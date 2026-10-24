"""SQL warehouse access: queries, parameterised writes, live schema and enum discovery.

``databricks`` imports are deferred so the module (and ``SchemaContext``) can be loaded
anywhere. One connection is cached per process and re-opened once if it dies.

Writes always use named parameters (``:question``); SQL text is never built from
user-controlled values, so there is nothing to escape.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd

from .config import Settings

logger = logging.getLogger(__name__)

DATA_SCHEMAS = ("sales", "hr", "finance")

# Low-cardinality columns worth showing the model, as (schema, table, column).
ENUM_CANDIDATES: list[tuple[str, str, str]] = [
    ("sales", "orders", "status"),
    ("sales", "orders", "payment_method"),
    ("sales", "products", "category"),
    ("sales", "products", "subcategory"),
    ("sales", "customers", "region"),
    ("sales", "customers", "segment"),
    ("sales", "returns", "return_reason"),
    ("sales", "returns", "return_status"),
    ("sales", "sales_channels", "channel_name"),
    ("sales", "sales_channels", "channel_type"),
    ("sales", "stores", "region"),
    ("hr", "departments", "dept_name"),
    ("hr", "employees", "job_level"),
    ("hr", "employees", "job_title"),
    ("hr", "employees", "status"),
    ("hr", "job_history", "change_reason"),
    ("hr", "payroll", "pay_status"),
    ("finance", "expenses", "approval_status"),
    ("finance", "expenses", "category"),
    ("finance", "expenses", "vendor"),
    ("finance", "invoices", "invoice_status"),
    ("finance", "gl_accounts", "account_type"),
    ("finance", "gl_accounts", "account_category"),
    ("finance", "budget_lines", "budget_year"),
]
ENUM_VALUE_LIMIT = 15


@dataclass
class SchemaContext:
    tables_text: str = ""
    views_text: str = ""
    known_identifiers: set[str] = field(default_factory=set)
    enum_hints: dict[str, list[str]] = field(default_factory=dict)
    as_of_date: str = ""


# --------------------------------------------------------------------------- #
# Connection management
# --------------------------------------------------------------------------- #

_conn = None
_lock = threading.RLock()
_settings: Settings | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def configure(settings: Settings) -> None:
    """Use explicit settings (the eval harness and tests); closes any cached connection."""
    global _settings
    with _lock:
        _settings = settings
        close()


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
        _conn = None


def _open(settings: Settings, with_timeout: bool = True):
    from databricks import sql
    from databricks.sdk.core import Config

    cfg = Config()
    kwargs = {}
    if with_timeout:
        kwargs["session_configuration"] = {"statement_timeout": str(settings.query_timeout_s)}
    return sql.connect(
        server_hostname=cfg.host.removeprefix("https://").rstrip("/"),
        http_path=f"/sql/1.0/warehouses/{settings.require_warehouse_id()}",
        credentials_provider=lambda: cfg.authenticate,
        **kwargs,
    )


def _connect():
    global _conn
    with _lock:
        if _conn is not None:
            return _conn
        settings = _get_settings()
        try:
            _conn = _open(settings, with_timeout=True)
        except Exception as exc:
            logger.warning("connect with statement_timeout failed (%s); retrying without it", exc)
            _conn = _open(settings, with_timeout=False)
        return _conn


_CONNECTION_ERROR_NAMES = {
    "RequestError", "MaxRetryDurationError", "OperationalError", "InterfaceError",
    "SessionAlreadyClosedError", "CursorAlreadyClosedError", "ConnectionError", "TimeoutError",
}


def _is_connection_error(exc: Exception) -> bool:
    if type(exc).__name__ in _CONNECTION_ERROR_NAMES or isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return "invalid sessionhandle" in str(exc).lower()


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Decimal columns become float so they chart and compare cleanly."""
    for col in df.columns:
        if df[col].dtype == object:
            non_null = [v for v in df[col] if v is not None]
            if non_null and all(isinstance(v, Decimal) for v in non_null):
                df[col] = df[col].map(lambda v: None if v is None else float(v)).astype("float64")
    return df


def _fetch(sql: str, max_rows: int | None, params: dict | None = None) -> pd.DataFrame:
    with _lock:
        conn = _connect()
        with conn.cursor() as cur:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows + 1) if max_rows else cur.fetchall()
    df = pd.DataFrame([tuple(r) for r in rows], columns=cols)
    truncated = bool(max_rows) and len(df) > max_rows
    if truncated:
        df = df.iloc[:max_rows].reset_index(drop=True)
    df = _coerce(df)
    df.attrs["truncated"] = truncated
    return df


def run_query(sql: str, max_rows: int | None = None) -> tuple[pd.DataFrame | None, str | None]:
    """Execute a read query. Returns ``(DataFrame, None)`` or ``(None, error text)``.

    ``df.attrs["truncated"]`` is True when more than ``max_rows`` rows existed.
    A dead connection is re-opened and the query retried exactly once.
    """
    if max_rows is None:
        max_rows = _get_settings().max_rows
    for attempt in (1, 2):
        try:
            return _fetch(sql, max_rows), None
        except Exception as exc:
            if attempt == 1 and _is_connection_error(exc):
                logger.warning("connection error (%s); reconnecting once", exc)
                close()
                continue
            return None, str(exc)
    return None, "unreachable"  # pragma: no cover


def execute_params(sql: str, params: dict) -> None:
    """Run a write with NAMED parameters (``:name``). Raises on failure; callers decide."""
    for attempt in (1, 2):
        try:
            with _lock:
                conn = _connect()
                with conn.cursor() as cur:
                    cur.execute(sql, params)
            return
        except Exception as exc:
            if attempt == 1 and _is_connection_error(exc):
                close()
                continue
            raise


def query_params(sql: str, params: dict) -> pd.DataFrame:
    """Read with named parameters. Raises on failure."""
    return _fetch(sql, None, params)


# --------------------------------------------------------------------------- #
# Schema and enum discovery
# --------------------------------------------------------------------------- #


def build_enum_query(catalog: str, candidates=ENUM_CANDIDATES, limit: int = ENUM_VALUE_LIMIT) -> str:
    """One UNION ALL query returning ``(k, v)`` rows: ``k`` is ``schema.table.column``."""
    parts = [
        f"(SELECT '{s}.{t}.{c}' AS k, CAST({c} AS STRING) AS v FROM {catalog}.{s}.{t} "
        f"WHERE {c} IS NOT NULL GROUP BY {c} ORDER BY {c} LIMIT {limit})"
        for s, t, c in candidates
    ]
    return "\nUNION ALL\n".join(parts)


def get_enum_hints(catalog: str, run=None) -> dict[str, list[str]]:
    """Distinct values of the categorical columns, in a single round trip.

    ``run`` is any ``(sql, max_rows) -> (DataFrame | None, error | None)`` callable; the default is
    :func:`run_query`. The eval notebook passes a Spark-backed runner.
    """
    run = run or run_query
    df, err = run(build_enum_query(catalog), len(ENUM_CANDIDATES) * ENUM_VALUE_LIMIT)
    if err is not None:
        # One missing column fails the whole UNION; fall back to per-column queries.
        logger.warning("combined enum query failed (%s); sampling columns individually", err)
        hints: dict[str, list[str]] = {}
        for cand in ENUM_CANDIDATES:
            one, e = run(build_enum_query(catalog, [cand]), ENUM_VALUE_LIMIT)
            if e is None and one is not None and len(one):
                hints["{}.{}.{}".format(*cand)] = [str(v) for v in one["v"]]
            elif e is not None:
                logger.warning("enum sampling failed for %s: %s", ".".join(cand), e)
        return hints
    hints = {}
    for k, v in zip(df["k"], df["v"], strict=True):
        hints.setdefault(k, []).append(str(v))
    return hints


def _render(objects: dict[str, dict], enum_hints: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for fqn, obj in objects.items():
        lines.append(fqn + (f"  -- {obj['comment']}" if obj["comment"] else ""))
        for col, dtype, comment, key in obj["cols"]:
            notes = []
            if comment:
                notes.append(comment)
            if key in enum_hints:
                notes.append("Values: " + ", ".join(enum_hints[key]))
            lines.append(f"  - {col} ({dtype})" + (f"  -- {'; '.join(notes)}" if notes else ""))
    return "\n".join(lines)


def get_as_of_date(settings: Settings, run=None) -> str:
    """Latest data date: ``AS_OF_DATE`` if configured, otherwise ``max(order_date)``."""
    if settings.as_of_date:
        return settings.as_of_date
    df, err = (run or run_query)(f"SELECT CAST(max(order_date) AS STRING) AS d FROM {settings.fq('sales', 'orders')}", 1)
    if err or df is None or df.empty or df.iloc[0, 0] is None:
        raise RuntimeError(f"could not determine the data's as-of date: {err or 'no orders found'}")
    return str(df.iloc[0, 0])


def get_live_schema(settings: Settings | None = None, run=None) -> SchemaContext:
    """Read tables, views, comments, enum values and the as-of date from the catalog.

    Metric views (``mv_*``) are excluded: they come from the registry, which is the
    source of truth for their dimensions and measures.
    """
    settings = settings or _get_settings()
    run = run or run_query
    cat = settings.catalog
    schemas = ", ".join(f"'{s}'" for s in DATA_SCHEMAS)
    df, err = run(
        f"""
        SELECT c.table_schema AS s, c.table_name AS t, c.column_name AS c, c.data_type AS dt,
               c.comment AS col_comment, tb.table_type AS tt, tb.comment AS tbl_comment
        FROM {cat}.information_schema.columns c
        JOIN {cat}.information_schema.tables tb
          ON c.table_schema = tb.table_schema AND c.table_name = tb.table_name
        WHERE c.table_schema IN ({schemas}) AND NOT startswith(lower(c.table_name), 'mv_')
        ORDER BY c.table_schema, c.table_name, c.ordinal_position
        """,
        20000,
    )
    if err is not None or df is None:
        raise RuntimeError(f"schema discovery failed: {err}")

    enum_hints = get_enum_hints(cat, run)
    tables: dict[str, dict] = {}
    views: dict[str, dict] = {}
    known: set[str] = set()
    for s, t, c, dt, ccom, tt, tcom in df.itertuples(index=False, name=None):
        if "METRIC" in str(tt).upper():
            continue
        target = views if str(tt).upper() == "VIEW" else tables
        fqn = f"{cat}.{s}.{t}"
        obj = target.setdefault(fqn, {"comment": (tcom or "").strip(), "cols": []})
        obj["cols"].append((c, dt, (ccom or "").strip(), f"{s}.{t}.{c}"))
        for ident in (
            f"{s}.{t}.{c}", f"{cat}.{s}.{t}.{c}", f"{t}.{c}", t, f"{s}.{t}", f"{cat}.{s}.{t}", c, s, cat,
        ):
            known.add(ident.upper())

    return SchemaContext(
        tables_text=_render(tables, enum_hints),
        views_text=_render(views, enum_hints),
        known_identifiers=known,
        enum_hints=enum_hints,
        as_of_date=get_as_of_date(settings, run),
    )
