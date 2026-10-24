"""Result-set equivalence for the eval harness. PURE (stdlib + pandas).

Two result sets are equivalent when they contain the same values, regardless of
column names. This is deliberate: a model may alias ``revenue`` as ``total`` and
still be right. Numbers are rounded, strings trimmed and casefolded, dates compared
as ISO strings, and ``None`` equals ``NaN``.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import math
from decimal import Decimal

import pandas as pd

MAX_PERMUTED_COLUMNS = 6
_MAX_ASSIGNMENTS = 720


def _norm(value, round_dp: int):
    """Normalise one cell to None, float, or str."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, Decimal)) or hasattr(value, "dtype") and getattr(value.dtype, "kind", "") in "iuf":
        f = float(value)
        return None if math.isnan(f) else round(f, round_dp)
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, _dt.datetime):
        return value.date().isoformat() if value.time() == _dt.time(0) else value.isoformat(sep=" ")
    if isinstance(value, _dt.date):
        return value.isoformat()
    return str(value).strip().casefold()


def _sort_key(v):
    if v is None:
        return (0, 0.0, "")
    if isinstance(v, float):
        return (1, v, "")
    return (2, 0.0, v)


def _eq(a, b, rel_tol: float, round_dp: int) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=0.5 * 10 ** (-round_dp) + 1e-12)
    return a == b


def _column_values(df: pd.DataFrame, idx: int, round_dp: int) -> list:
    return [_norm(v, round_dp) for v in df.iloc[:, idx].tolist()]


def _cols_equal(a: list, b: list, order_matters: bool, rel_tol: float, round_dp: int) -> bool:
    if len(a) != len(b):
        return False
    if not order_matters:
        a, b = sorted(a, key=_sort_key), sorted(b, key=_sort_key)
    return all(_eq(x, y, rel_tol, round_dp) for x, y in zip(a, b, strict=True))


def _rows(cols: list[list], order_matters: bool) -> list[tuple]:
    rows = list(zip(*cols, strict=True)) if cols else []
    if not order_matters:
        rows.sort(key=lambda r: tuple(_sort_key(v) for v in r))
    return rows


def _rows_equal(exp_cols, act_cols, order_matters, rel_tol, round_dp) -> bool:
    er, ar = _rows(exp_cols, order_matters), _rows(act_cols, order_matters)
    return all(
        all(_eq(x, y, rel_tol, round_dp) for x, y in zip(r1, r2, strict=True)) for r1, r2 in zip(er, ar, strict=True)
    )


def results_match(
    df_expected: pd.DataFrame | None,
    df_actual: pd.DataFrame | None,
    order_matters: bool,
    rel_tol: float = 1e-6,
    round_dp: int = 2,
) -> tuple[bool, str]:
    """Return ``(matches, reason)``. ``reason`` is ``"match"`` on success."""
    if df_expected is None or df_actual is None:
        return False, "missing result set"
    n_rows = len(df_expected)
    if len(df_actual) != n_rows:
        return False, f"row count differs: expected {n_rows}, got {len(df_actual)}"
    m, k = df_expected.shape[1], df_actual.shape[1]
    if m == 0:
        return (k == 0 or n_rows == 0), "match" if (k == 0 or n_rows == 0) else "expected no columns"
    if k < m:
        return False, f"column count differs: expected {m}, got {k}"

    exp_cols = [_column_values(df_expected, i, round_dp) for i in range(m)]
    act_cols = [_column_values(df_actual, j, round_dp) for j in range(k)]

    if m > MAX_PERMUTED_COLUMNS:
        # Too many columns to permute: require the same column order.
        if _rows_equal(exp_cols, act_cols[:m], order_matters, rel_tol, round_dp):
            return True, "match"
        return False, f"values differ (more than {MAX_PERMUTED_COLUMNS} columns, same order required)"

    candidates: list[list[int]] = []
    for i, col in enumerate(exp_cols):
        # Column-level screen: same values ignoring row alignment.
        cand = [j for j, a in enumerate(act_cols) if _cols_equal(col, a, False, rel_tol, round_dp)]
        if not cand:
            name = df_expected.columns[i]
            return False, f"no actual column matches expected column {i} ({name!r})"
        candidates.append(cand)

    tried = 0
    for assign in itertools.product(*candidates):
        if len(set(assign)) != m:
            continue
        tried += 1
        if tried > _MAX_ASSIGNMENTS:
            break
        if _rows_equal(exp_cols, [act_cols[j] for j in assign], order_matters, rel_tol, round_dp):
            return True, "match"
    suffix = " (row order matters)" if order_matters else ""
    return False, f"column values match individually but rows do not line up{suffix}"
