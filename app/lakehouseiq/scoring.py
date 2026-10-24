"""Code-verified quality checks and the final verdict. PURE (stdlib + semantic).

Three kinds of evidence feed the verdict:

* schema compliance - every dotted identifier in the SQL exists in the live schema
  or the metric-view registry (checked in code, not by a model)
* value compliance  - literals compared against categorical columns match values
  actually present in the data
* metric-view rules - measures are read with ``MEASURE()`` and never with
  ``SELECT *`` or a plain aggregate

The judge model only adds the subjective scores (relevance, SQL quality); it can
lower confidence but never raise it above what the code checks allow.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .guardrails import _scan, strip_sql_comments
from .semantic import MetricView

# --------------------------------------------------------------------------- #
# Identifier extraction and schema compliance
# --------------------------------------------------------------------------- #

_DOTTED_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_AS_ALIAS_RE = re.compile(r"\bAS\s+(\w+)", re.IGNORECASE)
_CTE_RE = re.compile(r"(?:\bWITH\b|,)\s*(\w+)\s*(?:\([^)]*\)\s*)?AS\s*\(", re.IGNORECASE)
_MEASURE_RE = re.compile(r"\bMEASURE\s*\(\s*(\w+)\s*\)", re.IGNORECASE)


def _code_view(sql: str) -> str:
    """Comments removed, literals blanked, backticks removed: identifiers only."""
    return _scan(sql, blank_literals=True, unquote_backticks=True)[0]


def extract_identifiers(sql: str) -> list[str]:
    """Dot-separated identifiers (``schema.table.col``, ``alias.col``), upper-cased."""
    return [i.upper() for i in _DOTTED_RE.findall(_code_view(sql))]


def _referenced_views(sql: str, registry: Iterable[MetricView]) -> list[MetricView]:
    code = _code_view(sql).lower()
    return [
        mv
        for mv in registry
        if re.search(rf"(?<![\w.]){re.escape(mv.schema)}\s*\.\s*{re.escape(mv.name)}\b", code)
        or re.search(rf"\.\s*{re.escape(mv.schema)}\s*\.\s*{re.escape(mv.name)}\b", code)
    ]


def unknown_measures(sql: str, registry: Iterable[MetricView]) -> list[str]:
    """``MEASURE(x)`` calls where ``x`` is not a measure of a referenced metric view."""
    registry = list(registry)
    views = _referenced_views(sql, registry)
    if not views:
        return []
    allowed = {m.lower() for mv in views for m in mv.measure_names}
    return [m for m in _MEASURE_RE.findall(_code_view(sql)) if m.lower() not in allowed]


def compute_schema_compliance(
    sql: str,
    known_identifiers: set[str],
    registry: Iterable[MetricView] | None = None,
) -> tuple[int, str]:
    """Return ``(score 1-10, reason)``. Identifiers compare case-insensitively."""
    registry = list(registry or [])
    code = _code_view(sql)
    local_names = {a.upper() for a in _AS_ALIAS_RE.findall(code)}
    local_names |= {c.upper() for c in _CTE_RE.findall(code)}

    identifiers = extract_identifiers(sql)
    bad_measures = unknown_measures(sql, registry)
    if not identifiers and not bad_measures:
        return 10, "No table/column references to verify (code-verified)"

    valid = 0
    invalid: list[str] = []
    for ident in identifiers:
        parts = ident.split(".")
        if (
            ident in known_identifiers
            or (len(parts) >= 2 and f"{parts[-2]}.{parts[-1]}" in known_identifiers)
            or parts[-1] in known_identifiers
            or parts[0] in local_names  # cte.col / subquery.col: defined in this query
            or parts[-1] in local_names  # alias.computed_col
        ):
            valid += 1
        else:
            invalid.append(ident.lower())
    for m in bad_measures:
        invalid.append(f"measure({m.lower()})")

    total = len(identifiers) + len(bad_measures)
    valid_total = valid
    score = max(1, min(10, round(valid_total / total * 10)))
    if invalid:
        reason = (
            f"{valid_total}/{total} identifiers verified; unrecognized: "
            f"{', '.join(invalid[:3])} (code-verified)"
        )
    else:
        reason = f"All {total} identifiers verified against live schema (code-verified)"
    return score, reason


# --------------------------------------------------------------------------- #
# Value compliance
# --------------------------------------------------------------------------- #

_LIT = r"""'(?:[^'\\]|\\.|'')*'|"(?:[^"\\]|\\.|"")*\""""
_LIT_RE = re.compile(_LIT)
# Only WHERE-style comparisons count. A literal in a THEN/AS/SELECT position is an
# output label (e.g. CASE ... THEN '0-30') and must never be flagged.
_EQUALITY_RE = re.compile(
    r"(?:\w+\.)?`?(\w+)`?\s*(?:==|=|!=|<>)\s*(" + _LIT + ")", re.IGNORECASE
)
_IN_RE = re.compile(
    r"(?:\w+\.)?`?(\w+)`?\s+IN\s*\(((?:" + _LIT + r"|[^)'\"])*)\)", re.IGNORECASE
)


def _unquote(lit: str) -> str:
    q = lit[0]
    body = lit[1:-1].replace(q + q, q)
    return re.sub(r"\\(.)", r"\1", body)


def extract_string_literals(sql: str) -> list[str]:
    """All string literal values (single or double quoted) outside comments."""
    return [_unquote(m.group(0)) for m in _LIT_RE.finditer(strip_sql_comments(sql))]


def check_value_compliance(sql: str, enum_hints: dict[str, list[str]]) -> tuple[int, str, list[str]]:
    """Check literals compared against known categorical columns.

    ``enum_hints`` maps ``schema.table.column`` to the values present in the data.
    Returns ``(score 1-10, reason, bad_values)``.
    """
    # The same column name can appear in several tables (e.g. status), so a value
    # is accepted if it is valid for any table's column of that name.
    values_by_column: dict[str, set[str]] = {}
    for key, values in enum_hints.items():
        bucket = values_by_column.setdefault(key.split(".")[-1].upper(), set())
        bucket.update(str(v).upper() for v in values)

    if not values_by_column:
        return 10, "No sampled enum values available to validate against", []

    cleaned = strip_sql_comments(sql)
    compared: list[tuple[str, str]] = []
    for col, lit in _EQUALITY_RE.findall(cleaned):
        compared.append((col.upper(), _unquote(lit)))
    for col, in_list in _IN_RE.findall(cleaned):
        for lit in _LIT_RE.findall(in_list):
            compared.append((col.upper(), _unquote(lit)))

    bad_values: list[str] = []
    checked = 0
    for col, lit in compared:
        allowed = values_by_column.get(col)
        if not allowed:
            continue  # not a column we sampled
        if "%" in lit or lit.strip() == "":
            continue  # LIKE-style pattern or empty
        checked += 1
        if lit.upper() not in allowed:
            bad_values.append(lit)

    if checked == 0:
        return 10, "No enum column comparisons to validate", []

    valid = checked - len(bad_values)
    score = max(1, min(10, round(valid / checked * 10)))
    if bad_values:
        reason = (
            f"{valid}/{checked} filter values valid; unrecognized: "
            f"{', '.join(repr(v) for v in bad_values[:3])}"
        )
    else:
        reason = f"All {checked} filter values match sampled column values"
    return score, reason, bad_values


# --------------------------------------------------------------------------- #
# Metric-view rules
# --------------------------------------------------------------------------- #

_AGGS = r"(?:SUM|AVG|MIN|MAX|COUNT|MEDIAN|STDDEV|VARIANCE)"
_BARE_STAR_RE = re.compile(r"\bSELECT\s+(?:DISTINCT\s+)?\*", re.IGNORECASE)
_QUAL_STAR_RE = re.compile(r"(\w+)\s*\.\s*\*")
_SOURCE_RE = re.compile(r"\b(?:FROM|JOIN)\s+(?:\w+\s*\.\s*)*(\w+)(?:\s+AS)?\s+(\w+)", re.IGNORECASE)
_NOT_ALIAS = {
    "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "JOIN", "LEFT", "RIGHT", "INNER", "FULL",
    "CROSS", "ON", "USING", "UNION", "EXCEPT", "INTERSECT", "WINDOW", "QUALIFY", "OFFSET",
}


def metric_view_misuse(sql: str, registry: Iterable[MetricView]) -> list[str]:
    """Rule violations when ``sql`` queries a metric view. Empty list means clean.

    (a) a measure aggregated without ``MEASURE()``, or ``MEASURE()`` wrapped in a
        second aggregate;
    (b) ``SELECT *`` against a metric view;
    (c) a measure referenced outside ``MEASURE()`` (e.g. inside ``CASE``);
    (d) ``MEASURE(x)`` where ``x`` is not a measure of the queried view(s).
    """
    views = _referenced_views(sql, registry)
    if not views:
        return []
    code = _code_view(sql)
    problems: list[str] = []

    for mv in views:
        for m in mv.measure_names:
            if re.search(rf"\b{_AGGS}\s*\(\s*(?:DISTINCT\s+)?(?:\w+\s*\.\s*)?{re.escape(m)}\s*\)", code, re.IGNORECASE):
                problems.append(
                    f"measure '{m}' of {mv.name} is aggregated directly; use MEASURE({m})"
                )
    if re.search(rf"\b{_AGGS}\s*\(\s*MEASURE\s*\(", code, re.IGNORECASE):
        problems.append("MEASURE() must not be wrapped in another aggregate")

    # (c) a measure used bare (inside CASE, arithmetic, HAVING ...) instead of through MEASURE().
    #     Output aliases are fine: MEASURE(x) AS x ... ORDER BY x.
    aliases = {a.lower() for a in re.findall(r"\bAS\s+(\w+)", code, re.IGNORECASE)}
    outside_measure = re.sub(r"\bMEASURE\s*\(\s*\w+\s*\)", " ", code, flags=re.IGNORECASE)
    outside_measure = re.sub(r"\bAS\s+\w+", " ", outside_measure, flags=re.IGNORECASE)
    for mv in views:
        for m in mv.measure_names:
            if m.lower() in aliases:
                continue
            if re.search(rf"(?<![\w.]){re.escape(m)}\b", outside_measure, re.IGNORECASE) and not any(
                m in p_ for p_ in problems
            ):
                problems.append(f"measure '{m}' of {mv.name} is used without MEASURE(); write MEASURE({m})")

    # (d) MEASURE(x) where x is not a measure of any metric view in FROM.
    known = {m.lower() for mv in views for m in mv.measure_names}
    for name in re.findall(r"\bMEASURE\s*\(\s*(\w+)\s*\)", code, re.IGNORECASE):
        if name.lower() not in known:
            owners = [mv.name for mv in registry if name.lower() in {x.lower() for x in mv.measure_names}]
            hint = f"; it belongs to {', '.join(owners)} (query that view, or a cross-fact view)" if owners else ""
            problems.append(f"'{name}' is not a measure of {', '.join(v.name for v in views)}{hint}")

    names = {mv.name.lower() for mv in views}
    mv_aliases = set(names)
    for m in _SOURCE_RE.finditer(code):
        if m.group(1).lower() in names and m.group(2).upper() not in _NOT_ALIAS:
            mv_aliases.add(m.group(2).lower())
    qualified = {q.lower() for q in _QUAL_STAR_RE.findall(code)}
    if _BARE_STAR_RE.search(code) or qualified & mv_aliases:
        problems.append("SELECT * on a metric view is not allowed; list dimensions and MEASURE() columns")
    return problems


# --------------------------------------------------------------------------- #
# Judge output and final verdict
# --------------------------------------------------------------------------- #

_REL_RE = re.compile(r"RELEVANCE:\s*(\d+)\s*/\s*10\s*\|?\s*(.*)", re.IGNORECASE)
_QUAL_RE = re.compile(r"SQL_QUALITY:\s*(\d+)\s*/\s*10\s*\|?\s*(.*)", re.IGNORECASE)
_VERD_RE = re.compile(r"VERDICT:\s*(GOOD|NEEDS REVIEW|RISKY)", re.IGNORECASE)


def parse_judge_response(raw: str) -> dict:
    """Parse the judge's three-line reply. ``parsed`` is False if scores are missing."""
    out = {
        "relevance": 0,
        "relevance_reason": "",
        "sql_quality": 0,
        "sql_quality_reason": "",
        "judge_verdict": None,
        "parsed": False,
        "raw": raw or "",
    }
    for line in (raw or "").splitlines():
        line = line.strip().lstrip("*-• ").replace("**", "")
        m = _REL_RE.match(line)
        if m:
            out["relevance"], out["relevance_reason"] = min(10, int(m.group(1))), m.group(2).strip()
            continue
        m = _QUAL_RE.match(line)
        if m:
            out["sql_quality"], out["sql_quality_reason"] = min(10, int(m.group(1))), m.group(2).strip()
            continue
        m = _VERD_RE.match(line)
        if m:
            out["judge_verdict"] = m.group(1).upper()
    out["parsed"] = out["relevance"] > 0 and out["sql_quality"] > 0
    return out


def final_verdict(
    schema_score: int,
    value_score: int,
    judge: dict | None,
    misuse: list[str] | None = None,
) -> str:
    """Combine code checks and the judge into GOOD / NEEDS REVIEW / RISKY / UNSCORED.

    * RISKY        schema < 5, value < 5, or any metric-view misuse
    * UNSCORED     the judge replied but its scores could not be parsed
    * NEEDS REVIEW any score < 8, or no judge result at all
    * GOOD         everything else
    """
    if schema_score < 5 or value_score < 5 or misuse:
        return "RISKY"
    if judge is None:
        return "NEEDS REVIEW"
    if not judge.get("parsed", False):
        return "UNSCORED"
    scores = [schema_score, value_score, judge.get("relevance", 0), judge.get("sql_quality", 0)]
    if any(s < 8 for s in scores):
        return "NEEDS REVIEW"
    return "GOOD"
