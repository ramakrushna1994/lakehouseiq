"""Parsing of model output into SQL, plus display formatting. PURE (stdlib only).

The generation contract is a JSON object ``{"sql": "...", "view": "<fqn or null>"}``.
Models do not always honour it, so parsing is deliberately forgiving about fences,
surrounding prose and bare SQL, but strict about the ``view`` field: it is only
trusted when it names a view the product actually publishes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

_SQL_FIELD_RE = re.compile(r'"sql"\s*:\s*"(.*?)"\s*(?:,\s*"view"|\})', re.DOTALL)


def strip_markdown_fences(text: str) -> str:
    """Remove ```sql / ```json / ``` fences the model sometimes wraps its answer in."""
    return re.sub(r"```(?:sql|json)?", "", text).strip()


def _load_json(text: str):
    # strict=False tolerates raw newlines/tabs inside JSON strings, which models
    # emit constantly when the SQL is multi-line.
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        return None


def _normalise_view(view, known_views: Iterable[str]) -> str | None:
    if not isinstance(view, str):
        return None
    cleaned = view.replace("`", "").strip()
    by_lower = {v.lower(): v for v in known_views}
    return by_lower.get(cleaned.lower())


def parse_sql_response(raw: str, known_views: Iterable[str]) -> dict:
    """Parse a model reply into ``{"sql": str, "view": str|None, "error": str|None}``."""
    known = list(known_views)
    text = strip_markdown_fences(raw or "")

    parsed = _load_json(text)
    if parsed is None:
        # The model may wrap the JSON in prose; pull out the outermost object.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = _load_json(match.group())

    if isinstance(parsed, dict) and parsed.get("sql"):
        return {
            "sql": str(parsed["sql"]).strip(),
            "view": _normalise_view(parsed.get("view"), known),
            "error": None,
        }

    # Last resort for almost-JSON, e.g. SQL containing an invalid escape like \'.
    if text.lstrip().startswith("{"):
        m = _SQL_FIELD_RE.search(text)
        if m and m.group(1).strip():
            view_match = re.search(r'"view"\s*:\s*"([^"]*)"', text)
            return {
                "sql": m.group(1).replace('\\"', '"').replace("\\n", "\n").strip(),
                "view": _normalise_view(view_match.group(1) if view_match else None, known),
                "error": None,
            }

    # Fallback: the model returned bare SQL instead of JSON.
    if text.upper().lstrip().startswith(("SELECT", "WITH")):
        return {"sql": text, "view": None, "error": None}

    return {"sql": "", "view": None, "error": "Could not parse LLM response"}


# --------------------------------------------------------------------------- #
# Display formatting (never used for execution)
# --------------------------------------------------------------------------- #

_LITERAL_RE = re.compile(
    r"""'(?:[^'\\]|\\.|'')*'|"(?:[^"\\]|\\.|"")*"|`[^`]*`"""
)
_FORMAT_KEYWORDS = re.compile(
    r"\b(SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|"
    r"LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|CROSS\s+JOIN|INNER\s+JOIN|JOIN|"
    r"ON|AND|OR|UNION\s+ALL|UNION|EXCEPT|INTERSECT|WITH|AS\s*\(|"
    r"CASE|WHEN|THEN|ELSE|END)\b",
    re.IGNORECASE,
)
_INDENT_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT",
    "OFFSET", "UNION ALL", "UNION", "EXCEPT", "INTERSECT", "WITH",
}
_SUB_INDENT_KEYWORDS = {
    "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "CROSS JOIN", "INNER JOIN", "JOIN",
    "ON", "AND", "OR", "WHEN", "THEN", "ELSE", "END",
}


def format_sql(sql: str) -> str:
    """Best-effort line-breaking for display. Literals and quoted names are protected."""
    if not sql:
        return sql
    protected: list[str] = []

    def _hide(m: re.Match) -> str:
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    s = " ".join(_LITERAL_RE.sub(_hide, sql).split())
    lines: list[str] = []
    for part in _FORMAT_KEYWORDS.split(s):
        token = part.strip()
        if not token:
            continue
        upper = " ".join(token.upper().split())
        if upper in _INDENT_KEYWORDS:
            lines.append(token.upper())
        elif upper in _SUB_INDENT_KEYWORDS:
            lines.append("  " + token.upper())
        elif lines:
            lines[-1] += " " + token
        else:
            lines.append(token)
    text = "\n".join(lines)
    return re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], text)
