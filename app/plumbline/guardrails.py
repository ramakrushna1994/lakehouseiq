"""SQL guardrails: validate generated SQL and cap result size.

This module is PURE (standard library only) so it runs offline in CI.

Defence in depth: the checks here are the *first* layer. They reject anything that
is not a single read-only ``SELECT``/``WITH`` statement, anything that touches
``system`` / ``information_schema`` / the assistant's own ``app`` schema, and a list
of Databricks statements that have side effects. The *enforced* boundary is Unity
Catalog grants (``notebooks/05_grant_app.py``): the App's service principal has
``SELECT`` only on ``sales``, ``hr`` and ``finance``, and ``MODIFY`` only on the two
``app`` tables it writes. Regexes can be bypassed by a sufficiently creative
statement; grants cannot.

Lexical rules follow Spark SQL, which differs from ANSI quoting:

* ``'text'`` and ``"text"`` are both string literals.
* Backslash escapes apply inside both quote styles (``'it\\'s'``); a doubled quote
  (``'it''s'``) is also an escape. ``r'...'`` raw literals have no backslash escape.
* Backticks delimit identifiers, not literals.
* ``--`` and ``/* ... */`` start comments only outside literals and identifiers.

All of that is handled by one character-level scanner rather than regexes, so a
``--`` inside a literal is data, and a quote inside a comment cannot hide code.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #


def _scan(
    sql: str,
    *,
    drop_comments: bool = True,
    blank_literals: bool = False,
    blank_backticks: bool = False,
    unquote_backticks: bool = False,
) -> tuple[str, bool]:
    """Walk ``sql`` once and return ``(text, terminated)``.

    ``terminated`` is False when a string literal, backtick identifier or block
    comment is still open at the end of input.

    * ``drop_comments``: replace comments with a single space.
    * ``blank_literals``: keep the quote characters but empty the literal body.
    * ``blank_backticks``: keep the backticks but empty the identifier body.
    * ``unquote_backticks``: drop the backticks and keep the identifier body.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    terminated = True
    while i < n:
        ch = sql[i]

        # -- comments -------------------------------------------------------
        if ch == "-" and sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j == -1 else j
            out.append(" " if drop_comments else sql[i:j])
            i = j
            continue
        if ch == "/" and sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            if j == -1:
                terminated = False
                j = n
            else:
                j += 2
            out.append(" " if drop_comments else sql[i:j])
            i = j
            continue

        # -- string literals ------------------------------------------------
        if ch in ("'", '"'):
            raw = i > 0 and sql[i - 1] in "rR" and (i < 2 or not (sql[i - 2].isalnum() or sql[i - 2] == "_"))
            j = i + 1
            closed = False
            while j < n:
                c = sql[j]
                if c == "\\" and not raw:
                    j += 2
                    continue
                if c == ch:
                    if j + 1 < n and sql[j + 1] == ch:  # doubled quote
                        j += 2
                        continue
                    closed = True
                    break
                j += 1
            if not closed:
                terminated = False
                j = n
                out.append(sql[i:j] if not blank_literals else ch)
                i = j
                continue
            out.append(ch + ch if blank_literals else sql[i : j + 1])
            i = j + 1
            continue

        # -- backtick identifiers ------------------------------------------
        if ch == "`":
            j = i + 1
            closed = False
            while j < n:
                if sql[j] == "`":
                    if j + 1 < n and sql[j + 1] == "`":
                        j += 2
                        continue
                    closed = True
                    break
                j += 1
            if not closed:
                terminated = False
                out.append(sql[i:])
                i = n
                continue
            body = sql[i + 1 : j]
            if blank_backticks:
                out.append("``")
            elif unquote_backticks:
                out.append(body)
            else:
                out.append(sql[i : j + 1])
            i = j + 1
            continue

        out.append(ch)
        i += 1
    return "".join(out), terminated


def strip_sql_comments(sql: str) -> str:
    """Remove ``--`` and ``/* */`` comments (never inside literals) and trim."""
    return _scan(sql)[0].strip()


def blank_string_literals(sql: str) -> str:
    """Empty the body of every string literal, keeping the quotes.

    Keeps statement structure intact but stops data from being read as SQL, so a
    legitimate filter such as ``return_reason = 'Needs update'`` never trips the
    DDL/DML keyword scan.
    """
    return _scan(sql, drop_comments=False, blank_literals=True)[0]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

FORBIDDEN_KEYWORDS = (
    # classic DDL / DML / privileges
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "GRANT", "MERGE",
    "CREATE", "REVOKE", "EXEC", "EXECUTE",
    # Databricks statements with side effects or session impact
    "OPTIMIZE", "VACUUM", "COPY", "REFRESH", "RESTORE", "CLONE", "CALL", "USE",
    "SET", "RESET", "CACHE", "UNCACHE", "MSCK", "ANALYZE", "LOAD", "REPAIR",
    "ADD", "LIST", "PUT", "GET", "REMOVE",
)  # REPLACE is deliberately absent: it is a string function; CREATE OR REPLACE is caught by CREATE.

_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
_START_RE = re.compile(r"^\(*\s*(SELECT|WITH)\b", re.IGNORECASE)
_VALUES_START_RE = re.compile(r"^\(*\s*VALUES\b", re.IGNORECASE)

# Schemas the assistant must never read. Matched as a qualifier: ``system.x``,
# ``cat.app.x``, ``app.x``.
BLOCKED_SCHEMAS = ("system", "app", "information_schema")
_BLOCKED_SCHEMA_RE = re.compile(
    r"(?<![\w$])(" + "|".join(BLOCKED_SCHEMAS) + r")\s*\.", re.IGNORECASE
)


def validate_sql(sql: str) -> tuple[bool, str | None]:
    """Check that ``sql`` is one read-only SELECT. Returns ``(ok, error)``."""
    if not sql or not sql.strip():
        return False, "Generated SQL is empty."

    code, terminated = _scan(sql, blank_literals=True, blank_backticks=True)
    if not terminated:
        return False, "Generated SQL has an unterminated string, identifier or comment."
    code = code.strip()
    if not code:
        return False, "Generated SQL is empty."

    if _VALUES_START_RE.match(code):
        return False, "VALUES-only statements are not allowed."
    if not _START_RE.match(code):
        return False, "Generated SQL must be a SELECT or WITH statement."

    # Stacked statements: a single trailing semicolon is fine, anything after is not.
    if ";" in code.rstrip().rstrip(";"):
        return False, "Generated SQL must be a single statement."

    match = _FORBIDDEN_RE.search(code)
    if match:
        return False, f"Generated SQL contains forbidden keyword: {match.group(0).upper()}"

    # Qualifier check runs on a view where backticks are removed so that
    # `system`.`access`.`audit` is seen as system.access.audit.
    ident_view, _ = _scan(sql, blank_literals=True, unquote_backticks=True)
    blocked = _BLOCKED_SCHEMA_RE.search(ident_view)
    if blocked:
        return False, f"Generated SQL references a restricted schema: {blocked.group(1).lower()}"
    return True, None


# --------------------------------------------------------------------------- #
# Row cap
# --------------------------------------------------------------------------- #

# A row cap only applies if LIMIT is the *last* clause. A LIMIT inside a CTE or
# subquery caps that subquery, not the result set.
_TRAILING_ROW_CAP = re.compile(
    r"\b(?:LIMIT\s+\d+(?:\s+OFFSET\s+\d+)?"
    r"|FETCH\s+(?:FIRST|NEXT)\s+\d+\s+ROWS?\s+ONLY)\s*$",
    re.IGNORECASE,
)


def ensure_limit(sql: str, max_rows: int = 500) -> str:
    """Append ``LIMIT max_rows`` unless the query already ends with a row cap."""
    body = strip_sql_comments(sql).rstrip().rstrip(";").rstrip()
    if _TRAILING_ROW_CAP.search(body):
        return sql
    return sql.rstrip().rstrip(";").rstrip() + f"\nLIMIT {max_rows}"
