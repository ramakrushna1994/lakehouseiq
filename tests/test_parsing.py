"""Offline tests for plumbline.parsing."""

from plumbline.parsing import format_sql, parse_sql_response, strip_markdown_fences

VIEW = "workspace.sales.mv_orders"
KNOWN = [VIEW, "workspace.sales.v_product_performance"]


def test_parses_plain_json():
    out = parse_sql_response('{"sql": "SELECT 1", "view": null}', KNOWN)
    assert (out["sql"], out["view"], out["error"]) == ("SELECT 1", None, None)


def test_parses_known_view():
    raw = '{"sql": "SELECT 1", "view": "%s"}' % VIEW
    assert parse_sql_response(raw, KNOWN)["view"] == VIEW


def test_view_match_is_case_and_backtick_insensitive():
    raw = '{"sql": "SELECT 1", "view": "`Workspace`.`sales`.`MV_ORDERS`"}'
    assert parse_sql_response(raw, KNOWN)["view"] == VIEW


def test_rejects_unknown_view_but_keeps_sql():
    out = parse_sql_response('{"sql": "SELECT 1", "view": "some.made.up_view"}', KNOWN)
    assert out["view"] is None and out["sql"] == "SELECT 1"


def test_non_string_view_is_ignored():
    assert parse_sql_response('{"sql": "SELECT 1", "view": 7}', KNOWN)["view"] is None


def test_strips_markdown_fences():
    raw = '```json\n{"sql": "SELECT 1", "view": null}\n```'
    assert parse_sql_response(raw, KNOWN)["sql"] == "SELECT 1"
    assert strip_markdown_fences("```sql\nSELECT 1\n```") == "SELECT 1"


def test_extracts_json_wrapped_in_prose():
    raw = 'Here you go:\n{"sql": "SELECT 1", "view": null}\nHope that helps.'
    assert parse_sql_response(raw, KNOWN)["sql"] == "SELECT 1"


def test_falls_back_to_bare_sql():
    out = parse_sql_response("SELECT a FROM t", KNOWN)
    assert (out["sql"], out["error"]) == ("SELECT a FROM t", None)


def test_bare_with_statement():
    assert parse_sql_response("WITH x AS (SELECT 1) SELECT * FROM x", KNOWN)["error"] is None


def test_reports_error_when_unparseable():
    out = parse_sql_response("I cannot answer that question.", KNOWN)
    assert out["sql"] == "" and out["error"]


def test_empty_input():
    assert parse_sql_response("", KNOWN)["error"]
    assert parse_sql_response(None, KNOWN)["error"]


def test_handles_sql_containing_braces():
    raw = '{"sql": "SELECT regexp_extract(a, \'x{2}\', 0) FROM t", "view": null}'
    assert "regexp_extract" in parse_sql_response(raw, KNOWN)["sql"]


def test_multiline_sql_with_raw_newlines():
    raw = '{"sql": "SELECT a\nFROM t\nWHERE b = 1", "view": null}'
    assert parse_sql_response(raw, KNOWN)["sql"] == "SELECT a\nFROM t\nWHERE b = 1"


def test_invalid_backslash_escape_still_parsed():
    raw = '{"sql": "SELECT * FROM t WHERE n = \'O\\\'Brien\'", "view": null}'
    out = parse_sql_response(raw, KNOWN)
    assert out["error"] is None and "Brien" in out["sql"]


def test_format_sql_breaks_clauses():
    out = format_sql("select a from t where b = 1 order by a limit 5")
    assert out.splitlines()[0].startswith("SELECT") and "LIMIT 5" in out


def test_format_sql_does_not_touch_literals():
    out = format_sql("select a from t where n = 'black and white from here'")
    assert "'black and white from here'" in out
