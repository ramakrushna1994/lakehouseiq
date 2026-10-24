"""Offline tests for lakehouseiq.guardrails (no warehouse, no model calls)."""

import pytest

from lakehouseiq.guardrails import (
    blank_string_literals,
    ensure_limit,
    strip_sql_comments,
    validate_sql,
)


class TestValidateAccepts:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT a FROM t",
            "WITH x AS (SELECT 1) SELECT * FROM x",
            "(SELECT a FROM t) UNION ALL (SELECT a FROM u)",
            "SELECT a FROM t;",
            "select replace(name, 'a', 'b') from t",
            "SELECT a FROM t -- DROP TABLE t",
            "SELECT /* DROP TABLE t */ a FROM t",
        ],
    )
    def test_accepts(self, sql):
        assert validate_sql(sql) == (True, None)

    def test_rejects_non_select(self):
        ok, err = validate_sql("SHOW TABLES")
        assert not ok and "SELECT" in err

    def test_rejects_empty(self):
        assert validate_sql("   ")[0] is False
        assert validate_sql("-- only a comment")[0] is False

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1; DROP TABLE customers",
            "SELECT * FROM t; DELETE FROM t",
            "WITH x AS (SELECT 1) SELECT * FROM x; TRUNCATE TABLE t",
        ],
    )
    def test_blocks_stacked_ddl(self, sql):
        assert validate_sql(sql)[0] is False


class TestLiteralsAreData:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM returns WHERE return_reason = 'Needs update'",
            "SELECT name FROM customers WHERE segment = 'Enterprise Create'",
            "SELECT * FROM job_history WHERE change_reason = 'Insert transfer'",
            "SELECT CASE WHEN x > 0 THEN 'Merge candidate' ELSE 'None' END FROM t",
            'SELECT * FROM t WHERE reason = "Needs update"',
            "SELECT * FROM t WHERE note = 'set; drop table x'",
        ],
    )
    def test_keyword_inside_literal_allowed(self, sql):
        ok, err = validate_sql(sql)
        assert ok, err

    def test_keyword_outside_literal_still_blocked(self):
        assert validate_sql("SELECT 'safe update' FROM t; DROP TABLE t")[0] is False

    def test_double_quoted_literal_is_a_literal(self):
        # In Spark SQL "x" is a string, so a keyword in it is data.
        assert validate_sql('SELECT "drop table" FROM t')[0] is True

    def test_backslash_escaped_quote_does_not_end_literal(self):
        # 'it\'s a drop' is ONE literal in Spark SQL; the keyword stays inside it.
        assert validate_sql(r"SELECT 'it\'s a drop' FROM t")[0] is True

    def test_backslash_escape_cannot_hide_code(self):
        # The literal ends at the second quote; DROP is real code.
        assert validate_sql(r"SELECT 'a\\' ; DROP TABLE t --'")[0] is False

    def test_doubled_quote_escape(self):
        assert validate_sql("SELECT 'it''s a drop' FROM t")[0] is True

    def test_raw_literal_has_no_backslash_escape(self):
        # r'\' is a complete literal; what follows is real code.
        assert validate_sql("SELECT r'\\' ; DROP TABLE t; --'")[0] is False

    def test_comment_marker_inside_literal_is_data(self):
        assert validate_sql("SELECT '--' AS dash, 'x' AS y FROM t")[0] is True
        assert blank_string_literals("SELECT '-- x' FROM t -- c") == "SELECT '' FROM t -- c"

    def test_block_comment_marker_inside_literal_is_data(self):
        sql = "SELECT '/*' AS a FROM t WHERE x = 1; DROP TABLE t; --*/"
        assert validate_sql(sql)[0] is False  # the DROP is real, not hidden by '/*'

    def test_unterminated_literal_rejected(self):
        ok, err = validate_sql("SELECT 'abc FROM t")
        assert not ok and "unterminated" in err

    def test_unterminated_block_comment_rejected(self):
        assert validate_sql("SELECT 1 /* oops")[0] is False

    def test_backtick_identifier_is_not_a_literal(self):
        assert validate_sql("SELECT `order date` FROM t")[0] is True
        # keyword spelled as a quoted identifier is a column name, not a statement
        assert validate_sql("SELECT `update` FROM t")[0] is True

    def test_backtick_not_blanked_as_literal(self):
        assert "`a b`" in blank_string_literals("SELECT `a b`, 'x' FROM t")


class TestForbiddenKeywords:
    @pytest.mark.parametrize(
        "kw",
        [
            "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "GRANT", "MERGE", "CREATE",
            "REVOKE", "EXEC", "EXECUTE", "OPTIMIZE", "VACUUM", "COPY", "REFRESH", "RESTORE", "CLONE",
            "CALL", "USE", "SET", "RESET", "CACHE", "UNCACHE", "MSCK", "ANALYZE", "LOAD", "REPAIR",
            "ADD", "LIST", "PUT", "GET", "REMOVE",
        ],
    )
    def test_keyword_blocked(self, kw):
        ok, err = validate_sql(f"SELECT 1 FROM t WHERE x IN (SELECT 1) {kw} y")
        assert not ok and kw in err

    def test_replace_function_allowed(self):
        assert validate_sql("SELECT replace(a, 'x', 'y') FROM t")[0] is True

    def test_create_or_replace_blocked_by_create(self):
        assert validate_sql("SELECT 1; CREATE OR REPLACE VIEW v AS SELECT 1")[0] is False

    def test_function_names_containing_keywords_allowed(self):
        assert validate_sql("SELECT add_months(d, 1), date_add(d, 1), get_json_object(j, '$.a') FROM t")[0] is True

    def test_values_only_rejected(self):
        ok, err = validate_sql("VALUES (1), (2)")
        assert not ok and "VALUES" in err
        assert validate_sql("(VALUES (1))")[0] is False

    def test_values_inside_cte_allowed(self):
        assert validate_sql("WITH v AS (SELECT * FROM VALUES (1), (2) AS t(a)) SELECT a FROM v")[0] is True


class TestRestrictedSchemas:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM system.access.audit",
            "SELECT * FROM `system`.`access`.`audit`",
            "SELECT * FROM workspace.app.conversation_history",
            "SELECT * FROM app.query_corrections",
            "SELECT * FROM workspace.APP.eval_results",
            "SELECT * FROM workspace . app . eval_results",
            "SELECT * FROM workspace.information_schema.tables",
        ],
    )
    def test_blocked(self, sql):
        ok, err = validate_sql(sql)
        assert not ok and "restricted schema" in err

    def test_not_blocked_in_literal_or_longer_names(self):
        assert validate_sql("SELECT * FROM t WHERE note = 'app.x'")[0] is True
        assert validate_sql("SELECT * FROM workspace.sales.apple_orders")[0] is True
        assert validate_sql("SELECT mapp.x FROM t mapp")[0] is True
        assert validate_sql("SELECT * FROM workspace.hr.employees")[0] is True


class TestEnsureLimit:
    def test_appends_when_missing(self):
        assert "LIMIT 500" in ensure_limit("SELECT a FROM t")

    def test_custom_max_rows(self):
        assert ensure_limit("SELECT a FROM t", 25).endswith("LIMIT 25")

    def test_preserves_trailing_limit(self):
        sql = "SELECT a FROM t ORDER BY a DESC LIMIT 5"
        assert ensure_limit(sql) == sql

    def test_preserves_limit_offset(self):
        sql = "SELECT a FROM t LIMIT 10 OFFSET 20"
        assert ensure_limit(sql) == sql

    def test_strips_trailing_semicolon(self):
        out = ensure_limit("SELECT a FROM t;")
        assert "LIMIT 500" in out and ";" not in out

    def test_trailing_comment_does_not_swallow_limit(self):
        out = ensure_limit("SELECT a FROM t -- note")
        assert out.rstrip().endswith("LIMIT 500") and "\n" in out

    def test_limit_in_comment_does_not_count(self):
        assert ensure_limit("SELECT a FROM t /* LIMIT 5 */").rstrip().endswith("LIMIT 500")

    def test_limit_in_literal_does_not_count(self):
        assert ensure_limit("SELECT 'LIMIT 5' AS x FROM t").rstrip().endswith("LIMIT 500")

    def test_inner_cte_limit_still_gets_outer_cap(self):
        sql = (
            "WITH top10 AS (SELECT product_id FROM products ORDER BY rev DESC LIMIT 10) "
            "SELECT o.* FROM orders o JOIN top10 t ON o.product_id = t.product_id"
        )
        assert ensure_limit(sql).rstrip().endswith("LIMIT 500")

    def test_subquery_limit_still_gets_outer_cap(self):
        sql = "SELECT * FROM orders WHERE id IN (SELECT id FROM t LIMIT 3)"
        assert ensure_limit(sql).rstrip().endswith("LIMIT 500")


class TestStripComments:
    def test_removes_both_comment_styles(self):
        out = strip_sql_comments("SELECT 1 -- a\n/* b */ FROM t")
        assert "--" not in out and "/*" not in out and out.split() == ["SELECT", "1", "FROM", "t"]

    def test_keeps_markers_inside_literals(self):
        assert strip_sql_comments("SELECT 'a--b', \"c/*d*/\" FROM t") == "SELECT 'a--b', \"c/*d*/\" FROM t"
