"""Offline tests for plumbline.scoring."""

import pytest

from plumbline.scoring import (
    check_value_compliance,
    compute_schema_compliance,
    final_verdict,
    metric_view_misuse,
    parse_judge_response,
    unknown_measures,
)
from plumbline.semantic import MetricView

KNOWN = {"SALES.ORDERS.STATUS", "ORDERS.STATUS", "ORDERS", "SALES.ORDERS", "STATUS"}

MV = MetricView(
    name="mv_orders",
    schema="sales",
    description="orders",
    dimensions=[{"name": "order_month", "expr": "x", "comment": ""}],
    measures=[
        {"name": "order_revenue", "expr": "x", "comment": "", "synonyms": []},
        {"name": "total_orders", "expr": "x", "comment": "", "synonyms": []},
    ],
)
OTHER = MetricView(
    name="mv_budget",
    schema="finance",
    description="b",
    dimensions=[{"name": "dept_name", "expr": "x", "comment": ""}],
    measures=[{"name": "budget_amount", "expr": "x", "comment": "", "synonyms": []}],
)
REGISTRY = [MV, OTHER]


class TestSchemaCompliance:
    def test_known_identifiers_score_high(self):
        assert compute_schema_compliance("SELECT orders.status FROM sales.orders", KNOWN)[0] == 10

    def test_case_insensitive(self):
        assert compute_schema_compliance("select Orders.Status from SALES.orders", KNOWN)[0] == 10

    def test_unknown_identifier_lowers_score(self):
        score, reason = compute_schema_compliance("SELECT madeup.col FROM nowhere.tbl", KNOWN)
        assert score < 10 and "unrecognized" in reason

    def test_no_identifiers_scores_ten(self):
        assert compute_schema_compliance("SELECT 1", KNOWN)[0] == 10

    def test_backticked_names_are_checked(self):
        assert compute_schema_compliance("SELECT `orders`.`status` FROM `sales`.`orders`", KNOWN)[0] == 10

    def test_literals_are_not_identifiers(self):
        assert compute_schema_compliance("SELECT 'a.b.c' FROM sales.orders", KNOWN)[0] == 10

    def test_cte_and_alias_columns_are_not_penalised(self):
        sql = "WITH t AS (SELECT status AS s FROM sales.orders) SELECT t.s FROM t"
        assert compute_schema_compliance(sql, KNOWN)[0] == 10

    def test_metric_view_dimensions_and_measures_are_known(self):
        from plumbline.semantic import known_identifiers

        ids = KNOWN | known_identifiers(REGISTRY, "workspace")
        sql = "SELECT mv_orders.order_month, MEASURE(order_revenue) FROM workspace.sales.mv_orders GROUP BY ALL"
        assert compute_schema_compliance(sql, ids, REGISTRY)[0] == 10

    def test_unknown_measure_is_unknown_identifier(self):
        sql = "SELECT MEASURE(not_a_measure) FROM workspace.sales.mv_orders"
        score, reason = compute_schema_compliance(sql, KNOWN, REGISTRY)
        assert score < 10 and "not_a_measure" in reason

    def test_measure_of_other_view_is_unknown(self):
        # budget_amount belongs to mv_budget, but only mv_orders is in FROM
        assert unknown_measures("SELECT MEASURE(budget_amount) FROM workspace.sales.mv_orders", REGISTRY) == [
            "budget_amount"
        ]

    def test_measure_of_each_joined_view_is_known(self):
        sql = (
            "SELECT MEASURE(budget_amount) FROM workspace.finance.mv_budget "
            "UNION ALL SELECT MEASURE(order_revenue) FROM workspace.sales.mv_orders"
        )
        assert unknown_measures(sql, REGISTRY) == []


class TestValueCompliance:
    HINTS = {
        "sales.orders.status": ["Completed", "Cancelled", "Pending"],
        "sales.products.category": ["Electronics", "Apparel"],
        "finance.invoices.invoice_status": ["Paid", "Sent", "Overdue"],
    }

    def test_valid_value(self):
        assert check_value_compliance("SELECT * FROM orders WHERE status = 'Completed'", self.HINTS)[::2] == (10, [])

    def test_invalid_value_flagged(self):
        score, _, bad = check_value_compliance("SELECT * FROM orders WHERE status = 'Shipped'", self.HINTS)
        assert bad == ["Shipped"] and score < 10

    def test_qualified_column(self):
        assert check_value_compliance("SELECT * FROM orders o WHERE o.status = 'Nope'", self.HINTS)[2] == ["Nope"]

    def test_double_quoted_literal_checked(self):
        assert check_value_compliance('SELECT * FROM orders WHERE status = "Nope"', self.HINTS)[2] == ["Nope"]

    def test_in_list(self):
        assert check_value_compliance("SELECT * FROM o WHERE status IN ('Completed', 'Bogus')", self.HINTS)[2] == ["Bogus"]

    def test_case_insensitive_value(self):
        assert check_value_compliance("SELECT * FROM o WHERE status = 'completed'", self.HINTS)[2] == []

    def test_unknown_column_not_checked(self):
        assert check_value_compliance("SELECT * FROM t WHERE free_text = 'anything'", self.HINTS)[::2] == (10, [])

    def test_like_pattern_skipped(self):
        assert check_value_compliance("SELECT * FROM o WHERE status LIKE '%Complete%'", self.HINTS)[::2] == (10, [])

    def test_case_labels_are_not_flagged(self):
        sql = (
            "SELECT CASE WHEN d <= 30 THEN '0-30' WHEN d <= 60 THEN '31-60' ELSE '90+' END AS bucket, "
            "SUM(x) FROM invoices WHERE invoice_status = 'Overdue' GROUP BY 1"
        )
        assert check_value_compliance(sql, self.HINTS)[::2] == (10, [])

    def test_alias_label_not_flagged(self):
        assert check_value_compliance("SELECT 'Q1' AS quarter, SUM(x) FROM t", self.HINTS)[::2] == (10, [])

    def test_escaped_quote_in_literal(self):
        hints = {"hr.employees.job_title": ["Manager"]}
        assert check_value_compliance(r"SELECT 1 FROM e WHERE job_title = 'O\'Brien'", hints)[2] == ["O'Brien"]

    def test_comment_is_ignored(self):
        assert check_value_compliance("SELECT 1 FROM o -- status = 'Bogus'", self.HINTS)[::2] == (10, [])

    def test_no_hints(self):
        assert check_value_compliance("SELECT 1 FROM o WHERE status = 'x'", {})[0] == 10


class TestMetricViewMisuse:
    def test_clean_query(self):
        sql = "SELECT order_month, MEASURE(order_revenue) AS r FROM workspace.sales.mv_orders GROUP BY ALL"
        assert metric_view_misuse(sql, REGISTRY) == []

    def test_alias_equal_to_measure_name_is_fine(self):
        sql = (
            "SELECT order_month, MEASURE(order_revenue) AS order_revenue FROM workspace.sales.mv_orders "
            "GROUP BY ALL ORDER BY order_revenue DESC"
        )
        assert metric_view_misuse(sql, REGISTRY) == []

    def test_plain_aggregate_of_measure(self):
        sql = "SELECT order_month, SUM(order_revenue) FROM workspace.sales.mv_orders GROUP BY order_month"
        problems = metric_view_misuse(sql, REGISTRY)
        assert problems and "MEASURE(order_revenue)" in problems[0]

    def test_qualified_aggregate_of_measure(self):
        sql = "SELECT AVG(m.total_orders) FROM workspace.sales.mv_orders m"
        assert metric_view_misuse(sql, REGISTRY)

    def test_aggregate_around_measure_call(self):
        sql = "SELECT SUM(MEASURE(order_revenue)) FROM workspace.sales.mv_orders"
        assert any("wrapped" in p for p in metric_view_misuse(sql, REGISTRY))

    def test_select_star(self):
        assert metric_view_misuse("SELECT * FROM workspace.sales.mv_orders", REGISTRY)
        assert metric_view_misuse("SELECT DISTINCT * FROM workspace.sales.mv_orders", REGISTRY)

    def test_qualified_star_on_view_alias(self):
        assert metric_view_misuse("SELECT m.* FROM workspace.sales.mv_orders AS m", REGISTRY)

    def test_qualified_star_on_other_table_ok(self):
        sql = "SELECT c.* FROM workspace.sales.customers c WHERE c.customer_id IN (SELECT 1 FROM workspace.sales.mv_orders GROUP BY ALL)"
        assert metric_view_misuse(sql, REGISTRY) == []

    def test_count_star_is_not_select_star(self):
        sql = "SELECT order_month, COUNT(*) FROM workspace.sales.mv_orders GROUP BY order_month"
        assert metric_view_misuse(sql, REGISTRY) == []

    def test_non_metric_queries_never_flagged(self):
        assert metric_view_misuse("SELECT * FROM workspace.sales.orders", REGISTRY) == []
        assert metric_view_misuse("SELECT SUM(order_revenue) FROM workspace.sales.orders", REGISTRY) == []


class TestJudgeParsing:
    def test_parses_three_lines(self):
        raw = "RELEVANCE: 9/10 | answers it\nSQL_QUALITY: 8/10 | clean\nVERDICT: GOOD"
        j = parse_judge_response(raw)
        assert j["parsed"] and (j["relevance"], j["sql_quality"], j["judge_verdict"]) == (9, 8, "GOOD")
        assert j["relevance_reason"] == "answers it"

    def test_tolerates_markdown_decoration(self):
        raw = "**RELEVANCE:** 7/10 | ok\n- SQL_QUALITY: 6 / 10 | meh\nVERDICT: NEEDS REVIEW"
        j = parse_judge_response(raw)
        assert j["parsed"] and (j["relevance"], j["sql_quality"]) == (7, 6)

    def test_garbage_is_unparsed(self):
        assert parse_judge_response("I think it is fine")["parsed"] is False
        assert parse_judge_response("")["parsed"] is False


class TestFinalVerdict:
    GOOD_JUDGE = {"parsed": True, "relevance": 9, "sql_quality": 9}

    def test_good(self):
        assert final_verdict(10, 10, self.GOOD_JUDGE, []) == "GOOD"

    @pytest.mark.parametrize("schema,value", [(4, 10), (10, 4), (1, 1)])
    def test_risky_on_low_code_scores(self, schema, value):
        assert final_verdict(schema, value, self.GOOD_JUDGE, []) == "RISKY"

    def test_risky_on_misuse_even_with_perfect_scores(self):
        assert final_verdict(10, 10, self.GOOD_JUDGE, ["SELECT * on a metric view"]) == "RISKY"

    @pytest.mark.parametrize("schema,value", [(7, 10), (10, 7), (5, 10), (10, 5)])
    def test_needs_review_on_mid_code_scores(self, schema, value):
        assert final_verdict(schema, value, self.GOOD_JUDGE, []) == "NEEDS REVIEW"

    def test_needs_review_on_mid_judge_scores(self):
        assert final_verdict(10, 10, {"parsed": True, "relevance": 7, "sql_quality": 9}, []) == "NEEDS REVIEW"
        assert final_verdict(10, 10, {"parsed": True, "relevance": 9, "sql_quality": 7}, []) == "NEEDS REVIEW"

    def test_missing_judge_needs_review(self):
        assert final_verdict(10, 10, None, []) == "NEEDS REVIEW"

    def test_unparseable_judge_is_unscored(self):
        assert final_verdict(10, 10, {"parsed": False, "relevance": 0, "sql_quality": 0}, []) == "UNSCORED"

    def test_boundary_eight_is_good(self):
        assert final_verdict(8, 8, {"parsed": True, "relevance": 8, "sql_quality": 8}, []) == "GOOD"

    def test_misuse_none_is_accepted(self):
        assert final_verdict(10, 10, self.GOOD_JUDGE, None) == "GOOD"


# --- misuse patterns observed live (held-out run); each must be caught before execution ---------------
def _live_registry():
    from pathlib import Path

    from plumbline.semantic import load_registry

    return load_registry(Path(__file__).resolve().parent.parent / "app" / "semantic")


def test_measure_inside_case_is_flagged():
    from plumbline.scoring import metric_view_misuse

    sql = ("SELECT dept_name, AVG(CASE WHEN review_period = '2026-Q1' THEN avg_rating END) AS q1_avg "
           "FROM workspace.hr.mv_performance GROUP BY dept_name")
    assert any("avg_rating" in p for p in metric_view_misuse(sql, _live_registry()))


def test_measure_from_another_view_is_flagged_with_owner_hint():
    from plumbline.scoring import metric_view_misuse

    sql = "SELECT category, MEASURE(return_count) / MEASURE(line_count) * 100 AS r FROM workspace.sales.mv_returns GROUP BY category"
    problems = metric_view_misuse(sql, _live_registry())
    assert any("line_count" in p and "mv_order_lines" in p for p in problems)


def test_sum_over_measures_is_flagged():
    from plumbline.scoring import metric_view_misuse

    sql = "SELECT dept_name, SUM(total_overtime) / SUM(total_base) * 100 AS pct FROM workspace.hr.mv_payroll GROUP BY ALL"
    problems = metric_view_misuse(sql, _live_registry())
    assert any("total_overtime" in p for p in problems) and any("total_base" in p for p in problems)


def test_measure_alias_reuse_is_not_flagged():
    from plumbline.scoring import metric_view_misuse

    sql = ("SELECT product_name, MEASURE(net_revenue) AS net_revenue FROM workspace.sales.mv_order_lines "
           "WHERE category = 'Electronics' GROUP BY ALL ORDER BY net_revenue DESC LIMIT 5")
    assert metric_view_misuse(sql, _live_registry()) == []


def test_measure_in_having_through_measure_is_clean():
    from plumbline.scoring import metric_view_misuse

    sql = ("SELECT dept_name, MEASURE(actual_vs_budget_pct) AS pct FROM workspace.finance.mv_budget "
           "GROUP BY ALL HAVING MEASURE(actual_vs_budget_pct) > 10")
    assert metric_view_misuse(sql, _live_registry()) == []
