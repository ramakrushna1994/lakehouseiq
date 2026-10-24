"""End-to-end pipeline tests with a fake model client and a fake warehouse (offline)."""

from pathlib import Path

import pandas as pd
import pytest

from lakehouseiq.config import Settings
from lakehouseiq.pipeline import TIMING_KEYS, AnswerResult, answer_question
from lakehouseiq.semantic import load_registry
from lakehouseiq.warehouse import SchemaContext

REGISTRY = load_registry(Path(__file__).resolve().parent.parent / "app" / "semantic")
SETTINGS = Settings(catalog="cat", generator_endpoint="gen-ep", judge_endpoint="judge-ep", max_rows=500)
CTX = SchemaContext(
    tables_text="cat.sales.orders\n  - status (string)",
    views_text="",
    known_identifiers={"SALES.ORDERS", "CAT.SALES.ORDERS", "ORDERS", "STATUS", "CAT.SALES.ORDERS.STATUS"},
    enum_hints={"sales.orders.status": ["Completed", "Shipped", "Processing"]},
    as_of_date="2026-06-30",
)
GOOD_JUDGE = "RELEVANCE: 9/10 | answers it\nSQL_QUALITY: 9/10 | clean\nVERDICT: GOOD"
MV_SQL = (
    "SELECT order_month, MEASURE(order_revenue) AS rev FROM cat.sales.mv_orders GROUP BY ALL ORDER BY order_month"
)


def reply(sql, view=None):
    import json

    return json.dumps({"sql": sql, "view": view})


class FakeLLM:
    """Dispatches on the prompt's role line. ``script`` maps role -> reply or exception or list."""

    def __init__(self, **script):
        self.script = {"generate": reply(MV_SQL, "cat.sales.mv_orders"), "judge": GOOD_JUDGE, "summary": "**Headline**"}
        self.script.update(script)
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def role(prompt):
        for marker, role in (
            ("SQL quality reviewer", "judge"),
            ("SQL (Spark SQL) debugger", "repair"),
            ("data analyst", "summary"),
            ("SQL (Spark SQL) generator", "generate"),
        ):
            if marker in prompt:
                return role
        raise AssertionError("unrecognised prompt")

    def complete(self, prompt, endpoint, **kw):
        role = self.role(prompt)
        self.calls.append((role, endpoint))
        out = self.script[role]
        if isinstance(out, list):
            out = out.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    def roles(self):
        return [r for r, _ in self.calls]


class FakeWH:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.sqls: list[str] = []

    def run_query(self, sql, max_rows=None):
        self.sqls.append(sql)
        out = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        return out


DF = pd.DataFrame({"order_month": ["2026-05-01", "2026-06-01"], "rev": [10.0, 20.0]})
OK = (DF, None)


def run(llm, wh, question="Revenue by month?", **kw):
    return answer_question(question, settings=SETTINGS, schema_ctx=CTX, registry=REGISTRY, llm=llm, wh=wh, **kw)


def test_happy_path():
    llm, wh = FakeLLM(), FakeWH(OK)
    r = run(llm, wh)
    assert isinstance(r, AnswerResult)
    assert r.verdict == "GOOD" and r.error is None and r.blocked_reason is None and not r.repaired
    assert r.source_view == "cat.sales.mv_orders"
    assert r.sql.rstrip().endswith("LIMIT 500") and wh.sqls == [r.sql]
    assert r.df is DF and r.summary == "**Headline**"
    assert set(TIMING_KEYS) <= set(r.timings_ms) and r.timings_ms["total"] >= 0
    assert r.scores["schema"] == 10 and r.scores["relevance"] == 9 and r.scores["misuse"] == []
    assert llm.roles() == ["generate", "judge", "summary"]


def test_generator_and_judge_use_separate_endpoints():
    llm = FakeLLM()
    run(llm, FakeWH(OK))
    assert dict(llm.calls) == {"generate": "gen-ep", "judge": "judge-ep", "summary": "gen-ep"}


def test_prompt_receives_as_of_date_and_catalog():
    seen = []

    class Spy(FakeLLM):
        def complete(self, prompt, endpoint, **kw):
            seen.append(prompt)
            return super().complete(prompt, endpoint, **kw)

    run(Spy(), FakeWH(OK))
    gen = seen[0]
    assert "Today = 2026-06-30" in gen and "cat.sales.mv_order_lines" in gen and "cat.sales.mv_orders" in gen


def test_without_summary_skips_the_call():
    llm = FakeLLM()
    r = run(llm, FakeWH(OK), with_summary=False)
    assert r.summary is None and "summary" not in llm.roles()


def test_empty_result_skips_summary_but_is_still_scored():
    llm = FakeLLM()
    r = run(llm, FakeWH((DF.iloc[0:0], None)))
    assert r.summary is None and "summary" not in llm.roles() and r.verdict == "GOOD"


def test_repair_path():
    bad = reply("SELECT nope FROM cat.sales.orders")
    llm = FakeLLM(generate=bad, repair=reply(MV_SQL, "cat.sales.mv_orders"))
    wh = FakeWH((None, "UNRESOLVED_COLUMN nope"), OK)
    r = run(llm, wh)
    assert r.repaired and r.error is None and r.verdict == "GOOD"
    assert llm.roles() == ["generate", "repair", "judge", "summary"]
    assert len(wh.sqls) == 2 and "mv_orders" in r.sql and r.source_view == "cat.sales.mv_orders"
    assert r.timings_ms["repair"] >= 0


def test_repair_prompt_contains_the_warehouse_error():
    prompts_seen = []

    class Spy(FakeLLM):
        def complete(self, prompt, endpoint, **kw):
            prompts_seen.append(prompt)
            return super().complete(prompt, endpoint, **kw)

    run(Spy(repair=reply(MV_SQL)), FakeWH((None, "UNRESOLVED_COLUMN.WITH_SUGGESTION `nope`"), OK))
    assert any("UNRESOLVED_COLUMN.WITH_SUGGESTION" in p for p in prompts_seen)


def test_repair_that_also_fails_reports_error():
    llm = FakeLLM(repair=reply("SELECT 2 FROM cat.sales.orders"))
    r = run(llm, FakeWH((None, "first error"), (None, "second error")))
    assert r.repaired and r.error == "second error" and r.df is None and r.verdict == "ERROR"
    assert "judge" not in llm.roles()


def test_repair_producing_blocked_sql_keeps_original_error():
    llm = FakeLLM(repair=reply("DROP TABLE cat.sales.orders"))
    wh = FakeWH((None, "boom"))
    r = run(llm, wh)
    assert r.error == "boom" and not r.repaired and len(wh.sqls) == 1


def test_repair_llm_failure_keeps_original_error():
    from lakehouseiq.llm import LLMError

    r = run(FakeLLM(repair=LLMError("gen-ep", "down")), FakeWH((None, "boom")))
    assert r.error == "boom"


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE cat.sales.orders",
        "SELECT 1; DELETE FROM cat.sales.orders",
        "SELECT * FROM cat.app.query_corrections",
        "SELECT * FROM system.access.audit",
    ],
)
def test_blocked_sql_never_reaches_the_warehouse(sql):
    llm, wh = FakeLLM(generate=reply(sql)), FakeWH(OK)
    r = run(llm, wh)
    assert r.verdict == "BLOCKED" and r.blocked_reason and r.df is None
    assert wh.sqls == [] and llm.roles() == ["generate"]


@pytest.mark.parametrize("q", ["", "   ", "x" * 1001])
def test_input_guard(q):
    llm, wh = FakeLLM(), FakeWH(OK)
    r = run(llm, wh, question=q)
    assert r.verdict == "BLOCKED" and r.blocked_reason and llm.calls == [] and wh.sqls == []


def test_question_is_stripped():
    assert run(FakeLLM(), FakeWH(OK), question="  hi  ").question == "hi"


def test_judge_failure_is_needs_review_not_an_exception():
    r = run(FakeLLM(judge=RuntimeError("endpoint down")), FakeWH(OK))
    assert r.verdict == "NEEDS REVIEW" and r.df is not None
    assert r.scores["judge_error"] and r.scores["relevance"] is None


def test_unparseable_judge_is_unscored():
    r = run(FakeLLM(judge="looks fine to me"), FakeWH(OK))
    assert r.verdict == "UNSCORED" and r.scores["judge_parsed"] is False


def test_low_judge_scores_need_review():
    r = run(FakeLLM(judge="RELEVANCE: 6/10 | meh\nSQL_QUALITY: 9/10 | ok\nVERDICT: NEEDS REVIEW"), FakeWH(OK))
    assert r.verdict == "NEEDS REVIEW"


def test_metric_view_misuse_is_risky():
    sql = "SELECT order_month, SUM(order_revenue) FROM cat.sales.mv_orders GROUP BY order_month"
    r = run(FakeLLM(generate=reply(sql, "cat.sales.mv_orders")), FakeWH(OK))
    assert r.verdict == "RISKY" and r.scores["misuse"]


def test_select_star_on_metric_view_is_risky():
    r = run(FakeLLM(generate=reply("SELECT * FROM cat.sales.mv_orders")), FakeWH(OK))
    assert r.verdict == "RISKY"


def test_invented_filter_value_is_risky():
    sql = "SELECT count(*) FROM cat.sales.orders WHERE status = 'Vanished'"
    r = run(FakeLLM(generate=reply(sql)), FakeWH(OK))
    assert r.verdict == "RISKY" and r.scores["bad_values"] == ["Vanished"]


def test_unknown_table_is_risky():
    r = run(FakeLLM(generate=reply("SELECT x.y FROM cat.sales.ghost x")), FakeWH(OK))
    assert r.verdict == "RISKY"


def test_generation_failure_surfaces_error():
    from lakehouseiq.llm import LLMError

    r = run(FakeLLM(generate=LLMError("gen-ep", "429")), FakeWH(OK))
    assert r.verdict == "ERROR" and "generation failed" in r.error


def test_unparseable_generation():
    r = run(FakeLLM(generate="I cannot help with that."), FakeWH(OK))
    assert r.verdict == "ERROR" and "Could not generate" in r.error


def test_summary_failure_does_not_break_the_answer():
    r = run(FakeLLM(summary=RuntimeError("quota")), FakeWH(OK))
    assert r.summary is None and r.verdict == "GOOD" and r.df is not None


def test_existing_limit_is_kept():
    sql = "SELECT order_month, MEASURE(order_revenue) FROM cat.sales.mv_orders GROUP BY ALL LIMIT 5"
    r = run(FakeLLM(generate=reply(sql)), FakeWH(OK))
    assert r.sql == sql


def test_view_outside_the_published_set_is_dropped():
    r = run(FakeLLM(generate=reply(MV_SQL, "cat.sales.invented_view")), FakeWH(OK))
    assert r.source_view is None


def test_history_and_corrections_reach_the_generator():
    seen = []

    class Spy(FakeLLM):
        def complete(self, prompt, endpoint, **kw):
            seen.append(prompt)
            return super().complete(prompt, endpoint, **kw)

    hist = [{"question": "earlier question", "sql": "SELECT 1"}]
    corr = [{"question": "q", "bad_sql": "bad", "corrected_sql": "good", "reason": "why"}]
    run(Spy(), FakeWH(OK), history=hist, corrections=corr)
    assert "earlier question" in seen[0] and "Known Corrections" in seen[0]
