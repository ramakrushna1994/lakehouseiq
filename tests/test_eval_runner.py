"""Eval harness tests using fakes: no warehouse, model or Genie access."""

from types import SimpleNamespace

import pandas as pd

from eval import runner
from plumbline.pipeline import AnswerResult


def _q(i="t-01", order=False):
    return {"id": i, "question": "How many?", "reference_sql": "SELECT 1 AS n", "order_matters": order}


def _run_query(sql, max_rows):
    if "BAD" in sql:
        return None, "boom"
    return pd.DataFrame({"n": [3]}), None


def _answer(df, verdict="GOOD", repaired=False, error=None):
    def fn(question):
        return AnswerResult(question=question, sql="SELECT 3 AS n", df=df, verdict=verdict, repaired=repaired,
                            error=error, scores={"relevance": 9}, timings_ms={"total": 120})
    return fn


def test_questions_load_with_substitution():
    qs = runner.load_questions(catalog="cat", as_of_date="2026-01-31")
    assert len(qs) == 30
    assert all("${" not in q["reference_sql"] for q in qs)
    assert len({q["id"] for q in qs}) == 30


def test_load_questions_rejects_bad_catalog():
    import pytest

    with pytest.raises(ValueError):
        runner.load_questions(catalog="x; DROP", as_of_date="2026-01-31")


def test_assistant_match_and_mismatch():
    rows = runner.run_eval([_q()], answer_fn=_answer(pd.DataFrame({"count": [3]})), run_query=_run_query,
                           run_id="r", sleep_s=0)
    assert rows[0]["exec_match"] is True and rows[0]["judge_relevance"] == 9
    rows = runner.run_eval([_q()], answer_fn=_answer(pd.DataFrame({"count": [4]})), run_query=_run_query,
                           run_id="r", sleep_s=0)
    assert rows[0]["exec_match"] is False and "mismatch" in rows[0]["error"]


def test_pipeline_exception_is_recorded():
    def boom(question):
        raise RuntimeError("quota")

    rows = runner.run_eval([_q()], answer_fn=boom, run_query=_run_query, run_id="r", sleep_s=0)
    assert rows[0]["verdict"] == "ERROR" and "quota" in rows[0]["error"]


def test_reference_failure_never_counts_as_match():
    q = _q()
    q["reference_sql"] = "BAD"
    rows = runner.run_eval([q], answer_fn=_answer(pd.DataFrame({"n": [3]})), run_query=_run_query, run_id="r", sleep_s=0)
    assert rows[0]["exec_match"] is False and "reference SQL failed" in rows[0]["error"]


def test_resume_and_limit_and_callback():
    seen = []
    qs = [_q("a"), _q("b"), _q("c")]
    rows = runner.run_eval(qs, answer_fn=_answer(pd.DataFrame({"n": [3]})), run_query=_run_query, run_id="r",
                           done={("assistant", "a")}, limit=2, sleep_s=0, on_rows=seen.append)
    assert [r["question_id"] for r in rows] == ["b"]
    assert len(seen) == 1


def test_genie_paths():
    ok = lambda q: ("SELECT 3 AS n", None)  # noqa: E731
    row = runner.evaluate_genie(_q(), pd.DataFrame({"n": [3]}), ask_fn=ok, run_query=_run_query, run_id="r")
    assert row["exec_match"] is True
    blocked = runner.evaluate_genie(_q(), None, ask_fn=lambda q: ("DROP TABLE x", None), run_query=_run_query, run_id="r")
    assert blocked["verdict"] == "BLOCKED"
    err = runner.evaluate_genie(_q(), None, ask_fn=lambda q: (None, "down"), run_query=_run_query, run_id="r")
    assert err["verdict"] == "ERROR" and err["error"] == "down"


def test_extract_genie_sql_defensive():
    assert runner.extract_genie_sql(SimpleNamespace(attachments=None)) is None
    msg = SimpleNamespace(attachments=[SimpleNamespace(query=None),
                                       SimpleNamespace(query=SimpleNamespace(query=" SELECT 1 "))])
    assert runner.extract_genie_sql(msg) == "SELECT 1"


def test_ask_genie_never_raises():
    class W:
        genie = SimpleNamespace(start_conversation_and_wait=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))

    sql, err = runner.ask_genie(W(), "space", "q")
    assert sql is None and "genie call failed" in err


def test_summarize():
    base = {"verdict": "GOOD", "latency_ms": 100, "repaired": False}
    rows = [
        {**base, "system": "assistant", "exec_match": True},
        {**base, "system": "assistant", "exec_match": False, "verdict": "BLOCKED", "latency_ms": 300, "repaired": True},
        {**base, "system": "genie", "exec_match": True},
    ]
    m = runner.summarize(rows)
    assert m["exec_accuracy_assistant"] == 0.5 and m["blocked_rate"] == 0.5 and m["repair_rate"] == 0.5
    assert m["p50_latency_ms"] == 200.0 and m["exec_accuracy_genie"] == 1.0


def test_insert_uses_named_params_only():
    calls = []
    row = runner._row("r", "assistant", _q(), generated_sql="SELECT 'x'; --", error="it's bad")
    runner.insert_rows(lambda sql, p: calls.append((sql, p)), "cat", [row])
    sql, params = calls[0]
    assert "it's" not in sql and "SELECT 'x'" not in sql
    assert set(params) == set(runner.TABLE_COLUMNS) and params["error"] == "it's bad"


def test_run_id_validation():
    import pytest

    with pytest.raises(ValueError):
        runner._check_run_id("a' OR '1'='1")


def test_summarize_reports_holdout_separately():
    from eval.runner import summarize

    rows = [
        {"system": "assistant", "question_id": "sales-01", "exec_match": True, "verdict": "GOOD", "latency_ms": 10},
        {"system": "assistant", "question_id": "holdout-01", "exec_match": False, "verdict": "GOOD", "latency_ms": 20},
        {"system": "assistant", "question_id": "holdout-02", "exec_match": True, "verdict": "GOOD", "latency_ms": 30},
    ]
    m = summarize(rows)
    assert m["exec_accuracy_assistant_curated"] == 1.0
    assert m["exec_accuracy_assistant_holdout"] == 0.5
    assert abs(m["exec_accuracy_assistant"] - 2 / 3) < 1e-9


def test_holdout_file_loads_and_substitutes_catalog():
    from eval.runner import HOLDOUT_PATH, load_questions

    qs = load_questions(HOLDOUT_PATH, catalog="cat_x")
    assert len(qs) == 10 and all(q["id"].startswith("holdout-") for q in qs)
    assert all("${catalog}" not in q["reference_sql"] and "cat_x." in q["reference_sql"] for q in qs)


def test_holdout_v2_has_30_frozen_questions_with_definitions():
    from eval.runner import HOLDOUT_V2_PATH, load_questions, split_of

    qs = load_questions(HOLDOUT_V2_PATH, catalog="cat_x")
    assert len(qs) == 30 and len({q["id"] for q in qs}) == 30
    assert all(split_of(q["id"]) == "holdout_v2" for q in qs)
    assert split_of("holdout-03") == "holdout" and split_of("sales-01") == "curated"


def test_summarize_reports_v2_split():
    from eval.runner import summarize

    rows = [
        {"system": "assistant", "question_id": "h2-01", "exec_match": True, "verdict": "GOOD", "latency_ms": 1},
        {"system": "assistant", "question_id": "h2-02", "exec_match": False, "verdict": "GOOD", "latency_ms": 1},
    ]
    assert summarize(rows)["exec_accuracy_assistant_holdout_v2"] == 0.5


def test_genie_question_carries_the_as_of_date():
    from eval.runner import genie_question

    q = genie_question("What is net revenue by month for the last 6 months?", "2026-06-30")
    assert q.startswith("Treat 2026-06-30 as today's date") and q.endswith("last 6 months?")


def test_genie_errors_reported_separately():
    from eval.runner import summarize

    rows = [
        {"system": "genie", "question_id": "h2-01", "exec_match": False, "verdict": "ERROR", "latency_ms": 1},
        {"system": "genie", "question_id": "h2-02", "exec_match": True, "verdict": "GOOD", "latency_ms": 1},
    ]
    assert summarize(rows)["error_rate_genie"] == 0.5
