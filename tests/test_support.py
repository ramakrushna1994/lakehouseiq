"""Offline tests for config, llm retry, tracing, store ranking and warehouse helpers."""

from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from plumbline import llm, store, tracing, warehouse
from plumbline.config import Settings

# ---- config ---------------------------------------------------------------- #


def test_settings_defaults_and_no_warehouse_needed_at_import():
    s = Settings.from_env({})
    assert (s.catalog, s.max_rows, s.query_timeout_s) == ("workspace", 500, 120)
    assert s.warehouse_id is None and s.as_of_date is None and s.mlflow_experiment_id is None
    with pytest.raises(RuntimeError):
        s.require_warehouse_id()


def test_settings_from_env():
    s = Settings.from_env(
        {
            "CATALOG": "main", "DATABRICKS_WAREHOUSE_ID": "abc", "GENERATOR_ENDPOINT": "g", "JUDGE_ENDPOINT": "j",
            "MAX_ROWS": "50", "QUERY_TIMEOUT_S": "9", "MLFLOW_EXPERIMENT_ID": "7", "AS_OF_DATE": "2026-06-30",
        }
    )
    assert s.require_warehouse_id() == "abc" and s.fq("sales", "orders") == "main.sales.orders"
    assert (s.generator_endpoint, s.judge_endpoint, s.max_rows, s.mlflow_experiment_id) == ("g", "j", 50, "7")


def test_blank_env_values_fall_back_to_defaults():
    assert Settings.from_env({"CATALOG": "  ", "JUDGE_ENDPOINT": ""}).catalog == "workspace"


def test_catalog_must_be_an_identifier():
    with pytest.raises(ValueError):
        Settings.from_env({"CATALOG": "x; DROP"})


# ---- llm ------------------------------------------------------------------- #


class Resp:
    def __init__(self, content):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class FakeClient:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        self.kw = kw
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class HttpError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


@pytest.fixture
def fast(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm, "_sleep", sleeps.append)

    def install(client):
        monkeypatch.setattr(llm, "get_client", lambda: client)
        return client

    return SimpleNamespace(install=install, sleeps=sleeps)


def test_complete_returns_text(fast):
    c = fast.install(FakeClient(Resp("  hello  ")))
    assert llm.complete("hi", "ep") == "hello"
    assert c.kw["model"] == "ep" and c.kw["messages"] == [{"role": "user", "content": "hi"}] and c.kw["temperature"] == 0.0


def test_structured_content_keeps_text_parts_only(fast):
    parts = [{"type": "reasoning", "summary": [{"text": "thinking"}]}, {"type": "text", "text": "SELECT 1"}]
    fast.install(FakeClient(Resp(parts)))
    assert llm.complete("q", "ep") == "SELECT 1"


def test_retries_429_and_5xx_with_backoff(fast):
    c = fast.install(FakeClient(HttpError(429), HttpError(503), Resp("ok")))
    assert llm.complete("q", "ep") == "ok"
    assert c.calls == 3 and len(fast.sleeps) == 2
    assert 2 <= fast.sleeps[0] < 3 and 4 <= fast.sleeps[1] < 5


def test_gives_up_after_three_retries_and_names_endpoint(fast):
    c = fast.install(FakeClient(*[HttpError(500)] * 4))
    with pytest.raises(llm.LLMError, match="my-endpoint"):
        llm.complete("q", "my-endpoint")
    assert c.calls == 4 and len(fast.sleeps) == 3


def test_does_not_retry_client_errors(fast):
    c = fast.install(FakeClient(HttpError(400)))
    with pytest.raises(llm.LLMError):
        llm.complete("q", "ep")
    assert c.calls == 1 and fast.sleeps == []


def test_empty_content_raises(fast):
    fast.install(FakeClient(Resp(""), Resp(None)))
    with pytest.raises(llm.LLMError, match="empty"):
        llm.complete("q", "ep")
    with pytest.raises(llm.LLMError, match="empty"):
        llm.complete("q", "ep")


def test_llm_import_needs_no_openai_or_databricks():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(llm.__file__).read_text())
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert {getattr(n, "module", None) or n.names[0].name for n in top_level} <= {"__future__", "logging", "random", "threading", "time"}


# ---- tracing ---------------------------------------------------------------- #


def test_tracing_is_a_noop_without_an_experiment():
    tracing.reset()
    assert tracing.configure(None) is False
    with tracing.span("x", a=1) as s:
        s.set(b=2)
        assert s.trace_id is None


def test_tracing_never_swallows_body_exceptions():
    with pytest.raises(ZeroDivisionError), tracing.span("x"):
        raise ZeroDivisionError


def test_tracing_survives_a_broken_mlflow():
    class Boom:
        def start_span(self, **kw):
            raise RuntimeError("mlflow down")

    tracing._state.update(configured=True, mlflow=Boom())
    try:
        with tracing.span("x") as s:
            s.set(a=1)
    finally:
        tracing.reset()


def test_traced_decorator():
    @tracing.traced("f")
    def f(x):
        return x + 1

    assert f(1) == 2


# ---- store ranking ----------------------------------------------------------- #


ROWS = [
    {"question": "What is the total revenue by region?", "bad_sql": "b1", "corrected_sql": "c1"},
    {"question": "Average salary by department", "bad_sql": "b2", "corrected_sql": "c2"},
    {"question": "Top vendors by expense amount", "bad_sql": "b3", "corrected_sql": "c3"},
]


def test_rank_returns_similar_corrections_only():
    out = store.rank_corrections("total revenue by region please", ROWS)
    assert [r["corrected_sql"] for r in out] == ["c1"]


def test_rank_has_no_recent_fallback():
    assert store.rank_corrections("completely unrelated astronomy question", ROWS) == []


def test_rank_respects_limit_and_order():
    rows = [
        {"question": "revenue by region", "corrected_sql": "weak"},
        {"question": "revenue by region and segment", "corrected_sql": "weaker"},
        {"question": "revenue by region", "corrected_sql": "same"},
    ]
    out = store.rank_corrections("revenue by region", rows, limit=2)
    assert [r["corrected_sql"] for r in out] == ["weak", "same"]


def test_rank_threshold_is_strict():
    assert store.jaccard({"a", "b", "c", "d", "e"}, {"a", "x", "y", "z", "w"}) < 0.2
    assert store.rank_corrections("alpha beta gamma delta epsilon", [{"question": "alpha one two three four"}]) == []


def test_tokenize_drops_stopwords():
    assert store.tokenize("What is the Revenue of the Top regions?") == {"revenue", "regions"}


class RecordingWH:
    def __init__(self, fail=False, df=None):
        self.fail, self.df, self.calls = fail, df, []

    def execute_params(self, sql, params):
        self.calls.append((sql, params))
        if self.fail:
            raise RuntimeError("warehouse down")

    def run_query(self, sql, max_rows=None):
        return (self.df, None) if self.df is not None else (None, "boom")


def test_save_turn_uses_named_parameters_only():
    wh = RecordingWH()
    s = Settings(catalog="cat")
    hostile = "x'); DROP TABLE t; --"
    ok = store.save_turn(
        settings=s, session_id="s", turn_id=1, user_email="u@x.com", question=hostile, generated_sql="SELECT 1",
        source_view=None, verdict="GOOD", row_count=3, error=None, latency_ms=12, wh=wh,
    )
    sql, params = wh.calls[0]
    assert ok and "cat.app.conversation_history" in sql and params["question"] == hostile
    assert "DROP" not in sql and all(f":{k}" in sql for k in params)


def test_save_correction_uses_named_parameters_only():
    wh = RecordingWH()
    store.save_correction(
        settings=Settings(catalog="cat"), user_email="u", question="q", bad_sql="b", corrected_sql="c", bad_values=["x", "y"], wh=wh
    )
    sql, params = wh.calls[0]
    assert "cat.app.query_corrections" in sql and params["bad_values"] == "x, y" and len(params["correction_id"]) == 36
    assert "'" not in sql.split("VALUES")[1]


def test_store_failures_are_swallowed():
    wh = RecordingWH(fail=True)
    s = Settings(catalog="cat")
    assert store.save_turn(
        settings=s, session_id="s", turn_id=1, user_email="u", question="q", generated_sql=None, source_view=None,
        verdict=None, row_count=None, error="e", latency_ms=None, wh=wh,
    ) is False
    assert store.save_correction(settings=s, user_email="u", question="q", bad_sql="b", corrected_sql="c", wh=wh) is False
    assert store.get_relevant_corrections("q", settings=s, wh=RecordingWH(df=None)) == []


def test_get_relevant_corrections_ranks_rows():
    df = pd.DataFrame(ROWS).assign(reason="", bad_values="")
    out = store.get_relevant_corrections("total revenue by region", settings=Settings(catalog="cat"), wh=RecordingWH(df=df))
    assert [r["corrected_sql"] for r in out] == ["c1"]


# ---- warehouse helpers -------------------------------------------------------- #


def test_enum_query_is_a_single_union_with_limits():
    q = warehouse.build_enum_query("cat")
    assert q.count("UNION ALL") == len(warehouse.ENUM_CANDIDATES) - 1
    assert "'sales.orders.status' AS k" in q and "cat.sales.orders" in q and q.count("LIMIT 15") == len(warehouse.ENUM_CANDIDATES)


def test_enum_candidates_are_lowercase_and_unique():
    assert len(set(warehouse.ENUM_CANDIDATES)) == len(warehouse.ENUM_CANDIDATES) >= 20
    assert all(p == p.lower() for c in warehouse.ENUM_CANDIDATES for p in c)


def test_decimal_columns_become_float():
    df = warehouse._coerce(pd.DataFrame({"a": [Decimal("1.50"), None], "b": ["x", "y"]}))
    assert df["a"].dtype.kind == "f" and df["a"].iloc[0] == 1.5


def test_render_includes_comments_and_enum_values():
    objs = {"c.s.t": {"comment": "tbl", "cols": [("status", "string", "order status", "s.t.status")]}}
    out = warehouse._render(objs, {"s.t.status": ["A", "B"]})
    assert "c.s.t  -- tbl" in out and "order status; Values: A, B" in out


def test_connection_error_classification():
    assert warehouse._is_connection_error(ConnectionError("x"))
    assert warehouse._is_connection_error(RuntimeError("Invalid SessionHandle: abc"))
    assert not warehouse._is_connection_error(ValueError("PARSE_SYNTAX_ERROR"))


def test_run_query_reconnects_once(monkeypatch):
    calls = []

    def fake_fetch(sql, max_rows, params=None):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("dropped")
        return pd.DataFrame({"a": [1]})

    monkeypatch.setattr(warehouse, "_fetch", fake_fetch)
    monkeypatch.setattr(warehouse, "close", lambda: None)
    df, err = warehouse.run_query("SELECT 1", 10)
    assert err is None and len(df) == 1 and len(calls) == 2


def test_run_query_returns_sql_errors_without_retry(monkeypatch):
    calls = []

    def fake_fetch(sql, max_rows, params=None):
        calls.append(1)
        raise ValueError("PARSE_SYNTAX_ERROR")

    monkeypatch.setattr(warehouse, "_fetch", fake_fetch)
    df, err = warehouse.run_query("SELEC", 10)
    assert df is None and "PARSE_SYNTAX_ERROR" in err and len(calls) == 1
