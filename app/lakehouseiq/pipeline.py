"""The only orchestration: question in, :class:`AnswerResult` out.

Both the Streamlit app and the eval harness call :func:`answer_question`, so the
numbers the eval reports are the numbers users get. There is no Streamlit import here;
the model client and the warehouse are injectable so the whole flow runs offline with
fakes.

Verdicts: GOOD, NEEDS REVIEW, RISKY, UNSCORED (judge reply unparseable), BLOCKED
(rejected by the guardrails or the input check) and ERROR (nothing usable was produced).
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import pandas as pd

from . import llm as llm_module
from . import prompts, scoring, tracing
from . import semantic as semantic_module
from . import warehouse as warehouse_module
from .config import Settings
from .guardrails import ensure_limit, validate_sql
from .parsing import parse_sql_response
from .warehouse import SchemaContext

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1000
TIMING_KEYS = ("generate", "execute", "repair", "judge", "summary", "total")


@dataclass
class AnswerResult:
    question: str
    sql: str | None = None
    source_view: str | None = None
    df: pd.DataFrame | None = None
    error: str | None = None
    blocked_reason: str | None = None
    repaired: bool = False
    scores: dict = field(default_factory=dict)
    verdict: str = "ERROR"
    summary: str | None = None
    timings_ms: dict = field(default_factory=lambda: dict.fromkeys(TIMING_KEYS, 0))
    trace_id: str | None = None


@contextmanager
def _step(name: str, timings: dict, **attrs):
    """Time a step and wrap it in a tracing span. Timing accumulates if a step repeats."""
    start = time.perf_counter()
    with tracing.span(name, **attrs) as s:
        try:
            yield s
        finally:
            timings[name] = timings.get(name, 0) + int((time.perf_counter() - start) * 1000)


def _context_kwargs(settings: Settings, schema_ctx: SchemaContext, registry) -> dict:
    return dict(
        catalog=settings.catalog,
        as_of_date=schema_ctx.as_of_date,
        metric_views_context=semantic_module.render_prompt_context(registry, settings.catalog),
        tables_text=schema_ctx.tables_text,
        views_text=schema_ctx.views_text,
    )


def _generate(question, settings, schema_ctx, registry, history, corrections, llm, known_views):
    kwargs = _context_kwargs(settings, schema_ctx, registry)
    prompt = prompts.build_generate_prompt(
        question,
        cross_fact_views={f"{settings.catalog}.{k}": v for k, v in semantic_module.CROSS_FACT_VIEWS.items()},
        history=history,
        corrections=corrections,
        **kwargs,
    )
    return parse_sql_response(llm.complete(prompt, settings.generator_endpoint), known_views)


def _repair(question, failed_sql, error, settings, schema_ctx, registry, llm, known_views):
    prompt = prompts.build_retry_prompt(
        question, failed_sql, error, **_context_kwargs(settings, schema_ctx, registry)
    )
    return parse_sql_response(llm.complete(prompt, settings.generator_endpoint), known_views)


def _score(question, sql, df, settings, schema_ctx, registry, llm, timings) -> tuple[dict, str]:
    known = set(schema_ctx.known_identifiers) | semantic_module.known_identifiers(registry, settings.catalog)
    schema_score, schema_reason = scoring.compute_schema_compliance(sql, known, registry)
    value_score, value_reason, bad_values = scoring.check_value_compliance(sql, schema_ctx.enum_hints)
    misuse = scoring.metric_view_misuse(sql, registry)

    judge = None
    judge_error = None
    with _step("judge", timings, endpoint=settings.judge_endpoint):
        try:
            # Reasoning models spend max_tokens on thinking, so leave generous headroom.
            raw = llm.complete(prompts.build_judge_prompt(question, sql), settings.judge_endpoint, max_tokens=3000)
            judge = scoring.parse_judge_response(raw)
        except Exception as exc:  # the judge must never break an answer
            judge_error = str(exc)
            logger.warning("judge failed: %s", exc)

    scores = {
        "schema": schema_score,
        "schema_reason": schema_reason,
        "value": value_score,
        "value_reason": value_reason,
        "bad_values": bad_values,
        "misuse": misuse,
        "relevance": judge["relevance"] if judge else None,
        "relevance_reason": judge["relevance_reason"] if judge else (judge_error or "judge unavailable"),
        "sql_quality": judge["sql_quality"] if judge else None,
        "sql_quality_reason": judge["sql_quality_reason"] if judge else (judge_error or "judge unavailable"),
        "judge_verdict": judge["judge_verdict"] if judge else None,
        "judge_parsed": bool(judge and judge["parsed"]),
        "judge_error": judge_error,
    }
    return scores, scoring.final_verdict(schema_score, value_score, judge, misuse)


def _summarise(question, df, settings, llm) -> str | None:
    try:
        prompt = prompts.build_summary_prompt(
            question, df.head(20).to_string(index=False), col_info=", ".join(map(str, df.columns))
        )
        return llm.complete(prompt, settings.generator_endpoint, max_tokens=700)
    except Exception as exc:
        logger.warning("summary failed: %s", exc)
        return None


def answer_question(
    question: str,
    *,
    settings: Settings,
    schema_ctx: SchemaContext,
    registry,
    history: list[dict] | None = None,
    corrections: list[dict] | None = None,
    llm=llm_module,
    wh=warehouse_module,
    with_summary: bool = True,
) -> AnswerResult:
    """Generate, validate, execute (with one repair), score and summarise one question."""
    t0 = time.perf_counter()
    question = (question or "").strip()
    result = AnswerResult(question=question)
    timings = result.timings_ms

    def finish() -> AnswerResult:
        timings["total"] = int((time.perf_counter() - t0) * 1000)
        return result

    with tracing.span("answer_question", question=question[:200]) as root:
        result.trace_id = root.trace_id

        # 1. input guard
        if not question or len(question) > MAX_QUESTION_CHARS:
            result.blocked_reason = f"Question must be between 1 and {MAX_QUESTION_CHARS} characters."
            result.verdict = "BLOCKED"
            return finish()

        known_views = semantic_module.known_view_fqns(registry, settings.catalog)

        # 2. generate -> parse -> validate -> cap
        with _step("generate", timings, endpoint=settings.generator_endpoint):
            try:
                parsed = _generate(question, settings, schema_ctx, registry, history, corrections, llm, known_views)
            except Exception as exc:
                result.error = f"Query generation failed: {exc}"
                return finish()
        if not parsed["sql"]:
            result.error = f"Could not generate a query: {parsed['error'] or 'empty response'}"
            return finish()
        ok, why = validate_sql(parsed["sql"])
        if not ok:
            result.sql, result.blocked_reason, result.verdict = parsed["sql"], why, "BLOCKED"
            return finish()
        sql = ensure_limit(parsed["sql"], settings.max_rows)
        result.sql, result.source_view = sql, parsed["view"]

        # 3. execute, with one repair attempt
        with _step("execute", timings):
            df, err = wh.run_query(sql, settings.max_rows)
        if err:
            with _step("repair", timings):
                try:
                    fixed = _repair(question, sql, err, settings, schema_ctx, registry, llm, known_views)
                except Exception as exc:
                    fixed = {"sql": "", "view": None, "error": str(exc)}
                if fixed["sql"]:
                    ok, why = validate_sql(fixed["sql"])
                    if ok:
                        repaired_sql = ensure_limit(fixed["sql"], settings.max_rows)
                        with tracing.span("execute_repaired"):  # timed as part of "repair"
                            df2, err2 = wh.run_query(repaired_sql, settings.max_rows)
                        result.repaired = True
                        result.sql, result.source_view = repaired_sql, fixed["view"]
                        df, err = df2, err2
                    else:
                        logger.warning("repaired SQL blocked: %s", why)
        if err or df is None:
            result.error = err or "Query returned no result set."
            return finish()
        result.df = df

        # 4. score: code checks plus an independent judge
        result.scores, result.verdict = _score(question, result.sql, df, settings, schema_ctx, registry, llm, timings)

        # 5. summary
        if with_summary and len(df) > 0:
            with _step("summary", timings, endpoint=settings.generator_endpoint):
                result.summary = _summarise(question, df, settings, llm)
        root.set(verdict=result.verdict, repaired=result.repaired, rows=len(df))
        return finish()
