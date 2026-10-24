"""Eval harness: run the assistant (and optionally Genie) on curated questions and score them.

For every question the harness

1. runs the reference SQL and the system's SQL on the same warehouse,
2. compares the two result sets with :func:`eval.compare.results_match` (values, not column names),
3. records one row per question per system in ``app.eval_results`` and one MLflow run.

The assistant is called through :func:`lakehouseiq.pipeline.answer_question`, the same function the
app uses, so the accuracy shown here is the accuracy users get. Genie's SQL is validated with the same
guardrails and executed through the same runner, so both systems are judged on identical data.

Usage (from the repo root, with DATABRICKS_* auth and DATABRICKS_WAREHOUSE_ID set)::

    python -m eval.runner --limit 5
    python -m eval.runner --run-id nightly-1 --genie-space-id 01ef...

Reference SQL requires human review before the numbers are published (see questions.yaml).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import yaml

from eval.compare import results_match

logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.yaml"
HOLDOUT_PATH = Path(__file__).resolve().parent / "holdout.yaml"  # v1: used once to choose the generator model
HOLDOUT_V2_PATH = Path(__file__).resolve().parent / "holdout_v2.yaml"  # frozen; the number to report
SPLIT_PREFIXES = (("h2-", "holdout_v2"), ("holdout-", "holdout"))  # anything else is "curated"
TABLE_COLUMNS = (
    "run_id", "ts", "system", "question_id", "question", "generated_sql", "exec_match",
    "judge_relevance", "verdict", "latency_ms", "error",
)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

Runner = Callable[[str, "int | None"], tuple]  # (sql, max_rows) -> (DataFrame | None, error | None)


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #


def load_questions(path: str | Path = QUESTIONS_PATH, catalog: str = "workspace", as_of_date: str = "2026-06-30") -> list[dict]:
    """Load questions with ``${catalog}`` and ``${as_of_date}`` substituted in the reference SQL."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", catalog):
        raise ValueError(f"catalog must be a simple identifier, got {catalog!r}")
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", as_of_date):
        raise ValueError(f"as_of_date must be YYYY-MM-DD, got {as_of_date!r}")
    questions = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for q in questions:
        q["reference_sql"] = q["reference_sql"].replace("${catalog}", catalog).replace("${as_of_date}", as_of_date)
    return questions


# --------------------------------------------------------------------------- #
# Per-question evaluation
# --------------------------------------------------------------------------- #


def _row(run_id: str, system: str, q: dict, **kw) -> dict:
    row = {
        "run_id": run_id,
        "ts": dt.datetime.now(dt.timezone.utc),
        "system": system,
        "question_id": q["id"],
        "question": q["question"],
        "generated_sql": None,
        "exec_match": False,
        "judge_relevance": None,
        "verdict": None,
        "latency_ms": None,
        "error": None,
        "repaired": False,
    }
    row.update(kw)
    return row


def evaluate_assistant(q: dict, expected, *, answer_fn, run_id: str) -> dict:
    """Score the assistant on one question. ``expected`` is the reference result DataFrame (or None)."""
    start = time.perf_counter()
    try:
        res = answer_fn(q["question"])
    except Exception as exc:  # the harness must survive any single failure
        return _row(run_id, "assistant", q, error=f"pipeline raised: {exc}", verdict="ERROR",
                    latency_ms=int((time.perf_counter() - start) * 1000))
    latency = res.timings_ms.get("total") or int((time.perf_counter() - start) * 1000)
    row = _row(
        run_id, "assistant", q,
        generated_sql=res.sql, verdict=res.verdict, latency_ms=latency, repaired=res.repaired,
        judge_relevance=res.scores.get("relevance") if res.scores else None,
        error=res.error or res.blocked_reason,
    )
    if res.df is not None and expected is not None:
        row["exec_match"], reason = results_match(expected, res.df, bool(q.get("order_matters")),
            ignore_zero_rows=bool(q.get("ignore_zero_rows")))
        if not row["exec_match"]:
            row["error"] = f"result mismatch: {reason}"
    return row


def extract_genie_sql(message) -> str | None:
    """Pull the query text out of a Genie message, tolerating SDK shape changes."""
    for att in getattr(message, "attachments", None) or []:
        query = getattr(att, "query", None)
        text = getattr(query, "query", None) if query is not None else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def genie_question(question: str, as_of_date: str) -> str:
    """Give Genie the same "today" the assistant's prompt gets, so relative dates resolve against the data.

    Without this, "last 6 months" resolves against the calendar date and the frozen demo data looks empty,
    which penalises Genie for an artifact of the eval rather than a wrong answer.
    """
    return f"Treat {as_of_date} as today's date (the data ends on that day). {question}"


def ask_genie(workspace_client, space_id: str, question: str, timeout_minutes: int = 5):
    """Ask Genie and return ``(sql | None, error | None)``. Never raises."""
    try:
        msg = workspace_client.genie.start_conversation_and_wait(
            space_id, question, timeout=dt.timedelta(minutes=timeout_minutes)
        )
    except Exception as exc:
        return None, f"genie call failed: {exc}"
    sql = extract_genie_sql(msg)
    if sql:
        return sql, None
    err = getattr(msg, "error", None)
    return None, f"genie returned no query attachment{f' ({err})' if err else ''}"


def evaluate_genie(q: dict, expected, *, ask_fn, run_query: Runner, run_id: str, max_rows: int = 500) -> dict:
    from lakehouseiq.guardrails import validate_sql

    start = time.perf_counter()
    sql, err = ask_fn(q["question"])
    latency = int((time.perf_counter() - start) * 1000)
    if err or not sql:
        return _row(run_id, "genie", q, error=err or "no SQL", verdict="ERROR", latency_ms=latency)
    ok, why = validate_sql(sql)
    if not ok:
        return _row(run_id, "genie", q, generated_sql=sql, verdict="BLOCKED", latency_ms=latency, error=why)
    df, run_err = run_query(sql, max_rows)
    if run_err or df is None:
        return _row(run_id, "genie", q, generated_sql=sql, verdict="ERROR", latency_ms=latency, error=run_err)
    row = _row(run_id, "genie", q, generated_sql=sql, verdict="EXECUTED", latency_ms=latency)
    if expected is not None:
        row["exec_match"], reason = results_match(expected, df, bool(q.get("order_matters")),
            ignore_zero_rows=bool(q.get("ignore_zero_rows")))
        if not row["exec_match"]:
            row["error"] = f"result mismatch: {reason}"
    return row


# --------------------------------------------------------------------------- #
# Whole run
# --------------------------------------------------------------------------- #


def run_eval(
    questions: list[dict],
    *,
    answer_fn,
    run_query: Runner,
    run_id: str,
    genie_ask_fn=None,
    done: set | None = None,
    limit: int | None = None,
    sleep_s: float = 2.0,
    on_rows: Callable[[list[dict]], None] | None = None,
    max_rows: int = 500,
) -> list[dict]:
    """Evaluate ``questions``. ``done`` holds ``(system, question_id)`` pairs to skip (resume).

    ``on_rows`` is called after each question with its new rows, so a quota cut-off loses nothing.
    """
    done = done or set()
    rows: list[dict] = []
    selected = questions[:limit] if limit else questions
    for i, q in enumerate(selected):
        systems = [s for s in (["assistant"] + (["genie"] if genie_ask_fn else [])) if (s, q["id"]) not in done]
        if not systems:
            continue
        expected, ref_err = run_query(q["reference_sql"], max_rows)
        new: list[dict] = []
        if ref_err or expected is None:
            logger.error("reference SQL failed for %s: %s", q["id"], ref_err)
        if "assistant" in systems:
            row = evaluate_assistant(q, expected, answer_fn=answer_fn, run_id=run_id)
            if expected is None:
                row["exec_match"], row["error"] = False, f"reference SQL failed: {ref_err}"
            new.append(row)
        if "genie" in systems:
            row = evaluate_genie(q, expected, ask_fn=genie_ask_fn, run_query=run_query, run_id=run_id, max_rows=max_rows)
            if expected is None:
                row["exec_match"], row["error"] = False, f"reference SQL failed: {ref_err}"
            new.append(row)
        rows.extend(new)
        if on_rows:
            on_rows(new)
        if i < len(selected) - 1 and sleep_s:
            time.sleep(sleep_s)  # Free Edition rate limits
    return rows


def split_of(question_id: str) -> str:
    for prefix, split in SPLIT_PREFIXES:
        if str(question_id).startswith(prefix):
            return split
    return "curated"


def _accuracy(rows: list[dict]) -> float:
    return sum(r["exec_match"] for r in rows) / len(rows)


def summarize(rows: list[dict]) -> dict:
    """MLflow metrics: accuracy per system (overall, curated, holdout), blocked/repair rates, assistant p50 latency.

    ``holdout_v2`` (ids ``h2-``) is the number to report: it is frozen and never used for tuning. ``holdout`` (v1)
    was used once to choose the generator model. ``curated`` has been used for debugging and is optimistic.
    """
    a = [r for r in rows if r["system"] == "assistant"]
    g = [r for r in rows if r["system"] == "genie"]
    metrics: dict[str, float] = {}
    for system, rs in (("assistant", a), ("genie", g)):
        if not rs:
            continue
        metrics[f"exec_accuracy_{system}"] = _accuracy(rs)
        for split in ("curated", "holdout", "holdout_v2"):
            part = [r for r in rs if split_of(r.get("question_id", "")) == split]
            if part:
                metrics[f"exec_accuracy_{system}_{split}"] = _accuracy(part)
    if g:
        # Infrastructure failures (Genie call did not complete) are not wrong answers; report them separately.
        metrics["error_rate_genie"] = sum(r["verdict"] == "ERROR" for r in g) / len(g)
    if a:
        metrics["blocked_rate"] = sum(r["verdict"] == "BLOCKED" for r in a) / len(a)
        metrics["repair_rate"] = sum(bool(r.get("repaired")) for r in a) / len(a)
        lat = [r["latency_ms"] for r in a if r["latency_ms"] is not None]
        if lat:
            metrics["p50_latency_ms"] = float(statistics.median(lat))
    return metrics


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _check_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.match(run_id):
        raise ValueError("run_id may contain only letters, digits, '_', '.' and '-' (max 64)")
    return run_id


def load_done(run_query: Runner, catalog: str, run_id: str) -> set:
    """``(system, question_id)`` pairs already stored for ``run_id`` (resume after a quota cut-off)."""
    df, err = run_query(
        f"SELECT DISTINCT system, question_id FROM {catalog}.app.eval_results WHERE run_id = '{_check_run_id(run_id)}'",
        None,
    )
    if err or df is None:
        logger.warning("could not read existing eval rows (%s); starting fresh", err)
        return set()
    return set(zip(df["system"], df["question_id"], strict=True))


def insert_rows(execute: Callable[[str, dict], None], catalog: str, rows: list[dict]) -> None:
    """Insert result rows with named parameters. ``execute(sql, params)`` runs one statement."""
    sql = (
        f"INSERT INTO {catalog}.app.eval_results ({', '.join(TABLE_COLUMNS)}) VALUES "
        "(CAST(:run_id AS STRING), CAST(:ts AS TIMESTAMP), CAST(:system AS STRING), CAST(:question_id AS STRING), "
        "CAST(:question AS STRING), CAST(:generated_sql AS STRING), CAST(:exec_match AS BOOLEAN), "
        "CAST(:judge_relevance AS INT), CAST(:verdict AS STRING), CAST(:latency_ms AS BIGINT), CAST(:error AS STRING))"
    )
    for r in rows:
        params = {c: r.get(c) for c in TABLE_COLUMNS}
        params["ts"] = r["ts"].strftime("%Y-%m-%d %H:%M:%S")
        execute(sql, params)


def log_to_mlflow(rows: list[dict], metrics: dict, run_id: str, experiment_id: str | None = None) -> str | None:
    """Log metrics and the results table to MLflow. Returns the MLflow run id, or None if unavailable."""
    try:
        import mlflow
        import pandas as pd

        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)
        with mlflow.start_run(run_name=run_id) as run:
            mlflow.log_metrics(metrics)
            table = pd.DataFrame(rows).drop(columns=["ts"], errors="ignore")
            mlflow.log_table(data=table, artifact_file="eval_results.json")
            return run.info.run_id
    except Exception as exc:
        logger.warning("MLflow logging skipped: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LakehouseIQ eval harness.")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--run-id", default=None, help="reuse a run id to resume; default is a fresh timestamp")
    parser.add_argument("--genie-space-id", default=None, help="also evaluate this Genie space (or set GENIE_SPACE_ID)")
    parser.add_argument("--sleep", type=float, default=2.0, help="seconds between questions")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import os
    from functools import partial

    from lakehouseiq import semantic, warehouse
    from lakehouseiq.config import Settings
    from lakehouseiq.pipeline import answer_question

    settings = Settings.from_env()
    warehouse.configure(settings)
    registry = semantic.load_registry(Path(__file__).resolve().parent.parent / "app" / "semantic")
    schema_ctx = warehouse.get_live_schema(settings)
    run_id = _check_run_id(args.run_id or dt.datetime.now(dt.timezone.utc).strftime("run-%Y%m%d-%H%M%S"))

    genie_space = args.genie_space_id or os.environ.get("GENIE_SPACE_ID") or None
    ask = None
    if genie_space:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        ask = lambda question: ask_genie(w, genie_space, genie_question(question, schema_ctx.as_of_date))  # noqa: E731

    questions = load_questions(catalog=settings.catalog, as_of_date=schema_ctx.as_of_date)
    for extra in (HOLDOUT_PATH, HOLDOUT_V2_PATH):
        if extra.exists():
            questions += load_questions(extra, catalog=settings.catalog, as_of_date=schema_ctx.as_of_date)
    done = load_done(warehouse.run_query, settings.catalog, run_id)
    rows = run_eval(
        questions,
        answer_fn=partial(answer_question, settings=settings, schema_ctx=schema_ctx, registry=registry, with_summary=False),
        run_query=warehouse.run_query,
        run_id=run_id,
        genie_ask_fn=ask,
        done=done,
        limit=args.limit,
        sleep_s=args.sleep,
        on_rows=lambda new: insert_rows(warehouse.execute_params, settings.catalog, new),
        max_rows=settings.max_rows,
    )
    metrics = summarize(rows)
    print(f"run {run_id}: {len(rows)} new rows")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
    if not args.no_mlflow:
        log_to_mlflow(rows, metrics, run_id, settings.mlflow_experiment_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
