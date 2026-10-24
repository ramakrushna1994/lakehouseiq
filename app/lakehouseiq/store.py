"""Conversation history and corrections in Delta, written with named parameters.

Every call is best-effort: a failure is logged and swallowed, because audit and
learning must never stop a user from getting an answer. The ranking function is pure.
"""

from __future__ import annotations

import logging
import re
import uuid

from . import warehouse as warehouse_module
from .config import Settings

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "a an the of in on for to by and or is are was were be been what which who whom how many much "
    "show me give list get do does did with from at as it its this that these those i we you our "
    "my your per over under than then vs versus top".split()
)
_WORD_RE = re.compile(r"[a-z0-9_]+")
MIN_SCORE = 0.2


def tokenize(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS and len(w) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def rank_corrections(question: str, rows: list[dict], limit: int = 3, min_score: float = MIN_SCORE) -> list[dict]:
    """Top ``limit`` corrections by token overlap with ``question``, score > ``min_score``.

    There is deliberately no fallback to "most recent": an unrelated correction in the
    prompt is worse than none.
    """
    q = tokenize(question)
    scored = []
    for i, row in enumerate(rows):
        s = jaccard(q, tokenize(row.get("question", "")))
        if s > min_score:
            scored.append((s, -i, row))  # ties keep the more recent row (rows arrive newest first)
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [r for _, _, r in scored[:limit]]


def save_turn(
    *,
    settings: Settings,
    session_id: str,
    turn_id: int,
    user_email: str,
    question: str,
    generated_sql: str | None,
    source_view: str | None,
    verdict: str | None,
    row_count: int | None,
    error: str | None,
    latency_ms: int | None,
    wh=warehouse_module,
) -> bool:
    sql = f"""
        INSERT INTO {settings.fq('app', 'conversation_history')}
          (session_id, turn_id, ts, user_email, question, generated_sql, source_view, verdict,
           row_count, error, latency_ms, generator_endpoint, judge_endpoint)
        VALUES (CAST(:session_id AS STRING), CAST(:turn_id AS INT), current_timestamp(),
                CAST(:user_email AS STRING), CAST(:question AS STRING), CAST(:generated_sql AS STRING),
                CAST(:source_view AS STRING), CAST(:verdict AS STRING), CAST(:row_count AS INT),
                CAST(:error AS STRING), CAST(:latency_ms AS BIGINT), CAST(:generator_endpoint AS STRING),
                CAST(:judge_endpoint AS STRING))
    """
    params = {
        "session_id": session_id,
        "turn_id": turn_id,
        "user_email": user_email,
        "question": question,
        "generated_sql": generated_sql,
        "source_view": source_view,
        "verdict": verdict,
        "row_count": row_count,
        "error": error,
        "latency_ms": latency_ms,
        "generator_endpoint": settings.generator_endpoint,
        "judge_endpoint": settings.judge_endpoint,
    }
    try:
        wh.execute_params(sql, params)
        return True
    except Exception as exc:
        logger.warning("save_turn failed: %s", exc)
        return False


def save_correction(
    *,
    settings: Settings,
    user_email: str,
    question: str,
    bad_sql: str,
    corrected_sql: str,
    reason: str = "",
    bad_values: list[str] | None = None,
    wh=warehouse_module,
) -> bool:
    sql = f"""
        INSERT INTO {settings.fq('app', 'query_corrections')}
          (correction_id, ts, user_email, question, bad_sql, corrected_sql, reason, bad_values)
        VALUES (CAST(:correction_id AS STRING), current_timestamp(), CAST(:user_email AS STRING),
                CAST(:question AS STRING), CAST(:bad_sql AS STRING), CAST(:corrected_sql AS STRING),
                CAST(:reason AS STRING), CAST(:bad_values AS STRING))
    """
    params = {
        "correction_id": str(uuid.uuid4()),
        "user_email": user_email,
        "question": question,
        "bad_sql": bad_sql,
        "corrected_sql": corrected_sql,
        "reason": reason,
        "bad_values": ", ".join(bad_values or []),
    }
    try:
        wh.execute_params(sql, params)
        return True
    except Exception as exc:
        logger.warning("save_correction failed: %s", exc)
        return False


def get_relevant_corrections(
    question: str, limit: int = 3, *, settings: Settings | None = None, wh=warehouse_module
) -> list[dict]:
    """Fetch the latest 200 corrections and keep those similar to ``question``."""
    settings = settings or Settings.from_env()
    try:
        df, err = wh.run_query(
            f"SELECT question, bad_sql, corrected_sql, reason, bad_values "
            f"FROM {settings.fq('app', 'query_corrections')} ORDER BY ts DESC LIMIT 200",
            200,
        )
        if err or df is None:
            logger.warning("correction lookup failed: %s", err)
            return []
        return rank_corrections(question, df.to_dict("records"), limit)
    except Exception as exc:
        logger.warning("correction lookup failed: %s", exc)
        return []
