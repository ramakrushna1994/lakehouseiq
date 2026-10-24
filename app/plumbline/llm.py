"""Foundation Model API client with retry and backoff.

``openai`` and ``databricks`` are loaded lazily inside functions so the rest of the
package (and its tests) import without either installed, and so tests can replace
``get_client``.
"""

from __future__ import annotations

import logging
import random
import threading
import time

logger = logging.getLogger(__name__)

RETRY_DELAYS_S = (2.0, 4.0, 8.0)
_RETRYABLE_STATUS = {408, 409, 425, 429}
_client = None
_client_lock = threading.Lock()
_sleep = time.sleep  # replaced in tests


class LLMError(RuntimeError):
    """A model call failed for good. The message names the endpoint."""

    def __init__(self, endpoint: str, message: str):
        super().__init__(f"[{endpoint}] {message}")
        self.endpoint = endpoint


def get_client():
    """Return a cached OpenAI-compatible client for Databricks Model Serving."""
    global _client
    with _client_lock:
        if _client is not None:
            return _client
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                _client = w.serving_endpoints.get_open_ai_client()
        except Exception as exc:  # older/newer SDKs, or openai missing from the extra
            logger.info("get_open_ai_client unavailable (%s); building client from the auth header", exc)
            from openai import OpenAI

            headers = w.config.authenticate()
            token = headers.get("Authorization", "").removeprefix("Bearer ").strip()
            _client = OpenAI(api_key=token, base_url=f"{w.config.host.rstrip('/')}/serving-endpoints")
        return _client


def reset_client() -> None:
    global _client
    with _client_lock:
        _client = None


def _extract_text(content) -> str:
    """Flatten message content. Reasoning models return a list of typed parts; keep only text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
        if ptype not in (None, "text", "output_text"):
            continue  # reasoning / tool parts
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS or status >= 500
    name = type(exc).__name__
    return name in {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"} or isinstance(
        exc, (ConnectionError, TimeoutError)
    )


def complete(prompt: str, endpoint: str, *, max_tokens: int = 1500, temperature: float = 0.0) -> str:
    """Send one user message to ``endpoint`` and return the text reply.

    Retries 429/5xx and connection errors after 2s, 4s and 8s (with jitter).
    Raises :class:`LLMError` naming the endpoint on final failure or empty output.
    """
    attempts = len(RETRY_DELAYS_S) + 1
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            client = get_client()
            resp = client.chat.completions.create(
                model=endpoint,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = _extract_text(resp.choices[0].message.content).strip()
            if not text:
                raise LLMError(endpoint, "empty response (a reasoning model may have exhausted max_tokens)")
            return text
        except LLMError:
            raise
        except Exception as exc:
            last = exc
            if attempt < attempts - 1 and _is_retryable(exc):
                delay = RETRY_DELAYS_S[attempt] + random.uniform(0, 0.5)
                logger.warning("%s failed (%s); retry %d in %.1fs", endpoint, exc, attempt + 1, delay)
                _sleep(delay)
                continue
            break
    raise LLMError(endpoint, f"{type(last).__name__}: {last}") from last
