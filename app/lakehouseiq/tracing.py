"""Safe MLflow tracing. Every function here is a no-op unless MLflow is importable and
an experiment is configured, and no tracing failure can ever break answering."""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_state: dict = {"configured": False, "mlflow": None}


def configure(experiment_id: str | None = None) -> bool:
    """Point MLflow at the workspace once. Returns True when tracing is active."""
    if _state["configured"]:
        return _state["mlflow"] is not None
    _state["configured"] = True
    experiment_id = experiment_id or os.environ.get("MLFLOW_EXPERIMENT_ID")
    if not experiment_id:
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(experiment_id=experiment_id)
        _state["mlflow"] = mlflow
        return True
    except Exception as exc:
        logger.warning("tracing disabled: %s", exc)
        return False


def reset() -> None:
    _state.update(configured=False, mlflow=None)


class SpanHandle:
    """What the body of ``with span(...) as s`` sees. Safe whether or not tracing is on."""

    def __init__(self, real=None):
        self._real = real

    @property
    def trace_id(self) -> str | None:
        try:
            return getattr(self._real, "trace_id", None) if self._real is not None else None
        except Exception:
            return None

    def set(self, **attrs) -> None:
        if self._real is None:
            return
        try:
            self._real.set_attributes({k: _plain(v) for k, v in attrs.items()})
        except Exception as exc:
            logger.debug("set_attributes failed: %s", exc)


def _plain(value):
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


@contextmanager
def span(name: str, **attrs):
    """Open a child span. Exceptions from the body propagate; tracing's own never do."""
    cm = real = None
    mlflow = _state["mlflow"]
    if mlflow is not None:
        try:
            cm = mlflow.start_span(name=name, attributes={k: _plain(v) for k, v in attrs.items()})
            real = cm.__enter__()
        except Exception as exc:
            logger.debug("start_span failed: %s", exc)
            cm = real = None
    handle = SpanHandle(real)
    try:
        yield handle
    except BaseException as exc:
        if cm is not None:
            try:
                cm.__exit__(type(exc), exc, exc.__traceback__)
            except BaseException:  # the context manager re-raises exc; we re-raise it below
                pass
        raise
    else:
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception as exc:
                logger.debug("span exit failed: %s", exc)


def traced(name: str | None = None):
    """Decorator form of :func:`span`."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with span(name or fn.__name__):
                return fn(*args, **kwargs)

        return wrapper

    return deco
