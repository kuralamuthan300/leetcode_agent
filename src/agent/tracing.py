"""MLflow tracing bootstrap.

`mlflow.langchain.autolog()` instruments LangChain and LangGraph, so graph
nodes and every ChatOpenAI call show up as spans without decorating them.
Each run's trace is renamed/tagged by the reporter (`tag_run`) so the trace
list reads `run_id · problem` instead of a wall of `LangGraph`.

Disable with `MLFLOW_TRACING_DISABLED=1` (tests set this so `cli.main()` stays
hermetic) or the `--no-tracing` CLI flag.
"""

from __future__ import annotations

import logging
import os

from .config import get_config

logger = logging.getLogger(__name__)

_experiment_id: str | None = None


def tracing_enabled() -> bool:
    return os.getenv("MLFLOW_TRACING_DISABLED", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


def init_tracing() -> bool:
    """Configure the tracking server and enable LangGraph autologging.

    Returns True when tracing is active. Never raises: observability must not
    take down a solve when the tracking server is unreachable.
    """
    global _experiment_id
    if not tracing_enabled():
        return False
    try:
        # Keep CLI runs quiet: MLflow prints an agent-skill hint on import.
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

        import mlflow
        import mlflow.langchain as mlflow_langchain

        config = get_config()
        mlflow.set_tracking_uri(config.gateway.tracking_uri)
        # MLFLOW_EXPERIMENT_ID (standard MLflow variable) wins when set.
        experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
        if not experiment_id:
            experiment = mlflow.set_experiment(config.gateway.experiment)
            experiment_id = getattr(experiment, "experiment_id", None)
        _experiment_id = experiment_id
        mlflow_langchain.autolog(log_traces=True)
        logger.info(
            "MLflow tracing enabled (experiment=%s, tracking_uri=%s)",
            config.gateway.experiment,
            config.gateway.tracking_uri,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort
        logger.warning("MLflow tracing unavailable: %s", exc)
        return False


def tag_run(state: dict) -> None:
    """Name and tag the active run trace (called by the reporter node).

    MLflow displays the reserved `mlflow.traceName` tag as the trace name, so
    runs stop looking identical in the trace list. Payload previews are kept
    short instead of dumping the whole graph state.
    """
    if not tracing_enabled():
        return
    try:
        import mlflow

        if mlflow.get_current_active_span() is None:
            return

        run_id = str(state.get("run_id") or "")
        requirement = " ".join(str(state.get("requirement") or "").split())
        name = f"{run_id} · {requirement[:60]}".strip(" ·")

        evaluation = state.get("eval") or {}
        passed = evaluation.get("passed_count")
        total = evaluation.get("total")
        eval_tag = f"{passed}/{total}" if total else ""
        culprit = str((state.get("failure_report") or {}).get("culprit") or "")
        status = str(state.get("status") or "ok")

        tags = {
            "run_id": run_id,
            "status": status,
            "gather_mode": str(state.get("gather_mode") or ""),
            "eval": eval_tag,
            "mlflow.traceName": name,
        }
        response = f"status={status}" + (f" eval={eval_tag}" if eval_tag else "")
        if culprit:
            response += f" culprit={culprit}"
        mlflow.update_current_trace(
            tags={key: value for key, value in tags.items() if value},
            request_preview=requirement[:300] or run_id,
            response_preview=response[:300],
        )
    except Exception:  # noqa: BLE001 - tracing is best-effort
        logger.debug("failed to tag trace", exc_info=True)


def last_trace_id() -> str | None:
    """Trace id of the run that just finished, or None when tracing is off."""
    if not tracing_enabled():
        return None
    try:
        import mlflow

        return mlflow.get_last_active_trace_id()
    except Exception:  # noqa: BLE001 - tracing is best-effort
        return None


def trace_url(trace_id: str | None) -> str | None:
    """Deep link to a trace in the MLflow UI, or None when unavailable."""
    if not trace_id or not tracing_enabled():
        return None
    try:
        import mlflow

        config = get_config()
        experiment_id = _experiment_id
        if not experiment_id:
            experiment = mlflow.get_experiment_by_name(config.gateway.experiment)
            experiment_id = getattr(experiment, "experiment_id", None)
        if not experiment_id:
            return None
        base = config.gateway.tracking_uri.rstrip("/")
        return f"{base}/#/experiments/{experiment_id}/traces/{trace_id}"
    except Exception:  # noqa: BLE001 - tracing is best-effort
        return None
