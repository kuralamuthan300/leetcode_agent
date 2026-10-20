"""MLflow tracing bootstrap.

`mlflow.langchain.autolog()` instruments LangChain and LangGraph, so graph
nodes and every ChatOpenAI call show up as spans without decorating them.

Disable with `MLFLOW_TRACING_DISABLED=1` (tests set this so `cli.main()` stays
hermetic).
"""

from __future__ import annotations

import logging
import os

from .config import get_config

logger = logging.getLogger(__name__)


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
        if not os.getenv("MLFLOW_EXPERIMENT_ID"):
            mlflow.set_experiment(config.gateway.experiment)
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
