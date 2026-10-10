"""MLflow tracking shared by every pipeline step.

Runs are logged straight to a local SQLite store (no server needed to log);
`scripts/mlflow_ui.sh` starts the browser UI on top of the same store.
Nothing here leaves the machine.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}"
ARTIFACT_ROOT = (ROOT / "mlartifacts").as_uri()

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def start(experiment: str, run_name: str) -> mlflow.ActiveRun:
    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name)
