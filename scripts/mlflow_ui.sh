#!/usr/bin/env bash
# Browse logged runs at http://127.0.0.1:5000 (local only).
cd "$(dirname "$0")/.." || exit 1
export MLFLOW_DISABLE_AGENT_HINT=1
exec .venv/bin/mlflow ui \
  --backend-store-uri "sqlite:///$PWD/mlruns/mlflow.db" \
  --default-artifact-root "file://$PWD/mlartifacts" \
  --host 127.0.0.1 --port 5000
