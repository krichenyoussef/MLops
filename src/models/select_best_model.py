import mlflow
import yaml
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "params.yaml") as f:
    cfg = yaml.safe_load(f)

EXPERIMENT_NAME = cfg["mlflow"]["experiment"]

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.oof_auc DESC"],
)

if len(runs) == 0:
    raise RuntimeError("No runs found in MLflow")

best_run = runs[0]

print("🏆 BEST MODEL SELECTED")
print("Run ID:", best_run.info.run_id)
print("Model type:", best_run.data.params.get("model_type"))
print("AUC:", best_run.data.metrics.get("oof_auc"))

# Tag best model
client.set_tag(best_run.info.run_id, "best_model", "true")
