import mlflow
import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "params.yaml") as f:
    cfg = yaml.safe_load(f)

EXPERIMENT = cfg["mlflow"]["experiment"]
MODEL_PATH = PROJECT_ROOT / "models/xgboost_model.pkl"

mlflow.set_experiment(EXPERIMENT)

print("Tracking URI:", mlflow.get_tracking_uri())

with mlflow.start_run(run_name="xgboost-model"):
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_metric("auc", 0.94)  # example
    mlflow.log_artifact(MODEL_PATH, artifact_path="models")

print("✅ Model logged to MLflow")
