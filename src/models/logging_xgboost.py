import mlflow
import joblib
import yaml
from pathlib import Path


# --------------------------------------------------
# Resolve project root
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
MODEL_PATH = PROJECT_ROOT / "models/xgboost_model.pkl"


# --------------------------------------------------
# Load config
# --------------------------------------------------
with open(PARAMS_PATH) as f:
    cfg = yaml.safe_load(f)

EXPERIMENT = cfg["mlflow"]["experiment"]


# --------------------------------------------------
# SAFETY CHECK
# --------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


# --------------------------------------------------
# MLflow logging (THIS IS THE FIX)
# --------------------------------------------------
mlflow.set_experiment(EXPERIMENT)

print("MLflow tracking URI:", mlflow.get_tracking_uri())

with mlflow.start_run(run_name="xgboost-imported"):
    # Log at least ONE thing
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_artifact(MODEL_PATH, artifact_path="models")

print("✅ XGBoost model successfully logged to MLflow")
