import mlflow
import yaml
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "params.yaml") as f:
    cfg = yaml.safe_load(f)

EXPERIMENT = cfg["mlflow"]["experiment"]
MODELS_DIR = PROJECT_ROOT / "models"

OOF_AUC = 0.93865  # <-- replace with CatBoost OOF AUC


mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="catboost_cv"):
    mlflow.log_param("model_type", "catboost")
    mlflow.log_metric("oof_auc", OOF_AUC)

    for pkl in MODELS_DIR.glob("catboost_fold_*.pkl"):
        mlflow.log_artifact(pkl, artifact_path="models")

print("✅ CatBoost logged to MLflow")
