import mlflow
import yaml
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "params.yaml") as f:
    cfg = yaml.safe_load(f)

EXPERIMENT = cfg["mlflow"]["experiment"]
MODELS_DIR = PROJECT_ROOT / "models"

# You should already know this value from training
OOF_AUC = 0.94178  # <-- replace with real LightGBM OOF AUC


mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="lightgbm_cv"):
    mlflow.log_param("model_type", "lightgbm")
    mlflow.log_metric("oof_auc", OOF_AUC)

    # log all fold models
    for pkl in MODELS_DIR.glob("lightgbm_fold_*.pkl"):
        mlflow.log_artifact(pkl, artifact_path="models")

print("✅ LightGBM logged to MLflow")
