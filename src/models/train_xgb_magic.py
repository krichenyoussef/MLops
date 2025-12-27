import pandas as pd
import yaml
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import joblib


# --------------------------------------------------
# Resolve project root
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


# --------------------------------------------------
# Load config
# --------------------------------------------------
with open(PARAMS_PATH) as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg["data"]
XGB_CFG = cfg["xgboost"]


# --------------------------------------------------
# Load data
# --------------------------------------------------
X = pd.read_parquet(PROJECT_ROOT / "data/features/X_train.parquet")
y = pd.read_parquet(PROJECT_ROOT / "data/features/y_train.parquet")[DATA_CFG["target_col"]]

train_idx = pd.read_pickle(PROJECT_ROOT / "data/splits/train_idx.pkl")
val_idx = pd.read_pickle(PROJECT_ROOT / "data/splits/val_idx.pkl")

X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]


# --------------------------------------------------
# Handle class imbalance
# --------------------------------------------------
scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()


# --------------------------------------------------
# Train model
# --------------------------------------------------
model = xgb.XGBClassifier(
    **XGB_CFG,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    verbose=100,
)

# --------------------------------------------------
# Evaluate
# --------------------------------------------------
val_preds = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_preds)

print(f"✅ XGBoost Validation AUC: {auc:.5f}")


# --------------------------------------------------
# Save model
# --------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODEL_DIR / "xgboost_model.pkl")
