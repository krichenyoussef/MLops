import pandas as pd
import yaml
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score
import joblib

from src.models.utils import make_stratified_folds


# --------------------------------------------------
# Resolve project root & load config
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "params.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg["data"]
TRAIN_CFG = cfg["train"]
LGB_CFG = cfg["lightgbm"]


# --------------------------------------------------
# Paths
# --------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------
X = pd.read_parquet(PROJECT_ROOT / "data/features/X_train.parquet")
y = pd.read_parquet(
    PROJECT_ROOT / "data/features/y_train.parquet"
)[DATA_CFG["target_col"]]


# --------------------------------------------------
# Cross-validation folds
# --------------------------------------------------
folds = make_stratified_folds(
    y,
    n_splits=TRAIN_CFG["n_splits"],
    seed=TRAIN_CFG["seed"],
)

oof_preds = pd.Series(0.0, index=y.index)
fold_aucs = {}


# --------------------------------------------------
# Train LightGBM with params.yaml config
# --------------------------------------------------
for fold, (tr_idx, val_idx) in enumerate(folds):
    print(f"\n🔹 Training fold {fold + 1}/{TRAIN_CFG['n_splits']}")

    model = lgb.LGBMClassifier(**LGB_CFG)

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model.fit(X_tr, y_tr)

    val_preds = model.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_val, val_preds)

    print(f"   ✅ Fold {fold} AUC: {fold_auc:.5f}")

    oof_preds.iloc[val_idx] = val_preds
    fold_aucs[f"fold_{fold}_auc"] = fold_auc

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    model_path = MODELS_DIR / f"lightgbm_fold_{fold}.pkl"
    joblib.dump(model, model_path)


# --------------------------------------------------
# Overall OOF evaluation
# --------------------------------------------------
oof_auc = roc_auc_score(y, oof_preds)
print(f"\n🎯 LightGBM OOF AUC: {oof_auc:.5f}")
