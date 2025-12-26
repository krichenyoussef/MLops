import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import subprocess

def gpu_available():
    try:
        subprocess.check_output("nvidia-smi", stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

import mlflow
import mlflow.xgboost

mlflow.set_experiment("ieee-cis-fraud")


df = pd.read_parquet("../../data/data_features/train_features.parquet")

y = df["isFraud"]
X = df.drop(columns=["isFraud", "TransactionID"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

# 🔥 Enable GPU if available
if gpu_available():
    print("🚀 Training on GPU")
    params.update({
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "max_bin": 255
    })
else:
    print("💻 Training on CPU")
    params["device"] = "cpu"


model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=2000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=50)
    ]
)

preds = model.predict(X_val)
print("AUC:", roc_auc_score(y_val, preds))
