from __future__ import annotations

import os
import json
import subprocess
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from src.config.settings import load_params, load_settings
from src.features.magic_features import (
    build_magic_features_train_val,
    finalize_for_xgb,
)


# --------------------------------------------------
# MLflow configuration (DAGsHub-compatible)
# --------------------------------------------------
def configure_mlflow():
    s = load_settings()
    if s.mlflow_tracking_uri:
        mlflow.set_tracking_uri(s.mlflow_tracking_uri)


# --------------------------------------------------
# Main training
# --------------------------------------------------
def main():
    p = load_params()
    configure_mlflow()

    # ----------------------------
    # Config
    # ----------------------------
    processed_dir = p["data"]["processed_dir"]
    target = p["data"]["target_col"]
    exp_name = p["mlflow"]["experiment"]

    seed = int(p["train"]["seed"])
    n_splits = int(p["train"]["n_splits"])

    xgb_cfg = p["xgboost"]

    # ----------------------------
    # Load data
    # ----------------------------
    train_df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_dir, "val.parquet"))

    if target not in train_df.columns:
        raise ValueError(f"Target '{target}' not found in train.parquet")

    y_train = train_df[target].astype(int)
    y_val = val_df[target].astype(int)

    train_df = train_df.drop(columns=[target])
    val_df = val_df.drop(columns=[target])

    # ----------------------------
    # Feature engineering
    # ----------------------------
    train_feat, val_feat = build_magic_features_train_val(train_df, val_df)

    train_group = train_feat.get("DT_M")
    if train_group is None:
        raise RuntimeError("DT_M missing — check TransactionDT preprocessing.")

    X_train = finalize_for_xgb(train_feat, drop_cols=["TransactionID"])
    X_val = finalize_for_xgb(val_feat, drop_cols=["TransactionID"])
    X_val = X_val.reindex(columns=X_train.columns, fill_value=-1)

    # ----------------------------
    # MLflow experiment
    # ----------------------------
    mlflow.set_experiment(exp_name)

    xgb_params = {
        "n_estimators": xgb_cfg["n_estimators"],
        "max_depth": xgb_cfg["max_depth"],
        "learning_rate": xgb_cfg["learning_rate"],
        "subsample": xgb_cfg["subsample"],
        "colsample_bytree": xgb_cfg["colsample_bytree"],
        "reg_lambda": xgb_cfg["reg_lambda"],
        "tree_method": xgb_cfg["tree_method"],
        "eval_metric": xgb_cfg["eval_metric"],
        "missing": xgb_cfg["missing"],
        "random_state": seed,
        "n_jobs": -1,
    }

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(X_train), dtype="float32")

    # ----------------------------
    # Training + Tracking
    # ----------------------------
    with mlflow.start_run(run_name="xgb_magic_processed"):
        # Log params
        mlflow.log_params(xgb_params)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("feature_count", int(X_train.shape[1]))
        mlflow.log_param("train_rows", int(len(X_train)))
        mlflow.log_param("val_rows", int(len(X_val)))
        mlflow.log_param("processed_dir", processed_dir)

        # Log git commit (reproducibility)
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            mlflow.log_param("git_commit", commit)
        except Exception:
            pass

        # ----------------------------
        # CV on TRAIN
        # ----------------------------
        for fold, (idx_tr, idx_va) in enumerate(
            gkf.split(X_train, y_train, groups=train_group)
        ):
            model = XGBClassifier(**xgb_params)

            model.fit(
                X_train.iloc[idx_tr],
                y_train.iloc[idx_tr],
                eval_set=[(X_train.iloc[idx_va], y_train.iloc[idx_va])],
                verbose=200,
                callbacks=[EarlyStopping(rounds=200, save_best=True)],
            )

            oof[idx_va] = model.predict_proba(X_train.iloc[idx_va])[:, 1]
            fold_auc = roc_auc_score(y_train.iloc[idx_va], oof[idx_va])
            mlflow.log_metric(f"fold_{fold}_auc", float(fold_auc))
            print(f"Fold {fold} AUC: {fold_auc:.6f}")

        oof_auc = roc_auc_score(y_train, oof)
        mlflow.log_metric("oof_auc", float(oof_auc))
        print(f"✅ OOF AUC: {oof_auc:.6f}")

        # ----------------------------
        # Train final model (VAL)
        # ----------------------------
        final_model = XGBClassifier(**xgb_params)
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=200,
            callbacks=[EarlyStopping(rounds=200, save_best=True)],
        )

        val_proba = final_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)

        mlflow.log_metric("val_auc", float(val_auc))
        mlflow.log_metric("primary_auc", float(val_auc))

        print(f"✅ VAL AUC: {val_auc:.6f}")

        # ----------------------------
        # Register model (DAGsHub)
        # ----------------------------
        mlflow.xgboost.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="ieee-cis-xgb-magic",
        )

        # ----------------------------
        # Save metrics for DVC
        # ----------------------------
        os.makedirs("reports", exist_ok=True)
        with open("reports/metrics.json", "w") as f:
            json.dump(
                {
                    "oof_auc": float(oof_auc),
                    "val_auc": float(val_auc),
                },
                f,
                indent=2,
            )

    print("✅ Training + MLflow logging completed successfully.")


if __name__ == "__main__":
    main()
