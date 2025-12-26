from __future__ import annotations

import os
import pandas as pd
from src.config.settings import load_params


def time_split(df: pd.DataFrame, time_col: str, val_ratio: float, test_ratio: float):
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_df = df.iloc[n - n_test :]
    val_df = df.iloc[n - n_test - n_val : n - n_test]
    train_df = df.iloc[: n - n_test - n_val]
    return train_df, val_df, test_df


def main():
    p = load_params()
    raw_dir = p["data"]["raw_dir"]
    out_dir = p["data"]["processed_dir"]

    id_col = p["data"]["id_col"]
    time_col = p["data"]["time_col"]
    target = p["data"]["target_col"]

    os.makedirs(out_dir, exist_ok=True)

    tx_path = os.path.join(raw_dir, "train_transaction.csv")
    id_path = os.path.join(raw_dir, "train_identity.csv")

    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Missing: {tx_path}")

    tx = pd.read_csv(tx_path)

    if os.path.exists(id_path):
        ident = pd.read_csv(id_path)
        df = tx.merge(ident, on=id_col, how="left")
    else:
        df = tx

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in train_transaction.csv")

    train_df, val_df, test_df = time_split(
        df, time_col=time_col,
        val_ratio=float(p["data"]["val_ratio"]),
        test_ratio=float(p["data"]["test_ratio"])
    )

    train_df.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)

    print("✅ Created processed splits in:", out_dir)
    print(f"Sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")


if __name__ == "__main__":
    main()
