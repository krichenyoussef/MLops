import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("../../data/raw")
OUT_DIR = Path("../../data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utility functions
# -----------------------------

def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith("int"):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
            else:
                df[col] = df[col].astype(np.float32)
    return df


def frequency_encoding(df, cols):
    for col in cols:
        freq = df[col].value_counts(dropna=False)
        df[f"{col}_freq"] = df[col].map(freq).astype(np.float32)
    return df


# -----------------------------
# Main preprocessing
# -----------------------------

def main():
    print("📥 Loading data...")
    trans = pd.read_csv(RAW_DIR / "train_transaction.csv")
    ident = pd.read_csv(RAW_DIR / "train_identity.csv")

    df = trans.merge(ident, on="TransactionID", how="left")

    # -----------------------------
    # Drop extremely sparse columns
    # -----------------------------
    missing_ratio = df.isna().mean()
    drop_cols = missing_ratio[missing_ratio > 0.90].index.tolist()
    df.drop(columns=drop_cols, inplace=True)

    # -----------------------------
    # Time features
    # -----------------------------
    df["TransactionDT_days"] = df["TransactionDT"] / (60 * 60 * 24)
    df.drop(columns=["TransactionDT"], inplace=True)

    # -----------------------------
    # Amount features
    # -----------------------------
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # -----------------------------
    # Identify column types
    # -----------------------------
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    id_cols = [c for c in df.columns if c.startswith("id_")]
    v_cols = [c for c in df.columns if c.startswith("V")]

    # -----------------------------
    # Frequency encoding (CRUCIAL)
    # -----------------------------
    freq_cols = [
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "P_emaildomain", "R_emaildomain"
    ]
    freq_cols = [c for c in freq_cols if c in df.columns]
    df = frequency_encoding(df, freq_cols)

    # -----------------------------
    # Categorical handling
    # -----------------------------
    for col in cat_cols:
        df[col] = df[col].fillna("missing")

    # -----------------------------
    # Numeric missing values
    # -----------------------------
    num_cols = df.select_dtypes(exclude="object").columns
    df[num_cols] = df[num_cols].fillna(-1)

    # -----------------------------
    # Memory optimization
    # -----------------------------
    df = reduce_mem_usage(df)

    # -----------------------------
    # Save
    # -----------------------------
    df.to_parquet(OUT_DIR / "train.parquet", index=False)
    print("✅ Preprocessing complete")

if __name__ == "__main__":
    main()
