import pandas as pd
from pathlib import Path
    
RAW_DIR = Path("../../data/raw")
OUT_DIR = Path("../../data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    trans = pd.read_csv(RAW_DIR / "train_transaction.csv")
    ident = pd.read_csv(RAW_DIR / "train_identity.csv")

    df = trans.merge(ident, on="TransactionID", how="left")

    # Basic cleaning
    drop_cols = [c for c in df.columns if df[c].isna().mean() > 0.9]
    df.drop(columns=drop_cols, inplace=True)

    df.to_parquet(OUT_DIR / "train.parquet", index=False)
    print("✅ Processed data saved")

if __name__ == "__main__":
    main()
