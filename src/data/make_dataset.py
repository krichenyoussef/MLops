import argparse
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Project root resolution (robust)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


# ---------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------
def load_params():
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"params.yaml not found at {PARAMS_PATH}")

    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# Validation (NO data modification)
# ---------------------------------------------------------------------
def validate_dataframe(df: pd.DataFrame, split: str, cfg: dict):
    data_cfg = cfg["data"]
    interim_cfg = cfg["interim"]

    id_col = data_cfg["id_col"]
    time_col = data_cfg["time_col"]
    target_col = data_cfg["target_col"]
    missing_thr = interim_cfg["missing_threshold_warning"]

    print("Running interim validation checks...")

    if split == "train" and interim_cfg["fail_on"]["target_missing_in_train"]:
        if target_col not in df.columns:
            raise ValueError(
                f"❌ Target column '{target_col}' missing in TRAIN data"
            )

    if split == "test" and interim_cfg["fail_on"]["target_in_test"]:
        if target_col in df.columns:
            raise ValueError(
                f"❌ Target leakage detected: '{target_col}' found in TEST data"
            )

    for col in [id_col, time_col]:
        if col not in df.columns:
            raise ValueError(f"❌ Required column '{col}' missing")

    missing_ratio = df.isnull().mean()
    high_missing = missing_ratio[missing_ratio > missing_thr]

    if not high_missing.empty:
        print(f"\n⚠️ Columns with missing ratio > {missing_thr}:")
        print(high_missing.sort_values(ascending=False))

    dup_count = df[id_col].duplicated().sum()
    if dup_count > 0:
        print(f"⚠️ Found {dup_count} duplicated values in '{id_col}'")

    print("✔ Interim validation completed\n")


# ---------------------------------------------------------------------
# Prepare one split
# ---------------------------------------------------------------------
def prepare_split(split: str, cfg: dict):
    print(f"\n🔹 Preparing {split} data")

    raw_dir = PROJECT_ROOT / cfg["data"]["raw_dir"]
    interim_dir = PROJECT_ROOT / "data/interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    trans_path = raw_dir / split / f"{split}_transaction.csv"
    ident_path = raw_dir / split / f"{split}_identity.csv"

    if not trans_path.exists():
        raise FileNotFoundError(f"Missing file: {trans_path}")

    trans = pd.read_csv(trans_path)

    if ident_path.exists():
        ident = pd.read_csv(ident_path)
        df = trans.merge(
            ident,
            on=cfg["data"]["id_col"],
            how="left",
            validate="one_to_one",
        )
    else:
        if cfg["interim"]["allow_missing_identity"]:
            print("⚠️ identity.csv missing — proceeding without it")
            df = trans.copy()
        else:
            raise FileNotFoundError(f"Missing file: {ident_path}")

    time_col = cfg["data"]["time_col"]
    df = df.sort_values(time_col).reset_index(drop=True)

    validate_dataframe(df, split, cfg)

    out_path = interim_dir / f"{split}_merged.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create clean interim parquet data"
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "test"],
        help="Dataset split to process",
    )
    args = parser.parse_args()

    cfg = load_params()
    prepare_split(args.split, cfg)


if __name__ == "__main__":
    main()
