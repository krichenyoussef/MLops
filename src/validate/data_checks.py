from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import yaml


@dataclass
class CheckResult:
    ok: bool
    msg: str


def load_params(path: str = "params.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fail(msg: str) -> CheckResult:
    return CheckResult(False, msg)


def pass_(msg: str) -> CheckResult:
    return CheckResult(True, msg)


def check_files_exist(processed_dir: Path) -> List[CheckResult]:
    res = []
    for name in ["train.parquet", "val.parquet", "test.parquet"]:
        p = processed_dir / name
        if not p.exists():
            res.append(fail(f"Missing processed file: {p}"))
        else:
            res.append(pass_(f"Found {p}"))
    return res


def check_required_columns(df: pd.DataFrame, required: List[str], split_name: str) -> List[CheckResult]:
    res = []
    missing = [c for c in required if c not in df.columns]
    if missing:
        res.append(fail(f"[{split_name}] Missing required columns: {missing}"))
    else:
        res.append(pass_(f"[{split_name}] All required columns present"))
    return res


def check_target_binary(df: pd.DataFrame, target: str, split_name: str) -> List[CheckResult]:
    if target not in df.columns:
        return [fail(f"[{split_name}] Target column '{target}' not found")]

    # Drop NaN targets (shouldn't happen, but avoid crashing)
    y = df[target].dropna()
    uniq = sorted(set(y.astype(int).unique().tolist()))
    if not set(uniq).issubset({0, 1}):
        return [fail(f"[{split_name}] Target '{target}' has non-binary values: {uniq}")]
    return [pass_(f"[{split_name}] Target '{target}' is binary: {uniq}")]


def check_missing_ratio(df: pd.DataFrame, max_missing_ratio: float, split_name: str) -> List[CheckResult]:
    miss = df.isna().mean().sort_values(ascending=False)
    bad = miss[miss > max_missing_ratio]
    if len(bad) > 0:
        # show only top few columns to keep logs clean
        top = bad.head(15).to_dict()
        return [fail(f"[{split_name}] Columns exceed max_missing_ratio_per_col={max_missing_ratio}: {top}")]
    return [pass_(f"[{split_name}] No columns exceed missing ratio threshold ({max_missing_ratio})")]


def check_duplicate_ids(df: pd.DataFrame, id_col: str, max_dup_ratio: float, split_name: str) -> List[CheckResult]:
    if id_col not in df.columns:
        return [fail(f"[{split_name}] ID column '{id_col}' not found")]

    n = len(df)
    if n == 0:
        return [fail(f"[{split_name}] Empty dataset")]

    dup_count = int(df[id_col].duplicated().sum())
    dup_ratio = dup_count / n
    if dup_ratio > max_dup_ratio:
        return [fail(f"[{split_name}] Duplicate {id_col} ratio too high: {dup_ratio:.6f} (count={dup_count}/{n})")]
    return [pass_(f"[{split_name}] Duplicate {id_col} ratio OK: {dup_ratio:.6f} (count={dup_count}/{n})")]


def check_fraud_rate(df: pd.DataFrame, target: str, min_rate: float, max_rate: float, split_name: str) -> List[CheckResult]:
    if target not in df.columns:
        return [fail(f"[{split_name}] Target column '{target}' not found")]

    y = df[target].dropna().astype(int)
    if len(y) == 0:
        return [fail(f"[{split_name}] Target '{target}' is empty after dropping NaNs")]

    rate = float(y.mean())
    if not (min_rate <= rate <= max_rate):
        return [fail(f"[{split_name}] Fraud rate out of bounds: {rate:.6f} (expected {min_rate}..{max_rate})")]
    return [pass_(f"[{split_name}] Fraud rate OK: {rate:.6f}")]


def check_time_order(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, time_col: str) -> List[CheckResult]:
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if time_col not in df.columns:
            return [fail(f"[{name}] Time column '{time_col}' not found")]

    # We expect time-based split => train times <= val times <= test times
    train_max = train_df[time_col].max()
    val_min = val_df[time_col].min()
    val_max = val_df[time_col].max()
    test_min = test_df[time_col].min()

    if train_max > val_min:
        return [fail(f"Time split invalid: train_max({train_max}) > val_min({val_min})")]
    if val_max > test_min:
        return [fail(f"Time split invalid: val_max({val_max}) > test_min({test_min})")]

    return [pass_(f"Time split OK: train_max={train_max}, val=[{val_min}..{val_max}], test_min={test_min}")]


def main() -> int:
    p = load_params()

    processed_dir = Path(p["data"]["processed_dir"])
    id_col = p["data"]["id_col"]
    time_col = p["data"]["time_col"]
    target_col = p["data"]["target_col"]

    vcfg = p.get("validate", {})
    required_cols = vcfg.get("required_cols", [id_col, time_col, "TransactionAmt", target_col])
    max_missing_ratio = float(vcfg.get("max_missing_ratio_per_col", 0.999))
    max_dup_ratio = float(vcfg.get("max_duplicate_id_ratio", 0.0001))
    min_fraud_rate = float(vcfg.get("min_fraud_rate", 0.0001))
    max_fraud_rate = float(vcfg.get("max_fraud_rate", 0.05))
    enforce_time_order = bool(vcfg.get("enforce_time_order", True))

    results: List[CheckResult] = []
    results += check_files_exist(processed_dir)

    if not all(r.ok for r in results):
        print("❌ File checks failed. Fix build_dataset outputs first.")
        for r in results:
            print(("✅" if r.ok else "❌"), r.msg)
        return 1

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        results += check_required_columns(df, required_cols, split_name)
        results += check_duplicate_ids(df, id_col, max_dup_ratio, split_name)
        results += check_missing_ratio(df, max_missing_ratio, split_name)
        results += check_target_binary(df, target_col, split_name)
        results += check_fraud_rate(df, target_col, min_fraud_rate, max_fraud_rate, split_name)

    if enforce_time_order:
        results += check_time_order(train_df, val_df, test_df, time_col)

    # Print summary
    print("\n==== DATA CHECKS SUMMARY ====")
    for r in results:
        print(("✅" if r.ok else "❌"), r.msg)

    ok = all(r.ok for r in results)
    if ok:
        print("\n✅ All data checks passed.")
        return 0

    print("\n❌ Data checks failed. Fix the issues above before training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
