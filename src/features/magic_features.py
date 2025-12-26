from __future__ import annotations

import datetime
from typing import Iterable, Sequence, Dict, Tuple

import numpy as np
import pandas as pd


def add_dt_m(df: pd.DataFrame, time_col: str = "TransactionDT") -> pd.DataFrame:
    start = datetime.datetime.strptime("2017-11-30", "%Y-%m-%d")
    dt = df[time_col].apply(lambda x: (start + datetime.timedelta(seconds=float(x))))
    df["DT_M"] = (dt.dt.year - 2017) * 12 + dt.dt.month
    return df


def normalize_d_columns(df: pd.DataFrame, time_col: str = "TransactionDT", d_cols: Sequence[str] | None = None) -> pd.DataFrame:
    if d_cols is None:
        d_cols = [f"D{i}" for i in range(1, 16)]
    if time_col not in df.columns:
        return df
    t_days = df[time_col] / np.float32(24 * 60 * 60)
    for c in d_cols:
        if c in df.columns and c != time_col:
            df[c] = df[c] - t_days
    return df


def add_cents(df: pd.DataFrame, amt_col: str = "TransactionAmt") -> pd.DataFrame:
    if amt_col in df.columns:
        df["cents"] = (df[amt_col] - np.floor(df[amt_col])).astype("float32")
    return df


def _safe_str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("NA")


def fit_frequency_maps(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """
    Train-only frequency maps. No Kaggle-test mixing.
    """
    maps: Dict[str, Dict[str, float]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = _safe_str(df[col])
        vc = s.value_counts(dropna=False, normalize=True).to_dict()
        maps[col] = vc
    return maps


def apply_frequency_maps(df: pd.DataFrame, maps: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    for col, mp in maps.items():
        if col not in df.columns:
            continue
        s = _safe_str(df[col])
        df[f"{col}_FE"] = s.map(mp).fillna(0.0).astype("float32")
    return df


def fit_label_maps(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, Dict[str, int]]:
    """
    Train-only label encodings.
    """
    maps: Dict[str, Dict[str, int]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = _safe_str(df[col])
        uniques = pd.Index(s.unique())
        mp = {k: i for i, k in enumerate(uniques.tolist())}
        maps[col] = mp
    return maps


def apply_label_maps(df: pd.DataFrame, maps: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    for col, mp in maps.items():
        if col not in df.columns:
            continue
        s = _safe_str(df[col])
        df[col] = s.map(mp).fillna(-1).astype("int32")
    return df


def make_combo(df: pd.DataFrame, col1: str, col2: str, out: str) -> pd.DataFrame:
    if col1 in df.columns and col2 in df.columns:
        df[out] = _safe_str(df[col1]) + "_" + _safe_str(df[col2])
    return df


def fit_group_agg_maps(
    df: pd.DataFrame,
    main_cols: Sequence[str],
    group_cols: Sequence[str],
    aggs: Sequence[str] = ("mean", "std"),
) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """
    Train-only group aggregation maps:
      key=(main, group, agg) -> mapping {group_value -> agg_value}
    """
    maps: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for g in group_cols:
        if g not in df.columns:
            continue
        for m in main_cols:
            if m not in df.columns:
                continue
            for agg in aggs:
                temp = df[[g, m]].copy()
                # ensure numeric for aggregations
                temp[m] = pd.to_numeric(temp[m], errors="coerce")
                mp = temp.groupby(g)[m].agg(agg).to_dict()
                maps[(m, g, agg)] = mp
    return maps


def apply_group_agg_maps(df: pd.DataFrame, maps: Dict[Tuple[str, str, str], Dict[str, float]]) -> pd.DataFrame:
    for (m, g, agg), mp in maps.items():
        if g not in df.columns:
            continue
        out = f"{m}_{g}_{agg}"
        df[out] = df[g].map(mp).astype("float32")
        df[out] = df[out].fillna(-1.0).astype("float32")
    return df


def finalize_for_xgb(df: pd.DataFrame, drop_cols: Sequence[str]) -> pd.DataFrame:
    """
    Final numeric matrix for XGB:
    - factorize remaining object/categorical (train-only already handled by label maps for chosen cols)
    - fill NaNs with -1
    """
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    for c in df.columns:
        if df[c].dtype.name in ("object", "category", "string"):
            # last-resort factorize per split (OK for leftovers)
            codes, _ = pd.factorize(_safe_str(df[c]), sort=True)
            df[c] = codes.astype("int32")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].fillna(-1)

    return df


def build_magic_features_train_val(train: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build features using TRAIN-ONLY fitting (production-safe), then apply to VAL.
    """

    # Basic “magic” base features
    train = train.copy()
    val = val.copy()

    train = normalize_d_columns(train)
    val = normalize_d_columns(val)

    train = add_cents(train)
    val = add_cents(val)

    train = add_dt_m(train)
    val = add_dt_m(val)

    # FE columns (only if exist)
    fe_cols = ["addr1", "card1", "card2", "card3", "P_emaildomain"]
    fe_maps = fit_frequency_maps(train, fe_cols)
    train = apply_frequency_maps(train, fe_maps)
    val = apply_frequency_maps(val, fe_maps)

    # Combo cols + label enc
    train = make_combo(train, "card1", "addr1", "card1_addr1")
    val = make_combo(val, "card1", "addr1", "card1_addr1")

    train = make_combo(train, "card1_addr1", "P_emaildomain", "card1_addr1_P_emaildomain")
    val = make_combo(val, "card1_addr1", "P_emaildomain", "card1_addr1_P_emaildomain")

    le_cols = ["card1_addr1", "card1_addr1_P_emaildomain", "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
    le_maps = fit_label_maps(train, le_cols)
    train = apply_label_maps(train, le_maps)
    val = apply_label_maps(val, le_maps)

    # Group aggregates (train-only)
    group_cols = [c for c in ["card1", "card1_addr1"] if c in train.columns]
    main_cols = [c for c in ["TransactionAmt", "D9", "D11", "D4", "D10", "D15"] if c in train.columns]
    ag_maps = fit_group_agg_maps(train, main_cols=main_cols, group_cols=group_cols, aggs=("mean", "std"))
    train = apply_group_agg_maps(train, ag_maps)
    val = apply_group_agg_maps(val, ag_maps)

    # outsider15 if D1 and D15 exist
    if "D1" in train.columns and "D15" in train.columns:
        train["outsider15"] = (np.abs(train["D1"] - train["D15"]) > 3).astype("int8")
    if "D1" in val.columns and "D15" in val.columns:
        val["outsider15"] = (np.abs(val["D1"] - val["D15"]) > 3).astype("int8")

    return train, val
