import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

from src.utils.paths import get_project_root, get_params_path


# --------------------------------------------------
# Load config
# --------------------------------------------------
PROJECT_ROOT = get_project_root()
with open(get_params_path(), "r") as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg["data"]
FEAT_CFG = cfg["features"]

INTERIM_DIR = PROJECT_ROOT / "data/interim"
FEATURES_DIR = PROJECT_ROOT / "data/features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def encode_categoricals(train_df, test_df):
    cat_cols = train_df.select_dtypes(include="object").columns

    for col in cat_cols:
        le = LabelEncoder()

        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

        le.fit(train_df[col])

        train_df[col] = le.transform(train_df[col])
        test_df[col] = test_df[col].map(
            lambda x: le.transform([x])[0]
            if x in le.classes_
            else FEAT_CFG["unknown_category_value"]
        )

    return train_df, test_df


def main():
    print("🔹 Loading interim data")
    train_df = pd.read_parquet(INTERIM_DIR / "train_merged.parquet")
    test_df = pd.read_parquet(INTERIM_DIR / "test_merged.parquet")

    target = DATA_CFG["target_col"]
    id_col = DATA_CFG["id_col"]

    y_train = train_df[target]
    X_train = train_df.drop(columns=[target, id_col], errors="ignore")
    X_test = test_df.drop(columns=[id_col], errors="ignore")

    print("🔹 Encoding categoricals (model-agnostic)")
    X_train, X_test = encode_categoricals(X_train, X_test)

    print("🔹 Saving model-agnostic features")
    X_train.to_parquet(FEATURES_DIR / "X_train.parquet")
    y_train.to_frame(target).to_parquet(FEATURES_DIR / "y_train.parquet")
    X_test.to_parquet(FEATURES_DIR / "X_test.parquet")

    print("✅ Feature engineering completed")


if __name__ == "__main__":
    main()
