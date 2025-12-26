import pandas as pd
import numpy as np
df = pd.read_parquet("../../data/processed/train.parquet")

# Time features
df["TransactionDT"] = df["TransactionDT"] / (60*60*24)
df["TransactionAmt_log"] = df["TransactionAmt"].apply(lambda x: 0 if x <= 0 else np.log1p(x))

# Frequency encoding
for col in ["card1", "card2", "addr1"]:
    freq = df[col].value_counts()
    df[f"{col}_freq"] = df[col].map(freq)

df.to_parquet("../../data/data_features/train_features.parquet")
print("✅ Features built and saved")