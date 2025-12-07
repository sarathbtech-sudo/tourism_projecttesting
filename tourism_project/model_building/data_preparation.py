
import os
from pathlib import Path
import pandas as pd

def main():
    DATA_DIR = Path("tourism_project1/data")
    RAW_PATH = DATA_DIR / "tourism.csv"
    PROCESSED_PATH = DATA_DIR / "processed_tourism.csv"

    df = pd.read_csv(RAW_PATH)

    df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns], errors="ignore")

    df = df.rename(columns={"ProdTaken": "will_purchase"})

    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    df.to_csv(PROCESSED_PATH, index=False)

if __name__ == "__main__":
    main()
