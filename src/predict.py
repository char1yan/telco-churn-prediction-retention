"""Score customers with the saved churn model.

Usage (from the repo root):
    python src/predict.py data/raw/Telco-Customer-Churn.csv outputs/scored.csv

The input needs the same columns as the IBM Telco dataset (the Churn column is optional).
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def score(df: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(ROOT / "models" / "churn_model.joblib")
    X = df.copy()
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"].astype(str).str.strip().replace("", np.nan)).fillna(0)
    proba = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
    out = df[["customerID"]].copy()
    out["churn_probability"] = proba.round(4)
    out["target_for_offer"] = proba >= bundle["threshold"]
    return out.sort_values("churn_probability", ascending=False)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    result = score(pd.read_csv(src))
    result.to_csv(dst, index=False)
    print(f"Scored {len(result):,} customers -> {dst} ({result.target_for_offer.sum():,} flagged for an offer)")
