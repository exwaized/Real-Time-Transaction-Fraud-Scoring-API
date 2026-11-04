"""
Train and evaluate the fraud model on REAL card transactions (ULB / Worldline
credit-card fraud dataset: 284,807 transactions over two days, 492 frauds).

Differences from train.py (synthetic demo):
  * real labels, real class imbalance (~0.17% fraud), no SMOTE / class re-weighting
    (scale_pos_weight cut PR-AUC from ~0.86 to ~0.24 in testing, so it is not used)
  * exact duplicate rows are dropped before splitting
  * stratified 64/16/20 train/val/test split (the OpenML copy has no Time column,
    so a chronological split is not possible; scores may be slightly optimistic
    versus a time-ordered split)
  * decision threshold tuned on a validation slice, never on the test set
  * reports ROC-AUC, PR-AUC (average precision), precision/recall at the threshold

Data: set FRAUD_DATA to a local creditcard.csv, otherwise it is fetched from
OpenML (data_id=1597) and cached next to this file.

Run:  python train_real.py
"""
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score,
                             roc_auc_score)

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
MODEL_DIR = HERE / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = ["amount_log"] + V_COLS


def load_data() -> pd.DataFrame:
    path = Path(os.environ.get("FRAUD_DATA", HERE / "creditcard.csv"))
    if path.exists():
        return pd.read_csv(path)
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=1597, as_frame=True, parser="auto").frame
    df.to_csv(path, index=False)
    return df


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = df[V_COLS].copy()
    out.insert(0, "amount_log", np.log1p(df["Amount"]))
    return out[FEATURES]


def main():
    df = load_data()
    raw_n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)  # exact duplicate rows would leak across splits
    print(f"dropped {raw_n - len(df)} exact duplicate rows ({raw_n} -> {len(df)})")
    df["Class"] = df["Class"].astype(int)
    X, y = featurize(df), df["Class"].values

    X_rest, X_te, y_rest, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_rest, y_rest, test_size=0.2, stratify=y_rest, random_state=42)
    print(f"train={len(X_tr)} (fraud {y_tr.sum()}) val={len(X_val)} "
          f"(fraud {y_val.sum()}) test={len(X_te)} (fraud {y_te.sum()})")

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=-1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(X_tr, y_tr)

    # threshold on validation slice (max F1), then frozen for the test period
    p, r, t = precision_recall_curve(y_val, model.predict_proba(X_val)[:, 1])
    f1 = 2 * p * r / (p + r + 1e-9)
    threshold = float(t[int(np.argmax(f1[:-1]))])

    probs = model.predict_proba(X_te)[:, 1]
    pred = (probs >= threshold).astype(int)
    metrics = {
        "dataset": f"ULB credit-card fraud (OpenML 1597), {len(df)} unique txns after dedup, {int(df.Class.sum())} frauds",
        "split": "stratified random 64/16/20 train/val/test (no Time column in OpenML copy)",
        "n_test": int(len(y_te)), "n_test_fraud": int(y_te.sum()),
        "roc_auc": round(float(roc_auc_score(y_te, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_te, probs)), 4),
        "threshold": round(threshold, 4),
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "frauds_caught": int(((pred == 1) & (y_te == 1)).sum()),
        "false_alarms": int(((pred == 1) & (y_te == 0)).sum()),
    }

    # single-row inference latency on this machine (CPU)
    row = X_te.iloc[[0]]
    lat = []
    for _ in range(500):
        t0 = time.perf_counter()
        model.predict_proba(row)
        lat.append((time.perf_counter() - t0) * 1000)
    metrics["latency_ms_p50"] = round(float(np.percentile(lat, 50)), 2)
    metrics["latency_ms_p95"] = round(float(np.percentile(lat, 95)), 2)
    print(json.dumps(metrics, indent=2))

    with open(MODEL_DIR / "model_real.pkl", "wb") as f:
        pickle.dump(model, f)
    meta = {"features": FEATURES, "threshold": threshold, "metrics": metrics,
            "feature_importances": dict(zip(FEATURES, model.feature_importances_.tolist()))}
    with open(MODEL_DIR / "meta_real.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("saved app/models/model_real.pkl and meta_real.json")


if __name__ == "__main__":
    main()
