"""
Benchmark the champion/challenger recipe from train.py on a real public dataset.

Dataset: ULB / Worldline credit card fraud (Kaggle: mlg-ulb/creditcardfraud).
284,807 European card transactions over two days, 492 of them fraud (0.17%).
Columns: Time, V1..V28 (anonymised PCA components), Amount, Class.

The split is by time (first 60% train, next 20% validation, last 20% test) so
the model is scored on transactions that happen after the ones it learned from.
The decision threshold is chosen on the validation slice only; the test slice is
used once, for reporting.

Usage:
    python benchmarks/real_data_benchmark.py /path/to/creditcard.csv --out results.json
"""
import argparse
import json
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REQUIRED = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"not the ULB credit card dataset, missing columns: {missing[:5]}")
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, train=0.6, val=0.2):
    n = len(df)
    a, b = int(n * train), int(n * (train + val))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[int(np.argmax(f1[:-1]))])


def recall_at_fpr(y, p, fpr_target=0.01) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))


def evaluate(y, p, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "recall_at_1pct_fpr": round(recall_at_fpr(y, p), 4),
        "threshold": round(thr, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def run(df: pd.DataFrame, seed: int = 42) -> dict:
    train, val, test = time_split(df)
    Xtr, ytr = train[FEATURES].values, train["Class"].values
    Xva, yva = val[FEATURES].values, val["Class"].values
    Xte, yte = test[FEATURES].values, test["Class"].values

    # Same recipe as train.py: SMOTE on the training split only, scaler fit on the
    # resampled training data, LightGBM champion, XGBoost challenger.
    X_res, y_res = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    scaler = StandardScaler().fit(X_res)

    champion = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                  class_weight="balanced", random_state=seed, verbose=-1)
    champion.fit(scaler.transform(X_res), y_res)

    spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    challenger = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                   scale_pos_weight=spw, random_state=seed,
                                   eval_metric="logloss", verbosity=0)
    challenger.fit(scaler.transform(Xtr), ytr)

    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    baseline.fit(scaler.transform(Xtr), ytr)

    models = {
        "champion_lightgbm_smote": champion,
        "challenger_xgboost": challenger,
        "baseline_logistic_regression": baseline,
    }
    results = {}
    for name, model in models.items():
        pva = model.predict_proba(scaler.transform(Xva))[:, 1]
        pte = model.predict_proba(scaler.transform(Xte))[:, 1]
        thr = best_f1_threshold(yva, pva)          # chosen on validation only
        results[name] = evaluate(yte, pte, thr)    # reported on the untouched test slice

    return {
        "dataset": "ULB credit card fraud (mlg-ulb/creditcardfraud)",
        "split": "time-ordered 60/20/20 (train/validation/test)",
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "fraud": {"train": int(ytr.sum()), "validation": int(yva.sum()), "test": int(yte.sum())},
        "seed": seed,
        "models": results,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="path to creditcard.csv")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    result = run(load(args.csv), seed=args.seed)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
