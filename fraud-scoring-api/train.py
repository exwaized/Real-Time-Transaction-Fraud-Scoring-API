"""
Training pipeline for Real-Time Transaction Fraud Scoring.
Uses PaySim synthetic dataset (fallback to synthetic if not present).
Trains Champion (LightGBM) + Challenger (XGBoost) models.
Logs to MLflow: AUC-ROC, KS Statistic, Gini, precision@top5%.
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy import stats
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_PATH = "data/PS_20174392719_1491204439457_log.csv"  # PaySim
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        log.info("Loading PaySim dataset...")
        df = pd.read_csv(DATA_PATH)
        df = df.rename(columns={
            "amount": "TransactionAmt",
            "isFraud": "isFraud",
            "type": "ProductCD",
            "nameOrig": "card_id",
            "step": "TransactionDT",
            "oldbalanceOrg": "balance_before",
            "newbalanceOrig": "balance_after",
            "nameDest": "merchant_id",
        })
        df["TransactionID"] = range(len(df))
    else:
        log.info("PaySim not found — generating synthetic dataset (50K rows)...")
        df = _generate_synthetic(50_000)
    return df


def _generate_synthetic(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_fraud = int(n * 0.035)
    n_legit = n - n_fraud

    def make_block(size, fraud):
        hours = rng.integers(0, 24, size)
        return pd.DataFrame({
            "TransactionID": range(size),
            "TransactionDT": rng.integers(0, 720, size) * 3600,
            "TransactionAmt": (
                rng.exponential(800, size) if fraud
                else rng.exponential(120, size)
            ).clip(1, 15000),
            "ProductCD": rng.choice(["W", "H", "C", "S", "R"], size,
                                     p=[0.5, 0.2, 0.15, 0.1, 0.05]),
            "card_id": rng.integers(1000, 9999, size),
            "card_country": rng.choice(
                ["US", "MX", "NG", "CN", "RU", "UK"], size,
                p=[0.7, 0.05, 0.08, 0.07, 0.05, 0.05] if fraud
                else [0.88, 0.03, 0.02, 0.03, 0.02, 0.02]
            ),
            "balance_before": rng.uniform(0, 5000, size),
            "balance_after": rng.uniform(0, 5000, size),
            "merchant_id": rng.integers(100, 999, size),
            "hour": hours,
            "isFraud": int(fraud),
        })

    df = pd.concat([make_block(n_legit, False), make_block(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["TransactionID"] = range(len(df))
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time features
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].between(22, 6).astype(int)

    # Behavioral deviation: amt vs card historical avg
    card_stats = df.groupby("card_id")["TransactionAmt"].agg(
        card_avg_amt="mean", card_std_amt="std", card_txn_count="count"
    ).reset_index()
    df = df.merge(card_stats, on="card_id", how="left")
    df["amt_vs_avg_ratio"] = df["TransactionAmt"] / (df["card_avg_amt"] + 1e-6)
    df["amt_zscore"] = (df["TransactionAmt"] - df["card_avg_amt"]) / (df["card_std_amt"] + 1e-6)

    # Velocity features (simulated via card_txn_count bucketing)
    df["velocity_1hr_bucket"] = pd.cut(
        df["card_txn_count"], bins=[0, 5, 20, 50, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Balance drain ratio (PaySim feature)
    if "balance_before" in df.columns:
        df["balance_drain_ratio"] = np.where(
            df["balance_before"] > 0,
            (df["balance_before"] - df["balance_after"]) / (df["balance_before"] + 1e-6),
            0
        )
    else:
        df["balance_drain_ratio"] = 0

    # Foreign transaction flag
    df["is_foreign_txn"] = (df.get("card_country", "US") != "US").astype(int)

    # Merchant category risk encoding
    risk_map = {"W": 0.02, "C": 0.08, "H": 0.04, "S": 0.03, "R": 0.12}
    df["merchant_risk_score"] = df["ProductCD"].map(risk_map).fillna(0.05)

    # Rule-based pre-filter score (used as feature too)
    df["rule_flag"] = (
        (df["amt_vs_avg_ratio"] > 3) |
        (df["is_foreign_txn"] == 1)
    ).astype(int)

    # Merchant txn count (proxy for frequency at merchant)
    merchant_freq = df.groupby("merchant_id")["TransactionID"].count().rename("merchant_txn_freq")
    df = df.merge(merchant_freq, on="merchant_id", how="left")

    return df


FEATURE_COLS = [
    "TransactionAmt", "hour", "day_of_week", "is_weekend", "is_night",
    "amt_vs_avg_ratio", "amt_zscore", "velocity_1hr_bucket",
    "balance_drain_ratio", "is_foreign_txn", "merchant_risk_score",
    "rule_flag", "merchant_txn_freq", "card_txn_count",
]


# ─────────────────────────────────────────────
# 3. METRICS
# ─────────────────────────────────────────────
def ks_statistic(y_true, y_prob):
    fraud_scores = y_prob[y_true == 1]
    legit_scores = y_prob[y_true == 0]
    return stats.ks_2samp(fraud_scores, legit_scores).statistic


def gini_coefficient(y_true, y_prob):
    return 2 * roc_auc_score(y_true, y_prob) - 1


def precision_at_top_k(y_true, y_prob, k_pct=0.05):
    k = max(1, int(len(y_true) * k_pct))
    top_idx = np.argsort(y_prob)[::-1][:k]
    return precision_score(y_true, np.isin(np.arange(len(y_true)), top_idx).astype(int))


# ─────────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────────
def train():
    mlflow.set_experiment("fraud_scoring_v1")

    df = load_data()
    df = engineer_features(df)

    X = df[FEATURE_COLS].fillna(0)
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    log.info(f"Train: {len(X_train)} | Test: {len(X_test)} | Fraud rate: {y_train.mean():.3%}")

    # SMOTE for class imbalance
    log.info("Applying SMOTE...")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    log.info(f"After SMOTE: {len(X_res)} samples | Fraud rate: {y_res.mean():.3%}")

    # ── Champion: LightGBM ──
    with mlflow.start_run(run_name="champion_lgbm"):
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1,  # SMOTE handles imbalance
            "verbose": -1,
            "random_state": 42,
        }
        champion = lgb.LGBMClassifier(**lgb_params)
        champion.fit(X_res, y_res, eval_set=[(X_test, y_test)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])

        y_prob = champion.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        ks = ks_statistic(y_test.values, y_prob)
        gini = gini_coefficient(y_test.values, y_prob)
        p5 = precision_at_top_k(y_test.values, y_prob)

        mlflow.log_params(lgb_params)
        mlflow.log_metrics({"auc_roc": auc, "ks_statistic": ks, "gini": gini, "precision_top5pct": p5})
        mlflow.lightgbm.log_model(champion, "champion_model")

        log.info(f"Champion  AUC={auc:.4f} | KS={ks:.4f} | Gini={gini:.4f} | P@5%={p5:.4f}")

        with open(f"{MODEL_DIR}/champion_lgbm.pkl", "wb") as f:
            pickle.dump(champion, f)

    # ── Challenger: XGBoost ──
    with mlflow.start_run(run_name="challenger_xgb"):
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": int((y_train == 0).sum() / (y_train == 1).sum()),
            "random_state": 42,
            "use_label_encoder": False,
        }
        challenger = xgb.XGBClassifier(**xgb_params, verbosity=0)
        challenger.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                       verbose=False)

        y_prob_c = challenger.predict_proba(X_test)[:, 1]
        auc_c = roc_auc_score(y_test, y_prob_c)
        ks_c = ks_statistic(y_test.values, y_prob_c)
        gini_c = gini_coefficient(y_test.values, y_prob_c)

        mlflow.log_params(xgb_params)
        mlflow.log_metrics({"auc_roc": auc_c, "ks_statistic": ks_c, "gini": gini_c})
        mlflow.xgboost.log_model(challenger, "challenger_model")

        log.info(f"Challenger AUC={auc_c:.4f} | KS={ks_c:.4f} | Gini={gini_c:.4f}")

        with open(f"{MODEL_DIR}/challenger_xgb.pkl", "wb") as f:
            pickle.dump(challenger, f)

    # Save feature list + calibration stats
    model_meta = {
        "feature_cols": FEATURE_COLS,
        "champion_metrics": {"auc_roc": round(auc, 4), "ks": round(ks, 4),
                              "gini": round(gini, 4), "precision_top5pct": round(p5, 4)},
        "challenger_metrics": {"auc_roc": round(auc_c, 4), "ks": round(ks_c, 4)},
        "card_stats": card_stats.to_dict(orient="records") if "card_stats" in dir() else [],
    }

    # Rebuild card_stats for saving
    card_stats_save = df.groupby("card_id")["TransactionAmt"].agg(
        card_avg_amt="mean", card_std_amt="std"
    ).reset_index()
    model_meta["card_avg"] = card_stats_save.set_index("card_id")["card_avg_amt"].to_dict()
    model_meta["card_std"] = card_stats_save.set_index("card_id")["card_std_amt"].to_dict()

    with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
        json.dump(model_meta, f, indent=2)

    # Threshold calibration: find cutoff at FPR=0.01
    from sklearn.metrics import roc_curve
    fpr_arr, tpr_arr, thresholds = roc_curve(y_test, y_prob)
    idx = np.searchsorted(fpr_arr, 0.01)
    calibrated_threshold = float(thresholds[min(idx, len(thresholds) - 1)])
    model_meta["default_threshold"] = calibrated_threshold
    with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
        json.dump(model_meta, f, indent=2)

    log.info(f"Calibrated threshold @ FPR=1%: {calibrated_threshold:.4f}")
    log.info("Training complete. Models saved to /models/")


if __name__ == "__main__":
    train()
