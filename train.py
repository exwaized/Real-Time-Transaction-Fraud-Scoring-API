"""
Train fraud classifier: SMOTE oversampling + LightGBM + threshold tuning.
Saves model, scaler, threshold, and feature list to /app/models/.
"""
import pandas as pd
import numpy as np
import pickle, json, os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import warnings; warnings.filterwarnings("ignore")

os.makedirs("/home/claude/fraud-api/app/models", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("/home/claude/fraud-api/data/transactions.csv")

FEATURES = ["amount", "hour", "day_of_week",
            "cat_electronics", "cat_online", "cat_atm", "cat_travel",
            "cat_grocery", "cat_fuel", "cat_restaurant", "cat_pharmacy"]

for cat in ["electronics","online","atm","travel","grocery","fuel","restaurant","pharmacy"]:
    df[f"cat_{cat}"] = (df["merchant_category"] == cat).astype(int)

X = df[FEATURES].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ── SMOTE ──────────────────────────────────────────────────────────────────────
sm = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE — class 0: {(y_res==0).sum()}, class 1: {(y_res==1).sum()}")

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_res_s = scaler.fit_transform(X_res)
X_test_s = scaler.transform(X_test)

# ── Train LightGBM ─────────────────────────────────────────────────────────────
model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    num_leaves=31, class_weight="balanced", random_state=42, verbose=-1
)
model.fit(X_res_s, y_res)

# ── Threshold tuning (max F1 on fraud class) ───────────────────────────────────
probs = model.predict_proba(X_test_s)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_thresh = float(thresholds[np.argmax(f1s[:-1])])
best_f1 = float(np.max(f1s[:-1]))

y_pred = (probs >= best_thresh).astype(int)
auc = roc_auc_score(y_test, probs)

print(f"\nBest threshold : {best_thresh:.3f}")
print(f"Best F1 (fraud): {best_f1:.3f}")
print(f"ROC-AUC        : {auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["legit","fraud"]))

# ── Feature importances ────────────────────────────────────────────────────────
importances = dict(zip(FEATURES, model.feature_importances_.tolist()))

# ── Save artifacts ─────────────────────────────────────────────────────────────
with open("/home/claude/fraud-api/app/models/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("/home/claude/fraud-api/app/models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

meta = {
    "threshold": best_thresh,
    "features": FEATURES,
    "roc_auc": round(auc, 4),
    "best_f1": round(best_f1, 4),
    "feature_importances": importances
}
with open("/home/claude/fraud-api/app/models/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n✓ Artifacts saved to app/models/")
