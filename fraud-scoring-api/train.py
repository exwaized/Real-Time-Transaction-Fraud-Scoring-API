import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
warnings.filterwarnings("ignore")

os.makedirs("app/models", exist_ok=True)

# ── Synthetic data ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
n, n_fraud = 10000, 300
cats = ["electronics","online","atm","travel","grocery","fuel","restaurant","pharmacy"]

legit = pd.DataFrame({
    "amount": rng.exponential(120, n-n_fraud).clip(1,5000),
    "hour": rng.integers(6, 22, n-n_fraud),
    "day_of_week": rng.integers(0, 7, n-n_fraud),
    "merchant_category": rng.choice(cats[:5], n-n_fraud),
    "label": 0
})
fraud = pd.DataFrame({
    "amount": rng.exponential(800, n_fraud).clip(1,15000),
    "hour": rng.choice([0,1,2,3,23], n_fraud),
    "day_of_week": rng.integers(0, 7, n_fraud),
    "merchant_category": rng.choice(["electronics","online","atm"], n_fraud),
    "label": 1
})
df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
for cat in cats:
    df[f"cat_{cat}"] = (df["merchant_category"] == cat).astype(int)

FEATURES = ["amount","hour","day_of_week"] + [f"cat_{c}" for c in cats]
X = df[FEATURES].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ── SMOTE ──────────────────────────────────────────────────────────────────────
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
print(f"After SMOTE — 0: {(y_res==0).sum()} 1: {(y_res==1).sum()}")

# ── Scale + train ──────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_res_s = scaler.fit_transform(X_res)
X_test_s = scaler.transform(X_test)

model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            max_depth=6, class_weight="balanced",
                            random_state=42, verbose=-1)
model.fit(X_res_s, y_res)

# ── Threshold tuning ───────────────────────────────────────────────────────────
probs = model.predict_proba(X_test_s)[:, 1]
p, r, t = precision_recall_curve(y_test, probs)
f1s = 2*p*r/(p+r+1e-9)
best_thresh = float(t[np.argmax(f1s[:-1])])
auc = roc_auc_score(y_test, probs)
print(f"Threshold: {best_thresh:.3f} | ROC-AUC: {auc:.3f}")
print(classification_report(y_test, (probs>=best_thresh).astype(int), target_names=["legit","fraud"]))

# ── Save ───────────────────────────────────────────────────────────────────────
pickle.dump(model,  open("app/models/model.pkl",  "wb"))
pickle.dump(scaler, open("app/models/scaler.pkl", "wb"))
json.dump({
    "threshold": best_thresh, "features": FEATURES,
    "roc_auc": round(auc,4),
    "best_f1": round(float(np.max(f1s[:-1])),4),
    "feature_importances": dict(zip(FEATURES, model.feature_importances_.tolist()))
}, open("app/models/meta.json","w"), indent=2)
print("✓ Saved to app/models/")