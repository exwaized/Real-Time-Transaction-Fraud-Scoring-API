"""
Generates a synthetic 10K transaction dataset with ~3% fraud rate.
Fraud patterns: card-testing bursts, large amount anomalies, odd hours.
Run this before train.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Reproducibility
np.random.seed(42)

N_TOTAL = 10_000
FRAUD_RATE = 0.03
N_FRAUD = int(N_TOTAL * FRAUD_RATE)
N_LEGIT = N_TOTAL - N_FRAUD

CATEGORIES = ["electronics", "online", "atm", "travel",
               "grocery", "fuel", "restaurant", "pharmacy"]

# Legit transaction weights — grocery/restaurant most common
LEGIT_CAT_WEIGHTS = [0.05, 0.20, 0.05, 0.05, 0.30, 0.10, 0.20, 0.05]

# Fraud transaction weights — electronics/online/atm skewed higher
FRAUD_CAT_WEIGHTS = [0.30, 0.30, 0.20, 0.05, 0.05, 0.02, 0.05, 0.03]


def generate_legit(n: int) -> pd.DataFrame:
    """Normal spending: daytime hours, moderate amounts, mixed categories."""
    return pd.DataFrame({
        "card_id":           [f"card_{np.random.randint(0, 500):04d}" for _ in range(n)],
        "merchant_id":       [f"merch_{np.random.randint(0, 200):03d}" for _ in range(n)],
        "amount":            np.round(np.random.lognormal(mean=4.5, sigma=0.8, size=n), 2),
        "merchant_category": np.random.choice(CATEGORIES, size=n, p=LEGIT_CAT_WEIGHTS),
        # Daytime hours: 8am-10pm
        "hour":              np.random.choice(range(8, 22), size=n),
        "day_of_week":       np.random.randint(0, 7, size=n),
        "label":             0,
    })


def generate_fraud(n: int) -> pd.DataFrame:
    """
    Fraud patterns:
    - High amounts (amount anomaly)
    - Late night hours (0-5am)
    - Electronics/online/atm skew (card-testing categories)
    """
    return pd.DataFrame({
        "card_id":           [f"card_{np.random.randint(0, 50):04d}" for _ in range(n)],
        "merchant_id":       [f"merch_{np.random.randint(0, 200):03d}" for _ in range(n)],
        # Fraud amounts: bimodal — small card-testing probes + large cashout attempts
        "amount":            np.round(
            np.where(
                np.random.rand(n) < 0.4,
                np.random.uniform(1, 20, n),           # small probe transactions
                np.random.lognormal(mean=7.5, sigma=1.0, size=n)  # large cashouts
            ), 2
        ),
        "merchant_category": np.random.choice(CATEGORIES, size=n, p=FRAUD_CAT_WEIGHTS),
        # Late night: 0-5am
        "hour":              np.random.choice(range(0, 6), size=n),
        "day_of_week":       np.random.randint(0, 7, size=n),
        "label":             1,
    })


if __name__ == "__main__":
    legit = generate_legit(N_LEGIT)
    fraud = generate_fraud(N_FRAUD)

    # Combine and shuffle so fraud isn't all at the end
    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    df = df.reset_index(drop=True)

    # Output path — relative so it works on any machine
    out_path = Path(__file__).parent / "transactions.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df):,} transactions — fraud rate: {df['label'].mean():.1%}")
    print(f"Saved to {out_path}")
