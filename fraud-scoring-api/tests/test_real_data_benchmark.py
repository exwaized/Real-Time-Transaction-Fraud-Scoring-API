"""
Smoke test for benchmarks/real_data_benchmark.py using a tiny fake frame in the
ULB credit card schema (Time, V1..V28, Amount, Class). It checks the plumbing
(split, threshold selection, metrics), not model quality.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

import real_data_benchmark as bench  # noqa: E402


def _fake_ulb(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.arange(n, dtype=float)
    df["Amount"] = rng.exponential(50, n)
    df["Class"] = (rng.random(n) < 0.05).astype(int)
    df.loc[df["Class"] == 1, "V1"] += 3.0     # give the models something to find
    return df


def test_time_split_is_ordered_and_disjoint():
    df = _fake_ulb()
    train, val, test = bench.time_split(df)
    assert train["Time"].max() < val["Time"].min() <= val["Time"].max() < test["Time"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_run_reports_all_models_with_sane_metrics():
    result = bench.run(_fake_ulb())
    assert set(result["models"]) == {
        "champion_lightgbm_smote", "challenger_xgboost", "baseline_logistic_regression"}
    for m in result["models"].values():
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["pr_auc"] <= 1.0
        assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == result["rows"]["test"]


def test_load_rejects_wrong_dataset(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(p, index=False)
    try:
        bench.load(str(p))
    except ValueError as e:
        assert "missing columns" in str(e)
    else:
        raise AssertionError("expected ValueError")
