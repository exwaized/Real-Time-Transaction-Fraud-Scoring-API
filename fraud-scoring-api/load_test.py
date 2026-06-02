"""
Latency benchmark — measures p50/p95/p99 against local API.
Run after docker-compose up: python load_test.py
"""

import time
import random
import statistics
import concurrent.futures
import httpx

URL = "http://localhost:8000/score"
N_REQUESTS = 500
CONCURRENCY = 20

SAMPLE_TXNS = [
    {
        "transaction_id": f"TXN_{i:06d}",
        "card_id": random.randint(1000, 9999),
        "transaction_amt": round(random.uniform(10, 2000), 2),
        "product_cd": random.choice(["W", "H", "C", "S", "R"]),
        "transaction_dt": int(time.time()) - random.randint(0, 86400),
        "balance_before": round(random.uniform(100, 5000), 2),
        "balance_after": round(random.uniform(0, 4000), 2),
        "merchant_id": random.randint(100, 999),
        "card_country": random.choices(["US", "MX", "NG", "CN"], weights=[80, 5, 8, 7])[0],
        "card_txn_count_24hr": random.randint(1, 25),
        "merchant_txn_freq": random.randint(10, 500),
    }
    for i in range(N_REQUESTS)
]


def score_one(txn):
    t0 = time.perf_counter()
    r = httpx.post(URL, json=txn, timeout=5)
    latency = (time.perf_counter() - t0) * 1000
    return latency, r.status_code


def run():
    latencies = []
    errors = 0

    print(f"Sending {N_REQUESTS} requests with concurrency={CONCURRENCY}...")
    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(score_one, txn) for txn in SAMPLE_TXNS]
        for f in concurrent.futures.as_completed(futures):
            lat, status = f.result()
            if status == 200:
                latencies.append(lat)
            else:
                errors += 1

    total_time = time.perf_counter() - t_start
    latencies.sort()

    def pct(p):
        idx = int(len(latencies) * p / 100)
        return round(latencies[min(idx, len(latencies) - 1)], 2)

    print(f"\n{'─'*40}")
    print(f"  Requests:    {len(latencies)} ok / {errors} errors")
    print(f"  Duration:    {total_time:.2f}s")
    print(f"  Throughput:  {len(latencies)/total_time:.1f} req/s")
    print(f"  p50 latency: {pct(50)} ms")
    print(f"  p95 latency: {pct(95)} ms")
    print(f"  p99 latency: {pct(99)} ms  ← target <50ms")
    print(f"  max latency: {pct(100)} ms")
    print(f"{'─'*40}")

    if pct(99) < 50:
        print("  ✅ p99 < 50ms TARGET MET")
    else:
        print("  ⚠️  p99 > 50ms — consider model quantization or caching")


if __name__ == "__main__":
    run()
