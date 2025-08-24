
import os
import sys
import time
import json
import math
import random
from pathlib import Path

# --- Try to import project modules ---
# If this file is at repo root, current dir will contain 'books_recommender'
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Fallback: also try parent dir (if script sits in a tools/ folder etc.)
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

# Project imports (no external deps beyond scikit-learn / pandas / numpy / scipy)
from books_recommender.config.configuration import AppConfiguration
from books_recommender.exception.exception_handler import AppException

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def sizeof_fmt(num, suffix="B"):
    if num < 1024:
        return f"{num} {suffix}"
    for unit in ["Ki","Mi","Gi","Ti"]:
        num /= 1024.0
        if num < 1024.0:
            return f"{num:.2f} {unit}{suffix}"
    return f"{num:.2f} Pi{suffix}"


def main():
    print("=== Readora Quick Benchmark ===")

    # 1) Load config
    cfg = AppConfiguration().get_recommendation_config()

    book_pivot_path = Path(cfg.book_pivot_serialized_objects)
    final_rating_path = Path(cfg.final_rating_serialized_objects)
    model_path       = Path(cfg.trained_model_path)
    artifacts = {
        "book_pivot.pkl": str(book_pivot_path),
        "final_rating.pkl": str(final_rating_path),
        "model.pkl": str(model_path),
    }

    # 2) Load pivot (DataFrame) and create CSR
    t0 = time.perf_counter()
    book_pivot: pd.DataFrame = pd.read_pickle(book_pivot_path)
    load_pivot_s = time.perf_counter() - t0

    # Ensure NaNs are zero (safety; should already be 0)
    book_pivot = book_pivot.fillna(0)

    # build CSR
    t1 = time.perf_counter()
    csr = csr_matrix(book_pivot.values)
    build_csr_s = time.perf_counter() - t1

    n_rows, n_cols = book_pivot.shape
    nnz = int(csr.nnz)
    density = nnz / (n_rows * n_cols) if n_rows and n_cols else 0.0

    # Approx memory estimates
    dense_bytes = n_rows * n_cols * 8  # float64
    csr_bytes = csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes

    # 3) Train KNN (brute + cosine, k=10)
    t2 = time.perf_counter()
    model = NearestNeighbors(algorithm="brute", metric="cosine", n_neighbors=10)
    model.fit(csr)
    train_s = time.perf_counter() - t2

    # 4) Query latency (random sample of titles)
    # Warm-up
    rng = np.random.default_rng(42)
    warm_idx = int(rng.integers(low=0, high=n_rows))
    _ = model.kneighbors(csr[warm_idx, :], n_neighbors=6)

    # Measure
    trials = min(50, n_rows)  # up to 50 queries
    idxs = rng.integers(low=0, high=n_rows, size=trials)
    latencies = []
    for ix in idxs:
        q0 = time.perf_counter()
        _ = model.kneighbors(csr[ix, :], n_neighbors=6)  # 5 + self
        latencies.append(time.perf_counter() - q0)

    lat_ms = [x * 1000.0 for x in latencies]
    p50 = float(np.percentile(lat_ms, 50)) if lat_ms else math.nan
    p90 = float(np.percentile(lat_ms, 90)) if lat_ms else math.nan
    p99 = float(np.percentile(lat_ms, 99)) if lat_ms else math.nan
    mean_ms = float(np.mean(lat_ms)) if lat_ms else math.nan

    # 5) Artifact sizes (on disk)
    sizes = {}
    for name, path in artifacts.items():
        try:
            sizes[name] = os.path.getsize(path)
        except OSError:
            sizes[name] = None

    # 6) Save & print report
    report = {
        "matrix": {
            "shape": [int(n_rows), int(n_cols)],
            "nnz": nnz,
            "density": density,
        },
        "load_times_sec": {
            "load_pivot": round(load_pivot_s, 4),
            "build_csr": round(build_csr_s, 4),
            "train_knn": round(train_s, 4),
        },
        "memory_bytes": {
            "dense_estimate": dense_bytes,
            "csr_actual": csr_bytes,
            "savings_ratio": (1 - (csr_bytes / dense_bytes)) if dense_bytes else None
        },
        "query_latency_ms": {
            "mean": round(mean_ms, 2) if not math.isnan(mean_ms) else None,
            "p50": round(p50, 2) if not math.isnan(p50) else None,
            "p90": round(p90, 2) if not math.isnan(p90) else None,
            "p99": round(p99, 2) if not math.isnan(p99) else None,
            "trials": int(trials),
        },
        "artifact_sizes_bytes": sizes,
    }

    out_json = ROOT / "readora_bench.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    # Pretty print
    print("\n--- Matrix ---")
    print(f"Shape           : {n_rows} x {n_cols}")
    print(f"Non-zeros       : {nnz} ({density*100:.4f}% density)")

    print("\n--- Memory (estimate) ---")
    print(f"Dense (float64) : {dense_bytes} bytes ({sizeof_fmt(dense_bytes)})")
    print(f"CSR actual      : {csr_bytes} bytes ({sizeof_fmt(csr_bytes)})")
    if dense_bytes:
        print(f"Savings         : {(1 - csr_bytes/dense_bytes)*100:.2f}%")

    print("\n--- Timings (sec) ---")
    print(f"Load pivot      : {load_pivot_s:.4f}s")
    print(f"Build CSR       : {build_csr_s:.4f}s")
    print(f"Train KNN       : {train_s:.4f}s")

    print("\n--- Query latency (ms) ---")
    print(f"Mean            : {mean_ms:.2f} ms")
    print(f"P50 / P90 / P99 : {p50:.2f} / {p90:.2f} / {p99:.2f} ms")
    print(f"Trials          : {trials}")

    print("\n--- Artifact sizes (disk) ---")
    for k, v in sizes.items():
        if v is None:
            print(f"{k:17}: (not found)")
        else:
            print(f"{k:17}: {v} bytes ({sizeof_fmt(v)})")

    print(f"\nReport saved to: {out_json}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except AppException as e:
        print('AppException:', e)
        sys.exit(1)
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
