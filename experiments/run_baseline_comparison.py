
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import config
from core.trajectory_utils import load_trajectory, segment_trajectory, downsample_trajectory
from evaluation.retrieval_metrics import evaluate_matrix, format_row, block_bootstrap_ci
from loop_closure import LoopClosureDetector

USE_NUMBA = False
try:
    if USE_NUMBA:
        from numba import njit
except ImportError:
    USE_NUMBA = False



# =====================================================================
# Classical distances
# =====================================================================

def dtw_distance(A, B, window=None):
    """
    Exact DTW with an optional Sakoe-Chiba band.

    Pointwise distances come from one vectorised cdist call instead of
    np.linalg.norm per cell. Identical output, ~5x faster.
    """
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        return np.inf

    window = max(n, m) if window is None else max(window, abs(n - m))
    P = cdist(A, B, metric="euclidean")

    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - window), min(m, i + window)
        cur, prev, Pi = cost[i], cost[i - 1], P[i - 1]
        for j in range(lo, hi + 1):
            cur[j] = Pi[j - 1] + min(prev[j], cur[j - 1], prev[j - 1])
    return float(cost[n, m])


def discrete_frechet_distance(A, B):
    """Discrete Frechet distance; pointwise distances precomputed."""
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        return np.inf

    P = cdist(A, B, metric="euclidean")
    ca = np.full((n, m), np.inf, dtype=np.float64)
    ca[0, 0] = P[0, 0]
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], P[0, j])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], P[i, 0])
        row, prev, Pi = ca[i], ca[i - 1], P[i]
        for j in range(1, m):
            row[j] = max(min(prev[j], prev[j - 1], row[j - 1]), Pi[j])
    return float(ca[n - 1, m - 1])


def hausdorff_distance(A, B):
    """Symmetric Hausdorff distance (already vectorised)."""
    if len(A) == 0 or len(B) == 0:
        return np.inf
    P = cdist(A, B, metric="euclidean")
    return float(max(P.min(axis=1).max(), P.min(axis=0).max()))


def _matrix(fn, queries, references, label, **kw):
    D = np.zeros((len(queries), len(references)), dtype=np.float64)
    for i, q in enumerate(queries):
        for j, r in enumerate(references):
            D[i, j] = fn(q, r, **kw)
        if (i + 1) % 25 == 0 or i == len(queries) - 1:
            print(f"  {label} {i + 1}/{len(queries)}", flush=True)
    return D


def compute_dtw_matrix(queries, references, window=None):
    window = config.DTW_WINDOW if window is None else window
    return _matrix(dtw_distance, queries, references, "DTW", window=window)


def compute_frechet_matrix(queries, references):
    return _matrix(discrete_frechet_distance, queries, references, "Frechet")


def compute_hausdorff_matrix(queries, references):
    return _matrix(hausdorff_distance, queries, references, "Hausdorff")


# =====================================================================
# Data
# =====================================================================

def load_canonical_ground_truth(n_query):
    path = (repo_root / "ground_truth" / "files"
            / f"gt_{config.DATASET_SHORT}_{config.SPATIAL_TOLERANCE}m.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun: python ground_truth/generate_ground_truth.py"
        )

    gt = pd.read_csv(path).sort_values("query_index").reset_index(drop=True)
    need = {"query_index", "nearest_ref_index", "nearest_distance_m", "label"}
    missing = need - set(gt.columns)
    if missing:
        raise ValueError(f"GT missing columns: {sorted(missing)}")
    if not np.array_equal(gt["query_index"].to_numpy(), np.arange(len(gt))):
        raise ValueError("GT query indices are not consecutive from zero")
    if len(gt) != n_query:
        raise ValueError(
            f"GT has {len(gt)} rows, experiment has {n_query} query segments. "
            "Preprocessing differs between the GT generator and this script."
        )
    return gt["label"].to_numpy(dtype=bool), gt["nearest_ref_index"].to_numpy()


from core.trajectory_utils import prepare_segments

def build_segments(df):
    """Helper to prepare segments safely using prepare_segments which computes fps."""
    # If df is a DataFrame containing trajectory points and timestamps
    if isinstance(df, pd.DataFrame):
        raw_segs, fps = prepare_segments(
            df, 
            target_points=config.TARGET_POINTS, 
            segment_length=config.SEGMENT_LENGTH, 
            stride=config.STRIDE
        )
        return raw_segs
    else:
        # Fallback if it's already a numpy array of points
        # Assuming a default fps of 10.0 if timestamps aren't present
        from core.trajectory_utils import segment_trajectory
        return segment_trajectory(
            df, 
            segment_length=config.SEGMENT_LENGTH, 
            stride=config.STRIDE, 
            fps=10.0, 
            target_points=config.TARGET_POINTS
        )

# =====================================================================
# Main
# =====================================================================

def main(bootstrap: bool = False):
    print("=" * 74)
    print("BASELINE COMPARISON")
    print("=" * 74)
    print(f"Dataset: {config.ACTIVE_DATASET}")

    poses = repo_root / config.POSES_DIR
    ref_xyz = load_trajectory(poses / config.BAG3_CSV)[
        ["PosX", "PosY", "PosZ"]].to_numpy()
    q_xyz = load_trajectory(poses / config.BAG7_CSV)[
        ["PosX", "PosY", "PosZ"]].to_numpy()

    ref_segs, q_segs_all = build_segments(ref_xyz), build_segments(q_xyz)
    n = min(len(ref_segs), len(q_segs_all))
    ref_segs, q_segs_all = ref_segs[:n], q_segs_all[:n]
    print(f"Segments: {n}")

    y_all, gt_ref_all = load_canonical_ground_truth(n)
    print(f"GT positives: {y_all.sum()}/{n}  (base rate {y_all.mean():.1%})")

    # Queries subsampled; REFERENCE DATABASE KEPT WHOLE (see note 1).
    max_q = getattr(config, "MAX_BASELINE_QUERIES", 150)
    if config.BASELINE_SEGMENT_POLICY == "full" or n <= max_q:
        q_idx = np.arange(n)
    else:
        q_idx = np.linspace(0, n - 1, max_q, dtype=int)

    queries = [q_segs_all[i] for i in q_idx]
    y_true = y_all[q_idx]
    gt_ref = gt_ref_all[q_idx]
    print(f"Queries: {len(queries)}   References: {len(ref_segs)}")
    print(f"Positives in subset: {y_true.sum()}/{len(y_true)} "
          f"(base rate {y_true.mean():.1%})")

    rows = []

    # ---------------- LPGW ----------------
    print("\n" + "=" * 74)
    print("[1/4] LPGW")
    print("=" * 74)

    detector = LoopClosureDetector(
        segment_length=config.SEGMENT_LENGTH, fps=config.FPS, stride=config.STRIDE,
        lambdaa=config.LPGW_LAMBDA,
        downsample_points=config.LPGW_SEGMENT_POINTS,
        reference_strategy=config.LPGW_REFERENCE_STRATEGY,
        cost_mode=config.LPGW_COST_MODE,
        huber_delta_frac=config.LPGW_HUBER_DELTA_FRAC,
        cost_scale=config.LPGW_COST_SCALE,
    )

    diag = detector.run_diagnostics(ref_segs)
    if diag["transported_mass_min"] > 0.999:
        print("\n*** Partial transport is INACTIVE. Every lambda term in")
        print("    distance() is identically zero, so these results measure")
        print("    plain GW. Lower config.LPGW_LAMBDA and re-run.\n")

    t0 = time.perf_counter()
    D = detector.compute_distance_matrix(queries, ref_segs)
    rt = time.perf_counter() - t0
    m = evaluate_matrix(D, y_true, gt_ref, config.RECALL_AT_1_INDEX_TOL,
                        percentile=config.PERCENTILE)
    m.update({"method": "LPGW", "runtime_sec": rt})
    rows.append(m)
    print(format_row("LPGW", m))
    np.save(repo_root / "results" / f"D_lpgw_{config.DATASET_SHORT}.npy", D)

    # ---------------- classical ----------------
    for k, (name, fn) in enumerate(
        [("DTW", compute_dtw_matrix),
         ("Frechet", compute_frechet_matrix),
         ("Hausdorff", compute_hausdorff_matrix)], start=2
    ):
        print("\n" + "=" * 74)
        print(f"[{k}/4] {name}")
        print("=" * 74)
        t0 = time.perf_counter()
        Dm = fn(queries, ref_segs)
        rt = time.perf_counter() - t0
        mm = evaluate_matrix(Dm, y_true, gt_ref, config.RECALL_AT_1_INDEX_TOL,
                             percentile=config.PERCENTILE)
        mm.update({"method": name, "runtime_sec": rt})
        rows.append(mm)
        print(format_row(name, mm))

    # ---------------- trivial reference rows ----------------
    base = float(y_true.mean())
    rng = np.random.default_rng(config.RANDOM_SEED)
    D_rand = rng.random((len(queries), len(ref_segs)))
    mr = evaluate_matrix(D_rand, y_true, gt_ref, config.RECALL_AT_1_INDEX_TOL)
    mr.update({"method": "Random (chance)", "runtime_sec": 0.0})
    rows.append(mr)
    rows.append({
        "method": "All-positive", "max_f1": 2 * base / (1 + base),
        "average_precision": base, "recall_at_1": float("nan"),
        "recall_at_100_precision": 0.0, "base_rate": base,
        "random_ap": base, "all_positive_f1": 2 * base / (1 + base),
        "runtime_sec": 0.0, "n_queries": len(y_true),
        "n_positives": int(y_true.sum()),
    })

    df = pd.DataFrame(rows)
    for k, v in {
        "dataset": config.ACTIVE_DATASET, "seed": config.RANDOM_SEED,
        "num_query_segments": len(queries),
        "num_reference_segments": len(ref_segs),
        "lambda": config.LPGW_LAMBDA, "cost_mode": config.LPGW_COST_MODE,
        "ground_truth": "canonical",
    }.items():
        df[k] = v

    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"baseline_accuracy_{config.DATASET_SHORT}.csv"
    df.to_csv(out, index=False)

    if bootstrap:
        print("\nBlock bootstrap on LPGW max F1...")
        pt, lo, hi = block_bootstrap_ci(
            D, y_true, block_size=config.BOOTSTRAP_BLOCK_SIZE,
            n_resamples=config.BOOTSTRAP_RESAMPLES, seed=config.RANDOM_SEED,
        )
        print(f"  max F1 = {pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    cols = ["method", "max_f1", "average_precision", "recall_at_1",
            "recall_at_100_precision", "runtime_sec"]
    print(df[cols].to_string(index=False))
    print(f"\nBase rate {base:.4f} -> random AP {base:.4f}, "
          f"all-positive F1 {2 * base / (1 + base):.4f}")
    print("A method that does not clearly beat BOTH has no measurable signal.")
    print(f"\nSaved to: {out}")
    return df


if __name__ == "__main__":
    main(bootstrap="--bootstrap" in sys.argv)
