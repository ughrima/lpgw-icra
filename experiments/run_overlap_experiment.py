# experiments/run_overlap_experiment.py
"""
Partial-overlap robustness: LPGW vs classical trajectory distances.

THE HYPOTHESIS
==============
Partial Gromov-Wasserstein relaxes the equal-mass constraint so unmatched
trajectory mass can be left untransported. The claim is that this helps
when a robot re-enters a previously visited region only partway through a
motion window -- so LPGW's advantage should GROW as overlap falls.

PROTOCOL
========
For each ratio in {1.0, 0.8, 0.6, 0.4}, retain a contiguous fraction of
each QUERY segment. Reference segments and ground-truth labels are
untouched, so the experiment measures partial observation and nothing else.

Contiguous truncation, not random point removal: randomly subsampling a
curve returns approximately the same curve at lower density, leaving the
intra-segment distance structure nearly unchanged. PGW is close to
invariant to that, so it tests nothing.

WHAT WAS WRONG BEFORE
=====================

(1) THE COST SCALE MOVED WITH THE PERTURBATION.
    A detector was constructed inside the ratio loop with no cost_scale,
    so it recalibrated at every ratio. Truncation changes segment extent,
    which changes the calibrated scale, which changes the distances -- the
    sweep measured normalisation drift as much as overlap. Visible in the
    old output as threshold_tau wandering non-monotonically:
    0.0788 -> 0.0711 -> 0.0624 -> 0.0805.
    cost_scale, lambda, distance_lambda and the reference index are now
    pinned ONCE, outside the loop.

(2) THRESHOLD-BASED METRICS AT THE BASE-RATE PERCENTILE.
    PERCENTILE = 68 is tuned to this dataset's base rate, so F1 tracks the
    base rate rather than the ranking. The old best LPGW F1 of 0.7134 was
    BELOW the all-positive classifier's 0.7208. Primary metrics are now
    threshold-free (max F1, AP, Recall@1) with the trivial baselines
    printed alongside.

(3) ONLY DTW AS A BASELINE.
    Hausdorff is the strongest competitor on this data and was absent.
    All four methods now run on identical inputs.

Saves: results/overlap_experiment_{DATASET_SHORT}.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import config
from core.trajectory_utils import load_trajectory, prepare_segments
from loop_closure import LoopClosureDetector
from experiments.run_baseline_comparison import (
    compute_dtw_matrix, compute_frechet_matrix, compute_hausdorff_matrix,
)


# ---------------------------------------------------------------------
# Partial observation
# ---------------------------------------------------------------------

def truncate_contiguous(seg, ratio, rng, min_points=20):
    """
    Retain a contiguous `ratio` fraction of a segment, random start.

    Models the robot entering or leaving the revisited region partway
    through the observation window while preserving local curve geometry.
    """
    seg = np.asarray(seg, dtype=np.float64)
    n = len(seg)
    if ratio >= 1.0:
        return seg.copy()

    keep = max(min_points, int(np.floor(n * ratio)))
    if keep >= n:
        return seg.copy()

    start = int(rng.integers(0, n - keep + 1))
    return seg[start:start + keep].copy()


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def retrieval_metrics(D, y_true, gt_nearest_ref=None, index_tol=2):
    """Threshold-free scoring with the trivial baselines alongside."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = -np.min(np.asarray(D, dtype=float), axis=1)

    base = float(y_true.mean())
    out = {
        "base_rate": base,
        "all_positive_f1": (2 * base / (1 + base)) if base > 0 else 0.0,
        "random_ap": base,
    }

    if y_true.any() and not y_true.all():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        b = int(np.argmax(f1))
        out["max_f1"] = float(f1[b])
        out["precision_at_max_f1"] = float(prec[b])
        out["recall_at_max_f1"] = float(rec[b])
        out["average_precision"] = float(average_precision_score(y_true, scores))
        perfect = prec >= 0.999
        out["recall_at_100_precision"] = float(rec[perfect].max()) if perfect.any() else 0.0
    else:
        for k in ("max_f1", "precision_at_max_f1", "recall_at_max_f1",
                  "average_precision", "recall_at_100_precision"):
            out[k] = float("nan")

    if gt_nearest_ref is not None and y_true.any():
        pos = np.flatnonzero(y_true)
        hits = sum(
            abs(int(np.argmin(D[i])) - int(gt_nearest_ref[i])) <= index_tol
            for i in pos
        )
        out["recall_at_1"] = hits / len(pos)
    else:
        out["recall_at_1"] = float("nan")

    out["f1_over_all_positive"] = out["max_f1"] - out["all_positive_f1"]
    out["ap_over_random"] = out["average_precision"] - out["random_ap"]
    return out


def load_canonical_ground_truth(n_query):
    path = (repo_root / "ground_truth" / "files"
            / f"gt_{config.DATASET_SHORT}_{config.SPATIAL_TOLERANCE}m.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun: python ground_truth/generate_ground_truth.py"
        )

    gt = pd.read_csv(path).sort_values("query_index").reset_index(drop=True)
    missing = {"query_index", "nearest_ref_index", "label"} - set(gt.columns)
    if missing:
        raise ValueError(f"GT missing columns: {sorted(missing)}")
    if len(gt) != n_query:
        raise ValueError(
            f"GT has {len(gt)} rows but experiment has {n_query} query "
            "segments. Regenerate ground truth."
        )
    return gt["label"].to_numpy(dtype=bool), gt["nearest_ref_index"].to_numpy()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(ratios=None, skip_slow=False):
    ratios = list(
        getattr(config, "OVERLAP_RATIOS", [1.0, 0.8, 0.6, 0.4])
        if ratios is None else ratios
    )

    print("=" * 74)
    print("PARTIAL-OVERLAP ROBUSTNESS  (contiguous truncation)")
    print("=" * 74)
    print(f"Dataset : {config.DATASET_SHORT}")
    print(f"Ratios  : {ratios}")

    poses = repo_root / config.POSES_DIR
    ref_df = load_trajectory(poses / config.BAG3_CSV)
    query_df = load_trajectory(poses / config.BAG7_CSV)

    print("\nReference trajectory:")
    ref_segs_raw, fps_ref = prepare_segments(
        ref_df, config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE)
    print("Query trajectory:")
    query_segs_raw, fps_q = prepare_segments(
        query_df, config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE)

    # No truncation: reference and query counts are independent.
    n_ref, n_q = len(ref_segs_raw), len(query_segs_raw)
    print(f"\nReference segments: {n_ref}   Query segments: {n_q}")

    y_true_full, gt_ref_full = load_canonical_ground_truth(n_q)
    print(f"GT positives    : {y_true_full.sum()}/{n_q}  "
          f"(base rate {y_true_full.mean():.1%})")

    max_q = getattr(config, "MAX_BASELINE_QUERIES",
                    getattr(config, "MAX_BASELINE_SEGMENTS", 150))
    q_idx = (np.arange(n_q) if n_q <= max_q
             else np.linspace(0, n_q - 1, max_q, dtype=int))
    y_true, gt_ref = y_true_full[q_idx], gt_ref_full[q_idx]
    print(f"Query subset    : {len(q_idx)}   References: {n_ref} (full DB)")

    # --- detector built ONCE, everything pinned -----------------------
    detector = LoopClosureDetector(
        segment_length=config.SEGMENT_LENGTH, stride=config.STRIDE, fps=fps_ref,
        lambdaa=config.LPGW_LAMBDA,
        downsample_points=None,          # we downsample explicitly below
        reference_strategy=config.LPGW_REFERENCE_STRATEGY,
        cost_mode=config.LPGW_COST_MODE,
        huber_delta_frac=config.LPGW_HUBER_DELTA_FRAC,
    )
    dl = getattr(config, "LPGW_DISTANCE_LAMBDA", None)
    if dl is not None and hasattr(detector.lpgw, "distance_lambda"):
        detector.lpgw.distance_lambda = float(dl)
    print(f"lambda={config.LPGW_LAMBDA}  distance_lambda="
          f"{getattr(detector.lpgw, 'distance_lambda', config.LPGW_LAMBDA)}")

    npts = config.LPGW_SEGMENT_POINTS
    ref_segs = [detector.downsample_segment(s, npts) for s in ref_segs_raw]
    print(f"Points per segment: {len(ref_segs[0])}")

    cost_scale = getattr(config, "LPGW_COST_SCALE", None)
    if cost_scale is None:
        cost_scale = detector.lpgw.calibrate(ref_segs)
    else:
        detector.lpgw.cost_scale = float(cost_scale)
    print(f"Cost scale      : {cost_scale:.6g} (pinned)")

    ref_seg, ref_idx = detector.select_reference(
        ref_segs, config.LPGW_REFERENCE_STRATEGY)
    detector.reference_index = ref_idx
    print(f"Reference index : {ref_idx}")

    print("\nVerifying PGW is active before the sweep...")
    diag = detector.lpgw.diagnose(ref_seg, ref_segs)
    if diag["transported_mass_min"] > 0.999:
        raise SystemExit(
            "\nABORT: partial transport inactive (full mass always "
            "transported).\nThis experiment tests the partial-mass "
            "mechanism specifically, so it\nis meaningless while that "
            "mechanism is off. Re-run calibrate_lambda.py."
        )

    methods = [("LPGW", None), ("Hausdorff", compute_hausdorff_matrix)]
    if not skip_slow:
        methods += [("DTW", compute_dtw_matrix), ("Frechet", compute_frechet_matrix)]

    rows = []
    for ratio in ratios:
        print("\n" + "=" * 74)
        print(f"CONTIGUOUS RETENTION = {ratio:.0%}")
        print("=" * 74)

        # Seeded per ratio so truncation is reproducible.
        rng = np.random.default_rng(config.RANDOM_SEED + int(ratio * 1000))
        truncated = [
            truncate_contiguous(query_segs_raw[i], ratio, rng) for i in q_idx
        ]
        # Downsample AFTER truncation so every method sees identical arrays.
        q_segs = [detector.downsample_segment(s, npts) for s in truncated]
        print(f"  query points: {len(query_segs_raw[q_idx[0]])} -> "
              f"{len(truncated[0])} -> {len(q_segs[0])} after downsample")

        for name, fn in methods:
            t0 = time.perf_counter()
            if name == "LPGW":
                detector.reference_index = ref_idx
                D = detector.compute_distance_matrix(q_segs, ref_segs)
            else:
                D = fn(q_segs, ref_segs)
            elapsed = time.perf_counter() - t0

            m = retrieval_metrics(
                D, y_true, gt_ref,
                getattr(config, "RECALL_AT_1_INDEX_TOL", 2),
            )
            m.update({
                "overlap_ratio": ratio,
                "method": name,
                "runtime_sec": elapsed,
                "query_points": len(q_segs[0]),
                "num_query_segments": len(q_segs),
                "num_reference_segments": len(ref_segs),
                "cost_scale": cost_scale,
                "lambda": config.LPGW_LAMBDA,
                "distance_lambda": getattr(
                    detector.lpgw, "distance_lambda", config.LPGW_LAMBDA),
                "seed": config.RANDOM_SEED,
            })
            rows.append(m)

            print(f"  {name:10s} maxF1={m['max_f1']:.4f} "
                  f"({m['f1_over_all_positive']:+.4f} vs all-pos)  "
                  f"AP={m['average_precision']:.4f} "
                  f"({m['ap_over_random']:+.4f} vs random)  "
                  f"R@1={m['recall_at_1']:.4f}")

    df = pd.DataFrame(rows)
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"overlap_experiment_{config.DATASET_SHORT}.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 74)
    print("SUMMARY  (max F1 by retention)")
    print("=" * 74)
    print(df.pivot_table(index="overlap_ratio", columns="method",
                         values="max_f1").to_string())
    print("\n(AP by retention)")
    print(df.pivot_table(index="overlap_ratio", columns="method",
                         values="average_precision").to_string())
    print("\n(Recall@1 by retention)")
    print(df.pivot_table(index="overlap_ratio", columns="method",
                         values="recall_at_1").to_string())

    b = float(y_true.mean())
    print(f"\nChance levels: all-positive F1 = {2*b/(1+b):.4f}, "
          f"random AP = {b:.4f}")
    print(f"Saved to: {out_path}")
    print("\nWHAT TO LOOK FOR: the claim is that LPGW's advantage GROWS as")
    print("retention falls. If the gap narrows instead, partial matching is")
    print("not delivering what Table I claims for it -- report that honestly.")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", type=float, nargs="+", default=None)
    ap.add_argument("--skip-slow", action="store_true",
                    help="Skip DTW and Frechet (keep LPGW + Hausdorff)")
    a = ap.parse_args()
    run(ratios=a.ratios, skip_slow=a.skip_slow)