# experiments/check_label_leakage.py
"""
Does each method's score approximate the GROUND-TRUTH LABELLING FUNCTION?

WHY
===
The canonical label is

    positive  <=>  min_i ||centroid(query_j) - centroid(ref_i)|| <= tolerance

which is a statement about ABSOLUTE POSITION. DTW, Frechet and Hausdorff
all compare absolute coordinates, so their per-query best-match score is
close to a monotone function of that same centroid distance. Gromov-
Wasserstein, by construction, discards position entirely.

On KITTI 00 this shows up starkly: the coordinate baselines report
AP ~ 0.998 and Recall@1 = 1.000. Near-perfect place recognition does not
happen on KITTI -- Scan Context, a purpose-built LiDAR descriptor, sits
around 0.8-0.9. The baselines are not solving the task; they are
re-deriving the answer key.

This script quantifies that, so the paper can state it with evidence
rather than assertion.

HOW TO READ THE OUTPUT
======================
Spearman correlation between each method's best-match score and the true
centroid distance:

    > 0.90   the method is essentially computing the labelling function
    0.5-0.9  substantial overlap with the label definition
    < 0.5    the method is measuring something else

An AUC-style number is also reported: how well the score alone separates
positives from negatives, which is the same thing seen from the other side.

USAGE
=====
    python experiments/check_label_leakage.py
    python experiments/check_label_leakage.py --skip-slow
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import config
from core.trajectory_utils import load_trajectory, prepare_segments
from loop_closure import LoopClosureDetector
from experiments.run_baseline_comparison import (
    compute_dtw_matrix, compute_frechet_matrix, compute_hausdorff_matrix,
)


def main(skip_slow=False):
    print("=" * 74)
    print("LABEL LEAKAGE CHECK")
    print("=" * 74)
    print(f"Dataset   : {config.DATASET_SHORT}")
    print(f"Tolerance : {config.SPATIAL_TOLERANCE} m")

    poses = repo_root / config.POSES_DIR
    ref_df = load_trajectory(poses / config.BAG3_CSV)
    query_df = load_trajectory(poses / config.BAG7_CSV)

    print("\nReference trajectory:")
    ref_raw, fps_ref = prepare_segments(
        ref_df, config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE)
    print("Query trajectory:")
    q_raw, fps_q = prepare_segments(
        query_df, config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE)

    n_ref, n_q = len(ref_raw), len(q_raw)
    print(f"\nReference segments: {n_ref}   Query segments: {n_q}")

    max_q = getattr(config, "MAX_BASELINE_QUERIES",
                    getattr(config, "MAX_BASELINE_SEGMENTS", 150))
    q_idx = (np.arange(n_q) if n_q <= max_q
             else np.linspace(0, n_q - 1, max_q, dtype=int))

    detector = LoopClosureDetector(
        segment_length=config.SEGMENT_LENGTH, stride=config.STRIDE, fps=fps_ref,
        lambdaa=config.LPGW_LAMBDA,
        downsample_points=None,
        reference_strategy=config.LPGW_REFERENCE_STRATEGY,
        cost_mode=config.LPGW_COST_MODE,
        huber_delta_frac=config.LPGW_HUBER_DELTA_FRAC,
    )
    dl = getattr(config, "LPGW_DISTANCE_LAMBDA", None)
    if dl is not None and hasattr(detector.lpgw, "distance_lambda"):
        detector.lpgw.distance_lambda = float(dl)

    npts = config.LPGW_SEGMENT_POINTS
    ref_segs = [detector.downsample_segment(s, npts) for s in ref_raw]
    q_segs = [detector.downsample_segment(q_raw[i], npts) for i in q_idx]

    cost_scale = getattr(config, "LPGW_COST_SCALE", None)
    if cost_scale is None:
        cost_scale = detector.lpgw.calibrate(ref_segs)
    else:
        detector.lpgw.cost_scale = float(cost_scale)

    _, ref_idx = detector.select_reference(ref_segs, config.LPGW_REFERENCE_STRATEGY)
    detector.reference_index = ref_idx
    print(f"lambda={config.LPGW_LAMBDA}  distance_lambda="
          f"{getattr(detector.lpgw, 'distance_lambda', config.LPGW_LAMBDA)}  "
          f"cost_scale={cost_scale:.6g}  ref_idx={ref_idx}")

    # ---- the quantity the LABEL is built from -----------------------
    rc = np.array([s.mean(axis=0) for s in ref_segs])
    qc = np.array([s.mean(axis=0) for s in q_segs])
    true_dist = cKDTree(rc).query(qc, k=1)[0]
    y_true = true_dist <= config.SPATIAL_TOLERANCE

    print(f"Queries evaluated : {len(q_segs)}")
    print(f"Positives         : {y_true.sum()}/{len(y_true)} "
          f"({y_true.mean():.1%})")

    # ---- distance matrices ------------------------------------------
    methods = [("LPGW", None), ("Hausdorff", compute_hausdorff_matrix)]
    if not skip_slow:
        methods += [("DTW", compute_dtw_matrix), ("Frechet", compute_frechet_matrix)]

    print("\nComputing distance matrices...")
    rows = []
    for name, fn in methods:
        print(f"  {name}...")
        D = (detector.compute_distance_matrix(q_segs, ref_segs)
             if name == "LPGW" else fn(q_segs, ref_segs))
        score = np.min(D, axis=1)

        rho = spearmanr(score, true_dist).correlation
        r = pearsonr(score, true_dist).statistic

        # Same thing from the other side: how separable are the classes?
        pos, neg = score[y_true], score[~y_true]
        auc = float(np.mean(
            (neg[None, :] > pos[:, None]).astype(float)
            + 0.5 * (neg[None, :] == pos[:, None])
        )) if pos.size and neg.size else float("nan")

        rows.append({"method": name, "spearman": rho, "pearson": r, "auc": auc})

    print("\n" + "=" * 74)
    print("CORRELATION WITH THE LABELLING FUNCTION")
    print("=" * 74)
    print(f"{'method':12s} {'spearman':>10} {'pearson':>10} {'AUC':>8}   verdict")
    print("-" * 74)
    for row in rows:
        rho = row["spearman"]
        if rho > 0.90:
            verdict = "COMPUTES THE LABEL"
        elif rho > 0.50:
            verdict = "substantial overlap with label"
        else:
            verdict = "measures something else"
        print(f"{row['method']:12s} {rho:10.3f} {row['pearson']:10.3f} "
              f"{row['auc']:8.3f}   {verdict}")
    print("-" * 74)

    df = pd.DataFrame(rows)
    df["dataset"] = config.DATASET_SHORT
    df["tolerance_m"] = config.SPATIAL_TOLERANCE
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"label_leakage_{config.DATASET_SHORT}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to: {out}")

    print("\nINTERPRETATION")
    print("A coordinate-based distance correlating above ~0.9 with the")
    print("centroid distance is not competing on the task -- it is")
    print("approximating the oracle that generated the labels. Report this")
    print("explicitly; the comparison is not a fair contest, and the")
    print("apparent near-perfect baseline scores follow from the choice of")
    print("ground-truth definition rather than from method quality.")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true",
                    help="Skip DTW and Frechet (keep LPGW + Hausdorff)")
    a = ap.parse_args()
    main(skip_slow=a.skip_slow)
