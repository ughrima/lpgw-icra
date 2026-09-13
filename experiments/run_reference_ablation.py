# experiments/run_reference_ablation.py
"""
How much does the arbitrary choice of fixed reference segment matter?

WHY THIS EXISTS
===============
LPGW embeds every segment against ONE reference space. That choice is
arbitrary, and a reviewer will ask whether the results hinge on it. This
sweeps the reference index across the trajectory and reports the spread.

WHAT IT CONTROLS FOR
====================
Only reference_index varies. cost_scale, lambda, distance_lambda, the
query subset and the ground truth are all pinned, so the spread reflects
reference choice and nothing else. Recalibrating per reference would
confound two variables.

A NOTE ON THE OLD 'robust' HEURISTIC
====================================
The previous selector scored 0.4*size + 0.3*complexity + 0.3*density.
Every segment holds the same number of points, so the size term was a
constant 1.0 and contributed nothing. Density was points / bbox-volume;
near-planar segments (very common) drove one bbox dimension toward zero,
so volume collapsed, density exploded, and after normalising by the max
that one segment took 1.0 while everything else sat near 0. The heuristic
therefore selected the FLATTEST segment. It is replaced by median-diameter
selection, which is one line and defensible; this ablation checks that the
choice barely matters either way.

USAGE
=====
    python experiments/run_reference_ablation.py
    python experiments/run_reference_ablation.py --n-refs 9
    python experiments/run_reference_ablation.py --skip-slow
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


def retrieval_metrics(D, y_true, gt_ref=None, tol=2):
    y = np.asarray(y_true, dtype=bool)
    s = -np.min(np.asarray(D, dtype=float), axis=1)
    base = float(y.mean())
    out = {"base_rate": base,
           "all_positive_f1": 2 * base / (1 + base) if base > 0 else 0.0,
           "random_ap": base}

    if y.any() and not y.all():
        prec, rec, _ = precision_recall_curve(y, s)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        out["max_f1"] = float(np.max(f1))
        out["average_precision"] = float(average_precision_score(y, s))
        perfect = prec >= 0.999
        out["recall_at_100_precision"] = float(rec[perfect].max()) if perfect.any() else 0.0
    else:
        for k in ("max_f1", "average_precision", "recall_at_100_precision"):
            out[k] = float("nan")

    if gt_ref is not None and y.any():
        pos = np.flatnonzero(y)
        hits = sum(abs(int(np.argmin(D[i])) - int(gt_ref[i])) <= tol for i in pos)
        out["recall_at_1"] = hits / len(pos)
    else:
        out["recall_at_1"] = float("nan")
    return out


def main(n_refs=None, skip_slow=False):
    print("=" * 74)
    print("REFERENCE SEGMENT ABLATION")
    print("=" * 74)
    print(f"Dataset: {config.DATASET_SHORT}")

    poses = repo_root / config.POSES_DIR
    ref_df = load_trajectory(poses / config.BAG3_CSV)
    q_df = load_trajectory(poses / config.BAG7_CSV)

    print("\nReference trajectory:")
    ref_raw, fps_ref = prepare_segments(
        ref_df, config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE)
    print("Query trajectory:")
    q_raw, _ = prepare_segments(
        q_df, config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE)

    n_ref, n_q = len(ref_raw), len(q_raw)
    print(f"\nReference segments: {n_ref}   Query segments: {n_q}")

    gt = pd.read_csv(
        repo_root / "ground_truth" / "files"
        / f"gt_{config.DATASET_SHORT}_{config.SPATIAL_TOLERANCE}m.csv"
    ).sort_values("query_index").reset_index(drop=True)
    if len(gt) != n_q:
        raise ValueError(f"GT has {len(gt)} rows, experiment has {n_q} queries")

    max_q = getattr(config, "MAX_BASELINE_QUERIES",
                    getattr(config, "MAX_BASELINE_SEGMENTS", 150))
    q_idx = (np.arange(n_q) if n_q <= max_q
             else np.linspace(0, n_q - 1, max_q, dtype=int))
    y_true = gt["label"].to_numpy(bool)[q_idx]
    gt_ref = gt["nearest_ref_index"].to_numpy()[q_idx]
    print(f"Queries: {len(q_idx)}   positives: {y_true.sum()} "
          f"({y_true.mean():.1%})")

    detector = LoopClosureDetector(
        segment_length=config.SEGMENT_LENGTH, stride=config.STRIDE, fps=fps_ref,
        lambdaa=config.LPGW_LAMBDA, downsample_points=None,
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

    # Pinned: only reference_index varies below.
    cs = getattr(config, "LPGW_COST_SCALE", None)
    if cs is None:
        cs = detector.lpgw.calibrate(ref_segs)
    else:
        detector.lpgw.cost_scale = float(cs)
    print(f"lambda={config.LPGW_LAMBDA}  distance_lambda="
          f"{getattr(detector.lpgw, 'distance_lambda', config.LPGW_LAMBDA)}  "
          f"cost_scale={cs:.6g} (all pinned)")

    _, default_idx = detector.select_reference(
        ref_segs, config.LPGW_REFERENCE_STRATEGY)

    fracs = (getattr(config, "REFERENCE_ABLATION_FRACTIONS",
                     [0.0, 0.25, 0.5, 0.75, 1.0])
             if n_refs is None else list(np.linspace(0, 1, n_refs)))
    indices = sorted(set(
        [int(round(f * (n_ref - 1))) for f in fracs] + [default_idx]
    ))
    print(f"\nTesting reference indices: {indices}")
    print(f"  (median-diameter default is {default_idx})")

    tol = getattr(config, "RECALL_AT_1_INDEX_TOL", 2)
    rows = []
    for idx in indices:
        detector.reference_index = idx
        t0 = time.perf_counter()
        D = detector.compute_distance_matrix(q_segs, ref_segs)
        elapsed = time.perf_counter() - t0

        m = retrieval_metrics(D, y_true, gt_ref, tol)

        # Degeneracy check: additive row+column structure means argmin
        # collapses onto one column for every row.
        row, col = D.mean(1, keepdims=True), D.mean(0, keepdims=True)
        resid = D - row - col + D.mean()
        m["additive_variance"] = float(
            1 - resid.var() / D.var()) if D.var() > 0 else 1.0
        m["unique_argmins"] = int(len(np.unique(D.argmin(axis=1))))

        m.update({
            "reference_index": idx,
            "reference_fraction": idx / max(n_ref - 1, 1),
            "is_default": idx == default_idx,
            "diameter_m": float(np.max(
                np.linalg.norm(
                    ref_segs[idx][:, None] - ref_segs[idx][None], axis=2))),
            "runtime_sec": elapsed,
            "cost_scale": cs,
            "lambda": config.LPGW_LAMBDA,
        })
        rows.append(m)

        star = " <- default" if idx == default_idx else ""
        print(f"  idx {idx:4d} (frac {m['reference_fraction']:.2f}, "
              f"diam {m['diameter_m']:6.1f} m)  "
              f"maxF1={m['max_f1']:.4f}  AP={m['average_precision']:.4f}  "
              f"R@1={m['recall_at_1']:.4f}{star}")

    df = pd.DataFrame(rows)
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"reference_ablation_{config.DATASET_SHORT}.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 74)
    print("SENSITIVITY TO REFERENCE CHOICE")
    print("=" * 74)
    for k in ("max_f1", "average_precision", "recall_at_1"):
        v = df[k].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size:
            print(f"  {k:18s} min {v.min():.4f}  max {v.max():.4f}  "
                  f"range {v.max()-v.min():.4f}  sd {v.std():.4f}")

    b = float(y_true.mean())
    print(f"\nChance: all-positive F1 = {2*b/(1+b):.4f}, random AP = {b:.4f}")
    print(f"Saved to: {out}")
    print("\nHOW TO READ THIS: a small range means the linearisation is")
    print("insensitive to the arbitrary reference, which is what you want to")
    print("claim. A large range means the reference is a hidden hyper-")
    print("parameter and must be reported as such. Compare the range against")
    print("the bootstrap CI width -- if it is smaller, reference choice is")
    print("not a meaningful source of variation.")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-refs", type=int, default=None,
                    help="Number of evenly spaced reference indices to test")
    ap.add_argument("--skip-slow", action="store_true")
    a = ap.parse_args()
    main(n_refs=a.n_refs, skip_slow=a.skip_slow)
