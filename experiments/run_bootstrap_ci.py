#experiments/run_bootstrap_ci.py
"""
Block-bootstrap confidence intervals for the headline metrics.

WHY BLOCK BOOTSTRAP AND NOT THE ORDINARY KIND
=============================================
Segments overlap in time. With a 5 s window at 0.5 s stride, neighbouring
query segments share 90% of their points, so their outcomes are strongly
dependent. Resampling queries independently pretends there are far more
independent observations than there are, and the intervals come out too
narrow.

Resampling contiguous BLOCKS of queries preserves that dependence. Block
size must exceed the overlap span: window/stride = 10 segments here, so
BOOTSTRAP_BLOCK_SIZE = 10 is the minimum sensible value.

WHY IT MATTERS FOR THIS PAPER
=============================
The UZH-FPV comparison rests on differences like AP 0.653 (LPGW) versus
0.549 (Hausdorff) on 126 queries. Whether that gap survives resampling is
exactly what a reviewer will ask, and the honest answer may be that the
interval is wide. Report it either way.

The script also bootstraps the DIFFERENCE between LPGW and each baseline
on the same resampled blocks, which is the quantity the claim depends on
and is tighter than comparing two separately-computed intervals.

USAGE
=====
    python experiments/run_bootstrap_ci.py
    python experiments/run_bootstrap_ci.py --resamples 2000 --block-size 10
    python experiments/run_bootstrap_ci.py --skip-slow
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


# Add the Linearized_Partial_Gromov_Wasserstein submodule directory to sys.path
submodule_path = repo_root / "Linearized_Partial_Gromov_Wasserstein"
if submodule_path.exists():
    sys.path.insert(0, str(submodule_path))
else:
    raise RuntimeError(
        f"Submodule directory not found at: {submodule_path}\n"
        "Please ensure Linearized_Partial_Gromov_Wasserstein is cloned inside your repository root."
    )


import config
from core.trajectory_utils import load_trajectory, prepare_segments
from loop_closure import LoopClosureDetector
from experiments.run_baseline_comparison import (
    compute_dtw_matrix, compute_frechet_matrix, compute_hausdorff_matrix,
)


def point_metrics(D, y, gt_ref=None, tol=2):
    """max F1, AP and Recall@1 for one (possibly resampled) query set."""
    y = np.asarray(y, dtype=bool)
    if not y.any() or y.all():
        return {"max_f1": np.nan, "average_precision": np.nan,
                "recall_at_1": np.nan}

    s = -np.min(D, axis=1)
    prec, rec, _ = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    out = {"max_f1": float(np.max(f1)),
           "average_precision": float(average_precision_score(y, s))}

    if gt_ref is not None:
        pos = np.flatnonzero(y)
        hits = sum(abs(int(np.argmin(D[i])) - int(gt_ref[i])) <= tol for i in pos)
        out["recall_at_1"] = hits / len(pos)
    else:
        out["recall_at_1"] = np.nan
    return out


def blocks_for(n, block_size):
    starts = np.arange(0, n, block_size)
    return [np.arange(s, min(s + block_size, n)) for s in starts]


def bootstrap(Ds, y, gt_ref, block_size, n_resamples, seed, tol=2):
    """
    Resample blocks ONCE per iteration and score every method on the same
    resample, so method differences are paired rather than independent.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    blks = blocks_for(n, block_size)
    n_needed = int(np.ceil(n / block_size))

    draws = {m: {k: [] for k in ("max_f1", "average_precision", "recall_at_1")}
             for m in Ds}
    diffs = {m: {k: [] for k in ("max_f1", "average_precision", "recall_at_1")}
             for m in Ds if m != "LPGW"}

    kept = 0
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(blks), size=n_needed)
        idx = np.concatenate([blks[c] for c in chosen])[:n]
        yb = y[idx]
        if not yb.any() or yb.all():
            continue
        gb = None if gt_ref is None else gt_ref[idx]

        scored = {}
        for m, D in Ds.items():
            try:
                scored[m] = point_metrics(D[idx], yb, gb, tol)
            except Exception:
                scored = {}
                break
        if not scored:
            continue

        kept += 1
        for m, res in scored.items():
            for k, v in res.items():
                if v == v:
                    draws[m][k].append(v)
        if "LPGW" in scored:
            for m in diffs:
                if m in scored:
                    for k in diffs[m]:
                        a, b = scored["LPGW"][k], scored[m][k]
                        if a == a and b == b:
                            diffs[m][k].append(a - b)

    return draws, diffs, kept


def ci(vals, alpha=0.05):
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if v.size == 0:
        return np.nan, np.nan
    return (float(np.percentile(v, 100 * alpha / 2)),
            float(np.percentile(v, 100 * (1 - alpha / 2))))


def main(resamples=None, block_size=None, skip_slow=False, seed=None):
    resamples = resamples or getattr(config, "BOOTSTRAP_RESAMPLES", 10000)
    block_size = block_size or getattr(config, "BOOTSTRAP_BLOCK_SIZE", 10)
    seed = seed or config.RANDOM_SEED

    print("=" * 74)
    print("BLOCK BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 74)
    print(f"Dataset    : {config.DATASET_SHORT}")
    print(f"Resamples  : {resamples}   block size: {block_size} segments")
    overlap_span = int(round(config.SEGMENT_LENGTH / config.STRIDE))
    print(f"Overlap span: {overlap_span} segments "
          f"({config.SEGMENT_LENGTH}s window / {config.STRIDE}s stride)")
    if block_size < overlap_span:
        print(f"  WARNING: block size {block_size} < overlap span "
              f"{overlap_span}; intervals will be too narrow.")

    poses = repo_root / config.POSES_DIR
    ref_df = load_trajectory(poses / config.BAG3_CSV)
    q_df = load_trajectory(poses / config.BAG7_CSV)

    print("\nReference trajectory:")
    ref_raw, fps_ref = prepare_segments(
        ref_df, config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE)
    print("Query trajectory:")
    q_raw, _ = prepare_segments(
        q_df, config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE)

    n_q = len(q_raw)
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
    y = gt["label"].to_numpy(bool)[q_idx]
    gt_ref = gt["nearest_ref_index"].to_numpy()[q_idx]
    print(f"\nQueries: {len(q_idx)}   positives: {y.sum()} "
          f"({y.mean():.1%})   references: {len(ref_raw)}")

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

    cs = getattr(config, "LPGW_COST_SCALE", None)
    if cs is None:
        cs = detector.lpgw.calibrate(ref_segs)
    else:
        detector.lpgw.cost_scale = float(cs)
    _, ref_idx = detector.select_reference(ref_segs, config.LPGW_REFERENCE_STRATEGY)
    detector.reference_index = ref_idx

    print("\nComputing distance matrices (once)...")
    Ds = {"LPGW": detector.compute_distance_matrix(q_segs, ref_segs)}
    Ds["Hausdorff"] = compute_hausdorff_matrix(q_segs, ref_segs)
    if not skip_slow:
        Ds["DTW"] = compute_dtw_matrix(q_segs, ref_segs)
        Ds["Frechet"] = compute_frechet_matrix(q_segs, ref_segs)

    tol = getattr(config, "RECALL_AT_1_INDEX_TOL", 2)
    point = {m: point_metrics(D, y, gt_ref, tol) for m, D in Ds.items()}

    print(f"\nBootstrapping ({resamples} resamples)...")
    draws, diffs, kept = bootstrap(Ds, y, gt_ref, block_size, resamples, seed, tol)
    print(f"  {kept}/{resamples} resamples usable "
          f"(rest dropped: single-class draw)")

    print("\n" + "=" * 74)
    print("POINT ESTIMATES WITH 95% BLOCK-BOOTSTRAP CI")
    print("=" * 74)
    rows = []
    for m in ["LPGW", "Hausdorff", "DTW", "Frechet"]:
        if m not in Ds:
            continue
        print(f"\n{m}")
        for k in ("max_f1", "average_precision", "recall_at_1"):
            lo, hi = ci(draws[m][k])
            print(f"  {k:18s} {point[m][k]:.4f}   [{lo:.4f}, {hi:.4f}]")
            rows.append({"method": m, "metric": k, "point": point[m][k],
                         "ci_low": lo, "ci_high": hi})

    print("\n" + "=" * 74)
    print("PAIRED DIFFERENCES  (LPGW minus baseline, same resamples)")
    print("=" * 74)
    print("An interval excluding zero means the difference is significant.")
    for m in diffs:
        print(f"\nLPGW - {m}")
        for k in ("max_f1", "average_precision", "recall_at_1"):
            lo, hi = ci(diffs[m][k])
            d = point["LPGW"][k] - point[m][k]
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
            print(f"  {k:18s} {d:+.4f}   [{lo:+.4f}, {hi:+.4f}]   {sig}")
            rows.append({"method": f"LPGW-{m}", "metric": k, "point": d,
                         "ci_low": lo, "ci_high": hi})

    df = pd.DataFrame(rows)
    df["dataset"] = config.DATASET_SHORT
    df["n_resamples"] = kept
    df["block_size"] = block_size
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"bootstrap_ci_{config.DATASET_SHORT}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to: {out}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--skip-slow", action="store_true")
    a = ap.parse_args()
    main(resamples=a.resamples, block_size=a.block_size,
         skip_slow=a.skip_slow, seed=a.seed)
