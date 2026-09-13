"""
Drift robustness: LPGW vs classical trajectory distances.

THE HYPOTHESIS
==============
Odometry drift moves absolute coordinates but leaves a segment's INTERNAL
geometry almost unchanged. Gromov-Wasserstein compares intra-segment
distance matrices, so it should be nearly invariant. DTW / Frechet /
Hausdorff compare absolute coordinates, so they should degrade.

This is the one experiment where LPGW's position-invariance is an asset
rather than a handicap.

WHAT TO LOOK FOR
================
LPGW roughly flat across drift levels while the coordinate-based baselines
fall. If LPGW also falls, its invariance is not surviving the barycentric
projection -- investigate K.

PROTOCOL NOTES
==============
- FPS is derived per file from timestamps. Reference and query are at
  different rates (101.01 vs 74.96 Hz here) so no single constant works.
- Reference and query segment counts are INDEPENDENT (90 vs 126). The
  reference is a database, the query a set of probes. No min_len.
- cost_scale and the reference segment index are pinned ONCE from the
  clean data and reused at every drift level, so the sweep measures drift
  rather than a shifting normalisation.
- Segments are downsampled BEFORE reference selection so the reference has
  the same point count as the targets.
- Zero-centering is OFF by default. Over a 5 s window drift is essentially
  a constant offset, which is exactly what subtracting the segment mean
  removes -- it would hand the coordinate-based baselines the same
  translation-invariance that LPGW gets from its mathematics, making the
  hypothesis untestable. Available as --center for an ablation.

Saves: results/drift_experiment_{DATASET_SHORT}.csv
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
from core.trajectory_utils import (
    load_trajectory, prepare_segments, segment_trajectory,
    downsample_trajectory, effective_fps,
)
from loop_closure import LoopClosureDetector
from experiments.run_baseline_comparison import (
    compute_dtw_matrix, compute_frechet_matrix, compute_hausdorff_matrix,
)


# ---------------------------------------------------------------------
# Drift model
# ---------------------------------------------------------------------

def make_drift_basis(traj, rng):
    """One fixed integrated-random-walk realisation, to be scaled."""
    walk = np.cumsum(rng.normal(0.0, 1.0, size=traj.shape), axis=0)
    return walk - walk[0]


def scale_drift(basis, traj, drift_fraction):
    """
    Scale the walk so its final displacement equals
    drift_fraction * path_length.

    Real odometry drift is smooth, accumulating, and conventionally
    reported as a percentage of distance travelled (typically 1-2%).
    Scaling to path length also makes the sweep comparable across
    datasets with different sampling rates and extents.
    """
    if drift_fraction <= 0:
        return np.zeros_like(traj)
    path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    final = float(np.linalg.norm(basis[-1]))
    if final < 1e-12:
        return np.zeros_like(traj)
    return basis * (drift_fraction * path_len / final)


def zero_center_segments(segments):
    """Ablation only -- see PROTOCOL NOTES."""
    return [s - s.mean(axis=0) for s in segments]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def retrieval_metrics(D, y_true, gt_nearest_ref=None, index_tol=2):
    """Threshold-free scoring, with the trivial baselines alongside."""
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
    need = {"query_index", "nearest_ref_index", "label"}
    missing = need - set(gt.columns)
    if missing:
        raise ValueError(f"GT missing columns: {sorted(missing)}")
    if len(gt) != n_query:
        raise ValueError(
            f"GT has {len(gt)} rows but experiment has {n_query} query "
            "segments. Regenerate ground truth, or check that this script "
            "segments identically to generate_ground_truth.py."
        )
    return gt["label"].to_numpy(dtype=bool), gt["nearest_ref_index"].to_numpy()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_drift_experiment(center=False, fractions=None, skip_slow=False):
    fractions = list(
        getattr(config, "DRIFT_FRACTIONS", [0.0, 0.005, 0.01, 0.02, 0.05])
        if fractions is None else fractions
    )

    print("=" * 74)
    print("DRIFT ROBUSTNESS EXPERIMENT")
    print("=" * 74)
    print(f"Dataset         : {config.DATASET_SHORT}")
    print(f"Drift fractions : {[f'{f:.1%}' for f in fractions]}")
    print(f"Zero-centering  : {'ON (ablation)' if center else 'OFF (default)'}")

    poses = repo_root / config.POSES_DIR
    ref_df = load_trajectory(poses / config.BAG3_CSV)
    query_df = load_trajectory(poses / config.BAG7_CSV)

    print("\nReference trajectory:")
    ref_segs_raw, fps_ref = prepare_segments(
        ref_df, config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE)
    print("Query trajectory:")
    query_segs_raw, fps_q = prepare_segments(
        query_df, config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE)

    # NO truncation: reference and query counts are independent.
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

    detector = LoopClosureDetector(
        segment_length=config.SEGMENT_LENGTH, stride=config.STRIDE, fps=fps_ref,
        lambdaa=config.LPGW_LAMBDA,
        downsample_points=None,          # we downsample explicitly below
        reference_strategy=config.LPGW_REFERENCE_STRATEGY,
        cost_mode=config.LPGW_COST_MODE,
        huber_delta_frac=config.LPGW_HUBER_DELTA_FRAC,
    )
    # distance_lambda is optional; set it if lpgw.py supports the split.
    dl = getattr(config, "LPGW_DISTANCE_LAMBDA", None)
    if dl is not None and hasattr(detector.lpgw, "distance_lambda"):
        detector.lpgw.distance_lambda = float(dl)
    print(f"lambda={config.LPGW_LAMBDA}  distance_lambda="
          f"{getattr(detector.lpgw, 'distance_lambda', config.LPGW_LAMBDA)}")

    # Downsample BEFORE reference selection so the reference has the same
    # point count as the targets, and matches what calibration measured.
    npts = config.LPGW_SEGMENT_POINTS
    ref_segs = [detector.downsample_segment(s, npts) for s in ref_segs_raw]
    print(f"Points per segment: {len(ref_segs[0])}")

    # Pin cost scale and reference index from CLEAN data, reuse everywhere.
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
            "transported).\nEvery lambda term in distance() is identically "
            "zero, so this would\nmeasure plain GW. Re-run calibrate_lambda.py."
        )

    # One fixed drift realisation, scaled per level.
    rng = np.random.default_rng(config.RANDOM_SEED)
    q_xyz = query_df[["PosX", "PosY", "PosZ"]].to_numpy(dtype=float)
    basis = make_drift_basis(q_xyz, rng)
    path_len = float(np.sum(np.linalg.norm(np.diff(q_xyz, axis=0), axis=1)))
    print(f"Query path length: {path_len:.1f} m")

    # Rebuild perturbed segments with the CLEAN rate: drift moves positions,
    # not the clock, so the rate must not be re-derived from perturbed data.
    def build_noisy(xyz):
        return segment_trajectory(
            downsample_trajectory(xyz, config.TARGET_POINTS),
            segment_length=config.SEGMENT_LENGTH, fps=fps_q,
            stride=config.STRIDE,
        )

    methods = [("LPGW", None), ("Hausdorff", compute_hausdorff_matrix)]
    if not skip_slow:
        methods += [("DTW", compute_dtw_matrix), ("Frechet", compute_frechet_matrix)]

    rows = []
    for frac in fractions:
        print("\n" + "=" * 74)
        print(f"DRIFT = {frac:.1%} of path length  ({frac * path_len:.2f} m)")
        print("=" * 74)

        offset = scale_drift(basis, q_xyz, frac)
        noisy_raw = build_noisy(q_xyz + offset)
        if len(noisy_raw) < n_q:
            raise RuntimeError(
                f"perturbed query gave {len(noisy_raw)} segments, expected "
                f">= {n_q}"
            )
        noisy = [detector.downsample_segment(s, npts) for s in noisy_raw[:n_q]]

        q_segs = [noisy[i] for i in q_idx]
        r_segs = list(ref_segs)
        if center:
            q_segs = zero_center_segments(q_segs)
            r_segs = zero_center_segments(r_segs)

        for name, fn in methods:
            t0 = time.perf_counter()
            if name == "LPGW":
                detector.reference_index = ref_idx
                D = detector.compute_distance_matrix(q_segs, r_segs)

                if frac == fractions[0]:  # Run only on the first iteration
                    from scipy.stats import spearmanr
                    D_new = D.copy()
            else:
                D = fn(q_segs, r_segs)
            elapsed = time.perf_counter() - t0

            m = retrieval_metrics(
                D, y_true, gt_ref,
                getattr(config, "RECALL_AT_1_INDEX_TOL", 2),
            )
            m.update({
                "drift_fraction": frac,
                "drift_metres": frac * path_len,
                "method": name,
                "runtime_sec": elapsed,
                "num_query_segments": len(q_segs),
                "num_reference_segments": len(r_segs),
                "centered": center,
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
    out_path = out_dir / (
        f"drift_experiment_{config.DATASET_SHORT}"
        f"{'_centered' if center else ''}.csv"
    )
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 74)
    print("SUMMARY  (max F1 by drift level)")
    print("=" * 74)
    print(df.pivot_table(index="drift_fraction", columns="method",
                         values="max_f1").to_string())
    print("\n(AP by drift level)")
    print(df.pivot_table(index="drift_fraction", columns="method",
                         values="average_precision").to_string())
    print("\n(Recall@1 by drift level)")
    print(df.pivot_table(index="drift_fraction", columns="method",
                         values="recall_at_1").to_string())

    b = float(y_true.mean())
    print(f"\nChance levels: all-positive F1 = {2*b/(1+b):.4f}, "
          f"random AP = {b:.4f}")
    print(f"Saved to: {out_path}")
    print("\nWHAT TO LOOK FOR: LPGW roughly flat while the coordinate-based")
    print("baselines fall. If LPGW also falls, its invariance is not")
    print("surviving the barycentric projection -- investigate K.")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", action="store_true",
                    help="Zero-center segments (ablation; removes the drift signal)")
    ap.add_argument("--fractions", type=float, nargs="+", default=None,
                    help="Drift levels as fractions of path length")
    ap.add_argument("--skip-slow", action="store_true",
                    help="Skip DTW and Frechet (keep LPGW + Hausdorff)")
    a = ap.parse_args()
    run_drift_experiment(center=a.center, fractions=a.fractions,
                         skip_slow=a.skip_slow)
