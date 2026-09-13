# experiments/make_heatmap_figures.py
"""
LPGW distance-matrix heatmaps and precision-recall curves, for both datasets.

Computes the matrices from scratch (no cached .npy needed), caches them for
reuse, and renders:

    figures/{short}/fig_heatmap.png     LPGW discrepancy matrix
    figures/{short}/fig_pr_curve.png    PR curve with the operating point
    results/D_lpgw_{short}.npy          cached matrix

WHY THE HEATMAP MATTERS BEYOND DECORATION
=========================================
The title of each heatmap reports two diagnostics:

  additive variance  -- how much of the matrix is explained by row + column
                        effects alone. A distance dominated by per-segment
                        offsets has argmin collapsing onto the same column
                        for every row, which looks like a working method on
                        threshold-based metrics but retrieves nothing.
                        Above ~0.9 is degenerate.

  unique argmins     -- how many distinct references are ever retrieved.

These are the numbers that distinguish real structure from a matrix with
none, so they belong on the figure rather than buried in text.

PER-DATASET PARAMETERS
======================
cost_scale, lambda and tolerance differ per dataset and must not be
crossed. They are listed in PARAMS below -- edit them to match config.py
if you re-calibrate.

USAGE
=====
    python experiments/make_heatmap_figures.py            # current config
    python experiments/make_heatmap_figures.py --both     # uzhfpv + kitti00
    python experiments/make_heatmap_figures.py --dataset kitti
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import config
from core.trajectory_utils import load_trajectory, prepare_segments
from loop_closure import LoopClosureDetector

# Edit these to match config.py after any re-calibration.
PARAMS = {
    "uzh_fpv": {
        "short": "uzhfpv", "nice": "UZH-FPV",
        "ref_csv": "poses_bag-3.csv", "query_csv": "poses_bag-7.csv",
        "tolerance": 1.0, "cost_scale": 56.53432887902825,
        "lambdaa": 1.0, "distance_lambda": 0.2,
    },
    "kitti": {
        "short": "kitti00", "nice": "KITTI 00",
        "ref_csv": "kitti_00_reference.csv", "query_csv": "kitti_00_query.csv",
        "tolerance": 5.0, "cost_scale": 97.9785,
        "lambdaa": 1.0, "distance_lambda": 0.2,
    },
}

SINGLE = 3.5
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def compute_matrix(p):
    """LPGW discrepancy matrix plus the labels that go with it."""
    poses = repo_root / config.POSES_DIR

    print(f"\n{'=' * 70}\n{p['nice']}\n{'=' * 70}")
    ref_df = load_trajectory(poses / p["ref_csv"])
    q_df = load_trajectory(poses / p["query_csv"])

    print("Reference trajectory:")
    ref_raw, fps_ref = prepare_segments(
        ref_df, config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE)
    print("Query trajectory:")
    q_raw, _ = prepare_segments(
        q_df, config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE)

    n_ref, n_q = len(ref_raw), len(q_raw)
    print(f"Reference segments: {n_ref}   Query segments: {n_q}")

    gt_path = (repo_root / "ground_truth" / "files"
               / f"gt_{p['short']}_{p['tolerance']}m.csv")
    if not gt_path.exists():
        raise FileNotFoundError(
            f"{gt_path}\nSet ACTIVE_DATASET and SPATIAL_TOLERANCE to match, "
            "then run ground_truth/generate_ground_truth.py"
        )
    gt = pd.read_csv(gt_path).sort_values("query_index").reset_index(drop=True)
    if len(gt) != n_q:
        raise ValueError(
            f"GT has {len(gt)} rows but segmentation gives {n_q} queries. "
            "Regenerate ground truth with the current settings."
        )

    max_q = getattr(config, "MAX_BASELINE_QUERIES",
                    getattr(config, "MAX_BASELINE_SEGMENTS", 150))
    q_idx = (np.arange(n_q) if n_q <= max_q
             else np.linspace(0, n_q - 1, max_q, dtype=int))
    y_true = gt["label"].to_numpy(bool)[q_idx]
    print(f"Queries used: {len(q_idx)}   positives: {y_true.sum()} "
          f"({y_true.mean():.1%})")

    detector = LoopClosureDetector(
        segment_length=config.SEGMENT_LENGTH, stride=config.STRIDE, fps=fps_ref,
        lambdaa=p["lambdaa"], downsample_points=None,
        reference_strategy=getattr(config, "LPGW_REFERENCE_STRATEGY",
                                   "median_diameter"),
        cost_mode=getattr(config, "LPGW_COST_MODE", "squared"),
        huber_delta_frac=getattr(config, "LPGW_HUBER_DELTA_FRAC", 0.1),
    )
    if hasattr(detector.lpgw, "distance_lambda"):
        detector.lpgw.distance_lambda = float(p["distance_lambda"])

    npts = config.LPGW_SEGMENT_POINTS
    ref_segs = [detector.downsample_segment(s, npts) for s in ref_raw]
    q_segs = [detector.downsample_segment(q_raw[i], npts) for i in q_idx]

    detector.lpgw.cost_scale = float(p["cost_scale"])
    detector.cost_scale = float(p["cost_scale"])
    print(f"lambda={p['lambdaa']}  distance_lambda={p['distance_lambda']}  "
          f"cost_scale={p['cost_scale']:.4f} (pinned)")

    D = detector.compute_distance_matrix(q_segs, ref_segs)

    out = repo_root / "results" / f"D_lpgw_{p['short']}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, D)
    print(f"Saved matrix: {out.relative_to(repo_root)}  shape {D.shape}")
    return D, y_true


def diagnostics(D):
    row, col = D.mean(1, keepdims=True), D.mean(0, keepdims=True)
    resid = D - row - col + D.mean()
    add = float(1 - resid.var() / D.var()) if D.var() > 0 else 1.0
    return add, int(len(np.unique(D.argmin(axis=1))))


def fig_heatmap(D, p, outdir):
    add, uniq = diagnostics(D)

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.92))
    im = ax.imshow(D, aspect="auto", origin="lower", cmap="viridis",
                   interpolation="nearest")
    ax.set_xlabel("Reference segment index")
    ax.set_ylabel("Query segment index")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("LPGW discrepancy", fontsize=7)
    ax.set_title(f"{p['nice']}  —  additive variance {add:.2f}, "
                 f"{uniq}/{D.shape[0]} unique argmins", fontsize=7.5)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "fig_heatmap.png")
    plt.close(fig)
    print(f"  wrote {(outdir / 'fig_heatmap.png').relative_to(repo_root)}")
    print(f"  additive variance {add:.3f}  |  unique argmins {uniq}/{D.shape[0]}")
    if add > 0.9:
        print("    >> above 0.9: matrix is dominated by per-segment offsets")


def fig_pr(D, y, p, outdir):
    if not y.any() or y.all():
        print("  skip PR curve: labels are single-class")
        return

    s = -np.min(D, axis=1)
    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    b = int(np.argmax(f1))
    base = float(y.mean())

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.85))
    ax.plot(rec, prec, lw=1.6, color="#d62728",
            label=f"LPGW (AP = {ap:.3f})")
    ax.axhline(base, ls="--", lw=1.1, color="0.35",
               label=f"random (AP = {base:.3f})")
    ax.scatter([rec[b]], [prec[b]], s=34, color="k", zorder=5,
               label=f"max F1 = {f1[b]:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title(p["nice"], fontsize=8)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "fig_pr_curve.png")
    plt.close(fig)
    print(f"  wrote {(outdir / 'fig_pr_curve.png').relative_to(repo_root)}")
    print(f"  AP {ap:.4f} vs random {base:.4f}  ({ap - base:+.4f})")


def run(key):
    p = PARAMS[key]
    D, y = compute_matrix(p)
    outdir = repo_root / "figures" / p["short"]
    fig_heatmap(D, p, outdir)
    fig_pr(D, y, p, outdir)


def main(dataset=None, both=False):
    if both:
        keys = list(PARAMS)
    elif dataset:
        keys = [dataset]
    else:
        keys = [config.ACTIVE_DATASET]

    for k in keys:
        if k not in PARAMS:
            raise SystemExit(f"Unknown dataset {k!r}; expected one of "
                             f"{list(PARAMS)}")
        try:
            run(k)
        except Exception as exc:
            print(f"\n  FAILED for {k}: {type(exc).__name__}: {exc}")
            if both:
                print("  (continuing with the other dataset)")
            else:
                raise

    print("\nLaTeX:")
    for k in keys:
        s = PARAMS[k]["short"]
        for n in ("fig_heatmap", "fig_pr_curve"):
            if (repo_root / "figures" / s / f"{n}.png").exists():
                print(rf"  \includegraphics[width=\linewidth]{{{s}/{n}.png}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(PARAMS), default=None)
    ap.add_argument("--both", action="store_true")
    a = ap.parse_args()
    main(dataset=a.dataset, both=a.both)
