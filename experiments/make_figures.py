"""
Generate paper figures from the drift, overlap, ablation, bootstrap, and .npy files.

SOURCES
=======
    results/drift_experiment_{short}.csv
    results/overlap_experiment_{short}.csv
    results/reference_ablation_{short}.csv
    results/bootstrap_ci_{short}.csv
    results/D_lpgw_{short}.npy

OUTPUT  ->  figures/{short}/
    fig1_trajectories.png   top-down reference vs query overlay
    fig2_drift.png          AP / max-F1 / Recall@1 vs drift
    fig3_overlap.png        AP / max-F1 / Recall@1 vs contiguous retention
    fig4_method_bars.png    method comparison at zero perturbation
    fig5_summary.png        drift and overlap AP side by side, both datasets
    fig6_ablation.png       performance spread across reference indices
    fig7_bootstrap.png      point estimates with 95% block-bootstrap CIs
    fig8_heatmap.png        LPGW pairwise distance matrix heatmap

USAGE
=====
    python experiments/make_figures.py                 # current dataset
    python experiments/make_figures.py --dataset kitti00
    python experiments/make_figures.py --all           # both, plus fig5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import config

RESULTS = repo_root / "results"

COLORS = {"LPGW": "#d62728", "Hausdorff": "#1f77b4",
          "DTW": "#2ca02c", "Frechet": "#ff7f0e"}
MARKERS = {"LPGW": "o", "Hausdorff": "s", "DTW": "^", "Frechet": "D"}
ORDER = ["LPGW", "Hausdorff", "DTW", "Frechet"]

SINGLE, DOUBLE = 3.5, 7.16           # IEEE column widths, inches
NICE = {"uzhfpv": "UZH-FPV", "kitti00": "KITTI 00"}

# Prettier publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _order(ms):
    known = [m for m in ORDER if m in ms]
    return known + sorted(set(ms) - set(known))


def _chance(df):
    """Chance levels straight from the CSV, so they match the run."""
    b = float(df["base_rate"].iloc[0]) if "base_rate" in df.columns else np.nan
    ap = 2 * b / (1 + b) if b == b else np.nan
    return b, ap


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(repo_root)}")


def _load(short, kind):
    p = RESULTS / f"{kind}_experiment_{short}.csv"
    if not p.exists():
        print(f"  skip: {p.name} not found")
        return None
    df = pd.read_csv(p)
    if "max_f1" not in df.columns and kind in ["drift", "overlap"]:
        print(f"  skip: {p.name} predates threshold-free metrics; re-run")
        return None
    return df


# ----------------------------------------------------------------------
# Fig 1: trajectories
# ----------------------------------------------------------------------

def fig_trajectories(short, outdir):
    from core.trajectory_utils import load_trajectory
    poses = repo_root / config.POSES_DIR
    try:
        ref = load_trajectory(poses / config.BAG3_CSV)[
            ["PosX", "PosY", "PosZ"]].to_numpy()
        qry = load_trajectory(poses / config.BAG7_CSV)[
            ["PosX", "PosY", "PosZ"]].to_numpy()
    except Exception as exc:
        print(f"  skip fig1: {exc}")
        return

    if short.startswith("kitti"):
        rx, ry, qx, qy = ref[:, 0], ref[:, 2], qry[:, 0], qry[:, 2]
        xlab, ylab = "X (m)", "Z (m)"
    else:
        rx, ry, qx, qy = ref[:, 0], ref[:, 1], qry[:, 0], qry[:, 1]
        xlab, ylab = "X (m)", "Y (m)"

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.85))
    ax.plot(rx, ry, lw=1.0, color="#1f77b4", alpha=0.9, label="Reference")
    ax.plot(qx, qy, lw=1.0, color="#d62728", alpha=0.9, label="Query")
    ax.scatter([rx[0]], [ry[0]], s=25, color="#1f77b4", zorder=5, edgecolors="k", linewidths=0.5)
    ax.scatter([qx[0]], [qy[0]], s=25, color="#d62728", zorder=5, edgecolors="k", linewidths=0.5)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", framealpha=0.9)
    _save(fig, outdir / "fig1_trajectories.png")


# ----------------------------------------------------------------------
# Fig 2 / 3: perturbation sweeps
# ----------------------------------------------------------------------

def _sweep(df, xcol, xlabel, outpath, invert=False, title=None):
    base, allpos = _chance(df)
    methods = _order(df["method"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.3))
    panels = [
        ("average_precision", "Average precision", base, "random"),
        ("max_f1", "Max F1", allpos, "all-positive"),
        ("recall_at_1", "Recall@1", None, None),
    ]
    for ax, (col, lab, chance, cname) in zip(axes, panels):
        for m in methods:
            sub = df[df.method == m].sort_values(xcol)
            ax.plot(sub[xcol], sub[col], marker=MARKERS.get(m, "o"), ms=3.5,
                    lw=1.5, color=COLORS.get(m), label=m)
        if chance is not None and chance == chance:
            ax.axhline(chance, ls="--", lw=1.0, color="0.4",
                       label=f"{cname} ({chance:.2f})")
        ax.set_xlabel(xlabel); ax.set_ylabel(lab)
        ax.set_ylim(-0.02, 1.05)
        if invert:
            ax.invert_xaxis()
    axes[0].legend(loc="lower left", framealpha=0.9)
    if title:
        fig.suptitle(title, y=1.03, fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, outpath)


def fig_drift(short, outdir):
    df = _load(short, "drift")
    if df is None:
        return
    df = df.copy()
    df["drift_pct"] = df["drift_fraction"] * 100
    _sweep(df, "drift_pct", "Drift (% of path length)",
           outdir / "fig2_drift.png", title=NICE.get(short, short))


def fig_overlap(short, outdir):
    df = _load(short, "overlap")
    if df is None:
        return
    df = df.copy()
    df["retention_pct"] = df["overlap_ratio"] * 100
    _sweep(df, "retention_pct", "Contiguous retention (%)",
           outdir / "fig3_overlap.png", invert=True,
           title=NICE.get(short, short))


# ----------------------------------------------------------------------
# Fig 4: method comparison at zero perturbation
# ----------------------------------------------------------------------

def fig_method_bars(short, outdir):
    df = _load(short, "drift")
    if df is None:
        return
    df = df[df.drift_fraction == df.drift_fraction.min()]
    base, allpos = _chance(df)
    methods = _order(df["method"].unique())
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.2))
    panels = [
        ("average_precision", "Average precision", base, "random"),
        ("max_f1", "Max F1", allpos, "all-positive"),
        ("recall_at_1", "Recall@1", None, None),
    ]
    for ax, (col, lab, chance, cname) in zip(axes, panels):
        vals = [float(df[df.method == m][col].iloc[0]) for m in methods]
        bars = ax.bar(x, vals, width=0.55,
                      color=[COLORS.get(m, "0.6") for m in methods], edgecolor="k", linewidth=0.5)
        if chance is not None and chance == chance:
            ax.axhline(chance, ls="--", lw=1.0, color="0.4",
                       label=f"{cname} ({chance:.2f})")
            ax.legend(loc="upper right", framealpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel(lab); ax.set_ylim(0, 1.12)
        for xi, v in zip(x, vals):
            if v == v:
                ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=6, fontweight="bold")
    fig.suptitle(f"{NICE.get(short, short)} — Method Comparison",
                 y=1.03, fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir / "fig4_method_bars.png")


# ----------------------------------------------------------------------
# Fig 5: summary across datasets
# ----------------------------------------------------------------------

def fig_summary(shorts, outdir):
    panels = []
    for s in shorts:
        d, o = _load(s, "drift"), _load(s, "overlap")
        if d is not None:
            panels.append((f"{NICE.get(s, s)}: drift", d, "drift_fraction",
                           "Drift (% of path)", 100, False))
        if o is not None:
            panels.append((f"{NICE.get(s, s)}: overlap", o, "overlap_ratio",
                           "Retention (%)", 100, True))
    if not panels:
        print("  skip fig5: no CSVs")
        return

    ncol = min(len(panels), 4)
    fig, axes = plt.subplots(1, ncol, figsize=(DOUBLE, 2.2), squeeze=False)
    for ax, (title, df, xcol, xlab, mult, inv) in zip(axes[0], panels):
        base, _ = _chance(df)
        for m in _order(df["method"].unique()):
            sub = df[df.method == m].sort_values(xcol)
            ax.plot(sub[xcol] * mult, sub["average_precision"],
                    marker=MARKERS.get(m, "o"), ms=3, lw=1.2,
                    color=COLORS.get(m), label=m)
        if base == base:
            ax.axhline(base, ls="--", lw=1.0, color="0.4")
        ax.set_title(title, fontsize=7.5, fontweight="bold")
        ax.set_xlabel(xlab); ax.set_ylim(-0.02, 1.05)
        if inv:
            ax.invert_xaxis()
    axes[0][0].set_ylabel("Average precision")
    axes[0][0].legend(loc="lower left", fontsize=6, framealpha=0.9)
    fig.tight_layout()
    _save(fig, outdir / "fig5_summary.png")


# ----------------------------------------------------------------------
# Fig 6: Reference Ablation
# ----------------------------------------------------------------------

def fig_ablation(short, outdir):
    p = RESULTS / f"reference_ablation_{short}.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.2))
    metrics = [("max_f1", "Max F1"), ("average_precision", "Average precision"), ("recall_at_1", "Recall@1")]
    for ax, (col, lab) in zip(axes, metrics):
        ax.plot(df["reference_fraction"] * 100, df[col], marker="o", ms=4, lw=1.3, color="#7f7f7f")
        # Highlight default
        default_row = df[df["is_default"] == True]
        if not default_row.empty:
            ax.scatter(default_row["reference_fraction"] * 100, default_row[col],
                       color="#d62728", s=40, zorder=5, label="Default")
        ax.set_xlabel("Reference Index Fraction (%)")
        ax.set_ylabel(lab)
        ax.set_ylim(-0.02, 1.05)
    axes[1].legend(loc="best", framealpha=0.9)
    fig.suptitle(f"{NICE.get(short, short)} — Reference Segment Ablation", y=1.03, fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir / "fig6_ablation.png")


# ----------------------------------------------------------------------
# Fig 7: Bootstrap CIs
# ----------------------------------------------------------------------

def fig_bootstrap(short, outdir):
    p = RESULTS / f"bootstrap_ci_{short}.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    # Filter for main point metrics
    df = df[df["metric"].isin(["max_f1", "average_precision", "recall_at_1"])]
    methods = [m for m in ORDER if m in df["method"].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.2))
    metrics = [("max_f1", "Max F1"), ("average_precision", "Average precision"), ("recall_at_1", "Recall@1")]
    
    for ax, (col, lab) in zip(axes, metrics):
        sub = df[df["metric"] == col]
        y_pos = np.arange(len(methods))
        points, err_low, err_high = [], [], []
        for m in methods:
            row = sub[sub["method"] == m]
            if not row.empty:
                pt = row["point"].values[0]
                lo = row["ci_low"].values[0]
                hi = row["ci_high"].values[0]
                points.append(pt)
                err_low.append(pt - lo)
                err_high.append(hi - pt)
            else:
                points.append(0); err_low.append(0); err_high.append(0)
        
        ax.errorbar(points, y_pos, xerr=[err_low, err_high], fmt="o", color="#1f77b4",
                    ecolor="#aec7e8", elinewidth=2, capsize=3, ms=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods)
        ax.set_xlabel(lab)
        ax.set_xlim(-0.05, 1.05)
    fig.suptitle(f"{NICE.get(short, short)} — 95% Block-Bootstrap CIs", y=1.03, fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir / "fig7_bootstrap.png")


# ----------------------------------------------------------------------
# Fig 8: Distance Matrix Heatmap
# ----------------------------------------------------------------------

def fig_heatmap(short, outdir):
    p = RESULTS / f"D_lpgw_{short}.npy"
    if not p.exists():
        return
    D = np.load(p)

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.9))
    im = ax.imshow(D, aspect="auto", cmap="viridis_r")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("LPGW Distance")
    ax.set_xlabel("Reference Segment Index")
    ax.set_ylabel("Query Segment Index")
    ax.set_title(f"{NICE.get(short, short)} — LPGW Distance Matrix", fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir / "fig8_heatmap.png")


# ----------------------------------------------------------------------

def run_one(short):
    outdir = repo_root / "figures" / short
    print(f"\n--- {NICE.get(short, short)}  ->  figures/{short}/")
    fig_trajectories(short, outdir)
    fig_drift(short, outdir)
    fig_overlap(short, outdir)
    fig_method_bars(short, outdir)
    fig_ablation(short, outdir)
    fig_bootstrap(short, outdir)
    fig_heatmap(short, outdir)


def main(dataset=None, do_all=False):
    print("=" * 74)
    print("GENERATING POLISHED PUBLICATION FIGURES")
    print("=" * 74)

    if do_all:
        shorts = [s for s in ("uzhfpv", "kitti00")
                  if (RESULTS / f"drift_experiment_{s}.csv").exists()
                  or (RESULTS / f"overlap_experiment_{s}.csv").exists()]
        if not shorts:
            print("No experiment CSVs found in results/")
            return
        for s in shorts:
            run_one(s)
        print("\n--- combined summary  ->  figures/")
        fig_summary(shorts, repo_root / "figures")
    else:
        run_one(dataset or config.DATASET_SHORT)

    print("\nLaTeX snippets:")
    for p in sorted((repo_root / "figures").rglob("fig*.png")):
        rel = p.relative_to(repo_root / "figures")
        print(rf"  \includegraphics[width=\linewidth]{{{rel.as_posix()}}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="uzhfpv or kitti00 (defaults to config.DATASET_SHORT)")
    ap.add_argument("--all", action="store_true",
                    help="Both datasets plus the combined summary figure")
    a = ap.parse_args()
    main(dataset=a.dataset, do_all=a.all)