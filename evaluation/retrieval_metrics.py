# evaluation/retrieval_metrics.py
"""
Shared scoring for every experiment.

WHY THIS EXISTS
===============
Each experiment previously scored its own way, so numbers were not
comparable across tables. Worse, all of them used a percentile threshold
taken from config -- and that percentile was tuned to UZH-FPV's base rate
(68% of queries are loops). Applied to KITTI, whose base rate is ~15%, it
flags 68% of queries regardless of how good the ranking is, which pins
precision near the base rate.

The numbers that actually matter are threshold-free:

    max_f1       best achievable F1 over all thresholds
    average_precision (AP)
    recall_at_100_precision   how much recall before the first mistake
    recall_at_1  does argmin land on the right reference?

TRIVIAL BASELINES ARE REPORTED ALONGSIDE, ALWAYS
================================================
A random ranker scores AP ~= base rate. An all-positive classifier scores
F1 = 2b/(1+b). If a method does not clearly beat both, it has no signal,
and a reviewer will compute these in thirty seconds. Report them first.

Reference: on a synthetic matrix with pure additive row+column structure
(the signature of a distance function whose d(Y,Y) != 0), these metrics
return max_f1 ~= all_positive_f1, AP ~= base rate, R@100%P = 0 and
R@1 ~= chance. If your real results look like that, the matrix is
degenerate, not merely weak.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def scores_from_matrix(D: np.ndarray) -> np.ndarray:
    """Per-query loop-likelihood: higher = more loop-like."""
    return -np.min(np.asarray(D, dtype=float), axis=1)


def trivial_baselines(y_true) -> dict:
    """What a random ranker and an all-positive classifier would score."""
    y = np.asarray(y_true, dtype=bool)
    b = float(y.mean())
    return {
        "base_rate": b,
        "random_ap": b,
        "all_positive_f1": (2 * b / (1 + b)) if b > 0 else 0.0,
    }


def recall_at_1(D, y_true, gt_nearest_ref, index_tol: int = 2) -> float:
    """
    For queries that ARE genuine loops, does the nearest retrieved
    reference land within `index_tol` segments of the true one?

    This separates retrieval quality from threshold choice and is the most
    diagnostic single number available. Chance level is roughly
    (2*index_tol + 1) / n_reference.
    """
    y = np.asarray(y_true, dtype=bool)
    pos = np.flatnonzero(y)
    if pos.size == 0:
        return float("nan")
    hits = sum(
        abs(int(np.argmin(D[i])) - int(gt_nearest_ref[i])) <= index_tol
        for i in pos
    )
    return hits / pos.size


def evaluate_matrix(
    D,
    y_true,
    gt_nearest_ref=None,
    index_tol: int = 2,
    percentile: float | None = None,
) -> dict:
    """
    Score a query x reference distance matrix.

    `percentile`, if given, adds the thresholded operating point as
    SECONDARY output. The threshold is taken over per-query best-match
    scores (not over all matrix entries -- row minima live far out in the
    left tail of the all-pairs distribution, so a percentile of the full
    matrix means something different for every method).
    """
    D = np.asarray(D, dtype=float)
    y = np.asarray(y_true, dtype=bool)

    if D.shape[0] != y.size:
        raise ValueError(f"D has {D.shape[0]} rows but y_true has {y.size}")
    if not np.all(np.isfinite(D)):
        raise ValueError("distance matrix contains NaN or inf")

    out = trivial_baselines(y)
    s = scores_from_matrix(D)
    out["n_queries"] = int(y.size)
    out["n_positives"] = int(y.sum())

    if y.any() and not y.all():
        prec, rec, _ = precision_recall_curve(y, s)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        b = int(np.argmax(f1))
        out["max_f1"] = float(f1[b])
        out["precision_at_max_f1"] = float(prec[b])
        out["recall_at_max_f1"] = float(rec[b])
        out["average_precision"] = float(average_precision_score(y, s))
        perfect = prec >= 0.999
        out["recall_at_100_precision"] = (
            float(rec[perfect].max()) if perfect.any() else 0.0
        )
    else:
        for k in ("max_f1", "precision_at_max_f1", "recall_at_max_f1",
                  "average_precision", "recall_at_100_precision"):
            out[k] = float("nan")

    out["recall_at_1"] = (
        recall_at_1(D, y, gt_nearest_ref, index_tol)
        if gt_nearest_ref is not None else float("nan")
    )

    # Headroom over the trivial baselines. Negative means no signal.
    out["f1_over_all_positive"] = out["max_f1"] - out["all_positive_f1"]
    out["ap_over_random"] = out["average_precision"] - out["random_ap"]

    if percentile is not None:
        best = np.min(D, axis=1)
        tau = float(np.percentile(best, percentile))
        pred = best <= tau
        tp = int(np.sum(y & pred))
        fp = int(np.sum(~y & pred))
        fn = int(np.sum(y & ~pred))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        out.update({
            "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-12),
            "tp": tp, "fp": fp, "fn": fn,
            "threshold_tau": tau,
            "num_predicted_positives": int(pred.sum()),
            "percentile": percentile,
        })
    return out


def block_bootstrap_ci(
    D, y_true, block_size: int = 10, n_resamples: int = 10000,
    metric: str = "max_f1", alpha: float = 0.05, seed: int = 42,
) -> tuple:
    """
    95% CI via BLOCK bootstrap over contiguous query blocks.

    Segments overlap in time (5 s windows at 1 s stride share 80% of their
    points), so neighbouring queries are strongly dependent. Resampling
    them independently understates uncertainty. Blocks of `block_size`
    consecutive queries keep the dependence intact.

    With stride 1 s and 5 s windows, block_size >= 5 is the minimum to
    break the overlap; 10 is a reasonable default.
    """
    D = np.asarray(D, dtype=float)
    y = np.asarray(y_true, dtype=bool)
    rng = np.random.default_rng(seed)

    n = y.size
    starts = np.arange(0, n, block_size)
    blocks = [np.arange(s, min(s + block_size, n)) for s in starts]
    n_needed = int(np.ceil(n / block_size))

    vals = []
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(blocks), size=n_needed)
        idx = np.concatenate([blocks[c] for c in chosen])[:n]
        yb = y[idx]
        if not yb.any() or yb.all():
            continue
        try:
            vals.append(evaluate_matrix(D[idx], yb)[metric])
        except Exception:
            continue

    if not vals:
        return float("nan"), float("nan"), float("nan")

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    point = evaluate_matrix(D, y)[metric]
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def format_row(name: str, m: dict) -> str:
    """One-line summary with the trivial baselines inline."""
    return (
        f"  {name:11s} maxF1={m['max_f1']:.4f} (allpos {m['all_positive_f1']:.4f}, "
        f"{m['f1_over_all_positive']:+.4f})  "
        f"AP={m['average_precision']:.4f} (rand {m['random_ap']:.4f}, "
        f"{m['ap_over_random']:+.4f})  "
        f"R@1={m['recall_at_1']:.4f}  R@100%P={m['recall_at_100_precision']:.4f}"
    )
