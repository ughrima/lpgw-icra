"""
STEP 1. Run before any experiment, and again after any change to
segmentation (SEGMENT_LENGTH, STRIDE, TARGET_POINTS, sampling rate).

WHAT IT DOES
============
Sweeps lambda and reports how much mass PGW actually transports at each
value. Recommends the largest lambda at which mass is genuinely discarded,
and prints the two lines to paste into config.py.

WHY IT MATTERS
==============
In the PGW objective, discarding all mass costs
lambda*(|mu|^2 + |nu|^2) = 2*lambda, while transporting everything costs 0.
If lambda is large relative to the cost matrix, dropping even a sliver
costs more than any possible geometric gain, so the solver always
transports FULL mass. Then, for every segment:

    transported_mass == 1.0
    q_e              == p == uniform, IDENTICAL across segments
    penalty_1        == lambda*(1 + 1 - 2*1) == 0
    gamma_c_abs      == 0  ->  penalty_2 == 0

Every partial-transport term is identically zero and the method silently
degenerates to plain linear GW. The check is one number: if
transported_mass is 1.0, nothing downstream means anything.

cost_scale is the median squared pairwise distance inside a segment, so it
moves with segmentation. A lambda calibrated on 0.55 s windows does not
transfer to 5 s windows.

USAGE
=====
    python experiments/calibrate_lambda.py
    python experiments/calibrate_lambda.py --lambdas 1000 500 200 100 50 20
    python experiments/calibrate_lambda.py --n-probe 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np

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
from lpgw import LPGW
from loop_closure import LoopClosureDetector


DEFAULT_LAMBDAS = [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.1, 0.01]

# Target band for mean transported mass. Near 1.0 means partial transport
# is inactive; very low means segments are being shredded, not partially
# matched.
TARGET_LO, TARGET_HI = 0.70, 0.98


def main(lambdas=None, n_probe=12):
    lambdas = sorted(DEFAULT_LAMBDAS if lambdas is None else lambdas, reverse=True)

    print("=" * 96)
    print("LAMBDA CALIBRATION")
    print("=" * 96)
    print(f"Dataset         : {config.ACTIVE_DATASET}")
    print(f"Cost mode       : {config.LPGW_COST_MODE}")
    print(f"Segment length  : {config.SEGMENT_LENGTH} s   stride: {config.STRIDE} s")
    print(f"Spatial tol     : {config.SPATIAL_TOLERANCE} m")
    print(f"LPGW points/seg : {config.LPGW_SEGMENT_POINTS}")

    poses = repo_root / config.POSES_DIR

    print("\nReference trajectory:")
    ref_segs, fps_ref = prepare_segments(
        load_trajectory(poses / config.BAG3_CSV),
        config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE,
    )
    print("Query trajectory:")
    q_segs, fps_q = prepare_segments(
        load_trajectory(poses / config.BAG7_CSV),
        config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE,
    )

    # NO min_len truncation. The reference is a DATABASE, the query a set of
    # PROBES; nothing requires the counts to match, and truncating to the
    # shorter one silently discarded the tail of the longer trajectory.
    print(f"\nReference segments: {len(ref_segs)}   Query segments: {len(q_segs)}")

    det = LoopClosureDetector(
        segment_length=config.SEGMENT_LENGTH, stride=config.STRIDE, fps=fps_ref,
        lambdaa=1.0,
        downsample_points=config.LPGW_SEGMENT_POINTS,
        reference_strategy=config.LPGW_REFERENCE_STRATEGY,
    )

    # Downsample to the points-per-segment LPGW will actually use, so the
    # calibration matches the embedding. PGW is O(n^2) in points.
    proc_ref = [det.downsample_segment(s, config.LPGW_SEGMENT_POINTS) for s in ref_segs]
    proc_q = [det.downsample_segment(s, config.LPGW_SEGMENT_POINTS) for s in q_segs]
    print(f"Points per segment after downsampling: {len(proc_ref[0])}")

    probe = LPGW(cost_mode=config.LPGW_COST_MODE,
                 huber_delta_frac=config.LPGW_HUBER_DELTA_FRAC)
    cost_scale = probe.calibrate(proc_ref + proc_q)
    print(f"\nCalibrated cost_scale: {cost_scale:.6g}")
    if config.LPGW_COST_MODE == "squared":
        print(f"  (median pairwise distance within a segment: "
              f"{np.sqrt(cost_scale):.2f} m)")

    ref_seg, ref_idx = det.select_reference(proc_ref, config.LPGW_REFERENCE_STRATEGY)
    print(f"Reference segment index: {ref_idx}")

    segs = proc_q[:n_probe]
    print(f"\nProbing {len(segs)} query segments per lambda...\n")
    print(f"{'lambda':>9} {'mass mean':>10} {'mass min':>10} {'gamma_c':>11} "
          f"{'d(Y,Y)':>10} {'med dist':>11} {'geo share':>10} {'uniq':>9}  status")
    print("-" * 96)

    results = []
    for lam in lambdas:
        lpgw = LPGW(lambdaa=lam, cost_mode=config.LPGW_COST_MODE,
                    huber_delta_frac=config.LPGW_HUBER_DELTA_FRAC,
                    cost_scale=cost_scale)
        try:
            embs = [lpgw.embed(ref_seg, s) for s in segs]
        except Exception as exc:
            print(f"{lam:9.4g} {'--':>10} {'--':>10} {'--':>11} {'--':>10} "
                  f"{'--':>11} {'--':>10} {'--':>9}  SOLVER ERROR: "
                  f"{type(exc).__name__}: {exc}")
            continue

        mass = np.array([e["transported_mass"] for e in embs])
        gc = np.array([e["gamma_c_abs"] for e in embs])
        self_d = lpgw.distance(embs[0], embs[0])

        # Split distance into geometry vs mass penalties. With a large
        # lambda the mass terms can swamp the geometry, which reintroduces
        # the degenerate row+column structure from the other direction.
        geo, pen = [], []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                a, b = embs[i], embs[j]
                q12 = np.minimum(a["q_e"], b["q_e"])
                t = float(q12.sum())
                geo.append(float(q12 @ ((a["K"] - b["K"]) ** 2) @ q12)
                           / max(t * t, 1e-30))
                pen.append(lam * (float(a["q_e"].sum()) ** 2
                                  + float(b["q_e"].sum()) ** 2 - 2 * t ** 2)
                           + lam * abs(a["gamma_c_abs"] - b["gamma_c_abs"]))
        mg, mp = float(np.median(geo)), float(np.median(pen))
        geo_share = mg / max(mg + mp, 1e-30)

        D = np.array([[lpgw.distance(a, b) for b in embs] for a in embs])
        off = D[~np.eye(len(embs), dtype=bool)]
        med = float(np.median(off)) if off.size else 0.0
        uniq = len(np.unique(D.argmin(axis=1)))

        if mass.min() > 0.999:
            status = "INACTIVE (lambda too high)"
        elif self_d > 1e-10:
            status = "d(Y,Y) != 0  BUG"
        elif mass.mean() < 0.3:
            status = "over-discarding (lambda too low)"
        elif uniq <= 2:
            status = "degenerate argmin"
        elif geo_share < 0.2:
            status = "mass terms dominate geometry"
        elif TARGET_LO <= mass.mean() <= TARGET_HI:
            status = "GOOD"
        else:
            status = "usable"

        print(f"{lam:9.4g} {mass.mean():10.5f} {mass.min():10.5f} {gc.mean():11.3e} "
              f"{self_d:10.2e} {med:11.3e} {geo_share:9.1%} "
              f"{uniq:4d}/{len(embs):<4d}  {status}")

        results.append({"lambda": lam, "mass_mean": float(mass.mean()),
                        "mass_min": float(mass.min()), "geo_share": geo_share,
                        "status": status, "uniq": uniq})

    print("-" * 96)
    print()

    good = [r for r in results if r["status"] == "GOOD"]
    usable = [r for r in results if r["status"] == "usable"]
    pick = good or usable

    if not pick:
        print("NO USABLE LAMBDA FOUND.")
        if results and all(r["mass_min"] > 0.999 for r in results):
            lo = min(r["lambda"] for r in results)
            print("  Every lambda transported full mass. Sweep lower:")
            print(f"    --lambdas {lo/10:g} {lo/100:g} {lo/1000:g}")
        elif results and all(r["mass_mean"] < 0.3 for r in results):
            hi = max(r["lambda"] for r in results)
            print("  Everything over-discarded. Sweep higher:")
            print(f"    --lambdas {hi*10:g} {hi*100:g}")
        elif any(r["geo_share"] < 0.2 for r in results):
            print("  Mass penalties dominate the geometric term at every")
            print("  workable lambda. The solver needs a large lambda but")
            print("  distance() needs a small one. These can be decoupled --")
            print("  ask for the lambdaa / distance_lambda split.")
        return results

    best = max(pick, key=lambda r: r["lambda"])
    print(f"RECOMMENDED:  LPGW_LAMBDA = {best['lambda']}")
    print(f"  mean transported mass {best['mass_mean']:.4f}  "
          f"-> partial transport IS active")
    print(f"  geometric share {best['geo_share']:.1%}  -> "
          f"{'geometry dominates, good' if best['geo_share'] > 0.5 else 'watch this'}")
    print()
    print("Paste into config.py:")
    print(f"    LPGW_LAMBDA     = {best['lambda']}")
    print(f"    LPGW_COST_SCALE = {cost_scale!r}")
    print()
    print("Pinning cost_scale matters: drift and overlap runs must share one")
    print("normalisation, or the sweep measures a shifting baseline instead")
    print("of the perturbation.")

    if best["lambda"] == max(r["lambda"] for r in results):
        print()
        print("NOTE: the winner sits at the TOP of the sweep, so it may be a")
        print("boundary rather than an optimum. Check above it:")
        print(f"    --lambdas {best['lambda']*10:g} {best['lambda']*5:g} "
              f"{best['lambda']*2:g} {best['lambda']:g}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+", default=None)
    ap.add_argument("--n-probe", type=int, default=12)
    a = ap.parse_args()
    main(lambdas=a.lambdas, n_probe=a.n_probe)