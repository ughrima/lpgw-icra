"""
SWEEP DISTANCE LAMBDA
=====================
Finds the distance_lambda where the geometric term accounts for 50-80% 
of the distance value, ensuring geometry leads while mass penalties still contribute.
"""

import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

submodule_path = repo_root / "Linearized_Partial_Gromov_Wasserstein"
if submodule_path.exists():
    sys.path.insert(0, str(submodule_path))

import config
from core.trajectory_utils import load_trajectory, prepare_segments
from lpgw import LPGW
from loop_closure import LoopClosureDetector

def main():
    lambdas_d = [2.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0]
    
    poses = repo_root / config.POSES_DIR
    ref_segs, fps_ref = prepare_segments(
        load_trajectory(poses / config.BAG3_CSV),
        config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE,
    )
    q_segs, fps_q = prepare_segments(
        load_trajectory(poses / config.BAG7_CSV),
        config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE,
    )

    det = LoopClosureDetector(
        segment_length=config.SEGMENT_LENGTH, stride=config.STRIDE, fps=fps_ref,
        lambdaa=config.LPGW_LAMBDA,  # e.g. 2.0
        downsample_points=config.LPGW_SEGMENT_POINTS,
        reference_strategy=config.LPGW_REFERENCE_STRATEGY,
    )

    proc_ref = [det.downsample_segment(s, config.LPGW_SEGMENT_POINTS) for s in ref_segs]
    proc_q = [det.downsample_segment(s, config.LPGW_SEGMENT_POINTS) for s in q_segs]

    # Initialize solver with locked solver lambda (2.0)
    lpgw_probe = LPGW(lambdaa=config.LPGW_LAMBDA, cost_mode=config.LPGW_COST_MODE)
    cost_scale = lpgw_probe.calibrate(proc_ref + proc_q)
    
    ref_seg, _ = det.select_reference(proc_ref, config.LPGW_REFERENCE_STRATEGY)
    segs = proc_q[:12] # Probe first 12 segments

    print(f"\nSweeping distance_lambda with solver lambdaa = {config.LPGW_LAMBDA}...")
    print(f"{'dist_lam':>10} {'geo share':>10} {'med dist':>12} {'self_d':>10}  status")
    print("-" * 60)

    for lam_d in lambdas_d:
        lpgw = LPGW(
            lambdaa=config.LPGW_LAMBDA,
            distance_lambda=lam_d,
            cost_mode=config.LPGW_COST_MODE,
            cost_scale=cost_scale
        )
        embs = [lpgw.embed(ref_seg, s) for s in segs]
        
        geo, pen = [], []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                a, b = embs[i], embs[j]
                q12 = np.minimum(a["q_e"], b["q_e"])
                t = float(q12.sum())
                raw_g = float(q12 @ ((a["K"] - b["K"]) ** 2) @ q12)
                g_val = raw_g / (t ** 2) if t > 1e-12 else raw_g
                geo.append(g_val)
                
                p1 = lam_d * (float(a["q_e"].sum()) ** 2 + float(b["q_e"].sum()) ** 2 - 2 * t ** 2)
                p2 = lam_d * abs(a["gamma_c_abs"] - b["gamma_c_abs"])
                pen.append(p1 + p2)

        mg, mp = float(np.median(geo)), float(np.median(pen))
        geo_share = mg / max(mg + mp, 1e-30)

        D = np.array([[lpgw.distance(a, b) for b in embs] for a in embs])
        off = D[~np.eye(len(embs), dtype=bool)]
        med = float(np.median(off)) if off.size else 0.0
        self_d = lpgw.distance(embs[0], embs[0])

        if 0.50 <= geo_share <= 0.85:
            status = "GOOD (Target 50-80%)"
        elif geo_share > 0.85:
            status = "geometry dominant (mass ignored)"
        else:
            status = "mass terms swamp geometry"

        print(f"{lam_d:10.4f} {geo_share:9.1%} {med:12.3e} {self_d:10.2e}  {status}")

    print("-" * 60)

if __name__ == "__main__":
    main()