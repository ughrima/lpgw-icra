import argparse
import sys
from pathlib import Path

# --- MOVE THESE LINES TO THE VERY TOP ---
repo_root = Path(__file__).resolve().parent
submodule_path = repo_root / "Linearized_Partial_Gromov_Wasserstein"
if submodule_path.exists():
    sys.path.insert(0, str(submodule_path))
else:
    # Fallback to Desktop symlink if needed
    desktop_submodule = Path("/Users/richajain/Desktop/Linearized_Partial_Gromov_Wasserstein")
    if desktop_submodule.exists():
        sys.path.insert(0, str(desktop_submodule))
# ----------------------------------------

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff

import config
from core.trajectory_utils import load_trajectory, prepare_segments
from lpgw import LPGW
from loop_closure import LoopClosureDetector


# 1. Load segments
poses = config.POSES_DIR
ref_segs, _ = prepare_segments(
    load_trajectory(f"{poses}/{config.BAG3_CSV}"),
    config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE
)
q_segs, _ = prepare_segments(
    load_trajectory(f"{poses}/{config.BAG7_CSV}"),
    config.TARGET_POINTS, config.SEGMENT_LENGTH, config.STRIDE
)

# 2. Compute ground-truth center distances (what the label uses)
rc = np.array([s.mean(0) for s in ref_segs])
qc = np.array([s.mean(0) for s in q_segs])
true_dist = cKDTree(rc).query(qc, k=1)[0]

print(f"Loaded {len(ref_segs)} ref segments and {len(q_segs)} query segments.")

# 3. Compute Hausdorff distance matrix
print("Computing Hausdorff distance matrix...")
D_haus = np.zeros((len(q_segs), len(ref_segs)))
for i, q in enumerate(q_segs):
    for j, r in enumerate(ref_segs):
        h1 = directed_hausdorff(q, r)[0]
        h2 = directed_hausdorff(r, q)[0]
        D_haus[i, j] = max(h1, h2)

# 4. Compute DTW distance matrix (using Euclidean norm fallback or fastdtw if available)
print("Computing DTW distance matrix...")
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    has_fastdtw = True
except ImportError:
    has_fastdtw = False

D_dtw = np.zeros((len(q_segs), len(ref_segs)))
for i, q in enumerate(q_segs):
    for j, r in enumerate(ref_segs):
        if has_fastdtw:
            D_dtw[i, j], _ = fastdtw(q, r, dist=euclidean)
        else:
            # Fallback mean point-to-point Euclidean distance approximation
            D_dtw[i, j] = np.mean([np.linalg.norm(q_pt - r_pt) for q_pt, r_pt in zip(q, r)])

# 5. Compute LPGW distance matrix
print("Computing LPGW distance matrix...")
det = LoopClosureDetector(
    segment_length=config.SEGMENT_LENGTH, stride=config.STRIDE, fps=10.0,
    lambdaa=getattr(config, 'LPGW_LAMBDA', 2.0),
    downsample_points=config.LPGW_SEGMENT_POINTS,
    reference_strategy=config.LPGW_REFERENCE_STRATEGY,
)
proc_ref = [det.downsample_segment(s, config.LPGW_SEGMENT_POINTS) for s in ref_segs]
proc_q = [det.downsample_segment(s, config.LPGW_SEGMENT_POINTS) for s in q_segs]

lpgw = LPGW(lambdaa=det.lambdaa, cost_mode=config.LPGW_COST_MODE,
            huber_delta_frac=config.LPGW_HUBER_DELTA_FRAC,
            cost_scale=getattr(config, 'LPGW_COST_SCALE', None))
if lpgw.cost_scale is None:
    lpgw.calibrate(proc_ref + proc_q)

# Embed reference and query segments
ref_embs = [lpgw.embed(proc_ref[0], r) for r in proc_ref]
q_embs = [lpgw.embed(proc_ref[0], q) for q in proc_q]

D_lpgw = np.array([[lpgw.distance(qe, re) for re in ref_embs] for qe in q_embs])

# 6. Evaluate label leakage correlations
print("\n--- Label Leakage Results ---")
for name, D in [("Hausdorff", D_haus), ("DTW", D_dtw), ("LPGW", D_lpgw)]:
    min_d = D.min(axis=1)
    corr, _ = spearmanr(min_d, true_dist)
    print(f"{name}: {round(corr, 3)}")