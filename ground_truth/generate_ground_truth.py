# ground_truth/generate_ground_truth.py
"""
Canonical ground-truth generator.

Uses the SAME segmentation as all LPGW experiments:
  - Downsample to config.TARGET_POINTS
  - Segment via core.trajectory_utils.prepare_segments
  - Build GT by nearest reference segment center
  - Save labels + segment metadata so any future script can verify alignment.

Supports multiple datasets via config.ACTIVE_DATASET and names files as:
  - gt_{DATASET_SHORT}_{tolerance}m.csv
  - gt_{DATASET_SHORT}_{tolerance}m_meta.json
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import json

from core.trajectory_utils import prepare_segments 
from core.trajectory_utils import load_trajectory
import config


def generate_canonical_ground_truth(
    ref_csv: Path,
    query_csv: Path,
    tolerance_m: float,
    output_csv: Path,
    metadata_json: Path,
):
    # Load trajectories
    ref_df = load_trajectory(ref_csv)
    query_df = load_trajectory(query_csv)

    # Downsample + segment independently (without min_len truncation)
    target_points = getattr(config, "TARGET_POINTS", 5000)
    segment_length = getattr(config, "SEGMENT_LENGTH", 5.0)
    fps = getattr(config, "FPS", 10)
    stride = getattr(config, "STRIDE", 1.0)

    segments_3, fps_ref = prepare_segments(ref_df, target_points, segment_length, stride)
    segments_7, fps_qry = prepare_segments(query_df, target_points, segment_length, stride)

    # Compute segment centers for full sets independently
    ref_centers = np.array([seg.mean(axis=0) for seg in segments_3])
    query_centers = np.array([seg.mean(axis=0) for seg in segments_7])

    # Nearest reference segment for each query segment
    from sklearn.neighbors import KDTree

    tree = KDTree(ref_centers)
    dists, idxs = tree.query(query_centers, k=1)
    dists = dists[:, 0]
    idxs = idxs[:, 0]

    # Labels at given tolerance
    labels = (dists <= tolerance_m).astype(int)

    # Save ground truth CSV
    gt_df = pd.DataFrame({
        "query_index": np.arange(len(labels)),
        "nearest_ref_index": idxs,
        "nearest_distance_m": dists,
        "label": labels,
    })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    gt_df.to_csv(output_csv, index=False)

    # Save metadata so any script can verify it's using the same segmentation
    metadata = {
        "dataset": config.ACTIVE_DATASET,
        "segment_length": segment_length,
        "stride": stride,
        "target_points": target_points,
        "fps_ref": fps_ref,
        "fps_query": fps_qry,
        "ref_csv": str(ref_csv),
        "query_csv": str(query_csv),
        "tolerance_m": tolerance_m,
        "num_query_segments": len(labels),
        "num_ref_segments": len(ref_centers),
        "config": {
            "TARGET_POINTS": target_points,
            "SEGMENT_LENGTH": segment_length,
            "FPS": fps,
            "STRIDE": stride,
        },
        "ref_centers": ref_centers.tolist(),
        "query_centers": query_centers.tolist(),
        "positive_count": int(labels.sum()),
        "negative_count": int(len(labels) - labels.sum()),
    }

    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Ground truth saved to {output_csv}")
    print(f"Metadata saved to {metadata_json}")
    print(f"Positive segments: {labels.sum()} / {len(labels)}")


def main():
    poses_dir = Path("data/poses")
    ref_csv = poses_dir / config.BAG3_CSV
    query_csv = poses_dir / config.BAG7_CSV

    gt_dir = Path("ground_truth/files")
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Use dataset-specific prefixes so UZH and KITTI GT don't collide
    dataset = config.DATASET_SHORT

    # Primary GT at 2.0 m (used by all experiments)
    generate_canonical_ground_truth(
        ref_csv,
        query_csv,
        tolerance_m=config.SPATIAL_TOLERANCE,  # 2.0 m
        output_csv=gt_dir / f"gt_{dataset}_{config.SPATIAL_TOLERANCE}m.csv",
        metadata_json=gt_dir / f"gt_{dataset}_{config.SPATIAL_TOLERANCE}m_meta.json",
    )

    # Optional: GT at 0.5 m (if you want it for ablations)
    generate_canonical_ground_truth(
        ref_csv,
        query_csv,
        tolerance_m=config.GT_SEGMENT_CENTER_THRESHOLD,  # 0.5 m
        output_csv=gt_dir / f"gt_{dataset}_{config.GT_SEGMENT_CENTER_THRESHOLD}m.csv",
        metadata_json=gt_dir / f"gt_{dataset}_{config.GT_SEGMENT_CENTER_THRESHOLD}m_meta.json",
    )


if __name__ == "__main__":
    main()