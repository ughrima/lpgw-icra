"""
loop_closure.py
---------------

Trajectory-specific wrapper around the mathematically faithful LPGW
core in lpgw.py.

This file deliberately separates:
    1. LPGW mathematics          -> lpgw.py
    2. trajectory segmentation  -> this file
    3. reference selection      -> this file
    4. thresholding/evaluation  -> this file

That makes the experimental pipeline easier to describe and reproduce.
"""


from __future__ import annotations
import sys
from pathlib import Path

# Automatically find the repository root
repo_root = Path(__file__).resolve().parent
if repo_root.name == "experiments":
    repo_root = repo_root.parent

sys.path.insert(0, str(repo_root))

# Check for the submodule in both possible locations
submodule_path = repo_root / "Linearized_Partial_Gromov_Wasserstein"
if not submodule_path.exists():
    submodule_path = repo_root.parent / "Linearized_Partial_Gromov_Wasserstein"

if submodule_path.exists():
    sys.path.insert(0, str(submodule_path))
else:
    raise RuntimeError(
        f"Could not find Linearized_Partial_Gromov_Wasserstein folder. "
        f"Checked {repo_root / 'Linearized_Partial_Gromov_Wasserstein'} and "
        f"{repo_root.parent / 'Linearized_Partial_Gromov_Wasserstein'}"
    )

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.mixture import GaussianMixture

from lpgw import LPGW


class LoopClosureDetector:
    """
    LPGW-based trajectory loop-closure detector.

    Parameters
    ----------
    segment_length:
        Segment duration in seconds.
    fps:
        Sampling frequency used by the trajectory files.
    stride:
        Segment stride in seconds.
    lambdaa:
        LPGW partial-matching penalty.
    downsample_points:
        If not None, each segment is uniformly downsampled to at most
        this many points before LPGW.
    reference_strategy:
        "robust", "first", "middle", or "last".
    """
    def __init__(
        self,
        segment_length=5.0,
        fps=100.0,
        stride=0.5,
        lambdaa=0.5,
        distance_lambda=0.05,
        downsample_points=50,
        reference_strategy="robust",
        cost_mode="squared",
        huber_delta_frac=0.1,
    ):
        self.segment_length = segment_length
        self.fps = fps
        self.stride = stride
        self.lambdaa = lambdaa
        self.distance_lambda = distance_lambda
        self.downsample_points = downsample_points
        self.reference_strategy = reference_strategy
        self.cost_mode = cost_mode
        self.huber_delta_frac = huber_delta_frac
        
        # Add these tracking attributes:
        self.reference_index = None
        self.reference_segment = None
        self.cost_scale = None

        
        self.lpgw = LPGW(
            lambdaa=self.lambdaa,
            distance_lambda=self.distance_lambda,
            cost_mode=self.cost_mode,
            huber_delta_frac=self.huber_delta_frac,
            num_itermax_gw=5000,  # <--- Increased to prevent early stopping warnings
            num_itermax=5000,     # <--- Increased
            tol=1e-9,
        )

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def segment_trajectory(
        trajectory: np.ndarray,
        segment_length: float = 5.0,
        fps: float = 10.0,
        stride: float = 1.0,
    ) -> list[np.ndarray]:
        trajectory = np.asarray(trajectory, dtype=np.float64)

        if trajectory.ndim != 2:
            raise ValueError("trajectory must have shape (N, D)")
        if len(trajectory) == 0:
            raise ValueError("trajectory cannot be empty")

        num_points = int(round(segment_length * fps))
        stride_points = int(round(stride * fps))

        if num_points <= 0 or stride_points <= 0:
            raise ValueError("segment_length and stride must be positive")

        segments = []

        for start in range(
            0,
            len(trajectory) - num_points + 1,
            stride_points,
        ):
            seg = trajectory[start:start + num_points]

            if len(seg) == num_points:
                segments.append(seg.copy())

        return segments

    @staticmethod
    def downsample_segment(
        segment: np.ndarray,
        target_points: int | None,
    ) -> np.ndarray:
        if target_points is None:
            return np.asarray(segment, dtype=np.float64)

        segment = np.asarray(segment, dtype=np.float64)

        if len(segment) <= target_points:
            return segment.copy()

        indices = np.linspace(
            0,
            len(segment) - 1,
            target_points,
        ).astype(int)

        return segment[indices]

    # ------------------------------------------------------------------
    # Reference selection
    # ------------------------------------------------------------------

    @staticmethod
    def select_reference(
        segments: list[np.ndarray],
        strategy: str = "robust",
    ) -> tuple[np.ndarray, int]:

        if not segments:
            raise ValueError("segments cannot be empty")

        strategy = strategy.lower()

        if strategy == "first":
            return segments[0], 0

        if strategy == "middle":
            idx = len(segments) // 2
            return segments[idx], idx

        if strategy == "last":
            return segments[-1], len(segments) - 1

        if strategy in ("robust", "median_diameter"):
            diameters = np.array(
                [
                    float(np.max(cdist(seg, seg)))
                    for seg in segments
                ],
                dtype=np.float64,
            )
            idx = int(
                np.argmin(
                    np.abs(diameters - np.median(diameters))
                )
            )
            return segments[idx], idx

        if strategy != "median_diameter":
            raise ValueError(
                "strategy must be one of: median_diameter, robust, "
                "first, middle, last"
            )

    # ------------------------------------------------------------------
    # Distance matrix
    # ------------------------------------------------------------------
    def compute_distance_matrix(self, segments1, segments2, verbose=True):
        proc_q = [self.downsample_segment(s, self.downsample_points) for s in segments1]
        proc_ref = [self.downsample_segment(s, self.downsample_points) for s in segments2]

        # Use a pinned reference if one was set; only fall back to the strategy.
        if self.reference_index is not None:
            self.reference_segment = proc_ref[self.reference_index]
        else:
            self.reference_segment, self.reference_index = self.select_reference(
                proc_ref, self.reference_strategy
            )

        # Use a pinned cost scale if one was set; only calibrate if missing.
        if self.lpgw.cost_scale is None:
            probe = LPGW(cost_mode=self.cost_mode,
                        huber_delta_frac=self.huber_delta_frac)
            self.lpgw.cost_scale = probe.calibrate(proc_ref + proc_q)
        self.cost_scale = self.lpgw.cost_scale

        print(f"Reference index: {self.reference_index}   "
            f"cost_scale: {self.cost_scale:.4f}")

        embeddings_q = [self.lpgw.embed(self.reference_segment, s) for s in proc_q]
        embeddings_r = [self.lpgw.embed(self.reference_segment, s) for s in proc_ref]
        D = self.lpgw.distance_matrix(embeddings_q, embeddings_r)
        return D

    def embedding_vectors(self, embeddings):
        """Return fixed-reference vectors for approximate retrieval."""
        return np.stack(
            [self.lpgw.embedding_vector(embedding) for embedding in embeddings]
        )

    def approximate_nearest_references(
        self,
        query_embeddings,
        reference_embeddings,
    ):
        """Retrieve candidates using the fixed-reference geometric vectors.

        This is an approximate retrieval path. Exact LPGW still includes
        pair-dependent partial-mass terms and should be used to rerank the
        returned candidates when those terms matter.
        """
        reference_vectors = self.embedding_vectors(reference_embeddings)
        query_vectors = self.embedding_vectors(query_embeddings)
        tree = cKDTree(reference_vectors)
        distances, indices = tree.query(query_vectors, k=1)
        return distances, indices

    # ------------------------------------------------------------------
    # Loop-closure detection
    # ------------------------------------------------------------------

    @staticmethod
    def choose_threshold(
        D: np.ndarray,
        method: str = "percentile",
        percentile: float = 1.0,
    ) -> float:

        D = np.asarray(D, dtype=np.float64)
        values = D[np.isfinite(D)]

        if len(values) == 0:
            raise ValueError("Distance matrix contains no finite values")

        method = method.lower()

        if method == "percentile":
            if not 0 <= percentile <= 100:
                raise ValueError(
                    "percentile must be between 0 and 100."
                )

            finite_rows = np.isfinite(D).any(axis=1)
            if not np.any(finite_rows):
                raise ValueError(
                    "Distance matrix contains no finite row scores"
                )

            best_per_query = np.min(
                np.where(np.isfinite(D[finite_rows]), D[finite_rows], np.inf),
                axis=1,
            )
            return float(np.percentile(best_per_query, percentile))

        if method == "gmm":
            if len(values) < 10:
                raise ValueError(
                    "GMM threshold requires more distance samples"
                )

            gmm = GaussianMixture(
                n_components=2,
                random_state=42,
            )
            gmm.fit(values.reshape(-1, 1))

            means = np.sort(gmm.means_.ravel())

            return float(np.mean(means))

        if method == "mad":
            median = float(np.median(values))
            mad = float(stats.median_abs_deviation(values))

            return float(median - 2.0 * mad)

        raise ValueError(
            "Unknown threshold method. "
            "Use 'percentile', 'gmm', or 'mad'."
        )

    @staticmethod
    def detect_loop_closures(
        D: np.ndarray,
        threshold: float,
    ) -> tuple[list[int], list[int], list[float]]:

        D = np.asarray(D, dtype=np.float64)

        flags = []
        matches = []
        min_distances = []

        for i in range(D.shape[0]):

            row = D[i]

            valid = np.isfinite(row)

            if not np.any(valid):
                flags.append(0)
                matches.append(-1)
                min_distances.append(np.inf)
                continue

            valid_indices = np.flatnonzero(valid)

            local_idx = int(
                np.argmin(row[valid])
            )

            best_idx = int(
                valid_indices[local_idx]
            )

            best_distance = float(
                row[best_idx]
            )

            flags.append(
                int(best_distance <= threshold)
            )

            matches.append(best_idx)
            min_distances.append(best_distance)

        return flags, matches, min_distances

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(
        ground_truth: np.ndarray,
        predictions: np.ndarray,
    ) -> dict:

        gt = np.asarray(ground_truth).astype(int)
        pred = np.asarray(predictions).astype(int)

        if gt.shape != pred.shape:
            raise ValueError(
                "ground_truth and predictions must have the same shape"
            )

        tp = int(np.sum((gt == 1) & (pred == 1)))
        fp = int(np.sum((gt == 0) & (pred == 1)))
        fn = int(np.sum((gt == 1) & (pred == 0)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        f1 = (
            2 * precision * recall
            / max(precision + recall, 1e-12)
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }


def load_xyz_trajectory(csv_path: str):
    """
    Load trajectory CSV with:
        PosX, PosY, PosZ, Timestamp
    """
    df = pd.read_csv(csv_path)

    required = {
        "PosX",
        "PosY",
        "PosZ",
        "Timestamp",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV is missing columns: {sorted(missing)}"
        )

    xyz = df[
        ["PosX", "PosY", "PosZ"]
    ].to_numpy(dtype=np.float64)

    timestamps = df["Timestamp"].to_numpy()

    return xyz, timestamps
