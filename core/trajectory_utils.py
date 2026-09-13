# core/trajectory_utils.py

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Segments:
    """
    Metadata describing trajectory segments.
    """

    centers: np.ndarray
    indices: np.ndarray
    timestamps: np.ndarray

def load_trajectory(path: Path) -> pd.DataFrame:
    """
    Load a trajectory CSV.

    Required columns:
        Timestamp
        PosX
        PosY
        PosZ
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Trajectory file not found: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "Timestamp",
        "PosX",
        "PosY",
        "PosZ",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns {missing} in {path}"
        )

    if len(df) == 0:
        raise ValueError(
            f"Trajectory file is empty: {path}"
        )

    # KITTI datasets often use frame indices (step = 1.0) instead of seconds.
    # Scale them to true 10 Hz timestamps (0.1s per frame) if detected.
    if "kitti" in str(path).lower():
        t_diff = np.diff(df["Timestamp"].to_numpy(dtype=float))
        if len(t_diff) > 0 and np.allclose(t_diff, 1.0, atol=1e-2):
            df["Timestamp"] = df["Timestamp"] * 0.1

    return df


def resample_to_target_points(
    df: pd.DataFrame,
    target_points: int,
) -> pd.DataFrame:
    """
    Uniformly subsample a trajectory by sample index.

    If the trajectory already contains fewer than or equal
    to target_points samples, it is returned unchanged.

    This function assumes the input trajectory is sampled
    at approximately uniform temporal intervals.
    """

    if target_points <= 0:
        raise ValueError(
            "target_points must be greater than zero."
        )

    N = len(df)

    if N == 0:
        raise ValueError(
            "Cannot resample an empty trajectory."
        )

    if N <= target_points:
        return df.reset_index(drop=True)

    indices = np.linspace(
        0,
        N - 1,
        target_points,
        dtype=int,
    )

    return df.iloc[indices].reset_index(drop=True)

def segment_trajectory(
    traj,
    segment_length=3.0,
    fps=10,
    stride=1.0,
):
    """
    Segment a trajectory into overlapping windows.

    Parameters
    ----------
    traj : np.ndarray
        Trajectory array of shape (N, 3).
    segment_length : float
        Segment duration in seconds.
    fps : float
        Sampling frequency.
    stride : float
        Distance between consecutive segment starts in seconds.

    Returns
    -------
    list[np.ndarray]
        List of trajectory segments.
    """
    if len(traj) < 1:
        raise ValueError("Trajectory cannot be empty.")

    num_points = int(segment_length * fps)
    stride_pts = int(stride * fps)

    if num_points <= 0:
        raise ValueError("segment_length * fps must be greater than zero.")

    if stride_pts <= 0:
        raise ValueError("stride * fps must be greater than zero.")

    segments = []

    for start in range(
        0,
        len(traj) - num_points + 1,
        stride_pts,
    ):
        segment = traj[start:start + num_points]

        if segment.shape[0] == num_points:
            segments.append(np.array(segment))

    return segments


def downsample_trajectory(
    traj,
    target_points=1000,
):
    """
    Uniformly downsample a trajectory.

    If the trajectory already contains fewer than or equal
    to target_points samples, it is returned unchanged.
    """
    if target_points <= 0:
        raise ValueError(
            "target_points must be greater than zero."
        )

    if traj.shape[0] == 0:
        raise ValueError(
            "Cannot downsample an empty trajectory."
        )

    if traj.shape[0] <= target_points:
        return traj

    indices = np.linspace(
        0,
        traj.shape[0] - 1,
        target_points,
        dtype=int,
    )

    return traj[indices]

def effective_fps(df, target_points=None, timestamp_col="Timestamp"):
    """
    Sampling rate in Hz AFTER downsampling to `target_points`.

    Derived from timestamps rather than assumed, because the raw rate
    differs per dataset (UZH-FPV Leica ~453 Hz, KITTI 10 Hz) and
    downsampling changes it again.
    """
    if timestamp_col not in df.columns:
        raise ValueError(
            f"No {timestamp_col!r} column; cannot derive sampling rate. "
            "Do not fall back to a hardcoded FPS -- that is the bug."
        )

    t = np.asarray(df[timestamp_col], dtype=float)
    if len(t) < 2:
        raise ValueError("need at least two samples")

    duration = float(t[-1] - t[0])
    if duration <= 0:
        raise ValueError(
            f"non-positive duration ({duration}); check timestamp units "
            "(seconds vs nanoseconds)"
        )

    n_after = len(t) if target_points is None else min(len(t), target_points)
    return n_after / duration


def prepare_segments(
    df,
    target_points,
    segment_length,
    stride,
    xyz_cols=("PosX", "PosY", "PosZ"),
    timestamp_col="Timestamp",
    verbose=True,
):
    """
    Downsample and segment a trajectory at its TRUE sampling rate.

    Returns (segments, fps). Use this everywhere instead of calling
    downsample_trajectory + segment_trajectory with config.FPS.

    Asserts the resulting segments are `segment_length` seconds long, so
    a wrong rate fails loudly instead of silently producing sub-second
    windows.
    """
    xyz = df[list(xyz_cols)].to_numpy(dtype=float)
    fps = effective_fps(df, target_points, timestamp_col)

    xyz_ds = downsample_trajectory(xyz, target_points)
    segments = segment_trajectory(
        xyz_ds,
        segment_length=segment_length,
        fps=fps,
        stride=stride,
    )

    if not segments:
        raise ValueError(
            f"no segments produced: {len(xyz_ds)} points, "
            f"{int(round(segment_length * fps))} needed per window"
        )

    actual = len(segments[0]) / fps
    if abs(actual - segment_length) > 0.1:
        raise AssertionError(
            f"segments are {actual:.3f}s but SEGMENT_LENGTH={segment_length}s. "
            f"fps={fps:.2f}, points/segment={len(segments[0])}. "
            "This is the bug that made UZH-FPV segments 0.55s."
        )

    if verbose:
        t = np.asarray(df[timestamp_col], dtype=float)
        print(f"  raw {len(xyz)} pts over {t[-1] - t[0]:.2f}s "
              f"({len(xyz) / (t[-1] - t[0]):.1f} Hz raw) "
              f"-> {len(xyz_ds)} pts @ {fps:.2f} Hz")
        print(f"  {len(segments)} segments x {len(segments[0])} pts "
              f"= {actual:.2f}s each")

    return segments, fps