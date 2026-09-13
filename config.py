
# ----------------- Dataset selection -----------------

ACTIVE_DATASET = "uzh_fpv"  # "uzh_fpv" or "kitti"

DATASETS = {
    "uzh_fpv": {
        "ref_csv": "poses_bag-3.csv",
        "query_csv": "poses_bag-7.csv",
        "label_ref": "Reference (Bag 3)",
        "label_query": "Query (Bag 7)",
        "short_name": "uzhfpv",
        # Secondary operating point only. Roughly this dataset's base
        # rate; see note (1) -- do not read F1 here as a headline result.
        "percentile": 68.0,
    },
    "kitti": {
        "ref_csv": "kitti_00_reference.csv",
        "query_csv": "kitti_00_query.csv",
        "label_ref": "Reference (KITTI Seq 00)",
        "label_query": "Query (KITTI Seq 00 query)",
        "short_name": "kitti00",
        # KITTI's base rate is ~15%, NOT 68%.
        "percentile": 15.0,
    },
}

_d = DATASETS[ACTIVE_DATASET]
BAG3_CSV = _d["ref_csv"]
BAG7_CSV = _d["query_csv"]
DATASET_LABEL_REF = _d["label_ref"]
DATASET_LABEL_QUERY = _d["label_query"]
DATASET_SHORT = _d["short_name"]

POSES_DIR = "data/poses"


# ----------------- Trajectory preprocessing -----------------

TARGET_POINTS = 5000       # whole-trajectory cap
SEGMENT_LENGTH = 5.0       # seconds
STRIDE = 0.5               # seconds  -> 50-point windows, 80% overlap

# Points per segment fed to LPGW. Segments hold 50 points, so None is a
# no-op; set lower only to trade accuracy for speed.


# ----------------- Operating points -----------------

# SECONDARY. Primary reporting is threshold-free (max F1, AP, Recall@1).
PERCENTILE = _d["percentile"]

SPATIAL_TOLERANCE = 1.0 #uzhfpv
# SPATIAL_TOLERANCE = 5.0 #kitti
TOLERANCE_SWEEP = [0.5, 1.0, 1.5, 2.0]
PERCENTILE_SWEEP = [1, 5, 10, 20, 50]

# Index tolerance for Recall@1: a retrieved reference counts as correct if
# it is within this many segments of the canonical nearest reference.
RECALL_AT_1_INDEX_TOL = 2


# ----------------- Ground truth -----------------

GT_SEGMENT_CENTER_THRESHOLD = 0.5   # secondary GT for the tolerance ablation


# ----------------- LPGW -----------------

# config.py settings for LPGW
LPGW_LAMBDA = 2.0  #uzhfpv
# LPGW_LAMBDA = 1.0          # kitti
LPGW_DISTANCE_LAMBDA = 0.5 #uzhfpv
# LPGW_DISTANCE_LAMBDA = 0.2   # kitti
LPGW_SEGMENT_POINTS = 100    # Upgraded from 50 to prevent heavy smoothing on aggressive flight


# "squared" is faithful to Eq. (24). "huber" is available for outlier
# robustness; its delta is a FRACTION of segment extent, not metres.
LPGW_COST_MODE = "squared"
LPGW_HUBER_DELTA_FRAC = 0.1


# Filled by lpgw.calibrate() at runtime. Pin it here once measured so that
# drift / overlap / ablation runs share one normalisation and therefore
# measure the perturbation rather than a shifting baseline.
LPGW_COST_SCALE = 56.53432887902825 #uzhfpv
# LPGW_COST_SCALE = 97.9785 #kitti

LPGW_REFERENCE_STRATEGY = "median_diameter"   # median_diameter|first|middle|last


# ----------------- Perturbation experiments -----------------

# Drift as a fraction of path length (real odometry drift is 1-2%).
DRIFT_FRACTIONS = [0.0, 0.005, 0.01, 0.02, 0.05]

# Contiguous fraction of each query segment retained. Random point removal
# leaves the curve's shape unchanged and does not model partial overlap.
OVERLAP_RATIOS = [1.0, 0.8, 0.6, 0.4]

REFERENCE_ABLATION_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ----------------- Bootstrap -----------------

BOOTSTRAP_RESAMPLES = 10000
# >= 5 needed to break the 5 s / 1 s window overlap between neighbours.
BOOTSTRAP_BLOCK_SIZE = 10


# ----------------- Reproducibility -----------------

RANDOM_SEED = 42


# ----------------- Misc -----------------

TIME_TOLERANCE = 2.0
DEFAULT_POSE_TOPIC = "/groundtruth/pose"


# ----------------- Baseline experiment -----------------

# Queries are subsampled to control classical-baseline cost. The REFERENCE
# DATABASE IS ALWAYS KEPT WHOLE: subsampling references would discard the
# true match for most positives and invalidate the canonical labels.
BASELINE_SEGMENT_POLICY = "subsample_fixed"   # "full" | "subsample_fixed"
MAX_BASELINE_QUERIES = 150
MAX_BASELINE_SEGMENTS = MAX_BASELINE_QUERIES  # backwards-compat alias

DTW_WINDOW = 10    # Sakoe-Chiba band

FPS = None                      # Derived per dataset via effective_fps()
