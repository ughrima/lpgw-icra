"""
lpgw.py  -- CORRECTED
---------------------

Linear Partial Gromov-Wasserstein core for trajectory loop-closure.

Reference:
    Y. Bai, A. Kothapalli, H. Du, R. Diaz Martin, S. Kolouri,
    "Linear Partial Gromov-Wasserstein Embedding", arXiv:2410.16669.


WHAT CHANGED AND WHY
====================

The previous version computed cost matrices like this:

    Xn = X / global_scale                                  # ~30-60 m
    Cx = huber(Xn, delta = 0.15 / global_scale)            # delta ~ 0.005

Two things went wrong.

(1) THE PARTIAL TRANSPORT NEVER ACTIVATED.
    With delta ~= 0.005 in normalised units, ~98% of pairwise residuals sit
    in Huber's LINEAR regime, so Cx ~= delta * r and the whole cost matrix
    is O(1e-3).  The geometric term of the PGW objective then lands around
    1e-12, while the penalty for discarding all mass is lambda*(|mu|^2 +
    |nu|^2) = 1.0.  Lambda outweighed the geometry by ~1e11, so the solver
    always transported FULL mass.  Consequently, for every segment:

        transported_mass == 1.0
        q_e              == p == uniform          (identical for all)
        gamma_c_abs      == 0
        penalty_1        == lambda*(1 + 1 - 2*1) == 0
        penalty_2        == lambda*|0 - 0|       == 0

    i.e. every partial-matching term was exactly zero and the method
    silently degenerated to plain linear GW on a near-zero cost matrix.

(2) THE COST WAS NOT A SQUARED DISTANCE.
    Eq. (24) of the paper is defined with ||y_i - y_j||^2.  Huber in its
    linear regime supplies delta*||y_i - y_j|| instead -- a different
    objective, scaled by a tiny constant.

THE FIX
-------
Costs are now built in METRES (so huber_delta means what it says), then
divided by ONE dataset-wide constant `cost_scale` chosen so the median
cost entry is 1.  With costs O(1), lambda = 0.5 is a real trade-off and
the solver can actually choose to discard mass.

Call `calibrate(reference, segments)` once per dataset before embedding.

THE CHECK THAT MATTERS
----------------------
After any change, run:

    e = lpgw.embed(reference, segment)
    print(e["transported_mass"])     # MUST be < 1.0
    print(e["gamma_c_abs"])          # MUST be > 0

If transported_mass is 1.0 you are running GW, not PGW, and every lambda
term in distance() is identically zero.  Use `diagnose()` below.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

_SOLVER_IMPORT_ERROR: Exception | None = None
try:  # lazy so the pure math stays unit-testable without the repo
    from lib.gromov import partial_gromov_ver1 as _default_solver
except Exception as exc:  # pragma: no cover
    _default_solver = None
    _SOLVER_IMPORT_ERROR = exc


_COST_MODES = ("squared", "huber")


class LPGW:
    """
    Linear Partial Gromov-Wasserstein embedding.

    Parameters
    ----------
    lambdaa : float
        Partial-transport penalty. Meaningful only because cost matrices
        are normalised to median 1 (see `calibrate`).
    cost_mode : {"squared", "huber"}
        "squared"  -- ||x_i - x_j||^2, faithful to Eq. (24). Default.
        "huber"    -- Huber loss of the residual, for outlier robustness.
    huber_delta_frac : float
        Huber transition point, as a FRACTION of the segment scale
        (not metres). 0.15 m on a 60 m segment puts ~98% of residuals in
        the linear regime, which is why the old absolute value failed.
        Ignored when cost_mode == "squared".
    cost_scale : float or None
        Dataset-wide divisor applied to every cost matrix. Set by
        `calibrate()`. If None, costs are used unnormalised (NOT
        recommended -- lambda will not behave).
    """

    def __init__(
        self,
        lambdaa=0.5,
        distance_lambda=None,
        cost_mode="squared",
        huber_delta_frac=0.1,
        cost_scale=None,
        num_itermax_gw=5000, 
        num_itermax=5000,    
        tol=1e-9,            
        line_search=True,    
        seed=None,           
    ):
        self.lambdaa = float(lambdaa)
        # Solver lambda and discrepancy lambda play different roles: one sets
        # how much mass PGW discards, the other weights the mass penalties
        # against the geometric term in Eq. (25).
        self.distance_lambda = (
            float(lambdaa) if distance_lambda is None else float(distance_lambda)
        )
        self.cost_mode = cost_mode
        self.huber_delta_frac = huber_delta_frac
        self.cost_scale = cost_scale

        # Bind solver and optimization parameters
        self._solver = _default_solver
        self.num_itermax_gw = num_itermax_gw
        self.num_itermax = num_itermax
        self.tol = tol
        self.line_search = line_search
        self.seed = seed
        

    # ------------------------------------------------------------------
    # Cost matrices
    # ------------------------------------------------------------------

    @staticmethod
    def diameter(points: np.ndarray) -> float:
        X = np.asarray(points, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("points must be a non-empty (N, D) array")
        if X.shape[0] == 1:
            return 0.0
        return float(np.max(cdist(X, X, metric="euclidean")))

    def raw_cost_matrix(self, points: np.ndarray) -> np.ndarray:
        """
        Cost matrix in physical units, BEFORE cost_scale normalisation.

        Coordinates stay in metres here on purpose: that is what makes
        huber_delta interpretable, and what keeps the squared-distance
        cost faithful to Eq. (24).
        """
        X = np.asarray(points, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("points must be a non-empty (N, D) array")

        r = cdist(X, X, metric="euclidean")

        if self.cost_mode == "squared":
            return r ** 2

        # Huber, with delta tied to this segment's own extent so it lands
        # in a sensible regime instead of collapsing to the linear branch.
        delta = self.huber_delta_frac * max(self.diameter(X), 1e-12)
        return np.where(
            r <= delta,
            0.5 * r ** 2,
            delta * (r - 0.5 * delta),
        )

    def cost_matrix(self, points: np.ndarray) -> np.ndarray:
        """Normalised cost matrix (median entry ~1 after calibration)."""
        C = self.raw_cost_matrix(points)
        if self.cost_scale is None:
            return C
        return C / self.cost_scale

    def calibrate(self, segments, max_probe: int = 40) -> float:
        """
        Choose ONE dataset-wide cost_scale so the median cost entry is 1.

        Call this once, on reference + query segments, BEFORE embedding.
        Store the returned value and reuse it for every experiment on that
        dataset (drift, overlap, ablations) so those experiments measure
        the perturbation rather than a shifting normalisation.
        """
        vals = []
        for seg in list(segments)[:max_probe]:
            C = self.raw_cost_matrix(seg)
            off = C[~np.eye(C.shape[0], dtype=bool)]
            if off.size:
                vals.append(float(np.median(off)))

        if not vals:
            raise ValueError("no usable segments for calibration")

        self.cost_scale = float(np.median(vals))
        if self.cost_scale <= 0:
            raise ValueError("calibrated cost_scale is non-positive")
        return self.cost_scale

    @staticmethod
    def uniform_mass(n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        return np.full(n, 1.0 / n, dtype=np.float64)

    @staticmethod
    def _normalize_mass(mass, n: int) -> np.ndarray:
        if mass is None:
            return LPGW.uniform_mass(n)
        mass = np.asarray(mass, dtype=np.float64).reshape(-1)
        if len(mass) != n:
            raise ValueError(f"Mass has length {len(mass)}, expected {n}")
        if np.any(mass < 0):
            raise ValueError("Masses must be non-negative")
        total = mass.sum()
        if total <= 0:
            raise ValueError("Mass vector must have positive total mass")
        return mass / total

    # ------------------------------------------------------------------
    # PGW
    # ------------------------------------------------------------------

    def solve_pgw(self, Cx, Cy, p, q) -> np.ndarray:
        if self._solver is None:  # pragma: no cover
            raise ImportError(
                "Could not import lib.gromov.partial_gromov_ver1. Clone "
                "https://github.com/mint-vu/Linearized_Partial_Gromov_Wasserstein"
                " and put it on PYTHONPATH.\n"
                f"Original error: {_SOLVER_IMPORT_ERROR}"
            )

        gamma = self._solver(
            Cx, Cy, p, q,
            Lambda=self.lambdaa,
            numItermax_gw=self.num_itermax_gw,
            numItermax=self.num_itermax,
            tol=self.tol,
            log=False,
            verbose=False,
            line_search=self.line_search,
            seed=self.seed,
        )

        gamma = np.asarray(gamma, dtype=np.float64)
        expected = (len(p), len(q))
        if gamma.shape != expected:
            raise RuntimeError(f"PGW returned {gamma.shape}, expected {expected}")
        if not np.all(np.isfinite(gamma)):
            raise RuntimeError("PGW returned non-finite values")
        gamma[gamma < 0] = 0.0
        return gamma

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, X, Y, p=None, q=None) -> dict:
        """Embed target Y relative to reference X."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must have shape (N, D)")
        if len(X) == 0 or len(Y) == 0:
            raise ValueError("X and Y cannot be empty")

        # Costs in normalised units. NOTE: coordinates are NOT rescaled --
        # only the cost matrices are, by one shared constant.
        Cx = self.cost_matrix(X)
        Cy = self.cost_matrix(Y)

        p = self._normalize_mass(p, len(X))
        q = self._normalize_mass(q, len(Y))

        gamma = self.solve_pgw(Cx, Cy, p, q)

        q_e = gamma.sum(axis=1)

        # Barycentric projection of Y onto the reference support.
        Y_proj = np.zeros((len(X), Y.shape[1]), dtype=np.float64)
        valid = q_e > 1e-15
        if np.any(valid):
            Y_proj[valid] = (gamma[valid] @ Y) / q_e[valid, None]

        # Eq. (24): compare projected-target geometry against reference
        # geometry, both through the SAME cost function and scale.
        K = self.cost_matrix(Y_proj) - Cx

        transported_mass = float(q_e.sum())
        gamma_c_abs = max(0.0, float(p.sum()) ** 2 - transported_mass ** 2)

        return {
            "K": K,
            "q_e": q_e,
            "gamma_c_abs": gamma_c_abs,
            "gamma": gamma,
            "Y_projected": Y_proj,
            "transported_mass": transported_mass,
        }

    # ------------------------------------------------------------------
    # Pairwise discrepancy  (Eq. 25)
    # ------------------------------------------------------------------

    def distance(self, emb1: dict, emb2: dict) -> float:
        """
        Guarantees: distance(e, e) == 0, symmetric, non-negative.
        """
        K1, K2 = emb1["K"], emb2["K"]
        if K1.shape != K2.shape:
            raise ValueError("Embeddings use different references")

        q1, q2 = emb1["q_e"], emb2["q_e"]
        q12 = np.minimum(q1, q2)
        total_q12 = float(q12.sum())
        t = total_q12  # match expected variable name for penalties

        raw_geom = float(q12 @ ((K1 - K2) ** 2) @ q12)
        geometric = raw_geom / (t ** 2) if t > 1e-12 else raw_geom

        penalty_1 = self.distance_lambda * (
            float(emb1["q_e"].sum()) ** 2
            + float(emb2["q_e"].sum()) ** 2
            - 2 * t ** 2
        )
        penalty_2 = self.distance_lambda * abs(
            emb1["gamma_c_abs"] - emb2["gamma_c_abs"]
        )

        return max(0.0, geometric + penalty_1 + penalty_2)

    # ------------------------------------------------------------------
    # Fixed-length vector  ->  approximate O(K) retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def embedding_vector(emb, reference_mass=None) -> np.ndarray:
        """
        Flattened geometric embedding for KD-tree / FAISS retrieval.

        Replaces the pair-dependent weight min(q1,q2) with the fixed
        reference marginal, so this is an APPROXIMATION of distance().
        Use it to shortlist candidates, then rerank with distance().
        """
        K = np.asarray(emb["K"], dtype=np.float64)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("K must be square")
        if reference_mass is None:
            reference_mass = LPGW.uniform_mass(K.shape[0])
        reference_mass = LPGW._normalize_mass(reference_mass, K.shape[0])
        W = np.sqrt(np.outer(reference_mass, reference_mass))
        return (W * K).ravel()

    # ------------------------------------------------------------------
    # Diagnostics  -- run these before trusting any result
    # ------------------------------------------------------------------

    def diagnose(self, reference, segments, n_probe: int = 12) -> dict:
        """
        Verify the PGW problem is actually well posed.

        Prints, and returns, the handful of numbers that distinguish
        'working' from 'silently degenerate'.
        """
        embs = [self.embed(reference, s) for s in list(segments)[:n_probe]]

        masses = np.array([e["transported_mass"] for e in embs])
        creations = np.array([e["gamma_c_abs"] for e in embs])
        self_d = self.distance(embs[0], embs[0])

        D = np.array([[self.distance(a, b) for b in embs] for a in embs])
        row, col = D.mean(1, keepdims=True), D.mean(0, keepdims=True)
        resid = D - row - col + D.mean()
        additive = 1.0 - (resid.var() / D.var()) if D.var() > 0 else 1.0
        n_unique = len(np.unique(D.argmin(axis=1)))

        out = {
            "cost_scale": self.cost_scale,
            "transported_mass_mean": float(masses.mean()),
            "transported_mass_min": float(masses.min()),
            "gamma_c_abs_mean": float(creations.mean()),
            "self_distance": float(self_d),
            "distance_median": float(np.median(D[D > 0])) if np.any(D > 0) else 0.0,
            "additive_variance_explained": float(additive),
            "unique_argmins": int(n_unique),
            "n_probe": len(embs),
        }

        print("=" * 64)
        print("LPGW DIAGNOSTICS")
        print("=" * 64)
        print(f"  cost_scale                 : {out['cost_scale']}")
        print(f"  transported_mass (mean/min): {out['transported_mass_mean']:.6f} / "
              f"{out['transported_mass_min']:.6f}")
        if out["transported_mass_min"] > 0.999:
            print("    >> FAIL: full mass always transported. Partial transport is")
            print("       INACTIVE and every lambda term in distance() is zero.")
            print("       Lower lambdaa (try 0.05, 0.01) or re-check cost_scale.")
        else:
            print("    >> ok: mass is being discarded, PGW is active.")
        print(f"  gamma_c_abs (mean)         : {out['gamma_c_abs_mean']:.6e}")
        print(f"  distance(e, e)             : {out['self_distance']:.3e}")
        if out["self_distance"] > 1e-10:
            print("    >> FAIL: d(Y,Y) != 0. distance() is not a metric.")
        print(f"  median pairwise distance   : {out['distance_median']:.3e}")
        if 0 < out["distance_median"] < 1e-6:
            print("    >> WARN: distances collapsed near zero; cost_scale suspect.")
        print(f"  additive (row+col) variance: {out['additive_variance_explained']:.3f}")
        if out["additive_variance_explained"] > 0.9:
            print("    >> FAIL: matrix is row+column structure. argmin is degenerate.")
        print(f"  unique argmins             : {out['unique_argmins']} / {out['n_probe']}")
        print("=" * 64)
        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def embed_many(self, reference, segments) -> list[dict]:
        return [self.embed(reference, seg) for seg in segments]

    def distance_matrix(self, embeddings_q, embeddings_r) -> np.ndarray:
        D = np.zeros((len(embeddings_q), len(embeddings_r)), dtype=np.float64)
        for i, e1 in enumerate(embeddings_q):
            for j, e2 in enumerate(embeddings_r):
                D[i, j] = self.distance(e1, e2)
        return D