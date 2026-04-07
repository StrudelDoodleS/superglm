"""B-spline basis with optional SSP reparametrisation.

Knots are penalised via P-spline (Eilers & Marx, 1996), so 15-20 interior
knots is a safe default. More knots gives the penalty more flexibility to
capture the shape — it will not cause overfitting because the second-difference
penalty controls smoothness, not the knot count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline as BSpl

from superglm.types import GroupInfo, LambdaPolicy, LinearConstraintSet, TensorMarginalInfo

if TYPE_CHECKING:
    from superglm.solvers.scop import SCOPSolverReparam


def _weighted_quantile_knots(x: NDArray, n_knots: int, alpha: float) -> NDArray:
    """Compute interior knots via weighted quantiles of unique values.

    Parameters
    ----------
    x : 1-D array
        Covariate values (already clipped to boundary if needed).
    n_knots : int
        Desired number of interior knots.
    alpha : float
        Tempering exponent.  ``alpha=0`` gives equal weight to every
        unique value (same as ``"quantile"``).  Higher alpha concentrates
        knots where more rows occur.

    Returns
    -------
    interior : 1-D array
        Unique interior knot positions (may be fewer than ``n_knots``
        if ties collapse them).
    """
    ux, counts = np.unique(x, return_counts=True)
    if len(ux) < 2:
        return ux
    w = counts.astype(np.float64) ** alpha
    cw = np.cumsum(w)
    # CDF mapped to [0, 1]: cdf[0]=0, cdf[-1]=1, intermediate
    # proportional to cumulative weight.  For alpha=0 (equal weights)
    # this gives cdf[i]=i/(N-1), matching np.percentile exactly.
    denom = cw[-1] - w[0]
    if denom <= 0:
        return ux[:1]
    cdf = (cw - w[0]) / denom
    probs = np.linspace(0, 1, n_knots + 2)[1:-1]
    raw = np.interp(probs, cdf, ux)
    return np.unique(raw)


class _SplineBase:
    """Base class for all spline feature specs.

    Subclasses must implement ``_build_penalty()`` and may override
    ``_apply_constraints()`` to add boundary constraints.

    Basis capability attributes (override in subclasses):
        _penalty_semantics : how the penalty is computed
        _max_penalty_order : static upper bound on m (None = K-dependent)
        _multi_m_supported : whether tuple m is allowed
        _select_supported : whether select=True is allowed at all
        _tensor_supported : whether tensor marginal ingredients are available

    Parameters
    ----------
    n_knots : int
        Number of interior knots. 10 is a good default balancing
        flexibility and refit stability. The second-difference penalty
        controls smoothness, so more knots won't overfit — but fewer
        knots reduce curve drift between refitting periods.
    degree : int
        B-spline polynomial degree. 3 (cubic, default) gives C2 smooth
        curves and is the standard choice. 1 (linear) and 2 (quadratic)
        are acceptable alternatives. Values above 3 are not advised —
        numerical issues increase with no practical benefit.
    knot_strategy : str
        "uniform" (default) spaces knots evenly across the data range.
        "quantile" places knots at quantiles of ``unique(x)`` (mgcv
        convention — resistant to ties). "quantile_rows" places knots
        at quantiles of all rows (``pd.qcut``-style — more knots in
        dense regions). "quantile_tempered" places knots via weighted
        quantiles of ``unique(x)`` with weights ``count^knot_alpha``
        (density-weighted support quantiles; ``alpha=0`` recovers
        ``"quantile"``, higher values concentrate knots in dense regions).
    penalty : str
        "ssp" enables SSP reparametrisation, "none" disables it.
    knots : array-like or None
        Explicit interior knot positions. When provided, ``n_knots`` and
        ``knot_strategy`` are ignored. Use this for reproducibility
        across refits with different data.
    discrete : bool or None
        If True, discretize the covariate into ``n_bins`` equal-width bins
        at fit time, reducing matrix algebra from O(n) to O(n_bins).
        If None (default), defers to the model-level ``discrete`` setting.
    n_bins : int or None
        Number of bins for discretization. Only used when
        ``discrete=True``. If None (default), defers to the model-level
        ``n_bins`` setting (which defaults to 256).
    extrapolation : {"clip", "extend", "error"}
        Prediction-time behavior outside the training range. ``"clip"``
        (default) freezes the spline at the boundary value. ``"extend"``
        evaluates the spline basis outside the training range using its
        native continuation. ``"error"`` raises on out-of-range values.
    """

    # ── Basis capability metadata (override in subclasses) ────────
    _penalty_semantics: str = "difference"
    _max_penalty_order: int | None = None  # None = K-dependent (phase 2)
    _multi_m_supported: bool = True
    _select_supported: bool = True
    _tensor_supported: bool = True

    def _select_compatible(self, m_orders: tuple[int, ...]) -> bool:
        """Whether select=True is supported with these m orders.

        Default (BS): supported when max(m_orders) <= 2.
        Subclasses override for different null-space structures.
        """
        if not self._select_supported:
            return False
        return max(m_orders) <= 2

    # ── Validation ────────────────────────────────────────────────

    def _validate_m_orders(self) -> None:
        """Phase 1: static m validation. Called from __init__."""
        if len(self._m_orders) > 1 and not self._multi_m_supported:
            raise NotImplementedError(
                f"{type(self).__name__} does not support multi-order penalties "
                f"(m tuple). Use a single m value."
            )
        if self._max_penalty_order is not None:
            for o in self._m_orders:
                if o > self._max_penalty_order:
                    raise ValueError(
                        f"{type(self).__name__} supports penalty orders "
                        f"up to {self._max_penalty_order}, got m={o}."
                    )

    def _validate_m_orders_build(self) -> None:
        """Phase 2: dimension-dependent m validation. Called after knot placement."""
        for o in self._m_orders:
            if o >= self._n_basis:
                raise ValueError(
                    f"Penalty order m={o} requires at least {o + 1} basis functions, "
                    f"but this spline has n_basis={self._n_basis}. "
                    f"Increase n_knots or reduce m."
                )

    def _validate_select(self) -> None:
        """Phase 1: static select validation. Called from __init__."""
        if not self.select:
            return
        if not self._select_compatible(self._m_orders):
            if not self._select_supported:
                raise NotImplementedError(
                    f"select=True is not supported for {type(self).__name__}."
                )
            raise NotImplementedError(
                f"select=True with m={self._m_orders} is not supported for "
                f"{type(self).__name__}. "
                f"This is a current capability policy, not a mathematical impossibility."
            )

    def __init__(
        self,
        n_knots: int = 10,
        degree: int = 3,
        knot_strategy: str = "uniform",
        penalty: str = "ssp",
        knots: ArrayLike | None = None,
        discrete: bool | None = None,
        n_bins: int | None = None,
        extrapolation: str = "clip",
        boundary: tuple[float, float] | None = None,
        knot_alpha: float = 0.2,
        select: bool = False,
        monotone: str | None = None,
        monotone_mode: str = "postfit",
        m: int | tuple[int, ...] = 2,
        lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
    ):
        if monotone is not None and monotone not in ("increasing", "decreasing"):
            raise ValueError(
                f"monotone must be None, 'increasing', or 'decreasing', got {monotone!r}"
            )
        if monotone_mode not in ("postfit", "fit"):
            raise ValueError(f"monotone_mode must be 'postfit' or 'fit', got {monotone_mode!r}")
        self.monotone = monotone
        self.monotone_mode = monotone_mode

        self.select = select

        # Derivative penalty orders
        self._m_orders = (m,) if isinstance(m, int) else tuple(m)
        if not all(isinstance(o, int) and o >= 1 for o in self._m_orders):
            raise ValueError(f"m must contain positive integers, got {m}")
        self._validate_m_orders()
        self._validate_select()
        if knots is not None:
            knots = np.asarray(knots, dtype=np.float64).ravel()
            if knots.ndim != 1 or len(knots) < 1:
                raise ValueError("knots must be a non-empty 1D array")
            if not np.all(np.diff(knots) > 0):
                raise ValueError("knots must be strictly increasing")
            self.n_knots = len(knots)
            self._explicit_knots = knots
        else:
            self.n_knots = n_knots
            self._explicit_knots = None

        self.degree = degree
        self.knot_strategy = knot_strategy
        self.penalty = penalty
        # self.select already set above (before validation)
        self.discrete = discrete
        self.n_bins = n_bins
        if extrapolation not in {"clip", "extend", "error"}:
            raise ValueError(
                f"extrapolation must be one of ('clip', 'extend', 'error'), got {extrapolation!r}"
            )
        self.extrapolation = extrapolation
        self.knot_alpha = knot_alpha

        if boundary is not None:
            lo_b, hi_b = float(boundary[0]), float(boundary[1])
            if lo_b >= hi_b:
                raise ValueError(f"boundary must satisfy lo < hi, got boundary=({lo_b}, {hi_b})")
            if self._explicit_knots is not None:
                if self._explicit_knots[0] <= lo_b or self._explicit_knots[-1] >= hi_b:
                    raise ValueError(
                        f"explicit knots must lie strictly inside boundary=({lo_b}, {hi_b}), "
                        f"got knots in [{self._explicit_knots[0]}, {self._explicit_knots[-1]}]"
                    )
            self._explicit_boundary: tuple[float, float] | None = (lo_b, hi_b)
        else:
            self._explicit_boundary = None

        # State set during build()
        self._knots: NDArray = np.array([])
        self._n_basis: int = 0
        self._lo: float = 0.0
        self._hi: float = 1.0
        self._knot_strategy_actual: str = knot_strategy
        self._R_inv: NDArray | None = None
        self._interaction_projection: NDArray | None = None
        self._basis_lo: NDArray | None = None
        self._basis_hi: NDArray | None = None
        self._basis_d1_lo: NDArray | None = None
        self._basis_d1_hi: NDArray | None = None

        # select=True state (set during _eigendecompose_select)
        self._U_null: NDArray | None = None
        self._U_range: NDArray | None = None
        self._omega_range: NDArray | None = None

        # Multi-m penalty components (set during build / build_knots_and_penalty)
        self._penalty_components: list[tuple[str, NDArray]] | None = None

        # Lambda policy (user-specified)
        self._lambda_policy = lambda_policy

    def _prepare_eval_points(self, x: NDArray) -> tuple[NDArray, bool]:
        """Apply the configured extrapolation policy for basis evaluation."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if self.extrapolation == "clip":
            return np.clip(x, self._lo, self._hi), False
        if self.extrapolation == "extend":
            return x, True

        # extrapolation == "error"
        scale = max(1.0, abs(self._lo), abs(self._hi), abs(self._hi - self._lo))
        tol = 1e-12 * scale
        lo_mask = x < (self._lo - tol)
        hi_mask = x > (self._hi + tol)
        if np.any(lo_mask) or np.any(hi_mask):
            raise ValueError(
                f"Spline received values outside training range "
                f"[{self._lo:.6g}, {self._hi:.6g}] with extrapolation='error'."
            )
        return x, False

    def _basis_matrix(self, x: NDArray):
        """Evaluate the raw B-spline basis under the extrapolation policy."""
        x_eval, extrapolate = self._prepare_eval_points(x)
        return BSpl.design_matrix(x_eval, self._knots, self.degree, extrapolate=extrapolate)

    def _raw_basis_matrix(self, x: NDArray) -> NDArray:
        """Evaluate the raw (pre-projection) basis at points clipped to training range.

        Returns a dense ``(n, n_basis)`` array.  All code outside spline.py
        that needs to evaluate a spline's basis should call this method
        rather than constructing a ``BSpline.design_matrix`` directly, so
        that non-B-spline subclasses (e.g. :class:`CardinalCRSpline`) get
        their own evaluation strategy.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        x_clip = np.clip(x, self._lo, self._hi)
        return BSpl.design_matrix(x_clip, self._knots, self.degree, extrapolate=False).toarray()

    def _basis_value_and_slope_at(self, x0: float) -> tuple[NDArray, NDArray]:
        """Return the raw basis row and its first derivative at ``x0``."""
        basis = BSpl.design_matrix(
            np.array([x0], dtype=np.float64), self._knots, self.degree, extrapolate=False
        ).toarray()[0]
        slope = np.zeros(self._n_basis)
        for j in range(self._n_basis):
            c = np.zeros(self._n_basis)
            c[j] = 1.0
            slope[j] = BSpl(self._knots, c, self.degree)(x0, nu=1)
        return basis, slope

    def _boundary_linear_rows(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Cache basis value/slope rows for linear continuation at the boundaries."""
        if self._basis_lo is None or self._basis_d1_lo is None:
            self._basis_lo, self._basis_d1_lo = self._basis_value_and_slope_at(self._lo)
        if self._basis_hi is None or self._basis_d1_hi is None:
            self._basis_hi, self._basis_d1_hi = self._basis_value_and_slope_at(self._hi)
        return self._basis_lo, self._basis_d1_lo, self._basis_hi, self._basis_d1_hi

    def _linear_tail_basis_matrix(self, x: NDArray):
        """Evaluate the raw basis with explicit linear continuation outside the fit range."""
        x = np.asarray(x, dtype=np.float64).ravel()
        lo_mask = x < self._lo
        hi_mask = x > self._hi
        mid_mask = ~(lo_mask | hi_mask)

        rows = np.zeros((len(x), self._n_basis))
        if np.any(mid_mask):
            rows[mid_mask] = BSpl.design_matrix(
                x[mid_mask], self._knots, self.degree, extrapolate=False
            ).toarray()

        if np.any(lo_mask) or np.any(hi_mask):
            basis_lo, slope_lo, basis_hi, slope_hi = self._boundary_linear_rows()
            if np.any(lo_mask):
                rows[lo_mask] = basis_lo + (x[lo_mask, None] - self._lo) * slope_lo
            if np.any(hi_mask):
                rows[hi_mask] = basis_hi + (x[hi_mask, None] - self._hi) * slope_hi

        return sp.csr_matrix(rows)

    def _place_knots(self, x: NDArray) -> None:
        """Place interior knots and build the full knot vector."""
        if self._explicit_boundary is not None:
            self._lo, self._hi = self._explicit_boundary
        else:
            self._lo, self._hi = float(x.min()), float(x.max())
        self._basis_lo = None
        self._basis_hi = None
        self._basis_d1_lo = None
        self._basis_d1_hi = None

        if self._explicit_knots is not None:
            interior = self._explicit_knots
            self._knot_strategy_actual = "explicit"
        elif self.knot_strategy in ("quantile", "quantile_rows", "quantile_tempered"):
            # When boundary is frozen, restrict to [lo, hi] so quantile
            # knots cannot land outside the boundary.
            x_q = x[(x >= self._lo) & (x <= self._hi)] if self._explicit_boundary is not None else x
            if len(x_q) == 0:
                x_q = np.array([self._lo, self._hi])
            if self.knot_strategy == "quantile_tempered":
                interior = _weighted_quantile_knots(x_q, self.n_knots, self.knot_alpha)
            else:
                probs = np.linspace(0, 100, self.n_knots + 2)[1:-1]
                if self.knot_strategy == "quantile":
                    source = np.unique(x_q)
                else:
                    source = x_q
                interior = np.unique(np.percentile(source, probs))
            if len(interior) < self.n_knots:
                interior = np.linspace(self._lo, self._hi, self.n_knots + 2)[1:-1]
                self._knot_strategy_actual = "uniform"
            else:
                self._knot_strategy_actual = self.knot_strategy
        else:  # "uniform" (default)
            interior = np.linspace(self._lo, self._hi, self.n_knots + 2)[1:-1]
            self._knot_strategy_actual = "uniform"

        self._assemble_knot_vector(interior)

    def _assemble_knot_vector(self, interior: NDArray) -> None:
        """Build the full knot vector from interior knots.

        Default: clamped (repeated-end) construction.  Subclasses may
        override for open knot vectors.
        """
        pad = (self._hi - self._lo) * 1e-6
        self._knots = np.concatenate(
            [
                np.repeat(self._lo - pad, self.degree + 1),
                interior,
                np.repeat(self._hi + pad, self.degree + 1),
            ]
        )
        self._n_basis = len(self._knots) - self.degree - 1

    def __repr__(self) -> str:
        cls = type(self).__name__
        parts = [f"n_knots={self.n_knots}"]
        if self.select:
            parts.append("select=True")
        if self.degree != 3:
            parts.append(f"degree={self.degree}")
        return f"{cls}({', '.join(parts)})"

    def _build_penalty(self) -> NDArray:
        """Return (K, K) penalty matrix. Subclasses must implement."""
        raise NotImplementedError

    @property
    def fitted_knots(self) -> NDArray | None:
        """Interior knot locations from the fitted spline, or None before fit.

        These are the data-driven (or explicit) interior knot positions,
        excluding the boundary knots.  After fitting, these are frozen and
        reused on every subsequent ``transform()`` / ``predict()`` call.
        Pass them back via ``Spline(knots=..., boundary=...)`` to
        guarantee identical placement *and* boundary on a refit with
        different data.
        """
        if self._n_basis == 0:
            return None
        return self._knots[self.degree + 1 : -(self.degree + 1)].copy()

    @property
    def fitted_boundary(self) -> tuple[float, float] | None:
        """Training-range boundary ``(lo, hi)``, or None before fit."""
        if self._n_basis == 0:
            return None
        return (self._lo, self._hi)

    @property
    def absorbs_intercept(self) -> bool:
        """Whether the smooth should absorb the intercept-like direction.

        Centered smooth terms are constrained so that the unweighted
        mean contribution over the training data is zero, making the model
        intercept carry the constant part.  This is True by default for all
        spline kinds.

        When ``select=True`` the eigendecompose step removes the
        constant direction instead, so identifiability is skipped.
        """
        return not self.select

    def _build_multi_m_components(
        self, x: NDArray, B: Any, final_projection: NDArray | None
    ) -> list[tuple[str, NDArray]]:
        """Build per-order penalty components projected through the same constraints.

        Each per-order penalty is projected through the same constraint Z
        and identifiability projection that was used for the summed penalty.
        The constraint projections are basis-geometric (independent of the
        specific penalty values), so they can be reused for each order.
        """
        components: list[tuple[str, NDArray]] = []
        for order in self._m_orders:
            omega_raw = self._build_penalty_for_order(order)
            # Apply constraint projection (Z from natural boundary etc.)
            _, omega_c, _, constraint_proj = self._apply_constraints(None, omega_raw)
            # Apply identifiability projection
            omega_c, _, _ = self._apply_identifiability(x, omega_c, constraint_proj)
            components.append((f"d{order}", omega_c))
        return components

    def _apply_constraints(self, B, omega: NDArray) -> tuple[Any, NDArray, int, NDArray | None]:
        """Apply boundary constraints. Returns (B, omega, n_cols, projection).

        Default: no constraints (identity).
        """
        return B, omega, self._n_basis, None

    def _identifiability_projection(
        self,
        x: NDArray,
        constraint_projection: NDArray | None,
    ) -> NDArray | None:
        """Compute the projection that removes the intercept-confounded direction.

        Returns the (possibly composed) projection from the raw basis
        space to the identified+constrained space.  For BS this is
        ``(K, K-1)``; for CR it is ``(K, K-3)`` (natural + identifiability).

        Used by both the standard ``_apply_identifiability`` and the
        ``select=True`` path (where the main effect handles constant
        removal via eigendecomposition, but interactions still need the
        identified projection to avoid rank deficiency).
        """
        n_cols = (
            constraint_projection.shape[1] if constraint_projection is not None else self._n_basis
        )
        if n_cols <= 1:
            return constraint_projection

        x = np.asarray(x, dtype=np.float64).ravel()
        support, inverse = np.unique(x, return_inverse=True)
        basis = self._basis_matrix(support).toarray()
        if constraint_projection is not None:
            basis = basis @ constraint_projection

        counts = np.bincount(inverse, minlength=len(support)).astype(np.float64)
        constraint = counts @ basis
        if np.linalg.norm(constraint) < 1e-12:
            return constraint_projection

        q, _ = np.linalg.qr(constraint.reshape(-1, 1), mode="complete")
        z = q[:, 1:]
        return z if constraint_projection is None else constraint_projection @ z

    def _apply_identifiability(
        self,
        x: NDArray,
        omega: NDArray,
        projection: NDArray | None,
    ) -> tuple[NDArray, int, NDArray | None]:
        """Remove the intercept-confounded smooth direction.

        The constraint ``sum_i B(x_i) @ beta = 0`` (unweighted over
        training observations) centres the smooth so that the mean
        contribution over the covariate distribution is zero.  The model
        intercept carries the constant part.  Exposure/weights are a
        likelihood concept and do not enter the identifiability constraint.
        """
        if not self.absorbs_intercept:
            return omega, omega.shape[0], projection

        projection_ident = self._identifiability_projection(x, projection)
        if projection_ident is projection:
            return omega, omega.shape[0], projection

        # Extract the local identifiability direction to transform omega.
        # projection_ident = projection @ z (or just z if projection is None).
        # We need z to compute omega_ident = z.T @ omega @ z.
        if projection is not None:
            # z = pinv(projection) @ projection_ident, but since projection
            # has orthonormal columns, pinv = projection.T
            z = projection.T @ projection_ident
        else:
            z = projection_ident
        omega_ident = z.T @ omega @ z
        return omega_ident, omega_ident.shape[0], projection_ident

    def _natural_constraint_null_space(self) -> NDArray:
        """Compute (K, K-2) null space of the natural boundary constraints.

        Builds a 2xK constraint matrix C where each row is the second
        derivative of each B-spline basis function evaluated at a boundary.
        The null space of C (via QR of C.T) gives the subspace satisfying
        f''(boundary) = 0.

        Uses ``BSpline.__call__(x, nu=2)`` (de Boor algorithm) rather than
        ``BSpline.derivative(2)`` to avoid the repeated-knot division error
        at clamped boundaries.
        """
        K = self._n_basis
        C = np.zeros((2, K))
        for j in range(K):
            c = np.zeros(K)
            c[j] = 1.0
            spl = BSpl(self._knots, c, self.degree)
            C[0, j] = spl(self._lo, nu=2)
            C[1, j] = spl(self._hi, nu=2)
        Q, _ = np.linalg.qr(C.T, mode="complete")
        return Q[:, 2:]  # (K, K-2)

    def _eigendecompose_select(self, omega_c: NDArray, Z: NDArray | None) -> None:
        """Eigendecompose the constrained penalty for select=True splitting.

        Populates ``_U_null``, ``_U_range``, ``_omega_range`` on self.
        Works for any spline kind whose constrained penalty has exactly
        2 null eigenvalues (constant + linear).

        Parameters
        ----------
        omega_c : (d, d) constrained penalty matrix
        Z : (K, d) constraint projection, or None if no constraints
        """
        eigvals, eigvecs = np.linalg.eigh(omega_c)
        null_mask = eigvals < 1e-10
        n_null = int(np.sum(null_mask))
        if n_null != 2:
            raise ValueError(
                f"select=True requires exactly 2 null eigenvalues in the "
                f"constrained penalty, got {n_null}. "
                f"Spline kind {type(self).__name__} may not support select=True."
            )

        U_null_raw = eigvecs[:, null_mask]  # (d, 2)
        U_range = eigvecs[:, ~null_mask]  # (d, d-2)
        omega_range = np.diag(eigvals[~null_mask])

        # Remove constant: ones_c = Z.T @ ones if Z else ones
        ones_c = (Z.T @ np.ones(self._n_basis)) if Z is not None else np.ones(omega_c.shape[0])
        ones_in_null = U_null_raw.T @ ones_c
        ones_in_null /= np.linalg.norm(ones_in_null)
        U_null_centered = U_null_raw - U_null_raw @ np.outer(ones_in_null, ones_in_null)
        u, s, _ = np.linalg.svd(U_null_centered, full_matrices=False)
        U_null_1d = u[:, :1]  # (d, 1)

        # Compose projections to raw basis space (K-dimensional)
        self._U_null = Z @ U_null_1d if Z is not None else U_null_1d  # (K, 1)
        self._U_range = Z @ U_range if Z is not None else U_range  # (K, n_range)
        self._omega_range = omega_range

    def _resolve_lambda_policies(self, info: GroupInfo) -> dict[str, LambdaPolicy] | None:
        """Resolve lambda_policy parameter into a per-component dict."""
        if self._lambda_policy is None:
            return None

        # Determine valid component names
        if info.penalty_components is not None:
            valid_names = {name for name, _ in info.penalty_components}
        else:
            # Single-penalty spline: use canonical name "wiggle"
            valid_names = {"wiggle"}

        if isinstance(self._lambda_policy, LambdaPolicy):
            return {name: self._lambda_policy for name in valid_names}

        # Dict: validate keys, fill unspecified with estimate
        policy_dict = self._lambda_policy
        unknown = set(policy_dict) - valid_names
        if unknown:
            raise ValueError(
                f"lambda_policy contains unknown component names: {unknown}. "
                f"Valid names: {sorted(valid_names)}"
            )
        result = {}
        for name in valid_names:
            result[name] = policy_dict.get(name, LambdaPolicy.estimate())
        return result

    def _build_select(self, x: NDArray, B) -> GroupInfo:
        """Build select=True GroupInfo with null + wiggle/per-order penalty components."""
        # Use the highest derivative order for eigendecomposition — it has
        # the most null eigenvalues (constant + linear for m=2).
        if len(self._m_orders) == 1:
            omega_for_eigen = self._build_penalty()
        else:
            max_order = max(self._m_orders)
            omega_for_eigen = self._build_penalty_for_order(max_order)
        _, omega_c, _, Z = self._apply_constraints(None, omega_for_eigen)

        # Store the full constraint + identifiability projection for
        # interactions (SplineCategorical).  The main effect handles
        # constant removal via eigendecomposition, but interactions
        # need the standard identified projection to stay full-rank.
        self._interaction_projection = self._identifiability_projection(x, Z)

        self._eigendecompose_select(omega_c, Z)

        n_null = 1
        n_range = self._U_range.shape[1]
        n_combined = n_null + n_range

        # Combined projection: [U_null | U_range]
        U_combined = np.hstack([self._U_null, self._U_range])  # (K, n_combined)
        # Constrained-space projections for per-order penalty mapping
        U_null_c = self._U_null if Z is None else np.linalg.lstsq(Z, self._U_null, rcond=None)[0]
        U_range_c = self._U_range if Z is None else np.linalg.lstsq(Z, self._U_range, rcond=None)[0]
        U_combined_c = np.hstack([U_null_c, U_range_c])  # (d, n_combined)

        # Null-space selection penalty (identity on null block, zero on range)
        omega_null = np.zeros((n_combined, n_combined))
        omega_null[:n_null, :n_null] = np.eye(n_null)

        components: list[tuple[str, np.ndarray]] = [("null", omega_null)]
        component_types: dict[str, str] = {"null": "selection"}

        if len(self._m_orders) == 1:
            # Single-m: one "wiggle" component (block-diagonal by construction)
            omega_wiggle = np.zeros((n_combined, n_combined))
            omega_wiggle[n_null:, n_null:] = self._omega_range
            components.append(("wiggle", omega_wiggle))
        else:
            # Multi-m: project each per-order constrained penalty into the
            # combined [U_null | U_range] basis.
            for order in self._m_orders:
                omega_raw_j = self._build_penalty_for_order(order)
                _, omega_c_j, _, _ = self._apply_constraints(None, omega_raw_j)
                omega_combined_j = U_combined_c.T @ omega_c_j @ U_combined_c
                components.append((f"d{order}", omega_combined_j))

        penalty_matrix = sum(omega for _, omega in components)

        info = GroupInfo(
            columns=B,
            n_cols=n_combined,
            penalty_matrix=penalty_matrix,
            reparametrize=True,
            penalized=True,
            projection=U_combined,
            penalty_components=components,
            component_types=component_types,
        )
        info.lambda_policies = self._resolve_lambda_policies(info)
        return info

    def build(
        self, x: NDArray, sample_weight: NDArray | None = None
    ) -> GroupInfo | list[GroupInfo]:
        """Build B-spline basis and penalty matrix."""
        if self.monotone is not None and self.monotone_mode == "fit":
            if not hasattr(self, "_build_monotone_constraints_raw") and not hasattr(
                self, "_build_scop_reparameterization"
            ):
                raise NotImplementedError(
                    f"{type(self).__name__} does not support "
                    f"monotone_mode='fit'. Use monotone_mode='postfit'."
                )
            if self.select:
                raise NotImplementedError(
                    "Monotone fit-time constraints are not supported with "
                    "select=True. Use select=False or monotone_mode='postfit'."
                )
        x = np.asarray(x, dtype=np.float64).ravel()
        self._place_knots(x)
        self._validate_m_orders_build()
        B = self._basis_matrix(x).tocsr()

        if self.select:
            return self._build_select(x, B)

        omega = self._build_penalty()

        # SCOP monotone path: bypass standard SSP, use SCAM-style centering
        if (
            self.monotone is not None
            and self.monotone_mode == "fit"
            and hasattr(self, "_build_scop_reparameterization")
        ):
            B_dense = B.toarray() if hasattr(B, "toarray") else B
            B_centered, S_scop, scop_reparam = self._build_scop_reparameterization(B_dense, omega)
            n_cols_scop = B_centered.shape[1]
            return GroupInfo(
                columns=B_centered,
                n_cols=n_cols_scop,
                penalty_matrix=S_scop,
                reparametrize=False,
                penalized=True,
                scop_reparameterization=scop_reparam,
                monotone_engine="scop",
            )

        B, omega, n_cols, projection = self._apply_constraints(B, omega)
        omega, n_cols, projection = self._apply_identifiability(x, omega, projection)
        self._interaction_projection = projection

        # Multi-m: build per-order penalty components through the same projection
        penalty_components = None
        if len(self._m_orders) > 1:
            penalty_components = self._build_multi_m_components(x, B, projection)
            # Summed penalty for R_inv
            omega = sum(om for _, om in penalty_components)

        # Monotone constraint emission: compose raw constraints through
        # identifiability projection into post-identifiability space.
        # NOTE: constraints are NOT solver-ready after build() — they become
        # solver-coordinate only after the DM builder composes with R_inv.
        constraints = None
        monotone_engine = None
        raw_to_solver_map = None
        if self.monotone is not None and self.monotone_mode == "fit":
            cs_raw = self._build_monotone_constraints_raw()
            if projection is not None:
                constraints = cs_raw.compose(projection)
            else:
                constraints = cs_raw
            monotone_engine = "qp"
            raw_to_solver_map = projection

        info = GroupInfo(
            columns=B,
            n_cols=n_cols,
            penalty_matrix=omega,
            reparametrize=(self.penalty == "ssp"),
            projection=projection,
            penalty_components=penalty_components,
            constraints=constraints,
            monotone_engine=monotone_engine,
            raw_to_solver_map=raw_to_solver_map,
        )
        # Promote single-penalty spline to explicit component when lambda_policy is set
        if self._lambda_policy is not None and info.penalty_components is None:
            info.penalty_components = [("wiggle", info.penalty_matrix)]
            info.component_types = {"wiggle": "difference"}
        info.lambda_policies = self._resolve_lambda_policies(info)
        return info

    def build_knots_and_penalty(
        self, x: NDArray, sample_weight: NDArray | None = None
    ) -> tuple[NDArray, int, NDArray | None]:
        """Place knots and return penalty info, without building the full basis.

        Used by the discretization path to avoid the O(n) basis construction.
        Applies boundary constraints (NaturalSpline/CRS) and identifiability
        so the returned penalty and column count match the exact ``build()``
        path.

        For ``select=True``, also runs the eigendecompose step to populate
        ``_U_null``, ``_U_range``, ``_omega_range``.

        Returns
        -------
        omega : (n_cols, n_cols) penalty matrix (projected if constrained).
        n_cols : effective number of basis columns.
        projection : (K, n_cols) constraint projection, or None.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        self._place_knots(x)
        self._validate_m_orders_build()
        omega = self._build_penalty()
        _, omega_c, n_cols, projection = self._apply_constraints(None, omega)

        if self.select:
            Z = projection  # constraint projection (or None)
            # For multi-m, eigendecompose the highest-order penalty (most nulls)
            if len(self._m_orders) > 1:
                max_order = max(self._m_orders)
                omega_for_eigen = self._build_penalty_for_order(max_order)
                _, omega_c_eigen, _, _ = self._apply_constraints(None, omega_for_eigen)
                self._eigendecompose_select(omega_c_eigen, Z)
            else:
                self._eigendecompose_select(omega_c, Z)
            # Store constraint + identifiability projection for interactions
            self._interaction_projection = self._identifiability_projection(x, Z)
            return omega_c, n_cols, projection

        omega_c, n_cols, projection = self._apply_identifiability(x, omega_c, projection)
        self._interaction_projection = projection

        # Multi-m: store per-order components for the discrete dm_builder to pick up
        if len(self._m_orders) > 1:
            self._penalty_components = self._build_multi_m_components(x, None, projection)
            omega_c = sum(om for _, om in self._penalty_components)
        else:
            self._penalty_components = None

        return omega_c, n_cols, projection

    def transform(self, x: NDArray) -> NDArray:
        """Build design matrix using knots learned during build()."""
        B = self._basis_matrix(x).toarray()
        if hasattr(self, "_scop_Sigma") and self._scop_Sigma is not None:
            # SCOP monotone term: apply Sigma, drop constant column, center
            X_sigma = B @ self._scop_Sigma
            return X_sigma[:, 1:] - self._scop_col_means
        if self._R_inv is not None:
            B = B @ self._R_inv
        return B

    def set_reparametrisation(self, R_inv: NDArray) -> None:
        self._R_inv = R_inv

    def reconstruct(self, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
        x_grid = np.linspace(self._lo, self._hi, n_points)
        if hasattr(self, "_scop_Sigma") and self._scop_Sigma is not None:
            # SCOP term: beta is gamma_eff, reconstruct via centered design
            B_grid = self.transform(x_grid)
            log_rels = B_grid @ beta
            beta_orig = beta
        else:
            beta_orig = self._R_inv @ beta if self._R_inv is not None else beta
            B_grid = self._basis_matrix(x_grid).toarray()
            log_rels = B_grid @ beta_orig
        return {
            "x": x_grid,
            "log_relativity": log_rels,
            "relativity": np.exp(log_rels),
            "knots_interior": self._knots[self.degree + 1 : -(self.degree + 1)],
            "coefficients_original": beta_orig,
        }

    def tensor_marginal_ingredients(self, x: NDArray) -> TensorMarginalInfo:
        """Compute marginal basis, penalty, and projection for tensor products.

        Must be called on an already-built spec (after ``build()`` or
        ``_place_knots()``).  Reuses the parent's knot vector, penalty
        type, and boundary constraints so that tensor marginals inherit
        the parent spline geometry.

        Returns a ``TensorMarginalInfo`` with the centered+constrained
        marginal basis, penalty, and a projection from the raw B-spline
        space to the effective (centered) space.
        """
        if not self._tensor_supported:
            raise NotImplementedError(
                f"{type(self).__name__} does not support tensor marginal ingredients. "
                f"Use kind='cr' or kind='ps' for tensor product interactions."
            )
        reasons: list[str] = []
        if self.select:
            reasons.append("select=True")
        if len(self._m_orders) > 1:
            reasons.append(f"m={self._m_orders}")
        if reasons:
            detail = " and ".join(reasons)
            raise NotImplementedError(
                f"Tensor interactions require single-penalty parent smooths, but "
                f"{type(self).__name__} was configured with {detail}. "
                "This matches the mgcv te()/ti() marginal-smooth contract."
            )
        x = np.asarray(x, dtype=np.float64).ravel()

        # 1. Raw basis
        B_raw = self._raw_basis_matrix(x)  # (n, K)

        # 2. Penalty in raw space (single-penalty, multi-m rejected above)
        omega = self._build_penalty()  # (K, K)

        # 3. Apply boundary constraints (e.g. natural f''=0)
        _, omega_c, _, Z = self._apply_constraints(None, omega)
        # Z is (K, K-c) or None

        # 4. Apply constraints to basis
        if Z is not None:
            B_c = B_raw @ Z  # (n, K-c)
        else:
            B_c = B_raw  # (n, K)

        # 5. Center (remove intercept direction)
        c = B_c.sum(axis=0)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-12:
            # Degenerate: no centering needed
            P_ident = np.eye(B_c.shape[1])
        else:
            c = c / c_norm
            q, _ = np.linalg.qr(c[:, None], mode="complete")
            P_ident = q[:, 1:]  # (K-c, K-c-1)

        # 6. Centered basis and penalty
        B_centered = B_c @ P_ident  # (n, K_eff)
        omega_centered = P_ident.T @ omega_c @ P_ident  # (K_eff, K_eff)

        # 7. Full projection: raw → centered+constrained
        if Z is not None:
            projection = Z @ P_ident  # (K, K_eff)
        else:
            projection = P_ident  # (K, K_eff)

        K_eff = projection.shape[1]

        return TensorMarginalInfo(
            basis=B_centered,
            penalty=omega_centered,
            knots=self._knots.copy(),
            lo=self._lo,
            hi=self._hi,
            projection=projection,
            K_eff=K_eff,
            degree=self.degree,
        )


class _BSplineBase(_SplineBase):
    """Shared base for unconstrained B-spline smooths (PSpline, BSplineSmooth).

    Provides the open knot-vector assembly used by both PSpline and
    BSplineSmooth. CubicRegressionSpline has its own clamped knot assembly
    and inherits from _SplineBase directly.
    """

    def _assemble_knot_vector(self, interior: NDArray) -> None:
        """Open knot vector with 0.001*range edge padding.

        Instead of repeating the boundary knot ``degree + 1`` times
        (clamped construction), extend the knot vector beyond the data
        range at regular spacing.  The internal effective boundary is
        expanded by ``0.001 * range`` on each side.  This keeps the
        spline basis open while leaving the public fitted boundary tied
        to the observed training range.

        The public boundary (``self._lo``, ``self._hi``) is unchanged,
        so ``fitted_boundary``, clipping, and extrapolation still refer
        to the data range.
        """
        xr = self._hi - self._lo
        lo_eff = self._lo - 0.001 * xr
        hi_eff = self._hi + 0.001 * xr

        # Interior knots bracketed by the effective boundary
        inner = np.concatenate([[lo_eff], interior, [hi_eff]])

        # Extension spacing: use the nearest interior gap at each end
        dx_lo = inner[1] - inner[0]
        dx_hi = inner[-1] - inner[-2]

        lower = lo_eff - dx_lo * np.arange(self.degree, 0, -1)
        upper = hi_eff + dx_hi * np.arange(1, self.degree + 1)

        self._knots = np.concatenate([lower, inner, upper])
        self._n_basis = len(self._knots) - self.degree - 1


class PSpline(_BSplineBase):
    """P-spline: B-spline basis + discrete-difference penalty.

    This is the concrete P-spline implementation. For the recommended
    public API, use :func:`Spline` which dispatches to ``PSpline``,
    ``NaturalSpline``, or ``CubicRegressionSpline`` based on ``kind``.

    The ``m`` parameter controls the discrete difference order(s) for the
    penalty (default 2, second-difference).

    Parameters
    ----------
    n_knots : int
        Number of interior knots.
    degree : int
        B-spline polynomial degree.
    knot_strategy : str
        "uniform" (default), "quantile", "quantile_rows", or
        "quantile_tempered".
    penalty : str
        "ssp" enables SSP reparametrisation, "none" disables it.
    select : bool
        If True, decompose the spline into null-space (linear) and
        range-space (wiggly) subgroups for three-way selection:
        nonlinear -> linear -> dropped.

        **Double penalty**: The null-space (linear) subgroup is
        penalised with a ridge penalty (``penalty_matrix=eye(1)``).
        With ``fit_reml()``, REML estimates separate lambdas for the
        linear and spline subgroups — driving the linear lambda to
        infinity effectively zeros the linear component (three-way
        selection: nonlinear -> linear -> dropped). See Wood (2011).

    knots : array-like or None
        Explicit interior knot positions.
    monotone : str or None
        Monotonicity constraint direction. ``None`` (default) means no
        constraint. ``"increasing"`` or ``"decreasing"`` requests
        monotone repair.
    monotone_mode : str
        ``"postfit"`` (default) applies isotonic regression after fitting.
        ``"fit"`` is reserved for future constrained IRLS (not yet implemented).
    """

    def __init__(
        self,
        n_knots: int = 10,
        degree: int = 3,
        knot_strategy: str = "uniform",
        penalty: str = "ssp",
        select: bool = False,
        knots: ArrayLike | None = None,
        discrete: bool | None = None,
        n_bins: int | None = None,
        extrapolation: str = "clip",
        boundary: tuple[float, float] | None = None,
        knot_alpha: float = 0.2,
        monotone: str | None = None,
        monotone_mode: str = "postfit",
        m: int | tuple[int, ...] = 2,
        lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
    ):
        super().__init__(
            n_knots,
            degree,
            knot_strategy,
            penalty,
            knots,
            discrete,
            n_bins,
            extrapolation,
            boundary,
            knot_alpha,
            select=select,
            monotone=monotone,
            monotone_mode=monotone_mode,
            m=m,
            lambda_policy=lambda_policy,
        )

    def _build_scop_reparameterization(
        self, B: NDArray, omega: NDArray
    ) -> tuple[NDArray, NDArray, SCOPSolverReparam]:
        """Build SCOP reparameterization for monotone PSpline.

        Returns (B_centered, S_scop, scop_reparam) where:
        - B_centered: design matrix with SCAM-style centering (n, q_eff)
        - S_scop: SCOP penalty matrix (first-difference on beta), (q_eff, q_eff)
        - scop_reparam: solver-space SCOPSolverReparam
        """
        from superglm.solvers.scop import build_scop_reparam, build_scop_solver_reparam

        q = self._n_basis
        reparam = build_scop_reparam(q, direction=self.monotone)

        # SCAM-style centering: B @ Sigma, drop constant column, center remaining
        X_sigma = B @ reparam.Sigma
        # Drop column 0 (constant level absorbed into intercept)
        col_means = X_sigma[:, 1:].mean(axis=0)
        X_centered = X_sigma[:, 1:] - col_means

        # Store Sigma and centering for use in transform()
        self._scop_Sigma = reparam.Sigma
        self._scop_col_means = col_means

        # Build solver-space reparam
        solver_reparam = build_scop_solver_reparam(q, direction=self.monotone)

        # SCOP penalty in solver space
        S_scop = solver_reparam.penalty_matrix()

        return X_centered, S_scop, solver_reparam

    def _build_penalty_for_order(self, order: int) -> NDArray:
        if order >= self._n_basis:
            raise ValueError(
                f"Difference order {order} >= n_basis {self._n_basis}. "
                f"Increase n_knots or reduce m."
            )
        Dm = np.diff(np.eye(self._n_basis), n=order, axis=0)
        return Dm.T @ Dm

    def _build_penalty(self) -> NDArray:
        return self._build_penalty_for_order(self._m_orders[0])


# Backward-compatible alias — existing code using BasisSpline still works.
BasisSpline = PSpline


class BSplineSmooth(_BSplineBase):
    """B-spline smooth with integrated-derivative penalty.

    Same raw B-spline basis as ``PSpline``, but penalised via the
    *integrated squared m-th derivative* rather than the discrete
    difference penalty.  This is the analogue of mgcv's ``"bs"`` smooth.

    The penalty matrix is::

        omega_ij = int B_i^(m)(x) B_j^(m)(x) dx

    computed by Gauss--Legendre quadrature over each knot span.  ``m``
    is the integrated derivative order (default 2 = integrated
    second-derivative penalty).  Compare with ``PSpline`` where ``m``
    is the finite-difference order on the coefficient vector.

    Cubic by default (``degree=3``) but general degree is allowed.

    Parameters
    ----------
    n_knots : int
        Number of interior knots.
    degree : int
        B-spline polynomial degree.
    knot_strategy : str
        ``"uniform"`` or ``"quantile"``.
    penalty : str
        ``"ssp"`` enables SSP reparametrisation, ``"none"`` for raw.
    select : bool
        If True, add double-penalty shrinkage (null + range space).
    knots : array-like or None
        Explicit interior knot positions.
    monotone : str or None
        Monotonicity constraint direction.
    monotone_mode : str
        ``"postfit"`` (default) applies isotonic regression after fitting.
    m : int or tuple of int
        Integrated derivative order(s) for the penalty.
    lambda_policy : LambdaPolicy or dict or None
        Per-component lambda control.
    """

    _penalty_semantics = "integrated_derivative"
    _max_penalty_order: int | None = None  # validated dynamically in _build_penalty_for_order

    def __init__(
        self,
        n_knots: int = 10,
        degree: int = 3,
        knot_strategy: str = "uniform",
        penalty: str = "ssp",
        select: bool = False,
        knots: ArrayLike | None = None,
        discrete: bool | None = None,
        n_bins: int | None = None,
        extrapolation: str = "clip",
        boundary: tuple[float, float] | None = None,
        knot_alpha: float = 0.2,
        monotone: str | None = None,
        monotone_mode: str = "postfit",
        m: int | tuple[int, ...] = 2,
        lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
    ):
        super().__init__(
            n_knots,
            degree,
            knot_strategy,
            penalty,
            knots,
            discrete,
            n_bins,
            extrapolation,
            boundary,
            knot_alpha,
            select=select,
            monotone=monotone,
            monotone_mode=monotone_mode,
            m=m,
            lambda_policy=lambda_policy,
        )

    def _build_penalty_for_order(self, order: int) -> NDArray:
        """Integrated f^(m) squared penalty via Gauss-Legendre quadrature.

        omega_ij = int B_i^(m)(x) B_j^(m)(x) dx

        For degree-d B-splines, the m-th derivative is degree (d-m), so
        the product is degree 2*(d-m).  Quadrature with max(m+1, d)
        points is sufficient.
        """
        if order > self.degree:
            raise ValueError(
                f"Derivative order {order} > spline degree {self.degree}. "
                f"Integrated-derivative penalty requires order <= degree."
            )
        K = self._n_basis
        unique_knots = np.unique(self._knots)
        omega = np.zeros((K, K))
        n_quad = max(order + 1, self.degree)

        for a, b in zip(unique_knots[:-1], unique_knots[1:]):
            if b - a < 1e-15:
                continue
            xi, wi = np.polynomial.legendre.leggauss(n_quad)
            x_q = 0.5 * (b - a) * xi + 0.5 * (a + b)
            w_q = 0.5 * (b - a) * wi

            Dm_q = np.zeros((len(x_q), K))
            for j in range(K):
                c = np.zeros(K)
                c[j] = 1.0
                spl = BSpl(self._knots, c, self.degree)
                Dm_q[:, j] = spl(x_q, nu=order)

            omega += Dm_q.T @ (Dm_q * w_q[:, None])

        return omega

    def _build_monotone_constraints_raw(self) -> LinearConstraintSet:
        """Build monotone constraints on raw B-spline coefficients.

        For monotone increasing: D @ beta_raw >= 0 where D is the
        first-difference matrix (beta_{i+1} - beta_i >= 0).
        For monotone decreasing: -D @ beta_raw >= 0.

        Returns constraints on K raw (pre-projection) coefficients.
        """
        K = self._n_basis
        # First-difference matrix: row i = e_{i+1} - e_i
        D = np.diff(np.eye(K), axis=0)
        if self.monotone == "decreasing":
            D = -D
        return LinearConstraintSet(A=D, b=np.zeros(K - 1))

    def _build_penalty(self) -> NDArray:
        return self._build_penalty_for_order(self._m_orders[0])


class NaturalSpline(_SplineBase):
    """Natural P-spline: f''=0 at boundaries, linear tails.

    Applies natural boundary constraints: f''(boundary) = 0 at both
    ends. The underlying basis therefore has linear tails beyond the
    boundary knots, preventing the tail explosions common with
    unconstrained B-splines. Prediction behavior outside the training
    range is then controlled by ``extrapolation``: ``"clip"``
    (default) freezes at the boundary, while ``"extend"`` exposes the
    linear tails.

    Uses a second-difference penalty (like BS) rather than the
    integrated-f'' penalty of ``CubicRegressionSpline``.  The
    boundary constraints reduce the penalty null space to 1 dimension
    (constant only), so ``select=True`` is not supported — use
    ``kind="cr"`` or ``kind="ps"`` for double-penalty selection.
    """

    _select_supported = False

    def __init__(
        self,
        n_knots: int = 10,
        degree: int = 3,
        knot_strategy: str = "uniform",
        penalty: str = "ssp",
        select: bool = False,
        knots: ArrayLike | None = None,
        discrete: bool | None = None,
        n_bins: int | None = None,
        extrapolation: str = "clip",
        boundary: tuple[float, float] | None = None,
        knot_alpha: float = 0.2,
        m: int | tuple[int, ...] = 2,
        lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
    ):
        super().__init__(
            n_knots,
            degree,
            knot_strategy,
            penalty,
            knots,
            discrete,
            n_bins,
            extrapolation,
            boundary,
            knot_alpha,
            select=select,
            m=m,
            lambda_policy=lambda_policy,
        )
        self._Z: NDArray | None = None

    def _build_penalty_for_order(self, order: int) -> NDArray:
        if order >= self._n_basis:
            raise ValueError(
                f"Difference order {order} >= n_basis {self._n_basis}. "
                f"Increase n_knots or reduce m."
            )
        Dm = np.diff(np.eye(self._n_basis), n=order, axis=0)
        return Dm.T @ Dm

    def _build_penalty(self) -> NDArray:
        return self._build_penalty_for_order(self._m_orders[0])

    def _basis_matrix(self, x: NDArray):
        if self.extrapolation != "extend" or self.degree < 3:
            return super()._basis_matrix(x)
        return self._linear_tail_basis_matrix(x)

    def _apply_constraints(self, B, omega: NDArray) -> tuple[Any, NDArray, int, NDArray | None]:
        if self.degree < 3:
            return B, omega, self._n_basis, None
        Z = self._natural_constraint_null_space()
        self._Z = Z
        omega_nat = Z.T @ omega @ Z  # (K-2, K-2) projected penalty
        return B, omega_nat, self._n_basis - 2, Z


class CubicRegressionSpline(_SplineBase):
    """CR spline: integrated f'' squared penalty + natural boundary constraints.

    Compatible with the standard cubic regression spline construction
    used in GAM packages. Always cubic (degree=3).
    Natural boundary constraints (f''=0 at boundaries) are mandatory.
    The basis has linear tails, but default prediction still clips at
    the training boundary unless ``extrapolation="extend"`` is used.

    The penalty matrix is the wiggliness penalty: omega_ij = int B_i''(x) B_j''(x) dx,
    computed via Gauss-Legendre quadrature over each knot interval.

    Multi-order penalties (m tuple) are a SuperGLM extension, not strict
    mgcv ``bs="cr"`` parity.
    """

    _penalty_semantics = "integrated_derivative"
    _max_penalty_order = 3

    def _select_compatible(self, m_orders: tuple[int, ...]) -> bool:
        """CR: natural constraints always produce 2 null eigenvalues."""
        return True  # any m <= 3 works with select

    def __init__(
        self,
        n_knots: int = 10,
        knot_strategy: str = "uniform",
        penalty: str = "ssp",
        select: bool = False,
        knots: ArrayLike | None = None,
        discrete: bool | None = None,
        n_bins: int | None = None,
        extrapolation: str = "clip",
        boundary: tuple[float, float] | None = None,
        knot_alpha: float = 0.2,
        monotone: str | None = None,
        monotone_mode: str = "postfit",
        m: int | tuple[int, ...] = 2,
        lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
    ):
        super().__init__(
            n_knots,
            degree=3,
            knot_strategy=knot_strategy,
            penalty=penalty,
            knots=knots,
            discrete=discrete,
            n_bins=n_bins,
            extrapolation=extrapolation,
            boundary=boundary,
            knot_alpha=knot_alpha,
            select=select,
            monotone=monotone,
            monotone_mode=monotone_mode,
            m=m,
            lambda_policy=lambda_policy,
        )
        self._Z: NDArray | None = None

    def _assemble_knot_vector(self, interior: NDArray) -> None:
        """Clamped knot vector with exact boundary knots (no padding).

        CRS applies f''=0 constraints at the boundary, so the boundary
        knots must be exactly at the data range — not offset by an
        epsilon like the base-class default.
        """
        self._knots = np.concatenate(
            [
                np.repeat(self._lo, self.degree + 1),
                interior,
                np.repeat(self._hi, self.degree + 1),
            ]
        )
        self._n_basis = len(self._knots) - self.degree - 1

    def _build_penalty_for_order(self, order: int) -> NDArray:
        """Integrated f^(m) squared penalty via Gauss-Legendre quadrature.

        omega_ij = int B_i^(m)(x) B_j^(m)(x) dx

        For cubic B-splines, the m-th derivative is degree (3-m), so the
        product is degree 2*(3-m). Quadrature with max(m+1, 3) points
        is sufficient.
        """
        K = self._n_basis
        unique_knots = np.unique(self._knots)
        omega = np.zeros((K, K))
        n_quad = max(order + 1, 3)

        for a, b in zip(unique_knots[:-1], unique_knots[1:]):
            if b - a < 1e-15:
                continue
            xi, wi = np.polynomial.legendre.leggauss(n_quad)
            x_q = 0.5 * (b - a) * xi + 0.5 * (a + b)
            w_q = 0.5 * (b - a) * wi

            Dm_q = np.zeros((len(x_q), K))
            for j in range(K):
                c = np.zeros(K)
                c[j] = 1.0
                spl = BSpl(self._knots, c, self.degree)
                Dm_q[:, j] = spl(x_q, nu=order)

            omega += Dm_q.T @ (Dm_q * w_q[:, None])

        return omega

    def _build_penalty(self) -> NDArray:
        return self._build_penalty_for_order(self._m_orders[0])

    def _basis_matrix(self, x: NDArray):
        if self.extrapolation != "extend":
            return super()._basis_matrix(x)
        return self._linear_tail_basis_matrix(x)

    def _build_monotone_constraints_raw(self) -> LinearConstraintSet:
        """Build monotone constraints on raw B-spline coefficients.

        CRS is built on a raw B-spline basis projected through a
        natural-boundary Z matrix. Adjacent-coefficient-difference
        constraints on the raw B-spline coefficients (D @ beta_raw >= 0)
        guarantee monotonicity because B-spline functions with monotone
        coefficients are monotone.

        The composition through Z and identifiability is handled by
        _SplineBase.build() via cs_raw.compose(projection).
        """
        K = self._n_basis
        D = np.diff(np.eye(K), axis=0)
        if self.monotone == "decreasing":
            D = -D
        return LinearConstraintSet(A=D, b=np.zeros(K - 1))

    def _apply_constraints(self, B, omega: NDArray) -> tuple[Any, NDArray, int, NDArray | None]:
        """Natural boundary constraints: f''(lo) = f''(hi) = 0."""
        Z = self._natural_constraint_null_space()
        self._Z = Z
        omega_nat = Z.T @ omega @ Z
        return B, omega_nat, self._n_basis - 2, Z


class CardinalCRSpline(_SplineBase):
    """Cardinal cubic regression spline (mgcv ``bs="cr"`` parameterisation).

    The basis functions are the natural cubic spline cardinal functions:
    basis function *j* is the unique natural cubic spline that equals 1
    at knot *j* and 0 at every other knot.  The penalty is the exact
    integrated squared second derivative, ``S = B_d^T D^{-1} B_d``,
    computed from the tridiagonal second-derivative system.

    Natural boundary conditions (``f''=0`` at endpoints) are built into
    the cardinal construction — no Z-projection is needed.

    This is an **experimental** implementation for A/B comparison against
    the existing ``CubicRegressionSpline`` (which uses a B-spline basis
    projected via Z).  It is not yet the default for ``kind="cr"``.

    Parameters
    ----------
    n_knots : int
        Number of interior knots.  Total knots K = n_knots + 2
        (boundaries are added automatically).
    knot_strategy : str
        ``"uniform"`` (default), ``"quantile"``, ``"quantile_rows"``,
        or ``"quantile_tempered"``.
    penalty : str
        ``"ssp"`` enables SSP reparametrisation, ``"none"`` disables it.
    select : bool
        If True, decompose into linear + wiggly subgroups (double penalty).
    knots : array-like or None
        Explicit interior knot positions.
    monotone : str or None
        Monotonicity constraint direction.
    monotone_mode : str
        ``"postfit"`` (default) or ``"fit"`` (not yet implemented).
    """

    _penalty_semantics = "fixed"
    _max_penalty_order = 2
    _multi_m_supported = False
    _tensor_supported = False

    def _select_compatible(self, m_orders: tuple[int, ...]) -> bool:
        """CardinalCR: only m=(2,) supports select."""
        return m_orders == (2,)

    def __init__(
        self,
        n_knots: int = 10,
        knot_strategy: str = "uniform",
        penalty: str = "ssp",
        select: bool = False,
        knots: ArrayLike | None = None,
        discrete: bool | None = None,
        n_bins: int | None = None,
        extrapolation: str = "clip",
        boundary: tuple[float, float] | None = None,
        knot_alpha: float = 0.2,
        monotone: str | None = None,
        monotone_mode: str = "postfit",
        m: int | tuple[int, ...] = 2,
        lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
    ):
        super().__init__(
            n_knots,
            degree=3,
            knot_strategy=knot_strategy,
            penalty=penalty,
            knots=knots,
            discrete=discrete,
            n_bins=n_bins,
            extrapolation=extrapolation,
            boundary=boundary,
            knot_alpha=knot_alpha,
            select=select,
            monotone=monotone,
            monotone_mode=monotone_mode,
            m=m,
            lambda_policy=lambda_policy,
        )
        self._cr_knots: NDArray | None = None
        self._cr_M: NDArray | None = None
        self._cr_S: NDArray | None = None

    def _place_knots(self, x: NDArray) -> None:
        """Place K = n_knots + 2 knots and build the cardinal CR matrices."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if self._explicit_boundary is not None:
            self._lo, self._hi = self._explicit_boundary
        else:
            self._lo, self._hi = float(x.min()), float(x.max())
        self._basis_lo = None
        self._basis_hi = None
        self._basis_d1_lo = None
        self._basis_d1_hi = None

        if self._explicit_knots is not None:
            interior = self._explicit_knots
            self._knot_strategy_actual = "explicit"
        elif self.knot_strategy in ("quantile", "quantile_rows", "quantile_tempered"):
            # Restrict to [lo, hi] when boundary is frozen (same logic as _SplineBase).
            x_q = x[(x >= self._lo) & (x <= self._hi)] if self._explicit_boundary is not None else x
            if len(x_q) == 0:
                x_q = np.array([self._lo, self._hi])
            if self.knot_strategy == "quantile_tempered":
                interior = _weighted_quantile_knots(x_q, self.n_knots, self.knot_alpha)
            else:
                probs = np.linspace(0, 100, self.n_knots + 2)[1:-1]
                if self.knot_strategy == "quantile":
                    source = np.unique(x_q)
                else:
                    source = x_q
                interior = np.unique(np.percentile(source, probs))
            if len(interior) < self.n_knots:
                interior = np.linspace(self._lo, self._hi, self.n_knots + 2)[1:-1]
                self._knot_strategy_actual = "uniform"
            else:
                self._knot_strategy_actual = self.knot_strategy
        else:
            interior = np.linspace(self._lo, self._hi, self.n_knots + 2)[1:-1]
            self._knot_strategy_actual = "uniform"

        # Cardinal CR knots include the boundaries
        self._cr_knots = np.concatenate([[self._lo], interior, [self._hi]])
        K = len(self._cr_knots)
        self._n_basis = K
        # Store for base-class compatibility (reconstruct, etc.)
        self._knots = self._cr_knots

        self._build_cr_matrices()

    def _build_cr_matrices(self) -> None:
        """Build the tridiagonal system matrices for the cardinal CR spline.

        Given K knots with spacings h_j = x_{j+1} - x_j:

        * B_d (K-2, K): divided-difference band matrix mapping knot values
          to second-derivative RHS.
        * D (K-2, K-2): tridiagonal matrix for the interior second
          derivatives gamma_2 ... gamma_{K-1}.
        * M (K, K): full mapping delta -> gamma, with M[0,:] = M[K-1,:] = 0
          encoding the natural boundary conditions gamma_1 = gamma_K = 0.
        * S (K, K): penalty S = B_d^T D^{-1} B_d.
        """
        knots = self._cr_knots
        K = len(knots)
        h = np.diff(knots)

        # B_d: (K-2, K) — divided-difference band matrix
        B_d = np.zeros((K - 2, K))
        for i in range(K - 2):
            B_d[i, i] = 1.0 / h[i]
            B_d[i, i + 1] = -1.0 / h[i] - 1.0 / h[i + 1]
            B_d[i, i + 2] = 1.0 / h[i + 1]

        # D: (K-2, K-2) — tridiagonal for interior second derivatives
        D = np.zeros((K - 2, K - 2))
        for i in range(K - 2):
            D[i, i] = (h[i] + h[i + 1]) / 3.0
            if i < K - 3:
                D[i, i + 1] = h[i + 1] / 6.0
                D[i + 1, i] = h[i + 1] / 6.0

        # Penalty: S = B_d^T D^{-1} B_d
        D_inv_Bd = np.linalg.solve(D, B_d)
        self._cr_S = B_d.T @ D_inv_Bd

        # M: K×K mapping knot values (delta) → second derivatives (gamma)
        # gamma_1 = gamma_K = 0 (natural BCs); interior via D^{-1} B_d
        self._cr_M = np.zeros((K, K))
        self._cr_M[1 : K - 1, :] = D_inv_Bd

    def _cardinal_boundary_slopes(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Basis value and slope at the boundary knots for linear extrapolation.

        Returns ``(basis_lo, slope_lo, basis_hi, slope_hi)`` where each is
        a length-K vector in the raw cardinal basis.
        """
        knots = self._cr_knots
        K = len(knots)
        h = np.diff(knots)
        M = self._cr_M

        # At lo (knot 0): basis = e_0
        basis_lo = np.zeros(K)
        basis_lo[0] = 1.0
        # Slope at lo via derivative of cardinal formula at t=0 in interval [x_0, x_1]:
        # df/dx = (-1/h)*delta_0 + (1/h)*delta_1 + h*((-2/6)*gamma_0 + (-1/6)*gamma_1)
        # gamma_0 = 0 (natural BC), so:
        slope_lo = np.zeros(K)
        slope_lo[0] = -1.0 / h[0]
        slope_lo[1] = 1.0 / h[0]
        slope_lo -= (h[0] / 6.0) * M[1, :]

        # At hi (knot K-1): basis = e_{K-1}
        basis_hi = np.zeros(K)
        basis_hi[K - 1] = 1.0
        # Slope at hi via derivative at t=1 in interval [x_{K-2}, x_{K-1}]:
        # df/dx = (-1/h)*delta_{K-2} + (1/h)*delta_{K-1} + h*((1/6)*gamma_{K-2} + (1/3)*gamma_{K-1})
        # gamma_{K-1} = 0 (natural BC), so:
        slope_hi = np.zeros(K)
        slope_hi[K - 2] = -1.0 / h[K - 2]
        slope_hi[K - 1] = 1.0 / h[K - 2]
        slope_hi += (h[K - 2] / 6.0) * M[K - 2, :]

        return basis_lo, slope_lo, basis_hi, slope_hi

    def _build_penalty(self) -> NDArray:
        return self._cr_S

    def _basis_matrix(self, x: NDArray):
        """Evaluate the cardinal CR basis at data points."""
        x_eval, extrapolate = self._prepare_eval_points(x)
        if not extrapolate:
            return self._eval_cardinal_basis(x_eval)
        return self._linear_tail_cardinal_basis(x_eval)

    def _raw_basis_matrix(self, x: NDArray) -> NDArray:
        x = np.asarray(x, dtype=np.float64).ravel()
        x_clip = np.clip(x, self._lo, self._hi)
        return self._eval_cardinal_basis(x_clip).toarray()

    def _linear_tail_cardinal_basis(self, x: NDArray) -> sp.csr_matrix:
        """Evaluate cardinal basis with natural-spline linear tails outside range."""
        x = np.asarray(x, dtype=np.float64).ravel()
        lo_mask = x < self._lo
        hi_mask = x > self._hi
        mid_mask = ~(lo_mask | hi_mask)

        K = len(self._cr_knots)
        X = np.zeros((len(x), K))

        if np.any(mid_mask):
            X[mid_mask] = self._eval_cardinal_basis(x[mid_mask]).toarray()

        if np.any(lo_mask) or np.any(hi_mask):
            basis_lo, slope_lo, basis_hi, slope_hi = self._cardinal_boundary_slopes()
            if np.any(lo_mask):
                X[lo_mask] = basis_lo + (x[lo_mask, None] - self._lo) * slope_lo
            if np.any(hi_mask):
                X[hi_mask] = basis_hi + (x[hi_mask, None] - self._hi) * slope_hi

        return sp.csr_matrix(X)

    def _eval_cardinal_basis(self, x: NDArray) -> sp.csr_matrix:
        """Vectorised cardinal cubic regression spline evaluation.

        For x in [x_j, x_{j+1}] with t = (x - x_j) / h_j:

            f(x) = (1-t)*delta_j + t*delta_{j+1}
                   + h_j^2 * [((1-t)^3 - (1-t))/6 * gamma_j
                              + (t^3 - t)/6 * gamma_{j+1}]

        where gamma = M @ delta.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        knots = self._cr_knots
        K = len(knots)
        h = np.diff(knots)
        M = self._cr_M
        n = len(x)

        # Interval index for each point (clipped to valid range)
        j = np.searchsorted(knots, x, side="right") - 1
        j = np.clip(j, 0, K - 2)

        # Local parameter t ∈ [0, 1]
        hj = h[j]
        t = (x - knots[j]) / hj

        # Linear interpolation part
        X = np.zeros((n, K))
        rows = np.arange(n)
        X[rows, j] = 1.0 - t
        X[rows, j + 1] = t

        # Cubic correction via second-derivative mapping M
        c1 = hj**2 * ((1.0 - t) ** 3 - (1.0 - t)) / 6.0  # coeff of gamma_j
        c2 = hj**2 * (t**3 - t) / 6.0  # coeff of gamma_{j+1}
        X += c1[:, None] * M[j, :] + c2[:, None] * M[j + 1, :]

        return sp.csr_matrix(X)

    @property
    def fitted_knots(self) -> NDArray | None:
        if self._cr_knots is None:
            return None
        return self._cr_knots[1:-1].copy()

    def _apply_constraints(self, B, omega: NDArray) -> tuple[Any, NDArray, int, NDArray | None]:
        """No Z-projection needed — natural BCs are built into M."""
        return B, omega, self._n_basis, None

    def reconstruct(self, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
        beta_orig = self._R_inv @ beta if self._R_inv is not None else beta
        x_grid = np.linspace(self._lo, self._hi, n_points)
        B_grid = self._basis_matrix(x_grid).toarray()
        log_rels = B_grid @ beta_orig
        return {
            "x": x_grid,
            "log_relativity": log_rels,
            "relativity": np.exp(log_rels),
            "knots_interior": self._cr_knots[1:-1],
            "coefficients_original": beta_orig,
        }

    # tensor_marginal_ingredients: rejected by base class via _tensor_supported = False


# ═══════════════════════════════════════════════════════════════════
# Public Spline factory
# ═══════════════════════════════════════════════════════════════════

_KIND_MAP = {
    "bs": PSpline,
    "ps": PSpline,
    "ns": NaturalSpline,
    "cr": CubicRegressionSpline,
    "cr_cardinal": CardinalCRSpline,
}


def n_knots_from_k(kind: str, k: int, degree: int = 3) -> int:
    """Convert basis dimension ``k`` to interior knot count.

    ``k`` is the number of basis functions *before* identifiability,
    following the common GAM basis-dimension convention.  The built column count is
    ``k - 1`` for all spline kinds because the identifiability
    constraint (unweighted sum-to-zero) removes one direction.

    Parameters
    ----------
    kind : str
        Spline kind: ``"ps"``, ``"bs"``, ``"ns"``, or ``"cr"``.
    k : int
        Basis dimension (number of basis functions before
        identifiability).  The built column count is always ``k - 1``.
    degree : int
        B-spline polynomial degree (default 3, cubic).

    Returns
    -------
    int
        Number of interior knots for the chosen kind.

    Mapping
    -------
    - ``"ps"``/``"bs"``: ``n_knots = k - degree - 1``  (identifiability
      removes 1, ``build().n_cols == k - 1``)
    - ``"ns"``: ``n_knots = k - degree + 1``  (2 natural constraints +
      identifiability, ``build().n_cols == k - 1``)
    - ``"cr"``: ``n_knots = k - degree + 1``  (2 natural constraints +
      identifiability, ``build().n_cols == k - 1``)
    """
    if kind not in _KIND_MAP:
        raise ValueError(f"Unknown spline kind {kind!r}, expected one of {sorted(_KIND_MAP)}")

    if kind in ("bs", "ps"):
        n_knots = k - degree - 1
        min_k = degree + 2  # need at least 1 interior knot
    elif kind == "cr_cardinal":
        # Cardinal CR: K total knots = n_knots + 2, K = k (basis dim before ident)
        n_knots = k - 2
        min_k = 3  # need at least 1 interior knot (K >= 3)
    else:
        # ns and cr: n_basis = n_knots + degree + 1, natural constraints remove 2
        # → pre-identifiability cols = n_knots + degree - 1 = k
        # → n_knots = k - degree + 1
        # For cr, absorbs_intercept removes 1 more → build().n_cols = k - 1
        n_knots = k - degree + 1
        min_k = degree  # need at least 1 interior knot → k >= degree

    if k < min_k:
        raise ValueError(
            f"k={k} is too small for kind={kind!r} with degree={degree}. Minimum k is {min_k}."
        )

    return n_knots


def Spline(
    kind: str = "ps",
    *,
    k: int | None = None,
    n_knots: int | None = None,
    degree: int = 3,
    knot_strategy: str = "uniform",
    penalty: str = "ssp",
    select: bool = False,
    knots: ArrayLike | None = None,
    discrete: bool | None = None,
    n_bins: int | None = None,
    extrapolation: str = "clip",
    boundary: tuple[float, float] | None = None,
    knot_alpha: float = 0.2,
    monotone: str | None = None,
    monotone_mode: str = "postfit",
    m: int | tuple[int, ...] = 2,
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
) -> _SplineBase:
    """Create a spline feature spec.

    This is the recommended public API for creating spline features.
    Dispatches to ``PSpline``, ``NaturalSpline``, or
    ``CubicRegressionSpline`` based on ``kind``.

    Parameters
    ----------
    kind : str
        Spline type:

        - ``"ps"`` — P-spline (B-spline basis + discrete-difference penalty).
          Default. Equivalent to ``PSpline``.
        - ``"bs"`` — **Deprecated** alias for ``"ps"``. Currently creates a
          P-spline (same as ``"ps"``). In a future release, ``"bs"`` will
          create a proper B-spline smooth with integrated-derivative penalty.
        - ``"ns"`` — Natural P-spline (f''=0 at boundaries, linear tails).
          Equivalent to ``NaturalSpline``.
        - ``"cr"`` — Cubic regression spline (integrated f'' penalty +
          natural constraints + identifiability).
          Equivalent to ``CubicRegressionSpline``.
        - ``"cr_cardinal"`` — **Experimental** cardinal cubic regression
          spline.  Uses a cardinal natural cubic spline parameterisation
          where basis functions are interpolants at the knot locations.
          Penalty is ``B_d^T D^{-1} B_d`` from the tridiagonal second-
          derivative system.

    k : int, optional
        Basis dimension (number of basis functions before
        identifiability), following the common GAM ``k`` convention. The built column
        count is always ``k - 1`` because the identifiability
        constraint (unweighted sum-to-zero) removes one
        direction.  Internally converted to ``n_knots`` via
        :func:`n_knots_from_k`.  Cannot be used together with
        ``n_knots``.
    n_knots : int, optional
        Number of interior knots (lower-level parameter). Cannot be
        used together with ``k``. Defaults to 10 if neither ``k`` nor
        ``n_knots`` is given.
    degree : int
        B-spline polynomial degree (default 3). Ignored for ``kind="cr"``
        which is always cubic.
    knot_strategy : str
        ``"uniform"`` (default), ``"quantile"``, ``"quantile_rows"``,
        or ``"quantile_tempered"``.
    penalty : str
        ``"ssp"`` enables SSP reparametrisation (default), ``"none"``
        disables it.
    select : bool
        If True, add a null-space selection penalty (mgcv-style double
        penalty). Support depends on basis and ``m``:
        ``"bs"`` (max(m)<=2), ``"cr"`` (any m<=3), ``"cr_cardinal"``
        (m=2 only). Not supported for ``"ns"``.
    knots : array-like, optional
        Explicit interior knot positions. Overrides ``k`` / ``n_knots``.
    discrete : bool, optional
        Enable covariate discretization.
    n_bins : int, optional
        Number of discretization bins.
    extrapolation : {"clip", "extend", "error"}
        Prediction-time behavior outside the training range.

        - ``"clip"`` (default): freeze at boundary value.
        - ``"extend"``: evaluate basis outside training range. For
          ``"ns"`` and ``"cr"`` this gives linear tails; for ``"bs"``
          this uses the B-spline's native polynomial continuation.
        - ``"error"``: raise on out-of-range values.
    boundary : tuple of float, optional
        Explicit ``(lo, hi)`` boundary. When set, the boundary is
        frozen across refits instead of being inferred from the data
        range. Use together with ``knots`` to fully freeze knot
        placement.
    knot_alpha : float
        Tempering exponent for ``knot_strategy="quantile_tempered"``.
        Default 0.2.  ``alpha=0`` gives equal weight per unique value
        (same as ``"quantile"``); higher values concentrate knots in
        dense regions.  Ignored by other strategies.

    Returns
    -------
    _SplineBase
        A concrete spline feature spec (``PSpline``,
        ``NaturalSpline``, or ``CubicRegressionSpline``).

    Examples
    --------
    >>> Spline(kind="ps", k=20)           # 20-column P-spline
    >>> Spline(kind="cr", k=10)           # 9-column cubic regression spline (k-1)
    >>> Spline(kind="ns", n_knots=8)      # 8 interior knots, natural spline
    >>> Spline(n_knots=10, penalty="ssp")  # backward-compatible, defaults to "ps"
    >>> Spline(kind="cr", k=12, select=True)  # CR with double-penalty selection
    """
    if kind not in _KIND_MAP:
        raise ValueError(f"Unknown spline kind {kind!r}, expected one of {sorted(_KIND_MAP)}")

    if k is not None and n_knots is not None:
        raise ValueError(
            "Cannot specify both k and n_knots. Use k (public basis size) or n_knots (interior knots), not both."
        )

    if monotone is not None and kind == "ns":
        raise NotImplementedError(
            "monotone is not supported for kind='ns'. "
            "Use kind='cr' or kind='ps' with monotone='increasing' or 'decreasing'."
        )

    # Deprecation warning for kind="bs"
    if kind == "bs":
        import warnings

        warnings.warn(
            "Spline(kind='bs') currently creates a P-spline (discrete-difference "
            "penalty). Use kind='ps' for this behavior. In a future release, "
            "kind='bs' will create a proper B-spline smooth with "
            "integrated-derivative penalty.",
            FutureWarning,
            stacklevel=2,
        )

    # Resolve n_knots
    if k is not None:
        if kind in ("cr", "cr_cardinal"):
            resolved_n_knots = n_knots_from_k(kind, k, degree=3)
        else:
            resolved_n_knots = n_knots_from_k(kind, k, degree)
    elif n_knots is not None:
        resolved_n_knots = n_knots
    else:
        resolved_n_knots = 10  # default

    # Dispatch to concrete class
    cls = _KIND_MAP[kind]

    if kind in ("bs", "ps"):
        return cls(
            n_knots=resolved_n_knots,
            degree=degree,
            knot_strategy=knot_strategy,
            penalty=penalty,
            select=select,
            knots=knots,
            discrete=discrete,
            n_bins=n_bins,
            extrapolation=extrapolation,
            boundary=boundary,
            knot_alpha=knot_alpha,
            monotone=monotone,
            monotone_mode=monotone_mode,
            m=m,
            lambda_policy=lambda_policy,
        )
    elif kind in ("cr", "cr_cardinal"):
        if kind == "cr":
            return cls(
                n_knots=resolved_n_knots,
                knot_strategy=knot_strategy,
                penalty=penalty,
                select=select,
                knots=knots,
                discrete=discrete,
                n_bins=n_bins,
                extrapolation=extrapolation,
                boundary=boundary,
                knot_alpha=knot_alpha,
                monotone=monotone,
                monotone_mode=monotone_mode,
                m=m,
                lambda_policy=lambda_policy,
            )
        else:  # cr_cardinal
            return cls(
                n_knots=resolved_n_knots,
                knot_strategy=knot_strategy,
                penalty=penalty,
                select=select,
                knots=knots,
                discrete=discrete,
                n_bins=n_bins,
                extrapolation=extrapolation,
                boundary=boundary,
                knot_alpha=knot_alpha,
                monotone=monotone,
                monotone_mode=monotone_mode,
                m=m,
                lambda_policy=lambda_policy,
            )
    else:  # "ns"
        return cls(
            n_knots=resolved_n_knots,
            degree=degree,
            knot_strategy=knot_strategy,
            penalty=penalty,
            select=select,
            knots=knots,
            discrete=discrete,
            n_bins=n_bins,
            extrapolation=extrapolation,
            boundary=boundary,
            knot_alpha=knot_alpha,
            m=m,
            lambda_policy=lambda_policy,
        )
