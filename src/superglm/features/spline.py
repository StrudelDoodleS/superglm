"""B-spline basis with optional SSP reparametrisation.

Knots are penalised via P-spline (Eilers & Marx, 1996), so 15-20 interior
knots is a safe default. More knots gives the penalty more flexibility to
capture the shape — it will not cause overfitting because the second-difference
penalty controls smoothness, not the knot count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from superglm.features import (
    _spline_build,
    _spline_cardinal_spec,
    _spline_config,
    _spline_constraints,
    _spline_factory,
    _spline_identifiability,
    _spline_knots,
    _spline_multi_penalty,
    _spline_penalties,
    _spline_runtime,
    _spline_select,
    _spline_subclass_ops,
)
from superglm.types import GroupInfo, LambdaPolicy, LinearConstraintSet, TensorMarginalInfo

if TYPE_CHECKING:
    from superglm.solvers.scop import SCOPSolverReparam


def _weighted_quantile_knots(x: NDArray, n_knots: int, alpha: float) -> NDArray:
    """Compatibility wrapper for the private weighted-quantile knot helper."""
    return _spline_knots.weighted_quantile_knots(x, n_knots, alpha)


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
        _spline_config.validate_m_orders(self)

    def _validate_m_orders_build(self) -> None:
        """Phase 2: dimension-dependent m validation. Called after knot placement."""
        _spline_config.validate_m_orders_build(self)

    def _validate_select(self) -> None:
        """Phase 1: static select validation. Called from __init__."""
        _spline_config.validate_select(self)

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
        constraint=None,
        m: int | tuple[int, ...] = 2,
        lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
    ):
        _spline_config.initialize_spec(
            self,
            n_knots=n_knots,
            degree=degree,
            knot_strategy=knot_strategy,
            penalty=penalty,
            knots=knots,
            discrete=discrete,
            n_bins=n_bins,
            extrapolation=extrapolation,
            boundary=boundary,
            knot_alpha=knot_alpha,
            select=select,
            constraint=constraint,
            m=m,
            lambda_policy=lambda_policy,
        )

    def _prepare_eval_points(self, x: NDArray) -> tuple[NDArray, bool]:
        """Apply the configured extrapolation policy for basis evaluation."""
        return _spline_runtime.prepare_eval_points(self, x)

    def _basis_matrix(self, x: NDArray):
        """Evaluate the raw B-spline basis under the extrapolation policy."""
        return _spline_runtime.basis_matrix(self, x)

    def _raw_basis_matrix(self, x: NDArray) -> NDArray:
        """Evaluate the raw (pre-projection) basis at points clipped to training range.

        Returns a dense ``(n, n_basis)`` array.  All code outside spline.py
        that needs to evaluate a spline's basis should call this method
        rather than constructing a ``BSpline.design_matrix`` directly, so
        that non-B-spline subclasses (e.g. :class:`CardinalCRSpline`) get
        their own evaluation strategy.
        """
        return _spline_runtime.raw_basis_matrix(self, x)

    def _basis_value_and_slope_at(self, x0: float) -> tuple[NDArray, NDArray]:
        """Return the raw basis row and its first derivative at ``x0``."""
        return _spline_runtime.basis_value_and_slope_at(self, x0)

    def _boundary_linear_rows(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Cache basis value/slope rows for linear continuation at the boundaries."""
        return _spline_runtime.boundary_linear_rows(self)

    def _linear_tail_basis_matrix(self, x: NDArray):
        """Evaluate the raw basis with explicit linear continuation outside the fit range."""
        return _spline_runtime.linear_tail_basis_matrix(self, x)

    def _place_knots(self, x: NDArray) -> None:
        """Place interior knots and build the full knot vector."""
        _spline_runtime.place_knots(self, x)

    def _assemble_knot_vector(self, interior: NDArray) -> None:
        """Build the full knot vector from interior knots.

        Default: clamped (repeated-end) construction.  Subclasses may
        override for open knot vectors.
        """
        _spline_runtime.assemble_clamped_knot_vector(self, interior)

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
        """Build per-order penalty components projected through the same constraints."""
        del B, final_projection
        return _spline_multi_penalty.build_multi_m_components(
            x=x,
            m_orders=self._m_orders,
            build_penalty_for_order=self._build_penalty_for_order,
            apply_constraints=self._apply_constraints,
            apply_identifiability=self._apply_identifiability,
        )

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
        """Compute the projection that removes the intercept-confounded direction."""
        return _spline_identifiability.build_identifiability_projection_for_spec(
            self, x, constraint_projection
        )

    def _apply_identifiability(
        self,
        x: NDArray,
        omega: NDArray,
        projection: NDArray | None,
    ) -> tuple[NDArray, int, NDArray | None]:
        """Remove the intercept-confounded smooth direction."""
        return _spline_identifiability.apply_identifiability_for_spec(self, x, omega, projection)

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
        return _spline_constraints.build_natural_constraint_null_space(
            self._knots,
            self.degree,
            lo=self._lo,
            hi=self._hi,
        )

    def _eigendecompose_select(self, omega_c: NDArray, Z: NDArray | None) -> None:
        """Eigendecompose the constrained penalty for select=True splitting."""
        self._U_null, self._U_range, self._omega_range = _spline_select.eigendecompose_select(
            omega_c,
            Z,
            n_basis=self._n_basis,
            spline_kind=type(self).__name__,
        )

    def _resolve_lambda_policies(self, info: GroupInfo) -> dict[str, LambdaPolicy] | None:
        """Resolve lambda_policy parameter into a per-component dict."""
        return _spline_select.resolve_lambda_policies(self._lambda_policy, info)

    def _build_select(self, x: NDArray, B) -> GroupInfo:
        """Build select=True GroupInfo with null + wiggle/per-order penalty components."""
        return _spline_select.build_select(self, x, B)

    def build(
        self, x: NDArray, sample_weight: NDArray | None = None
    ) -> GroupInfo | list[GroupInfo]:
        """Build B-spline basis and penalty matrix."""
        return _spline_build.build_group_info(self, x, sample_weight)

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
        return _spline_build.build_knots_and_penalty(self, x, sample_weight)

    def transform(self, x: NDArray) -> NDArray:
        """Build design matrix using knots learned during build()."""
        return _spline_runtime.transform(self, x)

    def score(self, x: NDArray, beta: NDArray) -> NDArray:
        """Score the fitted spline contribution on new data."""
        return _spline_runtime.score(self, x, beta)

    def set_reparametrisation(self, R_inv: NDArray) -> None:
        self._R_inv = R_inv

    def reconstruct(self, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
        return _spline_runtime.reconstruct(self, beta, n_points=n_points)

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
        return _spline_build.tensor_marginal_info(self, x)


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
        _spline_subclass_ops.assemble_open_knot_vector(self, interior)


class PSpline(_BSplineBase):
    """P-spline: B-spline basis with a discrete-difference penalty.

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
    monotone : {None, "increasing", "decreasing"}
        ``None`` (default) leaves the spline unconstrained. ``"increasing"``
        requests a nondecreasing monotone spline and ``"decreasing"``
        requests a nonincreasing monotone spline.
    monotone_mode : {"postfit", "fit"}
        ``"postfit"`` (default) fits first and then allows isotonic repair.
        ``"fit"`` keeps the monotone constraint in the optimization problem
        and uses the SCOP monotone engine for ``PSpline``. With
        ``fit_reml()``, fixed lambdas work directly and automatic lambda
        estimation uses the dedicated monotone-aware SCOP REML / EFS path.
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
        constraint=None,
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
            constraint=constraint,
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
        return _spline_subclass_ops.build_scop_reparameterization(self, B, omega)

    def _build_penalty_for_order(self, order: int) -> NDArray:
        return _spline_penalties.build_difference_penalty(self._n_basis, order)

    def _build_penalty(self) -> NDArray:
        return self._build_penalty_for_order(self._m_orders[0])


class BSplineSmooth(_BSplineBase):
    """B-spline smooth: B-spline basis with an integrated-derivative penalty.

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
    monotone : {None, "increasing", "decreasing"}
        ``None`` (default) leaves the spline unconstrained. ``"increasing"``
        requests a nondecreasing monotone spline and ``"decreasing"``
        requests a nonincreasing monotone spline.
    monotone_mode : {"postfit", "fit"}
        ``"postfit"`` (default) fits first and then allows isotonic repair.
        ``"fit"`` uses the constrained QP monotone solver path for
        ``BSplineSmooth``. With ``fit_reml()``, fixed lambdas work directly;
        automatic lambda estimation uses the QP passthrough heuristic
        (unconstrained REML followed by constrained refit), not exact joint
        constrained REML.
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
        constraint=None,
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
            constraint=constraint,
            m=m,
            lambda_policy=lambda_policy,
        )

    def _build_penalty_for_order(self, order: int) -> NDArray:
        """Integrated f^(m) squared penalty via Gauss-Legendre quadrature."""
        return _spline_penalties.build_integrated_derivative_penalty(
            self._knots, self.degree, order
        )

    def _build_monotone_constraints_raw(self) -> LinearConstraintSet:
        """Build monotone constraints on raw B-spline coefficients.

        For monotone increasing: D @ beta_raw >= 0 where D is the
        first-difference matrix (beta_{i+1} - beta_i >= 0).
        For monotone decreasing: -D @ beta_raw >= 0.

        Returns constraints on K raw (pre-projection) coefficients.
        """
        return _spline_subclass_ops.build_monotone_constraints_raw(self)

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
        return _spline_penalties.build_difference_penalty(self._n_basis, order)

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

    Parameters
    ----------
    monotone : {None, "increasing", "decreasing"}
        ``None`` (default) leaves the spline unconstrained. ``"increasing"``
        requests a nondecreasing monotone spline and ``"decreasing"``
        requests a nonincreasing monotone spline.
    monotone_mode : {"postfit", "fit"}
        ``"postfit"`` (default) fits first and then allows isotonic repair.
        ``"fit"`` uses the constrained QP monotone solver path for
        ``CubicRegressionSpline``. With ``fit_reml()``, fixed lambdas work
        directly; automatic lambda estimation uses the QP passthrough
        heuristic (unconstrained REML followed by constrained refit), not
        exact joint constrained REML.
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
        constraint=None,
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
            constraint=constraint,
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
        _spline_subclass_ops.assemble_clamped_knot_vector(self, interior)

    def _build_penalty_for_order(self, order: int) -> NDArray:
        """Integrated f^(m) squared penalty via Gauss-Legendre quadrature."""
        return _spline_penalties.build_integrated_derivative_penalty(
            self._knots, self.degree, order
        )

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
        return _spline_subclass_ops.build_monotone_constraints_raw(self)

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
        constraint=None,
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
            constraint=constraint,
            m=m,
            lambda_policy=lambda_policy,
        )
        self._cr_knots: NDArray | None = None
        self._cr_M: NDArray | None = None
        self._cr_S: NDArray | None = None

    def _place_knots(self, x: NDArray) -> None:
        """Place K = n_knots + 2 knots and build the cardinal CR matrices."""
        _spline_cardinal_spec.place_knots(self, x)

    def _build_cr_matrices(self) -> None:
        """Build the tridiagonal system matrices for the cardinal CR spline."""
        _spline_cardinal_spec.build_cr_matrices(self)

    def _cardinal_boundary_slopes(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Basis value and slope at the boundary knots for linear extrapolation.

        Returns ``(basis_lo, slope_lo, basis_hi, slope_hi)`` where each is
        a length-K vector in the raw cardinal basis.
        """
        return _spline_cardinal_spec.cardinal_boundary_slopes(self)

    def _build_penalty(self) -> NDArray:
        return self._cr_S

    def _basis_matrix(self, x: NDArray):
        """Evaluate the cardinal CR basis at data points."""
        return _spline_cardinal_spec.basis_matrix(self, x)

    def _raw_basis_matrix(self, x: NDArray) -> NDArray:
        return _spline_cardinal_spec.raw_basis_matrix(self, x)

    def _linear_tail_cardinal_basis(self, x: NDArray) -> sp.csr_matrix:
        """Evaluate cardinal basis with natural-spline linear tails outside range."""
        return _spline_cardinal_spec.linear_tail_cardinal_basis(self, x)

    def _eval_cardinal_basis(self, x: NDArray) -> sp.csr_matrix:
        """Vectorised cardinal cubic regression spline evaluation.

        For x in [x_j, x_{j+1}] with t = (x - x_j) / h_j:

            f(x) = (1-t)*delta_j + t*delta_{j+1}
                   + h_j^2 * [((1-t)^3 - (1-t))/6 * gamma_j
                              + (t^3 - t)/6 * gamma_{j+1}]

        where gamma = M @ delta.
        """
        return _spline_cardinal_spec.eval_cardinal_basis(self, x)

    @property
    def fitted_knots(self) -> NDArray | None:
        if self._cr_knots is None:
            return None
        return self._cr_knots[1:-1].copy()

    def _apply_constraints(self, B, omega: NDArray) -> tuple[Any, NDArray, int, NDArray | None]:
        """No Z-projection needed — natural BCs are built into M."""
        return B, omega, self._n_basis, None

    def reconstruct(self, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
        return _spline_cardinal_spec.reconstruct(self, beta, n_points=n_points)

    # tensor_marginal_ingredients: rejected by base class via _tensor_supported = False


# ═══════════════════════════════════════════════════════════════════
# Public Spline factory
# ═══════════════════════════════════════════════════════════════════


def n_knots_from_k(kind: str, k: int, degree: int = 3) -> int:
    """Convert basis dimension ``k`` to interior knot count."""
    return cast(int, _spline_factory.n_knots_from_k(kind, k, degree))


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
    constraint=None,
    m: int | tuple[int, ...] = 2,
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
) -> _SplineBase:
    """Create a spline feature spec."""
    return cast(
        _SplineBase,
        _spline_factory.Spline(
            kind=kind,
            k=k,
            n_knots=n_knots,
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
            constraint=constraint,
            m=m,
            lambda_policy=lambda_policy,
        ),
    )


__all__ = [
    "BSplineSmooth",
    "CardinalCRSpline",
    "CubicRegressionSpline",
    "NaturalSpline",
    "PSpline",
    "Spline",
    "n_knots_from_k",
]
