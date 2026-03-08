"""B-spline basis with optional SSP reparametrisation.

Knots are penalised via P-spline (Eilers & Marx, 1996), so 15-20 interior
knots is a safe default. More knots gives the penalty more flexibility to
capture the shape — it will not cause overfitting because the second-difference
penalty controls smoothness, not the knot count.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline as BSpl

from superglm.types import GroupInfo


class _SplineBase:
    """Base class for all spline feature specs.

    Subclasses must implement ``_build_penalty()`` and may override
    ``_apply_constraints()`` to add boundary constraints.

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
        "quantile" places knots at data quantiles.
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
    ):
        if knots is not None:
            knots = np.asarray(knots, dtype=np.float64).ravel()
            if knots.ndim != 1 or len(knots) < 1:
                raise ValueError("knots must be a non-empty 1D array")
            self.n_knots = len(knots)
            self._explicit_knots = knots
        else:
            self.n_knots = n_knots
            self._explicit_knots = None

        self.degree = degree
        self.knot_strategy = knot_strategy
        self.penalty = penalty
        self.discrete = discrete
        self.n_bins = n_bins
        if extrapolation not in {"clip", "extend", "error"}:
            raise ValueError(
                f"extrapolation must be one of ('clip', 'extend', 'error'), got {extrapolation!r}"
            )
        self.extrapolation = extrapolation

        # State set during build()
        self._knots: NDArray = np.array([])
        self._n_basis: int = 0
        self._lo: float = 0.0
        self._hi: float = 1.0
        self._R_inv: NDArray | None = None
        self._basis_lo: NDArray | None = None
        self._basis_hi: NDArray | None = None
        self._basis_d1_lo: NDArray | None = None
        self._basis_d1_hi: NDArray | None = None

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
        self._lo, self._hi = float(x.min()), float(x.max())
        pad = (self._hi - self._lo) * 1e-6
        self._basis_lo = None
        self._basis_hi = None
        self._basis_d1_lo = None
        self._basis_d1_hi = None

        if self._explicit_knots is not None:
            interior = self._explicit_knots
        elif self.knot_strategy == "quantile":
            probs = np.linspace(0, 100, self.n_knots + 2)[1:-1]
            interior = np.percentile(x, probs)
        else:  # "uniform" (default)
            interior = np.linspace(self._lo, self._hi, self.n_knots + 2)[1:-1]

        self._knots = np.concatenate(
            [
                np.repeat(self._lo - pad, self.degree + 1),
                interior,
                np.repeat(self._hi + pad, self.degree + 1),
            ]
        )
        self._n_basis = len(self._knots) - self.degree - 1

    def _build_penalty(self) -> NDArray:
        """Return (K, K) penalty matrix. Subclasses must implement."""
        raise NotImplementedError

    @property
    def absorbs_intercept(self) -> bool:
        """Whether the smooth should absorb the intercept-like direction."""
        return False

    @property
    def supports_linear_split(self) -> bool:
        """Whether this spline can split linear and wiggly subspaces."""
        return False

    def _apply_constraints(self, B, omega: NDArray) -> tuple[Any, NDArray, int, NDArray | None]:
        """Apply boundary constraints. Returns (B, omega, n_cols, projection).

        Default: no constraints (identity).
        """
        return B, omega, self._n_basis, None

    def _apply_identifiability(
        self, x: NDArray, omega: NDArray, projection: NDArray | None
    ) -> tuple[NDArray, int, NDArray | None]:
        """Optionally remove the intercept-confounded smooth direction.

        mgcv-style smooth terms are constrained to sum to zero over the fit
        data so the model intercept carries the constant part. For CR splines
        this removes one null-space direction after the natural constraints.
        """
        if not self.absorbs_intercept:
            return omega, omega.shape[0], projection

        n_cols = omega.shape[0]
        if n_cols <= 1:
            return omega, n_cols, projection

        x = np.asarray(x, dtype=np.float64).ravel()
        support, counts = np.unique(x, return_counts=True)
        basis = self._basis_matrix(support).toarray()
        if projection is not None:
            basis = basis @ projection

        constraint = counts.astype(np.float64) @ basis
        if np.linalg.norm(constraint) < 1e-12:
            return omega, n_cols, projection

        q, _ = np.linalg.qr(constraint.reshape(-1, 1), mode="complete")
        z = q[:, 1:]
        omega_ident = z.T @ omega @ z
        projection_ident = z if projection is None else projection @ z
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

    def build(self, x: NDArray, exposure: NDArray | None = None) -> GroupInfo:
        """Build B-spline basis and penalty matrix."""
        x = np.asarray(x, dtype=np.float64).ravel()
        self._place_knots(x)
        B = self._basis_matrix(x).tocsr()
        omega = self._build_penalty()
        B, omega, n_cols, projection = self._apply_constraints(B, omega)
        omega, n_cols, projection = self._apply_identifiability(x, omega, projection)
        return GroupInfo(
            columns=B,
            n_cols=n_cols,
            penalty_matrix=omega,
            reparametrize=(self.penalty == "ssp"),
            projection=projection,
        )

    def build_knots_and_penalty(self, x: NDArray) -> tuple[NDArray, int, NDArray | None]:
        """Place knots and return penalty info, without building the full basis.

        Used by the discretization path to avoid the O(n) basis construction.
        Applies boundary constraints (NaturalSpline/CRS) so the returned
        penalty and column count match the exact ``build()`` path.

        Returns
        -------
        omega : (n_cols, n_cols) penalty matrix (projected if constrained).
        n_cols : effective number of basis columns.
        projection : (K, n_cols) constraint projection, or None.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        self._place_knots(x)
        omega = self._build_penalty()
        _, omega, n_cols, projection = self._apply_constraints(None, omega)
        omega, n_cols, projection = self._apply_identifiability(x, omega, projection)
        return omega, n_cols, projection

    def transform(self, x: NDArray) -> NDArray:
        """Build design matrix using knots learned during build()."""
        B = self._basis_matrix(x).toarray()
        if self._R_inv is not None:
            B = B @ self._R_inv
        return B

    def set_reparametrisation(self, R_inv: NDArray) -> None:
        self._R_inv = R_inv

    def reconstruct(self, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
        beta_orig = self._R_inv @ beta if self._R_inv is not None else beta
        x_grid = np.linspace(self._lo, self._hi, n_points)
        B_grid = self._basis_matrix(x_grid).toarray()
        log_rels = B_grid @ beta_orig
        return {
            "x": x_grid,
            "log_relativity": log_rels,
            "relativity": np.exp(log_rels),
            "knots_interior": self._knots[self.degree + 1 : -(self.degree + 1)],
            "coefficients_original": beta_orig,
        }


class Spline(_SplineBase):
    """P-spline: B-spline basis + second-difference penalty.

    Parameters
    ----------
    n_knots : int
        Number of interior knots.
    degree : int
        B-spline polynomial degree.
    knot_strategy : str
        "uniform" (default) or "quantile".
    penalty : str
        "ssp" enables SSP reparametrisation, "none" disables it.
    split_linear : bool
        If True, decompose the spline into null-space (linear) and
        range-space (wiggly) subgroups for mgcv-style three-way
        selection: nonlinear -> linear -> dropped.

        **mgcv double penalty**: The null-space (linear) subgroup is
        penalised with a ridge penalty (``penalty_matrix=eye(1)``).
        With ``fit_reml()``, REML estimates separate lambdas for the
        linear and spline subgroups — driving the linear lambda to
        infinity effectively zeros the linear component (three-way
        selection: nonlinear -> linear -> dropped). See Wood (2011).

    knots : array-like or None
        Explicit interior knot positions.
    """

    def __init__(
        self,
        n_knots: int = 10,
        degree: int = 3,
        knot_strategy: str = "uniform",
        penalty: str = "ssp",
        split_linear: bool = False,
        knots: ArrayLike | None = None,
        discrete: bool | None = None,
        n_bins: int | None = None,
        extrapolation: str = "clip",
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
        )
        self.split_linear = split_linear
        self._U_null: NDArray | None = None
        self._U_range: NDArray | None = None
        self._omega_range: NDArray | None = None

    @property
    def supports_linear_split(self) -> bool:
        return True

    def _build_penalty(self) -> NDArray:
        D2 = np.diff(np.eye(self._n_basis), n=2, axis=0)
        return D2.T @ D2

    def build_knots_and_penalty(self, x: NDArray) -> tuple[NDArray, int, NDArray | None]:
        """Place knots and return penalty + eigendecompose for linear-split terms."""
        omega, n_cols, projection = super().build_knots_and_penalty(x)
        if self.split_linear:
            self._eigendecompose_penalty(omega)
        return omega, n_cols, projection

    def _eigendecompose_penalty(self, omega: NDArray) -> None:
        """Eigendecompose omega into null/range spaces for linear splitting."""
        eigvals, eigvecs = np.linalg.eigh(omega)
        null_mask = eigvals < 1e-10
        n_null_eig = int(np.sum(null_mask))
        if n_null_eig != 2:
            raise ValueError(
                f"Expected 2 null eigenvalues for second-difference penalty, "
                f"got {n_null_eig}. Check penalty matrix or B-spline degree."
            )
        U_null_raw = eigvecs[:, null_mask]
        self._U_range = eigvecs[:, ~null_mask]
        self._omega_range = np.diag(eigvals[~null_mask])
        ones = np.ones(self._n_basis)
        ones_in_null = U_null_raw.T @ ones
        ones_in_null = ones_in_null / np.linalg.norm(ones_in_null)
        U_null_centered = U_null_raw - U_null_raw @ np.outer(ones_in_null, ones_in_null)
        u, s, _ = np.linalg.svd(U_null_centered, full_matrices=False)
        self._U_null = u[:, :1]

    def build(self, x: NDArray, exposure: NDArray | None = None) -> GroupInfo | list[GroupInfo]:
        if not self.split_linear:
            return super().build(x, exposure)

        # split_linear=True path: eigendecompose, return [linear, spline] GroupInfos
        x = np.asarray(x, dtype=np.float64).ravel()
        self._place_knots(x)
        B = self._basis_matrix(x).tocsr()
        omega = self._build_penalty()

        eigvals, eigvecs = np.linalg.eigh(omega)
        null_mask = eigvals < 1e-10
        n_null_eig = int(np.sum(null_mask))
        if n_null_eig != 2:
            raise ValueError(
                f"Expected 2 null eigenvalues for second-difference penalty, "
                f"got {n_null_eig}. Check penalty matrix or B-spline degree."
            )
        U_null_raw = eigvecs[:, null_mask]  # (K, 2): constant + linear
        self._U_range = eigvecs[:, ~null_mask]  # (K, K-2)
        omega_range = np.diag(eigvals[~null_mask])

        # Identifiability: remove the constant from the null space.
        # B-splines have partition of unity (rows sum to 1), so the
        # constant function in coefficient space is proportional to
        # the ones vector. Project it out to keep only the linear part.
        ones = np.ones(self._n_basis)
        ones_in_null = U_null_raw.T @ ones  # project ones onto null basis
        ones_in_null = ones_in_null / np.linalg.norm(ones_in_null)
        # Gram-Schmidt: remove the constant direction
        U_null_centered = U_null_raw - U_null_raw @ np.outer(ones_in_null, ones_in_null)
        # SVD to get the 1D orthonormal basis for the linear part
        u, s, _ = np.linalg.svd(U_null_centered, full_matrices=False)
        self._U_null = u[:, :1]  # (K, 1): linear trend only

        n_null = 1
        n_range = self._U_range.shape[1]

        return [
            GroupInfo(
                columns=B,
                n_cols=n_null,
                penalty_matrix=np.eye(n_null),
                reparametrize=False,
                penalized=True,
                subgroup_name="linear",
                projection=self._U_null,
            ),
            GroupInfo(
                columns=B,
                n_cols=n_range,
                penalty_matrix=omega_range,
                reparametrize=True,
                subgroup_name="spline",
                projection=self._U_range,
            ),
        ]


class NaturalSpline(_SplineBase):
    """Natural P-spline: f''=0 at boundaries, linear tails.

    Applies natural boundary constraints: f''(boundary) = 0 at both
    ends. The underlying basis therefore has linear tails beyond the
    boundary knots, preventing the tail explosions common with
    unconstrained B-splines. Prediction behavior outside the training
    range is then controlled by ``extrapolation``: ``"clip"``
    (default) freezes at the boundary, while ``"extend"`` exposes the
    linear tails. Equivalent to R's ``splines::ns()`` or mgcv's ``cr``
    basis.

    Parameters
    ----------
    n_knots : int
        Number of interior knots.
    degree : int
        B-spline polynomial degree.
    knot_strategy : str
        "uniform" (default) or "quantile".
    penalty : str
        "ssp" enables SSP reparametrisation, "none" disables it.
    knots : array-like or None
        Explicit interior knot positions.
    """

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
        )
        self._Z: NDArray | None = None

    def _build_penalty(self) -> NDArray:
        D2 = np.diff(np.eye(self._n_basis), n=2, axis=0)
        return D2.T @ D2

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

    Equivalent to mgcv's ``s(x, bs="cr")``. Always cubic (degree=3).
    Natural boundary constraints (f''=0 at boundaries) are mandatory.
    The basis has linear tails, but default prediction still clips at
    the training boundary unless ``extrapolation="extend"`` is used.

    The penalty matrix is the wiggliness penalty: omega_ij = int B_i''(x) B_j''(x) dx,
    computed via Gauss-Legendre quadrature over each knot interval.

    Parameters
    ----------
    n_knots : int
        Number of interior knots.
    knot_strategy : str
        "uniform" (default) or "quantile".
    penalty : str
        "ssp" enables SSP reparametrisation, "none" disables it.
    knots : array-like or None
        Explicit interior knot positions.
    """

    def __init__(
        self,
        n_knots: int = 10,
        knot_strategy: str = "uniform",
        penalty: str = "ssp",
        knots: ArrayLike | None = None,
        discrete: bool | None = None,
        n_bins: int | None = None,
        extrapolation: str = "clip",
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
        )
        self._Z: NDArray | None = None

    @property
    def absorbs_intercept(self) -> bool:
        """Match mgcv's identified CR smooth by removing the constant direction."""
        return True

    def _build_penalty(self) -> NDArray:
        """Integrated f'' squared penalty: omega_ij = int B_i''(x) B_j''(x) dx.

        Uses Gauss-Legendre quadrature over each knot interval.
        For cubic B-splines, B'' is piecewise linear, so the product
        is quadratic -- 3-point GL is more than sufficient.
        """
        K = self._n_basis
        unique_knots = np.unique(self._knots)
        omega = np.zeros((K, K))

        for a, b in zip(unique_knots[:-1], unique_knots[1:]):
            if b - a < 1e-15:
                continue
            # 3-point Gauss-Legendre (exact for degree <= 5)
            xi, wi = np.polynomial.legendre.leggauss(3)
            x_q = 0.5 * (b - a) * xi + 0.5 * (a + b)
            w_q = 0.5 * (b - a) * wi

            D2_q = np.zeros((len(x_q), K))
            for j in range(K):
                c = np.zeros(K)
                c[j] = 1.0
                spl = BSpl(self._knots, c, self.degree)
                D2_q[:, j] = spl(x_q, nu=2)

            omega += D2_q.T @ (D2_q * w_q[:, None])

        return omega

    def _basis_matrix(self, x: NDArray):
        if self.extrapolation != "extend":
            return super()._basis_matrix(x)
        return self._linear_tail_basis_matrix(x)

    def _apply_constraints(self, B, omega: NDArray) -> tuple[Any, NDArray, int, NDArray | None]:
        """Natural boundary constraints: f''(lo) = f''(hi) = 0."""
        Z = self._natural_constraint_null_space()
        self._Z = Z
        omega_nat = Z.T @ omega @ Z
        return B, omega_nat, self._n_basis - 2, Z
