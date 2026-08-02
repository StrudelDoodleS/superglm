"""Interaction features for SuperGLM.

Seven interaction types covering all supported feature combinations:

- SplineCategorical: per-level spline curves (varying coefficient)
- PolynomialCategorical: per-level polynomial curves (varying coefficient)
- NumericCategorical: per-level slopes (varying slope)
- CategoricalInteraction: cross-product indicator columns
- NumericInteraction: product of two numerics
- PolynomialInteraction: cross-product of two polynomial bases
- TensorInteraction: ti()-style tensor product spline interaction
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.polynomial.legendre import legvander
from numpy.typing import NDArray

from superglm.features.categorical import _validate_categorical_levels
from superglm.group_matrix import _discretize_column
from superglm.types import DiscreteTensorBuildResult, GroupInfo, TensorMarginalInfo


def interaction_spline_spec(spec, x: NDArray, n_knots_override: int | None = None):
    """The spline spec an INTERACTION marginal is built from.

    Cubic regression splines carry two implementations.  A main effect uses
    ``CubicRegressionSpline`` -- a B-spline basis projected via Z.  Every
    interaction marginal instead uses ``CardinalCRSpline``, which is much
    closer to mgcv's ``bs="cr"`` geometry than the older projected path.

    That routing belongs to *interactions as a class*, not to one interaction
    type.  Applying it in ``TensorInteraction`` alone left ``SplineCategorical``
    building a different basis from the ``ti`` beside it and, more sharply,
    from the screening probe that names it as its refit target: the probe could
    span directions the refit could not build at all.  How badly depends on the
    margin -- containment of probe span in refit span measured 0.9996 on a
    uniform margin but 0.675 on a skewed one at ``n_knots=16`` (issue #191).
    Both go through here so the invariant holds by construction rather than by
    coincidence.

    The knots move too, not just the basis: a default ``knot_strategy="uniform"``
    parent is re-placed on quantiles, because a cardinal basis on knots that
    ignore the margin's density is the weaker half of the geometry this routing
    exists to get right.  So a ``cr`` main effect and the interaction beside it
    sit on DIFFERENT interior knots -- on a gamma margin at ``n_knots=8``, 1.92
    to 15.22 for the main effect against 0.87 to 5.69 here.  That is the same
    placement the screening probe uses, which is what keeps probe and refit
    identical; an explicitly knotted or already-quantile parent is left alone.

    Returns *spec* unchanged for every other spline kind, so ``ps`` -- the
    default -- is untouched, and for the cr configurations the cardinal basis
    cannot express (below).  The returned spec has its knots already placed on
    *x*; callers must STORE it and read the basis back from the stored copy,
    since a predict-time basis rebuilt from the original spec would disagree
    with the design that was fitted.
    """
    from superglm.features.spline import CardinalCRSpline, CubicRegressionSpline

    if not isinstance(spec, CubicRegressionSpline):
        return spec

    # A select=True parent's double-penalty shrinkage lives in its own
    # identifiability projection.  The cardinal spec below is built with
    # select=False, so substituting drops that shrinkage, widening the
    # interaction block and leaving the design rank-deficient.
    if getattr(spec, "select", False):
        return spec
    # ``CardinalCRSpline._build_penalty`` is the exact integrated squared
    # SECOND derivative and nothing else -- the class advertises
    # _penalty_semantics="fixed" and rejects m > 2 outright.  So any other
    # requested order has to keep the parent's quadrature penalty: routing it
    # here would raise for m=3 and silently swap m=1's first-derivative
    # penalty for a second-derivative one.
    if getattr(spec, "_m_orders", (2,)) != (2,):
        return spec

    knot_strategy = spec.knot_strategy
    if knot_strategy == "uniform" and spec._explicit_knots is None:
        knot_strategy = "quantile"
    cardinal = CardinalCRSpline(
        n_knots=n_knots_override if n_knots_override is not None else spec.n_knots,
        knot_strategy=knot_strategy,
        penalty=spec.penalty,
        knots=spec._explicit_knots,
        discrete=spec.discrete,
        n_bins=spec.n_bins,
        extrapolation=spec.extrapolation,
        boundary=spec._explicit_boundary,
        knot_alpha=spec.knot_alpha,
        select=False,
        constraint=None,
        m=spec._m_orders[0],
        lambda_policy=None,
    )
    cardinal._place_knots(np.asarray(x, dtype=np.float64).ravel())
    return cardinal


def _varying_coefficient_spline_spec(spec, x: NDArray):
    """``interaction_spline_spec`` for a per-level masked block, centered.

    A resolved spec is freshly constructed, so unlike a fitted parent it
    carries no identifiability projection until something builds it.  A
    varying-coefficient block cannot go without one: the cardinal basis
    reproduces the constant function exactly -- its columns sum to 1 at every
    x -- so masking the uncentered block to a level rebuilds that level's own
    categorical main-effect indicator, and the combined design loses one rank
    per non-base level.  ``build_knots_and_penalty()`` runs the same
    constraint + identifiability steps a main effect runs and leaves the
    projection on the spec.

    ``tensor_marginal_ingredients()`` centers its own marginals over the
    compressed support it is handed, so the tensor path resolves the spec
    without paying for this one.
    """
    resolved = interaction_spline_spec(spec, x)
    if resolved is not spec:
        resolved.build_knots_and_penalty(x)
    return resolved


def _plan_spline_cat_support(B: sp.spmatrix, x: NDArray, active: NDArray, *, n_levels: int):
    """Lossless row-support compression for a shared ``spline_cat`` basis.

    A varying-coefficient term stores one CSR basis shared by its levels plus a
    row subset per level, and every level runs its own weighted gram over its
    rows.  When the spline covariate repeats -- a rating factor recorded in
    whole years, the common insurance case -- those rows repeat too, and one
    dense block of distinct rows serves every level.

    This is the same deduplication ``_build_ssp_group`` applies to a main
    effect, and the same gate decides it, with three differences.

    The levels partition the rows between them but each reads the whole shared
    support, so the cost model is told how many grams that is.

    The grouping is offered rather than detected: the basis is a function of
    *x* alone, so equal *x* should give equal rows, which makes the decline a
    sort of one column instead of a scan of the whole basis.
    ``plan_verified_row_support`` still checks that grouping against the basis
    before using it, so the saving is on the detection scan and not exactness.

    And only ``active`` rows count.  The base level of the categorical parent
    is absorbed into the main effect, so its rows appear in NO emitted block --
    and with the default ``base="most_exposed"`` that is the level carrying the
    most exposure, routinely the majority of the book.  Charging the CSR side
    for work it will never do overstates the win, and keeping support rows that
    only a base-level observation ever reaches makes every level's dense gram
    scan a support wider than it can use.  Both errors point the same way, so
    the gate is derived from the union of the non-base masks.

    Returns ``None`` when the gate declines, leaving CSR in place.
    """
    # Function-local ON PURPOSE, not merely to dodge a circular import: these
    # two are read fresh on every call, so a test that lowers the module global
    # reaches the pre-gate below.  Hoisting this to module scope would bind them
    # once at import and silently re-freeze that gate while the one inside
    # plan_verified_row_support kept resolving -- the two disagreeing again, in
    # the opposite direction.  The pass-through below is what keeps them equal;
    # this comment is what stops the import being "tidied" upward.
    from superglm._group_matrix._group_matrix_support import (
        DEFAULT_MAX_SUPPORT_BYTES,
        DEFAULT_MIN_SPEEDUP,
        _passes_support_gates,
        plan_verified_row_support,
    )

    active = np.asarray(active, dtype=np.intp).ravel()
    if n_levels <= 0 or active.size == 0:
        return None

    codes = np.unique(np.asarray(x, dtype=np.float64).ravel()[active], return_inverse=True)[1]
    codes = np.ravel(codes).astype(np.intp, copy=False)
    # Active nnz off the indptr rather than off a slice: the gate has to be
    # cheap on the DECLINED path, which is every continuous covariate, and
    # slicing the basis there would cost a copy of most of it.
    nnz_active = int(np.diff(B.indptr)[active].sum())
    # One resolution feeds BOTH gates.  These are read here at call time, so
    # without passing them through, the pre-gate below would see a patched
    # threshold while the gate inside plan_verified_row_support saw the frozen
    # default -- two gates documented as agreeing, disagreeing under a patch,
    # with the block silently falling back to CSR and a test believing it.
    min_speedup = DEFAULT_MIN_SPEEDUP
    max_support_bytes = DEFAULT_MAX_SUPPORT_BYTES
    if not _passes_support_gates(
        int(active.size),
        int(codes.max()) + 1,
        int(B.shape[1]),
        nnz_active,
        min_speedup,
        max_support_bytes,
        n_levels,
    ):
        return None

    planned = plan_verified_row_support(
        sp.csr_matrix(B)[active],
        codes,
        min_speedup=min_speedup,
        max_support_bytes=max_support_bytes,
        gram_repeats=n_levels,
    )
    if planned is None:
        return None
    b_unique, active_codes = planned
    # Full-length, because the builder indexes it by each level's own row ids.
    # Base-level entries are never read; zero is simply a valid index.
    row_index = np.zeros(B.shape[0], dtype=np.intp)
    row_index[active] = active_codes
    return b_unique, row_index


# ── SplineCategorical ──────────────────────────────────────────


class SplineCategorical:
    """Varying-coefficient interaction: spline curve per categorical level.

    For each non-base level of the categorical, produces one group of K
    B-spline columns masked by the level indicator. The base level's effect
    is absorbed into the main spline term.
    """

    def __init__(self, spline_name: str, cat_name: str):
        self.spline_name = spline_name
        self.cat_name = cat_name

        self._knots: NDArray = np.array([])
        self._n_basis: int = 0
        self._degree: int = 3
        self._lo: float = 0.0
        self._hi: float = 1.0
        self._non_base: list[str] = []
        self._base_level: str = ""
        self._R_inv_dict: dict[str, NDArray] | NDArray = {}
        self._projection: NDArray | None = None

    @property
    def parent_names(self) -> tuple[str, str]:
        return (self.spline_name, self.cat_name)

    def build(
        self,
        x_spline: NDArray,
        x_cat: NDArray,
        parent_specs: dict,
        sample_weight: NDArray | None = None,
    ) -> list[GroupInfo]:
        from superglm.features.categorical import Categorical
        from superglm.features.spline import _SplineBase

        spline_spec = parent_specs[self.spline_name]
        cat_spec = parent_specs[self.cat_name]
        if not isinstance(spline_spec, _SplineBase):
            raise TypeError(f"Expected a spline spec for {self.spline_name}")
        if not isinstance(cat_spec, Categorical):
            raise TypeError(f"Expected Categorical spec for {self.cat_name}")

        x_spline = np.asarray(x_spline, dtype=np.float64).ravel()
        x_cat = np.asarray(x_cat).ravel()
        # Interaction marginals use the cardinal cr basis; store the resolved
        # spec so transform()/score() rebuild the SAME basis at predict time.
        spline_spec = _varying_coefficient_spline_spec(spline_spec, x_spline)

        self._spline_spec = spline_spec
        self._knots = spline_spec._knots
        self._n_basis = spline_spec._n_basis
        self._degree = spline_spec.degree
        self._lo = spline_spec._lo
        self._hi = spline_spec._hi
        self._non_base = list(cat_spec._non_base)
        self._base_level = cat_spec._base_level
        self._projection = getattr(spline_spec, "_interaction_projection", None)

        B = sp.csr_matrix(spline_spec._raw_basis_matrix(x_spline))
        # Union of the non-base masks: exactly the rows some emitted block owns.
        active_rows = np.flatnonzero(np.isin(x_cat, self._non_base))
        compressed = _plan_spline_cat_support(
            B, x_spline, active_rows, n_levels=len(self._non_base)
        )

        omega = spline_spec._build_penalty()

        # Project penalty through the full constraint projection (natural
        # constraints + identifiability).  The basis columns stay sparse —
        # the projection is passed via GroupInfo so dm_builder folds it
        # into R_inv (SparseSSPGroupMatrix keeps the factored form).
        if self._projection is not None:
            omega = self._projection.T @ omega @ self._projection
            n_cols = self._projection.shape[1]
        else:
            n_cols = self._n_basis

        groups: list[GroupInfo] = []
        for level in self._non_base:
            mask = x_cat == level
            shared = dict(
                columns=None,
                n_cols=n_cols,
                penalty_matrix=omega,
                reparametrize=True,
                projection=self._projection,
                spline_cat_mask=mask,
                spline_cat_level=str(level),
                spline_cat_feature=self.cat_name,
            )
            if compressed is None:
                groups.append(GroupInfo(spline_cat_basis=B, **shared))
            else:
                b_unique, row_index = compressed
                groups.append(
                    GroupInfo(
                        spline_cat_basis_unique=b_unique,
                        spline_cat_bin_idx=row_index,
                        spline_cat_support_lossless=True,
                        **shared,
                    )
                )
        return groups

    def build_discrete(
        self,
        x_spline: NDArray,
        x_cat: NDArray,
        parent_specs: dict,
        n_bins: int,
        sample_weight: NDArray | None = None,
    ) -> list[GroupInfo]:
        """Build spline-by-category groups from compressed spline support."""
        from superglm.features.categorical import Categorical
        from superglm.features.spline import _SplineBase

        spline_spec = parent_specs[self.spline_name]
        cat_spec = parent_specs[self.cat_name]
        if not isinstance(spline_spec, _SplineBase):
            raise TypeError(f"Expected a spline spec for {self.spline_name}")
        if not isinstance(cat_spec, Categorical):
            raise TypeError(f"Expected Categorical spec for {self.cat_name}")

        x_spline = np.asarray(x_spline, dtype=np.float64).ravel()
        x_cat = np.asarray(x_cat).ravel()
        # Same resolution as build(); see _varying_coefficient_spline_spec.
        spline_spec = _varying_coefficient_spline_spec(spline_spec, x_spline)

        self._spline_spec = spline_spec
        self._knots = spline_spec._knots
        self._n_basis = spline_spec._n_basis
        self._degree = spline_spec.degree
        self._lo = spline_spec._lo
        self._hi = spline_spec._hi
        self._non_base = list(cat_spec._non_base)
        self._base_level = cat_spec._base_level
        self._projection = getattr(spline_spec, "_interaction_projection", None)

        support, bin_idx = _discretize_column(x_spline, int(n_bins))
        B_unique = np.asarray(spline_spec._raw_basis_matrix(support), dtype=np.float64)

        omega = spline_spec._build_penalty()
        if self._projection is not None:
            omega = self._projection.T @ omega @ self._projection
            n_cols = self._projection.shape[1]
        else:
            n_cols = self._n_basis

        groups: list[GroupInfo] = []
        for level in self._non_base:
            mask = x_cat == level
            groups.append(
                GroupInfo(
                    columns=None,
                    n_cols=n_cols,
                    penalty_matrix=omega,
                    reparametrize=True,
                    projection=self._projection,
                    spline_cat_mask=mask,
                    spline_cat_basis_unique=B_unique,
                    spline_cat_bin_idx=bin_idx,
                    spline_cat_level=str(level),
                    spline_cat_feature=self.cat_name,
                )
            )
        return groups

    def set_reparametrisation(self, R_inv_dict: dict[str, NDArray] | NDArray) -> None:
        if isinstance(R_inv_dict, dict):
            self._R_inv_dict = R_inv_dict
            return

        arr = np.asarray(R_inv_dict, dtype=np.float64)
        if len(self._non_base) == 0:
            self._R_inv_dict = {}
            return

        n_cols = arr.shape[1] // len(self._non_base)
        if n_cols * len(self._non_base) != arr.shape[1]:
            raise ValueError("SplineCategorical R_inv array width is not divisible by level count.")

        self._R_inv_dict = {
            level: arr[:, i * n_cols : (i + 1) * n_cols] for i, level in enumerate(self._non_base)
        }

    def transform(self, x_spline: NDArray, x_cat: NDArray) -> NDArray:
        x_spline = np.asarray(x_spline, dtype=np.float64).ravel()
        x_cat = np.asarray(x_cat).ravel()
        _validate_categorical_levels(
            x_cat, set(self._non_base) | {self._base_level}, context=self.cat_name
        )
        B = self._spline_spec._raw_basis_matrix(x_spline)

        blocks = []
        for level in self._non_base:
            indicator = (x_cat == level).astype(np.float64)
            B_level = B * indicator[:, None]
            R_inv = self._R_inv_dict.get(level)
            if R_inv is not None:
                # R_inv already includes projection (P @ R_inv_local)
                B_level = B_level @ R_inv
            elif self._projection is not None:
                B_level = B_level @ self._projection
            blocks.append(B_level)
        return np.hstack(blocks)

    def score(self, x_spline: NDArray, x_cat: NDArray, beta: NDArray) -> NDArray:
        """Score the interaction directly from its public runtime blocks."""
        x_spline = np.asarray(x_spline, dtype=np.float64).ravel()
        x_cat = np.asarray(x_cat).ravel()
        _validate_categorical_levels(
            x_cat, set(self._non_base) | {self._base_level}, context=self.cat_name
        )

        B = sp.csr_matrix(self._spline_spec._raw_basis_matrix(x_spline))
        beta = np.asarray(beta, dtype=np.float64).ravel()
        out = np.zeros(len(x_spline), dtype=np.float64)

        offset = 0
        for level in self._non_base:
            R_inv = self._R_inv_dict.get(level)
            if R_inv is not None:
                n_cols = R_inv.shape[1]
                beta_raw = R_inv @ beta[offset : offset + n_cols]
            elif self._projection is not None:
                n_cols = self._projection.shape[1]
                beta_raw = self._projection @ beta[offset : offset + n_cols]
            else:
                n_cols = self._n_basis
                beta_raw = beta[offset : offset + n_cols]
            offset += n_cols

            mask = x_cat == level
            if np.any(mask):
                out[mask] = np.asarray(B[mask] @ beta_raw, dtype=np.float64).ravel()
        return out

    def reconstruct(self, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
        x_grid = np.linspace(self._lo, self._hi, n_points)

        per_level: dict[str, dict[str, Any]] = {}
        for level in self._non_base:
            level_codes = np.full(n_points, level, dtype=object)
            log_rels = self.score(x_grid, level_codes, beta)
            per_level[level] = {
                "log_relativity": log_rels,
                "relativity": np.exp(log_rels),
            }

        return {
            "x": x_grid,
            "levels": self._non_base,
            "per_level": per_level,
            "base_level": self._base_level,
            "interaction": True,
        }


# ── PolynomialCategorical ─────────────────────────────────────


class PolynomialCategorical:
    """Varying-coefficient interaction: polynomial curve per categorical level.

    For each non-base level, produces one group of ``degree`` Legendre
    columns masked by the level indicator.
    """

    def __init__(self, poly_name: str, cat_name: str):
        self.poly_name = poly_name
        self.cat_name = cat_name

        self._degree: int = 3
        self._lo: float = 0.0
        self._hi: float = 1.0
        self._non_base: list[str] = []
        self._base_level: str = ""

    @property
    def parent_names(self) -> tuple[str, str]:
        return (self.poly_name, self.cat_name)

    def _scale(self, x: NDArray) -> NDArray:
        span = self._hi - self._lo
        if span < 1e-12:
            return np.zeros_like(x)
        return 2.0 * (x - self._lo) / span - 1.0

    def _basis(self, x_scaled: NDArray) -> NDArray:
        return legvander(x_scaled, self._degree)[:, 1:]

    def build(
        self,
        x_poly: NDArray,
        x_cat: NDArray,
        parent_specs: dict,
        sample_weight: NDArray | None = None,
    ) -> list[GroupInfo]:
        from superglm.features.categorical import Categorical
        from superglm.features.polynomial import Polynomial

        poly_spec = parent_specs[self.poly_name]
        cat_spec = parent_specs[self.cat_name]
        if not isinstance(poly_spec, Polynomial):
            raise TypeError(f"Expected Polynomial spec for {self.poly_name}")
        if not isinstance(cat_spec, Categorical):
            raise TypeError(f"Expected Categorical spec for {self.cat_name}")

        self._degree = poly_spec.degree
        self._lo = poly_spec._lo
        self._hi = poly_spec._hi
        self._non_base = list(cat_spec._non_base)
        self._base_level = cat_spec._base_level

        x_poly = np.asarray(x_poly, dtype=np.float64).ravel()
        x_cat = np.asarray(x_cat).ravel()
        P = self._basis(self._scale(x_poly))

        groups: list[GroupInfo] = []
        for level in self._non_base:
            indicator = (x_cat == level).astype(np.float64)
            P_level = P * indicator[:, None]
            groups.append(GroupInfo(columns=P_level, n_cols=self._degree))
        return groups

    def transform(self, x_poly: NDArray, x_cat: NDArray) -> NDArray:
        x_poly = np.asarray(x_poly, dtype=np.float64).ravel()
        x_cat = np.asarray(x_cat).ravel()
        _validate_categorical_levels(
            x_cat, set(self._non_base) | {self._base_level}, context=self.cat_name
        )
        P = self._basis(self._scale(x_poly))

        blocks = []
        for level in self._non_base:
            indicator = (x_cat == level).astype(np.float64)
            blocks.append(P * indicator[:, None])
        return np.hstack(blocks)

    def score(self, x_poly: NDArray, x_cat: NDArray, beta: NDArray) -> NDArray:
        x_poly = np.asarray(x_poly, dtype=np.float64).ravel()
        x_cat = np.asarray(x_cat).ravel()
        _validate_categorical_levels(
            x_cat, set(self._non_base) | {self._base_level}, context=self.cat_name
        )
        P = self._basis(self._scale(x_poly))
        beta = np.asarray(beta, dtype=np.float64).ravel()
        out = np.zeros(len(x_poly), dtype=np.float64)
        offset = 0
        for level in self._non_base:
            b_level = beta[offset : offset + self._degree]
            offset += self._degree
            mask = x_cat == level
            if np.any(mask):
                out[mask] = P[mask] @ b_level
        return out

    def reconstruct(self, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
        x_grid = np.linspace(self._lo, self._hi, n_points)
        P_grid = self._basis(self._scale(x_grid))

        per_level: dict[str, dict[str, Any]] = {}
        offset = 0
        for level in self._non_base:
            b_level = beta[offset : offset + self._degree]
            offset += self._degree
            log_rels = P_grid @ b_level
            per_level[level] = {
                "log_relativity": log_rels,
                "relativity": np.exp(log_rels),
            }

        return {
            "x": x_grid,
            "levels": self._non_base,
            "per_level": per_level,
            "base_level": self._base_level,
            "interaction": True,
        }


# ── NumericCategorical ─────────────────────────────────────────


class NumericCategorical:
    """Varying-slope interaction: per-level numeric slope.

    Single group of L-1 columns, each ``x_num * I(cat == level)``.
    Group lasso selects/deselects the entire interaction as a unit.
    """

    def __init__(self, num_name: str, cat_name: str):
        self.num_name = num_name
        self.cat_name = cat_name

        self._non_base: list[str] = []
        self._base_level: str = ""

    @property
    def parent_names(self) -> tuple[str, str]:
        return (self.num_name, self.cat_name)

    def build(
        self,
        x_num: NDArray,
        x_cat: NDArray,
        parent_specs: dict,
        sample_weight: NDArray | None = None,
    ) -> GroupInfo:
        from superglm.features.categorical import Categorical
        from superglm.features.numeric import Numeric

        num_spec = parent_specs[self.num_name]
        cat_spec = parent_specs[self.cat_name]
        if not isinstance(num_spec, Numeric):
            raise TypeError(f"Expected Numeric spec for {self.num_name}")
        if not isinstance(cat_spec, Categorical):
            raise TypeError(f"Expected Categorical spec for {self.cat_name}")

        self._non_base = list(cat_spec._non_base)
        self._base_level = cat_spec._base_level

        x_num = np.asarray(x_num, dtype=np.float64).ravel()
        x_cat = np.asarray(x_cat).ravel()

        cols = []
        for level in self._non_base:
            indicator = (x_cat == level).astype(np.float64)
            cols.append(x_num * indicator)

        columns = np.column_stack(cols)
        return GroupInfo(columns=columns, n_cols=len(self._non_base))

    def transform(self, x_num: NDArray, x_cat: NDArray) -> NDArray:
        x_num = np.asarray(x_num, dtype=np.float64).ravel()
        x_cat = np.asarray(x_cat).ravel()
        _validate_categorical_levels(
            x_cat, set(self._non_base) | {self._base_level}, context=self.cat_name
        )

        cols = []
        for level in self._non_base:
            indicator = (x_cat == level).astype(np.float64)
            cols.append(x_num * indicator)
        return np.column_stack(cols)

    def score(self, x_num: NDArray, x_cat: NDArray, beta: NDArray) -> NDArray:
        x_num = np.asarray(x_num, dtype=np.float64).ravel()
        x_cat = np.asarray(x_cat).ravel()
        _validate_categorical_levels(
            x_cat, set(self._non_base) | {self._base_level}, context=self.cat_name
        )
        beta = np.asarray(beta, dtype=np.float64).ravel()
        out = np.zeros(len(x_num), dtype=np.float64)
        for i, level in enumerate(self._non_base):
            mask = x_cat == level
            if np.any(mask):
                out[mask] = x_num[mask] * beta[i]
        return out

    def reconstruct(self, beta: NDArray) -> dict[str, Any]:
        log_rels_per_unit: dict[str, float] = {}
        rels_per_unit: dict[str, float] = {}
        for i, level in enumerate(self._non_base):
            b = float(beta[i])
            log_rels_per_unit[level] = b
            rels_per_unit[level] = float(np.exp(b))
        return {
            "levels": self._non_base,
            "base_level": self._base_level,
            "log_relativities_per_unit": log_rels_per_unit,
            "relativities_per_unit": rels_per_unit,
            "interaction": True,
        }


# ── CategoricalInteraction ────────────────────────────────────


class CategoricalInteraction:
    """Cross-product interaction between two categorical features.

    Produces a single group of (L1-1) * (L2-1) indicator columns for
    all non-base level pairs.
    """

    def __init__(self, cat1_name: str, cat2_name: str):
        self.cat1_name = cat1_name
        self.cat2_name = cat2_name

        self._non_base1: list[str] = []
        self._non_base2: list[str] = []
        self._base1: str = ""
        self._base2: str = ""
        self._pairs: list[tuple[str, str]] = []

    @property
    def parent_names(self) -> tuple[str, str]:
        return (self.cat1_name, self.cat2_name)

    def build(
        self,
        x_cat1: NDArray,
        x_cat2: NDArray,
        parent_specs: dict,
        sample_weight: NDArray | None = None,
    ) -> GroupInfo:
        from superglm.features.categorical import Categorical

        cat1_spec = parent_specs[self.cat1_name]
        cat2_spec = parent_specs[self.cat2_name]
        if not isinstance(cat1_spec, Categorical):
            raise TypeError(f"Expected Categorical spec for {self.cat1_name}")
        if not isinstance(cat2_spec, Categorical):
            raise TypeError(f"Expected Categorical spec for {self.cat2_name}")

        self._non_base1 = list(cat1_spec._non_base)
        self._non_base2 = list(cat2_spec._non_base)
        self._base1 = cat1_spec._base_level
        self._base2 = cat2_spec._base_level

        x_cat1 = np.asarray(x_cat1).ravel()
        x_cat2 = np.asarray(x_cat2).ravel()
        n = len(x_cat1)

        self._pairs = []
        rows_list = []
        cols_list = []
        col_idx = 0
        for lev1 in self._non_base1:
            for lev2 in self._non_base2:
                self._pairs.append((lev1, lev2))
                mask = np.where((x_cat1 == lev1) & (x_cat2 == lev2))[0]
                rows_list.append(mask)
                cols_list.append(np.full(len(mask), col_idx))
                col_idx += 1

        n_pairs = len(self._pairs)
        if n_pairs == 0:
            raise ValueError(
                f"CategoricalInteraction {self.cat1_name}:{self.cat2_name} "
                "produced 0 pairs — at least one parent has only 1 level."
            )

        rows_arr = np.concatenate(rows_list) if rows_list else np.array([], dtype=int)
        cols_arr = np.concatenate(cols_list) if cols_list else np.array([], dtype=int)
        data = np.ones(len(rows_arr), dtype=np.float64)
        columns = sp.csr_matrix((data, (rows_arr, cols_arr)), shape=(n, n_pairs))

        return GroupInfo(columns=columns, n_cols=n_pairs)

    def transform(self, x_cat1: NDArray, x_cat2: NDArray) -> NDArray:
        x_cat1 = np.asarray(x_cat1).ravel()
        x_cat2 = np.asarray(x_cat2).ravel()
        _validate_categorical_levels(
            x_cat1, set(self._non_base1) | {self._base1}, context=self.cat1_name
        )
        _validate_categorical_levels(
            x_cat2, set(self._non_base2) | {self._base2}, context=self.cat2_name
        )
        n = len(x_cat1)
        cols = []
        for lev1, lev2 in self._pairs:
            cols.append(((x_cat1 == lev1) & (x_cat2 == lev2)).astype(np.float64))
        return np.column_stack(cols) if cols else np.empty((n, 0))

    def score(self, x_cat1: NDArray, x_cat2: NDArray, beta: NDArray) -> NDArray:
        x_cat1 = np.asarray(x_cat1).ravel()
        x_cat2 = np.asarray(x_cat2).ravel()
        _validate_categorical_levels(
            x_cat1, set(self._non_base1) | {self._base1}, context=self.cat1_name
        )
        _validate_categorical_levels(
            x_cat2, set(self._non_base2) | {self._base2}, context=self.cat2_name
        )
        beta = np.asarray(beta, dtype=np.float64).ravel()
        out = np.zeros(len(x_cat1), dtype=np.float64)
        for i, (lev1, lev2) in enumerate(self._pairs):
            mask = (x_cat1 == lev1) & (x_cat2 == lev2)
            if np.any(mask):
                out[mask] = beta[i]
        return out

    def reconstruct(self, beta: NDArray) -> dict[str, Any]:
        log_rels = {}
        rels = {}
        for i, (lev1, lev2) in enumerate(self._pairs):
            label = f"{lev1}:{lev2}"
            log_rels[label] = float(beta[i])
            rels[label] = float(np.exp(beta[i]))
        return {
            "pairs": self._pairs,
            "log_relativities": log_rels,
            "relativities": rels,
            "levels1": [self._base1] + self._non_base1,
            "levels2": [self._base2] + self._non_base2,
            "base_level1": self._base1,
            "base_level2": self._base2,
            "interaction": True,
        }


# ── NumericInteraction ─────────────────────────────────────────


class NumericInteraction:
    """Product interaction between two numeric features.

    Single group of 1 column: ``x1 * x2``.
    """

    def __init__(self, num1_name: str, num2_name: str):
        self.num1_name = num1_name
        self.num2_name = num2_name

    @property
    def parent_names(self) -> tuple[str, str]:
        return (self.num1_name, self.num2_name)

    def _prep(self, x1: NDArray, x2: NDArray) -> tuple[NDArray, NDArray]:
        x1 = np.asarray(x1, dtype=np.float64).ravel()
        x2 = np.asarray(x2, dtype=np.float64).ravel()
        return x1, x2

    def build(
        self,
        x1: NDArray,
        x2: NDArray,
        parent_specs: dict,
        sample_weight: NDArray | None = None,
    ) -> GroupInfo:
        from superglm.features.numeric import Numeric

        s1 = parent_specs[self.num1_name]
        s2 = parent_specs[self.num2_name]
        if not isinstance(s1, Numeric):
            raise TypeError(f"Expected Numeric spec for {self.num1_name}")
        if not isinstance(s2, Numeric):
            raise TypeError(f"Expected Numeric spec for {self.num2_name}")

        x1s, x2s = self._prep(x1, x2)
        return GroupInfo(columns=(x1s * x2s).reshape(-1, 1), n_cols=1)

    def transform(self, x1: NDArray, x2: NDArray) -> NDArray:
        x1s, x2s = self._prep(x1, x2)
        return (x1s * x2s).reshape(-1, 1)

    def score(self, x1: NDArray, x2: NDArray, beta: NDArray) -> NDArray:
        x1s, x2s = self._prep(x1, x2)
        return x1s * x2s * float(np.asarray(beta, dtype=np.float64).ravel()[0])

    def reconstruct(self, beta: NDArray) -> dict[str, Any]:
        b = float(beta[0])
        return {
            "coef": b,
            "relativity_per_unit_unit": float(np.exp(b)),
            "interaction": True,
        }


# ── PolynomialInteraction ─────────────────────────────────────


class PolynomialInteraction:
    """Cross-product of two polynomial bases.

    Single group of ``d1 * d2`` columns formed by all pairwise products
    of Legendre basis terms (excluding degree 0).
    """

    def __init__(self, poly1_name: str, poly2_name: str):
        self.poly1_name = poly1_name
        self.poly2_name = poly2_name

        self._degree1: int = 3
        self._degree2: int = 3
        self._lo1: float = 0.0
        self._hi1: float = 1.0
        self._lo2: float = 0.0
        self._hi2: float = 1.0

    @property
    def parent_names(self) -> tuple[str, str]:
        return (self.poly1_name, self.poly2_name)

    @staticmethod
    def _scale(x: NDArray, lo: float, hi: float) -> NDArray:
        span = hi - lo
        if span < 1e-12:
            return np.zeros_like(x)
        return 2.0 * (x - lo) / span - 1.0

    @staticmethod
    def _basis(x_scaled: NDArray, degree: int) -> NDArray:
        return legvander(x_scaled, degree)[:, 1:]

    def _cross_design(self, x1: NDArray, x2: NDArray) -> NDArray:
        x1 = np.asarray(x1, dtype=np.float64).ravel()
        x2 = np.asarray(x2, dtype=np.float64).ravel()
        P1 = self._basis(self._scale(x1, self._lo1, self._hi1), self._degree1)
        P2 = self._basis(self._scale(x2, self._lo2, self._hi2), self._degree2)
        n = len(x1)
        n_cols = self._degree1 * self._degree2
        cols = np.empty((n, n_cols))
        idx = 0
        for j in range(self._degree1):
            for k in range(self._degree2):
                cols[:, idx] = P1[:, j] * P2[:, k]
                idx += 1
        return cols

    def build(
        self,
        x1: NDArray,
        x2: NDArray,
        parent_specs: dict,
        sample_weight: NDArray | None = None,
    ) -> GroupInfo:
        from superglm.features.polynomial import Polynomial

        s1 = parent_specs[self.poly1_name]
        s2 = parent_specs[self.poly2_name]
        if not isinstance(s1, Polynomial):
            raise TypeError(f"Expected Polynomial spec for {self.poly1_name}")
        if not isinstance(s2, Polynomial):
            raise TypeError(f"Expected Polynomial spec for {self.poly2_name}")

        self._degree1, self._lo1, self._hi1 = s1.degree, s1._lo, s1._hi
        self._degree2, self._lo2, self._hi2 = s2.degree, s2._lo, s2._hi

        cols = self._cross_design(x1, x2)
        return GroupInfo(columns=cols, n_cols=cols.shape[1])

    def transform(self, x1: NDArray, x2: NDArray) -> NDArray:
        return self._cross_design(x1, x2)

    def score(self, x1: NDArray, x2: NDArray, beta: NDArray) -> NDArray:
        x1 = np.asarray(x1, dtype=np.float64).ravel()
        x2 = np.asarray(x2, dtype=np.float64).ravel()
        P1 = self._basis(self._scale(x1, self._lo1, self._hi1), self._degree1)
        P2 = self._basis(self._scale(x2, self._lo2, self._hi2), self._degree2)
        C = np.asarray(beta, dtype=np.float64).reshape(self._degree1, self._degree2)
        return np.einsum("ij,jk,ik->i", P1, C, P2, optimize=True)

    def reconstruct(self, beta: NDArray, n_points: int = 50) -> dict[str, Any]:
        x1_grid = np.linspace(self._lo1, self._hi1, n_points)
        x2_grid = np.linspace(self._lo2, self._hi2, n_points)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        cols = self._cross_design(X1.ravel(), X2.ravel())
        log_rels = cols @ beta
        return {
            "x1": x1_grid,
            "x2": x2_grid,
            "log_relativity": log_rels.reshape(n_points, n_points),
            "relativity": np.exp(log_rels).reshape(n_points, n_points),
            "interaction": True,
        }


# ── Row-wise Kronecker utility ────────────────────────────────


def _row_kron(B1: sp.spmatrix, B2: sp.spmatrix) -> sp.csr_matrix:
    """Row-wise Kronecker product of two sparse matrices.

    For B1 (n, k1) and B2 (n, k2), returns T (n, k1*k2) where
    T[i, :] = B1[i, :] ⊗ B2[i, :].

    Column ordering: column j1*k2 + j2 corresponds to B1[:,j1] * B2[:,j2].
    """
    # Ensure 2D csc_matrix (not csc_array) for correct column slicing
    B1c = sp.csc_matrix(B1)
    B2c = sp.csc_matrix(B2)
    k2 = B2c.shape[1]
    blocks = []
    for j1 in range(B1c.shape[1]):
        c1 = B1c[:, j1]
        for j2 in range(k2):
            blocks.append(c1.multiply(B2c[:, j2]))
    return sp.csr_matrix(sp.hstack(blocks, format="csr"))


def _row_kron_dense(B1: NDArray, B2: NDArray) -> NDArray:
    """Dense row-wise Kronecker product for discretized tensor support."""
    return np.einsum("ij,ik->ijk", B1, B2).reshape(B1.shape[0], B1.shape[1] * B2.shape[1])


def _normalize_tensor_penalty(S: NDArray) -> NDArray:
    """Scale a marginal tensor penalty to unit leading eigenvalue.

    mgcv rescales marginal penalties before constructing tensor penalties.
    Matching that convention keeps different
    margins on comparable penalty scales.
    """
    eigvals = np.linalg.eigvalsh(S)
    max_eig = float(np.max(eigvals)) if eigvals.size else 0.0
    if max_eig <= 1e-12:
        return S
    return S / max_eig


_TENSOR_SCORE_CHUNK_SIZE = 8192
_TENSOR_SCORE_SAMPLE_SIZE = 4096
_MAX_TENSOR_SCORE_SUPPORT_CELLS = 2_000_000


# ── TensorInteraction ─────────────────────────────────────────


class TensorInteraction:
    """Interaction-only tensor product spline (`ti`) term.

    Builds centered marginal bases from parent spline specs, inheriting
    their knot vectors, penalties, and boundary constraints.  Forms
    the row-wise Kronecker product to yield an interaction-only surface:
    constant and main-effect directions are excluded structurally.

    The tensor penalty is ``kron(S1, I) + kron(I, S2)`` on the centered
    marginals.  This leaves the bilinear ``x1 * x2`` direction in the
    tensor null space while excluding the constant, ``x1`` and ``x2``
    lower-order pieces.  Group lasso can still zero the whole
    interaction block cleanly.

    Parameters
    ----------
    feat1_name, feat2_name : str
        Names of the parent spline features.
    n_knots : tuple of int or None
        ``(n_knots1, n_knots2)`` interior knots for each marginal basis.
        When None (default), the parent's knot count is used directly.
    decompose : bool
        If True, split the centered tensor basis into a 1D bilinear subgroup
        and a wiggly subgroup. This is useful when you want the bilinear null
        space to be selectable/shrinkable separately from the higher-order
        interaction surface.
    """

    def __init__(
        self,
        feat1_name: str,
        feat2_name: str,
        *,
        n_knots: tuple[int, int] | None = None,
        decompose: bool = False,
    ):
        self.feat1_name = feat1_name
        self.feat2_name = feat2_name
        self._n_knots = n_knots
        self._decompose = decompose

        # State set during build()
        self._marginal1: TensorMarginalInfo | None = None
        self._marginal2: TensorMarginalInfo | None = None
        self._p1: int = 0
        self._p2: int = 0
        self._R_inv: NDArray | None = None

    @property
    def parent_names(self) -> tuple[str, str]:
        return (self.feat1_name, self.feat2_name)

    @staticmethod
    def _marginal_from_spec(
        spec,
        x: NDArray,
        n_knots_override: int | None,
        *,
        support: NDArray | None = None,
        counts: NDArray | None = None,
    ) -> TensorMarginalInfo:
        """Get marginal ingredients from a parent spec, optionally overriding n_knots.

        Enforces the mgcv te()/ti() contract on the original spec before
        any cloning, so unsupported configurations (select, multi-m) can't
        bypass the check via n_knots override reconstruction.
        """
        # Reject unsupported parent configs before cloning
        reasons: list[str] = []
        if getattr(spec, "select", False):
            reasons.append("select=True")
        if hasattr(spec, "_m_orders") and len(spec._m_orders) > 1:
            reasons.append(f"m={spec._m_orders}")
        if reasons:
            detail = " and ".join(reasons)
            raise NotImplementedError(
                f"Tensor interactions require single-penalty parent smooths, but "
                f"{type(spec).__name__} was configured with {detail}. "
                "This matches the mgcv te()/ti() marginal-smooth contract."
            )

        def marginal_ingredients(candidate) -> TensorMarginalInfo:
            method = candidate.tensor_marginal_ingredients
            if support is None:
                return method(x)

            def compact_legacy(legacy: TensorMarginalInfo) -> TensorMarginalInfo:
                support_clipped = np.clip(
                    np.asarray(support, dtype=np.float64), legacy.lo, legacy.hi
                )
                compact_basis = np.asarray(legacy.raw_basis_eval(support_clipped), dtype=np.float64)
                compact_basis = compact_basis @ legacy.projection
                return replace(legacy, basis=compact_basis)

            try:
                parameters = inspect.signature(method).parameters
            except (TypeError, ValueError):
                parameters = {}
            accepts_keywords = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
            if accepts_keywords or {"support", "counts"} <= parameters.keys():
                compact = method(x, support=support, counts=counts)
                if np.asarray(compact.basis).shape[0] == len(support):
                    return compact
                return compact_legacy(compact)

            # Compatibility for custom spline subclasses overriding the old
            # one-argument method. Preserve their projection/penalty geometry,
            # but retain only its evaluation on the discrete support.
            return compact_legacy(method(x))

        # Route cubic regression splines through the cardinal CR
        # implementation, which is much closer to mgcv's bs="cr" geometry than
        # the older projected-B-spline CR path.  A cr parameterisation the
        # cardinal basis cannot express comes back unchanged and takes the
        # ordinary marginal path below, penalty order and all.
        cardinal = interaction_spline_spec(spec, x, n_knots_override)
        if cardinal is not spec:
            info = marginal_ingredients(cardinal)
            info.normalize_penalty = True
            return info

        if n_knots_override is not None and n_knots_override != spec.n_knots:
            kwargs: dict = dict(
                n_knots=n_knots_override,
                knot_strategy=spec.knot_strategy,
                penalty=spec.penalty,
                boundary=(spec._lo, spec._hi),
                knot_alpha=spec.knot_alpha,
                # Single-penalty by the guard above, so the parent's order
                # carries over intact; dropping it would rebuild a cr m=1 or
                # m=3 margin on the default second-derivative penalty.
                m=spec._m_orders[0],
            )
            # CubicRegressionSpline/CardinalCRSpline hardcode degree=3
            if "degree" in inspect.signature(type(spec).__init__).parameters:
                kwargs["degree"] = spec.degree
            clone = type(spec)(**kwargs)
            clone._place_knots(x)
            return marginal_ingredients(clone)
        return marginal_ingredients(spec)

    def _centered_marginal_basis(self, x: NDArray, info: TensorMarginalInfo) -> sp.csr_matrix:
        x = np.asarray(x, dtype=np.float64).ravel()
        x_clip = np.clip(x, info.lo, info.hi)
        B = np.asarray(info.raw_basis_eval(x_clip), dtype=np.float64)
        return sp.csr_matrix(B @ info.projection)

    def _prepare_marginal_infos(
        self,
        x1: NDArray,
        x2: NDArray,
        parent_specs: dict,
        *,
        discrete_supports: tuple[
            tuple[NDArray, NDArray],
            tuple[NDArray, NDArray],
        ]
        | None = None,
    ) -> tuple[TensorMarginalInfo, TensorMarginalInfo]:
        from superglm.features.spline import _SplineBase

        spec1 = parent_specs[self.feat1_name]
        spec2 = parent_specs[self.feat2_name]
        if not isinstance(spec1, _SplineBase):
            raise TypeError(f"Expected a spline spec for {self.feat1_name}")
        if not isinstance(spec2, _SplineBase):
            raise TypeError(f"Expected a spline spec for {self.feat2_name}")

        x1 = np.asarray(x1, dtype=np.float64).ravel()
        x2 = np.asarray(x2, dtype=np.float64).ravel()

        nk1 = self._n_knots[0] if self._n_knots is not None else None
        nk2 = self._n_knots[1] if self._n_knots is not None else None

        # tensor_marginal_ingredients() raises TypeError for CardinalCRSpline
        if discrete_supports is None:
            support1 = counts1 = support2 = counts2 = None
        else:
            (support1, counts1), (support2, counts2) = discrete_supports
        self._marginal1 = self._marginal_from_spec(
            spec1,
            x1,
            nk1,
            support=support1,
            counts=counts1,
        )
        self._marginal2 = self._marginal_from_spec(
            spec2,
            x2,
            nk2,
            support=support2,
            counts=counts2,
        )

        self._p1 = self._marginal1.K_eff
        self._p2 = self._marginal2.K_eff
        return self._marginal1, self._marginal2

    def _prepare_centered_marginals(
        self,
        x1: NDArray,
        x2: NDArray,
        parent_specs: dict,
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, NDArray, NDArray]:
        m1, m2 = self._prepare_marginal_infos(x1, x2, parent_specs)

        B1 = sp.csr_matrix(m1.basis)
        B2 = sp.csr_matrix(m2.basis)
        S1 = _normalize_tensor_penalty(m1.penalty) if m1.normalize_penalty else m1.penalty
        S2 = _normalize_tensor_penalty(m2.penalty) if m2.normalize_penalty else m2.penalty
        return B1, B2, S1, S2

    def _build_group_infos(
        self,
        omega_1: NDArray,
        omega_2: NDArray,
        projection: NDArray | None = None,
    ) -> GroupInfo | list[GroupInfo]:
        omega = omega_1 + omega_2
        n_cols = omega.shape[0]
        if self._decompose:
            eigvals, eigvecs = np.linalg.eigh(omega)
            tol = 1e-8 * max(float(np.max(eigvals)), 1e-12)
            null_mask = eigvals < tol
            n_null = int(np.sum(null_mask))
            if n_null != 1:
                raise ValueError(
                    f"Expected 1 null eigenvalue for centered tensor penalty, got {n_null}."
                )
            U_null = eigvecs[:, null_mask]
            U_range = eigvecs[:, ~null_mask]
            omega_1_range = U_range.T @ omega_1 @ U_range
            omega_2_range = U_range.T @ omega_2 @ U_range
            omega_range = 0.5 * (
                (omega_1_range + omega_2_range) + (omega_1_range + omega_2_range).T
            )
            return [
                GroupInfo(
                    columns=None,
                    n_cols=1,
                    penalty_matrix=None,
                    reparametrize=False,
                    penalized=False,
                    subgroup_name="bilinear",
                    projection=U_null,
                ),
                GroupInfo(
                    columns=None,
                    n_cols=n_cols - 1,
                    penalty_matrix=omega_range,
                    reparametrize=False,
                    subgroup_name="wiggly",
                    projection=U_range if projection is None else projection @ U_range,
                    penalty_components=[
                        (f"margin_{self.feat1_name}", omega_1_range),
                        (f"margin_{self.feat2_name}", omega_2_range),
                    ],
                ),
            ]

        # Non-decompose: emit one penalty component per marginal,
        # matching mgcv te()/ti() single-penalty marginal contract.
        return GroupInfo(
            columns=None,
            n_cols=n_cols,
            penalty_matrix=omega,
            reparametrize=False,
            projection=projection,
            penalty_components=[
                (f"margin_{self.feat1_name}", omega_1),
                (f"margin_{self.feat2_name}", omega_2),
            ],
        )

    def build(
        self,
        x1: NDArray,
        x2: NDArray,
        parent_specs: dict,
        sample_weight: NDArray | None = None,
    ) -> GroupInfo | list[GroupInfo]:
        B1, B2, S1, S2 = self._prepare_centered_marginals(x1, x2, parent_specs)

        # Row-wise Kronecker product
        T = _row_kron(B1, B2)

        # Tensor product penalty on the centered marginal spaces.
        omega_1 = np.kron(S1, np.eye(self._p2))
        omega_2 = np.kron(np.eye(self._p1), S2)
        infos = self._build_group_infos(omega_1, omega_2)
        if isinstance(infos, list):
            for info in infos:
                info.columns = T
            return infos
        infos.columns = T
        return infos

    def build_discrete(
        self,
        x1: NDArray,
        x2: NDArray,
        parent_specs: dict,
        n_bins: tuple[int, int],
        sample_weight: NDArray | None = None,
    ) -> DiscreteTensorBuildResult:
        """Build a discretized tensor basis on observed joint support pairs."""
        support1, idx1 = _discretize_column(x1, int(n_bins[0]))
        support2, idx2 = _discretize_column(x2, int(n_bins[1]))
        counts1 = np.bincount(idx1, minlength=len(support1))
        counts2 = np.bincount(idx2, minlength=len(support2))
        m1, m2 = self._prepare_marginal_infos(
            x1,
            x2,
            parent_specs,
            discrete_supports=((support1, counts1), (support2, counts2)),
        )
        S1 = _normalize_tensor_penalty(m1.penalty) if m1.normalize_penalty else m1.penalty
        S2 = _normalize_tensor_penalty(m2.penalty) if m2.normalize_penalty else m2.penalty
        B1_unique = np.asarray(m1.basis, dtype=np.float64)
        B2_unique = np.asarray(m2.basis, dtype=np.float64)

        # Encode joint support pairs into one integer to avoid the much slower
        # np.unique(..., axis=0) path on large observation arrays.
        n_support2 = len(support2)
        pair_codes = idx1.astype(np.int64) * n_support2 + idx2.astype(np.int64)
        observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
        observed_i1 = (observed_codes // n_support2).astype(np.intp)
        observed_i2 = (observed_codes % n_support2).astype(np.intp)
        B_joint = _row_kron_dense(
            B1_unique[observed_i1],
            B2_unique[observed_i2],
        )
        omega_1 = np.kron(S1, np.eye(self._p2))
        omega_2 = np.kron(np.eye(self._p1), S2)
        infos = self._build_group_infos(omega_1, omega_2)
        return DiscreteTensorBuildResult(
            infos=infos,
            B_joint=B_joint,
            pair_idx=pair_idx.astype(np.intp),
            B1_unique=B1_unique,
            B2_unique=B2_unique,
            idx1=idx1.astype(np.intp),
            idx2=idx2.astype(np.intp),
        )

    def set_reparametrisation(self, R_inv: NDArray) -> None:
        self._R_inv = R_inv

    def transform(self, x1: NDArray, x2: NDArray) -> NDArray:
        x1 = np.asarray(x1, dtype=np.float64).ravel()
        x2 = np.asarray(x2, dtype=np.float64).ravel()
        B1 = self._centered_marginal_basis(x1, self._marginal1)
        B2 = self._centered_marginal_basis(x2, self._marginal2)
        T = _row_kron(B1, B2)

        if self._R_inv is not None:
            return T @ self._R_inv
        return T.toarray()

    def score(self, x1: NDArray, x2: NDArray, beta: NDArray) -> NDArray:
        """Score the tensor interaction without materialising the row-Kronecker block."""
        x1 = np.asarray(x1, dtype=np.float64).ravel()
        x2 = np.asarray(x2, dtype=np.float64).ravel()
        if x1.shape != x2.shape:
            raise ValueError("tensor interaction margins must have the same number of rows")

        beta_eff = np.asarray(beta, dtype=np.float64).ravel()
        if self._R_inv is not None:
            beta_eff = self._R_inv @ beta_eff
        C = beta_eff.reshape(self._p1, self._p2)

        if x1.size == 0:
            return np.empty(0, dtype=np.float64)

        sample_step = max(1, len(x1) // _TENSOR_SCORE_SAMPLE_SIZE)
        sample1 = x1[::sample_step][:_TENSOR_SCORE_SAMPLE_SIZE]
        sample2 = x2[::sample_step][:_TENSOR_SCORE_SAMPLE_SIZE]
        repeated_support = np.unique(sample1).size <= max(1, sample1.size // 2) and np.unique(
            sample2
        ).size <= max(1, sample2.size // 2)
        if repeated_support:
            support1, inverse1 = np.unique(x1, return_inverse=True)
            support2, inverse2 = np.unique(x2, return_inverse=True)
            support_cells = support1.size * self._p1 + support2.size * self._p2
            if support_cells <= _MAX_TENSOR_SCORE_SUPPORT_CELLS:
                B1_support = self._centered_marginal_basis(support1, self._marginal1).toarray()
                B2_support = self._centered_marginal_basis(support2, self._marginal2).toarray()
                pair_codes = inverse1.astype(np.int64) * len(support2) + inverse2
                observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
                cells_per_pair = self._p1 + self._p2 + min(self._p1, self._p2)
                batch_size = max(1, _MAX_TENSOR_SCORE_SUPPORT_CELLS // cells_per_pair)
                support_values = np.empty(len(observed_codes), dtype=np.float64)
                for start in range(0, len(observed_codes), batch_size):
                    stop = min(start + batch_size, len(observed_codes))
                    batch_codes = observed_codes[start:stop]
                    left = B1_support[batch_codes // len(support2)]
                    right = B2_support[batch_codes % len(support2)]
                    support_values[start:stop] = np.einsum(
                        "ij,jk,ik->i",
                        left,
                        C,
                        right,
                        optimize=True,
                    )
                return np.asarray(support_values[pair_idx], dtype=np.float64)

        result = np.empty(len(x1), dtype=np.float64)
        for start in range(0, len(x1), _TENSOR_SCORE_CHUNK_SIZE):
            stop = min(start + _TENSOR_SCORE_CHUNK_SIZE, len(x1))
            B1 = self._centered_marginal_basis(x1[start:stop], self._marginal1)
            B2 = self._centered_marginal_basis(x2[start:stop], self._marginal2)

            if self._p1 <= self._p2:
                tmp = np.asarray(B1 @ C, dtype=np.float64)
                result[start:stop] = np.asarray(
                    B2.multiply(tmp).sum(axis=1), dtype=np.float64
                ).ravel()
            else:
                tmp = np.asarray(B2 @ C.T, dtype=np.float64)
                result[start:stop] = np.asarray(
                    B1.multiply(tmp).sum(axis=1), dtype=np.float64
                ).ravel()

        return result

    def reconstruct(self, beta: NDArray, n_points: int = 50) -> dict[str, Any]:
        # Map from SSP space to original space
        if self._R_inv is not None:
            beta_orig = self._R_inv @ beta
        else:
            beta_orig = beta

        m1, m2 = self._marginal1, self._marginal2

        # Reshape to the centered marginal coefficient layout.
        C = beta_orig.reshape(self._p1, self._p2)

        # Evaluate on grid
        x1_grid = np.linspace(m1.lo, m1.hi, n_points)
        x2_grid = np.linspace(m2.lo, m2.hi, n_points)

        B1_grid = self._centered_marginal_basis(x1_grid, m1).toarray()
        B2_grid = self._centered_marginal_basis(x2_grid, m2).toarray()

        # surface[j, i] = f(x1_grid[i], x2_grid[j]) — matches meshgrid convention
        surface = B2_grid @ C.T @ B1_grid.T

        return {
            "x1": x1_grid,
            "x2": x2_grid,
            "log_relativity": surface,
            "relativity": np.exp(surface),
            "interaction": True,
        }
