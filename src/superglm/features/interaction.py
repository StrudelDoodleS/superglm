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

from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.polynomial.legendre import legvander
from numpy.typing import NDArray
from scipy.interpolate import BSpline as BSpl

from superglm.features.categorical import _validate_categorical_levels
from superglm.group_matrix import _discretize_column
from superglm.types import DiscreteTensorBuildResult, GroupInfo, TensorMarginalInfo

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
        self._R_inv_dict: dict[str, NDArray] = {}
        self._projection: NDArray | None = None

    @property
    def parent_names(self) -> tuple[str, str]:
        return (self.spline_name, self.cat_name)

    def build(
        self,
        x_spline: NDArray,
        x_cat: NDArray,
        parent_specs: dict,
        exposure: NDArray | None = None,
    ) -> list[GroupInfo]:
        from superglm.features.categorical import Categorical
        from superglm.features.spline import _SplineBase

        spline_spec = parent_specs[self.spline_name]
        cat_spec = parent_specs[self.cat_name]
        if not isinstance(spline_spec, _SplineBase):
            raise TypeError(f"Expected a spline spec for {self.spline_name}")
        if not isinstance(cat_spec, Categorical):
            raise TypeError(f"Expected Categorical spec for {self.cat_name}")

        self._spline_spec = spline_spec
        self._knots = spline_spec._knots
        self._n_basis = spline_spec._n_basis
        self._degree = spline_spec.degree
        self._lo = spline_spec._lo
        self._hi = spline_spec._hi
        self._non_base = list(cat_spec._non_base)
        self._base_level = cat_spec._base_level
        self._projection = getattr(spline_spec, "_interaction_projection", None)

        x_spline = np.asarray(x_spline, dtype=np.float64).ravel()
        x_cat = np.asarray(x_cat).ravel()
        B = sp.csr_matrix(spline_spec._raw_basis_matrix(x_spline))

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
            indicator = (x_cat == level).astype(np.float64)
            B_level = B.multiply(indicator[:, None]).tocsr()
            groups.append(
                GroupInfo(
                    columns=B_level,
                    n_cols=n_cols,
                    penalty_matrix=omega,
                    reparametrize=True,
                    projection=self._projection,
                )
            )
        return groups

    def set_reparametrisation(self, R_inv_dict: dict[str, NDArray]) -> None:
        self._R_inv_dict = R_inv_dict

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

    def reconstruct(self, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
        x_grid = np.linspace(self._lo, self._hi, n_points)
        B_grid = self._spline_spec._raw_basis_matrix(x_grid)

        per_level: dict[str, dict[str, Any]] = {}
        offset = 0
        for level in self._non_base:
            R_inv = self._R_inv_dict.get(level)
            if R_inv is not None:
                n_cols = R_inv.shape[1]
            elif self._projection is not None:
                n_cols = self._projection.shape[1]
            else:
                n_cols = self._n_basis
            b_level = beta[offset : offset + n_cols]
            offset += n_cols

            # R_inv already includes projection (P @ R_inv_local)
            if R_inv is not None:
                beta_orig = R_inv @ b_level
            elif self._projection is not None:
                beta_orig = self._projection @ b_level
            else:
                beta_orig = b_level
            log_rels = B_grid @ beta_orig
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
        exposure: NDArray | None = None,
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
        exposure: NDArray | None = None,
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
        exposure: NDArray | None = None,
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
        exposure: NDArray | None = None,
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
        exposure: NDArray | None = None,
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
    def _marginal_from_spec(spec, x: NDArray, n_knots_override: int | None) -> TensorMarginalInfo:
        """Get marginal ingredients from a parent spec, optionally overriding n_knots."""
        if n_knots_override is not None and n_knots_override != spec.n_knots:
            kwargs: dict = dict(
                n_knots=n_knots_override,
                knot_strategy=spec.knot_strategy,
                penalty=spec.penalty,
                boundary=(spec._lo, spec._hi),
                knot_alpha=spec.knot_alpha,
            )
            # CubicRegressionSpline/CardinalCRSpline hardcode degree=3
            import inspect

            if "degree" in inspect.signature(type(spec).__init__).parameters:
                kwargs["degree"] = spec.degree
            clone = type(spec)(**kwargs)
            clone._place_knots(x)
            return clone.tensor_marginal_ingredients(x)
        return spec.tensor_marginal_ingredients(x)

    def _centered_marginal_basis(self, x: NDArray, info: TensorMarginalInfo) -> sp.csr_matrix:
        x = np.asarray(x, dtype=np.float64).ravel()
        x_clip = np.clip(x, info.lo, info.hi)
        B = BSpl.design_matrix(x_clip, info.knots, info.degree).tocsr()
        return sp.csr_matrix(B @ info.projection)

    def _prepare_centered_marginals(
        self,
        x1: NDArray,
        x2: NDArray,
        parent_specs: dict,
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, NDArray, NDArray]:
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
        self._marginal1 = self._marginal_from_spec(spec1, x1, nk1)
        self._marginal2 = self._marginal_from_spec(spec2, x2, nk2)

        self._p1 = self._marginal1.K_eff
        self._p2 = self._marginal2.K_eff

        B1 = sp.csr_matrix(self._marginal1.basis)
        B2 = sp.csr_matrix(self._marginal2.basis)
        S1 = self._marginal1.penalty
        S2 = self._marginal2.penalty
        return B1, B2, S1, S2

    def _build_group_infos(self, omega: NDArray) -> GroupInfo | list[GroupInfo]:
        n_cols = self._p1 * self._p2
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
            omega_range = np.diag(eigvals[~null_mask])
            return [
                GroupInfo(
                    columns=None,
                    n_cols=1,
                    penalty_matrix=np.eye(1),
                    reparametrize=False,
                    subgroup_name="bilinear",
                    projection=U_null,
                ),
                GroupInfo(
                    columns=None,
                    n_cols=n_cols - 1,
                    penalty_matrix=omega_range,
                    reparametrize=True,
                    subgroup_name="wiggly",
                    projection=U_range,
                ),
            ]

        return GroupInfo(
            columns=None,
            n_cols=n_cols,
            penalty_matrix=omega,
            reparametrize=True,
        )

    def build(
        self,
        x1: NDArray,
        x2: NDArray,
        parent_specs: dict,
        exposure: NDArray | None = None,
    ) -> GroupInfo | list[GroupInfo]:
        B1, B2, S1, S2 = self._prepare_centered_marginals(x1, x2, parent_specs)

        # Row-wise Kronecker product
        T = _row_kron(B1, B2)

        # Tensor product penalty on the centered marginal spaces.
        omega = np.kron(S1, np.eye(self._p2)) + np.kron(np.eye(self._p1), S2)
        infos = self._build_group_infos(omega)
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
        exposure: NDArray | None = None,
    ) -> DiscreteTensorBuildResult:
        """Build a discretized tensor basis on observed joint support pairs."""
        B1, B2, S1, S2 = self._prepare_centered_marginals(x1, x2, parent_specs)
        omega = np.kron(S1, np.eye(self._p2)) + np.kron(np.eye(self._p1), S2)
        infos = self._build_group_infos(omega)

        m1, m2 = self._marginal1, self._marginal2
        support1, idx1 = _discretize_column(x1, int(n_bins[0]))
        support2, idx2 = _discretize_column(x2, int(n_bins[1]))
        B1_unique = self._centered_marginal_basis(support1, m1).toarray()
        B2_unique = self._centered_marginal_basis(support2, m2).toarray()

        pair_codes = np.column_stack([idx1, idx2])
        observed_pairs, pair_idx = np.unique(pair_codes, axis=0, return_inverse=True)
        B_joint = _row_kron_dense(
            B1_unique[observed_pairs[:, 0]],
            B2_unique[observed_pairs[:, 1]],
        )
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
