"""Fully penalized smooth deviations by factor level."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.factor_smooth_geometry import expand_sum_to_zero_blocks
from superglm.types import GroupInfo, LambdaPolicy


def _natural_parameterization(
    basis: NDArray,
    penalty: NDArray,
    *,
    rank: int,
) -> tuple[NDArray, tuple[tuple[str, NDArray], ...]]:
    """Reproduce ``mgcv:::nat.param(..., type=1)`` for one marginal smooth."""
    X = np.asarray(basis, dtype=np.float64)
    S = np.asarray(penalty, dtype=np.float64)
    if X.ndim != 2 or S.shape != (X.shape[1], X.shape[1]):
        raise ValueError("factor-smooth basis and penalty dimensions do not agree")
    if X.shape[0] < X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError(
            "FactorSmooth marginal basis is rank deficient; use more distinct numeric values "
            "or a smaller k, or choose a suitable non-smooth feature."
        )

    _Q, R = np.linalg.qr(X, mode="reduced")
    R_inv = la.solve_triangular(R, np.eye(R.shape[0]), lower=False)
    transformed_penalty = R_inv.T @ S @ R_inv
    # R's ``eigen(..., symmetric=TRUE)`` uses LAPACK's MRRR driver.  The
    # zero-eigenvalue subspace is otherwise free to rotate, which matters here
    # because ``bs="fs"`` gives each null coordinate its own smoothing
    # parameter.  Pinning ``evr`` reproduces mgcv's stable orientation.
    eigenvalues, eigenvectors = la.eigh(
        0.5 * (transformed_penalty + transformed_penalty.T),
        driver="evr",
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues[:rank]
    if rank < 1 or rank > X.shape[1] or np.any(positive <= 0.0):
        raise ValueError("FactorSmooth marginal penalty has an invalid numerical rank")

    natural_map = R_inv @ eigenvectors
    natural_map[:, :rank] /= np.sqrt(positive)
    natural_basis = X @ natural_map

    penalized_scale = 1.0 / np.sqrt(np.mean(natural_basis[:, :rank] ** 2))
    natural_map[:, :rank] *= penalized_scale
    wiggle_diagonal = np.full(rank, penalized_scale**2, dtype=np.float64)

    null_dim = X.shape[1] - rank
    if null_dim:
        null_scale = 1.0 / np.sqrt(np.mean(natural_basis[:, rank:] ** 2))
        natural_map[:, rank:] *= null_scale

    wiggle = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    wiggle[np.arange(rank), np.arange(rank)] = wiggle_diagonal
    components: list[tuple[str, NDArray]] = [("wiggle", wiggle)]
    for null_index in range(null_dim):
        component = np.zeros_like(wiggle)
        coordinate = rank + null_index
        component[coordinate, coordinate] = 1.0
        components.append((f"null_{null_index}", component))
    return natural_map, tuple(components)


class FactorSmooth:
    """An mgcv-style factor-by-P-spline interaction.

    ``basis="fs"`` is fully penalized and retains independent level curves.
    ``basis="sz"`` represents centered sum-to-zero deviations; its specialized
    geometry is populated by the design-matrix builder.
    """

    structured_kind = "factor_smooth"
    requires_reml = True

    def __init__(
        self,
        variable: str,
        *,
        group: str,
        basis: Literal["fs", "sz"] = "fs",
        kind: str = "ps",
        k: int = 6,
        m: int = 2,
        unseen: Literal["population", "error"] = "population",
        missing: Literal["error"] = "error",
        lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
        name: str | None = None,
    ):
        if not isinstance(variable, str) or not variable:
            raise ValueError("variable must be a non-empty column name")
        if not isinstance(group, str) or not group:
            raise ValueError("group must be a non-empty column name")
        if variable == group:
            raise ValueError("variable and group must name different columns")
        if basis not in ("fs", "sz"):
            raise ValueError(f"basis must be 'fs' or 'sz', got {basis!r}")
        if kind != "ps":
            raise NotImplementedError("FactorSmooth currently supports only kind='ps'.")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k < 5:
            raise ValueError("k must be at least 5 for a cubic P-spline basis")
        if isinstance(m, bool) or not isinstance(m, int):
            raise TypeError("m must be an integer")
        if not 1 <= m < k:
            raise ValueError(f"m must be between 1 and k - 1, got m={m}, k={k}")
        if unseen not in ("population", "error"):
            raise ValueError(f"unseen must be 'population' or 'error', got {unseen!r}")
        if missing != "error":
            raise ValueError(f"missing must be 'error', got {missing!r}")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("name must be a non-empty string when supplied")

        valid_components = (
            {"wiggle", *(f"null_{index}" for index in range(m))} if basis == "fs" else {"wiggle"}
        )
        if isinstance(lambda_policy, dict):
            unknown = set(lambda_policy) - valid_components
            if unknown:
                raise ValueError(
                    "lambda_policy contains unknown component names "
                    f"{sorted(unknown)!r}; valid names are {sorted(valid_components)!r}"
                )
            invalid = {
                component
                for component, policy in lambda_policy.items()
                if not isinstance(policy, LambdaPolicy)
            }
            if invalid:
                raise TypeError(
                    "lambda_policy values must be LambdaPolicy instances; "
                    f"invalid components: {sorted(invalid)!r}"
                )
        elif lambda_policy is not None and not isinstance(lambda_policy, LambdaPolicy):
            raise TypeError("lambda_policy must be a LambdaPolicy, a component mapping, or None")

        self.variable = variable
        self.group = group
        self.basis = basis
        self.kind = kind
        self.k = k
        self.m = m
        self.unseen = unseen
        self.missing = missing
        self._lambda_policy = lambda_policy
        self.name = name or f"{variable}:{group}:{basis}"

        self._levels: list[Any] = []
        self._level_to_code: dict[Any, int] = {}
        self._spline = None
        self._natural_map = None
        self._base_penalty_components: tuple[tuple[str, Any], ...] = ()

    @property
    def parent_names(self) -> tuple[str, str]:
        """The numeric marginal and grouping columns read by this interaction."""
        return (self.variable, self.group)

    @staticmethod
    def _validate_numeric(values: NDArray) -> NDArray[np.float64]:
        try:
            numeric = np.asarray(values, dtype=np.float64).ravel()
        except (TypeError, ValueError) as exc:
            raise TypeError("FactorSmooth variable must be numeric.") from exc
        if not np.all(np.isfinite(numeric)):
            raise ValueError("FactorSmooth variable contains missing or non-finite values.")
        return numeric

    def _factorize_group(self, values: NDArray) -> NDArray[np.intp]:
        group_values = np.asarray(values).ravel()
        if np.any(pd.isna(group_values)):
            raise ValueError("FactorSmooth group contains missing values (NaN or None).")
        codes, uniques = pd.factorize(group_values, sort=True)
        if len(uniques) < 1:
            raise ValueError("FactorSmooth requires at least one fitted group level.")
        if self.basis == "sz" and len(uniques) < 2:
            raise ValueError("FactorSmooth basis='sz' requires at least two fitted group levels.")
        self._levels = uniques.tolist()
        self._level_to_code = {level: code for code, level in enumerate(self._levels)}
        return codes.astype(np.intp, copy=False)

    def _resolve_lambda_policies(self) -> dict[str, LambdaPolicy] | None:
        if self._lambda_policy is None:
            return None
        names = [name for name, _component in self._base_penalty_components]
        if isinstance(self._lambda_policy, LambdaPolicy):
            return {name: self._lambda_policy for name in names}
        return {name: self._lambda_policy.get(name, LambdaPolicy.estimate()) for name in names}

    def _build_marginal(
        self,
        x: NDArray,
    ) -> tuple[sp.csr_matrix, NDArray]:
        from superglm.features.spline import Spline

        spline = Spline(kind="ps", k=self.k, penalty="none", m=self.m)
        spline._place_knots(x)
        # mgcv's ``ps`` constructor lays the whole equally spaced knot
        # sequence out from boundaries expanded by 0.1% of the data range.
        # Ordinary SuperGLM P-splines preserve their pre-expansion interior
        # knots for backwards compatibility, so align this owned marginal
        # explicitly before constructing the factor-smooth marginal basis.
        x_range = spline._hi - spline._lo
        expanded_lo = spline._lo - 0.001 * x_range
        expanded_hi = spline._hi + 0.001 * x_range
        interior = np.linspace(
            expanded_lo,
            expanded_hi,
            spline.n_knots + 2,
        )[1:-1]
        spline._assemble_knot_vector(interior)
        spline._validate_m_orders_build()
        exact_basis = sp.csr_matrix(spline._basis_matrix(x), dtype=np.float64)
        raw_dense = exact_basis.toarray()
        if self.basis == "sz" and (
            raw_dense.shape[0] < self.k or np.linalg.matrix_rank(raw_dense) < self.k
        ):
            raise ValueError(
                "FactorSmooth marginal basis is rank deficient; use more distinct "
                "numeric values or a smaller k, or choose a suitable non-smooth feature."
            )
        penalty = spline._build_penalty()
        if self.basis == "fs":
            natural_map, components = _natural_parameterization(
                raw_dense,
                penalty,
                rank=self.k - self.m,
            )
        else:
            natural_map = np.eye(self.k, dtype=np.float64)
            components = (("wiggle", np.asarray(penalty, dtype=np.float64)),)
        self._spline = spline
        self._natural_map = natural_map
        self._base_penalty_components = components
        return exact_basis, raw_dense

    def _group_info(
        self,
        *,
        codes: NDArray,
        basis: sp.spmatrix | None = None,
        basis_unique: NDArray | None = None,
        bin_idx: NDArray | None = None,
    ) -> GroupInfo:
        n_levels = len(self._levels)
        coefficient_levels = n_levels if self.basis == "fs" else n_levels - 1
        return GroupInfo(
            columns=None,
            n_cols=coefficient_levels * self.k,
            penalized=True,
            lambda_policies=self._resolve_lambda_policies(),
            structured_kind="factor_smooth",
            factor_smooth_factor_basis=self.basis,
            factor_smooth_codes=codes,
            factor_smooth_basis=basis,
            factor_smooth_basis_unique=basis_unique,
            factor_smooth_bin_idx=bin_idx,
            factor_smooth_n_levels=n_levels,
            factor_smooth_block_size=self.k,
            factor_smooth_transform=self._natural_map,
            factor_smooth_levels=tuple(self._levels),
            repeated_penalty_components=self._base_penalty_components,
        )

    def build(
        self,
        x: NDArray,
        group: NDArray,
        specs: dict[str, Any],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        """Build one exact compact factor-by-spline block."""
        del specs, sample_weight
        numeric = self._validate_numeric(x)
        codes = self._factorize_group(group)
        if len(numeric) != len(codes):
            raise ValueError("FactorSmooth variable and group lengths differ.")
        exact_basis, _raw_dense = self._build_marginal(numeric)
        return self._group_info(codes=codes, basis=exact_basis)

    def build_discrete(
        self,
        x: NDArray,
        group: NDArray,
        specs: dict[str, Any],
        n_bins: int,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        """Build compact support-space geometry with a fixed natural basis."""
        del specs, sample_weight
        from superglm.group_matrix import _discretize_column

        numeric = self._validate_numeric(x)
        codes = self._factorize_group(group)
        if len(numeric) != len(codes):
            raise ValueError("FactorSmooth variable and group lengths differ.")
        self._build_marginal(numeric)
        support, bin_idx = _discretize_column(numeric, n_bins)
        basis_unique = self._spline._raw_basis_matrix(support)
        return self._group_info(
            codes=codes,
            basis_unique=np.asarray(basis_unique, dtype=np.float64),
            bin_idx=np.asarray(bin_idx, dtype=np.intp),
        )

    def validate_prediction_values(
        self,
        x: NDArray,
        group: NDArray,
    ) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
        """Validate new rows and return the numeric marginal and fitted-level codes."""
        numeric = self._validate_numeric(x)
        group_values = np.asarray(group).ravel()
        if len(numeric) != len(group_values):
            raise ValueError("FactorSmooth variable and group lengths differ.")
        if np.any(pd.isna(group_values)):
            raise ValueError("FactorSmooth group contains missing values (NaN or None).")
        codes = pd.Index(self._levels).get_indexer(group_values).astype(np.intp, copy=False)
        unseen_mask = codes < 0
        if self.unseen == "error" and np.any(unseen_mask):
            unseen = pd.unique(group_values[unseen_mask]).tolist()
            raise ValueError(f"Encountered unseen FactorSmooth levels: {unseen}.")
        return numeric, codes

    def marginal_basis(self, x: NDArray) -> NDArray[np.float64]:
        """Evaluate the fitted natural marginal basis on requested numeric values."""
        numeric = self._validate_numeric(x)
        if self._spline is None or self._natural_map is None:
            raise RuntimeError("FactorSmooth has not been fitted.")
        raw = np.asarray(self._spline._raw_basis_matrix(numeric), dtype=np.float64)
        return np.asarray(raw @ self._natural_map, dtype=np.float64)

    def score(
        self,
        x: NDArray,
        group: NDArray,
        beta: NDArray,
    ) -> NDArray[np.float64]:
        """Score fitted level-specific deviations without expanding factor geometry."""
        numeric, codes = self.validate_prediction_values(x, group)
        basis = self.marginal_basis(numeric)
        blocks = self._level_blocks(beta)
        result = np.zeros(len(numeric), dtype=np.float64)
        known = codes >= 0
        result[known] = np.einsum(
            "ij,ij->i",
            basis[known],
            blocks[codes[known]],
            optimize=True,
        )
        return result

    def _level_blocks(self, beta: NDArray) -> NDArray[np.float64]:
        """Return coefficients for every fitted level in marginal coordinates."""
        coefficient_levels = len(self._levels) if self.basis == "fs" else len(self._levels) - 1
        expected = coefficient_levels * self.k
        coefficients = np.asarray(beta, dtype=np.float64)
        if coefficients.shape != (expected,):
            raise ValueError(f"beta must have shape ({expected},).")
        free = coefficients.reshape(coefficient_levels, self.k)
        return free if self.basis == "fs" else expand_sum_to_zero_blocks(free)

    def transform(
        self,
        x: NDArray,
        group: NDArray,
    ) -> NDArray[np.float64]:
        """Materialize a small prediction matrix for compatibility and references."""
        numeric, codes = self.validate_prediction_values(x, group)
        basis = self.marginal_basis(numeric)
        coefficient_levels = len(self._levels) if self.basis == "fs" else len(self._levels) - 1
        result = np.zeros((len(numeric), coefficient_levels * self.k), dtype=np.float64)
        free_rows = np.flatnonzero((codes >= 0) & (codes < coefficient_levels))
        if len(free_rows):
            columns = codes[free_rows, None] * self.k + np.arange(self.k)[None, :]
            result[free_rows[:, None], columns] = basis[free_rows]
        if self.basis == "sz":
            final_rows = np.flatnonzero(codes == len(self._levels) - 1)
            if len(final_rows):
                result[final_rows] = np.tile(-basis[final_rows], (1, coefficient_levels))
        return result

    def reconstruct(self, beta: NDArray) -> dict[str, Any]:
        """Return fitted natural-basis coefficients by level."""
        blocks = self._level_blocks(beta)
        return {
            "variable": self.variable,
            "group": self.group,
            "basis": self.basis,
            "levels": self._levels.copy(),
            "coefficients": {
                level: block.copy() for level, block in zip(self._levels, blocks, strict=True)
            },
        }


__all__ = ["FactorSmooth"]
