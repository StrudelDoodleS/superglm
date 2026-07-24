"""Fully penalized smooth deviations by factor level."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from numpy.typing import NDArray

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
            "or a smaller k."
        )

    _Q, R = np.linalg.qr(X, mode="reduced")
    R_inv = la.solve_triangular(R, np.eye(R.shape[0]), lower=False)
    transformed_penalty = R_inv.T @ S @ R_inv
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (transformed_penalty + transformed_penalty.T))
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
    """An mgcv-style ``bs="fs"`` P-spline interaction.

    The fitted representation is populated by the design-matrix builder.  One
    shared marginal basis is repeated implicitly across every fitted factor
    level, with no reference level and no centering side condition.
    """

    structured_kind = "factor_smooth"
    requires_reml = True

    def __init__(
        self,
        variable: str,
        *,
        group: str,
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

        valid_components = {"wiggle", *(f"null_{index}" for index in range(m))}
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
        self.kind = kind
        self.k = k
        self.m = m
        self.unseen = unseen
        self.missing = missing
        self._lambda_policy = lambda_policy
        self.name = name or f"{variable}:{group}:fs"

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
        spline._validate_m_orders_build()
        exact_basis = sp.csr_matrix(spline._basis_matrix(x), dtype=np.float64)
        raw_dense = exact_basis.toarray()
        penalty = spline._build_penalty()
        rank = self.k - self.m
        natural_map, components = _natural_parameterization(
            raw_dense,
            penalty,
            rank=rank,
        )
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
        return GroupInfo(
            columns=None,
            n_cols=n_levels * self.k,
            penalized=True,
            lambda_policies=self._resolve_lambda_policies(),
            structured_kind="factor_smooth",
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


__all__ = ["FactorSmooth"]
