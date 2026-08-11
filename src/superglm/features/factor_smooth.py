"""Fully penalized smooth deviations by factor level."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.factor_smooth_geometry import expand_sum_to_zero_blocks
from superglm.types import GroupInfo, LambdaPolicy

if TYPE_CHECKING:
    from superglm.features.spline import PSpline


_MARGINAL_QR_CHUNK_ROWS = 65_536
_MarginalBuildBackend = Literal["streamed_tsqr", "dense_qr_compat"]


def _combine_qr_r(
    current: NDArray | None,
    basis_chunk: sp.csr_matrix,
) -> NDArray:
    """Merge one bounded basis chunk into a tall-skinny QR factor."""
    chunk_r = np.asarray(np.linalg.qr(basis_chunk.toarray(), mode="r"), dtype=np.float64)
    if current is None:
        return chunk_r
    return np.asarray(
        np.linalg.qr(np.vstack((current, chunk_r)), mode="r"),
        dtype=np.float64,
    )


def _natural_parameterization_from_r(
    R: NDArray,
    penalty: NDArray,
    *,
    rank: int,
    n_rows: int,
) -> tuple[NDArray, tuple[tuple[str, NDArray], ...]]:
    """Build a QR-whitened natural parameterization without materializing ``Q``."""
    R_array = np.asarray(R, dtype=np.float64)
    S = np.asarray(penalty, dtype=np.float64)
    if R_array.ndim != 2 or R_array.shape[0] != R_array.shape[1]:
        raise ValueError("factor-smooth QR factor must be square")
    if S.shape != R_array.shape:
        raise ValueError("factor-smooth QR factor and penalty dimensions do not agree")
    k = R_array.shape[0]
    if n_rows < k or np.linalg.matrix_rank(R_array) < k:
        raise ValueError(
            "FactorSmooth marginal basis is rank deficient; use more distinct numeric values "
            "or a smaller k, or choose a suitable non-smooth feature."
        )

    R_inv = la.solve_triangular(R_array, np.eye(k), lower=False)
    transformed_penalty = R_inv.T @ S @ R_inv
    # The zero-eigenvalue eigenspace can rotate freely.  Each FS null
    # coordinate has its own smoothing parameter, so explicitly select the
    # MRRR driver to keep that coordinate system deterministic under the
    # tested numerical contract.
    eigenvalues, eigenvectors = la.eigh(
        0.5 * (transformed_penalty + transformed_penalty.T),
        driver="evr",
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues[:rank]
    if rank < 1 or rank > k or np.any(positive <= 0.0):
        raise ValueError("FactorSmooth marginal penalty has an invalid numerical rank")

    natural_map = R_inv @ eigenvectors
    natural_map[:, :rank] /= np.sqrt(positive)
    penalized_scale = np.sqrt(n_rows * rank / np.sum(1.0 / positive))
    natural_map[:, :rank] *= penalized_scale
    null_dim = k - rank
    if null_dim:
        natural_map[:, rank:] *= np.sqrt(n_rows)

    wiggle = np.zeros((k, k), dtype=np.float64)
    wiggle[np.arange(rank), np.arange(rank)] = penalized_scale**2
    components: list[tuple[str, NDArray]] = [("wiggle", wiggle)]
    for null_index in range(null_dim):
        component = np.zeros_like(wiggle)
        coordinate = rank + null_index
        component[coordinate, coordinate] = 1.0
        components.append((f"null_{null_index}", component))
    return natural_map, tuple(components)


class FactorSmooth:
    """A factor-by-P-spline interaction.

    ``basis="fs"`` is fully penalized and retains independent level curves.
    ``basis="sz"`` represents centered sum-to-zero deviations; its specialized
    geometry is populated by the design-matrix builder.

    ``levels=`` binds the grouping column's level universe (spec 2026-08-11,
    §3.1).  Under ``basis="fs"`` a declared level with no training rows keeps
    its own curve block and shrinks to zero through the penalty.  ``basis="sz"``
    rejects one: its sum-to-zero contrast is what identifies the deviations, and
    an empty level makes that constraint vacuous.
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
        levels=None,
        unseen: Literal["population", "error"] = "population",
        missing: Literal["error"] = "error",
        lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
        name: str | None = None,
    ):
        from superglm.features._level_source import resolve_level_source

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
        self.basis: Literal["fs", "sz"] = basis
        self.kind = kind
        self.k = k
        self.m = m
        self.unseen = unseen
        self.missing = missing
        self._lambda_policy = lambda_policy
        self.name = name or f"{variable}:{group}:{basis}"

        self._declared_levels: list | None = (
            None if levels is None else resolve_level_source(levels, context="FactorSmooth")
        )
        self._level_source: str = "declared" if levels is not None else "inferred"
        self._levels: list[Any] = []
        self._level_to_code: dict[Any, int] = {}
        self._spline: PSpline | None = None
        self._natural_map = None
        self._base_penalty_components: tuple[tuple[str, Any], ...] = ()
        self._marginal_build_backend: _MarginalBuildBackend | None = None

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

    def adopt_dtype_categories(self, categories: list) -> None:
        """Adopt a dtype-declared universe unless one is already declared.

        Not reached by the main-loop hooks this release: FactorSmooth lives in
        the interaction specs, and dm_builder/binding_ops bind main-loop
        features only. Declare ``levels=`` explicitly; this hook exists so the
        wiring lands in one place when interaction binding is added.
        """
        if self._declared_levels is None:
            from superglm.features._level_source import resolve_level_source

            self._declared_levels = resolve_level_source(list(categories), context="FactorSmooth")
            self._level_source = "dtype"

    def apply_level_binding(self, binding) -> None:
        """Adopt a full-frame universe when nothing more specific declared one.

        Only the levels are read: a penalized term has no base level, so its
        bindings carry ``base=None`` and there is nothing to pin.
        """
        if self._declared_levels is None and binding.levels is not None:
            self._declared_levels = list(binding.levels)
            self._level_source = "full-frame"

    def resolve_binding(self, values: NDArray, sample_weight=None):
        """Compute this spec's full-frame group binding without mutating the spec."""
        import copy

        from superglm.types import LevelBinding

        del sample_weight
        # Factorize on a throwaway copy so the universe and its NaN checks stay
        # single-sourced in `_factorize_group`.
        probe = copy.deepcopy(self)
        probe._factorize_group(values)
        return LevelBinding(levels=tuple(probe._levels), base=None)

    def _declared_codes(
        self,
        group_values: NDArray,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.intp]:
        """Code group values against the bound universe, rejecting anything outside it.

        Missing values are already rejected by the caller, so a -1 here can only
        mean data the declaration does not admit.

        ``sample_weight`` is read for the ``sz`` empty-level guard alone, which
        asks whether a declared level has any EFFECTIVE rows; ``None`` keeps the
        physical-row count. Nothing else here is weighted.
        """
        codes = pd.Index(self._levels).get_indexer(group_values).astype(np.intp, copy=False)
        if np.any(codes < 0):
            outside = group_values[codes < 0]
            raise ValueError(
                f"Training data contains levels outside the declared level universe: "
                f"{sorted(set(outside.tolist()), key=str)}. Declared: "
                f"{sorted(self._levels, key=str)}. Widen levels= or fix the column."
            )
        if self.basis == "sz":
            # An empty level does not shrink under sz, it breaks it: the
            # sum-to-zero constraint is what identifies these deviations
            # against the population smooth, and a level with no rows absorbs
            # any common curve, so the constraint stops binding.  Measured on
            # a three-level fit, adding one empty declared level moved the
            # penalized system's smallest eigenvalue 5.9e-1 -> 4.4e-10 and
            # max|beta| 1.8 -> 4.9e3.  fs has no such gap: every coordinate
            # carries a penalty, so an empty block sits at its own lambda and
            # the observed levels' coefficients do not move.
            #
            # Effective rows, not physical ones. A level whose every row carries
            # weight 0 contributes exactly nothing to the fitted system, so it
            # is as empty as a level with no rows at all and recreates the same
            # near-singularity -- but a physical `bincount` counts it as
            # present and waves it through. This mirrors `Categorical.build`,
            # which has always measured occupancy as total weight when weights
            # are supplied.
            if sample_weight is None:
                effective = np.bincount(codes, minlength=len(self._levels)).astype(np.float64)
            else:
                weights = np.asarray(sample_weight, dtype=np.float64).ravel()
                if weights.size != codes.size:
                    raise ValueError(
                        f"FactorSmooth sample_weight length {weights.size} != group length "
                        f"{codes.size}."
                    )
                effective = np.bincount(codes, weights=weights, minlength=len(self._levels))
            unobserved = [
                level
                for level, weight in zip(self._levels, effective, strict=True)
                if weight <= 0.0
            ]
            if unobserved:
                raise ValueError(
                    f"FactorSmooth basis='sz' cannot carry a declared group level with "
                    f"no training rows: {sorted(unobserved, key=str)}. Its sum-to-zero "
                    f"contrast stops identifying the deviations once a level is empty. "
                    f"Use basis='fs', which penalizes every coordinate, or narrow levels=."
                )
        return codes

    def _factorize_group(
        self,
        values: NDArray,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> NDArray[np.intp]:
        group_values = np.asarray(values).ravel()
        if np.any(pd.isna(group_values)):
            raise ValueError("FactorSmooth group contains missing values (NaN or None).")
        if self._declared_levels is not None:
            # A declared universe is >= 2 labels by construction, so the fitted
            # minimums below are already satisfied.
            self._levels = list(self._declared_levels)
            codes = self._declared_codes(group_values, sample_weight)
        else:
            codes, uniques = pd.factorize(group_values, sort=True)
            if len(uniques) < 1:
                raise ValueError("FactorSmooth requires at least one fitted group level.")
            if self.basis == "sz" and len(uniques) < 2:
                raise ValueError(
                    "FactorSmooth basis='sz' requires at least two fitted group levels."
                )
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

    def _streaming_safe(self) -> bool:
        """Return whether QR sign/null rotations preserve the declared penalty geometry."""
        if self.basis == "sz":
            return True
        if self.m > 2:
            return False
        if self.m <= 1:
            return True
        if self._lambda_policy is None or isinstance(self._lambda_policy, LambdaPolicy):
            return True
        policies = [
            self._lambda_policy.get(f"null_{index}", LambdaPolicy.estimate())
            for index in range(self.m)
        ]
        return all(policy == policies[0] for policy in policies[1:])

    def _initialize_marginal_spline(
        self,
        x: NDArray,
    ) -> tuple[PSpline, NDArray]:
        """Place the shared marginal knots and return its raw penalty."""
        from superglm.features.spline import PSpline, Spline

        spline = cast(PSpline, Spline(kind="ps", k=self.k, penalty="none", m=self.m))
        spline._place_knots(x)
        # Factor-smooth marginals place one equally spaced knot sequence
        # across boundaries expanded by 0.1% of the data range. Ordinary
        # SuperGLM P-splines preserve their pre-expansion interior knots for
        # backwards compatibility, so align this owned marginal explicitly.
        boundary = spline.fitted_boundary
        if boundary is None:  # pragma: no cover - populated by _place_knots
            raise RuntimeError("FactorSmooth marginal spline has no fitted boundary.")
        lo, hi = boundary
        x_range = hi - lo
        expanded_lo = lo - 0.001 * x_range
        expanded_hi = hi + 0.001 * x_range
        interior = np.linspace(
            expanded_lo,
            expanded_hi,
            self.k - 2,
        )[1:-1]
        spline._assemble_knot_vector(interior)
        spline._validate_m_orders_build()
        return spline, np.asarray(spline._build_penalty(), dtype=np.float64)

    def _build_marginal(
        self,
        x: NDArray,
        *,
        retain_basis: bool,
    ) -> sp.csr_matrix | None:
        """Build the marginal with bounded QR memory when its coordinates permit it."""
        spline, penalty = self._initialize_marginal_spline(x)
        exact_basis: sp.csr_matrix | None

        if self._streaming_safe():
            qr_r: NDArray | None = None
            chunks: list[sp.csr_matrix] | None = [] if retain_basis else None
            for start in range(0, len(x), _MARGINAL_QR_CHUNK_ROWS):
                basis_chunk = sp.csr_matrix(
                    spline._basis_matrix(x[start : start + _MARGINAL_QR_CHUNK_ROWS]),
                    dtype=np.float64,
                )
                qr_r = _combine_qr_r(qr_r, basis_chunk)
                if chunks is not None:
                    chunks.append(basis_chunk)
            if qr_r is None:  # pragma: no cover - group validation rejects zero rows
                raise RuntimeError("FactorSmooth marginal QR received no rows.")
            exact_basis = (
                sp.csr_matrix(sp.vstack(chunks, format="csr"), dtype=np.float64)
                if chunks is not None
                else None
            )
            self._marginal_build_backend = "streamed_tsqr"
        else:
            if retain_basis:
                exact_basis = sp.csr_matrix(spline._basis_matrix(x), dtype=np.float64)
                raw_dense = exact_basis.toarray()
            else:
                exact_basis = None
                raw_dense = np.asarray(spline._raw_basis_matrix(x), dtype=np.float64)
            qr_r = np.asarray(np.linalg.qr(raw_dense, mode="r"), dtype=np.float64)
            self._marginal_build_backend = "dense_qr_compat"

        if (
            qr_r.shape != (self.k, self.k)
            or len(x) < self.k
            or np.linalg.matrix_rank(qr_r) < self.k
        ):
            raise ValueError(
                "FactorSmooth marginal basis is rank deficient; use more distinct "
                "numeric values or a smaller k, or choose a suitable non-smooth feature."
            )
        if self.basis == "fs":
            natural_map, components = _natural_parameterization_from_r(
                qr_r,
                penalty,
                rank=self.k - self.m,
                n_rows=len(x),
            )
        else:
            natural_map = np.eye(self.k, dtype=np.float64)
            components = (("wiggle", penalty),)
        self._spline = spline
        self._natural_map = natural_map
        self._base_penalty_components = components
        return exact_basis

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
        del specs
        numeric = self._validate_numeric(x)
        # The weights reach the universe check only: they decide which declared
        # levels are EFFECTIVELY empty (see `_declared_codes`). The basis and
        # the penalty geometry stay weight-free as before.
        codes = self._factorize_group(group, sample_weight)
        if len(numeric) != len(codes):
            raise ValueError("FactorSmooth variable and group lengths differ.")
        exact_basis = self._build_marginal(numeric, retain_basis=True)
        if exact_basis is None:  # pragma: no cover - required by retain_basis
            raise RuntimeError("FactorSmooth exact marginal basis was not retained.")
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
        del specs
        from superglm.group_matrix import _discretize_column

        numeric = self._validate_numeric(x)
        # Same rule as the exact path: weights inform the empty-level check and
        # nothing else. The support grid and the natural basis are unweighted.
        codes = self._factorize_group(group, sample_weight)
        if len(numeric) != len(codes):
            raise ValueError("FactorSmooth variable and group lengths differ.")
        self._build_marginal(numeric, retain_basis=False)
        support, bin_idx = _discretize_column(numeric, n_bins)
        spline = self._spline
        if spline is None:  # pragma: no cover - populated by _build_marginal
            raise RuntimeError("FactorSmooth marginal spline was not initialized.")
        basis_unique = spline._raw_basis_matrix(support)
        return self._group_info(
            codes=codes,
            basis_unique=np.asarray(basis_unique, dtype=np.float64),
            bin_idx=np.asarray(bin_idx, dtype=np.intp),
        )

    def _validated_prediction_inputs(
        self,
        x: NDArray,
        group: NDArray,
    ) -> tuple[NDArray[np.float64], NDArray]:
        """Validate prediction shape and missingness without applying unseen policy."""
        numeric = self._validate_numeric(x)
        group_values = np.asarray(group).ravel()
        if len(numeric) != len(group_values):
            raise ValueError("FactorSmooth variable and group lengths differ.")
        if np.any(pd.isna(group_values)):
            raise ValueError("FactorSmooth group contains missing values (NaN or None).")
        return numeric, group_values

    def validate_population_prediction_values(
        self,
        x: NDArray,
        group: NDArray,
    ) -> None:
        """Validate rows for a population prediction that skips this deviation."""
        self._validated_prediction_inputs(x, group)

    def validate_prediction_values(
        self,
        x: NDArray,
        group: NDArray,
    ) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
        """Validate new rows and return the numeric marginal and fitted-level codes."""
        numeric, group_values = self._validated_prediction_inputs(x, group)
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
