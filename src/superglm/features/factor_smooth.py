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
    normalization_mass: float | None = None,
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
    mass = float(n_rows) if normalization_mass is None else float(normalization_mass)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("factor-smooth normalization mass must be finite and positive")

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
    penalized_scale = np.sqrt(mass * rank / np.sum(1.0 / positive))
    natural_map[:, :rank] *= penalized_scale
    null_dim = k - rank
    if null_dim:
        natural_map[:, rank:] *= np.sqrt(mass)

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

    Both bases store each level's curve in the natural parameterization of the
    marginal P-spline: per level, the coordinates in which the wiggle penalty
    is diagonal, followed by the unpenalized polynomial coordinates.  The
    coefficients are therefore neither the B-spline coefficients nor mgcv's
    ``s(x, g, bs="sz")`` coordinates; read a level's curve through
    ``SuperGLM.factor_smooth`` instead.  (An ``sz`` model saved by 0.35.0 or
    earlier keeps the marginal-basis coordinates it was fitted in.)

    ``levels=`` binds the grouping column's level universe (spec 2026-08-11,
    §3.1).  Under ``basis="fs"`` a declared level with no training rows keeps
    its own curve block and shrinks to zero through the penalty.  ``basis="sz"``
    rejects one: its sum-to-zero contrast is what identifies the deviations, and
    an empty level makes that constraint vacuous.

    ``basis="sz"`` leaves each level's polynomial part (its line, with
    ``m=2``) unpenalized, as mgcv's ``sz`` does.  When the fit's data leave
    those lines without a finite or unique estimate -- some level's line
    separates the response, or every level holds fewer distinct ``x`` values
    than the line has coefficients -- the fit adds a second penalty
    component, ``"null"``, on every level's line: Marra & Wood's (2011)
    null-space penalty (mgcv's ``select=TRUE``), with its own smoothing
    parameter, which reads the lines as random effects shrunk toward the
    population curve, as ``basis="fs"`` does with its ``null_j``
    components.  Its smoothing parameter is estimated by REML unless
    ``lambda_policy`` is a single policy for the whole term.  Every other
    ``sz`` fit keeps the lines unpenalized.
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
        # What the fit's data identify of each sz level (#432), by code
        # (``_record_unidentified_levels``; ``_free_directions`` None until a
        # fit records it): the thin levels, the directions of the penalty's null
        # space each leaves free, the weightless ones, and the levels whose
        # unpenalized line separates the response.
        self._unidentified_levels: tuple[int, ...] = ()
        self._free_directions: tuple[NDArray, ...] | None = None
        self._weightless_levels: tuple[int, ...] = ()
        self._separated_levels: tuple[int, ...] = ()
        self._population_null_space: NDArray | None = None
        # Whether the fit penalized the levels' lines (#444,
        # ``_record_unidentified_levels``); a model saved before has none.
        self._lines_penalized = False

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
        geometry_weight: NDArray | None,
    ) -> tuple[PSpline, NDArray]:
        """Place the shared marginal knots and return its raw penalty."""
        from superglm.features.spline import PSpline, Spline

        spline = cast(PSpline, Spline(kind="ps", k=self.k, penalty="none", m=self.m))
        spline._place_knots(x, geometry_weight)
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
        geometry_weight: NDArray | None,
    ) -> sp.csr_matrix | None:
        """Build the marginal with bounded QR memory when its coordinates permit it."""
        if geometry_weight is None:
            weights = None
            normalization_mass = float(len(x))
        else:
            weights = np.asarray(geometry_weight, dtype=np.float64).ravel()
            if weights.shape != x.shape:
                raise ValueError("FactorSmooth geometry weights must match its numeric rows.")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("FactorSmooth geometry weights must be finite and non-negative.")
            if not np.any(weights > 0.0):
                raise ValueError("FactorSmooth geometry weights must retain at least one row.")
            normalization_mass = float(np.sum(weights, dtype=np.float64))
        spline, penalty = self._initialize_marginal_spline(x, weights)
        exact_basis: sp.csr_matrix | None

        if self._streaming_safe():
            qr_r: NDArray | None = None
            chunks: list[sp.csr_matrix] | None = [] if retain_basis else None
            for start in range(0, len(x), _MARGINAL_QR_CHUNK_ROWS):
                basis_chunk = sp.csr_matrix(
                    spline._basis_matrix(x[start : start + _MARGINAL_QR_CHUNK_ROWS]),
                    dtype=np.float64,
                )
                if weights is None:
                    qr_basis = basis_chunk
                else:
                    row_scale = np.sqrt(weights[start : start + _MARGINAL_QR_CHUNK_ROWS])
                    qr_basis = basis_chunk.multiply(row_scale[:, None]).tocsr()
                qr_r = _combine_qr_r(qr_r, qr_basis)
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
            qr_basis = raw_dense if weights is None else raw_dense * np.sqrt(weights[:, None])
            qr_r = np.asarray(np.linalg.qr(qr_basis, mode="r"), dtype=np.float64)
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
        natural_map, components = _natural_parameterization_from_r(
            qr_r,
            penalty,
            rank=self.k - self.m,
            n_rows=len(x),
            normalization_mass=normalization_mass,
        )
        if self.basis == "sz":
            # One-engine design §3.5, decision 5: sz takes fs's natural
            # parameterization.  Its sum-to-zero constraint is coefficientwise,
            # so the same per-level change of basis preserves it; the wiggle
            # penalty becomes diagonal (its square root exact) and the
            # polynomial null coordinates separate.  sz penalizes the wiggle
            # component alone, as before: the null coordinates stay unpenalized.
            components = components[:1]
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
        # The supplied stream governs both effective-level occupancy and the
        # marginal geometry: physical rows under prior semantics, replicated
        # row mass under frequency semantics.
        codes = self._factorize_group(group, sample_weight)
        if len(numeric) != len(codes):
            raise ValueError("FactorSmooth variable and group lengths differ.")
        exact_basis = self._build_marginal(
            numeric,
            retain_basis=True,
            geometry_weight=sample_weight,
        )
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
        # Same geometry rule as the exact path; only the final evaluated basis
        # is compressed onto the unweighted numeric support.
        codes = self._factorize_group(group, sample_weight)
        if len(numeric) != len(codes):
            raise ValueError("FactorSmooth variable and group lengths differ.")
        self._build_marginal(
            numeric,
            retain_basis=False,
            geometry_weight=sample_weight,
        )
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

    def _record_unidentified_levels(
        self,
        design,
        prior_weights: NDArray | None,
        response: NDArray | None = None,
        boundaries: tuple[str, ...] = (),
        *,
        penalize: bool = False,
    ) -> tuple[int, ...]:
        """Record what the fit's data identify of each ``sz`` level (#432); return the separated.

        A level whose rows of positive weight hold fewer distinct ``x`` values
        than the penalty's null space ``N_P`` has directions of its polynomial
        deviation the data cannot tell from the main effect
        (``layout.sz_level_identification``).  A level whose unpenalized line
        separates the response (``boundaries``, the family's response
        boundaries reached at infinite ``eta``) has no finite line
        (``diagnostics.separation.separated_factor_smooth_levels``).  Both
        stay out of the population curve (``_population_map``).

        ``penalize`` (a fit about to run on ``design``, #444): when some line
        separates, or every level is thin, the lines have no finite or no
        unique estimate that a convention could fix away from the levels'
        own ``x`` values.  The design then takes the null-space penalty
        (``_level_line_penalty``) and no level is left unidentified, so the
        population curve is the main effect and every level predicts its
        own fitted curve.  The decision reads the wiggle penalty's null space
        alone, so it is the same whether or not ``design`` already carries
        the penalty from an earlier fit of the same data.
        """
        from superglm.diagnostics.separation import separated_factor_smooth_levels
        from superglm.solvers._structured.layout import sz_level_identification

        self._unidentified_levels = ()
        self._free_directions = ()
        self._weightless_levels = ()
        self._separated_levels = ()
        self._population_null_space = None
        if penalize:
            self._lines_penalized = False
        if self.basis != "sz":
            return ()
        if penalize:
            design.repeated_penalty_components = self._base_penalty_components
            design.lambda_policies = self._resolve_lambda_policies()
        rows = sz_level_identification(design, prior_weights)
        separated: tuple[int, ...] = ()
        if response is not None and boundaries:
            separated = separated_factor_smooth_levels(
                design, rows.null_space, prior_weights, response, boundaries
            )
        if penalize and (separated or len(rows.thin) == len(self._levels)):
            suffix, omega = self._level_line_penalty()
            design.repeated_penalty_components = (
                *self._base_penalty_components,
                (suffix, omega),
            )
            if isinstance(self._lambda_policy, LambdaPolicy):
                design.lambda_policies = {
                    **(design.lambda_policies or {}),
                    suffix: self._lambda_policy,
                }
            self._lines_penalized = True
            return separated
        self._unidentified_levels = rows.thin
        self._free_directions = rows.free
        self._weightless_levels = rows.weightless
        self._separated_levels = separated
        self._population_null_space = rows.null_space if (rows.thin or separated) else None
        return separated

    def _level_line_penalty(self) -> tuple[str, NDArray[np.float64]]:
        """``("null", S*)``: the penalty on each level's polynomial part (#444).

        Marra & Wood's (2011) null-space penalty: the wiggle penalty's
        eigenvectors with its zero eigenvalues set to one and the rest to
        zero.  In the natural parameterization the wiggle penalty is diagonal
        (``_natural_parameterization_from_r``), so ``S*`` is the indicator of
        its zero diagonal, exactly.  Those coordinates are orthonormal over
        the term's weighted rows (unit mean square each), so ``beta_l' S*
        beta_l`` is the mean square of level ``l``'s polynomial part over the
        data: invariant under any rotation of the null coordinates, as the
        streamed marginal QR may choose.
        """
        wiggle = np.asarray(self._base_penalty_components[0][1], dtype=np.float64)
        diagonal = np.diag(wiggle)
        if (
            np.count_nonzero(wiggle - np.diag(diagonal))
            or np.count_nonzero(diagonal == 0.0) != self.m
        ):
            raise RuntimeError("an sz level-line penalty needs the natural parameterization")
        return "null", np.diag((diagonal == 0.0).astype(np.float64))

    @property
    def _unidentified_level_names(self) -> tuple:
        """The fitted ``sz`` levels the data identify only in part (``_record_unidentified_levels``)."""
        return tuple(self._levels[code] for code in getattr(self, "_unidentified_levels", ()))

    @property
    def _has_population_offset(self) -> bool:
        """Whether some level stays out of the population curve, which then moves off the main effect."""
        return bool(getattr(self, "_unidentified_levels", ())) or bool(
            getattr(self, "_separated_levels", ())
        )

    @property
    def _population_convention(self) -> str:
        """How the population curve is fixed (``_population_map``).

        ``"main"`` (no level left out: the main effect, as the sum-to-zero
        constraint makes it), ``"mean"`` (the mean of the levels the data
        identify), ``"separated_mean"`` (every level the data identify
        separates: their mean, which follows how far the fit walked their
        lines) or ``"canonical"`` (every level thin).
        """
        if not self._has_population_offset:
            return "main"
        excluded = set(self._unidentified_levels) | set(self._separated_levels)
        if len(excluded) < len(self._levels):
            return "mean"
        if len(self._unidentified_levels) < len(self._levels):
            return "separated_mean"
        return "canonical"

    def _population_map(self) -> tuple[NDArray, NDArray] | None:
        """``(levels, V)``: the population offset ``c = sum_i V_i beta_{levels_i}`` (natural basis).

        A thin level ``t`` leaves a part ``r_t`` of its polynomial deviation
        free (``r_t`` in the span of ``_free_directions[t]``): shifting it,
        every level by ``-R / K`` and the main effect by ``+R / K`` (``R =
        sum_t r_t``) keeps the sum-to-zero constraint, every other level's
        curve and every level's fit at its own rows.  A separated level's
        line walks the same way without bound.  The fit's coefficients are
        one point of that family, and the main effect, so the population
        curve, moved with it.  The population curve is therefore
        ``main + b(x)' c`` with ``c`` the polynomial part of the mean of the
        levels the data identify, ``c = P_N sum_{l in I} beta_l / |I|``
        (``P_N = N_P N_P'``): along the family each of them moves by
        ``-R / K``, so ``c`` does as well and ``main + b(x)' c`` does not
        depend on the fit's point.  This is the constraint without the levels
        left out, as mgcv drops unused factor levels before it fits and lme4
        predicts the population for a level it did not see.  ``V`` is one
        ``(k, k)`` matrix shared by the levels, or one per level.

        With every level separated or thin there is no such mean.  If some
        level is not thin (every one of those separates) the mean is over
        them, and it follows the separated lines (a warning at fit).  If every
        level is thin the population is the canonical point of the family,
        where each level's free part is zero, ``Pi_t (beta_t + s_t - S / K)
        = 0`` (``Pi_t`` the projector on ``_free_directions[t]``, ``S = sum_t
        s_t``): ``S = -(I - M / K)^+ sum_t Pi_t beta_t`` with ``M = sum_t
        Pi_t``, unique when no direction of ``N_P`` is free in every level,
        and ``c = S / K``.  ``None`` when no level is left out.
        """
        if not self._has_population_offset:
            return None
        n_levels = len(self._levels)
        null_space = np.asarray(self._population_null_space, dtype=np.float64)
        projector = null_space @ null_space.T
        thin = tuple(self._unidentified_levels)
        excluded = set(thin) | set(self._separated_levels)
        identified = [level for level in range(n_levels) if level not in excluded]
        if not identified:
            identified = [level for level in range(n_levels) if level not in set(thin)]
        if identified:
            return np.asarray(identified, dtype=np.intp), projector / len(identified)
        free = [np.asarray(f, dtype=np.float64) for f in self._free_directions]
        projectors = np.stack([f @ f.T for f in free])
        local = null_space.T @ (np.eye(len(projector)) - projectors.sum(axis=0) / n_levels)
        local = local @ null_space
        values, vectors = np.linalg.eigh(0.5 * (local + local.T))
        # ``I - M / K`` has its eigenvalues in [0, 1]: the eigensolver resolves
        # them to ``m eps`` (its backward error on a unit-norm matrix)
        keep = values > len(values) * np.finfo(np.float64).eps
        inverse = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
        lifted = -(null_space @ inverse @ null_space.T) / n_levels
        return np.asarray(thin, dtype=np.intp), lifted @ projectors

    def _population_offset(self, blocks: NDArray) -> NDArray[np.float64]:
        """The population curve's offset ``c`` (natural basis) from the level blocks (``_population_map``)."""
        mapping = self._population_map()
        if mapping is None:
            return np.zeros(blocks.shape[1])
        levels, V = mapping
        if V.ndim == 2:
            return V @ np.sum(blocks[levels], axis=0)
        return np.einsum("lij,lj->i", V, blocks[levels], optimize=True)

    def _population_contrast(self) -> NDArray[np.float64] | None:
        """``c`` as a map of the term's ``(K - 1) k`` free coefficients, ``(k, (K - 1) k)``.

        The free blocks are levels ``0 .. K - 2``; the last level is minus
        their sum (``expand_sum_to_zero_blocks``).  ``None`` when no level is
        left out of the population.
        """
        mapping = self._population_map()
        if mapping is None:
            return None
        levels, V = mapping
        n_levels, k = len(self._levels), self.k
        raw = np.zeros((n_levels, k, k))
        raw[levels] = V
        free = raw[:-1] - raw[-1][None, :, :]
        return np.ascontiguousarray(free.transpose(1, 0, 2).reshape(k, (n_levels - 1) * k))

    def _identified_blocks(
        self, blocks: NDArray
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(blocks', c)``: each level's coefficients as predicted, and the population offset.

        A thin level keeps what its rows identify and takes the population's
        part in the directions they leave free, ``beta_t - Pi_t (beta_t -
        c)`` (``Pi_t`` the projector on ``_free_directions[t]``): along the
        family ``beta_t - c`` moves by ``r_t``, which ``Pi_t`` removes, so the
        level's prediction moves by ``-R / K`` exactly as the main effect
        moves by ``+R / K``, and at its own rows ``b(x)' Pi_t = 0`` keeps the
        fit.  A level without weight is predicted at the population value.
        Every other level keeps its coefficients.
        """
        coefficients = np.array(blocks, dtype=np.float64, copy=True)
        offset = self._population_offset(coefficients)
        weightless = set(getattr(self, "_weightless_levels", ()))
        for level, free in zip(self._unidentified_levels, self._free_directions, strict=True):
            if level in weightless:
                coefficients[level] = offset
                continue
            directions = np.asarray(free, dtype=np.float64)
            coefficients[level] -= directions @ (directions.T @ (blocks[level] - offset))
        return coefficients, offset

    def _score_identified(
        self,
        x: NDArray,
        group: NDArray,
        beta: NDArray,
        *,
        population: bool,
    ) -> tuple[NDArray[np.float64], tuple]:
        """Score the term as predicted when some level stays out of the population (#432).

        Returns the term's contribution and the thin levels among the rows.
        Each level takes its ``_identified_blocks`` coefficients; an unseen
        level, and every row under ``population``, the population offset
        ``b(x)' c``.
        """
        if population:
            numeric, _ = self._validated_prediction_inputs(x, group)
            codes = np.full(len(numeric), -1, dtype=np.intp)
        else:
            numeric, codes = self.validate_prediction_values(x, group)
        basis = self.marginal_basis(numeric)
        blocks, offset = self._identified_blocks(self._level_blocks(beta))
        coefficients = blocks[np.maximum(codes, 0)]
        coefficients[codes < 0] = offset
        result = np.einsum("ij,ij->i", basis, coefficients, optimize=True)
        thin = np.isin(codes, np.asarray(self._unidentified_levels, dtype=np.intp))
        named = tuple(self._levels[code] for code in np.unique(codes[thin]))
        return result, named

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
        """Return fitted natural-basis coefficients by level.

        With ``sz`` levels left out of the population (#432,
        ``_population_map``) each level's coefficients are its deviation from
        the population curve as predicted, ``_identified_blocks`` minus the
        offset ``c``, which the result carries as ``population_offset`` (the
        main effect's reported curve adds ``b(x)' c``).
        """
        blocks = self._level_blocks(beta)
        extra: dict[str, Any] = {}
        if self.basis == "sz" and self._has_population_offset:
            identified, offset = self._identified_blocks(blocks)
            blocks = identified - offset[None, :]
            extra["population_offset"] = offset
        return {
            "variable": self.variable,
            "group": self.group,
            "basis": self.basis,
            "levels": self._levels.copy(),
            "coefficients": {
                level: block.copy() for level, block in zip(self._levels, blocks, strict=True)
            },
            **extra,
        }


def population_curve_terms(name: Any, interaction_specs: Any) -> list[tuple[Any, FactorSmooth]]:
    """``(interaction name, spec)`` of the ``sz`` terms that move feature ``name``'s population curve.

    An ``sz`` term on ``x`` needs the global Spline on ``x`` and, with levels
    left out of its population (#432, ``FactorSmooth._population_map``), its
    population curve is ``main(x) + b(x)' c``: the reported main-effect curve
    carries ``b(x)' c``.
    """
    return [
        (key, spec)
        for key, spec in (interaction_specs or {}).items()
        if isinstance(spec, FactorSmooth)
        and spec.basis == "sz"
        and spec.variable == name
        and spec._has_population_offset
    ]


def population_curve_shift(
    name: Any, x: NDArray, interaction_specs: Any, groups: Any, beta: NDArray
) -> NDArray[np.float64] | None:
    """``sum_terms b(x)' c`` on ``x`` for feature ``name`` (``population_curve_terms``), or None."""
    terms = population_curve_terms(name, interaction_specs)
    if not terms:
        return None
    shift = np.zeros(len(x), dtype=np.float64)
    for key, spec in terms:
        coefficients = np.concatenate([beta[g.sl] for g in groups if g.feature_name == key])
        offset = spec._population_offset(spec._level_blocks(coefficients))
        shift += spec.marginal_basis(np.asarray(x, dtype=np.float64)) @ offset
    return shift


def with_population_curve(
    raw: dict[str, Any], name: Any, interaction_specs: Any, groups: Any, beta: NDArray
) -> dict[str, Any]:
    """A main-effect curve shifted onto its ``sz`` terms' population curve (#432).

    With ``sz`` levels left out of the population the curve the model
    predicts for the population is ``main(x) + b(x)' c``
    (``population_curve_shift``), so every report of the main effect's curve
    (relativities, reconstruct, term inference, bands) is that one; a curve
    no ``sz`` term moves is returned as it is.
    """
    if "x" not in raw or "log_relativity" not in raw:
        return raw
    shift = population_curve_shift(name, raw["x"], interaction_specs, groups, beta)
    if shift is None:
        return raw
    shifted = dict(raw)
    shifted["log_relativity"] = np.asarray(raw["log_relativity"], dtype=np.float64) + shift
    shifted["relativity"] = np.exp(shifted["log_relativity"])
    return shifted


__all__ = ["FactorSmooth"]
