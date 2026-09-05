"""Internal predictor configuration and family-ordered compilation."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm._frame import EagerFrame
from superglm._predictor_compiler import CompiledPredictorDesign, compile_predictor_design
from superglm.distributional.family import ParameterSpec
from superglm.distributional.weights import ResolvedLikelihoodWeights
from superglm.group_matrix import (
    CrossMatrixExecutionPlan,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm.links import Link, resolve_link
from superglm.model.reml_setup import collect_reml_groups
from superglm.reml.penalty_algebra import build_penalty_components
from superglm.types import FeatureSpec, PenaltyComponent


class ShapeConstraintIgnoredWarning(UserWarning):
    """A feature asked for a shape constraint the distributional path cannot apply yet."""


def _constrained_features(predictor: Predictor) -> list[tuple[str, str, str]]:
    """``(feature, kind, mode)`` for every feature spec carrying a shape constraint."""
    found: list[tuple[str, str, str]] = []
    for name, spec in predictor.features.items():
        inner = getattr(spec, "basis", spec)
        kind = getattr(inner, "constraint_kind", getattr(inner, "monotone", None))
        if kind is not None:
            mode = getattr(inner, "constraint_mode", getattr(inner, "monotone_mode", "postfit"))
            found.append((name, str(kind), str(mode)))
    return found


@dataclass(frozen=True)
class Predictor:
    """Immutable configuration for one natural-parameter predictor.

    Feature specifications remain caller-owned configuration objects here.
    ``compile_predictor_design`` deep-clones their complete graph before any
    build-time mutation occurs.
    """

    name: str
    features: Mapping[str, FeatureSpec]
    link: str | Link | None = None
    intercept: bool = True
    interactions: tuple[tuple[str, str], ...] = ()
    interaction_specs: Mapping[str, Any] = field(default_factory=dict)
    interaction_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("predictor name must be a non-empty string")
        if not isinstance(self.features, Mapping):
            raise TypeError("features must be a mapping")
        owned_features = dict(self.features)
        for name, feature in owned_features.items():
            if not isinstance(name, str) or not name:
                raise ValueError("feature names must be non-empty strings")
            if not isinstance(feature, FeatureSpec):
                raise TypeError(f"feature {name!r} does not implement FeatureSpec")

        interactions: list[tuple[str, str]] = []
        for pair in self.interactions:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("each interaction must be a two-name tuple")
            left, right = pair
            if not isinstance(left, str) or not isinstance(right, str):
                raise TypeError("interaction names must be strings")
            missing = [name for name in pair if name not in owned_features]
            if missing:
                raise ValueError(f"interaction references unknown features: {missing}")
            interactions.append((left, right))

        if not isinstance(self.interaction_specs, Mapping):
            raise TypeError("interaction_specs must be a mapping")
        owned_interaction_specs = dict(self.interaction_specs)
        for interaction_name, interaction in owned_interaction_specs.items():
            if not isinstance(interaction_name, str) or not interaction_name:
                raise ValueError("interaction spec names must be non-empty strings")
            parents = getattr(interaction, "parent_names", None)
            if (
                not isinstance(parents, tuple)
                or len(parents) != 2
                or not all(isinstance(parent, str) for parent in parents)
            ):
                raise TypeError("interaction specs must expose two string parent_names")
            missing = [parent for parent in parents if parent not in owned_features]
            if missing:
                raise ValueError(f"interaction spec references unknown features: {missing}")
        interaction_order = (
            tuple(owned_interaction_specs)
            if not self.interaction_order
            else tuple(self.interaction_order)
        )
        if len(set(interaction_order)) != len(interaction_order) or set(interaction_order) != set(
            owned_interaction_specs
        ):
            raise ValueError(
                "interaction_order must name every configured interaction exactly once"
            )

        if self.link is not None and not isinstance(self.link, str | Link):
            raise TypeError("link must be a link name, Link, or None")
        if not isinstance(self.intercept, bool):
            raise TypeError("intercept must be bool")

        object.__setattr__(self, "features", MappingProxyType(owned_features))
        object.__setattr__(self, "interactions", tuple(interactions))
        object.__setattr__(
            self,
            "interaction_specs",
            MappingProxyType(owned_interaction_specs),
        )
        object.__setattr__(self, "interaction_order", interaction_order)


@dataclass(frozen=True)
class PredictorExecutionPlan:
    """Grouped coefficient-space products for one predictor design.

    The intercept is implicit: no observation-by-coefficient augmented matrix
    is constructed. Slope products remain delegated to the design's cached
    matrix execution plan and rectangular cross-plan.
    """

    design: DesignMatrix
    intercept: bool

    def __post_init__(self) -> None:
        if not isinstance(self.design, DesignMatrix):
            raise TypeError("design must be a DesignMatrix")
        if not isinstance(self.intercept, bool):
            raise TypeError("intercept must be bool")

    @property
    def width(self) -> int:
        return self.design.p + int(self.intercept)

    def _validated_vector(self, values: NDArray, *, label: str) -> NDArray[np.float64]:
        try:
            vector = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} must be a finite vector matching the design rows") from exc
        if vector.shape != (self.design.n,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} must be a finite vector matching the design rows")
        return vector

    def score(self, values: NDArray) -> NDArray[np.float64]:
        """Return the implicit-intercept transpose product for one score channel."""
        vector = self._validated_vector(values, label="score channel")
        result = np.empty(self.width, dtype=np.float64)
        slope_start = int(self.intercept)
        if self.intercept:
            result[0] = np.sum(vector, dtype=np.float64)
        if self.design.p:
            result[slope_start:] = self.design.rmatvec(vector)
        return result

    def diagonal_moment(
        self,
        weights: NDArray,
    ) -> NDArray[np.float64]:
        """Return one symmetric ``X.T @ diag(weights) @ X`` predictor block."""
        moments = self.design.execution_plan.moments(
            weights,
            include_xtw=self.intercept,
            signed=True,
        )
        slope_start = int(self.intercept)
        result = np.zeros((self.width, self.width), dtype=np.float64)
        if self.intercept:
            weight_values = self._validated_vector(weights, label="curvature channel")
            result[0, 0] = np.sum(weight_values, dtype=np.float64)
            assert moments.xtw is not None
            result[0, 1:] = moments.xtw
            result[1:, 0] = moments.xtw
        result[slope_start:, slope_start:] = moments.gram

        # Copy one triangle so symmetry is bit-for-bit exact without a second
        # matrix-product evaluation or a global post-hoc average.
        upper = np.triu(result)
        return upper + np.triu(result, k=1).T

    def cross_moment(
        self,
        right: PredictorExecutionPlan,
        weights: NDArray,
    ) -> NDArray[np.float64]:
        """Return one rectangular cross-predictor weighted moment."""
        if not isinstance(right, PredictorExecutionPlan):
            raise TypeError("right must be a PredictorExecutionPlan")
        cross_plan = CrossMatrixExecutionPlan(
            self.design.execution_plan,
            right.design.execution_plan,
        )
        slopes = cross_plan.cross_moment(weights, signed=True)
        weight_values = np.asarray(weights, dtype=np.float64)
        result = np.zeros((self.width, right.width), dtype=np.float64)
        left_start = int(self.intercept)
        right_start = int(right.intercept)
        result[left_start:, right_start:] = slopes
        if self.intercept:
            if right.design.p:
                result[0, right_start:] = right.design.rmatvec(weight_values)
            if right.intercept:
                result[0, 0] = np.sum(weight_values, dtype=np.float64)
        if right.intercept and self.design.p:
            result[left_start:, 0] = self.design.rmatvec(weight_values)
        return result


@dataclass(frozen=True)
class CompiledPredictor:
    """One locally indexed compiled predictor before global layout embedding."""

    name: str
    parameter_index: int
    link: Link
    compiled: CompiledPredictorDesign
    intercept: bool
    offset: NDArray[np.float64]
    penalties: tuple[PenaltyComponent, ...]


@dataclass(frozen=True)
class _ParameterLinkDefaults:
    default_link: str | Link


def _centering_target(spec: Any) -> Any:
    """Return the spec whose ``transform`` evaluates the basis being centered.

    The centering constant is subtracted at predict time inside the spline
    runtime, which reads it off the spec it is evaluating.  For a numeric-axis
    ``Spline`` that is the feature spec itself; for a hosted basis
    (``OrderedCategorical(basis=Spline(...))``) the feature spec is the wrapper
    and it delegates every numeric path to its inner spline, so the constant
    has to land there or the predict-time half of the hook silently never
    fires.
    """
    if not hasattr(type(spec), "_basis_spline"):
        return spec
    if list(getattr(spec, "_active_special_cols", ())):
        # An ACTIVE special's rows are structural zeros in the smooth block.
        # Centering would shift them off zero at fit time and leave them at
        # zero at predict time, because the wrapper zero-fills them after the
        # inner basis has already subtracted the constant. Refuse rather than
        # ship the mismatch. Such a term also compiles to two groups, which the
        # caller refuses separately -- but on a message that says nothing about
        # specials=, so the refusal is spelled out here.
        #
        # The test is the ACTIVE list, not ``has_specials``: a declared special
        # with no effective training rows is pinned, emits no indicator column,
        # and -- because ``fit`` requires strictly positive weights -- can only
        # be pinned by being absent from the training column altogether. There
        # are then no special rows to hold a structural zero, the block is the
        # one an identical spec without ``specials=`` would build, and the
        # constant lands on the inner spline exactly as it does there. Refusing
        # that shape would fail a model on the fold where a rare level happens
        # not to appear.
        raise ValueError(
            "a centered select smooth cannot declare an active specials= level; "
            "drop specials= or select=False on the inner basis"
        )
    return spec._basis_spline


def _center_selected_smooths(
    compiled: CompiledPredictorDesign,
    geometry_weight: NDArray,
    *,
    intercept: bool,
) -> CompiledPredictorDesign:
    """Center selected smooths while preserving dense or compressed storage."""
    if not intercept:
        return compiled
    matrices = list(compiled.design.group_matrices)
    changed = False
    for feature_name in compiled.feature_order:
        spec = compiled.specs[feature_name]
        if not bool(getattr(spec, "select", False)):
            continue
        target = _centering_target(spec)
        group_indices = [
            index
            for index, group in enumerate(compiled.groups)
            if group.feature_name == feature_name
        ]
        if len(group_indices) != 1:
            raise ValueError("a centered select smooth must compile to exactly one group")
        group_index = group_indices[0]
        matrix = matrices[group_index]
        centered: SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix
        source: SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix
        if isinstance(matrix, SparseSSPGroupMatrix):
            source = matrix
            basis = matrix.B.toarray()
            basis_mean = np.average(basis, axis=0, weights=geometry_weight)
            centered = SparseSSPGroupMatrix(sp.csr_matrix(basis - basis_mean), matrix.R_inv)
        elif type(matrix) in (DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix):
            # Both storages are one row per distinct design row plus an index;
            # they differ only in whether that row set was binned. Rebuild
            # through ``type(matrix)`` so a lossless support block is not
            # silently downcast to the lossy binned parent.
            compressed = cast(DiscretizedSSPGroupMatrix, matrix)
            source = compressed
            support_weight = np.bincount(
                compressed.bin_idx,
                weights=geometry_weight,
                minlength=compressed.n_bins,
            )
            basis_mean = (support_weight @ compressed.B_unique) / np.sum(
                support_weight,
                dtype=np.float64,
            )
            centered = type(compressed)(
                compressed.B_unique - basis_mean,
                compressed.R_inv,
                compressed.bin_idx,
            )
        else:
            raise TypeError("distributional select centering requires an SSP spline group")
        centered.omega = source.omega
        centered.projection = source.projection
        centered.omega_components = source.omega_components
        centered.component_types = source.component_types
        centered.lambda_policies = source.lambda_policies
        matrices[group_index] = centered
        setattr(target, "_distributional_basis_mean", np.asarray(basis_mean, dtype=np.float64))
        changed = True
    if not changed:
        return compiled
    return replace(
        compiled,
        design=DesignMatrix(matrices, compiled.design.n, compiled.design.p),
    )


def _validate_predictor_order(
    parameters: Sequence[ParameterSpec], predictors: Sequence[Predictor]
) -> None:
    expected = tuple(parameter.name for parameter in parameters)
    actual = tuple(predictor.name for predictor in predictors)
    duplicates = sorted({name for name in actual if actual.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate predictor name: {', '.join(duplicates)}")

    unknown = tuple(name for name in actual if name not in expected)
    if unknown:
        raise ValueError(f"unknown predictor name: {', '.join(unknown)}")
    missing = tuple(name for name in expected if name not in actual)
    if missing:
        raise ValueError(f"missing predictor name: {', '.join(missing)}")
    if actual != expected:
        raise ValueError(f"predictor order must match family order {expected}; got {actual}")


def _validate_resolved_weights(
    resolved: ResolvedLikelihoodWeights,
    n_observations: int,
) -> ResolvedLikelihoodWeights:
    if not isinstance(resolved, ResolvedLikelihoodWeights):
        raise TypeError("weights must be ResolvedLikelihoodWeights")
    if len(resolved.values) != n_observations:
        raise ValueError(
            "resolved likelihood weights must match the retained predictor rows; "
            f"expected {n_observations}, got {len(resolved.values)}"
        )
    return resolved


def _resolve_offsets(
    offsets: Mapping[str, NDArray] | None,
    predictor_names: tuple[str, ...],
    n_observations: int,
) -> dict[str, NDArray[np.float64]]:
    supplied: Mapping[str, NDArray] = {} if offsets is None else offsets
    if not isinstance(supplied, Mapping):
        raise TypeError("offsets must be a predictor-keyed mapping")
    unknown = tuple(name for name in supplied if name not in predictor_names)
    if unknown:
        raise ValueError(f"unknown offset predictor name: {', '.join(unknown)}")

    resolved: dict[str, NDArray[np.float64]] = {}
    for name in predictor_names:
        if name not in supplied:
            offset = np.zeros(n_observations, dtype=np.float64)
        else:
            try:
                offset = np.asarray(supplied[name], dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"offset for {name!r} must contain finite numeric values") from exc
            if offset.ndim != 1 or len(offset) != n_observations:
                raise ValueError(
                    f"offset for {name!r} must be one-dimensional with length {n_observations}"
                )
            if not np.all(np.isfinite(offset)):
                raise ValueError(f"offset for {name!r} must contain only finite values")
            offset = np.array(offset, dtype=np.float64, copy=True)
        offset.setflags(write=False)
        resolved[name] = offset
    return resolved


def resolve_predictor_link(parameter: ParameterSpec, predictor: Predictor) -> Link:
    """Resolve one predictor's link against its natural parameter's default."""
    if predictor.link is None and isinstance(parameter.default_link, Link):
        return parameter.default_link
    return resolve_link(
        predictor.link,
        cast(Any, _ParameterLinkDefaults(parameter.default_link)),
    )


def resolve_predictor_links(
    parameters: Sequence[ParameterSpec],
    predictors: Sequence[Predictor],
) -> tuple[Link, ...]:
    """Resolve every predictor's link in ``parameters`` order."""
    return tuple(
        resolve_predictor_link(parameter, predictor)
        for parameter, predictor in zip(tuple(parameters), tuple(predictors), strict=True)
    )


def compile_predictors(
    X: EagerFrame,
    resolved_weights: ResolvedLikelihoodWeights,
    parameters: Sequence[ParameterSpec],
    predictors: Sequence[Predictor],
    *,
    offsets: Mapping[str, NDArray] | None = None,
    model_discrete: bool = False,
    n_bins_config: int | dict[str, int] = 256,
    lambda2: float | dict[str, float] = 0.1,
    separation_boundaries: Sequence[tuple[str, ...]] | None = None,
) -> tuple[CompiledPredictor, ...]:
    """Compile retained predictors under one resolved likelihood-weight root.

    Likelihood and matrix algebra use ``resolved_weights.values``. Learned
    model geometry uses ``resolved_weights.geometry_values``: physical rows
    for prior semantics and literal replication mass for frequency semantics.
    Zero-weight input positions have already been removed at the fit boundary.

    ``separation_boundaries`` carries one response-boundary tuple per
    predictor; a predictor with a non-empty tuple records its categorical
    terms on ``CompiledPredictorDesign.separation_records`` for the fit
    boundary to scan.  ``None`` records nothing.
    """
    parameter_tuple = tuple(parameters)
    predictor_tuple = tuple(predictors)
    if not parameter_tuple:
        raise ValueError("parameters must not be empty")
    if not all(isinstance(parameter, ParameterSpec) for parameter in parameter_tuple):
        raise TypeError("parameters must contain only ParameterSpec values")
    if not all(isinstance(predictor, Predictor) for predictor in predictor_tuple):
        raise TypeError("predictors must contain only Predictor values")
    for predictor in predictor_tuple:
        for feature, kind, mode in _constrained_features(predictor):
            warnings.warn(
                f"SuperLSS ignores shape constraints: {predictor.name}:{feature} asked for "
                f"{kind} ({mode}); the distributional path has no constraint machinery yet, "
                "so the smooth is fitted unconstrained",
                ShapeConstraintIgnoredWarning,
                stacklevel=3,
            )
    _validate_predictor_order(parameter_tuple, predictor_tuple)
    boundary_tuple = (
        tuple(() for _ in predictor_tuple)
        if separation_boundaries is None
        else tuple(tuple(boundary) for boundary in separation_boundaries)
    )
    if len(boundary_tuple) != len(predictor_tuple):
        raise ValueError("separation_boundaries must supply one tuple per predictor")

    n_observations = len(X)
    resolved = _validate_resolved_weights(resolved_weights, n_observations)
    weights = resolved.values
    geometry_weight = resolved.geometry_values
    physical_rows = resolved.provenance.contract.semantics == "prior"
    resolved_offsets = _resolve_offsets(
        offsets,
        tuple(parameter.name for parameter in parameter_tuple),
        n_observations,
    )

    built: list[CompiledPredictor] = []
    for parameter_index, (parameter, predictor) in enumerate(
        zip(parameter_tuple, predictor_tuple, strict=True)
    ):
        link = resolve_predictor_link(parameter, predictor)
        compiled = compile_predictor_design(
            X,
            weights,
            geometry_weight=geometry_weight,
            polynomial_weight=geometry_weight,
            categorical_reporting_weight=geometry_weight,
            ordered_reporting_weight=geometry_weight,
            physical_rows=physical_rows,
            specs=predictor.features,
            feature_order=tuple(predictor.features),
            interaction_specs=predictor.interaction_specs,
            interaction_order=predictor.interaction_order,
            pending_interactions=predictor.interactions,
            model_discrete=model_discrete,
            n_bins_config=n_bins_config,
            lambda2=lambda2,
            separation_boundaries=boundary_tuple[parameter_index],
        )
        compiled = _center_selected_smooths(
            compiled,
            geometry_weight,
            intercept=predictor.intercept,
        )
        local_groups = list(compiled.groups)
        local_matrices = list(compiled.design.group_matrices)
        reml_groups = collect_reml_groups(local_groups, local_matrices)
        penalties = tuple(
            build_penalty_components(
                local_matrices,
                cast(list[tuple[int, object]], reml_groups),
            )
        )
        built.append(
            CompiledPredictor(
                name=predictor.name,
                parameter_index=parameter_index,
                link=link,
                compiled=compiled,
                intercept=predictor.intercept,
                offset=resolved_offsets[predictor.name],
                penalties=penalties,
            )
        )
    return tuple(built)
