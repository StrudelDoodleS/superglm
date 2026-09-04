"""Internal dense distributional model harness for fixed smoothing parameters."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._blas_threads import allow_row_space_work, allow_wide_design
from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    DefaultPredictionFamily,
    DistributionalFamily,
    DistributionFunctionFamily,
    FamilyLikelihoodPlan,
)
from superglm.distributional.fit_state import (
    CompactNullModel,
    DistributionalFitState,
    prepare_distributional_fit_state,
)
from superglm.distributional.inference import JointInference
from superglm.distributional.layout import StackedLayout, build_stacked_layout
from superglm.distributional.prediction_design import (
    _as_contribution,
    _score_feature,
    _score_interaction,
    _term_indices,
)
from superglm.distributional.predictor import (
    CompiledPredictor,
    Predictor,
    compile_predictors,
    resolve_predictor_links,
)
from superglm.distributional.result import (
    DenseSolverConfig,
    DenseSolverResult,
    DistributionalEFSConfig,
    DistributionalEFSResult,
    DistributionalFitResult,
)
from superglm.distributional.separation import (
    apply_separation_policy,
    predictor_response_boundaries,
    scan_predictor_separation,
    validate_separation_policy,
)
from superglm.distributional.smoothing.loop import fit_distributional_efs
from superglm.distributional.solver.chunks import ChunkSize
from superglm.distributional.solver.solver import fit_dense_fixed_lambda
from superglm.distributional.timing import FitPhaseRecorder, measure_phase
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)


def _readonly(values: NDArray) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("prediction produced non-finite values")
    result.setflags(write=False)
    return result


def _readonly_default_prediction(values: NDArray) -> NDArray[np.float64]:
    """Freeze a default prediction while preserving a legitimate infinite mean."""
    result = np.array(values, dtype=np.float64, copy=True)
    if np.any(np.isnan(result)) or np.any(np.isneginf(result)):
        raise ValueError("prediction produced NaN or negative infinity")
    result.setflags(write=False)
    return result


def _unvalidated_response_shape(y: NDArray, n_observations: int) -> NDArray:
    """Check only the original positional shape before zero-row selection."""
    try:
        response = np.asarray(y)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("y must be one-dimensional with one value per input row") from exc
    if response.shape != (n_observations,):
        raise ValueError("y must be one-dimensional with one value per input row")
    return response


def _unvalidated_offset_shapes(
    offsets: Mapping[str, NDArray] | None,
    n_observations: int,
) -> Mapping[str, NDArray] | None:
    """Check only original offset row shapes, deferring values and names."""
    if offsets is None:
        return None
    if not isinstance(offsets, Mapping):
        raise TypeError("offsets must be a predictor-keyed mapping")
    shaped: dict[str, NDArray] = {}
    for name, values in offsets.items():
        try:
            offset = np.asarray(values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"offset for {name!r} must be one-dimensional with length {n_observations}"
            ) from exc
        if offset.shape != (n_observations,):
            raise ValueError(
                f"offset for {name!r} must be one-dimensional with length {n_observations}"
            )
        shaped[name] = offset
    return shaped


def _take_unvalidated_offsets(
    offsets: Mapping[str, NDArray] | None,
    positions: NDArray[np.integer],
) -> dict[str, NDArray] | None:
    if offsets is None:
        return None
    return {name: np.array(values[positions], copy=True) for name, values in offsets.items()}


def _prediction_offsets(
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
    result: dict[str, NDArray[np.float64]] = {}
    for name in predictor_names:
        if name not in supplied:
            result[name] = _readonly(np.zeros(n_observations, dtype=np.float64))
            continue
        try:
            values = np.asarray(supplied[name], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"offset for {name!r} must contain finite numeric values") from exc
        if values.ndim != 1 or len(values) != n_observations:
            raise ValueError(
                f"offset for {name!r} must be one-dimensional with length {n_observations}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"offset for {name!r} must contain only finite values")
        result[name] = _readonly(values)
    return result


def _predict_one_eta(
    frame: EagerFrame,
    predictor: CompiledPredictor,
    layout: StackedLayout,
    result: DenseSolverResult,
    offset: NDArray,
) -> NDArray[np.float64]:
    state = layout.predictors[predictor.parameter_index]
    local_coefficients = result.coefficients[state.coefficient_slice]
    intercept_width = int(state.intercept_index is not None)
    slopes = local_coefficients[intercept_width:]
    eta = np.full(
        len(frame),
        local_coefficients[0] if intercept_width else 0.0,
        dtype=np.float64,
    )
    assigned = np.zeros(len(slopes), dtype=np.bool_)

    for name in predictor.compiled.feature_order:
        indices = _term_indices(predictor.compiled.groups, name)
        contribution = _score_feature(
            predictor.compiled.specs[name],
            frame.column_array(name),
            slopes[indices],
        )
        eta += _as_contribution(contribution, n_observations=len(frame), term_name=name)
        assigned[indices] = True

    for name in predictor.compiled.interaction_order:
        interaction = predictor.compiled.interaction_specs[name]
        indices = _term_indices(predictor.compiled.groups, name)
        left_name, right_name = interaction.parent_names
        contribution = _score_interaction(
            interaction,
            frame.column_array(left_name),
            frame.column_array(right_name),
            slopes[indices],
        )
        eta += _as_contribution(contribution, n_observations=len(frame), term_name=name)
        assigned[indices] = True

    if not np.all(assigned):
        missing = np.flatnonzero(~assigned).tolist()
        raise RuntimeError(f"prediction plan did not assign local coefficient columns {missing}")
    eta += offset
    return _readonly(eta)


@dataclass(frozen=True)
class DenseDistributionalModel:
    """Internal model whose complete fitted revision is one atomic reference."""

    family: DistributionalFamily
    _fit_state: DistributionalFitState

    def __post_init__(self) -> None:
        if not isinstance(self.family, DistributionalFamily):
            raise TypeError("family must implement DistributionalFamily")
        self._validate_candidate(self._fit_state)

    def _validate_candidate(self, candidate: DistributionalFitState) -> None:
        if not isinstance(candidate, DistributionalFitState):
            raise TypeError("fit state must be DistributionalFitState")
        family_names = tuple(parameter.name for parameter in self.family.parameters)
        state_names = tuple(state.name for state in candidate.layout.predictors)
        if state_names != family_names:
            raise ValueError("fit-state predictors must follow family parameter order")

    def _publish(self, candidate: DistributionalFitState) -> None:
        """Install a prevalidated revision with one allocation-free reference swap."""
        self._validate_candidate(candidate)
        object.__setattr__(self, "_fit_state", candidate)

    @property
    def fit_state(self) -> DistributionalFitState:
        return self._fit_state

    @property
    def compiled_predictors(self) -> tuple[CompiledPredictor, ...]:
        return self._fit_state.compiled_predictors

    @property
    def layout(self) -> StackedLayout:
        return self._fit_state.layout

    @property
    def lambdas(self) -> Mapping[str, float]:
        return self._fit_state.lambdas

    @property
    def result(self) -> DenseSolverResult:
        return self._fit_state.solver_result

    @property
    def smoothing(self) -> DistributionalEFSResult | None:
        return self._fit_state.smoothing

    @property
    def fitted_result(self) -> DistributionalFitResult:
        return self._fit_state.result

    @property
    def inference(self) -> JointInference:
        return self._fit_state.inference

    @property
    def null_model(self) -> CompactNullModel:
        return self._fit_state.null_model

    @property
    def coefficients(self) -> NDArray[np.float64]:
        return self.fitted_result.coefficients

    @property
    def covariance(self) -> NDArray[np.float64]:
        return self.fitted_result.covariance

    @property
    def predictor_coefficients(self) -> Mapping[str, NDArray[np.float64]]:
        return self.fitted_result.predictor_coefficients

    @property
    def smoothing_parameters(self) -> Mapping[str, float]:
        return self.fitted_result.smoothing_parameters

    @property
    def telemetry(self):
        return self.fitted_result.curvature_telemetry

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.fitted_result.parameter_names

    def predict_eta(
        self,
        X: FrameLike | EagerFrame,
        *,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate every fitted predictor in family parameter order."""
        frame = as_eager_frame(X)
        predictor_names = tuple(predictor.name for predictor in self.compiled_predictors)
        resolved_offsets = _prediction_offsets(offsets, predictor_names, len(frame))
        required_columns = tuple(
            dict.fromkeys(
                name
                for predictor in self.compiled_predictors
                for name in (
                    *predictor.compiled.feature_order,
                    *(
                        parent
                        for interaction_name in predictor.compiled.interaction_order
                        for parent in predictor.compiled.interaction_specs[
                            interaction_name
                        ].parent_names
                    ),
                )
            )
        )
        frame.require_columns(required_columns)
        eta = np.column_stack(
            tuple(
                _predict_one_eta(
                    frame,
                    predictor,
                    self.layout,
                    self.result,
                    resolved_offsets[predictor.name],
                )
                for predictor in self.compiled_predictors
            )
        )
        return _readonly(eta)

    def predict_parameters(
        self,
        X: FrameLike | EagerFrame,
        *,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> NDArray[np.float64]:
        """Return fitted natural parameters in family order."""
        eta = self.predict_eta(X, offsets=offsets)
        theta = np.empty_like(eta)
        for parameter_index, state in enumerate(self.layout.predictors):
            values = np.asarray(state.link.inverse(eta[:, parameter_index]), dtype=np.float64)
            if values.shape != (len(eta),):
                raise ValueError(f"inverse link for {state.name!r} returned an invalid shape")
            theta[:, parameter_index] = values
        return _readonly(theta)

    def predict(
        self,
        X: FrameLike | EagerFrame,
        *,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> NDArray[np.float64]:
        """Return the family-defined default prediction quantity."""
        if not isinstance(self.family, DefaultPredictionFamily):
            raise NotImplementedError(
                "this distributional family has no default prediction; use predict_parameters()"
            )
        values = self.family.default_prediction(self.predict_parameters(X, offsets=offsets))
        return _readonly_default_prediction(np.asarray(values, dtype=np.float64))

    def _distribution_function_family(self) -> DistributionFunctionFamily:
        if not isinstance(self.family, DistributionFunctionFamily):
            raise NotImplementedError(
                "this distributional family has no distribution function; use predict_parameters()"
            )
        return self.family

    def predict_cdf(
        self,
        X: FrameLike | EagerFrame,
        y: NDArray | float,
        *,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> NDArray[np.float64]:
        """Return ``P(Y <= y)`` per row from the fitted natural parameters."""
        family = self._distribution_function_family()
        theta = self.predict_parameters(X, offsets=offsets)
        values = np.broadcast_to(np.asarray(y, dtype=np.float64), (len(theta),))
        return _readonly(np.asarray(family.cdf(values, theta), dtype=np.float64))

    def predict_quantile(
        self,
        X: FrameLike | EagerFrame,
        p: NDArray | float,
        *,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> NDArray[np.float64]:
        """Return the ``p``-quantile per row from the fitted natural parameters."""
        family = self._distribution_function_family()
        theta = self.predict_parameters(X, offsets=offsets)
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(theta),))
        return _readonly(np.asarray(family.quantile(probabilities, theta), dtype=np.float64))


def _clone_predictor_templates(predictors: Sequence[Predictor]) -> tuple[Predictor, ...]:
    templates: list[Predictor] = []
    for predictor in predictors:
        if not isinstance(predictor, Predictor):
            raise TypeError("predictors must contain only Predictor values")
        templates.append(
            Predictor(
                predictor.name,
                copy.deepcopy(dict(predictor.features)),
                link=copy.deepcopy(predictor.link),
                intercept=predictor.intercept,
                interactions=tuple(predictor.interactions),
                interaction_specs=copy.deepcopy(dict(predictor.interaction_specs)),
                interaction_order=tuple(predictor.interaction_order),
            )
        )
    if not templates:
        raise ValueError("predictors must not be empty")
    return tuple(templates)


def _fixed_fit_lambdas(
    layout: StackedLayout,
    supplied: Mapping[str, float] | None,
) -> dict[str, float]:
    values: Mapping[str, float] = {} if supplied is None else supplied
    if not isinstance(values, Mapping):
        raise TypeError("lambdas must be a qualified penalty-name mapping")
    unknown = set(values) - set(layout.penalty_names)
    if unknown:
        raise ValueError(f"unknown penalty lambda names: {sorted(unknown)}")

    resolved: dict[str, float] = {}
    missing: list[str] = []
    for component in layout.penalties:
        policy = component.lambda_policy
        if policy is not None and policy.mode == "fixed":
            if policy.value is None:
                raise ValueError(f"fixed policy for {component.name!r} has no value")
            value = policy.value
        elif component.name in values:
            value = values[component.name]
        else:
            missing.append(component.name)
            continue
        if isinstance(value, bool):
            raise ValueError(f"lambda for {component.name!r} must be finite and nonnegative")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"lambda for {component.name!r} must be finite and nonnegative")
        resolved[component.name] = numeric
    if missing:
        raise ValueError(f"missing penalty lambda names: {sorted(missing)}")
    return resolved


def _fit_candidate(
    frame: EagerFrame,
    y: NDArray,
    *,
    family: DistributionalFamily,
    predictor_templates: Sequence[Predictor],
    sample_weight: NDArray | None,
    weight_contract: WeightContract,
    offsets: Mapping[str, NDArray] | None,
    lambdas: Mapping[str, float] | None,
    config: DenseSolverConfig | None,
    efs_config: DistributionalEFSConfig | None,
    initial: NDArray | None,
    revision: int,
    retain_rows: bool,
    discrete: bool = False,
    n_bins: int | dict[str, int] = 256,
    chunk_size: ChunkSize | None = None,
    phase_recorder: FitPhaseRecorder | None = None,
    separation: str = "warn",
) -> DistributionalFitState:
    if not isinstance(weight_contract, WeightContract):
        raise TypeError("weight_contract must be a WeightContract")
    separation_policy = validate_separation_policy(separation)
    original_n = len(frame)
    original_y = _unvalidated_response_shape(y, original_n)
    original_offsets = _unvalidated_offset_shapes(offsets, original_n)
    resolved_weights = resolve_likelihood_weights(
        sample_weight,
        n_observations=original_n,
        contract=weight_contract,
    )
    positions = resolved_weights.input_positions
    retained_frame = as_eager_frame(frame.take_rows(positions))
    retained_y = np.array(original_y[positions], copy=True)
    retained_offsets = _take_unvalidated_offsets(original_offsets, positions)
    likelihood_plan = family.bind_likelihood(
        retained_y,
        resolved_weights,
        COMPLETE_OBSERVATION,
    )
    if not isinstance(likelihood_plan, FamilyLikelihoodPlan):
        raise UnsupportedLikelihoodContractError(
            "family.bind_likelihood() must return a FamilyLikelihoodPlan"
        )
    if not isinstance(likelihood_plan.weights, ResolvedLikelihoodWeights):
        raise UnsupportedLikelihoodContractError(
            "family likelihood plans must own resolved likelihood weights"
        )
    if likelihood_plan.weights.root_digest != resolved_weights.root_digest:
        raise UnsupportedLikelihoodContractError(
            "the family likelihood changed the fitted likelihood weights"
        )
    with measure_phase(phase_recorder, "predictor_compilation"):
        templates = _clone_predictor_templates(predictor_templates)
        # The scalar scanner reads each Categorical's built level universe,
        # so the compile records the scannable terms and the scan runs on
        # them before any coefficient is fitted.  A predictor-count mismatch
        # is left for compile_predictors to report.
        separation_boundaries = (
            predictor_response_boundaries(
                family,
                resolve_predictor_links(family.parameters, templates),
            )
            if separation_policy != "ignore" and len(family.parameters) == len(templates)
            else None
        )
        compiled = compile_predictors(
            retained_frame,
            resolved_weights,
            family.parameters,
            templates,
            offsets=retained_offsets,
            model_discrete=discrete,
            n_bins_config=n_bins,
            separation_boundaries=separation_boundaries,
        )
    if separation_boundaries is not None:
        apply_separation_policy(
            scan_predictor_separation(
                compiled,
                retained_y,
                resolved_weights.values,
                boundaries=separation_boundaries,
            ),
            separation_policy,
            stacklevel=4,
        )
    with measure_phase(phase_recorder, "layout_penalty_assembly"):
        layout = build_stacked_layout(compiled)
    allow_wide_design(layout.n_coefficients)
    allow_row_space_work(
        layout.predictors[0].design.n,
        [
            state.coefficient_slice.stop - state.coefficient_slice.start
            for state in layout.predictors
        ],
    )
    if efs_config is None:
        with measure_phase(phase_recorder, "layout_penalty_assembly"):
            ordered_lambdas = _fixed_fit_lambdas(layout, lambdas)
            penalty = layout.penalty_matrix(ordered_lambdas)
        result = fit_dense_fixed_lambda(
            family,
            layout,
            retained_y,
            likelihood_plan,
            penalty,
            initial=initial,
            config=config,
            chunk_size=chunk_size,
            phase_recorder=phase_recorder,
        )
        smoothing = None
    else:
        smoothing = fit_distributional_efs(
            family,
            layout,
            retained_y,
            likelihood_plan,
            lambdas=lambdas,
            solver_config=config,
            efs_config=efs_config,
            initial=initial,
            chunk_size=chunk_size,
            phase_recorder=phase_recorder,
        )
        ordered_lambdas = dict(smoothing.lambdas)
        result = smoothing.terminal_fit
    with measure_phase(phase_recorder, "inference_edf"):
        return prepare_distributional_fit_state(
            family,
            templates,
            compiled,
            layout,
            ordered_lambdas,
            result,
            smoothing,
            retained_y,
            likelihood_plan,
            revision=revision,
            retain_rows=retain_rows,
            requested_discrete=discrete,
            requested_n_bins=n_bins,
            requested_chunk_size=chunk_size,
        )


def fit_dense_distributional(
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    family: DistributionalFamily,
    predictors: Sequence[Predictor],
    weight_contract: WeightContract,
    sample_weight: NDArray | None = None,
    offsets: Mapping[str, NDArray] | None = None,
    lambdas: Mapping[str, float] | None = None,
    config: DenseSolverConfig | None = None,
    efs_config: DistributionalEFSConfig | None = None,
    initial: NDArray | None = None,
    retain_rows: bool = True,
    discrete: bool = False,
    n_bins: int | dict[str, int] = 256,
    chunk_size: ChunkSize | None = None,
    phase_recorder: FitPhaseRecorder | None = None,
    separation: str = "warn",
    revision: int = 1,
) -> DenseDistributionalModel:
    """Compile every terminal artifact privately, then publish one fitted revision.

    ``revision`` numbers the published fitted state; a facade that refits in
    place passes its next revision so successive fits stay distinguishable.
    """
    if not isinstance(family, DistributionalFamily):
        raise TypeError("family must implement DistributionalFamily")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("revision must be a positive integer")
    if not isinstance(retain_rows, bool):
        raise TypeError("retain_rows must be bool")
    if not isinstance(discrete, bool):
        raise TypeError("discrete must be bool")
    with measure_phase(phase_recorder, "fit_total"):
        with measure_phase(phase_recorder, "frame_normalization"):
            frame = as_eager_frame(X)
        candidate = _fit_candidate(
            frame,
            y,
            family=family,
            predictor_templates=tuple(predictors),
            sample_weight=sample_weight,
            weight_contract=weight_contract,
            offsets=offsets,
            lambdas=lambdas,
            config=config,
            efs_config=efs_config,
            initial=initial,
            revision=revision,
            retain_rows=retain_rows,
            discrete=discrete,
            n_bins=n_bins,
            chunk_size=chunk_size,
            phase_recorder=phase_recorder,
            separation=separation,
        )
        return DenseDistributionalModel(family=family, _fit_state=candidate)


def refit_dense_distributional(
    model: DenseDistributionalModel,
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    sample_weight: NDArray | None = None,
    offsets: Mapping[str, NDArray] | None = None,
    lambdas: Mapping[str, float] | None = None,
    config: DenseSolverConfig | None = None,
    efs_config: DistributionalEFSConfig | None = None,
    initial: NDArray | None = None,
    retain_rows: bool | None = None,
    phase_recorder: FitPhaseRecorder | None = None,
    separation: str = "warn",
) -> DenseDistributionalModel:
    """Replace a fitted revision only after its complete candidate validates."""
    if not isinstance(model, DenseDistributionalModel):
        raise TypeError("model must be a DenseDistributionalModel")
    with measure_phase(phase_recorder, "fit_total"):
        with measure_phase(phase_recorder, "frame_normalization"):
            frame = as_eager_frame(X)
        previous = model.fit_state
        resolved_lambdas = model.lambdas if lambdas is None else lambdas
        resolved_config = previous.requested_solver_config if config is None else config
        resolved_efs = efs_config
        if resolved_efs is None and model.smoothing is not None:
            resolved_efs = model.smoothing.config
        resolved_retain_rows = (
            previous.retained_rows is not None if retain_rows is None else retain_rows
        )
        candidate = _fit_candidate(
            frame,
            y,
            family=model.family,
            predictor_templates=previous.predictor_templates,
            # A refit re-states the published model's contract rather than
            # re-defaulting: what the weights mean was settled at the first fit.
            weight_contract=previous.weight_contract,
            sample_weight=sample_weight,
            offsets=offsets,
            lambdas=resolved_lambdas,
            config=resolved_config,
            efs_config=resolved_efs,
            initial=initial,
            revision=previous.revision + 1,
            retain_rows=resolved_retain_rows,
            discrete=previous.requested_discrete,
            n_bins=(
                previous.requested_n_bins
                if isinstance(previous.requested_n_bins, int)
                else dict(previous.requested_n_bins)
            ),
            chunk_size=previous.requested_chunk_size,
            phase_recorder=phase_recorder,
            # The fit state records no separation policy, so a refit takes
            # the caller's argument rather than re-stating a stored one.
            separation=separation,
        )
        model._publish(candidate)
        return model


__all__ = [
    "DenseDistributionalModel",
    "fit_dense_distributional",
    "refit_dense_distributional",
]
