"""Internal dense distributional model harness for fixed smoothing parameters."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from itertools import groupby
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._blas_threads import allow_wide_design
from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.family import (
    ChunkPreparedLikelihoodFamily,
    DefaultPredictionFamily,
    DistributionalFamily,
    DistributionFunctionFamily,
    FamilyLikelihoodPlan,
    ObservationContract,
    _likelihood_reuse_contract,
    bounded_observation,
    plan_observation,
)
from superglm.distributional.fit_state import (
    CompactNullModel,
    DistributionalFitState,
    prepare_distributional_fit_state,
)
from superglm.distributional.inference import JointInference
from superglm.distributional.layout import StackedLayout, build_stacked_layout
from superglm.distributional.likelihood_identity import legacy_identifier_problem
from superglm.distributional.observation import (
    BoundPair,
    CensoredRows,
    resolve_response_bounds,
    unvalidated_bound_arrays,
)
from superglm.distributional.offsets import (
    prediction_offsets,
    shape_fit_offsets,
    sum_offset_sources,
)
from superglm.distributional.prediction_design import (
    DESIGN_BLOCK_ROWS,
    ROW_INDEPENDENT_TERMS,
    _as_contribution,
    _score_feature,
    _score_interaction,
    _term_indices,
    group_level_values,
    label_columns,
    warn_population_levels,
)
from superglm.distributional.predictor import (
    CompiledPredictor,
    Predictor,
    compile_predictors,
    resolve_predictor_links,
)
from superglm.distributional.response import (
    response_spec,
    unvalidated_response,
    validated_response_array,
)
from superglm.distributional.results.fit import DistributionalFitResult
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.results.smoothing import DistributionalEFSResult
from superglm.distributional.results.solver import DenseSolverConfig, DenseSolverResult
from superglm.distributional.separation import (
    apply_separation_policy,
    censored_recession,
    format_censored_separation_message,
    predictor_response_boundaries,
    record_factor_smooth_levels,
    scan_censored_separation,
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
    """Legacy scalar-only boundary for scalar diagnostics and row operations."""
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


@dataclass(frozen=True)
class _EtaTask:
    """One term's contribution over ``rows`` of the frame: ``run()`` is ``(values, levels)``."""

    term: str
    rows: slice
    run: Callable[[], tuple[NDArray, list]]


def _score_term(
    term: str,
    n_rows: int,
    score: Callable[[], NDArray],
    levels: list | None = None,
) -> tuple[NDArray, list]:
    values = score()
    return _as_contribution(values, n_observations=n_rows, term_name=term), levels or []


def _level_effects(effects: NDArray, columns: NDArray) -> NDArray:
    """``effects`` at each row's column, the last (zero) entry for a row with none."""
    return effects[columns]


def _eta_tasks(
    frame: EagerFrame,
    predictor: CompiledPredictor,
    layout: StackedLayout,
    coefficients: NDArray[np.float64],
    spans: Sequence[slice] | None = None,
    labels: Mapping[tuple[str, str], NDArray] | None = None,
) -> list[_EtaTask]:
    """The scoring tasks of one predictor, in the order its terms are added to eta.

    ``spans`` splits each term made of numeric covariates alone (``ROW_INDEPENDENT_TERMS``), and
    each label term read from the frame's codes, into row ranges that can be scored on any thread;
    every other term is one task over the frame.
    """
    state = layout.predictors[predictor.parameter_index]
    intercept_width = int(state.intercept_index is not None)
    slopes = coefficients[state.coefficient_slice][intercept_width:]
    whole = slice(0, len(frame))
    tasks: list[_EtaTask] = []
    assigned = np.zeros(len(slopes), dtype=np.bool_)

    def ranges(spec: Any) -> Sequence[slice]:
        return spans if spans and type(spec) in ROW_INDEPENDENT_TERMS else (whole,)

    for name in predictor.compiled.feature_order:
        indices = _term_indices(predictor.compiled.groups, name)
        spec = predictor.compiled.specs[name]
        column, beta = frame.column_array(name), slopes[indices]
        coded = None if labels is None else labels.get((predictor.name, name))
        if coded is not None:
            # a label term's rows are its level effects, read from the frame's codes, which
            # checked the levels and warned once for the frame when they were made
            effects = np.append(beta, 0.0)
            for rows in spans or (whole,):
                tasks.append(
                    _EtaTask(
                        name,
                        rows,
                        partial(
                            _score_term,
                            name,
                            rows.stop - rows.start,
                            partial(_level_effects, effects, coded[rows]),
                        ),
                    )
                )
            assigned[indices] = True
            continue
        for rows in ranges(spec):
            values = column[rows]
            tasks.append(
                _EtaTask(
                    name,
                    rows,
                    partial(
                        _score_term,
                        name,
                        len(values),
                        partial(_score_feature, spec, values, beta),
                    ),
                )
            )
        assigned[indices] = True

    for name in predictor.compiled.interaction_order:
        interaction = predictor.compiled.interaction_specs[name]
        indices = _term_indices(predictor.compiled.groups, name)
        left_name, right_name = interaction.parent_names
        group = getattr(interaction, "group", None)
        constants = (
            group_level_values(
                predictor, interaction, slopes[_term_indices(predictor.compiled.groups, group)]
            )
            if group in predictor.compiled.feature_order
            else None
        )
        left, right = frame.column_array(left_name), frame.column_array(right_name)
        for rows in ranges(interaction):
            levels: list = []
            tasks.append(
                _EtaTask(
                    name,
                    rows,
                    partial(
                        _score_term,
                        name,
                        rows.stop - rows.start,
                        partial(
                            _score_interaction,
                            interaction,
                            left[rows],
                            right[rows],
                            slopes[indices],
                            named=levels,
                            level_constants=constants,
                        ),
                        levels,
                    ),
                )
            )
        assigned[indices] = True

    if not np.all(assigned):
        missing = np.flatnonzero(~assigned).tolist()
        raise RuntimeError(f"prediction plan did not assign local coefficient columns {missing}")
    return tasks


def _assemble_eta(
    predictor: CompiledPredictor,
    layout: StackedLayout,
    coefficients: NDArray[np.float64],
    offset: NDArray,
    tasks: Sequence[_EtaTask],
    results: Sequence[tuple[NDArray, list]],
    n_rows: int,
    named: list[str] | None,
) -> NDArray[np.float64]:
    state = layout.predictors[predictor.parameter_index]
    local_coefficients = coefficients[state.coefficient_slice]
    intercept_width = int(state.intercept_index is not None)
    eta = np.zeros(n_rows, dtype=np.float64)
    for task, (contribution, levels) in zip(tasks, results, strict=True):
        if levels and named is not None:
            named.append(
                f"{predictor.name} term {task.term!r} levels {', '.join(map(str, levels))}"
            )
        eta[task.rows] += contribution
    # Match geometry, value screens and terminal materialization: accumulate
    # the ordered slopes first, then add the intercept and finally the offset.
    if intercept_width:
        if len(local_coefficients) > 1:
            eta += local_coefficients[0]
        else:
            eta[:] = local_coefficients[0]
    eta += offset
    return _readonly(eta)


def _predict_one_eta(
    frame: EagerFrame,
    predictor: CompiledPredictor,
    layout: StackedLayout,
    coefficients: NDArray[np.float64],
    offset: NDArray,
    named: list[str] | None = None,
) -> NDArray[np.float64]:
    tasks = _eta_tasks(frame, predictor, layout, coefficients)
    results = [task.run() for task in tasks]
    return _assemble_eta(predictor, layout, coefficients, offset, tasks, results, len(frame), named)


#: Frames of at least this many rows are scored on the worker pool; a smaller one, which is every
#: quote, runs inline and starts no thread.
POOLED_PREDICTION_ROWS = 2 * DESIGN_BLOCK_ROWS


def _predict_eta_pooled(
    frame: EagerFrame,
    predictors: Sequence[CompiledPredictor],
    layout: StackedLayout,
    coefficients: NDArray[np.float64],
    offsets: Mapping[str, NDArray],
    named: list[str],
) -> NDArray[np.float64]:
    """Every predictor's eta, a block of rows at a time on ``superglm``'s pool.

    A term that is one task over the frame (a label term that checks its levels, an ``sz`` term)
    is scored first, so it checks and warns once for the frame. Each worker then forms one block
    of one predictor's eta, adding its terms in the order ``_predict_one_eta`` adds them, so the
    result is the same at any ``n_jobs``, and holds one term's block at a time: the only
    contributions kept for the whole frame are those of the terms that are one task over it.
    """
    from superglm._parallel import pooled_for

    n = len(frame)
    spans = [slice(lo, min(lo + DESIGN_BLOCK_ROWS, n)) for lo in range(0, n, DESIGN_BLOCK_ROWS)]
    labels = label_columns(frame, predictors)
    grouped = [
        [
            list(tasks)
            for _, tasks in groupby(
                _eta_tasks(frame, p, layout, coefficients, spans, labels), key=lambda t: t.term
            )
        ]
        for p in predictors
    ]
    keys = [
        (i, k)
        for i, terms in enumerate(grouped)
        for k, tasks in enumerate(terms)
        if len(tasks) == 1
    ]
    scored = (
        dict(
            zip(
                keys,
                # Each of these reads its term over the whole frame, in one task.
                pooled_for(
                    lambda key: grouped[key[0]][key[1]][0].run(),
                    keys,
                    max(n, DESIGN_BLOCK_ROWS) * 8 * 16,
                ),
                strict=True,
            )
        )
        if keys
        else {}
    )
    for (i, k), (_, levels) in scored.items():
        if levels:
            named.append(
                f"{predictors[i].name} term {grouped[i][k][0].term!r} levels "
                f"{', '.join(map(str, levels))}"
            )
    eta = np.empty((n, len(predictors)), dtype=np.float64)

    def form(job: tuple[int, int]) -> None:
        i, block = job
        rows = spans[block]
        state = layout.predictors[predictors[i].parameter_index]
        local = coefficients[state.coefficient_slice]
        out = np.zeros(rows.stop - rows.start, dtype=np.float64)
        for k, tasks in enumerate(grouped[i]):
            out += scored[(i, k)][0][rows] if len(tasks) == 1 else tasks[block].run()[0]
        # as ``_assemble_eta``: the ordered slopes, then the intercept, then the offset
        if state.intercept_index is not None:
            if len(local) > 1:
                out += local[0]
            else:
                out[:] = local[0]
        out += offsets[predictors[i].name][rows]
        eta[rows, i] = out

    # A job here holds one block of one term at a time, whatever the frame.
    pooled_for(
        form,
        [(i, b) for i in range(len(predictors)) for b in range(len(spans))],
        DESIGN_BLOCK_ROWS * 8 * 16,
    )
    return eta


@dataclass(frozen=True)
class DenseDistributionalModel:
    """Internal model whose complete fitted revision is one atomic reference."""

    family: DistributionalFamily
    _fit_state: DistributionalFitState
    # Whether the family supplied expected information at fit time, from a loaded
    # manifest (None if fitted here); schema-8 curvature scope depended on it.
    recorded_expected_information: bool | None = field(default=None, compare=False, repr=False)

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
        """The coefficients the fit reports: the hand edit's, else the fit result's."""
        return self._fit_state.reported_coefficients

    @property
    def prediction_coefficients(self) -> NDArray[np.float64]:
        """What predictions and posterior draws use: the hand edit's, else ``result``'s."""
        edit = self._fit_state.hand_edit
        return self.result.coefficients if edit is None else edit.coefficients

    @property
    def covariance(self) -> NDArray[np.float64]:
        return self.fitted_result.covariance

    @property
    def predictor_coefficients(self) -> Mapping[str, NDArray[np.float64]]:
        state = self._fit_state
        if state.hand_edit is None:
            return self.fitted_result.predictor_coefficients
        return MappingProxyType(
            {
                predictor.name: state.hand_edit.coefficients[predictor.coefficient_slice]
                for predictor in state.layout.predictors
            }
        )

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
        return self.predict_eta_at(self.prediction_coefficients, X, offsets=offsets)

    def predict_eta_at(
        self,
        coefficients: NDArray[np.float64],
        X: FrameLike | EagerFrame,
        *,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate every predictor at a full coefficient vector in the fit's layout."""
        frame = as_eager_frame(X)
        resolved_offsets = prediction_offsets(self.fit_state.predictor_templates, frame, offsets)
        return self._predict_eta_resolved(coefficients, frame, resolved_offsets)

    def _predict_parameters_resolved(self, X, offsets) -> NDArray[np.float64]:
        """Internal prediction with a complete, already-resolved link-scale sum."""
        return self.parameters_from_eta(
            self._predict_eta_resolved(self.prediction_coefficients, X, offsets)
        )

    def _predict_eta_resolved(self, coefficients, X, offsets) -> NDArray[np.float64]:
        """Internal path for retained rows and scoring; never resolve declarations twice."""
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
        named: list[str] = []
        if len(frame) >= POOLED_PREDICTION_ROWS:
            eta = _predict_eta_pooled(
                frame, self.compiled_predictors, self.layout, coefficients, resolved_offsets, named
            )
        else:
            eta = np.column_stack(
                tuple(
                    _predict_one_eta(
                        frame,
                        predictor,
                        self.layout,
                        coefficients,
                        resolved_offsets[predictor.name],
                        named,
                    )
                    for predictor in self.compiled_predictors
                )
            )
        warn_population_levels(named)
        return _readonly(eta)

    def predict_parameters(
        self,
        X: FrameLike | EagerFrame,
        *,
        offsets: Mapping[str, NDArray] | None = None,
    ) -> NDArray[np.float64]:
        """Return fitted natural parameters in family order."""
        return self.parameters_from_eta(self.predict_eta(X, offsets=offsets))

    def parameters_from_eta(self, eta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply each parameter's inverse link to its linear predictor column."""
        theta = np.empty_like(eta)
        for parameter_index, state in enumerate(self.layout.predictors):
            # Overflow is refused by the support check immediately below.
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.asarray(state.link.inverse(eta[:, parameter_index]), dtype=np.float64)
            if values.shape != (len(eta),):
                raise ValueError(f"inverse link for {state.name!r} returned an invalid shape")
            if not np.all(self.family.parameters[parameter_index].support.contains(values)):
                raise ValueError(
                    f"prediction parameter {state.name!r} is outside its finite support"
                )
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
                offset=getattr(predictor, "offset", None),
            )
        )
    if not templates:
        raise ValueError("predictors must not be empty")
    return tuple(templates)


def _fixed_fit_lambdas(
    layout: StackedLayout,
    supplied: Mapping[str, float] | None,
    selected_lines: tuple[str, ...] = (),
) -> dict[str, float]:
    """The fit's lambda for every penalty component, in layout order.

    Components of ``select=True`` ``sz`` terms (prefixes in ``selected_lines``)
    refuse zero: an unpenalized level with no rows would make the system singular.
    """
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
    unpenalized = sorted(
        name for name, value in resolved.items() if value == 0.0 and name.startswith(selected_lines)
    )
    if unpenalized:
        raise ValueError(
            f"select=True penalizes every level's line beside its wiggle, but lambdas fixes "
            f"{unpenalized!r} at zero, which leaves part of every level unpenalized. Use "
            "select=False for unpenalized lines, or pass a positive lambda."
        )
    if missing:
        raise ValueError(f"missing penalty lambda names: {sorted(missing)}")
    return resolved


def _selected_line_prefixes(compiled: Sequence[CompiledPredictor]) -> tuple[str, ...]:
    """Penalty-name prefixes of every ``select=True`` ``sz`` term, ``"<parameter>:<term>#"``."""
    from superglm.features.factor_smooth import FactorSmooth

    return tuple(
        f"{predictor.name}:{name}#"
        for predictor in compiled
        for name, spec in predictor.compiled.interaction_specs.items()
        if isinstance(spec, FactorSmooth) and spec._selects_lines
    )


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
    truncation: BoundPair = None,
    censored: CensoredRows = None,
    _input_positions: list[NDArray] | None = None,
) -> DistributionalFitState:
    if not isinstance(weight_contract, WeightContract):
        raise TypeError("weight_contract must be a WeightContract")
    separation_policy = validate_separation_policy(separation)
    original_n = len(frame)
    original_y = unvalidated_response(
        y, family=family, n_rows=original_n, input_index=getattr(frame.native, "index", None)
    )
    original_offsets = _unvalidated_offset_shapes(offsets, original_n)
    declared_offsets = shape_fit_offsets(predictor_templates, frame) if offsets is None else None
    if response_spec(family).shape and (truncation is not None or censored is not None):
        raise UnsupportedLikelihoodContractError(
            "scalar truncation and censoring bounds do not apply to vector responses"
        )
    original_bounds = unvalidated_bound_arrays(truncation, censored, original_n)
    resolved_weights = resolve_likelihood_weights(
        sample_weight,
        n_observations=original_n,
        contract=weight_contract,
    )
    positions = resolved_weights.input_positions
    if _input_positions is not None:
        _input_positions.append(positions)
    retained_frame = as_eager_frame(frame.take_rows(positions))
    retained_y = np.array(original_y[positions], copy=True)
    retained_offsets = (
        sum_offset_sources(declared_offsets, positions)
        if declared_offsets is not None
        else _take_unvalidated_offsets(original_offsets, positions)
    )
    observation = bounded_observation(
        resolve_response_bounds(
            retained_y,
            None if original_bounds is None else tuple(a[positions] for a in original_bounds),
            positions,
        )
    )
    if not observation.is_complete and not getattr(
        getattr(family, "capabilities", None), "censored_response", False
    ):
        raise UnsupportedLikelihoodContractError(
            f"{type(family).__name__} does not fit truncated or censored rows. "
            "All built-in scalar families support these observations; use a compatible family or drop "
            "truncation= and censored= to read every row as complete"
        )
    # Inherited capability must not bypass a custom family's ordinary binder.
    # Bounded rows are always bound eagerly.
    preparation_contract = _likelihood_reuse_contract(family)
    bind = (
        family.bind_chunked_likelihood
        if chunk_size is not None
        and observation.is_complete
        and preparation_contract is not None
        and preparation_contract.derived_prepared_fields
        and isinstance(family, ChunkPreparedLikelihoodFamily)
        else family.bind_likelihood
    )
    likelihood_plan = bind(retained_y, resolved_weights, observation)
    # The family receives retained raw values first; generic float conversion
    # must not hide nullable integers, bools or invalid composition inputs.
    retained_y = validated_response_array(retained_y, family=family, n_rows=len(positions))
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
    if not observation.is_complete and not isinstance(
        getattr(likelihood_plan, "observation", None), ObservationContract
    ):
        raise UnsupportedLikelihoodContractError(
            f"{type(family).__name__} bound truncated or censored rows into a plan with no "
            "observation contract; a bounds-aware plan exposes it as plan.observation, which "
            "the diagnostics, the corrected covariance and the curvature choice read"
        )
    certificate_problem = legacy_identifier_problem(likelihood_plan)
    if certificate_problem is not None:
        raise UnsupportedLikelihoodContractError(certificate_problem)
    with measure_phase(phase_recorder, "predictor_compilation"):
        templates = _clone_predictor_templates(predictor_templates)
        # The scalar scanner reads each Categorical's level universe, so scan the
        # compiled terms before fitting; compile_predictors reports count mismatches.
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
            separation_response=retained_y,
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
    bound_rows = plan_observation(likelihood_plan).bounds
    if bound_rows is not None and separation_policy != "ignore":
        apply_separation_policy(
            scan_censored_separation(
                compiled,
                retained_frame,
                bound_rows,
                resolved_weights.values,
                recession=censored_recession(family, bound_rows, compiled),
            ),
            separation_policy,
            stacklevel=4,
            formatter=format_censored_separation_message,
        )
    if len(family.parameters) == len(templates):
        # Recorded under every policy, as SuperGLM records them: "ignore"
        # silences the warning, never the prediction rule.
        record_factor_smooth_levels(
            compiled,
            retained_y,
            resolved_weights.values,
            boundaries=(
                separation_boundaries
                if separation_boundaries is not None
                else predictor_response_boundaries(
                    family, resolve_predictor_links(family.parameters, templates)
                )
            ),
            policy=separation_policy,
            stacklevel=4,
        )
    with measure_phase(phase_recorder, "layout_penalty_assembly"):
        layout = build_stacked_layout(compiled)
    allow_wide_design(layout.n_coefficients)
    if efs_config is None:
        with measure_phase(phase_recorder, "layout_penalty_assembly"):
            ordered_lambdas = _fixed_fit_lambdas(layout, lambdas, _selected_line_prefixes(compiled))
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
    truncation: BoundPair = None,
    censored: CensoredRows = None,
    _input_positions: list[NDArray] | None = None,
) -> DenseDistributionalModel:
    """Compile every terminal artifact privately, then publish one fitted revision.

    ``revision`` numbers the published fitted state; a facade that refits in
    place passes its next revision so successive fits stay distinguishable.
    ``_input_positions``, when given, receives the input positions of the
    retained rows.
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
            truncation=truncation,
            censored=censored,
            _input_positions=_input_positions,
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
    truncation: BoundPair = None,
    censored: CensoredRows = None,
) -> DenseDistributionalModel:
    """Replace a fitted revision only after its complete candidate validates.

    Bounds describe the rows being fitted, so a refit takes the caller's
    ``truncation`` and ``censored`` as ``fit`` does; none are carried over.
    """
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
            truncation=truncation,
            censored=censored,
        )
        model._publish(candidate)
        return model


__all__ = [
    "DenseDistributionalModel",
    "fit_dense_distributional",
    "refit_dense_distributional",
]
