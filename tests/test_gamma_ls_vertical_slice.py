from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, special

from superglm import Spline, SuperLSS
from superglm.distributional import GammaLS, Predictor
from superglm.distributional.assembly import dense_predictor_matrices
from superglm.distributional.efs_acceleration import (
    MultisecantDecision,
    WindowedTypeIIAnderson,
)
from superglm.distributional.family import COMPLETE_OBSERVATION, InitialParameterState
from superglm.distributional.model import (
    DenseDistributionalModel,
    fit_dense_distributional,
    refit_dense_distributional,
)
from superglm.distributional.prediction_design import build_joint_prediction_design
from superglm.distributional.result import (
    DenseSolverConfig,
    DenseSolverResult,
    DistributionalEFSConfig,
)
from superglm.distributional.weights import (
    LikelihoodWeightError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.features import Numeric
from superglm.group_matrix import DiscretizedSSPGroupMatrix
from superglm.types import LambdaPolicy
from tests._gamma_lss_oracles import gamma_row_reference
from tests.test_distributional_efs import _spectral_logdet


class _RejectMarkerNumeric(Numeric):
    """Fail only if a discarded row reaches predictor compilation."""

    def build(self, x, sample_weight=None):
        values = np.asarray(x, dtype=np.float64)
        if np.any(values == 999.0):
            raise AssertionError("discarded row reached predictor geometry")
        return super().build(values, sample_weight)


def _roundoff_factor(operations: int) -> float:
    if isinstance(operations, bool) or not isinstance(operations, int) or operations < 1:
        raise ValueError("operations must be a positive integer")
    epsilon = np.finfo(np.float64).eps
    product = operations * epsilon
    if product >= 1.0:
        raise ValueError("operation count lies outside the float64 gamma regime")
    return float(np.nextafter(product / (1.0 - product), np.inf))


def _outward_error(*terms: float) -> float:
    if any(not math.isfinite(term) or term < 0.0 for term in terms):
        raise ValueError("error terms must be finite and non-negative")
    return float(np.nextafter(math.fsum(terms), np.inf))


def _gamma_sum_bound(absolute_sum: float, *, n_rows: int, dimension: int = 2) -> float:
    # Higham's γₘ with m=n+q covers one n-row accumulation followed by the
    # q-coordinate infinity norm and scalar normalization used by the KKT test.
    operations = n_rows + dimension
    return _outward_error(_roundoff_factor(operations) * absolute_sum)


def _bounded_mean(values: np.ndarray) -> tuple[float, float]:
    total = math.fsum(float(value) for value in values)
    absolute_sum = math.fsum(abs(float(value)) for value in values)
    sum_error = _gamma_sum_bound(absolute_sum, n_rows=len(values), dimension=1)
    mean = total / len(values)
    division_error = np.finfo(np.float64).eps * abs(total) / len(values)
    return mean, sum_error / len(values) + division_error


def _intercept_gamma_mle(response: np.ndarray) -> tuple[float, float, float, float]:
    mean, mean_error = _bounded_mean(response)
    ratio = response / mean
    mean_deviance, mean_deviance_error = _bounded_mean(ratio - 1.0 - np.log(ratio))

    def shape_score(shape: float) -> float:
        return math.log(shape) - float(special.digamma(shape)) - mean_deviance

    eps = np.finfo(np.float64).eps
    shape = optimize.brentq(
        shape_score,
        eps,
        1.0 / eps,
        xtol=eps,
        rtol=8.0 * eps,
    )
    return mean, float(shape), mean_error, mean_deviance_error


def _gamma_intercept_diagnostics(
    response: np.ndarray,
    mean: float,
    scale: float,
) -> tuple[float, np.ndarray, float, float, np.ndarray]:
    shape = 1.0 / scale**2
    ratio = response / mean
    deviance = ratio - 1.0 - np.log(ratio)
    normalizer = shape * math.log(shape) - shape - float(special.gammaln(shape))
    score_rows = np.column_stack(
        (
            shape * (ratio - 1.0),
            2.0 * shape * (float(special.digamma(shape)) - math.log(shape) + deviance),
        )
    )
    objective_rows = normalizer - shape * deviance
    score = np.array(
        [math.fsum(float(value) for value in score_rows[:, column]) for column in range(2)]
    )
    objective = math.fsum(float(value) for value in objective_rows)
    score_error = np.array(
        [
            _gamma_sum_bound(
                math.fsum(abs(float(value)) for value in score_rows[:, column]),
                n_rows=len(response),
            )
            for column in range(2)
        ]
    )
    objective_error = _gamma_sum_bound(
        math.fsum(abs(float(value)) for value in objective_rows),
        n_rows=len(response),
    )
    normalized_kkt = float(np.linalg.norm(score, ord=np.inf) / (1.0 + abs(objective)))
    return objective, score, normalized_kkt, objective_error, score_error


def _normalized_kkt_roundoff(
    diagnostics: tuple[float, np.ndarray, float, float, np.ndarray],
    tolerance: float,
) -> float:
    objective, _, _, objective_error, score_error = diagnostics
    return float((np.max(score_error) + tolerance * objective_error) / (1.0 + abs(objective)))


def test_public_gamma_fixed_fit_recovers_the_independent_intercept_mle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260828)
    n = 512
    true_mean = 2.4
    true_scale = 0.55
    response = rng.gamma(
        shape=1.0 / true_scale**2,
        scale=true_mean * true_scale**2,
        size=n,
    )
    frame = pd.DataFrame({"row": np.linspace(-1.0, 1.0, n)})
    initial_mean = 1.4 * true_mean
    initial_scale = 1.5 * true_scale

    def displaced_initialization(self, y, plan):
        del self, plan
        return InitialParameterState(
            theta=np.tile(np.array([initial_mean, initial_scale]), (len(y), 1))
        )

    monkeypatch.setattr(GammaLS, "initialize", displaced_initialization)
    inner_tolerance = float(np.sqrt(np.finfo(np.float64).eps))
    initial_diagnostics = _gamma_intercept_diagnostics(
        response,
        initial_mean,
        initial_scale,
    )
    assert initial_diagnostics[2] > inner_tolerance + _normalized_kkt_roundoff(
        initial_diagnostics,
        inner_tolerance,
    )
    model = SuperLSS(
        family=GammaLS(),
        predictors=(Predictor("mean", {}), Predictor("scale", {})),
    ).fit(frame, response, inner_tol=inner_tolerance)

    parameters = model.predict_parameters(frame)
    fitted_mean = float(parameters["mean"].iloc[0])
    fitted_scale = float(parameters["scale"].iloc[0])
    mle_mean, mle_shape, mle_mean_error, mle_deviance_error = _intercept_gamma_mle(response)
    fitted_shape = 1.0 / fitted_scale**2
    terminal_diagnostics = _gamma_intercept_diagnostics(
        response,
        fitted_mean,
        fitted_scale,
    )

    state = model._require_fitted()
    result = state.result
    inference = state.inference
    assert (
        abs(result.initial_penalized_optimizing_log_likelihood - initial_diagnostics[0])
        <= initial_diagnostics[3]
    )
    assert model.result_.coefficient_converged is True
    assert terminal_diagnostics[2] <= (
        result.config.tolerance
        + _normalized_kkt_roundoff(terminal_diagnostics, result.config.tolerance)
    )
    assert model.result_.rank == state.layout.n_coefficients == 2
    assert np.all(parameters.to_numpy() > 0.0)

    terminal_score = terminal_diagnostics[1]
    terminal_score_error = terminal_diagnostics[4]
    mean_score_fraction = (abs(float(terminal_score[0])) + float(terminal_score_error[0])) / (
        n * fitted_shape
    )
    assert mean_score_fraction < 1.0
    oracle_mean_fraction = mle_mean_error / mle_mean
    assert oracle_mean_fraction < 1.0
    oracle_mean_log_error = -math.log1p(-oracle_mean_fraction)
    mean_error_bound = -math.log1p(-mean_score_fraction) + oracle_mean_log_error
    assert abs(math.log(fitted_mean / mle_mean)) <= mean_error_bound

    mean_ratio = mle_mean / fitted_mean
    mean_deviance_gap = mean_ratio - 1.0 - math.log(mean_ratio)
    oracle_mean_deviance_error = mle_mean_error / fitted_mean + oracle_mean_log_error
    shape_equation_residual = (
        (abs(float(terminal_score[1])) + float(terminal_score_error[1])) / (2.0 * n * fitted_shape)
        + mean_deviance_gap
        + oracle_mean_deviance_error
        + mle_deviance_error
    )
    epsilon = np.finfo(np.float64).eps
    root_relative_error = 8.0 * epsilon + epsilon / mle_shape
    root_log_error = -math.log1p(-root_relative_error)
    shape_interval_upper = max(
        fitted_shape,
        mle_shape * math.exp(root_log_error),
    )
    # For k>0, k·trigamma(k)-1 ≥ 1/(2k). Therefore the magnitude of the
    # derivative of the shape equation in ρ=log(k) is at least
    # 1/(2·shape_interval_upper) throughout the interval joining both shapes.
    shape_error_bound = 2.0 * shape_interval_upper * shape_equation_residual + root_log_error
    assert abs(math.log(fitted_shape / mle_shape)) <= shape_error_bound
    np.testing.assert_array_equal(model.predict(frame), parameters["mean"].to_numpy())

    covariance = model.covariance_
    symmetry_tolerance = (
        64.0 * np.finfo(np.float64).eps * max(1.0, float(np.linalg.norm(covariance, ord=np.inf)))
    )
    assert np.all(np.isfinite(covariance))
    np.testing.assert_allclose(
        covariance,
        covariance.T,
        rtol=0.0,
        atol=symmetry_tolerance,
    )
    assert inference.rank == model.result_.rank
    assert inference.covariance_curvature_source == result.terminal_curvature.actual_source
    assert inference.edf_curvature_source == result.terminal_curvature.actual_source
    assert inference.slice_reconciliation_error <= inference.reconciliation_tolerance
    assert inference.predictor_reconciliation_error <= inference.reconciliation_tolerance
    edf_roundoff = (
        64.0
        * np.finfo(np.float64).eps
        * state.layout.n_coefficients
        * float(np.linalg.cond(result.terminal_penalized_curvature))
    )
    assert abs(inference.total_edf - state.layout.n_coefficients) <= edf_roundoff
    assert model.result_.total_effective_df == inference.total_edf


_WeightSemantics = Literal["prior", "frequency"]
_EPS = np.finfo(np.float64).eps
_COMPLETE_FIT_TOLERANCE = float(np.sqrt(4.0 * np.finfo(np.float64).eps))
_CROSS_FAMILY_OUTER_TOLERANCE = float(_EPS**0.25)
_OBJECTIVE_FIELDS = (
    "optimizing_log_likelihood",
    "parameter_independent_carrier",
    "log_likelihood",
    "penalty_value",
    "penalized_optimizing_log_likelihood",
    "penalized_log_likelihood",
)


def _fsum(values: np.ndarray) -> float:
    return math.fsum(float(value) for value in np.ravel(values))


@dataclass(frozen=True)
class _GammaFixture:
    frame: pd.DataFrame
    response: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class _GammaFitDiagnostics:
    objectives: dict[str, float]
    objective_envelope: float
    penalized_score: np.ndarray
    score_envelope: np.ndarray
    kkt_upper: float
    kkt_ceiling: float


def _pair_tolerance(resolution: float, *values: float, extra: float = 0.0) -> float:
    return _outward_error(resolution * max((1.0, *(abs(value) for value in values))), extra)


def _gamma_surface_fixture(*, repetitions: int) -> _GammaFixture:
    levels = np.linspace(-1.0, 1.0, 12)
    x = np.tile(levels, repetitions)
    z = np.tile(np.roll(levels, 3), repetitions)
    mean = np.exp(0.7 + 0.55 * np.sin(np.pi * x))
    scale = np.exp(-0.8 + 0.45 * np.cos(np.pi * z))
    return _GammaFixture(
        frame=pd.DataFrame({"x": x, "z": z}),
        response=np.random.default_rng(91).gamma(shape=1.0 / scale**2, scale=mean * scale**2),
        counts=np.tile([1, 4, 2, 3, 1, 5, 2, 4, 3, 1, 2, 5], repetitions).astype(float),
    )


def _gamma_predictors(
    *,
    n_knots: int = 5,
    estimate_smoothing: bool = False,
) -> tuple[Predictor, Predictor]:
    lambda_policy = LambdaPolicy.estimate() if estimate_smoothing else None
    return (
        Predictor(
            "mean",
            {
                "x": Spline(
                    "cr",
                    n_knots=n_knots,
                    knot_strategy="quantile_rows",
                    lambda_policy=lambda_policy,
                )
            },
        ),
        Predictor(
            "scale",
            {
                "z": Spline(
                    "cr",
                    n_knots=n_knots,
                    knot_strategy="quantile_rows",
                    lambda_policy=lambda_policy,
                )
            },
        ),
    )


def _gamma_complete_fit_diagnostics(
    model,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    semantics: _WeightSemantics,
) -> _GammaFitDiagnostics:
    result, layout = model.result, model.layout
    n_rows, width = len(response), layout.n_coefficients
    mean, scale = np.asarray(result.theta).T
    shape = weights / scale**2 if semantics == "prior" else 1.0 / scale**2
    multiplier = np.ones_like(weights) if semantics == "prior" else weights
    ratio = response / mean
    log_ratio = np.log(ratio)
    deviance = ratio - 1.0 - log_ratio
    carrier_rows = -np.log(response) if semantics == "prior" else -weights * np.log(response)
    reported_rows = gamma_row_reference(response, mean, scale, weights, semantics)
    optimizing_rows = reported_rows - carrier_rows

    digamma = special.digamma(shape)
    bracket = digamma - np.log(shape) + deviance
    score_eta = np.column_stack(
        (multiplier * shape * (ratio - 1.0), 2.0 * multiplier * shape * bracket)
    )
    row_scales = np.column_stack(
        (
            abs(multiplier * shape) * (abs(ratio) + 1.0),
            2.0
            * abs(multiplier * shape)
            * (abs(digamma) + abs(np.log(shape)) + abs(ratio) + 1.0 + abs(log_ratio)),
        )
    )

    data_score, data_error = np.empty(width), np.empty(width)
    for parameter, (state, matrix) in enumerate(
        zip(layout.predictors, dense_predictor_matrices(layout), strict=True)
    ):
        for local, column in enumerate(
            range(state.coefficient_slice.start, state.coefficient_slice.stop)
        ):
            terms = matrix[:, local] * score_eta[:, parameter]
            data_score[column] = _fsum(terms)
            data_error[column] = _roundoff_factor(32 + 2 * n_rows) * _fsum(
                abs(matrix[:, local]) * row_scales[:, parameter]
            )

    penalty_terms = result.penalty * result.coefficients[None, :]
    penalty_score = np.array([_fsum(row) for row in penalty_terms])
    penalized_score = data_score - penalty_score
    score_envelope = np.nextafter(
        data_error
        + _roundoff_factor(2 * width + 1) * np.array([_fsum(abs(row)) for row in penalty_terms]),
        np.inf,
    )

    optimizing, carrier = _fsum(optimizing_rows), _fsum(carrier_rows)
    reported = optimizing + carrier
    quadratic_terms = result.coefficients[:, None] * penalty_terms
    penalty_value, absolute_penalty = (
        0.5 * _fsum(quadratic_terms),
        0.5 * _fsum(abs(quadratic_terms)),
    )
    q_optimizing, q_reported = optimizing - penalty_value, reported - penalty_value
    objectives = dict(
        zip(
            _OBJECTIVE_FIELDS,
            (optimizing, carrier, reported, penalty_value, q_optimizing, q_reported),
            strict=True,
        )
    )
    objective_envelope = _outward_error(
        _roundoff_factor(64 * n_rows + 3 * width**2 + 2 * width + 16)
        * (_fsum(abs(reported_rows)) + _fsum(abs(carrier_rows)) + absolute_penalty)
    )

    kkt_ceiling = _outward_error(
        2.0 * result.config.tolerance,
        _roundoff_factor(64 * (n_rows * width + width**2)),
    )
    score_norm, score_error_norm = (
        float(np.linalg.norm(values, ord=np.inf)) for values in (penalized_score, score_envelope)
    )
    upper_numerator = np.nextafter(score_norm + score_error_norm, np.inf)
    upper_denominator = np.nextafter(
        1.0 + max(0.0, np.nextafter(abs(q_optimizing) - objective_envelope, -np.inf)),
        -np.inf,
    )
    kkt_upper = float(np.nextafter(upper_numerator / upper_denominator, np.inf))
    return _GammaFitDiagnostics(
        objectives,
        objective_envelope,
        penalized_score,
        score_envelope,
        kkt_upper,
        kkt_ceiling,
    )


def _assert_gamma_oracle_reconciliation(model, diagnostics: _GammaFitDiagnostics) -> None:
    result = model.result
    for field, expected in diagnostics.objectives.items():
        actual = getattr(result, field)
        assert actual is not None
        assert abs(actual - expected) <= diagnostics.objective_envelope
    assert np.all(
        np.abs(result.terminal_score - diagnostics.penalized_score) <= diagnostics.score_envelope
    )
    assert diagnostics.kkt_upper <= diagnostics.kkt_ceiling


def _assert_terminal_provenance(left, right) -> None:
    results = left.result, right.result
    assert results[0].terminal_rank.rank == results[1].terminal_rank.rank
    curvatures = tuple(result.terminal_curvature for result in results)
    for field in ("requested_source", "actual_source", "reason", "fallback_count"):
        assert getattr(curvatures[0], field) == getattr(curvatures[1], field)
    assert curvatures[0].fallback_count == 0
    assert curvatures[0].actual_source == curvatures[0].requested_source
    for model in (left, right):
        inference = model.fit_state.inference
        source = model.result.terminal_curvature.actual_source
        assert inference.rank == model.result.terminal_rank.rank
        assert inference.covariance_curvature_source == inference.edf_curvature_source == source


def _pair_budget(
    left,
    right,
    diagnostics: tuple[_GammaFitDiagnostics, _GammaFitDiagnostics],
    *,
    outer_tolerance: float = 0.0,
) -> float:
    width = left.layout.n_coefficients
    assert width == right.layout.n_coefficients
    assert left.result.terminal_rank.rank == right.result.terminal_rank.rank == width
    curvatures = (
        left.result.terminal_penalized_curvature,
        right.result.terminal_penalized_curvature,
    )
    assert all(np.all(np.isfinite(curvature)) for curvature in curvatures)
    condition = max(float(np.linalg.cond(curvature)) for curvature in curvatures)
    assert math.isfinite(condition) and condition <= _EPS**-0.25
    n_rows = max(len(left.result.theta), len(right.result.theta))
    resolution = (
        (width + 1)
        * (1.0 + condition)
        * (
            max(item.kkt_ceiling for item in diagnostics)
            + outer_tolerance
            + _roundoff_factor(64 * (n_rows * width + width**2 + width**3))
        )
    )
    assert resolution <= _EPS**0.125
    return resolution


def _assert_fit_reproducibility(
    left,
    right,
    frame: pd.DataFrame,
    left_law: tuple[np.ndarray, np.ndarray, _WeightSemantics],
    right_law: tuple[np.ndarray, np.ndarray, _WeightSemantics],
    *,
    outer_tolerance: float = 0.0,
) -> tuple[_GammaFitDiagnostics, _GammaFitDiagnostics, float]:
    left_diagnostics, right_diagnostics = (
        _gamma_complete_fit_diagnostics(model, response, weights, semantics=semantics)
        for model, (response, weights, semantics) in zip(
            (left, right), (left_law, right_law), strict=True
        )
    )
    for model, diagnostic in zip((left, right), (left_diagnostics, right_diagnostics), strict=True):
        assert model.result.converged is True
        _assert_gamma_oracle_reconciliation(model, diagnostic)
    diagnostics = left_diagnostics, right_diagnostics
    budget = _pair_budget(left, right, diagnostics, outer_tolerance=outer_tolerance)
    surfaces = left.predict_eta(frame), right.predict_eta(frame)
    assert float(np.max(abs(surfaces[0] - surfaces[1]), initial=0.0)) <= _pair_tolerance(
        budget, *(float(np.max(abs(surface), initial=0.0)) for surface in surfaces)
    ), "log-parameter surface is not reproducible"
    for field in _OBJECTIVE_FIELDS:
        values = tuple(item.objectives[field] for item in diagnostics)
        assert abs(values[0] - values[1]) <= _pair_tolerance(
            budget,
            *values,
            extra=left_diagnostics.objective_envelope + right_diagnostics.objective_envelope,
        ), f"{field} is not reproducible"
    edf = left.fit_state.inference.total_edf, right.fit_state.inference.total_edf
    assert abs(edf[0] - edf[1]) <= _pair_tolerance(budget, left.layout.n_coefficients, *edf)
    _assert_terminal_provenance(left, right)
    return left_diagnostics, right_diagnostics, budget


def _assert_subspace_reproducibility(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
) -> None:
    bases, conditions = [], []
    for matrix in (left_matrix, right_matrix):
        n_rows, width = matrix.shape
        basis, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
        condition = float(singular_values[0] / singular_values[-1])
        assert math.isfinite(condition) and condition <= _EPS**-0.25
        conditions.append(condition)
        bases.append(basis)
    n_rows = max(left_matrix.shape[0], right_matrix.shape[0])
    width = max(left_matrix.shape[1], right_matrix.shape[1])
    projector_budget = _outward_error(
        (conditions[0] + conditions[1]) * _roundoff_factor(64 * n_rows * width**2)
    )
    assert projector_budget <= _EPS**0.25
    projector_difference = float(
        np.linalg.norm(bases[0] @ bases[0].T - bases[1] @ bases[1].T, ord=2)
    )
    assert projector_difference <= projector_budget, "design subspace is not reproducible"


def _kkt_lower_bound(diagnostics: _GammaFitDiagnostics) -> float:
    lower_numerator = max(
        0.0,
        float(
            np.nextafter(
                float(np.linalg.norm(diagnostics.penalized_score, ord=np.inf))
                - float(np.linalg.norm(diagnostics.score_envelope, ord=np.inf)),
                -np.inf,
            )
        ),
    )
    objective = diagnostics.objectives["penalized_optimizing_log_likelihood"]
    lower_denominator = np.nextafter(
        1.0 + np.nextafter(abs(objective) + diagnostics.objective_envelope, np.inf),
        np.inf,
    )
    return max(0.0, float(np.nextafter(lower_numerator / lower_denominator, -np.inf)))


@dataclass(frozen=True)
class _ImplementedLAMLReceipt:
    value: float
    coefficient_objective_error: float
    hessian_logdet_error: float
    penalty_logdet_error: float


def _estimated_disjoint_components(model: DenseDistributionalModel):
    components = tuple(model.layout.penalties)
    assert tuple(component.name for component in components) == tuple(model.lambdas)
    occupied: set[int] = set()
    for component in components:
        assert component.lambda_policy is not None and component.lambda_policy.mode == "estimate"
        assert component.penalty_kind == "dense" and component.omega_ssp is not None
        indices = set(range(component.group_sl.start, component.group_sl.stop))
        assert indices and occupied.isdisjoint(indices), "EFS components must be disjoint"
        occupied.update(indices)
        width = component.group_sl.stop - component.group_sl.start
        assert np.asarray(component.omega_ssp).shape == (width, width)
        assert float(int(component.rank)) == component.rank and 0 < component.rank <= width
    return components


def _independent_penalty_matrix(
    model: DenseDistributionalModel,
    lambdas,
) -> np.ndarray:
    penalty = np.zeros((model.layout.n_coefficients,) * 2, dtype=np.float64)
    for component in _estimated_disjoint_components(model):
        penalty[component.group_sl, component.group_sl] += float(
            lambdas[component.name]
        ) * np.asarray(
            component.omega_ssp,
            dtype=np.float64,
        )
    assert np.array_equal(penalty, penalty.T)
    return penalty


def _direct_raw_efs_residuals(
    model: DenseDistributionalModel,
    fit: DenseSolverResult,
    lambdas,
) -> dict[str, tuple[float, float]]:
    """Rebuild q, trace, denominator, and raw log residual without EFS update code."""
    beta = np.asarray(fit.coefficients, dtype=np.float64)
    retained_inverse = fit.terminal_rank.pseudo_inverse()
    evidence: dict[str, tuple[float, float]] = {}
    for component in _estimated_disjoint_components(model):
        local = component.group_sl
        omega = np.asarray(component.omega_ssp, dtype=np.float64)
        q_terms = beta[local, None] * omega * beta[None, local]
        trace_terms = retained_inverse[local, local] * omega.T
        denominator = math.fsum((_fsum(q_terms), _fsum(trace_terms)))
        denominator_error = _roundoff_factor(8 * omega.size + 16) * (
            _fsum(abs(q_terms)) + _fsum(abs(trace_terms))
        )
        assert denominator > denominator_error
        lambda_value = float(lambdas[component.name])
        assert math.isfinite(lambda_value) and lambda_value > 0.0
        residual = math.log(int(component.rank) / denominator) - math.log(lambda_value)
        error = _outward_error(
            -math.log1p(-denominator_error / denominator),
            _roundoff_factor(32) * max(1.0, abs(residual), abs(math.log(lambda_value))),
        )
        evidence[component.name] = residual, error
    return evidence


def _assert_terminal_raw_efs_evidence(model: DenseDistributionalModel) -> None:
    smoothing = model.smoothing
    assert smoothing is not None and smoothing.converged and smoothing.coefficient_converged
    assert smoothing.unresolved_upper_bound == ()
    assert not smoothing.history or smoothing.history[-1].boundary_nominations == ()
    assert all(
        math.isfinite(value)
        and smoothing.config.minimum_lambda < value < smoothing.config.maximum_lambda
        for value in smoothing.lambdas.values()
    )
    evidence = _direct_raw_efs_residuals(model, smoothing.terminal_fit, smoothing.lambdas)
    assert tuple(evidence) == tuple(smoothing.lambdas)
    assert all(
        abs(residual) <= smoothing.config.tolerance + error for residual, error in evidence.values()
    )
    reconstructed = max(abs(residual) for residual, _ in evidence.values())
    maximum_error = _outward_error(
        max(error for _, error in evidence.values()),
        _roundoff_factor(8 * len(evidence) + 8)
        * max(1.0, reconstructed, smoothing.terminal_raw_max_log_step),
    )
    assert abs(smoothing.terminal_raw_max_log_step - reconstructed) <= maximum_error


def _fit_provenance(fit: DenseSolverResult) -> tuple[object, ...]:
    curvature = fit.terminal_curvature
    rank = fit.terminal_rank
    return (
        fit.family_likelihood_plan_identifier,
        fit.execution_backend_identifier,
        curvature.requested_source,
        curvature.actual_source,
        curvature.fallback_count,
        rank.policy_version,
        rank.method,
        rank.rank,
        tuple(int(index) for index in rank.active_columns),
    )


def _assert_cross_route_fit_provenance(
    raw: DenseDistributionalModel,
    multisecant: DenseDistributionalModel,
    *,
    require_rejected_candidate: bool = True,
) -> None:
    assert raw.smoothing is not None and multisecant.smoothing is not None
    assert dict(raw.smoothing.initial_lambdas) == dict(multisecant.smoothing.initial_lambdas)
    expected = _fit_provenance(raw.smoothing.coefficient_fits[0])
    assert all(
        _fit_provenance(fit) == expected
        and fit.terminal_curvature.requested_source == fit.terminal_curvature.actual_source
        and fit.terminal_curvature.fallback_count == 0
        for smoothing in (raw.smoothing, multisecant.smoothing)
        for fit in smoothing.coefficient_fits
    )
    if require_rejected_candidate:
        assert any(
            iteration.acceleration_outcome == "rejected"
            for iteration in multisecant.smoothing.history
        ), "fixture must exercise rejected-candidate provenance"


def _capture_multisecant_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[MultisecantDecision]:
    """Capture real proposal diagnostics without changing controller behavior."""
    decisions: list[MultisecantDecision] = []
    original_proposal = WindowedTypeIIAnderson.propose

    def capture_proposal(self, **kwargs) -> MultisecantDecision:
        decision = original_proposal(self, **kwargs)
        decisions.append(decision)
        return decision

    monkeypatch.setattr(WindowedTypeIIAnderson, "propose", capture_proposal)
    return decisions


def _assert_genuine_multisecant_trials(
    model: DenseDistributionalModel,
    decisions: list[MultisecantDecision],
) -> None:
    smoothing = model.smoothing
    assert smoothing is not None
    assert len(decisions) == len(smoothing.history) and smoothing.accelerated_trial_count > 0
    trial = next(
        (iteration, decision.proposal)
        for iteration, decision in zip(smoothing.history, decisions, strict=True)
        if iteration.acceleration_outcome in {"accepted", "rejected"}
    )
    names = tuple(smoothing.lambdas)
    iteration, proposal = trial
    assert proposal is not None and proposal.secant_depth > 0 and proposal.numerical_rank > 0
    assert math.isfinite(proposal.raw_residual_norm) and math.isfinite(proposal.model_residual_norm)
    source = smoothing.coefficient_fits[iteration.source_fit_index]
    raw = _direct_raw_efs_residuals(model, source, iteration.lambdas_before)
    raw_logs = np.array(
        [
            np.clip(
                math.log(iteration.lambdas_before[name])
                + np.clip(
                    raw[name][0], -smoothing.config.max_log_step, smoothing.config.max_log_step
                ),
                math.log(smoothing.config.minimum_lambda),
                math.log(smoothing.config.maximum_lambda),
            )
            for name in names
        ]
    )
    candidate_logs = np.asarray(proposal.log_lambdas)
    assert (
        hashlib.sha256(candidate_logs.tobytes()).digest()
        != hashlib.sha256(raw_logs.tobytes()).digest()
    )
    assert iteration.accelerated_fit_index is not None
    candidate_lambdas = dict(zip(names, np.exp(candidate_logs), strict=True))
    np.testing.assert_array_equal(
        smoothing.coefficient_fits[iteration.accelerated_fit_index].penalty,
        _independent_penalty_matrix(model, candidate_lambdas),
    )


def _implemented_laml_receipt(
    *,
    coefficient_objective: float,
    coefficient_objective_error: float,
    hessian: tuple[float, float],
    penalty: tuple[float, float],
) -> _ImplementedLAMLReceipt:
    return _ImplementedLAMLReceipt(
        value=-coefficient_objective + 0.5 * (hessian[0] - penalty[0]),
        coefficient_objective_error=coefficient_objective_error,
        hessian_logdet_error=hessian[1],
        penalty_logdet_error=penalty[1],
    )


def _assert_multisecant_laml_not_worse(
    raw: DenseDistributionalModel,
    multisecant: DenseDistributionalModel,
    raw_receipt: _ImplementedLAMLReceipt,
    multisecant_receipt: _ImplementedLAMLReceipt,
) -> None:
    assert raw.smoothing is not None and multisecant.smoothing is not None
    receipts = raw_receipt, multisecant_receipt
    for model, receipt in zip((raw, multisecant), receipts, strict=True):
        scale = max(1.0, abs(receipt.value), abs(model.smoothing.objective))
        reconstruction_error = _outward_error(
            receipt.coefficient_objective_error,
            0.5 * receipt.hessian_logdet_error,
            0.5 * receipt.penalty_logdet_error,
            _roundoff_factor(32) * scale,
        )
        assert abs(model.smoothing.objective - receipt.value) <= reconstruction_error

    scale = 1.0 + max(abs(receipt.value) for receipt in receipts)
    normalized_interval = _outward_error(
        sum(receipt.coefficient_objective_error for receipt in receipts) / scale,
        0.5 * sum(receipt.hessian_logdet_error for receipt in receipts) / scale,
        0.5 * sum(receipt.penalty_logdet_error for receipt in receipts) / scale,
        _roundoff_factor(32) * max(1.0, *(abs(receipt.value) for receipt in receipts)) / scale,
    )
    assert (multisecant_receipt.value - raw_receipt.value) / scale <= normalized_interval


def _assert_default_none_identity(
    default: DenseDistributionalModel,
    explicit: DenseDistributionalModel,
    frame: pd.DataFrame,
) -> None:
    assert default.smoothing is not None and explicit.smoothing is not None
    np.testing.assert_array_equal(default.coefficients, explicit.coefficients)
    assert default.lambdas == explicit.lambdas
    assert default.smoothing.initial_objective == explicit.smoothing.initial_objective
    assert default.smoothing.objective == explicit.smoothing.objective
    assert default.smoothing.history == explicit.smoothing.history
    np.testing.assert_array_equal(default.covariance, explicit.covariance)
    assert default.inference.total_edf == explicit.inference.total_edf
    assert default.inference.predictor_edf == explicit.inference.predictor_edf
    np.testing.assert_array_equal(default.predict(frame), explicit.predict(frame))


def _assert_same_basin_then_observables(
    raw: DenseDistributionalModel,
    multisecant: DenseDistributionalModel,
    raw_continuation: DenseDistributionalModel,
    multisecant_continuation: DenseDistributionalModel,
    frame: pd.DataFrame,
) -> None:
    endpoints = raw, multisecant
    continuations = raw_continuation, multisecant_continuation
    models = (*endpoints, *continuations)
    for model in models:
        _assert_terminal_raw_efs_evidence(model)
    for endpoint, continuation in zip(endpoints, continuations, strict=True):
        assert endpoint.smoothing is not None and continuation.smoothing is not None
        assert continuation.smoothing.config.acceleration == "none"
        assert continuation.smoothing.config.tolerance < endpoint.smoothing.config.tolerance
        assert continuation.result.config.tolerance < endpoint.result.config.tolerance
        assert dict(continuation.smoothing.initial_lambdas) == dict(endpoint.smoothing.lambdas)

    smoothings = tuple(model.smoothing for model in models)
    assert all(smoothing is not None for smoothing in smoothings)
    assert (
        len({_fit_provenance(smoothing.terminal_fit) for smoothing in smoothings if smoothing}) == 1
    )
    names = tuple(raw.smoothing.lambdas)
    assert all(tuple(model.smoothing.lambdas) == names for model in continuations)
    continuation_logs = tuple(
        np.log([model.smoothing.lambdas[name] for name in names]) for model in continuations
    )
    curvature_condition = max(
        float(np.linalg.cond(model.result.terminal_penalized_curvature)) for model in models
    )
    assert math.isfinite(curvature_condition)
    width = raw.layout.n_coefficients
    arithmetic = _roundoff_factor(64 * (width**3 + len(frame) * width + 1))

    def resolution(selected, dimension):
        return (
            dimension
            * (1.0 + math.sqrt(curvature_condition))
            * (
                sum(model.smoothing.terminal_raw_max_log_step for model in selected)
                + sum(model.result.config.tolerance for model in selected)
                + arithmetic
            )
        )

    continuation_resolution = resolution(continuations, len(names) + 1)
    assert continuation_resolution < 1.0
    assert float(np.max(np.abs(continuation_logs[0] - continuation_logs[1]), initial=0.0)) <= (
        continuation_resolution
    )

    observable_resolution = resolution(endpoints, width + 1)
    assert observable_resolution < 1.0
    for left, right in (
        (raw.predict_eta(frame), multisecant.predict_eta(frame)),
        (raw.predict(frame), multisecant.predict(frame)),
    ):
        scale = max(
            1.0, float(np.max(abs(left), initial=0.0)), float(np.max(abs(right), initial=0.0))
        )
        assert float(np.max(abs(left - right), initial=0.0)) <= observable_resolution * scale
    edf_scale = max(
        1.0,
        width,
        abs(raw.inference.total_edf),
        abs(multisecant.inference.total_edf),
    )
    assert abs(raw.inference.total_edf - multisecant.inference.total_edf) / edf_scale <= (
        observable_resolution
    )


def test_gamma_two_surface_efs_is_start_stable_and_algorithm_matched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _gamma_surface_fixture(repetitions=20)
    inner_tolerance = _COMPLETE_FIT_TOLERANCE
    outer_tolerance = _COMPLETE_FIT_TOLERANCE

    family = GammaLS()
    resolved = resolve_likelihood_weights(
        np.ones(len(fixture.response)),
        n_observations=len(fixture.response),
        contract=WeightContract("prior"),
    )
    plan = family.bind_likelihood(fixture.response, resolved, COMPLETE_OBSERVATION)
    reference_initial_eta = np.log(family.initialize(fixture.response, plan).theta)
    displaced_initial_eta = reference_initial_eta.copy()
    displaced_initial_eta[:, 0] += math.log1p(math.sqrt(2.0))
    displaced_initial_eta[:, 1] -= 0.5 * math.log(2.0)

    def fit_model() -> SuperLSS:
        return SuperLSS(
            family=GammaLS(),
            predictors=_gamma_predictors(),
        ).fit_reml(
            fixture.frame,
            fixture.response,
            max_reml_iter=100,
            reml_tol=outer_tolerance,
            max_inner_iter=100,
            inner_tol=inner_tolerance,
            retain_rows=False,
            practical_reml=False,
            outer="efs",
        )

    reference = fit_model()
    original_initialize = GammaLS.initialize

    def displaced_initialization(self, y, plan):
        theta = np.array(original_initialize(self, y, plan).theta, copy=True)
        theta[:, 0] *= 1.0 + math.sqrt(2.0)
        theta[:, 1] /= math.sqrt(2.0)
        return InitialParameterState(theta=theta)

    monkeypatch.setattr(GammaLS, "initialize", displaced_initialization)
    displaced = fit_model()
    models = [reference._require_fitted(), displaced._require_fitted()]

    for model in models:
        smoothing = model.smoothing
        assert smoothing is not None
        assert smoothing.converged is True
        assert smoothing.convergence_reason in {"lambda_change", "objective_plateau"}
        assert smoothing.coefficient_converged is True
        assert model.result.convergence_reason in {
            "score",
            "objective_and_step",
            "objective_and_score",
        }
        assert all(math.isfinite(value) and value > 0.0 for value in smoothing.lambdas.values())
        assert math.isfinite(smoothing.objective)
        assert smoothing.matched_certified is True
        smoothing.assert_matched_certified()

    left, right = models
    unit_law = (fixture.response, np.ones(len(fixture.response)), "prior")
    _, _, budget = _assert_fit_reproducibility(
        left,
        right,
        fixture.frame,
        unit_law,
        unit_law,
        outer_tolerance=2.0
        * max(left.smoothing.config.tolerance, right.smoothing.config.tolerance),
    )
    for name in left.smoothing.lambdas:
        assert abs(
            math.log(left.smoothing.lambdas[name] / right.smoothing.lambdas[name])
        ) <= _pair_tolerance(budget)
    assert abs(left.smoothing.objective - right.smoothing.objective) <= _pair_tolerance(
        budget, left.smoothing.objective, right.smoothing.objective
    )
    for initial, model in zip((reference_initial_eta, displaced_initial_eta), (left, right)):
        assert float(np.max(abs(initial - model.result.eta))) > _pair_tolerance(
            budget, float(np.max(abs(initial))), float(np.max(abs(model.result.eta)))
        )


def _fit_gamma_cross_family_route(
    fixture: _GammaFixture,
    *,
    acceleration: Literal["none", "multisecant"] | None,
    lambdas: dict[str, float] | None = None,
    initial: np.ndarray | None = None,
    outer_tolerance: float = _CROSS_FAMILY_OUTER_TOLERANCE,
    inner_tolerance: float = _COMPLETE_FIT_TOLERANCE,
) -> DenseDistributionalModel:
    config_options: dict[str, object] = {
        "max_iterations": 100,
        "tolerance": outer_tolerance,
        "max_log_step": 5.0,
        "objective_tolerance": _COMPLETE_FIT_TOLERANCE,
        # The cross-family receipts describe the Fellner--Schall route.
        "outer": "efs",
    }
    if acceleration is not None:
        config_options["acceleration"] = acceleration
    return fit_dense_distributional(
        fixture.frame,
        fixture.response,
        family=GammaLS(),
        predictors=_gamma_predictors(estimate_smoothing=True),
        weight_contract=WeightContract("prior"),
        sample_weight=fixture.counts,
        lambdas=({"mean:x#wiggle": 0.1, "scale:z#wiggle": 0.1} if lambdas is None else lambdas),
        initial=initial,
        config=DenseSolverConfig(
            max_iterations=100,
            tolerance=inner_tolerance,
            coefficient_curvature="observed",
        ),
        efs_config=DistributionalEFSConfig(**config_options),
        retain_rows=False,
        discrete=False,
        chunk_size=None,
    )


def _gamma_laml_receipt(
    model: DenseDistributionalModel,
    diagnostics: _GammaFitDiagnostics,
) -> _ImplementedLAMLReceipt:
    hessian = _spectral_logdet(model.result.terminal_penalized_curvature, full_rank=True)
    assert hessian.rank == model.result.terminal_rank.rank == model.layout.n_coefficients
    penalty_rank = sum(int(component.rank) for component in model.layout.penalties)
    penalty = _spectral_logdet(_independent_penalty_matrix(model, model.lambdas), full_rank=False)
    assert penalty.rank == penalty_rank
    return _implemented_laml_receipt(
        coefficient_objective=diagnostics.objectives["penalized_optimizing_log_likelihood"],
        coefficient_objective_error=diagnostics.objective_envelope,
        hessian=(hessian.logdet, hessian.logdet_error),
        penalty=(penalty.logdet, penalty.logdet_error),
    )


def test_gamma_multisecant_cross_family_correctness_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the independent Gamma row oracle while certifying generic EFS."""
    fixture = _gamma_surface_fixture(repetitions=20)
    default = _fit_gamma_cross_family_route(fixture, acceleration=None)
    raw = _fit_gamma_cross_family_route(fixture, acceleration="none")
    decisions = _capture_multisecant_decisions(monkeypatch)
    multisecant = _fit_gamma_cross_family_route(fixture, acceleration="multisecant")

    _assert_genuine_multisecant_trials(multisecant, decisions)
    _assert_default_none_identity(default, raw, fixture.frame)
    _assert_cross_route_fit_provenance(raw, multisecant)
    diagnostics = tuple(
        _gamma_complete_fit_diagnostics(
            model,
            fixture.response,
            fixture.counts,
            semantics="prior",
        )
        for model in (raw, multisecant)
    )
    for model, item in zip((raw, multisecant), diagnostics, strict=True):
        _assert_gamma_oracle_reconciliation(model, item)
    _assert_multisecant_laml_not_worse(
        raw,
        multisecant,
        _gamma_laml_receipt(raw, diagnostics[0]),
        _gamma_laml_receipt(multisecant, diagnostics[1]),
    )

    continuation_tolerance = _CROSS_FAMILY_OUTER_TOLERANCE / 4.0
    continuation_inner_tolerance = _COMPLETE_FIT_TOLERANCE / 4.0
    continuations = tuple(
        _fit_gamma_cross_family_route(
            fixture,
            acceleration="none",
            lambdas=dict(endpoint.lambdas),
            initial=endpoint.coefficients,
            outer_tolerance=continuation_tolerance,
            inner_tolerance=continuation_inner_tolerance,
        )
        for endpoint in (raw, multisecant)
    )
    _assert_same_basin_then_observables(
        raw,
        multisecant,
        continuations[0],
        continuations[1],
        fixture.frame,
    )


def test_gamma_prior_and_integer_frequency_complete_fits_obey_distinct_laws() -> None:
    fixture = _gamma_surface_fixture(repetitions=4)
    expanded_take = np.repeat(np.arange(len(fixture.response)), fixture.counts.astype(np.intp))
    expanded_frame = fixture.frame.iloc[expanded_take].reset_index(drop=True)
    expanded_response = fixture.response[expanded_take]
    prior = _fit_gamma_compact_route(fixture, semantics="prior")
    frequency = _fit_gamma_compact_route(fixture, semantics="frequency")
    expanded = _fit_gamma_compact_route(
        fixture,
        frame=expanded_frame,
        response=expanded_response,
        weights=None,
        semantics="frequency",
    )
    prior_diagnostics = _gamma_complete_fit_diagnostics(
        prior,
        fixture.response,
        fixture.counts,
        semantics="prior",
    )
    assert prior.result.converged is True
    _assert_gamma_oracle_reconciliation(prior, prior_diagnostics)
    frequency_diagnostics, expanded_diagnostics, _ = _assert_fit_reproducibility(
        frequency,
        expanded,
        fixture.frame,
        (fixture.response, fixture.counts, "frequency"),
        (expanded_response, np.ones(len(expanded_response)), "frequency"),
    )

    prior_state = prior.fit_state
    assert prior_state.weight_contract == WeightContract("prior")
    assert prior_state.weight_provenance.physical_count == len(fixture.response)
    probabilities = np.linspace(0.0, 1.0, 6)[1:-1]
    for state, feature in zip(prior.fit_state.compiled_predictors, ("x", "z"), strict=True):
        expected = np.quantile(fixture.frame[feature], probabilities)
        np.testing.assert_allclose(
            state.compiled.specs[feature].fitted_knots,
            expected,
            rtol=0.0,
            atol=_roundoff_factor(len(fixture.frame)) * max(1.0, float(np.max(abs(expected)))),
        )

    frequency_design = build_joint_prediction_design(
        fixture.frame,
        frequency.fit_state.compiled_predictors,
        frequency.layout,
    )
    expanded_design = build_joint_prediction_design(
        fixture.frame,
        expanded.fit_state.compiled_predictors,
        expanded.layout,
    )
    for name in ("mean", "scale"):
        _assert_subspace_reproducibility(frequency_design.local[name], expanded_design.local[name])
    penalty_budget = _roundoff_factor(frequency.layout.n_coefficients**2) * max(
        1.0,
        float(np.linalg.norm(frequency.result.penalty, ord=np.inf)),
        float(np.linalg.norm(expanded.result.penalty, ord=np.inf)),
    )
    assert (
        float(np.linalg.norm(frequency.result.penalty - expanded.result.penalty, ord=np.inf))
        <= penalty_budget
    )

    wrong_law_diagnostics = _gamma_complete_fit_diagnostics(
        frequency,
        fixture.response,
        fixture.counts,
        semantics="prior",
    )
    assert _kkt_lower_bound(wrong_law_diagnostics) > wrong_law_diagnostics.kkt_ceiling

    separation_budget = _pair_budget(prior, frequency, (prior_diagnostics, frequency_diagnostics))
    surfaces = prior.predict_eta(fixture.frame), frequency.predict_eta(fixture.frame)
    assert float(np.max(abs(surfaces[0] - surfaces[1]), initial=0.0)) > _pair_tolerance(
        separation_budget, *(float(np.max(abs(surface), initial=0.0)) for surface in surfaces)
    )


def _fit_gamma_compact_route(
    fixture: _GammaFixture,
    *,
    frame: pd.DataFrame | None = None,
    response: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    semantics: _WeightSemantics = "prior",
    discrete: bool = False,
    n_bins: int = 16,
    chunk_size: int | str | None = None,
):
    return fit_dense_distributional(
        fixture.frame if frame is None else frame,
        fixture.response if response is None else response,
        family=GammaLS(),
        predictors=_gamma_predictors(n_knots=4),
        weight_contract=WeightContract(semantics),
        sample_weight=fixture.counts if weights is None and response is None else weights,
        lambdas={"mean:x#wiggle": 0.25, "scale:z#wiggle": 0.5},
        config=DenseSolverConfig(tolerance=_COMPLETE_FIT_TOLERANCE),
        retain_rows=False,
        discrete=discrete,
        n_bins=n_bins,
        chunk_size=chunk_size,
    )


@pytest.mark.parametrize(
    ("route", "expected_chunk", "n_bins", "lossless"),
    [
        ("chunked", 7, 16, True),
        ("exact-discrete", "auto", 16, True),
        ("lossy-discrete", "auto", 2, False),
    ],
)
def test_gamma_fixed_fit_matches_dense_on_generic_compact_routes(
    route: str,
    expected_chunk: int | str,
    n_bins: int,
    lossless: bool,
) -> None:
    fixture = _gamma_surface_fixture(repetitions=4)
    dense = _fit_gamma_compact_route(fixture)
    actual = _fit_gamma_compact_route(
        fixture,
        discrete=route != "chunked",
        n_bins=n_bins,
        chunk_size=expected_chunk,
    )
    assert dense.result.execution_backend_identifier == "distributional-dense-v1"
    assert dense.result.resolved_chunk_size is None
    resolved_chunk = len(fixture.response) if expected_chunk == "auto" else expected_chunk
    assert actual.result.execution_backend_identifier == "distributional-chunked-v1"
    assert actual.result.resolved_chunk_size == resolved_chunk
    assert actual.fit_state.requested_discrete is (route != "chunked")
    assert actual.fit_state.requested_n_bins == n_bins
    assert actual.fit_state.requested_chunk_size == expected_chunk
    if route != "chunked":
        groups = [
            group for state in actual.layout.predictors for group in state.design.group_matrices
        ]
        assert groups
        assert all(type(group) is DiscretizedSSPGroupMatrix for group in groups)
        if route == "exact-discrete":
            assert max(fixture.frame["x"].nunique(), fixture.frame["z"].nunique()) <= n_bins
            assert all(group.B_unique.shape[0] <= n_bins for group in groups)
        else:
            assert all(group.B_unique.shape[0] < 12 for group in groups)

    law = (fixture.response, fixture.counts, "prior")
    assert dense.fit_state.retained_rows is None
    assert actual.fit_state.retained_rows is None
    if lossless:
        _assert_fit_reproducibility(dense, actual, fixture.frame, law, law)
    else:
        with pytest.raises(AssertionError, match="log-parameter surface|log_likelihood"):
            _assert_fit_reproducibility(dense, actual, fixture.frame, law, law)


def _support_model() -> SuperLSS:
    return SuperLSS(
        family=GammaLS(),
        predictors=(
            Predictor("mean", {"x": _RejectMarkerNumeric()}),
            Predictor("scale", {}),
        ),
    )


def test_zero_weight_invalid_gamma_row_is_dropped_before_support_and_geometry() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 999.0, 2.0, 3.0, 4.0]})
    response = np.array([0.8, 1.2, 0.0, 1.8, 2.2, 2.7])
    weights = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])

    model = _support_model().fit(frame, response, sample_weight=weights)

    assert model._require_fitted().null_model.n_observations == 5


def test_positive_weight_invalid_gamma_row_refuses_before_geometry() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 999.0, 2.0, 3.0, 4.0]})
    response = np.array([0.8, 1.2, 0.0, 1.8, 2.2, 2.7])

    with pytest.raises(ValueError, match="strictly positive"):
        _support_model().fit(frame, response, sample_weight=np.ones(len(frame)))


def test_all_zero_gamma_weights_keep_the_public_all_dropped_refusal() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    response = np.array([0.8, 1.2, 1.8])

    with pytest.raises(LikelihoodWeightError, match="retain at least one row"):
        _support_model().fit(frame, response, sample_weight=np.zeros(len(frame)))


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_public_gamma_artifact_restore_and_refit_preserve_weight_semantics(
    semantics: _WeightSemantics,
) -> None:
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 16)})
    response = np.exp(0.45 + 0.25 * frame["x"].to_numpy()) * np.array(
        [
            0.72,
            0.91,
            1.18,
            0.83,
            1.27,
            0.95,
            1.36,
            0.88,
            1.12,
            1.41,
            0.79,
            1.23,
            1.04,
            1.32,
            0.86,
            1.16,
        ]
    )
    weights = (
        np.linspace(0.6, 1.8, len(frame))
        if semantics == "prior"
        else np.tile(np.array([1.0, 3.0, 2.0, 4.0]), 4)
    )
    predictors = (Predictor("mean", {"x": Numeric()}), Predictor("scale", {}))
    fitted = SuperLSS(
        family=GammaLS(),
        predictors=predictors,
        weight_semantics=semantics,
    ).fit(
        frame,
        response,
        sample_weight=weights,
        retain_rows=False,
    )
    restored = SuperLSS.from_bytes(fitted.to_bytes())

    assert type(restored.family) is GammaLS
    assert restored.family.to_config() == {"type": "GammaLS", "parameterization": "mean_cv"}
    assert type(restored.family_) is GammaLS
    assert restored.family_.to_config() == {"type": "GammaLS", "parameterization": "mean_cv"}
    assert restored.parameter_names_ == ("mean", "scale")
    assert restored.weight_semantics == semantics
    assert restored.discrete is False
    restored_predictors = restored.predictors
    assert tuple(predictor.name for predictor in restored_predictors) == (
        "mean",
        "scale",
    )
    assert tuple(tuple(predictor.features) for predictor in restored_predictors) == (
        ("x",),
        (),
    )
    assert (
        type(restored_predictors[0].features["x"]) is type(predictors[0].features["x"]) is Numeric
    )
    assert (
        tuple(predictor.link for predictor in restored_predictors)
        == tuple(predictor.link for predictor in predictors)
        == (None, None)
    )
    assert tuple(predictor.intercept for predictor in restored_predictors) == (True, True)
    assert all(
        predictor.interactions == ()
        and not predictor.interaction_specs
        and predictor.interaction_order == ()
        for predictor in restored_predictors
    )
    restored_state = restored._require_fitted().fit_state
    assert restored_state.retained_rows is None
    np.testing.assert_array_equal(
        restored.predict_parameters(frame), fitted.predict_parameters(frame)
    )
    np.testing.assert_array_equal(restored.predict(frame), fitted.predict(frame))

    previous_identifier = restored_state.family_likelihood_plan_identifier
    refit_dense_distributional(
        restored._require_fitted(),
        frame,
        response * np.linspace(1.01, 1.05, len(response)),
        sample_weight=weights,
    )
    refitted = restored._require_fitted().fit_state
    assert refitted.revision == 2
    assert refitted.weight_contract == WeightContract(semantics)
    assert refitted.retained_rows is None
    assert refitted.family_likelihood_plan_identifier != previous_identifier
