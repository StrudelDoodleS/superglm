"""First-trial dispatch and numerical correctness are independent contracts."""

import weakref
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels._common import _NumericalEvaluationError
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.result import DenseSolverConfig
from superglm.distributional.solver import chunks
from superglm.distributional.solver import solver as solver_api
from superglm.distributional.solver._global_moments import GlobalMomentRefusalError
from superglm.distributional.weights import UnsupportedLikelihoodContractError
from superglm.features import Numeric, Spline

from ._gaussian_lss_oracles import gamma, gaussian_row_oracle
from .test_distributional_automatic_panels import _literal_design
from .test_distributional_chunk_execution import _problem


def _context(problem=None, *, source="observed"):
    family, layout, y, plan, coefficients = _problem() if problem is None else problem
    context = solver_api._validated_context(
        family,
        layout,
        y,
        plan,
        np.eye(layout.n_coefficients) * 0.2,
        coefficient_curvature=source,
        chunk_size=7,
        coefficient_face=None,
    )
    return context, coefficients


def _run(context, coefficients, *, config=None, stop_policy="ordinary"):
    initial = solver_api._evaluate_state(context, coefficients)
    assert initial is not None
    geometry = solver_api._geometry(context, initial, context.coefficient_curvature)
    if config is None:
        config = DenseSolverConfig(
            max_iterations=1,
            tolerance=1e-12,
            max_predictor_step=0.1,
            coefficient_curvature=context.coefficient_curvature,
        )
    return initial, geometry, config


def test_accepted_first_trial_does_not_repeat_the_likelihood_pass(monkeypatch):
    context, coefficients = _context()
    initial, geometry, config = _run(context, coefficients)
    orders = []
    original = type(context.family).evaluate_natural

    def evaluate(self, *args, **kwargs):
        orders.append(kwargs.get("derivative_order", 2))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(context.family), "evaluate_natural", evaluate)
    run = solver_api._run_iterations(
        context, initial, config, initial_geometry=geometry, stop_policy="ordinary"
    )
    assert len(run.history) == 1
    assert run.history[0].backtracks == 0
    assert orders == [2] * 4, "accepted first trial repeated the value/geometry row pass"


def _combined(context, coefficients, **kwargs):
    return chunks._evaluate_chunked_geometry(
        context.family,
        context.layout,
        context.response,
        context.likelihood_plan,
        coefficients,
        penalty=context.penalty,
        chunk_size=context.chunk_size,
        curvature_source=context.coefficient_curvature,
        likelihood_cache=context.likelihood_cache,
        **kwargs,
    )


@pytest.mark.parametrize("failure", ["add", "finish"])
def test_partial_global_fallback_resets_likelihood_sums_and_closes_workspace(monkeypatch, failure):
    context, coefficients = _context()
    expected_geometry, expected_sums = _combined(
        context, coefficients, small_group_panel_byte_budget=None
    )
    events, refs = [], []

    class Plan:
        def reset(self, **kwargs):
            self.buffer = np.empty((4, 4))
            refs.append(weakref.ref(self.buffer))
            self.count = 0

        def add_chunk(self, *args):
            self.count += 1
            if self.count == 2 and failure == "add":
                raise GlobalMomentRefusalError("numerical refusal", recoverable=True)

        def finish(self):
            raise GlobalMomentRefusalError("numerical refusal", recoverable=True)

        def close(self):
            self.buffer = None
            events.append("closed")

    original = chunks._assemble_grouped_chunk_geometry

    def replay(*args, **kwargs):
        assert events == ["closed"]
        assert all(ref() is None for ref in refs)
        assert kwargs["_likelihood_sums"] == [0.0, 0.0]
        return original(*args, **kwargs)

    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: 64 << 20)
    monkeypatch.setattr(
        chunks, "build_global_moment_plan", lambda *args, **kwargs: SimpleNamespace(plan=Plan())
    )
    monkeypatch.setattr(chunks, "_assemble_grouped_chunk_geometry", replay)
    geometry, sums = _combined(context, coefficients)
    assert sums == expected_sums
    np.testing.assert_allclose(geometry.score_data, expected_geometry.score_data, rtol=0, atol=0)
    np.testing.assert_allclose(
        geometry.data_curvature, expected_geometry.data_curvature, rtol=0, atol=0
    )


def test_derivative_failure_releases_partial_trial_before_value_retry(monkeypatch):
    context, coefficients = _context()
    initial, geometry, config = _run(context, coefficients)
    original = type(context.family).evaluate_natural
    accumulator_type = chunks.GroupedGeometryAccumulator
    orders, refs = [], []

    def accumulator(*args, **kwargs):
        result = accumulator_type(*args, **kwargs)
        refs.append(weakref.ref(result._curvature))
        return result

    def evaluate(self, *args, **kwargs):
        order = kwargs.get("derivative_order", 2)
        orders.append(order)
        if len(orders) == 2:
            raise FloatingPointError("temporary derivative domain failure")
        if order == 0:
            assert all(ref() is None for ref in refs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(chunks, "GroupedGeometryAccumulator", accumulator)
    monkeypatch.setattr(type(context.family), "evaluate_natural", evaluate)
    run = solver_api._run_iterations(
        context, initial, config, initial_geometry=geometry, stop_policy="ordinary"
    )
    assert run.history[0].backtracks == 0
    assert orders == [2, 2] + [0] * 4 + [2] * 4


@pytest.mark.parametrize("error", [UnsupportedLikelihoodContractError, ValueError])
def test_hard_source_failure_never_retries_a_value_screen(monkeypatch, error):
    context, coefficients = _context()
    initial, geometry, config = _run(context, coefficients)
    orders = []
    original = type(context.family).evaluate_natural

    def evaluate(self, *args, **kwargs):
        orders.append(kwargs.get("derivative_order", 2))
        if error is UnsupportedLikelihoodContractError:
            raise error("invalid source contract")
        return original(self, *args, **kwargs)

    def invalid_source(*args, **kwargs):
        raise error("invalid source contract")

    monkeypatch.setattr(type(context.family), "evaluate_natural", evaluate)
    if error is ValueError:
        monkeypatch.setattr(chunks, "_take_likelihood_rows", invalid_source)
    with pytest.raises(error, match="invalid source contract"):
        solver_api._run_iterations(
            context, initial, config, initial_geometry=geometry, stop_policy="ordinary"
        )
    assert 0 not in orders


def test_rejected_fused_geometry_is_released_before_later_value_screen(monkeypatch):
    context, coefficients = _context()
    initial, geometry, config = _run(context, coefficients)
    fused_original = solver_api._evaluate_fused_trial
    value_original = solver_api._evaluate_state
    refs = []

    def rejected(*args, **kwargs):
        result = fused_original(*args, **kwargs)
        assert result is not None
        candidate, trial_geometry = result
        refs.append(weakref.ref(trial_geometry))
        return replace(candidate, penalized_optimizing_log_likelihood=-1e100), trial_geometry

    def screened(*args, **kwargs):
        assert len(refs) == 1 and refs[0]() is None
        return value_original(*args, **kwargs)

    monkeypatch.setattr(solver_api, "_evaluate_fused_trial", rejected)
    monkeypatch.setattr(solver_api, "_evaluate_state", screened)
    run = solver_api._run_iterations(
        context, initial, config, initial_geometry=geometry, stop_policy="ordinary"
    )
    assert len(run.history) == 1
    assert run.history[0].backtracks == 1
    assert len(refs) == 1


@pytest.mark.parametrize("unsupported", ["score_only", "family", "link"])
def test_unsupported_trials_keep_existing_value_and_geometry_calls(monkeypatch, unsupported):
    context, coefficients = _context()
    initial, geometry, config = _run(context, coefficients)
    if unsupported == "family":

        class CustomFamily(type(context.family)):
            pass

        context = replace(context, family=CustomFamily(scale_floor=context.family.scale_floor))
    elif unsupported == "link":

        class CustomLink(type(context.links[0])):
            pass

        context = replace(context, links=(CustomLink(), *context.links[1:]))

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported trial entered fused evaluation")

    monkeypatch.setattr(solver_api, "_evaluate_fused_trial", forbidden)
    run = solver_api._run_iterations(
        context,
        initial,
        config,
        initial_geometry=geometry,
        stop_policy="score_only" if unsupported == "score_only" else "ordinary",
    )
    assert len(run.history) == 1


def test_malformed_derivative_shape_is_hard_and_not_a_value_retry(monkeypatch):
    context, coefficients = _context()
    initial, geometry, config = _run(context, coefficients)
    original = chunks.transform_natural_derivatives

    def malformed(natural, eta, links):
        return original(natural, eta[:, :1], links)

    def forbidden(*args, **kwargs):
        raise AssertionError("malformed derivative shape reached a value retry")

    monkeypatch.setattr(chunks, "transform_natural_derivatives", malformed)
    monkeypatch.setattr(solver_api, "_evaluate_state", forbidden)
    with pytest.raises(ValueError, match="eta shape"):
        solver_api._run_iterations(
            context, initial, config, initial_geometry=geometry, stop_policy="ordinary"
        )


def test_live_layout_link_replacement_refuses_fused_admission():
    context, _ = _context()
    assert solver_api._fused_first_trial_eligible(context)

    class CustomLink(type(context.links[0])):
        pass

    layout = replace(
        context.layout,
        predictors=(
            replace(context.layout.predictors[0], link=CustomLink()),
            *context.layout.predictors[1:],
        ),
    )
    assert not solver_api._fused_first_trial_eligible(replace(context, layout=layout))


def test_nonfinite_completed_likelihood_sums_refuse_only_the_fused_trial(monkeypatch):
    context, coefficients = _context()
    original = chunks._assemble_chunked_geometry

    def nonfinite(*args, **kwargs):
        geometry = original(*args, **kwargs)
        kwargs["_likelihood_sums"][0] = -np.inf
        return geometry

    monkeypatch.setattr(chunks, "_assemble_chunked_geometry", nonfinite)
    assert solver_api._evaluate_fused_trial(context, coefficients, "observed", None) is None


def _family_problem(family_type):
    _, _, y, old_plan, _ = _problem()
    family = family_type()
    x = np.linspace(-0.9, 1.1, len(y))
    location, scale = (spec.name for spec in family.parameters)
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(pd.DataFrame({"x": x, "z": np.sin(2.3 * x)})),
            old_plan.weights,
            family.parameters,
            (
                Predictor(
                    location, {"x": Numeric(), "z": Spline(kind="cr", n_knots=4, discrete=True)}
                ),
                Predictor(scale, {"x": Numeric()}),
            ),
            offsets={location: 0.03 * x**2, scale: -0.1 + 0.02 * x},
            model_discrete=True,
            n_bins_config=7,
        )
    )
    plan = family.bind_likelihood(y, old_plan.weights, COMPLETE_OBSERVATION)
    coefficients = np.linspace(-0.2, 0.3, layout.n_coefficients)
    return family, layout, y, plan, coefficients


def test_gamma_unrepresentable_derivative_keeps_its_finite_value_screen():
    context, coefficients = _context(_family_problem(GammaLS))
    coefficients[:] = 0
    coefficients[0] = np.log(1e-200)
    # This is a real numeric refusal: likelihood O(1e200) is representable,
    # whereas the natural mean score O(1e400) exceeds float64 range.
    with pytest.raises(_NumericalEvaluationError, match="not representable"):
        chunks.assemble_chunked_geometry(
            context.family,
            context.layout,
            context.response,
            context.likelihood_plan,
            coefficients,
            penalty=context.penalty,
            chunk_size=7,
            curvature_source="observed",
        )
    assert solver_api._evaluate_fused_trial(context, coefficients, "observed", None) is None
    screened = solver_api._evaluate_state(context, coefficients)
    assert screened is not None
    assert np.isfinite(screened.penalized_optimizing_log_likelihood)


def test_same_chunk_authority_mutation_wins_over_numerical_derivative_failure(monkeypatch):
    context, coefficients = _context()
    group = next(
        group
        for state in context.layout.predictors
        for group in state.design.group_matrices
        if hasattr(group, "B_unique")
    )
    original = type(context.family).evaluate_natural
    calls = []

    def mutate(self, *args, **kwargs):
        calls.append(kwargs.get("derivative_order", 2))
        if len(calls) == 2:
            group.B_unique = group.B_unique.copy()
            group.B_unique[0, 0] += 0.5
            raise FloatingPointError("numeric derivative failure after live mutation")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: 64 << 20)
    monkeypatch.setattr(type(context.family), "evaluate_natural", mutate)
    with pytest.raises(GlobalMomentRefusalError, match="authority") as error:
        solver_api._evaluate_fused_trial(context, coefficients, "observed", None)
    assert not error.value.recoverable
    assert calls == [2, 2]


@pytest.mark.parametrize("global_moments", [False, True])
def test_fused_gaussian_cancellation_obeys_independent_backward_bounds(monkeypatch, global_moments):
    context, coefficients = _context()
    location, scale = context.layout.predictors
    coefficients[scale.coefficient_slice] = 0
    coefficients[0] = 2.0**20
    offset = location.offset - 2.0**20
    layout = replace(context.layout, predictors=(replace(location, offset=offset), scale))
    penalty = context.penalty.copy()
    penalty[0, 0] = 0
    context = replace(context, layout=layout, penalty=penalty)
    if global_moments:
        monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: 64 << 20)
    matrices = _literal_design(layout)
    eta = np.column_stack(
        [
            np.asarray(matrix, dtype=np.longdouble) @ coefficients[state.coefficient_slice]
            + np.asarray(state.offset, dtype=np.longdouble)
            for matrix, state in zip(matrices, layout.predictors, strict=True)
        ]
    ).astype(np.float64)
    sigma = context.family.scale_floor + np.exp(eta[:, 1])
    rows = gaussian_row_oracle(
        context.response,
        eta[:, 0],
        sigma,
        context.likelihood_plan.weights.values,
        semantics="prior",
        scale_floor=context.family.scale_floor,
    )
    geometry, sums = _combined(context, coefficients)
    # The location offset cancels a large intercept. Bound predictor roundoff
    # from absolute operands, then propagate it through the exact residual
    # polynomial; relative tolerance on the tiny residual would be inappropriate.
    n, q = len(context.response), len(coefficients)
    x, z = (np.asarray(matrix, dtype=np.longdouble) for matrix in matrices)
    eta_error = gamma(64 * q) * (
        np.abs(x) @ np.abs(coefficients[location.coefficient_slice]) + np.abs(offset) + 1
    )
    residual = context.response - eta[:, 0]
    weight = context.likelihood_plan.weights.values
    d = sigma - context.family.scale_floor
    residual2_error = 2 * np.abs(residual) * eta_error + eta_error**2
    ll_error = weight / (2 * sigma**2) * residual2_error
    score_error = np.column_stack(
        (
            weight / sigma**2 * eta_error,
            d * weight / sigma**3 * residual2_error,
        )
    )
    curvature_error = np.column_stack(
        (
            np.zeros(n),
            2 * weight * d / sigma**3 * eta_error,
            weight * (3 * d**2 / sigma**4 + d / sigma**3) * residual2_error,
        )
    )
    score_expected = np.concatenate((x.T @ rows.link_score[:, 0], z.T @ rows.link_score[:, 1]))
    score_bound = np.concatenate((np.abs(x).T @ score_error[:, 0], np.abs(z).T @ score_error[:, 1]))
    score_bound += gamma(64 * n + 64 * q) * np.maximum(
        1,
        np.concatenate(
            (
                np.abs(x).T @ np.abs(rows.link_score[:, 0]),
                np.abs(z).T @ np.abs(rows.link_score[:, 1]),
            )
        ),
    )
    assert np.all(np.abs(geometry.score_data - score_expected) <= score_bound)
    expected = np.zeros((q, q), dtype=np.longdouble)
    bound = np.zeros_like(expected)
    for channel, (a, b) in enumerate(((0, 0), (0, 1), (1, 1))):
        left, right = (x, z)[a], (x, z)[b]
        index = (layout.predictors[a].coefficient_slice, layout.predictors[b].coefficient_slice)
        expected[index] = left.T @ (rows.observed_link_curvature_packed[:, channel, None] * right)
        absolute = np.abs(left).T @ (
            np.abs(rows.observed_link_curvature_packed[:, channel, None]) * np.abs(right)
        )
        bound[index] = np.abs(left).T @ (curvature_error[:, channel, None] * np.abs(right)) + gamma(
            96 * n + 64 * q
        ) * np.maximum(1, absolute)
        if a != b:
            expected[index[::-1]] = expected[index].T
            bound[index[::-1]] = bound[index].T
    assert np.all(np.abs(geometry.data_curvature - expected) <= bound)
    likelihood_bound = np.sum(ll_error) + gamma(64 * n) * np.sum(
        np.abs(rows.optimizing_log_likelihood)
    )
    assert (
        abs(
            sums.optimizing_log_likelihood
            - np.sum(rows.optimizing_log_likelihood, dtype=np.longdouble)
        )
        <= likelihood_bound
    )
    carrier_bound = gamma(64 * n) * max(1, np.sum(np.abs(rows.parameter_independent_carrier)))
    assert (
        abs(
            sums.parameter_independent_carrier
            - np.sum(rows.parameter_independent_carrier, dtype=np.longdouble)
        )
        <= carrier_bound
    )


@pytest.mark.parametrize("family_type", [GaussianLS, GammaLS])
def test_small_complete_fit_preserves_stationarity_and_armijo_progress(monkeypatch, family_type):
    context, coefficients = _context(_family_problem(family_type))
    config = DenseSolverConfig(tolerance=1e-9, max_iterations=80)

    def fit():
        return solver_api.fit_dense_fixed_lambda(
            context.family,
            context.layout,
            context.response,
            context.likelihood_plan,
            context.penalty,
            initial=coefficients,
            chunk_size=7,
            config=config,
        )

    fused = fit()
    monkeypatch.setattr(solver_api, "_fused_first_trial_eligible", lambda context: False)
    baseline = fit()
    assert fused.converged and baseline.converged
    for result in (fused, baseline):
        assert all(step.objective_after >= step.objective_before for step in result.history)
    smallest = min(
        np.linalg.eigvalsh(result.terminal_penalized_curvature)[0] for result in (fused, baseline)
    )
    assert smallest > 0
    # On this well-conditioned fixture, observed residuals give a coefficient
    # backward-error budget through the positive curvature spectral gap.
    bound = (
        4 * sum(np.linalg.norm(result.terminal_score) for result in (fused, baseline)) / smallest
    )
    bound += gamma(128 * len(context.response) * len(coefficients)) * (
        1 + np.linalg.norm(coefficients)
    )
    assert np.linalg.norm(fused.coefficients - baseline.coefficients) <= bound
