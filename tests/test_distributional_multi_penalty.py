"""Finite shared-penalty LAML geometry and independent public profile checks."""

import functools
import math
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest
from scipy.special import digamma, gammaln, polygamma

from superglm import SuperLSS
from superglm.distributional import GammaLS, GaussianLS, Predictor
from superglm.distributional._row_design import bounded_predictor_matrices
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.smoothing import endpoint_laml
from superglm.distributional.smoothing.derivatives import laml_derivatives
from superglm.distributional.smoothing.face_efs import projected_component_states
from superglm.distributional.smoothing.objective import joint_laplace_objective
from superglm.features import Spline
from superglm.reml.penalty_algebra import _compute_penalty_logdet_evaluation
from superglm.types import LambdaPolicy
from tests.test_distributional_endpoint_laml import (
    _axis_aligned_projected_face,
    _projected_penalty_problem,
)


def _shared_face_problem():
    layout, face, lambdas, _, _ = _projected_penalty_problem()
    selected, left, right = layout.penalties
    left_matrix = np.diag([1.0, 1.0, 0.0])
    right_matrix = np.diag([0.0, 1.0, 1.0])
    left = replace(left, omega_raw=left_matrix, omega_ssp=left_matrix)
    right = replace(right, omega_raw=right_matrix, omega_ssp=right_matrix)
    inactive = replace(right, name="location:group#inactive", lambda_policy=LambdaPolicy.fixed(0.0))
    zero = replace(
        right,
        name="location:group#zero",
        omega_raw=np.zeros((3, 3)),
        omega_ssp=np.zeros((3, 3)),
        rank=0,
    )
    layout = replace(layout, penalties=(selected, left, right, inactive, zero))
    face, _ = _axis_aligned_projected_face(layout, face, left)
    lambdas = {
        selected.name: 13.0,
        left.name: 1e12,
        right.name: 3.0,
        inactive.name: 0.0,
        zero.name: 2.0,
    }
    return layout, face, lambdas


def test_projected_shared_penalty_retains_positive_support_and_inactive_names():
    """Kills weighted rank loss after projection to a disjoint exact face."""
    layout, face, lambdas = _shared_face_problem()
    result = endpoint_laml.projected_finite_penalty_logdet(
        layout=layout, face=face, lambdas=lambdas
    )
    assert result.rank == 3
    evaluation = endpoint_laml._projected_finite_penalty_evaluation(
        layout=layout, face=face, lambdas=lambdas
    )
    expected = np.log(1e12) + np.log(3) + np.log(1e12 + 3)
    assert abs(evaluation.logdet - expected) <= (
        evaluation.logdet_error + 4 * np.finfo(float).eps * abs(expected)
    )
    states = projected_component_states(layout=layout, face=face, lambdas=lambdas)
    assert tuple(state.name for state in states) == result.component_names
    for state in states:
        assert state.rank == evaluation.gradient[state.name]
    assert evaluation.gradient["location:group#inactive"] == 0
    assert evaluation.gradient["location:group#zero"] == 0


def test_projected_production_uses_source_roots_without_reforming_a_gram(monkeypatch):
    layout, face, lambdas = _shared_face_problem()

    def obsolete_projection(**_kwargs):
        pytest.fail("finite face geometry must project component roots")

    monkeypatch.setattr(endpoint_laml, "_projected_finite_penalty_inputs", obsolete_projection)
    result = endpoint_laml.projected_finite_penalty_logdet(
        layout=layout, face=face, lambdas=lambdas
    )
    assert result.rank == 3


def test_nonzero_finite_component_cannot_be_projected_away_by_a_disjoint_face():
    layout, face, lambdas = _shared_face_problem()
    selected, left, *others = layout.penalties
    overlap = replace(
        left,
        group_sl=selected.group_sl,
        omega_ssp=np.ones((1, 1)),
        omega_raw=np.ones((1, 1)),
        rank=1,
    )
    invalid_layout = replace(layout, penalties=(selected, overlap, *others))
    with pytest.raises(ValueError, match="shared|overlapping|layout"):
        endpoint_laml.projected_finite_penalty_logdet(
            layout=invalid_layout, face=face, lambdas=lambdas
        )


def _public_data(kind):
    x = np.linspace(-1.0, 1.0, 384)
    frame = pd.DataFrame({"x": x})
    if kind == "gaussian":
        rng = np.random.default_rng(248)
        y = 0.4 + 0.8 * x + np.exp(-0.4 + 0.35 * x) * rng.normal(size=x.size)
        family, names = GaussianLS(scale_floor=0.0), ("location", "scale")
    else:
        rng = np.random.default_rng(249)
        mean, cv = np.exp(0.4 + 0.8 * x), np.exp(-0.8 + 0.25 * x)
        y = mean * rng.gamma(shape=1 / cv**2, scale=cv**2)
        family, names = GammaLS(), ("mean", "scale")
    predictors = (
        Predictor(names[0], {"x": Spline(kind="cr", k=6, m=(1, 2))}),
        Predictor(names[1], {"x": Spline(kind="cr", k=5, m=(1, 2))}),
    )
    return frame, y, family, predictors


def _weights(kind):
    first = "location" if kind == "gaussian" else "mean"
    return {f"{first}:x#d1": 3.0, f"{first}:x#d2": 2.0, "scale:x#d1": 1.0, "scale:x#d2": 2.0}


@functools.lru_cache(maxsize=256)
def _public_fit(kind, discrete, rho=(0.0, 0.0), origin=None):
    frame, y, family, predictors = _public_data(kind)
    weights = _weights(kind) if origin is None else dict(origin)
    for name, step in zip(tuple(weights)[:2], rho, strict=True):
        weights[name] *= math.exp(step)
    model = SuperLSS(family=family, predictors=predictors, discrete=discrete, n_bins=512)
    model.fit(frame, y, lambdas=weights, max_inner_iter=150, inner_tol=1e-10)
    return model


@dataclass
class _IndependentMode:
    objective: float
    error: float
    mode_error: float
    provenance: tuple
    matrices: tuple


def _independent_mode(model, kind):
    """Analytic likelihood/observed Hessian and unweighted component roots.

    This oracle never calls the production objective, penalty/rank kernel or
    derivative helper. Its mode error follows a local Newton residual bound;
    derivative differences also account for this error at every public refit.
    """
    fitted = model._require_fitted()
    state = fitted.fit_state
    fit, layout = state.solver_result, state.layout
    y = np.asarray(state.retained_rows.response)
    beta = np.asarray(fit.coefficients)
    matrices = tuple(
        np.column_stack([np.ones(len(y)), predictor.design.toarray()])
        for predictor in layout.predictors
    )
    slices = [predictor.coefficient_slice for predictor in layout.predictors]
    eta = np.column_stack([matrix @ beta[block] for matrix, block in zip(matrices, slices)])
    p = len(beta)
    u = np.finfo(float).eps / 2
    gamma = 8 * (len(y) + p) * u / (1 - 8 * (len(y) + p) * u)
    if kind == "gaussian":
        residual = y - eta[:, 0]
        v = np.exp(-2 * eta[:, 1])
        log_likelihood = -0.5 * (np.log(2 * np.pi) + 2 * eta[:, 1] + residual**2 * v)
        score_rows = np.column_stack([residual * v, residual**2 * v - 1])
        curvature = np.array([[v, 2 * residual * v], [2 * residual * v, 2 * residual**2 * v]])
        third = np.array(
            [
                [[np.zeros(len(y)), -2 * v], [-2 * v, -4 * residual * v]],
                [[-2 * v, -4 * residual * v], [-4 * residual * v, -4 * residual**2 * v]],
            ]
        )
    else:
        a, t = np.exp(-2 * eta[:, 1]), y * np.exp(-eta[:, 0])
        log_t = np.log(y) - eta[:, 0]
        b = np.log(a) + 1 + log_t - t - digamma(a)
        c = b + 1 - a * polygamma(1, a)
        # The model's optimizing law separates the fixed carrier -log(y).
        log_likelihood = a * (np.log(a) + log_t - t) - gammaln(a)
        score_rows = np.column_stack([a * (t - 1), -2 * a * b])
        curvature = np.array([[a * t, 2 * a * (t - 1)], [2 * a * (t - 1), -4 * a * c]])
        third = np.array(
            [
                [[-a * t, -2 * a * t], [-2 * a * t, -4 * a * (t - 1)]],
                [
                    [-2 * a * t, -4 * a * (t - 1)],
                    [
                        -4 * a * (t - 1),
                        8 * a * (c + 1 - 2 * a * polygamma(1, a) - a**2 * polygamma(2, a)),
                    ],
                ],
            ]
        )
    penalty = np.zeros((p, p))
    groups = {}
    for component in layout.penalties:
        matrix = np.asarray(component.omega_ssp)
        values, vectors = np.linalg.eigh(matrix)
        threshold = 16 * matrix.shape[0] * u * np.max(np.abs(values))
        positive = values > threshold
        root = np.sqrt(values[positive])[:, None] * vectors[:, positive].T
        reconstruction = np.linalg.norm(root.T @ root - matrix)
        assert reconstruction <= 4 * threshold * np.sqrt(matrix.shape[0])
        weight = state.lambdas[component.name]
        penalty[component.group_sl, component.group_sl] += weight * matrix
        groups.setdefault(component.group_name, []).append(np.sqrt(weight) * root)
    penalty_logdet = 0.0
    penalty_log_scale = 0.0
    penalty_condition = 0.0
    for roots in groups.values():
        stacked = np.vstack(roots)
        _, triangular = np.linalg.qr(stacked, mode="reduced")
        logs = 2 * np.log(np.abs(np.diag(triangular)))
        penalty_logdet += math.fsum(logs)
        penalty_log_scale += math.fsum(np.abs(logs))
        penalty_condition += np.linalg.cond(stacked)
    hessian = penalty.copy()
    score = -(penalty @ beta)
    score_scale = np.abs(penalty) @ np.abs(beta)
    for q, (matrix, block) in enumerate(zip(matrices, slices)):
        score[block] += matrix.T @ score_rows[:, q]
        score_scale[block] += np.abs(matrix).T @ np.abs(score_rows[:, q])
        for r, (other, other_block) in enumerate(zip(matrices, slices)):
            hessian[block, other_block] += matrix.T @ (curvature[q, r, :, None] * other)
    sign, hessian_logdet = np.linalg.slogdet(hessian)
    assert sign > 0
    inverse = np.linalg.inv(hessian)
    coefficient_objective = -math.fsum(log_likelihood) + 0.5 * (beta @ penalty @ beta)
    objective = coefficient_objective + 0.5 * (hessian_logdet - penalty_logdet)
    score_bound = gamma * score_scale
    score_threshold = fit.config.tolerance * (1 + abs(coefficient_objective))
    if np.max(np.abs(score)) > score_threshold + np.max(score_bound):
        # Check the independent objective-resolution invariant even when the
        # public coefficient solver labels this mode "objective_and_step".
        enclosed_score = np.abs(score) + score_bound
        decrement_bound = 0.5 * enclosed_score @ np.abs(inverse) @ enclosed_score
        decrement_bound *= 1 + gamma * np.linalg.cond(hessian)
        assert decrement_bound <= np.spacing(abs(coefficient_objective))
    assert np.linalg.norm(score - fit.terminal_score) <= 2 * np.linalg.norm(score_bound)
    assert fit.converged and fit.terminal_curvature.actual_source == "observed"
    assert fit.terminal_rank.rank == p
    inverse_norm = np.linalg.norm(inverse, 2)
    correction = np.linalg.solve(hessian, score)
    solve_residual = hessian @ correction - score
    solve_bound = gamma * (
        np.linalg.norm(hessian, 2) * np.linalg.norm(correction) + np.linalg.norm(score)
    )
    assert np.linalg.norm(solve_residual) <= solve_bound
    radius = 2 * (
        np.linalg.norm(correction)
        + inverse_norm * (np.linalg.norm(solve_residual) + np.linalg.norm(score_bound))
    )
    drift_gradient = np.zeros(p)
    drift_norm_bound = 0.0
    for t_index, (matrix, block) in enumerate(zip(matrices, slices)):
        row_drift = np.zeros(len(y))
        for q, (left, left_block) in enumerate(zip(matrices, slices)):
            for r, (right, right_block) in enumerate(zip(matrices, slices)):
                leverage = np.einsum("ij,jk,ik->i", left, inverse[left_block, right_block], right)
                row_drift += leverage * third[t_index, q, r]
                drift_norm_bound += (
                    np.linalg.norm(left, 2)
                    * np.linalg.norm(right, 2)
                    * np.max(np.abs(third[t_index, q, r]))
                    * np.linalg.norm(matrix, 2)
                )
        drift_gradient[block] = 0.5 * matrix.T @ row_drift
    # The observed Hessian stays invertible throughout the inferred Newton
    # ball. This is a local test-fixture check, not a global uniqueness claim.
    assert inverse_norm * drift_norm_bound * radius < 0.25
    mode_error = 2 * np.linalg.norm(drift_gradient - score) * radius
    mode_error += 2 * np.linalg.norm(hessian, 2) * radius**2
    error = gamma * (
        math.fsum(np.abs(log_likelihood))
        + abs(beta @ penalty @ beta)
        + p * np.linalg.cond(hessian)
        + penalty_condition
        + abs(hessian_logdet)
        + penalty_log_scale
    )
    scale_radius = np.linalg.norm(matrices[1], 2) * radius
    assert np.ptp(eta[:, 1]) > 4 * scale_radius
    assert error + mode_error < 1e-5
    return _IndependentMode(
        float(objective),
        error,
        mode_error,
        (fit.terminal_rank.method, p, "observed", tuple(groups)),
        matrices,
    )


def _derivatives(model, *, chunk=53, want_hessian=True, reuse=None):
    fitted = model._require_fitted()
    state = fitted.fit_state
    rows = state.retained_rows
    plan = fitted.family.bind_likelihood(
        rows.response, rows.likelihood_weights, COMPLETE_OBSERVATION
    )
    return laml_derivatives(
        fitted.family,
        state.layout,
        rows.response,
        plan,
        lambdas=state.lambdas,
        fit=state.solver_result,
        dense_matrices=bounded_predictor_matrices(state.layout, chunk_size=chunk),
        want_hessian=want_hessian,
        reuse=reuse,
    )


@pytest.mark.parametrize("kind", ["gaussian", "gamma"])
@pytest.mark.parametrize("discrete", [False, True])
def test_public_varying_scale_shared_penalty_profile(kind, discrete):
    """Every profile mode enters through SuperLSS.fit, including all stencil probes."""
    for rho_left in (-0.4, 0.0, 0.4):
        for rho_right in (-0.4, 0.0, 0.4):
            rho = np.array([rho_left, rho_right])
            model = _public_fit(kind, discrete, tuple(rho))
            oracle = _independent_mode(model, kind)
            state = model._require_fitted().fit_state
            fit = state.solver_result
            expected_backend = (
                "distributional-chunked-v1" if discrete else "distributional-dense-v1"
            )
            assert fit.execution_backend_identifier == expected_backend
            assert (fit.resolved_chunk_size is not None) == discrete
            production = joint_laplace_objective(fit, layout=state.layout, lambdas=state.lambdas)
            assert abs(production - oracle.objective) <= oracle.error
            derivatives = _derivatives(model)

            def probe(delta):
                result = _independent_mode(_public_fit(kind, discrete, tuple(rho + delta)), kind)
                assert result.provenance == oracle.provenance
                return result.objective, result.error + result.mode_error

            def stencil(direction, h, order):
                plus, e_plus = probe(h * direction)
                minus, e_minus = probe(-h * direction)
                if order == 1:
                    return (plus - minus) / (2 * h), (e_plus + e_minus) / (2 * h)
                return (
                    (plus - 2 * oracle.objective + minus) / h**2,
                    (e_plus + e_minus + 2 * (oracle.error + oracle.mode_error)) / h**2,
                )

            h = 2e-2
            for index in range(2):
                direction = np.eye(2)[index]
                for order in (1, 2):
                    coarse, coarse_error = stencil(direction, h, order)
                    fine, fine_error = stencil(direction, h / 2, order)
                    richardson = (4 * fine - coarse) / 3
                    truncation = abs(fine - coarse) / 3
                    rounding = (4 * fine_error + coarse_error) / 3
                    value = (
                        derivatives.gradient[index]
                        if order == 1
                        else derivatives.hessian[index, index]
                    )
                    certificate = (
                        derivatives.gradient_certificate[index]
                        if order == 1
                        else derivatives.hessian_certificate[index, index]
                    )
                    assert abs(value - richardson) <= truncation + rounding + certificate

            def mixed(step):
                probes = [
                    (left * right, probe(step * np.array([left, right])))
                    for left in (-1, 1)
                    for right in (-1, 1)
                ]
                return (
                    sum(sign * value for sign, (value, _) in probes) / (4 * step**2),
                    sum(error for _, (_, error) in probes) / (4 * step**2),
                )

            coarse, coarse_error = mixed(h)
            fine, fine_error = mixed(h / 2)
            expected = (4 * fine - coarse) / 3
            tolerance = (abs(fine - coarse) + 4 * fine_error + coarse_error) / 3
            assert abs(derivatives.hessian[0, 1] - expected) <= (
                tolerance + derivatives.hessian_certificate[0, 1]
            )


@pytest.mark.parametrize("kind", ["gaussian", "gamma"])
def test_public_shared_penalty_replay_units_and_coordinate_permutations(kind):
    model = _public_fit(kind, True)
    state = model._require_fitted().fit_state
    evaluation = _compute_penalty_logdet_evaluation(
        dict(state.lambdas), list(state.layout.penalties)
    )
    copied = []
    weights = dict(state.lambdas)
    for index, component in enumerate(state.layout.penalties):
        factor = 2.0 ** (32 if index % 2 else -32)
        order = np.arange(component.omega_ssp.shape[0])[::-1]
        matrix = factor * component.omega_ssp[np.ix_(order, order)]
        copied.append(replace(component, omega_ssp=matrix.copy(), omega_raw=matrix.copy()))
        weights[component.name] /= factor
    changed = _compute_penalty_logdet_evaluation(weights, copied)
    assert changed.rank == evaluation.rank
    assert abs(changed.logdet - evaluation.logdet) <= changed.logdet_error + evaluation.logdet_error
    for name in evaluation.gradient:
        assert abs(changed.gradient[name] - evaluation.gradient[name]) <= (
            changed.gradient_error[name] + evaluation.gradient_error[name]
        )
    frame, _, _, _ = _public_data(kind)
    replayed = SuperLSS.from_bytes(model.to_bytes())
    restored = replayed._require_fitted().fit_state
    actual = _compute_penalty_logdet_evaluation(
        dict(restored.lambdas), list(restored.layout.penalties)
    )
    assert actual.rank == evaluation.rank
    assert abs(actual.logdet - evaluation.logdet) <= actual.logdet_error + evaluation.logdet_error
    np.testing.assert_array_equal(
        restored.solver_result.coefficients, state.solver_result.coefficients
    )
    np.testing.assert_array_equal(
        replayed.predict_parameters(frame), model.predict_parameters(frame)
    )


@pytest.mark.parametrize("kind", ["gaussian", "gamma"])
def test_public_shared_penalty_row_order_and_chunk_boundaries(kind, monkeypatch):
    from superglm.distributional.solver import chunks

    reference = _public_fit(kind, True)
    expected = _independent_mode(reference, kind)
    frame, y, family, predictors = _public_data(kind)
    order = np.random.default_rng(924).permutation(len(y))
    real_selector = chunks._resolve_fitting_chunk_size
    observed_bounds = []

    def small_chunks(family, layout, plan, requested):
        selected = real_selector(family, layout, plan, requested)
        return None if selected is None else 37

    original_chunks = chunks.iter_row_chunks

    def record_chunks(*args, **kwargs):
        for chunk in original_chunks(*args, **kwargs):
            observed_bounds.append(chunk.stop - chunk.start)
            yield chunk

    monkeypatch.setattr(chunks, "_resolve_fitting_chunk_size", small_chunks)
    monkeypatch.setattr(chunks, "iter_row_chunks", record_chunks)
    reordered = SuperLSS(family=family, predictors=predictors, discrete=True, n_bins=512).fit(
        frame.iloc[order], y[order], lambdas=_weights(kind), max_inner_iter=150, inner_tol=1e-10
    )
    oracle = _independent_mode(reordered, kind)
    assert observed_bounds and max(observed_bounds) <= 37
    assert reordered._require_fitted().fit_state.solver_result.resolved_chunk_size == 37
    assert abs(oracle.objective - expected.objective) <= (
        oracle.error + oracle.mode_error + expected.error + expected.mode_error
    )
    first, second = _derivatives(reference, chunk=31), _derivatives(reference, chunk=67)
    np.testing.assert_array_less(
        np.abs(first.gradient - second.gradient),
        first.gradient_certificate
        + second.gradient_certificate
        + 128 * np.finfo(float).eps * np.maximum(np.abs(first.gradient), 1.0),
    )
    np.testing.assert_array_less(
        np.abs(first.hessian - second.hessian),
        first.hessian_certificate
        + second.hessian_certificate
        + 128 * np.finfo(float).eps * np.maximum(np.abs(first.hessian), 1.0),
    )


def _decimal_root_logdet(roots, weights):
    """80-digit determinant of the frozen full-support root representative."""
    with localcontext() as context:
        context.prec = 80
        width = roots[0].shape[1]
        matrix = [[Decimal(0) for _ in range(width)] for _ in range(width)]
        for root, weight in zip(roots, weights, strict=True):
            for row in root:
                values = [Decimal.from_float(float(value)) for value in row]
                for i in range(width):
                    for j in range(width):
                        matrix[i][j] += Decimal.from_float(float(weight)) * values[i] * values[j]
        logdet = Decimal(0)
        for i in range(width):
            pivot = matrix[i][i]
            assert pivot > 0
            logdet += pivot.ln()
            for j in range(i + 1, width):
                factor = matrix[j][i] / pivot
                for k in range(i + 1, width):
                    matrix[j][k] -= factor * matrix[i][k]
        return float(logdet)


@pytest.mark.parametrize("discrete", [False, True])
def test_public_high_ratio_retains_rank_or_reports_numerical_refusal(discrete):
    from superglm.reml.multi_penalty import _evaluate_penalty_support
    from superglm.reml.penalty_support import PenaltyNumericalError, _penalty_support

    weights = _weights("gaussian")
    weights["location:x#d2"] = 1e12
    model = _public_fit("gaussian", discrete, origin=tuple(weights.items()))
    state = model._require_fitted().fit_state
    components = state.layout.penalties[:2]
    support = _penalty_support([component.omega_ssp for component in components])
    assert support.rank == 5
    values = np.array([weights[component.name] for component in components])
    expected = _decimal_root_logdet(support.component_roots, values)
    try:
        result = _evaluate_penalty_support(support, values)
    except PenaltyNumericalError as exc:
        assert "certif" in str(exc) or "accuracy" in str(exc) or "error" in str(exc)
        assert state.smoothing is None
        with pytest.raises(PenaltyNumericalError):
            joint_laplace_objective(state.solver_result, layout=state.layout, lambdas=state.lambdas)
    else:
        assert result.rank == 5
        assert abs(result.logdet_s_plus - expected) <= (
            result._certificate.logdet_error + 2 * np.finfo(float).eps * abs(expected)
        )


def test_penalty_derivative_errors_are_retained_across_reused_passes(monkeypatch):
    from superglm.distributional.smoothing import derivatives as module

    model = _public_fit("gaussian", False)
    baseline = _derivatives(model)
    real_evaluate = module._compute_penalty_logdet_evaluation
    calls = 0

    def enlarged_evidence(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = real_evaluate(*args, **kwargs)
        return replace(
            result,
            gradient_error={key: value + 0.02 for key, value in result.gradient_error.items()},
            hessian_error={key: value + 0.04 for key, value in result.hessian_error.items()},
        )

    monkeypatch.setattr(module, "_compute_penalty_logdet_evaluation", enlarged_evidence)
    fitted = model._require_fitted()
    state, rows = fitted.fit_state, fitted.fit_state.retained_rows
    plan = fitted.family.bind_likelihood(
        rows.response, rows.likelihood_weights, COMPLETE_OBSERVATION
    )
    matrices = bounded_predictor_matrices(state.layout, chunk_size=53)
    workspace = module.LamlDerivativeWorkspace()
    kwargs = dict(
        lambdas=state.lambdas, fit=state.solver_result, dense_matrices=matrices, reuse=workspace
    )
    gradient = module.laml_derivatives(
        fitted.family, state.layout, rows.response, plan, want_hessian=False, **kwargs
    )
    complete = module.laml_derivatives(
        fitted.family, state.layout, rows.response, plan, want_hessian=True, **kwargs
    )
    assert calls == 1
    np.testing.assert_array_equal(complete.gradient, baseline.gradient)
    np.testing.assert_array_equal(complete.hessian, baseline.hessian)
    np.testing.assert_allclose(
        gradient.gradient_certificate - baseline.gradient_certificate,
        0.01,
        rtol=0,
        atol=4 * np.finfo(float).eps,
    )
    for i in range(4):
        for j in range(4):
            expected = 0.02 if i // 2 == j // 2 else 0.0
            assert complete.hessian_certificate[i, j] - baseline.hessian_certificate[i, j] == (
                pytest.approx(expected, abs=4 * np.finfo(float).eps)
            )


def test_numerical_penalty_refusal_preserves_derivative_cause(monkeypatch):
    from superglm.distributional.smoothing import derivatives as module
    from superglm.reml.penalty_support import PenaltyNumericalError

    model = _public_fit("gaussian", False)
    failure = PenaltyNumericalError("reference action cannot meet accuracy")

    def refuse(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(module, "_compute_penalty_logdet_evaluation", refuse)
    with pytest.raises(module.LamlDerivativeError, match="penalty geometry") as caught:
        _derivatives(model)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("outer", ["efs", "efs+newton"])
def test_public_reml_shared_penalty_mode_has_an_independent_local_profile(outer):
    frame, _, family, predictors = _public_data("gaussian")
    x = frame["x"].to_numpy()
    # Curvature in both predictors keeps this automatic smoothing fit in a
    # moderate regime. The linear-truth profile above has a near-infinite
    # second-difference optimum, unsuitable for coefficient-forward checks.
    y = (
        0.4
        + 0.8 * x
        + 0.8 * np.sin(np.pi * x)
        + np.exp(-0.4 + 0.35 * x + 0.3 * np.cos(np.pi * x))
        * np.random.default_rng(248).normal(size=x.size)
    )
    model = SuperLSS(family=family, predictors=predictors).fit_reml(
        frame,
        y,
        lambdas=_weights("gaussian"),
        max_inner_iter=150,
        inner_tol=1e-10,
        max_reml_iter=100,
        practical_reml=False,
        outer=outer,
    )
    state = model._require_fitted().fit_state
    oracle = _independent_mode(model, "gaussian")
    derivatives = _derivatives(model)
    production = joint_laplace_objective(
        state.solver_result, layout=state.layout, lambdas=state.lambdas
    )
    assert abs(production - oracle.objective) <= oracle.error

    def profile(delta):
        weights = dict(state.lambdas)
        for name, step in zip(tuple(weights)[:2], delta, strict=True):
            weights[name] *= math.exp(step)
        fitted = SuperLSS(family=family, predictors=predictors).fit(
            frame, y, lambdas=weights, max_inner_iter=150, inner_tol=1e-10
        )
        return _independent_mode(fitted, "gaussian")

    for index in range(2):
        values, errors = [], []
        for step in (2e-2, 1e-2):
            direction = np.eye(2)[index] * step
            plus = profile(direction)
            minus = profile(-direction)
            assert plus.provenance == minus.provenance == oracle.provenance
            values.append((plus.objective - minus.objective) / (2 * step))
            errors.append(
                (plus.error + plus.mode_error + minus.error + minus.mode_error) / (2 * step)
            )
        expected = (4 * values[1] - values[0]) / 3
        tolerance = (abs(values[1] - values[0]) + 4 * errors[1] + errors[0]) / 3
        assert (
            abs(derivatives.gradient[index] - expected)
            <= tolerance + derivatives.gradient_certificate[index]
        )
