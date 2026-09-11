"""Public varying-scale fits, units, replay and shared-penalty LAML evidence.

The likelihood derivatives below follow directly from the Gaussian density and
the Gamma density with shape w / scale**2. They do not call family evaluators.
Wood (2011, section 3.1 and Appendix B) and Wood, Pya and Safken (2016,
sections 3.1.1--3.1.2) motivate fixed penalty support and observed terminal
curvature. The arithmetic bounds and refit checks here are our test policy.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import digamma, polygamma

from superglm import GammaLS, GaussianLS, Predictor, Spline, SuperLSS, TensorInteraction
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.smoothing.derivatives import laml_derivatives
from superglm.distributional.smoothing.objective import joint_laplace_objective
from superglm.distributional.solver.solver import fit_dense_fixed_lambda
from tests.bound_predictor_fixtures import model_from_templates

_N = 480
_EPS = np.finfo(float).eps


def _gamma(count):
    """Accumulation bound gamma_k, including explicitly counted work."""
    return count * _EPS / (1.0 - count * _EPS)


@functools.cache
def _data():
    rng = np.random.default_rng(2026091001)
    x = rng.permutation(np.linspace(0.0, 1.0, _N))
    z = rng.permutation(np.linspace(0.0, 1.0, _N))
    weights = np.exp(rng.uniform(-0.4, 0.4, _N))
    location = 0.8 * np.sin(2 * np.pi * x) + 0.3 * np.cos(2 * np.pi * z)
    scale = np.exp(-0.5 + 0.35 * np.sin(2 * np.pi * z))
    gaussian = location + scale / np.sqrt(weights) * rng.standard_normal(_N)
    mean = np.exp(0.5 + 0.5 * location)
    shape = weights / scale**2
    gamma = rng.gamma(shape, mean / shape)
    return pd.DataFrame({"x": x, "z": z}), weights, {"gaussian": gaussian, "gamma": gamma}


def _predictors(kind, *, shared=False):
    names = ("location", "scale") if kind == "gaussian" else ("mean", "scale")
    return [
        Predictor(
            name,
            {
                key: Spline(n_knots=4) if shared else Spline(kind="cr", n_knots=6)
                for key in ("x", "z")
            },
            interaction_specs=(
                {"x:z": TensorInteraction("x", "z", n_knots=(4, 4))} if shared else {}
            ),
        )
        for name in names
    ]


@functools.cache
def _fit(kind, smoothing=False, discrete=False, transform="original", shared=False):
    frame, weights, responses = _data()
    frame, y, weights = frame.copy(), responses[kind].copy(), weights.copy()
    if transform == "feature_units":
        frame *= 10.0
    elif transform == "response_units":
        y *= 10.0
    elif transform == "permutation":
        order = np.random.default_rng(71023).permutation(_N)
        frame, y, weights = frame.iloc[order].reset_index(drop=True), y[order], weights[order]
    family = GaussianLS(scale_floor=0.0) if kind == "gaussian" else GammaLS()
    model = model_from_templates(
        family=family,
        predictors=_predictors(kind, shared=shared),
        discrete=discrete,
        n_bins=_N,
    )
    options = dict(
        sample_weight=weights, max_inner_iter=200, inner_tol=1.0e-11 if shared else 1.0e-9
    )
    if smoothing:
        model.fit_reml(
            frame,
            y,
            **options,
            max_reml_iter=100,
            outer="efs+newton",
            practical_reml=False,
        )
    else:
        names = ("location", "scale") if kind == "gaussian" else ("mean", "scale")
        lambdas = {f"{name}:{key}#wiggle": 1.0 for name in names for key in ("x", "z")}
        if shared:
            for name in names:
                lambdas[f"{name}:x:z#margin_x"] = 0.7
                lambdas[f"{name}:x:z#margin_z"] = 1.3
        if transform == "feature_units":
            lambdas = {name: value * 1000.0 for name, value in lambdas.items()}
        elif transform == "response_units" and kind == "gaussian":
            lambdas = {
                name: value / 100.0 if name.startswith("location:") else value
                for name, value in lambdas.items()
            }
        model.fit(frame, y, lambdas=lambdas, **options)
    return model, frame, y, weights


def _matrices(layout):
    # Independent assembly of the represented design, including each intercept.
    return tuple(
        np.column_stack((np.ones(state.design.n), state.design.toarray()))
        for state in layout.predictors
    )


def _rows(kind, y, weights, theta):
    mean, scale = theta.T
    a = weights / scale**2
    if kind == "gaussian":
        d = y - mean
        score = np.column_stack((a * d, a * d**2 - 1.0))
        curvature = (a, 2.0 * a * d, 2.0 * a * d**2)
        absolute_d = abs(y) + abs(mean)
        score_magnitude = np.column_stack((a * absolute_d, a * absolute_d**2 + 1))
        curvature_magnitude = (a, 2 * a * absolute_d, 2 * a * absolute_d**2)
    else:
        ratio = y / mean
        b = np.log(a) + 1.0 - digamma(a) + np.log(ratio) - ratio
        absolute_b = abs(np.log(a)) + 1 + abs(digamma(a)) + abs(np.log(ratio)) + ratio
        score = np.column_stack((a * (ratio - 1.0), -2.0 * a * b))
        curvature = (a * ratio, 2.0 * a * (ratio - 1.0), 4.0 * a * (a * polygamma(1, a) - 1 - b))
        score_magnitude = np.column_stack((a * (ratio + 1), 2 * a * absolute_b))
        curvature_magnitude = (
            a * ratio,
            2 * a * (ratio + 1),
            4 * a * (a * polygamma(1, a) + 1 + absolute_b),
        )
    return score, curvature, score_magnitude, curvature_magnitude


@dataclass
class _Evidence:
    score: np.ndarray
    curvature: np.ndarray
    score_error: np.ndarray
    curvature_error: float
    lower_eigenvalue: float
    matrices: tuple


def _spectrum(matrix, rank):
    """Refuse unresolved support; certify eigenpairs by residual and orthogonality.

    For a symmetric represented matrix, Weyl's absolute perturbation bound
    applies. The residual, loss of orthogonality and gamma_(16p) assembly
    allowance bound the eigenvalue error; no relative-small-eigenvalue claim
    follows from backward stability alone.
    """
    values, vectors = np.linalg.eigh(matrix)
    norm = np.linalg.norm(matrix, 2)
    p = len(matrix)
    residual = np.linalg.norm(matrix @ vectors - vectors * values, 2)
    residual += _gamma(p + 2) * np.linalg.norm(
        abs(matrix) @ abs(vectors) + abs(vectors) * abs(values), 2
    )
    orthogonality = np.linalg.norm(vectors.T @ vectors - np.eye(p), 2)
    orthogonality += _gamma(p + 2) * np.linalg.norm(abs(vectors).T @ abs(vectors) + np.eye(p), 2)
    assert orthogonality < 0.5
    polar_error = orthogonality / (1 + np.sqrt(1 - orthogonality))
    # Compare A first with V diag(d) V.T and then with its orthogonal polar
    # counterpart U diag(d) U.T, whose eigenvalues are exactly d.
    error = residual * np.sqrt(1 + orthogonality) + norm * orthogonality
    error += max(abs(values)) * polar_error * (2 + polar_error) + _gamma(16 * p) * norm
    assert np.max(np.abs(values[: p - rank]), initial=0.0) <= error
    positive = values[p - rank :]
    assert len(positive) == rank and positive[0] > 2.0 * error, "unresolved spectral support"
    logdet = float(np.log(positive).sum())
    log_error = (
        float(np.sum(-np.log1p(-error / positive))) + _gamma(p) * np.abs(np.log(positive)).sum()
    )
    return logdet, log_error, positive[0] - error


def _evidence(kind, layout, fit, y, weights):
    matrices = _matrices(layout)
    row_score, row_curvature, row_magnitude, curvature_magnitude = _rows(
        kind, y, weights, fit.theta
    )
    score = np.concatenate([x.T @ row_score[:, k] for k, x in enumerate(matrices)])
    magnitude = np.concatenate([abs(x).T @ abs(row_score[:, k]) for k, x in enumerate(matrices)])
    score -= fit.penalty @ fit.coefficients
    magnitude += abs(fit.penalty) @ abs(fit.coefficients)
    x, z = matrices
    a, b, c = row_curvature
    data = np.block(
        [
            [x.T @ (a[:, None] * x), x.T @ (b[:, None] * z)],
            [z.T @ (b[:, None] * x), z.T @ (c[:, None] * z)],
        ]
    )
    absolute = np.block(
        [
            [abs(x).T @ (abs(a[:, None]) * abs(x)), abs(x).T @ (abs(b[:, None]) * abs(z))],
            [abs(z).T @ (abs(b[:, None]) * abs(x)), abs(z).T @ (abs(c[:, None]) * abs(z))],
        ]
    )
    row_error = np.concatenate([abs(x).T @ row_magnitude[:, k] for k, x in enumerate(matrices)])
    aa, bb, cc = curvature_magnitude
    row_curvature_error = np.block(
        [
            [abs(x).T @ (aa[:, None] * abs(x)), abs(x).T @ (bb[:, None] * abs(z))],
            [abs(z).T @ (bb[:, None] * abs(x)), abs(z).T @ (cc[:, None] * abs(z))],
        ]
    )
    # Separate n-row accumulation from row evaluation. The latter uses the
    # magnitudes BEFORE cancellation, with a 64-operation allowance for the
    # density/link expressions (including the special-function evaluations).
    arithmetic = _gamma(len(y) + len(score) + 4)
    score_error = arithmetic * (1.0 + magnitude) + _gamma(64) * row_error
    curvature_error = arithmetic * np.linalg.norm(absolute + abs(fit.penalty), 2)
    curvature_error += _gamma(64) * np.linalg.norm(row_curvature_error, 2)
    curvature = (data + data.T) / 2.0 + fit.penalty
    assert np.linalg.norm(fit.terminal_score - score, np.inf) <= np.max(score_error)
    assert np.linalg.norm(fit.terminal_data_curvature - data, 2) <= curvature_error
    assert np.linalg.norm(fit.terminal_penalized_curvature - curvature, 2) <= curvature_error
    _, _, lower = _spectrum(curvature, len(score))
    assert lower > 2 * curvature_error
    return _Evidence(score, curvature, score_error, curvature_error, lower, matrices)


def _successful(case, kind):
    model, frame, y, weights = case
    state = model._require_fitted().fit_state
    fit = state.solver_result
    diagnostic = model.diagnose()
    assert fit.converged, diagnostic.render(detail="full")
    theta = model.predict_parameters(frame).to_numpy()
    assert theta.shape == (_N, 2) and np.all(np.isfinite(theta))
    assert np.all(theta[:, 1] > 0)
    if kind == "gamma":
        assert np.all(theta[:, 0] > 0)
    prediction_bound = (
        _gamma(32 * len(fit.coefficients))
        * (1 + np.linalg.norm(fit.coefficients))
        * (1 + abs(fit.theta))
    )
    assert np.all(abs(theta - fit.theta) <= prediction_bound)
    assert state.layout.predictor("scale").design.p > 0
    assert np.ptp(theta[:, 1]) > np.sqrt(_EPS) * np.linalg.norm(theta[:, 1], np.inf)
    assert fit.coefficient_face is None
    assert fit.terminal_curvature.actual_source == "observed"
    assert fit.terminal_rank.rank == len(fit.coefficients) == model.result_.rank
    evidence = _evidence(kind, state.layout, fit, y, weights)
    denominator = 1.0 + abs(fit.penalized_optimizing_log_likelihood)
    if np.linalg.norm(evidence.score, np.inf) > (
        fit.config.tolerance * denominator + np.max(evidence.score_error)
    ):
        # A resolution-limited stop needs an independent improvement bound,
        # rather than a relaxed score tolerance.
        assert kind == "gaussian"
        improvement, _ = _gaussian_refit_error(state.layout, fit, y, weights)
        assert improvement <= np.spacing(abs(fit.penalized_optimizing_log_likelihood))
    assert model.result_.coefficient_converged
    if state.smoothing is not None:
        smoothing = state.smoothing
        assert smoothing.converged, diagnostic.render(detail="full")
        assert smoothing.terminal_fit is fit
        assert smoothing.convergence_reason == "stationary"
        assert smoothing.terminal_projected_gradient_norm <= smoothing.stationarity_bar
        assert all(
            v <= smoothing.stationarity_bar
            for v in smoothing.terminal_gradient_certificate.values()
        )
        assert dict(smoothing.lambdas) == dict(model.result_.smoothing_parameters)
        assert model.result_.smoothing_converged
    return evidence


@pytest.mark.parametrize("kind", ["gaussian", "gamma"])
@pytest.mark.parametrize("smoothing", [False, True], ids=["fixed", "strict_reml"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "exact_discrete"])
def test_public_varying_scale_fit_and_replay(kind, smoothing, discrete):
    """Catches stale success/rank evidence, discarded scale terms and backend changes."""
    case = _fit(kind, smoothing, discrete)
    _successful(case, kind)
    model, frame, _, _ = case
    fit = model._require_fitted().fit_state.solver_result
    assert len(fit.coefficients) == 30
    assert (fit.resolved_chunk_size is not None) == discrete
    assert fit.execution_backend_identifier == (
        "distributional-chunked-v1" if discrete else "distributional-dense-v1"
    )
    restored = SuperLSS.from_bytes(model.to_bytes())
    replay = restored._require_fitted().fit_state.solver_result
    np.testing.assert_array_equal(
        restored.predict_parameters(frame), model.predict_parameters(frame)
    )
    assert replay.terminal_rank.policy_version == fit.terminal_rank.policy_version
    assert replay.terminal_rank.rank == fit.terminal_rank.rank
    assert replay.execution_backend_identifier == fit.execution_backend_identifier
    assert replay.resolved_chunk_size == fit.resolved_chunk_size
    assert restored.result_.coefficient_converged == model.result_.coefficient_converged
    assert restored.result_.smoothing_converged == model.result_.smoothing_converged
    assert dict(restored.result_.smoothing_parameters) == dict(model.result_.smoothing_parameters)
    if smoothing:
        assert dict(restored._require_fitted().smoothing.initial_lambdas) == dict(
            model._require_fitted().smoothing.initial_lambdas
        )


def _curvature_lipschitz(kind, y, weights, theta, matrices, radius):
    """Bound ||H(beta+d)-H(beta)||_2 / ||d||_2 in the coefficient ball.

    The squared Frobenius norm of the two predictor rows bounds the embedding
    operator norm squared. Bound its two partial derivatives throughout the
    ball, then sum their embedded norms. Polygamma magnitudes decrease with
    positive shape, so the lower endpoint bounds both Gamma derivatives.
    """
    xnorm, znorm = (np.linalg.norm(x, axis=1) for x in matrices)
    mean, scale = theta.T
    a = weights / scale**2
    upper_a = a * np.exp(2 * znorm * radius)
    if kind == "gaussian":
        d = abs(y - mean) + xnorm * radius
        du = upper_a * np.sqrt(8.0 + 16.0 * d**2)
        dv = 2 * upper_a * np.sqrt(1.0 + 8.0 * d**2 + 4.0 * d**4)
    else:
        lower_a = a * np.exp(-2 * znorm * radius)
        ratio = y / mean
        upper_ratio = ratio * np.exp(xnorm * radius)
        b = (
            abs(np.log(a))
            + 2 * znorm * radius
            + 1
            + np.maximum(abs(digamma(lower_a)), abs(digamma(upper_a)))
            + abs(np.log(ratio))
            + xnorm * radius
            + upper_ratio
        )
        du = upper_a * np.sqrt(9 * upper_ratio**2 + 16 * (upper_ratio + 1) ** 2)
        last = (
            8
            * upper_a
            * (
                3 * upper_a * polygamma(1, lower_a)
                + upper_a**2 * abs(polygamma(2, lower_a))
                + 2
                + b
            )
        )
        dv = np.sqrt(
            (2 * upper_a * upper_ratio) ** 2 + 2 * (4 * upper_a * (upper_ratio + 1)) ** 2 + last**2
        )
    return float(np.sum((xnorm**2 + znorm**2) * (xnorm * du + znorm * dv)))


@dataclass
class _CoefficientBall:
    radius: float
    matrices: tuple
    lower: float
    geometry_error: float
    score: np.ndarray
    score_bound: float


def _coefficient_ball(kind, fit, y, weights, evidence):
    width = len(fit.coefficients)
    root = np.linalg.solve(np.linalg.cholesky(evidence.curvature).T, np.eye(width))
    x, z = evidence.matrices
    p = x.shape[1]
    u, v = x @ root[:p], z @ root[p:]
    # Certify the transformed solve against the represented H, including its
    # assembly error. Conditioning alone would not certify the score.
    geometry_error = np.linalg.norm(root.T @ evidence.curvature @ root - np.eye(width), 2)
    geometry_error += evidence.curvature_error * np.linalg.norm(root, 2) ** 2
    lower = 1 - geometry_error
    assert lower > 0
    score = root.T @ evidence.score
    score_error = abs(root.T) @ evidence.score_error
    score_bound = np.linalg.norm(score) + np.linalg.norm(score_error)
    radius = 2 * score_bound / lower
    # The frozen-Hessian Newton map contracts this ball. This certifies a
    # local error radius in the observed-Hessian metric, not global uniqueness.
    change = radius * _curvature_lipschitz(kind, y, weights, fit.theta, (u, v), radius)
    assert change < lower / 2, "coefficient error ball has unresolved curvature"
    return _CoefficientBall(radius, (u, v), lower - change, geometry_error, score, score_bound)


def _derivatives(model, fit=None, lambdas=None):
    fitted = model._require_fitted()
    state = fitted.fit_state
    rows = state.retained_rows
    plan = fitted.family.bind_likelihood(
        rows.response, rows.likelihood_weights, COMPLETE_OBSERVATION
    )
    fit = state.solver_result if fit is None else fit
    lambdas = dict(model.result_.smoothing_parameters) if lambdas is None else lambdas
    return laml_derivatives(
        fitted.family,
        state.layout,
        rows.response,
        plan,
        lambdas=lambdas,
        fit=fit,
        dense_matrices=_matrices(state.layout),
    )


def _smoothing_radius(model):
    smoothing = model._require_fitted().smoothing
    fresh = _derivatives(model)
    recorded = np.array([smoothing.terminal_gradient[name] for name in fresh.names])
    recorded_error = np.array(
        [smoothing.terminal_gradient_certificate[name] for name in fresh.names]
    )
    error = fresh.gradient_certificate + recorded_error + _gamma(32 * len(fresh.names))
    assert np.all(abs(fresh.gradient - recorded) <= error), "stale terminal smoothing evidence"
    assert np.linalg.norm(fresh.gradient, np.inf) <= smoothing.stationarity_bar + max(error)
    _, _, lower = _spectrum(fresh.hessian, len(fresh.names))
    lower -= np.linalg.norm(fresh.hessian_certificate, 2)
    assert lower > 0, "this fixture no longer identifies its smoothing profile"
    # A local observed-Hessian error estimate for the resolved smoothing
    # profile, including the derivative stencil's reported uncertainty.
    radius = (
        2 * (np.linalg.norm(fresh.gradient) + np.linalg.norm(fresh.gradient_certificate)) / lower
    )
    return radius, fresh.names


def _prediction_error(case, kind, evidence, *, smoothing_radius=0.0):
    model, _, y, weights = case
    fit = model._require_fitted().fit_state.solver_result
    ball = _coefficient_ball(kind, fit, y, weights, evidence)
    eta_errors = np.column_stack(
        [np.linalg.norm(matrix, axis=1) * ball.radius for matrix in ball.matrices]
    )
    if smoothing_radius:
        layout = model._require_fitted().layout
        rhs = []
        for component in layout.penalties:
            direction = np.zeros_like(fit.coefficients)
            direction[component.group_sl] = (
                model.result_.smoothing_parameters[component.name]
                * component.omega_ssp
                @ fit.coefficients[component.group_sl]
            )
            rhs.append(direction)
        sensitivity = np.linalg.solve(evidence.curvature, np.column_stack(rhs))
        for k, state in enumerate(layout.predictors):
            response = evidence.matrices[k] @ sensitivity[state.coefficient_slice]
            eta_errors[:, k] += 2 * np.linalg.norm(response, axis=1) * smoothing_radius
    columns = []
    for k in range(2):
        eta_error = eta_errors[:, k]
        if kind == "gaussian" and k == 0:
            columns.append(eta_error)
        else:
            columns.append(fit.theta[:, k] * np.expm1(eta_error))
    return np.column_stack(columns)


@pytest.mark.parametrize("kind", ["gaussian", "gamma"])
@pytest.mark.parametrize("smoothing", [False, True], ids=["fixed", "strict_reml"])
@pytest.mark.parametrize(
    "transform", ["feature_units", "response_units", "permutation", "exact_discrete"]
)
def test_varying_scale_fit_is_invariant_to_units_rows_and_execution(kind, smoothing, transform):
    """Catches wrong penalty units, order-sensitive fits and inexact binning."""
    base = _fit(kind, smoothing)
    changed = (
        _fit(kind, smoothing, True)
        if transform == "exact_discrete"
        else _fit(kind, smoothing, transform=transform)
    )
    left, right = _successful(base, kind), _successful(changed, kind)
    radii = (0.0, 0.0)
    if smoothing:
        starts = [case[0]._require_fitted().smoothing.initial_lambdas for case in (base, changed)]
        mapping = {
            name: 1000.0
            if transform == "feature_units"
            else 0.01
            if transform == "response_units" and name.startswith("location:")
            else 1.0
            for name in starts[0]
        }
        # The start is compared before terminal smoothing/prediction output.
        for name, multiplier in mapping.items():
            assert starts[1][name] == pytest.approx(
                multiplier * starts[0][name], rel=_gamma(4096 * len(left.score))
            )
        resolved = [_smoothing_radius(case[0]) for case in (base, changed)]
        radii = (resolved[0][0], resolved[1][0])
        assert resolved[0][1] == resolved[1][1]
        names = resolved[0][1]
        rho_delta = np.array(
            [
                np.log(
                    changed[0].result_.smoothing_parameters[name]
                    / (mapping[name] * base[0].result_.smoothing_parameters[name])
                )
                for name in names
            ]
        )
        assert np.linalg.norm(rho_delta) <= sum(radii) + _gamma(4096 * len(left.score))
    error = _prediction_error(base, kind, left, smoothing_radius=radii[0])
    changed_error = _prediction_error(changed, kind, right, smoothing_radius=radii[1])
    expected = base[0].predict_parameters(base[1]).to_numpy()
    actual = changed[0].predict_parameters(changed[1]).to_numpy()
    if transform == "permutation":
        undo = np.argsort(np.random.default_rng(71023).permutation(_N))
        actual, changed_error = actual[undo], changed_error[undo]
    if transform == "response_units":
        units = np.array([10.0, 10.0 if kind == "gaussian" else 1.0])
        actual, changed_error = actual / units, changed_error / units
    arithmetic = _gamma(32 * (_N + len(left.score))) * (1 + abs(expected) + abs(actual))
    assert np.all(abs(actual - expected) <= error + changed_error + arithmetic)


@pytest.mark.parametrize("kind", ["gaussian", "gamma"])
@pytest.mark.parametrize("chunk_size", [17, 37])
def test_explicit_chunk_boundaries_preserve_the_represented_fit(kind, chunk_size):
    """Catches omitted/repeated boundary rows on the same design and weights."""
    case = _fit(kind)
    model, _, y, weights = case
    evidence = _successful(case, kind)
    fitted = model._require_fitted()
    base = fitted.fit_state.solver_result
    rows = fitted.fit_state.retained_rows
    plan = fitted.family.bind_likelihood(y, rows.likelihood_weights, COMPLETE_OBSERVATION)
    chunked = fit_dense_fixed_lambda(
        fitted.family,
        fitted.layout,
        y,
        plan,
        base.penalty,
        config=base.config,
        chunk_size=chunk_size,
    )
    assert chunked.converged
    assert chunked.resolved_chunk_size == chunk_size
    assert chunked.execution_backend_identifier == "distributional-chunked-v1"
    assert chunked.terminal_curvature.actual_source == "observed"
    assert chunked.terminal_rank.rank == base.terminal_rank.rank == 30
    other = _evidence(kind, fitted.layout, chunked, y, weights)
    left = _coefficient_ball(kind, base, y, weights, evidence)
    right = _coefficient_ball(kind, chunked, y, weights, other)
    for k in range(2):
        eta_bound = np.linalg.norm(left.matrices[k], axis=1) * left.radius
        eta_bound += np.linalg.norm(right.matrices[k], axis=1) * right.radius
        bound = (
            eta_bound if kind == "gaussian" and k == 0 else base.theta[:, k] * np.expm1(eta_bound)
        )
        assert np.all(abs(chunked.theta[:, k] - base.theta[:, k]) <= bound)


def _independent_penalty(layout, lambdas):
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients))
    for component in layout.penalties:
        assert component.omega_ssp is not None
        penalty[component.group_sl, component.group_sl] += (
            lambdas[component.name] * component.omega_ssp
        )
    return penalty


def _penalty_support(layout):
    balanced = np.zeros((layout.n_coefficients, layout.n_coefficients))
    for component in layout.penalties:
        omega = component.omega_ssp
        balanced[component.group_sl, component.group_sl] += omega / np.linalg.norm(omega, 2)
    values = np.linalg.eigvalsh(balanced)
    error = _gamma(16 * len(balanced)) * np.linalg.norm(balanced, 2)
    rank = np.count_nonzero(values > 2 * error)
    _spectrum(balanced, rank)
    return rank


def _independent_objective(layout, fit, lambdas, support):
    penalty = _independent_penalty(layout, lambdas)
    log_s, error_s, _ = _spectrum(penalty, support)
    log_h, error_h, _ = _spectrum(fit.terminal_penalized_curvature, len(fit.coefficients))
    value = -fit.penalized_optimizing_log_likelihood + 0.5 * (log_h - log_s)
    error = 0.5 * (error_h + error_s) + _gamma(8 * len(fit.coefficients)) * (
        1 + abs(fit.penalized_optimizing_log_likelihood) + abs(log_h) + abs(log_s)
    )
    return value, error


def _gaussian_refit_error(layout, fit, y, weights):
    evidence = _evidence("gaussian", layout, fit, y, weights)
    x, z = evidence.matrices
    p = x.shape[1]
    width = len(fit.coefficients)
    root = np.linalg.solve(np.linalg.cholesky(evidence.curvature).T, np.eye(width))
    # Use the observed-Hessian metric: harmless coefficient units must not
    # inflate an error bound for a stable fitted observable.
    u, v = x @ root[:p], z @ root[p:]
    geometry_error = np.linalg.norm(root.T @ evidence.curvature @ root - np.eye(width), 2)
    geometry_error += evidence.curvature_error * np.linalg.norm(root, 2) ** 2
    lower = 1 - geometry_error
    assert lower > 0
    score = root.T @ evidence.score
    score_error = abs(root.T) @ evidence.score_error
    score_bound = np.linalg.norm(score) + np.linalg.norm(score_error)
    radius = 2 * score_bound / lower
    xx, xz, zz = np.sum(u**2, axis=1), np.sum(u * v, axis=1), np.sum(v**2, axis=1)
    a = weights / fit.theta[:, 1] ** 2
    d = y - fit.theta[:, 0]
    trace_mu = -4 * a * xz - 4 * a * d * zz
    trace_scale = -2 * a * xx - 8 * a * d * xz - 4 * a * d**2 * zz
    gradient = -score + 0.5 * (u.T @ trace_mu + v.T @ trace_scale)
    xn, zn = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    upper_a = a * np.exp(2 * zn * radius)
    upper_d = abs(d) + xn * radius
    du = upper_a * np.sqrt(8 + 16 * upper_d**2)
    dv = 2 * upper_a * np.sqrt(1 + 8 * upper_d**2 + 4 * upper_d**4)
    # Whitened predictor rows share coordinates. Their squared Frobenius norm
    # bounds the observation's embedding operator norm.
    operator_squared = xn**2 + zn**2
    lipschitz = np.sum(operator_squared * (xn * du + zn * dv))
    second = np.sum(operator_squared * (4 * upper_a * xn**2 + 4 * du * xn * zn + 2 * dv * zn**2))
    change = lipschitz * radius
    assert change < lower / 2, "unresolved refit error ball"
    lower -= change
    # d log|H| = tr(H^-1 dH). The second derivative has the two terms
    # tr(H^-1 d2H) and -tr(H^-1 dH H^-1 dH); their norm bounds control
    # the Taylor remainder as the refitted coefficients approach their mode.
    hessian_bound = 1 + geometry_error + change
    hessian_bound += 0.5 * width * ((lipschitz / lower) ** 2 + second / lower)
    improvement = score_bound**2 / (2 * lower)
    uncertainty = np.linalg.norm(gradient) * radius + 0.5 * hessian_bound * radius**2
    return float(improvement), float(uncertainty)


def _shared_case():
    case = _fit("gaussian", shared=True)
    _successful(case, "gaussian")
    layout = case[0]._require_fitted().layout
    for name in ("location", "scale"):
        components = [c for c in layout.penalties if c.name.startswith(f"{name}:x:z#")]
        assert {c.name for c in components} == {f"{name}:x:z#margin_x", f"{name}:x:z#margin_z"}
        assert len(components) == 2
        assert components[0].group_sl == components[1].group_sl
        assert components[0].group_sl.stop - components[0].group_sl.start > 1
    return case


def test_shared_varying_scale_objective_preserves_positive_penalty_support(monkeypatch):
    """Kills omission of a positive determinant direction on shared coefficients."""
    from superglm.distributional.smoothing import objective as objective_module

    case = _shared_case()
    model = case[0]
    layout = model._require_fitted().layout
    fit = model._require_fitted().fit_state.solver_result
    lambdas = dict(model.result_.smoothing_parameters)
    support = _penalty_support(layout)
    expected, error = _independent_objective(layout, fit, lambdas, support)
    actual = joint_laplace_objective(fit, layout=layout, lambdas=lambdas)
    assert abs(actual - expected) <= error

    # Remove a resolved positive eigen-direction from one interaction block.
    sl = next(c.group_sl for c in layout.penalties if c.name == "scale:x:z#margin_x")
    penalty = _independent_penalty(layout, lambdas)
    values, vectors = np.linalg.eigh(penalty[sl, sl])
    lost_logdet = np.log(values[-1])
    damaged = penalty.copy()
    damaged[sl, sl] -= values[-1] * np.outer(vectors[:, -1], vectors[:, -1])
    with pytest.raises(AssertionError, match="unresolved spectral support"):
        _spectrum(damaged, support)
    original = objective_module._compute_penalty_logdet_evaluation

    def drop_direction(*args):
        result = original(*args)
        return replace(result, logdet=result.logdet - lost_logdet)

    with monkeypatch.context() as patch:
        patch.setattr(objective_module, "_compute_penalty_logdet_evaluation", drop_direction)
        mutated = joint_laplace_objective(fit, layout=layout, lambdas=lambdas)
        with pytest.raises(AssertionError):
            assert abs(mutated - expected) <= error


def _ladder(values, errors):
    coarse = abs(values[1] - values[0])
    fine = abs(values[2] - values[1])
    # Refuse a ladder that has not entered second-order convergence, unless
    # its differences are already covered by refit/arithmetic uncertainty.
    assert fine <= 0.5 * coarse + sum(errors), "unresolved difference ladder"
    estimate = (4 * values[2] - values[1]) / 3
    uncertainty = 2 * fine + (4 * errors[2] + errors[1]) / 3
    return estimate, uncertainty


def test_shared_varying_scale_laml_derivatives_match_refitted_objective():
    """Catches missing shared-block terms in first and mixed log-lambda derivatives."""
    case = _shared_case()
    model, _, y, weights = case
    fitted = model._require_fitted()
    layout, base = fitted.layout, fitted.fit_state.solver_result
    lambdas = dict(model.result_.smoothing_parameters)
    support = _penalty_support(layout)
    derivatives = _derivatives(model)
    rows = fitted.fit_state.retained_rows
    plan = fitted.family.bind_likelihood(y, rows.likelihood_weights, COMPLETE_OBSERVATION)
    provenance = (
        base.terminal_rank.rank,
        base.terminal_rank.method,
        base.terminal_rank.policy_version,
        base.terminal_curvature.actual_source,
    )

    @functools.cache
    def probe(offsets):
        values = {name: lambdas[name] * np.exp(shift) for name, shift in offsets}
        fit = fit_dense_fixed_lambda(
            fitted.family,
            layout,
            y,
            plan,
            layout.penalty_matrix(values),
            initial=base.coefficients,
            config=replace(base.config, tolerance=1.0e-11),
        )
        assert fit.converged, "unresolved objective refit"
        assert fit.coefficient_face is None
        assert (
            fit.terminal_rank.rank,
            fit.terminal_rank.method,
            fit.terminal_rank.policy_version,
            fit.terminal_curvature.actual_source,
        ) == provenance
        value, arithmetic = _independent_objective(layout, fit, values, support)
        return value, arithmetic + _gaussian_refit_error(layout, fit, y, weights)[1]

    def at(shifts):
        return probe(tuple((name, shifts.get(name, 0.0)) for name in lambdas))

    steps = (0.08, 0.04, 0.02)
    for predictor in ("location", "scale"):
        names = [f"{predictor}:x:z#margin_x", f"{predictor}:x:z#margin_z"]
        for name in names:
            values, errors = [], []
            for h in steps:
                plus, minus = at({name: h}), at({name: -h})
                values.append((plus[0] - minus[0]) / (2 * h))
                errors.append((plus[1] + minus[1]) / (2 * h))
            estimate, uncertainty = _ladder(values, errors)
            index = derivatives.names.index(name)
            assert uncertainty < abs(estimate) / 4, "first derivative oracle is unresolved"
            assert abs(derivatives.gradient[index] - estimate) <= (
                uncertainty + derivatives.gradient_certificate[index]
            )
        values, errors = [], []
        for h in steps:
            pp, pm, mp, mm = [
                at({names[0]: s * h, names[1]: t * h})
                for s, t in ((1, 1), (1, -1), (-1, 1), (-1, -1))
            ]
            values.append((pp[0] - pm[0] - mp[0] + mm[0]) / (4 * h**2))
            errors.append((pp[1] + pm[1] + mp[1] + mm[1]) / (4 * h**2))
        estimate, uncertainty = _ladder(values, errors)
        i, j = (derivatives.names.index(name) for name in names)
        assert uncertainty < abs(estimate) / 4, "mixed derivative oracle is unresolved"
        assert abs(derivatives.hessian[i, j] - estimate) <= (
            uncertainty + derivatives.hessian_certificate[i, j]
        )
