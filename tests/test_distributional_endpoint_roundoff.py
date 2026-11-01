from __future__ import annotations

import math
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.smoothing.authority as smoothing_authority
import superglm.distributional.smoothing.faces as smoothing_faces
import superglm.distributional.solver.solver as solver_module
import tests.test_distributional_endpoint_laml as endpoint_tests
from superglm import (
    GaussianLS,
    GeneralizedParetoLSS,
    Numeric,
    Predictor,
    Spline,
    SuperLSS,
    cat,
    re,
    s,
    ti,
)
from superglm._frame import as_eager_frame
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import compile_predictors
from superglm.distributional.results.solver import (
    DenseSolverConfig,
    DenseSolverResult,
    _assessment_is_numerically_stationary,
    _assessment_retained_kkt_ratio,
)
from superglm.distributional.smoothing.penalty_face import build_penalty_face
from superglm.distributional.solver import fit_dense_fixed_lambda
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import RandomEffect


def _spline_face():
    family = GaussianLS()
    weights = resolve_likelihood_weights(
        np.ones(20, dtype=np.float64),
        n_observations=20,
        contract=WeightContract(semantics="prior"),
    )
    frame = as_eager_frame(pd.DataFrame({"x": np.linspace(0.0, 1.0, 20)}))
    predictors = (
        Predictor(
            "location",
            {"x": Spline(kind="cr", n_knots=5)},
            intercept=True,
        ),
        Predictor(
            "scale",
            {"x": Spline(kind="cr", n_knots=5)},
            intercept=True,
        ),
    )
    layout = build_stacked_layout(compile_predictors(frame, weights, family.parameters, predictors))
    face = build_penalty_face(layout, (layout.penalty_names[0],))
    return face


def test_endpoint_revalidation_projection_roundoff_has_an_arithmetic_bound() -> None:
    face = _spline_face()
    canonical = face.lift_vector(np.linspace(1.0, 2.0, face.reduced_width))
    projected = face.project(canonical)
    movement = float(np.linalg.norm(projected - canonical, ord=2))

    bound = smoothing_faces._endpoint_revalidation_projection_bound(face, canonical)

    assert bound is not None
    assert movement <= bound


def test_endpoint_revalidation_projection_bound_rejects_a_material_scalar_move() -> None:
    family = GaussianLS()
    weights = resolve_likelihood_weights(
        np.ones(1, dtype=np.float64),
        n_observations=1,
        contract=WeightContract(semantics="prior"),
    )
    frame = as_eager_frame(pd.DataFrame({"effect": ["only"]}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor("location", {"effect": RandomEffect()}),
                Predictor("scale", {"effect": RandomEffect()}),
            ),
        )
    )
    face = build_penalty_face(layout, (layout.penalty_names[0],))
    canonical = np.zeros(face.width, dtype=np.float64)
    altered = canonical + 1.0e-14 * face.constraint_basis[:, 0]

    bound = smoothing_faces._endpoint_revalidation_projection_bound(face, altered)
    projected = face.project(altered)
    movement = float(np.linalg.norm(projected - altered, ord=2))

    assert bound is not None
    assert movement > bound


def test_stationary_endpoint_revalidation_accepts_a_subnormal_projection_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_check = smoothing_faces._check_face_direction

    def perturb_revalidation_start(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            endpoint_initial = np.array(kwargs["endpoint_initial"], copy=True)
            endpoint_initial[0] = np.nextafter(0.0, np.inf)
            kwargs["endpoint_initial"] = endpoint_initial
        return real_check(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(smoothing_faces, "_check_face_direction", perturb_revalidation_start)
    component_name, smoothing = endpoint_tests._scalar_efs_fit(0.5)

    assert calls == 2
    assert smoothing.converged is True
    assert smoothing.terminal_fit.coefficient_face is not None
    assert smoothing.history[-1].revalidated_face_components == (component_name,)


def test_endpoint_revalidation_records_material_state_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_check = smoothing_faces._check_face_direction
    real_fit = smoothing_faces._fit_endpoint_authority_stationary
    inject_result = False
    injected = False

    def inject_endpoint_result(*args: object, **kwargs: object):
        nonlocal injected
        fit = real_fit(*args, **kwargs)  # type: ignore[arg-type]
        if inject_result and not injected and kwargs.get("face") is not None:
            endpoint_face = kwargs["face"]
            assert endpoint_face is not None
            assert endpoint_face.null_basis.shape[1] > 0
            injected = True
            fit = replace(
                fit,
                coefficients=fit.coefficients + 1.0e-6 * endpoint_face.null_basis[:, 0],
            )
        return fit

    def perturb_revalidation_start(*args: object, **kwargs: object):
        nonlocal calls, inject_result
        calls += 1
        if calls == 2:
            np.testing.assert_array_equal(kwargs["endpoint_initial"], kwargs["initial"])
            inject_result = True
        try:
            return real_check(*args, **kwargs)  # type: ignore[arg-type]
        finally:
            inject_result = False

    monkeypatch.setattr(
        smoothing_faces, "_fit_endpoint_authority_stationary", inject_endpoint_result
    )
    monkeypatch.setattr(smoothing_faces, "_check_face_direction", perturb_revalidation_start)
    (
        _family,
        _layout,
        _response,
        _plan,
        _lambdas,
        component_name,
        _finite,
        _config,
        smoothing,
    ) = endpoint_tests._gamma_isolated_cap_face_problem()

    terminal_event = smoothing.history[-1]
    assert calls == 2
    assert smoothing.converged is False
    assert terminal_event.deactivated_face_components == (component_name,)
    assert terminal_event.endpoint_assessment_failure_reason == "endpoint_state_changed"


def test_result_rejects_a_within_bound_state_change_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_check = smoothing_faces._check_face_direction
    real_fit = smoothing_faces._fit_endpoint_authority_stationary
    inject_result = False
    injected = False

    def inject_endpoint_result(*args: object, **kwargs: object):
        nonlocal injected
        fit = real_fit(*args, **kwargs)  # type: ignore[arg-type]
        if inject_result and not injected and kwargs.get("face") is not None:
            endpoint_face = kwargs["face"]
            assert endpoint_face is not None
            assert endpoint_face.null_basis.shape[1] > 0
            injected = True
            fit = replace(
                fit,
                coefficients=fit.coefficients + 1.0e-6 * endpoint_face.null_basis[:, 0],
            )
        return fit

    def material_refusal(*args: object, **kwargs: object):
        nonlocal calls, inject_result
        calls += 1
        if calls == 2:
            np.testing.assert_array_equal(kwargs["endpoint_initial"], kwargs["initial"])
            inject_result = True
        try:
            return real_check(*args, **kwargs)  # type: ignore[arg-type]
        finally:
            inject_result = False

    monkeypatch.setattr(
        smoothing_faces, "_fit_endpoint_authority_stationary", inject_endpoint_result
    )
    monkeypatch.setattr(smoothing_faces, "_check_face_direction", material_refusal)
    _values = endpoint_tests._gamma_isolated_cap_face_problem()
    smoothing = _values[-1]
    retraction = next(item for item in smoothing.history if item.deactivated_face_components)
    cap_index, endpoint_index = retraction.coefficient_fit_indices[:2]
    cap_fit = smoothing.coefficient_fits[cap_index]
    endpoint_fit = smoothing.coefficient_fits[endpoint_index]
    source_fit = smoothing.coefficient_fits[retraction.source_fit_index]
    assert endpoint_fit.coefficient_face is not None
    within_bound = replace(
        endpoint_fit,
        coefficients=source_fit.coefficients
        + 1.0e-14 * endpoint_fit.coefficient_face.null_basis[:, 0],
    )
    fits = list(smoothing.coefficient_fits)

    # Beside a numerically stationary cap, a replay that stepped changed the
    # endpoint state only beyond the refit envelope.
    assert _assessment_is_numerically_stationary(cap_fit, retraction.coefficient_tolerances[0])
    fits[endpoint_index] = replace(within_bound, iterations=within_bound.iterations + 1)
    with pytest.raises(ValueError, match="within the refit envelope"):
        replace(smoothing, coefficient_fits=tuple(fits))
    # The strict rule a saved v0.40.0 result was made under refused any
    # stepped replay beside a strictly stationary cap.
    fits[cap_index] = replace(cap_fit, terminal_score=np.zeros_like(cap_fit.terminal_score))
    with pytest.raises(ValueError, match="within the refit envelope"):
        replace(smoothing, coefficient_fits=tuple(fits))
    replace(smoothing, coefficient_fits=tuple(fits), rounding_floor_stationarity=False)

    # Beside a cap that is not stationary, direct face authority also needs
    # full observed rank.
    fits[cap_index] = endpoint_tests._score_above_rounding_floor(cap_fit)
    fits[endpoint_index] = within_bound
    with pytest.raises(ValueError, match="within the refit envelope"):
        replace(smoothing, coefficient_fits=tuple(fits))

    assert endpoint_fit.terminal_reduced_rank is not None
    bad_rank = replace(
        endpoint_fit.terminal_reduced_rank,
        rank=endpoint_fit.terminal_reduced_rank.rank - 1,
    )
    bad_endpoint = replace(endpoint_fit, terminal_reduced_rank=bad_rank)
    fits[endpoint_index] = bad_endpoint
    with pytest.raises(ValueError, match="full observed rank"):
        replace(smoothing, coefficient_fits=tuple(fits))


def _capped_spline_problem():
    """A Gaussian location spline with its smoothing parameter at the cap."""
    family = GaussianLS()
    n = 40
    x = np.linspace(0.0, 1.0, n)
    weights = resolve_likelihood_weights(
        np.ones(n, dtype=np.float64),
        n_observations=n,
        contract=WeightContract(semantics="prior"),
    )
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(pd.DataFrame({"x": x})),
            weights,
            family.parameters,
            (
                Predictor("location", {"x": Spline(kind="cr", n_knots=8)}, intercept=True),
                Predictor("scale", {}, intercept=True),
            ),
        )
    )
    response = 1.0 + np.sin(2.0 * np.pi * x) + 0.1 * np.cos(17.0 * x)
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    lambdas = dict.fromkeys(layout.penalty_names, 1.0e10)
    return family, layout, response, plan, lambdas


def _capped_spline_fit(
    *, max_iterations: int, initial: np.ndarray | None = None, curvature: str = "observed"
) -> DenseSolverResult:
    family, layout, response, plan, lambdas = _capped_spline_problem()
    return fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        layout.penalty_matrix(lambdas),
        initial=initial,
        config=DenseSolverConfig(
            max_iterations=max_iterations,
            tolerance=1.0e-12,
            coefficient_curvature=curvature,  # type: ignore[arg-type]
        ),
    )


def _one_ulp_from(fit: DenseSolverResult) -> np.ndarray:
    """``fit``'s coefficients with its most penalized one moved by one unit in the last place."""
    stiff = int(np.argmax(np.abs(fit.penalty) @ np.abs(fit.coefficients)))
    start = np.array(fit.coefficients, copy=True)
    start[stiff] = np.nextafter(start[stiff], np.inf)
    return start


def test_a_capped_fit_stops_at_its_rounding_floor() -> None:
    """Kills stepping on round-off once the score is within its rounding floor."""
    fit = _capped_spline_fit(max_iterations=150)
    assert fit.converged is True
    start = _one_ulp_from(fit)

    replay = _capped_spline_fit(max_iterations=150, initial=start)

    tolerance = replay.config.tolerance
    assert replay.convergence_reason == "score"
    assert replay.iterations == 0
    np.testing.assert_array_equal(replay.coefficients, start)
    # One ulp of a penalized coefficient moves the score beyond the bare tolerance.
    assert _assessment_retained_kkt_ratio(replay) > tolerance
    assert _assessment_is_numerically_stationary(replay, tolerance)


def test_endpoint_authority_does_not_polish_a_score_within_its_rounding_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills a Newton polish that can only follow round-off."""
    family, layout, response, plan, lambdas = _capped_spline_problem()
    start = _one_ulp_from(_capped_spline_fit(max_iterations=150))
    real_fit = smoothing_authority._fit_fixed_state
    calls = 0

    def counted_fit(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return real_fit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(smoothing_authority, "_fit_fixed_state", counted_fit)
    config = smoothing_authority._face_authority_config(DenseSolverConfig())
    fit = smoothing_authority._fit_endpoint_authority_stationary(
        family,
        layout,
        response,
        plan,
        lambdas=lambdas,
        face=None,
        initial=start,
        config=config,
        chunk_size=None,
        phase_recorder=None,
    )

    assert calls == 1
    assert _assessment_retained_kkt_ratio(fit) > config.tolerance
    assert _assessment_is_numerically_stationary(fit, config.tolerance)


def test_numerical_stationarity_admits_a_score_within_its_rounding_floor() -> None:
    """Kills the bare relative score test and a floor taken from another coordinate."""
    fit = _capped_spline_fit(max_iterations=150)
    assert fit.converged is True
    tolerance = fit.config.tolerance
    objective = fit.penalized_optimizing_log_likelihood
    assert objective is not None
    scale = 1.0 + abs(objective)
    unit_roundoff = 0.5 * np.finfo(np.float64).eps
    nonzeros = np.count_nonzero(fit.penalty, axis=1)
    # P b loses up to gamma_k (|P| |b|)_i to rounding: far above the tolerance at a cap.
    rounding = nonzeros * unit_roundoff * (np.abs(fit.penalty) @ np.abs(fit.coefficients))
    stiff = int(np.argmax(rounding))
    assert rounding[stiff] > 1.0e3 * tolerance * scale

    def with_score(index: int, value: float) -> DenseSolverResult:
        score = np.zeros_like(fit.terminal_score)
        score[index] = value
        return replace(
            fit, terminal_score=score, converged=False, convergence_reason="max_iterations"
        )

    within_floor = with_score(stiff, 0.5 * rounding[stiff])
    assert _assessment_retained_kkt_ratio(within_floor) > tolerance
    assert _assessment_is_numerically_stationary(within_floor, tolerance)

    unpenalized = int(np.flatnonzero(nonzeros == 0)[0])
    above_floor = with_score(unpenalized, 2.0 * tolerance * scale)
    assert not _assessment_is_numerically_stationary(above_floor, tolerance)

    trial = _capped_spline_fit(max_iterations=1)
    assert trial.converged is False
    assert not _assessment_is_numerically_stationary(trial, tolerance)


def test_endpoint_revalidation_accepts_a_stepped_replay_within_the_refit_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills demanding a zero-step replay beside a numerically stationary cap, and
    accepting one beside a strictly stationary cap in a result saved before the
    rounding-floor rule."""
    calls = 0
    real_check = smoothing_faces._check_face_direction
    real_fit = smoothing_faces._fit_endpoint_authority_stationary
    inject_result = False
    injected = False

    def step_endpoint_replay(*args: object, **kwargs: object):
        nonlocal injected
        fit = real_fit(*args, **kwargs)  # type: ignore[arg-type]
        if inject_result and not injected and kwargs.get("face") is not None:
            endpoint_face = kwargs["face"]
            assert endpoint_face is not None
            injected = True
            fit = replace(
                fit,
                coefficients=fit.coefficients + 1.0e-14 * endpoint_face.null_basis[:, 0],
                iterations=fit.iterations + 1,
            )
        return fit

    def replay_revalidation(*args: object, **kwargs: object):
        nonlocal calls, inject_result
        calls += 1
        inject_result = calls == 2
        try:
            return real_check(*args, **kwargs)  # type: ignore[arg-type]
        finally:
            inject_result = False

    monkeypatch.setattr(smoothing_faces, "_fit_endpoint_authority_stationary", step_endpoint_replay)
    monkeypatch.setattr(smoothing_faces, "_check_face_direction", replay_revalidation)
    values = endpoint_tests._gamma_isolated_cap_face_problem()
    component_name, smoothing = values[5], values[-1]

    assert calls == 2 and injected
    assert smoothing.converged is True
    revalidation = smoothing.history[-1]
    assert revalidation.revalidated_face_components == (component_name,)
    cap_fit, endpoint_fit = (
        smoothing.coefficient_fits[index] for index in revalidation.coefficient_fit_indices
    )
    source_fit = smoothing.coefficient_fits[revalidation.source_fit_index]
    tolerance = revalidation.coefficient_tolerances[0]
    assert _assessment_is_numerically_stationary(cap_fit, tolerance)
    assert endpoint_fit.iterations > 0
    assert not np.array_equal(endpoint_fit.coefficients, source_fit.coefficients)
    assert smoothing.terminal_fit is endpoint_fit

    # The strict rule a saved result was made with requires the source's exact
    # projection for every replay beside a strictly stationary cap.
    fits = list(smoothing.coefficient_fits)
    fits[revalidation.coefficient_fit_indices[0]] = replace(
        cap_fit, terminal_score=np.zeros_like(cap_fit.terminal_score)
    )
    replace(smoothing, coefficient_fits=tuple(fits))
    with pytest.raises(ValueError, match="changed the canonical endpoint state"):
        replace(smoothing, coefficient_fits=tuple(fits), rounding_floor_stationarity=False)


def _flagged_generalized_pareto_problem() -> tuple[pd.DataFrame, np.ndarray]:
    """The release differential's mid GeneralizedParetoLSS book, its 4,000 training rows."""
    rng = np.random.default_rng(5187)
    n_train, total = 4000, 6000
    x1, x2 = rng.uniform(0, 10, total), rng.uniform(-2, 2, total)
    levels = [f"a{i}" for i in range(6)]
    c1 = rng.choice(levels, total, p=[0.3, 0.25, 0.2, 0.12, 0.08, 0.05])
    c2 = rng.choice([f"g{i:02d}" for i in range(25)], total)
    linear = rng.normal(0, 1, total)
    rng.uniform(0, 1, total)
    rng.uniform(0.2, 1.0, total)
    c1_effect = dict(zip(levels, [0, 0.2, -0.15, 0.3, -0.1, 0.25], strict=True))
    c2_effect = dict(zip(sorted(set(c2)), rng.normal(0, 0.15, 25), strict=False))
    eta = (
        0.06 * x1
        + 0.3 * np.sin(1.3 * x2)
        + np.array([c1_effect[level] for level in c1])
        + np.array([c2_effect[level] for level in c2])
        + 0.1 * linear
        + 0.04 * x1 * np.tanh(x2)
    )
    eta -= eta.mean()
    response = rng.pareto(4.0, total) * np.exp(0.3 * eta) + 1.0e-3
    frame = pd.DataFrame({"x1": x1, "x2": x2, "c1": c1, "c2": c2})
    return frame.iloc[:n_train], response[:n_train]


def test_capped_generalized_pareto_fit_revalidates_its_exact_face() -> None:
    """A smoothing parameter at the cap leaves a score only resolved to its rounding floor.

    At the 1e10 cap the rounding of ``P b`` exceeds the 1e-12 relative score
    tolerance, so the exact-face revalidation passed or failed on round-off and
    this fit stopped with ``endpoint_revalidation_failed``.
    """
    frame, response = _flagged_generalized_pareto_problem()
    family = GeneralizedParetoLSS()
    model = SuperLSS(
        family,
        family.scale(
            s("x1", kind="cr", k=8), cat("c1"), s("x2", kind="cr", k=6), re("c2"), ti("x1", "x2")
        ),
        family.shape(s("x2", kind="cr", k=5)),
        discrete=True,
    )
    model.fit_reml(frame, response, outer="efs+newton")
    smoothing = model._require_fitted().smoothing
    assert smoothing is not None
    assert smoothing.convergence_reason == "stationary"
    assert smoothing.converged is True
    # No worse than v0.40.0's fit of this book, within the release
    # differential's objective tolerance.
    released = -395.1090527
    assert smoothing.objective <= released + 1.0e-5 * (1.0 + abs(released))

    # Saving keeps the rule the receipts were made under. Its face and cap fits
    # are stationary only within their rounding floors, so a load that read the
    # result as made before the floor would validate it under the strict rule.
    loaded = pickle.loads(pickle.dumps(model))._require_fitted().smoothing
    assert loaded is not None
    assert loaded.rounding_floor_stationarity is True
    assert replace(loaded).convergence_reason == "stationary"


def _null_direction_problem(offset: float = -2.0e8):
    """Two constant location columns tied by a penalty at the cap, offset to cancel.

    At ``b = (1e8, 1e8)`` the penalty term ``P b`` is exactly zero, but its
    rounding floor ``gamma_2 (|P| |b|)`` is about 440. The location score
    along ``b1 + b2``, the penalty's null direction, is a genuine likelihood
    gain that no rounding of ``P b`` produces. With ``offset = 0`` the fit
    starts from ``b = (0.5, 0.5)`` instead, where the floor is about 2e-6.
    """
    family = GaussianLS(scale_floor=0)
    n = 40
    weights = resolve_likelihood_weights(
        np.ones(n), n_observations=n, contract=WeightContract(semantics="prior")
    )
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(pd.DataFrame({"x": np.ones(n), "z": np.ones(n)})),
            weights,
            family.parameters,
            (
                Predictor("location", {"x": Numeric(), "z": Numeric()}, intercept=False),
                Predictor("scale", {}, intercept=True),
            ),
            offsets={"location": np.full(n, offset)},
        )
    )
    response = np.tile([0.0, 2.0], n // 2)
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    penalty = np.zeros((3, 3))
    penalty[:2, :2] = 1.0e10 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    return family, layout, response, plan, penalty


def test_a_score_inside_its_floor_that_a_newton_step_removes_is_not_stationary() -> None:
    """Kills a rounding-floor stop that a material Newton correction does not refuse.

    From ``b = (1e8, 1e8)``, location 0 and scale sqrt(2), the location scores
    are 20 inside a floor of about 440, and the floor alone stopped the fit
    there. The responses alternate 0 and 2, so the maximum likelihood location
    and scale are both 1.
    """
    family, layout, response, plan, penalty = _null_direction_problem()
    start = np.array([1.0e8, 1.0e8, math.log(math.sqrt(2.0))])
    unit_roundoff = 0.5 * np.finfo(np.float64).eps
    floor = 2.0 * unit_roundoff * (np.abs(penalty) @ np.abs(start))
    assert np.all(floor[:2] > 400.0)

    fit = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        penalty,
        initial=start,
        config=DenseSolverConfig(max_iterations=100, tolerance=1.0e-12),
    )

    assert fit.converged is True
    assert fit.iterations > 0
    objective = fit.penalized_optimizing_log_likelihood
    assert objective is not None
    # The stop certifies a gap of at most tol (1 + |f|) in the quadratic model,
    # whose location and log-scale curvatures at the optimum are n and 2n; the
    # location sums three terms of magnitude up to 2e8, so it also carries
    # their rounding.
    n = len(response)
    gap = fit.config.tolerance * (1.0 + abs(objective))
    location, scale = fit.theta[0]
    assert abs(location - 1.0) <= math.sqrt(2.0 * gap / n) + 3.0 * unit_roundoff * 4.0e8
    assert abs(math.log(scale)) <= math.sqrt(2.0 * gap / (2.0 * n)) + 4.0 * unit_roundoff

    # The same score forged onto the converged state is still inside its floor
    # (P b = 0), and its Newton correction still moves the objective.
    tolerance = fit.config.tolerance
    score = np.array(fit.terminal_score, copy=True)
    score[:2] = 20.0
    unconverged = replace(
        fit, terminal_score=score, converged=False, convergence_reason="max_iterations"
    )
    assert not _assessment_is_numerically_stationary(unconverged, tolerance)
    with pytest.raises(ValueError, match="rounding floor"):
        replace(fit, terminal_score=score, convergence_reason="score")


def _cancelled_score_problem():
    """Review round 3's counterexample, rebuilt so that every computed ``P b`` is exact.

    Location columns ``(1, -1)`` with offset ``-2t`` at ``b = (t, -t)``, where
    ``t = 2**24``, give location 0, and the penalty ``2**33 [[1, 1], [1, 1]]``
    has ``b`` in its null space. The location score lies along that null
    direction, ``(1, -1)``. The rounding floor ``gamma_2 (|P| |b|)`` is 64, yet
    each product is a coefficient times a power of two and each row adds two
    terms within a factor of two of each other, so every BLAS computes ``P b``
    exactly at every ``b`` the fit visits. The responses alternate
    ``delta - 1`` and ``delta + 1``, so the maximum likelihood estimates are
    location ``delta`` and scale 1, and the start's scale is the estimate at
    location 0.
    """
    family = GaussianLS(scale_floor=0)
    n = 128
    t = 2.0**24
    delta = -0.25
    weights = resolve_likelihood_weights(
        np.ones(n), n_observations=n, contract=WeightContract(semantics="prior")
    )
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(pd.DataFrame({"x": np.ones(n), "z": -np.ones(n)})),
            weights,
            family.parameters,
            (
                Predictor("location", {"x": Numeric(), "z": Numeric()}, intercept=False),
                Predictor("scale", {}, intercept=True),
            ),
            offsets={"location": np.full(n, -2.0 * t)},
        )
    )
    response = np.tile([delta - 1.0, delta + 1.0], n // 2)
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    penalty = np.zeros((3, 3))
    penalty[:2, :2] = 2.0**33
    start = np.array([t, -t, math.log(math.sqrt(1.0 + delta**2))])
    return family, layout, response, plan, penalty, start, delta


def test_a_floor_stop_holds_for_every_score_its_rounding_admits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills certifying a floor stop's Newton correction on the computed score alone.

    In the reviewer's case a fused multiply-add BLAS kernel returned
    ``P b = (128, -128)`` where it is exactly zero, and that rounding cancelled
    the location score along the penalty's null direction. The rest lay along
    the penalized direction, inside its floor and with a Newton correction far
    below the tolerance, so the fit stopped at iteration 0 at location 0. Here
    the start's score is forged to that outcome on every platform: rounding
    inside the floor takes the null-direction score away and leaves 1/16 along
    ``(1, 1)``.
    """
    family, layout, response, plan, penalty, start, delta = _cancelled_score_problem()
    stiff = 0.0625
    real_geometry = solver_module._geometry
    implied_products = []

    def cancelled_geometry(context, state, source):
        geometry = real_geometry(context, state, source)
        if not np.array_equal(state.coefficients, start):
            return geometry
        score = np.array([stiff, stiff, geometry.score_penalized[2]])
        implied_products.append(geometry.score_data - score)
        return replace(geometry, score_penalized=score)

    monkeypatch.setattr(solver_module, "_geometry", cancelled_geometry)
    fit = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        penalty,
        initial=start,
        config=DenseSolverConfig(max_iterations=100, tolerance=1.0e-12),
    )

    # The forged score is one a rounding of P b = 0 within its floor returns.
    unit_roundoff = 0.5 * np.finfo(np.float64).eps
    floor = 2.0 * unit_roundoff * (np.abs(penalty) @ np.abs(start))
    assert implied_products
    assert np.all(np.abs(implied_products[0]) <= floor)
    assert fit.converged is True
    assert fit.iterations > 0
    objective = fit.penalized_optimizing_log_likelihood
    assert objective is not None
    # As in the null-direction test: the stop certifies a gap of at most
    # tol (1 + |f|) in a quadratic model with location and log-scale
    # curvatures n and 2n, and the location sums three terms up to 2t.
    n = len(response)
    gap = fit.config.tolerance * (1.0 + abs(objective))
    location, scale = fit.theta[0]
    assert abs(location - delta) <= math.sqrt(2.0 * gap / n) + 3.0 * unit_roundoff * 4.0 * 2.0**24
    assert abs(math.log(scale)) <= math.sqrt(2.0 * gap / (2.0 * n)) + 4.0 * unit_roundoff

    # The same score on the converged state is inside its floor with a Newton
    # correction far below the tolerance, but a rounding inside that floor can
    # hide a null-direction score whose correction is not.
    tolerance = fit.config.tolerance
    score = np.array([stiff, stiff, 0.0])
    unconverged = replace(
        fit, terminal_score=score, converged=False, convergence_reason="max_iterations"
    )
    assert _assessment_retained_kkt_ratio(unconverged) > tolerance
    assert not _assessment_is_numerically_stationary(unconverged, tolerance)
    with pytest.raises(ValueError, match="rounding floor"):
        replace(fit, terminal_score=score, convergence_reason="score")


def test_a_rounding_floor_stop_needs_unfallbacked_observed_curvature() -> None:
    """Kills certifying a floor stop's Newton correction on a fallback curvature."""
    family, layout, response, plan, penalty = _null_direction_problem(offset=0.0)
    fit = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        penalty,
        initial=np.array([0.5, 0.5, math.log(math.sqrt(2.0))]),
        config=DenseSolverConfig(max_iterations=100, tolerance=1.0e-12),
    )
    tolerance = fit.config.tolerance
    # A score along the penalized direction, above the bare tolerance and
    # inside its floor of about 2e-6, whose Newton correction is far below the
    # tolerance for every score within that floor. The fit's own scale score
    # is not kept: a decrement stop bounds it through the curvature, not by
    # its floor.
    score = np.array([1.0e-6, -1.0e-6, 0.0])
    within = replace(fit, terminal_score=score, convergence_reason="score")
    assert _assessment_retained_kkt_ratio(within) > tolerance
    assert _assessment_is_numerically_stationary(within, tolerance)

    fallback = replace(fit.terminal_curvature, actual_source="fisher", fallback_count=1)
    unconverged = replace(within, converged=False, convergence_reason="max_iterations")
    assert not _assessment_is_numerically_stationary(
        replace(unconverged, terminal_curvature=fallback), tolerance
    )
    with pytest.raises(ValueError, match="rounding floor"):
        replace(within, terminal_curvature=fallback)


def test_a_fisher_scoring_solve_does_not_take_the_rounding_floor_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills certifying a floor stop on Fisher information instead of observed curvature."""
    start = _one_ulp_from(_capped_spline_fit(max_iterations=150))
    real_floor = solver_module._score_within_rounding_floor
    calls = 0

    def counted_floor(**kwargs: object) -> bool:
        nonlocal calls
        calls += 1
        return real_floor(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(solver_module, "_score_within_rounding_floor", counted_floor)
    _capped_spline_fit(max_iterations=150, initial=start, curvature="fisher")
    assert calls == 0
    _capped_spline_fit(max_iterations=150, initial=start)
    assert calls > 0
