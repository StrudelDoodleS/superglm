from __future__ import annotations

import math
import pickle
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from benchmarks import release_differential

import superglm.distributional.smoothing.authority as smoothing_authority
import superglm.distributional.smoothing.faces as smoothing_faces
import superglm.distributional.solver.solver as solver_module
import tests.test_distributional_endpoint_laml as endpoint_tests
from superglm import (
    GaussianLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    LambdaPolicy,
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
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.posterior import posterior_covariance
from superglm.distributional.predictor import compile_predictors
from superglm.distributional.results.solver import (
    DenseSolverConfig,
    DenseSolverResult,
    _assessment_is_numerically_stationary,
    _assessment_retained_kkt_ratio,
)
from superglm.distributional.smoothing.derivatives import LamlDerivativeError
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

    assert bound is not None and bound > 0.0
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
    *,
    max_iterations: int,
    initial: np.ndarray | None = None,
    curvature: str = "observed",
    tolerance: float = 1.0e-12,
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
            tolerance=tolerance,
            coefficient_curvature=curvature,  # type: ignore[arg-type]
        ),
    )


def _one_ulp_from(fit: DenseSolverResult) -> np.ndarray:
    """``fit``'s coefficients with its most penalized one moved by one unit in the last place."""
    stiff = int(np.argmax(np.abs(fit.penalty) @ np.abs(fit.coefficients)))
    start = np.array(fit.coefficients, copy=True)
    start[stiff] = np.nextafter(start[stiff], np.inf)
    return start


def _floor_start() -> np.ndarray:
    """One ulp from the capped fit polished to a 1000x tighter tolerance.

    An endpoint that only just met the tolerance leaves its one-ulp neighbour's
    Newton correction near the objective bound, and other BLAS builds stop at
    other such endpoints; polishing first starts the move from rounding level.
    """
    fit = _capped_spline_fit(max_iterations=150)
    assert fit.converged is True
    polished = _capped_spline_fit(max_iterations=150, initial=fit.coefficients, tolerance=1.0e-15)
    assert polished.converged is True
    return _one_ulp_from(polished)


def test_a_capped_fit_stops_at_its_rounding_floor() -> None:
    """Kills stepping on round-off once the score is within its rounding floor."""
    start = _floor_start()

    # At 1e-10 the one-ulp score is still ~1000x the bare tolerance, while the
    # certified decrement sits ~250x inside its limit on every BLAS kernel; at
    # 1e-12 that margin is 2-3x and some builds step.
    replay = _capped_spline_fit(max_iterations=150, initial=start, tolerance=1.0e-10)

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
    start = _floor_start()
    real_fit = smoothing_authority._fit_fixed_state
    calls = 0

    def counted_fit(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return real_fit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(smoothing_authority, "_fit_fixed_state", counted_fit)
    # At 1e-10, as in the replay test above, the certificate has margin on every BLAS kernel.
    config = replace(
        smoothing_authority._face_authority_config(DenseSolverConfig()), tolerance=1.0e-10
    )
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


def test_endpoint_authority_keeps_a_polish_that_meets_the_contract_its_source_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills keeping a source that fails stationarity because a passing polish's raw score is larger."""
    family, layout, response, plan, lambdas = _capped_spline_problem()
    # At 1e-10, as in the replay test above, the certificate has margin on every BLAS kernel.
    config = replace(
        smoothing_authority._face_authority_config(DenseSolverConfig()), tolerance=1.0e-10
    )
    arguments = {
        "lambdas": lambdas,
        "face": None,
        "config": config,
        "chunk_size": None,
        "phase_recorder": None,
    }
    fit = smoothing_authority._fit_fixed_state(
        family, layout, response, plan, initial=_floor_start(), **arguments
    )
    assert fit.converged is True
    objective = fit.penalized_optimizing_log_likelihood
    assert objective is not None
    nonzeros = np.count_nonzero(fit.penalty, axis=1)
    unit_roundoff = 0.5 * np.finfo(np.float64).eps
    rounding = nonzeros * unit_roundoff * (np.abs(fit.penalty) @ np.abs(fit.coefficients))

    # A score on an unpenalized row has no floor: above the bar, this fit fails the contract.
    source_score = np.zeros_like(fit.terminal_score)
    source_score[np.flatnonzero(nonzeros == 0)[0]] = 2.0 * config.tolerance * (1.0 + abs(objective))
    source = replace(fit, terminal_score=source_score, convergence_reason="objective_and_step")
    assert not _assessment_is_numerically_stationary(source, config.tolerance)

    # A larger score on the stiffest penalized row is inside that row's floor.
    polished_score = np.zeros_like(fit.terminal_score)
    polished_score[np.argmax(rounding)] = 0.5 * rounding.max()

    def scripted_fit(
        *args: object, initial: np.ndarray, score_only: bool = False, **kwargs: object
    ):
        if not score_only:
            return source
        return replace(
            fit, coefficients=initial, terminal_score=polished_score, convergence_reason="score"
        )

    monkeypatch.setattr(smoothing_authority, "_fit_fixed_state", scripted_fit)
    result = smoothing_authority._fit_endpoint_authority_stationary(
        family, layout, response, plan, initial=fit.coefficients, **arguments
    )

    assert _assessment_retained_kkt_ratio(result) > _assessment_retained_kkt_ratio(source)
    assert _assessment_is_numerically_stationary(result, config.tolerance)


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
    """Kills demanding a zero-step replay beside a numerically stationary cap."""
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

    The shape smooth's parameter is held just below the cap, so the scale
    smooth is the sole capped component. Selected, it ends either 1.5e-8 below
    the cap or exactly on it, depending on the summation order; exactly on it,
    two components share the cap and the face recheck retracts.
    """
    frame, response = _flagged_generalized_pareto_problem()
    family = GeneralizedParetoLSS()
    model = SuperLSS(
        family,
        family.scale(
            s("x1", kind="cr", k=8), cat("c1"), s("x2", kind="cr", k=6), re("c2"), ti("x1", "x2")
        ),
        family.shape(s("x2", kind="cr", k=5, lambda_policy=LambdaPolicy.fixed(9.99e9))),
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


def test_a_stationary_endgame_keeps_an_exact_face_beside_a_capped_tensor_margin() -> None:
    """A scalar exact face revalidates beside another cap the exact gradient certified.

    The release differential's GeneralizedGammaLSS rich book ends its Newton
    endgame stationary with ``mean:x1#wiggle`` on an exact face and the tensor
    margin ``mean:x1:x2#margin_x1`` at the cap. A margin that shares its block
    cannot join a face, so the sole-cap rule alone retracted the face and the
    fit stopped ``endpoint_revalidation_failed``. The refreshed Fellner--Schall
    evidence then nominates that margin, and the revalidation refit moves the
    coefficients by a rounding, so the gradient the stop published must be
    retaken at the published fit for the corrected covariance to replay it.
    Mutation checks: without the waiver this fit retracts; without the
    nomination rule it keeps ``unresolved_upper_bound`` nonempty; without the
    retaken gradient the corrected covariance falls back to the fixed one.
    """
    frame, eta, rng = release_differential.make_data(4000, 5155)
    response = rng.gamma(3.0, np.exp(0.5 * eta + 1.0) / 3.0)
    family = GeneralizedGammaLSS()
    model = SuperLSS(
        family,
        family.mean(
            s("x1", kind="cr", k=8), cat("c1"), s("x2", kind="cr", k=6), re("c2"), ti("x1", "x2")
        ),
        family.scale(s("x2", kind="cr", k=5)),
        family.shape(),
    )
    model.fit_reml(frame.iloc[:4000], response[:4000], outer="efs+newton")
    smoothing = model._require_fitted().smoothing
    assert smoothing is not None
    promoted = {name for item in smoothing.history for name in item.activated_face_components}
    beside = [
        name
        for name, value in smoothing.lambdas.items()
        if name not in promoted and value == smoothing.config.maximum_lambda
    ]
    if not promoted or not beside:
        # Another BLAS build can round onto a path that never caps the margin.
        pytest.skip("this build's endgame left no cap beside an exact face")
    assert smoothing.convergence_reason == "stationary"
    assert smoothing.converged is True
    assert smoothing.unresolved_upper_bound == ()
    assert set(model.result_.exact_face_components) == promoted
    fitted = model._require_fitted()
    # The retake makes the replayed gradient agree; the positive-definiteness
    # gate's verdict on the capped margin's near-zero curvature varies by platform.
    fallbacks = _corrected_covariance_fallbacks(fitted)
    assert _only_the_hessian_gate_declined(fallbacks), fallbacks
    loaded = pickle.loads(pickle.dumps(model))._require_fitted().smoothing
    assert loaded is not None
    assert loaded.convergence_reason == "stationary"


def test_a_stationary_endgame_voids_a_refreshed_cap_nomination(monkeypatch) -> None:
    """The endgame's exact gradient outranks a Fellner--Schall cap nomination.

    An exact-face revalidation after a stationary endgame refreshes the
    Fellner--Schall evidence, and that evidence can nominate a capped component
    outside the face whose exact gradient the endgame found flat. The nomination
    is forced here on every component outside the face, all of which that
    gradient judged. The fit still stops stationary. Mutation check: without the
    rule the stationary result kept the nomination.
    """
    import superglm.distributional.smoothing.loop as loop

    endgame = loop.run_newton_endgame
    fresh = loop._fresh_raw_evidence
    stationary: list[bool] = []
    nominated: list[tuple[str, ...]] = []

    def flagged_endgame(*args, **kwargs):
        outcome = endgame(*args, **kwargs)
        if outcome.kind == "stationary":
            stationary.append(True)
        return outcome

    def nominating(layout, lambdas, fit, config, *, face=None, **options):
        evidence = fresh(layout, lambdas, fit, config, face=face, **options)
        if stationary and face is not None:
            outside = evidence.estimated_names
            assert outside and not set(outside) & set(face.component_names)
            nominated.append(outside)
            return replace(evidence, unresolved_upper_bound=outside)
        return evidence

    monkeypatch.setattr(loop, "run_newton_endgame", flagged_endgame)
    monkeypatch.setattr(loop, "_fresh_raw_evidence", nominating)
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
    assert nominated, "the fixture's stationary endgame revalidates an exact face"
    assert smoothing is not None
    assert smoothing.convergence_reason == "stationary"
    assert smoothing.converged is True
    assert smoothing.unresolved_upper_bound == ()


def _flagged_generalized_pareto_model(**shape_options) -> SuperLSS:
    """The release differential's mid GeneralizedParetoLSS model, as in the tests above."""
    family = GeneralizedParetoLSS()
    return SuperLSS(
        family,
        family.scale(
            s("x1", kind="cr", k=8), cat("c1"), s("x2", kind="cr", k=6), re("c2"), ti("x1", "x2")
        ),
        family.shape(s("x2", kind="cr", k=5, **shape_options)),
        discrete=True,
    )


def test_a_stationary_endgame_revalidates_its_face_beside_another_cap(monkeypatch) -> None:
    """The sole-cap waiver, forced: the terminal recheck sees another cap outside the face.

    A tensor margin at the cap beside a scalar face is the case it covers. Here
    the recheck is told the face's component is not the sole cap, on a fixture
    whose endgame ends stationary with an exact face. That endgame's exact
    gradient certified every other cap, so the face is kept. The shape smooth
    is held below the cap so the face is a scalar one: at the cap the two
    components can form a joint face, which takes another route. Mutation
    check: without the waiver the recheck retracts the face and the fit stops
    ``endpoint_revalidation_failed``.
    """
    import superglm.distributional.smoothing.loop as loop

    recheck = loop._recheck_exact_face
    sole_cap = smoothing_faces._is_sole_cap_outside_face
    waivers: list[bool] = []

    def not_sole(*args, **kwargs) -> bool:
        sole_cap(*args, **kwargs)
        return False

    def recheck_beside_another_cap(*args, **kwargs):
        waivers.append(kwargs["other_caps_stationary"])
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(smoothing_faces, "_is_sole_cap_outside_face", not_sole)
            return recheck(*args, **kwargs)

    monkeypatch.setattr(loop, "_recheck_exact_face", recheck_beside_another_cap)
    frame, response = _flagged_generalized_pareto_problem()
    model = _flagged_generalized_pareto_model(lambda_policy=LambdaPolicy.fixed(9.99e9)).fit_reml(
        frame, response, outer="efs+newton"
    )
    smoothing = model._require_fitted().smoothing
    assert smoothing is not None
    assert waivers == [True], "the fixture's stationary endgame revalidates an exact face"
    assert smoothing.convergence_reason == "stationary"
    promoted = {name for item in smoothing.history for name in item.activated_face_components}
    assert promoted and set(model.result_.exact_face_components) == promoted


_HESSIAN_GATE_REFUSALS = (
    "the smoothing Hessian is not positive definite",
    "the smoothing Hessian certificate is too large",
    "does not establish positive invertibility",
)


def _corrected_covariance_fallbacks(fitted) -> list[str]:
    """The corrected covariance's fallback disclosures, after checking it against them.

    The covariance is finite, and it equals the fixed covariance exactly when a
    fallback is disclosed.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        corrected = posterior_covariance(fitted, kind="corrected")
    assert np.all(np.isfinite(corrected))
    fallbacks = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
        and "corrected covariance is unavailable" in str(item.message)
    ]
    assert bool(fallbacks) == np.array_equal(corrected, posterior_covariance(fitted, kind="fixed"))
    return fallbacks


def _only_the_hessian_gate_declined(fallbacks: list[str]) -> bool:
    """Whether every fallback is the positive-definiteness gate's own refusal."""
    return all(
        any(refusal in message for refusal in _HESSIAN_GATE_REFUSALS) for message in fallbacks
    )


@pytest.mark.parametrize("retake", ["published", "unavailable", "declined"])
def test_a_revalidated_stationary_stop_publishes_derivatives_at_its_fit(
    monkeypatch, retake: str
) -> None:
    """The stationary derivatives are retaken at the fit the face revalidation published.

    The corrected covariance replays its gradient at that fit and compares, so
    a retaken gradient always agrees with its replay. The covariance's separate
    positive-definiteness gate may still decline the correction here: the
    shape smooth ends near the lambda cap with a curvature of about 3e-8 of the
    largest, and where the endgame leaves that flat direction varies by
    platform, so a published retake may fall back only through that gate; the
    declined case forces its refusal. When the retake fails, the stop keeps
    the endgame's derivatives. In every case the corrected covariance is
    finite, and it falls back to the fixed one only with a disclosure.
    Mutation checks: without the retake there is no pass at the published fit,
    and a fallback without its ConvergenceWarning fails the declined case.
    """
    import superglm.distributional.smoothing.loop as loop

    retaken = loop.published_derivatives
    fits: list[DenseSolverResult] = []

    def spy(*args, **kwargs):
        fits.append(kwargs["fit"])
        if retake == "unavailable":
            raise LamlDerivativeError("injected unavailable derivatives at the published fit")
        return retaken(*args, **kwargs)

    monkeypatch.setattr(loop, "published_derivatives", spy)
    frame, response = _flagged_generalized_pareto_problem()
    model = _flagged_generalized_pareto_model().fit_reml(frame, response, outer="efs+newton")
    fitted = model._require_fitted()
    smoothing = fitted.smoothing
    assert smoothing is not None and smoothing.convergence_reason == "stationary"
    assert smoothing.history[-1].revalidated_face_components
    assert len(fits) == 1 and fits[0] is smoothing.terminal_fit
    if retake == "declined":
        import superglm.distributional.posterior as posterior

        def declined(*args, **kwargs):
            raise RuntimeError("the smoothing Hessian is not positive definite")

        monkeypatch.setattr(posterior, "_trusted_smoothing_hessian", declined)
    fallbacks = _corrected_covariance_fallbacks(fitted)
    if retake == "published":
        assert _only_the_hessian_gate_declined(fallbacks), fallbacks
    if retake == "declined":
        assert fallbacks


def test_a_revalidated_face_keeps_the_endgames_published_hessian(monkeypatch) -> None:
    """The retake keeps a smoothing Hessian the endgame published.

    The endgame is made to stop with its exact Hessian, as its Newton-step
    exits do. Without retained rows the corrected covariance can only use the
    published Hessian, so it may fall back only through the positive-
    definiteness gate. Mutation check: a gradient-only retake publishes no
    Hessian, and the correction falls back for want of retained rows.
    """
    import superglm.distributional.smoothing.loop as loop
    import superglm.distributional.smoothing.newton as newton

    endgame = loop.run_newton_endgame

    def with_hessian(*args, **kwargs):
        outcome = endgame(*args, **kwargs)
        if outcome.kind != "stationary" or outcome.derivatives.hessian is not None:
            return outcome
        derivatives = newton.published_derivatives(
            *args[:4],
            lambdas=outcome.state.lambdas,
            fit=outcome.state.fit,
            efs_config=kwargs["efs_config"],
            phase_recorder=None,
            reuse_session=kwargs["reuse_session"],
            want_hessian=True,
        )
        return replace(outcome, derivatives=derivatives)

    monkeypatch.setattr(loop, "run_newton_endgame", with_hessian)
    frame, response = _flagged_generalized_pareto_problem()
    model = _flagged_generalized_pareto_model().fit_reml(
        frame, response, retain_rows=False, outer="efs+newton"
    )
    fitted = model._require_fitted()
    smoothing = fitted.smoothing
    assert smoothing is not None and smoothing.convergence_reason == "stationary"
    assert smoothing.history[-1].revalidated_face_components
    assert smoothing.smoothing_hessian is not None
    # Without retained rows only the published Hessian can form the correction;
    # the positive-definiteness gate's verdict on it varies by platform here.
    fallbacks = _corrected_covariance_fallbacks(fitted)
    assert _only_the_hessian_gate_declined(fallbacks), fallbacks


def test_the_iteration_budget_does_not_refuse_a_stationary_endgames_face_check() -> None:
    """``max_reml_iter`` counts search iterations; the terminal face revalidation is not one.

    The fit is refitted with exactly the budget its own search used. Mutation
    check: when the revalidation counted against the budget, the refit stopped
    ``max_iterations`` with its face never revalidated.
    """
    frame, response = _flagged_generalized_pareto_problem()
    reference = _flagged_generalized_pareto_model().fit_reml(frame, response, outer="efs+newton")
    smoothing = reference._require_fitted().smoothing
    assert smoothing is not None and smoothing.convergence_reason == "stationary"
    searched = len(smoothing.history)
    while searched and smoothing.history[searched - 1].revalidated_face_components:
        searched -= 1
    assert searched < len(smoothing.history), "the fixture's stop revalidates an exact face"
    refit = _flagged_generalized_pareto_model().fit_reml(
        frame, response, max_reml_iter=searched, outer="efs+newton"
    )
    again = refit._require_fitted().smoothing
    assert again is not None
    assert again.convergence_reason == "stationary"
    assert refit.result_.exact_face_components == reference.result_.exact_face_components


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


def test_a_fisher_scoring_solve_does_not_take_the_rounding_floor_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills certifying a floor stop on Fisher information instead of observed curvature."""
    start = _floor_start()
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
