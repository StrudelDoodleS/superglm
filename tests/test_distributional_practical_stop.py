from __future__ import annotations

import math
import pickle
import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sympy

from superglm import ConvergenceWarning, SuperLSS, TwoPieceNormalLSS, cat, re, s, term, ti
from superglm.distributional import GaussianLS, Predictor
from superglm.distributional import fit_diagnostics as diagnostics_module
from superglm.distributional.fit_diagnostics import diagnose_distributional_fit
from superglm.distributional.model import DenseDistributionalModel
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.serialization import pickle_state
from superglm.distributional.smoothing.evidence import _FreshRawEvidence, _lower_bound_pressure
from superglm.distributional.smoothing.objective import _stable_isolated_gfs_update
from superglm.distributional.smoothing.proposals import _accelerated_proposal, _scaled_proposal
from superglm.features import Numeric, RandomEffect, Spline
from superglm.reml.efs_update import EFSComponentState, wood_fasiolo_update
from superglm.solvers.rank import decompose_symmetric_deflated
from superglm.types import LambdaPolicy
from tests.bound_predictor_fixtures import model_from_templates


def test_saturated_update_checks_inverse_products_before_cancellation() -> None:
    factor = np.array([1.0, math.sqrt(2.0)])
    penalty = np.outer(factor, factor)
    eigenvalue = float(factor @ factor)
    projector = penalty / eigenvalue
    null_projector = np.eye(2) - projector
    lam = 1.0e10
    # H = I + λS: its two eigenspaces have eigenvalues 1 and 1 + λ||factor||².
    inverse = null_projector + projector / (1.0 + eigenvalue * lam)
    beta = np.array([factor[1], -factor[0]]) + 2.0**-40 * factor
    component = EFSComponentState("smooth", slice(0, 2), penalty, 1.0, lam, LambdaPolicy.estimate())
    fit = SimpleNamespace(
        coefficients=beta,
        coefficient_face=None,
        terminal_rank=SimpleNamespace(rank=2, row_decided=False),
        terminal_data_curvature=np.eye(2),
        terminal_penalized_curvature=np.eye(2) + lam * penalty,
    )
    config = DistributionalEFSConfig()
    raw = wood_fasiolo_update((component,), beta, inverse, inverse_scale=1.0)
    stable, names = _stable_isolated_gfs_update((component,), fit, inverse, raw, config)

    assert names == {"smooth"}
    # The exact multiplier is 1 / ((1 + 3λ) λ 9·2^-80), safely above one.
    assert stable.raw_log_steps["smooth"] > math.log(100.0)
    assert stable.proposal_kinds["smooth"] == "gfs"
    assert stable.stationarity_log_residuals["smooth"] > 0.0
    np.testing.assert_allclose(
        math.exp(-stable.stationarity_log_residuals["smooth"]),
        lam * (stable.quadratic_forms["smooth"] + stable.trace_terms["smooth"]),
        rtol=8.0 * np.finfo(float).eps,
        atol=0.0,
    )
    projection_bound = (
        64.0 * beta.size * np.finfo(float).eps * np.linalg.norm(factor) * np.linalg.norm(beta)
    )
    assert (
        abs(math.sqrt(stable.quadratic_forms["smooth"]) - abs(float(factor @ beta)))
        <= projection_bound
    )

    # A trace rounded below the nominal saturation threshold still needs repair.
    rounded = replace(
        raw, trace_terms={"smooth": (1.0 - 2.0 * math.sqrt(np.finfo(float).eps)) / lam}
    )
    retried, names = _stable_isolated_gfs_update((component,), fit, inverse, rounded, config)
    assert names == {"smooth"}
    assert retried.raw_log_steps == stable.raw_log_steps
    assert retried.stationarity_log_residuals == stable.stationarity_log_residuals
    assert retried.trace_terms == stable.trace_terms

    _, refused = _stable_isolated_gfs_update((component,), fit, 2.0 * inverse, raw, config)
    assert refused == frozenset()


def test_saturated_update_refuses_an_unresolved_residual_product() -> None:
    penalty = np.array([[0.5, -0.5], [-0.5, 0.5]])
    common = np.eye(2) - penalty
    large = 2.0**40
    small = np.spacing(large)
    lam = 2.0**16
    curvature = large * np.ones((2, 2)) + small * np.eye(2)
    inverse = common / (2.0 * large + small) + penalty / (lam + small)
    beta = np.array([0.001, -0.001])
    component = EFSComponentState("smooth", slice(0, 2), penalty, 1.0, lam, LambdaPolicy.estimate())
    fit = SimpleNamespace(
        coefficients=beta,
        coefficient_face=None,
        terminal_rank=SimpleNamespace(rank=2, row_decided=False),
        terminal_data_curvature=curvature,
        terminal_penalized_curvature=curvature + lam * penalty,
    )
    update = wood_fasiolo_update((component,), beta, inverse)

    # The true residual is smaller than the rounding bound of the matrix product.
    scale = float(np.sum(np.abs(penalty) * (np.abs(inverse) @ np.abs(curvature)).T))
    assert small / (lam + small) < beta.size * np.finfo(float).eps * scale
    stable, names = _stable_isolated_gfs_update(
        (component,), fit, inverse, update, DistributionalEFSConfig()
    )
    assert names == frozenset()
    assert stable is update


def _rational(values) -> sympy.Matrix:
    return sympy.Matrix([[sympy.Rational(float(v)) for v in row] for row in np.atleast_2d(values)])


@pytest.mark.parametrize("case", ["penalized", "null_space"])
def test_a_row_decided_saturated_step_pairs_its_grey_columns_with_the_rows(case: str) -> None:
    """The stable step of a rank decided from the rows matches the exact curvature's.

    ``z = x + d h`` puts a near pair in the grey band, and ``h`` is a column of
    the saturated block, so the grey direction reaches the penalty.  The rows
    are exact, and the reference is the step of ``tr(P H^-1 A)`` for the exact
    ``H``, in rationals.  ``"penalized"``: the formed curvature couples the
    pair to ``h`` by ``64 u``, inside its formation bound.  ``"null_space"``:
    a first-difference penalty, whose constant null direction the grey column
    also reaches.  The representative keeps the formed top block, so the step
    is accurate to ``u`` times that block's condition number (``4e9`` in the
    null-space case).  Mutation check: pairing the grey columns with the formed
    curvature (the formed numerator) moves the penalized step by ``8e-3``, and
    declines the null-space one, whose generic quotient is negative.
    """
    rng = np.random.default_rng(0 if case == "penalized" else 5)
    grid = 2.0**-8
    columns = [np.round(rng.normal(size=64) / grid) * grid for _ in range(2)]
    linear = np.round(np.linspace(-1.0, 1.0, 64) / grid) * grid
    x = np.round(rng.normal(size=64) / grid) * grid
    if case == "penalized":
        block_penalty, lam, d, plant = np.diag([1.0, 1.0, 0.0]), 2.0**40, 2.0**-23, 64.0
    else:
        difference = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        block_penalty, lam, d, plant = difference.T @ difference, 2.0**36, 2.0**-22, 0.0
    rows = np.column_stack([*columns, linear, x, x + d * columns[0]])
    penalty = np.zeros((5, 5))
    penalty[:3, :3] = block_penalty
    exact = _rational(rows).T * _rational(rows) + sympy.Rational(lam) * _rational(penalty)
    data = rows.T @ rows
    scale = np.sqrt(np.diag(data + lam * penalty))
    data[0, 3] = data[3, 0] = data[0, 3] + plant * 2.0**-53 * scale[0] * scale[3]

    def project(basis):
        product = exact * _rational(basis)
        cross = np.array(product, dtype=float)
        block = np.array(_rational(basis).T * product, dtype=float)
        return block, cross, 2.0**-53 * np.abs(block), 2.0**-53 * np.abs(cross)

    rank = decompose_symmetric_deflated(
        data + lam * penalty, project, formation_bound=512.0 * 2.0**-53
    )
    assert rank.row_decided and rank.rank == 5 and np.any(rank.grey_columns)
    beta = np.array([2.0**-38, -(2.0**-38), 1.0, 0.5, -0.5])
    component = EFSComponentState(
        "smooth", slice(0, 3), block_penalty, 2.0, lam, LambdaPolicy.estimate()
    )
    fit = SimpleNamespace(
        coefficients=beta,
        coefficient_face=None,
        terminal_rank=rank,
        terminal_data_curvature=data,
        terminal_penalized_curvature=data + lam * penalty,
    )
    inverse = rank.pseudo_inverse()
    raw = wood_fasiolo_update((component,), beta, inverse, inverse_scale=1.0)
    stable, names = _stable_isolated_gfs_update(
        (component,), fit, inverse, raw, DistributionalEFSConfig()
    )

    span = sympy.Matrix.hstack(*_rational(block_penalty).columnspace())
    projector = sympy.zeros(5, 5)
    projector[:3, :3] = span * (span.T * span).inv() * span.T
    residual = projector * exact.LUsolve(exact - sympy.Rational(lam) * _rational(penalty))
    quadratic = (_rational(beta) * _rational(penalty) * _rational(beta).T)[0, 0]
    reference = sympy.log(residual.trace() / (quadratic * sympy.Rational(lam)))
    top = np.asarray(rank.retained_values)[~np.asarray(rank.grey_columns)]
    tolerance = 1.0e-7 + 64.0 * 2.0**-53 * float(np.max(top) / np.min(top))
    assert names == {"smooth"}
    assert abs(stable.raw_log_steps["smooth"] - float(reference)) <= tolerance


@pytest.mark.parametrize(
    "old_value", [49.99, 3.7, 1.0e-3, 1.0e9, 9.999999999e9, 0.125, 12.0, 2.0e9]
)
def test_scaled_proposal_lands_exactly_on_the_maximum(old_value: float) -> None:
    config = DistributionalEFSConfig(maximum_lambda=1.0e10, minimum_lambda=1.0e-6)
    cap = config.maximum_lambda if old_value > 1.0 else 50.0
    config = DistributionalEFSConfig(maximum_lambda=cap, minimum_lambda=1.0e-6)
    step = math.log(cap) - math.log(old_value)
    lambdas, log_steps = _scaled_proposal(
        {"a": old_value},
        {"a": step},
        ("a",),
        1.0,
        config,
    )
    assert lambdas["a"] == cap
    assert log_steps["a"] == math.log(cap) - math.log(old_value)


def test_a_bound_limited_step_lands_on_the_bound_from_every_start() -> None:
    """A step equal to the distance to a bound is assigned that bound, bit for bit.

    The update's bound-limited step is ``log(bound) - log(old)``. Recomputing the
    proposal as ``old * exp(step)`` rounds through a log and an exp and, at a cap
    of 2e10, lands up to about 15 float spacings short of it, which a snap inside
    8 spacings does not recover; every later test of "at the cap" is an exact
    comparison, so the search then reads the term as not capped. Mutation check:
    with the proposal computed as ``old * exp(step)`` and then snapped, about a
    third of these starts miss a cap of 2e10.
    """
    configs = {
        "cap 3e5": DistributionalEFSConfig(maximum_lambda=3.0e5),
        "cap 5e7": DistributionalEFSConfig(maximum_lambda=5.0e7),
        "cap 2e10": DistributionalEFSConfig(maximum_lambda=2.0e10),
        "relative box": DistributionalEFSConfig(relative_box=True, lambda_units=(("a", 2.0**7),)),
    }
    for label, config in configs.items():
        lower, upper = config.lower("a"), config.upper("a")
        span = math.log(upper) - math.log(lower)
        for k in range(1, 400):
            old = lower * math.exp(span * k / 400.0)
            rise = math.log(upper) - math.log(old)
            lambdas, log_steps = _scaled_proposal({"a": old}, {"a": rise}, ("a",), 1.0, config)
            assert lambdas["a"] == upper, (label, old)
            assert log_steps["a"] == rise
            fall = math.log(lower) - math.log(old)
            lambdas, _ = _scaled_proposal({"a": old}, {"a": fall}, ("a",), 1.0, config)
            assert lambdas["a"] == lower, (label, old)


def test_scaled_proposal_lands_exactly_on_the_minimum() -> None:
    config = DistributionalEFSConfig(maximum_lambda=1.0e10, minimum_lambda=1.0e-6)
    old_value = 2.5e-6
    step = math.log(config.minimum_lambda) - math.log(old_value)
    lambdas, _ = _scaled_proposal({"a": old_value}, {"a": step}, ("a",), 1.0, config)
    assert lambdas["a"] == config.minimum_lambda


def test_accelerated_proposal_snaps_a_near_cap_log_value() -> None:
    config = DistributionalEFSConfig(maximum_lambda=50.0, minimum_lambda=1.0e-6)
    log_value = math.log(50.0) - 1.0e-15
    result = _accelerated_proposal(
        {"a": 40.0},
        ("a",),
        np.array([log_value]),
        np.array([log_value - math.log(40.0)]),
        config,
    )
    assert result is not None
    lambdas, _ = result
    assert lambdas["a"] == 50.0


def _noise_random_effect_fit(*, practical: bool, max_lambda: float, levels: int = 20):
    rng = np.random.default_rng(3)
    labels = np.repeat(np.array([f"l{i}" for i in range(levels)]), 20)
    y = rng.normal(size=len(labels))
    frame = pd.DataFrame({"effect": labels})
    model = model_from_templates(
        family=GaussianLS(scale_floor=1.0e-4),
        predictors=(Predictor("location", {"effect": RandomEffect()}), Predictor("scale", {})),
    )
    model.fit_reml(frame, y, max_lambda=max_lambda, practical_reml=practical)
    return model, model._require_fitted().smoothing


def test_practical_window_replays_non_increasing_steps(monkeypatch) -> None:
    """A practical plateau's window is a full run of non-increasing accepted steps after the last longer one.

    The one-observation random effect whose mean sits 1e-12 above its prior
    has a harmonic tail toward an infinite smoothing parameter.  Its ordinary
    fit, with a practical parameter tolerance of 1e-3, stops on the practical
    plateau's own run (the stop that
    ``test_practical_plateau_stops_an_immaterial_tail_without_certifying_infinity``
    pins), and the history it records ends in that run.

    The rule itself is driven with controlled steps: each iteration's
    proposal keeps the lambdas it reached but reports a scripted step,
    ``1 / (1 + i)`` at iteration ``i`` with one longer step (10) at
    iteration ``jump``, and the objective and parameter tests are opened
    (``plateau_tolerance=1``, ``practical_parameter_tolerance=1e300``) so
    the step test alone decides.  A longer step restarts the run, so the stop
    comes a whole window after it, at ``jump + 1 + plateau_iterations``
    iterations; with no longer step inside the first window it comes at
    ``plateau_iterations``.  Mutation check: dropping ``max_accepted_step <=
    previous_accepted_step`` from the practical run stops every scripted fit
    at ``plateau_iterations``.
    """
    from superglm.distributional.smoothing import loop
    from tests.test_distributional_endpoint_laml import _scalar_efs_fit

    _name, ordinary = _scalar_efs_fit(
        1.000000000001,
        maximum_lambda=1.0e10,
        starting_lambda=0.1,
        max_iterations=300,
        practical_convergence=True,
        practical_parameter_tolerance=1.0e-3,
    )
    assert ordinary.convergence_reason == "practical_plateau"
    window = ordinary.config.plateau_iterations
    assert all(item.accepted for item in ordinary.history)
    steps = [item.max_accepted_log_step for item in ordinary.history]
    run = steps[-window - 1 :]
    assert run == sorted(run, reverse=True), run

    def scripted(jump: int | None) -> list[float]:
        real, full_scale = loop._scaled_proposal, []

        def proposal(current, log_steps, names, scale, config):
            lambdas, steps = real(current, log_steps, names, scale, config)
            full_scale.append(scale == 1.0)
            i = sum(full_scale) - 1  # every iteration's first proposal is at full scale
            step = 10.0 if i == jump else 1.0 / (1.0 + i)
            return lambdas, {name: step for name in steps}

        with monkeypatch.context() as patch:
            patch.setattr(loop, "_scaled_proposal", proposal)
            _name, smoothing = _scalar_efs_fit(
                1.000000000001,
                maximum_lambda=1.0e10,
                starting_lambda=0.1,
                max_iterations=300,
                practical_convergence=True,
                practical_parameter_tolerance=1.0e300,
                plateau_tolerance=1.0,
            )
        assert smoothing.convergence_reason == "practical_plateau"
        return [item.max_accepted_log_step for item in smoothing.history]

    assert len(scripted(None)) == window
    for jump in range(1, window):
        steps = scripted(jump)
        assert steps[jump] == 10.0
        assert len(steps) == jump + 1 + window, (jump, steps)


def _backtracked_gaussian(monkeypatch, *, score_longer_trials):
    from superglm.distributional.smoothing import loop

    rng = np.random.default_rng(4)
    u = np.repeat(rng.uniform(-1, 1, 256), 4)
    x = np.repeat(rng.normal(size=256), 4)
    z = x + 1e-7 * np.tile([1, 1, -1, -1], 256)
    y = 2 * u + np.sin(3 * u) + np.tile([1, -1, -1, 1], 256)
    frame = pd.DataFrame({"u": u, "x": x, "z": z})
    family = GaussianLS(scale_floor=0)
    model = SuperLSS(
        family,
        family.location(s("u", kind="cr", k=8), term("x", Numeric()), term("z", Numeric())),
        family.scale(),
        discrete=False,
    )
    raw, objective = loop._scaled_proposal, loop._trial_laplace_objective
    trial_scale = 1.0

    def proposal(current, steps, names, scale, config):
        nonlocal trial_scale
        trial_scale = scale
        return raw(current, steps, names, scale, config)

    def trial_objective(*args, **kwargs):
        if trial_scale > 1 / 256:
            if not score_longer_trials:
                return None
            score = objective(*args, **kwargs)
            assert score is not None
            return score + 10.0
        return objective(*args, **kwargs)

    monkeypatch.setattr(loop, "_scaled_proposal", proposal)
    monkeypatch.setattr(loop, "_trial_laplace_objective", trial_objective)
    return model.fit_reml(
        frame, y, initial_lambda=2.45, max_lambda=1e12, max_reml_iter=4, outer="efs"
    )


def test_refused_longer_trials_do_not_make_a_practical_plateau(monkeypatch, recwarn):
    """Only the 1/256 trials have a score. The outer search must disclose its residual."""
    model = _backtracked_gaussian(monkeypatch, score_longer_trials=False)

    smoothing = model._require_fitted().smoothing
    assert model.result_.coefficient_converged
    assert model.result_.smoothing_converged is False
    assert model.smoothing_convergence_reason_ == "max_iterations"
    assert smoothing.iterations == 4
    assert all(item.accepted and item.raw_backtracks == 8 for item in smoothing.history)
    assert smoothing.history[-1].max_proposed_log_step > 0.075
    assert "max_iterations" in str(recwarn.pop(ConvergenceWarning).message)


@pytest.mark.pins_behaviour
def test_scored_backtracking_can_still_support_a_practical_plateau(monkeypatch):
    """Pin the practical plateau after longer trials are scored and have worse objectives."""
    model = _backtracked_gaussian(monkeypatch, score_longer_trials=True)

    smoothing = model._require_fitted().smoothing
    assert model.result_.smoothing_converged is True
    assert model.smoothing_convergence_reason_ == "practical_plateau"
    assert smoothing.iterations == smoothing.config.plateau_iterations
    assert all(item.accepted and item.raw_backtracks == 8 for item in smoothing.history)


@pytest.mark.usefixtures("no_endpoint_trial")
def test_outward_plateau_requires_a_window_of_scorable_trials(monkeypatch, recwarn):
    """A refused trial breaks the window even when the accepted moves span a factor of e."""
    from superglm.distributional.smoothing import loop

    first_trial = True
    objective = loop._trial_laplace_objective

    def trial_objective(*args, **kwargs):
        nonlocal first_trial
        if first_trial:
            first_trial = False
            return None
        return objective(*args, **kwargs)

    monkeypatch.setattr(loop, "_trial_laplace_objective", trial_objective)
    frame, y = _preempt_fixture()
    model = model_from_templates(
        family=GaussianLS(scale_floor=1.0e-4),
        predictors=(Predictor("location", {"effect": RandomEffect()}), Predictor("scale", {})),
    )
    model.fit_reml(
        frame,
        y,
        initial_lambda=1.0e6,
        max_lambda=1.0e6 * math.exp(1.5),
        max_log_step=0.6,
        max_reml_iter=3,
        reml_tol=1.0e-8,
        inner_tol=1.0e-10,
        reml_plateau_tol=1.0e-6,
        outer="efs",
    )

    smoothing = model._require_fitted().smoothing
    assert [item.raw_backtracks for item in smoothing.history] == [1, 0, 0]
    assert model.result_.smoothing_converged is False
    assert model.smoothing_convergence_reason_ == "lambda_cap_unresolved"
    assert "lambda_cap_unresolved" in str(recwarn.pop(ConvergenceWarning).message)


def test_lower_bound_pressure_names_components_pinned_at_the_minimum() -> None:
    config = DistributionalEFSConfig(minimum_lambda=1.0e-6, tolerance=1.0e-6)
    evidence = _FreshRawEvidence(
        components=(),
        estimated_names=("a", "b"),
        update=None,
        maximum=0.0,
        working_infinity=(),
        unresolved_upper_bound=(),
    )
    lambdas = {"a": 1.0e-6, "b": 0.5}
    raw = {"a": -0.3, "b": -0.3}
    assert _lower_bound_pressure(evidence, lambdas, raw, config) == ("a",)
    raw_inward = {"a": 0.2, "b": -0.3}
    assert _lower_bound_pressure(evidence, lambdas, raw_inward, config) == ()


def _preempt_fixture():
    rng = np.random.default_rng(7)
    labels = np.repeat(np.array(["a", "b", "c", "d"]), 10)
    y = rng.normal(size=len(labels))
    return pd.DataFrame({"effect": labels}), y


def _preempt_fit(practical: bool):
    frame, y = _preempt_fixture()
    model = model_from_templates(
        family=GaussianLS(scale_floor=1.0e-4),
        predictors=(Predictor("location", {"effect": RandomEffect()}), Predictor("scale", {})),
    )
    model.fit_reml(
        frame,
        y,
        lambdas={"location:effect#wiggle": 1000.0},
        max_lambda=1002.5,
        max_log_step=1.0e-3,
        max_reml_iter=10,
        reml_tol=1.0e-8,
        inner_tol=1.0e-10,
        reml_plateau_tol=1.0e-6,
        practical_reml=practical,
    )
    return model, model._require_fitted().smoothing


def test_practical_stop_does_not_preempt_the_exact_face() -> None:
    practical_model, practical = _preempt_fit(True)
    strict_model, strict = _preempt_fit(False)
    assert strict.convergence_reason == "lambda_change"
    assert strict_model.exact_face_components_ == ("location:effect#wiggle",)
    assert practical.convergence_reason != "practical_plateau"
    assert practical_model.exact_face_components_ == strict_model.exact_face_components_
    assert practical.converged is True
    assert practical.unresolved_upper_bound == ()
    assert abs(practical.objective - strict.objective) <= 1.0e-9 * (1.0 + abs(strict.objective))


def _start_fixture():
    rng = np.random.default_rng(11)
    n = 600
    x = rng.uniform(-1.0, 1.0, n)
    y = 0.4 + 0.9 * np.sin(np.pi * x) + rng.normal(scale=0.35, size=n)
    return pd.DataFrame({"x": x}), y


def _start_fit(**kwargs):
    frame, y = _start_fixture()
    model = model_from_templates(
        family=GaussianLS(),
        predictors=(Predictor("location", {"x": Spline(kind="cr", k=8)}), Predictor("scale", {})),
    )
    model.fit_reml(frame, y, practical_reml=False, **kwargs)
    return model._require_fitted().smoothing


def test_fit_reml_initial_lambda_sets_the_search_start() -> None:
    low = _start_fit(initial_lambda=1.0e-3)
    high = _start_fit(initial_lambda=1.0e3)
    assert low.initial_lambdas["location:x#wiggle"] == 1.0e-3
    assert high.initial_lambdas["location:x#wiggle"] == 1.0e3
    assert low.config.initial_lambda == 1.0e-3
    assert high.config.initial_lambda == 1.0e3


def test_fit_reml_initial_lambda_is_capped_by_max_lambda_and_validated() -> None:
    capped = _start_fit(initial_lambda=10.0, max_lambda=5.0)
    assert capped.config.initial_lambda == 5.0
    frame, y = _start_fixture()
    model = model_from_templates(
        family=GaussianLS(),
        predictors=(Predictor("location", {"x": Spline(kind="cr", k=8)}), Predictor("scale", {})),
    )
    with pytest.raises(ValueError, match="initial_lambda"):
        model.fit_reml(frame, y, initial_lambda=0.0)
    with pytest.raises(ValueError, match="initial_lambda"):
        model.fit_reml(frame, y, initial_lambda=float("nan"))


@pytest.mark.parametrize(
    ("residual", "expected"),
    [
        (5.0e-7, "info"),
        (9.0e-3, "info"),
        (1.5e-2, "warning"),
        (0.9, "warning"),
        (1.5, "error"),
    ],
)
def test_residual_severity_bands(residual: float, expected: str) -> None:
    assert diagnostics_module._residual_severity(residual, 1.0e-6) == expected


@pytest.mark.parametrize("reason", ["practical_plateau", "lambda_change"])
@pytest.mark.parametrize("residual", [9.0e-3, 1.5e-2, 1.5])
def test_trajectory_unsettled_severity_is_label_independent(reason, residual) -> None:
    """The unsettled-trajectory finding's severity is its residual's band, whatever the stop's label.

    The fit settles below tolerance, so the terminal residual and the stop's
    label are set on its smoothing result: one value in each severity band, under
    a practical and a strict label.
    """
    model, smoothing = _noise_random_effect_fit(practical=True, max_lambda=1.0e10)
    fitted = model._require_fitted()
    unsettled = replace(smoothing)
    object.__setattr__(unsettled, "convergence_reason", reason)
    object.__setattr__(unsettled, "terminal_raw_max_log_step", residual)
    state = replace(fitted.fit_state, smoothing=unsettled)
    report = diagnose_distributional_fit(
        DenseDistributionalModel(family=fitted.family, _fit_state=state)
    )
    findings = [f for f in report.findings if f.code == "smoothing.trajectory_unsettled"]
    expected = diagnostics_module._residual_severity(residual, smoothing.config.tolerance)
    assert [f.severity for f in findings] == [expected]


def _ridge_outward_window(*, t=0.5, moves=(0.3, 0.4, 0.5), second_moves=None):
    """Exact unit-information Gaussian ridge profile, including its covariance.

    beta=t/(1+lambda), Var(beta)=1/(1+lambda), and the rank-one LAML
    F=.5*(log(1+1/lambda)+t**2*lambda/(1+lambda)).  Its EFS raw
    log step is log((1+lambda)/(t**2*lambda)), so t<1 pushes outward
    while t>1 at large lambda returns toward a finite optimum.
    """
    names = ("mean:a",) if second_moves is None else ("mean:a", "mean:b")
    movements = [moves] if second_moves is None else [moves, second_moves]
    paths = [1.0e6 * np.exp(np.r_[0.0, np.cumsum(values)]) for values in movements]
    fits = []
    objectives = []
    for index in range(4):
        penalties = np.array([path[index] for path in paths])
        means = t / (1.0 + penalties)
        fits.append(
            SimpleNamespace(
                theta=np.column_stack((means, np.ones_like(means))),
                covariance=np.diag(1.0 / (1.0 + penalties)),
            )
        )
        objectives.append(
            float(0.5 * np.sum(np.log1p(1.0 / penalties) + t**2 * penalties / (1 + penalties)))
        )
    history = []
    for index in range(3):
        history.append(
            SimpleNamespace(
                accepted=True,
                stage="efs",
                source_fit_index=index,
                accepted_fit_index=index + 1,
                lambdas_before={name: paths[k][index] for k, name in enumerate(names)},
                lambdas_after={name: paths[k][index + 1] for k, name in enumerate(names)},
                accepted_log_steps={name: movements[k][index] for k, name in enumerate(names)},
                objective_before=objectives[index],
                objective_after=objectives[index + 1],
                objective_relative_change=abs(objectives[index + 1] - objectives[index])
                / (1.0 + abs(objectives[index])),
                activated_face_components=(),
                deactivated_face_components=(),
                revalidated_face_components=(),
                refused_face_components=(),
            )
        )
    raw_steps = {
        name: math.log((1.0 + paths[k][-1]) / (t**2 * paths[k][-1])) for k, name in enumerate(names)
    }
    config = DistributionalEFSConfig(
        practical_convergence=True,
        plateau_tolerance=1.0e-6,
        practical_parameter_tolerance=1.0e-4,
    )
    return history, fits, raw_steps, config


def _outward_window_matches(history, fits, raw_steps, config):
    from superglm.distributional.results.smoothing import _practical_outward_window

    return _practical_outward_window(
        history=history,
        coefficient_fits=fits,
        terminal_raw_log_steps=raw_steps,
        config=config,
    )


def test_practical_outward_window_distinguishes_ridge_drift_from_finite_optimum() -> None:
    outward = _ridge_outward_window(t=0.5)
    inward = _ridge_outward_window(t=2.0)
    assert _outward_window_matches(*outward)
    assert not _outward_window_matches(*inward)
    # The accepted finite ridge fit is already close to its exact nullspace:
    # both prediction and covariance error are bounded by 1/lambda.
    history, fits, _raw, _config = outward
    terminal_lambda = history[-1].lambdas_after["mean:a"]
    assert np.max(np.abs(fits[-1].theta[:, 0])) <= 0.5 / terminal_lambda
    assert np.linalg.norm(fits[-1].covariance, ord=2) <= 1.0 / terminal_lambda


@pytest.mark.parametrize(
    "moves",
    [
        (0.02, 0.03, 0.05),  # a small accepted probe is not evidence of flatness
        (0.6, -0.1, 0.7),  # net outward distance cannot conceal an oscillation
        (0.5, 0.7, 0.0),  # a clipped duplicate cannot complete the window
    ],
)
def test_practical_outward_window_requires_substantial_monotone_probes(moves) -> None:
    assert not _outward_window_matches(*_ridge_outward_window(moves=moves))


def test_practical_outward_window_checks_each_coordinate_span() -> None:
    assert not _outward_window_matches(*_ridge_outward_window(second_moves=(0.02, 0.03, 0.05)))
    assert _outward_window_matches(*_ridge_outward_window(second_moves=(0.4, 0.4, 0.4)))


@pytest.mark.parametrize("parameter", [0, 1, 2])
def test_practical_outward_window_checks_cumulative_all_parameter_change(parameter: int) -> None:
    history, fits, raw_steps, config = _ridge_outward_window()
    for index, fit in enumerate(fits):
        if parameter == 2:
            fit.theta = np.column_stack((fit.theta, np.ones(fit.theta.shape[0])))
        fit.theta[:, parameter] = 1.0 + index * 1.5e-4
    # Every individual change is below 1e-4 relative to (1+|theta|), but
    # the complete first-source-to-terminal window moves by more than it.
    assert not _outward_window_matches(history, fits, raw_steps, config)


def test_practical_outward_window_checks_cumulative_objective_change() -> None:
    history, fits, raw_steps, config = _ridge_outward_window()
    for index, item in enumerate(history):
        item.objective_before = 1.0 - index * 1.5e-6
        item.objective_after = 1.0 - (index + 1) * 1.5e-6
        item.objective_relative_change = 1.5e-6 / (1.0 + abs(item.objective_before))
    assert not _outward_window_matches(history, fits, raw_steps, config)


def test_practical_outward_window_rejects_missing_coordinate_pressure() -> None:
    history, fits, raw_steps, config = _ridge_outward_window(second_moves=(0.02, 0.03, 0.05))
    raw_steps.pop("mean:b")
    assert not _outward_window_matches(history, fits, raw_steps, config)


def test_practical_outward_window_preserves_lower_bound_authority() -> None:
    history, fits, raw_steps, config = _ridge_outward_window(second_moves=(0.0, 0.0, 0.0))
    config = replace(config, minimum_lambda=1.0e6, initial_lambda=1.0e6)
    raw_steps["mean:b"] = -0.1
    assert not _outward_window_matches(history, fits, raw_steps, config)


def test_practical_outward_window_cannot_omit_unchanged_lower_pressure() -> None:
    history, fits, raw_steps, config = _ridge_outward_window(second_moves=(0.0, 0.0, 0.0))
    config = replace(config, minimum_lambda=1.0e6, initial_lambda=1.0e6)
    raw_steps["mean:b"] = -0.1
    assert not _outward_window_matches(history, fits, raw_steps, config)
    # The coordinate is pinned by clipping, not stationary. Removing its
    # pressure used to bypass the veto because its accepted lambda never moved.
    raw_steps.pop("mean:b")
    assert not _outward_window_matches(history, fits, raw_steps, config)


@pytest.mark.parametrize("fixed_value", [0.0, 1.0e10])
def test_practical_outward_window_allows_unchanged_zero_pressure_fixed_coordinate(fixed_value):
    history, fits, raw_steps, config = _ridge_outward_window()
    for item in history:
        item.lambdas_before["mean:fixed"] = fixed_value
        item.lambdas_after["mean:fixed"] = fixed_value
        item.accepted_log_steps["mean:fixed"] = 0.0
    raw_steps["mean:fixed"] = 0.0
    assert _outward_window_matches(history, fits, raw_steps, config)


def test_practical_outward_window_requires_pressure_in_terminal_coordinate_order() -> None:
    history, fits, raw_steps, config = _ridge_outward_window(second_moves=(0.4, 0.4, 0.4))
    assert not _outward_window_matches(history, fits, dict(reversed(raw_steps.items())), config)


@pytest.mark.parametrize("window_size", [1, 2])
def test_practical_outward_window_honors_configured_probe_count(window_size) -> None:
    history, fits, raw_steps, config = _ridge_outward_window(moves=(0.1, 0.2, 1.2))
    config = replace(
        config,
        plateau_iterations=window_size,
        maximum_lambda=history[-1].lambdas_after["mean:a"],
    )
    assert _outward_window_matches(history[-window_size:], fits, raw_steps, config)


def _outward_random_effect_fit(*, practical=True, max_log_step=0.5, span=1.5, fixed_lambda=None):
    frame, y = _preempt_fixture()
    features = {"effect": RandomEffect()}
    if fixed_lambda is not None:
        frame["fixed"] = np.tile(["u", "v"], len(frame) // 2)
        features["fixed"] = RandomEffect(lambda_policy=LambdaPolicy.fixed(fixed_lambda))
    model = model_from_templates(
        family=GaussianLS(scale_floor=1.0e-4),
        predictors=(Predictor("location", features), Predictor("scale", {})),
    )
    model.fit_reml(
        frame,
        y,
        lambdas={"location:effect#wiggle": 1.0e6},
        max_lambda=1.0e6 * math.exp(span),
        max_log_step=max_log_step,
        max_reml_iter=20,
        reml_tol=1.0e-8,
        inner_tol=1.0e-10,
        reml_plateau_tol=1.0e-6,
        practical_reml=practical,
    )
    return model, model._require_fitted().smoothing


@pytest.mark.usefixtures("no_endpoint_trial")
@pytest.mark.parametrize("fixed_value", [0.0, 1.0e6 * math.exp(1.5)])
def test_practical_outward_result_records_fixed_coordinate_pressure(fixed_value):
    model, smoothing = _outward_random_effect_fit(fixed_lambda=fixed_value)
    assert smoothing.convergence_reason == "practical_plateau"
    assert tuple(smoothing.terminal_raw_log_steps) == tuple(smoothing.lambdas)
    assert smoothing.terminal_raw_log_steps["location:fixed#wiggle"] == 0.0
    pressure = dict(smoothing.terminal_raw_log_steps)
    pressure.pop("location:fixed#wiggle")

    restored = pickle.loads(pickle_state(model._require_fitted())).smoothing
    assert restored.terminal_raw_log_steps == smoothing.terminal_raw_log_steps


@pytest.mark.usefixtures("no_endpoint_trial")
def test_practical_outward_cap_stop_keeps_pressure_and_replays_finite_fit() -> None:
    _model, smoothing = _outward_random_effect_fit()
    assert smoothing.convergence_reason == "practical_plateau"
    assert smoothing.converged and not smoothing.matched_certified
    assert smoothing.unresolved_upper_bound == ("location:effect#wiggle",)
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.terminal_raw_log_steps["location:effect#wiggle"] > 0.0
    assert len(smoothing.history) == smoothing.config.plateau_iterations

    _strict_model, strict = _outward_random_effect_fit(practical=False)
    assert strict.converged and strict.terminal_fit.coefficient_face is not None
    finite_theta = smoothing.terminal_fit.theta
    exact_theta = strict.terminal_fit.theta
    relative = np.abs(finite_theta - exact_theta) / (
        1.0 + np.maximum(np.abs(finite_theta), np.abs(exact_theta))
    )
    assert np.max(relative) < smoothing.config.practical_parameter_tolerance
    assert (
        abs(smoothing.objective - strict.objective) / (1.0 + abs(strict.objective))
        < smoothing.config.plateau_tolerance
    )
    finite_covariance = smoothing.terminal_fit.terminal_pseudo_inverse()
    exact_covariance = strict.terminal_fit.terminal_pseudo_inverse()
    assert (
        np.linalg.norm(finite_covariance - exact_covariance, ord=2)
        / (1.0 + np.linalg.norm(exact_covariance, ord=2))
        < smoothing.config.practical_parameter_tolerance
    )


@pytest.mark.usefixtures("no_endpoint_trial")
def test_practical_outward_pressure_survives_a_pickle_round_trip() -> None:
    model, smoothing = _outward_random_effect_fit()
    restored = pickle.loads(pickle_state(model._require_fitted())).smoothing
    assert restored.convergence_reason == "practical_plateau"
    assert restored.unresolved_upper_bound == smoothing.unresolved_upper_bound
    assert restored.terminal_raw_log_steps == smoothing.terminal_raw_log_steps
    assert not restored.matched_certified
    with pytest.raises(TypeError):
        restored.terminal_raw_log_steps["location:effect#wiggle"] = -0.1


_CAP = 20.0
_FREE, _CAPPED = "location:x#wiggle", "location:z#wiggle"


def _cap_fit(*, z_policy=None, second_weak=False, **kwargs):
    """A strong smooth in ``x`` beside a weak one in ``z``, under a cap of 20.

    ``second_weak`` adds a smooth in ``w``, a covariate the response ignores.
    """
    rng = np.random.default_rng(21)
    x, z = rng.uniform(-1.0, 1.0, 600), rng.uniform(-1.0, 1.0, 600)
    y = 0.4 + 0.9 * np.sin(2.0 * np.pi * x) + 0.05 * z**2 + rng.normal(scale=0.35, size=600)
    weak = (
        Spline(kind="cr", k=8)
        if z_policy is None
        else Spline(kind="cr", k=8, lambda_policy=z_policy)
    )
    frame, terms = pd.DataFrame({"x": x, "z": z}), {"x": Spline(kind="cr", k=8), "z": weak}
    if second_weak:
        frame["w"] = np.random.default_rng(22).uniform(-1.0, 1.0, 600)
        terms["w"] = Spline(kind="cr", k=8)
    model = model_from_templates(
        family=GaussianLS(),
        predictors=(Predictor("location", terms), Predictor("scale", {})),
    )
    options = {"reml_tol": 1.0e-8, "practical_reml": False, **kwargs}
    model.fit_reml(
        frame,
        y,
        max_lambda=_CAP,
        inner_tol=1.0e-10,
        outer="efs",
        **options,
    )
    return model, model._require_fitted().smoothing


def _cap_start() -> tuple[SuperLSS, dict[str, float]]:
    """The fit with ``z`` fixed on the cap, and a start that puts ``x`` three log units above it."""
    reference, smoothing = _cap_fit(z_policy=LambdaPolicy.fixed(_CAP))
    assert smoothing.converged
    start = smoothing.lambdas[_FREE] * math.exp(3.0)
    assert start < _CAP
    return reference, {_CAPPED: _CAP, _FREE: start}


def _refuse_every_promotion(monkeypatch) -> list[dict[str, float]]:
    """Make every cap unpromotable; return the smoothing states the search judged."""
    from superglm.distributional.smoothing import loop

    judged: list[dict[str, float]] = []

    def joint(*args, lambdas, **kwargs):
        judged.append(dict(lambdas))
        return None, None

    monkeypatch.setattr(loop, "_try_joint_exact_face", joint)
    monkeypatch.setattr(loop, "_try_exact_face", lambda *args, **kwargs: (None, None))
    return judged


@pytest.mark.parametrize("stop", ["tolerance", "plateau", "budget"])
def test_a_refused_cap_does_not_stop_the_free_terms(monkeypatch, stop: str) -> None:
    """A cap the search cannot promote is held while the other terms finish, then judged again.

    ``z`` starts on its cap and every promotion is refused, so the cap event is
    forced; ``x`` starts three log units from its optimum. The fit ends with ``x``
    where the fit with ``z`` fixed on the cap puts it, and reports the cap as
    unresolved after judging it at the returned state. With the budget spent
    first, the free term has not stopped: the fit says it ran out of rounds and
    names the cap it still holds. Mutation checks: returning at the first
    refusal leaves ``x`` at its start; without the release at the tolerance
    stop, or at the plateau stop, the fit reports ``converged=True`` with the
    cap never judged again.
    """
    reference, start = _cap_start()
    judged = _refuse_every_promotion(monkeypatch)
    options = {
        "tolerance": {},
        "plateau": {"practical_reml": True, "reml_tol": 1.0e-13, "max_log_step": 0.5},
        "budget": {"max_log_step": 0.5, "max_reml_iter": 3},
    }[stop]
    model, smoothing = _cap_fit(lambdas=start, **options)

    assert smoothing.converged is False
    assert smoothing.unresolved_upper_bound == (_CAPPED,)
    assert smoothing.lambdas[_CAPPED] == _CAP
    moved = math.log(start[_FREE] / smoothing.lambdas[_FREE])
    if stop == "budget":
        assert smoothing.convergence_reason == "max_iterations"
        assert len(smoothing.history) == 3
        assert len(judged) == 1
        assert moved == pytest.approx(1.5)
        return
    assert smoothing.convergence_reason == "lambda_cap_unresolved"
    assert judged == [start, dict(smoothing.lambdas)]
    assert moved == pytest.approx(3.0, abs=1.0e-6)
    fitted = reference._require_fitted().smoothing.terminal_fit.theta
    # To 1e-6 of the predictors' scale: a row near zero has no relative accuracy.
    scale = float(np.abs(fitted).max())
    np.testing.assert_allclose(
        smoothing.terminal_fit.theta, fitted, rtol=1.0e-6, atol=1.0e-6 * scale
    )
    assert model.result_.total_effective_df == pytest.approx(
        reference.result_.total_effective_df, rel=1.0e-6
    )


@pytest.mark.parametrize("accepted_rounds", [0, 2])
def test_a_search_that_cannot_step_still_reports_its_refused_cap(
    monkeypatch, accepted_rounds: int
) -> None:
    """When no trial improves the objective, a refused cap stays the reason for the stop.

    Every trial after ``accepted_rounds`` accepted rounds is scored far worse than
    the fit it starts from. The cap is judged again only when a step was accepted
    since its refusal. Mutation checks: without the release at this stop the
    reason is ``objective_rejected``; judging the cap twice at one fit makes two
    judgements where no round was accepted.
    """
    from superglm.distributional.smoothing import loop

    _reference, start = _cap_start()
    judged = _refuse_every_promotion(monkeypatch)
    objective, scored = loop._trial_laplace_objective, []

    def trial_objective(*args, **kwargs):
        scored.append(objective(*args, **kwargs))
        return scored[-1] + (1.0e9 if len(scored) > accepted_rounds else 0.0)

    monkeypatch.setattr(loop, "_trial_laplace_objective", trial_objective)
    _model, smoothing = _cap_fit(lambdas=start, max_log_step=0.5)

    assert smoothing.convergence_reason == "lambda_cap_unresolved"
    assert smoothing.unresolved_upper_bound == (_CAPPED,)
    assert [item.accepted for item in smoothing.history] == [True] * accepted_rounds + [False]
    assert len(judged) == (2 if accepted_rounds else 1)


def test_a_held_cap_is_promoted_where_the_free_terms_stop(monkeypatch) -> None:
    """The second judgement can put the cap on its exact face, and the fit converges there.

    The first promotion is refused by force and later ones are the search's own.
    ``z`` has no optimum short of infinity, so once ``x`` has stopped the search
    promotes it and ends where the unforced search ends. Mutation check: without
    the release at the tolerance stop the fit returns with no exact face.
    """
    from superglm.distributional.smoothing import loop

    _reference, start = _cap_start()
    unforced_model, unforced = _cap_fit(lambdas=start)
    assert unforced_model.exact_face_components_ == (_CAPPED,)
    joint, single, calls = loop._try_joint_exact_face, loop._try_exact_face, []

    def refused_once(assessor):
        def assess(*args, **kwargs):
            calls.append(assessor)
            return (None, None) if len(calls) <= 2 else assessor(*args, **kwargs)

        return assess

    monkeypatch.setattr(loop, "_try_joint_exact_face", refused_once(joint))
    monkeypatch.setattr(loop, "_try_exact_face", refused_once(single))
    model, smoothing = _cap_fit(lambdas=start)

    assert smoothing.converged is True
    assert smoothing.unresolved_upper_bound == ()
    assert model.exact_face_components_ == (_CAPPED,)
    assert smoothing.objective == pytest.approx(unforced.objective, rel=1.0e-9)
    assert smoothing.lambdas[_FREE] == pytest.approx(unforced.lambdas[_FREE], rel=1.0e-6)


def test_a_result_names_every_cap_still_pressed(monkeypatch) -> None:
    """A fit that runs out of rounds as a second term reaches its cap reports both caps.

    ``z`` is refused on its cap at the start and held. ``w`` starts one log unit
    below the cap and reaches it in the budget's two rounds of half a log unit,
    so the fit returns without a round to judge either. Mutation check: a result
    built from the evidence that leaves held caps out names ``w`` alone.
    """
    _reference, start = _cap_start()
    _refuse_every_promotion(monkeypatch)
    second = "location:w#wiggle"
    _model, smoothing = _cap_fit(
        second_weak=True,
        lambdas={**start, second: _CAP * math.exp(-1.0)},
        max_log_step=0.5,
        max_reml_iter=2,
    )

    assert smoothing.convergence_reason == "lambda_cap_unresolved"
    assert len(smoothing.history) == 2
    assert smoothing.lambdas[second] == _CAP
    assert smoothing.unresolved_upper_bound == (_CAPPED, second)


def test_a_held_cap_evidence_follows_the_sign_of_its_step(monkeypatch) -> None:
    """A held cap stays out of the stop evidence only while its raw step is not inward.

    The evidence is rebuilt at the cap fit's first state, ``x`` at its optimum,
    with ``z``'s raw step and proposal kind scripted. Held with an outward step, a working
    infinity or a step inside the tolerance, ``z`` is no part of the residual,
    the cap pressure or the working infinities. With an inward step it is
    ordinary evidence again, as if it had never been held. Mutation checks:
    holding whatever the sign keeps ``z`` out of the residual at an inward step;
    holding a name that is not on its cap drops ``x`` from the residual; leaving
    a held name in the residual, the working infinities or the cap pressure
    fails the first three cases.
    """
    from superglm.distributional.smoothing import evidence as smoothing_evidence
    from superglm.distributional.smoothing import loop

    states = []
    rebuild = loop._fresh_raw_evidence

    def capture(layout, lambdas, fit, config, **kwargs):
        states.append((layout, dict(lambdas), fit, config))
        return rebuild(layout, lambdas, fit, config, **kwargs)

    monkeypatch.setattr(loop, "_fresh_raw_evidence", capture)
    _reference, start = _cap_start()
    states.clear()
    _refuse_every_promotion(monkeypatch)
    _cap_fit(lambdas={**start, _FREE: start[_FREE] * math.exp(-3.0)}, max_reml_iter=1)
    layout, lambdas, fit, config = states[0]
    assert lambdas[_CAPPED] == config.upper(_CAPPED)
    update = smoothing_evidence.wood_fasiolo_update
    tolerance = config.tolerance

    def scripted(raw_step: float, kind: str):
        def scripted_update(*args, **kwargs):
            real = update(*args, **kwargs)
            return replace(
                real,
                raw_log_steps={**real.raw_log_steps, _CAPPED: raw_step},
                stationarity_log_residuals={
                    **real.stationarity_log_residuals,
                    _CAPPED: math.copysign(min(abs(raw_step), 1.0e-3), raw_step),
                },
                proposal_kinds={**real.proposal_kinds, _CAPPED: kind},
            )

        return scripted_update

    def evidence_at(raw_step: float, kind: str, held: frozenset[str]):
        with monkeypatch.context() as patch:
            patch.setattr(smoothing_evidence, "wood_fasiolo_update", scripted(raw_step, kind))
            return rebuild(layout, lambdas, fit, config, held=held)

    for raw_step, kind in ((2.0, "gfs"), (700.0, "working_infinity"), (-0.5 * tolerance, "gfs")):
        held = evidence_at(raw_step, kind, frozenset({_CAPPED, _FREE}))
        assert held.cap_held == (_CAPPED,)  # ``x`` is not on its cap, so it is never held
        assert held.unresolved_upper_bound == () and held.working_infinity == ()
        assert held.maximum == abs(held.update.stationarity_log_residuals[_FREE])
    free = evidence_at(2.0, "gfs", frozenset())
    assert free.unresolved_upper_bound == (_CAPPED,) and free.maximum == 2.0
    for held_names in (frozenset({_CAPPED}), frozenset()):
        inward = evidence_at(-0.3, "gfs", held_names)
        assert inward.cap_held == ()
        assert inward.unresolved_upper_bound == () and inward.maximum == 0.3


_GROUP, _SPREAD = "location:g#wiggle", "scale:h#wiggle"


def _coupled_fit(**kwargs):
    """Group effects that only a correct noise level reveals.

    Twelve groups of ten rows have effects of spread 0.5 under noise 0.3, and
    as many ungrouped rows have noise 5. With one pooled noise level the group
    means look quieter than chance, so their penalty wants infinity; once the
    scale's own effect separates the two noise levels, their optimum is finite.
    """
    rng = np.random.default_rng(5)
    groups, per = 12, 10
    grouped = groups * per
    g = np.r_[np.repeat([f"g{i:02d}" for i in range(groups)], per), np.repeat("rest", grouped)]
    h = np.r_[np.repeat("low", grouped), np.repeat("high", grouped)]
    effect = np.r_[np.repeat(rng.normal(scale=0.5, size=groups), per), np.zeros(grouped)]
    y = 1.0 + effect + np.where(h == "low", 0.3, 5.0) * rng.normal(size=2 * grouped)
    model = model_from_templates(
        family=GaussianLS(),
        predictors=(
            Predictor("location", {"g": RandomEffect()}),
            Predictor("scale", {"h": RandomEffect()}),
        ),
    )
    model.fit_reml(
        pd.DataFrame({"g": g, "h": h}),
        y,
        max_lambda=1.0e3,
        reml_tol=1.0e-8,
        inner_tol=1.0e-10,
        practical_reml=False,
        outer="efs",
        **kwargs,
    )
    return model, model._require_fitted().smoothing


@pytest.mark.parametrize("promotion", ["refused by force", "the search's own"])
def test_a_held_cap_is_released_when_the_free_term_makes_its_optimum_interior(
    monkeypatch, promotion: str
) -> None:
    """A held cap takes its inward step once the other term's move gives it a finite optimum.

    Both penalties start on the cap. There the group penalty's raw step points
    outward and its promotion is refused (by force, or by the search's own joint
    judgement, which the scale penalty's inward step fails). The scale penalty
    then moves in, the group penalty's step turns inward, and the fit ends at the
    interior optimum an uncapped start reaches, reporting no cap. Mutation
    checks: returning at the first refusal reports the cap at the start; holding
    whatever the sign leaves the group penalty out of the residual at its first
    inward step.
    """
    from superglm.distributional.smoothing import loop

    reference_model, reference = _coupled_fit(initial_lambda=1.0)
    assert reference.converged and reference.unresolved_upper_bound == ()
    if promotion == "refused by force":
        _refuse_every_promotion(monkeypatch)
    seen = []
    rebuild = loop._fresh_raw_evidence

    def record(layout, lambdas, fit, config, **kwargs):
        evidence = rebuild(layout, lambdas, fit, config, **kwargs)
        if _GROUP in kwargs.get("held", ()) and lambdas[_GROUP] == config.upper(_GROUP):
            seen.append((evidence.update.raw_log_steps[_GROUP], evidence))
        return evidence

    monkeypatch.setattr(loop, "_fresh_raw_evidence", record)
    model, smoothing = _coupled_fit(lambdas={_GROUP: 1.0e3, _SPREAD: 1.0e3})

    assert smoothing.converged is True
    assert smoothing.unresolved_upper_bound == ()
    for name in (_GROUP, _SPREAD):
        assert smoothing.lambdas[name] == pytest.approx(reference.lambdas[name], rel=1.0e-5)
    assert smoothing.objective == pytest.approx(reference.objective, rel=1.0e-9)
    assert model.result_.total_effective_df == pytest.approx(
        reference_model.result_.total_effective_df, rel=1.0e-6
    )
    (outward, held), (inward, released) = seen
    assert outward > 0.0 and held.cap_held == (_GROUP,)
    assert inward < -smoothing.config.tolerance
    assert released.cap_held == () and released.maximum >= -inward


def test_a_face_the_other_term_makes_wrong_is_released_and_the_fit_converges() -> None:
    """A penalty on its exact face is released once the other term's move gives it a finite optimum.

    The group penalty starts on the cap beside a scale penalty of 300. There
    its endpoint derivative is positive, and the search puts it on the exact
    face. The scale penalty then moves in, and where it stops the group
    penalty's endpoint derivative is negative: infinity is no longer its
    optimum. The search releases the face and ends at the interior optimum an
    uncapped start reaches, with no exact face. Mutation check: a search that
    ends at that recheck reports ``endpoint_revalidation_failed`` with the
    group penalty left on its cap, at 3.3 effective degrees of freedom
    against 14.3.
    """
    reference_model, reference = _coupled_fit(initial_lambda=1.0)
    assert reference.converged and reference_model.exact_face_components_ == ()
    model, smoothing = _coupled_fit(lambdas={_GROUP: 1.0e3, _SPREAD: 300.0})

    promotions = [item for item in smoothing.history if item.activated_face_components]
    releases = [item for item in smoothing.history if item.deactivated_face_components]
    assert [item.activated_face_components for item in promotions] == [(_GROUP,)]
    assert [item.deactivated_face_components for item in releases] == [(_GROUP,)]
    assert promotions[0].endpoint_direction_evidence.lower_bound > 0.0
    assert releases[0].endpoint_direction_evidence.decision == "finite"
    assert releases[0].endpoint_direction_evidence.upper_bound < 0.0
    assert smoothing.converged is True
    assert smoothing.unresolved_upper_bound == ()
    assert model.exact_face_components_ == ()
    for name in (_GROUP, _SPREAD):
        assert smoothing.lambdas[name] == pytest.approx(reference.lambdas[name], rel=1.0e-4)
    assert smoothing.objective == pytest.approx(reference.objective, rel=1.0e-9)
    assert model.result_.total_effective_df == pytest.approx(
        reference_model.result_.total_effective_df, rel=1.0e-5
    )


@pytest.mark.pins_behaviour
def test_a_fit_with_no_capped_term_never_holds_a_cap(monkeypatch) -> None:
    """Pin the search of a fit whose smoothing parameters stay inside their bounds.

    No promotion is judged, no evidence is rebuilt with a cap held, and the stop,
    the round count and the fitted smoothing parameter are those the search gave
    before a refused cap was held.
    """
    from superglm.distributional.smoothing import loop

    judged = _refuse_every_promotion(monkeypatch)
    held = []
    rebuild = loop._fresh_raw_evidence

    def record(*args, **kwargs):
        held.append(kwargs.get("held", frozenset()))
        return rebuild(*args, **kwargs)

    monkeypatch.setattr(loop, "_fresh_raw_evidence", record)
    smoothing = _start_fit(initial_lambda=1.0e-3, outer="efs")

    assert judged == [] and not any(held)
    assert smoothing.convergence_reason == "objective_plateau"
    assert len(smoothing.history) == 6
    assert smoothing.lambdas["location:x#wiggle"] == pytest.approx(2.775967034672, rel=1.0e-6)
    assert smoothing.objective == pytest.approx(271.937954313925, rel=1.0e-9)


def _two_piece_wide_fit() -> SuperLSS:
    """``TwoPieceNormalLSS`` on 20,000 rows with nine location terms, three of them tensors.

    Three tensor margins and the smooths of two covariates the response barely
    uses reach the cap while other penalties are still moving.
    """
    from benchmarks import release_differential

    rows = 20000
    frame, eta, rng = release_differential.make_data(rows, 5217)
    sign = np.where(rng.uniform(size=len(eta)) < 0.6, -1, 2)
    y = 1 + eta + sign * np.abs(rng.normal(0, 0.5, len(eta)))
    family = TwoPieceNormalLSS()
    model = SuperLSS(
        family,
        family.location(
            s("x1", kind="cr", k=8),
            cat("c1"),
            s("x2", kind="cr", k=6),
            re("c2"),
            ti("x1", "x2"),
            s("lin", kind="cr", k=10),
            s("noise", kind="cr", k=10),
            ti("x1", "lin"),
            ti("x2", "noise"),
        ),
        family.scale(s("x2", kind="cr", k=5), s("x1", kind="cr", k=6), cat("c1")),
        family.skew(),
        discrete=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit_reml(frame.iloc[:rows], y[:rows])


@pytest.mark.slow
def test_two_piece_wide_fit_is_one_answer_on_every_kernel() -> None:
    """A search with refused caps ends where its free terms end, whatever the BLAS kernel.

    Stopping at the first refused cap left this fit at 55.857 effective degrees
    of freedom on two OpenBLAS kernels and 56.057 on a third, with penalties still
    moving 1.27 log units a round. Holding the caps, the search runs until no
    step improves the objective and gives the reference below on Haswell,
    SkylakeX and SandyBridge, to nine digits in the degrees of freedom.
    """
    model = _two_piece_wide_fit()
    grid = pd.DataFrame(
        {
            "x1": np.linspace(0.5, 9.5, 7),
            "x2": np.linspace(-1.8, 1.8, 7),
            "c1": "a0",
            "c2": "g00",
            "lin": np.linspace(-1.5, 1.5, 7),
            "noise": np.linspace(0.1, 0.9, 7),
        }
    )
    link, se = model.predict_link(grid, se_fit=True)

    assert model.smoothing_convergence_reason_ == "lambda_cap_unresolved"
    assert model.result_.total_effective_df == pytest.approx(56.813074895, rel=1.0e-6)
    reference = {
        "location": (
            [0.127143667996, 0.091858726260, 0.303324479440, 0.744350579349]
            + [1.240164531587, 1.595506543693, 1.773411116268],
            [0.038583486107, 0.029081212562, 0.027030440960, 0.026299219198]
            + [0.026970730556, 0.028730359629, 0.038882832422],
        ),
        "scale": (
            [-0.412954323122, -0.410695248933, -0.407977351141, -0.404369462375]
            + [-0.399562862067, -0.393955727274, -0.388001647662],
            [0.014975161881, 0.011939619642, 0.010299351784, 0.009845486928]
            + [0.010325905390, 0.011979631450, 0.015003969878],
        ),
        "skew": ([1.182453070781] * 7, [0.023224571821] * 7),
    }
    for parameter, (curve, error) in reference.items():
        np.testing.assert_allclose(link[parameter], curve, rtol=1.0e-6)
        np.testing.assert_allclose(se[parameter], error, rtol=1.0e-6)
