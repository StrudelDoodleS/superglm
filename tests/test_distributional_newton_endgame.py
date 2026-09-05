"""Newton-endgame primitives and their integration into the smoothing loop.

The fits budgets measured at the strict and practical EFS stops are recorded
in ``test_budget_is_no_worse_than_practical_efs``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from superglm.distributional import NegativeBinomialLS, Predictor, TweedieLSS
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.result import DenseSolverConfig
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.smoothing.newton import (
    bfgs_direction,
    bfgs_update,
    bracket_beyond_cap,
    newton_direction,
    should_hand_off,
)
from superglm.distributional.smoothing.objective import joint_laplace_objective
from superglm.distributional.weights import WeightContract
from superglm.features import Spline
from tests._laml_oracle import probe_fit
from tests.test_distributional_endpoint_direction import (
    _cap_start,
    _GammaWithoutDirection,
    _linear_x_fixture,
    _mean_predictor,
    _tweedie_response,
)
from tests.test_distributional_laml_derivatives import _interior_case, _interior_data


def test_newton_direction_is_descent_and_capped() -> None:
    g = np.array([0.4, -0.2, 0.05])
    # indefinite third direction
    h = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, -0.5]])
    step, ridge = newton_direction(g, h, np.array([True, True, True]), max_log_step=5.0)
    assert ridge > 0.0  # perturbed to positive definite
    assert float(g @ step) < 0.0  # descent
    assert np.max(np.abs(step)) <= 5.0
    step2, ridge2 = newton_direction(g[:2], h[:2, :2], np.array([True, True]), max_log_step=0.1)
    assert ridge2 == 0.0 and np.max(np.abs(step2)) == pytest.approx(0.1)  # proportional cap
    frozen, _ = newton_direction(g, h, np.array([True, False, True]), max_log_step=5.0)
    assert frozen[1] == 0.0


def test_bfgs_update_keeps_positive_definiteness_under_bad_curvature() -> None:
    B = np.eye(2)
    s = np.array([0.1, -0.05])
    yv = np.array([-0.3, 0.2])  # s'y < 0
    B2 = bfgs_update(B, s, yv)
    assert np.all(np.linalg.eigvalsh(B2) > 0.0)
    step = bfgs_direction(np.array([1.0, 1.0]), B2, np.array([True, True]), max_log_step=5.0)
    assert float(np.array([1.0, 1.0]) @ step) < 0.0


def test_should_hand_off_rules() -> None:
    cfg = DistributionalEFSConfig(outer="efs+newton", handoff_step=0.5, handoff_iterations=10)
    assert should_hand_off(None, max_accepted_step=0.4, iterations=2, config=cfg)
    assert not should_hand_off(None, max_accepted_step=2.0, iterations=2, config=cfg)
    assert should_hand_off(None, max_accepted_step=2.0, iterations=10, config=cfg)
    for reason in (
        "lambda_change",
        "objective_plateau",
        "practical_plateau",
        "objective_rejected",
        "max_iterations",
        "lambda_cap_unresolved",
    ):
        assert should_hand_off(reason, max_accepted_step=3.0, iterations=1, config=cfg)
    for reason in ("fixed_only", "coefficient_not_converged", "endpoint_revalidation_failed"):
        assert not should_hand_off(reason, max_accepted_step=0.0, iterations=1, config=cfg)
    assert not should_hand_off(
        "lambda_change",
        max_accepted_step=0.0,
        iterations=1,
        config=DistributionalEFSConfig(outer="efs"),
    )


def test_bracket_beyond_cap_finds_the_root_in_log_tau() -> None:
    # phi(u) = dF/dtau as a function of u = log(lambda/lambda_cap) in [0, log_span]; root at u = 2.3
    calls = []

    def phi(u):
        calls.append(u)
        return -(u - 2.3)

    out = bracket_beyond_cap(
        phi_at_cap=phi(0.0), phi_at_endpoint=-1.0, evaluate=phi, log_span=math.log(1e4)
    )
    assert out.found and abs(out.log_lambda_ratio - 2.3) < 1e-3 and len(calls) <= 13
    none = bracket_beyond_cap(
        phi_at_cap=0.5, phi_at_endpoint=0.2, evaluate=lambda u: 0.5, log_span=math.log(1e4)
    )
    assert not none.found and none.evaluations == 0


# --- The endgame inside the loop ---------------------------------------------


def _fit_smoothing(kind: str, *, outer: str, family=None, **efs_overrides):
    """Fit the interior fixture with the given outer method."""
    frame, y, default_family, predictors = _interior_data(kind)
    efs = {"max_iterations": 250, "practical_convergence": False, "outer": outer}
    efs.update(efs_overrides)
    model = fit_dense_distributional(
        frame,
        y,
        family=default_family if family is None else family,
        predictors=predictors,
        weight_contract=WeightContract("prior"),
        config=DenseSolverConfig(max_iterations=500, tolerance=1.0e-11),
        efs_config=DistributionalEFSConfig(**efs),
        retain_rows=True,
    )
    smoothing = model.smoothing
    assert smoothing is not None
    return smoothing


def _efs_stop(kind: str, **cfg):
    """Strict (or practical) EFS result at the Fellner-Schall fixed point, outer="efs"."""
    return _fit_smoothing(kind, outer="efs", **cfg)


def _newton_stop(kind: str, *, family=None, **cfg):
    """The same problem under outer="efs+newton"."""
    return _fit_smoothing(kind, outer="efs+newton", family=family, **cfg)


def _face_components(result) -> tuple[str, ...]:
    face = result.terminal_fit.coefficient_face
    return () if face is None else face.component_names


def _scipy_optimum(kind: str, start) -> dict[str, float]:
    """L-BFGS-B on the LAML objective with tight warm refits (the optimum_cost method)."""
    family, layout, y, plan, lambdas, fit, config, _session = _interior_case(kind, "fisher")
    names = list(lambdas)
    x0 = np.array([math.log(start[name]) for name in names])

    def unpack(x):
        return {name: float(math.exp(value)) for name, value in zip(names, x, strict=True)}

    def f(x):
        values = unpack(x)
        try:
            probe = probe_fit(family, layout, y, plan, values, fit, config)
        except RuntimeError:
            return 1.0e30
        return joint_laplace_objective(probe, layout=layout, lambdas=values)

    def g(x, h=0.01):
        out = np.empty_like(x)
        for i in range(x.size):
            central = []
            for step in (h, 0.5 * h):
                xp, xm = x.copy(), x.copy()
                xp[i] += step
                xm[i] -= step
                central.append((f(xp) - f(xm)) / (2.0 * step))
            out[i] = (4.0 * central[1] - central[0]) / 3.0
        return out

    res = minimize(
        f, x0, jac=g, method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-16, "gtol": 1e-10}
    )
    return unpack(res.x)


def _linear_effect_config(outer: str) -> tuple[DenseSolverConfig, DistributionalEFSConfig]:
    """The settings the endpoint tests fit with (``fit_reml`` values)."""
    return (
        DenseSolverConfig(max_iterations=150, tolerance=1.0e-9, coefficient_curvature="observed"),
        DistributionalEFSConfig(
            max_iterations=120, tolerance=1.0e-6, practical_convergence=False, outer=outer
        ),
    )


def _tweedie_linear_effect_fit(*, outer: str):
    frame, mu = _linear_x_fixture(seed=6)
    y = _tweedie_response(mu, seed=6)
    solver, efs = _linear_effect_config(outer)
    model = fit_dense_distributional(
        frame,
        y,
        family=TweedieLSS(),
        predictors=(_mean_predictor(), Predictor("dispersion", {}), Predictor("power", {})),
        weight_contract=WeightContract("prior"),
        lambdas=_cap_start(),
        config=solver,
        efs_config=efs,
        retain_rows=True,
    )
    smoothing = model.smoothing
    assert smoothing is not None
    return smoothing


def _nb2_linear_effect_fit(*, seed: int, outer: str):
    """NB2 cap-start fixture with n = 4,000 and an exactly linear ``x`` effect."""
    rng = np.random.default_rng(seed)
    n = 4000
    x = rng.uniform(-1.0, 1.0, n)
    w = rng.uniform(-1.0, 1.0, n)
    eta = 1.0 + 0.55 * np.sin(np.pi * w) + 0.15 * w + 0.3 * x
    mean = np.exp(eta)
    theta = 2.5
    y = rng.negative_binomial(theta, theta / (mean + theta)).astype(np.float64)
    frame = pd.DataFrame({"x": x, "w": w})
    solver, efs = _linear_effect_config(outer)
    model = fit_dense_distributional(
        frame,
        y,
        family=NegativeBinomialLS(),
        predictors=(
            Predictor(
                "mean", {"x": Spline(kind="cr", n_knots=5), "w": Spline(kind="cr", n_knots=5)}
            ),
            Predictor("theta", {}),
        ),
        weight_contract=WeightContract("prior"),
        lambdas=_cap_start(),
        config=solver,
        efs_config=efs,
        retain_rows=True,
    )
    smoothing = model.smoothing
    assert smoothing is not None
    return smoothing


@pytest.mark.parametrize("kind", ["gaussian", "gamma"])
def test_endgame_reaches_the_optimum_scipy_reaches(kind) -> None:
    efs = _efs_stop(kind)
    # matched precision: scipy runs to gtol 1e-10, so the engine runs at 1e-9 here
    newton = _newton_stop(kind, tolerance=1.0e-9)
    assert newton.converged and newton.convergence_reason == "stationary"
    assert newton.objective < efs.objective
    score_scale = 1.0 + abs(newton.objective)
    assert newton.terminal_projected_gradient_norm <= 1e-6 * score_scale
    # scipy L-BFGS-B on the finite-difference gradient from the EFS stop (the optimum_cost method)
    reference = _scipy_optimum(kind, start=efs.lambdas)
    for name in newton.lambdas:
        assert abs(math.log(newton.lambdas[name]) - math.log(reference[name])) < 1e-4, name
    newton_steps = [it for it in newton.history if it.stage == "newton"]
    assert 1 <= len(newton_steps) <= 8
    drops = [abs(it.objective_before - it.objective_after) for it in newton_steps if it.accepted]
    assert all(later <= earlier for earlier, later in zip(drops, drops[1:], strict=False))


@pytest.mark.parametrize("start", [1e-4, 0.1, 1e3])
def test_start_independence(start) -> None:
    result = _newton_stop("gamma", initial_lambda=start, tolerance=1.0e-9)
    reference = _newton_stop("gamma", initial_lambda=0.1, tolerance=1.0e-9)
    assert result.convergence_reason == "stationary"
    for name in result.lambdas:
        assert abs(math.log(result.lambdas[name]) - math.log(reference.lambdas[name])) < 1e-4


def test_flat_smooth_is_certified_not_capped() -> None:
    # Tweedie linear effect at a cap start: exact face, converged, no unresolved bound
    result = _tweedie_linear_effect_fit(outer="efs+newton")
    assert result.converged and not result.unresolved_upper_bound
    assert "mean:x#wiggle" in _face_components(result)
    # These NB2 samples ended lambda_cap_unresolved under the fixed-point route;
    # the seeds pin the exact-face regression.
    for seed in (2, 3):
        nb = _nb2_linear_effect_fit(seed=seed, outer="efs+newton")
        assert nb.convergence_reason in {"stationary", "lambda_change"} or _face_components(nb)
        assert nb.convergence_reason != "lambda_cap_unresolved"
        # the cap never certifies: a component resting at maximum_lambda without an
        # endpoint decision is stationary at a box bound, not a certified infinity
        at_cap = [
            name
            for name, value in nb.lambdas.items()
            if value == nb.config.maximum_lambda and name not in _face_components(nb)
        ]
        if at_cap:
            assert not nb.matched_certified, (seed, at_cap)


def test_budget_is_no_worse_than_practical_efs() -> None:
    """Coefficient fits, measured on the interior fixtures (inner 1e-11, outer 1e-6).

    Gaussian: strict EFS 17 fits (objective_plateau, F = 2385.075338603),
    practical EFS 10 fits (F = 2385.075333326), efs+newton 5 fits -- two EFS
    warm-up iterations, two Newton steps with drops 9.1e-4 and 3.6e-8 -- at
    F = 2385.074519794, the value scipy's L-BFGS-B reaches to 1e-9.
    Gamma: strict 12, practical 8, efs+newton 5 fits (drops 3.5e-3, 7.9e-7),
    F = 1538.766716693 against the strict stop's 1538.768410418.
    """
    for kind in ("gaussian", "gamma"):
        practical = _efs_stop(kind, practical_convergence=True)
        newton = _newton_stop(kind)
        # +2: the derivative-pass refits at the hand-off
        assert len(newton.coefficient_fits) <= len(practical.coefficient_fits) + 2, kind
        assert newton.objective <= practical.objective + 1e-9 * (1 + abs(practical.objective))


def test_stationary_results_are_validated_on_replay() -> None:
    newton = _newton_stop("gaussian")
    with pytest.raises(ValueError, match="projected gradient"):
        replace(newton, terminal_projected_gradient_norm=1.0)
    with pytest.raises(ValueError, match="stationary"):
        replace(
            newton, convergence_reason="lambda_change"
        )  # FS evidence is not fresh at the optimum
    # the FS residual is NOT zero at the REML optimum
    assert newton.terminal_raw_max_log_step > newton.config.tolerance


def test_gradient_unresolved_when_certificate_dominates() -> None:
    """A finite-difference step so small the certificate exceeds the stationarity bar.

    The certificate is finite-difference round-off, so the family must take the
    finite-difference path (GammaLS's analytic third derivative carries a zero
    certificate at any step).  Measured on this fixture: the gradient
    certificate passes ``tol * (1 + |F|) = 1.5e-3`` only at ``step = 1e-14``
    (2.9e-3; it is 3.4e-8 at ``step = 1e-9``).
    """
    result = _newton_stop("gamma", family=_GammaWithoutDirection(), derivative_step=1e-14)
    assert not result.converged and result.convergence_reason == "gradient_unresolved"
    assert max(result.terminal_gradient_certificate.values()) > 0.0


def test_bfgs_fallback_when_the_hessian_certificate_fails() -> None:
    """At ``step = 1e-9`` the Hessian certificate (2.1e2) dwarfs a tenth of its
    smallest diagonal (7.6e-2) while the gradient certificate stays 3.4e-8: the
    iteration must step by damped BFGS and still reach the optimum."""
    result = _newton_stop("gamma", family=_GammaWithoutDirection(), derivative_step=1e-9)
    assert result.converged and result.convergence_reason == "stationary"
    assert result.bfgs_fallback_iterations >= 1
    assert any(item.step_source == "bfgs" for item in result.history)
    reference = _newton_stop("gamma")
    for name in result.lambdas:
        assert abs(math.log(result.lambdas[name]) - math.log(reference.lambdas[name])) < 1e-3


# --- Derivative-pass cost: gradient first, Hessian on demand ------------------


def _derivative_pass_spy(monkeypatch) -> list[bool]:
    """Record every ``laml_derivatives`` call the endgame makes, as its ``want_hessian`` flag."""
    from superglm.distributional.smoothing import newton as newton_module

    calls: list[bool] = []
    real = newton_module.laml_derivatives

    def spy(*args, **kwargs):
        calls.append(bool(kwargs.get("want_hessian", True)))
        return real(*args, **kwargs)

    monkeypatch.setattr(newton_module, "laml_derivatives", spy)
    return calls


def test_endgame_forms_a_hessian_only_when_it_steps(monkeypatch) -> None:
    """Every pass evaluates the gradient; a Hessian is formed only to step, never to stop.

    A gradient-only pass leaves ``hessian_certificate`` ``None`` on its record; the terminal
    convergence check makes no trial fit, so it leaves no record at all (a
    ``DistributionalEFSIteration`` is one proposed update plus the refits that judged it) and
    the published smoothing Hessian is absent: it is exact at the terminal point or not there.
    """
    calls = _derivative_pass_spy(monkeypatch)
    newton = _newton_stop("gaussian")
    assert newton.convergence_reason == "stationary"
    passes = [it for it in newton.history if it.stage == "newton"]
    with_hessian = [it for it in passes if it.hessian_certificate is not None]
    # a Hessian is formed for a step, never for a convergence check
    assert len(with_hessian) <= len([it for it in passes if it.accepted])
    assert (with_hessian and with_hessian[0] is passes[0]) or passes[0].hessian_certificate is None
    # the first and the last derivative pass are gradient-only; every Hessian pass left a record
    assert calls[0] is False and calls[-1] is False
    assert sum(calls) == len(with_hessian)
    assert newton.smoothing_hessian is None


def test_stationary_handoff_never_forms_a_hessian(monkeypatch) -> None:
    """A hand-off point that already satisfies Wood's criterion ends stationary after one
    gradient pass: no Hessian, no step, no Newton record (a record needs a trial fit)."""
    calls = _derivative_pass_spy(monkeypatch)
    handed = _newton_stop("gaussian", tolerance=1.0e-3)  # loose bar: the warm-up lands inside it
    assert handed.convergence_reason == "stationary"
    assert calls == [False]
    assert handed.newton_iterations == 0 and handed.smoothing_hessian is None
    assert all(it.stage == "efs" for it in handed.history)
