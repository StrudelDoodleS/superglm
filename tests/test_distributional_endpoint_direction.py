# tests/test_distributional_endpoint_direction.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm.distributional import GammaLS, Predictor, TweedieLSS
from superglm.distributional.derivatives import transform_natural_derivatives
from superglm.distributional.endpoint_direction import (
    DEFAULT_STEP,
    FiniteDifferenceDirection,
    finite_difference_curvature_direction,
)
from superglm.distributional.endpoint_laml import EndpointLaplaceDerivative
from superglm.distributional.families.gamma import GammaLS as _GammaLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.result import (
    EndpointDirectionEvidence,
    JointEndpointDirectionEvidence,
)
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import Spline
from superglm.links import LogLink


def _gamma_rows(n: int = 810, seed: int = 5):
    rng = np.random.default_rng(seed)
    eta = np.column_stack((rng.uniform(-3.0, 3.0, n), rng.uniform(-2.5, 1.5, n)))
    mean = np.exp(eta[:, 0])
    cv = np.exp(eta[:, 1])
    y = rng.gamma(1.0 / cv**2, mean * cv**2)
    # extremes: tiny and huge responses relative to the mean, huge scale
    y[:30] = mean[:30] * 1.0e-8
    y[30:60] = mean[30:60] * 50.0
    eta[60:90, 1] = 4.0
    direction = rng.normal(size=eta.shape)
    return y, eta, direction


def _gamma_plan(y):
    weights = resolve_likelihood_weights(
        None, n_observations=len(y), contract=WeightContract("prior")
    )
    return GammaLS().bind_likelihood(y, weights, COMPLETE_OBSERVATION)


def test_finite_difference_matches_the_analytic_gamma_direction() -> None:
    family = GammaLS()
    y, eta, direction = _gamma_rows()
    plan = _gamma_plan(y)
    links = (LogLink(), LogLink())
    analytic = np.asarray(
        family.predictor_curvature_directional_derivative(y, eta, direction, links, plan)
    )
    numeric = finite_difference_curvature_direction(family, y, eta, direction, links, plan)
    assert isinstance(numeric, FiniteDifferenceDirection)
    assert numeric.values.shape == analytic.shape == (len(y), 3)
    assert numeric.evaluations == 4
    scale = np.max(np.abs(analytic), axis=1, keepdims=True) + 1.0e-300
    relative = np.abs(numeric.values - analytic) / scale
    assert np.max(relative) < 1.0e-9
    assert np.median(relative) < 1.0e-11


def test_certificate_bounds_the_actual_error() -> None:
    """The Richardson certificate bounds |FD - analytic| up to finite-difference round-off.

    ``certificate = |D4 - D2(step/2)|`` is a truncation estimate: it cannot see
    the family's own evaluation round-off, which a central difference amplifies
    by ``1/step``.  The floor is therefore that round-off at the row's curvature
    scale, ``64 eps max|curvature_row| ||d|| / step``, not per entry: the
    (mean, mean) channel at ``y = 1e-8 mu`` is ``alpha y / mu`` computed as the
    difference of two O(alpha) terms, so it carries the row's round-off, not
    the entry's.  At the default step truncation and round-off are balanced by
    design, so the bound is asserted one decade coarser, where the certificate
    is the operative term: measured over five seeds it bounds the error by
    itself in every entry above the floor (worst ratio 0.03) and the excess
    beyond it stays under 0.7 of the floor.
    """
    step = 10.0 * DEFAULT_STEP
    family = GammaLS()
    y, eta, direction = _gamma_rows(seed=9)
    plan = _gamma_plan(y)
    links = (LogLink(), LogLink())
    analytic = np.asarray(
        family.predictor_curvature_directional_derivative(y, eta, direction, links, plan)
    )
    numeric = finite_difference_curvature_direction(
        family, y, eta, direction, links, plan, step=step
    )
    theta = np.column_stack([link.inverse(eta[:, index]) for index, link in enumerate(links)])
    curvature = transform_natural_derivatives(
        family.evaluate_natural(y, theta, plan, derivative_order=2), eta, links
    ).curvature_packed
    row_scale = np.max(np.abs(curvature), axis=1) * np.linalg.norm(direction, axis=1)
    floor = 64.0 * np.finfo(np.float64).eps * row_scale[:, None] / step
    error = np.abs(numeric.values - analytic)
    assert np.all(error <= numeric.certificate + floor)
    operative = numeric.certificate > floor
    assert np.all(error[operative] <= numeric.certificate[operative])


def test_direction_scale_invariance() -> None:
    family = GammaLS()
    y, eta, direction = _gamma_rows(seed=13, n=200)
    plan = _gamma_plan(y)
    links = (LogLink(), LogLink())
    base = finite_difference_curvature_direction(family, y, eta, direction, links, plan).values
    scaled = finite_difference_curvature_direction(
        family, y, eta, 1.0e3 * direction, links, plan
    ).values
    assert np.allclose(scaled, 1.0e3 * base, rtol=1.0e-11, atol=0.0)


def test_non_finite_inputs_fail_closed() -> None:
    family = TweedieLSS()
    rng = np.random.default_rng(1)
    n = 50
    y = rng.gamma(2.0, 1.0, n)
    eta = np.zeros((n, 3))
    weights = resolve_likelihood_weights(None, n_observations=n, contract=WeightContract("prior"))
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    links = tuple(p.default_link for p in family.parameters)
    direction = rng.normal(size=eta.shape)
    direction[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        finite_difference_curvature_direction(family, y, eta, direction, links, plan)
    with pytest.raises(ValueError, match="step"):
        finite_difference_curvature_direction(
            family, y, eta, np.abs(direction) + 1.0, links, plan, step=0.0
        )


def test_invalid_perturbed_rows_fail_closed() -> None:
    """A family that flags a perturbed row invalid must make the derivative refuse."""

    class _FlaggingGamma(GammaLS):
        def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
            evaluation = super().evaluate_natural(y, theta, plan, derivative_order=derivative_order)
            valid = np.ones(len(y), dtype=bool)
            valid[0] = False
            from dataclasses import replace as _replace

            return _replace(evaluation, valid=valid)

    y, eta, direction = _gamma_rows(n=40, seed=2)
    plan = _gamma_plan(y)
    with pytest.raises(ValueError, match="invalid rows"):
        finite_difference_curvature_direction(
            _FlaggingGamma(), y, eta, direction, (LogLink(), LogLink()), plan
        )


def _evidence(authority: str, derivative: float = 2.0) -> EndpointDirectionEvidence:
    return EndpointDirectionEvidence(
        authority_identifier=authority,
        decision="endpoint",
        endpoint_objective=10.0,
        analytic_derivative=derivative,
        profile_score_term=0.0,
        curvature_schur_term=4.0,
        curvature_drift_term=0.0,
        numerical_error=1.0e-6,
        lower_bound=derivative - 1.0e-6,
        upper_bound=derivative + 1.0e-6,
    )


def test_evidence_accepts_the_finite_difference_authority() -> None:
    evidence = _evidence("finite-difference-curvature-direction/v1")
    assert evidence.decision == "endpoint"
    with pytest.raises(ValueError, match="authority"):
        _evidence("some-other-authority/v1")


def test_joint_evidence_authority_follows_its_components() -> None:
    analytic = _evidence("analytic-observed-curvature-direction/v1")
    numeric = _evidence("finite-difference-curvature-direction/v1")
    joint = JointEndpointDirectionEvidence(
        authority_identifier="joint-finite-difference-curvature-direction/v1",
        component_directions=(("a", analytic), ("b", numeric)),
    )
    assert joint.authority_identifier == "joint-finite-difference-curvature-direction/v1"
    with pytest.raises(ValueError, match="authority"):
        JointEndpointDirectionEvidence(
            authority_identifier="joint-analytic-observed-curvature-direction/v1",
            component_directions=(("a", analytic), ("b", numeric)),
        )


def test_laplace_derivative_accepts_the_finite_difference_authority() -> None:
    derivative = EndpointLaplaceDerivative(
        authority_identifier="finite-difference-curvature-direction/v1",
        decision="endpoint",
        derivative=2.0,
        profile_score_term=0.0,
        curvature_schur_term=4.0,
        curvature_drift_term=0.0,
        numerical_error=1.0e-6,
        lower_bound=2.0 - 1.0e-6,
        upper_bound=2.0 + 1.0e-6,
    )
    assert derivative.decision == "endpoint"


class _GammaWithoutDirection(_GammaLS):
    """GammaLS with the analytic protocol hidden, so the engine must fall back."""

    predictor_curvature_directional_derivative = None  # type: ignore[assignment]


def _linear_x_fixture(n: int = 1000, seed: int = 4):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    w = rng.uniform(-1.0, 1.0, n)
    eta = 0.4 + 0.3 * x + 0.55 * np.sin(np.pi * w) + 0.15 * w
    return pd.DataFrame({"x": x, "w": w}), np.exp(eta)


def _gamma_response(mean, seed: int = 4):
    rng = np.random.default_rng(seed + 100)
    return rng.gamma(4.0, mean / 4.0)


def _tweedie_response(mu, seed: int = 4, phi: float = 0.8, power: float = 1.5):
    rng = np.random.default_rng(seed + 200)
    lam = mu ** (2.0 - power) / (phi * (2.0 - power))
    alpha = (2.0 - power) / (power - 1.0)
    scale = phi * (power - 1.0) * mu ** (power - 1.0)
    counts = rng.poisson(lam)
    y = np.zeros(len(mu))
    positive = counts > 0
    y[positive] = rng.gamma(alpha * counts[positive], scale[positive])
    return y


def _mean_predictor():
    return Predictor("mean", {"x": Spline(kind="cr", n_knots=5), "w": Spline(kind="cr", n_knots=5)})


def _cap_start():
    return {"mean:x#wiggle": 1.0e10, "mean:w#wiggle": 0.5}


def test_gamma_decisions_agree_between_analytic_and_finite_difference() -> None:
    from superglm import SuperLSS

    frame, mean = _linear_x_fixture()
    y = _gamma_response(mean)
    outcomes = {}
    for label, family in (("analytic", GammaLS()), ("fd", _GammaWithoutDirection())):
        model = SuperLSS(family=family, predictors=(_mean_predictor(), Predictor("scale", {})))
        model.fit_reml(frame, y, lambdas=_cap_start(), practical_reml=False)
        smoothing = model._require_fitted().smoothing
        outcomes[label] = (
            smoothing.convergence_reason,
            model.exact_face_components_,
            round(smoothing.lambdas["mean:w#wiggle"], 3),
        )
    assert outcomes["analytic"] == outcomes["fd"]
    assert outcomes["fd"][1] == ("mean:x#wiggle",)


def test_tweedie_certifies_a_genuine_infinity_through_finite_differences() -> None:
    """TweedieLSS has no analytic direction; the finite-difference fallback certifies.

    With ``fit_reml`` defaults the same fit certifies the same face with the
    same evidence but stops on ``objective_plateau`` rather than
    ``lambda_change``.
    """
    from superglm import SuperLSS

    frame, mu = _linear_x_fixture(seed=6)
    y = _tweedie_response(mu, seed=6)
    for practical in (False, True):
        model = SuperLSS(
            family=TweedieLSS(),
            predictors=(_mean_predictor(), Predictor("dispersion", {}), Predictor("power", {})),
        )
        model.fit_reml(
            frame,
            y,
            lambdas=_cap_start(),
            practical_reml=practical,
            max_reml_iter=120,
            reml_tol=1.0e-6,
            max_inner_iter=150,
            inner_tol=1.0e-9,
            outer="efs",
        )
        smoothing = model._require_fitted().smoothing
        assert smoothing.converged is True
        assert smoothing.convergence_reason == "lambda_change"
        assert model.exact_face_components_ == ("mean:x#wiggle",)
        evidence = smoothing.terminal_endpoint_directions["mean:x#wiggle"]
        assert evidence.authority_identifier == "finite-difference-curvature-direction/v1"
        assert evidence.decision == "endpoint"
        assert evidence.numerical_error < 0.01 * abs(evidence.analytic_derivative)


def test_a_finite_optimum_is_not_certified() -> None:
    """A genuinely wiggly ``x`` effect must leave the cap through the fallback.

    At amplitude 0.2 the analytic and finite-difference fits agree on a finite
    optimum. A smaller amplitude on this seeded sample can legitimately place
    the optimum at infinity and therefore cannot test this refusal.
    """
    from superglm import SuperLSS

    rng = np.random.default_rng(21)
    n = 1000
    x = rng.uniform(-1.0, 1.0, n)
    w = rng.uniform(-1.0, 1.0, n)
    eta = 0.4 + 0.2 * np.sin(np.pi * x) + 0.55 * np.sin(np.pi * w) + 0.15 * w
    y = _gamma_response(np.exp(eta), seed=21)
    frame = pd.DataFrame({"x": x, "w": w})
    model = SuperLSS(
        family=_GammaWithoutDirection(),
        predictors=(_mean_predictor(), Predictor("scale", {})),
    )
    model.fit_reml(frame, y, lambdas=_cap_start(), practical_reml=False)
    smoothing = model._require_fitted().smoothing
    assert model.exact_face_components_ == ()
    assert smoothing.lambdas["mean:x#wiggle"] < 1.0e10
    refusals = [item.endpoint_assessment_failure_reason for item in smoothing.history]
    assert "analytic_unavailable" not in refusals
