"""Binomial/log fits stay inside the binomial mean space.

The log link's inverse ``exp(eta)`` covers ``(0, inf)``, wider than the
binomial probabilities: a row with ``eta >= 0`` has no binomial likelihood, so
the parameter space is ``{beta : X beta + offset < 0}`` and the maximum can lie
on its boundary (Wacholder 1986; Marschner & Gillett 2012).  PIRLS rejects a
trial that leaves the space, as R's glm and glm2 step-halving do (Donoghoe &
Marschner 2018, J. Stat. Softw. 86(9), section 3.2), rather than letting
``clip_mu`` rewrite its means; a state holding rows at the boundary is never a
converged mode; and REML, whose Laplace approximation needs an interior
stationary maximum (Wood, Pya & Saefken 2016, section 3.1), restores an interior
start or says why none exists.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import (
    BSplineSmooth,
    Categorical,
    Constraint,
    LambdaPolicy,
    Numeric,
    RandomEffect,
    SuperGLM,
)
from superglm.diagnostics.separation import SeparationWarning
from superglm.distributions import Binomial, Gamma, Poisson
from superglm.links import CauchitLink, CloglogLink, LogitLink, LogLink, ProbitLink
from superglm.model.input_validation import FractionalFrequencyWeightWarning
from superglm.reml.observed_geometry import ObservedModeNotConvergedError
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.irls_state import (
    _mean_space_halving_budget,
    mean_space_boundary_rows,
    mean_space_clipped_rows,
    mean_space_score_rows,
    mean_space_violation,
)
from superglm.solvers.mode_score import MODE_CERTIFICATION_BAR, mode_certification_bar

_U = 2.0**-53  # unit roundoff, and the spacing of float64 just below one
_EPS = 2.0**-52  # machine epsilon, the spacing of float64 above one


def _thin_event_levels() -> tuple[pd.DataFrame, np.ndarray]:
    """A random effect beside a slope, plus four one-row levels whose row is an event.

    An event row's log-likelihood ``eta`` is linear, so its level's coefficient
    at a penalty ``lambda`` is ``1 / lambda`` and the row's mode is interior only
    while ``1 / lambda`` is below its distance ``|eta_0| ~ 2.3`` to the boundary:
    the near-unpenalized REML bootstrap (``lambda = 1e-4``) puts it there.
    """
    rng = np.random.default_rng(7)
    n, levels = 1500, 60
    g = rng.integers(0, levels, n)
    x = rng.uniform(size=n)
    eta = -2.3 + 0.4 * x + rng.normal(0.0, 0.3, levels)[g]
    y = (rng.uniform(size=n) < np.exp(eta)).astype(float)
    extra = pd.DataFrame({"x": rng.uniform(size=4), "g": [f"s{j}" for j in range(4)]})
    frame = pd.concat(
        [pd.DataFrame({"x": x, "g": [f"g{c:03d}" for c in g]}), extra], ignore_index=True
    )
    return frame, np.concatenate([y, np.ones(4)])


def _fit(direct_solve: str) -> SuperGLM:
    X, y = _thin_event_levels()
    model = SuperGLM(
        family="binomial",
        link="log",
        direct_solve=direct_solve,
        features={"x": Numeric(), "g": RandomEffect()},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        model.fit_reml(X, y)
    return model


def _eta(model: SuperGLM) -> np.ndarray:
    return model._dm.matvec(model.result.beta) + model.result.intercept


def test_only_a_link_that_leaves_the_mean_space_declares_a_boundary() -> None:
    """Declared by family and link: the binomial links that stay in ``(0, 1)`` declare none."""
    contained = [
        (Binomial(), LogitLink()),
        (Binomial(), ProbitLink()),
        (Binomial(), CloglogLink()),
        (Binomial(), CauchitLink()),
        (Poisson(), LogLink()),
        (Gamma(), LogLink()),
    ]
    for family, link in contained:
        assert mean_space_violation(family, link) is None
        assert mean_space_boundary_rows(family, link, np.array([5.0]), np.ones(1)) == 0
    violates = mean_space_violation(Binomial(), LogLink())
    assert violates is not None
    # eta = 0 is mu = 1: outside, unless the row carries no likelihood
    assert violates(np.array([0.0]), np.ones(1))
    assert not violates(np.array([-1e-300, 0.0]), np.array([1.0, 0.0]))
    # the boundary rows are the rows clip_mu caps, exp(eta) >= 1 - 1e-7
    cap = np.log1p(-1e-7)
    eta = np.array([cap, np.nextafter(cap, -np.inf), 0.0])
    assert mean_space_boundary_rows(Binomial(), LogLink(), eta, np.array([1.0, 1.0, 0.0])) == 1


def test_a_near_unpenalized_fit_stops_at_the_boundary_and_says_so() -> None:
    """At the bootstrap's ``lambda = 1e-4`` the event rows' maximum is on the boundary.

    Every accepted state keeps ``eta < 0`` on the rows with weight, and the
    state the fit returns is reported as the constrained point it is, not as a
    converged mode.
    """
    model = _fit("gram")
    X, y = _thin_event_levels()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, _ = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=np.ones(len(y)),
            family=Binomial(),
            link=LogLink(),
            groups=model._groups,
            lambda2={name: 1e-4 for name in model._reml_result.lambdas},
            reml_penalties=model._reml_penalties,
            direct_solve="gram",
            weight_semantics="prior",
        )
    eta = model._dm.matvec(result.beta) + result.intercept
    assert float(eta.max()) < 0.0
    assert not result.converged
    assert result.termination_reason == "mean_space_boundary"
    assert result.mean_space_boundary_rows == int(np.count_nonzero(eta >= np.log1p(-1e-7)))
    assert result.mean_space_boundary_rows >= 1


def test_reml_reaches_the_interior_maximum_on_both_backends() -> None:
    """The REML optimum's mode is interior: REML reaches it, certified, on structured and gram.

    The two backends stop within the REML stopping rule ``reml_tol (1 + |V|)``
    of the same optimum, so their objectives agree within twice it.
    """
    fits = {direct_solve: _fit(direct_solve) for direct_solve in ("auto", "gram")}
    for model in fits.values():
        assert model._reml_result.converged
        assert model.result.converged
        assert float(_eta(model).max()) < 0.0
    objectives = [float(model._reml_result.objective) for model in fits.values()]
    tolerance = float(fits["auto"]._reml_profile["reml_tol_resolved"])
    assert abs(objectives[0] - objectives[1]) <= 2.0 * tolerance * (1.0 + abs(objectives[0]))


def test_reml_says_why_a_maximum_on_the_boundary_has_no_criterion() -> None:
    """An unpenalized level whose rows are all events is at probability one at every lambda."""
    rng = np.random.default_rng(3)
    n = 3000
    region = rng.choice(["A", "B", "C"], n)
    p = np.where(region == "A", 0.30, np.where(region == "B", 0.10, 1.0))
    y = (rng.random(n) < p).astype(np.float64)
    X = pd.DataFrame(
        {
            "region": region,
            "x": rng.uniform(size=n),
            "g": [f"g{c}" for c in rng.integers(0, 40, n)],
        }
    )
    model = SuperGLM(
        family="binomial",
        link="log",
        features={"region": Categorical(base="first"), "x": Numeric(), "g": RandomEffect()},
    )
    with pytest.raises(ObservedModeNotConvergedError, match="no interior maximum"):
        model.fit_reml(X, y)


def _offset_fit(offset: float, *, plain: bool, direct_solve: str) -> SuperGLM:
    """Sol's #431 (c) fixture: an interior optimum, probabilities 0.16 to 0.69."""
    rng = np.random.default_rng(2)
    n = 300
    x = rng.uniform(-1.0, 1.0, n)
    g = np.tile(np.arange(30), 10)
    y = (rng.random(n) < 0.4).astype(np.float64)
    X = pd.DataFrame({"x": x, "g": [f"g{c:02d}" for c in g]})
    level = (
        Categorical(base="first") if plain else RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0))
    )
    model = SuperGLM(
        family="binomial",
        link="log",
        direct_solve=direct_solve,
        selection_penalty=0.0,
        features={"x": Numeric(), "g": level},
    )
    if plain:
        model.fit(X, y, offset=np.full(n, offset))
    else:
        model.fit_reml(X, y, offset=np.full(n, offset))
    return model


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_a_positive_offset_starts_inside_the_mean_space(direct_solve: str) -> None:
    """A constant offset of +2 moves the intercept by -2, not the fit to the boundary (#431 c).

    The default intercept is chosen before the offset, so every row started
    at ``eta >= 0``, outside the space, and the fit ended "no interior
    maximum" (fit_reml) or unconverged at the boundary (fit).  The start is
    now lowered into the space, and the offset fit is a certified mode of the
    same problem: its REML criterion agrees with the no-offset fit's within
    the error the certificate allows each, ``reml_tol (1 + |V|)``, and a plain
    fit's deviance within the deviance stop's own ``tol |D|`` each.
    """
    for plain in (False, True):
        base = _offset_fit(0.0, plain=plain, direct_solve=direct_solve)
        shifted = _offset_fit(2.0, plain=plain, direct_solve=direct_solve)
        assert shifted.result.converged
        assert shifted.result.termination_reason == "converged"
        eta = shifted._dm.matvec(shifted.result.beta) + shifted.result.intercept + 2.0
        assert float(eta.max()) < 0.0
        if plain:
            deviances = [float(base.result.deviance), float(shifted.result.deviance)]
            bound = 2.0 * base._tol * max(abs(d) for d in deviances)
            assert abs(deviances[1] - deviances[0]) <= bound
        else:
            objectives = [float(base._reml_result.objective), float(shifted._reml_result.objective)]
            tolerance = float(base._reml_profile["reml_tol_resolved"])
            assert abs(objectives[1] - objectives[0]) <= 2.0 * tolerance * (
                1.0 + abs(objectives[0])
            )


def _two_level_fit(offset_a: float, offset_b: float, direct_solve: str) -> SuperGLM:
    """Two Categorical levels, one event and one non-event each: both MLE probabilities 1/2."""
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        features={"g": Categorical(base="first")},
    )
    model.fit(
        pd.DataFrame({"g": ["a", "a", "b", "b"]}),
        np.array([0.0, 1.0, 0.0, 1.0]),
        offset=np.array([offset_a, offset_a, offset_b, offset_b]),
        record_diagnostics=True,
    )
    return model


def _assert_at_the_two_level_mle(model: SuperGLM, offset_a: float, offset_b: float) -> None:
    """Converged, with both levels at ``log(1/2)`` within what the deviance stop resolves.

    The deviance stop is ``|D - D_prev| < tol (|D_prev| + 1)``
    (``_irls_objective_relative_change``).  Each level's deviance ``-2 eta - 2
    log(1 - e^eta)`` has curvature 4 at its maximum, where Fisher scoring's
    curvature equals the observed one (``2p/(1-p) = p/(1-p)^2`` at ``p =
    1/2``), so the iteration contracts at least by half near it and the excess
    left after the stop is at most the last change.  A level ``delta`` from its
    maximum carries an excess ``2 delta^2``, so ``delta <= sqrt(tol (|D_prev|
    + 1) / 2)``, within ``sqrt(tol |D|)`` at ``|D| = 8 log 2 >= 1``.
    """
    assert model.result.converged
    assert model.result.termination_reason == "converged"
    offset = np.array([offset_a, offset_a, offset_b, offset_b])
    eta = model._dm.matvec(model.result.beta) + model.result.intercept + offset
    bound = np.sqrt(model._tol * abs(float(model.result.deviance)))
    assert float(np.max(np.abs(eta - np.log(0.5)))) <= bound


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_a_lowered_start_reaches_a_trial_inside_the_space(direct_solve: str) -> None:
    """Sol's #437 fixture: the first step needs more than the ordinary 20 halvings.

    With offsets -16 and +1.3 the lowered start puts level a at eta ~ -18 and
    the Fisher proposal moves it by ~3e7, so only a fraction below ~6e-7 of
    the step stays inside ``eta < 0``: past 2^-20.  Backtracking reaches the
    fraction to the boundary instead of rejecting the step, and the fit
    reaches the maximum.
    """
    model = _two_level_fit(-16.0, 1.3, direct_solve)
    assert model.result.iteration_log[0].step_halvings > 20
    _assert_at_the_two_level_mle(model, -16.0, 1.3)


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
@pytest.mark.parametrize("depth", range(20, 41))
def test_a_level_below_the_clip_floor_is_not_a_converged_fit(direct_solve: str, depth: int) -> None:
    """Sol's #437 fixture over offsets -20 to -40: no stop on the flat clipped deviance.

    The lowered start puts level a below ``clip_mu``'s floor ``1e-7``, where
    the clipped deviance is flat in ``eta``.  A step that leaves the level
    there changed nothing the deviance stop can see, which reported a wrong
    fit as converged at -35 and -40 (probabilities ``1e-7`` and 1/2,
    deviance 35.0088).  A stop is accepted only where the binomial/log score
    certifies it, so the fit goes on to the maximum.  Master returns these
    fits unconverged from an infeasible start, with a "probability" of 1.83.
    """
    model = _two_level_fit(-float(depth), 1.3, direct_solve)
    _assert_at_the_two_level_mle(model, -float(depth), 1.3)


def _three_row_fit(y, offset, weights, direct_solve: str) -> SuperGLM:
    """Sol's #437 fixture: two levels of three rows, the first with its own offset and weight."""
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        weight_semantics="frequency",
        features={"g": Categorical(base="first")},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FractionalFrequencyWeightWarning)
        model.fit(
            pd.DataFrame({"g": ["a", "a", "a", "b", "b", "b"]}),
            np.tile(np.asarray(y, dtype=float), 2),
            offset=np.tile(np.asarray(offset, dtype=float), 2),
            sample_weight=np.tile(np.asarray(weights, dtype=float), 2),
        )
    return model


def _common_rows(model: SuperGLM, offset) -> np.ndarray:
    """Each level's probability on its two rows at the common offset."""
    eta = model._dm.matvec(model.result.beta) + model.result.intercept
    return np.exp(eta + np.tile(np.asarray(offset, dtype=float), 2))[[1, 4]]


# Both three-row fixtures have a common-row probability near 1/2 at their
# maximum, where the six rows' scores sum in size to under 4.001 and each
# level's score falls by more than 1.99 per unit of its shift (the common
# non-event's observed curvature p / (1 - p)^2 = 2).  The certificate holds the
# intercept's score within bar sum|s| and level b's centred score within
# bar sum|s| / 2, so each level's score is within 2.5 bar sum|s| and its log
# probability within 2.5 bar 4.001 / 1.99 < 5.1 bar of the maximum.
_THREE_ROW_LOG_P_BOUND = 5.1 * MODE_CERTIFICATION_BAR


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_a_clipped_non_event_row_does_not_certify_a_stop(direct_solve: str) -> None:
    """Sol's #437 fixture: a non-event of weight 1e14 per level at offset -30, the rest at +10.

    The heavy rows sit far below ``clip_mu``'s floor, where each adds a flat
    ``-2 w log(1 - 1e-7) ~ 2e7`` to the clipped deviance: that constant made
    the relative deviance stop pass after one step, at a common-row
    probability of 0.049.  The maximum solves the level's score equation ``1 -
    p / (1 - p) - w q / (1 - q) = 0``, ``q = p e^-40``: ``p = 0.4999468955724079``
    (Sol's 70-digit oracle).  The stop is accepted only on the binomial/log
    score, so the fit goes on to it.
    """
    offset = [-30.0, 10.0, 10.0]
    model = _three_row_fit([0, 0, 1], offset, [1e14, 1, 1], direct_solve)
    assert model.result.converged
    assert model.result.termination_reason == "converged"
    p_star = 0.4999468955724079
    assert np.max(np.abs(np.log(_common_rows(model, offset) / p_star))) <= _THREE_ROW_LOG_P_BOUND


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_an_optimum_beside_a_negligible_event_row_is_certified(direct_solve: str) -> None:
    """Sol's #437 fixture: an event of weight 1e-12 per level at offset -20, the rest at +1.3.

    The light row sits below ``clip_mu``'s floor, and the stop refused while
    any event row did ran to max_iter at the maximum.  The certificate weighs
    the row's score ``1e-12`` against the whole score instead, and accepts
    the maximum ``p / (1 - p) = 1 + 1e-12``; the same model with every offset
    shifted by -2 starts inside and reaches the same probabilities.
    """
    p_star = (1.0 + 1e-12) / (2.0 + 1e-12)
    for offset in ([-20.0, 1.3, 1.3], [-22.0, -0.7, -0.7]):
        model = _three_row_fit([1, 0, 1], offset, [1e-12, 1, 1], direct_solve)
        assert model.result.converged
        assert model.result.termination_reason == "converged"
        p = _common_rows(model, offset)
        assert np.max(np.abs(np.log(p / p_star))) <= _THREE_ROW_LOG_P_BOUND


def _event_row_below_the_floor(shift: float, direct_solve: str):
    """claude's #437 fixture: 200 rows at offset +2 and one event row at offset -25."""
    rng = np.random.default_rng(0)
    x = np.append(rng.uniform(-1.0, 1.0, 200), 0.3)
    y = np.append((rng.uniform(size=200) < 0.2).astype(np.float64), 1.0)
    offset = np.append(np.full(200, 2.0), -25.0) + shift
    model = SuperGLM(
        family="binomial",
        link="log",
        selection_penalty=0.0,
        direct_solve=direct_solve,
        features={"x": Numeric()},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        model.fit(pd.DataFrame({"x": x}), y, offset=offset)
    eta = model._dm.matvec(model.result.beta) + model.result.intercept + offset
    return model, x, y, eta


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_an_event_row_the_clipped_iteration_ignores_is_not_certified(direct_solve: str) -> None:
    """The event row's score is ``w y = 1`` wherever its mean is, but PIRLS does not see it.

    Below ``clip_mu``'s floor PIRLS weights the row by ``mu'^2 / V(1e-7)``
    and credits it with ``exp(eta) / 1e-7`` (about 4e-6) of its score, so the
    clipped iteration settles on the maximum without the row, 0.13 standard
    errors from the model's.  Master certified that state when the start was
    inside (every offset shifted by -1); the previous refusal ran the other
    to max_iter and a SeparationWarning.  The binomial/log score refuses both
    stops, the score stops contracting, and both fits end uncertified at the
    same state.  At it, the true score is the row's uncredited part,
    ``(1 - c) (1, x_row)`` with ``c = exp(eta_row) / 1e-7``: the clipped score
    is zero to the certificate's bar against ``sum |s|``.
    """
    fits = [_event_row_below_the_floor(shift, direct_solve) for shift in (0.0, -1.0)]
    for model, x, y, eta in fits:
        assert not model.result.converged
        assert model.result.termination_reason == "score_stagnated"
        score, _ = mean_space_score_rows(y, np.ones_like(y), eta)
        uncredited = 1.0 - np.exp(eta[-1]) / 1e-7
        tolerance = MODE_CERTIFICATION_BAR * float(np.sum(np.abs(score)))
        assert abs(float(np.sum(score)) - uncredited) <= tolerance
        assert abs(float(score @ x) - 0.3 * uncredited) <= tolerance
    # one model: each fit's clipped score is within bar sum|s| per component,
    # so their coefficients are within 2 sqrt(2) bar sum|s| / lambda_min of
    # the clipped maximum's and every eta, |(1, x)| <= sqrt(2), within 4 bar
    # sum|s| / lambda_min of the other's (lambda_min: the Fisher curvature's)
    (first, x, y, eta_first), (second, _, _, eta_second) = fits
    design = np.column_stack([np.ones_like(x), x])
    score, fisher = mean_space_score_rows(y, np.ones_like(y), eta_first)
    smallest = float(np.linalg.eigvalsh(design.T @ (fisher[:, None] * design))[0])
    bound = 4.0 * MODE_CERTIFICATION_BAR * float(np.sum(np.abs(score))) / smallest
    assert np.max(np.abs(eta_first - eta_second)) <= bound
    assert abs(float(first.result.beta[0] - second.result.beta[0])) <= bound


@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_a_constrained_lowered_fit_is_certified_with_its_multipliers(direct_solve: str) -> None:
    """An increasing spline on a probability that rises then falls, offset by +1.5.

    The lowered start brings the fit under the binomial/log certificate, and
    its maximum has active monotonicity constraints: the penalized score is
    balanced by their multipliers, ``G + A' m = 0`` with ``m >= 0``, not
    zero.  The certificate tests that residual (without the multipliers this
    fit ended ``score_stagnated``) and accepts the mode the no-offset fit
    reaches, its deviance within the deviance stop's tolerance of it.
    """
    rng = np.random.default_rng(5)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(size=n) < 0.1 + 0.3 * np.sin(np.pi * x)).astype(np.float64)
    deviances = []
    for shift in (0.0, 1.5):
        model = SuperGLM(
            family="binomial",
            link="log",
            selection_penalty=0.0,
            direct_solve=direct_solve,
            features={"x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing)},
        )
        model.fit(pd.DataFrame({"x": x}), y, offset=np.full(n, shift))
        assert model.result.converged
        assert model.result.termination_reason == "converged"
        deviances.append(float(model.result.deviance))
    assert abs(deviances[1] - deviances[0]) <= 2.0 * model._tol * (max(deviances) + 1.0)


def test_the_clip_and_the_true_score_are_read_off_the_unclipped_eta() -> None:
    """``mean_space_clipped_rows`` and ``mean_space_score_rows`` on each of their branches."""
    family, link = Binomial(), LogLink()
    # clip_mu holds 1e-7 <= mu <= 1 - 1e-7: rows below and above it count,
    # events or not, when they carry weight, and so does exp's underflow
    eta = np.array([-17.0, -17.0, -16.0, -1e-8, -1e-6, -800.0, -17.0])
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    assert mean_space_clipped_rows(family, link, eta, weights) == 4
    assert mean_space_clipped_rows(family, LogitLink(), eta, weights) == 0
    # the score w [y - (1 - y) mu / (1 - mu)] and Fisher weight w mu / (1 - mu)
    # against 400-digit references (eta = -1e-300 needs 1 - mu to 300 digits):
    # exp, expm1, the division and the product by w round once each, within 4
    # eps of the reference; exp's underflow below eta ~ -745 gives the exact
    # limits w y and 0
    from decimal import Decimal, localcontext

    eta = np.array([-800.0, -40.0, -1.0, -1e-12, -1e-300, -40.0, -1e-12])
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    weights = np.array([3.0, 1e14, 0.5, 2.0, 1.0, 1e-12, 7.0])
    score, fisher = mean_space_score_rows(y, weights, eta)
    with localcontext() as context:
        context.prec = 400
        for k in range(len(eta)):
            mu = Decimal(float(eta[k])).exp()
            odds = mu / (1 - mu)
            w = Decimal(float(weights[k]))
            response = Decimal(float(y[k]))
            expected_score = float(w * (response - (1 - response) * odds))
            expected_fisher = float(w * odds)
            assert abs(score[k] - expected_score) <= 4.0 * _EPS * abs(expected_score)
            assert abs(fisher[k] - expected_fisher) <= 4.0 * _EPS * abs(expected_fisher)
    # outside the space: no likelihood with weight, nothing without
    score, fisher = mean_space_score_rows(np.zeros(2), np.array([1.0, 0.0]), np.zeros(2))
    assert not np.isfinite(score[0])
    assert score[1] == 0.0 and fisher[1] == 0.0


def test_the_halving_budget_reaches_the_fraction_to_the_boundary() -> None:
    """``_mean_space_halving_budget`` on each of its branches."""
    family, link = Binomial(), LogLink()
    weights = np.ones(2)

    def budget(eta, proposal, default=20, link=link):
        return _mean_space_halving_budget(
            committed=SimpleNamespace(eta_unclipped=np.asarray(eta, dtype=float)),
            proposal=SimpleNamespace(eta_unclipped=np.asarray(proposal, dtype=float)),
            weights=weights,
            family=family,
            link=link,
            default=default,
        )

    # a link that stays in (0, 1), a start outside the space, no row moving up,
    # and a fraction the ordinary budget reaches all keep the default
    assert budget([-1.0, -1.0], [5.0, -1.0], link=LogitLink()) == 20
    assert budget([0.0, -1.0], [5.0, -1.0]) == 20
    assert budget([-1.0, -1.0], [-2.0, -1.5]) == 20
    assert budget([-1.0, -1.0], [1.0, -1.0]) == 20
    # t = 1 / 3e7: the first halving below t, then the ordinary budget
    eta, step = -1.0, 3e7
    depth = budget([eta, -1.0], [eta + step, -1.0]) - 20
    fraction = -eta / step
    assert 2.0**-depth < fraction <= 2.0 ** -(depth - 1)
    # float64's halving depth caps it
    assert budget([-1e-320, -1.0], [1.0, -1.0]) == 1074


def _frequency_weighted_events(*, plain: bool, direct_solve: str) -> SuperGLM:
    """Sol's #431 (d) fixture: per level, nine events of weight 1e7 and a non-event of weight 1."""
    n = 20
    g = np.tile(np.arange(2), 10)
    y = np.ones(n)
    y[:2] = 0.0
    weights = np.where(y > 0.0, 1e7, 1.0)
    level = (
        Categorical(base="first") if plain else RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0))
    )
    model = SuperGLM(
        family="binomial",
        link="log",
        direct_solve=direct_solve,
        selection_penalty=0.0,
        weight_semantics="frequency",
        features={"g": level},
    )
    X = pd.DataFrame({"g": [f"g{c}" for c in g]})
    if plain:
        model.fit(X, y, sample_weight=weights)
    else:
        model.fit_reml(X, y, sample_weight=weights)
    return model


@pytest.mark.xfail(
    strict=True,
    raises=ObservedModeNotConvergedError,
    reason="#431 (d): the 1 - 1e-7 clip is still the boundary; redone in #431",
)
@pytest.mark.parametrize("direct_solve", ["auto", "gram"])
def test_an_interior_maximum_above_the_old_clip_is_reached(direct_solve: str) -> None:
    """``p = 9e7 / (9e7 + 1)`` is interior (#431 d), and the fit should reach it.

    Not yet: the boundary rule still treats every probability at or above
    ``1 - 1e-7`` as the boundary.  Kept as the target of the #431 redo.

    Both levels hold the same data, so the penalized maximum has a zero
    random effect and an intercept solving the score equation
    ``S(eta) = 18e7 - 2 p / (1 - p) = 0``: ``eta* = -log1p(1 / 9e7)``, above
    the clip's ``log1p(-1e-7)``, which calls it the boundary.  The
    certified mode has ``|S| <= bar sum |s|`` with ``sum |s| = 18e7 + 2 p /
    (1 - p)`` and ``|S'| = 2 p / (1 - p)^2``, so it lies within ``bar sum |s|
    / |S'|`` of ``eta*``.  The score is evaluated at ``p = fl(exp(eta))``,
    whose ``1 - p`` is exact but carries ``p``'s rounding, so the computed
    root can sit two units of ``2^-53`` (``exp`` and the rounding of ``p``)
    from the exact one; four allow for ``exp``'s own error.
    """
    model = _frequency_weighted_events(plain=False, direct_solve=direct_solve)
    assert model.result.converged
    assert model.result.termination_reason == "converged"
    eta = model._dm.matvec(model.result.beta) + model.result.intercept
    eta_star = -np.log1p(1.0 / 9e7)
    odds = 9e7
    total = 18e7 + 2.0 * odds
    slope = 2.0 * odds * (odds + 1.0)
    bar = mode_certification_bar(float(model._reml_profile["reml_tol_resolved"]))
    assert float(np.max(np.abs(eta - eta_star))) <= bar * total / slope + 4.0 * _U
    # a plain fit stops on its deviance test, not the certificate, and is
    # checked for the verdict alone: an interior mode, not the boundary
    plain = _frequency_weighted_events(plain=True, direct_solve=direct_solve)
    assert plain.result.converged
    assert plain.result.termination_reason == "converged"
    eta = plain._dm.matvec(plain.result.beta) + plain.result.intercept
    assert float(np.max(np.abs(eta - eta_star))) < 0.5 * abs(eta_star)
