"""The discrete terminal PIRLS refit certifies its fixed point to first order.

Fisher scoring on a non-canonical link contracts only linearly, so the old
objective-change stop certified the coefficients to about ``sqrt(tol)``: an
objective change of ``tau |F|`` leaves ``1/2 e' H e`` of that size and ``e``
of order ``sqrt(tau)`` (Gill, Murray and Wright, Practical Optimization,
section 8.2.3).  A discrete Gamma fit held at fixed smoothing parameters was
left 1e-6 from its fixed point at ``pirls_tol = 1e-10``.  The terminal refit
now stops when either first-order certificate holds, as glum's IRLS stops on
``gradient_tol`` or ``step_size_tol``:

- the penalized score ``g = [1 X]' s - S beta``, ``s = w (y - mu) h' / V``,
  satisfies ``||g||_inf <= tau sum |s|``, the intercept column of the
  pre-cancellation magnitude ``|X|' |s|``; or
- the step into the retained state satisfies ``|d_j| <= tau max(1, |beta_j|)``.

Fisher scoring reaches the fixed point at a linear rate.  Gamma/log, whose
observed rows the solver approves, takes observed-Newton steps to the same
root instead, quadratically, and keeps the Fisher geometry.

To first order ``beta* - beta = H_O^-1 g`` with ``H_O`` the observed penalized
Hessian.  The step certificate bounds the score too: the retained state is one
Fisher step past ``beta_prev``, so ``g = (H_F - H_O) d`` (one observed-Newton
step leaves only the second-order remainder in ``d``).  Either way
``|g| <= c = tau max(sum |s|, |H_F - H_O| max(1, |beta|))`` componentwise, and
``|beta - beta*| <= |H_O^-1| (c + 2 err)`` with ``err`` the rounding of the
score evaluated here and in the reference: the sums of products rounded once
(``2 eps G``, ``G = |X|' |s| + |S| |beta|``), each score row within
``16 eps w |h' / V| (|y| + |mu|)``, and ``eta`` within
``(m + 1) eps (|X| |beta| + |offset|)`` for ``m`` nonzero products, moving the
row score by ``W_O |d eta|``.  The reference ``beta*`` is the dense Newton
fixed point on the observed Hessian.

A level whose responses are all zero has no finite Tweedie coefficient.  The
score of that level vanishes with its fitted mass while its Fisher step stays
at one unit of log-mean per iteration, so a step-only rule walks it to the
link's overflow guard; the score certificate stops it where the search left it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, LambdaPolicy, Numeric, RandomEffect, Spline, SuperGLM, Tweedie
from superglm.links import stabilize_eta
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.solvers.irls_direct import fit_irls_direct

EPS = np.finfo(np.float64).eps
PIRLS_TOL = 1e-10
HELD = {"x": 3.0, "grp": 10.0}


def _frame(seed: int, n: int = 2000) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=n)
    cat = rng.integers(0, 4, n)
    grp = rng.integers(0, 40, n)
    exposure = rng.uniform(0.5, 2.0, n)
    eta = (
        -0.4
        + 0.4 * np.sin(2 * np.pi * x)
        + np.array([0.0, 0.2, -0.1, 0.3])[cat]
        + rng.normal(0.0, 0.3, 40)[grp]
    )
    frame = pd.DataFrame(
        {
            "x": x,
            "cat": np.array([f"c{c}" for c in cat], dtype=object),
            "grp": np.array([f"g{g}" for g in grp], dtype=object),
        }
    )
    return frame, eta, exposure, cat


def _response(family: str):
    frame, eta, exposure, _ = _frame(11)
    rng = np.random.default_rng(12)
    if family == "gamma":
        return frame, rng.gamma(3.0, np.exp(eta) / 3.0), None
    counts = rng.poisson(0.8 * exposure * np.exp(eta))
    y = np.array([rng.gamma(2.0, 0.6, count).sum() for count in counts])
    return frame, y, np.log(exposure)


def _model(family: str, *, held: bool) -> SuperGLM:
    def policy(name: str):
        return LambdaPolicy.fixed(HELD[name]) if held else None

    return SuperGLM(
        family=Tweedie(p=1.5) if family == "tweedie" else family,
        features={
            "x": Spline(n_knots=8, lambda_policy=policy("x")),
            "cat": Categorical(),
            "grp": RandomEffect(lambda_policy=policy("grp")),
        },
        selection_penalty=0,
        discrete=True,
    )


def _rows(model: SuperGLM, y, offset, beta):
    """``(mu, s, W_F, W_O, d_eta)`` at intercept-augmented ``beta``, solver coordinates."""
    X = np.hstack([model._dm.toarray(), np.ones((len(y), 1))])
    eta = X @ beta + (0.0 if offset is None else offset)
    link, family = model._link, model._distribution
    mu, d1, d2 = link.inverse(eta), link.deriv_inverse(eta), link.deriv2_inverse(eta)
    variance = family.variance(mu)
    slope = d1 / variance
    curvature = d2 / variance - slope * d1 * family.variance_derivative(mu) / variance
    fisher = d1 * slope
    observed = fisher - (y - mu) * curvature
    offset_size = 0.0 if offset is None else np.abs(offset)
    d_eta = (np.count_nonzero(X, axis=1) + 1) * EPS * (np.abs(X) @ np.abs(beta) + offset_size)
    rounding = 16 * EPS * np.abs(slope) * (np.abs(y) + np.abs(mu))
    return X, (y - mu) * slope, fisher, observed, np.abs(observed) * d_eta + rounding


def _newton_fixed_point(model: SuperGLM, y, offset, S, beta):
    """The dense observed-Newton fixed point from ``beta``.

    Newton converges quadratically, so the step contracts until rounding sets
    its floor; the first step that fails to contract is rounding and is not taken.
    """
    previous = np.inf
    for _ in range(20):
        X, score, _, observed, _ = _rows(model, y, offset, beta)
        step = np.linalg.solve(X.T @ (observed[:, None] * X) + S, X.T @ score - S @ beta)
        size = float(np.max(np.abs(step)))
        if size >= previous:
            return beta
        beta, previous = beta + step, size
    raise AssertionError("the dense Newton reference did not settle")


@pytest.mark.parametrize("family", ["gamma", "tweedie"])
def test_discrete_terminal_fit_reaches_the_certified_fixed_point(family: str) -> None:
    frame, y, offset = _response(family)
    model = _model(family, held=True)
    # the default pirls_tol (1e-6): the terminal refit tightens it to PIRLS_TOL
    model.fit_reml(frame, y, offset=offset)
    dm, p = model._dm, model._dm.shape[1]
    S = np.zeros((p + 1, p + 1))
    S[:p, :p] = build_penalty_matrix(
        dm.group_matrices, model._groups, model._reml_lambdas, p, model._reml_penalties
    )
    shift = model._runtime_canonical_state["intercept_shift"]
    beta = np.append(model.result.beta, model.result.intercept - shift)
    X, score, fisher, observed, rows = _rows(model, y, offset, beta)
    H_O = X.T @ (observed[:, None] * X) + S
    H_F = X.T @ (fisher[:, None] * X) + S

    certified = PIRLS_TOL * np.maximum(
        np.sum(np.abs(score)),
        np.abs(H_F - H_O) @ np.maximum(1.0, np.abs(beta)),
    )
    err = np.abs(X).T @ (rows + 2 * EPS * np.abs(score)) + 2 * EPS * np.abs(S) @ np.abs(beta)
    bound = np.abs(np.linalg.inv(H_O)) @ (certified + 2 * err)
    distance = np.abs(beta - _newton_fixed_point(model, y, offset, S, beta))
    assert np.all(distance <= bound), float(np.max(distance / bound))
    if family == "gamma":
        # quadratic from the held start: two Newton steps where Fisher scoring takes six
        assert model.result.n_iter <= 2


def test_score_mode_stops_no_later_than_the_step_rule() -> None:
    """A column in large units makes ``tau sum |s|`` a strict bar on its score.

    The stopping rule does not change the iterates, and ``"score"`` stops on
    the smaller of the score ratio and the step, so on the same iterates it
    stops no later than the step rule the exact path uses.
    """
    frame, y, _ = _response("gamma")
    frame = frame.assign(big=1e6 * np.random.default_rng(3).standard_normal(len(frame)))
    model = SuperGLM(
        family="gamma",
        features={
            "x": Spline(n_knots=8, lambda_policy=LambdaPolicy.fixed(HELD["x"])),
            "cat": Categorical(),
            "grp": RandomEffect(lambda_policy=LambdaPolicy.fixed(HELD["grp"])),
            "big": Numeric(),
        },
        selection_penalty=0,
        discrete=True,
    )
    model.fit_reml(frame, y, pirls_tol=PIRLS_TOL)
    results = {
        convergence: fit_irls_direct(
            X=model._dm,
            y=y,
            weights=np.ones(len(y)),
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=dict(model._reml_lambdas),
            reml_penalties=model._reml_penalties,
            tol=PIRLS_TOL,
            convergence=convergence,
            weight_semantics="prior",
        )[0]
        for convergence in ("score", "coefficients")
    }
    assert results["score"].converged and results["coefficients"].converged
    assert results["score"].n_iter <= results["coefficients"].n_iter


def test_separated_level_is_not_walked_to_the_overflow_guard() -> None:
    frame, eta, exposure, cat = _frame(5, n=4000)
    rng = np.random.default_rng(6)
    counts = rng.poisson(exposure * np.exp(eta))
    y = np.array([rng.gamma(2.0, 0.6, count).sum() for count in counts])
    y[cat == 3] = 0.0
    model = _model("tweedie", held=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the build-time scan names the separated cell
        model.fit_reml(frame, y, offset=np.log(exposure))
    assert model.result.converged
    shift = model._runtime_canonical_state["intercept_shift"]
    eta_fit = (
        model._dm.matvec(model.result.beta) + (model.result.intercept - shift) + np.log(exposure)
    )
    np.testing.assert_array_equal(stabilize_eta(eta_fit, model._link), eta_fit)
