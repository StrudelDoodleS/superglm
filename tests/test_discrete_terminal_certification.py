"""The discrete terminal PIRLS refit certifies its fixed point to first order.

Fisher scoring on a non-canonical link contracts only linearly, so the old
objective-change stop certified the coefficients to about ``sqrt(tol)``: an
objective change of ``tau |F|`` leaves ``1/2 e' H e`` of that size and ``e``
of order ``sqrt(tau)`` (Gill, Murray and Wright, Practical Optimization,
section 8.2.3).  A discrete Gamma fit held at fixed smoothing parameters was
left 1e-6 from its fixed point at ``pirls_tol = 1e-10``.  The terminal refit
of every route auto uses now stops on the mode certificate
(``convergence="mode_score"``, one-engine design §3.8): the penalized score
``g = [1 X~]' s - S beta``, ``s = w (y - mu) h' / V``, satisfies ``|g_0| <=
bar sum |s|`` and ``|g_j| <= bar (zeta sqrt(D_jj + S_jj) + |S beta|_j)``,
``zeta = sum |s| / sqrt(sum w_F)`` and ``D_jj = sum w_F x~_j^2``
(``solvers.mode_score``), with ``bar`` the certificate's bar for the fit's
REML tolerance (``mode_certification_bar``).

To first order ``beta* - beta = H_O^-1 g`` with ``H_O`` the observed penalized
Hessian, so ``|beta - beta*| <= |H_O^-1| (c + 2 err)`` componentwise, with
``c`` the certified score bound above and ``err`` the rounding of the score
evaluated here and in the reference: the sums of products rounded once
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

import superglm.model.reml_finalize as reml_finalize
from superglm import Categorical, LambdaPolicy, RandomEffect, Spline, SuperGLM, Tweedie
from superglm.links import stabilize_eta
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.solvers import irls_direct
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.irls_state import _IRLSStepDecision
from superglm.solvers.mode_score import MODE_CERTIFICATION_BAR, mode_certification_bar

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


def _certified_score(model: SuperGLM, y, offset, S, beta, bar):
    """The penalized score ``g`` at ``beta``, the certificate's bound on it and its rounding.

    ``|g_0| <= bar sum |s|`` and ``|g_j| <= bar (zeta sqrt(D_jj + S_jj) +
    |S beta|_j)`` (module docstring, ``solvers.mode_score``), recomputed here
    from the dense design; ``err`` bounds the rounding of this evaluation and
    of the solver's (sums of products rounded once, ``2 eps G``, plus the
    rows' own rounding).
    """
    p = S.shape[0] - 1
    X, score, fisher, _, rows = _rows(model, y, offset, beta)
    total = float(np.sum(np.abs(score)))
    centred = X[:, :p] - (fisher @ X[:, :p]) / np.sum(fisher)
    zeta = total / np.sqrt(np.sum(fisher))
    curvature = fisher @ centred**2 + np.diag(S)[:p]
    certified = bar * np.append(
        zeta * np.sqrt(curvature) + np.abs(S[:p, :p] @ beta[:p]),
        total,
    )
    err = np.abs(X).T @ (rows + 2 * EPS * np.abs(score)) + 2 * EPS * np.abs(S) @ np.abs(beta)
    return X.T @ score - S @ beta, certified, err


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

    assert model._reml_profile["reml_terminal_mode_certified"] is True
    bar = mode_certification_bar(model._reml_profile["reml_tol_resolved"])
    total = float(np.sum(np.abs(score)))
    centred = X[:, :p] - (fisher @ X[:, :p]) / np.sum(fisher)
    zeta = total / np.sqrt(np.sum(fisher))
    curvature = fisher @ centred**2 + np.diag(S)[:p]
    certified = bar * np.append(
        zeta * np.sqrt(curvature) + np.abs(S[:p, :p] @ beta[:p]),
        total,
    )
    err = np.abs(X).T @ (rows + 2 * EPS * np.abs(score)) + 2 * EPS * np.abs(S) @ np.abs(beta)
    bound = np.abs(np.linalg.inv(H_O)) @ (certified + 2 * err)
    distance = np.abs(beta - _newton_fixed_point(model, y, offset, S, beta))
    assert np.all(distance <= bound), float(np.max(distance / bound))


@pytest.mark.parametrize("family", ["poisson", "gamma"])
def test_score_is_certified_no_finer_than_the_penalty_rounding(family: str) -> None:
    """A tensor block's margin penalties cancel in ``S beta`` at the fixed point.

    At lambda 1e10 the terms of ``S beta`` are about 1e10 times the score they
    cancel to, so the score settles at the rounding of ``S beta``: the mode
    certificate's bar carries ``|S beta|_j`` in its scale for that reason, and
    the fit certifies within 30 iterations.  The accepted score must sit within
    the certificate, recomputed here from the dense design.
    """
    rng = np.random.default_rng(3)
    n = 2000
    x, z = rng.uniform(size=n), rng.uniform(size=n)
    eta = -0.5 + 0.8 * x - 0.6 * z + 1.5 * (x - 0.5) * (z - 0.5)
    y = rng.poisson(np.exp(eta)) if family == "poisson" else rng.gamma(3.0, np.exp(eta) / 3.0)
    model = SuperGLM(
        family=family,
        features={
            name: Spline(kind="cr", n_knots=8, penalty="ssp", discrete=True) for name in "xz"
        },
        interactions=[("x", "z")],
        selection_penalty=0,
        discrete=True,
    )
    model.fit_reml(pd.DataFrame({"x": x, "z": z}), y, max_reml_iter=1)  # builds the design
    lambdas = {name: 1e10 if ":" in name else 0.1 for name in model._reml_lambdas}
    result = fit_irls_direct(
        X=model._dm,
        y=y,
        weights=np.ones(n),
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        lambda2=lambdas,
        reml_penalties=model._reml_penalties,
        max_iter=30,
        tol=PIRLS_TOL,
        convergence="mode_score",
        weight_semantics="prior",
    )[0]
    assert result.converged

    dm, p = model._dm, model._dm.shape[1]
    S = np.zeros((p + 1, p + 1))
    S[:p, :p] = build_penalty_matrix(
        dm.group_matrices, model._groups, lambdas, p, model._reml_penalties
    )
    beta = np.append(result.beta, result.intercept)
    score, certified, err = _certified_score(model, y, None, S, beta, MODE_CERTIFICATION_BAR)
    assert np.all(np.abs(score) <= certified + 2 * err)


def test_a_damped_step_does_not_certify_the_fixed_point(monkeypatch) -> None:
    """A backtracked step is ``alpha d``, so its size says nothing about ``d``.

    The first line search is forced to keep ``2^-40`` of the Fisher step from
    zero coefficients: the step test reads about 1e-12 while the score is far
    from zero.  The fit must go on until a certificate holds at the published
    state, which the dense score recomputed here confirms.
    """
    frame, y, _ = _response("gamma")
    model = _model("gamma", held=True)
    model.fit_reml(frame, y)
    original = irls_direct._select_irls_trial
    forced = []

    def damp_first(**kwargs):
        if forced:
            return original(**kwargs)
        alpha = 2.0**-40
        kwargs["evaluate_state"](alpha)
        forced.append(alpha)
        return _IRLSStepDecision(alpha=alpha, step_halvings=40, step_rejected=False)

    monkeypatch.setattr(irls_direct, "_select_irls_trial", damp_first)
    lambdas = dict(model._reml_lambdas)
    result = fit_irls_direct(
        X=model._dm,
        y=y,
        weights=np.ones(len(y)),
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        lambda2=lambdas,
        reml_penalties=model._reml_penalties,
        tol=PIRLS_TOL,
        convergence="mode_score",
        weight_semantics="prior",
    )[0]
    assert forced and result.converged

    dm, p = model._dm, model._dm.shape[1]
    S = np.zeros((p + 1, p + 1))
    S[:p, :p] = build_penalty_matrix(
        dm.group_matrices, model._groups, lambdas, p, model._reml_penalties
    )
    beta = np.append(result.beta, result.intercept)
    score, certified, err = _certified_score(model, y, None, S, beta, MODE_CERTIFICATION_BAR)
    assert np.all(np.abs(score) <= certified + 2 * err)


@pytest.mark.parametrize(
    ("family", "link"),
    [("poisson", "log"), ("gamma", "log"), ("poisson", "sqrt"), ("binomial", "probit")],
)
def test_every_discrete_terminal_refit_stops_on_the_mode_certificate(
    monkeypatch, family: str, link: str
) -> None:
    """One stop rule for every route auto uses (one-engine design §3.8).

    Fisher scoring off a canonical link contracts only linearly, and at the
    old 1e-10 target that could spend the whole iteration budget; the bar the
    REML tolerance needs (``mode_certification_bar``, 1e-8 at the default) is
    reached at a linear rate too, so every pair certifies its terminal mode.
    """
    modes = []
    original = reml_finalize.fit_irls_direct

    def recording(*args, **kwargs):
        if kwargs.get("trace_purpose") == "reml_final":
            modes.append(kwargs["convergence"])
        return original(*args, **kwargs)

    monkeypatch.setattr(reml_finalize, "fit_irls_direct", recording)
    frame, eta, exposure, _ = _frame(21)
    rng = np.random.default_rng(22)
    mean = np.exp(eta)
    y = {
        "poisson": lambda: rng.poisson(exposure * mean).astype(float),
        "gamma": lambda: rng.gamma(3.0, mean / 3.0),
        "binomial": lambda: (rng.uniform(size=len(eta)) < 1.0 / (1.0 + np.exp(-eta))).astype(float),
    }[family]()
    model = SuperGLM(
        family=family,
        link=link,
        features={"x": Spline(n_knots=8), "cat": Categorical()},
        selection_penalty=0,
        discrete=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame[["x", "cat"]], y)
    assert modes == ["mode_score"]
    assert model._reml_profile["reml_terminal_mode_certified"] is True


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
