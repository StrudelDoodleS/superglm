"""The mode certificate's bar, its stop rule and its disclosure (one-engine design §3.8).

The stage-1 verifier's findings on the score stop, each pinned by a test that
fails under the named mutation (shown by running it, change record):

- a constant dense column centres to exact zeros on gram's centred system, so
  the REML objective, gradient and Hessian share one rank at every iterate
  (mutation: the unshifted weighted mean);
- a coefficient with no data rows has a finite relative score scale
  (mutation: the scale without the penalty's curvature);
- the score stop is reachable on gram at a raw offset (mutation: a bar below
  what the REML criterion needs, 1e-10);
- a score that stops contracting ends the solve (mutation: no stagnation stop);
- a mode short of the bar is published as not converged, never refused
  (mutation: the certificate's refusal restored);
- the bar resolves the REML criterion to its tolerance (mutation: bar 1e-6);
- the discrete terminal refit stops on the same certificate (mutation: its
  unconstrained ``score`` stop).

Bounds derive from the REML stop's own resolution, ``reml_tol (1 + |V|)``
over the criterion's curvature, and from eps; none is tuned.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import superglm.reml.direct as direct_module
import superglm.solvers.mode_score as mode_score
from superglm import Categorical, LambdaPolicy, Numeric, RandomEffect, Spline, SuperGLM
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.solvers.centered_system import build_centered_system
from superglm.solvers.mode_score import penalized_mode_residual

REML_TOL = 1e-9


def _no_re_frame(n: int, variant: str, seed: int = 343):
    """The stage-1 verifier's no-random-effect design (gram under auto)."""
    rng = np.random.default_rng(seed)
    frame = {f"s{j}": rng.uniform(size=n) for j in range(4)}
    cat = rng.integers(0, 12, n)
    frame["cat"] = np.array([f"c{c:03d}" for c in cat], dtype=object)
    frame["x1"] = rng.normal(size=n)
    frame = pd.DataFrame(frame)
    eta = (
        -1.0
        + sum(0.3 * np.sin(2 * np.pi * frame[f"s{j}"]) for j in range(4))
        + rng.normal(0, 0.2, 12)[cat]
        + 0.2 * frame["x1"]
    )
    features = {f"s{j}": Spline(kind="ps", k=10) for j in range(4)}
    features["cat"] = Categorical()
    features["x1"] = Numeric()
    if variant == "const":
        frame["k"] = 3.0
        features["k"] = Numeric()
    elif variant == "raw8":
        frame["xr"] = 1e8 + rng.normal(size=n)
        features["xr"] = Numeric()
        eta = eta + 0.1 * (frame["xr"] - 1e8)
    return frame, np.asarray(eta, dtype=float), features, rng


def _reml_stop_bound(model) -> dict[str, float]:
    """Each fit's ``rho`` is within ``reml_tol (1 + |V|)`` over the criterion's curvature of the optimum."""
    decision = model._reml_profile["reml_freeze_decision"]
    bound = {}
    for position, name in enumerate(decision["names"]):
        curvature = abs(decision["hess_diag"][position])
        bound[name] = 2.0 * REML_TOL * decision["score_scale"] / curvature
    return bound


# ----------------------------------------------------------------- centring
def test_a_constant_dense_column_centres_to_exact_zeros_on_gram():
    """Mutation: the unshifted weighted mean ``sum W x / sum W``.

    For these weights ``sum W * 3 / sum W`` rounds to ``3 - 4.4e-16``; centred
    on it the constant column keeps a diagonal of ``sum W (4.4e-16)^2`` whose
    Jacobi-scaled rank decision then keeps or drops the direction with each
    iterate's weights (the stage-1 verifier's Gamma/log constant-column fit).
    Averaged about one of its own rows it is exactly 3 and centres to zeros.
    """
    rng = np.random.default_rng(0)
    n = 200
    weights = rng.gamma(2.0, 0.5, n)
    dm = DesignMatrix(
        [DenseGroupMatrix(rng.normal(size=(n, 1))), DenseGroupMatrix(np.full((n, 1), 3.0))],
        n=n,
        p=2,
    )
    system = build_centered_system(
        dm=dm, W=weights, z_off=rng.normal(size=n), penalty=np.zeros((2, 2))
    )
    assert system.mean_x[1] == 3.0
    assert np.all(system.data_gram[1] == 0.0)
    assert np.all(system.data_gram[:, 1] == 0.0)
    assert system.rhs[1] == 0.0


@pytest.mark.slow
def test_a_constant_column_leaves_the_gram_smoothing_parameters_alone():
    """Stage-1 verifier HIGH finding: Gamma/log without a random effect, auto (gram).

    A constant numeric column is an exact alias of the intercept: the REML
    problem is the same with or without it, so the smoothing parameters must
    agree to what the REML stop resolves.  Under the unshifted mean the
    objective jumped by ``~log(eps^2)/2`` between iterates, the line search
    failed and the fit ended 3.2% off, unconverged.
    """
    fits = {}
    for variant in ("plain", "const"):
        frame, eta, features, rng = _no_re_frame(67_000, variant)
        y = rng.gamma(2.0, np.exp(eta) / 2.0)
        model = SuperGLM(family="gamma", features=features, selection_penalty=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(frame, y)
        assert model._reml_result.converged, variant
        fits[variant] = model
    bound = _reml_stop_bound(fits["plain"])
    for name, limit in bound.items():
        gap = abs(
            np.log(fits["const"]._reml_lambdas[name]) - np.log(fits["plain"]._reml_lambdas[name])
        )
        assert gap <= limit, (name, gap, limit)


# ------------------------------------------------------------ relative score
def test_a_coefficient_without_rows_has_a_finite_score_scale():
    """Mutation: the slope scale ``sum|s| scale_j + |S beta|_j`` without ``S_jj``.

    A random-effect level whose rows all carry zero weight has no data: its
    score is the penalty term ``-lambda beta_j`` alone, and rounding leaves
    ``beta_j`` at ~1e-9 on gram (the stage-1 verifier's census).  Scaled by
    ``|S beta|_j`` alone its relative score is identically 1, whatever
    ``beta_j``.  With the Jacobi scale ``zeta sqrt(S_jj)`` it is
    ``sqrt(lambda) |beta_j| / zeta``: below the bar.
    """
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    level = np.zeros((n, 1))  # the empty level: no row carries it
    dm = DesignMatrix([DenseGroupMatrix(x[:, None]), DenseGroupMatrix(level)], n=n, p=2)
    weights = np.ones(n)
    row_score = rng.normal(size=n)
    row_score -= row_score.mean()
    lam = 2.0
    beta = np.array([0.3, 1e-9])
    mean_x = np.array([float(x.mean()), 0.0])
    scale = np.sqrt(np.array([float(np.sum((x - x.mean()) ** 2)), 0.0]) / n)
    residual = penalized_mode_residual(
        dm=dm,
        row_score=row_score,
        fisher_weights=weights,
        positive_prior=weights > 0,
        mean_x=mean_x,
        centered_scale=scale,
        alpha=0.0,
        eta_tilde=x * beta[0],
        penalty_score=np.array([0.0, lam * beta[1]]),
        penalty_magnitude=np.array([0.0, lam * beta[1]]),
        penalty_curvature=np.array([0.0, lam]),
        sum_w=float(n),
        bar=mode_score.MODE_CERTIFICATION_BAR,
    )
    zeta = float(np.sum(np.abs(row_score))) / np.sqrt(n)
    expected = lam * beta[1] / (zeta * np.sqrt(lam) + lam * beta[1])
    assert residual.relative[2] == pytest.approx(expected, rel=8 * mode_score._EPS)
    assert residual.relative[2] < mode_score.MODE_CERTIFICATION_BAR


# ----------------------------------------------------------------- stop rule
def _recorded_calls(monkeypatch) -> list:
    calls: list = []
    real = direct_module.fit_irls_direct

    def recording(*args, **kwargs):
        output = real(*args, **kwargs)
        calls.append(output[0].termination_reason)
        return output

    monkeypatch.setattr(direct_module, "fit_irls_direct", recording)
    return calls


def test_the_score_stop_is_reachable_on_gram_at_a_raw_offset(monkeypatch):
    """Stage-1 verifier HIGH finding: a column at 1e8 + N(0, 1), Poisson, no random effect.

    Mutations: a bar tighter than the REML criterion needs (1e-10,
    ``REML_TOL_BAR_RATIO`` 0.1), or gram's PIRLS state in raw coordinates.
    Every PIRLS the criterion reads then ran to its budget (stage 1: 2,882
    iterations for 30 calls at 5,000 rows).  In raw coordinates the column's
    1e8 offset leaves the penalized deviance noisy at 1e-11 relative, so near
    the mode no Newton step passes the line search and one call stalls at a
    relative score of 1e-6.  About the prior-weighted centre, at the derived
    bar, every call certifies its mode.
    """
    frame, eta, features, rng = _no_re_frame(5_000, "raw8")
    y = rng.poisson(np.exp(eta)).astype(float)
    calls = _recorded_calls(monkeypatch)
    model = SuperGLM(family="poisson", features=features, selection_penalty=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    assert model._reml_profile["direct_backend"] == "gram"
    assert calls and set(calls) == {"converged"}, calls
    assert model._reml_result.converged
    assert model._reml_profile["reml_terminal_mode_certified"] is True


def _stalling_fit(**fit_kwargs):
    """A binomial level with one separated row at lambda 1e-7 and weights x1e4.

    Its linear predictor walks to the link's overflow guard, where the score
    stops moving (the stage-1 verifier's frozen 6.9e-6): no step can certify
    this mode.
    """
    rng = np.random.default_rng(5)
    n, big = 1500, 35
    popularity = rng.gamma(1.5, size=big)
    g = np.concatenate(
        [rng.choice(big, size=n - 5, p=popularity / popularity.sum()), np.arange(big, 40)]
    )
    g = g[rng.permutation(n)]
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "u": rng.uniform(size=n),
            "cat": np.array([f"c{c}" for c in rng.integers(0, 8, n)], dtype=object),
            "g": np.array([f"g{c:02d}" for c in g], dtype=object),
        }
    )
    eta = 0.1 + 0.3 * frame["x1"].to_numpy() + 0.4 * np.sin(4 * frame["u"].to_numpy())
    eta = eta + rng.normal(0, 0.3, 40)[g]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    model = SuperGLM(
        family="binomial",
        features={
            "x1": Numeric(),
            "u": Spline(kind="ps", k=8),
            "cat": Categorical(),
            "g": RandomEffect(
                levels=[f"g{c:02d}" for c in range(40)], lambda_policy=LambdaPolicy.fixed(1e-7)
            ),
        },
        selection_penalty=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y, sample_weight=np.full(n, 1e4), **fit_kwargs)
    return model


@pytest.mark.slow
def test_a_score_that_stops_contracting_ends_the_solve(monkeypatch):
    """Mutation: no stagnation stop (every such call runs its 100 iterations).

    A resolved score that does not halve within ``stagnation_window`` cannot
    reach the bar within the budget (``mode_score.stagnation_window``): the
    solve ends there and the fit publishes its mode as not converged.
    """
    calls = _recorded_calls(monkeypatch)
    model = _stalling_fit()
    assert "score_stagnated" in calls
    assert "max_iter" not in calls
    window = mode_score.stagnation_window(100)
    assert model._reml_profile["irls_iters"] <= len(calls) * 3 * window


def test_a_mode_short_of_the_bar_is_published_as_not_converged():
    """Owner decision 3 (2026-09-30): never refused, disclosed as not converged.

    Mutation: the terminal certificate's ``ObservedModeNotCertifiedError``
    restored.  One PIRLS iteration per solve cannot certify a Gamma/log mode.
    """
    frame, eta, features, rng = _no_re_frame(3_000, "plain")
    y = rng.gamma(2.0, np.exp(eta) / 2.0)
    model = SuperGLM(family="gamma", features=features, selection_penalty=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y, max_pirls_iter=1)
    assert model._reml_profile["reml_terminal_mode_certified"] is False
    assert model._reml_result.converged is False
    assert model.diagnostics()["_model"]["converged"] is False
    assert np.all(np.isfinite(model.result.beta))


@pytest.mark.slow
def test_the_bar_resolves_the_reml_criterion(monkeypatch):
    """Mutation: bar 1e-6.  The inexact-Newton forcing argument (``mode_score``, **The bar**).

    The same Gamma/log fit at the production bar and at 1e-10 must select
    the same smoothing parameters to what the REML stop resolves.  At 1e-6
    the criterion's evaluations are too noisy for its line search: the
    lambdas land 27x outside the bound (measured).
    """
    frame, eta, features, rng = _no_re_frame(67_000, "plain")
    y = rng.gamma(2.0, np.exp(eta) / 2.0)
    fits = []
    for ratio in (mode_score.REML_TOL_BAR_RATIO, 0.1):
        # the bar is REML_TOL_BAR_RATIO * reml_tol: production, then 1e-10
        monkeypatch.setattr(mode_score, "REML_TOL_BAR_RATIO", ratio)
        model = SuperGLM(family="gamma", features=dict(features), selection_penalty=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(frame, y)
        assert model._reml_result.converged
        fits.append(model)
    bound = _reml_stop_bound(fits[1])
    for name, limit in bound.items():
        gap = abs(np.log(fits[0]._reml_lambdas[name]) - np.log(fits[1]._reml_lambdas[name]))
        assert gap <= limit, (name, gap, limit)


def test_the_discrete_terminal_refit_stops_on_the_certificate():
    """Mutation: the discrete terminal's unconstrained ``score`` stop.

    One stop rule for every route auto uses: the discrete terminal refit
    certifies its mode with the same residual, and the fit says whether it
    did.
    """
    frame, eta, features, rng = _no_re_frame(3_000, "plain")
    y = rng.poisson(np.exp(eta)).astype(float)
    model = SuperGLM(family="poisson", features=features, selection_penalty=0, discrete=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    assert model._reml_profile["reml_terminal_mode_certified"] is True
