"""Complete nested random-effect fits (fix D) against the dense ``gram`` backend.

The factor algebra is tested in ``test_nested_schur_factor.py`` and the
plumbing in ``test_nested_structured_plumbing.py``; this file runs whole
``SuperGLM`` fits on a make / model / variant chain beside a spline, a
categorical and a crossed random effect.  Section numbers refer to
``notes/research/2026-09-26-nested-random-effect-elimination.md``.

Where the tolerances come from.

- REML optimum.  The Newton engines stop when the projected gradient and the
  objective change are below ``tau (1 + |V|)`` with ``tau = REML_TOL``, so each
  fit's objective is within that of the optimum: two fits of one surface
  differ by at most ``tau (1 + |V|)``.  The nested fit's smoothing parameters
  are certified the same way on the dense surface: the dense fit held at
  them must be within ``tau (1 + |V|)`` of the dense optimum (exact path; on
  the discrete path a cold refit at fixed lambdas is not the engine's
  warm-started final state, which is a property of that engine alone).
  Coordinates of flat REML directions are not certified by anything and are
  compared through the fits they produce.
- One fixed set of smoothing parameters.  Both backends run the same PIRLS
  iteration.  Each step is a backward-stable solve (a Cholesky of the
  Jacobi-scaled ``H`` on the dense side; the tree recursion and a Cholesky of
  ``Q_s`` on the nested side), so the iterates agree to ``(p + k) eps
  kappa_s(H)`` relative per step, ``kappa_s`` of the intercept-augmented ``H``.
- The contraction.  PIRLS is Fisher scoring, ``beta <- beta + H_F^-1 g`` with
  ``H_F = X' W_F X + S``, ``W_F = w h'^2 / V`` and ``h`` the inverse link.  At the
  fixed point its Jacobian is ``J = H_F^-1 (H_F - H_O)``, where the observed
  ``H_O`` has rows ``W_F - w (y - mu) (h'' / V - h'^2 V' / V^2)``.  ``J`` is
  self-adjoint in the ``H_F`` inner product, so its norm there is its spectral
  radius ``rho``, and a difference made at one step reaches the end at most
  ``1 / (1 - rho)`` times over.  ``_budget`` asserts ``rho <= 1/2``: the 2
  below.  Poisson/log and Binomial/logit are canonical, ``h'' V = h'^2 V'``,
  so ``J = 0`` and scoring is Newton's method (Lange, Numerical Analysis for
  Statisticians, ch. 14).  Gamma/log and Tweedie/log have
  ``W_F - W_O = (1 - p) w (y - mu) mu^(1 - p)``, whose ratio to ``W_F`` exceeds
  one on these fixtures (3.2 and 10.7), so ``rho`` depends on the data
  (Osborne 1992, Int. Stat. Rev. 60) and is measured: 0.18 and 0.12 to 0.13.
- Stopping.  The retained fit is one PIRLS step past the engine's last state.
  It stops on the coefficient change for exact Gamma and Tweedie fits, on the
  penalized score or the step for discrete fits (a discrete Gamma fit ends
  within 1e-9 of its fixed point, against 1e-6 on the objective change), and
  on the objective change for exact canonical fits, which are Newton's method.
  Each fit is within its bounded fixed-point gap, so two fits held at the same
  smoothing parameters are within twice the larger and ``gamma`` gains that
  (``_held_gap``) whatever iteration stopped them.  The nested REML fit and
  the dense fit held at its smoothing parameters come by different paths, so
  each is asserted to be within ``PIRLS_TOL`` of the fixed point and
  ``gamma`` gains ``2 PIRLS_TOL``.  ``gamma`` bounds the coefficients
  relative to ``max(1, ||beta||_inf)``, and
  ``|d eta| <= gamma max(1, ||beta||_inf) (1 + ||X||_inf)``.  Deviance,
  dispersion, the REML objective, edf, edf1, standard errors and the REML
  derivatives are smooth functions of ``eta`` and ``H^-1`` evaluated by the
  same code on both sides, apart from the ``H^-1`` quantities each factor
  forms, which carry the same ``gamma``; they are compared at small integer
  multiples of that budget, relative to their own size or, for signed
  matrices, to the Cauchy-Schwarz scale ``sqrt(|A_ii A_jj|)``.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction
from functools import cache

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from superglm import (
    Categorical,
    LambdaPolicy,
    Numeric,
    RandomEffect,
    Spline,
    SuperGLM,
    Tweedie,
)
from superglm.inference.covariance import StructuredCovarianceAccessor
from superglm.model.reml_setup import collect_reml_groups
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.penalty_algebra import build_penalty_context, build_penalty_matrix
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.mode_score import mode_certification_bar
from superglm.solvers.structured import (
    NestedSchurFactor,
    ProfiledNestedSchurFactor,
)

EPS = np.finfo(np.float64).eps
PIRLS_TOL = 1e-10
REML_TOL = 1e-9
CHAIN = ("make", "model", "variant")
TERMS = ("x", "cat", "crossed", *CHAIN)
PENALIZED = ("x", "crossed", *CHAIN)
# response seeds with an interior REML optimum at every level (see ``_data``)
SEEDS = {"poisson": 8, "gamma": 1, "tweedie": 2, "binomial": 3, "gausslog": 4}
FAMILIES = tuple(SEEDS)


# ── Data ──────────────────────────────────────────────────────────────────


@cache
def _data(sizes: tuple[int, ...] = (5, 20, 80), n: int = 3000, seed: int = 3):
    """A strict chain with explicit interaction labels, a spline, a categorical,
    a crossed random effect and an exposure; two variants carry zero weight.

    Every level has a clear variance, so REML has an interior optimum: a flat
    boundary direction is frozen wherever its gradient drops below
    ``max(0.1 tau, 1e-7) (1 + |V|)``, which the ``tau (1 + |V|)`` bound does not cover.
    """
    rng = np.random.default_rng(seed)
    parents = [
        np.concatenate([np.arange(coarse), rng.integers(0, coarse, fine - coarse)])
        for coarse, fine in zip(sizes[:-1], sizes[1:], strict=True)
    ]
    popularity = rng.gamma(1.5, size=sizes[-1])
    leaf = rng.choice(sizes[-1], size=n, p=popularity / popularity.sum())
    codes = [leaf]
    for parent in reversed(parents):
        codes.insert(0, parent[codes[0]])
    make, model, variant = codes[0], codes[-2], leaf
    x = rng.uniform(size=n)
    cat = rng.integers(0, 4, n)
    crossed = rng.integers(0, 5, n)
    exposure = rng.uniform(0.5, 2.0, n)
    eta = (
        -0.6
        + 0.3 * np.sin(2 * np.pi * x)
        + np.array([0.0, 0.2, -0.1, 0.3])[cat]
        + rng.normal(0.0, 0.2, 5)[crossed]
        + sum(
            rng.normal(0.0, scale, size)[code]
            for scale, size, code in zip((0.3, 0.25, 0.25), sizes, codes, strict=False)
        )
    )
    frame = pd.DataFrame(
        {
            "x": x,
            "cat": np.array([f"c{c}" for c in cat], dtype=object),
            "crossed": np.array([f"r{c}" for c in crossed], dtype=object),
            "make": np.array([f"mk{c}" for c in make], dtype=object),
            "model": np.array(
                [f"mk{a}:md{b}" for a, b in zip(make, model, strict=True)], dtype=object
            ),
            "variant": np.array(
                [f"md{b}:v{c}" for b, c in zip(model, variant, strict=True)], dtype=object
            ),
        }
    )
    weight = np.where(np.isin(variant, np.unique(variant)[[3, 7]]), 0.0, 1.0)
    return frame, eta, exposure, weight


@cache
def _response(family: str):
    """Frame, response, offset and prior weights; Tweedie refuses zero prior weights."""
    frame, eta, exposure, weight = _data()
    rng = np.random.default_rng(SEEDS[family])
    mean = np.exp(eta)
    if family == "poisson":
        return frame, rng.poisson(exposure * mean).astype(float), np.log(exposure), weight
    if family == "gamma":
        return frame, rng.gamma(3.0, mean / 3.0), None, weight
    if family == "tweedie":
        counts = rng.poisson(0.8 * exposure * mean)
        y = np.array([rng.gamma(2.0, 0.6, count).sum() for count in counts])
        return frame, y, np.log(exposure), np.ones(len(y))
    if family == "gausslog":
        # Gaussian with a log link: observed rows w mu (2 mu - y) are negative
        # wherever y > 2 mu, so the chain factors signed rows (design §3.3)
        # (about 3% of the rows here)
        return frame, mean + rng.normal(0.0, 0.2, len(eta)), None, weight
    probability = 1.0 / (1.0 + np.exp(-(eta + 0.3)))
    return frame, (rng.uniform(size=len(eta)) < probability).astype(float), None, weight


def _holdout(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows with an unseen make, an unseen model of a seen make and an unseen variant."""
    rows = frame.iloc[:6].copy()
    make, model = rows["make"].iloc[1], rows["model"].iloc[2]
    rows.loc[rows.index[0], list(CHAIN)] = ["mkNEW", "mkNEW:mdNEW", "mdNEW:vNEW"]
    rows.loc[rows.index[1], ["model", "variant"]] = [f"{make}:mdNEW", "mdNEW:vNEW"]
    rows.loc[rows.index[2], "variant"] = f"{model.split(':')[1]}:vNEW"
    return rows


def _family(name: str):
    return {"tweedie": Tweedie(p=1.5), "gausslog": "gaussian"}.get(name, name)


def _link(name: str) -> str | None:
    return "log" if name == "gausslog" else None


def _fit(
    family: str,
    direct_solve: str,
    *,
    discrete: bool = False,
    lambdas: dict[str, float] | None = None,
    data=None,
) -> SuperGLM:
    frame, y, offset, weight = _response(family) if data is None else data
    policy = {
        name: None if lambdas is None else LambdaPolicy.fixed(lambdas[name]) for name in PENALIZED
    }
    features = {
        "x": Spline(n_knots=8, lambda_policy=policy["x"]),
        "cat": Categorical(),
        **{name: RandomEffect(lambda_policy=policy[name]) for name in ("crossed", *CHAIN)},
    }
    model = SuperGLM(
        family=_family(family),
        link=_link(family),
        features=features,
        selection_penalty=0,
        direct_solve=direct_solve,
        discrete=discrete,
    )
    with warnings.catch_warnings():
        # the binary response leaves a few boundary cells; separation is not under test
        warnings.simplefilter("ignore")
        model.fit_reml(
            frame, y, sample_weight=weight, offset=offset, pirls_tol=PIRLS_TOL, reml_tol=REML_TOL
        )
    return model


# ── Bounds ────────────────────────────────────────────────────────────────


def _scaled_condition(H: np.ndarray) -> float:
    scale = 1.0 / np.sqrt(np.diag(H))
    return float(np.linalg.cond(scale[:, None] * H * scale[None, :]))


def _pirls_rows(model: SuperGLM, y, offset, weight):
    """Design, penalty and working rows at the retained fit, in solver coordinates.

    The solver's intercept is the published one less the canonical column-mean
    shift, and its column is last.  Returns ``(X, S, beta, mu, slope, fisher,
    residual)`` with ``slope = h' / V``, ``fisher = W_F`` and
    ``residual = W_F - W_O``, which a canonical link makes zero.
    """
    dm = model._dm
    X = np.hstack([dm.toarray(), np.ones((dm.shape[0], 1))])
    p = dm.shape[1]
    S = np.zeros((p + 1, p + 1))
    S[:p, :p] = build_penalty_matrix(
        dm.group_matrices, model._groups, model._reml_lambdas, p, model._reml_penalties
    )
    shift = model._runtime_canonical_state["intercept_shift"]
    beta = np.append(model.result.beta, model.result.intercept - shift)
    eta = X @ beta + (0.0 if offset is None else offset)
    link, family = model._link, model._distribution
    mu, d1, d2 = link.inverse(eta), link.deriv_inverse(eta), link.deriv2_inverse(eta)
    variance = family.variance(mu)
    slope = d1 / variance
    curvature = d2 / variance - slope * d1 * family.variance_derivative(mu) / variance
    return X, S, beta, mu, slope, weight * d1 * slope, weight * (y - mu) * curvature


def _contraction(model: SuperGLM, y, offset, weight) -> float:
    """``rho = ||J||_H_F``: the largest ``|lambda|`` of the pencil ``(H_F - H_O, H_F)``."""
    X, S, _, _, _, fisher, residual = _pirls_rows(model, y, offset, weight)
    H_F = X.T @ (fisher[:, None] * X) + S
    pencil = scipy.linalg.eigh(X.T @ (residual[:, None] * X), H_F, eigvals_only=True)
    return float(np.max(np.abs(pencil)))


def _fixed_point_gap(model: SuperGLM, y, offset, weight) -> float:
    """A bound on ``max |beta - beta*| / max(1, ||beta||_inf)`` at the retained fit.

    To first order ``beta - beta* = H_O^-1 g``, ``g`` the penalized score.  ``g``
    is summed exactly over products rounded once, an error of at most ``2 eps G``
    with ``G = |X|' |s| + |S| |beta|``.  Each score row ``s = w (y - mu) h' / V``
    takes under eight operations within 2 ulp, ``16 eps w |h' / V| (|y| + |mu|)``,
    and moves by ``W_O |d eta|``, ``eta`` summing the row's ``m`` nonzero products
    and the offset (an exact zero adds no rounding), so
    ``|d eta| <= (m + 1) eps (|X| |beta| + |offset|)``.  The bound adds these,
    through ``|H_O^-1|``, to ``|H_O^-1 g|``.
    """
    X, S, beta, mu, slope, fisher, residual = _pirls_rows(model, y, offset, weight)
    observed = fisher - residual
    inverse = np.linalg.inv(X.T @ (observed[:, None] * X) + S)
    score = weight * (y - mu) * slope
    terms = np.vstack([X * score[:, None], -(S * beta).T])
    g = np.array([math.fsum(column) for column in terms.T])
    offset_size = 0.0 if offset is None else np.abs(offset)
    d_eta = (np.count_nonzero(X, axis=1) + 1) * EPS * (np.abs(X) @ np.abs(beta) + offset_size)
    rows = np.abs(observed) * d_eta + 16 * EPS * weight * np.abs(slope) * (np.abs(y) + np.abs(mu))
    error = np.abs(X).T @ (rows + 2 * EPS * np.abs(score)) + 2 * EPS * np.abs(S) @ np.abs(beta)
    gap = np.abs(inverse @ g) + np.abs(inverse) @ error
    return float(np.max(gap)) / max(1.0, float(np.max(np.abs(beta))))


def _certified_gap(model: SuperGLM, y, offset, weight) -> float:
    """What the mode certificate lets ``_fixed_point_gap`` be at a published mode.

    PIRLS stops once ``|g_0| <= bar sum |s|`` and ``|g_j| <= bar (zeta
    sqrt(D_jj + S_jj) + |S beta|_j)``, ``zeta = sum |s| / sqrt(sum W_F)``,
    ``D_jj = sum W_F x~_j^2`` (``solvers.mode_score``), with ``bar`` the
    certificate's bar for the fit's REML tolerance: the same first-order map
    through ``|H_O^-1|``, with the score's evaluation error added as in
    ``_fixed_point_gap``.
    """
    X, S, beta, mu, slope, fisher, residual = _pirls_rows(model, y, offset, weight)
    observed = fisher - residual
    inverse = np.linalg.inv(X.T @ (observed[:, None] * X) + S)
    score = weight * (y - mu) * slope
    bar = mode_certification_bar(model._reml_profile["reml_tol_resolved"])
    total = float(np.sum(np.abs(score)))
    slopes = X[:, :-1]
    centred = slopes - (fisher @ slopes) / np.sum(fisher)
    zeta = total / np.sqrt(np.sum(fisher))
    certified = bar * np.append(
        zeta * np.sqrt(fisher @ centred**2 + np.diag(S)[:-1]) + np.abs(S[:-1, :-1] @ beta[:-1]),
        total,
    )
    offset_size = 0.0 if offset is None else np.abs(offset)
    d_eta = (np.count_nonzero(X, axis=1) + 1) * EPS * (np.abs(X) @ np.abs(beta) + offset_size)
    rows = np.abs(observed) * d_eta + 16 * EPS * weight * np.abs(slope) * (np.abs(y) + np.abs(mu))
    error = np.abs(X).T @ (rows + 2 * EPS * np.abs(score)) + 2 * EPS * np.abs(S) @ np.abs(beta)
    gap = np.abs(inverse) @ (certified + error)
    return float(np.max(gap)) / max(1.0, float(np.max(np.abs(beta))))


def _held_gap(nested: SuperGLM, dense: SuperGLM, y, offset, weight) -> float:
    """The larger bounded distance to the fixed point of two fits held at the same lambdas."""
    return max(_fixed_point_gap(model, y, offset, weight) for model in (nested, dense))


def _budget(nested: SuperGLM, y, offset, weight, stopping_gap: float = 0.0) -> float:
    """``gamma max(1, ||beta||_inf) (1 + ||X||_inf)``: the same-lambda budget on ``eta``.

    ``gamma = 2 (p + k) eps kappa_s(H) + 2 stopping_gap``, the 2 on the rounding
    term being ``1 / (1 - rho)`` at the asserted contraction.  ``kappa_s`` is
    taken on the intercept-augmented ``H`` both backends solve: a level whose
    penalty is nearly zero is nearly aliased with the intercept, which the slope
    block alone does not show.
    """
    rho = _contraction(nested, y, offset, weight)
    assert rho <= 0.5, f"PIRLS contraction {rho:.3g} exceeds the budget's 1/2"
    state = nested._linear_system_state
    p = state.system.operator.shape[0] + 1
    k = state.system.operator.tree.n_nodes
    H = state.augmented_factor.operator.matvec(np.eye(p))
    gamma = 2.0 * (p + k) * EPS * _scaled_condition(0.5 * (H + H.T)) + 2.0 * stopping_gap
    beta = np.append(nested.result.beta, nested.result.intercept)
    row_norm = float(np.max(np.abs(nested._dm.toarray()).sum(axis=1)))
    return gamma * max(1.0, float(np.max(np.abs(beta)))) * (1.0 + row_norm)


def _relative(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), np.finfo(float).tiny)))


def _assert_nested(model: SuperGLM) -> None:
    profile = model._reml_profile
    assert profile["direct_backend"] == "structured"
    assert profile["structured_chain"] == CHAIN
    assert profile["structured_nested_fallback_reason"] is None
    state = model._linear_system_state
    assert isinstance(state.profiled_factor, ProfiledNestedSchurFactor)
    assert isinstance(state.augmented_factor, NestedSchurFactor)
    assert state.profiled_factor.chain_group_names == CHAIN
    # the retained state's edf and edf1 take the O(k) identity route: the
    # centred data operator wraps the factor's own data operator by identity
    factor = state.profiled_factor
    assert factor._is_centered_data(*factor._pieces(state.centered_data_operator))


def _assert_same_fit(nested: SuperGLM, dense: SuperGLM, frame, y, offset, weight, budget) -> None:
    """Every published quantity of two fits at the same smoothing parameters."""
    scale = budget / (1.0 + float(np.max(np.abs(nested._dm.toarray()).sum(axis=1))))
    np.testing.assert_array_less(np.abs(nested.result.beta - dense.result.beta), scale)
    assert abs(nested.result.intercept - dense.result.intercept) <= scale
    for rows, rows_offset in (
        (frame, offset),
        (_holdout(frame), None if offset is None else offset[:6]),
    ):
        eta_nested = np.log(nested.predict(rows, offset=rows_offset))
        eta_dense = np.log(dense.predict(rows, offset=rows_offset))
        assert np.max(np.abs(eta_nested - eta_dense)) <= budget
    assert _relative(nested.result.deviance, dense.result.deviance) <= 4 * budget
    assert _relative(nested.result.phi, dense.result.phi) <= 4 * budget
    assert abs(nested._reml_result.objective - dense._reml_result.objective) <= 4 * budget * (
        1.0 + abs(dense._reml_result.objective)
    )
    nested_metrics = nested.metrics(frame, y, sample_weight=weight, offset=offset)
    dense_metrics = dense.metrics(frame, y, sample_weight=weight, offset=offset)
    # the nested fit's inference reads its retained factors, not a dense rebuild
    assert isinstance(nested_metrics._active_info[3], StructuredCovarianceAccessor)
    np.testing.assert_array_equal(
        nested_metrics._current_coefficient_estimable,
        dense_metrics._current_coefficient_estimable,
    )
    for nested_edf, dense_edf in zip(
        nested_metrics._influence_edf, dense_metrics._influence_edf, strict=True
    ):
        assert np.max(np.abs(nested_edf - dense_edf)) <= 8 * budget
    for term in TERMS:
        nested_se, dense_se = nested_metrics.feature_se(term), dense_metrics.feature_se(term)
        key = "se" if "se" in dense_se else "se_log_relativity"
        assert _relative(nested_se[key], dense_se[key]) <= 8 * budget


# ── Complete fits ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
@pytest.mark.parametrize("family", FAMILIES)
def test_nested_fit_reproduces_the_dense_fit(family: str, discrete: bool) -> None:
    frame, y, offset, weight = _response(family)
    dense = _fit(family, "gram", discrete=discrete)
    nested = _fit(family, "structured", discrete=discrete)
    assert dense._reml_profile["direct_backend"] == "gram"
    _assert_nested(nested)
    assert dense._reml_result.converged and nested._reml_result.converged
    objective = dense._reml_result.objective
    bound = REML_TOL * (1.0 + abs(objective))
    assert abs(nested._reml_result.objective - objective) <= bound

    lambdas = dict(nested._reml_lambdas)
    dense_held = _fit(family, "gram", discrete=discrete, lambdas=lambdas)
    nested_held = _fit(family, "structured", discrete=discrete, lambdas=lambdas)
    _assert_nested(nested_held)
    gap = _held_gap(nested_held, dense_held, y, offset, weight)
    budget = _budget(nested_held, y, offset, weight, stopping_gap=gap)
    _assert_same_fit(nested_held, dense_held, frame, y, offset, weight, budget)
    if not discrete:
        # the nested optimum is an optimum of the dense surface, and the nested
        # REML fit is the dense fit at its own smoothing parameters; the two
        # come by different paths, so each must have reached the fixed point
        assert abs(dense_held._reml_result.objective - objective) <= bound
        certified = max(_certified_gap(model, y, offset, weight) for model in (nested, dense_held))
        for model in (nested, dense_held):
            assert _fixed_point_gap(model, y, offset, weight) <= certified
        budget = _budget(nested_held, y, offset, weight, stopping_gap=certified)
        _assert_same_fit(nested, dense_held, frame, y, offset, weight, budget)
    assert isinstance(str(nested_held.summary()), str)


def test_a_discrete_fit_converges_whichever_way_its_trial_objectives_round(monkeypatch) -> None:
    """The discrete line search does not ask the objective for digits it lacks.

    Near the optimum a Newton step predicts a decrease ``-g'd`` far below the
    stopping rule's resolution ``tau (1 + |V|)`` (1.7e-14 here, below one ulp
    of ``V = 40.76``), so whether its trial objective lands above or below the
    candidate's is rounding.  On OpenBLAS's Haswell kernels the Gamma/log
    discrete gram fit's trials landed above: a strict ``trial < obj`` halved to
    steps of 2^-24 chosen by rounding until the budget ran out with ``|g|``
    at 1.9e-7 against a bar of 4.2e-8 (CI py3.14 D; Linux ARM64 in #427).
    Every cached trial objective is raised here by ``64 eps (1 + |V|)``, a
    rounding-level error the objective's sums carry, and the fit must still
    converge, to the unperturbed fit's optimum within ``tau (1 + |V|)``.
    Mutation: the strict ``trial_obj < obj`` acceptance (the fit stops at the
    iteration budget, not converged).
    """
    import superglm.reml.discrete as discrete_module

    reference = _fit("gamma", "gram", discrete=True)
    assert reference._reml_result.converged
    original = discrete_module.reml_laml_objective

    def rounded_up(*args, **kwargs):
        value = original(*args, **kwargs)
        if args[5].n_iter == 0 and isinstance(value, float):  # a cached line-search trial
            return value + 64.0 * EPS * (1.0 + abs(value))
        return value

    monkeypatch.setattr(discrete_module, "reml_laml_objective", rounded_up)
    perturbed = _fit("gamma", "gram", discrete=True)
    assert perturbed._reml_result.converged
    objective = reference._reml_result.objective
    assert abs(perturbed._reml_result.objective - objective) <= REML_TOL * (1.0 + abs(objective))


@pytest.mark.parametrize("lam", [1e-6, 1e10], ids=["nearly_free", "nearly_zero"])
@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
def test_extreme_fixed_penalty_on_a_middle_level(lam: float, discrete: bool) -> None:
    frame, y, offset, weight = _response("poisson")
    lambdas = {"x": 3.0, "crossed": 20.0, "make": 5.0, "model": lam, "variant": 20.0}
    dense = _fit("poisson", "gram", discrete=discrete, lambdas=lambdas)
    nested = _fit("poisson", "structured", discrete=discrete, lambdas=lambdas)
    _assert_nested(nested)
    gap = _held_gap(nested, dense, y, offset, weight)
    budget = _budget(nested, y, offset, weight, stopping_gap=gap)
    _assert_same_fit(nested, dense, frame, y, offset, weight, budget)


def _retained_budget(nested: SuperGLM) -> float:
    """``_budget`` with ``kappa_s`` of the retained spectrum of the scaled augmented ``H``.

    Both fits are held, so no stopping gap enters, and Poisson/log is canonical,
    ``rho = 0``; there is nothing to measure (``H`` is singular, the pencil not definite).
    """
    state = nested._linear_system_state
    p = state.system.operator.shape[0] + 1
    k = state.system.operator.tree.n_nodes
    H = state.augmented_factor.operator.matvec(np.eye(p))
    scale = 1.0 / np.sqrt(np.diag(H))
    eigenvalues = np.linalg.eigvalsh(scale[:, None] * (0.5 * (H + H.T)) * scale[None, :])
    retained = eigenvalues[eigenvalues > 1e-10 * eigenvalues[-1]]
    gamma = 2.0 * (p + k) * EPS * float(retained[-1] / retained[0])
    beta = np.append(nested.result.beta, nested.result.intercept)
    row_norm = float(np.max(np.abs(nested._dm.toarray()).sum(axis=1)))
    return gamma * max(1.0, float(np.max(np.abs(beta)))) * (1.0 + row_norm)


def test_aliased_border_categoricals_publish_the_dense_effective_df() -> None:
    """A categorical nested in another aliases border columns: the retained factor
    is Schur-truncated, and its identity routes read ``diag(H^+ H)``, so the
    effective degrees of freedom agree with the dense fit instead of counting
    the nullity (§3.7)."""
    frame, y, offset, weight = _response("poisson")
    area = np.random.default_rng(4).integers(0, 12, len(frame))
    frame = frame.assign(
        area=np.array([f"a{code}" for code in area], dtype=object),
        region=np.array([f"g{code // 4}" for code in area], dtype=object),
    )
    lambdas = {"x": 3.0, "crossed": 20.0, "make": 5.0, "model": 8.0, "variant": 20.0}
    fits = {}
    for direct_solve in ("gram", "structured"):
        model = SuperGLM(
            family="poisson",
            features={
                "x": Spline(n_knots=8, lambda_policy=LambdaPolicy.fixed(lambdas["x"])),
                "cat": Categorical(),
                "region": Categorical(),
                "area": Categorical(),
                **{
                    name: RandomEffect(lambda_policy=LambdaPolicy.fixed(lambdas[name]))
                    for name in ("crossed", *CHAIN)
                },
            },
            selection_penalty=0,
            direct_solve=direct_solve,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(frame, y, sample_weight=weight, offset=offset, pirls_tol=PIRLS_TOL)
        fits[direct_solve] = model
    nested, dense = fits["structured"], fits["gram"]
    _assert_nested(nested)
    factor = nested._linear_system_state.profiled_factor
    # the border is factored by the verified pivoted Cholesky, never a dense fallback
    assert factor.rank_truncated
    nullity = factor.shape[0] - factor.rank
    assert nullity > 0
    budget = _retained_budget(nested)
    assert abs(nested.result.effective_df - dense.result.effective_df) <= 8 * budget * (
        1.0 + dense.result.effective_df
    )
    assert abs(nested.result.effective_df - dense.result.effective_df) < 0.5 * nullity
    nested_metrics = nested.metrics(frame, y, sample_weight=weight, offset=offset)
    dense_metrics = dense.metrics(frame, y, sample_weight=weight, offset=offset)
    for nested_edf, dense_edf in zip(
        nested_metrics._influence_edf, dense_metrics._influence_edf, strict=True
    ):
        assert abs(np.sum(nested_edf) - np.sum(dense_edf)) <= 8 * budget * (
            1.0 + abs(np.sum(dense_edf))
        )
    assert abs(nested_metrics.aic - dense_metrics.aic) <= 8 * budget * (
        1.0 + abs(dense_metrics.aic)
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_reml_derivatives_with_weight_derivatives_match_the_dense_ones(family: str) -> None:
    """The REML gradient and the W-corrected Newton Hessian at fixed lambdas (§3.6).

    The Hessian's pairs go through ``derivative_cross_traces`` of the profiled
    nested factor with the signed operators of ``reml_w_correction``; the REML
    optimum does not depend on them, so the complete fits above cannot see
    them.
    """
    frame, y, offset, weight = _response(family)
    held = {"x": 3.0, "crossed": 20.0, "make": 5.0, "model": 8.0, "variant": 20.0}
    nested_model = _fit(family, "structured", lambdas=held)
    budget = _budget(nested_model, y, offset, weight)
    lambdas = dict(nested_model._reml_lambdas)  # keyed by penalty component
    dm, groups = nested_model._dm, nested_model._groups
    matrices = list(dm.group_matrices)
    penalties, _, _ = build_penalty_context(matrices, collect_reml_groups(groups, matrices))
    family_object, link = nested_model._distribution, nested_model._link
    offset_array = np.zeros(len(y)) if offset is None else offset
    derivatives = {}
    for direct_solve in ("gram", "structured"):
        result, factor = fit_irls_direct(
            X=dm,
            y=y,
            weights=weight,
            family=family_object,
            link=link,
            groups=groups,
            lambda2=lambdas,
            offset=offset_array,
            direct_solve=direct_solve,
            reml_penalties=penalties,
            tol=PIRLS_TOL,
            weight_semantics="prior",
        )
        gradient = reml_direct_gradient(matrices, result, factor, lambdas, reml_penalties=penalties)
        correction = reml_w_correction(
            dm,
            link,
            groups,
            result,
            factor,
            lambdas,
            sample_weight=weight,
            offset_arr=offset_array,
            distribution=family_object,
            w_correction_order=2,
            reml_penalties=penalties,
        )
        operators, second = (None, None) if correction is None else correction[1:]
        hessian = reml_direct_hessian(
            matrices,
            family_object,
            factor,
            lambdas,
            gradient=gradient,
            dH_extra=operators,
            dH2_cross=second,
            reml_penalties=penalties,
        )
        derivatives[direct_solve] = (factor, gradient, correction, hessian)
    factor, gradient, correction, hessian = derivatives["structured"]
    _, dense_gradient, dense_correction, dense_hessian = derivatives["gram"]
    assert isinstance(factor, ProfiledNestedSchurFactor)
    # Gamma/log has constant Fisher weights: no W derivative on either side
    assert (correction is None) == (dense_correction is None) == (family == "gamma")
    assert np.max(np.abs(gradient - dense_gradient)) <= 8 * budget * (
        1.0 + np.max(np.abs(dense_gradient))
    )
    if correction is not None:
        first, second = correction[0], correction[2]
        dense_first, dense_second = dense_correction[0], dense_correction[2]
        assert np.max(np.abs(first - dense_first)) <= 8 * budget * np.max(np.abs(dense_first))
        cs = np.sqrt(np.outer(np.abs(np.diag(dense_second)), np.abs(np.diag(dense_second))))
        assert np.all(np.abs(second - dense_second) <= 8 * budget * cs)
    cs = np.sqrt(np.outer(np.abs(np.diag(dense_hessian)), np.abs(np.diag(dense_hessian))))
    assert np.all(np.abs(hessian - dense_hessian) <= 8 * budget * cs)


# ── Routing ───────────────────────────────────────────────────────────────


def test_implicit_nesting_is_not_chained() -> None:
    """Variant labels reused under several models are crossed by their counts (§3.7).

    Each label now names two variants of different models, so the variant term
    is still the largest random effect but no function of the model code: it is
    a chain of one.
    """
    frame, y, offset, weight = _response("poisson")
    codes = pd.factorize(frame["variant"], sort=True)[0]
    implicit = frame.assign(variant=np.array([f"v{code % 40}" for code in codes], dtype=object))
    assert implicit.groupby("variant")["model"].nunique().max() > 1
    model = _fit("poisson", "structured", data=(implicit, y, offset, weight))
    assert model._reml_profile["structured_chain"] == ("variant",)
    assert isinstance(model._linear_system_state.profiled_factor, ProfiledNestedSchurFactor)


def test_a_parent_with_more_levels_than_the_block_cap_reports_standard_errors() -> None:
    """A parent random effect over the 256-coefficient block cap (§6) through ``feature_se``."""
    frame, eta, exposure, _ = _data(sizes=(300, 900), n=4000, seed=5)
    frame = frame.drop(columns="model")
    y = np.random.default_rng(6).poisson(exposure * np.exp(eta)).astype(float)
    offset = np.log(exposure)  # one array: metrics recognise the fit rows by it
    lambdas = {"make": 4.0, "variant": 10.0}
    fits = {}
    for direct_solve in ("gram", "structured"):
        model = SuperGLM(
            family="poisson",
            features={
                "x": Spline(n_knots=6, lambda_policy=LambdaPolicy.fixed(2.0)),
                **{
                    name: RandomEffect(lambda_policy=LambdaPolicy.fixed(lambdas[name]))
                    for name in lambdas
                },
            },
            selection_penalty=0,
            direct_solve=direct_solve,
        )
        model.fit_reml(frame, y, offset=offset, pirls_tol=PIRLS_TOL)
        fits[direct_solve] = model
    nested, dense = fits["structured"], fits["gram"]
    assert nested._reml_profile["structured_chain"] == ("make", "variant")
    assert (
        frame["make"].nunique()
        > nested._linear_system_state.profiled_factor.max_structured_inverse_block
    )
    nested_metrics = nested.metrics(frame, y, offset=offset)
    assert isinstance(nested_metrics._active_info[3], StructuredCovarianceAccessor)
    nested_se = nested_metrics.feature_se("make")["se"]
    dense_se = dense.metrics(frame, y, offset=offset).feature_se("make")["se"]
    assert np.all(np.isfinite(nested_se)) and np.all(nested_se > 0.0)
    ones = np.ones(len(y))
    gap = _held_gap(nested, dense, y, offset, ones)
    assert _relative(nested_se, dense_se) <= 8 * _budget(nested, y, offset, ones, stopping_gap=gap)


# ── Estimability ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("leaf_constant", [False, True])
def test_border_estimability_is_invariant_to_a_column_offset(leaf_constant: bool) -> None:
    """Adding a constant to a numeric column cannot change what the data identify.

    The border's decision is taken on the Schur complement of the leaf block,
    the within-leaf scatter (Lovell 1963).  The raw moments of
    ``x = 1e7 + N(0, 1)`` are ``1e14`` times that scatter, beyond what a
    float64 difference resolves, so the reduction runs on the rows ``x - c``
    the leaf statistics hold (Chan, Golub & LeVeque 1983, §3); ``1e12`` is an
    epoch-millisecond column.  A column constant within every leaf lies in the
    span of the leaf indicators at any offset and stays non-estimable.  The
    compact reduction is called directly: the public dispatcher's dense
    fallback for narrow systems would hide a reduction that raises.
    """
    from superglm import Numeric
    from superglm.solvers._structured.geometry import _nested_centered_estimability

    rng = np.random.default_rng(0)
    n, roots, leaves = 900, 10, 30
    parent = np.concatenate([np.arange(roots), rng.integers(0, roots, leaves - roots)])
    leaf = rng.integers(0, leaves, n)
    spread = rng.normal(size=leaves)[leaf] if leaf_constant else rng.normal(size=n)
    y = (
        0.5 * spread
        + rng.normal(0.0, 0.5, roots)[parent[leaf]]
        + rng.normal(0.0, 0.3, leaves)[leaf]
        + rng.normal(size=n)
    )
    frame = pd.DataFrame(
        {
            "root": np.array([f"r{code}" for code in parent[leaf]], dtype=object),
            "leaf": np.array([f"l{code}" for code in leaf], dtype=object),
        }
    )
    decisions = []
    for offset in (0.0, 1e7, 1e12):
        data = frame.assign(x=offset + spread)
        model = SuperGLM(
            family="gaussian",
            features={"x": Numeric(), "root": RandomEffect(), "leaf": RandomEffect()},
            selection_penalty=0,
            direct_solve="structured",
        )
        model.fit_reml(data, y)
        state = model._linear_system_state
        assert model._reml_profile["structured_chain"] == ("root", "leaf")
        assert isinstance(state.augmented_factor, NestedSchurFactor)
        operator = state.centered_data_operator
        estimable = _nested_centered_estimability(operator, operator.raw)
        x = next(group.sl for group in model._groups if group.name == "x")
        assert bool(estimable[x][0]) is not leaf_constant
        se = model.metrics(data, y).coefficient_se["x"]
        assert bool(np.isfinite(se[0])) is not leaf_constant
        decisions.append(estimable)
    for decision in decisions[1:]:
        np.testing.assert_array_equal(decision, decisions[0])


# ── A chain of one ────────────────────────────────────────────────────────
#
# A lone random effect is a chain of one on NestedSchurFactor.  These fixtures
# are the classes on which the retired ScalarSchurFactor refused or erred (selection.py):
# aliased or constant border columns, extreme lambda times weight, and a
# column offset.  Gaussian/identity fits at fixed lambda make PIRLS one exact
# solve (rho = 0, no stopping gap), so the backends differ only by rounding.

LONE_LEVELS = 30


@cache
def _lone_level_data(extra: str | None = None, seed: int = 3):
    """One random effect of 30 declared levels (one never observed, two with zero
    weight) beside a normal column, a column of mean 10, a level attribute and a
    12-level categorical; ``extra`` adds one border column."""
    rng = np.random.default_rng(seed)
    n = 1500
    popularity = rng.gamma(1.5, size=LONE_LEVELS - 1)
    g = rng.choice(LONE_LEVELS - 1, size=n, p=popularity / popularity.sum())
    cat = rng.integers(0, 12, n)
    x1, x10 = rng.normal(size=n), 10.0 + rng.normal(size=n)
    attribute = rng.normal(size=LONE_LEVELS)[g]
    frame = pd.DataFrame(
        {
            "x1": x1,
            "x10": x10,
            "attr": attribute,
            "cat": np.array([f"c{c:02d}" for c in cat], dtype=object),
            "g": np.array([f"g{c:02d}" for c in g], dtype=object),
        }
    )
    eta = (
        0.2
        + 0.3 * x1
        + 0.05 * (x10 - 10.0)
        + 0.2 * attribute
        + rng.normal(0.0, 0.2, 12)[cat]
        + rng.normal(0.0, 0.3, LONE_LEVELS)[g]
    )
    columns = {
        None: {},
        "duplicate": {"x1dup": x1.copy()},
        "alias": {"catval": 0.5 * cat - 1.0},
        "constant": {"const": np.full(n, 3.0)},
        "offset": {"x1": 1e6 + x1},
    }[extra]
    frame = frame.assign(**columns)
    weight = np.where(np.isin(g, np.unique(g)[[3, 7]]), 0.0, 1.0)
    y = eta + rng.normal(0.0, 0.5, n)
    return frame, y, weight


def _lone_level_fit(extra, direct_solve, lam, weight_scale=1.0, *, data=None, link=None):
    """A fit of ``_lone_level_data(extra)``, or of ``data``, a binary response
    under ``link``."""
    frame, y, weight = _lone_level_data(extra) if data is None else data
    numerics = [name for name in frame.columns if name not in ("cat", "g")]
    model = SuperGLM(
        family="gaussian" if link is None else "binomial",
        link=link,
        features={
            **{name: Numeric() for name in numerics},
            "cat": Categorical(),
            "g": RandomEffect(
                levels=[f"g{c:02d}" for c in range(LONE_LEVELS)],
                lambda_policy=LambdaPolicy.fixed(lam),
            ),
        },
        selection_penalty=0,
        direct_solve=direct_solve,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y, sample_weight=weight * weight_scale, pirls_tol=PIRLS_TOL)
    return model


def _centred_rows(model: SuperGLM, weight) -> tuple[np.ndarray, np.ndarray]:
    """The design less its weighted column means, by the shifted two-pass form (a
    constant column is exactly zero), and ``H_c = X_c' W X_c + S``."""
    dm = model._dm
    X = dm.toarray()
    shifted = X - X[np.flatnonzero(weight)[0]]
    Xc = shifted - (weight @ shifted) / np.sum(weight)
    S = build_penalty_matrix(
        dm.group_matrices, model._groups, model._reml_lambdas, dm.shape[1], model._reml_penalties
    )
    return Xc, Xc.T @ (weight[:, None] * Xc) + S


def _centred_budget(chain: SuperGLM, weight) -> tuple[float, float]:
    """``(gamma, budget)``: the same-lambda bound on ``eta`` in centred coordinates.

    ``gamma = 2 (p + 1 + k) eps kappa_s(H_c)`` and ``|d eta| <= gamma max(1,
    ||beta||_inf) (1 + ||X_c||_inf)``, as ``_budget``, but ``kappa_s`` is that of
    the intercept-profiled ``H_c`` both backends solve, not of the augmented
    ``H``, whose Jacobi scaling a column offset of 1e6 conditions by its mean.
    It is taken over the retained spectrum, the rank the fit reports; exactly
    zero centred columns (a constant) carry no curvature and are left out.
    """
    Xc, H = _centred_rows(chain, weight)
    live = np.diag(H) > 0.0
    scale = 1.0 / np.sqrt(np.diag(H)[live])
    eigenvalues = np.linalg.eigvalsh(scale[:, None] * H[np.ix_(live, live)] * scale[None, :])
    retained = eigenvalues[-(chain.result.reml_hessian_rank - 1) :]
    p = H.shape[0] + 1
    gamma = 2.0 * (p + LONE_LEVELS) * EPS * float(retained[-1] / retained[0])
    beta = max(1.0, float(np.max(np.abs(chain.result.beta))))
    return gamma, gamma * beta * (1.0 + float(np.max(np.abs(Xc).sum(axis=1))))


def _assert_chain_of_one_is_gram(chain: SuperGLM, gram: SuperGLM, extra, weight) -> None:
    """The chain of one and gram at one lambda: rank, estimability and every fitted
    value.  The objective and the pseudo-determinant take ``4 budget``, as
    ``_assert_same_fit``; ``log|H|`` does not depend on the fit here, so its
    retained eigenvalues move by ``gamma / 2`` relatively at most, each."""
    assert isinstance(chain._linear_system_state.profiled_factor, ProfiledNestedSchurFactor)
    assert chain._reml_profile["structured_chain"] == ("g",)
    assert chain.result.direct_fallback_reason is None
    assert chain.result.reml_hessian_rank == gram.result.reml_hessian_rank
    frame, y, _ = _lone_level_data(extra)
    gamma, budget = _centred_budget(chain, weight)
    # each prediction also rounds its own sum X beta + alpha, whose terms a column
    # offset makes large: (m + 1) eps (|X| |beta| + |alpha|) per row, m its nonzeros
    X = chain._dm.toarray()
    evaluation = sum(
        (np.count_nonzero(X, axis=1) + 1)
        * EPS
        * (np.abs(X) @ np.abs(model.result.beta) + abs(model.result.intercept))
        for model in (chain, gram)
    )
    assert np.all(np.abs(chain.predict(frame) - gram.predict(frame)) <= budget + evaluation)
    objective = gram._reml_result.objective
    assert abs(chain._reml_result.objective - objective) <= 4 * budget * (1.0 + abs(objective))
    rank = gram.result.reml_hessian_rank
    assert abs(chain.result.log_det_H - gram.result.log_det_H) <= rank * gamma
    chain_metrics = chain.metrics(frame, y, sample_weight=weight)
    gram_metrics = gram.metrics(frame, y, sample_weight=weight)
    np.testing.assert_array_equal(
        chain_metrics._current_coefficient_estimable, gram_metrics._current_coefficient_estimable
    )


def _lone_level_pair(extra, lam, weight_scale=1.0):
    fits = [_lone_level_fit(extra, solve, lam, weight_scale) for solve in ("auto", "gram")]
    return (*fits, _lone_level_data(extra)[2] * weight_scale)


@pytest.mark.parametrize(
    ("extra", "weight_scale"),
    [pytest.param("duplicate", 1e2, id="duplicate-x1e2"), pytest.param("alias", 1.0, id="alias")],
)
def test_a_chain_of_one_decides_an_aliased_border_as_gram_does(extra, weight_scale) -> None:
    """A duplicated column under weights x1e2, and a numeric that is a function of
    the categorical: the border has one exact null direction.  The chain forms Q
    as a sum of PSD pieces and decides rank on the Jacobi-scaled Q, so it keeps
    nullity 1 and gram's estimability where the retired scalar factor, forming Q by
    subtraction, refused 12 of 22 and 20 of 44 such fits."""
    chain, gram, weight = _lone_level_pair(extra, 0.7, weight_scale)
    width = chain._dm.shape[1] + 1
    assert width - chain.result.reml_hessian_rank == 1
    _assert_chain_of_one_is_gram(chain, gram, extra, weight)


@pytest.mark.parametrize("lam", [0.7, 1e8], ids=["moderate", "nearly-zero"])
def test_a_chain_of_one_drops_a_constant_column_as_gram_does(lam) -> None:
    """A column equal to 3 is the intercept's alias.  The chain reduces the border on
    the rows less the global centre, where the column is exactly zero: it is
    non-estimable and nothing else is, as in gram, and the objective carries no
    pseudo-determinant offset (the retired scalar factor's was 0.5 ln(1 + 3^2) = 1.1513)."""
    chain, gram, weight = _lone_level_pair("constant", lam)
    const = next(group.sl for group in chain._groups if group.name == "const")
    frame, y, _ = _lone_level_data("constant")
    estimable = chain.metrics(frame, y, sample_weight=weight)._current_coefficient_estimable
    assert not estimable[const][0]
    _assert_chain_of_one_is_gram(chain, gram, "constant", weight)


def test_a_chain_of_one_certifies_an_observed_mode_beside_a_constant_column() -> None:
    """Binomial/probit, whose observed rows reach the mode score, beside the
    constant column.  The chain's centred operator gives the column an exactly
    zero diagonal, so its score is ``(3 - mean_x) sum(r)``, the weighted mean's
    rounding times the intercept score; normalised by ``tiny`` it scored 5.4e28
    and auto raised ObservedModeNotCertifiedError where gram fits.  Normalised
    at the centring's resolution it certifies, and the fit is gram's on the
    model without the column, whose objective it must meet to
    ``reml_tol (1 + |V|)``."""
    frame, latent, weight = _lone_level_data("constant", seed=4)
    y = (latent > 0.2).astype(float)  # the latent-variable form of a probit response
    reduced = frame.drop(columns="const")
    chain = _lone_level_fit(None, "auto", 0.7, data=(frame, y, weight), link="probit")
    gram = _lone_level_fit(None, "gram", 0.7, data=(reduced, y, weight), link="probit")
    assert isinstance(chain._linear_system_state.profiled_factor, ProfiledNestedSchurFactor)
    assert chain.result.direct_fallback_reason is None
    assert chain.result.reml_hessian_rank == gram.result.reml_hessian_rank
    objective = gram._reml_result.objective
    assert abs(chain._reml_result.objective - objective) <= REML_TOL * (1.0 + abs(objective))
    estimable = chain.metrics(frame, y, sample_weight=weight)._current_coefficient_estimable
    const = next(group.sl for group in chain._groups if group.name == "const")
    assert not estimable[const][0]
    np.testing.assert_array_equal(
        np.delete(estimable, const),
        gram.metrics(reduced, y, sample_weight=weight)._current_coefficient_estimable,
    )


def test_a_chain_of_one_fits_a_column_offset_by_a_million() -> None:
    """``1e6 + x``: the raw moments carry the offset, which the chain's global centre
    removes before any leaf statistic is formed (Chan, Golub & LeVeque 1983).  The
    retired scalar factor refused 16 of 16 such fits."""
    chain, gram, weight = _lone_level_pair("offset", 0.7)
    _assert_chain_of_one_is_gram(chain, gram, "offset", weight)


def _dyadic(values: np.ndarray) -> tuple[np.ndarray, int]:
    """``(N, e)`` with ``values = N / 2^e`` exactly, ``N`` Python integers."""
    exponent = max(
        float(value).as_integer_ratio()[1].bit_length() - 1 for value in np.unique(values)
    )
    return np.array(
        [int(Fraction(float(value)) * 2**exponent) for value in values.ravel()], dtype=object
    ).reshape(values.shape), exponent


def _exact_logdet_and_edf(model: SuperGLM, weight) -> tuple[float, float, float]:
    """``log|H_aug|``, the edf and ``kappa_s`` of the centred border Schur complement,
    from the float64 rows in exact integer arithmetic.

    Every float64 is dyadic, so ``H_aug = [X 1]' W [X 1] + blockdiag(S, 0)`` is an
    integer matrix over one power of two.  Fraction-free Gauss-Jordan (Bareiss
    1968) gives its determinant and adjugate; the edf is ``p + 1 - tr(H_aug^-1
    S_aug)``, the slope block of ``H_aug^-1`` being ``H_c^-1``.  ``Q`` eliminates
    the random effect's diagonal block, centred at the weighted column means.
    """
    dm = model._dm
    X = np.hstack([dm.toarray(), np.ones((dm.shape[0], 1))])
    size = X.shape[1]
    S = np.zeros((size, size))
    S[:-1, :-1] = build_penalty_matrix(
        dm.group_matrices, model._groups, model._reml_lambdas, size - 1, model._reml_penalties
    )
    (Xi, ex), (wi, ew), (Si, es) = _dyadic(X), _dyadic(weight), _dyadic(S)
    total = max(2 * ex + ew, es)
    H = (Xi.T @ (wi[:, None] * Xi)) * 2 ** (total - 2 * ex - ew) + Si * 2 ** (total - es)
    M = [list(row) + [int(i == j) for j in range(size)] for i, row in enumerate(H.tolist())]
    previous = 1
    for k in range(size):
        pivot = M[k][k]
        for i in range(size):
            if i != k:
                factor = M[i][k]
                M[i] = [(a * pivot - factor * b) // previous for a, b in zip(M[i], M[k])]
        previous = pivot
    adjugate = np.array([row[size:] for row in M], dtype=object)
    logdet = math.log(previous) - size * total * math.log(2.0)
    trace = Fraction(int((adjugate * Si.T).sum()) * 2 ** (total - es), previous)
    level = next(group.sl for group in model._groups if group.name == "g")
    tree = np.arange(level.start, level.stop)
    border = np.setdiff1d(np.arange(size), tree)
    Q = [
        [
            Fraction(H[i, j]) - sum(Fraction(H[i, t] * H[j, t], H[t, t]) for t in tree)
            for j in border
        ]
        for i in border
    ]
    # the intercept, last, is sheared onto the weighted column means (its own is 0)
    mean = [Fraction(H[-1, j], H[-1, -1]) for j in border[:-1]] + [Fraction(0)]
    R = np.array(
        [[Fraction(int(i == j)) for j in range(len(border))] for i in range(len(border))],
        dtype=object,
    )
    R[-1] -= np.array(mean, dtype=object)
    Qc = R.T @ np.array(Q, dtype=object) @ R
    Qc = np.array([[float(value) for value in row] for row in Qc])
    scale = 1.0 / np.sqrt(np.diag(Qc))
    kappa = float(np.linalg.cond(scale[:, None] * Qc * scale[None, :]))
    return logdet, float(size - trace), kappa


def test_a_chain_of_one_is_exact_at_a_tiny_lambda_under_large_weights() -> None:
    """lambda = 1e-8 on the random effect under weights x1e4.  The intercept's Schur
    pivot is ``sum_l w_l lambda / (w_l + lambda)``, about ``K lambda``: formed by
    subtraction it cancels to a relative 1e-2, which the retired scalar factor refused
    and gram's Cholesky of the augmented H carries into log|H|.  The chain sums
    it from positive terms.  Bounds as in ``test_nested_schur_factor.py``:
    ``gamma_tree = n_tree eps`` for the per-level sums (the largest level's rows
    and four operations), ``gamma_Q = (n + k + q + 10) eps`` for the PSD-sum Q
    and ``gamma_border = q kappa_s(Q) gamma_Q``; ``log|H|`` takes ``k gamma_tree +
    q gamma_border`` and the edf, a sum of ``p + 1`` terms at most 1, ``(p + 1)
    (gamma_tree + gamma_border)``.
    """
    chain = _lone_level_fit(None, "auto", 1e-8, weight_scale=1e4)
    assert isinstance(chain._linear_system_state.profiled_factor, ProfiledNestedSchurFactor)
    frame, _, weight = _lone_level_data(None)
    logdet, edf, kappa = _exact_logdet_and_edf(chain, weight * 1e4)
    n, size = len(frame), chain._dm.shape[1] + 1
    q = size - LONE_LEVELS
    rows = int(np.max(np.bincount(pd.factorize(frame["g"])[0])))
    gamma_tree = (rows + 4) * EPS
    gamma_border = q * kappa * (n + LONE_LEVELS + q + 10) * EPS
    assert abs(chain.result.log_det_H - logdet) <= LONE_LEVELS * gamma_tree + q * gamma_border
    assert abs(chain.result.effective_df - edf) <= size * (gamma_tree + gamma_border)
