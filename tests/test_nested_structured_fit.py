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
from functools import cache

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from superglm import (
    Categorical,
    LambdaPolicy,
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
from superglm.solvers.structured import (
    NestedSchurFactor,
    ProfiledNestedSchurFactor,
    ProfiledScalarSchurFactor,
)

EPS = np.finfo(np.float64).eps
PIRLS_TOL = 1e-10
REML_TOL = 1e-9
CHAIN = ("make", "model", "variant")
TERMS = ("x", "cat", "crossed", *CHAIN)
PENALIZED = ("x", "crossed", *CHAIN)
# response seeds with an interior REML optimum at every level (see ``_data``)
SEEDS = {"poisson": 8, "gamma": 1, "tweedie": 2, "binomial": 3}
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
    return Tweedie(p=1.5) if name == "tweedie" else name


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
    assert isinstance(state.coefficient_factor, NestedSchurFactor)
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
        for model in (nested, dense_held):
            assert _fixed_point_gap(model, y, offset, weight) <= PIRLS_TOL
        budget = _budget(nested_held, y, offset, weight, stopping_gap=PIRLS_TOL)
        _assert_same_fit(nested, dense_held, frame, y, offset, weight, budget)
    assert isinstance(str(nested_held.summary()), str)


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
    assert factor.used_dense_fallback and factor.rank_truncated
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


def test_gaussian_log_declines_the_chain_and_keeps_the_single_level_backend() -> None:
    """Observed Gaussian/log weights can be negative (§3.7): the chain is declined, recorded."""
    frame, eta, _, weight = _data()
    y = np.exp(eta) * (1.0 + 0.2 * np.random.default_rng(9).normal(size=len(eta)))
    model = SuperGLM(
        family="gaussian",
        link="log",
        features={
            "x": Spline(n_knots=8),
            "cat": Categorical(),
            **{name: RandomEffect() for name in ("crossed", *CHAIN)},
        },
        selection_penalty=0,
        direct_solve="structured",
    )
    model.fit_reml(frame, y, sample_weight=weight, pirls_tol=PIRLS_TOL, reml_tol=REML_TOL)
    profile = model._reml_profile
    assert profile["direct_backend"] == "structured"
    assert profile["structured_chain"] == ("variant",)
    assert "Gaussian/LogLink" in profile["structured_nested_fallback_reason"]
    assert isinstance(model._linear_system_state.profiled_factor, ProfiledScalarSchurFactor)


def test_implicit_nesting_is_not_chained() -> None:
    """Variant labels reused under several models are crossed by their counts (§3.7).

    Each label now names two variants of different models, so the variant term
    is still the largest random effect but no function of the model code.
    """
    frame, y, offset, weight = _response("poisson")
    codes = pd.factorize(frame["variant"], sort=True)[0]
    implicit = frame.assign(variant=np.array([f"v{code % 40}" for code in codes], dtype=object))
    assert implicit.groupby("variant")["model"].nunique().max() > 1
    model = _fit("poisson", "structured", data=(implicit, y, offset, weight))
    assert model._reml_profile["structured_chain"] == ("variant",)
    assert isinstance(model._linear_system_state.profiled_factor, ProfiledScalarSchurFactor)


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
