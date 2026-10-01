"""The centred state at large column offsets (issue #430): no raw reconstruction left.

A ``Numeric`` column far from zero (an epoch time, an ID) makes the raw intercept
``alpha - c' beta`` cancel against ``X beta`` (one-engine design §3.8).  Each test
translates such a column exactly (its values on a grid the offset represents)
and checks that the fit, its REML, its certificates and its diagnostics read the
same model as at no offset.  Every test fails under the mutation its
docstring names; every test but the review-edge guard of the unprofiled
W-correction also fails on the code before #430.

Bounds derive from dimensions, ``eps``, the REML stopping rule and the fits'
own predictors (``mode_score.linear_predictor``, which reads the centred
state), never from what passed locally.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from superglm import (
    Categorical,
    FactorSmooth,
    LambdaPolicy,
    Numeric,
    Polynomial,
    RandomEffect,
    Spline,
    SuperGLM,
)
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.solvers.mode_score import linear_predictor
from tests.test_factor_smooth_sz_thin_and_influence import _frame, _model, _penalized_objective

EPS = float(np.finfo(np.float64).eps)
_U = EPS / 2.0


def _gamma(count: float) -> float:
    return count * _U / (1.0 - count * _U)


def _reml_tol(model: SuperGLM) -> float:
    """The stopping tolerance the fit's REML engine resolved and ran with."""
    return float(model._reml_profile["reml_tol_resolved"])


def _fit_reml(model: SuperGLM, frame, y) -> SuperGLM:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit_reml(frame, y)


def _assert_same_reml(base: SuperGLM, shifted: SuperGLM) -> None:
    """Two REML runs on one model, translated: the same termination at the same optimum.

    The optimizer stops once the objective resolves no change beyond
    ``reml_tol (1 + |V|)``, so two runs that stop certified sit within twice
    that of each other.  Reading the stop rule as a bound on the distance to
    the optimum's value holds once the outer Newton steps converge
    superlinearly, as they do on these well-determined fixtures.
    """
    first, second = base._reml_result, shifted._reml_result
    assert first.converged and second.converged
    assert second.termination_reason == first.termination_reason
    objective = abs(float(first.objective))
    assert abs(float(second.objective) - float(first.objective)) <= 2.0 * _reml_tol(base) * (
        1.0 + objective
    )


def _offset_frame(shift: float, n: int = 240, seed: int = 10):
    """Sol's fixture: an even-integer column (exact at a 1e16 offset) beside 40 levels."""
    rng = np.random.default_rng(seed)
    g = np.tile(np.arange(40), n // 40)
    z = 2.0 * rng.integers(-4, 5, n)
    return rng, g, z, pd.DataFrame({"x": shift + z, "g": g})


# ---------------------------------------------------------- 1. cached trials
@pytest.mark.parametrize("shift", [1e12, 1e16])
def test_discrete_cached_trials_carry_the_factors_centred_intercept(shift):
    """A discrete structured trial keeps the factor's centred ``alpha`` (item 1).

    The cached trial took ``alpha`` back from the factor's rounded raw
    intercept, ``intercept + c' beta``, which errs by ``u |c' beta|`` in every
    row's eta (0.25 at 1e16): the trial objectives were noise and REML ran to
    ``max_reml_iter`` with lambda 2.84% off (Sol's reproduction).  Mutation:
    ``centred_intercept_trial = cached_solution.intercept + fsum(c * beta)``
    in ``reml.discrete``.  A Gaussian fit's one-step candidates do not depend
    on their start, so this isolates the trial from the warm start (item 4).
    """
    rng, g, z, _ = _offset_frame(0.0)
    y = 3.0 + 0.2 * z + 0.3 * np.sin(g) + 0.05 * rng.normal(size=len(z))

    def fit(offset: float) -> SuperGLM:
        model = SuperGLM(
            family="gaussian",
            features={"x": Numeric(), "g": RandomEffect()},
            selection_penalty=0,
            discrete=True,
        )
        return _fit_reml(model, pd.DataFrame({"x": offset + z, "g": g}), y)

    base, shifted = fit(0.0), fit(shift)
    assert shifted._reml_profile["direct_backend"] == "structured"
    _assert_same_reml(base, shifted)


@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_discrete_factor_smooth_trials_carry_the_leaf_centred_intercept(basis):
    """Item 1 through the fs and sz leaf factors' cached solves.

    ``solve_cached_block_structured`` and ``solve_cached_sum_to_zero_structured``
    read the border centre from the leaf system (``system.leaf.center``), the
    nested solve from its operator's.  ``x1`` sits in the border at 1e16.
    Mutation: a zero centre for a leaf system in
    ``assembly._cached_centred_solution`` (the trial's ``alpha`` is then
    carried as if it were about zero).
    """
    frame, _, rng, _ = _frame(K=12, n=1500, seed=5)
    z = 2.0 * rng.integers(-4, 5, len(frame)).astype(float)
    signal = 0.5 + 0.2 * z + np.sin(2 * np.pi * frame["x"].to_numpy())
    y = signal + rng.normal(0.0, 0.3, len(z))

    def fit(shift: float) -> SuperGLM:
        translated = frame.copy()
        translated["x1"] = shift + z
        features = {"x1": Numeric(), "cat": Categorical()}
        if basis == "sz":
            features["x"] = Spline(n_knots=6)
        model = SuperGLM(
            family="gaussian",
            features=features,
            interactions=[FactorSmooth("x", group="g", basis=basis, k=5)],
            selection_penalty=0,
            direct_solve="structured",
            discrete=True,
        )
        return _fit_reml(model, translated, y)

    base, shifted = fit(0.0), fit(1e16)
    assert shifted._reml_profile["direct_backend"] == "structured"
    _assert_same_reml(base, shifted)


# --------------------------------------------------- 2. sz minimum-norm step
def test_sz_minimum_norm_step_forms_its_score_about_the_centre():
    """A truncated ``sz`` factor's increment solves the centred score (item 2).

    Two one-row levels truncate the balance tree, so every step is the
    minimum-norm increment ``H^+ g``.  Its score took the border's raw ``X' r``
    and the factor subtracted ``c`` times the intercept's ``sum r``: at a 1e16
    offset no digit was left, the step was rejected and ``x1``'s coefficient
    moved in the third digit (Sol).  Both fits are certified modes at the
    same lambdas, so their penalized deviances are within ``4 phi reml_tol
    (1 + |V|)`` (``test_factor_smooth_sz_thin_and_influence._assert_fits_as_gram``,
    the envelope argument).  Mutation: ``dm.rmatvec`` for
    ``centred_data_score`` and no ``border_centred`` in ``irls_direct``.
    """
    frame, _, rng, _ = _frame(K=8, n=800, seed=3, one_row=2)
    z = 2.0 * rng.integers(-4, 5, len(frame)).astype(float)
    y = 3.0 + 0.2 * z + rng.normal(0.0, 0.15, len(z))
    fits = {}
    for shift in (0.0, 1e16):
        translated = frame.copy()
        translated["x1"] = shift + z
        fits[shift] = _fit_reml(_model("gaussian", "structured", k=5), translated, y)
    base, shifted = fits[0.0], fits[1e16]
    for model in (base, shifted):
        assert model._reml_profile["direct_backend"] == "structured"
        assert model.result.converged
        assert model.result.termination_reason == "converged"
    criterion = abs(float(base._reml_result.objective))
    tolerance = 4.0 * float(base.result.phi) * _reml_tol(base) * (1.0 + criterion)
    assert abs(_penalized_objective(shifted) - _penalized_objective(base)) <= tolerance


# --------------------------------------------- 3. the certificates' intercept
def _gamma_log_fit(shift: float, spy) -> tuple[SuperGLM, dict]:
    """Gamma/log (observed REML geometry) at fixed lambdas: the terminal certificate."""
    rng, g, z, frame = _offset_frame(shift, n=480, seed=4)
    eta = 0.4 + 0.1 * z + 0.3 * np.sin(g)
    y = np.random.default_rng(5).gamma(4.0, np.exp(eta) / 4.0)
    model = SuperGLM(
        family="gamma",
        link="log",
        features={"x": Numeric(), "g": RandomEffect(lambda_policy=LambdaPolicy.fixed(2.0))},
        selection_penalty=0,
    )
    captured: dict = {}
    spy(captured)
    _fit_reml(model, frame, y)
    return model, captured


@pytest.mark.parametrize("module", ["reml.observed_geometry", "solvers.irls_direct"])
def test_the_mode_certificates_read_the_centred_intercept(module, monkeypatch):
    """The observed-mode and PIRLS certificates read ``alpha`` and ``X~ beta`` centred (item 3).

    Their backward-error floors take the mode's intercept about the working
    means, ``alpha = intercept + mean_x' beta``, and ``X~ beta = X beta -
    mean_x' beta``: from the raw intercept and ``X beta`` both cancel ``c'
    beta`` (~0.1 here at 1e16).  ``alpha`` is the weighted mean of ``eta``
    under the geometry's weights, so it is translation invariant: two fits
    whose predictors differ by ``d = max|eta_1 - eta_0|`` move it by at most
    ``d`` through the rows and ``2 d max|eta - alpha|`` through the weights
    (relative changes of ``d``), and ``X~ beta`` row by row by ``d`` plus
    that; each evaluation rounds within ``gamma_n`` of ``max|eta|``.
    Mutation: ``alpha = float(result.intercept) + float(mean_x @ beta)`` in
    ``observed_penalized_mode_score``, or ``intercept_value + shift`` in
    ``irls_direct``'s ``mode_residual``.
    """
    import importlib

    target = importlib.import_module(f"superglm.{module}")
    real = target.penalized_mode_residual

    def install(captured: dict) -> None:
        def spy(**kwargs):
            captured["alpha"] = float(kwargs["alpha"])
            captured["eta_tilde"] = np.array(kwargs["eta_tilde"], dtype=np.float64)
            return real(**kwargs)

        monkeypatch.setattr(target, "penalized_mode_residual", spy)

    base, base_inputs = _gamma_log_fit(0.0, install)
    shifted, shifted_inputs = _gamma_log_fit(1e16, install)
    assert base_inputs and shifted_inputs
    eta_base = linear_predictor(base._dm, base._solver_pirls_result(), None)
    eta_shifted = linear_predictor(shifted._dm, shifted._solver_pirls_result(), None)
    moved = float(np.max(np.abs(eta_shifted - eta_base)))
    alpha = base_inputs["alpha"]
    spread = float(np.max(np.abs(eta_base - alpha)))
    rounding = 4.0 * _gamma(len(eta_base)) * float(np.max(np.abs(eta_base)))
    alpha_bound = moved * (1.0 + 2.0 * spread) + rounding
    assert abs(shifted_inputs["alpha"] - alpha) <= alpha_bound
    np.testing.assert_array_less(
        np.abs(shifted_inputs["eta_tilde"] - base_inputs["eta_tilde"]),
        moved + alpha_bound + rounding,
    )


# ------------------------------------------------------------ 4. warm starts
@pytest.mark.parametrize("family", ["poisson", "gamma"])
def test_discrete_reml_warm_starts_carry_the_centred_state(family):
    """Each discrete candidate starts from the warm state's ``(alpha, c)`` (item 4).

    A discrete REML candidate is one PIRLS step from the previous state,
    with that state's deviance supplied.  Started from its raw intercept the
    state's eta erred by ``u |c' beta|`` (~0.1 at 1e16) under the correct
    deviance: Poisson ended 0.5% off in lambda and Gamma at
    ``max_reml_iter``.  Mutation: drop ``_centred_init`` in ``reml.discrete``
    (item 1's trials kept centred).
    """
    rng = np.random.default_rng(10)
    n = 2400
    g = np.tile(np.arange(40), n // 40)
    z = 2.0 * rng.integers(-4, 5, n)
    s = rng.uniform(size=n)
    eta = 0.5 + 0.1 * z + 0.3 * np.sin(g) + 0.4 * np.sin(2 * np.pi * s)
    if family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        y = rng.gamma(3.0, np.exp(eta) / 3.0)

    def fit(shift: float) -> SuperGLM:
        model = SuperGLM(
            family=family,
            features={"x": Numeric(), "g": RandomEffect(), "s": Spline(kind="ps", k=8)},
            selection_penalty=0,
            discrete=True,
        )
        return _fit_reml(model, pd.DataFrame({"x": shift + z, "g": g, "s": s}), y)

    _assert_same_reml(fit(0.0), fit(1e16))


def test_a_centred_warm_state_without_its_beta_is_not_a_start():
    """``_centred_init`` is ``alpha + (X - 1 c') beta_init``: without ``beta_init`` it is ignored.

    The fit then starts from ``intercept_init`` with zero slopes, as without
    the centred state.  One Poisson step depends on its start, so the two
    fits agree bit for bit only from the same start.  Mutation: drop
    ``beta_init is not None`` from ``irls_direct``'s warm-start guard (the fit
    then starts from the warm ``alpha`` instead).
    """
    from superglm.distributions import Poisson
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import LogLink
    from superglm.solvers.irls_direct import fit_irls_direct
    from superglm.types import GroupSlice

    rng = np.random.default_rng(3)
    n = 200
    X = 5.0 + rng.normal(size=(n, 1))
    y = rng.poisson(np.exp(0.2 + 0.3 * (X[:, 0] - 5.0))).astype(float)
    common = {
        "X": DesignMatrix([DenseGroupMatrix(X)], n=n, p=1),
        "y": y,
        "weights": np.ones(n),
        "family": Poisson(),
        "link": LogLink(),
        "groups": [GroupSlice(name="x", start=0, end=1)],
        "lambda2": 1.0,
        "S_override": np.array([[0.1]]),
        "intercept_init": 0.1,
        "max_iter": 1,
        "weight_semantics": "frequency",
    }
    plain, _ = fit_irls_direct(**common)
    warm, _ = fit_irls_direct(**common, _centred_init=(2.0, np.array([5.0])))
    np.testing.assert_array_equal(warm.beta, plain.beta)
    assert warm.intercept == plain.intercept


# --------------------------------------- 4b. the exact REML weight derivative
@pytest.mark.parametrize("family", ["poisson", "gamma", "binomial"])
def test_exact_reml_weight_derivative_reads_the_centred_direction(family):
    """The W(rho) correction's ``deta = (X - 1 mean_x') dbeta`` is formed about the centre.

    Exact REML differentiates the working weights through ``deta/drho``.  Formed
    as ``X dbeta - mean_x' dbeta`` it cancels ``c' dbeta`` at a column's offset,
    ``u |c' dbeta|`` in every row: at 1e16 Poisson, Gamma (observed geometry) and
    binomial ended ``line_search_failed`` with lambda about 1e-4 off.  The dense
    column is now applied about the state's centre and its mean offset formed on
    centred rows (``reml.w_derivatives``).  Mutation: the raw ``centered_matvec``.
    """
    rng = np.random.default_rng(10)
    n = 2400
    g = np.tile(np.arange(40), n // 40)
    z = 2.0 * rng.integers(-4, 5, n)
    s = rng.uniform(size=n)
    eta = 0.5 + 0.1 * z + 0.3 * np.sin(g) + 0.4 * np.sin(2 * np.pi * s)
    if family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif family == "gamma":
        y = rng.gamma(3.0, np.exp(eta) / 3.0)
    else:
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(1.0 - eta))).astype(float)

    def fit(shift: float) -> SuperGLM:
        model = SuperGLM(
            family=family,
            features={"x": Numeric(), "g": RandomEffect(), "s": Spline(kind="ps", k=8)},
            selection_penalty=0,
            **({"link": "log"} if family == "gamma" else {}),
        )
        return _fit_reml(model, pd.DataFrame({"x": shift + z, "g": g, "s": s}), y)

    _assert_same_reml(fit(0.0), fit(1e16))


def _dense_binomial_state(columns: int) -> dict:
    """Sol's reproduction: even-integer columns, binomial/logit, an identity penalty each at 4."""
    from superglm.distributions import Binomial
    from superglm.links import LogitLink
    from superglm.solvers.irls_direct import fit_irls_direct
    from superglm.types import GroupSlice, PenaltyComponent

    rng = np.random.default_rng(10)
    n = 2400
    X = 2.0 * rng.integers(-4, 5, size=(n, columns)).astype(float)
    signal = 0.5 + X @ np.full(columns, 0.1)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-signal))).astype(float)
    dm = DesignMatrix([DenseGroupMatrix(X[:, j : j + 1]) for j in range(columns)], n=n, p=columns)
    groups = [GroupSlice(name=f"x{j}", start=j, end=j + 1) for j in range(columns)]
    penalties = [
        PenaltyComponent(
            name=g.name,
            group_name=g.name,
            group_index=j,
            group_sl=slice(j, j + 1),
            omega_raw=np.ones((1, 1)),
            omega_ssp=np.ones((1, 1)),
            rank=1.0,
            log_det_omega_plus=0.0,
            eigvals_omega=np.ones(1),
        )
        for j, g in enumerate(groups)
    ]
    lambdas = {g.name: 4.0 for g in groups}
    weights = np.ones(n)
    result, inverse, _ = fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Binomial(),
        link=LogitLink(),
        groups=groups,
        lambda2=lambdas,
        offset=np.zeros(n),
        return_xtwx=True,
        reml_penalties=penalties,
        weight_semantics="frequency",
    )
    assert result.converged
    return {
        "dm": dm,
        "link": LogitLink(),
        "groups": groups,
        "pirls_result": result,
        "XtWX_S_inv": inverse,
        "lambdas": lambdas,
        "sample_weight": weights,
        "offset_arr": np.zeros(n),
        "distribution": Binomial(),
        "reml_penalties": penalties,
    }


def _translated_state(state: dict, shift: float) -> dict:
    """The same converged state read with every column translated by ``shift``.

    The slope inverse is translation invariant; the state's centre is the
    translated design's own (``prior_weighted_centre``), its centred
    intercept carried to it exactly, and the stored working means translated
    as a fit at that offset rounds them.
    """
    import dataclasses
    import math

    from superglm.solvers.mode_score import prior_weighted_centre

    dm = state["dm"]
    values = dm.toarray()
    translated = DesignMatrix(
        [DenseGroupMatrix(values[:, j : j + 1] + shift) for j in range(dm.p)], n=dm.n, p=dm.p
    )
    result = state["pirls_result"]
    centre = prior_weighted_centre(translated, state["sample_weight"])
    beta = np.asarray(result.beta, dtype=np.float64)
    alpha = float(result.centred_intercept) + math.fsum(
        ((centre - shift) - result.state_center) * beta
    )
    rank_info = result.rank_info
    if rank_info is not None:
        rank_info = dataclasses.replace(rank_info, mean_x=rank_info.mean_x + shift)
    summary = result.reml_geometry
    if summary is not None:
        summary = dataclasses.replace(summary, mean_x=summary.mean_x + shift)
    moved = dataclasses.replace(
        result,
        state_center=centre,
        centred_intercept=alpha,
        intercept=alpha - math.fsum(centre * beta),
        rank_info=rank_info,
        reml_geometry=summary,
    )
    return {**state, "dm": translated, "pirls_result": moved}


def _correction_magnitude(state: dict, *, profiled: bool = True) -> NDArray:
    """``M_j``: the sum of the magnitudes both W-correction routes add, per penalty.

    Both routes sum, over rows, ``0.5 dW_i (x~_i' dbeta_j) (x~_i' H^-1 x~_i +
    1 / sum W)``; their rounding is at most ``gamma_k`` of ``M_j``, the same sum
    of magnitudes, ``x~`` the centred rows and ``dbeta_j = -H^-1 S_j beta``.
    Without a profiled intercept the rows are raw and the ``1 / sum W`` term
    is absent.
    """
    dm, result = state["dm"], state["pirls_result"]
    X = dm.toarray()
    eta = linear_predictor(dm, result, None)
    mu = 1.0 / (1.0 + np.exp(-eta))
    weights = mu * (1.0 - mu)
    centred = X - (weights @ X) / weights.sum() if profiled else X
    dweights = np.abs(weights * (1.0 - 2.0 * mu))
    inverse = np.asarray(state["XtWX_S_inv"], dtype=np.float64)
    leverage = np.einsum("ij,jk,ik->i", np.abs(centred), np.abs(inverse), np.abs(centred))
    intercept_term = 1.0 / weights.sum() if profiled else 0.0
    magnitudes = []
    for j in range(dm.p):
        penalty_beta = np.zeros(dm.p)
        penalty_beta[j] = 4.0 * result.beta[j]
        dbeta = np.abs(inverse @ penalty_beta)
        magnitudes.append(
            0.5 * float(np.sum(dweights * (np.abs(centred) @ dbeta) * (leverage + intercept_term)))
        )
    return np.array(magnitudes)


@pytest.mark.parametrize("columns", [1, 4])
@pytest.mark.parametrize("route", ["ordinary", "leverage"])
def test_the_weight_correction_centres_its_grams_and_leverage_rows(columns, route):
    """The W(rho) correction's signed Grams and leverage rows read the exact centre pair.

    Sol's reproduction: the same converged binomial/logit state, its columns
    translated exactly by 1e16.  The derivative Grams and
    ``_leverage_gradient_rhs`` centred rows about the rounded ``mean_x``: the
    correction moved 9.2% (ordinary route) and 17.1% (leverage route, the
    default for binomial/logit exact REML with four or more penalties).
    Centred as ``(x - c) - d`` the two readings agree to the rounding of the
    sums, ``4 gamma_{n+2p+5} M_j`` (``_correction_magnitude``).  Mutation:
    ``mean_lo`` dropped from ``centered_signed_grams`` or from
    ``_leverage_gradient_rhs``.
    """
    from superglm.reml.w_derivatives import reml_w_correction

    state = _dense_binomial_state(columns)
    moved = _translated_state(state, 1e16)
    leverage = route == "leverage"
    base = reml_w_correction(**state, gradient_only=leverage)
    shifted = reml_w_correction(**moved, gradient_only=leverage)
    assert base is not None and shifted is not None
    n, p = state["dm"].n, state["dm"].p
    bound = 4.0 * _gamma(n + 2 * p + 5) * _correction_magnitude(state)
    np.testing.assert_array_less(np.abs(shifted[0] - base[0]), bound)


def test_exact_reml_leverage_route_is_translation_invariant():
    """Binomial/logit exact REML with four penalties (the leverage route) at a 1e16 offset.

    A numeric column on the even-integer grid beside four P-splines, no
    random effect, so the gram backend's dense factor and ``gradient_only``
    W-correction run.  Mutation: as the previous test, or the gram rows
    centred about the rounded ``mean_x``.
    """
    rng = np.random.default_rng(11)
    n = 2400
    z = 2.0 * rng.integers(-4, 5, n)
    s = rng.uniform(size=(n, 4))
    eta = 0.2 + 0.1 * z + 0.6 * np.sin(2 * np.pi * s).sum(axis=1)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)

    def fit(shift: float) -> SuperGLM:
        features = {"x": Numeric(), **{f"s{j}": Spline(kind="ps", k=8) for j in range(4)}}
        frame = pd.DataFrame({"x": shift + z, **{f"s{j}": s[:, j] for j in range(4)}})
        model = SuperGLM(family="binomial", features=features, selection_penalty=0)
        return _fit_reml(model, frame, y)

    base, shifted = fit(0.0), fit(1e16)
    assert shifted._reml_profile["direct_backend"] == "gram"
    _assert_same_reml(base, shifted)


def test_without_a_profiled_intercept_the_weight_correction_centres_no_column():
    """No profiled intercept (a bare inverse, no rank or geometry summary): no column is centred.

    ``sum_w`` is then None and the operator is ``X`` itself, whatever a
    column's type.  381bdd39 still centred the dense columns there about the
    state's centre, mixing a profiled and an unprofiled intercept.  The same
    column stored dense and sparse must give the same correction, to the
    rounding of the sums over raw rows, ``4 gamma_{n+2p+5} M_j``.  Mutation:
    the dense pair formed whatever ``sum_w``, divided by ``np.sum`` of the
    weights.  This guards the review edge; master never centred here.
    """
    import dataclasses

    from scipy import sparse

    from superglm.group_matrix import SparseGroupMatrix
    from superglm.reml.w_derivatives import reml_w_correction

    state = _translated_state(_dense_binomial_state(1), 8.0)
    bare = {
        **state,
        "pirls_result": dataclasses.replace(
            state["pirls_result"], rank_info=None, reml_geometry=None
        ),
        "XtWX_S_inv": np.asarray(state["XtWX_S_inv"], dtype=np.float64),
    }
    dm = state["dm"]
    values = dm.toarray()
    stored_sparse = DesignMatrix(
        [SparseGroupMatrix(sparse.csr_matrix(values[:, j : j + 1])) for j in range(dm.p)],
        n=dm.n,
        p=dm.p,
    )
    dense_reading = reml_w_correction(**bare)
    sparse_reading = reml_w_correction(**{**bare, "dm": stored_sparse})
    assert dense_reading is not None and sparse_reading is not None
    bound = 4.0 * _gamma(dm.n + 2 * dm.p + 5) * _correction_magnitude(state, profiled=False)
    np.testing.assert_array_less(np.abs(dense_reading[0] - sparse_reading[0]), bound)


# ------------------------------------------------------ 5. drop-term holdout
def test_holdout_drop_term_reads_the_public_predictor():
    """``term_drop_diagnostics(mode="holdout")`` scores the fit's centred predictor (item 5).

    It formed ``eta`` from the raw intercept plus ``X beta``, which cancel at a
    1e16 offset (errors of 0.15 per row in this fixture), so the deviances it
    differences were not the model's.  A random effect's delta is translation
    invariant: its two deviances move by at most ``2 n d (max|r| + max|b| + d)``
    for predictors ``d`` apart, plus the rounding of each sum.  Mutation:
    ``np.full(n, model.result.intercept)`` plus raw term scores.
    """
    rng, g, z, _ = _offset_frame(0.0)
    y = 3.0 + 0.2 * z + 0.05 * np.sin(g) + 0.01 * rng.normal(size=len(z))
    fits, deltas, predictors = {}, {}, {}
    for shift in (0.0, 1e16):
        frame = pd.DataFrame({"x": shift + z, "g": g})
        model = SuperGLM(
            family="gaussian",
            features={"x": Numeric(), "g": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0))},
            selection_penalty=0,
        )
        fits[shift] = _fit_reml(model, frame, y)
        table = model.term_drop_diagnostics(frame, y, mode="holdout", X_val=frame, y_val=y)
        deltas[shift] = float(table.set_index("feature").loc["g", "delta_deviance"])
        predictors[shift] = np.asarray(model.predict(frame), dtype=np.float64)
    base = fits[0.0]
    g_group = next(group for group in base._groups if group.name == "g")
    effect = float(np.max(np.abs(base.result.beta[g_group.sl])))
    residual = float(np.max(np.abs(y - predictors[0.0])))
    moved = float(np.max(np.abs(predictors[1e16] - predictors[0.0])))
    full = float(np.sum((y - predictors[0.0]) ** 2))
    dropped = full + deltas[0.0]
    bound = 2.0 * len(y) * moved * (residual + effect + moved) + 4.0 * _gamma(len(y)) * (
        full + dropped
    )
    assert abs(deltas[1e16] - deltas[0.0]) <= bound


# ------------------------------------------- 6. the compensated intercept kept
@pytest.mark.parametrize("base", [3e-8, 1e100])
def test_a_folded_compensated_intercept_still_predicts_the_fit(base):
    """A Gaussian identity fit keeps its pair ``(alpha, alpha_lo)``, by type (item 6).

    ``Polynomial(degree=1)`` on ``x`` in {0, 1} is the column ``+-1`` about
    its exact zero centre, so canonicalization folds it and the public
    result dropped the centred state with its remainder ``alpha_lo``: the
    levels ``y = base`` and ``nextafter(base)`` predicted ``alpha +- beta``,
    which ties to one even float at the midpoint ``alpha*``, while the fit
    published the zero deviance of ``alpha + (+-beta + alpha_lo)``.  Every
    quantity is an exact multiple of half an ulp, so the predictor is ``y``
    itself.  Mutation: return ``(None, None, None)`` for a zero public centre.
    """
    x = np.tile([0.0, 1.0], 50)
    X = pd.DataFrame({"x": x})
    y = np.where(x == 0.0, base, np.nextafter(base, np.inf))
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, features={"x": Polynomial(degree=1)}
    ).fit(X, y)

    solver = model._solver_pirls_result()
    assert solver.centred_intercept_lo is not None
    assert solver.deviance == 0.0
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.metrics(X, y).deviance == 0.0


# ------------------------------------- 7. the gram and QR paths at 1e16
def _even_grid_frame(shift: float):
    """An even-integer column (exact at 1e16) beside a four-level categorical."""
    rng = np.random.default_rng(1)
    n = 2400
    z = 2.0 * rng.integers(-4, 5, n)
    s = rng.uniform(size=n)
    c = rng.integers(0, 4, n)
    eta = 0.3 + 0.1 * z + 0.4 * np.sin(2 * np.pi * s) + 0.1 * c
    frame = pd.DataFrame(
        {"x": shift + z, "s": s, "c": np.array([f"c{k}" for k in c], dtype=object)}
    )
    return rng, eta, frame


@pytest.mark.parametrize("direct_solve", ["gram", "qr"])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_gram_and_qr_fits_are_translation_invariant_at_1e16(family, direct_solve):
    """The gram and QR systems centre a dense column about an exact pair (items 1 and 7).

    Both formed the profiled Gram (gram) or the QR's data block (QR) on rows
    ``x - mean_x``, ``mean_x`` the one-float weighted mean.  It rounds at ``u
    s`` at an offset ``s``, which adds ``sum W d d'`` to the Gram (``d`` that
    rounding): at 1e16 a Gaussian fit's eta moved 1.1e-3 while reporting
    converged and a Poisson fit ended ``step_rejected``.  The rows are now
    ``(x - x_ref) - lo`` (``centered_system.weighted_mean_pair``): ``x - x_ref``
    is the same exact difference at every offset and ``lo`` is formed on it,
    so the two fits' centred systems agree to rounding and the ``(u s /
    sigma)^2`` term is gone.  Their predictors then agree to the forward error
    of the solves, ``gamma_n kappa(H) max|eta|``, ``kappa`` the centred
    Hessian's condition; the intercept reads ``mean_x - c`` on centred rows
    too (``centre_offset_mean``).  Mutation: ``mean_lo`` dropped in
    ``build_centered_system`` (gram) or ``_centred_rows`` (QR).
    """
    fits = {}
    for shift in (0.0, 1e16):
        rng, eta, frame = _even_grid_frame(shift)
        if family == "poisson":
            y = rng.poisson(np.exp(eta)).astype(float)
        else:
            y = 3.0 + eta + 0.3 * rng.normal(size=len(eta))
        model = SuperGLM(
            family=family,
            features={"x": Numeric(), "c": Categorical()},
            selection_penalty=0.0,
            direct_solve=direct_solve,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits[shift] = model.fit(frame.drop(columns="s"), y)
    base, shifted = fits[0.0], fits[1e16]
    solver, shifted_solver = base._solver_pirls_result(), shifted._solver_pirls_result()
    assert solver.converged and shifted_solver.converged
    assert shifted_solver.termination_reason == solver.termination_reason
    eta_base = linear_predictor(base._dm, solver, None)
    eta_shifted = linear_predictor(shifted._dm, shifted_solver, None)
    n = len(eta_base)
    weights = np.exp(eta_base) if family == "poisson" else np.ones(n)
    design = np.column_stack([np.ones(n), np.asarray(base._dm.toarray(), dtype=np.float64)])
    kappa = float(np.linalg.cond(design.T @ (weights[:, None] * design)))
    bound = _gamma(n) * kappa * float(np.max(np.abs(eta_base)))
    np.testing.assert_array_less(np.abs(eta_shifted - eta_base), bound)


def _gram_qr_reml(family: str, direct_solve: str, shift: float) -> SuperGLM:
    """REML on ``Numeric(x) + Spline(s)`` from ``_even_grid_frame``, no random effect."""
    rng, eta, frame = _even_grid_frame(shift)
    if family == "gamma":
        y = rng.gamma(3.0, np.exp(eta) / 3.0)
    elif family == "gaussian":
        y = np.exp(eta) + 0.5 * rng.normal(size=len(eta))
    else:
        y = (rng.uniform(size=len(eta)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    model = SuperGLM(
        family=family,
        features={"x": Numeric(), "s": Spline(kind="ps", k=8)},
        selection_penalty=0,
        direct_solve=direct_solve,
        **({"link": "log"} if family in ("gamma", "gaussian") else {}),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit_reml(frame.drop(columns="c"), y)


@pytest.mark.parametrize("direct_solve", ["gram", "qr"])
@pytest.mark.parametrize("family", ["binomial", "gamma", "gaussian"])
def test_reml_on_the_gram_and_qr_paths_at_1e16(family, direct_solve):
    """REML with a spline and a numeric column at 1e16, no random effect.

    On the default (gram) route binomial ended ``line_search_failed``.
    Gamma/log has constant Fisher weights, so PIRLS reuses the data Gram and
    refreshes only the right-hand side, whose dense columns are read on
    centred rows.  Gaussian/log has signed observed weights, so its REML
    Hessian comes from the observed geometry's dense branch, which centres
    rows about the pair.  Mutation: as the previous test, or the one-float
    mean back in the right-hand-side refresh or in the observed geometry's
    signed Gram.
    """
    _assert_same_reml(
        _gram_qr_reml(family, direct_solve, 0.0), _gram_qr_reml(family, direct_solve, 1e16)
    )


def test_the_second_order_weight_correction_reads_the_centred_transpose(monkeypatch):
    """``w_correction_order=2`` differentiates the weighted mean through ``X_c' a``.

    That transpose was formed ``X' a - mean_x sum a``, which cancels ``c sum a``
    at a column's offset (at 1e16 ``u c`` is 1, the column's spread 8).  It
    reads the dense column on centred rows about the pair the direction
    reads: on Sol's state translated by 1e16, every transpose the correction
    forms equals ``sum ((x_i - c) - d) a_i`` on the exact rows to its
    rounding, ``gamma_{n+2} sum |x~_i| |a_i|``.  A REML fit cannot show this:
    the outer Hessian shapes only the steps, and the fit reaches the same
    optimum.  Mutation: the raw ``centered_rmatvec``.
    """
    from fractions import Fraction

    import superglm.reml.w_derivatives as w_derivatives

    moved = _translated_state(_dense_binomial_state(1), 1e16)
    directions, transposes = [], []
    real_matvec = w_derivatives.dense_centred_matvec
    real_rmatvec = w_derivatives.dense_centred_rmatvec

    def matvec(dm, values, center, center_lo=None):
        directions.append((np.copy(center), np.copy(center_lo)))
        return real_matvec(dm, values, center, center_lo)

    def rmatvec(dm, rows, center, center_lo=None):
        result = real_rmatvec(dm, rows, center, center_lo)
        transposes.append((np.copy(rows), np.copy(center), np.copy(center_lo), result))
        return result

    monkeypatch.setattr(w_derivatives, "dense_centred_matvec", matvec)
    monkeypatch.setattr(w_derivatives, "dense_centred_rmatvec", rmatvec)
    correction = w_derivatives.reml_w_correction(**moved, w_correction_order=2)
    assert correction is not None and len(correction) == 3
    assert directions and transposes
    x = moved["dm"].toarray()[:, 0]
    for rows, center, center_lo, result in transposes:
        assert any(
            np.array_equal(center, c) and np.array_equal(center_lo, lo) for c, lo in directions
        )
        exact = sum(
            (Fraction(xi) - Fraction(center[0]) - Fraction(center_lo[0])) * Fraction(ai)
            for xi, ai in zip(x.tolist(), rows.tolist(), strict=True)
        )
        magnitude = float(np.sum(np.abs((x - center[0]) - center_lo[0]) * np.abs(rows)))
        assert abs(Fraction(float(result[0])) - exact) <= Fraction(_gamma(len(x) + 2) * magnitude)


# ----------------------------------------------- 8. the null model's mean
@pytest.mark.parametrize("base", [3e-8, 1e100])
def test_the_null_deviance_reads_a_correctly_rounded_mean(base):
    """The null model's weighted mean is refined once with an exact residual (item 8).

    On two levels of adjacent floats the mean is their midpoint, which no
    float holds; ``np.average`` landed 1.5 ulp from it, a null deviance of
    ``250 ulp^2`` against the ``50 ulp^2`` of either neighbour.  The refined
    mean (``mode_score.compensated_weighted_mean``) forms every quantity
    exactly here and rounds the midpoint to an even neighbour.  Mutation:
    ``np.average`` in ``fit_ops._compute_null_mu``.
    """
    x = np.tile([0.0, 1.0], 50)
    X = pd.DataFrame({"x": x})
    adjacent = np.nextafter(base, np.inf)
    y = np.where(x == 0.0, base, adjacent)
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": Numeric()}).fit(X, y)
    assert model.metrics(X, y).null_deviance == 50.0 * (adjacent - base) ** 2


def _mean_bound(values, weights, exact):
    """The stated bound of ``compensated_weighted_mean``, in exact arithmetic."""
    from fractions import Fraction

    u = Fraction(_U)
    largest_product = max(
        abs(Fraction(v) * Fraction(w)) for v, w in zip(values, weights, strict=True)
    )
    largest_weight = max(Fraction(w) for w in weights)
    total = sum(Fraction(w) for w in weights)
    scaling = Fraction(2) ** -1074 * (
        1 + 9 * len(values) * (largest_product + abs(exact) * largest_weight) / total
    )
    return (u + 10 * u * u) * abs(exact) + scaling


def _exact_mean(values, weights):
    from fractions import Fraction

    return sum(Fraction(w) * Fraction(v) for v, w in zip(values, weights, strict=True)) / sum(
        Fraction(w) for w in weights
    )


@pytest.mark.parametrize(
    "values, weights",
    [
        ([0.0, 1.0, 1.0], [5e-324] * 3),
        ([1e300, -1e300, 1.0], [1.0] * 3),
        ([0.0, 1e300], [1e300, 1e-300]),
    ],
    ids=["subnormal_weights", "wide_range", "mixed_exponents"],
)
def test_the_compensated_mean_keeps_every_product_and_error_term(values, weights):
    """``compensated_weighted_mean`` on subnormal weights, values of +-1e300 and mixed exponents.

    Every product is split exactly on the operands' mantissas with its
    exponent kept apart, and each sum is rounded once, so the mean meets its
    stated bound (``_mean_bound``); here every piece stays in range and the
    mean is within ``2 u |m*|`` of the exact rational mean as well.  The first
    version returned 0.333 for 2/3 (subnormal products), 0.556 for 1/3 (each
    error term absorbed into its head), and its fix, scaling the weights by
    the largest one's power of two, erased ``1e-300`` beside ``1e300`` and
    returned 0 for ``1e-300`` (Sol; ``np.average`` gets it).  Mutations: the
    weights scaled by the largest one with raw products, or ``W (h + e)``
    summed per row.
    """
    from fractions import Fraction

    from superglm.solvers.mode_score import compensated_weighted_mean

    exact = _exact_mean(values, weights)
    mean = compensated_weighted_mean(np.array(values), np.array(weights))
    assert abs(Fraction(mean) - exact) <= _mean_bound(values, weights, exact)
    assert abs(Fraction(mean) - exact) <= 2 * Fraction(_U) * abs(exact)


def test_the_compensated_mean_meets_its_bound_across_the_exponent_range():
    """Signed values and weights drawn across ``1e-300 .. 1e300``, against ``fractions.Fraction``.

    Products under- and overflow the binary64 range in pairs here; the bound
    (``_mean_bound``) holds for every draw, and the null model's Gaussian
    predictor on Sol's mixed-exponent rows is their mean, not zero.
    Mutation: as the previous test.
    """
    from fractions import Fraction

    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.model.fit_ops import _compute_null_mu
    from superglm.solvers.mode_score import compensated_weighted_mean

    rng = np.random.default_rng(430)
    for _ in range(100):
        n = int(rng.integers(2, 40))
        values = (rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(-300, 300, n)).tolist()
        weights = (10.0 ** rng.uniform(-300, 300, n)).tolist()
        exact = _exact_mean(values, weights)
        mean = compensated_weighted_mean(np.array(values), np.array(weights))
        assert abs(Fraction(mean) - exact) <= _mean_bound(values, weights, exact)

    null = _compute_null_mu(
        np.array([0.0, 1e300]),
        np.array([1e300, 1e-300]),
        None,
        Gaussian(),
        IdentityLink(),
        weight_semantics="prior",
    )
    np.testing.assert_array_equal(null, np.full(2, 1e-300))


# ------------------------------------------- 10. the anchor of the exact pair
def _pair_gram_bound(x: NDArray, w: NDArray) -> float:
    """How far a Gram centred about the corrected two-pass pair may sit from the exact ``G*``.

    With ``m*`` the exact mean and ``x_ref`` the first row that carries
    weight, pass one puts the anchor within ``L = u |m*| + gamma_{n+2} sum
    |w| |x - x_ref| / |sum w|`` of ``m*``.  With ``D_i = |x_i - m*|`` and ``A
    = sum |w| (D + L) / |sum w|``, the remainder ``lo`` errs by at most
    ``gamma_{n+2} A`` and each centred row ``(x - hi) - lo`` by ``E_i =
    gamma_{n+4} (A + 2 D_i + 4 L)``; the Gram of those rows then errs by
    ``sum |w| (2 D_i E_i + E_i^2) + gamma_{n+2} sum |w| (D_i + E_i)^2``.
    """
    from fractions import Fraction

    total = sum(Fraction(v) for v in w)
    mean = sum(Fraction(a) * Fraction(b) for a, b in zip(x, w, strict=True)) / total
    deviation = np.array([float(abs(Fraction(a) - mean)) for a in x])
    magnitude = np.abs(w)
    seed = x[np.flatnonzero(w != 0.0)[0]]
    offset = _U * abs(float(mean)) + _gamma(len(x) + 2) * float(
        np.sum(magnitude * np.abs(x - seed)) / abs(float(total))
    )
    spread = float(np.sum(magnitude * (deviation + offset)) / abs(float(total)))
    n = len(x)
    row = _gamma(n + 4) * (spread + 2.0 * deviation + 4.0 * offset)
    return float(
        np.sum(magnitude * (2.0 * deviation * row + row**2))
        + _gamma(n + 2) * np.sum(magnitude * (deviation + row) ** 2)
    )


@pytest.mark.parametrize(
    "route, w",
    [
        ("signed", [0.0, -0.1, 1.0, 1.0]),
        ("signed", [1e-30, -0.1, 1.0, 1.0]),
        ("gram", [1e-30, 0.5, 1.0, 1.0]),
    ],
    ids=["signed_zero_weight", "signed_negligible_weight", "gram_negligible_weight"],
)
def test_the_exact_pair_does_not_anchor_on_a_far_row(route, w):
    """The pair is seeded by a row that carries weight, then refined about the rounded mean.

    The signed observed route anchored on row 0 whatever its weight: Sol's
    ``x = [0, 1e16 - 2, 1e16, 1e16 + 2]`` at ``w = [0, -0.1, 1, 1]`` formed
    the remainder at the offset's scale and read a centred Gram of 3.6 for
    1.0526315789.  A seed row of negligible weight far from the rest did the
    same on either route.  The anchor is now the rounded mean
    (``corrected_two_pass_pair``), within pass one's rounding of it, so the
    same rows in any order give the exact Gram to ``_pair_gram_bound``.
    Mutations: the anchor back on row 0, or pass two dropped.
    """
    from fractions import Fraction

    from superglm._group_matrix._group_matrix_centered import centered_gram_rhs
    from superglm.reml.observed_geometry import _stable_signed_mean_pair
    from superglm.solvers.centered_system import weighted_mean_pair

    x = np.array([0.0, 1e16 - 2.0, 1e16, 1e16 + 2.0])
    w = np.array(w)
    total = sum(Fraction(v) for v in w)
    mean = sum(Fraction(a) * Fraction(b) for a, b in zip(x, w, strict=True)) / total
    exact = sum(Fraction(b) * (Fraction(a) - mean) ** 2 for a, b in zip(x, w, strict=True))
    pair = _stable_signed_mean_pair if route == "signed" else weighted_mean_pair
    for order in ([0, 1, 2, 3], [1, 2, 3, 0], [3, 0, 2, 1]):
        rows, weights = x[order], w[order]
        dm = DesignMatrix([DenseGroupMatrix(rows[:, None])], n=4, p=1)
        _, hi, lo = pair(dm, weights, float(np.sum(weights)))
        gram, _ = centered_gram_rhs(dm=dm, W=weights, mean_x=hi, z_centered=np.zeros(4), mean_lo=lo)
        assert abs(Fraction(float(gram[0, 0])) - exact) <= Fraction(_pair_gram_bound(rows, weights))


def test_gaussian_log_reml_ignores_where_a_zero_weight_row_sits():
    """Public Gaussian/log REML (signed observed weights) with a zero-weight row far from the data.

    The row's place in the frame decided the observed geometry's anchor, and
    moving it from first to last moved the converged objective by 0.29
    against a stopping resolution near 2e-6 (Sol).  Both orders now stop
    with the same termination within that resolution.  Mutation: the anchor
    back on row 0.
    """
    rng, eta, frame = _even_grid_frame(1e16)
    y = np.exp(eta) + 0.5 * rng.normal(size=len(eta))
    weights = np.ones(len(y))
    far = pd.DataFrame({"x": [0.0], "s": [0.5], "c": ["c0"]})

    def fit(first: bool) -> SuperGLM:
        rows = pd.concat([far, frame] if first else [frame, far], ignore_index=True)
        response = np.concatenate([[1.0], y] if first else [y, [1.0]])
        prior = np.concatenate([[0.0], weights] if first else [weights, [0.0]])
        model = SuperGLM(
            family="gaussian",
            link="log",
            features={"x": Numeric(), "s": Spline(kind="ps", k=8)},
            selection_penalty=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit_reml(rows.drop(columns="c"), response, sample_weight=prior)

    _assert_same_reml(fit(False), fit(True))


# ---------------------------------------------- 11. the metrics certificates
def test_metrics_certify_the_rank_about_the_exact_pair():
    """``metrics()`` on new rows certifies rank about the exact pair, like the fit.

    Sol's two Numeric columns at 1e16, one a translate of the other: the fit
    reads slope rank 1 and NaN standard errors for both.  The streamed
    certificates of ``metrics`` (data rank and profiled covariance) centred
    their rows about the one-float ``X'W1 / sum W`` and read rank 2, so
    ``metrics(X.copy(), y)`` published finite standard errors.  Mutation: the
    one-float centre in either certificate.
    """
    from superglm.inference import metrics as metrics_module
    from superglm.inference._metrics_design import weighted_moments

    z = np.tile([-2.0, 0.0, 2.0, 4.0], 100)
    X = pd.DataFrame({"a": 1e16 + z, "b": 1e16 + z + 2.0})
    y = 0.3 + 0.1 * z + np.tile([0.01, -0.02, 0.03, -0.02], 100)
    model = SuperGLM(
        family="gaussian", features={"a": Numeric(), "b": Numeric()}, selection_penalty=0
    ).fit(X, y)
    for name in ("a", "b"):
        assert np.all(np.isnan(model.metrics(X, y).coefficient_se[name]))
        assert np.all(np.isnan(model.metrics(X.copy(), y).coefficient_se[name]))

    s = np.tile([0.1, 0.7, 0.3, 0.9, 0.5], 80)
    design = np.column_stack((1e16 + z, 1e16 + z + 2.0, s))
    W = np.ones(len(z))
    _, xtw1, data_gram = weighted_moments(design, W)
    data_rank = metrics_module._certified_data_rank(design, W, data_gram, xtw1)
    profile_rank = metrics_module._certified_profile_rank(
        design, W, data_gram, xtw1, np.diag([0.0, 0.0, 1.0]), data_rank
    )
    assert data_rank.rank == 2
    assert profile_rank.rank == 2


@pytest.mark.parametrize("fit", ["gram", "gram_reml", "gamma_reml", "proximal"])
def test_aliased_columns_at_1e16_certify_the_rank_of_the_fit_at_zero(fit):
    """Two Numeric columns at 1e16, ``b = 2 a - 1e16``, beside a P-spline.

    ``a = s + z`` and ``b = s + 2 z`` alias with the intercept, so the centred
    Gram is singular and the rank comes from a factor certificate:
    ``certify_centered_factor`` and the terminal data rank on the direct
    path, the observed geometry's certificate (Gamma/log REML), the
    covariance factors (``state_ops``), the proximal fit's post-fit rank
    (``pirls``) and the metrics certificates.  Each centres its rows about
    the exact pair, as the Gram does; about the one-float means ``2 m_a -
    m_b`` rounds away from the offset and the pair stops aliasing (the proximal certificate
    read rank 9 against 8 at no offset, Codex).  The certified rank and the
    NaN standard errors of the pair match the fit at no offset.  Mutation:
    the one-float centre at a certification site.
    """
    rng = np.random.default_rng(5)
    n = 800
    z = np.tile([-2.0, 0.0, 2.0, 4.0], n // 4)
    s = rng.uniform(size=n)
    eta = 0.3 + 0.1 * z + 0.4 * np.sin(2 * np.pi * s)
    if fit == "proximal":
        y = 3.0 + eta + 0.3 * rng.normal(size=n)
    elif fit == "gamma_reml":
        y = rng.gamma(3.0, np.exp(eta) / 3.0)
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    family = {"proximal": "gaussian", "gamma_reml": "gamma"}.get(fit, "poisson")
    ranks = []
    for shift in (0.0, 1e16):
        frame = pd.DataFrame({"a": shift + z, "b": shift + 2.0 * z, "s": s})
        model = SuperGLM(
            family=family,
            features={"a": Numeric(), "b": Numeric(), "s": Spline(kind="ps", k=8)},
            selection_penalty=0.01 if fit == "proximal" else 0,
            **({"link": "log"} if fit == "gamma_reml" else {}),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if fit.endswith("reml"):
                model.fit_reml(frame, y)
            else:
                model.fit(frame, y)
        ranks.append(int(model.result.rank_info.data.rank))
        for rows in (frame, frame.copy()):
            se = model.metrics(rows, y).coefficient_se
            assert np.all(np.isnan(se["a"])) and np.all(np.isnan(se["b"]))
    assert ranks[1] == ranks[0]


# ------------------------------------------- 12. dense columns take the pair
@pytest.mark.parametrize("rung", ["raw_moment", "tabmat"])
def test_a_dense_column_takes_the_exact_pair_at_every_offset(rung):
    """A ``DenseGroupMatrix`` column never takes a raw-moment rung, at any offset.

    Above the raw rungs' size crossover, Sol's even-integer column (n =
    108,000) took the raw-moment rung at offset 0 and the exact pair at
    offset 10, where ``_raw_centering_well_scaled`` rejected it; beside a
    150-level categorical the tabmat rung did the same.  The arithmetic
    changed with the column's location.  Both offsets now centre about the
    exact pair.  Mutation: dense designs admitted to the raw rungs.
    """
    from superglm.group_matrix import CategoricalGroupMatrix
    from superglm.solvers.centered_system import TabmatCenteringState, build_centered_system

    column = 2.0 * np.tile(np.arange(-4, 5), 12000)
    n = len(column)
    codes = np.random.default_rng(3).integers(0, 150, n)
    for shift in (0.0, 10.0):
        groups = [DenseGroupMatrix((column + shift)[:, None])]
        if rung == "tabmat":
            groups.append(CategoricalGroupMatrix(codes, 150))
        dm = DesignMatrix(groups, n=n, p=sum(group.shape[1] for group in groups))
        profile: dict = {}
        system = build_centered_system(
            dm=dm,
            W=np.ones(n),
            z_off=np.zeros(n),
            penalty=np.zeros((dm.p, dm.p)),
            tabmat_split=dm.tabmat_centering_split,
            tabmat_state=TabmatCenteringState(),
            profile=profile,
        )
        assert system.mean_lo is not None
        assert "centered_raw_moment_hits" not in profile
