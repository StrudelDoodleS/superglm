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

import math
import warnings
from fractions import Fraction

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
    PSpline,
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
        # the translated state is read without the untranslated system's pair
        summary = dataclasses.replace(
            summary, mean_x=summary.mean_x + shift, mean_hi=None, mean_lo=None
        )
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


@pytest.mark.parametrize("folded", ["spline", "polynomial"])
def test_the_public_intercept_pair_carries_the_fold_exactly(folded):
    """Canonicalization folds ``(m - c)' beta`` into the compensated pair, error included.

    A materialized term's public columns lose their training means ``m``,
    and the solver centred them about its own ``c``, two roundings of the
    same means: the fold ``(m - c)' beta`` (``_public_centred_state``) is of
    the order of an ulp of ``alpha``, so ``alpha + fold`` rounds away part of
    it, which the TwoSum's error carries into ``alpha_lo``.  The public pair
    must then be the solver's pair plus the fold, to the fold's own rounding
    (a difference and a product per column, then a correctly rounded
    ``fsum``: ``gamma_3`` on the terms' magnitudes, Higham 2002, Lemma 3.1)
    and the one addition that forms the public ``alpha_lo`` (``gamma_1``);
    the TwoSum itself is exact (Knuth, TAOCP vol. 2, 4.2.2, Theorem B).  The
    previous test's fold is zero, so it cannot see this.  Mutations: the
    TwoSum's error dropped, or ``alpha + fold`` added plainly (Claude review
    of #425, Low): both leave the gap at the rounded part of the fold.
    """
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(5.0, 1.0, n)
    s = rng.uniform(0.0, 1.0, n)
    y = 2.0 + 0.3 * x + np.sin(4.0 * s) + 0.1 * rng.normal(size=n)
    term = Spline(n_knots=6) if folded == "spline" else Polynomial(degree=3)
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, features={"x": Numeric(), "s": term}
    ).fit(pd.DataFrame({"x": x, "s": s}), y)

    solver, public = model._solver_pirls_result(), model.result
    alpha, alpha_lo = solver.centred_intercept, solver.centred_intercept_lo
    assert alpha_lo is not None and public.centred_intercept_lo is not None
    centre = np.asarray(solver.state_center, dtype=np.float64)
    beta = np.asarray(solver.beta, dtype=np.float64)
    shifts = np.zeros(centre.size)
    for term_state in model._runtime_canonical_state["terms"].values():
        if term_state["applied_to_public_model"]:
            for group_state in term_state["groups"]:
                lo, hi = group_state["solver_slice"]
                shifts[lo:hi] = group_state["column_means"]
    columns = np.asarray(public.state_center) == 0.0
    terms = [
        (Fraction(float(m)) - Fraction(float(c))) * Fraction(float(b))
        for m, c, b in zip(shifts[columns], centre[columns], beta[columns], strict=True)
    ]
    fold = sum(terms, Fraction(0))
    # the fold is real and alpha + fold rounds it, so only alpha_lo can carry the rest
    assert fold != 0 and Fraction(alpha + float(fold)) != Fraction(alpha) + fold
    gap = Fraction(public.centred_intercept) + Fraction(public.centred_intercept_lo)
    gap -= Fraction(alpha) + Fraction(alpha_lo) + fold
    bound = Fraction(_gamma(3)) * sum(map(abs, terms), Fraction(0))
    bound += Fraction(_gamma(1)) * abs(Fraction(public.centred_intercept_lo))
    assert abs(gap) <= bound, f"gap {float(gap):.3g}, fold {float(fold):.3g}"


@pytest.mark.parametrize("change", [0.0, 0.0123], ids=["no_intercept_change", "intercept_change"])
def test_a_revision_publishes_one_predictor_in_both_coordinates(change):
    """A revision's published pair and its solver predictor read the revised coefficients.

    A ``PSpline``'s columns are sparse, so the solver centres none of them
    (``c = 0``) and the public pair folds the means its public columns lose:
    ``alpha_pub = alpha + m' beta``, ``m`` the unweighted means, which differ
    from the weighted centre under unequal weights.  The revision halves
    ``beta`` and moves the public intercept by ``change``, as an editor edit
    (with and without an intercept change) or a shape repair (its profiled
    shift) does.  The published pair moves by the change alone, bit for bit
    without one, since the moved columns carry no centre, and predicts the
    raw public predictor ``intercept_pub + X_pub beta`` of the same revision to
    the remainder ``alpha_lo``, two roundings of each intercept and each
    evaluation's ``gamma_(p+3)`` (Higham 2002, section 3.1).  The solver
    predictor reads the same rows from ``alpha_pub - m' beta`` about the solver
    columns, to both evaluations and the fold's ``gamma_2``.  Before #447 the
    solver predictor kept the old ``m' beta`` (off by ``m' beta / 2``), and
    re-reading the pair from it moved the public one by as much (#433, a7871319).
    """
    from superglm.model import shape_ops
    from superglm.model.fit_state import (
        FittedStateRevision,
        move_public_intercept,
        publish_revised_coefficients,
    )

    x = np.linspace(0.0, 1.0, 60)
    y = 1.5 - 1.1 * x + 0.08 * np.sin(7.0 * x)
    weights = np.resize(np.array([1.0, 3.0, 2.0, 4.0]), x.size)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.8,
        features={"x": PSpline(n_knots=6, knot_strategy="uniform")},
        weight_semantics="frequency",
    ).fit(frame, y, sample_weight=weights)
    solver = model._solver_pirls_result()
    assert solver.state_center is not None and not np.any(solver.state_center)
    assert model.result.centred_intercept != solver.centred_intercept  # m' beta is folded in
    published = (model.result.centred_intercept, model.result.centred_intercept_lo)

    revision = FittedStateRevision.start(model)
    work = revision.model
    before = np.array(work.result.beta, dtype=np.float64)
    beta = 0.5 * before
    shape_ops._replace_result_beta(work, beta)
    move_public_intercept(work, change)
    publish_revised_coefficients(work, before)
    revised = revision.commit()
    public = revised.result
    if change == 0.0:
        assert (public.centred_intercept, public.centred_intercept_lo) == published
    lo = Fraction(public.centred_intercept_lo or 0.0)
    moved = Fraction(public.centred_intercept) + lo
    moved -= Fraction(published[0]) + Fraction(published[1] or 0.0)
    assert abs(moved - Fraction(change)) <= Fraction(_U) * abs(lo)

    columns = np.asarray(revised._specs["x"].transform(x), dtype=np.float64)
    raw = float(public.intercept) + columns @ beta
    magnitude = np.abs(columns) @ np.abs(beta) + abs(float(public.centred_intercept))
    bound = (
        2.0 * _gamma(beta.size + 3) * magnitude
        + abs(float(public.centred_intercept_lo or 0.0))
        + 2.0 * _U * (abs(float(public.centred_intercept)) + abs(float(public.intercept)))
    )
    predicted = revised.predict(frame)
    assert np.all(np.abs(predicted - raw) <= bound)

    solver = revised._solver_pirls_result()
    means = np.asarray(
        revised._runtime_canonical_state["terms"]["x"]["groups"][0]["column_means"],
        dtype=np.float64,
    )
    solver_magnitude = (np.abs(columns) + np.abs(means)) @ np.abs(beta) + abs(
        float(solver.centred_intercept)
    )
    solver_bound = 2.0 * _gamma(beta.size + 3) * (magnitude + solver_magnitude) + _gamma(2) * (
        np.abs(means) @ np.abs(beta)
    )
    solver_bound += 2.0 * _U * abs(float(public.centred_intercept))
    assert np.all(np.abs(linear_predictor(revised._dm, solver, None) - predicted) <= solver_bound)


def _edit_moves_predictions_by_its_columns(model, edited, frame, term: str) -> None:
    """The edited predictions are the pre-edit ones moved by the edited term's own columns.

    With no intercept change, ``eta_after - eta_before = X_t (beta_t_new -
    beta_t_old)`` exactly; the centred predictor evaluates each side to
    ``gamma_(p+3)`` of its magnitudes ``|alpha| + |x - c| |beta| + |X| |beta|
    + |alpha_lo|`` (Higham 2002, section 3.1), and the reference adds its own
    product and one addition.  Read from the raw intercept at an offset of
    1e16 the predictions are off by tenths, so a dropped centring fails it.
    """
    assert float(edited.result.intercept) == float(model.result.intercept)
    group = next(g for g in model._groups if g.feature_name == term)
    old = np.asarray(model.result.beta, dtype=np.float64)
    new = np.asarray(edited.result.beta, dtype=np.float64)
    columns = np.asarray(model._specs[term].transform(frame[term].to_numpy()), dtype=np.float64)
    if columns.ndim == 1:
        columns = columns[:, None]
    change = new[group.sl] - old[group.sl]
    expected = model.predict(frame) + columns @ change
    public = model.result
    centre = np.asarray(public.state_center, dtype=np.float64)
    x_slot = next(g for g in model._groups if g.feature_name == "x").sl  # the centred column
    x = frame["x"].to_numpy(dtype=np.float64)
    spread = np.abs(x - centre[x_slot][0]) * abs(float(old[x_slot][0]))
    term_size = np.abs(columns) @ (np.abs(old[group.sl]) + np.abs(new[group.sl]))
    magnitude = (
        abs(float(public.centred_intercept))
        + abs(float(public.centred_intercept_lo or 0.0))
        + spread
        + term_size
    )
    p = old.size
    bound = 2.0 * _gamma(p + 3) * magnitude + _gamma(p + 1) * (
        np.abs(columns) @ np.abs(change) + np.abs(expected)
    )
    error = np.abs(edited.predict(frame) - expected)
    assert np.all(error <= bound), f"max error {float(np.max(error)):.3g}"


@pytest.mark.parametrize("offset", [1e12, 1e16])
def test_editing_a_spline_keeps_the_numeric_columns_centring(offset):
    """An editor edit of a spline beside an offset numeric keeps the centred predictor (#445).

    Sol's review of eab2d550 (P2): halving the spline's effect moves the
    intercept by 2.6e-18, which the editor skips, so the solver relation still
    holds; eab2d550 then declined to republish and cleared the whole centred
    state, numeric column included, and the edited predictions came from the
    raw intercept: 0.19 off at 1e16, 2.3e-5 at 1e12.  Master kept them to
    8.9e-16.  Check: ``_edit_moves_predictions_by_its_columns``.
    """
    from superglm.editor import EditorSession

    rng = np.random.default_rng(445)
    n = 120
    z = 2.0 * rng.integers(-4, 5, n)
    s = rng.uniform(0.0, 1.0, n)
    frame = pd.DataFrame({"x": offset + z, "s": s})
    y = 3.0 + 0.2 * z + 0.3 * np.sin(4.0 * s)
    weights = np.resize(np.array([1.0, 3.0, 2.0, 4.0]), n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            spline_penalty=0.8,
            features={"x": Numeric(), "s": PSpline(n_knots=6, knot_strategy="uniform")},
            weight_semantics="frequency",
        ).fit(frame, y, sample_weight=weights)
        session = EditorSession.from_model(model, terms=["s"], train_data=(frame, y, weights))
        term = session.terms["s"]
        term.edited_log_effect = 0.5 * np.asarray(term.edited_log_effect, dtype=np.float64)
        edited = session.to_model()
    assert edited.result.centred_intercept is not None
    _edit_moves_predictions_by_its_columns(model, edited, frame, "s")


@pytest.mark.parametrize("offset", [1e12, 1e16])
def test_an_edit_without_the_design_keeps_the_published_centred_pair(offset):
    """With ``retain_fit_state=False`` an edit keeps the published pair it cannot rebuild (#445).

    Sol's review of eab2d550 (P2): after a pickle reload the model holds no
    design, ``_public_centred_state`` returned ``(None, None, None)`` and the
    republication erased the published pair; the edited categorical's
    predictions came from the raw intercept, 0.15 off at 1e16.  The edit now
    carries the published pair (``publish_revised_coefficients``), which needs
    no design.  Check: the pair is unchanged (the edit moves
    only folded, uncentred columns) and ``_edit_moves_predictions_by_its_columns``.
    """
    import pickle

    from superglm.editor import EditorSession

    rng = np.random.default_rng(445)
    n = 120
    z = 2.0 * rng.integers(-4, 5, n)
    g = np.resize(np.array(["a", "b", "c", "d"], dtype=object), n)
    frame = pd.DataFrame({"x": offset + z, "g": g})
    y = 3.0 + 0.2 * z + np.resize(np.array([0.0, 0.2, 0.3, -0.4]), n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric(), "g": Categorical(base="first")},
            retain_fit_state=False,
        ).fit(frame, y)
        model = pickle.loads(pickle.dumps(fitted))
        assert model._dm is None
        session = EditorSession.from_model(model, terms=["g"], train_data=(frame, y))
        term = session.terms["g"]
        term.edited_log_effect = 0.5 * np.asarray(term.edited_log_effect, dtype=np.float64)
        edited = session.to_model()
    published = (model.result.centred_intercept, model.result.centred_intercept_lo)
    assert published[0] is not None
    assert (edited.result.centred_intercept, edited.result.centred_intercept_lo) == published
    _edit_moves_predictions_by_its_columns(model, edited, frame, "g")


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
    ``(x - hi) - lo`` (``centered_system.weighted_mean_pair``), ``hi`` the
    rounded weighted mean and ``lo`` the remainder formed on rows differenced
    from it.  ``hi`` lands on a different float at each offset, so the two
    fits' centred rows are not bitwise equal: ``x - hi`` is exact by Sterbenz
    at 1e16 and rounds at the spread's scale at 0, and ``lo`` absorbs the
    difference in ``hi``.  They agree to ``O(u spread)`` per entry, and the
    ``(u s / sigma)^2`` term is gone.  Their predictors then agree to the forward error
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
    # L = 1 on the unscaled path, the products' scale on the overflow fallback
    scale = max(Fraction(1), largest_product + abs(exact) * largest_weight)
    scaling = Fraction(2) ** -1074 * (2 + 16 * (len(values) + 5) * scale / total)
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
        ([1e150, -1e150, 1e-200], [1.0] * 3),
        ([float(np.finfo(float).max)] * 2, [0.01, 0.02]),
        ([1e-20, 3e-20, 3e-20], [1e-300] * 3),
        ([1e10, 1e10], [1e-300, 1e-300]),
        ([1e-310, 2e-310], [1e10, 1e10]),
    ],
    ids=[
        "subnormal_weights",
        "wide_range",
        "mixed_exponents",
        "cancelling_giants",
        "largest_float",
        "subnormal_products",
        "weights_below_the_merge",
        "values_below_the_merge",
    ],
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
    returned 0 for ``1e-300`` (Sol; ``np.average`` gets it).  Its successor,
    scaling every product by the largest one's power of two, lost ``1e-200``
    before ``1e150 - 1e150`` cancelled (0 for ``3.3e-201``), and the quotient
    of two ``finfo.max`` raised ``OverflowError`` in ``math.ldexp`` (Sol):
    the products are now summed unscaled, the subnormal ones carried at a
    shifted scale, and a quotient past the range falls back instead of
    raising.  The weight total and the numerator can then sit at different
    scales (``2^-1126`` and ``2^0``); dividing their significands first
    rounded ``1e10 * 2^-1126`` to 0 for a mean of 1e10 (Claude).  Mutations:
    the weights scaled by the largest one, the TwoSum error dropped, or the
    significands divided before their scales.
    """
    from fractions import Fraction

    from superglm.solvers.mode_score import compensated_weighted_mean

    exact = _exact_mean(values, weights)
    mean = compensated_weighted_mean(np.array(values), np.array(weights))
    assert abs(Fraction(mean) - exact) <= _mean_bound(values, weights, exact)
    # beyond the subnormal results, the mean is within two roundings of exact
    assert abs(Fraction(mean) - exact) <= 2 * Fraction(_U) * abs(exact) + Fraction(2) ** -1075


def test_the_scaled_ratio_divides_mantissas_whichever_sum_is_larger(monkeypatch):
    """Every finite non-zero pair of scaled sums takes the mantissa path.

    Products below ``2^-969`` are carried at ``K = -1126`` and a weight total
    above ``2^-968`` at ``K = 0``.  The quotient of the stored sums is then
    ``2^146 / 2^-960 = 2^1106``: the guard tested that quotient, returned
    ``inf``, and the compensated mean fell back to ``np.average`` (Claude).
    The mirror arrangement is the one ``weights_below_the_merge`` pins.  Here
    the fallback is made to raise, and the mean must equal the exact
    ``Fraction`` mean.  Mutation: the guard on the quotient.
    """
    from fractions import Fraction

    from superglm.solvers.mode_score import _scaled_ratio, compensated_weighted_mean

    assert _scaled_ratio((2.0**146, -1126), (2.0**-960, 0)) == 2.0**-20
    assert _scaled_ratio((2.0**-20, 0), (2.0**146, -1126)) == 2.0**960
    assert math.isnan(_scaled_ratio((1.0, 0), (0.0, 0)))  # no quotient: nan, never a raise

    def no_fallback(*args, **kwargs):
        raise AssertionError("the compensated mean fell back to np.average")

    monkeypatch.setattr(np, "average", no_fallback)
    values = np.array([2.0**-20, 3.0 * 2.0**-21])
    weights = np.array([2.0**-960, 2.0**-959])
    exact = sum(Fraction(v) * Fraction(w) for v, w in zip(values, weights, strict=True)) / sum(
        Fraction(w) for w in weights
    )
    assert compensated_weighted_mean(values, weights) == float(exact)


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
    for draw in range(120):
        n = int(rng.integers(2, 40))
        values = rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(-300, 300, n)
        if draw % 3 == 0:
            # giants that cancel exactly, beside small terms
            half = n // 2
            values[:half] = 10.0 ** rng.uniform(100, 300, half)
            values[half : 2 * half] = -values[:half]
        values = values.tolist()
        weights = (10.0 ** rng.uniform(-300, 300, n) if draw % 4 else np.ones(n)).tolist()
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

    largest = float(np.finfo(float).max)
    frame = pd.DataFrame(index=range(2))
    model = SuperGLM(family="gaussian", features={}, selection_penalty=0).fit(
        frame, np.array([largest, largest]), sample_weight=np.array([0.01, 0.02])
    )
    np.testing.assert_array_equal(model.predict(frame), np.full(2, largest))
    assert model.result.deviance == 0.0


def test_the_intercept_remainder_never_worsens_the_mean():
    """An intercept-only Gaussian fit on ``[1e12, -1e12, 1]`` predicts the correctly rounded 1/3.

    ``centred_intercept_remainder`` added each residual's TwoSum error back into
    its large head before an ordinary reduction, so the errors were absorbed
    and the remainder was noise: the fit predicted 0.3333062 where master
    predicts 1/3 (Sol).  The residual is now carried as four floats and summed
    exactly, so the pair ``(alpha, alpha_lo)`` is the mean to ``3u
    |alpha_lo|``, and the prediction rounds to the correctly rounded mean.
    Mutation: the remainder from the absorbed residual.
    """
    from fractions import Fraction

    frame = pd.DataFrame(index=range(3))
    y = np.array([1e12, -1e12, 1.0])
    model = SuperGLM(family="gaussian", features={}, selection_penalty=0).fit(frame, y)
    exact = Fraction(1, 3)
    np.testing.assert_array_equal(model.predict(frame), np.full(3, float(exact)))
    solver = model._solver_pirls_result()
    pair = Fraction(float(solver.intercept)) + Fraction(float(solver.centred_intercept_lo))
    assert abs(pair - exact) <= 3 * Fraction(_U) * abs(Fraction(solver.centred_intercept_lo))


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


def test_the_proximal_certificate_rejects_an_unresolved_slope():
    """The composite KKT certificate reads a dense column's score and curvature about its pair.

    ``x = 1e12 + z``, ``y = 3 + 0.2 z`` and the slope at 1e-12, the predictor
    ``3.2 + 1e-12 z``.  Raw, the curvature ``offset^2 sum W`` shrank the
    proximal step to 1e-24 against a slope of 1e-12, a violation of 1.6e-13,
    and the state passed as converged: the false convergence of the raw
    solver, which stopped with its slope near zero.  Centred, the step is the
    slope's whole error and the violation is 1.  Mutation: the raw
    certificate.
    """
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.penalties.group_lasso import GroupLasso
    from superglm.solvers.irls_state import _evaluate_irls_state
    from superglm.solvers.pirls import _composite_kkt_violation
    from superglm.types import GroupSlice

    z = np.tile([-2.0, 0.0, 2.0, 4.0], 100)
    x = 1e12 + z
    y = 3.0 + 0.2 * z
    dm = DesignMatrix([DenseGroupMatrix(x[:, None])], n=len(x), p=1)
    groups = [GroupSlice("x", 0, 1, weight=1.0)]
    weights, offset = np.ones(len(x)), np.zeros(len(x))
    beta = np.array([1e-12])
    state = _evaluate_irls_state(dm, y, weights, Gaussian(), IdentityLink(), offset, beta, 2.2)
    tol = 1e-6
    violation = _composite_kkt_violation(
        dm=dm,
        state=state,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        offset=offset,
        groups=groups,
        penalty=GroupLasso(lambda1=0.01),
        S=None,
        has_smooth_penalty=False,
        tol=tol,
    )
    assert violation >= tol


def _proximal_fixture(family: str, shift: float):
    """An even-integer column (exact at every offset) and a response from it."""
    z = np.tile([-2.0, 0.0, 2.0, 4.0], 100)
    rng = np.random.default_rng(40)
    if family == "poisson":
        y = rng.poisson(np.exp(0.3 + 0.1 * z)).astype(float)
    else:
        y = 3.0 + 0.2 * z + np.tile([0.01, -0.02, 0.03, -0.02], 100)
    return pd.DataFrame({"x": shift + z}), y


@pytest.mark.parametrize("shift", [40.0, 2010.0, 1e5, 1e12])
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_the_proximal_solver_centres_a_dense_column(family, shift):
    """A selection fit and its path reach the same certified optimum at every offset.

    The proximal solver ran block coordinate descent in raw coordinates: the
    intercept and a column at offset ``m`` with spread ``s`` shared a curvature
    ``(m^2 + s^2) sum W``, so each sweep shrank the slope's error by only
    ``m^2 / (m^2 + s^2)``.  On master an offset of 40 took 99 iterations to a
    slope 5e-4 off, 2010 never converged, and 1e5 and 1e12 claimed
    convergence with the slope at zero.  A dense column is now updated about
    its exact pair and the state is kept about the prior-weighted centre (the
    unpenalized intercept separates from centred columns; Friedman, Hastie &
    Tibshirani 2010, §2.6), so the fit reaches the optimum it reaches at no
    offset; its iteration counts are measured in the PR's selection panel,
    not asserted here.  One penalized column: its curvature is its own strong convexity,
    so the certificate puts each fit within ``tol`` of the optimum's slope and
    the two within ``2 tol``.  Mutation: the raw block update.
    """
    model = SuperGLM(family=family, features={"x": Numeric()}, selection_penalty=0.01)
    tol = model._tol

    def fit(shift: float):
        frame, y = _proximal_fixture(family, shift)
        model = SuperGLM(family=family, features={"x": Numeric()}, selection_penalty=0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(frame, y)
            path = SuperGLM(
                family=family, features={"x": Numeric()}, selection_penalty=0.01
            ).fit_path(frame, y, n_lambda=8)
        return model.result, path

    base, base_path = fit(0.0)
    result, path = fit(shift)
    assert base.converged and result.converged
    slope = float(base.beta[0])
    assert abs(float(result.beta[0]) - slope) <= 2.0 * tol * abs(slope)
    assert np.all(path.converged_path)
    scale = float(np.max(np.abs(base_path.coef_path)))
    assert np.max(np.abs(path.coef_path - base_path.coef_path)) <= 2.0 * tol * scale


@pytest.mark.parametrize("entry", ["fit", "fit_path", "refit_at_a_path_point"])
@pytest.mark.parametrize("shift", [0.0, 40.0, 1e12, 1e16])
def test_a_proximal_fit_predicts_the_deviance_it_reports(entry, shift):
    """``predict()`` on the training rows reproduces a selection fit's reported deviance.

    The proximal solver keeps its intercept centred (item 13), but its result
    published only the raw reading ``alpha - c' beta``, which cancels at a
    column's offset: at 1e16 a Gaussian fit reported deviance 0.172 and
    predicted the levels [2.75, 3, 3.5, 3.75], a squared error of 2.58, and
    ``fit_path``'s final model did the same (Sol).  A path's other points
    are predicted by refitting at ``lambda_seq[i]`` (``PathResult``), the
    third entry.  Both predictors are
    ``alpha + (x - c) beta``, each row formed with at most four roundings,
    with ``|x - c| <= R = max|z| + 2`` (``c`` lies within one grid spacing,
    at most 2 here, of the mean) and ``|alpha| <= max|mu| + R |beta|``.  Per
    row they differ by at most ``delta = 2 gamma_4 (max|mu| + 2 R |beta|)``,
    so the deviances differ by at most ``2 delta sum|r| + n delta^2 + 2
    gamma_n D``.

    The refit must also be path point 3 itself, as ``PathResult`` promises.
    One penalized column of a Gaussian identity fit: the slope's objective is
    quadratic with curvature ``L = sum (z - zbar)^2``, so the certificate's
    proximal step ``d`` lands on the optimum and ``|d| <= tol s + a`` bounds
    each converged fit's distance from it
    (``test_the_proximal_solver_centres_a_dense_column``).  The scale ``s``
    holds the score step ``|beta - b0|``, ``b0`` the unpenalized slope, so ``s
    <= |b0| + |d|`` (the shrunk slope lies between 0 and ``b0``), not
    ``|beta|``; ``a`` is the certificate's arithmetic allowance, ``gamma'_{n
    + 5} (||y|| + ||mu||) / sqrt(L) + gamma'_6 3 |b0|`` (``gamma'`` counts
    ``eps``).  Each slope is within ``(tol |b0| + a) / (1 - tol)`` of the
    optimum, the two within twice that, and the deviances, quadratic in the
    slope, differ by at most ``L |beta_r - beta_p| (|beta_r - b0| + |beta_p -
    b0|)``, plus each intercept's error, at most ``((tol + gamma'_{n+2})
    sum(|y| + |mu|))^2 / n``, and each deviance's rounding, half the bound
    above.  ``fit_path`` runs ``fit_pirls`` at its default ``tol``, equal to
    the model's default.  Mutations: the result published without its
    centred state; path point 3 fitted at ``lambda_seq[4]`` or at
    ``lambda_seq[3] (1 + 1e-3)``; ``deviance_path`` published one point off.
    """
    z = np.tile([-2.0, 0.0, 2.0, 4.0], 100)
    y = 3.0 + 0.2 * z + np.tile([0.01, -0.02, 0.03, -0.02], 100)
    frame = pd.DataFrame({"x": shift + z})
    model = SuperGLM(family="gaussian", features={"x": Numeric()}, selection_penalty=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if entry == "fit":
            model.fit(frame, y)
            reported = float(model.result.deviance)
        elif entry == "fit_path":
            reported = float(model.fit_path(frame, y, n_lambda=8).deviance_path[-1])
        else:
            path = model.fit_path(frame, y, n_lambda=8)
            model = SuperGLM(
                family="gaussian",
                features={"x": Numeric()},
                selection_penalty=float(path.lambda_seq[3]),
            ).fit(frame, y)
            reported = float(model.result.deviance)
    assert model.result.converged
    mu = np.asarray(model.predict(frame), dtype=np.float64)
    residual = y - mu
    n = len(y)
    u = np.finfo(np.float64).eps / 2.0

    def gamma(k: int) -> float:
        return k * u / (1.0 - k * u)

    spread = float(np.max(np.abs(z))) + 2.0
    slope = float(np.max(np.abs(model.result.beta)))
    delta = 2.0 * gamma(4) * (float(np.max(np.abs(mu))) + 2.0 * spread * slope)
    bound = 2.0 * delta * float(np.sum(np.abs(residual))) + n * delta**2 + 2.0 * gamma(n) * reported
    assert abs(float(np.sum(residual**2)) - reported) <= bound
    if entry != "refit_at_a_path_point":
        return
    assert path.converged_path[3]
    tol = model._tol
    centred = z - np.mean(z)
    curvature = float(np.sum(centred**2))
    unpenalized = float(np.sum(centred * (y - np.mean(y)))) / curvature
    allowance = gamma(2 * (n + 5)) * float(np.linalg.norm(y) + np.linalg.norm(mu)) / np.sqrt(
        curvature
    ) + gamma(12) * 3.0 * abs(unpenalized)
    distance = (tol * abs(unpenalized) + allowance) / (1.0 - tol)
    refit, point = float(model.result.beta[0]), float(path.coef_path[3, 0])
    assert abs(refit - point) <= 2.0 * distance
    moved = 2.0 * distance * curvature * (abs(refit - unpenalized) + abs(point - unpenalized))
    intercept = ((tol + gamma(2 * (n + 2))) * float(np.sum(np.abs(y) + np.abs(mu)))) ** 2 / n
    assert abs(reported - float(path.deviance_path[3])) <= moved + 2.0 * intercept + bound


def test_cross_validation_scores_a_selection_fit_the_same_at_an_offset():
    """``cross_validate`` of a selection fit reads the fold deviances it reads at no offset.

    Each fold refits and predicts its held-out rows, so the published centred
    state reaches cross-validation too: on 5e8988d1 a Numeric at 1e16 scored
    the folds 0.007 to 0.036 against 0.0004 at no offset.  Each fold's slope
    is within ``2 tol |beta|`` of its unshifted fit (one penalized column,
    ``test_the_proximal_solver_centres_a_dense_column``) and its intercept the
    held-out mean's, so a held-out row moves by at most ``delta = 2 R 2 tol
    |beta|``, ``R = 6`` the grid's range, and a fold's mean squared residual
    ``m`` by at most ``2 sqrt(m) delta + delta^2 + 2 gamma_n m``.  Mutation: the
    result published without its centred state.
    """
    from sklearn.model_selection import KFold

    from superglm import cross_validate

    z = np.tile([-2.0, 0.0, 2.0, 4.0], 100)
    y = 3.0 + 0.2 * z + np.tile([0.01, -0.02, 0.03, -0.02], 100)
    scores = {}
    for shift in (0.0, 1e16):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cross_validate(
                SuperGLM(family="gaussian", features={"x": Numeric()}, selection_penalty=0.01),
                pd.DataFrame({"x": shift + z}),
                y,
                cv=KFold(4, shuffle=True, random_state=0),
                scoring=("deviance",),
                return_estimators=True,
            )
        scores[shift] = result
    base, shifted = scores[0.0], scores[1e16]
    tol = SuperGLM(family="gaussian", features={"x": Numeric()})._tol
    u = np.finfo(np.float64).eps / 2.0
    for fold, estimator in enumerate(base.estimators):
        m = float(base.fold_scores["deviance"].iloc[fold])
        n = int(base.fold_scores["n_test"].iloc[fold])
        delta = 2.0 * 6.0 * 2.0 * tol * abs(float(estimator.result.beta[0]))
        gamma = n * u / (1.0 - n * u)
        bound = 2.0 * np.sqrt(m) * delta + delta**2 + 2.0 * gamma * m
        assert abs(float(shifted.fold_scores["deviance"].iloc[fold]) - m) <= bound


def _merge_rounding_count(sizes: list[int]) -> int:
    """``N``: each entry of ``_anchored_weighted_moments``'s merge is within ``gamma_N sqrt(C_ii C_jj)``.

    ``sizes`` are the row counts of the ``K`` chunks that carry weight, ``n``
    the largest.  The co-moment is ``C = S + B``: ``S`` the chunks'
    co-moments ``C^t`` about their own means, ``B`` the merge terms ``T_t =
    m_t d_t d_t'``, which sum to the chunks' scatter ``sum_t w_t (mean_t -
    M)(mean_t - M)'`` about the overall mean ``M`` (Chan, Golub & LeVeque
    1979).  Every part is positive semidefinite, so Cauchy-Schwarz bounds a
    sum of ``|part_ij|`` by its diagonal pair.  Counting each rounding
    relative to the value it rounds, to first order in ``u`` (Higham 2002,
    §3.1, Lemma 3.3, §4.2):

    - a chunk co-moment, ``n_t + 7`` of ``sum w |r_i| |r_j|``, ``r`` the rows
      about the chunk's pair ``(hi, lo)``: the rows ``(x - hi) - lo``, 2 in
      each factor; ``w`` times a row, 1; the dot product, ``n_t``; the
      profiled intercept's subtraction and the symmetrising sum, 2.  The
      profiled term is the square of the pair's error;
    - merge term ``t``, ``t + 13`` of ``|T_t,ij|``: ``d``, 2 in each factor;
      ``sqrt(m)``, 4 in each (two square roots, their product, the share's sum
      and quotient); ``sqrt(m) d``, 1 in each; the outer product, 1; the
      running total, ``t - 2`` additions;
    - the sum of the ``2K - 1`` parts, ``2K - 2``;
    - a chunk's weight, ``n_t - 1`` additions, and its mean, off by
      ``gamma_{n_t+1} sum w |x - hi| / w_t`` (``lo``'s dot product).  ``B`` is
      stationary in ``M``, so these move it only through each chunk's own
      term, by at most ``gamma_{n-1} sqrt(B_ii B_jj)`` and ``gamma_{n+1}
      (sqrt(B_ii S_jj) + sqrt(S_ii B_jj))`` in all;
    - the running mean, moved from the heavier side by the lighter side's
      share ``l`` of ``d``: ``t + 4`` roundings of ``l |d|`` (``d``, 2; the
      share, 2 and the running total's ``t - 2``; the product, 1; the tail's
      sum, 1; TwoSum is exact).  An error ``e`` in the mean ``M_t`` of the
      first ``t`` chunks, of weight ``W_t``, moves ``B`` by ``W_t ((M_t - M)
      e' + e (M_t - M)')``.  With ``W_t (M_t - M)_i^2 <= B_ii`` and ``sqrt(W_t) l |d| <=
      sqrt(T_t,ii)``, the ``K - 1`` steps add ``2 (K + 4) sqrt(K - 1) u
      sqrt(B_ii B_jj)``.

    So ``sqrt(S_ii S_jj)`` carries at most ``n + 2K + 5`` roundings,
    ``sqrt(B_ii B_jj)`` ``n + 3K + 10 + 2 (K + 4) sqrt(K - 1)`` and the mixed
    pair ``n + 1``.  As a 2x2 form in the vectors ``(sqrt S_ii, sqrt B_ii)``,
    of length ``sqrt C_ii``, the total is at most the larger diagonal plus
    the off-diagonal (Gershgorin), ``N``.  Dropped: products of roundings,
    and the tails' roundings, ``u^2`` times a mean, below ``u`` times a
    spread while each mean is within ``1/u`` spreads of zero (``1e12`` here).
    """
    count = len(sizes)
    steps = 2.0 * (count + 4) * math.sqrt(count - 1)
    return 2 * max(sizes) + 3 * count + 11 + math.ceil(steps)


def test_the_metrics_gram_does_not_anchor_on_a_far_first_chunk():
    """The streamed centred Gram merges chunk co-moments, so no one chunk sets its anchor.

    Codex's case: the first chunk holds 100 rows of negligible weight at
    ``1e8 + z``, the rest 900 rows of unit weight at ``z``.  About the first
    chunk's mean the Gram subtracted ``sum w (x - a)^2`` and ``(sum w (x -
    a))^2 / sum w``, both near ``9e18``, and kept nothing of the 900.  Each
    chunk is now centred about its own mean and the co-moments merged
    pairwise (Chan, Golub & LeVeque 1979), so the result is within the
    merge's bound, ``gamma_N C`` with ``N = _merge_rounding_count`` of ten
    chunks of 100 rows (325), against the exact co-moment; the error is
    3.3e-3 of it.  Mutation: the one-anchor formula.
    """
    from superglm.inference._metrics_design import _anchored_weighted_moments

    rng = np.random.default_rng(440)
    x = np.concatenate((1e8 + rng.normal(size=100), rng.normal(size=900)))
    W = np.concatenate((np.full(100, 1e-20), np.ones(900)))

    def chunks():
        for start in range(0, len(x), 100):
            yield start, start + 100, x[start : start + 100, None]

    _, _, centred = _anchored_weighted_moments(chunks, W, 1)
    total = _exact_product_sum((value,) for value in W)
    first = _exact_product_sum(zip(W, x, strict=True))
    exact = _exact_product_sum(zip(W, x, x, strict=True)) - first * first / total
    bound = _gamma(_merge_rounding_count([100] * 10)) * float(exact)
    assert abs(Fraction(float(centred[0, 0])) - exact) <= Fraction(bound)


@pytest.mark.parametrize(
    "gap, weight_a, weight_b",
    [
        (1e160, 1e-24, 1e-24),
        (1e-200, 1e100, 1e100),
        (1e100, 1e200, 1e-200),
        (1e100, 1e-200, 1e200),
        (1e160, 1e296, 1e-28),
    ],
    ids=[
        "large_gap_small_weight",
        "small_gap_large_weight",
        "heavy_chunk_first",
        "light_chunk_first",
        "heavy_total_far_light_chunk",
    ],
)
def test_the_metrics_gram_merges_chunks_without_over_or_underflow(gap, weight_a, weight_b):
    """The pairwise merge's term ``m d d'`` neither over- nor underflows where its value does not.

    Two chunks of 8192 rows, at 0 with weight ``weight_a`` and at ``gap``
    with weight ``weight_b``: the centred Gram is ``m gap^2``, ``m = W_a W_b /
    (W_a + W_b)``.  Master reads 4.096e299, 4.096e-297 and 8192 here; the
    last case is Codex's, a total near 1e300 beside a chunk near 1e-24 at a
    separation of 1e160, a term near 8e295.  Formed
    as ``d d'`` before the weight, the merge read ``inf`` and 0 (Sol, on
    a40b61c8's parent); with ``w_b / (w_a + w_b)`` formed before the square
    root, 1e-200 beside 1e200 underflowed and read 0, and the rank 0 (Sol, on
    d7971d77).  The result must lie within ``4 gamma_{n+4}`` of the exact
    value.  Mutations: ``m * outer(d, d)``; the raw share.
    """
    from superglm.inference._metrics_design import _anchored_weighted_moments

    x = np.repeat([0.0, gap], 8192)
    W = np.repeat([weight_a, weight_b], 8192)

    def chunks():
        for start in range(0, x.size, 8192):
            yield start, start + 8192, x[start : start + 8192, None]

    _, _, centred = _anchored_weighted_moments(chunks, W, 1)
    total_a, total_b = 8192 * Fraction(weight_a), 8192 * Fraction(weight_b)
    exact = float(total_a * total_b / (total_a + total_b) * Fraction(gap) ** 2)
    u = np.finfo(np.float64).eps / 2.0
    k = x.size + 4
    assert np.isfinite(centred[0, 0]) and centred[0, 0] > 0.0
    assert abs(float(centred[0, 0]) - exact) <= 4.0 * (k * u / (1.0 - k * u)) * exact


def _exact_product_sum(terms) -> Fraction:
    """The exact sum of products of floats: each term a tuple of floats."""
    numerators, exponents = [], []
    for factors in terms:
        numerator, exponent = 1, 0
        for value in factors:
            top, bottom = float(value).as_integer_ratio()
            numerator *= top
            exponent += bottom.bit_length() - 1
        numerators.append(numerator)
        exponents.append(exponent)
    scale = max(exponents)
    shifted = sum(n << (scale - e) for n, e in zip(numerators, exponents, strict=True))
    return Fraction(shifted, 1 << scale)


def test_the_metrics_gram_merge_meets_its_bound_against_exact_arithmetic():
    """The merged centred Gram against exact rational arithmetic, over draws spanning +-300 decades.

    Each draw has 2 to 6 chunks of 1 to 400 rows; a chunk's weights sit near
    ``10^e_w``, ``e_w`` over +-300.  Two draws in three put each chunk's two
    columns near their own ``+-10^e_x`` with 50% spread; the third translates
    every chunk to one offset ``+-10^e_c`` with a spread of 1e-12 to 1e-4 of
    it, the chunks' means a few spreads apart.  The exponents are drawn so a
    row's ``w x^2`` lies within 1e+-290 (every exact entry then
    representable, the products normal).  Every entry lies within ``gamma_N
    sqrt(C_ii C_jj)`` of the exact co-moment, ``N`` the merge's rounding
    count (``_merge_rounding_count``: ``2 n + 3K + 11 + 2 (K + 4) sqrt(K -
    1)``, ``n`` the largest chunk, ``K`` the chunks), and is never inf, NaN,
    or a zero diagonal.  On these 600 draws the largest error is 0.11 of
    that bound.  Against the looser ``16 gamma_{n + 2K}``, which ``gamma_N``
    never exceeds here, master's row-0 anchor failed 271 (483 entries NaN or
    infinite, 108 zero diagonals) and d7971d77's merge 204.  Mutations: the
    raw share; the running mean stepped from the lighter side; the means'
    tails dropped from ``d``; ``m`` applied after ``d d'``; the running
    mean's tail dropped.
    """
    from superglm.inference._metrics_design import _anchored_weighted_moments

    rng = np.random.default_rng(4300)
    for draw in range(600):
        blocks, weights = [], []
        # every third draw translates all its chunks to one far offset: a
        # spread 10^-12 to 10^-4 of it, the chunks' means apart by a few spreads
        translated = draw % 3 == 2
        e_c = rng.uniform(-140.0, 140.0, size=2)
        offset = rng.choice([-1.0, 1.0], size=2) * 10.0**e_c
        spread = 10.0 ** rng.uniform(-12.0, -4.0)
        for _chunk in range(int(rng.integers(2, 7))):
            size = int(rng.choice([1, 2, 7, 60, 400]))
            if translated:
                e_w = rng.uniform(
                    max(-300.0, -290.0 - 2.0 * e_c.min()), min(300.0, 290.0 - 2.0 * e_c.max())
                )
                shift = rng.normal(scale=3.0)
                blocks.append(offset * (1.0 + spread * (shift + rng.standard_normal((size, 2)))))
            else:
                e_w = rng.uniform(-300.0, 300.0)
                low, high = max(-300.0, (-290.0 - e_w) / 2.0), min(300.0, (290.0 - e_w) / 2.0)
                centre = rng.choice([-1.0, 1.0], size=2) * 10.0 ** rng.uniform(low, high, size=2)
                blocks.append(centre * (1.0 + 0.5 * rng.standard_normal((size, 2))))
            weights.append(10.0 ** (e_w + 0.3 * rng.standard_normal(size)))
        x, w = np.vstack(blocks), np.concatenate(weights)
        edges = np.cumsum([0] + [len(block) for block in blocks])

        def chunks(blocks=blocks, edges=edges):
            for index, block in enumerate(blocks):
                yield int(edges[index]), int(edges[index + 1]), block

        _, _, centred = _anchored_weighted_moments(chunks, w, 2)
        total = _exact_product_sum((value,) for value in w)
        sums = [_exact_product_sum(zip(w, x[:, j], strict=True)) for j in range(2)]
        exact = [
            [
                _exact_product_sum(zip(w, x[:, i], x[:, j], strict=True))
                - sums[i] * sums[j] / total
                for j in range(2)
            ]
            for i in range(2)
        ]
        diagonal = [float(exact[i][i]) for i in range(2)]
        gamma = _gamma(_merge_rounding_count([len(block) for block in blocks]))
        for i in range(2):
            assert centred[i, i] > 0.0
            for j in range(2):
                assert np.isfinite(centred[i, j])
                bound = gamma * np.sqrt(diagonal[i]) * np.sqrt(diagonal[j])
                assert abs(Fraction(float(centred[i, j])) - exact[i][j]) <= Fraction(bound)


@pytest.mark.parametrize("position", ["first", "last"])
def test_metrics_on_new_rows_ignore_where_a_zero_weight_row_sits(position):
    """``metrics()`` on rows with a zero-weight row far from the data reads the fit's standard error.

    The evaluation design anchored its raw-moment subtraction on row 0
    whatever its weight: a zero-weight row at ``x = 0`` before Numeric rows
    at 1e12 formed a centred Gram of 2^36 for 2000, which passed as
    authoritative, and the slope's standard error read 7.9e-8 for 4.65e-4
    (Sol; also on master).  The moments are now anchored on the rounded
    weighted mean, so the row's position does not matter.  The bound: a 1x1
    Gram on ``n`` rows and the scale from the same residuals, ``8 gamma_n``
    relative.  Mutation: the anchor back on row 0.
    """
    z = np.tile([-2.0, 0.0, 2.0, 4.0], 100)
    frame = pd.DataFrame({"x": 1e12 + z})
    y = 3.0 + 0.2 * z + np.tile([0.01, -0.02, 0.03, -0.02], 100)
    model = SuperGLM(family="gaussian", features={"x": Numeric()}, selection_penalty=0).fit(
        frame, y
    )
    far = pd.DataFrame({"x": [0.0]})
    if position == "first":
        rows = pd.concat([far, frame], ignore_index=True)
        response, weights = np.r_[0.0, y], np.r_[0.0, np.ones(len(z))]
    else:
        rows = pd.concat([frame, far], ignore_index=True)
        response, weights = np.r_[y, 0.0], np.r_[np.ones(len(z)), 0.0]
    reference = float(model.metrics(frame, y).coefficient_se["x"][0])
    se = float(model.metrics(rows, response, sample_weight=weights).coefficient_se["x"][0])
    assert abs(se - reference) <= 8.0 * _gamma(len(rows)) * reference


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


@pytest.mark.parametrize("bounded", ["discrete_categorical", "spline"])
def test_the_dense_split_matches_the_exact_pair_reference(bounded, monkeypatch):
    """``build_centered_system``'s dense/bounded split against the full exact-pair system.

    Beside a dense column the bounded columns' block comes from a raw rung
    and the dense columns join it centred about their exact pair, with the
    cross block ``N' (W D~) - m_N (1' W D~)`` (``_attach_dense_split``).  The
    reference centres the whole design about ``weighted_mean_pair``'s ``(hi,
    lo)``.  Each entry agrees to ``4 gamma_{n+p+4}`` times the same entry
    formed on absolute rows: a dense column's centred rows, a bounded
    column's ``|x| + |m|`` (its raw rung subtracts the mean's outer product).
    Mutation: the cross block formed without ``W``, which is the same at
    every offset, so the translation tests cannot see it.
    """
    import scipy.sparse as sp

    from superglm._group_matrix._group_matrix_centered import centered_gram_rhs
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        SparseSSPGroupMatrix,
    )
    from superglm.solvers import centered_system

    rng = np.random.default_rng(439)
    n = 9000  # above the raw rungs' size crossovers for both bounded halves
    dense = DenseGroupMatrix(2010.0 + 5.0 * rng.normal(size=(n, 2)))
    if bounded == "discrete_categorical":
        support = rng.normal(size=(8, 3))
        support -= support.mean(axis=0)
        others = [
            DiscretizedSSPGroupMatrix(support, np.eye(3), np.arange(n, dtype=np.intp) % 8),
            CategoricalGroupMatrix(np.arange(n, dtype=np.intp) % 120, n_levels=120),
        ]
    else:
        others = []
        for width, phase in ((12, 0), (10, 3)):
            rows = np.repeat(np.arange(n, dtype=np.intp), 4)
            columns = (rows + np.tile(np.arange(4, dtype=np.intp), n) + phase) % width
            values = np.tile(np.array([0.1, 0.4, 0.4, 0.1]), n)
            basis = sp.csr_matrix((values, (rows, columns)), shape=(n, width))
            others.append(SparseSSPGroupMatrix(basis, np.eye(width)))
    groups = [dense, *others]
    dm = DesignMatrix(groups, n=n, p=sum(group.shape[1] for group in groups))
    W = rng.uniform(0.5, 2.0, size=n)
    z = rng.normal(size=n)
    attached = []
    real_attach = centered_system._attach_dense_split

    def recording_attach(split, **kwargs):
        attached.append(split)
        return real_attach(split, **kwargs)

    monkeypatch.setattr(centered_system, "_attach_dense_split", recording_attach)
    system = centered_system.build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_split=dm.tabmat_centering_split,
        tabmat_state=centered_system.TabmatCenteringState(),
    )
    assert attached, "the bounded half took no raw rung: the split was not exercised"

    sum_w = float(np.sum(W))
    mean_x, hi, lo = centered_system.weighted_mean_pair(dm, W, sum_w)
    z_centered = z - system.mean_z
    gram, rhs = centered_gram_rhs(dm=dm, W=W, mean_x=hi, z_centered=z_centered, mean_lo=lo)
    X = dm.toarray()
    is_dense = np.zeros(dm.p, dtype=bool)
    is_dense[: dense.shape[1]] = True
    magnitude = np.where(is_dense, np.abs((X - hi) - lo), np.abs(X) + np.abs(mean_x))
    u = np.finfo(np.float64).eps / 2.0
    k = n + dm.p + 4
    gamma = k * u / (1.0 - k * u)
    assert np.all(
        np.abs(system.data_gram - gram) <= 4.0 * gamma * (magnitude.T @ (W[:, None] * magnitude))
    )
    assert np.all(
        np.abs(system.rhs - rhs) <= 4.0 * gamma * (magnitude.T @ (W * np.abs(z_centered)))
    )
    assert np.all(np.abs(system.mean_x - mean_x) <= 4.0 * gamma * (np.abs(X).T @ W) / sum_w)


# One pass for the working mean (the #439 follow-up).  Every dense column takes
# the corrected two-pass algorithm about its working-weighted mean, by type:
# pass one rounds the mean ``a`` (shifted by the first weighted row), pass two
# forms the rows ``x - a`` once and from them ``e = sum W (x - a)``, the Gram
# ``G`` and the right-hand side, and Björck's correction ``G - e e' / sum W``
# folds in the remainder that #439 formed in a pass of its own (Chan, Golub &
# LeVeque 1983, eq. 1.7 and Table 1).


def _exact_centred_gram(values: NDArray, weights: NDArray) -> list[list[Fraction]]:
    """``sum w (x_j - m_j)(x_k - m_k)`` about the exact weighted means, in rationals."""
    total = sum(Fraction(w) for w in weights)
    columns = [[Fraction(v) for v in values[:, j]] for j in range(values.shape[1])]
    fractions = [Fraction(w) for w in weights]
    means = [
        sum(w * v for w, v in zip(fractions, column, strict=True)) / total for column in columns
    ]
    centred = [[v - mean for v in column] for column, mean in zip(columns, means, strict=True)]
    width = values.shape[1]
    return [
        [
            sum(w * a * b for w, a, b in zip(fractions, centred[j], centred[k], strict=True))
            for k in range(width)
        ]
        for j in range(width)
    ]


def _two_pass_bound(values: NDArray, weights: NDArray) -> NDArray:
    """How far the corrected two-pass Gram may sit from the exact ``S``, entry by entry.

    Pass one puts the anchor within ``L = u |m| + gamma_{n+2} sum |w| |x -
    x_ref| / |sum w|`` of the exact mean ``m`` (as ``_pair_gram_bound``).
    With ``D = |x - m|`` and ``A = D + L``, the rows ``fl(x - a)`` carry
    ``R = u A``; centred, the perturbation is at most ``F = R + sum |w| R /
    |sum w|``, which moves the exact centred Gram by ``sum |w| (D_j F_k + F_j
    D_k + F_j F_k)``.  The pass's dot products err by ``gamma_{n+3} sum |w|
    (A + R)_j (A + R)_k`` and ``e`` by ``E_err = gamma_{n+3} sum |w| (A +
    R)``, against ``|e| <= E = |sum w| L + sum |w| R``.  Each of the
    correction's three terms ``e l'``, ``l e'`` and ``l l' sum w``
    (``two_pass_centred_gram``, ``l = e / sum w``) then errs from ``e e' / sum
    w`` by at most ``(E_j E_err,k + E_err,j E_k + E_err,j E_err,k + gamma_{n+3}
    (E + E_err)_j (E + E_err)_k) (1 + gamma_n) / |sum w|``, and the three
    additions and the symmetrization by ``4u`` of the terms.
    """
    n, width = values.shape
    total = sum(Fraction(w) for w in weights)
    magnitude = np.abs(weights)
    seed = values[np.flatnonzero(weights != 0.0)[0]]
    D = np.empty_like(values)
    L = np.empty(width)
    for j in range(width):
        mean = sum(Fraction(a) * Fraction(b) for a, b in zip(values[:, j], weights, strict=True))
        mean /= total
        D[:, j] = [float(abs(Fraction(a) - mean)) for a in values[:, j]]
        L[j] = _U * abs(float(mean)) + _gamma(n + 2) * float(
            np.sum(magnitude * np.abs(values[:, j] - seed[j])) / abs(float(total))
        )
    A = D + L
    R = _U * A
    F = R + (magnitude @ R) / abs(float(total))
    data = D.T @ (magnitude[:, None] * F) + F.T @ (magnitude[:, None] * D)
    data += F.T @ (magnitude[:, None] * F)
    rows = A + R
    gram = _gamma(n + 3) * (rows.T @ (magnitude[:, None] * rows))
    E = abs(float(total)) * L + magnitude @ R
    E_err = _gamma(n + 3) * (magnitude @ rows)
    correction = np.outer(E, E_err) + np.outer(E_err, E) + np.outer(E_err, E_err)
    correction += _gamma(n + 3) * np.outer(E + E_err, E + E_err)
    correction *= 3.0 * (1.0 + _gamma(n)) / abs(float(total))
    final = (
        4.0
        * _U
        * (
            rows.T @ (magnitude[:, None] * rows)
            + 3.0 * np.outer(E + E_err, E + E_err) / abs(float(total))
        )
    )
    return (data + gram + correction + final) * (1.0 + 8.0 * _gamma(n + 4))


def _split_design(n: int, shift: float, *, tilt: bool):
    """An even-integer column beside a 120-level categorical, above the raw rungs' crossovers.

    The working weights spread over ``[0.5, 2]``.  With ``tilt`` the first row
    sits ``2e9`` from the rest and carries no working weight, so the working
    mean sits ``2e9 / n``, about 4e4 working standard deviations, from the
    unit-weighted mean: ``kappa^2`` of the rows about that mean is about 2e9.
    """
    from superglm.group_matrix import CategoricalGroupMatrix

    values = shift + 2.0 * np.tile(np.arange(-4.0, 5.0), n // 9 + 1)[:n]
    W = np.random.default_rng(5).uniform(0.5, 2.0, n)
    if tilt:
        values[0] = shift + 2.0e9
        W[0] = 0.0
    groups = [
        DenseGroupMatrix(values[:, None]),
        CategoricalGroupMatrix(np.arange(n, dtype=np.intp) % 120, n_levels=120),
    ]
    return DesignMatrix(groups, n=n, p=121), W


def _split_system(dm, W, z, monkeypatch):
    """``build_centered_system`` on a design whose bounded half takes a raw rung (the split)."""
    from superglm.solvers import centered_system

    attached = []
    real_attach = centered_system._attach_dense_split

    def recording_attach(split, **kwargs):
        attached.append(split)
        return real_attach(split, **kwargs)

    monkeypatch.setattr(centered_system, "_attach_dense_split", recording_attach)
    system = centered_system.build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_split=dm.tabmat_centering_split,
        tabmat_state=centered_system.TabmatCenteringState(),
    )
    assert attached, "the bounded half took no raw rung: the split was not exercised"
    return system


@pytest.mark.parametrize("shift", [0.0, 1e8, 1e16])
def test_the_dense_block_takes_the_corrected_two_pass_at_every_offset(shift, monkeypatch):
    """The split's dense block, by the corrected two-pass, against the exact pair and exact rationals.

    The system agrees with #439's reference, centred row by row about
    ``weighted_mean_pair``'s ``(hi, lo)``, to ``5 gamma_{n+p+4}`` of the
    entry formed on absolute rows (a dense column's ``|x - hi| + |(x - hi) -
    lo|``, a bounded column's ``|x| + |m|``), and its dense diagonal meets
    ``_two_pass_bound`` against the exact rational value.  The pair it
    publishes is the exact pair's anchor and a remainder below the anchor's
    rounding.  Mutations: Björck's correction dropped (``G`` for ``G - e e' /
    sum W``), which at 1e16 leaves the anchor's rounding, a ulp of the
    offset, in the Gram; pass one taken without the working weights.
    """
    from superglm._group_matrix._group_matrix_centered import centered_gram_rhs
    from superglm.solvers.centered_system import weighted_mean_pair

    n = 9000
    dm, W = _split_design(n, shift, tilt=False)
    z = np.random.default_rng(6).normal(size=n)
    system = _split_system(dm, W, z, monkeypatch)
    sum_w = float(np.sum(W))
    mean_x, hi, lo = weighted_mean_pair(dm, W, sum_w)
    assert system.mean_hi is not None and system.mean_lo is not None
    assert system.mean_hi[0] == hi[0]
    z_centered = z - system.mean_z
    gram, rhs = centered_gram_rhs(dm=dm, W=W, mean_x=hi, z_centered=z_centered, mean_lo=lo)
    X = dm.toarray()
    magnitude = np.abs(X) + np.abs(mean_x)
    magnitude[:, 0] = np.abs(X[:, 0] - hi[0]) + np.abs((X[:, 0] - hi[0]) - lo[0])
    gamma = _gamma(n + dm.p + 4)
    assert np.all(
        np.abs(system.data_gram - gram) <= 5.0 * gamma * (magnitude.T @ (W[:, None] * magnitude))
    )
    assert np.all(
        np.abs(system.rhs - rhs) <= 5.0 * gamma * (magnitude.T @ (W * np.abs(z_centered)))
    )
    exact = _exact_centred_gram(X[:, :1], W)[0][0]
    bound = float(_two_pass_bound(X[:, :1], W)[0, 0])
    assert abs(Fraction(float(system.data_gram[0, 0])) - exact) <= Fraction(bound)


@pytest.mark.parametrize("shift", [0.0, 1e16])
@pytest.mark.parametrize("route", ["gram", "proximal"])
def test_a_working_mean_far_from_the_unweighted_mean_keeps_the_two_pass_bound(
    route, shift, monkeypatch
):
    """A working mean 4e4 working standard deviations from the rows' unit-weighted mean.

    One row 2e9 away carries no working weight.  About the rows' unweighted
    mean, ``kappa^2`` is about 2e9, and the textbook formula on rows shifted
    there would lose ``n u kappa^2``, two parts in a thousand.  The corrected
    two-pass algorithm anchors at the working mean whatever the rows, so on
    the gram route's split and in the proximal solver's block Gram the result
    meets ``_two_pass_bound`` against exact rationals, as it does untilted.
    Mutations: pass one taken without the working weights; Björck's
    correction dropped (at 1e16).
    """
    from superglm.solvers.pirls import _dense_group_centring, _dense_rows_buffer
    from superglm.types import GroupSlice

    n = 9000
    dm, W = _split_design(n, shift, tilt=True)
    if route == "gram":
        system = _split_system(dm, W, np.zeros(n), monkeypatch)
        value = float(system.data_gram[0, 0])
    else:
        groups = [GroupSlice("x", 0, 1), GroupSlice("c", 1, dm.p)]
        centring = _dense_group_centring(dm, groups, W, buffer=_dense_rows_buffer(dm, groups))
        assert centring is not None and centring[0] is not None
        value = float(centring[0][3][0, 0])
    x = dm.group_matrices[0].M[:, :1]
    exact = _exact_centred_gram(x, W)[0][0]
    assert abs(Fraction(value) - exact) <= Fraction(float(_two_pass_bound(x, W)[0, 0]))


def test_the_corrected_two_pass_meets_its_bound_against_exact_arithmetic():
    """The corrected two-pass Gram against exact rational arithmetic, at any tilt of the weights.

    Draws of 3 to 40 rows and 1 to 3 columns on grids exact at offsets 0,
    1e4, 1e8 and 1e16.  The working weights are prior weights tilted by
    ``exp(N(0, sigma))``, ``sigma`` 0.1, 1 or 3, some rows weightless, and in
    one draw in four a row 2e9 grid steps away carries no working weight, so
    the working mean sits up to ``kappa^2 ~ 1e9`` from the rows' unweighted
    mean.  Every entry of ``dense_anchor`` + ``anchored_dense_moments`` +
    ``two_pass_centred_gram`` meets ``_two_pass_bound``, which does not grow
    with that tilt, and #439's exact pair (``weighted_mean_pair`` and rows
    ``(x - hi) - lo``) meets ``_pair_gram_bound`` on the same draws.  On every
    draw the two-pass bound is within a factor two of #439's.  Mutations:
    Björck's correction dropped; pass one taken without the working weights.
    """
    from superglm._group_matrix._group_matrix_centered import centered_gram_rhs
    from superglm.solvers.centered_system import (
        anchored_dense_moments,
        dense_anchor,
        two_pass_centred_gram,
        weighted_mean_pair,
    )

    rng = np.random.default_rng(2026)
    tilted = 0
    for draw in range(240):
        n = int(rng.integers(3, 41))
        width = int(rng.integers(1, 4))
        shift = (0.0, 1e4, 1e8, 1e16)[draw % 4]
        spacing = 2.0 if shift >= 1e15 else 0.5
        values = shift + spacing * rng.integers(-20, 21, size=(n, width)).astype(np.float64)
        prior = np.exp(rng.normal(0.0, 0.5, n))
        working = prior * np.exp(rng.normal(0.0, (0.1, 1.0, 3.0)[draw % 3], n))
        working[rng.uniform(size=n) < 0.1] = 0.0
        if draw % 4 == 1:
            values[0] = shift + spacing * 2.0e9
            working[0] = 0.0
            tilted += 1
        if not np.any(working > 0.0):
            working[-1] = 1.0
        dm = DesignMatrix([DenseGroupMatrix(values)], n=n, p=width)
        sum_w = float(np.sum(working))
        anchor = dense_anchor(dm, working, sum_w)
        first, gram, _ = anchored_dense_moments(dm, working, anchor)
        centred = two_pass_centred_gram(first, gram, first / sum_w, sum_w)
        exact = _exact_centred_gram(values, working)
        bound = _two_pass_bound(values, working)
        _, hi, lo = weighted_mean_pair(dm, working, sum_w)
        pair, _ = centered_gram_rhs(dm=dm, W=working, mean_x=hi, z_centered=np.zeros(n), mean_lo=lo)
        for j in range(width):
            reference = _pair_gram_bound(values[:, j], working)
            assert bound[j, j] <= 2.0 * reference
            assert abs(Fraction(float(pair[j, j])) - exact[j][j]) <= Fraction(reference)
            for k in range(width):
                error = abs(Fraction(float(centred[j, k])) - exact[j][k])
                assert error <= Fraction(float(bound[j, k]))
    assert tilted >= 50


def test_the_proximal_rows_serve_the_block_products_of_their_outer_iteration():
    """The proximal solver's rows, formed once per outer iteration, against the chunked products.

    ``_dense_group_centring`` writes each dense group's rows ``fl(x - a)``
    into the fit's buffer in the pass that forms its Gram.  The block updates
    read them: ``X~' v = R' v - lo (1' v)`` (``_centred_group_score``) and
    ``X~ d = R d - lo' d`` (``_centred_group_step``).  Against the chunked
    products about the same pair (``dense_centred_rmatvec``,
    ``dense_centred_matvec``) they differ only in their summation order, at
    most ``2 gamma_{n+2}`` of the products formed on absolute rows; the Gram
    is the one formed without a buffer.  A two-column group at a 1e16 offset,
    where the pair's remainder is not negligible.  Mutations: ``- lo (1'v)``
    dropped from the score; ``- lo'd`` dropped from the step.
    """
    from superglm.solvers.mode_score import dense_centred_matvec, dense_centred_rmatvec
    from superglm.solvers.pirls import (
        _centred_group_score,
        _centred_group_step,
        _dense_group_centring,
        _dense_rows_buffer,
    )
    from superglm.types import GroupSlice

    rng = np.random.default_rng(458)
    n = 20000  # three chunks of the chunked products
    values = 1e16 + 2.0 * rng.integers(-50, 51, size=(n, 2)).astype(np.float64)
    dm = DesignMatrix([DenseGroupMatrix(values)], n=n, p=2)
    groups = [GroupSlice("x", 0, 2)]
    W = rng.uniform(0.2, 3.0, n)
    buffered = _dense_group_centring(dm, groups, W, buffer=_dense_rows_buffer(dm, groups))
    formed = _dense_group_centring(dm, groups, W)
    assert buffered is not None and formed is not None
    entry = buffered[0]
    assert entry is not None and entry[5] is not None and formed[0][5] is None
    design, hi, lo, gram, _, rows = entry
    assert np.array_equal(hi, formed[0][1]) and np.array_equal(lo, formed[0][2])
    assert np.array_equal(rows, values - hi)
    assert np.all(np.abs(lo) > 0.0)
    np.testing.assert_array_equal(gram, formed[0][3])
    v = W * rng.normal(1.0, 1.0, n)
    absolute = np.abs(rows).T @ np.abs(v) + np.abs(lo) * float(np.sum(np.abs(v)))
    score = _centred_group_score(entry, v)
    reference = dense_centred_rmatvec(design, v, hi, lo)
    assert np.all(np.abs(score - reference) <= 2.0 * _gamma(n + 2) * absolute)
    d = rng.normal(size=2)
    step = _centred_group_step(entry, d)
    reference = dense_centred_matvec(design, d, hi, lo)
    row_scale = np.abs(rows) @ np.abs(d) + float(np.abs(lo) @ np.abs(d))
    assert np.all(np.abs(step - reference) <= 2.0 * _gamma(4) * row_scale)


def test_a_fit_forms_each_working_mean_once(monkeypatch):
    """Beside a dense column, every working mean comes from the pass that needs it.

    Gram REML (binomial/logit, a ``Numeric``, a spline and a 120-level
    categorical, so the bounded half takes a raw rung) and a selection fit
    (Poisson): no PIRLS iteration forms #439's separate remainder pass
    (``corrected_two_pass_pair``) or the offset of the working mean from the
    state centre (``centre_offset_mean``), and the weight derivative forms no
    pair, reading the final system's.  The pass counts per iteration are in
    the PR.  Mutations, each failing it alone: the split's remainder formed by
    ``dense_mean_pair``; the offset formed by ``centre_offset_mean``; the
    geometry summary without its pair.
    """
    import collections

    from superglm.reml import w_derivatives
    from superglm.solvers import centered_system, irls_direct

    calls: collections.Counter = collections.Counter()

    def count(module, name):
        real = getattr(module, name)

        def counted(*args, **kwargs):
            calls[name] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    count(centered_system, "corrected_two_pass_pair")
    count(irls_direct, "centre_offset_mean")
    count(w_derivatives, "dense_mean_pair")
    rng = np.random.default_rng(12)
    n = 9000
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "s": rng.uniform(size=n),
            "c": pd.Categorical(rng.integers(0, 120, n).astype(str)),
        }
    )
    eta = -0.5 + 0.3 * frame["x"] + np.sin(4.0 * frame["s"])
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    features = {"x": Numeric(), "s": Spline(k=8), "c": Categorical()}
    model = _fit_reml(SuperGLM(family="binomial", features=features), frame, y)
    assert model._reml_result.converged
    assert model._reml_profile.get("direct_backend") == "gram"
    assert sum(calls.values()) == 0, dict(calls)
    counts = rng.poisson(np.exp(-1.0 + 0.1 * frame["x"] + 0.3 * np.sin(4.0 * frame["s"])))
    selection = SuperGLM(
        family="poisson", features={"x": Numeric(), "s": Spline(k=8)}, selection_penalty=0.01
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selection.fit(frame[["x", "s"]], counts.astype(float))
    assert selection.result.converged
    assert sum(calls.values()) == 0, dict(calls)
