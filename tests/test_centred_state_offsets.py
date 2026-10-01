"""The centred state at large column offsets (issue #430): no raw reconstruction left.

A ``Numeric`` column far from zero (an epoch time, an ID) makes the raw intercept
``alpha - c' beta`` cancel against ``X beta`` (one-engine design §3.8).  Each test
translates such a column exactly (its values on a grid the offset represents)
and checks that the fit, its REML, its certificates and its diagnostics read the
same model as at no offset.  Every test fails on the code before #430 and
under the mutation its docstring names.

Bounds derive from dimensions, ``eps``, the REML stopping rule and the fits'
own predictors (``mode_score.linear_predictor``, which reads the centred
state), never from what passed locally.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

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


# ------------------------------------------- 7. the gram path's intercept
@pytest.mark.parametrize("direct_solve", ["gram", "qr"])
@pytest.mark.parametrize("family", ["poisson", "gamma"])
def test_gram_and_qr_intercepts_read_the_working_means_about_the_centre(family, direct_solve):
    """The gram and QR iterations' ``alpha = mean_z - (mean_x - c)' beta`` (item 7).

    ``mean_x`` of a column at ``s`` rounds at ``u s``, so ``mean_x - c`` erred
    by ``u s |beta|`` in every row's eta (2.7e-12 at 1e6; 4e-8 here at 1e10):
    a non-Gaussian fit has no compensated intercept to absorb it.  The column
    is now differenced from ``c`` before its weighted sum
    (``mode_score.centre_offset_mean``), and the translated fit runs the same
    centred arithmetic, so the two predictors agree to the forward error of
    the solves, ``gamma_n kappa(H) max|eta|`` with ``kappa`` the centred
    Hessian's condition; QR is backward stable too, so one bound covers both
    branches.  Mutation: ``centered.mean_x - _state_center`` in ``irls_direct``'s
    gram branch, or in its QR branch.
    """
    rng = np.random.default_rng(7)
    n = 3000
    z = np.round(rng.normal(size=n) * 2.0**19) / 2.0**19  # exact beside 1e10
    cat = rng.integers(0, 5, n)
    eta = 0.3 + 0.3 * z + 0.2 * (cat - 2)
    if family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        y = rng.gamma(3.0, np.exp(eta) / 3.0)
    labels = np.array([f"c{c}" for c in cat], dtype=object)
    fits = {}
    for shift in (0.0, 1e10):
        model = SuperGLM(
            family=family,
            features={"x": Numeric(), "cat": Categorical()},
            selection_penalty=0.0,
            direct_solve=direct_solve,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits[shift] = model.fit(pd.DataFrame({"x": shift + z, "cat": labels}), y)
    base, shifted = fits[0.0], fits[1e10]
    solver = base._solver_pirls_result()
    assert shifted._solver_pirls_result().n_iter == solver.n_iter
    eta_base = linear_predictor(base._dm, solver, None)
    eta_shifted = linear_predictor(shifted._dm, shifted._solver_pirls_result(), None)
    weights = np.exp(eta_base) if family == "poisson" else np.ones(n)
    design = np.column_stack([np.ones(n), np.asarray(base._dm.toarray(), dtype=np.float64)])
    kappa = float(np.linalg.cond(design.T @ (weights[:, None] * design)))
    bound = _gamma(n) * kappa * float(np.max(np.abs(eta_base)))
    np.testing.assert_array_less(np.abs(eta_shifted - eta_base), bound)


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
