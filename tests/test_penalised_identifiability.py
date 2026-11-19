"""Standard errors from the penalised fit, blank only where the fit cannot identify a coefficient.

``solvers.identifiability`` decides ``null(H) = N_S null(X N_S)`` for the penalised Hessian
``H``: a coefficient's standard error is blank only on a coordinate that neither the data nor
any penalty reaches, with a footnote naming what it is aliased with.  Master blanked every
coefficient the data alone left undetermined; each structural case below fails there.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import superglm
import superglm.dm_builder as dm_builder
from superglm import Categorical, Numeric, RandomEffect, Spline, SuperGLM
from superglm.model.fit_state import fitted_lambda2
from superglm.reml.penalty_algebra import build_penalty_matrix

FIXTURES = Path(__file__).parent / "fixtures"


def _fit(features, X, y, *, interactions=(), method="fit", **kwargs):
    model = SuperGLM(
        family=superglm.families.gamma(),
        link="log",
        features=features,
        interactions=list(interactions),
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(model, method)(X, y)
    return model


def _blank(model, X, y) -> dict[str, list[int]]:
    """Each term's coordinates without a standard error in ``metrics().coefficient_se``."""
    se = model.metrics(X, y).coefficient_se
    return {
        name: np.flatnonzero(np.isnan(values)).tolist()
        for name, values in se.items()
        if np.any(np.isnan(values))
    }


def _rows(model) -> dict[str, float]:
    return {row.name: row.se for row in model.summary()._coef_rows if not row.is_spline}


def _dense_face_se(model, fixed: list[int]) -> np.ndarray:
    """``sqrt(phi diag Z (Z'HZ)^-1 Z')`` from the dense design: the fit without ``fixed``.

    ``H = X_c'X_c + S`` (a Gamma log link's Fisher weights are one), ``X_c`` centred on the
    intercept, ``S`` the fitted penalty, and ``Z`` drops the coordinates ``fixed``.
    """
    n, p = model._dm.n, model._dm.p
    X = np.hstack([np.asarray(m.toarray(), float).reshape(n, -1) for m in model._dm.group_matrices])
    X = X - X.mean(axis=0)
    S = build_penalty_matrix(
        list(model._dm.group_matrices),
        list(model._groups),
        fitted_lambda2(model),
        p,
        reml_penalties=getattr(model, "_reml_penalties", None),
    )
    kept = np.setdiff1d(np.arange(p), fixed)
    covariance = np.full(p, np.nan)
    covariance[kept] = np.diag(np.linalg.inv((X.T @ X + S)[np.ix_(kept, kept)]))
    return np.sqrt(covariance * model.result.phi)


rng = np.random.default_rng(7)
N = 2000
X_SPREAD = rng.uniform(0, 10, N)
X1 = rng.normal(size=N)
LEVEL = rng.choice(list("ABCDE"), N)
Y = rng.gamma(2.0, np.exp(0.1 * np.sin(X_SPREAD)) / 2.0)


# ------------------------------------------------------------- the mgcv oracle
def test_standard_errors_are_mgcvs_bayesian_ones_on_a_factor_by_spline_model():
    """A spline by a factor with a thin level: every SE is printed, and is mgcv's ``Vp``.

    The fixture is mgcv 1.9-3 run on superglm's own design and penalties at pinned smoothing
    parameters (``paraPen``), so both engines solve one penalised problem; it reports all 50
    coefficients.  Refit here at those lambdas, the covariance is the dense ``(X_c'X_c +
    S)^-1`` of the refit's design (condition about 40, so ``kappa u`` ~ 1e-14; rtol 1e-10),
    and ``SE^2 / phi`` is mgcv's ``Vp / phi`` to 1e-6: the refit's SSP whitening is formed at
    the pinned lambdas, the export's at REML's start, so its slope columns differ by up to
    2e-7, which moves their variances by up to 4e-8.  Master blanked ``f[E]`` and 18
    coefficients.
    """
    fixture = json.loads((FIXTURES / "mgcv_factor_by_spline_gamma.json").read_text())
    data = np.load(FIXTURES / "mgcv_factor_by_spline_gamma_data.npz", allow_pickle=False)
    X = pd.DataFrame({"x": data["x"], "f": data["f"].astype(str)})
    y = data["y"]
    model = SuperGLM(
        family=superglm.families.gamma(),
        link="log",
        features={"x": Spline(n_knots=8), "f": Categorical()},
        interactions=[("x", "f")],
    )
    model.lambda2 = dict(fixture["lambda_solver"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, tol=1e-12, max_iter=500)
    metrics = model.metrics(X, y)
    se = np.concatenate(
        [[metrics.intercept_se]]
        + [np.atleast_1d(metrics.coefficient_se[g.name]) for g in model._groups]
    )
    assert np.all(np.isfinite(se))
    assert all(np.isfinite(value) for value in _rows(model).values())
    np.testing.assert_allclose(se[1:], _dense_face_se(model, []), rtol=1e-10)
    phi = model.result.phi
    vp = np.asarray(fixture["mgcv_se_vp"]) ** 2 / fixture["phi"]
    np.testing.assert_allclose(se**2 / phi, vp, rtol=1e-6)
    beta = np.r_[model.result.intercept, model.result.beta]
    mgcv = np.asarray(fixture["mgcv_coef"])
    assert np.max(np.abs(beta - mgcv)) <= 1e-6 * np.max(np.abs(mgcv))
    assert model.result.effective_df == pytest.approx(fixture["measured"]["edf_mgcv"], rel=1e-8)


# ------------------------------------------------------------ structural aliases
def test_nested_factors_blank_the_later_factor_alone():
    """``g`` is a union of ``f``'s levels: ``g[P]`` is aliased; ``f`` is the fit without ``g``."""
    g = np.where(np.isin(LEVEL, ["A", "B"]), "P", "Q")
    X = pd.DataFrame({"x": X_SPREAD, "f": LEVEL, "g": g})
    features = {"x": Spline(n_knots=6), "f": Categorical(), "g": Categorical()}
    model = _fit(features, X, Y)
    alone = _fit({"x": features["x"], "f": Categorical()}, X[["x", "f"]], Y)
    assert _blank(model, X, Y) == {"g": [0]}
    rows, reference = _rows(model), _rows(alone)
    for level in ("f[A]", "f[C]", "f[D]", "f[E]"):
        assert rows[level] == pytest.approx(reference[level], rel=1e-8)
    assert "g[P] is aliased with f: each level of g is a union of levels of f." in str(
        model.summary()
    )


def test_a_duplicated_numeric_blanks_the_duplicate_alone():
    """``x2 = 2 x1``: ``x2`` is aliased at float64; ``x1``'s SE is the fit without ``x2``."""
    X = pd.DataFrame({"x": X_SPREAD, "x1": X1, "x2": 2.0 * X1})
    features = {"x": Spline(n_knots=6), "x1": Numeric(), "x2": Numeric()}
    model = _fit(features, X, Y)
    alone = _fit({"x": features["x"], "x1": Numeric()}, X[["x", "x1"]], Y)
    assert _blank(model, X, Y) == {"x2": [0]}
    assert _rows(model)["x1"] == pytest.approx(_rows(alone)["x1"], rel=1e-8)
    assert "x2 is aliased with x1: no data separates them at float64." in str(model.summary())


@pytest.mark.parametrize("case", ["nested", "duplicate"])
def test_superlss_drops_the_coefficient_superglm_blanks(case):
    """SuperGLM and SuperLSS take one alias decision on nested factors and on ``x2 = 2 x1``.

    SuperGLM's record names the later term's coordinate (``solvers.identifiability``);
    SuperLSS's terminal rank drops that coordinate, refits without it as a design alias
    (``solver._drops_design_aliases``) and publishes it at zero.
    """
    from superglm import SuperLSS, cat, s, term
    from superglm.distributional import GaussianLS

    if case == "nested":
        g = np.where(np.isin(LEVEL, ["A", "B"]), "P", "Q")
        X = pd.DataFrame({"x": X_SPREAD, "f": LEVEL, "g": g})
        features = {"x": Spline(n_knots=6), "f": Categorical(), "g": Categorical()}
        location, expected = (s("x"), cat("f"), cat("g")), {"g"}
    else:
        X = pd.DataFrame({"x": X_SPREAD, "x1": X1, "x2": 2.0 * X1})
        features = {"x": Spline(n_knots=6), "x1": Numeric(), "x2": Numeric()}
        location, expected = (s("x"), term("x1", Numeric()), term("x2", Numeric())), {"x2"}
    glm = _fit(features, X, Y)
    owner = {
        next(g.name for g in glm._groups if g.start <= alias.coordinate < g.end)
        for alias in glm._fit_inference_info["identifiability"].aliases
    }
    family = GaussianLS(scale_floor=0)
    lss = SuperLSS(family, family.location(*location), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lss.fit_reml(X, Y)
    state = lss._require_fitted().fit_state
    result = state.solver_result
    dropped = np.flatnonzero(result.terminal_rank.structural_aliases)
    terms = {
        name.split(":", 1)[1]
        for name, block in state.layout.term_slices.items()
        if any(block.start <= index < block.stop for index in dropped)
    }
    assert owner == terms == expected
    assert result.converged and np.all(result.coefficients[dropped] == 0.0)


def test_a_truncated_direction_on_a_column_of_tiny_curvature_is_read():
    """Two truncated directions, one on a column whose Jacobi scale is 1e-9 of the other's.

    A structured factor truncates a column carried by rounding-level weights beside an
    ordinary alias.  Scaled by ``sqrt(diag H)`` and read as one basis, the second direction's
    singular value falls under the share floor and its column looks determined; each
    direction is normalised first, so both are read (``fixed_coordinates``).  Mutation:
    without the normalisation coordinate 1 is reported determined.
    """
    from superglm.solvers.identifiability import TruncatedDecomposition, fixed_coordinates

    decomposition = TruncatedDecomposition(
        null_vectors=np.eye(3)[:, :2],
        rank=1,
        column_scale=np.array([1.0, 1e-9, 1.0]),
        pre_truncation_condition=1.0,
    )
    assert fixed_coordinates(decomposition).tolist() == [0, 1]


def test_a_level_at_one_value_blanks_its_slope_alone():
    """Level E sits at one ``x``: its slope coordinate is aliased with ``f[E]``, which keeps an SE.

    The by-level block's penalty leaves its linear part free, and with one distinct ``x`` that
    part is ``f[E]``'s dummy; the later term absorbs it.  ``f[E]``'s SE is the dense fit
    without that coordinate.  Master blanked ``f[E]`` and all seven of the level's coefficients.
    """
    X = pd.DataFrame({"x": np.where(LEVEL == "E", 5.0, X_SPREAD), "f": LEVEL})
    model = _fit(
        {"x": Spline(n_knots=6), "f": Categorical()},
        X,
        Y,
        interactions=[("x", "f")],
        method="fit_reml",
    )
    blank = _blank(model, X, Y)
    assert list(blank) == ["x:f[E]"] and len(blank["x:f[E]"]) == 1
    group = next(g for g in model._groups if g.name == "x:f[E]")
    fixed = group.start + blank["x:f[E]"][0]
    level = next(g for g in model._groups if g.name == "f")
    oracle = _dense_face_se(model, [fixed])
    se = model.metrics(X, Y).coefficient_se["f"]
    np.testing.assert_allclose(se, oracle[level.sl], rtol=1e-8)
    assert "x:f[E] is aliased with f: level E of f has 1 distinct value of x." in str(
        model.summary()
    )


def test_a_level_at_one_value_of_a_numeric_by_level_prints_no_se_of_rounding():
    """``x:f[E] = 5 f[E]``: the owner finds the alias whatever the solver's decomposition did.

    Whether the published decomposition zeroes ``x:f[E]`` or keeps the alias depends on the
    BLAS kernel: the formed Gram's eigenvalue there (2e-14 of the largest) passes its
    certification band on one kernel and falls below its cut on another.  Zeroed, the later
    term absorbs the alias and ``f[E]`` is the fit without it; kept, the fit's split along it
    is arbitrary and its variance rounding (SEs near 1e5), so both are blank.  Mutation:
    blanking only the decomposition's own zeros prints those SEs.
    """
    X = pd.DataFrame({"x": np.where(LEVEL == "E", 5.0, X_SPREAD), "f": LEVEL})
    model = _fit(
        {"x": Numeric(), "f": Categorical()},
        X,
        Y,
        interactions=[("x", "f")],
        method="fit_reml",
    )
    published = model.result.rank_info.augmented
    kept = published.rank == published.width
    blank = _blank(model, X, Y)
    assert blank == ({"f": [3], "x:f": [3]} if kept else {"x:f": [3]})
    reason = "level E of f has 1 distinct value of x."
    text = str(model.summary())
    assert f"x:f[E] is aliased with f: {reason}" in text
    if not kept:
        alone = _dense_face_se(model, [model._dm.p - 1])
        level = next(g for g in model._groups if g.name == "f")
        np.testing.assert_allclose(
            model.metrics(X, Y).coefficient_se["f"], alone[level.sl], rtol=1e-8
        )


def test_basis_functions_without_data_or_penalty_are_blank_with_their_count():
    """An unpenalised spline with a data gap: exactly its coordinates the gap leaves free.

    Master blanked the same three coordinates, with no footnote.
    """
    gap = np.where((X_SPREAD > 3) & (X_SPREAD < 7), X_SPREAD - 3.0, X_SPREAD)
    X = pd.DataFrame({"x": gap})
    spline = Spline(knots=np.linspace(0.5, 9.5, 19), penalty="none")
    model = _fit({"x": spline}, X, Y, spline_penalty=0.0)
    assert _blank(model, X, Y) == {"x": [8, 9, 10]}
    assert "x (3 coefficients): 3 of its basis functions have no data" in str(model.summary())


def test_a_penalised_factors_empty_level_reports_its_prior_sd():
    """A random effect's level with no rows: the ridge identifies it, ``SE = sqrt(phi / lambda)``.

    Master blanked every level of the term, the observed ones included.
    """
    X = pd.DataFrame({"x": X_SPREAD, "f": pd.Categorical(LEVEL, categories=list("ABCDEZ"))})
    model = _fit({"x": Spline(n_knots=6), "f": RandomEffect()}, X, Y, method="fit_reml")
    se = model.metrics(X, Y).coefficient_se["f"]
    assert np.all(np.isfinite(se))
    component = next(c for c in model._reml_penalties if c.group_name == "f")
    prior = np.sqrt(model.result.phi / fitted_lambda2(model)[component.name])
    assert se[-1] == pytest.approx(prior, rel=1e-8)
    assert "not identified" not in str(model.summary())


# ------------------------------------------------------------------ invariance
def test_a_rotation_of_a_spline_block_changes_no_identifiability_decision(monkeypatch):
    """An orthogonal turn of the main spline's coordinates moves no SE outside that block.

    ``null(H) = N_S null(X N_S)`` holds no coordinates, so nothing is blank either way and
    every other coefficient's SE is unchanged (rounding, ``kappa u`` ~ 1e-14; rtol 1e-9).
    Master's data-only flags moved under such a turn and blanked ``f[E]`` here.
    """
    fixture = json.loads((FIXTURES / "mgcv_factor_by_spline_gamma.json").read_text())
    data = np.load(FIXTURES / "mgcv_factor_by_spline_gamma_data.npz", allow_pickle=False)
    X = pd.DataFrame({"x": data["x"], "f": data["f"].astype(str)})
    y = data["y"]

    def fit():
        model = SuperGLM(
            family=superglm.families.gamma(),
            link="log",
            features={"x": Spline(n_knots=8), "f": Categorical()},
            interactions=[("x", "f")],
        )
        model.lambda2 = dict(fixture["lambda_solver"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y, tol=1e-12, max_iter=500)
        return model

    plain = fit()
    whitening, turned = dm_builder._whitening_map, []
    turn = np.linalg.qr(np.random.default_rng(3).normal(size=(9, 9)))[0]

    def rotated(G, omega, lam2):
        result = whitening(G, omega, lam2)
        if not turned and result.shape[1] == turn.shape[0]:
            turned.append(True)
            return result @ turn
        return result

    monkeypatch.setattr(dm_builder, "_whitening_map", rotated)
    rotated_model = fit()
    assert turned
    x = next(g for g in plain._groups if g.name == "x")
    assert not np.allclose(rotated_model.result.beta[x.sl], plain.result.beta[x.sl])
    np.testing.assert_allclose(rotated_model.predict(X), plain.predict(X), rtol=1e-9)
    se_plain, se_rotated = plain.metrics(X, y).coefficient_se, rotated_model.metrics(X, y)
    for name, values in se_plain.items():
        assert np.all(np.isfinite(values)) and np.all(np.isfinite(se_rotated.coefficient_se[name]))
        if name != "x":
            np.testing.assert_allclose(se_rotated.coefficient_se[name], values, rtol=1e-9)
    for model in (plain, rotated_model):
        assert model.result.rank_info.identifiability.aliases == ()
    assert rotated_model.result.rank_info.identifiability.fixed_effect_condition == pytest.approx(
        plain.result.rank_info.identifiability.fixed_effect_condition, rel=1e-9
    )


# ----------------------------------------------------------------- saved models
def test_a_saved_record_without_the_decision_keeps_its_flags_unless_full_rank():
    """A record from before fits recorded identifiability reads its data-only flags.

    Full rank, ``null(H) = 0`` and every coefficient is identified with no design needed; with
    a truncated Hessian it keeps the flags it was saved with, so it loses no SE it showed.
    """
    X = pd.DataFrame({"x": X_SPREAD, "x1": X1, "x2": 2.0 * X1})
    aliased = _fit({"x": Spline(n_knots=6), "x1": Numeric(), "x2": Numeric()}, X, Y)
    full = _fit({"x": Spline(n_knots=6), "x1": Numeric()}, X[["x", "x1"]], Y)
    for model in (aliased, full):
        record = model.result.rank_info
        assert record.identifiability_policy == "penalised"
    from superglm.model.state_ops import fit_data_rank

    saved = dataclasses.replace(
        fit_data_rank(aliased), identifiability_policy="data", identifiability=None
    )
    expected = np.zeros(len(saved.mean_x), dtype=bool)
    expected[saved.selected_columns] = saved.data.coefficient_estimable()
    np.testing.assert_array_equal(saved.coefficient_identified(), expected)
    assert not expected[-2:].any()
    old_full = dataclasses.replace(
        full.result.rank_info, identifiability_policy="data", identifiability=None, data=None
    )
    assert old_full.coefficient_identified().all()
