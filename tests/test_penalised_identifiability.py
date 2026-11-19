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


def _dense_constraint_se(model, functionals: np.ndarray) -> np.ndarray:
    """``sqrt(phi diag Z (Z'HZ)^-1 Z')`` from the dense design, ``Z`` spanning ``null(G')``.

    The fit with the constraints ``G' beta = 0`` held (``functionals`` ``G``, fitted
    coordinates); a unit functional's coordinate has variance zero there.  ``H`` as in
    ``_dense_face_se``.
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
    Z = np.linalg.svd(np.asarray(functionals).T, full_matrices=True)[2][functionals.shape[1] :].T
    covariance = Z @ np.linalg.inv(Z.T @ (X.T @ X + S) @ Z) @ Z.T
    return np.sqrt(np.maximum(np.diag(covariance), 0.0) * model.result.phi)


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


def test_a_level_at_one_value_holds_its_slope_alone():
    """Level E sits at one ``x``: its by-level spline's unpenalised part is aliased with ``f[E]``.

    The by-level block's penalty leaves its linear part free, and with one distinct ``x`` that
    part is ``f[E]``'s dummy; the later term absorbs it.  The published constraint is that part
    of ``x:f[E]`` in its own raw basis (``identifiability.ConstructionColumns``), so no
    coordinate is blank: ``f[E]`` and the block's coefficients have the standard errors of the
    dense fit with that constraint held.  Master blanked ``f[E]`` and all seven of the level's
    coefficients; ``1d18eaf47`` blanked the block's last whitened coordinate, a choice a rotation
    of the whitening moved (``test_a_rotation_of_a_level_spline_leaves_its_constraint``).
    """
    X = pd.DataFrame({"x": np.where(LEVEL == "E", 5.0, X_SPREAD), "f": LEVEL})
    model = _fit(
        {"x": Spline(n_knots=6), "f": Categorical()},
        X,
        Y,
        interactions=[("x", "f")],
        method="fit_reml",
    )
    assert _blank(model, X, Y) == {}
    record = model._fit_inference_info["identifiability"]
    (alias,) = record.aliases
    assert alias.coordinate is None and alias.term == "x:f[E]'s unpenalised part"
    held = np.flatnonzero(record.gauge.functionals[:, 0])
    group = next(g for g in model._groups if g.name == "x:f[E]")
    assert np.all((held >= group.start) & (held < group.end))
    oracle = _dense_constraint_se(model, record.gauge.functionals)
    se = model.metrics(X, Y).coefficient_se
    level = next(g for g in model._groups if g.name == "f")
    np.testing.assert_allclose(se["f"], oracle[level.sl], rtol=1e-8)
    np.testing.assert_allclose(se["x:f[E]"], oracle[group.sl], rtol=1e-6)
    assert (
        "x:f[E]'s unpenalised part is aliased with f: level E of f has 1 distinct value of x."
        in (" ".join(str(model.summary()).split()))
    )


def _republish(model, decomposition) -> None:
    """Publish ``model`` again as if its solver had decomposed its Hessian as ``decomposition``."""
    from superglm.model import runtime_canonicalize

    solver = model._solver_pirls_result()
    record = dataclasses.replace(solver.rank_info, augmented=decomposition, identifiability=None)
    model._solver_result = dataclasses.replace(solver, rank_info=record)
    for name in ("_fit_inference_info", "_coef_covariance", "_group_edf"):
        model.__dict__.pop(name, None)
    model._summary_cache = None
    model._fit_metrics_cache = None
    runtime_canonicalize.canonicalize_fitted_model(model)


def test_a_level_at_one_value_of_a_numeric_by_level_takes_one_decision_on_any_kernel():
    """``x:f[E] = 5 f[E]``: one blank, one SE and one EDF whether the solver kept the alias or not.

    The formed Gram's eigenvalue along the alias (``2e-14`` of the largest) passes the dense
    solver's certification band on one BLAS kernel and falls below its cut on another.  Both
    events are forced here on the fit's own formed Hessian: kept (a ridge of ``1e-13`` of the
    diagonal) and truncated (a rank cut of ``1e-8``).  Either way the owner certifies only what
    clears the Gram's formation bound, reads the rest from the rows, and publishes ``x:f[E]``
    blank, ``f[E]`` with the SE of the fit without that coordinate and EDF 9, never the kept
    split's SE of rounding or its EDF of ``9.988``.  At ``1d18eaf47`` the kept decomposition
    blanked ``f[E]`` too, with EDF ``9.988``.
    """
    from superglm.solvers.rank import SHARED_RANK_POLICY, decompose_gram

    X = pd.DataFrame({"x": np.where(LEVEL == "E", 5.0, X_SPREAD), "f": LEVEL})
    model = _fit(
        {"x": Numeric(), "f": Categorical()},
        X,
        Y,
        interactions=[("x", "f")],
        method="fit_reml",
    )
    # unpenalised, and a Gamma log link's Fisher weights are one: ``H = X_c'X_c``
    n = model._dm.n
    design = np.hstack(
        [np.asarray(m.toarray(), float).reshape(n, -1) for m in model._dm.group_matrices]
    )
    centred = design - design.mean(axis=0)
    hessian = centred.T @ centred
    kept = decompose_gram(hessian + 1e-13 * np.diag(np.diag(hessian)))
    truncated = decompose_gram(
        hessian, policy=dataclasses.replace(SHARED_RANK_POLICY, gram_rcond=1e-8)
    )
    assert kept.rank == kept.width and truncated.rank == truncated.width - 1
    alone = _dense_face_se(model, [model._dm.p - 1])
    level = next(g for g in model._groups if g.name == "f")
    for decomposition in (kept, truncated):
        _republish(model, decomposition)
        assert _blank(model, X, Y) == {"x:f": [3]}
        np.testing.assert_allclose(
            model.metrics(X, Y).coefficient_se["f"], alone[level.sl], rtol=1e-8
        )
        assert model.result.effective_df == pytest.approx(9.0, abs=1e-8)
        assert model.result.beta[-1] == 0.0
        assert "x:f[E] is aliased with f: level E of f has 1 distinct value of x." in str(
            model.summary()
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


def test_a_rotation_of_a_level_spline_leaves_its_constraint_where_it_was(monkeypatch):
    """Level E at one ``x``: a change of every smooth's coordinates moves no output.

    Each smooth's whitening is turned by a rotation and rescaled (an invertible, not orthogonal,
    change of its coordinates, as a different smoothing parameter or weighting of the SSP map
    makes).  The published constraint holds ``x:f[E]``'s unpenalised part in its raw basis,
    which no whitening moves, so ``f[E]``'s coefficient and SE, every prediction at E (on its one
    training value and off it) and the blanks stand under each change (``kappa u`` ~ 1e-13 on
    this fit; rtol 1e-8).  At ``1d18eaf47`` the gauge held the block's last whitened coordinate,
    and a rotation alone moved ``f[E]``'s SE from 1.824 to 80.37 and 1.062 (Astra's review).
    Mutation: the constraint in the solver's coordinates (orthogonal projection there) survives
    a rotation and fails the rescaling.
    """
    X = pd.DataFrame({"x": np.where(LEVEL == "E", 5.0, X_SPREAD), "f": LEVEL})
    grid = pd.DataFrame({"x": [4.0, 5.0, 6.0], "f": ["E"] * 3})
    whitening = dm_builder._whitening_map

    def fit(seed):
        turns = np.random.default_rng(seed)

        def rotated(G, omega, lam2):
            result = whitening(G, omega, lam2)
            if seed is None:
                return result
            width = result.shape[1]
            turn = np.linalg.qr(turns.normal(size=(width, width)))[0]
            return result @ (turn * np.linspace(0.5, 2.0, width))

        with monkeypatch.context() as patch:
            patch.setattr(dm_builder, "_whitening_map", rotated)
            return _fit(
                {"x": Spline(n_knots=6), "f": Categorical()}, X, Y, interactions=[("x", "f")]
            )

    plain = fit(None)
    level = next(g for g in plain._groups if g.name == "f")
    reference = plain.metrics(X, Y).coefficient_se["f"]
    for seed in (7, 21):
        turned = fit(seed)
        main = next(g for g in plain._groups if g.name == "x")
        assert not np.allclose(turned.result.beta[main.sl], plain.result.beta[main.sl])
        assert _blank(turned, X, Y) == _blank(plain, X, Y) == {}
        np.testing.assert_allclose(turned.metrics(X, Y).coefficient_se["f"], reference, rtol=1e-8)
        np.testing.assert_allclose(
            turned.result.beta[level.sl], plain.result.beta[level.sl], rtol=1e-8, atol=1e-12
        )
        np.testing.assert_allclose(turned.predict(grid), plain.predict(grid), rtol=1e-8)


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


# ------------------------------------------------------------ what the rows read
def _row_passes(monkeypatch) -> list[int]:
    """The width of every row pass the owner makes (``grouped_projected_factor``)."""
    import superglm.solvers.centered_system as centered_system

    widths: list[int] = []
    original = centered_system.grouped_projected_factor

    def counted(dm, W, basis, **kwargs):
        widths.append(int(np.shape(basis)[1]))
        return original(dm, W, basis, **kwargs)

    monkeypatch.setattr(centered_system, "grouped_projected_factor", counted)
    return widths


def _redecide(model):
    """The fit's record decided again from its retained state, as the fit decides it."""
    from superglm.model import state_ops

    W = state_ops._solver_space_working_weights(model)
    state = state_ops.retained_linear_state(model)
    if state is not None:
        object.__setattr__(state, "identifiability", None)
        return state_ops._structured_identifiability(model, state, W, publish=True)
    rank_info = model.result.rank_info
    object.__setattr__(rank_info, "identifiability", None)
    design, groups = state_ops._rank_active_state(model, rank_info)
    gram = state_ops._rank_centered_data_gram(design, W)
    return state_ops._record_identifiability(
        model, rank_info, design, groups, W, gram, publish=True
    )


def test_a_structured_fit_reads_its_truncated_directions_alone(monkeypatch):
    """Five one-row ``sz`` levels among 60: the rows are read on the factor's 4 truncated directions.

    The factor certified its kept pivots, so its truncated directions are the only undecided
    ones; the weighted rows put each under its rounding bound and so does the design at unit
    weight, so they are ``null(H)``: two passes of 4 columns, and construction columns only for
    the terms they reach.  At ``633289397`` the owner formed the Gram of every level's line and
    factored ``X_F`` whole (120 columns), which took 54 s and 1.4 GiB at 1000 levels.
    """
    from superglm import FactorSmooth, LambdaPolicy
    from superglm.model import state_ops

    rows = np.random.default_rng(1)
    K, n = 60, 2400
    g = rows.integers(5, K, n)
    g[:5] = np.arange(5)
    x = rows.uniform(size=n)
    frame = pd.DataFrame({"x": x, "z": rows.normal(size=n), "g": [f"g{v:03d}" for v in g]})
    y = np.sin(3 * x) + 0.3 * np.sin(5 * x + g % 7) + 0.1 * frame["z"] + rows.normal(0, 0.3, n)
    model = SuperGLM(
        family="gaussian",
        features={
            "z": Numeric(),
            "x": Spline(kind="ps", n_knots=6, lambda_policy=LambdaPolicy.fixed(1.0)),
        },
        interactions=[
            FactorSmooth(
                "x", group="g", basis="sz", k=6, lambda_policy={"wiggle": LambdaPolicy.fixed(0.7)}
            )
        ],
        selection_penalty=0,
        direct_solve="structured",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    truncated = state_ops.retained_linear_state(model).profiled_factor.null_directions().shape[1]
    assert truncated == 4

    formed: list[str] = []
    construction = state_ops._raw_construction

    def counted(null, blocks, group, matrix):
        formed.append(group.name)
        return construction(null, blocks, group, matrix)

    monkeypatch.setattr(state_ops, "_raw_construction", counted)
    widths = _row_passes(monkeypatch)
    record = _redecide(model)
    assert widths == [truncated, truncated] and record.row_directions == truncated
    assert len(record.aliases) == truncated and record.gauge is not None
    assert set(formed) <= {"x", "x:g:sz"}


def test_a_dense_alias_reads_the_grams_undecided_direction_alone(monkeypatch):
    """``x2 = 2 x1`` on the dense solver: one direction read from the rows, not ``X_F`` whole.

    The decomposition certifies everything but the direction it truncated; that one is read
    from the rows, weighted and at unit weight, falls under its rounding bound both times, and
    is the alias.  At ``633289397`` the owner factored all of ``X_F`` from the rows.
    """
    X = pd.DataFrame({"x": X_SPREAD, "x1": X1, "x2": 2.0 * X1})
    model = _fit({"x": Spline(n_knots=6), "x1": Numeric(), "x2": Numeric()}, X, Y)
    widths = _row_passes(monkeypatch)
    record = _redecide(model)
    assert widths == [1, 1] and record.row_directions == 1
    assert [alias.term for alias in record.aliases] == ["x2"]


def test_a_near_alias_is_fitted_from_its_undecided_direction(monkeypatch):
    """``z = x + 1e-8 h``: the Gram cannot tell its smallest direction from zero; the rows can.

    The Jacobi-scaled design has singular value ``7e-9`` there, its Gram ``5e-17``, under the
    Gram's formation bound, so the dense solver truncates it; read from the rows it clears the
    rounding bound (``7e-13``), so the penalised fit identifies ``z`` and the fit takes the step
    the rows give along it (``identifiability.WeakDirections``).  Coefficients and SEs are the
    least-squares fit's, from a QR of ``[1, x, z]`` (``numpy.linalg.lstsq``; ``kappa u`` ~ 1e-8,
    rtol 1e-6), and one direction is read.  At ``1d18eaf47`` ``z`` was blank as fixed by the
    solver's rank decision, and ``x`` showed the SE of the fit without ``z``, eight orders too
    small.  Mutation: skipping the step (``WeakDirections.step`` zero) leaves ``z`` at zero.
    """
    draws = np.random.default_rng(11).standard_normal((8, 3000))
    h, x = draws[4], draws[5]
    frame = pd.DataFrame({"x": x, "z": x + 1e-8 * h})
    y = 1.0 + 0.5 * x + 3e-6 * h + 0.5 * draws[7]
    model = SuperGLM(family="gaussian", features={"x": Numeric(), "z": Numeric()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, y)
    design = np.column_stack((np.ones_like(x), frame["x"], frame["z"]))
    coefficients, residual, _, _ = np.linalg.lstsq(design, y, rcond=None)
    r = np.linalg.qr(design)[1]
    inverse = np.linalg.inv(r)
    scale = float(residual[0]) / (len(y) - 3)
    se = np.sqrt(scale * np.sum(inverse * inverse, axis=1))
    beta = np.r_[model.result.intercept, model.result.beta]
    np.testing.assert_allclose(beta, coefficients, rtol=1e-6)
    metrics = model.metrics(frame, y)
    ours = np.r_[metrics.intercept_se, metrics.coefficient_se["x"], metrics.coefficient_se["z"]]
    np.testing.assert_allclose(ours, se, rtol=1e-6)
    published = model._fit_inference_info["identifiability"]
    assert published.aliases == () and published.weak is not None and published.weak.fitted
    widths = _row_passes(monkeypatch)
    record = _redecide(model)
    assert widths == [1] and record.row_directions == 1


# ------------------------------------------------------------------ one gauge
def test_fit_at_its_default_lambda_publishes_the_owners_constraint():
    """Level E at one ``x``, ``fit()`` at lambda 0.1: the slope part is held, ``f[E]`` reported.

    The dense solver's rank decision here fixes ``f[E]`` at zero, not the later term's slope
    (stage 1, follow-up 3); the record's gauge moves the coefficients along the alias onto the
    later term's constraint (``identifiability.Gauge``), so ``f[E]`` reports its coefficient and
    the SE of the fit with that constraint held, as ``fit_reml`` does, and no fitted value
    moves.  Mutation: no gauge published blanks ``f[E]`` and reports it at zero.
    """
    X = pd.DataFrame({"x": np.where(LEVEL == "E", 5.0, X_SPREAD), "f": LEVEL})
    model = _fit({"x": Spline(n_knots=6), "f": Categorical()}, X, Y, interactions=[("x", "f")])
    record = model._fit_inference_info["identifiability"]
    assert record.gauge is not None
    assert _blank(model, X, Y) == {}
    level = next(g for g in model._groups if g.name == "f")
    functionals = record.gauge.functionals
    assert abs(float(functionals[:, 0] @ model.result.beta)) <= 1e-12
    assert model.result.beta[level.start + 3] != 0.0
    np.testing.assert_allclose(
        model.metrics(X, Y).coefficient_se["f"],
        _dense_constraint_se(model, functionals)[level.sl],
        rtol=1e-8,
    )
    np.testing.assert_allclose(model.predict(X), model._fit_mu, rtol=1e-12)


@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_a_duplicate_beside_a_factor_smooth_publishes_one_gauge(basis, monkeypatch):
    """``x2 = x1`` beside an ``fs`` or ``sz`` term: both backends blank ``x2`` alone.

    The structured factor truncates the duplicate's direction and splits the effect evenly;
    it publishes the dense solver's zero on ``x2`` instead, with the same coefficients, SEs and
    footnote, and predicts what it predicted before (``assert_one_gauge``).  Mutation: no gauge
    leaves ``x1`` blank at half the effect.
    """
    from superglm import FactorSmooth, LambdaPolicy
    from tests._identifiability_parity import assert_one_gauge, without_gauge

    rows = np.random.default_rng(4)
    n, K = 3000, 25
    g = rows.integers(0, K, n)
    x, x1 = rows.uniform(size=n), rows.normal(size=n)
    frame = pd.DataFrame({"x": x, "x1": x1, "x2": x1.copy(), "g": [f"g{v:02d}" for v in g]})
    y = np.sin(3 * x) + 0.2 * x1 + 0.2 * np.sin(4 * x + g % 5) + rows.normal(0, 0.3, n)

    def fit(solve):
        model = SuperGLM(
            family="gaussian",
            features={
                "x1": Numeric(),
                "x2": Numeric(),
                "x": Spline(kind="ps", n_knots=6, lambda_policy=LambdaPolicy.fixed(1.0)),
            },
            interactions=[
                FactorSmooth(
                    "x",
                    group="g",
                    basis=basis,
                    k=5,
                    lambda_policy={"wiggle": LambdaPolicy.fixed(0.7)},
                )
            ],
            selection_penalty=0,
            direct_solve=solve,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit_reml(frame, y)

    structured, dense = fit("structured"), fit("gram")
    assert structured._fit_inference_info.get("structured_covariance")
    before = [
        without_gauge(monkeypatch, lambda solve=solve: fit(solve))
        for solve in ("structured", "gram")
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = [model.metrics(frame, y) for model in (structured, dense)]
    assert_one_gauge(*metrics, frame, before=before, rtol=1e-8)
    assert _blank(structured, frame, y) == {"x2": [0]}


# ----------------------------------------------------------- the mgcv oracle, stage 6a
def _near_duplicate_frame(d: float):
    """Opus's design gate probe: ``x2 = x1 + d h`` beside a spline, Gaussian."""
    rng = np.random.default_rng(1)
    n = 2000
    z, x1, h = rng.uniform(size=n), rng.normal(size=n), rng.normal(size=n)
    y = 1 + x1 + np.sin(2 * np.pi * z) + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x1 + d * h, "z": z}), y


def test_a_near_duplicate_column_is_fitted_as_mgcv_fits_it():
    """``x2 = x1 + 1e-8 h``: the penalised fit identifies the pair, as mgcv's does (rank 12/12).

    The fixture is mgcv 1.9-3 run on superglm's own design and penalty at the pinned lambda
    (``paraPen``).  The dense solver truncates the pair's difference (its Gram eigenvalue is
    below the formation bound); the rows identify it, so the fit takes their step along it and
    reports its variance (``identifiability.WeakDirections``): no blank, the coefficients near
    ``+-9e5``, every SE^2 / phi mgcv's ``Vp / sig2`` and the EDF and scale mgcv's (rtol 1e-6:
    the direction's curvature is read from the rows to ``kappa u`` ~ 1e-8).  At ``1d18eaf47``
    ``x2`` was blank and ``x1`` showed SE 0.0224 against mgcv's 2.2e6.
    """
    fixture = json.loads((FIXTURES / "mgcv_near_duplicate_and_aliased_columns.json").read_text())
    oracle = fixture["near_duplicate"]
    X, y = _near_duplicate_frame(oracle["d"])
    model = SuperGLM(
        family="gaussian", features={"x1": Numeric(), "x2": Numeric(), "z": Spline(n_knots=8)}
    )
    model.lambda2 = oracle["lambda"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, tol=1e-12, max_iter=500)
    design = np.hstack(
        [np.asarray(m.toarray(), float).reshape(model._dm.n, -1) for m in model._dm.group_matrices]
    )
    assert float(np.sum(np.abs(design))) == pytest.approx(oracle["design_abs_sum"], rel=1e-12)
    assert oracle["rank"] == oracle["width"]
    assert _blank(model, X, y) == {}
    metrics = model.metrics(X, y)
    se = np.concatenate(
        [[metrics.intercept_se]]
        + [np.atleast_1d(metrics.coefficient_se[g.name]) for g in model._groups]
    )
    vp = np.diag(np.asarray(oracle["Vp"])) / oracle["sig2"]
    np.testing.assert_allclose(se**2 / model.result.phi, vp, rtol=1e-6)
    beta = np.r_[model.result.intercept, model.result.beta]
    mgcv = np.asarray(oracle["coef"])
    assert np.max(np.abs(beta - mgcv)) <= 1e-6 * np.max(np.abs(mgcv))
    assert model.result.phi == pytest.approx(oracle["sig2"], rel=1e-6)
    assert model.result.effective_df == pytest.approx(oracle["edf"], rel=1e-6)


def _three_column_frame(d: float):
    """Sol's review case: ``b = a + d u`` on integer columns, an exact alias at float64."""
    rng = np.random.default_rng(42)
    n = 500
    a, u = rng.integers(-128, 129, (2, n)).astype(float)
    v = rng.normal(size=n)
    y = 2 + 0.01 * a + 0.01 * u + v * 0.1
    return pd.DataFrame({"a": a, "b": a + d * u, "u": u}), y, rng


@pytest.mark.parametrize(("d", "held"), [(2.11e-8, "b"), (1e-3, "u")])
def test_an_exact_three_column_alias_takes_one_constraint_in_any_row_order(d, held):
    """``b = a + d u``: the latest column is held unless that is ill-conditioned, in any order.

    Holding ``u`` amplifies the kept columns' conditioning by ``sqrt(2) / d``: at ``d = 1e-3``
    that is ``1.4e3`` and ``u`` is held, as ``lm`` does; at ``d = 2.11e-8`` it is ``6.7e7``,
    past ``1e7``, and the conditioning-first choice holds ``b``, as ``lm`` does.  The choice
    reads the construction columns alone, so two row permutations give the same blank,
    coefficients and SEs, which are mgcv's (``gam(y ~ a + b + u)``, a minimum-norm split, moved
    along the alias ``(1, -1, d)`` to the held column's zero; rtol 1e-6).  At ``1d18eaf47`` one
    permutation at ``d = 2.11e-8`` held ``u`` and printed SE 2886 for ``a`` against 6.2e-5.
    """
    fixture = json.loads((FIXTURES / "mgcv_near_duplicate_and_aliased_columns.json").read_text())
    oracle = fixture["three_column"][repr(d)]
    X, y, rng = _three_column_frame(d)
    columns = ["a", "b", "u"]
    alias = np.array([0.0, 1.0, -1.0, d])
    position = 1 + columns.index(held)
    coefficients = np.asarray(oracle["coef"])
    vp = np.asarray(oracle["Vp"]) / oracle["sig2"]
    gauge = np.eye(4) - np.outer(alias, np.eye(4)[position]) / alias[position]
    reference, covariance = gauge @ coefficients, gauge @ vp @ gauge.T
    for order in (np.arange(len(y)), rng.permutation(len(y)), rng.permutation(len(y))):
        frame, response = X.iloc[order].reset_index(drop=True), y[order]
        model = SuperGLM(family="gaussian", features={c: Numeric() for c in columns})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(frame, response)
        assert _blank(model, frame, response) == {held: [0]}
        metrics = model.metrics(frame, response)
        se = np.r_[metrics.intercept_se, [metrics.coefficient_se[c][0] for c in columns]]
        shown = np.arange(4) != position
        np.testing.assert_allclose(
            se[shown] ** 2 / model.result.phi, np.diag(covariance)[shown], rtol=1e-6
        )
        beta = np.r_[model.result.intercept, model.result.beta]
        assert beta[position] == 0.0
        assert np.max(np.abs(beta - reference)) <= 1e-6 * np.max(np.abs(reference))


# ------------------------------------------------------------------------- cost
@pytest.mark.parametrize("case", ["zero_claims", "near_duplicate"])
def test_a_large_sz_fit_reads_only_its_undecided_directions(case, monkeypatch):
    """Zero-claim ``sz`` levels and a near-duplicate beside an ``sz`` term: a few rows read.

    Five Poisson levels with no claims: their working weights vanish where the fit stopped, so
    the weighted rows tell a direction's curvature from zero or not depending on that stop (a
    separation, which the exact test of stage 6b decides); the design identifies every one, so
    none is an alias: each is held as not resolved at float64 or reported with its curvature,
    never moved along, with no coordinate blank.  ``zz = z + 1e-8 h``:
    the rows identify the pair and fit it.  Either way the owner reads only the factor's
    truncated directions (a pass of at most two columns, twice) and forms one small construction
    map per term they reach.  At ``1d18eaf47`` both fell back to factoring the whole
    restricted design from the rows (2003 columns at 1000 levels, 1.6 GiB; the zero-claim case
    then refused the gauge and blanked 2306 coefficients).
    """
    from superglm import FactorSmooth, LambdaPolicy
    from superglm.model import state_ops

    K = 200
    if case == "zero_claims":
        rows = np.random.default_rng(2)
        n = 20 * K
        g = rows.integers(0, K, n)
        x = rows.uniform(size=n)
        frame = pd.DataFrame({"x": x, "z": rows.normal(size=n), "g": [f"g{v:04d}" for v in g]})
        mu = np.exp(-1.0 + 0.5 * np.sin(3 * x) + 0.2 * np.sin(5 * x + g % 7))
        y = rows.poisson(mu).astype(float)
        y[g < 5] = 0.0
        model = SuperGLM(
            family="poisson",
            features={"z": Numeric(), "x": Spline(kind="ps", n_knots=6)},
            interactions=[FactorSmooth("x", group="g", basis="sz", k=6)],
            selection_penalty=0,
            direct_solve="structured",
        )
    else:
        rows = np.random.default_rng(1)
        n = K * 24
        g = rows.integers(0, K, n)
        x = rows.uniform(size=n)
        frame = pd.DataFrame({"x": x, "z": rows.normal(size=n), "g": [f"g{v:04d}" for v in g]})
        y = np.sin(3 * x) + 0.3 * np.sin(5 * x + g % 7) + 0.1 * frame["z"] + rows.normal(0, 0.3, n)
        frame["zz"] = frame["z"] + 1e-8 * rows.normal(size=n)
        model = SuperGLM(
            family="gaussian",
            features={
                "z": Numeric(),
                "zz": Numeric(),
                "x": Spline(kind="ps", n_knots=6, lambda_policy=LambdaPolicy.fixed(1.0)),
            },
            interactions=[
                FactorSmooth(
                    "x",
                    group="g",
                    basis="sz",
                    k=6,
                    lambda_policy={"wiggle": LambdaPolicy.fixed(0.7)},
                )
            ],
            selection_penalty=0,
            direct_solve="structured",
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    assert not any(np.any(np.isnan(v)) for v in model.metrics(frame, y).coefficient_se.values())
    published = model._fit_inference_info["identifiability"]
    truncated = state_ops.retained_linear_state(model).profiled_factor.null_directions().shape[1]
    assert 0 < truncated <= 5
    formed: list[str] = []
    construction = state_ops._raw_construction

    def counted(null, blocks, group, matrix):
        formed.append(group.name)
        return construction(null, blocks, group, matrix)

    monkeypatch.setattr(state_ops, "_raw_construction", counted)
    widths = _row_passes(monkeypatch)
    record = _redecide(model)
    assert widths and all(width <= truncated for width in widths) and len(widths) <= 2
    assert record.row_directions == truncated
    # one small map per term the directions reach, never one per level
    assert len(formed) == len(set(formed)) <= 3
    if case == "zero_claims":
        assert {a.source for a in published.aliases} <= {"float64"}
        assert published.weak is None or not published.weak.fitted
        assert published.gauge is None or not np.any(published.gauge.moved)
    else:
        assert published.aliases == () and published.weak is not None and published.weak.fitted
