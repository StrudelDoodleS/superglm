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


U = 2.0**-53  # the unit roundoff


def _gamma(k: int) -> float:
    """``gamma_k = k u / (1 - k u)`` (Higham 2002, sec. 3.1)."""
    return k * U / (1.0 - k * U)


def _footnote(model, term: str):
    """The record's alias on ``term`` and the summary's footnote line that reports it.

    The line is one of the record's notes, and the printed summary carries it (wrapped).
    """
    (alias,) = [a for a in model._fit_inference_info["identifiability"].aliases if a.term == term]
    summary = model.summary()
    notes = [" ".join(note.split()) for note in summary._info["identifiability_notes"]]
    (line,) = [note for note in notes if note.startswith((f"{term} ", f"{term}:"))]
    assert line in " ".join(str(summary).split())
    return alias, line


def _design_rows(model, X) -> np.ndarray:
    """``[1, X_D]`` on the rows of ``X``: each coefficient's column, scored as ``predict`` does."""
    from superglm._frame import as_eager_frame
    from superglm.model import base

    frame, p = as_eager_frame(X), len(model.result.beta)
    columns = [np.ones(len(X))]
    for unit in np.eye(p):
        columns.append(
            sum(
                base._score_prediction_term_exact(term, frame, unit)
                for term in base._prediction_plan(model)["features"]
            )
        )
    return np.column_stack(columns)


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
    # the footnote names the held level, its partner and the nesting that aliases them
    alias, note = _footnote(model, "g[P]")
    assert alias.aliased_with == ("f",) and alias.source == "construction"
    assert "aliased with f" in note and "each level of g" in note
    assert "union of levels of f" in note


def test_a_duplicated_numeric_blanks_the_duplicate_alone():
    """``x2 = 2 x1``: ``x2`` is aliased at float64; ``x1``'s SE is the fit without ``x2``."""
    X = pd.DataFrame({"x": X_SPREAD, "x1": X1, "x2": 2.0 * X1})
    features = {"x": Spline(n_knots=6), "x1": Numeric(), "x2": Numeric()}
    model = _fit(features, X, Y)
    alone = _fit({"x": features["x"], "x1": Numeric()}, X[["x", "x1"]], Y)
    assert _blank(model, X, Y) == {"x2": [0]}
    assert _rows(model)["x1"] == pytest.approx(_rows(alone)["x1"], rel=1e-8)
    # the footnote names the held duplicate, the term it keeps and that no row separates them
    alias, note = _footnote(model, "x2")
    assert alias.aliased_with == ("x1",) and alias.source == "rank"
    assert "aliased with x1" in note and "no data separates them" in note


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


def _drifted_lss(monkeypatch, location, X, y, width: int, drift: dict[int, float]):
    """A Gaussian SuperLSS fit whose start is moved by ``drift`` (coordinate: step) along an alias."""
    import superglm.distributional.solver.solver as solver
    from superglm import SuperLSS
    from superglm.distributional import GaussianLS

    start = solver._initial_coefficients

    def drifted(context):
        coefficients = np.array(start(context))
        if context.layout.n_coefficients == width:
            coefficients[list(drift)] += list(drift.values())
        return coefficients

    monkeypatch.setattr(solver, "_initial_coefficients", drifted)
    family = GaussianLS(scale_floor=0)
    model = SuperLSS(family, family.location(*location), family.scale(), discrete=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("case", ["duplicate", "level", "constant"])
def test_superlss_refits_without_the_coordinate_superglm_blanks(monkeypatch, case):
    """An exact alias, the start moved 3 along it: SuperLSS refits without SuperGLM's blank.

    ``x2 = x1``; ``x = 1[g = b]`` beside ``g``, in that order, so the latest construction
    column is ``g[b]``; a constant ``c = 2`` beside the intercept.  The refit drops the
    coordinate SuperGLM's gauge holds (``aliases.held_coordinates``), so the two fits blank the
    same coefficient, at zero with no variance (DESIGN6's invariance).  It pins what
    ``253c6d353`` already did here, where the rank's earliest representatives in coordinate
    order are the term order's.  Mutation checks: holding the earliest construction column
    drops ``x1``, ``x`` and the intercept; reading the design without its intercept column
    calls the constant no alias and publishes it walked, nonzero.
    """
    from superglm import cat, term

    rng = np.random.default_rng(3)
    n = 600
    x1, g = rng.normal(size=n), rng.choice(["a", "b"], n)
    y = 1.0 + 0.5 * x1 + 0.7 * (g == "b") + rng.normal(size=n)
    if case == "duplicate":
        X = pd.DataFrame({"x1": x1, "x2": x1})
        features = {"x1": Numeric(), "x2": Numeric()}
        location, drift = (term("x1", Numeric()), term("x2", Numeric())), {1: 3.0, 2: -3.0}
    elif case == "level":
        X = pd.DataFrame({"x": (g == "b").astype(float), "g": g})
        features = {"x": Numeric(), "g": Categorical(base="a")}
        location, drift = (term("x", Numeric()), cat("g", base="a")), {1: 3.0, 2: -3.0}
    else:
        X = pd.DataFrame({"x1": x1, "c": np.full(n, 2.0)})
        features = {"x1": Numeric(), "c": Numeric()}
        location, drift = (term("x1", Numeric()), term("c", Numeric())), {0: 6.0, 2: -3.0}
    glm = SuperGLM(family="gaussian", features=features, selection_penalty=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm.fit(X, y)
    ((name, (local,)),) = _blank(glm, X, y).items()
    width = 2 + X.shape[1]
    lss = _drifted_lss(monkeypatch, location, X, y, width, drift)
    fitted = lss._require_fitted()
    blank = fitted.layout.term_slices[f"location:{name}"].start + local
    variance = np.diag(np.asarray(lss.covariance_))
    location_block = fitted.layout.predictors[0].coefficient_slice
    zero = [i for i in range(location_block.stop) if fitted.coefficients[i] == 0.0]
    assert fitted.fit_state.solver_result.converged
    assert zero == [blank] and variance[blank] == 0.0
    assert np.all(variance[np.arange(width) != blank] > 0.0)


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "chunked"])
def test_superlss_withholds_the_standard_errors_of_a_near_duplicate_it_did_not_fit(discrete):
    """``x2 = x1 + 1e-8 h`` (the design gate's probe): nothing blank, no standard error for the pair.

    SuperGLM identifies the pair and fits it, blanking nothing
    (``test_a_near_duplicate_column_is_fitted_as_mgcv_fits_it``, SE about 2.2e6).  SuperLSS's
    terminal rank drops ``x1 - x2``, which the design identifies at 2000 rows, so it keeps the
    walked state (fitting the truncated direction, D6, is not built for SuperLSS) and its
    summary withholds both standard errors with a note.  At ``253c6d353`` it refitted without
    ``x2``, printed ``x1``'s SE as 0.0224, eight orders too small, and ``x2`` at zero.
    Mutation checks: the ``rho <= 1e-7`` gate refits; no reached coordinate prints 0.0224.
    """
    from superglm import SuperLSS, s, term
    from superglm.distributional import GaussianLS

    X, y = _near_duplicate_frame(1e-8)
    glm = SuperGLM(
        family="gaussian", features={"x1": Numeric(), "x2": Numeric(), "z": Spline(n_knots=8)}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm.fit(X, y)
    assert _blank(glm, X, y) == {}
    family = GaussianLS(scale_floor=0)
    location = family.location(term("x1", Numeric()), term("x2", Numeric()), s("z", n_knots=8))
    lss = SuperLSS(family, location, family.scale(), discrete=discrete)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lss.fit_reml(X, y)
    table = lss.summary(X_train=X).set_index(["parameter", "term"])
    pair = table.loc[[("location", "x1"), ("location", "x2")]]
    assert pair["se"].isna().all() and np.all(pair["estimate"] != 0.0)
    from superglm.distributional.terms import UNFITTED_NOTE

    assert (pair["note"] == UNFITTED_NOTE).all()
    intercepts = table.loc[[("location", "(intercept)"), ("scale", "(intercept)")]]
    assert np.all(np.isfinite(intercepts["se"]))


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
    # the footnote names the held smooth part, its partner, the level and its one value of x
    alias, note = _footnote(model, "x:f[E]'s unpenalised part")
    assert alias.aliased_with == ("f",) and alias.source == "construction"
    assert "aliased with f" in note and "level E of f" in note
    assert "1 distinct value of x" in note


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
        # the footnote names the blank coefficient, its partner, the level and its one value
        alias, note = _footnote(model, "x:f[E]")
        assert alias.coordinate == model._dm.p - 1 and alias.aliased_with == ("f",)
        assert "aliased with f" in note and "level E of f" in note
        assert "1 distinct value of x" in note


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
    a truncated Hessian it keeps the flags it was saved with, so it loses no SE it showed.  A
    model that kept its fit state does not read them: its first inference decides the record
    again at its saved coefficients (``test_saved_master_records``); this is the record a
    model reads where it cannot be decided again (an edited one, whose inference stays as
    saved).
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


def _structured_rows(model, W) -> tuple[np.ndarray, np.ndarray]:
    """``B = sqrt(W) X_c`` (``X_c`` centred at ``W``) and the penalty ``S``, dense, of a structured fit."""
    from superglm.model import state_ops

    state = state_ops.retained_linear_state(model)
    n = model._dm.n
    X = np.hstack([np.asarray(m.toarray(), float).reshape(n, -1) for m in model._dm.group_matrices])
    B = np.sqrt(W)[:, None] * (X - (W @ X) / np.sum(W))
    S = state_ops._structured_penalty_product(state.penalized_operator)(np.eye(X.shape[1]))[0]
    return B, S


def _stand_in_record(model, directions, W, B, S):
    """The structured fit's record decided at weights ``W`` by a factor that truncated ``directions``.

    The stand-in inverts the dense ``H = B'B + S`` (``_structured_rows``) off ``directions`` and
    presents them as its only truncated ones, so the owner reads exactly those whatever the
    fit's own factor truncated.  Its condition estimate (it only enters the accuracy note) is 1.
    """
    from types import SimpleNamespace

    from superglm.model import state_ops

    state = state_ops.retained_linear_state(model)
    H = B.T @ B + S
    complement = np.linalg.svd(directions, full_matrices=True)[0][:, directions.shape[1] :]
    inverse = complement @ np.linalg.solve(complement.T @ H @ complement, complement.T)
    factor = SimpleNamespace(
        shape=H.shape,
        rank=H.shape[0] - directions.shape[1],
        schur_condition_estimate=1.0,
        null_directions=lambda: directions,
        solve=lambda vectors: inverse @ vectors,
        selected_inverse_diagonal=lambda indices: np.diag(inverse)[indices],
    )
    forced = dataclasses.replace(state, profiled_factor=factor, identifiability=None)
    return state_ops._structured_identifiability(
        model, forced, W, publish=True, residual=state_ops._stepped_residual(model, W)
    )


@pytest.mark.pins_behaviour
def test_a_structured_fit_reads_its_truncated_directions_alone(monkeypatch):
    """Five one-row ``sz`` levels among 60: the rows are read on the factor's truncated directions alone.

    Pins the work the owner does on a structured fit, which already held; forced so that it no
    longer rests on the fit's own factor truncating exactly 4: a stand-in factor
    (``_stand_in_record``) truncates exactly the dense Hessian's null
    directions, the right singular vectors of the Jacobi-scaled ``[B; S]`` under ``gamma_n`` of
    its largest (each one-row level's line is one), with a gap to the next above ``sqrt(u)``, so
    the count is the design's and not the fit's factor's.  Those are the only undecided
    directions; the weighted rows put each under its rounding bound and so does the design at
    unit weight, so they are ``null(H)``: two passes of that many columns, one alias each, and
    construction columns only for the terms they reach.  At ``633289397`` the owner formed the
    Gram of every level's line and factored ``X_F`` whole (120 columns), which took 54 s and
    1.4 GiB at 1000 levels.
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
    W = np.array(state_ops._solver_space_working_weights(model), dtype=np.float64)
    B, S = _structured_rows(model, W)
    jacobi = np.sqrt(np.diag(B.T @ B + S))
    _, singular, right = np.linalg.svd(np.vstack((B, S)) / jacobi, full_matrices=False)
    exact = singular <= _gamma(len(y)) * singular[0]
    assert np.min(singular[~exact]) > np.sqrt(U) * singular[0]
    directions = right[exact].T / jacobi[:, None]
    truncated = directions.shape[1]
    assert truncated > 0

    formed: list[str] = []
    construction = state_ops._raw_construction

    def counted(null, blocks, group, matrix):
        formed.append(group.name)
        return construction(null, blocks, group, matrix)

    monkeypatch.setattr(state_ops, "_raw_construction", counted)
    widths = _row_passes(monkeypatch)
    record = _stand_in_record(model, directions, W, B, S)
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


def _near_alias():
    """``z = x + 1e-8 h`` on the dense solver, and the least-squares oracle of ``[1, x, z]``.

    Returns the fit, its rows, the design, held-out rows with theirs, and the oracle's
    coefficients (``numpy.linalg.lstsq``) and SEs (from a QR of the design).
    """
    draws = np.random.default_rng(11).standard_normal((8, 3000))
    h, x = draws[4], draws[5]
    frame = pd.DataFrame({"x": x, "z": x + 1e-8 * h})
    held = pd.DataFrame({"x": draws[1], "z": draws[1] + 1e-8 * draws[0]})
    y = 1.0 + 0.5 * x + 3e-6 * h + 0.5 * draws[7]
    model = SuperGLM(family="gaussian", features={"x": Numeric(), "z": Numeric()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, y)
    design, held_design = (
        np.column_stack((np.ones(len(rows)), rows["x"], rows["z"])) for rows in (frame, held)
    )
    coefficients, residual, _, _ = np.linalg.lstsq(design, y, rcond=None)
    inverse = np.linalg.inv(np.linalg.qr(design)[1])
    scale = float(residual[0]) / (len(y) - 3)
    se = np.sqrt(scale * np.sum(inverse * inverse, axis=1))
    return model, frame, y, design, (held, held_design), coefficients, se


def _near_alias_se(model, frame, y) -> np.ndarray:
    metrics = model.metrics(frame, y)
    return np.r_[metrics.intercept_se, metrics.coefficient_se["x"], metrics.coefficient_se["z"]]


def test_a_near_alias_is_fitted_from_its_undecided_direction():
    """``z = x + 1e-8 h``: the Gram cannot tell its smallest direction from zero; the rows can.

    The Jacobi-scaled design has singular value ``7e-9`` there, its Gram ``5e-17``, under the
    Gram's formation bound, so the dense solver truncates it; read from the rows it clears the
    rounding bound (``7e-13``), so the penalised fit identifies ``z`` and the fit takes the step
    the rows give along it (``identifiability.WeakDirections``).  The fit is the least-squares
    fit of ``[1, x, z]``: its fitted values and held-out predictions are the oracle's to about
    ``kappa u`` of their size (``kappa`` the Jacobi-scaled design's condition, ``2e8``; a
    least-squares fit's values move by ``O(kappa)`` times a perturbation, Higham 2002, Thm 20.1),
    its normal equations hold to the rounding of evaluating them (``gamma_(n+4)``, componentwise),
    and its SEs are the oracle's (``kappa u`` ~ 1e-8, rtol 1e-6).  At ``1d18eaf47`` ``z`` was
    blank as fixed by the solver's rank decision, and ``x`` showed the SE of the fit without
    ``z``, eight orders too small.  Mutation: skipping the step (``WeakDirections.step`` zero)
    leaves ``z`` at zero, which moves the predictions by ``3e-3``.
    """
    model, frame, y, design, (held, held_design), coefficients, se = _near_alias()
    condition = np.linalg.cond(design / np.linalg.norm(design, axis=0))
    for rows, rows_design in ((frame, design), (held, held_design)):
        reference = rows_design @ coefficients
        error = np.max(np.abs(model.predict(rows) - reference))
        assert error <= 10.0 * condition * U * np.max(np.abs(reference))
    beta = np.r_[model.result.intercept, model.result.beta]
    residual = y - design @ beta
    magnitude = np.abs(design).T @ (np.abs(residual) + np.abs(design) @ np.abs(beta))
    assert np.max(np.abs(design.T @ residual) / magnitude) <= _gamma(len(y) + 4)
    np.testing.assert_allclose(_near_alias_se(model, frame, y), se, rtol=1e-6)
    published = model._fit_inference_info["identifiability"]
    assert published.aliases == () and published.weak is not None and published.weak.fitted


@pytest.mark.parametrize("kept", [True, False], ids=["kept", "truncated"])
def test_a_near_alias_reads_one_direction_whether_the_solver_kept_it_or_not(kept, monkeypatch):
    """``z = x + 1e-8 h`` with the solver's decision forced both ways: one direction is read.

    The fit's formed Gram is decomposed with the near alias kept (a ridge of ``1e-13`` of the
    diagonal) or truncated (a rank cut of ``1e-8``).  Either way that one direction is all the
    decomposition leaves undecided (a kept eigenvalue under the formation bound is recomputed as
    undecided, ``state_ops._dense_base``), so the publication reads one direction from the rows,
    once, and the SEs are the least-squares oracle's.  At ``633289397`` the owner factored all of
    ``X_F`` from the rows.
    """
    from superglm.solvers.rank import SHARED_RANK_POLICY, decompose_gram

    model, frame, y, design, _, _, se = _near_alias()
    centred = design[:, 1:] - design[:, 1:].mean(axis=0)
    hessian = centred.T @ centred  # unpenalised Gaussian: ``H = X_c'X_c``
    if kept:
        decomposition = decompose_gram(hessian + 1e-13 * np.diag(np.diag(hessian)))
        assert decomposition.rank == decomposition.width
    else:
        policy = dataclasses.replace(SHARED_RANK_POLICY, gram_rcond=1e-8)
        decomposition = decompose_gram(hessian, policy=policy)
        assert decomposition.rank == decomposition.width - 1
    widths = _row_passes(monkeypatch)
    _republish(model, decomposition)
    record = model._fit_inference_info["identifiability"]
    assert widths == [1] and record.row_directions == 1
    assert record.aliases == () and record.weak is not None
    np.testing.assert_allclose(_near_alias_se(model, frame, y), se, rtol=1e-6)


# ------------------------------------------------------------------ one gauge
def test_fit_at_its_default_lambda_publishes_the_owners_constraint():
    """Level E at one ``x``, ``fit()`` at lambda 0.1: the slope part is held, ``f[E]`` reported.

    The dense solver's rank decision here fixes ``f[E]`` at zero, not the later term's slope
    (stage 1, follow-up 3); the record's gauge moves the coefficients along the alias onto the
    later term's constraint (``identifiability.Gauge``), so ``f[E]`` reports its coefficient and
    the SE of the fit with that constraint held, as ``fit_reml`` does, and no fitted value
    moves.  The published coefficients are ``fl(beta - N s)``, ``s = (G'N)^-1 G' beta``, so
    ``G'`` of them is zero to the rounding of forming ``s``, ``N s``, the difference and the
    product itself: within ``3 gamma_(p+2) |G|'(|beta| + |N||s|)``.  Mutation: no gauge published
    blanks ``f[E]`` and reports it at zero; a gauge not applied leaves ``G' beta`` at ``1e2``.
    """
    from superglm.solvers.identifiability import Gauge

    moves: list[tuple[np.ndarray, np.ndarray]] = []
    coefficients = Gauge.coefficients

    def recorded(gauge, beta, intercept):
        moved = coefficients(gauge, beta, intercept)
        moves.append((moved[0], moved[2]))
        return moved

    X = pd.DataFrame({"x": np.where(LEVEL == "E", 5.0, X_SPREAD), "f": LEVEL})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Gauge, "coefficients", recorded)
        model = _fit({"x": Spline(n_knots=6), "f": Categorical()}, X, Y, interactions=[("x", "f")])
    record = model._fit_inference_info["identifiability"]
    assert record.gauge is not None
    assert _blank(model, X, Y) == {}
    level = next(g for g in model._groups if g.name == "f")
    functionals = record.gauge.functionals
    beta = model.result.beta
    step = next(step for moved, step in moves if np.array_equal(moved, beta))
    G, null = np.abs(functionals[:, 0]), np.abs(record.gauge.null)
    allowance = 3.0 * _gamma(len(beta) + 2) * float(G @ np.abs(beta) + G @ (null @ np.abs(step)))
    assert abs(float(functionals[:, 0] @ beta)) <= allowance
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
    reports its variance (``identifiability.WeakDirections``): no blank, every SE^2 / phi
    mgcv's ``Vp / sig2`` and the EDF and scale mgcv's (rtol 1e-6).  The direction's curvature is
    read from the rows to the record's bound ``WeakDirections.accuracy``, ``3.8e-7`` here
    (``kappa u`` with ``kappa = 3.4e9``, the penalised Hessian's condition), so 1e-6 is about
    ``2.7 kappa u``; measured, the SEs^2 agree to ``6.0e-9``, the scale to ``2.5e-13`` and the
    EDF to ``3.3e-10``.  The coefficients near ``+-9e5`` cancel in every
    prediction, so the fit is compared where a user reads it: the link predictions on the
    training rows and on held-out rows are mgcv's to about ``kappa u`` of their size (``kappa``
    the Jacobi-scaled design's condition, ``2.5e8``; Higham 2002, Thm 20.1), each engine
    carrying that much.  At ``1d18eaf47`` ``x2`` was blank and ``x1`` showed SE 0.0224 against
    mgcv's 2.2e6.  Mutation: no step along the direction moves the predictions by ``3e-2``.
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
    mgcv = np.asarray(oracle["coef"])
    rows = np.random.default_rng(2)
    z, x1, h = rows.uniform(size=500), rows.normal(size=500), rows.normal(size=500)
    held = pd.DataFrame({"x1": x1, "x2": x1 + oracle["d"] * h, "z": z})
    training = np.column_stack((np.ones(len(y)), design))
    condition = np.linalg.cond(training / np.linalg.norm(training, axis=0))
    for frame, rows_design in ((X, training), (held, _design_rows(model, held))):
        reference = rows_design @ mgcv
        error = np.max(np.abs(model.predict(frame) - reference))
        assert error <= 10.0 * condition * U * np.max(np.abs(reference))
    assert model.result.phi == pytest.approx(oracle["sig2"], rel=1e-6)
    assert model.result.effective_df == pytest.approx(oracle["edf"], rel=1e-6)


def test_a_near_duplicates_leverage_is_the_published_fits_on_its_rows_and_their_copies():
    """The rows and equal copies of them give one leverage, which sums to mgcv's EDF.

    Leverage is the hat diagonal of the published fit, the weak direction included, so its
    sum is the fit's EDF: mgcv's 11.3156 to the ``1e-6`` the direction's curvature is read to
    (``test_a_near_duplicate_column_is_fitted_as_mgcv_fits_it``).  The weak direction's term is
    read as ``(r' P U)^2``; through the dense covariance's ``1 / C`` entries (``C`` about
    ``2e-13``) it carries their rounding, so at ``253c6d353`` the rows' own leverage summed to
    12.23, and an equal copy, re-formed on the solver's truncated inverse, to 10.32.
    Mutations: the weak root read only for a released fit; copies recognised only for
    structured fits.
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
    own, copy = model.metrics(X, y), model.metrics(X.copy(), y.copy())
    np.testing.assert_array_equal(copy.leverage, own.leverage)
    assert float(np.sum(own.leverage)) == pytest.approx(oracle["edf"], rel=1e-6)
    np.testing.assert_array_equal(copy.coefficient_se["x1"], own.coefficient_se["x1"])


# --------------------------------------------- rows other than the fit's, in its gauge
def _gauge_fit(columns: dict[str, np.ndarray]):
    """A Gaussian fit of unpenalised numeric columns (``x2 = x1`` is held by the gauge)."""
    rng = np.random.default_rng(3)
    y = 1.0 + columns["x1"] + rng.normal(size=len(columns["x1"]))
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, features={name: Numeric() for name in columns}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(pd.DataFrame(columns), y)
    return model


def _gauge_oracle(model, frame) -> tuple[np.ndarray, float]:
    """``B (B'HB)^+ B'``'s diagonal on ``frame`` and the condition of ``B'HB`` on its range.

    ``H = A'A`` (a Gaussian identity fit's Fisher weights are one, the columns unpenalised),
    ``A = [1, X]`` scored as ``predict`` does, and ``B`` spans the coefficients the fit's gauge
    leaves free: every coordinate but the ones it holds at zero.  An eigenvalue of ``B'HB``
    under ``n u`` of the largest is its null space, whose coordinates are blank; the others'
    variances are those of every generalised inverse.
    """
    A = _design_rows(model, frame)
    held = 1 + np.flatnonzero(~np.asarray(model._fit_inference_info["coefficient_identified"]))
    B = np.eye(A.shape[1])[:, np.setdiff1d(np.arange(A.shape[1]), held)]
    values, vectors = np.linalg.eigh(B.T @ (A.T @ A) @ B)
    kept = values > len(A) * U * values[-1]
    inverse = (vectors[:, kept] / values[kept]) @ vectors[:, kept].T
    return np.diag(B @ inverse @ B.T), float(values[-1] / values[kept][0])


def _raw_se(model, frame) -> np.ndarray:
    metrics = model.metrics(frame, np.zeros(len(frame)))
    return np.r_[
        metrics.intercept_se_raw,
        [metrics.coefficient_se_raw[name][0] for name in frame.columns],
    ]


@pytest.mark.parametrize("correlation", [0.0, 0.5], ids=["identity", "correlated"])
def test_rows_that_resolve_a_training_alias_read_the_covariance_in_the_fits_gauge(correlation):
    """Training ``x2 = x1``, held by the gauge at ``beta2 = 0``; rows that separate the two.

    The coefficients are the fit's, so ``x2`` stays blank, and the covariance is the fixed
    gauge's, ``B (B'HB)^+ B'`` (w5m2's counterexample: with the slopes' Hessian ``I`` the fixed
    gauge's ``var(beta1)`` is 1 where the oblique ``P H^+ P'`` gives 2; correlated at 0.5 it is
    1 where ``H^+`` gives 4/3), to ``10 kappa n u`` of the oracle's variance.  At
    ``253c6d353`` these rows printed x2's SE and, correlated, ``H^+``'s x1.  Mutations: no
    gauge restriction; the fit's blanks dropped.
    """
    rng = np.random.default_rng(1)
    x1 = rng.normal(size=200)
    model = _gauge_fit({"x1": x1, "x2": x1.copy()})
    assert model._fit_inference_info["coefficient_identified"].tolist() == [True, False]
    # rows whose centred slopes have the Gram ``[[1, c], [c, 1]]``
    a, b = np.array([1.0, -1.0, 1.0, -1.0]) / 2.0, np.array([1.0, 1.0, -1.0, -1.0]) / 2.0
    frame = pd.DataFrame({"x1": a, "x2": correlation * a + np.sqrt(1.0 - correlation**2) * b})
    variance, condition = _gauge_oracle(model, frame)
    se = _raw_se(model, frame)
    assert np.isnan(se[2]) and np.all(np.isfinite(se[:2]))
    np.testing.assert_allclose(se[:2] ** 2, variance[:2], rtol=10 * condition * len(frame) * U)


@pytest.mark.parametrize(
    ("training", "evaluation", "blank"),
    [
        ("aliased", {"x3": "x1"}, [1, 2, 3]),
        ("aliased", {"x3": "x2"}, [2]),
        ("full", {"x3": "x1"}, [1, 3]),
    ],
    ids=["new-alias-in-the-gauge", "alias-the-gauge-holds", "new-alias"],
)
def test_rows_that_introduce_an_alias_blank_what_it_reaches_in_the_fits_gauge(
    training, evaluation, blank
):
    """Rows on which one column copies another: blank where ``null(B'HB)`` reaches.

    The rows leave undetermined ``null(H) cap null(G')``, not the difference of the two null
    spaces (w5m2): beside a training gauge holding ``beta2 = 0``, ``x3 = x1`` blanks x1 and x3,
    and ``x3 = x2`` blanks nothing new, since the gauge already holds that direction, so x3
    keeps the gauge's SE.  A full-rank fit's ``x3 = x1`` blanks both, since the coefficients
    do not move to hold either at zero.  Every other SE is the oracle's ``B (B'HB)^+ B'``.  At
    ``253c6d353`` the rows' own decomposition blanked one pivot of each new alias and none of
    the fit's, and blanked x3 beside the gauge.  Mutations: the restricted null space read
    off ``H`` alone; the owner's decision skipped.
    """
    rng = np.random.default_rng(1)
    x1, other, x3 = rng.normal(size=(3, 200))
    model = _gauge_fit({"x1": x1, "x2": x1.copy() if training == "aliased" else other, "x3": x3})
    e = rng.normal(size=(3, 400))
    columns = {"x1": e[0], "x2": e[1], "x3": e[2]}
    for name, source in evaluation.items():
        columns[name] = columns[source]
    frame = pd.DataFrame(columns)
    se = _raw_se(model, frame)
    assert np.flatnonzero(np.isnan(se)).tolist() == blank
    variance, condition = _gauge_oracle(model, frame)
    shown = np.setdiff1d(np.arange(4), blank)
    np.testing.assert_allclose(
        se[shown] ** 2, variance[shown], rtol=10 * condition * len(frame) * U
    )


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

    Five Poisson levels with no claims: their working weights vanish (where the fit stops, a
    separation, which the exact test of stage 6b decides).  The decision is taken here at those
    rows' weights set to zero, by a stand-in factor that truncated the directions it cannot
    certify (``_stand_in_record``): those whose Jacobi-scaled ``[B; S]`` singular value is
    under ``sqrt(u)``, so their curvature is under the rounding of ``H``.  The ones under
    ``gamma_n``, which no weighted row reaches, are not resolved at float64: the design at unit
    weight identifies each, so none is an alias, none is moved along and no coordinate is blank;
    the rest the rows identify, with their curvature.  ``zz = z + 1e-8 h``: the factor truncates
    the pair's difference, and the rows identify it and fit it.  Either way the owner reads only
    the truncated directions (a pass of their width, at most twice) and forms one small
    construction map per term they reach.  At ``1d18eaf47`` both fell back to factoring the
    whole restricted design from the rows (2003 columns at 1000 levels, 1.6 GiB; the zero-claim
    case then refused the gauge and blanked 2306 coefficients).  Mutations: candidates of every
    coordinate fail the work count; skipping the design's read at unit weight moves the
    zero-claim directions as aliases.
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
    W = np.array(state_ops._solver_space_working_weights(model), dtype=np.float64)
    if case == "zero_claims":
        W[g < 5] = 0.0
        B, S = _structured_rows(model, W)
        jacobi = np.sqrt(np.diag(B.T @ B + S))
        _, singular, right = np.linalg.svd(np.vstack((B, S)) / jacobi, full_matrices=False)
        directions = right[singular <= np.sqrt(U) * singular[0]].T / jacobi[:, None]
        exact = int(np.count_nonzero(singular <= _gamma(len(y)) * singular[0]))
    else:
        assert [group.name for group in model._groups[:2]] == ["z", "zz"]
        B, S = _structured_rows(model, W)
        directions = np.zeros((B.shape[1], 1))
        directions[[0, 1], 0] = np.sqrt(0.5) * np.array([1.0, -1.0])  # ``z - zz``
    truncated = directions.shape[1]
    formed: list[str] = []
    construction = state_ops._raw_construction

    def counted(null, blocks, group, matrix):
        formed.append(group.name)
        return construction(null, blocks, group, matrix)

    monkeypatch.setattr(state_ops, "_raw_construction", counted)
    widths = _row_passes(monkeypatch)
    record = _stand_in_record(model, directions, W, B, S)
    assert widths and all(width <= truncated for width in widths) and len(widths) <= 2
    assert record.row_directions == truncated
    # one small map per term the directions reach, never one per level
    assert len(formed) == len(set(formed)) <= 3
    assert record.coefficient_identified().all()
    if case == "zero_claims":
        assert 0 < exact < truncated
        assert len(record.aliases) == exact
        assert {a.source for a in record.aliases} == {"float64"}
        assert record.gauge is not None and not np.any(record.gauge.moved)
        assert record.weak is not None and record.weak.root.shape[1] == truncated - exact
        assert not record.weak.fitted  # Poisson: the publication resumes IRLS on them
    else:
        assert record.aliases == () and record.gauge is None
        assert record.weak is not None and record.weak.fitted
        assert record.weak.root.shape[1] == 1
