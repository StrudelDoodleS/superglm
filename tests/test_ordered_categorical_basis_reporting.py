"""Reporting and export for OrderedCategorical Piecewise/Polynomial bases.

Pins the inner-basis reporting vocabulary: segmented terms report STRUCTURAL
CONTRASTS -- one slope-change Wald row per stated break and one curvature row
per degree>=2 segment, never per-segment per-power z rows (under C0 seams that
geometry does not exist; the rows are ordinary fixed-knot inference, Smith
1979) -- while ``basis=Polynomial(powers=[...])`` keeps one clean-z row per
stated power (the main-effect property of the exposure-orthonormal ordinal
contrasts). Both whole-term rows are plain Wald chi-squares on an unpenalized
block, counted in the parametric df bucket, and never trigger the Wood smooth
footnote on their own. Rating-table export stays one row per band for every
inner basis, exact against the fitted model.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    FactorSmooth,
    GroupLasso,
    Numeric,
    OrderedCategorical,
    Piecewise,
    Polynomial,
    RandomEffect,
    Spline,
    SuperGLM,
)
from superglm.solvers.rank import SHARED_RANK_POLICY
from superglm.types import LambdaPolicy

LEVELS = [f"Mi{i:03d}" for i in range(10)]


def _frame(n: int = 4000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"band": rng.choice(LEVELS, n)})
    position = {level: index for index, level in enumerate(LEVELS)}
    signal = np.array(
        [0.02 * min(position[b], 4) ** 2 + (0.05 if position[b] > 6 else 0.0) for b in X["band"]]
    )
    y = signal + rng.normal(0.0, 0.05, n)
    return X, y


def _log_link_frame(n: int = 4000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """``_frame``'s band signal as a Poisson count, for the export tests.

    The rating-table export takes log-link models only -- its table is a
    product of factors -- so the tests that go through it need a response the
    log link can carry.  Same levels and same monotone-then-flat shape as
    ``_frame``, so the level axis under test is unchanged.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"band": rng.choice(LEVELS, n)})
    position = {level: index for index, level in enumerate(LEVELS)}
    mu = np.exp(
        -1.0
        + np.array(
            [
                0.02 * min(position[b], 4) ** 2 + (0.05 if position[b] > 6 else 0.0)
                for b in X["band"]
            ]
        )
    )
    return X, rng.poisson(mu).astype(float)


def _fit(spec: OrderedCategorical) -> SuperGLM:
    X, y = _frame()
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def segmented_model() -> SuperGLM:
    return _fit(
        OrderedCategorical(
            order=LEVELS, basis=Piecewise(breaks=["Mi004", "Mi007"], degrees=[2, 1, 0])
        )
    )


@pytest.fixture(scope="module")
def polynomial_model() -> SuperGLM:
    return _fit(OrderedCategorical(order=LEVELS, basis=Polynomial(powers=[1, 3])))


# ── summary rows ─────────────────────────────────────────────────────


def test_segmented_summary_reports_structural_contrasts(segmented_model) -> None:
    text = str(segmented_model.summary())
    # Knots [0, 4, 7, 9] with degrees [2, 1, 0]: the flat tail merges knots 7
    # and 9 into one value column -> 3 knot-value groups minus the base, plus
    # one curvature column = 3 parameters.
    assert "[ordered piecewise, 3 params" in text
    assert "band[slope-change @ Mi004]" in text
    assert "band[slope-change @ Mi007]" in text
    assert "band[curvature Mi000..Mi004]" in text
    # One curvature row only: the degree-1 and degree-0 segments state none.
    assert text.count("curvature") == 1


def test_segmented_summary_has_no_per_power_rows(segmented_model) -> None:
    text = str(segmented_model.summary())
    assert "band[P1]" not in text
    assert "band[P2]" not in text


def test_segmented_slope_change_rows_are_real_wald_rows(segmented_model) -> None:
    for line in str(segmented_model.summary()).splitlines():
        if "slope-change @ Mi004" in line:
            # coef, se, z, p and CI all populated (no "---" placeholders in
            # the numeric fields).
            assert line.count("---") == 1  # the QS column only
            return
    raise AssertionError("slope-change row missing")


def test_segmented_df_lands_in_the_parametric_bucket(segmented_model) -> None:
    text = str(segmented_model.summary())
    assert "(0 smooth)" in text
    assert "Wood (2013)" not in text


def test_segmented_level_rows_one_per_band(segmented_model) -> None:
    text = str(segmented_model.summary())
    for level in LEVELS:
        assert f"band[{level}]" in text


def test_polynomial_summary_reports_stated_powers_with_clean_z(polynomial_model) -> None:
    text = str(polynomial_model.summary())
    assert "[ordered polynomial, 2 params" in text
    assert "band[P1]" in text
    assert "band[P3]" in text
    assert "band[P2]" not in text
    for line in text.splitlines():
        if "band[P1]" in line:
            assert line.count("---") == 1  # z/p/CI populated; QS column only
            return
    raise AssertionError("per-power row missing")


def test_polynomial_summary_has_no_structural_rows(polynomial_model) -> None:
    text = str(polynomial_model.summary())
    assert "slope-change" not in text
    assert "curvature" not in text


def test_curvature_family_of_two_freedoms_reports_a_joint_test() -> None:
    model = _fit(
        OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi005"], degrees=[3, 1]))
    )
    text = str(model.summary())
    for line in text.splitlines():
        if "curvature Mi000..Mi005" in line:
            assert "[curvature, 2 params, chi2(2.0)=" in line
            return
    raise AssertionError("curvature family row missing")


_TWIN_LABELS = list("abcdefgh")


def _twin_frame(names, extra=None):
    codes = np.tile(_TWIN_LABELS, 20)
    frame = pd.DataFrame({name: codes for name in names})
    for name, values in (extra or {}).items():
        frame[name] = values
    return frame, np.tile(np.arange(8.0) ** 2, 20) + np.sin(np.arange(160.0))


def _twin_fit(names, degrees, extra_features=None, extra=None, *, basis=None, bases=None):
    """Ordered piecewise terms on one column (issue #737), with any ``extra_features``.

    ``basis`` makes each term's basis instead, and ``bases`` maps a term to its base level.
    """
    settings = {} if degrees is None else {"degrees": degrees}
    features = {
        name: OrderedCategorical(
            order=_TWIN_LABELS,
            base=(bases or {}).get(name, "a"),
            basis=Piecewise(breaks=["d"], **settings) if basis is None else basis(),
        )
        for name in names
    }
    features.update(extra_features or {})
    X, y = _twin_frame(names, extra)
    return SuperGLM(family="gaussian", features=features).fit(X, y), X, y


def _structural_contrasts(model, name):
    """Each structural row of ``name`` as rows over the fitted slopes, with its label."""
    group = next(g for g in model._groups if g.feature_name == name)
    width = len(model.result.beta)
    out = []
    for row in model._specs[name]._basis_spline.ordered_structural_rows():
        if row.kind == "slope_change":
            block = row.contrast[None, :]
        else:
            block = np.eye(group.size)[np.asarray(row.column_indices, dtype=np.intp)]
        full = np.zeros((len(block), width))
        full[:, group.sl] = block
        out.append((row.kind, full))
    return out


def _design(model):
    return np.column_stack([np.ones(model._dm.n), *[g.toarray() for g in model._dm.group_matrices]])


def _rank_gain(model, contrasts, rtol=None):
    """How much the slope contrasts raise the rank of ``[1, X]`` when appended.

    ``rtol`` is the singular-value cut relative to the largest, numpy's default when omitted.
    """
    design = _design(model)
    stacked = np.vstack([design, np.column_stack([np.zeros(len(contrasts)), contrasts])])
    return int(np.linalg.matrix_rank(stacked, rtol=rtol) - np.linalg.matrix_rank(design, rtol=rtol))


@pytest.mark.parametrize("degrees", [None, [2, 2], [3, 2]])
def test_structural_rows_of_twin_ordered_piecewise_terms_carry_no_default_error(degrees) -> None:
    """Two piecewise ordered terms on one column: no slope change or curvature is determined.

    Moving the curve from one term to the other leaves the fit, so every structural
    contrast raises the design's rank when appended. The default summary and the
    metrics summary keep each row's estimate and give no error, interval or test:
    the slope-change row, a one-freedom curvature row (degree 2) and the curvature
    family (degree 3), on both terms.
    """
    model, X, y = _twin_fit(["a", "b"], degrees)
    for name in ("a", "b"):
        for _, contrasts in _structural_contrasts(model, name):
            assert _rank_gain(model, contrasts) > 0
    for summary in (model.summary(), model.metrics(X, y).summary()):
        rows = [r for r in summary._coef_rows if "slope-change" in r.name or "curvature" in r.name]
        assert {r.name.split("[")[0] for r in rows} == {"a", "b"}
        for row in rows:
            assert row.se is None and row.z is None and row.p is None, row.name
            assert row.ci_low is None and row.ci_high is None, row.name
            assert row.wald_chi2 is None and row.wald_p is None, row.name
            assert row.ref_df is None or row.n_params == 1, row.name
        slope_change = next(r for r in rows if r.name == "a[slope-change @ d]")
        assert np.isfinite(slope_change.coef)
        text = str(summary)
        assert not any("curvature" in line and "chi2(" in line for line in text.splitlines())


@pytest.mark.pins_behaviour
def test_determined_structural_rows_keep_their_default_errors() -> None:
    """Control (pins): a structural contrast the fit determines keeps its least-squares error.

    One unaliased piecewise term: each row's error is ``sqrt(phi c'(X'X)^-1 c)`` of the
    Gaussian fit, with ``phi`` the residual mean square. A linear term on the level
    positions aliases the term's slopes, so no slope coefficient is determined, but the
    slope change is: appending it leaves the design's rank, and its row keeps the
    single term's error.
    """
    single, _, y = _twin_fit(["a"], [2, 2])
    design = _design(single)
    gram_inverse = np.linalg.inv(design.T @ design)
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    phi = float(residual @ residual) / (len(y) - design.shape[1])
    rows = {r.name: r for r in single.summary()._coef_rows}
    expected = {}
    for (_, contrasts), name in zip(
        _structural_contrasts(single, "a"),
        ["a[slope-change @ d]", "a[curvature a..d]", "a[curvature d..h]"],
    ):
        augmented = np.column_stack([np.zeros(len(contrasts)), contrasts])[0]
        expected[name] = np.sqrt(phi * augmented @ gram_inverse @ augmented)
        np.testing.assert_allclose(rows[name].se, expected[name], rtol=1e-9)
        assert rows[name].p is not None

    positions = np.tile(np.arange(8.0), 20)
    aliased, X, y = _twin_fit(
        ["a"], [2, 2], extra_features={"x": Numeric()}, extra={"x": positions}
    )
    slope_change = _structural_contrasts(aliased, "a")[0][1]
    shown = aliased.result.rank_info.coefficient_estimable()
    assert not np.all(shown[slope_change[0] != 0.0])
    assert _rank_gain(aliased, slope_change) == 0
    for summary in (aliased.summary(), aliased.metrics(X, y).summary()):
        row = next(r for r in summary._coef_rows if r.name == "a[slope-change @ d]")
        np.testing.assert_allclose(row.se, expected["a[slope-change @ d]"], rtol=1e-9)
    # other rows re-form the covariance, and its own data rank decides the contrast
    other = aliased.metrics(X.iloc[:120].reset_index(drop=True), y[:120] + 0.5).summary()
    row = next(r for r in other._coef_rows if r.name == "a[slope-change @ d]")
    assert row.se is not None and np.isfinite(row.se) and row.p is not None


def test_metrics_on_other_rows_pickle_with_their_structural_decisions() -> None:
    """Metrics re-formed on other rows pickle once read, and keep their structural decisions.

    A metrics object on other rows keeps its re-formed data rank's contrast decision for
    its summary; at dceef692b that was a local closure, and ``pickle.dumps`` raised.
    Restored, the twin terms' slope change stays blank, and the aliased term's determined
    one keeps the error a fresh object gives.
    """
    twin, X, y = _twin_fit(["a", "b"], None)
    positions = np.tile(np.arange(8.0), 20)
    aliased, X_aliased, _ = _twin_fit(
        ["a"], [2, 2], extra_features={"x": Numeric()}, extra={"x": positions}
    )
    for model, frame, shown in ((twin, X, False), (aliased, X_aliased, True)):
        other = frame.iloc[:120].reset_index(drop=True), y[:120] + 0.5
        metrics = model.metrics(*other)
        _ = metrics.coefficient_se
        restored = pickle.loads(pickle.dumps(metrics))
        row, fresh = (
            next(r for r in m.summary()._coef_rows if r.name == "a[slope-change @ d]")
            for m in (restored, model.metrics(*other))
        )
        assert (row.se is not None) is shown
        assert row.se == fresh.se and row.p == fresh.p


def _structured_fit(names, term, *, linear=False, direct_solve="auto"):
    """Ordered piecewise ``names`` on eight bands crossed with 40 levels, twice per cell,
    beside a structured ``term`` on the levels: a random effect, declared first so its
    levels sit before the border columns, or an ``fs`` or ``sz`` factor smooth; with a
    linear term on the band positions if ``linear``.
    """
    band = np.repeat(np.arange(8), 80)
    level = np.tile(np.repeat(np.arange(40), 2), 8)
    row = np.arange(len(band), dtype=float)
    z = 0.5 + 0.5 * np.sin(0.7 * row)
    X = pd.DataFrame({name: np.asarray(_TWIN_LABELS)[band] for name in names})
    X["g"], X["x"], X["z"] = level.astype(str), band.astype(float), z
    y = band**2.0 + 2.0 * np.cos(level) + np.sin(row) + z
    features = {"g": RandomEffect()} if term == "random effect" else {}
    for name in names:
        features[name] = OrderedCategorical(
            order=_TWIN_LABELS, base="a", basis=Piecewise(breaks=["d"])
        )
    if linear:
        features["x"] = Numeric()
    interactions = []
    if term != "random effect":
        features["z"] = Spline(n_knots=5)
        interactions.append(FactorSmooth("z", group="g", basis=term, k=5))
    model = SuperGLM(
        family="gaussian",
        features=features,
        interactions=interactions,
        direct_solve=direct_solve,
    )
    return model.fit_reml(X, y), X, y


@pytest.mark.parametrize("term", ["random effect", "fs", "sz"])
def test_structured_fits_decide_each_slope_change_on_their_own_data(term) -> None:
    """A structured fit decides whether its data determine a slope change, as a dense fit does.

    The terms cover the three compact geometries: a random effect's nested chain, an ``fs``
    factor smooth's independent blocks and an ``sz`` one's balance tree.  With a linear term
    on the band positions no slope coefficient is determined, but the slope change is:
    it keeps the dense fit's error in both summaries, where dceef692b blanked it.  Twin
    terms determine neither slope change, and both stay blank, where fe8a6eb9b showed them.
    """
    model, X, y = _structured_fit(["a"], term, linear=True)
    assert model._fit_inference_info.get("structured_covariance")
    slope_change = _structural_contrasts(model, "a")[0][1]
    shown = model._fit_inference_info["coefficient_estimable"]
    assert not np.all(shown[slope_change[0] != 0.0])
    assert _rank_gain(model, slope_change) == 0
    dense, _, _ = _structured_fit(["a"], term, linear=True, direct_solve="gram")
    expected = next(r for r in dense.summary()._coef_rows if r.name == "a[slope-change @ d]").se
    for summary in (model.summary(), model.metrics(X, y).summary()):
        row = next(r for r in summary._coef_rows if r.name == "a[slope-change @ d]")
        assert row.se is not None and row.p is not None
        np.testing.assert_allclose(row.se, expected, rtol=1e-9)

    twin, X, y = _structured_fit(["a", "b"], term)
    assert twin._fit_inference_info.get("structured_covariance")
    for name in ("a", "b"):
        assert _rank_gain(twin, _structural_contrasts(twin, name)[0][1]) > 0
    for summary in (twin.summary(), twin.metrics(X, y).summary()):
        rows = [r for r in summary._coef_rows if "slope-change" in r.name]
        assert len(rows) == 2
        assert all(r.se is None and r.p is None for r in rows), [r.name for r in rows]


def test_a_summary_saved_beside_an_ordered_piecewise_term_is_formed_again() -> None:
    """A summary cached before the structural rows were checked is dropped on restore.

    Pickled after ``summary()`` at fe8a6eb9b, the twin terms' model served that summary at
    dceef692b: ``a[slope-change @ d]`` kept SE 0.2179 and p = 0, which ``summary(alpha=0.1)``
    of the same restored model left blank.
    """
    model, _, _ = _twin_fit(["a", "b"], None)
    stale = next(r for r in model.summary()._coef_rows if r.name == "a[slope-change @ d]")
    stale.se, stale.z, stale.p = 0.21788071784804042, 35.3, 0.0  # what fe8a6eb9b cached
    restored = pickle.loads(pickle.dumps(model))
    row = next(r for r in restored.summary()._coef_rows if r.name == "a[slope-change @ d]")
    assert row.se is None and row.z is None and row.p is None
    assert np.isfinite(row.coef)


_TWIN_BASES = {"piecewise": None, "spline": lambda: Spline(n_knots=4)}


def _level_contrasts(model, name):
    """Each non-base level of ``name`` as a row over the fitted slopes, with its label."""
    spec = model._specs[name]
    start = next(g for g in model._groups if g.feature_name == name).start
    rows = np.asarray(spec.transform(np.asarray(_TWIN_LABELS, dtype=object)), dtype=float)
    base = np.asarray(spec.transform([spec._base_level]), dtype=float).ravel()
    out = []
    for label, row in zip(_TWIN_LABELS, rows):
        if label != spec._base_level:
            full = np.zeros((1, len(model.result.beta)))
            full[0, start : start + row.size] = row - base
            out.append((f"{name}[{label}]", full))
    return out


@pytest.mark.parametrize("basis", list(_TWIN_BASES))
def test_levels_of_twin_ordered_terms_carry_no_error_on_any_summary(basis) -> None:
    """Two ordered terms on one column: no level relativity is determined (issue #740).

    Moving the effect from one term to the other leaves the fit, so each level's contrast
    raises the design's rank when appended.  The default summary, both metrics summaries
    and the robust one keep each level's estimate with no error or interval, and flag it
    ``estimable=False``, as they do each structural row; the default summary's dictionary
    gives no error for a coefficient the fit does not determine.  At 71f10edce the default
    ones showed ``a[b]`` SE 0.1406 (piecewise) and 1.319 (spline), and every flag was True.
    """
    model, X, y = _twin_fit(["a", "b"], None, basis=_TWIN_BASES[basis])
    for name in ("a", "b"):
        for _, contrast in _level_contrasts(model, name):
            assert _rank_gain(model, contrast) > 0
    other = X.iloc[:120].reset_index(drop=True), y[:120] + 0.5
    summaries = {
        "default": model.summary(),
        "metrics": model.metrics(X, y).summary(),
        "other rows": model.metrics(*other).summary(),
        "robust": model.summary(covariance=model.sandwich_covariance()),
    }
    for kind, summary in summaries.items():
        rows = {r.name: r for r in summary._coef_rows}
        for name in ("a", "b"):
            for label, _ in _level_contrasts(model, name):
                row = rows[label]
                assert np.isfinite(row.coef), (kind, label)
                assert row.se is None and row.ci_low is None and row.ci_high is None, (kind, label)
                assert row.estimable is False, (kind, label)
        for row in summary._coef_rows:
            if "slope-change" in row.name or "curvature" in row.name:
                assert row.estimable is False, (kind, row.name)
    assert model.result.rank_info.coefficient_estimable().sum() == 0
    for se in summaries["default"]["standard_errors"].values():
        if isinstance(se, dict):
            assert all(np.all(np.isnan(v)) for v in se.values()), se


@pytest.mark.parametrize("basis", list(_TWIN_BASES))
def test_an_intercept_at_two_bases_the_fit_does_not_determine_shows_no_error(basis) -> None:
    """Twin ordered terms on different base levels: the reported intercept is not determined.

    The intercept is reported at each ordered term's base level, so with ``a`` based at
    ``a`` and ``b`` at ``h`` it moves when the effect moves between the terms: its
    functional is not orthogonal to the design's null space.  The default summary keeps
    its estimate and gives no error, test or interval, flagged ``estimable=False``; with
    one base both summaries keep the error, and the null-space check agrees.
    """
    from superglm.inference._ordered_reference import ordered_reference_beta_contrast

    def intercept_determined(model):
        design = _design(model)
        _, s, vt = np.linalg.svd(design)
        null = vt[int(np.sum(s > s[0] * 1e-10)) :].T
        width = len(model.result.beta)
        reference = ordered_reference_beta_contrast(
            width, list(model._specs), model._specs, model._groups
        )
        return np.linalg.norm(np.concatenate([[1.0], reference]) @ null) < 1e-8

    split, X, y = _twin_fit(["a", "b"], None, basis=_TWIN_BASES[basis], bases={"b": "h"})
    assert not intercept_determined(split)
    for summary in (split.summary(), split.metrics(X, y).summary()):
        row = summary._coef_rows[0]
        assert row.name == "Intercept" and np.isfinite(row.coef)
        assert row.se is None and row.z is None and row.p is None and row.ci_low is None
        assert row.estimable is False
    same, X, y = _twin_fit(["a", "b"], None, basis=_TWIN_BASES[basis])
    assert intercept_determined(same)
    for summary in (same.summary(), same.metrics(X, y).summary()):
        row = summary._coef_rows[0]
        assert row.se is not None and np.isfinite(row.se) and row.estimable is True


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("basis", list(_TWIN_BASES))
def test_a_single_ordered_term_keeps_its_level_errors(basis) -> None:
    """Control (pins): one unaliased ordered term keeps every level's error and flag.

    Piecewise is unpenalized, so each level's error is ``sqrt(phi c'(X'X)^-1 c)`` of the
    Gaussian fit; the penalized spline's are those ``term_inference`` reads from the same
    covariance.  The intercept keeps its error and every row stays ``estimable``.
    """
    model, X, y = _twin_fit(["a"], None, basis=_TWIN_BASES[basis])
    if basis == "piecewise":
        design = _design(model)
        gram_inverse = np.linalg.inv(design.T @ design)
        residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        phi = float(residual @ residual) / (len(y) - design.shape[1])
        expected = {
            label: np.sqrt(phi * np.r_[0.0, c[0]] @ gram_inverse @ np.r_[0.0, c[0]])
            for label, c in _level_contrasts(model, "a")
        }
    else:
        inference = model.term_inference("a")
        expected = {
            f"a[{level}]": se
            for level, se in zip(inference.levels, inference.se_log_relativity)
            if level != "a"
        }
    for summary in (model.summary(), model.metrics(X, y).summary()):
        rows = {r.name: r for r in summary._coef_rows}
        for label, se in expected.items():
            np.testing.assert_allclose(rows[label].se, se, rtol=1e-9)
        assert all(r.estimable for r in summary._coef_rows)
        assert summary._coef_rows[0].se is not None


@pytest.mark.pins_behaviour
def test_a_dropped_ordered_spline_leaves_its_free_level_and_the_intercept_their_errors() -> None:
    """Control (pins): an ordered spline the selection drops adds nothing undetermined.

    Its coefficients are held at zero, although the mask blanks them, so neither its
    base-level effect in the reported intercept nor its share of the free level ``h``'s
    contrast leaves those rows undetermined: both keep their errors.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": np.tile(_TWIN_LABELS, 20), "x": rng.normal(size=160)})
    y = 3.0 * X["x"].to_numpy() + 2.0 * (X["a"] == "h").to_numpy() + rng.normal(size=160)
    spline = Spline(n_knots=4)
    features = {
        "a": OrderedCategorical(order=_TWIN_LABELS, base="a", basis=spline, specials=["h"]),
        "x": Numeric(),
    }
    model = SuperGLM(family="gaussian", features=features, penalty=GroupLasso(lambda1=20.0))
    model.fit(X, y)
    group = next(g for g in model._groups if g.name == "a")
    assert np.all(model.result.beta[group.sl] == 0.0)
    assert not np.any(model._fit_inference_info["coefficient_estimable"][group.sl])
    for summary in (model.summary(), model.metrics(X, y).summary()):
        rows = {r.name: r for r in summary._coef_rows}
        for row in (rows["Intercept"], rows["a[h]"]):
            assert row.se is not None and np.isfinite(row.se) and row.se > 0.0, row.name
            assert row.estimable is True


def test_a_summary_saved_before_levels_were_decided_is_formed_again() -> None:
    """A summary cached before undetermined levels and dictionary errors were blanked is dropped.

    Twin ordered splines (no piecewise term, which #744's hook covered) and twin numerics,
    whose aliased coefficients the dictionary gave errors for: a summary holding what
    71f10edce cached is served again unless the restore drops it.
    """
    spline, _, _ = _twin_fit(["a", "b"], None, basis=_TWIN_BASES["spline"])
    stale = next(r for r in spline.summary()._coef_rows if r.name == "a[b]")
    stale.se, stale.estimable = 1.3189453074629316, True  # what 71f10edce cached
    restored = pickle.loads(pickle.dumps(spline))
    row = next(r for r in restored.summary()._coef_rows if r.name == "a[b]")
    assert row.se is None and row.estimable is False

    x = np.linspace(0.0, 1.0, 160)
    frame = pd.DataFrame({"x": x, "x2": x})
    numeric = SuperGLM(family="gaussian", features={"x": Numeric(), "x2": Numeric()}).fit(
        frame, 2.0 * x + np.sin(np.arange(160.0))
    )
    assert not np.all(numeric.result.rank_info.coefficient_estimable())
    numeric.summary()["standard_errors"]["coefficient_se"]["x2"][:] = 0.1  # a stale figure
    restored = pickle.loads(pickle.dumps(numeric))
    se = restored.summary()["standard_errors"]["coefficient_se"]
    assert np.all(np.isnan(se["x"])) and np.all(np.isnan(se["x2"]))


@pytest.mark.parametrize("term", ["random effect", "fs", "sz"])
def test_structured_fits_decide_each_level_on_their_own_data(term) -> None:
    """A structured fit blanks the levels its data do not determine, and keeps the rest.

    Twin ordered terms beside a random effect, an ``fs`` or an ``sz`` factor smooth: no
    level of either term is determined, and both summaries give none an error, flagged
    ``estimable=False``.  One term keeps the dense fit's level errors.
    """
    twin, X, y = _structured_fit(["a", "b"], term)
    assert twin._fit_inference_info.get("structured_covariance")
    for summary in (twin.summary(), twin.metrics(X, y).summary()):
        rows = {r.name: r for r in summary._coef_rows}
        for name in ("a", "b"):
            for label, contrast in _level_contrasts(twin, name):
                assert rows[label].se is None and rows[label].estimable is False, label
    assert _rank_gain(twin, _level_contrasts(twin, "a")[0][1]) > 0

    single, X, y = _structured_fit(["a"], term)
    dense, _, _ = _structured_fit(["a"], term, direct_solve="gram")
    expected = {r.name: r.se for r in dense.summary()._coef_rows if r.name.startswith("a[")}
    for summary in (single.summary(), single.metrics(X, y).summary()):
        rows = {r.name: r for r in summary._coef_rows}
        for label, _ in _level_contrasts(single, "a"):
            assert rows[label].estimable is True
            np.testing.assert_allclose(rows[label].se, expected[label], rtol=1e-9)


def _whole_term_rows(summary):
    return {r.name: r for r in summary._coef_rows if r.subgroup_type and r.name in ("a", "b")}


@pytest.mark.parametrize("basis", list(_TWIN_BASES))
def test_twin_ordered_terms_carry_no_whole_term_test(basis) -> None:
    """Twin ordered terms: neither whole-term test is formed on any default summary (#740).

    The piecewise row's chi-square tests ``beta_term = 0``, and the spline row's Wood test
    the term's fitted function ``X_term beta_term``; moving the effect between the twins
    leaves the fit, so the coefficients (and those functions) raise the design's rank when
    appended.  At a80058a35 the default summary showed piecewise ``a`` chi2 17,782 (p = 0)
    and the spline twins Wood p 0.092 and 0.059.
    """
    model, X, y = _twin_fit(["a", "b"], None, basis=_TWIN_BASES[basis])
    for name in ("a", "b"):
        group = next(g for g in model._groups if g.name == name)
        design = _design(model)[:, 1 + group.start : 1 + group.end]
        function = np.zeros((len(design), len(model.result.beta)))
        function[:, group.sl] = design
        assert _rank_gain(model, np.eye(len(model.result.beta))[group.sl]) > 0
        assert _rank_gain(model, function) > 0
    other = X.iloc[:120].reset_index(drop=True), y[:120] + 0.5
    for summary in (
        model.summary(),
        model.metrics(X, y).summary(),
        model.metrics(*other).summary(),
    ):
        rows = _whole_term_rows(summary)
        assert set(rows) == {"a", "b"}
        for row in rows.values():
            assert row.wald_chi2 is None and row.wald_p is None and row.ref_df is None, row.name
            assert row.estimable is False, row.name


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("basis", list(_TWIN_BASES))
def test_a_single_ordered_term_keeps_its_whole_term_test(basis) -> None:
    """Control (pins): one unaliased ordered term keeps its whole-term test on both summaries.

    The piecewise row is the Wald chi-square ``b' V^-1 b`` of its unpenalized block, ``V``
    the Gaussian ``phi (X'X)^-1`` block; the spline row is Wood's test, at the figures
    a80058a35 gave (tolerances for the REML fit's convergence, not the test).
    """
    model, X, y = _twin_fit(["a"], None, basis=_TWIN_BASES[basis])
    if basis == "piecewise":
        design = _design(model)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coef
        phi = float(residual @ residual) / (len(y) - design.shape[1])
        V = phi * np.linalg.inv(design.T @ design)[1:, 1:]
        b = coef[1:]
        expected = (float(b @ np.linalg.solve(V, b)), 2.0)
    for summary in (model.summary(), model.metrics(X, y).summary()):
        row = _whole_term_rows(summary)["a"]
        assert row.estimable is True
        if basis == "piecewise":
            np.testing.assert_allclose([row.wald_chi2, row.ref_df], expected, rtol=1e-9)
        else:
            np.testing.assert_allclose(
                [row.wald_chi2, row.ref_df], [85640.68738570434, 4.973720572547127], rtol=1e-6
            )
            np.testing.assert_allclose(
                np.log(row.wald_p), np.log(1.0034931805512945e-209), rtol=1e-4
            )


_NUMERIC_TWIN_BASES = {
    "spline": lambda: Spline(n_knots=6),
    "piecewise": lambda: Piecewise(breaks=[0.5]),
}


def _numeric_twin_fit(names, basis):
    """Numeric terms of one kind on one column, as in #740's numeric case."""
    x = np.tile(np.linspace(0.0, 1.0, 40), 4)
    y = np.sin(3.0 * x) + np.cos(np.arange(160.0))
    X = pd.DataFrame({name: x for name in names})
    features = {name: _NUMERIC_TWIN_BASES[basis]() for name in names}
    return SuperGLM(family="gaussian", features=features).fit(X, y), X, y


def _numeric_test_rows(summary, names):
    return {r.name: r for r in summary._coef_rows if r.coef is None and r.name in names}


@pytest.mark.parametrize("basis", list(_NUMERIC_TWIN_BASES))
def test_twin_numeric_terms_carry_no_whole_term_test(basis) -> None:
    """Twin numeric terms: neither whole-term test is formed on any default summary (#740).

    The ordered rule, on a spline's Wood test of ``X_term beta_term`` and a piecewise
    term's chi-square of ``beta_term = 0``: moving the effect between the twins leaves the
    fit, so both raise the design's rank when appended.  At b7e5b3d98 the default summary
    showed the spline twins Wood p 0.99999 and 0.9997, and piecewise ``u`` chi2 25.2.
    """
    model, X, y = _numeric_twin_fit(["u", "v"], basis)
    for name in ("u", "v"):
        group = next(g for g in model._groups if g.name == name)
        design = _design(model)[:, 1 + group.start : 1 + group.end]
        function = np.zeros((len(design), len(model.result.beta)))
        function[:, group.sl] = design
        assert _rank_gain(model, np.eye(len(model.result.beta))[group.sl]) > 0
        assert _rank_gain(model, function) > 0
    other = X.iloc[:120].reset_index(drop=True), y[:120] + 0.5
    for summary in (
        model.summary(),
        model.metrics(X, y).summary(),
        model.metrics(*other).summary(),
    ):
        rows = _numeric_test_rows(summary, ("u", "v"))
        assert set(rows) == {"u", "v"}
        for row in rows.values():
            assert row.wald_chi2 is None and row.wald_p is None and row.ref_df is None, row.name
            assert row.estimable is False, row.name


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("basis", list(_NUMERIC_TWIN_BASES))
def test_a_single_numeric_term_keeps_its_whole_term_test(basis) -> None:
    """Control (pins): one numeric term keeps its whole-term test on both summaries.

    The piecewise row is the Wald chi-square ``b' V^-1 b`` of its unpenalized block, ``V``
    the Gaussian ``phi (X'X)^-1`` block; the spline row is Wood's test, at the figures
    b7e5b3d98 gave (tolerances for the REML fit's convergence, not the test).
    """
    model, X, y = _numeric_twin_fit(["u"], basis)
    if basis == "piecewise":
        design = _design(model)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coef
        phi = float(residual @ residual) / (len(y) - design.shape[1])
        V = phi * np.linalg.inv(design.T @ design)[1:, 1:]
        b = coef[1:]
        expected = [float(b @ np.linalg.solve(V, b)), 2.0]
    else:
        expected = [25.11705310752398, 6.91538004484533]
    for summary in (model.summary(), model.metrics(X, y).summary()):
        row = _numeric_test_rows(summary, ("u",))["u"]
        assert row.estimable is True
        rtol = 1e-9 if basis == "piecewise" else 1e-6
        np.testing.assert_allclose([row.wald_chi2, row.ref_df], expected, rtol=rtol)
        if basis == "spline":
            np.testing.assert_allclose(row.wald_p, 0.0012851369121335832, rtol=1e-5)


def _three_point_spline(extra=None, *, group=False):
    """``Spline(n_knots=4)`` on a column with three support points, beside ``extra`` numeric
    columns and, with ``group``, a random effect on 40 levels."""
    n = 720 if group else 120
    x = np.tile([0.0, 0.4, 1.0], n // 3)
    X = pd.DataFrame({"x": x, **{name: column(x) for name, column in (extra or {}).items()}})
    y = 2.0 + x + np.sin(np.arange(float(n)))
    features = {"x": Spline(n_knots=4), **{name: Numeric() for name in (extra or {})}}
    if group:
        level = np.tile(np.repeat(np.arange(40), 2), 9)
        X["g"], y = level.astype(str), y + 2.0 * np.cos(level)
        model = SuperGLM(family="gaussian", features={"g": RandomEffect(), **features})
        return model.fit_reml(X, y), X, y
    model = SuperGLM(family="gaussian", features=features, spline_penalty=0)
    return model.fit(X, y), X, y


def _spline_test_rows(model, X, y):
    """Term ``x``'s whole-term row on the default summary and the three metrics summaries."""
    weights = np.linspace(0.5, 1.5, len(y))
    part = slice(0, 3 * len(y) // 4)
    summaries = [
        model.summary(),
        model.metrics(X, y).summary(),
        model.metrics(X, y, sample_weight=weights).summary(),
        model.metrics(X.iloc[part].reset_index(drop=True), y[part]).summary(),
    ]
    return [_numeric_test_rows(summary, ("x",))["x"] for summary in summaries]


def test_a_term_aliased_only_within_itself_keeps_its_whole_term_test() -> None:
    """Three support points alias a spline's coefficients, not its fitted function (#740).

    Every design row lies in the row space, so Wood's test is formed: for this unpenalized
    Gaussian fit it is the one-way layout's F test of the three support points, with
    chi-square ``2 F`` on 2 degrees of freedom.  b2c29ad87 asked each row of the Gram's
    untruncated square root, three of them rounding of norm 1e-8 to 1e-7 and outside the
    row space, and withheld the test on every summary.  The gate reads the null space the
    coefficient mask reads: the function is determined when each null direction keeps to
    the term's coefficients or leaves them alone.

    Beside ``2**-24 x`` as a numeric term the spline's linear part can move to it, however
    small that column is: the function raises the design's rank when appended, and the
    test stays withheld, on a dense fit and beside a random effect.
    """
    from scipy.stats import f as f_distribution

    model, X, y = _three_point_spline()
    assert not np.any(model.result.rank_info.coefficient_estimable())
    cells = [y[X["x"] == value] for value in (0.0, 0.4, 1.0)]
    between = sum(len(cell) * (cell.mean() - y.mean()) ** 2 for cell in cells) / 2
    within = sum(float(np.sum((cell - cell.mean()) ** 2)) for cell in cells) / (len(y) - 3)
    rows = _spline_test_rows(model, X, y)
    for row in rows:
        assert row.estimable is True and row.wald_chi2 > 0.0 and 0.0 < row.wald_p < 1e-4
    for row in rows[:2]:
        np.testing.assert_allclose([row.wald_chi2, row.ref_df], [2 * between / within, 2], 1e-9)
        expected = f_distribution.sf(between / within, 2, len(y) - 3)
        np.testing.assert_allclose(row.wald_p, expected, rtol=1e-6)

    for group in (False, True):
        model, X, y = _three_point_spline({"z": lambda x: 2.0**-24 * x}, group=group)
        term = next(g for g in model._groups if g.name == "x")
        function = np.zeros((len(y), len(model.result.beta)))
        function[:, term.sl] = _design(model)[:, 1 + term.start : 1 + term.end]
        assert _rank_gain(model, function) > 0
        for row in _spline_test_rows(model, X, y):
            assert row.wald_chi2 is None and row.wald_p is None and row.ref_df is None
            assert row.estimable is False


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("group", [False, True], ids=["dense", "random effect"])
def test_twins_elsewhere_do_not_take_a_self_aliased_terms_test(group) -> None:
    """Control (pins): numeric twins beside the three-point spline leave it its test.

    The twins leave the penalized fit uncertified, so the data's mask blanks every
    coefficient the three support points alias; no null direction joins the spline to
    another term, and its test is the one of the fit with one twin, on a dense fit and
    beside a random effect, whose reader keeps only the border rows of the null space.
    b2c29ad87 withheld it on both.
    """
    twin = {"u": lambda x: 0.5 + 0.5 * np.sin(0.7 * np.arange(float(len(x))))}
    model, X, y = _three_point_spline({**twin, "v": twin["u"]}, group=group)
    assert bool(model._fit_inference_info.get("structured_covariance")) is group
    reference, _, _ = _three_point_spline(twin, group=group)
    expected = _numeric_test_rows(reference.summary(), ("x",))["x"]
    for summary in (model.summary(), model.metrics(X, y).summary()):
        row = _numeric_test_rows(summary, ("x",))["x"]
        assert row.estimable is True
        np.testing.assert_allclose(row.wald_chi2, expected.wald_chi2, rtol=1e-6)


def _term_inference_errors(model, name):
    """``term_inference(name)``'s level and curve errors, the curve's at the base removed."""
    inference = model.term_inference(name)
    base = inference.levels.index(model._specs[name]._base_level)
    levels = np.delete(np.asarray(inference.se_log_relativity), base)
    curve = inference.smooth_curve
    base_x = curve.level_x[base]
    return levels, np.asarray(curve.se_log_relativity)[curve.x != base_x], inference


@pytest.mark.parametrize("fit", ["piecewise", "spline", "random effect"])
def test_term_inference_gives_no_error_the_fit_does_not_determine(fit) -> None:
    """Twin ordered terms: ``term_inference`` gives no level or curve error (#740).

    Each non-base level and each curve point away from the base is a contrast
    ``f(x) - f(base)`` the twins can trade (their levels raise the design's rank); the
    summary blanks the levels, and the same rule leaves NaN errors and intervals here,
    on a dense fit and beside a random effect.  At a80058a35 ``term_inference("a")`` gave
    ``a[b]`` 0.1406 (piecewise) and 1.319 (spline).
    """
    if fit == "random effect":
        model, _, _ = _structured_fit(["a", "b"], fit)
    else:
        model, _, _ = _twin_fit(["a", "b"], None, basis=_TWIN_BASES[fit])
    assert _rank_gain(model, _level_contrasts(model, "a")[0][1]) > 0
    for name in ("a", "b"):
        levels, curve, inference = _term_inference_errors(model, name)
        assert np.all(np.isnan(levels)), (name, levels)
        assert np.all(np.isnan(curve)), (name, curve)
        base = inference.levels.index("a")
        assert np.all(np.isnan(np.delete(inference.ci_lower, base))), name
        assert np.isfinite(inference.log_relativity).all()


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("basis", list(_TWIN_BASES))
def test_term_inference_of_a_single_ordered_term_keeps_its_errors(basis) -> None:
    """Control (pins): one unaliased ordered term keeps every ``term_inference`` error.

    The piecewise levels' are ``sqrt(phi c'(X'X)^-1 c)`` of the Gaussian fit, each curve
    vertex is a level, and the spline's are the summary's level errors; every curve error
    is finite and positive away from the base.
    """
    model, X, y = _twin_fit(["a"], None, basis=_TWIN_BASES[basis])
    levels, curve, _ = _term_inference_errors(model, "a")
    if basis == "piecewise":
        design = _design(model)
        gram_inverse = np.linalg.inv(design.T @ design)
        residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        phi = float(residual @ residual) / (len(y) - design.shape[1])
        expected = [
            np.sqrt(phi * np.r_[0.0, c[0]] @ gram_inverse @ np.r_[0.0, c[0]])
            for _, c in _level_contrasts(model, "a")
        ]
    else:
        rows = {r.name: r.se for r in model.summary()._coef_rows}
        expected = [rows[label] for label, _ in _level_contrasts(model, "a")]
    np.testing.assert_allclose(levels, expected, rtol=1e-9)
    assert np.all(np.isfinite(curve)) and np.all(curve > 0.0)


def _relativity_errors(model, name, centering):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = model.relativities(with_se=True, centering=centering)[name]
    return frame.set_index("level")["se_log_relativity"]


@pytest.mark.parametrize("fit", ["piecewise", "spline", "random effect"])
def test_relativities_give_no_level_error_the_fit_does_not_determine(fit) -> None:
    """Twin ordered terms: ``relativities(with_se=True)`` gives no non-base level error (#740).

    It decides each level ``L_l - L_base`` (centred: less the levels' mean) by the rule
    ``summary()`` and ``term_inference()`` use, so the three agree.  At b7e5b3d98 it gave
    ``a[b]`` 0.1406 (piecewise) and 1.319 (spline), and every centred error.
    """
    if fit == "random effect":
        model, _, _ = _structured_fit(["a", "b"], fit)
    else:
        model, _, _ = _twin_fit(["a", "b"], None, basis=_TWIN_BASES[fit])
    for name in ("a", "b"):
        native = _relativity_errors(model, name, "native")
        assert native["a"] == 0.0, name
        assert native.drop("a").isna().all(), (name, native)
        assert _relativity_errors(model, name, "mean").isna().all(), name
        inference = model.term_inference(name)
        np.testing.assert_array_equal(native.to_numpy(), inference.se_log_relativity)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("basis", list(_TWIN_BASES))
def test_relativities_of_a_single_ordered_term_keep_their_errors(basis) -> None:
    """Control (pins): one unaliased ordered term keeps every ``relativities`` level error.

    The native errors are ``term_inference``'s, bit for bit, and the centred ones are
    ``feature_se_from_cov(center=True)``'s; all are finite.
    """
    from superglm.inference._term_covariance import feature_se_from_cov

    model, _, _ = _twin_fit(["a"], None, basis=_TWIN_BASES[basis])
    native = _relativity_errors(model, "a", "native")
    np.testing.assert_array_equal(native.to_numpy(), model.term_inference("a").se_log_relativity)
    covariance, active = model._coef_covariance
    centred = feature_se_from_cov(
        "a",
        covariance,
        active,
        model.result,
        model._groups,
        model._specs,
        model._interaction_specs,
        center=True,
    )
    np.testing.assert_array_equal(_relativity_errors(model, "a", "mean").to_numpy(), centred)
    assert np.isfinite(centred).all() and native.drop("a").gt(0.0).all()


def test_an_intercept_a_design_spanning_the_constant_leaves_open_shows_no_error() -> None:
    """``x`` and ``5 - x``: the columns span the constant, so the intercept is not determined.

    The reported intercept is the centred one, which every fit determines, less the column
    means times the slopes; moving ``c`` from the intercept to ``(x + z) c / 5`` leaves the
    fit.  The default summary and the metrics summaries (fit rows, other weights, other
    rows) keep the estimate with no error, flagged ``estimable=False``, as the robust one
    does; at a80058a35 they showed SE 0.1488.  ``x`` alone keeps the OLS error.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 4.0, 200)
    y = 1.0 + 0.5 * x + rng.normal(size=200)
    X = pd.DataFrame({"x": x, "z": 5.0 - x})
    model = SuperGLM(family="gaussian", features={"x": Numeric(), "z": Numeric()}).fit(X, y)
    _, s, vt = np.linalg.svd(_design(model))
    null = vt[int(np.sum(s > s[0] * 1e-10)) :]
    assert np.linalg.norm(null[:, 0]) > 0.1
    weights = np.linspace(0.5, 1.5, 200)
    for summary in (
        model.summary(),
        model.metrics(X, y).summary(),
        model.metrics(X, y, sample_weight=weights).summary(),
        model.metrics(X.iloc[:150].reset_index(drop=True), y[:150]).summary(),
    ):
        row = summary._coef_rows[0]
        assert row.name == "Intercept" and np.isfinite(row.coef)
        assert row.se is None and row.z is None and row.p is None and row.ci_low is None
        assert row.estimable is False

    alone = SuperGLM(family="gaussian", features={"x": Numeric()}).fit(X[["x"]], y)
    design = _design(alone)
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    phi = float(residual @ residual) / (len(y) - design.shape[1])
    expected = np.sqrt(phi * np.linalg.inv(design.T @ design)[0, 0])
    for summary in (alone.summary(), alone.metrics(X[["x"]], y).summary()):
        row = summary._coef_rows[0]
        assert row.estimable is True
        np.testing.assert_allclose(row.se, expected, rtol=1e-9)


def _intercept_rows(model, X, y):
    """The intercept row of the default summary and of the three metrics summaries, the
    last on the first three quarters of each half of the rows."""
    weights = np.linspace(0.5, 1.5, len(y))
    half, kept = len(y) // 2, 3 * len(y) // 8
    part = np.r_[0:kept, half : half + kept]
    return [
        model.summary()._coef_rows[0],
        model.metrics(X, y).summary()._coef_rows[0],
        model.metrics(X, y, sample_weight=weights).summary()._coef_rows[0],
        model.metrics(X.iloc[part].reset_index(drop=True), y[part]).summary()._coef_rows[0],
    ]


def test_an_intercept_is_decided_with_its_own_weight_not_by_what_a_zero_mean_rounds_to() -> None:
    """Aliased columns that do not span the constant leave the intercept its error (#740).

    ``x`` and ``3 x`` with ``x`` symmetric about zero: the columns are collinear and each
    sums to zero, so the constant is outside their span and the intercept is the OLS one
    of ``y`` on ``x``.  The recorded means, about 4e-18 and 9e-18, are what the sum rounds
    to; b2c29ad87 decided ``-mean' beta`` on its own, relative to that rounding, and
    blanked the intercept.  The reported intercept is the centred one less that product,
    one estimate with unit weight on the centred intercept, and is decided as such, on
    the fit's rows and on a symmetric part of them a metrics reader centres again.

    A spline on three support points aliases its own coefficients; its columns sum to
    zero by construction, to the rounding of the basis.  The intercept is the mean
    response, with the one-way layout's error, which b2c29ad87 blanked too.

    ``x + 2**-20`` beside ``3 x``: now ``3 (x + 2**-20) - 3 x`` is a constant, the columns
    span it, and the intercept stays blank on every summary.
    """
    a = np.random.default_rng(0).integers(-(2**50), 2**50, 100) / 2**50
    x = np.r_[a, -a]
    y = 2.0 + np.sin(np.arange(200.0))
    X = pd.DataFrame({"x": x, "z": 3.0 * x})
    model = SuperGLM(family="gaussian", features={"x": Numeric(), "z": Numeric()}).fit(X, y)
    design = _design(model)
    assert np.linalg.matrix_rank(design) == np.linalg.matrix_rank(design[:, 1:]) + 1
    alone = SuperGLM(family="gaussian", features={"x": Numeric()}).fit(X[["x"]], y)
    ols = _design(alone)
    residual = y - ols @ np.linalg.lstsq(ols, y, rcond=None)[0]
    phi = float(residual @ residual) / (len(y) - ols.shape[1])
    expected = _intercept_rows(alone, X[["x"]], y)
    np.testing.assert_allclose(
        expected[0].se, np.sqrt(phi * np.linalg.inv(ols.T @ ols)[0, 0]), rtol=1e-9
    )
    for row, reference in zip(_intercept_rows(model, X, y), expected):
        assert row.estimable is True
        np.testing.assert_allclose(row.se, reference.se, rtol=1e-9)

    support = [0.0, 0.7, 1.0]
    u = np.tile(support, 40)
    v = 2.0 + u + np.sin(np.arange(120.0))
    frame = pd.DataFrame({"x": u})
    spline = SuperGLM(family="gaussian", features={"x": Spline(n_knots=4)}, spline_penalty=0)
    spline.fit(frame, v)
    assert not np.all(spline.result.rank_info.coefficient_estimable())
    within = sum(float(np.sum((v[u == s] - v[u == s].mean()) ** 2)) for s in support)
    for row in (spline.summary()._coef_rows[0], spline.metrics(frame, v).summary()._coef_rows[0]):
        assert row.estimable is True
        np.testing.assert_allclose(row.coef, v.mean(), rtol=1e-9)
        np.testing.assert_allclose(row.se, np.sqrt(within / (120 - 3) / 120), rtol=1e-9)

    shifted = pd.DataFrame({"x": x + 2.0**-20, "z": 3.0 * x})
    model = SuperGLM(family="gaussian", features={"x": Numeric(), "z": Numeric()}).fit(shifted, y)
    design = _design(model)
    assert np.linalg.matrix_rank(design) == np.linalg.matrix_rank(design[:, 1:])
    for row in _intercept_rows(model, shifted, y):
        assert row.se is None and row.ci_low is None and row.estimable is False


def _curve_contrasts(model, name, x):
    """Spline term ``name``'s curve at ``x`` as rows over the fitted slopes."""
    group = next(g for g in model._groups if g.name == name)
    rows = np.zeros((len(x), len(model.result.beta)))
    rows[:, group.sl] = np.asarray(model._specs[name].transform(np.asarray(x)), dtype=float)
    return rows


def test_curves_of_twin_splines_carry_no_error_on_any_reader() -> None:
    """Twin splines on one column: no point of either curve is determined (issue #752).

    Moving the effect between the twins leaves the fit, so each curve point raises the
    design's rank when appended.  ``term_inference`` and ``relativities(with_se=True)``,
    native and centred, give NaN errors and intervals around a finite curve, no
    simultaneous band is drawn, and the summaries print no curve-error range.  At
    09fe0c01d the readers gave errors from 0.60 to 2.84, as the ordered twins did
    before #740.
    """
    model, X, y = _numeric_twin_fit(["u", "v"], "spline")
    for name in ("u", "v"):
        assert _rank_gain(model, _curve_contrasts(model, name, [0.25, 0.75])) > 0
        for centering in ("native", "mean"):
            inference = model.term_inference(name, centering=centering)
            assert np.all(np.isnan(inference.se_log_relativity)), (name, centering)
            assert np.all(np.isnan(inference.ci_lower)) and np.all(np.isnan(inference.ci_upper))
            assert np.isfinite(inference.log_relativity).all()
            frame = model.relativities(with_se=True, centering=centering)[name]
            assert frame["se_log_relativity"].isna().all(), (name, centering)
        banded = model.term_inference(name, simultaneous=True, n_sim=200)
        assert banded.ci_lower_simultaneous is None and banded.ci_upper_simultaneous is None
    other = X.iloc[:120].reset_index(drop=True), y[:120] + 0.5
    for summary in (
        model.summary(),
        model.metrics(X, y).summary(),
        model.metrics(*other).summary(),
    ):
        for row in _numeric_test_rows(summary, ("u", "v")).values():
            assert np.isnan(row.curve_se_min) and np.isnan(row.curve_se_max), row.name
        assert "curve SE" not in str(summary)


def test_a_self_aliased_spline_keeps_the_curve_errors_its_data_determine() -> None:
    """Three support points determine a spline's curve at those points and nowhere else (#752).

    Each curve point is decided on its own: at a support point it is a design row's
    part, which leaves the design's rank when appended, and keeps its error; between
    them the unpenalized fit has no information, the point raises the rank, and its
    error is NaN.  The centred curve subtracts the grid's mean, which the fit does not
    determine, so every centred error is NaN.  At 09fe0c01d every error was a figure.

    The rank is cut at ``1e-10`` of the largest singular value: the basis is formed to
    about ``1e-13`` of it, where numpy's default cut also sits, and a point between the
    support points adds a singular value of the design's own size.
    """
    model, X, _ = _three_point_spline()
    inference = model.term_inference("x", n_points=11)
    support = np.isin(inference.x, [0.0, 0.4, 1.0])
    assert support.sum() == 3
    gains = [_rank_gain(model, _curve_contrasts(model, "x", [x]), 1e-10) for x in inference.x]
    np.testing.assert_array_equal(np.asarray(gains) == 0, support)
    errors = np.asarray(inference.se_log_relativity)
    np.testing.assert_array_equal(np.isfinite(errors), support)
    assert np.all(errors[support] > 0.0)
    centred = model.term_inference("x", n_points=11, centering="mean").se_log_relativity
    assert np.all(np.isnan(centred))


@pytest.mark.pins_behaviour
def test_a_single_spline_keeps_its_curve_errors() -> None:
    """Control (pins): one unaliased spline keeps every curve error.

    ``term_inference`` and ``relativities`` give ``sqrt(diag(M V M'))`` with ``M`` the
    basis on the grid (less its column means when centred) and ``V`` the term's
    covariance block, to the rounding of that product; the summary prints their range,
    the one 09fe0c01d gave, and the simultaneous band is drawn.
    """
    model, X, y = _numeric_twin_fit(["u"], "spline")
    covariance, active = model._coef_covariance
    block = covariance[np.ix_(*[np.arange(active[0].start, active[0].end)] * 2)]
    inference = model.term_inference("u")
    M = np.asarray(model._specs["u"].transform(inference.x), dtype=float)
    for centering, basis in (("native", M), ("mean", M - M.mean(axis=0))):
        expected = np.sqrt(np.maximum(np.sum((basis @ block) * basis, axis=1), 0.0))
        got = model.term_inference("u", centering=centering).se_log_relativity
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        frame = model.relativities(with_se=True, centering=centering)["u"]
        np.testing.assert_array_equal(frame["se_log_relativity"].to_numpy(), got)
    native = np.asarray(inference.se_log_relativity)
    np.testing.assert_allclose(
        [native.min(), native.max()], [0.12079868655384633, 0.2486961468140573], rtol=1e-6
    )
    banded = model.term_inference("u", simultaneous=True, n_sim=200)
    assert np.all(banded.ci_upper_simultaneous >= banded.ci_upper)
    for summary in (model.summary(), model.metrics(X, y).summary()):
        row = _numeric_test_rows(summary, ("u",))["u"]
        np.testing.assert_allclose(
            [row.curve_se_min, row.curve_se_max], [native.min(), native.max()], rtol=1e-9
        )


def test_a_centred_curve_the_fit_determines_keeps_its_simultaneous_band() -> None:
    """Centring can determine every point of a curve whose own points are open (issue #752).

    ``z = (x - 0.5)**2`` beside an unpenalized spline on ``x`` lies in the spline's span,
    so the curve is open by a multiple of it.  On the two endpoints that multiple is one
    constant, which the grid's mean removes: each native point raises the design's rank
    and has no error, each centred point leaves it and has ``sqrt(diag(M V M'))`` on the
    centred map.  The band is of the reported curve, so ``centering="mean"`` draws it.
    At 07ceaec50 the band was decided on the native errors, and the centred report,
    0.2477 at both ends, had none.
    """
    x = np.tile(np.linspace(0.0, 1.0, 33), 4)
    X = pd.DataFrame({"x": x, "z": (x - 0.5) ** 2})
    y = np.sin(3.0 * x) + np.cos(np.arange(float(len(x))))
    features = {"x": Spline(kind="ps", n_knots=6), "z": Numeric()}
    model = SuperGLM(family="gaussian", features=features, spline_penalty=0).fit(X, y)
    ends = _curve_contrasts(model, "x", [0.0, 1.0])
    assert [_rank_gain(model, row[None], 1e-10) for row in ends] == [1, 1]
    assert _rank_gain(model, ends - ends.mean(axis=0), 1e-10) == 0

    native = model.term_inference("x", n_points=2, simultaneous=True, n_sim=200)
    assert np.all(np.isnan(native.se_log_relativity))
    assert native.ci_lower_simultaneous is None and native.critical_value_simultaneous is None

    centred = model.term_inference("x", n_points=2, simultaneous=True, n_sim=200, centering="mean")
    covariance, active = model._coef_covariance
    block = covariance[np.ix_(*[np.arange(active[0].start, active[0].end)] * 2)]
    M = np.asarray(model._specs["x"].transform(centred.x), dtype=float)
    M = M - M.mean(axis=0)
    expected = np.sqrt(np.sum((M @ block) * M, axis=1))
    np.testing.assert_allclose(centred.se_log_relativity, expected, rtol=1e-12)
    np.testing.assert_allclose(expected, [0.2477361, 0.2477361], rtol=1e-6)
    assert np.isfinite(centred.critical_value_simultaneous)
    half = centred.critical_value_simultaneous * expected
    upper, lower = np.exp(centred.log_relativity + half), np.exp(centred.log_relativity - half)
    np.testing.assert_allclose(centred.ci_upper_simultaneous, upper, rtol=1e-12)
    np.testing.assert_allclose(centred.ci_lower_simultaneous, lower, rtol=1e-12)


def _population_fit(names, direct_solve, bound=None):
    """Splines ``names`` on one column beside an ``sz`` smooth of ``x`` by four levels, every
    lambda fixed.  The last level has one row: it is thin and left out of the population,
    so the reported curve of ``x`` is the population curve ``main(x) + b(x)' c``.  ``bound``
    adds a term the smooth shares its levels with: an ordered ``"group term"`` on ``g``, or a
    ``"second smooth"`` of another column by ``g``."""
    x = np.r_[np.tile(np.linspace(0.0, 1.0, 30), 3), 0.4]
    g = np.array([*np.repeat(["a", "b", "c"], 30), "d"])
    row = np.arange(float(len(x)))
    y = np.sin(3.0 * x) + 0.3 * (g == "b") * x + 0.2 * np.cos(row)
    X = pd.DataFrame({name: x for name in names}).assign(g=g, w=np.cos(1.7 * row))
    features = {name: Spline(n_knots=5, lambda_policy=LambdaPolicy.fixed(1.2)) for name in names}
    smooths = ["x"]
    if bound == "group term":
        features["g"] = OrderedCategorical(order=list("abcd"), basis=Polynomial(degree=2))
    if bound == "second smooth":
        features["w"] = Spline(n_knots=5, lambda_policy=LambdaPolicy.fixed(1.2))
        smooths.append("w")
    wiggle = {"wiggle": LambdaPolicy.fixed(1.0)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the thin level, and the group term it repeats
        model = SuperGLM(
            family="gaussian",
            features=features,
            interactions=[
                FactorSmooth(name, group="g", basis="sz", k=5, lambda_policy=wiggle)
                for name in smooths
            ],
            selection_penalty=0,
            direct_solve=direct_solve,
        ).fit_reml(X, y)
    assert bool(model._fit_inference_info.get("structured_covariance")) == (
        direct_solve == "structured"
    )
    assert model._interaction_specs["x:g:sz"]._has_population_offset
    return model


def _assert_population_errors_are_the_contrasts(model) -> np.ndarray:
    """``x``'s reported errors are ``sqrt(diag(G V G'))`` on its population contrast ``G``,
    less its column means when centred, to the rounding of that sum of products
    (``gamma_(2 p)`` of ``|G| |V| |G|'``, Higham 2002, section 3.5); returns the native ones."""
    covariance = model._coef_covariance[0]
    width = covariance.shape[0]
    solve = getattr(covariance, "solve", None)
    V = np.asarray(covariance if solve is None else solve(np.eye(width)), dtype=float)
    inference = model.term_inference("x")
    G = _population_contrasts(model, inference.x, active=True)
    for centering, contrast in (("native", G), ("mean", G - G.mean(axis=0))):
        variance = np.einsum("ij,jk,ik->i", contrast, V, contrast)
        reach = np.einsum("ij,jk,ik->i", np.abs(contrast), np.abs(V), np.abs(contrast))
        got = np.asarray(model.term_inference("x", centering=centering).se_log_relativity)
        assert np.all(np.abs(got * got - variance) <= 8 * width * np.finfo(float).eps * reach)
        frame = model.relativities(with_se=True, centering=centering)["x"]
        np.testing.assert_array_equal(frame["se_log_relativity"].to_numpy(), got)
    native = np.asarray(inference.se_log_relativity)
    assert np.all(native > 0.0)
    return native


def _population_contrasts(model, grid, active=False):
    """The population curve of ``x`` on ``grid`` as rows over the fitted slopes (the active
    covariance's with ``active``): its basis, and the smooth's times its population contrast."""
    term = model._interaction_specs["x:g:sz"]
    groups = model._coef_covariance[1] if active else model._groups
    rows = np.zeros((len(grid), max(group.end for group in groups)))
    for group in groups:
        if group.feature_name == "x":
            rows[:, group.start : group.end] = model._specs["x"].transform(np.asarray(grid))
        if group.feature_name == "x:g:sz":
            rows[:, group.start : group.end] = term.marginal_basis(np.asarray(grid)) @ (
                term._population_contrast()
            )
    return rows


@pytest.mark.parametrize("direct_solve", ["structured", "gram"])
def test_a_population_curve_traded_against_a_twin_carries_no_error(direct_solve) -> None:
    """A curve an ``sz`` term moves onto its population curve is decided too (issue #752).

    ``x`` beside its copy ``z``: the reported curve of ``x`` is a contrast of its main
    term and the smooth together.  Moving the effect between ``x`` and ``z`` leaves the
    fit and moves that curve, so each of its points raises the design's rank, as each
    point of ``z`` does.  Both terms give NaN errors on every reader, native and centred,
    no simultaneous band and no curve-error range.  At 07ceaec50 ``x`` returned its
    errors before the rule was read: 200 figures, a band and a range.
    """
    model = _population_fit(["x", "z"], direct_solve)
    grid = np.linspace(0.0, 1.0, 5)
    for rows in (_population_contrasts(model, grid), _curve_contrasts(model, "z", grid)):
        assert [_rank_gain(model, row[None], 1e-10) for row in rows] == [1] * len(grid)
    for name in ("x", "z"):
        for centering in ("native", "mean"):
            inference = model.term_inference(name, centering=centering)
            assert np.all(np.isnan(inference.se_log_relativity)), (name, centering)
            assert np.all(np.isnan(inference.ci_lower)) and np.all(np.isnan(inference.ci_upper))
            assert np.isfinite(inference.log_relativity).all()
            frame = model.relativities(with_se=True, centering=centering)[name]
            assert frame["se_log_relativity"].isna().all(), (name, centering)
        banded = model.term_inference(name, simultaneous=True, n_sim=200)
        assert banded.ci_lower_simultaneous is None and banded.critical_value_simultaneous is None
    summary = model.summary()
    for row in _numeric_test_rows(summary, ("x", "z")).values():
        assert np.isnan(row.curve_se_min) and np.isnan(row.curve_se_max), row.name
    assert "curve SE" not in str(summary)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("direct_solve", ["structured", "gram"])
def test_a_population_curve_keeps_the_errors_of_its_contrast(direct_solve) -> None:
    """Control (pins): a population curve no other term is traded against keeps every error.

    The thin level leaves two directions open, both inside ``x`` and its smooth, and the
    mask shows none of ``x``'s coefficients: the curve is decided as the fitted function
    of the terms its smooth binds, which no other term shares.  Its errors are those of
    the population contrast (``_assert_population_errors_are_the_contrasts``), the band is
    drawn and the summary prints the range, as at 07ceaec50.
    """
    model = _population_fit(["x"], direct_solve)
    group = next(g for g in model._groups if g.name == "x")
    assert not np.any(model._fit_inference_info["coefficient_estimable"][group.sl])
    native = _assert_population_errors_are_the_contrasts(model)
    banded = model.term_inference("x", simultaneous=True, n_sim=200)
    assert np.all(banded.ci_upper_simultaneous >= banded.ci_upper)
    row = _numeric_test_rows(model.summary(), ("x",))["x"]
    np.testing.assert_allclose(
        [row.curve_se_min, row.curve_se_max], [native.min(), native.max()], rtol=1e-9
    )


@pytest.mark.pins_behaviour
@pytest.mark.parametrize(
    ("bound", "direct_solve"),
    [("group term", "structured"), ("group term", "gram"), ("second smooth", "gram")],
)
def test_a_population_curve_keeps_its_errors_beside_the_terms_its_smooth_shares(
    bound, direct_solve
) -> None:
    """Control (pins): the terms an ``sz`` smooth shares its levels with do not blank its curve.

    An ordered term on the group repeats each level's constant, and a second ``sz`` smooth
    on the group shares them (``share_group_levels``): null directions of the data then
    run between ``x``'s smooth and those terms, so the smooth and its main term alone are
    not a function the data determine.  The curve is decided on the whole family
    (``population_curve_family``), which no other term is traded against, and keeps the
    errors of its contrast, as at 07ceaec50.
    """
    model = _population_fit(["x"], direct_solve, bound)
    marked = np.zeros(len(model.result.beta), dtype=bool)
    for group in model._groups:
        marked[group.sl] = group.feature_name in ("x", "x:g:sz")
    decide = model._solver_pirls_result().rank_info
    if decide is not None:
        assert not decide.is_estimable(marked, design_rows=True)
    _assert_population_errors_are_the_contrasts(model)


@pytest.mark.pins_behaviour
def test_a_population_curve_closes_its_family_across_connected_sz_groups() -> None:
    """Control (pins): connected ``sz`` groups keep an unaliased curve's errors (#752).

    ``sz(x, g)``, ``sz(w, g)``, and ``sz(w, h)`` with thin levels in both ``g`` and ``h``
    couple ``sz(x, g)``'s level constants to ``sz(w, g)`` and ``sz(w, h)``.  Because all
    null directions split cleanly into those level constants plus the other terms' blocks,
    ``x``'s curve stays determined and keeps its SEs, band, and summary range.
    """
    x = np.r_[np.tile(np.linspace(0.0, 1.0, 30), 3), 0.4]
    g = np.array([*np.repeat(["a", "b", "c"], 30), "d"])
    row = np.arange(float(len(x)))
    h = np.where(row == 17, "r", np.where(row % 2 == 0, "p", "q"))
    w = np.cos(1.7 * row)
    y = np.sin(3.0 * x) + 0.3 * (g == "b") * x + 0.2 * np.sin(2.5 * w) + 0.1 * np.cos(row)
    X = pd.DataFrame({"x": x, "w": w, "g": g, "h": h})
    wiggle = {"wiggle": LambdaPolicy.fixed(1.0)}
    fixed = LambdaPolicy.fixed(1.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(
            family="gaussian",
            features={
                "x": Spline(n_knots=5, lambda_policy=fixed),
                "w": Spline(n_knots=5, lambda_policy=fixed),
            },
            interactions=[
                FactorSmooth("x", group="g", basis="sz", k=5, lambda_policy=wiggle),
                FactorSmooth("w", group="g", basis="sz", k=5, lambda_policy=wiggle),
                FactorSmooth("w", group="h", basis="sz", k=5, lambda_policy=wiggle),
            ],
            selection_penalty=0,
            direct_solve="gram",
        ).fit_reml(X, y)
    assert model._interaction_specs["w:h:sz"]._has_population_offset
    native = _assert_population_errors_are_the_contrasts(model)
    banded = model.term_inference("x", simultaneous=True, n_sim=200)
    assert np.all(banded.ci_upper_simultaneous >= banded.ci_upper)
    row_x = _numeric_test_rows(model.summary(), ("x",))["x"]
    np.testing.assert_allclose(
        [row_x.curve_se_min, row_x.curve_se_max], [native.min(), native.max()], rtol=1e-9
    )


@pytest.mark.parametrize("chained", [False, True], ids=["two-smooth", "three-smooth-chain"])
def test_a_population_curve_keeps_its_errors_when_only_a_sibling_sz_term_is_aliased(
    chained: bool,
) -> None:
    """An alias confined to sibling term ``w`` leaves ``x``'s population curve determined (#752).

    ``x`` and ``w`` each carry a thin-level ``sz`` smooth on group ``g`` (and ``sz(w, h)`` when
    ``chained``), while ``z = w`` aliases only ``w``'s linear component.  Previously, closing
    over ``population_curve_family`` pulled ``w`` into ``x``'s test and blanked ``x``; now
    ``x`` keeps all 200 errors, band, and summary range while ``w`` is blanked.
    """
    x = np.r_[np.tile(np.linspace(0.0, 1.0, 30), 3), 0.4]
    g = np.array([*np.repeat(["a", "b", "c"], 30), "d"])
    row = np.arange(float(len(x)))
    h = np.where(row == 17, "r", np.where(row % 2 == 0, "p", "q"))
    w = np.cos(1.7 * row)
    y = np.sin(3.0 * x) + 0.3 * (g == "b") * x + 0.2 * np.sin(2.5 * w) + 0.1 * np.cos(row)
    wiggle = {"wiggle": LambdaPolicy.fixed(1.0)}
    fixed = LambdaPolicy.fixed(1.2)
    interactions = [
        FactorSmooth("x", group="g", basis="sz", k=5, lambda_policy=wiggle),
        FactorSmooth("w", group="g", basis="sz", k=5, lambda_policy=wiggle),
    ]
    base_data = {"x": x, "w": w, "g": g}
    if chained:
        interactions.append(FactorSmooth("w", group="h", basis="sz", k=5, lambda_policy=wiggle))
        base_data["h"] = h
    base_features = {
        "x": Spline(n_knots=5, lambda_policy=fixed),
        "w": Spline(n_knots=5, lambda_policy=fixed),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = SuperGLM(
            family="gaussian",
            features=base_features,
            interactions=interactions,
            selection_penalty=0,
            direct_solve="gram",
        ).fit_reml(pd.DataFrame(base_data), y)
        model = SuperGLM(
            family="gaussian",
            features={**base_features, "z": Numeric()},
            interactions=interactions,
            selection_penalty=0,
            direct_solve="gram",
        ).fit_reml(pd.DataFrame({**base_data, "z": w}), y)

    native = _assert_population_errors_are_the_contrasts(model)
    ref_ti = ref.term_inference("x", centering="mean")
    mod_ti = model.term_inference("x", centering="mean")
    np.testing.assert_allclose(mod_ti.log_relativity, ref_ti.log_relativity, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        mod_ti.se_log_relativity, ref_ti.se_log_relativity, rtol=1e-10, atol=1e-12
    )
    banded_x = model.term_inference("x", simultaneous=True, n_sim=200)
    assert np.all(banded_x.ci_upper_simultaneous >= banded_x.ci_upper)
    row_x = _numeric_test_rows(model.summary(), ("x",))["x"]
    np.testing.assert_allclose(
        [row_x.curve_se_min, row_x.curve_se_max], [native.min(), native.max()], rtol=1e-9
    )

    for centering in ("native", "mean"):
        inf_w = model.term_inference("w", centering=centering)
        assert np.all(np.isnan(inf_w.se_log_relativity))
        frame_w = model.relativities(with_se=True, centering=centering)["w"]
        assert frame_w["se_log_relativity"].isna().all()
    banded_w = model.term_inference("w", simultaneous=True, n_sim=200)
    assert banded_w.ci_lower_simultaneous is None and banded_w.critical_value_simultaneous is None
    row_w = _numeric_test_rows(model.summary(), ("w",))["w"]
    assert np.isnan(row_w.curve_se_min) and np.isnan(row_w.curve_se_max)


def test_a_population_curve_blanks_twin_variables_sharing_an_sz_group() -> None:
    """Mixed/twin external aliases blank population curves while pure group constants do not (#752).

    - Twin variables ``w = x`` with ``sz(x, g)`` + ``sz(w, g)`` leave both curves open -> both blank.
    - Mixed alias ``z = x + 1_{g=='a'}`` beside ``x + sz(x, g)`` couples ``x`` to ``z`` -> blank.
    - Pure group constant ``z = 1_{g=='a'}`` couples only to level constants -> keeps ``x``'s errors.
    """
    x = np.r_[np.tile(np.linspace(0.0, 1.0, 30), 3), 0.4]
    g = np.array([*np.repeat(["a", "b", "c"], 30), "d"])
    row = np.arange(float(len(x)))
    y = np.sin(3.0 * x) + 0.3 * (g == "b") * x + 0.1 * np.cos(row)
    wiggle = {"wiggle": LambdaPolicy.fixed(1.0)}
    fixed = LambdaPolicy.fixed(1.2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        twin = SuperGLM(
            family="gaussian",
            features={
                "x": Spline(n_knots=5, lambda_policy=fixed),
                "w": Spline(n_knots=5, lambda_policy=fixed),
            },
            interactions=[
                FactorSmooth("x", group="g", basis="sz", k=5, lambda_policy=wiggle),
                FactorSmooth("w", group="g", basis="sz", k=5, lambda_policy=wiggle),
            ],
            selection_penalty=0,
            direct_solve="gram",
        ).fit_reml(pd.DataFrame({"x": x, "w": x, "g": g}), y)
        mixed = SuperGLM(
            family="gaussian",
            features={"x": Spline(n_knots=5, lambda_policy=fixed), "z": Numeric()},
            interactions=[FactorSmooth("x", group="g", basis="sz", k=5, lambda_policy=wiggle)],
            selection_penalty=0,
            direct_solve="gram",
        ).fit_reml(pd.DataFrame({"x": x, "z": x + (g == "a").astype(float), "g": g}), y)
        base = SuperGLM(
            family="gaussian",
            features={"x": Spline(n_knots=5, lambda_policy=fixed)},
            interactions=[FactorSmooth("x", group="g", basis="sz", k=5, lambda_policy=wiggle)],
            selection_penalty=0,
            direct_solve="gram",
        ).fit_reml(pd.DataFrame({"x": x, "g": g}), y)
        pure_g = SuperGLM(
            family="gaussian",
            features={"x": Spline(n_knots=5, lambda_policy=fixed), "z": Numeric()},
            interactions=[FactorSmooth("x", group="g", basis="sz", k=5, lambda_policy=wiggle)],
            selection_penalty=0,
            direct_solve="gram",
        ).fit_reml(pd.DataFrame({"x": x, "z": (g == "a").astype(float), "g": g}), y)

    for model, names in ((twin, ("x", "w")), (mixed, ("x",))):
        rows = _numeric_test_rows(model.summary(), names)
        for name in names:
            for centering in ("native", "mean"):
                assert np.all(
                    np.isnan(model.term_inference(name, centering=centering).se_log_relativity)
                )
                assert (
                    model.relativities(with_se=True, centering=centering)[name]["se_log_relativity"]
                    .isna()
                    .all()
                )
            assert np.isnan(rows[name].curve_se_min) and np.isnan(rows[name].curve_se_max)

    native = _assert_population_errors_are_the_contrasts(pure_g)
    ref_ti = base.term_inference("x", centering="mean")
    got_ti = pure_g.term_inference("x", centering="mean")
    np.testing.assert_allclose(got_ti.log_relativity, ref_ti.log_relativity, rtol=1e-10, atol=1e-12)
    row_g = _numeric_test_rows(pure_g.summary(), ("x",))["x"]
    np.testing.assert_allclose(
        [row_g.curve_se_min, row_g.curve_se_max], [native.min(), native.max()], rtol=1e-9
    )


_INTERACTION_KINDS = {
    "spline by level": (lambda: Spline(n_knots=5), "c", "wood"),
    "tensor": (lambda: Spline(n_knots=4), "w", "wood"),
    "polynomial by level": (lambda: Polynomial(degree=2), "c", "chi2"),
    "polynomial by polynomial": (lambda: Polynomial(degree=3), "w", "chi2"),
}


def _interaction_fit(kind, twin):
    """``x`` crossed with a second term, as each kind of interaction with a whole-term
    test; with ``twin``, a second copy ``x2`` of ``x`` crossed with it too."""
    main, other, _ = _INTERACTION_KINDS[kind]
    rng = np.random.default_rng(0)
    x = np.tile(np.linspace(0.0, 1.0, 40), 12)
    c = np.tile(np.repeat(["p", "q", "r"], 40), 4)
    w = rng.uniform(0.0, 1.0, len(x))
    y = np.sin(3.0 * x) * (1.0 + (c == "q")) + 0.5 * w + np.cos(np.arange(float(len(x))))
    names = ["x", "x2"] if twin else ["x"]
    X = pd.DataFrame({name: x for name in names})
    features = {name: main() for name in names}
    if other == "c":
        X["c"], features["c"] = c, Categorical()
    else:
        X["w"] = w
        features["w"] = Spline(n_knots=4) if kind == "tensor" else Polynomial(degree=2)
    model = SuperGLM(
        family="gaussian", features=features, interactions=[(name, other) for name in names]
    )
    return model.fit(X, y), X, y


def _interaction_rows(model, summary):
    names = {g.name for g in model._groups if g.feature_name in model._interaction_specs}
    return {r.name: r for r in summary._coef_rows if r.coef is None and r.name in names}


@pytest.mark.parametrize("kind", list(_INTERACTION_KINDS))
def test_twin_interactions_carry_no_whole_term_test(kind) -> None:
    """Twin interactions: no whole-term test is formed on any default summary (issue #752).

    ``x:c`` beside ``x2:c`` on a copy of ``x`` (and likewise against ``w``): moving the
    effect between the twins leaves the fit.  Wood's test of a by-level spline or a
    tensor is of the block's fitted function, and the chi-square of a by-level
    polynomial or a large polynomial product is of ``beta_block = 0``; both raise the
    design's rank when appended.  The rule is the one inside the two test helpers, which
    these four producers reach without stating it.  At 09fe0c01d the default summary
    showed Wood p 0.999 on ``x:c[q]`` and 1.0 on ``x:w``, chi2 20.65 (p = 3e-5) on the
    polynomial ``x:c[q]`` and 0.657 on the polynomial ``x:w``, with NaN on their twins;
    the robust summary left the two chi-squares blank but flagged them estimable.
    """
    model, X, y = _interaction_fit(kind, twin=True)
    design = _design(model)
    width = len(model.result.beta)
    for group in model._groups:
        if group.feature_name not in model._interaction_specs:
            continue
        function = np.zeros((len(design), width))
        function[:, group.sl] = design[:, 1 + group.start : 1 + group.end]
        tested = function if _INTERACTION_KINDS[kind][2] == "wood" else np.eye(width)[group.sl]
        assert _rank_gain(model, tested) > 0, group.name
    other = X.iloc[:360].reset_index(drop=True), y[:360] + 0.5
    summaries = [model.summary(), model.metrics(X, y).summary(), model.metrics(*other).summary()]
    if _INTERACTION_KINDS[kind][2] == "chi2":
        summaries.append(model.summary(covariance=model.sandwich_covariance()))
    for summary in summaries:
        rows = _interaction_rows(model, summary)
        assert len(rows) == (4 if _INTERACTION_KINDS[kind][1] == "c" else 2)
        for row in rows.values():
            assert row.wald_chi2 is None and row.wald_p is None and row.ref_df is None, row.name
            assert row.estimable is False, row.name


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("kind", list(_INTERACTION_KINDS))
def test_a_single_interaction_keeps_its_whole_term_test(kind) -> None:
    """Control (pins): one unaliased interaction keeps its whole-term test on both summaries.

    The polynomial rows are the Wald chi-square ``b' V^-1 b`` of their unpenalized block,
    ``V`` the Gaussian ``phi (X'X)^-1`` block; the spline rows are Wood's test, at the
    figures 09fe0c01d gave.
    """
    model, X, y = _interaction_fit(kind, twin=False)
    wood = {
        "spline by level": {
            "x:c[q]": (21.01900558308689, 5.834075571440447),
            "x:c[r]": (3.2500155228990493, 5.834075571440447),
        },
        "tensor": {"x:w": (11.219280813499378, 24.983943859133674)},
    }
    design = _design(model)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    V = float(residual @ residual) / (len(y) - design.shape[1]) * np.linalg.inv(design.T @ design)
    for summary in (model.summary(), model.metrics(X, y).summary()):
        rows = _interaction_rows(model, summary)
        for group in model._groups:
            if group.feature_name not in model._interaction_specs:
                continue
            row = rows[group.name]
            assert row.estimable is True
            if kind in wood:
                np.testing.assert_allclose(
                    [row.wald_chi2, row.ref_df], wood[kind][group.name], rtol=1e-6
                )
                continue
            block = slice(1 + group.start, 1 + group.end)
            expected = float(coef[block] @ np.linalg.solve(V[block, block], coef[block]))
            np.testing.assert_allclose([row.wald_chi2, row.ref_df], [expected, group.size], 1e-8)


_INTERCEPT_COLUMNS = {
    # x and 3 - x span the constant
    "spanning": lambda x, a: {"x": x, "z": 3.0 - x},
    # a and 3 a, symmetric about zero, are collinear and do not span it
    "collinear": lambda x, a: {"x": a, "z": 3.0 * a},
    "alone": lambda x, a: {"x": x},
}


def _intercept_fit(term, columns, direct_solve="auto"):
    """Numeric terms on ``columns`` beside a structured ``term`` on 40 levels (``_structured_fit``'s)."""
    rng = np.random.default_rng(0)
    n = 720
    level = np.tile(np.repeat(np.arange(40), 2), 9)
    half = rng.integers(-(2**50), 2**50, n // 2) / 2**50
    x = rng.uniform(0.0, 3.0, n)
    z = 0.5 + 0.5 * np.sin(0.7 * np.arange(float(n)))
    X = pd.DataFrame({"g": level.astype(str), "s": z})
    numeric = _INTERCEPT_COLUMNS[columns](x, np.r_[half, -half])
    X = X.assign(**numeric)
    y = 1.0 + 0.5 * x + 2.0 * np.cos(level) + z + rng.normal(size=n)
    features: dict = {"g": RandomEffect()} if term == "random effect" else {}
    features.update({name: Numeric() for name in numeric})
    interactions = []
    if term != "random effect":
        features["s"] = Spline(n_knots=5)
        interactions.append(FactorSmooth("s", group="g", basis=term, k=5))
    model = SuperGLM(
        family="gaussian", features=features, interactions=interactions, direct_solve=direct_solve
    )
    return model.fit_reml(X, y), X, y


def _intercept_share_of_the_null_space(model):
    """The largest weight a null direction of ``[1, X]`` puts on the intercept."""
    _, s, vt = np.linalg.svd(_design(model))
    null = vt[int(np.sum(s > s[0] * 1e-10)) :]
    return float(np.max(np.abs(null[:, 0]), initial=0.0))


@pytest.mark.parametrize(
    ("term", "columns"),
    [
        ("random effect", "spanning"),
        ("fs", "spanning"),
        ("sz", "spanning"),
        ("random effect", "collinear"),
        ("fs", "collinear"),
    ],
)
def test_structured_fits_show_no_error_on_an_intercept_their_data_leave_open(term, columns) -> None:
    """A structured fit decides its reported intercept on its own data, as a dense fit does (#752).

    The reported intercept weighs every column's mean, the structured ones included, which
    the compact geometry read as outside its border and left undecided: only the ordered
    terms' base-level part was.  Beside ``x`` and ``3 - x`` the columns span the constant;
    beside any aliased pair, a random effect's or an ``fs`` smooth's levels, which only
    their penalty holds, alias it too.  A null direction of ``[1, X]`` then moves the
    intercept, and the default and metrics summaries keep its estimate with no error,
    flagged ``estimable=False``, the dense fit's answer.  At 09fe0c01d the random effect
    beside ``x`` and ``3 - x`` showed SE 0.233.
    """
    model, X, y = _intercept_fit(term, columns)
    assert model._fit_inference_info.get("structured_covariance")
    assert _intercept_share_of_the_null_space(model) > 0.01
    dense, _, _ = _intercept_fit(term, columns, direct_solve="gram")
    assert not dense._fit_inference_info.get("structured_covariance")
    assert dense.summary()._coef_rows[0].se is None
    for summary in (model.summary(), model.metrics(X, y).summary()):
        row = summary._coef_rows[0]
        assert row.name == "Intercept" and np.isfinite(row.coef)
        assert row.se is None and row.z is None and row.p is None and row.ci_low is None
        assert row.estimable is False


@pytest.mark.pins_behaviour
@pytest.mark.parametrize(
    ("term", "columns"),
    [("sz", "collinear"), ("random effect", "alone"), ("fs", "alone"), ("sz", "alone")],
)
def test_structured_fits_keep_an_intercept_they_determine(term, columns) -> None:
    """Control (pins): a structured fit keeps the dense fit's intercept error where it is determined.

    Beside one numeric term the penalized fit is certified and every error is shown.  An
    ``sz`` smooth's levels sum to zero, so beside a collinear pair that does not span the
    constant no null direction of ``[1, X]`` moves the intercept: the geometry decides
    the columns' mean on its level coordinates and keeps the error.
    """
    model, X, y = _intercept_fit(term, columns)
    assert model._fit_inference_info.get("structured_covariance")
    if columns == "collinear":
        assert not np.all(model._fit_inference_info["coefficient_estimable"])
        assert _intercept_share_of_the_null_space(model) < 1e-8
    dense, _, _ = _intercept_fit(term, columns, direct_solve="gram")
    expected = dense.summary()._coef_rows[0]
    assert expected.estimable is True
    for summary in (model.summary(), model.metrics(X, y).summary()):
        row = summary._coef_rows[0]
        assert row.estimable is True
        np.testing.assert_allclose(row.se, expected.se, rtol=1e-9)


@pytest.mark.parametrize("multiple", [30, 35])
def test_an_sz_fit_sizes_its_intercept_in_the_centred_columns(multiple) -> None:
    """An ``sz`` fit decides its reported intercept as a dense fit does (issue #752).

    ``x`` near 16 and ``z = shift - x``: ``x + z`` is the constant ``shift``, so one null
    direction of ``[1, X]`` moves the intercept by ``shift``.  In the centred columns'
    equilibrated coordinates that is a share ``shift / (sqrt(2) s)`` of the estimate's
    size ``hypot(sqrt(2) m / s, 1 / sqrt(n))``, with ``m`` the columns' mean and ``s``
    their centred norm; the rule's cut is ``factor_rcond``.  The share is 0.94 of the cut
    at ``30 * 2**-26`` and 1.09 at ``35 * 2**-26``: shown at the first, withheld at the
    second, on both backends.  At 07ceaec50 the tree sized the contrast in the uncentred
    columns, where the intercept's own weight is a third of the size, and showed both.
    """
    shift, n = multiple * 2.0**-26, 160
    row = np.arange(float(n))
    x = 16.0 + np.tile(np.arange(-4, 4) + 0.5, 20) / 4.0
    s = np.tile(np.linspace(0.0, 1.0, 20), 8)
    X = pd.DataFrame({"x": x, "z": shift - x, "g": np.repeat(np.arange(8), 20).astype(str), "s": s})
    y = 1.0 + 0.3 * x + np.sin(3.0 * s) + np.cos(row)
    spread = float(np.linalg.norm(x - x.mean()))
    size = np.hypot(np.sqrt(2.0) * x.mean() / spread, 1.0 / np.sqrt(n))
    share = shift / (np.sqrt(2.0) * spread) / size / SHARED_RANK_POLICY.factor_rcond
    assert abs(share - 1.0) > 0.05
    rows = {}
    for solve in ("structured", "gram"):
        wiggle = {"wiggle": LambdaPolicy.fixed(1.0)}
        features = {"x": Numeric(), "z": Numeric()}
        features["s"] = Spline(n_knots=5, lambda_policy=LambdaPolicy.fixed(1.0))
        model = SuperGLM(
            family="gaussian",
            features=features,
            interactions=[FactorSmooth("s", group="g", basis="sz", k=5, lambda_policy=wiggle)],
            selection_penalty=0,
            direct_solve=solve,
        ).fit_reml(X, y)
        structured = bool(model._fit_inference_info.get("structured_covariance"))
        assert structured == (solve == "structured")
        rows[solve] = model.summary()._coef_rows[0]
        assert rows[solve].estimable is bool(share < 1.0), solve
    if share < 1.0:
        np.testing.assert_allclose(rows["structured"].se, rows["gram"].se, rtol=1e-6)


def test_the_low_level_readers_still_take_a_bare_decider(polynomial_model) -> None:
    """``determines`` as a callable, the form before #752, still decides each contrast.

    The low-level ``term_inference`` took one decider of a contrast and ``relativities``
    one per term, ``determines(name)``; both still resolve from ``superglm.inference``.
    A decider that determines nothing leaves the base level, whose contrast is zero, at
    error 0 and every other level NaN; one that determines everything changes nothing.
    At 07ceaec50 both raised ``AttributeError`` on the ordered term.
    """
    from superglm.inference._term_model_ops import relativities
    from superglm.inference._term_ops import term_inference

    model = polynomial_model
    shared = (model._specs, model._interaction_specs, model._groups, model.result)
    covariance_fn = lambda: model._coef_covariance  # noqa: E731
    width = len(model.result.beta)

    def read(decide):
        asked = []

        def decider(contrast):
            asked.append(np.asarray(contrast).shape)
            return decide

        inference = term_inference(
            "band",
            result=model.result,
            groups=model._groups,
            specs=model._specs,
            interaction_specs=model._interaction_specs,
            covariance_fn=covariance_fn,
            reml_lambdas=None,
            lambda2=0.0,
            determines=None if decide is None else decider,
        )
        frame = relativities(
            ["band"],
            [],
            *shared,
            with_se=True,
            covariance_fn=covariance_fn,
            determines=None if decide is None else (lambda name: decider),
        )["band"]
        assert set(asked) <= {(width,)}
        return np.asarray(inference.se_log_relativity), frame["se_log_relativity"].to_numpy()

    plain = read(None)
    base = np.asarray(LEVELS) == model._specs["band"]._base_level
    assert np.all(plain[0][~base] > 0.0) and plain[0][base] == 0.0
    for errors, reference in zip(read(True), plain, strict=True):
        np.testing.assert_array_equal(errors, reference)
    for errors in read(False):
        np.testing.assert_array_equal(np.isnan(errors), ~base)
        assert errors[base] == 0.0


def test_readers_form_the_rule_only_for_a_term_that_reads_it(monkeypatch) -> None:
    """Work counts: no mask or centre is formed where no reader would use it (issue #752).

    ``relativities(with_se=True)`` and ``term_inference`` put a spline's curve and an
    ordered term's levels to the covariance's rule, and no other class, so a numeric
    term forms no coefficient mask.  A structured summary reads its columns' mean only
    to decide the intercept beside a coefficient the mask leaves open.  At 07ceaec50 the
    numeric-only readers formed the mask once each and the certified summary the mean.
    """
    from superglm.model import state_ops

    calls = {"mask": 0, "centre": 0}
    mask, centre = state_ops.fit_coefficient_mask, state_ops.fit_intercept_centre

    def counted(name, function):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(state_ops, "fit_coefficient_mask", counted("mask", mask))
    monkeypatch.setattr(state_ops, "fit_intercept_centre", counted("centre", centre))

    x = np.tile(np.linspace(0.0, 1.0, 40), 4)
    X = pd.DataFrame({"u": x, "v": np.cos(7.0 * x)})
    y = np.sin(3.0 * x) + np.cos(np.arange(160.0))
    numeric = SuperGLM(family="gaussian", features={"u": Numeric(), "v": Numeric()}).fit(X, y)
    numeric.relativities(with_se=True)
    numeric.term_inference("u")
    assert calls["mask"] == 0
    mixed = SuperGLM(family="gaussian", features={"u": Spline(n_knots=5), "v": Numeric()})
    mixed.fit(X, y).term_inference("v")
    assert calls["mask"] == 0
    mixed.term_inference("u")
    mixed.relativities(with_se=True)
    assert calls["mask"] == 2

    certified, _, _ = _intercept_fit("random effect", "alone")
    assert np.all(certified._fit_inference_info["coefficient_estimable"])
    assert certified.summary()._coef_rows[0].estimable is True and calls["centre"] == 0
    aliased, _, _ = _intercept_fit("random effect", "collinear")
    assert aliased.summary()._coef_rows[0].estimable is False and calls["centre"] == 1


def _robust_block_fit(direct_solve):
    """A random effect on 40 levels beside the collinear pair ``x``, ``2 x``, the twin
    piecewise terms ``u`` and ``v`` on one column, and a piecewise term ``k`` on its own."""
    n = 720
    row = np.arange(float(n))
    level = np.tile(np.repeat(np.arange(40), 2), 9)
    x = np.tile(np.linspace(-1.0, 1.0, 45), 16)
    k, u = np.cos(1.3 * row), np.sin(2.1 * row)
    X = pd.DataFrame({"g": level.astype(str), "x": x, "z": 2.0 * x, "k": k, "u": u, "v": u})
    y = 1.0 + 0.5 * x + 2.0 * np.cos(level) + 0.4 * np.maximum(k, 0.0) + 0.2 * u + np.sin(row)
    features = {"g": RandomEffect(), "x": Numeric(), "z": Numeric()}
    features.update({name: Piecewise(breaks=[0.0]) for name in ("k", "u", "v")})
    model = SuperGLM(family="gaussian", features=features, direct_solve=direct_solve)
    return model.fit_reml(X, y)


@pytest.mark.parametrize("clustered", [False, True])
def test_a_truncated_structured_bread_keeps_the_test_of_a_block_its_mask_shows(clustered) -> None:
    """A robust summary tests a block whose every coefficient its mask shows (issue #752).

    A structured fit's truncated bread keeps the fit's coefficient mask and no null
    direction, so it decides no contrast by itself.  ``k`` is aliased with nothing and
    its two coefficients are shown: its chi-square is ``b' V^-1 b`` on its robust block,
    the dense fit's figure, under HC0 and CR0.  The twins ``u`` and ``v`` are not shown
    and stay withheld, flagged ``estimable=False``, on both fits.  At 07ceaec50 the
    structured summary withheld ``k`` too.
    """
    summaries = {}
    for solve in ("structured", "gram"):
        model = _robust_block_fit(solve)
        cluster = np.arange(720) % 15 if clustered else None
        covariance = model.sandwich_covariance(cluster)
        truncated = covariance.undetermined is not None and covariance.undetermined.basis is None
        assert truncated == (solve == "structured")
        group = next(g for g in model._groups if g.name == "k")
        assert np.all(covariance.coefficient_estimable[group.sl])
        rows = _numeric_test_rows(model.summary(covariance=covariance), ("k", "u", "v"))
        for name in ("u", "v"):
            twin = next(g for g in model._groups if g.name == name)
            assert not np.any(covariance.coefficient_estimable[twin.sl])
            assert rows[name].wald_chi2 is None and rows[name].wald_p is None
            assert rows[name].estimable is False
        active = next(a for a in covariance.active_groups if a.name == "k")
        block = covariance.covariance[1 + active.start : 1 + active.end][:, 1 + active.start :]
        b = model.result.beta[group.sl]
        expected = float(b @ np.linalg.solve(block[:, : group.size], b))
        assert rows["k"].estimable is True and rows["k"].ref_df == group.size
        np.testing.assert_allclose(rows["k"].wald_chi2, expected, rtol=1e-9)
        summaries[solve] = rows["k"]
    np.testing.assert_allclose(
        summaries["structured"].wald_chi2, summaries["gram"].wald_chi2, rtol=1e-8
    )


def test_a_geometry_saved_before_it_kept_the_mean_is_formed_again() -> None:
    """A saved fit's old contrast geometry does not blank an intercept its data determine (#752).

    A structured fit's inference record keeps the geometry its contrasts were decided on.
    One pickled before that geometry kept the columns' mean decides no intercept, so the
    reader forms it again from the retained factor, with no rows and no refit, and the
    loaded summary shows the fresh fit's error bit for bit.  The fit is compact, with
    varying and zero weights and an offset, beside the collinear pair ``a``, ``2 a``.
    At 07ceaec50 the loaded summary showed the intercept with no error.
    """
    n = 720
    row = np.arange(float(n))
    level = np.tile(np.repeat(np.arange(40), 2), 9)
    a = np.tile([-0.5, 0.5], n // 2)
    s = 0.5 + 0.5 * np.sin(0.7 * row)
    X = pd.DataFrame({"g": level.astype(str), "s": s, "a": a, "b": 2.0 * a})
    y = 1.0 + 0.5 * a + 2.0 * np.cos(level) + s + np.cos(row)
    weights = np.linspace(0.5, 1.5, n)
    weights[::9] = 0.0
    model = SuperGLM(
        family="gaussian",
        features={"a": Numeric(), "b": Numeric(), "s": Spline(n_knots=5)},
        interactions=[FactorSmooth("s", group="g", basis="sz", k=5)],
        retain_fit_state=False,
    ).fit_reml(X, y, sample_weight=weights, offset=0.3 * np.cos(0.11 * row))
    inference = model._fit_inference_info
    assert inference.get("structured_covariance") and model._dm is None
    assert not np.all(inference["coefficient_estimable"])
    fresh = model.summary()._coef_rows[0]
    assert fresh.estimable is True and fresh.se > 0.0

    saved = vars(inference["contrast_estimability"])
    for name in ("centre", "centre_along", "centre_size"):
        del saved[name]
    loaded = pickle.loads(pickle.dumps(model))
    assert loaded._fit_inference_info["contrast_estimability"].centre is None
    intercept = loaded.summary()._coef_rows[0]
    assert intercept.estimable is True
    assert (intercept.se, intercept.z, intercept.ci_low) == (fresh.se, fresh.z, fresh.ci_low)
    assert loaded._fit_inference_info["contrast_estimability"].centre is not None


def test_editor_stale_summary_keeps_the_basis_vocabulary(segmented_model) -> None:
    from superglm.editor.apply import apply_edits_to_model_copy_with_data
    from superglm.editor.session import EditorSession

    session = EditorSession.from_model(segmented_model)
    term = session._require_term("band")
    term.edited_log_effect = term.edited_log_effect + 0.01
    edited = apply_edits_to_model_copy_with_data(segmented_model, {"band": term})
    text = str(edited.summary())
    assert "[ordered piecewise" in text
    assert "band[Mi000]" in text


# ── export ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "basis",
    [
        Piecewise(breaks=["Mi004", "Mi007"], degrees=[2, 1, 0]),
        Polynomial(powers=[1, 2]),
        Spline(kind="cr", n_knots=4),
    ],
    ids=["piecewise", "polynomial", "spline"],
)
def test_rating_table_is_one_row_per_band_for_every_basis(basis, tmp_path) -> None:
    """One row per band whatever the basis, on the link a rating table is for.

    Poisson rather than this module's shared gaussian ``_fit``: the exported
    table is multiplicative, so the export accepts log-link models only, the
    same restriction ``test_workbook_reconstruction_is_exact_for_a_segmented_
    term`` below already relies on.  What is under test here is the level axis
    -- every band gets a row regardless of the basis fitted through them --
    which the family does not touch.
    """
    X, y = _log_link_frame()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"band": OrderedCategorical(order=LEVELS, basis=basis)},
    )
    model.fit(X, y)
    path = model.export_rating_tables(tmp_path / "tables.xlsx", X, y)
    sheet = pd.read_excel(path, sheet_name="Rating Tables", header=None)
    column = sheet[0].astype(str)
    for level in LEVELS:
        assert (column == level).any(), f"missing band row {level!r}"


def test_workbook_reconstruction_is_exact_for_a_segmented_term() -> None:
    """Per-band table ratios equal per-band prediction ratios exactly.

    Under the log link the exported per-band relativity IS the model: any
    degree is table-exact on the level axis, because the table carries one row
    per band and scoring a band reads that row.
    """
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"band": rng.choice(LEVELS, 4000)})
    position = {level: index for index, level in enumerate(LEVELS)}
    mu = np.exp(-1.0 + 0.05 * np.array([min(position[b], 6) for b in X["band"]]))
    y = rng.poisson(mu).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=LEVELS, basis=Piecewise(breaks=["Mi004", "Mi007"], degrees=[2, 1, 0])
            )
        },
    )
    model.fit(X, y)
    ti = model.term_inference("band")
    table = dict(zip(ti.levels, ti.relativity))
    grid = pd.DataFrame({"band": LEVELS})
    predicted = model.predict(grid)
    ratios = predicted / predicted[0]
    table_ratios = np.array([table[b] for b in LEVELS]) / table[LEVELS[0]]
    assert np.allclose(ratios, table_ratios, rtol=1e-12, atol=0.0)


def _grouping(X, groups: dict[str, list[str]]):
    """A LevelGrouping over LEVELS with everything unnamed left as a singleton."""
    from superglm.features.grouping import collapse_levels

    covered = {member for members in groups.values() for member in members}
    full = dict(groups)
    for level in LEVELS:
        if level not in covered:
            full[level] = [level]
    return collapse_levels(X["band"].to_numpy(dtype=object), groups=full, order=LEVELS)


def _grouped_curve(X, y, basis, groups: dict[str, list[str]]):
    spec = OrderedCategorical(order=LEVELS, basis=basis, grouping=_grouping(X, groups))
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(X, y)
    inference = model.term_inference("band", with_se=True)
    assert inference.se_log_relativity is not None
    assert len(inference.se_log_relativity) == len(inference.levels)
    return inference


def _reference_curve(X, y, basis, groups: dict[str, list[str]]):
    """The same fit stated WITHOUT a grouping: one level per group, at its position.

    An independent reference for what the grouped term's curve should be. The
    data is relabelled to group labels and the group positions are declared
    directly through ``values=``, so no grouping code runs at all -- if the
    expansion touches a single value of the curve, this stops matching.
    """
    grouping = _grouping(X, groups)
    declared = dict(zip(LEVELS, np.linspace(0.0, 1.0, len(LEVELS)).tolist()))
    values = {
        label: float(np.mean([declared[str(member)] for member in members]))
        for label, members in grouping.group_to_originals.items()
    }
    relabelled = pd.DataFrame({"band": [grouping.original_to_group[b] for b in X["band"]]})
    spec = OrderedCategorical(values=values, basis=basis)
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(relabelled, y)
    return model.term_inference("band", with_se=True).smooth_curve


def test_a_grouped_display_curve_is_the_fitted_curve_bit_for_bit() -> None:
    """The expansion moves the markers and touches nothing else (issue #282).

    It used to interpolate a fresh 200-point PCHIP through the expanded markers
    wherever the collapse moved the display axis, and export it with no band.
    Both halves were damage. ``resolve_grouped_level_display("auto", ...)``
    returns ``"collapsed"`` for every grouped ``OrderedCategorical``, so the
    DEFAULT panel re-collapses the markers straight back onto the fitted
    positions -- exactly, because both halves compute a group's position with
    ``group_axis_position`` over the same members. The markers and the per-level
    SEs therefore made the round trip intact and the curve was the only casualty:
    on ``Spline(kind="cr", n_knots=4)`` over ten levels merging ``Mi001+Mi002``
    the default panel drew a shape up to 0.0474 in log-relativity -- 4.86% in
    relativity -- from the function that had been fitted, with no band.

    Stated against an INDEPENDENT reference rather than against itself: the same
    model declared with no grouping at all, one level per group at the group's
    position. Equality is exact on all six curve fields, which is the strongest
    form of "the curve is the fitted curve" available here -- and it is not a
    tolerance that a re-exported band or a slightly different interpolant could
    slip through.
    """
    X, y = _frame()

    for merge in ({"Mi000+Mi001": ["Mi000", "Mi001"]}, {"Mi001+Mi002": ["Mi001", "Mi002"]}):
        basis = Spline(kind="cr", n_knots=4)
        ti = _grouped_curve(X, y, basis, merge)
        reference = _reference_curve(X, y, Spline(kind="cr", n_knots=4), merge)
        for field in ("x", "log_relativity", "relativity", "se_log_relativity", "ci_lower"):
            ours = getattr(ti.smooth_curve, field)
            theirs = getattr(reference, field)
            assert (ours is None) == (theirs is None), (merge, field)
            np.testing.assert_array_equal(
                np.asarray(ours, dtype=np.float64),
                np.asarray(theirs, dtype=np.float64),
                err_msg=f"{merge}: the expansion changed {field}",
            )
        # not an equality of two Nones: the band is real and rides along
        assert ti.smooth_curve.se_log_relativity is not None, merge
        assert ti.smooth_curve.ci_lower is not None and ti.smooth_curve.ci_upper is not None
        # the rated quantities are untouched by any of this
        assert ti.se_log_relativity is not None
        assert len(ti.se_log_relativity) == len(ti.levels)


def test_a_grouped_display_curve_does_not_interpolate_its_markers() -> None:
    """The mechanism check, not just the values: no marker is a knot of the curve.

    A PCHIP through the expanded markers passes through every one of them by
    construction, so ``curve(level_x[i]) == log_relativity[i]`` held to machine
    precision for all i. A fitted spline does not: a merged pair is one
    parameter reported at two marker positions, and the curve cannot be at both
    values. This separates "the curve is fitted" from "the curve happens to
    agree numerically", and it is the assertion a future re-interpolation would
    trip on even if it reproduced the band.

    A leading merge also puts the first marker OUTSIDE the fitted curve's own
    domain -- the group sits at the mean of its members, so the fitted axis
    starts inside the declared one. That is the ggplot2 ``fullrange=FALSE``
    default (the smoothing line is not expanded to the range of the plot), and
    the renderers already union the two extents when setting x-limits.
    """
    X, y = _frame()
    merge = {"Mi000+Mi001": ["Mi000", "Mi001"]}
    ti = _grouped_curve(X, y, Spline(kind="cr", n_knots=4), merge)
    curve = ti.smooth_curve

    curve_x = np.asarray(curve.x, dtype=np.float64)
    curve_y = np.asarray(curve.log_relativity, dtype=np.float64)
    level_x = np.asarray(curve.level_x, dtype=np.float64)
    level_y = np.asarray(ti.log_relativity, dtype=np.float64)

    # the leading merge leaves the first declared marker off the fitted domain
    outside = level_x < curve_x.min()
    assert outside.sum() == 1, level_x
    inside = ~outside
    gap = np.abs(np.interp(level_x[inside], curve_x, curve_y) - level_y[inside]).max()
    # A 200-point grid over [0.0556, 1] steps 0.00474, so linear interpolation
    # of a curve whose second derivative is order 1 here carries at most ~1e-5
    # of its own error -- three orders below the 0.022 measured separation, and
    # five above the ~1e-16 an interpolant through the markers would give.
    assert gap > 1e-3, f"the curve still passes through its markers (max gap {gap:.3e})"


def test_a_grouped_display_band_is_centred_on_the_curve_it_ships_with() -> None:
    """#277's own measurement, on #277's own fixture, now zero.

    #277 removed the band because it was a statement about a curve that had been
    replaced: on an interior merge the fitted and rebuilt grids were the same
    200-point linspace, so the fitted band zipped against interpolated values.
    Measured then: 61 of 200 drawn points strictly outside their own exported
    band, with the band centre reproducing the FITTED curve to 1.1e-16 -- a
    ribbon around a line nobody drew.

    That reason lapses rather than being overruled. Nothing replaces any value
    now, so the band's centre and the drawn curve are the same array. This test
    is the guard that keeps it that way: it recomputes exactly #277's two
    numbers and requires 0 and machine zero.

    Tolerance is derived, not observed: ``ci_lower``/``ci_upper`` are
    ``exp(log_relativity -/+ z*se)``, so recovering the centre costs one ``exp``
    and one ``log`` per edge and a halved sum -- at most a handful of ulp of
    ``|log_relativity|``. 8 ulp bounds it. The failure it separates is 0.0474,
    fourteen orders away.

    Also corrects a label carried from #277 into #282: the "max band half-width
    = 0.00302" quoted there is the maximum STANDARD ERROR. The 95% half-width is
    1.96x it, 0.00587.
    """
    X, y = _frame()
    merge = {"Mi001+Mi002": ["Mi001", "Mi002"]}
    curve = _grouped_curve(X, y, Spline(kind="cr", n_knots=4), merge).smooth_curve

    # First, or the comparisons below are vacuous: a missing band reaches them
    # as NaN, and every `<` against NaN is False, so "0 points outside" would
    # pass on a panel that exports no band at all.
    assert curve.se_log_relativity is not None
    assert curve.ci_lower is not None and curve.ci_upper is not None

    log_rel = np.asarray(curve.log_relativity, dtype=np.float64)
    se = np.asarray(curve.se_log_relativity, dtype=np.float64)
    lo = np.log(np.asarray(curve.ci_lower, dtype=np.float64))
    hi = np.log(np.asarray(curve.ci_upper, dtype=np.float64))

    outside = int(np.count_nonzero((log_rel < lo) | (log_rel > hi)))
    assert outside == 0, f"{outside}/{len(log_rel)} drawn points outside their own band"

    tolerance = 8 * np.finfo(np.float64).eps * float(np.abs(log_rel).max())
    assert np.abs(0.5 * (lo + hi) - log_rel).max() <= tolerance

    # the band is real and the mislabelled figure is the SE, not a half-width
    assert se.max() == pytest.approx(0.0030, abs=5e-4)
    assert (hi - log_rel).max() == pytest.approx(1.96 * se.max(), rel=1e-3)


def test_a_grouped_display_band_brackets_the_curve_it_is_drawn_around() -> None:
    """Wherever a grouped panel exports a band, the drawn line is inside it.

    This is the invariant behind the rule above, stated on the values rather
    than on which branch produced them: ``ci_lower`` and ``ci_upper`` are
    ``exp(log_relativity -/+ z * se)``, so the curve they describe lies between
    them by construction. Exporting the fitted curve's band beside a rebuilt
    curve breaks it -- on the ``Mi001+Mi002`` spline case 61 of 200 points sat
    strictly outside, which a renderer draws as the line leaving its own
    ribbon.

    The tolerance bounds the round-trip error of that construction.
    ``relativity`` and the two edges each come from one ``exp`` of a
    quantity carrying at most a couple of rounding errors, so 8 ulp of the
    larger magnitude bounds any legitimate crossing. It is a null allowance
    against the failure it is here for, which is 1.4e14 ulp.
    """
    X, y = _frame()
    cases = [
        ("piecewise, merge after the break", Piecewise(breaks=["Mi004"]), ["Mi007", "Mi008"]),
        ("polynomial, interior merge", Polynomial(powers=[1, 2]), ["Mi001", "Mi002"]),
        ("spline, interior merge", Spline(kind="cr", n_knots=4), ["Mi001", "Mi002"]),
        ("spline, first-pair merge", Spline(kind="cr", n_knots=4), ["Mi000", "Mi001"]),
    ]
    banded = 0
    for label, basis, members in cases:
        curve = _grouped_curve(X, y, basis, {"+".join(members): members}).smooth_curve
        if curve.se_log_relativity is None:
            assert curve.ci_lower is None and curve.ci_upper is None, label
            continue
        banded += 1
        rel = np.asarray(curve.relativity, dtype=np.float64)
        lo = np.asarray(curve.ci_lower, dtype=np.float64)
        hi = np.asarray(curve.ci_upper, dtype=np.float64)
        assert len(lo) == len(hi) == len(rel) == len(curve.x), label
        slack = 8 * np.finfo(np.float64).eps * np.maximum(np.abs(rel), np.abs(hi))
        outside = int(np.count_nonzero((rel < lo - slack) | (rel > hi + slack)))
        assert outside == 0, f"{label}: {outside}/{len(rel)} drawn points outside their own band"
    # Not vacuous, and no longer partly vacuous: EVERY case exports a band now.
    # Two of these four are the spline merges #277 stripped, which the expansion
    # no longer rebuilds (issue #282), so the check runs on all of them.
    assert banded == len(cases), f"only {banded} of {len(cases)} grouped cases exported a band"


def test_a_grouping_that_merges_nothing_leaves_the_spline_curve_alone() -> None:
    """An identity collapse moves no level, so it may not redraw the curve.

    ``_grouping`` makes every uncovered level a singleton, so a grouping with
    no merges at all is the identity: ``grouped_ltv[glev]`` copies the original
    value and a singleton's ``np.mean`` returns its one element unchanged, so
    the expanded positions ARE the fitted curve's ``level_x``. This is a spline
    -- the basis whose collapse path is otherwise unchanged -- and it lands on
    the kept-curve branch for the geometric reason, not because of the basis
    name.

    Before, the identity grouping went through the same 200-point PCHIP as a
    real merge: same grid, different function. Measured here, the drawn curve
    now reproduces the ungrouped fit to 0.0 on all of x, log_relativity,
    relativity, level_x and the bands.
    """
    X, y = _frame()
    plain = OrderedCategorical(order=LEVELS, basis=Spline(kind="cr", n_knots=4))
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": plain})
    model.fit(X, y)
    ungrouped = model.term_inference("band", with_se=True).smooth_curve

    identity = _grouped_curve(X, y, Spline(kind="cr", n_knots=4), {}).smooth_curve

    for field in ("x", "log_relativity", "relativity", "level_x", "se_log_relativity"):
        theirs = getattr(ungrouped, field)
        ours = getattr(identity, field)
        assert (theirs is None) == (ours is None), field
        if theirs is None:
            continue
        np.testing.assert_array_equal(
            np.asarray(ours, dtype=np.float64),
            np.asarray(theirs, dtype=np.float64),
            err_msg=f"identity grouping changed {field}",
        )
    # the bands are real, so the equality above is not equality of two Nones
    assert identity.se_log_relativity is not None


def test_grouped_segmented_piecewise_keeps_every_stated_break_as_a_vertex() -> None:
    """The headline claim on the SEGMENTED grid, not just the all-degree-1 one.

    ``reconstruct`` returns the knots themselves when every segment is linear,
    which is the easy case; ``_reconstruct_segmented`` builds a 25-point
    linspace per curved segment and unions them, so the grid is 27 points here
    and the breaks have to survive that union. This module's own fixture is
    ``degrees=[2, 1, 0]``, and grouping ``Mi002+Mi003`` merges INSIDE the first
    segment, which is allowed and slides both breaks down one position.

    Measured on this fit: breaks at positions 3.0 and 6.0, both exact vertices
    of the 27-point grid, bands of matching length, and a first segment that is
    a genuine quadratic -- constant second difference to 2.50e-16 spread on a
    mean of 1.2359e-03, where the PCHIP rebuild this replaced left ~1e-04.
    """
    X, y = _frame()
    ti = _grouped_curve(
        X,
        y,
        Piecewise(breaks=["Mi004", "Mi007"], degrees=[2, 1, 0]),
        {"Mi002+Mi003": ["Mi002", "Mi003"]},
    )
    curve = ti.smooth_curve
    curve_x = np.asarray(curve.x, dtype=np.float64)
    curve_y = np.asarray(curve.log_relativity, dtype=np.float64)
    level_x = np.asarray(curve.level_x, dtype=np.float64)
    levels = list(ti.levels)
    by_position = dict(zip(level_x.tolist(), np.asarray(ti.log_relativity, dtype=np.float64)))

    assert len(curve_x) > 4, "a segmented grid is denser than its knots; this one is not"
    for name in ("Mi004", "Mi007"):
        position = level_x[levels.index(name)]
        assert np.any(curve_x == position), (
            f"the stated break {name} sits at x={position} but is not a vertex of {curve_x!r}"
        )
        corner = int(np.flatnonzero(curve_x == position)[0])
        assert curve_y[corner] == by_position[position]

    # the bands belong to the curve being drawn, so they zip against it
    assert curve.se_log_relativity is not None
    assert len(curve.se_log_relativity) == len(curve_x)

    # The degree-2 segment is a quadratic on a uniform sub-grid, so its second
    # difference is constant. Tolerance: the second-difference stencil is
    # (1, -2, 1), so it magnifies each entry's representation error at most
    # fourfold; allowing a handful of ulp for the reconstruction that produced
    # the entries gives 16 * eps * max|curve_y|. The defect it separates is
    # ~1e-04, eleven orders away.
    first_break = level_x[levels.index("Mi004")]
    segment = curve_y[curve_x <= first_break]
    # The sub-grid is uniform TO MACHINE PRECISION, not exactly: this fixture's
    # first segment is linspace(0, 3, 25) and 3/24 = 0.125 is dyadic, so every
    # point lands exact -- but that is a property of where the break sits, not
    # of the grid construction. Move the break one level and the step is 4/24 =
    # 1/6, which is not representable; measured ptp(diff) is then 4.44e-16 and
    # an exact test would fail on grid spacing before the quadratic it guards
    # was ever examined. Tolerance: each point carries at most eps*max|x| of
    # representation error, a first difference of two points at most twice
    # that, and ptp is a difference of two such differences, so 4 * eps *
    # max|curve_x|; a handful of ulp for linspace's own reconstruction gives 8.
    # The bound scales with the break, so each is measured against its OWN
    # bound: break 3 (this fixture) 0.0 against 5.3e-15, break 4 4.4e-16
    # against 7.1e-15, break 5 8.9e-16 against 8.9e-15, break 6 0.0 against
    # 1.1e-14, break 7 8.9e-16 against 1.2e-14 -- tightest margin 10x, at
    # break 5. The defect it separates -- a genuinely non-uniform grid -- is of
    # order the step itself, 0.125, thirteen orders away.
    segment_x = curve_x[curve_x <= first_break]
    spacing_tolerance = 8 * np.finfo(np.float64).eps * float(np.abs(segment_x).max())
    assert np.ptp(np.diff(segment_x)) < spacing_tolerance
    tolerance = 16 * np.finfo(np.float64).eps * float(np.abs(curve_y).max())
    assert np.ptp(np.diff(segment, 2)) < tolerance


def test_grouped_piecewise_with_specials_smooths_only_the_ordered_levels() -> None:
    """Grouping, ``specials=`` and a ``Piecewise`` basis in one term.

    The interaction is correct but load-bearing across two files: the special
    is masked out of ``smooth_levels`` BEFORE the expanded positions are built,
    and ``_install_position_axis`` maps only the declared smooth levels, so a
    special is never looked up in a map with no key for it. Nothing combined
    all three, so a change on either side would have surfaced as a ``KeyError``
    out of the public ``term_inference`` path rather than as a test failure.

    Measured: 11 reported levels, one of them special, ``level_x`` covering the
    10 smoothed ones only, and the stated break still a vertex of [0, 4, 8].
    """
    X, y = _frame()
    rng = np.random.default_rng(9)
    X = X.copy()
    X.loc[rng.random(len(X)) < 0.08, "band"] = "MISSING"

    from superglm.features.grouping import collapse_levels

    universe = LEVELS + ["MISSING"]
    members = {"Mi007+Mi008": ["Mi007", "Mi008"]}
    covered = {m for group in members.values() for m in group}
    full = dict(members)
    for level in universe:
        if level not in covered:
            full[level] = [level]
    grouping = collapse_levels(X["band"].to_numpy(dtype=object), groups=full, order=universe)

    spec = OrderedCategorical(
        order=LEVELS,
        basis=Piecewise(breaks=["Mi004"]),
        specials=["MISSING"],
        grouping=grouping,
    )
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(X, y)
    ti = model.term_inference("band", with_se=True)
    curve = ti.smooth_curve

    special = np.asarray(ti.level_is_special, dtype=bool)
    assert list(ti.levels) == universe
    assert special.sum() == 1 and special[list(ti.levels).index("MISSING")]
    # level_x carries the SMOOTHED levels only; the special keeps a detached row
    assert len(curve.level_x) == int((~special).sum())
    np.testing.assert_array_equal(
        np.asarray(curve.level_x, dtype=np.float64), [0, 1, 2, 3, 4, 5, 6, 7, 7, 8]
    )
    # and the fitted corner is still drawn, not interpolated around
    np.testing.assert_array_equal(np.asarray(curve.x, dtype=np.float64), [0.0, 4.0, 8.0])
    assert curve.se_log_relativity is not None
    assert len(curve.se_log_relativity) == len(curve.x)


def test_grouped_display_curve_keeps_the_stated_c0_corner() -> None:
    """A grouped Piecewise term draws its stated corner, not a smoothed bend.

    The whole contract of ``Piecewise`` is STATED breaks with C0 joins, so a
    display path that rounds the join shows a different function from the one
    that was fitted. The grouped expansion used to replace the fitted curve
    with a 200-point ``PchipInterpolator`` through the expanded level markers.
    PCHIP is C1 by construction -- one derivative per node -- so the drawn
    curve had no corner anywhere, and the stated break was not even a point on
    its grid. Measured on this fit: the drawn secants straddling the break
    differed by -0.008956 against the fitted slope change of -0.057576 (6.4x
    too shallow, and what little difference remained was the interpolant's own
    curvature rather than a join), and the drawn curve ran 0.006570 in
    log-relativity -- 0.66% in relativity -- away from the fitted shape. The
    same rebuild on a merge that also shifts the break gave 0.012590 (1.27%)
    and 5.9x; on a sharper stated kink it reached 0.0393 (4.0%) and 6.3x.

    Grouping a pair AFTER the break leaves the break at position 4, so the
    fitted shape is exactly the polyline through the three knot bands
    (Mi000, Mi004, Mi009) and this test needs no model internals to state it.
    """
    X, y = _frame()
    ti = _grouped_curve(X, y, Piecewise(breaks=["Mi004"]), {"Mi007+Mi008": ["Mi007", "Mi008"]})
    curve = ti.smooth_curve

    curve_x = np.asarray(curve.x, dtype=np.float64)
    curve_y = np.asarray(curve.log_relativity, dtype=np.float64)
    level_x = np.asarray(curve.level_x, dtype=np.float64)
    by_position = dict(zip(level_x.tolist(), np.asarray(ti.log_relativity, dtype=np.float64)))
    break_x = 4.0

    # The corner is a point ON the drawn curve, not something a renderer has to
    # interpolate its way through.
    assert np.any(curve_x == break_x), (
        f"the stated break sits at x={break_x} but the drawn grid is {curve_x!r}"
    )
    corner = int(np.flatnonzero(curve_x == break_x)[0])
    # ...and the curve meets the break band at exactly the value that band rates.
    assert curve_y[corner] == by_position[break_x]

    # The drawn curve IS the fitted piecewise-linear function: exactly the
    # polyline through the three knot bands, to the last bit.
    knot_positions = np.array([0.0, break_x, 8.0])
    knot_values = np.array([by_position[position] for position in knot_positions])
    assert np.abs(curve_y - np.interp(curve_x, knot_positions, knot_values)).max() == 0.0

    # And the join is a genuine corner: the two drawn segments meeting at the
    # break have different slopes, by the amount the fit states.
    left = (curve_y[corner] - curve_y[corner - 1]) / (curve_x[corner] - curve_x[corner - 1])
    right = (curve_y[corner + 1] - curve_y[corner]) / (curve_x[corner + 1] - curve_x[corner])
    fitted_left = (knot_values[1] - knot_values[0]) / (knot_positions[1] - knot_positions[0])
    fitted_right = (knot_values[2] - knot_values[1]) / (knot_positions[2] - knot_positions[1])
    assert right - left == pytest.approx(fitted_right - fitted_left, abs=1e-15)
    # Guards the line above against vacuity: a corner that small would be a
    # straight line either way. Measured -0.0576 here, against the -0.0090 the
    # PCHIP's own curvature produced.
    assert abs(right - left) > 0.04

    # The bands now belong to the curve they are drawn around, so they zip.
    assert curve.se_log_relativity is not None
    assert len(curve.se_log_relativity) == len(curve_x)


def test_grouped_piecewise_panel_draws_the_corner_on_the_canvas() -> None:
    """The corner survives all the way to the rendered line, not just the term.

    ``_collapsed_smooth_curve`` keeps whatever curve the term inference hands
    it, so a rounded curve upstream is a rounded curve on the canvas.
    """
    import matplotlib

    matplotlib.use("Agg")

    X, y = _frame()
    spec = OrderedCategorical(
        order=LEVELS,
        basis=Piecewise(breaks=["Mi004"]),
        grouping=_grouping(X, {"Mi007+Mi008": ["Mi007", "Mi008"]}),
    )
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(X, y)
    curve = model.term_inference("band").smooth_curve
    expected_x = np.asarray(curve.x, dtype=np.float64)
    expected_y = np.asarray(curve.relativity, dtype=np.float64)

    ax = model.plot("band", X=X).axes[0]
    drawn = [line for line in ax.lines if len(line.get_xdata()) == len(expected_x)]
    assert len(drawn) == 1, f"expected one curve line of {len(expected_x)} vertices, got {drawn}"
    np.testing.assert_array_equal(np.asarray(drawn[0].get_xdata(), dtype=np.float64), expected_x)
    np.testing.assert_array_equal(np.asarray(drawn[0].get_ydata(), dtype=np.float64), expected_y)
    # Three vertices: the two boundary knots and the stated break. A polyline
    # through them is the fitted shape exactly; anything denser is a rebuild.
    np.testing.assert_array_equal(expected_x, [0.0, 4.0, 8.0])


def test_grouped_polynomial_display_curve_is_the_fitted_polynomial() -> None:
    """The same rebuild also replaced a hosted Polynomial's fitted curve.

    A Polynomial inner basis shares the Piecewise level-position axis, so a
    collapse does not move it either -- and there the rebuilt 200-point grid
    happened to MATCH the fitted one, so the pre-expansion bands were carried
    onto a curve that was not the one they were computed for. There is no
    corner to lose here; the shape itself was wrong.

    ``powers=[1, 2]`` fits a parabola, and on the uniform display grid a
    parabola has a constant second difference. Measured: 2.8e-16 spread on a
    mean of -3.08e-05 once the fitted curve is kept, against 9.59e-05 through
    the PCHIP -- a spread 3.1x the mean itself, i.e. not a parabola at all.
    """
    X, y = _frame()
    ti = _grouped_curve(X, y, Polynomial(powers=[1, 2]), {"Mi001+Mi002": ["Mi001", "Mi002"]})
    curve_x = np.asarray(ti.smooth_curve.x, dtype=np.float64)
    curve_y = np.asarray(ti.smooth_curve.log_relativity, dtype=np.float64)

    # Uniform grid, so second differences compare. Same derivation as the
    # segmented test's spacing bound: a point carries at most eps*max|x|, a
    # first difference at most twice that, and ptp is a difference of two such
    # differences, so 8 * eps * max|curve_x| once a handful of ulp is allowed
    # for linspace's own reconstruction. That is ~1.4e-14 here, against the
    # 1e-12 magic constant this replaces, which was ~70x looser and stated
    # nothing.
    spacing_tolerance = 8 * np.finfo(np.float64).eps * float(np.abs(curve_x).max())
    assert np.ptp(np.diff(curve_x)) < spacing_tolerance
    second = np.diff(curve_y, 2)
    # The tolerance follows from the arithmetic. The (1, -2, 1) stencil
    # magnifies each entry's error at most fourfold, and allowing a handful of
    # ulp for the reconstruction that produced the entries gives
    # 16 * eps * max|curve_y|. Measured spread 2.8e-16 against a bound of
    # 1.1e-15; the defect it separates is 9.59e-05, eleven orders away.
    tolerance = 16 * np.finfo(np.float64).eps * float(np.abs(curve_y).max())
    assert np.ptp(second) < tolerance


def test_plateau_bands_share_one_table_row_value(segmented_model) -> None:
    relativities = segmented_model.relativities()["band"]
    tail = relativities.set_index("level").loc[["Mi007", "Mi008", "Mi009"], "relativity"]
    assert float(tail.max()) == pytest.approx(float(tail.min()), abs=0.0)


def test_export_summary_group_kind_says_group_not_smooth() -> None:
    from superglm.export.summary import _group_test_kind

    X, y = _frame()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"band": OrderedCategorical(order=LEVELS, basis=Piecewise(breaks=["Mi004"]))},
    )
    model.fit(X, y)
    groups = tuple(group for group in model._groups if group.feature_name == "band")

    class _Row:
        name = "band"
        subgroup_type = None

    assert _group_test_kind(model, _Row(), groups) == "group"


def test_specials_ride_alongside_each_basis_in_the_summary() -> None:
    """The Fit column's word matches the whole-term classification.

    A level carried by an unpenalized parametric block reported "smooth"
    before the fix, while the whole-term row and the exported workbook said
    the block is a plain group -- the word must match the model.
    """
    X, y = _frame()
    rng = np.random.default_rng(9)
    X = X.copy()
    missing = rng.random(len(X)) < 0.08
    X.loc[missing, "band"] = "MISSING"
    for basis, fit_word in (
        (Piecewise(breaks=["Mi004"]), "piecewise"),
        (Polynomial(powers=[1, 2]), "polynomial"),
    ):
        spec = OrderedCategorical(order=LEVELS, basis=basis, specials=["MISSING"])
        model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
        model.fit(X, y)
        text = str(model.summary())
        assert "band[MISSING]" in text
        assert "free" in text
        level_line = next(line for line in text.splitlines() if "band[Mi000]" in line)
        assert fit_word in level_line
        assert "smooth" not in level_line

        # The editor-stale row builder carries the same vocabulary.
        from superglm.editor.apply import apply_edits_to_model_copy_with_data
        from superglm.editor.session import EditorSession

        session = EditorSession.from_model(model)
        term = session._require_term("band")
        term.edited_log_effect = term.edited_log_effect + 0.01
        edited = apply_edits_to_model_copy_with_data(model, {"band": term})
        stale_line = next(
            line for line in str(edited.summary()).splitlines() if "band[Mi000]" in line
        )
        assert fit_word in stale_line
        assert "smooth" not in stale_line
