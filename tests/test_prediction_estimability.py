"""Predictions that lean on what neither the data nor the penalty determine are named once.

R's ``predict.lm(rankdeficient = "warnif")``: a row ``r`` is off the determined
space when ``||r N|| > 1e-6 ||r||`` (Lenth, R Journal 7(1), 2015), with ``N``
the null space the fit's data rank recorded, restricted to the null space of
every penalty component, read in the record's equilibrated scale
(``model.estimability_ops``).  ``predict`` returns master's price and gives one
``NonEstimablePredictionWarning`` whose ``rows`` names them; the rating-table
export marks the entries that lean.

Each fixture forces its direction by construction: a level seen at one value of
its by-level slope, or a column equal to another on every training row.  A test
of a warning or a mark fails on master, which has neither (the class is matched
by name, so it fails on the assertion).  A test of silence is a control
(``pins_behaviour``) and reads only the public API, so it passes on master;
several pin a defect a review found at ``ce4a5b2f2``.  Every fixture's shares,
training rows included, are asserted at least three orders of magnitude on the
right side of the tolerance, so the tolerance does not pick the answer.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import superglm
from superglm import (
    Categorical,
    FactorSmooth,
    Numeric,
    OrderedCategorical,
    Polynomial,
    RandomEffect,
    Spline,
    SuperGLM,
)
from superglm._frame import as_eager_frame
from superglm.model import base

TOL = 1e-6  # R's predict.lm tol, the module's PREDICTION_TOLERANCE
MARGIN = 1e3  # each side of TOL by at least this factor

rng = np.random.default_rng(7)
N = 2000
X_SPREAD = rng.uniform(0, 10, N)
LEVEL = rng.choice(list("ABCDE"), N)
Y = rng.gamma(2.0, np.exp(0.1 * np.sin(X_SPREAD)) / 2.0)
# Row J: level E sits at x = 5 alone, so its by-level slope (or its spline's
# unpenalised line) is aliased with its level on every training row.
ROW_J = pd.DataFrame({"x": np.where(LEVEL == "E", 5.0, X_SPREAD), "f": LEVEL})
# E at x - 1 and x + 1 is off the span; E at 5 and the other levels are on it.
ROW_J_NEW = pd.DataFrame({"x": [5.0, 4.0, 6.0, 3.0, 5.0], "f": ["E", "E", "E", "A", "B"]})
ALIASED_NEW = pd.DataFrame(
    {"x1": [1.0, 1.0, 2.0, 0.5], "x2": [5.0, 6.0, 7.0, 2.0], "f": ["A", "B", "C", "D"]}
)

# master (bfff7a074)'s ``predict`` on ROW_J_NEW and ALIASED_NEW, as float.hex (native
# OpenBLAS; master's own Haswell values differ by at most 1.3e-12 relative)
MASTER_PRICES = {
    ("numeric", "fit"): [
        "0x1.ff76266bff2d6p-1",
        "0x1.f4dd62de00f88p-1",
        "0x1.0524275eb8bf7p+0",
        "0x1.053b0f2502ed4p+0",
        "0x1.06b7686fbb86dp+0",
    ],
    ("numeric", "fit_reml"): [
        "0x1.ff76266bff2d6p-1",
        "0x1.f4dd62de00f88p-1",
        "0x1.0524275eb8bf7p+0",
        "0x1.053b0f2502ed4p+0",
        "0x1.06b7686fbb86dp+0",
    ],
    ("spline", "fit"): [
        "0x1.ff76266bff2f5p-1",
        "0x1.7c3e5525756d4p+8",
        "0x1.7ce6e4ab73386p-9",
        "0x1.01c464ccbd8d9p+0",
        "0x1.bb027f00887e8p-1",
    ],
    ("spline", "fit_reml"): [
        "0x1.ff76266bff2d1p-1",
        "0x1.0368ce18b7269p+0",
        "0x1.121eb5112a7c2p+0",
        "0x1.00425a81e50e1p+0",
        "0x1.cdf5cce3b3ea6p-1",
    ],
    ("aliased", "fit"): [
        "0x1.66a690271d49bp-1",
        "0x1.7dde80ae777bbp-1",
        "0x1.f2a45af4acd70p-1",
        "0x1.55c668d688a61p-1",
    ],
    ("aliased", "fit_reml"): [
        "0x1.66a690271d49bp-1",
        "0x1.7dde80ae777bbp-1",
        "0x1.f2a45af4acd70p-1",
        "0x1.55c668d688a61p-1",
    ],
}


def _master(key):
    return np.array([float.fromhex(v) for v in MASTER_PRICES[key]])


def _fit(X, y, features, interactions=(), *, method="fit", family=None, **kwargs):
    model = SuperGLM(
        family=family or superglm.families.gamma(),
        link="log",
        features=features,
        interactions=list(interactions),
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(model, method)(X, y)
    return model


X_SPECS = {"numeric": Numeric, "spline": lambda: Spline(n_knots=6)}


def _row_j(x_name, method):
    return _fit(ROW_J, Y, {"x": X_SPECS[x_name](), "f": Categorical()}, [("x", "f")], method=method)


def _disclosures(call):
    """``call``'s value and its off-span warnings, matched by class name."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = call()
    found = [
        c.message for c in caught if type(c.message).__name__ == "NonEstimablePredictionWarning"
    ]
    return value, found


def _shares(model, X, random_effects="conditional"):
    from superglm.model.estimability_ops import prediction_shares

    return prediction_shares(model, as_eager_frame(X), base._prediction_plan(model), random_effects)


def _assert_gap(model, on_span, off_span):
    """The tolerance sits at least ``MARGIN`` from every share on either side."""
    assert np.max(_shares(model, on_span)) < TOL / MARGIN
    assert np.min(_shares(model, off_span)) > TOL * MARGIN


def _aliased():
    """``x2 = 2 x1 + 3`` on every training row: one null direction across two terms."""
    local = np.random.default_rng(17)
    x1 = local.uniform(0, 4, N)
    X = pd.DataFrame({"x1": x1, "x2": 2.0 * x1 + 3.0, "f": LEVEL})
    y = local.poisson(np.exp(0.2 * x1 - 0.5)).astype(float)
    features = {"x1": Numeric(), "x2": Numeric(), "f": Categorical()}
    return X, y, features


@pytest.mark.parametrize("method", ["fit", "fit_reml"])
@pytest.mark.parametrize("x_name", list(X_SPECS))
def test_row_j_off_its_level_value_is_warned_at_masters_price(x_name, method):
    """Level E quoted at x = 4 and 6 leans on the alias; the price is master's.

    ``rows`` names 1 and 2 only.  The price is master's to ``1e-9`` relative:
    every row, the off-span ones included, whose value is the fit's gauge.
    The fit's own rows sit on the span with the other new rows.
    """
    model = _row_j(x_name, method)
    mu, found = _disclosures(lambda: model.predict(ROW_J_NEW))
    assert len(found) == 1
    assert found[0].rows.tolist() == [1, 2]
    assert "rows 1, 2" in str(found[0])
    np.testing.assert_allclose(mu, _master((x_name, method)), rtol=1e-9)
    on_span = pd.concat([ROW_J, ROW_J_NEW.iloc[[0, 3, 4]]], ignore_index=True)
    _assert_gap(model, on_span, ROW_J_NEW.iloc[[1, 2]])


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("x_name", list(X_SPECS))
def test_row_j_training_rows_and_their_copy_are_never_warned(x_name):
    """Control (passes on master): the fit's own rows, a copy and a reversed copy."""
    model = _row_j(x_name, "fit")
    for frame in (ROW_J, ROW_J.copy(), ROW_J.iloc[::-1].reset_index(drop=True)):
        _, found = _disclosures(lambda frame=frame: model.predict(frame))
        assert not found


@pytest.mark.parametrize("method", ["fit", "fit_reml"])
def test_an_aliased_column_quoted_off_its_span_is_warned(method):
    """Rows with ``x2 != 2 x1 + 3`` are named at master's price; the training rows are not."""
    X, y, features = _aliased()
    model = _fit(X, y, features, method=method, family=superglm.families.poisson())
    mu, found = _disclosures(lambda: model.predict(ALIASED_NEW))
    assert [w.rows.tolist() for w in found] == [[1, 3]]
    np.testing.assert_allclose(mu, _master(("aliased", method)), rtol=1e-9)
    on_span = pd.concat([X, ALIASED_NEW.iloc[[0, 2]]], ignore_index=True)
    _assert_gap(model, on_span, ALIASED_NEW.iloc[[1, 3]])
    _, found = _disclosures(lambda: model.predict(X.copy()))
    assert not found


@pytest.mark.pins_behaviour
def test_a_full_rank_fit_is_never_warned():
    """Control (passes on master): a full-rank fit, quoted far outside its range."""
    X = pd.DataFrame({"x": X_SPREAD, "f": LEVEL})
    model = _fit(X, Y, {"x": Spline(n_knots=6), "f": Categorical()}, [("x", "f")])
    far = pd.DataFrame({"x": [-50.0, 0.0, 10.0, 60.0], "f": ["E", "A", "B", "C"]})
    for frame in (X, X.copy(), far):
        _, found = _disclosures(lambda frame=frame: model.predict(frame))
        assert not found


@pytest.mark.pins_behaviour
def test_predicts_readers_do_not_repeat_the_warning():
    """Control (passes on master): ``predict_priced`` reads ``predict``'s means without its test."""
    model = _row_j("numeric", "fit")
    mu, found = _disclosures(lambda: base.predict_priced(model, ROW_J_NEW, None))
    assert not found
    np.testing.assert_allclose(mu, _master(("numeric", "fit")), rtol=1e-9)


def _random_effect_fit():
    local = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame(
        {"x": local.uniform(0, 10, n), "g": local.choice([f"g{i}" for i in range(20)], n)}
    )
    y = local.poisson(np.exp(-1 + 0.05 * X["x"].to_numpy())).astype(float)
    return _fit(
        X,
        y,
        {"x": Spline(n_knots=6), "g": RandomEffect()},
        method="fit_reml",
        family=superglm.families.poisson(),
    ), X


@pytest.mark.pins_behaviour
def test_a_healthy_random_effect_fit_is_never_warned():
    """Control (passes on master; at ce4a5b2f2 every population row warned).

    The random effect's level sum is aliased with the intercept in the data,
    and its zero-mean prior fixes it: the population prediction and an unseen
    level read the fit's ordinary mean.
    """
    model, X = _random_effect_fit()
    new = pd.DataFrame({"x": [2.0, 2.0, 7.0], "g": ["g1", "NEW", "g2"]})
    for frame, mode in ((X, "population"), (new, "conditional"), (new, "population")):
        _, found = _disclosures(
            lambda frame=frame, mode=mode: model.predict(frame, random_effects=mode)
        )
        assert not found


def test_the_penalty_fixes_a_random_effects_level_sum_and_no_row_is_read(monkeypatch):
    """The data leave the level sum free; the ridge prior curves it; ``predict`` does no extra pass.

    The record's data null space has one direction, whose curvature under the
    normalised prior is of order one, far above ``PENALTY_RANK_CUT``; Row J's
    alias reads rounding there.  No tested direction means no design pass.
    """
    from superglm._roundoff import PENALTY_RANK_CUT
    from superglm.model import estimability_ops

    model, X = _random_effect_fit()
    record = model.result.rank_info.data
    assert record.width - record.rank == 1
    assert estimability_ops.data_null_directions(model) is None
    dual = estimability_ops._dual_null_basis(record)
    scale = np.where(record.column_scale > 0.0, record.column_scale, 1.0)
    _, curvature = estimability_ops._penalty_free(model, model.result.rank_info, dual, scale)
    assert np.min(curvature) > PENALTY_RANK_CUT * MARGIN
    row_j = estimability_ops.data_null_directions(_row_j("numeric", "fit"))
    assert np.max(np.abs(row_j.curvature)) < PENALTY_RANK_CUT / MARGIN

    def no_pass(*args, **kwargs):
        raise AssertionError("a design pass on a fit with no tested direction")

    monkeypatch.setattr(estimability_ops, "_ScaledRows", no_pass)
    model.predict(X, random_effects="population")


def test_a_huge_finite_row_is_warned_without_overflow():
    """A row at ``1e170`` along an aliased column: priced as master, warned, no overflow.

    The share is formed on rows scaled by powers of two, so ``over="raise"``
    finds nothing to raise.
    """
    x = np.arange(-2.0, 3.0).repeat(50)
    X = pd.DataFrame({"x": x, "dup": x})
    y = np.random.default_rng(913).poisson(2, len(X)).astype(float)
    model = _fit(
        X,
        y,
        {"x": Numeric(), "dup": Numeric()},
        family=superglm.families.poisson(),
        selection_penalty=0.0,
    )
    new = pd.DataFrame({"x": [0.0], "dup": [1e170]})
    with np.errstate(over="raise", invalid="raise"):
        mu, found = _disclosures(lambda: model.predict(new))
    np.testing.assert_allclose(mu, [2.02383993], rtol=1e-8)
    assert [w.rows.tolist() for w in found] == [[0]]
    share = _shares(model, new)[0]
    np.testing.assert_allclose(share, np.sqrt(0.5), rtol=1e-6)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("shift", [0.0, 2.0**40, 2.0**46])
def test_a_translated_aliased_column_never_warns_its_training_rows(shift):
    """Control (passes on master; at ce4a5b2f2 every row warned at ``2**40``).

    ``x - dup + 2 [group B]`` is constant on every row; a row is read about the
    centre pair the record was decided at, so the translation moves nothing.
    """
    x = np.tile(np.arange(9.0), 100)
    f = np.tile([0.0, 1.0], x.size // 2)
    frame = pd.DataFrame({"x": shift + x, "dup": x + 2 * f, "group": np.where(f, "B", "A")})
    y = np.random.default_rng(913).poisson(2, len(frame)).astype(float)
    model = _fit(
        frame,
        y,
        {"x": Numeric(), "dup": Numeric(), "group": Categorical(base="first")},
        family=superglm.families.poisson(),
        selection_penalty=0.0,
    )
    _, found = _disclosures(lambda: model.predict(frame.copy()))
    assert not found


@pytest.mark.pins_behaviour
def test_a_binned_fits_training_rows_are_read_as_it_binned_them():
    """Control (passes on master; at ce4a5b2f2 all 300 rows warned).

    An unpenalised spline on four bins: the binned design has rank 3 of 7, and
    a training row's exact basis row is off it, but the fit read it at its
    bin's support point, which is on it.
    """
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 300)})
    y = np.random.default_rng(913).poisson(2, len(X)).astype(float)
    model = _fit(
        X,
        y,
        {"x": Spline(kind="ps", k=8)},
        family=superglm.families.poisson(),
        selection_penalty=0.0,
        spline_penalty=0.0,
        discrete=True,
        n_bins=4,
    )
    for frame in (X, X.copy()):
        _, found = _disclosures(lambda frame=frame: model.predict(frame))
        assert not found


def test_the_rating_table_marks_the_entries_the_data_do_not_determine(tmp_path):
    """Row J (spline): f's level E and the E column of ``x:f`` are marked; nothing else is.

    Mean-centred, every level of ``f`` reads the block's mean, which E moves,
    so every level is marked.  In the workbook a mark is a shaded cell and a
    note.
    """
    model = _row_j("spline", "fit")
    for centering, levels in (("native", [4]), ("mean", [0, 1, 2, 3, 4])):
        payload = model.rating_table_payload(ROW_J, Y, centering=centering, impact_bins=())
        blocks = {block.name: block for block in payload.main_effects}
        assert np.flatnonzero(blocks["f"].non_estimable).tolist() == levels
        assert not np.any(blocks["x"].non_estimable)
        (cells,) = payload.interactions
        columns = list(cells.table.columns[1:])
        marked = np.flatnonzero(np.any(cells.non_estimable, axis=0)).tolist()
        assert [columns[j] for j in marked] == ["E"]
        # every E cell away from x = 5 is marked
        keys = cells.table.iloc[:, 0].tolist()
        e = cells.non_estimable[:, columns.index("E")]
        assert e.sum() >= len(keys) - 1

    import openpyxl

    path = tmp_path / "rating.xlsx"
    model.export_rating_tables(path, ROW_J, Y, impact_bins=())
    notes = [
        cell.value
        for sheet in openpyxl.load_workbook(path).worksheets
        for row in sheet.iter_rows()
        for cell in row
        if isinstance(cell.value, str) and cell.value.startswith("Not determined")
    ]
    assert len(notes) == 2


def test_the_aliased_columns_per_unit_blocks_are_marked():
    """``x1`` and ``x2`` trade one for the other: each per-unit factor leans; ``f`` does not."""
    X, y, features = _aliased()
    model = _fit(X, y, features, family=superglm.families.poisson())
    payload = model.rating_table_payload(X, y, impact_bins=())
    blocks = {block.name: block for block in payload.main_effects}
    assert blocks["x1"].non_estimable.tolist() == [True]
    assert blocks["x2"].non_estimable.tolist() == [True]
    assert not np.any(blocks["f"].non_estimable)


def _poisson(X, features, interactions=(), **kwargs):
    y = np.random.default_rng(913).poisson(2, len(X)).astype(float)
    model = _fit(
        X,
        y,
        features,
        interactions,
        family=superglm.families.poisson(),
        selection_penalty=0.0,
        **kwargs,
    )
    return model, y


def _ordinal_frame():
    levels = list("ABCDEFG")
    code = np.tile(np.arange(7), 60)
    X = pd.DataFrame(
        {
            "band": np.array(levels)[code],
            "dup": code.astype(float),
            "x": np.random.default_rng(9).uniform(size=len(code)),
        }
    )
    band = OrderedCategorical(order=levels, base="first", basis=Spline(k=6))
    return X, band


def test_an_ordered_factors_native_table_marks_against_its_reference_level():
    """The ordered factor's line is aliased with ``dup``: A, the reference at 1.0, is not marked.

    Each other level's relativity is its contrast with A, which the aliased
    slope moves, so B to G are marked.
    """
    X, band = _ordinal_frame()
    model, y = _poisson(X, {"band": band, "dup": Numeric()})
    payload = model.rating_table_payload(X, y, impact_bins=(), n_bins=5)
    block = payload.main_effects[0]
    assert block.table.iloc[0, 1] == 1.0
    assert block.non_estimable.tolist() == [False] + [True] * 6


@pytest.mark.pins_behaviour
def test_an_ordered_factor_by_spline_surface_exports():
    """Control (passes on master; at ce4a5b2f2 the export raised on the grid's scores)."""
    X, band = _ordinal_frame()
    model, y = _poisson(X, {"band": band, "dup": Numeric(), "x": Spline(k=5)}, [("band", "x")])
    payload = model.rating_table_payload(X, y, impact_bins=(), n_bins=5)
    assert [block.table.shape for block in payload.interactions] == [(5, 6)]


def test_a_binned_entry_is_tested_as_its_rows_weighted_mean():
    """A bin of ``x`` values -1, 1, 1: its factor averages the aliased line's rows, so it is marked.

    The polynomial's linear part is aliased with ``dup``; at the bin's
    midpoint, 0, it vanishes, but its mean over the bin's rows does not.
    """
    x = np.tile(np.arange(-2.0, 3.0), 60)
    model, _ = _poisson(
        pd.DataFrame({"x": x, "dup": x}), {"x": Polynomial(degree=2), "dup": Numeric()}
    )
    quoted = np.array([-1.0, 1.0, 1.0])
    payload = model.rating_table_payload(
        pd.DataFrame({"x": quoted, "dup": quoted}), np.ones(3), impact_bins=(), n_bins=1
    )
    assert payload.main_effects[0].non_estimable.tolist() == [True]


def test_a_healthy_splines_binned_entries_are_not_marked():
    """A penalised spline on an 8-valued covariate: each bin is a mean of training rows."""
    local = np.random.default_rng(3)
    x = local.integers(1, 9, 5000).astype(float)
    y = local.poisson(np.exp(-1 + 0.1 * x)).astype(float)
    model = _fit(pd.DataFrame({"x": x}), y, {"x": Spline()}, family=superglm.families.poisson())
    payload = model.rating_table_payload(pd.DataFrame({"x": x}), y, impact_bins=())
    block = payload.main_effects[0]
    assert block.kind == "continuous"
    assert block.non_estimable is None or not np.any(block.non_estimable)


def test_a_polynomial_piece_is_marked_where_its_coefficients_lean():
    """An unpenalised cubic spline aliased with ``x (x - 1/2)(x + 1/2)``: every piece leans.

    On ``[-0.5, 0.5)`` the aliased cubic vanishes at the ends and the middle
    but not at the thirds, which the four coefficients read.
    """
    x = np.linspace(-1.0, 1.0, 513)
    X = pd.DataFrame({"x": x, "dup": x * (x - 0.5) * (x + 0.5)})
    model, y = _poisson(
        X,
        {"x": Spline(kind="bs", knots=[-0.5, 0.5], boundary=(-1.0, 1.0)), "dup": Numeric()},
        spline_penalty=0.0,
    )
    payload = model.rating_table_payload(X, y, impact_bins=(), continuous_kind="ppform")
    block = payload.main_effects[0]
    middle = block.table.iloc[:, 0].tolist().index("[-0.5, 0.5)")
    assert block.non_estimable[middle]


@pytest.mark.pins_behaviour
def test_an_sz_levels_held_rows_read_the_population_curve_they_are_priced_at():
    """Control (passes on master; before the sz rows were read through ``predict``'s own
    scoring, the disclosure crashed on them).

    Level E sits at x = 5 alone, so its line is free; ``predict`` prices E on the
    population curve and says so (``SZ_POPULATION_PREDICTION``), so E at other x
    leans on nothing and is not named again.
    """
    model = _fit(
        ROW_J,
        Y,
        {"x": Spline(n_knots=6)},
        [FactorSmooth("x", group="f", basis="sz", k=6)],
        method="fit_reml",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mu = model.predict(ROW_J_NEW)
    assert np.all(np.isfinite(mu))
    assert not [c for c in caught if type(c.message).__name__ == "NonEstimablePredictionWarning"]
    _, found = _disclosures(lambda: model.predict(ROW_J))
    assert not found
