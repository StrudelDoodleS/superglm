"""Predictions that lean on what neither the data nor the penalty determine are named once.

R's ``predict.lm(rankdeficient = "warnif")``: a row ``r`` is off the determined
space when ``||r N|| > 1e-6 ||r||`` (Lenth, R Journal 7(1), 2015), with ``N``
the null space the fit's data rank recorded, restricted to the null space of
every penalty component, read in the record's equilibrated scale
(``model.estimability_ops``).  ``predict`` returns master's price and gives one
``NonEstimablePredictionWarning`` whose ``rows`` names them. Dense fits with
ordinary or binned splines are tested on the exact prediction rows. The
silence and price of older binned saves, factor smooths and structured fits
are pinned here.

Each fixture forces its direction by construction: a level seen at one value of
its by-level slope, or a column equal to another on every training row.  A test
of a warning fails on master, which has none, on its assertion (the class is
matched by name).  Two tests read the module's own penalty restriction and
row blocks: they import ``superglm.model.estimability_ops``, so on master they
fail on that import, not on a behaviour.  A test of silence is
a control (``pins_behaviour``) and reads only the public API, so it passes on
master; several pin a defect a review found on this branch.  Every fixture's shares,
training rows included, are asserted at least three orders of magnitude on the
right side of the tolerance, so the tolerance does not pick the answer.
"""

from __future__ import annotations

import pickle
import threading
import warnings

import numpy as np
import pandas as pd
import pytest

import superglm
from superglm import (
    Categorical,
    FactorSmooth,
    Numeric,
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

# REML's default stop leaves a by-level spline's smoothing parameters to about 3e-6
# relative, where its criterion is flat, and that moves a price by 1e-8: master's default
# price sits 3.4e-9 from its own converged one. The spline's ``fit_reml`` price is fitted
# and recorded at these tolerances, where master's two kernels and this tree's agree to 2e-14.
CONVERGED_REML = {"reml_tol": 1e-12, "pirls_tol": 1e-12, "max_reml_iter": 400}

# master (bfff7a074)'s ``predict`` on ROW_J_NEW and ALIASED_NEW, as float.hex (native
# OpenBLAS; master's own Haswell values differ by at most 1.3e-12 relative). The spline's
# ``fit_reml`` row is master (daf11a862)'s at ``CONVERGED_REML``.
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
        "0x1.ff76266bff2d2p-1",
        "0x1.0368ce116321ap+0",
        "0x1.121eb50bb1ab9p+0",
        "0x1.00425a8fa85bap+0",
        "0x1.cdf5ccfdc7202p-1",
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


def _fit(X, y, features, interactions=(), *, method="fit", family=None, fit_kwargs=None, **kwargs):
    model = SuperGLM(
        family=family or superglm.families.gamma(),
        link="log",
        features=features,
        interactions=list(interactions),
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(model, method)(X, y, **(fit_kwargs or {}))
    return model


X_SPECS = {"numeric": Numeric, "spline": lambda: Spline(n_knots=6)}


def _row_j(x_name, method, **fit_kwargs):
    features = {"x": X_SPECS[x_name](), "f": Categorical()}
    return _fit(ROW_J, Y, features, [("x", "f")], method=method, fit_kwargs=fit_kwargs)


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


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("method", ["fit", "fit_reml"])
@pytest.mark.parametrize("x_name", list(X_SPECS))
def test_row_j_off_its_level_value_is_warned_at_masters_price(x_name, method):
    """Level E quoted at x = 4 and 6 leans on the alias; the price is master's.

    ``rows`` names 1 and 2 only.  The price is master's to ``1e-9`` relative:
    every row, the off-span ones included, whose value is the fit's gauge.
    The fit's own rows sit on the span with the other new rows.  The spline's
    REML price is compared at REML's optimum (``CONVERGED_REML``), which does
    not depend on where the default stop lands.
    """
    converged = CONVERGED_REML if (x_name, method) == ("spline", "fit_reml") else {}
    model = _row_j(x_name, method, **converged)
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

    The record's data null space has one direction, and the penalised Hessian's
    has none, so its squared sine to that space is one, far above
    ``FREE_SINE_CUT``.  Row J's penalised spline alias, which no penalty
    curves, is shared by both records: its squared sine is rounding.  No
    tested direction means no design pass.
    """
    from superglm.model import estimability_ops

    cut = estimability_ops.FREE_SINE_CUT
    model, X = _random_effect_fit()
    record = model.result.rank_info.data
    assert record.width - record.rank == 1
    assert estimability_ops.data_null_directions(model) is None
    dual = estimability_ops._dual_null_basis(record)
    scale = np.where(record.column_scale > 0.0, record.column_scale, 1.0)
    _, outside = estimability_ops._penalty_free(model.result.rank_info, dual, scale)
    assert np.min(outside) > cut * MARGIN
    row_j = estimability_ops.data_null_directions(_row_j("spline", "fit_reml"))
    shared = row_j.outside[row_j.outside <= cut]
    assert shared.size == row_j.dual.shape[1] >= 1
    assert np.max(shared) < cut / MARGIN
    assert np.min(row_j.outside[row_j.outside > cut], initial=1.0) > cut * MARGIN

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


def _nested_geography():
    """The Poisson tariff from #692, with regions nested within areas."""
    rng = np.random.default_rng(1)
    n = 4000
    region = rng.choice(["r1", "r2", "r3", "r4"], n)
    area = np.where(np.isin(region, ["r1", "r2"]), "N", "S")
    x = rng.uniform(0, 1, n)
    grp = rng.choice(list("abcdef"), n)
    y = rng.poisson(0.1 * np.exp(0.5 * x) * np.where(area == "S", 1.3, 1.0)).astype(float)
    X = pd.DataFrame({"area": area, "region": region, "x": x, "grp": grp})
    new = pd.DataFrame(
        {"area": ["N", "S", "N"], "region": ["r1"] * 3, "x": [0.5] * 3, "grp": ["a"] * 3},
        index=[12, -9, 12],
    )
    return X, y, new


def _binned_geography(X, y, *, per_term=False, method="fit_reml", **kwargs):
    return _fit(
        X,
        y,
        {
            "area": Categorical(base="N"),
            "region": Categorical(base="r1"),
            "x": Spline(n_knots=6, m=2, discrete=True if per_term else None),
        },
        family=superglm.families.poisson(),
        method=method,
        selection_penalty=0.0,
        discrete=not per_term,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("per_term", "method"), [(False, "fit_reml"), (True, "fit_reml"), (False, "fit")]
)
def test_binned_nested_geography_warns_once(per_term, method):
    from superglm._group_matrix._group_matrix_discretized import DiscretizedSSPGroupMatrix

    X, y, new = _nested_geography()

    def alias(frame):
        return (frame["area"] == "S").astype(int) - frame["region"].isin(["r3", "r4"])

    np.testing.assert_array_equal(alias(X), 0)
    np.testing.assert_array_equal(alias(new), [0, 1, 0])
    model = _binned_geography(X, y, per_term=per_term, method=method)
    assert model.result.direct_backend == "gram"
    block = model._dm.group_matrices[-1]
    assert type(block) is DiscretizedSSPGroupMatrix
    assert block.n_bins < len(X)

    mu, found = _disclosures(lambda: model.predict(new))
    np.testing.assert_array_equal(mu, model._predict_exact(new))
    assert [w.rows.tolist() for w in found] == [[1]]
    assert "the training data do not determine (rows 1)" in str(found[0])


@pytest.mark.pins_behaviour
def test_default_binned_spline_is_silent_on_a_training_copy():
    """Pin silence on training quotes with default bins and a second-order penalty."""
    X, y, _ = _nested_geography()
    model = _binned_geography(X, y)
    _, found = _disclosures(lambda: model.predict(X.copy()))
    assert not found


def test_coarse_binned_fit_checks_exact_rows_against_fitted_span():
    from superglm._group_matrix._group_matrix_bins import discretize_positive_support
    from superglm.features._spline_runtime import bin_knots

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
    spline = model._specs["x"]
    x = X["x"].to_numpy()
    weights = np.ones(len(X))
    representatives, codes = discretize_positive_support(
        x, 4, weights, bin_knots(spline, x, weights)
    )
    block = model._dm.group_matrices[0]
    np.testing.assert_array_equal(codes, block.bin_idx)
    assert representatives.size == 4

    def design(x):
        return np.column_stack((np.ones(len(x)), spline.transform(x)))

    # An independent SVD of the public-basis design actually seen by the fit.
    _, singular, basis = np.linalg.svd(design(representatives[codes]), full_matrices=False)
    assert np.count_nonzero(singular > singular[0] * 1e-12) == 4
    assert np.linalg.matrix_rank(design(x)) == basis.shape[1]
    new = X.iloc[[299, 0, 180, 77]].copy()
    new.iloc[[0, 2, 3], 0] = representatives[[3, 2, 1]]
    projection = np.linalg.norm(design(new["x"].to_numpy()) @ basis[4:].T, axis=1)
    assert np.max(projection[[0, 2, 3]]) < 1e-10
    assert projection[1] > 1e-3

    mu, found = _disclosures(lambda: model.predict(new))
    np.testing.assert_array_equal(mu, model._predict_exact(new))
    assert [w.rows.tolist() for w in found] == [[1]]
    _, reordered = _disclosures(lambda: model.predict(new.iloc[[2, 3, 0, 1]].copy()))
    assert [w.rows.tolist() for w in reordered] == [[3]]


def test_new_compact_binned_fit_keeps_warning_on_pickle_round_trip():
    X, y, new = _nested_geography()
    model = _binned_geography(X, y, retain_fit_state=False)
    saved = pickle.loads(pickle.dumps(model))
    assert saved._dm is None
    mu, found = _disclosures(lambda: saved.predict(new))
    np.testing.assert_array_equal(mu, model._predict_exact(new))
    assert [w.rows.tolist() for w in found] == [[1]]


@pytest.mark.pins_behaviour
def test_legacy_binned_pickle_stays_silent():
    """Pin silence for a binned save with rank records and no binned warning record."""
    X, y, new = _nested_geography()
    model = _binned_geography(X, y, retain_fit_state=False)
    assert model.result.rank_info.data is not None
    assert model.result.rank_info.augmented is not None
    model.__dict__.pop("_binned_estimability", None)
    saved = pickle.loads(pickle.dumps(model))
    mu, found = _disclosures(lambda: saved.predict(new))
    np.testing.assert_array_equal(mu, model._predict_exact(new))
    assert not found


@pytest.mark.pins_behaviour
def test_an_sz_fit_is_silent():
    """Control (passes on master; the disclosure once crashed on these rows).

    Level E sits at x = 5 alone; ``predict`` prices E on the population curve
    and says so (``SZ_POPULATION_PREDICTION``).  A fit with a factor smooth is
    not tested.
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


@pytest.mark.parametrize("x_name", list(X_SPECS))
def test_a_compact_fits_rows_are_tested_from_its_records(x_name):
    """``retain_fit_state=False`` keeps no design but both rank records, saved and loaded too."""
    model = _fit(
        ROW_J,
        Y,
        {"x": X_SPECS[x_name](), "f": Categorical()},
        [("x", "f")],
        retain_fit_state=False,
    )
    for fitted in (model, pickle.loads(pickle.dumps(model))):
        mu, found = _disclosures(lambda fitted=fitted: fitted.predict(ROW_J_NEW))
        assert [w.rows.tolist() for w in found] == [[1, 2]]
        np.testing.assert_allclose(mu, _master((x_name, "fit")), rtol=1e-9)


def test_the_shares_are_the_same_across_row_blocks(monkeypatch):
    """Rows formed in blocks of a few rows read the shares of one block (``prediction_shares``).

    Spline, categorical, numeric and interaction terms cross the boundary; the
    scaling is per row, so the shares agree to rounding and the same rows are
    named.
    """
    from superglm.model import estimability_ops

    X, y, features = _aliased()
    fixtures = [
        (_row_j("spline", "fit"), pd.concat([ROW_J, ROW_J_NEW], ignore_index=True)),
        (
            _fit(X, y, features, family=superglm.families.poisson()),
            pd.concat([X, ALIASED_NEW], ignore_index=True),
        ),
    ]
    for model, frame in fixtures:
        whole = _shares(model, frame)
        monkeypatch.setattr(estimability_ops, "_BLOCK_ENTRIES", 61 * 8)
        blocked = _shares(model, frame)
        monkeypatch.undo()
        # a share is at most one; the training rows read rounding either way
        np.testing.assert_allclose(blocked, whole, rtol=1e-12, atol=16 * 2.0**-53)
        assert np.array_equal(blocked > TOL, whole > TOL)


# ------------------------------------------------------------ round-2 review reproducers


def _duplicated(scale=1.0):
    x = np.arange(-2.0, 3.0).repeat(50) * scale
    return pd.DataFrame({"x": x, "dup": x})


@pytest.mark.pins_behaviour
def test_a_slotted_custom_feature_spec_fits_and_predicts():
    """Control (passes on master; at b839308ef the fit raised on a slotted spec)."""

    class SlottedNumeric:
        __slots__ = ()
        build = Numeric.build
        transform = Numeric.transform
        reconstruct = Numeric.reconstruct

    X = pd.DataFrame({"x": np.arange(-2.0, 3.0).repeat(50)})
    model = _fit(
        X,
        np.full(len(X), 2.0),
        {"x": SlottedNumeric()},
        family=superglm.families.poisson(),
        selection_penalty=0.0,
    )
    mu, found = _disclosures(lambda: model.predict(X.iloc[:1]))
    np.testing.assert_allclose(mu, [2.0], rtol=1e-9)
    assert not found


def test_a_row_at_the_float64_limit_is_warned_without_overflow():
    """``dup = 1e307`` over columns of norm 3.2e-3: priced as master, warned, nothing overflows.

    Each row is scaled by a power of two before its division by the column
    scale, so neither the default nor ``over="raise"`` sees an overflow.
    """
    X = _duplicated(1e-4)
    model = _fit(
        X,
        np.full(len(X), 2.0),
        {"x": Numeric(), "dup": Numeric()},
        family=superglm.families.poisson(),
        selection_penalty=0.0,
    )
    new = pd.DataFrame({"x": [0.0], "dup": [1e307]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(all="raise"):
            mu = model.predict(new)
    np.testing.assert_allclose(mu, [2.0], rtol=1e-9)
    assert [type(c.message).__name__ for c in caught] == ["NonEstimablePredictionWarning"]
    assert caught[0].message.rows.tolist() == [0]
    np.testing.assert_allclose(_shares(model, new), [np.sqrt(0.5)], rtol=1e-6)


def test_the_warning_pickles_with_its_rows():
    """A warning raised in a worker process reaches its parent whole (``__reduce__``)."""
    X = _duplicated()
    model = _fit(
        X,
        np.full(len(X), 2.0),
        {"x": Numeric(), "dup": Numeric()},
        family=superglm.families.poisson(),
        selection_penalty=0.0,
    )
    _, (warning,) = _disclosures(lambda: model.predict(pd.DataFrame({"x": [0.0], "dup": [1.0]})))
    restored = pickle.loads(pickle.dumps(warning))
    assert type(restored) is type(warning)
    assert restored.rows.tolist() == [0] and str(restored) == str(warning)


def test_two_concurrent_predictions_each_warn(monkeypatch):
    """No warning filter moves, so two threads predicting at once each get their warning."""
    X = _duplicated()
    model = _fit(
        X,
        np.full(len(X), 2.0),
        {"x": Numeric(), "dup": Numeric()},
        family=superglm.families.poisson(),
        selection_penalty=0.0,
    )
    new = pd.DataFrame({"x": [0.0], "dup": [1.0]})
    entered, leave = threading.Event(), threading.Event()
    original = Numeric.transform

    def transform(self, values):
        if not entered.is_set():
            entered.set()
            assert leave.wait(10.0)
        return original(self, values)

    monkeypatch.setattr(Numeric, "transform", transform)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        worker = threading.Thread(target=model.predict, args=(new,))
        worker.start()
        assert entered.wait(10.0)
        model.predict(new)
        leave.set()
        worker.join(10.0)
    rows = [c.message.rows.tolist() for c in caught if hasattr(c.message, "rows")]
    assert rows == [[0], [0]]


@pytest.mark.pins_behaviour
def test_an_sz_population_quote_of_an_unseen_level_is_priced():
    """Control (passes on master at 0.91635282; at b839308ef ``unseen="error"`` raised)."""
    model = _fit(
        ROW_J,
        Y,
        {"x": Spline(n_knots=6)},
        [FactorSmooth("x", group="f", basis="sz", k=6, unseen="error")],
        method="fit_reml",
    )
    new = pd.DataFrame({"x": [4.0], "f": ["unseen"]})
    mu, found = _disclosures(lambda: model.predict(new, random_effects="population"))
    np.testing.assert_allclose(mu, [0.91635282], rtol=1e-7)
    assert not found


@pytest.mark.pins_behaviour
def test_an_fs_fit_with_an_aliased_column_is_silent():
    """Control (passes on master): a factor smooth's fit is not tested, conditional or population."""
    local = np.random.default_rng(913)
    v = local.uniform(0.0, 1.0, 2000)
    X = pd.DataFrame({"x": v, "u": v, "g": local.choice(list("ABCDE"), len(v))})
    model = _fit(
        X,
        local.poisson(2.0, len(X)).astype(float),
        {"x": Numeric(), "u": Numeric()},
        [FactorSmooth("x", group="g", basis="fs", k=5)],
        method="fit_reml",
        family=superglm.families.poisson(),
        selection_penalty=0.0,
        direct_solve="gram",
    )
    new = pd.DataFrame({"x": [0.2, 0.2], "u": [0.2, 0.8], "g": ["unseen", "unseen"]})
    for mode in ("conditional", "population"):
        _, found = _disclosures(lambda mode=mode: model.predict(new, random_effects=mode))
        assert not found


@pytest.mark.pins_behaviour
def test_a_structured_fit_is_silent():
    """Control (passes on master): a structured fit keeps no dense rank records, so it is silent."""
    local = np.random.default_rng(952)
    n = 10000
    x = local.integers(-4, 5, n).astype(float)
    g = local.integers(0, 80, n)
    X = pd.DataFrame({"x": x, "dup": x, "g": g})
    y = local.poisson(np.exp(0.01 * x + 0.1 * np.sin(g))).astype(float)
    model = _fit(
        X,
        y,
        {"x": Numeric(), "dup": Numeric(), "g": RandomEffect()},
        method="fit_reml",
        family=superglm.families.poisson(),
        selection_penalty=0.0,
        direct_solve="auto",
    )
    assert model.result.rank_info is None
    mu, found = _disclosures(
        lambda: model.predict(pd.DataFrame({"x": [0.0], "dup": [1.0], "g": [0]}))
    )
    assert np.all(np.isfinite(mu)) and not found


def test_a_binned_spline_on_four_values_warns_between_them():
    """An unpenalised spline on four values, quoted between them at 1.5.

    Binning preserves all four values, so both fits warn at the same price.
    """
    X = pd.DataFrame({"x": np.tile([0.0, 1.0, 2.0, 3.0], 100)})
    y = np.random.default_rng(913).poisson(2.0, len(X)).astype(float)
    new = pd.DataFrame({"x": [1.5]})
    prices = {}
    for discrete in (False, True):
        model = _fit(
            X,
            y,
            {"x": Spline(kind="ps", k=8, knot_strategy="uniform")},
            family=superglm.families.poisson(),
            selection_penalty=0.0,
            spline_penalty=0.0,
            discrete=discrete,
            n_bins=256,
        )
        prices[discrete], found = _disclosures(lambda model=model: model.predict(new))
        assert [w.rows.tolist() for w in found] == [[0]]
    np.testing.assert_allclose(prices[True], prices[False], rtol=1e-9)


@pytest.mark.pins_behaviour
def test_the_rating_table_carries_no_marks():
    """Control (passes on master): exact banding of a spline-by-level term exports, unmarked."""
    local = np.random.default_rng(913)
    xv = np.tile(np.arange(4.0).repeat(2), 100)
    X = pd.DataFrame({"x": xv, "f": np.tile(["A", "B"], len(xv) // 2)})
    y = local.poisson(2.0, len(X)).astype(float)
    model = _fit(
        X,
        y,
        {"x": Spline(kind="ps", k=8, knot_strategy="uniform"), "f": Categorical(base="first")},
        [("x", "f")],
        family=superglm.families.poisson(),
        selection_penalty=0.0,
        spline_penalty=0.0,
    )
    take = np.repeat(np.arange(len(X)), 1 + (xv == 0.0).astype(int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = model.rating_table_payload(
            X.iloc[take],
            y[take],
            n_bins=1,
            bin_strategy="exact",
            impact_bins=(),
            band_max_error=100.0,
            band_se=100.0,
        )
    for block in (*payload.main_effects, *payload.interactions):
        assert not hasattr(block, "non_estimable")


def test_a_huge_row_over_tiny_columns_is_warned_without_overflow():
    """Columns of scale ``1e-150`` quoted at ``dup = 1e170``: the share is ``sqrt(1/2)``, not NaN.

    The row is scaled by a power of two before it is divided by the column
    norms, so the warning is given and ``over="raise"`` finds nothing.
    """
    X = _duplicated(1e-150)
    model = _fit(
        X,
        np.full(len(X), 2.0),
        {"x": Numeric(), "dup": Numeric()},
        family=superglm.families.poisson(),
        selection_penalty=0.0,
    )
    new = pd.DataFrame({"x": [0.0], "dup": [1e170]})
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mu, found = _disclosures(lambda: model.predict(new))
    assert [w.rows.tolist() for w in found] == [[0]]
    np.testing.assert_array_equal(mu, base.predict_exact(model, new, warn=False))
    np.testing.assert_allclose(_shares(model, new), [np.sqrt(0.5)], rtol=1e-6)


@pytest.mark.pins_behaviour
def test_a_subnormal_entry_is_priced_without_overflow():
    """Control (passes on master; at 6260329c6 the disclosure pass overflowed).

    ``dup = 1e-309`` (subnormal): priced as master, no overflow, and no warning.

    The intercept is part of the row, so the row is on the determined space. Each
    power-of-two scaling is applied to the entries themselves, so no
    multiplier overflows where the scaled entry is finite.
    """
    X = _duplicated()
    model = _fit(
        X,
        np.full(len(X), 2.0),
        {"x": Numeric(), "dup": Numeric()},
        family=superglm.families.poisson(),
        selection_penalty=0.0,
    )
    new = pd.DataFrame({"x": [0.0], "dup": [1e-309]})
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mu, found = _disclosures(lambda: model.predict(new))
    np.testing.assert_array_equal(mu, base.predict_exact(model, new, warn=False))
    assert not found
