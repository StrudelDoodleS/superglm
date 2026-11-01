"""Compensated numeric edits preserve drop diagnostics and bounded replay (#456)."""

from __future__ import annotations

import warnings
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from superglm import Constraint, Numeric, PSpline, SuperGLM
from superglm.group_matrix import SparseGroupMatrix
from superglm.solvers import mode_score
from tests.test_coefficient_revisions import (
    _centred_magnitude,
    _far_slope_edit,
    _gamma,
    _numeric_fit,
    _public_columns,
    _slope_edited,
)


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("mixed", [False, True], ids=["lower_limit", "both_limits"])
def test_dropping_an_overflowed_numeric_term_rebuilds_the_remaining_predictor(sign, mixed):
    """Removing x must not subtract infinity from infinity or retain its carried centre.

    All-negative eta keeps the full deviance moderate, so the delta also
    distinguishes the correct retained intercept from a dropped intercept.
    Mixed signs cover the issue's original two clipping limits.
    """
    _, edited, frame, y = _far_slope_edit("poisson", sign * 1e308, ("x", "z"))
    x = sign * np.array([1e300, -1e300] if mixed else [-1e300, -2e300])
    holdout = pd.DataFrame({"x": x, "z": [2.0, 3.0]})
    target = np.array([2.0, 3.0])
    z_group = next(group for group in edited._groups if group.feature_name == "z")
    eta_drop = float(edited.result.intercept) + holdout["z"].to_numpy() * float(
        edited.result.beta[z_group.sl][0]
    )
    mu_drop = np.exp(eta_drop)
    # Independent Poisson unit deviance, applied to the raw predictor with x removed.
    dev_drop = float(np.sum(2.0 * (target * np.log(target / mu_drop) - (target - mu_drop))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu_full = edited.predict(holdout)
        table = edited.term_drop_diagnostics(
            frame, y, mode="holdout", X_val=holdout, y_val=target
        ).set_index("feature")
    dev_full = float(np.sum(2.0 * (target * np.log(target / mu_full) - (target - mu_full))))
    delta = float(table.loc["x", "delta_deviance"])
    assert np.isfinite(delta)
    assert delta == pytest.approx(dev_drop - dev_full, rel=2e-14, abs=2e-12)
    assert table.loc["z", "delta_deviance"] == 0.0


def test_nonzero_holdout_rows_detect_an_ignored_drop_handle():
    """The earlier x=0 fixture cannot detect an ignored without= argument."""
    frame = pd.DataFrame({"x": 1e16 * np.resize([0.0, 1.0, 2.0], 60)})
    y = np.resize([2.0, 3.0, 2.0], 60)
    model = _numeric_fit(frame, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edited = _slope_edited(model, frame, y, {"x": 1.1})
    holdout = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    target = np.array([2.0, 3.0, 2.0])
    intercept = float(model.result.intercept)
    full = intercept + 1.1 * holdout["x"].to_numpy()
    expected = float(np.sum((target - intercept) ** 2) - np.sum((target - full) ** 2))
    table = edited.term_drop_diagnostics(
        frame, y, mode="holdout", X_val=holdout, y_val=target
    ).set_index("feature")
    assert table.loc["x", "delta_deviance"] == pytest.approx(expected, rel=2e-13, abs=2e-12)


def test_overflowed_drop_rebuilds_after_the_centred_pair_was_discarded():
    year = 2000.0 + np.resize(np.arange(20.0), 60)
    frame = pd.DataFrame({"x": year})
    y = np.exp(0.5 + 0.01 * (year - 2010.0)) + np.resize([0.1, -0.1, 0.05], 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(family="poisson", selection_penalty=0, features={"x": Numeric()}).fit(
            frame, y
        )
        edited = _slope_edited(model, frame, y, {"x": 1e306})
        assert not edited.result.centred_sum_compensated
        holdout = pd.DataFrame({"x": [2000.0, 2001.0]})
        target = np.array([2.0, 3.0])
        mu_full = edited.predict(holdout)
        table = edited.term_drop_diagnostics(
            frame, y, mode="holdout", X_val=holdout, y_val=target
        ).set_index("feature")
    mu_drop = np.exp(float(edited.result.intercept))
    dev_drop = np.sum(2 * (target * np.log(target / mu_drop) - (target - mu_drop)))
    dev_full = np.sum(2 * (target * np.log(target / mu_full) - (target - mu_full)))
    assert np.isfinite(table.loc["x", "delta_deviance"])
    assert table.loc["x", "delta_deviance"] == pytest.approx(dev_drop - dev_full, rel=2e-14)


def test_overflowed_drop_preserves_cancellation_in_a_remaining_numeric_term():
    x = np.resize([0.01, 0.02, 0.03], 120)
    z = 1e16 + np.resize([-4.0, -2.0, 0.0, 2.0, 4.0], 120)
    frame = pd.DataFrame({"x": x, "z": z})
    y = np.exp(0.5 + 0.1 * (z - 1e16))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(
            family="poisson", selection_penalty=0, features={"x": Numeric(), "z": Numeric()}
        ).fit(frame, y)
        edited = _slope_edited(model, frame, y, {"x": 1e308})
        holdout = pd.DataFrame({"x": [-1e300, -1e300], "z": [1e16, 1e16 + 2]})
        target = np.array([2.0, 3.0])
        mu_full = edited.predict(holdout)
        table = edited.term_drop_diagnostics(
            frame, y, mode="holdout", X_val=holdout, y_val=target
        ).set_index("feature")
    group = next(group for group in edited._groups if group.feature_name == "z")
    intercept = Fraction(float(edited.result.intercept))
    beta = Fraction(float(edited.result.beta[group.sl][0]))
    eta_drop = np.array([float(intercept + beta * Fraction(value)) for value in holdout["z"]])
    mu_drop = np.exp(eta_drop)
    dev_drop = np.sum(2 * (target * np.log(target / mu_drop) - (target - mu_drop)))
    dev_full = np.sum(2 * (target * np.log(target / mu_full) - (target - mu_full)))
    assert table.loc["x", "delta_deviance"] == pytest.approx(dev_drop - dev_full, abs=2e-12)


def test_rebuilt_drop_does_not_remove_an_uncompensated_centre_twice():
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 60)})
    y = np.exp(0.5 + 2.0 * frame["x"].to_numpy())
    model = SuperGLM(family="poisson", selection_penalty=0, features={"x": Numeric()}).fit(frame, y)
    assert model.result.state_center is not None
    assert not model.result.centred_sum_compensated
    holdout = pd.DataFrame({"x": [-1e308, -1e308]})
    target = np.array([2.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu_full = model.predict(holdout)
        table = model.term_drop_diagnostics(
            frame, y, mode="holdout", X_val=holdout, y_val=target
        ).set_index("feature")
    mu_drop = np.exp(float(model.result.intercept))
    dev_drop = np.sum(2 * (target * np.log(target / mu_drop) - (target - mu_drop)))
    dev_full = np.sum(2 * (target * np.log(target / mu_full) - (target - mu_full)))
    assert table.loc["x", "delta_deviance"] == pytest.approx(dev_drop - dev_full, abs=2e-12)


def test_shape_repair_after_a_numeric_edit_profiles_the_edited_predictor():
    """Repair must use the numeric edit's published state, not the old solver pair."""
    frame = pd.DataFrame({"x": np.resize([0.01, 0.02, 0.03], 60), "s": np.linspace(0.0, 1.0, 60)})
    y = 2.0 + 0.8 * frame["x"].to_numpy() - frame["s"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            spline_penalty=0.8,
            features={
                "x": Numeric(),
                "s": PSpline(
                    n_knots=6, knot_strategy="uniform", constraint=Constraint.postfit.increasing
                ),
            },
        ).fit(frame, y)
        edited = _slope_edited(model, frame, y, {"x": 1.1})
        assert edited.result.centred_sum_compensated
        before = edited.result.beta.copy()
        edited.apply_shape_postfit(frame, n_grid=80)
    assert not np.array_equal(edited.result.beta, before)
    # At the repair's coefficients, the Gaussian intercept is exactly the
    # mean residual.  Accumulate the public column products independently.
    products = [
        sum(
            (Fraction(float(c)) * Fraction(float(b)) for c, b in zip(row, edited.result.beta)),
            Fraction(0),
        )
        for row in _public_columns(edited, frame)
    ]
    intercept = sum(
        (Fraction(float(value)) - product for value, product in zip(y, products)), Fraction(0)
    ) / len(y)
    reference = np.array([float(intercept + product) for product in products])
    bound = (
        4.0
        * _gamma(len(y) + edited.result.beta.size + 4)
        * (_centred_magnitude(edited, frame) + np.abs(reference))
    )
    assert np.all(np.abs(edited.predict(frame) - reference) <= bound)
    assert np.all(
        np.abs(mode_score.linear_predictor(edited._dm, edited._solver_result, None) - reference)
        <= bound
    )


def _record_expansion_sizes(monkeypatch):
    sizes = []
    original = mode_score._Addend.expand

    def record(self, data, rows, **kwargs):
        sizes.append(len(range(*rows.indices(len(data)))) if isinstance(rows, slice) else len(rows))
        return original(self, data, rows, **kwargs)

    monkeypatch.setattr(mode_score._Addend, "expand", record)
    return sizes


def test_fallback_expands_only_row_chunks_and_recomputes_each_source_once(monkeypatch):
    """A full-array replay defeats the expansion's row-memory limit."""
    monkeypatch.setattr(mode_score, "_CHUNK", 3)
    sizes = _record_expansion_sizes(monkeypatch)
    calls = [0, 0, 0]
    total = mode_score.CompensatedSum(1e308, 0.25, n=11)
    for index, value in enumerate([1e308, 0.0, -1e308]):

        def source(index=index, value=value):
            calls[index] += 1
            return np.full(11, value)

        total.add(source)
    np.testing.assert_array_equal(total.value(), np.full(11, 1e308))
    assert max(sizes) <= 3
    assert calls == [2, 2, 2]


def test_solver_fallback_replays_distinct_loop_scored_blocks_in_row_chunks(monkeypatch):
    """The solver's three non-dense recipes must retain their own block and beta."""
    monkeypatch.setattr(mode_score, "_CHUNK", 3)
    sizes = _record_expansion_sizes(monkeypatch)
    dm = SimpleNamespace(
        n=11,
        group_matrices=[SparseGroupMatrix(sparse.csr_matrix(np.ones((11, 1)))) for _ in range(3)],
    )
    result = SimpleNamespace(
        centred_intercept=1e308,
        centred_intercept_lo=0.25,
        state_center=np.zeros(3),
        centred_sum_compensated=True,
        beta=np.array([1e308, 0.0, -1e308]),
    )
    np.testing.assert_array_equal(mode_score.linear_predictor(dm, result, None), np.full(11, 1e308))
    assert max(sizes) <= 3


def test_fallback_does_not_repeat_scorer_warnings():
    """A scorer warning belongs to its initial score, not its fallback replay."""
    total = mode_score.CompensatedSum(1e308, n=11)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        for index, value in enumerate([1e308, 0.0, -1e308]):

            def source(index=index, value=value):
                warnings.warn(f"scored term {index}", UserWarning, stacklevel=2)
                return np.full(11, value)

            total.add(source)
        np.testing.assert_array_equal(total.value(), np.full(11, 1e308))
    assert [str(item.message) for item in recorded] == [f"scored term {i}" for i in range(3)]


def test_centred_column_plain_product_reports_genuine_overflow():
    """The expansion's guards must not hide the plain numeric product warning."""
    with pytest.warns(RuntimeWarning, match="overflow encountered in multiply") as recorded:
        _, plain = mode_score.centred_column_expansion(np.array([1e300, -1e300]), 0.02, 1e308)
    np.testing.assert_array_equal(plain, [np.inf, -np.inf])
    assert len(recorded) == 1


def test_marked_predictor_reports_product_overflow_once_across_row_chunks(monkeypatch):
    _, edited, _, _ = _far_slope_edit("poisson", 1e308)
    monkeypatch.setattr(mode_score, "_CHUNK", 3)
    holdout = pd.DataFrame({"x": np.resize([1e300, -1e300], 11)})
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        prediction = edited.predict(holdout)
    assert np.all(np.isfinite(prediction))
    assert len(recorded) == 1
    assert recorded[0].category is RuntimeWarning
    assert "overflow encountered in multiply" in str(recorded[0].message)
