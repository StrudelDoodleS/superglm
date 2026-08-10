"""The two plot backends must draw the same fitted curve for an OC term."""

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, OrderedCategorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.plotting.group_display import (
    _collapsed_smooth_curve,
    project_grouped_term_for_display,
)

AGE_VALUES = {
    "18-24": 21.0,
    "25-34": 30.0,
    "35-49": 42.0,
    "50-64": 57.0,
    "65-80": 72.0,
}


def _age_band_frame(seed: int, n: int):
    rng = np.random.default_rng(seed)
    levels = list(AGE_VALUES)
    band = rng.choice(levels, n, p=[0.15, 0.25, 0.28, 0.20, 0.12])
    mileage = rng.normal(0.0, 1.0, n)
    sample_weight = rng.uniform(0.5, 1.5, n)
    age = np.array([AGE_VALUES[value] for value in band], dtype=np.float64)
    y = 0.8 + 0.25 * np.sin(age / 22.0) + 0.04 * mileage + rng.normal(0.0, 0.05, n)
    return pd.DataFrame({"age_band": band, "mileage": mileage}), y, sample_weight


@pytest.fixture
def ordered_spline_model():
    X, y, sample_weight = _age_band_frame(20260805, 800)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "age_band": OrderedCategorical(values=AGE_VALUES, basis=Spline(kind="ps", k=6)),
            "mileage": Numeric(),
        },
    )
    model.fit(X, y, sample_weight=sample_weight)
    return X, sample_weight, model


def _collapse_session():
    X, y, sample_weight = _age_band_frame(20260806, 700)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "age_band": OrderedCategorical(values=AGE_VALUES, basis=Spline(kind="ps", k=5)),
            "mileage": Numeric(),
        },
    )
    model.fit(X, y, sample_weight=sample_weight)
    session = EditorSession.from_model(
        model,
        terms=["age_band"],
        train_data=(X, y, sample_weight),
    )
    return X, sample_weight, session


@pytest.fixture
def collapsed_ordered_spline_model():
    X, sample_weight, session = _collapse_session()
    session.select_levels("age_band", ["18-24", "25-34", "35-49"])
    collapsed = session.replace_with_collapsed_levels("age_band", method="fit")
    return X, sample_weight, collapsed


@pytest.fixture
def single_group_collapsed_model():
    """Every level in one group: the degenerate one-marker display."""
    X, sample_weight, session = _collapse_session()
    session.select_levels("age_band", list(AGE_VALUES))
    # The clone still clamps ``n_knots`` hard here -- five levels collapse to
    # one -- but the clamp is the editor re-fitting the caller's own declared
    # basis to the levels they just merged, so the internal clone stays quiet
    # and only user-facing construction warns.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        collapsed = session.replace_with_collapsed_levels("age_band", method="fit")
    assert not [w for w in caught if "clamped" in str(w.message)]
    return X, sample_weight, collapsed


def _matplotlib_curve(ax, n_points: int):
    """The one drawn line with as many vertices as the fitted curve."""
    lines = [line for line in ax.lines if len(line.get_xdata()) == n_points]
    assert len(lines) == 1, f"expected exactly one curve line, got {len(lines)}"
    return (
        np.asarray(lines[0].get_xdata(), dtype=np.float64),
        np.asarray(lines[0].get_ydata(), dtype=np.float64),
    )


def test_matplotlib_ordered_panel_draws_the_fitted_curve(ordered_spline_model):
    # False today: the panel draws PchipInterpolator(arange(K), relativity) over
    # linspace(0, K-1, 200), so curve_x is [0 .. 4] while the fitted curve
    # spans the level values [21 .. 72].
    import matplotlib

    matplotlib.use("Agg")

    X, sample_weight, model = ordered_spline_model
    ti = model.term_inference("age_band")
    fig = model.plot("age_band", X=X, sample_weight=sample_weight)

    curve_x, curve_y = _matplotlib_curve(fig.axes[0], len(ti.smooth_curve.x))
    np.testing.assert_allclose(curve_x, np.asarray(ti.smooth_curve.x, dtype=np.float64))
    np.testing.assert_allclose(curve_y, np.asarray(ti.smooth_curve.relativity, dtype=np.float64))


def test_matplotlib_ordered_panel_places_levels_at_fitted_positions(ordered_spline_model):
    # False today: x_pos is arange(K), so the ticks are [0, 1, 2, 3, 4] and the
    # exposure bars are centred there instead of at level_x = [21 .. 72].
    import matplotlib

    matplotlib.use("Agg")

    X, sample_weight, model = ordered_spline_model
    ti = model.term_inference("age_band")
    level_x = np.asarray(ti.smooth_curve.level_x, dtype=np.float64)
    fig = model.plot("age_band", X=X, sample_weight=sample_weight)

    ax = fig.axes[0]
    np.testing.assert_allclose(np.asarray(ax.get_xticks(), dtype=np.float64), level_x)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == list(ti.levels)

    bars = fig.axes[1].patches
    centres = np.asarray(
        [patch.get_x() + patch.get_width() / 2.0 for patch in bars], dtype=np.float64
    )
    np.testing.assert_allclose(centres, level_x)


def test_both_backends_draw_the_same_ordered_curve(ordered_spline_model):
    # False today: matplotlib draws a PCHIP over [0, 4] and plotly draws the
    # fitted spline over [21, 72]; the x arrays do not even overlap.
    go = pytest.importorskip("plotly.graph_objects")
    import matplotlib

    matplotlib.use("Agg")

    X, sample_weight, model = ordered_spline_model
    ti = model.term_inference("age_band")
    mpl_fig = model.plot("age_band", X=X, sample_weight=sample_weight)
    mpl_x, mpl_y = _matplotlib_curve(mpl_fig.axes[0], len(ti.smooth_curve.x))

    plotly_fig = model.plot(
        ["age_band", "mileage"],
        engine="plotly",
        X=X,
        sample_weight=sample_weight,
    )
    curve = next(
        trace
        for trace in plotly_fig.data
        if isinstance(trace, go.Scatter) and trace.name == "Smooth curve"
    )
    np.testing.assert_allclose(mpl_x, np.asarray(curve.x, dtype=np.float64))
    np.testing.assert_allclose(mpl_y, np.asarray(curve.y, dtype=np.float64))


def test_collapsed_display_keeps_the_fitted_curve(collapsed_ordered_spline_model):
    # False today: _collapsed_smooth_curve replaces the curve with a PCHIP
    # through the collapsed relativities at arange(3), so curve.x becomes
    # [0 .. 2] and level_x becomes [0, 1, 2] rather than the group-mean
    # positions on the fitted axis.
    X, sample_weight, model = collapsed_ordered_spline_model
    ti = model.term_inference("age_band")
    display = project_grouped_term_for_display(model, ti, "auto")

    assert display.collapsed is True
    assert display.term.levels == ["18-24+25-34+35-49", "50-64", "65-80"]

    np.testing.assert_allclose(
        np.asarray(display.term.smooth_curve.x, dtype=np.float64),
        np.asarray(ti.smooth_curve.x, dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(display.term.smooth_curve.relativity, dtype=np.float64),
        np.asarray(ti.smooth_curve.relativity, dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(display.term.smooth_curve.level_x, dtype=np.float64),
        [np.mean([21.0, 30.0, 42.0]), 57.0, 72.0],
    )


def test_collapsed_panel_x_limits_hold_the_whole_curve(collapsed_ordered_spline_model):
    # Markers sit at group means (31, 57, 72) but the curve still spans the
    # original level range (21 .. 72), so limits derived from the markers alone
    # cut the leading 2.5 units of the fitted curve off-canvas.
    import matplotlib

    matplotlib.use("Agg")

    X, sample_weight, model = collapsed_ordered_spline_model
    curve = model.term_inference("age_band").smooth_curve
    curve_x = np.asarray(curve.x, dtype=np.float64)

    ax = model.plot("age_band", X=X, sample_weight=sample_weight).axes[0]
    lo, hi = ax.get_xlim()
    x_pos = np.asarray(ax.get_xticks(), dtype=np.float64)

    np.testing.assert_allclose(x_pos, [np.mean([21.0, 30.0, 42.0]), 57.0, 72.0])
    # The left edge is the curve's own start, below the marker padding.
    assert lo == pytest.approx(float(curve_x.min()))
    assert lo < float(x_pos.min())
    # The right edge still keeps the marker padding, which reaches past the curve.
    assert hi >= float(curve_x.max())
    assert hi > float(x_pos.max())

    drawn_x, _ = _matplotlib_curve(ax, len(curve_x))
    assert float(drawn_x.min()) >= lo and float(drawn_x.max()) <= hi


def test_single_group_collapse_still_shows_the_whole_curve(single_group_collapsed_model):
    # Degenerate display: every level lands in one group, so there is one marker
    # and no spacing to derive.  The panel must still show all of the curve
    # rather than a 1-unit window around the single marker.
    import matplotlib

    matplotlib.use("Agg")

    X, sample_weight, single = single_group_collapsed_model

    ti = single.term_inference("age_band")
    display = project_grouped_term_for_display(single, ti, "auto")
    assert len(display.term.levels) == 1

    curve_x = np.asarray(ti.smooth_curve.x, dtype=np.float64)
    ax = single.plot("age_band", X=X, sample_weight=sample_weight).axes[0]
    lo, hi = ax.get_xlim()

    assert np.asarray(ax.get_xticks()).size == 1
    assert lo == pytest.approx(float(curve_x.min()))
    assert hi == pytest.approx(float(curve_x.max()))


def test_collapsed_curve_is_dropped_when_it_has_no_level_positions(ordered_spline_model):
    # A curve without level_x cannot be re-positioned onto the collapsed
    # markers, and handing the uncollapsed curve to a display term with fewer
    # levels would draw it in level-value units against arange(n_collapsed).
    _, _, model = ordered_spline_model
    ti = model.term_inference("age_band")
    groups = [[0, 1, 2], [3], [4]]

    assert _collapsed_smooth_curve(ti, groups) is not None

    without_level_x = replace(ti, smooth_curve=replace(ti.smooth_curve, level_x=None))
    assert _collapsed_smooth_curve(without_level_x, groups) is None
    assert _collapsed_smooth_curve(replace(ti, smooth_curve=None), groups) is None


def test_ordered_bar_width_follows_the_shared_level_spacing(ordered_spline_model):
    # _ordered_bar_width delegates to _ordered_level_spacing; pin both the
    # spaced case and the degenerate single-position fallback.
    from superglm.plotting.common import _ordered_level_spacing
    from superglm.plotting.main_effects_plotly import _ordered_bar_width

    _, _, model = ordered_spline_model
    level_x = np.asarray(model.term_inference("age_band").smooth_curve.level_x, dtype=np.float64)

    assert _ordered_level_spacing(level_x) == pytest.approx(9.0)  # min gap 30 - 21
    assert _ordered_bar_width(level_x) == pytest.approx(9.0 * 0.72)
    assert _ordered_level_spacing(np.array([44.4])) == pytest.approx(1.0)
    assert _ordered_bar_width(np.array([44.4])) == pytest.approx(0.72)
