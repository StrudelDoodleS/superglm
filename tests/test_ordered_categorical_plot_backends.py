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
    # The whole curve is on canvas, and since issue #282 that is no longer a
    # near miss: the expansion stopped rebuilding the curve, so the drawn curve
    # is the FITTED one and spans the fitted axis (group means, 31 .. 72) --
    # which is exactly the collapsed markers' own range. The limits then hold it
    # with the marker padding to spare on both sides.
    #
    # Before, the expansion interpolated a PCHIP across the original level range
    # (21 .. 72) and the collapse passed that through, so the curve overhung the
    # markers by 10 units and the left limit had to be the curve's own start.
    # The union in ``_plot_ordered_spline_panel`` still guards that direction --
    # it is what keeps this an inequality rather than an assumption.
    import matplotlib

    matplotlib.use("Agg")

    X, sample_weight, model = collapsed_ordered_spline_model
    curve = model.term_inference("age_band").smooth_curve
    curve_x = np.asarray(curve.x, dtype=np.float64)

    ax = model.plot("age_band", X=X, sample_weight=sample_weight).axes[0]
    lo, hi = ax.get_xlim()
    x_pos = np.asarray(ax.get_xticks(), dtype=np.float64)

    np.testing.assert_allclose(x_pos, [np.mean([21.0, 30.0, 42.0]), 57.0, 72.0])
    # The fitted curve starts at the first marker and ends at the last: the
    # collapse is the exact inverse of the expansion on the level positions.
    assert float(curve_x.min()) == pytest.approx(float(x_pos.min()))
    assert float(curve_x.max()) == pytest.approx(float(x_pos.max()))
    # Both edges keep the marker padding, so the whole curve is inside them.
    assert lo < float(curve_x.min())
    assert hi > float(curve_x.max())

    drawn_x, _ = _matplotlib_curve(ax, len(curve_x))
    assert float(drawn_x.min()) >= lo and float(drawn_x.max()) <= hi


def test_single_group_collapse_still_shows_the_whole_curve(single_group_collapsed_model):
    # Degenerate display: every level lands in one group, so there is one marker
    # and no spacing to derive.  The panel must still show all of the curve.
    #
    # Since issue #282 that curve is the fitted one, and one group is ONE
    # parameter -- so the fitted function is a single number at a single
    # coordinate (the mean of the five level values, 44.4) and the curve is 200
    # samples of it there.  The panel shows exactly that.  It used to show a
    # PCHIP drawn from 21 to 72 through five markers that all carry the same
    # value, which is a line across a range the model no longer distinguishes:
    # Cattaneo, Crump, Farrell & Feng, *On Binscatter*, AER 114(5):1488-1514
    # (2024) -- "although the binned scatter plot invites the viewer to 'connect
    # the dots' smoothly, the actual estimator is piecewise constant".
    import matplotlib

    matplotlib.use("Agg")

    X, sample_weight, single = single_group_collapsed_model

    ti = single.term_inference("age_band")
    display = project_grouped_term_for_display(single, ti, "auto")
    assert len(display.term.levels) == 1

    curve_x = np.asarray(ti.smooth_curve.x, dtype=np.float64)
    marker = float(np.mean(list(AGE_VALUES.values())))
    assert float(curve_x.min()) == pytest.approx(marker)
    assert float(curve_x.max()) == pytest.approx(marker)

    ax = single.plot("age_band", X=X, sample_weight=sample_weight).axes[0]
    lo, hi = ax.get_xlim()

    x_pos = np.asarray(ax.get_xticks(), dtype=np.float64)
    assert x_pos.size == 1
    assert float(x_pos[0]) == pytest.approx(marker)
    # a degenerate curve still has to be on canvas, with a window around it
    assert lo < float(curve_x.min()) and hi > float(curve_x.max())


def test_plotly_expanded_panel_shows_a_curve_narrower_than_its_markers(
    collapsed_ordered_spline_model,
):
    """The combination issue #282 creates, on the backend that does not union.

    ``expanded`` mode is the only place the axis mismatch survives: the markers
    sit at the declared level values (21 .. 72) while the curve spans the axis
    the smooth was fitted on (31 .. 72), because the leading group sits at its
    members' mean. That is the ggplot2 ``geom_smooth(fullrange=FALSE)``
    arrangement -- the line is not expanded to the range of the plot -- and the
    panel has to hold both.

    matplotlib guarantees it by unioning the marker padding with the curve's
    extent, which ``test_collapsed_panel_x_limits_hold_the_whole_curve`` pins.
    Plotly gets there differently: it sets no x-range at all and lets autorange
    cover every trace. That is arguably more robust, since it also covers the
    exposure bars and a detached ``specials=`` block -- but it is an absence
    rather than a guarantee, and a single ``update_xaxes(range=...)`` added
    later for an unrelated reason would clip the curve with nothing failing.
    So the property is asserted here rather than left to the absence.
    """
    go = pytest.importorskip("plotly.graph_objects")

    X, sample_weight, model = collapsed_ordered_spline_model
    ti = model.term_inference("age_band")
    curve_x = np.asarray(ti.smooth_curve.x, dtype=np.float64)
    declared = np.asarray(sorted(AGE_VALUES.values()), dtype=np.float64)

    # the premise: the fitted curve really is narrower than the drawn markers
    assert float(curve_x.min()) > float(declared.min())
    assert float(curve_x.max()) == pytest.approx(float(declared.max()))

    fig = model.plot(
        ["age_band", "mileage"],
        engine="plotly",
        X=X,
        sample_weight=sample_weight,
        grouped_level_display="expanded",
    )
    curve = next(
        trace
        for trace in fig.data
        if isinstance(trace, go.Scatter) and trace.name == "Smooth curve"
    )
    np.testing.assert_allclose(np.asarray(curve.x, dtype=np.float64), curve_x)

    # No explicit range is set, on either the initial layout or the term
    # dropdown's update, so autorange covers every trace on the axis.
    for axis in ("xaxis", "xaxis2"):
        layout_axis = fig.layout[axis] if axis in fig.layout else None
        if layout_axis is not None:
            assert layout_axis.range is None, f"{axis} pins a range, which can clip the curve"
    # The Y range IS pinned per term (it is computed from that term's
    # relativities); it is only an X range that could clip the curve.
    for button in fig.layout.updatemenus[-1].buttons:
        for key in button.args[1] if len(button.args) > 1 else {}:
            assert not (key.startswith("xaxis") and key.endswith(".range")), (
                f"{key} pins an x-range on a term switch, which can clip the curve"
            )

    # and the declared markers really are on the panel beside the narrower curve
    tickvals = np.asarray(fig.layout.xaxis.tickvals, dtype=np.float64)
    np.testing.assert_allclose(np.sort(tickvals), declared)


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
