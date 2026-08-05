"""The two plot backends must draw the same fitted curve for an OC term."""

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, OrderedCategorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.plotting.group_display import project_grouped_term_for_display

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


@pytest.fixture
def collapsed_ordered_spline_model():
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
    session.select_levels("age_band", ["18-24", "25-34", "35-49"])
    collapsed = session.replace_with_collapsed_levels("age_band", method="fit")
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
