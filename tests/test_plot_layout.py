"""Readable effect/support layouts without changing the plotted values."""

from dataclasses import replace
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from superglm.inference._term_types import SmoothCurve, TermInference
from superglm.plotting import plot_relativities, plot_term


def _term(kind):
    rel = np.array([0.8, 1.0, 1.4])
    ti = TermInference(
        name="feature",
        kind="categorical",
        active=True,
        levels=["A", "B", "Unknown"],
        relativity=rel,
        ci_lower=rel * 0.9,
        ci_upper=rel * 1.1,
    )
    if kind == "ordered":
        x = np.linspace(2.0, 5.0, 20)
        curve_rel = np.exp(np.linspace(np.log(0.8), 0.0, len(x)))
        ti = replace(
            ti,
            smooth_curve=SmoothCurve(x, np.log(curve_rel), curve_rel, np.array([2.0, 5.0])),
            level_is_special=np.array([False, False, True]),
        )
    elif kind == "spline":
        ti = replace(ti, kind="spline", levels=None, x=np.array([0.0, 0.5, 1.0]))
    return ti


@pytest.mark.parametrize("kind", ["categorical", "ordered", "spline"])
@pytest.mark.parametrize("grid", [False, True])
def test_support_sits_below_effect_with_shared_x_limits(kind, grid):
    ti = _term(kind)
    values = ["B", "A", "Unknown", "B"] if kind != "spline" else [0.0, 0.2, 0.6, 1.0]
    X = pd.DataFrame({"feature": values})
    plot = plot_relativities if grid else plot_term
    fig = plot([ti] if grid else ti, X=X, sample_weight=[1.0, 2.0, 3.0, 4.0])
    try:
        fig.canvas.draw()
        effect, support = fig.axes
        assert support.get_position().y1 < effect.get_position().y0
        np.testing.assert_array_equal(effect.get_xlim(), support.get_xlim())
        assert not effect.get_xticklabels()
        assert support.get_xticklabels()
        if kind != "spline":
            np.testing.assert_array_equal([bar.get_height() for bar in support.patches], [2, 5, 3])
            positions = [2.0, 5.0, 11.0] if kind == "ordered" else [0.0, 1.0, 2.0]
            np.testing.assert_allclose(
                [bar.get_x() + bar.get_width() / 2 for bar in support.patches], positions
            )
            assert [tick.get_text() for tick in support.get_xticklabels()] == ti.levels
        support.set_xlim(-1.0, 12.0)
        assert effect.get_xlim() == (-1.0, 12.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("grid", [False, True])
def test_figure_heading_and_legend_do_not_overlap_panels(grid):
    terms = [replace(_term("spline"), name=f"Feature {i}") for i in range(4)]
    X = pd.DataFrame({ti.name: np.linspace(0.0, 1.0, 10) for ti in terms})
    plot = plot_relativities if grid else plot_term
    fig = plot(
        terms if grid else terms[0],
        X=X,
        title="Model effects",
        subtitle="Synthetic example\nEffects and observation support",
    )
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        headings = [text.get_window_extent(renderer) for text in fig.texts]
        headings += [legend.get_window_extent(renderer) for legend in fig.legends]
        assert all(not a.overlaps(b) for a, b in combinations(headings, 2))
        for ax in fig.axes:
            assert all(not heading.overlaps(ax.get_tightbbox(renderer)) for heading in headings)
    finally:
        plt.close(fig)


def test_plot_style_does_not_change_matplotlib_defaults():
    with matplotlib.rc_context({"axes.facecolor": "#eeeeee", "font.size": 17}):
        before = dict(matplotlib.rcParams)
        fig = plot_term(_term("categorical"))
        try:
            assert dict(matplotlib.rcParams) == before
        finally:
            plt.close(fig)


def test_tight_export_includes_the_figure_title():
    fig = plot_term(_term("categorical"), title="Model effects")
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title = fig._suptitle.get_window_extent(renderer).transformed(
            fig.dpi_scale_trans.inverted()
        )
        bounds = fig.get_tightbbox(renderer)
        assert bounds.contains(title.x0, title.y0)
        assert bounds.contains(title.x1, title.y1)
    finally:
        plt.close(fig)


def test_incomplete_grid_keeps_an_invisible_axis_for_the_unused_cell():
    terms = [replace(_term("categorical"), name=f"Feature {i}") for i in range(3)]
    fig = plot_relativities(terms, ncols=2, show_exposure=False)
    try:
        axes = np.asarray(fig.axes).reshape(2, 2)
        assert [ax.get_visible() for ax in axes.flat] == [True, True, True, False]
    finally:
        plt.close(fig)
