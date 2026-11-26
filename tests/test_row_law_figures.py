"""The figures of one row's law and of the two threshold plots, on both engines.

Each renderer is handed a payload whose fields are set here, and the test reads the figure's
artists (matplotlib) or traces and shapes (plotly): what is drawn, where, in which hue and
style. No image is compared. The last section draws each public plot end to end.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

go = pytest.importorskip("plotly.graph_objects")

from superglm import (  # noqa: E402
    GammaLS,
    GeneralizedParetoLSS,
    SplitSplice,
    SuperGLM,
    SuperLSS,
    mean_excess,
    plot_mean_excess,
    plot_threshold_stability,
    threshold_stability,
)
from superglm.distributional.row_law import DistributionCurve  # noqa: E402
from superglm.distributional.thresholds import (  # noqa: E402
    MeanExcessFigure,
    TailLine,
    ThresholdStabilityResult,
)
from superglm.plotting import (  # noqa: E402
    plot_distribution_curve,
    plot_mean_excess_figure,
    plot_threshold_stability_figure,
)
from superglm.plotting.distributional_plotly import (  # noqa: E402
    plotly_distribution_curve,
    plotly_mean_excess_figure,
    plotly_threshold_stability_figure,
)

ENGINES = ("matplotlib", "plotly")
RENDERERS = {
    "distribution": {"matplotlib": plot_distribution_curve, "plotly": plotly_distribution_curve},
    "mean_excess": {"matplotlib": plot_mean_excess_figure, "plotly": plotly_mean_excess_figure},
    "threshold_stability": {
        "matplotlib": plot_threshold_stability_figure,
        "plotly": plotly_threshold_stability_figure,
    },
}


@pytest.fixture(autouse=True)
def _label_already_shown(monkeypatch):
    monkeypatch.setattr("superglm._experimental._announced", True)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _draw(kind, engine, payload):
    return RENDERERS[kind][engine](payload)


# --------------------------------------------------------------------------- #
# Reading a figure
# --------------------------------------------------------------------------- #


def _artists(ax, gid):
    return [artist for artist in ax.get_children() if artist.get_gid() == gid]


def _one(ax, gid):
    (artist,) = _artists(ax, gid)
    return artist


def _xy(line):
    return np.asarray(line.get_xdata()).tolist(), np.asarray(line.get_ydata()).tolist()


def _traces(fig, name):
    return [trace for trace in fig.data if trace.name == name]


def _trace(fig, name):
    (trace,) = _traces(fig, name)
    return trace


def _shape(fig, name):
    (shape,) = [shape for shape in fig.layout.shapes if shape.name == name]
    return shape


def _hollow(collection):
    faces = collection.get_facecolors()
    return len(faces) == 0 or bool(np.all(faces[:, 3] == 0.0))


def _band_ends(fig, engine, name, ax=None):
    """The band's lower and upper values at its first and last abscissa."""
    if engine == "matplotlib":
        vertices = _one(ax, name).get_paths()[0].vertices
        ends = []
        for x in (vertices[:, 0].min(), vertices[:, 0].max()):
            at = np.sort(np.unique(vertices[vertices[:, 0] == x, 1]))
            ends.append((at[0], at[-1]))
        return ends
    band = _trace(fig, name)
    n = len(band.y) // 2
    upper, lower = np.asarray(band.y[:n]), np.asarray(band.y[n:])[::-1]
    return [(lower[0], upper[0]), (lower[-1], upper[-1])]


def _texts(fig, engine):
    if engine == "plotly":
        return fig.to_json()
    parts = [text.get_text() for text in fig.findobj(matplotlib.text.Text)]
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Payloads
# --------------------------------------------------------------------------- #


def _curve(kind="density", piece=("body",) * 3 + ("tail",) * 3, threshold=3.0, **fields):
    values = {
        "y": np.array([1.0, 2.0, 3.0, 3.0, 4.0, 5.0]),
        "values": np.array([0.10, 0.20, 0.30, 0.05, 0.04, 0.03]),
        "mean": 2.5,
        "family": "SplitSplice(GammaLS, GeneralizedParetoLSS)",
        "parameters": {"exceedance": 0.2},
        "amount_scale": "linear",
    }
    values.update(fields)
    return DistributionCurve(kind=kind, piece=tuple(piece), threshold=threshold, **values)


TABLE = pd.DataFrame(
    {
        "threshold": [1.0, 2.0, 3.0, 4.0, 5.0],
        "mean_excess": [2.0, 2.2, 2.5, 2.7, 3.1],
        "se": [0.1, 0.12, 0.2, 0.3, 0.5],
        "lower": [1.8, 1.96, 2.1, 2.1, 2.1],
        "upper": [2.2, 2.44, 2.9, 3.3, 4.1],
        "n_above": [400, 180, 70, 30, 9],
    }
)


def _line(converged=True, wall=None):
    return TailLine(
        threshold=2.0,
        scale=1.6,
        shape=0.2,
        n_above=180,
        converged=converged,
        wall=wall,
        x=(2.0, 5.0),
        y=(2.0, 2.75),
    )


def _mean_excess_figure(threshold=None, line=None, shaded_from=None, few_claims=50):
    return MeanExcessFigure(
        table=TABLE,
        level=0.95,
        threshold=threshold,
        line=line,
        shaded_from=shaded_from,
        few_claims=few_claims,
    )


def _stability(modified_scale=(1.0, 1.1, 0.9, 1.05), shape_label="tail shape"):
    scale = np.asarray(modified_scale, dtype=np.float64)
    table = pd.DataFrame(
        {
            "threshold": [1.0, 2.0, 3.0, 4.0],
            "n_above": [400, 200, 90, 40],
            "n_censored": [0, 0, 0, 0],
            "shape": [0.20, 0.22, 0.99, 0.25],
            "shape_lower": [0.15, 0.14, 0.80, 0.05],
            "shape_upper": [0.25, 0.30, 1.0, 0.45],
            "scale": [1.2, 1.4, 1.0, 2.0],
            "modified_scale": scale,
            "modified_scale_lower": scale - 0.2,
            "modified_scale_upper": scale + 0.3,
            # the second fit stopped before converging; the third sits at the upper wall
            "converged": [True, False, True, True],
            "wall": [None, None, "upper", None],
            "notes": ["", "stopped", "at the wall", ""],
        }
    )
    return ThresholdStabilityResult(
        table=table,
        dropped=pd.DataFrame(columns=["threshold", "n_above", "n_censored", "reason"]),
        level=0.95,
        min_claims=20,
        tail="one scale, one shape",
        shape_label=shape_label,
        shape_walls=(0.0, 1.0),
    )


# --------------------------------------------------------------------------- #
# One row's law
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("engine", ENGINES)
def test_a_splice_draws_body_and_tail_in_two_hues_joined_by_a_step_at_the_marked_threshold(
    engine,
):
    curve = _curve()
    fig = _draw("distribution", engine, curve)
    if engine == "matplotlib":
        ax = fig.axes[0]
        body, tail = _one(ax, "body"), _one(ax, "tail")
        assert body.get_xdata().tolist() == [1.0, 2.0, 3.0]
        assert tail.get_xdata().tolist() == [3.0, 4.0, 5.0]
        assert body.get_color() != tail.get_color()
        step = _one(ax, "step")
        assert _xy(step) == ([3.0, 3.0], [0.30, 0.05])
        assert _xy(_one(ax, "threshold"))[0] == [3.0, 3.0]
        assert (ax.get_xscale(), ax.get_yscale()) == ("linear", "linear")
    else:
        body, tail = _trace(fig, "body"), _trace(fig, "tail")
        assert list(body.x) == [1.0, 2.0, 3.0] and list(tail.x) == [3.0, 4.0, 5.0]
        assert body.line.color != tail.line.color
        step = _trace(fig, "step")
        assert list(step.x) == [3.0, 3.0] and list(step.y) == [0.30, 0.05]
        threshold = _shape(fig, "threshold")
        assert threshold.x0 == threshold.x1 == 3.0
        assert fig.layout.xaxis.type in (None, "linear")
        assert fig.layout.yaxis.type in (None, "linear")


@pytest.mark.parametrize("engine", ENGINES)
def test_a_single_law_is_one_curve_in_one_hue_with_no_threshold(engine):
    curve = _curve(piece=("law",) * 6, threshold=None, family="GammaLS")
    fig = _draw("distribution", engine, curve)
    if engine == "matplotlib":
        ax = fig.axes[0]
        assert len(ax.lines) == 1 and _one(ax, "law").get_xdata().tolist() == curve.y.tolist()
        assert _artists(ax, "step") == [] and _artists(ax, "threshold") == []
    else:
        assert [trace.name for trace in fig.data] == ["law"]
        assert len(fig.layout.shapes) == 0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("amount_scale", "amount_axis"), [("log", "log"), ("linear", "linear")])
def test_a_survival_curve_takes_its_amount_axis_from_the_payload(engine, amount_scale, amount_axis):
    curve = _curve(
        kind="survival",
        values=np.array([0.9, 0.5, 0.2, 0.2, 0.05, 0.01]),
        amount_scale=amount_scale,
    )
    fig = _draw("distribution", engine, curve)
    if engine == "matplotlib":
        ax = fig.axes[0]
        assert (ax.get_xscale(), ax.get_yscale()) == (amount_axis, "log")
    else:
        assert (fig.layout.xaxis.type or "linear", fig.layout.yaxis.type) == (amount_axis, "log")


@pytest.mark.parametrize("engine", ENGINES)
def test_an_infinite_mean_excess_is_left_undrawn_on_a_linear_value_axis(engine):
    values = np.array([4.0, 3.5, 3.0, 3.0, np.inf, np.inf])
    fig = _draw("distribution", engine, _curve(kind="mean_excess", values=values))
    if engine == "matplotlib":
        ax = fig.axes[0]
        drawn = np.asarray(_one(ax, "tail").get_ydata(), dtype=np.float64)
        assert ax.get_yscale() == "linear"
    else:
        drawn = np.asarray(_trace(fig, "tail").y, dtype=np.float64)
        assert fig.layout.yaxis.type in (None, "linear")
    assert drawn[0] == 3.0 and np.isnan(drawn[1:]).all()


@pytest.mark.parametrize("engine", ENGINES)
def test_a_mean_excess_infinite_everywhere_says_so(engine):
    curve = _curve(kind="mean_excess", values=np.full(6, np.inf))
    assert "infinite at every amount" in _texts(_draw("distribution", engine, curve), engine)


# --------------------------------------------------------------------------- #
# The mean excess plot
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("engine", ENGINES)
def test_the_mean_excess_draws_every_point_and_the_band_through_the_tables_ends(engine):
    fig = _draw("mean_excess", engine, _mean_excess_figure())
    if engine == "matplotlib":
        ax = fig.axes[0]
        points = _one(ax, "mean excess").get_offsets()
        assert np.asarray(points).tolist() == TABLE[["threshold", "mean_excess"]].values.tolist()
        ends = _band_ends(fig, engine, "band", ax)
        assert [_artists(ax, name) for name in ("threshold", "tail line", "few claims")] == [[]] * 3
    else:
        points = _trace(fig, "mean excess")
        assert list(points.x) == TABLE["threshold"].tolist()
        assert list(points.y) == TABLE["mean_excess"].tolist()
        ends = _band_ends(fig, engine, "95% band")
        assert len(fig.data) == 2 and len(fig.layout.shapes) == 0
    assert ends == [(1.8, 2.2), (2.1, 4.1)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("converged", "wall", "dashed"),
    [(True, None, False), (False, None, True), (True, "upper", True)],
    ids=["settled", "not_converged", "at_a_wall"],
)
def test_a_threshold_is_marked_with_its_fitted_line_and_the_few_claims_stretch_shaded(
    engine, converged, wall, dashed
):
    payload = _mean_excess_figure(
        threshold=2.0, line=_line(converged, wall), shaded_from=4.0, few_claims=50
    )
    fig = _draw("mean_excess", engine, payload)
    if engine == "matplotlib":
        ax = fig.axes[0]
        assert _xy(_one(ax, "threshold"))[0] == [2.0, 2.0]
        line = _one(ax, "tail line")
        assert _xy(line) == ([2.0, 5.0], [2.0, 2.75])
        assert (line.get_linestyle() == "--") is dashed
        shade = _one(ax, "few claims")
        assert (shade.get_x(), shade.get_x() + shade.get_width()) == (4.0, 5.0)
        assert shade.get_label() == "fewer than 50 claims above"
    else:
        assert _shape(fig, "threshold").x0 == 2.0
        (line,) = [trace for trace in fig.data if trace.name.startswith("fitted tail")]
        assert (list(line.x), list(line.y)) == ([2.0, 5.0], [2.0, 2.75])
        assert (line.line.dash == "dash") is dashed
        shade = _shape(fig, "few claims")
        assert (shade.x0, shade.x1) == (4.0, 5.0)
        assert "fewer than 50 claims above" in [note.text for note in fig.layout.annotations]


# --------------------------------------------------------------------------- #
# The threshold stability plot
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("engine", ENGINES)
def test_flagged_candidates_are_hollow_beside_the_band_the_walls_and_the_scale_panel(engine):
    result = _stability()
    fig = _draw("threshold_stability", engine, result)
    if engine == "matplotlib":
        shape_ax, scale_ax = fig.axes
        filled, hollow = _one(shape_ax, "fitted"), _one(shape_ax, "flagged")
        assert np.asarray(filled.get_offsets())[:, 0].tolist() == [1.0, 4.0]
        assert np.asarray(hollow.get_offsets())[:, 0].tolist() == [2.0, 3.0]
        assert _hollow(hollow) and not _hollow(filled)
        walls = [_one(shape_ax, f"{side} wall").get_ydata()[0] for side in ("lower", "upper")]
        shape_ends = _band_ends(fig, engine, "band", shape_ax)
        scale_ends = _band_ends(fig, engine, "band", scale_ax)
        assert _hollow(_one(scale_ax, "flagged"))
        assert shape_ax.get_ylabel() == "tail shape"
    else:
        filled, hollow = _trace(fig, "tail shape"), _traces(fig, "did not converge or at a wall")
        assert list(filled.x) == [1.0, 4.0]
        assert [list(trace.x) for trace in hollow] == [[2.0, 3.0]] * 2
        assert {trace.marker.symbol for trace in hollow} == {"circle-open"}
        assert filled.marker.symbol in (None, "circle")
        walls = [_shape(fig, f"{side} wall").y0 for side in ("lower", "upper")]
        shape_ends = _band_ends(fig, engine, "tail shape band")
        scale_ends = _band_ends(fig, engine, "threshold-free scale band")
        assert fig.layout.yaxis.title.text == "tail shape"
    assert walls == [0.0, 1.0]
    assert shape_ends == [(0.15, 0.25), (0.05, 0.45)]
    assert np.allclose(scale_ends, [(0.8, 1.3), (0.85, 1.35)], rtol=0, atol=1e-15)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_scale_column_without_numbers_has_no_panel_and_the_shape_label_is_printed(engine):
    label = "mean fitted shape over the book"
    fig = _draw(
        "threshold_stability", engine, _stability(modified_scale=[np.nan] * 4, shape_label=label)
    )
    if engine == "matplotlib":
        (ax,) = fig.axes
        assert ax.get_ylabel() == label and label in ax.get_title()
    else:
        assert _traces(fig, "threshold-free scale") == []
        assert "xaxis2" not in fig.layout.to_plotly_json()
        assert fig.layout.yaxis.title.text == label
    assert label in _texts(fig, engine)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("distribution", _curve()),
        ("mean_excess", _mean_excess_figure(2.0, _line(), 4.0)),
        ("threshold_stability", _stability()),
    ],
    ids=["distribution", "mean_excess", "threshold_stability"],
)
def test_no_figure_names_a_currency(engine, kind, payload):
    text = _texts(_draw(kind, engine, payload), engine)
    assert not any(symbol in text for symbol in "$£€¥") and "currency" not in text


# --------------------------------------------------------------------------- #
# End to end: every public plot returns a figure
# --------------------------------------------------------------------------- #


def _is_figure(drawn, engine):
    return isinstance(drawn, Figure if engine == "matplotlib" else go.Figure)


@pytest.fixture(scope="module")
def gamma_model():
    rng = np.random.default_rng(3)
    y = rng.gamma(3.0, 20.0, 600)
    X = pd.DataFrame({"x": np.zeros(len(y))})
    family = GammaLS()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperLSS(family, family.mean(), family.scale()).fit(X, y)
    return model, X


@pytest.fixture(scope="module")
def splice():
    rng = np.random.default_rng(570)
    n = 1500
    large = rng.random(n) < 0.25
    y = rng.gamma(3.0, 20.0, n)
    y = np.where(y >= 100.0, 100.0 * rng.random(n), y)
    tail = 100.0 + 35.0 * np.expm1(-0.25 * np.log(rng.random(n))) / 0.25
    y = np.where(large, tail, y)
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, n)})
    body, gpd = GammaLS(), GeneralizedParetoLSS()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SplitSplice.fit(
            X,
            y,
            threshold=100.0,
            body=SuperLSS(body, body.mean(), body.scale()),
            exceedance=SuperGLM(family="binomial", features={}),
            tail=SuperLSS(gpd, gpd.scale(), gpd.shape()),
        )
    return model, X, y


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("kind", ["density", "survival", "mean_excess"])
def test_superlss_plot_distribution_returns_a_figure(gamma_model, engine, kind):
    model, X = gamma_model
    drawn = model.plot_distribution(X.iloc[[0]], kind=kind, engine=engine, n_points=40)
    assert _is_figure(drawn, engine)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("kind", ["density", "survival", "mean_excess"])
def test_splitsplice_plot_distribution_returns_a_figure_with_its_step(splice, engine, kind):
    model, X, _ = splice
    drawn = model.plot_distribution(X.iloc[[0]], kind=kind, engine=engine, n_points=40)
    assert _is_figure(drawn, engine)
    if engine == "matplotlib":
        assert [len(_artists(drawn.axes[0], gid)) for gid in ("body", "tail", "step")] == [1] * 3
    else:
        assert [len(_traces(drawn, name)) for name in ("body", "tail", "step")] == [1] * 3


@pytest.mark.parametrize("engine", ENGINES)
def test_plot_mean_excess_and_plot_threshold_stability_return_figures(splice, engine):
    _, _, y = splice
    result = mean_excess(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drawn = plot_mean_excess(result, threshold=100.0, engine=engine)
        stability = threshold_stability(None, y, [80.0, 100.0, 140.0])
    assert _is_figure(drawn, engine)
    assert _is_figure(plot_threshold_stability(stability, engine=engine), engine)
