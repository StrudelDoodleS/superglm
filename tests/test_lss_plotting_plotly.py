"""Contract tests for the plotly renderers of the LSS inference suite.

Every figure is asserted against the editor's grammar: the registered
``superglm_editor`` template, the trace inventory each payload earns, and the
three colours the grammar reserves -- the band fill, the flagged point and the
green of an interactive selection.  plotly is a development dependency, so the
whole module skips when it is absent.
"""

from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Spline, SuperLSS
from superglm.distributional import Predictor
from superglm.distributional.checks.binned import binned_check, binned_check_2d
from superglm.distributional.checks.calibration import (
    ActualExpected,
    actual_expected_check,
    calibration_payload,
)
from superglm.distributional.checks.compare import compare_models
from superglm.distributional.checks.pit import pit_payload
from superglm.distributional.checks.qq import qq_payload
from superglm.distributional.checks.worm import worm_payload
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.residuals import compute_residuals
from superglm.distributional.surfaces import (
    density_fan,
    parameter_spread,
    portfolio,
    risk_curves,
)
from superglm.distributional.terms import term_effect

go = pytest.importorskip("plotly.graph_objects")
pio = pytest.importorskip("plotly.io")
dp = importlib.import_module("superglm.plotting.distributional_plotly")

BAND_FILL = "rgba(9, 105, 218, 0.13)"
WHISKER = "rgba(9, 105, 218, 0.55)"
SELECTED = "rgba(22, 163, 74, 0.62)"
BLUE = "#0969da"
GREY = "#8c959f"
RED = "#d1242f"


# --------------------------------------------------------------------------- #
# Fixtures: one small Gaussian fit, every payload built from it
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def case():
    """A location-and-scale Gaussian fit on 400 simulated rows."""
    rng = np.random.default_rng(20260903)
    n = 400
    x = rng.uniform(-1.0, 1.0, n)
    g = rng.choice(["a", "b", "c"], n)
    scale = np.exp(-1.0 + 0.5 * np.cos(1.8 * x))
    level = np.where(g == "a", 0.3, np.where(g == "b", -0.2, 0.0))
    X = pd.DataFrame({"x": x, "g": g})
    y = 0.6 * np.sin(2.4 * x) + level + scale * rng.standard_normal(n)
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"x": Spline("cr", k=6), "g": Categorical()}),
            Predictor("scale", {"x": Spline("cr", k=5)}),
        ],
    ).fit_reml(X, y)
    fitted = model._require_fitted()
    return fitted, X, y, compute_residuals(fitted, X, y)


@pytest.fixture(scope="module")
def misfit(case):
    """A constant-scale fit of the same rows, for the comparison payload."""
    _, X, y, _ = case
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"x": Spline("cr", k=6)}),
            Predictor("scale", {}),
        ],
    ).fit_reml(X, y)
    return model._require_fitted()


@pytest.fixture(scope="module")
def qq(case):
    fitted, X, _, residuals = case
    return qq_payload(fitted, residuals, n_sim=20, seed=3, X=X)


@pytest.fixture(scope="module")
def worm(case):
    _, _, _, residuals = case
    return worm_payload(residuals, n_points=40)


@pytest.fixture(scope="module")
def worm_by_x(case):
    _, X, _, residuals = case
    return worm_payload(
        residuals, covariate=X["x"].to_numpy(), covariate_name="x", n_intervals=4, n_points=40
    )


@pytest.fixture(scope="module")
def pit(case):
    _, _, _, residuals = case
    return pit_payload(residuals, n_bins=10)


@pytest.fixture(scope="module")
def binned(case):
    _, X, _, residuals = case
    return binned_check(residuals, X["x"].to_numpy(), name="x", n_bins=8, n_boot=40, seed=5)


@pytest.fixture(scope="module")
def binned_levels(case):
    _, X, _, residuals = case
    return binned_check(residuals, X["g"].to_numpy(), name="g", n_bins=8, n_boot=40, seed=5)


@pytest.fixture(scope="module")
def binned2d(case):
    _, X, _, residuals = case
    return binned_check_2d(
        residuals, X["x"].to_numpy(), residuals.eta[:, 0], names=("x", "eta"), n_bins=(6, 5)
    )


@pytest.fixture(scope="module")
def actual_expected(case):
    fitted, X, y, _ = case
    return actual_expected_check(
        fitted, X, y, X["x"].to_numpy(), name="x", n_bins=8, n_draws=40, seed=7
    )


@pytest.fixture(scope="module")
def calibration_no_tails(case):
    fitted, X, y, residuals = case
    return calibration_payload(
        fitted,
        X,
        y,
        residuals=residuals,
        levels=(0.5, 0.9),
        thresholds=(),
        quantile_grid=(0.25, 0.5, 0.75),
        by_parameter_deciles=False,
    )


@pytest.fixture(scope="module")
def calibration_with_tails(case):
    fitted, X, y, residuals = case
    return calibration_payload(
        fitted,
        X,
        y,
        residuals=residuals,
        levels=(0.9,),
        thresholds=(float(np.quantile(y, 0.9)),),
        quantile_grid=(0.5, 0.9),
        by_parameter_deciles=False,
    )


@pytest.fixture(scope="module")
def comparison(case, misfit):
    fitted, X, y, _ = case
    return compare_models(fitted, misfit, X, y, which="log", by="g", murphy_quantile=0.9)


@pytest.fixture(scope="module")
def comparison_bare(case, misfit):
    fitted, X, y, _ = case
    return compare_models(fitted, misfit, X, y, which="log")


@pytest.fixture(scope="module")
def smooth_effect(case):
    fitted, X, _, _ = case
    return term_effect(fitted, X, "location", "x", n_points=25, n_sim=200, seed=2)


@pytest.fixture(scope="module")
def level_effect(case):
    fitted, X, _, _ = case
    return term_effect(fitted, X, "location", "g", n_sim=200, seed=2)


@pytest.fixture(scope="module")
def scale_effect(case):
    fitted, X, _, _ = case
    return term_effect(fitted, X, "scale", "x", n_points=25, n_sim=200, seed=2)


@pytest.fixture(scope="module")
def curves(case):
    fitted, X, _, _ = case
    return risk_curves(fitted, X, {"g": "a"}, "x", n_points=12, n_draws=60, seed=4)


@pytest.fixture(scope="module")
def curves_by_level(case):
    fitted, X, _, _ = case
    return risk_curves(fitted, X, {"x": 0.0}, "g", n_points=5, n_draws=60, seed=4)


@pytest.fixture(scope="module")
def fan(case):
    fitted, X, _, _ = case
    return density_fan(fitted, X, {"g": "a"}, "x", n_points=10, n_y=40)


@pytest.fixture(scope="module")
def spread(case):
    fitted, X, y, _ = case
    return parameter_spread(fitted, X, threshold=float(np.quantile(y, 0.9)), n_bins=6)


@pytest.fixture(scope="module")
def book(case):
    fitted, X, _, _ = case
    return portfolio(fitted, X, n_draws=40, by="g", seed=6, return_draws=True)


@pytest.fixture(scope="module")
def book_total_only(case):
    fitted, X, _, _ = case
    return portfolio(fitted, X, n_draws=40, seed=6, parameter_uncertainty=False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def assert_editor_template(fig) -> None:
    """Every figure carries the editor template, resolved on the layout."""
    assert isinstance(fig, go.Figure)
    template = fig.layout.template
    assert template.layout.plot_bgcolor == "#ffffff"
    assert template.layout.paper_bgcolor == "#ffffff"
    assert template.layout.font.size == 13
    assert template.layout.xaxis.gridcolor == "rgba(140, 149, 159, 0.22)"
    assert template.layout.yaxis.gridcolor == "rgba(140, 149, 159, 0.22)"


def band_traces(fig) -> list:
    return [trace for trace in fig.data if getattr(trace, "fillcolor", None) == BAND_FILL]


def as_list(value) -> list:
    return [] if value is None else list(np.asarray(value).ravel())


# --------------------------------------------------------------------------- #
# The template
# --------------------------------------------------------------------------- #


def test_importing_the_module_registers_the_editor_template() -> None:
    assert "superglm_editor" in pio.templates
    assert dp.TEMPLATE == "superglm_editor"


# --------------------------------------------------------------------------- #
# Q-Q, worm, PIT
# --------------------------------------------------------------------------- #


def test_qq_draws_an_envelope_a_reference_and_the_order_statistics(qq) -> None:
    fig = dp.plotly_qq(qq)
    assert_editor_template(fig)
    assert len(fig.data) == 3

    envelope, reference, observed = fig.data
    assert envelope.fillcolor == BAND_FILL
    assert envelope.line.width == 0
    assert reference.line.color == GREY
    assert reference.line.dash == "7px,5px"
    assert observed.mode == "markers"
    assert observed.selected.marker.color == SELECTED
    assert set(as_list(observed.marker.color)) <= {"#ffffff", RED}
    assert "%{x" in observed.hovertemplate


def test_qq_flags_the_order_statistics_outside_the_envelope(qq) -> None:
    fig = dp.plotly_qq(qq)
    observed = fig.data[2]
    outside = (qq.observed < qq.envelope_lower) | (qq.observed > qq.envelope_upper)
    colors = np.asarray(observed.marker.color)
    assert list(colors[outside]) == [RED] * int(outside.sum())
    assert set(colors[~outside]) <= {"#ffffff"}


def test_worm_draws_three_traces_per_interval(worm, worm_by_x) -> None:
    single = dp.plotly_worm(worm)
    assert_editor_template(single)
    assert len(worm.panels) == 1
    assert len(single.data) == 3
    assert single.data[0].fillcolor == BAND_FILL
    assert single.data[1].line.dash == "7px,5px"
    assert single.data[2].line.color == BLUE

    grid = dp.plotly_worm(worm_by_x)
    assert len(worm_by_x.panels) == 4
    assert len(grid.data) == 12
    titles = [note.text for note in grid.layout.annotations]
    assert [panel.label in " ".join(titles) for panel in worm_by_x.panels] == [True] * 4


def test_worm_reports_the_q_statistics_in_the_panel_titles(worm_by_x) -> None:
    fig = dp.plotly_worm(worm_by_x)
    table = worm_by_x.q_statistics
    flagged = {str(name) for name in table.loc[table["flagged"], "group"]}
    labels = [panel.label for panel in worm_by_x.panels]
    assert flagged & set(labels), "the fixture must flag at least one interval"

    seen = {str(note.text).split(" \u00b7 ")[0]: note.font.color for note in fig.layout.annotations}
    assert set(seen) == set(labels)
    assert seen == {label: (RED if label in flagged else "#24292f") for label in labels}


def test_pit_draws_the_band_the_uniform_line_and_the_bars(pit) -> None:
    fig = dp.plotly_pit(pit)
    assert_editor_template(fig)
    assert len(fig.data) == 3

    band, expected, bars = fig.data
    assert band.fillcolor == BAND_FILL
    assert expected.line.dash == "4px,4px"
    assert expected.line.color == "#d0d7de"
    assert isinstance(bars, go.Bar)
    assert set(as_list(bars.marker.color)) <= {"rgba(244, 211, 94, 0.95)", RED}
    assert bars.marker.line.color == "#d8a10f"
    assert list(bars.y) == list(pit.counts)


def test_pit_flags_the_bins_outside_the_consistency_band(pit) -> None:
    bars = dp.plotly_pit(pit).data[2]
    counts = np.asarray(pit.counts, dtype=np.float64)
    outside = (counts < pit.band_lower) | (counts > pit.band_upper)
    colors = np.asarray(bars.marker.color)
    assert list(colors[outside]) == [RED] * int(outside.sum())


# --------------------------------------------------------------------------- #
# Binned checks
# --------------------------------------------------------------------------- #


def test_binned_draws_three_statistics_and_the_bin_counts(binned) -> None:
    fig = dp.plotly_binned(binned)
    assert_editor_template(fig)
    assert len(fig.data) == 10
    assert len(band_traces(fig)) == 3

    counts = fig.data[-1]
    assert isinstance(counts, go.Bar)
    assert list(counts.y) == list(binned.n)
    assert "n" in fig.data[2].hovertemplate

    references = [fig.data[index] for index in (1, 4, 7)]
    assert [float(np.unique(trace.y)[0]) for trace in references] == [0.0, 1.0, 0.0]
    assert all(trace.line.dash == "4px,4px" for trace in references)


def test_binned_flags_the_bins_whose_band_excludes_the_reference(binned) -> None:
    fig = dp.plotly_binned(binned)
    excluded = (binned.skew_lower > 0.0) | (binned.skew_upper < 0.0)
    assert excluded.any(), "the fixture must flag at least one skewness bin"

    skew = fig.data[8]
    assert list(np.asarray(skew.marker.color)) == list(np.where(excluded, RED, "#ffffff"))
    assert skew.selected.marker.color == SELECTED


def test_binned_on_a_categorical_covariate_labels_the_levels(binned_levels) -> None:
    fig = dp.plotly_binned(binned_levels)
    assert binned_levels.levels is not None
    assert list(fig.layout.xaxis4.ticktext) == list(binned_levels.levels)
    assert list(fig.layout.xaxis4.tickvals) == list(binned_levels.centers)


def test_binned_2d_is_one_diverging_heatmap_centred_at_zero(binned2d) -> None:
    fig = dp.plotly_binned_2d(binned2d)
    assert_editor_template(fig)
    assert len(fig.data) == 1

    heatmap = fig.data[0]
    assert isinstance(heatmap, go.Heatmap)
    assert heatmap.zmid == 0.0
    assert [stop[1] for stop in heatmap.colorscale] == [RED, "#ffffff", BLUE]
    assert np.asarray(heatmap.z).shape == binned2d.mean.T.shape


# --------------------------------------------------------------------------- #
# Actual versus expected and calibration
# --------------------------------------------------------------------------- #


def test_actual_expected_draws_the_model_the_data_the_ratio_and_the_exposure(
    actual_expected,
) -> None:
    fig = dp.plotly_actual_expected(actual_expected)
    assert_editor_template(fig)
    assert len(fig.data) == 5

    expected, actual, reference, ratio, weight = fig.data
    assert expected.line.color == BLUE
    assert actual.mode == "markers"
    assert reference.line.dash == "4px,4px"
    assert ratio.error_y.color == WHISKER
    assert list(ratio.error_y.array) == list(actual_expected.ratio_se)
    assert ratio.selected.marker.color == SELECTED
    assert isinstance(weight, go.Bar)
    assert list(weight.y) == list(actual_expected.weight)


def test_actual_expected_flags_the_bins_further_than_two_standard_errors(
    actual_expected,
) -> None:
    ratio = dp.plotly_actual_expected(actual_expected).data[3]
    off = np.abs(actual_expected.ratio - 1.0) > 2.0 * actual_expected.ratio_se
    colors = np.asarray(ratio.marker.color)
    assert list(colors) == list(np.where(off, RED, "#ffffff"))
    assert list(np.asarray(ratio.marker.line.color)) == list(np.where(off, RED, BLUE))


def _level_totals(n_levels: int, *, ratio: np.ndarray | None = None) -> ActualExpected:
    """An actual-versus-expected payload over ``n_levels`` labelled bands."""
    expected = np.full(n_levels, 100.0)
    values = np.linspace(0.9, 1.1, n_levels) if ratio is None else np.asarray(ratio, dtype=float)
    return ActualExpected(
        covariate="band",
        edges=None,
        levels=tuple(f"band {index:02d}" for index in range(n_levels)),
        centers=np.arange(n_levels, dtype=float),
        n=np.full(n_levels, 50, dtype=np.int64),
        weight=np.full(n_levels, 50.0),
        actual=values * expected,
        expected=expected,
        ratio=values,
        ratio_se=np.full(n_levels, 0.02),
        variance_law="family",
        weight_semantics="prior",
    )


def test_actual_expected_angles_crowded_level_ticks_and_floors_the_ratio() -> None:
    crowded = dp.plotly_actual_expected(_level_totals(16))
    assert crowded.layout.xaxis3.tickangle == 45
    assert crowded.layout.yaxis2.minallowed == 0.0

    sparse = dp.plotly_actual_expected(_level_totals(3))
    assert sparse.layout.xaxis3.tickangle == 0

    signed = dp.plotly_actual_expected(_level_totals(3, ratio=np.array([-0.4, 1.0, 1.2])))
    assert float(signed.layout.yaxis2.minallowed) < -0.4


def test_calibration_without_thresholds_draws_two_panels(calibration_no_tails) -> None:
    fig = dp.plotly_calibration(calibration_no_tails)
    assert_editor_template(fig)
    assert len(fig.data) == 4

    nominal, realised, one_minus_p, exceedance = fig.data
    assert nominal.line.dash == "4px,4px"
    assert realised.error_y.color == WHISKER
    assert one_minus_p.line.dash == "4px,4px"
    assert list(exceedance.x) == list(calibration_no_tails.quantiles["p"])


def test_calibration_with_thresholds_adds_the_tails_and_the_reliability_curves(
    calibration_with_tails,
) -> None:
    fig = dp.plotly_calibration(calibration_with_tails)
    assert len(calibration_with_tails.thresholds) == 1
    assert len(fig.data) == 9
    assert len(band_traces(fig)) == 1
    assert sum(isinstance(trace, go.Bar) for trace in fig.data) == 1


def test_calibration_colours_each_threshold_and_names_the_tail_ticks(case) -> None:
    fitted, X, y, residuals = case
    thresholds = tuple(float(np.quantile(y, p)) for p in (0.5, 0.8, 0.95))
    payload = calibration_payload(
        fitted,
        X,
        y,
        residuals=residuals,
        levels=(0.9,),
        thresholds=thresholds,
        quantile_grid=(0.5,),
        by_parameter_deciles=False,
    )
    fig = dp.plotly_calibration(payload)

    curves = [trace for trace in fig.data if str(trace.name).startswith("reliability at ")]
    assert len(curves) == len(thresholds)
    assert [trace.line.color for trace in curves] == [
        "#dbeafe",
        dp._sequential_color(0.5),
        BLUE,
    ]

    assert list(fig.layout.xaxis3.ticktext) == [
        f"{threshold:.3g} \u00b7 all" for threshold in thresholds
    ]


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def test_comparison_draws_segments_the_murphy_curves_and_the_difference(comparison) -> None:
    fig = dp.plotly_comparison(comparison)
    assert_editor_template(fig)
    assert comparison.murphy is not None
    assert len(fig.data) == 7

    zero, segments, curve_a, curve_b, band, difference, murphy_zero = fig.data
    assert zero.line.dash == "4px,4px"
    assert segments.error_y.color == WHISKER
    assert list(segments.x) == list(comparison.by_segment.index.astype(str))
    assert curve_a.line.color == BLUE
    assert curve_b.line.dash == "7px,5px"
    assert band.fillcolor == BAND_FILL
    assert difference.line.color == BLUE
    assert murphy_zero.line.dash == "4px,4px"


def test_comparison_without_segments_or_murphy_draws_the_overall_difference(
    comparison_bare,
) -> None:
    fig = dp.plotly_comparison(comparison_bare)
    assert comparison_bare.by_segment is None and comparison_bare.murphy is None
    assert len(fig.data) == 2
    assert list(fig.data[1].x) == ["all"]
    assert float(fig.data[1].y[0]) == pytest.approx(comparison_bare.overall["mean_diff"])


# --------------------------------------------------------------------------- #
# Term effects
# --------------------------------------------------------------------------- #


def test_term_effect_of_a_smooth_fills_the_pointwise_band_and_outlines_the_simultaneous(
    smooth_effect,
) -> None:
    fig = dp.plotly_term_effect(smooth_effect)
    assert_editor_template(fig)
    assert smooth_effect.lower_simultaneous is not None
    assert len(fig.data) == 5

    band, sim_lower, sim_upper, zero, curve = fig.data
    assert band.fillcolor == BAND_FILL
    assert sim_lower.line.color == WHISKER and sim_upper.line.color == WHISKER
    assert sim_lower.line.width == 1.4
    assert zero.line.dash == "4px,4px"
    assert curve.line.color == BLUE and curve.line.width == 2.3
    assert "se" in curve.hovertemplate


def test_term_effect_without_a_simultaneous_band_drops_its_outline(case) -> None:
    fitted, X, _, _ = case
    effect = term_effect(fitted, X, "location", "x", n_points=15, simultaneous=False)
    fig = dp.plotly_term_effect(effect)
    assert effect.lower_simultaneous is None
    assert len(fig.data) == 3
    assert [trace.name for trace in fig.data][-1] == "location:x"


def test_term_effect_with_an_exposure_strip_adds_one_bar(smooth_effect, case) -> None:
    _, X, _, _ = case
    exposure = np.histogram(X["x"].to_numpy(), bins=len(smooth_effect.x))[0]
    fig = dp.plotly_term_effect(smooth_effect, exposure=exposure)
    assert len(fig.data) == 6
    bar = fig.data[-1]
    assert isinstance(bar, go.Bar)
    assert bar.marker.line.color == "#d8a10f"
    assert list(bar.y) == list(exposure.astype(float))


def test_term_effect_of_a_categorical_draws_levels_with_whiskers(level_effect) -> None:
    fig = dp.plotly_term_effect(level_effect)
    assert level_effect.levels is not None
    assert len(fig.data) == 3

    zero, markers, simultaneous = fig.data
    assert zero.line.dash == "4px,4px"
    assert markers.mode == "markers"
    assert list(markers.x) == list(level_effect.levels)
    assert markers.error_y.color == WHISKER
    assert markers.selected.marker.color == SELECTED
    assert simultaneous.line.color == WHISKER


def test_term_effect_draws_the_special_levels_as_their_own_trace(level_effect) -> None:
    marked = replace(level_effect, special=(False, False, True))
    fig = dp.plotly_term_effect(marked)

    assert len(fig.data) == 4
    special = next(trace for trace in fig.data if trace.name == "special")
    assert list(special.x) == [level_effect.levels[-1]]
    assert special.marker.color == RED
    assert special.marker.line.color == RED

    ordinary = next(trace for trace in fig.data if trace.name.endswith(":g"))
    assert list(ordinary.x) == list(level_effect.levels[:-1])
    assert len(dp.plotly_term_effect(level_effect).data) == 3


def test_term_effect_exposure_length_must_match_the_domain(smooth_effect) -> None:
    with pytest.raises(ValueError, match="one exposure"):
        dp.plotly_term_effect(smooth_effect, exposure=np.ones(3))


def test_term_grid_lays_out_one_panel_per_effect(smooth_effect, level_effect, scale_effect) -> None:
    effects = [smooth_effect, level_effect, scale_effect]
    fig = dp.plotly_term_grid(effects)
    assert_editor_template(fig)
    assert len(fig.data) == 5 + 3 + 5
    assert len(fig.layout.annotations) >= 3

    location_only = dp.plotly_term_grid(effects, parameter="location")
    assert len(location_only.data) == 5 + 3


def test_term_grid_refuses_an_empty_selection(smooth_effect) -> None:
    with pytest.raises(ValueError, match="no term"):
        dp.plotly_term_grid([smooth_effect], parameter="nonesuch")


# --------------------------------------------------------------------------- #
# Surfaces
# --------------------------------------------------------------------------- #


def test_risk_curves_draw_a_band_and_a_line_per_quantile(curves) -> None:
    fig = dp.plotly_risk_curves(curves)
    assert_editor_template(fig)
    assert len(curves.quantiles) == 3
    assert len(fig.data) == 6
    assert len(band_traces(fig)) == 3
    assert fig.data[-1].line.color == BLUE
    assert "%{y" in fig.data[-1].hovertemplate


def test_risk_curves_on_a_categorical_sweep_label_the_levels(curves_by_level) -> None:
    fig = dp.plotly_risk_curves(curves_by_level)
    assert curves_by_level.levels is not None
    assert list(fig.layout.xaxis.ticktext) == list(curves_by_level.levels)


def test_density_fan_is_one_sequential_heatmap(fan) -> None:
    fig = dp.plotly_density_fan(fan)
    assert_editor_template(fig)
    assert len(fig.data) == 5  # heatmap, iso-density contour, three quantile curves
    assert isinstance(fig.data[1], go.Contour)
    assert [trace.name for trace in fig.data[2:]] == ["q0.5", "q0.9", "q0.99"]

    heatmap = fig.data[0]
    assert isinstance(heatmap, go.Heatmap)
    assert [stop[1] for stop in heatmap.colorscale] == ["#dbeafe", BLUE]
    assert np.asarray(heatmap.z).shape == fan.density.T.shape


def test_spread_draws_a_histogram_per_parameter_and_the_identically_priced_band(spread) -> None:
    fig = dp.plotly_spread(spread)
    assert_editor_template(fig)
    assert len(spread.parameters) == 2
    assert len(fig.data) == 2 + 1 + 3
    assert sum(isinstance(trace, go.Bar) for trace in fig.data) == 3
    assert len(band_traces(fig)) == 1
    assert "ratio" in fig.data[-1].hovertemplate


def test_portfolio_draws_the_total_and_the_segments(book) -> None:
    fig = dp.plotly_portfolio(book)
    assert_editor_template(fig)
    assert book.total_draws is not None and book.by_segment is not None
    assert len(fig.data) == 3

    draws, means, spread_of_totals = fig.data
    assert isinstance(draws, go.Histogram)
    assert isinstance(means, go.Bar)
    assert list(means.x) == list(book.by_segment["segment"].astype(str))
    assert spread_of_totals.error_y.color == WHISKER
    quantile_lines = [shape for shape in fig.layout.shapes if shape.type == "line"]
    assert len(quantile_lines) == len(book.quantiles) + 1


def test_portfolio_without_draws_or_segments_plots_the_quantiles(book_total_only) -> None:
    fig = dp.plotly_portfolio(book_total_only)
    assert book_total_only.total_draws is None and book_total_only.by_segment is None
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == list(book_total_only.quantiles)
    assert list(fig.data[0].y) == [
        book_total_only.total_quantiles[q] for q in book_total_only.quantiles
    ]


# --------------------------------------------------------------------------- #
# The six-panel diagnostics figure
# --------------------------------------------------------------------------- #


def test_diagnostics_figure_draws_six_panels(qq, worm, pit, case) -> None:
    _, _, _, residuals = case
    fig = dp.plotly_diagnostics_figure(qq, worm, pit, residuals)
    assert_editor_template(fig)
    assert len(fig.data) == 15
    assert len(band_traces(fig)) == 3
    assert sum(isinstance(trace, go.Histogram2d) for trace in fig.data) == 0
    assert sum(isinstance(trace, go.Histogram) for trace in fig.data) == 1
    assert len([note for note in fig.layout.annotations if note.text]) >= 6

    axes = {trace.xaxis for trace in fig.data}
    assert axes == {"x", "x2", "x3", "x4", "x5", "x6"}


def test_diagnostics_figure_boxes_the_worm_q_statistics_in_the_corner(qq, worm, pit, case) -> None:
    _, _, _, residuals = case
    fig = dp.plotly_diagnostics_figure(qq, worm, pit, residuals)

    boxed = [note for note in fig.layout.annotations if note.bgcolor == "#ffffff"]
    assert len(boxed) == 1
    note = boxed[0]
    assert note.bordercolor == "#d0d7de"
    assert note.xref == "x2 domain" and note.yref == "y2 domain"
    assert note.xanchor == "left" and note.yanchor == "top"
    assert "mean" in note.text and "kurt" in note.text


def test_diagnostics_figure_bins_the_scatter_panels_above_max_points(qq, worm, pit, case) -> None:
    _, _, _, residuals = case
    fig = dp.plotly_diagnostics_figure(qq, worm, pit, residuals, max_points=10)
    assert len(fig.data) == 15

    dense = [trace for trace in fig.data if isinstance(trace, go.Histogram2d)]
    assert len(dense) == 2
    assert [stop[1] for stop in dense[0].colorscale] == ["#dbeafe", BLUE]
    assert not any(trace.mode == "markers" for trace in fig.data if isinstance(trace, go.Scatter))
