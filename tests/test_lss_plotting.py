"""Contract tests for the matplotlib renderers of the LSS inference suite.

Every renderer takes a payload the builders produced from a real (small,
simulated) fit and returns a figure drawn in the editor's chart grammar: the
panel frame on the figure patch, the axis token on the spines, 11 pt tick
labels and the ``.ci`` fill at alpha 0.13 wherever a band is drawn.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from superglm import Categorical, Spline, SuperLSS
from superglm.distributional import Predictor
from superglm.distributional.checks.binned import (
    BinnedCheck2D,
    binned_check,
    binned_check_2d,
)
from superglm.distributional.checks.calibration import (
    ActualExpected,
    actual_expected_check,
    calibration_payload,
)
from superglm.distributional.checks.compare import compare_models
from superglm.distributional.checks.pit import pit_payload
from superglm.distributional.checks.qq import QQPayload, qq_payload
from superglm.distributional.checks.worm import WormPayload, worm_payload
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.residuals import compute_residuals
from superglm.distributional.surfaces import (
    density_fan,
    parameter_spread,
    portfolio,
    risk_curves,
)
from superglm.distributional.terms import ParameterTermEffect, term_effect
from superglm.plotting.distributional import (
    plot_actual_expected,
    plot_binned,
    plot_binned_2d,
    plot_calibration,
    plot_comparison,
    plot_density_fan,
    plot_diagnostics_figure,
    plot_pit,
    plot_portfolio,
    plot_qq,
    plot_risk_curves,
    plot_spread,
    plot_term_effect,
    plot_term_grid,
    plot_worm,
)
from superglm.plotting.editor_style import CHART, LABEL_PT, PANEL, TOKENS, sequential_cmap

FRAME = to_rgba(PANEL["frame"])
SPINE = to_rgba("#8c959f")
BAND_ALPHA = 0.13


# --------------------------------------------------------------------------- #
# Fixtures: one small Gaussian location-scale fit and its payloads
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def case():
    """A seeded Gaussian location-scale fit with a numeric and a level covariate."""
    rng = np.random.default_rng(20260903)
    n = 1000
    x = rng.uniform(-1.0, 1.0, n)
    g = rng.choice(["a", "b", "c"], n)
    location = 0.6 * np.sin(2.4 * x) + np.where(g == "a", 0.3, np.where(g == "b", -0.2, 0.0))
    scale = np.exp(-1.0 + 0.5 * np.cos(1.8 * x))
    frame = pd.DataFrame({"x": x, "g": g})
    y = location + scale * rng.standard_normal(n)
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"x": Spline("cr", k=6), "g": Categorical()}),
            Predictor("scale", {"x": Spline("cr", k=5)}),
        ],
    ).fit_reml(frame, y)
    fitted = model._require_fitted()
    return fitted, frame, y, compute_residuals(fitted, frame, y)


@pytest.fixture(scope="module")
def flat_fit(case):
    """A comparison model with no scale predictor, for the paired score figures."""
    _, frame, y, _ = case
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"x": Spline("cr", k=6)}),
            Predictor("scale", {}),
        ],
    ).fit_reml(frame, y)
    return model._require_fitted()


# --------------------------------------------------------------------------- #
# Style assertions shared by every renderer
# --------------------------------------------------------------------------- #


def _assert_panels(fig, styled: int, *, total: int | None = None) -> None:
    """Assert the figure is the editor's panel: frame, spines, 11 pt ticks."""
    assert isinstance(fig, Figure)
    assert len(fig.axes) == (styled if total is None else total)
    assert to_rgba(fig.patch.get_edgecolor()) == FRAME
    for ax in fig.axes[:styled]:
        assert to_rgba(ax.spines["left"].get_edgecolor()) == SPINE
        assert ax.spines["left"].get_linewidth() == pytest.approx(1.0)
        assert not ax.spines["top"].get_visible()
        assert all(label.get_fontsize() == LABEL_PT for label in ax.get_yticklabels())
        assert all(label.get_fontsize() == LABEL_PT for label in ax.get_xticklabels())


def _band_alphas(ax) -> list[float]:
    """Facecolour alphas of every filled band on ``ax``."""
    alphas = []
    for collection in ax.collections:
        face = collection.get_facecolor()
        if len(face):
            alphas.append(float(face[0][3]))
    return alphas


def _has_band(ax) -> bool:
    return any(abs(alpha - BAND_ALPHA) < 1e-9 for alpha in _band_alphas(ax))


# --------------------------------------------------------------------------- #
# Distribution checks
# --------------------------------------------------------------------------- #


def test_qq_panel_draws_points_inside_its_envelope(case) -> None:
    fitted, frame, _, residuals = case
    payload = qq_payload(fitted, residuals, n_sim=20, X=frame, seed=3)
    fig = plot_qq(payload)
    ax = fig.axes[0]
    _assert_panels(fig, 1)
    assert _has_band(ax)
    assert str(payload.n_rows) in ax.get_title()
    assert str(payload.n_sim) in ax.get_title()
    # The observed curve is a point cloud below the line threshold.
    assert any(isinstance(collection, PolyCollection) for collection in ax.collections)
    assert len(ax.lines) == 1


def test_qq_draws_a_line_when_the_cloud_is_dense() -> None:
    grid = np.linspace(-3.0, 3.0, 2500)
    payload = QQPayload(
        theoretical=grid,
        observed=grid + 0.05,
        envelope_lower=grid - 0.2,
        envelope_upper=grid + 0.2,
        n_sim=10,
        n_rows=2500,
        subsampled=True,
        seed=1,
    )
    fig = plot_qq(payload)
    _assert_panels(fig, 1)
    assert len(fig.axes[0].lines) == 2
    assert "subsampled" in fig.axes[0].get_title()


def test_worm_grid_has_one_panel_per_interval(case) -> None:
    _, frame, _, residuals = case
    payload = worm_payload(
        residuals, covariate=frame["x"].to_numpy(), covariate_name="x", n_intervals=3
    )
    fig = plot_worm(payload, ncols=2)
    _assert_panels(fig, 3)
    assert _has_band(fig.axes[0])
    assert "all" in (fig.get_suptitle() or "")
    for ax, panel in zip(fig.axes, payload.panels, strict=True):
        assert panel.label in ax.get_title()
        assert ax.texts


def test_worm_flags_a_panel_in_red_and_survives_no_statistics(case) -> None:
    _, _, _, residuals = case
    payload = worm_payload(residuals)
    flagged = payload.q_statistics.copy()
    flagged["flagged"] = True
    fig = plot_worm(WormPayload(payload.panels, None, flagged))
    _assert_panels(fig, 1)
    assert to_rgba(fig.axes[0].texts[0].get_color()) == to_rgba(TOKENS["red"])

    bare = plot_worm(WormPayload(payload.panels, None, None))
    _assert_panels(bare, 1)
    assert not bare.axes[0].texts


def test_pit_histogram_carries_bars_and_a_band(case) -> None:
    _, _, _, residuals = case
    payload = pit_payload(residuals, n_bins=10)
    fig = plot_pit(payload)
    ax = fig.axes[0]
    _assert_panels(fig, 1)
    assert len(ax.patches) == payload.n_bins
    assert _has_band(ax)


# --------------------------------------------------------------------------- #
# Binned checks
# --------------------------------------------------------------------------- #


def test_binned_check_stacks_three_moment_panels(case) -> None:
    _, frame, _, residuals = case
    payload = binned_check(residuals, frame["x"].to_numpy(), name="x", n_bins=6, n_boot=20)
    fig = plot_binned(payload)
    _assert_panels(fig, 3)
    assert all(_has_band(ax) for ax in fig.axes)
    assert fig.axes[-1].get_xlabel() == "x"


def test_binned_check_labels_levels_on_the_axis(case) -> None:
    _, frame, _, residuals = case
    payload = binned_check(residuals, frame["g"].to_numpy(), name="g", n_boot=20)
    fig = plot_binned(payload)
    _assert_panels(fig, 3)
    assert [text.get_text() for text in fig.axes[-1].get_xticklabels()] == list(payload.levels)


def test_binned_2d_annotates_every_populated_cell(case) -> None:
    _, frame, _, residuals = case
    payload = binned_check_2d(
        residuals,
        frame["x"].to_numpy(),
        np.asarray(residuals.eta[:, 0]),
        names=("x", "eta"),
        n_bins=(4, 4),
    )
    fig = plot_binned_2d(payload)
    _assert_panels(fig, 1, total=2)
    assert len(fig.axes[0].texts) == int(np.count_nonzero(payload.count))
    assert all(text.get_fontsize() == LABEL_PT for text in fig.axes[0].texts)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_actual_expected_bars_exposure_beside_the_ratio(case) -> None:
    fitted, frame, y, _ = case
    payload = actual_expected_check(fitted, frame, y, frame["x"].to_numpy(), name="x", n_bins=6)
    fig = plot_actual_expected(payload)
    _assert_panels(fig, 2)
    assert len(fig.axes[1].patches) == len(payload.centers)
    overall = float(payload.actual.sum()) / float(payload.expected.sum())
    assert f"{overall:.3f}" in fig.axes[0].get_title()


def test_actual_expected_uses_levels_as_ticks(case) -> None:
    fitted, frame, y, _ = case
    payload = actual_expected_check(fitted, frame, y, frame["g"].to_numpy(), name="g")
    fig = plot_actual_expected(payload)
    _assert_panels(fig, 2)
    assert [text.get_text() for text in fig.axes[0].get_xticklabels()] == list(payload.levels)


def _level_totals(n_levels: int, *, ratio: NDArray | None = None) -> ActualExpected:
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


def test_actual_expected_rotates_crowded_level_ticks_and_floors_the_ratio() -> None:
    """Two dozen band labels overlap flat, and a ratio of totals starts at zero."""
    crowded = plot_actual_expected(_level_totals(16))
    assert [label.get_rotation() for label in crowded.axes[0].get_xticklabels()] == [45.0] * 16
    assert crowded.axes[0].get_ylim()[0] == pytest.approx(0.0)

    sparse = plot_actual_expected(_level_totals(3))
    assert [label.get_rotation() for label in sparse.axes[0].get_xticklabels()] == [0.0] * 3

    # A signed target really can price a band below parity; the floor follows
    # the data there rather than hiding it.
    signed = plot_actual_expected(_level_totals(3, ratio=np.array([-0.4, 1.0, 1.2])))
    assert signed.axes[0].get_ylim()[0] < -0.4


def test_calibration_draws_its_four_panels(case) -> None:
    fitted, frame, y, residuals = case
    payload = calibration_payload(
        fitted,
        frame,
        y,
        residuals=residuals,
        levels=(0.5, 0.9),
        thresholds=(float(np.quantile(y, 0.9)),),
        quantile_grid=(0.5, 0.9),
    )
    fig = plot_calibration(payload)
    _assert_panels(fig, 4)
    assert _has_band(fig.axes[2])
    assert _has_band(fig.axes[3])
    assert fig.axes[1].patches  # paired expected/realised bars


def test_calibration_colours_each_threshold_and_shortens_the_tail_ticks(case) -> None:
    fitted, frame, y, residuals = case
    thresholds = tuple(float(np.quantile(y, p)) for p in (0.5, 0.8, 0.95))
    payload = calibration_payload(
        fitted,
        frame,
        y,
        residuals=residuals,
        levels=(0.9,),
        thresholds=thresholds,
        quantile_grid=(0.5,),
    )
    fig = plot_calibration(payload)

    reliability = fig.axes[3]
    curves = [line.get_color() for line in reliability.lines[::2]]
    assert len(curves) == len(thresholds)
    assert len({tuple(np.round(colour, 6)) for colour in curves}) == len(thresholds)
    scale = sequential_cmap()
    assert curves[0] == pytest.approx(scale(0.0))
    assert curves[-1] == pytest.approx(scale(1.0))

    labels = [label.get_text() for label in fig.axes[1].get_xticklabels()]
    assert labels[0] == f"{thresholds[0]:.3g} \u00b7 all"
    assert labels[1] == f"{thresholds[0]:.3g} \u00b7 d1"
    assert labels[10] == f"{thresholds[0]:.3g} \u00b7 d10"
    assert labels[11] == f"{thresholds[1]:.3g} \u00b7 all"


def test_calibration_without_thresholds_says_so(case) -> None:
    fitted, frame, y, residuals = case
    payload = calibration_payload(
        fitted,
        frame,
        y,
        residuals=residuals,
        levels=(0.9,),
        quantile_grid=(0.9,),
        by_parameter_deciles=False,
    )
    fig = plot_calibration(payload)
    _assert_panels(fig, 4)
    assert not fig.axes[1].patches
    assert fig.axes[1].texts and fig.axes[3].texts


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def test_comparison_with_a_murphy_diagram(case, flat_fit) -> None:
    fitted, frame, y, _ = case
    payload = compare_models(
        fitted,
        flat_fit,
        frame,
        y,
        by="g",
        murphy_quantile=0.9,
        thresholds=np.linspace(float(y.min()), float(y.max()), 12),
    )
    fig = plot_comparison(payload)
    _assert_panels(fig, 3)
    assert len(fig.axes[0].get_xticks()) >= len(payload.by_segment)
    assert _has_band(fig.axes[2])


def test_comparison_without_segments_or_murphy(case, flat_fit) -> None:
    fitted, frame, y, _ = case
    payload = compare_models(fitted, flat_fit, frame, y)
    fig = plot_comparison(payload)
    _assert_panels(fig, 1)
    assert [text.get_text() for text in fig.axes[0].get_xticklabels()] == ["all"]


# --------------------------------------------------------------------------- #
# Term effects
# --------------------------------------------------------------------------- #


def test_smooth_term_effect_bands_and_titles(case) -> None:
    fitted, frame, _, _ = case
    effect = term_effect(fitted, frame, "location", "x", n_points=40, n_sim=200)
    fig = plot_term_effect(effect)
    ax = fig.axes[0]
    _assert_panels(fig, 1)
    assert _has_band(ax)
    assert "location" in ax.get_title() and "edf" in ax.get_title()
    assert effect.link in ax.get_ylabel()


def test_smooth_term_effect_takes_an_exposure_strip(case) -> None:
    fitted, frame, _, _ = case
    effect = term_effect(fitted, frame, "location", "x", n_points=40, n_sim=200)
    exposure = np.linspace(1.0, 5.0, len(effect.x))
    fig = plot_term_effect(effect, exposure=exposure)
    _assert_panels(fig, 1)
    assert len(fig.axes[0].patches) == len(effect.x)


def test_categorical_term_effect_puts_levels_on_the_axis(case) -> None:
    fitted, frame, _, _ = case
    effect = term_effect(fitted, frame, "location", "g", n_sim=200)
    fig = plot_term_effect(effect, exposure=np.array([3.0, 2.0, 1.0]))
    ax = fig.axes[0]
    _assert_panels(fig, 1)
    assert [text.get_text() for text in ax.get_xticklabels()] == list(effect.levels)
    assert len(ax.patches) == len(effect.levels)


def test_log_link_term_gets_a_multiplier_axis(case) -> None:
    fitted, frame, _, _ = case
    effect = term_effect(fitted, frame, "scale", "x", n_points=40, n_sim=200)
    assert effect.multiplier is not None
    fig = plot_term_effect(effect)
    _assert_panels(fig, 2)
    assert fig.axes[1].get_ylabel() == "multiplier"


def test_special_levels_are_drawn_as_flagged_markers(case) -> None:
    """A free special level is a different kind of number and looks like one."""
    fitted, frame, _, _ = case
    plain = term_effect(fitted, frame, "location", "g", n_sim=200)
    marked = replace(plain, special=(False, False, True))

    fig = plot_term_effect(marked)
    ax = fig.axes[0]
    _assert_panels(fig, 1)
    faces = [
        tuple(np.round(collection.get_facecolor()[0], 6))
        for collection in ax.collections
        if isinstance(collection, PathCollection) and len(collection.get_offsets()) > 0
    ]
    assert tuple(np.round(to_rgba(CHART["point_selected"]["face"]), 6)) in faces
    assert "special" in [text.get_text() for text in ax.get_legend().get_texts()]

    assert plot_term_effect(plain).axes[0].get_legend() is None


def test_term_grid_filters_by_parameter(case) -> None:
    fitted, frame, _, _ = case
    effects = [
        term_effect(fitted, frame, "location", "x", n_points=30, n_sim=200),
        term_effect(fitted, frame, "location", "g", n_sim=200),
        term_effect(fitted, frame, "scale", "x", n_points=30, n_sim=200),
    ]
    fig = plot_term_grid(effects, parameter="location", ncols=2)
    _assert_panels(fig, 2)
    everything = plot_term_grid(effects, ncols=3)
    _assert_panels(everything, 3, total=4)  # the scale panel adds a multiplier axis
    with pytest.raises(ValueError, match="no term"):
        plot_term_grid(effects, parameter="nonesuch")


# --------------------------------------------------------------------------- #
# Surfaces
# --------------------------------------------------------------------------- #


def test_risk_curves_one_line_per_quantile(case) -> None:
    fitted, frame, _, _ = case
    payload = risk_curves(
        fitted, frame, {"g": "a"}, "x", quantiles=(0.5, 0.99), n_points=12, n_draws=60
    )
    fig = plot_risk_curves(payload)
    ax = fig.axes[0]
    _assert_panels(fig, 1)
    assert len(ax.lines) == len(payload.quantiles)
    assert _has_band(ax)
    widths = [line.get_linewidth() for line in ax.lines]
    assert widths[0] < widths[-1]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["q0.5", "q0.99"]


def test_risk_curves_over_levels_uses_level_ticks(case) -> None:
    fitted, frame, _, _ = case
    payload = risk_curves(fitted, frame, {"x": 0.0}, "g", quantiles=(0.9,), n_draws=60)
    fig = plot_risk_curves(payload)
    _assert_panels(fig, 1)
    assert [text.get_text() for text in fig.axes[0].get_xticklabels()] == list(payload.levels)


def test_density_fan_is_a_mesh_with_a_colorbar(case) -> None:
    fitted, frame, _, _ = case
    payload = density_fan(fitted, frame, {"g": "a"}, "x", n_points=10, n_y=24)
    fig = plot_density_fan(payload)
    ax = fig.axes[0]
    assert any(type(artist).__name__.endswith("ContourSet") for artist in ax.collections) or any(
        type(child).__name__.endswith("ContourSet") for child in ax.get_children()
    )
    assert [line.get_label() for line in ax.get_lines()] == ["q0.5", "q0.9", "q0.99"]
    _assert_panels(fig, 1, total=2)
    assert fig.axes[0].collections


def test_spread_panels_cover_parameters_and_the_priced_alike(case) -> None:
    fitted, frame, y, _ = case
    payload = parameter_spread(fitted, frame, threshold=float(np.quantile(y, 0.9)), n_bins=5)
    fig = plot_spread(payload)
    _assert_panels(fig, len(payload.parameters) + 2)
    priced = fig.axes[-1]
    assert len(priced.texts) == int(np.isfinite(payload.identically_priced["ratio"]).sum())


def test_portfolio_rows_and_the_total_histogram(case) -> None:
    fitted, frame, _, _ = case
    payload = portfolio(fitted, frame, n_draws=40, by="g", seed=4, return_draws=True)
    fig = plot_portfolio(payload)
    _assert_panels(fig, 2)
    assert [text.get_text() for text in fig.axes[0].get_yticklabels()] == [
        *[str(label) for label in payload.by_segment["segment"]],
        "all",
    ]
    assert fig.axes[1].patches


def test_portfolio_without_draws_is_one_panel(case) -> None:
    fitted, frame, _, _ = case
    payload = portfolio(fitted, frame, n_draws=40, seed=4, quantiles=(0.25, 0.75))
    fig = plot_portfolio(payload)
    _assert_panels(fig, 1)
    assert [text.get_text() for text in fig.axes[0].get_yticklabels()] == ["all"]


# --------------------------------------------------------------------------- #
# Caller-supplied axes, degenerate payloads and the optional bands
# --------------------------------------------------------------------------- #


def test_renderers_take_a_caller_axes_or_figure(case) -> None:
    fitted, frame, _, residuals = case
    payload = qq_payload(fitted, residuals, n_sim=20, X=frame, seed=3)
    caller = plt.figure(figsize=(6.0, 4.0))
    assert plot_qq(payload, ax=caller.add_subplot()) is caller
    _assert_panels(caller, 1)

    grid = plt.figure(figsize=(8.0, 4.0))
    assert plot_worm(worm_payload(residuals), fig=grid) is grid
    _assert_panels(grid, 1)

    book = portfolio(fitted, frame, n_draws=30, seed=4, return_draws=True)
    single = plt.figure(figsize=(6.0, 4.0))
    plot_portfolio(book, ax=single.add_subplot())
    _assert_panels(single, 1)


def test_term_effect_edges_of_the_optional_pieces(case) -> None:
    fitted, frame, _, _ = case
    plain = term_effect(fitted, frame, "location", "x", n_points=20, simultaneous=False)
    assert plain.lower_simultaneous is None
    fig = plot_term_effect(plain)
    _assert_panels(fig, 1)
    assert len(fig.axes[0].lines) == 2  # the fitted curve and the zero line

    with pytest.raises(ValueError, match="one value per grid point"):
        plot_term_effect(plain, exposure=np.ones(3))
    empty = plot_term_effect(plain, exposure=np.zeros(len(plain.x)))
    assert not empty.axes[0].patches

    flat = ParameterTermEffect(
        parameter="scale",
        link="log",
        term="x",
        kind="spline",
        x=np.linspace(0.0, 1.0, 5),
        levels=None,
        special=None,
        effect=np.zeros(5),
        se=np.full(5, 0.1),
        lower=np.full(5, -0.2),
        upper=np.full(5, 0.2),
        lower_simultaneous=None,
        upper_simultaneous=None,
        critical_value=None,
        multiplier=np.ones(5),
        edf=1.0,
        lambdas={},
        covariance_kind="fixed",
        alpha=0.05,
    )
    _assert_panels(plot_term_effect(flat), 1)  # a flat effect gets no multiplier axis


def test_degenerate_payloads_still_draw() -> None:
    empty_cells = BinnedCheck2D(
        covariates=("a", "b"),
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0, 2.0]),
        mean=np.full((2, 2), np.nan),
        count=np.zeros((2, 2), dtype=np.int64),
    )
    fig = plot_binned_2d(empty_cells)
    _assert_panels(fig, 1, total=2)
    assert not fig.axes[0].texts

    one_bin = ActualExpected(
        covariate="x",
        edges=np.array([0.0, 1.0]),
        levels=None,
        centers=np.array([0.5]),
        n=np.array([10], dtype=np.int64),
        weight=np.array([10.0]),
        actual=np.array([12.0]),
        expected=np.array([10.0]),
        ratio=np.array([1.2]),
        ratio_se=np.array([0.05]),
        variance_law="family",
        weight_semantics="prior",
    )
    ratios = plot_actual_expected(one_bin)
    _assert_panels(ratios, 2)
    assert len(ratios.axes[1].patches) == 1


# --------------------------------------------------------------------------- #
# The six-panel diagnostics figure
# --------------------------------------------------------------------------- #


def test_diagnostics_figure_has_six_panels(case) -> None:
    fitted, frame, _, residuals = case
    qq = qq_payload(fitted, residuals, n_sim=20, X=frame, seed=3)
    worm = worm_payload(residuals)
    pit = pit_payload(residuals, n_bins=10)
    fig = plot_diagnostics_figure(qq, worm, pit, residuals)
    _assert_panels(fig, 6)
    scatter = fig.axes[4]
    assert not any(isinstance(item, PolyCollection) for item in scatter.collections)
    assert fig.axes[3].patches  # the residual density histogram


def test_diagnostics_worm_annotation_sits_in_a_halo_box(case) -> None:
    fitted, frame, _, residuals = case
    qq = qq_payload(fitted, residuals, n_sim=20, X=frame, seed=3)
    fig = plot_diagnostics_figure(
        qq, worm_payload(residuals), pit_payload(residuals, n_bins=10), residuals
    )
    worm_axes = fig.axes[1]
    annotation = worm_axes.texts[0]
    box = annotation.get_bbox_patch()

    assert box is not None
    assert to_rgba(box.get_facecolor()) == to_rgba("#ffffff")
    assert to_rgba(box.get_edgecolor()) == to_rgba("#d0d7de")
    assert annotation.get_position()[1] > 0.5
    assert annotation.get_va() == "top"


def test_diagnostics_figure_hexbins_above_max_points(case) -> None:
    fitted, frame, _, residuals = case
    qq = qq_payload(fitted, residuals, n_sim=20, X=frame, seed=3)
    fig = plot_diagnostics_figure(
        qq, worm_payload(residuals), pit_payload(residuals, n_bins=10), residuals, max_points=1
    )
    _assert_panels(fig, 6)
    assert any(isinstance(item, PolyCollection) for item in fig.axes[4].collections)
