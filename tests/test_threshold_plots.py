"""The two threshold plots, through their payloads: ``mean_excess`` and ``threshold_stability``.

References are independent of the functions: a hand-computed fixture, NumPy on repeated
rows, SciPy's generalised Pareto likelihood maximised in the test with a finite-difference
Hessian, the same tail fitted by hand, and finite differences of a reported quantity.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats
from scipy.special import expit, logit

from superglm import (
    GammaLS,
    GeneralizedParetoLSS,
    Numeric,
    SplitSplice,
    SuperGLM,
    SuperLSS,
    cat,
    s,
    term,
)
from superglm._experimental import ExperimentalWarning
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.distributional import thresholds as module
from superglm.distributional.rebuild import clone_with_replaced_terms, unfitted_clone
from superglm.distributional.thresholds import (
    mean_excess,
    plot_mean_excess,
    plot_threshold_stability,
    threshold_stability,
)
from superglm.distributional.walls import wall_rows

U = 2.0**-53
Z95 = 1.959963984540054  # the 0.975 quantile of the standard normal
Z50 = 0.6744897501960817  # its 0.75 quantile
HAND = np.array([1.0, 2.0, 2.0, 4.0, 7.0, 11.0])


@pytest.fixture(autouse=True)
def _label_already_shown(monkeypatch):
    monkeypatch.setattr("superglm._experimental._announced", True)


def _gpd(rng, n, shape, scale=1.0):
    """Generalised Pareto excesses by inversion; a negative shape gives a bounded tail."""
    return scale * np.expm1(-shape * np.log(rng.random(n))) / shape


def _scipy_tail(excess, counts=None, censored=None):
    """SciPy's maximum-likelihood tail in (log scale, logit shape), and its covariance.

    The optimum is polished from ``genpareto.fit``; the covariance inverts a Hessian of central
    second differences. A censored excess contributes its log survival function.
    """
    counts = np.ones(len(excess)) if counts is None else np.asarray(counts, dtype=float)
    capped = np.zeros(len(excess), dtype=bool) if censored is None else np.asarray(censored)

    def negative(parameters):
        law = stats.genpareto(expit(parameters[1]), scale=math.exp(parameters[0]))
        return -np.sum(counts * np.where(capped, law.logsf(excess), law.logpdf(excess)))

    shape, _, scale = stats.genpareto.fit(excess[~capped], floc=0)
    best = optimize.minimize(
        negative,
        [math.log(scale), logit(np.clip(shape, 0.02, 0.98))],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 4000},
    ).x
    step = 1e-4 * np.eye(2)
    hessian = np.array(
        [
            [
                (
                    negative(best + step[a] + step[b])
                    - negative(best + step[a] - step[b])
                    - negative(best - step[a] + step[b])
                    + negative(best - step[a] - step[b])
                )
                / 4e-8
                for b in range(2)
            ]
            for a in range(2)
        ]
    )
    return math.exp(best[0]), float(expit(best[1])), np.linalg.inv(hessian)


def _row(result, threshold):
    table = result.table
    return table[table["threshold"] == threshold].iloc[0]


def _plain_tail(**kwargs):
    family = GeneralizedParetoLSS()
    return SuperLSS(family, family.scale(), family.shape(), **kwargs)


# --------------------------------------------------------------------------- #
# The experimental label
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "call",
    [
        lambda: mean_excess(HAND, min_above=2),
        lambda: threshold_stability(None, 1.0 + _gpd(np.random.default_rng(0), 60, 0.2), [1.0]),
        lambda: plot_mean_excess(mean_excess(HAND, min_above=2)),
        lambda: plot_threshold_stability(None),
    ],
    ids=["mean_excess", "threshold_stability", "plot_mean_excess", "plot_threshold_stability"],
)
def test_each_entry_point_warns_on_its_first_call(call, request, monkeypatch):
    monkeypatch.setattr(module, "_renderer", lambda engine, kind: lambda payload: payload)
    name = request.node.callspec.id
    if name == "plot_mean_excess":
        # build the argument first, so the plot is the first experimental call
        argument = mean_excess(HAND, min_above=2)
        call = lambda: plot_mean_excess(argument)  # noqa: E731
    monkeypatch.setattr("superglm._experimental._announced", False)
    with pytest.warns(ExperimentalWarning, match=rf"^{name} is experimental") as caught:
        call()
    assert caught[0].filename == __file__


def test_the_plots_hand_their_payload_to_the_renderer_of_the_engine(monkeypatch):
    monkeypatch.setattr(module, "_renderer", lambda engine, kind: lambda p: (engine, kind, p))
    rng = np.random.default_rng(5)
    y = 1.0 + _gpd(rng, 800, 0.2)
    result = mean_excess(y)
    engine, kind, figure = plot_mean_excess(result, 1.5, few_claims=30, engine="plotly")
    assert (engine, kind) == ("plotly", "mean_excess")
    assert figure.to_json() == result.figure(1.5, few_claims=30).to_json()
    assert figure.threshold == 1.5 and figure.few_claims == 30 and figure.line is not None
    stability = threshold_stability(None, y, [1.5])
    assert plot_threshold_stability(stability) == ("matplotlib", "threshold_stability", stability)


# --------------------------------------------------------------------------- #
# B1: the empirical mean excess
# --------------------------------------------------------------------------- #


def test_the_hand_fixture_pins_means_errors_and_both_band_ends_at_two_levels():
    # above 1: 1, 1, 3, 6, 10; above 2 (the two claims at 2 are not above it): 2, 5, 9;
    # above 4: 3, 7; above 7 only one claim is left
    mean = np.array([4.2, 16.0 / 3.0, 5.0])
    se = np.array([math.sqrt(14.7 / 5.0), math.sqrt(37.0 / 9.0), 2.0])
    for level, z in ((0.95, Z95), (0.5, Z50)):
        result = mean_excess(HAND, min_above=2, level=level)
        table = result.table
        assert list(table.columns) == [
            "threshold",
            "mean_excess",
            "se",
            "lower",
            "upper",
            "n_above",
        ]
        assert table["threshold"].tolist() == [1.0, 2.0, 4.0]
        assert table["n_above"].tolist() == [5, 3, 2]
        np.testing.assert_allclose(table["mean_excess"], mean, rtol=16 * U)
        np.testing.assert_allclose(table["se"], se, rtol=16 * U)
        np.testing.assert_allclose(table["lower"], mean - z * se, rtol=64 * U)
        np.testing.assert_allclose(table["upper"], mean + z * se, rtol=64 * U)
        assert result.n_claims == 6 and result.level == level and result.dropped.empty
    asked = mean_excess(HAND, thresholds=[4.0, 1.0, 7.0, 2.0, 12.0, 2.0], min_above=2)
    assert asked.table["threshold"].tolist() == [1.0, 2.0, 4.0]
    assert asked.dropped.to_dict("list") == {"threshold": [7.0, 12.0], "n_above": [1, 0]}
    assert json.loads(json.dumps(asked.to_json()))["dropped"]["n_above"] == [1, 0]


def test_frequency_counts_equal_repeated_rows_at_every_kind_of_candidate():
    rng = np.random.default_rng(11)
    y = np.round(rng.gamma(2.0, 3.0, 400), 1) + 0.1  # ties
    counts = rng.integers(0, 5, 400)
    y[counts == 0] = np.nan  # a row with no count is never read
    repeated = np.repeat(y, counts)
    top = np.sort(repeated)[-8]
    # tied losses, an empty stretch between two candidates, adjacent and far-apart candidates
    asked = [0.05, 0.3, 2.0, 2.0000001, 2.05, 9.0, 9.01, 9.02, top]
    weighted = mean_excess(y, counts, thresholds=asked)
    assert np.any(np.diff(weighted.table["n_above"]) == 0)
    pd.testing.assert_frame_equal(
        weighted.table, mean_excess(repeated, thresholds=asked).table, rtol=1e-12
    )
    for line in weighted.table.itertuples():
        excess = repeated[repeated > line.threshold] - line.threshold
        assert line.n_above == len(excess)
        np.testing.assert_allclose(line.mean_excess, excess.mean(), rtol=1e-12)
        np.testing.assert_allclose(line.se, excess.std(ddof=1) / math.sqrt(len(excess)), rtol=1e-12)
    default = mean_excess(y, counts, max_points=25)
    pd.testing.assert_frame_equal(default.table, mean_excess(repeated, max_points=25).table)
    assert 5 <= default.table["n_above"].iloc[-1] and len(default.table) <= 25
    assert default.n_claims == counts.sum()


def test_a_threshold_far_below_tightly_spaced_losses_costs_no_digits():
    small = np.arange(1.0, 41.0) / 8.0  # exact in float64 beside 1e9
    near = np.array([0.4375, 1.3125, 2.5625, 4.0625])  # sixteenths, exact too
    result = mean_excess(1e9 + small, thresholds=[1.0, *(1e9 + near)])
    above_one = small - small.mean()
    assert result.table["threshold"].iloc[0] == 1.0
    np.testing.assert_allclose(
        result.table["se"].iloc[0], math.sqrt(above_one @ above_one / (40 * 39)), rtol=1e-9
    )
    np.testing.assert_allclose(
        result.table["mean_excess"].iloc[0], (1e9 - 1.0) + small.mean(), rtol=4 * U
    )
    for line, u in zip(result.table.iloc[1:].itertuples(), near, strict=True):
        excess = small[small > u] - u
        np.testing.assert_allclose(line.mean_excess, excess.mean(), rtol=1e-9)
        np.testing.assert_allclose(line.se, excess.std(ddof=1) / math.sqrt(len(excess)), rtol=1e-9)


def test_the_known_line_of_a_simulated_tail_lies_inside_the_band():
    excess = stats.genpareto.rvs(0.2, scale=1.0, size=20_000, random_state=np.random.default_rng(3))
    table = mean_excess(excess, level=0.999).table
    read = table[table["n_above"] >= 200]
    line = (1.0 + 0.2 * read["threshold"]) / 0.8
    assert len(read) > 50
    assert np.all((read["lower"] < line) & (line < read["upper"]))


def test_the_fitted_line_is_the_classical_fit_and_the_shading_starts_where_claims_run_short():
    rng = np.random.default_rng(21)
    y = 2.0 + _gpd(rng, 3_000, 0.25, scale=1.5)
    result = mean_excess(y)
    u = 3.0
    figure = result.figure(threshold=u, few_claims=80)
    line, fitted = figure.line, _row(threshold_stability(None, y, [u]), u)
    assert (line.threshold, line.n_above) == (u, int(np.sum(y > u)))
    # the same fit on the same excesses, there as distinct losses with their counts
    np.testing.assert_allclose(
        [line.scale, line.shape], [fitted["scale"], fitted["shape"]], rtol=1e-8
    )
    assert (line.converged, line.wall) == (True, None)
    # the line's two ordinates from SciPy's fit of the same excesses
    scale, shape, _ = _scipy_tail(y[y > u] - u)
    last = float(result.table["threshold"].iloc[-1])
    assert line.x == (u, last) and last > u
    np.testing.assert_allclose(
        line.y, [scale / (1 - shape), (scale + shape * (last - u)) / (1 - shape)], rtol=1e-5
    )
    candidates = result.table["threshold"].to_numpy()
    short = [t for t in candidates if np.sum(y > t) < 80]
    assert figure.shaded_from == short[0] and result.figure(few_claims=1).shaded_from is None
    payload = json.loads(json.dumps(figure.to_json()))
    assert payload["line"]["y"] == list(line.y) and payload["threshold"] == u
    assert payload["table"]["n_above"] == result.table["n_above"].tolist()
    with pytest.raises(ValueError, match="tail line needs 5 or more claims.*lower threshold"):
        result.tail_line(float(np.sort(y)[-3]))


# --------------------------------------------------------------------------- #
# B2: the classical stability fit
# --------------------------------------------------------------------------- #


def test_the_default_fit_and_both_bands_match_scipy_at_nonzero_thresholds():
    rng = np.random.default_rng(140)
    y = 1.5 + _gpd(rng, 2_500, 0.08, scale=2.0)
    asked = [1.5, 2.5, 5.0]
    result = threshold_stability(None, y, asked)
    assert result.table["threshold"].tolist() == asked
    assert result.tail == "one scale, one shape" and result.shape_walls == (0.0, 1.0)
    for line in result.table.itertuples():
        u = line.threshold
        scale, shape, covariance = _scipy_tail(y[y > u] - u)
        assert (line.n_above, line.n_censored) == (int(np.sum(y > u)), 0)
        assert line.converged and line.wall is None and line.notes == ""
        np.testing.assert_allclose([line.scale, line.shape], [scale, shape], rtol=1e-5)
        # the shape's band: a Wald interval on the logit scale, mapped back
        se_eta = math.sqrt(covariance[1, 1])
        np.testing.assert_allclose(
            [line.shape_lower, line.shape_upper],
            expit(logit(shape) + Z95 * se_eta * np.array([-1.0, 1.0])),
            rtol=1e-4,
        )
        # the threshold-free scale by the delta method in (log scale, logit shape)
        gradient = np.array([scale, -u * shape * (1.0 - shape)])
        se = math.sqrt(gradient @ covariance @ gradient)
        np.testing.assert_allclose(line.modified_scale, scale - shape * u, rtol=1e-4)
        np.testing.assert_allclose(
            [line.modified_scale_lower, line.modified_scale_upper],
            scale - shape * u + Z95 * se * np.array([-1.0, 1.0]),
            rtol=0,
            atol=1e-4 * se,
        )
        # the two parameters are correlated, so the cross term carries weight here
        assert abs(2.0 * gradient[0] * gradient[1] * covariance[0, 1]) > 0.2 * se**2
    # a band on the shape's own scale would cross the lower wall on the smallest sample
    last = result.table.iloc[-1]
    natural_se = math.sqrt(_scipy_tail(y[y > 5.0] - 5.0)[2][1, 1]) * last["shape"]
    assert last["shape"] - Z95 * natural_se * (1 - last["shape"]) < 0.0 < last["shape_lower"]


def test_a_wall_and_a_stopped_fit_are_the_fits_own_flags_and_stay_in_the_table():
    rng = np.random.default_rng(8)
    bounded, heavy = 1.0 + _gpd(rng, 3_000, -0.3), 1.0 + _gpd(rng, 3_000, 1.5)
    for y, side in ((bounded, "lower"), (heavy, "upper")):
        # a tail with an end point, and one with no mean, put the shape beyond a wall
        with pytest.warns(ConvergenceWarning, match="1 of 1 candidate thresholds") as caught:
            result = threshold_stability(None, y, [1.0])
        assert len(caught) == 1
        line = result.table.iloc[0]
        assert line["wall"] == side and line["n_above"] == 3_000
        assert f"at the {side} wall" in line["notes"]
        assert json.loads(json.dumps(result.to_json()))["table"]["wall"] == [side]
    ordinary = 1.0 + _gpd(rng, 500, 0.3)
    with pytest.warns(ConvergenceWarning, match="1 of 1 candidate thresholds"):
        stopped = threshold_stability(None, ordinary, [1.0], fit_options={"max_inner_iter": 1})
    assert not stopped.table["converged"].iloc[0] and stopped.table["wall"].iloc[0] is None
    assert threshold_stability(None, ordinary, [1.0]).table["converged"].iloc[0]


def _shape_line_tail():
    family = GeneralizedParetoLSS()
    return SuperLSS(family, family.scale(), family.shape(term("x", Numeric())))


def test_a_shape_near_both_walls_is_flagged_only_when_the_fit_pressed_it_there():
    # an interior optimum whose fitted shapes come within 1e-4 of both walls: near, not pressed
    x = np.linspace(-1.0, 1.0, 5_000)
    shape = expit(10.0 * x)
    interior = np.expm1(-shape * np.log(np.random.default_rng(4).random(5_000))) / shape
    frame = pd.DataFrame({"x": x})
    result = threshold_stability(
        frame, 2.0 + interior, [2.0], tail=_shape_line_tail(), fit_options={"inner_tol": 1e-10}
    )
    assert result.table["wall"].iloc[0] is None and result.table["converged"].iloc[0]
    by_hand = _shape_line_tail().fit(frame, interior, inner_tol=1e-10)
    fitted = by_hand._require_fitted()
    near = wall_rows(fitted.family, fitted.layout, fitted.result.coefficients, fitted.result.theta)
    assert near is not None and near.near.any()  # the proximity screen alone would flag it
    assert by_hand.result_.wall is not None and by_hand.result_.wall.side is None

    # a slope that runs off towards both walls
    x = np.linspace(-1.0, 1.0, 1_000)
    with pytest.warns(ConvergenceWarning, match="candidate thresholds"):
        pressed = threshold_stability(
            pd.DataFrame({"x": x}),
            2.0 + np.where(x < 0.0, 1.0, 100.0),
            [2.0],
            tail=_shape_line_tail(),
            fit_options={"inner_tol": 1e-5},
        )
    assert pressed.table["wall"].iloc[0] == "both"


@pytest.mark.filterwarnings("ignore:the method-of-moments generalized Pareto start")
def test_the_fit_result_records_the_wall_decision_behind_its_disclosure():
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(index=np.arange(3_000))
    with pytest.warns(ConvergenceWarning, match="lower wall"):
        lower = _plain_tail().fit(frame, _gpd(rng, 3_000, -0.3))
    decision = lower.result_.wall
    assert (decision.pressed, decision.clipped, decision.n_rows) == ((3_000, 0), (0, 0), 3_000)
    assert decision.side == "lower" and decision.settled
    assert "3000 of 3000 rows have a tail shape at the lower wall" in lower.result_.warnings[0]
    with pytest.warns(ConvergenceWarning, match="upper wall"):
        upper = _plain_tail().fit(frame, _gpd(rng, 3_000, 1.5))
    assert upper.result_.wall.side == "upper" and upper.result_.wall.pressed == (0, 3_000)
    assert _plain_tail().fit(frame, _gpd(rng, 3_000, 0.3)).result_.wall is None


def test_censored_claims_are_kept_while_their_limit_reaches_the_candidate():
    rng = np.random.default_rng(77)
    n = 4_000
    loss = 1.0 + _gpd(rng, n, 0.3, scale=2.0)
    limit = rng.choice([2.0, 4.0, np.inf], size=n, p=[0.15, 0.25, 0.6])  # independent of the loss
    capped = loss > limit
    y = np.minimum(loss, limit)
    result = threshold_stability(None, y, [2.0, 4.0], censored=capped, min_claims=20)
    for u in (2.0, 4.0):
        line = _row(result, u)
        # a limit equal to the candidate is kept (it is known to exceed); one below is not
        kept = np.where(capped, y >= u, y > u)
        assert line["n_above"] == kept.sum() and line["n_censored"] == (kept & capped).sum()
        scale, shape, _ = _scipy_tail(y[kept] - u, censored=capped[kept])
        np.testing.assert_allclose([line["scale"], line["shape"]], [scale, shape], rtol=1e-5)
    assert np.sum(capped & (y == 4.0)) > 0 and np.sum(capped & (y == 2.0)) > 0
    assert _row(result, 2.0)["notes"] == ""
    left_out = int(np.sum(capped & (y < 4.0)))
    assert f"{left_out} censored claims with a limit below" in _row(result, 4.0)["notes"]
    # the same fit as on the rows without the claims that were left out
    rest = ~(capped & (y < 4.0))
    without = _row(threshold_stability(None, y[rest], [4.0], censored=capped[rest]), 4.0)
    columns = ["n_above", "n_censored", "shape", "shape_lower", "scale", "modified_scale_upper"]
    assert _row(result, 4.0)[columns].tolist() == without[columns].tolist()


def test_counts_equal_repeated_rows_from_the_default_thresholds_on():
    rng = np.random.default_rng(31)
    y = np.round(1.0 + _gpd(rng, 1_200, 0.35), 2)
    counts = rng.integers(0, 4, 1_200)
    capped = rng.random(1_200) < 0.05
    y[counts == 0] = -1.0  # never read
    weighted = threshold_stability(None, y, sample_weight=counts, censored=capped, min_claims=150)
    repeated = threshold_stability(
        None, np.repeat(y, counts), censored=np.repeat(capped, counts), min_claims=150
    )
    # the default candidates: order statistic ceil(k N / 100) of the complete losses
    complete = np.sort(np.repeat(y, counts)[~np.repeat(capped, counts)])
    expected = np.unique([complete[-(-k * len(complete) // 100) - 1] for k in range(80, 100)])
    for result in (weighted, repeated):
        candidates = [*result.table["threshold"], *result.dropped["threshold"]]
        assert candidates == expected.tolist()
        assert len(result.table) > 5 and len(result.dropped) > 5
        assert result.table["wall"].isna().all() and result.table["converged"].all()
    for column in ("n_above", "n_censored"):
        assert weighted.table[column].tolist() == repeated.table[column].tolist()
        assert weighted.dropped[column].tolist() == repeated.dropped[column].tolist()
    assert weighted.table["n_censored"].min() > 0
    for column in ("shape", "scale", "modified_scale", "shape_lower", "modified_scale_upper"):
        np.testing.assert_allclose(weighted.table[column], repeated.table[column], rtol=1e-8)


# --------------------------------------------------------------------------- #
# B3: the user's own tail
# --------------------------------------------------------------------------- #


def _book(n=3_000, seed=52):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"group": rng.choice(["a", "b", "c"], size=n), "x": rng.uniform(-1.0, 1.0, n)}
    )
    scale = np.exp(0.2 + 0.4 * (frame["group"] == "b") + 0.3 * np.sin(2.0 * frame["x"]))
    return frame, 1.0 + _gpd(rng, n, 0.25) * scale.to_numpy()


def test_the_supplied_tail_is_refitted_and_never_touched():
    frame, y = _book()
    family = GeneralizedParetoLSS()

    def build():
        return SuperLSS(family, family.scale(cat("group")), family.shape())

    unfitted = build()
    result = threshold_stability(frame, y, [1.5, 2.5], tail=unfitted)
    assert unfitted._model is None  # the object the user passed is not fitted
    assert "scale ~ 1 + group; shape ~ 1" in result.tail and result.shape_label == "tail shape"
    for u in (1.5, 2.5):
        keep = y > u
        by_hand = build().fit(frame[keep], y[keep] - u)
        line = _row(result, u)
        np.testing.assert_allclose(
            line["shape"], by_hand.predict_parameters(frame[:1])["shape"].iloc[0], rtol=1e-10
        )
        # with terms on the scale there is no one scale, and no threshold-free scale
        assert np.isnan(
            line[["scale", "modified_scale", "modified_scale_lower"]].astype(float)
        ).all()
        assert np.isfinite(line[["shape_lower", "shape_upper"]].astype(float)).all()

    fitted = build().fit(frame[y > 1.5], y[y > 1.5] - 1.5)
    before = (fitted.result_.fit_id, fitted.result_.coefficients.copy(), fitted.predict(frame[:5]))
    again = threshold_stability(frame, y, [1.5, 2.5], tail=fitted)
    assert fitted.result_.fit_id == before[0]
    np.testing.assert_array_equal(fitted.result_.coefficients, before[1])
    np.testing.assert_array_equal(fitted.predict(frame[:5]), before[2])
    np.testing.assert_allclose(again.table["shape"], result.table["shape"], rtol=1e-10)


def test_a_tail_taken_from_a_splice_fitted_with_censored_rows_is_accepted():
    frame, y = _book(2_000)
    capped = y > 9.0
    y = np.minimum(y, 9.0)
    body = GammaLS()
    tail = _plain_tail()
    SplitSplice.fit(
        frame,
        y,
        threshold=2.0,
        body=SuperLSS(body, body.mean(), body.scale()),
        exceedance=SuperGLM(family="binomial", features={}),
        tail=tail,
        censored=capped,
    )
    assert tail._model.fit_state.censored_row_count > 0
    with pytest.raises(ValueError, match="truncated or censored rows"):
        clone_with_replaced_terms(tail, {})  # the editor's replay still refuses it
    assert unfitted_clone(tail)._model is None and unfitted_clone(tail)._fit_settings is None
    result = threshold_stability(frame, y, [2.0, 3.0], tail=tail, censored=capped)
    classical = threshold_stability(None, y, [2.0, 3.0], censored=capped)
    np.testing.assert_allclose(result.table["shape"], classical.table["shape"], rtol=1e-10)
    np.testing.assert_allclose(
        result.table["modified_scale_lower"], classical.table["modified_scale_lower"], rtol=1e-10
    )


def _smooth_tail(**kwargs):
    family = GeneralizedParetoLSS()
    return SuperLSS(family, family.scale(s("x", kind="cr", k=6)), family.shape(), **kwargs)


def _shape_of(model, frame):
    return model.predict_parameters(frame[:1])["shape"].iloc[0]


def test_a_fitted_tail_is_refitted_the_way_it_was_fitted():
    frame, y = _book()
    keep = y > 2.0
    rows, excess = frame[keep], y[keep] - 2.0
    strength = {"scale:x#wiggle": 2.5}

    fixed = _smooth_tail().fit(frame[y > 1.2], y[y > 1.2] - 1.2, lambdas=strength)
    by_hand = _smooth_tail().fit(rows, excess, lambdas=strength)
    line = _row(threshold_stability(frame, y, [2.0], tail=fixed), 2.0)
    np.testing.assert_allclose(line["shape"], _shape_of(by_hand, frame), rtol=1e-10)

    searched = _smooth_tail().fit_reml(frame[y > 1.2], y[y > 1.2] - 1.2)
    by_reml = _smooth_tail().fit_reml(rows, excess)
    line = _row(threshold_stability(frame, y, [2.0], tail=searched), 2.0)
    np.testing.assert_allclose(line["shape"], _shape_of(by_reml, frame), rtol=1e-10)
    assert abs(_shape_of(by_reml, frame) - _shape_of(by_hand, frame)) > 1e-6  # the two differ

    # the caller's settings come first, and an unfitted tail follows ``reml``
    overridden = threshold_stability(
        frame, y, [2.0], tail=searched, reml=False, fit_options={"lambdas": strength}
    )
    np.testing.assert_allclose(
        overridden.table["shape"].iloc[0], _shape_of(by_hand, frame), rtol=1e-10
    )
    unfitted = threshold_stability(frame, y, [2.0], tail=_smooth_tail(), reml=True)
    np.testing.assert_allclose(
        unfitted.table["shape"].iloc[0], _shape_of(by_reml, frame), rtol=1e-10
    )
    with pytest.raises(ValueError, match="missing penalty lambda names"):
        threshold_stability(frame, y, [2.0], tail=_smooth_tail())  # the first fit raises as itself


def test_an_array_offset_is_cut_to_the_rows_of_each_candidate():
    frame, y = _book(1_500)
    rng = np.random.default_rng(9)
    offset = 0.4 * np.sin(np.arange(1_500.0))
    counts = rng.integers(0, 3, 1_500)
    y[counts == 0] = np.nan  # a row with no count is never read
    family = GeneralizedParetoLSS()

    def build(values):
        return SuperLSS(
            family, family.scale(offset=values), family.shape(), weight_semantics="frequency"
        )

    result = threshold_stability(frame, y, [1.6], tail=build(offset), sample_weight=counts)
    keep = (counts > 0) & (y > 1.6)
    assert np.any(np.diff(np.flatnonzero(keep)) > 1)  # the kept rows are not consecutive
    by_hand = build(offset[keep]).fit(frame[keep], y[keep] - 1.6, counts[keep])
    line = result.table.iloc[0]
    by_hand_shape = by_hand.predict_parameters(frame[:1], offsets={"scale": offset[:1]})["shape"]
    np.testing.assert_allclose(line["shape"], by_hand_shape.iloc[0], rtol=1e-10)
    assert line["n_above"] == counts[keep].sum()
    # an offset makes the scale differ by row, so no single scale is reported
    assert np.isnan(line["scale"]) and np.isnan(line["modified_scale"])
    assert "scale ~ 1 + offset" in result.tail


def test_a_level_with_no_claim_left_drops_that_candidate_with_the_fits_reason():
    # one shape, a scale by group: the small-scale group has no claim above 2
    rng = np.random.default_rng(122)
    y = 0.2 + np.r_[_gpd(rng, 100, 0.3, scale=0.1), _gpd(rng, 300, 0.3, scale=3.0)]
    frame = pd.DataFrame({"group": ["A"] * 100 + ["B"] * 300})
    assert y[:100].max() < 2.0 and np.sum(y > 2.0) > 100
    family = GeneralizedParetoLSS()
    tail = SuperLSS(family, family.scale(cat("group")), family.shape())
    result = threshold_stability(frame, y, [0.2, 2.0, 500.0], tail=tail)
    assert result.table["threshold"].tolist() == [0.2] and result.table["wall"].isna().all()
    dropped = result.dropped.set_index("threshold")
    assert dropped.loc[2.0, "n_above"] == np.sum(y > 2.0)
    assert "Categorical needs" in dropped.loc[2.0, "reason"]
    assert "fewer than 20 complete claims" in dropped.loc[500.0, "reason"]
    # at the lowest candidate the same error is the caller's to see
    with pytest.raises(ValueError, match="Categorical needs"):
        threshold_stability(frame, y, [2.0], tail=tail)


def test_the_band_of_a_shape_with_terms_follows_the_mean_shape_over_the_whole_book():
    frame, y = _book(2_500, seed=64)
    rng = np.random.default_rng(64)
    counts = rng.integers(1, 4, 2_500)
    lift = 0.5 * np.cos(np.arange(2_500.0))
    family = GeneralizedParetoLSS()

    def build(values):
        return SuperLSS(
            family,
            family.scale(),
            family.shape(term("x", Numeric()), cat("group"), offset=values),
            weight_semantics="frequency",
        )

    u = 1.8
    result = threshold_stability(frame, y, [u], tail=build(lift), sample_weight=counts)
    line = result.table.iloc[0]
    assert result.shape_label == "mean fitted shape over the book"
    keep = y > u
    by_hand = build(lift[keep]).fit(frame[keep], y[keep] - u, counts[keep])
    fitted = by_hand._require_fitted()

    def mean_shape(coefficients):
        # every row of the book, above the threshold or not, with its count
        eta = fitted.predict_eta_at(coefficients, frame, offsets={"shape": lift})[:, 1]
        return counts @ expit(eta) / counts.sum()

    beta = by_hand.result_.coefficients
    np.testing.assert_allclose(line["shape"], mean_shape(beta), rtol=1e-12)
    np.testing.assert_allclose(
        line["shape"],
        counts @ by_hand.predict_parameters(frame, offsets={"shape": lift})["shape"] / counts.sum(),
        rtol=1e-12,
    )
    gradient = np.array(
        [
            (mean_shape(beta + 1e-6 * e) - mean_shape(beta - 1e-6 * e)) / 2e-6
            for e in np.eye(len(beta))
        ]
    )
    assert np.count_nonzero(gradient) == 4  # the intercept, the slope and two group contrasts
    se = math.sqrt(gradient @ by_hand.covariance_ @ gradient)
    m = line["shape"]
    np.testing.assert_allclose(
        [line["shape_lower"], line["shape_upper"]],
        expit(logit(m) + Z95 * se / (m * (1.0 - m)) * np.array([-1.0, 1.0])),
        rtol=1e-6,
    )
    # the mean over the claims above the threshold alone is another number
    assert abs(counts[keep] @ expit(fitted.result.eta[:, 1]) / counts[keep].sum() - m) > 1e-4


# --------------------------------------------------------------------------- #
# Warnings from the fits
# --------------------------------------------------------------------------- #


def test_fit_warnings_go_to_the_notes_and_one_summary_obeys_the_filters(monkeypatch):
    rng = np.random.default_rng(8)
    y = 1.0 + _gpd(rng, 3_000, -0.3)
    quiet = 1.0 + _gpd(rng, 500, 0.3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = threshold_stability(None, y, [1.0, 1.05])
    assert [w.category for w in caught] == [ConvergenceWarning]
    assert "2 of 2 candidate thresholds" in str(caught[0].message)
    assert caught[0].filename == __file__
    for notes in result.table["notes"]:
        # the wall disclosure is both emitted and stored, and the start warning comes twice
        assert notes.count("at the lower wall 0") == 1
        assert notes.count("method-of-moments") == 1
    with warnings.catch_warnings():
        # an ambient error filter does not stop the fits; it raises the one summary
        warnings.simplefilter("error")
        assert threshold_stability(None, quiet, [1.0]).table["notes"].iloc[0] == ""
        with pytest.raises(ConvergenceWarning, match="threshold_stability: 1 of 1"):
            threshold_stability(None, y, [1.0])

    class Noisy(SuperLSS):
        def fit(self, *args, **kwargs):
            warnings.warn("from somewhere else", RuntimeWarning, stacklevel=1)
            return super().fit(*args, **kwargs)

    family = GeneralizedParetoLSS()
    monkeypatch.setattr(
        module,
        "_classical_tail",
        lambda: Noisy(family, family.scale(), family.shape(), weight_semantics="frequency"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = threshold_stability(None, quiet, [1.0, 1.2])
    assert [str(w.message) for w in caught] == ["from somewhere else"] * 2
    assert result.table["notes"].tolist() == ["", ""]


# --------------------------------------------------------------------------- #
# C: refusals
# --------------------------------------------------------------------------- #


def test_mean_excess_refuses_what_it_cannot_read():
    flags = [False, False, True, False, True, False]
    with pytest.raises(ValueError, match=r"mean_excess reads complete losses, and 2 rows.*row 2"):
        mean_excess(HAND, censored=flags)
    with pytest.raises(ValueError, match=r"threshold_stability\(\.\.\., censored="):
        mean_excess(HAND, censored=flags)
    for bad in (np.nan, 0.0, -3.0, np.inf):
        with pytest.raises(ValueError, match=r"Losses must be positive amounts; 1 .*row 3.*Drop"):
            mean_excess([1.0, 2.0, 3.0, bad, 5.0, 6.0, 7.0])
    with pytest.raises(ValueError, match="No candidate threshold leaves 5 or more.*min_above"):
        mean_excess(HAND[1:])
    with pytest.raises(ValueError, match="No candidate threshold leaves 2 or more"):
        mean_excess(HAND, thresholds=[10.5, 30.0], min_above=2)
    for level in (0.0, 1.0, 1.5, np.nan):
        with pytest.raises(ValueError, match="level must lie strictly between 0 and 1, not"):
            mean_excess(HAND, level=level)
    for asked in ([0.0, 2.0], [-1.0], [np.inf], [np.nan], []):
        with pytest.raises(ValueError, match="thresholds must be finite, positive amounts"):
            mean_excess(HAND, thresholds=asked, min_above=2)


def test_threshold_stability_refuses_what_it_cannot_fit():
    rng = np.random.default_rng(2)
    y = 1.0 + _gpd(rng, 300, 0.2)
    frame = pd.DataFrame({"x": np.zeros(300)})
    top = float(np.sort(y)[-15])
    with pytest.raises(ValueError) as refused:
        threshold_stability(None, y, [top, 2.0 * y.max()])
    message = str(refused.value)
    assert "No candidate threshold leaves 20 or more complete claims above it" in message
    assert f"the most is 14, above {top:g}" in message and "smaller min_claims" in message
    listed = threshold_stability(None, y, [1.0, top]).dropped
    assert listed[["threshold", "n_above", "n_censored"]].values.tolist() == [[top, 14, 0]]
    body = GammaLS()
    with pytest.raises(TypeError, match="tail must be a SuperLSS with GeneralizedParetoLSS"):
        threshold_stability(frame, y, [1.0], tail=SuperLSS(body, body.mean(), body.scale()))
    with pytest.raises(TypeError, match="X is None. Pass the frame"):
        threshold_stability(None, y, [1.0], tail=_plain_tail())
    with pytest.raises(ValueError, match="counts other than one.*weight_semantics='frequency'"):
        threshold_stability(frame, y, [1.0], tail=_plain_tail(), sample_weight=np.full(300, 2))
    with pytest.raises(ValueError, match="level must lie strictly between 0 and 1"):
        threshold_stability(None, y, [1.0], level=1.0)
    with pytest.raises(ValueError, match="thresholds must be finite, positive amounts"):
        threshold_stability(None, y, [0.0])
    with pytest.raises(ValueError, match="Losses must be positive amounts; 1 .*row 4"):
        threshold_stability(None, np.r_[y[:4], -1.0, y[5:]], [1.0])
    with pytest.raises(ValueError, match="threshold_stability supplies censored to each fit"):
        threshold_stability(None, y, [1.0], fit_options={"censored": True})
