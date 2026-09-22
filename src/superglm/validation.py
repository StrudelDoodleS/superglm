"""Actuarial validation toolkit for model comparison and calibration assessment.

Provides lift charts, double lift charts, Lorenz curves with Gini coefficients,
and loss ratio charts following CAS RPM 2016 methodology.

All functions accept raw numpy arrays and are usable with any model framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from superglm._utils import _default_weights, _ensure_array

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@dataclass(frozen=True)
class LiftChartResult:
    """Result from :func:`superglm.lift_chart`.

    Attributes
    ----------
    bins : pd.DataFrame
        One row per quantile bin with columns: ``bin``, ``exposure_share``,
        ``observed``, ``predicted``, ``obs_pred_ratio``.
    figure : matplotlib.figure.Figure or None
        The generated figure, or ``None`` if an external ``ax`` was provided.
    """

    bins: pd.DataFrame
    figure: Figure | None


@dataclass(frozen=True)
class DoubleLiftChartResult:
    """Result from :func:`superglm.double_lift_chart`.

    Attributes
    ----------
    bins : pd.DataFrame
        One row per quantile bin with columns: ``bin``, ``n_rows``,
        ``exposure_sum``, ``exposure_share``, ``target_sum``,
        ``actual_avg``, ``model_avg``, ``current_avg``,
        ``actual_index``, ``model_index``, ``current_index``,
        ``sort_score_min``, ``sort_score_median``, ``sort_score_max``.
    figure : matplotlib.figure.Figure or None
        The generated figure, or ``None`` if an external ``ax`` was provided.
    """

    bins: pd.DataFrame
    figure: Figure | None


@dataclass(frozen=True)
class LorenzCurveResult:
    """Result from :func:`superglm.lorenz_curve`.

    Attributes
    ----------
    curve : pd.DataFrame
        Lorenz curve data with columns: ``cum_exposure_share``,
        ``cum_loss_share_ordered``, ``cum_loss_share_model``,
        ``cum_loss_share_perfect``.
    gini_model : float
        Gini coefficient for the model ordering.
    gini_perfect : float
        Gini coefficient for perfect-foresight ordering.
    gini_ratio : float
        Normalised Gini: ``gini_model / gini_perfect``.
    figure : object or None
        The generated matplotlib or Plotly figure, or ``None`` if an external
        matplotlib ``ax`` was provided.
    """

    curve: pd.DataFrame
    gini_model: float
    gini_perfect: float
    gini_ratio: float
    figure: Any | None


@dataclass(frozen=True)
class LossRatioChartResult:
    """Result from :func:`superglm.loss_ratio_chart`.

    Attributes
    ----------
    bins : pd.DataFrame
        One row per quantile bin with columns: ``bin``, ``exposure_share``,
        ``observed``, ``predicted``.
    figure : matplotlib.figure.Figure or None
        The generated figure, or ``None`` if an external ``ax`` was provided.
    """

    bins: pd.DataFrame
    figure: Figure | None


# ── Private helpers ──────────────────────────────────────────────


def _scaled_ratio(numerator, denominator, name: str) -> float:
    """Quotient of two (significand, exponent) pairs, refused if it overflows."""
    if denominator[0] == 0:
        return float("nan")
    with np.errstate(over="ignore", under="ignore"):
        result = float(np.ldexp(numerator[0] / denominator[0], numerator[1] - denominator[1]))
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _two_product(left: float, right: float) -> tuple[float, float]:
    """Error-free product within these reductions' bounded exponent ranges."""
    product = left * right
    splitter = float(2**27 + 1)
    a, b = splitter * left, splitter * right
    ah, bh = a - (a - left), b - (b - right)
    al, bl = left - ah, right - bh
    return product, al * bl - (((product - ah * bh) - al * bh) - ah * bl)


def _two_sum(left, right):
    total = left + right
    right_part = total - left
    return total, (left - (total - right_part)) + (right - right_part)


def _span_limit(factors: int) -> int:
    """Widest column span, in binades, for exact products of this many factors.

    A column rescaled so that its largest magnitude lies in [1, 2) keeps every
    nonzero entry above 2**-limit, hence a multiple of 2**-(limit + 52).
    Dekker's product and TwoSum stay error-free while every exact intermediate
    is normal, which holds when factors * (limit + 52) <= 1022: two factors for
    weighted totals and means, three with exposure, and five for a Gini
    contraction (weight, exposure and target times a weight*exposure prefix).
    """
    return 1022 // factors - 52


def _live_rows(mass, columns):
    """Zero every column on rows without mass: they contribute nothing, so
    they must not widen a column's range check."""
    live = np.logical_and.reduce([column != 0 for column in mass])
    return tuple(np.where(live, column, 0.0) for column in columns)


def _ordinary_scaling(columns, factors):
    """Exactly rescaled operand columns and their power-of-two shifts.

    Only the dynamic range within each column decides, never its units. A
    wider column is refused: flushing its small entries could drop a term
    that another column's large factor makes significant.
    """
    scaled, shifts = [], []
    for column in columns:  # the two or three operands of one reduction
        mantissa, exponent = np.frexp(column)
        top = math.frexp(max(column.max(), -column.min()))[1]
        lowest = np.min(exponent, where=mantissa != 0, initial=top)
        if lowest < top - _span_limit(factors):
            raise ValueError(
                f"validation inputs must span at most 2**{_span_limit(factors)} within a column"
            )
        scaled.append(column if top == 1 else np.ldexp(column, 1 - top))
        shifts.append(top - 1)
    return scaled, shifts


def _require_resolved(value, magnitude, count: int, name: str) -> None:
    """Refuse a compensated result that its rounding error could swamp.

    A double-length prefix is within gamma_(n+8)**2 times the sum of |terms|
    (Ogita, Rump and Oishi 2005, Proposition 4.5, widened for the plain
    correction sums), and the contraction's products are exact, so four such
    errors bound the result. A 2**24 margin keeps ratios of it within 1e-7.
    """
    unit = (count + 8) * np.finfo(float).eps / 2
    bound = 4 * 2.0**24 * (unit / (1 - unit)) ** 2 * magnitude[0], magnitude[1]
    if value[0] <= 0 or _scaled_ratio(value, bound, name) <= 1:
        raise ValueError(f"{name} cancel below binary64 resolution")


def _ordinary_product_terms(left, right, exposure=None):
    """Exact two/three-factor terms, omitting only units and zero channels."""
    terms = (left,)
    for factor in (right,) if exposure is None else (exposure, right):
        expanded = []
        unit = np.all(factor == 1)
        for part in terms:
            if unit:
                expanded.append(part)
            elif np.all(part == 1):
                expanded.append(factor)
            else:
                high, low = _two_product(part, factor)
                expanded.append(high)
                if np.any(low):
                    expanded.append(low)
        terms = tuple(expanded)
    return terms


def _ordinary_prefix_parts(terms):
    """Compensated prefix sums: the running sum and its accumulated errors.

    Each add.accumulate prefix plus the exact prefix of its TwoSum errors
    equals the exact prefix of the first term. The remaining terms join the
    correction, so high + correction is a double-length prefix (Ogita, Rump
    and Oishi, SIAM J. Sci. Comput. 26(6), 2005, Algorithm 4.4).
    """
    high = np.add.accumulate(terms[0])
    _, low = _two_sum(np.r_[0.0, high[:-1]], terms[0])
    for term in terms[1:]:  # the product residuals, at most three
        low = low + term
    return high, np.add.accumulate(low)


def _ordinary_product_sums(factors, ends):
    """Compensated prefix sums of rescaled row products at ends.

    Dekker's TwoProduct is exact on the rescaled columns: their products,
    splitter products and nonzero residuals are all normal binary64 values.
    Feed both scalar terms into the accumulator before cancellation. An
    optional exposure contributes both product terms before the final loss
    product; callers must not round left*exposure first.
    """
    if len(ends) == 1 and ends[0] == len(factors[0]):
        nonzero = np.logical_and.reduce([factor != 0 for factor in factors])
        terms = _ordinary_product_terms(*(factor[nonzero] for factor in factors))
        return np.array([math.fsum(chain.from_iterable(term.tolist() for term in terms))])
    high, low = _ordinary_prefix_parts(_ordinary_product_terms(*factors))
    return high[ends - 1] + low[ends - 1]


def _scaled_product_sums(
    left: NDArray, right: NDArray, ends: NDArray, *, exposure: NDArray | None = None
):
    """Weighted prefix sums at ends as (significand, exponent) snapshots.

    Totals are correctly rounded by math.fsum; interior prefixes carry a
    compensated error of about one ulp. Snapshots do not alter state.
    """
    factors = (left, right) if exposure is None else (left, right, exposure)
    scaled, shifts = _ordinary_scaling(_live_rows(factors, factors), len(factors))
    mantissa, exponent = np.frexp(_ordinary_product_sums(scaled, ends))
    return mantissa, exponent + sum(shifts)


def _scaled_product_total(
    left: NDArray, right: NDArray, *, exposure: NDArray | None = None
) -> tuple[float, int]:
    sums, powers = _scaled_product_sums(left, right, np.array([len(left)]), exposure=exposure)
    return float(sums[0]), int(powers[0])


def _validated_vector(name: str, value, n_rows: int | None = None) -> NDArray:
    """Return one finite numeric public-chart vector."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if n_rows is None and raw.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    if getattr(raw.dtype, "kind", None) in {"M", "m"}:
        raise ValueError(f"{name} must contain only real numeric values")
    if n_rows is not None and len(raw) != n_rows:
        raise ValueError(f"{name} must have length {n_rows}, got {len(raw)}")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain only real numeric values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validated_n_bins(n_bins) -> int:
    """Validate the shared quantile-bin boundary."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, int | np.integer) or n_bins <= 0:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    return int(n_bins)


def _validated_chart_inputs(
    y_obs,
    *,
    sample_weight=None,
    optional_vectors: tuple[str, ...] = (),
    **vectors,
) -> tuple[NDArray, dict[str, NDArray], NDArray]:
    """Validate and remove zero-effective-weight rows for every public chart."""
    observed = _validated_vector("y_obs", y_obs)
    n_rows = len(observed)
    normalized = {
        name: _validated_vector(name, value, n_rows)
        for name, value in vectors.items()
        if value is not None or name not in optional_vectors
    }
    weights = (
        np.ones(n_rows, dtype=np.float64)
        if sample_weight is None
        else _validated_vector("sample_weight", sample_weight, n_rows)
    )
    if np.any(weights < 0.0):
        raise ValueError("sample_weight must be nonnegative")
    if not np.any(weights > 0.0):
        raise ValueError("sample_weight must not be all zero")

    exposure = normalized.get("exposure")
    if exposure is not None and np.any(exposure < 0.0):
        raise ValueError("exposure must be nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        effective_weight = weights if exposure is None else weights * exposure
    if not np.all(np.isfinite(effective_weight)):
        raise ValueError("sample_weight * exposure must contain only finite values")
    positive = effective_weight > 0.0
    if not np.any(positive):
        raise ValueError("sample_weight * exposure must not be all zero")
    try:
        total_effective_weight = math.fsum(effective_weight[positive].tolist())
    except OverflowError:
        total_effective_weight = float("inf")
    if not np.isfinite(total_effective_weight):
        source = "sample_weight * exposure" if exposure is not None else "sample_weight"
        raise ValueError(f"{source} must have a finite total")
    if not np.all(positive):
        observed = observed[positive]
        normalized = {name: values[positive] for name, values in normalized.items()}
        weights = weights[positive]

    return observed, normalized, weights


def _weighted_mean(values: NDArray, weights: NDArray, name: str) -> float:
    """Scaled numerator over fsum(weights), within the weighted values' hull.

    Both sums are correctly rounded, so the quotient is within 3u relative.
    """
    total_weight = math.fsum(np.asarray(weights, dtype=float).tolist())
    if total_weight <= 0 or not np.isfinite(total_weight):
        raise ValueError(f"{name} weights must have a finite positive total")
    numerator = _scaled_product_total(values, weights)
    if numerator[0] == 0.0:
        # Exact zero has no comparison exponent and is already in the hull.
        return 0.0
    denominator, power = math.frexp(total_weight)
    mean, shift = math.frexp(numerator[0] / denominator)
    exponent = numerator[1] - power + shift
    # Only rows with weight enter the mean, so only they bound it.
    weighted = values[weights != 0]
    minimum, maximum = float(np.min(weighted)), float(np.max(weighted))
    # Compare the hull in the mean's units before reconstructing it, so a
    # mean near the float64 limits neither overflows nor underflows here.
    with np.errstate(over="ignore", under="ignore"):
        lower, upper = np.ldexp([minimum, maximum], -exponent)
    if mean <= lower:
        return minimum
    if mean >= upper:
        return maximum
    return _scaled_ratio((mean, exponent), (1.0, 0), f"{name} weighted mean")


def _weighted_total(values: NDArray, weights: NDArray, name: str) -> float:
    """Refuse an overflowing result after range-safe products and cancellation."""
    return _scaled_ratio(_scaled_product_total(values, weights), (1.0, 0), f"{name} weighted total")


def _finite_ratio(numerator: float, denominator: float, name: str) -> float:
    """Divide two finite scalars, retaining the established zero-denominator NaN."""
    if denominator == 0.0:
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = float(np.float64(numerator) / np.float64(denominator))
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _quantile_bins(sort_values: NDArray, weights: NDArray, n_bins: int) -> NDArray:
    """Assign observations to equal-weight quantile bins.

    Returns an integer array of bin indices (0-based).
    """
    n = len(sort_values)
    order = np.argsort(sort_values, kind="stable")
    cum_w = np.cumsum(weights[order])
    total_w = cum_w[-1]
    if total_w <= 0:
        return np.zeros(n, dtype=int)
    bin_edges = np.linspace(0, total_w, n_bins + 1)
    # Assign each observation to the appropriate bin
    bins = np.searchsorted(bin_edges[1:], cum_w, side="left")
    bins = np.clip(bins, 0, n_bins - 1)
    # Map back to original order
    result = np.empty(n, dtype=int)
    result[order] = bins
    return result


def _lorenz_cumulative_by_score(
    scores: NDArray,
    weights: NDArray,
    exposure: NDArray,
    losses: NDArray,
    *,
    total_exp: tuple[float, int],
    total_loss: tuple[float, int],
) -> tuple[NDArray, NDArray, tuple[NDArray, NDArray]]:
    """Final Lorenz shares and reusable order/tie starts for the same scores."""
    order = np.argsort(scores, kind="stable")
    _, starts = np.unique(scores[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    cumulative_exposure = _scaled_product_sums(weights[order], exposure[order], ends)
    cumulative_loss = _scaled_product_sums(
        weights[order], losses[order], ends, exposure=exposure[order]
    )
    shares = []
    for (mantissa, exponent), total in (
        (cumulative_exposure, total_exp),
        (cumulative_loss, total_loss),
    ):
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            result = np.ldexp(mantissa / total[0], exponent - total[1])
        if total[0] != 0 and not np.all(np.isfinite(result)):
            raise ValueError("Lorenz cumulative shares must be finite")
        shares.append(result if total[0] != 0 else np.full_like(result, np.nan))
    exposure_shares, loss_shares = shares
    return exposure_shares, loss_shares, (order, starts)


def _ordinary_pair_concordance(order, starts, weights, target, exposure):
    """Tie-block contraction on compensated prefixes, summed by math.fsum."""
    w, t = weights[order], target[order]
    weight_terms = (w,) if exposure is None else _ordinary_product_terms(w, exposure[order])
    high, low = (np.r_[0.0, part] for part in _ordinary_prefix_parts(weight_terms))
    ends = np.append(starts[1:], len(order))
    # For every row in a score block, W_before - W_after equals
    # W_before + W_through - W_total. Excluding the whole block removes ties.
    coefficient, first = _two_sum(high[starts], high[ends])
    coefficient, second = _two_sum(coefficient, -high[-1])
    correction = first + second + low[starts] + low[ends] - low[-1]
    counts = ends - starts
    coefficient, correction = (np.repeat(part, counts) for part in (coefficient, correction))
    # A zero target contributes no contraction, but its weight must remain
    # in the prefixes and tie blocks above.
    nonzero = t != 0
    coefficient, correction, t = (part[nonzero] for part in (coefficient, correction, t))
    weight_terms = tuple(part[nonzero] for part in weight_terms)
    mass = tuple(value for part in weight_terms for value in _two_product(part, t) if np.any(value))
    products = tuple(
        value
        for part in mass
        for factor in (coefficient, correction)
        for value in _two_product(part, factor)
        if np.any(value)
    )
    return math.frexp(math.fsum(chain.from_iterable(part.tolist() for part in products)))


def _weighted_pair_concordance(
    scores, weights, target, *, exposure=None, score_order=None
) -> tuple[float, int]:
    """Contract compensated prefix/block parts; rounded prefix snapshots are unsafe.

    For each strict-score block, add W_previous*T_block - T_previous*W_block.
    Prefixes keep their compensation until after contraction, including
    product residuals. Tied rows enter the prefix only after the block's
    contribution, so no within-tie pair is counted.
    """
    if score_order is None:
        order = np.argsort(scores, kind="stable")
        _, starts = np.unique(np.asarray(scores)[order], return_index=True)
    else:
        order, starts = score_order
    factors = (weights, target) if exposure is None else (weights, target, exposure)
    mass = factors[:1] + factors[2:]
    # Targets stay raw; the caller's resolution check bounds the error from
    # their raw magnitudes.
    scaled, shifts = _ordinary_scaling(_live_rows(mass, factors), 2 * len(mass) + 1)
    value, power = _ordinary_pair_concordance(
        order, starts, scaled[0], scaled[1], None if exposure is None else scaled[2]
    )
    # Quadratic in the weight*exposure mass, linear in the target.
    return value, power + 2 * (shifts[0] + sum(shifts[2:])) + shifts[1]


def _gini_coefficients(
    y_obs, y_pred, sample_weight=None, *, exposure=None, totals=None, score_orders=None
) -> tuple[float, float, float]:
    """Pair concordance; Lorenz may supply its same-operand totals and orders."""
    y_obs = _ensure_array(y_obs)
    y_pred = _ensure_array(y_pred)
    weights = _default_weights(sample_weight, len(y_obs))
    if y_obs.size == 0 or not np.any(weights > 0):
        return 0.0, 0.0, 0.0
    if totals is None:
        mass = np.ones_like(weights) if exposure is None else exposure
        total_weight = _scaled_product_total(weights, mass)
        total_loss = _scaled_product_total(weights, y_obs, exposure=exposure)
    else:
        total_weight, total_loss = totals
    if total_loss[0] <= 0:
        return 0.0, 0.0, 0.0
    # Constant targets give an exact zero without a contraction.
    live = weights != 0 if exposure is None else (weights != 0) & (exposure != 0)
    lowest = np.min(y_obs[live])
    if lowest == np.max(y_obs[live]):
        return 0.0, 0.0, 0.0
    perfect_order, model_order = (None, None) if score_orders is None else score_orders
    perfect = _weighted_pair_concordance(
        y_obs, weights, y_obs, exposure=exposure, score_order=perfect_order
    )
    # Both contractions share the error bound W * sum|w*y| * 4 gamma**2, so a
    # perfect ordering that clears it by the margin also fixes the model's
    # ratio to about 1e-7.
    magnitude = (
        _scaled_product_total(weights, np.abs(y_obs), exposure=exposure)
        if lowest < 0
        else total_loss
    )
    pair_magnitude = total_weight[0] * magnitude[0], total_weight[1] + magnitude[1]
    _require_resolved(perfect, pair_magnitude, len(y_obs), "Gini pair sums")
    model = _weighted_pair_concordance(
        y_pred, weights, y_obs, exposure=exposure, score_order=model_order
    )
    denominator = total_weight[0] * total_loss[0], total_weight[1] + total_loss[1]
    model_gini = _scaled_ratio(model, denominator, "Gini coefficients")
    perfect_gini = _scaled_ratio(perfect, denominator, "Gini coefficients")
    ratio = _scaled_ratio(model, perfect, "Gini ratio")
    return model_gini, perfect_gini, float(np.clip(ratio, -1.0, 1.0))


def _normalized_gini(y_obs, y_pred, sample_weight=None) -> float:
    """Return a stable, tie-collapsed Gini ratio without creating a plot."""
    # Unlike lorenz_curve, this scorer entry has no validated boundary, and a
    # NaN target or weight would otherwise collapse to a plausible zero score.
    operands = (y_obs,) if sample_weight is None else (y_obs, sample_weight)
    if not all(np.all(np.isfinite(_ensure_array(value))) for value in operands):
        raise ValueError("Gini coefficients must be finite")
    return _gini_coefficients(y_obs, y_pred, sample_weight)[2]


def _make_ax(ax: Axes | None):
    """Return (ax, fig_or_None). If ax is None, create a new figure."""
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax, None
    fig, ax_new = plt.subplots()
    return ax_new, fig


# ── Public functions ─────────────────────────────────────────────


def lift_chart(
    y_obs,
    y_pred,
    sample_weight=None,
    exposure=None,
    *,
    n_bins: int = 10,
    ax: Axes | None = None,
) -> LiftChartResult:
    """Lift chart: observed vs predicted across equal-exposure quantile bins.

    Parameters
    ----------
    y_obs : array-like
        Observed response values.
    y_pred : array-like
        Predicted response values.
    sample_weight : array-like or None
        Observation weights for aggregation.
    exposure : array-like or None
        Exposure measure for rate models. When provided, bins are
        equal-exposure quantiles and averages are exposure-weighted.
    n_bins : int
        Number of quantile bins.
    ax : matplotlib Axes or None
        If provided, plot onto this axes (``figure`` in result will be None).

    Returns
    -------
    LiftChartResult
        Contains a ``bins`` DataFrame and an optional ``figure``.
    """
    y_obs, vectors, w = _validated_chart_inputs(
        y_obs,
        y_pred=y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        optional_vectors=("exposure",),
    )
    y_pred = vectors["y_pred"]
    n_bins = _validated_n_bins(n_bins)
    n = len(y_obs)
    exp = vectors.get("exposure", np.ones(n, dtype=float))

    # Bin by predicted value, using exposure as bin weights
    bin_weights = w * exp
    bins_idx = _quantile_bins(y_pred, bin_weights, n_bins)

    rows = []
    total_exp = (w * exp).sum()
    for b in range(n_bins):
        mask = bins_idx == b
        if not mask.any():
            continue
        wb = w[mask]
        eb = exp[mask]
        we = wb * eb
        we_sum = we.sum()
        obs_mean = _weighted_mean(y_obs[mask], we, "y_obs")
        pred_mean = _weighted_mean(y_pred[mask], we, "y_pred")
        exp_share = we_sum / total_exp if total_exp > 0 else 0.0
        ratio = _finite_ratio(obs_mean, pred_mean, "observed / predicted ratio")
        rows.append(
            {
                "bin": b + 1,
                "exposure_share": exp_share,
                "observed": obs_mean,
                "predicted": pred_mean,
                "obs_pred_ratio": ratio,
            }
        )

    df = pd.DataFrame(rows)

    ax_plot, fig = _make_ax(ax)
    x = np.arange(len(df))
    width = 0.35
    ax_plot.bar(x - width / 2, df["observed"], width, label="Observed", color="C0")
    ax_plot.bar(x + width / 2, df["predicted"], width, label="Predicted", color="C1")
    ax2 = ax_plot.twinx()
    ax2.plot(x, df["obs_pred_ratio"], "ko-", markersize=4, label="A/E ratio")
    ax2.axhline(1.0, color="grey", linewidth=0.7, linestyle="--")
    ax2.set_ylabel("A/E ratio")
    ax_plot.set_xticks(x)
    ax_plot.set_xticklabels(df["bin"].astype(int))
    ax_plot.set_xlabel("Bin")
    ax_plot.set_ylabel("Mean value")
    ax_plot.set_title("Lift Chart")
    ax_plot.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    return LiftChartResult(bins=df, figure=fig)


def double_lift_chart(
    y_obs,
    y_pred_model,
    y_pred_current,
    sample_weight=None,
    exposure=None,
    *,
    n_bins: int = 10,
    labels: tuple[str, str, str] = ("Actual", "Model", "Current"),
    ax: Axes | None = None,
) -> DoubleLiftChartResult:
    """CAS-style double lift chart (CAS RPM 2016 methodology).

    Sorts by the ratio ``y_pred_model / y_pred_current``, bins into
    equal-exposure quantiles, and plots three indexed series: Actual,
    Model, and Current — each indexed to its own overall average.

    This is the standard actuarial double lift chart for comparing a
    new model against a current/baseline model on holdout data.

    Parameters
    ----------
    y_obs : array-like
        Observed response values (frequency, severity, or loss ratio).
    y_pred_model : array-like
        New model predictions (holdout).
    y_pred_current : array-like
        Current/baseline/manual predictions (holdout).
    sample_weight : array-like or None
        Observation weights, read as replication mass by this comparison.
    exposure : array-like or None
        Exposure measure for rate models.
    n_bins : int
        Number of equal-exposure quantile bins.
    labels : tuple of (str, str, str)
        Display labels as ``(Actual, Model, Current)``. Each element
        names the corresponding series in the plot legend and axis labels.
    ax : matplotlib Axes or None
        If provided, plot onto this axes (``figure`` in result will be None).

    Returns
    -------
    DoubleLiftChartResult
        Contains a ``bins`` DataFrame and an optional ``figure``.

    References
    ----------
    CAS RPM 2016, "Predictive Modeling — Lift and Double Lift Charts",
    https://www.casact.org/sites/default/files/presentation/rpm_2016_presentations_pm-lm-4.pdf
    """
    y_obs, vectors, w = _validated_chart_inputs(
        y_obs,
        y_pred_model=y_pred_model,
        y_pred_current=y_pred_current,
        sample_weight=sample_weight,
        exposure=exposure,
        optional_vectors=("exposure",),
    )
    y_pred_model = vectors["y_pred_model"]
    y_pred_current = vectors["y_pred_current"]
    n_bins = _validated_n_bins(n_bins)
    n = len(y_obs)
    exp = vectors.get("exposure", np.ones(n, dtype=float))

    # Sort score: model / current (with epsilon guard)
    eps = 1e-10
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sort_score = y_pred_model / np.maximum(y_pred_current, eps)
    if not np.all(np.isfinite(sort_score)):
        raise ValueError("y_pred_model / y_pred_current must contain only finite values")

    # Equal-exposure bins based on sort score
    bin_weights = w * exp
    bins_idx = _quantile_bins(sort_score, bin_weights, n_bins)

    # Overall exposure-weighted averages (for indexing)
    total_we = bin_weights.sum()
    overall_actual = _weighted_mean(y_obs, bin_weights, "y_obs")
    overall_model = _weighted_mean(y_pred_model, bin_weights, "y_pred_model")
    overall_current = _weighted_mean(y_pred_current, bin_weights, "y_pred_current")

    rows = []
    for b in range(n_bins):
        mask = bins_idx == b
        if not mask.any():
            continue
        we = bin_weights[mask]
        we_sum = we.sum()
        if we_sum <= 0:
            continue

        actual_avg = _weighted_mean(y_obs[mask], we, "y_obs")
        model_avg = _weighted_mean(y_pred_model[mask], we, "y_pred_model")
        current_avg = _weighted_mean(y_pred_current[mask], we, "y_pred_current")

        rows.append(
            {
                "bin": b + 1,
                "n_rows": int(mask.sum()),
                "exposure_sum": float(we_sum),
                "exposure_share": we_sum / total_we,
                "target_sum": _weighted_total(y_obs[mask], we, "y_obs"),
                "actual_avg": actual_avg,
                "model_avg": model_avg,
                "current_avg": current_avg,
                "actual_index": _finite_ratio(
                    actual_avg,
                    overall_actual,
                    "actual index",
                ),
                "model_index": _finite_ratio(
                    model_avg,
                    overall_model,
                    "model index",
                ),
                "current_index": _finite_ratio(
                    current_avg,
                    overall_current,
                    "current index",
                ),
                "sort_score_min": float(sort_score[mask].min()),
                "sort_score_median": float(np.median(sort_score[mask])),
                "sort_score_max": float(sort_score[mask].max()),
            }
        )

    df = pd.DataFrame(rows)

    # ── Plot ──────────────────────────────────────────────────────
    ax_plot, fig = _make_ax(ax)

    x = np.arange(len(df))
    lbl_actual, lbl_model, lbl_current = labels

    # Exposure-share bars on secondary axis (behind lines)
    ax_exp = ax_plot.twinx()
    ax_exp.bar(
        x,
        df["exposure_share"],
        width=0.8,
        alpha=0.08,
        color="grey",
        label="Exposure share",
        zorder=1,
    )
    ax_exp.set_ylabel("Exposure share", fontsize=8, color="grey")
    ax_exp.tick_params(axis="y", colors="grey")

    # Three indexed series
    ax_plot.plot(
        x,
        df["actual_index"],
        "o-",
        label=lbl_actual,
        color="C0",
        markersize=5,
        linewidth=1.5,
        zorder=3,
    )
    ax_plot.plot(
        x,
        df["model_index"],
        "s-",
        label=lbl_model,
        color="C1",
        markersize=5,
        linewidth=1.5,
        zorder=3,
    )
    ax_plot.plot(
        x,
        df["current_index"],
        "^-",
        label=lbl_current,
        color="C2",
        markersize=5,
        linewidth=1.5,
        zorder=3,
    )
    ax_plot.axhline(1.0, color="grey", linewidth=0.7, linestyle="--")

    ax_plot.set_xticks(x)
    ax_plot.set_xticklabels(df["bin"].astype(int))
    ax_plot.set_xlabel(f"Bin (sorted by {lbl_model} / {lbl_current} predicted rate)")
    ax_plot.set_ylabel("Indexed rate (bin avg / overall avg)")
    ax_plot.set_title("Double Lift Chart")
    ax_plot.legend(loc="upper left", fontsize=8)
    ax_exp.legend(loc="upper right", fontsize=7)

    return DoubleLiftChartResult(bins=df, figure=fig)


def lorenz_curve(
    y_obs,
    y_pred,
    sample_weight=None,
    exposure=None,
    *,
    engine: str = "matplotlib",
    ax: Axes | None = None,
) -> LorenzCurveResult:
    """Lorenz curve with Gini coefficient computation.

    Parameters
    ----------
    y_obs : array-like
        Observed response values.
    y_pred : array-like
        Predicted response values.
    sample_weight : array-like or None
        Observation weights.
    exposure : array-like or None
        Exposure measure. When provided, the Lorenz curve uses
        cumulative exposure share on the x-axis.
    engine : {"matplotlib", "plotly"}
        Plotting backend. ``"plotly"`` requires the optional plotly dependency.
    ax : matplotlib Axes or None
        If provided, plot onto this axes. Only valid with
        ``engine="matplotlib"``.

    Returns
    -------
    LorenzCurveResult
        Contains ``curve`` DataFrame, ``gini_model``, ``gini_perfect``,
        ``gini_ratio``, and an optional ``figure``.
    """
    if engine not in {"matplotlib", "plotly"}:
        raise ValueError(f"engine={engine!r} is not valid, expected 'matplotlib' or 'plotly'.")
    if engine == "plotly" and ax is not None:
        raise ValueError("ax= is only supported with engine='matplotlib'.")

    y_obs, vectors, w = _validated_chart_inputs(
        y_obs,
        y_pred=y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        optional_vectors=("exposure",),
    )
    y_pred = vectors["y_pred"]
    n = len(y_obs)
    exp = vectors.get("exposure", np.ones(n, dtype=float))

    losses = y_obs
    total_loss = _scaled_product_total(w, losses, exposure=exp)
    total_exp = _scaled_product_total(w, exp)

    if total_loss[0] > 0 and np.min(losses[(w != 0) & (exp != 0)], initial=0.0) < 0:
        # Signed losses can cancel: every prefix must stay resolved against
        # the total that normalizes it. A degenerate total is handled below.
        magnitude = _scaled_product_total(w, np.abs(losses), exposure=exp)
        _require_resolved(total_loss, magnitude, n, "Lorenz loss prefixes")
    if total_loss[0] <= 0 or total_exp[0] <= 0:
        # Degenerate: all zeros or no exposure
        curve_df = pd.DataFrame(
            {
                "cum_exposure_share": [0.0, 1.0],
                "cum_loss_share_ordered": [0.0, 1.0],
                "cum_loss_share_model": [0.0, 1.0],
                "cum_loss_share_perfect": [0.0, 1.0],
            }
        )
        if engine == "plotly":
            try:
                import plotly.graph_objects as go
            except ImportError:
                raise ImportError(
                    "plotly is required for engine='plotly'. Install it with: pip install plotly"
                ) from None

            from superglm.plotting.common import _PLOTLY_TEXT, _apply_plotly_theme

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[0.0, 1.0],
                    y=[0.0, 1.0],
                    mode="lines",
                    name="Random",
                    line=dict(color=_PLOTLY_TEXT, dash="dash", width=1.2),
                )
            )
            _apply_plotly_theme(
                fig, hovermode="x unified", height=460, margin=dict(t=72, r=28, b=72, l=72)
            )
            fig.update_layout(title="Lorenz Curve (degenerate)")
            fig.update_xaxes(title_text="Cumulative exposure share", range=[0.0, 1.0])
            fig.update_yaxes(title_text="Cumulative loss share", range=[0.0, 1.0])
        else:
            ax_plot, fig = _make_ax(ax)
            ax_plot.plot([0, 1], [0, 1], "k--", linewidth=0.7, label="Random")
            ax_plot.set_title("Lorenz Curve (degenerate)")
            ax_plot.legend(fontsize=7)
        return LorenzCurveResult(
            curve=curve_df, gini_model=0.0, gini_perfect=0.0, gini_ratio=0.0, figure=fig
        )

    # Order by model predictions (ascending = lowest risk first), collapsing
    # tied predictions into a single block so within-tie row order carries no
    # fake ranking information.
    cum_exp_model, cum_loss_model, model_order = _lorenz_cumulative_by_score(
        y_pred,
        w,
        exp,
        losses,
        total_exp=total_exp,
        total_loss=total_loss,
    )

    # Order by actual loss ratio (ascending = lowest actual risk first)
    # For perfect foresight ordering
    positive_exposure = exp > 0
    loss_ratio = np.where(positive_exposure, y_obs, 0.0)
    cum_exp_perfect, cum_loss_perfect, perfect_order = _lorenz_cumulative_by_score(
        loss_ratio,
        w,
        exp,
        losses,
        total_exp=total_exp,
        total_loss=total_loss,
    )

    # Random ordering = diagonal
    # Prepend (0, 0)
    cum_exp_m = np.concatenate([[0.0], cum_exp_model])
    cum_loss_m = np.concatenate([[0.0], cum_loss_model])
    cum_exp_p = np.concatenate([[0.0], cum_exp_perfect])
    cum_loss_p = np.concatenate([[0.0], cum_loss_perfect])

    # Gini coefficients
    gini_model, gini_perfect, gini_ratio = _gini_coefficients(
        y_obs,
        y_pred,
        w,
        exposure=vectors.get("exposure"),
        totals=(total_exp, total_loss),
        score_orders=(perfect_order if np.all(positive_exposure) else None, model_order),
    )

    # Build curve DataFrame — use model ordering x-axis for all curves
    # Random ordering diagonal: cum_loss_share == cum_exposure_share
    curve_df = pd.DataFrame(
        {
            "cum_exposure_share": cum_exp_m,
            "cum_loss_share_ordered": cum_exp_m,
            "cum_loss_share_model": cum_loss_m,
            "cum_loss_share_perfect": np.interp(cum_exp_m, cum_exp_p, cum_loss_p),
        }
    )

    # Plot
    if engine == "plotly":
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly is required for engine='plotly'. Install it with: pip install plotly"
            ) from None

        from superglm.plotting.common import (
            _PLOTLY_LINE_COLOR,
            _PLOTLY_SIM_FILL,
            _PLOTLY_TEXT,
            _apply_plotly_theme,
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                name="Random",
                line=dict(color=_PLOTLY_TEXT, dash="dash", width=1.2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cum_exp_m,
                y=cum_loss_m,
                mode="lines",
                name="Model",
                line=dict(color=_PLOTLY_LINE_COLOR, width=2.4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cum_exp_p,
                y=cum_loss_p,
                mode="lines",
                name="Perfect",
                line=dict(color=_PLOTLY_SIM_FILL, width=2.0),
                opacity=0.85,
            )
        )
        _apply_plotly_theme(
            fig,
            hovermode="x unified",
            height=460,
            margin=dict(t=72, r=28, b=72, l=72),
        )
        fig.update_layout(title=f"Lorenz Curve (Gini ratio = {gini_ratio:.3f})")
        fig.update_xaxes(title_text="Cumulative exposure share", range=[0.0, 1.0])
        fig.update_yaxes(title_text="Cumulative loss share", range=[0.0, 1.0])
    else:
        ax_plot, fig = _make_ax(ax)
        ax_plot.plot([0, 1], [0, 1], "k--", linewidth=0.7, label="Random")
        ax_plot.plot(cum_exp_m, cum_loss_m, "-", color="C0", linewidth=1.2, label="Model")
        ax_plot.plot(
            cum_exp_p, cum_loss_p, "-", color="C2", linewidth=1.0, alpha=0.7, label="Perfect"
        )
        ax_plot.set_xlabel("Cumulative exposure share")
        ax_plot.set_ylabel("Cumulative loss share")
        ax_plot.set_title(f"Lorenz Curve (Gini ratio = {gini_ratio:.3f})")
        ax_plot.legend(fontsize=8)

    return LorenzCurveResult(
        curve=curve_df,
        gini_model=gini_model,
        gini_perfect=gini_perfect,
        gini_ratio=gini_ratio,
        figure=fig,
    )


def loss_ratio_chart(
    y_obs,
    y_pred,
    sample_weight=None,
    exposure=None,
    *,
    n_bins: int = 10,
    feature_values=None,
    feature_name: str | None = None,
    ax: Axes | None = None,
) -> LossRatioChartResult:
    """Loss ratio chart: observed vs predicted loss ratios per bin.

    Parameters
    ----------
    y_obs : array-like
        Observed response values.
    y_pred : array-like
        Predicted response values.
    sample_weight : array-like or None
        Observation weights.
    exposure : array-like or None
        Exposure measure for rate models.
    n_bins : int
        Number of quantile bins.
    feature_values : array-like or None
        If provided, bin by this feature's values instead of predicted values.
    feature_name : str or None
        Label for the feature axis.
    ax : matplotlib Axes or None
        If provided, plot onto this axes.

    Returns
    -------
    LossRatioChartResult
        Contains a ``bins`` DataFrame and an optional ``figure``.
    """
    y_obs, vectors, w = _validated_chart_inputs(
        y_obs,
        y_pred=y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        feature_values=feature_values,
        optional_vectors=("exposure", "feature_values"),
    )
    y_pred = vectors["y_pred"]
    n_bins = _validated_n_bins(n_bins)
    n = len(y_obs)
    exp = vectors.get("exposure", np.ones(n, dtype=float))

    # Determine what to bin by
    if feature_values is not None:
        sort_vals = vectors["feature_values"]
        x_label = feature_name or "Feature"
    else:
        sort_vals = y_pred
        x_label = "Predicted value"

    bin_weights = w * exp
    bins_idx = _quantile_bins(sort_vals, bin_weights, n_bins)

    rows = []
    total_exp = bin_weights.sum()
    for b in range(n_bins):
        mask = bins_idx == b
        if not mask.any():
            continue
        we = w[mask] * exp[mask]
        we_sum = we.sum()
        obs_lr = _weighted_mean(y_obs[mask], we, "y_obs")
        pred_lr = _weighted_mean(y_pred[mask], we, "y_pred")
        rows.append(
            {
                "bin": b + 1,
                "exposure_share": we_sum / total_exp if total_exp > 0 else 0.0,
                "observed": obs_lr,
                "predicted": pred_lr,
            }
        )

    df = pd.DataFrame(rows)

    ax_plot, fig = _make_ax(ax)
    x = np.arange(len(df))
    width = 0.35
    ax_plot.bar(x - width / 2, df["observed"], width, label="Observed", color="C0")
    ax_plot.bar(x + width / 2, df["predicted"], width, label="Predicted", color="C1")

    # Volume overlay
    ax2 = ax_plot.twinx()
    ax2.bar(x, df["exposure_share"], width=0.8, alpha=0.15, color="grey", label="Exposure share")
    ax2.set_ylabel("Exposure share")

    ax_plot.set_xticks(x)
    ax_plot.set_xticklabels(df["bin"].astype(int))
    ax_plot.set_xlabel(x_label)
    ax_plot.set_ylabel("Loss ratio")
    ax_plot.set_title("Loss Ratio Chart")
    ax_plot.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    return LossRatioChartResult(bins=df, figure=fig)
