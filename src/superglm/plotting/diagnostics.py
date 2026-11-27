"""GLM/GAM diagnostic plots for SuperGLM models.

Four-panel figure using quantile residuals (Dunn & Smyth 1996) with
simulation-based Q-Q envelopes. Designed for Poisson, Gamma, NB2,
Tweedie, Binomial, and Gaussian families.
"""

from __future__ import annotations

import html
import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import scipy.spatial
from numba import njit
from scipy import stats

from superglm._utils import bin_rows, occurrence_rows, score_bins
from superglm._weight_units import dispersion_standard_deviation, likelihood_scale
from superglm.plotting.common import (
    _LINE_COLOR,
    _LINE_WIDTH,
    _NEUTRAL_FILL,
    _REF_COLOR,
    _SPECIAL_COLOR,
    _TEXT_COLOR,
    _style_matplotlib_axis,
)
from superglm.solvers.dispersion import PRIOR_WEIGHTS, model_weight_semantics

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from superglm.model import SuperGLM


# ── LOWESS smoother (Numba-accelerated) ───────────────────────────


@njit(cache=True)
def _lowess_core(x_s, y_s, k, n_steps):
    """Numba-accelerated LOWESS with tricube kernel and sliding window."""
    n = len(x_s)
    y_hat = y_s.copy()
    robustness_w = np.ones(n)

    for _iteration in range(1 + n_steps):
        left = 0
        for i in range(n):
            while left + k < n:
                if x_s[i] - x_s[left] > x_s[left + k] - x_s[i]:
                    left += 1
                else:
                    break

            right = min(left + k, n)
            h = max(x_s[i] - x_s[left], x_s[right - 1] - x_s[i]) + 1e-10
            # The weights are a function of distance: a row tied with either
            # end of the window is as near as that end and joins it, so which
            # rows of a tier the window holds never depends on their order.
            lo, hi = left, right
            while lo > 0 and x_s[lo - 1] == x_s[left]:
                lo -= 1
            while hi < n and x_s[hi] == x_s[right - 1]:
                hi += 1

            sw = 0.0
            swx = 0.0
            swy = 0.0
            swxx = 0.0
            swxy = 0.0
            for j in range(lo, hi):
                u = abs(x_s[j] - x_s[i]) / h
                if u < 1.0:
                    u3 = u * u * u
                    t = 1.0 - u3
                    w = t * t * t * robustness_w[j]
                else:
                    w = 0.0
                sw += w
                wx = w * x_s[j]
                swx += wx
                swy += w * y_s[j]
                swxx += wx * x_s[j]
                swxy += w * x_s[j] * y_s[j]

            if sw < 1e-12:
                # Every robustness weight in the window vanished, which only a
                # reweighting pass can do: keep the previous pass's fit, which
                # rows sharing an x share.
                continue

            mx = swx / sw
            my = swy / sw
            den = swxx - swx * swx / sw
            if abs(den) < 1e-12:
                y_hat[i] = my
            else:
                num = swxy - swx * swy / sw
                y_hat[i] = my + (num / den) * (x_s[i] - mx)

        if _iteration < n_steps:
            abs_resid = np.empty(n)
            for i in range(n):
                abs_resid[i] = abs(y_s[i] - y_hat[i])
            med_resid = np.median(abs_resid)
            if med_resid < 1e-12:
                break
            inv6m = 1.0 / (6.0 * med_resid)
            for i in range(n):
                u = (y_s[i] - y_hat[i]) * inv6m
                au = abs(u)
                if au < 1.0:
                    t = 1.0 - u * u
                    robustness_w[i] = t * t
                else:
                    robustness_w[i] = 0.0

    return y_hat


def _lowess(x: NDArray, y: NDArray, *, frac: float = 0.6, n_steps: int = 2) -> NDArray:
    """LOWESS smoother (numba-accelerated, tricube kernel, robustness iterations)."""
    n = len(x)
    if n < 3:
        return y.copy()

    order = np.argsort(x)
    x_s = np.ascontiguousarray(x[order], dtype=np.float64)
    y_s = np.ascontiguousarray(y[order], dtype=np.float64)
    k = max(int(np.ceil(frac * n)), 3)

    y_hat = _lowess_core(x_s, y_s, k, n_steps)

    result = np.empty(n)
    result[order] = y_hat
    return result


# ── Simulation helpers ────────────────────────────────────────────


def _validate_diagnostic_weights(
    family,
    sample_weight,
    n_rows: int,
    *,
    weight_semantics: str,
) -> NDArray:
    """Normalize diagnostic weights under the declared weight contract.

    Strict positivity is a Tweedie density requirement: only the prior
    contract puts ``log w`` inside the compound-Poisson normalizer. A
    replication weight of zero is a row that appears no times, as in fitting.
    """
    from superglm.distributions import prior_weights_in_normalizer

    if n_rows == 0:
        raise ValueError("y and sample_weight must be non-empty")
    if sample_weight is None:
        return np.ones(n_rows, dtype=np.float64)

    message = f"sample_weight must be a numeric one-dimensional array of length {n_rows}"
    try:
        raw = np.asarray(sample_weight)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if (
        raw.ndim != 1
        or len(raw) != n_rows
        or np.iscomplexobj(raw)
        or getattr(raw.dtype, "kind", None) in {"M", "m"}
    ):
        raise ValueError(message)
    try:
        weights = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight must contain only finite values")
    if prior_weights_in_normalizer(family) and weight_semantics == PRIOR_WEIGHTS:
        if np.any(weights <= 0.0):
            raise ValueError("Tweedie sample_weight must be strictly positive")
    else:
        if np.any(weights < 0.0):
            raise ValueError("sample_weight must be nonnegative")
        if not np.any(weights > 0.0):
            raise ValueError("sample_weight must contain at least one positive value")
    return weights


def _simulate_response(
    family,
    mu,
    phi,
    sample_weight,
    rng,
    *,
    weight_semantics: str,
    row_scale: NDArray | None = None,
) -> NDArray | None:
    """Simulate one response vector from the fitted model, or None for an unsupported family.

    A replication weight does not alter a row's distribution, so the frequency contract
    simulates at ``w = 1``; a prior weight stays in the draw. Draws come from exactly the
    marginals ``_quantile_residuals`` inverts (Gaussian ``phi / w`` variance, Gamma ``w / phi``
    shape, the ``w``-scaled Poisson and negative binomial lattice, binomial ``w`` trials), or a
    correct fit falls outside its own envelope. ``row_scale`` is each row's ``phi / w`` (``phi``
    under replication) where the caller formed it in the fit's weight unit, because ``phi``
    itself underflows there.
    """
    from superglm.distributions import (
        Binomial,
        Gamma,
        Gaussian,
        NegativeBinomial,
        Poisson,
        Tweedie,
    )

    prior = weight_semantics == PRIOR_WEIGHTS
    weights = np.asarray(sample_weight, dtype=np.float64)

    if isinstance(family, Poisson):
        # w Y ~ Poisson(w mu): draw on the scaled lattice and return the rate.
        if prior:
            return np.asarray(rng.poisson(lam=weights * mu), dtype=np.float64) / weights
        return np.asarray(rng.poisson(lam=mu), dtype=np.float64)

    if isinstance(family, Gaussian):
        variance = row_scale if row_scale is not None else phi / weights if prior else phi
        return rng.normal(loc=mu, scale=np.sqrt(variance))

    if isinstance(family, Gamma):
        if row_scale is not None:
            return rng.gamma(shape=1.0 / row_scale, scale=mu * row_scale)
        shape = (weights if prior else 1.0) / phi
        scale = mu * phi / (weights if prior else 1.0)
        return rng.gamma(shape=shape, scale=scale)

    if isinstance(family, NegativeBinomial):
        from scipy.stats import nbinom as nbinom_dist

        theta = family.theta
        # w Y ~ NB2(w mu, w theta) scales the size parameter and leaves the
        # success probability theta/(mu + theta) untouched.
        p_nb = theta / (mu + theta)
        n_param = weights * theta if prior else theta
        draws = nbinom_dist.rvs(n=n_param, p=p_nb, random_state=rng).astype(float)
        return draws / weights if prior else draws

    if isinstance(family, Tweedie):
        from superglm._tweedie import generate_tweedie_cpg

        if row_scale is not None:
            row_phi = row_scale
        else:
            row_phi = phi / weights if prior else np.full_like(weights, phi)
        return generate_tweedie_cpg(len(mu), mu, row_phi, family.p, rng=rng)

    if isinstance(family, Binomial):
        # As in `_quantile_residuals`: the prior likelihood's masses at y in {0, 1}
        # do not sum to one for w != 1, so no uniform PIT exists; Bernoulli(mu)
        # keeps the envelope a valid reference band.
        return rng.binomial(n=1, p=mu).astype(float)

    return None


# ── Trend helpers ─────────────────────────────────────────────────

_LOWESS_CAP = 5_000
_LARGE_N = 20_000
_VERY_LARGE_N = 100_000
_QQ_GRID = 1000
_ENVELOPE_CAP = 50_000
_ENVELOPE_SIM_CAP = 50
_BIN_COUNT = 50
_FREQUENCY_EXPANSION_CAP = 100_000


def _stream_seed(stream: np.random.SeedSequence) -> int:
    """Return an integer seed for ``stream``, for APIs that take an integer ``seed``."""
    return int(stream.generate_state(1, np.uint64)[0])


def _diagnostic_rows(
    model,
    y,
    mu,
    eta,
    weights,
    qresid,
    seed,
    rng,
    *,
    weight_semantics: str,
):
    """Return row-level plot arrays honoring frequency-weight replication.

    The frequency contract, not the family, decides whether rows expand. Under
    the prior contract the weight lives in each row's marginal, which the
    quantile residual already carries, so there is nothing to expand.
    """
    if np.all(weights == 1.0):
        return y, mu, eta, weights, qresid

    positive = weights > 0.0
    if weight_semantics == PRIOR_WEIGHTS:
        # A zero prior weight is an unobserved row; it leaves the plot as it
        # leaves the likelihood.
        if np.all(positive):
            return y, mu, eta, weights, qresid
        return (
            y[positive],
            mu[positive],
            eta[positive],
            weights[positive],
            qresid[positive],
        )

    rounded = np.rint(weights)
    if not np.array_equal(weights, rounded):
        # Fractional replication weights have no literal row array. Keep their
        # weighted calibration representation while dropping zero-mass rows.
        return (
            y[positive],
            mu[positive],
            eta[positive],
            weights[positive],
            qresid[positive],
        )

    row_idx = occurrence_rows(rounded, _FREQUENCY_EXPANSION_CAP, rng)
    from superglm.inference.metrics import ModelMetrics

    row_y = y[row_idx]
    row_mu = mu[row_idx]
    row_weights = np.ones(len(row_idx), dtype=np.float64)
    row_metrics = ModelMetrics(
        model,
        y=row_y,
        sample_weight=row_weights,
        _mu=row_mu,
    )
    row_qresid = row_metrics.residuals("quantile", seed=seed)
    return row_y, row_mu, eta[row_idx], row_weights, row_qresid


def _trend_line(ax, x, y, n, rng):
    """Add a trend line appropriate for the data size."""
    if n < 20:
        return
    if n <= _LARGE_N:
        # Full LOWESS
        sm = _lowess(x, y)
        order = np.argsort(x)
        _plot_trend(ax, x[order], sm[order])
    elif n <= _VERY_LARGE_N:
        # Sampled LOWESS
        idx = rng.choice(n, _LOWESS_CAP, replace=False)
        sm = _lowess(x[idx], y[idx])
        order = np.argsort(x[idx])
        _plot_trend(ax, x[idx][order], sm[order])
    else:
        # Bin medians, in score bins with ties kept together
        _binned_median_line(ax, x, y)


def _plot_trend(ax, x_sorted, trend):
    """Draw a trend through points sorted by x; one distinct x is drawn as a point.

    Rows sharing an x share their trend value, and a line through one
    coordinate draws nothing, so a single tier is marked instead.
    """
    if x_sorted[0] == x_sorted[-1]:
        ax.plot(x_sorted[:1], trend[:1], color=_SPECIAL_COLOR, marker="o")
    else:
        ax.plot(x_sorted, trend, color=_SPECIAL_COLOR, linewidth=_LINE_WIDTH)


def _binned_median_line(ax, x, y, n_bins=_BIN_COUNT):
    """Plot the median trend in score bins of ``x``, ties kept together."""
    _, groups = bin_rows(score_bins(x, np.ones(len(x)), n_bins))
    mids = np.array([np.median(x[group]) for group in groups])
    medians = np.array([np.median(y[group]) for group in groups])
    _plot_trend(ax, mids, medians)


def _scatter_or_hexbin(ax, x, y, n, max_points):
    """Scatter for small n, log-density hexbin for large n."""
    from matplotlib.colors import LogNorm

    if n <= max_points:
        ax.scatter(x, y, s=6, alpha=0.3, edgecolors="none", color=_LINE_COLOR)
    else:
        gridsize = min(80, max(30, n // 5000))
        ax.hexbin(
            x,
            y,
            gridsize=gridsize,
            cmap="Blues",
            mincnt=1,
            norm=LogNorm(),
        )


def _stratified_sample(mu, n_sample, rng):
    """Sample indices stratified by mu quantiles to preserve tails."""
    n = len(mu)
    if n <= n_sample:
        return np.arange(n)
    n_strata = 20
    strata_edges = np.percentile(mu, np.linspace(0, 100, n_strata + 1))
    strata_edges[-1] += 1e-10  # include max
    idx_list = []
    per_stratum = n_sample // n_strata
    for i in range(n_strata):
        mask = (mu >= strata_edges[i]) & (mu < strata_edges[i + 1])
        stratum_idx = np.where(mask)[0]
        if len(stratum_idx) == 0:
            continue
        k = min(per_stratum, len(stratum_idx))
        idx_list.append(rng.choice(stratum_idx, k, replace=False))
    return np.concatenate(idx_list)


def _panel_calibration(ax, y, mu, w, n_bins=20):
    """Panel 2: Weight-balanced calibration by predicted-value bins."""
    codes = score_bins(mu, w, n_bins)
    mass = np.bincount(codes, weights=w, minlength=n_bins)
    filled = mass > 0
    pred_means = np.bincount(codes, weights=w * mu, minlength=n_bins)[filled] / mass[filled]
    obs_means = np.bincount(codes, weights=w * y, minlength=n_bins)[filled] / mass[filled]

    # Calibration scatter + y=x
    ax.scatter(
        pred_means, obs_means, s=30, color=_LINE_COLOR, zorder=3, edgecolors="white", linewidth=0.5
    )
    lims = [
        min(pred_means.min(), obs_means.min()) * 0.9,
        max(pred_means.max(), obs_means.max()) * 1.1,
    ]
    ax.plot(lims, lims, color=_REF_COLOR, linewidth=0.8, linestyle="--", label="y = x")
    ax.set_xlabel("Predicted (bin mean)")
    ax.set_ylabel("Observed (bin mean)")
    bins = f"{len(pred_means)} score bin" + ("" if len(pred_means) == 1 else "s")
    ax.set_title(f"Calibration ({bins}, ties kept together)")
    ax.legend(fontsize=7, loc="upper left")


# ── Main diagnostic function ─────────────────────────────────────


def plot_diagnostics(
    model: SuperGLM,
    X,
    y: NDArray,
    sample_weight: NDArray | None = None,
    offset: NDArray | None = None,
    *,
    n_sim: int = 100,
    figsize: tuple[float, float] | None = None,
    max_points: int = 50_000,
    seed: int = 42,
    residual_type: str = "auto",
) -> Figure:
    """GLM/GAM diagnostic figure with simulation-based Q-Q envelope.

    Four panels using quantile residuals (Dunn & Smyth 1996):

    1. **Q-Q with simulation envelope** — observed quantile residuals vs
       simulated reference, with 95% pointwise envelope.
    2. **Calibration** — observed vs predicted values in score bins of equal
       weight, ties kept together: rows with one predicted value share a bin, so
       tied predictions give fewer, uneven bins. Integer replication weights
       are represented as replicated rows; prior weights are used as mass.
    3. **Residuals vs Linear Predictor** — quantile residuals vs eta.
    4. **Residual distribution** — histogram with N(0,1) overlay.

    Parameters
    ----------
    model : SuperGLM
        A fitted SuperGLM model.
    X : pandas or eager Polars DataFrame
        Design matrix.
    y : NDArray
        Response vector.
    sample_weight : NDArray or None
        Optional observation weights, read under the model's declared
        ``weight_semantics``.  Under ``"frequency"`` integer values are
        represented as literal row replication in the randomized diagnostics
        up to 100,000 occurrences; larger populations read a reproducible
        subset of 100,000 occurrences drawn without replacement, and all-unit
        weights read every row.  Under ``"prior"`` the weight is part of
        the row's distribution through ``phi / w``, and a Tweedie fit
        additionally requires it finite and strictly positive.
    offset : NDArray or None
        Optional offset.
    n_sim : int
        Number of simulation replicates for the Q-Q envelope. Default 100.
    figsize : tuple or None
        Figure size ``(width, height)`` in inches. Defaults to ``(10, 8)``.
    max_points : int
        Threshold for scatter vs hexbin rendering. Default 50,000.
    seed : int
        Random seed for quantile residuals, simulation, and subsampling.
    residual_type : str
        .. deprecated::
            All panels now use quantile residuals. This parameter is
            ignored. Pass ``"auto"`` (default) to suppress the warning.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with 4 diagnostic subplots.
    """
    if residual_type != "auto":
        warnings.warn(
            "residual_type is deprecated and ignored — all panels now use "
            "quantile residuals (Dunn & Smyth 1996). Pass 'auto' or omit "
            "to suppress this warning.",
            FutureWarning,
            stacklevel=2,
        )

    if figsize is None:
        figsize = (10, 8)

    rng = np.random.default_rng(seed)
    # A response draw and its quantile-residual jitter must not share uniforms
    # (for a Bernoulli draw they would coincide), nor the occurrence selection
    # and the jitter of the occurrences it selects: each gets its own stream.
    selection_stream, observed_stream, simulation_stream = np.random.SeedSequence(seed).spawn(3)
    observed_seed = _stream_seed(observed_stream)

    family = getattr(model, "_distribution")
    y_input = np.asarray(y)
    if y_input.ndim != 1:
        raise ValueError("y must be one-dimensional")
    # Resolved once, and handed to every helper that has to agree with the
    # likelihood the residuals were built under.
    weight_semantics = model_weight_semantics(model)
    diagnostic_weights = _validate_diagnostic_weights(
        family,
        sample_weight,
        len(y_input),
        weight_semantics=weight_semantics,
    )

    # Build metrics object
    metrics_weights = None if sample_weight is None else diagnostic_weights
    m = model.metrics(X, y, metrics_weights, offset)
    mu = m._mu
    eta = m.eta
    y_arr = m._y
    original_weights = diagnostic_weights

    # Quantile residuals for all panels
    qresid = m.residuals("quantile", seed=observed_seed)
    y_arr, mu, eta, w, qresid = _diagnostic_rows(
        model,
        y_arr,
        mu,
        eta,
        original_weights,
        qresid,
        observed_seed,
        np.random.default_rng(selection_stream),
        weight_semantics=weight_semantics,
    )
    n = len(y_arr)

    # Family/link label for suptitle
    family_name = type(family).__name__
    link_name = type(getattr(model, "_link")).__name__

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.set_facecolor("white")
    fig.suptitle(
        f"Diagnostics: {family_name}({link_name})",
        fontsize=14,
        fontweight="bold",
        color=_TEXT_COLOR,
        y=0.98,
    )

    # ── Panel 1: Q-Q with simulation envelope ────────────────────
    ax1 = axes[0, 0]
    _panel_qq_envelope(
        ax1,
        model,
        m,
        mu,
        w,
        qresid,
        n,
        n_sim,
        simulation_stream,
        rng,
        weight_semantics=weight_semantics,
    )

    # ── Panel 2: Calibration plot (family-weight aware) ─────────
    ax2 = axes[0, 1]
    _panel_calibration(ax2, y_arr, mu, w)

    # ── Panel 3: Residuals vs Linear Predictor ───────────────────
    ax3 = axes[1, 0]
    _scatter_or_hexbin(ax3, eta, qresid, n, max_points)
    ax3.axhline(0, color=_REF_COLOR, linewidth=0.7, linestyle="--")
    _trend_line(ax3, eta, qresid, n, rng)
    ax3.set_xlabel("Linear predictor")
    ax3.set_ylabel("Quantile residuals")
    ax3.set_title("Residuals vs Linear Predictor")

    # ── Panel 4: Residual distribution ───────────────────────────
    ax4 = axes[1, 1]
    finite = qresid[np.isfinite(qresid)]
    lo, hi = np.percentile(finite, [0.5, 99.5])
    clipped = finite[(finite >= lo) & (finite <= hi)]
    ax4.hist(clipped, bins=80, density=True, color=_NEUTRAL_FILL, edgecolor="none")
    # N(0,1) overlay
    x_norm = np.linspace(lo, hi, 200)
    ax4.plot(
        x_norm, stats.norm.pdf(x_norm), color=_LINE_COLOR, linewidth=_LINE_WIDTH, label="N(0,1)"
    )
    ax4.legend(fontsize=8)
    ax4.set_xlabel("Quantile residuals")
    ax4.set_ylabel("Density")
    # Annotate with phi and Pearson chi2/df
    from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom

    phi = m.phi_likelihood  # the scale the quantile residuals above were drawn at
    resid_df = pearson_residual_degrees_of_freedom(
        original_weights,
        m.effective_df,
        weight_semantics=m._weight_semantics,
    )
    chi2_ratio = m.pearson_chi2 / resid_df
    ax4.set_title(f"Residual Distribution (φ={phi:.3g}, χ²/df={chi2_ratio:.2f})")

    for ax in axes.flat:
        _style_matplotlib_axis(ax)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _panel_qq_envelope(
    ax, model, m, mu, w, qresid, n, n_sim, simulation_stream, rng, *, weight_semantics
):
    """Panel 1: Q-Q plot with simulation envelope."""
    family = model._distribution
    phi = m.phi_likelihood  # the envelope simulates the fitted density

    # Determine sample/grid sizes for large n
    effective_n = n
    sim_n = n_sim
    sim_idx = None
    title_suffix = ""

    if n > _VERY_LARGE_N:
        sim_idx = _stratified_sample(mu, _ENVELOPE_CAP, rng)
        effective_n = len(sim_idx)
        sim_n = min(n_sim, _ENVELOPE_SIM_CAP)
        title_suffix = f" ({effective_n // 1000}k sample)"

    # Observed sorted quantile residuals (possibly on sample)
    obs_qr = np.sort(qresid[sim_idx] if sim_idx is not None else qresid)

    # Use a quantile grid for large n
    if effective_n > _QQ_GRID:
        grid_idx = np.linspace(0, effective_n - 1, _QQ_GRID).astype(int)
        obs_grid = obs_qr[grid_idx]
    else:
        grid_idx = np.arange(effective_n)
        obs_grid = obs_qr

    n_grid = len(grid_idx)
    # Blom's positions (k - 3/8) / (n + 1/4) count ranks k from 1; grid_idx counts from 0.
    theoretical = stats.norm.ppf((grid_idx + 0.625) / (effective_n + 0.25))

    # Simulate envelope
    envelope_ok = False
    sim_mu = mu[sim_idx] if sim_idx is not None else mu
    sim_w = w[sim_idx] if sim_idx is not None else w
    row_scale = None
    density_scale = likelihood_scale(m._result)
    if density_scale.solver_phi is not None:
        # A lifted weight unit: phi / w is formed in that unit, where both are normal.
        root = dispersion_standard_deviation(
            sim_w if weight_semantics == PRIOR_WEIGHTS else np.ones_like(sim_w), density_scale
        )
        row_scale = root * root

    try:
        from superglm.inference.metrics import ModelMetrics

        sim_sorted = np.empty((sim_n, n_grid))
        for i, replicate in enumerate(simulation_stream.spawn(sim_n)):
            response_stream, jitter_stream = replicate.spawn(2)
            y_sim = _simulate_response(
                family,
                sim_mu,
                phi,
                sim_w,
                np.random.default_rng(response_stream),
                weight_semantics=weight_semantics,
                row_scale=row_scale,
            )
            if y_sim is None:
                break
            # Compute quantile residuals for simulated data
            m_sim = ModelMetrics(model, y=y_sim, sample_weight=sim_w, _mu=sim_mu)
            qr_sim = m_sim.residuals("quantile", seed=_stream_seed(jitter_stream))
            sorted_sim = np.sort(qr_sim)
            if effective_n > _QQ_GRID:
                sim_sorted[i] = sorted_sim[grid_idx]
            else:
                sim_sorted[i] = sorted_sim
        else:
            envelope_ok = True
    except Exception:
        envelope_ok = False

    if envelope_ok:
        median_sim = np.median(sim_sorted, axis=0)
        lo_env = np.percentile(sim_sorted, 2.5, axis=0)
        hi_env = np.percentile(sim_sorted, 97.5, axis=0)

        ax.fill_between(
            theoretical,
            lo_env,
            hi_env,
            alpha=0.5,
            color=_NEUTRAL_FILL,
            label="95% pointwise envelope",
        )
        ax.plot(
            theoretical,
            obs_grid,
            color=_LINE_COLOR,
            linewidth=_LINE_WIDTH,
            zorder=3,
            label="Observed",
        )
        ax.plot(
            theoretical,
            median_sim,
            color=_REF_COLOR,
            linewidth=0.8,
            linestyle="--",
            label="Simulated median",
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title(f"Q-Q Envelope{title_suffix}")
    else:
        # Fallback: plain Q-Q without envelope
        warnings.warn(
            f"Simulation envelope not available for "
            f"{type(family).__name__}. Showing plain Q-Q plot.",
            stacklevel=3,
        )
        ax.plot(
            theoretical,
            obs_grid,
            color=_LINE_COLOR,
            linewidth=_LINE_WIDTH,
        )
        lims = [
            min(theoretical.min(), obs_grid.min()),
            max(theoretical.max(), obs_grid.max()),
        ]
        ax.plot(lims, lims, color=_REF_COLOR, linewidth=0.8, linestyle="--")
        ax.set_title(f"Q-Q Plot (no envelope){title_suffix}")

    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Quantile residuals")
