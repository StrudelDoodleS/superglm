"""Characterise the Tweedie and NB2 profilers before the profiling rebuild.

Runs the pre-rebuild code and writes ``tests/fixtures/tweedie_nb_characterisation.json``,
the outputs the rebuilt code is compared against:

* ``logpdf``: ``tweedie_logpdf`` on a (y, mu, phi, p, w) grid. Each row is
  evaluated alone, so it takes the route the old evaluator gives a single row
  (Wright, the p = 1.5 Bessel form, the series, or the saddlepoint), recorded
  as ``route``. ``logpdf_exact`` is the 50-digit mpmath log-density of the same
  float64 inputs, or ``None`` where the series peak index exceeds 3e5.
  ``old_route_max_rel_error`` is the largest ``|logpdf - logpdf_exact| /
  max(1, |logpdf|)`` over the rows not sent to the saddlepoint; the float64
  cancellation between log W and the canonical term is part of it, because
  every float64 evaluator pays it.
* ``reml_phi``: ``fit_reml`` dispersion on the ``re_*.csv`` fixtures at their
  powers (``reml_phi_refused`` where ``fit_reml`` refuses the design), and
  ``reml_phi_books`` on the two synthetic books at p = 1.5.
* ``estimate_p`` / ``estimate_theta``: estimates and 95% profile intervals, and
  under ``*_refused`` the cases the pre-rebuild code refuses, with the error.
  An interval the pre-rebuild code raises on is ``None``, with ``ci95_error``.
  ``ci95_slopes`` is |d nll / dp| at each interval endpoint under the local
  quadratic profile, ``lr / (ll_scale |endpoint - p_hat|)`` (``None`` for a
  side that is not a root). ``phi_log_curvature`` is d2 nll / d(log phi)^2 at
  the published fit, a central second difference with step 1e-2 in log phi.
  ``mean_logpdf_scale`` is the mean of ``max(1, |logpdf_i|)`` there. With
  ``old_route_max_rel_error`` these give the derived fit tolerances.

``--series-scan`` instead writes stage 0 measurement (a): for every positive
row of every Tweedie case, the Dunn-Smyth series work at p in {1.05, 1.2, 1.5,
1.8, 1.95} and phi at 1e-3, 0.1, 1, 10 and 1e3 times the Pearson estimate.

The logpdf, scan and estimate arms read the pre-rebuild code's private names
and result fields, so they document what was measured and run only against a
pre-rebuild checkout.
``build_case`` uses public API alone and is imported by the tests on either
side of the rebuild.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    uv run --with mpmath python benchmarks/tweedie_nb_characterisation.py \
        --out tests/fixtures/tweedie_nb_characterisation.json
"""

from __future__ import annotations

import argparse
import datetime
import functools
import itertools
import json
import math
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import superglm
from superglm import (
    Categorical,
    CubicRegressionSpline,
    NegativeBinomial,
    RandomEffect,
    Spline,
    SuperGLM,
    Tweedie,
    generate_tweedie_cpg,
    tweedie_logpdf,
)

REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURES = REPOSITORY / "tests" / "fixtures"

BOOK_SEED = 20260926
BOOK_ROWS = 30_000
LEVEL_EFFECTS = np.array([0.0, 0.2, -0.3, 0.1, 0.4])
# (intercept, phi, p): about 90% zeros, and about 96% positive rows (the editor-demo shape).
BOOK_SHAPES = {"zeros90": (-2.4, 6.0, 1.5), "positive96": (0.15, 0.6, 1.45)}
RE_POWERS = {"re_ident_p15": 1.5, "re_flat_p15_lowzero": 1.5, "re_flat_p18": 1.8}
NB_KNOTS = {"nb_clamp005": 10, "nb_worst": 20, "nb_poisson": 10}
TWEEDIE_CASES = (*RE_POWERS, *BOOK_SHAPES, "unpen")
FIT_MODES = ("fit", "reml")

LOGPDF_POWERS = (1.01, 1.05, 1.1, 1.3, 1.5, 1.7, 1.9, 1.95, 1.99)
LOGPDF_PHIS = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
LOGPDF_RESPONSES = (0.0, 1e-3, 0.1, 1.0, 10.0, 1e3)
LOGPDF_WEIGHTS = (1.0, 3.5)
REFERENCE_DIGITS = 50
REFERENCE_MAX_MODE = 3e5
# The old evaluator's default Wright-argument ceiling.
OLD_T_ARG_LIMIT = 1e14

SCAN_POWERS = (1.05, 1.2, 1.5, 1.8, 1.95)
SCAN_PHI_MULTIPLIERS = (1e-3, 0.1, 1.0, 10.0, 1e3)
SERIES_LOG_CUTOFF = 37.0
SERIES_MAX_ROW_TERMS = 1_000_000
SERIES_MAX_SAFE_MODE = float(2**52)
THREAD_POOLS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _book(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intercept, phi, p = BOOK_SHAPES[name]
    rng = np.random.default_rng(BOOK_SEED)
    x = rng.uniform(0.0, 1.0, BOOK_ROWS)
    level = rng.integers(0, len(LEVEL_EFFECTS), BOOK_ROWS)
    mu = np.exp(intercept + 0.5 * np.sin(2.0 * np.pi * x) + LEVEL_EFFECTS[level])
    y = generate_tweedie_cpg(BOOK_ROWS, mu, phi, p, rng=rng)
    return x, np.char.add("L", level.astype(str)), y


def _random_effect_case(name: str, weight_semantics: str):
    data = pd.read_csv(FIXTURES / f"{name}.csv")
    X = pd.DataFrame(
        {
            "u": pd.Categorical(data["u"].astype(str)),
            "r": pd.Categorical(data["r"].astype(str)),
        }
    )
    model = SuperGLM(
        features={"u": Categorical(), "r": RandomEffect()},
        family=Tweedie(p=RE_POWERS[name]),
        weight_semantics=weight_semantics,
    )
    return model, X, data["y"].to_numpy(dtype=np.float64)


def _negative_binomial_case(name: str, weight_semantics: str):
    if name == "nb_poisson":
        rng = np.random.default_rng(7)
        x = rng.uniform(0.0, 1.0, 5000)
        y = rng.poisson(np.exp(1.2 + 0.8 * np.sin(2.0 * np.pi * x))).astype(np.float64)
        X = pd.DataFrame({"x": x})
    else:
        data = pd.read_csv(FIXTURES / f"{name}.csv")
        X, y = data[["x"]], data["y"].to_numpy(dtype=np.float64)
    model = SuperGLM(
        features={"x": CubicRegressionSpline(n_knots=NB_KNOTS[name])},
        family=NegativeBinomial(theta=1.0),
        weight_semantics=weight_semantics,
    )
    return model, X, y


def _book_case(name: str, weight_semantics: str):
    x, level, y = _book("zeros90" if name == "unpen" else name)
    if name == "unpen":
        # Categorical-only: the design on which estimate_p's joint ML path runs.
        band = np.char.add("B", np.minimum((x * 8.0).astype(np.int64), 7).astype(str))
        X = pd.DataFrame({"band": band, "level": level})
        features = {"band": Categorical(), "level": Categorical()}
    else:
        X = pd.DataFrame({"x": x, "level": level})
        features = {"x": Spline(n_knots=10), "level": Categorical()}
    model = SuperGLM(family=Tweedie(p=1.5), features=features, weight_semantics=weight_semantics)
    return model, X, y


def build_case(name: str, weight_semantics: str = "prior"):
    """A fresh unfitted model and its data ``(model, X, y)`` for one named case."""
    if name in RE_POWERS:
        return _random_effect_case(name, weight_semantics)
    if name in NB_KNOTS:
        return _negative_binomial_case(name, weight_semantics)
    return _book_case(name, weight_semantics)


# ---------------------------------------------------------------------------
# logpdf grid against the old routes and a 50-digit reference
# ---------------------------------------------------------------------------


def _logpdf_grid():
    """(y, mu, phi, p, w) rows: mu in {y/2, y, 2y}, and mu = 1 for a zero response."""
    for p, phi, y, w, factor in itertools.product(
        LOGPDF_POWERS, LOGPDF_PHIS, LOGPDF_RESPONSES, LOGPDF_WEIGHTS, (0.5, 1.0, 2.0)
    ):
        if y > 0.0 or factor == 1.0:
            yield y, factor * y if y > 0.0 else 1.0, phi, p, w


def _log_t(y: float, phi: float, p: float, w: float) -> float:
    a = (2.0 - p) / (p - 1.0)
    return a * (math.log(y) - math.log(p - 1.0)) - math.log(2.0 - p) + (a + 1.0) * math.log(w / phi)


def _series_mode(log_t: float, a: float) -> float:
    # Closed-form peak index of the Dunn-Smyth series (Dunn & Smyth 2005).
    return math.exp((log_t - a * math.log(a)) / (a + 1.0))


def _old_route(y: float, phi: float, p: float, w: float, saddlepoint: bool) -> str:
    """Which branch the old evaluator gives a single positive row."""
    from scipy.special import wright_bessel

    if saddlepoint:
        return "saddlepoint"
    log_t = _log_t(y, phi, p, w)
    a = (2.0 - p) / (p - 1.0)
    if log_t < math.log(OLD_T_ARG_LIMIT):
        with np.errstate(all="ignore"):
            wright = float(wright_bessel(a, a + 1.0, math.exp(log_t)))
        if math.isfinite(wright) and wright > 0.0:
            return "wright"
    return "bessel_p15" if p == 1.5 else "series"


def _reference_side(term, j: int, step: int, peak, floor):
    """Sum exp(term(j) - peak) from j in one direction until a term falls below floor."""
    import mpmath as mp

    total = mp.mpf(0)
    while j >= 1:
        q = term(j)
        total += mp.exp(q - peak)
        if q < floor:
            return total
        j += step
    return total


@functools.cache
def _reference_log_w(y: float, phi: float, p: float, w: float):
    """log W at 50 digits for the float64 inputs, summed outward from the peak term.

    This is the reference sum of the rebuild's series oracle: outward from the
    closed-form peak index until a term is 120 log-units below the peak.
    """
    import mpmath as mp

    p_, w_, phi_ = mp.mpf(p), mp.mpf(w), mp.mpf(phi)
    a = (2 - p_) / (p_ - 1)
    log_t = a * (mp.log(mp.mpf(y)) - mp.log(p_ - 1)) - mp.log(2 - p_) + (a + 1) * mp.log(w_ / phi_)

    def term(j: int):
        return j * log_t - mp.loggamma(j + 1) - mp.loggamma(a * j)

    mode = max(1, int(mp.floor(mp.exp((log_t - a * mp.log(a)) / (a + 1)))))
    peak = max(term(j) for j in range(max(1, mode - 2), mode + 3))
    total = _reference_side(term, mode, 1, peak, peak - 120) + _reference_side(
        term, mode - 1, -1, peak, peak - 120
    )
    return peak + mp.log(total)


def _exact_logpdf(y: float, mu: float, phi: float, p: float, w: float) -> float | None:
    """50-digit log-density of the float64 inputs, or None past the reference's mode cap."""
    import mpmath as mp

    y_, mu_, phi_, p_, w_ = (mp.mpf(v) for v in (y, mu, phi, p, w))
    if y == 0.0:
        return float(-w_ * mu_ ** (2 - p_) / ((2 - p_) * phi_))
    a = (2.0 - p) / (p - 1.0)
    if _series_mode(_log_t(y, phi, p, w), a) > REFERENCE_MAX_MODE:
        return None
    canonical = y_ * mu_ ** (1 - p_) / (1 - p_) - mu_ ** (2 - p_) / (2 - p_)
    return float(_reference_log_w(y, phi, p, w) - mp.log(y_) + canonical * w_ / phi_)


def _logpdf_row(y: float, mu: float, phi: float, p: float, w: float) -> dict:
    from superglm.profiling.tweedie import _evaluate_tweedie_density, _prepare_tweedie_density

    prepared = _prepare_tweedie_density(np.array([y]), np.array([mu]), p, weights=np.array([w]))
    evaluation = _evaluate_tweedie_density(prepared, phi)
    saddlepoint = y > 0.0 and bool(evaluation.positive_saddlepoint_mask[0])
    return {
        "y": y,
        "mu": mu,
        "phi": phi,
        "p": p,
        "w": w,
        "logpdf": float(evaluation.logpdf[0]),
        "saddlepoint": saddlepoint,
        "route": "zero" if y == 0.0 else _old_route(y, phi, p, w, saddlepoint),
        "logpdf_exact": _exact_logpdf(y, mu, phi, p, w),
    }


def _route_errors(rows: list[dict]) -> dict[str, float]:
    errors: dict[str, float] = {}
    for row in rows:
        if row["route"] in ("zero", "saddlepoint") or row["logpdf_exact"] is None:
            continue
        error = abs(row["logpdf"] - row["logpdf_exact"]) / max(1.0, abs(row["logpdf"]))
        errors[row["route"]] = max(errors.get(row["route"], 0.0), error)
    return errors


def characterise_logpdf() -> dict:
    import mpmath as mp

    mp.mp.dps = REFERENCE_DIGITS
    rows, refused = [], []
    for y, mu, phi, p, w in _logpdf_grid():
        try:
            rows.append(_logpdf_row(y, mu, phi, p, w))
        except (ValueError, FloatingPointError) as exc:
            refused.append({"y": y, "mu": mu, "phi": phi, "p": p, "w": w, "error": repr(exc)})
    by_route = _route_errors(rows)
    excluded = [r for r in rows if r["y"] > 0.0 and r["logpdf_exact"] is None]
    return {
        "old_route_max_rel_error": max(by_route.values()),
        "old_route_max_rel_error_by_route": by_route,
        "reference_excluded": {
            "rows": len(excluded),
            "saddlepoint_rows": sum(r["saddlepoint"] for r in excluded),
            "max_mode": REFERENCE_MAX_MODE,
        },
        "logpdf": rows,
        "logpdf_refused": refused,
    }


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def _recorded_warnings(caught, result_warnings=()) -> list[str]:
    return sorted({str(item.message) for item in caught} | set(result_warnings))


def _ci_slopes(result, details) -> list[float | None]:
    def slope(endpoint) -> float | None:
        if endpoint.status != "root_found":
            return None
        return endpoint.lr_statistic / (result._ll_scale * abs(endpoint.value - result.p_hat))

    return [slope(details.lower), slope(details.upper)]


def _interval(result) -> dict:
    """The 95% profile interval, or the error the pre-rebuild interval raises instead."""
    try:
        details = result.ci_details(0.05)
    except RuntimeError as exc:
        return {"ci95": None, "ci95_status": None, "ci95_slopes": None, "ci95_error": str(exc)}
    return {
        "ci95": [float(v) for v in details.interval],
        "ci95_status": [details.lower.status, details.upper.status],
        "ci95_slopes": _ci_slopes(result, details),
    }


def _published_phi_terms(y, mu, weights, p: float, phi: float) -> tuple[float, float]:
    """(d2 nll / d(log phi)^2, mean max(1, |logpdf|)) at the published fit."""

    def mean_nll(log_phi: float) -> float:
        return -float(np.mean(tweedie_logpdf(y, mu, math.exp(log_phi), p, weights=weights)))

    step, centre = 1e-2, math.log(phi)
    curvature = (
        mean_nll(centre + step) - 2.0 * mean_nll(centre) + mean_nll(centre - step)
    ) / step**2
    logpdf = tweedie_logpdf(y, mu, phi, p, weights=weights)
    return curvature, float(np.mean(np.maximum(1.0, np.abs(logpdf))))


def characterise_estimate_p(
    case: str, model, X, y, fit_mode: str, *, sample_weight=None, offset=None
) -> dict:
    """One ``estimate_p`` with its 95% interval, as a fixture row."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.estimate_p(X, y, sample_weight, offset, fit_mode=fit_mode)
        interval = _interval(result)
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(model.predict(X, offset=offset), dtype=np.float64)
    curvature, logpdf_scale = _published_phi_terms(
        y, mu, sample_weight, float(result.p_hat), float(result.phi_hat)
    )
    return {
        "case": case,
        "fit_mode": fit_mode,
        "p_hat": float(result.p_hat),
        "phi_hat": float(result.phi_hat),
        "nll": float(result.nll),
        "search_nll": float(result.nll if result.search_nll is None else result.search_nll),
        **interval,
        "n_evaluations": len(result.search_trace),
        "method": str(result.method),
        "converged": bool(result.converged),
        "phi_log_curvature": curvature,
        "mean_logpdf_scale": logpdf_scale,
        "n": int(y.size),
        "n_positive": int(np.count_nonzero(y > 0.0)),
        "warnings": _recorded_warnings(caught, result.warnings),
    }


def characterise_reml_phi(case: str) -> dict:
    model, X, y = build_case(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y)
    return {"case": case, "p": float(model._distribution.p), "phi": float(model.result.phi)}


def characterise_estimate_theta(case: str, fit_mode: str) -> dict:
    model, X, y = build_case(case)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.estimate_theta(X, y, fit_mode=fit_mode)
        ci = result.ci(0.05)
    return {
        "case": case,
        "fit_mode": fit_mode,
        "theta_hat": float(result.theta_hat),
        "nll": float(result.nll),
        "ci95": [float(v) for v in ci],
        "converged": bool(result.converged),
        "warnings": _recorded_warnings(caught),
    }


def _estimate_p_case(case: str, fit_mode: str) -> dict:
    model, X, y = build_case(case)
    return characterise_estimate_p(case, model, X, y, fit_mode)


def _collect(calls) -> tuple[list[dict], list[dict]]:
    """Rows from ``(case, context, thunk)`` calls; a raised error is recorded as a refusal."""
    rows, refused = [], []
    for case, context, thunk in calls:
        try:
            rows.append(thunk())
        except Exception as exc:  # the refusal itself is the characterised output
            refused.append({"case": case, **context, "error": f"{type(exc).__name__}: {exc}"})
    return rows, refused


def characterise_fits() -> dict:
    reml_phi, reml_refused = _collect(
        (case, {"p": p}, functools.partial(characterise_reml_phi, case))
        for case, p in RE_POWERS.items()
    )
    estimate_p, estimate_p_refused = _collect(
        (case, {"fit_mode": mode}, functools.partial(_estimate_p_case, case, mode))
        for case, mode in itertools.product(TWEEDIE_CASES, FIT_MODES)
    )
    estimate_theta, estimate_theta_refused = _collect(
        (case, {"fit_mode": mode}, functools.partial(characterise_estimate_theta, case, mode))
        for case, mode in itertools.product(NB_KNOTS, FIT_MODES)
    )
    return {
        "reml_phi": reml_phi,
        "reml_phi_refused": reml_refused,
        "reml_phi_books": [characterise_reml_phi(case) for case in BOOK_SHAPES],
        "estimate_p": estimate_p,
        "estimate_p_refused": estimate_p_refused,
        "estimate_theta": estimate_theta,
        "estimate_theta_refused": estimate_theta_refused,
    }


# ---------------------------------------------------------------------------
# Stage 0 (a): series work per row
# ---------------------------------------------------------------------------


def series_work(y, mu, weights, p: float, edf: float, dataset: str) -> list[dict]:
    """Series work for the positive rows at phi around the Pearson estimate at ``mu``."""
    from superglm._tweedie_profile_kernel import series_moments

    y, mu, weights = (np.asarray(v, dtype=np.float64) for v in (y, mu, weights))
    pearson = float(np.sum(weights * (y - mu) ** 2 / mu**p) / (y.size - edf))
    positive = y > 0.0
    a = (2.0 - p) / (p - 1.0)
    log_t_unit_phi = (
        a * (np.log(y[positive]) - math.log(p - 1.0))
        - math.log(2.0 - p)
        + (a + 1.0) * np.log(weights[positive])
    )
    rows = []
    for multiplier in SCAN_PHI_MULTIPLIERS:
        phi = pearson * multiplier
        log_t = log_t_unit_phi - (a + 1.0) * math.log(phi)
        mode = np.exp((log_t - a * math.log(a)) / (a + 1.0))
        terms = 2.0 * np.sqrt(2.0 * SERIES_LOG_CUTOFF * mode / (a + 1.0))
        ok = series_moments(log_t, a, max_terms=SERIES_MAX_ROW_TERMS, max_total_terms=2**62)[0]
        rows.append(
            {
                "dataset": dataset,
                "p": p,
                "phi_multiplier": multiplier,
                "phi": phi,
                "n_positive": int(log_t.size),
                "max_mode": float(mode.max()),
                "max_terms": float(terms.max()),
                "rows_over_1e5_terms": int(np.count_nonzero(terms > 1e5)),
                "rows_over_1e6_terms": int(np.count_nonzero(terms > 1e6)),
                "rows_mode_unsafe": int(np.count_nonzero(mode > SERIES_MAX_SAFE_MODE)),
                "rows_refused_at_1e6_terms": int(np.count_nonzero(~ok)),
            }
        )
    return rows


def scan_fitted(model, X, y, dataset: str, *, sample_weight=None, offset=None) -> list[dict]:
    """``series_work`` at every scan power, the mean refitted at each power."""
    rows = []
    weights = np.ones(len(y)) if sample_weight is None else sample_weight
    for p in SCAN_POWERS:
        model.family = Tweedie(p=p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y, sample_weight=sample_weight, offset=offset)
        mu = model.predict(X, offset=offset)
        rows += series_work(y, mu, weights, p, float(model.result.effective_df), dataset)
    return rows


def scan_logpdf_grid() -> dict:
    """The logpdf grid's own rows at its own phi: largest series work."""
    rows = [(y, phi, p, w) for y, _, phi, p, w in _logpdf_grid() if y > 0.0]
    modes = [_series_mode(_log_t(y, phi, p, w), (2.0 - p) / (p - 1.0)) for y, phi, p, w in rows]
    terms = [
        2.0 * math.sqrt(2.0 * SERIES_LOG_CUTOFF * m / ((2.0 - p) / (p - 1.0) + 1.0))
        for m, (_, _, p, _) in zip(modes, rows, strict=True)
    ]
    return {
        "dataset": "logpdf_grid",
        "n_positive": len(rows),
        "max_mode": max(modes),
        "max_terms": max(terms),
        "rows_over_1e5_terms": sum(t > 1e5 for t in terms),
        "rows_over_1e6_terms": sum(t > 1e6 for t in terms),
    }


def _scan_model(case: str):
    model, X, y = build_case(case)
    if case in RE_POWERS:
        # fit() refuses RandomEffect; the unshrunk categorical mean serves a Pearson estimate.
        model = SuperGLM(features={"u": Categorical(), "r": Categorical()}, family=Tweedie(p=1.5))
    return model, X, y


def series_scan() -> dict:
    rows = []
    for case in TWEEDIE_CASES:
        rows += scan_fitted(*_scan_model(case), case)
    return {"cases": rows, "logpdf_grid": scan_logpdf_grid()}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPOSITORY, check=True, capture_output=True, text=True
    ).stdout.strip()


def provenance() -> dict:
    import numba
    import scipy

    master = _git("merge-base", "HEAD", "origin/master")
    if _git("diff", "--name-only", master, "HEAD", "--", "src"):
        raise RuntimeError(f"src differs from master {master}; run on the pre-rebuild code")
    return {
        "master_sha": master,
        "head_sha": _git("rev-parse", "HEAD"),
        "script": "benchmarks/tweedie_nb_characterisation.py",
        "generated": datetime.date.today().isoformat(),
        "reference_digits": REFERENCE_DIGITS,
        # BLAS reduction order follows the pool sizes; the suite runs pinned to 1.
        "thread_pools": {name: os.environ.get(name) for name in THREAD_POOLS},
        "versions": {
            "superglm": superglm.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--series-scan", action="store_true", help="write measurement (a) instead")
    args = parser.parse_args()
    superglm.warmup()
    payload = {"provenance": provenance()}
    payload |= series_scan() if args.series_scan else characterise_logpdf() | characterise_fits()
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
