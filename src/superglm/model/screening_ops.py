"""PSST interaction screening over a fitted mains model.

PSST — Penalized Smooth Score Test — ranks every candidate ``ti(a, b)`` pair
by how much of the fitted model's leftover working signal the pair's actual
tensor-product smooth could absorb at a fixed screening complexity.  One
fused O(n) cell pass per pair, no refits; the confirmatory ``fit_reml``
refit of the top-ranked pairs is the gate.  Ranking-only: the statistic is
not a calibrated p-value and must not be reported as one.

The statistic is reported on the ``T / phi`` scale, with ``phi`` the mains
fit's Pearson dispersion estimate: under the null ``E[T] = phi * edf0``, so
without this scaling the ``edf0`` noise floor is only honest for
unit-dispersion families and a dispersed Gaussian null would swamp the scan.
The Pearson denominator is ``n - edf`` per the sample_weight=exposure
contract (Var(y) = phi*V(mu)/w), deliberately NOT the frequency-weight
``sum(w) - edf`` convention of ``solvers.dispersion`` — measured on
known-dispersion exposure nulls, only ``n - edf`` is calibrated.

Per-feature work (unique support, codes, marginal basis menus, marginal
penalties) is cached across the sweep, so an all-pairs screen over P spline
features builds P marginals, not P*(P-1).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from superglm._frame import as_eager_frame
from superglm.distributions import _VARIANCE_FLOOR
from superglm.features.spline import _SplineBase
from superglm.screening import (
    pair_cell_moments,
    pair_score_curvature,
    penalized_score_statistic,
    working_score,
)
from superglm.screening._overlap import pair_overlap_moments, tensor_penalty

_RESULT_COLUMNS = [
    "feature_a",
    "feature_b",
    "statistic",
    "z",
    "edf0",
    "lambda0",
    "n_cells",
    "approx",
]

# The V-assembly einsum carries (n_a, k_b, k_b) and (n_b, k_a, k_a)
# intermediates that max_cells alone does not bound (a lopsided 1e6 x 5 pair
# passes the cell budget while the intermediate is 40x larger).  Budget them
# against a small multiple of max_cells; the marginal dimension is estimated
# from the parent spec's k, which is a guard-grade bound, not an exact count.
_INTERMEDIATE_BUDGET_FACTOR = 4


def _quantile_binned(x, bins):
    """Replace x by per-bin mean representatives on an empirical quantile grid.

    Screening-only lossy compression for continuous covariates whose joint
    support would blow the cell budget: the spline basis is evaluated at the
    within-bin mean, so the probe sees the same marginal geometry at <= bins
    support points.  Never applied when the pair fits the budget exactly.
    """
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    codes = np.searchsorted(edges, x, side="right")
    _, codes = np.unique(codes, return_inverse=True)
    reps = np.bincount(codes, weights=x) / np.bincount(codes)
    return reps[codes]


def _validated_budgets(edf0) -> tuple[float, ...]:
    budgets = np.atleast_1d(np.asarray(edf0, dtype=np.float64)).ravel()
    if budgets.size == 0:
        raise ValueError("edf0 must be a positive budget or a non-empty sequence of budgets")
    if not np.all(np.isfinite(budgets)) or np.any(budgets <= 0.0):
        raise ValueError(f"edf0 budgets must be finite and positive, got {edf0!r}")
    return tuple(float(b) for b in budgets)


def _validated_pairs(candidates, spline_names):
    if candidates is None:
        return list(combinations(spline_names, 2))
    valid = set(spline_names)
    pairs = []
    for raw in candidates:
        pair = tuple(raw)
        if len(pair) != 2 or pair[0] == pair[1] or not valid.issuperset(pair):
            raise ValueError(
                "candidates entries must pair two distinct fitted spline features; "
                f"got {raw!r} (screenable features: {spline_names})"
            )
        pairs.append(pair)
    return pairs


def screen_interactions(
    model,
    X,
    y,
    sample_weight=None,
    *,
    offset=None,
    candidates=None,
    edf0=(2.0, 4.0, 8.0, 16.0),
    max_cells: int = 5_000_000,
    screen_bins: int = 256,
) -> pd.DataFrame:
    """Rank candidate spline-pair interactions of a fitted model by PSST.

    ``edf0`` is the probe bandwidth: a smooth surface is detected best by a
    small budget, a high-frequency one only by a budget at least as complex
    as its shape (measured: a sin x sin signal is invisible at edf0<=4).  The
    default is therefore a LADDER — each pair is evaluated at every budget,
    each T is normalized against its own noise floor,
    ``z = (T - edf0) / sqrt(2 * edf0)``, and the pair is ranked by its best
    normalized score, a scan statistic over bandwidths.  Pass a single float
    to probe one bandwidth.  The expensive per-pair work (cells, menus,
    profiling) happens once; the ladder re-solves a small system per rung.

    ``offset`` and ``sample_weight`` both default to the values the model
    was fitted with, so the screen linearizes at the fitted likelihood; pass
    either explicitly only to override it.  ``T`` is scaled by the fit's
    Pearson dispersion estimate before normalization (see module docstring).

    Returns a frame sorted by ``z`` (descending) with one row per screened
    pair: ``feature_a, feature_b, statistic, z, edf0, lambda0, n_cells,
    approx``, where ``statistic``/``edf0``/``lambda0`` describe the winning
    rung.  ``statistic`` is therefore NOT comparable across rows (rungs
    differ) — rank by ``z``.  At a clamped rung ``edf0`` holds the achieved
    value and ``lambda0`` is a bracket edge, not an interpretable smoothing
    parameter.

    Pairs whose support exceeds the cell or intermediate budgets fall back
    to quantile binning, largest margin first: a margin with more than
    ``screen_bins`` unique values is compressed to ``screen_bins``
    empirical-quantile bins (basis evaluated at within-bin means) and the
    row is flagged ``approx=True``.  Pairs within budget are always computed
    exactly and flagged ``approx=False`` — the fallback never touches them.
    A pair still over budget after binning is skipped with a NaN row
    (``n_cells`` reports the grid that was attempted), as is a pair whose
    statistic degenerates.
    """
    if getattr(model, "_result", None) is None:
        raise RuntimeError("screen_interactions requires a fitted model; call fit_reml first")
    from superglm.features.interaction import TensorInteraction

    frame = as_eager_frame(X)
    n_rows = len(frame)
    y = np.asarray(y, dtype=np.float64)
    if sample_weight is None:
        sample_weight = getattr(model, "_fit_weights", None)
    weights = (
        np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )
    if y.shape != (n_rows,):
        raise ValueError(
            f"y must be one-dimensional with one entry per row of X; got shape {y.shape} "
            f"for {n_rows} rows"
        )
    if weights.shape != (n_rows,):
        raise ValueError(
            f"sample_weight must be one-dimensional with one entry per row of X; got shape "
            f"{weights.shape} for {n_rows} rows"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("screen_interactions requires finite y")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("sample_weight must be finite and non-negative with a positive sum")
    budgets = _validated_budgets(edf0)
    if int(screen_bins) < 2:
        raise ValueError(f"screen_bins must be at least 2, got {screen_bins!r}")
    screen_bins = int(screen_bins)

    spline_names = [
        name for name in model._feature_order if isinstance(model._specs.get(name), _SplineBase)
    ]
    pairs = _validated_pairs(candidates, spline_names)
    selected = sorted(
        {name for pair in pairs for name in pair if getattr(model._specs[name], "select", False)}
    )
    if selected:
        raise ValueError(
            f"screen_interactions requires single-penalty parent smooths, but {selected} were "
            "fitted with select=True; ti() terms cannot be built on such parents either — "
            "refit the mains without select"
        )

    raw_x: dict[str, np.ndarray] = {}

    def _raw(name):
        if name not in raw_x:
            x = frame.column_array(name, dtype=np.float64)
            if not np.all(np.isfinite(x)):
                raise ValueError(
                    f"screen_interactions requires finite covariates; {name!r} contains "
                    "non-finite values"
                )
            raw_x[name] = x
        return raw_x[name]

    for name in sorted({name for pair in pairs for name in pair}):
        _raw(name)

    distribution, link = model._distribution, model._link
    if offset is None:
        offset = getattr(model, "_fit_offset", None)
    mu = np.asarray(model.predict(X, offset), dtype=np.float64)
    eta = np.asarray(link.link(mu), dtype=np.float64)
    score = working_score(y, mu, eta, weights, distribution, link)
    dmu_deta = link.deriv_inverse(eta)
    var_mu = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
    working_weights = weights * dmu_deta**2 / var_mu

    # Residual d.f. is n - edf, NOT the frequency-weight form sum(w) - edf of
    # pearson_residual_degrees_of_freedom: the screen follows the library's
    # protected sample_weight=exposure contract, under which Var(y) =
    # phi*V(mu)/w and E[Pearson] = n*phi.  Measured on known-phi exposure
    # nulls, n - edf tracks truth exactly while sum(w) - edf misreads phi 1.0
    # as ~1.55 and demotes refit-confirmed structure (see the plan log,
    # dispersion-denominator entry).
    edf_mains = float(getattr(model._result, "effective_df", float("nan")))
    denom = y.size - edf_mains if np.isfinite(edf_mains) else float(y.size)
    phi_hat = float(np.sum(weights * (y - mu) ** 2 / var_mu)) / max(denom, 1.0)
    phi_hat = max(phi_hat, float(np.finfo(np.float64).tiny))

    support_cache: dict[tuple[str, bool], dict] = {}
    marginal_cache: dict[tuple[str, bool], tuple[np.ndarray, np.ndarray]] = {}

    def _support(name, binned):
        key = (name, binned)
        if key not in support_cache:
            x = _raw(name)
            if binned:
                x = _quantile_binned(x, screen_bins)
            _, first, codes = np.unique(x, return_index=True, return_inverse=True)
            support_cache[key] = {"x": x, "first": first, "codes": codes, "n": len(first)}
        return support_cache[key]

    def _marginals(feat_a, bin_a, feat_b, bin_b):
        key_a, key_b = (feat_a, bin_a), (feat_b, bin_b)
        if key_a not in marginal_cache or key_b not in marginal_cache:
            spec = TensorInteraction(feat_a, feat_b)
            sa, sb = _support(feat_a, bin_a), _support(feat_b, bin_b)
            B1, B2, S1, S2 = spec._prepare_centered_marginals(sa["x"], sb["x"], model._specs)
            if key_a not in marginal_cache:
                marginal_cache[key_a] = (
                    np.asarray(B1[sa["first"]].todense(), dtype=np.float64),
                    S1,
                )
            if key_b not in marginal_cache:
                marginal_cache[key_b] = (
                    np.asarray(B2[sb["first"]].todense(), dtype=np.float64),
                    S2,
                )
        return marginal_cache[key_a], marginal_cache[key_b]

    def _k_est(name):
        return max(int(getattr(model._specs[name], "k", 10)) - 1, 1)

    def _within_budget(n_a, n_b, k_a, k_b):
        cells_ok = n_a * n_b <= max_cells
        inter_ok = (
            n_a * k_b * k_b + n_b * k_a * k_a <= _INTERMEDIATE_BUDGET_FACTOR * max_cells
        )
        return cells_ok and inter_ok

    rows = []
    for feat_a, feat_b in pairs:
        k_a, k_b = _k_est(feat_a), _k_est(feat_b)
        bin_flag = {feat_a: False, feat_b: False}
        while True:
            n_a = _support(feat_a, bin_flag[feat_a])["n"]
            n_b = _support(feat_b, bin_flag[feat_b])["n"]
            if _within_budget(n_a, n_b, k_a, k_b):
                break
            # bin the largest not-yet-binned margin that binning can shrink
            binnable = sorted(
                (
                    (n, name)
                    for n, name in ((n_a, feat_a), (n_b, feat_b))
                    if not bin_flag[name] and n > screen_bins
                ),
                reverse=True,
            )
            if not binnable:
                break
            bin_flag[binnable[0][1]] = True
        sa = _support(feat_a, bin_flag[feat_a])
        sb = _support(feat_b, bin_flag[feat_b])
        n_a, n_b = sa["n"], sb["n"]
        approx = bin_flag[feat_a] or bin_flag[feat_b]
        if not _within_budget(n_a, n_b, k_a, k_b):
            rows.append((feat_a, feat_b, np.nan, np.nan, np.nan, np.nan, n_a * n_b, False))
            continue

        (menu_a, S1), (menu_b, S2) = _marginals(
            feat_a, bin_flag[feat_a], feat_b, bin_flag[feat_b]
        )
        S_cell, W_cell = pair_cell_moments(
            sa["codes"], sb["codes"], n_a, n_b, score, working_weights, max_cells=max_cells
        )
        U, V = pair_score_curvature(menu_a, menu_b, S_cell, W_cell)
        M, C, u_m = pair_overlap_moments(menu_a, menu_b, S_cell, W_cell)
        S_ti = tensor_penalty(S1, S2)
        best_z, best = -np.inf, None
        for budget in budgets:
            result = penalized_score_statistic(U, V, C, M, S_ti, edf0=budget, U_nuisance=u_m)
            statistic = result.statistic / phi_hat
            z = (statistic - result.edf0) / np.sqrt(2.0 * result.edf0)
            if z > best_z:
                best_z, best = z, (statistic, result.edf0, result.lambda0)
        if best is None:
            rows.append((feat_a, feat_b, np.nan, np.nan, np.nan, np.nan, n_a * n_b, approx))
        else:
            rows.append((feat_a, feat_b, best[0], best_z, best[1], best[2], n_a * n_b, approx))

    table = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    return table.sort_values("z", ascending=False, ignore_index=True)
