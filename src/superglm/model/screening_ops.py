"""PSST interaction screening over a fitted mains model.

PSST — Penalized Smooth Score Test — ranks every candidate ``ti(a, b)`` pair
by how much of the fitted model's leftover working signal the pair's actual
tensor-product smooth could absorb at a fixed screening complexity.  One
fused O(n) cell pass per pair, no refits; the confirmatory ``fit_reml``
refit of the top-ranked pairs is the gate.  Ranking-only: the statistic is
not a calibrated p-value and must not be reported as one.
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


def screen_interactions(
    model,
    X,
    y,
    sample_weight=None,
    *,
    candidates=None,
    edf0=(2.0, 4.0, 8.0, 16.0),
    max_cells: int = 5_000_000,
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

    Returns a frame sorted by ``z`` (descending) with one row per screened
    pair: ``feature_a, feature_b, statistic, z, edf0, lambda0, n_cells``,
    where ``statistic``/``edf0``/``lambda0`` describe the winning rung.
    Pairs whose joint cell grid exceeds ``max_cells`` are skipped with NaN
    rather than silently binned.
    """
    if getattr(model, "_result", None) is None:
        raise RuntimeError("screen_interactions requires a fitted model; call fit_reml first")
    from superglm.features.interaction import TensorInteraction

    frame = as_eager_frame(X)
    y = np.asarray(y, dtype=np.float64)
    weights = (
        np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )

    distribution, link = model._distribution, model._link
    mu = np.asarray(model.predict(X), dtype=np.float64)
    eta = np.asarray(link.link(mu), dtype=np.float64)
    score = working_score(y, mu, eta, weights, distribution, link)
    dmu_deta = link.deriv_inverse(eta)
    working_weights = weights * dmu_deta**2 / np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)

    spline_names = [
        name for name in model._feature_order if isinstance(model._specs.get(name), _SplineBase)
    ]
    pairs = (
        [tuple(pair) for pair in candidates]
        if candidates is not None
        else list(combinations(spline_names, 2))
    )

    rows = []
    for feat_a, feat_b in pairs:
        x_a = frame.column_array(feat_a, dtype=np.float64)
        x_b = frame.column_array(feat_b, dtype=np.float64)
        uniq_a, first_a, codes_a = np.unique(x_a, return_index=True, return_inverse=True)
        uniq_b, first_b, codes_b = np.unique(x_b, return_index=True, return_inverse=True)
        n_a, n_b = len(uniq_a), len(uniq_b)
        if n_a * n_b > max_cells:
            rows.append((feat_a, feat_b, np.nan, np.nan, np.nan, np.nan, n_a * n_b))
            continue

        spec = TensorInteraction(feat_a, feat_b)
        B1, B2, S1, S2 = spec._prepare_centered_marginals(x_a, x_b, model._specs)
        menu_a = np.asarray(B1[first_a].todense(), dtype=np.float64)
        menu_b = np.asarray(B2[first_b].todense(), dtype=np.float64)

        S_cell, W_cell = pair_cell_moments(
            codes_a, codes_b, n_a, n_b, score, working_weights, max_cells=max_cells
        )
        U, V = pair_score_curvature(menu_a, menu_b, S_cell, W_cell)
        M, C, u_m = pair_overlap_moments(menu_a, menu_b, S_cell, W_cell)
        budgets = (edf0,) if np.isscalar(edf0) else tuple(edf0)
        S_ti = tensor_penalty(S1, S2)
        best_z, best = -np.inf, None
        for budget in budgets:
            result = penalized_score_statistic(U, V, C, M, S_ti, edf0=float(budget), U_nuisance=u_m)
            z = (result.statistic - result.edf0) / np.sqrt(2.0 * result.edf0)
            if z > best_z:
                best_z, best = z, result
        rows.append((feat_a, feat_b, best.statistic, best_z, best.edf0, best.lambda0, n_a * n_b))

    table = pd.DataFrame(
        rows,
        columns=["feature_a", "feature_b", "statistic", "z", "edf0", "lambda0", "n_cells"],
    )
    return table.sort_values("z", ascending=False, ignore_index=True)
