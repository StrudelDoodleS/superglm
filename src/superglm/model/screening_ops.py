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
    edf0: float = 4.0,
    max_cells: int = 5_000_000,
) -> pd.DataFrame:
    """Rank candidate spline-pair interactions of a fitted model by PSST.

    Returns a frame sorted by ``statistic`` (descending) with one row per
    screened pair: ``feature_a, feature_b, statistic, edf0, lambda0,
    n_cells``.  Pairs whose joint cell grid exceeds ``max_cells`` are
    skipped with ``statistic = NaN`` rather than silently binned.
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
            rows.append((feat_a, feat_b, np.nan, np.nan, np.nan, n_a * n_b))
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
        result = penalized_score_statistic(
            U, V, C, M, tensor_penalty(S1, S2), edf0=edf0, U_nuisance=u_m
        )
        rows.append((feat_a, feat_b, result.statistic, result.edf0, result.lambda0, n_a * n_b))

    table = pd.DataFrame(
        rows,
        columns=["feature_a", "feature_b", "statistic", "edf0", "lambda0", "n_cells"],
    )
    return table.sort_values("statistic", ascending=False, ignore_index=True)
