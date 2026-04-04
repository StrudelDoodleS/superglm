"""Spline redundancy diagnostics.

# Internal submodules: import siblings directly, not through this __init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class SplineRedundancyReport:
    """Redundancy diagnostics for one spline feature."""

    feature_name: str
    n_knots: int
    knot_locations: NDArray
    knot_spacing: NDArray
    support_mass: NDArray  # fraction of data near each knot
    adjacent_basis_corr: NDArray
    coef_energy_penalized: NDArray
    effective_rank: float
    small_singular_values: NDArray = field(default_factory=lambda: np.array([]))


def spline_redundancy(
    model,
    X: pd.DataFrame,
    sample_weight: NDArray | None = None,
) -> dict[str, SplineRedundancyReport]:
    """Spline redundancy diagnostics for all spline features.

    Diagnostic-only. No auto-pruning. Interpretation: "try lower k and refit".
    """
    from superglm.features.spline import _SplineBase

    if model._result is None:
        raise RuntimeError("Model must be fitted.")

    results = {}

    for name, spec in model._specs.items():
        if not isinstance(spec, _SplineBase):
            continue

        x_col = np.asarray(X[name], dtype=np.float64)

        # Knot info
        interior_knots = spec.fitted_knots
        if interior_knots is None or len(interior_knots) == 0:
            continue

        knot_spacing = np.diff(interior_knots)

        # Support mass: fraction of data near each knot
        n_knots = len(interior_knots)
        support_mass = np.zeros(n_knots)
        for i, kn in enumerate(interior_knots):
            # Count data within half a knot spacing on each side
            if i == 0:
                lo = spec._lo
            else:
                lo = 0.5 * (interior_knots[i - 1] + kn)
            if i == n_knots - 1:
                hi = spec._hi
            else:
                hi = 0.5 * (kn + interior_knots[i + 1])
            support_mass[i] = np.sum((x_col >= lo) & (x_col <= hi)) / len(x_col)

        # Adjacent basis correlation
        B = spec.transform(x_col)
        n_cols = B.shape[1]
        adj_corr = np.zeros(max(n_cols - 1, 0))
        for j in range(n_cols - 1):
            c1, c2 = B[:, j], B[:, j + 1]
            s1, s2 = np.std(c1), np.std(c2)
            if s1 > 1e-12 and s2 > 1e-12:
                adj_corr[j] = float(np.corrcoef(c1, c2)[0, 1])

        # Coefficient energy in penalized directions
        beta = model.result.beta
        groups = [g for g in model._groups if g.feature_name == name]
        beta_combined = np.concatenate([beta[g.sl] for g in groups])
        coef_energy = beta_combined**2

        # Effective rank via singular values of transformed basis
        sv = np.linalg.svd(B, compute_uv=False)
        sv_norm = sv / sv[0] if sv[0] > 1e-12 else sv
        effective_rank = float(np.sum(sv_norm > 1e-4))
        small_sv = sv_norm[sv_norm < 0.01]

        results[name] = SplineRedundancyReport(
            feature_name=name,
            n_knots=n_knots,
            knot_locations=interior_knots,
            knot_spacing=knot_spacing,
            support_mass=support_mass,
            adjacent_basis_corr=adj_corr,
            coef_energy_penalized=coef_energy,
            effective_rank=effective_rank,
            small_singular_values=small_sv,
        )

    return results
