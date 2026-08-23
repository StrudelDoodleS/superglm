"""Spline redundancy diagnostics.

# Internal submodules: import siblings directly, not through this __init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from superglm._frame import FrameLike, as_eager_frame


def _redundancy_geometry_weights(model, sample_weight, n_rows: int) -> NDArray:
    """Validate weights and return the row mass used by spline geometry."""
    from superglm.distributions import Tweedie
    from superglm.solvers.dispersion import PRIOR_WEIGHTS, model_weight_semantics

    if n_rows == 0:
        raise ValueError("X must be non-empty")
    weight_semantics = model_weight_semantics(model)
    if sample_weight is None:
        weights = np.ones(n_rows, dtype=np.float64)
    elif isinstance(model._distribution, Tweedie) and weight_semantics == PRIOR_WEIGHTS:
        from superglm._utils import _validate_strict_prior_weights

        weights = _validate_strict_prior_weights(sample_weight, n_rows)
    else:
        from superglm.model.input_validation import _finite_vector

        # The weights are validated under either contract; only what counts as
        # admissible differs, and only for Tweedie above.  What the contract
        # decides here is the row mass the geometry follows, below.
        weights = _finite_vector("sample_weight", sample_weight, n_rows)
        if np.any(weights < 0.0):
            raise ValueError("sample_weight must be nonnegative")
        if not np.any(weights > 0.0):
            raise ValueError("sample_weight must not be all zero")

    if weight_semantics == PRIOR_WEIGHTS:
        # Prior weights are precision, not replicated design rows, so learned
        # geometry stays a function of the rows themselves -- excluding a row
        # carrying no weight, which under this contract was not observed.
        weights = (weights > 0.0).astype(np.float64)

    total = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(total):
        raise ValueError("sample_weight must have a finite sum")
    return weights


def _weighted_correlation(x: NDArray, y: NDArray, weights: NDArray) -> float:
    """Correlation equivalent to replicating rows by integer weights."""
    positive = weights > 0.0
    x_active = np.asarray(x[positive], dtype=np.float64)
    y_active = np.asarray(y[positive], dtype=np.float64)
    mass = np.asarray(weights[positive], dtype=np.float64)
    mass /= np.sum(mass, dtype=np.float64)
    x_centered = x_active - float(np.sum(mass * x_active))
    y_centered = y_active - float(np.sum(mass * y_active))
    var_x = float(np.sum(mass * x_centered**2))
    var_y = float(np.sum(mass * y_centered**2))
    if np.sqrt(var_x) <= 1e-12 or np.sqrt(var_y) <= 1e-12:
        return 0.0
    correlation = float(np.sum(mass * x_centered * y_centered) / np.sqrt(var_x * var_y))
    return float(np.clip(correlation, -1.0, 1.0))


@dataclass
class SplineRedundancyReport:
    """Redundancy diagnostics for one spline feature.

    ``support_mass``, ``adjacent_basis_corr``, and ``effective_rank`` use
    replication mass under the frequency contract. Integer weights therefore
    match literal row replication. Under the prior contract the weights are
    validated but these geometric summaries use the physical rows rather than
    treating a precision as a row frequency; a Tweedie fit additionally
    requires them finite and strictly positive.
    """

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
    X: FrameLike,
    sample_weight: NDArray | None = None,
) -> dict[str, SplineRedundancyReport]:
    """Spline redundancy diagnostics for all spline features.

    Diagnostic-only. No auto-pruning. Interpretation: "try lower k and refit".

    Under the frequency contract ``sample_weight`` is replication mass, so the support
    fractions, correlations, and weighted-basis singular values match literal
    integer row replication. Under the prior contract ``sample_weight`` is a
    precision; it is validated but does not alter the physical-row spline
    geometry summarized here.
    """
    from superglm.features.spline import _SplineBase

    if model._result is None:
        raise RuntimeError("Model must be fitted.")

    frame = as_eager_frame(X)
    geometry_weight = _redundancy_geometry_weights(model, sample_weight, len(frame))
    geometry_total = float(np.sum(geometry_weight, dtype=np.float64))
    results = {}

    for name, spec in model._specs.items():
        if not isinstance(spec, _SplineBase):
            continue

        x_col = frame.column_array(name, dtype=np.float64)

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
            support_mass[i] = float(
                np.sum(geometry_weight[(x_col >= lo) & (x_col <= hi)], dtype=np.float64)
                / geometry_total
            )

        # Adjacent basis correlation
        B = spec.transform(x_col)
        n_cols = B.shape[1]
        adj_corr = np.zeros(max(n_cols - 1, 0))
        for j in range(n_cols - 1):
            c1, c2 = B[:, j], B[:, j + 1]
            adj_corr[j] = _weighted_correlation(c1, c2, geometry_weight)

        # Coefficient energy in penalized directions
        beta = model.result.beta
        groups = [g for g in model._groups if g.feature_name == name]
        beta_combined = np.concatenate([beta[g.sl] for g in groups])
        coef_energy = beta_combined**2

        # Effective rank via singular values of transformed basis
        active = geometry_weight > 0.0
        weighted_basis = B[active] * np.sqrt(geometry_weight[active, None])
        sv = np.linalg.svd(weighted_basis, compute_uv=False)
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
