"""Subclass-specific spline helper operations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.features import _spline_constraints


def assemble_open_knot_vector(spec: Any, interior: NDArray) -> None:
    """Build the padded open knot vector used by unconstrained B-splines."""
    x_range = spec._hi - spec._lo
    lo_effective = spec._lo - 0.001 * x_range
    hi_effective = spec._hi + 0.001 * x_range

    inner = np.concatenate([[lo_effective], interior, [hi_effective]])
    dx_lo = inner[1] - inner[0]
    dx_hi = inner[-1] - inner[-2]

    lower = lo_effective - dx_lo * np.arange(spec.degree, 0, -1)
    upper = hi_effective + dx_hi * np.arange(1, spec.degree + 1)

    spec._knots = np.concatenate([lower, inner, upper])
    spec._n_basis = len(spec._knots) - spec.degree - 1


def assemble_clamped_knot_vector(spec: Any, interior: NDArray) -> None:
    """Build the exact-boundary clamped knot vector used by cubic regression splines."""
    spec._knots = np.concatenate(
        [
            np.repeat(spec._lo, spec.degree + 1),
            interior,
            np.repeat(spec._hi, spec.degree + 1),
        ]
    )
    spec._n_basis = len(spec._knots) - spec.degree - 1


def build_scop_reparameterization(
    spec: Any,
    basis: NDArray,
    omega: NDArray,
):
    """Build the SCOP reparameterization for monotone P-splines."""
    del omega
    from superglm.solvers.scop import build_scop_reparam, build_scop_solver_reparam

    q = spec._n_basis
    reparam = build_scop_reparam(q, direction=spec.monotone)

    x_sigma = basis @ reparam.Sigma
    col_means = x_sigma[:, 1:].mean(axis=0)
    x_centered = x_sigma[:, 1:] - col_means

    spec._scop_Sigma = reparam.Sigma
    spec._scop_col_means = col_means

    solver_reparam = build_scop_solver_reparam(q, direction=spec.monotone)
    scop_penalty = solver_reparam.penalty_matrix()
    return x_centered, scop_penalty, solver_reparam


def build_monotone_constraints_raw(spec: Any):
    """Build first-difference monotone constraints on raw spline coefficients."""
    return _spline_constraints.build_monotone_difference_constraints(spec._n_basis, spec.monotone)
