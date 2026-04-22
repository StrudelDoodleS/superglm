"""Post-fit canonicalization for the public runtime model state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import clip_mu
from superglm.links import stabilize_eta
from superglm.solvers.pirls import PIRLSResult


@dataclass(frozen=True)
class CanonicalizationDiagnostics:
    """Diagnostics for the runtime canonicalization pass."""

    max_abs_eta_delta: float
    max_abs_mu_delta: float
    intercept_shift: float
    term_means_before: dict[str, float]
    term_means_after: dict[str, float]


def _is_spline_backed_spec(spec: Any) -> bool:
    """Whether this feature exposes spline-style runtime evaluation hooks."""
    return hasattr(spec, "_basis_matrix")


def _feature_group_indices(model, feature_name: str) -> tuple[int, ...]:
    """Return all fitted group indices belonging to one feature."""
    return tuple(i for i, group in enumerate(model._groups) if group.feature_name == feature_name)


def _group_column_means(group_matrix, n_rows: int) -> NDArray[np.float64]:
    """Compute unweighted training-row column means for one fitted block."""
    ones = np.ones(n_rows, dtype=np.float64)
    return np.asarray(group_matrix.rmatvec(ones), dtype=np.float64) / float(n_rows)


def _term_contribution(
    model, solver: PIRLSResult, group_indices: tuple[int, ...]
) -> NDArray[np.float64]:
    """Evaluate one fitted term on the training rows in solver space."""
    contribution = np.zeros(model._dm.n, dtype=np.float64)
    for idx in group_indices:
        group = model._groups[idx]
        contribution += np.asarray(
            model._dm.group_matrices[idx].matvec(solver.beta[group.sl]),
            dtype=np.float64,
        )
    return contribution


def _regular_spline_transform(model, spec: Any, group_idx: int) -> dict[str, Any]:
    """Center an SSP-style runtime block by rewriting its public transform."""
    group = model._groups[group_idx]
    group_matrix = model._dm.group_matrices[group_idx]
    column_means = _group_column_means(group_matrix, model._dm.n)
    r_inv = np.asarray(spec._R_inv, dtype=np.float64)
    centered = r_inv - np.ones((r_inv.shape[0], 1), dtype=np.float64) @ column_means[None, :]
    spec._R_inv = centered
    return {
        "mode": "blockwise",
        "group_indices": (group_idx,),
        "solver_slice": (group.start, group.end),
        "column_means": column_means,
    }


def _constrained_spline_transform(model, spec: Any, group_idx: int) -> dict[str, Any]:
    """Center a constrained runtime block by updating its affine offset."""
    group = model._groups[group_idx]
    group_matrix = model._dm.group_matrices[group_idx]
    column_means = _group_column_means(group_matrix, model._dm.n)
    spec._scop_col_means = np.asarray(spec._scop_col_means, dtype=np.float64) + column_means
    return {
        "mode": "affine",
        "group_indices": (group_idx,),
        "solver_slice": (group.start, group.end),
        "column_means": column_means,
    }


def _compile_runtime_state(model, solver: PIRLSResult) -> dict[str, Any]:
    """Compile term-level canonicalization state and diagnostics."""
    term_states: dict[str, dict[str, Any]] = {}
    term_means_before: dict[str, float] = {}
    term_means_after: dict[str, float] = {}
    total_shift = 0.0

    for feature_name in model._feature_order:
        spec = model._specs[feature_name]
        if not _is_spline_backed_spec(spec):
            continue

        group_indices = _feature_group_indices(model, feature_name)
        if not group_indices:
            continue

        contribution_before = _term_contribution(model, solver, group_indices)
        mean_before = float(np.mean(contribution_before))
        state: dict[str, Any] = {
            "mode": "skipped",
            "group_indices": group_indices,
            "solver_slice": None,
            "term_mean_before": mean_before,
            "term_mean_after": mean_before,
            "intercept_shift": 0.0,
        }

        if len(group_indices) == 1 and getattr(spec, "_scop_Sigma", None) is not None:
            state |= _constrained_spline_transform(model, spec, group_indices[0])
            state["term_mean_after"] = 0.0
            state["intercept_shift"] = mean_before
            total_shift += mean_before
        elif len(group_indices) == 1 and getattr(spec, "_R_inv", None) is not None:
            state |= _regular_spline_transform(model, spec, group_indices[0])
            state["term_mean_after"] = 0.0
            state["intercept_shift"] = mean_before
            total_shift += mean_before

        term_states[feature_name] = state
        term_means_before[feature_name] = mean_before
        term_means_after[feature_name] = float(state["term_mean_after"])

    eta_before = model._dm.matvec(solver.beta) + float(solver.intercept)
    if model._fit_offset is not None:
        eta_before = eta_before + model._fit_offset
    eta_before = stabilize_eta(eta_before, model._link)
    mu_before = clip_mu(model._link.inverse(eta_before), model._distribution)

    eta_after = eta_before.copy()
    mu_after = clip_mu(model._link.inverse(eta_after), model._distribution)

    diagnostics = CanonicalizationDiagnostics(
        max_abs_eta_delta=float(np.max(np.abs(eta_after - eta_before))) if eta_before.size else 0.0,
        max_abs_mu_delta=float(np.max(np.abs(mu_after - mu_before))) if mu_before.size else 0.0,
        intercept_shift=float(total_shift),
        term_means_before=term_means_before,
        term_means_after=term_means_after,
    )

    return {
        "terms": term_states,
        "diagnostics": asdict(diagnostics),
        "intercept_shift": float(total_shift),
        "solver_to_public": np.eye(model._dm.p, dtype=np.float64),
    }


def _build_public_result(solver: PIRLSResult, state: dict[str, Any]) -> PIRLSResult:
    """Build the public PIRLS result from the private solver fit."""
    return PIRLSResult(
        beta=np.asarray(solver.beta, dtype=np.float64).copy(),
        intercept=float(solver.intercept) + float(state["intercept_shift"]),
        n_iter=solver.n_iter,
        deviance=solver.deviance,
        converged=solver.converged,
        phi=solver.phi,
        effective_df=solver.effective_df,
        iteration_log=solver.iteration_log,
        log_det_H=solver.log_det_H,
    )


def canonicalize_fitted_model(model) -> None:
    """Finalize the public runtime result after solver-space fitting completes."""
    solver = model._solver_pirls_result()
    state = _compile_runtime_state(model, solver)
    model._result = _build_public_result(solver, state)
    model._runtime_canonical_state = state
    model._prediction_plan = None
