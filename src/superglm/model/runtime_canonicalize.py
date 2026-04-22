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


def _is_spline_backed_feature_spec(spec: Any) -> bool:
    """Whether this main-effect spec exposes spline-style runtime hooks."""
    return hasattr(spec, "_basis_matrix")


def _is_spline_backed_interaction(model, spec: Any) -> bool:
    """Whether this interaction is backed by spline parent terms."""
    parent_names = getattr(spec, "parent_names", None)
    if parent_names is None:
        return False
    left, right = parent_names
    return (
        left in model._specs
        and right in model._specs
        and _is_spline_backed_feature_spec(model._specs[left])
        and _is_spline_backed_feature_spec(model._specs[right])
    )


def _iter_spline_backed_terms(model):
    """Yield spline-backed main effects and interactions."""
    for feature_name in model._feature_order:
        if feature_name in model._interaction_specs:
            continue
        spec = model._specs[feature_name]
        if _is_spline_backed_feature_spec(spec):
            yield feature_name, spec, "feature"

    for interaction_name in model._interaction_order:
        if interaction_name in model._specs:
            continue
        spec = model._interaction_specs[interaction_name]
        if _is_spline_backed_interaction(model, spec):
            yield interaction_name, spec, "interaction"


def _feature_group_indices(model, feature_name: str) -> tuple[int, ...]:
    """Return all fitted group indices belonging to one term."""
    return tuple(i for i, group in enumerate(model._groups) if group.feature_name == feature_name)


def _group_column_means(group_matrix, n_rows: int) -> NDArray[np.float64]:
    """Compute unweighted training-row column means for one fitted block."""
    ones = np.ones(n_rows, dtype=np.float64)
    return np.asarray(group_matrix.rmatvec(ones), dtype=np.float64) / float(n_rows)


def _term_contribution(
    model,
    solver: PIRLSResult,
    group_indices: tuple[int, ...],
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


def _term_group_metadata(model, group_indices: tuple[int, ...]) -> list[dict[str, Any]]:
    """Collect blockwise group metadata for one term."""
    groups: list[dict[str, Any]] = []
    for idx in group_indices:
        group = model._groups[idx]
        groups.append(
            {
                "group_index": idx,
                "group_name": group.name,
                "solver_slice": (group.start, group.end),
                "column_means": _group_column_means(model._dm.group_matrices[idx], model._dm.n),
            }
        )
    return groups


def _runtime_training_feature_column_means(
    model,
    feature_name: str,
    spec: Any,
) -> NDArray[np.float64]:
    """Compute exact public runtime column means on the stored training rows."""
    X_ref = model._fit_X_ref
    if X_ref is None:
        raise RuntimeError("Training feature reference is required for runtime canonicalization")

    values = np.asarray(X_ref[feature_name], dtype=np.float64)
    return np.asarray(np.mean(spec.transform(values), axis=0), dtype=np.float64)


def _replace_group_column_means(
    groups: list[dict[str, Any]],
    column_means: NDArray[np.float64],
) -> list[dict[str, Any]]:
    """Return group metadata with the exact runtime column means substituted in."""
    if len(groups) != 1:
        raise ValueError("Public runtime mean substitution currently requires a single group")

    group_state = dict(groups[0])
    group_state["column_means"] = np.asarray(column_means, dtype=np.float64)
    return [group_state]


def _term_shift_from_groups(
    solver: PIRLSResult,
    groups: list[dict[str, Any]],
) -> float:
    """Compute the scalar intercept shift implied by the blockwise means."""
    shift = 0.0
    for group_state in groups:
        start, end = group_state["solver_slice"]
        shift += float(np.dot(group_state["column_means"], solver.beta[start:end]))
    return shift


def _group_mode(model, group_indices: tuple[int, ...], spec: Any) -> str:
    """Return the canonicalization mode for this term."""
    groups = [model._groups[idx] for idx in group_indices]
    if getattr(spec, "_scop_Sigma", None) is not None:
        return "affine"
    if any(group.monotone_engine == "qp" or group.constraints is not None for group in groups):
        return "affine"
    return "blockwise"


def _apply_r_inv_centering(spec: Any, column_means: NDArray[np.float64]) -> None:
    """Apply centering to an SSP-style runtime transform."""
    r_inv = np.asarray(spec._R_inv, dtype=np.float64)
    spec._R_inv = r_inv - np.ones((r_inv.shape[0], 1), dtype=np.float64) @ column_means[None, :]


def _apply_scop_centering(spec: Any, column_means: NDArray[np.float64]) -> None:
    """Apply centering to a SCOP runtime transform."""
    spec._scop_col_means = np.asarray(spec._scop_col_means, dtype=np.float64) + column_means


def _materialize_public_term(
    term_kind: str,
    spec: Any,
    mode: str,
    groups: list[dict[str, Any]],
) -> bool:
    """Apply canonicalization directly to the current public term when possible."""
    if not _can_materialize_public_term(term_kind, spec, mode, groups):
        return False

    column_means = groups[0]["column_means"]
    if getattr(spec, "_scop_Sigma", None) is not None:
        _apply_scop_centering(spec, column_means)
        return True

    _apply_r_inv_centering(spec, column_means)
    return True


def _can_materialize_public_term(
    term_kind: str,
    spec: Any,
    mode: str,
    groups: list[dict[str, Any]],
) -> bool:
    """Whether this term can be canonicalized directly in the public model."""
    if term_kind != "feature" or len(groups) != 1:
        return False

    if getattr(spec, "_scop_Sigma", None) is not None:
        return True

    if getattr(spec, "_R_inv", None) is not None and mode in {"blockwise", "affine"}:
        return True

    return False


def _compile_runtime_terms(
    model,
    solver: PIRLSResult,
) -> tuple[dict[str, dict[str, Any]], float]:
    """Compile spline-backed term state and apply the Task 2 public mutations."""
    term_states: dict[str, dict[str, Any]] = {}
    public_intercept_shift = 0.0

    for term_name, spec, term_kind in _iter_spline_backed_terms(model):
        group_indices = _feature_group_indices(model, term_name)
        if not group_indices:
            continue

        groups = _term_group_metadata(model, group_indices)
        mode = _group_mode(model, group_indices, spec)
        can_materialize = _can_materialize_public_term(term_kind, spec, mode, groups)
        if can_materialize:
            groups = _replace_group_column_means(
                groups,
                _runtime_training_feature_column_means(model, term_name, spec),
            )

        shift = _term_shift_from_groups(solver, groups)
        mean_before = (
            float(shift)
            if can_materialize
            else float(np.mean(_term_contribution(model, solver, group_indices)))
        )
        applied_to_public_model = _materialize_public_term(term_kind, spec, mode, groups)

        if applied_to_public_model:
            public_intercept_shift += shift

        term_states[term_name] = {
            "term_kind": term_kind,
            "mode": mode,
            "applied_to_public_model": applied_to_public_model,
            "group_indices": group_indices,
            "groups": groups,
            "term_mean_before": mean_before,
            "term_mean_after": None,
            "intercept_shift": float(shift),
        }

    return term_states, public_intercept_shift


def _live_public_runtime_state(
    model,
    public_result: PIRLSResult,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the training rows through the live public scoring contract."""
    from superglm.model import base

    X_ref = model._fit_X_ref
    if X_ref is None:
        raise RuntimeError("Training feature reference is required for runtime diagnostics")

    plan = base._build_prediction_plan(model)
    beta_all = public_result.beta
    eta = np.full(len(X_ref), public_result.intercept, dtype=np.float64)
    contributions: dict[str, NDArray[np.float64]] = {}

    for term in plan["features"]:
        if term["name"] in model._interaction_specs:
            continue
        values = np.asarray(X_ref[term["name"]])
        beta = beta_all[term["beta_idx"]]
        contribution = np.asarray(
            base._score_feature(term["spec"], values, beta),
            dtype=np.float64,
        ).ravel()
        contributions[term["name"]] = contribution
        eta += contribution

    for term in plan["interactions"]:
        if term["name"] in model._specs:
            continue
        spec = term["spec"]
        left_name, right_name = spec.parent_names
        beta = beta_all[term["beta_idx"]]
        contribution = np.asarray(
            base._score_interaction(
                spec,
                np.asarray(X_ref[left_name]),
                np.asarray(X_ref[right_name]),
                beta,
            ),
            dtype=np.float64,
        ).ravel()
        contributions[term["name"]] = contribution
        eta += contribution

    if model._fit_offset is not None:
        eta = eta + model._fit_offset

    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    return contributions, eta, mu


def _compute_public_parity_diagnostics(
    model,
    solver: PIRLSResult,
    public_result: PIRLSResult,
    term_states: dict[str, dict[str, Any]],
    intercept_shift: float,
) -> tuple[CanonicalizationDiagnostics, dict[str, float]]:
    """Compute diagnostics from the live public runtime state."""
    eta_before = model._dm.matvec(solver.beta) + float(solver.intercept)
    if model._fit_offset is not None:
        eta_before = eta_before + model._fit_offset
    eta_before = stabilize_eta(eta_before, model._link)
    mu_before = clip_mu(model._link.inverse(eta_before), model._distribution)

    contributions_after, eta_after, mu_after = _live_public_runtime_state(model, public_result)
    live_term_means_after = {
        term_name: float(np.mean(contributions_after[term_name]))
        for term_name in term_states
        if term_name in contributions_after
    }

    diagnostics = CanonicalizationDiagnostics(
        max_abs_eta_delta=float(np.max(np.abs(eta_after - eta_before))) if eta_before.size else 0.0,
        max_abs_mu_delta=float(np.max(np.abs(mu_after - mu_before))) if mu_before.size else 0.0,
        intercept_shift=float(intercept_shift),
        term_means_before={
            term_name: float(term_state["term_mean_before"])
            for term_name, term_state in term_states.items()
        },
        term_means_after=live_term_means_after,
    )
    return diagnostics, live_term_means_after


def _solver_to_public_state(
    model,
    term_states: dict[str, dict[str, Any]],
) -> tuple[NDArray[np.float64] | None, bool]:
    """Return the honest solver-to-public mapping state for Task 2."""
    complete = all(term_state["applied_to_public_model"] for term_state in term_states.values())
    if complete:
        return np.eye(model._dm.p, dtype=np.float64), True
    return None, False


def _build_public_result(solver: PIRLSResult, state: dict[str, Any]) -> PIRLSResult:
    """Build the public PIRLS result from the private solver fit."""
    public_result = PIRLSResult(
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
    if hasattr(solver, "scop_states"):
        public_result.scop_states = solver.scop_states
    return public_result


def canonicalize_fitted_model(model) -> None:
    """Finalize the public runtime result after solver-space fitting completes."""
    from superglm.model import base

    solver = model._solver_pirls_result()
    term_states, public_intercept_shift = _compile_runtime_terms(model, solver)

    state = {"intercept_shift": float(public_intercept_shift)}
    public_result = _build_public_result(solver, state)
    diagnostics, live_term_means_after = _compute_public_parity_diagnostics(
        model,
        solver,
        public_result,
        term_states,
        public_intercept_shift,
    )

    for term_name, term_state in term_states.items():
        if term_state["applied_to_public_model"]:
            term_state["term_mean_after"] = live_term_means_after.get(term_name)

    solver_to_public, solver_to_public_complete = _solver_to_public_state(model, term_states)

    model._result = public_result
    model._runtime_canonical_state = {
        "terms": term_states,
        "diagnostics": asdict(diagnostics),
        "intercept_shift": float(public_intercept_shift),
        "solver_to_public": solver_to_public,
        "solver_to_public_complete": solver_to_public_complete,
    }
    base.freeze_prediction_plan(model)


def canonicalize_intercept_path(
    model,
    coef_path: NDArray[np.float64],
    intercept_path: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map a solver-space intercept path into the public canonical intercept path."""
    state = getattr(model, "_runtime_canonical_state", None)
    if not isinstance(state, dict):
        return np.asarray(intercept_path, dtype=np.float64)

    terms = state.get("terms", {})
    intercept_public = np.asarray(intercept_path, dtype=np.float64).copy()
    if coef_path.size == 0 or not terms:
        return intercept_public

    for term_state in terms.values():
        if not term_state.get("applied_to_public_model"):
            continue
        for group_state in term_state["groups"]:
            start, end = group_state["solver_slice"]
            means = np.asarray(group_state["column_means"], dtype=np.float64)
            intercept_public += coef_path[:, start:end] @ means

    return intercept_public
