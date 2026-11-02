"""Post-fit canonicalization for the public runtime model state."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._frame import EagerFrame, as_eager_frame
from superglm.solvers.mode_score import two_sum
from superglm.solvers.pirls import PIRLSResult

_RUNTIME_MEAN_CHUNK_SIZE = 8192


@dataclass(frozen=True)
class _IdentityCoefficientMap:
    """Lazy identity map from solver coefficients to public coefficients."""

    size: int

    __array_priority__ = 1000.0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    @property
    def dtype(self) -> np.dtype[np.float64]:
        return np.dtype(np.float64)

    def __matmul__(self, other: Any) -> NDArray[Any]:
        array = np.asarray(other)
        if array.ndim == 0 or array.shape[0] != self.size:
            raise ValueError(
                f"Identity map with shape {self.shape} cannot multiply an array "
                f"with shape {array.shape}"
            )
        return array.copy()

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        copy: bool | None = None,
    ) -> NDArray[Any]:
        """Materialize only when an ndarray consumer explicitly requests it."""
        del copy
        return np.eye(self.size, dtype=np.float64 if dtype is None else dtype)


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
    frame: EagerFrame | None = None,
) -> NDArray[np.float64]:
    """Compute exact public runtime column means on the stored training rows."""
    X_ref = model._fit_X_ref
    if X_ref is None:
        raise RuntimeError("Training feature reference is required for runtime canonicalization")

    if frame is None:
        frame = as_eager_frame(X_ref)
    values = frame.column_array(feature_name, dtype=np.float64)
    support, counts = np.unique(values, return_counts=True)
    total: NDArray[np.float64] | None = None
    compensation: NDArray[np.float64] | None = None
    for start in range(0, len(support), _RUNTIME_MEAN_CHUNK_SIZE):
        stop = min(start + _RUNTIME_MEAN_CHUNK_SIZE, len(support))
        transformed = np.asarray(spec.transform(support[start:stop]), dtype=np.float64)
        chunk_total = np.asarray(counts[start:stop] @ transformed, dtype=np.float64)
        if total is None:
            total = np.zeros_like(chunk_total)
            compensation = np.zeros_like(chunk_total)
        assert compensation is not None
        corrected = chunk_total - compensation
        updated = total + corrected
        compensation[...] = (updated - total) - corrected
        total[...] = updated

    if total is None:
        return np.asarray(np.mean(spec.transform(values), axis=0), dtype=np.float64)
    return total / float(len(values))


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
    fit_frame: EagerFrame | None = None

    for term_name, spec, term_kind in _iter_spline_backed_terms(model):
        group_indices = _feature_group_indices(model, term_name)
        if not group_indices:
            continue

        groups = _term_group_metadata(model, group_indices)
        mode = _group_mode(model, group_indices, spec)
        can_materialize = _can_materialize_public_term(term_kind, spec, mode, groups)
        if can_materialize:
            if fit_frame is None and model._fit_X_ref is not None:
                fit_frame = as_eager_frame(model._fit_X_ref)
            groups = _replace_group_column_means(
                groups,
                _runtime_training_feature_column_means(
                    model,
                    term_name,
                    spec,
                    fit_frame,
                ),
            )

        shift = _term_shift_from_groups(solver, groups)
        applied_to_public_model = _materialize_public_term(term_kind, spec, mode, groups)

        if applied_to_public_model:
            public_intercept_shift += shift

        term_states[term_name] = {
            "term_kind": term_kind,
            "mode": mode,
            "applied_to_public_model": applied_to_public_model,
            "group_indices": group_indices,
            "groups": groups,
            "intercept_shift": float(shift),
        }

    return term_states, public_intercept_shift


def _solver_to_public_state(
    model,
    term_states: dict[str, dict[str, Any]],
) -> tuple[_IdentityCoefficientMap | None, bool]:
    """Return the honest solver-to-public mapping state for Task 2."""
    complete = all(term_state["applied_to_public_model"] for term_state in term_states.values())
    if complete:
        return _IdentityCoefficientMap(model._dm.p), True
    return None, False


def _public_centred_state(
    model,
    solver: PIRLSResult,
    term_states: dict[str, dict[str, Any]],
) -> tuple[float | None, NDArray[np.float64] | None, float | None]:
    """The solver's centred state ``(alpha, c, alpha_lo)`` read in the public coordinates.

    The fit's predictor is ``eta = alpha + (X - 1 c') beta`` (``mode_score.linear_predictor``),
    free of the cancellation between ``X beta`` and the raw intercept. Only a
    ``DenseGroupMatrix`` column its term scores centred (``base.scores_centred``) keeps its
    centre; every other column's ``(m - c)' beta`` is folded into ``alpha`` once, its rounding
    kept in ``alpha_lo`` by TwoSum. Returns ``(None, None, None)`` when there is no centred
    state, or no remainder and no kept centre, so the model predicts from its raw intercept.
    """
    from superglm.group_matrix import DenseGroupMatrix
    from superglm.model.base import scores_centred

    alpha = getattr(solver, "centred_intercept", None)
    centre = getattr(solver, "state_center", None)
    dm = getattr(model, "_dm", None)
    if alpha is None or centre is None or dm is None:
        return None, None, None
    centre = np.asarray(centre, dtype=np.float64)
    beta = np.asarray(solver.beta, dtype=np.float64)
    dense = np.zeros(centre.size, dtype=bool)
    start = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        dense[start : start + width] = type(matrix) is DenseGroupMatrix
        start += width
    differenced = np.zeros(centre.size, dtype=bool)
    specs = {**model._specs, **model._interaction_specs}
    for group in model._groups:
        if scores_centred(specs.get(group.feature_name)):
            differenced[group.start : group.end] = True
    shifts = _public_column_shifts(term_states, centre.size)
    public_centre = np.where(dense & differenced & (centre != 0.0), centre - shifts, 0.0)
    alpha_lo = getattr(solver, "centred_intercept_lo", None)
    # A compensated intercept keeps its pair even when every column folds into
    # it: the remainder is part of the fit's predictor.
    if not np.any(public_centre != 0.0) and alpha_lo is None:
        return None, None, None
    alpha_public, alpha_public_lo = _fold_public_pair(
        alpha, alpha_lo, centre, public_centre, shifts, beta
    )
    return alpha_public, public_centre, alpha_public_lo


def _public_column_shifts(term_states: dict[str, dict[str, Any]], size: int) -> NDArray[np.float64]:
    """The training means ``m`` each materialized public column loses, zero elsewhere."""
    shifts = np.zeros(size, dtype=np.float64)
    for term_state in term_states.values():
        if not term_state["applied_to_public_model"]:
            continue
        for group_state in term_state["groups"]:
            lo, hi = group_state["solver_slice"]
            shifts[lo:hi] = np.asarray(group_state["column_means"], dtype=np.float64)
    return shifts


def _fold_public_pair(
    alpha: float,
    alpha_lo: float | None,
    centre: NDArray[np.float64],
    public_centre: NDArray[np.float64],
    shifts: NDArray[np.float64],
    beta: NDArray[np.float64],
) -> tuple[float, float | None]:
    """``alpha + (m - c)' beta`` over the folded columns (zero public centre), compensated.

    The fold is one correctly rounded ``fsum``; with a remainder ``alpha_lo``
    the sum with ``alpha`` is a TwoSum whose error joins the remainder.
    """
    folded = public_centre == 0.0
    fold = math.fsum((shifts[folded] - centre[folded]) * beta[folded])
    if alpha_lo is None:
        return float(alpha) + fold, None
    alpha_public, fold_error = two_sum(float(alpha), fold)
    return alpha_public, float(alpha_lo) + fold_error


def _build_public_result(solver: PIRLSResult, state: dict[str, Any]) -> PIRLSResult:
    """Build the public PIRLS result from the private solver fit."""
    public_result = replace(
        solver,
        beta=np.asarray(solver.beta, dtype=np.float64).copy(),
        intercept=float(solver.intercept) + float(state["intercept_shift"]),
        centred_intercept=state.get("centred_intercept"),
        state_center=state.get("state_center"),
        centred_intercept_lo=state.get("centred_intercept_lo"),
    )
    if hasattr(solver, "scop_states"):
        public_result.scop_states = solver.scop_states
    return public_result


def canonicalize_fitted_model(model) -> None:
    """Finalize the public runtime result after solver-space fitting completes."""
    from superglm.model import base

    solver = model._solver_pirls_result()
    term_states, public_intercept_shift = _compile_runtime_terms(model, solver)

    centred_intercept, state_center, centred_intercept_lo = _public_centred_state(
        model, solver, term_states
    )
    state = {
        "intercept_shift": float(public_intercept_shift),
        "centred_intercept": centred_intercept,
        "state_center": state_center,
        "centred_intercept_lo": centred_intercept_lo,
    }
    public_result = _build_public_result(solver, state)
    solver_to_public, solver_to_public_complete = _solver_to_public_state(model, term_states)

    model._result = public_result
    model._runtime_canonical_state = {
        "terms": term_states,
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
