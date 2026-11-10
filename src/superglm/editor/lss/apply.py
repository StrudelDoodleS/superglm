"""Turn edited SuperLSS curves into a revised model.

An edited curve lives on the term's grid, on the link scale of its parameter.
A categorical term takes each non-base level's coefficient straight from the
edited curve. Every other term is projected onto its basis by weighted least
squares, the way ``editor.apply`` projects a SuperGLM spline; the term's design
comes from the compiled spec at the reference row, so a select-centred basis is
projected through the columns the model predicts with. What a curve cannot
carry, its level, moves the intercept of its parameter.

Only terms whose curve differs from the fitted one are touched; every other
coefficient keeps its fitted value exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.api import SuperLSS
from superglm.distributional.terms import _level_domain, _resolve_term, _term_design
from superglm.editor import apply as editor_apply
from superglm.editor._types import EditableTerm
from superglm.editor.errors import EditorValueError
from superglm.editor.lss.view import LSSModelView, split_key
from superglm.editor.terms import native_log_effect_values
from superglm.features.categorical import Categorical

_SPECIAL_SUFFIX = ":special"


def lss_model_with_edits(
    view: LSSModelView,
    terms: Mapping[str, EditableTerm],
    *,
    keep_pinned: bool = False,
) -> SuperLSS:
    """The fitted model with the edited curves of ``terms`` written into its coefficients.

    Returns ``view.lss`` itself when no curve differs from its fitted one. The
    rows come from ``view.X_train``. ``keep_pinned`` is for carrying edits onto
    another fit, where a level that fit never saw keeps its pin and the rest of
    the term is edited.
    """
    changed = [term for term in terms.values() if _is_edited(term)]
    if not changed:
        return view.lss
    coefficients = np.array(view._fitted.coefficients, dtype=np.float64)
    for term in changed:
        _write_term(view, term, coefficients, keep_pinned=keep_pinned)
    return view.lss._with_revised_coefficients(
        coefficients, edited_terms=[term.name for term in changed], X_train=view.X_train
    )


def _is_edited(term: EditableTerm) -> bool:
    return not np.allclose(term.edited_log_effect, term.original_log_effect, rtol=0.0, atol=1e-14)


def _write_term(
    view: LSSModelView,
    term: EditableTerm,
    coefficients: NDArray[np.float64],
    *,
    keep_pinned: bool,
) -> None:
    state, term_slice, block, shift = _term_block(view, term, keep_pinned=keep_pinned)
    coefficients[term_slice] = block
    coefficients[state.intercept_index] += shift


def _term_block(
    view: LSSModelView, term: EditableTerm, *, keep_pinned: bool
) -> tuple[Any, slice, NDArray[np.float64], float]:
    """The term's state and slice, its edited coefficients and the level it puts on the intercept."""
    parameter, label = split_key(term.name)
    fitted = view._fitted
    qualified, state, term_slice = _resolve_term(fitted, parameter, label)
    if f"{qualified}{_SPECIAL_SUFFIX}" in fitted.layout.term_slices:
        raise EditorValueError(
            f"{term.name!r} has special levels, whose coefficients the editor does not write yet. "
            "Edit a term without specials, or refit with the change."
        )
    if state.intercept_index is None:
        raise EditorValueError(
            f"The {parameter!r} parameter is fitted without an intercept, so an edit to "
            f"{term.name!r} has nowhere to put its level. Refit with an intercept to edit it."
        )
    spec = view._specs[term.name]
    if isinstance(spec, Categorical):
        block, shift = _level_block(term, spec, keep_pinned=keep_pinned)
    else:
        block, shift = _curve_block(view, term, label, state, term_slice)
    if block.size != term_slice.stop - term_slice.start:
        raise EditorValueError(
            f"The edit to {term.name!r} has {block.size} coefficients, but the term holds "
            f"{term_slice.stop - term_slice.start}. Reopen the editor on this model."
        )
    return state, term_slice, block, shift


def _level_block(
    term: EditableTerm, spec: Categorical, *, keep_pinned: bool
) -> tuple[NDArray[np.float64], float]:
    """Each non-base level's coefficient, and the base level's edit for the intercept.

    The reference level owns no design column, so its fitted effect is exactly
    zero and its edit is the whole shift of the intercept.
    """
    pinned = sorted(getattr(spec, "_pinned_levels", ()), key=str)
    if pinned and not keep_pinned:
        raise EditorValueError(
            f"Editable term {term.name!r} cannot be edited: level(s) {pinned} had no effective "
            "training rows in this fit and are pinned to the base level, so they have no fitted "
            "coefficient to edit. Refit on data carrying those level(s) with weight, or remove "
            "them from the term's level universe (levels=)."
        )
    target = editor_apply._level_target_map(term, spec)
    base = target[str(spec._base_level)]
    block = np.array([target[str(level)] - base for level in spec._non_base], dtype=np.float64)
    return block, base


def _curve_block(
    view: LSSModelView, term: EditableTerm, label: str, state: Any, term_slice: slice
) -> tuple[NDArray[np.float64], float]:
    """The weighted least-squares fit of the edited curve by the term's basis.

    The curve is read at ``term.x``, or at the levels of a term with levels, each
    read from the value the term was fitted on. The constant the basis cannot
    carry is returned for the intercept.
    """
    if term.x is not None:
        grid: Any = np.asarray(term.x)
    elif term.levels is not None:
        native = _native_levels(view._specs[term.name])
        grid = [native[str(level)] for level in term.levels]
    else:
        raise EditorValueError(f"{term.name!r} has no grid to project an edit from.")
    rows = view.reference_row()
    rows = rows.iloc[np.zeros(len(grid), dtype=np.intp)].reset_index(drop=True)
    rows[label] = grid
    design = _term_design(view._fitted, rows, state, term_slice)
    fitted = np.asarray(view._fitted.coefficients, dtype=np.float64)[term_slice]
    target = native_log_effect_values(term) + float(view.reference_design(term.name) @ fitted)
    shift, block = _projection(design, target, editor_apply._term_weights(term))
    return block, shift


def _native_levels(spec: Any) -> dict[str, Any]:
    """Each level label the editor shows, mapped to the value the term reads it from."""
    grouping = getattr(spec, "_grouping", None)
    levels = _level_domain(spec) if grouping is None else grouping.all_original_levels
    return {str(level): level for level in levels or ()}


def _projection(
    design: NDArray[np.float64], target: NDArray[np.float64], weights: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    """The weighted least-squares fit of ``target`` by an intercept and ``design``.

    The rank is decided on columns scaled to unit weighted norm and centred, under
    weights that sum to one, so neither a covariate's units and origin nor the
    weights' scale decide which directions are kept; the intercept takes the
    centring back.
    """
    weights = weights / weights.sum()
    norm = np.sqrt(weights @ np.square(design))
    norm[norm == 0.0] = 1.0
    scaled = design / norm
    centre = weights @ scaled
    shift, coefficients = editor_apply._solve_with_intercept(scaled - centre, target, weights)
    return shift - float(centre @ coefficients), coefficients / norm


__all__ = ["lss_model_with_edits"]
