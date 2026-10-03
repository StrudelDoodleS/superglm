"""Carry hand-edited curves onto a refitted model (spec D2, D5).

A Refit re-specifies some terms and refits all of them. A term it left alone
keeps its rows and ``n_points``, so its display grid and level labels are the
same, and its hand-edited curve can go back onto the refitted model exactly
(:func:`carried_curve`).

Run CV refits the in-force structure on each fold and Final fit refits it on
train and validation rows together; both then put the hand edits back as
set (D5). Each of those refits draws its own grid: a fold's numeric range is
its own training range and its level universe is its own.
:func:`model_with_edited_curves` carries every edited curve onto that grid
and patches the refit with the same code an editor export uses
(:func:`apply.apply_edits_to_model_copy_with_data`). A level that refit never
saw (one only other folds' rows hold) is pinned there, so it keeps its pin
and the rest of the edit lands.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.editor import apply
from superglm.editor._types import EditableTerm
from superglm.editor.terms import native_log_effect_values, term_offset_values


def carried_curve(edited: EditableTerm, refitted: EditableTerm) -> NDArray | None:
    """``edited``'s curve as ``refitted`` would display it, or None when their grids differ.

    The curve kept is the one the model scores, the native one. Shown natively
    it is ``edited``'s own values; shown mean-centred it moves by the refit's
    own centring constant, so that the scored curve does not.
    """
    if edited.size != refitted.size or edited.levels != refitted.levels:
        return None
    if (edited.x is None) != (refitted.x is None):
        return None
    if edited.x is not None and refitted.x is not None and not np.array_equal(edited.x, refitted.x):
        return None
    return native_log_effect_values(edited) - _centring_offset(refitted)


def _centring_offset(term: EditableTerm) -> NDArray | float:
    """``term``'s native curve less its displayed one: zero unless it is centred for display."""
    native = term.metadata.get("native_original_log_effect")
    if native is None:
        return 0.0
    native = np.asarray(native, dtype=np.float64).ravel()
    if native.shape != term.original_log_effect.shape:
        return 0.0
    return native - np.asarray(term.original_log_effect, dtype=np.float64)


def model_with_edited_curves(
    model,
    edited: dict[str, EditableTerm],
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    n_points: int = 200,
):
    """Return a copy of fitted ``model`` with each ``edited`` curve put back as set.

    ``edited`` maps term names to the editor's terms, each a main effect of
    ``model``. A numeric curve is carried by ``np.interp`` on ``x``, held at
    its end values past either end of the editor's grid (a piecewise term
    follows its own extrapolation, as its offset does). A level curve is
    carried by label; a level the editor never showed keeps its refit value,
    and a level ``model`` holds pinned (no training rows) keeps its pin.

    The carried curve keeps the edited shape exactly and moves by one
    constant: the exposure-weighted mean of the refit curve minus the
    editor's original curve. That constant is the difference between the
    two fits' centring, a fold's own centring or a reference level that
    re-resolved on more rows, so it does not move predictions; an edit that
    shifted the whole curve keeps its shift.

    ``X``, ``y``, ``sample_weight`` and ``offset`` are the rows ``model`` was
    fitted on. They weight the projection onto the refit's basis and refresh
    the copy's fit statistics. ``model`` itself is not changed.
    """
    from superglm.editor.session import EditorSession

    if not edited:
        return model
    session = EditorSession.from_model(
        model,
        list(edited),
        n_points=n_points,
        centering="native",
        with_se=False,
        train_data=(X, y, sample_weight, offset),
    )
    for name, source in edited.items():
        target = session.terms[name]
        target.edited_log_effect = _carried_values(source, target)
    return apply.apply_edits_to_model_copy_with_data(
        session.model,
        session.terms,
        X=X,
        y=y,
        sample_weight=sample_weight,
        offset=offset,
        keep_pinned=True,
    )


def _carried_values(source: EditableTerm, target: EditableTerm) -> NDArray[np.float64]:
    """``source``'s edited curve on ``target``'s grid, in ``target``'s centring."""
    refit = np.asarray(target.edited_log_effect, dtype=np.float64)
    edited = native_log_effect_values(source)
    if source.size == 1:
        # A linear term's one value is its slope, which carries no centring.
        return edited.copy()
    original = _original_curve(source)
    weights = _weights(target)
    if target.levels is not None:
        position = {label: index for index, label in enumerate(source.levels or ())}
        shared = np.array([label in position for label in target.levels], dtype=bool)
        index = [position[label] for label in target.levels if label in position]
        values = refit.copy()
        anchor = _anchor(refit[shared], native_log_effect_values(original)[index], weights[shared])
        values[shared] = edited[index] + anchor
        return values
    x = np.asarray(target.x, dtype=np.float64)
    anchor = _anchor(refit, term_offset_values(original, x), weights)
    return term_offset_values(source, x) + anchor


def _original_curve(term: EditableTerm) -> EditableTerm:
    """``term`` with its edits undone, so the offset helpers read the original curve."""
    unedited = term.copy()
    unedited.edited_log_effect = unedited.original_log_effect.copy()
    return unedited


def _weights(term: EditableTerm) -> NDArray[np.float64]:
    if term.weights is None:
        return np.ones(term.size, dtype=np.float64)
    return np.asarray(term.weights, dtype=np.float64)


def _anchor(refit: NDArray, original: NDArray, weights: NDArray) -> float:
    """Exposure-weighted mean of ``refit - original``; unweighted when no row backs it."""
    if refit.size == 0:
        return 0.0
    if not float(np.sum(weights)) > 0.0:
        weights = np.ones_like(refit)
    return float(np.average(refit - original, weights=weights))


__all__ = ["carried_curve", "model_with_edited_curves"]
