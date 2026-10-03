"""Carry hand-edited curves onto a refitted model (spec D2).

A Refit re-specifies some terms and refits all of them. A term it left alone
keeps its rows and ``n_points``, so its display grid and level labels are the
same, and its hand-edited curve can go back onto the refitted model exactly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.editor._types import EditableTerm
from superglm.editor.terms import native_log_effect_values


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
    if edited.x is not None and not np.array_equal(edited.x, refitted.x):
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
