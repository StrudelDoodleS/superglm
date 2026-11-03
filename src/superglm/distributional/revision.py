"""Hand-edited coefficients: predictions follow the edit, the fit's records do not.

A hand edit is a full coefficient vector that may differ from the fit's only
inside the terms it names, and at the intercept of each named term's
parameter. It lives in the fit state beside the fit's own solver result,
inference and covariance, which keep describing the fit.

Models and row states are typed ``Any``: ``fit_state`` imports this module for
``HandEdit``, so it cannot import ``fit_state`` or ``model`` back
(``tests/test_distributional_family_kernel_architecture.py`` forbids import
cycles even under ``TYPE_CHECKING``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional._common import readonly_finite
from superglm.distributional.layout import StackedLayout

NO_TRAINING_FRAME = (
    "Pass X_train= to revise the coefficients of a model whose training frame is not attached."
)
CORRECTED_COVARIANCE_NEEDS_THE_FIT = (
    "The corrected covariance needs the fitted coefficients; use covariance='fixed', or refit."
)


class EditedCoefficientsWarning(UserWarning):
    """Standard errors, bands or tests are quoted for a model edited by hand."""


@dataclass(frozen=True, eq=False)
class HandEdit:
    """Prediction coefficients that differ from the fit's inside ``edited_terms``.

    ``coefficients`` is the full vector in the fit's layout, read-only.
    ``edited_terms`` are qualified term names such as ``"scale:age"``.
    """

    coefficients: NDArray[np.float64]
    edited_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.edited_terms, str):
            raise TypeError("edited_terms must be a sequence of term names, not one string")
        terms = tuple(dict.fromkeys(self.edited_terms))
        if not terms or not all(isinstance(term, str) for term in terms):
            raise ValueError("edited_terms must name at least one term, as '<parameter>:<term>'")
        coefficients = readonly_finite(self.coefficients, name="hand-edited coefficients")
        if coefficients.ndim != 1:
            raise ValueError("hand-edited coefficients must be one-dimensional")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "edited_terms", terms)


def check_hand_edit(edit: HandEdit, layout: StackedLayout, fitted: NDArray[np.float64]) -> None:
    """Refuse an edit that moves a coefficient outside its edited terms.

    A term's own slice may move, and so may the intercept of its parameter,
    since a centred curve leaves its level to the intercept. The first other
    entry that differs from ``fitted`` is named.
    """
    if edit.coefficients.shape != (layout.n_coefficients,):
        raise ValueError(
            f"Hand-edited coefficients must hold {layout.n_coefficients} values, one per entry "
            f"of the fit; got {edit.coefficients.shape[0]}."
        )
    unknown = [term for term in edit.edited_terms if term not in layout.term_slices]
    if unknown:
        raise ValueError(
            f"Edited term {unknown[0]!r} is not a term of the fit. "
            "Name terms as '<parameter>:<term>', for example 'scale:age'."
        )
    allowed = np.zeros(layout.n_coefficients, dtype=np.bool_)
    for term in edit.edited_terms:
        allowed[layout.term_slices[term]] = True
        intercept = layout.predictor(term.partition(":")[0]).intercept_index
        if intercept is not None:
            allowed[intercept] = True
    stray = np.flatnonzero((edit.coefficients != fitted) & ~allowed)
    if stray.size:
        raise ValueError(
            f"Coefficient {layout.coefficient_names[stray[0]]!r} differs from the fit but is "
            f"outside the edited terms ({', '.join(edit.edited_terms)}). "
            "Change only the edited terms, or name this one in edited_terms."
        )


def _rows_of_the_fit(frame: EagerFrame, rows: Any) -> EagerFrame:
    """The rows the fit kept, from either the kept rows or the frame it was given."""
    provenance = rows.likelihood_weights.provenance
    if len(frame) == provenance.retained_count:
        return frame
    if len(frame) == provenance.original_count:
        return as_eager_frame(frame.take_rows(rows.likelihood_weights.input_positions))
    raise ValueError(
        f"X_train has {len(frame)} rows, but the fit was given {provenance.original_count} "
        f"and kept {provenance.retained_count}. Pass the frame the model was fitted on."
    )


def _recomputed_rows(
    model: Any,
    rows: Any,
    edit: HandEdit,
    X_train: FrameLike | EagerFrame | None,
) -> Any:
    """The retained rows with fitted ``eta`` and parameters taken at the edit.

    They are evaluated through the prediction design, so on a discrete fit they
    are the exact design's values, not the binned ones the fit itself used.
    """
    if X_train is None:
        raise ValueError(NO_TRAINING_FRAME)
    eta = model.predict_eta_at(
        edit.coefficients,
        _rows_of_the_fit(as_eager_frame(X_train), rows),
        offsets=rows.offsets,
    )
    return replace(rows, fitted_eta=eta, fitted_parameters=model.parameters_from_eta(eta))


def revised_model(
    model: Any,
    coefficients: NDArray,
    edited_terms: Sequence[str],
    X_train: FrameLike | EagerFrame | None,
) -> Any:
    """A model predicting with ``coefficients``; ``model`` is left as it was.

    Terms edited earlier stay edited, and every difference is judged against the
    fit's coefficients. The retained fitted rows are recomputed from the edit,
    which needs the training rows.
    """
    if isinstance(edited_terms, str):
        raise TypeError("edited_terms must be a sequence of term names, not one string")
    state = model.fit_state
    earlier = () if state.hand_edit is None else state.hand_edit.edited_terms
    edit = HandEdit(coefficients, (*earlier, *edited_terms))
    check_hand_edit(edit, state.layout, state.result.coefficients)
    rows = state.retained_rows
    if rows is not None:
        rows = _recomputed_rows(model, rows, edit, X_train)
    return replace(model, _fit_state=replace(state, hand_edit=edit, retained_rows=rows))


__all__ = [
    "CORRECTED_COVARIANCE_NEEDS_THE_FIT",
    "EditedCoefficientsWarning",
    "HandEdit",
    "check_hand_edit",
    "revised_model",
]
