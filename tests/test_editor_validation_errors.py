"""Intentional editor validation is distinguishable from backend failures."""

import numpy as np
import pytest

from superglm.editor import EditableTerm, EditorSession
from superglm.editor.collapse import collapsed_feature_spec
from superglm.editor.controls import control_curve_after_move
from superglm.editor.level_order import level_order_for_direction
from superglm.editor.refit import fit_refit_model
from superglm.editor.terms import term_offset_values


@pytest.fixture
def session():
    curve = EditableTerm(
        name="curve",
        kind="spline",
        original_log_effect=np.zeros(3),
        edited_log_effect=np.zeros(3),
        x=np.arange(3, dtype=float),
    )
    category = EditableTerm(
        name="category",
        kind="categorical",
        original_log_effect=np.zeros(2),
        edited_log_effect=np.zeros(2),
        levels=["A", "B"],
    )
    return EditorSession(None, {"curve": curve, "category": category})


@pytest.mark.parametrize(
    ("operation", "error_type", "message"),
    [
        (
            lambda session: session.select_indices("curve", [3]),
            IndexError,
            "Selection indices out of range for term 'curve'.",
        ),
        (
            lambda session: session.selection("missing"),
            KeyError,
            "Unknown editable term: 'missing'",
        ),
        (
            lambda session: session.select_levels("curve", ["A"]),
            TypeError,
            "Term 'curve' does not have levels.",
        ),
        (
            lambda session: session.select_levels("category", ["C"]),
            KeyError,
            "Unknown level(s) for term 'category': ['C']",
        ),
        (
            lambda session: session.smooth("curve", strength=2),
            ValueError,
            "strength must be between 0 and 1, got 2",
        ),
        (
            lambda session: control_curve_after_move(None, session.terms["curve"], 99, 0.0),
            IndexError,
            "Control handle index out of range for term 'curve'.",
        ),
        (
            lambda session: collapsed_feature_spec(
                None, session.terms["category"], np.array([0]), X=None
            ),
            ValueError,
            "Select at least two levels to collapse term 'category'.",
        ),
        (
            lambda session: level_order_for_direction(2, np.array([0]), "up"),
            ValueError,
            "direction must be 'left' or 'right', got 'up'",
        ),
        (
            lambda session: fit_refit_model(None, None, method="invalid", X=None, y=None),
            ValueError,
            "method must be 'auto', 'fit', or 'fit_reml'.",
        ),
        (
            lambda session: term_offset_values(session.terms["category"], ["C"]),
            KeyError,
            "Offset data contains unseen level(s) for 'category': ['C']",
        ),
    ],
)
def test_intentional_validation_has_public_message(session, operation, error_type, message):
    with pytest.raises(error_type) as caught:
        operation(session)
    assert getattr(caught.value, "public_message", None) == message


@pytest.mark.parametrize("error_type", [KeyError, ValueError, TypeError, IndexError])
def test_arbitrary_builtin_errors_are_not_public(error_type):
    assert getattr(error_type("backend failure"), "public_message", None) is None


def test_array_conversion_failure_is_not_promoted_to_public_validation(session):
    with pytest.raises(ValueError) as caught:
        session.select_indices("curve", ["invalid index"])
    assert getattr(caught.value, "public_message", None) is None
