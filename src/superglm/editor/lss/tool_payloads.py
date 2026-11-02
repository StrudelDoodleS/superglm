"""The tool sub-payloads of a SuperLSS term, in their "not available" shape.

The shape, knot, basis, new-level and reference tools read a SuperGLM's
declarations, which a SuperLSS term does not keep in that form. A SuperLSS term
carries the same keys the browser reads, with every tool off and no reason.
"""

from __future__ import annotations

from typing import Any

from superglm.editor.shapes import EDITOR_JOINS

_KNOT_FIELDS = (
    "positions",
    "count",
    "strategy",
    "alpha",
    "lo",
    "hi",
    "min_gap",
    "basis",
    "waiting_basis",
    "difference_penalty",
    "max_count",
    "even_only",
    "kind",
    "select",
)


def lss_shape_payload() -> dict[str, Any]:
    return {
        "available": False,
        "reason": None,
        "ranges": [],
        "support": None,
        "specials": [],
        "returnable": [],
        "joins": list(EDITOR_JOINS),
        "join_reason": None,
    }


def lss_knots_payload() -> dict[str, Any]:
    return {
        **dict.fromkeys(_KNOT_FIELDS),
        "available": False,
        "reason": None,
        "from_editor": False,
        "resettable": False,
        "kinds": [],
        "select_available": False,
        "select_reason": None,
    }
