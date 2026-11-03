"""The shape tool's sub-payload of a SuperLSS term, in its "not available" shape.

The shape tool reads a SuperGLM's declarations, which a SuperLSS term does not
keep in that form. A SuperLSS term carries the same keys the browser reads, with the
tool off; ``lss_availability`` adds the reason.
"""

from __future__ import annotations

from typing import Any

from superglm.editor.shapes import EDITOR_JOINS


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
