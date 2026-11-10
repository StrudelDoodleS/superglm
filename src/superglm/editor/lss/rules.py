"""A term's shape rules, as the editor names them, and a SuperLSS term's rows ``A b >= c``.

A ``Constraint.fit`` rule reaches a SuperLSS fit as linear rows ``A b >= c`` on its term's
coefficients (``GroupSlice.constraints``), which the fit enforces
(``distributional/solver/shape.py``). The editor keeps a hand edit of such a term inside the same
rows (``lss.apply.kept_inside_rules``), so a curve the editor shows, and the model it writes, meet
every rule the fit met. A flat stretch is part of the term's basis rather than a row, and holds on
any curve the basis draws.

A SuperGLM term's rules are only named: its hand edits are not held to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.terms import _resolve_term
from superglm.editor.lss.view import LSSModelView, split_key
from superglm.features.constraint import rules_of


@dataclass(frozen=True)
class TermRows:
    """A SuperLSS term's rules as written, and its rows ``A b >= c`` on the term's coefficients.

    ``A`` has one column per coefficient of the term's slice, in its order, and no rows when
    every rule is held by the basis.
    """

    text: str
    A: NDArray[np.float64]
    c: NDArray[np.float64]


def term_rows(view: LSSModelView, key: str) -> TermRows | None:
    """The rows of the SuperLSS term ``key`` names, or None when it has no fit-time rule.

    Read once per view: the rows belong to its fitted model.
    """
    if key not in view._term_rows:
        view._term_rows[key] = _read_rows(view, key)
    return view._term_rows[key]


def _read_rows(view: LSSModelView, key: str) -> TermRows | None:
    rules = rules_of(view._specs[key])
    if not rules or any(rule.mode != "fit" for rule in rules):
        return None
    parameter, label = split_key(key)
    _qualified, state, term_slice = _resolve_term(view._fitted, parameter, label)
    width = term_slice.stop - term_slice.start
    slope_start = state.coefficient_slice.start + int(state.intercept_index is not None)
    blocks: list[NDArray[np.float64]] = []
    bounds: list[NDArray[np.float64]] = []
    for group in state.groups:
        start = slope_start + group.start - term_slice.start
        if group.constraints is None or not 0 <= start < width:
            continue
        local = np.asarray(group.constraints.A, dtype=np.float64)
        block = np.zeros((local.shape[0], width), dtype=np.float64)
        block[:, start : start + local.shape[1]] = local
        blocks.append(block)
        bounds.append(np.asarray(group.constraints.b, dtype=np.float64))
    A = np.vstack(blocks) if blocks else np.zeros((0, width), dtype=np.float64)
    c = np.concatenate(bounds) if bounds else np.zeros(0, dtype=np.float64)
    return TermRows(rules.describe(), A, c)


def holds_rules(model: Any, name: str) -> bool:
    """Whether the editor keeps hand edits of term ``name`` inside fit-time rules."""
    return isinstance(model, LSSModelView) and term_rows(model, name) is not None


def rule_payload(model: Any, name: str) -> dict[str, Any] | None:
    """The rules term ``name`` carries, as the context bar names them, or None when it has none.

    ``fit`` says the fit enforces them, rather than repairing the curve after it;
    ``kept`` that hand edits are kept inside them, which a SuperLSS term's are.
    """
    rules = rules_of(model._specs[name])
    if not rules:
        return None
    fit = all(rule.mode == "fit" for rule in rules)
    return {"text": rules.describe(), "fit": fit, "kept": holds_rules(model, name)}


__all__ = ["TermRows", "holds_rules", "rule_payload", "term_rows"]
