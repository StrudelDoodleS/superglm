"""Turn edited SuperLSS curves into a revised model.

An edited curve lives on the term's grid, on the link scale of its parameter.
A categorical term takes each non-base level's coefficient straight from the
edited curve. Every other term is projected onto its basis by weighted least
squares, the way ``editor.apply`` projects a SuperGLM spline; the term's design
comes from the compiled spec at the reference row, so a select-centred basis is
projected through the columns the model predicts with. What a curve cannot
carry, its level, moves the intercept of its parameter.

A term with fit-time shape rules is projected under its rows ``A b >= c``
(``lss.rules``): onto the nearest curve, in the same weighted least squares,
that the term's basis draws with coefficients meeting them. With the intercept
free that is the convex QP ``min x' H x / 2 - g' x`` subject to ``[0 A] x >= c``,
``H = D' W D`` and ``g = D' W t`` for ``D`` the design beside a column of ones,
solved by the repository's active-set QP (``solvers/constrained_qp.py``). A
categorical term with a cap is projected the same way, through its level design.
An edit of some levels leaves the base where it is, and on a term that gives each
level its own coefficient the other levels it leaves as well (``_holds``). The
session keeps every edit of such a term on that curve
(``kept_inside_rules``), so the curve it shows is the curve the model takes, and
writing that curve into the model returns it unchanged.

Only terms whose curve differs from the fitted one are touched; every other
coefficient keeps its fitted value exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import gamma
from superglm.distributional.api import SuperLSS
from superglm.distributional.terms import _level_domain, _resolve_term, _term_design
from superglm.editor import apply as editor_apply
from superglm.editor._types import EditableTerm
from superglm.editor.errors import EditorValueError
from superglm.editor.lss.rules import TermRows, term_rows
from superglm.editor.lss.view import LSSModelView, split_key
from superglm.editor.terms import native_log_effect_values
from superglm.features.categorical import Categorical
from superglm.solvers.irls_direct._qp import (
    _QP_FEASIBILITY_TOL,
    _solve_constrained_qp_with_cold_retry,
)

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
    label, state, term_slice = _editable_slice(view, term)
    spec = view._specs[term.name]
    rows = term_rows(view, term.name)
    if isinstance(spec, Categorical):
        _require_unpinned(term, spec, keep_pinned=keep_pinned)
    if isinstance(spec, Categorical) and rows is None:
        block, shift = _level_block(term, spec)
    else:
        design, centre = _grid_design(view, term, label, state, term_slice)
        target = native_log_effect_values(term) + centre
        # The fitted curve meets the term's rules: the QP starts from it.
        inside = term.original_log_effect + centre
        shift, block = _projection(
            design, target, editor_apply._term_weights(term), rows, term.name, inside=inside
        )
    if block.size != term_slice.stop - term_slice.start:
        raise EditorValueError(
            f"The edit to {term.name!r} has {block.size} coefficients, but the term holds "
            f"{term_slice.stop - term_slice.start}. Reopen the editor on this model."
        )
    return state, term_slice, block, shift


def _editable_slice(view: LSSModelView, term: EditableTerm) -> tuple[str, Any, slice]:
    """The term's column, predictor state and coefficient slice, refusing a term none is written to."""
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
    return label, state, term_slice


def _require_unpinned(term: EditableTerm, spec: Categorical, *, keep_pinned: bool) -> None:
    """Refuse an edit of a categorical term with pinned levels, unless they keep their pin."""
    pinned = sorted(getattr(spec, "_pinned_levels", ()), key=str)
    if pinned and not keep_pinned:
        raise EditorValueError(
            f"Editable term {term.name!r} cannot be edited: level(s) {pinned} had no effective "
            "training rows in this fit and are pinned to the base level, so they have no fitted "
            "coefficient to edit. Refit on data carrying those level(s) with weight, or remove "
            "them from the term's level universe (levels=)."
        )


def _level_block(term: EditableTerm, spec: Categorical) -> tuple[NDArray[np.float64], float]:
    """Each non-base level's coefficient, and the base level's edit for the intercept.

    The reference level owns no design column, so its fitted effect is exactly
    zero and its edit is the whole shift of the intercept.
    """
    target = editor_apply._level_target_map(term, spec)
    base = target[str(spec._base_level)]
    block = np.array([target[str(level)] - base for level in spec._non_base], dtype=np.float64)
    return block, base


def _grid_design(
    view: LSSModelView, term: EditableTerm, label: str, state: Any, term_slice: slice
) -> tuple[NDArray[np.float64], float]:
    """The term's design on the editor's grid, and the shown curve's distance from its own.

    A term with levels is read at its levels, each from the value the term was
    fitted on, in the order the editor shows them; any other term at ``term.x``.
    A reordered level term carries display positions in ``term.x``, which are
    not values the term reads. The shown curve is measured from
    the reference level, so the term's own effect is the shown curve plus the
    returned constant.
    """
    if term.levels is not None:
        native = _native_levels(view._specs[term.name])
        grid: Any = [native[str(level)] for level in term.levels]
    elif term.x is not None:
        grid = np.asarray(term.x)
    else:
        raise EditorValueError(f"{term.name!r} has no grid to project an edit from.")
    rows = view.reference_row()
    rows = rows.iloc[np.zeros(len(grid), dtype=np.intp)].reset_index(drop=True)
    rows[label] = grid
    design = _term_design(view._fitted, rows, state, term_slice)
    fitted = np.asarray(view._fitted.coefficients, dtype=np.float64)[term_slice]
    return design, float(view.reference_design(term.name) @ fitted)


_OUTSIDE_RULES = (
    "The edit to {term!r} could not be kept inside its rules ({rules}): the nearest curve they "
    "allow could not be confirmed, so the edit was not made. Undo it, or make a smaller edit."
)

_ABSORBED = (
    "The edit to {term!r} changed nothing: its rules ({rules}) already hold the values it "
    "moves as far as they can go{held}. Edit values the rules leave room for, or remove the "
    "term's shape constraint and refit to go past them."
)


_NO_ROWS = (
    "The edit to {term!r} changed nothing: no training rows sit at the values it moves, so the "
    "curve there is set by the values around them. Edit values that have training rows, or a "
    "range that takes some in."
)


def absorbed_message(model: Any, term: EditableTerm, asked: NDArray[np.float64]) -> str:
    """Why an edit of ``term`` to ``asked`` that the projection took back entirely changed nothing.

    The projection weighs each grid value by its training rows, so an edit of
    values with none is taken back whatever the rules say; the rules are blamed
    only for an edit that moves a value with rows.
    """
    if moves_only_empty_values(term, asked):
        return _NO_ROWS.format(term=term.name)
    rows = term_rows(model, term.name)
    held = ", with the levels the edit leaves held where they are" if term.levels else ""
    return _ABSORBED.format(term=term.name, rules="" if rows is None else rows.text, held=held)


def moves_only_empty_values(term: EditableTerm, asked: NDArray[np.float64]) -> bool:
    """Whether every value ``asked`` changes on ``term`` has no training rows.

    A ruled edit is kept by a projection weighed by training rows, so it keeps
    nothing of such an edit, whatever the rules say.
    """
    if term.weights is None:
        return False
    moved = np.asarray(asked) != term.edited_log_effect
    return bool(moved.any()) and not np.any(np.asarray(term.weights)[moved] > 0.0)


# A kept curve that moves no value by more than this fraction of the change asked for, or of
# the curve's own size, leaves the curve where it was: a billionth of a log-relativity is
# nothing a chart or a price shows.
_NO_MOVE = 1e-9


def kept_inside_rules(
    model: Any, term: EditableTerm, curve: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """``curve``, edited on ``term``'s grid, as the nearest curve inside the term's rules.

    None when ``model`` is not a SuperLSS view or the term has no fit-time rule.
    The returned curve is the one the model takes when the edit is written into
    its coefficients, so it is drawn and recorded in place of ``curve``.

    On a term with levels, the base stays where it is unless the edit moves it.
    On a term that gives each level its own coefficient, an unordered
    categorical, so do the other levels the edit leaves: a level pushed past its
    cap stops at it. An edit of the base moves the rest only where their rules,
    read from the base, make them give way, and then as little as the nearest
    curve needs. On an ordered term the curve through the levels is smooth, so
    the levels the edit leaves follow the nearest curve, as on an unruled term.
    The current curve meets the rules and every hold, and the QP starts from it.
    """
    if not isinstance(model, LSSModelView):
        return None
    rows = term_rows(model, term.name)
    if rows is None:
        return None
    label, state, term_slice = _editable_slice(model, term)
    design, centre = _grid_design(model, term, label, state, term_slice)
    # A SuperLSS term opens on its native scale (``NATIVE_ONLY``), so the curve is the effect.
    curve = np.asarray(curve, dtype=np.float64)
    held = base = None
    if term.levels is not None:
        held = curve == term.edited_log_effect
        base = _base_mask(model._specs[term.name], term.levels)
    weights = editor_apply._term_weights(term)
    shift, block = _projection(
        design,
        curve + centre,
        weights,
        rows,
        term.name,
        held=held,
        base=base,
        inside=term.edited_log_effect + centre,
    )
    return design @ block + (shift - centre)


def prepare_ruled_edit(model: Any, key: str) -> None:
    """Build what an edit of the ruled term ``key`` reads, so its first edit does not wait.

    That is the view's reference row, read from every training row once, and the
    term's design at its reference level. Nothing for a term without rules, or a
    model that is not a SuperLSS view. Opening builds neither: only the terms a
    user shows pay for it, before they edit them.
    """
    if isinstance(model, LSSModelView) and term_rows(model, key) is not None:
        model.reference_row()
        model.reference_design(key)


def leaves_curve(
    kept: NDArray[np.float64], current: NDArray[np.float64], asked: NDArray[np.float64]
) -> bool:
    """Whether the rules absorbed the edit ``asked`` entirely, leaving ``current`` as ``kept``."""
    scale = max(float(np.max(np.abs(asked - current))), float(np.max(np.abs(current))))
    return bool(np.max(np.abs(kept - current)) <= _NO_MOVE * scale)


def _native_levels(spec: Any) -> dict[str, Any]:
    """Each level label the editor shows, mapped to the value the term reads it from."""
    grouping = getattr(spec, "_grouping", None)
    levels = _level_domain(spec) if grouping is None else grouping.all_original_levels
    return {str(level): level for level in levels or ()}


def _base_mask(spec: Any, shown: list[str]) -> NDArray[np.bool_] | None:
    """Which of the ``shown`` levels read the term's base level, or None for a term without one.

    A grouped term's base is a group, so each of its members reads it.
    """
    base = getattr(spec, "_base_level", None)
    if base is None:
        return None
    grouping = getattr(spec, "_grouping", None)
    if grouping is None:
        fitted = {level: level for level in shown}
    else:
        fitted = {
            str(level): str(grouping.original_to_group[level])
            for level in grouping.all_original_levels
        }
    mask = np.array([fitted.get(str(level)) == str(base) for level in shown], dtype=bool)
    return mask if mask.any() else None


def _projection(
    design: NDArray[np.float64],
    target: NDArray[np.float64],
    weights: NDArray[np.float64],
    rows: TermRows | None = None,
    name: str = "",
    *,
    held: NDArray[np.bool_] | None = None,
    base: NDArray[np.bool_] | None = None,
    inside: NDArray[np.float64] | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """The weighted least-squares fit of ``target`` by an intercept and ``design``, under ``rows``.

    The rank is decided on columns scaled to unit weighted norm and centred, under
    weights that sum to one, so neither a covariate's units and origin nor the
    weights' scale decide which directions are kept; the intercept takes the
    centring back. Rows on the coefficients are rescaled with their columns.

    ``held`` marks the levels of a level design the edit leaves, and ``base`` the
    levels that read its base level; ``_holds`` says which of them the fit keeps
    at ``target``, written on the coefficients themselves. ``inside`` is a curve
    on the same grid that meets the rows, which the QP starts from.
    """
    weights = weights / weights.sum()
    ruled = rows is not None and rows.A.shape[0] > 0
    # A rule the basis holds, a flat stretch, has no rows, and the base is held all the same.
    hold = None if rows is None else _holds(design, target, held, base)
    if rows is not None and hold is not None:
        with_intercept = np.column_stack([np.ones(design.shape[0]), design])
        A = np.column_stack([np.zeros(rows.A.shape[0]), rows.A])
        x = _ruled_fit(with_intercept, A, target, weights, rows, name, *hold, inside=inside)
        return float(x[0]), np.asarray(x[1:], dtype=np.float64)
    norm = np.sqrt(weights @ np.square(design))
    norm[norm == 0.0] = 1.0
    scaled = design / norm
    centre = weights @ scaled
    if rows is None or not ruled:
        shift, coefficients = editor_apply._solve_with_intercept(scaled - centre, target, weights)
    else:
        with_intercept = np.column_stack([np.ones(design.shape[0]), scaled - centre])
        A = np.column_stack([np.zeros(rows.A.shape[0]), rows.A / norm])
        width = with_intercept.shape[1]
        x = _ruled_fit(
            with_intercept,
            A,
            target,
            weights,
            rows,
            name,
            np.zeros(width),
            np.eye(width),
            inside=inside,
        )
        shift, coefficients = float(x[0]), np.asarray(x[1:], dtype=np.float64)
    return shift - float(centre @ coefficients), coefficients / norm


# In ``[intercept, coefficients]`` the base reads the intercept and its own
# columns. None, nothing held, when the edit moves the base or the term has
# none: every level then follows the nearest curve, and on a design with one
# coefficient per level a level the edit leaves moves only where a row on its
# coefficient, read from the moved base, makes it.
#
# With the base held, a design that gives each other level its own coefficient,
# an indicator per level with the base reading none, holds every level the edit
# leaves: a held level's coefficient is fixed and the free directions are unit
# vectors, so a row on held coefficients alone is exactly zero along them, and
# an edited level stops where its rows bind. Any other design, an ordered term's
# smooth curve through its levels, cannot move one level alone, so only the
# base is held, by its own row: the intercept moves against the columns.
def _holds(
    design: NDArray[np.float64],
    target: NDArray[np.float64],
    held: NDArray[np.bool_] | None,
    base: NDArray[np.bool_] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """The fits of a level design that keep the levels an edit leaves where they are: ``start + free z``.

    The comment above gives which designs hold which levels, and why.
    """
    if held is None or base is None or not np.all(held[base]):
        return None
    width = design.shape[1] + 1
    start = np.zeros(width)
    start[0] = target[base][0]
    indicator = design == 1.0
    one_each = np.all(indicator | (design == 0.0)) and not np.any(indicator.sum(axis=1) > 1)
    level = np.where(indicator.any(axis=1), indicator.argmax(axis=1), -1)
    if not (one_each and np.array_equal(level < 0, base)):
        row = design[np.flatnonzero(base)[0]]
        return start, np.vstack([-row[None, :], np.eye(design.shape[1])])
    own = held & (level >= 0)
    start[1 + level[own]] = target[own] - start[0]
    pinned = np.zeros(width, dtype=bool)
    pinned[0] = True
    pinned[1 + level[own]] = True
    return start, np.eye(width)[:, ~pinned]


# ``design`` and ``A`` carry the intercept as their first column. When the
# unconstrained nearest fit meets the rows it is the answer, as the QP would
# return it, so a curve already inside the rules is kept as drawn, and writing
# it into the model again returns it unchanged. Otherwise the QP finds the
# nearest fit that meets them, over the rows the free directions move; a row
# they do not move keeps the slack it has at ``start``, checked with the rest.
#
# The QP starts from the fit of ``inside``, a curve that meets the rows, when
# its coefficients do: the active-set method needs a feasible start, and its
# own repair of the unconstrained fit need not reach one. The rows that fit
# breaks are its first working rows when they are linearly independent; on a
# capped level design they are the rows the answer binds. Dependent rows, an
# ordered term's caps and steps (a cap row less the next is a step row), are
# not seeded at all: the solver would keep the first independent ones, which
# need not bind, so the outcome would hang on the order the rules are written.
def _ruled_fit(
    design: NDArray[np.float64],
    A: NDArray[np.float64],
    target: NDArray[np.float64],
    weights: NDArray[np.float64],
    rows: TermRows,
    name: str,
    start: NDArray[np.float64],
    free: NDArray[np.float64],
    *,
    inside: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """The coefficients ``x`` nearest ``target`` with ``A x >= rows.c``, among ``start + free z``.

    The comment above gives the solve and its fallbacks.
    """
    reduced = design @ free
    rest = target - design @ start
    root = np.sqrt(weights)
    weighted = reduced * root[:, None]
    z = np.zeros(free.shape[1])
    if free.shape[1]:
        z = np.linalg.lstsq(weighted, rest * root, rcond=None)[0]
    x = start + free @ z
    magnitude = max(float(np.max(np.abs(target))), float(np.max(np.abs(x))))
    if _meets_rows(A, x, rows.c, magnitude):
        return x
    moving = A @ free
    moved = np.any(moving != 0.0, axis=1)
    if free.shape[1] and np.any(moved):
        H = reduced.T @ (weights[:, None] * reduced)
        g = reduced.T @ (weights * rest)
        bound = rows.c[moved] - A[moved] @ start
        z0 = None
        if inside is not None:
            z0 = np.linalg.lstsq(weighted, (inside - design @ start) * root, rcond=None)[0]
            if not _meets_rows(A, start + free @ z0, rows.c, magnitude):
                z0 = None
        broken = _independent(moving[moved], np.flatnonzero(moving[moved] @ z < bound))
        z = _nearest_inside(H, g, moving[moved], bound, rows, name, start=z0, active=broken)
        x = start + free @ z
    if not _meets_rows(A, x, rows.c, max(magnitude, float(np.max(np.abs(x))))):
        raise EditorValueError(_OUTSIDE_RULES.format(term=name, rules=rows.text))
    return x


def _meets_rows(
    A: NDArray[np.float64], x: NDArray[np.float64], c: NDArray[np.float64], magnitude: float
) -> bool:
    """Whether ``A x >= c`` holds to the QP's own feasibility tolerance, on the fit's scale.

    Each row is read against ``max(|c_i|, |A_i| 1 magnitude)``, its action on
    coefficients the size of the fit (``magnitude``, the largest of the target and
    the coefficients, in the fit's own columns), not against its own action: on a
    flat curve every coefficient is rounding of the solve, a row's own action is
    that same rounding, and reading one by the other turns noise of either sign
    into a violation. ``_QP_FEASIBILITY_TOL`` is the relative tolerance the QP
    accepts its own answers at, and ``gamma`` covers the rounding of each row's
    dot product (Higham 2002, section 3.1). ``irls_direct._qp._binding_rows``
    reads a term the constraint holds flat on the same kind of scale.
    """
    scale = np.maximum(np.abs(c), np.abs(A).sum(axis=1) * magnitude)
    bar = (_QP_FEASIBILITY_TOL + gamma(A.shape[1] + 2)) * scale
    return bool(np.all(A @ x - c >= -bar))


def _independent(A: NDArray[np.float64], seeds: NDArray[np.intp]) -> NDArray[np.intp]:
    """``seeds`` when those rows of ``A`` are linearly independent, else none of them.

    The rows are read at unit norm, so a row's scale does not decide its rank;
    the rank is NumPy's, at its SVD tolerance.
    """
    if seeds.size < 2:
        return seeds
    unit = A[seeds] / np.linalg.norm(A[seeds], axis=1)[:, None]
    return seeds if np.linalg.matrix_rank(unit) == seeds.size else seeds[:0]


def _nearest_inside(
    H: NDArray[np.float64],
    g: NDArray[np.float64],
    A: NDArray[np.float64],
    c: NDArray[np.float64],
    rows: TermRows,
    name: str,
    *,
    start: NDArray[np.float64] | None,
    active: NDArray[np.intp],
) -> NDArray[np.float64]:
    """The minimiser of ``z' H z / 2 - g' z`` subject to ``A z >= c``, by the repository's QP.

    ``start`` meets the rows, or is None; ``active`` are the first working rows.
    An active-set step adds or drops one row, so the solve gets the room to visit
    every row twice beyond the solver's default. A seeded solve that stops short
    of its KKT certificate is retried from ``start`` without them, as the fits
    retry; one that still stops short refuses the edit, rather than keep a curve
    whose nearness is unproven.
    """
    steps = max(200, 2 * (A.shape[0] + H.shape[0]))
    result = _solve_constrained_qp_with_cold_retry(
        H, g, A, c, active.tolist(), max_iter=steps, beta_start=start
    )
    if not result.converged:
        raise EditorValueError(_OUTSIDE_RULES.format(term=name, rules=rows.text))
    return np.asarray(result.beta, dtype=np.float64)


__all__ = [
    "absorbed_message",
    "kept_inside_rules",
    "leaves_curve",
    "moves_only_empty_values",
    "lss_model_with_edits",
    "prepare_ruled_edit",
]
