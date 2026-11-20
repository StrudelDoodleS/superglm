"""Saved models from before the identifiability record had a lifecycle (``RECORD_MOVES``).

Such a pickle holds no record tag.  Loading it is one move (``migrate_saved_record``, from
``SuperGLM.__setstate__``):

- a model that kept its fit state is ``"legacy"``: its saved inference (covariance, EDF map,
  summary and metrics caches) predates the record and is dropped, and its first inference
  decides the record at the saved coefficients (``decide_at_saved_coefficients``), which it
  never moves;
- a compact one (``retain_fit_state=False``) is ``"frozen"``: it has no rows to decide from,
  so ``summary()`` and ``metrics()`` on its training rows read the inference it was saved with,
  and the summary says so (``frozen_notes``; the summary and metrics it cached are dropped);
- an edited one keeps its inference as saved, reference-only, as every edited model's is.

**The gauge of a saved singular fit.**  No gauge is published on load: the saved coefficients
already satisfy one, and the record reads the one they satisfy, in this order:

1. the gauge the fit would publish (``identifiability._construction_gauge``), when every
   functional ``g`` has ``|g'beta| <= sqrt(eps) |g|'|beta|`` (a held coordinate is exactly zero);
2. the coordinates the saved fit holds at exactly zero (a dense solver's truncation), when there
   are as many as null directions and they determine them;
3. otherwise every coordinate the null directions reach (the intercept included) has no
   estimate and no standard error (its saved value is the saved solver's split), and every
   estimable contrast keeps its own.

The first test separates rounding from a different gauge (order ``|beta|``) by orders of
magnitude.  Zeros alone do not identify a convention: when more coordinates the null
directions reach are zero than there are directions, several gauges pass the test and report
standard errors on different coefficients, so the record falls to (3) (the w5m2 gate review,
sec. 7).  The footnote names the coordinates held.  A structured fit's saved split is not
recoverable from what it saved: master's ``sz`` coefficients are not the retained factor's
pseudo-inverse solution (``|beta - H^+ H beta|`` is 4e-2 of ``|beta|``, under master's own
code), and ``N'beta`` vanishes in neither the public, the Jacobi nor the inverse Jacobi metric
(5.7e-2 to 7.0e-2 of ``|beta|``), so such a fit takes (3).

**Directions the saved coefficients were never fitted along.**  The rows identify a direction
the saved solver truncated (a near-duplicate pair), but the coefficients cannot move, so their
values there are not the penalised fit's: the coordinates such a direction reaches have no
standard error, the footnote says why, and the EDF stays the saved one.

**A saved separated fit.**  Existence is decided from the saved rows.  A verified separation
withholds its rows; the other standard errors are read from the kept rows at the saved
coefficients, with no refit, so every coefficient and prediction stays as saved, and the
footnote says the coefficients are the walk's, with the kept rows' Newton decrement there.

**Prediction disclosure follows the first inference.**  ``predict`` never builds a record (a
saved model serving quotes pays nothing for it), so a legacy model's predictions say nothing
about directions the fit cannot determine until its first ``summary()``, ``metrics()`` or
other inference has decided the record; from then on they do.  A frozen model never has one.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import SQRT_EPS

_LEGACY_CACHES = ("_fit_inference_info", "_coef_covariance", "_group_edf", "_fit_active_info")
_LEGACY_READERS = ("_fit_metrics_cache", "_fit_metrics_cache_signature", "_summary_cache")

FROZEN_NOTE = (
    "Standard errors as saved by an earlier superglm: the identifiability and separation "
    "checks need the training rows, which this model did not keep; refit with "
    "retain_fit_state=True to have them."
)


def migrate_saved_record(model) -> None:
    """``SuperGLM.__setstate__``'s part: the load move of a pickle saved before the record."""
    from superglm.model.state_ops import move_record, record_tag

    if record_tag(model) != "saved":
        return
    if model.__dict__.get("_editor_inference_stale", False):
        move_record(model, "load stale")
        return
    # what the readers rendered from the saved inference: their footnote changes now
    for name in _LEGACY_READERS:
        if name in model.__dict__:
            model.__dict__[name] = None
    if model.__dict__.get("_dm") is None:
        move_record(model, "load compact")
    else:
        for name in _LEGACY_CACHES:
            model.__dict__.pop(name, None)
        move_record(model, "load warm")


def frozen_notes(model) -> tuple[str, ...]:
    """The summary's footnote for a model whose inference is as an earlier superglm saved it."""
    from superglm.model.state_ops import record_tag

    return (FROZEN_NOTE,) if record_tag(model) == "frozen" else ()


def decide_at_saved_coefficients(model) -> None:
    """The record of a legacy model, decided at its saved coefficients (module docstring).

    Left to the saved flags, as publication leaves them: a constrained face, a SCOP fit and a
    selection penalty that can zero a group.
    """
    from superglm.model import state_ops
    from superglm.penalties.base import penalty_can_zero_groups
    from superglm.solvers._structured.state import StructuredLinearSystemState

    solver = model._solver_pirls_result()
    if model._dm is None or getattr(solver, "scop_inference", None) is not None:
        return
    if penalty_can_zero_groups(state_ops.fitted_penalty(model)):
        return
    state = state_ops.retained_linear_state(model)
    if isinstance(state, StructuredLinearSystemState):
        _decide_structured(model, state)
    elif solver.rank_info is not None and solver.rank_info.face is None:
        _decide_dense(model, solver)


def _set_rank_info(model, before, after) -> None:
    """Every result holding the rank record ``before`` holds ``after`` (results are immutable:
    each is replaced, the same object by the same copy)."""
    from superglm.model import state_ops

    copies: dict[int, Any] = {}

    def moved(result):
        if result is None or getattr(result, "rank_info", None) is not before:
            return result
        return copies.setdefault(id(result), dataclasses.replace(result, rank_info=after))

    model._result = moved(model._result)
    model._solver_result = moved(model._solver_result)
    reml = state_ops._reml_pirls(model)
    if reml is not None:
        state_ops._set_reml_pirls(model, moved(reml))


def _decide_dense(model, solver) -> None:
    from superglm.model import state_ops
    from superglm.model.separation_ops import separation_boundaries

    saved = solver.rank_info
    rank_info = _formed_at_saved(model, saved)
    _set_rank_info(model, saved, rank_info)
    solver = model._solver_pirls_result()
    boundaries = separation_boundaries(model)
    context = state_ops._publication_context(model, boundaries)
    if context is None or context.record is None:
        return
    if boundaries:
        public, reml = model._result, state_ops._reml_pirls(model)
        context = state_ops._publish_existence(
            model, context, boundaries, refit=_kept_rows_at_saved
        )
        kept = model._solver_pirls_result()
        if kept is not solver:
            # the kept rows' record on the saved results: every coefficient, its public form
            # and every fit statistic stay as saved
            model._result, model._solver_result = public, solver
            state_ops._set_reml_pirls(model, reml)
            _set_rank_info(model, rank_info, kept.rank_info)
    record = context.record
    rank_info = model._solver_pirls_result().rank_info
    columns = np.asarray(rank_info.selected_columns, dtype=np.intp)
    scale = np.ones(len(solver.beta))
    scale[columns] = np.asarray(rank_info.augmented.column_scale, dtype=np.float64)
    reach = _Reach(scale, float(rank_info.sum_w))
    record = _without_unfitted(model, record, reach)
    record = _saved_gauge(model, record, np.asarray(solver.beta, dtype=np.float64), reach)
    if context.withheld is not None:
        record = _walked(model, record, rank_info)
    state_ops._install_record(context, record)


def _decide_structured(model, state) -> None:
    from superglm.model import state_ops
    from superglm.model.separation_ops import separation_boundaries
    from superglm.solvers._structured.operators import compact_operator_diagonal

    boundaries = separation_boundaries(model)
    context = state_ops._publication_context(model, boundaries)
    if context is None or context.record is None:
        return
    if boundaries:
        # a structured fit's kept rows would need its factor formed again: the verified
        # direction is kept as a lower bound ("partial"), at the saved state
        context = state_ops._publish_existence(model, context, boundaries, refit=_no_refit)
    operator = state.centered_data_operator
    diagonal = compact_operator_diagonal(
        state_ops._spread_centred_operator(operator)
    ) + state_ops._structured_penalty_diagonal(state.penalized_operator)
    scale = np.sqrt(np.maximum(diagonal, 0.0))
    reach = _Reach(scale, float(operator.total))
    beta = np.asarray(model._solver_pirls_result().beta, dtype=np.float64)
    record = _without_unfitted(model, context.record, reach)
    record = _saved_gauge(model, record, beta, reach)
    state_ops._install_record(context, record)


def _no_refit(model, rows):
    raise NotImplementedError("a structured fit's kept rows are not formed at load")


def _formed_at_saved(model, saved):
    """The saved rank record formed again at the Fisher weights of the saved coefficients.

    Its decompositions and data Gram are those weights' (``state_ops.face_record_at``), so the
    record and the existence certificate read the matrix they decide; a fit's record was formed
    at the same weights, so its covariance is the saved one to rounding.  Its EDF stays the
    saved one, and its policy is the penalised fit's.
    """
    from superglm.model import state_ops

    formed = state_ops.face_record_at(model, state_ops._solver_space_working_weights(model))
    return dataclasses.replace(
        formed,
        feature_edf=saved.feature_edf,
        feature_edf1=saved.feature_edf1,
        group_edf=saved.group_edf,
        identifiability_policy="penalised",
        identifiability=None,
    )


def _kept_rows_at_saved(model, rows: NDArray):
    """The saved coefficients with their rank record formed on the kept rows (no refit)."""
    from superglm.model.separation_ops import WithheldFit

    solver = model._solver_pirls_result()
    model._separated_rows = np.asarray(rows, dtype=np.intp)
    return WithheldFit(
        result=dataclasses.replace(solver, rank_info=_formed_at_saved(model, solver.rank_info)),
        state=None,
        rows=np.asarray(rows, dtype=np.intp),
    )


@dataclasses.dataclass(frozen=True)
class _Reach:
    """Which coefficients a set of directions reaches, read in the Hessian's Jacobi scale.

    ``scale`` is each fitted coordinate's (one where none is recorded) and ``total`` the fit's
    weight sum, the intercept's squared scale.  A coordinate is reached when its row of an
    orthonormal basis of the scaled directions (intercept first) is above the square root of
    eps, ``identifiability._reached``'s floor.
    """

    scale: NDArray
    total: float

    def __call__(self, directions: NDArray, intercept: NDArray) -> tuple[NDArray, bool]:
        safe = np.where(self.scale > 0.0, self.scale, 1.0)
        root = math.sqrt(max(self.total, 0.0))
        scaled = np.vstack((root * np.asarray(intercept)[None, :], safe[:, None] * directions))
        if not scaled.shape[1]:
            return np.zeros(0, dtype=np.intp), False
        basis = np.linalg.qr(scaled)[0]
        reached = np.linalg.norm(basis, axis=1) > SQRT_EPS
        return np.flatnonzero(reached[1:]).astype(np.intp), bool(reached[0])


def _names(model, coordinates) -> list[str]:
    from superglm.solvers.identifiability import _coordinate_name

    specs = {**getattr(model, "_specs", {}), **getattr(model, "_interaction_specs", {})}
    owner = {c: g for g in model._groups for c in range(g.start, g.end)}
    names: list[str] = []
    for coordinate in coordinates:
        name = _coordinate_name(int(coordinate), owner[int(coordinate)], specs)
        if name not in names:
            names.append(name)
    return names


def _without_unfitted(model, record, reach: _Reach):
    """``record`` with its weak directions withheld: the saved coefficients never fitted them."""
    weak = record.weak
    if weak is None:
        return record
    coordinates, intercept = reach(np.asarray(weak.root), np.asarray(weak.intercept))
    withheld = tuple(int(c) for c in coordinates) + ((-1,) if intercept else ())
    count = int(np.shape(weak.root)[1])
    names = (["Intercept"] if intercept else []) + _names(model, coordinates)
    note = (
        f"{count} {'direction' if count == 1 else 'directions'} the saved fit's solver "
        f"truncated {'is' if count == 1 else 'are'} identified by the rows, but the saved "
        "coefficients were never fitted along "
        f"{'it' if count == 1 else 'them'}: {', '.join(names)} have no standard error "
        "(shown as ---) and the degrees of freedom are as saved; refit to estimate them."
    )
    return dataclasses.replace(
        record,
        weak=None,
        withheld=tuple(sorted(set(record.withheld) | set(withheld))),
        provenance=(*record.provenance, note),
    )


def _saved_gauge(model, record, beta: NDArray, reach: _Reach):
    """``record`` in the gauge the saved coefficients satisfy (module docstring, 1 to 3)."""
    from superglm.solvers.identifiability import Alias

    gauge = record.gauge
    if gauge is None:
        return record
    moved = gauge._moved()
    if not np.any(moved):
        return record
    null, functionals = gauge.null[:, moved], gauge.functionals[:, moved]
    k = null.shape[1]
    reached, intercept = reach(null, np.asarray(gauge.intercept)[moved])
    zeros = reached[beta[reached] == 0.0]
    ambiguous = zeros.size > k
    residual = np.abs(functionals.T @ beta)
    if not ambiguous and np.all(residual <= SQRT_EPS * (np.abs(functionals).T @ np.abs(beta))):
        held = ", ".join(alias.term for alias in record.aliases)
        note = (
            f"The saved coefficients satisfy the constraints this fit publishes ({held} held "
            "at zero), so the standard errors are given there."
        )
        return dataclasses.replace(record, provenance=(*record.provenance, note))
    # the directions held where the fit left them (separated, or not resolved) stay held; the
    # fit's intercept decision on every direction stays with them (read through ``any``)
    rest = None
    if not np.all(moved):
        rest = dataclasses.replace(
            gauge,
            null=gauge.null[:, ~moved],
            functionals=gauge.functionals[:, ~moved],
            intercept=np.asarray(gauge.intercept)[~moved],
            moved=np.zeros(int(np.count_nonzero(~moved)), dtype=bool),
        )
    others = _names(model, reached)

    def aliases(coordinates, reason) -> tuple:
        named = []
        for coordinate in coordinates:
            name = _names(model, [coordinate])[0]
            named.append(
                Alias(
                    coordinate=int(coordinate),
                    term=name,
                    aliased_with=tuple(other for other in others if other != name),
                    reason=reason,
                    source="rank",
                )
            )
        return tuple(named)

    if not ambiguous and zeros.size == k and np.linalg.matrix_rank(null[zeros]) == k:
        unit = np.zeros_like(gauge.functionals)
        unit[zeros, np.flatnonzero(moved)] = 1.0
        unit[:, ~moved] = gauge.functionals[:, ~moved]
        held = ", ".join(_names(model, zeros))
        note = (
            f"The saved coefficients hold {held} at zero, so the standard errors are given "
            "in that gauge."
        )
        return dataclasses.replace(
            record,
            gauge=dataclasses.replace(gauge, functionals=unit),
            aliases=aliases(zeros, "the saved coefficients hold it at zero"),
            provenance=(*record.provenance, note),
        )
    reason = (
        "several gauges fit the saved coefficients, so no standard error is given on any "
        "coordinate an aliased direction reaches"
        if ambiguous
        else "its saved value depends on how the saved solver split an aliased direction"
    )
    provenance = record.provenance
    if intercept:
        note = (
            "The intercept has no standard error either, nor an estimate (both shown as ---): "
            "the aliased directions reach it."
        )
        provenance = (*provenance, note)
    return dataclasses.replace(
        record,
        gauge=rest,
        aliases=aliases(reached, reason),
        intercept_aliased=record.intercept_aliased or intercept,
        provenance=provenance,
    )


def _walked(model, record, rank_info):
    """``record`` with the note a saved separated fit carries: its coefficients are the walk's."""
    from superglm.model import state_ops
    from superglm.model.separation_ops import fisher_rows, withheld_weights

    solver = model._solver_pirls_result()
    design, groups = state_ops._rank_active_state(model, rank_info)
    columns = np.asarray(rank_info.selected_columns, dtype=np.intp)
    W, residual = fisher_rows(model, solver, withheld_weights(model))
    r = W * residual
    total = float(np.sum(r))
    slopes = np.asarray(solver.beta, dtype=np.float64)[columns]
    score = (
        np.asarray(design.rmatvec(r), dtype=np.float64)
        - np.asarray(rank_info.mean_x, dtype=np.float64)[columns] * total
        - state_ops._rank_curvature(model, groups) @ slopes
    )
    inverse = record.published_covariance(rank_info.augmented.pseudo_inverse(), columns)
    # ``g' H^+ g`` with the intercept's own term, in the fit's units (``weight_exponent``)
    decrement = float(score @ inverse @ score) + total * total / float(rank_info.sum_w)
    decrement = float(np.ldexp(decrement, -int(solver.weight_exponent)))
    note = (
        "The saved coefficients are the separated fit's walk, kept as saved: no fit on the "
        "kept rows was run, so the standard errors are read from the kept rows at the saved "
        f"coefficients, whose Newton decrement there is {decrement:.3g} (zero at the kept "
        "rows' own fit)."
    )
    return dataclasses.replace(record, provenance=(*record.provenance, note))
