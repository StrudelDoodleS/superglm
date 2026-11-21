"""Build-time detection of separated categorical cells and levels.

A categorical level, or crossed-categorical cell, with exposure whose responses all
sit on the response distribution's boundary (all ``y == 0`` for a log-link Poisson /
Tweedie / negative-binomial fit; all 0 or all 1 for binomial) has no finite
maximum-likelihood estimate: IRLS drifts until the objective stagnates and returns
fitted values collapsed to the boundary, while rank metrics still look healthy. This is
nonexistence of the exponential-family MLE (Haberman 1974; Fienberg and Rinaldo 2012,
*Ann. Statist.* 40(2); Albert and Anderson 1984, *Biometrika* 71(1) for binomial). In
general it is a linear program (Konis 2007), but for indicator blocks the directions of
recession are exactly the cells with exposure and boundary-only response, so the O(n)
scan is exact. It runs at design-build time on the dense and ``discrete=True`` paths. A
term bounded by an active selection penalty, or a ``Categorical`` under a fit-time cap, is
skipped (the bounded optimum is finite); only ``selection_penalty`` is consulted, since
``Categorical`` blocks carry no ``lambda2``. A SuperGLM fit call states what it found once,
when it returns, priced by the fit it publishes (``disclosure_scope``).
"""

from __future__ import annotations

import contextlib
import sys
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import gamma
from superglm._warn import _PACKAGE_PREFIX, warn_caller

_CONTEXTLIB = contextlib.__file__

#: Working-weight ratio past which an exhausted, stagnant IRLS run is treated
#: as the runtime signature of separation (see ``format_runtime_message``).
EXTREME_WEIGHT_RATIO = 1e12

#: Relative deviance change below which the objective counts as stagnant for
#: the runtime backstop.  Deliberately far below any convergence tolerance:
#: separation plateaus print ``delta=0.00e+00`` while coefficients still walk.
STAGNANT_DEVIANCE_DELTA = 1e-10


class SeparationWarning(UserWarning):
    """A term contains cells whose maximum-likelihood estimate is infinite."""


class SeparationError(ValueError):
    """Refusal to fit (or certify) a design containing separated cells.

    ``levels`` is true when the finding names factor levels or crossed cells
    (the build-time scans), whose remedy is grouping them; an intercept-only
    finding and the runtime backstop's, which names model terms, are not.
    """

    def __init__(self, *args: object, levels: bool = False) -> None:
        super().__init__(*args)
        self.levels = levels


# What a public fit call says about separation.  A call can build its design
# several times (a theta or power search, then its publication refit) and run
# many coefficient solves; it names each design finding once, and the runtime
# finding of the state it publishes only.

#: The design notices this public fit call has emitted; ``None`` outside one.
_DISCLOSED: ContextVar[set[str] | None] = ContextVar("superglm_separation_disclosed", default=None)

#: The separated levels and cells this public fit call's builds found, held for
#: the one disclosure it makes when it returns; ``None`` outside one.
_PENDING: ContextVar[list[SeparatedLevels] | None] = ContextVar(
    "superglm_separation_pending", default=None
)


@contextmanager
def disclosure_scope(model: Any) -> Iterator[None]:
    """``model``'s public fit call: each distinct separation notice inside it is emitted once.

    Each call opens its own scope, restoring the enclosing one on exit, so an
    independent fit started inside another call (from a progress callback)
    reports what it finds.  A fit the enclosing call runs and discards
    (``_discarded_fit``: a power-search candidate's ``fit_reml``, a ``drop1``
    comparison fit) shares that call's scope.

    The separated levels and cells its builds find are held for one disclosure.
    The fit the call publishes states them, priced, before it is installed
    (``take_held_levels``, read by ``fit_state.capture_fit_state``), so a warning
    raised as an error leaves the previous fit in place.  A call that publishes
    none (``drop1``), or fails, states them as its builds counted them when it
    returns.
    """
    if getattr(model, "_discarded_fit", False) and _DISCLOSED.get() is not None:
        yield
        return
    token = _DISCLOSED.set(set())
    held: list[SeparatedLevels] = []
    pending_token = _PENDING.set(held)
    try:
        yield
    finally:
        _PENDING.reset(pending_token)
        _DISCLOSED.reset(token)
        if held:
            lines = [line for line, _, _ in held_lines(held)]
            message = format_separated_levels(lines, example=lines_example(lines, held[-1].specs))
            warnings.warn(message, SeparationWarning, stacklevel=_caller_level())


def _caller_level() -> int:
    """``warnings.warn``'s ``stacklevel``, from its caller, of the first frame outside superglm.

    ``contextlib``'s frames are stepped over too, since a scope's ``finally`` runs
    under its ``__exit__``.  Counted here: ``skip_file_prefixes`` does not step over
    that frame for a generator's warning (CPython 3.13 lands it in ``contextlib``).
    """
    frame, level = sys._getframe(2), 2
    while frame is not None and frame.f_code.co_filename.startswith((_PACKAGE_PREFIX, _CONTEXTLIB)):
        frame, level = frame.f_back, level + 1
    return level


def take_held_levels() -> list[SeparatedLevels]:
    """What this fit call's builds found separated, handed to the fit it publishes; then none."""
    held = _PENDING.get()
    if not held:
        return []
    taken = list(held)
    held.clear()
    return taken


def warn_separation(message: str, *, stacklevel: int) -> None:
    """Warn ``message`` at the caller's ``stacklevel``, unless this fit call already has."""
    disclosed = _DISCLOSED.get()
    if disclosed is not None:
        if message in disclosed:
            return
        disclosed.add(message)
    warnings.warn(message, SeparationWarning, stacklevel=stacklevel + 1)


def disclose_runtime_separation(message: str | None, mode: str) -> None:
    """Warn, or refuse under ``"error"``, with the runtime finding of a fit the caller receives.

    The solvers record the backstop's finding on the state they return
    (``PIRLSResult.separation_message``) and warn about nothing: a REML
    candidate, a theta or power evaluation, a line-search trial or a discarded
    refit is evidence about that solve, not about the fit the caller receives.
    """
    if message is None or mode == "ignore":
        return
    if mode == "error":
        raise SeparationError(message)
    held = _HELD.get()
    if held is not None:
        held.append(message)
        return
    warn_caller(message, SeparationWarning)


#: The runtime warnings a caller holds back to decide on later; ``None`` when none does.
_HELD: ContextVar[list[str] | None] = ContextVar("superglm_separation_held", default=None)


@contextmanager
def held_runtime_warnings() -> Iterator[list[str]]:
    """Collect, rather than emit, the runtime warnings of the fits run inside.

    For a caller that revises the fit before returning it (the editor's Final
    fit carrying edits), which then warns about what it returns.  A refusal
    under ``"error"`` is not held.
    """
    held: list[str] = []
    token = _HELD.set(held)
    try:
        yield held
    finally:
        _HELD.reset(token)


def refuses_on_publication(model: Any) -> bool:
    """Whether a runtime finding on the fit ``model`` is solving now refuses it.

    Under ``"error"``, a fit the caller receives is refused as soon as its
    state is known, before anything else describes it.  A discarded fit
    (``_discarded_fit``) and a refit of an NB2 theta alternation, which a later
    refit may supersede, are decided where the call publishes
    (``capture_fit_state``).
    """
    return (
        getattr(model, "_separation", "warn") == "error"
        and not getattr(model, "_discarded_fit", False)
        and not getattr(model, "_nb_theta_search", False)
    )


def refuse_published_separation(model: Any, result: Any) -> None:
    """Refuse ``result``, the state ``model``'s fit returns, when it carries a finding."""
    message = getattr(result, "separation_message", None)
    if message is not None and refuses_on_publication(model):
        raise SeparationError(message)


def disclose_solver_separation(message: str | None, mode: str) -> None:
    """An exported optimizer's own disclosure of the state it returns, as a solver's.

    ``"warn"`` warns at the optimizer's caller, ``"error"`` refuses, and
    ``"ignore"`` and ``"record"`` say nothing.
    """
    if mode != "record":
        disclose_runtime_separation(message, mode)


def format_path_runtime_message(lambdas: NDArray, findings: Sequence[str | None]) -> str | None:
    """One statement for a regularization path: the points whose solve has a runtime finding."""
    hit = [i for i, finding in enumerate(findings) if finding is not None]
    if not hit:
        return None
    points = ", ".join(f"{float(lambdas[i]):.4g}" for i in hit)
    return (
        f"fit_path: {len(hit)} of {len(findings)} path points (lambda {points}) carry the "
        f"runtime separation finding. At lambda {float(lambdas[hit[0]]):.4g}: {findings[hit[0]]}"
    )


def format_drop1_runtime_message(findings: Sequence[tuple[str, str]]) -> str | None:
    """One statement for ``drop1``: the dropped features whose comparison fit has a runtime finding."""
    if not findings:
        return None
    if len(findings) == 1:
        opening = (
            f"the comparison fit without {findings[0][0]!r} carries the runtime separation "
            "finding, so the deviance change and test for dropping it compare"
        )
    else:
        names = " and ".join(f"without {name!r}" for name, _ in findings)
        opening = (
            f"the comparison fits {names} each carry the runtime separation finding, so the "
            "deviance change and test for dropping each compare"
        )
    return (
        f"drop1: {opening} with a fit that is not a maximum-likelihood fit. "
        f"Without {findings[0][0]!r}: {findings[0][1]}"
    )


@dataclass(frozen=True)
class SeparatedTerm:
    """Separated cells found in one categorical or crossed-categorical term."""

    term: str
    kind: str  # "levels" (main effect), "cells" (crossed) or "intercept"
    boundary: str  # "zero" or "one"
    labels: list[Any]  # level names, or (level1, level2) pairs
    n_occupied: int  # occupied levels/cells scanned in this term
    # Each label's positive-weight training rows, in label order: what the
    # disclosure counts and prices. Only a build's own findings carry them.
    rows: tuple[NDArray, ...] = field(default=(), compare=False, repr=False)
    # A crossed term's occupied cells (positive weight), base row and column
    # included: the design rows its readers test contrasts against.
    occupied: tuple = field(default=(), compare=False, repr=False)


def response_boundaries(distribution: Any, link: Any) -> tuple[str, ...]:
    """Boundaries of the response support reachable only at infinite eta.

    Returns a subset of ``("zero", "one")``.  ``"zero"`` means the fitted mean
    reaches 0 only as ``eta -> -inf`` and the distribution puts positive mass
    on ``y == 0``; ``"one"`` is the binomial upper boundary at ``eta -> +inf``.
    An empty tuple disables the separation scan for this family/link: with the
    boundary at finite eta (identity, sqrt) the MLE sits on the parameter-space
    boundary instead of escaping to infinity, which is a different (bounded)
    failure mode.
    """
    from superglm.distributions import (
        Binomial,
        NegativeBinomial,
        Poisson,
        Tweedie,
    )
    from superglm.links import CauchitLink, CloglogLink, LogitLink, LogLink, ProbitLink

    if isinstance(distribution, Binomial):
        if isinstance(link, LogitLink | ProbitLink | CloglogLink | CauchitLink):
            return ("zero", "one")
        if isinstance(link, LogLink):
            return ("zero",)
        return ()
    mass_at_zero = isinstance(distribution, Poisson | NegativeBinomial) or (
        isinstance(distribution, Tweedie) and 1.0 <= float(distribution.p) < 2.0
    )
    if mass_at_zero and isinstance(link, LogLink):
        return ("zero",)
    return ()


def _level_codes(
    x: NDArray, spec: Any, *, context: str, codes: NDArray | None = None
) -> tuple[NDArray, list[Any]]:
    """Row codes against a built Categorical spec's full level universe.

    Applies the same raw -> collapsed -> fitted-domain label contract the
    interaction builder uses, then maps rows an ``unseen="base"`` policy
    routes to the base level onto the base index so their exposure and
    response anchor the cell they actually train in.

    ``codes``, when given, are the codes the term's own build already made
    for these rows (``Categorical._build_with_codes``): the same labels
    against the same universe, so coding the column again would only repeat
    them.
    """
    levels = list(spec._levels)
    if codes is not None:
        codes = codes.astype(np.intp, copy=False)
    else:
        from superglm.features.categorical import _codes_against
        from superglm.features.interaction import _categorical_build_labels

        labels = _categorical_build_labels(x, spec, context=context)
        codes = _codes_against(labels, levels)
    if getattr(spec, "unseen", "error") == "base" and (codes < 0).any():
        base_index = levels.index(spec._base_level)
        codes = np.where(codes < 0, base_index, codes)
    return codes, levels


def _separated_flags(
    codes: NDArray,
    n_cells: int,
    y: NDArray,
    sample_weight: NDArray,
    boundary: str,
) -> tuple[NDArray, int]:
    """Per-cell separation flags and the occupied-cell count.

    A cell separates when it has positive effective exposure and no
    positive-weight row off the boundary.  Zero-weight rows contribute no
    likelihood, so they neither occupy nor anchor a cell.
    """
    valid = (codes >= 0) & (sample_weight > 0)
    exposure = np.bincount(codes[valid], weights=sample_weight[valid], minlength=n_cells)
    off_boundary = (y > 0) if boundary == "zero" else (y < 1)
    anchor = valid & off_boundary
    anchored = np.bincount(codes[anchor], weights=sample_weight[anchor], minlength=n_cells) > 0
    occupied = exposure > 0
    return occupied & ~anchored, int(np.count_nonzero(occupied))


def _flagged_rows(codes: NDArray, flags: NDArray, sample_weight: NDArray) -> tuple[NDArray, ...]:
    """Each flagged cell's positive-weight rows, in cell order, in one pass over the rows."""
    hit = np.flatnonzero((codes >= 0) & (sample_weight > 0))
    hit = hit[flags[codes[hit]]]
    hit = hit[np.argsort(codes[hit], kind="stable")]
    cells = np.flatnonzero(flags)
    sorted_codes = codes[hit]
    starts = np.searchsorted(sorted_codes, cells, side="left")
    ends = np.searchsorted(sorted_codes, cells, side="right")
    return tuple(hit[start:end] for start, end in zip(starts, ends, strict=True))


def scan_categorical_term(
    name: str,
    spec: Any,
    x: NDArray,
    y: NDArray,
    sample_weight: NDArray,
    boundaries: tuple[str, ...],
    build_codes: NDArray | None = None,
    *,
    include_base: bool = True,
) -> list[SeparatedTerm]:
    """Scan one main-effect Categorical term for separated levels.

    A level with its own indicator separates along that coordinate. The base
    level needs an intercept that can walk against every non-base indicator.
    ``build_codes`` are the term's build codes, when the caller holds them;
    the scan reads them instead of coding the column again.
    """
    codes, levels = _level_codes(x, spec, context=name, codes=build_codes)
    findings: list[SeparatedTerm] = []
    for boundary in boundaries:
        flags, n_occupied = _separated_flags(codes, len(levels), y, sample_weight, boundary)
        if not include_base:
            flags[levels.index(spec._base_level)] = False
        if flags.any():
            findings.append(
                SeparatedTerm(
                    term=name,
                    kind="levels",
                    boundary=boundary,
                    labels=[levels[i] for i in np.flatnonzero(flags)],
                    n_occupied=n_occupied,
                    rows=_flagged_rows(codes, flags, sample_weight),
                )
            )
    return findings


def scan_interaction_term(
    name: str,
    spec1: Any,
    spec2: Any,
    x1: NDArray,
    x2: NDArray,
    p1: str,
    p2: str,
    y: NDArray,
    sample_weight: NDArray,
    boundaries: tuple[str, ...],
) -> list[SeparatedTerm]:
    """Scan one CategoricalInteraction term for separated crossed cells.

    The scan covers the full ``L1 x L2`` grid, base rows and columns
    included: with both mains and the full non-base interaction grid in the
    design, the spanned space contains every single-cell indicator, so any
    occupied cell with boundary-only response is a direction of recession.
    Cells the builder prunes (empty, or aliased under an exactly nested
    parent) still separate through the parent main effect, so pruning does
    not exempt them.
    """
    codes1, levels1 = _level_codes(x1, spec1, context=p1)
    codes2, levels2 = _level_codes(x2, spec2, context=p2)
    n2 = len(levels2)
    on_grid = (codes1 >= 0) & (codes2 >= 0)
    cell_codes = np.where(on_grid, codes1 * n2 + codes2, -1)
    weighted = on_grid & (sample_weight > 0)
    occupied = tuple(
        (levels1[i // n2], levels2[i % n2])
        for i in np.flatnonzero(np.bincount(cell_codes[weighted], minlength=len(levels1) * n2))
    )
    findings: list[SeparatedTerm] = []
    for boundary in boundaries:
        flags, n_occupied = _separated_flags(
            cell_codes, len(levels1) * n2, y, sample_weight, boundary
        )
        if flags.any():
            labels = [(levels1[i // n2], levels2[i % n2]) for i in np.flatnonzero(flags)]
            findings.append(
                SeparatedTerm(
                    term=name,
                    kind="cells",
                    boundary=boundary,
                    labels=labels,
                    n_occupied=n_occupied,
                    rows=_flagged_rows(cell_codes, flags, sample_weight),
                    occupied=occupied,
                )
            )
    return findings


_MAX_LISTED_LABELS = 15


def _format_labels(labels: list[Any]) -> str:
    shown = labels[:_MAX_LISTED_LABELS]
    parts = [
        f"({item[0]!r} x {item[1]!r})" if isinstance(item, tuple) else repr(item) for item in shown
    ]
    text = ", ".join(parts)
    if len(labels) > _MAX_LISTED_LABELS:
        text += f", ... and {len(labels) - _MAX_LISTED_LABELS} more"
    return text


def format_separation_message(findings: list[SeparatedTerm]) -> str:
    """One actionable message naming every separated cell and the remedies."""
    n_total = sum(len(f.labels) for f in findings)
    lines = [
        f"Separation detected: {n_total} categorical cell(s) have exposure but only "
        "boundary response values, so their maximum-likelihood effects are infinite. "
        "IRLS will drift until the objective stagnates and the affected fitted values "
        "collapse to the boundary instead of converging. Rank and aggregate metrics "
        "(gini, balance) will look healthy on such a fit; only out-of-sample "
        "likelihood/deviance exposes it."
    ]
    for f in findings:
        response = "no positive response" if f.boundary == "zero" else "all-one response"
        unit = "occupied cells" if f.kind == "cells" else "occupied levels"
        lines.append(
            f"  {f.term!r}: {len(f.labels)} of {f.n_occupied} {unit} have exposure "
            f"but {response}: {_format_labels(f.labels)}"
        )
    lines.append(
        "Remedies: collapse the affected levels into neighbours "
        "(collapse_levels / Categorical(grouping=...)), model the crossed factor "
        "with RandomEffect (its ridge bounds every cell), or target the term with "
        "a selection penalty. Pass separation='error' to refuse such designs, or "
        "separation='ignore' to disable this check."
    )
    return "\n".join(lines)


# ── A SuperGLM fit's one disclosure of its separated levels and cells ───────


@dataclass(frozen=True)
class SeparatedLevels:
    """What one design build found separated, with the rows the disclosure counts.

    ``specs`` holds every term of the build by name, main effects and
    interactions, and ``frame`` the build's training frame, which the disclosure
    scores.  ``exposure`` says the build had an offset under a log link, whose
    exponent is a row's exposure.  Totals weight each row by its
    ``sample_weight``, whatever its semantics; exposure counts each row once.
    """

    findings: list[SeparatedTerm]
    specs: dict[str, Any]
    sample_weight: NDArray
    offset: NDArray | None
    exposure: bool
    frame: Any


@dataclass(frozen=True)
class SeparatedLine:
    """What the disclosure says about one separated level or crossed cell."""

    term: str
    kind: str  # "levels" or "cells"
    label: Any  # a level, or a cell's (level1, level2)
    boundary: str  # "zero" or "one"
    rows: int
    # The rows' weight total when their weights are not all one, and their exposure
    # (exp(offset) summed, each row once) under a log link with an offset.
    weight: float | None
    exposure: float | None
    base: Any  # the base level, or the base cell
    # Read from a published fit: the response total the rows expect at the base's
    # rate, when the base is priced (not for the base itself, nor a separated base,
    # nor when the move reaches an unoccupied crossed cell), and the fitted response
    # total over the same rows, both weighted by ``sample_weight``; and a level's
    # fitted factor against its base on a log link when no crossed term involves it.
    expected: float | None = None
    fitted: float | None = None
    factor: float | None = None

    @property
    def is_base(self) -> bool:
        return bool(self.label == self.base)


def level_lines(found: SeparatedLevels) -> list[SeparatedLine]:
    """One line per separated label of ``found``, with what its build counted."""
    lines = []
    for finding in found.findings:
        spec = found.specs[finding.term]
        cells = finding.kind == "cells"
        base = (spec._base1, spec._base2) if cells else spec._base_level
        for label, rows in zip(finding.labels, finding.rows, strict=True):
            weights = found.sample_weight[rows]
            # a total past float64's range is stated so (``_total``), never raised
            with np.errstate(over="ignore"):
                weight = None if np.all(weights == 1.0) else float(np.sum(weights))
                exposure = (
                    float(np.sum(np.exp(found.offset[rows])))
                    if found.exposure and found.offset is not None
                    else None
                )
            lines.append(
                SeparatedLine(
                    term=finding.term,
                    kind=finding.kind,
                    label=label,
                    boundary=finding.boundary,
                    rows=len(rows),
                    weight=weight,
                    exposure=exposure,
                    base=base,
                )
            )
    return lines


def held_lines(
    held: Sequence[SeparatedLevels],
) -> list[tuple[SeparatedLine, SeparatedLevels, NDArray]]:
    """Each label the builds found, once: as the latest build counted it, that build, its rows."""
    latest: dict[tuple, tuple[SeparatedLine, SeparatedLevels, NDArray]] = {}
    for found in held:
        rows = (part for finding in found.findings for part in finding.rows)
        for line, part in zip(level_lines(found), rows, strict=True):
            latest[(line.term, line.kind, line.label)] = (line, found, part)
    return list(latest.values())


def grouping_example(term: str, spec: Any, separated: Sequence[Any]) -> str | None:
    """A call that groups ``term``'s first separated level with its nearest priced level.

    Nearest in the term's level order, which is the data's order for an ordered
    factor, among the levels with weight in the fit: a declared level the data
    never hold (``_pinned_levels``) is not one ``collapse_levels`` can group.
    ``None`` when the term already has a grouping (its levels are groups then)
    or no level is priced.
    """
    if getattr(spec, "_grouping", None) is not None:
        return None
    levels = list(spec._levels)
    unpriced = {*separated, *(getattr(spec, "_pinned_levels", ()) or ())}
    priced = [i for i, level in enumerate(levels) if level not in unpriced]
    for label in separated:
        if label not in levels or not priced:
            continue
        here = levels.index(label)
        near = min(priced, key=lambda i: (abs(i - here), i))
        members = [str(levels[i]) for i in sorted((here, near))]
        return (
            f"Categorical(grouping=collapse_levels(X[{term!r}], "
            f"groups={{{'+'.join(members)!r}: {members!r}}}))"
        )
    return None


def lines_example(lines: Sequence[SeparatedLine], specs: dict[str, Any]) -> str | None:
    """``grouping_example`` for the first term of ``lines`` that has one."""
    for line in lines:
        if line.kind == "levels" and line.term in specs:
            held = [other.label for other in lines if other.term == line.term]
            example = grouping_example(line.term, specs[line.term], held)
            if example is not None:
                return example
    return None


def _counted(n: int, noun: str) -> str:
    return "" if n == 0 else f"{n} {noun}" + ("" if n == 1 else "s")


def _total(value: float, digits: str) -> str:
    """A stated total, or "beyond float64" for one float64 cannot hold."""
    return format(value, digits) if np.isfinite(value) else "beyond float64"


def _line_text(line: SeparatedLine, *, parameter_effects: bool = False) -> str:
    if line.kind == "cells":
        name = f"{line.term!r} cell {line.label[0]!r} x {line.label[1]!r}"
        base = f"base cell {line.base[0]!r} x {line.base[1]!r}"
    else:
        name, base = f"{line.term!r} level {line.label!r}", f"base level {line.base!r}"
    every = "0" if line.boundary == "zero" else "1"
    counted = f"{line.rows} rows"
    if line.weight is not None:
        counted += f", weight {_total(line.weight, '.4g')}"
    if line.exposure is not None:
        counted += f", exposure {_total(line.exposure, '.4g')}"
    text = f"  {name}: {counted}, every response {every}"
    if line.is_base:
        effects = "effects" if parameter_effects else "relativities"
        return f"{text}; it is the {base}, which the term's other {effects} are measured against"
    if line.fitted is None:
        return text
    totals = [f"fitted {_total(line.fitted, '.3g')} in total"]
    if line.expected is not None:
        totals.insert(0, f"{_total(line.expected, '.3g')} expected in total at the rate of {base}")
    weighted = "weighted by sample_weight, " if line.weight is not None else ""
    factor = "" if line.factor is None else f"; fitted factor {_total(line.factor, '.3g')}"
    return f"{text}; {weighted}{', '.join(totals)}{factor}"


def _rank(line: SeparatedLine) -> tuple[float, float]:
    """Most expected first, then most exposed: the order a term's lines are listed in."""
    size = next(
        (value for value in (line.exposure, line.weight) if value is not None), float(line.rows)
    )
    expected = 0.0 if line.expected is None or np.isnan(line.expected) else line.expected
    return (-expected, -size)


def format_separated_levels(
    lines: Sequence[SeparatedLine],
    *,
    mode: str = "warn",
    fitted: bool = False,
    example: str | None = None,
    parameter_effects: bool = False,
) -> str:
    """The disclosure of separated levels and cells, one line each, and what to do.

    ``fitted`` says the lines carry a published fit's numbers; a refusal
    (``mode="error"``) and a build outside a fit call carry what the build counted.
    Every line is listed: each names a price that is not an estimate.
    """
    n_levels = sum(line.kind == "levels" for line in lines)
    what = " and ".join(
        part
        for part in (
            _counted(n_levels, "categorical level"),
            _counted(len(lines) - n_levels, "crossed cell"),
        )
        if part
    )
    one = len(lines) == 1
    boundaries = {line.boundary for line in lines}
    every = (
        "all 0, or all 1" if len(boundaries) > 1 else ("all 0" if "zero" in boundaries else "all 1")
    )
    opening = (
        f"{what} {'has' if one else 'have'} rows whose responses are {every}, so the data give "
        f"{'its' if one else 'their'} effect no finite estimate."
    )
    if parameter_effects:
        opening += " Each affected effect is where the fit stopped."
        if n_levels:
            opening += (
                " Inference flags the separated levels and limits their terms' joint tests "
                "to the identified contrasts."
            )
    elif fitted:
        opening += (
            f" The fitted price for {'it' if one else 'each'} is where the fit stopped, not an "
            "estimate, and predict, summary() and the rating tables carry it; summary() leaves "
            "its standard error and p-value blank."
        )
    else:
        opening += f" A fit prices {'it' if one else 'each'} where it stops, not at an estimate."
    body = []
    by_term: dict[str, list[SeparatedLine]] = {}
    for line in lines:
        by_term.setdefault(line.term, []).append(line)
    for term_lines in by_term.values():
        body.extend(
            _line_text(line, parameter_effects=parameter_effects)
            for line in sorted(term_lines, key=_rank)
        )
    group = "group a level with similar levels " + (
        f"({example})" if example else "(collapse_levels builds the grouping)"
    )
    random_effect = "fit the term as a RandomEffect"
    if parameter_effects:
        remedies = f"{group} or {random_effect}"
    elif n_levels:
        remedies = (
            f"{group}, {random_effect}, or bound it with Constraint.fit.cap "
            "or a selection penalty on the term"
        )
    else:
        remedies = f"{group}, {random_effect}, or put a selection penalty on the crossed term"
    remedy = f"To price {'it' if one else 'them'}: {remedies}."
    if mode == "error":
        remedy += (
            " Pass separation='warn' to fit with this as a warning, or separation='ignore' to "
            "fit without it."
        )
    else:
        remedy += " separation='error' refuses such fits; separation='ignore' silences this."
    return "\n".join([opening, *body, remedy])


def runtime_finding(
    design: Any,
    groups: list,
    beta: NDArray,
    *,
    ratio: float,
    n_iter: int,
    exhausted_stagnant: bool,
    eta: NDArray,
    eta_unclipped: NDArray,
    weights: NDArray,
) -> RuntimeFinding | None:
    """The runtime backstop's finding on one coefficient state, or ``None`` (issue #341).

    Under an extreme working-weight range (``ratio``), two terminal signatures
    mark a coefficient walking toward +/-infinity: the linear predictor pinned
    at the link's guard on rows that carry weight, and an exhausted budget at a
    stagnant deviance (``exhausted_stagnant``).  The guard is read as it
    stands, offset included, so a row its offset alone holds there counts too.
    """
    if not ratio > EXTREME_WEIGHT_RATIO:
        return None
    pinned = bool(np.any((eta != eta_unclipped) & (weights > 0)))
    if not (pinned or exhausted_stagnant):
        return None
    drifting = drifting_groups(design, groups, beta, eta, eta_unclipped, weights)
    return RuntimeFinding(format_runtime_message(ratio, n_iter, drifting, pinned), tuple(drifting))


def drifting_groups(
    design: Any,
    groups: list,
    beta: NDArray,
    eta: NDArray,
    eta_unclipped: NDArray,
    weights: NDArray,
) -> list[str]:
    """The groups that carry the linear predictor to where the walk stopped.

    The rows are those pinned at the link's guard; with none pinned, the rows whose predictor
    lies farthest from the bulk (at least half as far as the farthest). A group is scored by
    how far its own contribution moves those rows from where it puts the bulk (its median over
    rows whose weight did not stop). Coefficient size is not comparable across groups (each is
    in its own reparametrised scale) and the raw contribution includes a constant the intercept
    cancels. Groups reaching at least half the farthest are named, at most five.
    """
    live = weights > 0
    rows = np.flatnonzero((eta != eta_unclipped) & live)
    if not rows.size and np.any(live):
        spread = np.abs(eta_unclipped - np.median(eta_unclipped[live]))
        spread[~live] = 0.0
        top = float(np.max(spread))
        rows = np.flatnonzero(spread >= 0.5 * top) if top > 0.0 else rows
    if not rows.size:
        return []
    rest = live.copy()
    rest[rows] = False
    bulk = rest if rest.any() else live
    reach = []
    for group, matrix in zip(groups, design.group_matrices, strict=True):
        coefficients = beta[group.sl]
        if not coefficients.size or group.scop_reparameterization is not None:
            continue
        contribution = np.asarray(matrix.matvec(coefficients), dtype=np.float64)
        moved = contribution[rows] - np.median(contribution[bulk])
        reach.append((float(np.max(np.abs(moved))), group.name))
    farthest = max((value for value, _ in reach), default=0.0)
    if farthest <= 0.0:
        # No group moves the stopped rows from the rest: none is named.
        return []
    ranked = sorted(reach, key=lambda item: -item[0])
    return [name for value, name in ranked if value >= 0.5 * farthest][:5]


@dataclass(frozen=True)
class RuntimeFinding:
    """The runtime backstop's statement, and the groups it names, farthest first.

    A solve records both on the state it returns (``PIRLSResult.separation_message``
    and ``separation_groups``).
    """

    message: str
    groups: tuple[str, ...]


def restates_separated_level(groups: Sequence[str], separated: set[str]) -> bool:
    """Whether a runtime finding is the walk of levels or cells the fit already discloses.

    True when every group the finding names (``RuntimeFinding.groups``) is a term in
    ``separated``, the terms a recorded separated level or cell belongs to.
    """
    return bool(groups) and set(groups) <= separated


def format_runtime_message(
    w_ratio: float,
    n_iter: int,
    drifting_groups: list[str],
    pinned: bool,
) -> str:
    """Terminal-state message for the in-solver backstop (issue #341)."""
    where = ""
    if drifting_groups:
        where = f" (the rows it reached are carried there by group(s): {drifting_groups})"
    if pinned:
        how = (
            f"IRLS stopped after {n_iter} iterations with the linear predictor pinned "
            f"at the link's overflow guard on rows that carry weight and an extreme "
            f"working-weight range (ratio {w_ratio:.1e}): the walk was stopped by "
            f"the guard, not by the likelihood."
        )
    else:
        how = (
            f"IRLS exhausted its iteration budget ({n_iter} iterations) with a "
            f"stagnant deviance and an extreme working-weight range "
            f"(ratio {w_ratio:.1e})."
        )
    return (
        f"{how} This is the signature of separation -- one or more coefficients "
        f"are drifting to +/-infinity and the affected fitted values have collapsed "
        f"to the response boundary{where}. The returned coefficients are not "
        "maximum-likelihood estimates and the collapsed cells' predictions are "
        "unusable, even though rank and aggregate metrics will look healthy. The "
        "build-time check (separation='warn', the default) names separated "
        "categorical cells, and holds spline-by-factor curves over separated "
        "ranges, before fitting; separation this check reports at runtime "
        "involves structure that check cannot see. Remove it (collapse levels, "
        "RandomEffect, or a selection penalty on the term), or pass "
        "separation='ignore' to silence this."
    )


def emit_separation_findings(found: SeparatedLevels, mode: str) -> None:
    """Refuse under ``"error"``; otherwise hold the finding for the fit call's one disclosure.

    Held in the public call's ``disclosure_scope`` and disclosed when it returns,
    priced by the fit it published; a build outside any such call warns now,
    with what the build counted.
    """
    if not found.findings or mode == "ignore":
        return
    held = _PENDING.get()
    if mode != "error" and held is not None:
        held.append(found)
        return
    lines = level_lines(found)
    message = format_separated_levels(lines, mode=mode, example=lines_example(lines, found.specs))
    if mode == "error":
        raise SeparationError(message, levels=True)
    # stacklevel points past the builder into the user's fit call region; the
    # exact frame depth varies by entrypoint, so the message carries the term
    # names rather than relying on the reported line.
    warn_separation(message, stacklevel=3)


def emit_curve_separation(curves: list, mode: str) -> None:
    """Warn, or refuse, for spline-by-factor curves held over a separated range."""
    if not curves or mode == "ignore":
        return
    message = format_curve_separation(curves)
    if mode == "error" and any(curve.refusable for curve in curves):
        raise SeparationError(message, levels=True)
    warn_separation(message, stacklevel=3)


def validate_separation_mode(mode: str) -> str:
    """Return ``mode`` unchanged when it names a supported separation policy."""

    if mode not in ("warn", "error", "ignore"):
        raise ValueError(f"separation must be 'warn', 'error', or 'ignore', got {mode!r}")
    return mode


def validate_solver_separation_mode(mode: str) -> str:
    """``validate_separation_mode`` for an exported solver, which also takes ``"record"``."""
    if mode not in ("warn", "error", "ignore", "record"):
        raise ValueError(f"separation must be 'warn', 'error', 'ignore', or 'record', got {mode!r}")
    return mode


# ── sz factor smooths: each level's unpenalized line ───────────────────────


def _recedes(sign: NDArray, natural: NDArray, null_space: NDArray, free: NDArray) -> bool:
    """Whether some ``g = b(x)' N_P Z h`` has ``sign * g >= 0`` on every row, ``> 0`` on one.

    ``natural`` are the boundary rows' natural basis rows, ``sign`` the side each must keep
    (``-1`` at the lower boundary, ``+1`` at the upper), and ``free`` (``Z``) spans the
    directions of ``N_P`` the level's interior rows do not see. A least-squares direction with
    ``sign * g = 1`` settles complete separation; one free direction is its two signs;
    otherwise a linear program (Konis 2007; Geyer 2009) maximizes ``sum sign * g`` within
    ``|h| <= 1``. A direction counts only when its evaluation clears its own rounding,
    ``gamma_(k + m + q) |b| |N_P| |Z| |h|`` (Higham 2002, section 3.5).
    """
    rows = natural @ null_space @ free
    D = sign[:, None] * rows
    k, m, q = natural.shape[1], null_space.shape[1], free.shape[1]
    scale = np.abs(natural) @ (np.abs(null_space) @ np.abs(free))
    slack = gamma(k + m + q)

    def certified(h: NDArray) -> bool:
        g = D @ h
        rho = slack * (scale @ np.abs(h))
        return bool(np.all(g >= -rho) and np.any(g > rho))

    direction = np.linalg.lstsq(D, np.ones(len(D)), rcond=None)[0]
    if certified(direction):
        return True
    if q == 1:
        return certified(np.ones(1)) or certified(-np.ones(1))
    from scipy.optimize import linprog

    program = linprog(
        -D.sum(axis=0),
        A_ub=-D,
        b_ub=np.zeros(len(D)),
        bounds=[(-1.0, 1.0)] * q,
        method="highs",
    )
    return bool(program.status == 0 and certified(np.asarray(program.x, dtype=np.float64)))


def _recession_span(sign: NDArray, natural: NDArray, null_space: NDArray, free: NDArray) -> NDArray:
    """``H`` ``(q, r)``, orthonormal: the span of a separated level's directions of recession.

    Arguments as in ``_recedes``. The directions form the cone ``C = {h : D h >= 0}``, ``D``
    the boundary rows' ``sign * b(x)' N_P Z``; its span is ``{h : D_I h = 0}`` for the implicit
    equalities ``I``, and the likelihood is flat along it (Geyer 2009, section 3; Correia,
    Guimaraes and Zylkin 2021, Proposition 3). One linear program (maximize ``sum t`` under
    ``t <= D h``, ``0 <= t <= 1``) finds a relative-interior direction, which clears every row
    outside ``I`` at once. With one free direction the cone is a ray; if the program cannot be
    solved the span is the whole of ``Z``.
    """
    q = free.shape[1]
    if q == 1:
        return np.ones((1, 1))
    from scipy.optimize import linprog

    D = sign[:, None] * (natural @ null_space @ free)
    n = len(D)
    from scipy import sparse

    # the slack identity is sparse: ``n (q + 1)`` stored entries, not ``n (q + n)``
    program = linprog(
        np.concatenate([np.zeros(q), -np.ones(n)]),
        A_ub=sparse.hstack([sparse.csr_matrix(-D), sparse.identity(n, format="csr")], format="csr"),
        b_ub=np.zeros(n),
        bounds=[(None, None)] * q + [(0.0, 1.0)] * n,
        method="highs",
    )
    if program.status != 0:
        return np.eye(q)
    implicit = D[np.asarray(program.x, dtype=np.float64)[q:] < 0.5]
    if not len(implicit):
        return np.eye(q)
    return _rank_and_null_space(implicit)[1]


def _numerical_rank(values: NDArray, shape: tuple[int, ...]) -> int:
    """Rank from singular ``values``: the rule of ``numpy.linalg.matrix_rank``, ``max(shape) eps`` of the largest."""
    return int(np.sum(values > 2.0 * max(shape) * gamma(1) * values[0])) if len(values) else 0


def _rank_and_null_space(rows: NDArray) -> tuple[int, NDArray]:
    """The rank of ``rows`` and an orthonormal basis ``(width, width - rank)`` of its null space.

    The columns are scaled to unit length first (equilibration: van der
    Sluis 1969; Higham 2002, section 7.3), so the rank does not depend on a
    column's units: a covariate at ``1e8`` beside the constant otherwise puts
    two anchors one apart below the cut, a unit-free fact decided by the
    scale.  A null vector of the scaled rows is mapped back by the column
    scales and orthonormalised.  Only the right factor is read: a tall matrix
    takes the economy factorization, since its ``M x M`` left factor is never
    used; a wide one needs full matrices for its null directions.
    """
    width = rows.shape[1]
    if rows.shape[0] == 0:
        return 0, np.eye(width)
    scale = np.linalg.norm(rows, axis=0)
    scale[scale == 0.0] = 1.0
    _, values, right = np.linalg.svd(rows / scale, full_matrices=rows.shape[0] < width)
    rank = _numerical_rank(values, rows.shape)
    null = right[rank:].T / scale[:, None]
    return rank, np.ascontiguousarray(np.linalg.qr(null)[0]) if null.size else null


def _level_recession(
    natural: NDArray,
    lower: NDArray,
    upper: NDArray,
    inside: NDArray,
    null_space: NDArray,
    *,
    span: bool = True,
) -> NDArray | None:
    """One level's directions of recession, read from its rows: ``None`` when it has none.

    ``natural`` holds the level's rows ``b(x)'`` and ``lower``, ``upper`` and
    ``inside`` their responses' sides; ``null_space`` the unpenalized
    coordinates.  Some ``g = b(x)' N_P h`` recedes when it is ``<= 0`` on the
    rows at the lower boundary, ``>= 0`` on those at the upper one, zero on
    the interior rows and not zero throughout (Geyer 2009, Theorem 4):
    ``_recedes`` on the level's distinct rows.  Returns the span of the
    directions in natural coordinates (``_recession_span``).  With ``span``
    false only whether there is one is decided (``_recedes`` alone), and the
    array returned is the free directions it was decided on, not their span.
    """
    nullity = null_space.shape[1]
    unique, inverse = np.unique(natural, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    at_lower = np.zeros(len(unique), dtype=bool)
    at_upper = np.zeros(len(unique), dtype=bool)
    within = np.zeros(len(unique), dtype=bool)
    np.logical_or.at(at_lower, inverse, lower)
    np.logical_or.at(at_upper, inverse, upper)
    np.logical_or.at(within, inverse, inside)
    # an x with responses on both boundaries holds g at zero as an interior one does
    anchor = within | (at_lower & at_upper)
    anchored = unique[anchor] @ null_space
    # the anchors' rank, not their count: a gapped ``Polynomial(powers=...)``
    # has distinct anchors that are linearly dependent in ``b(x)' N_P``
    held, free = _rank_and_null_space(anchored)
    if held >= nullity:
        return None
    side = np.where(anchor, 0.0, np.where(at_lower, -1.0, 1.0))
    edge = side != 0.0
    if not (np.any(edge) and _recedes(side[edge], unique[edge], null_space, free)):
        return None
    if not span:
        return null_space @ free
    directions = _recession_span(side[edge], unique[edge], null_space, free)
    return np.ascontiguousarray(null_space @ free @ directions)


def separated_level_ranges(
    spec: Any, x: NDArray, x_cat: NDArray, prior_weights: NDArray | None, y: NDArray, boundaries
) -> dict[Any, tuple[float, float]]:
    """The levels of a varying-coefficient term whose own curve separates, with their ``x`` range.

    ``NumericCategorical`` and ``PolynomialCategorical`` give each non-base
    level an unpenalized curve ``phi(x)' theta``, ``phi(x) = (1, x)`` or
    ``(1, P(x))`` with the ``Categorical`` main effect's constant.  A level
    whose curve has a direction of recession (``_level_recession``, every
    coordinate unpenalized) has no finite estimate, and its fitted curve is
    where its walk stopped: inside the range of its rows the walk only takes
    fitted values toward the boundary its responses sit on, but beyond it the
    curve runs on toward the other.  Returns ``{level: (min x, max x)}``,
    which prediction holds the level's ``x`` to.
    """
    from superglm.features.interaction import PolynomialCategorical, _categorical_predict_labels

    x = np.asarray(x, dtype=np.float64).ravel()
    labels = _categorical_predict_labels(
        x_cat,
        grouping=spec._cat_grouping,
        universe=spec._cat_levels,
        unseen=spec._cat_unseen,
        context=spec.cat_name,
    )
    phi = spec._poly_basis(x) if isinstance(spec, PolynomialCategorical) else x[:, None]
    basis = np.column_stack([np.ones(len(x)), phi])
    weights = np.ones(len(x)) if prior_weights is None else np.asarray(prior_weights, np.float64)
    response = np.asarray(y, dtype=np.float64)
    positive = weights > 0.0
    lower = positive & (response <= 0.0) if "zero" in boundaries else np.zeros(len(x), bool)
    upper = positive & (response >= 1.0) if "one" in boundaries else np.zeros(len(x), bool)
    inside = positive & ~lower & ~upper
    ranges = {}
    for level in spec._non_base:
        rows = np.flatnonzero(positive & (labels == level))
        if np.any(lower[rows] | upper[rows]) and (
            _level_recession(
                basis[rows],
                lower[rows],
                upper[rows],
                inside[rows],
                np.eye(basis.shape[1]),
                span=False,
            )
            is not None
        ):
            ranges[level] = (float(x[rows].min()), float(x[rows].max()))
    return ranges


def emit_level_curve_separation(terms: list) -> None:
    """Name the varying-coefficient levels whose curve separates (``separated_level_ranges``).

    Never refused: prediction holds each such level's curve to its rows' range.
    """
    if terms:
        lines = "; ".join(f"{name!r} levels {_format_labels(labels)}" for name, labels in terms)
        warn_separation(
            "Separation in a varying-coefficient term: these levels' own curves separate the "
            f"response and have no finite estimate: {lines}. Predictions hold each such curve "
            "at its values over the x range of the level's rows, as extrapolation='clip' "
            "does; merging the levels (Categorical(grouping=...)) or a penalized term "
            "(FactorSmooth) gives them finite curves.",
            stacklevel=3,
        )


def separated_factor_smooth_spans(
    dominant: Any,
    null_space: NDArray,
    prior_weights: NDArray | None,
    y: NDArray,
    boundaries: tuple[str, ...],
) -> dict[int, NDArray]:
    """The ``sz`` levels whose unpenalized line separates the response, each with its span.

    Returns ``{level: S}`` in level order, ``S`` ``(k, r)`` an orthonormal basis, in natural
    coordinates, of the level's directions of recession (``_recession_span``).

    An ``sz`` level's polynomial part ``b(x)' N_P f`` is unpenalized, and shifting it with
    every level by ``-1/K`` and the main effect's by ``+1/K`` changes that level's predictor
    alone. The level has no finite estimate when some such ``g`` is ``<= 0`` on its
    lower-boundary rows, ``>= 0`` on its upper-boundary rows, zero on interior rows and not
    zero throughout (Geyer 2009, Theorem 4). Only a level with fewer than ``m`` distinct
    interior ``x`` values can have one. A level whose rows all sit on one boundary separates
    along the constant when the basis reproduces it, tested for all such levels by one sparse
    product; any other candidate is decided on its own rows (``_recedes``).
    """
    from superglm.solvers._structured.layout import (
        first_distinct_level_rows,
        level_natural_rows,
    )

    nullity = null_space.shape[1]
    if not boundaries or not nullity:
        return {}
    n_levels = int(dominant.n_levels)
    codes = np.asarray(dominant.codes, dtype=np.intp)
    weights = (
        np.ones(len(codes)) if prior_weights is None else np.asarray(prior_weights, np.float64)
    )
    response = np.asarray(y, dtype=np.float64)
    positive = weights > 0.0
    lower = positive & (response <= 0.0) if "zero" in boundaries else np.zeros(len(codes), bool)
    upper = positive & (response >= 1.0) if "one" in boundaries else np.zeros(len(codes), bool)
    inside = positive & ~lower & ~upper
    count, _ = first_distinct_level_rows(dominant, inside.astype(np.float64), nullity)
    n_lower = np.bincount(codes[lower], minlength=n_levels)
    n_upper = np.bincount(codes[upper], minlength=n_levels)
    candidates = (count < nullity) & ((n_lower > 0) | (n_upper > 0))
    if not np.any(candidates):
        return {}
    separated = np.zeros(n_levels, dtype=bool)
    spans: dict[int, NDArray] = {}

    one_sided = candidates & (count == 0) & ((n_lower == 0) | (n_upper == 0))
    if np.any(one_sided):
        natural_map = np.asarray(dominant.natural_map, dtype=np.float64)
        spanned = natural_map @ null_space
        unit = np.ones(natural_map.shape[0])
        v = np.linalg.lstsq(spanned, unit, rcond=None)[0]
        w = spanned @ v
        # the representation test of ``balance_tree._penalized_aliases``
        if np.linalg.norm(w - unit) <= np.sqrt(np.finfo(np.float64).eps) * np.linalg.norm(unit):
            if dominant.is_discrete:
                support = np.asarray(dominant.B_unique, dtype=np.float64)
                value = (support @ w)[dominant.bin_idx]
                bound = (np.abs(support) @ np.abs(w))[dominant.bin_idx]
                nnz = support.shape[1]
            else:
                value = dominant.B @ w
                bound = abs(dominant.B) @ np.abs(w)
                nnz = int(np.max(np.diff(dominant.B.indptr), initial=0))
            clear = value > gamma(nnz) * bound
            unclear = np.bincount(codes[positive & ~clear], minlength=n_levels)
            separated |= one_sided & (unclear == 0)

    general = np.flatnonzero(candidates & ~separated)
    if len(general):
        rows = np.flatnonzero(positive & np.isin(codes, general))
        rows = rows[np.argsort(codes[rows], kind="stable")]
        starts = np.searchsorted(codes[rows], general, side="left")
        stops = np.searchsorted(codes[rows], general, side="right")
        for level, start, stop in zip(general, starts, stops, strict=True):
            level_rows = rows[start:stop]
            found = _level_recession(
                level_natural_rows(dominant, level_rows),
                lower[level_rows],
                upper[level_rows],
                inside[level_rows],
                null_space,
            )
            if found is not None:
                separated[level] = True
                spans[int(level)] = found
    whole = np.ascontiguousarray(null_space)
    return {int(level): spans.get(int(level), whole) for level in np.flatnonzero(separated)}


#: What a term whose ``sz`` lines are unpenalized can do about it (#444): the
#: lines' null-space penalty is the term's own option, never the data's.
SZ_SELECT_REMEDY = (
    "To give every level a finite, unique curve, pass select=True to this FactorSmooth, "
    "which penalizes every level's line with a smoothing parameter of its own (the "
    "null-space penalty of mgcv's select=TRUE), or use basis='fs'."
)


def format_factor_smooth_separation(name: str, labels: list[Any], n_levels: int) -> str:
    """The warning for ``sz`` levels whose unpenalized line separates (``separated_factor_smooth_spans``)."""
    return (
        f"FactorSmooth {name!r} (basis='sz'): {len(labels)} of {n_levels} levels have an "
        f"unpenalized line that separates the response: {_format_labels(labels)}. The "
        "likelihood keeps increasing along each such line, so their fitted values walk to "
        "the response boundary for as long as the fit runs. They are left out of the "
        "population curve, the mean of the levels the data identify, and predict follows "
        "the population curve along each such line, where the fit has no finite estimate. "
        f"{SZ_SELECT_REMEDY} "
        "separation='ignore' silences this warning."
    )


def format_factor_smooth_canonical(name: str, separated: bool) -> str:
    """The warning for an ``sz`` term no level of which identifies the population curve."""
    if not separated:
        return (
            f"FactorSmooth {name!r} (basis='sz'): every level holds fewer distinct x "
            "values than the penalty's null space, so no level identifies the population "
            "curve. Its polynomial part is fixed by convention (each level's unidentified "
            "polynomial part zero); away from the levels' own x values the population curve, "
            "and every level's curve, still follow the fit's point along the main effect's "
            "unpenalized curve wherever that curve is not a polynomial. Predictions on the "
            f"training rows reproduce the fit. {SZ_SELECT_REMEDY}"
        )
    return (
        f"FactorSmooth {name!r} (basis='sz'): every level holds fewer distinct x "
        "values than the penalty's null space or separates the response along its line, "
        "so no level identifies the population curve. Its polynomial part "
        "is fixed by convention (each level's unidentified polynomial part zero); away "
        "from the levels' own x values the population curve, and every level's curve, "
        "still follow the fit's point along the main effect's unpenalized curve wherever "
        f"that curve is not a polynomial. {SZ_SELECT_REMEDY}"
    )


# ── spline x factor curves: a level's curve over a range it separates ──────


@dataclass(frozen=True)
class SeparatedRange:
    """One run of a level's covariate values whose responses all sit on a boundary."""

    low: float
    high: float
    n_rows: int
    weight: float
    boundary: str  # "zero", "one", or "mixed" when the run holds both
    # Past the level's first or last anchor: the covariate value the curve is
    # held at there.  ``None`` for a run between anchors, where ``interpolated``
    # says whether the curve directions only it sees are held.
    held_at: float | None = None
    interpolated: bool = False


@dataclass(frozen=True)
class SeparatedCurve:
    """A factor level whose penalised curve separates the response over some ranges."""

    term: str
    covariate: str
    level: Any
    ranges: tuple[SeparatedRange, ...]
    note: str = ""
    # Whether ``separation="error"`` refuses on this finding; a factor base's
    # range, found on the main effect alone, is reported only.
    refusable: bool = True


def _unseen_directions(rows: NDArray) -> NDArray:
    """Orthonormal directions ``rows`` do not see, at the shared factor rank cut.

    A direction whose image is below ``factor_rcond`` times the largest
    singular value moves those rows by less than ``sqrt(eps)`` of what the
    best-seen direction does: for separation it is as good as unseen, since
    a walk along it changes their fitted values only after it has carried the
    rest ``1 / sqrt(eps)`` times further.
    """
    from superglm.solvers.rank import SHARED_RANK_POLICY

    width = rows.shape[1]
    if rows.shape[0] == 0:
        return np.eye(width)
    # Only the right factor is read.  A tall matrix (a level's distinct values on
    # the exact path) already yields the full width x width one without the
    # M x M left factor; a wide one needs full matrices for its null directions.
    _, sigma, vt = np.linalg.svd(rows, full_matrices=rows.shape[0] < width)
    cut = SHARED_RANK_POLICY.factor_rcond * float(sigma[0]) if sigma.size else 0.0
    return vt[int(np.count_nonzero(sigma > cut)) :].T


#: Distance a program must move a unit-length row for it to count as moved: 100x HiGHS's
#: default primal feasibility tolerance (1e-7), so solver residual never reads as a held row.
_PROGRAM_TOL = 1e-5


#: Largest weight ``_unmoved_rows`` gives a row, against the ``1`` every certified row reaches.
#: The rounding of ``D' y`` at this weight, about ``u 1e8 ~ 1e-8`` per column of unit-length
#: rows, stays three digits below ``_PROGRAM_TOL``; overlap by a margin of ``1e-5`` or more
#: needs no wider spread.
_CERTIFICATE_SPREAD = 1e8


def _unmoved_rows(D: NDArray) -> NDArray:
    """Rows no direction ``a`` with ``D a >= 0`` and ``|a| <= 1`` moves by more than ``_PROGRAM_TOL``.

    Row ``j`` is held at zero by the whole cone exactly when some ``y >= 0``, ``y_j > 0``, has
    ``D' y = 0`` (Stiemke's alternative; Konis 2007, chapter 4). One program, maximising
    ``sum t`` under ``D' y = 0``, ``t <= y``, ``0 <= t <= 1``, ``y <= _CERTIFICATE_SPREAD``,
    finds a ``y`` for all such rows (unbounded, HiGHS returns weights near ``1e13`` too coarse to
    certify). It is checked, not trusted: ``y_j D_j a <= (D' y)' a <= |D' y|_1``, so a row counts
    when ``|D' y|_1 / y_j``, with the rounding of ``D' y`` (Higham 2002, section 3.5), is at most
    ``_PROGRAM_TOL``. Other rows, and all rows if the program fails, are left to ``_held_rows``.
    """
    from scipy import sparse
    from scipy.optimize import linprog

    n_rows, width = D.shape
    identity = sparse.identity(n_rows, format="csr")
    program = linprog(
        np.concatenate([np.zeros(n_rows), -np.ones(n_rows)]),
        A_ub=sparse.hstack([-identity, identity], format="csr"),
        b_ub=np.zeros(n_rows),
        A_eq=sparse.hstack([sparse.csr_matrix(D.T), sparse.csr_matrix((width, n_rows))]),
        b_eq=np.zeros(width),
        bounds=[(0.0, _CERTIFICATE_SPREAD)] * n_rows + [(0.0, 1.0)] * n_rows,
        method="highs",
    )
    if program.status != 0:
        return np.zeros(n_rows, dtype=bool)
    y = np.maximum(np.asarray(program.x[:n_rows], dtype=np.float64), 0.0)
    residual = np.abs(D.T @ y) + gamma(n_rows) * (np.abs(D.T) @ y)
    bound = float(np.sum(residual)) * (1.0 + gamma(width + 2))
    return (y > 0.0) & (bound <= _PROGRAM_TOL * y)


def _held_rows(D: NDArray) -> NDArray:
    """The rows some direction ``a`` moves strictly the right way, ``D a > 0``, with the others unmoved.

    These directions form a cone whose moved rows are the complement of the linearity (Geyer
    2009, section 3.12). One program maximises how many rows a direction moves (``sum t`` under
    ``D a >= t``, ``0 <= t <= 1``, ``|a| <= 1``); each row it leaves gets its own, maximising
    ``D_j a`` under ``D a >= 0``, unless ``_unmoved_rows`` certifies it, which also drops a
    counting hit on it as solver tolerance. Overlapping rows are decided by these two programs.
    Rows are scaled to unit length and the box keeps the programs bounded (HiGHS fails on free
    ones once rows differ in scale); they are those the free directions see by more than the
    rank cut (``separated_curve_support``).
    """
    from scipy import sparse
    from scipy.optimize import linprog

    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    n_rows, width = D.shape
    box = [(-1.0, 1.0)] * width
    program = linprog(
        np.concatenate([np.zeros(width), -np.ones(n_rows)]),
        A_ub=sparse.hstack(
            [sparse.csr_matrix(-D), sparse.identity(n_rows, format="csr")], format="csr"
        ),
        b_ub=np.zeros(n_rows),
        bounds=box + [(0.0, 1.0)] * n_rows,
        method="highs",
    )
    held = (
        np.asarray(program.x[width:]) > _PROGRAM_TOL
        if program.status == 0
        else np.zeros(n_rows, dtype=bool)
    )
    unmoved = _unmoved_rows(D) if n_rows > 1 else np.zeros(n_rows, dtype=bool)
    held &= ~unmoved
    for row in np.flatnonzero(~held & ~unmoved):
        single = linprog(-D[row], A_ub=-D, b_ub=np.zeros(n_rows), bounds=box, method="highs")
        held[row] = single.status == 0 and -float(single.fun) > _PROGRAM_TOL
    return held


def separated_curve_support(
    raw: NDArray,
    projection: NDArray,
    lower: NDArray,
    upper: NDArray,
    inside: NDArray,
) -> tuple[NDArray, NDArray] | None:
    """The support points over which one level's curve separates, and the directions only they see.

    ``raw`` holds the level's distinct raw basis rows, ``projection`` maps scanned coefficients
    to raw ones, and ``lower`` / ``upper`` / ``inside`` say whether a point holds rows on the
    lower boundary, the upper boundary or strictly between. A point holding an interior row, or
    rows on both boundaries, anchors the curve. A direction ``h`` with ``g = raw projection h``
    zero at every anchor, ``g <= 0`` on lower-boundary points, ``g >= 0`` on upper ones and not
    zero throughout is a direction of recession (Geyer 2009, Theorem 4; Konis 2007); the points
    some such direction moves are the separated ones.

    Returns ``(separated, free)``: a mask over points and an orthonormal basis of the directions
    every other point leaves unseen (``_unseen_directions``), in which every direction of
    recession lies; ``None`` when no point separates. A point no direction in ``free`` moves is
    returned to the others, which only shrinks ``free``, so the loop ends.
    """
    anchor = inside | (lower & upper)
    edge = ~anchor & (lower | upper)
    if not np.any(edge):
        return None
    from superglm.solvers.rank import SHARED_RANK_POLICY

    rows = raw @ projection
    sign = np.where(lower, -1.0, 1.0)
    free = _unseen_directions(rows[anchor])
    # A point the free directions do not see -- its row within the rank
    # cut's angle of the anchors' span -- cannot be held; leaving it out keeps
    # the program to the ranges the anchors leave open, which is what bounds
    # its size on a covariate with many distinct values.
    norms = np.linalg.norm(rows, axis=1)
    separated = edge.copy()
    while True:
        separated &= np.linalg.norm(rows @ free, axis=1) > SHARED_RANK_POLICY.factor_rcond * norms
        if not free.shape[1] or not separated.any():
            return None
        points = np.flatnonzero(separated)
        held = _held_rows(sign[points, None] * (rows[points] @ free))
        if held.all():
            return separated, free
        separated[points[~held]] = False
        free = _unseen_directions(rows[~separated])


def continuation_basis(
    free: NDArray,
    omega: NDArray,
    omega_null: NDArray,
    projection: NDArray,
    projected_null: NDArray,
) -> NDArray | None:
    """Block coefficients whose separated part is the smooth continuation of the rest.

    ``free`` (raw coefficients, orthonormal columns) spans the curve directions only a level's
    separated rows see. The penalised likelihood keeps rising along them, so the curve is held
    where the penalty alone puts it given the rest: ``free' omega f = 0``, which reads
    ``free' omega projection h = 0`` on the block coefficients ``h`` since the level's constant
    lies in ``omega``'s null space. A ``free`` direction the penalty does not see either
    (``omega_null``) has no continuation; the block's unpenalised directions (``projected_null``)
    are then held at zero. Returns an orthonormal basis of the block coefficients left, or
    ``None`` when none is.
    """
    from superglm.solvers.rank import SHARED_RANK_POLICY

    width = projection.shape[1]
    outside = free - omega_null @ (omega_null.T @ free) if omega_null.size else free
    _, sines, vt = np.linalg.svd(outside, full_matrices=True)
    # Sines of the principal angles between span(free) and null(omega).
    n_line = free.shape[1] - int(np.count_nonzero(sines > SHARED_RANK_POLICY.factor_rcond))
    penalised = free @ vt[: free.shape[1] - n_line].T
    held = [(omega @ penalised).T @ projection]
    if n_line:
        held.append(projected_null.T)
    # Rows of the two kinds are orthogonal (projection maps projected_null
    # into omega's null space), and each kind has full row rank, so the rank
    # is known and no cut decides it.
    rank = penalised.shape[1] + (projected_null.shape[1] if n_line else 0)
    if rank >= width:
        return None
    return np.linalg.svd(np.vstack(held), full_matrices=True)[2][rank:].T


def format_curve_separation(findings: list[SeparatedCurve]) -> str:
    """One plain message naming each level, the covariate range and what the fit did there."""
    words = {
        "zero": "every response 0",
        "one": "every response 1",
        "mixed": "every response 0 or 1",
    }
    lines = []
    for item in findings[:_MAX_LISTED_LABELS]:
        spans = []
        for r in item.ranges:
            where = f"{r.low:g}" if r.low == r.high else f"{r.low:g} to {r.high:g}"
            if r.held_at is not None:
                how = f"held at its value at {item.covariate} {r.held_at:g}"
            elif r.interpolated:
                how = "held at the smooth interpolation of the curve on either side"
            else:
                how = "left to the smoothing penalty"
            spans.append(
                f"{item.covariate} {where} ({r.n_rows} rows, total weight {r.weight:.4g}, "
                f"{words[r.boundary]}): {how}"
            )
        note = f" ({item.note})" if item.note else ""
        lines.append(f"  {item.term!r}, level {item.level!r}{note}: " + "; ".join(spans))
    if len(findings) > _MAX_LISTED_LABELS:
        lines.append(f"  ... and {len(findings) - _MAX_LISTED_LABELS} more levels")
    upper = any(r.boundary != "zero" for item in findings for r in item.ranges)
    edge = "0 (or 1)" if upper else "0"
    head = (
        f"Separation in a spline-by-factor term: {len(findings)} level(s) have a range of the "
        f"covariate where every response is {edge}. The data there show the mean is near "
        f"{edge} but not how near (its maximum-likelihood value is {edge} itself), so nothing "
        "in the data stops the level's curve heading to that limit on those rows; only its "
        "smoothing penalty does, and the rest of the curve can set that penalty weak. The "
        "fit holds the curve over each such range it can, as each line says:"
    )
    tail = (
        "Past the first or last value where a level's responses vary, its curve keeps its "
        "value there, as a spline keeps its value past its training range "
        "(extrapolation='clip'), in the fit and in predict(). To let the data shape these "
        "ranges, merge the level into a similar one (Categorical(grouping=...)) or drop the "
        "interaction. separation='ignore' leaves the curves free there, as earlier versions "
        "did; separation='error' refuses the design"
        + (
            "."
            if any(item.refusable for item in findings)
            else ", except on a factor base's range, which it names and does not refuse."
        )
    )
    return "\n".join([head, *lines, tail])
