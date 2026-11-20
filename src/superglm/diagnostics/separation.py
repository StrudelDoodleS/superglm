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
term bounded by an active selection penalty is skipped (the penalised optimum is finite);
only ``selection_penalty`` is consulted, since ``Categorical`` blocks carry no ``lambda2``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import UNIT_ROUNDOFF, gamma
from superglm._warn import warn_caller

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


@contextmanager
def disclosure_scope(model: Any) -> Iterator[None]:
    """``model``'s public fit call: each distinct separation notice inside it is emitted once.

    Each call opens its own scope, restoring the enclosing one on exit, so an
    independent fit started inside another call (from a progress callback)
    reports what it finds.  A fit the enclosing call runs and discards
    (``_discarded_fit``: a power-search candidate's ``fit_reml``, a ``drop1``
    comparison fit) shares that call's scope.
    """
    if getattr(model, "_discarded_fit", False) and _DISCLOSED.get() is not None:
        yield
        return
    token = _DISCLOSED.set(set())
    named = _NAMED_ROWS.set({})
    try:
        yield
    finally:
        _NAMED_ROWS.reset(named)
        _DISCLOSED.reset(token)


#: The rows this public fit call's build scans have named, by row count; ``None`` outside one.
_NAMED_ROWS: ContextVar[dict[int, NDArray] | None] = ContextVar(
    "superglm_separation_named_rows", default=None
)


def _record_named_rows(findings: list[SeparatedTerm]) -> None:
    """Keep the rows ``findings`` name in this call's scope (``named_by_scan``)."""
    named = _NAMED_ROWS.get()
    if named is None:
        return
    for finding in findings:
        if finding.rows is None:
            continue
        held = named.get(finding.rows.size)
        named[finding.rows.size] = finding.rows if held is None else held | finding.rows


def named_by_scan(rows: NDArray) -> bool:
    """Whether this call's build scans already named every one of ``rows``.

    A separation the exact decision finds on rows the categorical scan named is the same
    finding: the fit gives one warning for it, the scan's, and the summary footnote adds which
    coefficients it reaches.  A row the scans did not name is news, and the decision says so.
    A scan of other rows (another row count) names none of these.
    """
    named = _NAMED_ROWS.get()
    rows = np.asarray(rows, dtype=np.intp)
    if not named or not rows.size:
        return False
    return any(int(rows.max()) < size and bool(np.all(mask[rows])) for size, mask in named.items())


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
    A model's fit passes its published statement
    (``separation_ops.disclosed_text``), which its existence decision makes
    where one is built.
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
    """Whether a separation statement on the fit ``model`` is publishing now refuses it.

    Under ``"error"``, a fit the caller receives is refused as soon as its
    statement is known, before anything else describes it
    (``runtime_canonicalize.refuse_published_statement``).  A discarded fit
    (``_discarded_fit``) and a refit of an NB2 theta alternation, which a later
    refit supersedes, are never refused: the alternation's published refit is,
    once its search flag is cleared.
    """
    return (
        getattr(model, "_separation", "warn") == "error"
        and not getattr(model, "_discarded_fit", False)
        and not getattr(model, "_nb_theta_search", False)
    )


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


def format_drop1_runtime_message(findings: Sequence[tuple[str, str, str]]) -> str | None:
    """One statement for ``drop1``: the dropped features whose comparison fit says it is separated.

    Each finding is ``(feature, statement, source)``.  A comparison fit the backstop flagged
    (``"heuristic"``) is not a maximum-likelihood fit; one whose decision proved separation is
    the limit fit, whose deviance is the supremum the likelihood approaches while some of its
    coefficients have no finite estimate.
    """
    if not findings:
        return None
    if len(findings) == 1:
        opening = (
            f"the comparison fit without {findings[0][0]!r} carries the runtime separation "
            "finding, so the deviance change and test for dropping it compare"
        )
    else:
        names = " and ".join(f"without {name!r}" for name, _, _ in findings)
        opening = (
            f"the comparison fits {names} each carry the runtime separation finding, so the "
            "deviance change and test for dropping each compare"
        )
    if all(source == "heuristic" for _, _, source in findings):
        compared = "a fit that is not a maximum-likelihood fit"
    else:
        compared = (
            "the limit of a fit some of whose coefficients have no finite estimate (its "
            "deviance is the supremum the likelihood approaches)"
        )
    return f"drop1: {opening} with {compared}. Without {findings[0][0]!r}: {findings[0][1]}"


@dataclass(frozen=True)
class SeparatedTerm:
    """Separated cells found in one categorical or crossed-categorical term."""

    term: str
    kind: str  # "levels" (main effect), "cells" (crossed) or "intercept"
    boundary: str  # "zero" or "one"
    labels: list[Any]  # level names, or (level1, level2) pairs
    n_occupied: int  # occupied levels/cells scanned in this term
    # the positive-weight rows of the cells named, a mask over the fit's rows (``named_by_scan``)
    rows: NDArray | None = field(default=None, compare=False, repr=False)


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


def _flagged_rows(codes: NDArray, flags: NDArray, sample_weight: NDArray) -> NDArray:
    """The positive-weight rows whose cell ``flags`` marks, as a mask."""
    on = codes >= 0
    rows = np.zeros(codes.shape, dtype=bool)
    rows[on] = flags[codes[on]]
    return rows & (np.asarray(sample_weight) > 0)


def scan_categorical_term(
    name: str,
    spec: Any,
    x: NDArray,
    y: NDArray,
    sample_weight: NDArray,
    boundaries: tuple[str, ...],
    build_codes: NDArray | None = None,
) -> list[SeparatedTerm]:
    """Scan one main-effect Categorical term for separated levels.

    Every level of the universe is scanned, base included: with the level's
    own indicator the recession direction is that coordinate; for the base
    level it is the intercept walking against every non-base indicator.
    ``build_codes`` are the term's build codes, when the caller holds them;
    the scan reads them instead of coding the column again.
    """
    codes, levels = _level_codes(x, spec, context=name, codes=build_codes)
    findings: list[SeparatedTerm] = []
    for boundary in boundaries:
        flags, n_occupied = _separated_flags(codes, len(levels), y, sample_weight, boundary)
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
) -> str | None:
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
    return format_runtime_message(ratio, n_iter, drifting, pinned)


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


def emit_separation_findings(findings: list[SeparatedTerm], mode: str) -> None:
    """Warn or raise per the model's ``separation`` mode."""
    if not findings or mode == "ignore":
        return
    message = format_separation_message(findings)
    if mode == "error":
        raise SeparationError(message, levels=True)
    # stacklevel points past the builder into the user's fit call region; the
    # exact frame depth varies by entrypoint, so the message carries the term
    # names rather than relying on the reported line.
    _record_named_rows(findings)
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


# ── Existence of the penalised estimate ────────────────────────────────────
#
# A fit maximises ``sum_i w_i l_i(eta_i) - beta'S beta / 2`` with ``S`` semidefinite.  For the
# families ``response_boundaries`` names every row's log-likelihood is bounded above, so along
# a direction with ``S gamma != 0`` the objective falls like ``-t^2``: only ``gamma`` in
# ``null(S)`` can recede.  There the maximiser exists unless some ``z = A gamma != 0`` has
# ``z_i = 0`` on the interior rows, ``z_i <= 0`` where ``y_i`` is at the lower boundary and
# ``z_i >= 0`` where it is at the upper one (Verbeek 1989; Santos Silva & Tenreyro 2010;
# Correia, Guimaraes & Zylkin 2019, arXiv:1903.01633, Prop. 1; for the binomial, Albert &
# Anderson 1984 and Silvapulle 1981).  Withholding the rows such a direction separates
# leaves the fitted values of the others and every coefficient the direction does not move
# unchanged (Prop. 3), so a separated fit publishes the fit on the rows it keeps.
#
# Every decision here is certified or reported unresolved, never guessed: a separating
# direction is verified on the rows exactly (``exact_row_signs``), and absence is proved by
# the fitted score (``recession_excluded``).

SeparationState = Literal["absent", "present", "partial", "unresolved"]

# Snapping grids tried on a proposed direction before it is verified exactly: the
# verification is the proof, so any proposal that passes it is a certificate.
_SNAP_BITS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30)

#: The largest linear program (rows it reads times candidate directions) the exact path runs.
#: HiGHS's interior-point method took seven iterations on every logistic case measured, but each
#: grows faster than its rows (0.9 s at 20,000 rows of nine directions, 4.7 s at 50,000, 56 s at
#: 200,000): a size, not an iteration count, bounds its work, and one every platform reads
#: alike.  A healthy fit's score proves existence first (``recession_excluded``), so the
#: program reads only the rows a candidate moves on fits the score could not certify.  Past the
#: cap it reads a subset of the cap's size (``program_rows``): its answer is a proposal like any
#: other, verified on every row exactly, so the subset bounds the work, never the proof.
PROGRAM_CAP = 2**19


@dataclass(frozen=True)
class Separation:
    """Whether a fit's penalised estimate exists, decided from its responses and design.

    ``state`` is one of:

    - ``"absent"``: certified; every coefficient has a finite estimate.
    - ``"present"``: certified; ``rows`` (the fit's row indices) are every separated row and
      ``span`` (``(1 + width) x k``, ``[intercept; slopes]`` in fitted coordinates, verified on
      the rows exactly) spans the directions along which the estimate is infinite.
    - ``"partial"``: a separating direction verified on the rows exactly, ``rows`` the rows it
      separates, a proven lower bound: the fit on the other rows was not shown to separate no
      further (or not run, for a fit refused under ``"error"``), so more rows may be separated.
      ``span`` holds the direction, its entries below the square root of eps cut for reporting.
    - ``"unresolved"``: float64 cannot decide; no direction verified.  Past the program's cap
      (``work["capped"]``) the direction proposed from ``work["sampled"]`` of its rows did not
      verify on every row.

    ``decided_by`` names the route, ``boundaries`` the family's response boundaries, and
    ``unique`` is false for a link whose likelihood is not concave (cauchit: the sign test
    proves a maximiser exists, not that it is unique).  ``labels`` names the categorical
    levels whose rows are all separated, ``names`` the coefficients ``span`` moves, and
    ``work`` counts what the decision read.  ``complete`` marks a ``"present"`` decision
    with no row kept, and ``labelled_rows`` counts the separated rows the levels in ``labels``
    hold.
    """

    state: SeparationState
    decided_by: str
    rows: NDArray = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    span: NDArray = field(default_factory=lambda: np.zeros((1, 0)))
    boundaries: tuple[str, ...] = ()
    unique: bool = True
    labels: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    work: Mapping[str, int] = field(default_factory=dict)
    complete: bool = False
    labelled_rows: int = 0

    def notes(self) -> tuple[str, ...]:
        """The summary's footnote for this decision (empty when the estimate exists and is unique)."""
        if self.state == "absent":
            if self.unique:
                return ()
            return (
                "The cauchit link's likelihood is not concave: no direction separates the "
                "responses, so a maximiser exists, but it need not be unique.",
            )
        if self.state == "unresolved":
            return (
                "Whether every coefficient has a finite estimate is not resolved at float64: "
                f"{self._unresolved_reason()}.",
            )
        words = self._words()
        if self.complete:
            unreached = self.span.shape[0] - len(self.names)
            penalised = (
                " The standard errors shown come from the penalty alone: no row informs them."
                if unreached > 0
                else ""
            )
            return (
                f"{words['count']} {words['has']} no finite estimate (estimate and standard "
                f"error shown as ---): {words['listed']}. Every row is separated ({words['rows']} with "
                f"responses at {words['edges']}): a direction moving {words['them']} fits every "
                f"row exactly at the boundary, so no row is kept and the coefficients are where "
                f"the fit stopped.{penalised} {_SEPARATION_REMEDY}",
            )
        if self.state == "partial":
            return (
                f"{words['count']} {words['has']} no finite estimate (estimate and standard "
                f"error shown as ---): {words['listed']}. At least {words['rows']} with responses at "
                f"{words['edges']} ({words['where']}) {words['are']} separated along a direction moving "
                f"{words['them']}; the fit on the remaining rows was not shown to separate no "
                "further, so more rows may be. The coefficients are where the fit stopped, not "
                f"a fit on the remaining rows. {_SEPARATION_REMEDY}",
            )
        return (
            f"{words['count']} {words['has']} no finite estimate (estimate and standard error "
            f"shown as ---): {words['listed']}. {words['rows']} with responses at {words['edges']} "
            f"({words['where']}) {words['are']} separated: a direction moving {words['them']} fits those "
            f"rows exactly at the boundary, so {words['estimate']} infinite. The other "
            "coefficients and their standard errors are the fit on the remaining rows. "
            f"{_SEPARATION_REMEDY}",
        )

    def statement(self, stops: tuple[float, float] | None, guard: str | None) -> str | None:
        """What the fit says about this decision when it is published: the warning, or ``None``.

        ``stops`` is the range of the linear predictor on the separated rows where the fit
        stopped; ``guard`` the runtime evidence on an unresolved decision (the walk's rows at
        the link's guard, or an exhausted budget), without which an unresolved decision says
        nothing at fit time (its footnote still does).
        """
        if self.state == "absent":
            return None
        if self.state == "unresolved":
            if guard is None:
                return None
            return (
                "Separation not resolved: whether every coefficient has a finite estimate is "
                f"not decided at float64 ({self._unresolved_reason()}). {guard}, which "
                "separation would explain: those rows' predictions may have collapsed to the "
                f"response boundary. {_SEPARATION_REMEDY} Pass separation='ignore' to silence "
                "this."
            )
        words = self._words()
        held = f" (linear predictor {_stopping_range(stops)})" if stops is not None else ""
        if self.complete:
            return (
                f"Complete separation: every row's response is at {words['edges']} along a "
                f"direction moving {words['listed']}, so no row is kept and {words['count']} "
                f"{words['has']} no finite estimate. The coefficients are where the fit "
                f"stopped{held}, a finite approximation of a limit at infinity, with no "
                f"standard error. {_SEPARATION_REMEDY} Pass separation='ignore' to silence this."
            )
        if self.state == "partial":
            return (
                f"Separation: at least {words['rows']} with responses at {words['edges']} "
                f"({words['where']}) {words['are']} fitted exactly at the boundary along a direction moving "
                f"{words['listed']}, so {words['count']} {words['has']} no finite estimate; more "
                f"rows may be separated. The coefficients are where the fit stopped{held}, not a "
                f"fit on the other rows, and {words['their']} standard {words['errors']} "
                f"withheld. {_SEPARATION_REMEDY} Pass separation='ignore' to silence this."
            )
        return (
            f"Separation: {words['rows']} with responses at {words['edges']} "
            f"({words['where']}) {words['are']} fitted exactly at the boundary along a direction moving "
            f"{words['listed']}, so {words['count']} {words['has']} no finite estimate. The fit "
            "returned is the fit on the other rows; the separated rows' linear predictor is "
            f"held where the fit stopped{held}, and {words['their']} standard {words['errors']} "
            f"withheld. {_SEPARATION_REMEDY} Pass separation='ignore' to silence this."
        )

    def _unresolved_reason(self) -> str:
        edges = " or ".join({"zero": "0", "one": "1"}[edge] for edge in self.boundaries)
        capped = self.work.get("capped")
        if capped:
            sampled = self.work.get("sampled", 0)
            return (
                "the fit's score does not prove it, and the rows' linear program "
                f"({self.work.get('lp_rows', 0)} rows by {self.work.get('candidates', 0)} "
                f"directions) exceeds its size cap of {capped} entries"
                + (
                    f"; no direction it proposed from {sampled} of those rows separates the "
                    f"responses at {edges} on every row"
                    if sampled
                    else ", so it was not run"
                )
            )
        return (
            "the fit's score does not prove it, and no direction separating the responses at "
            f"{edges} verifies exactly on the rows"
        )

    def _words(self) -> dict[str, str]:
        """The pieces the decision's texts share: the coefficients, the rows and the edges."""
        # a term's coordinates share its name: one entry per term, with its count
        counted: dict[str, int] = {}
        for name in self.names:
            counted[name] = counted.get(name, 0) + 1
        entries = [
            name if count == 1 else f"{name} ({count} coefficients)"
            for name, count in counted.items()
        ]
        listed = ", ".join(entries[:_MAX_LISTED_LABELS])
        if len(entries) > _MAX_LISTED_LABELS:
            listed += f" and {len(entries) - _MAX_LISTED_LABELS} more"
        one = len(self.names) == 1
        other = self.rows.size - (self.labelled_rows if self.labels else 0)
        if other:
            shown = ", ".join(str(int(row)) for row in self.rows[:6])
            more = f" and {self.rows.size - 6} more" if self.rows.size > 6 else ""
            rows = f"{'row' if self.rows.size == 1 else 'rows'} {shown}{more}"
            where = "; ".join(self.labels) + f", and {other} other rows" if self.labels else rows
        else:
            where = "; ".join(self.labels)
        return {
            "listed": listed,
            "count": f"{len(self.names)}",
            "has": "coefficient has" if one else "coefficients have",
            "them": "it" if one else "them",
            "estimate": "its estimate is" if one else "their estimates are",
            "their": "its" if one else "their",
            "errors": "error is" if one else "errors are",
            "edges": " or ".join({"zero": "0", "one": "1"}[edge] for edge in self.boundaries),
            "where": where,
            "rows": f"{self.rows.size} row" + ("" if self.rows.size == 1 else "s"),
            "are": "is" if self.rows.size == 1 else "are",
        }

    @property
    def reached(self) -> NDArray:
        """The slope coordinates ``span`` moves: the coefficients with no finite estimate."""
        if self.span.shape[1] == 0:
            return np.zeros(0, dtype=np.intp)
        return np.flatnonzero(np.any(self.span[1:] != 0.0, axis=1)).astype(np.intp)

    @property
    def intercept_reached(self) -> bool:
        """Whether ``span`` moves the intercept."""
        return bool(self.span.shape[1] and np.any(self.span[0] != 0.0))


_SEPARATION_REMEDY = (
    "Merging or re-basing the levels named, or penalising the term, gives every coefficient a "
    "finite estimate."
)


def _stopping_range(stops: tuple[float, float]) -> str:
    low, high = (float(f"{value:.4g}") for value in stops)
    if low == high:
        return f"at {low:g}"
    return f"between {low:g} and {high:g}"


def response_sides(y: NDArray, prior_weights: NDArray, boundaries: tuple[str, ...]) -> NDArray:
    """Each row's side of the response's support, as an ``int8``.

    ``-1`` on the lower boundary (``y <= 0`` where ``"zero"`` is a boundary), ``+1`` on the
    upper (``y >= 1`` where ``"one"`` is), ``0`` inside, and ``2`` for a row with no prior
    weight, which carries no likelihood.  Log-binomial's boundaries are ``("zero",)``: a
    success sits at the mean space's finite edge, reached at a finite predictor, so it is an
    interior row (an equality), never separation.
    """
    y = np.asarray(y, dtype=np.float64)
    sides = np.zeros(y.shape, dtype=np.int8)
    # one pass per boundary (a boolean mask's scatter costs several)
    if "zero" in boundaries:
        np.subtract(sides, y <= 0.0, out=sides, casting="unsafe")
    if "one" in boundaries:
        np.add(sides, y >= 1.0, out=sides, casting="unsafe")
    dead = ~(np.asarray(prior_weights, dtype=np.float64) > 0.0)
    if dead.any():
        sides[dead] = 2
    return sides


def recession_excluded(margin: float, defect: float, curvature: float) -> bool:
    """Whether a fitted score proves that no direction of recession exists.

    Stiemke's alternative (Stiemke 1915; Mangasarian, *Nonlinear Programming*, 1969, ch. 2;
    the form Konis 2007, ch. 4 uses): either some ``h`` with ``S h = 0`` and ``z = A h``
    separates (``z_i = 0`` inside, ``s_i z_i >= 0`` on the boundary, ``z != 0``), or a vector
    ``mu > 0`` on the boundary rows balances them, never both.  A fit supplies ``mu`` from its
    score after one Newton step ``v``: ``r' = W (z - eta - A v)``, ``mu_i = s_i r'_i``, with
    ``A'r' - S (beta + v) = e`` its defect.  For a separating ``h`` the interior rows and the
    penalty drop out of ``h'e``, so with ``margin = c = min_B mu_i / sqrt(W_i)`` (weighted
    Cauchy-Schwarz, the gate review k3p8)

        h'e = sum_B mu_i |z_i| >= c sum_B sqrt(W_i) |z_i| >= c (sum W_i z_i^2)^(1/2)
            = c (h'H h)^(1/2) >= c sqrt(lambda) ||h~||

    in Jacobi coordinates ``h~``, ``lambda`` a certified lower bound on the Jacobi-scaled
    Hessian's smallest eigenvalue (``W`` is the Hessian's own, and positive on every boundary
    row).  With ``h'e <= ||h~|| defect`` (``defect`` a bound on the scaled ``e``), a separating
    ``h`` needs ``c sqrt(lambda) <= defect``; so when the margin clears it no ``h`` separates.
    It is never weaker than ``min mu sqrt(lambda / max W)``, since ``c >= min mu / sqrt(max
    W)``: a zero-response row's multiplier is about its own working weight, so the unweighted
    form failed on healthy fits with a row far into the tail.  A sufficient test (Sol's gate
    review): when it fails the exact path decides.
    """
    if not (margin > 0.0 and curvature > 0.0):
        return False
    if not (math.isfinite(margin) and math.isfinite(defect)):
        return False
    return margin * math.sqrt(curvature) > defect


def exact_row_signs(matrix, direction: NDArray) -> NDArray:
    """The exact sign of each row of ``matrix @ direction``, as an ``int8``.

    ``matrix`` is a SciPy CSR matrix.  Every product is split exactly (Dekker's TwoProduct,
    ``mode_score.two_product``).  A row whose plain sum clears its rounding bound,
    ``gamma_(2k) sum |terms|`` over its ``2k`` addends (Higham 2002, eq. 4.4), has that sum's
    sign.  A row whose products are exact (no tail) and whose terms lie on one dyadic grid
    narrow enough to hold their sum (their exponents within ``53 - log2(count)`` bits of the
    finest bit any of them sets) sums exactly in float64 in any order, so its plain sum is
    exact: a structural row (indicator columns, a dyadic direction) is decided so, with no row
    loop.  Every other row is summed exactly by ``math.fsum`` (Shewchuk's algorithm, correctly
    rounded, so its sign is the exact sum's).  A row whose product leaves the normal range
    (``0 < |p| < 2^-969``, where TwoProduct's tail is lost) is ``2``: not decidable here.
    """
    from superglm.solvers.mode_score import two_product

    indptr = np.asarray(matrix.indptr, dtype=np.intp)
    rows = indptr.size - 1
    signs = np.zeros(rows, dtype=np.int8)
    if matrix.nnz == 0:
        return signs
    direction = np.asarray(direction, dtype=np.float64)
    high, low = two_product(np.asarray(matrix.data, dtype=np.float64), direction[matrix.indices])
    counts = np.diff(indptr)
    owner = np.repeat(np.arange(rows), counts)
    plain = np.bincount(owner, weights=high + low, minlength=rows)
    magnitude = np.bincount(owner, weights=np.abs(high) + np.abs(low), minlength=rows)
    nonzero = high != 0.0
    lost = np.bincount(owner, weights=nonzero & (np.abs(high) < 2.0**-969), minlength=rows) > 0
    bound = gamma(2 * counts + 2) * magnitude
    clear = np.abs(plain) > bound
    signs[clear] = np.sign(plain[clear]).astype(np.int8)
    # the dyadic grid of each row's terms: the finest bit any sets, the largest exponent
    mantissa, exponent = np.frexp(high[nonzero])
    significand = np.ldexp(np.abs(mantissa), 53).astype(np.int64)
    finest = exponent - 53 + np.log2((significand & -significand).astype(np.float64)).astype(int)
    big = np.iinfo(np.int64).max // 4
    lowest = np.full(rows, big, dtype=np.int64)
    highest = np.full(rows, -big, dtype=np.int64)
    np.minimum.at(lowest, owner[nonzero], finest)
    np.maximum.at(highest, owner[nonzero], exponent)
    terms = np.bincount(owner[nonzero], minlength=rows)
    tailed = np.bincount(owner, weights=low != 0.0, minlength=rows) > 0
    span = highest - lowest + np.ceil(np.log2(np.maximum(terms, 1))).astype(np.int64) + 1
    gridded = ~clear & ~lost & ~tailed & ((terms == 0) | (span <= 53))
    exact = np.bincount(owner, weights=high, minlength=rows)
    signs[gridded] = np.sign(exact[gridded]).astype(np.int8)
    for row in np.flatnonzero(~clear & ~gridded & ~lost & (counts > 0)):
        lo, hi = indptr[row], indptr[row + 1]
        total = math.fsum(np.concatenate((high[lo:hi], low[lo:hi])).tolist())
        signs[row] = int(np.sign(total))
    signs[lost & ~clear] = 2
    return signs


def _verified_separation(matrix, sides: NDArray, direction: NDArray) -> NDArray | None:
    """The rows ``direction`` separates, if it is exactly a direction of recession; else ``None``.

    Exact (``exact_row_signs``): zero on every interior row, the boundary's sign or zero on
    every boundary row, and strictly the boundary's sign on at least one.
    """
    signs = exact_row_signs(matrix, direction)
    live = sides != 2
    if np.any(signs[live] == 2) or np.any(signs[sides == 0] != 0):
        return None
    boundary = np.flatnonzero((sides == 1) | (sides == -1))
    signed = sides[boundary].astype(np.int16) * signs[boundary]
    if np.any(signed < 0):
        return None
    strict = boundary[signed > 0]
    return strict if strict.size else None


def _snapped(direction: NDArray) -> list[NDArray]:
    """Proposals near ``direction``, scaled to unit maximum: on dyadic grids, coarsest first, then
    itself (a structural direction verifies on a coarse grid, where its rows sum exactly)."""
    largest = float(np.max(np.abs(direction), initial=0.0))
    if not (largest > 0.0 and math.isfinite(largest)):
        return []
    unit = direction / largest
    proposals = []
    for bits in _SNAP_BITS:
        scale = float(2**bits)
        proposals.append(np.round(unit * scale) / scale)
    proposals.append(unit)
    return proposals


def interior_null_directions(matrix, interior: NDArray) -> NDArray:
    """Directions the interior rows of ``matrix`` cannot tell from zero: the candidates.

    The first step of Correia, Guimaraes & Zylkin's App. A.2 (after Larch et al. 2019): a
    direction of recession is zero on every interior row, so it lies in the null space of
    their rows.  Read from the column-equilibrated Gram; a direction whose eigenvalue clears
    ``gamma_(n + 2) q`` of the largest is certainly not null (Weyl, the Gram's formation
    bound), and the rest are proposed.  The candidates only steer the linear program: the
    direction it finds is verified exactly (``_verified_separation``), so a candidate kept in
    excess costs work, never a decision.
    """
    width = matrix.shape[1]
    rows = matrix[np.flatnonzero(interior)]
    if rows.shape[0] == 0:
        return np.eye(width)
    gram = np.asarray((rows.T @ rows).todense(), dtype=np.float64)
    scale = np.sqrt(np.maximum(np.diag(gram), 0.0))
    untouched = scale == 0.0
    safe = np.where(untouched, 1.0, scale)
    values, vectors = np.linalg.eigh(gram / np.outer(safe, safe))
    largest = float(np.max(np.abs(values), initial=0.0))
    kept = values <= gamma(rows.shape[0] + 2) * width * max(largest, 1.0)
    candidates = vectors[:, kept] / safe[:, None]
    # an eigenvector's entries at its own rounding are a structural zero's noise: cut them, so a
    # row the candidates do not reach gives an exact zero and constrains no proposal
    reach = np.max(np.abs(candidates), axis=0, initial=0.0)
    candidates[np.abs(candidates) <= 2.0**-40 * reach] = 0.0
    return candidates


def program_rows(rows: NDArray, budget: int, priority: NDArray | None = None) -> NDArray:
    """The ``budget`` of ``rows`` (ascending row indices) a capped program reads, ascending.

    Half are the rows nearest their boundary at the walk (largest ``priority``, ``s_i eta_i``:
    a separated row is driven there, however few they are), ties in row order; the rest an
    evenly strided sample of the others, whose overlap keeps a direction that merely separates
    the first half from verifying.  Row positions only, so every platform reads the same rows
    wherever the priorities do not tie at the cut.
    """
    budget = max(int(budget), 0)
    if rows.size <= budget:
        return rows
    first = np.zeros(0, dtype=np.intp)
    if priority is not None:
        order = np.argsort(-np.asarray(priority, dtype=np.float64), kind="stable")
        first = np.sort(order[: budget // 2])
    rest = np.setdiff1d(np.arange(rows.size), first, assume_unique=True)
    count = budget - first.size
    strided = rest[(np.arange(count, dtype=np.int64) * rest.size) // max(count, 1)]
    return rows[np.sort(np.concatenate((first, strided)))]


def separating_direction(
    matrix, sides: NDArray, priority: Callable[[], NDArray | None] | None = None
) -> tuple[NDArray | None, NDArray, dict]:
    """A direction of recession of the rows of ``matrix``, verified exactly, with its rows.

    ``matrix`` (CSR) holds the rows' explicit columns, ``sides`` each row's side
    (``response_sides``).  The linear program is Correia, Guimaraes & Zylkin's App. A.2
    (eq. 19) on the candidates ``Phi`` (``interior_null_directions``): with ``Z = M_B Phi`` on
    the boundary rows, maximise ``sum t`` under ``t_i <= s_i Z_i phi / ||Z_i||``, ``0 <= t <= 1``
    and ``phi`` free.  Summing separating rays gives one strict on every separable row, and
    scaling it makes every such ``t_i`` one, so the optimum separates every row that can be
    separated (Silvapulle & Burridge 1986; Konis 2007; the capped-slack form Sol's gate review
    checks).  The program's answer is a proposal: ``Phi phi``, and its snaps to dyadic grids,
    are verified on every row exactly (``_verified_separation``), and the one separating the
    most rows is returned with them.  ``None`` when no proposal verifies.  The third value
    counts the work.

    A program past ``PROGRAM_CAP`` reads ``program_rows`` of its rows, ``priority`` (each row's
    ``s_i eta_i`` at the walk, formed only then; ``None`` for row order alone) choosing half:
    constraint sampling
    for a linear program of few variables and many constraints (Clarkson, *Las Vegas algorithms
    for linear and integer programming when the dimension is small*, J. ACM 42(2), 1995).  Its
    answer is verified on every row like any other, so a sample changes the work, not what a
    verified direction proves; a proposal that does not verify leaves the decision unresolved.
    """
    from scipy import sparse
    from scipy.optimize import linprog

    work = {"rows": int(matrix.shape[0]), "columns": int(matrix.shape[1])}
    candidates = interior_null_directions(matrix, sides == 0)
    work["candidates"] = int(candidates.shape[1])
    boundary = np.flatnonzero((sides == 1) | (sides == -1))
    if candidates.shape[1] == 0 or boundary.size == 0:
        return None, np.zeros(0, dtype=np.intp), work
    rows = matrix[boundary]
    values = np.asarray(rows @ candidates, dtype=np.float64)
    reach = np.asarray(abs(rows) @ np.abs(candidates), dtype=np.float64)
    largest = np.max(np.abs(values), axis=1)
    # a row the candidates move only at their own rounding constrains no proposal
    moved = largest > math.sqrt(UNIT_ROUNDOFF) * np.max(reach, axis=1)
    work["lp_rows"] = int(np.count_nonzero(moved))
    if not np.any(moved):
        return None, np.zeros(0, dtype=np.intp), work
    read = np.flatnonzero(moved)
    if read.size * candidates.shape[1] > PROGRAM_CAP:
        # past the cap the program reads a subset of the cap's size; its proposal is verified
        # on every row below, as any proposal is
        work["capped"] = PROGRAM_CAP
        ranked = None if priority is None else priority()
        ranked = None if ranked is None else np.asarray(ranked)[boundary][read]
        read = program_rows(read, PROGRAM_CAP // candidates.shape[1], ranked)
        work["sampled"] = int(read.size)
        if read.size == 0:
            return None, np.zeros(0, dtype=np.intp), work
    signed = (sides[boundary][read, None] * values[read]) / largest[read, None]
    count, width = signed.shape
    # HiGHS's interior-point method: on 200,000 boundary rows its dual simplex took minutes
    program = linprog(
        np.concatenate((np.zeros(width), -np.ones(count))),
        A_ub=sparse.hstack(
            (sparse.csr_matrix(-signed), sparse.identity(count, format="csr")), format="csr"
        ),
        b_ub=np.zeros(count),
        bounds=[(None, None)] * width + [(0.0, 1.0)] * count,
        method="highs-ipm",
    )
    work["lp_status"] = int(program.status)
    if program.status != 0 or program.x is None:
        return None, np.zeros(0, dtype=np.intp), work
    proposal = candidates @ np.asarray(program.x[:width], dtype=np.float64)
    # the rows the program separates: a snap that keeps them all is the answer
    target = int(np.count_nonzero(np.asarray(program.x[width:]) > 0.5))
    best: tuple[NDArray, NDArray] | None = None
    for direction in _snapped(proposal):
        strict = _verified_separation(matrix, sides, direction)
        if strict is not None and (best is None or strict.size > best[1].size):
            best = (direction, strict)
            if strict.size >= target:
                break
    work["verified"] = int(best is not None)
    if best is None:
        return None, np.zeros(0, dtype=np.intp), work
    return best[0], best[1], work


def exactly_null(matrix, direction: NDArray, rows: NDArray) -> bool:
    """Whether ``direction`` is exactly zero on ``rows`` of ``matrix`` (``exact_row_signs``)."""
    if rows.size == 0:
        return True
    return bool(np.all(exact_row_signs(matrix[rows], direction) == 0))


def exact_null_basis(matrix, directions: NDArray, rows: NDArray) -> NDArray | None:
    """A basis of ``span(directions)`` verified to vanish exactly on ``rows``, or ``None``.

    Each direction is reduced to a row-echelon basis of the span (pivoted on its largest
    entries, so a structural direction becomes small integers or dyadic fractions) and snapped
    (``_snapped``); the snap that vanishes exactly on ``rows`` is kept.  ``None`` when some
    direction has no exact snap: the span is then not known exactly at float64.
    """
    count = directions.shape[1]
    if count == 0:
        return np.zeros((directions.shape[0], 0))
    import scipy.linalg

    # an echelon basis: pivot on the largest entries, unit there and zero on the others' pivots
    _, _, pivots = scipy.linalg.qr(directions.T, pivoting=True, mode="economic")
    pivots = np.asarray(pivots[:count])
    echelon = directions @ np.linalg.solve(directions[pivots], np.eye(count))
    basis = []
    for column in echelon.T:
        exact = next((snap for snap in _snapped(column) if exactly_null(matrix, snap, rows)), None)
        if exact is None:
            return None
        basis.append(exact)
    return np.column_stack(basis)
