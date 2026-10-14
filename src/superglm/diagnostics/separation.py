"""Build-time detection of separated categorical cells and levels.

A categorical level, or a crossed-categorical cell, that carries exposure but
whose responses all sit on the response distribution's boundary (all ``y == 0``
for a log-link Poisson / Tweedie / negative-binomial fit; all ``y == 0`` or all
``y == 1`` for a binomial fit) has no finite maximum-likelihood estimate.  The
likelihood increases monotonically as the cell's linear predictor walks toward
the boundary, so IRLS drifts until the objective stagnates, declares
convergence, and returns fitted values collapsed to the boundary.  Rank and
aggregate metrics (gini, balance) look healthy on such a fit; only the
out-of-sample likelihood/deviance exposes it.

This is the classical nonexistence problem for exponential-family maximum
likelihood: the estimate exists iff the sufficient statistic lies in the
relative interior of its marginal cone (Haberman 1974, *The Analysis of
Frequency Data*; Fienberg & Rinaldo 2012, *Ann. Statist.* 40(2) 996-1023), and
in the binomial case it is complete / quasi-complete separation (Albert &
Anderson 1984, *Biometrika* 71(1) 1-10).  Detection in a general design is a
linear program over the columns (Konis 2007; Kosmidis' ``detectseparation``),
but for indicator blocks the coordinate directions of recession are exactly
the cells with exposure and boundary-only response, so the scan below is exact
for the block structure it covers -- and O(n) rather than an LP.

The scan runs at design-build time, before any IRLS iteration, on both the
dense and ``discrete=True`` paths (they share the builder).  A term whose
block is bounded by an active SELECTION penalty is skipped: that penalty grows
without bound along any recession direction, so the penalised optimum is finite
and the term is estimable as specified.  Only ``selection_penalty`` is
consulted (``dm_builder`` at the exemption site) -- a ridge would bound the
term too, but ``Categorical`` blocks carry no ``lambda2``, so there is nothing
to check and claiming otherwise would describe a test that does not run.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

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
    """Refusal to fit (or certify) a design containing separated cells."""


@dataclass(frozen=True)
class SeparatedTerm:
    """Separated cells found in one categorical or crossed-categorical term."""

    term: str
    kind: str  # "levels" (main effect) or "cells" (crossed)
    boundary: str  # "zero" or "one"
    labels: list[Any]  # level names, or (level1, level2) pairs
    n_occupied: int  # occupied levels/cells scanned in this term


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


def _level_codes(x: NDArray, spec: Any, *, context: str) -> tuple[NDArray, list[Any]]:
    """Row codes against a built Categorical spec's full level universe.

    Applies the same raw -> collapsed -> fitted-domain label contract the
    interaction builder uses, then maps rows an ``unseen="base"`` policy
    routes to the base level onto the base index so their exposure and
    response anchor the cell they actually train in.
    """
    from superglm.features.categorical import _codes_against
    from superglm.features.interaction import _categorical_build_labels

    labels = _categorical_build_labels(x, spec, context=context)
    levels = list(spec._levels)
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


def scan_categorical_term(
    name: str,
    spec: Any,
    x: NDArray,
    y: NDArray,
    sample_weight: NDArray,
    boundaries: tuple[str, ...],
) -> list[SeparatedTerm]:
    """Scan one main-effect Categorical term for separated levels.

    Every level of the universe is scanned, base included: with the level's
    own indicator the recession direction is that coordinate; for the base
    level it is the intercept walking against every non-base indicator.
    """
    codes, levels = _level_codes(x, spec, context=name)
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


def drifting_groups(
    design: Any,
    groups: list,
    beta: NDArray,
    eta: NDArray,
    eta_unclipped: NDArray,
    weights: NDArray,
) -> list[str]:
    """The groups that carry the linear predictor to where the walk stopped.

    The rows are those pinned at the link's guard; with none pinned (an
    exhausted, stagnant walk) the rows whose predictor lies farthest from the
    bulk, at least half as far as the farthest.  A group is named by how far
    its own contribution moves those rows from where it puts the bulk (its
    median over the rows with weight), which compares across groups; the size
    of its coefficients does not, since each group's coefficients are in its
    own reparametrised scale (a tensor's can dwarf the curve that walked), and
    nor does its raw contribution, whose constant part -- ``mean * beta`` for a
    covariate far from zero -- the intercept cancels.  Groups reaching at least
    half the farthest are named, at most five.
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
    reach = []
    for group, matrix in zip(groups, design.group_matrices, strict=True):
        coefficients = beta[group.sl]
        if not coefficients.size or group.scop_reparameterization is not None:
            continue
        contribution = np.asarray(matrix.matvec(coefficients), dtype=np.float64)
        moved = contribution[rows] - np.median(contribution[live])
        reach.append((float(np.max(np.abs(moved))), group.name))
    if not reach:
        return []
    farthest = max(value for value, _ in reach)
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
        raise SeparationError(message)
    # stacklevel points past the builder into the user's fit call region; the
    # exact frame depth varies by entrypoint, so the message carries the term
    # names rather than relying on the reported line.
    warnings.warn(message, SeparationWarning, stacklevel=3)


def emit_curve_separation(curves: list, mode: str, distribution: Any) -> None:
    """Warn, or refuse, for spline-by-factor curves held over a separated range."""
    if not curves or mode == "ignore":
        return
    message = format_curve_separation(curves, distribution)
    if mode == "error":
        raise SeparationError(message)
    warnings.warn(message, SeparationWarning, stacklevel=3)


def validate_separation_mode(mode: str) -> str:
    """Return ``mode`` unchanged when it names a supported separation policy."""

    if mode not in ("warn", "error", "ignore"):
        raise ValueError(f"separation must be 'warn', 'error', or 'ignore', got {mode!r}")
    return mode


# ── sz factor smooths: each level's unpenalized line ───────────────────────


def _rounding(count: int) -> float:
    """Higham's ``gamma_n = n u / (1 - n u)`` (2002, Lemma 3.1), ``u = eps / 2``."""
    unit = float(np.finfo(np.float64).eps) / 2.0
    return count * unit / (1.0 - count * unit)


def _recedes(sign: NDArray, natural: NDArray, null_space: NDArray, free: NDArray) -> bool:
    """Whether some ``g = b(x)' N_P Z h`` has ``sign * g >= 0`` on every row, ``> 0`` on one.

    ``natural`` are the boundary rows' natural basis rows and ``sign`` the
    side each must keep (``-1``: ``g <= 0`` where the response sits on the
    lower boundary, ``+1``: the upper one); ``free`` (``Z``) spans the
    directions of ``N_P`` the level's interior rows do not see.  A
    least-squares direction with ``sign * g = 1`` settles complete
    separation; one free direction is its two signs; otherwise the linear
    program over the rows (Konis 2007; Geyer 2009, directions of recession)
    maximizes ``sum sign * g`` within ``|h| <= 1``.  A direction counts only
    when its evaluation clears its own rounding, ``gamma_(k + m + q)`` times
    ``|b| |N_P| |Z| |h|`` (Higham 2002, section 3.5).
    """
    rows = natural @ null_space @ free
    D = sign[:, None] * rows
    k, m, q = natural.shape[1], null_space.shape[1], free.shape[1]
    scale = np.abs(natural) @ (np.abs(null_space) @ np.abs(free))
    slack = _rounding(k + m + q)

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


def separated_factor_smooth_levels(
    dominant: Any,
    null_space: NDArray,
    prior_weights: NDArray | None,
    y: NDArray,
    boundaries: tuple[str, ...],
) -> tuple[int, ...]:
    """The ``sz`` levels whose unpenalized line separates the response.

    An ``sz`` level's deviation keeps its polynomial part ``b(x)' N_P f``
    unpenalized, and moving it with every level by ``-1 / K`` and the main
    effect's polynomial by ``+1 / K`` changes the level's linear predictor
    alone.  So the level has no finite estimate when some such ``g`` is
    ``<= 0`` on its rows at the lower boundary (``y == 0`` under a log link
    with mass at zero), ``>= 0`` on those at the upper one (``y == 1``,
    binomial), zero on its interior rows and not zero throughout: a direction
    of recession (Geyer 2009, Theorem 4), the categorical scan's rule above
    for a block of indicators.  Only a level with fewer than ``m`` distinct
    interior ``x`` values can have one.  A level whose rows all sit on one
    boundary separates along the constant when the basis reproduces it
    (``b(x)' N_P v = raw(x)' w`` with ``w`` the unit vector, a partition of
    unity, positive on every row): one sparse product for every such level.
    Any other candidate is decided on its own rows (``_recedes``).
    """
    from superglm.solvers._structured.layout import (
        first_distinct_level_rows,
        level_natural_rows,
    )

    nullity = null_space.shape[1]
    if not boundaries or not nullity:
        return ()
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
        return ()
    separated = np.zeros(n_levels, dtype=bool)

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
            clear = value > _rounding(nnz) * bound
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
            natural = level_natural_rows(dominant, level_rows)
            unique, inverse = np.unique(natural, axis=0, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            at_lower = np.zeros(len(unique), dtype=bool)
            at_upper = np.zeros(len(unique), dtype=bool)
            within = np.zeros(len(unique), dtype=bool)
            np.logical_or.at(at_lower, inverse, lower[level_rows])
            np.logical_or.at(at_upper, inverse, upper[level_rows])
            np.logical_or.at(within, inverse, inside[level_rows])
            # an x with responses on both boundaries holds g at zero as an interior one does
            anchor = within | (at_lower & at_upper)
            anchored = unique[anchor] @ null_space
            held = min(len(anchored), nullity)
            if held >= nullity:
                continue
            free = (
                np.eye(nullity)
                if not held
                else np.linalg.svd(anchored, full_matrices=True)[2][held:].T
            )
            side = np.where(anchor, 0.0, np.where(at_lower, -1.0, 1.0))
            edge = side != 0.0
            if np.any(edge) and _recedes(side[edge], unique[edge], null_space, free):
                separated[level] = True
    return tuple(int(level) for level in np.flatnonzero(separated))


#: What a term whose ``sz`` lines are unpenalized can do about it (#444): the
#: lines' null-space penalty is the term's own option, never the data's.
SZ_SELECT_REMEDY = (
    "To give every level a finite, unique curve, pass select=True to this FactorSmooth, "
    "which penalizes every level's line with a smoothing parameter of its own (the "
    "null-space penalty of mgcv's select=TRUE), or use basis='fs'."
)


def format_factor_smooth_separation(name: str, labels: list[Any], n_levels: int) -> str:
    """The warning for ``sz`` levels whose unpenalized line separates (``separated_factor_smooth_levels``)."""
    return (
        f"FactorSmooth {name!r} (basis='sz'): {len(labels)} of {n_levels} levels have an "
        f"unpenalized line that separates the response: {_format_labels(labels)}. The "
        "likelihood keeps increasing along each such line, so their fitted values walk to "
        "the response boundary for as long as the fit runs. They are left out of the "
        f"population curve, the mean of the levels the data identify. {SZ_SELECT_REMEDY} "
        "separation='ignore' silences this warning."
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


#: How far a linear program must move a row, of unit length, for it to count as
#: moved: a hundred times HiGHS's default primal feasibility tolerance (1e-7),
#: so a solver residual never reads as a held row.  Rows a direction does hold
#: move by a property of the cone's geometry, measured at 0.018 and above on
#: the freMTPL2 guide model's levels.
_PROGRAM_TOL = 1e-5


def _held_rows(D: NDArray) -> NDArray:
    """The rows some direction ``a`` moves strictly the right way, ``D a > 0``, with the others unmoved.

    Such directions form a cone, and the rows it moves are the complement of
    the linearity in Geyer (2009, section 3.12), which he finds with one
    linear program per row.  One program first maximises how many rows a
    direction moves, ``sum t`` under ``D a >= t``, ``0 <= t <= 1``, with
    ``|a| <= 1``; a row the cone moves only together with a larger cost to
    the others can be left at ``t = 0`` there, so each row it leaves gets its
    own program, maximising ``D_j a`` under ``D a >= 0``.  Each row is scaled
    to unit length first, which leaves its sign, and so the cone, unchanged
    and keeps a row the free directions see only faintly within the
    solver's tolerances; the box keeps the programs bounded, as HiGHS fails
    on the free ones once rows differ in scale.  The rows passed in are ones
    the free directions see by more than the rank cut
    (``separated_curve_support``).
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
    for row in np.flatnonzero(~held):
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

    ``raw`` holds the level's distinct raw basis rows (one per covariate value,
    or bin, that carries positive weight), ``projection`` maps the scanned
    coefficients to raw ones, and ``lower`` / ``upper`` / ``inside`` say
    whether a point holds rows on the lower boundary (``y == 0`` under a log
    link with mass at zero), the upper one (``y == 1``, binomial) or strictly
    between.  A point holding an interior row, or rows on both boundaries,
    anchors the curve: any direction of recession leaves it unmoved.  The
    others hold rows on one boundary only, and a direction ``h`` with
    ``g = raw projection h`` zero at every anchor, ``g <= 0`` where the
    responses sit on the lower boundary and ``g >= 0`` on the upper, and not
    zero throughout, raises the likelihood without bound: a direction of
    recession (Geyer 2009, Theorem 4; the linear-programming test of Konis
    2007), here inside the curve's penalised range.  Points some such
    direction moves are the separated ones.

    Returns ``(separated, free)``: a mask over the points and an orthonormal
    basis of the scanned directions every other point leaves unseen
    (``_unseen_directions``), in which every direction of recession lies;
    ``None`` when no point separates.  The set is settled when a direction in
    ``free`` moves every point of it the right way; a point no such direction
    moves is returned to the others, which can only shrink ``free``, so the
    loop ends.
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

    ``free`` (raw coefficients, orthonormal columns) spans the curve
    directions only a level's separated rows see.  The penalised likelihood
    keeps rising along a direction of recession, so those rows cannot fix
    the curve there; it is held where the penalty alone puts it given the
    rest of the curve -- its minimum-penalty continuation, ``free' omega f =
    0`` for the level's raw curve ``f``, which is how the penalty sets a curve
    over a range with no rows.  The level's own constant lies in ``omega``'s
    null space, so the condition reads on the block coefficients ``h`` alone,
    ``free' omega projection h = 0``.  A ``free`` direction the penalty does
    not see either (``omega_null``: a straight line no unseparated row pins)
    has no continuation; the block's own unpenalised directions
    (``projected_null``) are then held at zero, so the line comes from the
    rest of the model.  Returns an orthonormal basis of the block coefficients
    left, or ``None`` when none is.
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


def format_curve_separation(findings: list[SeparatedCurve], distribution: Any = None) -> str:
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
        lines.append(f"  {item.term!r}, level {item.level!r}: " + "; ".join(spans))
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
        "fit holds the curve over each such range instead:"
    )
    tail = (
        "Past the first or last value where a level's responses vary, its curve keeps its "
        "value there, as a spline keeps its value past its training range "
        "(extrapolation='clip'), in the fit and in predict(). To let the data shape these "
        "ranges, merge the level into a similar one (Categorical(grouping=...)) or drop the "
        "interaction. separation='ignore' leaves the curves free there, as earlier versions "
        "did; separation='error' refuses the design."
    )
    return "\n".join([head, *lines, tail])
