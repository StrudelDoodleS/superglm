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


def format_runtime_message(
    w_ratio: float,
    n_iter: int,
    drifting_groups: list[str],
    pinned: bool,
) -> str:
    """Terminal-state message for the in-solver backstop (issue #341)."""
    where = ""
    if drifting_groups:
        where = f" (largest drifting coefficients in group(s): {drifting_groups})"
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
        "categorical cells before fitting; separation this check reports at "
        "runtime instead involves non-categorical structure the design scan "
        "cannot see. Remove it (collapse levels, RandomEffect, or a selection "
        "penalty on the term), or pass separation='ignore' to silence this."
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


def validate_separation_mode(mode: str) -> str:
    if mode not in ("warn", "error", "ignore"):
        raise ValueError(f"separation must be 'warn', 'error', or 'ignore', got {mode!r}")
    return mode
