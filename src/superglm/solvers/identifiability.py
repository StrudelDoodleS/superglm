"""Which coefficients a penalised fit identifies: the one decision every standard error reads.

**The question.** A fit reports the penalised estimate, so a coefficient is identified when the
penalised Hessian ``H = X'WX + sum_j lambda_j S_j`` determines it, not when the data alone would
(classical estimability, which assumes no penalty).  With every ``lambda_j > 0`` and ``W > 0``,
``H v = 0`` exactly when ``X v = 0`` and ``S_j v = 0`` for every ``j`` (a sum of semidefinite
terms vanishes only when each does), so

    null(H) = N_S null(X_F),   X_F = X_c N_S,

with ``N_S`` the penalty's null space and ``X_c`` the design centred on the intercept: a direction
is lost only when neither the data nor any penalty reaches it, whatever the smoothing parameters
and however a term is parametrised (Wood, Pya & Saefken 2016, JASA 111, sec. 3.1.2; the
mixed-model split of Ruppert, Wand & Carroll 2003, sec. 4.9).  ``X_F`` holds the intercept, the
factor dummies, the numeric columns and the unpenalised part of each smooth: small, and on real
books well conditioned.  A partially aliased coefficient is identified; its standard error is
only larger, and the summary prints it.

**Deciding null(X_F).** Its Gram ``N_S' D N_S`` is a congruence of the centred data Gram ``D``
the fit already forms.  Jacobi-scaled, every eigenvalue above that Gram's error bound (the
eigensolver's ``p(n) eps`` and the formation bound ``gamma_n |X|'|X|``, as
``rank.decompose_gram_deflated`` bounds them) is certainly positive (Weyl), and when all are,
``X_F`` has full rank with no pass over the rows; on real books it does, by orders of magnitude.
Otherwise the rows are read on the Gram's undecided directions only, the eigenvectors whose
eigenvalue it cannot tell from zero, as ``rank.decompose_gram_deflated`` reads them for the data
rank: a singular value of that block at or below the rounding bound of its computation cannot be
told from zero in float64, and is a null direction of ``X_F`` by interlacing; one above it clears
the bound by more than what leaks from the certified directions (Li & Li's quadratic bound, the
coupling bounded by the Gram's own error) and is identified.  That is a rounding bound, not a
tolerance: an exactly aliased direction always falls under it, and an identified one above it
never does.  A structured factor certified its kept pivots itself (``border.factor_border``), so
its truncated directions are the only undecided ones: when the rows put every one of them under
the bound, they are ``null(H)`` and neither ``N_S`` nor ``X_F``'s Gram is formed.  Whatever these
reads leave unresolved, ``X_F`` is factored from the rows whole, which changes the cost of a
decision and never the decision.

**Which coordinate is blank.** Where ``H`` is singular and its published decomposition's null
directions ``N`` are ``null(H)``, one coordinate per direction is blank, the latest the direction
reaches (R's and ``gam.side``'s convention, the later or higher-order term absorbing the alias;
``rank._earliest_representatives``).  The fit publishes the coefficients with those coordinates
``E`` at zero and their covariance (``Gauge``), whichever backend it ran on: the dense solver
fixes them already unless its rank decision chose others for conditioning, and a structured
factor's minimum-norm split moves along ``N``, which changes no fitted value.  A
direction the published decomposition kept although ``null(X_F)`` holds it carries a split and a
variance of rounding, so every coordinate it reaches is blank; a coordinate the fit fixed that the
penalised problem identifies is blank too, since the fit reports zero there, and the record says
why.

**What each alias is.** An alias is named from the model's construction when counting proves it:
a column with no data and no penalty; factor levels nested in or disconnected from another
factor's (connectedness, Weeks & Williams, Technometrics 6, 1964); a level of a by-level term
with fewer distinct covariate values than its alias spans; basis functions with no data in an
unpenalised term.  Otherwise the record says no data separates the terms at float64.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import PENALTY_RANK_CUT, SQRT_EPS, UNIT_ROUNDOFF, gamma_up

# A coordinate takes part in a null direction when its share there, in Jacobi-scaled
# coordinates, is above this: the representative scan's floor (``rank._earliest_representatives``),
# below which a unit-norm null direction carries no information.  It names terms and reads which
# coordinates a direction reaches; whether a direction is null is the certificate's.
_SHARE_FLOOR = SQRT_EPS

# A standard error carries ``-log10(kappa u)`` digits (Demmel, LAPACK Working Note 14, 1989:
# Cholesky attains the accuracy of the diagonally scaled matrix); past this the summary says so.
_ACCURACY_NOTE = 1e-4

# The Gram's undecided directions are extended up to the first eigenvalue spacing of this many
# errors (``rank._GREY_GAP``), so the certified directions above them clear by a margin; past
# ``_GREY_CAP`` of them ``X_F`` is factored whole.
_GREY_GAP = 8.0
_GREY_CAP = 64


@dataclass(frozen=True)
class Gauge:
    """The coefficients a singular fit publishes: zero on ``coordinates`` ``E``, along ``null`` ``N``.

    ``null`` (``width x k``, fitted coordinates) spans ``null(H)``; with every lambda positive
    ``X_c N = 0`` and ``S N = 0``, so moving along it changes no fitted value and no penalty.
    ``E'N`` is nonsingular, so ``P = I - N (E'N)^-1 E'`` maps a coefficient vector to the one
    with ``E' beta = 0`` and the same fitted values, and a covariance ``V``, any generalised
    inverse of ``H``, to ``P V P'`` (each row of ``P`` is orthogonal to ``N``, and every
    generalised inverse has one bilinear form on ``range(H) = null(H)^perp``).  ``intercept``
    holds each direction's solver-intercept component, ``-c'N`` with ``c`` the fit's weighted
    column centre.
    """

    null: NDArray
    coordinates: NDArray
    intercept: NDArray

    def lift(self, columns: NDArray | None = None) -> NDArray:
        """``N (E'N)^-1`` on the fitted ``columns`` (every coordinate by default)."""
        null = self.null if columns is None else self.null[np.asarray(columns, dtype=np.intp)]
        return np.linalg.solve(self.null[self.coordinates].T, null.T).T

    def coefficients(self, beta: NDArray, intercept: float) -> tuple[NDArray, float, NDArray]:
        """``P beta``, the solver intercept that keeps every fitted value, and the step taken."""
        beta = np.asarray(beta, dtype=np.float64)
        step = np.linalg.solve(self.null[self.coordinates], beta[self.coordinates])
        gauged = beta - self.null @ step
        gauged[self.coordinates] = 0.0
        return gauged, float(intercept) - float(self.intercept @ step), step

    def covariance(self, covariance: NDArray, columns: NDArray) -> NDArray:
        """``P V P'`` of a dense covariance ``V`` held on the fitted ``columns``."""
        columns = np.asarray(columns, dtype=np.intp)
        V = np.asarray(covariance, dtype=np.float64)
        E = np.searchsorted(columns, self.coordinates)
        lift = self.lift(columns)
        VE = V[:, E]
        gauged = V - lift @ VE.T - VE @ lift.T + lift @ V[np.ix_(E, E)] @ lift.T
        gauged = 0.5 * (gauged + gauged.T)
        gauged[E, :] = 0.0
        gauged[:, E] = 0.0
        return gauged


@dataclass(frozen=True)
class Alias:
    """One fitted coordinate the fit does not identify, and what it is aliased with."""

    coordinate: int
    term: str
    aliased_with: tuple[str, ...]
    reason: str
    source: Literal["construction", "rank", "fit"]


@dataclass(frozen=True)
class Identifiability:
    """The identifiability record of one fit, in fitted (solver) coordinates.

    ``aliases`` are the coordinates without a standard error; every other coefficient is
    identified.  ``penalised_condition`` is the published decomposition's condition number, whose
    product with ``u`` is the standard errors' relative accuracy, and ``fixed_effect_condition``
    that of ``X_F`` (NaN when a certificate left ``X_F`` unformed).  ``gauge`` moves a singular
    fit's coefficients and covariance to the blanked coordinates' zeros where its decomposition
    did not fix them there; ``row_directions`` counts the directions read from the rows.
    """

    width: int
    aliases: tuple[Alias, ...]
    penalised_condition: float
    fixed_effect_condition: float
    gauge: Gauge | None = None
    row_directions: int = 0

    def coefficient_identified(self) -> NDArray:
        identified = np.ones(self.width, dtype=bool)
        identified[[alias.coordinate for alias in self.aliases]] = False
        return identified

    def alias_table(self):
        """Every alias, one row per coordinate: R's ``alias()`` "Complete" table."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "term": alias.term,
                    "coordinate": alias.coordinate,
                    "aliased_with": ", ".join(alias.aliased_with),
                    "reason": alias.reason,
                    "source": alias.source,
                }
                for alias in self.aliases
            ],
            columns=["term", "coordinate", "aliased_with", "reason", "source"],
        )

    def notes(self) -> tuple[str, ...]:
        """The summary's footnote: each alias with what it is aliased with, and the accuracy."""
        notes: list[str] = []
        if self.aliases:
            count = len(self.aliases)
            plural = "coefficient" if count == 1 else "coefficients"
            notes.append(f"{count} {plural} not identified (standard error shown as ---):")
            grouped: dict[tuple[str, tuple[str, ...], str], int] = {}
            for alias in self.aliases:
                key = (alias.term, alias.aliased_with, alias.reason)
                grouped[key] = grouped.get(key, 0) + 1
            for (term, others, reason), size in grouped.items():
                label = term if size == 1 else f"{term} ({size} coefficients)"
                link = f" is aliased with {', '.join(others)}" if others else ""
                notes.append(f"  {label}{link}: {reason}.")
        accuracy = self.penalised_condition * UNIT_ROUNDOFF
        if not self.aliases and accuracy > _ACCURACY_NOTE:
            digits = max(0, int(math.floor(-math.log10(accuracy)))) if accuracy < 1.0 else 0
            notes.append(
                f"Standard errors carry about {digits} significant digits: the penalised "
                f"Hessian's condition number is {self.penalised_condition:.1e}."
            )
        return tuple(notes)


@dataclass(frozen=True)
class TruncatedDecomposition:
    """A published decomposition of ``H`` known by the directions it truncated, rank ``rank``.

    A structured factor inverts ``H`` on the complement of its truncated directions
    ``null_vectors`` (``width x t``, the Moore-Penrose inverse of its Jacobi-scaled border), so
    every coordinate they reach is left undetermined, as a spectral decomposition's
    (``fixed_coordinates``).  ``column_scale`` is ``sqrt(diag H)``, and
    ``pre_truncation_condition`` the condition number the factor certified.
    """

    null_vectors: NDArray
    rank: int
    column_scale: NDArray
    pre_truncation_condition: float

    @property
    def width(self) -> int:
        return int(self.null_vectors.shape[0])

    @property
    def active_columns(self) -> NDArray:
        return np.arange(self.width)

    def null_basis(self) -> NDArray:
        return self.null_vectors


def certified_identifiability(decomposition, width: int) -> Identifiability | None:
    """The record of a fit whose structured factor certified ``H`` nonsingular; ``None`` otherwise.

    A structured factor keeps full rank only on certified pivots: its border by Rump's verified
    Cholesky for every matrix within its formation bound (``border.factor_border``), its levels
    positive definite by construction or kept above ``20 k eps`` of their columns.  ``H`` is then
    nonsingular, so ``null(H) = N_S null(X_F)`` is zero and every coefficient is identified, with
    no ``X_F`` Gram formed (``N_S`` holds every ``sz`` level's line).  A dense decomposition's full
    rank is not such a certificate, so only ``TruncatedDecomposition`` gives one.
    """
    if not isinstance(decomposition, TruncatedDecomposition):
        return None
    if decomposition.rank != decomposition.width:
        return None
    return Identifiability(
        width=width,
        aliases=(),
        penalised_condition=float(decomposition.pre_truncation_condition),
        fixed_effect_condition=float("nan"),
    )


def fixed_coordinates(decomposition) -> NDArray:
    """The coordinates a published decomposition of ``H`` leaves undetermined, in its own order.

    Full rank: none.  A representative decomposition (its rank equals its active columns) fixes
    the columns outside its face at zero.  A spectral one reports the minimum-norm split, which
    leaves every coordinate its null space reaches undetermined.
    """
    width = decomposition.width
    if decomposition.rank == width:
        return np.zeros(0, dtype=np.intp)
    fixed = np.setdiff1d(np.arange(width), np.asarray(decomposition.active_columns, dtype=np.intp))
    if decomposition.rank == width - fixed.size:
        return fixed
    return np.union1d(fixed, _reached(_orthonormal(decomposition, decomposition.null_basis())))


def coordinates_identified(decomposition) -> NDArray:
    """Whether each coordinate of a published decomposition of ``H`` is determined by it.

    A record without the explained decision (``decide_identifiability``) reads this.
    """
    identified = np.ones(decomposition.width, dtype=bool)
    identified[fixed_coordinates(decomposition)] = False
    return identified


def functional_identified(decomposition, contrast: NDArray) -> bool:
    """Whether a published decomposition of ``H`` determines the functional ``contrast' beta``.

    Wood, Pya & Saefken's criterion on the decomposition itself: in its Jacobi-scaled dual
    coordinates the contrast has no share above the floor on its null space, which with every
    lambda positive is ``null(H)``.  A unit contrast is identified exactly when no null direction
    reaches its coordinate (``_reached``), whatever the decomposition's representative.
    """
    null = _orthonormal(decomposition, decomposition.null_basis())
    if null.shape[1] == 0:
        return True
    scale = np.asarray(decomposition.column_scale, dtype=np.float64)
    scaled = np.asarray(contrast, dtype=np.float64) / np.where(scale > 0.0, scale, 1.0)
    norm = float(np.linalg.norm(scaled))
    return norm == 0.0 or float(np.linalg.norm(null.T @ scaled)) <= _SHARE_FLOOR * norm


def penalty_null_space(blocks: Sequence[NDArray], width: int) -> NDArray:
    """A basis of the null space a term's penalty blocks share, ``N_S,g``.

    The balanced penalty ``sum_j S_j / ||S_j||_F`` (Wood, Pya & Saefken 2016, sec. 3.1.2) has
    the null space of every ``S_j`` and no smoothing parameter in it; its eigenvalues are cut at
    ``PENALTY_RANK_CUT`` of the largest, the cut every penalty rank uses.  No penalty: the
    identity.
    """
    balanced = np.zeros((width, width))
    for block in blocks:
        norm = float(np.linalg.norm(block))
        if norm > 0.0:
            balanced += np.asarray(block, dtype=np.float64) / norm
    if not np.any(balanced):
        return np.eye(width)
    values, vectors = np.linalg.eigh(0.5 * (balanced + balanced.T))
    return vectors[:, values <= PENALTY_RANK_CUT * values[-1]]


def decide_identifiability(
    decomposition,
    *,
    columns: NDArray,
    width: int,
    groups: Sequence,
    null_spaces: Sequence[NDArray] | Callable[[], Sequence[NDArray]],
    data_gram,
    rows: int,
    design=None,
    specs: Mapping[Any, Any] | None = None,
    W: NDArray | None = None,
    data_diagonal: NDArray | None = None,
    penalty: Callable[[NDArray], tuple[NDArray, NDArray]] | None = None,
    gauge: bool = False,
) -> Identifiability:
    """The identifiability record of a fit with published decomposition ``decomposition`` of ``H``.

    ``columns`` maps the decomposition's coordinates to the ``width`` fitted ones, ``groups``
    are its term blocks (``GroupSlice``, in its coordinates) and ``null_spaces`` their penalty
    null spaces (``penalty_null_space``), or a callable that forms them when they are read.
    ``data_gram`` is the centred data Gram at the fit's weights, over ``rows`` rows: a dense
    matrix, or a compact operator (``matvec``) with its diagonal ``data_diagonal``, which a
    structured fit holds in place of the matrix; only its products with ``N_S`` are read.
    ``design`` (their rows, a ``DesignMatrix`` of those groups) and ``W`` read the directions the
    Gram cannot resolve from the rows, and ``specs`` (feature specs by name) name levels; without
    the rows such a direction counts as aliased.  ``penalty`` applies a structured factor's
    penalty to its truncated directions (``S T`` and the bound ``|S| |T|``), so that they decide
    the record alone (``_truncated_null``).  ``gauge`` lets the record move the published
    coefficients to its blanked coordinates' zeros (``Gauge``, its intercept components from the
    rows' weighted column means); without it, or without the rows, a split the decomposition did
    not publish at those zeros keeps every coordinate it reaches blank.
    """
    columns = np.asarray(columns, dtype=np.intp)
    owner_null, fixed_effect_condition, row_directions = None, float("nan"), 0
    if isinstance(decomposition, TruncatedDecomposition) and penalty is not None:
        owner_null, row_directions = _truncated_null(
            decomposition, groups, design, W, penalty, data_diagonal
        )
    if owner_null is None:
        basis = _null_space_basis(
            groups,
            cast(Callable[[], Sequence[NDArray]], null_spaces)()
            if callable(null_spaces)
            else null_spaces,
            decomposition.width,
        )
        if isinstance(data_gram, np.ndarray):
            gram_basis, diagonal = data_gram @ basis, np.diag(data_gram)
        else:
            gram_basis = data_gram.matvec(basis) if basis.shape[1] else np.zeros_like(basis)
            diagonal = np.asarray(data_diagonal, dtype=np.float64)
        owner_null, fixed_effect_condition, read = _fixed_effect_null(
            basis, basis.T @ gram_basis, diagonal, rows, groups, design, W
        )
        row_directions += read
    owner = np.empty(decomposition.width, dtype=np.intp)
    for index, group in enumerate(groups):
        owner[group.sl] = index
    null = _orthonormal(decomposition, owner_null)
    chosen = _gauge_coordinates(decomposition, null)
    zeros = np.setdiff1d(np.arange(decomposition.width), decomposition.active_columns)
    published_gauge = None
    movable = gauge and design is not None and W is not None
    if chosen is not None and (movable or np.all(np.isin(chosen, zeros))):
        blank = chosen
        if not np.all(np.isin(chosen, zeros)):
            published_gauge = _gauge(decomposition, chosen, columns, width, design, W)
    else:
        fixed = fixed_coordinates(decomposition)
        # The fit's zeros account for ``null(H)`` when no null direction vanishes on them.
        spanned = null.shape[1] == 0 or (
            fixed.size >= null.shape[1]
            and np.linalg.svd(null[fixed], compute_uv=False)[null.shape[1] - 1] > _SHARE_FLOOR
        )
        blank = fixed if spanned else np.union1d(fixed, _reached(null))
    facts = _RowFacts(design, W)
    aliases = []
    for coordinate in blank:
        share = np.abs(null @ null[coordinate])
        if not share[coordinate] > _SHARE_FLOOR**2:
            aliases.append(
                Alias(
                    coordinate=int(columns[coordinate]),
                    term=groups[owner[coordinate]].name,
                    aliased_with=(),
                    reason="the fit's rank decision fixed it at zero; the data and the "
                    "penalty identify it",
                    source="fit",
                )
            )
            continue
        term = int(owner[coordinate])
        involved = sorted({term, *(int(owner[i]) for i in _reached_by(share))})
        name = _coordinate_name(int(coordinate), groups[term], specs or {})
        reason = _construction_reason(
            int(coordinate), term, involved, decomposition, groups, facts, specs or {}, share
        )
        aliases.append(
            Alias(
                coordinate=int(columns[coordinate]),
                term=name,
                aliased_with=tuple(groups[i].name for i in involved if i != term),
                reason="no data separates them at float64" if reason is None else reason,
                source="rank" if reason is None else "construction",
            )
        )
    return Identifiability(
        width=width,
        aliases=tuple(aliases),
        penalised_condition=float(decomposition.pre_truncation_condition),
        fixed_effect_condition=fixed_effect_condition,
        gauge=published_gauge,
        row_directions=row_directions,
    )


def _gauge_coordinates(decomposition, null: NDArray) -> NDArray | None:
    """The coordinates ``E`` a singular fit publishes at zero, one per direction of ``null(H)``.

    ``null`` is the owner's ``null(H)``, orthonormal in Jacobi-scaled coordinates.  The published
    decomposition's null directions must span it, since they are what its covariance inverts
    off; then ``E`` holds the latest coordinate each direction reaches, by leverage above the
    share floor after the later directions are taken out (``rank._earliest_representatives``), so
    ``E'N`` is nonsingular at float64's resolution.  It reads the null space alone, so every
    backend makes the same choice.  ``None`` when the spans differ or no choice exists.
    """
    k = null.shape[1]
    if k == 0:
        return None
    published = _orthonormal(decomposition, decomposition.null_basis())
    if published.shape[1] != k:
        return None
    if np.linalg.svd(null.T @ published, compute_uv=False)[-1] < 1.0 - _SHARE_FLOOR:
        return None
    from superglm.solvers.rank import _earliest_representatives

    keep = _earliest_representatives(published, published.shape[0] - k)
    if keep is None:
        return None
    return np.setdiff1d(np.arange(published.shape[0]), keep)


def _gauge(decomposition, chosen: NDArray, columns: NDArray, width: int, design, W) -> Gauge:
    """The published null directions and ``chosen`` in fitted coordinates (``Gauge``).

    ``X_c N = 0`` on the fit's rows, so ``X N`` is ``c'N`` there, ``c`` the weighted column
    means of ``design`` at ``W``: the solver intercept moves by ``-c'N`` per unit step.
    """
    from superglm.solvers.centered_system import weighted_mean_pair

    raw = np.asarray(decomposition.null_basis(), dtype=np.float64)
    null = np.zeros((width, raw.shape[1]))
    null[columns] = raw
    weights = np.asarray(W, dtype=np.float64)
    mean, _, _ = weighted_mean_pair(design, weights, float(np.sum(weights)))
    intercept = -(np.asarray(mean, dtype=np.float64) @ raw)
    return Gauge(null=null, coordinates=columns[chosen], intercept=intercept)


def _coordinate_name(coordinate: int, group, specs: Mapping[Any, Any]) -> str:
    """A factor level's coefficient by its level, ``f[E]``; any other by its term."""
    levels = getattr(specs.get(group.feature_name), "_non_base", None)
    local = coordinate - group.start
    if group.name == group.feature_name and levels is not None and len(levels) == group.size:
        return f"{group.name}[{levels[local]}]"
    return group.name


def _null_space_basis(groups: Sequence, null_spaces: Sequence[NDArray], width: int) -> NDArray:
    """``N_S``, block diagonal by term, in the decomposition's coordinates."""
    blocks = [np.asarray(space, dtype=np.float64) for space in null_spaces]
    basis = np.zeros((width, sum(block.shape[1] for block in blocks)))
    column = 0
    for group, block in zip(groups, blocks, strict=True):
        basis[group.sl, column : column + block.shape[1]] = block
        column += block.shape[1]
    return basis


def _truncated_null(
    decomposition: TruncatedDecomposition,
    groups: Sequence,
    design,
    W: NDArray | None,
    penalty: Callable[[NDArray], tuple[NDArray, NDArray]],
    data_diagonal: NDArray | None,
) -> tuple[NDArray | None, int]:
    """A structured factor's truncated directions when the rows put each under its bound.

    The factor certified its kept pivots (Rump's verified Cholesky on its border, its levels
    positive definite), so ``nullity(H) <= t``, ``t`` its truncated directions ``T``.  On an
    orthonormal (Jacobi-scaled) basis of them the rows give the Rayleigh block ``T' H T = B'B +
    T'ST``, ``B = sqrt(W) X_c T`` (``_row_factor``).  By Courant-Fischer its eigenvalues bound
    the ``t`` smallest of ``H`` from above, so when every one is under the rounding bound of its
    computation, ``H`` has ``t`` eigenvalues that cannot be told from zero, and ``span(T)`` is
    ``null(H)``.  Otherwise (``None``) some truncated direction is identified, and the caller
    decides ``null(X_F)`` from the rows.  Returns the directions (raw coordinates) and how many
    were read from the rows.
    """
    if design is None or W is None or data_diagonal is None:
        return None, 0
    scale = np.asarray(decomposition.column_scale, dtype=np.float64)
    scaled = _orthonormal(decomposition, decomposition.null_vectors)
    if scaled.shape[1] == 0:
        return None, 0
    candidates = scaled / np.where(scale > 0.0, scale, 1.0)[:, None]
    factor, bound = _row_factor(candidates, groups, design, W, data_diagonal)
    applied, magnitude = penalty(candidates)
    curvature = candidates.T @ applied
    rayleigh = factor.T @ factor + 0.5 * (curvature + curvature.T)
    # ``|fl(T'ST) - T'ST| <= gamma_(2p) |T|'|S||T|`` (Higham 2002, eq. 3.13)
    rounding = gamma_up(2 * candidates.shape[0]) * float(
        np.linalg.norm(np.abs(candidates).T @ magnitude)
    )
    values = np.linalg.eigvalsh(rayleigh)
    if not np.all(np.sqrt(np.maximum(values, 0.0)) <= bound + math.sqrt(rounding)):
        return None, candidates.shape[1]
    return candidates, candidates.shape[1]


def _fixed_effect_null(
    basis: NDArray,
    gram: NDArray,
    data_diagonal: NDArray,
    rows: int,
    groups: Sequence,
    design,
    W: NDArray | None,
) -> tuple[NDArray, float, int]:
    """A basis of ``N_S null(X_F)`` in the decomposition's coordinates, ``X_F``'s condition.

    ``gram`` is ``X_F``'s Gram ``N_S' D N_S`` and ``data_diagonal`` the diagonal of ``D``.  A
    column of ``X_F`` whose Gram diagonal is zero has no data at all.  The rest is decided by
    the Gram when every Jacobi-scaled eigenvalue clears its error bound, which then certifies
    full rank (Weyl); otherwise by the rows on the eigenvectors it leaves undecided
    (``_grey_null``), and failing that by the rows whole (``_row_factor``): an eigenvector below
    the bound is a null vector only to within that bound over the spectral gap, so it is not
    read as one.  The condition is ``X_F``'s, in singular values.  The third value counts the
    directions read from the rows.
    """
    gram = 0.5 * (gram + gram.T)
    diagonal = np.diag(gram)
    active = np.flatnonzero(diagonal > 0.0)
    nulls = [basis[:, np.flatnonzero(~(diagonal > 0.0))]]
    if active.size == 0:
        return nulls[0], 1.0, 0
    scale = np.sqrt(diagonal[active])
    values, vectors = np.linalg.eigh(gram[np.ix_(active, active)] / np.outer(scale, scale))
    largest = float(values[-1])
    # ``|dD_ij| <= gamma_n s_i s_j`` (Cauchy-Schwarz on the centred columns), so the
    # congruence moves entry ``kl`` by at most ``gamma_n r_k r_l``, ``r = |N_S|' s``.
    from superglm.solvers.rank import _eigensolver_relative_bar, _formation_slack

    reach = (np.abs(basis[:, active]).T @ np.sqrt(np.maximum(data_diagonal, 0.0))) ** 2
    error = _eigensolver_relative_bar(active.size + 1) * largest + _formation_slack(
        gamma_up(rows + 2) * reach, diagonal[active]
    )
    if values[0] > error:
        return nulls[0], math.sqrt(largest / float(values[0])), 0
    columns = basis[:, active] / scale
    grey = int(np.searchsorted(values, error, side="right"))
    while grey < values.size and values[grey] - values[grey - 1] < _GREY_GAP * error:
        grey += 1
    read = 0
    if grey <= _GREY_CAP and grey < values.size and design is not None and W is not None:
        decided = _grey_null(
            columns @ vectors[:, :grey], values, grey, error, groups, design, W, data_diagonal
        )
        read = grey
        if decided is not None:
            null, smallest = decided
            nulls.append(null)
            return np.hstack(nulls), math.sqrt(largest) / smallest, read
    if design is None or W is None:
        return np.hstack((nulls[0], columns)), 1.0, read
    factor, bound = _row_factor(columns, groups, design, W, data_diagonal)
    _, singular, right_t = np.linalg.svd(factor)
    nulls.append(columns @ right_t.T[:, ~(singular > bound)])
    kept = singular[singular > bound]
    condition = float(kept[0] / kept[-1]) if kept.size else 1.0
    return np.hstack(nulls), condition, read + columns.shape[1]


def _grey_null(
    grey: NDArray,
    values: NDArray,
    count: int,
    error: float,
    groups: Sequence,
    design,
    W: NDArray,
    data_diagonal: NDArray,
) -> tuple[NDArray, float] | None:
    """``null(X_F)`` from the rows on the Gram's ``count`` undecided eigenvectors, or ``None``.

    In the computed eigenbasis ``[V_1, V_2]`` (``grey = N_S V_2 / s``) the Jacobi-scaled Gram
    is ``[[G_1, E], [E', G_2]]``: ``G_1`` has eigenvalues at least ``L_1 = w_(count+1) - e``
    (Weyl), ``||E|| <= e`` (the computed eigenpairs are exact for a matrix within ``e``), and
    ``G_2 = B'B`` with ``B = A grey`` read from the rows.  A singular value of ``B`` at or below
    its rounding bound is a null direction of ``X_F`` by interlacing (Courant-Fischer); one
    whose square clears the bound's by Li & Li's shift ``2 e^2 / (eta + sqrt(eta^2 + 4 e^2))``
    (Linear Algebra Appl. 395, 2005; ``eta`` the gap between the blocks) is identified.  Any
    other leaves the block undecided.  Returns the null directions and ``X_F``'s smallest kept
    singular value.
    """
    from superglm.solvers.rank import _li_li_shift

    factor, bound = _row_factor(grey, groups, design, W, data_diagonal)
    _, singular, right_t = np.linalg.svd(factor)
    floor = float(values[count]) - error
    gap = floor - (float(singular[0]) + bound) ** 2
    shift = _li_li_shift(2.0 * error, gap)
    null = singular <= bound
    identified = np.sqrt(np.maximum(singular * singular - shift, 0.0)) > bound
    if not np.all(null | identified):
        return None
    kept = singular[identified]
    smallest = min(float(kept[-1]), math.sqrt(floor)) if kept.size else math.sqrt(floor)
    return grey @ right_t.T[:, null], smallest


def _row_factor(
    columns: NDArray,
    groups: Sequence,
    design,
    W: NDArray,
    data_diagonal: NDArray,
) -> tuple[NDArray, float]:
    """The triangle of ``B = sqrt(W) X_c columns`` from the rows, and its singular values' bound.

    One pass forms ``B`` about the exact centre pair and its triangle
    (``grouped_projected_factor``).  An entry of ``B`` is within ``gamma_(p + T + 3)`` of
    ``|a_r|' |n_j|`` (``a_r`` a centred row, inflated where a centre is added after the
    product), and a column of ``|A|`` has norm ``sqrt(D_ii)``; the triangle adds ``gamma_(n k)
    ||B||_F`` (Higham 2002, Thm 19.4) and the SVD ``gamma_(k^2)``.  A singular value within
    their sum of zero cannot be told from zero in float64 (Weyl).
    """
    from superglm.solvers.centered_system import grouped_projected_factor, weighted_mean_pair

    width = columns.shape[1]
    weights = np.asarray(W, dtype=np.float64)
    _, centre, centre_lo = weighted_mean_pair(design, weights, float(np.sum(weights)))
    factor, _, _, inflated = grouped_projected_factor(
        design, weights, columns, center=centre, center_lo=centre_lo
    )
    factor = np.asarray(factor)
    if factor.shape[0] < width:
        factor = np.vstack((factor, np.zeros((width - factor.shape[0], width))))
    magnitude = np.maximum(
        np.sqrt(np.maximum(data_diagonal, 0.0)), np.asarray(inflated, dtype=np.float64)
    )
    formation = gamma_up(columns.shape[0] + len(groups) + 3) * (magnitude @ np.abs(columns))
    bound = float(np.linalg.norm(formation)) + gamma_up(
        int(design.n) * width + width * width
    ) * float(np.linalg.norm(factor))
    return factor, bound


def _orthonormal(decomposition, vectors: NDArray) -> NDArray:
    """An orthonormal basis of ``vectors`` in the decomposition's Jacobi-scaled coordinates.

    Each scaled vector is normalised first, so a direction on a column of tiny curvature (a
    column carried by rounding-level weights) is not cut as dependent on one of ordinary scale.
    """
    scale = np.asarray(decomposition.column_scale, dtype=np.float64)
    scaled = np.asarray(vectors, dtype=np.float64) * np.where(scale > 0.0, scale, 1.0)[:, None]
    norms = np.linalg.norm(scaled, axis=0)
    scaled = scaled[:, norms > 0.0] / norms[norms > 0.0]
    if scaled.shape[1] == 0:
        return scaled
    left, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    return left[:, singular > _SHARE_FLOOR * singular[0]]


def _reached(orthonormal: NDArray) -> NDArray:
    """The coordinates an orthonormal null basis reaches: leverage above the share floor."""
    leverage = np.einsum("ij,ij->i", orthonormal, orthonormal)
    return np.flatnonzero(leverage > _SHARE_FLOOR**2)


def _reached_by(share: NDArray) -> NDArray:
    largest = float(np.max(share, initial=0.0))
    return np.flatnonzero(share > _SHARE_FLOOR * largest) if largest > 0.0 else np.zeros(0, int)


class _RowFacts:
    """What the footnote reasons count on the rows, each read once per term for every alias.

    ``None`` throughout without the rows.  A factor's level rows come from one stable sort of
    its codes on the positive-weight rows; a spline block's empty basis functions from one pass.
    """

    def __init__(self, design, W: NDArray | None):
        self.matrices = None if design is None or W is None else design.group_matrices
        self.positive = None if W is None else np.asarray(W, dtype=np.float64) > 0.0
        self._codes: dict[int, NDArray] = {}
        self._levels: dict[int, tuple[NDArray, NDArray]] = {}
        self._empty: dict[int, int] = {}

    def is_factor(self, index: int) -> bool:
        from superglm._group_matrix._group_matrix_core import (
            CategoricalGroupMatrix,
            RandomEffectGroupMatrix,
        )

        matrix = cast(Sequence, self.matrices)[index]
        return isinstance(matrix, CategoricalGroupMatrix) and not isinstance(
            matrix, RandomEffectGroupMatrix
        )

    def codes(self, index: int) -> NDArray:
        """The factor's codes on the positive-weight rows."""
        if index not in self._codes:
            codes = np.asarray(cast(Sequence, self.matrices)[index].codes, dtype=np.intp)
            self._codes[index] = codes[cast(NDArray, self.positive)]
        return self._codes[index]

    def level_rows(self, index: int, level: int) -> NDArray:
        """The positive-weight rows of one level of a factor."""
        if index not in self._levels:
            rows = np.flatnonzero(cast(NDArray, self.positive))
            order = np.argsort(self.codes(index), kind="stable")
            self._levels[index] = (self.codes(index)[order], rows[order])
        codes, rows = self._levels[index]
        lo, hi = np.searchsorted(codes, [level, level + 1])
        return rows[lo:hi]

    def empty_basis_functions(self, index: int) -> int:
        if index not in self._empty:
            self._empty[index] = _empty_basis_functions(
                cast(Sequence, self.matrices)[index], cast(NDArray, self.positive)
            )
        return self._empty[index]


def _construction_reason(
    coordinate: int,
    term: int,
    involved: list[int],
    decomposition,
    groups: Sequence,
    facts: _RowFacts,
    specs: Mapping[Any, Any],
    share: NDArray,
) -> str | None:
    """The construction that proves this alias by counting, or ``None``."""
    if facts.matrices is None:
        return (
            "it has no data and no penalty" if decomposition.column_scale[coordinate] == 0 else None
        )
    if len(involved) >= 2 and all(facts.is_factor(i) for i in involved):
        return _factor_reason(term, involved, groups, facts)
    factors = [i for i in involved if facts.is_factor(i)]
    if len(involved) == 2 and len(factors) == 1:
        by_level = next(i for i in involved if i != factors[0])
        return _level_reason(by_level, factors[0], groups, facts, specs, share)
    if involved == [term]:
        empty = facts.empty_basis_functions(term)
        if empty:
            plural = "function has" if empty == 1 else "functions have"
            return f"{empty} of its basis {plural} no data, and it has no penalty there"
    if float(decomposition.column_scale[coordinate]) == 0.0:
        # no penalty, and a centred column of zero: constant on the fit's rows
        unit = np.zeros(groups[term].size)
        unit[coordinate - groups[term].start] = 1.0
        column = np.asarray(facts.matrices[term].matvec(unit))[facts.positive]
        if np.any(column != 0.0):
            return (
                "it is constant on the fit's rows, which the intercept carries, and has no penalty"
            )
        return "it has no data and no penalty"
    return None


def _factor_reason(term, involved, groups, facts: _RowFacts) -> str | None:
    """Nested or disconnected factor levels, by the cross-classification's connectivity."""
    codes = [facts.codes(i) for i in involved]
    later = involved.index(term)
    if len(involved) == 2:
        earlier = 1 - later
        pairs = np.unique(np.stack((codes[earlier], codes[later])), axis=1)
        if np.unique(pairs[0]).size == pairs.shape[1]:
            name, coarse = groups[involved[earlier]].name, groups[term].name
            return f"each level of {coarse} is a union of levels of {name}"
    sets = _connected_sets(codes)
    if sets < 2:
        return None
    names = [groups[i].name for i in involved]
    listed = " and ".join(names) if len(names) == 2 else ", ".join(names[:-1]) + f" and {names[-1]}"
    return f"the levels of {listed} fall into {sets} sets that share no rows"


def _connected_sets(codes: list[NDArray]) -> int:
    """Connected components of the levels, two levels joined by a row holding both (union-find)."""
    offsets = np.cumsum([0] + [int(code.max(initial=-1)) + 1 for code in codes])
    parent = np.arange(int(offsets[-1]))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    present = np.zeros(int(offsets[-1]), dtype=bool)
    for factor, code in enumerate(codes):
        present[offsets[factor] + code] = True
    for row in np.unique(np.stack(codes), axis=1).T:
        nodes = [int(offsets[factor] + level) for factor, level in enumerate(row)]
        root = find(nodes[0])
        for node in nodes[1:]:
            other = find(node)
            if other != root:
                parent[other] = root
    return len({find(int(node)) for node in np.flatnonzero(present)})


def _level_reason(term, factor, groups, facts: _RowFacts, specs, share) -> str | None:
    """A level of a by-level term with too few distinct covariate rows for its alias, counted."""
    largest = float(np.max(share, initial=0.0))
    levels = np.flatnonzero(share[groups[factor].sl] > _SHARE_FLOOR * largest)
    if levels.size != 1:
        return None
    level = int(levels[0])
    rows = facts.level_rows(factor, level)
    own = groups[term].sl
    spanned = int(np.count_nonzero(share[own] > _SHARE_FLOOR * largest))
    distinct = (
        0
        if rows.size == 0
        else np.unique(
            cast(Sequence, facts.matrices)[term].row_subset(rows).toarray(), axis=0
        ).shape[0]
    )
    # ``distinct`` points carry at most that many independent functions, and with the level's
    # dummy the alias spans ``spanned + 1`` of them: the count proves the dependence.
    if distinct > spanned:
        return None
    factor_name = groups[factor].feature_name or groups[factor].name
    labels = getattr(specs.get(factor_name), "_non_base", None)
    label = labels[level] if labels is not None and level < len(labels) else level
    spec = specs.get(groups[term].feature_name)
    parents = [p for p in (getattr(spec, "parent_names", None) or ()) if p != factor_name]
    covariate = parents[0] if parents else groups[term].name
    plural = "value" if distinct == 1 else "values"
    return f"level {label} of {factor_name} has {distinct} distinct {plural} of {covariate}"


def _empty_basis_functions(matrix, positive: NDArray) -> int:
    """How many raw basis functions of a spline block no positive-weight row reaches."""
    raw = getattr(matrix, "B", None)
    if raw is not None and hasattr(raw, "tocsr"):
        reached = np.asarray((raw.tocsr()[np.flatnonzero(positive)] != 0).sum(axis=0)).ravel()
        return int(np.count_nonzero(reached == 0))
    table, bins = getattr(matrix, "B_unique", None), getattr(matrix, "bin_idx", None)
    if table is not None and bins is not None:
        occupied = np.bincount(np.asarray(bins)[positive], minlength=np.shape(table)[0]) > 0
        reached = np.count_nonzero(np.asarray(table)[occupied] != 0, axis=0)
        return int(np.count_nonzero(reached == 0))
    return 0


__all__ = [
    "Alias",
    "Gauge",
    "Identifiability",
    "TruncatedDecomposition",
    "certified_identifiability",
    "coordinates_identified",
    "decide_identifiability",
    "fixed_coordinates",
    "functional_identified",
    "penalty_null_space",
]
