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
Otherwise ``X_F`` is factored from the rows in one pass, and a singular value at or below the
rounding bound of that computation cannot be told from zero in float64.  That is a rounding
bound, not a tolerance: an exactly aliased direction always falls under it, and an identified
one above it never does.

**Which coordinate is blank.** Where ``H`` is singular, the fit fixes one coordinate per null
direction at zero, the latest the direction reaches (R's and ``gam.side``'s convention, the later
or higher-order term absorbing the alias; ``rank._earliest_representatives``), and its covariance
is the one on the face it keeps.  When those coordinates span ``null(H)`` they are the aliases.  A
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

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
    that of ``X_F`` (NaN when a certificate left ``X_F`` unformed).
    """

    width: int
    aliases: tuple[Alias, ...]
    penalised_condition: float
    fixed_effect_condition: float

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
    null_spaces: Sequence[NDArray],
    data_gram,
    rows: int,
    design=None,
    specs: Mapping[Any, Any] | None = None,
    W: NDArray | None = None,
    data_diagonal: NDArray | None = None,
) -> Identifiability:
    """The identifiability record of a fit with published decomposition ``decomposition`` of ``H``.

    ``columns`` maps the decomposition's coordinates to the ``width`` fitted ones, ``groups``
    are its term blocks (``GroupSlice``, in its coordinates) and ``null_spaces`` their penalty
    null spaces (``penalty_null_space``).  ``data_gram`` is the centred data Gram at the fit's
    weights, over ``rows`` rows: a dense matrix, or a compact operator (``matvec``) with its
    diagonal ``data_diagonal``, which a structured fit holds in place of the matrix; only its
    products with ``N_S`` are read.  ``design`` (their rows, a ``DesignMatrix`` of those groups)
    and ``W`` read the directions the Gram cannot resolve from the rows, and ``specs``
    (feature specs by name) name levels; without the rows such a direction counts as aliased.
    """
    columns = np.asarray(columns, dtype=np.intp)
    basis = _null_space_basis(groups, null_spaces, decomposition.width)
    if isinstance(data_gram, np.ndarray):
        gram_basis, diagonal = data_gram @ basis, np.diag(data_gram)
    else:
        gram_basis = data_gram.matvec(basis) if basis.shape[1] else np.zeros_like(basis)
        diagonal = np.asarray(data_diagonal, dtype=np.float64)
    owner_null, fixed_effect_condition = _fixed_effect_null(
        basis, basis.T @ gram_basis, diagonal, rows, groups, design, W
    )
    owner = np.empty(decomposition.width, dtype=np.intp)
    for index, group in enumerate(groups):
        owner[group.sl] = index
    fixed = fixed_coordinates(decomposition)
    null = _orthonormal(decomposition, owner_null)
    # The fit's zeros account for ``null(H)`` when no null direction vanishes on them.
    spanned = null.shape[1] == 0 or (
        fixed.size >= null.shape[1]
        and np.linalg.svd(null[fixed], compute_uv=False)[null.shape[1] - 1] > _SHARE_FLOOR
    )
    blank = fixed if spanned else np.union1d(fixed, _reached(null))
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
            int(coordinate), term, involved, decomposition, groups, design, specs or {}, W, share
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
    )


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


def _fixed_effect_null(
    basis: NDArray,
    gram: NDArray,
    data_diagonal: NDArray,
    rows: int,
    groups: Sequence,
    design,
    W: NDArray | None,
) -> tuple[NDArray, float]:
    """A basis of ``N_S null(X_F)`` in the decomposition's coordinates, and ``X_F``'s condition.

    ``gram`` is ``X_F``'s Gram ``N_S' D N_S`` and ``data_diagonal`` the diagonal of ``D``.  A
    column of ``X_F`` whose Gram diagonal is zero has no data at all.  The rest is decided by
    the Gram when every Jacobi-scaled eigenvalue clears its error bound, which then certifies
    full rank (Weyl), and otherwise by the rows (``_factor_singular_values``): an eigenvector
    below the bound is a null vector only to within that bound over the spectral gap, so it is
    not read as one.  The condition is ``X_F``'s, in singular values.
    """
    gram = 0.5 * (gram + gram.T)
    diagonal = np.diag(gram)
    active = np.flatnonzero(diagonal > 0.0)
    nulls = [basis[:, np.flatnonzero(~(diagonal > 0.0))]]
    if active.size == 0:
        return nulls[0], 1.0
    scale = np.sqrt(diagonal[active])
    values = np.linalg.eigvalsh(gram[np.ix_(active, active)] / np.outer(scale, scale))
    largest = float(values[-1])
    # ``|dD_ij| <= gamma_n s_i s_j`` (Cauchy-Schwarz on the centred columns), so the
    # congruence moves entry ``kl`` by at most ``gamma_n r_k r_l``, ``r = |N_S|' s``.
    from superglm.solvers.rank import _eigensolver_relative_bar, _formation_slack

    reach = (np.abs(basis[:, active]).T @ np.sqrt(np.maximum(data_diagonal, 0.0))) ** 2
    error = _eigensolver_relative_bar(active.size + 1) * largest + _formation_slack(
        gamma_up(rows + 2) * reach, diagonal[active]
    )
    if values[0] > error:
        return nulls[0], math.sqrt(largest / float(values[0]))
    singular, right = _factor_singular_values(
        basis[:, active], scale, groups, design, W, data_diagonal
    )
    nulls.append((basis[:, active] / scale) @ right[:, ~(singular > 0.0)])
    kept = singular[singular > 0.0]
    condition = float(kept[0] / kept[-1]) if kept.size else 1.0
    return np.hstack(nulls), condition


def _factor_singular_values(
    columns: NDArray,
    scale: NDArray,
    groups: Sequence,
    design,
    W: NDArray | None,
    data_diagonal: NDArray,
) -> tuple[NDArray, NDArray]:
    """``X_F``'s Jacobi-scaled singular values from the rows, zero where rounding explains them.

    One pass forms ``B = sqrt(W) X_c N_S`` about the exact centre pair and its triangle
    (``grouped_projected_factor``).  An entry of ``B`` is within ``gamma_(p + T + 3)`` of ``|a_r|'
    |n_j|`` (``a_r`` a centred row, inflated where a centre is added after the product), and a
    column of ``|A|`` has norm ``sqrt(D_ii)``; the triangle adds ``gamma_(n k) ||B||_F`` (Higham
    2002, Thm 19.4) and the SVD ``gamma_(k^2)``.  A singular value within their sum of zero cannot
    be told from zero in float64 (Weyl), and is zero, as is every one without the rows.  Returns
    the singular values and right singular vectors.
    """
    width = columns.shape[1]
    if design is None or W is None:
        return np.zeros(width), np.eye(width)
    from superglm.solvers.centered_system import grouped_projected_factor, weighted_mean_pair

    weights = np.asarray(W, dtype=np.float64)
    _, centre, centre_lo = weighted_mean_pair(design, weights, float(np.sum(weights)))
    factor, _, _, inflated = grouped_projected_factor(
        design, weights, columns, center=centre, center_lo=centre_lo
    )
    factor = np.asarray(factor)
    if factor.shape[0] < width:
        factor = np.vstack((factor, np.zeros((width - factor.shape[0], width))))
    factor = factor / scale
    _, singular, right_t = np.linalg.svd(factor)
    magnitude = np.maximum(
        np.sqrt(np.maximum(data_diagonal, 0.0)), np.asarray(inflated, dtype=np.float64)
    )
    formation = gamma_up(columns.shape[0] + len(groups) + 3) * (magnitude @ np.abs(columns)) / scale
    bound = float(np.linalg.norm(formation)) + gamma_up(
        int(design.n) * width + width * width
    ) * float(np.linalg.norm(factor))
    return np.where(singular > bound, singular, 0.0), right_t.T


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


def _construction_reason(
    coordinate: int,
    term: int,
    involved: list[int],
    decomposition,
    groups: Sequence,
    design,
    specs: Mapping[Any, Any],
    W: NDArray | None,
    share: NDArray,
) -> str | None:
    """The construction that proves this alias by counting, or ``None``."""
    if design is None or W is None:
        return (
            "it has no data and no penalty" if decomposition.column_scale[coordinate] == 0 else None
        )
    from superglm._group_matrix._group_matrix_core import (
        CategoricalGroupMatrix,
        RandomEffectGroupMatrix,
    )

    matrices = design.group_matrices
    positive = np.asarray(W, dtype=np.float64) > 0.0

    def is_factor(index: int) -> bool:
        matrix = matrices[index]
        return isinstance(matrix, CategoricalGroupMatrix) and not isinstance(
            matrix, RandomEffectGroupMatrix
        )

    if len(involved) >= 2 and all(is_factor(i) for i in involved):
        return _factor_reason(term, involved, groups, matrices, positive)
    factors = [i for i in involved if is_factor(i)]
    if len(involved) == 2 and len(factors) == 1:
        by_level = next(i for i in involved if i != factors[0])
        return _level_reason(by_level, factors[0], groups, matrices, specs, positive, share)
    if involved == [term]:
        empty = _empty_basis_functions(matrices[term], positive)
        if empty:
            plural = "function has" if empty == 1 else "functions have"
            return f"{empty} of its basis {plural} no data, and it has no penalty there"
    if float(decomposition.column_scale[coordinate]) == 0.0:
        # no penalty, and a centred column of zero: constant on the fit's rows
        unit = np.zeros(groups[term].size)
        unit[coordinate - groups[term].start] = 1.0
        column = np.asarray(matrices[term].matvec(unit))[positive]
        if np.any(column != 0.0):
            return (
                "it is constant on the fit's rows, which the intercept carries, and has no penalty"
            )
        return "it has no data and no penalty"
    return None


def _factor_reason(term, involved, groups, matrices, positive) -> str | None:
    """Nested or disconnected factor levels, by the cross-classification's connectivity."""
    codes = [np.asarray(matrices[i].codes, dtype=np.intp)[positive] for i in involved]
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


def _level_reason(term, factor, groups, matrices, specs, positive, share) -> str | None:
    """A level of a by-level term with too few distinct covariate rows for its alias, counted."""
    largest = float(np.max(share, initial=0.0))
    levels = np.flatnonzero(share[groups[factor].sl] > _SHARE_FLOOR * largest)
    if levels.size != 1:
        return None
    level = int(levels[0])
    rows = np.flatnonzero(positive & (np.asarray(matrices[factor].codes) == level))
    own = groups[term].sl
    spanned = int(np.count_nonzero(share[own] > _SHARE_FLOOR * largest))
    distinct = (
        0
        if rows.size == 0
        else np.unique(matrices[term].row_subset(rows).toarray(), axis=0).shape[0]
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
    "Identifiability",
    "TruncatedDecomposition",
    "certified_identifiability",
    "coordinates_identified",
    "decide_identifiability",
    "fixed_coordinates",
    "functional_identified",
    "penalty_null_space",
]
