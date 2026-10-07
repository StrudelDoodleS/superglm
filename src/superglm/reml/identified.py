"""The identified part of the Laplace approximation (one-engine design §3.9).

**The class.**  A slope is weakly identified by its specification when no
penalty touches it and its column, centred on its shifted prior-weighted
mean (design §3.2), carries curvature only through rows whose prior weight
is within the rounding of an accumulation at the largest prior weight:

    sum_r w_r x~_rj^2 <= gamma_{n+} max_r(w_r) sum_{w_r > 0} x~_rj^2,

``gamma_k = k u / (1 - k u)`` (Higham 2002, Lemma 3.1), ``n+`` the rows with
positive prior weight: the design's §3.9 test with the working weights taken
at unit variance factor, so that it is a function of the design and the prior
weights alone -- rows weighted 1e-15 beside rows weighted 1.  A penalty
identifies a slope however small its data curvature, so only slopes outside
every penalty component are candidates (a penalty at the rounding of the data
is the border's step-5 truncation instead, design §3.6).

**Why the Laplace approximation leaves it out.**  Along such a slope the
likelihood is flat to within float64: PIRLS reaches no reproducible value of
it (a deviance-stopped iteration never converges it, a certificate-stopped
one excludes it by design §3.8), yet its log-curvature enters ``log|H|`` at
full weight, because ``log`` is scale-free.  ``log|H|`` then moves with
wherever PIRLS stopped, and REML's smoothing parameters for every other term
move with it: path-dependent, and silently wrong (the stage-0 verifier's
``offset_rare`` Poisson, lambda_u 19.0 against 10.47).  The approximation is
therefore taken over the identified slopes ``I`` with the weakly identified
slopes ``W`` held fixed at the fitted values (decision 7: flag, keep, and
certify the identified part).  ``log|H_II|``, the inverse ``[H_II^+, 0; 0,
0]`` (exactly zero ``W`` rows and columns) and the coefficient rank are read
from one factorization of ``H_II`` itself (``IdentifiedLaplace``), so they
describe the same matrix even where ``H`` or ``H_II`` is singular.  The REML
objective, its gradient, the ``W(rho)`` correction and the outer Hessian all
read these, so they are one function of the smoothing parameters whatever
value of ``beta_W`` PIRLS leaves: the identified block depends on ``beta_W``
only through rows weighted at the rounding.  The fit keeps ``beta_W``, and
its standard error comes from the full ``H``, never from this part.

The set is decided once per fit, before any iterate, from the design and the
prior weights: no classification changes between REML candidates, and none
selects a route (design §6).  With ``W`` empty every method returns its
argument, so every other fit is unchanged bit for bit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    RandomEffectGroupMatrix,
)

_UNIT_ROUNDOFF = 2.0**-53
_CHUNK = 8192


class WeakIdentificationWarning(UserWarning):
    """Some coefficients carry information only at the noise level of the data.

    Fires once per ``fit_reml`` and names the coefficients.  Each belongs to a
    factor level or column with little or no weight, few observations or
    little information: the data pin it down no better than the rounding of
    the computation does.  The fit keeps these coefficients in the model and
    says so plainly: those whose information is at noise level are left out
    of smoothing-parameter (REML) selection, and every named coefficient's
    estimate and standard error carry little information.  They are listed
    in ``model.diagnostics()`` (per group under ``"weakly_identified"`` and,
    with a note, under ``"_model"``).  To remove the warning, drop the column
    or merge the level, or give its rows more weight or data.
    """


WEAK_IDENTIFICATION_NOTE = (
    "These coefficients carry information only at the noise level of the data (a factor "
    "level or column with little or no weight, observations or information). They stay in "
    "the model; those whose information is at noise level are left out of "
    "smoothing-parameter selection, and their estimates and standard errors carry little "
    "information."
)


def _gamma(count: int) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)`` for ``k`` roundings, ``u = 2^-53``."""
    product = count * _UNIT_ROUNDOFF
    return product / (1.0 - product) if product < 1.0 else float("inf")


def penalized_columns(width: int, penalties: Sequence | None) -> NDArray:
    """``(width,)`` bool: the slopes some penalty component's block covers.

    A component whose ``lambda_policy`` fixes its smoothing parameter at zero
    (``LambdaPolicy.off()`` or ``fixed(0.0)``) adds nothing to ``S`` for the
    whole fit, so it identifies nothing: its slopes stay candidates.  The
    policy is part of the specification, so the classification still reads
    no data value.
    """
    mask = np.zeros(width, dtype=bool)
    for component in penalties or ():
        policy = getattr(component, "lambda_policy", None)
        if policy is not None and policy.mode == "fixed" and float(policy.value) == 0.0:
            continue
        mask[component.group_sl] = True
    return mask


def random_effect_columns(dm: DesignMatrix) -> NDArray:
    """Slope indices of every ``RandomEffectGroupMatrix`` block, ascending.

    In a structured factor's border such a block is complete, so its exposed
    levels sum to the intercept: a structural generator of the border's data
    part (``_structured.moments._border_generators``), which the factor's
    rebuild cannot leave out (``IdentifiedLaplace``).
    """
    columns, offset = [], 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        if isinstance(matrix, RandomEffectGroupMatrix):
            columns.append(np.arange(offset, offset + width))
        offset += width
    return np.concatenate(columns).astype(np.intp) if columns else np.zeros(0, dtype=np.intp)


def laplace_excluded_coefficients(
    dm: DesignMatrix,
    sample_weight: NDArray,
    penalties: Sequence | None,
) -> NDArray:
    """The slopes the Laplace approximation leaves out (module docstring), ascending.

    One-hot columns are evaluated in closed form from two transpose products
    (entries 0 and 1 about a centre in [0, 1] cannot cancel); a
    ``DenseGroupMatrix`` column is centred row by row in fixed chunks about its
    shifted prior-weighted mean, so its rounding scales with ``|x - c|`` and
    not with a column offset; any other unpenalized column is formed once by a
    design product.  A column with no centred mass on the weighted rows is an
    exact alias of the intercept, a rank decision, and is never in the set.
    """
    weights = np.asarray(sample_weight, dtype=np.float64)
    positive = weights > 0.0
    row_count = int(np.count_nonzero(positive))
    # Rows of weight 0, masked to an exact 0 (#369); None in the usual fit.
    zero_rows = None if row_count == weights.size else ~positive
    candidates = ~penalized_columns(dm.p, penalties)
    if row_count == 0 or not np.any(candidates):
        return np.zeros(0, dtype=np.intp)
    total = float(np.sum(weights))
    bar = _gamma(row_count) * float(np.max(weights))
    first = int(np.argmax(positive))
    curvature = np.zeros(dm.p)
    mass = np.zeros(dm.p)
    evaluated = np.zeros(dm.p, dtype=bool)
    on_weight = on_rows = None
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        columns = np.arange(offset, offset + width)
        offset += width
        take = candidates[columns]
        if not np.any(take):
            continue
        evaluated[columns] = True
        if isinstance(matrix, CategoricalGroupMatrix):
            if on_weight is None:
                on_weight = dm.rmatvec(weights)
                on_rows = dm.rmatvec(positive.astype(np.float64))
            assert on_rows is not None
            level_weight, level_rows = on_weight[columns], on_rows[columns]
            share = level_weight / total
            curvature[columns] = (1.0 - share) ** 2 * level_weight + share**2 * (
                total - level_weight
            )
            mass[columns] = (1.0 - share) ** 2 * level_rows + share**2 * (row_count - level_rows)
        elif isinstance(matrix, DenseGroupMatrix):
            values = matrix.M
            reference = np.array(values[first], dtype=np.float64)
            shift = np.zeros(width)
            for lo in range(0, dm.n, _CHUNK):
                hi = min(lo + _CHUNK, dm.n)
                shift += weights[lo:hi] @ (values[lo:hi] - reference)
            centre = reference + shift / total
            for lo in range(0, dm.n, _CHUNK):
                hi = min(lo + _CHUNK, dm.n)
                centred = values[lo:hi] - centre
                if zero_rows is not None:
                    centred[zero_rows[lo:hi]] = 0.0  # a zero-weight row adds exactly 0 (#369)
                squares = centred**2
                curvature[columns] += weights[lo:hi] @ squares
                mass[columns] += positive[lo:hi].astype(np.float64) @ squares
        else:
            for column in columns[take]:
                unit = np.zeros(dm.p)
                unit[column] = 1.0
                values = dm.matvec(unit)
                reference = float(values[first])
                centre = reference + float(weights @ (values - reference)) / total
                centred = values - centre
                if zero_rows is not None:
                    centred[zero_rows] = 0.0
                squares = centred**2
                curvature[column] = float(weights @ squares)
                mass[column] = float(np.sum(squares[positive]))
    weak = evaluated & candidates & (mass > 0.0) & (curvature <= bar * mass)
    return np.flatnonzero(weak).astype(np.intp)


def binomial_log(distribution: Any, link: Any) -> bool:
    """Whether the family is binomial with the log link: the true-score mean space (#437)."""
    from superglm.distributions import Binomial
    from superglm.links import LogLink

    return isinstance(distribution, Binomial) and isinstance(link, LogLink)


def separated_directions(
    dm: DesignMatrix,
    y: NDArray,
    sample_weight: NDArray,
    penalized: NDArray,
    generator_columns: NDArray,
    weak: NDArray | Sequence[int] = (),
) -> tuple[NDArray, tuple]:
    """``(pivots, sets)``: a binomial/log fit's separated unpenalized directions.

    **The class.**  A set of rows the one-hot blocks move on their own
    (``mode_score.row_sets``: a level, a block's reference rows, a kept joint
    set; a set held without its direction, past the bridge budget, is judged
    but not left out here) whose positive-weight responses are all 0, or all 1, along a
    direction no penalty touches has no interior maximum.  The likelihood's
    supremum along it is at ``eta -> -infinity`` (or at the mean space's
    boundary), where the rows' log-likelihood and their observed curvature
    are both 0.  A fit stops somewhere along that drift.  Where it stops moves
    ``log|H|`` by the log of the rows' vanishing curvature, so the criterion
    and its smoothing parameters move with the stopping point: with a
    constant offset, with the start, with the solver's path.  The Laplace
    approximation is therefore taken at the supremum along these directions:
    they are left out of it, as the weakly identified slopes are.  (Their
    rows' log-likelihood, about the sum of their means, has underflowed to
    0 where the fits measured for #437 stop, so it is left as it is.)  The
    set is decided once per fit from the design,
    the responses, the prior weights and which columns a penalty covers, so
    no classification changes between REML candidates.  A penalized set (a
    random-effect level inside a level without events) is not in the class:
    its penalized maximum is finite.

    **Pivots.**  ``IdentifiedLaplace`` leaves out coordinates.  Each
    direction is reduced by the directions already taken (Gaussian
    elimination) and stands for one coordinate where it is largest, among the
    unpenalized columns outside a random-effect block (``generator_columns``,
    which a structured rebuild cannot leave out).  A level's direction is its
    own column.  A block's reference direction is the intercept less the
    block's columns, ``-1_B`` once the intercept is profiled, and stands for
    one of the block's other columns.  The change of basis that makes it a
    coordinate is unimodular, and at the supremum ``H`` is singular along
    it, so the determinant left over differs from the pseudo-determinant of
    the rest by a constant that no smoothing parameter moves.  A direction
    with no such column, or spanned by those already taken, adds none.  The
    ``weak`` slopes, which the Laplace term leaves out already, come first in
    the elimination: no pivot is a weak column, and a set whose direction a
    weak slope already spans (a separated level that is also weak) takes none
    and is disclosed once, as weak.  ``pivots`` are in the order the sets are
    found, and ``sets`` match them one to one, each a tuple of ``(block's
    first column, level code or None for its reference)`` over the blocks it
    names (``separated_set_labels``).
    """
    from superglm.solvers.mode_score import row_sets

    width = dm.p
    response = np.asarray(y, dtype=np.float64)
    positive = np.asarray(sample_weight, dtype=np.float64) > 0.0
    covered = np.asarray(penalized, dtype=bool)
    allowed = ~covered
    allowed[np.asarray(generator_columns, dtype=np.intp)] = False
    seeded = np.zeros(width, dtype=bool)
    seeded[np.asarray(weak, dtype=np.intp)] = True
    allowed &= ~seeded
    rising = (positive & (response > 0.0)).astype(np.float64)
    falling = (positive & (response < 1.0)).astype(np.float64)
    carried = positive.astype(np.float64)
    candidates: list[tuple[NDArray, tuple]] = []
    sets = row_sets(dm)
    for start, matrix in sets.blocks:
        levels = matrix.n_levels
        codes = matrix.codes
        count, up, down = (
            np.bincount(codes, weights=values, minlength=levels + 1)
            for values in (carried, rising, falling)
        )
        separated = (count > 0.0) & ((up == 0.0) | (down == 0.0))
        for level in np.flatnonzero(separated).tolist():
            if level < levels:
                if covered[start + level]:
                    continue
                direction = np.zeros(width)
                direction[start + level] = 1.0
            else:
                if np.any(covered[start : start + levels]):
                    continue
                direction = np.zeros(width)
                direction[start : start + levels] = -1.0
            candidates.append((direction, ((start, level if level < levels else None),)))
    if sets.row_cell is not None:
        held = sets.cell_directions
        membership = sets.cell_sets.T.tocsr()
        cells = sets.cell_sets.shape[0]
        count, up, down = (
            np.asarray(
                membership @ np.bincount(sets.row_cell, weights=values, minlength=cells)
            ).ravel()
            for values in (carried, rising, falling)
        )
        separated = (count > 0.0) & ((up == 0.0) | (down == 0.0)) & ~sets.bounded
        for index in np.flatnonzero(separated).tolist():
            direction = held[[index]].toarray().ravel()
            if np.any(covered & (direction != 0.0)):
                continue
            described = tuple(
                (start, code if code < matrix.n_levels else None)
                for (start, matrix), code in zip(
                    sets.blocks, sets.cell_codes[index].tolist(), strict=True
                )
                if code >= 0
            )
            candidates.append((direction, described))
    pivots: list[int] = []
    taken: list[NDArray] = []
    named: list[tuple] = []
    for direction, described in candidates:
        reduced = direction.copy()
        reduced[seeded] = 0.0  # the weak slopes' unit directions, taken first
        for pivot, basis in zip(pivots, taken, strict=True):
            if reduced[pivot] != 0.0:
                reduced = reduced - (reduced[pivot] / basis[pivot]) * basis
        size = float(np.max(np.abs(direction)))
        magnitude = np.where(allowed, np.abs(reduced), 0.0)
        pivot = int(np.argmax(magnitude))
        if magnitude[pivot] <= width * _UNIT_ROUNDOFF * size:
            continue
        pivots.append(pivot)
        taken.append(reduced)
        named.append(described)
    return np.asarray(pivots, dtype=np.intp), tuple(named)


def separated_set_labels(groups: Sequence, sets: Sequence[tuple]) -> tuple[str, ...]:
    """``group[level]`` or ``group[reference]`` per block a separated set names, joined by `` x ``."""
    labels = []
    for described in sets:
        parts = []
        for start, level in described:
            group = next((g for g in groups if g.start == int(start)), None)
            name = f"coef[{int(start)}]" if group is None else group.name
            parts.append(f"{name}[{'reference' if level is None else int(level)}]")
        labels.append(" x ".join(parts))
    return tuple(labels)


def dense_hessian(cache: dict | None) -> tuple[NDArray, float] | None:
    """``(H_c, sum w)`` a dense PIRLS left in its ``cache_out``; ``None`` without one.

    ``H_c`` is the centred slope Hessian the fit's slope decomposition was
    taken of: the dense identified part restricts exactly that matrix.
    """
    if not cache or "centered_hessian" not in cache:
        return None
    return cache["centered_hessian"], float(cache["sum_W"])


@dataclasses.dataclass(frozen=True)
class _IdentifiedPart:
    """One input's identified part: its inverse (``None`` when not asked), ``log|H_II|`` and rank."""

    inverse: Any
    log_det: float
    rank: int


class IdentifiedLaplace:
    """The identified part of one fit's Laplace approximation (module docstring).

    ``excluded`` holds the slope indices ``W`` the Laplace term leaves out:
    the weakly identified slopes (``weak``, which the fit also leaves ungated
    and discloses) and, for binomial/log, the pivots standing for its
    separated unpenalized directions (``separated_directions``).  The identified part is read
    from ONE factorization of ``H_II`` itself, so its inverse, ``log|H_II|``
    and coefficient rank always describe the same matrix: a structured factor
    is rebuilt by its own engine with the ``W`` border columns left out (its
    ``logdet`` and ``rank``), and a dense system's centred Hessian ``H_c`` is
    restricted to ``I`` and decomposed by the shared rank rule
    (``solvers.rank.decompose_gram``), ``log|H_II| = log(sum w) + log
    pdet(H_c[I, I])`` as for ``H``.  Jacobi's complementary-minor identity
    ``det H_II = det H det((H^-1)_WW)`` is a statement about an invertible
    ``H``; once the full factor truncates a direction, the ``W`` block of its
    generalized inverse no longer determines ``H_II`` (the Jacobi-type
    identities for generalized inverses carry nullity terms instead), so it
    is not used.  A dense input therefore needs its centred Hessian
    (``dense=(H_c, sum_w)``), which every dense caller holds.

    Every method returns its argument unchanged when ``W`` is empty.
    ``unsupported`` counts the inputs this cannot restrict -- an excluded
    slope outside a structured factor's border, or on a random-effect block,
    whose complete one-hot sum is a structural generator of the border that
    the rebuild cannot leave out -- for which every method keeps the full
    ``H``, inverse, determinant and rank alike; the count is published.

    Cache: the last input's part is memoised (owner: this object, one fit;
    lifetime: until an input of another identity arrives; invalidation: the
    identity of the input factor or Hessian, which are immutable once built),
    so the inverse, determinant and rank reads of one evaluation share one
    rebuild.  A structured refusal while restricting raises
    ``StructuredSolverError``, as every structured factor operation does.
    """

    def __init__(
        self,
        excluded: NDArray | Sequence[int] = (),
        *,
        generator_columns: NDArray | Sequence[int] = (),
        weak: NDArray | Sequence[int] | None = None,
        separated_pivots: NDArray | Sequence[int] = (),
        separated_sets: tuple = (),
    ):
        self.excluded = np.asarray(excluded, dtype=np.intp)
        self.weak = self.excluded if weak is None else np.asarray(weak, dtype=np.intp)
        self.separated_pivots = np.asarray(separated_pivots, dtype=np.intp)
        self.separated_sets = tuple(separated_sets)
        self.generator_columns = np.asarray(generator_columns, dtype=np.intp)
        self.unsupported = 0
        self._memo: tuple | None = None

    @classmethod
    def for_design(
        cls,
        dm: DesignMatrix,
        sample_weight: NDArray,
        penalties: Sequence | None,
        *,
        y: NDArray | None = None,
        distribution: Any = None,
        link: Any = None,
    ) -> IdentifiedLaplace:
        """The fit's identified part: ``laplace_excluded_coefficients``, the separated directions' pivots and the generator columns."""
        weak = laplace_excluded_coefficients(dm, sample_weight, penalties)
        pivots: NDArray = np.zeros(0, dtype=np.intp)
        separated: tuple = ()
        if y is not None and binomial_log(distribution, link):
            pivots, separated = separated_directions(
                dm,
                y,
                sample_weight,
                penalized_columns(dm.p, penalties),
                random_effect_columns(dm),
                weak,
            )
        if not pivots.size:
            if not weak.size:  # nothing to restrict: no generator test either
                return cls(weak)
            return cls(weak, generator_columns=random_effect_columns(dm))
        return cls(
            np.union1d(weak, pivots).astype(np.intp),
            generator_columns=random_effect_columns(dm),
            weak=weak,
            separated_pivots=pivots,
            separated_sets=separated,
        )

    def __bool__(self) -> bool:
        return bool(self.excluded.size)

    @property
    def disclosed(self) -> tuple[int, ...]:
        """The left-out coordinates in disclosure order: the weak slopes, then each separated set's pivot.

        Pairs one to one with ``coefficient_labels`` of the weak slopes
        followed by ``separated_set_labels`` of ``separated_sets``.
        """
        return tuple(int(index) for index in self.weak) + tuple(
            int(index) for index in self.separated_pivots
        )

    def _part(self, inverse, dense) -> _IdentifiedPart | None:
        hessian = None if dense is None else dense[0]
        memo = self._memo
        if memo is not None and memo[0] is inverse and memo[1] is hessian:
            return memo[2]
        part = self._restrict(inverse, dense)
        self._memo = (inverse, hessian, part)
        return part

    def _restrict(self, inverse, dense) -> _IdentifiedPart | None:
        from superglm.solvers._structured.balance_tree import ProfiledSumToZeroTreeFactor
        from superglm.solvers._structured.block_leaves import ProfiledFactorSmoothLeafFactor
        from superglm.solvers._structured.nested import ProfiledNestedSchurFactor
        from superglm.solvers.hessian_factor import DenseHessianFactor

        if isinstance(
            inverse,
            ProfiledNestedSchurFactor
            | ProfiledFactorSmoothLeafFactor
            | ProfiledSumToZeroTreeFactor,
        ):
            return self._structured_part(inverse)
        if inverse is None or isinstance(inverse, np.ndarray | DenseHessianFactor):
            if dense is None:
                raise RuntimeError(
                    "The identified part of a dense system needs its centred Hessian."
                )
            part = self._dense_part(*dense, with_inverse=inverse is not None)
            if isinstance(inverse, DenseHessianFactor):
                part = dataclasses.replace(
                    part, inverse=DenseHessianFactor(inverse=part.inverse, log_det=part.log_det)
                )
            return part
        self.unsupported += 1
        return None

    def _dense_part(self, hessian, sum_w: float, *, with_inverse: bool) -> _IdentifiedPart:
        """``H_c[I, I]`` decomposed by the shared rule; ``W`` rows and columns of the inverse zero."""
        from superglm.solvers.rank import decompose_gram

        matrix = np.asarray(hessian, dtype=np.float64)
        kept = np.ones(matrix.shape[0], dtype=bool)
        kept[self.excluded] = False
        block = matrix[np.ix_(kept, kept)]
        decomposition = decompose_gram(0.5 * (block + block.T))
        inverse = None
        if with_inverse:
            inverse = np.zeros_like(matrix)
            inverse[np.ix_(kept, kept)] = decomposition.pseudo_inverse()
        return _IdentifiedPart(
            inverse=inverse,
            log_det=float(np.log(sum_w) + decomposition.log_pdet),
            rank=1 + int(decomposition.rank),
        )

    def _structured_part(self, inverse) -> _IdentifiedPart | None:
        """The structured factor rebuilt by its own engine with ``W`` left out of its border."""
        from superglm.solvers._structured.balance_tree import (
            ProfiledSumToZeroTreeFactor,
            SumToZeroTreeFactor,
        )
        from superglm.solvers._structured.block_leaves import (
            FactorSmoothLeafFactor,
            ProfiledFactorSmoothLeafFactor,
        )
        from superglm.solvers._structured.nested import (
            NestedSchurFactor,
            ProfiledNestedSchurFactor,
        )
        from superglm.solvers.irls_direct import _structured_solver_errors

        augmented = inverse.augmented_factor
        positions = self.excluded + 1
        if not np.all(np.isin(positions, augmented.small_indices)) or np.any(
            np.isin(self.excluded, self.generator_columns)
        ):
            self.unsupported += 1
            return None
        excluded = tuple(int(index) for index in positions)
        with _structured_solver_errors():
            if isinstance(inverse, ProfiledNestedSchurFactor):
                rebuilt = NestedSchurFactor(
                    augmented.operator,
                    chain_group_names=augmented.chain_group_names,
                    chain_group_indices=augmented.chain_group_indices,
                    intercept=True,
                    max_structured_inverse_block=augmented.max_structured_inverse_block,
                    excluded=excluded,
                )
                profiled = ProfiledNestedSchurFactor(
                    augmented_factor=rebuilt,
                    sum_w=inverse.sum_w,
                    xtw=inverse.xtw,
                    data_operator=inverse.data_operator,
                )
            elif isinstance(inverse, ProfiledFactorSmoothLeafFactor):
                rebuilt = FactorSmoothLeafFactor(
                    augmented.system,
                    augmented.penalized,
                    max_structured_inverse_block=augmented.max_structured_inverse_block,
                    excluded=excluded,
                )
                profiled = ProfiledFactorSmoothLeafFactor(
                    augmented_factor=rebuilt, sum_w=inverse.sum_w, xtw=inverse.xtw
                )
            else:
                assert isinstance(inverse, ProfiledSumToZeroTreeFactor)
                rebuilt = SumToZeroTreeFactor(
                    augmented.system,
                    augmented.penalized,
                    max_structured_inverse_block=augmented.max_structured_inverse_block,
                    excluded=excluded,
                )
                profiled = ProfiledSumToZeroTreeFactor(
                    augmented_factor=rebuilt, sum_w=inverse.sum_w, xtw=inverse.xtw
                )
            log_det = float(rebuilt.logdet())
        return _IdentifiedPart(inverse=profiled, log_det=log_det, rank=int(rebuilt.rank))

    def inverse(self, inverse, dense: tuple | None = None):
        """The identified inverse ``[H_II^+, 0; 0, 0]`` in the same representation."""
        if not self or inverse is None:
            return inverse
        part = self._part(inverse, dense)
        return inverse if part is None else part.inverse

    def log_det(self, inverse, log_det: float | None, dense: tuple | None = None) -> float | None:
        """``log|H_II|`` (a pseudo-determinant where ``H_II`` is singular) of the same factorization."""
        if not self or log_det is None or (inverse is None and dense is None):
            return log_det
        part = self._part(inverse, dense)
        return log_det if part is None else part.log_det

    def rank(self, rank: int | None, inverse, dense: tuple | None = None) -> int | None:
        """The coefficient rank of the identified part, from the same factorization."""
        if not self or rank is None or (inverse is None and dense is None):
            return rank
        part = self._part(inverse, dense)
        return rank if part is None else part.rank

    def geometry(self, geometry):
        """An observed REML geometry with its inverse, determinant and rank restricted."""
        if not self or geometry is None:
            return geometry
        inverse = geometry.hessian_inverse
        hessian = geometry.centered_hessian
        dense = (hessian, float(geometry.sum_w)) if isinstance(hessian, np.ndarray) else None
        if inverse is None and dense is None:
            return geometry
        part = self._part(inverse, dense)
        if part is None:
            return geometry
        return dataclasses.replace(
            geometry,
            hessian_inverse=part.inverse if inverse is not None else None,
            log_det_H=part.log_det,
            hessian_rank=part.rank,
        )


def penalty_diagonal(width: int, lambdas: dict[str, float], penalties: Sequence | None) -> NDArray:
    """``diag(S)`` ``(width,)`` from the penalty components, never forming ``S``.

    Identity blocks add ``lambda``; dense blocks ``lambda diag(Omega)``;
    repeated blocks tile it; a sum-to-zero block is ``(C'C) kron Omega`` with
    ``C = [I; -1]``, whose diagonal is ``2 diag(Omega)`` per level.
    """
    diagonal = np.zeros(width)
    for component in penalties or ():
        lam = float(lambdas.get(component.name, 0.0))
        if lam == 0.0:
            continue
        block = component.group_sl
        size = len(range(*block.indices(width)))
        if component.penalty_kind == "identity":
            diagonal[block] += lam
            continue
        local = np.diag(np.asarray(component.omega_ssp, dtype=np.float64))
        if component.penalty_kind == "repeated":
            local = np.tile(local, int(component.repeat_count))
        elif component.penalty_kind == "sum_to_zero":
            local = 2.0 * np.tile(local, int(component.repeat_count) - 1)
        if local.shape != (size,):
            raise ValueError(f"Penalty component {component.name!r} does not match its block.")
        diagonal[block] += lam * local
    return diagonal


def final_mode_weak_slopes(
    *,
    dm: DesignMatrix,
    distribution,
    link,
    sample_weight: NDArray,
    offset_arr: NDArray,
    result,
    lambdas: dict[str, float],
    penalties: Sequence | None,
) -> NDArray:
    """The §3.9 weak test at a fit's final mode, from its Fisher rows (``mode_score.weakly_identified_mask``)."""
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.solvers.mode_score import linear_predictor, weakly_identified_mask
    from superglm.solvers.working_rows import fisher_working_weights

    weights = np.asarray(sample_weight, dtype=np.float64)
    eta = stabilize_eta(linear_predictor(dm, result, offset_arr), link)
    mu = clip_mu(link.inverse(eta), distribution)
    fisher = fisher_working_weights(
        distribution=distribution, link=link, mu=mu, eta=eta, sample_weight=weights
    )
    fisher = np.where(np.isfinite(fisher), np.maximum(fisher, 0.0), 0.0)
    total = float(np.sum(fisher))
    if not total > 0.0:
        return np.zeros(0, dtype=np.intp)
    mask = weakly_identified_mask(
        dm=dm,
        fisher_weights=fisher,
        positive_prior=weights > 0.0,
        mean_x=dm.rmatvec(fisher) / total,
        penalty_diagonal=penalty_diagonal(dm.p, lambdas, penalties),
        unpenalized=~penalized_columns(dm.p, penalties),
    )
    return np.flatnonzero(mask).astype(np.intp)


def coefficient_labels(groups: Sequence, indices: Sequence[int]) -> tuple[str, ...]:
    """``group[column]`` names of slope indices (the column within its group)."""
    labels = []
    for index in indices:
        group = next((g for g in groups if g.start <= int(index) < g.end), None)
        labels.append(
            f"coef[{int(index)}]" if group is None else f"{group.name}[{int(index) - group.start}]"
        )
    return tuple(labels)
