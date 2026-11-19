"""Which coefficients a penalised fit identifies: the one decision every standard error reads.

**The question.** A fit reports the penalised estimate, so a coefficient is identified when the
penalised Hessian ``H = X'WX + sum_j lambda_j S_j`` determines it, not when the data alone would
(classical estimability, which assumes no penalty).  With every ``lambda_j > 0`` and ``W > 0``,
``H v = 0`` exactly when ``X v = 0`` and ``S_j v = 0`` for every ``j`` (a sum of semidefinite
terms vanishes only when each does), so

    null(H) = N_S null(X_F),   X_F = X_c N_S,

with ``N_S`` the penalty's null space and ``X_c`` the design centred on the intercept, on the
rows with positive prior weight: a direction is lost only when neither the data nor any penalty
reaches it, whatever the smoothing parameters and however a term is parametrised (Wood, Pya &
Saefken 2016, JASA 111, sec. 3.1.2; the mixed-model split of Ruppert, Wand & Carroll 2003, sec.
4.9).  A partially aliased coefficient is identified; its standard error is only larger.

**One decision, read from the rows.** The solver's decomposition of ``H`` is a speed hint: it
inverts ``H`` on a part it certified positive definite (a structured factor's verified pivots; a
dense decomposition whose smallest kept eigenvalue clears the Gram's formation bound, or one
recomputed so that it does, ``state_ops._dense_base``) and leaves the rest, ``T``, undecided.
In the basis ``[K, T]`` the nullity of ``H`` is that of the Schur complement on ``T``, which one
pass over the rows gives (``_read_rows``): ``T'HT = B'B + T'ST``, ``B = sqrt(W) X_c T``, less the
coupling to ``K`` the same pass reads.  An eigenvalue whose root is within the rounding bound of
``B``'s singular values cannot be told from zero in float64; one that clears it is identified
(Weyl); the decision never compares the solver's own rank.  A direction the weighted rows cannot
tell from zero is an alias only if the design cannot either, read once more at unit weight on the
rows with positive prior weight (``_design_null``): a vanished working weight carries no
information about identifiability, so such a direction is never moved along, and its
coordinates' standard errors are not resolved at float64.  Counting (a level's rows, its distinct
covariate values) only names the reason for an alias; it decides nothing.

**The margin grows with the rows.** In Jacobi-scaled coordinates (unit diagonal of ``H``) the
rounding bound on ``B``'s singular values is the formation term ``gamma_(p+T+3) || |A| |T| ||``,
at most about ``(p + T + 3) sqrt(p) u`` with ``u = 2^-53``, plus the triangle's ``gamma_(n k)
||B||_F`` for ``n`` rows and ``k`` directions read together (Higham 2002, Thm 19.4).  So a
near-duplicate column separated from another by a relative ``d`` is aliased when ``d`` is under
about ``p^(3/2) u`` read alone, and under about ``n k u`` when it is read beside a direction of
unit curvature: at ``n = 10^6`` rows that is ``1e-10``, and ``d = 1e-8`` stays identified.  A
direction whose root falls inside the coupling's uncertainty is reported as not resolved at
float64, never decided.

**Directions the solver truncated that the rows identify** are fitted and reported, as mgcv's
penalised fit has them and ``lm``'s rank decision does not (``WeakDirections``): their
covariance is ``T C^-1 T'`` on the refined directions, and their coefficients take the Newton
step their score gives, which for a Gaussian family with the identity link is the penalised fit;
under any other family the publication resumes penalised IRLS on the identified subspace
(``state_ops._resumed_weak_fit``).

**Whether each estimate is finite** is the fit's existence decision
(``model.separation_ops``, ``Identifiability.separation``): every coefficient a certified
separating direction moves has no finite estimate and no standard error.

**Which constraint is published.** Where ``H`` is singular, the fit publishes the coefficients
with one constraint per null direction (``Gauge``): each null direction is written on the terms'
construction columns of ``X_F`` (``ConstructionColumns``: an unpenalised term's coordinates, a
smooth's penalty null space in its raw basis), and the latest column each reaches in term order
is held at zero, so the first-defined term keeps its coefficients (R's and ``gam.side``'s
convention).  The choice reads no row, weight, smoothing parameter or solver coordinate, so a
row permutation, a BLAS kernel or a rotation of a spline's coordinates leaves it where it was.
It is kept while it amplifies the conditioning by at most ``1e7`` (``lm.fit``'s ``tol = 1e-7``),
and the conditioning-first selector replaces it past that.  A held unpenalised column is blank;
a held smooth part blanks no coordinate, and the term's coefficients have the standard errors of
the fit with that part at zero.  The covariance is ``P (V + U U') P'``, with ``V`` the inverse on
the certified part, never a near-singular inverse's.

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

# The term-order gauge is kept while it amplifies the retained coordinates' conditioning by at
# most ``1 / tol`` with ``tol = 1e-7``, the scale at which R's ``lm.fit`` moves a column it
# cannot separate from the earlier ones to the end (``?lm.fit``); past it the conditioning-first
# selector chooses (threshold pivoting: Arioli, Duff, Gould & Reid, SIAM J. Sci. Stat. Comput.
# 11(5), 1990).
_AMPLIFICATION_LIMIT = 1e7


@dataclass(frozen=True)
class Gauge:
    """The coefficients a singular fit publishes: ``G' beta = 0`` along ``null`` ``N``.

    ``null`` (``width x k``, fitted coordinates) spans ``null(H)``; with every lambda positive
    ``X_c N = 0`` and ``S N = 0``, so moving along it changes no fitted value and no penalty.
    ``functionals`` ``G`` (``width x k``) are the constraints the fit publishes, each a term's
    construction column (``ConstructionColumns``): a unit coordinate for an unpenalised column,
    a smooth's unpenalised part otherwise.  ``G'N`` is nonsingular, so ``P = I - N (G'N)^-1 G'``
    maps a coefficient vector to the one with ``G' beta = 0`` and the same fitted values, and a
    covariance ``V``, any generalised inverse of ``H``, to ``P V P'`` (each row of ``P`` is
    orthogonal to ``N``, and every generalised inverse has one bilinear form on ``range(H) =
    null(H)^perp``).  ``intercept`` holds each direction's solver-intercept component, ``-c'N``
    with ``c`` the fit's weighted column centre.  ``moved`` marks the directions of ``null(H)``;
    the others are directions the rows cannot resolve at float64 (``_design_null``), which do
    move fitted values: the coefficients are never moved along them, and the covariance is the
    fit's with them held where the fit left them.
    """

    null: NDArray
    functionals: NDArray
    intercept: NDArray
    moved: NDArray | None = None

    def _moved(self) -> NDArray:
        count = self.null.shape[1]
        return np.ones(count, dtype=bool) if self.moved is None else np.asarray(self.moved, bool)

    @property
    def coordinates(self) -> NDArray:
        """The coordinates held at exactly zero: those of the unit functionals."""
        nonzero = self.functionals != 0.0
        unit = np.count_nonzero(nonzero, axis=0) == 1
        return np.sort(np.argmax(nonzero[:, unit], axis=0)).astype(np.intp)

    def lift(self, columns: NDArray | None = None) -> NDArray:
        """``N (G'N)^-1`` on the fitted ``columns`` (every coordinate by default)."""
        null = self.null if columns is None else self.null[np.asarray(columns, dtype=np.intp)]
        return np.linalg.solve((self.functionals.T @ self.null).T, null.T).T

    def coefficients(self, beta: NDArray, intercept: float) -> tuple[NDArray, float, NDArray]:
        """``beta`` moved along ``null(H)`` onto its constraints, the solver intercept that keeps
        every fitted value, and the step taken (along the moved directions)."""
        beta = np.asarray(beta, dtype=np.float64)
        moved = self._moved()
        if not np.any(moved):
            return beta.copy(), float(intercept), np.zeros(0)
        null, functionals = self.null[:, moved], self.functionals[:, moved]
        step = np.linalg.solve(functionals.T @ null, functionals.T @ beta)
        gauged = beta - null @ step
        nonzero = functionals != 0.0
        unit = np.count_nonzero(nonzero, axis=0) == 1
        gauged[np.argmax(nonzero[:, unit], axis=0)] = 0.0
        return gauged, float(intercept) - float(self.intercept[moved] @ step), step

    def covariance(self, covariance: NDArray, columns: NDArray) -> NDArray:
        """``P V P'`` of a dense covariance ``V`` held on the fitted ``columns``."""
        columns = np.asarray(columns, dtype=np.intp)
        V = np.asarray(covariance, dtype=np.float64)
        lift = self.lift(columns)
        VG = V @ self.functionals[columns]
        GVG = self.functionals[columns].T @ VG
        gauged = V - lift @ VG.T - VG @ lift.T + lift @ GVG @ lift.T
        gauged = 0.5 * (gauged + gauged.T)
        fixed = np.searchsorted(columns, self.coordinates)
        gauged[fixed, :] = 0.0
        gauged[:, fixed] = 0.0
        return gauged


@dataclass(frozen=True)
class WeakDirections:
    """Directions the solver truncated that the penalised fit identifies, fitted from the rows.

    In fitted coordinates: ``root`` ``U`` (``width x d``) with ``U U'`` their covariance,
    ``T C^-1 T'`` for the refined directions ``T`` and their curvature ``C`` read from the rows
    (``decide_identifiability``); ``step`` the coefficients' move ``T C^-1 g`` by the score
    ``g`` along them, the one Newton step that fits them at the fit's working weights;
    ``intercept`` and ``intercept_step`` the solver intercept's components of each, ``-c'U`` and
    ``-c' step``.  ``accuracy`` bounds the relative error of their curvature, and ``data`` is
    ``D U``, the centred data Gram's product read from the rows.  ``fitted`` is false where the
    fit keeps its solver's coefficients along them (``step`` zero; ``state_ops``).
    """

    root: NDArray
    intercept: NDArray
    step: NDArray
    intercept_step: float
    accuracy: float
    data: NDArray
    fitted: bool = True

    def covariance(self, covariance: NDArray, columns: NDArray) -> NDArray:
        """``V + U U'`` of a dense covariance ``V`` held on the fitted ``columns``."""
        U = self.root[np.asarray(columns, dtype=np.intp)]
        return np.asarray(covariance, dtype=np.float64) + U @ U.T


@dataclass(frozen=True)
class Alias:
    """One direction the fit does not identify, and what it is aliased with.

    ``coordinate`` is the fitted coordinate held at zero, without a standard error; ``None``
    when the published constraint is a smooth's unpenalised part, whose coefficients keep their
    standard errors in the published gauge.
    """

    coordinate: int | None
    term: str
    aliased_with: tuple[str, ...]
    reason: str
    source: Literal["construction", "rank", "float64", "fit"]


@dataclass(frozen=True)
class Identifiability:
    """The identifiability record of one fit, in fitted (solver) coordinates.

    ``aliases`` are the directions the fit does not identify; every coefficient off their
    ``coordinate`` is identified.  ``penalised_condition`` is the condition number whose product
    with ``u`` bounds the standard errors' relative accuracy, and ``fixed_effect_condition`` that
    of the directions read from the rows (NaN when none were).  ``gauge`` moves a singular fit's
    coefficients and covariance onto its published constraints, ``weak`` fits the directions
    its solver truncated that the penalised fit identifies, and ``covariance`` (selected
    coordinates) replaces a dense decomposition's own inverse where its rank decision kept a
    direction the rows put in ``null(H)``.  ``edf`` and ``edf1`` (fitted coordinates) are the
    published fit's, where it differs from its solver's.  ``row_directions`` counts the
    directions read from the rows, and ``rank`` the slope directions the published fit
    identifies (``None`` for a record that describes its solver's coefficients).
    ``curvature_floor`` is a certified lower bound on the Jacobi-scaled Hessian's eigenvalues on
    the directions the record identifies (zero where none is certified: a direction not resolved
    at float64), which the existence certificate reads (``separation.recession_excluded``).
    ``separation`` is the fit's existence decision (``diagnostics.separation.Separation``): the
    coefficients a certified separating direction moves have no finite estimate and no standard
    error, and an unresolved one withholds the standard errors its proposal moves.
    """

    width: int
    aliases: tuple[Alias, ...]
    penalised_condition: float
    fixed_effect_condition: float
    gauge: Gauge | None = None
    row_directions: int = 0
    weak: WeakDirections | None = None
    covariance: NDArray | None = None
    edf: NDArray | None = None
    edf1: NDArray | None = None
    rank: int | None = None
    curvature_floor: float = 0.0
    separation: Any = None

    def weak_root(self, columns: NDArray) -> NDArray | None:
        """``[-c'PU; PU]`` on ``columns``, the weak directions' published root with its intercept.

        ``P (V + U U') P' = P V P' + (P U)(P U)'``, so a reader forming a quadratic form (the
        intercept's variance ``c' V c``, a row's leverage) adds this term as ``(r' PU)^2``,
        never through the ``1 / C`` entries of the dense matrix, whose rounding is larger than
        the form.  ``None`` without weak directions.
        """
        if self.weak is None:
            return None
        columns = np.asarray(columns, dtype=np.intp)
        root = np.vstack((self.weak.intercept[None, :], self.weak.root[columns]))
        if self.gauge is not None:
            null = np.vstack((self.gauge.intercept[None, :], self.gauge.null[columns]))
            functionals = np.vstack((np.zeros((1, null.shape[1])), self.gauge.functionals[columns]))
            root = root - null @ np.linalg.solve(functionals.T @ null, functionals.T @ root)
        return root

    def coefficient_identified(self) -> NDArray:
        identified = np.ones(self.width, dtype=bool)
        identified[[alias.coordinate for alias in self.aliases if alias.coordinate is not None]] = (
            False
        )
        if self.separation is not None:
            identified[np.asarray(self.separation.reached, dtype=np.intp)] = False
        return identified

    def intercept_identified(self) -> bool:
        """Whether the intercept has a finite estimate: false when separation moves it."""
        return self.separation is None or not self.separation.intercept_reached

    def published_covariance(
        self, covariance: NDArray, columns: NDArray, *, weak: bool = True
    ) -> NDArray:
        """The published fit's slope covariance from its solver's ``covariance`` on ``columns``.

        Without ``weak`` the weak directions' term ``P U U' P'`` is left out, for a reader that
        adds it in low-rank form (``weak_root``).
        """
        if self.covariance is not None:
            covariance = self.covariance
        if self.weak is not None and weak:
            covariance = self.weak.covariance(covariance, columns)
        if self.gauge is not None:
            covariance = self.gauge.covariance(covariance, columns)
        return covariance

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
            ]
            + [
                {
                    "term": name,
                    "coordinate": None if name == "Intercept" else int(coordinate),
                    "aliased_with": "",
                    "reason": "no finite estimate: separated"
                    if self.separation.state == "present"
                    else "not resolved at float64: may be separated",
                    "source": "separation",
                }
                for name, coordinate in self._separation_coordinates()
            ],
            columns=["term", "coordinate", "aliased_with", "reason", "source"],
        )

    def _separation_coordinates(self) -> list[tuple[str, int]]:
        """``(name, fitted coordinate)`` of each coefficient separation withholds, intercept first."""
        if self.separation is None:
            return []
        coordinates = [-1] if self.separation.intercept_reached else []
        coordinates += [int(c) for c in self.separation.reached]
        return list(zip(self.separation.names, coordinates, strict=False))

    def notes(self) -> tuple[str, ...]:
        """The summary's footnote: each alias with what it is aliased with, and the accuracy."""
        notes: list[str] = []
        blank = [alias for alias in self.aliases if alias.coordinate is not None]
        held = [alias for alias in self.aliases if alias.coordinate is None]
        for aliases, header in (
            (blank, "{count} {plural} not identified (standard error shown as ---):"),
            (
                held,
                "{count} smooth-term {plural} not identified, held at zero (the terms' "
                "standard errors are the fit's with {pronoun} held):",
            ),
        ):
            if not aliases:
                continue
            count = len(aliases)
            if aliases is blank:
                plural = "coefficient" if count == 1 else "coefficients"
            else:
                plural = "part" if count == 1 else "parts"
            pronoun = "it" if count == 1 else "them"
            notes.append(header.format(count=count, plural=plural, pronoun=pronoun))
            grouped: dict[tuple[str, tuple[str, ...], str], int] = {}
            for alias in aliases:
                key = (alias.term, alias.aliased_with, alias.reason)
                grouped[key] = grouped.get(key, 0) + 1
            for (term, others, reason), size in grouped.items():
                unit = "coefficients" if aliases is blank else "parts"
                label = term if size == 1 else f"{term} ({size} {unit})"
                link = f" is aliased with {', '.join(others)}" if others else ""
                notes.append(f"  {label}{link}: {reason}.")
        if self.weak is not None and not self.weak.fitted:
            count = self.weak.root.shape[1]
            plural = "direction" if count == 1 else "directions"
            notes.append(
                f"{count} {plural} the solver's factorisation truncated "
                f"{'is' if count == 1 else 'are'} identified by the rows: the standard errors "
                "include them, and the coefficients along them stay where the solver stopped, "
                "so those estimates are not the penalised optimum."
            )
        if self.separation is not None:
            notes.extend(self.separation.notes())
        accuracy = self.penalised_condition * UNIT_ROUNDOFF
        # a held separated span is null on the kept rows, as an alias is: its condition is not
        # the standard errors'
        separated = self.separation is not None and self.separation.state == "present"
        if not self.aliases and not separated and accuracy > _ACCURACY_NOTE:
            digits = max(0, int(math.floor(-math.log10(accuracy)))) if accuracy < 1.0 else 0
            notes.append(
                f"Standard errors carry about {digits} significant digits: the penalised "
                f"Hessian's condition number is {self.penalised_condition:.1e}."
            )
        return tuple(notes)


@dataclass(frozen=True)
class TruncatedDecomposition:
    """A decomposition of ``H`` known by the part it certified and the directions it did not.

    It inverts ``H`` on a part it certified positive definite and leaves the directions
    ``null_vectors`` (``width x t``) undecided: a structured factor's truncated directions (the
    Moore-Penrose inverse of its Jacobi-scaled border), or a dense decomposition's truncated and
    uncertified ones (``state_ops``), so every coordinate they reach is left undetermined, as a
    spectral decomposition's (``fixed_coordinates``).  ``column_scale`` is ``sqrt(diag H)``,
    ``pre_truncation_condition`` the condition number of the certified part, ``solve`` the
    inverse on that part (``x -> V x``) and ``covariance`` that inverse as a matrix where it
    replaces the published decomposition's own.  ``floor`` is a certified lower bound on the
    certified part's Jacobi-scaled eigenvalues (zero when none is stated).
    """

    null_vectors: NDArray
    rank: int
    column_scale: NDArray
    pre_truncation_condition: float
    solve: Callable[[NDArray], NDArray] | None = None
    covariance: NDArray | None = None
    floor: float = 0.0

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


@dataclass(frozen=True)
class ConstructionColumns:
    """One term's columns of ``X_F`` in its own coordinates, in the order its construction defines.

    ``local`` (``m x w``) maps a block of ``w`` coefficients to the coefficients of the term's
    unpenalised construction columns on it, and ``basis`` (``w x m``, ``local basis = I``) holds
    those columns in the term's coordinates; ``blocks`` repeats the block (one per level of a
    factor smooth), so the term is ``blocks w`` wide and its columns run block by block, ``m``
    each.  An unpenalised term's columns are its coordinates (``local = I``), a ridge-penalised
    term has none, and a smooth's are its penalty null space, ordered and projected in the raw
    basis its construction defines (``state_ops``), so they depend on no row, weight, smoothing
    parameter or whitening.  ``name(block, j)`` names one.
    """

    local: NDArray
    basis: NDArray
    blocks: int
    name: Callable[[int, int], str]

    @property
    def unit(self) -> bool:
        width = self.local.shape[1]
        return self.local.shape[0] == width and bool(np.array_equal(self.local, np.eye(width)))


@dataclass(frozen=True)
class _Split:
    """The undecided directions read from the rows (``_read_rows``), in decomposition coordinates.

    ``refined`` are the directions ``T - V (H T)`` and ``values`` their curvature's eigenvalues
    in the eigenvectors ``vectors``: ``null`` under the rounding bound, ``identified`` above it
    by more than the coupling's uncertainty, the rest not resolved at float64.  ``score`` is the
    penalised score along ``T``, ``data`` the data Gram's product ``D T`` read from the rows.
    """

    refined: NDArray
    values: NDArray
    vectors: NDArray
    null: NDArray
    identified: NDArray
    limit: float
    score: NDArray | None
    data: NDArray
    candidates: NDArray
    penalised: NDArray
    shift: float = 0.0

    def directions(self, mask: NDArray) -> NDArray:
        return self.refined @ self.vectors[:, mask]


def decide_identifiability(
    decomposition: TruncatedDecomposition,
    *,
    columns: NDArray,
    width: int,
    groups: Sequence,
    design=None,
    W: NDArray | None = None,
    data_diagonal: NDArray | None = None,
    data_apply: Callable[[NDArray], NDArray] | None = None,
    penalty: Callable[[NDArray], tuple[NDArray, NDArray]] | None = None,
    specs: Mapping[Any, Any] | None = None,
    published=None,
    construction: Callable[[int], ConstructionColumns | None] | None = None,
    residual: NDArray | None = None,
    beta: NDArray | None = None,
    support: NDArray | None = None,
    separated: NDArray | None = None,
) -> Identifiability:
    """The identifiability record of a fit whose decomposition of ``H`` is ``decomposition``.

    ``columns`` maps the decomposition's coordinates to the ``width`` fitted ones, ``groups``
    are its term blocks (``GroupSlice``, in its coordinates).  ``design`` (their rows) and
    ``W`` read the undecided directions from the rows, with ``data_diagonal`` the diagonal of
    the centred data Gram ``D`` and ``data_apply`` its product (``V -> D V``); ``penalty``
    applies the penalty to them (``S T`` and the bound ``|S| |T|``), and ``specs`` (feature
    specs by name) name levels.  Without the rows every undecided direction counts as aliased.
    ``support`` marks the rows with positive prior weight, on which the design decides whether a
    direction the weighted rows cannot tell from zero is an alias (``_design_null``).

    The decision is ``_read_rows``'s, the one every backend takes.  A fit that publishes its
    coefficients (``construction`` given, with ``residual``, the working residual
    ``sqrt(W) (z - eta)``, and ``beta``) gets the gauge of its aliases on its terms'
    construction columns (``_construction_gauge``) and the fit of its weak directions
    (``WeakDirections``); one that cannot move them describes them as ``published``, its
    solver's decomposition, holds them (``_published_blanks``).

    ``separated`` (``width x k``, fitted slope coordinates) spans the directions a certified
    separation leaves infinite (``diagnostics.separation.Separation``) on a fit whose separated
    rows are withheld, where they are null: they are held where the fit left them, never moved,
    and no alias names them (the separation record does).  The record's ``curvature_floor`` is
    the certified part's (``TruncatedDecomposition.floor``), with the weak directions' curvature
    over their Jacobi norm, halved, where they are read; zero when a direction is not resolved.
    """
    columns = np.asarray(columns, dtype=np.intp)
    condition = float(decomposition.pre_truncation_condition)
    candidates = _orthonormal(decomposition, decomposition.null_vectors)
    if candidates.shape[1] == 0:
        return Identifiability(
            width, (), condition, float("nan"), curvature_floor=float(decomposition.floor)
        )
    scale = np.asarray(decomposition.column_scale, dtype=np.float64)
    candidates = candidates / np.where(scale > 0.0, scale, 1.0)[:, None]
    rows = design is not None and W is not None and data_diagonal is not None
    split = (
        _read_rows(decomposition, candidates, groups, design, W, data_diagonal, penalty, residual)
        if rows and penalty is not None
        else None
    )
    floating = np.zeros((decomposition.width, 0))
    if split is None:
        null, weak_mask = candidates, None
        fixed_effect_condition = float("nan")
    else:
        null = split.directions(~split.identified)
        weak_mask = split.identified
        kept = split.values[split.identified]
        fixed_effect_condition = (
            math.sqrt(float(np.max(split.values)) / float(np.min(kept))) if kept.size else 1.0
        )
        if support is not None and null.shape[1]:
            null, floating = _design_null(
                decomposition, null, groups, design, support, cast(Callable, penalty)
            )
    floor = _curvature_floor(decomposition, split, scale, floating.shape[1])
    quiet = 0
    if separated is not None and np.shape(separated)[1] and null.shape[1] and not floating.shape[1]:
        null, floating = _hold_separated(np.asarray(separated)[columns], null, scale)
        quiet = floating.shape[1]
    owner = np.empty(decomposition.width, dtype=np.intp)
    for index, group in enumerate(groups):
        owner[group.sl] = index
    facts = _RowFacts(design, W)
    weak = None
    if construction is None:
        both = np.hstack((null, floating))
        unresolved = np.r_[np.zeros(null.shape[1], bool), np.ones(floating.shape[1], bool)]
        chosen = _published_blanks(published, decomposition, both)
        aliases = _coordinate_aliases(
            chosen, both, unresolved, decomposition, columns, groups, owner, facts, specs or {}
        )
        gauge = None
    else:
        held = np.r_[np.zeros(null.shape[1], bool), np.ones(floating.shape[1], bool)]
        gauge, aliases = _construction_gauge(
            np.hstack((null, floating)),
            held,
            decomposition,
            columns,
            width,
            groups,
            owner,
            construction,
            data_diagonal,
            design,
            W,
            facts,
            specs or {},
        )
        if quiet:
            # the held separated directions follow the aliases; the separation record names them
            aliases = aliases[: _orthonormal(decomposition, null).shape[1]]
        if split is not None and weak_mask is not None and np.any(weak_mask):
            weak = _weak_directions(
                split,
                weak_mask,
                decomposition,
                columns,
                width,
                design,
                cast(NDArray, W),
                beta,
                data_apply,
            )
            accuracy = weak.accuracy / UNIT_ROUNDOFF
            condition = max(condition, accuracy)
    return Identifiability(
        width=width,
        aliases=tuple(aliases),
        penalised_condition=condition,
        fixed_effect_condition=fixed_effect_condition,
        gauge=gauge,
        row_directions=0 if split is None else candidates.shape[1],
        weak=weak,
        covariance=decomposition.covariance if construction is not None else None,
        rank=(
            None
            if construction is None
            else int(decomposition.rank) + (0 if weak is None else weak.root.shape[1])
        ),
        curvature_floor=floor,
    )


def _curvature_floor(
    decomposition: TruncatedDecomposition, split: _Split | None, scale: NDArray, floating: int
) -> float:
    """A certified lower bound on the Jacobi-scaled Hessian on the directions the record identifies.

    The certified part's is the decomposition's (``TruncatedDecomposition.floor``).  The weak
    directions ``U`` (``T - V H T``, H-orthogonal to it) have curvature at least their smallest
    Schur eigenvalue less the coupling's uncertainty, ``mu - shift``, per unit of ``c`` in
    ``U Q c``, so per Jacobi unit at least ``(mu - shift) / ||D^1/2 U Q||^2``; and for
    ``h = k + U Q c``, ``||h~||^2 <= 2 (||k~||^2 + ||U~ c||^2)``, so the joint bound is half the
    smaller.  A direction not resolved at float64 (``split``'s, or one the design identifies
    and the weights do not) leaves no bound: zero.  The null directions are the quotient.
    """
    floor = float(decomposition.floor)
    if split is None:
        return 0.0 if decomposition.null_vectors.shape[1] else floor
    unresolved = ~split.null & ~split.identified
    if floating or np.any(unresolved):
        return 0.0
    if not np.any(split.identified):
        return floor
    jacobi = scale[:, None] * split.directions(split.identified)
    norm = float(np.linalg.norm(jacobi, 2))
    if not norm > 0.0:
        return 0.0
    weak = (float(np.min(split.values[split.identified])) - split.shift) / norm**2
    return 0.5 * min(floor, weak) if weak > 0.0 else 0.0


def _hold_separated(separated: NDArray, null: NDArray, scale: NDArray) -> tuple[NDArray, NDArray]:
    """``null`` less the separated span, and the separated span, held (``decide_identifiability``).

    In Jacobi coordinates the separated directions' orthonormal basis ``Q`` is projected out of
    ``null``, and what keeps a singular value above the share floor stays an alias.
    """
    safe = np.where(scale > 0.0, scale, 1.0)[:, None]
    basis = np.linalg.qr(separated * safe)[0]
    rest = null * safe
    rest = rest - basis @ (basis.T @ rest)
    left, singular, _ = np.linalg.svd(rest, full_matrices=False)
    return left[:, singular > _SHARE_FLOOR] / safe, np.asarray(separated, dtype=np.float64)


def _read_rows(
    decomposition: TruncatedDecomposition,
    candidates: NDArray,
    groups: Sequence,
    design,
    W: NDArray,
    data_diagonal: NDArray,
    penalty: Callable[[NDArray], tuple[NDArray, NDArray]],
    residual: NDArray | None,
) -> _Split:
    """The decision on the undecided directions ``T``, from one pass over the rows.

    In the basis ``[K, T]``, ``K`` the part the decomposition certified positive definite, ``H``
    is singular exactly where the Schur complement ``C = T'HT - T'HK (K'HK)^-1 K'HT`` is (a
    positive definite block's Schur complement carries the nullity).  The rows give ``T'HT =
    B'B + T'ST``, ``B = sqrt(W) X_c T``, and ``HT = D T + S T`` with ``D T = A'B`` from the same
    pass (``grouped_projected_factor``); the decomposition's inverse on ``K`` gives ``V H T``, so
    ``C = T'HT - (HT)' V (HT)`` and ``T - V H T`` are the directions it is the curvature of.
    The decomposition left ``K'HT`` at its rounding (its own ``HT`` vanishes on ``K``), so the
    correction is that rounding's square over ``K``'s curvature, and its own error is bounded
    by ``||HT|| ||V HT||`` (taken as the shift).  An eigenvalue of ``C`` whose root is within
    the rounding bound of ``B``'s singular values (``_row_bound``) and of ``T'ST`` cannot be told
    from zero in float64: a direction of ``null(H)``.  One whose root clears it after the shift
    is identified (Weyl); any other is not resolved at float64.  The bound grows with the rows
    (``gamma_(n k) ||B||``, Higham 2002, Thm 19.4), so a near-duplicate column separated by
    about ``n u`` of its norm is aliased on a large book and identified on a small sample.
    """
    from superglm.solvers.centered_system import grouped_projected_factor, weighted_mean_pair

    weights = np.asarray(W, dtype=np.float64)
    _, centre, centre_lo = weighted_mean_pair(design, weights, float(np.sum(weights)))
    count = candidates.shape[1]
    score = None if residual is None else np.zeros(count)
    factor, cross, _, inflated = grouped_projected_factor(
        design,
        weights,
        candidates,
        center=centre,
        center_lo=centre_lo,
        response=None if residual is None else np.asarray(residual, dtype=np.float64),
        score_out=score,
    )
    factor = np.asarray(factor)
    if factor.shape[0] < count:
        factor = np.vstack((factor, np.zeros((count - factor.shape[0], count))))
    bound = _row_bound(candidates, groups, design, factor, data_diagonal, inflated)
    applied, magnitude = penalty(candidates)
    hessian = np.asarray(cross, dtype=np.float64) + applied
    corrected = (
        np.zeros_like(hessian)
        if decomposition.solve is None
        else np.asarray(decomposition.solve(hessian), dtype=np.float64)
    )
    curvature = candidates.T @ applied
    coupling = hessian.T @ corrected
    schur = factor.T @ factor + 0.5 * (curvature + curvature.T) - 0.5 * (coupling + coupling.T)
    # ``|fl(T'ST) - T'ST| <= gamma_(2p) |T|'|S||T|`` (Higham 2002, eq. 3.13)
    rounding = gamma_up(2 * candidates.shape[0]) * float(
        np.linalg.norm(np.abs(candidates).T @ magnitude)
    )
    limit = bound + math.sqrt(rounding)
    shift = float(np.linalg.norm(hessian, 2)) * float(np.linalg.norm(corrected, 2))
    values, vectors = np.linalg.eigh(schur)
    null = np.sqrt(np.maximum(values, 0.0)) <= limit
    identified = np.sqrt(np.maximum(values - shift, 0.0)) > limit
    return _Split(
        refined=candidates - corrected,
        values=values,
        vectors=vectors,
        null=null,
        identified=identified & ~null,
        limit=limit,
        score=score,
        data=np.asarray(cross, dtype=np.float64),
        candidates=candidates,
        penalised=applied,
        shift=shift,
    )


def _row_bound(
    candidates: NDArray,
    groups: Sequence,
    design,
    factor: NDArray,
    data_diagonal: NDArray,
    inflated: NDArray,
) -> float:
    """The rounding bound on the singular values of ``B = sqrt(W) X_c T`` read from the rows.

    One pass forms ``B`` about the exact centre pair and its triangle
    (``grouped_projected_factor``).  An entry of ``B`` is within ``gamma_(p + T + 3)`` of
    ``|a_r|' |t_j|`` (``a_r`` a centred row, inflated where a centre is added after the
    product), and a column of ``|A|`` has norm ``sqrt(D_ii)``; the triangle adds ``gamma_(n k)
    ||B||_F`` (Higham 2002, Thm 19.4) and the SVD ``gamma_(k^2)``.  A singular value within
    their sum of zero cannot be told from zero in float64 (Weyl).
    """
    width = candidates.shape[1]
    magnitude = np.maximum(
        np.sqrt(np.maximum(data_diagonal, 0.0)), np.asarray(inflated, dtype=np.float64)
    )
    formation = gamma_up(candidates.shape[0] + len(groups) + 3) * (magnitude @ np.abs(candidates))
    return float(np.linalg.norm(formation)) + gamma_up(
        int(design.n) * width + width * width
    ) * float(np.linalg.norm(factor))


def _design_null(
    decomposition: TruncatedDecomposition,
    directions: NDArray,
    groups: Sequence,
    design,
    support: NDArray,
    penalty: Callable[[NDArray], tuple[NDArray, NDArray]],
) -> tuple[NDArray, NDArray]:
    """Of the directions the weighted rows cannot tell from zero, those the design cannot either.

    With every lambda positive ``null(H) = N_S null(X N_S)`` on the rows with positive prior
    weight: a working weight carries no information about identifiability, only about its
    precision.  So the directions are read once more at unit weight on those rows
    (``grouped_projected_factor``): the curvature ``B_1'B_1 + T'ST`` of the Hessian at unit
    weight, and an eigenvalue whose root is within the rounding bound of ``B_1``'s singular
    values (``_row_bound``, the pass's own column norms) and of ``T'ST`` is an alias of the
    design, as in ``_read_rows``.  The rest are
    identified by the design and carry no weighted curvature the fit can resolve: they move
    fitted values, so they are never a gauge, and their coordinates' standard errors are not
    resolved at float64.  Returns the aliases and the rest, in decomposition coordinates.
    """
    from superglm.solvers.centered_system import grouped_projected_factor, weighted_mean_pair

    oriented = _orthonormal(decomposition, directions)
    scale = np.asarray(decomposition.column_scale, dtype=np.float64)
    directions = oriented / np.where(scale > 0.0, scale, 1.0)[:, None]
    count = directions.shape[1]
    if count == 0:
        return directions, directions
    weights = np.asarray(support, dtype=np.float64)
    _, centre, centre_lo = weighted_mean_pair(design, weights, float(np.sum(weights)))
    factor, _, _, magnitudes = grouped_projected_factor(
        design, weights, directions, center=centre, center_lo=centre_lo
    )
    factor = np.asarray(factor)
    if factor.shape[0] < count:
        factor = np.vstack((factor, np.zeros((count - factor.shape[0], count))))
    bound = _row_bound(
        directions, groups, design, factor, np.zeros(directions.shape[0]), magnitudes
    )
    applied, magnitude = penalty(directions)
    curvature = directions.T @ applied
    rounding = gamma_up(2 * directions.shape[0]) * float(
        np.linalg.norm(np.abs(directions).T @ magnitude)
    )
    values, vectors = np.linalg.eigh(factor.T @ factor + 0.5 * (curvature + curvature.T))
    rotated = directions @ vectors
    null = np.sqrt(np.maximum(values, 0.0)) <= bound + math.sqrt(rounding)
    return rotated[:, null], rotated[:, ~null]


def _weak_directions(
    split: _Split,
    mask: NDArray,
    decomposition: TruncatedDecomposition,
    columns: NDArray,
    width: int,
    design,
    W: NDArray,
    beta: NDArray | None,
    data_apply: Callable[[NDArray], NDArray] | None,
) -> WeakDirections:
    """The identified directions the decomposition truncated, fitted from the rows (``WeakDirections``).

    On the refined directions ``T Q`` (``Q`` the eigenvectors of ``C`` they take) the curvature
    is ``diag(mu)``, so their covariance is ``T Q mu^-1 Q'T'`` and the Newton step at the fit's
    weights ``T Q mu^-1 Q' g``, ``g = B'r - T'S beta`` the score the pass read (the decomposition
    solved its part of the score, so ``K``'s share of it is the fit's own tolerance).
    """
    from superglm.solvers.centered_system import weighted_mean_pair

    vectors = split.vectors[:, mask]
    values = split.values[mask]
    directions = split.refined @ vectors
    root = directions / np.sqrt(values)
    step = np.zeros(decomposition.width)
    fitted = split.score is not None and beta is not None
    if fitted:
        # the score along ``T``; along ``T - V H T`` it differs by ``(V H T)' g``, ``V g`` the
        # decomposition's solve of a score it has already solved
        score = split.score - split.penalised.T @ np.asarray(beta, dtype=np.float64)
        step = directions @ ((vectors.T @ score) / values)
    weights = np.asarray(W, dtype=np.float64)
    mean, _, _ = weighted_mean_pair(design, weights, float(np.sum(weights)))
    mean = np.asarray(mean, dtype=np.float64)
    # ``D (T - V H T)``: ``D T`` from the rows, ``D V H T`` at the formed Gram's rounding
    data = split.data - (
        0.0 if data_apply is None else np.asarray(data_apply(split.candidates - split.refined))
    )
    fitted_root, fitted_step = np.zeros((width, root.shape[1])), np.zeros(width)
    fitted_data = np.zeros((width, root.shape[1]))
    fitted_root[columns], fitted_step[columns] = root, step
    fitted_data[columns] = (data @ vectors) / np.sqrt(values)
    return WeakDirections(
        root=fitted_root,
        intercept=-(mean @ root),
        step=fitted_step,
        intercept_step=-float(mean @ step),
        accuracy=float(split.limit / np.sqrt(np.min(values))),
        data=fitted_data,
        fitted=fitted,
    )


def _construction_gauge(
    null: NDArray,
    unresolved: NDArray,
    decomposition: TruncatedDecomposition,
    columns: NDArray,
    width: int,
    groups: Sequence,
    owner: NDArray,
    construction: Callable[[int], ConstructionColumns | None],
    data_diagonal: NDArray | None,
    design,
    W: NDArray | None,
    facts: _RowFacts,
    specs: Mapping[Any, Any],
) -> tuple[Gauge | None, list[Alias]]:
    """The published constraints on ``null(H)``: the latest construction columns, in term order.

    Each null direction is written on the terms' construction columns of ``X_F``
    (``ConstructionColumns``), unit-normed by the centred data Gram's diagonal (a column's norm
    for an unpenalised term, its block's diagonal for a smooth's, the scale the conditioning
    check reads), and one column per
    direction is held at zero: the latest each still reaches in term order
    (``rank._earliest_representatives``, so the first-defined term keeps its coefficients), a
    choice of the construction alone.  It is kept unless it amplifies the retained columns'
    conditioning by more than ``_AMPLIFICATION_LIMIT`` (``rank._selection_amplification``); then
    the conditioning-first choice (``rank._leverage_pivot_representatives``) takes its place.
    Directions the construction columns cannot represent (an unresolved direction off the
    penalty null space) are held on the fitted coordinates instead.  The ``unresolved``
    directions (after the aliases) are held with them but never moved along (``Gauge.moved``).
    """
    from superglm.solvers.rank import (
        _earliest_representatives,
        _leverage_pivot_representatives,
        _selection_amplification,
    )

    aliases_first = _orthonormal(decomposition, null[:, ~unresolved])
    rest = _orthonormal(decomposition, null[:, unresolved])
    if rest.shape[1]:
        # the unresolved directions off the aliases' span, so the aliases keep their own
        rest = rest - aliases_first @ (aliases_first.T @ rest)
        left, singular, _ = np.linalg.svd(rest, full_matrices=False)
        rest = left[:, singular > _SHARE_FLOOR] if singular.size else left
    oriented = np.hstack((aliases_first, rest))
    unresolved = np.r_[np.zeros(aliases_first.shape[1], bool), np.ones(rest.shape[1], bool)]
    k = oriented.shape[1]
    if k == 0:
        return None, []
    scale = np.asarray(decomposition.column_scale, dtype=np.float64)
    safe = np.where(scale > 0.0, scale, 1.0)
    null = oriented / safe[:, None]
    touched = np.einsum("ij,ij->i", oriented, oriented) > float(np.finfo(float).eps) ** 2
    entries: list[tuple[int, int, int]] = []
    coefficients: list[NDArray] = []
    norms: list[NDArray] = []
    diagonal = (
        np.ones(decomposition.width)
        if data_diagonal is None
        else np.maximum(np.asarray(data_diagonal, dtype=np.float64), 0.0)
    )
    for index in np.unique(owner[touched]):
        group = groups[int(index)]
        own = construction(int(index))
        if own is None or own.local.shape[0] == 0:
            continue
        block, count = own.local.shape[1], own.local.shape[0]
        rows = null[group.sl].reshape(own.blocks, block, k)
        reached = np.flatnonzero(touched[group.sl].reshape(own.blocks, block).any(axis=1))
        coefficients.append(np.einsum("mw,bwk->bmk", own.local, rows[reached]).reshape(-1, k))
        local = diagonal[group.sl].reshape(own.blocks, block)[reached]
        norms.append(np.sqrt(local @ (own.basis * own.basis)).ravel())
        entries += [(int(index), int(level), j) for level in reached for j in range(count)]
    norms_array = np.concatenate(norms) if norms else np.zeros(0)
    norms_array = np.where(norms_array > 0.0, norms_array, 1.0)
    chosen: list[tuple[int, int, int]] = []
    if len(entries) >= k:
        written = np.vstack(coefficients) * norms_array[:, None]
        left, singular, _ = np.linalg.svd(written, full_matrices=False)
        if singular[-1] > _SHARE_FLOOR * singular[0]:
            keep = _earliest_representatives(left, len(entries) - k)
            if keep is not None and _selection_amplification(left, keep) > _AMPLIFICATION_LIMIT:
                pivoted = _leverage_pivot_representatives(left, len(entries) - k)
                if pivoted is not None and _selection_amplification(
                    left, pivoted
                ) < _selection_amplification(left, keep):
                    keep = pivoted
            if keep is not None:
                rejected = np.setdiff1d(np.arange(len(entries)), keep)
                chosen = [entries[int(i)] for i in rejected]
    functionals = np.zeros((decomposition.width, k))
    if len(chosen) == k:
        for i, (index, level, j) in enumerate(chosen):
            own = cast(ConstructionColumns, construction(index))
            block = own.local.shape[1]
            start = groups[index].start + level * block
            functionals[start : start + block, i] = own.local[j]
    else:
        keep = _earliest_representatives(oriented, decomposition.width - k)
        held = (
            np.argsort(-np.einsum("ij,ij->i", oriented, oriented))[:k]
            if keep is None
            else np.setdiff1d(np.arange(decomposition.width), keep)
        )
        functionals[held, np.arange(k)] = 1.0
        chosen = []
    if np.linalg.cond(functionals.T @ null) > 1.0 / UNIT_ROUNDOFF:
        return None, []
    from superglm.solvers.centered_system import weighted_mean_pair

    intercept = np.zeros(k)
    if design is not None and W is not None:
        weights = np.asarray(W, dtype=np.float64)
        mean, _, _ = weighted_mean_pair(design, weights, float(np.sum(weights)))
        intercept = -(np.asarray(mean, dtype=np.float64) @ null)
    fitted_null, fitted_functionals = np.zeros((width, k)), np.zeros((width, k))
    fitted_null[columns], fitted_functionals[columns] = null, functionals
    gauge = Gauge(
        null=fitted_null,
        functionals=fitted_functionals,
        intercept=intercept,
        moved=None if not np.any(unresolved) else ~unresolved,
    )
    aliases = []
    dual_lift = np.linalg.inv(functionals.T @ null)
    for i in range(k):
        functional = functionals[:, i]
        support = np.flatnonzero(functional)
        dual = functional / safe
        dual /= float(np.linalg.norm(dual))
        share = np.abs(oriented @ (oriented.T @ dual))
        representative = int(support[np.argmax(np.abs(functional[support]))])
        term = int(owner[representative])
        if support.size == 1:
            name = _coordinate_name(representative, groups[term], specs)
            coordinate: int | None = int(columns[representative])
        else:
            index, level, j = chosen[i]
            name = cast(ConstructionColumns, construction(index)).name(level, j)
            coordinate = None
        # the direction this constraint holds, ``N (G'N)^-1 e_i``, and whether it is unresolved
        held = oriented @ dual_lift[:, i]
        floating = np.linalg.norm(oriented[:, unresolved].T @ held) > np.linalg.norm(
            oriented[:, ~unresolved].T @ held
        )
        aliases.append(
            _alias(
                coordinate,
                name,
                representative,
                term,
                share,
                floating,
                decomposition,
                groups,
                owner,
                facts,
                specs,
            )
        )
    return gauge, aliases


def _published_blanks(published, decomposition, null: NDArray) -> NDArray:
    """The coordinates a fit that cannot move its coefficients leaves without an SE.

    Its coefficients are its solver's: where the published decomposition's zeros account for
    ``null(H)`` (no null direction vanishes on them) those zeros; otherwise its split along a
    null direction is arbitrary, and every coordinate the direction reaches is blank, with the
    zeros.  A zero the penalised fit identifies stays blank, since the fit reports zero there.
    """
    oriented = _orthonormal(decomposition, null)
    fixed = fixed_coordinates(decomposition if published is None else published)
    k = oriented.shape[1]
    spanned = k == 0 or (
        fixed.size >= k and np.linalg.svd(oriented[fixed], compute_uv=False)[k - 1] > _SHARE_FLOOR
    )
    return fixed if spanned else np.union1d(fixed, _reached(oriented))


def _coordinate_aliases(
    blank: NDArray,
    null: NDArray,
    unresolved: NDArray,
    decomposition,
    columns: NDArray,
    groups: Sequence,
    owner: NDArray,
    facts: _RowFacts,
    specs: Mapping[Any, Any],
) -> list[Alias]:
    """One alias per blank coordinate, as ``_alias`` names it."""
    oriented = _orthonormal(decomposition, null)
    aliases = []
    for coordinate in blank:
        share = np.abs(oriented @ oriented[coordinate]) if oriented.shape[1] else np.zeros(0)
        if not share.size or not share[coordinate] > _SHARE_FLOOR**2:
            aliases.append(
                Alias(
                    coordinate=int(columns[coordinate]),
                    term=groups[owner[coordinate]].name,
                    aliased_with=(),
                    reason="the published coefficients hold it at zero; the data and the "
                    "penalty identify it",
                    source="fit",
                )
            )
            continue
        term = int(owner[coordinate])
        floating = bool(
            np.any(unresolved)
            and np.linalg.norm(oriented[coordinate, unresolved])
            >= 0.5 * np.linalg.norm(oriented[coordinate])
        )
        aliases.append(
            _alias(
                int(columns[coordinate]),
                _coordinate_name(int(coordinate), groups[term], specs),
                int(coordinate),
                term,
                share,
                floating,
                decomposition,
                groups,
                owner,
                facts,
                specs,
            )
        )
    return aliases


def _alias(
    coordinate: int | None,
    name: str,
    representative: int,
    term: int,
    share: NDArray,
    floating: bool,
    decomposition,
    groups: Sequence,
    owner: NDArray,
    facts: _RowFacts,
    specs: Mapping[Any, Any],
) -> Alias:
    """An alias with what it is aliased with and why: counted where the construction proves it."""
    involved = sorted({term, *(int(owner[i]) for i in _reached_by(share))})
    others = tuple(groups[i].name for i in involved if i != term)
    if floating:
        return Alias(
            coordinate=coordinate,
            term=name,
            aliased_with=others,
            reason="not resolved at float64: the rows cannot tell its curvature from zero",
            source="float64",
        )
    reason = _construction_reason(
        representative, term, involved, decomposition, groups, facts, specs, share
    )
    return Alias(
        coordinate=coordinate,
        term=name,
        aliased_with=others,
        reason="no data separates them at float64" if reason is None else reason,
        source="rank" if reason is None else "construction",
    )


def published_edf(
    base_edf: NDArray,
    base_square: NDArray,
    solve: Callable[[NDArray], NDArray],
    data_apply: Callable[[NDArray], NDArray],
    weak: WeakDirections | None,
    columns: NDArray,
) -> tuple[NDArray, NDArray]:
    """``diag(F)`` and ``diag(2F - F^2)`` of the published fit's influence ``F``, on ``columns``.

    ``F_b = V D`` is the solver's (``base_edf = diag(F_b)``, ``base_square = diag(F_b^2)``,
    ``solve`` its ``V``, ``data_apply`` the data Gram ``D``).  The weak directions add ``U (D
    U)'``, ``U`` their root and ``D U`` read from the rows (``WeakDirections.data``), so ``F =
    F_b + U Y'`` and both diagonals follow from ``d`` products: Wood's EDF and ``edf1`` (2017,
    sec. 6.1.2) of the published fit.  The gauge moves along ``null(H)``, where ``D N = 0``, so
    it leaves the trace and is not read here.
    """
    columns = np.asarray(columns, dtype=np.intp)
    if weak is None:
        return base_edf.copy(), 2.0 * base_edf - base_square
    X, Y = weak.root[columns], weak.data[columns]
    edf = base_edf + np.einsum("ij,ij->i", X, Y)
    FX = np.asarray(solve(np.asarray(data_apply(X))), dtype=np.float64)
    FtY = np.asarray(data_apply(np.asarray(solve(Y))), dtype=np.float64)
    square = (
        base_square
        + np.einsum("ij,ij->i", FX, Y)
        + np.einsum("ij,ij->i", X, FtY)
        + np.einsum("ij,ij->i", X @ (Y.T @ X), Y)
    )
    return edf, 2.0 * edf - square


def _coordinate_name(coordinate: int, group, specs: Mapping[Any, Any]) -> str:
    """A factor level's coefficient by its level, ``f[E]``; any other by its term."""
    levels = getattr(specs.get(group.feature_name), "_non_base", None)
    local = coordinate - group.start
    if group.name == group.feature_name and levels is not None and len(levels) == group.size:
        return f"{group.name}[{levels[local]}]"
    return group.name


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
    "ConstructionColumns",
    "Gauge",
    "Identifiability",
    "TruncatedDecomposition",
    "WeakDirections",
    "certified_identifiability",
    "coordinates_identified",
    "decide_identifiability",
    "fixed_coordinates",
    "functional_identified",
    "penalty_null_space",
    "published_edf",
]
