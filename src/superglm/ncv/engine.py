"""The neighbourhood cross-validation criterion and its gradient at given smoothing parameters.

Wood, "On Neighbourhood Cross Validation" (arXiv 2404.16490v3), sections 2 and 2.2.  Rows ``i``
have the loss ``l_i(eta) = w_i d(y_i, mu(eta)) / 2`` and the scored loss ``D_i = 2 l_i``
(:mod:`superglm.ncv.rows`).  At the penalised mode ``theta = (alpha, beta)`` of the smoothing
``rho = log lambda``, in the centred coordinates of the fit's own Hessian
``H = blockdiag(sum W, H_c)``, a neighbourhood that removes rows ``a`` and scores rows ``d`` has

    g_a = Z_a' l'_a        H_a = H - Z_a' W_a Z_a        Delta = H_a^-1 g_a
    eta_i^(-a) = eta_i + z_i' Delta   (i in d)            V = sum_k sum_(i in d(k)) D_i(eta_i^(-a))

one Newton step of the fit without the rows ``a`` (exact for a Gaussian model).  ``eta`` carries
the offset, so the held-out predictor does too.  With ``r = Z_d' D'_d`` at the held-out predictor,
``u = H_a^-1 r`` and ``M = sum_k u_k Delta_k'``, the gradient is

    dV/d rho_j = -lambda_j [ sum(S_j * M) + (H^-1 q)' S_j beta ]
    q = sum_k (r_k + Z_a' W_a Z_a u_k) - sum_i l'''_i z_i (z_i' M z_i)
        + sum_k sum_(i in a(k)) l'''_i z_i (z_i' u_k) (z_i' Delta_k)

from ``d theta / d rho_j = -lambda_j H^-1 S_j theta`` and one adjoint solve per neighbourhood.

**Solves with** ``H_a``.  Two routes give the same numbers to rounding.  The neighbourhood-sized
system is the Woodbury identity on the whole neighbourhood (Wood, equation 4): with
``Y = Z_a H^-1``, ``B = Y Z_a'`` and the Cholesky factor of ``I - W^(1/2) B W^(1/2)``,
``H_a^-1 Z_a' r = Y' (r + W^(1/2) (I - W^(1/2) B W^(1/2))^-1 W^(1/2) B r)``.  Form and factor
accumulates ``Z_a' W_a Z_a`` over the neighbourhood's rows in bounded blocks and factors the
Jacobi-scaled ``H - Z_a' W_a Z_a``.  Beyond the row product both share, the first costs
``3 |a|^2 P + |a|^3`` and the second ``P^3`` (``P = p + 1``), which is the dispatch rule, with
``|a|`` every row the neighbourhood removes: a stack carries a removed row of zero weight as a
row of the identity (:func:`_stacked`).  A neighbourhood of one row takes the scalar form
``1 - W_i h_i``, with no factorisation.

**Stacks.**  Neighbourhoods on the first route are read in blocks of whole neighbourhoods,
sorted by size, and those of one size are solved together, as ``(members, size, size)``
arrays of ``B`` and of the systems: the single rows of a block by the scalar form, whether
they come from the single-row form or from groups or folds of one row, and each larger size
by one call of LAPACK's ``xGESV`` a right-hand side (``numpy.linalg.solve``).  The cost is
per row and per distinct size in a block, not per neighbourhood.  The stacked call fails as
a whole on a singular member and says nothing of which, where the batched LAPACK interface
returns a status per matrix (Abdelfattah et al., "A set of batched basic linear algebra
subprograms and LAPACK routines", ACM Trans. Math. Softw. 47(3), 2021), and it has no
condition estimate.  A stack therefore takes the members it can show sound beforehand:
those whose Gershgorin discs all lie right of ``sqrt(size)`` times the level this route
accepts a margin above (below).
Such a system is symmetric and strictly diagonally dominant, so it is positive definite
with its smallest eigenvalue above that bound; its margin, the reciprocal of
``||A^-1||_1 <= sqrt(size) ||A^-1||_2``, is above that level; and elimination on it
meets no zero pivot and grows no entry beyond twice the largest (Wilkinson; Higham 2002,
Theorem 9.9), so the solve is backward stable.  Every other member is factored and decided
on its own, in the neighbourhoods' order and before its block's stacks are scored.  The
route a neighbourhood takes, and whether it is flagged, fitted from its kept rows or
refused, are what they would be for the neighbourhood alone.  A forced route takes every
neighbourhood alone.

**Row work.**  Rows enter through four products, ``Z v``, ``Z' v``, ``Z' W Z`` and ``z' M z``,
and dense blocks of ``rows.chunk_rows(P)`` rows serve all four.  A form-and-factor
neighbourhood within one block is read once, in a block of whole neighbourhoods, for its
cross-products, its scores and its third-derivative term.  A set of rows of at least
``rows.COMPACT_ENTRIES`` dense entries is served by the design's stored form instead
(``rows.CentredRows``, ``rows.serves_compact``), and no row of it is made dense:

- A form-and-factor neighbourhood is a row subset of the design.  The solver's centred system
  on it is its Gram about its own weighted mean ``m``; about the fit's centre ``c`` that is
  ``G_m + w (m - c)(m - c)'`` with the intercept cross term ``w (m - c)``, two positive
  semidefinite terms, each formed on rows centred about an exact pair.  Its scores and its
  third-derivative term are products with the subset.
- The third-derivative pass over every weighted row needs ``z' M z`` for ``M = sum_k u_k
  Delta_k'``.  A design of lookups and dense columns reads it from its term-pair tables,
  whatever the rank of ``M``.  Any other design keeps ``sum_k (z' u_k)(z' Delta_k)`` row by
  row as the pairs arrive, while ``2 K <= P``: two products a pair against a block's ``P`` a
  row.

The neighbourhood-sized system and the single-row form multiply each row with the rows of
its own neighbourhood, and keep their dense blocks.  The compact products and the blocks
give the same numbers to rounding, so the rule decides cost only.

Rows of zero weight are in no system and no sum; a scored one is predicted at the full fit plus
the step.  A trial counts only at a usable inner mode (a certified one, or one at the limit of
float64, below), and with every weighted scored row on the scale where its mean is the
unclipped inverse link.

**The inner fit's two bars.**  The criterion is not stationary in the coefficients, so the
error an inner fit stops with enters the criterion and its gradient at first order, and a line
search on values that uncertain finds steps only while the slope exceeds about the square root
of the uncertainty times the criterion's curvature (Xie, Byrd and Nocedal, "Analysis of the
BFGS method with errors", arXiv:1901.09063, Theorems 3.4 and 3.16).  A solve therefore runs to
``ncv_mode_aim``, a thousandth of the search's tolerance.  Newton's method in floating point
contracts to a limiting accuracy and no further (Tisseur 2001, as
``solvers.mode_score.stagnation_window`` cites it), and that limit can lie above the aim: a
Gaussian response 1e7 of its spread from zero is resolved to between 1e-10 and 2e-8 of the
score's scale, by how its rounding falls.  A solve that ends short of the aim is continued, from
the state it stopped on, to ``ncv_mode_bar``, a hundredth of the tolerance
(:func:`two_level`).  A fit certified at either bar is a certified mode.  The limit is taken
to be the data's more than the smoothing's, so a fit carries the bar its solve ran to and a
solve it warms runs to that bar at once: a search pays once for an aim its data cannot reach.

**A fit at the limit of float64.**  The limit can lie above the acceptance bar as well: one
240-row response 1e8 of its spread from zero is resolved to 1e-7 of the score's scale.  The
solver ends such a solve as stagnated (``"score_stagnated"``), a verdict it gives on two
grounds only (``solvers.irls_direct``): a full Newton step left the state bitwise unchanged,
or a score within ``mode_score.MODE_RESOLVE_CAP`` of zero on every coefficient did not halve
over ``mode_score.stagnation_window`` iterations.  On either the state is as near the mode
as float64 brings it for the data, so a stagnated fit is used as the mode
(:attr:`ModeFit.usable`): its trial has a criterion, it warms the next solve, and no solve
from another start follows, which would end at the same limit.  The continuation of a
stagnated solve takes one step from the state it stopped on, so the state a trial is
evaluated at is the same whether the acceptance bar is then met or not, and the criterion
does not depend on which side of that bar the rounding falls.  A stagnated fit is not a
certified mode: an evaluation made at one is counted (``WorkCounts.stagnated_fits``), and
the search and the published fit that rest on one are not converged.  A solve that ends
short of its bar in any other way (its iteration limit, a rejected step, a deviance that is
not finite) says nothing of where the mode is: it is solved once more from the solver's own
start, and its trial has no criterion if that solve ends the same way.

**A rank-deficient fit.**  When the fit's own slope Hessian has a null space (a duplicated
column, a column constant on the weighted rows), every weighted row and every penalty with a
positive smoothing parameter annihilates it, so no held-out prediction on a weighted row
depends on it.  The evaluation runs in the coordinates the fit identified, ``theta = T phi``:
rows ``Z T``, the Hessian ``T' H T`` and the solver's own inverse of it.  ``T`` selects the
solver's representative slopes, a gather, or is its retained basis where it kept one (a weak
direction beside an alias), in whose coordinates the Hessian is the diagonal of its retained
values.  ``q`` and ``M`` are lifted back, ``T q`` and ``T M T'``, for the penalty contractions.
A full-rank fit has no ``T`` and pays nothing.

**A questionable leave-out system.**  A downdate subtracts two Grams, each known to
``gamma_n`` of ``H``, and its step carries the full fit's rounding divided by its margin
(Stewart, "The effects of rounding error on an algorithm for downdating a Cholesky
factorization", J. Inst. Math. Appl. 23, 1979: the algorithm is stable, the downdated factor
can be an ill-conditioned function of its data).  A factorisation margin therefore decides
nothing: at or below ``max(sqrt(eps), 2 P gamma_n)`` it only flags the neighbourhood.  The
margin of a factored system, on its unit diagonal, is the reciprocal of LAPACK's estimate of
the 1-norm of its inverse (``xPOCON``, ``O(P^2)`` from the Cholesky factor).  That is at
most the smallest eigenvalue up to the estimator's slack, whose estimate of the norm is a
lower bound and in practice within a factor of ten (oneMKL Developer Reference, ``?pocon``);
the smallest pivot only bounds that eigenvalue from above.  For a single row it is
``1 - W_i h_i``, the one eigenvalue of its system that is not one.  A flagged neighbourhood
is fitted from the rows it keeps.  Its one-step fit is the penalised weighted least-squares
fit of the working response ``eta - o - l' / l''`` to them, centred about their own weighted
mean as the solver centres a fit, on the solver's row factor and under its rank rule
(``rank.decompose_factor``).  Where that factor is short of rank, the kept rows and the
penalties either leave a direction free at every smoothing (a null direction of the kept rows'
own factor that no penalty with a positive smoothing parameter sees), which is refused with
:class:`LeaveOutRankError`, or they do not, and the trial is infeasible at this smoothing.

**A margin the inverse does not resolve.**  The neighbourhood-sized system is built on
``H^-1``, so its margin carries that inverse's error, which grows with the condition of
``H``.  With ``D = diag(H)^(-1/2)``, the Hessian the solver formed differs from the exact one
by ``dH`` with ``||D dH D||_2 <= P gamma_n`` (each entry of a Gram within ``gamma_n`` of a
unit diagonal; Higham 2002, section 3.5).  The rows of the fit have
``||W^(1/2) Z H^(-1/2)||_2 <= 1``, so the exact inverse of the formed Hessian moves
``W^(1/2) Z H^-1 Z' W^(1/2)``, and with it every eigenvalue of a leave-out system (Weyl), by
at most ``eta / (1 - eta)``, ``eta = P gamma_n ||(D H D)^-1||_2``.  The computed inverse and
the two products with it add terms of the same form whose constants hold ``P`` and no ``n``
(Du Croz and Higham, "Stability of methods for matrix inversion", IMA J. Numer. Anal. 12,
1992: the inverse from a Cholesky factor has a residual of order ``u kappa``), and the
identity is not backward stable where ``H`` is ill conditioned, whatever solves with ``H``
(Hashemi and Nakatsukasa, "Instability of the Sherman-Morrison formula and stabilization by
iterative refinement", arXiv:2510.01696: a backward error of order ``u kappa(H)``).  A leverage
from the inverse of a Gram is therefore known to ``u`` times the Gram's condition, the
square of the rows' own, where the leverage itself is conditioned by the rows' (Holodnak,
Ipsen and Wentworth, "Conditioning of leverage scores and computation by QR decomposition",
arXiv:1402.0957, Theorem 2.4).  The margin is known to ``2 P gamma_n ||(D H D)^-1||_1``: the
Gram's term twice over, which is at least ``eta / (1 - eta)`` wherever it is below one and
leaves the other terms as much again while ``eta`` is small.  A margin only above that bound
fixes its sign and not its size, and the step it divides carries the margin's relative error:
``U / m`` for a margin ``m`` known to ``U``.  A system is therefore solved on this route
only when its margin exceeds :data:`RESOLVED` times the bound, which leaves the step within
1/63 of the exact one at worst.  The 1-norm is at least the 2-norm and is ``O(P^2)`` to read
from the Hessian's diagonal and the solver's inverse.  The product of the system's margin and the
full fit's own, which bounds the smallest eigenvalue of its Jacobi-scaled leave-out Hessian
from below, then clears the error of that Hessian's two Grams.  A system at or below the
level, and above the flag level, is formed and factored instead, where the margin is read
from ``H - Z_a' W_a Z_a`` itself and owes nothing to ``H^-1``; that margin flags it for its
kept rows or does not.

**A single row the inverse does not resolve.**  That bound is the worst case over every row.
A row's own leverage ``h = s' A^-1 s``, ``A = D H D`` as the solver formed it and ``s = D z``,
moves under a perturbation ``E`` of ``A`` by ``x' E y`` for ``x = A^-1 s`` and
``y = (A + E)^-1 s``, at most ``||E|| ||x||^2 / (1 - eta)``, ``eta = ||E|| ||A^-1||_2``.  The
exact Hessian's leverage is that of such an ``A + E``, ``E`` the formed Gram's error
(``P gamma_n``).  Taken through the Cholesky factor of ``A``, one an evaluation, by a forward
and a back substitution, the computed leverage is that of another, and the computed solved
row ``x`` that of a third, each ``E`` of the form of a Gram's error: the scaling's
(``gamma_2`` an entry), the factorisation's (``gamma_(P+1) |L||L'|``, Higham 2002, Theorem
10.3) and the two substitutions' (``2 gamma_P |L||L'|``, Theorem 8.5), where ``|L||L'|`` is
at most one an entry on a unit diagonal.  With ``||E||_2 <= P gamma_(n + 3P + 6)`` for the
two together and the computed ``x``, the leverage is known to
``|w| (||E|| ||x||^2 (1 + eta)^2 / (1 - eta) + gamma_(P+2) h)`` while ``eta <= 1/2``.  A single row whose margin ``1 - w h`` from the factor clears
:data:`RESOLVED` times that is solved by the scalar form on the factor's leverage and solved
row (:func:`_through_factor`); the bound's own roundings leave its step within 1/63.  The
Rayleigh quotient of ``A - w s s'`` at ``x`` is ``h (1 - w h) / ||x||^2``, so a row it leaves
has a leave-out Hessian whose smallest eigenvalue on a unit diagonal is below
``RESOLVED (||E|| (1 + eta)^2 / (1 - eta) + P gamma_(P+2))``, about ``64 P gamma_n``: near the
flag level, where its own factorisation decides it as above.

**Gaussian cross-products.**  For a Gaussian identity model ``l'' = w`` and the centre do not
move with ``rho``, so a form-and-factor neighbourhood's ``G_a = Z_a' W_a Z_a`` is formed once
and kept by the problem, with ``G_d`` over its scored rows and both scores at a reference fit
``theta_0``.  At a later fit, with ``delta = theta - theta_0``,

    g_a = g_a(theta_0) + G_a delta        g_d = g_d(theta_0) + G_d delta
    V_k = sum_(i in d) w_i (eta_i - y_i)^2 + 2 Delta' g_d + Delta' G_d Delta
    r_k = 2 (g_d + G_d Delta)             q += r_k + G_a u_k

and no row is read.  The sum of squares is taken over the fit's own residuals, one term per
scored occurrence, and the cross-products hold residual scores, so a large response level
cancels nowhere.  ``V_k`` gives no prediction per pair: an evaluation asked for its held-out
pairs reads the scored rows.

**Memory.**  No ``rows x P`` array of the whole design is formed.  Rows are read in blocks of
``rows.chunk_rows(P)``; a form-and-factor neighbourhood within one block is held from a block
of whole neighbourhoods while it is solved, a larger one is streamed through such blocks pass
by pass, and the neighbourhoods on the neighbourhood-sized route come in blocks of at most
that many rows and one neighbourhood more (under the cost rule each has fewer than ``0.54 P``
weighted rows).  An evaluation holds a block and its solved rows, one more array of that
size at most (a stack's ``(u, Delta)`` pairs, or two blocks of pairs from neighbourhoods
taken one at a time), ``size x size`` per stacked neighbourhood of the block (at most
``0.54`` of the block), a handful of ``P x P`` matrices, and row-length vectors: the mode's
rows and five entries per scored pair.  Single rows the inverse does not resolve take two
more arrays of a block's size at most through the fit's factor.  A neighbourhood served as a row subset holds that subset's
stored form (an index per lookup term, its dense columns) and the solver's plan for it, one
neighbourhood at a time; the compact third pass adds row-length vectors and reads the design's
own lookup plan, which the design holds until the fit is published.  Between evaluations only a
Gaussian problem keeps anything:
the reference predictor and ``8 (P^2 + P)`` bytes per cached neighbourhood, twice that when it
scores part of what it removes, within ``rows.cache_allowance()``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, cholesky, get_lapack_funcs, solve_triangular

from superglm._roundoff import EPS, gamma
from superglm.distributions import Distribution
from superglm.group_matrix import DesignMatrix
from superglm.links import Link
from superglm.ncv.neighbourhoods import Neighbourhoods
from superglm.ncv.rows import (
    CentredRows,
    RowLoss,
    augmented_rows,
    cache_allowance,
    chunk_rows,
    row_chunks,
    row_loss,
    serves_compact,
)
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.solvers.centered_system import (
    grouped_augmented_factor_rhs,
    grouped_weighted_factor,
    penalty_factor,
    weighted_mean_pair,
)
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.mode_score import centred_warm_start, linear_predictor
from superglm.solvers.pirls import PIRLSResult
from superglm.solvers.rank import SHARED_RANK_POLICY, decompose_factor
from superglm.types import GroupSlice, PenaltyComponent

Route = Literal["woodbury", "factor"]
"""A leave-out solve route: the neighbourhood-sized system, or form and factor.

Forced on an evaluation, a route takes every neighbourhood by it, one at a time.
"""

InfeasibleReason = Literal[
    "uncertified_mode", "clipped_prediction", "non_finite", "leave_out_rank", "rank_deficient_fit"
]
"""Why a trial gave no criterion.

``"leave_out_rank"``: a leave-out system is unusable at this smoothing though its kept rows
and penalties identify the model, or more systems needed their kept rows than ``KEPT_FITS``.
``"rank_deficient_fit"``: the fit is rank deficient and hands back no coordinates to work in.
"""

DEFAULT_NCV_TOL = 1e-6

KEPT_FITS = 8
"""Leave-out systems one evaluation fits from their kept rows.

Each is a pass of the solver's row factor over the design, so an evaluation that needs more
is an infeasible trial: a limit on cost, not a tolerance.
"""

RESOLVED = 64.0
"""How many times the error bound of its leverages a margin solved from ``H^-1`` must exceed.

The margin is then known to a sixty-fourth of itself, so the leave-out step is within 1/63
of the exact one at worst.  A margin at or below that level is decided by the system's own
factorisation (module docstring, "A margin the inverse does not resolve").
"""

_POCON = get_lapack_funcs("pocon", dtype=np.float64)


def ncv_mode_bar(ncv_tol: float = DEFAULT_NCV_TOL) -> float:
    """The certificate bar an inner fit is accepted at, for a search stopping at ``ncv_tol``.

    A hundredth of the tolerance: the looser of the inner fit's two bars (module docstring),
    which a fit short of :func:`ncv_mode_aim` is taken to.  A fit short of it is certified
    at neither bar, and is used only where its solve ended at the limit of float64
    (:attr:`ModeFit.usable`).  Never below ``100 eps``, the least the certificate expresses.
    """
    return max(0.01 * float(ncv_tol), 100.0 * EPS)


def ncv_mode_aim(ncv_tol: float = DEFAULT_NCV_TOL) -> float:
    """The certificate bar an inner fit first runs to, for a search stopping at ``ncv_tol``.

    A thousandth of the tolerance: at a hundredth, two fits at one smoothing differ in the
    criterion by about 1e-9 of it at the default tolerance, the size of a search's last
    gains.  Never below ``100 eps``, so never above :func:`ncv_mode_bar`.
    """
    return max(0.001 * float(ncv_tol), 100.0 * EPS)


class LeaveOutRankError(ValueError):
    """The rows left when a neighbourhood is removed do not identify the model.

    ``neighbourhood`` is its index; ``margin`` the factorisation margin that flagged it.
    """

    def __init__(self, neighbourhood: int, where: str, margin: float) -> None:
        super().__init__(
            f"With {where} held out, the rows that remain do not identify the model: two of "
            "its columns coincide on them, or a term has no rows left. Merge the levels "
            "involved, or choose neighbourhoods that divide their rows."
        )
        self.neighbourhood = neighbourhood
        self.margin = margin


@dataclass
class WorkCounts:
    """Work done by one evaluation, counted where it happens.

    ``rows_read`` are design rows materialised.  ``rows_multiplied`` are rows that entered a
    product costing ``(p + 1)^2`` each: solved rows, leave-out Grams, the signed Gram and the
    third-derivative pass.  ``rows_streamed`` are rows a pass took through compact products
    in place of a dense block (``rows.CentredRows``), once per pass, and are in neither of
    the other two.  Divide any of them by the row count for passes over the data.
    ``small_factorisations`` are neighbourhood-sized systems factored one at a time, and
    ``solves`` the solves with one factored system, of either size.  ``stacked`` are the
    neighbourhood-sized systems solved together, in ``stacks`` of one size, each stack by one
    LAPACK call a right-hand side; a single row's scalar system is in none of these.
    ``resolved_by_factor`` are single rows whose margin the full fit's inverse did not
    resolve and its Cholesky factor did (:func:`_through_factor`).  ``kept_fits`` are
    leave-out systems fitted from their kept rows, each one or two passes of the solver's row
    factor over the design, which neither row count includes.
    ``fallback_solves`` are the inner solves, of ``inner_solves``, made because an earlier one
    at the same smoothing ended short of its bar (:func:`two_level`).  ``stagnated_fits``
    are the evaluations that took their criterion from a fit at the limit of float64, short
    of its bar (:attr:`ModeFit.stagnated`): 0 or 1 for one evaluation.
    """

    inner_solves: int = 0
    fallback_solves: int = 0
    stagnated_fits: int = 0
    pirls_iterations: int = 0
    factorisations: int = 0
    small_factorisations: int = 0
    stacks: int = 0
    stacked: int = 0
    solves: int = 0
    rows_read: int = 0
    rows_multiplied: int = 0
    rows_streamed: int = 0
    resolved_by_factor: int = 0
    kept_fits: int = 0


@dataclass(frozen=True)
class _Moments:
    """One neighbourhood's cross-products, in the design's own columns.

    ``gram`` and ``score`` are ``Z' W Z`` and ``Z' l'`` over its weighted removed rows;
    ``scored_gram`` and ``scored_score`` the same over its scored rows, the same arrays when
    it scores every row it removes.  Held by a cache, the scores are at its reference fit.
    """

    gram: NDArray[np.float64]
    score: NDArray[np.float64]
    scored_gram: NDArray[np.float64]
    scored_score: NDArray[np.float64]

    def moved(self, shift: NDArray[np.float64]) -> _Moments:
        """The cross-products at a fit ``shift`` away: ``l'`` is linear in the coefficients."""
        score = self.score + self.gram @ shift
        if self.scored_gram is self.gram:
            return _Moments(self.gram, score, self.gram, score)
        scored_score = self.scored_score + self.scored_gram @ shift
        return _Moments(self.gram, score, self.scored_gram, scored_score)


def _same(left: NDArray | None, right: NDArray | None) -> bool:
    return left is right or (left is not None and right is not None and np.array_equal(left, right))


@dataclass
class _GaussianCache:
    """A Gaussian problem's cross-products, kept between evaluations.

    Owner: its :class:`NCVProblem`, which builds its own and shares it with no other problem,
    so a problem built again holds nothing.  Lifetime: the problem's.  It holds one
    neighbourhood structure about one centre pair, and an evaluation with another drops
    everything held.  ``afforded`` marks the neighbourhoods ``rows.cache_allowance()`` pays
    for after the reference predictor, largest first; the others are formed from their rows
    at every evaluation.  ``eta`` and ``beta`` are the reference fit's, empty while nothing
    is afforded.
    """

    nb: Neighbourhoods | None = None
    centre_hi: NDArray[np.float64] | None = None
    centre_lo: NDArray[np.float64] | None = None
    eta: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    beta: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    afforded: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    held: dict[int, _Moments] = field(default_factory=dict)

    def attach(
        self,
        nb: Neighbourhoods,
        fit: ModeFit,
        eta: NDArray[np.float64],
        weighted: NDArray[np.intp],
        width: int,
    ) -> None:
        """Keep what is held if it was formed for ``nb`` about ``fit``'s centre; else start again at ``fit``.

        ``weighted`` are the weighted removed rows per neighbourhood, ``width`` is ``P``.
        """
        if (
            self.nb is nb
            and _same(self.centre_hi, fit.centre_hi)
            and _same(self.centre_lo, fit.centre_lo)
        ):
            return
        beta = fit.result.beta
        entry = 8 * (width * width + width) * (1 if nb.score_ptr is None else 2)
        room = cache_allowance() - eta.nbytes - beta.nbytes
        count = min(nb.m, max(room, 0) // entry)
        self.afforded = np.zeros(nb.m, dtype=bool)
        self.afforded[np.argsort(-weighted, kind="stable")[:count]] = True
        self.nb, self.centre_hi, self.centre_lo = nb, fit.centre_hi, fit.centre_lo
        self.eta, self.beta = (eta, beta) if count else (np.zeros(0), np.zeros(0))
        self.held = {}

    @property
    def nbytes(self) -> int:
        """Bytes held: the reference fit's rows and every cached cross-product."""
        arrays = {
            id(array): array.nbytes
            for entry in self.held.values()
            for array in (entry.gram, entry.score, entry.scored_gram, entry.scored_score)
        }
        return self.eta.nbytes + self.beta.nbytes + sum(arrays.values())


@dataclass(frozen=True)
class NCVProblem:
    """What the engine reads of a model: its design, family, rows and penalties.

    ``names`` are the estimated penalty components, in the order of ``rho``; ``blocks`` holds
    each one's columns and unit penalty matrix on them.  ``fixed_lambdas`` are the components
    held at a value, which every inner fit keeps.  ``mode_aim`` and ``mode_bar`` are the
    certificate bars an inner fit runs to and is accepted at (:func:`two_level`).  ``cache``
    is this problem's own: no constructor argument sets it, so a problem built or copied anew
    starts with nothing held.
    """

    dm: DesignMatrix
    y: NDArray[np.float64]
    weights: NDArray[np.float64]
    offset: NDArray[np.float64]
    distribution: Distribution
    link: Link
    groups: list[GroupSlice]
    penalties: list[PenaltyComponent]
    weight_semantics: str
    names: tuple[str, ...]
    blocks: tuple[tuple[slice, NDArray[np.float64]], ...]
    fixed_lambdas: Mapping[str, float]
    loss: RowLoss
    active: NDArray[np.bool_]
    mode_aim: float
    mode_bar: float
    pirls_tol: float
    max_pirls_iter: int
    cache: _GaussianCache = field(
        init=False, default_factory=_GaussianCache, compare=False, repr=False
    )

    @property
    def n(self) -> int:
        """Rows of the design."""
        return int(self.dm.n)

    @property
    def width(self) -> int:
        """Coefficients with the intercept, ``P = p + 1``."""
        return int(self.dm.p) + 1

    def lambdas(self, rho: NDArray[np.float64]) -> dict[str, float]:
        """Every component's smoothing parameter at the estimated ``rho``."""
        estimated = dict(zip(self.names, np.exp(rho).tolist(), strict=True))
        return {**self.fixed_lambdas, **estimated}


def build_problem(
    *,
    dm: DesignMatrix,
    groups: list[GroupSlice],
    penalties: list[PenaltyComponent],
    distribution: Distribution,
    link: Link,
    y: NDArray,
    weights: NDArray,
    offset: NDArray | None,
    weight_semantics: str,
    fixed_lambdas: Mapping[str, float] | None = None,
    mode_bar: float | None = None,
    mode_aim: float | None = None,
    pirls_tol: float = 1e-8,
    max_pirls_iter: int = 100,
) -> NCVProblem:
    """The engine's problem for a built design.

    ``penalties`` are the design's components (``penalty_algebra.build_penalty_context``).  A
    component named in ``fixed_lambdas`` is held at that value; every other one is estimated.
    ``mode_bar`` and ``mode_aim`` default to the bars of the default search tolerance
    (:func:`ncv_mode_bar`, :func:`ncv_mode_aim`), and an aim above the bar is the bar: a
    problem given a bar below the default aim solves to that bar alone.
    """
    held = dict(fixed_lambdas or {})
    estimated = [component for component in penalties if component.name not in held]
    weights = np.asarray(weights, dtype=np.float64)
    bar = ncv_mode_bar() if mode_bar is None else float(mode_bar)
    aim = ncv_mode_aim() if mode_aim is None else float(mode_aim)
    return NCVProblem(
        dm=dm,
        y=np.asarray(y, dtype=np.float64),
        weights=weights,
        offset=np.zeros(dm.n) if offset is None else np.asarray(offset, dtype=np.float64),
        distribution=distribution,
        link=link,
        groups=groups,
        penalties=penalties,
        weight_semantics=weight_semantics,
        names=tuple(component.name for component in estimated),
        blocks=tuple(_penalty_block(dm, groups, penalties, component) for component in estimated),
        fixed_lambdas=held,
        loss=row_loss(distribution, link),
        active=weights > 0.0,
        mode_aim=min(aim, bar),
        mode_bar=bar,
        pirls_tol=float(pirls_tol),
        max_pirls_iter=int(max_pirls_iter),
    )


def _penalty_block(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    penalties: list[PenaltyComponent],
    component: PenaltyComponent,
) -> tuple[slice, NDArray[np.float64]]:
    """``component``'s columns and its penalty matrix on them at ``lambda = 1``, as the solver builds it."""
    unit = {other.name: float(other.name == component.name) for other in penalties}
    full = build_penalty_matrix(
        list(dm.group_matrices), groups, unit, dm.p, reml_penalties=penalties
    )
    columns = component.group_sl
    block = full[columns, columns].copy()
    full[columns, columns] = 0.0
    if full.any():
        raise ValueError(
            f"Penalty component {component.name!r} reaches outside the columns of its own term, "
            "which neighbourhood cross-validation does not support; choose this model's "
            "smoothing with fit_reml."
        )
    return columns, block


@dataclass(frozen=True)
class ModeFit:
    """A fixed-smoothing fit with the geometry the leave-out solves read.

    ``slope_inverse`` and ``slope_hessian`` are ``H_c^+`` and ``H_c`` of the final system, whose
    rows are centred as ``(x - centre_hi) - centre_lo``; with ``1 / sum_w`` the first is ``H^-1``.
    ``data_gram`` is that system's ``X' W X``, the Hessian without its penalty, as the solver
    formed it; a fit not made by :meth:`from_solve` may carry none.  All three are in the
    design's own columns, the ones ``NCVProblem.blocks`` index, whatever the fit's rank: the
    coordinates a rank-deficient fit identified are read from ``result``, never from a shape.
    ``bar`` is the certificate bar the solve ran to; a fit made without one carries ``None``,
    which says nothing of what its data can reach.
    """

    lambdas: dict[str, float]
    result: PIRLSResult
    slope_inverse: NDArray[np.float64]
    slope_hessian: NDArray[np.float64]
    sum_w: float
    centre_hi: NDArray[np.float64]
    centre_lo: NDArray[np.float64] | None
    data_gram: NDArray[np.float64] | None = None
    bar: float | None = None

    @classmethod
    def from_solve(
        cls,
        lambdas: Mapping[str, float],
        result: PIRLSResult,
        slope_inverse: NDArray[np.float64],
        cache: Mapping[str, Any],
        bar: float | None = None,
    ) -> ModeFit:
        """From what ``fit_irls_direct(..., lambda2=lambdas, return_xtwx=True, cache_out=cache)`` hands back.

        Solve with ``_retain_reml_decomposition=True``: a rank-deficient fit whose solver kept a
        basis and not representative columns is evaluated in that basis, read from ``result``.
        ``bar`` is the solve's ``_mode_bar``.
        """
        centre_hi, centre_lo = cache["centre_pair"]
        return cls(
            lambdas=dict(lambdas),
            result=result,
            slope_inverse=slope_inverse,
            slope_hessian=cache["centered_hessian"],
            sum_w=float(cache["sum_W"]),
            centre_hi=centre_hi,
            centre_lo=centre_lo,
            data_gram=cache.get("centered_XtWX"),
            bar=bar,
        )

    @property
    def certified(self) -> bool:
        """Whether the solve ended on its mode certificate."""
        return bool(self.result.converged)

    @property
    def stagnated(self) -> bool:
        """Whether the solve ended at the limit of float64 accuracy, short of its bar.

        The solver's own verdict, ``"score_stagnated"`` (module docstring): no threshold is
        applied here.
        """
        return not self.certified and self.result.termination_reason == "score_stagnated"

    @property
    def usable(self) -> bool:
        """Whether the fit stands for the mode: certified, or stagnated at the limit of float64."""
        return self.certified or self.stagnated


def warm_start(fit: ModeFit) -> dict[str, Any]:
    """The solver's start arguments for the state ``fit``'s solve ended on."""
    return {
        "beta_init": fit.result.beta,
        "intercept_init": float(fit.result.intercept),
        "_centred_init": centred_warm_start(fit.result),
    }


def first_bar(warm: ModeFit | None, aim: float, bar: float) -> float:
    """The bar a solve warmed by ``warm`` first runs to.

    The bar ``warm``'s solve ran to, within ``[aim, bar]``: a fit accepted only at ``bar``, or
    stagnated short of it, shows the aim is beyond what its data resolve, and the next solve
    does not try for it.  ``aim`` with no fit, or with one that carries no bar.
    """
    if warm is None or warm.bar is None:
        return aim
    return max(aim, min(warm.bar, bar))


def two_level(
    solve: Callable[[dict[str, Any], float], ModeFit], start: dict[str, Any], aim: float, bar: float
) -> ModeFit:
    """A fit by ``solve(start, bar)`` under the inner fit's two-level rule (module docstring).

    ``start`` holds the solver's start arguments and is empty for its own start.  The solve
    runs to ``aim`` from ``start``.  One that ends short of it is continued to ``bar`` from
    the state it stopped on, which is as near the mode as that solve could bring it.  A fit
    that is then neither certified nor stagnated at the limit of float64
    (:attr:`ModeFit.usable`) is solved to ``bar`` from the solver's own start, unless it
    began there; a stagnated one is not, since every start ends at that limit.  At most
    three solves, each within the solver's iteration budget; the last one is returned,
    usable or not.
    """
    fit = solve(start, aim)
    if not fit.certified and aim < bar:
        fit = solve(warm_start(fit), bar)
    if not fit.usable and start:
        fit = solve({}, bar)
    return fit


def solve_mode(
    problem: NCVProblem,
    rho: NDArray[np.float64],
    *,
    warm: ModeFit | None = None,
    work: WorkCounts | None = None,
) -> ModeFit:
    """The penalised fit at ``rho`` by :func:`two_level`, from ``warm``'s coefficients when given.

    The first solve runs to the bar ``warm`` carries (:func:`first_bar`).
    """
    lambdas = problem.lambdas(np.asarray(rho, dtype=np.float64))
    counts = WorkCounts() if work is None else work
    before = counts.inner_solves
    fit = two_level(
        lambda start, bar: _solve(problem, lambdas, start, bar, counts),
        {} if warm is None else warm_start(warm),
        first_bar(warm, problem.mode_aim, problem.mode_bar),
        problem.mode_bar,
    )
    counts.fallback_solves += counts.inner_solves - before - 1
    return fit


def _solve(
    problem: NCVProblem,
    lambdas: dict[str, float],
    start: dict[str, Any],
    bar: float,
    work: WorkCounts,
) -> ModeFit:
    cache: dict[str, Any] = {}
    solved = fit_irls_direct(
        X=problem.dm,
        y=problem.y,
        weights=problem.weights,
        family=problem.distribution,
        link=problem.link,
        groups=problem.groups,
        lambda2=lambdas,
        offset=problem.offset,
        reml_penalties=problem.penalties,
        max_iter=problem.max_pirls_iter,
        tol=problem.pirls_tol,
        convergence="mode_score",
        _mode_bar=bar,
        return_xtwx=True,
        cache_out=cache,
        compute_rank_info=False,
        _compensate_centred_intercept=False,
        _retain_reml_decomposition=True,
        separation="record",
        weight_semantics=problem.weight_semantics,
        **start,
    )
    work.inner_solves += 1
    work.pirls_iterations += int(solved[0].n_iter)
    return ModeFit.from_solve(lambdas, solved[0], solved[1], cache, bar)


@dataclass(frozen=True)
class HeldOutPairs:
    """One entry per scored (neighbourhood, row) pair, in neighbourhood order.

    A row scored by several neighbourhoods has an entry for each.  ``predictor`` is the held-out
    linear predictor with its offset, ``mean`` its mean as ``predict`` reports one, ``deviance``
    the weighted unit deviance there (zero on a row of zero weight).
    """

    row: NDArray[np.intp]
    neighbourhood: NDArray[np.intp]
    predictor: NDArray[np.float64]
    mean: NDArray[np.float64]
    deviance: NDArray[np.float64]


@dataclass(frozen=True)
class Evaluation:
    """The criterion at one smoothing.

    ``value`` is the summed held-out deviance, ``gradient`` its derivative in the estimated
    ``rho`` and ``heldout`` its pairs, each when asked for.  An infeasible trial has
    ``feasible`` False, its reason, a NaN value and no gradient or held-out pairs.  ``fit`` is
    the usable mode (:attr:`ModeFit.usable`), ``None`` without one: the state to start the
    next solve from.
    """

    feasible: bool
    infeasible_reason: InfeasibleReason | None
    value: float
    gradient: NDArray[np.float64] | None
    heldout: HeldOutPairs | None
    fit: ModeFit | None
    work: WorkCounts

    @property
    def past_kept_fits(self) -> bool:
        """Whether the trial ended at its limit on kept-rows fits (``KEPT_FITS``), a limit on cost."""
        return self.work.kept_fits > KEPT_FITS


def evaluate(
    problem: NCVProblem,
    nb: Neighbourhoods,
    rho: NDArray[np.float64],
    *,
    warm: ModeFit | None = None,
    gradient: bool = True,
    heldout: bool = True,
    route: Route | None = None,
) -> Evaluation:
    """Solve the mode at ``rho`` and evaluate the criterion there.

    ``heldout=False`` leaves the held-out pairs out of the result, and a Gaussian neighbourhood
    with cached cross-products then reads no rows.
    """
    work = WorkCounts()
    fit = solve_mode(problem, rho, warm=warm, work=work)
    return _evaluate(problem, nb, fit, gradient, heldout, route, work)


def evaluate_fit(
    problem: NCVProblem,
    nb: Neighbourhoods,
    fit: ModeFit,
    *,
    gradient: bool = True,
    heldout: bool = True,
    route: Route | None = None,
) -> Evaluation:
    """Evaluate the criterion at a supplied fit and its geometry; no solve runs."""
    return _evaluate(problem, nb, fit, gradient, heldout, route, WorkCounts())


class _InfeasibleTrial(Exception):  # noqa: N818 - control flow, not an error
    def __init__(self, reason: InfeasibleReason) -> None:
        super().__init__(reason)
        self.reason: InfeasibleReason = reason


@dataclass(frozen=True)
class _Identified:
    """The coordinates a rank-deficient fit identified: ``theta = T phi``, intercept first.

    ``T`` gathers the intercept and the solver's representative slopes (``columns``, with no
    ``basis``), or is ``blockdiag(1, B)`` for its retained slope basis ``B`` (``basis``),
    whose left inverse ``dual'`` reads a slope vector's coordinates.  ``hessian`` and ``inverse`` are
    ``T' H T`` and its inverse on the slopes, as the solver holds them.
    """

    columns: NDArray[np.intp]
    basis: NDArray[np.float64] | None
    dual: NDArray[np.float64] | None
    hessian: NDArray[np.float64]
    inverse: NDArray[np.float64]

    @property
    def width(self) -> int:
        """Identified coefficients with the intercept."""
        return len(self.hessian) + 1

    def rows(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        """``Z T`` for rows of width ``P``; ``T' v`` for one vector."""
        return block[..., self.columns] if self.basis is None else block @ self.basis

    def slopes(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        """``rows`` for rows of slope width ``p``, without the intercept."""
        if self.basis is None:
            return block[..., self.columns[1:] - 1]
        return block @ self.basis[1:, 1:]

    def gram(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """``T' M T`` for a matrix of width ``P``."""
        if self.basis is None:
            return matrix[np.ix_(self.columns, self.columns)]
        return self.basis.T @ matrix @ self.basis

    def coordinates(self, beta: NDArray[np.float64]) -> NDArray[np.float64]:
        """``phi`` on the slopes for coefficients ``beta``, up to the null space."""
        return beta[self.columns[1:] - 1] if self.dual is None else self.dual.T @ beta

    def directions(self, slopes: NDArray[np.float64], width: int) -> NDArray[np.float64]:
        """Columns of slope coordinates as directions in the design's ``width`` slopes."""
        if self.basis is not None:
            return self.basis[1:, 1:] @ slopes
        full = np.zeros((width, slopes.shape[1]))
        full[self.columns[1:] - 1] = slopes
        return full

    def lift(self, vector: NDArray[np.float64], width: int) -> NDArray[np.float64]:
        """``T v`` in the design's ``width`` columns."""
        if self.basis is not None:
            return self.basis @ vector
        full = np.zeros(width)
        full[self.columns] = vector
        return full

    def lift_gram(self, matrix: NDArray[np.float64], width: int) -> NDArray[np.float64]:
        """``T M T'`` in the design's ``width`` columns."""
        if self.basis is not None:
            return self.basis @ matrix @ self.basis.T
        full = np.zeros((width, width))
        full[np.ix_(self.columns, self.columns)] = matrix
        return full


def _coordinates(fit: ModeFit) -> _Identified | None:
    """The coordinates ``fit`` identified where its slope Hessian is rank deficient, else ``None``.

    The solver's pseudo-inverse is zero outside the slopes it kept.  Where those number its
    rank they are its representatives; where there are more it kept a basis, read with its
    retained values from the decomposition the fit carries.
    """
    width = len(fit.slope_inverse)
    rank = fit.result.reml_hessian_rank
    if rank is None or int(rank) == width + 1:
        return None
    kept = np.flatnonzero(np.any(fit.slope_inverse != 0.0, axis=0))
    if len(kept) == int(rank) - 1:
        block = np.ix_(kept, kept)
        columns = np.concatenate(([0], kept + 1))
        return _Identified(columns, None, None, fit.slope_hessian[block], fit.slope_inverse[block])
    retained = fit.result.reml_slope_decomposition
    slopes = None if retained is None else retained.solution_basis
    dual = None if retained is None else retained.estimable_functional_basis
    values = None if retained is None else retained.retained_values
    if slopes is None or dual is None or values is None or slopes.shape[1] != int(rank) - 1:
        raise _InfeasibleTrial("rank_deficient_fit")
    basis = np.zeros((width + 1, int(rank)))
    basis[0, 0] = 1.0
    basis[1:, 1:] = slopes
    none = np.zeros(0, dtype=np.intp)
    return _Identified(none, basis, dual, np.diag(values), np.diag(1.0 / values))


def _evaluate(
    problem: NCVProblem,
    nb: Neighbourhoods,
    fit: ModeFit,
    gradient: bool,
    heldout: bool,
    route: Route | None,
    work: WorkCounts,
) -> Evaluation:
    if nb.n_rows != problem.n:
        raise ValueError(
            f"The neighbourhoods index {nb.n_rows} rows and the design has {problem.n}; build "
            "them for the rows being fitted."
        )
    if not fit.usable:
        return Evaluation(False, "uncertified_mode", float("nan"), None, None, None, work)
    try:
        trial = _Trial.start(problem, nb, fit, gradient and bool(problem.names), heldout, work)
        if nb.drop_rows is None and route is None:
            _small_route(trial, nb, np.flatnonzero(problem.active), stacked=True)
        else:
            _neighbourhoods(trial, nb, route)
    except _InfeasibleTrial as stop:
        return Evaluation(False, stop.reason, float("nan"), None, None, fit, work)
    value = float(np.sum(trial.deviance)) + trial.cached_value
    slopes = None
    if gradient:
        slopes = trial.rho_gradient() if trial.gradient else np.zeros(0)
    if not np.isfinite(value) or (slopes is not None and not np.all(np.isfinite(slopes))):
        return Evaluation(False, "non_finite", float("nan"), None, None, fit, work)
    pairs = None
    if heldout:
        pairs = HeldOutPairs(
            row=nb.scored_rows(),
            neighbourhood=np.repeat(np.arange(nb.m, dtype=np.intp), np.diff(trial.pair_ptr)),
            predictor=trial.predictor,
            mean=trial.mean,
            deviance=trial.deviance,
        )
    work.stagnated_fits += int(fit.stagnated)
    return Evaluation(True, None, value, slopes, pairs, fit, work)


@dataclass
class _Trial:
    """One evaluation in progress: the mode's rows, the per-pair results and the gradient's sums."""

    problem: NCVProblem
    fit: ModeFit
    gradient: bool
    heldout: bool
    work: WorkCounts
    coords: _Identified | None
    width: int
    eta: NDArray[np.float64]
    score: NDArray[np.float64]
    curvature: NDArray[np.float64]
    third: NDArray[np.float64]
    pair_ptr: NDArray[np.intp]
    predictor: NDArray[np.float64]
    mean: NDArray[np.float64]
    deviance: NDArray[np.float64]
    q: NDArray[np.float64]
    cross: NDArray[np.float64]
    cache: _GaussianCache | None = None
    cached_value: float = 0.0
    forms: NDArray[np.float64] | None = None
    _left: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    _right: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    _held: int = 0
    _subset: tuple[NDArray[np.intp], CentredRows] | None = None
    _block: tuple[NDArray[np.intp], NDArray[np.float64]] | None = None

    @classmethod
    def start(
        cls,
        problem: NCVProblem,
        nb: Neighbourhoods,
        fit: ModeFit,
        gradient: bool,
        heldout: bool,
        work: WorkCounts,
    ) -> _Trial:
        """The trial before any neighbourhood: every pair predicted at the full fit, unscored.

        ``width`` is that of the trial's coordinates: ``P``, or the identified ones' of a
        rank-deficient fit, in which every row, solve and sum of the trial is taken.
        """
        coords = _coordinates(fit)
        eta = linear_predictor(problem.dm, fit.result, problem.offset)
        score, curvature, third = problem.loss.derivatives(problem.y, problem.weights, eta)
        scored = np.diff(nb.score_ptr) if nb.score_ptr is not None else nb.sizes()
        pair_ptr = np.concatenate(([0], np.cumsum(scored))).astype(np.intp)
        predictor = eta[nb.scored_rows()]
        width = problem.width if coords is None else coords.width
        return cls(
            problem=problem,
            fit=fit,
            gradient=gradient,
            heldout=heldout,
            work=work,
            coords=coords,
            width=width,
            eta=eta,
            score=score,
            curvature=curvature,
            third=third,
            pair_ptr=pair_ptr,
            predictor=predictor,
            mean=problem.loss.mean(predictor),
            deviance=np.zeros(len(predictor)),
            q=np.zeros(width),
            cross=np.zeros((width, width)),
        )

    @cached_property
    def slope_inverse(self) -> NDArray[np.float64]:
        """``H_c^-1`` in the trial's coordinates."""
        return self.fit.slope_inverse if self.coords is None else self.coords.inverse

    @cached_property
    def hessian(self) -> NDArray[np.float64]:
        """``H = blockdiag(sum W, H_c)``."""
        full = np.zeros((self.width, self.width))
        full[0, 0] = self.fit.sum_w
        full[1:, 1:] = self.fit.slope_hessian if self.coords is None else self.coords.hessian
        return full

    @cached_property
    def jacobi(self) -> NDArray[np.float64]:
        """``diag(H)^(-1/2)``, the scaling of a factored leave-out Hessian."""
        return 1.0 / np.sqrt(np.diag(self.hessian))

    @cached_property
    def questionable(self) -> float:
        """The factorisation margin at or below which a leave-out system is fitted from its kept rows.

        The solver's own cut between a Gram and its row factor, or the worst a downdate's two
        Grams can be off by when that is larger: each within ``gamma_n`` of a unit diagonal an
        entry (Higham 2002, section 3.5), ``2 P gamma_n`` in norm.
        """
        return max(SHARED_RANK_POLICY.factor_rcond, 2.0 * self.width * gamma(self.problem.n))

    @cached_property
    def unresolved(self) -> float:
        """The margin at or below which a neighbourhood-sized system is not solved as one.

        Its margin is built on ``H^-1`` and is known to ``2 P gamma_n ||(D H D)^-1||_1``,
        ``D = diag(H)^(-1/2)`` (module docstring): the error of the Hessian's Gram, twice
        over, times the 1-norm of the inverse of the Jacobi-scaled Hessian.  That norm is
        read from the solver's own inverse in the design's columns, where the intercept's
        entry is one.  A margin is resolved at :data:`RESOLVED` times that error.  Never
        below :attr:`questionable`.
        """
        error = 2.0 * self.width * gamma(self.problem.n) * self.inverse_norm
        return max(self.questionable, RESOLVED * error)

    @cached_property
    def inverse_norm(self) -> float:
        """``||(D H D)^-1||_1``, read from the solver's own inverse in the design's columns."""
        fit = self.fit
        root = np.sqrt(np.maximum(np.diag(fit.slope_hessian), 0.0))
        return float(np.max(root * (np.abs(fit.slope_inverse) @ root), initial=1.0))

    @cached_property
    def factor(self) -> tuple[NDArray[np.float64], float] | None:
        """The Cholesky factor of ``D H D`` and a single row's leverage error through it per ``|w| ||x||^2``.

        The error is ``||E|| (1 + eta)^2 / (1 - eta)`` for ``||E|| = P gamma_(n + 3P + 6)`` and
        ``eta = ||E|| ||(D H D)^-1||_1`` (module docstring, "A single row the inverse does not
        resolve").  ``None`` where ``eta`` exceeds a half or the factorisation fails.
        """
        width = self.width
        error = width * gamma(self.problem.n + 3 * width + 6)
        eta = error * self.inverse_norm
        if not eta <= 0.5:
            return None
        scale = self.jacobi
        try:
            factor = cholesky(
                self.hessian * scale[:, None] * scale[None, :], lower=True, check_finite=False
            )
        except np.linalg.LinAlgError:
            return None
        return factor, error * (1.0 + eta) ** 2 / (1.0 - eta)

    def raw(self, rows: NDArray[np.intp]) -> NDArray[np.float64]:
        """The augmented centred rows of ``rows`` in the design's own columns.

        Taken from the block held for their neighbourhood (:func:`_factor_routes`), which
        lists its rows in ascending order, else read.
        """
        if self._block is not None:
            held, block = self._block
            return block if len(rows) == len(held) else block[np.searchsorted(held, rows)]
        self.work.rows_read += len(rows)
        return augmented_rows(self.problem.dm, rows, self.fit.centre_hi, self.fit.centre_lo)

    def reads_twice(self, nb: Neighbourhoods, k: int) -> bool:
        """Whether at least two passes read form-and-factor neighbourhood ``k``'s rows in blocks.

        Its cross-products, unless the cache holds them; its scores, unless they come from
        cached cross-products; its third-derivative term.  A neighbourhood served by compact
        products is read by none.
        """
        if serves_compact(len(nb.drop(k)), self.problem.width):
            return False
        cache = self.cache
        kept = cache is not None and bool(cache.afforded[k])
        held = cache is not None and k in cache.held
        third = self.gradient and self.problem.loss.has_third_derivative
        return int(not held) + int(not kept or self.heldout) + int(third) >= 2

    def read(self, rows: NDArray[np.intp]) -> NDArray[np.float64]:
        """The augmented centred rows ``Z`` of ``rows``."""
        block = self.raw(rows)
        return block if self.coords is None else self.coords.rows(block)

    @cached_property
    def design(self) -> CentredRows:
        """Every row about the fit's centre pair, in the design's own columns, never dense."""
        return CentredRows(self.problem.dm, self.fit.centre_hi, self.fit.centre_lo)

    def compact(self, rows: NDArray[np.intp]) -> CentredRows | None:
        """``rows`` as a design of their own where compact products are the cheaper route, else ``None``.

        The rule is ``rows.serves_compact``.  The trial holds the one it made last and hands
        it back while the same rows are asked for, so a neighbourhood's cross-products, scores
        and third-derivative term share one; the next neighbourhood's replaces it.
        """
        if not serves_compact(len(rows), self.problem.width):
            return None
        self.work.rows_streamed += len(rows)
        if self._subset is None or not np.array_equal(self._subset[0], rows):
            self._subset = (rows, self.design.subset(rows))
        return self._subset[1]

    def lifted(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """A vector of the trial's coordinates in the design's own columns, ``T v``."""
        return vector if self.coords is None else self.coords.lift(vector, self.problem.width)

    def reduced(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """A vector of the design's own columns in the trial's coordinates, ``T' v``."""
        return vector if self.coords is None else self.coords.rows(vector)

    def within(self, moments: _Moments) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``moments``' Gram and score over the removed rows, in the trial's coordinates."""
        if self.coords is None:
            return moments.gram, moments.score
        return self.coords.gram(moments.gram), self.coords.rows(moments.score)

    def slopes(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rows or a vector of slope width ``p``, in the trial's coordinates."""
        return block if self.coords is None else self.coords.slopes(block)

    def apply_inverse(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """``vectors @ H^-1`` for row vectors of width ``P`` (one vector, or a block of them)."""
        out = np.empty_like(vectors)
        out[..., 0] = vectors[..., 0] / self.fit.sum_w
        out[..., 1:] = vectors[..., 1:] @ self.slope_inverse
        return out

    def solved(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        """``Z H^-1`` for a block of rows."""
        self.work.rows_multiplied += len(block)
        return self.apply_inverse(block)

    def pairs_of(self, k: int) -> slice:
        """Neighbourhood ``k``'s entries among the scored pairs."""
        return slice(int(self.pair_ptr[k]), int(self.pair_ptr[k + 1]))

    def score_pairs(
        self,
        pairs: slice | NDArray[np.intp],
        rows: NDArray[np.intp],
        predictor: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Record the held-out results of ``pairs`` and return ``D'`` there."""
        mean, deviance, slope, usable = self.problem.loss.scored(
            self.problem.y[rows], self.problem.weights[rows], predictor
        )
        if not usable:
            raise _InfeasibleTrial("clipped_prediction")
        self.predictor[pairs] = predictor
        self.mean[pairs] = mean
        self.deviance[pairs] = deviance
        return slope

    def moments(self, rows: NDArray[np.intp], scored: NDArray[np.bool_] | None) -> _Moments:
        """``rows``' cross-products at this fit; ``scored`` marks the scored ones.

        ``None`` for ``scored`` is every row, and the cross-products over the scored rows are
        then those over ``rows``.  Through the rows' own compact design (:meth:`compact`), or
        block by block.
        """
        design = self.compact(rows)
        if design is not None:
            gram, score = design.gram(self.curvature[rows]), design.rmatvec(self.score[rows])
            if scored is None:
                return _Moments(gram, score, gram, score)
            inside, part = design.subset(np.flatnonzero(scored)), rows[scored]
            return _Moments(
                gram, score, inside.gram(self.curvature[part]), inside.rmatvec(self.score[part])
            )
        width = self.problem.width
        gram, score = np.zeros((width, width)), np.zeros(width)
        scored_gram, scored_score = gram, score
        if scored is not None:
            scored_gram, scored_score = np.zeros((width, width)), np.zeros(width)
        for chunk in row_chunks(np.arange(len(rows), dtype=np.intp), width):
            members = rows[chunk]
            block = self.raw(members)
            weighted = self.curvature[members][:, None] * block
            gram += block.T @ weighted
            score += block.T @ self.score[members]
            if scored is not None:
                part = scored[chunk]
                scored_gram += block[part].T @ weighted[part]
                scored_score += block[part].T @ self.score[members][part]
        self.work.rows_multiplied += len(rows) + (0 if scored is None else int(scored.sum()))
        return _Moments(gram, score, scored_gram, scored_score)

    @cached_property
    def shift(self) -> NDArray[np.float64]:
        """``theta - theta_0`` from the cache's reference fit, in the design's own columns.

        The rows are centred about their weighted mean, so the intercept's part is the
        weighted mean of the two fits' difference in predictor.
        """
        cache = self.problem.cache
        level = float(self.problem.weights @ (self.eta - cache.eta)) / self.fit.sum_w
        return np.concatenate(([level], self.fit.result.beta - cache.beta))

    def cross_products(
        self, nb: Neighbourhoods, k: int, rows: NDArray[np.intp], active: NDArray[np.bool_]
    ) -> tuple[_Moments, bool]:
        """Neighbourhood ``k``'s cross-products at this fit, and whether its scored ones are among them.

        Read from the cache where it holds them; otherwise formed from the weighted rows
        ``rows[active]`` and, where the cache affords the neighbourhood, kept at its
        reference fit.
        """
        cache = self.cache
        if cache is None or not cache.afforded[k]:
            return self.moments(rows[active], None), False
        if k in cache.held:
            return cache.held[k].moved(self.shift), True
        pos = nb.scored_positions(k)
        scored = None
        if pos is not None:
            scored = np.zeros(len(rows), dtype=bool)
            scored[pos] = True
            scored = scored[active]
        moments = self.moments(rows[active], scored)
        cache.held[k] = moments.moved(-self.shift)
        return moments, True

    def score_cached(
        self, nb: Neighbourhoods, k: int, moments: _Moments, delta: NDArray[np.float64]
    ) -> NDArray[np.float64] | None:
        """Add neighbourhood ``k``'s held-out sum of squares at the step ``delta``; returns ``Z_d' D'``.

        ``sum_d w (eta - y)^2`` from the fit's residuals, one term per scored row of this
        neighbourhood, and ``2 delta' g_d + delta' G_d delta`` from its cross-products, with
        ``g_d`` moved from the reference fit by ``G_d`` times the shift.  The sum is known to
        ``gamma_m`` times the same terms taken in absolute value, ``m`` the rows and columns
        summed.  Where that exceeds both the inner fit's bar times the sum and
        :data:`RESOLVED` times the ``gamma_m`` of the sum the rows themselves give, the terms
        cancel: nothing is added, None is returned, and the rows are scored directly.
        """
        pos = nb.scored_positions(k)
        rows = nb.drop(k) if pos is None else nb.drop(k)[pos]
        gram, score = moments.scored_gram, moments.scored_score
        lifted = self.lifted(delta)
        step = gram @ lifted
        at_fit = float(self.score[rows] @ (self.eta[rows] - self.problem.y[rows]))
        value = at_fit + float(lifted @ (2.0 * score + step))
        reference = np.abs(self.cache.held[k].scored_score) + np.abs(gram) @ np.abs(self.shift)
        size = at_fit + float(np.abs(lifted) @ (2.0 * reference + np.abs(gram) @ np.abs(lifted)))
        rounding = gamma(len(rows) + 2 * self.problem.width + 2)
        if rounding * size > max(self.problem.mode_bar, RESOLVED * rounding) * value:
            return None
        self.cached_value += value
        return self.reduced(2.0 * (score + step))

    def score_removed(
        self, k: int, rows: NDArray[np.intp], delta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score neighbourhood ``k``'s rows at the step ``delta``; returns ``Z_d' D'``.

        Through the rows' own compact design (:meth:`compact`), or block by block.
        """
        first = int(self.pair_ptr[k])
        design = self.compact(rows)
        if design is not None:
            predictor = self.eta[rows] + design.matvec(self.lifted(delta))
            slope = self.score_pairs(slice(first, first + len(rows)), rows, predictor)
            return self.reduced(design.rmatvec(slope))
        total = np.zeros(self.width)
        for chunk in row_chunks(np.arange(len(rows), dtype=np.intp), self.problem.width):
            members = rows[chunk]
            block = self.read(members)
            pairs = slice(first + int(chunk[0]), first + int(chunk[-1]) + 1)
            total += block.T @ self.score_pairs(pairs, members, self.eta[members] + block @ delta)
        return total

    def third_inside(
        self, rows: NDArray[np.intp], u: NDArray[np.float64], delta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """``sum_i l'''_i z_i (z_i' u) (z_i' delta)`` over ``rows``.

        Through the rows' own compact design (:meth:`compact`), or block by block.
        """
        design = self.compact(rows)
        if design is not None:
            along = design.matvec(self.lifted(u)) * design.matvec(self.lifted(delta))
            return self.reduced(design.rmatvec(self.third[rows] * along))
        total = np.zeros(self.width)
        for chunk in row_chunks(rows, self.problem.width):
            block = self.read(chunk)
            total += block.T @ (self.third[chunk] * (block @ u) * (block @ delta))
        return total

    def penalty(self, lambdas: Mapping[str, float]) -> NDArray[np.float64]:
        """The slope penalty at ``lambdas``, as the solver builds it."""
        problem = self.problem
        return build_penalty_matrix(
            list(problem.dm.group_matrices),
            problem.groups,
            dict(lambdas),
            problem.dm.p,
            reml_penalties=problem.penalties,
        )

    def kept_fit(
        self, nb: Neighbourhoods, k: int, margin: float
    ) -> tuple[NDArray[np.float64], Callable[[NDArray[np.float64]], NDArray[np.float64]]]:
        """Neighbourhood ``k``'s one-step fit from the rows it keeps: its step, and solves with its Hessian.

        The kept rows are centred about their own weighted mean ``m``, where the leave-out
        Hessian is ``blockdiag(sum W, F' F)`` for the solver's factor ``F`` of the centred
        rows and the penalty root.  The trial's rows are those plus ``d = m - c``, so a solve
        with the leave-out Hessian there is that factor's between two shears by ``d``.
        Raises :class:`LeaveOutRankError` or ends the trial where ``F`` is short of rank.
        """
        self.work.kept_fits += 1
        if self.work.kept_fits > KEPT_FITS:
            raise _InfeasibleTrial("leave_out_rank")
        problem, fit = self.problem, self.fit
        weight = self.curvature.copy()
        weight[nb.drop(k)] = 0.0
        total = float(np.sum(weight))
        if not total > 0.0:
            raise LeaveOutRankError(k, nb.describe(k), margin)
        on = weight > 0.0
        linear = self.eta - problem.offset
        working = np.zeros(problem.n)
        working[on] = linear[on] - self.score[on] / weight[on]
        level = float(weight @ working) / total
        _, hi, lo = weighted_mean_pair(problem.dm, weight, total)
        factor, response = grouped_augmented_factor_rhs(
            problem.dm,
            weight,
            self.penalty(fit.lambdas),
            response=working - level,
            center=hi,
            center_lo=lo,
        )
        system = decompose_factor(self.slopes(factor), retain_factor_solve=True)
        if system.rank < self.width - 1:
            if self._left_free(weight, hi, lo):
                raise LeaveOutRankError(k, nb.describe(k), margin)
            raise _InfeasibleTrial("leave_out_rank")
        moved = hi - fit.centre_hi
        if lo is not None:
            moved = moved + lo
        if fit.centre_lo is not None:
            moved = moved - fit.centre_lo
        moved = self.slopes(moved)
        beta = fit.result.beta
        at_fit = beta if self.coords is None else self.coords.coordinates(beta)
        slopes = system.solve_factor_rhs(response)
        full_level = float(self.curvature @ linear) / fit.sum_w
        delta = np.concatenate(([level - moved @ slopes - full_level], slopes - at_fit))

        def solve(rhs: NDArray[np.float64]) -> NDArray[np.float64]:
            self.work.solves += 1
            inner = system.solve(rhs[1:] - moved * rhs[0])
            return np.concatenate(([rhs[0] / total - moved @ inner], inner))

        self.work.solves += 1
        return delta, solve

    def _left_free(
        self, weight: NDArray[np.float64], hi: NDArray[np.float64], lo: NDArray[np.float64] | None
    ) -> bool:
        """Whether the rows of ``weight`` and the penalties leave a direction free at every smoothing.

        The null directions of the rows' own factor, under the solver's rank rule, are taken
        as an orthonormal set ``N`` in the design's slopes.  Each penalty with a positive
        smoothing parameter enters at unit trace, so their root ``R`` has singular values of
        at most one each and holds no smoothing parameter.  A singular value of ``R N`` within
        the rule's cut of zero is a direction neither the rows nor any penalty sees; the rows
        and the penalties are never weighed against each other.
        """
        problem, lambdas = self.problem, self.fit.lambdas
        slopes = problem.dm.p
        data = self.slopes(grouped_weighted_factor(problem.dm, weight, center=hi, center_lo=lo))
        null = decompose_factor(data).null_basis()
        if not null.shape[1]:
            return False
        if self.coords is not None:
            null = self.coords.directions(null, slopes)
        null, _ = np.linalg.qr(null)
        units = [
            self.penalty({other: float(other == name) for other in lambdas})
            for name, value in lambdas.items()
            if value > 0.0
        ]
        at_unit_trace = sum(
            (unit / np.trace(unit) for unit in units if np.trace(unit) > 0.0),
            np.zeros((slopes, slopes)),
        )
        seen = np.linalg.svd(penalty_factor(at_unit_trace) @ null, compute_uv=False)
        cut = SHARED_RANK_POLICY.factor_rcond
        return int(np.count_nonzero(seen > cut)) < null.shape[1]

    def reserve(self, count: int) -> None:
        """Room for the ``(u, Delta)`` pairs of ``count`` neighbourhoods, a bounded block at a time."""
        capacity = min(max(count, 1), chunk_rows(self.problem.width))
        self._left = np.empty((capacity, self.width))
        self._right = np.empty((capacity, self.width))

    def keep_forms(self, pairs: int) -> None:
        """Keep ``z' M z`` row by row as the pairs arrive, where that is the cheapest third pass.

        ``M = sum_k u_k Delta_k'`` has rank at most ``pairs`` and ``z' M z = sum_k (z' u_k)
        (z' Delta_k)``: two products with the design a pair, where a dense block takes ``P`` of
        them a row.  Kept when ``2 pairs <= P`` and the weighted rows are many enough for
        compact products (``rows.serves_compact``), on a design without lookup tables: those
        read the forms of any ``M`` in one pass (:meth:`_third_pass`).
        """
        rows = int(np.count_nonzero(self.problem.active))
        if (
            self.gradient
            and self.problem.loss.has_third_derivative
            and 2 * pairs <= self.problem.width
            and serves_compact(rows, self.problem.width)
            and not self.design.has_tables
        ):
            self.forms = np.zeros(self.problem.n)

    def pair(self, u: NDArray[np.float64], delta: NDArray[np.float64]) -> None:
        """Add ``u Delta'`` to ``M``, and its form to every row's where those are kept."""
        if self.forms is not None:
            design = self.design
            self.forms += design.matvec(self.lifted(u)) * design.matvec(self.lifted(delta))
        if not len(self._left):
            self.reserve(1)
        self._left[self._held] = u
        self._right[self._held] = delta
        self._held += 1
        if self._held == len(self._left):
            self.flush()

    def pairs(self, left: NDArray[np.float64], right: NDArray[np.float64]) -> None:
        """Add ``sum_k u_k Delta_k'`` to ``M`` for a block of pairs, and their forms where those are kept.

        Row ``k`` of ``left`` and of ``right`` need only have ``u_k Delta_k'`` as their outer
        product.
        """
        if self.forms is not None:
            design = self.design
            for u, delta in zip(left, right, strict=True):
                self.forms += design.matvec(self.lifted(u)) * design.matvec(self.lifted(delta))
        self.cross += left.T @ right
        self.work.rows_multiplied += len(left)

    def flush(self) -> None:
        """Add the held pairs' ``sum u Delta'`` to ``M``."""
        held, self._held = self._held, 0
        if held:
            self.cross += self._left[:held].T @ self._right[:held]
            self.work.rows_multiplied += held

    def rho_gradient(self) -> NDArray[np.float64]:
        """``dV / d rho`` from ``q`` and ``M``, after every neighbourhood has added to them."""
        self.flush()
        if self.problem.loss.has_third_derivative:
            self._third_pass()
        z, cross = self.apply_inverse(self.q), self.cross
        if self.coords is not None:
            width = self.problem.width
            z, cross = self.coords.lift(z, width), self.coords.lift_gram(cross, width)
        z, slopes = z[1:], cross[1:, 1:]
        beta = self.fit.result.beta
        return np.array(
            [
                -self.fit.lambdas[name]
                * (
                    float(np.sum(S * slopes[columns, columns]))
                    + float(z[columns] @ S @ beta[columns])
                )
                for name, (columns, S) in zip(self.problem.names, self.problem.blocks, strict=True)
            ]
        )

    def _third_pass(self) -> None:
        """Subtract ``sum_i l'''_i z_i (z_i' M z_i)`` over every weighted row from ``q``.

        The forms are the pairs' own where they were kept (:meth:`keep_forms`); else, where
        the rows are many enough and the design has lookup tables, they are read from those;
        else they are formed block by block.
        """
        active = self.problem.active
        rows = np.flatnonzero(active)
        symmetric = 0.5 * (self.cross + self.cross.T)
        forms = self.forms
        if forms is None and serves_compact(len(rows), self.problem.width):
            lifted = symmetric
            if self.coords is not None:
                lifted = self.coords.lift_gram(symmetric, self.problem.width)
            forms = self.design.quadratic_forms(lifted, active)
        if forms is None:
            for chunk in row_chunks(rows, self.problem.width):
                block = self.read(chunk)
                within = np.einsum("ij,ij->i", block @ symmetric, block)
                self.q -= block.T @ (self.third[chunk] * within)
            self.work.rows_multiplied += len(rows)
            return
        weighted = np.zeros(self.problem.n)
        weighted[rows] = self.third[rows] * forms[rows]
        self.q -= self.reduced(self.design.rmatvec(weighted))
        self.work.rows_streamed += len(rows)


def _leave_out_factor(
    system: NDArray[np.float64], level: float
) -> tuple[tuple[NDArray[np.float64], bool] | None, float]:
    """A leave-out system's Cholesky factor and its margin; no factor where the margin is at or below ``level``.

    The margin is ``1 / ||A^-1||_1`` as ``xPOCON`` estimates it from the factor, on a diagonal
    of at most one, and zero when the factorisation fails.  It only flags:
    :meth:`_Trial.kept_fit` decides what a flagged system is.
    """
    try:
        factor = cho_factor(system, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return None, 0.0
    norm = float(np.linalg.norm(system, 1))
    margin = float(_POCON(factor[0], norm, uplo="L")[0]) * norm
    return (factor if margin > level else None), margin


@dataclass(frozen=True)
class _Stack:
    """The neighbourhoods of one size in a block of rows, one neighbourhood a row of each array.

    ``ks`` are the neighbourhoods; ``rows`` their removed rows and ``slots`` each row's scored
    pair, -1 where it is not scored, both ``(members, size)``.  ``Z`` and ``Y`` are those rows
    and ``Z H^-1``, ``(members, size, P)`` views of the block.  ``B`` is ``Y Z'`` per member
    and ``sound`` marks the members the stack solves together.
    """

    ks: NDArray[np.intp]
    rows: NDArray[np.intp]
    slots: NDArray[np.intp]
    Z: NDArray[np.float64]
    Y: NDArray[np.float64]
    B: NDArray[np.float64]
    sound: NDArray[np.bool_]

    def solved_together(self) -> _Stack:
        """The members marked sound, as a stack of their own."""
        keep = self.sound
        if keep.all():
            return self
        return _Stack(
            self.ks[keep],
            self.rows[keep],
            self.slots[keep],
            self.Z[keep],
            self.Y[keep],
            self.B[keep],
            keep[keep],
        )


def _systems(B: NDArray[np.float64], root: NDArray[np.float64]) -> NDArray[np.float64]:
    """``I - W^(1/2) B W^(1/2)`` for each member of a stack, ``root`` its ``W^(1/2)``.

    A row of zero weight has the identity's row and column, so it is in no system.
    """
    system = -(root[:, :, None] * B * root[:, None, :])
    np.einsum("nii->ni", system)[...] += 1.0
    return system


def _dominant(system: NDArray[np.float64], level: float) -> NDArray[np.bool_]:
    """Members of a stack whose smallest eigenvalue Gershgorin's discs put above ``level``.

    Every disc of such a member, its diagonal entry less the absolute sum of the rest of
    the row, lies right of ``level``.  The member is then strictly diagonally dominant, so
    Gaussian elimination meets no zero pivot and grows no entry beyond twice the largest
    (Wilkinson; Higham 2002, Theorem 9.9).  A member that is not finite is not marked.
    """
    diagonal = np.einsum("nii->ni", system)
    reach = np.abs(system).sum(axis=2) - np.abs(diagonal)
    return np.min(diagonal - reach, axis=1) > level


def _stack(
    trial: _Trial,
    stacked: bool,
    ks: NDArray[np.intp],
    rows: NDArray[np.intp],
    slots: NDArray[np.intp],
    Z: NDArray[np.float64],
    Y: NDArray[np.float64],
) -> _Stack:
    """A stack with ``B`` formed and the members it solves together marked; none unless ``stacked``.

    Single rows: every member whose margin ``1 - W h`` is above the level the inverse
    resolves (:attr:`_Trial.unresolved`), and of the others those the fit's Cholesky factor
    resolves (:func:`_through_factor`), whose leverage in ``B`` and solved row in ``Y`` are
    then the factor's: written into the block's solved rows, which nothing else reads.
    Larger systems: the members Gershgorin's discs put above ``sqrt(size)`` times that
    level, whose margin as :func:`_leave_out_factor` estimates it is then above the level
    (module docstring): taken alone they would be factored and solved on this route.
    """
    size = rows.shape[1]
    curvature = trial.curvature[rows]
    if size == 1:
        B = np.einsum("ij,ij->i", Y[:, 0], Z[:, 0])[:, None, None]
        sound = 1.0 - curvature[:, 0] * B[:, 0, 0] > trial.unresolved
        pending = np.flatnonzero(~sound & stacked)
        through = _through_factor(trial, rows[pending, 0], Z[pending, 0]) if len(pending) else None
        if through is not None:
            B[pending, 0, 0], Y[pending, 0], sound[pending] = through
    else:
        B = np.matmul(Y, Z.transpose(0, 2, 1))
        sound = _dominant(_systems(B, np.sqrt(curvature)), np.sqrt(size) * trial.unresolved)
    return _Stack(ks, rows, slots, Z, Y, B, sound & stacked)


def _through_factor(
    trial: _Trial, rows: NDArray[np.intp], block: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]] | None:
    """Single rows by the full fit's Cholesky factor: their leverages, solved rows and which it resolves.

    ``rows`` and ``block`` are the rows and ``Z``; the solved rows are ``Z H^-1``.  A row is
    resolved where its margin ``1 - w h`` exceeds :data:`RESOLVED` times the error of its own
    leverage (module docstring, "A single row the inverse does not resolve").  ``None`` where
    the trial has no factor (:attr:`_Trial.factor`).
    """
    held = trial.factor
    if held is None:
        return None
    factor, error = held
    scale = trial.jacobi
    forward = solve_triangular(
        factor, (block * scale).T, lower=True, overwrite_b=True, check_finite=False
    )
    leverage = np.einsum("ij,ij->j", forward, forward)
    back = solve_triangular(
        factor, forward, lower=True, trans="T", overwrite_b=True, check_finite=False
    )
    curvature = trial.curvature[rows]
    known = error * np.einsum("ij,ij->j", back, back) + gamma(trial.width + 2) * leverage
    resolved = 1.0 - curvature * leverage > RESOLVED * np.abs(curvature) * known
    trial.work.rows_multiplied += len(rows)
    trial.work.resolved_by_factor += int(np.count_nonzero(resolved))
    return leverage, back.T * scale, resolved


def _alone(trial: _Trial, nb: Neighbourhoods, stack: _Stack, at: int, stacked: bool) -> None:
    """Member ``at`` of ``stack``, which the stack does not solve.

    A single row goes on with its margin ``1 - W h``, the fit's Cholesky factor's where the
    trial has one (:func:`_stack`), to :func:`_factor_route`; any other member, and every
    member unless ``stacked``, is decided by its own factorisation (:func:`_woodbury`).
    """
    k = int(stack.ks[at])
    if not stacked or stack.rows.shape[1] > 1:
        _woodbury(trial, nb, k, stack.rows[at], stack.Z[at], stack.Y[at])
        return
    margin = 1.0 - trial.curvature[stack.rows[at, 0]] * stack.B[at, 0, 0]
    _factor_route(trial, nb, k, flagged=float(margin))


def _single_rows(trial: _Trial, stack: _Stack) -> None:
    """A stack of single rows by the scalar form ``1 - W h``, with no factorisation."""
    rows, block, solved = stack.rows[:, 0], stack.Z[:, 0], stack.Y[:, 0]
    leverage = stack.B[:, 0, 0]
    curvature = trial.curvature[rows]
    denominator = 1.0 - curvature * leverage
    a = trial.score[rows] / denominator
    step = a * leverage
    scored = stack.slots[:, 0] >= 0
    slope = np.zeros(len(rows))
    slope[scored] = trial.score_pairs(
        stack.slots[scored, 0], rows[scored], (trial.eta[rows] + step)[scored]
    )
    if not trial.gradient:
        return
    b = slope / denominator
    xu = b * leverage
    trial.q += block.T @ (slope + curvature * xu + trial.third[rows] * xu * step)
    trial.pairs(solved, (a * b)[:, None] * solved)


def _stacked(trial: _Trial, stack: _Stack) -> None:
    """A stack of neighbourhood-sized systems, each right-hand side by one ``numpy.linalg.solve``.

    The arithmetic of :func:`_woodbury` with one row of each array a member, all of them
    shown sound beforehand (:func:`_stack`).  A removed row of zero weight sits in its
    member's system as a row of the identity with a zero score: its part of every sum is an
    exact zero, and its prediction is the full fit plus the step.
    """
    rows, Z, Y, B = stack.rows, stack.Z, stack.Y, stack.B
    curvature = trial.curvature[rows]
    root = np.sqrt(curvature)
    system = _systems(B, root)

    def lift(residual: NDArray[np.float64]) -> NDArray[np.float64]:
        """``m`` with ``H_a^-1 Z' residual = Y' m`` per member, for residuals on its removed rows."""
        inner = root * np.einsum("nij,nj->ni", B, residual)
        return residual + root * np.linalg.solve(system, inner[..., None])[..., 0]

    m = lift(trial.score[rows])
    step = np.einsum("nji,nj->ni", B, m)
    scored = stack.slots >= 0
    slope = np.zeros(rows.shape)
    slope[scored] = trial.score_pairs(
        stack.slots[scored], rows[scored], (trial.eta[rows] + step)[scored]
    )
    trial.work.stacks += 1
    trial.work.stacked += len(rows)
    if not trial.gradient:
        return
    adjoint = lift(slope)
    xu = np.einsum("nji,nj->ni", B, adjoint)
    weight = slope + curvature * xu + trial.third[rows] * xu * step
    trial.q += weight.reshape(-1) @ Z.reshape(rows.size, -1)
    trial.pairs(np.einsum("ni,nip->np", adjoint, Y), np.einsum("ni,nip->np", m, Y))


def _neighbourhoods(trial: _Trial, nb: Neighbourhoods, route: Route | None) -> None:
    """Every neighbourhood by one of the two routes: the cost rule's, or the one forced.

    A Gaussian problem's cache is attached here, for neighbourhoods that list their rows.
    """
    problem, width = trial.problem, trial.width
    active = problem.active
    if nb.drop_rows is None:
        weighted = active.astype(np.intp)
    else:
        weighted = np.bincount(nb.owners()[active[nb.drop_rows]], minlength=nb.m)
        if problem.loss.quadratic:
            problem.cache.attach(nb, trial.fit, trial.eta, weighted, problem.width)
            trial.cache = problem.cache
    size = np.minimum(nb.sizes(), width).astype(np.int64)
    small = 3 * size * size * width + size**3 < width**3
    if route is not None:
        small = np.full(nb.m, route == "woodbury")
    formed = np.flatnonzero(~small & (weighted > 0))
    trial.reserve(len(formed) if route is None else int(np.count_nonzero(weighted)))
    trial.keep_forms(int(np.count_nonzero(weighted)))
    _small_route(trial, nb, np.flatnonzero(small & (weighted > 0)), stacked=route is None)
    _factor_routes(trial, nb, formed)


def _batches(sizes: NDArray[np.intp], limit: int) -> Iterator[tuple[int, int]]:
    """Consecutive runs ``[first, last)`` of items, each run within ``limit`` plus one item in total size."""
    if not len(sizes):
        return iter(())
    ends = np.cumsum(sizes)
    marks = np.arange(limit, int(ends[-1]) + limit, limit)
    cuts = np.unique(
        np.concatenate(([0], np.searchsorted(ends, marks, side="right"), [len(sizes)]))
    )
    return zip(cuts[:-1].tolist(), cuts[1:].tolist(), strict=True)


def _runs(sizes: list[int], limit: int) -> Iterator[tuple[int, int]]:
    """Consecutive runs ``[first, last)`` of items, each of at most ``limit`` in total size or of one item."""
    first = total = 0
    for index, size in enumerate(sizes):
        if index > first and total + size > limit:
            yield first, index
            first = index
            total = 0
        total += size
    if sizes:
        yield first, len(sizes)


def _factor_routes(trial: _Trial, nb: Neighbourhoods, ks: NDArray[np.intp]) -> None:
    """Neighbourhoods ``ks`` by forming their leave-out Hessians, in order.

    A neighbourhood that fits in one block and is read by at least two of its passes
    (:meth:`_Trial.reads_twice`) is read once: its rows come in a block shared with the
    neighbourhoods beside it, of at most ``rows.chunk_rows`` rows in all, and every pass
    takes them from it.  Any other reads its rows pass by pass.
    """
    limit = chunk_rows(trial.problem.width)
    members = ks.tolist()
    shared = [
        len(nb.drop(k)) if len(nb.drop(k)) <= limit and trial.reads_twice(nb, k) else 0
        for k in members
    ]
    for first, last in _runs(shared, limit):
        sizes = shared[first:last]
        rows = [nb.drop(k) for k, size in zip(members[first:last], sizes, strict=True) if size]
        block = trial.raw(np.concatenate(rows)) if rows else None
        stops = np.cumsum(sizes).tolist()
        for k, size, stop in zip(members[first:last], sizes, stops, strict=True):
            if block is not None and size:
                trial._block = (nb.drop(k), block[stop - size : stop])
            _factor_route(trial, nb, k)
            trial._block = None


def _ranges(
    ptr: NDArray[np.intp], ks: NDArray[np.intp]
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Every position of the ranges ``ptr[k]:ptr[k + 1]`` of ``ks``, range by range.

    With them, each position's index in ``ks`` and where each range starts among the positions.
    """
    sizes = ptr[ks + 1] - ptr[ks]
    first = np.cumsum(sizes) - sizes
    owner = np.repeat(np.arange(len(ks), dtype=np.intp), sizes)
    return ptr[ks][owner] + np.arange(len(owner), dtype=np.intp) - first[owner], owner, first


def _removed(
    nb: Neighbourhoods, ks: NDArray[np.intp]
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """The rows neighbourhoods ``ks`` remove, neighbourhood by neighbourhood.

    Each row, its scored pair (-1 where the neighbourhood does not score it) and its
    neighbourhood's index in ``ks``.
    """
    if nb.drop_ptr is None or nb.drop_rows is None:
        return ks, ks, np.arange(len(ks), dtype=np.intp)
    entries, owner, first = _ranges(nb.drop_ptr, ks)
    if nb.score_ptr is None or nb.score_pos is None:
        return nb.drop_rows[entries], entries, owner
    pairs, scorer, _ = _ranges(nb.score_ptr, ks)
    slots = np.full(len(entries), -1, dtype=np.intp)
    slots[first[scorer] + nb.score_pos[pairs]] = pairs
    return nb.drop_rows[entries], slots, owner


def _small_route(trial: _Trial, nb: Neighbourhoods, ks: NDArray[np.intp], stacked: bool) -> None:
    """Neighbourhoods ``ks`` by the neighbourhood-sized system, in bounded blocks of whole neighbourhoods.

    A block's rows are read sorted by the size of their neighbourhood, so the neighbourhoods
    of one size are a stack (:class:`_Stack`): views of the block and of its solved rows.
    A stack solves together the members it can show sound (:func:`_stack`).  Every other
    member is decided one at a time (:func:`_alone`), in the order of ``ks`` and before the
    block's stacks are solved.  ``stacked`` False is a forced route: every neighbourhood
    one at a time.
    """
    sizes = nb.sizes()[ks]
    for first, last in _batches(sizes, chunk_rows(trial.problem.width)):
        members, size = ks[first:last], sizes[first:last]
        rows, slots, owner = _removed(nb, members)
        order = np.argsort(size[owner], kind="stable")
        rows, slots = rows[order], slots[order]
        block = trial.read(rows)
        solved = trial.solved(block)
        stacks, stop = [], 0
        for removed in np.unique(size).tolist():
            inside = np.flatnonzero(size == removed)
            start, stop = stop, stop + removed * len(inside)
            shape = (len(inside), removed)
            stack = _stack(
                trial,
                stacked,
                members[inside],
                rows[start:stop].reshape(shape),
                slots[start:stop].reshape(shape),
                block[start:stop].reshape(*shape, -1),
                solved[start:stop].reshape(*shape, -1),
            )
            stacks.append(stack)
        alone = [(stack, at) for stack in stacks for at in np.flatnonzero(~stack.sound).tolist()]
        for stack, at in sorted(alone, key=lambda member: member[0].ks[member[1]]):
            _alone(trial, nb, stack, at, stacked)
        for stack in stacks:
            if stack.sound.any():
                solve = _single_rows if stack.rows.shape[1] == 1 else _stacked
                solve(trial, stack.solved_together())


def _woodbury(
    trial: _Trial,
    nb: Neighbourhoods,
    k: int,
    rows: NDArray[np.intp],
    block: NDArray[np.float64],
    solved: NDArray[np.float64],
) -> None:
    """One neighbourhood by the system of its weighted removed rows.

    One whose margin the inverse does not resolve goes on with it (:func:`_factor_route`),
    unless it has one weighted row and the fit's Cholesky factor resolves that row's margin,
    which then takes the factor's leverage and solved row (:func:`_through_factor`).
    """
    on = trial.problem.active[rows]
    every = bool(on.all())
    members = rows if every else rows[on]
    Z, Y = (block, solved) if every else (block[on], solved[on])
    root = np.sqrt(trial.curvature[members])
    B = Y @ Z.T
    factor, margin = _leave_out_factor(
        np.eye(len(members)) - root[:, None] * B * root[None, :], trial.unresolved
    )
    through = _through_factor(trial, members, Z) if factor is None and len(members) == 1 else None
    if through is not None:
        leverage, Y, resolved = through
        B = leverage[:, None]
        margin = float(1.0 - trial.curvature[members[0]] * leverage[0])
        if resolved[0]:
            factor = (np.sqrt([[margin]]), True)
    if factor is None:
        _factor_route(trial, nb, k, flagged=margin)
        return
    trial.work.small_factorisations += 1

    def lift(residual: NDArray[np.float64]) -> NDArray[np.float64]:
        """``H_a^-1 Z' residual`` for a residual on the weighted removed rows."""
        trial.work.solves += 1
        inner = cho_solve(factor, root * (B @ residual), check_finite=False)
        return Y.T @ (residual + root * inner)

    delta = lift(trial.score[members])
    pos = nb.scored_positions(k)
    scored, scored_block = (rows, block) if pos is None else (rows[pos], block[pos])
    slope = trial.score_pairs(trial.pairs_of(k), scored, trial.eta[scored] + scored_block @ delta)
    if not trial.gradient:
        return
    spread = np.zeros(len(rows))
    spread[slice(None) if pos is None else pos] = slope
    residual = spread if every else spread[on]
    u = lift(residual)
    xu = Z @ u
    inside = trial.third[members] * xu * (Z @ delta)
    trial.q += Z.T @ (residual + trial.curvature[members] * xu + inside)
    trial.pair(u, delta)


def _factor_route(trial: _Trial, nb: Neighbourhoods, k: int, flagged: float | None = None) -> None:
    """One neighbourhood by forming its leave-out Hessian.

    The Hessian is factored as ``H - Z_a' W_a Z_a``.  Where that factorisation's margin is
    questionable, the neighbourhood is fitted from the rows it keeps.  ``flagged`` is the
    margin of a neighbourhood-sized system that route did not solve: questionable, the
    neighbourhood is fitted from its kept rows with no factorisation here; above that, the
    inverse did not resolve it and this factorisation decides.  A Gaussian neighbourhood
    with cached cross-products is scored from them unless the held-out pairs are wanted.
    """
    rows = nb.drop(k)
    active = trial.problem.active[rows]
    members = rows[active]
    moments, cached = trial.cross_products(nb, k, rows, active)
    gram, score = trial.within(moments)
    solve = None
    if flagged is None or flagged > trial.questionable:
        scale = trial.jacobi
        factor, flagged = _leave_out_factor(
            (trial.hessian - gram) * scale[:, None] * scale[None, :], trial.questionable
        )
        if factor is not None:
            trial.work.factorisations += 1
            solve = _scaled_solver(trial.work, factor, scale)
    if solve is None:
        delta, solve = trial.kept_fit(nb, k, flagged)
    else:
        delta = solve(score)
    r = trial.score_cached(nb, k, moments, delta) if cached and not trial.heldout else None
    if r is None:
        pos = nb.scored_positions(k)
        r = trial.score_removed(k, rows if pos is None else rows[pos], delta)
    if not trial.gradient:
        return
    u = solve(r)
    trial.q += r + gram @ u
    if trial.problem.loss.has_third_derivative:
        trial.q += trial.third_inside(members, u, delta)
    trial.pair(u, delta)


def _scaled_solver(
    work: WorkCounts, factor: tuple[NDArray[np.float64], bool], scale: NDArray[np.float64]
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Solves with a leave-out Hessian from the factor of its Jacobi-scaled form."""

    def solve(rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        work.solves += 1
        return scale * cho_solve(factor, scale * rhs, check_finite=False)

    return solve


__all__ = [
    "DEFAULT_NCV_TOL",
    "Evaluation",
    "HeldOutPairs",
    "LeaveOutRankError",
    "ModeFit",
    "NCVProblem",
    "WorkCounts",
    "build_problem",
    "evaluate",
    "evaluate_fit",
    "first_bar",
    "ncv_mode_aim",
    "ncv_mode_bar",
    "solve_mode",
    "two_level",
    "warm_start",
]
