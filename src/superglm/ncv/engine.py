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
``3 |a|^2 P + |a|^3`` and the second ``P^3`` (``P = p + 1``), which is the dispatch rule.  Single
rows take the scalar form ``1 - W_i h_i`` in blocks of rows, with no factorisation.

Rows of zero weight are in no system and no sum; a scored one is predicted at the full fit plus
the step.  A trial counts only at a certified inner mode, and with every weighted scored row on
the scale where its mean is the unclipped inverse link.

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
``rows.chunk_rows(P)``; a form-and-factor neighbourhood is streamed through such blocks, and a
neighbourhood on the neighbourhood-sized route is held whole (under the cost rule it has fewer
than ``0.54 P`` weighted rows).  An evaluation holds a block, its two products and two blocks of
``(u, Delta)`` pairs, a handful of ``P x P`` matrices, and row-length vectors: the mode's rows
and five entries per scored pair.  Between evaluations only a Gaussian problem keeps anything:
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
from scipy.linalg import cho_factor, cho_solve, get_lapack_funcs

from superglm._roundoff import EPS, gamma
from superglm.distributions import Distribution
from superglm.group_matrix import DesignMatrix
from superglm.links import Link
from superglm.ncv.neighbourhoods import Neighbourhoods
from superglm.ncv.rows import (
    RowLoss,
    augmented_rows,
    cache_allowance,
    chunk_rows,
    row_chunks,
    row_loss,
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
"""A leave-out solve route: the neighbourhood-sized system, or form and factor."""

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

_POCON = get_lapack_funcs("pocon", dtype=np.float64)


def ncv_mode_bar(ncv_tol: float = DEFAULT_NCV_TOL) -> float:
    """The inner fit's certificate bar for a search stopping at ``ncv_tol``: a tenth of it.

    The criterion is not stationary in the coefficients, so a mode error enters it and its
    gradient at first order.  Never below ``100 eps``, the least the certificate expresses.
    """
    return max(0.1 * float(ncv_tol), 100.0 * EPS)


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
    third-derivative pass.  Divide either by the row count for passes over the data.
    ``kept_fits`` are leave-out systems fitted from their kept rows, each one or two passes of
    the solver's row factor over the design, which neither row count includes.
    """

    inner_solves: int = 0
    pirls_iterations: int = 0
    factorisations: int = 0
    small_factorisations: int = 0
    solves: int = 0
    rows_read: int = 0
    rows_multiplied: int = 0
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
    held at a value, which every inner fit keeps.  ``cache`` is this problem's own: no
    constructor argument sets it, so a problem built or copied anew starts with nothing held.
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
    pirls_tol: float = 1e-8,
    max_pirls_iter: int = 100,
) -> NCVProblem:
    """The engine's problem for a built design.

    ``penalties`` are the design's components (``penalty_algebra.build_penalty_context``).  A
    component named in ``fixed_lambdas`` is held at that value; every other one is estimated.
    """
    held = dict(fixed_lambdas or {})
    estimated = [component for component in penalties if component.name not in held]
    weights = np.asarray(weights, dtype=np.float64)
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
        mode_bar=ncv_mode_bar() if mode_bar is None else float(mode_bar),
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
    """

    lambdas: dict[str, float]
    result: PIRLSResult
    slope_inverse: NDArray[np.float64]
    slope_hessian: NDArray[np.float64]
    sum_w: float
    centre_hi: NDArray[np.float64]
    centre_lo: NDArray[np.float64] | None
    data_gram: NDArray[np.float64] | None = None

    @classmethod
    def from_solve(
        cls,
        lambdas: Mapping[str, float],
        result: PIRLSResult,
        slope_inverse: NDArray[np.float64],
        cache: Mapping[str, Any],
    ) -> ModeFit:
        """From what ``fit_irls_direct(..., lambda2=lambdas, return_xtwx=True, cache_out=cache)`` hands back.

        Solve with ``_retain_reml_decomposition=True``: a rank-deficient fit whose solver kept a
        basis and not representative columns is evaluated in that basis, read from ``result``.
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
        )

    @property
    def certified(self) -> bool:
        """Whether the solve ended on its mode certificate."""
        return bool(self.result.converged)


def solve_mode(
    problem: NCVProblem,
    rho: NDArray[np.float64],
    *,
    warm: ModeFit | None = None,
    work: WorkCounts | None = None,
) -> ModeFit:
    """The penalised fit at ``rho``, started from ``warm``'s coefficients when given.

    A warm-started solve that ends without its certificate is solved once more from the
    solver's own start, within the same iteration budget.
    """
    lambdas = problem.lambdas(np.asarray(rho, dtype=np.float64))
    counts = WorkCounts() if work is None else work
    fit = _solve(problem, lambdas, warm, counts)
    if warm is not None and not fit.certified:
        fit = _solve(problem, lambdas, None, counts)
    return fit


def _solve(
    problem: NCVProblem, lambdas: dict[str, float], warm: ModeFit | None, work: WorkCounts
) -> ModeFit:
    start: dict[str, Any] = {}
    if warm is not None:
        start = {
            "beta_init": warm.result.beta,
            "intercept_init": float(warm.result.intercept),
            "_centred_init": centred_warm_start(warm.result),
        }
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
        _mode_bar=problem.mode_bar,
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
    return ModeFit.from_solve(lambdas, solved[0], solved[1], cache)


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
    the certified mode, ``None`` without one: the state to start the next solve from.
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
    if not fit.certified:
        return Evaluation(False, "uncertified_mode", float("nan"), None, None, None, work)
    try:
        trial = _Trial.start(problem, nb, fit, gradient and bool(problem.names), heldout, work)
        if nb.drop_rows is None and route is None:
            _single_rows(trial, nb)
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
    _left: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    _right: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    _held: int = 0

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

    def raw(self, rows: NDArray[np.intp]) -> NDArray[np.float64]:
        """The augmented centred rows of ``rows`` in the design's own columns."""
        self.work.rows_read += len(rows)
        return augmented_rows(self.problem.dm, rows, self.fit.centre_hi, self.fit.centre_lo)

    def read(self, rows: NDArray[np.intp]) -> NDArray[np.float64]:
        """The augmented centred rows ``Z`` of ``rows``."""
        block = self.raw(rows)
        return block if self.coords is None else self.coords.rows(block)

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
        """``rows``' cross-products at this fit, block by block; ``scored`` marks the scored ones.

        ``None`` for ``scored`` is every row, and the cross-products over the scored rows are
        then those over ``rows``.
        """
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
    ) -> NDArray[np.float64]:
        """Add neighbourhood ``k``'s held-out sum of squares at the step ``delta``; returns ``Z_d' D'``.

        ``sum_d w (eta - y)^2`` from the fit's residuals, one term per scored row of this
        neighbourhood, and ``2 delta' g_d + delta' G_d delta`` from its cross-products.
        """
        pos = nb.scored_positions(k)
        rows = nb.drop(k) if pos is None else nb.drop(k)[pos]
        gram, score = moments.scored_gram, moments.scored_score
        if self.coords is not None:
            gram, score = self.coords.gram(gram), self.coords.rows(score)
        step = gram @ delta
        at_fit = float(self.score[rows] @ (self.eta[rows] - self.problem.y[rows]))
        self.cached_value += at_fit + float(delta @ (2.0 * score + step))
        return 2.0 * (score + step)

    def score_removed(
        self, k: int, rows: NDArray[np.intp], delta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score neighbourhood ``k``'s rows at the step ``delta``, block by block; returns ``Z_d' D'``."""
        first = int(self.pair_ptr[k])
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
        """``sum_i l'''_i z_i (z_i' u) (z_i' delta)`` over ``rows``, block by block."""
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

    def pair(self, u: NDArray[np.float64], delta: NDArray[np.float64]) -> None:
        """Add ``u Delta'`` to ``M``."""
        if not len(self._left):
            self.reserve(1)
        self._left[self._held] = u
        self._right[self._held] = delta
        self._held += 1
        if self._held == len(self._left):
            self.flush()

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
        """Subtract ``sum_i l'''_i z_i (z_i' M z_i)`` over every weighted row from ``q``."""
        symmetric = 0.5 * (self.cross + self.cross.T)
        rows = np.flatnonzero(self.problem.active)
        for chunk in row_chunks(rows, self.problem.width):
            block = self.read(chunk)
            forms = np.einsum("ij,ij->i", block @ symmetric, block)
            self.q -= block.T @ (self.third[chunk] * forms)
        self.work.rows_multiplied += len(rows)


def _leave_out_factor(
    system: NDArray[np.float64], questionable: float
) -> tuple[tuple[NDArray[np.float64], bool] | None, float]:
    """A leave-out system's Cholesky factor and its margin; no factor where the margin is questionable.

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
    return (factor if margin > questionable else None), margin


def _single_rows(trial: _Trial, nb: Neighbourhoods) -> None:
    """Every weighted row removed and scored on its own, in blocks of rows.

    A row whose ``1 - W h`` is questionable is taken out of its block and fitted from the
    rows it leaves (:func:`_factor_route`).
    """
    problem = trial.problem
    for chunk in row_chunks(np.flatnonzero(problem.active), problem.width):
        block = trial.read(chunk)
        solved = trial.solved(block)
        leverage = np.einsum("ij,ij->i", solved, block)
        curvature = trial.curvature[chunk]
        denominator = 1.0 - curvature * leverage
        sound = denominator > trial.questionable
        if not sound.all():
            for at in np.flatnonzero(~sound).tolist():
                _factor_route(trial, nb, int(chunk[at]), flagged=float(denominator[at]))
            chunk, block, solved = chunk[sound], block[sound], solved[sound]
            leverage, curvature, denominator = leverage[sound], curvature[sound], denominator[sound]
        a = trial.score[chunk] / denominator
        step = a * leverage
        slope = trial.score_pairs(chunk, chunk, trial.eta[chunk] + step)
        if not trial.gradient:
            continue
        b = slope / denominator
        xu = b * leverage
        trial.q += block.T @ (slope + curvature * xu + trial.third[chunk] * xu * step)
        trial.cross += solved.T @ ((a * b)[:, None] * solved)
        trial.work.rows_multiplied += len(chunk)


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
    size = np.minimum(weighted, width).astype(np.int64)
    small = 3 * size * size * width + size**3 < width**3
    if route is not None:
        small = np.full(nb.m, route == "woodbury")
    trial.reserve(int(np.count_nonzero(weighted)))
    _woodbury_route(trial, nb, np.flatnonzero(small & (weighted > 0)))
    for k in np.flatnonzero(~small & (weighted > 0)).tolist():
        _factor_route(trial, nb, k)


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


def _woodbury_route(trial: _Trial, nb: Neighbourhoods, ks: NDArray[np.intp]) -> None:
    """Neighbourhoods ``ks`` by the neighbourhood-sized system, their rows read in bounded batches."""
    sizes = nb.sizes()[ks]
    for first, last in _batches(sizes, chunk_rows(trial.problem.width)):
        members = ks[first:last].tolist()
        rows = np.concatenate([nb.drop(k) for k in members])
        block = trial.read(rows)
        solved = trial.solved(block)
        stops = np.cumsum(sizes[first:last]).tolist()
        for k, start, stop in zip(members, [0, *stops[:-1]], stops, strict=True):
            _woodbury(trial, nb, k, rows[start:stop], block[start:stop], solved[start:stop])


def _woodbury(
    trial: _Trial,
    nb: Neighbourhoods,
    k: int,
    rows: NDArray[np.intp],
    block: NDArray[np.float64],
    solved: NDArray[np.float64],
) -> None:
    """One neighbourhood by the system of its weighted removed rows.

    One whose system is questionable is fitted from the rows it keeps (:func:`_factor_route`).
    """
    on = trial.problem.active[rows]
    every = bool(on.all())
    members = rows if every else rows[on]
    Z, Y = (block, solved) if every else (block[on], solved[on])
    root = np.sqrt(trial.curvature[members])
    B = Y @ Z.T
    factor, margin = _leave_out_factor(
        np.eye(len(members)) - root[:, None] * B * root[None, :], trial.questionable
    )
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
    questionable, or another route's was (``flagged``, its margin), the neighbourhood is
    fitted from the rows it keeps.  A Gaussian neighbourhood with cached cross-products is
    scored from them unless the held-out pairs are wanted.
    """
    rows = nb.drop(k)
    active = trial.problem.active[rows]
    members = rows[active]
    moments, cached = trial.cross_products(nb, k, rows, active)
    gram, score = trial.within(moments)
    solve = None
    if flagged is None:
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
    if cached and not trial.heldout:
        r = trial.score_cached(nb, k, moments, delta)
    else:
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
    "ncv_mode_bar",
    "solve_mode",
]
