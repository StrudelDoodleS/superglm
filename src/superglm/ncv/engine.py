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

**Memory.**  No ``rows x P`` array of the whole design is formed.  Rows are read in blocks of
``rows.chunk_rows(P)``; a form-and-factor neighbourhood is streamed through such blocks, and a
neighbourhood on the neighbourhood-sized route is held whole (under the cost rule it has fewer
than ``0.54 P`` weighted rows).  An evaluation holds a block, its two products and two blocks of
``(u, Delta)`` pairs, a handful of ``P x P`` matrices, and row-length vectors: the mode's rows
and five entries per scored pair.  Nothing is kept between evaluations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from superglm._roundoff import EPS
from superglm.distributions import Distribution
from superglm.group_matrix import DesignMatrix
from superglm.links import Link
from superglm.ncv.neighbourhoods import Neighbourhoods
from superglm.ncv.rows import RowLoss, augmented_rows, chunk_rows, row_chunks, row_loss
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.mode_score import centred_warm_start, linear_predictor
from superglm.solvers.pirls import PIRLSResult
from superglm.types import GroupSlice, PenaltyComponent

Route = Literal["woodbury", "factor"]
"""A leave-out solve route: the neighbourhood-sized system, or form and factor."""

InfeasibleReason = Literal["uncertified_mode", "clipped_prediction", "non_finite"]

DEFAULT_NCV_TOL = 1e-6


def ncv_mode_bar(ncv_tol: float = DEFAULT_NCV_TOL) -> float:
    """The inner fit's certificate bar for a search stopping at ``ncv_tol``: a tenth of it.

    The criterion is not stationary in the coefficients, so a mode error enters it and its
    gradient at first order.  Never below ``100 eps``, the least the certificate expresses.
    """
    return max(0.1 * float(ncv_tol), 100.0 * EPS)


class LeaveOutRankError(ValueError):
    """The rows left when a neighbourhood is removed do not identify the model.

    ``neighbourhood`` is its index; ``margin`` the factorisation margin that failed.
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
    """

    inner_solves: int = 0
    pirls_iterations: int = 0
    factorisations: int = 0
    small_factorisations: int = 0
    solves: int = 0
    rows_read: int = 0
    rows_multiplied: int = 0


@dataclass(frozen=True)
class NCVProblem:
    """What the engine reads of a model: its design, family, rows and penalties.

    ``names`` are the estimated penalty components, in the order of ``rho``; ``blocks`` holds
    each one's columns and unit penalty matrix on them.  ``fixed_lambdas`` are the components
    held at a value, which every inner fit keeps.
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
    """

    lambdas: dict[str, float]
    result: PIRLSResult
    slope_inverse: NDArray[np.float64]
    slope_hessian: NDArray[np.float64]
    sum_w: float
    centre_hi: NDArray[np.float64]
    centre_lo: NDArray[np.float64] | None

    @classmethod
    def from_solve(
        cls,
        lambdas: Mapping[str, float],
        result: PIRLSResult,
        slope_inverse: NDArray[np.float64],
        cache: Mapping[str, Any],
    ) -> ModeFit:
        """From what ``fit_irls_direct(..., lambda2=lambdas, return_xtwx=True, cache_out=cache)`` hands back."""
        centre_hi, centre_lo = cache["centre_pair"]
        return cls(
            lambdas=dict(lambdas),
            result=result,
            slope_inverse=slope_inverse,
            slope_hessian=cache["centered_hessian"],
            sum_w=float(cache["sum_W"]),
            centre_hi=centre_hi,
            centre_lo=centre_lo,
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

    ``value`` is the summed held-out deviance and ``gradient`` its derivative in the estimated
    ``rho``, when asked for.  An infeasible trial has ``feasible`` False, its reason, a NaN
    value and no gradient or held-out pairs.  ``fit`` is the certified mode, ``None`` without
    one: the state to start the next solve from.
    """

    feasible: bool
    infeasible_reason: InfeasibleReason | None
    value: float
    gradient: NDArray[np.float64] | None
    heldout: HeldOutPairs | None
    fit: ModeFit | None
    work: WorkCounts


def evaluate(
    problem: NCVProblem,
    nb: Neighbourhoods,
    rho: NDArray[np.float64],
    *,
    warm: ModeFit | None = None,
    gradient: bool = True,
    route: Route | None = None,
) -> Evaluation:
    """Solve the mode at ``rho`` and evaluate the criterion there."""
    work = WorkCounts()
    fit = solve_mode(problem, rho, warm=warm, work=work)
    return _evaluate(problem, nb, fit, gradient, route, work)


def evaluate_fit(
    problem: NCVProblem,
    nb: Neighbourhoods,
    fit: ModeFit,
    *,
    gradient: bool = True,
    route: Route | None = None,
) -> Evaluation:
    """Evaluate the criterion at a supplied fit and its geometry; no solve runs."""
    return _evaluate(problem, nb, fit, gradient, route, WorkCounts())


class _InfeasibleTrial(Exception):  # noqa: N818 - control flow, not an error
    def __init__(self, reason: InfeasibleReason) -> None:
        super().__init__(reason)
        self.reason: InfeasibleReason = reason


def _evaluate(
    problem: NCVProblem,
    nb: Neighbourhoods,
    fit: ModeFit,
    gradient: bool,
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
    trial = _Trial.start(problem, nb, fit, gradient and bool(problem.names), work)
    try:
        if nb.drop_rows is None and route is None:
            _single_rows(trial, nb)
        else:
            _neighbourhoods(trial, nb, route)
    except _InfeasibleTrial as stop:
        return Evaluation(False, stop.reason, float("nan"), None, None, fit, work)
    value = float(np.sum(trial.deviance))
    slopes = None
    if gradient:
        slopes = trial.rho_gradient() if trial.gradient else np.zeros(0)
    if not np.isfinite(value) or (slopes is not None and not np.all(np.isfinite(slopes))):
        return Evaluation(False, "non_finite", float("nan"), None, None, fit, work)
    heldout = HeldOutPairs(
        row=nb.scored_rows(),
        neighbourhood=np.repeat(np.arange(nb.m, dtype=np.intp), np.diff(trial.pair_ptr)),
        predictor=trial.predictor,
        mean=trial.mean,
        deviance=trial.deviance,
    )
    return Evaluation(True, None, value, slopes, heldout, fit, work)


@dataclass
class _Trial:
    """One evaluation in progress: the mode's rows, the per-pair results and the gradient's sums."""

    problem: NCVProblem
    fit: ModeFit
    gradient: bool
    work: WorkCounts
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
    _left: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    _right: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    _held: int = 0

    @classmethod
    def start(
        cls, problem: NCVProblem, nb: Neighbourhoods, fit: ModeFit, gradient: bool, work: WorkCounts
    ) -> _Trial:
        """The trial before any neighbourhood: every pair predicted at the full fit, unscored."""
        eta = linear_predictor(problem.dm, fit.result, problem.offset)
        score, curvature, third = problem.loss.derivatives(problem.y, problem.weights, eta)
        scored = np.diff(nb.score_ptr) if nb.score_ptr is not None else nb.sizes()
        pair_ptr = np.concatenate(([0], np.cumsum(scored))).astype(np.intp)
        predictor = eta[nb.scored_rows()]
        width = problem.width
        return cls(
            problem=problem,
            fit=fit,
            gradient=gradient,
            work=work,
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

    @property
    def width(self) -> int:
        return self.problem.width

    @cached_property
    def hessian(self) -> NDArray[np.float64]:
        """``H = blockdiag(sum W, H_c)``."""
        full = np.zeros((self.width, self.width))
        full[0, 0] = self.fit.sum_w
        full[1:, 1:] = self.fit.slope_hessian
        return full

    @cached_property
    def jacobi(self) -> NDArray[np.float64]:
        """``diag(H)^(-1/2)``, the scaling of a factored leave-out Hessian."""
        return 1.0 / np.sqrt(np.diag(self.hessian))

    def read(self, rows: NDArray[np.intp]) -> NDArray[np.float64]:
        """The augmented centred rows ``Z`` of ``rows``."""
        self.work.rows_read += len(rows)
        return augmented_rows(self.problem.dm, rows, self.fit.centre_hi, self.fit.centre_lo)

    def apply_inverse(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """``vectors @ H^-1`` for row vectors of width ``P`` (one vector, or a block of them)."""
        out = np.empty_like(vectors)
        out[..., 0] = vectors[..., 0] / self.fit.sum_w
        out[..., 1:] = vectors[..., 1:] @ self.fit.slope_inverse
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

    def moments(self, rows: NDArray[np.intp]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``Z' W Z`` and ``Z' l'`` over ``rows``, block by block."""
        gram = np.zeros((self.width, self.width))
        score = np.zeros(self.width)
        for chunk in row_chunks(rows, self.width):
            block = self.read(chunk)
            gram += block.T @ (self.curvature[chunk][:, None] * block)
            score += block.T @ self.score[chunk]
        self.work.rows_multiplied += len(rows)
        return gram, score

    def score_removed(
        self, k: int, rows: NDArray[np.intp], delta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Score neighbourhood ``k``'s rows at the step ``delta``, block by block; returns ``Z_d' D'``."""
        first = int(self.pair_ptr[k])
        total = np.zeros(self.width)
        for chunk in row_chunks(np.arange(len(rows), dtype=np.intp), self.width):
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
        for chunk in row_chunks(rows, self.width):
            block = self.read(chunk)
            total += block.T @ (self.third[chunk] * (block @ u) * (block @ delta))
        return total

    def reserve(self, count: int) -> None:
        """Room for the ``(u, Delta)`` pairs of ``count`` neighbourhoods, a bounded block at a time."""
        capacity = min(max(count, 1), chunk_rows(self.width))
        self._left = np.empty((capacity, self.width))
        self._right = np.empty((capacity, self.width))

    def pair(self, u: NDArray[np.float64], delta: NDArray[np.float64]) -> None:
        """Add ``u Delta'`` to ``M``."""
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
        z = self.apply_inverse(self.q)[1:]
        beta = self.fit.result.beta
        slopes = self.cross[1:, 1:]
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
        for chunk in row_chunks(rows, self.width):
            block = self.read(chunk)
            forms = np.einsum("ij,ij->i", block @ symmetric, block)
            self.q -= block.T @ (self.third[chunk] * forms)
        self.work.rows_multiplied += len(rows)


def _leave_out_verdict(nb: Neighbourhoods, k: int, margin: float, width: int) -> None:
    """Refuse leave-out system ``k`` when its factorisation margin cannot be told from zero.

    ``margin`` is the smallest squared pivot of the system's Cholesky factor, on a diagonal of
    at most one: ``1 - W h`` for a single row, zero when the factorisation failed.  The one
    place the engine decides what a questionable leave-out system means.
    """
    if not margin > width * EPS:
        raise LeaveOutRankError(k, nb.describe(k), margin)


def _leave_out_factor(
    nb: Neighbourhoods, k: int, system: NDArray[np.float64], width: int
) -> tuple[NDArray[np.float64], bool]:
    """The Cholesky factor of leave-out system ``k``, once :func:`_leave_out_verdict` passes it."""
    try:
        factor = cho_factor(system, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        _leave_out_verdict(nb, k, 0.0, width)
        raise
    _leave_out_verdict(nb, k, float(np.min(np.diag(factor[0]))) ** 2, width)
    return factor


def _single_rows(trial: _Trial, nb: Neighbourhoods) -> None:
    """Every weighted row removed and scored on its own, in blocks of rows."""
    problem, width = trial.problem, trial.width
    for chunk in row_chunks(np.flatnonzero(problem.active), width):
        block = trial.read(chunk)
        solved = trial.solved(block)
        leverage = np.einsum("ij,ij->i", solved, block)
        curvature = trial.curvature[chunk]
        denominator = 1.0 - curvature * leverage
        worst = int(np.argmin(denominator))
        _leave_out_verdict(nb, int(chunk[worst]), float(denominator[worst]), width)
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
    """Every neighbourhood by one of the two routes: the cost rule's, or the one forced."""
    active, width = trial.problem.active, trial.width
    if nb.drop_rows is None:
        weighted = active.astype(np.intp)
    else:
        weighted = np.bincount(nb.owners()[active[nb.drop_rows]], minlength=nb.m)
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
    for first, last in _batches(sizes, chunk_rows(trial.width)):
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
    """One neighbourhood by the system of its weighted removed rows."""
    on = trial.problem.active[rows]
    every = bool(on.all())
    members = rows if every else rows[on]
    Z, Y = (block, solved) if every else (block[on], solved[on])
    root = np.sqrt(trial.curvature[members])
    B = Y @ Z.T
    factor = _leave_out_factor(
        nb, k, np.eye(len(members)) - root[:, None] * B * root[None, :], trial.width
    )
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


def _factor_route(trial: _Trial, nb: Neighbourhoods, k: int) -> None:
    """One neighbourhood by forming and factoring its leave-out Hessian."""
    rows = nb.drop(k)
    members = rows[trial.problem.active[rows]]
    gram, score = trial.moments(members)
    scale = trial.jacobi
    factor = _leave_out_factor(
        nb, k, (trial.hessian - gram) * scale[:, None] * scale[None, :], trial.width
    )
    trial.work.factorisations += 1
    solve = _scaled_solver(trial.work, factor, scale)
    delta = solve(score)
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
