"""Immutable IRLS snapshots and atomic trial-step selection."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import Binomial, Distribution, Poisson, clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import _BINOMIAL_CLIP_MU_EPS, Link, LogLink, SqrtLink, stabilize_eta

_MAX_FLOAT64_HALVING_DEPTH = 1074


@dataclass(frozen=True)
class SolverState:
    """A complete, immutable evaluated coefficient-state snapshot."""

    beta: NDArray
    intercept: float
    eta_unclipped: NDArray
    eta: NDArray
    mu: NDArray
    deviance: float
    penalized_deviance: float | None = None
    state_id: int | None = None
    evaluation_id: int | None = None
    state_space: str = "solver"
    basis_id: int | None = None
    lambdas: tuple[tuple[str, object], ...] = ()
    dispersion: float | None = None
    # the centred intercept alpha of a solver that keeps its state centred
    # (one-engine design §3.8); ``intercept`` is then its raw reading
    centred_intercept: float | None = None


# Migration alias retained while direct IRLS and downstream private callers
# move onto the shared state identity contract.
_IRLSState = SolverState


@dataclass(frozen=True)
class _IRLSStepDecision:
    """The accepted fraction of a proposal, or an atomic rejection."""

    alpha: float
    step_halvings: int
    step_rejected: bool
    trials_attempted: int = 1
    # halvings refused unevaluated, outside the mean space (``mean_space_first_halving``)
    skipped: int = 0


def _immutable_array(values: NDArray) -> NDArray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _freeze_owned_array(values: NDArray) -> NDArray:
    """Freeze an array produced inside state evaluation without copying it."""
    result = np.asarray(values, dtype=float)
    result.setflags(write=False)
    return result


def _evaluate_irls_state(
    dm: DesignMatrix,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    offset: NDArray,
    beta: NDArray,
    intercept: float,
    *,
    deviance: float | None = None,
    eta_unclipped: NDArray | None = None,
    penalized_deviance: float | None = None,
    state_id: int | None = None,
    evaluation_id: int | None = None,
    state_space: str = "solver",
    basis_id: int | None = None,
    lambdas: tuple[tuple[str, object], ...] = (),
    dispersion: float | None = None,
) -> _IRLSState:
    """Evaluate and freeze all state derived from one coefficient vector."""
    retained_beta = _immutable_array(beta)
    if eta_unclipped is None:
        eta_unclipped = dm.matvec(retained_beta) + intercept + offset
    else:
        eta_unclipped = np.asarray(eta_unclipped, dtype=float)
        if eta_unclipped.shape != (dm.n,):
            raise ValueError(f"eta_unclipped must have shape {(dm.n,)}, got {eta_unclipped.shape}")
    eta = stabilize_eta(eta_unclipped, link)
    mu = clip_mu(link.inverse(eta), family)
    retained_deviance = (
        float(np.sum(weights * family.deviance_unit(y, mu)))
        if deviance is None
        else float(deviance)
    )
    eta_unclipped = _freeze_owned_array(eta_unclipped)
    eta = _freeze_owned_array(eta)
    mu = _freeze_owned_array(mu)
    return _IRLSState(
        beta=retained_beta,
        intercept=float(intercept),
        eta_unclipped=eta_unclipped,
        eta=eta,
        mu=mu,
        deviance=retained_deviance,
        penalized_deviance=(None if penalized_deviance is None else float(penalized_deviance)),
        state_id=state_id,
        evaluation_id=evaluation_id,
        state_space=state_space,
        basis_id=basis_id,
        lambdas=lambdas,
        dispersion=dispersion,
    )


def mean_space_violation(
    family: Distribution, link: Link
) -> Callable[[NDArray, NDArray], bool] | None:
    """The test that a linear predictor leaves the family's mean space, or ``None``.

    Declared by the family and link alone: ``None`` when every ``eta`` the
    link accepts maps into the family's means.  The log link's inverse
    ``exp(eta)`` covers ``(0, inf)``, which is wider than the binomial
    probability space ``(0, 1)``: a positive-weight row with ``eta >= 0`` has
    no binomial likelihood (``log(1 - mu)`` is undefined), so it is not a
    state at all.  ``clip_mu`` would hide it behind a mean of ``1 - 1e-7``,
    whose deviance is finite and flat in ``eta`` and whose Fisher weight
    ``exp(2 eta) / V(mu)`` pins the row where it landed: replacing an
    out-of-range fitted value instead of the coefficients is Wacholder's
    (1986) device, which can return estimates whose fitted probabilities are
    invalid.  The feasible set ``{beta : X beta + offset < 0}`` is convex and
    the log-likelihood concave, so a line search from a feasible state finds
    a feasible step by halving, as R's ``glm`` and ``glm2`` do (Donoghoe &
    Marschner 2018, J. Stat. Softw. 86(9), sections 2 and 3.2).
    """
    if isinstance(family, Binomial) and isinstance(link, LogLink):

        def violates(eta_unclipped: NDArray, weights: NDArray) -> bool:
            return bool(np.any((eta_unclipped >= 0.0) & (weights > 0.0)))

        return violates
    return None


# ``clip_mu`` caps a binomial mean at ``1 - _BINOMIAL_CLIP_MU_EPS``: under the
# log link that cap is reached at ``eta = log1p(-eps)``.
_BINOMIAL_LOG_CAP_ETA = math.log1p(-_BINOMIAL_CLIP_MU_EPS)


def mean_space_boundary_rows(
    family: Distribution, link: Link, eta_unclipped: NDArray, weights: NDArray
) -> int:
    """Positive-weight rows whose mean ``clip_mu`` caps at the mean-space boundary.

    Zero for a family and link whose means cannot leave the mean space
    (``mean_space_violation`` is ``None``).  A binomial/log row with
    ``exp(eta) >= 1 - 1e-7`` has its mean replaced by the cap, so its deviance
    is flat in ``eta`` there: the fitted objective is no longer the binomial
    likelihood, and a state holding such a row is at the boundary of the
    parameter space ``{beta : X beta + offset < 0}``, where the maximum is
    not a stationary point (Donoghoe & Marschner 2018, J. Stat. Softw.
    86(9), sections 2 and 4.1).
    """
    if mean_space_violation(family, link) is None:
        return 0
    return int(np.count_nonzero((eta_unclipped >= _BINOMIAL_LOG_CAP_ETA) & (weights > 0.0)))


def mean_space_clipped_rows(
    family: Distribution, link: Link, eta_unclipped: NDArray, weights: NDArray
) -> int:
    """Positive-weight rows whose mean ``clip_mu`` replaces, for a family with a mean space.

    Zero for a family and link whose means cannot leave the mean space
    (``mean_space_violation`` is ``None``).  ``clip_mu`` holds a binomial mean
    inside ``[1e-7, 1 - 1e-7]``.  A row whose mean leaves that band, event or
    non-event, carries a deviance flat in ``eta`` and a score that is not the
    binomial/log score, so a stop rule read off the clipped objective there
    says nothing about the model's likelihood (``mean_space_score_rows``).
    """
    if mean_space_violation(family, link) is None:
        return 0
    with np.errstate(under="ignore"):
        mean = link.inverse(stabilize_eta(np.asarray(eta_unclipped, dtype=float), link))
    return int(np.count_nonzero((clip_mu(mean, family) != mean) & (weights > 0.0)))


def mean_space_score_rows(
    y: NDArray, weights: NDArray, eta_unclipped: NDArray
) -> tuple[NDArray, NDArray]:
    """The binomial/log row score and Fisher weight in ``eta``, from the unclipped ``eta``.

    With ``mu = exp(eta)`` and ``1 - mu = -expm1(eta)``, exact near ``eta =
    0``, the row log-likelihood ``w [y log mu + (1 - y) log(1 - mu)]`` has
    derivative ``s = w [y - (1 - y) mu / (1 - mu)]`` and Fisher weight ``w mu
    / (1 - mu)`` (``w mu'^2 / V(mu)``).  The score is the difference of two
    non-negative terms, so it cancels only where it vanishes; the textbook
    ``w (y - mu) / (1 - mu)`` loses ``y - mu`` as ``mu -> 1``.  ``clip_mu``'s
    band does not enter.  Below ``eta ~ -745`` ``exp`` underflows to the
    exact limits ``s = w y`` and weight 0.  A positive-weight row outside the
    space (``eta >= 0``) has no likelihood and gives a non-finite score; a
    zero-weight row gives 0.
    """
    eta = np.asarray(eta_unclipped, dtype=np.float64)
    response = np.asarray(y, dtype=np.float64)
    prior = np.asarray(weights, dtype=np.float64)
    with np.errstate(under="ignore", over="ignore", divide="ignore", invalid="ignore"):
        odds = np.exp(eta) / -np.expm1(eta)
        odds = np.where(eta < 0.0, odds, np.inf)
        score = prior * response - prior * (1.0 - response) * odds
        fisher = prior * odds
    # a zero-weight row carries nothing, wherever its eta is
    carried = prior > 0.0
    return np.where(carried, score, 0.0), np.where(carried, fisher, 0.0)


def mean_space_newton_rows(
    y: NDArray, weights: NDArray, eta_unclipped: NDArray
) -> tuple[NDArray, NDArray] | None:
    """Newton rows for binomial/log: ``(observed curvature, score)`` in ``eta``, or ``None``.

    The row log-likelihood is concave in ``eta``, with observed information
    ``-d2l/deta2 = w (1 - y) mu / (1 - mu)^2``: never negative, and zero on
    an event row, whose log-likelihood ``w eta`` is linear.  Fisher scoring
    weights that row by ``w mu / (1 - mu)`` instead, which grows without
    bound as ``mu -> 1``; near the boundary the two informations part, and
    scoring converges linearly at a rate set by their mismatch (Osborne
    1992, Int. Stat. Rev. 60; Green 1984, JRSSB 46, sections 1.2 and 2.3).
    Newton's method on the observed information converges quadratically
    there.  The step is taken in score form, ``(X' W X + S) delta = X' u - S
    beta`` with these ``W`` and the score ``u`` of ``mean_space_score_rows``,
    so an event row's zero curvature never divides its score.  ``None`` when
    a row is not finite or no row carries curvature (every positive-weight
    row an event: the boundary supremum), and the caller keeps its rows.
    """
    eta = np.asarray(eta_unclipped, dtype=np.float64)
    prior = np.asarray(weights, dtype=np.float64)
    score, _ = mean_space_score_rows(y, weights, eta)
    with np.errstate(under="ignore", over="ignore", divide="ignore", invalid="ignore"):
        complement = -np.expm1(eta)
        curvature = prior * (1.0 - np.asarray(y, dtype=np.float64)) * (np.exp(eta) / complement)
        curvature = curvature / complement
    curvature = np.where(prior > 0.0, curvature, 0.0)
    total = float(np.sum(curvature))
    finite = bool(np.all(np.isfinite(curvature)) and np.all(np.isfinite(score)))
    if not (finite and math.isfinite(total) and total > 0.0 and np.all(curvature >= 0.0)):
        return None
    return curvature, score


def mean_space_log_likelihood_rows(y: NDArray, weights: NDArray, eta_unclipped: NDArray) -> NDArray:
    """Each row's binomial/log log-likelihood ``w [y eta + (1 - y) log(1 - e^eta)]`` at the unclipped eta.

    The model's own likelihood, up to the saturated term: ``clip_mu``'s band
    does not enter, so a row below its floor still counts.  ``log(1 -
    e^eta)`` is Maechler's (2012) ``log1mexp``, ``log(-expm1(eta))`` above
    ``-log 2`` and ``log1p(-exp(eta))`` below, each branch evaluated on its
    own non-event rows only.  Zero on a zero-weight row.
    """
    eta = np.asarray(eta_unclipped, dtype=np.float64)
    response = np.asarray(y, dtype=np.float64)
    prior = np.asarray(weights, dtype=np.float64)
    rows = np.zeros_like(eta)
    carried = prior > 0.0
    near = carried & (response < 1.0) & (eta > -math.log(2.0))
    far = carried & (response < 1.0) & ~near
    complement = np.zeros_like(eta)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        complement[near] = np.log(-np.expm1(eta[near]))
        complement[far] = np.log1p(-np.exp(eta[far]))
        rows[carried] = prior[carried] * (
            response[carried] * eta[carried] + (1.0 - response[carried]) * complement[carried]
        )
    return rows


def interior_start_intercept(
    family: Distribution,
    link: Link,
    eta_unclipped: NDArray,
    weights: NDArray,
    intercept: float,
    level: float,
) -> float:
    """An intercept that puts a starting state inside the mean space.

    ``intercept`` unchanged when the family and link cannot leave their mean
    space or every positive-weight row starts inside it (``eta < 0`` for
    binomial/log, ``mean_space_violation``).  Otherwise it is lowered until
    the largest positive-weight ``eta`` is ``level``, the intercept-only
    start's own ``log(mean) < 0``.  Every log-binomial solver but the
    augmented-Lagrangian ones needs a start inside the parameter space, and
    with an intercept the simple one is ``(a, 0, ..., 0)`` with ``a < 0``
    (Schwendinger, Gruen & Hornik 2021, Comput. Stat. 36, section 4.2); an
    offset moves that bound to ``a < -max(offset)``, which the default
    intercept, chosen before the offset, ignores.  A constant offset ``c``
    then starts at ``intercept - c``, the no-offset start shifted with the
    optimum.  A start already inside is left alone, so a fit without an
    offset starts exactly as before.
    """
    violates = mean_space_violation(family, link)
    eta = np.asarray(eta_unclipped, dtype=float)
    if violates is None or not violates(eta, weights):
        return float(intercept)
    top = float(np.max(eta[weights > 0.0]))
    if not math.isfinite(top):
        return float(intercept)
    return float(intercept) - (top - float(level))


StateInvalid = Callable[[_IRLSState], bool]
MeritDelta = Callable[[_IRLSState, _IRLSState], float]
MeritRoundoff = Callable[[_IRLSState, _IRLSState], float]


def _stable_penalized_deviance_delta(
    candidate: _IRLSState,
    committed: _IRLSState,
    penalty_matvec: Callable[[NDArray], NDArray] | NDArray | None = None,
    nonsmooth_penalty: Callable[[NDArray], float] | None = None,
    deviance_delta: float | None = None,
) -> float:
    """Compare penalized deviances without subtracting two large quadratics.

    In an ill-conditioned smooth basis, the two penalty quadratics can each be
    accurately evaluated while their tiny difference loses enough digits to
    reverse the sign of an otherwise safe terminal step.  The polarization
    identity evaluates that difference directly from the coefficient update.

    ``penalty_matvec`` supplies the quadratic penalty ``S`` (a matrix or a
    matvec); pass ``None`` when the fit carries no quadratic penalty. **It
    must apply a symmetric operator.** The polarization identity used here
    evaluates ``(b1 - b0)' S (b1 + b0)``, which equals the difference of the
    two merit quadratics ``b' S b`` only when ``S = S'``; for an asymmetric
    ``S`` the antisymmetric part cancels out of ``b' S b`` but not out of the
    polarized form, so the merit and its delta would disagree. Every in-tree
    penalty is symmetric by construction; the constraint is stated because
    ``S_override`` accepts an arbitrary ``(p, p)`` array on a shape check
    alone.
    ``nonsmooth_penalty`` supplies any non-quadratic penalty term as a
    function of ``beta``, already scaled to match the caller's merit
    convention; its two evaluations enter the same ``math.fsum``.
    ``deviance_delta``, when given, replaces the two states' deviances with
    an already-formed difference of the data term.
    """
    terms = (
        [float(candidate.deviance), -float(committed.deviance)]
        if deviance_delta is None
        else [float(deviance_delta)]
    )

    if penalty_matvec is not None:
        delta_beta = candidate.beta - committed.beta
        summed_beta = candidate.beta + committed.beta
        penalty_direction = (
            penalty_matvec(summed_beta)
            if callable(penalty_matvec)
            else np.asarray(penalty_matvec, dtype=np.float64) @ summed_beta
        )
        terms.append(
            math.fsum(
                float(delta_value * direction_value)
                for delta_value, direction_value in zip(
                    delta_beta,
                    penalty_direction,
                    strict=True,
                )
            )
        )

    if nonsmooth_penalty is not None:
        terms.append(float(nonsmooth_penalty(candidate.beta)))
        terms.append(-float(nonsmooth_penalty(committed.beta)))

    return math.fsum(terms)


def _state_is_finite(state: _IRLSState) -> bool:
    return bool(
        np.isfinite(state.intercept)
        and np.all(np.isfinite(state.beta))
        and np.all(np.isfinite(state.eta_unclipped))
        and np.all(np.isfinite(state.eta))
        and np.all(np.isfinite(state.mu))
        and np.isfinite(state.deviance)
        and (state.penalized_deviance is None or np.isfinite(state.penalized_deviance))
    )


def _state_merit(state: _IRLSState) -> float:
    """Return the fitted objective used to judge an IRLS trial."""
    if state.penalized_deviance is not None:
        return float(state.penalized_deviance)
    return float(state.deviance)


def _irls_objective_scale(
    *,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
) -> float:
    """Return the natural additive scale for an IRLS objective."""
    if type(family) is not Poisson or type(link) is not SqrtLink:
        return 1.0

    with np.errstate(over="ignore", invalid="ignore"):
        response_mass = float(np.sum(weights * y, dtype=np.float64))
    if not np.isfinite(response_mass):
        return 1.0
    return max(response_mass, np.finfo(np.float64).tiny)


def _irls_objective_relative_change(
    *,
    objective: float,
    previous: float,
    objective_scale: float,
) -> float:
    """Return a convergence ratio with link-appropriate objective units."""
    # Poisson deviance is homogeneous of degree one when y and μ are rescaled
    # together.  A fixed ``+1`` denominator therefore declares convergence
    # prematurely for genuinely tiny sqrt-link means.  Response mass carries
    # the same units and makes this stopping rule scale-equivariant.  Callers
    # compute this fit-invariant scale once and reuse it for every iteration.
    reference = max(abs(objective), abs(previous), objective_scale)
    if reference == 0.0:
        return 0.0
    change = abs(objective - previous)
    change = (
        change / reference
        if np.isfinite(change)
        else abs(objective / reference - previous / reference)
    )
    scale = abs(previous) / reference + objective_scale / reference
    if scale == 0.0:
        return 0.0 if change == 0.0 else float("inf")
    return change / scale


def _poisson_sqrt_halving_budget(
    *,
    committed: _IRLSState,
    proposal: _IRLSState,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    default: int,
) -> int:
    """Scale a binary backtracking budget for an exact Poisson/sqrt step.

    For a proposal direction ``d_eta``, let

        R = max_i |d_eta_i| / sqrt(y_i)

    over positive-weight, positive-response rows.  ``R`` is dimensionless and
    invariant when the response is rescaled by ``c`` and eta by ``sqrt(c)``.
    Adding ``ceil(log2(R))`` to the ordinary budget gives the search the same
    number of halvings *after* the proposal displacement reaches its natural
    response scale.  This changes only how far backtracking may look; it does
    not alter the Fisher direction, score, or fitted objective.
    """
    if default < 1:
        raise ValueError("default halving budget must be at least 1")
    if type(family) is not Poisson or type(link) is not SqrtLink:
        return default

    y_values = np.asarray(y, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    active = (weight_values > 0.0) & (y_values > 0.0)
    if not np.any(active):
        return default

    eta_step = np.abs(proposal.eta_unclipped[active] - committed.eta_unclipped[active])
    moving = eta_step > 0.0
    if not np.any(moving):
        return default

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log2_ratio = np.log2(eta_step[moving]) - 0.5 * np.log2(y_values[active][moving])
    # A non-finite endpoint direction cannot be recovered by multiplying it by
    # a positive alpha: IEEE ``alpha * inf`` remains infinite.  Retain the
    # ordinary bounded rejection path instead of attempting 1074 futile trials.
    if np.any(np.isposinf(log2_ratio)):
        return default
    finite = log2_ratio[np.isfinite(log2_ratio)]
    if finite.size == 0:
        return default

    extra_depth = max(0, int(np.ceil(float(np.max(finite)))))
    return min(default + extra_depth, _MAX_FLOAT64_HALVING_DEPTH)


def _irls_trial_is_unsafe(
    candidate: _IRLSState,
    committed: _IRLSState,
    invalid_state: StateInvalid | None = None,
    merit_delta: MeritDelta | None = None,
    merit_scale: float = 1.0,
    merit_roundoff: MeritRoundoff | None = None,
) -> bool:
    """Reject invalid states or a material increase in the fitted objective."""
    if not _state_is_finite(candidate):
        return True
    if invalid_state is not None and invalid_state(candidate):
        return True
    if (candidate.penalized_deviance is None) != (committed.penalized_deviance is None):
        return True

    candidate_merit = _state_merit(candidate)
    committed_merit = _state_merit(committed)
    if not np.isfinite(committed_merit):
        return False
    if merit_roundoff is not None:
        roundoff = float(merit_roundoff(candidate, committed))
        if not np.isfinite(roundoff) or roundoff < 0.0:
            return True
    else:
        roundoff = (
            64.0
            * np.finfo(float).eps
            * max(merit_scale, abs(candidate_merit), abs(committed_merit))
        )
    if merit_delta is not None:
        delta = float(merit_delta(candidate, committed))
        return not np.isfinite(delta) or bool(delta > roundoff)
    return bool(candidate_merit > committed_merit + roundoff)


def _mean_space_halving_budget(
    *,
    committed: _IRLSState,
    proposal: _IRLSState,
    weights: NDArray,
    family: Distribution,
    link: Link,
    default: int,
) -> int:
    """Extend backtracking from a state inside the mean space until a trial is back inside.

    ``default`` unless the family and link can leave their mean space
    (``mean_space_violation``), the committed state is inside it and the
    proposal is not.  The binomial/log space ``{eta < 0}`` is open and
    convex, so from a committed ``eta`` strictly inside it the trial at
    fraction ``alpha`` of the proposal step ``d`` stays inside exactly when
    ``alpha < t = min(-eta / d)`` over positive-weight rows with ``d > 0``:
    the fraction to the boundary of interior-point methods (Nocedal & Wright
    2006, Numerical Optimization, chapter 19).  A feasible trial always
    exists, but when ``t <= 2^-default`` every budgeted halving leaves the
    space and the step is rejected.  The budget then reaches the first
    halving below ``t``, depth ``floor(log2(1 / t)) + 1``, plus the ordinary
    budget for the objective test from there, within float64's halving
    depth.  ``_select_irls_trial`` reads it only after the ordinary budget
    has failed, and ``fit_irls_direct`` asks for it only in a fit whose start
    ``interior_start_intercept`` lowered, so every other fit is unchanged.
    """
    violates = mean_space_violation(family, link)
    if violates is None or violates(committed.eta_unclipped, weights):
        return default
    eta = np.asarray(committed.eta_unclipped, dtype=np.float64)
    step = np.asarray(proposal.eta_unclipped, dtype=np.float64) - eta
    rising = (weights > 0.0) & (step > 0.0)
    if not np.any(rising):
        return default
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        fraction = float(np.min(-eta[rising] / step[rising]))
    if not (math.isfinite(fraction) and 0.0 < fraction <= 2.0**-default):
        return default
    depth = math.floor(-math.log2(fraction)) + 1
    return min(depth + default, _MAX_FLOAT64_HALVING_DEPTH)


def mean_space_first_halving(
    *,
    committed: _IRLSState,
    proposal: _IRLSState,
    weights: NDArray,
    family: Distribution,
    link: Link,
) -> int:
    """The first halving depth whose trial the mean-space test can accept: ``1`` or deeper.

    ``1`` unless the family and link can leave their mean space
    (``mean_space_violation``), the committed state is inside it and the
    proposal is not.  The binomial/log space ``{eta < 0}`` is open and convex,
    so along the proposal step ``d = eta_prop - eta`` the trial at fraction
    ``alpha`` stays inside exactly when ``alpha < t = min(-eta_i / d_i)`` over
    positive-weight rows with ``d_i > 0``: the largest step to the boundary,
    which interior-point methods compute once by this ratio test before they
    backtrack below it (Nocedal & Wright 2006, Numerical Optimization,
    section 19.2, eq. (19.9) and section 19.3, eq. (19.27); Waechter & Biegler
    2006, Math. Program. 106, eq. (15)).  Halving from the full step instead
    evaluates every power of two above ``t``, each refused by the space test:
    about ``log2(1 / t)`` deviance passes per step, and dozens per step where
    the iterate creeps towards the boundary.

    Each depth ``j`` whose trial the row attaining ``t`` leaves outside, with
    ``fl(eta_r + fl(2^-j d_r)) >= 0`` formed exactly as the trial forms it, is
    a trial the space test refuses, so the search can start at the first depth
    that row admits.  That depth is found by scanning from ``floor(log2(1 /
    t))`` on that one row.  No trial the test could accept is skipped, so the
    accepted step, and every fit, is the halving's own bit for bit; only the
    refused evaluations go.  Backing off by a fixed fraction ``tau t`` instead
    moves the iterate's path, and on #437's extreme sweep that path left 20
    light levels the halving certified short of the bar (``score_stagnated``).
    """
    violates = mean_space_violation(family, link)
    if violates is None or violates(committed.eta_unclipped, weights):
        return 1
    if not violates(proposal.eta_unclipped, weights):
        return 1
    eta = np.asarray(committed.eta_unclipped, dtype=np.float64)
    positive = np.asarray(weights) > 0.0
    # the direction the trials step along, formed as ``evaluate_trial`` forms it
    step = np.asarray(proposal.eta_unclipped, dtype=np.float64) - eta
    if not np.all(np.isfinite(step[positive])):
        return 1
    rising = np.flatnonzero(positive & (step > 0.0))
    if rising.size == 0:
        return 1
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        quotient = -eta[rising] / step[rising]
    row = int(rising[int(np.argmin(quotient))])
    base, direction = eta[row], step[row]
    fraction = float(np.min(quotient))

    def inside(depth: int) -> bool:
        with np.errstate(under="ignore"):
            return bool(base + 2.0**-depth * direction < 0.0)

    limit = _MAX_FLOAT64_HALVING_DEPTH
    depth = 1
    if math.isfinite(fraction) and fraction > 0.0:
        depth = min(max(1, math.floor(-math.log2(fraction))), limit)
    while depth > 1 and inside(depth - 1):
        depth -= 1
    while depth < limit and not inside(depth):
        depth += 1
    return depth


def _select_irls_trial(
    *,
    committed: _IRLSState,
    proposal: _IRLSState,
    evaluate_state: Callable[[float], _IRLSState],
    invalid_state: StateInvalid | None = None,
    max_halving: int = 20,
    extended_max_halving: Callable[[], int] | None = None,
    merit_delta: MeritDelta | None = None,
    merit_scale: float = 1.0,
    merit_roundoff: MeritRoundoff | None = None,
    first_halving: Callable[[], int] | None = None,
) -> _IRLSStepDecision:
    """Return the largest safe fixed-endpoint trial, or reject atomically.

    ``extended_max_halving`` is resolved only after the ordinary budget is
    exhausted.  Extreme Poisson/sqrt fits can therefore search beyond the
    default float depth without paying to derive that bound on normal steps.

    ``first_halving``, resolved only once the full proposal is refused,
    gives the first depth whose trial the space test can accept
    (``mean_space_first_halving``): shallower depths are refused without
    being evaluated, so the decision is the one the halving reaches, with
    fewer evaluations.  ``trials_attempted`` counts the states evaluated.
    """
    if max_halving < 1:
        raise ValueError("max_halving must be at least 1")
    if not _irls_trial_is_unsafe(
        proposal,
        committed,
        invalid_state,
        merit_delta,
        merit_scale,
        merit_roundoff,
    ):
        return _IRLSStepDecision(1.0, 0, False, trials_attempted=1)

    first = 1 if first_halving is None else first_halving()
    trials_attempted = 1
    skipped = 0
    for depth in range(1, max_halving + 1):
        alpha = 2.0**-depth
        if alpha == 0.0:
            return _IRLSStepDecision(0.0, 0, True, trials_attempted=trials_attempted)
        if depth < first:
            skipped += 1  # outside the space: the test would refuse it
            continue
        candidate = evaluate_state(alpha)
        trials_attempted += 1
        if not _irls_trial_is_unsafe(
            candidate,
            committed,
            invalid_state,
            merit_delta,
            merit_scale,
            merit_roundoff,
        ):
            return _IRLSStepDecision(
                alpha, depth, False, trials_attempted=trials_attempted, skipped=skipped
            )

    if extended_max_halving is not None:
        extended_limit = extended_max_halving()
        if extended_limit < max_halving:
            raise ValueError("extended_max_halving must be at least max_halving")
        for depth in range(max_halving + 1, extended_limit + 1):
            alpha = 2.0**-depth
            if alpha == 0.0:
                return _IRLSStepDecision(
                    0.0,
                    0,
                    True,
                    trials_attempted=trials_attempted,
                )
            if depth < first:
                skipped += 1
                continue
            candidate = evaluate_state(alpha)
            trials_attempted += 1
            if not _irls_trial_is_unsafe(
                candidate,
                committed,
                invalid_state,
                merit_delta,
                merit_scale,
                merit_roundoff,
            ):
                return _IRLSStepDecision(
                    alpha,
                    depth,
                    False,
                    trials_attempted=trials_attempted,
                    skipped=skipped,
                )

    return _IRLSStepDecision(0.0, 0, True, trials_attempted=trials_attempted, skipped=skipped)
