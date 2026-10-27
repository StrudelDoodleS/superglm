"""The binomial/log mean space's overlay on the direct IRLS fit: Newton steps and the true-score certificate.

PR #437's machinery for the binomial/log pair, whose log link can leave the
probability space and whose clipped deviance is flat where ``clip_mu`` holds
a row.  ``_IRLSDirectMeanSpaceOverlay`` is all of it, so a later change can
replace it whole (the constrained-QP method on ``eta <= 0`` for the event
rows, stopped by its KKT conditions; design ``2026-10-23-irls-direct-split``).
The mean-space and link-domain trial predicates (#425's guard, #498) are the
shared line search's, not this overlay's.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import EPS as _EPS_FLOAT
from superglm._roundoff import UNIT_ROUNDOFF as _UNIT_ROUNDOFF_FLOAT
from superglm._roundoff import gamma as _gamma
from superglm.solvers.irls_state import (
    _IRLSState,
    _mean_space_halving_budget,
    mean_space_boundary_rows,
    mean_space_clipped_rows,
    mean_space_first_halving,
    mean_space_log_likelihood_rows,
    mean_space_newton_rows,
    mean_space_score_rows,
)
from superglm.solvers.mode_score import (
    MODE_RESOLVE_CAP,
    ModeResidual,
    centred_data_score,
    null_basis_angle,
    penalized_mode_residual,
    row_set_quadratics,
    row_set_residual,
    row_sets,
    truncated_direction_ratio,
    weighted_column_centring,
)

if TYPE_CHECKING:
    from superglm.group_matrix import DesignMatrix
    from superglm.solvers.centered_system import CenteredSystem
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit
    from superglm.solvers.pirls import TerminationReason
    from superglm.solvers.rank import RankDecomposition

# the package's logger: its records keep the name they had when the solver was one module
logger = logging.getLogger(__package__)

# The spacing of the subnormal numbers, 2^-1074, the smallest positive
# float64.  Gradual underflow adds at most half of it to one operation
# (Higham 2002, section 2.1); half is not representable (2.0**-1075 rounds to
# 0), so the bounds below count the whole spacing per operation.
_SUBNORMAL_SPACING = 2.0**-1074


def _underflow_allowance(rows: int) -> float:
    """The absolute error gradual underflow can add to a sum over ``rows`` rows and its two companions."""
    return (rows + 2) * _SUBNORMAL_SPACING


# The largest a coordinate's penalty terms may grow to in the certificate's
# units, short of float64's 2^1024 by enough for the sums that use them.
_PENALTY_HEADROOM_EXPONENT = 1000


def _even_exponent(value: float) -> int:
    """The even ``e`` with ``value * 2^-e`` in ``[1/4, 1)``; 0 for zero or non-finite ``value``."""
    if not (value > 0.0 and math.isfinite(value)):
        return 0
    exponent = math.frexp(value)[1]
    return exponent + (exponent % 2)


def _column_shift(penalty_size: NDArray, weight_exponent: int) -> NDArray:
    """Per coordinate, the even shift that keeps its penalty below ``2^1000`` in the weights' units.

    ``penalty_size`` in the penalty's own units; the shift is 0 wherever the
    penalty, divided by ``2^weight_exponent``, already fits.
    """
    size = np.asarray(penalty_size, dtype=np.float64)
    exponents = np.where(size > 0.0, np.frexp(size)[1], 0)
    excess = np.maximum(exponents - weight_exponent - _PENALTY_HEADROOM_EXPONENT, 0)
    return (excess + excess % 2).astype(np.int64)


class _IRLSDirectMeanSpaceOverlay:
    """The binomial/log mean space on every route: #437's Newton steps and the true-score certificate.

    The fit holds one (``mean_space``) when its family and link can leave
    their mean space (``irls_state.mean_space_violation``), else ``None``, and
    every consult is a test of that.  It keeps no reference to the fit, whose
    state each method is handed.

    **The Newton half** acts on the dense, QP and structured steps once the
    certificate engages it (``engaged``); it is available (``available``)
    off the QR route, whose least squares cannot carry a zero-curvature row's
    score, the SCOP route, which takes its own Newton steps, and the fREML
    performance iteration.  Its parts:

    - ``newton_rows``: the working rows (L02) on the observed information;
    - ``centred_rhs``: the score in the centred right-hand side, and its
      centred data part, which the dense step's increment reads (the dense and
      QP steps, L06 and L07); the structured step puts the score in its block
      right-hand side and its truncated increment itself, from the fit's
      ``newton_score``;
    - ``newton_iterate``: the dense step's increment form (L06);
    - ``line_search``: the shared line search's first halving and a lowered
      start's budget, on every binomial/log fit, and under Newton steps the
      merit on the model's own deviance (L12, L14).

    **The certificate half** acts on every route, through the step's
    ``certificate_inputs``: ``judge`` (L17) inside the stop rule, ``certify``
    (the true-score certificate it reads), and ``final`` (F04, the boundary
    verdict), whose numbers the fit publishes.
    """

    __slots__ = (
        "available engaged start_lowered _committed_rows _true_ratios true_stop"
        " true_confirmed true_stop_ratio last_true_residual _judged_truncated"
        " _last_truncated _row_set_curvature boundary_rows published_truncated"
        " truncated_boundary"
    ).split()

    def __init__(self, *, available: bool, start_lowered: bool) -> None:
        self.available = available
        # Whether the default start was lowered into the mean space
        # (``_IRLSDirectFit._setup_design``): its line search may halve past
        # the ordinary budget, and its stop is certified.
        self.start_lowered = start_lowered
        # Newton steps on the observed information once the certificate refused
        # a stop near the mode or its refused score stalled (``judge``)
        self.engaged = False
        # the committed state's log-likelihood rows under Newton steps, which
        # the line search's merit reads
        self._committed_rows: NDArray | None = None
        # the certificate ratio at every stop it refused, for the stagnation stop
        self._true_ratios: list[float] = []
        # Whether the certificate decided the last iteration under Newton steps,
        # its ratio there, and its last residual: the diagnostics then publish
        # that rule's numbers, and ``PIRLSResult.mean_space_true_mode`` tells a
        # criterion read at the state that it is the model's own mode.
        self.true_stop = False
        self.true_confirmed = False
        self.true_stop_ratio = math.inf
        self.last_true_residual: ModeResidual | None = None
        # The latest certificate's judgement of the truncated directions
        # (``truncated_direction_ratio``) with the state it judged (its eta),
        # and the latest judgement that found any.  Only a judgement of the
        # returned state decides the fit's verdict; an earlier one is
        # published as history (``final``).
        self._judged_truncated: tuple = (None, ())
        self._last_truncated: tuple = (None, ())
        # d'Sd along each one-hot reference and joint-cell direction
        # (``mode_score.row_set_quadratics``): one penalty product each, formed
        # once per fit on first use (S and the design are fixed within a fit)
        self._row_set_curvature: NDArray | None = None
        # what ``final`` decides, which the fit publishes
        self.boundary_rows = 0
        self.published_truncated: tuple = ()
        self.truncated_boundary = False

    # ── the Newton half ──────────────────────────────────────────────

    def newton_rows(
        self, fit: _IRLSDirectFit, committed: _IRLSState
    ) -> tuple[NDArray, NDArray, NDArray] | None:
        """L02 once engaged: ``(W, score, z)`` on the observed information, or ``None``.

        The observed curvature and the score of the unclipped eta, the step in
        score form.  ``z`` is then the predictor itself, so ``W (z - eta)`` adds
        nothing and the score enters on its own.  A state holding rows at the
        boundary keeps Fisher's rows (``None``).
        """
        family, link, y, weights = fit.family, fit.link, fit.y, fit.weights
        eta = committed.eta_unclipped
        self._committed_rows = None
        if mean_space_boundary_rows(family, link, eta, weights):
            return None
        newton_rows = mean_space_newton_rows(y, weights, eta)
        if newton_rows is None:
            return None
        W, newton_score = newton_rows
        z = np.array(eta, dtype=np.float64)
        self._committed_rows = mean_space_log_likelihood_rows(y, weights, eta)
        fit.recorder.count("irls_mean_space_newton_iters")
        return W, newton_score, z

    @staticmethod
    def centred_rhs(
        centered: CenteredSystem, newton_score: NDArray, dm: DesignMatrix
    ) -> tuple[CenteredSystem, NDArray]:
        """The centred system with the score in its right-hand side, and the score's ``X~' u``.

        The working mean of ``z_off`` is the centred intercept; the score moves
        it by ``sum(u) / sum(W)`` and the slopes' right-hand side by ``X~' u``,
        which the dense step's increment reads too (``newton_iterate``), so it
        is formed once.  ``replace`` keeps the system's arrays, so the factor
        certificate's cache, which tests them by identity, still hits.
        """
        data_score = centred_data_score(dm, newton_score, centered.mean_x)
        system = replace(
            centered,
            mean_z=centered.mean_z + float(np.sum(newton_score)) / centered.sum_w,
            rhs=centered.rhs + data_score,
        )
        return system, data_score

    @staticmethod
    def newton_iterate(
        fit: _IRLSDirectFit, rank: RankDecomposition, data_score: NDArray
    ) -> NDArray:
        """The dense step's Newton iterate: the committed coefficients plus the increment.

        ``data_score`` is the score's ``X~' u`` (``centred_rhs``).  The
        increment, not the iterate: a direction the factor truncates keeps the
        iterate's component instead of the minimum-norm reset of solving for it
        (Pes & Rodriguez 2021, eqs. 1.2-1.4).
        """
        return fit.committed.beta + rank.solve(data_score - fit.penalty_matvec(fit.committed.beta))

    def line_search(
        self,
        fit: _IRLSDirectFit,
        committed: _IRLSState,
        proposal: _IRLSState,
        newton_score: NDArray | None,
    ) -> tuple[
        Callable[[], int],
        Callable[[], int] | None,
        Callable[[_IRLSState, _IRLSState], float] | None,
        Callable[[_IRLSState, _IRLSState], float] | None,
    ]:
        """The shared line search's mean-space hooks for one iteration (L12, L14).

        ``(first_halving, lowered_budget, deviance_delta, merit_roundoff)``:

        - the first halving whose trial the mean-space test can accept
          (``irls_state.mean_space_first_halving``): the halvings above the
          boundary are refused without a deviance pass each;
        - a lowered start's budget (``irls_state._mean_space_halving_budget``),
          which can sit far inside the space with a proposal far outside it,
          past the ordinary budget; ``None`` otherwise;
        - under Newton steps, the merit's deviance change on the model's own
          deviance, which still sees the rows ``clip_mu`` holds (Fisher's is
          judged on the clipped one), and its rounding; ``None`` otherwise.
        """
        weights, family, link, max_halving = fit.weights, fit.family, fit.link, fit.max_halving

        def first_halving() -> int:
            return mean_space_first_halving(
                committed=committed,
                proposal=proposal,
                weights=weights,
                family=family,
                link=link,
            )

        lowered_budget = None
        if self.start_lowered:

            def lowered_budget() -> int:
                return _mean_space_halving_budget(
                    committed=committed,
                    proposal=proposal,
                    weights=weights,
                    family=family,
                    link=link,
                    default=max_halving,
                )

        if newton_score is None:
            return first_halving, lowered_budget, None, None
        y, newton_committed_rows = fit.y, self._committed_rows
        # the last trial's rows, read by both the decrease and its
        # round-off allowance: one log1mexp pass per trial
        _trial_rows: list = [None, None]

        def _newton_rows_at(state: _IRLSState) -> NDArray:
            if state is committed and newton_committed_rows is not None:
                return newton_committed_rows
            if _trial_rows[0] is not state:
                _trial_rows[0] = state
                _trial_rows[1] = mean_space_log_likelihood_rows(y, weights, state.eta_unclipped)
            return _trial_rows[1]

        def _newton_deviance_delta(candidate: _IRLSState, base: _IRLSState) -> float:
            # each row's change first, then numpy's pairwise sum
            return float(-2.0 * np.sum(_newton_rows_at(candidate) - _newton_rows_at(base)))

        def _newton_merit_roundoff(candidate: _IRLSState, base: _IRLSState) -> float:
            # The true deviance's own magnitudes: the rows' sizes (the
            # log1mexp and product roundings, and the pairwise sum's
            # gamma_{log2 n + 8}, are within 64 eps of them for any n
            # below 2^50), plus each state's penalty part.
            size = float(np.sum(np.abs(_newton_rows_at(candidate))))
            size += float(np.sum(np.abs(_newton_rows_at(base))))
            for state in (candidate, base):
                if state.penalized_deviance is not None:
                    size += abs(float(state.penalized_deviance) - float(state.deviance))
            return 64.0 * _EPS_FLOAT * 2.0 * size

        return first_halving, lowered_budget, _newton_deviance_delta, _newton_merit_roundoff

    # ── the certificate half ─────────────────────────────────────────

    def judge(self, fit: _IRLSDirectFit, stagnation_window: int) -> None:
        """L17: a stop the clip can fake is certified on the model's own score, after the stop rule.

        The stop rule runs it (``_IRLSDirectConvergence.judge``) after its
        criterion and the step-rejected veto, and before the feasibility
        contract's veto, which may still veto the stop; its refused stops
        stagnate over the rule's window.  It may refuse a stop, stop a fit
        under Newton steps, end one whose refused score stagnates, and engage
        the Newton steps (``engaged``).
        """
        engaged = self.engaged
        _true_ratios, converged_this_iter = self._true_ratios, fit.converged_this_iter
        family, link, retained, weights = fit.family, fit.link, fit.retained, fit.weights
        score_stagnated = fit.score_stagnated
        # Once Newton steps are engaged the model's own certificate is the
        # stop, every iteration: the clipped rule a mode_score fit stops on
        # cannot pass at a mode it does not see (below).
        newton_stop = bool(engaged and np.isfinite(fit.dev) and not fit.step_rejected)
        # A state holding rows at the boundary is never published as converged
        # (``mean_space_boundary`` below), so its stop stands as on master.
        at_boundary = bool(
            (converged_this_iter or newton_stop)
            and mean_space_boundary_rows(family, link, retained.eta_unclipped, weights)
        )
        # rows clip_mu holds, at the step's base or where it landed
        clipped_rows = (
            mean_space_clipped_rows(family, link, fit.committed.eta_unclipped, weights)
            + mean_space_clipped_rows(family, link, retained.eta_unclipped, weights)
            if converged_this_iter and not at_boundary
            else 0
        )
        _true_confirmed = False
        if not at_boundary and (
            newton_stop or (converged_this_iter and (clipped_rows or self.start_lowered))
        ):
            # On a row whose mean leaves clip_mu's band the clipped deviance
            # is flat and the step's score is not the binomial/log score, and
            # a lowered start can sit far below that band: a stop read off
            # the clipped objective, or off a step taken from it, is accepted
            # only where the model's own penalized score certifies it
            # (``certify``).  A refused stop goes on.  Each refused
            # stop sits where the iteration's own rule passed, so a score that
            # has stopped contracting across them (``stagnation_window``) will
            # not reach the bar: the fit ends uncertified, as under
            # ``mode_score``.  Under Newton steps the certificate alone
            # decides, at every iteration, and its own ratios alone stagnate.
            true_ratio, true_resolved = self.certify(
                fit,
                retained.beta,
                retained.intercept,
                retained.eta_unclipped,
                fit.step.certificate_inputs(),
            )
            # the state is certified on the model's own score, Newton or not,
            # where the clip holds a row (or Newton decided): a criterion read
            # at it reads the model's own likelihood and curvature
            _true_confirmed = bool(true_ratio <= 1.0 and (newton_stop or clipped_rows))
            if newton_stop:
                converged_this_iter = bool(true_ratio <= 1.0)
                score_stagnated = False
                # this rule decided: its numbers are the ones published
                fit.convergence_label = "mean_space_mode_score"
                fit.convergence_value = fit.mode_bar * true_ratio
                fit.convergence_tolerance = fit.mode_bar
                self.true_stop_ratio = true_ratio
            if not true_ratio <= 1.0:
                converged_this_iter = False
                _true_ratios.append(true_ratio)
                score_stagnated = len(_true_ratios) > stagnation_window and not (
                    min(_true_ratios[-stagnation_window:])
                    < 0.5 * min(_true_ratios[:-stagnation_window])
                )
                if self.available and not engaged and (true_resolved or score_stagnated):
                    # Fisher scoring has carried the fit to where its own
                    # stop passes but the model's score does not: near the
                    # mode (resolved), or where scoring has stalled.  The
                    # iterations left take Newton steps on the observed
                    # information (``irls_state.mean_space_newton_rows``),
                    # which converge quadratically where scoring is linear;
                    # far from the mode Fisher's steps stay the safer ones
                    # (Green 1984, section 2.3 and its discussion).
                    engaged = True
                    _true_ratios.clear()
                    score_stagnated = False

        self.true_stop = bool(newton_stop and not at_boundary)
        fit.converged_this_iter, fit.score_stagnated = converged_this_iter, score_stagnated
        self.true_confirmed, self.engaged = _true_confirmed, engaged

    def certify(
        self,
        fit: _IRLSDirectFit,
        beta_values: NDArray,
        intercept_value: float,
        eta_values: NDArray,
        inputs: tuple,
    ) -> tuple[float, bool]:
        """The mode certificate on the binomial/log score itself, for a stop the clip can fool.

        ``(ratio, resolved)``: the largest relative score over its bar
        (``ModeResidual.ratio``), ``inf`` when the arithmetic cannot resolve
        it, and whether every relative score is within ``MODE_RESOLVE_CAP``
        (near the mode).  ``mode_residual``'s relative penalized score, bar,
        floors and weak tests, evaluated on the score and Fisher weights of
        the unclipped ``eta_values`` (``irls_state.mean_space_score_rows``)
        and centred on those weights (``mode_score.weighted_column_centring``),
        not on the clipped solve's system.

        **Units.** Each relative score is a ratio of quantities of degree one
        in the weights and the penalty taken together, so it does not depend
        on their units; the arithmetic that forms it does.
        - The prior weights are first brought by an even power of two to a
          largest entry in ``[1/4, 1)``, before any row product, so no row
          product underflows that would not at unit scale.
        - Each coordinate then takes its own even power of two: the shift
          that keeps its penalty terms, in the weights' new units, below
          ``2^1000``, applied to its data terms too (``penalized_mode_residual``'s
          ``column_shift``).  A data term that underflows there lies more
          than ``2^-2000`` below the penalty beside it.
        - Powers of two are exact on normal and subnormal values, and an even
          one keeps ``zeta``'s square root exact.
        - Gradual underflow adds at most half the subnormal spacing
          ``2^-1074`` of absolute error to each operation (Higham 2002,
          section 2.1), so a sum over ``n`` rows carries at most ``(n + 2)
          2^-1074`` beyond its ``gamma_n`` relative error
          (``_underflow_allowance``, the whole spacing per operation since
          half of it is not representable).  Where that is not below the
          bar's allowance, ``bar`` times an identified coordinate's scale,
          the certificate cannot resolve the score and refuses: ``inf``,
          never a pass.
        - Slopes the weak test finds at rounding-level curvature are left out
          of the ratio only while the iterate is at the mode along them: half
          their block Newton decrement, over their penalised curvature block
          (``weak_penalty_block`` for the penalty's part), within the
          log-likelihood's own rounding over the rows the block touches,
          ``gamma_{|S| + 4} sum_{i in S} |l_i|`` (``penalized_mode_residual``'s
          ``decrement_noise``).  The penalty's rounding is left out of that
          noise, which can only refuse.

        **What the step holds** (``inputs``, the step's ``certificate_inputs``):
        the decomposition its solve holds, the constraint rows active at the
        iterate, the columns tested in latent coordinates instead, and the
        latent groups.

        **Constraints.** The active rows are the hard constraints ``a' beta >=
        b`` active at the iterate.  A constrained mode is stationary when ``G
        + A' m = 0`` with ``m >= 0``, so the residual tested is ``G + A' m``
        at the non-negative least-squares ``m`` of its Jacobi-scaled form.
        Each row is put in the coordinates' units, brought to a largest entry
        in ``[1/2, 1)`` by a power of two and then to unit length: this leaves
        the cone it spans unchanged, and its own units can neither overflow
        nor vanish in the norm.

        **SCOP.** The latent groups pair each shape-constrained group's state
        with its spec.  A shape-constrained group's coefficients are its
        latent ``theta``, with ``gamma = forward(theta)``.  Its stationarity is
        the latent score ``J' B' s - lambda S theta`` (``J = d gamma / d
        theta``, diagonal).  The groups' latent scores are projected together off the
        joint directions their Newton step could not resolve, and tested in
        the same relative form, bar and floor, in the weights' units (a latent
        penalty that overflows there is refused).
        """
        dm, lambda2, p, y, weights = fit.dm, fit.lambda2, fit.p, fit.y, fit.weights
        mode_bar, _penalty_curvature = fit.mode_bar, fit._penalty_curvature
        penalty_matvec, penalty_rounding = fit.penalty_matvec, fit.penalty_rounding
        held_rank, active_rows, scop_columns, scop_groups = inputs
        self.last_true_residual = None
        # this state is judged: no truncated direction unless one is found below
        self._judged_truncated = (eta_values, ())
        positive = weights > 0.0
        largest_weight = float(np.max(weights[positive], initial=0.0)) if np.any(positive) else 0.0
        weight_exponent = _even_exponent(largest_weight)
        score, fisher = mean_space_score_rows(y, np.ldexp(weights, -weight_exponent), eta_values)
        if not (np.all(np.isfinite(score)) and np.all(np.isfinite(fisher))):
            return math.inf, False
        if not _penalty_curvature:
            _penalty_curvature.append(fit.penalty_curvature())
        penalty_score = penalty_matvec(beta_values)
        penalty_magnitude = penalty_matvec(beta_values, magnitude=True)
        curvature = np.asarray(_penalty_curvature[0], dtype=np.float64)
        if not (
            np.all(np.isfinite(penalty_score))
            and np.all(np.isfinite(penalty_magnitude))
            and np.all(np.isfinite(curvature))
        ):
            return math.inf, False
        shift = _column_shift(
            np.maximum(np.abs(penalty_magnitude), np.abs(curvature)), weight_exponent
        )
        penalty_score = np.ldexp(penalty_score, -weight_exponent - shift)
        penalty_magnitude = np.ldexp(penalty_magnitude, -weight_exponent - shift)
        curvature = np.ldexp(curvature, -weight_exponent - shift)
        mean_x, sum_w, diagonal = weighted_column_centring(dm, fisher, positive)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.sqrt(np.maximum(diagonal, 0.0) / sum_w)
        excluded = np.zeros(p, dtype=bool)
        excluded[list(fit._laplace_excluded)] = True
        if scop_columns is not None:
            excluded |= scop_columns
        tiny = float(np.finfo(np.float64).tiny)
        zeta = float(np.sum(np.abs(score))) / math.sqrt(max(sum_w, tiny))
        if active_rows is not None and active_rows.size:
            rows = np.ldexp(np.asarray(active_rows, dtype=np.float64), -shift[None, :])
            largest_entry = np.max(np.abs(rows), axis=1)
            rows = rows[largest_entry > 0.0]
            exponents = np.frexp(largest_entry[largest_entry > 0.0])[1]
            rows = np.ldexp(rows, -exponents[:, None])
            rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
            gradient = np.ldexp(centred_data_score(dm, score, mean_x), -shift) - penalty_score
            jacobi = np.ldexp(np.full(p, zeta), -(shift // 2)) * np.sqrt(
                np.ldexp(np.maximum(diagonal, 0.0), -shift) + np.maximum(curvature, 0.0)
            )
            jacobi = jacobi + np.abs(penalty_score)
            # any positive diagonal gives a valid m >= 0; keeping it within
            # eps of its largest entry keeps the scaled rows finite
            jacobi = np.maximum(jacobi, _EPS_FLOAT * max(float(np.max(jacobi, initial=0.0)), 1.0))
            if rows.size and np.all(np.isfinite(gradient)) and np.all(np.isfinite(jacobi)):
                from scipy.optimize import nnls

                try:
                    multipliers = nnls(rows.T / jacobi[:, None], -gradient / jacobi)[0]
                except RuntimeError:  # the active-set iteration's own limit
                    multipliers = np.zeros(rows.shape[0])
                penalty_score = penalty_score - rows.T @ multipliers
                penalty_magnitude = penalty_magnitude + np.abs(rows).T @ multipliers
        centre_shift = float(mean_x @ beta_values)

        likelihood_rows: list = []

        def likelihood_noise(support: NDArray) -> float:
            # the log-likelihood's rounding over the rows a weak block touches
            # (``penalized_mode_residual``): gamma for the terms summed there,
            # plus each row's own few roundings
            if not likelihood_rows:
                likelihood_rows.append(
                    np.abs(
                        mean_space_log_likelihood_rows(
                            y, np.ldexp(weights, -weight_exponent), eta_values
                        )
                    )
                )
            touched = np.asarray(support, dtype=bool) & positive
            noise = _gamma(int(np.count_nonzero(touched)) + 4) * float(
                np.sum(likelihood_rows[0][touched])
            )
            return noise if math.isfinite(noise) else 0.0

        def weak_penalty_block(columns: NDArray) -> NDArray:
            # the penalty's block on a few weakly identified slopes, in the
            # weights' units: one penalty product per slope
            block = np.empty((len(columns), len(columns)))
            for position, column in enumerate(columns):
                unit = np.zeros(p)
                unit[column] = 1.0
                block[:, position] = penalty_matvec(unit)[columns]
            with np.errstate(over="ignore"):
                return np.ldexp(0.5 * (block + block.T), -weight_exponent)

        residual = penalized_mode_residual(
            dm=dm,
            row_score=score,
            fisher_weights=fisher,
            positive_prior=positive,
            mean_x=mean_x,
            centered_scale=np.where(np.isfinite(scale), scale, 0.0),
            alpha=float(intercept_value) + centre_shift,
            eta_tilde=eta_values - fit.offset - float(intercept_value) - centre_shift,
            penalty_score=penalty_score,
            penalty_magnitude=penalty_magnitude,
            penalty_curvature=curvature,
            sum_w=sum_w,
            bar=mode_bar,
            excluded=excluded,
            resolve_cap=MODE_RESOLVE_CAP,
            column_shift=shift,
            decrement_noise=likelihood_noise,
            penalty_block=weak_penalty_block,
            penalty_rounding=penalty_rounding(),
        )
        self.last_true_residual = residual
        rows_n = int(np.count_nonzero(positive))
        underflow = _underflow_allowance(rows_n)
        assert residual.scale is not None
        identified = np.concatenate(([True], ~residual.excluded))
        if np.any(mode_bar * residual.scale[identified] <= underflow):
            return math.inf, False
        ratio = residual.ratio()
        # every set of rows the one-hot blocks move on their own, certified on
        # its own rows (``mode_score.row_set_residual``); the sets are the
        # design's, formed once per design (``mode_score.row_sets``)
        sets = row_sets(dm)
        if self._row_set_curvature is None:
            self._row_set_curvature = row_set_quadratics(
                sets, p, penalty_matvec, penalty_rounding()
            )
        with np.errstate(over="ignore", invalid="ignore"):
            column_penalty = np.ldexp(penalty_score, shift)
            column_penalty_size = np.ldexp(penalty_magnitude, shift)
            column_curvature = np.ldexp(curvature, shift)
            set_curvature = np.ldexp(self._row_set_curvature, -weight_exponent)
        ratio = max(
            ratio,
            row_set_residual(
                sets=sets,
                row_score=score,
                response=y,
                fisher_weights=fisher,
                positive_prior=positive,
                eta=eta_values,
                column_penalty=column_penalty,
                column_penalty_size=column_penalty_size,
                column_curvature=column_curvature,
                set_curvature=set_curvature,
                bar=mode_bar,
                underflow=underflow,
                penalty_rounding=penalty_rounding(),
            ),
        )
        # the directions the factorization this iteration holds truncates,
        # judged on the rows they move (``mode_score.truncated_direction_ratio``):
        # a light row set the relative score sums beside the heavy rows'
        # rounding.  The gram and qr routes hold one; unconstrained, no SCOP.
        if (
            held_rank is not None
            and held_rank.parameter_null_basis is not None
            and (active_rows is None or not active_rows.size)
            and not scop_groups
        ):
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                truncated_ratio, truncated = truncated_direction_ratio(
                    dm=dm,
                    null_basis=held_rank.parameter_null_basis,
                    angle=null_basis_angle(held_rank, fit.n),
                    mean_x=mean_x,
                    row_score=score,
                    fisher_weights=fisher,
                    response=y,
                    positive_prior=positive,
                    penalty_gradient=np.ldexp(penalty_matvec(beta_values), -weight_exponent),
                    penalty_size=np.ldexp(
                        penalty_matvec(beta_values, magnitude=True), -weight_exponent
                    ),
                    penalty_apply=lambda v: np.ldexp(penalty_matvec(v), -weight_exponent),
                    bar=mode_bar,
                    underflow=underflow,
                    column_scale=held_rank.column_scale,
                    eta=eta_values,
                    penalty_size_apply=lambda v: np.ldexp(
                        penalty_matvec(v, magnitude=True), -weight_exponent
                    ),
                    penalty_rounding=penalty_rounding(),
                )
            ratio = max(ratio, truncated_ratio)
            self._judged_truncated = (eta_values, truncated)
            if truncated:
                self._last_truncated = (eta_values, truncated)
            if truncated_ratio == math.inf:
                # refused because the basis or the arithmetic cannot resolve
                # the truncated rows (an ``unresolved_basis`` record)
                fit.recorder.count("irls_truncated_unresolved")
        resolved = bool(np.max(residual.relative, initial=0.0) <= MODE_RESOLVE_CAP)
        latent_parts = []
        for group_state, spec in scop_groups or ():
            group = spec.group
            lam = float(lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2)
            theta = np.asarray(group_state.beta_eff, dtype=np.float64)
            penalty_matrix = np.asarray(spec.S_scop, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                pull = np.ldexp(lam * (penalty_matrix @ theta), -weight_exponent)
                pull_size = np.ldexp(
                    lam * (np.abs(penalty_matrix) @ np.abs(theta)), -weight_exponent
                )
                pull_diagonal = np.ldexp(lam * np.diag(penalty_matrix), -weight_exponent)
            # a latent penalty past float64 beside the data's units: refused
            # (the shape-constrained groups are tested in the weights' units)
            if not (
                np.all(np.isfinite(pull))
                and np.all(np.isfinite(pull_size))
                and np.all(np.isfinite(pull_diagonal))
            ):
                return math.inf, False
            bins = spec.bin_idx
            basis = np.asarray(spec.B_scop, dtype=np.float64)

            def by_bin(values: NDArray, bins=bins, basis=basis) -> NDArray:
                if bins is None:
                    return values
                return np.bincount(bins, weights=values, minlength=basis.shape[0])

            jacobian = spec.reparam.jacobian_diagonal(theta)
            latent_parts.append(
                (
                    group_state,
                    theta,
                    jacobian * (basis.T @ by_bin(score)) - pull,
                    jacobian**2 * ((basis**2).T @ by_bin(fisher)) + pull_diagonal,
                    np.abs(pull),
                    pull_size,
                    np.abs(jacobian) * (np.abs(basis).T @ by_bin(np.abs(score))),
                )
            )
        if latent_parts:
            # The joint step truncates one factor across every SCOP group and
            # hands each group its own columns of each discarded direction, so
            # the groups' blocks reassemble row by row into the joint
            # directions (a group absent here holds zeros in them).  The
            # concatenated latent score is projected once off their span,
            # which an SVD finds with its rank (a QR would invent a direction
            # for a zero block).
            widths = [part[1].size for part in latent_parts]
            offsets = np.concatenate(([0], np.cumsum(widths)))
            blocks = [
                (k, np.asarray(part[0].discarded_directions, dtype=np.float64))
                for k, part in enumerate(latent_parts)
                if part[0].discarded_directions is not None
                and np.asarray(part[0].discarded_directions).ndim == 2
                and np.asarray(part[0].discarded_directions).shape[0] > 0
            ]
            gradient = np.concatenate([part[2] for part in latent_parts])
            if blocks:
                counts = {rows.shape[0] for _, rows in blocks}
                if len(counts) != 1:  # pragma: no cover - joint-step invariant
                    raise RuntimeError(
                        "the joint SCOP step handed its groups different discarded directions"
                    )
                directions = np.zeros((counts.pop(), int(offsets[-1])))
                for k, rows in blocks:
                    directions[:, offsets[k] : offsets[k + 1]] = rows
                left, singular, _ = np.linalg.svd(directions.T, full_matrices=False)
                cutoff = float(np.max(singular, initial=0.0)) * directions.shape[1] * _EPS_FLOAT
                frozen = left[:, singular > cutoff]
                gradient = gradient - frozen @ (frozen.T @ gradient)
            curvature_all = np.concatenate([part[3] for part in latent_parts])
            pull_all = np.concatenate([part[4] for part in latent_parts])
            size_all = np.concatenate([part[5] for part in latent_parts])
            evaluation = np.concatenate([part[6] for part in latent_parts])
            latent_scale = np.maximum(
                tiny, zeta * np.sqrt(np.maximum(curvature_all, 0.0)) + pull_all
            )
            if np.any(mode_bar * latent_scale <= underflow):
                return math.inf, False
            relative = np.abs(gradient) / latent_scale
            floor = (
                _gamma(rows_n) * evaluation
                + _gamma(int(offsets[-1]) + 2) * size_all
                + _UNIT_ROUNDOFF_FLOAT * size_all
            ) / latent_scale
            ratio = max(ratio, float(np.max(relative / np.maximum(mode_bar, floor), initial=0.0)))
            resolved = resolved and bool(np.max(relative, initial=0.0) <= MODE_RESOLVE_CAP)
        return ratio, resolved

    def final(
        self,
        fit: _IRLSDirectFit,
        eta_unclipped: NDArray,
        converged: bool,
        termination_reason: TerminationReason,
    ) -> tuple[bool, TerminationReason]:
        """F04: the retained state's verdict at the mean-space boundary, after the constraints' (F03).

        A returned state with rows at the mean-space boundary is not a mode of
        the model: their capped mean makes the deviance flat there, and the
        maximum it approaches is constrained, not stationary.  It is never
        converged, whatever stopped the loop (``mean_space_boundary_rows``).
        ``eta_unclipped`` is the retained state's own array, which the
        truncated-direction verdict compares by identity.  It returns the
        fit's convergence and reason, and keeps what the fit publishes:
        ``boundary_rows``, ``published_truncated`` and ``truncated_boundary``.
        """
        _boundary_rows = mean_space_boundary_rows(fit.family, fit.link, eta_unclipped, fit.weights)
        if _boundary_rows:
            converged = False
            termination_reason = "mean_space_boundary"
            logger.info(
                "fit_irls_direct: %d row(s) at the boundary of the family's mean space; "
                "the penalized maximum is constrained, fit is not converged.",
                _boundary_rows,
            )
        # The truncated directions published.  A judgement of the returned state
        # (the certificate read its eta) is the fit's own: published as it is, and
        # the only one that may decide the verdict.  The returned state of a
        # rejected step is the committed one, so its judgement counts.  When the
        # returned state was never judged, a fit that is not converged publishes
        # the latest judgement that found any as history (``earlier``), so a
        # refusal stays disclosed, and it decides nothing.
        _judged_eta, _judged = self._judged_truncated
        if _judged_eta is not None and _judged_eta is eta_unclipped:
            _published_truncated = _judged
        elif converged:
            _published_truncated = ()
        else:
            _published_truncated = tuple(
                replace(record, earlier=True) for record in self._last_truncated[1]
            )
        _boundary_sets = [
            record for record in _published_truncated if record.boundary and not record.earlier
        ]
        # rows rising to the boundary along a direction the factorization
        # truncates relabel only an exit the loop ran out on, never a
        # non-finite deviance or a constraint's verdict
        _truncated_boundary = bool(
            not converged
            and not _boundary_rows
            and _boundary_sets
            and termination_reason in ("max_iter", "score_stagnated", "step_rejected")
        )
        if _truncated_boundary:
            # rows whose supremum along a direction the factorization truncates
            # is the boundary (``TruncatedDirection.boundary``): the same verdict
            # as rows held at it, though float64 cannot carry them there
            termination_reason = "mean_space_boundary"
            _boundary_rows = sum(record.row_count for record in _boundary_sets)
            logger.info(
                "fit_irls_direct: %d row(s) rise towards the boundary of the family's mean "
                "space along a direction float64 cannot resolve; fit is not converged.",
                _boundary_rows,
            )
        self.boundary_rows, self.published_truncated = _boundary_rows, _published_truncated
        self.truncated_boundary = _truncated_boundary
        return converged, termination_reason
