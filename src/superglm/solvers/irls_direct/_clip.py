"""The direct IRLS fit's reading of binomial rows the working clip holds (``_IRLSDirectClipOverlay``)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm.solvers.clip_rows import (
    ClipRows,
    count_held_against,
    declares_tail_rows,
    exported_working_rows,
    likelihood_deviance,
    likelihood_working_rows,
    note_held_against,
)

if TYPE_CHECKING:
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit
    from superglm.solvers.irls_state import _IRLSState
    from superglm.solvers.pirls import TerminationReason
    from superglm.solvers.working_rows import CoefficientWorkingRows

# the loop's own exits: the reasons a held row may rename
_LOOP_EXITS = ("converged", "max_iter", "step_rejected", "score_stagnated")


class _IRLSDirectClipOverlay:
    """A binomial fit's rows where the working clip holds one, on every route.

    Held (``clip``) when the clip can hold a row against its outcome
    (``clip_rows.clip_can_hold_against``), else ``None``; it keeps no reference to the fit.
    While ``ClipRows`` is ``working`` it does nothing until the loop ends, so a fit ending
    clear of such rows is the solver's own bit for bit; ``moves_to_likelihood`` then counts
    rows held against their outcome, once. With any, a pair declaring tail rows moves to
    ``likelihood`` and carries on; one declaring none, or past float64, ends ``clip_held``.
    Under ``likelihood`` held rows read the likelihood in their unclipped predictor
    (``clip_rows.tail_rows``): ``deviance`` (merit, stop, published) and ``patched``
    (working weight and response); rows inside the guard keep the solver's arithmetic.
    """

    __slots__ = ("rows", "declared", "held_against", "unfinished", "information_underflowed")

    def __init__(self, family: object, link: object, shared: ClipRows | None) -> None:
        self.rows = ClipRows() if shared is None else shared
        self.declared = declares_tail_rows(family, link)
        # rows held against their outcome at the state the working rows' loop ended on
        self.held_against = 0
        # the move to the likelihood's rows came with no iteration left to solve them
        self.unfinished = False
        # float64 holds none of the likelihood's information on the returned rows (``exported``)
        self.information_underflowed = False

    @property
    def likelihood(self) -> bool:
        return self.rows.state == "likelihood"

    def deviance(self, fit: _IRLSDirectFit, state: _IRLSState) -> float:
        """A state's deviance with every held row's term read from the likelihood."""
        return likelihood_deviance(
            fit.family,
            fit.link,
            fit.y,
            fit.weights,
            state.eta_unclipped,
            state.eta,
            state.mu,
            state.deviance,
        )

    def patched(
        self,
        fit: _IRLSDirectFit,
        rows: CoefficientWorkingRows,
        eta_unclipped: NDArray,
        eta: NDArray,
        mu: NDArray,
        *,
        observed: bool,
    ) -> CoefficientWorkingRows:
        """``rows`` with every held row's weight and response read from the likelihood.

        A held row's response is about its unclipped predictor
        (``clip_rows.tail_working_rows``), which is every other row's own, so
        ``weights * (response - eta_unclipped)`` is the likelihood's score.
        """
        weights, response = likelihood_working_rows(
            fit.family,
            fit.link,
            fit.y,
            fit.weights,
            eta_unclipped,
            eta,
            mu,
            rows.weights,
            rows.response,
            observed=observed,
        )
        if weights is rows.weights:
            return rows
        return replace(rows, weights=weights, response=response)

    def exported(
        self,
        fit: _IRLSDirectFit,
        rows: CoefficientWorkingRows,
        eta_unclipped: NDArray,
        eta: NDArray,
        mu: NDArray,
    ) -> CoefficientWorkingRows:
        """``rows`` as the fit returns them: every held row's weight its Fisher information alone.

        Where float64 holds none of that information on any row
        (``clip_rows.exported_working_rows``), ``rows`` as they are, and
        ``information_underflowed`` says so.
        """
        weights, response, self.information_underflowed = exported_working_rows(
            fit.family,
            fit.link,
            fit.y,
            fit.weights,
            eta_unclipped,
            eta,
            mu,
            rows.weights,
            rows.response,
        )
        if weights is rows.weights:
            return rows
        return replace(rows, weights=weights, response=response)

    def moves_to_likelihood(self, fit: _IRLSDirectFit, it: int) -> bool:
        """Whether the working rows' loop ended on a row held against its outcome and carries on.

        One pass over the retained state's rows.  With such a row and tail
        rows float64 can hold there, the fit's ``ClipRows`` moves to
        ``likelihood``, the retained state is evaluated again on those rows
        and becomes the base of the iterations left.  ``True`` when one is
        left.
        """
        if not np.isfinite(fit.dev):
            return False
        retained = fit.retained
        self.held_against = count_held_against(
            fit.family,
            fit.link,
            fit.y,
            fit.weights,
            retained.eta_unclipped,
            retained.eta,
            retained.mu,
        )
        if not self.held_against or not self.declared:
            return False
        if not np.isfinite(self.deviance(fit, retained)):
            return False
        note_held_against(self.rows)
        fit.curvature.newton_on_likelihood_rows()
        state = fit.evaluate_state(
            retained.beta,
            retained.intercept,
            phase="initial",
            iteration=it + 1,
            eta_unclipped=retained.eta_unclipped,
            centred_intercept=retained.centred_intercept,
        )
        fit.committed = fit.retained = state
        fit.eta_unclipped, fit.eta, fit.mu, fit.dev = (
            state.eta_unclipped,
            state.eta,
            state.mu,
            state.deviance,
        )
        fit.converged = False
        fit.stop_rule.restart()
        fit.recorder.count("irls_clip_likelihood_switches")
        self.unfinished = it + 1 >= fit.max_iter
        return not self.unfinished

    def final(
        self, converged: bool, termination_reason: TerminationReason
    ) -> tuple[bool, TerminationReason]:
        """The retained state's verdict where the clip held a row against its outcome.

        A fit left on the working rows with such a row stopped where the clip,
        not the likelihood, put it: never converged, and ``clip_held`` in place
        of the loop's own exit.  A fit that moved to the likelihood's rows with
        no iteration left to solve them ran out of budget.
        """
        if self.held_against and not self.likelihood:
            return False, "clip_held" if termination_reason in _LOOP_EXITS else termination_reason
        if self.unfinished:
            return False, "max_iter" if termination_reason == "converged" else termination_reason
        return converged, termination_reason
