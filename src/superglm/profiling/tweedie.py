"""Profile-likelihood estimation of the Tweedie power (Dunn & Smyth 2005).

At each candidate p the mean is refitted (warm-started) and phi is profiled
at that mean by `solve_log_phi`; the power is the bounded-Brent minimiser of
the resulting mean negative log-likelihood, and its interval inverts the
likelihood-ratio test on the same curve.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._tweedie import (
    NearPoissonDispersionError,
    PhiSolve,
    TweedieRows,
    solve_log_phi,
    weighted_deviance,
)
from superglm.distributions import Tweedie, clip_mu
from superglm.links import stabilize_eta
from superglm.model.base import (
    model_build_design_matrix,
    model_has_lambda1_targets,
    resolve_selection_penalty_for_fit,
)
from superglm.model.fit_ops import (
    _reject_monotone_fit_conflicts,
    _solve_coefficients,
    _uses_direct_solver,
)
from superglm.model.fit_state import configured_lambda2, configured_penalty
from superglm.profiling._scalar import (
    Interval,
    RecordedObjective,
    censoring_warnings,
    credible_lower_point,
    likelihood_ratio_interval,
    minimize_profile,
    profile_plot,
    warn_caller,
)
from superglm.reml.observed_geometry import ObservedModeNotCertifiedError

# Candidate REML fits only rank powers; the published refit at p_hat runs at the
# tight publication default. At this bar a candidate's mean NLL was within
# 2.5e-6 relative of a reml_tol = 1e-11 fit, typically 1e-10 to 1e-7, on the
# characterisation REML books at p in {1.3, 1.5, 1.7}.
_SEARCH_REML_TOL = 1e-6
# A recorded value's error per unit of its fit's certificate (_Candidate.error).
# A REML fit certifies its objective V to tol (1 + |V|), not the likelihood: the
# NLL errs through the smoothing parameters' residual error, two-sided and first
# order, by |J' H_V^-1| times the certificate, J = d(n nll)/d log lambda. That
# ratio has no bound in theory; it is measured: at most 5.4 over 198 candidate fits
# at reml_tol = 1e-6 against 1e-11 fits (five book shapes, 3k to 100k rows, p in
# {1.3, 1.5, 1.7}; benchmarks/tweedie_candidate_noise.py reproduces both ratios).
# A ratio set too low costs a spurious caution; one set too high can only widen
# an interval: a lower point the floor ignores leaves the centre at most the floor
# above the minimum, and {2 n (nll - nll_hat) <= cutoff} is then {2 n (nll -
# nll_min) <= cutoff + drop}, a superset of the interval from the minimum. At
# reml_tol = 1e-6 the floor at p_hat is 1.4 in the statistic on a smooth
# 100k-row book and 5.8 on the flat-lambda fixture at 100k rows.
_CANDIDATE_ERROR_RATIO = 8.0
# A fit without a REML objective certifies its penalized objective's relative
# change to the model's tol; the NLL errs first order through the penalty's
# gradient, by a ratio the solver's rate of convergence sets: fast for Fisher
# scoring (Osborne 1992), linear for proximal block descent (Tseng & Yun 2009).
# Replaying estimate_p's warm-started evaluations and continuing each fit to
# tol 1e-13 (1.2k to 100k rows): Fisher scoring, the route with no active
# selection penalty and no shape constraint, measured at most 9.0e-4, at the
# smallest tol that returns each fit. The proximal route of a selection penalty
# (0.21, tol 1e-4 to 1e-10) and a shape constraint's route (0.14 at tol 1e-6,
# rising to 6.7 at 1e-9) keep _CANDIDATE_ERROR_RATIO.
_SCORING_ERROR_RATIO = 2e-3
# The interval may reach past the default search bounds (1.05, 1.95), as
# master's did; the series is exact from p = 1.001 to 1.99 (its 50-digit oracle).
_CI_BOUNDS = (1.02, 1.98)
# Endpoints are reported to three decimals; master located them to the same 1e-4.
_CI_XTOL = 1e-4


def profile_phi_at(
    y: NDArray, mu: NDArray, weights: NDArray, p: float, *, grouping: Any = True
) -> PhiSolve:
    """Maximum-likelihood phi at a fitted mean: Q with M = 0.

    ``grouping`` is TweedieRows.profile_grouping(y, weights), which a search
    computes once for all its powers; True computes it here.
    """
    rows = TweedieRows.profile(y, weights, p, grouping=grouping)
    return solve_log_phi(rows, weighted_deviance(y, mu, p, weights))


@dataclass(frozen=True)
class _Candidate:
    phi: float
    pirls_converged: bool
    # A REML candidate also needs its smoothing-parameter iterations to settle.
    reml_converged: bool = True
    # Bound on the recorded mean NLL's error: the fit's certificate in mean-NLL
    # units times its measured ratio (_CANDIDATE_ERROR_RATIO, _SCORING_ERROR_RATIO);
    # an infeasible power has none.
    error: float = math.nan

    @property
    def fit_converged(self) -> bool:
        return self.pirls_converged and self.reml_converged


_INFEASIBLE_CANDIDATE = _Candidate(math.nan, False)


class _PowerProfile:
    """Mean NLL of the profile at a power: refit mu(p), then phi(p) at that mean."""

    def __init__(self, model, X, y, sample_weight, offset, fit_mode: str):
        X, y, sample_weight, offset = _snapshot_profile_inputs(X, y, sample_weight, offset)
        self.clone = _clone_profile_model(model, X, sample_weight)
        self.candidates: dict[float, _Candidate] = {}
        self.infeasible: dict[float, str] = {}
        self.on_evaluation: Callable[[dict], None] | None = None
        if fit_mode == "fit_reml":
            self._prepare_reml(X, y, sample_weight, offset)
        else:
            self._prepare_ml(X, y, sample_weight, offset)
        self.n = float(self.y.size)
        # Which rows repeat does not depend on p: sort once for the whole search.
        self.grouping = TweedieRows.profile_grouping(self.y, self.w)

    def __call__(self, p: float) -> float:
        try:
            mu, pirls_converged, reml = self._fit(p)
            solved = profile_phi_at(self.y, mu, self.w, p, grouping=self.grouping)
        except (ObservedModeNotCertifiedError, NearPoissonDispersionError) as exc:
            # A REML candidate whose penalized mode cannot be differentiated
            # through, or a power too close to 1 to profile phi globally, has no
            # objective to report, so the power is scored infeasible and the
            # search routes around it instead of failing on a point it did not need.
            self.infeasible[p] = str(exc).partition("\n")[0]
            return math.inf
        nll = solved.criterion / self.n
        # A fit's tolerance certifies its objective to tol (1 + |objective|): the
        # REML objective, or for a fit without one the likelihood itself, n nll.
        if reml is None:
            error = self._pirls_ratio() * self.clone._tol * (1.0 + abs(solved.criterion))
        else:
            error = _CANDIDATE_ERROR_RATIO * _SEARCH_REML_TOL * (1.0 + abs(reml.objective))
        candidate = _Candidate(
            solved.phi, pirls_converged, reml is None or bool(reml.converged), error / self.n
        )
        fit_converged = candidate.fit_converged
        self.candidates[p] = candidate
        if self.on_evaluation is not None:
            self.on_evaluation(
                {"p": p, "nll": nll, "phi": solved.phi, "fit_converged": fit_converged}
            )
        return nll

    def _pirls_ratio(self) -> float:
        """The error ratio of the route a fit without a REML objective took."""
        shaped = any(
            group.constraints is not None or group.monotone_engine == "scop"
            for group in self.clone._groups
        )
        return _CANDIDATE_ERROR_RATIO if shaped or self.selecting else _SCORING_ERROR_RATIO

    def _prepare_ml(self, X, y, sample_weight, offset) -> None:
        clone = self.clone
        # The design does not depend on p: build it once for every candidate.
        self.y, self.w, offset = model_build_design_matrix(clone, X, y, sample_weight, offset)
        self.penalty = configured_penalty(clone)
        resolve_selection_penalty_for_fit(clone, self.penalty, self.y, self.w)
        self.has_lambda1_targets = model_has_lambda1_targets(clone)
        _reject_monotone_fit_conflicts(clone, self.penalty, self.has_lambda1_targets)
        self.offset = np.zeros_like(self.y) if offset is None else offset
        self.warm_beta = self.warm_intercept = None
        # An active selection penalty sends every fit down the proximal route.
        self.selecting = not _uses_direct_solver(clone, self.penalty, self.has_lambda1_targets)
        self._fit = self._fit_ml

    def _fit_ml(self, p: float) -> tuple[NDArray, bool, None]:
        clone = self.clone
        clone._distribution = Tweedie(p)
        result = _solve_coefficients(
            clone,
            self.y,
            self.w,
            self.offset,
            penalty=self.penalty,
            lambda2=configured_lambda2(clone),
            has_lambda1_targets=self.has_lambda1_targets,
            max_iter=clone._max_iter,
            tol=clone._tol,
            record_diagnostics=False,
            convergence=clone._convergence,
            beta_init=self.warm_beta,
            intercept_init=self.warm_intercept,
        )
        self.warm_beta, self.warm_intercept = result.beta, result.intercept
        eta = clone._dm.matvec(result.beta) + result.intercept + self.offset
        mu = clip_mu(clone._link.inverse(stabilize_eta(eta, clone._link)), clone._distribution)
        return mu, bool(result.converged), None

    def _prepare_reml(self, X, y, sample_weight, offset) -> None:
        clone = self.clone
        # Candidate fits are ranked and discarded: skip the reporting tables,
        # and share one design build across candidates
        # (fit_ops._fetch_or_build_design) since the design does not depend on p.
        clone._suppress_reporting_support = True
        clone._profile_design_cache = {}
        self.X, self.y, self.w, self.offset = X, y, sample_weight, offset
        # fit_reml refuses a selection penalty.
        self.selecting = False
        self._fit = self._fit_reml

    def _fit_reml(self, p: float) -> tuple[NDArray, bool, Any]:
        clone = self.clone
        clone.family = Tweedie(p)
        # The post-fit runtime parity check certifies published state; candidate
        # fits are never published, and it cost 42% of every candidate on master.
        # No lambda warm start: the direct and discrete REML engines bootstrap
        # their own starting lambdas, so a previous candidate's lambdas leave
        # the fit bitwise unchanged.
        clone.fit_reml(
            self.X,
            self.y,
            sample_weight=self.w,
            offset=self.offset,
            runtime_validation="skip",
            reml_tol=_SEARCH_REML_TOL,
        )
        # The clone follows the model's retain_fit_state; a released fit keeps
        # its coefficients but not its fitted mean.
        mu = clone._fit_mu if clone._retain_fit_state else clone.predict(self.X, self.offset)
        # None for a model with no REML-eligible term: fit_reml is an ordinary fit.
        return mu, bool(clone.result.converged), clone._reml_result


def _clone_profile_model(model, X, sample_weight):
    """Clone configured profile state and resolve shorthand only on the clone."""
    profile_model = model._clone_without_features(
        set(),
        lambda2=copy.deepcopy(configured_lambda2(model)),
    )
    profile_model._interaction_specs = copy.deepcopy(model._interaction_specs)
    profile_model._interaction_order = list(model._interaction_order)
    profile_model._pending_interactions = copy.deepcopy(model._pending_interactions)
    if model._splines is not None and not model._specs:
        # clone_without_features() normally clones resolved specs. Preserve
        # unresolved shorthand metadata and resolve it only on the scratch model.
        profile_model._splines = copy.deepcopy(model._splines)
        profile_model._n_knots = copy.deepcopy(model._n_knots)
        profile_model._degree = model._degree
        profile_model._categorical_base = model._categorical_base
        profile_model._auto_detect_features(X, sample_weight)
    # Fit attempts rematerialize immutable constructor intent. The private
    # runtime copies above therefore need one matching configuration snapshot;
    # otherwise a resolved custom interaction falls back to its shorthand when
    # the profile model enters a transactional fit workspace.
    profile_model._config = type(profile_model._config).capture(profile_model)
    profile_model._config_revision += 1
    return profile_model


def _snapshot_profile_inputs(X, y, sample_weight, offset):
    """Own the inputs the result's lazy interval evaluations keep refitting."""
    return (
        copy.deepcopy(X),
        np.array(y, dtype=np.float64, copy=True),
        np.array(sample_weight, dtype=np.float64, copy=True),
        None if offset is None else np.array(offset, dtype=np.float64, copy=True),
    )


def search_power(
    model,
    X,
    y,
    sample_weight,
    offset,
    *,
    fit_mode: str,
    p_bounds: tuple[float, float] = (1.05, 1.95),
    xatol: float = 1e-3,
    maxiter: int = 30,
    on_evaluation: Callable[[dict], None] | None = None,
) -> TweedieProfileResult:
    """Bounded Brent on the profile mean NLL over ``p_bounds``, refitting mu at every p.

    ``fit_mode`` is ``"fit"`` or ``"fit_reml"``. Inputs are validated by the
    caller (``estimate_p``). ``on_evaluation`` receives each feasible search
    candidate as ``{"p", "nll", "phi", "fit_converged"}`` while the search runs.
    """
    profile = _PowerProfile(model, X, y, sample_weight, offset, fit_mode)
    objective = RecordedObjective(profile)
    profile.on_evaluation = on_evaluation
    search_converged = minimize_profile(objective, p_bounds, xatol=xatol, maxiter=maxiter)
    # The interval evaluates the same profile later, after the caller's
    # progress display has finished with the search.
    profile.on_evaluation = None
    if not profile.candidates:
        power, reason = next(iter(profile.infeasible.items()))
        # A REML mode that cannot be certified, or a power too close to 1 to
        # profile phi, under either fit mode.
        raise RuntimeError(
            f"No evaluated power in [{p_bounds[0]:.10g}, {p_bounds[1]:.10g}] was feasible; "
            f"the first refusal, at p={power:.10g}: {reason}"
        )
    return _searched_result(profile, objective, fit_mode, p_bounds, search_converged)


def _searched_result(profile, objective, fit_mode, p_bounds, search_converged):
    """The search's estimate, its evaluations in search order and its warnings."""
    p_hat, nll_hat = objective.best()
    best = profile.candidates[p_hat]
    candidates = [profile.candidates.get(p, _INFEASIBLE_CANDIDATE) for p in objective.values]
    evaluations = pd.DataFrame(
        {
            "p": list(objective.values),
            "nll": list(objective.values.values()),
            "phi": [candidate.phi for candidate in candidates],
            "fit_converged": [candidate.fit_converged for candidate in candidates],
        }
    )
    skipped, cautions = _search_warnings(objective.values, profile.infeasible, p_hat)
    winner_caution = _winner_caution(best, p_hat)
    if winner_caution is not None:
        # Disclosed with the estimate, not only once an interval is asked for.
        cautions.append(winner_caution)
    if not search_converged:
        # Brent ran out of steps: p_hat is the best power evaluated, not a located minimum.
        cautions.append(
            f"The power search stopped at its iteration limit; p_hat={p_hat:.6g} is the best "
            "evaluated power and the result reports converged=False."
        )
    # A caution about p_hat is raised as well as recorded: a record alone is
    # easily missed, and an unraised warning is no warning.
    for caution in cautions:
        warn_caller(caution)
    return TweedieProfileResult(
        p_hat=p_hat,
        phi_hat=best.phi,
        nll=nll_hat,
        converged=search_converged and best.fit_converged,
        _caution=winner_caution,
        fit_mode=fit_mode,
        search_fit_mode=fit_mode,
        evaluations=evaluations,
        warnings=skipped + cautions,
        search_nll=nll_hat,
        _objective=objective,
        _candidates=profile.candidates,
        _ll_scale=profile.n,
        _ci_bounds=_interval_bounds(p_hat, p_bounds),
    )


def _winner_caution(best: _Candidate, p_hat: float) -> str | None:
    """What the interval rests on when the winner's fit did not settle, per cause.

    The interval inverts the searched curve from its value at p_hat; a fit that
    stopped short can leave that value short of the optimum. Imperfect
    convergence is disclosed, never a reason to withhold the interval.
    """
    if not best.pirls_converged:
        return (
            f"p_hat={p_hat:.6g} rests on a coefficient fit that stopped at its iteration "
            "limit, and so does its interval; raise max_iter for a settled profile value there."
        )
    if not best.reml_converged:
        return (
            f"p_hat={p_hat:.6g} rests on a candidate REML fit whose smoothing parameters had "
            "not settled within the search's own budget, and so does its interval; "
            "search_fit_mode='fit' searches p under ML."
        )
    return None


def _search_warnings(
    values: dict[float, float], infeasible: dict[float, str], p_hat: float
) -> tuple[list[str], list[str]]:
    """Skipped powers, and cautions for every edge p_hat touches with nothing beyond it.

    Brent evaluates only inside the bounds, so p_hat is a search bound exactly
    when it is the first or last evaluated power. An infeasible power next to
    p_hat leaves the profile beyond it unknown as well.
    """
    skipped = [f"p={p:.6g} skipped as infeasible: {reason}" for p, reason in infeasible.items()]
    cautions = []
    ordered = sorted(values)
    index = ordered.index(p_hat)
    if index in (0, len(ordered) - 1):
        # Rounded responses can make the likelihood rise as p -> 1 (Dunn &
        # Smyth 2005): only an estimate on the lower bound can be that artefact.
        artefact = (
            " (a maximum as p -> 1 can be an artefact of rounded responses, Dunn & Smyth 2005)"
            if index == 0
            else ""
        )
        cautions.append(
            f"p_hat={p_hat:.6g} is at a search bound; the optimum may lie beyond it{artefact}."
        )
    cautions.extend(
        f"p_hat={p_hat:.6g} is next to p={p:.6g}, where the fit was infeasible; the optimum "
        "may lie beyond it, so p_hat is a censored estimate."
        for p in ordered[max(index - 1, 0) : index + 2]
        if math.isinf(values[p])
    )
    return skipped, cautions


def _interval_bounds(p_hat: float, p_bounds: tuple[float, float]) -> tuple[float, float]:
    """Where the interval may look for its endpoints.

    Past the search bounds, except on a side where p_hat sits on the bound:
    the profile beyond it was never searched, so that side is censored there.
    """
    lower = p_bounds[0] if p_hat == p_bounds[0] else min(_CI_BOUNDS[0], p_bounds[0])
    upper = p_bounds[1] if p_hat == p_bounds[1] else max(_CI_BOUNDS[1], p_bounds[1])
    return lower, upper


@dataclass
class TweedieProfileResult:
    """Profile-likelihood estimate of the Tweedie power p and dispersion phi.

    ``nll`` is the mean negative log-likelihood of the published fit and
    ``search_nll`` the searched profile's value at ``p_hat``, which the
    interval and the plot measure against. ``evaluations`` lists every searched
    power in order; an infeasible power has ``nll = inf``. ``fit_mode`` is the
    published fit's regime and ``search_fit_mode`` the searched profile's, which
    ``search_nll``, the interval and the plot describe. The interval is
    inverted from the lowest point of that curve, which is ``p_hat`` unless an
    interval's own evaluations found one lower by more than the fits' own
    tolerances resolve; a caution then says so.
    """

    p_hat: float
    phi_hat: float
    nll: float
    converged: bool
    fit_mode: str
    evaluations: pd.DataFrame
    warnings: list[str]
    search_nll: float
    _objective: Any = field(repr=False)
    _ll_scale: float = field(repr=False)
    _ci_bounds: tuple[float, float] = field(repr=False)
    _ci_cache: dict[float, Interval] = field(default_factory=dict, repr=False)
    search_fit_mode: str | None = None
    _caution: str | None = field(default=None, repr=False)
    # Each searched power's fit, including those an interval evaluates later.
    _candidates: dict[float, _Candidate] = field(default_factory=dict, repr=False)
    # Cautions about one interval: what its own evaluations found.
    _ci_cautions: dict[float, list[str]] = field(default_factory=dict, repr=False)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickle; a result saved by superglm 0.35.0 is restated in this layout.

        One path for a result pickled on its own (what ``estimate_p`` returns)
        and for one inside a saved model (`restate_v0_35_profile_result`).
        """
        if "search_trace" in state:
            restated = restate_v0_35_profile_result(state)
            self.__class__ = type(restated)
            state = vars(restated)
        self.__dict__.update(state)

    def interval(self, alpha: float = 0.05) -> Interval:
        """Likelihood-ratio interval for p on the searched curve, with censoring flags.

        A censored side, a winner whose fit did not settle, and what the
        interval's own evaluations found (a fit that did not settle, or a power
        below p_hat's value by more than the fits resolve) are recorded in
        ``warnings`` when first computed and warned about on every call.
        """
        interval = self._interval(alpha)
        for message in [self._caution] * (self._caution is not None) + self._interval_warnings(
            alpha
        ):
            warn_caller(message)
        return interval

    def _interval_warnings(self, alpha: float) -> list[str]:
        """What one computed interval adds to the estimate's own cautions."""
        alpha = float(alpha)
        censored = censoring_warnings(self._ci_cache[alpha], alpha, "p", self._stopped_at)
        return self._ci_cautions[alpha] + censored

    def _interval(self, alpha: float) -> Interval:
        """The interval, computed once and recorded; reports read it without a warning."""
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self._caution is not None and self._caution not in self.warnings:
            self.warnings.append(self._caution)
        if alpha not in self._ci_cache:
            searched = set(self._objective.values)
            interval, centre, centre_nll = likelihood_ratio_interval(
                self._objective,
                *self._centre(),
                self._ci_bounds,
                alpha=alpha,
                scale=self._ll_scale,
                xtol=_CI_XTOL,
                error=self._evaluation_error,
            )
            self._ci_cache[alpha] = interval
            self._ci_cautions[alpha] = self._evaluation_cautions(
                alpha, [p for p in self._objective.values if p not in searched], centre, centre_nll
            )
            self.warnings.extend(self._interval_warnings(alpha))
        return self._ci_cache[alpha]

    def _evaluation_error(self, p: float) -> float:
        """Bound on the error of the value recorded at p, from its fit's own certificate."""
        return self._candidates[p].error

    def _centre(self) -> tuple[float, float]:
        """Where an interval starts and the plot measures from: p_hat, or the lowest
        recorded power where that lies below p_hat by more than the fits resolve."""
        lower = credible_lower_point(
            self._objective, self.p_hat, self.search_nll, self._evaluation_error
        )
        return (self.p_hat, self.search_nll) if lower is None else lower

    def _evaluation_cautions(self, alpha, evaluated, centre, centre_nll) -> list[str]:
        """What an interval's own evaluations found: a lower power, or fits that did not settle."""
        level = f"{100.0 * (1.0 - alpha):g}%"
        cautions = []
        if centre != self.p_hat:
            drop = 2.0 * self._ll_scale * (self.search_nll - centre_nll)
            error = self._evaluation_error(self.p_hat) + self._evaluation_error(centre)
            cautions.append(
                f"the {level} interval's search found p={centre:.6g} below p_hat={self.p_hat:.6g} "
                f"on the searched curve, by {drop:.3g} in the likelihood-ratio statistic, more "
                f"than the {2.0 * self._ll_scale * error:.3g} its fits resolve: p_hat is a local "
                "minimum there, and the interval is inverted from that lower power."
            )
        lower = credible_lower_point(self._objective, centre, centre_nll, self._evaluation_error)
        if lower is not None:
            cautions.append(
                f"the {level} interval stopped re-centring with p={lower[0]:.6g} still below "
                f"p={centre:.6g} by more than its fits resolve; it is inverted from p={centre:.6g}, "
                "not from the lowest power found."
            )
        # An infeasible power has no fit to report on.
        fits = {p: self._candidates[p] for p in evaluated if p in self._candidates}
        by_cause = (
            (
                [p for p, fit in fits.items() if not fit.pirls_converged],
                "coefficient fits at p={} that stopped at their iteration limit; raise "
                "max_iter for settled profile values there.",
            ),
            (
                [p for p, fit in fits.items() if fit.pirls_converged and not fit.reml_converged],
                "candidate REML fits at p={} whose smoothing parameters had not settled within "
                "the search's own budget; search_fit_mode='fit' searches p under ML.",
            ),
        )
        for powers, cause in by_cause:
            if powers:
                listed = ", ".join(f"{p:.6g}" for p in sorted(powers))
                cautions.append(f"the {level} interval rests on " + cause.format(listed))
        return cautions

    def _stopped_at(self, end: float) -> str:
        return "a search bound" if end in self._ci_bounds else "next to an infeasible power"

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """``(lower, upper)`` of :meth:`interval`; a censored side is where its search stopped."""
        interval = self.interval(alpha)
        return interval.lower, interval.upper

    def profile_plot(self, alpha: float = 0.05, ax=None):
        """Likelihood-ratio statistic over every evaluated power, with any computed interval."""
        # Measured from the point the interval is inverted from: a lower value
        # within the fits' error moves neither.
        centre, centre_nll = self._centre()
        ax = profile_plot(
            self._objective.values,
            centre,
            centre_nll,
            scale=self._ll_scale,
            alpha=alpha,
            interval=self._ci_cache.get(float(alpha)),
            label="p",
            ax=ax,
        )
        if centre != self.p_hat:
            ax.axvline(self.p_hat, c="k", lw=1, ls=":")
        return ax


@dataclass
class SavedTweedieProfileResult(TweedieProfileResult):
    """A Tweedie power estimate saved by superglm v0.35.0, restated in the current layout.

    It keeps what v0.35.0 reported: ``p_hat``, ``phi_hat``, ``nll``, the
    searched powers (``evaluations``) and every interval computed before it
    was saved.  The profile search that would evaluate new powers was
    retired with v0.35.0's code, so an interval at another ``alpha`` needs a
    new ``estimate_p``; a value from the current engine is never mixed into
    v0.35.0's curve.
    """

    def _interval(self, alpha: float) -> Interval:
        alpha = float(alpha)
        if alpha not in self._ci_cache:
            raise RuntimeError(
                f"This Tweedie power estimate was saved by superglm 0.35.0, which computed "
                f"its interval only at alpha in {sorted(self._ci_cache)}; its profile search "
                "has been retired, so call estimate_p again to compute an interval at "
                f"alpha={alpha:g}."
            )
        return super()._interval(alpha)

    def _centre(self) -> tuple[float, float]:
        return self.p_hat, self.search_nll


def _retired_v0_35_search(p: float) -> float:
    """The objective of a restated v0.35.0 profile: its search cannot evaluate new powers."""
    raise RuntimeError(
        f"The profile search saved by superglm 0.35.0 has been retired; p={p:g} cannot be "
        "evaluated on it, so call estimate_p again."
    )


def restate_v0_35_profile_result(fields_: dict[str, Any]) -> SavedTweedieProfileResult:
    """The current layout of a ``TweedieProfileResult`` unpickled from v0.35.0.

    ``fields_`` is the pickled state.  v0.35.0 recorded the searched powers in
    ``search_trace``, each interval as a ``(lower, upper)`` tuple in
    ``_ci_cache`` and its endpoints' status in ``_ci_details_cache`` (now inert
    stand-ins, ``__getattr__`` below).  An endpoint is censored unless its
    status was ``"root_found"``, the only one v0.35.0 located as a
    likelihood-ratio crossing.
    """
    trace = fields_["search_trace"]
    values = dict(zip(trace["p"].astype(float), trace["nll"].astype(float), strict=True))
    objective = RecordedObjective(_retired_v0_35_search)
    objective.values.update(values)
    intervals: dict[float, Interval] = {}
    cautions: dict[float, list[str]] = {}
    details_cache = fields_.get("_ci_details_cache") or {}
    for alpha, bounds in (fields_.get("_ci_cache") or {}).items():
        details = getattr(details_cache.get(alpha), "_retired_state", None)
        if not isinstance(details, dict):
            continue
        status = [getattr(details[side], "_retired_state", {}) for side in ("lower", "upper")]
        intervals[float(alpha)] = Interval(
            float(bounds[0]),
            float(bounds[1]),
            status[0].get("status") != "root_found",
            status[1].get("status") != "root_found",
        )
        cautions[float(alpha)] = [str(message) for message in details.get("warnings", ())]
    search_nll = fields_.get("search_nll")
    ci_lower, ci_upper = fields_["_ci_p_range"]
    return SavedTweedieProfileResult(
        p_hat=float(fields_["p_hat"]),
        phi_hat=float(fields_["phi_hat"]),
        nll=float(fields_["nll"]),
        converged=bool(fields_["converged"]),
        fit_mode=str(fields_["fit_mode"]),
        search_fit_mode=fields_.get("search_fit_mode"),
        evaluations=pd.DataFrame(
            {
                "p": trace["p"].astype(float).to_numpy(),
                "nll": trace["nll"].astype(float).to_numpy(),
                "phi": trace["phi"].astype(float).to_numpy(),
                "fit_converged": trace["fit_converged"].astype(bool).to_numpy(),
            }
        ),
        warnings=list(fields_.get("warnings") or ()),
        search_nll=float(fields_["nll"] if search_nll is None else search_nll),
        _objective=objective,
        _ll_scale=float(fields_["_ll_scale"]),
        _ci_bounds=(float(ci_lower), float(ci_upper)),
        _ci_cache=intervals,
        _ci_cautions=cautions,
    )


# v0.35.0's profile search, density and interval classes, which a model saved
# after estimate_p pickles (its TweedieProfileResult binds the search's
# methods).  The names stay importable as inert stand-ins (PEP 562) so such a
# result loads, on its own or in a model; ``TweedieProfileResult.__setstate__``
# restates it (``restate_v0_35_profile_result``).
_RETIRED_V0_35 = frozenset(
    {
        "TweedieProfileCIDensityProvenance",
        "TweedieProfileCIDetails",
        "TweedieProfileCIEndpoint",
        "TweedieProfileCIEvaluation",
        "_CIDensityAggregate",
        "_CPGRNG",
        "_DensitySummary",
        "_ExactPhiNewtonOutcome",
        "_PhiBoundedResult",
        "_PhiBranchMask",
        "_PhiCandidate",
        "_PhiEvaluationCache",
        "_PhiProfilePoint",
        "_PhiProfileResult",
        "_PhiScoreSearchResult",
        "_PreparedTweedieDensity",
        "_ProfileContext",
        "_ProfileContextREML",
        "_ProfileEvaluation",
        "_TweedieDensityEvaluation",
        "_TweedieLogpdfDiagnostics",
    }
)


def __getattr__(name: str):
    if name in _RETIRED_V0_35:
        from superglm.solvers._structured.retired import RetiredSearchState, retired_class

        return retired_class(__name__, name, RetiredSearchState)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
