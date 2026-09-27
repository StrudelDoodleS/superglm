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

from superglm._tweedie import PhiSolve, TweedieRows, solve_log_phi, tweedie_unit_deviance
from superglm.distributions import Tweedie, clip_mu
from superglm.links import stabilize_eta
from superglm.model.base import (
    model_build_design_matrix,
    model_has_lambda1_targets,
    resolve_selection_penalty_for_fit,
)
from superglm.model.fit_ops import _reject_monotone_fit_conflicts, _solve_coefficients
from superglm.model.fit_state import configured_lambda2, configured_penalty
from superglm.profiling._scalar import (
    Interval,
    RecordedObjective,
    likelihood_ratio_interval,
    minimize_profile,
    profile_plot,
)
from superglm.reml.observed_geometry import ObservedModeNotCertifiedError

# Candidate REML fits only rank powers. On flat-lambda designs this bar leaves
# the candidate NLL determined only to ~4e-4 relative, but the ranking holds:
# p_hat agreed with tight-bar searches to ~1e-11 on the benchmark fixture. The
# published refit at p_hat runs at the tight publication default.
_SEARCH_REML_TOL = 1e-6
# The interval may reach past the default search bounds (1.05, 1.95), as
# master's did; the series is exact from p = 1.01 to 1.99 (its 50-digit oracle).
_CI_BOUNDS = (1.02, 1.98)
# Endpoints are reported to three decimals; master located them to the same 1e-4.
_CI_XTOL = 1e-4


def profile_phi_at(y: NDArray, mu: NDArray, weights: NDArray, p: float) -> PhiSolve:
    """Maximum-likelihood phi at a fitted mean: Q with M = 0."""
    deviance = float(np.sum(weights * tweedie_unit_deviance(y, mu, p)))
    return solve_log_phi(TweedieRows.prepare(y, weights, p), deviance)


@dataclass(frozen=True)
class _Candidate:
    phi: float
    fit_converged: bool


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

    def __call__(self, p: float) -> float:
        try:
            mu, fit_converged = self._fit(p)
        except ObservedModeNotCertifiedError as exc:
            # A REML candidate whose penalized mode cannot be differentiated
            # through has no objective to report. Conditioning worsens toward
            # p = 2, so the power is scored infeasible and the search routes
            # around it instead of failing on a point it did not need.
            self.infeasible[p] = str(exc).partition("\n")[0]
            return math.inf
        solved = profile_phi_at(self.y, mu, self.w, p)
        nll = solved.criterion / self.n
        self.candidates[p] = _Candidate(solved.phi, fit_converged)
        if self.on_evaluation is not None:
            self.on_evaluation(
                {"p": p, "nll": nll, "phi": solved.phi, "fit_converged": fit_converged}
            )
        return nll

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
        self._fit = self._fit_ml

    def _fit_ml(self, p: float) -> tuple[NDArray, bool]:
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
        return mu, bool(result.converged)

    def _prepare_reml(self, X, y, sample_weight, offset) -> None:
        clone = self.clone
        # Candidate fits are ranked and discarded: keep compact state, skip the
        # reporting tables, and share one design build across candidates
        # (fit_ops._fetch_or_build_design) since the design does not depend on p.
        clone._retain_fit_state = False
        clone._suppress_reporting_support = True
        clone._profile_design_cache = {}
        self.X, self.y, self.sample_weight, self.offset = X, y, sample_weight, offset
        self.w = np.ones_like(y) if sample_weight is None else sample_weight
        self._fit = self._fit_reml

    def _fit_reml(self, p: float) -> tuple[NDArray, bool]:
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
            sample_weight=self.sample_weight,
            offset=self.offset,
            runtime_validation="skip",
            reml_tol=_SEARCH_REML_TOL,
        )
        reml = clone._reml_result
        # A model with no REML-eligible term makes fit_reml an ordinary fit.
        converged = clone.result.converged and (reml is None or reml.converged)
        return clone._fit_mu, bool(converged)


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
        (None if sample_weight is None else np.array(sample_weight, dtype=np.float64, copy=True)),
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
        raise RuntimeError(
            "REML could not certify a penalized coefficient mode at any evaluated power in "
            f"[{p_bounds[0]:.6g}, {p_bounds[1]:.6g}]; the first refusal, at p={power:.6g}: {reason}"
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
    return TweedieProfileResult(
        p_hat=p_hat,
        phi_hat=best.phi,
        nll=nll_hat,
        converged=search_converged and best.fit_converged,
        fit_mode=fit_mode,
        evaluations=evaluations,
        warnings=_search_warnings(objective.values, profile.infeasible, p_hat),
        search_nll=nll_hat,
        _objective=objective,
        _ll_scale=profile.n,
        _ci_bounds=_interval_bounds(p_hat, p_bounds),
    )


def _search_warnings(values: dict[float, float], infeasible: dict[float, str], p_hat: float):
    """Skipped powers, and every edge p_hat touches with nothing evaluated in between.

    Brent evaluates only inside the bounds, so p_hat is a search bound exactly
    when it is the first or last evaluated power. An infeasible power next to
    p_hat leaves the profile beyond it unknown as well.
    """
    warnings = [f"p={p:.6g} skipped as infeasible: {reason}" for p, reason in infeasible.items()]
    ordered = sorted(values)
    index = ordered.index(p_hat)
    if index in (0, len(ordered) - 1):
        warnings.append(
            f"p_hat={p_hat:.6g} is at a search bound; the optimum may lie beyond it "
            "(a maximum as p -> 1 can be an artefact of rounded responses, Dunn & Smyth 2005)."
        )
    warnings.extend(
        f"p_hat={p_hat:.6g} is next to p={p:.6g}, where the fit was infeasible; the optimum "
        "may lie beyond it, so p_hat is a censored estimate."
        for p in ordered[max(index - 1, 0) : index + 2]
        if math.isinf(values[p])
    )
    return warnings


def _interval_bounds(p_hat: float, p_bounds: tuple[float, float]) -> tuple[float, float]:
    """Where the interval may look for its endpoints.

    Past the search bounds, except on a side where p_hat sits on the bound:
    the profile beyond it was never searched, so that side is censored there.
    """
    lower = p_bounds[0] if p_hat == p_bounds[0] else min(_CI_BOUNDS[0], p_bounds[0])
    upper = p_bounds[1] if p_hat == p_bounds[1] else max(_CI_BOUNDS[1], p_bounds[1])
    return lower, upper


def _censoring_warnings(interval: Interval, alpha: float, bounds: tuple[float, float]):
    """One warning per censored side, naming where the interval stopped and why."""
    sides = (
        ("lower", interval.lower, interval.lower_censored),
        ("upper", interval.upper, interval.upper_censored),
    )
    return [
        f"the {100.0 * (1.0 - alpha):g}% interval for p is censored at its {side} end "
        f"p={end:.6g}, "
        + ("a search bound" if end in bounds else "next to an infeasible power")
        + ": the likelihood-ratio statistic does not reach its cutoff there, so the "
        "interval may extend beyond it."
        for side, end, censored in sides
        if censored
    ]


@dataclass
class TweedieProfileResult:
    """Profile-likelihood estimate of the Tweedie power p and dispersion phi.

    ``nll`` is the mean negative log-likelihood of the published fit and
    ``search_nll`` the searched profile's value at ``p_hat``, which the
    interval and the plot measure against. ``evaluations`` lists every searched
    power in order; an infeasible power has ``nll = inf``.
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

    def interval(self, alpha: float = 0.05) -> Interval:
        """Likelihood-ratio interval for p on the searched curve, with censoring flags.

        A censored side is also recorded in ``warnings`` when it is computed.
        """
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if alpha not in self._ci_cache:
            interval = likelihood_ratio_interval(
                self._objective,
                self.p_hat,
                self.search_nll,
                self._ci_bounds,
                alpha=alpha,
                scale=self._ll_scale,
                xtol=_CI_XTOL,
            )
            self._ci_cache[alpha] = interval
            self.warnings.extend(_censoring_warnings(interval, alpha, self._ci_bounds))
        return self._ci_cache[alpha]

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """``(lower, upper)`` of :meth:`interval`; a censored side is where its search stopped."""
        interval = self.interval(alpha)
        return interval.lower, interval.upper

    def profile_plot(self, alpha: float = 0.05, ax=None):
        """Likelihood-ratio statistic over every evaluated power, with any computed interval."""
        return profile_plot(
            self._objective.values,
            self.p_hat,
            self.search_nll,
            scale=self._ll_scale,
            alpha=alpha,
            interval=self._ci_cache.get(float(alpha)),
            label="p",
            ax=ax,
        )
