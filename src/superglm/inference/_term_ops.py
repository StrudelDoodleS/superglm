"""Operational term-inference assembly entry points."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from superglm.features.constraint import rules_of
from superglm.inference._term_covariance import (
    as_determinacy,
    feature_se_from_cov,
    piecewise_knot_covariance,
    simultaneous_band_and_critical_value,
)
from superglm.inference._term_helpers import (
    _VALID_CENTERING,
    _build_spline_metadata,
    _compute_term_edf,
    _expand_grouped_term,
    _recenter_term,
    _resolve_term_lambda,
    _spline_se,
    spline_groups,
)
from superglm.inference._term_interactions import _interaction_inference
from superglm.inference._term_types import (
    InteractionInference,
    SmoothCurve,
    TermInference,
    _safe_exp,
)
from superglm.solvers.rank import selected_group_name_set

if TYPE_CHECKING:
    from superglm.solvers.pirls import PIRLSResult
    from superglm.types import GroupSlice


def _maybe_array(value: NDArray | float | None) -> NDArray | None:
    """Normalize optional scalar-or-array results to ndarray for typed dataclasses."""
    if value is None:
        return None
    return cast(NDArray, np.asarray(value))


def _centered_se(
    centering: str,
    name: str,
    Cov_active,
    active_groups_cov,
    result: PIRLSResult,
    groups: list[GroupSlice],
    specs: dict[str, Any],
    interaction_specs: dict[str, Any],
    *,
    n_points: int = 200,
    flags=None,
    determines=None,
) -> NDArray | None:
    """The errors of the mean-centered report, or ``None`` when not centering.

    ``centering="mean"`` reports the estimable contrast ``C b`` rather than
    ``b``, so its errors come from ``C V C'`` and cannot be recovered from the
    against-base errors the native report carries.  Computed here, beside the
    covariance, and handed to ``_recenter_term``, which has no access to it.
    """
    if centering != "mean" or Cov_active is None:
        return None
    return feature_se_from_cov(
        name,
        Cov_active,
        active_groups_cov,
        result,
        groups,
        specs,
        interaction_specs,
        n_points=n_points,
        center=True,
        flags=flags,
        determines=determines,
    )


@cache
def _term_reports() -> tuple[tuple[type, Any], ...]:
    """Each main-effect class's report, in the order the classes are tested."""
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.piecewise import Piecewise
    from superglm.features.polynomial import Polynomial
    from superglm.features.random_effect import RandomEffect
    from superglm.features.spline import _SplineBase

    return (
        (RandomEffect, _TermReport.random_effect),
        (OrderedCategorical, _TermReport.ordered),
        (_SplineBase, _TermReport.spline),
        (Categorical, _TermReport.categorical),
        (Polynomial, _TermReport.polynomial),
        (Numeric, _TermReport.numeric),
        (Piecewise, _TermReport.piecewise),
    )


@dataclass(kw_only=True, eq=False)
class _TermReport:
    """One main effect's ``term_inference``: its arguments, and one report per feature class."""

    name: str
    spec: Any
    result: PIRLSResult
    groups: list[GroupSlice]
    specs: dict[str, Any]
    interaction_specs: dict[str, Any]
    covariance_fn: Any
    reml_lambdas: dict[str, float] | None
    lambda2: float
    group_edf: dict[str, float] | None
    with_se: bool
    simultaneous: bool
    n_points: int
    alpha: float
    n_sim: int
    seed: int
    centering: str
    determines: Any

    def __post_init__(self) -> None:
        import scipy.spatial
        from scipy.stats import norm

        name, groups = self.name, self.groups
        self.beta = self.result.beta
        self.feature_groups = [g for g in groups if g.feature_name == name]
        self.beta_combined = np.concatenate([self.beta[g.sl] for g in self.feature_groups])
        # Deliberately FEATURE-wide, unlike the smooth row's `active` in coef_tables and the
        # export: this one gates the level table, and a free special is genuinely still
        # fitted when the spline block is dropped. The smooth's own survival is
        # `smooth_active`.
        self.selected_names = selected_group_name_set(self.result, groups)
        self.active = any(group.name in self.selected_names for group in self.feature_groups)
        self.smooth_active = any(
            group.name in self.selected_names for group in spline_groups(self.feature_groups)
        )
        self.Cov_active = self.active_groups_cov = None
        if self.with_se and self.active:
            self.Cov_active, self.active_groups_cov = self.covariance_fn()
        # Per-group edf, over the SMOOTH's blocks, as coef_tables and report_ops scope it:
        # TermInference.edf is the copy the editor context bar and the plot data read.
        self.edf = _compute_term_edf(name, spline_groups(self.feature_groups), self.group_edf)
        self.lam = _resolve_term_lambda(name, self.feature_groups, self.reml_lambdas, self.lambda2)
        self.z_alpha = float(norm.ppf(1.0 - self.alpha / 2.0))

    def inference(self) -> TermInference:
        for kind, report in _term_reports():
            if isinstance(self.spec, kind):
                return report(self)
        raise TypeError(f"Unknown feature type: {type(self.spec).__name__}")

    # ── Errors ────────────────────────────────────────────────────

    def feature_errors(self, **options: Any) -> NDArray | None:
        """The term's standard errors (``feature_se_from_cov``), or ``None`` without them.

        NaN where ``determines`` leaves a reported value open.
        """
        if not (self.with_se and self.active and self.Cov_active is not None):
            return None
        assert self.active_groups_cov is not None
        return feature_se_from_cov(
            self.name,
            self.Cov_active,
            self.active_groups_cov,
            self.result,
            self.groups,
            self.specs,
            self.interaction_specs,
            determines=self.determines,
            **options,
        )

    def interval(self, log_rel: NDArray, se: NDArray | None) -> tuple[Any, Any]:
        """The pointwise interval on the relativity scale, or ``None`` without errors."""
        if se is None:
            return None, None
        return (
            _maybe_array(_safe_exp(log_rel - self.z_alpha * se)),
            _maybe_array(_safe_exp(log_rel + self.z_alpha * se)),
        )

    def centred_errors(self, **options: Any) -> NDArray | None:
        """The errors of the mean-centred report (``_centered_se``)."""
        return _centered_se(
            self.centering,
            self.name,
            self.Cov_active,
            self.active_groups_cov,
            self.result,
            self.groups,
            self.specs,
            self.interaction_specs,
            determines=self.determines,
            **options,
        )

    # ── Reports ───────────────────────────────────────────────────

    def random_effect(self) -> TermInference:
        raw = self.spec.reconstruct(self.beta_combined)
        levels = raw["levels"]
        log_rels = np.asarray([raw["effects"][level] for level in levels])
        se = self.feature_errors()
        ci_lo, ci_hi = self.interval(log_rels, se)
        return _recenter_term(
            TermInference(
                name=self.name,
                kind="categorical",
                active=self.active,
                levels=levels,
                log_relativity=log_rels,
                relativity=np.exp(log_rels),
                se_log_relativity=se,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                absorbs_intercept=False,
                centering_mode="population_zero",
                edf=self.edf,
                smoothing_lambda=self.lam,
                alpha=self.alpha,
            ),
            self.centering,
            se_centered=self.centred_errors(),
        )

    def ordered(self) -> TermInference:
        """K levels with SEs against the base level, and on a spline basis the fitted curve.

        A step-mode pickle reaches here and raises the migration error from
        ``spec.reconstruct``.  Specials are free levels with no position on the spline
        axis: they stay out of ``level_x`` and are flagged for the renderers instead,
        matched on the labels ``reconstruct`` exported, not on ``spec._specials``, which is
        string-coerced (``specials=[9]`` holds "9" there and 9 in ``levels``).  The base
        level is already at 0/1, so the report is not recentred.
        """
        from superglm.features.factor_smooth import with_population_curve
        from superglm.features.spline import _SplineBase

        spec, name = self.spec, self.name
        inner = spec._spline
        raw = with_population_curve(
            spec.reconstruct(self.beta_combined),
            name,
            self.interaction_specs,
            self.groups,
            self.beta,
            spec,
        )
        levels = raw["levels"]
        level_log_rels = np.array([raw["level_log_relativities"][lv] for lv in levels])
        special_labels = set(raw.get("special_levels") or ())
        se = self.feature_errors()
        ci_lo, ci_hi = self.interval(level_log_rels, se)
        curve = (
            self.ordered_curve(raw, [lv for lv in levels if lv not in special_labels])
            if self.smooth_active
            else None
        )
        report = TermInference(
            name=name,
            kind="categorical",
            active=self.active,
            levels=levels,
            log_relativity=level_log_rels,
            relativity=np.array([raw["level_relativities"][lv] for lv in levels]),
            se_log_relativity=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            absorbs_intercept=False,
            centering_mode="base_level",
            edf=self.edf,
            smoothing_lambda=self.lam,
            smooth_curve=curve,
            level_is_special=(
                np.array([lv in special_labels for lv in levels], dtype=bool)
                if special_labels
                else None
            ),
            # A Piecewise or Polynomial inner basis has no knot strategy or B-spline degree;
            # the editor reads None as "no spline metadata", which is exactly true.
            spline=_build_spline_metadata(inner) if isinstance(inner, _SplineBase) else None,
            monotone=rules_of(spec).describe() or None,
            alpha=self.alpha,
        )
        if spec._grouping is None:
            return report
        return _expand_grouped_term(report, spec._grouping, spec._original_level_to_value)

    def ordered_curve(self, raw: dict, smooth_levels: list) -> SmoothCurve:
        """An ordered spline's fitted curve, with its errors when the term carries them.

        The curve is a statement about the spline block alone, so its errors are too: the
        free-level block is filtered out of the covariance groups.
        """
        spec = self.spec
        inner = spec._spline
        level_x = np.array([raw["level_values"][lv] for lv in smooth_levels])
        if not (self.with_se and self.Cov_active is not None):
            return SmoothCurve(
                x=raw["x"],
                log_relativity=raw["log_relativity"],
                relativity=raw["relativity"],
                level_x=level_x,
            )
        assert self.active_groups_cov is not None
        smooth_groups = spline_groups(self.feature_groups)
        smooth_active_cov = [
            ag
            for ag in self.active_groups_cov
            if not (ag.feature_name == self.name and ag.subgroup_type == "special")
        ]
        reference_x = np.array([raw["level_values"][spec._base_level]], dtype=np.float64)
        curve_se = _spline_se(
            inner,
            self.name,
            smooth_groups,
            smooth_active_cov,
            self.Cov_active,
            x_eval=raw["x"],
            reference_x=reference_x,
            determines=self.determines,
        )
        ci_lo, ci_hi = self.interval(raw["log_relativity"], curve_se)
        return SmoothCurve(
            x=raw["x"],
            log_relativity=raw["log_relativity"],
            relativity=raw["relativity"],
            level_x=level_x,
            se_log_relativity=curve_se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
        )

    def spline(self) -> TermInference:
        from superglm.features.factor_smooth import with_population_curve

        spec, name = self.spec, self.name
        raw = spec.reconstruct(self.beta_combined, n_points=self.n_points)
        # the population curve when an sz term moves it, as its errors are
        raw = with_population_curve(raw, name, self.interaction_specs, self.groups, self.beta, spec)
        log_rel = raw["log_relativity"]
        se = self.feature_errors(n_points=self.n_points)
        # _recenter_term leaves a one-point report at its native value.
        se_centered = self.centred_errors(n_points=self.n_points) if len(log_rel) > 1 else None
        ci_lo, ci_hi = self.interval(log_rel, se)
        ci_lo_sim = ci_hi_sim = c_sim = None
        # A simultaneous band is one statement about the whole curve, the centred one
        # under ``centering="mean"``: none where the covariance leaves a point of it open.
        banded = se if se_centered is None else se_centered
        if banded is not None and self.simultaneous and not np.isnan(banded).any():
            bands, c_sim = simultaneous_band_and_critical_value(
                name,
                result=self.result,
                groups=self.groups,
                specs=self.specs,
                interaction_specs=self.interaction_specs,
                covariance_fn=self.covariance_fn,
                alpha=self.alpha,
                n_sim=self.n_sim,
                n_points=self.n_points,
                seed=self.seed,
                center=se_centered is not None,
            )
            ci_lo_sim = bands["ci_lower_simultaneous"].values
            ci_hi_sim = bands["ci_upper_simultaneous"].values
        return _recenter_term(
            TermInference(
                name=name,
                kind="spline",
                active=self.active,
                x=raw["x"],
                log_relativity=log_rel,
                relativity=raw["relativity"],
                se_log_relativity=se,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                ci_lower_simultaneous=ci_lo_sim,
                ci_upper_simultaneous=ci_hi_sim,
                critical_value_simultaneous=c_sim,
                absorbs_intercept=spec.absorbs_intercept,
                edf=self.edf,
                smoothing_lambda=self.lam,
                spline=_build_spline_metadata(spec),
                monotone=rules_of(spec).describe() or None,
                monotone_repaired=False,  # caller can override if repairs exist
                alpha=self.alpha,
            ),
            self.centering,
            se_centered=se_centered,
        )

    def categorical(self) -> TermInference:
        """One row per level; the separated levels and the levels a mean-centred report
        centres on come from one reading of the fit's separation record."""
        from superglm.features.factor_smooth import with_population_curve
        from superglm.model.separated_levels import level_flags, separation_flags

        spec, name = self.spec, self.name
        raw = with_population_curve(
            spec.reconstruct(self.beta_combined),
            name,
            self.interaction_specs,
            self.groups,
            self.beta,
            spec,
        )
        levels = raw["levels"]
        log_rels = np.array([raw["log_relativities"][lv] for lv in levels])
        flags = separation_flags(self.groups, self.specs, self.interaction_specs)
        separated, centre_on = level_flags(
            self.groups,
            self.specs,
            self.interaction_specs,
            name,
            levels,
            centering=self.centering,
            flags=flags,
        )
        se = self.feature_errors(flags=flags)
        ci_lo, ci_hi = self.interval(log_rels, se)
        report = _recenter_term(
            TermInference(
                name=name,
                kind="categorical",
                active=self.active,
                levels=levels,
                log_relativity=log_rels,
                relativity=np.array([raw["relativities"][lv] for lv in levels]),
                se_log_relativity=se,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                absorbs_intercept=False,
                centering_mode="base_level",
                edf=self.edf,
                smoothing_lambda=self.lam,
                monotone=rules_of(spec).describe() or None,
                alpha=self.alpha,
                separated=separated,
            ),
            self.centering,
            centre_on=centre_on,
            se_centered=self.centred_errors(flags=flags),
        )
        if spec._grouping is None:
            return report
        return _expand_grouped_term(report, spec._grouping)

    def polynomial(self) -> TermInference:
        # Reconstructed at the caller's resolution, as ``feature_se_from_cov`` reads it, so
        # the curve and its errors have one length at every ``n_points``.
        raw = self.spec.reconstruct(self.beta_combined, n_points=self.n_points)
        log_rel = raw["log_relativity"]
        se = self.feature_errors(n_points=self.n_points)
        ci_lo, ci_hi = self.interval(log_rel, se)
        return _recenter_term(
            TermInference(
                name=self.name,
                kind="polynomial",
                active=self.active,
                x=raw["x"],
                log_relativity=log_rel,
                relativity=raw["relativity"],
                se_log_relativity=se,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                absorbs_intercept=True,
                edf=self.edf,
                smoothing_lambda=self.lam,
                alpha=self.alpha,
            ),
            self.centering,
            se_centered=self.centred_errors(n_points=self.n_points),
        )

    def numeric(self) -> TermInference:
        raw = self.spec.reconstruct(self.beta_combined)
        log_rel = np.array([raw["coef"]])
        se = self.feature_errors()
        ci_lo = ci_hi = None
        if se is not None:
            # Unclipped, as ``reconstruct`` exponentiates the point, so the interval
            # contains the point out to float64's range.
            with np.errstate(over="ignore", under="ignore"):
                ci_lo = np.exp(log_rel - self.z_alpha * se)
                ci_hi = np.exp(log_rel + self.z_alpha * se)
        return _recenter_term(
            TermInference(
                name=self.name,
                kind="numeric",
                active=self.active,
                log_relativity=log_rel,
                relativity=np.array([raw["relativity_per_unit"]]),
                se_log_relativity=se,
                ci_lower=_maybe_array(ci_lo),
                ci_upper=_maybe_array(ci_hi),
                absorbs_intercept=False,
                centering_mode="none",
                edf=self.edf,
                smoothing_lambda=self.lam,
                alpha=self.alpha,
            ),
            self.centering,
        )

    def piecewise(self) -> TermInference:
        """The term at its knots, with the full per-knot covariance for the band between them.

        ``sqrt(diag)`` of the knot covariance IS the per-knot SE (the same basis map as
        ``feature_se_from_cov``'s Piecewise branch), so the two stay one computation.  It
        is also why no ``se_centered`` is passed: ``_recenter_term`` transforms the
        published covariance and reads the centred errors off its diagonal.
        """
        from superglm.inference.covariance import BinaryScaledCovariance

        raw = self.spec.reconstruct(self.beta_combined)
        knots = raw["knots"]
        log_rel = raw["log_relativity"]
        Cov_active, exponent = self.Cov_active, 0
        if isinstance(Cov_active, BinaryScaledCovariance):
            Cov_active, exponent = Cov_active.base, Cov_active.exponent
        se = knot_cov = None
        if self.with_se and self.active and Cov_active is not None:
            assert self.active_groups_cov is not None
            knot_cov = piecewise_knot_covariance(
                self.name, Cov_active, self.active_groups_cov, self.specs
            )
            se = (
                np.sqrt(np.maximum(np.diag(knot_cov), 0.0))
                if knot_cov is not None
                else np.zeros(knots.size)
            )
        ci_lo, ci_hi = self.interval(log_rel, se)
        inference = _recenter_term(
            TermInference(
                name=self.name,
                kind="piecewise",
                active=self.active,
                # The knot vector itself, not a display grid: a piecewise term is exactly
                # determined by its values at the knots, so the curve, the editor handles
                # and the workbook rows are the same J+2 points.
                x=knots,
                log_relativity=log_rel,
                relativity=_maybe_array(_safe_exp(log_rel)),
                se_log_relativity=se,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                knot_covariance=knot_cov,
                absorbs_intercept=False,
                centering_mode="base_knot",
                # The MEASURED edf, not the nominal J+1, which holds only while the group
                # carries no shrinkage; J+1 is for callers that asked for no edf at all.
                edf=self.edf if self.edf is not None else float(self.spec._non_base_indices.size),
                # No smoothing lambda, never the global lambda2 fallback: the group carries
                # no penalty matrix, and selection shrinkage is not a smoothing lambda.
                smoothing_lambda=None,
                alpha=self.alpha,
            ),
            self.centering,
        )
        if not exponent or inference.se_log_relativity is None:
            return inference
        errors = np.ldexp(inference.se_log_relativity, exponent // 2)
        with np.errstate(over="ignore"):
            knot_cov = np.ldexp(inference.knot_covariance, exponent)
        return replace(
            inference,
            se_log_relativity=errors,
            knot_covariance=knot_cov,
            ci_lower=_safe_exp(inference.log_relativity - self.z_alpha * errors),
            ci_upper=_safe_exp(inference.log_relativity + self.z_alpha * errors),
        )


# ── Main Entry Point ──────────────────────────────────────────────


def term_inference(
    name: str,
    *,
    result: PIRLSResult,
    groups: list[GroupSlice],
    specs: dict[str, Any],
    interaction_specs: dict[str, Any],
    covariance_fn,
    reml_lambdas: dict[str, float] | None,
    lambda2: float,
    group_edf: dict[str, float] | None = None,
    with_se: bool = True,
    simultaneous: bool = False,
    n_points: int = 200,
    alpha: float = 0.05,
    n_sim: int = 10_000,
    seed: int = 42,
    centering: str = "native",
    determines=None,
) -> TermInference | InteractionInference:
    """Build a per-term inference object.

    Parameters
    ----------
    name : str
        Feature or interaction name.
    result : PIRLSResult
        Fitted model result.
    groups : list[GroupSlice]
        Group definitions from the fitted model.
    specs, interaction_specs : dict
        Feature and interaction specs.
    covariance_fn : callable
        Zero-arg callable returning ``(Cov_active, active_groups)``.
    reml_lambdas : dict or None
        REML-estimated per-group lambdas (from model._reml_lambdas).
    lambda2 : float
        Global smoothing penalty.
    group_edf : dict[str, float] or None
        Per-group effective degrees of freedom (keyed by group name).
    with_se : bool
        Compute standard errors and pointwise CIs.
    simultaneous : bool
        Compute simultaneous bands (spline only, requires with_se).
    n_points : int
        Grid size for spline/polynomial curves.
    alpha : float
        Significance level for CIs.
    n_sim : int
        Number of simulations for simultaneous bands.
    seed : int
        Random seed for simultaneous bands.
    centering : {"native", "mean"}
        ``"native"`` (default) returns the canonical fitted term
        contribution under the model's identifiability constraint.
        ``"mean"`` is a reporting convenience that shifts so the
        geometric mean of relativities = 1.  The shift is a function of the
        same fitted coefficients, so the errors and intervals are propagated
        through the centering contrast rather than translated with the
        values -- see ``_recenter_term``.
    determines : Determinacy, callable or None
        The covariance's rule for a contrast over the fitted coefficients
        (``contrast_determined``); the errors of a spline's curve and of an ordered term's
        levels and curve that it does not determine are NaN.  A callable, the earlier
        form, decides each contrast itself.  ``None`` reports every one.

    Returns
    -------
    TermInference or InteractionInference
    """
    if centering not in _VALID_CENTERING:
        raise ValueError(f"centering must be one of {_VALID_CENTERING}, got {centering!r}")
    if name in specs and name in interaction_specs:
        raise ValueError(
            f"Ambiguous name {name!r}: exists as both a main effect "
            f"and an interaction. Use the feature or interaction spec "
            f"directly to disambiguate."
        )
    if name in interaction_specs:
        return _interaction_inference(
            name,
            result=result,
            groups=groups,
            interaction_specs=interaction_specs,
            specs=specs,
        )
    spec = specs.get(name)
    if spec is None:
        raise KeyError(f"Feature not found: {name}")
    return _TermReport(
        name=name,
        spec=spec,
        result=result,
        groups=groups,
        specs=specs,
        interaction_specs=interaction_specs,
        covariance_fn=covariance_fn,
        reml_lambdas=reml_lambdas,
        lambda2=lambda2,
        group_edf=group_edf,
        with_se=with_se,
        simultaneous=simultaneous,
        n_points=n_points,
        alpha=alpha,
        n_sim=n_sim,
        seed=seed,
        centering=centering,
        determines=as_determinacy(determines, len(result.beta)),
    ).inference()


__all__ = [
    "term_inference",
]
