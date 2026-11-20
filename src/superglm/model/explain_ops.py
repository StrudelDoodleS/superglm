"""Inference, plotting delegation, and feature diagnostics."""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import warnings

import numpy as np

from superglm.inference._term_covariance import simultaneous_band_and_critical_value
from superglm.inference._term_model_ops import (
    drop1 as _drop1,
)
from superglm.inference._term_model_ops import (
    refit_unpenalised as _refit_unpenalised,
)
from superglm.inference._term_model_ops import (
    relativities as _relativities,
)
from superglm.inference._term_ops import (
    term_inference as _term_inference,
)
from superglm.inference.covariance import _SelectedBlockAccess
from superglm.model.fit_state import fitted_lambda2

_EDITOR_STALE_INFERENCE_MESSAGE = (
    "Editor coefficient edits make fitted standard errors and confidence "
    "intervals reference-only; returning term inference without SE/CI."
)


def _shape_repaired_features(model) -> set[str]:
    """The features whose published coefficients came out of a shape repair.

    Routed through ``report_ops.shape_repaired_feature_names`` rather than
    reading ``_shape_repairs`` directly, so this half of the contract cannot
    disagree with the summary's half about which terms are repaired -- that
    helper falls back to ``_monotone_repairs`` for a model pickled before
    ``_shape_repairs`` existed, and reading the newer attribute alone would
    withhold in ``summary()`` and publish here on exactly the input the
    fallback was written for.
    """
    from superglm.model.report_ops import shape_repaired_feature_names

    return {str(key) for key in shape_repaired_feature_names(model)}


def _shape_repaired(model, name) -> bool:
    """Whether this term's published coefficients came out of a shape repair.

    A repair replaces them with a projection onto the shape cone. The
    projection is a constrained estimator whose reference distribution is not
    the unconstrained fit's -- the effective dimension depends on the number of
    active cone edges, a random variable (Meyer & Woodroofe, *Ann. Statist.*
    28(4):1083-1104, 2000) -- and a binding repair sits on the boundary, where
    the constrained and unconstrained estimators stop agreeing. The curve and
    the point estimate are an improvement whatever produced them (Chernozhukov,
    Fernandez-Val & Galichon, *Biometrika* 96(3):559-575, 2009, Prop. 1); the
    band around them belongs to a model that was not fitted.
    """
    return str(name) in _shape_repaired_features(model)


_SHAPE_REPAIRED_INFERENCE_MESSAGE = (
    "This term's coefficients were replaced by a post-fit shape projection, so "
    "fitted standard errors and confidence intervals reference the "
    "unconstrained fit rather than the published one; returning term inference "
    "without SE/CI. Fit the constraint (Constraint.fit.*) if you need a band."
)


class _RootCovariance(_SelectedBlockAccess):
    """``R R'`` read as a covariance: a kind's directions ``q q'`` (``_curve_labels``).

    The term readers' standard error of a point is ``sqrt(r V r')``, so at ``V = Q Q'`` it is
    ``||r Q||``, the point's share along the kind, and at ``V = diag(D^-2)`` it is ``||r D^-1||``,
    its dual norm: the readers evaluate the off-support test on the very functional they report.
    """

    def __init__(self, root):
        self.root = np.asarray(root, dtype=np.float64)
        self.shape = (self.root.shape[0], self.root.shape[0])

    def selected_block(self, indices):
        rows = self.root[indices]
        return rows @ rows.T

    def selected_diagonal(self, indices):
        rows = self.root[indices]
        return np.einsum("ij,ij->i", rows, rows)

    def solve(self, vector):
        return self.root @ (self.root.T @ vector)


class _DiagonalCovariance(_SelectedBlockAccess):
    """``diag(d)`` read as a covariance (``_RootCovariance``)."""

    def __init__(self, diagonal):
        self.diagonal = np.asarray(diagonal, dtype=np.float64)
        self.shape = (self.diagonal.size, self.diagonal.size)

    def selected_block(self, indices):
        return np.diag(self.diagonal[indices])

    def selected_diagonal(self, indices):
        return self.diagonal[indices]

    def solve(self, vector):
        return self.diagonal * vector


def _active_layout(model):
    """The selected groups re-indexed to the active columns, and those columns' coordinates.

    The layout of the coefficient covariance the term readers take (``_coef_covariance``).
    """
    from superglm.solvers.rank import selected_group_name_set

    selected = selected_group_name_set(model.result, model._groups)
    groups, columns, start = [], [], 0
    for group in model._groups:
        if group.name in selected:
            width = group.end - group.start
            groups.append(dataclasses.replace(group, start=start, end=start + width))
            columns.append(np.arange(group.start, group.end, dtype=np.intp))
            start += width
    return groups, (np.concatenate(columns) if columns else np.zeros(0, dtype=np.intp))


def _curve_labels(model, read):
    """Per kind, ``read(covariance_fn)``'s standard errors at the probes, as off-support labels.

    ``read`` takes a ``covariance_fn`` and returns a mapping of point arrays (a term's points,
    or each term's), the readers' standard errors at that covariance.  ``None`` when the fit
    has no direction to disclose (``estimability_ops.published_directions``): then nothing is
    read.  One read per kind and one for the norms, all in coefficient space.
    """
    from superglm.diagnostics.estimability import off_support
    from superglm.model.estimability_ops import published_directions

    directions = published_directions(model)
    if directions is None:
        return None
    groups, columns = _active_layout(model)
    norms = read(lambda: (_DiagonalCovariance(1.0 / directions.scale[columns] ** 2), groups))
    shares = {
        kind: read(lambda slopes=slopes: (_RootCovariance(slopes[columns]), groups))
        for kind, slopes in directions.slopes.items()
    }
    return {
        name: off_support(
            {kind: np.asarray(values[name], dtype=np.float64) for kind, values in shares.items()},
            np.asarray(norm, dtype=np.float64),
        )
        for name, norm in norms.items()
    }


# Set while a reader that shows each point's ``non_estimable`` kind itself (the editor's curves)
# reads term inference: the warning would repeat what it shows.  A context variable, so one
# thread's reader does not silence another's call.
_CURVE_DISCLOSURE_QUIET: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "superglm_curve_disclosure_quiet", default=False
)


@contextlib.contextmanager
def curve_disclosure_quiet():
    """Within the block, term values off the fit's support are flagged without the warning."""
    token = _CURVE_DISCLOSURE_QUIET.set(True)
    try:
        yield
    finally:
        _CURVE_DISCLOSURE_QUIET.reset(token)


def _curve_warning(flagged) -> None:
    """One warning for the term points ``flagged`` (name -> labels) whose standard error is NaN.

    Its ``points`` map each term to its labels; ``rows`` and ``kinds`` are the points of a single
    term, empty when several are named.  Silent under ``curve_disclosure_quiet``.
    """
    if _CURVE_DISCLOSURE_QUIET.get():
        return
    from superglm._warn import warn_caller
    from superglm.diagnostics.estimability import NonEstimablePredictionWarning, reasons

    parts = [f"{name!r} ({reasons(labels)})" for name, labels in flagged.items()]
    single = next(iter(flagged.values())) if len(flagged) == 1 else np.zeros(0, dtype=object)
    rows = np.flatnonzero(single != "")
    warn_caller(
        NonEstimablePredictionWarning(
            "Term values that lean on what the fit does not determine are returned at the "
            "published fit's values, and where a standard error is reported it is NaN: "
            + "; ".join(parts)
            + ". The warning's `points` name them by term.",
            rows,
            [str(kind) for kind in single[rows]],
            points=dict(flagged),
        )
    )


def metrics(model, X, y, sample_weight=None, offset=None):
    """Compute comprehensive diagnostics for the fitted model.

    Every statistic of the fitted mean is computed from the model's
    predictions on ``X``, the values ``predict`` returns.  The fit's own mean,
    likelihood and null statistics are not reused, even for the fit's own
    objects.  On a discrete fit they belong to the binned design, and on any
    fit they come from a different evaluation than ``predict``, so reusing
    them for identical objects only made the result depend on object identity
    and on whether the model had been saved (#441).  What the fit's own
    objects still keep, when the fit's data guard confirms they are unchanged,
    is the cached result object and the fit geometry that leverage reads.
    The weight-contract check and custom-family prior-weight report already
    given by the fit are not repeated for those guarded objects. Prediction
    disclosures about separated or partly identified levels are suppressed
    on every dataset, including holdout rows.
    """
    from superglm.inference.metrics import ModelMetrics

    cache_signature = (id(model.result), id(getattr(model, "_reml_penalties", None)))
    same_fit_objects = (
        X is getattr(model, "_fit_X_ref", None)
        and y is getattr(model, "_fit_y_ref", None)
        and sample_weight is getattr(model, "_fit_sample_weight_ref", None)
        and offset is getattr(model, "_fit_offset_ref", None)
    )
    fit_data_guard = getattr(model, "_fit_data_guard", None)
    same_fit_refs = bool(
        same_fit_objects
        and fit_data_guard is not None
        and fit_data_guard.matches(
            X,
            y,
            sample_weight,
            offset,
            fit_weights=getattr(model, "_fit_weights", None),
            fit_offset=getattr(model, "_fit_offset", None),
            own_frame=True,
        )
    )
    if (
        same_fit_refs
        and model._fit_metrics_cache is not None
        and model._fit_metrics_cache_signature == cache_signature
    ):
        return model._fit_metrics_cache

    metrics_obj = ModelMetrics(
        model,
        X,
        y,
        sample_weight,
        offset,
        _fit_data_matches=same_fit_refs,
        _contract_checked=same_fit_refs,
    )
    if same_fit_refs:
        model._fit_metrics_cache = metrics_obj
        model._fit_metrics_cache_signature = cache_signature
    return metrics_obj


def drop1(model, X, y, sample_weight=None, offset=None, test="Chisq"):
    """Drop-one deviance analysis for each feature."""
    return _drop1(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        test=test,
    )


def refit_unpenalised(model, X, y, sample_weight=None, offset=None, keep_smoothing=True):
    """Refit with only active features and no selection penalty."""
    return _refit_unpenalised(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        keep_smoothing=keep_smoothing,
    )


def relativities(model, with_se=False, centering="native"):
    """Extract plot-ready relativity DataFrames for all features.

    By default returns the canonical fitted term contributions under
    the model's identifiability constraint (``centering="native"``).
    Pass ``centering="mean"`` to shift so geometric mean of
    relativities = 1 — a reporting convenience, not the fitted term.
    """
    if getattr(model, "_editor_inference_stale", False) and with_se:
        warnings.warn(_EDITOR_STALE_INFERENCE_MESSAGE, UserWarning, stacklevel=2)
        with_se = False
    from superglm.model.fit_ops import _ensure_factor_smooth_levels_recorded

    _ensure_factor_smooth_levels_recorded(model)
    frames = _relativities(
        model._feature_order,
        model._interaction_order,
        model._specs,
        model._interaction_specs,
        model._groups,
        model.result,
        with_se=with_se,
        covariance_fn=(lambda: model._coef_covariance) if with_se else None,
        centering=centering,
    )
    if with_se:
        frames = _disclosed_relativities(model, frames, centering)
    # Per FEATURE, not for the whole call: this returns every term at once, and
    # dropping the band on unrepaired terms because one term was projected
    # would be the over-reach the repair record already avoids. Only the
    # repaired frames lose their SE column; the curve stays in all of them.
    if with_se:
        repaired = _shape_repaired_features(model)
        # Iterate the FRAMES and normalize on the way in, rather than
        # intersecting the two key sets: `_shape_repaired_features` stringifies
        # so that one spelling answers for every surface, while `_relativities`
        # keys its output by the original feature label -- which need not be a
        # string. A feature named `0` or `None` would make that intersection
        # empty and leave this guard silently inert, which is the failure mode
        # the whole PR is about.
        for name in [key for key in frames if str(key) in repaired]:
            frame = frames[name]
            if "se_log_relativity" in frame.columns:
                warnings.warn(_SHAPE_REPAIRED_INFERENCE_MESSAGE, UserWarning, stacklevel=2)
                frames[name] = frame.drop(columns=["se_log_relativity"])
    return frames


def _interaction_points(raw):
    """An interaction reconstruction's reported log values, flat, and the keys that name them.

    The keys follow ``InteractionInference``: ``(level, i)`` for point ``i`` of a per-level
    curve, the ``(level1, level2)`` pair of a cell, the level of a per-unit slope, ``"coef"``
    for a product coefficient.  ``None`` for a surface, which reports no value here.
    """
    from superglm.features.interaction import interaction_cells

    if "per_level" in raw and "x" in raw:
        keys = [(level, i) for level in raw["levels"] for i in range(len(raw["x"]))]
        values = [np.asarray(raw["per_level"][level]["log_relativity"]) for level in raw["levels"]]
        return keys, np.concatenate(values) if values else np.zeros(0)
    if "pairs" in raw:
        cells = interaction_cells(raw, "log_relativities")
        return list(cells), np.fromiter(cells.values(), dtype=np.float64, count=len(cells))
    if "relativities_per_unit" in raw:
        slopes = raw["log_relativities_per_unit"]
        levels = list(raw["levels"])
        return levels, np.array([slopes[level] for level in levels], dtype=np.float64)
    if "relativity_per_unit_unit" in raw:
        return ["coef"], np.array([raw["coef"]], dtype=np.float64)
    return None


def _interaction_labels(model, name):
    """``{key: kind}`` for an interaction's reported points (``_interaction_points``), or ``None``.

    The same test as every term's (``estimability_ops.point_labels``), on the reconstruction's
    log values at the term's coefficients; ``None`` when no point is flagged.
    """
    from superglm.model.estimability_ops import point_labels, term_coordinates

    ispec = model._interaction_specs[name]
    beta = np.asarray(model.result.beta)[term_coordinates(model, name)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = _interaction_points(ispec.reconstruct(beta))
        if fitted is None:
            return None
        keys, _ = fitted
        labels = point_labels(model, name, lambda b: _interaction_points(ispec.reconstruct(b))[1])
    if labels is None or not np.any(labels != ""):
        return None
    return dict(zip(keys, (str(kind) for kind in labels), strict=True))


def _relativity_frame_labels(model, name, labels):
    """An interaction's labels (``_interaction_labels``) by the relativity frames that hold them."""
    first = next(iter(labels))
    if isinstance(first, tuple) and isinstance(first[1], int):
        levels = dict.fromkeys(key[0] for key in labels)
        return {
            f"{name}[{level}]": np.array(
                [kind for (lv, _), kind in labels.items() if lv == level], dtype=object
            )
            for level in levels
        }
    return {name: np.array(list(labels.values()), dtype=object)}


def _disclosed_relativities(model, frames, centering):
    """``frames`` with NaN standard errors on the points off the fit's support, and one warning.

    A relativity is a term functional (its coordinates alone, no intercept), tested as
    ``predict``'s rows are (``_curve_labels``) on the functional its standard error reports.
    """

    def read(covariance_fn):
        probed = _relativities(
            model._feature_order,
            model._interaction_order,
            model._specs,
            model._interaction_specs,
            model._groups,
            model.result,
            with_se=True,
            covariance_fn=covariance_fn,
            centering=centering,
        )
        return {
            name: frame["se_log_relativity"].to_numpy()
            for name, frame in probed.items()
            if "se_log_relativity" in frame.columns
        }

    labels = _curve_labels(model, read)
    if labels is None:
        return frames
    flagged = {name: values for name, values in labels.items() if np.any(values != "")}
    for name in model._interaction_order:
        cells = _interaction_labels(model, name)
        if cells is not None:
            # an interaction's frames carry no standard error: the warning names its points
            by_frame = _relativity_frame_labels(model, name, cells)
            flagged.update({key: kinds for key, kinds in by_frame.items() if np.any(kinds != "")})
    for name, values in flagged.items():
        if "se_log_relativity" not in frames[name].columns:
            continue
        frame = frames[name].copy()
        se = frame["se_log_relativity"].to_numpy(dtype=np.float64, copy=True)
        se[values != ""] = np.nan
        frame["se_log_relativity"] = se
        frames[name] = frame
    if flagged:
        _curve_warning(flagged)
    return frames


def _disclosed_term(model, inference, read_kwargs):
    """``term_inference``'s result with its off-support points flagged (``_curve_labels``).

    ``non_estimable`` holds each point's kind (``""`` where the fit determines it); with
    standard errors, a flagged point's error, intervals and bands are NaN, with one warning.
    """
    from superglm.inference._term_types import InteractionInference, TermInference

    if isinstance(inference, InteractionInference) and inference.active:
        cells = _interaction_labels(model, inference.name)
        if cells is None:
            return inference
        if read_kwargs.get("with_se"):
            _curve_warning({inference.name: np.array(list(cells.values()), dtype=object)})
        flagged = {key: kind for key, kind in cells.items() if kind}
        return dataclasses.replace(inference, non_estimable=flagged)
    if not isinstance(inference, TermInference) or not inference.active:
        return inference
    read_kwargs = {key: value for key, value in read_kwargs.items() if key != "with_se"}
    name = inference.name

    def read(covariance_fn):
        probed = _term_inference(
            name,
            result=model.result,
            groups=model._groups,
            specs=model._specs,
            interaction_specs=model._interaction_specs,
            covariance_fn=covariance_fn,
            reml_lambdas=getattr(model, "_reml_lambdas", None),
            lambda2=fitted_lambda2(model),
            group_edf=model._group_edf,
            with_se=True,
            simultaneous=False,
            **read_kwargs,
        )
        values = {"points": probed.se_log_relativity}
        curve = getattr(probed, "smooth_curve", None)
        if curve is not None and curve.se_log_relativity is not None:
            values["curve"] = curve.se_log_relativity
        return values if values["points"] is not None else {}

    labels = _curve_labels(model, read)
    points = None if labels is None else labels.get("points")
    if points is None or not np.any(points != ""):
        return inference
    flagged = points != ""

    def blank(values):
        if values is None:
            return None
        values = np.array(values, dtype=np.float64, copy=True)
        values[flagged] = np.nan
        return values

    changes = {"non_estimable": points}
    if inference.se_log_relativity is not None:
        for field in (
            "se_log_relativity",
            "ci_lower",
            "ci_upper",
            "ci_lower_simultaneous",
            "ci_upper_simultaneous",
        ):
            changes[field] = blank(getattr(inference, field))
        if inference.knot_covariance is not None and inference.knot_covariance.shape[0] == (
            flagged.size
        ):
            covariance = np.array(inference.knot_covariance, dtype=np.float64, copy=True)
            covariance[flagged, :] = np.nan
            covariance[:, flagged] = np.nan
            changes["knot_covariance"] = covariance
        curve_labels = labels.get("curve")
        curve = inference.smooth_curve
        if curve is not None and curve_labels is not None and np.any(curve_labels != ""):
            on_curve = curve_labels != ""
            changes["smooth_curve"] = dataclasses.replace(
                curve,
                **{
                    field: np.where(on_curve, np.nan, getattr(curve, field))
                    for field in ("se_log_relativity", "ci_lower", "ci_upper")
                    if getattr(curve, field) is not None
                },
            )
        _curve_warning({name: points})
    return dataclasses.replace(inference, **changes)


def simultaneous_bands(model, feature, *, alpha=0.05, n_sim=10_000, n_points=200, seed=42):
    """Simultaneous confidence bands for a spline feature."""
    if model._result is None:
        raise RuntimeError("Model must be fitted before calling simultaneous_bands().")
    if getattr(model, "_editor_inference_stale", False):
        raise RuntimeError(
            "Editor coefficient edits make simultaneous bands unavailable until the model is refit."
        )
    if _shape_repaired(model, feature):
        raise RuntimeError(
            f"Feature {feature!r} carries a post-fit shape repair, so simultaneous bands "
            "would be simulated from the unconstrained fit's covariance around constrained "
            "coefficients. Fit the constraint (Constraint.fit.*) if you need a band."
        )
    from superglm.model.fit_ops import _ensure_factor_smooth_levels_recorded

    _ensure_factor_smooth_levels_recorded(model)
    bands, _ = simultaneous_band_and_critical_value(
        feature,
        result=model.result,
        groups=model._groups,
        specs=model._specs,
        interaction_specs=model._interaction_specs,
        covariance_fn=lambda: model._coef_covariance,
        alpha=alpha,
        n_sim=n_sim,
        n_points=n_points,
        seed=seed,
    )
    return bands


def term_inference(
    model,
    name,
    *,
    with_se=True,
    simultaneous=False,
    n_points=200,
    alpha=0.05,
    n_sim=10_000,
    seed=42,
    centering="native",
):
    """Per-term inference: curve, uncertainty, and metadata in one object."""
    if model._result is None:
        raise RuntimeError("Model must be fitted before calling term_inference().")
    if getattr(model, "_editor_inference_stale", False) and with_se:
        warnings.warn(_EDITOR_STALE_INFERENCE_MESSAGE, UserWarning, stacklevel=2)
        with_se = False
        simultaneous = False
    repaired = _shape_repaired(model, name)
    if repaired and with_se:
        warnings.warn(_SHAPE_REPAIRED_INFERENCE_MESSAGE, UserWarning, stacklevel=2)
        with_se = False
        simultaneous = False
    from superglm.model.fit_ops import _ensure_factor_smooth_levels_recorded

    _ensure_factor_smooth_levels_recorded(model)
    inference = _term_inference(
        name,
        result=model.result,
        groups=model._groups,
        specs=model._specs,
        interaction_specs=model._interaction_specs,
        covariance_fn=lambda: model._coef_covariance,
        reml_lambdas=getattr(model, "_reml_lambdas", None),
        lambda2=fitted_lambda2(model),
        group_edf=model._group_edf,
        with_se=with_se,
        simultaneous=simultaneous,
        n_points=n_points,
        alpha=alpha,
        n_sim=n_sim,
        seed=seed,
        centering=centering,
    )
    inference = _disclosed_term(
        model,
        inference,
        {"n_points": n_points, "alpha": alpha, "centering": centering, "with_se": with_se},
    )
    # `_term_ops` has no model handle, so it hard-codes `monotone_repaired=False`
    # with a comment inviting the caller to override. This is that caller: a
    # consumer that reads the object rather than the summary text has no other
    # way to learn the curve came out of a projection.
    if repaired and not inference.monotone_repaired:
        inference = dataclasses.replace(inference, monotone_repaired=True)
    # A path fits the term without the rows its build made; the rule line says so.
    from superglm.features.constraint import rules_of
    from superglm.inference.summary import unenforced_rules_tag
    from superglm.model.report_ops import rules_enforced

    if (
        not rules_enforced(model)
        and getattr(inference, "monotone", None)
        and any(g.feature_name == name and g.monotone_engine for g in model._groups)
    ):
        tag = unenforced_rules_tag(rules_of(model._specs[name]))
        inference = dataclasses.replace(inference, monotone=f"{inference.monotone} ({tag})")
    return inference


def term_importance(model, X, sample_weight=None):
    """Weighted variance of each term's contribution to eta."""
    from superglm.diagnostics.term_diagnostics import term_importance as _term_importance

    return _term_importance(model, X, sample_weight)


def term_drop_diagnostics(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    mode="refit",
    X_val=None,
    y_val=None,
    sample_weight_val=None,
    offset_val=None,
):
    """Drop-term diagnostics wrapper."""
    from superglm.diagnostics.term_diagnostics import (
        term_drop_diagnostics as _term_drop_diagnostics,
    )

    return _term_drop_diagnostics(
        model,
        X,
        y,
        sample_weight,
        offset,
        mode=mode,
        X_val=X_val,
        y_val=y_val,
        sample_weight_val=sample_weight_val,
        offset_val=offset_val,
    )


def spline_redundancy(model, X, sample_weight=None):
    """Spline redundancy diagnostics."""
    from superglm.diagnostics.spline_checks import spline_redundancy as _spline_redundancy

    return _spline_redundancy(model, X, sample_weight)


def discretization_impact(model, X, y, sample_weight=None, **kwargs):
    """Analyse the impact of discretizing this fit's smooth terms.

    Spline and polynomial main effects into bins, and a
    continuous-by-continuous interaction onto the grid its rating-table
    block is sampled on."""
    from superglm.diagnostics.discretize import discretization_impact as _disc_impact

    return _disc_impact(model, X, y, sample_weight, **kwargs)
