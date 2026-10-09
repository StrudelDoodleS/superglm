"""An ordered term's curve beside its levels fitted free.

The model is fitted again with the term's levels each free, a plain
categorical with the same groups and reference, and every other term as it
is declared: the comparison practice draws between a smoothed or simplified
factor and the factor fitted unconstrained, estimates with their standard
errors beside the curve (Anderson et al., *A Practitioner's Guide to
Generalized Linear Models*, CAS 2007, sections 2.16, 2.27-2.33 and 2.103).
A selection penalty, if the model has one, is lifted from this term only, so
its levels are not shrunk and the other terms are penalised as before.

A level is flagged where the curve lies outside its interval. The curve has
already moved toward each level's own data, so the gap between them is the
smoother's residual at that level, the statistic of the mean-shift outlier
test (She and Owen, "Outlier detection using nonconvex penalized regression",
JASA 2011). Both fits are linear in the response to first order, so the gap
is too, and its variance comes from both linearisations on the same rows,
taken under the free fit, which holds whether or not the level lies on the
curve: the cross-model covariance of seemingly unrelated estimation (Weesie,
Stata Technical Bulletin 52, sg121, 1999), with the family's variance in place
of the empirical one. The free estimate's variance less the curve's is the
special case of an efficient curve (Hausman, Econometrica 46(6), 1978), which
a penalised curve fitted with its own working weights is not. The interval is
drawn round the free estimate with the gap's variance, so a flag and an
interval that misses the curve are the same thing, and it is widened for the
number of levels compared (Sidak), so a flag is rarely chance. Each level is
judged on its own: making one level special changes the curve, and with it
every other level's flag.

Both fits are centred on the levels' mean weighted by their prior weight
(their exposure), the same weights on both sides, so a level with little
weight moves no other level's gap or interval. A level whose every response
is at the family's bound has no finite free value, and one whose rows another
term covers exactly has no free value of its own: both are left out and named.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from superglm._frame import as_eager_frame
from superglm.diagnostics.separation import _separated_flags, response_boundaries
from superglm.distributions import clip_mu
from superglm.editor.collapse import _require_not_interaction_parent
from superglm.editor.errors import EditorClientError, EditorTypeError, EditorValueError
from superglm.editor.refit import fit_refit_model
from superglm.features.categorical import Categorical, _grouping_labels
from superglm.features.grouping import LevelGrouping
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.rebuild import clone_with_replaced_features, special_labels
from superglm.inference._term_types import _safe_exp
from superglm.inference.covariance import covariance_selected_block
from superglm.links import stabilize_eta
from superglm.model.explain_ops import _shape_repaired
from superglm.model.state_ops import _grouped_active_state, _legacy_active_state
from superglm.solvers.constrained_qp import _feasibility_slack, _roundoff_tolerance
from superglm.solvers.mode_score import linear_predictor
from superglm.solvers.working_rows import (
    coefficient_working_rows,
    fisher_working_weights,
    supports_observed_newton,
)

# The chance, across all the levels of one term, that no level's interval
# misses the curve when every level lies on it.
CONFIDENCE = 0.95
# Past this leverage the curve all but passes through the level: what is
# left of the gap is the two fits' own tolerance, not data, so it is judged
# on the free estimate's variance instead.
_MAX_LEVERAGE = 0.99
_NOT_ORDERED = (
    "Compare with free levels is for ordered terms: every level of {term!r} is free already."
)
_NOT_FITTED = "The free levels of {term!r} could not be estimated, so there is nothing to compare."
_UNDETERMINED = (
    "No free value is drawn for {levels}: the model's other terms cover the same rows, "
    "so the data cannot separate {whose} value from theirs."
)
_ONE_LEVEL = (
    "Comparing {term!r} with free levels needs rows of positive weight in at least two of its "
    "levels, and the data the refit reads has them in {count}."
)
_SEPARATED = (
    "No free value is drawn for {levels}: every response on {whose} rows is {value}, so "
    "{whose} free value has no finite estimate."
)
_INDEPENDENT = (
    "Each interval takes the curve and the free estimate as independent, without the curve's "
    "pull toward the level: "
)
_NO_DESIGN = _INDEPENDENT + (
    "the model keeps no fitted design to measure it. Refit it with retain_fit_state=True to "
    "include it."
)
_OTHER_ROWS = _INDEPENDENT + "the curve was fitted on other rows than the comparison reads."
_SIGNED = (
    _INDEPENDENT
    + "under this family and link, rows far from their fitted mean leave it unmeasured."
)
_UNMEASURED = _INDEPENDENT + "this fit cannot measure it."
_UNCONVERGED = (
    "The {fit} stopped before it converged, so no level is judged: raise the model's max_iter "
    "to compare."
)
_NO_DATA = (
    "Comparing with free levels refits the model, which needs its training data: open the "
    "editor with train_data, or from a model fitted with its data kept."
)


def free_level_comparison(session, name: str) -> dict[str, Any]:
    """Refit the in-force model with ``name``'s levels free and compare them with its curve.

    Returns the payload the chart draws: for each level on the curve, its
    free estimate and interval as relativities on the chart's own scale,
    placed by the gap between the two fits centred on the levels' weighted
    mean, and the levels whose interval misses the curve. Hand edits are not
    part of either: both are fits.
    """
    term = session._require_term(name)
    spec = session.model._specs[name]
    if not isinstance(spec, OrderedCategorical):
        raise EditorTypeError(_NOT_ORDERED.format(term=name))
    _require_not_interaction_parent(session.model, name, operation="compare with free levels")
    try:
        X, y, sample_weight, offset = session._resolve_refit_data(None, None, None, None)
    except RuntimeError as exc:
        # No training data was given and the model kept none.
        raise EditorValueError(_NO_DATA) from exc
    if y is None:
        raise EditorValueError(_NO_DATA)
    column = as_eager_frame(X).column_array(name)
    response = np.asarray(y, dtype=np.float64).ravel()
    weights = (
        np.ones(len(response))
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64).ravel()
    )
    # The column read once: each row's text, as the categorical reads it.
    values = np.asarray(column, dtype=object).ravel()
    codes, texts = pd.factorize(_grouping_labels(values))
    first = np.unique(codes, return_index=True)[1]
    free, to_group = _free_categorical(spec, values[first], texts)
    supported = np.bincount(codes, weights=weights > 0.0, minlength=len(texts)) > 0.0
    held = {to_group[str(text)] for text in texts[supported]}
    if len(held) < 2:
        raise EditorValueError(_ONE_LEVEL.format(term=name, count=len(held)))
    free_model = clone_with_replaced_features(session.model, {name: free})
    shrunk = _lift_selection(free_model, session.model, name)
    # A level whose every response is at the family's bound is named below,
    # whatever the model's own rule for separated levels says.
    free_model._config = free_model._config.with_value(separation="ignore")
    try:
        fit_refit_model(
            session.model,
            free_model,
            method="auto",
            X=X,
            y=y,
            sample_weight=sample_weight,
            offset=offset,
        )
    except EditorClientError:
        raise
    except ValueError as exc:
        raise EditorValueError(_NOT_FITTED.format(term=name)) from exc
    # Each row's free level, and from it each level's exposure and whether
    # its every positive-weight response is at the family's bound.
    levels = [str(level) for level in free_model._specs[name]._levels]
    index = {level: i for i, level in enumerate(levels)}
    rows = np.array([index[to_group[str(text)]] for text in texts], dtype=np.intp)[codes]
    separated = {}
    for boundary in response_boundaries(free_model._distribution, free_model._link):
        flags, _occupied = _separated_flags(rows, len(levels), response, weights, boundary)
        if flags.any():
            separated[boundary] = [levels[i] for i in np.flatnonzero(flags)]
    # Both on the centred scale: a gap measured from the reference would
    # carry the curve's misfit at the reference into every level.
    exposure = np.bincount(rows, weights=weights, minlength=len(levels))
    gaps = _gaps(
        free_model,
        session.model,
        name,
        dict(zip(levels, exposure.tolist(), strict=True)),
        {label for labels in separated.values() for label in labels},
    )
    # A fit that stopped before converging holds no estimates to judge by:
    # its diamonds are drawn, and no level is flagged.
    unconverged = [
        fit
        for fit, model in (("free fit", free_model), ("model in force", session.model))
        if not bool(getattr(model.result, "converged", True))
    ]
    payload = _comparison(term, spec, gaps, shrunk=shrunk, judge=not unconverged)
    notes = [
        _SEPARATED.format(
            levels=", ".join(labels),
            whose="its" if len(labels) == 1 else "their",
            value="0" if boundary == "zero" else "1",
        )
        for boundary, labels in separated.items()
    ]
    if gaps.undetermined:
        notes.append(
            _UNDETERMINED.format(
                levels=", ".join(gaps.undetermined),
                whose="its" if len(gaps.undetermined) == 1 else "their",
            )
        )
    notes.extend(_UNCONVERGED.format(fit=fit) for fit in unconverged)
    if gaps.note is not None:
        notes.append(gaps.note)
    payload["notice"] = " ".join(notes) or None
    return payload


@dataclass(frozen=True)
class _Gaps:
    """The compared levels' centred values on both fits, and the variances that judge them."""

    labels: list[str]
    share: np.ndarray
    free: np.ndarray
    curve: np.ndarray
    free_var: np.ndarray
    gap_var: np.ndarray
    undetermined: list[str]
    note: str | None


def _gaps(free_model, model, name: str, prior: dict[str, float], separated: set[str]) -> _Gaps:
    """Both fits' level values centred on one weighted mean of one set of levels, and each gap's variance.

    The set is the levels the free fit estimated, less those ``separated``
    (every response at the family's bound) and those the data cannot tell
    apart from other terms, which are returned apart. Each level weighs its
    ``prior`` weight in the centre, on both sides alike.
    """
    free_spec, spec = free_model._specs[name], model._specs[name]
    pinned = {str(level) for level in getattr(free_spec, "_pinned_levels", ())}
    estimated = {str(level) for level in free_spec._levels} - pinned - separated
    declared = {str(level): level for level in spec._ordered_levels}
    labels, undetermined = _determined(
        free_model, name, [label for label in declared if label in estimated]
    )
    share = np.array([prior.get(label, 0.0) for label in labels], dtype=np.float64)
    share = share / share.sum() if share.sum() > 0.0 else share
    free_rows = _categorical_contrasts(free_model, name, labels, share)
    curve_rows = _ordered_contrasts(model, name, [declared[x] for x in labels], share)
    free_var = _contrast_variances(free_model, name, free_rows)
    gap_var, note = _gap_variances(free_model, free_rows, model, curve_rows, name)
    if gap_var is None:
        # Unmeasured, the two fits are taken as independent: exact for fits on
        # disjoint rows, and a bound for fits on shared rows, which correlate
        # positively.
        gap_var = free_var + (
            np.zeros(len(labels))
            if curve_rows is None
            else _contrast_variances(model, name, curve_rows)
        )
    free_values, curve_values = (_values(fitted, name, labels) for fitted in (free_model, model))
    return _Gaps(
        labels=labels,
        share=share,
        free=free_values - share @ free_values,
        curve=curve_values - share @ curve_values,
        free_var=free_var,
        gap_var=gap_var,
        undetermined=undetermined,
        note=note,
    )


def _determined(model, name: str, labels: list[str]) -> tuple[list[str], list[str]]:
    """The largest set of ``labels`` the free fit tells apart, and the rest.

    A level aliased with another term's columns (a categorical whose one
    level is exactly this level's rows) has no estimable free value, and a
    mean taken over it is not estimable either, so it would move every gap.
    Estimable differences fall into classes, since ``a - c = (a - b) + (b - c)``,
    and a mean over levels of two classes is not estimable: the gaps are
    measured within the largest class.
    """
    rank = getattr(model.result, "rank_info", None)
    if rank is None or _term_covariance(model, name) is None:
        return labels, []
    starts = [group.start for group in model._groups if group.feature_name == name]
    column = {str(level): starts[0] + j for j, level in enumerate(model._specs[name]._non_base)}

    def coordinate(label: str) -> np.ndarray:
        # The reference is the zero contrast: its value is fixed, not fitted.
        vector = np.zeros(len(rank.mean_x))
        if label in column:
            vector[column[label]] = 1.0
        return vector

    classes: list[list[str]] = []
    for label in labels:
        home = next(
            (c for c in classes if rank.is_estimable(coordinate(label) - coordinate(c[0]))), None
        )
        if home is None:
            classes.append([label])
        else:
            home.append(label)
    kept = max(classes, key=len, default=[])
    return kept, [label for label in labels if label not in kept]


def _values(model, name: str, labels: list[str]) -> np.ndarray:
    """The native log-relativities of fitted levels ``labels``; a grouped fit may report members."""
    native = model.term_inference(name, with_se=False)
    reported = {
        str(level): float(value)
        for level, value in zip(native.levels, native.log_relativity, strict=True)
    }
    grouping = getattr(model._specs[name], "_grouping", None)

    def value(label: str) -> float:
        members = [] if grouping is None else grouping.group_to_originals.get(label, [])
        found = [reported[str(m)] for m in [label, *members] if str(m) in reported]
        return found[0] if found else 0.0

    return np.array([value(label) for label in labels], dtype=np.float64)


def _term_covariance(model, name: str):
    """The fit's covariance and active groups, or None when selection removed the term."""
    covariance, active = model._coef_covariance
    if not any(group.feature_name == name for group in active):
        return None
    return covariance, active


def _term_columns(active, name: str) -> np.ndarray:
    """The term's columns in the fit's active coordinates."""
    return np.concatenate([np.arange(g.start, g.end) for g in active if g.feature_name == name])


def _categorical_contrasts(model, name: str, labels: list[str], share: np.ndarray) -> np.ndarray:
    """Rows over a categorical's columns giving its levels ``labels``, centred on ``share``."""
    column = {str(level): j for j, level in enumerate(model._specs[name]._non_base)}
    rows = np.zeros((len(labels), len(column)))
    for i, label in enumerate(labels):
        if label in column:
            rows[i, column[label]] = 1.0
    return rows - share @ rows


def _ordered_contrasts(model, name: str, levels: list, share: np.ndarray) -> np.ndarray | None:
    """Rows over an ordered term's active columns giving its ``levels``, centred on ``share``.

    None where the curve is taken as fixed: the selection penalty removed the
    term, or the fit's covariance no longer describes it, as ``term_inference``
    decides (hand edits an export baked in, or a shape repair after the fit).
    """
    found = _term_covariance(model, name)
    if (
        found is None
        or getattr(model, "_editor_inference_stale", False)
        or _shape_repaired(model, name)
    ):
        return None
    active = {group.name for group in found[1] if group.feature_name == name}
    groups = [group for group in model._groups if group.feature_name == name]
    columns = np.concatenate(
        [np.arange(g.start, g.end) - groups[0].start for g in groups if g.name in active]
    )
    spec = model._specs[name]
    rows = np.asarray(spec.transform(np.array(levels, dtype=object)), dtype=np.float64)[:, columns]
    return rows - share @ rows


def _contrast_variances(model, name: str, rows: np.ndarray) -> np.ndarray:
    """Each contrast's variance from the fit's covariance; zero when selection removed the term."""
    found = _term_covariance(model, name)
    if found is None:
        return np.zeros(len(rows))
    block = covariance_selected_block(found[0], _term_columns(found[1], name))
    return np.maximum(np.sum((rows @ block) * rows, axis=1), 0.0)


def _gap_variances(
    free_model, free_rows, model, curve_rows, name: str
) -> tuple[np.ndarray | None, str | None]:
    """Each gap's variance from both fits linearised on the same rows, or None and why not.

    To first order a fit's contrast moves with the response as ``s'y``, with
    ``s = (w / (V g')) u`` and ``u`` from :func:`_influence`, so the gap moves
    with ``s_free - s_curve``. Its variance under the free fit,
    ``var(y) = phi V / w``, is ``phi sum W (u_free - r u_curve)^2``, with ``W``
    the free fit's Fisher weight and ``r`` the ratio of the two fits' ``V g'``:
    the prior weights cancel, so no product leaves the float range that the
    fits themselves stay in. A curve taken as fixed (``curve_rows`` None), or
    one that fits exactly, contributes nothing. The sum runs row by row, so
    the two fits must have read the same rows, response, weights and offset,
    which their fit fingerprints decide.
    """
    phi = float(free_model.result.phi)
    if not phi > 0.0:
        return np.zeros(len(free_rows)), None
    if not float(model.result.phi) > 0.0:
        curve_rows = None
    if curve_rows is not None:
        fingerprints = [getattr(fit, "_fit_geometry_guard", None) for fit in (free_model, model)]
        if None in fingerprints:
            return None, _UNMEASURED
        if fingerprints[0] != fingerprints[1]:
            return None, _OTHER_ROWS
    free_side = _influence(free_model, name)
    if isinstance(free_side, str):
        return None, free_side
    if free_side.width != free_rows.shape[1]:
        return None, _UNMEASURED
    curve_side, ratio = None, None
    if curve_rows is not None:
        found = _influence(model, name)
        if isinstance(found, str):
            return None, found
        if found.width != curve_rows.shape[1]:
            return None, _UNMEASURED
        # The curve's influence goes onto the free fit's rows by this ratio.
        curve_side, ratio = found, free_side.spread / found.spread
    informative = free_side.fisher > 0.0
    root = np.sqrt(free_side.fisher[informative])
    variances = np.empty(len(free_rows))
    for i, row in enumerate(free_rows):
        moved = free_side.influence(row)
        if curve_side is not None and ratio is not None and curve_rows is not None:
            moved = moved - ratio * curve_side.influence(curve_rows[i])
        variances[i] = phi * float(np.sum((root * moved[informative]) ** 2))
    return variances, None


def _observed_curvature(model, mu, eta, prior, fisher) -> np.ndarray | None:
    """Each row's curvature of the fit's own objective at its optimum; ``fisher`` itself if equal.

    Gamma and Tweedie with a log link take the library's own observed
    kernels; every other pair :func:`_observed_rows`. None when the family or
    link does not give the derivatives.
    """
    distribution, link = model._distribution, model._link
    y = np.asarray(model._fit_y_ref, dtype=np.float64).ravel()
    if supports_observed_newton(distribution, link):
        return coefficient_working_rows(
            distribution=distribution,
            link=link,
            y=y,
            mu=mu,
            eta=eta,
            sample_weight=prior,
            prefer_observed=True,
        ).weights
    return _observed_rows(distribution, link, y, mu, eta, fisher)


def _observed_rows(distribution, link, y, mu, eta, fisher) -> np.ndarray | None:
    """The observed rows ``alpha W``, or ``fisher`` itself under a canonical link.

    ``alpha = 1 + (y - mu)(V'/V + g''/g')`` (Wood, JRSSB 73(1), 2011, section
    3), whatever rows the fit iterated on: the optimum is the same. With ``h``
    the inverse link, ``g''/g' = -h''/h'^2``. The bracket vanishes under a
    canonical link, where Fisher's rows are the observed ones; it is taken as
    vanishing when every row's value is within the rounding of its two terms.
    None when the family or link does not give the derivatives.
    """
    second = getattr(link, "deriv2_inverse", None)
    variance_slope = getattr(distribution, "variance_derivative", None)
    if second is None or variance_slope is None:
        return None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        first = np.asarray(link.deriv_inverse(eta), dtype=np.float64)
        spreading = np.asarray(variance_slope(mu), dtype=np.float64) / np.asarray(
            distribution.variance(mu), dtype=np.float64
        )
        bending = np.asarray(second(eta), dtype=np.float64) / first**2
        bracket = spreading - bending
    if not np.all(np.isfinite(bracket)):
        return None
    u = np.finfo(np.float64).eps / 2
    if np.all(np.abs(bracket) <= 8 * u * (np.abs(spreading) + np.abs(bending))):
        return fisher
    return fisher * (1.0 + (y - mu) * bracket)


def _on_binding_face(model, active, width: int, apply):
    """``apply`` restricted to the face of the fit's binding shape constraints.

    A constraint ``A theta >= b`` that binds at the fit holds the solution on
    its face under any small change of the response, so the fit moves with
    ``F - F A'(A F A')^+ A F``, the restricted estimator's form, rather than
    ``F``; a curve held flat by its monotone constraint does not move at all.
    A row binds when its slack, in the QP's own units, is within the fit's
    coefficient precision, about ``sqrt(tol)``, and the dot product's
    rounding: a row that close cannot be told from a binding one, and taken
    as binding it only widens an interval.
    """
    full = {group.name: group for group in model._groups}
    beta = np.asarray(model.result.beta, dtype=np.float64)
    rows = []
    for group in active:
        constraints = getattr(full.get(group.name), "constraints", None)
        if constraints is None or not constraints.n_constraints:
            continue
        source = full[group.name]
        theta = beta[source.start : source.end]
        A = np.asarray(constraints.A, dtype=np.float64)
        slack = _feasibility_slack(A, theta, np.asarray(constraints.b, dtype=np.float64))
        within = float(np.sqrt(model._tol)) + _roundoff_tolerance(A.shape[1])
        for row in A[slack <= within]:
            embedded = np.zeros(width)
            embedded[group.start : group.end] = row
            rows.append(embedded)
    if not rows:
        return apply
    R = np.array(rows)
    moves = np.column_stack([apply(row) for row in R])
    held = np.linalg.pinv(R @ moves, hermitian=True)

    def restricted(contrast: np.ndarray) -> np.ndarray:
        moved = apply(contrast)
        return moved - moves @ (held @ (R @ moved))

    return restricted


@dataclass(frozen=True)
class _Influence:
    """A fit's first-order response to its data, for contrasts over one term's columns."""

    influence: Any
    width: int
    fisher: np.ndarray
    spread: np.ndarray


def _influence(model, name: str) -> _Influence | str:
    """``c -> X_c F c`` over the term's active columns, or the sentence saying why not.

    At the fitted optimum a contrast moves with the response as
    ``(w / (V g')) X_c F c`` (the implicit function theorem), conditional on
    the fitted smoothing parameters, as a REML fit's ``Vp`` is, with ``F`` the
    inverse of the penalised curvature of the fit's own objective, its
    observed curvature (:func:`_observed_curvature`), and the intercept
    profiled out, so the design is centred on the curvature weights' mean.
    Under a canonical link that is the expected curvature, and the fit's
    own covariance serves. Comes with the Fisher weights, which carry each
    row's variance, and ``V g'``. The sentence instead when the fit kept no
    design, its curvature cannot be formed or factored or is negative on some
    row (a Gaussian/log fit far below a row, say), or its design and curvature
    do not share coordinates. The fit has a positive dispersion.
    """
    if getattr(model, "_dm", None) is None:
        return _NO_DESIGN
    distribution, link = model._distribution, model._link
    solver = model._solver_pirls_result()
    eta = stabilize_eta(linear_predictor(model._dm, solver, model._fit_offset), link)
    mu = clip_mu(link.inverse(eta), distribution)
    prior = (
        np.ones(model._dm.n)
        if model._fit_weights is None
        else np.asarray(model._fit_weights, dtype=np.float64)
    )
    fisher = fisher_working_weights(
        distribution=distribution, link=link, mu=mu, eta=eta, sample_weight=prior
    )
    spread = np.asarray(distribution.variance(mu) * link.deriv(mu), dtype=np.float64)
    observed = _observed_curvature(model, mu, eta, prior, fisher)
    if observed is None:
        return _UNMEASURED
    if observed is not fisher:
        if np.any(observed < 0.0):
            return _SIGNED
        if getattr(solver, "scop_inference", None) is not None:
            # A SCOP term's shape lives in its reparametrisation, whose
            # geometry only the fit's own covariance reads.
            return _UNMEASURED
        curvature = observed
        try:
            X, active, _inverse, _augmented, _gram, inverse, _rank = _legacy_active_state(
                model, solver, curvature
            )
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            return _UNMEASURED
        if not np.all(np.isfinite(inverse)):
            return _UNMEASURED

        def apply(contrast: np.ndarray) -> np.ndarray:
            return inverse @ contrast

        width = inverse.shape[0]
    else:
        curvature = fisher
        covariance, active = model._coef_covariance
        X, _active, _columns = _grouped_active_state(model, {group.name for group in active})
        scale = float(model.result.phi)

        def apply(contrast: np.ndarray) -> np.ndarray:
            if hasattr(covariance, "solve"):
                return np.asarray(covariance.solve(contrast), dtype=np.float64) / scale
            return np.asarray(covariance @ contrast, dtype=np.float64) / scale

        width = covariance.shape[0]
    if X.p != width:
        return _UNMEASURED
    move = _on_binding_face(model, active, width, apply)
    columns = _term_columns(active, name)
    # The weighted mean's sums, on weights rescaled by a power of two (exact),
    # stay in range whatever the prior weights' size.
    largest = float(np.max(np.abs(curvature), initial=0.0))
    unit = np.ldexp(curvature, -int(np.frexp(largest)[1])) if largest > 0.0 else curvature
    total = float(np.sum(unit))

    def influence(row: np.ndarray) -> np.ndarray:
        contrast = np.zeros(width)
        contrast[columns] = row
        moved = X.matvec(move(contrast))
        return moved - float(unit @ moved) / total

    return _Influence(influence=influence, width=len(columns), fisher=fisher, spread=spread)


def _free_categorical(spec: OrderedCategorical, raw, texts) -> tuple[Categorical, dict[str, str]]:
    """``spec``'s levels as a plain categorical: the same groups, the same reference.

    The ordered term reads its column through its declaration (a column of
    1.0, 2.0 against ``order=[1, 2]`` is levels 1 and 2), so the categorical
    groups the column's own texts under the term's levels and groups, and is
    named as the term names them. ``texts`` are the column's distinct texts as
    the categorical reads them, ``raw`` a value of each: 1 and 1.0 are one
    value to a hash but two texts to the column, and each needs its level. A
    reference the rows do not hold gives way to the first level they do; the
    comparison is centred, so the reference moves nothing. Comes with the
    map from each text to its level or group.
    """
    levels = [str(level) for level in spec._canonical(raw)]
    grouping = getattr(spec, "_grouping", None)
    to_group = {
        str(text): level if grouping is None else str(grouping.original_to_group.get(level, level))
        for text, level in zip(texts, levels, strict=True)
    }
    members: dict[str, list[str]] = {}
    for text, group in to_group.items():
        members.setdefault(group, []).append(text)
    present = [
        label for label in (str(level) for level in spec._ordered_levels) if label in members
    ]
    base = str(spec._base_level)
    free = Categorical(
        base=base if base in members or not present else present[0],
        grouping=LevelGrouping(
            original_to_group=to_group,
            group_to_originals=members,
            all_original_levels=list(to_group),
            grouped_levels=list(members),
        ),
    )
    # The reference is a level, whatever it is called ("first" included).
    free._base_is_level = True
    return free, to_group


def _lift_selection(free_model, model, name: str) -> bool:
    """Keep the selection penalty off ``name`` only; True where it cannot be lifted.

    A penalty that names its features names them again without ``name``; one
    that covers every feature names each of the others. A penalty object with
    no feature list cannot be restricted, and the levels are shrunk with it.
    """
    penalty = free_model.penalty
    if not getattr(penalty, "lambda1", None):
        return False
    if not hasattr(penalty, "features"):
        return True
    groups = getattr(model, "_groups", None) or ()
    targets = penalty.features
    if targets is None:
        targets = {group.feature_name for group in groups}
    own = {name} | {group.name for group in groups if group.feature_name == name}
    kept = frozenset(target for target in targets if target not in own)
    if kept:
        penalty.features = kept
    else:
        penalty.lambda1 = None
    free_model.penalty = penalty
    return False


def _comparison(
    term, spec: OrderedCategorical, gaps: _Gaps, *, shrunk: bool, judge: bool = True
) -> dict[str, Any]:
    """Each level's centred free estimate against the centred curve.

    The gap is drawn on the chart's own scale: the diamond sits that far from
    the curve, whatever the chart centres it on.
    """
    levels = [str(level) for level in term.levels]
    specials = special_labels(spec)
    grouping = getattr(spec, "_grouping", None)
    shown = np.asarray(term.original_log_effect, dtype=np.float64)
    at = {level: i for i, level in enumerate(levels)}
    compared = {label: i for i, label in enumerate(gaps.labels)}
    found = []
    for level in levels:
        group = level if grouping is None else str(grouping.original_to_group.get(level, level))
        if level in specials or group not in compared:
            continue
        i = compared[group]
        gap = float(gaps.free[i] - gaps.curve[i])
        free_var, gap_var = float(gaps.free_var[i]), float(gaps.gap_var[i])
        if gap_var <= (1.0 - _MAX_LEVERAGE) * free_var:
            gap_var = free_var
        found.append((level, group, gap, gap_var, judge and free_var > 0.0))
    # One comparison per group judged: a group's members share one free
    # estimate. A level with no free variance has nothing to judge.
    judged_groups = {group for _, group, _, _, judged in found if judged}
    z = float(norm.ppf(0.5 + 0.5 * CONFIDENCE ** (1.0 / max(len(judged_groups), 1))))
    rows = []
    for level, _group, gap, gap_var, judged in found:
        on_curve = float(shown[at[level]])
        half = z * float(np.sqrt(gap_var))
        rows.append(
            {
                "level": level,
                # A level with almost no weight has an interval past what
                # float64 holds; its ends stay finite, as the chart's own do.
                "y": float(_safe_exp(on_curve + gap)),
                "lower": float(_safe_exp(on_curve + gap - half)),
                "upper": float(_safe_exp(on_curve + gap + half)),
                "flagged": bool(judged and abs(gap) > half),
            }
        )
    return {
        "term": term.name,
        "levels": [row["level"] for row in rows],
        "y": [row["y"] for row in rows],
        "lower": [row["lower"] for row in rows],
        "upper": [row["upper"] for row in rows],
        "flagged": [row["level"] for row in rows if row["flagged"]],
        "confidence": CONFIDENCE,
        "z": z,
        "shrunk": shrunk,
    }
