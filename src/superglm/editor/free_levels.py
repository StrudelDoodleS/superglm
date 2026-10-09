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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from superglm._frame import as_eager_frame
from superglm.distributions import clip_mu
from superglm.editor.collapse import _require_not_interaction_parent
from superglm.editor.errors import EditorTypeError, EditorValueError
from superglm.editor.refit import fit_refit_model
from superglm.features.categorical import Categorical, _grouping_labels
from superglm.features.grouping import LevelGrouping
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.rebuild import clone_with_replaced_features, special_labels
from superglm.inference._term_types import _safe_exp
from superglm.inference.covariance import covariance_selected_block
from superglm.links import stabilize_eta
from superglm.model.explain_ops import _shape_repaired
from superglm.model.state_ops import _grouped_active_state
from superglm.solvers.mode_score import linear_predictor
from superglm.solvers.working_rows import fisher_working_weights

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
_NOT_LINEARISED = (
    "Each interval is the free estimate's own, without the curve's pull toward the level: the "
    "model keeps no fitted design on these rows to measure it. Refit it with "
    "retain_fit_state=True to include it."
)
_NO_DATA = (
    "Comparing with free levels refits the model, which needs its training data: open the "
    "editor with train_data, or from a model fitted with its data kept."
)


def free_level_comparison(session, name: str) -> dict[str, Any]:
    """Refit the in-force model with ``name``'s levels free and compare them with its curve.

    Returns the payload the chart draws: for each level on the curve, its
    free estimate and interval as relativities on the chart's own scale,
    placed by the gap between the two fits centred on the mean of the levels,
    and the levels whose interval misses the curve. Hand edits are not part
    of either: both are fits.
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
    free = _free_categorical(spec, as_eager_frame(X).column_array(name))
    free_model = clone_with_replaced_features(session.model, {name: free})
    shrunk = _lift_selection(free_model, session.model, name)
    fit_refit_model(
        session.model,
        free_model,
        method="auto",
        X=X,
        y=y,
        sample_weight=sample_weight,
        offset=offset,
    )
    # Both on the mean-centred scale: a gap measured from the reference
    # would carry the curve's misfit at the reference into every level.
    gaps = _gaps(free_model, session.model, name)
    payload = _comparison(term, spec, gaps, shrunk=shrunk)
    notes = []
    if gaps.undetermined:
        notes.append(
            _UNDETERMINED.format(
                levels=", ".join(gaps.undetermined),
                whose="its" if len(gaps.undetermined) == 1 else "their",
            )
        )
    if not gaps.linearised:
        notes.append(_NOT_LINEARISED)
    payload["notice"] = " ".join(notes) or None
    return payload


@dataclass(frozen=True)
class _Gaps:
    """The compared levels' centred values on both fits, and the variances that judge them."""

    labels: list[str]
    free: np.ndarray
    curve: np.ndarray
    free_var: np.ndarray
    gap_var: np.ndarray
    undetermined: list[str]
    linearised: bool


def _gaps(free_model, model, name: str) -> _Gaps:
    """Both fits' level values centred on the mean of one set of levels, and each gap's variance.

    The set is the levels the free fit estimated: a declared level with no
    rows has a value on the curve but none free, and centring the two over
    different sets would move every gap by the same amount. The levels whose
    free value the data cannot separate from other terms are returned apart.
    """
    free_spec, spec = free_model._specs[name], model._specs[name]
    pinned = {str(level) for level in getattr(free_spec, "_pinned_levels", ())}
    estimated = {str(level) for level in free_spec._levels} - pinned
    declared = {str(level): level for level in spec._ordered_levels}
    labels, undetermined = _determined(
        free_model, name, [label for label in declared if label in estimated]
    )
    free_rows = _categorical_contrasts(free_model, name, labels)
    free_var = _free_variances(free_model, name, free_rows)
    gap_var = _gap_variances(
        free_model,
        free_rows,
        model,
        _ordered_contrasts(model, name, [declared[x] for x in labels]),
        name,
    )
    free_values, curve_values = (_values(fitted, name, labels) for fitted in (free_model, model))
    return _Gaps(
        labels=labels,
        free=free_values - free_values.mean(),
        curve=curve_values - curve_values.mean(),
        free_var=free_var,
        gap_var=free_var if gap_var is None else gap_var,
        undetermined=undetermined,
        linearised=gap_var is not None,
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


def _categorical_contrasts(model, name: str, labels: list[str]) -> np.ndarray:
    """Rows over a categorical's columns giving its levels ``labels``, centred on their mean."""
    column = {str(level): j for j, level in enumerate(model._specs[name]._non_base)}
    rows = np.zeros((len(labels), len(column)))
    for i, label in enumerate(labels):
        if label in column:
            rows[i, column[label]] = 1.0
    return rows - rows.mean(axis=0)


def _ordered_contrasts(model, name: str, levels: list) -> np.ndarray | None:
    """Rows over an ordered term's active columns giving its ``levels``, centred on their mean.

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
    return rows - rows.mean(axis=0)


def _free_variances(model, name: str, rows: np.ndarray) -> np.ndarray:
    """Each free contrast's variance from the free fit's covariance; zero when selection removed it."""
    found = _term_covariance(model, name)
    if found is None:
        return np.zeros(len(rows))
    block = covariance_selected_block(found[0], _term_columns(found[1], name))
    return np.maximum(np.sum((rows @ block) * rows, axis=1), 0.0)


def _gap_variances(free_model, free_rows, model, curve_rows, name: str) -> np.ndarray | None:
    """Each gap's variance, from both fits linearised on the same rows; None without a design.

    To first order a fit's contrast moves with the response as ``s'y``, with
    ``s = W g'(mu) X_c F c``, so the gap moves with ``s_free - s_curve``. Its
    variance is ``sum (s_free - s_curve)^2 var(y)``, with ``var(y) = phi V(mu) / w
    = phi / (W g'(mu)^2)`` under the free fit. A curve taken as fixed
    (``curve_rows`` None) contributes nothing.
    """
    phi = float(free_model.result.phi)
    if not phi > 0.0:
        return np.zeros(len(free_rows))
    free_side = _influence(free_model, name)
    curve_side = None if curve_rows is None else _influence(model, name)
    if free_side is None or (curve_rows is not None and curve_side is None):
        return None
    free_influence, weights, slope = free_side
    if curve_side is not None and len(curve_side[1]) != len(weights):
        # The curve was fitted on other rows than the refit reads.
        return None
    informative = weights > 0.0
    precision = weights[informative] * slope[informative] ** 2
    variances = np.empty(len(free_rows))
    for i, row in enumerate(free_rows):
        moved = free_influence(row)
        if curve_side is not None:
            moved = moved - curve_side[0](curve_rows[i])
        variances[i] = phi * float(np.sum(moved[informative] ** 2 / precision))
    return variances


def _influence(model, name: str):
    """``c -> W g'(mu) X_c F c`` over the term's active columns, with ``W`` and ``g'(mu)``; or None.

    ``F`` is the fit's slope covariance over its dispersion, the inverse of
    the penalised information with the intercept profiled out, so the design
    is centred on the working weights' mean. None when the fit kept no design
    (``retain_fit_state=False``) or the fit has no dispersion to divide out.
    """
    scale = float(model.result.phi)
    if getattr(model, "_dm", None) is None or not scale > 0.0:
        return None
    covariance, active = model._coef_covariance
    X, _active, _columns = _grouped_active_state(model, {group.name for group in active})
    if X.p != covariance.shape[0]:
        return None
    solver = model._solver_pirls_result()
    eta = stabilize_eta(linear_predictor(model._dm, solver, model._fit_offset), model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    weights = fisher_working_weights(
        distribution=model._distribution,
        link=model._link,
        mu=mu,
        eta=eta,
        sample_weight=model._fit_weights,
    )
    slope = np.asarray(model._link.deriv(mu), dtype=np.float64)
    lever = weights * slope
    total = float(np.sum(weights))
    columns = _term_columns(active, name)

    def apply(contrast: np.ndarray) -> np.ndarray:
        if hasattr(covariance, "solve"):
            return np.asarray(covariance.solve(contrast), dtype=np.float64)
        return np.asarray(covariance @ contrast, dtype=np.float64)

    def influence(row: np.ndarray) -> np.ndarray:
        contrast = np.zeros(covariance.shape[0])
        contrast[columns] = row
        moved = X.matvec(apply(contrast) / scale)
        return lever * (moved - float(weights @ moved) / total)

    return influence, weights, slope


def _free_categorical(spec: OrderedCategorical, column) -> Categorical:
    """``spec``'s levels as a plain categorical: the same groups, the same reference.

    The ordered term reads its column through its declaration (a column of
    1.0, 2.0 against ``order=[1, 2]`` is levels 1 and 2), so the categorical
    groups the column's own texts under the term's levels and groups, and is
    named as the term names them. Each text the categorical reads gets one:
    1 and 1.0 are one value to a hash but two texts to the column. A
    reference the rows do not hold gives way to the first level they do; the
    comparison is centred on the levels' mean, so the reference moves nothing.
    """
    values = np.asarray(column, dtype=object).ravel()
    texts, first = np.unique(_grouping_labels(values), return_index=True)
    levels = [str(level) for level in spec._canonical(values[first])]
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
    return free


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


def _comparison(term, spec: OrderedCategorical, gaps: _Gaps, *, shrunk: bool) -> dict[str, Any]:
    """Each level's mean-centred free estimate against the mean-centred curve.

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
        found.append((level, group, gap, gap_var, free_var > 0.0))
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
