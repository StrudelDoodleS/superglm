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
smoother's residual at that level, whose variance is the free estimate's less
the curve's (``var(b - f) = var(b)(1 - h)``, with ``h`` the level's leverage
on the curve): the standardised residual of the mean-shift outlier test (She
and Owen, "Outlier detection using nonconvex penalized regression", JASA
2011). Using the curve's Bayesian variance makes the gap's variance an upper
bound, so the flag errs towards none. The interval is drawn round the free
estimate with that variance, so a flag and an interval that misses the curve
are the same thing, and it is widened for the number of levels compared
(Sidak), so a flag is rarely chance. Each level is judged on its own: making
one level special changes the curve, and with it every other level's flag.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from superglm._frame import as_eager_frame
from superglm.editor.collapse import _require_not_interaction_parent
from superglm.editor.errors import EditorTypeError, EditorValueError
from superglm.editor.refit import fit_refit_model
from superglm.features.categorical import Categorical
from superglm.features.grouping import LevelGrouping
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.rebuild import clone_with_replaced_features, special_labels
from superglm.inference._term_helpers import _spline_se
from superglm.inference._term_types import _safe_exp
from superglm.inference.covariance import covariance_selected_block
from superglm.model.explain_ops import _shape_repaired

# The chance, across all the levels of one term, that no level's interval
# misses the curve when every level lies on it.
CONFIDENCE = 0.95
# Past this leverage the curve's variance leaves the residual's bound no
# room, so the gap is judged on the free estimate's own variance, which
# bounds it too: a penalised smoother's residual ``(I - H) b`` varies no
# more than ``b``.
_MAX_LEVERAGE = 0.99
_NOT_ORDERED = (
    "Compare with free levels is for ordered terms: every level of {term!r} is free already."
)
_NOT_FITTED = "The free levels of {term!r} could not be estimated, so there is nothing to compare."
_UNDETERMINED = (
    "No free value is drawn for {levels}: the model's other terms cover the same rows, "
    "so the data cannot separate {whose} value from theirs."
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
    free_levels, curve, undetermined = _centred_pair(free_model, session.model, name)
    # The two fits estimate their dispersion apart; the curve's variance is
    # put on the free fit's, so the two variances share one scale. A response
    # the curve fits exactly has no dispersion, and no variance to rescale.
    curve_phi = float(session.model.result.phi)
    scale = float(free_model.result.phi) / curve_phi if curve_phi > 0.0 else 0.0
    payload = _comparison(term, spec, free_levels, curve, curve_scale=scale, shrunk=shrunk)
    payload["notice"] = (
        _UNDETERMINED.format(
            levels=", ".join(undetermined), whose="its" if len(undetermined) == 1 else "their"
        )
        if undetermined
        else None
    )
    return payload


def _centred_pair(free_model, model, name: str) -> tuple[dict, dict, list[str]]:
    """Both fits' level values and errors, centred on the mean of one set of levels.

    The set is the levels the free fit estimated: a declared level with no
    rows has a value on the curve but none free, and centring the two over
    different sets would move every gap by the same amount. The errors are
    those of that one contrast, each from its fit's covariance. A term the
    selection penalty removed is flat, with no variance. The levels whose
    free value the data cannot separate from other terms are returned apart.
    """
    free_spec, spec = free_model._specs[name], model._specs[name]
    pinned = {str(level) for level in getattr(free_spec, "_pinned_levels", ())}
    estimated = {str(level) for level in free_spec._levels} - pinned
    declared = {str(level): level for level in spec._ordered_levels}
    labels, undetermined = _determined(
        free_model, name, [label for label in declared if label in estimated]
    )
    pair = []
    for fitted, se in (
        (free_model, _categorical_centred_se(free_model, name, labels)),
        (model, _ordered_centred_se(model, name, [declared[label] for label in labels])),
    ):
        values = _values(fitted, name, labels)
        centred = values - values.mean()
        pair.append(
            {label: (float(c), float(s)) for label, c, s in zip(labels, centred, se, strict=True)}
        )
    return pair[0], pair[1], undetermined


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


def _categorical_centred_se(model, name: str, labels: list[str]) -> np.ndarray:
    """Errors of a categorical's levels centred on the mean of ``labels``."""
    found = _term_covariance(model, name)
    if found is None:
        return np.zeros(len(labels))
    covariance, active = found
    spec = model._specs[name]
    indices = np.concatenate([np.arange(g.start, g.end) for g in active if g.feature_name == name])
    block = covariance_selected_block(covariance, indices)
    column = {str(level): j for j, level in enumerate(spec._non_base)}
    rows = np.zeros((len(labels), len(column)))
    for i, label in enumerate(labels):
        if label in column:
            rows[i, column[label]] = 1.0
    rows -= rows.mean(axis=0)
    return np.sqrt(np.maximum(np.sum((rows @ block) * rows, axis=1), 0.0))


def _ordered_centred_se(model, name: str, levels: list) -> np.ndarray:
    """Errors of an ordered term's ``levels`` centred on their mean.

    Zero where the fit's covariance no longer describes the curve, as
    ``term_inference`` decides: hand edits an export baked in, or a shape
    repair after the fit. The curve is then taken as fixed, and each gap is
    judged on the free estimate's variance, which bounds it as at high leverage.
    """
    found = _term_covariance(model, name)
    if (
        found is None
        or getattr(model, "_editor_inference_stale", False)
        or _shape_repaired(model, name)
    ):
        return np.zeros(len(levels))
    covariance, active = found
    spec = model._specs[name]
    return _spline_se(
        spec,
        name,
        model.result.beta,
        [group for group in model._groups if group.feature_name == name],
        active,
        covariance,
        x_eval=np.array(levels, dtype=object),
        reference_x=np.array([spec._base_level], dtype=object),
        center=True,
    )


def _free_categorical(spec: OrderedCategorical, column) -> Categorical:
    """``spec``'s levels as a plain categorical: the same groups, the same reference.

    The ordered term reads its column through its declaration (a column of
    1.0, 2.0 against ``order=[1, 2]`` is levels 1 and 2), so the categorical
    groups the column's own texts under the term's levels and groups, and is
    named as the term names them.
    """
    raw = pd.unique(np.asarray(column, dtype=object).ravel())
    levels = [str(level) for level in spec._canonical(raw)]
    grouping = getattr(spec, "_grouping", None)
    to_group = {
        str(text): level if grouping is None else str(grouping.original_to_group.get(level, level))
        for text, level in zip(raw, levels, strict=True)
    }
    members: dict[str, list[str]] = {}
    for text, group in to_group.items():
        members.setdefault(group, []).append(text)
    free = Categorical(
        base=str(spec._base_level),
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


def _comparison(
    term,
    spec: OrderedCategorical,
    free: dict[str, tuple[float, float]],
    curve: dict[str, tuple[float, float]],
    *,
    curve_scale: float,
    shrunk: bool,
) -> dict[str, Any]:
    """Each level's mean-centred free estimate against the mean-centred curve.

    The gap is drawn on the chart's own scale: the diamond sits that far from
    the curve, whatever the chart centres it on.
    """
    levels = [str(level) for level in term.levels]
    specials = special_labels(spec)
    grouping = getattr(spec, "_grouping", None)
    shown = np.asarray(term.original_log_effect, dtype=np.float64)
    at = {level: i for i, level in enumerate(levels)}
    found = []
    for level in levels:
        group = level if grouping is None else str(grouping.original_to_group.get(level, level))
        if level in specials or group not in free or group not in curve:
            continue
        (free_value, free_se), (curve_value, curve_se) = free[group], curve[group]
        gap = free_value - curve_value
        free_var = free_se**2
        gap_var = free_var - curve_scale * curve_se**2
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
