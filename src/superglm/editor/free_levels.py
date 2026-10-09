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
from superglm.inference._term_covariance import feature_se_from_cov

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
_NO_DATA = (
    "Comparing with free levels refits the model, which needs its training data: open the "
    "editor with train_data, or from a model fitted with its data kept."
)


def free_level_comparison(session, name: str) -> dict[str, Any]:
    """Refit the in-force model with ``name``'s levels free and compare them with its curve.

    Returns the payload the chart draws: for each level on the curve, its
    free estimate and interval as relativities on the chart's own scale (at
    the reference both agree), and the levels whose interval misses the
    curve. Hand edits are not part of either: both are fits.
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
    free_levels = _centred(free_model, name)
    curve = _centred(session.model, name)
    # The two fits estimate their dispersion apart; the curve's variance is
    # put on the free fit's, so the two variances share one scale.
    scale = float(free_model.result.phi) / float(session.model.result.phi)
    return _comparison(term, spec, free_levels, curve, curve_scale=scale, shrunk=shrunk)


def _centred(model, name: str) -> dict[str, tuple[float, float]]:
    """Each fitted level's log-relativity and its standard error, centred on the mean of the levels.

    The mean is over the term's fitted levels, its groups and specials
    included, as for a mean-centred report; the errors are those of that
    contrast, from the fit's covariance. A term the selection penalty
    removed is flat, with no variance.
    """
    spec = model._specs[name]
    grouping = getattr(spec, "_grouping", None)
    fitted = spec._ordered_levels if isinstance(spec, OrderedCategorical) else spec._levels
    labels = [str(level) for level in fitted]
    native = model.term_inference(name, with_se=False)
    reported = {
        str(level): float(value)
        for level, value in zip(native.levels, native.log_relativity, strict=True)
    }

    def value(label: str) -> float:
        # A grouped fit may report its group's members rather than the group.
        members = [] if grouping is None else grouping.group_to_originals.get(label, [])
        found = [reported[str(m)] for m in [label, *members] if str(m) in reported]
        return found[0] if found else 0.0

    values = np.array([value(label) for label in labels])
    if native.active:
        covariance, active = model._coef_covariance
        se = feature_se_from_cov(
            name,
            covariance,
            active,
            model.result,
            model._groups,
            model._specs,
            model._interaction_specs,
            center=True,
        )
    else:
        se = np.zeros(len(labels))
    centred = values - values.mean()
    return {label: (float(c), float(s)) for label, c, s in zip(labels, centred, se, strict=True)}


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
                "y": float(np.exp(on_curve + gap)),
                "lower": float(np.exp(on_curve + gap - half)),
                "upper": float(np.exp(on_curve + gap + half)),
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
