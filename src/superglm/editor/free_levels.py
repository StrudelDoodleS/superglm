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
from scipy.stats import norm

from superglm._frame import as_eager_frame
from superglm.editor.collapse import _require_not_interaction_parent
from superglm.editor.errors import EditorTypeError, EditorValueError
from superglm.editor.refit import fit_refit_model
from superglm.features.categorical import Categorical
from superglm.features.grouping import native_by_text
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.rebuild import clone_with_replaced_features, special_labels

# The chance, across all the levels of one term, that no level's interval
# misses the curve when every level lies on it.
CONFIDENCE = 0.95
# Past this leverage the curve all but passes through the level's own
# estimate, so their gap says nothing and the level is not judged.
_MAX_LEVERAGE = 0.99
_NOT_ORDERED = (
    "Compare with free levels is for ordered terms: every level of {term!r} is free already."
)
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
    column = as_eager_frame(X).column_array(name)
    free_model = clone_with_replaced_features(
        session.model, {name: _free_categorical(spec, column)}
    )
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
    inference = free_model.term_inference(name, with_se=True)
    curve = session.model.term_inference(name, with_se=True)
    return _comparison(term, spec, inference, curve, shrunk=shrunk)


def _curve_se(inference, grouping) -> dict[str, float]:
    """The curve's standard error at each displayed level, a member taking its group's."""
    se = {
        str(level): float(value)
        for level, value in zip(inference.levels, inference.se_log_relativity, strict=True)
    }
    if grouping is not None:
        for original, group in grouping.original_to_group.items():
            if str(original) not in se and str(group) in se:
                se[str(original)] = se[str(group)]
    return se


def _free_categorical(spec: OrderedCategorical, column) -> Categorical:
    """``spec``'s levels as a plain categorical: the same groups, the same reference."""
    grouping = getattr(spec, "_grouping", None)
    base = spec._base_level
    if grouping is None:
        base = native_by_text(np.asarray(column, dtype=object).ravel()).get(str(base), base)
    return Categorical(base=base, grouping=grouping)


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
    term, spec: OrderedCategorical, inference, curve_inference, *, shrunk: bool
) -> dict[str, Any]:
    levels = [str(level) for level in term.levels]
    specials = special_labels(spec)
    grouping = getattr(spec, "_grouping", None)
    fitted = {str(level): i for i, level in enumerate(inference.levels)}
    curve_se = _curve_se(curve_inference, grouping)
    curve = np.asarray(term.original_log_effect, dtype=np.float64)
    compared = [level for level in levels if level not in specials]
    reference = str(spec._base_level)
    at = {level: i for i, level in enumerate(levels)}
    # Both are relativities to the same reference, so they agree there; the
    # chart may centre its curve elsewhere, and the free levels move with it.
    if reference not in at and grouping is not None:
        # A group is the reference: its members share its value on the curve.
        reference = next(iter(grouping.group_to_originals.get(reference, [])), reference)
    anchor = curve[at[reference]] if reference in at else 0.0
    z = float(norm.ppf(0.5 + 0.5 * CONFIDENCE ** (1.0 / max(len(compared), 1))))
    rows = []
    for level in compared:
        # A grouped fit reports each member under its own name, or the group's.
        group = level if grouping is None else str(grouping.original_to_group.get(level, level))
        index = fitted.get(level, fitted.get(group))
        if index is None:
            continue
        estimate = float(inference.log_relativity[index]) + anchor
        free_var = float(inference.se_log_relativity[index]) ** 2
        gap_var = free_var - curve_se.get(level, 0.0) ** 2
        judged = free_var > 0.0 and gap_var > (1.0 - _MAX_LEVERAGE) * free_var
        se = float(np.sqrt(gap_var if judged else free_var))
        lower, upper = estimate - z * se, estimate + z * se
        on_curve = float(curve[at[level]])
        rows.append(
            {
                "level": level,
                "y": float(np.exp(estimate)),
                "lower": float(np.exp(lower)),
                "upper": float(np.exp(upper)),
                "flagged": bool(judged and not lower <= on_curve <= upper),
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
