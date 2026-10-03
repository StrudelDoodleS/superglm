"""Comparison data builders for labeled fitted-model term overlays."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.piecewise import Piecewise
from superglm.features.polynomial import Polynomial
from superglm.features.spline import _SplineBase
from superglm.plotting.common import _exposure_kde


def _normalize_models(models: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the labeled model mapping and validate that it is non-empty."""
    normalized = dict(models)
    if not normalized:
        raise ValueError("models must contain at least one fitted model.")
    return normalized


def _comparison_family(spec) -> str | None:
    """Map a feature spec to its comparison family."""
    if isinstance(spec, Categorical | OrderedCategorical):
        return "level"
    # Piecewise belongs here: the continuous path evaluates `spec.score` on a
    # shared grid, which a piecewise term answers exactly (and past both
    # boundary knots).  Left out, the term was reported as "missing or
    # unsupported in one or more models" even when both models declared it
    # identically, sending the reader after a column that is not absent.
    if isinstance(spec, Numeric | Piecewise | Polynomial | _SplineBase):
        return "continuous"
    return None


def _feature_beta(model, term: str) -> NDArray[np.float64]:
    """Extract the fitted coefficient block for one feature."""
    groups = model._feature_groups(term)
    return np.concatenate([np.asarray(model.result.beta[g.sl], dtype=np.float64) for g in groups])


def _resolve_comparable_terms(
    models: Mapping[str, Any],
    terms: str | list[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Resolve overlapping comparable terms across all supplied models."""
    normalized = _normalize_models(models)
    model_items = list(normalized.items())
    first_label, first_model = model_items[0]
    del first_label

    if terms is None:
        candidate_terms = list(first_model._feature_order)
    elif isinstance(terms, str):
        candidate_terms = [terms]
    else:
        candidate_terms = list(terms)

    resolved: list[str] = []
    skipped: dict[str, str] = {}

    for term in candidate_terms:
        if term not in first_model._specs:
            skipped[term] = "not a main effect in the reference model"
            continue

        families = []
        missing = False
        for _, model in model_items:
            if term not in model._specs:
                missing = True
                break
            family = _comparison_family(model._specs[term])
            if family is None:
                missing = True
                break
            families.append(family)

        if missing:
            skipped[term] = "missing or unsupported in one or more models"
            continue

        if len(set(families)) != 1:
            skipped[term] = "incompatible term families across models"
            continue

        resolved.append(term)

    return resolved, skipped


def _shared_continuous_domain(
    X: EagerFrame, term: str, n_points: int
) -> dict[str, NDArray[np.float64]]:
    """Build a shared continuous x-grid from the passed comparison data."""
    values = X.column_array(term, dtype=np.float64)
    return {"x": np.linspace(float(values.min()), float(values.max()), n_points)}


def _model_level_order(spec) -> list[str]:
    """The level order a fitted level term reports in, as text.

    An ordered term's declared order; a grouped categorical's original levels,
    which is how its term inference expands the groups; otherwise the fitted
    universe, which keeps native order (1, 2, 10, not "1", "10", "2").
    """
    if isinstance(spec, OrderedCategorical):
        return [str(level) for level in spec._ordered_levels]
    grouping = getattr(spec, "_grouping", None)
    if grouping is not None:
        return [str(level) for level in grouping.all_original_levels]
    return [str(level) for level in spec._levels]


def _shared_level_domain(
    models: Mapping[str, Any],
    X: EagerFrame,
    term: str,
) -> dict[str, list[str]]:
    """Build a shared categorical/ordered level domain in model order.

    An ordered term's order wins; otherwise the first model's fitted order.
    Observed labels the model order lacks follow in row order.
    """
    specs = [model._specs[term] for model in models.values()]
    spec = next((s for s in specs if isinstance(s, OrderedCategorical)), specs[0])

    observed_levels = [
        str(level)
        for level in pd.Series(X.column_array(term), name=term)
        .astype(str)
        .drop_duplicates()
        .tolist()
    ]
    observed = set(observed_levels)
    merged = [level for level in _model_level_order(spec) if level in observed]
    placed = set(merged)
    merged.extend(level for level in observed_levels if level not in placed)
    return {"levels": merged}


def _native_level_values(X: EagerFrame, term: str, labels: list[str]) -> NDArray:
    """The column's own values for the domain's text labels, as predict receives them.

    A fitted universe keeps native types, so an integer-coded categorical
    refuses the text "1" as an unseen level.
    """
    native: dict[str, Any] = {}
    for value in pd.Series(X.column_array(term), name=term).drop_duplicates().tolist():
        native.setdefault(str(value), value)
    return np.asarray([native.get(label, label) for label in labels], dtype=object)


def _score_levels(spec, levels: NDArray, beta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Score a level term on ``levels``, NaN at a level the model refuses.

    A fold model that never saw a level the comparison frame holds refuses
    it as unseen. It has no value there, so that level is a gap in its
    curve rather than a failure of the whole comparison. The levels are
    scored one by one only when the model refuses the whole vector.
    """
    try:
        return np.asarray(spec.score(levels, beta), dtype=np.float64)
    except ValueError:
        pass
    values = np.full(len(levels), np.nan, dtype=np.float64)
    for index in range(len(levels)):
        try:
            values[index] = spec.score(levels[index : index + 1], beta)[0]
        except ValueError:
            continue
    return values


def _support_payload(
    family: str,
    X: EagerFrame,
    term: str,
    sample_weight: NDArray[np.float64] | None,
    domain: dict[str, Any],
) -> dict[str, Any] | None:
    """Build one shared support payload from the passed comparison data."""
    if sample_weight is None:
        sample_weight = np.ones(len(X), dtype=np.float64)

    if family == "continuous":
        values = X.column_array(term, dtype=np.float64)
        weights = np.asarray(sample_weight, dtype=np.float64)
        grid = np.asarray(domain["x"], dtype=np.float64)
        density = _exposure_kde(values, weights, grid)
        return {"x": grid, "density": density}

    level_series = pd.Series(X.column_array(term), name=term).astype(str)
    grouped = (
        pd.DataFrame({"level": level_series, "sample_weight": sample_weight})
        .groupby("level", sort=False)["sample_weight"]
        .sum()
    )
    levels = list(domain["levels"])
    weights = np.array([float(grouped.get(level, 0.0)) for level in levels], dtype=np.float64)
    return {"levels": levels, "density": weights}


def _build_term_comparison_data(
    *,
    models: Mapping[str, Any],
    terms: str | list[str] | tuple[str, ...] | None,
    X: FrameLike | EagerFrame,
    sample_weight: NDArray | None = None,
    support_by_label: Mapping[str, dict[str, Any]] | None = None,
    n_points: int = 200,
) -> dict[str, Any]:
    """Build normalized per-term comparison data for labeled fitted models."""
    normalized_models = _normalize_models(models)
    frame = as_eager_frame(X)
    resolved_terms, skipped = _resolve_comparable_terms(normalized_models, terms=terms)
    if not resolved_terms:
        raise ValueError("No comparable main-effect terms were found for the supplied models.")

    sample_weight_arr = (
        None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )
    normalized_support = (
        None
        if support_by_label is None
        else {
            label: {**support_data, "X": as_eager_frame(support_data["X"])}
            for label, support_data in support_by_label.items()
        }
    )
    payload_terms: list[dict[str, Any]] = []

    for term in resolved_terms:
        family = _comparison_family(next(iter(normalized_models.values()))._specs[term])
        if family == "continuous":
            domain = _shared_continuous_domain(frame, term, n_points)
            x = np.asarray(domain["x"], dtype=np.float64)
            series = {
                label: {
                    "link": np.asarray(
                        model._specs[term].score(x, _feature_beta(model, term)), dtype=np.float64
                    ),
                }
                for label, model in normalized_models.items()
            }
        else:
            domain = _shared_level_domain(normalized_models, frame, term)
            levels = _native_level_values(frame, term, domain["levels"])
            series = {
                label: {
                    "link": _score_levels(model._specs[term], levels, _feature_beta(model, term)),
                }
                for label, model in normalized_models.items()
            }

        for entry in series.values():
            entry["response"] = np.exp(entry["link"])

        if normalized_support is None:
            support = _support_payload(family, frame, term, sample_weight_arr, domain)
        else:
            support_series: dict[str, Any] = {}
            for label, support_data in normalized_support.items():
                support_series[label] = _support_payload(
                    family,
                    support_data["X"],
                    term,
                    cast(NDArray | None, support_data.get("sample_weight")),
                    domain,
                )
            support = {"mode": "by_label", "series": support_series}

        payload_terms.append(
            {
                "name": term,
                "family": family,
                "domain": domain,
                "series": series,
                "support": support,
            }
        )

    return {
        "kind": "term_comparison",
        "terms": payload_terms,
        "skipped_terms": skipped,
    }


def plot_term_comparison(
    *,
    models: Mapping[str, Any],
    terms: str | list[str] | tuple[str, ...] | None = None,
    X: FrameLike,
    sample_weight: NDArray | None = None,
    support_by_label: Mapping[str, dict[str, Any]] | None = None,
    engine: str = "plotly",
    n_points: int = 200,
    title: str | None = None,
    subtitle: str | None = None,
    plotly_style: dict[str, Any] | None = None,
):
    """Public comparison entry point for labeled fitted-model overlays."""
    payload = _build_term_comparison_data(
        models=models,
        terms=terms,
        X=X,
        sample_weight=sample_weight,
        support_by_label=support_by_label,
        n_points=n_points,
    )
    if engine != "plotly":
        raise ValueError("engine='plotly' is the only supported comparison backend.")
    from superglm.plotting.comparison_plotly import plot_term_comparison_plotly

    return plot_term_comparison_plotly(
        payload,
        title=title,
        subtitle=subtitle,
        style=plotly_style,
    )
