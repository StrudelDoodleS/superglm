"""Comparison data builders for labeled fitted-model term overlays."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
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
    if isinstance(spec, Numeric | Polynomial | _SplineBase):
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
    X: pd.DataFrame, term: str, n_points: int
) -> dict[str, NDArray[np.float64]]:
    """Build a shared continuous x-grid from the passed comparison data."""
    values = np.asarray(X[term], dtype=np.float64)
    return {"x": np.linspace(float(values.min()), float(values.max()), n_points)}


def _shared_level_domain(
    models: Mapping[str, Any],
    X: pd.DataFrame,
    term: str,
) -> dict[str, list[str]]:
    """Build a shared categorical/ordered level domain."""
    ordered_levels: list[str] | None = None
    for model in models.values():
        spec = model._specs[term]
        if isinstance(spec, OrderedCategorical):
            ordered_levels = [str(level) for level in spec._ordered_levels]
            break

    observed_levels = [
        str(level) for level in pd.Series(X[term]).astype(str).drop_duplicates().tolist()
    ]
    if ordered_levels is None:
        return {"levels": observed_levels}

    merged = [level for level in ordered_levels if level in observed_levels]
    for level in observed_levels:
        if level not in merged:
            merged.append(level)
    return {"levels": merged}


def _support_payload(
    family: str,
    X: pd.DataFrame,
    term: str,
    sample_weight: NDArray[np.float64] | None,
    domain: dict[str, Any],
) -> dict[str, Any] | None:
    """Build one shared support payload from the passed comparison data."""
    if sample_weight is None:
        sample_weight = np.ones(len(X), dtype=np.float64)

    if family == "continuous":
        values = np.asarray(X[term], dtype=np.float64)
        weights = np.asarray(sample_weight, dtype=np.float64)
        grid = np.asarray(domain["x"], dtype=np.float64)
        density = _exposure_kde(values, weights, grid)
        return {"x": grid, "density": density}

    level_series = pd.Series(X[term]).astype(str)
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
    X: pd.DataFrame,
    sample_weight: NDArray | None = None,
    support_by_label: Mapping[str, dict[str, Any]] | None = None,
    n_points: int = 200,
) -> dict[str, Any]:
    """Build normalized per-term comparison data for labeled fitted models."""
    normalized_models = _normalize_models(models)
    resolved_terms, skipped = _resolve_comparable_terms(normalized_models, terms=terms)
    if not resolved_terms:
        raise ValueError("No comparable main-effect terms were found for the supplied models.")

    sample_weight_arr = (
        None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )
    payload_terms: list[dict[str, Any]] = []

    for term in resolved_terms:
        family = _comparison_family(next(iter(normalized_models.values()))._specs[term])
        if family == "continuous":
            domain = _shared_continuous_domain(X, term, n_points)
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
            domain = _shared_level_domain(normalized_models, X, term)
            levels = np.asarray(domain["levels"], dtype=object)
            series = {
                label: {
                    "link": np.asarray(
                        model._specs[term].score(levels, _feature_beta(model, term)),
                        dtype=np.float64,
                    ),
                }
                for label, model in normalized_models.items()
            }

        for entry in series.values():
            entry["response"] = np.exp(entry["link"])

        if support_by_label is None:
            support = _support_payload(family, X, term, sample_weight_arr, domain)
        else:
            support_series: dict[str, Any] = {}
            for label, support_data in support_by_label.items():
                support_series[label] = _support_payload(
                    family,
                    support_data["X"],
                    term,
                    support_data.get("sample_weight"),
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
    X: pd.DataFrame,
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
