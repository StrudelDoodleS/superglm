"""Structured rating-table export for fitted SuperGLM models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.polynomial import Polynomial
from superglm.features.spline import _SplineBase

if TYPE_CHECKING:
    from superglm.model import SuperGLM


@dataclass(frozen=True)
class RatingTableBlock:
    """One one-dimensional rating-table block."""

    name: str
    kind: str
    table: pd.DataFrame


@dataclass(frozen=True)
class InteractionTableBlock:
    """One two-way interaction rating-table block."""

    name: str
    table: pd.DataFrame


@dataclass(frozen=True)
class RatingTablePayload:
    """Renderer-independent rating-table export payload."""

    base_relativity: float
    selected_n_bins: int
    main_effects: list[RatingTableBlock]
    interactions: list[InteractionTableBlock]
    discretization_impact: pd.DataFrame
    summary_lines: list[str]


def _resolve_format(file_path: str | Path, format: str | None) -> str:
    if format is not None:
        fmt = format.lower().lstrip(".")
    else:
        suffix = Path(file_path).suffix.lower()
        fmt = suffix.lstrip(".")
    if fmt in {"xlsx", "xlsm", "excel"}:
        return "excel"
    raise ValueError(
        f"Unsupported rating table export format: {format or Path(file_path).suffix!r}"
    )


def _continuous_features(model: SuperGLM) -> list[str]:
    return [
        name
        for name in model._feature_order
        if isinstance(model._specs.get(name), _SplineBase | Polynomial)
    ]


def _format_interval(left: float, right: float) -> str:
    return f"[{left:.10g}, {right:.10g})"


def _continuous_block(name: str, table: pd.DataFrame) -> RatingTableBlock:
    out = pd.DataFrame(
        {
            "Level": [
                _format_interval(float(row.bin_from), float(row.bin_to))
                for row in table.itertuples(index=False)
            ],
            "Relativity": table["relativity"].astype(float).to_numpy(),
            "Weight": table["sample_weight"].astype(float).to_numpy(),
        }
    )
    return RatingTableBlock(name=name, kind="continuous", table=out)


def _weights_by_level(
    X: pd.DataFrame,
    name: str,
    levels: list[str],
    sample_weight: NDArray | None,
) -> np.ndarray:
    weights = (
        np.ones(len(X), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    grouped = (
        pd.DataFrame({"level": X[name].astype(str), "weight": weights})
        .groupby("level", sort=False)["weight"]
        .sum()
    )
    return np.array([float(grouped.get(level, 0.0)) for level in levels], dtype=np.float64)


def _categorical_block(
    model: SuperGLM,
    X: pd.DataFrame,
    name: str,
    sample_weight: NDArray | None,
    centering: str,
) -> RatingTableBlock:
    ti = model.term_inference(name, with_se=False, centering=centering)
    levels = list(ti.levels or [])
    return RatingTableBlock(
        name=name,
        kind="categorical",
        table=pd.DataFrame(
            {
                "Level": levels,
                "Relativity": np.asarray(ti.relativity, dtype=np.float64),
                "Weight": _weights_by_level(X, name, levels, sample_weight),
            }
        ),
    )


def _numeric_block(model: SuperGLM, name: str, centering: str) -> RatingTableBlock:
    ti = model.term_inference(name, with_se=False, centering=centering)
    return RatingTableBlock(
        name=name,
        kind="numeric",
        table=pd.DataFrame(
            {
                "Level": ["per_unit"],
                "Relativity": np.asarray(ti.relativity, dtype=np.float64),
                "Weight": [0.0],
            }
        ),
    )


def _interaction_blocks(model: SuperGLM) -> list[InteractionTableBlock]:
    blocks: list[InteractionTableBlock] = []
    for name in model._interaction_order:
        raw = model.reconstruct_feature(name)
        if "pairs" not in raw:
            raise NotImplementedError(
                f"Interaction {name!r} is not yet exportable as a rating table."
            )

        ispec = model._interaction_specs[name]
        parent1, _ = ispec.parent_names
        levels1 = raw["levels1"]
        levels2 = raw["levels2"]
        rows = []
        for level1 in levels1:
            row: dict[str, str | float] = {parent1: level1}
            for level2 in levels2:
                key = f"{level1}:{level2}"
                row[level2] = float(raw["relativities"].get(key, 1.0))
            rows.append(row)
        blocks.append(InteractionTableBlock(name=name, table=pd.DataFrame(rows)))
    return blocks


def _empty_impact_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "n_bins",
            "feature",
            "actual_bins",
            "deviance_original",
            "deviance_discretized",
            "deviance_change",
            "deviance_change_pct",
            "mean_abs_prediction_change_pct",
            "max_abs_prediction_change_pct",
            "prediction_correlation",
        ]
    )


def _impact_sweep(
    model: SuperGLM,
    X: pd.DataFrame,
    y: NDArray,
    sample_weight: NDArray | None,
    *,
    impact_bins: tuple[int, ...],
    bin_strategy: str,
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if not features:
        return _empty_impact_frame()

    for n_bins in impact_bins:
        result = model.discretization_impact(
            X,
            y,
            sample_weight=sample_weight,
            n_bins=int(n_bins),
            bin_strategy=bin_strategy,
            features=features,
        )
        for feature, table in result.tables.items():
            row: dict[str, float | int | str] = {
                "n_bins": int(n_bins),
                "feature": feature,
                "actual_bins": int(len(table)),
            }
            row.update(result.metrics)
            rows.append(row)
    if not rows:
        return _empty_impact_frame()
    return pd.DataFrame(rows)


def build_rating_table_payload(
    model: SuperGLM,
    X: pd.DataFrame,
    y: NDArray,
    sample_weight: NDArray | None = None,
    *,
    n_bins: int = 150,
    impact_bins: tuple[int, ...] = (20, 50, 100, 200, 250),
    bin_strategy: str = "exposure_quantile",
    centering: str = "native",
) -> RatingTablePayload:
    if model._result is None:
        raise RuntimeError("Model must be fitted before exporting rating tables.")

    y_arr = np.asarray(y, dtype=np.float64)
    if len(X) != len(y_arr):
        raise ValueError("X and y must have the same length.")
    if sample_weight is not None and len(sample_weight) != len(X):
        raise ValueError("sample_weight must have the same length as X.")

    continuous = _continuous_features(model)
    selected = (
        model.discretization_impact(
            X,
            y_arr,
            sample_weight=sample_weight,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            features=continuous,
        )
        if continuous
        else None
    )

    main_effects: list[RatingTableBlock] = []
    for name in model._feature_order:
        spec = model._specs[name]
        if selected is not None and name in selected.tables:
            main_effects.append(_continuous_block(name, selected.tables[name]))
        elif isinstance(spec, Categorical | OrderedCategorical):
            main_effects.append(_categorical_block(model, X, name, sample_weight, centering))
        elif isinstance(spec, Numeric):
            main_effects.append(_numeric_block(model, name, centering))

    impact = _impact_sweep(
        model,
        X,
        y_arr,
        sample_weight,
        impact_bins=impact_bins,
        bin_strategy=bin_strategy,
        features=continuous,
    )
    return RatingTablePayload(
        base_relativity=float(np.exp(model.result.intercept)),
        selected_n_bins=int(n_bins),
        main_effects=main_effects,
        interactions=_interaction_blocks(model),
        discretization_impact=impact,
        summary_lines=str(model.summary(detail="compact")).splitlines(),
    )


def export_rating_tables(
    model: SuperGLM,
    file_path: str | Path,
    X: pd.DataFrame,
    y: NDArray,
    sample_weight: NDArray | None = None,
    *,
    n_bins: int = 150,
    impact_bins: tuple[int, ...] = (20, 50, 100, 200, 250),
    bin_strategy: str = "exposure_quantile",
    format: str | None = None,
    sheet_name: str = "Rating Tables",
    summary_sheet_name: str = "Model Summary",
    impact_sheet_name: str = "Discretization Impact",
    centering: str = "native",
) -> Path:
    out = Path(file_path)
    fmt = _resolve_format(out, format)
    if fmt != "excel":
        raise ValueError(f"Unsupported rating table export format: {fmt!r}")

    payload = build_rating_table_payload(
        model,
        X,
        y,
        sample_weight=sample_weight,
        n_bins=n_bins,
        impact_bins=impact_bins,
        bin_strategy=bin_strategy,
        centering=centering,
    )

    from superglm.export.excel import write_rating_table_workbook

    write_rating_table_workbook(
        payload,
        out,
        sheet_name=sheet_name,
        summary_sheet_name=summary_sheet_name,
        impact_sheet_name=impact_sheet_name,
    )
    return out
