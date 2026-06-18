"""Structured rating-table export for fitted SuperGLM models."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
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
from superglm.links import LogLink

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


def _format_axis_value(value: float) -> str:
    return f"{value:.10g}"


def _continuous_block(name: str, table: pd.DataFrame) -> RatingTableBlock:
    out = pd.DataFrame(
        {
            name: [
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
                name: levels,
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
                name: ["per_unit"],
                "Relativity": np.asarray(ti.relativity, dtype=np.float64),
                "Weight": [0.0],
            }
        ),
    )


def _fit_used_offset(model: SuperGLM) -> bool:
    return bool(
        getattr(
            model,
            "_fit_used_offset",
            getattr(model, "_fit_offset", None) is not None,
        )
    )


def _require_log_link_offset_export(model: SuperGLM) -> None:
    if not isinstance(model._link, LogLink):
        raise ValueError(
            "Rating-table offset relativities are currently supported only for log-link models."
        )


def _resolve_export_offset(
    offset,
    model: SuperGLM,
    X: pd.DataFrame,
) -> NDArray | None:
    if offset is not None:
        offset_arr = np.asarray(offset, dtype=np.float64).ravel()
        if len(offset_arr) != len(X):
            raise ValueError("offset must have the same length as X.")
        return offset_arr

    fit_offset = getattr(model, "_fit_offset", None)
    if X is getattr(model, "_fit_X_ref", None) and fit_offset is not None:
        offset_arr = np.asarray(fit_offset, dtype=np.float64).ravel()
        if len(offset_arr) != len(X):
            raise ValueError(
                "The fitted offset has a different length from X; pass offset= "
                "when exporting a frame other than the original fit frame."
            )
        return offset_arr

    raise ValueError("Pass offset= when exporting a frame other than the original fit frame.")


def _resolve_offset_source(
    offset_source,
    X: pd.DataFrame,
    *,
    offset_name: str | None,
) -> tuple[pd.Series, str]:
    if isinstance(offset_source, str):
        if offset_source not in X:
            raise ValueError(f"offset_source column {offset_source!r} is not present in X.")
        source = pd.Series(X[offset_source].to_numpy(), name=offset_source)
        name = offset_name if offset_name is not None else offset_source
    elif isinstance(offset_source, pd.Series):
        source = offset_source.reset_index(drop=True)
        if offset_name is not None:
            name = offset_name
        elif offset_source.name is not None:
            name = str(offset_source.name)
        else:
            raise ValueError(
                "offset_name is required when offset_source is an unnamed array-like object."
            )
    else:
        if offset_name is None:
            raise ValueError(
                "offset_name is required when offset_source is an unnamed array-like object."
            )
        source = pd.Series(offset_source)
        name = offset_name

    if len(source) != len(X):
        raise ValueError("offset_source must have the same length as X.")
    if source.isna().any():
        raise ValueError("offset_source cannot contain missing values.")
    return source.reset_index(drop=True), name


def _weights_array(n_rows: int, sample_weight: NDArray | None) -> NDArray:
    if sample_weight is None:
        return np.ones(n_rows, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64).ravel()
    if len(weights) != n_rows:
        raise ValueError("sample_weight must have the same length as X.")
    return weights


def _offset_multiplier_block(
    offset: NDArray,
    n_rows: int,
    sample_weight: NDArray | None,
    *,
    n_bins: int,
    bin_strategy: str,
) -> RatingTableBlock | None:
    offset_arr = np.asarray(offset, dtype=np.float64).ravel()
    if len(offset_arr) != n_rows:
        raise ValueError("offset must have the same length as X.")

    weights = _weights_array(n_rows, sample_weight)
    multiplier = np.exp(offset_arr)
    exact_multiplier = np.round(multiplier, 12)
    levels, inverse = np.unique(exact_multiplier, return_inverse=True)

    if len(levels) < 20:
        exposure = np.bincount(inverse, weights=weights, minlength=len(levels))
        table = pd.DataFrame(
            {
                "Offset Multiplier": levels.astype(float),
                "Relativity": levels.astype(float),
                "Weight": exposure.astype(float),
            }
        )
        return RatingTableBlock(name="Offset Multiplier", kind="offset", table=table)

    from superglm.diagnostics.discretize import _compute_edges

    edges = _compute_edges(multiplier, weights, n_bins, bin_strategy)
    actual_n_bins = len(edges) - 1
    bin_idx = np.digitize(multiplier, edges, right=False)
    bin_idx = np.clip(bin_idx, 1, actual_n_bins) - 1

    rows: list[dict[str, str | float]] = []
    for b in range(actual_n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            avg_multiplier = 0.0
            exposure = 0.0
        else:
            exposure = float(weights[mask].sum())
            avg_multiplier = float(np.average(multiplier[mask], weights=weights[mask]))
        rows.append(
            {
                "Offset Multiplier": _format_interval(float(edges[b]), float(edges[b + 1])),
                "Relativity": avg_multiplier,
                "Weight": exposure,
            }
        )
    return RatingTableBlock(
        name="Offset Multiplier",
        kind="offset",
        table=pd.DataFrame(rows),
    )


def _offset_source_block(
    offset: NDArray,
    offset_source,
    X: pd.DataFrame,
    sample_weight: NDArray | None,
    *,
    offset_name: str | None,
    offset_kind: str,
    offset_max_exact_levels: int,
    offset_mapping_rtol: float,
    offset_mapping_atol: float,
) -> RatingTableBlock:
    if offset_kind not in {"auto", "discrete"}:
        raise ValueError("offset_kind must be 'auto' or 'discrete'.")

    source, source_name = _resolve_offset_source(offset_source, X, offset_name=offset_name)
    n_unique = int(source.nunique(dropna=False))
    if n_unique > offset_max_exact_levels:
        raise ValueError(
            f"Offset source {source_name!r} has {n_unique} distinct values, exceeding "
            f"offset_max_exact_levels={offset_max_exact_levels}. Increase "
            "offset_max_exact_levels explicitly if all values are intended tariff levels."
        )

    offset_arr = np.asarray(offset, dtype=np.float64).ravel()
    weights = _weights_array(len(X), sample_weight)
    df = pd.DataFrame(
        {
            "__offset_source__": source,
            "__offset__": offset_arr,
            "__weight__": weights,
        }
    )

    rows: list[dict[str, object | float]] = []
    for level, group in df.groupby("__offset_source__", sort=False, dropna=False):
        offset_values = group["__offset__"].to_numpy(dtype=np.float64)
        multipliers = np.exp(offset_values)
        group_weights = group["__weight__"].to_numpy(dtype=np.float64)
        weight_sum = float(group_weights.sum())
        if weight_sum > 0.0:
            representative = float(np.exp(np.average(offset_values, weights=group_weights)))
        else:
            representative = float(multipliers[0])
        if not np.allclose(
            multipliers,
            representative,
            rtol=offset_mapping_rtol,
            atol=offset_mapping_atol,
        ):
            raise ValueError(
                f"Offset source {source_name!r} is not a valid discrete lookup: "
                f"level {level!r} maps to multiple offset multipliers. Pass a more "
                "granular offset_source, or keep the offset calculation outside the "
                "rating table."
            )
        rows.append(
            {
                source_name: level,
                "Relativity": representative,
                "Weight": weight_sum,
            }
        )

    return RatingTableBlock(
        name=source_name,
        kind="offset",
        table=pd.DataFrame(rows, columns=[source_name, "Relativity", "Weight"]),
    )


def _interaction_beta(model: SuperGLM, name: str) -> np.ndarray:
    groups = [g for g in model._groups if g.feature_name == name]
    return np.concatenate([model.result.beta[g.sl] for g in groups])


def _reconstruct_interaction(ispec, beta: NDArray, n_bins: int) -> dict:
    if "n_points" in signature(ispec.reconstruct).parameters:
        return ispec.reconstruct(beta, n_points=n_bins)
    return ispec.reconstruct(beta)


def _continuous_interaction_block(
    name: str,
    raw: dict,
    parent1: str,
    parent2: str,
) -> InteractionTableBlock:
    x1 = np.asarray(raw["x1"], dtype=np.float64)
    x2 = np.asarray(raw["x2"], dtype=np.float64)
    relativity = np.asarray(raw["relativity"], dtype=np.float64)
    if relativity.shape == (len(x2), len(x1)):
        relativity = relativity.T
    elif relativity.shape != (len(x1), len(x2)):
        raise ValueError(
            f"Interaction {name!r} returned a {relativity.shape} relativity grid, "
            f"expected {(len(x1), len(x2))} or {(len(x2), len(x1))}."
        )

    table = pd.DataFrame(relativity, columns=[_format_axis_value(v) for v in x2])
    table.insert(0, parent1, [_format_axis_value(v) for v in x1])
    return InteractionTableBlock(name=name, table=table)


def _interaction_blocks(model: SuperGLM, n_bins: int) -> list[InteractionTableBlock]:
    blocks: list[InteractionTableBlock] = []
    for name in model._interaction_order:
        ispec = model._interaction_specs[name]
        parent1, _ = ispec.parent_names
        raw = _reconstruct_interaction(ispec, _interaction_beta(model, name), n_bins)
        if {"x1", "x2", "relativity"} <= set(raw):
            parent2 = ispec.parent_names[1]
            blocks.append(_continuous_interaction_block(name, raw, parent1, parent2))
            continue

        if "pairs" not in raw:
            raise NotImplementedError(
                f"Interaction {name!r} is not yet exportable as a rating table."
            )

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
    offset: NDArray | None,
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
            offset=offset,
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
    offset: NDArray | None = None,
    offset_source=None,
    offset_name: str | None = None,
    offset_kind: str = "auto",
    offset_max_exact_levels: int = 20,
    offset_mapping_rtol: float = 1e-10,
    offset_mapping_atol: float = 1e-12,
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

    export_offset: NDArray | None = None
    if _fit_used_offset(model):
        _require_log_link_offset_export(model)
        export_offset = _resolve_export_offset(offset, model, X)
    elif offset is not None or offset_source is not None:
        raise ValueError("Offset rating-table export requires a model fitted with an offset.")

    continuous = _continuous_features(model)
    selected = (
        model.discretization_impact(
            X,
            y_arr,
            sample_weight=sample_weight,
            offset=export_offset,
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

    if export_offset is not None:
        if offset_source is None:
            offset_block = _offset_multiplier_block(
                export_offset,
                len(X),
                sample_weight,
                n_bins=n_bins,
                bin_strategy=bin_strategy,
            )
        else:
            offset_block = _offset_source_block(
                export_offset,
                offset_source,
                X,
                sample_weight,
                offset_name=offset_name,
                offset_kind=offset_kind,
                offset_max_exact_levels=offset_max_exact_levels,
                offset_mapping_rtol=offset_mapping_rtol,
                offset_mapping_atol=offset_mapping_atol,
            )
        main_effects.append(offset_block)

    impact = _impact_sweep(
        model,
        X,
        y_arr,
        sample_weight,
        offset=export_offset,
        impact_bins=impact_bins,
        bin_strategy=bin_strategy,
        features=continuous,
    )
    return RatingTablePayload(
        base_relativity=float(np.exp(model.result.intercept)),
        selected_n_bins=int(n_bins),
        main_effects=main_effects,
        interactions=_interaction_blocks(model, n_bins),
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
    offset: NDArray | None = None,
    offset_source=None,
    offset_name: str | None = None,
    offset_kind: str = "auto",
    offset_max_exact_levels: int = 20,
    offset_mapping_rtol: float = 1e-10,
    offset_mapping_atol: float = 1e-12,
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
        offset=offset,
        offset_source=offset_source,
        offset_name=offset_name,
        offset_kind=offset_kind,
        offset_max_exact_levels=offset_max_exact_levels,
        offset_mapping_rtol=offset_mapping_rtol,
        offset_mapping_atol=offset_mapping_atol,
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
