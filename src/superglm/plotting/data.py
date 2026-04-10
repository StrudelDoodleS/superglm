"""Plain plot-data exporters for SuperGLM terms and interactions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.inference.term import SmoothCurve, TermInference
from superglm.plotting.common import _exposure_kde, _kde_2d
from superglm.plotting.interactions import _highest_density_mass_field, _reconstruct_interaction


def build_main_effect_plot_data(
    model,
    terms: list[TermInference],
    *,
    X: pd.DataFrame | None = None,
    sample_weight: NDArray | None = None,
    show_density: bool = True,
    show_knots: bool = False,
    show_bases: bool = False,
) -> dict[str, Any]:
    """Return plot-ready data payloads for one or more main effects."""
    sample_weight = _normalize_sample_weight(X, sample_weight)
    return {
        "kind": "main_effects",
        "terms": [
            _build_single_main_effect_payload(
                model,
                ti,
                X=X,
                sample_weight=sample_weight,
                show_density=show_density,
                show_knots=show_knots,
                show_bases=show_bases,
            )
            for ti in terms
        ],
    }


def build_interaction_plot_data(
    model,
    name: str,
    *,
    n_points: int = 200,
    X: pd.DataFrame | None = None,
    sample_weight: NDArray | None = None,
) -> dict[str, Any]:
    """Return plot-ready data payload for one interaction."""
    if name not in model._interaction_specs:
        raise KeyError(
            f"Interaction not found: {name!r}. Available: {list(model._interaction_specs.keys())}"
        )

    ispec = model._interaction_specs[name]
    parent_names = ispec.parent_names
    raw = _reconstruct_interaction(model, name, n_points=n_points)
    sample_weight = _normalize_sample_weight(X, sample_weight)

    payload: dict[str, Any] = {
        "kind": "interaction",
        "name": name,
        "parents": list(parent_names),
        "plot_kind": _interaction_plot_kind(raw),
        "metadata": {"n_points": n_points},
    }

    if "per_level" in raw and "x" in raw:
        payload["effect"] = _varying_coefficient_dataframe(raw, parent_names)
        return payload

    if "pairs" in raw:
        payload["effect"] = _categorical_heatmap_dataframe(raw, parent_names)
        return payload

    if "relativities_per_unit" in raw:
        payload["effect"] = _numeric_categorical_dataframe(raw, parent_names)
        return payload

    if "relativity_per_unit_unit" in raw:
        payload["effect"] = _numeric_interaction_dataframe(raw, parent_names)
        return payload

    if "x1" in raw and "x2" in raw:
        x1 = np.asarray(raw["x1"], dtype=np.float64)
        x2 = np.asarray(raw["x2"], dtype=np.float64)
        z = np.asarray(raw["relativity"], dtype=np.float64)
        X1, X2 = np.meshgrid(x1, x2)
        payload["grid_axes"] = {
            parent_names[0]: x1.copy(),
            parent_names[1]: x2.copy(),
        }
        payload["effect"] = pd.DataFrame(
            {
                parent_names[0]: X1.ravel(),
                parent_names[1]: X2.ravel(),
                "relativity": z.ravel(),
                "log_relativity": np.log(np.maximum(z.ravel(), 1e-300)),
            }
        )
        if X is not None and sample_weight is not None:
            density = _surface_density_dataframe(X, sample_weight, parent_names, x1, x2)
            if density is not None:
                payload["density"] = density
        return payload

    raise ValueError(f"Cannot determine plot-data structure for interaction {name!r}.")


def _build_single_main_effect_payload(
    model,
    ti: TermInference,
    *,
    X: pd.DataFrame | None,
    sample_weight: NDArray | None,
    show_density: bool,
    show_knots: bool,
    show_bases: bool,
) -> dict[str, Any]:
    effect = ti.to_dataframe()
    if ti.smooth_curve is not None:
        level_x = ti.smooth_curve.level_x
        if level_x is not None and len(effect) == len(level_x):
            effect = effect.copy()
            effect["x_position"] = np.asarray(level_x, dtype=np.float64)

    payload: dict[str, Any] = {
        "name": ti.name,
        "term_kind": ti.kind,
        "effect": effect,
        "smooth_curve": _smooth_curve_dataframe(ti.smooth_curve),
        "density": None,
        "knots": None,
        "bases": None,
        "metadata": {
            "active": ti.active,
            "alpha": ti.alpha,
            "centering_mode": ti.centering_mode,
            "edf": ti.edf,
            "smoothing_lambda": ti.smoothing_lambda,
        },
    }
    if ti.spline is not None:
        payload["metadata"]["spline"] = {
            "kind": ti.spline.kind,
            "knot_strategy": ti.spline.knot_strategy,
            "interior_knots": np.asarray(ti.spline.interior_knots, dtype=np.float64).copy(),
            "boundary": tuple(float(v) for v in ti.spline.boundary),
            "n_basis": ti.spline.n_basis,
            "degree": ti.spline.degree,
            "extrapolation": ti.spline.extrapolation,
            "knot_alpha": ti.spline.knot_alpha,
        }

    if show_density:
        payload["density"] = _main_effect_density_dataframe(ti, X, sample_weight)
    if show_knots:
        payload["knots"] = _main_effect_knots_dataframe(ti)
    if show_bases:
        payload["bases"] = _main_effect_basis_dataframe(model, ti)
    return payload


def _normalize_sample_weight(
    X: pd.DataFrame | None, sample_weight: NDArray | None
) -> NDArray | None:
    if X is not None and sample_weight is None:
        return np.ones(len(X), dtype=np.float64)
    if sample_weight is None:
        return None
    return np.asarray(sample_weight, dtype=np.float64)


def _smooth_curve_dataframe(curve: SmoothCurve | None) -> pd.DataFrame | None:
    if curve is None:
        return None
    data: dict[str, Any] = {
        "x": np.asarray(curve.x, dtype=np.float64),
        "log_relativity": np.asarray(curve.log_relativity, dtype=np.float64),
        "relativity": np.asarray(curve.relativity, dtype=np.float64),
    }
    if curve.se_log_relativity is not None:
        data["se_log_relativity"] = np.asarray(curve.se_log_relativity, dtype=np.float64)
    if curve.ci_lower is not None:
        data["ci_lower"] = np.asarray(curve.ci_lower, dtype=np.float64)
        data["ci_upper"] = np.asarray(curve.ci_upper, dtype=np.float64)
    return pd.DataFrame(data)


def _main_effect_density_dataframe(
    ti: TermInference,
    X: pd.DataFrame | None,
    sample_weight: NDArray | None,
) -> pd.DataFrame | None:
    if X is None or sample_weight is None or ti.name not in X.columns:
        return None

    if ti.kind in ("spline", "polynomial"):
        x_grid = np.asarray(ti.x, dtype=np.float64)
        x_vals = X[ti.name].to_numpy(dtype=np.float64)
        density = _exposure_kde(x_vals, sample_weight, x_grid)
        return pd.DataFrame({"x": x_grid, "density": density})

    if ti.kind == "numeric":
        x_vals = X[ti.name].to_numpy(dtype=np.float64)
        x_grid = np.linspace(float(x_vals.min()), float(x_vals.max()), 200)
        density = _exposure_kde(x_vals, sample_weight, x_grid)
        return pd.DataFrame({"x": x_grid, "density": density})

    levels = list(ti.levels)
    exp = (
        pd.DataFrame({"level": X[ti.name].astype(str), "sample_weight": sample_weight})
        .groupby("level", sort=False)["sample_weight"]
        .sum()
    )
    weights = np.array([float(exp.get(level, 0.0)) for level in levels], dtype=np.float64)
    peak = float(weights.max()) if weights.size else 0.0
    if peak > 0:
        weights = weights / peak
    return pd.DataFrame({"level": levels, "relative_density": weights})


def _main_effect_knots_dataframe(ti: TermInference) -> pd.DataFrame | None:
    if ti.spline is None or ti.spline.interior_knots.size == 0 or ti.x is None:
        return None

    x_grid = np.asarray(ti.x, dtype=np.float64)
    knots = np.asarray(ti.spline.interior_knots, dtype=np.float64)
    return pd.DataFrame(
        {
            "x": knots,
            "relativity": np.interp(knots, x_grid, np.asarray(ti.relativity, dtype=np.float64)),
            "log_relativity": np.interp(
                knots, x_grid, np.asarray(ti.log_relativity, dtype=np.float64)
            ),
        }
    )


def _main_effect_basis_dataframe(model, ti: TermInference) -> pd.DataFrame | None:
    if ti.kind != "spline" or ti.x is None:
        return None

    spec = model._specs.get(ti.name)
    if spec is None or not hasattr(spec, "_basis_matrix"):
        return None

    x_grid = np.asarray(ti.x, dtype=np.float64)
    basis = spec._basis_matrix(x_grid).toarray()
    if basis.shape[1] == 0:
        return None

    from superglm.model.report_ops import feature_groups

    groups = feature_groups(model, ti.name)
    beta_combined = np.concatenate([model.result.beta[g.sl] for g in groups])
    R_inv = getattr(spec, "_R_inv", None)
    beta_orig = R_inv @ beta_combined if R_inv is not None else beta_combined
    if len(beta_orig) != basis.shape[1]:
        return None

    frames = []
    for idx in range(basis.shape[1]):
        basis_values = basis[:, idx]
        coef = float(beta_orig[idx])
        frames.append(
            pd.DataFrame(
                {
                    "x": x_grid,
                    "basis_index": idx + 1,
                    "basis_name": f"Basis {idx + 1}",
                    "basis_value": basis_values,
                    "coefficient": coef,
                    "contribution": basis_values * coef,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _interaction_plot_kind(raw: dict[str, Any]) -> str:
    if "per_level" in raw and "x" in raw:
        return "varying_coefficient"
    if "pairs" in raw:
        return "categorical_heatmap"
    if "relativities_per_unit" in raw:
        return "numeric_categorical"
    if "relativity_per_unit_unit" in raw:
        return "numeric_numeric"
    if "x1" in raw and "x2" in raw:
        return "surface"
    return "unknown"


def _varying_coefficient_dataframe(
    raw: dict[str, Any], parent_names: tuple[str, str]
) -> pd.DataFrame:
    x = np.asarray(raw["x"], dtype=np.float64)
    frames = []
    base = raw.get("base_level", "")
    if base:
        frames.append(
            pd.DataFrame(
                {
                    parent_names[0]: x,
                    parent_names[1]: base,
                    "log_relativity": np.zeros_like(x),
                    "relativity": np.ones_like(x),
                    "is_base_level": True,
                }
            )
        )
    for level in raw["levels"]:
        level_data = raw["per_level"][level]
        frame = pd.DataFrame(
            {
                parent_names[0]: x,
                parent_names[1]: level,
                "log_relativity": np.asarray(level_data["log_relativity"], dtype=np.float64),
                "relativity": np.asarray(level_data["relativity"], dtype=np.float64),
                "is_base_level": False,
            }
        )
        if "se_log_relativity" in level_data:
            frame["se_log_relativity"] = np.asarray(
                level_data["se_log_relativity"], dtype=np.float64
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _categorical_heatmap_dataframe(
    raw: dict[str, Any], parent_names: tuple[str, str]
) -> pd.DataFrame:
    levels1 = raw.get("levels1") or list(dict.fromkeys(p[0] for p in raw["pairs"]))
    levels2 = raw.get("levels2") or list(dict.fromkeys(p[1] for p in raw["pairs"]))
    log_rels = raw["log_relativities"]
    rels = raw["relativities"]

    rows = []
    for level1 in levels1:
        for level2 in levels2:
            key = f"{level1}:{level2}"
            rows.append(
                {
                    parent_names[0]: level1,
                    parent_names[1]: level2,
                    "log_relativity": float(log_rels.get(key, 0.0)),
                    "relativity": float(rels.get(key, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def _numeric_categorical_dataframe(
    raw: dict[str, Any], parent_names: tuple[str, str]
) -> pd.DataFrame:
    base = raw.get("base_level", "")
    levels = [base] + list(raw["levels"]) if base else list(raw["levels"])
    rows = []
    for level in levels:
        if base and level == base:
            rel = 1.0
            log_rel = 0.0
            is_base = True
        else:
            rel = float(raw["relativities_per_unit"][level])
            log_rel = float(raw["log_relativities_per_unit"][level])
            is_base = False
        rows.append(
            {
                parent_names[1]: level,
                "relativity_per_unit": rel,
                "log_relativity_per_unit": log_rel,
                "is_base_level": is_base,
            }
        )
    return pd.DataFrame(rows)


def _numeric_interaction_dataframe(
    raw: dict[str, Any], parent_names: tuple[str, str]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                parent_names[0]: "per_unit",
                parent_names[1]: "per_unit",
                "coef": float(raw["coef"]),
                "relativity_per_unit_unit": float(raw["relativity_per_unit_unit"]),
            }
        ]
    )


def _surface_density_dataframe(
    X: pd.DataFrame,
    sample_weight: NDArray,
    parent_names: tuple[str, str],
    x1: NDArray,
    x2: NDArray,
) -> pd.DataFrame | None:
    p0, p1 = parent_names
    if (
        p0 not in X.columns
        or p1 not in X.columns
        or not pd.api.types.is_numeric_dtype(X[p0])
        or not pd.api.types.is_numeric_dtype(X[p1])
    ):
        return None

    d1 = np.asarray(X[p0], dtype=np.float64)
    d2 = np.asarray(X[p1], dtype=np.float64)
    density = _kde_2d(d1, d2, sample_weight, x1, x2)
    mass_field = _highest_density_mass_field(density)
    X1, X2 = np.meshgrid(x1, x2)
    return pd.DataFrame(
        {
            p0: X1.ravel(),
            p1: X2.ravel(),
            "density": density.ravel(),
            "hdr_mass": mass_field.ravel(),
        }
    )
