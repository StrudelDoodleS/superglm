"""Internal model-facing term helpers used by explain/diagnostic surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from superglm.inference._term_covariance import feature_se_from_cov
from superglm.inference._term_helpers import _VALID_CENTERING
from superglm.inference._term_types import _safe_exp

if TYPE_CHECKING:
    from superglm.solvers.pirls import PIRLSResult
    from superglm.types import GroupSlice


def relativities(
    feature_order: list[str],
    interaction_order: list[str],
    specs: dict[str, Any],
    interaction_specs: dict[str, Any],
    groups: list[GroupSlice],
    result: PIRLSResult,
    *,
    with_se: bool = False,
    covariance_fn=None,
    centering: str = "native",
) -> dict[str, pd.DataFrame]:
    """Extract plot-ready relativity DataFrames for all features."""
    if centering not in _VALID_CENTERING:
        raise ValueError(f"centering must be one of {_VALID_CENTERING}, got {centering!r}")
    if with_se:
        Cov_active, active_groups = covariance_fn()

    def _feature_groups(name: str) -> list[GroupSlice]:
        return [g for g in groups if g.feature_name == name]

    def _reconstruct(name: str) -> dict[str, Any]:
        fgroups = _feature_groups(name)
        beta_combined = np.concatenate([result.beta[g.sl] for g in fgroups])
        if name in specs:
            return cast(dict[str, Any], specs[name].reconstruct(beta_combined))
        if name in interaction_specs:
            return cast(dict[str, Any], interaction_specs[name].reconstruct(beta_combined))
        raise KeyError(f"Feature not found: {name}")

    from superglm.features.ordered_categorical import OrderedCategorical

    def _center_df(df: pd.DataFrame) -> pd.DataFrame:
        if centering != "mean" or "log_relativity" not in df.columns:
            return df
        log_rel = df["log_relativity"].values.copy()
        shift = float(np.mean(log_rel))
        df = df.copy()
        df["log_relativity"] = log_rel - shift
        df["relativity"] = _safe_exp(df["log_relativity"].values)
        return df

    out: dict[str, pd.DataFrame] = {}
    for name in feature_order:
        raw = _reconstruct(name)
        spec_cur = specs.get(name)

        if isinstance(spec_cur, OrderedCategorical) and spec_cur.basis == "spline":
            levels = raw["levels"]
            df = pd.DataFrame(
                {
                    "level": levels,
                    "relativity": [raw["level_relativities"][lv] for lv in levels],
                    "log_relativity": [raw["level_log_relativities"][lv] for lv in levels],
                }
            )
            if with_se:
                df["se_log_relativity"] = feature_se_from_cov(
                    name,
                    Cov_active,
                    active_groups,
                    result,
                    groups,
                    specs,
                    interaction_specs,
                )
            out[name] = _center_df(df)
            continue

        if "x" in raw:
            df = pd.DataFrame(
                {
                    "x": raw["x"],
                    "relativity": raw["relativity"],
                    "log_relativity": raw["log_relativity"],
                }
            )
            if with_se:
                df["se_log_relativity"] = feature_se_from_cov(
                    name,
                    Cov_active,
                    active_groups,
                    result,
                    groups,
                    specs,
                    interaction_specs,
                    n_points=len(raw["x"]),
                )
            out[name] = _center_df(df)
        elif "levels" in raw:
            levels = raw["levels"]
            rels = raw["relativities"]
            log_rels = raw["log_relativities"]
            df = pd.DataFrame(
                {
                    "level": levels,
                    "relativity": [rels[lv] for lv in levels],
                    "log_relativity": [log_rels[lv] for lv in levels],
                }
            )
            if with_se:
                df["se_log_relativity"] = feature_se_from_cov(
                    name,
                    Cov_active,
                    active_groups,
                    result,
                    groups,
                    specs,
                    interaction_specs,
                )
            out[name] = _center_df(df)
        elif "relativity_per_unit" in raw:
            rel = raw["relativity_per_unit"]
            df = pd.DataFrame(
                {
                    "label": ["per_unit"],
                    "relativity": [rel],
                    "log_relativity": [np.log(rel)],
                }
            )
            if with_se:
                df["se_log_relativity"] = feature_se_from_cov(
                    name,
                    Cov_active,
                    active_groups,
                    result,
                    groups,
                    specs,
                    interaction_specs,
                )
            out[name] = df

    for iname in interaction_order:
        raw = _reconstruct(iname)

        if "per_level" in raw and "x" in raw:
            for level in raw["levels"]:
                level_data = raw["per_level"][level]
                key = f"{iname}[{level}]"
                out[key] = pd.DataFrame(
                    {
                        "x": raw["x"],
                        "relativity": level_data["relativity"],
                        "log_relativity": level_data["log_relativity"],
                    }
                )

        elif "pairs" in raw:
            pairs_labels = [f"{l1}:{l2}" for l1, l2 in raw["pairs"]]
            rels = raw["relativities"]
            log_rels = raw["log_relativities"]
            out[iname] = pd.DataFrame(
                {
                    "level": pairs_labels,
                    "relativity": [rels[k] for k in pairs_labels],
                    "log_relativity": [log_rels[k] for k in pairs_labels],
                }
            )

        elif "relativities_per_unit" in raw:
            levels = raw["levels"]
            rels = raw["relativities_per_unit"]
            log_rels = raw["log_relativities_per_unit"]
            out[iname] = pd.DataFrame(
                {
                    "level": levels,
                    "relativity_per_unit": [rels[lv] for lv in levels],
                    "log_relativity_per_unit": [log_rels[lv] for lv in levels],
                }
            )

        elif "relativity_per_unit_unit" in raw:
            out[iname] = pd.DataFrame(
                {
                    "label": ["per_unit_unit"],
                    "relativity": [raw["relativity_per_unit_unit"]],
                    "log_relativity": [raw["coef"]],
                }
            )

    return out


def drop1(
    model,
    X: pd.DataFrame,
    y: NDArray,
    sample_weight: NDArray | None = None,
    offset: NDArray | None = None,
    *,
    test: str = "Chisq",
) -> pd.DataFrame:
    """Drop-one deviance analysis for each feature."""
    from scipy.stats import chi2
    from scipy.stats import f as f_dist

    if model._result is None:
        raise RuntimeError("Model must be fitted before calling drop1().")

    dev_full = model._result.deviance
    edf_full = model._result.effective_df
    n = len(y) if not hasattr(y, "__len__") else len(y)
    phi = model._result.phi

    rows = []
    for name in model._feature_order:
        drop_set = {name}
        for iname in model._interaction_order:
            ispec = model._interaction_specs[iname]
            p1, p2 = ispec.parent_names
            if p1 == name or p2 == name:
                drop_set.add(iname)

        remaining = [f for f in model._feature_order if f not in drop_set]

        if not remaining:
            from superglm.distributions import Binomial, Gaussian, clip_mu
            from superglm.links import stabilize_eta

            y_arr = np.asarray(y, dtype=np.float64)
            w = (
                np.ones(n, dtype=np.float64)
                if sample_weight is None
                else np.asarray(sample_weight, dtype=np.float64)
            )
            y_mean = float(np.average(y_arr, weights=w))
            if isinstance(model._distribution, Binomial):
                y_mean = np.clip(y_mean, 1e-3, 1 - 1e-3)
            elif isinstance(model._distribution, Gaussian):
                y_mean = float(y_mean)
            else:
                y_mean = max(y_mean, 1e-10)

            if offset is not None:
                offset_arr = np.asarray(offset, dtype=np.float64)
                b0 = float(model._link.link(np.atleast_1d(y_mean))[0]) - np.average(
                    offset_arr, weights=w
                )
                eta0 = stabilize_eta(b0 + offset_arr, model._link)
                null_mu = clip_mu(model._link.inverse(eta0), model._distribution)
            else:
                null_mu = np.full(n, y_mean)
            dev_reduced = float(np.sum(w * model._distribution.deviance_unit(y_arr, null_mu)))
            edf_reduced = 1.0
        else:
            reduced = model._clone_without_features(drop_set)
            reduced.fit(X, y, offset=offset)
            dev_reduced = reduced.result.deviance
            edf_reduced = reduced.result.effective_df
        delta_dev = dev_reduced - dev_full
        delta_df = max(edf_full - edf_reduced, 1e-4)

        if test == "F":
            stat = (delta_dev / delta_df) / phi
            resid_df = max(n - edf_full, 1.0)
            p_value = float(f_dist.sf(stat, delta_df, resid_df))
        else:
            stat = delta_dev
            p_value = float(chi2.sf(stat, delta_df))

        rows.append(
            {
                "feature": name,
                "deviance_full": dev_full,
                "deviance_reduced": dev_reduced,
                "delta_deviance": delta_dev,
                "delta_df": delta_df,
                "statistic": stat,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def refit_unpenalised(
    model,
    X: pd.DataFrame,
    y: NDArray,
    sample_weight: NDArray | None = None,
    offset: NDArray | None = None,
    *,
    keep_smoothing: bool = True,
):
    """Refit the model with only the active features and no selection penalty."""
    del sample_weight
    if model._result is None:
        raise RuntimeError("Model must be fitted before calling refit_unpenalised().")

    beta = model._result.beta

    inactive = set()
    for name in model._feature_order:
        fgroups = [g for g in model._groups if g.feature_name == name]
        if all(np.linalg.norm(beta[g.sl]) < 1e-12 for g in fgroups):
            inactive.add(name)

    for iname in model._interaction_order:
        ispec = model._interaction_specs[iname]
        p1, p2 = ispec.parent_names
        if p1 in inactive or p2 in inactive:
            inactive.add(iname)

    lam2: Any
    if not keep_smoothing:
        lam2 = 0.0
    else:
        lam2 = ...

    new_model = model._clone_without_features(inactive, lambda1=0.0, lambda2=lam2)
    new_model.fit(X, y, offset=offset)
    return new_model


__all__ = [
    "drop1",
    "refit_unpenalised",
    "relativities",
]
