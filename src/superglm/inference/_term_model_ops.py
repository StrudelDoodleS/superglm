"""Internal model-facing term helpers used by explain/diagnostic surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from superglm._frame import FrameLike
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

    # The shift below is the mean of the same fitted coefficients the frame
    # reports, so a centered frame's errors are those of the contrast ``C b``
    # and have to be propagated through ``C``, not carried over.  They are
    # asked for at each SE call site rather than repaired here, because this
    # function only sees the shifted values and not the covariance.
    center_se = centering == "mean"

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
                    center=center_se,
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
                    center=center_se,
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
                    center=center_se,
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


def _requires_reml_term_names(model) -> list[str]:
    """Return configured terms that require variance-component fitting."""
    configured_terms = [(name, model._specs[name]) for name in model._feature_order] + [
        (name, model._interaction_specs[name]) for name in model._interaction_order
    ]
    return [name for name, spec in configured_terms if getattr(spec, "requires_reml", False)]


def _drop1_reduced_frame(model, X: FrameLike) -> FrameLike:
    """Select a reduced model's physical columns without changing label identity."""
    from superglm._frame import as_eager_frame

    frame = as_eager_frame(X)
    columns = list(model._feature_order)
    for name in model._interaction_order:
        columns.extend(model._interaction_specs[name].parent_names)
    selected = tuple(dict.fromkeys(columns))
    if not selected:
        # An intercept-only refit still needs the original row count.
        return X
    if frame.backend == "pandas":
        # A plain Python list makes pandas coerce a mixed label sequence such
        # as (None, "") to [nan, ""]. An object Index preserves the exact
        # arbitrary-hashable labels configured on the model.
        return cast(pd.DataFrame, frame.native).loc[:, pd.Index(selected, dtype=object)]
    return frame.select_native(selected)


def drop1(
    model,
    X: FrameLike,
    y: NDArray,
    sample_weight: NDArray | None = None,
    offset: NDArray | None = None,
    *,
    test: str = "Chisq",
) -> pd.DataFrame:
    """Drop-one deviance analysis for each feature.

    ``test="Chisq"`` compares the deviance change divided by the family's
    dispersion with a chi-square reference on the effective-d.f. difference.
    Estimated-scale families divide by the fitted ``phi``; known-scale
    families such as Poisson and Binomial divide by their unit scale, taken
    from the family rather than from the result because ``fit()`` pins
    ``phi`` to 1.0 while ``fit_path()`` publishes the solver's Pearson
    dispersion untouched. ``test="F"`` compares
    ``(delta_deviance / delta_df) / phi`` with an F reference whose residual
    d.f. follows the fitted family's sample-weight contract; an F reference
    presumes an estimated scale, so that path always divides by the fitted
    dispersion. For an estimated-scale fit with exactly zero dispersion, an
    unchanged reduced deviance is reported as statistic 0 and p-value 1; a
    nonzero deviance change is undefined and raises an explicit error.
    """
    from scipy.stats import chi2
    from scipy.stats import f as f_dist

    if model._result is None:
        raise RuntimeError("Model must be fitted before calling drop1().")
    if test not in {"Chisq", "F"}:
        raise ValueError(f"test must be 'Chisq' or 'F', got {test!r}")

    reml_only_terms = _requires_reml_term_names(model)
    if reml_only_terms:
        raise NotImplementedError(
            "drop1() does not support variance-component terms "
            f"{reml_only_terms!r}; boundary-aware REML comparison requires "
            "a dedicated model-comparison contract."
        )

    dev_full = model._result.deviance
    edf_full = model._result.effective_df
    n = len(y) if not hasattr(y, "__len__") else len(y)
    phi = model._result.phi
    # The chi-square LRT reference for a known-scale family is the family's
    # unit dispersion. Reading it off the result would make the test depend on
    # the fitting entry point: fit() pins phi to 1.0, but fit_path() publishes
    # the path solution's Pearson dispersion untouched.
    chisq_scale = 1.0 if getattr(model._distribution, "scale_known", True) else phi
    if sample_weight is None:
        diagnostic_weights = np.ones(n, dtype=np.float64)
    else:
        from superglm.distributions import Tweedie

        if isinstance(model._distribution, Tweedie):
            from superglm._utils import _validate_strict_prior_weights

            diagnostic_weights = _validate_strict_prior_weights(sample_weight, n)
        else:
            diagnostic_weights = np.asarray(sample_weight, dtype=np.float64)

    rows = []
    for name in model._feature_order:
        drop_set = {name}
        for iname in model._interaction_order:
            ispec = model._interaction_specs[iname]
            p1, p2 = ispec.parent_names
            if p1 == name or p2 == name:
                drop_set.add(iname)

        reduced = model._clone_without_features(drop_set)
        reduced_X = _drop1_reduced_frame(reduced, X)
        reduced.fit(reduced_X, y, sample_weight=sample_weight, offset=offset)
        dev_reduced = reduced.result.deviance
        edf_reduced = reduced.result.effective_df
        delta_dev = dev_reduced - dev_full
        delta_df = max(edf_full - edf_reduced, 1e-4)

        if not np.isfinite(phi) or phi < 0.0:
            raise ValueError(
                f"drop1 requires a finite nonnegative fitted dispersion; got phi={phi!r}"
            )
        if phi == 0.0:
            if delta_dev != 0.0:
                raise ValueError(
                    f"drop1 {test} statistic for feature {name!r} is undefined: "
                    "the fitted dispersion phi is zero but dropping the feature "
                    f"changes deviance by {delta_dev!r}"
                )
            stat = 0.0
            p_value = 1.0
        elif test == "F":
            from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom

            stat = (delta_dev / delta_df) / phi
            resid_df = pearson_residual_degrees_of_freedom(
                model._distribution,
                diagnostic_weights,
                edf_full,
            )
            p_value = float(f_dist.sf(stat, delta_df, resid_df))
        else:
            stat = delta_dev / chisq_scale
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
    X: FrameLike,
    y: NDArray,
    sample_weight: NDArray | None = None,
    offset: NDArray | None = None,
    *,
    keep_smoothing: bool = True,
):
    """Refit the model with only the active features and no selection penalty."""
    if model._result is None:
        raise RuntimeError("Model must be fitted before calling refit_unpenalised().")

    reml_only_terms = _requires_reml_term_names(model)
    if reml_only_terms:
        raise NotImplementedError(
            "refit_unpenalised() does not support variance-component terms "
            f"{reml_only_terms!r}; an ordinary unpenalised fit cannot preserve "
            "their REML variance-component contract."
        )

    from superglm.solvers.rank import selected_group_name_set

    selected_names = selected_group_name_set(model._result, model._groups)

    inactive = set()
    for name in model._feature_order:
        fgroups = [g for g in model._groups if g.feature_name == name]
        if all(group.name not in selected_names for group in fgroups):
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
    new_model.fit(
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
    )
    return new_model


__all__ = [
    "drop1",
    "refit_unpenalised",
    "relativities",
]
