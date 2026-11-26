"""Compare fit_mode="fit_ncv" against fit_mode="fit_reml" via cross_validate.

Runs 5-fold GroupKFold on the simulated grouped Gaussian book from
docs/scripts/ncv_grouped_curve.py and prints the fold-by-fold metrics and
summary totals a user reads from CrossValidationResult.

Run from the repository root:
    uv run python docs/scripts/ncv_cross_validate_check.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from superglm import Spline, SuperGLM, cross_validate


def true_curve(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * x) + 0.5 * x


def simulate_grouped_data(
    seed: int = 20261126,
    n_groups: int = 30,
    rows_per_group: int = 20,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Simulate rows grouped in local stretches of x with shared group shocks."""
    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    centers = np.linspace(0.03, 0.97, n_groups) + rng.uniform(-0.015, 0.015, n_groups)
    centers = np.clip(centers, 0.02, 0.98)
    x = np.clip(centers[groups] + rng.normal(0.0, 0.025, n), 0.0, 1.0)
    group_shock = rng.normal(0.0, 0.45, n_groups)
    y = true_curve(x) + group_shock[groups] + rng.normal(0.0, 0.12, n)
    return pd.DataFrame({"x": x}), y, groups


def main() -> None:
    X, y, groups = simulate_grouped_data()
    model = SuperGLM(family="gaussian", features={"x": Spline(kind="ps", k=25)})
    outer = GroupKFold(n_splits=5)

    with warnings.catch_warnings(record=True) as w_reml:
        warnings.simplefilter("always")
        reml_cv = cross_validate(
            model,
            X,
            y,
            cv=outer,
            groups=groups,
            fit_mode="fit_reml",
            return_estimators=True,
        )

    with warnings.catch_warnings(record=True) as w_ncv:
        warnings.simplefilter("always")
        ncv_cv = cross_validate(
            model,
            X,
            y,
            cv=outer,
            groups=groups,
            fit_mode="fit_ncv",
            return_estimators=True,
        )

    rf = reml_cv.fold_scores
    nf = ncv_cv.fold_scores
    print(
        f"CV_DATA n_rows={len(X)} n_groups={len(np.unique(groups))} "
        f"outer_splits={len(rf)} ncv_warm_start={ncv_cv.warm_start} "
        f"reml_warm_start={reml_cv.warm_start}"
    )

    ncv_better = 0
    for i in range(len(rf)):
        r_row = rf.iloc[i]
        n_row = nf.iloc[i]
        r_dev = float(r_row["deviance"])
        n_dev = float(n_row["deviance"])
        delta = n_dev - r_dev
        if n_dev < r_dev:
            ncv_better += 1
        n_inner_groups = int(ncv_cv.estimators[i].ncv_diagnostics()["neighbourhoods"]["count"])
        print(
            f"FOLD fold={int(r_row['fold'])} n_train={int(r_row['n_train'])} "
            f"n_test={int(r_row['n_test'])} inner_groups={n_inner_groups} "
            f"reml_dev={r_dev:.4f} reml_edf={float(r_row['effective_df']):.2f} "
            f"reml_iter={int(r_row['n_reml_iter'])} reml_conv={bool(r_row['converged'])} "
            f"reml_warm={bool(r_row['warm_started'])} "
            f"ncv_dev={n_dev:.4f} ncv_edf={float(n_row['effective_df']):.2f} "
            f"ncv_iter={int(n_row['n_ncv_iter'])} ncv_conv={bool(n_row['converged'])} "
            f"ncv_warm={bool(n_row['warm_started'])} "
            f"delta_dev={delta:+.4f}"
        )

    r_mean = float(reml_cv.mean_scores["deviance"])
    n_mean = float(ncv_cv.mean_scores["deviance"])
    r_pool = float(reml_cv.pooled_scores["deviance"])
    n_pool = float(ncv_cv.pooled_scores["deviance"])
    r_std = float(reml_cv.std_scores["deviance"])
    n_std = float(ncv_cv.std_scores["deviance"])

    print(
        f"SUMMARY_REML mean_dev={r_mean:.4f} pooled_dev={r_pool:.4f} "
        f"std_dev={r_std:.4f} warnings={len(w_reml)}"
    )
    print(
        f"SUMMARY_NCV mean_dev={n_mean:.4f} pooled_dev={n_pool:.4f} "
        f"std_dev={n_std:.4f} warnings={len(w_ncv)}"
    )
    print(
        f"SUMMARY_COMPARISON delta_mean={n_mean - r_mean:+.4f} "
        f"delta_pooled={n_pool - r_pool:+.4f} "
        f"pooled_reduction_pct={(1.0 - n_pool / r_pool) * 100.0:.1f} "
        f"ncv_better_folds={ncv_better}/{len(rf)}"
    )
    for label, caught in (("REML", w_reml), ("NCV", w_ncv)):
        for item in caught:
            print(f"WARNING mode={label} category={item.category.__name__} message={item.message}")


if __name__ == "__main__":
    main()
