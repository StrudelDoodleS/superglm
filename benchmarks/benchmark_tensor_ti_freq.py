"""Discrete MTPL frequency benchmark for a tensor interaction stress case.

Uses the same freMTPL2freq data preparation and 80/20 split as the other MTPL
frequency benchmarks, but compares:

    s(DrivAge) + s(VehAge) + s(BonusMalus) + Area

against:

    s(DrivAge) + s(VehAge) + s(BonusMalus) + ti(DrivAge, BonusMalus) + Area

The immediate goal is to provide a tracked reproduction of the current
``discrete=True fit_reml`` tensor-interaction failure mode so later refactors
have a fixed target.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.validation import lorenz_curve

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "freMTPL2freq.parquet"
if not DATA_PATH.exists() and ROOT.parent.name == ".worktrees":
    DATA_PATH = ROOT.parent.parent / "data" / "freMTPL2freq.parquet"
OUT_DIR = ROOT / "benchmarks" / "results"
OUT_JSON = OUT_DIR / "tensor_ti_freq.json"
OUT_TRAIN_CSV = OUT_DIR / "tensor_ti_freq_train.csv"
OUT_TEST_CSV = OUT_DIR / "tensor_ti_freq_test.csv"
OUT_FAIRNESS_JSON = OUT_DIR / "tensor_ti_freq_fairness.json"
OUT_VALIDATION_JSON = OUT_DIR / "tensor_ti_freq_runtime_validation.json"
OUT_ATTRIBUTION_JSON = OUT_DIR / "tensor_ti_freq_attribution.json"
TIMEOUT_S = 120.0
CASE_TIMEOUT_S = 180.0
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
PROFILE_TIMING_KEYS = (
    "reml_rebuild_dm_s",
    "reml_map_beta_s",
    "reml_penalty_context_s",
    "reml_tensor_summary_s",
    "fit_prime_caches_s",
    "fit_runtime_canonicalize_s",
    "fit_release_state_s",
    "irls_working_s",
    "irls_gram_s",
    "irls_solve_s",
    "irls_eta_s",
    "irls_deviance_s",
    "irls_deviance_eval_s",
    "irls_total_s",
    "irls_calls",
    "irls_iters",
    "block_diag_tensor_s",
    "block_diag_discrete_ssp_s",
    "block_diag_other_s",
    "block_cross_tensor_own_margin_s",
    "block_cross_tensor_main_s",
    "block_cross_tensor_tensor_s",
    "block_cross_tensor_spline_cat_s",
    "block_cross_spline_cat_spline_cat_s",
    "block_cross_disc_disc_s",
    "block_cross_disc_other_s",
    "block_cross_cat_cat_s",
    "block_cross_fallback_s",
    "block_hist2d_s",
    "block_tabmat_s",
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    interactions: tuple[tuple[str, str], ...] = ()

    @property
    def with_ti(self) -> bool:
        return ("DrivAge", "BonusMalus") in self.interactions


@dataclass(frozen=True)
class FitControls:
    name: str
    pirls_tol: float | None = None
    reml_tol: float = 1e-6
    max_reml_iter: int = 30
    max_pirls_iter: int | None = None
    interaction_mode: str = "full"
    runtime_validation: str = "auto"
    direct_solve: str = "auto"
    discrete: bool = True
    n_bins: int = 256
    discrete_strategy: str = "fixed_bins"

    def fit_kwargs(self) -> dict:
        kwargs = {
            "max_reml_iter": self.max_reml_iter,
            "reml_tol": self.reml_tol,
            "interaction_mode": self.interaction_mode,
            "runtime_validation": self.runtime_validation,
        }
        if self.pirls_tol is not None:
            kwargs["pirls_tol"] = self.pirls_tol
        if self.max_pirls_iter is not None:
            kwargs["max_pirls_iter"] = self.max_pirls_iter
        return kwargs

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "tol": 1e-6,
            "pirls_tol": 1e-6 if self.pirls_tol is None else self.pirls_tol,
            "pirls_tol_source": "model_default" if self.pirls_tol is None else "fit_reml_kwarg",
            "reml_tol": self.reml_tol,
            "max_iter": 100,
            "max_pirls_iter": 100 if self.max_pirls_iter is None else self.max_pirls_iter,
            "max_pirls_iter_source": "model_default"
            if self.max_pirls_iter is None
            else "fit_reml_kwarg",
            "max_reml_iter": self.max_reml_iter,
            "interaction_mode": self.interaction_mode,
            "runtime_validation": self.runtime_validation,
            "direct_solve": self.direct_solve,
            "discrete": self.discrete,
            "n_bins": self.n_bins,
            "discrete_strategy": self.discrete_strategy,
        }


def build_superglm_cases() -> tuple[BenchmarkCase, ...]:
    """Cases used to track one- and multi-interaction scaling.

    The first two cases intentionally preserve the original benchmark ordering:
    downstream PR summaries compare ``baseline_discrete`` against
    ``baseline_plus_ti_discrete``.
    """
    return (
        BenchmarkCase("baseline_discrete"),
        BenchmarkCase("baseline_plus_ti_discrete", (("DrivAge", "BonusMalus"),)),
        BenchmarkCase(
            "baseline_plus_2_tensors_discrete",
            (("DrivAge", "BonusMalus"), ("DrivAge", "VehAge")),
        ),
        BenchmarkCase(
            "baseline_plus_3_tensors_discrete",
            (("DrivAge", "BonusMalus"), ("DrivAge", "VehAge"), ("VehAge", "BonusMalus")),
        ),
        BenchmarkCase("baseline_plus_spline_cat_discrete", (("DrivAge", "Area"),)),
        BenchmarkCase(
            "baseline_plus_2_spline_cat_discrete",
            (("DrivAge", "Area"), ("BonusMalus", "Area")),
        ),
        BenchmarkCase(
            "baseline_plus_mixed_tensor_spline_cat_discrete",
            (("DrivAge", "BonusMalus"), ("VehAge", "Area")),
        ),
    )


def build_fairness_cases() -> tuple[BenchmarkCase, ...]:
    return build_superglm_cases()[:2]


def build_attribution_cases() -> tuple[BenchmarkCase, ...]:
    cases = build_superglm_cases()
    return (cases[0], cases[1], cases[4], cases[6])


def build_superglm_control_profiles() -> tuple[FitControls, ...]:
    return (
        FitControls("S0_current_default"),
        FitControls(
            "S1_strict",
            pirls_tol=1e-7,
            reml_tol=1e-7,
            max_reml_iter=20,
            runtime_validation="full",
        ),
        FitControls("S2_mgcv_ish", pirls_tol=1e-7, reml_tol=1e-6, max_reml_iter=20),
        FitControls("S3_practical", pirls_tol=1e-6, reml_tol=1e-6, max_reml_iter=20),
        FitControls(
            "S4_relaxed_candidate",
            pirls_tol=1e-5,
            reml_tol=1e-5,
            max_reml_iter=5,
            interaction_mode="fast_candidate",
        ),
        FitControls(
            "S5_very_relaxed_candidate",
            pirls_tol=1e-4,
            reml_tol=1e-4,
            max_reml_iter=3,
            interaction_mode="fast_candidate",
        ),
    )


def build_runtime_validation_profiles() -> tuple[FitControls, ...]:
    return (
        FitControls("runtime_auto", runtime_validation="auto"),
        FitControls("runtime_full", runtime_validation="full"),
        FitControls("runtime_skip", runtime_validation="skip"),
    )


def thread_control_metadata() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in THREAD_ENV_KEYS}


def select_named_items(items: tuple, names: str | None) -> tuple:
    if names is None or not names.strip():
        return items
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    by_name = {item.name: item for item in items}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        available = ", ".join(by_name)
        raise ValueError(f"Unknown names {missing}; available: {available}")
    return tuple(by_name[name] for name in wanted)


def _safe_delta(case: dict, baseline: dict, key: str) -> float | int | None:
    if case.get(key) is None or baseline.get(key) is None:
        return None
    value = case.get(key, 0.0) - baseline.get(key, 0.0)
    if isinstance(case.get(key), int) and isinstance(baseline.get(key), int):
        return int(value)
    return float(value)


def build_case_deltas(rows: list[dict]) -> dict:
    baseline = rows[0]
    one_tensor = next(
        (row for row in rows if row.get("model") == "baseline_plus_ti_discrete"),
        rows[1] if len(rows) > 1 else baseline,
    )
    legacy_keys = (
        "fit_s",
        "predict_test_median_s",
        "gini_model",
        "gini_ratio",
        "effective_df",
        "deviance",
        "reml_n_linesearch_fits",
        "reml_linesearch_s",
    )
    deltas = {key: _safe_delta(one_tensor, baseline, key) for key in legacy_keys}
    deltas.update({key: _safe_delta(one_tensor, baseline, key) for key in PROFILE_TIMING_KEYS})
    deltas["by_case"] = {
        str(row["model"]): {
            key: _safe_delta(row, baseline, key)
            for key in (
                "fit_s",
                "predict_test_median_s",
                "gini_model",
                "gini_ratio",
                "effective_df",
                "deviance",
                "coef_count",
                "n_groups",
                "n_smoothing_params",
                "n_reml_iter",
                "irls_calls",
                "irls_iters",
                "irls_gram_s",
                "irls_solve_s",
                "irls_eta_s",
                "reml_pirls_s",
                "reml_hessian_newton_s",
            )
        }
        for row in rows[1:]
    }
    return deltas


def _group_size(group) -> int:
    size = getattr(group, "size", None)
    if size is not None:
        return int(size)
    return int(getattr(group, "end") - getattr(group, "start"))


def _lambda_keys_for_group(
    group_name: str, feature_name: str, lambdas: dict[str, float]
) -> list[str]:
    return sorted(
        key
        for key in lambdas
        if key == group_name
        or key.startswith(f"{group_name}:")
        or key == feature_name
        or key.startswith(f"{feature_name}:")
    )


def summarize_group_attribution(
    groups,
    group_edf: dict[str, float] | None,
    lambdas: dict[str, float] | None,
) -> dict:
    """Summarise EDF/lambda attribution by solver group and feature term."""
    group_edf = group_edf or {}
    lambdas = lambdas or {}
    by_group: list[dict] = []
    feature_acc: dict[str, dict] = {}

    for group in groups:
        group_name = str(getattr(group, "name"))
        feature_name = str(getattr(group, "feature_name", group_name))
        edf = group_edf.get(group_name)
        coef_count = _group_size(group)
        lambda_keys = _lambda_keys_for_group(group_name, feature_name, lambdas)
        row = {
            "group_name": group_name,
            "feature_name": feature_name,
            "subgroup_type": getattr(group, "subgroup_type", None),
            "coef_count": coef_count,
            "edf": None if edf is None else float(edf),
            "lambda_keys": lambda_keys,
        }
        by_group.append(row)

        feature_row = feature_acc.setdefault(
            feature_name,
            {
                "feature_name": feature_name,
                "n_groups": 0,
                "coef_count": 0,
                "edf": 0.0,
                "group_names": [],
                "subgroup_types": set(),
                "lambda_keys": set(),
            },
        )
        feature_row["n_groups"] += 1
        feature_row["coef_count"] += coef_count
        feature_row["group_names"].append(group_name)
        if row["subgroup_type"] is not None:
            feature_row["subgroup_types"].add(row["subgroup_type"])
        feature_row["lambda_keys"].update(lambda_keys)
        if edf is not None:
            feature_row["edf"] += float(edf)

    by_feature = []
    for feature_row in feature_acc.values():
        by_feature.append(
            {
                "feature_name": feature_row["feature_name"],
                "n_groups": int(feature_row["n_groups"]),
                "coef_count": int(feature_row["coef_count"]),
                "edf": float(feature_row["edf"]),
                "group_names": list(feature_row["group_names"]),
                "subgroup_types": sorted(feature_row["subgroup_types"]),
                "lambda_keys": sorted(feature_row["lambda_keys"]),
            }
        )

    return {
        "by_group": by_group,
        "by_feature": by_feature,
        "total_group_edf": float(sum(row["edf"] or 0.0 for row in by_group)),
    }


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < x.size:
        stop = start + 1
        while stop < x.size and sorted_x[stop] == sorted_x[start]:
            stop += 1
        avg_rank = 0.5 * (start + stop - 1)
        ranks[order[start:stop]] = avg_rank
        start = stop
    return ranks


def summarize_eta_delta(eta: np.ndarray, reference_eta: np.ndarray) -> dict:
    """Return eta delta and rank-agreement diagnostics against a reference."""
    eta = np.asarray(eta, dtype=np.float64).ravel()
    reference_eta = np.asarray(reference_eta, dtype=np.float64).ravel()
    if eta.shape != reference_eta.shape:
        raise ValueError("eta and reference_eta must have the same shape")
    delta = eta - reference_eta
    if eta.size <= 1:
        rank_corr = 1.0
    else:
        eta_rank = _average_ranks(eta)
        ref_rank = _average_ranks(reference_eta)
        eta_rank = eta_rank - float(np.mean(eta_rank))
        ref_rank = ref_rank - float(np.mean(ref_rank))
        denom = float(np.linalg.norm(eta_rank) * np.linalg.norm(ref_rank))
        rank_corr = 1.0 if denom == 0.0 else float(np.dot(eta_rank, ref_rank) / denom)
    return {
        "max_abs_eta_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs_eta_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "rank_corr_eta": rank_corr,
    }


def load_freq() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(DATA_PATH)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(lower=0.01)
    df["DrivAge"] = df["DrivAge"].clip(18, 90)
    df["VehAge"] = df["VehAge"].clip(0, 20)
    df["BonusMalus"] = df["BonusMalus"].clip(50, 150)
    y_freq = (df["ClaimNb"] / df["Exposure"]).to_numpy(dtype=float)
    exposure = df["Exposure"].to_numpy(dtype=float)
    X = df[["DrivAge", "VehAge", "BonusMalus", "Area"]].copy()
    return X, y_freq, exposure


def split_data(
    X: pd.DataFrame, y: np.ndarray, w: np.ndarray, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(0.8 * len(idx))
    tr, te = idx[:n_train], idx[n_train:]
    return (
        X.iloc[tr].reset_index(drop=True),
        X.iloc[te].reset_index(drop=True),
        y[tr],
        y[te],
        w[tr],
        w[te],
    )


def build_features(discrete: bool) -> dict:
    return {
        "DrivAge": Spline(kind="cr", k=20, penalty="ssp", discrete=discrete),
        "VehAge": Spline(kind="cr", k=15, penalty="ssp", discrete=discrete),
        "BonusMalus": Spline(kind="cr", k=15, penalty="ssp", discrete=discrete),
        "Area": Categorical(base="most_exposed"),
    }


def _fit_case_result(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    *,
    interactions: tuple[tuple[str, str], ...],
    controls: FitControls | None = None,
    return_eta: bool = False,
    return_attribution: bool = False,
) -> dict:
    controls = controls or FitControls("S0_current_default")
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        spline_penalty=0.0,
        direct_solve=controls.direct_solve,
        discrete=controls.discrete,
        n_bins=controls.n_bins,
        features=build_features(discrete=controls.discrete),
        interactions=list(interactions) if interactions else None,
    )
    t0 = time.perf_counter()
    model.fit_reml(X_train, y_train, sample_weight=w_train, **controls.fit_kwargs())
    fit_s = time.perf_counter() - t0
    profile_timings = {key: float(model._reml_profile.get(key, 0.0)) for key in PROFILE_TIMING_KEYS}
    runtime_validate = bool(model._reml_profile.get("fit_runtime_canonicalize_validate", False))
    runtime_validate_reason = str(
        model._reml_profile.get("fit_runtime_canonicalize_validate_reason", "")
    )

    if fit_s > TIMEOUT_S:
        return {
            "model": name,
            "with_ti": ("DrivAge", "BonusMalus") in interactions,
            "interactions": [f"{a}:{b}" for a, b in interactions],
            "n_interactions": len(interactions),
            "control": controls.metadata(),
            "timed_out": True,
            "timeout_s": TIMEOUT_S,
            "fit_s": fit_s,
            "predict_test_median_s": None,
            "gini_model": None,
            "gini_ratio": None,
            "effective_df": None,
            "deviance": None,
            "lambdas": {},
            "n_reml_iter": int(model._reml_result.n_reml_iter),
            "converged": False,
            "reml_n_linesearch_fits": int(model._reml_profile.get("reml_n_linesearch_fits", 0)),
            "reml_linesearch_s": float(model._reml_profile.get("reml_linesearch_s", 0.0)),
            "reml_n_outer_iter": int(model._reml_profile.get("reml_n_outer_iter", 0)),
            "reml_pirls_s": float(model._reml_profile.get("reml_pirls_s", 0.0)),
            "fit_runtime_canonicalize_validate": runtime_validate,
            "fit_runtime_canonicalize_validate_reason": runtime_validate_reason,
            "coef_count": int(model._result.beta.size),
            "n_groups": len(model._groups),
            "n_smoothing_params": 0
            if model._reml_result is None
            else len(model._reml_result.lambdas),
            **profile_timings,
        }

    mu = model.predict(X_test)
    eta = model._predict_eta_exact(X_test) if return_eta else None
    lorenz = lorenz_curve(y_test, mu, exposure=w_test)

    _ = model.predict(X_test)
    predict_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = model.predict(X_test)
        predict_times.append(time.perf_counter() - t0)

    row = {
        "model": name,
        "with_ti": ("DrivAge", "BonusMalus") in interactions,
        "interactions": [f"{a}:{b}" for a, b in interactions],
        "n_interactions": len(interactions),
        "control": controls.metadata(),
        "fit_s": fit_s,
        "predict_test_median_s": float(np.median(predict_times)),
        "gini_model": float(lorenz.gini_model),
        "gini_ratio": float(lorenz.gini_ratio),
        "effective_df": float(model.result.effective_df),
        "deviance": float(model.result.deviance),
        "lambdas": {str(k): float(v) for k, v in model._reml_result.lambdas.items()},
        "coef_count": int(model._result.beta.size),
        "n_groups": len(model._groups),
        "n_smoothing_params": len(model._reml_result.lambdas),
        "n_reml_iter": int(model._reml_result.n_reml_iter),
        "converged": bool(model._reml_result.converged),
        "reml_n_linesearch_fits": int(model._reml_profile.get("reml_n_linesearch_fits", 0)),
        "reml_linesearch_s": float(model._reml_profile.get("reml_linesearch_s", 0.0)),
        "reml_n_outer_iter": int(model._reml_profile.get("reml_n_outer_iter", 0)),
        "reml_pirls_s": float(model._reml_profile.get("reml_pirls_s", 0.0)),
        "reml_objective_s": float(model._reml_profile.get("reml_objective_s", 0.0)),
        "reml_hessian_newton_s": float(model._reml_profile.get("reml_hessian_newton_s", 0.0)),
        "reml_n_analytical_iters": int(model._reml_profile.get("reml_n_analytical_iters", 0)),
        "fit_runtime_canonicalize_validate": runtime_validate,
        "fit_runtime_canonicalize_validate_reason": runtime_validate_reason,
        **profile_timings,
    }
    if return_eta:
        row["_eta_test"] = eta
    if return_attribution:
        row["group_attribution"] = summarize_group_attribution(
            model._groups,
            model._group_edf,
            model._reml_result.lambdas,
        )
    return row


def _fit_case_worker(
    queue: mp.Queue,
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    interactions: tuple[tuple[str, str], ...],
    controls: FitControls,
    return_eta: bool,
    return_attribution: bool,
) -> None:
    try:
        queue.put(
            _fit_case_result(
                name,
                X_train,
                X_test,
                y_train,
                y_test,
                w_train,
                w_test,
                interactions=interactions,
                controls=controls,
                return_eta=return_eta,
                return_attribution=return_attribution,
            )
        )
    except BaseException as exc:  # pragma: no cover - benchmark failure path
        queue.put(
            {
                "model": name,
                "with_ti": ("DrivAge", "BonusMalus") in interactions,
                "interactions": [f"{a}:{b}" for a, b in interactions],
                "n_interactions": len(interactions),
                "control": controls.metadata(),
                "error": repr(exc),
            }
        )


def fit_case(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    *,
    interactions: tuple[tuple[str, str], ...],
    controls: FitControls | None = None,
    return_eta: bool = False,
    return_attribution: bool = False,
    timeout_s: float = TIMEOUT_S,
) -> dict:
    controls = controls or FitControls("S0_current_default")
    if return_eta or return_attribution:
        result = _fit_case_result(
            name,
            X_train,
            X_test,
            y_train,
            y_test,
            w_train,
            w_test,
            interactions=interactions,
            controls=controls,
            return_eta=return_eta,
            return_attribution=return_attribution,
        )
        result.setdefault("timed_out", False)
        return result

    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_fit_case_worker,
        args=(
            queue,
            name,
            X_train,
            X_test,
            y_train,
            y_test,
            w_train,
            w_test,
            interactions,
            controls,
            return_eta,
            return_attribution,
        ),
    )
    proc.start()
    proc.join(CASE_TIMEOUT_S)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "model": name,
            "with_ti": ("DrivAge", "BonusMalus") in interactions,
            "interactions": [f"{a}:{b}" for a, b in interactions],
            "n_interactions": len(interactions),
            "control": controls.metadata(),
            "timed_out": True,
            "timeout_s": timeout_s,
            "fit_s": timeout_s,
            "predict_test_median_s": None,
            "gini_model": None,
            "gini_ratio": None,
            "effective_df": None,
            "deviance": None,
            "lambdas": {},
            "n_reml_iter": None,
            "converged": False,
            "reml_n_linesearch_fits": None,
            "reml_linesearch_s": None,
            "reml_n_outer_iter": None,
            "reml_pirls_s": None,
            "reml_objective_s": None,
            "reml_hessian_newton_s": None,
            "reml_n_analytical_iters": None,
            "coef_count": None,
            "n_groups": None,
            "n_smoothing_params": None,
        }

    result = queue.get()
    result.setdefault("timed_out", False)
    return result


def _prepare_split() -> tuple[
    pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    X, y, w = load_freq()
    X_train, X_test, y_train, y_test, w_train, w_test = split_data(X, y, w)

    export_train = X_train.copy()
    export_train["y_freq"] = y_train
    export_train["Exposure"] = w_train
    export_train.to_csv(OUT_TRAIN_CSV, index=False)

    export_test = X_test.copy()
    export_test["y_freq"] = y_test
    export_test["Exposure"] = w_test
    export_test.to_csv(OUT_TEST_CSV, index=False)
    return X_train, X_test, y_train, y_test, w_train, w_test


def _base_output(
    *,
    n_total: int,
    n_train: int,
    n_test: int,
    case_matrix: tuple[BenchmarkCase, ...],
) -> dict:
    return {
        "dataset": "freMTPL2freq",
        "n_total": int(n_total),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "target": "claim_rate",
        "weight": "exposure",
        "feature_set": ["DrivAge", "VehAge", "BonusMalus", "Area"],
        "interaction": "DrivAge:BonusMalus",
        "case_matrix": [
            {"name": case.name, "interactions": [f"{a}:{b}" for a, b in case.interactions]}
            for case in case_matrix
        ],
        "split_seed": 42,
        "timeout_s": TIMEOUT_S,
        "thread_controls": thread_control_metadata(),
    }


def _print_rows(rows: list[dict]) -> None:
    for row in rows:
        print(
            f"{row['model']:<48s} "
            f"fit={row['fit_s']:>7.2f}s  "
            f"predict={row['predict_test_median_s'] if row['predict_test_median_s'] is None else format(row['predict_test_median_s'], '.4f')}s  "
            f"gini={row['gini_model'] if row['gini_model'] is None else format(row['gini_model'], '.6f')}  "
            f"edf={row['effective_df'] if row['effective_df'] is None else format(row['effective_df'], '8.2f')}  "
            f"dev={row['deviance'] if row['deviance'] is None else format(row['deviance'], '.3f')}  "
            f"p={row['coef_count']}  "
            f"groups={row['n_groups']}  "
            f"sp={row['n_smoothing_params']}  "
            f"reml={row['n_reml_iter']}  "
            f"irls={row.get('irls_iters')}  "
            f"converged={row['converged']}  "
            f"timed_out={row['timed_out']}"
        )


def run_case_matrix() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, w_train, w_test = _prepare_split()
    controls = FitControls("S0_current_default")
    cases = build_superglm_cases()

    rows = []
    for case in cases:
        print(f"[SuperGLM matrix] fitting {case.name}", flush=True)
        row = fit_case(
            case.name,
            X_train,
            X_test,
            y_train,
            y_test,
            w_train,
            w_test,
            interactions=case.interactions,
            controls=controls,
        )
        print(f"[SuperGLM matrix] finished {case.name}: fit={row.get('fit_s')}s", flush=True)
        rows.append(row)

    out = _base_output(
        n_total=len(X_train) + len(X_test),
        n_train=len(X_train),
        n_test=len(X_test),
        case_matrix=cases,
    )
    out.update(
        {
            "suite": "case_matrix",
            "control": controls.metadata(),
            "results": rows,
            "deltas": build_case_deltas(rows),
        }
    )
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return out


def run_fairness_ladder(profile_names: str | None = None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, w_train, w_test = _prepare_split()
    cases = build_fairness_cases()
    controls = select_named_items(build_superglm_control_profiles(), profile_names)
    rows: list[dict] = []
    eta_by_key: dict[tuple[str, str], np.ndarray] = {}

    for control in controls:
        for case in cases:
            print(f"[SuperGLM fairness] fitting {control.name}/{case.name}", flush=True)
            row = fit_case(
                case.name,
                X_train,
                X_test,
                y_train,
                y_test,
                w_train,
                w_test,
                interactions=case.interactions,
                controls=control,
                return_eta=True,
            )
            eta = row.pop("_eta_test", None)
            if eta is not None:
                eta_by_key[(control.name, case.name)] = np.asarray(eta, dtype=np.float64)
            row["setting"] = control.name
            print(
                f"[SuperGLM fairness] finished {control.name}/{case.name}: fit={row.get('fit_s')}s",
                flush=True,
            )
            rows.append(row)

    for row in rows:
        strict_eta = eta_by_key.get(("S1_strict", row["model"]))
        eta = eta_by_key.get((row["setting"], row["model"]))
        if strict_eta is None or eta is None:
            row["max_abs_eta_delta_vs_strict"] = None
            row["mean_abs_eta_delta_vs_strict"] = None
            continue
        delta = np.abs(eta - strict_eta)
        row["max_abs_eta_delta_vs_strict"] = float(np.max(delta))
        row["mean_abs_eta_delta_vs_strict"] = float(np.mean(delta))

    out = _base_output(
        n_total=len(X_train) + len(X_test),
        n_train=len(X_train),
        n_test=len(X_test),
        case_matrix=cases,
    )
    out.update(
        {
            "suite": "fairness_ladder",
            "controls": [control.metadata() for control in controls],
            "results": rows,
            "deltas_by_setting": {
                control.name: build_case_deltas(
                    [row for row in rows if row.get("setting") == control.name]
                )
                for control in controls
            },
        }
    )
    OUT_FAIRNESS_JSON.write_text(json.dumps(out, indent=2))
    return out


def run_runtime_validation_ladder(profile_names: str | None = None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, w_train, w_test = _prepare_split()
    cases = build_fairness_cases()
    controls = select_named_items(build_runtime_validation_profiles(), profile_names)
    rows: list[dict] = []

    for control in controls:
        for case in cases:
            print(f"[SuperGLM runtime] fitting {control.name}/{case.name}", flush=True)
            row = fit_case(
                case.name,
                X_train,
                X_test,
                y_train,
                y_test,
                w_train,
                w_test,
                interactions=case.interactions,
                controls=control,
            )
            row["setting"] = control.name
            print(
                f"[SuperGLM runtime] finished {control.name}/{case.name}: fit={row.get('fit_s')}s",
                flush=True,
            )
            rows.append(row)

    out = _base_output(
        n_total=len(X_train) + len(X_test),
        n_train=len(X_train),
        n_test=len(X_test),
        case_matrix=cases,
    )
    out.update(
        {
            "suite": "runtime_validation_ladder",
            "controls": [control.metadata() for control in controls],
            "results": rows,
            "deltas_by_setting": {
                control.name: build_case_deltas(
                    [row for row in rows if row.get("setting") == control.name]
                )
                for control in controls
            },
        }
    )
    OUT_VALIDATION_JSON.write_text(json.dumps(out, indent=2))
    return out


def run_attribution_audit() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, w_train, w_test = _prepare_split()
    controls = FitControls("S0_current_default")
    cases = build_attribution_cases()
    rows: list[dict] = []
    eta_by_case: dict[str, np.ndarray] = {}

    for case in cases:
        print(f"[SuperGLM attribution] fitting {case.name}", flush=True)
        row = fit_case(
            case.name,
            X_train,
            X_test,
            y_train,
            y_test,
            w_train,
            w_test,
            interactions=case.interactions,
            controls=controls,
            return_eta=True,
            return_attribution=True,
        )
        eta = row.pop("_eta_test", None)
        if eta is not None:
            eta_by_case[case.name] = np.asarray(eta, dtype=np.float64)
        print(f"[SuperGLM attribution] finished {case.name}: fit={row.get('fit_s')}s", flush=True)
        rows.append(row)

    baseline_eta = eta_by_case.get("baseline_discrete")
    one_tensor_eta = eta_by_case.get("baseline_plus_ti_discrete")
    for row in rows:
        eta = eta_by_case.get(row["model"])
        row["eta_delta_vs_baseline"] = (
            None
            if baseline_eta is None or eta is None or row["model"] == "baseline_discrete"
            else summarize_eta_delta(eta, baseline_eta)
        )
        row["eta_delta_vs_one_tensor"] = (
            None
            if one_tensor_eta is None or eta is None or row["model"] == "baseline_plus_ti_discrete"
            else summarize_eta_delta(eta, one_tensor_eta)
        )

    out = _base_output(
        n_total=len(X_train) + len(X_test),
        n_train=len(X_train),
        n_test=len(X_test),
        case_matrix=cases,
    )
    out.update(
        {
            "suite": "attribution_audit",
            "control": controls.metadata(),
            "results": rows,
            "deltas": build_case_deltas(rows),
        }
    )
    OUT_ATTRIBUTION_JSON.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("matrix", "fairness", "runtime-validation", "attribution"),
        default="matrix",
        help="Benchmark suite to run.",
    )
    parser.add_argument(
        "--profiles",
        default=None,
        help="Comma-separated control/profile names for fairness or runtime-validation suites.",
    )
    args = parser.parse_args()

    if args.suite == "fairness":
        out = run_fairness_ladder(profile_names=args.profiles)
        out_path = OUT_FAIRNESS_JSON
    elif args.suite == "runtime-validation":
        out = run_runtime_validation_ladder(profile_names=args.profiles)
        out_path = OUT_VALIDATION_JSON
    elif args.suite == "attribution":
        out = run_attribution_audit()
        out_path = OUT_ATTRIBUTION_JSON
    else:
        out = run_case_matrix()
        out_path = OUT_JSON

    rows = out["results"]

    print("Discrete MTPL tensor benchmark")
    print("=" * 88)
    print(f"Suite: {out['suite']}")
    print(f"Train rows: {out['n_train']:,}  Test rows: {out['n_test']:,}")
    print(f"Thread controls: {out['thread_controls']}")
    print()
    _print_rows(rows)
    print()
    if "deltas" in out:
        print(
            "Delta vs baseline: "
            f"fit={out['deltas']['fit_s']:+.2f}s  "
            f"predict={out['deltas']['predict_test_median_s']}s  "
            f"gini={out['deltas']['gini_model']}  "
            f"gini_ratio={out['deltas']['gini_ratio']}  "
            f"edf={out['deltas']['effective_df']}  "
            f"dev={out['deltas']['deviance']}  "
            f"ls_fits={out['deltas']['reml_n_linesearch_fits']}  "
            f"ls_time={out['deltas']['reml_linesearch_s']}"
        )
    print()
    print(f"Saved JSON: {out_path}")
    print(f"Saved train CSV: {OUT_TRAIN_CSV}")
    print(f"Saved test CSV:  {OUT_TEST_CSV}")


if __name__ == "__main__":
    main()
