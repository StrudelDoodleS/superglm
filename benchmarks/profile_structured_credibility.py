"""Profile exact and discrete structured credibility fits.

The harness keeps data generation and model construction outside timed fit
calls.  It records uninstrumented wall repetitions separately from the single
instrumented fit used for cProfile and tracemalloc diagnostics.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import platform
import statistics
import time
import tracemalloc
import warnings
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from superglm import FactorSmooth, Numeric, RandomEffect, Spline, SuperGLM
from superglm.profiling.harness import (
    SystemSampler,
    dump_json,
    summarize_system_samples,
    write_pstats_summary,
    write_system_samples_csv,
    write_tracemalloc_report,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "benchmarks" / "results" / "structured_credibility"


@dataclass(frozen=True)
class CaseConfig:
    """One reproducible structured-credibility benchmark case."""

    n: int
    levels: int
    family: str
    discrete: bool
    random_effects: int
    secondary_levels: int | None
    small_width: int
    weights: str
    seed: int
    structured_term: str = "random_effect"
    block_size: int = 1
    global_spline: bool = False
    factor_basis: str = "fs"

    @property
    def dominant_width(self) -> int:
        """Coefficient width of the profiled structured term."""
        coefficient_levels = (
            self.levels - 1
            if self.structured_term == "factor_smooth" and self.factor_basis == "sz"
            else self.levels
        )
        return coefficient_levels * self.block_size

    @property
    def slug(self) -> str:
        mode = "discrete" if self.discrete else "exact"
        term = (
            "re"
            if self.structured_term == "random_effect"
            else f"{self.factor_basis}_k{self.block_size}"
        )
        global_suffix = "_global" if self.global_spline else ""
        return (
            f"{self.family}_{mode}_n{self.n}_k{self.levels}"
            f"_q{self.small_width}_{term}{global_suffix}_re{self.random_effects}"
        )


@dataclass(frozen=True)
class PreparedCase:
    """Generated rows and fit controls shared across backend repetitions."""

    config: CaseConfig
    X: pd.DataFrame
    y: np.ndarray
    sample_weight: np.ndarray | None
    offset: np.ndarray | None
    dominant_codes: np.ndarray
    secondary_level_count: int


CORE_MATRIX = (
    CaseConfig(2_000, 100, "gaussian", False, 1, None, 4, "unit", 7301),
    CaseConfig(6_000, 300, "poisson", False, 1, None, 4, "nonuniform", 7302),
    CaseConfig(20_000, 1_000, "poisson", False, 1, None, 6, "nonuniform", 7303),
    CaseConfig(20_000, 1_000, "poisson", True, 1, None, 6, "nonuniform", 7304),
    CaseConfig(30_000, 3_000, "gamma", False, 1, None, 4, "nonuniform", 7305),
    CaseConfig(50_000, 10_000, "poisson", True, 1, None, 4, "nonuniform", 7306),
    CaseConfig(20_000, 1_000, "poisson", False, 2, 80, 4, "nonuniform", 7307),
)

FACTOR_SMOOTH_MATRIX = (
    CaseConfig(
        3_000,
        30,
        "gaussian",
        False,
        0,
        None,
        4,
        "unit",
        7401,
        "factor_smooth",
        5,
        False,
    ),
    CaseConfig(
        6_000,
        50,
        "poisson",
        False,
        0,
        None,
        4,
        "nonuniform",
        7402,
        "factor_smooth",
        5,
        True,
    ),
    CaseConfig(
        10_000,
        100,
        "poisson",
        True,
        0,
        None,
        4,
        "nonuniform",
        7403,
        "factor_smooth",
        5,
        False,
    ),
    CaseConfig(
        8_000,
        40,
        "gamma",
        False,
        0,
        None,
        4,
        "nonuniform",
        7404,
        "factor_smooth",
        10,
        False,
    ),
    CaseConfig(
        20_000,
        300,
        "poisson",
        True,
        1,
        25,
        4,
        "nonuniform",
        7405,
        "factor_smooth",
        10,
        True,
    ),
    CaseConfig(
        30_000,
        600,
        "poisson",
        False,
        0,
        None,
        5,
        "nonuniform",
        7406,
        "factor_smooth",
        8,
        True,
    ),
)


def _balanced_codes(rng: np.random.Generator, n: int, levels: int) -> np.ndarray:
    if levels < 2:
        raise ValueError("levels must be at least 2")
    if n < levels:
        raise ValueError("n must be at least as large as levels so every level is represented")
    codes = np.resize(np.arange(levels, dtype=np.intp), n)
    rng.shuffle(codes)
    return codes


def prepare_case(config: CaseConfig) -> PreparedCase:
    """Generate a deterministic actuarial model with every level represented."""
    if config.family not in ("gaussian", "poisson", "gamma"):
        raise ValueError("family must be gaussian, poisson, or gamma")
    if config.structured_term not in ("random_effect", "factor_smooth"):
        raise ValueError("structured_term must be random_effect or factor_smooth")
    if config.factor_basis not in ("fs", "sz"):
        raise ValueError("factor_basis must be fs or sz")
    if config.structured_term == "random_effect" and config.factor_basis != "fs":
        raise ValueError("factor_basis has no meaning for a random-effect case")
    if config.structured_term == "random_effect" and config.random_effects not in (1, 2):
        raise ValueError("random-effect cases require random_effects=1 or 2")
    if config.structured_term == "factor_smooth" and config.random_effects not in (0, 1):
        raise ValueError("factor-smooth cases support zero or one secondary random effect")
    if config.structured_term == "factor_smooth" and config.block_size < 5:
        raise ValueError("factor-smooth block_size must be at least 5")
    if (
        config.structured_term == "factor_smooth"
        and config.factor_basis == "sz"
        and not config.global_spline
    ):
        raise ValueError("factor_basis='sz' requires global_spline=True")
    if config.structured_term == "random_effect" and config.block_size != 1:
        raise ValueError("random-effect cases require block_size=1")
    if config.small_width < 0:
        raise ValueError("small_width must be non-negative")
    if config.weights not in ("unit", "nonuniform"):
        raise ValueError("weights must be unit or nonuniform")

    rng = np.random.default_rng(config.seed)
    dominant_codes = _balanced_codes(rng, config.n, config.levels)
    frame: dict[str, np.ndarray] = {}
    eta = np.full(config.n, -0.3, dtype=np.float64)
    if config.structured_term == "random_effect":
        dominant_labels = np.asarray(
            [f"policy_{index:05d}" for index in range(config.levels)],
            dtype=object,
        )
        dominant_effect = rng.normal(scale=0.34, size=config.levels)
        dominant_effect -= dominant_effect.mean()
        frame["policy"] = dominant_labels[dominant_codes]
        eta += dominant_effect[dominant_codes]
    else:
        dominant_labels = np.asarray(
            [f"segment_{index:05d}" for index in range(config.levels)],
            dtype=object,
        )
        curve_x = rng.uniform(-1.25, 1.25, size=config.n)
        curve_amplitude = rng.normal(scale=0.24, size=config.levels)
        curve_amplitude -= curve_amplitude.mean()
        frame["curve_x"] = curve_x
        frame["curve_group"] = dominant_labels[dominant_codes]
        eta += 0.24 * np.sin(2.1 * curve_x)
        eta += curve_amplitude[dominant_codes] * (curve_x + 0.28 * curve_x**2)

    for column in range(config.small_width):
        values = rng.normal(size=config.n)
        frame[f"x{column}"] = values
        coefficient = 0.22 * np.cos(column + 0.5) / np.sqrt(column + 1.0)
        eta += coefficient * values

    secondary_level_count = 0
    has_secondary = (config.structured_term == "random_effect" and config.random_effects == 2) or (
        config.structured_term == "factor_smooth" and config.random_effects == 1
    )
    if has_secondary:
        secondary_level_count = config.secondary_levels or max(
            8,
            int(round(np.sqrt(config.levels))),
        )
        secondary_codes = _balanced_codes(rng, config.n, secondary_level_count)
        secondary_labels = np.asarray(
            [f"branch_{index:04d}" for index in range(secondary_level_count)],
            dtype=object,
        )
        secondary_effect = rng.normal(scale=0.18, size=secondary_level_count)
        secondary_effect -= secondary_effect.mean()
        frame["branch"] = secondary_labels[secondary_codes]
        eta += secondary_effect[secondary_codes]

    sample_weight = (
        None
        if config.weights == "unit"
        else rng.uniform(0.55, 1.45, size=config.n).astype(np.float64)
    )
    offset: np.ndarray | None
    if config.family == "gaussian":
        y = eta + rng.normal(scale=0.42, size=config.n)
        offset = None
    elif config.family == "poisson":
        exposure = rng.uniform(0.45, 2.2, size=config.n)
        offset = np.log(exposure)
        y = rng.poisson(exposure * np.exp(eta)).astype(np.float64)
    else:
        mean = np.exp(eta)
        y = rng.gamma(shape=4.0, scale=mean / 4.0).astype(np.float64)
        offset = None

    return PreparedCase(
        config=config,
        X=pd.DataFrame(frame),
        y=np.asarray(y, dtype=np.float64),
        sample_weight=sample_weight,
        offset=offset,
        dominant_codes=dominant_codes,
        secondary_level_count=secondary_level_count,
    )


def _new_model(prepared: PreparedCase, backend: str) -> SuperGLM:
    features = {f"x{column}": Numeric() for column in range(prepared.config.small_width)}
    interactions = []
    if prepared.config.structured_term == "random_effect":
        features["policy"] = RandomEffect()
        if prepared.config.random_effects == 2:
            features["branch"] = RandomEffect()
    else:
        if prepared.config.global_spline:
            features["curve_x"] = Spline(
                kind="ps",
                k=max(7, prepared.config.block_size),
            )
        if prepared.config.random_effects == 1:
            features["branch"] = RandomEffect()
        interactions.append(
            FactorSmooth(
                "curve_x",
                group="curve_group",
                basis=prepared.config.factor_basis,
                k=prepared.config.block_size,
            )
        )
    return SuperGLM(
        family=prepared.config.family,
        features=features,
        interactions=interactions,
        selection_penalty=0,
        direct_solve=backend,
        discrete=prepared.config.discrete,
        retain_fit_state=False,
        tol=1e-8,
        max_iter=100,
    )


def fit_once(
    prepared: PreparedCase,
    backend: str,
    *,
    max_reml_iter: int,
    reml_tol: float,
) -> tuple[SuperGLM, float]:
    """Fit one fresh model and return the unrounded wall time."""
    model = _new_model(prepared, backend)
    started = time.perf_counter()
    model.fit_reml(
        prepared.X,
        prepared.y,
        sample_weight=prepared.sample_weight,
        offset=prepared.offset,
        max_reml_iter=max_reml_iter,
        reml_tol=reml_tol,
        pirls_tol=1e-8,
        max_pirls_iter=100,
        runtime_validation="skip",
    )
    return model, time.perf_counter() - started


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def model_diagnostics(
    model: SuperGLM,
    prepared: PreparedCase,
    *,
    requested_backend: str,
) -> dict[str, Any]:
    """Collect compact numerical and phase diagnostics from a completed fit."""
    result = model.result
    reml_result = model._reml_result
    dominant_width = prepared.config.dominant_width
    p = len(result.beta)
    return {
        "requested_backend": requested_backend,
        "resolved_backend": result.direct_backend,
        "fallback_reason": result.direct_fallback_reason,
        "n": prepared.config.n,
        "coefficient_width": p,
        "dominant_width": dominant_width,
        "dominant_levels": prepared.config.levels,
        "dominant_block_size": prepared.config.block_size,
        "structured_term": prepared.config.structured_term,
        "factor_basis": prepared.config.factor_basis,
        "small_width": p - dominant_width,
        "secondary_level_count": prepared.secondary_level_count,
        "pirls_iterations": result.n_iter,
        "reml_iterations": reml_result.n_reml_iter,
        "converged": bool(reml_result.converged and result.converged),
        "termination_reason": reml_result.termination_reason,
        "objective": reml_result.objective,
        "deviance": result.deviance,
        "effective_df": result.effective_df,
        "phi": result.phi,
        "intercept": result.intercept,
        "lambdas": dict(model._reml_lambdas),
        "prediction_checksum": float(
            np.sum(
                model.predict(
                    prepared.X,
                    offset=prepared.offset,
                ),
                dtype=np.float64,
            )
        ),
        "phase_timings": _json_safe(dict(model._reml_profile)),
        "last_fit_meta": _json_safe(dict(model._last_fit_meta or {})),
    }


def parity_diagnostics(
    left: SuperGLM,
    right: SuperGLM,
    prepared: PreparedCase,
) -> dict[str, Any]:
    """Return dense-versus-structured numerical differences."""
    left_prediction = left.predict(prepared.X, offset=prepared.offset)
    right_prediction = right.predict(prepared.X, offset=prepared.offset)
    prediction_scale = np.maximum(np.abs(left_prediction), 1e-12)
    beta_scale = np.maximum(np.abs(left.result.beta), 1e-12)
    lambda_names = sorted(set(left._reml_lambdas) | set(right._reml_lambdas))
    lambda_relative = {}
    for name in lambda_names:
        left_lambda = float(left._reml_lambdas[name])
        right_lambda = float(right._reml_lambdas[name])
        lambda_relative[name] = abs(left_lambda - right_lambda) / max(
            abs(left_lambda),
            1e-12,
        )
    return {
        "prediction_max_abs": float(np.max(np.abs(left_prediction - right_prediction))),
        "prediction_max_rel": float(
            np.max(np.abs(left_prediction - right_prediction) / prediction_scale)
        ),
        "beta_max_abs": float(np.max(np.abs(left.result.beta - right.result.beta))),
        "beta_max_rel": float(np.max(np.abs(left.result.beta - right.result.beta) / beta_scale)),
        "intercept_abs": abs(float(left.result.intercept - right.result.intercept)),
        "deviance_abs": abs(float(left.result.deviance - right.result.deviance)),
        "effective_df_abs": abs(float(left.result.effective_df - right.result.effective_df)),
        "phi_abs": abs(float(left.result.phi - right.result.phi)),
        "objective_abs": abs(float(left._reml_result.objective - right._reml_result.objective)),
        "lambda_relative": lambda_relative,
    }


def _backend_sequence(requested: str) -> tuple[str, ...]:
    if requested == "both":
        return ("gram", "structured")
    return (requested,)


def _profile_backend(
    prepared: PreparedCase,
    backend: str,
    output_dir: Path,
    *,
    max_reml_iter: int,
    reml_tol: float,
) -> tuple[SuperGLM, float]:
    profiler = cProfile.Profile()
    profiler.enable()
    model, wall_time = fit_once(
        prepared,
        backend,
        max_reml_iter=max_reml_iter,
        reml_tol=reml_tol,
    )
    profiler.disable()
    profiler.dump_stats(str(output_dir / f"cprofile_{backend}.pstats"))
    write_pstats_summary(
        profiler,
        output_dir / f"cprofile_{backend}_top.txt",
        limit=100,
    )
    return model, wall_time


def run_case(
    prepared: PreparedCase,
    *,
    backend: str,
    repetitions: int,
    warmups: int,
    max_reml_iter: int,
    reml_tol: float,
    dense_parity: bool,
    dense_max_levels: int,
    sample_interval_ms: int,
    cprofile_enabled: bool,
    tracemalloc_enabled: bool,
    output_dir: Path,
) -> dict[str, Any]:
    """Run warmups, clean wall repetitions, and one instrumented fit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    backends = _backend_sequence(backend)
    if (
        dense_parity
        and "gram" not in backends
        and prepared.config.dominant_width <= dense_max_levels
    ):
        backends += ("gram",)

    sampler = SystemSampler(interval_s=sample_interval_ms / 1000.0)
    warning_records: list[dict[str, Any]] = []
    wall_results: dict[str, list[float]] = {}
    final_models: dict[str, SuperGLM] = {}
    profiled_wall: dict[str, float] = {}
    tracemalloc_wall: dict[str, float] = {}
    telemetry_wall: dict[str, float] = {}
    start_snapshot = None
    end_snapshot = None
    peak_bytes = 0
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for selected_backend in backends:
                for _ in range(warmups):
                    fit_once(
                        prepared,
                        selected_backend,
                        max_reml_iter=max_reml_iter,
                        reml_tol=reml_tol,
                    )
                    gc.collect()

                times = []
                for _ in range(repetitions):
                    model, elapsed = fit_once(
                        prepared,
                        selected_backend,
                        max_reml_iter=max_reml_iter,
                        reml_tol=reml_tol,
                    )
                    final_models[selected_backend] = model
                    times.append(elapsed)
                    gc.collect()
                wall_results[selected_backend] = times

            if cprofile_enabled:
                for selected_backend in backends:
                    profile_model, profile_wall = _profile_backend(
                        prepared,
                        selected_backend,
                        output_dir,
                        max_reml_iter=max_reml_iter,
                        reml_tol=reml_tol,
                    )
                    final_models[selected_backend] = profile_model
                    profiled_wall[selected_backend] = profile_wall

            if tracemalloc_enabled:
                tracemalloc.start(10)
                start_snapshot = tracemalloc.take_snapshot()
                for selected_backend in backends:
                    _, allocation_wall = fit_once(
                        prepared,
                        selected_backend,
                        max_reml_iter=max_reml_iter,
                        reml_tol=reml_tol,
                    )
                    tracemalloc_wall[selected_backend] = allocation_wall
                    gc.collect()
                _, peak_bytes = tracemalloc.get_traced_memory()
                end_snapshot = tracemalloc.take_snapshot()
                tracemalloc.stop()

            sampler.start()
            try:
                for selected_backend in backends:
                    _, sampled_wall = fit_once(
                        prepared,
                        selected_backend,
                        max_reml_iter=max_reml_iter,
                        reml_tol=reml_tol,
                    )
                    telemetry_wall[selected_backend] = sampled_wall
                    gc.collect()
            finally:
                sampler.stop()

            for item in caught:
                warning_records.append(
                    {
                        "message": str(item.message),
                        "category": item.category.__name__,
                        "filename": item.filename,
                        "lineno": item.lineno,
                    }
                )
    finally:
        total_wall = time.perf_counter() - started
        sampler.stop()
        if tracemalloc.is_tracing():
            _, peak_bytes = tracemalloc.get_traced_memory()
            end_snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()

    write_system_samples_csv(sampler.samples, output_dir / "system_timeseries.csv")
    if tracemalloc_enabled:
        write_tracemalloc_report(
            output_dir / "tracemalloc_top.txt",
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            peak_bytes=peak_bytes,
        )

    backend_results = {}
    for selected_backend, times in wall_results.items():
        backend_results[selected_backend] = {
            "wall_times_s": times,
            "wall_min_s": min(times),
            "wall_median_s": statistics.median(times),
            "wall_mean_s": statistics.mean(times),
            "profiled_wall_s": profiled_wall.get(selected_backend),
            "tracemalloc_wall_s": tracemalloc_wall.get(selected_backend),
            "telemetry_wall_s": telemetry_wall[selected_backend],
            "model": model_diagnostics(
                final_models[selected_backend],
                prepared,
                requested_backend=selected_backend,
            ),
        }

    parity = None
    if "gram" in final_models and "structured" in final_models:
        parity = parity_diagnostics(
            final_models["gram"],
            final_models["structured"],
            prepared,
        )
    elif "gram" in final_models and "auto" in final_models:
        parity = parity_diagnostics(
            final_models["gram"],
            final_models["auto"],
            prepared,
        )

    payload = {
        "schema_version": 1,
        "config": _json_safe(prepared.config.__dict__),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "instrumentation": {
            "repetitions": repetitions,
            "warmups": warmups,
            "max_reml_iter": max_reml_iter,
            "reml_tol": reml_tol,
            "cprofile_enabled": cprofile_enabled,
            "tracemalloc_enabled": tracemalloc_enabled,
            "total_wall_s": total_wall,
            "tracemalloc_peak_bytes": peak_bytes,
            "system_summary": summarize_system_samples(sampler.samples),
            "sampler_error": sampler.error,
        },
        "backends": backend_results,
        "parity": parity,
        "warnings": warning_records,
    }
    dump_json(output_dir / "summary.json", payload)
    return payload


def _case_from_args(args: argparse.Namespace) -> CaseConfig:
    return CaseConfig(
        n=args.n,
        levels=args.levels,
        family=args.family,
        discrete=args.discrete,
        random_effects=args.random_effects,
        secondary_levels=args.secondary_levels,
        small_width=args.small_width,
        weights=args.weights,
        seed=args.seed,
        structured_term=args.structured_term,
        block_size=(
            args.block_size
            if args.block_size is not None
            else (5 if args.structured_term == "factor_smooth" else 1)
        ),
        global_spline=args.global_spline,
        factor_basis=args.factor_basis,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--levels", type=int, default=1_000)
    parser.add_argument("--family", choices=("gaussian", "poisson", "gamma"), default="poisson")
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--random-effects", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--secondary-levels", type=int, default=None)
    parser.add_argument("--small-width", type=int, default=4)
    parser.add_argument(
        "--structured-term",
        choices=("random_effect", "factor_smooth"),
        default="random_effect",
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--factor-basis",
        choices=("fs", "sz"),
        default="fs",
    )
    parser.add_argument(
        "--global-spline",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--weights", choices=("unit", "nonuniform"), default="nonuniform")
    parser.add_argument(
        "--backend",
        choices=("gram", "structured", "auto", "both"),
        default="structured",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-reml-iter", type=int, default=10)
    parser.add_argument("--reml-tol", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--sample-interval-ms", type=int, default=500)
    parser.add_argument(
        "--cprofile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--tracemalloc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dense-max-levels", type=int, default=500)
    parser.add_argument(
        "--dense-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--matrix", choices=("core", "factor-smooth", "all"), default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        raise ValueError("repetitions must be positive and warmups must be non-negative")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir or RESULTS_ROOT / timestamp

    if args.matrix == "core":
        if args.factor_basis != "fs":
            raise ValueError("--factor-basis has no meaning for --matrix core")
        configs = CORE_MATRIX
    elif args.matrix == "factor-smooth":
        configs = tuple(
            replace(
                config,
                factor_basis=args.factor_basis,
                global_spline=config.global_spline or args.factor_basis == "sz",
            )
            for config in FACTOR_SMOOTH_MATRIX
        )
    elif args.matrix == "all":
        factor_configs = tuple(
            replace(
                config,
                factor_basis=args.factor_basis,
                global_spline=config.global_spline or args.factor_basis == "sz",
            )
            for config in FACTOR_SMOOTH_MATRIX
        )
        configs = (*CORE_MATRIX, *factor_configs)
    else:
        configs = (_case_from_args(args),)
    matrix_results = {}
    for config in configs:
        selected_backend = args.backend
        if args.matrix is not None:
            selected_backend = (
                "both" if config.dominant_width <= args.dense_max_levels else "structured"
            )
        case_output = output_root / config.slug if len(configs) > 1 else output_root
        payload = run_case(
            prepare_case(config),
            backend=selected_backend,
            repetitions=args.repetitions,
            warmups=args.warmups,
            max_reml_iter=args.max_reml_iter,
            reml_tol=args.reml_tol,
            dense_parity=args.dense_parity,
            dense_max_levels=args.dense_max_levels,
            sample_interval_ms=args.sample_interval_ms,
            cprofile_enabled=args.cprofile,
            tracemalloc_enabled=args.tracemalloc,
            output_dir=case_output,
        )
        matrix_results[config.slug] = payload
        print(f"{config.slug}: {case_output}")

    if len(configs) > 1:
        dump_json(output_root / "matrix_summary.json", matrix_results)


if __name__ == "__main__":
    main()
