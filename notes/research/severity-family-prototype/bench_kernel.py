"""Per-row cost of generated kernels against the hand-written GammaLS kernel.

Prototype for notes/research/2026-10-31-severity-family-framework.md; not production code.

    uv run python bench_kernel.py OUT_DIR    # pin every thread pool to 1 first

Best of seven calls on 10^6 rows, after a warm-up call, including the column
copies and output allocation each adapter call pays.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

import declarations as D
import sevgen
from superglm.distributional.kernels.gamma import evaluate_gamma_rows

N = 1_000_000
REPEATS = 7


def best_of(call) -> float:
    call()
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return min(times) / N * 1e9


def rows(declaration, theta_columns, rng):
    law = declaration.scipy_law(*theta_columns)
    return np.asarray(law.rvs(random_state=rng), dtype=np.float64)


def main() -> None:
    out_dir = Path(sys.argv[1])
    rng = np.random.default_rng(20261031)
    weights = np.ones(N)
    results = {}

    mean = 5000.0 * np.exp(0.3 * rng.standard_normal(N))
    cv = 1.2 * np.exp(0.1 * rng.standard_normal(N))
    y = stats.gamma(a=cv**-2, scale=mean * cv**2).rvs(random_state=rng)
    theta = np.column_stack((mean, cv))
    kernel, _ = sevgen.build(D.GAMMA, out_dir)
    for order in (0, 2):
        results[f"gamma_hand_o{order}"] = best_of(
            lambda o=order: evaluate_gamma_rows(y, mean, cv, weights, "prior", derivative_order=o)
        )
        results[f"gamma_generated_o{order}"] = best_of(
            lambda o=order: kernel.evaluate(y, theta, weights, o)
        )
    hand = evaluate_gamma_rows(y, mean, cv, weights, "prior", derivative_order=2)
    gen = kernel.evaluate(y, theta, weights, 2)
    results["gamma_max_rel_diff"] = {
        "value": float(np.max(np.abs(gen[0] - hand.optimizing_log_likelihood) / np.abs(hand.optimizing_log_likelihood))),
        "score": float(np.max(np.abs(gen[1] - hand.score) / np.max(np.abs(hand.score), axis=1, keepdims=True))),
        "hessian": float(np.max(np.abs(gen[2] - hand.hessian_packed) / np.max(np.abs(hand.hessian_packed), axis=1, keepdims=True))),
    }

    for declaration, columns in (
        (D.WEIBULL, (5000.0 * np.exp(0.3 * rng.standard_normal(N)), np.full(N, 0.8))),
        (D.WEIBULL_MEAN, (5000.0 * np.exp(0.3 * rng.standard_normal(N)), np.full(N, 0.8))),
        (D.BURR12, (5000.0 * np.exp(0.3 * rng.standard_normal(N)), np.full(N, 1.5), np.full(N, 2.0))),
        (D.LOG_LOGISTIC, (5000.0 * np.exp(0.3 * rng.standard_normal(N)), np.full(N, 2.5))),
        (D.LOMAX, (5000.0 * np.exp(0.3 * rng.standard_normal(N)), np.full(N, 2.5))),
    ):
        generated, _ = sevgen.build(declaration, out_dir)
        response = rows(declaration, columns, rng)
        stacked = np.column_stack(columns)
        name = declaration.name
        for order in (0, 2):
            results[f"{name}_o{order}"] = best_of(
                lambda o=order: generated.evaluate(response, stacked, weights, o)
            )
        results[f"{name}_logsf_o2"] = best_of(lambda: generated.tail("logsf", response, stacked, 2))
        results[f"{name}_logcdf_o2"] = best_of(lambda: generated.tail("logcdf", response, stacked, 2))
        results[f"{name}_d34"] = best_of(lambda: generated.third_fourth(response, stacked, weights))

    (out_dir / "bench_kernel.json").write_text(json.dumps(results, indent=1))
    for key, value in results.items():
        print(key, value if isinstance(value, dict) else f"{value:.1f} ns/row")


if __name__ == "__main__":
    main()
