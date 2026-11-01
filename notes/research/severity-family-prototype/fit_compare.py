"""A complete GammaLS fit with the hand-written row kernel against the generated one.

Prototype for notes/research/2026-10-31-severity-family-framework.md; not production code.

    uv run python fit_compare.py OUT_DIR [REPEATS]

``GeneratedGammaLS`` is ``GammaLS`` with only ``evaluate_natural`` (orders 0-2)
replaced by the generated kernel; plan, initialization, links, orders 3-4 and
every other protocol stay hand-written, so the comparison isolates the kernel.
Arms alternate, each after a small warm-up fit that compiles both kernels.
Unit weights, so prior and frequency semantics give the same law.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import declarations as D
import sevgen
from superglm import GammaLS, SuperLSS, s
from superglm.distributional import NaturalLikelihoodEvaluation

OUT = Path(sys.argv[1])
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
KERNEL, _ = sevgen.build(D.GAMMA, OUT / "kernels")


class GeneratedGammaLS(GammaLS):
    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        value, score, hessian = KERNEL.evaluate(y, theta, plan.weights.values, int(derivative_order))
        valid = np.isfinite(value)
        for channel in (score, hessian):
            if channel is not None:
                valid &= np.all(np.isfinite(channel), axis=1)
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=value,
            parameter_independent_carrier=plan.parameter_independent_carrier,
            score=score,
            hessian_packed=hessian,
            valid=valid,
        )


def book(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.uniform(size=n) for name in ("x1", "x2", "x3")})
    mean = np.exp(8.5 + 0.3 * np.sin(2 * np.pi * frame.x1) + 0.4 * frame.x2)
    cv = 1.2 * np.exp(0.25 * (frame.x3 - 0.5))
    y = stats.gamma(a=cv**-2, scale=mean * cv**2).rvs(random_state=rng)
    return frame, np.asarray(y)


def fit(family_type, frame, y):
    family = family_type()
    model = SuperLSS(family, family.mean(s("x1"), s("x2")), family.scale(s("x3")))
    start = time.perf_counter()
    model.fit_reml(frame, y)
    wall = time.perf_counter() - start
    return model, wall


def record(model, wall, frame) -> dict:
    result = model.result_
    profile = model.diagnose().profile
    phases = {p.name: [p.seconds, p.calls] for p in profile.phases}
    return {
        "wall_seconds": wall,
        "log_likelihood": result.log_likelihood,
        "total_edf": result.total_effective_df,
        "converged": result.converged,
        "n_inner_iter": result.n_inner_iter,
        "n_smoothing_iter": result.n_smoothing_iter,
        "smoothing_parameters": dict(result.smoothing_parameters),
        "phases": phases,
        "outer_iterations": profile.outer_iterations,
        "coefficient_fits": profile.coefficient_fits,
        "warnings": list(result.warnings),
    }


def main() -> None:
    small_frame, small_y = book(5_000, 1)
    for family_type in (GammaLS, GeneratedGammaLS):
        fit(family_type, small_frame, small_y)
    frame, y = book(200_000, 20261031)
    runs = {"GammaLS": [], "GeneratedGammaLS": []}
    predictions = {}
    for _ in range(REPEATS):
        for family_type in (GammaLS, GeneratedGammaLS):
            model, wall = fit(family_type, frame, y)
            runs[family_type.__name__].append(record(model, wall, frame))
            predictions[family_type.__name__] = model.predict_parameters(frame)
            print(family_type.__name__, f"{wall:.2f} s", flush=True)
    hand = np.asarray(predictions["GammaLS"], dtype=float)
    gen = np.asarray(predictions["GeneratedGammaLS"], dtype=float)
    summary = {
        "runs": runs,
        "max_rel_diff_predicted_parameters": float(np.max(np.abs(gen - hand) / np.abs(hand))),
    }
    (OUT / "fit_compare.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}))


if __name__ == "__main__":
    main()
