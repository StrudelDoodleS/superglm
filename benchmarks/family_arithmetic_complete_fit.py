"""Complete scalar fits for the shared working-arithmetic repair.

Select a source with PYTHONPATH. Run --profile separately from --measure-time;
the profile records executed Python backends and their caller/callee counts.
Each family has an untimed small warmup and a fresh model for the measured fit.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import pstats
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

import superglm
from superglm import SuperGLM
from superglm.distributions import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson, Tweedie
from superglm.features import Numeric, Spline

FAMILIES = ("gaussian", "gamma", "poisson", "binomial", "nb2", "tweedie")


def fixture(name: str, rows: int):
    rng = np.random.default_rng(9130 + FAMILIES.index(name))
    x = rng.uniform(-1.0, 1.0, rows)
    z = rng.normal(size=rows)
    eta = 0.3 + 0.4 * np.sin(np.pi * x) + 0.1 * z
    mean = np.exp(eta)
    weights = rng.integers(1, 4, size=rows).astype(float)
    if name == "gaussian":
        family, y = Gaussian(), eta + rng.normal(scale=0.5, size=rows)
    elif name == "gamma":
        family, y = Gamma(), rng.gamma(4.0, mean / 4.0)
    elif name == "poisson":
        family, y = Poisson(), rng.poisson(mean).astype(float)
    elif name == "binomial":
        family, y = Binomial(), rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    elif name == "nb2":
        family, y = NegativeBinomial(2.0), rng.negative_binomial(2.0, 2.0 / (2.0 + mean))
    else:
        family = Tweedie(1.5)
        count = rng.poisson(2.0 * np.sqrt(mean) / 0.6)
        y = rng.gamma(np.maximum(count, 1), 0.3 * np.sqrt(mean)) * (count > 0)
    return family, pd.DataFrame({"x": x, "z": z}), np.asarray(y, dtype=float), weights


def model_for(family):
    return SuperGLM(
        family=family,
        features={"x": Spline(n_knots=12), "z": Numeric()},
        selection_penalty=0.0,
        weight_semantics="frequency",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=12_000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", choices=FAMILIES, action="append")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--measure-time", action="store_true")
    mode.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(superglm.__file__).resolve().parent
    hashes = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*.py"))
    }
    payload = {
        "source": str(source),
        "source_files": hashes,
        "source_digest": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mode": "timing" if args.measure_time else "profile",
        "load_before": os.getloadavg(),
        "families": {},
    }
    predictions = {}
    with threadpool_limits(limits=1):
        for name in args.case or FAMILIES:
            family, frame, y, weights = fixture(name, args.rows)
            model_for(family).fit(frame.iloc[:300], y[:300], sample_weight=weights[:300])
            model = model_for(family)
            profile = cProfile.Profile()
            if args.profile:
                profile.enable()
            started = time.perf_counter()
            model.fit(frame, y, sample_weight=weights)
            elapsed = time.perf_counter() - started
            if args.profile:
                profile.disable()
                profile.dump_stats(str(args.out / f"{name}.prof"))
                with (args.out / f"{name}-profile.txt").open("w") as stream:
                    stats = (
                        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative")
                    )
                    stats.print_stats(35)
                    stats.print_callers("working_weights|working_rows|pearson|weighted_residual")
            result = model.result
            predictions[name] = model.predict(frame)
            record = {
                "fit_seconds": elapsed if args.measure_time else None,
                "rows": args.rows,
                "data_sha256": hashlib.sha256(
                    frame.to_numpy().tobytes() + y.tobytes() + weights.tobytes()
                ).hexdigest(),
                "converged": result.converged,
                "iterations": result.n_iter,
                "deviance": result.deviance,
                "phi": result.phi,
                "effective_df": result.effective_df,
                "coefficients": result.beta.tolist(),
                "intercept": result.intercept,
                "direct_backend": result.direct_backend,
                "fallback_reason": result.direct_fallback_reason,
                "data_rank": result.rank_info.data.rank,
                "coefficient_rank": result.rank_info.coefficient.rank,
                "group_matrix_classes": [type(g).__name__ for g in model._dm.group_matrices],
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "native_pools": threadpool_info(),
            }
            payload["families"][name] = record
            print(
                json.dumps(
                    {
                        "family": name,
                        "converged": result.converged,
                        "fit_seconds": record["fit_seconds"],
                    }
                ),
                flush=True,
            )
            if not result.converged:
                raise RuntimeError(f"{name} did not converge")
    payload["load_after"] = os.getloadavg()
    np.savez(args.out / "predictions.npz", **predictions)
    (args.out / "receipt.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
