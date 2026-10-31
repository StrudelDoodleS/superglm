"""Declared Weibull and Burr XII families fitted through SuperLSS, checked against scipy.

Prototype for notes/research/2026-10-31-severity-family-framework.md; not production code.

    uv run python fit_declared.py OUT_DIR

1. Intercept-only fits against an independent maximum-likelihood fit of the
   same law with ``scipy.optimize`` on ``scipy.stats`` log densities.
2. A smooth in the scale on 100,000 rows: recovery of the shape parameters and
   of the scale curve, with wall time, convergence and EDF.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

import declarations as D
import sevgen
from declared_family import DeclaredFamily, register
from superglm import SuperLSS, bind_predictor, s

OUT = Path(sys.argv[1])
FITS = {
    "weibull": lambda y: (stats.weibull_min.fit(y, floc=0)[2], stats.weibull_min.fit(y, floc=0)[0]),
    "burr12": lambda y: (np.median(y), 1.0, 1.0),
}
TRUTH = {"weibull": (0.8,), "burr12": (1.5, 2.0)}


def scipy_mle(declaration, y, start):
    def negative(log_theta):
        return -np.sum(declaration.scipy_law(*np.exp(log_theta)).logpdf(y))

    result = optimize.minimize(negative, np.log(start), method="BFGS", options={"gtol": 1e-10})
    return np.exp(result.x), -result.fun


def scale_curve(x):
    return np.exp(8.0 + 0.4 * np.sin(2 * np.pi * x))


def main() -> None:
    report = {}
    rng = np.random.default_rng(20261031)
    for declaration in (D.WEIBULL, D.BURR12):
        kernel, _ = sevgen.build(declaration, OUT / "kernels")
        register(declaration, kernel, FITS[declaration.name])
        family = DeclaredFamily(declaration.name)
        n = 100_000
        frame = pd.DataFrame({"x1": rng.uniform(size=n)})
        truth_scale = scale_curve(frame.x1.to_numpy())
        y = np.asarray(declaration.scipy_law(truth_scale, *TRUTH[declaration.name]).rvs(random_state=rng))
        names = declaration.parameters

        null = SuperLSS(family, *(bind_predictor(family, name) for name in names))
        start = time.perf_counter()
        null.fit(frame, y)
        null_wall = time.perf_counter() - start
        superglm_theta = np.asarray(null.predict_parameters(frame.iloc[:1]), dtype=float)[0]
        mle, mle_loglik = scipy_mle(declaration, y, superglm_theta * 1.05)

        smooth = SuperLSS(
            family,
            bind_predictor(family, names[0], s("x1")),
            *(bind_predictor(family, name) for name in names[1:]),
        )
        start = time.perf_counter()
        smooth.fit_reml(frame, y)
        smooth_wall = time.perf_counter() - start
        fitted = np.asarray(smooth.predict_parameters(frame), dtype=float)
        result = smooth.result_
        report[declaration.name] = {
            "intercept_only": {
                "superglm_theta": superglm_theta.tolist(),
                "scipy_mle_theta": mle.tolist(),
                "max_rel_diff_theta": float(np.max(np.abs(superglm_theta - mle) / mle)),
                "superglm_loglik": null.result_.log_likelihood,
                "scipy_mle_loglik": float(mle_loglik),
                "wall_seconds": null_wall,
            },
            "smooth_scale": {
                "wall_seconds": smooth_wall,
                "converged": result.converged,
                "smoothing_converged": result.smoothing_converged,
                "total_edf": result.total_effective_df,
                "shape_truth": TRUTH[declaration.name],
                "shape_fitted": fitted[0, 1:].tolist(),
                "scale_curve_max_rel_error": float(np.max(np.abs(fitted[:, 0] / truth_scale - 1))),
                "warnings": list(result.warnings),
                "mean_prediction_first_rows": np.asarray(smooth.predict(frame.iloc[:3])).tolist(),
            },
        }
        print(declaration.name, json.dumps(report[declaration.name]), flush=True)
    (OUT / "fit_declared.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
