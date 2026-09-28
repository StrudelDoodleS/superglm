"""NB2 auto-theta under ``fit_reml`` at 200,000 rows: the NB case of the profiling rebuild.

Tiles the ``nb_worst.csv`` design to 200,000 rows, jitters ``x`` with a fixed
seed, adds a six-level factor drawn with a fixed seed, and fits
``families.nb2()`` (auto theta) with one cubic regression spline and the factor
under ``fit_reml``. Prints one JSON line: wall and CPU seconds, peak RSS,
theta_hat, the REML fits the theta/lambda alternation ran and their total REML
iterations. Run each repetition in a fresh process with the thread pools pinned:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    uv run python benchmarks/nb_auto_theta_reml.py
"""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import superglm
from superglm import Categorical, CubicRegressionSpline, SuperGLM, families
from superglm.model import fit_ops

if __package__:
    from benchmarks import _platform
else:  # run by filename
    import _platform

ROWS = 200_000
SEED = 20260926
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "nb_worst.csv"


def replicated_design() -> tuple[pd.DataFrame, np.ndarray]:
    data = pd.read_csv(FIXTURE)
    reps = -(-ROWS // len(data))
    rng = np.random.default_rng(SEED)
    x = np.tile(data["x"].to_numpy(dtype=np.float64), reps)[:ROWS]
    y = np.tile(data["y"].to_numpy(dtype=np.float64), reps)[:ROWS]
    x = np.clip(x + rng.normal(0.0, 1e-3, ROWS), 0.0, 1.0)
    factor = np.char.add("g", rng.integers(0, 6, ROWS).astype(str))
    return pd.DataFrame({"x": x, "g": factor}), y


def main() -> None:
    X, y = replicated_design()
    model = SuperGLM(
        family=families.nb2(),
        features={"x": CubicRegressionSpline(n_knots=20), "g": Categorical()},
    )
    reml_iterations: list[int] = []
    fit_reml_once = fit_ops._fit_reml_in_workspace

    @functools.wraps(fit_reml_once)
    def counted(fitted_model, *args, **kwargs):
        recorder = fit_reml_once(fitted_model, *args, **kwargs)
        reml_iterations.append(int(fitted_model._reml_result.n_reml_iter))
        return recorder

    superglm.warmup()
    with patch.object(fit_ops, "_fit_reml_in_workspace", counted):
        wall, cpu = time.perf_counter(), time.process_time()
        model.fit_reml(X, y)
        wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    record = {
        "superglm": str(Path(superglm.__file__).resolve().parent),
        "rows": ROWS,
        "wall_s": wall,
        "cpu_s": cpu,
        "peak_rss_mb": _platform.peak_rss().bytes / 2**20,
        "load_average": _platform.load_average(),
        "theta_hat": float(model._nb_profile_result.theta_hat),
        "theta_converged": bool(model._nb_profile_result.converged),
        "reml_fits": len(reml_iterations),
        "reml_iterations": reml_iterations,
        "direct_backend": str(getattr(model._solver_pirls_result(), "direct_backend", None)),
    }
    print(json.dumps(record))


if __name__ == "__main__":
    main()
