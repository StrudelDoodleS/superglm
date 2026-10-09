"""Write the v0.36.0 fixtures of ``tests/test_saved_sz_models.py``: ``sz`` models whose levels the data identify only in part.

Run it with the v0.36.0 release on the path (tag ``v0.36.0``), for example from
``git worktree add <dir> v0.36.0``:

    PYTHONPATH=<dir>/src python scripts/make_saved_sz_v0_36_0_fixtures.py tests/fixtures/saved_v0_36_0

v0.36.0 recorded, at fit, which ``basis="sz"`` levels its data identify only in
part (#432) and predicted them by convention: a thin level keeps what its rows
identify, a level whose unpenalized line separates the response stays out of
the population curve, and with every level thin the population is the
canonical point.  Later releases let a term penalize its lines
(``select=True``, #444); a model saved by v0.36.0 has no such option and must
still predict as it did.  Each record holds the model (fit state
released), its training rows, a grid over every level and the linear
predictors v0.36.0 gave on them, conditional and population.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import superglm
from superglm import FactorSmooth, LambdaPolicy, Spline, SuperGLM
from superglm.model import base


def separated():
    """A zero-claim level, one whose claims sit below its other rows, and a two-sided one."""
    rng = np.random.default_rng(21)
    K, n = 12, 3600
    g = rng.integers(0, K, n)
    x = rng.uniform(size=n)
    y = rng.poisson(np.exp(0.2 + np.sin(3 * x) + rng.normal(0, 0.3, K)[g])).astype(float)
    y[g == 0] = 0.0
    one, two = np.flatnonzero(g == 1), np.flatnonzero(g == 2)
    x[one] = 0.25 + 0.7 * rng.uniform(size=len(one))
    x[one[:3]] = 0.2
    y[one] = 0.0
    y[one[:3]] = 1.0
    y[two] = 0.0
    x[two[:3]] = 0.5
    y[two[:3]] = 2.0
    frame = pd.DataFrame({"x": x, "g": np.array([f"g{v:03d}" for v in g], dtype=object)})
    return "poisson", frame, y


def all_thin():
    """Ten levels, each with 100 rows at one ``x`` (the Sol review's fixture)."""
    rng = np.random.default_rng(4)
    x = np.repeat(np.linspace(0.01, 0.99, 10), 100)
    y = np.sin(4 * x) + 0.2 * rng.normal(size=1000)
    frame = pd.DataFrame({"x": x, "g": np.repeat([f"g{i}" for i in range(10)], 100)})
    return "gaussian", frame, y


def one_row():
    """Eight levels, two of them with a single row."""
    rng = np.random.default_rng(31)
    K, n = 8, 800
    g = rng.integers(0, K, n)
    for level in (0, 1):
        rows = np.flatnonzero(g == level)
        g[rows[1:]] = K - 1
    x = rng.uniform(size=n)
    y = np.sin(3 * x) + rng.normal(0, 0.3, K)[g] * (1 + x) + rng.normal(0, 0.2, n)
    frame = pd.DataFrame({"x": x, "g": np.array([f"g{v:03d}" for v in g], dtype=object)})
    return "gaussian", frame, y


CASES = {
    "sz_poisson_separated": separated,
    "sz_gaussian_all_thin": all_thin,
    "sz_gaussian_one_row": one_row,
}


def main(out: str) -> None:
    assert superglm.__version__ == "0.36.0", superglm.__version__
    os.makedirs(out, exist_ok=True)
    for name, make in CASES.items():
        family, frame, y = make()
        model = SuperGLM(
            family=family,
            features={"x": Spline(n_knots=6, lambda_policy=LambdaPolicy.fixed(1.0))},
            interactions=[
                FactorSmooth(
                    "x", group="g", basis="sz", lambda_policy={"wiggle": LambdaPolicy.fixed(1.0)}
                )
            ],
            selection_penalty=0,
            retain_fit_state=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(frame, y)
        levels = sorted(frame["g"].unique())
        x = np.linspace(float(frame["x"].min()), float(frame["x"].max()), 9)
        grid = pd.DataFrame(
            {"x": np.tile(x, len(levels)), "g": np.repeat(np.array(levels, dtype=object), len(x))}
        )
        record = {
            "version": superglm.__version__,
            "model": model,
            "frame": frame,
            "grid": grid,
            "eta": base.predict_eta_exact(model, frame, warn=False),
            "eta_grid": base.predict_eta_exact(model, grid, warn=False),
            "eta_population": base.predict_eta_exact(
                model, grid, random_effects="population", warn=False
            ),
        }
        with open(os.path.join(out, f"{name}.pkl"), "wb") as handle:
            pickle.dump(record, handle, protocol=5)


if __name__ == "__main__":
    main(sys.argv[1])
