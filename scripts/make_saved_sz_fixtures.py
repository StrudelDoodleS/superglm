"""Write the saved-model fixtures of ``tests/test_saved_sz_models.py`` with superglm v0.35.0.

Run it with the v0.35.0 release on the path (commit ea837196), for example from
``git worktree add <dir> v0.35.0``:

    PYTHONPATH=<dir>/src python scripts/make_saved_sz_fixtures.py tests/fixtures/saved_v0_35_0

Each record holds the fitted model, its training rows, its predictions, the
class names its retained linear system pickles and, for the models that
retained their fit state, the standard errors v0.35.0 reported.  The models
take ``direct_solve="structured"``, which on v0.35.0 fits a ``basis="sz"`` term
with ``SumToZeroBlockFactor`` (the class the one engine retired), and every
record adds the covariance block of the implied last level.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import superglm
from superglm import Categorical, FactorSmooth, Numeric, Spline, SuperGLM

CASES = {
    "sz_gaussian_exact": {"family": "gaussian", "discrete": False, "retain": True},
    "sz_poisson_discrete": {"family": "poisson", "discrete": True, "retain": True},
    "sz_gaussian_released": {"family": "gaussian", "discrete": False, "retain": False},
}


def data(seed: int, n: int = 320, K: int = 6):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, K, n)
    x = rng.uniform(size=n)
    cat = rng.integers(0, 3, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "x1": rng.normal(size=n),
            "cat": np.array([f"c{c}" for c in cat], dtype=object),
            "g": np.array([f"g{c}" for c in g], dtype=object),
        }
    )
    eta = (
        0.2 * frame["x1"].to_numpy()
        + rng.normal(0, 0.2, 3)[cat]
        + np.sin(2 * np.pi * x) * (1 + rng.normal(0, 0.3, K)[g])
    )
    return frame, eta, rng


def main(out: str) -> None:
    assert superglm.__version__ == "0.35.0", superglm.__version__
    os.makedirs(out, exist_ok=True)
    for seed, (name, spec) in enumerate(CASES.items()):
        frame, eta, rng = data(3600 + seed)
        if spec["family"] == "gaussian":
            y = eta + rng.normal(0, 0.4, len(eta))
        else:
            y = rng.poisson(np.exp(0.5 * eta)).astype(float)
        model = SuperGLM(
            family=spec["family"],
            features={"x1": Numeric(), "cat": Categorical(), "x": Spline(n_knots=4)},
            interactions=[FactorSmooth("x", group="g", basis="sz", k=5)],
            selection_penalty=0,
            direct_solve="structured",
            discrete=spec["discrete"],
            retain_fit_state=spec["retain"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(frame, y)
        state = model._linear_system_state
        fields = ("profiled_factor", "augmented_factor", "system", "penalized_operator")
        types = {} if state is None else {key: type(getattr(state, key)).__name__ for key in fields}
        record = {
            "version": superglm.__version__,
            "model": model,
            "frame": frame,
            "y": y,
            "prediction": np.asarray(model.predict(frame)),
            "state_types": types,
        }
        if spec["retain"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = model.metrics(frame, y)
            record["se"] = {key: np.asarray(value) for key, value in metrics.coefficient_se.items()}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                curves = model.factor_smooth("x:g:sz", grid=7).curves
            # every level's curve, the implied last one included (its covariance
            # is the sum-to-zero constraint's)
            record["curve_effect"] = curves["effect"].to_numpy()
            record["curve_se"] = curves["posterior_se"].to_numpy()
        with open(os.path.join(out, f"{name}.pkl"), "wb") as handle:
            pickle.dump(record, handle, protocol=5)


if __name__ == "__main__":
    main(sys.argv[1])
