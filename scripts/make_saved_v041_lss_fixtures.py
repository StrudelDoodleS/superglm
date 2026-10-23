"""Write the saved ``SuperLSS`` fixtures of ``tests/test_saved_superlss_v041.py`` with superglm v0.41.0.

Run it with the v0.41.0 release on the path (commit 7b1732b7), for example from
``git worktree add <dir> v0.41.0``:

    PYTHONPATH=<dir>/src python scripts/make_saved_v041_lss_fixtures.py tests/fixtures/saved_v0_41_0

Each case writes two xz-compressed joblib files. ``<case>.model.joblib.xz`` is
the fitted model exactly as a user saved it, with ``joblib.dump(model, path)``;
``<case>.reference.joblib.xz`` holds the scoring frame, the linear predictors
``predict_link`` made, and the conditional means ``predict`` made. The two
cases are a Gaussian location-scale and a Gamma location-scale model, each
with one smooth and one categorical term, fitted by ``fit_reml`` on a few
hundred rows.
"""

from __future__ import annotations

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

import superglm as sg

N_ROWS = 300
LEVELS = np.array(["a", "b", "c"], dtype=object)


def data(seed: int = 41) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, N_ROWS)
    code = rng.integers(0, 3, N_ROWS)
    frame = pd.DataFrame({"x": x, "c": LEVELS[code]})
    location = 0.4 + 0.5 * np.sin(0.6 * x) + np.array([0.0, 0.3, -0.2])[code]
    spread = np.exp(-0.7 + 0.25 * np.array([0.0, 1.0, -1.0])[code])
    return frame, {
        "gaussian": location + spread * rng.standard_normal(N_ROWS),
        "gamma": rng.gamma(1.0 / spread**2, np.exp(location) * spread**2),
    }


def cases(frame: pd.DataFrame, responses: dict[str, np.ndarray]):
    smooth, group = sg.s("x", kind="cr", k=6), sg.cat("c")
    gaussian, gamma = sg.GaussianLS(), sg.GammaLS()
    return {
        "lss_gaussian": (
            sg.SuperLSS(gaussian, gaussian.location(smooth, group), gaussian.scale(group)),
            responses["gaussian"],
        ),
        "lss_gamma": (
            sg.SuperLSS(gamma, gamma.mean(smooth, group), gamma.scale(group)),
            responses["gamma"],
        ),
    }


def main(out: str) -> None:
    assert sg.__version__ == "0.41.0", f"run with superglm 0.41.0, not {sg.__version__}"
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    frame, responses = data()
    for name, (model, response) in cases(frame, responses).items():
        model.fit_reml(frame, response)
        joblib.dump(model, os.path.join(out, f"{name}.model.joblib.xz"), compress=("xz", 9))
        reference = {
            "version": sg.__version__,
            "frame": frame,
            "eta": model.predict_link(frame).to_numpy(dtype=np.float64),
            "prediction": np.asarray(model.predict(frame), dtype=np.float64),
        }
        joblib.dump(reference, os.path.join(out, f"{name}.reference.joblib.xz"), compress=("xz", 9))


if __name__ == "__main__":
    main(sys.argv[1])
