"""Write the saved-model fixtures of ``tests/test_saved_tweedie_models.py`` with superglm v0.35.0.

Run it with the v0.35.0 release on the path (commit ea837196), for example from
``git worktree add <dir> v0.35.0``:

    PYTHONPATH=<dir>/src python scripts/make_saved_tweedie_fixtures.py tests/fixtures/saved_v0_35_0

Case names after the directory write only those cases.

The data are small synthetic compound Poisson-Gamma draws.  A v0.35.0 REML fit
of a Tweedie model keeps its saturated-density memo
(``REMLResult.tweedie_scale_data``, a ``superglm.reml.scale.TweedieScaleProfileData``
holding a ``superglm.profiling.tweedie._PreparedTweedieDensity``), and
``estimate_p`` keeps the v0.35.0 ``TweedieProfileResult``.  Each record holds
the fitted model, its training rows, its predictions, the class names its
retained linear system pickles and the standard errors v0.35.0 reported;
``tweedie_estimate_p_result`` holds only the result ``estimate_p`` returned,
pickled on its own, and the estimate and interval v0.35.0 reported.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import superglm
from superglm import Categorical, FactorSmooth, Numeric, RandomEffect, Spline, SuperGLM, Tweedie

CASES = {
    "tweedie_re_exact": {"kind": "re", "discrete": False, "estimate_p": None, "n": 600},
    "tweedie_fs_discrete": {"kind": "fs", "discrete": True, "estimate_p": None, "n": 800},
    "tweedie_no_group": {"kind": "none", "discrete": False, "estimate_p": None, "n": 400},
    "tweedie_estimate_p_reml": {"kind": "none", "discrete": False, "estimate_p": "reml", "n": 300},
    "tweedie_estimate_p_fit": {"kind": "none", "discrete": False, "estimate_p": "fit", "n": 300},
    "tweedie_estimate_p_result": {
        "kind": "none",
        "discrete": False,
        "estimate_p": "fit",
        "n": 300,
        "bare": True,
    },
}


def data(seed: int, n: int = 800, K: int = 40):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, K, n)
    x = rng.uniform(size=n)
    cat = rng.integers(0, 3, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "x1": rng.normal(size=n),
            "cat": np.array([f"c{c}" for c in cat], dtype=object),
            "g": np.array([f"g{c:02d}" for c in g], dtype=object),
        }
    )
    eta = (
        0.2 * frame["x1"].to_numpy()
        + rng.normal(0, 0.2, 3)[cat]
        + 0.4 * np.sin(2 * np.pi * x)
        + rng.normal(0, 0.3, K)[g]
    )
    # compound Poisson-Gamma at p = 1.5, phi = 1: Poisson(mu^0.5 / 0.5) Gamma(1, 0.5 mu^0.5) terms
    mu = np.exp(eta)
    counts = rng.poisson(2.0 * np.sqrt(mu))
    y = np.array([rng.gamma(1.0, 0.5 * np.sqrt(m), c).sum() for c, m in zip(counts, mu)])
    return frame, y


def profile_record(profile) -> dict:
    return {
        "p_hat": float(profile.p_hat),
        "phi_hat": float(profile.phi_hat),
        "nll": float(profile.nll),
        "ci": {alpha: tuple(map(float, ci)) for alpha, ci in profile._ci_cache.items()},
    }


def main(out: str, names: list[str]) -> None:
    assert superglm.__version__ == "0.35.0", superglm.__version__
    os.makedirs(out, exist_ok=True)
    for seed, (name, spec) in enumerate(CASES.items()):
        if names and name not in names:
            continue
        frame, y = data(5100 + seed, n=spec["n"])
        features = {"x1": Numeric(), "cat": Categorical()}
        interactions = []
        if spec["kind"] == "re":
            features["x"] = Spline(kind="ps", k=6)
            features["g"] = RandomEffect()
        elif spec["kind"] == "fs":
            interactions = [FactorSmooth("x", group="g", basis="fs", k=5)]
        else:
            features["x"] = Spline(kind="ps", k=6)
        model = SuperGLM(
            family=Tweedie(p=1.5),
            features=features,
            interactions=interactions,
            selection_penalty=0,
            direct_solve="auto",
            discrete=spec["discrete"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if spec["estimate_p"] is not None:
                # ci_alpha computes the interval before the result is installed,
                # so the model pickles v0.35.0's interval details as well
                returned = model.estimate_p(frame, y, fit_mode=spec["estimate_p"], ci_alpha=0.05)
            else:
                model.fit_reml(frame, y)
        if spec.get("bare"):
            record = {"version": superglm.__version__, "result": returned}
            record["profile"] = profile_record(returned)
            with open(os.path.join(out, f"{name}.pkl"), "wb") as handle:
                pickle.dump(record, handle, protocol=5)
            print(name, type(returned).__name__, record["profile"])
            continue
        state = model._linear_system_state
        fields = ("profiled_factor", "augmented_factor", "system", "penalized_operator")
        types = {} if state is None else {key: type(getattr(state, key)).__name__ for key in fields}
        reml = model._reml_result
        profile = model._tweedie_profile_result
        # the no-group model is saved with its metrics cache filled; the others
        # are saved before any inference call, as a fresh fit is
        inspected = (
            model
            if spec["kind"] == "none" and profile is None
            else pickle.loads(pickle.dumps(model, protocol=5))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = inspected.metrics(frame, y)
        record = {
            "version": superglm.__version__,
            "model": model,
            "frame": frame,
            "y": y,
            "prediction": np.asarray(model.predict(frame)),
            "state_types": types,
            "scale_data_type": None if reml is None else type(reml.tweedie_scale_data).__name__,
            "profile": None if profile is None else profile_record(profile),
            "power": float(model._distribution.p),
            "phi": float(model._result.phi),
            "se": {key: np.asarray(value) for key, value in metrics.coefficient_se.items()},
        }
        with open(os.path.join(out, f"{name}.pkl"), "wb") as handle:
            pickle.dump(record, handle, protocol=5)
        print(name, types.get("augmented_factor"), record["scale_data_type"], record["profile"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
