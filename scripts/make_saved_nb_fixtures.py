"""Write the saved-model fixtures of ``tests/test_saved_nb_models.py`` with superglm v0.35.0.

Run it with the v0.35.0 release on the path (commit ea837196), for example from
``git worktree add <dir> v0.35.0``:

    PYTHONPATH=<dir>/src python scripts/make_saved_nb_fixtures.py tests/fixtures/saved_v0_35_0

The data are small synthetic NB2 counts.  ``estimate_theta`` keeps v0.35.0's
``NBProfileResult`` on the model, which records each interval as a
``(lower, upper)`` tuple and no endpoint status.  Each record holds the fitted
model (or, for ``nb_estimate_theta_result``, only the result ``estimate_theta``
returned), its training rows, its predictions, the estimate and intervals
v0.35.0 reported, and, per interval side, whether v0.35.0's likelihood-ratio
excess changed sign between the estimate and the end of its search range: a
side without a sign change is where v0.35.0's root search stopped.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2

import superglm
from superglm import Categorical, NegativeBinomial, Numeric, Spline, SuperGLM
from superglm.profiling.nb import _nb2_nll
from superglm.solvers.dispersion import dispersion_likelihood_size

# theta is the NB2 shape the counts are drawn with; "summary" saves the model
# after summary(), which computes its 95% interval; "bare" pickles the result
# estimate_theta returns, with its 95% interval, on its own.
CASES = {
    "nb_estimate_theta_fit": {"fit_mode": "fit", "theta": 2.0, "summary": True, "n": 400},
    "nb_estimate_theta_reml": {"fit_mode": "reml", "theta": 2.0, "summary": False, "n": 400},
    "nb_near_poisson": {"fit_mode": "fit", "theta": 200.0, "summary": True, "n": 300},
    "nb_estimate_theta_result": {"fit_mode": "fit", "theta": 2.0, "bare": True, "n": 300},
}


def data(seed: int, theta: float, n: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=n)
    cat = rng.integers(0, 3, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "x1": rng.normal(size=n),
            "cat": np.array([f"c{c}" for c in cat], dtype=object),
        }
    )
    eta = 0.5 + 0.2 * frame["x1"].to_numpy() + rng.normal(0, 0.2, 3)[cat] + 0.4 * np.sin(4 * x)
    mu = np.exp(eta)
    y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
    return frame, y


def no_crossing(result, alpha: float) -> tuple[bool, bool]:
    """Per side, whether v0.35.0's ``profile_ci_theta`` excess keeps its sign over its bracket."""
    y, mu, w = result._y, result._mu, result._weights
    semantics = result._weight_semantics
    size = dispersion_likelihood_size(w, weight_semantics=semantics)
    nll_hat = _nb2_nll(y, mu, w, result.theta_hat, weight_semantics=semantics)
    cutoff = chi2.ppf(1.0 - alpha, 1)

    def excess(theta: float) -> float:
        nll = _nb2_nll(y, mu, w, theta, weight_semantics=semantics)
        return 2.0 * size * (nll - nll_hat) - cutoff

    centre = excess(result.theta_hat)
    lower_end = min(0.01, result.theta_hat / 100.0)
    upper_end = max(500.0, result.theta_hat * 100.0)
    return bool(excess(lower_end) * centre > 0.0), bool(excess(upper_end) * centre > 0.0)


def profile_record(result) -> dict:
    return {
        "theta_hat": float(result.theta_hat),
        "nll": float(result.nll),
        "converged": bool(result.converged),
        "ci": {alpha: tuple(map(float, ci)) for alpha, ci in result._ci_cache.items()},
        "no_crossing": {alpha: no_crossing(result, alpha) for alpha in result._ci_cache},
        "iterates": {float(theta): float(nll) for theta, nll in result.cache.items()},
    }


def main(out: str) -> None:
    assert superglm.__version__ == "0.35.0", superglm.__version__
    os.makedirs(out, exist_ok=True)
    for seed, (name, spec) in enumerate(CASES.items()):
        frame, y = data(5200 + seed, spec["theta"], spec["n"])
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            features={"x": Spline(kind="ps", k=6), "x1": Numeric(), "cat": Categorical()},
            selection_penalty=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returned = model.estimate_theta(frame, y, fit_mode=spec["fit_mode"])
            if spec.get("bare"):
                returned.ci(0.05)
            elif spec["summary"]:
                str(model.summary())
        if spec.get("bare"):
            record = {"version": superglm.__version__, "result": returned}
            record["profile"] = profile_record(returned)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = pickle.loads(pickle.dumps(model, protocol=5)).metrics(frame, y)
            record = {
                "version": superglm.__version__,
                "model": model,
                "frame": frame,
                "y": y,
                "prediction": np.asarray(model.predict(frame)),
                "profile": profile_record(model._nb_profile_result),
                "se": {key: np.asarray(value) for key, value in metrics.coefficient_se.items()},
            }
        with open(os.path.join(out, f"{name}.pkl"), "wb") as handle:
            pickle.dump(record, handle, protocol=5)
        print(name, record["profile"]["theta_hat"], record["profile"]["ci"])
        print("   no crossing:", record["profile"]["no_crossing"])


if __name__ == "__main__":
    main(sys.argv[1])
