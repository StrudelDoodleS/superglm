"""Write the saved-model fixtures of ``tests/test_userspace_040.py`` with superglm v0.40.0.

Run it with the v0.40.0 release on the path (commit 542f5826), for example from
``git worktree add <dir> v0.40.0``:

    PYTHONPATH=<dir>/src python scripts/make_saved_v040_fixtures.py tests/fixtures/saved_v0_40_0

Each ``SuperGLM`` or wrapper record holds the pickled model, the fit method,
its rows and response, and the predictions v0.40.0 made; each ``SuperLSS`` record holds ``to_bytes()`` (v0.40.0
cannot pickle a ``SuperLSS``), the rows, its coefficients and the linear
predictors and means it predicted. The
models span the routes a saved book uses: exact and binned REML, a random
effect, an ``fs`` and an ``sz`` factor smooth, SCOP and QP monotone splines,
tensor and spline-by-category pairs (in SuperGLM: a SuperLSS ``ti`` or ``re`` term serialises to megabytes on v0.40.0), Tweedie, a penalised ``fit``, the sklearn
wrapper and every SuperLSS family.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import superglm as sg
from superglm import (
    Categorical,
    Constraint,
    FactorSmooth,
    Numeric,
    PSpline,
    RandomEffect,
    Spline,
    SuperGLM,
    SuperLSS,
    cat,
    s,
)


def data(n: int = 250, seed: int = 40):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "x": rng.uniform(0, 10, n),
            "z": rng.uniform(-2, 2, n),
            "lin": rng.normal(size=n),
            "c": np.array([f"c{v}" for v in rng.integers(0, 4, n)], dtype=object),
            "g": np.array([f"g{v:02d}" for v in rng.integers(0, 15, n)], dtype=object),
        }
    )
    eta = (
        0.08 * frame["x"].to_numpy()
        + 0.3 * np.sin(frame["z"].to_numpy())
        + rng.normal(0, 0.2, 4)[frame["c"].str[1:].astype(int).to_numpy()]
        + 0.1 * frame["lin"].to_numpy()
    )
    eta -= eta.mean()
    mu = np.exp(eta)
    counts = rng.poisson(0.6 * mu)
    responses = {
        "poisson": rng.poisson(mu).astype(float),
        "gamma": rng.gamma(3.0, mu / 3.0),
        "gaussian": 1.0 + eta + rng.normal(0, 0.4, n),
        "tweedie": np.array([rng.gamma(2.0, 1.0, k).sum() for k in counts]),
    }
    return frame, responses


MAINS = {
    "x": Spline(kind="cr", k=8),
    "z": Spline(kind="cr", k=6),
    "c": Categorical(),
    "lin": Numeric(),
}

GLM_CASES = {
    "glm_poisson_exact": dict(family="poisson", features=MAINS, fit="fit_reml"),
    "glm_poisson_binned": dict(family="poisson", features=MAINS, discrete=True, fit="fit_reml"),
    "glm_gamma_re": dict(family="gamma", features={**MAINS, "g": RandomEffect()}, fit="fit_reml"),
    "glm_poisson_fs": dict(
        family="poisson",
        features={"x": Spline(kind="cr", k=8)},
        interactions=[FactorSmooth("z", group="c", basis="fs", k=5)],
        fit="fit_reml",
    ),
    "glm_poisson_sz": dict(
        family="poisson",
        features={"x": Spline(kind="cr", k=8), "z": Spline(kind="cr", k=6)},
        interactions=[FactorSmooth("z", group="c", basis="sz", k=5)],
        fit="fit_reml",
    ),
    "glm_poisson_scop": dict(
        family="poisson",
        features={
            "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
            "c": Categorical(),
        },
        fit="fit_reml",
    ),
    "glm_poisson_qp": dict(
        family="poisson",
        features={
            "x": Spline(kind="bs", k=8, constraint=Constraint.fit.increasing),
            "c": Categorical(),
        },
        fit="fit_reml",
    ),
    "glm_poisson_pairs": dict(
        family="poisson",
        features={"x": Spline(kind="cr", k=6), "z": Spline(kind="cr", k=5), "c": Categorical()},
        interactions=[("x", "z"), ("z", "c")],
        fit="fit_reml",
    ),
    "glm_tweedie": dict(family="tweedie", features=MAINS, fit="fit_reml"),
    "glm_gaussian_fit": dict(family="gaussian", features=MAINS, fit="fit"),
}


def _lss_cases():
    gaussian = sg.GaussianLS()
    gamma = sg.GammaLS()
    nb = sg.NegativeBinomialLS()
    tweedie = sg.TweedieLSS()
    two_piece = sg.TwoPieceNormalLSS()
    return {
        "lss_gaussian": (
            "gaussian",
            [
                gaussian,
                gaussian.location(s("x", kind="cr", k=6), cat("c")),
                gaussian.scale(),
            ],
        ),
        "lss_gamma": (
            "gamma",
            [gamma, gamma.mean(s("x", kind="cr", k=8), cat("c")), gamma.scale()],
        ),
        "lss_nb": ("poisson", [nb, nb.mean(s("x", kind="cr", k=8), cat("c")), nb.theta()]),
        "lss_tweedie": (
            "tweedie",
            [tweedie, tweedie.mu(s("x", kind="cr", k=6)), tweedie.phi(), tweedie.p()],
        ),
        "lss_two_piece": (
            "gaussian",
            [
                two_piece,
                two_piece.location(s("x", kind="cr", k=6)),
                two_piece.scale(),
                two_piece.skew(),
            ],
        ),
    }


def main(out: str) -> None:
    assert sg.__version__ == "0.40.0", sg.__version__
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    frame, responses = data()
    for name, case in GLM_CASES.items():
        case = dict(case)
        fit = case.pop("fit")
        family = case.pop("family")
        if family == "tweedie":
            family = sg.families.tweedie(p=1.5)
        model = SuperGLM(family=family, **case)
        y = responses[name.split("_")[1]]
        getattr(model, fit)(frame, y)
        record = {
            "version": sg.__version__,
            "model": model,
            "fit": fit,
            "frame": frame,
            "y": y,
            "prediction": np.asarray(model.predict(frame), dtype=np.float64),
        }
        with open(os.path.join(out, f"{name}.pkl"), "wb") as handle:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)

    wrapper = sg.SuperGLMRegressor(
        family="poisson", features={"x": Spline(kind="cr", k=8), "c": Categorical()}
    )
    wrapper.fit(frame[["x", "c"]], responses["poisson"])
    record = {
        "version": sg.__version__,
        "model": wrapper,
        "fit": "fit",
        "frame": frame[["x", "c"]],
        "y": responses["poisson"],
        "prediction": np.asarray(wrapper.predict(frame[["x", "c"]]), dtype=np.float64),
    }
    with open(os.path.join(out, "sklearn_regressor.pkl"), "wb") as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for name, (response, declarations) in _lss_cases().items():
        model = SuperLSS(*declarations)
        model.fit_reml(frame, responses[response])
        record = {
            "version": sg.__version__,
            "serialized": model.to_bytes(),
            "frame": frame,
            "coefficients": dict(model.coef_),
            "eta": model.predict_link(frame).to_numpy(dtype=np.float64),
            "prediction": np.asarray(model.predict(frame), dtype=np.float64),
        }
        with open(os.path.join(out, f"{name}.pkl"), "wb") as handle:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(sys.argv[1])
