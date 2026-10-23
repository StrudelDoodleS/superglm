"""Write the saved-model fixtures of ``tests/test_userspace_040.py`` with superglm v0.40.0.

Run it with the v0.40.0 release on the path (commit 542f5826), for example from
``git worktree add <dir> v0.40.0``:

    PYTHONPATH=<dir>/src python scripts/make_saved_v040_fixtures.py tests/fixtures/saved_v0_40_0

Each ``SuperGLM`` or wrapper record holds the pickled model, the fit method,
its rows and response, and the predictions v0.40.0 made. The
models span the routes a saved book uses: exact and binned REML, a random
effect, an ``fs`` and an ``sz`` factor smooth, SCOP and QP monotone splines,
tensor and spline-by-category pairs, Tweedie, a penalised ``fit`` and the sklearn
wrapper.

Run with an older release instead (from ``git worktree add <dir> v0.17.0``, or
that release's own environment), it writes that saver's ``LEGACY_SAVERS``
records to ``tests/fixtures/saved_v0_17_0`` and so on: one small ``fit_reml``
model per era of terms that gained predict-time state after the saver.  The
0.41 audit found those terms' pickles failing at predict until
``__setstate__`` restored the state their saver scored with.  Their API is the
one these releases share, so ``data`` and the records stay identical in form.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import superglm as sg


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


def _glm_cases():
    from superglm import (
        Categorical,
        Constraint,
        FactorSmooth,
        Numeric,
        PSpline,
        RandomEffect,
        Spline,
    )

    mains = {
        "x": Spline(kind="cr", k=8),
        "z": Spline(kind="cr", k=6),
        "c": Categorical(),
        "lin": Numeric(),
    }

    return {
        "glm_poisson_exact": dict(family="poisson", features=mains, fit="fit_reml"),
        "glm_poisson_binned": dict(family="poisson", features=mains, discrete=True, fit="fit_reml"),
        "glm_gamma_re": dict(
            family="gamma", features={**mains, "g": RandomEffect()}, fit="fit_reml"
        ),
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
        "glm_tweedie": dict(family="tweedie", features=mains, fit="fit_reml"),
        "glm_gaussian_fit": dict(family="gaussian", features=mains, fit="fit"),
    }


# Saver -> the record it writes, one per class and era (rows 19-20 of the
# 0.41 release audit).  A spline-by-category term saved before 0.26 has no
# parent level universe or unseen policy, and before 0.18 no grouping either;
# an OrderedCategorical saved before 0.19 has no specials.
LEGACY_SAVERS = {
    "0.17.0": ("spline_cat",),
    "0.18.0": ("ordered_cat",),
    "0.25.0": ("spline_cat",),
}
BANDS = list("ABCDE")
LEGACY_ROWS = 60


def _legacy_case(name: str):
    """``(features, interactions)`` of one legacy record."""
    spline = sg.Spline(kind="cr", k=5)
    if name == "ordered_cat":
        return {"band": sg.OrderedCategorical(order=BANDS), "x": spline}, None
    return {"x": spline, "c": sg.Categorical()}, [("x", "c")]


def main_legacy(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    frame, responses = data(n=LEGACY_ROWS)
    frame["band"] = np.array(BANDS, dtype=object)[
        np.minimum(frame["x"].to_numpy() // 2, 4).astype(int)
    ]
    y = responses["poisson"]
    for name in LEGACY_SAVERS[sg.__version__]:
        features, interactions = _legacy_case(name)
        rows = frame[list(features)]
        model = sg.SuperGLM(family="poisson", features=features, interactions=interactions)
        model.fit_reml(rows, y)
        record = {
            "version": sg.__version__,
            "model": model,
            "fit": "fit_reml",
            "frame": rows,
            "y": y,
            "prediction": np.asarray(model.predict(rows), dtype=np.float64),
        }
        with open(os.path.join(out, f"{name}.pkl"), "wb") as handle:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(out: str) -> None:
    if sg.__version__ in LEGACY_SAVERS:
        return main_legacy(out)
    assert sg.__version__ == "0.40.0", sg.__version__
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    frame, responses = data()
    for name, case in _glm_cases().items():
        case = dict(case)
        fit = case.pop("fit")
        family = case.pop("family")
        if family == "tweedie":
            family = sg.families.tweedie(p=1.5)
        model = sg.SuperGLM(family=family, **case)
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
        family="poisson", features={"x": sg.Spline(kind="cr", k=8), "c": sg.Categorical()}
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


if __name__ == "__main__":
    main(sys.argv[1])
