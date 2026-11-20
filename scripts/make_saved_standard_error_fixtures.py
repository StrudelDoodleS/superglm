"""Write the saved-model fixtures of ``tests/test_penalized_standard_errors.py`` with master bce2910e5.

Run it with master bce2910e5 on the path, the last master that blanked every standard error the
data alone do not determine, for example from ``git worktree add --detach <dir> bce2910e5``:

    PYTHONPATH=<dir>/src python scripts/make_saved_standard_error_fixtures.py \\
        tests/fixtures/saved_master_bce2910e5

Each case writes ``<case>.pkl.xz``, an xz-compressed pickle of a record holding the fitted model,
its rows and response, and what that master reported on those rows: the predictions,
``metrics().coefficient_se`` over the model's groups, and the names of the summary rows it printed
without a standard error.  Two Poisson fits, each penalised to full rank where the data alone are
not:

- ``dense``: a ``RandomEffect`` with a level that has no claims, on the dense solver;
- ``structured``: an ``fs`` factor smooth with two such levels, on the structured solver.

Each is saved three ways: ``cold`` (pickled before any reader ran), ``warm`` (after ``summary()``
and ``metrics()`` cached their inference) and ``compact`` (``retain_fit_state=False``).
"""

from __future__ import annotations

import lzma
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import superglm as sg


def dense_case():
    rng = np.random.default_rng(3)
    n = 1200
    frame = pd.DataFrame(
        {
            "age": rng.uniform(18, 80, n),
            "veh": rng.choice(list("ABCDEF"), n, p=[0.3, 0.25, 0.2, 0.15, 0.08, 0.02]),
            "dens": rng.normal(size=n),
        }
    )
    eta = -1.0 + 0.3 * np.sin(frame["age"] / 10) + 0.2 * frame["dens"]
    y = rng.poisson(np.exp(eta)).astype(float)
    y[(frame["veh"] == "F").to_numpy()] = 0.0
    features = {"age": sg.Spline(n_knots=6), "veh": sg.RandomEffect(), "dens": sg.Numeric()}
    return frame, y, lambda **options: sg.SuperGLM(family="poisson", features=features, **options)


def structured_case():
    rng = np.random.default_rng(2)
    levels, n = 20, 500
    g = rng.integers(0, levels, n)
    x = rng.uniform(size=n)
    frame = pd.DataFrame({"x": x, "z": rng.normal(size=n), "g": [f"g{v:02d}" for v in g]})
    y = rng.poisson(np.exp(-0.5 + 0.5 * np.sin(3 * x) + 0.2 * np.sin(5 * x + g % 7))).astype(float)
    y[g < 2] = 0.0

    def model(**options):
        return sg.SuperGLM(
            family="poisson",
            features={"z": sg.Numeric(), "x": sg.Spline(kind="ps", n_knots=6)},
            interactions=[sg.FactorSmooth("x", group="g", basis="fs", k=6)],
            selection_penalty=0,
            direct_solve="structured",
            **options,
        )

    return frame, y, model


def reported(model, frame, y) -> dict:
    """What this build reports on the training rows; reading it caches the model's inference."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        se = model.metrics(frame, y).coefficient_se
        rows = model.summary()._coef_rows
    return {
        "prediction": np.asarray(model.predict(frame), dtype=np.float64),
        "metrics_se": np.concatenate(
            [np.atleast_1d(np.asarray(se[g.name], dtype=np.float64)) for g in model._groups]
        ),
        "summary_blank_rows": [
            row.name
            for row in rows
            if not row.is_spline
            and row.structured_kind is None
            and row.name != "Intercept"
            and (row.se is None or row.se != row.se)
        ],
    }


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    for name, case in (("dense", dense_case), ("structured", structured_case)):
        frame, y, build = case()
        for state in ("cold", "warm", "compact"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = build(retain_fit_state=state != "compact").fit_reml(frame, y)
            cold = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            record = {
                "version": sg.__version__,
                "commit": "bce2910e5",
                "state": state,
                "frame": frame,
                "y": y,
                **reported(model, frame, y),
            }
            record["model"] = pickle.loads(cold) if state == "cold" else model
            with lzma.open(os.path.join(out, f"{name}_{state}.pkl.xz"), "wb") as handle:
                pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
            blank = int(np.isnan(record["metrics_se"]).sum())
            print(f"{name}_{state}: {blank} of {record['metrics_se'].size} standard errors blank")


if __name__ == "__main__":
    main(sys.argv[1])
