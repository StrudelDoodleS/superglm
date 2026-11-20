"""Write ``tests/fixtures/saved_master_a4a7a64cd`` with master ``a4a7a64cd``, the base of the SE fix.

Run it with that commit on the path, from a detached worktree
(``git worktree add --detach <dir> a4a7a64cd``):

    PYTHONPATH=<dir>/src python scripts/make_saved_master_a4a7a64cd_fixtures.py \
        tests/fixtures/saved_master_a4a7a64cd

Each case is one ``<case>.pkl.xz`` record: the rows and response it was fitted on, a frame
of rows its fit cannot determine (``off``), and four pickles of the fitted model, each with
what master read from it after loading (``outputs``):

- ``warm_cold`` and ``warm_read``: ``retain_fit_state=True``, pickled before and after
  ``summary()`` and ``metrics()`` ran;
- ``compact_cold`` and ``compact_read``: ``retain_fit_state=False``, the same.

The outputs are the published coefficients (``intercept`` and ``beta``, bit for bit), the
predictions on both frames, the effective degrees of freedom, every ``summary()`` row's name
and standard error (NaN where blank) and ``metrics(X, y)``'s fitted-family and Pearson SEs.
The cases are those the SE fix changes on a saved model: a full-rank dense fit (the control),
an exactly duplicated numeric column, a Poisson factor whose reference level has every
response zero (separated), a Poisson near-duplicate, an ``sz`` factor smooth with one-row
levels, and a random effect with one-row levels.
"""

from __future__ import annotations

import lzma
import os
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

import superglm as sg
from superglm import Categorical, FactorSmooth, Numeric, RandomEffect, Spline

COMMIT = "a4a7a64cd"


def _dense_full():
    rng = np.random.default_rng(3)
    n = 300
    X = pd.DataFrame(
        {"x": rng.uniform(size=n), "z": rng.normal(size=n), "f": rng.choice(list("ABC"), n)}
    )
    eta = -0.3 + np.sin(2 * np.pi * X["x"]) + 0.2 * X["z"] + 0.4 * (X["f"] == "B")
    y = rng.poisson(np.exp(eta)).astype(float)
    off = pd.DataFrame({"x": [0.1, 0.5, 0.9], "z": [-3.0, 0.0, 3.0], "f": ["A", "B", "C"]})
    features = {"x": Spline(n_knots=6), "z": Numeric(), "f": Categorical()}
    return dict(family="poisson", features=features), "fit_reml", X, y, off


def _duplicate():
    """``x2 = x1`` exactly: one direction of ``[x1, x2]`` is aliased, master zeroed ``x2``."""
    rng = np.random.default_rng(4)
    n = 300
    x1 = rng.normal(size=n)
    X = pd.DataFrame({"x": rng.uniform(size=n), "x1": x1, "x2": x1.copy()})
    y = rng.poisson(np.exp(-0.2 + np.sin(2 * np.pi * X["x"]) + 0.3 * x1)).astype(float)
    off = pd.DataFrame({"x": [0.2, 0.6], "x1": [1.0, -1.0], "x2": [-1.0, 2.0]})
    features = {"x": Spline(n_knots=6), "x1": Numeric(), "x2": Numeric()}
    return dict(family="poisson", features=features), "fit_reml", X, y, off


def _reference_level():
    """The design's case C3 (``test_penalised_existence``) at 600 rows: reference level E all zero."""
    rng = np.random.default_rng(1)
    n = 600
    f = rng.choice(list("ABCDE"), n)
    x = rng.uniform(size=n)
    y = rng.poisson(np.exp(-1 + 0.5 * (f == "B") + np.sin(2 * np.pi * x))).astype(float)
    y[f == "E"] = 0.0
    X = pd.DataFrame({"f": f, "x": x})
    off = pd.DataFrame({"f": ["E", "A"], "x": [0.3, 0.3]})
    features = {"f": Categorical(base="E"), "x": Spline(n_knots=8)}
    return dict(family="poisson", features=features), "fit_reml", X, y, off


def _near_duplicate():
    """``x2 = x1 + 1e-8 h`` under Poisson (``test_penalised_existence``), at 600 rows."""
    rng = np.random.default_rng(1)
    n = 600
    z, x1, h = rng.uniform(size=n), rng.normal(size=n), rng.normal(size=n)
    y = rng.poisson(np.exp(-0.5 + 0.3 * x1 + np.sin(2 * np.pi * z))).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x1 + 1e-8 * h, "z": z})
    off = pd.DataFrame({"x1": [0.5, 0.5], "x2": [0.5, 1.5], "z": [0.5, 0.5]})
    features = {"x1": Numeric(), "x2": Numeric(), "z": Spline(n_knots=8)}
    return dict(family="poisson", features=features), "fit", X, y, off


def _sz_one_row():
    """An ``sz`` factor smooth over 12 levels, three of them one row each."""
    rng = np.random.default_rng(1)
    K, n = 12, 360
    g = rng.integers(3, K, n)
    g[:3] = np.arange(3)
    x = rng.uniform(size=n)
    X = pd.DataFrame({"x": x, "z": rng.normal(size=n), "g": [f"g{v:02d}" for v in g]})
    y = np.sin(3 * x) + 0.3 * np.sin(5 * x + g % 7) + 0.1 * X["z"] + rng.normal(0, 0.3, n)
    off = pd.DataFrame({"x": [0.05, 0.5, 0.95], "z": [0.0, 0.0, 0.0], "g": ["g00", "g01", "g05"]})
    features = {"z": Numeric(), "x": Spline(kind="ps", n_knots=6)}
    interactions = [FactorSmooth("x", group="g", basis="sz", k=6)]
    return (
        dict(family="gaussian", features=features, interactions=interactions),
        "fit_reml",
        X,
        np.asarray(y, dtype=float),
        off,
    )


def _random_effect():
    """A Poisson random effect over 15 levels, two of them one row each."""
    rng = np.random.default_rng(6)
    K, n = 15, 300
    g = rng.integers(2, K, n)
    g[:2] = np.arange(2)
    x = rng.uniform(size=n)
    effect = rng.normal(0, 0.3, K)
    X = pd.DataFrame({"x": x, "g": [f"g{v:02d}" for v in g]})
    y = rng.poisson(np.exp(-0.2 + np.sin(2 * np.pi * x) + effect[g])).astype(float)
    off = pd.DataFrame({"x": [0.5, 0.5, 0.5], "g": ["g00", "g07", "unseen"]})
    features = {"x": Spline(n_knots=6), "g": RandomEffect()}
    return dict(family="poisson", features=features), "fit_reml", X, y, off


CASES = {
    "dense_full": _dense_full,
    "duplicate": _duplicate,
    "reference_level": _reference_level,
    "near_duplicate": _near_duplicate,
    "sz_one_row": _sz_one_row,
    "random_effect": _random_effect,
}


def _readers(model, X, y) -> None:
    model.summary()
    model.metrics(X, y)


def _outputs(model, X, y, off) -> dict:
    """What master reads from a loaded model: coefficients, predictions and both SE readers."""
    rows = model.summary()._coef_rows
    metrics = model.metrics(X, y)
    return {
        "intercept": float(model.result.intercept),
        "beta": np.asarray(model.result.beta, dtype=np.float64).copy(),
        "effective_df": float(model.result.effective_df),
        "predict": np.asarray(model.predict(X), dtype=np.float64),
        "predict_off": np.asarray(model.predict(off), dtype=np.float64),
        "summary_names": [str(row.name) for row in rows],
        "summary_se": np.array([np.nan if row.se is None else float(row.se) for row in rows]),
        "summary_spline": np.array([bool(row.is_spline) for row in rows]),
        "metrics_se": {k: np.asarray(v, dtype=float) for k, v in metrics.coefficient_se.items()},
        "metrics_se_raw": {
            k: np.asarray(v, dtype=float) for k, v in metrics.coefficient_se_raw.items()
        },
        "metrics_intercept_se": float(metrics.intercept_se),
        "metrics_intercept_se_raw": float(metrics.intercept_se_raw),
    }


def main(out: str) -> None:
    head = subprocess.run(
        ["git", "-C", os.path.dirname(sg.__file__), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head.startswith(COMMIT), head
    os.makedirs(out, exist_ok=True)
    warnings.simplefilter("ignore")
    for name, make in CASES.items():
        kwargs, method, X, y, off = make()
        record = {"commit": COMMIT, "version": sg.__version__, "method": method}
        record.update(frame=X, y=y, off=off, models={}, outputs={})
        for retain in (True, False):
            model = sg.SuperGLM(selection_penalty=0.0, retain_fit_state=retain, **kwargs)
            getattr(model, method)(X, y)
            prefix = "warm" if retain else "compact"
            cold = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            _readers(model, X, y)
            read = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            for suffix, blob in (("cold", cold), ("read", read)):
                variant = f"{prefix}_{suffix}"
                record["models"][variant] = blob
                record["outputs"][variant] = _outputs(pickle.loads(blob), X, y, off)
        with lzma.open(os.path.join(out, f"{name}.pkl.xz"), "wb") as handle:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(name, os.path.getsize(os.path.join(out, f"{name}.pkl.xz")))


if __name__ == "__main__":
    main(sys.argv[1])
