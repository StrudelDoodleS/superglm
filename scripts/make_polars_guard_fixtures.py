"""Write the retained-Polars-frame fixture of ``tests/test_polars_guard_versions.py``.

A retained fit keeps a fingerprint of its Polars frame (the fit-data guard) and
a pickled model carries it to whatever Polars loads it.  The record holds a
small Gaussian random-effect model fitted on a Polars frame, the frame's
``random_effects`` table as the fitting Polars computed it, and the Polars that
wrote it.  Run it under Polars 1.x, once with the release that wrote the
fingerprint with Polars' own row hash (commit 931548cd, ``legacy``) and once
with the release that hashes the values itself (``values``):

    PYTHONPATH=<old>/src uv run --with polars==1.43.2 --with polars-runtime-32==1.43.2 \
        python scripts/make_polars_guard_fixtures.py tests/fixtures/saved_polars_guard legacy
    uv run --with polars==1.43.2 --with polars-runtime-32==1.43.2 \
        python scripts/make_polars_guard_fixtures.py tests/fixtures/saved_polars_guard values
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import polars as pl

import superglm as sg


def record() -> dict:
    rng = np.random.default_rng(518)
    n = 60
    levels = np.array(["a", "b", "c"])[np.arange(n) % 3]
    shift = {"a": 0.8, "b": -0.2, "c": -0.6}
    y = np.array([shift[v] for v in levels]) + rng.normal(0.0, 0.5, n)
    frame = pl.DataFrame({"group": levels})
    model = sg.SuperGLM(family="gaussian", features={"group": sg.RandomEffect()})
    model.fit_reml(frame, y)
    table = model.random_effects("group").table
    return {
        "polars": pl.__version__,
        "superglm_digest_scheme": getattr(model._fit_data_guard, "x_digest_scheme", None),
        "model": model,
        "table": table,
    }


def main() -> None:
    out, kind = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"re_retained_{kind}.pkl"), "wb") as handle:
        pickle.dump(record(), handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
