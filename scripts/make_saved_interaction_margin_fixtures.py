"""Write the saved interaction-margin fixtures of ``tests/test_interactions.py`` with master 5f28e8e80.

Run it with master 5f28e8e80 on the path, the last master whose tensor and
spline-by-category margins were held at the training range under every
extrapolation policy, for example from ``git worktree add --detach <dir> 5f28e8e80``:

    PYTHONPATH=<dir>/src python scripts/make_saved_interaction_margin_fixtures.py \\
        tests/fixtures/saved_master_5f28e8e80

Each case writes ``<case>.pkl.xz``, an xz-compressed pickle of a record holding
the fitted model, its rows and response, and the predictions that master made on
those rows. Both cases are Poisson fits on the rating cells of
``tests/test_interactions.py`` (``_rating_cells``) under ``extrapolation="extend"``,
with ``boundary=(30, 65)`` on age inside the ages 25 to 75 of the data, a tensor
``ti("age", "bonus")`` and a region curve over age: a P-spline (``ps``) and a
cubic regression spline (``cr``, whose interaction margins are cardinal). The
rows past the boundary were fitted on interaction margins held at it.
"""

from __future__ import annotations

import lzma
import os
import pickle
import sys

import numpy as np
import pandas as pd

import superglm as sg


def rating_cells() -> tuple[pd.DataFrame, np.ndarray]:
    """``tests/test_interactions.py::_rating_cells``."""
    age, bonus, region = (
        grid.ravel()
        for grid in np.meshgrid(
            np.linspace(25, 75, 21), np.linspace(50, 100, 11), ["A", "B"], indexing="ij"
        )
    )
    a, b = (age - 50) / 25, (bonus - 75) / 25
    eta = -2.2 + 0.35 * a + 0.14 * b + 0.51 * a * b + (region == "B") * (0.08 + 0.37 * a)
    return pd.DataFrame({"age": age, "bonus": bonus, "region": region}), np.exp(eta)


def model(kind: str) -> sg.SuperGLM:
    """``tests/test_interactions.py::_rating_model(kind, None, "extend", boundary=(30, 65))``."""
    return sg.SuperGLM(
        family="poisson",
        terms=[
            sg.s("age", kind=kind, n_knots=3, extrapolation="extend", boundary=(30.0, 65.0)),
            sg.s("bonus", kind=kind, n_knots=3, extrapolation="extend"),
            sg.cat("region", base="A"),
            sg.ti("age", "bonus"),
            sg.interaction(sg.SplineCategorical("age", "region")),
        ],
        selection_penalty=0.0,
        spline_penalty=0.03,
        n_jobs=1,
    )


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    frame, y = rating_cells()
    for kind in ("ps", "cr"):
        fitted = model(kind).fit(frame, y)
        record = {
            "version": sg.__version__,
            "model": fitted,
            "fit": "fit",
            "frame": frame,
            "y": y,
            "prediction": np.asarray(fitted.predict(frame), dtype=np.float64),
        }
        with lzma.open(os.path.join(out, f"interactions_past_boundary_{kind}.pkl.xz"), "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(sys.argv[1])
