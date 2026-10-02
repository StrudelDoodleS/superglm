"""Write the edited-model fixtures of ``tests/test_coefficient_revisions.py`` with an older superglm.

The editor of v0.35.0 and of master up to 31544462 (before #448) left a
model's solver state at its pre-edit coefficients, so a post-fit shape repair
of the saved edit profiled the wrong level (#447).  These fixtures are that
saved state.  Run the script with the release or commit on the path, for
example from ``git worktree add <dir> v0.35.0``:

    PYTHONPATH=<dir>/src python scripts/make_saved_editor_fixtures.py \\
        tests/fixtures/saved_v0_35_0 v0.35.0
    PYTHONPATH=<dir>/src python scripts/make_saved_editor_fixtures.py \\
        tests/fixtures/saved_31544462 31544462

The record holds the edited model (the issue's fixture: a P-spline with a
post-fit increasing constraint, its effect halved in the editor), its training
rows and weights, the source it was written with and its predictions.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import superglm
from superglm import Constraint, PSpline, SuperGLM
from superglm.editor import EditorSession


def main(out: str, source: str) -> None:
    os.makedirs(out, exist_ok=True)
    x = np.linspace(0.0, 1.0, 60)
    y = 1.5 - 1.1 * x + 0.08 * np.sin(7.0 * x)
    w = np.resize(np.array([1.0, 3.0, 2.0, 4.0]), x.size)
    frame = pd.DataFrame({"x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            spline_penalty=0.8,
            weight_semantics="frequency",
            features={
                "x": PSpline(
                    n_knots=6, knot_strategy="uniform", constraint=Constraint.postfit.increasing
                )
            },
        ).fit(frame, y, sample_weight=w)
        session = EditorSession.from_model(model, terms=["x"], train_data=(frame, y, w))
        session.terms["x"].edited_log_effect[:] = 0.5 * session.terms["x"].edited_log_effect
        edited = session.to_model()
    record = {
        "version": superglm.__version__,
        "source": source,
        "model": edited,
        "frame": frame,
        "y": y,
        "sample_weight": w,
        "prediction": np.asarray(edited.predict(frame)),
    }
    with open(os.path.join(out, "editor_halved_spline.pkl"), "wb") as handle:
        pickle.dump(record, handle, protocol=5)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
