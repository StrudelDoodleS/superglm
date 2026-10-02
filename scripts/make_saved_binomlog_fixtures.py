"""Write the saved-model fixtures of ``tests/test_saved_binomlog_models.py`` with superglm v0.36.0.

Run it with the v0.36.0 release on the path (commit f8e5ac01), for example from
``git archive v0.36.0 src | tar -x -C <dir>``:

    PYTHONPATH=<dir>/src python scripts/make_saved_binomlog_fixtures.py tests/fixtures/saved_v0_36_0

Each record holds the fitted model, its training rows, offsets and weights, its
predictions (without the offset, as ``tests/test_saved_fs_models.py``'s
``assert_predicts_as_saved`` reads them), and the rows of the ``TruncatedDirection`` records v0.36.0
published: a binomial/log fit whose factorization truncates a light cut
(``tests/test_binomial_log_mean_space.py::_light_cut``), at its own maximum
(weakly identified) and away from it (unresolved).
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import superglm
from superglm import Categorical, SuperGLM

LINKS = [("a1", "b2"), ("a1", "b3")]
CASES = {
    "binomlog_light_cut_at_maximum": 1.3,
    "binomlog_light_cut_unresolved": -20.0,
}


def light_cut(light_offset: float) -> pd.DataFrame:
    heavy = [(a, b) for a in ("a0", "a1") for b in ("b0", "b1")]
    heavy += [(a, b) for a in ("a2", "a3") for b in ("b2", "b3")]
    rows = [(a, b, response, 1e8, 1.3) for a, b in heavy for response in (0.0, 1.0)]
    rows += [(a, b, response, 1e-8, light_offset) for a, b in LINKS for response in (0.0, 1.0)]
    return pd.DataFrame(rows, columns=["A", "B", "y", "w", "off"])


def main(out: str) -> None:
    assert superglm.__version__ == "0.36.0", superglm.__version__
    os.makedirs(out, exist_ok=True)
    for name, light_offset in CASES.items():
        frame = light_cut(light_offset)
        model = SuperGLM(
            family="binomial",
            link="log",
            selection_penalty=0.0,
            direct_solve="gram",
            features={"A": Categorical(base="a0"), "B": Categorical(base="b0")},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                frame[["A", "B"]],
                frame["y"].to_numpy(),
                sample_weight=frame["w"].to_numpy(),
                offset=frame["off"].to_numpy(),
            )
        records = model.result.truncated_directions
        assert records, name
        record = {
            "model": model,
            "frame": frame,
            "prediction": np.asarray(model.predict(frame[["A", "B"]])),
            "truncated_rows": [list(r.rows) for r in records],
            "at_maximum": [bool(r.at_maximum) for r in records],
            "converged": bool(model.result.converged),
            "version": superglm.__version__,
        }
        with open(os.path.join(out, f"{name}.pkl"), "wb") as handle:
            pickle.dump(record, handle)
        print(name, record["converged"], record["truncated_rows"], record["at_maximum"])


if __name__ == "__main__":
    main(sys.argv[1])
