"""A fitted model's size in pickle does not grow with the rows its terms were fitted on.

A dense tensor interaction kept each margin's (n, K) training-row basis on its
spec: 280 MB of the 10-pair freMTPL2 model, 465 MB pickled even with
``retain_fit_state=False``. Scoring evaluates new rows from the margin's
``raw_basis_eval`` and ``projection``, so the bases are not needed after the
build. Master's spec pickle grew linearly in n.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, Spline, SuperGLM


def _fit(n, retain_fit_state):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
    y = rng.poisson(np.exp(0.3 * np.sin(6 * frame["a"]) + frame["b"])).astype(np.float64)
    model = SuperGLM(
        family="poisson",
        features={"a": Spline(n_knots=5), "b": Spline(n_knots=5), "c": Numeric()},
        interactions=[("a", "b")],
        retain_fit_state=retain_fit_state,
    )
    frame["c"] = rng.normal(size=n)
    model.fit_reml(frame, y)
    return model, frame


@pytest.mark.parametrize("retain_fit_state", [True, False])
def test_tensor_spec_pickle_does_not_grow_with_rows(retain_fit_state):
    small, _ = _fit(600, retain_fit_state)
    large, frame = _fit(2400, retain_fit_state)
    sizes = []
    for model in (small, large):
        (spec,) = model._interaction_specs.values()
        assert spec._marginal1.basis.shape[0] == 0
        sizes.append(len(pickle.dumps(spec)))
    # the knots, projections and penalties are fixed by the basis, not the rows
    assert sizes[1] <= sizes[0] + 1024
    # scoring new rows still goes through the margins' own evaluators
    assert np.all(np.isfinite(large.predict(frame.iloc[:50])))
