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

from superglm import Categorical, Constraint, Numeric, PSpline, Spline, SuperGLM


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
    # scoring new rows goes through the margins' own evaluators: a round trip
    # of the model scores them as the model does, and a retained model's own
    # rows reproduce its fitted mean
    scored = large.predict(frame)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(large)).predict(frame), scored)
    if retain_fit_state:
        np.testing.assert_allclose(scored, large._fit_mu, rtol=1e-10)


def _scop_fit(retain_fit_state):
    """A fixed-lambda fit with a SCOP monotone term and a wide categorical."""
    rng = np.random.default_rng(4)
    n = 600
    x = np.sort(rng.uniform(0.0, 1.0, n))
    frame = pd.DataFrame({"x": x, "band": rng.integers(0, 40, n).astype(str)})
    y = 1.2 * x + 0.1 * rng.normal(size=n) + 0.01 * rng.integers(0, 40, n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": PSpline(n_knots=7, constraint=Constraint.fit.increasing),
            "band": Categorical(),
        },
        retain_fit_state=retain_fit_state,
    )
    model.fit(frame, y)
    return model


@pytest.mark.parametrize("retain_fit_state", [True, False])
def test_scop_fit_pickles_no_data_gram_after_its_first_inference(retain_fit_state):
    """The fit's Gram is 8 p^2 bytes the SCOP branch of the inference must take."""
    model = _scop_fit(retain_fit_state)
    assert model._solver_result.scop_inference is not None
    if retain_fit_state:
        model.summary()
    rank_info = model._solver_result.rank_info
    assert rank_info.data_gram is None
    size = len(pickle.dumps(model))
    p = len(rank_info.mean_x)
    # a held Gram (as the first hand-off commit left it on SCOP fits) would add 8 p^2 bytes
    object.__setattr__(rank_info, "data_gram", np.zeros((p, p)))
    assert len(pickle.dumps(model)) >= size + 8 * p * p
