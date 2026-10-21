"""The first retained-fit inference reads the data Gram the fit formed; it does not form it again.

The fit decides the data rank from the centred data Gram at its final weights.
``fit_inference_info`` needs the same Gram (the EDF correction and the
factor of the test statistics), and master built it a second time from the
design: 2.0 s of the first ``summary()`` of the 10-pair freMTPL2 model, and a
share of the inference of a ``retain_fit_state=False`` fit, which runs it
before the fit returns.
"""

import numpy as np
import pandas as pd
import pytest

import superglm.model.state_ops as state_ops
from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline

U = np.finfo(np.float64).eps / 2.0


def _fit(retain_fit_state=True, n=500, seed=5):
    rng = np.random.default_rng(seed)
    levels = np.array([f"L{j}" for j in range(5)])
    idx = rng.integers(0, len(levels), n)
    z = rng.uniform(0.0, 10.0, n)
    frame = pd.DataFrame({"band": levels[idx], "z": z})
    y = rng.poisson(np.exp(-0.3 + 0.1 * idx + 0.4 * np.sin(z))).astype(np.float64)
    model = SuperGLM(
        family="poisson",
        features={"band": Categorical(), "z": Spline(kind="ps", k=6)},
        retain_fit_state=retain_fit_state,
    )
    model.fit_reml(frame, y)
    return model


@pytest.fixture
def gram_builds(monkeypatch):
    """Count the data Grams ``fit_inference_info`` forms from the design."""
    calls = []
    build = state_ops._rank_centered_data_gram

    def counted(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(state_ops, "_rank_centered_data_gram", counted)
    return calls


def test_first_inference_takes_the_fits_gram_and_leaves_none(gram_builds):
    model = _fit()
    rank_info = model.result.rank_info
    assert gram_builds == []
    assert rank_info.data_gram is not None
    model.summary()
    assert gram_builds == []
    assert rank_info.data_gram is None


def test_released_fit_never_forms_the_gram_again(gram_builds):
    model = _fit(retain_fit_state=False)
    # the release distilled the inference inside the fit, from the fit's Gram
    assert gram_builds == []
    assert model.result.rank_info.data_gram is None
    assert "_fit_inference_info" in model.__dict__


def test_the_fits_gram_gives_the_inference_the_design_gives():
    """Same inference as forming the Gram from the design at the summary's weights.

    The two Grams are sums of the same n weighted rows, so each entry agrees to
    ``2 gamma_n`` of the largest diagonal (Higham, section 3.1); the quantities
    built from them are compared at a bound of that size times the Gram's
    condition-free scale.
    """
    model = _fit()
    rank_info = model.result.rank_info
    fits_gram = rank_info.data_gram
    taken = model._fit_inference_info
    model.__dict__.pop("_fit_inference_info")
    assert rank_info.data_gram is None
    formed = model._fit_inference_info
    n = len(model._fit_weights)
    gamma_n = n * U / (1.0 - n * U)
    bound = 2.0 * gamma_n * float(np.max(np.diag(fits_gram)))
    p = fits_gram.shape[0]
    atol = bound * p * float(np.max(np.abs(taken["XtWX_inv"])))
    np.testing.assert_allclose(
        fits_gram,
        state_ops._rank_centered_data_gram(
            state_ops._rank_active_state(model, rank_info)[0],
            state_ops._solver_space_working_weights(model),
        ),
        rtol=0.0,
        atol=bound,
    )
    for key in ("edf", "edf1"):
        np.testing.assert_allclose(taken[key], formed[key], rtol=0.0, atol=max(atol, 1e-12))
    assert taken["group_edf_map"].keys() == formed["group_edf_map"].keys()


def test_a_gram_for_other_weights_is_not_taken():
    model = _fit()
    rank_info = model.result.rank_info
    weights = state_ops._solver_space_working_weights(model)
    assert rank_info.take_data_gram(weights * (1.0 + 1e-3)) is None
    # taking clears it either way: a record is read once
    assert rank_info.data_gram is None

    model = _fit()
    rank_info = model.result.rank_info
    gram = rank_info.data_gram
    weights = state_ops._solver_space_working_weights(model)
    assert rank_info.take_data_gram(weights) is gram
    assert rank_info.data_gram is None
