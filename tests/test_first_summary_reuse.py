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
from superglm import Constraint, SuperGLM
from superglm.features import Numeric, PSpline
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline

U = np.finfo(np.float64).eps / 2.0


def _data(n=500, seed=5):
    rng = np.random.default_rng(seed)
    levels = np.array([f"L{j}" for j in range(5)])
    idx = rng.integers(0, len(levels), n)
    z = rng.uniform(0.0, 10.0, n)
    frame = pd.DataFrame({"band": levels[idx], "z": z})
    y = rng.poisson(np.exp(-0.3 + 0.1 * idx + 0.4 * np.sin(z))).astype(np.float64)
    return frame, y


def _fit(retain_fit_state=True, n=500, seed=5):
    frame, y = _data(n, seed)
    model = SuperGLM(
        family="poisson",
        features={"band": Categorical(), "z": Spline(kind="ps", k=6)},
        retain_fit_state=retain_fit_state,
    )
    model.fit_reml(frame, y)
    return model


def _scop_fit(retain_fit_state=True, n=400, seed=1):
    """A fixed-lambda fit with a SCOP monotone term."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n))
    z = rng.normal(size=n)
    frame = pd.DataFrame({"z": z, "x": x})
    y = 0.3 + 0.7 * z + 1.2 * x + rng.normal(scale=0.1, size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "z": Numeric(),
            "x": PSpline(n_knots=7, constraint=Constraint.fit.increasing, shape_engine="scop"),
        },
        retain_fit_state=retain_fit_state,
    )
    model.fit(frame, y)
    return model, frame, y


def _weighted_gaussian_fit(n=300, seed=7):
    """Integer frequency weights: the Gaussian Fisher weights are the weights themselves."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    frame = pd.DataFrame({"x": x, "z": rng.normal(size=n)})
    y = np.sin(3.0 * x) + 0.5 * frame["z"] + rng.normal(scale=0.2, size=n)
    weights = rng.integers(1, 6, n).astype(np.float64)
    model = SuperGLM(
        family="gaussian",
        features={"x": Spline(kind="ps", k=6), "z": Numeric()},
    )
    model.fit_reml(frame, y, sample_weight=weights)
    return model


def _wood_rows(model, frame, y):
    """The summary's term-test rows (Wood statistic, reference df, p-value)."""
    rows = model.metrics(frame, y)._build_coef_rows()
    return {r.name: (r.wald_chi2, r.ref_df, r.wald_p) for r in rows if r.is_spline}


def _without_gram(model):
    """The same model with the fit's Gram dropped: its inference forms its own."""
    object.__setattr__(model.result.rank_info, "data_gram", None)
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


def _gram_bound(model, gram):
    """``2 gamma_n max(diag)``: the entrywise difference of two sums of the same n weighted rows."""
    n = len(model._fit_weights)
    gamma_n = n * U / (1.0 - n * U)
    return 2.0 * gamma_n * float(np.max(np.diag(gram)))


def test_the_fits_gram_gives_the_inference_the_design_gives():
    """Same inference as forming the Gram from the design at the summary's weights.

    The two Grams are sums of the same n weighted rows, so each entry agrees to
    ``2 gamma_n`` of the largest diagonal (Higham, section 3.1). ``R_a`` is
    compared through its Gram ``R_a' R_a``, which does not depend on the
    eigenvector basis, and ``edf1`` at that bound scaled by the inverse.
    """
    frame, y = _data()
    model = _fit()
    rank_info = model.result.rank_info
    fits_gram = rank_info.data_gram
    taken = model._fit_inference_info
    assert rank_info.data_gram is None
    model.__dict__.pop("_fit_inference_info")
    formed = model._fit_inference_info
    formed_gram = state_ops._rank_centered_data_gram(
        state_ops._rank_active_state(model, rank_info)[0],
        state_ops._solver_space_working_weights(model),
    )
    bound = _gram_bound(model, fits_gram)
    np.testing.assert_allclose(fits_gram, formed_gram, rtol=0.0, atol=bound)
    np.testing.assert_allclose(taken["R_a"].T @ taken["R_a"], formed_gram, rtol=0.0, atol=bound)
    atol = bound * fits_gram.shape[0] * float(np.max(np.abs(taken["XtWX_inv"])))
    np.testing.assert_allclose(taken["edf1"], formed["edf1"], rtol=0.0, atol=max(atol, 1e-12))
    assert taken["group_edf_map"].keys() == formed["group_edf_map"].keys()
    # the user-visible output the Gram feeds: Wood statistics, reference df, p-values
    reference = _wood_rows(_without_gram(_fit()), frame, y)
    rows = _wood_rows(_fit(), frame, y)
    assert rows.keys() == reference.keys() and rows
    for name, values in rows.items():
        np.testing.assert_allclose(values, reference[name], rtol=1e-8, atol=1e-12)


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


def test_weights_of_the_same_sum_do_not_take_the_gram(gram_builds):
    """Reordered integer weights sum to the same float, and make a different Gram."""
    model = _weighted_gaussian_fit()
    rank_info = model.result.rank_info
    weights = state_ops._solver_space_working_weights(model)
    permuted = np.ascontiguousarray(weights[::-1])
    assert float(np.sum(permuted)) == float(np.sum(weights))
    assert not np.array_equal(permuted, weights)
    # an editor copy reading reweighted rows: the inference follows the new weights
    model._fit_weights = np.ascontiguousarray(model._fit_weights[::-1])
    info = state_ops.fit_inference_info(model)
    assert gram_builds == [1]
    fresh = state_ops._rank_centered_data_gram(
        state_ops._rank_active_state(model, rank_info)[0], permuted
    )
    scale = float(np.max(np.diag(fresh)))
    np.testing.assert_allclose(
        info["R_a"].T @ info["R_a"], fresh, rtol=0.0, atol=_gram_bound(model, fresh) + 1e-14 * scale
    )
    assert rank_info.data_gram is None


def test_scop_fit_releases_the_gram_at_its_first_inference(gram_builds):
    model, frame, y = _scop_fit()
    rank_info = model._solver_result.rank_info
    assert model._solver_result.scop_inference is not None
    assert rank_info.data_gram is not None
    fits_gram = rank_info.data_gram
    taken = model._fit_inference_info  # the cached record the summary's rows read
    assert gram_builds == []
    assert rank_info.data_gram is None
    formed = state_ops._rank_centered_data_gram(
        state_ops._rank_active_state(model, rank_info)[0],
        state_ops._solver_space_working_weights(model),
    )
    bound = _gram_bound(model, fits_gram)
    np.testing.assert_allclose(taken["R_a"].T @ taken["R_a"], formed, rtol=0.0, atol=bound)
    reference_model, _, _ = _scop_fit()
    builds = len(gram_builds)  # the reference Gram above is the test's own build
    rows = _wood_rows(model, frame, y)
    assert len(gram_builds) == builds  # the rows used the fit's Gram, not a rebuilt one
    reference = _wood_rows(_without_gram(reference_model), frame, y)
    assert rows.keys() == reference.keys() and rows
    for name, values in rows.items():
        np.testing.assert_allclose(values, reference[name], rtol=1e-8, atol=1e-12)


def test_released_scop_fit_holds_no_gram():
    model, _, _ = _scop_fit(retain_fit_state=False)
    assert model._solver_result.rank_info.data_gram is None
