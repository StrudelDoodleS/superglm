"""A model saved by an earlier build with a nested factor keeps its inference (design §3.12).

The stage-0 verifier found that models pickled by master 94359786 or by the
uncommitted speed build, whose ``NestedSchurFactor`` carries another set of
derived attributes, loaded and predicted but raised ``AttributeError``
(``_center_star``) in standard errors, leverage and ``summary()``.  A factor
restored from a foreign state keeps the inputs it saved -- the operator it
factored, at the saved coefficients and smoothing parameters -- and is rebuilt
by the current engine on first use, with one notice.  The foreign states here
are what an earlier build pickles: its inputs, an attribute of its own, and no
format marker.  (The real master and speed-build pickles were checked the same
way outside the suite: SE equal to 4.9e-15, summaries and leverage working.)
"""

from __future__ import annotations

import copy
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import RandomEffect, Spline, SuperGLM

_INPUTS = (
    "operator",
    "chain_group_names",
    "chain_group_indices",
    "intercept",
    "max_structured_inverse_block",
)


def _model():
    rng = np.random.default_rng(0)
    n, K, H = 3000, 60, 25
    frame = pd.DataFrame(
        {
            "x": rng.uniform(size=n),
            "g": [f"g{c:02d}" for c in rng.integers(0, K, n)],
            "h": [f"h{c:02d}" for c in rng.integers(0, H, n)],
        }
    )
    signal = (
        0.3 * np.sin(4 * frame["x"].to_numpy())
        + rng.normal(0, 0.3, K)[frame["g"].str[1:].astype(int).to_numpy()]
    )
    y = rng.poisson(np.exp(signal)).astype(float)
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(kind="ps", k=6), "g": RandomEffect(), "h": RandomEffect()},
        selection_penalty=0,
        direct_solve="structured",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    return model, frame, y


def _observed(model, frame, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = model.metrics(frame, y)
        se = {name: np.asarray(values) for name, values in metrics.coefficient_se.items()}
        leverage = np.asarray(metrics.leverage)
        text = str(model.summary())
    return se, leverage, text


def _se_bound(model) -> float:
    """``2 p eps kappa_s``: two backward-stable solves with the retained pivots (Higham 2002, §14.1)."""
    factor = model._linear_system_state.augmented_factor
    retained = factor.scaled_schur_eigenvalues()
    retained = retained[retained > 0.0]
    p = factor.shape[0]
    return 2.0 * p * np.finfo(np.float64).eps * float(retained.max() / retained.min())


def _loaded_from_an_earlier_build(model):
    """``model`` as an earlier build pickles it, loaded: its inputs, an attribute of its own, no format marker."""
    foreign = pickle.loads(pickle.dumps(model))
    state = foreign._linear_system_state
    augmented, profiled = state.augmented_factor, state.profiled_factor
    old_augmented = {name: augmented.__dict__[name] for name in _INPUTS}
    old_augmented["_Q_scaled"] = np.eye(2)  # an attribute only the earlier build had
    old_profiled = {name: profiled.__dict__[name] for name in ("sum_w", "xtw", "data_operator")} | {
        "augmented_factor": augmented
    }
    augmented.__dict__.clear()
    augmented.__dict__.update(old_augmented)
    profiled.__dict__.clear()
    profiled.__dict__.update(old_profiled)
    return pickle.loads(pickle.dumps(foreign))


def test_a_nested_factor_saved_by_an_earlier_build_is_rebuilt_on_first_use():
    """T8 row "retired-class shim removed": fails with the nested factors' ``__setstate__``
    and ``__getattr__`` removed (``AttributeError`` in SE, leverage and ``summary``).

    The rebuilt factor factors the saved operator by the same code, so the
    standard errors, and the leverage (a quadratic form in the same ``H^+``),
    agree within the rounding of two evaluations from the same factor:
    the saved model had some of it cached from the fit, the rebuilt one forms
    it on first use, and even one model's first and second SE reads differ in
    the last bit.  Both are backward-stable solves with the retained pivots,
    so they agree to ``2 p eps kappa_s`` relative (Higham 2002, section 14.1),
    ``kappa_s`` the retained scaled border's condition.
    """
    model, frame, y = _model()
    assert type(model._linear_system_state.augmented_factor).__name__ == "NestedSchurFactor"
    se, leverage, _ = _observed(model, frame, y)
    loaded = _loaded_from_an_earlier_build(model)
    np.testing.assert_array_equal(loaded.predict(frame), model.predict(frame))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # the first read of the inference state rebuilds both factors
        rank = model._linear_system_state.profiled_factor.rank
        assert loaded._linear_system_state.profiled_factor.rank == rank
    notices = [w for w in caught if "saved by an earlier superglm build" in str(w.message)]
    assert len(notices) == 1
    loaded_se, loaded_leverage, text = _observed(loaded, frame, y)
    assert text
    bound = _se_bound(model)
    for name, values in se.items():
        np.testing.assert_array_equal(np.isnan(loaded_se[name]), np.isnan(values))
        finite = ~np.isnan(values)
        np.testing.assert_allclose(loaded_se[name][finite], values[finite], rtol=bound, atol=0.0)
    np.testing.assert_allclose(loaded_leverage, leverage, rtol=bound, atol=0.0)


def test_a_model_saved_by_an_earlier_build_saves_and_copies_again_before_first_use():
    """Re-saving or copying a loaded model before its first inference call keeps the foreign state.

    Fails without the nested factors' ``__getstate__``: the copy saves the
    pending wrapper itself, the next load wraps it again, and the first
    inference call raises ``KeyError: 'augmented_factor'``.  Each copy is then
    rebuilt once, with the notice, to the standard errors and summary of the
    model it was copied from.
    """
    model, frame, y = _model()
    se, leverage, _ = _observed(model, frame, y)
    rank = model._linear_system_state.profiled_factor.rank
    bound = _se_bound(model)
    loaded = _loaded_from_an_earlier_build(model)
    copies = {"saved again": pickle.loads(pickle.dumps(loaded)), "deepcopy": copy.deepcopy(loaded)}
    for label, again in copies.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert again._linear_system_state.profiled_factor.rank == rank, label
        notices = [w for w in caught if "saved by an earlier superglm build" in str(w.message)]
        assert len(notices) == 1, label
        again_se, again_leverage, text = _observed(again, frame, y)
        assert text, label
        for name, values in se.items():
            np.testing.assert_array_equal(np.isnan(again_se[name]), np.isnan(values))
            finite = ~np.isnan(values)
            np.testing.assert_allclose(
                again_se[name][finite], values[finite], rtol=bound, atol=0.0, err_msg=label
            )
        # a quadratic form in the same H^+, as the variances (the solver's BLAS
        # cap may differ from the fit's, so not bit for bit)
        np.testing.assert_allclose(again_leverage, leverage, rtol=bound, atol=0.0, err_msg=label)


def test_a_current_factor_round_trips_without_a_rebuild():
    """A state written by this format is restored as it is: no notice, the same object graph."""
    model, frame, y = _model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = pickle.loads(pickle.dumps(model))
        factor = loaded._linear_system_state.augmented_factor
        assert "_retired_state" not in factor.__dict__
        _observed(loaded, frame, y)
    assert not [w for w in caught if "saved by an earlier superglm build" in str(w.message)]


def test_a_retired_raw_coordinate_factor_is_restored_inert():
    """An earlier build's raw-coordinate factor (``intercept=False``), which nothing reads, loads inert."""
    model, _, _ = _model()
    factor = model._linear_system_state.augmented_factor
    old = {name: factor.__dict__[name] for name in _INPUTS} | {"intercept": False}
    restored = type(factor).__new__(type(factor))
    restored.__setstate__(old)
    assert restored.intercept is False  # first use restores the saved dictionary, no factorization
    with pytest.raises(AttributeError):
        _ = restored.border_certificate
