"""Early validation regressions for public Tweedie profile inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import superglm.profiling.tweedie as tweedie_module
from superglm import SuperGLM
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.model import profile_ops
from superglm.profiling.tweedie import estimate_tweedie_p


@pytest.fixture
def profile_problem():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 12)})
    y = np.array([0.0, 0.2, 0.4, 0.7, 1.0, 1.4, 1.8, 2.2, 2.7, 3.1, 3.6, 4.0])
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0,
        features={"x": Numeric()},
    )
    return model, X, y


@pytest.mark.parametrize(
    "offset",
    [
        0.0,
        np.array([]),
        np.zeros((12, 1)),
        np.zeros(11),
        np.r_[np.zeros(11), np.nan],
        np.r_[np.zeros(11), np.inf],
        np.zeros(12, dtype=np.bool_),
        np.zeros(12, dtype=object),
        np.ones(12, dtype=np.complex128),
        ["0"] * 12,
    ],
)
def test_bad_offset_fails_before_profile_context(monkeypatch, profile_problem, offset):
    model, X, y = profile_problem

    def unexpected(*args, **kwargs):
        raise AssertionError("profile context must not be built")

    monkeypatch.setattr(tweedie_module, "_build_profile_context", unexpected)
    with pytest.raises((TypeError, ValueError), match="offset"):
        model.estimate_p(X, y, offset=offset)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"p_bounds": 1.5}, "p_bounds"),
        ({"p_bounds": (1.9, 1.1)}, "p_bounds"),
        ({"p_bounds": (1.0, 1.9)}, "p_bounds"),
        ({"grid": np.array([[1.4, 1.5]])}, "grid"),
        ({"grid": np.array([1.4, np.nan])}, "grid"),
        ({"xatol": 0.0}, "xatol"),
        ({"maxiter": True}, "maxiter"),
        ({"n_grid": 1}, "n_grid"),
        ({"n_grid_coarse": 1.5}, "n_grid_coarse"),
        ({"trace_callback": 42}, "trace_callback"),
    ],
)
def test_search_controls_fail_before_context(monkeypatch, profile_problem, kwargs, match):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises((TypeError, ValueError), match=match):
        estimate_tweedie_p(model, X, y, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"verbose": 1}, "verbose"),
        ({"trace_iterations": 1}, "trace_iterations"),
        ({"optimizer": "unknown"}, "optimizer"),
        ({"optimizer": []}, "optimizer"),
        ({"method": []}, "method"),
        ({"fit_mode": []}, "fit_mode"),
        ({"phi_method": []}, "phi_method"),
    ],
)
def test_other_controls_fail_before_context(monkeypatch, profile_problem, kwargs, match):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises((TypeError, ValueError), match=match):
        estimate_tweedie_p(model, X, y, **kwargs)


def test_model_api_unhashable_fit_mode_has_stable_error(monkeypatch, profile_problem):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises(ValueError, match="fit_mode"):
        model.estimate_p(X, y, fit_mode=[])


def test_profile_wrapper_unhashable_fit_mode_has_stable_error(monkeypatch, profile_problem):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises(ValueError, match="fit_mode"):
        profile_ops.estimate_p(model, X, y, fit_mode=[])


def test_progress_callback_fails_before_profile_context(monkeypatch, profile_problem):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises(TypeError, match="progress_callback"):
        model.estimate_p(X, y, progress_callback=42)


class _UnhashableString(str):
    __hash__ = None  # type: ignore[assignment]


class _BadFloat(float):
    def __float__(self):
        raise RuntimeError("float conversion failed")


class _BadInt(int):
    def __int__(self):
        raise RuntimeError("integer conversion failed")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": _UnhashableString("brent")}, "method"),
        ({"fit_mode": _UnhashableString("fit")}, "fit_mode"),
        ({"phi_method": _UnhashableString("mle")}, "phi_method"),
        ({"optimizer": _UnhashableString("L-BFGS-B")}, "optimizer"),
    ],
)
def test_unhashable_string_subclass_has_stable_error(monkeypatch, profile_problem, kwargs, match):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises(ValueError, match=match):
        estimate_tweedie_p(model, X, y, **kwargs)


def test_model_api_string_subclass_fit_mode_has_stable_error(monkeypatch, profile_problem):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises(ValueError, match="fit_mode"):
        model.estimate_p(X, y, fit_mode=_UnhashableString("fit"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"xatol": _BadFloat(1e-3)}, "xatol"),
        ({"maxiter": _BadInt(30)}, "maxiter"),
        ({"n_grid": _BadInt(20)}, "n_grid"),
        ({"n_grid_coarse": _BadInt(10)}, "n_grid_coarse"),
    ],
)
def test_numeric_scalar_subclass_has_stable_error(monkeypatch, profile_problem, kwargs, match):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises((TypeError, ValueError), match=match):
        estimate_tweedie_p(model, X, y, **kwargs)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("y", "y"),
        ("sample_weight", "weights"),
        ("offset", "offset"),
    ],
)
def test_masked_row_vector_fails_before_profile_context(monkeypatch, profile_problem, field, match):
    model, X, y = profile_problem
    masked = np.ma.array(np.ones(len(y)), mask=np.arange(len(y)) == 3)
    kwargs = {field: masked}
    if field != "y":
        kwargs["y"] = y
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises((TypeError, ValueError), match=match):
        estimate_tweedie_p(model, X, **kwargs)


@pytest.mark.parametrize("name", ["maxiter", "n_grid", "n_grid_coarse"])
def test_unrepresentable_integer_control_fails_before_profile_context(
    monkeypatch, profile_problem, name
):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises(ValueError, match=name):
        estimate_tweedie_p(model, X, y, **{name: 10**400})


@pytest.mark.parametrize("name", ["n_grid", "n_grid_coarse"])
def test_runaway_grid_count_fails_before_profile_context(monkeypatch, profile_problem, name):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises(ValueError, match=name):
        estimate_tweedie_p(model, X, y, **{name: 10_001})


def test_runaway_explicit_grid_fails_before_profile_context(monkeypatch, profile_problem):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises(ValueError, match="grid"):
        estimate_tweedie_p(
            model,
            X,
            y,
            method="grid",
            grid=np.linspace(1.1, 1.9, 10_001),
        )


@pytest.mark.parametrize(
    "y",
    [
        1.0,
        np.array([]),
        np.ones((12, 1)),
        np.r_[np.ones(11), np.nan],
        np.r_[np.ones(11), np.inf],
        np.ones(12, dtype=np.bool_),
        np.ones(12, dtype=object),
        np.ones(12, dtype=np.complex128),
        ["1"] * 12,
        np.r_[np.ones(11), -1.0],
    ],
)
def test_bad_response_fails_before_profile_context(monkeypatch, profile_problem, y):
    model, X, _ = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises((TypeError, ValueError), match="y"):
        estimate_tweedie_p(model, X, y)


def test_row_mismatch_fails_before_profile_context(monkeypatch, profile_problem):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises(ValueError, match="X.*y"):
        estimate_tweedie_p(model, X.iloc[:-1], y)


def test_non_dataframe_X_fails_before_profile_context(monkeypatch, profile_problem):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises(TypeError, match="X.*DataFrame"):
        estimate_tweedie_p(model, X.to_numpy(), y)


def test_duplicate_dataframe_columns_fail_before_profile_context(monkeypatch, profile_problem):
    model, _, y = profile_problem
    X = pd.DataFrame(np.ones((len(y), 2)), columns=["duplicate", "duplicate"])
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises(ValueError, match="X.*column.*unique"):
        estimate_tweedie_p(model, X, y)


@pytest.mark.parametrize(
    "sample_weight",
    [
        1.0,
        np.array([]),
        np.ones((12, 1)),
        np.ones(11),
        np.r_[np.ones(11), np.nan],
        np.r_[np.ones(11), np.inf],
        np.r_[np.ones(11), 0.0],
        np.ones(12, dtype=np.bool_),
        np.ones(12, dtype=object),
        np.ones(12, dtype=np.complex128),
        ["1"] * 12,
        [10**400] * 12,
    ],
)
def test_bad_weight_fails_before_profile_context(monkeypatch, profile_problem, sample_weight):
    model, X, y = profile_problem
    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **values: pytest.fail("context built before validation"),
    )
    with pytest.raises((TypeError, ValueError), match="weights"):
        estimate_tweedie_p(model, X, y, sample_weight=sample_weight)


def test_valid_weight_is_copied_before_profile_context(monkeypatch, profile_problem):
    model, X, y = profile_problem
    sample_weight = np.linspace(0.5, 1.5, len(y))
    captured = {}

    class ContextReachedError(Exception):
        pass

    def capture_context(model, X, y, sample_weight, offset, *args, **kwargs):
        captured["sample_weight"] = sample_weight
        raise ContextReachedError

    monkeypatch.setattr(tweedie_module, "_build_profile_context", capture_context)
    with pytest.raises(ContextReachedError):
        estimate_tweedie_p(model, X, y, sample_weight=sample_weight)

    normalized = captured["sample_weight"]
    assert normalized is not sample_weight
    assert not np.shares_memory(normalized, sample_weight)
    expected = normalized.copy()
    sample_weight[:] = 99.0
    np.testing.assert_array_equal(normalized, expected)


def test_categorical_dataframe_reaches_profile_context(monkeypatch, profile_problem):
    model, _, y = profile_problem
    X = pd.DataFrame(
        {
            "x": pd.Categorical(["a", "b"] * 6),
            "label": [f"row-{index}" for index in range(12)],
        }
    )

    class ContextReachedError(Exception):
        pass

    monkeypatch.setattr(
        tweedie_module,
        "_build_profile_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(ContextReachedError),
    )
    with pytest.raises(ContextReachedError):
        estimate_tweedie_p(model, X, y)
