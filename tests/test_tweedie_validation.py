"""Early validation regressions for public Tweedie profile inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import superglm.profiling.tweedie as tweedie_module
from superglm import SuperGLM
from superglm._tweedie_numerics import (
    compound_poisson_gamma_parameters,
    pearson_dispersion,
    tweedie_unit_deviance,
)
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.model import profile_ops
from superglm.profiling.tweedie import (
    estimate_tweedie_p,
    generate_tweedie_cpg,
    tweedie_logpdf,
)


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


def _masked_vector(values):
    mask = np.zeros(len(values), dtype=np.bool_)
    mask[1] = True
    return np.ma.array(values, mask=mask)


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


@pytest.mark.parametrize("field", ["y", "mu", "weights"])
def test_masked_exact_density_input_is_rejected(field):
    values = {
        "y": np.array([0.2, 0.5, 1.0]),
        "mu": np.array([0.4, 0.8, 1.2]),
        "weights": np.ones(3),
    }
    values[field] = _masked_vector(values[field])

    with pytest.raises(TypeError, match=field):
        tweedie_logpdf(
            values["y"],
            values["mu"],
            0.8,
            1.5,
            weights=values["weights"],
        )


@pytest.mark.parametrize("field", ["y", "mu"])
def test_masked_unit_deviance_input_is_rejected(field):
    values = {
        "y": np.array([0.2, 0.5, 1.0]),
        "mu": np.array([0.4, 0.8, 1.2]),
    }
    values[field] = _masked_vector(values[field])

    with pytest.raises(TypeError, match=field):
        tweedie_unit_deviance(values["y"], values["mu"], 1.5)


@pytest.mark.parametrize(
    "operation",
    [
        lambda nested: tweedie_unit_deviance(nested, [[0.8]], 1.5),
        lambda nested: Tweedie(1.5).deviance_unit(nested, np.array([[0.8]])),
        lambda nested: Tweedie(1.5).variance(nested),
        lambda nested: Tweedie(1.5).variance_derivative(nested),
        lambda nested: Tweedie(1.5).variance_second_derivative(nested),
    ],
    ids=[
        "unit-deviance",
        "distribution-deviance",
        "variance",
        "variance-derivative",
        "variance-second-derivative",
    ],
)
def test_nested_masked_tweedie_numerical_input_is_rejected(operation):
    nested = [np.ma.array([0.5], mask=[True])]

    with pytest.raises(TypeError, match="mask"):
        operation(nested)


@pytest.mark.parametrize("field", ["y", "mu", "weights"])
def test_masked_pearson_input_is_rejected(field):
    values = {
        "y": np.array([0.2, 0.5, 1.0]),
        "mu": np.array([0.4, 0.8, 1.2]),
        "weights": np.ones(3),
    }
    values[field] = _masked_vector(values[field])

    with pytest.raises(TypeError, match=field):
        pearson_dispersion(
            values["y"],
            values["mu"],
            1.5,
            values["weights"],
            2.0,
        )


@pytest.mark.parametrize("field", ["mu", "phi", "weights"])
def test_masked_compound_parameters_input_is_rejected(field):
    values = {
        "mu": np.array([0.4, 0.8, 1.2]),
        "phi": np.full(3, 0.8),
        "weights": np.ones(3),
    }
    values[field] = _masked_vector(values[field])

    with pytest.raises(TypeError, match=field):
        compound_poisson_gamma_parameters(
            values["mu"],
            values["phi"],
            1.5,
            weights=values["weights"],
        )


def test_nested_masked_compound_dispersion_is_rejected_before_numpy_coercion():
    phi = [np.ma.array(0.8, mask=True), 0.8, 0.8]

    with pytest.raises(TypeError, match="phi.*mask"):
        compound_poisson_gamma_parameters(
            np.array([0.4, 0.8, 1.2]),
            phi,
            1.5,
        )


@pytest.mark.parametrize("field", ["n", "mu", "phi", "p"])
def test_masked_generator_input_is_rejected(field):
    values = {
        "n": 3,
        "mu": np.array([0.4, 0.8, 1.2]),
        "phi": np.full(3, 0.8),
        "p": 1.5,
    }
    if field == "n":
        values[field] = np.ma.array(3, mask=True)
    elif field == "p":
        values[field] = np.ma.array(1.5, mask=True)
    else:
        values[field] = _masked_vector(values[field])

    with pytest.raises((TypeError, ValueError), match=field):
        generate_tweedie_cpg(
            values["n"],
            values["mu"],
            values["phi"],
            values["p"],
            rng=np.random.default_rng(101),
        )


def test_masked_poisson_sampler_output_is_rejected_before_gamma() -> None:
    class MaskedPoissonRNG:
        def poisson(self, lam):
            return np.ma.array(np.ones(lam.shape, dtype=np.int64), mask=np.ones(lam.shape))

        def gamma(self, shape, *, scale):
            raise AssertionError("Gamma sampler must not run after masked Poisson output")

    with pytest.raises(RuntimeError, match="Poisson sampler output.*mask"):
        generate_tweedie_cpg(1, 1.0, 0.8, 1.5, rng=MaskedPoissonRNG())


def test_masked_gamma_sampler_output_is_rejected() -> None:
    class MaskedGammaRNG:
        def poisson(self, lam):
            return np.ones(lam.shape, dtype=np.int64)

        def gamma(self, shape, *, scale):
            return np.ma.array(np.full(shape.shape, 2.0), mask=np.ones(shape.shape))

    with pytest.raises(RuntimeError, match="Gamma sampler output.*mask"):
        generate_tweedie_cpg(1, 1.0, 0.8, 1.5, rng=MaskedGammaRNG())
