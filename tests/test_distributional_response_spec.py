"""Response metadata grants shape permission without replacing the family's law."""

from dataclasses import FrozenInstanceError
from importlib import import_module, util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm.distributional import GammaLS, GaussianLS, NegativeBinomialLS
from superglm.distributional import family as contracts
from superglm.distributional.families import _base
from superglm.distributional.families._base import response_row_count, validated_float_response
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights


def _api():
    name = "superglm.distributional.response"
    assert util.find_spec(name) is not None, "the structural response helpers are missing"
    return import_module(name)


def _vector():
    return SimpleNamespace(
        response_spec=_api().ResponseSpec(shape=(2,), component_names=("claim type", "expense")),
        default_prediction_columns=("claim type", "expense"),
    )


def test_scalar_default_and_vector_opt_in_do_not_flatten_observations():
    api = _api()
    scalar = GaussianLS()
    assert api.response_spec(scalar) == api.ResponseSpec()
    y = np.array([[0.25, 0.75], [0.5, 0.5], [0.8, 0.2]])
    with pytest.raises(ValueError, match="shape"):
        api.unvalidated_response(y, family=scalar, n_rows=3)
    vector = _vector()
    np.testing.assert_array_equal(api.unvalidated_response(y, family=vector, n_rows=3), y)
    assert response_row_count(y) == 3
    assert isinstance(vector, contracts.ResponseSpecFamily)
    assert isinstance(vector, contracts.VectorDefaultPredictionFamily)
    assert not isinstance(scalar, contracts.ResponseSpecFamily)
    with pytest.raises(FrozenInstanceError):
        vector.response_spec.shape = (3,)


def test_row_count_is_axis_zero_but_scalar_validator_is_still_strict():
    assert response_row_count(np.ones((3, 2))) == 3
    assert response_row_count(np.ones((3, 2, 4))) == 3
    for y in (1.0, np.array([])):
        with pytest.raises(ValueError):
            response_row_count(y)
    for y in (1.0, np.ones((3, 1)), np.ones((3, 2))):
        with pytest.raises(ValueError, match="scalar response"):
            validated_float_response(y, message="scalar response")


def test_scalar_count_companion_preserves_shape_only_validation_for_scalar_adapters():
    assert hasattr(_base, "scalar_response_row_count"), "scalar adapter companion is missing"
    assert _base.scalar_response_row_count(np.array([None, "unread"], dtype=object)) == 2
    for y in (1.0, np.array([]), np.ones((3, 1)), np.ones((3, 2))):
        with pytest.raises(ValueError):
            _base.scalar_response_row_count(y)


@pytest.mark.parametrize("y", [1.0, [1.0], [[1.0], [2.0]], np.ones((2, 2))])
def test_scalar_shape_is_not_broadcast_or_squeezed(y):
    with pytest.raises(ValueError, match="shape"):
        _api().unvalidated_response(y, family=GaussianLS(), n_rows=2)


def test_named_columns_are_canonicalized_without_reordering_rows_or_casting_values():
    api, family = _api(), _vector()
    frame = pd.DataFrame(
        {"expense": ["discard me", 0.75], "claim type": [None, 0.25]}, index=[7, 3]
    )
    raw = api.unvalidated_response(frame, family=family, n_rows=2, input_index=frame.index)
    assert raw.dtype == object
    assert raw[0, 1] == "discard me"
    weights = resolve_likelihood_weights(
        np.array([0, 3]), n_observations=2, contract=WeightContract("frequency")
    )
    retained = raw[weights.input_positions]
    np.testing.assert_array_equal(
        api.validated_response_array(retained, family=family, n_rows=1), [[0.25, 0.75]]
    )
    with pytest.raises(ValueError, match="index"):
        api.unvalidated_response(frame, family=family, n_rows=2, input_index=pd.Index([3, 7]))


@pytest.mark.parametrize(
    "columns",
    [
        ["claim type"],
        ["claim type", "claim type"],
        ["claim type", "extra"],
        ["claim type", "expense", "extra"],
    ],
)
def test_bad_component_labels_fail_before_weight_selection(columns):
    frame = pd.DataFrame([[None] * len(columns), [None] * len(columns)], columns=columns)
    with pytest.raises(ValueError, match="column|component"):
        _api().unvalidated_response(frame, family=_vector(), n_rows=2)


@pytest.mark.parametrize(
    "shape,names",
    [
        ((0,), ()),
        ((True,), ("a",)),
        ((2,), ()),
        ((2,), ("a",)),
        ((2,), ("a", "a")),
        ((2,), ("a", " ")),
        ((), ("a",)),
    ],
)
def test_response_metadata_rejects_ambiguous_components(shape, names):
    with pytest.raises((TypeError, ValueError)):
        _api().ResponseSpec(shape=shape, component_names=names)


def test_malformed_declared_spec_is_not_silently_treated_as_scalar():
    with pytest.raises(TypeError, match="ResponseSpec"):
        _api().response_spec(SimpleNamespace(response_spec=None))


@pytest.mark.parametrize(
    "y",
    [
        np.array([[True, False]]),
        np.array([[1 + 0j, 2 + 0j]]),
        np.array([[True, 0.5]], dtype=object),
        np.array([[1 + 0j, 0.5]], dtype=object),
        np.array([["0.5", 0.5]], dtype=object),
        np.array([[None, 0.5]], dtype=object),
        np.array([[np.nan, 0.5]]),
        np.array([[np.inf, 0.5]]),
    ],
)
def test_retained_values_cannot_be_legitimized_by_float_cast(y):
    api, family = _api(), _vector()
    raw = api.unvalidated_response(y, family=family, n_rows=1)
    assert raw.dtype == y.dtype
    with pytest.raises(ValueError):
        api.validated_response_array(raw, family=family, n_rows=1)


def test_raw_values_still_reach_family_domain_and_exact_integer_validation():
    api = _api()
    weights = resolve_likelihood_weights(None, n_observations=1, contract=WeightContract("prior"))
    for family, y in (
        (GammaLS(), np.array([-1.0])),
        (NegativeBinomialLS(), np.array([2**53 + 1], dtype=np.int64)),
        (NegativeBinomialLS(), np.array([True], dtype=object)),
    ):
        raw = api.unvalidated_response(y, family=family, n_rows=1)
        np.testing.assert_array_equal(raw, y)
        assert raw.dtype == y.dtype
        with pytest.raises(ValueError):
            family.bind_likelihood(raw, weights, COMPLETE_OBSERVATION)


def test_indexless_labeled_frames_are_positional():
    import polars as pl

    frame = pl.DataFrame({"expense": [0.75, 0.5], "claim type": [0.25, 0.5]})
    raw = _api().unvalidated_response(
        frame, family=_vector(), n_rows=2, input_index=pd.Index([8, 4])
    )
    np.testing.assert_array_equal(raw, [[0.25, 0.75], [0.5, 0.5]])


def test_python_sequences_and_mixed_columns_keep_individual_value_types():
    api, family = _api(), _vector()
    for source in ([[True, 0.5]], pd.DataFrame({"claim type": [True], "expense": [0.5]})):
        raw = api.unvalidated_response(source, family=family, n_rows=1)
        assert isinstance(raw[0, 0], bool | np.bool_)
        with pytest.raises(ValueError):
            api.validated_response_array(raw, family=family, n_rows=1)
    # Mixing numeric columns must not round an exact integer before family validation.
    frame = pd.DataFrame({"claim type": [2**53 + 1], "expense": [0.5]})
    raw = api.unvalidated_response(frame, family=family, n_rows=1)
    assert int(raw[0, 0]) == 2**53 + 1


@pytest.mark.parametrize("dtype", ["Int64", "UInt64"])
@pytest.mark.parametrize("container", [pd.Series, pd.Index, lambda values: values])
def test_nullable_scalar_values_reach_family_unchanged(dtype, container):
    original = 2**53 + 1
    source = container(pd.array([original, pd.NA], dtype=dtype))
    raw = _api().unvalidated_response(source, family=NegativeBinomialLS(), n_rows=2)
    weights = resolve_likelihood_weights(
        np.array([0.5, 0.0]), n_observations=2, contract=WeightContract("prior")
    )
    retained = raw[weights.input_positions]

    assert isinstance(retained[0], int | np.integer)
    assert int(retained[0]) == original
    assert pd.isna(raw[1])


def test_nullable_nb_prior_rates_are_checked_before_float_conversion():
    family = NegativeBinomialLS()
    source = pd.Series([2**53 + 1, pd.NA], dtype="Int64")
    weights = resolve_likelihood_weights(
        np.array([0.5, 0.0]), n_observations=2, contract=WeightContract("prior")
    )
    with pytest.raises(ValueError, match="lossless|exact float64"):
        family.bind_likelihood(np.array([2**53 + 1], dtype=object), weights, COMPLETE_OBSERVATION)

    raw = _api().unvalidated_response(source, family=family, n_rows=2)
    with pytest.raises(ValueError, match="lossless|exact float64"):
        family.bind_likelihood(raw[weights.input_positions], weights, COMPLETE_OBSERVATION)


@pytest.mark.parametrize("dtype", ["Int64", "UInt64"])
def test_nullable_vector_columns_preserve_retained_integers(dtype):
    frame = pd.DataFrame(
        {
            "expense": pd.array([0.75, pd.NA], dtype="Float64"),
            "claim type": pd.array([2**53 + 1, pd.NA], dtype=dtype),
        },
        index=[7, 3],
    )
    raw = _api().unvalidated_response(frame, family=_vector(), n_rows=2, input_index=frame.index)

    assert isinstance(raw[0, 0], int | np.integer)
    assert int(raw[0, 0]) == 2**53 + 1
    assert raw[0, 1] == 0.75
    assert pd.isna(raw[1, 0]) and pd.isna(raw[1, 1])


def test_nullable_composition_values_are_coerced_only_after_zero_drop():
    api, family = _api(), _vector()
    source = pd.DataFrame(
        {
            "expense": pd.array([0.75, pd.NA], dtype="Float64"),
            "claim type": pd.array([0.25, pd.NA], dtype="Float64"),
        }
    )
    raw = api.unvalidated_response(source, family=family, n_rows=2)
    weights = resolve_likelihood_weights(
        np.array([1, 0]), n_observations=2, contract=WeightContract("frequency")
    )

    np.testing.assert_array_equal(
        api.validated_response_array(raw[weights.input_positions], family=family, n_rows=1),
        [[0.25, 0.75]],
    )


def test_nullable_bool_component_still_fails_real_value_validation():
    frame = pd.DataFrame(
        {
            "expense": pd.array([0.75, pd.NA], dtype="Float64"),
            "claim type": pd.array([True, pd.NA], dtype="boolean"),
        }
    )
    raw = _api().unvalidated_response(frame, family=_vector(), n_rows=2)
    assert isinstance(raw[0, 0], bool | np.bool_)
    with pytest.raises(ValueError):
        _api().validated_response_array(raw[:1], family=_vector(), n_rows=1)
