"""Saved tensor coordinates and replay checks, with no response-model fits."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from superglm import Spline, TensorInteraction


def geometry():
    assert importlib.util.find_spec("interaction_compression_geometry") is not None, (
        "The saved tensor extraction helpers have not been implemented"
    )
    return importlib.import_module("interaction_compression_geometry")


def mapped_tensor():
    spec = TensorInteraction("x", "z")
    spec._p1, spec._p2 = 2, 3
    spec._R_inv = np.array(
        [[1, 0, 0, 2], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 3], [1, -1, 1, -1]],
        dtype=float,
    )
    return spec, np.array([2.0, 3.0, 5.0, 7.0])


def test_rectangular_runtime_map_precedes_nonsquare_row_major_reshape():
    extract = geometry().effective_tensor_coefficients
    spec, beta = mapped_tensor()
    # Exact integer multiplication gives (16, 3, 5, 5, 21, -3).
    expected = np.array([[16, 3, 5], [5, 21, -3]], dtype=float)
    np.testing.assert_array_equal(extract(spec, beta), expected)
    assert not np.array_equal(expected, (spec._R_inv @ beta).reshape(2, 3, order="F"))
    spec._R_inv = None
    with pytest.raises(ValueError, match="coefficient"):
        extract(spec, beta)


@pytest.mark.parametrize(
    "fault",
    [
        "decomposed",
        "class",
        "nan_beta",
        "nan_map",
        "map_shape",
        "map_rank",
        "zero_width",
        "complex_beta",
    ],
)
def test_invalid_coordinate_maps_are_refused(fault):
    extract = geometry().effective_tensor_coefficients
    spec, beta = mapped_tensor()
    if fault == "decomposed":
        spec._decompose = True
    elif fault == "class":
        spec = SimpleNamespace(**vars(spec))
    elif fault == "nan_beta":
        beta[0] = np.nan
    elif fault == "nan_map":
        spec._R_inv[0, 0] = np.inf
    elif fault == "map_shape":
        spec._R_inv = np.ones((5, 4))
    elif fault == "map_rank":
        spec._R_inv = np.ones(6)
    elif fault == "zero_width":
        spec._p1 = 0
    else:
        beta = beta.astype(complex) + 1j
    with pytest.raises((TypeError, ValueError)):
        extract(spec, beta)


@pytest.mark.parametrize("repeated", [False, True])
def test_saved_centering_and_clipping_replay_every_row_in_link_units(repeated):
    api = geometry()
    x = np.linspace(-2, 3, 61)
    z = np.sin(np.arange(61))
    parents = {"x": Spline(kind="cr", k=4), "z": Spline(kind="cr", k=5)}
    for name, values in (("x", x), ("z", z)):
        parents[name].build(values)
    spec = TensorInteraction("x", "z")
    spec.build(x, z, parents)
    size = spec._p1 * spec._p2
    spec._R_inv = np.eye(size) + np.tril(np.ones((size, size)), -1) / size
    beta = np.linspace(-4, 2, size)
    query_x = np.tile([-5.0, 0.0, 8.0], 29) if repeated else np.linspace(-5, 8, 97)
    query_z = np.tile([-4.0, 0.0, 4.0], 29) if repeated else np.cos(query_x)
    left, right = api.paired_centered_bases(spec, query_x, query_z)
    coefficients = api.effective_tensor_coefficients(spec, beta)
    # Saved projections, not means recomputed on the query population.
    for actual, values, info in (
        (left, query_x, spec._marginal1),
        (right, query_z, spec._marginal2),
    ):
        raw = info.raw_basis_eval(np.clip(values, info.lo, info.hi))
        np.testing.assert_array_equal(actual, raw @ info.projection)
    score = api.paired_tensor_score(left, coefficients, right)
    reference = spec.score(query_x, query_z, beta)
    estimate = api.contraction_roundoff_estimate(left, coefficients, right)
    assert np.all(np.abs(score - reference) <= estimate)
    assert np.any(score < 0), "Link contributions must not be exponentiated"
    receipt = api.replay_tensor_term(spec, beta, query_x, query_z)
    assert receipt["rows"] == len(query_x)
    assert receipt["within_roundoff_estimate"]
    assert receipt["roundoff_is_certificate"] is False


def test_roundoff_estimate_scales_with_uncancelled_products_and_detects_wrong_scores():
    api = geometry()
    left = np.array([[1.0, 1.0], [2.0, -3.0]])
    right = np.array([[1.0, 1.0, 1.0], [-1.0, 2.0, 3.0]])
    coefficients = np.array([[2.0**40, 1, -(2.0**40)], [-(2.0**40), 3, 2.0**40]])
    estimate = api.contraction_roundoff_estimate(left, coefficients, right)
    assert np.all(estimate > 0)
    np.testing.assert_array_equal(
        api.contraction_roundoff_estimate(left, 8 * coefficients, right), 8 * estimate
    )
    actual = api.paired_tensor_score(left, coefficients, right)
    explicit = np.sum(left[:, :, None] * coefficients * right[:, None, :], axis=(1, 2))
    assert np.all(np.abs(actual - explicit) <= estimate)
    assert np.any(
        np.abs(actual - api.paired_tensor_score(left, coefficients[:, ::-1], right)) > estimate
    )


def test_fitted_group_slice_is_the_only_source_of_term_coefficients():
    api = geometry()
    spec, expected = mapped_tensor()
    model = SimpleNamespace(
        _interaction_specs={"x:z": spec},
        _groups=[
            SimpleNamespace(feature_name="x", sl=slice(0, 2)),
            SimpleNamespace(feature_name="x:z", sl=slice(2, 6)),
        ],
        result=SimpleNamespace(beta=np.r_[99.0, -99.0, expected, 44.0]),
    )
    np.testing.assert_array_equal(api.fitted_term_beta(model, "x:z"), expected)
    model._groups.append(model._groups[-1])
    with pytest.raises(ValueError, match="one fitted group"):
        api.fitted_term_beta(model, "x:z")


def test_paired_bases_refuse_misaligned_or_nonfinite_coordinates():
    api = geometry()
    spec, _ = mapped_tensor()
    for left, right in (([1, 2], [1]), ([np.nan], [1])):
        with pytest.raises(ValueError):
            api.paired_centered_bases(spec, left, right)
