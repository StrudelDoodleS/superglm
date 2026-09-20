"""Saved tensor coordinates and replay checks, with no response-model fits."""

import importlib
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest

from superglm import Spline, TensorInteraction


def geometry():
    assert importlib.util.find_spec("interaction_compression_geometry") is not None, (
        "The saved tensor extraction helpers have not been implemented"
    )
    return importlib.import_module("interaction_compression_geometry")


def diagnostic(name):
    api = geometry()
    assert hasattr(api, name), f"The {name} diagnostic has not been implemented"
    return getattr(api, name)


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
    score = api.score_pairs(left, coefficients, right)
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
    actual = api.score_pairs(left, coefficients, right)
    explicit = np.sum(left[:, :, None] * coefficients * right[:, None, :], axis=(1, 2))
    assert np.all(np.abs(actual - explicit) <= estimate)
    assert np.any(np.abs(actual - api.score_pairs(left, coefficients[:, ::-1], right)) > estimate)


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


def test_rank_budget_uses_function_metric():
    truncate = diagnostic("truncate_in_product_metric")
    coefficients = np.diag([2.0, 1.0])
    left_factor = np.diag([1.0, 3.0])
    right_factor = np.eye(2)
    result = truncate(coefficients, left_factor, right_factor, rank=1)
    residual = left_factor @ (coefficients - result.coefficients) @ right_factor.T
    realized_error = float(np.linalg.norm(residual, ord="fro") ** 2)
    assert realized_error < (4.0 + 9.0) / 2.0
    assert result.factor_entries == 4


@pytest.mark.parametrize("rank,expected_energy", [(0, 13.0), (2, 0.0)])
def test_zero_and_full_rank_reconstruct_with_reported_allowance(rank, expected_energy):
    truncate = diagnostic("truncate_in_product_metric")
    result = truncate(np.diag([2.0, 1.0]), np.diag([1.0, 3.0]), np.eye(2), rank)
    assert abs(result.realized_product_energy - expected_energy) <= result.energy_allowance
    assert abs(result.discarded_product_energy - expected_energy) <= result.energy_allowance
    assert result.rank_budget == rank
    assert result.allowance_is_certificate is False


@pytest.mark.parametrize("rank,expected_energy", [(0, 40.0), (1, 4.0), (2, 0.0)])
def test_rectangular_rank_budgets_use_both_marginal_widths(rank, expected_energy):
    truncate = diagnostic("truncate_in_product_metric")
    C = np.array([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    result = truncate(C, np.diag([2.0, 1.0]), np.eye(3), rank)
    assert result.coefficients.shape == (2, 3)
    assert result.factor_entries == rank * 5
    assert abs(result.realized_product_energy - expected_energy) <= result.energy_allowance


@pytest.mark.parametrize("small", [0.0, np.finfo(float).eps])
def test_singular_and_illconditioned_metrics_refuse(small):
    truncate = diagnostic("truncate_in_product_metric")
    with pytest.raises(ValueError, match="condition|rank|budget"):
        truncate(np.eye(2), np.diag([1.0, small]), np.eye(2), 1)


def test_diagonal_observed_pairs_do_not_measure_product_energy():
    score = diagnostic("score_pairs")
    # e_1 and e_2 have equal marginal mass; off-diagonal changes are unseen.
    basis = np.eye(2)
    delta = np.array([[0.0, 2.0], [2.0, 0.0]])
    paired_energy = float(np.mean(score(basis, delta, basis) ** 2))
    product_energy = float(np.mean((basis @ delta @ basis.T) ** 2))
    assert paired_energy == 0.0
    assert product_energy == 2.0


def test_score_pairs_does_not_materialize_rowwise_tensor_products():
    score = diagnostic("score_pairs")
    left = np.ones((8193, 24))
    right = np.ones((8193, 32))
    coefficients = np.ones((24, 32))
    tracemalloc.start()
    try:
        actual = score(left, coefficients, right)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(actual, np.full(8193, 768.0))
    # A full row-by-width intermediate needs 2 MiB; the allowance is half that.
    # A rowwise tensor would need 48 MiB. Keep this separate from accuracy tests.
    assert peak < right.nbytes // 2


def test_modal_prefixes_are_nested_and_late_component_can_be_useful():
    prefix = diagnostic("modal_prefix")
    modes = np.eye(3)
    coefficients = np.diag([0.0, 0.0, 5.0])
    coarse = prefix(coefficients, modes, modes, 1)
    middle = prefix(coefficients, modes, modes, 2)
    full = prefix(coefficients, modes, modes, 3)
    np.testing.assert_array_equal(coarse, np.zeros((3, 3)))
    np.testing.assert_array_equal(prefix(middle, modes, modes, 1), coarse)
    np.testing.assert_array_equal(full, coefficients)
    assert np.linalg.norm(coefficients - coarse, ord="fro") ** 2 == 25.0
    populated = np.arange(2.0, 11.0).reshape(3, 3)
    expected_middle = np.array([[2, 3, 0], [5, 6, 0], [0, 0, 0]])
    np.testing.assert_array_equal(prefix(populated, modes, modes, 0), np.zeros((3, 3)))
    np.testing.assert_array_equal(prefix(populated, modes, modes, 2), expected_middle)
    np.testing.assert_array_equal(
        prefix(expected_middle, modes, modes, 1), prefix(populated, modes, modes, 1)
    )


def test_saved_penalty_modes_preserve_small_negative_eigenvalues_and_freeze_bases():
    saved = diagnostic("saved_penalty_modes")
    spec = TensorInteraction("x", "z")
    spec._p1 = spec._p2 = 3
    original = np.diag([-np.finfo(float).eps, 0.0, 2.0])
    spec._marginal1 = SimpleNamespace(penalty=original.copy())
    spec._marginal2 = SimpleNamespace(penalty=np.diag([0.0, 1.0, 1.0]))
    left, right = saved(spec)
    assert left.eigenvalues[0] < 0
    assert left.diagnostics["ambiguous_nullity"] == 2
    assert left.diagnostics["nullspace_certified"] is False
    assert right.diagnostics["tied_cutoffs"] == [2]
    assert not left.modes.flags.writeable
    assert not left.eigenvalues.flags.writeable
    spec._marginal1.penalty[:] = 0
    assert left.eigenvalues[-1] == 2.0
    reconstructed = left.modes @ np.diag(left.eigenvalues) @ left.modes.T
    assert norm_bound(reconstructed - original) <= left.diagnostics["eigenvalue_allowance"]


def test_modal_prefix_cutting_tie_depends_on_frozen_basis():
    modes = diagnostic("penalty_modes")
    prefix = diagnostic("modal_prefix")
    frozen = modes(np.diag([1.0, 1.0, 3.0]))
    assert frozen.diagnostics["tied_cutoffs"] == [1]
    C = np.diag([2.0, 1.0, 0.0])
    swapped = frozen.modes[:, [1, 0, 2]]
    first = prefix(C, frozen.modes, frozen.modes, 1)
    rotated = prefix(C, swapped, swapped, 1)
    assert np.linalg.norm(first - rotated) > 0


@pytest.mark.parametrize("penalty", [np.diag([-1.0, 2.0]), np.array([[1.0, 1.0], [0.0, 1.0]])])
def test_invalid_saved_penalties_are_refused(penalty):
    modes = diagnostic("penalty_modes")
    with pytest.raises(ValueError):
        modes(penalty)


@pytest.mark.parametrize("rotate", [False, True])
def test_modal_full_prefix_replays_coefficients_and_predictions_with_allowance(rotate):
    modes = diagnostic("penalty_modes")
    replay = diagnostic("modal_full_reconstruction")
    rotation = np.eye(4)
    if rotate:
        rotation, _ = np.linalg.qr(np.arange(16, dtype=float).reshape(4, 4) + np.eye(4))
    penalty = rotation @ np.diag([0.0, 1.0, 1.0, 4.0]) @ rotation.T
    frozen = modes(penalty)
    assert frozen.diagnostics["tied_cutoffs"] == [2]
    C = np.arange(16, dtype=float).reshape(4, 4) / 8
    left = np.arange(28, dtype=float).reshape(7, 4) / 8
    right = left[::-1]
    full, d = replay(C, frozen.modes, frozen.modes, left, right)
    assert np.linalg.norm(full - C) <= d["coefficient_allowance"]
    scores = geometry().score_pairs(left, full, right)
    reference = geometry().score_pairs(left, C, right)
    assert np.linalg.norm(scores - reference) <= d["score_allowance"]
    assert d["allowance_kind"] == "estimated"
    rotated_modes = frozen.modes[:, [0, 2, 1, 3]]
    rotated_full, rotated_d = replay(C, rotated_modes, rotated_modes, left, right)
    assert np.linalg.norm(rotated_full - C) <= rotated_d["coefficient_allowance"]


def test_penalty_symmetrization_is_explicit_and_roundoff_sized():
    modes = diagnostic("penalty_modes")
    penalty = np.diag([1.0, 2.0])
    penalty[0, 1] = np.finfo(float).eps / 2
    original = penalty.copy()
    result = modes(penalty)
    np.testing.assert_array_equal(penalty, original)
    assert result.diagnostics["target_changed"]
    assert result.diagnostics["target_change_norm"] > 0
    assert result.diagnostics["original_sha256"] != result.diagnostics["target_sha256"]
    penalty[0, 1] = np.sqrt(np.finfo(float).eps / 2)
    with pytest.raises(ValueError, match="asymmetr"):
        modes(penalty)


def test_zero_penalty_records_ambiguous_nullity_without_certification():
    modes = diagnostic("penalty_modes")
    result = modes(np.zeros((3, 3)))
    assert result.diagnostics["ambiguous_nullity"] == 3
    assert result.diagnostics["nullspace_certified"] is False
    np.testing.assert_array_equal(result.eigenvalues, np.zeros(3))


def hadamard_margins():
    J = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]], dtype=float)
    return J, np.diag([1.0, 2.0, 4.0]), np.diag([1.0, 0.5, 2.0])


def test_weighted_training_qr_represents_marginal_measure_and_weight_scale():
    metric = diagnostic("weighted_marginal_metric")
    J, left, _ = hadamard_margins()
    basis = J @ left
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    result = metric(basis, weights)
    scaled = metric(basis, np.ldexp(weights, 900))
    expected = basis.T @ ((weights / weights.sum())[:, None] * basis)
    actual = result.factor.T @ result.factor
    error = np.linalg.norm(actual - expected, ord="fro")
    d = result.diagnostics
    assert error <= d["metric_relative_error"] * np.linalg.norm(actual, ord="fro")
    assert d["metric_relative_error"] <= d["tau"]
    assert d["factor_condition"] >= 1
    assert d["allowance_kind"] == "estimated"
    scaled_metric = scaled.factor.T @ scaled.factor
    scaled_allowance = scaled.diagnostics["metric_relative_error"] * norm_bound(scaled_metric)
    assert norm_bound(scaled_metric - expected) <= scaled_allowance


@pytest.mark.parametrize("fault", ["duplicate", "near_rank", "zero_weights", "negative_weights"])
def test_weighted_training_metric_refuses_unsupported_geometry(fault):
    metric = diagnostic("weighted_marginal_metric")
    J, _, _ = hadamard_margins()
    weights = np.ones(4)
    if fault == "duplicate":
        J[:, 1] = J[:, 0]
    elif fault == "near_rank":
        J[:, 2] *= np.finfo(float).eps / 2
    elif fault == "zero_weights":
        weights[:] = 0
    else:
        weights[0] = -1
    with pytest.raises(ValueError):
        metric(J, weights)


@pytest.mark.parametrize(
    "basis,weights",
    [
        ([[2.0**600], [1.0]], [2.0**-600, 2.0**600]),
        ([[2.0**538], [1.0]], [3 * 2.0**-538, 2.0**537]),
        ([[1.0], [1.0], [1.0]], [2.0**-1074, 1.0, 1.0]),
        ([[2.0**-1074], [1.0]], [1.0, 3.0]),
    ],
    ids=[
        "normalized_zero",
        "normalized_positive_subnormal",
        "unit_sum_underflow",
        "formation_underflow",
    ],
)
def test_weighted_metric_refuses_underflow_outside_relative_error_model(basis, weights):
    # The first exact metrics are 2/(1+2**-1200) and 7/(1+3*2**-1075),
    # which round to 2 and 7. Lost normalization bits instead produced 1 and 9.
    # The last two cases isolate the unit-sum division and basis multiplication.
    with pytest.raises(ValueError, match="arithmetic scale|underflow"):
        geometry().weighted_marginal_metric(np.array(basis), np.array(weights))


def test_exact_subnormal_weight_scaling_remains_within_arithmetic_model():
    # All small intermediates are exact powers of two; the rounded metric is 2.
    result = geometry().weighted_marginal_metric(
        np.array([[2.0**537], [1.0]]), np.array([2.0**-1074, 1.0])
    )
    metric = (result.factor.T @ result.factor).item()
    assert abs(metric - 2.0) <= result.diagnostics["metric_relative_error"] * metric


def test_rank_energy_accuracy_uses_measured_allowance():
    truncate = diagnostic("truncate_in_product_metric")
    C = np.array([[2.0, 0.5], [0.25, 1.0]])
    result = truncate(C, np.diag([1.0, 3.0]), np.eye(2), 1)
    assert abs(result.realized_product_energy - result.discarded_product_energy) <= (
        result.energy_allowance
    )
    d = result.diagnostics
    assert d["residual_total"] <= d["tau_pair"] * d["whitened_norm"]
    assert d["svd_residual"] >= 0
    assert d["recovery_residual"] >= 0
    assert d["allowance_kind"] == "estimated"


def test_full_rank_one_ulp_recovery_perturbation_is_covered(monkeypatch):
    truncate = diagnostic("truncate_in_product_metric")
    api = geometry()
    solve = api.solve_triangular

    def perturbed_solve(*args, **kwargs):
        result = solve(*args, **kwargs)
        result.flat[0] = np.nextafter(result.flat[0], np.inf)
        return result

    monkeypatch.setattr(api, "solve_triangular", perturbed_solve)
    result = truncate(np.diag([2.0, 1.0]), np.eye(2), np.eye(2), 2)
    assert result.discarded_product_energy == 0
    assert 0 < result.realized_product_energy <= result.energy_allowance


def test_inaccurate_recovery_is_refused_instead_of_inflating_allowance(monkeypatch):
    truncate = diagnostic("truncate_in_product_metric")
    api = geometry()
    solve = api.solve_triangular

    def inaccurate_solve(*args, **kwargs):
        return 2 * solve(*args, **kwargs)

    monkeypatch.setattr(api, "solve_triangular", inaccurate_solve)
    with pytest.raises(ValueError, match="accuracy|budget"):
        truncate(np.eye(2), np.eye(2), np.eye(2), 1)


def gamma(count):
    scaled = count * np.finfo(float).eps / 2
    return scaled / (1 - scaled)


def norm_bound(value):
    return np.linalg.norm(value) / (1 - gamma(value.size + 2))


def polar_allowance(orthogonality):
    return orthogonality / (1 + np.sqrt(1 - orthogonality))


def coordinate_allowances(result, optimum_energy):
    d = result.diagnostics
    nu_left = polar_allowance(d["left_orthogonality"])
    nu_right = polar_allowance(d["right_orthogonality"])
    polar = (nu_left + nu_right + nu_left * nu_right) * norm_bound(result.singular_values)
    e = d["whitening_allowance"] + d["svd_residual"] + polar
    b = d["kept_product_allowance"] + d["recovery_residual"] + polar
    assert e + b <= d["tau_pair"] * d["whitened_norm"]
    z = 2 * e + b + d["evaluation_allowance"]
    metric = 2 * np.sqrt(optimum_energy) * z + z * z + d["energy_evaluation_allowance"]
    return e, b, metric


def test_separated_cutoff_predictions_survive_invertible_marginal_coordinates():
    truncate = diagnostic("truncate_in_product_metric")
    J, L, R = hadamard_margins()
    W = np.array([[5, 3, 0], [3, 5, 0], [0, 0, 0.5]])
    C = W / np.diag(L)[:, None] / np.diag(R)[None, :]
    T_left = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    T_right = np.array([[1, 0, 0], [0, 2, 0.25], [0, 0, 1]])
    changed = np.linalg.solve(T_left, C)
    changed = np.linalg.solve(T_right, changed.T).T
    # Exercise the saved runtime map before the diagnostic coefficient matrix.
    spec = TensorInteraction("x", "z")
    spec._p1 = spec._p2 = 3
    spec._R_inv = np.kron(T_left, T_right)
    reconstructed = geometry().effective_tensor_coefficients(spec, changed.ravel())
    np.testing.assert_array_equal(reconstructed, C)
    predictions, bounds, prediction_errors = [], [], []
    for coefficients, left, right in ((reconstructed, L, R), (changed, L @ T_left, R @ T_right)):
        result = truncate(coefficients, left, right, 1)
        e, b, metric_bound = coordinate_allowances(result, 17 / 4)
        assert abs(result.realized_product_energy - 17 / 4) <= metric_bound
        a = (16 + e) * e
        assert a < 60 / 2
        bounds.append(e + a * norm_bound(W) / (60 - a) + b)
        grid_left = np.repeat(J @ left, 4, axis=0)
        grid_right = np.tile(J @ right, (4, 1))
        predictions.append(geometry().score_pairs(grid_left, result.coefficients, grid_right))
        row_products = np.sum(grid_left**2, axis=1) * np.sum(grid_right**2, axis=1)
        prediction_errors.append(
            gamma(3 * 3 + 2) * norm_bound(result.coefficients) * np.sqrt(row_products.sum())
        )
    observed = norm_bound(predictions[0] - predictions[1])
    subtraction = gamma(1) * (norm_bound(predictions[0]) + norm_bound(predictions[1]))
    assert observed <= 4 * sum(bounds) + sum(prediction_errors) + subtraction


def test_tied_cutoff_preserves_error_and_rank_without_comparing_predictions():
    truncate = diagnostic("truncate_in_product_metric")
    C = np.diag([4.0, 4.0, 1.0])
    T_left = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    T_right = np.array([[1, 0, 0], [0, 2, 0.25], [0, 0, 1]])
    changed = np.linalg.solve(T_left, C)
    changed = np.linalg.solve(T_right, changed.T).T
    assert not np.array_equal(changed, C)
    for coefficients, left, right in (
        (C, np.eye(3), np.eye(3)),
        (changed, T_left, T_right),
    ):
        result = truncate(coefficients, left, right, 1)
        _, b, metric_bound = coordinate_allowances(result, 17)
        assert abs(result.realized_product_energy - 17) <= metric_bound
        recovered = left @ result.coefficients @ right.T
        U, s, Vt = np.linalg.svd(recovered, full_matrices=False)
        reconstructed = (U * s) @ Vt
        svd_error = (
            norm_bound(recovered - reconstructed)
            + gamma(1) * (norm_bound(recovered) + norm_bound(reconstructed))
            + gamma(4) * norm_bound(U) * norm_bound(s) * norm_bound(Vt)
        )
        o_left = norm_bound(U.T @ U - np.eye(3)) + gamma(3) * norm_bound(U) ** 2
        o_right = norm_bound(Vt @ Vt.T - np.eye(3)) + gamma(3) * norm_bound(Vt) ** 2
        # Include subtraction arithmetic in the measured orthogonality residuals.
        o_left += gamma(1) * (norm_bound(U.T @ U) + norm_bound(np.eye(3)))
        o_right += gamma(1) * (norm_bound(Vt @ Vt.T) + norm_bound(np.eye(3)))
        nu_left, nu_right = polar_allowance(o_left), polar_allowance(o_right)
        e_check = svd_error + (nu_left + nu_right + nu_left * nu_right) * norm_bound(s)
        recovery_product = (
            gamma(6) * norm_bound(left) * norm_bound(result.coefficients) * norm_bound(right)
        )
        threshold = b + recovery_product + e_check
        assert norm_bound(s[1:]) <= threshold < s[0]


def test_pair_metric_refuses_joint_conditioning_even_when_each_margin_passes():
    pair = diagnostic("product_metric_factors")
    metric = diagnostic("weighted_marginal_metric")
    basis = np.diag([1.0, 2.0**-16])
    for _ in range(2):
        metric(basis, np.ones(2))
    with pytest.raises(ValueError, match="two-sided"):
        pair(basis, basis, np.ones(2))


@pytest.mark.parametrize(
    "fault", ["negative_rank", "fractional_rank", "large_rank", "lower", "nan", "complex"]
)
def test_rank_input_boundary_refuses(fault):
    truncate = diagnostic("truncate_in_product_metric")
    C, L, R, rank = np.eye(2), np.eye(2), np.eye(2), 1
    if fault == "negative_rank":
        rank = -1
    elif fault == "fractional_rank":
        rank = 0.5
    elif fault == "large_rank":
        rank = 3
    elif fault == "lower":
        L[1, 0] = 1
    elif fault == "nan":
        C[0, 0] = np.nan
    else:
        C = C.astype(complex) + 1j
    with pytest.raises(ValueError):
        truncate(C, L, R, rank)


@pytest.mark.parametrize(
    "control", ["zero", "additive", "smooth", "localized", "multiple", "ridge"]
)
def test_small_centered_surface_controls(control):
    pair = diagnostic("product_metric_factors")
    truncate = diagnostic("truncate_in_product_metric")
    P = np.eye(8) - np.ones((8, 8)) / 8
    Q, _ = np.linalg.qr(P[:, :7], mode="reduced")
    B = np.sqrt(8) * Q
    if control in ("zero", "additive"):
        F, delta_surface, rank = np.zeros((8, 8)), 0.0, 0
    elif control == "smooth":
        a, b = np.arange(8), np.arange(8) ** 2
        x, y = P @ a, P @ b
        dx = gamma(8) * norm_bound(P) * norm_bound(a)
        dy = gamma(8) * norm_bound(P) * norm_bound(b)
        delta_surface = gamma(1) * norm_bound(x) * norm_bound(y)
        delta_surface += dx * norm_bound(y) + norm_bound(x) * dy + dx * dy
        F, rank = np.outer(x, y), 1
    elif control == "localized":
        x, y = P[:, 3], P[:, 5]
        delta_surface = gamma(1) * norm_bound(x) * norm_bound(y)
        F, rank = np.outer(x, y), 1
    elif control == "multiple":
        A, values = Q[:, :3], np.array([8, 2, 0.5])
        F, rank = (A * values) @ A.T, 3
        delta_surface = gamma(4) * norm_bound(A) ** 2 * norm_bound(values)
    else:
        shift = np.diag(np.ones(7), 1)
        T = np.eye(8) + (shift + shift.T) / 4
        F, rank = P @ T @ P, 7
        delta_surface = gamma(16) * norm_bound(P) ** 2 * norm_bound(T)
    C = Q.T @ F @ Q / 8
    left, right = pair(B, B, np.ones(8))
    result = truncate(C, left.factor, right.factor, rank)
    # Operand-based fixture allowance from task-2-fixture-design.md.
    delta_C = norm_bound(Q) ** 2 * (gamma(16) * norm_bound(F) + delta_surface) / 8
    delta_metric = np.linalg.norm(left.factor, 2) * np.linalg.norm(right.factor, 2) * delta_C
    e, _, _ = coordinate_allowances(result, 0)
    d = result.diagnostics
    assert (
        result.discarded_product_energy <= (delta_metric + e) ** 2 + d["tail_evaluation_allowance"]
    )
    assert (
        result.realized_product_energy <= result.discarded_product_energy + result.energy_allowance
    )
    if control == "ridge":
        rank_one = truncate(C, left.factor, right.factor, 1)
        assert rank_one.realized_product_energy > result.realized_product_energy
