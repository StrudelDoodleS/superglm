"""Tests for the shared centered numerical-rank policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import Gaussian
from superglm.features.numeric import Numeric
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
)
from superglm.links import IdentityLink
from superglm.penalties.group_lasso import GroupLasso
from superglm.solvers.centered_system import build_centered_system, refresh_centered_rhs
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import fit_pirls
from superglm.solvers.rank import SHARED_RANK_POLICY, decompose_factor, decompose_gram
from superglm.types import GroupSlice


def _dense_design_matrix(X: np.ndarray) -> DesignMatrix:
    return DesignMatrix([DenseGroupMatrix(X)], n=X.shape[0], p=X.shape[1])


def test_shared_rank_policy_matches_normal_equation_boundary() -> None:
    eps = np.finfo(float).eps

    assert SHARED_RANK_POLICY.factor_rcond == pytest.approx(np.sqrt(eps))
    assert SHARED_RANK_POLICY.gram_rcond == eps
    assert SHARED_RANK_POLICY.certification_band == 32.0
    assert SHARED_RANK_POLICY.warning_condition == pytest.approx(1.0 / np.sqrt(eps))
    assert SHARED_RANK_POLICY.severe_condition == pytest.approx(1.0 / eps)


def test_centered_system_avoids_raw_moment_cancellation() -> None:
    X = np.column_stack((np.full(8, 7.0), 1e9 + np.arange(8, dtype=float)))
    W = np.ones(8)
    z = 2.0 + np.arange(8, dtype=float)

    system = build_centered_system(
        dm=_dense_design_matrix(X),
        W=W,
        z_off=z,
        penalty=np.zeros((2, 2)),
    )

    centered = X - np.average(X, axis=0, weights=W)
    np.testing.assert_allclose(system.data_gram, centered.T @ (W[:, None] * centered))
    assert system.data_gram[0, 0] == pytest.approx(0.0, abs=1e-13)
    assert system.data_gram[1, 1] == pytest.approx(42.0)
    np.testing.assert_allclose(system.rhs, centered.T @ (W * (z - np.average(z, weights=W))))


def test_centered_rhs_is_stable_with_large_feature_and_response_means() -> None:
    delta = np.arange(12, dtype=float) - 5.5
    X = np.column_stack((1e12 + delta, -3e11 + 2.0 * delta))
    z = 8e12 - 4.0 * delta
    W = np.linspace(0.5, 2.0, len(delta))

    system = build_centered_system(
        dm=_dense_design_matrix(X),
        W=W,
        z_off=z,
        penalty=np.eye(2),
    )

    Xc = X - np.average(X, axis=0, weights=W)
    zc = z - np.average(z, weights=W)
    np.testing.assert_allclose(system.data_gram, Xc.T @ (W[:, None] * Xc))
    np.testing.assert_allclose(system.rhs, Xc.T @ (W * zc))
    np.testing.assert_allclose(system.hessian, system.data_gram + np.eye(2))
    for values in (
        system.mean_x,
        system.data_gram,
        system.rhs,
        system.penalty,
        system.hessian,
    ):
        assert not values.flags.writeable


def test_packed_centering_avoids_materializing_discrete_and_categorical_rows(
    monkeypatch,
) -> None:
    bin_idx = np.array([0, 0, 1, 2, 1, 0, 2, 2, 1, 0, 2, 1], dtype=np.intp)
    B_unique = np.column_stack(
        (
            1e12 + np.array([0.0, 2.0, 5.0]),
            -3e11 + np.array([0.0, -1.0, 4.0]),
        )
    )
    R_inv = np.array([[1.0, 0.25], [0.0, 1.0]])
    discrete = DiscretizedSSPGroupMatrix(B_unique, R_inv, bin_idx)
    categorical = CategoricalGroupMatrix(
        np.array([-1, 0, 1, 0, 1, -1, 0, 1, 0, -1, 1, 0]),
        n_levels=2,
    )
    dm = DesignMatrix([discrete, categorical], n=len(bin_idx), p=4)
    W = np.linspace(0.25, 2.0, len(bin_idx))
    W[3] = 0.0
    z = np.sin(np.arange(len(bin_idx), dtype=float))

    def centered_rows(support: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mass = np.bincount(codes, weights=W, minlength=len(support))
        anchor = int(np.argmax(mass))
        differences = support - support[anchor]
        mean_difference = mass @ differences / np.sum(W)
        return differences[codes] - mean_difference, support[anchor] + mean_difference

    discrete_centered_raw, discrete_mean_raw = centered_rows(B_unique, bin_idx)
    discrete_centered = discrete_centered_raw @ R_inv
    discrete_mean = discrete_mean_raw @ R_inv
    categorical_support = np.vstack((np.eye(2), np.zeros((1, 2))))
    categorical_centered, categorical_mean = centered_rows(
        categorical_support,
        categorical.codes,
    )
    X_centered = np.column_stack((discrete_centered, categorical_centered))
    mean_x = np.concatenate((discrete_mean, categorical_mean))
    mean_z = float(np.dot(W, z) / np.sum(W))
    z_centered = z - mean_z
    expected_gram = X_centered.T @ (W[:, None] * X_centered)
    expected_rhs = X_centered.T @ (W * z_centered)

    monkeypatch.setattr(
        dm,
        "row_subset",
        lambda _rows: pytest.fail("packed centering must not materialize observation rows"),
    )
    monkeypatch.setattr(
        DiscretizedSSPGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("packed centering must not materialize discrete rows"),
    )
    monkeypatch.setattr(
        CategoricalGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("packed centering must not materialize categorical rows"),
    )

    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((4, 4)),
    )

    np.testing.assert_allclose(system.mean_x, mean_x, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(system.data_gram, expected_gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(system.rhs, expected_rhs, rtol=1e-12, atol=1e-10)


def test_centered_system_reconstructs_raw_weighted_moments() -> None:
    rng = np.random.default_rng(147)
    X = rng.normal(size=(37, 4)) + np.array([0.0, 3.0, -7.0, 20.0])
    W = rng.uniform(0.2, 2.0, size=len(X))
    z = rng.normal(size=len(X)) + 5.0

    system = build_centered_system(
        dm=_dense_design_matrix(X),
        W=W,
        z_off=z,
        penalty=np.zeros((X.shape[1], X.shape[1])),
    )
    gram, xtw1, xtwz, sum_wz = system.raw_weighted_moments()

    np.testing.assert_allclose(gram, X.T @ (W[:, None] * X), rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(xtw1, X.T @ W, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(xtwz, X.T @ (W * z), rtol=1e-13, atol=1e-12)
    assert sum_wz == pytest.approx(float(np.dot(W, z)))


def test_well_scaled_rhs_refresh_uses_grouped_matvec(monkeypatch) -> None:
    rng = np.random.default_rng(148)
    X = rng.normal(size=(80, 4))
    W = rng.uniform(0.5, 1.5, size=len(X))
    first_z = rng.normal(size=len(X))
    next_z = rng.normal(size=len(X))
    dm = _dense_design_matrix(X)
    system = build_centered_system(dm=dm, W=W, z_off=first_z, penalty=np.eye(4))
    expected = build_centered_system(
        dm=_dense_design_matrix(X),
        W=W,
        z_off=next_z,
        penalty=np.eye(4),
    )

    monkeypatch.setattr(
        dm,
        "row_subset",
        lambda _rows: pytest.fail("well-scaled RHS refresh should use grouped rmatvec"),
    )
    refreshed = refresh_centered_rhs(system=system, dm=dm, W=W, z_off=next_z)

    np.testing.assert_allclose(refreshed.rhs, expected.rhs, rtol=1e-12, atol=1e-12)


def test_large_offset_rhs_refresh_retains_stable_centering() -> None:
    delta = np.arange(40, dtype=float) - 19.5
    X = np.column_stack((1e12 + delta, -3e11 + 2.0 * delta))
    W = np.linspace(0.5, 2.0, len(X))
    first_z = np.sin(delta)
    next_z = 8e12 - 4.0 * delta
    dm = _dense_design_matrix(X)
    system = build_centered_system(dm=dm, W=W, z_off=first_z, penalty=np.eye(2))
    refreshed = refresh_centered_rhs(system=system, dm=dm, W=W, z_off=next_z)
    expected = build_centered_system(
        dm=_dense_design_matrix(X),
        W=W,
        z_off=next_z,
        penalty=np.eye(2),
    )

    np.testing.assert_allclose(refreshed.rhs, expected.rhs, rtol=1e-12, atol=1e-12)


def test_centered_system_requires_positive_total_weight() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_centered_system(
            dm=_dense_design_matrix(np.ones((3, 1))),
            W=np.zeros(3),
            z_off=np.ones(3),
            penalty=np.zeros((1, 1)),
        )


def test_centered_system_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_centered_system(
            dm=_dense_design_matrix(np.ones((3, 1))),
            W=-np.ones(3),
            z_off=np.ones(3),
            penalty=np.zeros((1, 1)),
        )


def test_identity_uses_full_rank_cholesky_and_exact_operations() -> None:
    decomposition = decompose_gram(np.eye(3))

    assert decomposition.method == "cholesky"
    assert decomposition.rank == 3
    assert not decomposition.rank_truncated
    rhs = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(decomposition.solve(rhs), rhs)
    np.testing.assert_allclose(decomposition.pseudo_inverse(), np.eye(3))
    assert decomposition.log_pdet == pytest.approx(0.0)


def test_exact_duplicate_is_truncated_consistently() -> None:
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    decomposition = decompose_gram(matrix)

    assert decomposition.rank == 1
    assert decomposition.rank_truncated
    inverse = decomposition.pseudo_inverse()
    np.testing.assert_allclose(matrix @ inverse @ matrix, matrix, atol=1e-12)
    assert not decomposition.is_estimable(np.array([1.0, 0.0]))
    assert decomposition.is_estimable(np.array([1.0, 1.0]))


def test_shared_boundary_retains_above_and_truncates_below() -> None:
    eps = SHARED_RANK_POLICY.gram_rcond

    below = decompose_gram(np.array([[1.0, 1.0 - eps], [1.0 - eps, 1.0]]))
    above = decompose_gram(np.array([[1.0, 1.0 - 8 * eps], [1.0 - 8 * eps, 1.0]]))

    assert below.rank == 1
    assert above.rank == 2


def test_factor_and_gram_rules_agree_at_normal_equation_boundary() -> None:
    eps = SHARED_RANK_POLICY.gram_rcond
    gram = np.array([[1.0, 1.0 - eps], [1.0 - eps, 1.0]])
    factor = np.linalg.cholesky(gram).T

    factor_decomposition = decompose_factor(factor)
    gram_decomposition = decompose_gram(gram)

    assert factor_decomposition.rank == gram_decomposition.rank == 1


def test_column_rescaling_preserves_rank_and_fitted_projection() -> None:
    base = np.array([[2.0, 0.3], [0.3, 1.0]])
    rhs = np.array([1.0, -2.0])
    base_solution = decompose_gram(base).solve(rhs)

    scale = np.diag([1e-12, 1e12])
    scaled = scale @ base @ scale
    scaled_rhs = scale @ rhs
    scaled_solution = decompose_gram(scaled).solve(scaled_rhs)

    assert decompose_gram(base).rank == decompose_gram(scaled).rank == 2
    np.testing.assert_allclose(scale @ scaled_solution, base_solution, rtol=1e-10)


def test_zero_diagonal_column_is_inactive_and_nonestimable() -> None:
    decomposition = decompose_gram(np.diag([2.0, 0.0]))

    assert decomposition.rank == 1
    np.testing.assert_allclose(decomposition.solve(np.array([4.0, 9.0])), [2.0, 0.0])
    assert not decomposition.is_estimable(np.array([0.0, 1.0]))


def test_gram_and_qr_share_centered_alias_representation() -> None:
    x = np.linspace(-2.0, 2.0, 60)
    z = np.sin(x)
    X = np.column_stack((np.full_like(x, 7.0), x, x, np.zeros_like(x), z))
    y = 2.0 + 3.0 * x - 1.5 * z
    groups = [
        GroupSlice(name=name, start=index, end=index + 1)
        for index, name in enumerate(("constant", "x", "duplicate", "zero", "z"))
    ]
    results = {}

    for method in ("gram", "qr"):
        result, _ = fit_irls_direct(
            X,
            y,
            np.ones_like(y),
            Gaussian(),
            IdentityLink(),
            groups,
            lambda2=0.0,
            direct_solve=method,
            tol=1e-12,
        )
        results[method] = result
        assert result.rank_info is not None
        assert result.rank_info.data.rank == 2
        assert result.rank_info.augmented.rank == 2
        assert result.effective_df == pytest.approx(3.0)
        assert result.beta[0] == 0.0
        assert result.beta[2] == 0.0
        assert result.beta[3] == 0.0
        np.testing.assert_allclose(result.beta[[1, 4]], [3.0, -1.5], atol=1e-10)

    gram_prediction = results["gram"].intercept + X @ results["gram"].beta
    qr_prediction = results["qr"].intercept + X @ results["qr"].beta
    np.testing.assert_allclose(gram_prediction, y, atol=1e-10)
    np.testing.assert_allclose(qr_prediction, gram_prediction, atol=1e-10)


def test_pirls_selection_state_distinguishes_selected_zero_from_zeroed_group() -> None:
    x = np.linspace(-1.0, 1.0, 40)[:, None]
    group = [GroupSlice(name="x", start=0, end=1)]

    selected = fit_pirls(
        x,
        np.full(len(x), 2.0),
        np.ones(len(x)),
        Gaussian(),
        IdentityLink(),
        group,
        GroupLasso(lambda1=0.0),
        tol=1e-12,
    )
    zeroed = fit_pirls(
        x,
        2.0 + x[:, 0],
        np.ones(len(x)),
        Gaussian(),
        IdentityLink(),
        group,
        GroupLasso(lambda1=1e6),
        tol=1e-12,
    )

    assert selected.beta[0] == pytest.approx(0.0, abs=1e-14)
    assert selected.rank_info is not None
    assert selected.rank_info.selected_group_names == ("x",)
    np.testing.assert_array_equal(selected.rank_info.selected_columns, [0])
    assert zeroed.rank_info is not None
    assert zeroed.rank_info.selected_group_names == ()
    assert zeroed.rank_info.selected_columns.size == 0
    assert zeroed.effective_df == pytest.approx(1.0)


def test_alias_covariance_and_summary_suppress_nonestimable_coefficients() -> None:
    x = np.linspace(-2.0, 2.0, 80)
    frame = pd.DataFrame({"x": x, "duplicate": x})
    y = 1.0 + 3.0 * x + 0.03 * np.sin(5.0 * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric(), "duplicate": Numeric()},
    )
    model.fit(frame, y)

    rank_info = model._solver_pirls_result().rank_info
    assert rank_info is not None
    np.testing.assert_array_equal(rank_info.coefficient_estimable(), [False, False])
    assert rank_info.is_estimable(np.array([1.0, 1.0]))
    covariance, active_groups = model._coef_covariance
    assert [group.name for group in active_groups] == ["x", "duplicate"]
    assert np.linalg.matrix_rank(covariance) == 1
    rows = {row.name: row for row in model.summary()._coef_rows}
    for name in ("x", "duplicate"):
        assert not rows[name].estimable
        assert np.isnan(rows[name].se)
        assert np.isnan(rows[name].p)
