"""Tests for the shared centered numerical-rank policy."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import Gaussian
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
)
from superglm.links import IdentityLink
from superglm.model import state_ops
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.solvers.centered_system import (
    TabmatCenteringState,
    build_centered_system,
    refresh_centered_rhs,
)
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import fit_pirls
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    decompose_factor,
    decompose_gram,
    needs_factor_certification,
    selected_group_name_set,
)
from superglm.types import GroupSlice


def _dense_design_matrix(X: np.ndarray) -> DesignMatrix:
    return DesignMatrix([DenseGroupMatrix(X)], n=X.shape[0], p=X.shape[1])


def _count_tabmat_split_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    import tabmat

    calls = {"standardize": 0, "sandwich": 0, "transpose_matvec": 0}
    original_standardize = tabmat.SplitMatrix.standardize
    original_sandwich = tabmat.SplitMatrix.sandwich
    original_transpose_matvec = tabmat.SplitMatrix.transpose_matvec

    def counted_standardize(self, *args, **kwargs):
        calls["standardize"] += 1
        return original_standardize(self, *args, **kwargs)

    def counted_sandwich(self, *args, **kwargs):
        calls["sandwich"] += 1
        return original_sandwich(self, *args, **kwargs)

    def counted_transpose_matvec(self, *args, **kwargs):
        calls["transpose_matvec"] += 1
        return original_transpose_matvec(self, *args, **kwargs)

    monkeypatch.setattr(tabmat.SplitMatrix, "standardize", counted_standardize)
    monkeypatch.setattr(tabmat.SplitMatrix, "sandwich", counted_sandwich)
    monkeypatch.setattr(tabmat.SplitMatrix, "transpose_matvec", counted_transpose_matvec)
    return calls


def _fitted_discrete_tensor_state():
    idx1 = np.repeat(np.arange(3, dtype=np.intp), 4)
    idx2 = np.tile(np.repeat(np.arange(2, dtype=np.intp), 2), 3)
    B1 = np.array([[-1.0, 0.5], [0.25, -1.5], [1.5, 1.0]])
    B2 = np.array([[-0.75, 1.0], [1.25, 0.25]])
    pair_codes = idx1 * len(B2) + idx2
    observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
    B_joint = (
        B1[observed_codes // len(B2), :, None] * B2[observed_codes % len(B2), None, :]
    ).reshape(len(observed_codes), 4)
    R_inv = np.array(
        [
            [1.0, 0.0, 0.25],
            [0.0, 1.0, -0.5],
            [0.5, -0.25, 0.0],
            [0.0, 0.5, 1.0],
        ]
    )
    tensor = DiscretizedTensorGroupMatrix(
        B1,
        B2,
        idx1,
        idx2,
        B_joint,
        R_inv,
        pair_idx.astype(np.intp),
        tensor_id=153,
    )
    dm = DesignMatrix([tensor], n=len(idx1), p=R_inv.shape[1])
    weights = np.linspace(0.5, 1.5, len(idx1))
    y = 1.25 + dm.matvec(np.array([0.4, -0.7, 0.25])) + 0.01 * np.arange(len(idx1))
    groups = [GroupSlice(name="tensor", start=0, end=3, feature_name="tensor")]
    result, _ = fit_irls_direct(
        dm,
        y,
        weights,
        Gaussian(),
        IdentityLink(),
        groups,
        lambda2=0.0,
        tol=1e-12,
    )
    model = SimpleNamespace(
        _dm=dm,
        _groups=groups,
        _distribution=Gaussian(),
        _link=IdentityLink(),
        _fit_weights=weights,
        _fit_offset=None,
        _runtime_canonical_state=None,
    )
    model._solver_pirls_result = lambda: result
    return model


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


def test_mixed_categorical_centering_uses_tabmat_without_materializing_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(154)
    n = 600
    n_levels = 120
    dense = rng.normal(size=(n, 2))
    codes = np.resize(np.arange(n_levels, dtype=np.intp), n)
    rng.shuffle(codes)
    categorical = CategoricalGroupMatrix(codes, n_levels=n_levels)
    dm = DesignMatrix(
        [DenseGroupMatrix(dense), categorical],
        n=n,
        p=dense.shape[1] + n_levels,
    )
    W = rng.uniform(0.25, 2.0, size=n)
    z = rng.normal(size=n)
    X = np.column_stack((dense, categorical.toarray()))
    mean_x = np.average(X, axis=0, weights=W)
    mean_z = float(np.average(z, weights=W))
    X_centered = X - mean_x
    calls = _count_tabmat_split_calls(monkeypatch)

    monkeypatch.setattr(
        dm,
        "row_subset",
        lambda _rows: pytest.fail("certified Tabmat centering must not materialize rows"),
    )
    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_split=dm.tabmat_split,
    )

    assert calls == {"standardize": 1, "sandwich": 1, "transpose_matvec": 1}
    np.testing.assert_allclose(system.mean_x, mean_x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        system.data_gram,
        X_centered.T @ (W[:, None] * X_centered),
        rtol=1e-12,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        system.rhs,
        X_centered.T @ (W * (z - mean_z)),
        rtol=1e-12,
        atol=1e-11,
    )


def test_unsafe_mixed_tabmat_centering_falls_back_to_stable_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(155)
    n = 600
    n_levels = 120
    dense = np.column_stack((1e12 + np.arange(n, dtype=float), rng.normal(size=n)))
    codes = np.resize(np.arange(n_levels, dtype=np.intp), n)
    rng.shuffle(codes)
    categorical = CategoricalGroupMatrix(codes, n_levels=n_levels)
    dm = DesignMatrix(
        [DenseGroupMatrix(dense), categorical],
        n=n,
        p=dense.shape[1] + n_levels,
    )
    W = rng.uniform(0.25, 2.0, size=n)
    z = rng.normal(size=n)
    penalty = np.zeros((dm.p, dm.p))
    expected = build_centered_system(dm=dm, W=W, z_off=z, penalty=penalty)
    calls = _count_tabmat_split_calls(monkeypatch)
    tabmat_state = TabmatCenteringState()
    original_row_subset = dm.row_subset
    row_subset_calls = 0

    def counted_row_subset(rows):
        nonlocal row_subset_calls
        row_subset_calls += 1
        return original_row_subset(rows)

    monkeypatch.setattr(dm, "row_subset", counted_row_subset)
    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=penalty,
        tabmat_split=dm.tabmat_split,
        tabmat_state=tabmat_state,
    )
    second = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=penalty,
        tabmat_split=dm.tabmat_split,
        tabmat_state=tabmat_state,
    )

    assert calls == {"standardize": 1, "sandwich": 0, "transpose_matvec": 0}
    assert tabmat_state.eligible is False
    assert row_subset_calls == 2
    np.testing.assert_array_equal(system.mean_x, expected.mean_x)
    np.testing.assert_array_equal(system.data_gram, expected.data_gram)
    np.testing.assert_array_equal(system.rhs, expected.rhs)
    np.testing.assert_array_equal(second.data_gram, expected.data_gram)


def test_categorical_only_centering_keeps_packed_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(156)
    n = 600
    n_levels = 120
    codes = np.resize(np.arange(n_levels, dtype=np.intp), n)
    rng.shuffle(codes)
    categorical = CategoricalGroupMatrix(codes, n_levels=n_levels)
    dm = DesignMatrix([categorical], n=n, p=n_levels)
    calls = _count_tabmat_split_calls(monkeypatch)

    monkeypatch.setattr(
        dm,
        "row_subset",
        lambda _rows: pytest.fail("packed categorical centering must not materialize rows"),
    )
    system = build_centered_system(
        dm=dm,
        W=np.ones(n),
        z_off=rng.normal(size=n),
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_split=dm.tabmat_split,
    )

    assert calls == {"standardize": 0, "sandwich": 0, "transpose_matvec": 0}
    assert np.all(np.isfinite(system.data_gram))


@pytest.mark.parametrize("weight_layout", ["strided", "readonly"])
def test_tabmat_centering_normalizes_weight_buffers(
    weight_layout: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(157)
    n = 600
    n_levels = 120
    dense = rng.normal(size=(n, 2))
    codes = np.resize(np.arange(n_levels, dtype=np.intp), n)
    rng.shuffle(codes)
    categorical = CategoricalGroupMatrix(codes, n_levels=n_levels)
    dm = DesignMatrix(
        [DenseGroupMatrix(dense), categorical],
        n=n,
        p=dense.shape[1] + n_levels,
    )
    base_weights = rng.uniform(0.25, 2.0, size=n)
    if weight_layout == "strided":
        storage = np.empty(2 * n)
        storage[::2] = base_weights
        W = storage[::2]
        assert not W.flags.c_contiguous
    else:
        W = base_weights.copy()
        W.setflags(write=False)
        assert not W.flags.writeable
    z = rng.normal(size=n)
    penalty = np.zeros((dm.p, dm.p))
    expected = build_centered_system(dm=dm, W=base_weights, z_off=z, penalty=penalty)
    calls = _count_tabmat_split_calls(monkeypatch)

    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=penalty,
        tabmat_split=dm.tabmat_split,
    )

    assert calls == {"standardize": 1, "sandwich": 1, "transpose_matvec": 1}
    np.testing.assert_allclose(system.mean_x, expected.mean_x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(system.data_gram, expected.data_gram, rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(system.rhs, expected.rhs, rtol=1e-12, atol=1e-11)


def test_tabmat_split_uses_uniform_float64_solver_dtype() -> None:
    rng = np.random.default_rng(158)
    n = 600
    n_levels = 120
    dense = rng.normal(size=(n, 2)).astype(np.float32)
    codes = np.resize(np.arange(n_levels, dtype=np.intp), n)
    categorical = CategoricalGroupMatrix(codes, n_levels=n_levels)
    dm = DesignMatrix(
        [DenseGroupMatrix(dense), categorical],
        n=n,
        p=dense.shape[1] + n_levels,
    )
    W = rng.uniform(0.25, 2.0, size=n)
    z = rng.normal(size=n)
    penalty = np.zeros((dm.p, dm.p))
    expected = build_centered_system(dm=dm, W=W, z_off=z, penalty=penalty)
    split = dm.tabmat_split

    assert split is not None
    assert all(component.dtype == np.dtype(np.float64) for component in split.matrices)
    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=penalty,
        tabmat_split=split,
    )
    np.testing.assert_allclose(system.data_gram, expected.data_gram, rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(system.rhs, expected.rhs, rtol=1e-12, atol=1e-11)


def test_nonfinite_tabmat_raw_moments_fall_back_without_floating_point_error() -> None:
    rng = np.random.default_rng(159)
    n = 600
    n_levels = 120
    dense = (1e155 + 1e145 * rng.normal(size=n))[:, None]
    codes = np.resize(np.arange(n_levels, dtype=np.intp), n)
    categorical = CategoricalGroupMatrix(codes, n_levels=n_levels)
    dm = DesignMatrix(
        [DenseGroupMatrix(dense), categorical],
        n=n,
        p=dense.shape[1] + n_levels,
    )
    W = np.ones(n)
    z = rng.normal(size=n)
    penalty = np.zeros((dm.p, dm.p))
    expected = build_centered_system(dm=dm, W=W, z_off=z, penalty=penalty)
    tabmat_state = TabmatCenteringState(eligible=True)

    with np.errstate(over="raise", invalid="raise"):
        system = build_centered_system(
            dm=dm,
            W=W,
            z_off=z,
            penalty=penalty,
            tabmat_split=dm.tabmat_split,
            tabmat_state=tabmat_state,
        )

    assert tabmat_state.eligible is False
    assert np.all(np.isfinite(system.data_gram))
    np.testing.assert_array_equal(system.data_gram, expected.data_gram)
    np.testing.assert_array_equal(system.rhs, expected.rhs)


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


def test_packed_centering_avoids_materializing_tensor_rows(monkeypatch) -> None:
    idx1 = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2], dtype=np.intp)
    idx2 = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0], dtype=np.intp)
    B1 = np.column_stack(
        (
            1e6 + np.array([0.0, 2.0, 5.0]),
            np.array([1.0, -1.0, 3.0]),
        )
    )
    B2 = np.column_stack(
        (
            1e6 + np.array([0.0, 4.0]),
            np.array([-2.0, 2.0]),
        )
    )
    pair_codes = idx1 * len(B2) + idx2
    observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
    B_joint = (
        B1[observed_codes // len(B2), :, None] * B2[observed_codes % len(B2), None, :]
    ).reshape(len(observed_codes), 4)
    R_inv = np.array(
        [
            [1.0, 0.0, 0.25],
            [0.0, 1.0, -0.5],
            [0.0, 0.0, 1.0],
            [0.25, -0.25, 0.0],
        ]
    )
    tensor = DiscretizedTensorGroupMatrix(
        B1,
        B2,
        idx1,
        idx2,
        B_joint,
        R_inv,
        pair_idx.astype(np.intp),
        tensor_id=147,
    )
    dm = DesignMatrix([tensor], n=len(idx1), p=R_inv.shape[1])
    W = np.linspace(0.25, 2.0, len(idx1))
    z = np.cos(np.arange(len(idx1), dtype=float))
    support_mass = np.bincount(pair_idx, weights=W, minlength=len(B_joint))
    anchor = int(np.argmax(support_mass))
    support_differences = B_joint - B_joint[anchor]
    mean_difference = support_mass @ support_differences / np.sum(W)
    centered_support = (support_differences - mean_difference) @ R_inv
    mean_x = (B_joint[anchor] + mean_difference) @ R_inv
    X_centered = centered_support[pair_idx]
    mean_z = float(np.dot(W, z) / np.sum(W))

    monkeypatch.setattr(
        dm,
        "row_subset",
        lambda _rows: pytest.fail("packed tensor centering must not materialize rows"),
    )
    monkeypatch.setattr(
        DiscretizedTensorGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("packed tensor centering must not call toarray"),
    )

    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((R_inv.shape[1], R_inv.shape[1])),
    )

    np.testing.assert_allclose(system.mean_x, mean_x, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        system.data_gram,
        X_centered.T @ (W[:, None] * X_centered),
        rtol=1e-12,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        system.rhs,
        X_centered.T @ (W * (z - mean_z)),
        rtol=1e-12,
        atol=1e-8,
    )


def test_well_scaled_tensor_centering_reuses_factored_block_algebra(monkeypatch) -> None:
    from superglm._group_matrix import _group_matrix_centered as centered_algebra

    rng = np.random.default_rng(150)
    B1 = rng.normal(size=(5, 4))
    B2 = rng.normal(size=(4, 3))
    idx1 = rng.integers(0, len(B1), size=80, dtype=np.intp)
    idx2 = rng.integers(0, len(B2), size=80, dtype=np.intp)
    pair_codes = idx1 * len(B2) + idx2
    observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
    B_joint = (
        B1[observed_codes // len(B2), :, None] * B2[observed_codes % len(B2), None, :]
    ).reshape(len(observed_codes), B1.shape[1] * B2.shape[1])
    R_inv = rng.normal(size=(B_joint.shape[1], 7))
    tensor = DiscretizedTensorGroupMatrix(
        B1,
        B2,
        idx1,
        idx2,
        B_joint,
        R_inv,
        pair_idx.astype(np.intp),
        tensor_id=150,
    )
    categorical = CategoricalGroupMatrix(
        rng.integers(-1, 3, size=len(idx1), dtype=np.intp),
        n_levels=3,
    )
    dm = DesignMatrix([tensor, categorical], n=len(idx1), p=10)
    W = rng.uniform(0.25, 2.0, size=len(idx1))
    z = rng.normal(size=len(idx1))
    X = np.column_stack((tensor.toarray(), categorical.toarray()))
    mean_x = np.average(X, axis=0, weights=W)
    mean_z = float(np.average(z, weights=W))
    X_centered = X - mean_x

    monkeypatch.setattr(
        centered_algebra,
        "_anchor_center_support",
        lambda **_kwargs: pytest.fail("well-scaled tensor should retain factored algebra"),
    )
    monkeypatch.setattr(
        centered_algebra,
        "_try_factored_tensor_centering",
        lambda **_kwargs: pytest.fail("tensor path should use compressed support patterns"),
    )

    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
    )

    np.testing.assert_allclose(system.mean_x, mean_x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        system.data_gram,
        X_centered.T @ (W[:, None] * X_centered),
        rtol=1e-12,
        atol=1e-11,
    )
    pattern_plan = dm._centered_pattern_plan
    assert pattern_plan is not None
    assert pattern_plan.row_patterns.dtype == np.int32
    assert pattern_plan.unique_codes.shape[0] <= dm.n

    second = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
    )
    assert dm._centered_pattern_plan is pattern_plan
    np.testing.assert_allclose(second.data_gram, system.data_gram)
    np.testing.assert_allclose(
        system.rhs,
        X_centered.T @ (W * (z - mean_z)),
        rtol=1e-12,
        atol=1e-11,
    )


def test_pattern_tensor_centering_preserves_resolvable_near_collinearity() -> None:
    """Raw centering must not erase a direction retained by the shared rank policy."""
    B1 = np.array([[1.0, -1.0], [1.0, 1.0]])
    B2 = B1.copy()
    idx1 = np.repeat(np.arange(2, dtype=np.intp), 2)
    idx2 = np.tile(np.arange(2, dtype=np.intp), 2)
    B_joint = (B1[idx1, :, None] * B2[idx2, None, :]).reshape(4, 4)
    R_inv = np.array(
        [
            [1000.0, 1000.0],
            [0.0, 3e-6],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )
    tensor = DiscretizedTensorGroupMatrix(
        B1,
        B2,
        idx1,
        idx2,
        B_joint,
        R_inv,
        np.arange(4, dtype=np.intp),
        tensor_id=151,
    )
    dm = DesignMatrix([tensor], n=4, p=2)
    W = np.ones(4)
    z = np.arange(4, dtype=float)
    X = B_joint @ R_inv
    X_centered = X - np.average(X, axis=0, weights=W)
    expected = X_centered.T @ (W[:, None] * X_centered)

    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((2, 2)),
    )

    assert dm._centered_pattern_plan is not None
    assert decompose_gram(expected).rank == 2
    np.testing.assert_allclose(system.data_gram, expected, rtol=1e-12, atol=1e-13)
    assert decompose_gram(system.data_gram).rank == 2


def test_unsafe_pattern_tensor_centering_skips_duplicate_raw_assembly(monkeypatch) -> None:
    """A rejected pattern attempt should route directly to stable support centering."""
    from superglm._group_matrix import _group_matrix_centered as centered_algebra

    B1 = np.column_stack((1e12 + np.arange(3, dtype=float), np.ones(3)))
    B2 = np.column_stack((1e12 + np.arange(2, dtype=float), np.ones(2)))
    idx1 = np.repeat(np.arange(3, dtype=np.intp), 2)
    idx2 = np.tile(np.arange(2, dtype=np.intp), 3)
    B_joint = (B1[idx1, :, None] * B2[idx2, None, :]).reshape(6, 4)
    tensor = DiscretizedTensorGroupMatrix(
        B1,
        B2,
        idx1,
        idx2,
        B_joint,
        np.eye(4),
        np.arange(6, dtype=np.intp),
        tensor_id=152,
    )
    dm = DesignMatrix([tensor], n=6, p=4)

    monkeypatch.setattr(
        centered_algebra,
        "_try_factored_tensor_centering",
        lambda **_kwargs: pytest.fail("unsafe pattern centering should skip duplicate raw work"),
    )

    system = build_centered_system(
        dm=dm,
        W=np.ones(6),
        z_off=np.arange(6, dtype=float),
        penalty=np.zeros((4, 4)),
    )

    assert np.all(np.isfinite(system.data_gram))


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


def test_well_conditioned_gram_does_not_require_spectral_certification(monkeypatch) -> None:
    rng = np.random.default_rng(149)
    factor = rng.normal(size=(12, 12))
    matrix = factor.T @ factor + 2.0 * np.eye(12)
    rhs = rng.normal(size=12)

    monkeypatch.setattr(
        np.linalg,
        "eigh",
        lambda _matrix: pytest.fail("well-conditioned Gram should stay on Cholesky"),
    )

    decomposition = decompose_gram(matrix)

    assert decomposition.method == "cholesky"
    assert decomposition.rank == len(matrix)
    assert not decomposition.rank_truncated
    np.testing.assert_allclose(decomposition.solve(rhs), np.linalg.solve(matrix, rhs))


def test_condition_estimate_cannot_promote_boundary_rank(monkeypatch) -> None:
    import superglm.solvers.rank as rank_module

    eps = SHARED_RANK_POLICY.gram_rcond
    matrix = np.array([[1.0, 1.0 - eps], [1.0 - eps, 1.0]])
    original_get_lapack_funcs = rank_module.scipy.linalg.get_lapack_funcs

    def falsely_optimistic_condition(name, arrays=()):
        if name == "pocon":
            return lambda *_args, **_kwargs: (1.0, 0)
        return original_get_lapack_funcs(name, arrays)

    monkeypatch.setattr(
        rank_module.scipy.linalg,
        "get_lapack_funcs",
        falsely_optimistic_condition,
    )

    decomposition = decompose_gram(matrix)

    assert decomposition.rank == 1
    assert decomposition.rank_truncated


def test_exact_duplicate_is_truncated_consistently() -> None:
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    decomposition = decompose_gram(matrix)

    assert decomposition.rank == 1
    assert decomposition.rank_truncated
    assert decomposition.log_pdet == pytest.approx(np.log(2.0))
    inverse = decomposition.pseudo_inverse()
    np.testing.assert_allclose(matrix @ inverse @ matrix, matrix, atol=1e-12)
    assert not decomposition.is_estimable(np.array([1.0, 0.0]))
    assert decomposition.is_estimable(np.array([1.0, 1.0]))


def test_rank_truncated_log_pdet_is_stable_under_extreme_column_scaling() -> None:
    null_direction = np.ones(3) / np.sqrt(3.0)
    projection = np.eye(3) - np.outer(null_direction, null_direction)
    scales = np.array([1e150, 1.0, 1e-150])
    matrix = scales[:, None] * projection * scales[None, :]

    column_scale = np.sqrt(np.diag(matrix))
    equilibrated = matrix / np.outer(column_scale, column_scale)
    eigenvalues, eigenvectors = np.linalg.eigh(equilibrated)
    retained = eigenvalues > SHARED_RANK_POLICY.gram_rcond * eigenvalues[-1]
    retained_coordinates = column_scale[:, None] * eigenvectors[:, retained]
    _, coordinate_factor = np.linalg.qr(retained_coordinates, mode="reduced")
    expected_log_pdet = 2.0 * np.sum(np.log(np.abs(np.diag(coordinate_factor))))
    expected_log_pdet += np.sum(np.log(eigenvalues[retained]))

    decomposition = decompose_gram(matrix)

    assert decomposition.rank == 2
    assert np.isfinite(decomposition.log_pdet)
    assert decomposition.log_pdet == pytest.approx(expected_log_pdet, rel=1e-12)


@pytest.mark.parametrize("permutation", [[0, 1, 2], [2, 0, 1], [1, 2, 0]])
def test_factor_log_pdet_is_stable_for_extremely_scaled_retained_subspace(
    permutation: list[int],
) -> None:
    null_direction = np.ones(3) / np.sqrt(3.0)
    projection = np.eye(3) - np.outer(null_direction, null_direction)
    scales = np.array([1.0, 1e-20, 1e20])[permutation]
    factor = projection @ np.diag(scales)
    expected_log_pdet = 2.0 * float(np.sum(np.log(scales)))
    expected_log_pdet += float(np.log(np.sum((null_direction / scales) ** 2)))

    decomposition = decompose_factor(factor)

    assert decomposition.rank == 2
    assert decomposition.log_pdet == pytest.approx(expected_log_pdet, rel=1e-12)


def test_extreme_scale_factor_log_pdet_matches_cauchy_binet_and_column_permutation() -> None:
    rng = np.random.default_rng(20260710)
    base = rng.normal(size=(4, 10))
    log_scales = np.array([-150.0, -100.0, -40.0, -1.0, 1.0, 40.0, 80.0, 120.0, 145.0, 150.0])
    scales = 10.0**log_scales
    log_minors = []
    for columns in combinations(range(base.shape[1]), base.shape[0]):
        selected = np.asarray(columns, dtype=int)
        sign, log_abs_det = np.linalg.slogdet(base[:, selected])
        assert sign != 0.0
        log_minors.append(2.0 * (float(log_abs_det) + float(np.sum(np.log(scales[selected])))))
    expected_log_pdet = float(np.logaddexp.reduce(log_minors))

    factor = base * scales
    permutation = rng.permutation(factor.shape[1])
    decomposition = decompose_factor(factor)
    permuted = decompose_factor(factor[:, permutation])

    assert decomposition.rank == permuted.rank == base.shape[0]
    assert decomposition.log_pdet == pytest.approx(expected_log_pdet, rel=1e-12)
    assert permuted.log_pdet == pytest.approx(expected_log_pdet, rel=1e-12)


@pytest.mark.parametrize("columns", [np.array([1.0, 1.0]), np.array([2.0, 6.0])])
def test_factor_alias_log_pdet_matches_gram(columns: np.ndarray) -> None:
    factor = np.vstack([columns, np.zeros_like(columns)])

    factor_decomposition = decompose_factor(factor)
    gram_decomposition = decompose_gram(factor.T @ factor)

    assert factor_decomposition.rank == gram_decomposition.rank == 1
    assert factor_decomposition.log_pdet == pytest.approx(gram_decomposition.log_pdet)


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


def test_resolution_limited_gram_truncation_requests_factor_certification() -> None:
    rng = np.random.default_rng(0)
    orthonormal, _ = np.linalg.qr(rng.normal(size=(120, 2)))
    u, v = orthonormal.T
    separation = 3.0508168366745935e-8
    factor = np.column_stack((u, u + separation * v))

    gram_decomposition = decompose_gram(factor.T @ factor)
    factor_decomposition = decompose_factor(factor)

    assert gram_decomposition.rank == 1
    assert gram_decomposition.resolution_limited
    assert factor_decomposition.rank == 2
    assert needs_factor_certification(gram_decomposition)


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


@pytest.mark.parametrize(
    "penalty",
    [
        pytest.param(Ridge(lambda1=1.0), id="ridge"),
        pytest.param(GroupElasticNet(lambda1=1.0, alpha=0.0), id="elastic-net-pure-l2"),
    ],
)
def test_pirls_pure_l2_penalty_preserves_selected_zero_group(penalty) -> None:
    """Pure L2 shrinkage cannot deselect an exactly zero fitted group."""
    x = np.linspace(-1.0, 1.0, 40)[:, None]
    groups = [GroupSlice(name="x", start=0, end=1)]

    result = fit_pirls(
        x,
        np.full(len(x), 2.0),
        np.ones(len(x)),
        Gaussian(),
        IdentityLink(),
        groups,
        penalty,
        tol=1e-12,
    )

    assert result.beta[0] == pytest.approx(0.0, abs=1e-14)
    assert result.rank_info is not None
    assert result.rank_info.selected_group_names == ("x",)
    assert result.rank_info.group_edf == {"x": pytest.approx(1.0)}


@pytest.mark.parametrize(
    "penalty",
    [
        pytest.param(Ridge(lambda1=1.0), id="ridge"),
        pytest.param(GroupElasticNet(lambda1=1.0, alpha=0.0), id="elastic-net-pure-l2"),
    ],
)
def test_legacy_pure_l2_penalty_preserves_selected_zero_group(penalty) -> None:
    """Legacy selection fallback distinguishes smooth shrinkage from sparsity."""
    groups = [GroupSlice(name="x", start=0, end=1)]
    result = SimpleNamespace(rank_info=None, beta=np.zeros(1))

    assert selected_group_name_set(result, groups, penalty=penalty) == {"x"}


@pytest.mark.parametrize(
    "operation",
    [state_ops.coef_covariance, state_ops.fit_active_info, state_ops.fit_inference_info],
)
def test_rank_inference_paths_do_not_materialize_discrete_tensor_rows(
    monkeypatch, operation
) -> None:
    model = _fitted_discrete_tensor_state()

    monkeypatch.setattr(
        DiscretizedTensorGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("rank inference must not materialize tensor rows"),
    )

    output = operation(model)
    if operation is state_ops.fit_active_info:
        X_active = output[0]
        assert isinstance(X_active, DesignMatrix)
        assert X_active.shape == model._dm.shape
        beta = model._solver_pirls_result().beta
        np.testing.assert_allclose(X_active.matvec(beta), model._dm.matvec(beta))


def test_rank_active_state_rejects_selected_column_order_mismatch() -> None:
    """Group-name reconstruction must exactly match retained rank coordinates."""
    x = np.linspace(-1.0, 1.0, 60)
    frame = pd.DataFrame({"x": x, "z": np.cos(x)})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric(), "z": Numeric()},
    )
    model.fit(frame, 0.5 + x - 0.3 * np.cos(x))
    rank_info = model.result.rank_info
    assert rank_info is not None
    inconsistent = replace(rank_info, selected_columns=rank_info.selected_columns[::-1])

    with pytest.raises(ValueError, match="selected columns"):
        state_ops._rank_active_state(model, inconsistent)


def test_public_covariance_shift_uses_block_algebra(monkeypatch) -> None:
    """Intercept recentering must avoid dense identity-based cubic transforms."""
    rng = np.random.default_rng(20260720)
    factor = rng.normal(size=(4, 4))
    covariance = factor.T @ factor
    shift = np.array([0.2, -0.4, 0.1])
    transform = np.eye(4)
    transform[0, 1:] = shift
    expected = transform @ covariance @ transform.T
    model = SimpleNamespace(
        _runtime_canonical_state={
            "terms": {
                "x": {
                    "applied_to_public_model": True,
                    "groups": [{"group_name": "x", "column_means": shift}],
                }
            }
        }
    )
    active_groups = [GroupSlice(name="x", start=0, end=3)]

    monkeypatch.setattr(
        state_ops.np,
        "eye",
        lambda *_args, **_kwargs: pytest.fail("built a dense transform matrix"),
    )

    actual = state_ops._public_augmented_covariance(model, covariance, active_groups)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_rank_augmented_covariance_uses_profiled_blocks(monkeypatch) -> None:
    """Rank covariance maps the profiled intercept without cubic transforms."""
    hessian = np.array([[3.0, 0.4], [0.4, 1.8]])
    decomposition = decompose_gram(hessian)
    feature_covariance = decomposition.pseudo_inverse()
    mean_x = np.array([0.25, -0.6])
    rank_info = SimpleNamespace(
        selected_columns=np.array([0, 1]),
        sum_w=20.0,
        mean_x=mean_x,
        augmented=SimpleNamespace(pseudo_inverse=lambda: feature_covariance),
    )
    transform = np.eye(3)
    transform[0, 1:] = -mean_x
    centered = np.zeros((3, 3))
    centered[0, 0] = 1.0 / rank_info.sum_w
    centered[1:, 1:] = feature_covariance
    expected = transform @ centered @ transform.T
    model = SimpleNamespace(_runtime_canonical_state=None)
    active_groups = [GroupSlice(name="x", start=0, end=2)]

    monkeypatch.setattr(
        state_ops.np,
        "eye",
        lambda *_args, **_kwargs: pytest.fail("built a dense transform matrix"),
    )

    actual = state_ops._rank_augmented_covariance(model, rank_info, active_groups)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_rank_inference_edf1_counts_duplicate_retained_direction_once() -> None:
    x = np.linspace(-2.0, 2.0, 80)
    frame = pd.DataFrame({"x": x, "duplicate": x})
    y = 1.0 + 3.0 * x + 0.03 * np.sin(5.0 * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric(), "duplicate": Numeric()},
    )
    model.fit(frame, y)

    inference = model._fit_inference_info

    np.testing.assert_allclose(inference["edf"], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(inference["edf1"], [1.0, 0.0], atol=1e-12)


def test_rank_inference_edf1_spectral_branch_matches_direct_influence(
    monkeypatch,
) -> None:
    import superglm.solvers.rank as rank_module

    factor = np.array([[1.0, 0.3, -0.4], [0.2, 1.1, 0.7]])
    hessian = factor.T @ factor
    data_factor = np.array([[1.2, -0.5, 0.8], [0.3, 1.1, -0.4], [0.9, 0.2, 1.3]])
    data_gram = data_factor.T @ data_factor

    def reject_cholesky(*_args, **_kwargs):
        raise np.linalg.LinAlgError

    monkeypatch.setattr(
        rank_module.scipy.linalg,
        "cholesky",
        reject_cholesky,
    )
    decomposition = decompose_gram(hessian)
    assert decomposition.method == "gram_eigh"
    assert decomposition.rank == 2

    F = decomposition.pseudo_inverse() @ data_gram
    edf = np.diag(F).copy()
    expected = 2.0 * edf - np.diag(F @ F)
    rank_info = SimpleNamespace(augmented=decomposition)

    actual = state_ops._rank_edf1(rank_info, data_gram, edf)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


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


@pytest.mark.parametrize("alias_scale", [1.0, 1e8, 1e12])
def test_alias_estimability_is_invariant_to_column_scale(alias_scale: float) -> None:
    """Exact aliases remain non-estimable after arbitrary column rescaling."""
    x = np.linspace(-2.0, 2.0, 120)
    centered = x - np.mean(x)
    design = np.column_stack([centered, alias_scale * centered])
    decomposition = decompose_gram(design.T @ design)

    assert decomposition.rank == 1
    assert not decomposition.is_estimable(np.array([1.0, 0.0]))
    assert not decomposition.is_estimable(np.array([0.0, 1.0]))
    assert decomposition.is_estimable(np.array([1.0, alias_scale]))


@pytest.mark.parametrize("alias_scale", [1e4, 1e6, 1e10])
def test_scaled_alias_summary_suppresses_both_coefficients(alias_scale: float) -> None:
    """Scale-invariant rank metadata propagates through fitted summaries."""
    x = np.linspace(-2.0, 2.0, 120)
    frame = pd.DataFrame({"x": x, "scaled_duplicate": alias_scale * x})
    y = 1.0 + 2.5 * x + 0.03 * np.sin(4.0 * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric(), "scaled_duplicate": Numeric()},
    )
    model.fit(frame, y)
    assert model.result.beta[1] == pytest.approx(0.0, abs=1e-14)

    rows = {row.name: row for row in model.summary()._coef_rows}

    for name in ("x", "scaled_duplicate"):
        assert not rows[name].estimable
        assert np.isnan(rows[name].se)

    model.result.rank_info = None
    model._solver_pirls_result().rank_info = None
    model._summary_cache = None
    legacy_rows = {row.name: row for row in model.summary()._coef_rows}
    for name in ("x", "scaled_duplicate"):
        assert not legacy_rows[name].estimable
        assert np.isnan(legacy_rows[name].se)


def test_coefficient_estimability_mask_is_vectorized(monkeypatch) -> None:
    """Wide coefficient masks must not repeat full null-space projections."""
    x = np.linspace(-1.0, 1.0, 80)
    frame = pd.DataFrame({"x": x, "duplicate": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric(), "duplicate": Numeric()},
    )
    model.fit(frame, 1.0 + 2.0 * x)
    rank_info = model.result.rank_info
    assert rank_info is not None

    monkeypatch.setattr(
        type(rank_info.data),
        "is_estimable",
        lambda *_args, **_kwargs: pytest.fail("repeated scalar null-space projection"),
    )

    np.testing.assert_array_equal(rank_info.coefficient_estimable(), [False, False])


def test_diagonal_of_square_matches_matmul_without_forming_product() -> None:
    """EDF1's diag(F²) has an exact quadratic, not cubic, contraction."""
    from superglm.solvers.rank import diagonal_of_square

    rng = np.random.default_rng(20260721)
    matrix = rng.normal(size=(17, 17))

    np.testing.assert_allclose(
        diagonal_of_square(matrix),
        np.diag(matrix @ matrix),
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("scale_exponent", [8, 12])
def test_factor_log_pdet_handles_oppositely_scaled_full_rank_columns(
    scale_exponent: int,
) -> None:
    """Certification log-volume remains finite at extreme reciprocal scales."""
    epsilon = np.finfo(float).eps
    gram = np.array([[1.0, 1.0 - 8.0 * epsilon], [1.0 - 8.0 * epsilon, 1.0]])
    factor = np.linalg.cholesky(gram).T
    scaling = np.diag([10.0**-scale_exponent, 10.0**scale_exponent])

    decomposition = decompose_factor(factor @ scaling)

    assert decomposition.rank == 2
    assert np.isfinite(decomposition.log_pdet)
    assert decomposition.log_pdet == pytest.approx(
        np.linalg.slogdet(gram).logabsdet,
        rel=1e-10,
        abs=1e-10,
    )


def test_true_legacy_inference_matches_profiled_rank_state() -> None:
    """Old solver states without rank metadata retain centered EDF/covariance semantics."""
    rng = np.random.default_rng(20260722)
    x = np.linspace(-3.0, 3.0, 320)
    frame = pd.DataFrame({"x": x})
    weights = np.geomspace(0.05, 20.0, len(x))
    y = 0.4 + np.sin(x) + 0.08 * rng.normal(size=len(x))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=3.0,
        features={"x": Spline(n_knots=9, penalty="ssp")},
    )
    model.fit(frame, y, sample_weight=weights)
    baseline = state_ops.fit_inference_info(model)

    model.result.rank_info = None
    model._solver_pirls_result().rank_info = None
    legacy = state_ops.fit_inference_info(model)

    np.testing.assert_allclose(legacy["XtWX_inv_aug"], baseline["XtWX_inv_aug"], rtol=1e-9)
    np.testing.assert_allclose(legacy["edf"], baseline["edf"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(legacy["edf1"], baseline["edf1"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        legacy["R_a"].T @ legacy["R_a"],
        baseline["R_a"].T @ baseline["R_a"],
        rtol=1e-9,
        atol=1e-10,
    )
