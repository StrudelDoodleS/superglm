"""Focused tests for mixed observation/bin-space centering."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import scipy.sparse as sp

from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.solvers.centered_system import TabmatCenteringState, build_centered_system
from superglm.solvers.rank import decompose_gram


def _mixed_discrete_design(
    *,
    n: int = 800,
    seed: int = 20260723,
    aliased_dense: bool = False,
) -> tuple[DesignMatrix, np.ndarray]:
    """Return a centered numeric + compressed spline + native categorical design."""
    rng = np.random.default_rng(seed)
    dense = rng.normal(size=(n, 2))
    dense -= np.mean(dense, axis=0)
    if aliased_dense:
        dense[:, 1] = dense[:, 0]

    n_bins = 8
    support = rng.normal(size=(n_bins, 3))
    support -= np.mean(support, axis=0)
    bin_idx = np.arange(n, dtype=np.intp) % n_bins
    discrete = DiscretizedSSPGroupMatrix(support, np.eye(3), bin_idx)

    n_levels = 120
    codes = np.arange(n, dtype=np.intp) % n_levels
    categorical = CategoricalGroupMatrix(codes, n_levels=n_levels)
    dm = DesignMatrix(
        [DenseGroupMatrix(dense), discrete, categorical],
        n=n,
        p=dense.shape[1] + discrete.shape[1] + categorical.shape[1],
    )
    X = np.column_stack((dense, discrete.toarray(), categorical.toarray()))
    return dm, X


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


@pytest.mark.parametrize("aliased_dense", [False, True])
def test_mixed_discrete_centering_uses_execution_plan_without_materializing_rows(
    monkeypatch: pytest.MonkeyPatch,
    aliased_dense: bool,
) -> None:
    """Large mixed designs retain compressed spline algebra after profiling the intercept."""
    dm, X = _mixed_discrete_design(aliased_dense=aliased_dense)
    rng = np.random.default_rng(20260724)
    W = rng.uniform(0.25, 2.0, size=dm.n)
    W[::11] = 0.0
    z = rng.normal(size=dm.n)
    mean_x = np.average(X, axis=0, weights=W)
    mean_z = float(np.average(z, weights=W))
    X_centered = X - mean_x
    expected_gram = X_centered.T @ (W[:, None] * X_centered)
    expected_rhs = X_centered.T @ (W * (z - mean_z))
    state = TabmatCenteringState()
    assert dm.n * dm.p == 100_000

    monkeypatch.setattr(
        dm,
        "row_subset",
        lambda _rows: pytest.fail("mixed execution-plan centering must not materialize rows"),
    )
    monkeypatch.setattr(
        DenseGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("bin-space centering copied dense groups through toarray"),
    )
    monkeypatch.setattr(
        DiscretizedSSPGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("mixed centering must preserve compressed spline support"),
    )
    monkeypatch.setattr(
        CategoricalGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("mixed centering must preserve categorical codes"),
    )

    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
    )

    assert state.eligible is True
    assert dm._mixed_bin_space_centering_plan is not None
    assert dm._execution_plan is None
    assert not dm._tabmat_built
    np.testing.assert_allclose(system.mean_x, mean_x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(system.data_gram, expected_gram, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(system.rhs, expected_rhs, rtol=2e-12, atol=2e-11)
    if aliased_dense:
        assert decompose_gram(system.data_gram).rank < dm.p


def test_fragmented_mixed_discrete_centering_uses_bin_space_tabmat_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fragmented numeric groups are coalesced by one augmented Tabmat plan."""
    rng = np.random.default_rng(20260729)
    n = 12_500
    dense_groups = []
    dense_columns = []
    for _index in range(5):
        values = rng.normal(size=(n, 1))
        values -= np.mean(values, axis=0)
        dense_groups.append(DenseGroupMatrix(values))
        dense_columns.append(values)
    support = rng.normal(size=(16, 3))
    support -= np.mean(support, axis=0)
    discrete = DiscretizedSSPGroupMatrix(
        support,
        np.eye(3),
        np.arange(n, dtype=np.intp) % len(support),
    )
    dm = DesignMatrix([*dense_groups, discrete], n=n, p=8)
    X = np.column_stack((*dense_columns, discrete.toarray()))
    W = rng.uniform(0.5, 1.5, size=n)
    z = rng.normal(size=n)
    mean_x = np.average(X, axis=0, weights=W)
    mean_z = float(np.average(z, weights=W))
    X_centered = X - mean_x
    state = TabmatCenteringState()
    calls = _count_tabmat_split_calls(monkeypatch)

    monkeypatch.setattr(
        dm,
        "row_subset",
        lambda _rows: pytest.fail("fragmented mixed centering materialized row chunks"),
    )
    monkeypatch.setattr(
        DiscretizedSSPGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("fragmented mixed centering expanded spline rows"),
    )

    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
    )

    assert state.eligible is True
    assert calls == {"standardize": 1, "sandwich": 1, "transpose_matvec": 1}
    assert dm._mixed_bin_space_centering_plan is not None
    np.testing.assert_array_equal(
        dm._mixed_bin_space_centering_plan.ordinary_solver_indices,
        np.arange(5),
    )
    assert not hasattr(dm, "_mixed_centering_execution_plan")
    assert dm._execution_plan is None
    assert not dm._tabmat_built
    np.testing.assert_allclose(system.mean_x, mean_x, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        system.data_gram,
        X_centered.T @ (W[:, None] * X_centered),
        rtol=2e-12,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        system.rhs,
        X_centered.T @ (W * (z - mean_z)),
        rtol=2e-12,
        atol=2e-10,
    )


def test_mixed_bin_space_centering_uses_public_tabmat_without_numba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One augmented Tabmat plan must assemble mixed moments without compiled scatters."""
    from superglm._group_matrix import _group_matrix_algebra as grouped_algebra
    from superglm._group_matrix import _group_matrix_discretized as discrete_algebra

    rng = np.random.default_rng(20260801)
    n = 1_000
    dense_groups = [DenseGroupMatrix(rng.normal(size=(n, 1))) for _index in range(5)]
    support = rng.normal(size=(16, 3))
    support -= np.mean(support, axis=0)
    discrete = DiscretizedSSPGroupMatrix(
        support,
        np.eye(3),
        np.arange(n, dtype=np.intp) % len(support),
    )
    categorical = CategoricalGroupMatrix(
        np.arange(n, dtype=np.intp) % 120,
        n_levels=120,
    )
    dm = DesignMatrix(
        [*dense_groups, discrete, categorical],
        n=n,
        p=5 + discrete.shape[1] + categorical.shape[1],
    )
    X = np.column_stack(
        [*(group.toarray() for group in dense_groups), discrete.toarray(), categorical.toarray()]
    )
    W = rng.uniform(0.25, 2.0, size=n)
    W[::13] = 0.0
    z = rng.normal(size=n)
    mean_x = np.average(X, axis=0, weights=W)
    mean_z = float(np.average(z, weights=W))
    X_centered = X - mean_x
    calls = _count_tabmat_split_calls(monkeypatch)

    def forbidden_numba(*_args, **_kwargs):
        pytest.fail("accepted bin-space centering invoked a Numba scatter kernel")

    monkeypatch.setattr(discrete_algebra, "_fused_bincount_2", forbidden_numba)
    monkeypatch.setattr(grouped_algebra, "_weighted_bincount_2d", forbidden_numba)
    monkeypatch.setattr(grouped_algebra, "_disc_disc_2d_hist", forbidden_numba)
    monkeypatch.setattr(grouped_algebra, "_cat_weighted_bincount", forbidden_numba)
    monkeypatch.setattr(
        dm,
        "row_subset",
        lambda _rows: pytest.fail("bin-space centering materialized stable row chunks"),
    )
    monkeypatch.setattr(
        DiscretizedSSPGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("bin-space centering expanded spline rows"),
    )
    monkeypatch.setattr(
        CategoricalGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("bin-space centering expanded categorical rows"),
    )

    state = TabmatCenteringState()
    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
    )

    assert state.eligible is True
    assert calls == {"standardize": 1, "sandwich": 1, "transpose_matvec": 1}
    assert dm._mixed_bin_space_centering_plan is not None
    assert not hasattr(dm, "_mixed_centering_execution_plan")
    assert dm._execution_plan is None
    assert not dm._tabmat_built
    np.testing.assert_allclose(system.mean_x, mean_x, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        system.data_gram,
        X_centered.T @ (W[:, None] * X_centered),
        rtol=2e-12,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        system.rhs,
        X_centered.T @ (W * (z - mean_z)),
        rtol=2e-12,
        atol=2e-10,
    )


def test_mixed_bin_space_plan_is_cached_and_pickle_reset() -> None:
    """The augmented Tabmat slab is unique per live design and rebuilt after pickle."""
    import tabmat

    rng = np.random.default_rng(20260802)
    n = 2_000
    dense = DenseGroupMatrix(rng.normal(size=(n, 2)))
    support = rng.normal(size=(16, 3))
    discrete = DiscretizedSSPGroupMatrix(
        support,
        np.eye(3),
        np.arange(n, dtype=np.intp) % len(support),
    )
    categorical = CategoricalGroupMatrix(
        np.arange(n, dtype=np.intp) % 20,
        n_levels=20,
    )
    dm = DesignMatrix(
        [dense, discrete, categorical],
        n=n,
        p=dense.shape[1] + discrete.shape[1] + categorical.shape[1],
    )

    first = dm.mixed_bin_space_centering_plan
    assert first is dm.mixed_bin_space_centering_plan
    categorical_components = [
        component
        for component in first.split.matrices
        if isinstance(component, tabmat.CategoricalMatrix)
    ]
    assert len(categorical_components) == 2
    assert not any(
        isinstance(component, tabmat.DenseMatrix) and component.shape[1] >= categorical.n_levels
        for component in first.split.matrices
    )

    restored = pickle.loads(pickle.dumps(dm))
    assert restored._mixed_bin_space_centering_plan is None
    assert restored.mixed_bin_space_centering_plan is not first


def test_mixed_bin_space_plan_validates_declared_cached_and_moment_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid declared, mutated, or call-time shapes fail before numerical work."""
    import tabmat

    dm, _X = _mixed_discrete_design()
    with monkeypatch.context() as guarded:
        guarded.setattr(
            tabmat,
            "SplitMatrix",
            lambda *_args, **_kwargs: pytest.fail(
                "declared-shape validation ran after SplitMatrix construction"
            ),
        )
        with pytest.raises(ValueError, match="declared design shape"):
            DesignMatrix(list(dm.group_matrices), n=dm.n, p=dm.p + 1)

    plan = dm.mixed_bin_space_centering_plan
    assert plan is not None
    assert plan.n == dm.n
    assert plan.p == dm.p
    assert plan.shape == dm.shape
    with pytest.raises(ValueError, match="weights and weighted_z"):
        plan.moments(np.ones(dm.n - 1), np.ones(dm.n - 1))

    dm.p += 1
    dm.shape = (dm.n, dm.p)
    with pytest.raises(ValueError, match="cached bin-space plan"):
        _ = dm.mixed_bin_space_centering_plan


def test_mixed_bin_space_centering_preserves_low_cardinality_and_group_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-cardinality categoricals and multiple supports stay native and in solver order."""
    from superglm._group_matrix import _group_matrix_algebra as grouped_algebra
    from superglm._group_matrix import _group_matrix_discretized as discrete_algebra

    rng = np.random.default_rng(20260805)
    n = 16_000
    supports = [
        rng.normal(size=(8, 3)),
        rng.normal(size=(11, 2)),
        rng.normal(size=(9, 2)),
        rng.normal(size=(7, 3)),
    ]
    supports = [support - np.mean(support, axis=0) for support in supports]
    discrete_groups = [
        DiscretizedSSPGroupMatrix(
            support,
            np.eye(support.shape[1]),
            np.arange(n, dtype=np.intp) % len(support),
        )
        for support in supports
    ]
    dense_groups = [DenseGroupMatrix(rng.normal(size=(n, 1))) for _index in range(2)]
    categorical = CategoricalGroupMatrix(
        np.arange(n, dtype=np.intp) % 20,
        n_levels=20,
    )
    group_matrices = [
        discrete_groups[0],
        dense_groups[0],
        discrete_groups[2],
        categorical,
        dense_groups[1],
        discrete_groups[1],
        discrete_groups[3],
    ]
    dm = DesignMatrix(
        group_matrices,
        n=n,
        p=sum(group.shape[1] for group in group_matrices),
    )
    X = np.column_stack([group.toarray() for group in group_matrices])
    W = np.ones(n)
    z = rng.normal(size=n)
    mean_x = np.average(X, axis=0, weights=W)
    mean_z = float(np.average(z, weights=W))
    X_centered = X - mean_x
    calls = _count_tabmat_split_calls(monkeypatch)

    def forbidden_numba(*_args, **_kwargs):
        pytest.fail("accepted low-cardinality bin-space route invoked Numba")

    monkeypatch.setattr(discrete_algebra, "_fused_bincount_2", forbidden_numba)
    monkeypatch.setattr(grouped_algebra, "_weighted_bincount_2d", forbidden_numba)
    monkeypatch.setattr(grouped_algebra, "_disc_disc_2d_hist", forbidden_numba)
    monkeypatch.setattr(grouped_algebra, "_cat_weighted_bincount", forbidden_numba)
    monkeypatch.setattr(
        DiscretizedSSPGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("bin-space route expanded a compressed spline"),
    )
    monkeypatch.setattr(
        CategoricalGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("bin-space route expanded a low-cardinality categorical"),
    )

    state = TabmatCenteringState()
    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
    )

    assert state.eligible is True
    assert calls == {"standardize": 1, "sandwich": 1, "transpose_matvec": 1}
    np.testing.assert_allclose(system.mean_x, mean_x, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        system.data_gram,
        X_centered.T @ (W[:, None] * X_centered),
        rtol=3e-12,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        system.rhs,
        X_centered.T @ (W * (z - mean_z)),
        rtol=3e-12,
        atol=3e-10,
    )


def test_mixed_bin_space_route_excludes_discretized_tensor_subclass() -> None:
    """Tensor support must remain on its existing factored or stable route."""
    from superglm._group_matrix import _group_matrix_centered as centered_algebra

    rng = np.random.default_rng(20260803)
    n = 1_000
    B1 = rng.normal(size=(4, 2))
    B2 = rng.normal(size=(3, 2))
    idx1 = np.arange(n, dtype=np.intp) % len(B1)
    idx2 = (np.arange(n, dtype=np.intp) // len(B1)) % len(B2)
    pair_codes = idx1 * len(B2) + idx2
    observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
    B_joint = (
        B1[observed_codes // len(B2), :, None] * B2[observed_codes % len(B2), None, :]
    ).reshape(len(observed_codes), 4)
    tensor = DiscretizedTensorGroupMatrix(
        B1,
        B2,
        idx1,
        idx2,
        B_joint,
        np.eye(4),
        pair_idx.astype(np.intp),
        tensor_id=20260803,
    )
    categorical = CategoricalGroupMatrix(np.arange(n, dtype=np.intp) % 120, n_levels=120)
    dm = DesignMatrix(
        [DenseGroupMatrix(rng.normal(size=(n, 1))), tensor, categorical],
        n=n,
        p=1 + tensor.shape[1] + categorical.shape[1],
    )

    attempted, result = centered_algebra._try_mixed_discrete_centering(
        dm=dm,
        W=np.ones(n),
        z_centered=np.zeros(n),
        sum_w=float(n),
    )

    assert not attempted
    assert result is None
    assert dm._mixed_bin_space_centering_plan is None


@pytest.mark.parametrize(
    "limit_name",
    [
        "_MAX_DENSE_SLAB_BYTES",
        "_MAX_BIN_CODE_BYTES",
        "_MAX_AUGMENTED_GRAM_BYTES",
        "_MAX_SUPPORT_BYTES",
        "_MAX_RETAINED_AUXILIARY_BYTES",
        "_MAX_CONSTRUCTION_AUXILIARY_BYTES",
        "_MAX_TRANSIENT_AUXILIARY_BYTES",
    ],
)
def test_mixed_bin_space_plan_rejects_memory_estimate_before_build(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
) -> None:
    """Every retained or per-call allocation bound is checked before Tabmat construction."""
    import tabmat

    from superglm._group_matrix import _group_matrix_bin_space as bin_space

    dm, _X = _mixed_discrete_design()
    monkeypatch.setattr(bin_space, limit_name, 1)
    monkeypatch.setattr(
        tabmat,
        "SplitMatrix",
        lambda *_args, **_kwargs: pytest.fail("memory guard ran after SplitMatrix construction"),
    )
    for group_type in (
        DenseGroupMatrix,
        DiscretizedSSPGroupMatrix,
        CategoricalGroupMatrix,
    ):
        monkeypatch.setattr(
            group_type,
            "toarray",
            lambda _self: pytest.fail("memory guard materialized a group matrix"),
        )

    assert dm.mixed_bin_space_centering_plan is None
    assert dm._mixed_bin_space_centering_plan is None


def test_mixed_bin_space_estimates_category_metadata_and_construction_copies() -> None:
    """Hard memory bounds include retained categories and peak constructor copies."""
    dm, _X = _mixed_discrete_design()
    plan = dm.mixed_bin_space_centering_plan
    assert plan is not None
    dense, discrete, categorical = dm.group_matrices
    augmented_columns = dense.shape[1] + discrete.n_bins + categorical.n_levels
    dense_bytes = dm.n * dense.shape[1] * np.dtype(np.float64).itemsize
    code_bytes = dm.n * 2 * np.dtype(np.int32).itemsize
    support_bytes = discrete.n_bins * discrete.shape[1] * np.dtype(np.float64).itemsize
    category_bytes = (discrete.n_bins + categorical.n_levels + 1) * np.dtype(np.int32).itemsize
    index_bytes = (2 * augmented_columns + 2 * dm.p) * np.dtype(np.intp).itemsize
    expected_retained = dense_bytes + code_bytes + support_bytes + category_bytes + index_bytes
    construction_scratch = max(
        dense_bytes,
        dm.n * np.dtype(np.int32).itemsize
        + (categorical.n_levels + 1) * np.dtype(np.int32).itemsize,
        3 * augmented_columns * np.dtype(np.intp).itemsize,
    )

    assert plan.retained_bytes_estimate == expected_retained
    assert plan.construction_bytes_estimate == expected_retained + construction_scratch


def test_mixed_bin_space_transient_guard_counts_asymmetric_pair_intermediate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pair transforms count the previous solver width by current bin count."""
    import tabmat

    from superglm._group_matrix import _group_matrix_bin_space as bin_space

    n = 1_000
    previous_bins, previous_width = 100, 99
    current_bins, current_width = 170, 1
    previous = DiscretizedSSPGroupMatrix(
        np.ones((previous_bins, previous_width)),
        np.eye(previous_width),
        np.arange(n, dtype=np.intp) % previous_bins,
    )
    current = DiscretizedSSPGroupMatrix(
        np.ones((current_bins, current_width)),
        np.eye(current_width),
        np.arange(n, dtype=np.intp) % current_bins,
    )
    groups = [DenseGroupMatrix(np.ones((n, 1))), previous, current]
    p = sum(group.shape[1] for group in groups)
    augmented_columns = 1 + previous_bins + current_bins
    solver_moments = (p * p + 2 * p) * np.dtype(np.float64).itemsize
    augmented_moments = (augmented_columns * augmented_columns + 2 * augmented_columns) * np.dtype(
        np.float64
    ).itemsize
    previous_diagonal = (
        previous_bins**2 + previous_bins * previous_width + previous_width**2
    ) * np.dtype(np.float64).itemsize
    pair_intermediate = (
        previous_bins * current_bins
        + previous_width * current_bins
        + previous_width * current_width
    ) * np.dtype(np.float64).itemsize
    assert pair_intermediate > previous_diagonal
    monkeypatch.setattr(
        bin_space,
        "_MAX_TRANSIENT_AUXILIARY_BYTES",
        solver_moments + augmented_moments + (previous_diagonal + pair_intermediate) // 2,
    )
    monkeypatch.setattr(
        tabmat,
        "SplitMatrix",
        lambda *_args, **_kwargs: pytest.fail(
            "asymmetric pair scratch guard ran after SplitMatrix construction"
        ),
    )
    dm = DesignMatrix(groups, n=n, p=p)

    assert dm.mixed_bin_space_centering_plan is None


def test_mixed_bin_space_transient_guard_counts_tabmat_cross_result_liveness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SplitMatrix sandwich can retain the previous cross result while building the next."""
    import tabmat

    n = 1_000
    n_bins = 690
    groups = [DenseGroupMatrix(np.ones((n, 1)))]
    groups.extend(
        DiscretizedSSPGroupMatrix(
            np.ones((n_bins, 1)),
            np.eye(1),
            np.arange(n, dtype=np.intp) % n_bins,
        )
        for _index in range(4)
    )
    dm = DesignMatrix(groups, n=n, p=5)
    monkeypatch.setattr(
        tabmat,
        "SplitMatrix",
        lambda *_args, **_kwargs: pytest.fail(
            "sandwich cross-result liveness guard ran after SplitMatrix construction"
        ),
    )

    assert dm.mixed_bin_space_centering_plan is None


def test_mixed_bin_space_rejected_plan_is_attempted_once_and_pickle_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-budget ``None`` result is cached without surviving serialization."""
    import superglm.group_matrix as group_matrix_module
    from superglm._group_matrix import _group_matrix_bin_space as bin_space

    dm, _X = _mixed_discrete_design()
    monkeypatch.setattr(bin_space, "_MAX_RETAINED_AUXILIARY_BYTES", 1)
    build_calls = 0
    original_build = group_matrix_module.build_mixed_bin_space_centering_plan

    def counted_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(
        group_matrix_module,
        "build_mixed_bin_space_centering_plan",
        counted_build,
    )

    assert dm.mixed_bin_space_centering_plan is None
    assert dm.mixed_bin_space_centering_plan is None
    assert build_calls == 1
    assert dm._mixed_bin_space_centering_plan_attempted

    restored = pickle.loads(pickle.dumps(dm))
    assert restored._mixed_bin_space_centering_plan is None
    assert not restored._mixed_bin_space_centering_plan_attempted
    assert restored.mixed_bin_space_centering_plan is None
    assert build_calls == 2


def test_mixed_bin_space_plan_rejects_multiple_ordinary_categoricals_before_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supported topology has at most one ordinary categorical block."""
    import tabmat

    rng = np.random.default_rng(20260804)
    n = 1_000
    support = rng.normal(size=(8, 3))
    discrete = DiscretizedSSPGroupMatrix(
        support,
        np.eye(3),
        np.arange(n, dtype=np.intp) % len(support),
    )
    categoricals = [
        CategoricalGroupMatrix(np.arange(n, dtype=np.intp) % levels, n_levels=levels)
        for levels in (7, 11)
    ]
    dm = DesignMatrix(
        [DenseGroupMatrix(rng.normal(size=(n, 1))), discrete, *categoricals],
        n=n,
        p=1 + discrete.shape[1] + sum(group.shape[1] for group in categoricals),
    )
    monkeypatch.setattr(
        tabmat,
        "SplitMatrix",
        lambda *_args, **_kwargs: pytest.fail("topology guard ran after SplitMatrix construction"),
    )

    assert dm.mixed_bin_space_centering_plan is None
    assert dm._mixed_bin_space_centering_plan is None


def test_mixed_bin_space_plan_does_not_treat_zero_width_categorical_as_ordinary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-base zero-column categorical cannot make a compressed-only plan mixed."""
    import tabmat

    rng = np.random.default_rng(20260806)
    n = 1_000
    support = rng.normal(size=(8, 3))
    discrete = DiscretizedSSPGroupMatrix(
        support,
        np.eye(3),
        np.arange(n, dtype=np.intp) % len(support),
    )
    empty_categorical = CategoricalGroupMatrix(np.full(n, -1, dtype=np.intp), n_levels=0)
    dm = DesignMatrix([discrete, empty_categorical], n=n, p=discrete.shape[1])
    monkeypatch.setattr(
        tabmat,
        "SplitMatrix",
        lambda *_args, **_kwargs: pytest.fail("zero-width topology built a SplitMatrix"),
    )

    assert dm.mixed_bin_space_centering_plan is None


def test_mixed_bin_space_preflight_reuses_cached_compressed_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing weights reuses call-local Tabmat masses and cached support."""
    from superglm._group_matrix import _group_matrix_centered as centered_algebra

    dm, _X = _mixed_discrete_design()
    plan = dm.mixed_bin_space_centering_plan
    assert plan is not None

    def forbidden_matmul(*_args, **_kwargs):
        pytest.fail("mixed preflight recomputed B_unique @ R_inv")

    def forbidden_bincount(*_args, **_kwargs):
        pytest.fail("mixed preflight rescanned compressed bin codes")

    monkeypatch.setattr(centered_algebra.np, "matmul", forbidden_matmul)
    monkeypatch.setattr(centered_algebra.np, "bincount", forbidden_bincount)
    for W in (np.ones(dm.n), np.linspace(0.5, 1.5, dm.n)):
        assert (
            centered_algebra._mixed_raw_centering_preflight(
                plan=plan,
                W=W,
                sum_w=float(np.sum(W)),
            )
            is not None
        )


def test_mixed_bin_space_plan_supports_categorical_as_only_ordinary_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One native categorical can anchor a mixed plan without a retained dense slab."""
    import tabmat

    from superglm._group_matrix import _group_matrix_centered as centered_algebra

    rng = np.random.default_rng(20260807)
    n = 5_000
    support = rng.normal(size=(16, 3))
    support -= np.mean(support, axis=0)
    discrete = DiscretizedSSPGroupMatrix(
        support,
        np.eye(3),
        np.arange(n, dtype=np.intp) % len(support),
    )
    raw_codes = np.arange(n, dtype=np.intp) % 21
    categorical = CategoricalGroupMatrix(np.where(raw_codes == 20, -1, raw_codes), n_levels=20)
    dm = DesignMatrix([categorical, discrete], n=n, p=categorical.shape[1] + discrete.shape[1])
    X = np.column_stack((categorical.toarray(), discrete.toarray()))
    W = rng.uniform(0.25, 2.0, size=n)
    z = rng.normal(size=n)
    z_centered = z - np.average(z, weights=W)
    calls = _count_tabmat_split_calls(monkeypatch)

    attempted, centered = centered_algebra._try_mixed_discrete_centering(
        dm=dm,
        W=W,
        z_centered=z_centered,
        sum_w=float(np.sum(W)),
    )

    assert attempted
    assert centered is not None
    mean_x, gram, rhs = centered
    plan = dm.mixed_bin_space_centering_plan
    assert plan is not None
    assert all(isinstance(component, tabmat.CategoricalMatrix) for component in plan.split.matrices)
    assert not any(isinstance(component, tabmat.DenseMatrix) for component in plan.split.matrices)
    assert calls == {"standardize": 1, "sandwich": 1, "transpose_matvec": 1}
    expected_mean = np.average(X, axis=0, weights=W)
    X_centered = X - expected_mean
    np.testing.assert_allclose(mean_x, expected_mean, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(gram, X_centered.T @ (W[:, None] * X_centered), rtol=3e-12)
    np.testing.assert_allclose(rhs, X_centered.T @ (W * z_centered), rtol=3e-12)


def test_unmeasured_mixed_layouts_do_not_mutate_centering_route_state() -> None:
    """Sparse, all-support, and under-floor layouts stay on the stable route."""
    from superglm._group_matrix import _group_matrix_centered as centered_algebra

    rng = np.random.default_rng(20260731)

    def discrete_group(n: int, width: int = 3) -> DiscretizedSSPGroupMatrix:
        support = rng.normal(size=(8, width))
        support -= np.mean(support, axis=0)
        return DiscretizedSSPGroupMatrix(
            support,
            np.eye(width),
            np.arange(n, dtype=np.intp) % len(support),
        )

    n = 25_000
    sparse = SparseGroupMatrix(sp.csr_matrix(rng.normal(size=(n, 1))))
    sparse_dm = DesignMatrix([sparse, discrete_group(n)], n=n, p=4)

    support_n = 20_000
    exact_support = SparseSSPGroupMatrix(
        sp.csr_matrix(rng.normal(size=(support_n, 2))),
        np.eye(2),
    )
    support_dm = DesignMatrix(
        [exact_support, discrete_group(support_n)],
        n=support_n,
        p=5,
    )

    scaled_floor_n = 20_000
    scaled_floor_dm = DesignMatrix(
        [DenseGroupMatrix(rng.normal(size=(scaled_floor_n, 1)))]
        + [discrete_group(scaled_floor_n) for _index in range(4)],
        n=scaled_floor_n,
        p=13,
    )

    for dm in (sparse_dm, support_dm, scaled_floor_dm):
        attempted, result = centered_algebra._try_mixed_discrete_centering(
            dm=dm,
            W=np.ones(dm.n),
            z_centered=np.zeros(dm.n),
            sum_w=float(dm.n),
        )
        assert not attempted
        assert result is None
        assert dm._mixed_bin_space_centering_plan is None
        assert not hasattr(dm, "_mixed_centering_execution_plan")


def test_small_mixed_discrete_design_stays_on_stable_centered_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured size gate must avoid raw-moment overhead on small designs."""
    from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan

    dm, X = _mixed_discrete_design(n=799)
    rng = np.random.default_rng(20260725)
    W = rng.uniform(0.25, 2.0, size=dm.n)
    z = rng.normal(size=dm.n)
    state = TabmatCenteringState()
    mean_x = np.average(X, axis=0, weights=W)
    mean_z = float(np.average(z, weights=W))
    X_centered = X - mean_x

    monkeypatch.setattr(
        MatrixExecutionPlan,
        "_moments_prevalidated",
        lambda *_args, **_kwargs: pytest.fail("small mixed design crossed the raw-moment gate"),
    )
    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
    )

    assert state.eligible is None
    np.testing.assert_allclose(
        system.data_gram,
        X_centered.T @ (W[:, None] * X_centered),
        rtol=2e-12,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        system.rhs,
        X_centered.T @ (W * (z - mean_z)),
        rtol=2e-12,
        atol=2e-11,
    )


def test_small_low_cardinality_mixed_design_stays_on_stable_route() -> None:
    """Native low-cardinality setup cost must stay behind its measured row crossover."""
    from superglm._group_matrix import _group_matrix_centered as centered_algebra

    rng = np.random.default_rng(20260809)
    n = 2_000
    support = rng.normal(size=(16, 17))
    support -= np.mean(support, axis=0)
    discrete = DiscretizedSSPGroupMatrix(
        support,
        np.eye(17),
        np.arange(n, dtype=np.intp) % len(support),
    )
    categorical = CategoricalGroupMatrix(
        np.arange(n, dtype=np.intp) % 40,
        n_levels=40,
    )
    dm = DesignMatrix(
        [DenseGroupMatrix(rng.normal(size=(n, 1))), discrete, categorical],
        n=n,
        p=1 + discrete.shape[1] + categorical.shape[1],
    )
    assert dm.n * dm.p == 116_000

    attempted, centered = centered_algebra._try_mixed_discrete_centering(
        dm=dm,
        W=np.ones(n),
        z_centered=np.zeros(n),
        sum_w=float(n),
    )

    assert not attempted
    assert centered is None
    assert dm._mixed_bin_space_centering_plan is None


@pytest.mark.parametrize("failure_mode", ["constant", "large_location", "overflow"])
def test_unsafe_mixed_discrete_preflight_locks_out_raw_route(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    """An unsafe block is rejected before a full Gram and never retried in the fit."""
    from superglm._group_matrix import _group_matrix_centered as centered_algebra
    from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan

    dm, X = _mixed_discrete_design()
    dense = dm.group_matrices[0]
    assert isinstance(dense, DenseGroupMatrix)
    rng = np.random.default_rng(20260726)
    if failure_mode == "constant":
        dense.M[:, 0] = 2.0
    elif failure_mode == "large_location":
        dense.M[:, 0] = 1e12 + rng.normal(size=dm.n)
    else:
        dense.M[:, 0] = 1e155 + 1e145 * rng.normal(size=dm.n)
    X[:, :2] = dense.M
    W = rng.uniform(0.25, 2.0, size=dm.n)
    z = rng.normal(size=dm.n)
    state = TabmatCenteringState()
    preflight_calls = 0
    row_subset_calls = 0
    original_preflight = centered_algebra._mixed_raw_centering_preflight
    original_row_subset = dm.row_subset

    def counted_preflight(*args, **kwargs):
        nonlocal preflight_calls
        preflight_calls += 1
        return original_preflight(*args, **kwargs)

    def counted_row_subset(rows):
        nonlocal row_subset_calls
        row_subset_calls += 1
        return original_row_subset(rows)

    monkeypatch.setattr(centered_algebra, "_mixed_raw_centering_preflight", counted_preflight)
    monkeypatch.setattr(dm, "row_subset", counted_row_subset)
    monkeypatch.setattr(
        MatrixExecutionPlan,
        "_moments_prevalidated",
        lambda *_args, **_kwargs: pytest.fail("unsafe preflight built a full raw Gram"),
    )

    with np.errstate(over="raise", invalid="raise"):
        first = build_centered_system(
            dm=dm,
            W=W,
            z_off=z,
            penalty=np.zeros((dm.p, dm.p)),
            tabmat_state=state,
        )
        second = build_centered_system(
            dm=dm,
            W=W,
            z_off=z,
            penalty=np.zeros((dm.p, dm.p)),
            tabmat_state=state,
        )

    assert state.eligible is False
    assert preflight_calls == 1
    assert row_subset_calls == 2
    assert np.all(np.isfinite(first.data_gram))
    np.testing.assert_array_equal(second.data_gram, first.data_gram)
    np.testing.assert_array_equal(second.rhs, first.rhs)


def test_accepted_mixed_discrete_route_recertifies_changed_weights_and_locks_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later weights skip preflight but retain authoritative certificate lockout."""
    from superglm._group_matrix import _group_matrix_centered as centered_algebra
    from superglm._group_matrix._group_matrix_bin_space import MixedBinSpaceCenteringPlan

    dm, X = _mixed_discrete_design()
    rng = np.random.default_rng(20260727)
    z = rng.normal(size=dm.n)
    W_safe = np.ones(dm.n)
    W_changed_safe = rng.uniform(0.75, 1.25, size=dm.n)
    W_degenerate = np.zeros(dm.n)
    W_degenerate[0] = 1.0
    state = TabmatCenteringState()
    preflight_calls = 0
    moment_calls = 0
    original_preflight = centered_algebra._mixed_raw_centering_preflight
    original_moments = MixedBinSpaceCenteringPlan.moments
    calls = _count_tabmat_split_calls(monkeypatch)

    def counted_preflight(*args, **kwargs):
        nonlocal preflight_calls
        preflight_calls += 1
        return original_preflight(*args, **kwargs)

    def counted_moments(self, *args, **kwargs):
        nonlocal moment_calls
        moment_calls += 1
        return original_moments(self, *args, **kwargs)

    monkeypatch.setattr(centered_algebra, "_mixed_raw_centering_preflight", counted_preflight)
    monkeypatch.setattr(MixedBinSpaceCenteringPlan, "moments", counted_moments)

    first = build_centered_system(
        dm=dm,
        W=W_safe,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
    )
    assert state.eligible is True

    changed = build_centered_system(
        dm=dm,
        W=W_changed_safe,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
    )
    assert state.eligible is True
    changed_mean_x = np.average(X, axis=0, weights=W_changed_safe)
    changed_centered = X - changed_mean_x
    np.testing.assert_allclose(
        changed.data_gram,
        changed_centered.T @ (W_changed_safe[:, None] * changed_centered),
        rtol=2e-12,
        atol=2e-11,
    )

    rejected = build_centered_system(
        dm=dm,
        W=W_degenerate,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
    )
    assert state.eligible is False
    np.testing.assert_allclose(rejected.data_gram, 0.0, atol=1e-13)
    np.testing.assert_allclose(rejected.rhs, 0.0, atol=1e-13)

    third = build_centered_system(
        dm=dm,
        W=W_safe,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
    )
    mean_x = np.mean(X, axis=0)
    mean_z = float(np.mean(z))
    X_centered = X - mean_x
    np.testing.assert_allclose(third.data_gram, X_centered.T @ X_centered, rtol=2e-12)
    np.testing.assert_allclose(
        third.rhs,
        X_centered.T @ (z - mean_z),
        rtol=2e-12,
        atol=2e-11,
    )
    np.testing.assert_allclose(first.data_gram, third.data_gram, rtol=2e-12, atol=2e-11)
    assert state.eligible is False
    assert preflight_calls == 1
    assert moment_calls == 3
    assert calls == {"standardize": 1, "sandwich": 3, "transpose_matvec": 3}


def test_mixed_bin_space_later_iteration_derives_dense_xtw_without_weight_transpose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nonbinary dense means cannot be confused with sandwich diagonal moments."""
    from superglm._group_matrix import _group_matrix_centered as centered_algebra

    rng = np.random.default_rng(20260808)
    n = 25_000
    dense_values = rng.normal(size=(n, 3)) * np.array([0.5, 1.5, 3.0])
    support = rng.normal(size=(12, 2))
    support -= np.mean(support, axis=0)
    discrete = DiscretizedSSPGroupMatrix(
        support,
        np.eye(2),
        np.arange(n, dtype=np.intp) % len(support),
    )
    dm = DesignMatrix(
        [DenseGroupMatrix(dense_values), discrete],
        n=n,
        p=dense_values.shape[1] + discrete.shape[1],
    )
    X = np.column_stack((dense_values, discrete.toarray()))
    W = rng.uniform(0.2, 2.0, size=n)
    z = rng.normal(size=n)
    z_centered = z - np.average(z, weights=W)
    calls = _count_tabmat_split_calls(monkeypatch)

    attempted, centered = centered_algebra._try_mixed_discrete_centering(
        dm=dm,
        W=W,
        z_centered=z_centered,
        sum_w=float(np.sum(W)),
        preflight=False,
    )

    assert attempted
    assert centered is not None
    mean_x, gram, rhs = centered
    expected_mean = np.average(X, axis=0, weights=W)
    X_centered = X - expected_mean
    np.testing.assert_allclose(mean_x, expected_mean, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(gram, X_centered.T @ (W[:, None] * X_centered), rtol=3e-12)
    np.testing.assert_allclose(rhs, X_centered.T @ (W * z_centered), rtol=3e-12)
    assert calls == {"standardize": 0, "sandwich": 1, "transpose_matvec": 1}


def test_mixed_discrete_tensor_centering_stays_on_stable_route() -> None:
    """A mixed tensor is structurally excluded and leaves route state untouched."""
    rng = np.random.default_rng(20260728)
    n = 800
    dense_values = rng.normal(size=(n, 2))
    dense_values -= np.mean(dense_values, axis=0)
    B1 = rng.normal(size=(4, 2))
    B2 = rng.normal(size=(3, 2))
    B1 -= np.mean(B1, axis=0)
    B2 -= np.mean(B2, axis=0)
    idx1 = np.arange(n, dtype=np.intp) % len(B1)
    idx2 = (np.arange(n, dtype=np.intp) // len(B1)) % len(B2)
    pair_codes = idx1 * len(B2) + idx2
    observed_codes, pair_idx = np.unique(pair_codes, return_inverse=True)
    B_joint = (
        B1[observed_codes // len(B2), :, None] * B2[observed_codes % len(B2), None, :]
    ).reshape(len(observed_codes), 4)
    R_inv = rng.normal(size=(4, 3))
    tensor = DiscretizedTensorGroupMatrix(
        B1,
        B2,
        idx1,
        idx2,
        B_joint,
        R_inv,
        pair_idx.astype(np.intp),
        tensor_id=20260728,
    )
    categorical = CategoricalGroupMatrix(
        np.arange(n, dtype=np.intp) % 120,
        n_levels=120,
    )
    dm = DesignMatrix(
        [DenseGroupMatrix(dense_values), tensor, categorical],
        n=n,
        p=2 + tensor.shape[1] + categorical.shape[1],
    )
    X = np.column_stack((dense_values, tensor.toarray(), categorical.toarray()))
    W = rng.uniform(0.25, 2.0, size=n)
    z = rng.normal(size=n)
    mean_x = np.average(X, axis=0, weights=W)
    mean_z = float(np.average(z, weights=W))
    X_centered = X - mean_x
    state = TabmatCenteringState()

    system = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
    )

    assert state.eligible is None
    assert dm._mixed_bin_space_centering_plan is None
    np.testing.assert_allclose(
        system.data_gram,
        X_centered.T @ (W[:, None] * X_centered),
        rtol=3e-12,
        atol=3e-11,
    )
    np.testing.assert_allclose(
        system.rhs,
        X_centered.T @ (W * (z - mean_z)),
        rtol=3e-12,
        atol=3e-11,
    )
