from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import tabmat

from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
)


def _mixed_groups(seed: int = 170):
    rng = np.random.default_rng(seed)
    n = 720
    dense = DenseGroupMatrix(rng.normal(size=(n, 2)))

    sparse_values = rng.normal(size=(n, 3))
    sparse_values[rng.random(size=sparse_values.shape) < 0.9] = 0.0
    sparse = SparseGroupMatrix(sp.csr_matrix(sparse_values))

    codes = np.resize(np.arange(120, dtype=np.intp), n)
    rng.shuffle(codes)
    codes[::37] = -1
    categorical = CategoricalGroupMatrix(codes, n_levels=120)

    n_bins = 32
    bin_idx = rng.integers(0, n_bins, size=n, dtype=np.intp)
    bin_idx[:n_bins] = np.arange(n_bins)
    discrete = DiscretizedSSPGroupMatrix(
        rng.normal(size=(n_bins, 4)),
        rng.normal(size=(4, 3)),
        bin_idx,
    )

    n_bins1, n_bins2 = 8, 6
    B1 = rng.normal(size=(n_bins1, 2))
    B2 = rng.normal(size=(n_bins2, 2))
    idx1 = rng.integers(0, n_bins1, size=n, dtype=np.intp)
    idx2 = rng.integers(0, n_bins2, size=n, dtype=np.intp)
    pair_idx = idx1 * n_bins2 + idx2
    B_joint = np.einsum("ia,jb->ijab", B1, B2).reshape(n_bins1 * n_bins2, 4)
    tensor = DiscretizedTensorGroupMatrix(
        B1,
        B2,
        idx1,
        idx2,
        B_joint,
        np.eye(4),
        pair_idx,
        tensor_id=1,
    )
    groups = [dense, sparse, categorical, discrete, tensor]
    X = np.column_stack([group.toarray() for group in groups])
    return rng, groups, X


def _count_tabmat_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"sandwich": 0, "transpose_matvec": 0}
    original_sandwich = tabmat.SplitMatrix.sandwich
    original_transpose = tabmat.SplitMatrix.transpose_matvec

    def counted_sandwich(self, *args, **kwargs):
        calls["sandwich"] += 1
        return original_sandwich(self, *args, **kwargs)

    def counted_transpose(self, *args, **kwargs):
        calls["transpose_matvec"] += 1
        return original_transpose(self, *args, **kwargs)

    monkeypatch.setattr(tabmat.SplitMatrix, "sandwich", counted_sandwich)
    monkeypatch.setattr(tabmat.SplitMatrix, "transpose_matvec", counted_transpose)
    return calls


@pytest.mark.parametrize("signed", [False, True])
def test_mixed_execution_plan_moments_match_dense_without_materializing_compressed_rows(
    signed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng, groups, X = _mixed_groups()
    W = rng.normal(size=X.shape[0]) if signed else rng.uniform(0.0, 2.0, size=X.shape[0])
    W[::41] = 0.0
    rhs = (rng.normal(size=X.shape[0]), rng.normal(size=X.shape[0]))
    plan = MatrixExecutionPlan(groups, n=X.shape[0], ordinary_tabmat=True)
    calls = _count_tabmat_calls(monkeypatch)

    monkeypatch.setattr(
        DiscretizedSSPGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("the execution plan must keep compressed rows compressed"),
    )
    moments = plan.moments(W, rhs=rhs, include_xtw=True, signed=signed)

    assert calls == {"sandwich": 1, "transpose_matvec": 3}
    np.testing.assert_allclose(moments.gram, X.T @ (W[:, None] * X), rtol=2e-12, atol=2e-10)
    assert moments.xtw is not None
    np.testing.assert_allclose(moments.xtw, X.T @ W, rtol=2e-12, atol=2e-11)
    assert len(moments.xt_rhs) == 2
    for actual, vector in zip(moments.xt_rhs, rhs, strict=True):
        np.testing.assert_allclose(actual, X.T @ vector, rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("layout", ["strided", "readonly"])
def test_execution_plan_normalizes_all_tabmat_vector_buffers(layout: str) -> None:
    rng, groups, X = _mixed_groups(seed=171)
    base_W = rng.uniform(0.0, 2.0, size=X.shape[0])
    base_rhs = rng.normal(size=X.shape[0])
    if layout == "strided":
        W_storage = np.empty(2 * len(base_W))
        rhs_storage = np.empty(2 * len(base_rhs))
        W_storage[::2] = base_W
        rhs_storage[::2] = base_rhs
        W = W_storage[::2]
        rhs = rhs_storage[::2]
    else:
        W = base_W.copy()
        rhs = base_rhs.copy()
        W.setflags(write=False)
        rhs.setflags(write=False)

    moments = MatrixExecutionPlan(groups, n=X.shape[0], ordinary_tabmat=True).moments(
        W,
        rhs=(rhs,),
        include_xtw=True,
    )

    np.testing.assert_allclose(moments.gram, X.T @ (base_W[:, None] * X), atol=2e-10)
    np.testing.assert_allclose(moments.xtw, X.T @ base_W, atol=2e-11)
    np.testing.assert_allclose(moments.xt_rhs[0], X.T @ base_rhs, atol=2e-11)


def test_auto_plan_keeps_faster_specialized_assembly_for_mixed_discrete_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng, groups, X = _mixed_groups(seed=173)
    W = rng.uniform(0.0, 2.0, size=X.shape[0])
    rhs = rng.normal(size=X.shape[0])
    calls = _count_tabmat_calls(monkeypatch)

    moments = MatrixExecutionPlan(groups, n=X.shape[0]).moments(
        W,
        rhs=(rhs,),
        include_xtw=True,
    )

    assert calls == {"sandwich": 0, "transpose_matvec": 0}
    np.testing.assert_allclose(moments.gram, X.T @ (W[:, None] * X), atol=2e-10)
    np.testing.assert_allclose(moments.xtw, X.T @ W, atol=2e-11)
    np.testing.assert_allclose(moments.xt_rhs[0], X.T @ rhs, atol=2e-11)


def test_execution_plan_fuses_compressed_gram_and_transpose_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng, groups, _X = _mixed_groups(seed=174)
    discrete = groups[3]
    X = discrete.toarray()
    W = rng.uniform(0.0, 2.0, size=X.shape[0])
    rhs = rng.normal(size=X.shape[0])

    monkeypatch.setattr(
        DiscretizedSSPGroupMatrix,
        "gram",
        lambda _self, _W: pytest.fail("plan should use fused gram_rmatvec"),
    )
    monkeypatch.setattr(
        DiscretizedSSPGroupMatrix,
        "rmatvec",
        lambda _self, _v: pytest.fail("plan should use fused gram_rmatvec"),
    )
    moments = MatrixExecutionPlan([discrete], n=X.shape[0]).moments(
        W,
        rhs=(rhs,),
        include_xtw=True,
    )

    np.testing.assert_allclose(moments.gram, X.T @ (W[:, None] * X), atol=2e-10)
    np.testing.assert_allclose(moments.xtw, X.T @ W, atol=2e-11)
    np.testing.assert_allclose(moments.xt_rhs[0], X.T @ rhs, atol=2e-11)


def test_execution_plan_never_materializes_sparse_support_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(175)
    n = 720
    B = sp.random(n, 6, density=0.2, format="csr", random_state=rng)
    support = SparseSSPGroupMatrix(B, rng.normal(size=(6, 3)))
    codes = np.resize(np.arange(120, dtype=np.intp), n)
    rng.shuffle(codes)
    categorical = CategoricalGroupMatrix(codes, n_levels=120)
    X = np.column_stack((support.toarray(), categorical.toarray()))
    W = rng.uniform(0.0, 2.0, size=n)
    rhs = rng.normal(size=n)

    monkeypatch.setattr(
        SparseSSPGroupMatrix,
        "toarray",
        lambda _self: pytest.fail("support-space rows must remain factored"),
    )
    moments = MatrixExecutionPlan(
        [support, categorical],
        n=n,
        ordinary_tabmat=True,
    ).moments(W, rhs=(rhs,), include_xtw=True)

    np.testing.assert_allclose(moments.gram, X.T @ (W[:, None] * X), atol=2e-10)
    np.testing.assert_allclose(moments.xtw, X.T @ W, atol=2e-11)
    np.testing.assert_allclose(moments.xt_rhs[0], X.T @ rhs, atol=2e-11)


def test_execution_plan_scatters_interleaved_ordinary_columns_and_auto_accepts_ordinary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng, groups, _X = _mixed_groups(seed=176)
    interleaved = [groups[index] for index in (0, 3, 1, 4, 2)]
    X = np.column_stack([group.toarray() for group in interleaved])
    W = rng.uniform(0.0, 2.0, size=X.shape[0])
    rhs = rng.normal(size=X.shape[0])
    calls = _count_tabmat_calls(monkeypatch)

    forced = MatrixExecutionPlan(
        interleaved,
        n=X.shape[0],
        ordinary_tabmat=True,
    ).moments(W, rhs=(rhs,), include_xtw=True)

    assert calls == {"sandwich": 1, "transpose_matvec": 2}
    np.testing.assert_allclose(forced.gram, X.T @ (W[:, None] * X), atol=2e-10)
    np.testing.assert_allclose(forced.xtw, X.T @ W, atol=2e-11)
    np.testing.assert_allclose(forced.xt_rhs[0], X.T @ rhs, atol=2e-11)

    calls.update(sandwich=0, transpose_matvec=0)
    ordinary = [groups[index] for index in (0, 2, 1)]
    MatrixExecutionPlan(ordinary, n=X.shape[0]).moments(W, rhs=(rhs,), include_xtw=True)
    assert calls == {"sandwich": 1, "transpose_matvec": 2}


def test_design_matrix_caches_one_immutable_execution_plan() -> None:
    _rng, groups, X = _mixed_groups(seed=172)
    dm = DesignMatrix(groups, n=X.shape[0], p=X.shape[1])

    first = dm.execution_plan
    second = dm.execution_plan

    assert first is second
    assert first.shape == dm.shape
    assert first.group_spans == tuple(first.group_spans)
    with pytest.raises(TypeError):
        dm.group_matrices[0] = DenseGroupMatrix(np.zeros_like(groups[0].toarray()))
    with pytest.raises(AttributeError):
        first.shape = (0, 0)
    with pytest.raises(AttributeError):
        first.group_matrices = ()
