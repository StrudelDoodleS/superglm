from __future__ import annotations

import pickle
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import tabmat

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm._group_matrix._group_matrix_execution import GroupSpan, MatrixExecutionPlan
from superglm._group_matrix._group_matrix_tabmat import _build_tabmat_split
from superglm.distributions import Poisson
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.links import LogLink
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.types import GroupSlice


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


def _count_tabmat_vector_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"matvec": 0, "transpose_matvec": 0}
    original_matvec = tabmat.SplitMatrix.matvec
    original_transpose = tabmat.SplitMatrix.transpose_matvec

    def counted_matvec(self, *args, **kwargs):
        calls["matvec"] += 1
        return original_matvec(self, *args, **kwargs)

    def counted_transpose(self, *args, **kwargs):
        calls["transpose_matvec"] += 1
        return original_transpose(self, *args, **kwargs)

    monkeypatch.setattr(tabmat.SplitMatrix, "matvec", counted_matvec)
    monkeypatch.setattr(tabmat.SplitMatrix, "transpose_matvec", counted_transpose)
    return calls


def _ordinary_vector_groups(
    *,
    n: int,
    dense_blocks: int,
    seed: int,
    categorical_blocks: int = 1,
) -> tuple[np.random.Generator, list, np.ndarray]:
    rng = np.random.default_rng(seed)
    dense = [DenseGroupMatrix(rng.normal(size=n)) for _ in range(dense_blocks)]
    categoricals = [
        CategoricalGroupMatrix(
            rng.integers(-1, 160, size=n, dtype=np.intp),
            n_levels=160,
        )
        for _ in range(categorical_blocks)
    ]
    groups = [*dense, *categoricals]
    return rng, groups, np.column_stack([group.toarray() for group in groups])


@pytest.mark.parametrize("categorical_blocks", [0, 1, 2])
def test_retained_tabmat_split_accelerates_many_block_design_vectors(
    monkeypatch: pytest.MonkeyPatch,
    categorical_blocks: int,
) -> None:
    rng, groups, X = _ordinary_vector_groups(
        n=10_000,
        dense_blocks=8,
        categorical_blocks=categorical_blocks,
        seed=190 + categorical_blocks,
    )
    dm = DesignMatrix(groups, n=X.shape[0], p=X.shape[1])
    beta = rng.normal(size=X.shape[1])
    rows = rng.normal(size=X.shape[0])
    expected_eta = X @ beta
    expected_score = X.T @ rows
    beta.setflags(write=False)
    rows.setflags(write=False)

    assert dm.tabmat_split is not None
    calls = _count_tabmat_vector_calls(monkeypatch)

    np.testing.assert_allclose(dm.matvec(beta), expected_eta, rtol=2e-14, atol=2e-13)
    np.testing.assert_allclose(dm.rmatvec(rows), expected_score, rtol=2e-14, atol=2e-11)
    assert calls == {"matvec": 1, "transpose_matvec": 1}


def test_retained_tabmat_split_rejects_category_heavy_vector_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(195)
    n = 10_000
    dense_groups = [DenseGroupMatrix(rng.normal(size=n)) for _ in range(8)]
    categorical_groups = [
        CategoricalGroupMatrix(
            rng.integers(-1, 160, size=n, dtype=np.intp),
            n_levels=160,
        )
        for _ in range(4)
    ]
    groups = [*dense_groups, *categorical_groups]
    dm = DesignMatrix(groups, n=n, p=sum(group.shape[1] for group in groups))
    beta = rng.normal(size=dm.p)
    rows = rng.normal(size=n)
    expected_eta = np.zeros(n)
    expected_score = np.empty(dm.p)
    column = 0
    for group in groups:
        width = group.shape[1]
        expected_eta += group.matvec(beta[column : column + width])
        expected_score[column : column + width] = group.rmatvec(rows)
        column += width

    assert dm.tabmat_split is not None
    calls = _count_tabmat_vector_calls(monkeypatch)

    np.testing.assert_allclose(dm.matvec(beta), expected_eta, rtol=2e-14, atol=2e-13)
    np.testing.assert_allclose(dm.rmatvec(rows), expected_score, rtol=2e-14, atol=2e-11)
    assert calls == {"matvec": 0, "transpose_matvec": 0}


@pytest.mark.parametrize(
    ("categorical_blocks", "levels"),
    [(1, 2_000), (2, 1_000)],
)
def test_retained_tabmat_split_rejects_high_cardinality_vector_work(
    monkeypatch: pytest.MonkeyPatch,
    categorical_blocks: int,
    levels: int,
) -> None:
    rng = np.random.default_rng(196 + categorical_blocks)
    n = 10_000
    dense_groups = [DenseGroupMatrix(rng.normal(size=n)) for _ in range(8)]
    categorical_groups = [
        CategoricalGroupMatrix(
            rng.integers(-1, levels, size=n, dtype=np.intp),
            n_levels=levels,
        )
        for _ in range(categorical_blocks)
    ]
    groups = [*dense_groups, *categorical_groups]
    dm = DesignMatrix(groups, n=n, p=sum(group.shape[1] for group in groups))
    beta = rng.normal(size=dm.p)
    rows = rng.normal(size=n)
    expected_eta = np.zeros(n)
    expected_score = np.empty(dm.p)
    column = 0
    for group in groups:
        width = group.shape[1]
        expected_eta += group.matvec(beta[column : column + width])
        expected_score[column : column + width] = group.rmatvec(rows)
        column += width

    assert dm.tabmat_split is not None
    calls = _count_tabmat_vector_calls(monkeypatch)

    np.testing.assert_allclose(dm.matvec(beta), expected_eta, rtol=2e-14, atol=2e-13)
    np.testing.assert_allclose(dm.rmatvec(rows), expected_score, rtol=2e-14, atol=2e-11)
    assert calls == {"matvec": 0, "transpose_matvec": 0}


def test_design_vectors_do_not_build_an_eligible_tabmat_split() -> None:
    rng, groups, X = _ordinary_vector_groups(n=10_000, dense_blocks=8, seed=191)
    dm = DesignMatrix(groups, n=X.shape[0], p=X.shape[1])
    beta = rng.normal(size=X.shape[1])
    rows = rng.normal(size=X.shape[0])

    assert not dm._tabmat_built
    np.testing.assert_allclose(dm.matvec(beta), X @ beta, rtol=2e-14, atol=2e-13)
    np.testing.assert_allclose(dm.rmatvec(rows), X.T @ rows, rtol=2e-14, atol=2e-11)
    assert not dm._tabmat_built


def test_retained_tabmat_split_keeps_one_dense_block_on_group_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(192)
    n = 10_000
    groups = [
        DenseGroupMatrix(rng.normal(size=(n, 8))),
        CategoricalGroupMatrix(rng.integers(-1, 160, size=n), n_levels=160),
    ]
    X = np.column_stack([group.toarray() for group in groups])
    dm = DesignMatrix(groups, n=n, p=X.shape[1])
    beta = rng.normal(size=X.shape[1])
    rows = rng.normal(size=n)

    assert dm.tabmat_split is not None
    calls = _count_tabmat_vector_calls(monkeypatch)

    np.testing.assert_allclose(dm.matvec(beta), X @ beta, rtol=2e-14, atol=2e-13)
    np.testing.assert_allclose(dm.rmatvec(rows), X.T @ rows, rtol=2e-14, atol=2e-11)
    assert calls == {"matvec": 0, "transpose_matvec": 0}


def test_discrete_design_vectors_keep_compressed_group_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(193)
    n = 10_000
    n_bins = 64
    group = DiscretizedSSPGroupMatrix(
        rng.normal(size=(n_bins, 6)),
        rng.normal(size=(6, 4)),
        rng.integers(0, n_bins, size=n, dtype=np.intp),
    )
    dm = DesignMatrix([group], n=n, p=4)
    beta = rng.normal(size=4)
    rows = rng.normal(size=n)
    expected_eta = group.matvec(beta)
    expected_score = group.rmatvec(rows)

    assert dm.tabmat_split is None
    calls = _count_tabmat_vector_calls(monkeypatch)

    np.testing.assert_array_equal(dm.matvec(beta), expected_eta)
    np.testing.assert_array_equal(dm.rmatvec(rows), expected_score)
    assert calls == {"matvec": 0, "transpose_matvec": 0}


def test_retained_tabmat_vectors_preserve_full_fit_state() -> None:
    rng, groups, X = _ordinary_vector_groups(n=10_000, dense_blocks=8, seed=194)
    eta = 0.2 + X[:, :8] @ rng.normal(scale=0.08, size=8)
    y = rng.poisson(np.exp(eta))
    starts = np.cumsum([0, *[group.shape[1] for group in groups]])
    slices = [
        GroupSlice(name=f"g{index}", start=int(starts[index]), end=int(starts[index + 1]))
        for index in range(len(groups))
    ]
    penalty = 0.02 * np.eye(X.shape[1])
    baseline_dm = DesignMatrix(groups, n=X.shape[0], p=X.shape[1])
    candidate_dm = DesignMatrix(groups, n=X.shape[0], p=X.shape[1])
    baseline_dm._tabmat_vector_candidate = False

    baseline, _ = fit_irls_direct(
        baseline_dm,
        y,
        np.ones(len(y)),
        Poisson(),
        LogLink(),
        slices,
        lambda2=0.02,
        S_override=penalty,
        max_iter=20,
        tol=1e-10,
        record_diagnostics=True,
        compute_rank_info=False,
        _compute_fit_statistics=False,
    )
    candidate, _ = fit_irls_direct(
        candidate_dm,
        y,
        np.ones(len(y)),
        Poisson(),
        LogLink(),
        slices,
        lambda2=0.02,
        S_override=penalty,
        max_iter=20,
        tol=1e-10,
        record_diagnostics=True,
        compute_rank_info=False,
        _compute_fit_statistics=False,
    )

    assert candidate_dm._tabmat_vector_candidate is True
    assert candidate.n_iter == baseline.n_iter
    assert candidate.converged == baseline.converged
    assert candidate.deviance == pytest.approx(baseline.deviance, rel=2e-14, abs=2e-11)
    np.testing.assert_allclose(candidate.beta, baseline.beta, rtol=2e-13, atol=2e-13)
    baseline_mu = np.exp(baseline_dm.matvec(baseline.beta) + baseline.intercept)
    candidate_mu = np.exp(candidate_dm.matvec(candidate.beta) + candidate.intercept)
    np.testing.assert_allclose(candidate_mu, baseline_mu, rtol=2e-13, atol=2e-13)
    assert candidate.iteration_log is not None
    assert baseline.iteration_log is not None
    assert [row.accepted_alpha for row in candidate.iteration_log] == [
        row.accepted_alpha for row in baseline.iteration_log
    ]


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


def test_execution_plan_scatters_interleaved_columns_and_auto_rejects_small_ordinary(
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
    assert calls == {"sandwich": 0, "transpose_matvec": 0}


def test_execution_plan_auto_accepts_large_measured_tabmat_moment_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(182)
    n = 50_000
    dense = DenseGroupMatrix(rng.normal(size=(n, 3)))
    categorical = CategoricalGroupMatrix(np.resize(np.arange(120), n), n_levels=120)
    W = rng.uniform(0.1, 2.0, size=n)
    rhs = rng.normal(size=n)
    calls = _count_tabmat_calls(monkeypatch)

    MatrixExecutionPlan([dense, categorical], n=n).moments(
        W,
        rhs=(rhs,),
        include_xtw=True,
    )

    assert calls == {"sandwich": 1, "transpose_matvec": 2}


def test_execution_plan_auto_rejects_single_dense_column_even_at_large_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(183)
    n = 50_000
    dense = DenseGroupMatrix(rng.normal(size=(n, 1)))
    categorical = CategoricalGroupMatrix(np.resize(np.arange(120), n), n_levels=120)
    calls = _count_tabmat_calls(monkeypatch)

    MatrixExecutionPlan([dense, categorical], n=n).moments(
        np.ones(n),
        rhs=(np.ones(n),),
        include_xtw=True,
    )

    assert calls == {"sandwich": 0, "transpose_matvec": 0}


def test_execution_plan_auto_rejects_additional_categorical_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(184)
    n = 50_000
    dense = DenseGroupMatrix(rng.normal(size=(n, 3)))
    high = CategoricalGroupMatrix(np.resize(np.arange(120), n), n_levels=120)
    low = CategoricalGroupMatrix(np.resize(np.arange(20), n), n_levels=20)
    calls = _count_tabmat_calls(monkeypatch)

    MatrixExecutionPlan([dense, high, low], n=n).moments(
        np.ones(n),
        rhs=(np.ones(n),),
        include_xtw=True,
    )

    assert calls == {"sandwich": 0, "transpose_matvec": 0}


def _ordinary_partition_case(case: str):
    n = 49_999 if case == "row-threshold" else 50_000
    dense_width = 2 if case == "dense-width" else 3
    dense = DenseGroupMatrix(np.zeros((n, dense_width), dtype=np.float64))
    category_levels = 100 if case == "low-cardinality" else 120
    categorical = CategoricalGroupMatrix(np.zeros(n, dtype=np.intp), category_levels)
    groups = [dense, categorical]
    policy = None
    expected = frozenset()

    if case == "policy-disabled":
        policy = False
    elif case == "policy-forced":
        policy = True
        compressed = DiscretizedSSPGroupMatrix(
            np.zeros((4, 3)),
            np.zeros((3, 2)),
            np.zeros(n, dtype=np.intp),
        )
        groups.append(compressed)
        expected = frozenset({0, 1})
    elif case == "contains-compressed-group":
        groups.append(
            DiscretizedSSPGroupMatrix(
                np.zeros((4, 3)),
                np.zeros((3, 2)),
                np.zeros(n, dtype=np.intp),
            )
        )
    elif case == "no-categorical":
        groups = [dense]
    elif case == "multiple-categoricals":
        groups.append(CategoricalGroupMatrix(np.zeros(n, dtype=np.intp), 140))
    elif case == "contains-sparse-group":
        groups.append(SparseGroupMatrix(sp.csr_matrix((n, 1))))
    elif case == "auto-certified":
        expected = frozenset({0, 1})

    return groups, n, policy, expected


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("policy-disabled", "policy-disabled"),
        ("policy-forced", "policy-forced"),
        ("contains-compressed-group", "contains-compressed-group"),
        ("no-categorical", "categorical-layout"),
        ("multiple-categoricals", "categorical-layout"),
        ("low-cardinality", "categorical-layout"),
        ("dense-width", "dense-width"),
        ("contains-sparse-group", "contains-sparse-group"),
        ("row-threshold", "row-threshold"),
        ("auto-certified", "auto-certified"),
    ],
)
def test_ordinary_partition_reason_is_stable_and_does_not_build_split(case, reason):
    groups, n, policy, expected = _ordinary_partition_case(case)
    plan = MatrixExecutionPlan(groups, n=n, ordinary_tabmat=policy)

    assert plan._ordinary_split_built is False
    assert plan.ordinary_indices == tuple(sorted(expected))
    assert plan.ordinary_partition_reason == reason
    assert plan._ordinary_indices == expected
    assert plan._ordinary_split_built is False
    with pytest.raises(AttributeError):
        plan.ordinary_indices = ()
    with pytest.raises(AttributeError):
        plan.ordinary_partition_reason = "changed"


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


def test_design_matrix_pickle_round_trip_rebuilds_execution_plan() -> None:
    rng = np.random.default_rng(185)
    n = 80
    groups = [
        DenseGroupMatrix(rng.normal(size=(n, 2))),
        CategoricalGroupMatrix(np.resize(np.arange(7), n), n_levels=7),
    ]
    X = np.column_stack([group.toarray() for group in groups])
    dm = DesignMatrix(groups, n=X.shape[0], p=X.shape[1])
    original_plan = dm.execution_plan
    original_split = dm.tabmat_split

    restored = pickle.loads(pickle.dumps(dm, protocol=pickle.HIGHEST_PROTOCOL))

    assert restored._execution_plan is None
    assert isinstance(restored.group_matrices, tuple)
    assert restored._tabmat_built
    assert restored.tabmat_split is restored._tabmat_holder.split
    assert restored.tabmat_split is not original_split
    rebuilt_plan = restored.execution_plan
    assert rebuilt_plan is not original_plan
    assert rebuilt_plan.shape == dm.shape
    np.testing.assert_allclose(restored.toarray(), X)


@pytest.mark.parametrize("split_built", [False, True])
def test_design_matrix_restores_legacy_tabmat_pickle_state(split_built: bool) -> None:
    rng = np.random.default_rng(186)
    n = 50
    groups = [
        DenseGroupMatrix(rng.normal(size=(n, 2))),
        CategoricalGroupMatrix(np.resize(np.arange(7), n), n_levels=7),
    ]
    legacy = DesignMatrix.__new__(DesignMatrix)
    legacy.__dict__.update(
        {
            "group_matrices": list(groups),
            "n": n,
            "p": 9,
            "shape": (n, 9),
            "_tabmat_split": _build_tabmat_split(groups) if split_built else None,
            "_tabmat_built": split_built,
            "_tabmat_centering_candidate": None,
            "_execution_plan": None,
            "_centered_pattern_plan": None,
            "_centered_solver_supports": None,
        }
    )

    restored = pickle.loads(pickle.dumps(legacy, protocol=pickle.HIGHEST_PROTOCOL))

    assert isinstance(restored.group_matrices, tuple)
    assert "_tabmat_split" not in restored.__dict__
    assert "_tabmat_built" not in restored.__dict__
    assert restored._tabmat_built is split_built
    if split_built:
        assert restored.tabmat_split is restored._tabmat_holder.split
    else:
        assert restored._tabmat_split is None
        assert restored.tabmat_split is not None
        assert restored._tabmat_built
    assert restored.execution_plan.shape == restored.shape
    np.testing.assert_allclose(restored.toarray(), np.column_stack([g.toarray() for g in groups]))


def test_design_matrix_constructor_rejects_declared_width_mismatch() -> None:
    _rng, groups, X = _mixed_groups(seed=173)

    with pytest.raises(ValueError, match="declared design shape"):
        DesignMatrix(groups, n=X.shape[0], p=X.shape[1] + 1)


def test_design_matrix_constructor_rejects_declared_row_mismatch() -> None:
    _rng, groups, X = _mixed_groups(seed=174)

    with pytest.raises(ValueError, match="declared design shape"):
        DesignMatrix(groups, n=X.shape[0] + 1, p=X.shape[1])


def test_design_matrix_execution_plan_reuses_existing_tabmat_split() -> None:
    rng = np.random.default_rng(174)
    n = 50_000
    dense = DenseGroupMatrix(rng.normal(size=(n, 3)))
    categorical = CategoricalGroupMatrix(np.resize(np.arange(120), n), n_levels=120)
    dm = DesignMatrix([dense, categorical], n=n, p=123)

    split = dm.tabmat_split
    plan = dm.execution_plan
    plan.moments(np.ones(n))

    assert split is not None
    assert plan._ordinary_split is split


def test_design_matrix_execution_plan_publishes_lazily_built_tabmat_split() -> None:
    rng = np.random.default_rng(175)
    n = 50_000
    dense = DenseGroupMatrix(rng.normal(size=(n, 3)))
    categorical = CategoricalGroupMatrix(np.resize(np.arange(120), n), n_levels=120)
    dm = DesignMatrix([dense, categorical], n=n, p=123)
    plan = dm.execution_plan

    plan.moments(np.ones(n))

    assert plan._ordinary_split is not None
    assert dm.tabmat_split is plan._ordinary_split


def test_execution_plan_split_factory_does_not_retain_design_matrix_owner() -> None:
    rng = np.random.default_rng(181)
    n = 50_000
    dense = DenseGroupMatrix(rng.normal(size=(n, 3)))
    categorical = CategoricalGroupMatrix(np.resize(np.arange(120), n), n_levels=120)
    dm = DesignMatrix([dense, categorical], n=n, p=123)
    owner_ref = weakref.ref(dm)
    plan = dm.execution_plan

    del dm

    assert owner_ref() is None
    assert plan.moments(np.ones(n)).gram.shape == (123, 123)


@pytest.mark.parametrize("split_first", [False, True])
def test_prebuilt_tabmat_storage_does_not_change_dense_only_auto_policy(split_first: bool) -> None:
    rng = np.random.default_rng(176)
    n = 100
    dense = DenseGroupMatrix(rng.normal(size=(n, 3)))
    dm = DesignMatrix([dense], n=n, p=3)
    if split_first:
        assert dm.tabmat_split is not None

    plan = dm.execution_plan
    plan.moments(np.ones(n))

    assert plan._ordinary_indices == frozenset()
    assert plan._ordinary_split is None


def test_legacy_weighted_moment_entry_points_delegate_to_one_execution_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng, groups, _X = _mixed_groups(seed=177)
    n = groups[0].shape[0]
    starts = np.cumsum([0, *[group.shape[1] for group in groups]])
    spans = [
        SimpleNamespace(start=int(starts[i]), end=int(starts[i + 1])) for i in range(len(groups))
    ]
    W = rng.uniform(0.0, 2.0, size=n)
    Wz = rng.normal(size=n)
    signed_W = rng.normal(size=n)
    calls: list[tuple[bool, bool, int]] = []
    original = MatrixExecutionPlan.moments

    def counted(self, weights, *, rhs=(), include_xtw=False, signed=False, **kwargs):
        calls.append((signed, include_xtw, len(rhs)))
        return original(
            self,
            weights,
            rhs=rhs,
            include_xtw=include_xtw,
            signed=signed,
            **kwargs,
        )

    monkeypatch.setattr(MatrixExecutionPlan, "moments", counted)

    algebra._block_xtwx(groups, spans, W)
    algebra._block_xtwx_rhs(groups, spans, W, Wz)
    algebra._block_xtwx_signed(groups, spans, signed_W)

    assert calls == [
        (False, False, 0),
        (False, True, 1),
        (True, False, 0),
    ]


def test_legacy_entry_points_keep_validation_and_span_contracts() -> None:
    rng = np.random.default_rng(180)
    n = 30
    group = DenseGroupMatrix(rng.normal(size=(n, 2)))
    spans = [SimpleNamespace(start=0, end=2)]
    W = rng.uniform(0.1, 2.0, size=n)
    Wz = rng.normal(size=n)

    with pytest.raises(ValueError, match="row count"):
        algebra._block_xtwx([group], spans, W[:-1])
    with pytest.raises(ValueError, match="row count"):
        algebra._block_xtwx_rhs([group], spans, W, Wz[:-1])

    nonfinite = W.copy()
    nonfinite[0] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        algebra._block_xtwx([group], spans, nonfinite)
    with pytest.raises(ValueError, match="must be finite"):
        algebra._block_xtwx_signed([group], spans, nonfinite)

    negative = W.copy()
    negative[1] = -0.5
    with pytest.raises(ValueError, match="negative weights"):
        algebra._block_xtwx([group], spans, negative)
    np.testing.assert_allclose(
        algebra._block_xtwx_signed([group], spans, negative),
        group.toarray().T @ (negative[:, None] * group.toarray()),
    )

    nonfinite_rhs = Wz.copy()
    nonfinite_rhs[2] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        algebra._block_xtwx_rhs([group], spans, W, nonfinite_rhs)

    wrong_spans = [SimpleNamespace(start=1, end=3)]
    with pytest.raises(ValueError, match="group slices must be contiguous"):
        algebra._block_xtwx([group], wrong_spans, W)


def test_execution_plan_public_moments_reject_invalid_vectors() -> None:
    rng, groups, _X = _mixed_groups(seed=178)
    n = groups[0].shape[0]
    plan = MatrixExecutionPlan(groups, n=n, ordinary_tabmat=False)
    nonfinite = rng.uniform(0.0, 2.0, size=n)
    nonfinite[3] = np.nan
    negative = rng.uniform(0.0, 2.0, size=n)
    negative[4] = -1.0

    with pytest.raises(ValueError, match="must be finite"):
        plan.moments(nonfinite)
    with pytest.raises(ValueError, match="negative weights"):
        plan.moments(negative)


def test_prevalidated_signed_compressed_moments_use_gram_only_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng, groups, _X = _mixed_groups(seed=184)
    compressed_groups = [groups[0], groups[3], groups[4]]
    plan = MatrixExecutionPlan(
        compressed_groups,
        n=compressed_groups[0].shape[0],
        ordinary_tabmat=False,
    )
    W = rng.normal(size=plan.n)
    expected = np.full((plan.p, plan.p), 7.0)
    calls: list[tuple[np.ndarray, object]] = []

    def gram_only(self, weights, *, profile):
        calls.append((weights, profile))
        return expected

    monkeypatch.setattr(
        MatrixExecutionPlan,
        "_compressed_signed_gram",
        gram_only,
        raising=False,
    )
    monkeypatch.setattr(
        MatrixExecutionPlan,
        "_moments_impl",
        lambda *_args, **_kwargs: pytest.fail(
            "prevalidated signed Gram must bypass general moment dispatch"
        ),
    )

    moments = plan._moments_prevalidated(W, signed=True)

    assert calls == [(W, None)]
    assert moments.gram is expected
    assert moments.xtw is None
    assert moments.xt_rhs == ()


def test_prevalidated_signed_compressed_fast_path_matches_dense_and_profiles_once() -> None:
    rng, groups, _X = _mixed_groups(seed=187)
    compressed_groups = [groups[0], groups[3]]
    X = np.column_stack([group.toarray() for group in compressed_groups])
    plan = MatrixExecutionPlan(
        compressed_groups,
        n=X.shape[0],
        ordinary_tabmat=False,
    )
    W = rng.normal(size=plan.n)
    profile: dict[str, float | int] = {}

    moments = plan._moments_prevalidated(W, signed=True, profile=profile)

    np.testing.assert_allclose(moments.gram, X.T @ (W[:, None] * X), rtol=2e-12, atol=2e-10)
    assert moments.xtw is None
    assert moments.xt_rhs == ()
    assert profile["block_calls"] == 1
    assert set(profile) == {
        "block_calls",
        "block_diag_discrete_ssp_s",
        "block_diag_other_s",
        "block_cross_disc_other_s",
    }
    for key in set(profile) - {"block_calls"}:
        assert profile[key] > 0.0


def test_public_signed_compressed_moments_validate_before_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng, groups, _X = _mixed_groups(seed=188)
    compressed_groups = [groups[0], groups[3]]
    plan = MatrixExecutionPlan(
        compressed_groups,
        n=compressed_groups[0].shape[0],
        ordinary_tabmat=False,
    )
    W = rng.normal(size=plan.n)
    W[3] = np.nan

    monkeypatch.setattr(
        MatrixExecutionPlan,
        "_compressed_signed_gram",
        lambda *_args, **_kwargs: pytest.fail("public moments must validate before assembly"),
    )

    with pytest.raises(ValueError, match="must be finite"):
        plan.moments(W, signed=True)


def test_compressed_signed_gram_reuses_planned_column_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng, groups, _X = _mixed_groups(seed=185)
    compressed_groups = [groups[0], groups[3], groups[4]]
    plan = MatrixExecutionPlan(
        compressed_groups,
        n=compressed_groups[0].shape[0],
        ordinary_tabmat=False,
    )
    W = rng.normal(size=plan.n)

    monkeypatch.setattr(
        GroupSpan,
        "columns",
        property(lambda _self: pytest.fail("signed Gram must reuse planned slices")),
    )

    gram = plan._compressed_signed_gram(W, profile=None)

    assert gram.shape == (plan.p, plan.p)


def test_full_ordinary_tabmat_plan_returns_before_group_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(179)
    n = 200
    groups = [DenseGroupMatrix(rng.normal(size=(n, 2))) for _ in range(12)]
    split = _build_tabmat_split(groups)
    plan = MatrixExecutionPlan(
        groups,
        n=n,
        ordinary_tabmat=True,
        prepared_ordinary_split=split,
    )
    W = rng.uniform(0.1, 2.0, size=n)

    monkeypatch.setattr(
        "superglm._group_matrix._group_matrix_execution._runtime_group_matrix_types",
        lambda: pytest.fail("a full Tabmat result must not enter grouped dispatch"),
    )

    moments = plan.moments(W, include_xtw=True)

    assert moments.gram.shape == (plan.p, plan.p)
    assert moments.xtw is not None
