"""Disjoint raw levels save work without dropping centered cross products."""

import pickle

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm.dm_builder import rebuild_design_matrix_with_lambdas
from superglm.features import Categorical, Spline
from superglm.group_matrix import DesignMatrix, DiscretizedSplineCategoricalGroupMatrix
from superglm.solvers.centered_system import build_centered_system


def _model():
    rng = np.random.default_rng(7301)
    n = 240
    frame = pd.DataFrame(
        {"x": rng.uniform(size=n), "y": rng.uniform(size=n), "c": np.resize(list("ABC"), n)}
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=None,
        discrete=True,
        n_bins=16,
        features={"x": Spline(k=5), "y": Spline(k=5), "c": Categorical(base="first")},
        interactions=[("x", "c"), ("y", "c")],
    )
    model._build_design_matrix(frame, rng.normal(size=n), np.ones(n), None)
    return model


def test_builder_compiles_disjoint_raw_blocks_and_carries_through_whitening(monkeypatch):
    model = _model()
    dm = model._dm
    calls = []
    original = algebra._cross_gram_spline_categorical_spline_categorical

    def counted(left, right, *args, **kwargs):
        calls.append((left.spline_cat_level, right.spline_cat_level))
        return original(left, right, *args, **kwargs)

    monkeypatch.setattr(algebra, "_cross_gram_spline_categorical_spline_categorical", counted)
    for step in range(3):
        W = np.linspace(0.5 + step, 1.5 + step, dm.n)
        dm.execution_plan.moments(W, include_xtw=True)
        if step < 2:
            dm = rebuild_design_matrix_with_lambdas(
                dm,
                model._groups,
                {g.name: float(step + 2) for g in model._groups},
                np.ones(dm.n),
                1.0,
            )
    assert calls
    assert all(left == right for left, right in calls), calls


@pytest.mark.parametrize("same_label", [False, True])
def test_custom_overlapping_supports_do_not_trust_category_labels(same_label):
    B = np.array([[1.0, 2.0], [2.0, -1.0]])
    bins = np.arange(8) % 2
    groups = [
        DiscretizedSplineCategoricalGroupMatrix(B, np.eye(2), bins, rows)
        for rows in (np.array([0, 2, 4, 6]), np.array([2, 3, 4, 7]))
    ]
    for i, group in enumerate(groups):
        group.spline_cat_feature = "misleading parent"
        group.spline_cat_level = "equal" if same_label else str(i)
    dm = DesignMatrix(groups, n=8, p=4)
    X = np.column_stack([g.toarray() for g in groups])
    W = np.arange(1.0, 9.0)
    expected = X.T @ (W[:, None] * X)
    assert np.any(expected[:2, 2:])
    np.testing.assert_allclose(dm.execution_plan.moments(W).gram, expected, rtol=0, atol=0)


def test_raw_zero_preserves_centered_cross_intercept_rank_and_inference():
    from superglm._group_matrix._level_partition import certify_level_partitions

    # A well-conditioned two-column example with nonzero means. Both the raw
    # and centered results have independent literal row-design references.
    bins = np.arange(12) % 3
    groups = [
        DiscretizedSplineCategoricalGroupMatrix(
            np.array([[1.0], [2.0], [4.0]]), np.ones((1, 1)), bins, np.arange(i, 12, 2)
        )
        for i in range(2)
    ]
    for i, group in enumerate(groups):
        group.spline_cat_feature, group.spline_cat_level = "c", str(i)
    certify_level_partitions(groups, 12)
    dm = DesignMatrix(groups, 12, 2)
    X = np.column_stack([g.toarray() for g in groups])
    W, z = np.arange(1.0, 13.0), np.sin(np.arange(12.0))
    moments = dm.execution_plan.moments(W, rhs=(W * z,), include_xtw=True)
    assert moments.gram[0, 1] == 0
    centered = moments.gram - np.outer(moments.xtw, moments.xtw) / W.sum()
    assert centered[0, 1] != 0
    system = build_centered_system(dm=dm, W=W, z_off=z, penalty=np.eye(2))
    np.testing.assert_allclose(system.data_gram, centered, rtol=2e-14, atol=2e-13)
    A = np.column_stack([np.ones(12), X])
    penalty = np.diag([0.0, 1.0, 1.0])
    expected_gram = A.T @ (W[:, None] * A)
    actual_gram = np.block(
        [[np.array([[W.sum()]]), moments.xtw[None]], [moments.xtw[:, None], moments.gram]]
    )
    rhs = np.r_[W @ z, moments.xt_rhs[0]]
    covariance = np.linalg.inv(actual_gram + penalty)
    reference_covariance = np.linalg.inv(expected_gram + penalty)
    beta = covariance @ rhs
    reference_beta = reference_covariance @ (A.T @ (W * z))
    np.testing.assert_allclose(A @ beta, A @ reference_beta, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        np.einsum("ij,jk,ik->i", A, covariance, A),
        np.einsum("ij,jk,ik->i", A, reference_covariance, A),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.trace(covariance @ actual_gram),
        np.trace(reference_covariance @ expected_gram),
        rtol=1e-13,
    )
    assert np.linalg.matrix_rank(actual_gram) == np.linalg.matrix_rank(expected_gram) == 3


@pytest.mark.parametrize("signed", [False, True])
def test_support_replacement_invalidates_compiled_zero_and_subsets_are_fresh(signed):
    from superglm._group_matrix._level_partition import certify_level_partitions

    bins = np.arange(12) % 3
    groups = [
        DiscretizedSplineCategoricalGroupMatrix(
            np.arange(1.0, 4.0)[:, None], np.ones((1, 1)), bins, np.arange(i, 12, 2)
        )
        for i in range(2)
    ]
    for i, group in enumerate(groups):
        group.spline_cat_feature, group.spline_cat_level = "c", str(i)
    certify_level_partitions(groups, 12)
    dm = DesignMatrix(groups, 12, 2)
    plan = dm.execution_plan
    W = np.linspace(-1 if signed else 0.1, 1.0, 12) * 2.0**400
    plan.moments(W, signed=signed)
    # A mutable replacement creates overlap, despite unchanged metadata and
    # despite an execution plan compiled while the levels were disjoint.
    groups[1].row_idx = groups[0].row_idx.copy()
    X = dm.toarray()
    np.testing.assert_allclose(
        plan.moments(W, signed=signed).gram, X.T @ (W[:, None] * X), rtol=2e-14
    )
    idx = np.array([10, 0, 10, 3, 4])
    sub = dm.row_subset(idx)
    np.testing.assert_allclose(
        sub.execution_plan.moments(W[idx], signed=signed).gram,
        X[idx].T @ (W[idx, None] * X[idx]),
        rtol=2e-14,
    )


def test_once_only_partition_certification_checks_actual_supports():
    from superglm._group_matrix._level_partition import certify_level_partitions

    groups = [
        DiscretizedSplineCategoricalGroupMatrix(
            np.ones((1, 1)), np.ones((1, 1)), np.zeros(8, dtype=int), rows
        )
        for rows in (np.array([0, 1, 2]), np.array([2, 3, 4]))
    ]
    for i, group in enumerate(groups):
        group.spline_cat_feature, group.spline_cat_level = "c", str(i)
    certify_level_partitions(groups, 8)
    dm = DesignMatrix(groups, 8, 2)
    assert dm.execution_plan.moments(np.ones(8)).gram[0, 1] == 1.0


def test_partition_storage_cannot_be_mutated_and_pickle_drops_its_proof():
    model = _model()
    dm = model._dm
    group = next(g for g in dm.group_matrices if getattr(g, "_level_support", None) is not None)
    with pytest.raises(ValueError):
        group.row_idx.setflags(write=True)
    W = np.linspace(0.2, 1.3, dm.n)
    expected = dm.execution_plan.moments(W).gram
    restored = pickle.loads(pickle.dumps(dm))
    np.testing.assert_array_equal(restored.execution_plan.moments(W).gram, expected)
    assert all(getattr(g, "_level_support", None) is None for g in restored.group_matrices)
