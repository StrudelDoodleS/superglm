"""Cancellation regressions for projected spline moments."""

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from superglm import Spline, SuperGLM
from superglm._group_matrix._group_matrix_algebra import _cross_gram_tensor_own_margin
from superglm._group_matrix._group_matrix_centered import _TensorGridCache
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import DenseGroupMatrix, DiscretizedSSPGroupMatrix
from superglm.solvers import rank
from superglm.solvers.centered_system import build_centered_system


@pytest.mark.parametrize("dtype", [np.int64, np.float32])
@pytest.mark.parametrize("signed", [False, True])
def test_non_float64_supports_retain_weight_promoted_moments(dtype, signed):
    value = 2**32 if dtype == np.int64 else 1 + 2**-12
    basis = np.array([[value]], dtype=dtype)
    group = DiscretizedSSPGroupMatrix(basis, basis.copy(), np.zeros(1, dtype=int))
    weight = np.array([-1.0 if signed else 1.0])
    rhs = np.array([0.5])
    # Promotion by weights preceded both transform products before project-first.
    # The represented source factors have an exact binary64 product here.
    x = float(value) ** 2
    target = weight[0] * x**2
    bound = 16 * np.finfo(float).eps * abs(target)
    assert abs(group.gram(weight)[0, 0] - target) <= bound
    gram, xtw, xtrhs = group.gram_rmatvec(weight, rhs)
    assert abs(gram[0, 0] - target) <= bound
    np.testing.assert_allclose(xtw, weight * x, rtol=8 * np.finfo(float).eps, atol=0)
    np.testing.assert_allclose(xtrhs, rhs * x, rtol=8 * np.finfo(float).eps, atol=0)

    plan = MatrixExecutionPlan((group, DenseGroupMatrix(np.ones((1, 1)))), n=1)
    for fused in (False, True):
        moments = plan.moments(
            weight, signed=signed, rhs=(rhs,) if fused else (), include_xtw=fused
        )
        expected = weight[0] * np.array([[x**2, x], [x, 1]])
        np.testing.assert_allclose(moments.gram, expected, rtol=16 * np.finfo(float).eps, atol=0)
        if fused:
            np.testing.assert_allclose(
                moments.xtw, weight[0] * np.array([x, 1]), rtol=8 * np.finfo(float).eps, atol=0
            )
            np.testing.assert_allclose(
                moments.xt_rhs[0], rhs[0] * np.array([x, 1]), rtol=8 * np.finfo(float).eps, atol=0
            )


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("signed", [False, True])
def test_exceptional_moments_defer_projection_until_the_exact_range_guard(fused, signed):
    large = np.ldexp(1.0, 600)
    group = DiscretizedSSPGroupMatrix(
        np.array([[large, large, 1.0]]), np.array([[large], [-large], [1.0]]), np.zeros(1, int)
    )
    plan = MatrixExecutionPlan((group,), n=1)
    weights = np.array([-1.0 if signed else 1.0])
    with np.errstate(over="raise", invalid="raise"):
        moments = plan.moments(
            weights, rhs=(weights,) if fused else (), include_xtw=fused, signed=signed
        )
    np.testing.assert_array_equal(moments.gram, weights[:, None])
    if fused:
        np.testing.assert_array_equal(moments.xtw, weights)
        np.testing.assert_array_equal(moments.xt_rhs[0], weights)


def _year_frame(width=4e-5, side=1.0):
    rng = np.random.default_rng(20260919)
    n = 13_199
    others = np.round(np.linspace(1900.0, 2015.0, 97))[:-1]
    year = np.where(rng.random(n) < 0.78, 2000.0, others[np.arange(n) % 96])
    year[:4] = (2000.0 + side * 115.0 * 7 / width, 0.0, 4965.0, 1.0)
    frame = pd.DataFrame(
        {
            "year": (year - year.mean()) / year.std(),
            "continuous": rng.normal(size=n),
            "a": np.round(rng.normal(size=n), 1),
            "b": np.round(rng.normal(size=n), 1),
            "c": np.round(rng.normal(size=n), 1),
        }
    )
    return frame, rng


def _year_design(*, discrete=False, width=4e-5, side=1.0, pairs=()):
    frame, rng = _year_frame(width, side)
    model = SuperGLM(
        family="gaussian",
        features={name: Spline(kind="ps", k=10) for name in frame},
        discrete=discrete,
        n_bins=256,
        selection_penalty=None,
        interactions=list(pairs),
    )
    model._build_design_matrix(frame, rng.normal(size=len(frame)), np.ones(len(frame)), None)
    return model._dm, rng


def _reference(dm, weights):
    # Certify assembly against the represented solver rows. Rounding B @ R
    # itself has a separate factor-product error bound below; it is not Gram
    # assembly error. Center-first long-double products avoid raw cancellation.
    design = np.hstack([group.toarray() for group in dm.group_matrices]).astype(np.longdouble)
    w = weights.astype(np.longdouble)
    centered = design - (w @ design) / w.sum()
    return np.asarray(centered.T @ (w[:, None] * centered), dtype=float)


@pytest.mark.parametrize("discrete,pairs", [(False, ()), (True, ()), (True, (("a", "b"),))])
def test_whole_year_gram_resolves_cross_blocks_and_null_subspace(discrete, pairs):
    dm, rng = _year_design(discrete=discrete, pairs=pairs)
    weights = rng.uniform(0.5, 1.5, dm.n)
    moments = dm.execution_plan.moments(weights, include_xtw=True)
    data_gram = moments.gram - np.outer(moments.xtw, moments.xtw / weights.sum())
    reference = _reference(dm, weights)
    scale = np.sqrt(np.diag(reference))
    reference_eq = reference / np.outer(scale, scale)
    actual_eq = data_gram / np.outer(scale, scale)
    norm = np.linalg.norm(reference_eq, 2)
    bar = max(100 * np.finfo(float).eps, rank._eigensolver_relative_bar(dm.p)) * norm
    assert np.linalg.norm(actual_eq - reference_eq, 2) <= bar

    expected = rank.decompose_gram(reference)
    actual = rank.decompose_gram(data_gram)
    assert actual.rank == expected.rank < dm.p
    # Compare the resolved eigenspaces with the perturbation/gap bound. This
    # tests rank geometry without imposing a sign on a rounded null eigenvalue.
    eigenvalues, vectors = np.linalg.eigh(reference_eq)
    cutoff = dm.p - expected.rank
    gap = eigenvalues[cutoff] - eigenvalues[cutoff - 1]
    _, actual_vectors = np.linalg.eigh(actual_eq)
    angles = scipy.linalg.subspace_angles(vectors[:, :cutoff], actual_vectors[:, :cutoff])
    assert np.sin(angles).max(initial=0) <= 4 * bar / gap
    assert np.linalg.norm(actual_eq @ vectors[:, :cutoff], 2) <= 4 * bar


@pytest.mark.parametrize("width", [1e-3, 1e-5, 1e-7])
@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_year_gram_keeps_rank_certification_at_narrow_support(width, side):
    dm, rng = _year_design(width=width, side=side)
    weights = rng.uniform(0.5, 1.5, dm.n)
    system = build_centered_system(
        dm=dm,
        W=weights,
        z_off=rng.normal(size=dm.n),
        penalty=np.zeros((dm.p, dm.p)),
    )
    rank.decompose_gram_if_authoritative(system.data_gram)


def test_year_projection_matches_source_factors_with_product_error_bound():
    dm, _ = _year_design()
    group = dm.group_matrices[0]
    basis, transform = group.B_unique.astype(np.longdouble), group.R_inv.astype(np.longdouble)
    unit = np.finfo(float).eps / 2
    gamma = basis.shape[1] * unit / (1 - basis.shape[1] * unit)
    exact = basis @ transform
    error = abs((group.B_unique @ group.R_inv).astype(np.longdouble) - exact)
    assert np.all(error <= gamma * (abs(basis) @ abs(transform)))


@pytest.mark.parametrize("main_index", [2, 3])
def test_tensor_own_margin_accepts_the_centered_weight_grid_cache(main_index):
    dm, rng = _year_design(discrete=True, pairs=(("a", "b"),))
    main, tensor = dm.group_matrices[main_index], dm.group_matrices[-1]
    weights = rng.uniform(0.5, 1.5, dm.n)
    grid = np.zeros((tensor.n_bins1, tensor.n_bins2))
    np.add.at(grid, (tensor.idx1, tensor.idx2), weights)
    actual = _cross_gram_tensor_own_margin(tensor, main, weights, _TensorGridCache(grid))
    left, right = main.toarray(), tensor.toarray()
    target = left.T @ (weights[:, None] * right)
    scale = abs(left).T @ (weights[:, None] * abs(right))
    bound = (dm.n + 2 * dm.p) * np.finfo(float).eps * np.linalg.norm(scale, np.inf)
    assert np.linalg.norm(actual - target, np.inf) <= bound


@pytest.mark.parametrize("signed", [False, True])
def test_projected_support_is_reused_only_within_an_assembly(signed):
    dm, rng = _year_design()
    weights = rng.uniform(0.5, 1.5, dm.n)
    if signed:
        weights[::2] *= -1
    for _ in range(2):
        profile = {}
        moments = dm.execution_plan.moments(
            weights, include_xtw=not signed, signed=signed, profile=profile
        )
        design = np.hstack([group.toarray() for group in dm.group_matrices])
        target = design.T @ (weights[:, None] * design)
        scale = abs(design).T @ (abs(weights[:, None]) * abs(design))
        bound = (dm.n + 2 * dm.p) * np.finfo(float).eps * np.linalg.norm(scale, np.inf)
        assert np.linalg.norm(moments.gram - target, np.inf) <= bound
        assert profile["block_solver_support_builds"] == 4
        assert profile["block_solver_support_reuses"] >= 4
        dm.group_matrices[0].B_unique *= 0.5
        dm.group_matrices[0].R_inv[0, 0] += 0.125
        weights *= 0.5

    rows = np.arange(0, dm.n, 3)
    groups = [group.row_subset(rows) for group in dm.group_matrices]
    plan = MatrixExecutionPlan(groups, n=len(rows))
    result = plan.moments(weights[rows], signed=signed).gram
    design = np.hstack([group.toarray() for group in groups])
    target = design.T @ (weights[rows, None] * design)
    np.testing.assert_allclose(result, target, atol=bound, rtol=0)
