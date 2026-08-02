"""Correctness and policy gates for raw-basis Tabmat spline centering."""

from __future__ import annotations

import gc
import pickle
import weakref

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from superglm import Spline, SuperGLM
from superglm.distributions import Gaussian, Poisson
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix, SparseSSPGroupMatrix
from superglm.links import IdentityLink, LogLink
from superglm.model.base import model_build_design_matrix
from superglm.solvers.centered_system import (
    TabmatCenteringState,
    build_centered_system,
    grouped_weighted_factor,
)
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.rank import decompose_factor, decompose_gram, needs_factor_certification
from superglm.types import GroupSlice


class _StableSparseSSPGroupMatrix(SparseSSPGroupMatrix):
    """Test-only exact-kernel equivalent that is outside the auto policy."""


def _spline_group(
    n: int,
    width: int,
    *,
    phase: int,
    transform: np.ndarray | None = None,
) -> SparseSSPGroupMatrix:
    """Return a deterministic compact-support spline-like raw basis."""
    rows = np.repeat(np.arange(n, dtype=np.intp), 4)
    local = np.tile(np.arange(4, dtype=np.intp), n)
    columns = (np.repeat(np.arange(n, dtype=np.intp), 4) + local + phase) % width
    values = np.tile(np.array([0.1, 0.4, 0.4, 0.1]), n)
    basis = sp.csr_matrix((values, (rows, columns)), shape=(n, width))
    if transform is None:
        transform = np.eye(width)
        transform[np.arange(width - 1), np.arange(1, width)] = 0.05
    return SparseSSPGroupMatrix(basis, transform)


def _dense_centered_reference(
    dm: DesignMatrix,
    W: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = dm.toarray()
    mean_x = np.average(X, axis=0, weights=W)
    mean_z = float(np.average(z, weights=W))
    centered = X - mean_x
    return mean_x, centered.T @ (W[:, None] * centered), centered.T @ (W * (z - mean_z))


def _count_tabmat_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    import tabmat

    calls = {"standardize": 0, "sandwich": 0, "transpose_matvec": 0}
    for name in calls:
        original = getattr(tabmat.SplitMatrix, name)

        def counted(self, *args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(tabmat.SplitMatrix, name, counted)
    return calls


def test_multi_spline_centering_builds_one_lazy_raw_tabmat_plan() -> None:
    rng = np.random.default_rng(1701)
    n = 8_000
    groups = [_spline_group(n, 15, phase=0), _spline_group(n, 15, phase=2)]
    dm = DesignMatrix(groups, n=n, p=sum(group.shape[1] for group in groups))
    W = rng.uniform(0.25, 2.0, size=n)
    z = rng.normal(size=n)
    expected_mean, expected_gram, expected_rhs = _dense_centered_reference(dm, W, z)
    profile: dict[str, float | int] = {}
    state = TabmatCenteringState()

    assert dm.raw_spline_tabmat_plan_built is False
    first = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
        profile=profile,
    )
    plan = dm.get_raw_spline_tabmat_centering_plan()
    second = build_centered_system(
        dm=dm,
        W=W,
        z_off=z + 0.25,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
        profile=profile,
    )

    assert dm.raw_spline_tabmat_plan_built is True
    assert plan is dm.get_raw_spline_tabmat_centering_plan()
    assert state.raw_spline_eligible is True
    assert profile["centered_spline_tabmat_builds"] == 1
    assert profile["centered_spline_tabmat_attempts"] == 2
    assert profile["centered_spline_tabmat_accepts"] == 2
    assert profile["centered_spline_tabmat_retained_bytes"] > 0
    np.testing.assert_allclose(first.mean_x, expected_mean, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(first.data_gram, expected_gram, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(first.rhs, expected_rhs, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(second.data_gram, first.data_gram, rtol=2e-12, atol=2e-11)

    dm.release_raw_spline_tabmat_plan()
    assert dm.raw_spline_tabmat_plan_built is False


def test_dense_cardinal_spline_csr_is_rejected_before_tabmat_construction() -> None:
    """Sparse storage alone must not admit a structurally dense cardinal basis."""
    n = 8_000
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, n),
            "z": np.linspace(1.0, 0.0, n),
        }
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": Spline(kind="cr_cardinal", n_knots=15),
            "z": Spline(kind="cr_cardinal", n_knots=15),
        },
    )
    model_build_design_matrix(model, frame, np.zeros(n), np.ones(n), None)
    groups = model._dm.group_matrices
    assert all(type(group) is SparseSSPGroupMatrix for group in groups)
    assert all(group.B.nnz > 8 * n for group in groups)

    assert model._dm.get_raw_spline_tabmat_centering_plan() is None


def test_unbenchmarked_wide_support_sparse_basis_is_rejected() -> None:
    """The policy admits measured cubic support, not arbitrary sparse storage."""
    n = 8_000
    width = 20
    support = 6
    rows = np.repeat(np.arange(n, dtype=np.intp), support)
    local = np.tile(np.arange(support, dtype=np.intp), n)
    columns = (np.repeat(np.arange(n, dtype=np.intp), support) + local) % width
    values = np.full(n * support, 1.0 / support)
    basis = sp.csr_matrix((values, (rows, columns)), shape=(n, width))
    groups = [
        SparseSSPGroupMatrix(basis.copy(), np.eye(width)),
        SparseSSPGroupMatrix(basis.copy(), np.eye(width)),
    ]
    dm = DesignMatrix(groups, n=n, p=2 * width)

    assert dm.get_raw_spline_tabmat_centering_plan() is None


def test_dense_tensor_interaction_is_rejected_before_tabmat_construction() -> None:
    """A tensor stored as CSR is still dense in row support and loses badly."""
    n = 8_000
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, n),
            "z": np.linspace(1.0, 0.0, n),
        }
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Spline(n_knots=7), "z": Spline(n_knots=7)},
        interactions=[("x", "z")],
    )
    model_build_design_matrix(model, frame, np.zeros(n), np.ones(n), None)
    groups = model._dm.group_matrices
    assert len(groups) == 3
    assert all(type(group) is SparseSSPGroupMatrix for group in groups)
    assert groups[-1].B.nnz > 4 * n

    assert model._dm.get_raw_spline_tabmat_centering_plan() is None


def test_discrete_splines_never_build_the_raw_tabmat_plan() -> None:
    """The BAM-style support-space kernel remains authoritative when discrete=True."""
    n = 8_000
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, n),
            "z": np.linspace(1.0, 0.0, n),
        }
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        features={"x": Spline(n_knots=12), "z": Spline(n_knots=12)},
    )
    model_build_design_matrix(model, frame, np.zeros(n), np.ones(n), None)

    assert model._dm.get_raw_spline_tabmat_centering_plan() is None


def test_raw_spline_path_uses_one_sandwich_without_standardization_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(1707)
    n = 8_000
    groups = [_spline_group(n, 15, phase=0), _spline_group(n, 15, phase=2)]
    dm = DesignMatrix(groups, n=n, p=30)
    calls = _count_tabmat_calls(monkeypatch)

    build_centered_system(
        dm=dm,
        W=rng.uniform(0.25, 2.0, size=n),
        z_off=rng.normal(size=n),
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=TabmatCenteringState(),
    )

    assert calls == {"standardize": 0, "sandwich": 1, "transpose_matvec": 2}


def test_one_shot_centered_call_does_not_pay_lazy_plan_construction() -> None:
    rng = np.random.default_rng(1708)
    n = 8_000
    groups = [_spline_group(n, 8, phase=0), _spline_group(n, 7, phase=2)]
    dm = DesignMatrix(groups, n=n, p=15)

    build_centered_system(
        dm=dm,
        W=rng.uniform(0.25, 2.0, size=n),
        z_off=rng.normal(size=n),
        penalty=np.zeros((dm.p, dm.p)),
    )

    assert dm.raw_spline_tabmat_plan_built is False


def test_large_solver_translation_rejects_raw_moments_and_uses_stable_chunks() -> None:
    rng = np.random.default_rng(1702)
    n = 8_000
    width = 12
    transform = np.eye(width)
    transform[np.arange(width - 1), np.arange(1, width)] = 0.05
    shifted_transform = transform.copy()
    shifted_transform[:, 0] += 1.0e10
    ordinary_groups = [
        _spline_group(n, width, phase=0, transform=transform),
        _spline_group(n, width, phase=2, transform=transform),
    ]
    shifted_groups = [
        _spline_group(n, width, phase=0, transform=shifted_transform),
        _spline_group(n, width, phase=2, transform=transform),
    ]
    ordinary = DesignMatrix(ordinary_groups, n=n, p=2 * width)
    shifted = DesignMatrix(shifted_groups, n=n, p=2 * width)
    W = rng.uniform(0.25, 2.0, size=n)
    z = rng.normal(size=n)
    penalty = np.zeros((2 * width, 2 * width))
    expected = build_centered_system(dm=ordinary, W=W, z_off=z, penalty=penalty)
    profile: dict[str, float | int] = {}
    state = TabmatCenteringState()

    actual = build_centered_system(
        dm=shifted,
        W=W,
        z_off=z,
        penalty=penalty,
        tabmat_state=state,
        profile=profile,
    )

    assert state.raw_spline_eligible is False
    assert profile["centered_spline_tabmat_rejections"] == 1
    assert profile["centered_spline_tabmat_stable_fallbacks"] == 1
    np.testing.assert_allclose(actual.data_gram, expected.data_gram, rtol=2e-6, atol=2e-4)
    np.testing.assert_allclose(actual.rhs, expected.rhs, rtol=2e-6, atol=2e-4)


def test_aliased_spline_blocks_preserve_certified_rank_and_products() -> None:
    rng = np.random.default_rng(1703)
    n = 8_000
    width = 12
    first = _spline_group(n, width, phase=1)
    aliased = SparseSSPGroupMatrix(first.B.copy(), first.R_inv.copy())
    dm = DesignMatrix([first, aliased], n=n, p=2 * width)
    W = rng.uniform(0.25, 2.0, size=n)
    z = rng.normal(size=n)
    expected_mean, expected_gram, expected_rhs = _dense_centered_reference(dm, W, z)
    state = TabmatCenteringState()

    actual = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
    )

    assert state.raw_spline_eligible is True
    raw_rank = decompose_gram(actual.data_gram)
    certified_rank = decompose_factor(grouped_weighted_factor(dm, W, center=actual.mean_x))
    assert needs_factor_certification(raw_rank)
    assert certified_rank.rank == width - 2
    np.testing.assert_allclose(actual.mean_x, expected_mean, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual.data_gram, expected_gram, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(actual.rhs, expected_rhs, rtol=2e-12, atol=2e-11)


def test_psd_cleanup_preserves_structurally_zero_rows_and_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PSD projection must not give an empty-support coordinate numerical mass."""
    import superglm.solvers.centered_system as centered_system

    # The declared-PSD cleanup should retain this block's one positive mode.
    # With the structural zero coordinate included in the eigensolve, LAPACK
    # scatters reconstruction noise into its empty row and reports rank two.
    raw_hessian = np.array(
        [
            [-1.5569582453259823e-08, 0.0, -3.5450204197036575e-08],
            [0.0, 0.0, 0.0],
            [-3.5450204197036575e-08, 0.0, -7.923109584302772e-08],
        ]
    )
    assert np.linalg.eigvalsh(raw_hessian)[0] < -1e-12
    dm = DesignMatrix([DenseGroupMatrix(np.zeros((2, 3)))], n=2, p=3)

    monkeypatch.setattr(
        centered_system,
        "packed_centered_gram_rhs",
        lambda **_kwargs: (np.zeros(3), raw_hessian, np.zeros(3)),
    )
    system = centered_system.build_centered_system(
        dm=dm,
        W=np.ones(2),
        z_off=np.zeros(2),
        penalty=np.zeros((3, 3)),
    )

    np.testing.assert_array_equal(system.hessian[1, :], np.zeros(3))
    np.testing.assert_array_equal(system.hessian[:, 1], np.zeros(3))
    assert np.linalg.eigvalsh(system.hessian)[0] >= -1e-15
    assert np.linalg.matrix_rank(system.hessian) == 1

    decomposition = decompose_gram(system.hessian)
    assert decomposition.rank == 1
    assert decomposition.column_scale[1] == 0.0
    assert np.count_nonzero(decomposition.column_scale) == 2


def test_rectangular_spline_transforms_match_solver_coordinate_reference() -> None:
    rng = np.random.default_rng(1704)
    n = 8_000
    first_transform = np.eye(12)[:, :-1] - np.eye(12)[:, [-1]]
    second_transform = np.eye(12)[:, :-1] - np.eye(12)[:, [-1]]
    groups = [
        _spline_group(n, 12, phase=0, transform=first_transform),
        _spline_group(n, 12, phase=3, transform=second_transform),
    ]
    dm = DesignMatrix(groups, n=n, p=22)
    W = rng.uniform(0.25, 2.0, size=n)
    z = rng.normal(size=n)
    expected_mean, expected_gram, expected_rhs = _dense_centered_reference(dm, W, z)
    state = TabmatCenteringState()

    actual = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=state,
    )

    assert state.raw_spline_eligible is True
    np.testing.assert_allclose(actual.mean_x, expected_mean, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual.data_gram, expected_gram, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(actual.rhs, expected_rhs, rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("layout", ["strided", "readonly"])
def test_raw_spline_tabmat_normalizes_weight_buffers_without_mutation(layout: str) -> None:
    rng = np.random.default_rng(1705)
    n = 8_000
    groups = [_spline_group(n, 15, phase=0), _spline_group(n, 15, phase=2)]
    dm = DesignMatrix(groups, n=n, p=30)
    base_weights = rng.uniform(0.25, 2.0, size=n)
    if layout == "strided":
        storage = np.empty(2 * n)
        storage[::2] = base_weights
        W = storage[::2]
        assert not W.flags.c_contiguous
    else:
        W = base_weights.copy()
        W.setflags(write=False)
        assert not W.flags.writeable
    before = W.copy()
    z = rng.normal(size=n)
    expected_mean, expected_gram, expected_rhs = _dense_centered_reference(dm, base_weights, z)

    actual = build_centered_system(
        dm=dm,
        W=W,
        z_off=z,
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=TabmatCenteringState(),
    )

    np.testing.assert_array_equal(W, before)
    np.testing.assert_allclose(actual.mean_x, expected_mean, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual.data_gram, expected_gram, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(actual.rhs, expected_rhs, rtol=2e-12, atol=2e-11)


@pytest.mark.parametrize("layout", ["single", "large", "mixed"])
def test_raw_spline_tabmat_policy_rejects_uncertified_layouts(layout: str) -> None:
    if layout == "single":
        n = 8_000
        groups = [_spline_group(n, 20, phase=0)]
    elif layout == "large":
        n = 60_000
        groups = [_spline_group(n, 60, phase=0), _spline_group(n, 60, phase=2)]
    else:
        n = 8_000
        groups = [
            _spline_group(n, 10, phase=0),
            _spline_group(n, 10, phase=2),
            DenseGroupMatrix(np.ones((n, 1))),
        ]
    dm = DesignMatrix(groups, n=n, p=sum(group.shape[1] for group in groups))
    profile: dict[str, float | int] = {}

    plan = dm.get_raw_spline_tabmat_centering_plan(profile=profile)

    assert plan is None
    assert dm.raw_spline_tabmat_plan_built is True
    assert dm._tabmat_built is False
    assert profile["centered_spline_tabmat_policy_rejections"] == 1
    assert "centered_spline_tabmat_builds" not in profile


def test_raw_spline_tabmat_plan_is_releasable_and_not_serialized() -> None:
    n = 8_000
    groups = [_spline_group(n, 15, phase=0), _spline_group(n, 15, phase=2)]
    dm = DesignMatrix(groups, n=n, p=30)
    original = dm.get_raw_spline_tabmat_centering_plan()
    assert original is not None
    original_ref = weakref.ref(original)

    restored = pickle.loads(pickle.dumps(dm))

    assert dm.raw_spline_tabmat_plan_built is True
    assert restored.raw_spline_tabmat_plan_built is False
    dm.release_raw_spline_tabmat_plan()
    dm.release_raw_spline_tabmat_plan()
    assert dm.raw_spline_tabmat_plan_built is False
    del original
    gc.collect()
    assert original_ref() is None
    rebuilt = dm.get_raw_spline_tabmat_centering_plan()
    assert rebuilt is not None


def test_poisson_fit_matches_stable_spline_kernel_and_reuses_plan() -> None:
    rng = np.random.default_rng(1706)
    n = 8_000
    accelerated_groups = [_spline_group(n, 15, phase=0), _spline_group(n, 15, phase=2)]
    stable_groups = [
        _StableSparseSSPGroupMatrix(group.B.copy(), group.R_inv.copy())
        for group in accelerated_groups
    ]
    accelerated_dm = DesignMatrix(accelerated_groups, n=n, p=30)
    stable_dm = DesignMatrix(stable_groups, n=n, p=30)
    beta_true = rng.normal(scale=0.08, size=30)
    y = rng.poisson(np.exp(-0.15 + accelerated_dm.matvec(beta_true))).astype(float)
    weights = rng.uniform(0.5, 1.5, size=n)
    group_slices = [
        GroupSlice(name="spline_1", start=0, end=15),
        GroupSlice(name="spline_2", start=15, end=30),
    ]
    penalty = 0.2 * np.eye(30)
    profile: dict[str, float | int] = {}
    common = dict(
        y=y,
        weights=weights,
        family=Poisson(),
        link=LogLink(),
        groups=group_slices,
        lambda2=1.0,
        S_override=penalty,
        tol=1e-10,
        max_iter=50,
    )

    accelerated, _ = fit_irls_direct(X=accelerated_dm, profile=profile, **common)
    stable, _ = fit_irls_direct(X=stable_dm, **common)
    accelerated_second, _ = fit_irls_direct(
        X=accelerated_dm,
        profile=profile,
        **{
            **common,
            "S_override": 0.35 * np.eye(30),
            "beta_init": accelerated.beta,
            "intercept_init": accelerated.intercept,
        },
    )
    stable_second, _ = fit_irls_direct(
        X=stable_dm,
        **{
            **common,
            "S_override": 0.35 * np.eye(30),
            "beta_init": stable.beta,
            "intercept_init": stable.intercept,
        },
    )

    assert accelerated.converged is True
    assert stable.converged is True
    assert profile["centered_spline_tabmat_builds"] == 1
    assert profile["centered_spline_tabmat_attempts"] >= 2
    assert profile["centered_spline_tabmat_accepts"] == profile["centered_spline_tabmat_attempts"]
    np.testing.assert_allclose(accelerated.beta, stable.beta, rtol=2e-10, atol=2e-10)
    assert accelerated.intercept == pytest.approx(stable.intercept, rel=2e-11, abs=2e-11)
    assert accelerated.deviance == pytest.approx(stable.deviance, rel=2e-11, abs=2e-11)
    np.testing.assert_allclose(
        accelerated_second.beta,
        stable_second.beta,
        rtol=2e-10,
        atol=2e-10,
    )


def test_zero_width_non_spline_group_does_not_break_admitted_raw_plan() -> None:
    """Projected-away groups are ignored consistently by eligibility and width accounting."""
    rng = np.random.default_rng(1712)
    n = 8_000
    splines = [_spline_group(n, 15, phase=0), _spline_group(n, 15, phase=2)]
    dm = DesignMatrix(
        [DenseGroupMatrix(np.empty((n, 0))), *splines],
        n=n,
        p=30,
    )
    result, _ = fit_irls_direct(
        X=dm,
        y=rng.normal(size=n),
        weights=np.ones(n),
        family=Gaussian(),
        link=IdentityLink(),
        groups=[
            GroupSlice(name="empty", start=0, end=0),
            GroupSlice(name="spline_1", start=0, end=15),
            GroupSlice(name="spline_2", start=15, end=30),
        ],
        lambda2=1.0,
        S_override=0.2 * np.eye(30),
        compute_rank_info=False,
    )

    assert result.converged is True
    assert dm.raw_spline_tabmat_plan_built is True


def test_large_constant_weight_fit_defers_plan_unless_reml_will_reuse_it() -> None:
    rng = np.random.default_rng(1709)
    n = 50_000
    groups = [_spline_group(n, 15, phase=0), _spline_group(n, 15, phase=2)]
    dm = DesignMatrix(groups, n=n, p=30)
    beta_true = rng.normal(scale=0.05, size=30)
    y = 0.2 + dm.matvec(beta_true) + rng.normal(scale=0.2, size=n)
    slices = [
        GroupSlice(name="spline_1", start=0, end=15),
        GroupSlice(name="spline_2", start=15, end=30),
    ]
    common = dict(
        X=dm,
        y=y,
        weights=np.ones(n),
        family=Gaussian(),
        link=IdentityLink(),
        groups=slices,
        lambda2=1.0,
        S_override=0.2 * np.eye(30),
        compute_rank_info=False,
    )
    profile: dict[str, float | int] = {}

    ordinary, _ = fit_irls_direct(profile=profile, trace_purpose="fit", **common)

    assert ordinary.converged is True
    assert dm.raw_spline_tabmat_plan_built is False
    assert profile["centered_spline_tabmat_cold_policy_rejections"] == 1

    reml, _ = fit_irls_direct(
        profile=profile,
        trace_purpose="reml_candidate",
        beta_init=ordinary.beta,
        intercept_init=ordinary.intercept,
        **common,
    )
    assert reml.converged is True
    assert dm.raw_spline_tabmat_plan_built is True
    assert profile["centered_spline_tabmat_builds"] == 1
    attempts = profile["centered_spline_tabmat_attempts"]

    fit_irls_direct(
        profile=profile,
        trace_purpose="fit",
        beta_init=reml.beta,
        intercept_init=reml.intercept,
        **common,
    )
    assert profile["centered_spline_tabmat_attempts"] > attempts
    assert profile["centered_spline_tabmat_builds"] == 1

    final_dm = DesignMatrix(
        [_spline_group(n, 15, phase=0), _spline_group(n, 15, phase=2)],
        n=n,
        p=30,
    )
    final_profile: dict[str, float | int] = {}
    final, _ = fit_irls_direct(
        **{
            **common,
            "X": final_dm,
            "profile": final_profile,
            "trace_purpose": "reml_final",
            "beta_init": reml.beta,
            "intercept_init": reml.intercept,
        }
    )
    assert final.converged is True
    assert final_dm.raw_spline_tabmat_plan_built is False
    assert final_profile["centered_spline_tabmat_cold_policy_rejections"] == 1


def test_narrow_constant_weight_fit_defers_cold_plan_but_reml_can_reuse_it() -> None:
    rng = np.random.default_rng(1711)
    n = 8_000
    groups = [_spline_group(n, 12, phase=0), _spline_group(n, 12, phase=2)]
    dm = DesignMatrix(groups, n=n, p=24)
    slices = [
        GroupSlice(name="spline_1", start=0, end=12),
        GroupSlice(name="spline_2", start=12, end=24),
    ]
    common = dict(
        X=dm,
        y=0.2 + rng.normal(scale=0.2, size=n),
        weights=np.ones(n),
        family=Gaussian(),
        link=IdentityLink(),
        groups=slices,
        lambda2=1.0,
        S_override=0.2 * np.eye(24),
        compute_rank_info=False,
    )
    profile: dict[str, float | int] = {}

    ordinary, _ = fit_irls_direct(profile=profile, trace_purpose="fit", **common)
    assert ordinary.converged is True
    assert dm.raw_spline_tabmat_plan_built is False
    assert profile["centered_spline_tabmat_cold_policy_rejections"] == 1

    reml, _ = fit_irls_direct(
        profile=profile,
        trace_purpose="reml_candidate",
        beta_init=ordinary.beta,
        intercept_init=ordinary.intercept,
        **common,
    )
    assert reml.converged is True
    assert dm.raw_spline_tabmat_plan_built is True
    assert profile["centered_spline_tabmat_builds"] == 1
    attempts = profile["centered_spline_tabmat_attempts"]

    fit_irls_direct(
        profile=profile,
        trace_purpose="fit",
        beta_init=reml.beta,
        intercept_init=reml.intercept,
        **common,
    )
    assert profile["centered_spline_tabmat_attempts"] > attempts
    assert profile["centered_spline_tabmat_builds"] == 1


def test_cold_spline_policy_does_not_report_unrelated_dense_fit() -> None:
    rng = np.random.default_rng(1710)
    n = 50_000
    x = rng.normal(size=n)
    dm = DesignMatrix([DenseGroupMatrix(x)], n=n, p=1)
    profile: dict[str, float | int] = {}

    fit_irls_direct(
        X=dm,
        y=0.4 + 0.7 * x + rng.normal(scale=0.2, size=n),
        weights=np.ones(n),
        family=Gaussian(),
        link=IdentityLink(),
        groups=[GroupSlice(name="x", start=0, end=1)],
        lambda2=0.0,
        profile=profile,
        compute_rank_info=False,
    )

    assert "centered_spline_tabmat_cold_policy_rejections" not in profile
