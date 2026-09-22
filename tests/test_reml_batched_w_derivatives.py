"""Signed centered REML products: numerical invariants and separate dispatch checks."""

import gc
import math
import weakref

import numpy as np
import pytest
from scipy import sparse

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm._group_matrix import _group_matrix_centered
from superglm._group_matrix._group_matrix_centered import (
    centered_gram_rhs,
    centered_signed_grams,
)
from superglm.distributions import Poisson
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    SparseSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm.links import LogLink
from superglm.reml import w_derivatives
from superglm.solvers.hessian_factor import as_hessian_factor
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.types import GroupSlice, PenaltyComponent


def _kernel_fixture():
    n = 129
    t = (np.arange(n, dtype=float) - 64) / 16
    centered = np.column_stack((t, t[::-1], (np.arange(n) % 7) / 8))
    mean = np.array([2.0**30, -(2.0**29), 2.0**28])
    X = centered + mean
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=3)
    channels = [np.ones(n), np.where(np.arange(n) % 2, -1.0, 1.0), np.zeros(n)]
    return dm, centered, mean, channels


def _gram_bound(centered, weights):
    eps = np.finfo(float).eps
    gamma = (3 * len(weights) * eps) / (1 - 3 * len(weights) * eps)
    # Covers chunk products, compensated accumulation and symmetrization.
    return 8 * gamma * (np.abs(centered).T @ (np.abs(weights)[:, None] * np.abs(centered)))


def _serial_signed_grams(*, dm, weights, mean_x, chunk_size=8192):
    return [
        centered_gram_rhs(
            dm=dm, W=w, mean_x=mean_x, z_centered=np.zeros(dm.n), chunk_size=chunk_size
        )[0]
        for w in weights
    ]


def test_batched_signed_grams_match_centered_product_oracle():
    dm, centered, mean, channels = _kernel_fixture()
    results = centered_signed_grams(dm=dm, weights=channels, mean_x=mean, chunk_size=32)
    for weights, actual in zip(channels, results, strict=True):
        expected = np.array(
            [
                [math.fsum(weights * centered[:, j] * centered[:, k]) for k in range(dm.p)]
                for j in range(dm.p)
            ]
        )
        bound = _gram_bound(centered, weights)
        assert np.all(np.abs(actual - expected) <= bound)
        np.testing.assert_array_equal(actual, actual.T)
    # Mutation control: raw moments followed by large-mean subtraction fail.
    X = centered + mean
    raw = X.T @ X - np.outer(X.sum(axis=0), mean) - np.outer(mean, X.sum(axis=0))
    raw += dm.n * np.outer(mean, mean)
    assert np.any(np.abs(raw - results[0]) > _gram_bound(centered, channels[0]))


def test_batched_signed_grams_preserves_compensated_chunk_sum(monkeypatch):
    weights = np.concatenate(([2.0**53], np.ones(64), [-(2.0**53)]))
    dm = DesignMatrix([DenseGroupMatrix(np.ones((len(weights), 1)))], n=len(weights), p=1)
    kwargs = dict(dm=dm, weights=[weights], mean_x=np.zeros(1), chunk_size=1)
    expected = math.fsum(weights)
    # Each scalar product is exact. Compensation retains the unit residuals
    # exactly (they are -1, 0 or 1), and the final cancellation is exact.
    # Thus this fixture has no higher-order summation error; the usual
    # first-order compensated bound 2*eps*sum(abs(products)) suffices.
    bound = 2 * np.finfo(float).eps * math.fsum(np.abs(weights))
    actual = centered_signed_grams(**kwargs)[0][0, 0]
    assert abs(actual - expected) <= bound

    def ordinary_add(total, compensation, value):
        total += value

    monkeypatch.setattr(_group_matrix_centered, "_compensated_add", ordinary_add)
    mutated = centered_signed_grams(**kwargs)[0][0, 0]
    assert abs(mutated - expected) > bound


def test_batched_signed_grams_reuses_rows_and_preserves_inputs(monkeypatch):
    dm, centered, mean, channels = _kernel_fixture()
    original = DesignMatrix.row_subset
    calls = []

    def counted(self, rows):
        calls.append(rows.copy())
        return original(self, rows)

    monkeypatch.setattr(DesignMatrix, "row_subset", counted)
    saved = [w.copy() for w in channels]
    saved_mean = mean.copy()
    centered_signed_grams(dm=dm, weights=channels, mean_x=mean, chunk_size=32)
    assert len(calls) == 5
    calls.clear()
    _serial_signed_grams(dm=dm, weights=channels, mean_x=mean, chunk_size=32)
    assert len(calls) == 15  # Serial-wrapper mutation violates the five-call contract.
    for actual, expected in zip(channels, saved, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(mean, saved_mean)
    np.testing.assert_array_equal(dm.toarray(), centered + mean)


@pytest.mark.parametrize(
    "case", ["empty", "zero_columns", "weights", "mean", "chunk_zero", "chunk_negative"]
)
def test_batched_signed_grams_boundaries_without_materializing_rows(monkeypatch, case):
    dm, _, mean, channels = _kernel_fixture()
    chunk_size = 32
    if case == "empty":
        channels = []
    elif case == "zero_columns":
        dm = DesignMatrix([], n=129, p=0)
        mean = np.zeros(0)
    elif case == "weights":
        channels[1] = np.ones((129, 1))
    elif case == "mean":
        mean = np.zeros((3, 1))
    else:
        chunk_size = 0 if case == "chunk_zero" else -1

    def forbidden(*args):
        pytest.fail("boundary handling must precede row materialization")

    monkeypatch.setattr(DesignMatrix, "row_subset", forbidden)
    if case in {"empty", "zero_columns"}:
        result = centered_signed_grams(dm=dm, weights=channels, mean_x=mean, chunk_size=chunk_size)
        assert len(result) == len(channels)
        assert all(gram.shape == (0, 0) for gram in result)
    else:
        with pytest.raises(ValueError):
            centered_signed_grams(dm=dm, weights=channels, mean_x=mean, chunk_size=chunk_size)


def _correction_fixture(p=2, shift=1.0e8, storage=None):
    rng = np.random.default_rng(123)
    x = np.linspace(-1.5, 1.5, 320)
    X = np.column_stack((x, x**2 - np.mean(x**2), np.sin(3 * x)))[:, :p]
    if storage is not None:
        X[:, 2] = (np.arange(len(x)) % 3 == 0).astype(float)
    y = rng.poisson(np.exp(0.25 + X @ np.array([0.35, -0.15, 0.2])[:p])).astype(float)
    matrices = [DenseGroupMatrix(X[:, i : i + 1] + shift) for i in range(p)]
    if storage is not None:

        class CustomCategorical(CategoricalGroupMatrix):
            pass

        categorical = CustomCategorical if storage == "custom" else CategoricalGroupMatrix
        dtype = np.float32 if storage == "float32" else np.float64
        matrices = [
            SupportCompressedSSPGroupMatrix(
                X[:, :1].astype(dtype), np.ones((1, 1), dtype=dtype), np.arange(len(x))
            ),
            SparseSSPGroupMatrix(sparse.csr_matrix(X[:, 1:2]), np.ones((1, 1))),
            categorical(np.where(X[:, 2], 0, -1), n_levels=1),
        ]
    dm = DesignMatrix(matrices, n=len(x), p=p)
    groups = [GroupSlice(name=f"x{i}", start=i, end=i + 1) for i in range(p)]
    penalties = [
        PenaltyComponent(
            name=g.name,
            group_name=g.name,
            group_index=i,
            group_sl=slice(i, i + 1),
            omega_raw=np.ones((1, 1)),
            omega_ssp=np.ones((1, 1)),
            rank=1.0,
            log_det_omega_plus=0.0,
            eigvals_omega=np.ones(1),
        )
        for i, g in enumerate(groups)
    ]
    lambdas = {g.name: 4.0 + i for i, g in enumerate(groups)}
    weights, offset = np.ones_like(x), np.zeros_like(x)
    family, link = Poisson(), LogLink()
    result, inverse, _ = fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=family,
        link=link,
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        return_xtwx=True,
        reml_penalties=penalties,
        weight_semantics="frequency",
    )
    assert result.converged
    return dict(
        dm=dm,
        link=link,
        groups=groups,
        pirls_result=result,
        XtWX_S_inv=inverse,
        lambdas=lambdas,
        sample_weight=weights,
        offset_arr=offset,
        distribution=family,
        reml_penalties=penalties,
    )


@pytest.mark.parametrize("p,bounded", [(2, False), (3, True)])
def test_first_order_batch_matches_serial_grams_and_scalar_gradient(monkeypatch, p, bounded):
    kwargs = _correction_fixture(p=p)
    dm = kwargs["dm"]
    if bounded:
        bytes_per_direction = np.dtype(np.float64).itemsize * (dm.n + 3 * p * p)
        monkeypatch.setattr(w_derivatives, "_SIGNED_GRAM_BATCH_BYTES", 2 * bytes_per_direction)
    channels = []
    batches = []

    def recorded(**kernel_kwargs):
        batches.append(len(kernel_kwargs["weights"]))
        channels.extend(w.copy() for w in kernel_kwargs["weights"])
        return centered_signed_grams(**kernel_kwargs)

    monkeypatch.setattr(w_derivatives, "centered_signed_grams", recorded)
    actual = w_derivatives.reml_w_correction(**kwargs)
    assert batches == ([2, 1] if bounded else [2])
    monkeypatch.setattr(w_derivatives, "centered_signed_grams", _serial_signed_grams)
    expected = w_derivatives.reml_w_correction(**kwargs)
    inverse = as_hessian_factor(kwargs["XtWX_S_inv"]).inverse
    metadata = kwargs["pirls_result"].rank_info
    centered = dm.toarray() - metadata.mean_x
    eps = np.finfo(float).eps
    for i, weights in enumerate(channels):
        bound = 2 * _gram_bound(centered, weights)
        assert np.all(np.abs(actual[1][i] - expected[1][i]) <= bound)
        trace_bound = 0.5 * np.sum(np.abs(inverse) * bound)
        scalar = 0.5 * math.fsum(weights) / metadata.sum_w
        trace = 0.5 * np.sum(inverse * actual[1][i])
        reduction_bound = (
            8
            * dm.n
            * eps
            * (np.sum(np.abs(inverse * actual[1][i])) + np.sum(np.abs(weights)) / metadata.sum_w)
        )
        assert abs(actual[0][i] - expected[0][i]) <= trace_bound + reduction_bound
        assert abs(actual[0][i] - (trace + scalar)) <= reduction_bound
        assert abs(scalar) > reduction_bound  # Omitting log(sum(W)) is observable.


@pytest.mark.parametrize("order", [1, 2])
def test_only_second_order_computes_mean_derivative_transposes(monkeypatch, order):
    kwargs = _correction_fixture()
    original = DesignMatrix.rmatvec
    calls = 0

    def counted(self, values):
        nonlocal calls
        calls += 1
        return original(self, values)

    monkeypatch.setattr(DesignMatrix, "rmatvec", counted)
    correction = w_derivatives.reml_w_correction(**kwargs, w_correction_order=order)
    assert correction is not None
    assert calls == (0 if order == 1 else 5)  # Two means and three Hessian cross-terms.


@pytest.mark.parametrize(
    "route", ["well_scaled", "single", "second_order", "small_budget", "gradient_only"]
)
def test_other_routes_do_not_batch_signed_grams(monkeypatch, route):
    kwargs = _correction_fixture(
        p=1 if route == "single" else 2, shift=0 if route == "well_scaled" else 1e8
    )
    if route == "small_budget":
        monkeypatch.setattr(w_derivatives, "_SIGNED_GRAM_BATCH_BYTES", 1)

    def forbidden(**kwargs):
        pytest.fail("this route must not use the batched kernel")

    monkeypatch.setattr(w_derivatives, "centered_signed_grams", forbidden)
    result = w_derivatives.reml_w_correction(
        **kwargs,
        w_correction_order=2 if route == "second_order" else 1,
        gradient_only=route == "gradient_only",
    )
    assert result is not None


def test_derivative_channels_project_each_fixed_support_once(monkeypatch):
    # Removing derivative-local reuse repeats this real B_unique @ R_inv.
    kwargs = _correction_fixture(p=3, shift=0, storage="mixed")
    original = algebra._BlockWeightCache.solver_support
    builds = []
    group = kwargs["dm"].group_matrices[0]
    bounds = algebra._operand_exponent_bounds
    scans = {id(group.B_unique): 0, id(group.R_inv): 0}

    def counted(cache, group):
        if group not in cache._supports:
            builds.append(group)
        return original(cache, group)

    def scanned(values):
        if id(values) in scans:
            scans[id(values)] += 1
        return bounds(values)

    monkeypatch.setattr(algebra._BlockWeightCache, "solver_support", counted)
    monkeypatch.setattr(algebra, "_operand_exponent_bounds", scanned)
    result = w_derivatives.reml_w_correction(**kwargs)
    assert result is not None
    assert len(result[1]) == 3
    assert len(builds) == 1
    assert list(scans.values()) == [1, 1]


def _assert_poisson_correction(kwargs, correction):
    """Independent literal centered products, with dimension-scaled bounds."""
    dm = kwargs["dm"]
    result = kwargs["pirls_result"]
    centered = dm.toarray() - result.rank_info.mean_x
    dW = kwargs["sample_weight"] * np.exp(dm.matvec(result.beta) + result.intercept)
    inverse = as_hessian_factor(kwargs["XtWX_S_inv"])
    for i, pc in enumerate(kwargs["reml_penalties"]):
        rhs = np.zeros(dm.p)
        rhs[pc.group_sl] = kwargs["lambdas"][pc.name] * result.beta[pc.group_sl]
        weights = dW * (centered @ -inverse.solve(rhs))
        expected = np.array(
            [
                [math.fsum(weights * centered[:, j] * centered[:, k]) for k in range(dm.p)]
                for j in range(dm.p)
            ]
        )
        bound = 4 * _gram_bound(centered, weights)
        assert np.all(np.abs(correction[1][i] - expected) <= bound)
        trace = 0.5 * np.sum(inverse.inverse * expected)
        scalar = 0.5 * math.fsum(weights) / result.rank_info.sum_w
        allowance = 0.5 * np.sum(np.abs(inverse.inverse) * bound)
        allowance += 32 * dm.n * np.finfo(float).eps * (abs(trace) + abs(scalar))
        assert abs(correction[0][i] - trace - scalar) <= allowance


@pytest.mark.parametrize("replace", [False, True])
def test_derivative_reuse_observes_factors_and_weights_on_next_call(monkeypatch, replace):
    kwargs = _correction_fixture(p=3, shift=0, storage="mixed")
    group = kwargs["dm"].group_matrices[0]
    original = algebra._BlockWeightCache.solver_support
    projected = []

    def recorded(cache, group):
        result = original(cache, group)
        projected.append(weakref.ref(result))
        return result

    monkeypatch.setattr(algebra._BlockWeightCache, "solver_support", recorded)
    for iteration in range(2):
        if iteration:
            for name in ("B_unique", "R_inv"):
                changed = getattr(group, name) * 0.5
                if replace:
                    setattr(group, name, changed)
                else:
                    getattr(group, name)[:] = changed
            kwargs["sample_weight"] *= 2
        correction = w_derivatives.reml_w_correction(**kwargs)
        _assert_poisson_correction(kwargs, correction)
        assert projected and all(ref() is None for ref in projected)


@pytest.mark.parametrize("storage,builds_expected", [("custom", 3), ("float32", 0)])
def test_derivative_reuse_keeps_custom_and_dtype_fallback(monkeypatch, storage, builds_expected):
    kwargs = _correction_fixture(p=3, shift=0, storage=storage)
    original = algebra._BlockWeightCache.solver_support
    builds = []

    def counted(cache, group):
        if group not in cache._supports:
            builds.append(group)
        return original(cache, group)

    monkeypatch.setattr(algebra._BlockWeightCache, "solver_support", counted)
    correction = w_derivatives.reml_w_correction(**kwargs)
    _assert_poisson_correction(kwargs, correction)
    assert len(builds) == builds_expected


def test_derivative_weighted_cache_sharing_mutation_is_detected(monkeypatch):
    kwargs = _correction_fixture(p=3, shift=0, storage="mixed")
    # Reusing the weighted sparse Gram is wrong even though the design is fixed.
    monkeypatch.setattr(algebra._BlockWeightCache, "for_new_weights", lambda self: self)
    correction = w_derivatives.reml_w_correction(**kwargs)
    with pytest.raises(AssertionError):
        _assert_poisson_correction(kwargs, correction)


def test_derivative_reuse_releases_support_after_error(monkeypatch):
    kwargs = _correction_fixture(p=3, shift=0, storage="mixed")
    original = algebra._BlockWeightCache.solver_support
    projected = []
    calls = 0

    def interrupted(cache, group):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted derivative")
        result = original(cache, group)
        projected.append(weakref.ref(result))
        return result

    monkeypatch.setattr(algebra._BlockWeightCache, "solver_support", interrupted)
    with pytest.raises(RuntimeError, match="interrupted derivative"):
        w_derivatives.reml_w_correction(**kwargs)
    gc.collect()
    assert projected and all(ref() is None for ref in projected)


@pytest.mark.parametrize("weights", [np.zeros((320, 1)), np.full(320, np.nan)])
def test_fixed_support_moments_keep_weight_validation(weights):
    kwargs = _correction_fixture(p=3, shift=0, storage="mixed")
    plan = kwargs["dm"].execution_plan
    owner = plan._fixed_support_cache()
    with pytest.raises(ValueError):
        plan._signed_moments_fixed_support(weights, owner)
