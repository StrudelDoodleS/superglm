"""Signed centered REML products: numerical invariants and separate dispatch checks."""

import gc
import math
import weakref

import numpy as np
import pytest
from scipy import sparse

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm._group_matrix import _group_matrix_centered
from superglm._group_matrix import _group_matrix_kernels as kernels
from superglm._group_matrix._group_matrix_centered import (
    centered_gram_rhs,
    centered_signed_grams,
)
from superglm.distributions import Poisson
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedTensorGroupMatrix,
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
    monkeypatch.setattr(algebra._BlockWeightCache, "for_new_weights", lambda self, *_: self)
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


def _order_sensitive(rng, n, m):
    """Signed weights over twelve decades: another summation order changes the bits."""
    return rng.normal(size=(n, m)) * 10.0 ** rng.integers(-6, 7, size=(n, m))


def _channel_kernel_cases(rows):
    """Each single-weight row pass beside its channel kernel, over ``rows`` of one fixture."""
    rng = np.random.default_rng(8801)
    n = 400
    bins_a, bins_b = rng.integers(0, 6, n)[rows], rng.integers(0, 4, n)[rows]
    codes = rng.integers(0, 4, n)[rows]  # three levels and their sink code
    other = rng.integers(0, 3, n)[rows]
    bases = []
    for width in (5, 4):
        values = rng.normal(size=(n, width))
        values[rng.random(size=values.shape) < 0.5] = 0.0
        bases.append(sparse.csr_matrix(values[rows]))
    csr = (bases[0].data, bases[0].indices, bases[0].indptr)
    pair = (*csr, bases[1].data, bases[1].indices, bases[1].indptr)
    return {
        "hist": (
            lambda w: kernels._disc_disc_2d_hist(bins_a, bins_b, w, 6, 4),
            lambda W, s, k: kernels._weighted_hist_channels(bins_a, bins_b, W, s, k, 6, 4),
        ),
        "categorical": (
            lambda w: kernels._cat_weighted_bincount(codes, bins_a, w, 6, 3),
            lambda W, s, k: kernels._weighted_hist_channels(bins_a, codes, W, s, k, 6, 3),
        ),
        "crosstab": (
            lambda w: kernels._cat_cat_weighted_crosstab(codes, other, w, 3, 2),
            lambda W, s, k: kernels._weighted_hist_channels(codes, other, W, s, k, 3, 2),
        ),
        "bincount": (
            lambda w: np.bincount(codes, weights=w, minlength=4),
            lambda W, s, k: kernels._weighted_hist_channels(codes, None, W, s, k, 4, 1)[:, 0],
        ),
        "csr_gram": (
            lambda w: kernels._csr_weighted_gram(*csr, w, 5),
            lambda W, s, k: kernels._csr_weighted_gram_channels(*csr, W, s, k, 5, False),
        ),
        "csr_energy": (
            lambda w: kernels._csr_weighted_gram(*csr, w, 5, True),
            lambda W, s, k: kernels._csr_weighted_gram_channels(*csr, W, s, k, 5, True),
        ),
        "csr_cross": (
            lambda w: kernels._csr_weighted_cross(*pair, w, 5, 4),
            lambda W, s, k: kernels._csr_weighted_cross_channels(*pair, W, s, k, 5, 4),
        ),
        "csr_bincount": (
            lambda w: kernels._csr_weighted_bincount(*csr, 5, bins_a, w, 6),
            lambda W, s, k: kernels._csr_weighted_bincount_channels(*csr, 5, bins_a, W, s, k, 6),
        ),
    }


@pytest.mark.parametrize(
    "case",
    [
        "hist",
        "categorical",
        "crosstab",
        "bincount",
        "csr_gram",
        "csr_energy",
        "csr_cross",
        "csr_bincount",
    ],
)
def test_channel_kernels_repeat_each_single_weight_sum_bitwise(case):
    block = _order_sensitive(np.random.default_rng(8803), 400, 5)
    single, multi = _channel_kernel_cases(slice(None))[case]
    result = multi(block, 1, 3)
    assert result.shape[-1] == 3
    for channel in range(3):
        expected = single(np.ascontiguousarray(block[:, 1 + channel]))
        np.testing.assert_array_equal(result[..., channel], expected)
    # Control: the same terms added in reverse row order give other bits, so
    # equality above certifies each channel's row order, not just its value.
    weights = np.ascontiguousarray(block[:, 1])
    reverse = _channel_kernel_cases(slice(None, None, -1))[case][0]
    assert not np.array_equal(reverse(weights[::-1].copy()), single(weights))


def _channel_design(n=500):
    """Every batched block: support grids, categoricals, sparse SSP and a dense fallback."""
    rng = np.random.default_rng(8810)

    def support(bins, width):
        B, R = rng.normal(size=(bins, width)), rng.normal(size=(width, width - 1))
        return SupportCompressedSSPGroupMatrix(B, R, rng.integers(0, bins, n))

    def spline(width):
        values = rng.normal(size=(n, width))
        values[rng.random(size=values.shape) < 0.5] = 0.0
        return SparseSSPGroupMatrix(sparse.csr_matrix(values), rng.normal(size=(width, width - 1)))

    groups = [
        support(9, 4),
        CategoricalGroupMatrix(rng.integers(-1, 3, n), n_levels=3),
        support(11, 3),
        spline(5),
        CategoricalGroupMatrix(rng.integers(-1, 4, n), n_levels=4),
        spline(4),
        DenseGroupMatrix(rng.normal(size=(n, 2))),
    ]
    return DesignMatrix(groups, n=n, p=sum(g.shape[1] for g in groups))


_CHANNEL_KERNELS = (
    "_weighted_hist_channels",
    "_csr_weighted_gram_channels",
    "_csr_weighted_cross_channels",
    "_csr_weighted_bincount_channels",
)


def _record_channel_passes(monkeypatch):
    """Record each channel kernel call as (kernel, first channel, channel count)."""
    passes = []
    for name in _CHANNEL_KERNELS:
        kernel = getattr(algebra, name)

        def recorded(*args, _name=name, _kernel=kernel):
            block = next(i for i, a in enumerate(args) if getattr(a, "ndim", 0) == 2)
            passes.append((_name, args[block + 1], args[block + 2]))
            return _kernel(*args)

        monkeypatch.setattr(algebra, name, recorded)
    return passes


@pytest.mark.parametrize("fixed_support", [True, False])
def test_batched_moments_equal_each_direction_bitwise(monkeypatch, fixed_support):
    dm = _channel_design()
    plan = dm.execution_plan
    directions = list(_order_sensitive(np.random.default_rng(8811), dm.n, 4).T.copy())
    owner = plan._fixed_support_cache() if fixed_support else None
    assert (owner is not None) is fixed_support

    def serial():
        return [
            plan._signed_moments_fixed_support(w, plan._fixed_support_cache())
            if fixed_support
            else plan.moments(w, include_xtw=True, signed=True)
            for w in directions
        ]

    expected = serial()
    passes = _record_channel_passes(monkeypatch)
    actual = plan._signed_moments_channels(directions, owner)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got.gram, want.gram)
        np.testing.assert_array_equal(got.xtw, want.xtw)
        assert got.xt_rhs == ()
    # Every batched kernel ran, each pass serving all four directions at once.
    assert {name for name, _, _ in passes} == set(_CHANNEL_KERNELS)
    assert {(start, width) for _, start, width in passes} == {(0, 4)}


def test_channel_groups_follow_the_pass_budget(monkeypatch):
    dm = _channel_design()
    plan = dm.execution_plan
    directions = list(_order_sensitive(np.random.default_rng(8812), dm.n, 5).T.copy())
    expected = [plan.moments(w, include_xtw=True, signed=True) for w in directions]
    grids = []
    kernel = algebra._weighted_hist_channels

    def recorded(idx_a, idx_b, W, start, width, n_a, n_b):
        if (n_a, n_b) == (9, 11):
            grids.append((start, width))
        return kernel(idx_a, idx_b, W, start, width, n_a, n_b)

    monkeypatch.setattr(algebra, "_weighted_hist_channels", recorded)
    # The 9-by-11 support grid fits two channels; its fifth takes the single kernel.
    monkeypatch.setattr(algebra, "_CHANNEL_PASS_BYTES", 2 * 8 * 99)
    actual = plan._signed_moments_channels(directions, None)
    assert grids == [(0, 2), (2, 2)]
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got.gram, want.gram)
    # Below one channel every block keeps its single-weight kernel.
    monkeypatch.setattr(algebra, "_CHANNEL_PASS_BYTES", 0)
    passes = _record_channel_passes(monkeypatch)
    actual = plan._signed_moments_channels(directions, None)
    assert passes == []
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got.gram, want.gram)


def test_batched_grids_reform_the_layout_that_reuse_returns():
    rng = np.random.default_rng(8813)
    idx_a, idx_b = rng.integers(0, 5, 64), rng.integers(0, 7, 64)
    W, other = _order_sensitive(rng, 64, 2).T.copy()
    plain = algebra._BlockWeightCache()
    batch = algebra._ChannelBatch([W, other])
    cache = algebra._BlockWeightCache().for_new_weights(batch, 0)
    for first, second in (((idx_a, idx_b, 5, 7), (idx_b, idx_a, 7, 5)),) * 2:
        want = [plain.disc_disc_hist(*first[:2], W, *first[2:])]
        want.append(plain.disc_disc_hist(*second[:2], W, *second[2:]))
        got = [cache.disc_disc_hist(*first[:2], W, *first[2:])]
        batch.release([cache])  # A batched channel keeps no grid between blocks.
        got.append(cache.disc_disc_hist(*second[:2], W, *second[2:]))
        batch.release([cache])
        for g, w in zip(got, want, strict=True):
            np.testing.assert_array_equal(g, w)
            assert (g.flags.c_contiguous, g.flags.f_contiguous) == (
                w.flags.c_contiguous,
                w.flags.f_contiguous,
            )
    assert not want[1].flags.c_contiguous  # Reuse returned the retained grid's transpose.
    assert not cache._hist2d
    # Weights outside the batch, such as a derived vector, keep their own pass.
    third = _order_sensitive(rng, 64, 1)[:, 0]
    np.testing.assert_array_equal(
        cache.disc_disc_hist(idx_a, idx_b, third, 5, 7),
        kernels._disc_disc_2d_hist(idx_a, idx_b, third, 5, 7),
    )


def test_batched_bin_sums_keep_same_size_indices_apart_within_a_block(monkeypatch):
    rng = np.random.default_rng(8814)
    first, second = rng.integers(0, 6, 64), rng.integers(0, 6, 64)
    W, other = _order_sensitive(rng, 64, 2).T.copy()
    passes, kernel = [], algebra._weighted_hist_channels
    monkeypatch.setattr(
        algebra, "_weighted_hist_channels", lambda *a: passes.append(a[0]) or kernel(*a)
    )
    batch = algebra._ChannelBatch([W, other])
    caches = [algebra._BlockWeightCache().for_new_weights(batch, c) for c in range(2)]
    # Both indices have six bins; each channel's second request must not
    # return the first index's pass.
    for cache, weights in zip(caches, (W, other), strict=True):
        for idx in (first, second):
            np.testing.assert_array_equal(
                cache.batch_bin_sums(idx, weights, 6), np.bincount(idx, weights, minlength=6)
            )
    batch.release(caches)
    # One two-channel pass per index.
    assert len(passes) == 2 and passes[0] is first and passes[1] is second


@pytest.mark.parametrize("per_call", [None, 2])
def test_moment_route_requests_the_directions_together(monkeypatch, per_call):
    kwargs = _correction_fixture(p=3, shift=0, storage="mixed")
    dm = kwargs["dm"]
    monkeypatch.setattr(w_derivatives, "_SIGNED_GRAM_BATCH_BYTES", 1)
    expected = w_derivatives.reml_w_correction(**kwargs)
    monkeypatch.undo()
    if per_call is not None:
        bytes_per_direction = np.dtype(np.float64).itemsize * (2 * dm.n + 3 * dm.p * dm.p)
        monkeypatch.setattr(
            w_derivatives, "_SIGNED_GRAM_BATCH_BYTES", per_call * bytes_per_direction
        )
    plan_type = type(dm.execution_plan)
    original = plan_type._signed_moments_channels
    requests = []

    def recorded(plan, weights, owner):
        requests.append(len(weights))
        return original(plan, weights, owner)

    monkeypatch.setattr(plan_type, "_signed_moments_channels", recorded)
    actual = w_derivatives.reml_w_correction(**kwargs)
    assert actual is not None and expected is not None
    assert requests == ([3] if per_call is None else [2, 1])
    np.testing.assert_array_equal(actual[0], expected[0])
    grams, reference = actual[1], expected[1]
    assert grams is not None and reference is not None
    for i in range(3):
        np.testing.assert_array_equal(grams[i], reference[i])


def _support_correction_fixture(tensor):
    """An exact Poisson fit on four lossless supports and a categorical.

    With ``tensor``, a discretized tensor joins them: a design outside the
    shared-owner set, so each direction would project the supports itself.
    """
    rng = np.random.default_rng(8830)
    n, width = 8_000, 10
    matrices = [
        SupportCompressedSSPGroupMatrix(
            rng.normal(size=(n, width)) / 4, np.eye(width), np.arange(n)
        )
        for _ in range(4)
    ]
    if tensor:
        bins, k = 24, 4
        margins = rng.normal(size=(bins, k)) / 2, rng.normal(size=(bins, k)) / 2
        idx1, idx2 = rng.integers(0, bins, n), rng.integers(0, bins, n)
        cells, pair_idx = np.unique(idx1 * bins + idx2, return_inverse=True)
        joint = (
            margins[0][cells // bins][:, :, None] * margins[1][cells % bins][:, None, :]
        ).reshape(len(cells), k * k)
        matrices.append(
            DiscretizedTensorGroupMatrix(
                *margins, idx1, idx2, joint, np.eye(k * k), pair_idx, tensor_id=1
            )
        )
    matrices.append(CategoricalGroupMatrix(rng.integers(-1, 5, n), n_levels=5))
    widths = [matrix.shape[1] for matrix in matrices]
    starts = np.concatenate(([0], np.cumsum(widths)))
    dm = DesignMatrix(matrices, n=n, p=int(starts[-1]))
    groups = [
        GroupSlice(name=f"g{i}", start=int(starts[i]), end=int(starts[i + 1]))
        for i in range(len(matrices))
    ]
    penalties = [
        PenaltyComponent(
            name=g.name,
            group_name=g.name,
            group_index=i,
            group_sl=slice(g.start, g.end),
            omega_raw=np.eye(widths[i]),
            omega_ssp=np.eye(widths[i]),
            rank=float(widths[i]),
            log_det_omega_plus=0.0,
            eigvals_omega=np.ones(widths[i]),
        )
        for i, g in enumerate(groups)
    ]
    y = rng.poisson(np.exp(0.2 + 0.3 * dm.matvec(rng.normal(size=dm.p) / 4))).astype(float)
    lambdas = {g.name: 2.0 + i for i, g in enumerate(groups)}
    weights, offset = np.ones(n), np.zeros(n)
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


@pytest.mark.parametrize("tensor", [True, False], ids=["tensor-no-owner", "shared-owner"])
def test_batched_directions_stay_within_their_memory_budget(monkeypatch, tensor):
    """A batch holds no more than its per-direction charge beyond one direction's peak.

    Without a shared owner every direction projects the supports itself, so a
    batch kept one projection set per direction alive at once; such designs
    now take the directions one at a time.
    """
    import tracemalloc

    kwargs = _support_correction_fixture(tensor)
    dm = kwargs["dm"]
    plan_type = type(dm.execution_plan)
    assert (dm.execution_plan._fixed_support_cache() is None) is tensor
    original, requests = plan_type._signed_moments_channels, []

    def recorded(plan, weights, owner):
        requests.append(len(weights))
        return original(plan, weights, owner)

    def traced_peak(budget):
        with monkeypatch.context() as patch:
            patch.setattr(plan_type, "_signed_moments_channels", recorded)
            if budget is not None:
                patch.setattr(w_derivatives, "_SIGNED_GRAM_BATCH_BYTES", budget)
            gc.collect()
            tracemalloc.start()
            try:
                result = w_derivatives.reml_w_correction(**kwargs)
                return tracemalloc.get_traced_memory()[1], result
            finally:
                tracemalloc.stop()

    traced_peak(None)  # load every kernel the batch reaches before measuring
    requests.clear()
    serial, expected = traced_peak(1)  # every direction on its own
    assert requests == []
    batched, actual = traced_peak(None)
    m = len(kwargs["reml_penalties"])
    # The charge per direction: retained weights and their stacked copy (2n),
    # and the raw and centred Grams (3p^2), as w_derivatives budgets them.
    # Beyond it, each block of this design makes at most one shared pass,
    # live at most _CHANNEL_PASS_BYTES, beside one channel's copied slice.
    charge = np.dtype(np.float64).itemsize * (2 * dm.n + 3 * dm.p * dm.p)
    assert batched <= serial + m * charge + 3 * algebra._CHANNEL_PASS_BYTES // 2
    assert requests == ([] if tensor else [m])
    np.testing.assert_array_equal(actual[0], expected[0])
