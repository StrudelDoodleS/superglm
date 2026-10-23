"""The compiled NB2 near-Poisson series.

The series is the v0.40.0 per-row loop (``tests/_nb_series_reference.py``)
compiled operation for operation, so it is held to that loop's bits.
"""

from __future__ import annotations

import numpy as np
import pytest

import superglm.distributional.kernels._negative_binomial_numba as compiled
from superglm._parallel import parallel_config
from tests import _nb_series_reference as reference
from tests._distributional_family_kernels import negative_binomial as nb_kernel


def _stress_rows(seed: int, n_rows: int = 4000, max_count: float = 2.0**40):
    """Series candidates and near-candidates across the finite-theta domain."""
    rng = np.random.default_rng(seed)
    theta = np.exp(rng.uniform(-5.0, 25.0, n_rows))
    ratio = np.exp(rng.uniform(np.log(2.0**-40), np.log(1.0 / 15.0), n_rows))
    mean = theta * ratio
    counts = np.floor(
        rng.uniform(0.0, 1.0, n_rows) * (theta / 16.0 + 2.0) * rng.uniform(0, 1.1, n_rows)
    )
    counts = np.minimum(counts, max_count).astype(np.int64)
    return counts, mean, theta


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("order", [1, 2])
def test_compiled_series_is_the_reference_loop_bit_for_bit(seed: int, order: int) -> None:
    counts, mean, theta = _stress_rows(seed)
    _, remainder_error, remainder_scale = nb_kernel._ratio_remainder(mean / theta)
    arguments = (counts, mean, theta, remainder_error, remainder_scale, order)
    expected = reference._theta_series(*arguments)
    assert np.count_nonzero(np.any(expected[0], axis=1)) > 500
    for computed, oracle in zip(nb_kernel._theta_series(*arguments), expected, strict=True):
        assert computed.dtype == oracle.dtype
        np.testing.assert_array_equal(computed.view(np.uint8), oracle.view(np.uint8))


@pytest.mark.parametrize("workers", [1, 4])
def test_one_serial_pass_and_pooled_blocks_agree_bit_for_bit(workers: int) -> None:
    """The pooled blocks give the single serial pass's bits at every worker count."""
    counts, mean, theta = _stress_rows(5, 9000)
    candidate_scale = np.maximum(mean, np.maximum(counts.astype(float) - 1.0, 0.0))
    rows = np.flatnonzero(candidate_scale / theta <= nb_kernel._SERIES_RATIO).astype(np.int64)
    assert rows.size >= nb_kernel._PARALLEL_MIN_ROWS, "the pooled route must run"
    _, remainder_error, remainder_scale = nb_kernel._ratio_remainder(mean / theta)
    use = np.zeros((len(counts), 2), dtype=np.bool_)
    natural, direct = np.zeros((len(counts), 2)), np.zeros((len(counts), 2))
    compiled.theta_series_serial(
        rows,
        counts,
        mean,
        theta,
        candidate_scale,
        remainder_error,
        remainder_scale,
        2,
        nb_kernel._SERIES_ORDER,
        nb_kernel._SERIES_EXPONENTS,
        nb_kernel._SERIES_BERNOULLI,
        use,
        natural,
        direct,
    )
    with parallel_config(n_jobs=workers):
        pooled = nb_kernel._theta_series(counts, mean, theta, remainder_error, remainder_scale, 2)
    for serial, block in zip((use, natural, direct), pooled, strict=True):
        np.testing.assert_array_equal(serial.view(np.uint8), block.view(np.uint8))


def test_public_rows_are_unchanged_by_compiling_the_series(monkeypatch: pytest.MonkeyPatch) -> None:
    counts, mean, theta = _stress_rows(11, 3000, max_count=500.0)
    weights = np.ones(len(counts))
    compiled_rows = nb_kernel.evaluate_negative_binomial_rows(counts, mean, theta, weights, "prior")
    monkeypatch.setattr(nb_kernel, "_theta_series", reference._theta_series)
    reference_rows = nb_kernel.evaluate_negative_binomial_rows(
        counts, mean, theta, weights, "prior"
    )
    for name in ("optimizing_log_likelihood", "score", "hessian_packed"):
        np.testing.assert_array_equal(getattr(compiled_rows, name), getattr(reference_rows, name))


def test_series_rows_dispatch_to_the_compiled_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """No series row runs in Python: every candidate goes through one compiled call."""
    calls: list[int] = []
    serial = compiled.theta_series_serial

    def counted(rows, *arguments):
        calls.append(len(rows))
        return serial(rows, *arguments)

    monkeypatch.setattr(compiled, "theta_series_serial", counted)
    counts, mean, theta = _stress_rows(3, 7000, max_count=500.0)
    nb_kernel.evaluate_negative_binomial_rows(counts, mean, theta, np.ones(len(counts)), "prior")
    candidates = np.maximum(mean, np.maximum(counts - 1.0, 0.0)) / theta <= nb_kernel._SERIES_RATIO
    expected = int(np.count_nonzero(candidates))
    block = compiled._PARALLEL_BLOCK
    whole = [expected] if expected < nb_kernel._PARALLEL_MIN_ROWS else None
    assert calls == (whole or [min(block, expected - start) for start in range(0, expected, block)])
