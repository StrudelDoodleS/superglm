"""Row-local mathematical tests for the internal Tweedie LSS point kernel."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from numba import njit  # type: ignore[import-untyped]

from superglm import _tweedie_profile_kernel as profile_kernel
from tests._distributional_family_kernels import tweedie as tweedie_kernel
from tests._tweedie_lss_oracles import (
    CENTERED_RHO_MOMENT_ORACLE,
    FROZEN_ADVERSARIAL_EVALUATION,
    FROZEN_CUSTOM_CUTOFF_CASES,
    FROZEN_CUTOFF_CAP_BOUNDARIES,
    FROZEN_HUGE_CAP_EVALUATION,
    FROZEN_TWEEDIE_REFUSALS,
    TWEEDIE_LSS_CASES,
    TWEEDIE_OUTSIDE_POWER_RANGE_REFUSAL_CASES,
    TWEEDIE_POWER_RANGE_ZERO_CASES,
    TWEEDIE_UPPER_POWER_RANGE_SCALAR_DENSITY_ORACLE,
    FrozenTweedieCutoffCase,
    FrozenTweedieEvaluation,
    TweedieLSSOracleCase,
)

TweedieNumericalRefusal = tweedie_kernel.TweedieNumericalRefusal
evaluate_tweedie_rows = tweedie_kernel.evaluate_tweedie_rows


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_tweedie_initializer_is_a_primitive_kernel_operation(semantics: str) -> None:
    response = np.array([0.2, 0.7, 1.8, 3.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    theta = tweedie_kernel.initialize_tweedie(
        response,
        weights,
        semantics,
        power_lower=1.05,
        power_upper=1.95,
    )

    assert theta.shape == (4, 3)
    assert np.all(theta[:, :2] > 0.0)
    assert np.all((theta[:, 2] > 1.05) & (theta[:, 2] < 1.95))


def _case_arrays(case: TweedieLSSOracleCase) -> tuple[np.ndarray, ...]:
    return tuple(
        np.array([value], dtype=np.float64)
        for value in (case.y, case.mean, case.dispersion, case.power, case.weight)
    )


def _evaluate_case(case: TweedieLSSOracleCase, *, derivative_order: int = 2, **kwargs):
    return evaluate_tweedie_rows(
        *_case_arrays(case),
        case.semantics,
        derivative_order=derivative_order,
        **kwargs,
    )


def _dispatch_inputs(rows: int) -> tuple[np.ndarray, ...]:
    return (
        np.resize(np.array([0.0, 0.4, 1.7, 4.0], dtype=np.float64), rows),
        np.resize(np.array([0.8, 0.6, 1.4, 2.5], dtype=np.float64), rows),
        np.resize(np.array([0.4, 0.3, 1.2, 0.8], dtype=np.float64), rows),
        np.resize(np.array([1.4, 1.2, 1.5, 1.8], dtype=np.float64), rows),
        np.resize(np.array([1.0, 1.0, 0.7, 2.0], dtype=np.float64), rows),
    )


def test_ordinary_720_row_batch_uses_serial_nopython_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_parallel_core(*_args):
        raise AssertionError("ordinary batch called the parallel compiled core")

    monkeypatch.setattr(
        tweedie_kernel,
        "_evaluate_tweedie_batch_parallel_core",
        forbidden_parallel_core,
    )
    rows = 720
    result = evaluate_tweedie_rows(
        *_dispatch_inputs(rows),
        "prior",
        derivative_order=2,
    )

    assert np.all(result.valid)
    assert np.all(np.isfinite(result.log_likelihood))
    serial_core = tweedie_kernel._evaluate_tweedie_batch_core
    assert serial_core.nopython_signatures
    assert type(serial_core._cache).__name__ == "FunctionCache"
    target_options = serial_core.targetoptions
    assert target_options.get("fastmath", False) is False
    assert target_options.get("parallel", False) is False


@pytest.mark.parametrize(
    ("rows", "forbidden_core"),
    [
        (4_999, "_evaluate_tweedie_batch_parallel_core"),
        (5_000, "_evaluate_tweedie_batch_core"),
    ],
)
def test_production_dispatch_switches_at_measured_crossover_with_exact_outputs(
    rows: int,
    forbidden_core: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _dispatch_inputs(rows)
    serial = tweedie_kernel._compiled._evaluate_tweedie_batch_core(
        *arrays,
        0,
        2,
        100_000,
        37.0,
    )
    assert serial[5:] == (0, -1)

    def forbidden(*_args):
        raise AssertionError(f"{rows}-row batch selected {forbidden_core}")

    monkeypatch.setattr(tweedie_kernel, forbidden_core, forbidden)
    result = evaluate_tweedie_rows(
        *arrays,
        "prior",
        derivative_order=2,
    )

    np.testing.assert_array_equal(result.log_likelihood, serial[0])
    np.testing.assert_array_equal(result.score, serial[1])
    np.testing.assert_array_equal(result.hessian_packed, serial[2])
    np.testing.assert_array_equal(result.terms, serial[3])
    np.testing.assert_array_equal(result.valid, serial[4])
    if rows == 5_000:
        parallel_core = tweedie_kernel._compiled._evaluate_tweedie_batch_parallel_core
        assert parallel_core.nopython_signatures
        assert parallel_core.targetoptions.get("parallel", False) is True
        assert parallel_core.targetoptions.get("fastmath", False) is False


def test_public_warmup_normalizes_a_compiled_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import superglm

    def refusing_core(*_args):
        return (None, None, None, None, None, tweedie_kernel._compiled.KERNEL_MODE_RATIO, 0)

    monkeypatch.setattr(tweedie_kernel, "_evaluate_tweedie_batch_core", refusing_core)

    with pytest.raises(
        TweedieNumericalRefusal,
        match="row 0: series mode ratio is not representable",
    ):
        superglm.warmup()


@pytest.mark.parametrize("case", TWEEDIE_LSS_CASES, ids=lambda case: case.id)
def test_point_kernel_matches_independent_normalized_oracle(
    case: TweedieLSSOracleCase,
) -> None:
    result = _evaluate_case(case)

    assert result.valid.tolist() == [True]
    np.testing.assert_allclose(
        result.log_likelihood,
        [case.value],
        rtol=0.0,
        atol=case.value_atol,
    )
    assert result.score is not None
    np.testing.assert_allclose(
        result.score[0],
        case.score,
        rtol=case.score_rtol,
        atol=case.score_atol,
    )
    assert result.hessian_packed is not None
    np.testing.assert_allclose(
        result.hessian_packed[0],
        case.hessian,
        rtol=case.hessian_rtol,
        atol=case.hessian_atol,
    )
    if case.y > 0.0:
        assert 0 < result.terms[0] <= 100_000
    else:
        assert result.terms.tolist() == [0]


@pytest.mark.parametrize("case", TWEEDIE_POWER_RANGE_ZERO_CASES, ids=lambda case: case.id)
def test_exact_zero_route_matches_analytic_power_boundary_oracle(
    case: TweedieLSSOracleCase,
) -> None:
    # max_terms=1 would refuse every positive row, so success also proves that
    # zero rows never enter the positive-density series.
    result = _evaluate_case(case, max_terms=1)

    np.testing.assert_allclose(result.log_likelihood, [case.value], rtol=5e-14, atol=5e-14)
    assert result.score is not None
    np.testing.assert_allclose(result.score[0], case.score, rtol=5e-13, atol=5e-13)
    assert result.hessian_packed is not None
    np.testing.assert_allclose(result.hessian_packed[0], case.hessian, rtol=5e-13, atol=5e-13)
    assert result.terms.tolist() == [0]


@pytest.mark.parametrize(
    "case", TWEEDIE_OUTSIDE_POWER_RANGE_REFUSAL_CASES, ids=lambda case: case.id
)
def test_supported_external_wall_rows_are_typed_power_range_refusals(case) -> None:
    arrays = tuple(
        np.array([value], dtype=np.float64)
        for value in (case.y, case.mean, case.dispersion, case.power, case.weight)
    )

    with pytest.raises(TweedieNumericalRefusal, match="certified power range"):
        evaluate_tweedie_rows(*arrays, case.semantics, derivative_order=2)


def test_positive_upper_power_boundary_matches_external_scalar_density() -> None:
    case = TWEEDIE_UPPER_POWER_RANGE_SCALAR_DENSITY_ORACLE
    arrays = tuple(
        np.array([value], dtype=np.float64)
        for value in (case.y, case.mean, case.dispersion, case.power, case.weight)
    )

    result = evaluate_tweedie_rows(
        *arrays,
        case.semantics,
        derivative_order=0,
    )

    np.testing.assert_allclose(
        result.log_likelihood,
        [case.value],
        rtol=0.0,
        atol=case.value_atol,
    )


def test_derivative_orders_share_the_exact_value_window_and_suppress_work() -> None:
    case = TWEEDIE_LSS_CASES[2]
    order_zero = _evaluate_case(case, derivative_order=0)
    order_one = _evaluate_case(case, derivative_order=1)
    order_two = _evaluate_case(case, derivative_order=2)

    np.testing.assert_array_equal(order_zero.log_likelihood, order_one.log_likelihood)
    np.testing.assert_array_equal(order_zero.log_likelihood, order_two.log_likelihood)
    np.testing.assert_array_equal(order_zero.terms, order_one.terms)
    np.testing.assert_array_equal(order_zero.terms, order_two.terms)
    np.testing.assert_array_equal(order_zero.valid, order_two.valid)
    assert order_zero.score is None
    assert order_zero.hessian_packed is None
    assert order_one.score is not None
    assert order_one.hessian_packed is None
    assert order_two.score is not None
    assert order_two.hessian_packed is not None


def test_result_arrays_are_owned_read_only_and_row_aligned() -> None:
    result = _evaluate_case(TWEEDIE_LSS_CASES[3])
    arrays = (
        result.log_likelihood,
        result.score,
        result.hessian_packed,
        result.terms,
        result.valid,
    )

    for values in arrays:
        assert values is not None
        assert not values.flags.writeable
        assert values.shape[0] == 1
        with pytest.raises(ValueError):
            values.flat[0] = 0
    assert result.log_likelihood.dtype == np.float64
    assert result.score is not None and result.score.dtype == np.float64
    assert result.hessian_packed is not None and result.hessian_packed.dtype == np.float64
    assert np.issubdtype(result.terms.dtype, np.integer)
    assert result.valid.dtype == np.bool_


def _prior_batch_inputs() -> tuple[np.ndarray, ...]:
    cases = (
        TWEEDIE_LSS_CASES[0],
        TWEEDIE_LSS_CASES[2],
        TWEEDIE_LSS_CASES[3],
        TWEEDIE_LSS_CASES[5],
        TWEEDIE_POWER_RANGE_ZERO_CASES[1],
    )
    return tuple(
        np.array([getattr(case, field) for case in cases], dtype=np.float64)
        for field in ("y", "mean", "dispersion", "power", "weight")
    )


def _frozen_evaluation_arrays(case: FrozenTweedieEvaluation) -> tuple[np.ndarray, ...]:
    return tuple(np.array(column, dtype=np.float64) for column in zip(*case.rows, strict=True))


def _assert_evaluation_matches_frozen(
    actual,
    expected: FrozenTweedieEvaluation,
    *,
    derivative_order: int,
) -> None:
    np.testing.assert_array_equal(actual.valid, np.ones(len(expected.rows), dtype=np.bool_))
    np.testing.assert_array_equal(actual.terms, expected.terms)
    epsilon = np.finfo(np.float64).eps
    row_work = np.maximum(1.0, np.asarray(expected.terms, dtype=np.float64) + 1.0)

    def assert_channels(
        actual: np.ndarray,
        frozen: np.ndarray,
        *,
        factor: float,
        name: str,
    ) -> None:
        work = row_work.reshape((-1,) + (1,) * (frozen.ndim - 1))
        scale = np.maximum(1.0, np.abs(frozen))
        envelope = factor * epsilon * work * scale
        assert np.all(np.abs(actual - frozen) <= envelope), (
            f"compiled {name} differs from its frozen characterization envelope"
        )

    assert_channels(
        actual.log_likelihood,
        np.asarray(expected.log_likelihood),
        factor=64.0,
        name="value",
    )
    if derivative_order == 0:
        assert actual.score is None
        assert actual.hessian_packed is None
    else:
        assert actual.score is not None and expected.score is not None
        assert_channels(actual.score, np.asarray(expected.score), factor=128.0, name="score")
    if derivative_order < 2:
        assert actual.hessian_packed is None
    else:
        assert actual.hessian_packed is not None and expected.hessian is not None
        assert_channels(
            actual.hessian_packed,
            np.asarray(expected.hessian),
            factor=2048.0,
            name="Hessian",
        )


def _assert_external_batch(
    result,
    cases: tuple[TweedieLSSOracleCase, ...],
    *,
    derivative_order: int,
) -> None:
    assert result.valid.tolist() == [True] * len(cases)
    for row, case in enumerate(cases):
        np.testing.assert_allclose(
            result.log_likelihood[row],
            case.value,
            rtol=0.0,
            atol=case.value_atol,
        )
        if derivative_order >= 1:
            assert result.score is not None
            np.testing.assert_allclose(
                result.score[row],
                case.score,
                rtol=case.score_rtol,
                atol=case.score_atol,
            )
        if derivative_order == 2:
            assert result.hessian_packed is not None
            np.testing.assert_allclose(
                result.hessian_packed[row],
                case.hessian,
                rtol=case.hessian_rtol,
                atol=case.hessian_atol,
            )
    assert (result.score is None) == (derivative_order == 0)
    assert (result.hessian_packed is None) == (derivative_order < 2)


@pytest.mark.parametrize("derivative_order", [0, 1, 2])
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_external_oracle_grid_survives_permutations_and_partitions(
    derivative_order: int,
    semantics: str,
) -> None:
    cases = tuple(
        case
        for case in (*TWEEDIE_LSS_CASES, *TWEEDIE_POWER_RANGE_ZERO_CASES)
        if case.semantics == semantics
    )
    arrays = tuple(
        np.array([getattr(case, field) for case in cases], dtype=np.float64)
        for field in ("y", "mean", "dispersion", "power", "weight")
    )

    result = evaluate_tweedie_rows(
        *arrays,
        semantics,  # type: ignore[arg-type]
        derivative_order=derivative_order,
    )
    _assert_external_batch(result, cases, derivative_order=derivative_order)

    permutation = np.arange(len(cases) - 1, -1, -1)
    permuted_arrays = tuple(values[permutation] for values in arrays)
    permuted = evaluate_tweedie_rows(
        *permuted_arrays,
        semantics,  # type: ignore[arg-type]
        derivative_order=derivative_order,
    )
    _assert_external_batch(
        permuted,
        tuple(cases[index] for index in permutation),
        derivative_order=derivative_order,
    )

    split = max(1, len(cases) // 2)
    for start, stop in ((0, split), (split, len(cases))):
        if start == stop:
            continue
        partition = tuple(values[start:stop] for values in arrays)
        partitioned = evaluate_tweedie_rows(
            *partition,
            semantics,  # type: ignore[arg-type]
            derivative_order=derivative_order,
        )
        _assert_external_batch(
            partitioned,
            cases[start:stop],
            derivative_order=derivative_order,
        )


@pytest.mark.parametrize("derivative_order", [0, 1, 2])
def test_adversarial_grid_matches_frozen_characterization(
    derivative_order: int,
) -> None:
    case = FROZEN_ADVERSARIAL_EVALUATION
    result = evaluate_tweedie_rows(
        *_frozen_evaluation_arrays(case),
        case.semantics,
        derivative_order=derivative_order,
        max_terms=case.max_terms,
        log_cutoff=case.log_cutoff,
    )

    _assert_evaluation_matches_frozen(
        result,
        case,
        derivative_order=derivative_order,
    )


def test_huge_max_terms_matches_frozen_evidence_without_new_native_signature() -> None:
    case = FROZEN_HUGE_CAP_EVALUATION
    arrays = _frozen_evaluation_arrays(case)
    evaluate_tweedie_rows(*arrays, "prior", derivative_order=2, max_terms=100_000)
    signatures = tuple(tweedie_kernel._evaluate_tweedie_batch_core.signatures)

    result = evaluate_tweedie_rows(
        *arrays,
        case.semantics,
        derivative_order=case.derivative_order,
        max_terms=case.max_terms,
        log_cutoff=case.log_cutoff,
    )

    _assert_evaluation_matches_frozen(result, case, derivative_order=2)
    assert tuple(tweedie_kernel._evaluate_tweedie_batch_core.signatures) == signatures

    refusal = next(case for case in FROZEN_TWEEDIE_REFUSALS if case.id.startswith("huge-cap"))
    refusing = tuple(np.array([value], dtype=np.float64) for value in refusal.row)
    with pytest.raises(TweedieNumericalRefusal) as caught:
        evaluate_tweedie_rows(
            *refusing,
            refusal.semantics,
            derivative_order=refusal.derivative_order,
            max_terms=refusal.max_terms,
        )
    assert str(caught.value) == refusal.message
    assert tuple(tweedie_kernel._evaluate_tweedie_batch_core.signatures) == signatures


def _assert_identical_point_results(actual, expected) -> None:
    np.testing.assert_array_equal(actual.log_likelihood, expected.log_likelihood)
    np.testing.assert_array_equal(actual.score, expected.score)
    np.testing.assert_array_equal(actual.hessian_packed, expected.hessian_packed)
    np.testing.assert_array_equal(actual.terms, expected.terms)
    np.testing.assert_array_equal(actual.valid, expected.valid)


def _assert_identical_raw_results(actual, expected) -> None:
    for actual_values, expected_values in zip(actual[:5], expected[:5], strict=True):
        np.testing.assert_array_equal(actual_values, expected_values)
    assert actual[5:] == expected[5:]


@pytest.mark.parametrize("derivative_order", [0, 1, 2])
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_parallel_core_is_exactly_row_equivalent_under_permutations_and_partitions(
    derivative_order: int,
    semantics: str,
) -> None:
    cases = tuple(
        case
        for case in (*TWEEDIE_LSS_CASES, *TWEEDIE_POWER_RANGE_ZERO_CASES)
        if case.semantics == semantics
    )
    arrays = tuple(
        np.array([getattr(case, field) for case in cases], dtype=np.float64)
        for field in ("y", "mean", "dispersion", "power", "weight")
    )
    weight_mode = 0 if semantics == "prior" else 1
    variants = [arrays, tuple(values[::-1].copy() for values in arrays)]
    split = max(1, len(cases) // 2)
    variants.extend(
        tuple(values[start:stop].copy() for values in arrays)
        for start, stop in ((0, split), (split, len(cases)))
        if start < stop
    )

    for variant in variants:
        serial = tweedie_kernel._evaluate_tweedie_batch_core(
            *variant,
            weight_mode,
            derivative_order,
            100_000,
            37.0,
        )
        parallel = tweedie_kernel._evaluate_tweedie_batch_parallel_core(
            *variant,
            weight_mode,
            derivative_order,
            100_000,
            37.0,
        )
        _assert_identical_raw_results(parallel, serial)


def test_read_only_and_strided_inputs_reuse_normalized_native_signature() -> None:
    arrays = _prior_batch_inputs()
    baseline = evaluate_tweedie_rows(*arrays, "prior", derivative_order=2)
    signatures = tuple(tweedie_kernel._evaluate_tweedie_batch_core.signatures)

    read_only = tuple(values.copy() for values in arrays)
    for values in read_only:
        values.setflags(write=False)
    read_only_result = evaluate_tweedie_rows(
        *read_only,
        "prior",
        derivative_order=2,
    )

    strided = []
    for values in arrays:
        backing = np.empty(2 * values.size, dtype=np.float64)
        backing[::2] = values
        strided.append(backing[::2])
    strided_result = evaluate_tweedie_rows(
        *strided,
        "prior",
        derivative_order=2,
    )

    _assert_identical_point_results(read_only_result, baseline)
    _assert_identical_point_results(strided_result, baseline)
    assert tuple(tweedie_kernel._evaluate_tweedie_batch_core.signatures) == signatures


def test_parallel_boundary_preserves_read_only_and_strided_inputs_exactly() -> None:
    rows = 5_000
    arrays = tuple(np.resize(values, rows) for values in _prior_batch_inputs())
    serial = tweedie_kernel._evaluate_tweedie_batch_core(
        *arrays,
        0,
        2,
        100_000,
        37.0,
    )
    assert serial[5:] == (0, -1)

    read_only = tuple(values.copy() for values in arrays)
    for values in read_only:
        values.setflags(write=False)
    read_only_result = evaluate_tweedie_rows(
        *read_only,
        "prior",
        derivative_order=2,
    )

    strided = []
    for values in arrays:
        backing = np.empty(2 * values.size, dtype=np.float64)
        backing[::2] = values
        strided.append(backing[::2])
    strided_result = evaluate_tweedie_rows(
        *strided,
        "prior",
        derivative_order=2,
    )

    for result in (read_only_result, strided_result):
        np.testing.assert_array_equal(result.log_likelihood, serial[0])
        np.testing.assert_array_equal(result.score, serial[1])
        np.testing.assert_array_equal(result.hessian_packed, serial[2])
        np.testing.assert_array_equal(result.terms, serial[3])
        np.testing.assert_array_equal(result.valid, serial[4])
        assert not result.log_likelihood.flags.writeable
        assert result.score is not None and not result.score.flags.writeable
        assert result.hessian_packed is not None and not result.hessian_packed.flags.writeable
        assert not result.terms.flags.writeable
        assert not result.valid.flags.writeable


def _assert_raw_derivative_suppression(raw, derivative_order: int) -> None:
    assert raw[5] == 0
    if derivative_order == 0:
        assert raw[1].shape == (0, 3)
    else:
        assert raw[1].shape[1:] == (3,)
        assert np.all(np.isfinite(raw[1]))
    if derivative_order < 2:
        assert raw[2].shape == (0, 6), "raw core allocated suppressed order-2 rows"
    else:
        assert raw[2].shape[1:] == (6,)
        assert np.all(np.isfinite(raw[2]))


def test_raw_core_leaves_unrequested_derivative_channels_uncomputed() -> None:
    arrays = tuple(np.ascontiguousarray(values) for values in _prior_batch_inputs())

    order_zero = tweedie_kernel._evaluate_tweedie_batch_core(
        *arrays,
        0,
        0,
        100_000,
        37.0,
    )
    order_one = tweedie_kernel._evaluate_tweedie_batch_core(
        *arrays,
        0,
        1,
        100_000,
        37.0,
    )
    order_two = tweedie_kernel._evaluate_tweedie_batch_core(
        *arrays,
        0,
        2,
        100_000,
        37.0,
    )

    _assert_raw_derivative_suppression(order_zero, 0)
    _assert_raw_derivative_suppression(order_one, 1)
    _assert_raw_derivative_suppression(order_two, 2)


def test_compiled_order_zero_series_skips_all_special_function_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled_module = tweedie_kernel._compiled
    original = compiled_module._term_derivative_channels

    @njit
    def forbidden_channels(j, zeta_p, zeta_pp, inverse_r, derivative_order):
        if j > 0:
            raise RuntimeError("order-zero special-function poison")
        return zeta_p, zeta_pp, inverse_r, float(derivative_order)

    monkeypatch.setattr(compiled_module, "_term_derivative_channels", forbidden_channels)
    compiled_module._series_summary.recompile()
    coefficients = np.empty(10, dtype=np.float64)
    try:
        summary = compiled_module._series_summary(
            math.log(2.0),
            math.nan,
            math.nan,
            2.0,
            1.0,
            0,
            100_000,
            37.0,
            coefficients,
        )
        assert summary[0] == 0
        with pytest.raises(RuntimeError, match="order-zero special-function poison"):
            compiled_module._series_summary(
                math.log(2.0),
                1.0,
                math.nan,
                2.0,
                1.0,
                1,
                100_000,
                37.0,
                coefficients,
            )
    finally:
        compiled_module._term_derivative_channels = original
        compiled_module._series_summary.recompile()


def _recompile_compiled_positive_path(compiled_module) -> None:
    compiled_module._term_derivative_channels.recompile()
    compiled_module._series_summary.recompile()
    compiled_module._positive_row.recompile()
    compiled_module._evaluate_tweedie_batch_row.recompile()
    compiled_module._evaluate_tweedie_batch_core.recompile()


def test_production_order_one_batch_skips_poisoned_trigamma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled_module = tweedie_kernel._compiled
    original = compiled_module._digamma_trigamma_positive

    @njit
    def forbidden_trigamma(value):
        if value > 0.0:
            raise RuntimeError("order-one trigamma poison")
        return 0.0, 0.0

    monkeypatch.setattr(compiled_module, "_digamma_trigamma_positive", forbidden_trigamma)
    _recompile_compiled_positive_path(compiled_module)
    try:
        arrays = _case_arrays(TWEEDIE_LSS_CASES[3])
        order_one = evaluate_tweedie_rows(*arrays, "prior", derivative_order=1)
        assert order_one.score is not None
        assert order_one.hessian_packed is None
        with pytest.raises(RuntimeError, match="order-one trigamma poison"):
            evaluate_tweedie_rows(*arrays, "prior", derivative_order=2)
    finally:
        compiled_module._digamma_trigamma_positive = original
        _recompile_compiled_positive_path(compiled_module)


def test_compiled_special_functions_are_local_cache_dependencies() -> None:
    compiled_module = tweedie_kernel._compiled

    assert compiled_module._digamma_positive.__module__ == compiled_module.__name__
    assert compiled_module._digamma_trigamma_positive.__module__ == compiled_module.__name__


@pytest.mark.parametrize("value", [0.2, 1.7, 11.5, 12.0, 200.0])
def test_joint_digamma_trigamma_matches_existing_channels(value: float) -> None:
    compiled_module = tweedie_kernel._compiled

    digamma, trigamma = compiled_module._digamma_trigamma_positive(value)

    assert digamma == compiled_module._digamma_positive(value)
    assert trigamma == profile_kernel._trigamma_positive(value)


def test_external_values_and_weight_identity_reject_required_compiled_mutations() -> None:
    case = TWEEDIE_LSS_CASES[3]
    arrays = _case_arrays(case)
    compiled = _evaluate_case(case)
    assert compiled.score is not None
    assert compiled.hessian_packed is not None

    swapped = compiled.hessian_packed.copy()
    swapped[:, [1, 2]] = swapped[:, [2, 1]]
    with pytest.raises(AssertionError):
        _assert_external_batch(
            replace(compiled, hessian_packed=swapped),
            (case,),
            derivative_order=2,
        )

    omitted_phi_chain = compiled.hessian_packed.copy()
    omitted_phi_chain[:, 3] += compiled.score[:, 1] / arrays[2]
    with pytest.raises(AssertionError):
        _assert_external_batch(
            replace(compiled, hessian_packed=omitted_phi_chain),
            (case,),
            derivative_order=2,
        )

    y, mean, modeled_phi, power, weight = (float(values[0]) for values in arrays)
    rho = math.log(modeled_phi) - math.log(weight)
    log_mean = math.log(mean)
    r = power - 1.0
    s = 2.0 - power
    inverse_r = 1.0 / r
    inverse_s = 1.0 / s
    a_term = y * math.exp(-r * log_mean)
    c_term = math.exp(s * log_mean)
    f_term = math.exp(-rho)
    b_value = f_term * math.fsum((-a_term * inverse_r, -c_term * inverse_s))
    inverse_r2 = inverse_r * inverse_r
    b_p = f_term * math.fsum(
        (
            a_term * (log_mean * inverse_r + inverse_r2),
            c_term * (log_mean * inverse_s - inverse_s * inverse_s),
        )
    )
    mean_q_rho = modeled_phi * compiled.score[0, 1] + b_value
    mean_q_p = compiled.score[0, 2] - b_p
    raw_moments = compiled.hessian_packed.copy()
    raw_moments[0, 3] += mean_q_rho * mean_q_rho / (modeled_phi * modeled_phi)
    raw_moments[0, 4] += mean_q_rho * mean_q_p / modeled_phi
    raw_moments[0, 5] += mean_q_p * mean_q_p
    with pytest.raises(AssertionError):
        _assert_external_batch(
            replace(compiled, hessian_packed=raw_moments),
            (case,),
            derivative_order=2,
        )

    frequency_arrays = (
        np.array([1.1, 1.1], dtype=np.float64),
        np.array([0.9, 0.9], dtype=np.float64),
        np.array([0.3, 0.3], dtype=np.float64),
        np.array([1.49, 1.49], dtype=np.float64),
        np.array([1.0, 7.0], dtype=np.float64),
    )
    pre_law_count = evaluate_tweedie_rows(
        *frequency_arrays,
        "prior",
        derivative_order=2,
    )
    assert pre_law_count.score is not None
    assert pre_law_count.hessian_packed is not None

    def assert_frequency_replication(result) -> None:
        np.testing.assert_allclose(result.log_likelihood[1], 7.0 * result.log_likelihood[0])
        np.testing.assert_allclose(result.score[1], 7.0 * result.score[0])
        np.testing.assert_allclose(result.hessian_packed[1], 7.0 * result.hessian_packed[0])

    with pytest.raises(AssertionError):
        assert_frequency_replication(pre_law_count)

    frequency = evaluate_tweedie_rows(
        *frequency_arrays,
        "frequency",
        derivative_order=2,
    )
    assert_frequency_replication(frequency)


def test_batch_singleton_partition_and_permutation_are_bit_invariant() -> None:
    arrays = _prior_batch_inputs()
    batch = evaluate_tweedie_rows(*arrays, "prior", derivative_order=2)
    singletons = [
        evaluate_tweedie_rows(
            *(values[index : index + 1] for values in arrays),
            "prior",
            derivative_order=2,
        )
        for index in range(len(arrays[0]))
    ]

    np.testing.assert_array_equal(
        batch.log_likelihood, np.array([result.log_likelihood[0] for result in singletons])
    )
    assert batch.score is not None
    assert batch.hessian_packed is not None
    np.testing.assert_array_equal(batch.score, np.vstack([result.score for result in singletons]))
    np.testing.assert_array_equal(
        batch.hessian_packed,
        np.vstack([result.hessian_packed for result in singletons]),
    )
    np.testing.assert_array_equal(batch.terms, [result.terms[0] for result in singletons])

    permutation = np.array([4, 1, 3, 0, 2])
    permuted = evaluate_tweedie_rows(
        *(values[permutation] for values in arrays),
        "prior",
        derivative_order=2,
    )
    np.testing.assert_array_equal(permuted.log_likelihood, batch.log_likelihood[permutation])
    assert permuted.score is not None
    assert permuted.hessian_packed is not None
    np.testing.assert_array_equal(permuted.score, batch.score[permutation])
    np.testing.assert_array_equal(permuted.hessian_packed, batch.hessian_packed[permutation])
    np.testing.assert_array_equal(permuted.terms, batch.terms[permutation])


def test_max_terms_is_a_per_row_budget_not_a_batch_global_budget() -> None:
    arrays = _prior_batch_inputs()
    singletons = [
        evaluate_tweedie_rows(
            *(values[index : index + 1] for values in arrays),
            "prior",
            derivative_order=0,
        )
        for index in range(len(arrays[0]))
    ]
    per_row_cap = max(int(result.terms[0]) for result in singletons)

    batch = evaluate_tweedie_rows(
        *arrays,
        "prior",
        derivative_order=0,
        max_terms=per_row_cap,
    )

    np.testing.assert_array_equal(batch.terms, [result.terms[0] for result in singletons])


def test_frequency_count_multiplies_complete_row_without_changing_window() -> None:
    common = (
        np.array([1.1, 1.1], dtype=np.float64),
        np.array([0.9, 0.9], dtype=np.float64),
        np.array([0.3, 0.3], dtype=np.float64),
        np.array([1.49, 1.49], dtype=np.float64),
    )
    result = evaluate_tweedie_rows(
        *common,
        np.array([1.0, 7.0], dtype=np.float64),
        "frequency",
        derivative_order=2,
    )

    assert result.terms[0] == result.terms[1]
    np.testing.assert_allclose(result.log_likelihood[1], 7.0 * result.log_likelihood[0], rtol=0)
    assert result.score is not None
    assert result.hessian_packed is not None
    np.testing.assert_allclose(result.score[1], 7.0 * result.score[0], rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        result.hessian_packed[1], 7.0 * result.hessian_packed[0], rtol=0, atol=2e-12
    )


def test_prior_weight_changes_the_window_but_mean_never_does() -> None:
    result = evaluate_tweedie_rows(
        np.array([1.1, 1.1, 1.1, 1.1], dtype=np.float64),
        np.array([0.2, 0.9, 10.0, 0.9], dtype=np.float64),
        np.full(4, 0.3, dtype=np.float64),
        np.full(4, 1.49, dtype=np.float64),
        np.array([1.0, 1.0, 1.0, 4.0], dtype=np.float64),
        "prior",
        derivative_order=0,
    )

    assert result.terms[0] == result.terms[1] == result.terms[2]
    assert result.terms[3] != result.terms[1]


def test_prior_effective_dispersion_keeps_modeled_phi_natural_coordinates() -> None:
    case = TWEEDIE_LSS_CASES[5]
    prior = _evaluate_case(case)
    unit = _evaluate_case(replace(case, dispersion=case.dispersion / case.weight, weight=1.0))
    weight = case.weight

    np.testing.assert_array_equal(prior.log_likelihood, unit.log_likelihood)
    np.testing.assert_array_equal(prior.terms, unit.terms)
    assert prior.score is not None and unit.score is not None
    np.testing.assert_allclose(
        prior.score[0],
        [unit.score[0, 0], unit.score[0, 1] / weight, unit.score[0, 2]],
        rtol=2e-14,
        atol=2e-14,
    )
    assert prior.hessian_packed is not None and unit.hessian_packed is not None
    np.testing.assert_allclose(
        prior.hessian_packed[0],
        [
            unit.hessian_packed[0, 0],
            unit.hessian_packed[0, 1] / weight,
            unit.hessian_packed[0, 2],
            unit.hessian_packed[0, 3] / weight**2,
            unit.hessian_packed[0, 4] / weight,
            unit.hessian_packed[0, 5],
        ],
        rtol=2e-13,
        atol=2e-13,
    )


def test_ratio_bracket_finds_exact_plateau_and_adversarial_distant_mode() -> None:
    coefficients = np.empty(10, dtype=np.float64)
    # alpha=1 and zeta=log(2) make q_1 == q_2 exactly in the ratio formula.
    tweedie_kernel._compiled._fill_log_gamma_increment_coefficients(1.0, coefficients)
    assert tweedie_kernel._compiled._locate_series_mode(math.log(2.0), 1.0, coefficients) == (
        0,
        1,
    )

    # This literal defeats the old asymptotic estimate plus two adjacent q
    # checks.  The ratio transition itself uniquely brackets the mode.
    alpha = 0.1
    zeta = 16.75
    tweedie_kernel._compiled._fill_log_gamma_increment_coefficients(alpha, coefficients)
    status, mode = tweedie_kernel._compiled._locate_series_mode(zeta, alpha, coefficients)
    assert status == 0
    assert mode == 5_058_592
    before = tweedie_kernel._compiled._log_adjacent_ratio(
        mode - 1,
        zeta,
        alpha,
        coefficients,
    )
    after = tweedie_kernel._compiled._log_adjacent_ratio(
        mode,
        zeta,
        alpha,
        coefficients,
    )
    assert before > 0.0
    assert after <= 0.0


def test_peak_centered_moments_match_high_mode_direct_recurrence() -> None:
    case = CENTERED_RHO_MOMENT_ORACLE
    result = evaluate_tweedie_rows(
        np.array([case.y], dtype=np.float64),
        np.array([case.mean], dtype=np.float64),
        np.array([case.dispersion], dtype=np.float64),
        np.array([case.power], dtype=np.float64),
        np.array([case.weight], dtype=np.float64),
        "prior",
        derivative_order=2,
    )
    assert result.score is not None
    assert result.hessian_packed is not None
    score_rho = case.dispersion * result.score[0, 1]
    hessian_rho_rho = case.dispersion**2 * result.hessian_packed[0, 3] + score_rho

    assert score_rho == pytest.approx(
        case.expected_score_rho,
        rel=0.0,
        abs=case.score_rho_atol,
    )
    assert hessian_rho_rho == pytest.approx(
        case.expected_hessian_rho_rho,
        rel=0.0,
        abs=case.hessian_rho_rho_atol,
    )
    assert 10_000 < result.terms[0] < 100_000

    rho = math.log(case.dispersion / case.weight)
    d_value = math.log(case.power - 1.0) - math.log(case.y) + rho
    coefficients = np.empty(10, dtype=np.float64)
    series = tweedie_kernel._compiled._series_summary(
        math.log(1.0e12),
        4.0 * d_value,
        -16.0 * d_value + 24.0,
        2.0,
        1.0,
        2,
        100_000,
        37.0,
        coefficients,
    )
    assert series[0] == 0
    assert series[7] == pytest.approx(
        case.expected_covariance_q_rho_p,
        rel=0.0,
        abs=case.covariance_q_rho_p_atol,
    )
    assert series[8] == pytest.approx(
        case.expected_variance_q_p,
        rel=0.0,
        abs=case.variance_q_p_atol,
    )
    assert series[9] == result.terms[0] == 12_167


def test_positive_upper_power_boundary_is_order_batch_and_permutation_invariant() -> None:
    arrays = (
        np.array([1.7, 0.8, 2.4], dtype=np.float64),
        np.array([1.4, 0.9, 2.0], dtype=np.float64),
        np.array([3.0, 1.2, 4.0], dtype=np.float64),
        np.full(3, 1.95, dtype=np.float64),
        np.array([1.0, 0.7, 2.0], dtype=np.float64),
    )
    orders = [evaluate_tweedie_rows(*arrays, "prior", derivative_order=order) for order in range(3)]
    np.testing.assert_array_equal(orders[0].log_likelihood, orders[1].log_likelihood)
    np.testing.assert_array_equal(orders[0].log_likelihood, orders[2].log_likelihood)
    np.testing.assert_array_equal(orders[0].terms, orders[1].terms)
    np.testing.assert_array_equal(orders[0].terms, orders[2].terms)

    batch = orders[2]
    singletons = [
        evaluate_tweedie_rows(
            *(values[index : index + 1] for values in arrays),
            "prior",
            derivative_order=2,
        )
        for index in range(3)
    ]
    np.testing.assert_array_equal(
        batch.log_likelihood,
        np.array([result.log_likelihood[0] for result in singletons]),
    )
    assert batch.score is not None
    assert batch.hessian_packed is not None
    np.testing.assert_array_equal(batch.score, np.vstack([result.score for result in singletons]))
    np.testing.assert_array_equal(
        batch.hessian_packed,
        np.vstack([result.hessian_packed for result in singletons]),
    )
    permutation = np.array([2, 0, 1])
    permuted = evaluate_tweedie_rows(
        *(values[permutation] for values in arrays),
        "prior",
        derivative_order=2,
    )
    np.testing.assert_array_equal(permuted.log_likelihood, batch.log_likelihood[permutation])
    assert permuted.score is not None
    assert permuted.hessian_packed is not None
    np.testing.assert_array_equal(permuted.score, batch.score[permutation])
    np.testing.assert_array_equal(permuted.hessian_packed, batch.hessian_packed[permutation])
    np.testing.assert_array_equal(permuted.terms, batch.terms[permutation])


def _point_value(*, y: float, mean: float, dispersion: float, power: float) -> float:
    result = evaluate_tweedie_rows(
        np.array([y], dtype=np.float64),
        np.array([mean], dtype=np.float64),
        np.array([dispersion], dtype=np.float64),
        np.array([power], dtype=np.float64),
        np.ones(1, dtype=np.float64),
        "prior",
        derivative_order=0,
    )
    return float(result.log_likelihood[0])


def _roundoff_bound(
    coefficients: tuple[float, ...],
    values: tuple[float, ...],
    denominator: float,
    *,
    operation_factor: float = 8.0,
) -> float:
    """Bound binary64 stencil arithmetic by coefficient one-norm and scale."""
    epsilon = np.finfo(np.float64).eps
    scale = max(1.0, *(abs(value) for value in values))
    return (
        operation_factor
        * epsilon
        * math.fsum(abs(coefficient) for coefficient in coefficients)
        * scale
        / abs(denominator)
    )


def _backward_power_stencils(
    *,
    y: float,
    mean: float,
    dispersion: float,
    power: float,
    power_step: float,
    dispersion_step: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    first_coefficients = (25.0, -48.0, 36.0, -16.0, 3.0)
    second_coefficients = (45.0, -154.0, 214.0, -156.0, 61.0, -10.0)
    values = tuple(
        _point_value(
            y=y,
            mean=mean,
            dispersion=dispersion,
            power=power - index * power_step,
        )
        for index in range(6)
    )
    score_p = math.fsum(
        coefficient * value
        for coefficient, value in zip(first_coefficients, values[:5], strict=True)
    ) / (12.0 * power_step)
    hessian_pp = math.fsum(
        coefficient * value for coefficient, value in zip(second_coefficients, values, strict=True)
    ) / (12.0 * power_step * power_step)

    cross_values: list[float] = []
    score_phi_values: list[float] = []
    for index in range(5):
        row_power = power - index * power_step
        plus = _point_value(
            y=y,
            mean=mean,
            dispersion=dispersion + dispersion_step,
            power=row_power,
        )
        minus = _point_value(
            y=y,
            mean=mean,
            dispersion=dispersion - dispersion_step,
            power=row_power,
        )
        cross_values.extend((plus, minus))
        score_phi_values.append((plus - minus) / (2.0 * dispersion_step))
    hessian_phi_p = math.fsum(
        coefficient * value
        for coefficient, value in zip(
            first_coefficients,
            score_phi_values,
            strict=True,
        )
    ) / (12.0 * power_step)

    bounds = (
        _roundoff_bound(first_coefficients, values[:5], 12.0 * power_step),
        _roundoff_bound(
            tuple(outer * inner for outer in first_coefficients for inner in (1.0, -1.0)),
            tuple(cross_values),
            24.0 * power_step * dispersion_step,
        ),
        _roundoff_bound(
            second_coefficients,
            values,
            12.0 * power_step * power_step,
        ),
    )
    return (score_p, hessian_phi_p, hessian_pp), bounds


def test_positive_upper_power_boundary_has_internal_derivative_and_work_evidence() -> None:
    # The scalar value has an independent mgcv anchor above. Derivatives retain
    # analytic mean identities and centered differences of the complete
    # normalized row value in the coordinates that remain interior.
    y = 1.7
    mean = 1.4
    dispersion = 3.0
    power = 1.95
    result = evaluate_tweedie_rows(
        np.array([y], dtype=np.float64),
        np.array([mean], dtype=np.float64),
        np.array([dispersion], dtype=np.float64),
        np.array([power], dtype=np.float64),
        np.ones(1, dtype=np.float64),
        "prior",
        derivative_order=2,
    )
    assert result.score is not None
    assert result.hessian_packed is not None
    assert result.valid.tolist() == [True]
    assert np.all(np.isfinite(result.log_likelihood))
    assert np.all(np.isfinite(result.score))
    assert np.all(np.isfinite(result.hessian_packed))
    assert 0 < result.terms[0] <= 100_000

    expected_mean_score = (y - mean) / (dispersion * mean**power)
    expected_mean_hessian = ((power - 1.0) * mean - power * y) / (
        dispersion * mean ** (power + 1.0)
    )
    epsilon = np.finfo(np.float64).eps
    mean_score_atol = 32.0 * epsilon * max(1.0, abs(expected_mean_score))
    mean_hessian_atol = 32.0 * epsilon * max(1.0, abs(expected_mean_hessian))
    np.testing.assert_allclose(
        result.score[0, 0],
        expected_mean_score,
        rtol=0.0,
        atol=mean_score_atol,
    )
    np.testing.assert_allclose(
        result.hessian_packed[0, 0],
        expected_mean_hessian,
        rtol=0.0,
        atol=mean_hessian_atol,
    )
    np.testing.assert_allclose(
        result.hessian_packed[0, 1],
        -expected_mean_score / dispersion,
        rtol=0.0,
        atol=32.0 * epsilon * max(1.0, abs(expected_mean_score / dispersion)),
    )
    np.testing.assert_allclose(
        result.hessian_packed[0, 2],
        -math.log(mean) * expected_mean_score,
        rtol=0.0,
        atol=48.0 * epsilon * max(1.0, abs(math.log(mean) * expected_mean_score)),
    )

    fine, fine_roundoff = _backward_power_stencils(
        y=y,
        mean=mean,
        dispersion=dispersion,
        power=power,
        power_step=2.0e-3,
        dispersion_step=4.0e-4 * dispersion,
    )
    coarse, coarse_roundoff = _backward_power_stencils(
        y=y,
        mean=mean,
        dispersion=dispersion,
        power=power,
        power_step=4.0e-3,
        dispersion_step=2.0e-4 * dispersion,
    )
    refined_phi, refined_phi_roundoff = _backward_power_stencils(
        y=y,
        mean=mean,
        dispersion=dispersion,
        power=power,
        power_step=2.0e-3,
        dispersion_step=2.0e-4 * dispersion,
    )
    # The p stencils are fourth order, so their fine-grid error proxy is the
    # nested difference divided by 2**4 - 1.  The central phi pair is second
    # order; the refined phi-p estimate uses its difference divided by
    # 2**2 - 1.  Coefficient one-norm terms bound binary64 stencil arithmetic.
    stencil_atol = (
        abs(fine[0] - coarse[0]) / 15.0 + fine_roundoff[0] + coarse_roundoff[0],
        abs(refined_phi[1] - coarse[1]) / 15.0
        + abs(fine[1] - refined_phi[1]) / 3.0
        + fine_roundoff[1]
        + refined_phi_roundoff[1]
        + coarse_roundoff[1],
        abs(fine[2] - coarse[2]) / 15.0 + fine_roundoff[2] + coarse_roundoff[2],
    )
    np.testing.assert_allclose(
        result.score[0, 2],
        fine[0],
        rtol=0.0,
        atol=stencil_atol[0],
    )
    np.testing.assert_allclose(
        result.hessian_packed[0, 4],
        refined_phi[1],
        rtol=0.0,
        atol=stencil_atol[1],
    )
    np.testing.assert_allclose(
        result.hessian_packed[0, 5],
        fine[2],
        rtol=0.0,
        atol=stencil_atol[2],
    )

    mean_step = 4.0e-4 * mean
    phi_step = 4.0e-4 * dispersion
    center = float(result.log_likelihood[0])
    mean_plus = _point_value(y=y, mean=mean + mean_step, dispersion=dispersion, power=power)
    mean_minus = _point_value(y=y, mean=mean - mean_step, dispersion=dispersion, power=power)
    phi_plus = _point_value(y=y, mean=mean, dispersion=dispersion + phi_step, power=power)
    phi_minus = _point_value(y=y, mean=mean, dispersion=dispersion - phi_step, power=power)
    mixed = (
        _point_value(
            y=y,
            mean=mean + mean_step,
            dispersion=dispersion + phi_step,
            power=power,
        )
        - _point_value(
            y=y,
            mean=mean + mean_step,
            dispersion=dispersion - phi_step,
            power=power,
        )
        - _point_value(
            y=y,
            mean=mean - mean_step,
            dispersion=dispersion + phi_step,
            power=power,
        )
        + _point_value(
            y=y,
            mean=mean - mean_step,
            dispersion=dispersion - phi_step,
            power=power,
        )
    ) / (4.0 * mean_step * phi_step)
    finite_score = np.array(
        [
            (mean_plus - mean_minus) / (2.0 * mean_step),
            (phi_plus - phi_minus) / (2.0 * phi_step),
        ]
    )
    finite_hessian = np.array(
        [
            (mean_plus - 2.0 * center + mean_minus) / mean_step**2,
            mixed,
            (phi_plus - 2.0 * center + phi_minus) / phi_step**2,
        ]
    )
    np.testing.assert_allclose(result.score[0, :2], finite_score, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(
        result.hessian_packed[0, [0, 1, 3]],
        finite_hessian,
        rtol=3e-4,
        atol=3e-6,
    )


def test_positive_row_refuses_when_its_own_window_exceeds_work_cap() -> None:
    case = TWEEDIE_LSS_CASES[3]

    with pytest.raises(TweedieNumericalRefusal, match="max_terms"):
        _evaluate_case(case, derivative_order=0, max_terms=2)


def _assert_cutoff_channel_matches_frozen(
    actual: np.ndarray,
    frozen: tuple[float, ...],
    *,
    terms: int,
    factor: float,
) -> None:
    epsilon = np.finfo(np.float64).eps
    frozen_values = np.asarray(frozen)
    scale = np.maximum(1.0, np.abs(frozen_values))
    envelope = factor * epsilon * float(terms + 1) * scale
    assert np.all(np.abs(actual - frozen_values) <= envelope)


def _two_path_p_p_summation_gamma(*, terms: int) -> float:
    epsilon = np.finfo(np.float64).eps
    # The longest summation path has one update per selected term, followed by
    # anchor recombination and the final three-constituent composition.
    rounding_depth = terms + 2
    gamma = rounding_depth * epsilon / (1.0 - rounding_depth * epsilon)
    # Both the deleted mirror and compiled result contribute binary64 error.
    return 2.0 * gamma


def _p_p_characterization_envelope(case: FrozenTweedieCutoffCase, *, terms: int) -> float:
    constituent_l1 = math.fsum(abs(value) for value in case.p_p_constituents)
    return _two_path_p_p_summation_gamma(terms=terms) * constituent_l1


def test_frozen_cutoff_ledger_records_p_p_cancellation_constituents() -> None:
    for case in FROZEN_CUSTOM_CUTOFF_CASES:
        assert math.fsum(case.p_p_constituents) == case.hessian[5]
        assert case.channels_source == "frozen-python-mirror/v1"
        assert case.terms_source == "frozen-numba-evaluation/v1"


def test_lower_cutoff_p_p_output_scale_is_a_failing_mutation_witness() -> None:
    lower = FROZEN_CUSTOM_CUTOFF_CASES[0]
    arrays = tuple(np.array([value], dtype=np.float64) for value in lower.row)
    result = evaluate_tweedie_rows(
        *arrays,
        lower.semantics,
        derivative_order=2,
        log_cutoff=lower.cutoffs[1],
    )
    assert result.hessian_packed is not None
    error = abs(result.hessian_packed[0, 5] - lower.hessian[5])
    terms = int(result.terms[0])
    two_path_gamma = _two_path_p_p_summation_gamma(terms=terms)
    unjustified_output_scale = two_path_gamma * max(1.0, abs(lower.hessian[5]))
    cancellation_scale = _p_p_characterization_envelope(lower, terms=terms)

    assert error > unjustified_output_scale
    assert error <= cancellation_scale


def test_frozen_cutoff_ledger_covers_upper_power_and_frequency_regimes() -> None:
    by_id = {case.id: case for case in FROZEN_CUSTOM_CUTOFF_CASES}
    required_ids = {
        "mid-right-adjacent-cutoff",
        "three-quarter-frequency-adjacent-cutoff",
    }
    assert required_ids <= by_id.keys()

    upper = by_id["mid-right-adjacent-cutoff"]
    assert upper.row[3] > 1.5
    assert 0.0 < (2.0 - upper.row[3]) / (upper.row[3] - 1.0) < 1.0
    assert upper.semantics == "prior"
    assert upper.cutoffs == (
        37.883368320677135,
        37.88336832067714,
        37.88336832067715,
    )
    assert upper.terms == (32, 33, 33)
    assert np.nextafter(upper.cutoffs[0], math.inf) == upper.cutoffs[1]
    assert np.nextafter(upper.cutoffs[1], math.inf) == upper.cutoffs[2]

    frequency = by_id["three-quarter-frequency-adjacent-cutoff"]
    assert frequency.row[3] > 1.5
    assert frequency.row[4] == 3.0
    assert frequency.semantics == "frequency"
    assert frequency.cutoffs == (
        36.31621839739933,
        36.31621839739934,
        36.316218397399346,
    )
    assert frequency.terms == (29, 30, 30)
    assert np.nextafter(frequency.cutoffs[0], math.inf) == frequency.cutoffs[1]
    assert np.nextafter(frequency.cutoffs[1], math.inf) == frequency.cutoffs[2]


@pytest.mark.parametrize("case", FROZEN_CUSTOM_CUTOFF_CASES, ids=lambda case: case.id)
def test_selected_custom_cutoff_endpoints_match_frozen_windows(case) -> None:
    arrays = tuple(np.array([value], dtype=np.float64) for value in case.row)

    for cutoff, expected_terms in zip(case.cutoffs, case.terms, strict=True):
        result = evaluate_tweedie_rows(
            *arrays,
            case.semantics,
            derivative_order=2,
            log_cutoff=cutoff,
        )

        assert result.terms.tolist() == [expected_terms]
        _assert_cutoff_channel_matches_frozen(
            result.log_likelihood,
            (case.log_likelihood,),
            terms=expected_terms,
            factor=64.0,
        )
        assert result.score is not None
        _assert_cutoff_channel_matches_frozen(
            result.score[0],
            case.score,
            terms=expected_terms,
            factor=128.0,
        )
        assert result.hessian_packed is not None
        _assert_cutoff_channel_matches_frozen(
            result.hessian_packed[0, :5],
            case.hessian[:5],
            terms=expected_terms,
            factor=128.0,
        )
        assert abs(result.hessian_packed[0, 5] - case.hessian[5]) <= (
            _p_p_characterization_envelope(case, terms=expected_terms)
        )


@pytest.mark.parametrize("case", FROZEN_CUTOFF_CAP_BOUNDARIES, ids=lambda case: case.id)
def test_endpoint_tuned_custom_cap_matches_frozen_boundary(case) -> None:
    arrays = tuple(np.array([value], dtype=np.float64) for value in case.row)
    raw = tweedie_kernel._evaluate_tweedie_batch_core(
        *arrays,
        0 if case.semantics == "prior" else 1,
        0,
        case.max_terms,
        case.log_cutoff,
    )
    result = evaluate_tweedie_rows(
        *arrays,
        case.semantics,
        derivative_order=0,
        max_terms=case.max_terms,
        log_cutoff=case.log_cutoff,
    )

    assert raw[5:] == (case.status, case.failing_row)
    assert result.terms.tolist() == [case.terms]
    np.testing.assert_allclose(
        result.log_likelihood,
        [case.log_likelihood],
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps * float(case.terms + 1),
    )


def test_series_mode_above_exact_float_integer_range_refuses() -> None:
    with pytest.raises(TweedieNumericalRefusal, match="mode"):
        evaluate_tweedie_rows(
            np.ones(1, dtype=np.float64),
            np.ones(1, dtype=np.float64),
            np.array([2.0e-16], dtype=np.float64),
            np.array([1.5], dtype=np.float64),
            np.ones(1, dtype=np.float64),
            "prior",
            derivative_order=0,
        )


@pytest.mark.parametrize("case", FROZEN_TWEEDIE_REFUSALS, ids=lambda case: case.id)
def test_refusal_matches_frozen_raw_status_row_and_public_message(case) -> None:
    arrays = tuple(np.array([value], dtype=np.float64) for value in case.row)
    native_max_terms = min(case.max_terms, tweedie_kernel._compiled._MAX_SAFE_MODE + 1)
    serial = tweedie_kernel._evaluate_tweedie_batch_core(
        *arrays,
        0 if case.semantics == "prior" else 1,
        case.derivative_order,
        native_max_terms,
        case.log_cutoff,
    )
    assert serial[5:] == (case.status, case.failing_row)

    with pytest.raises(TweedieNumericalRefusal) as caught:
        evaluate_tweedie_rows(
            *arrays,
            case.semantics,
            derivative_order=case.derivative_order,
            max_terms=case.max_terms,
            log_cutoff=case.log_cutoff,
        )
    assert str(caught.value) == case.message


@pytest.mark.parametrize(
    "case",
    tuple(case for case in FROZEN_TWEEDIE_REFUSALS if case.id.startswith("complete-")),
    ids=lambda case: case.id,
)
def test_parallel_core_preserves_frozen_complete_output_statuses(case) -> None:
    arrays = tuple(np.array([value], dtype=np.float64) for value in case.row)
    parallel = tweedie_kernel._evaluate_tweedie_batch_parallel_core(
        *arrays,
        0 if case.semantics == "prior" else 1,
        case.derivative_order,
        case.max_terms,
        case.log_cutoff,
    )

    assert parallel[5:] == (case.status, case.failing_row)


def test_parallel_core_reports_the_deterministic_earliest_of_multiple_bad_rows() -> None:
    arrays = (
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        np.ones(4, dtype=np.float64),
        np.ones(4, dtype=np.float64),
        np.full(4, 1.5, dtype=np.float64),
        np.ones(4, dtype=np.float64),
    )
    expected = (4, 1)

    for _ in range(8):
        raw = tweedie_kernel._evaluate_tweedie_batch_parallel_core(
            *arrays,
            0,
            2,
            1,
            37.0,
        )
        assert raw[5:] == expected

    with pytest.raises(TweedieNumericalRefusal) as caught:
        evaluate_tweedie_rows(
            *arrays,
            "prior",
            derivative_order=2,
            max_terms=1,
        )
    assert str(caught.value) == "row 1: positive series window reached per-row max_terms=1"


def test_compiled_refusal_reports_the_frozen_failing_row() -> None:
    arrays = (
        np.array([1.1, 1.0e308], dtype=np.float64),
        np.array([0.9, 1.0e-308], dtype=np.float64),
        np.array([0.3, 1.0], dtype=np.float64),
        np.array([1.49, 1.5], dtype=np.float64),
        np.ones(2, dtype=np.float64),
    )
    raw = tweedie_kernel._evaluate_tweedie_batch_core(*arrays, 0, 2, 100_000, 37.0)
    assert raw[5:] == (17, 1)

    with pytest.raises(TweedieNumericalRefusal) as caught:
        evaluate_tweedie_rows(*arrays, "prior", derivative_order=2)

    assert str(caught.value) == "row 1: positive-row canonical scale is not representable"


def _valid_inputs() -> list[np.ndarray]:
    return [
        np.array([1.0], dtype=np.float64),
        np.array([1.2], dtype=np.float64),
        np.array([0.7], dtype=np.float64),
        np.array([1.5], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
    ]


@pytest.mark.parametrize("index", range(5))
def test_input_vectors_must_be_literal_float64_arrays(index: int) -> None:
    inputs = _valid_inputs()
    inputs[index] = inputs[index].astype(np.float32)

    with pytest.raises(TypeError, match="float64"):
        evaluate_tweedie_rows(*inputs, "prior", derivative_order=0)


def test_input_vectors_must_be_one_dimensional_and_row_aligned() -> None:
    inputs = _valid_inputs()
    inputs[0] = inputs[0].reshape(1, 1)
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate_tweedie_rows(*inputs, "prior", derivative_order=0)

    inputs = _valid_inputs()
    inputs[1] = np.ones(2, dtype=np.float64)
    with pytest.raises(ValueError, match="same shape"):
        evaluate_tweedie_rows(*inputs, "prior", derivative_order=0)


@pytest.mark.parametrize(
    ("index", "value", "message"),
    [
        (0, -1.0, "y must be finite with y >= 0"),
        (1, 0.0, "mean must be finite and strictly positive"),
        (2, 0.0, "dispersion must be finite and strictly positive"),
        (3, 1.0, "power must be finite and strictly between 1 and 2"),
        (3, 2.0, "power must be finite and strictly between 1 and 2"),
        (4, 0.0, "weight must be finite and strictly positive"),
        (0, math.nan, "y must be finite with y >= 0"),
    ],
)
def test_invalid_mathematical_inputs_are_value_errors_not_numerical_refusals(
    index: int, value: float, message: str
) -> None:
    inputs = _valid_inputs()
    inputs[index][0] = value

    with pytest.raises(ValueError, match=message) as caught:
        evaluate_tweedie_rows(*inputs, "prior", derivative_order=0)
    assert not isinstance(caught.value, TweedieNumericalRefusal)


def test_frequency_weights_must_be_exact_positive_integer_counts() -> None:
    inputs = _valid_inputs()
    inputs[4][0] = 1.5

    with pytest.raises(ValueError, match="integer replication counts"):
        evaluate_tweedie_rows(*inputs, "frequency", derivative_order=0)


@pytest.mark.parametrize("derivative_order", [-1, True, 3, 1.0])
def test_derivative_order_is_an_exact_supported_integer(derivative_order) -> None:
    with pytest.raises(ValueError, match="derivative_order"):
        evaluate_tweedie_rows(
            *_valid_inputs(),
            "prior",
            derivative_order=derivative_order,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_terms": 0}, "max_terms"),
        ({"max_terms": True}, "max_terms"),
        ({"log_cutoff": 0.0}, "log_cutoff"),
        ({"log_cutoff": math.inf}, "log_cutoff"),
    ],
)
def test_point_controls_are_strictly_validated(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_tweedie_rows(
            *_valid_inputs(),
            "prior",
            derivative_order=0,
            **kwargs,
        )


def test_unknown_weight_semantics_is_an_input_error() -> None:
    with pytest.raises(ValueError, match="semantics"):
        evaluate_tweedie_rows(
            *_valid_inputs(),
            "power",  # type: ignore[arg-type]
            derivative_order=0,
        )
