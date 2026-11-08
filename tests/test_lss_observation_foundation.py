"""Independent endpoint, positive-mixture, and normalized-contribution regressions."""

import itertools

import numpy as np
import pytest

from superglm.distributional import GammaLS, GaussianLS
from superglm.distributional.families import _base
from superglm.distributional.families import _response_bounds as bounded
from superglm.distributional.family import bounded_observation
from superglm.distributional.kernels import _response_tails as tails
from superglm.distributional.observation import (
    LEFT_CENSORED,
    RIGHT_CENSORED,
    ResponseBounds,
    resolve_response_bounds,
    unvalidated_bound_arrays,
)
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights


def _weights(values):
    return resolve_likelihood_weights(
        np.array(values), n_observations=len(values), contract=WeightContract("frequency")
    )


def _bounds(lower, upper, side):
    return ResponseBounds.from_arrays(np.array(lower), np.array(upper), np.array(side))


def _mixture(*args, **kwargs):
    assert hasattr(tails, "log_mixture_jets"), "the positive-mixture helper is missing"
    return tails.log_mixture_jets(*args, **kwargs)


def _stream(k, order):
    assert hasattr(tails, "CoreLogMixtureAccumulator"), "the streaming mixture helper is missing"
    return tails.CoreLogMixtureAccumulator(k, order)


def test_atom_support_keeps_zero_lower_bound_and_immutable_take_identity():
    tiny = np.nextafter(0.0, 1.0)
    source = _bounds([-tiny, 0.0, tiny], [2.0] * 3, [0, -1, 1])
    continuous = source.on_support(0.0)
    atomic = source.on_support(0.0, atom_at_lower=True)
    np.testing.assert_array_equal(continuous.lower, [-np.inf, -np.inf, tiny])
    np.testing.assert_array_equal(atomic.lower, [-np.inf, 0.0, tiny])
    assert continuous.digest != atomic.digest
    direct = atomic.take(np.array([2, 0, 2, 1]))
    nested = atomic.take(np.array([1, 2, 0])).take(np.array([1, 2, 1, 0]))
    assert direct == nested
    assert direct.digest == nested.digest
    assert atomic.take(np.array([0, 1])).digest != atomic.take(np.array([1, 0])).digest
    for a in (direct.lower, direct.upper, direct.side):
        with pytest.raises(ValueError):
            a.setflags(write=True)


@pytest.mark.parametrize("side", [0, LEFT_CENSORED])
def test_open_lower_endpoint_is_checked_on_original_rows_after_zero_drop(side):
    weights = _weights([0, 1, 0, 3])
    arrays = unvalidated_bound_arrays(
        ([None, 0.0, None, 2.0], None), [None, False, None, "none" if side == 0 else "left"], 4
    )
    taken = tuple(a[weights.input_positions] for a in arrays)
    with pytest.raises(ValueError, match="input row 3"):
        resolve_response_bounds(np.array([1.0, 2.0]), taken, weights.input_positions)


def test_direct_bind_checks_exact_endpoint_before_continuous_clipping():
    bounds = _bounds([0.0], [1.0], [0])
    with pytest.raises(ValueError, match="input row 2"):
        bounded.bind_observation(
            bounded_observation(bounds),
            np.array([0.0]),
            _weights([0, 0, 1]),
            support_lower=0.0,
            family_name="toy",
        )


@pytest.mark.parametrize("family_type", [GaussianLS, GammaLS])
@pytest.mark.parametrize(
    "side,response",
    [
        (RIGHT_CENSORED, 2.0),
        (RIGHT_CENSORED, 0.5),
        (LEFT_CENSORED, 1.0),
        (LEFT_CENSORED, 3.0),
    ],
)
def test_direct_continuous_binding_rejects_invalid_censor_windows(family_type, side, response):
    observation = bounded_observation(_bounds([1.0], [2.0], [side]))
    with pytest.raises(ValueError, match="input row 2"):
        family_type().bind_likelihood(np.array([response]), _weights([0, 0, 1]), observation)


@pytest.mark.parametrize("family_type", [GaussianLS, GammaLS])
@pytest.mark.parametrize("side,response", [(RIGHT_CENSORED, 1.0), (LEFT_CENSORED, 2.0)])
def test_direct_continuous_binding_accepts_certain_event_at_valid_boundary(
    family_type, side, response
):
    family = family_type()
    y = np.array([response])
    observation = bounded_observation(_bounds([1.0], [2.0], [side]))
    plan = family.bind_likelihood(y, _weights([1]), observation)
    evaluated = family.evaluate_natural(y, np.array([[1.5, 1.0]]), plan)

    assert evaluated.valid[0]
    np.testing.assert_array_equal(evaluated.optimizing_log_likelihood, [0.0])
    np.testing.assert_array_equal(evaluated.parameter_independent_carrier, [0.0])
    np.testing.assert_array_equal(evaluated.score, np.zeros((1, 2)))
    np.testing.assert_array_equal(evaluated.hessian_packed, np.zeros((1, 3)))


@pytest.mark.parametrize("atomic", [False, True])
def test_right_threshold_at_support_is_valid_and_left_threshold_depends_on_atom(atomic):
    weights = _weights([1])
    right = _bounds([-np.inf], [np.inf], [RIGHT_CENSORED])
    assert not bounded.bind_observation(
        bounded_observation(right),
        np.array([0.0]),
        weights,
        support_lower=0.0,
        family_name="toy",
        atom_at_lower=atomic,
    ).is_complete
    left = _bounds([-np.inf], [0.0], [LEFT_CENSORED])
    if atomic:
        bound = bounded.bind_observation(
            bounded_observation(left),
            np.array([0.0]),
            weights,
            support_lower=0.0,
            family_name="toy",
            atom_at_lower=True,
        )
        assert bound.bounds.upper[0] == 0.0
    else:
        with pytest.raises(ValueError, match="zero probability.*input row 0"):
            bounded.bind_observation(
                bounded_observation(left),
                np.array([0.0]),
                weights,
                support_lower=0.0,
                family_name="toy",
            )


def test_impossible_atomic_window_retains_input_positions_after_take():
    weights = _weights([0, 1, 0, 1]).take(np.array([1, 0]))
    bounds = _bounds([-2.0, -np.inf], [-1.0, 1.0], [LEFT_CENSORED, 0])
    with pytest.raises(ValueError, match="input row 3"):
        bounded.bind_observation(
            bounded_observation(bounds),
            np.array([-1.0, 0.0]),
            weights,
            support_lower=0.0,
            family_name="toy",
            atom_at_lower=True,
        )


def test_positive_only_atomic_window_cannot_observe_only_zero():
    bounds = _bounds([0.0], [1.0], [LEFT_CENSORED])
    with pytest.raises(ValueError, match="input row 2"):
        bounded.bind_observation(
            bounded_observation(bounds),
            np.array([0.0]),
            _weights([0, 0, 1]),
            support_lower=0.0,
            family_name="toy",
            atom_at_lower=True,
        )


def test_atomic_value_dispatch_matches_a_literal_discrete_probability_table():
    # A law with masses 1/4 at zero and 3/4 at one. No epsilon may move zero.
    tiny = np.nextafter(0.0, 1.0)
    lo = np.array([-np.inf, 0.0, -tiny, 0.0, -np.inf, tiny])
    hi = np.array([0.0, np.inf, 0.0, 0.0, np.inf, 1.0])

    def window(lower, upper, rows):
        mass = 0.25 * ((lower < 0) & (upper >= 0)) + 0.75 * ((lower < 1) & (upper >= 1))
        with np.errstate(divide="ignore"):
            return np.log(mass), None, None

    actual = bounded.log_window_values(
        lo, hi, len(lo), window, support_lower=0.0, atom_at_lower=True
    )
    np.testing.assert_allclose(
        actual, [np.log(0.25), np.log(0.75), np.log(0.25), -np.inf, 0.0, np.log(0.75)], rtol=1e-15
    )
    continuous = bounded.log_window_values(
        np.array([-np.inf]), np.array([0.0]), 1, window, support_lower=0.0
    )
    assert continuous[0] == -np.inf


def test_scalar_threshold_helper_preserves_raw_exact_validation_and_thresholds():
    assert hasattr(_base, "validated_observation_response"), "scalar observation validator missing"

    def positive_integer(y):
        # A family law stronger than generic finiteness; receives original objects.
        assert y.dtype == object
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in y):
            raise ValueError("exact positive integers required")
        return np.asarray(y, dtype=float)

    y = np.array([2, 0.0, 1.25], dtype=object)
    got = _base.validated_observation_response(
        y,
        censored=np.array([False, True, True]),
        exact_validator=positive_integer,
        input_positions=np.array([2, 5, 8]),
    )
    np.testing.assert_array_equal(got, [2.0, 0.0, 1.25])
    with pytest.raises(ValueError, match="input row 8"):
        _base.validated_observation_response(
            y,
            censored=np.array([False, True, False]),
            exact_validator=positive_integer,
            input_positions=np.array([2, 5, 8]),
        )
    y[2] = True
    with pytest.raises(ValueError, match="input row 8"):
        _base.validated_observation_response(
            y,
            censored=np.array([False, True, True]),
            exact_validator=positive_integer,
            input_positions=np.array([2, 5, 8]),
        )


@pytest.mark.parametrize("order", [0, 1, 2])
def test_mixture_matches_analytic_log_cosh_and_uses_packed_covariance(order):
    # exp(common + t) + exp(common - t), t = x - 2y.
    # Hessian(log cosh(t)) = sech(t)^2 * [[1,-2],[-2,4]].
    t = np.array([0.0, 0.5, -2.0])
    common = np.array([-1e200, -1000.0, 1000.0])
    logs = np.column_stack((common + t, common - t))
    scores = np.broadcast_to([[1.0, -2.0], [-1.0, 2.0]], (3, 2, 2)).copy()
    hessians = np.zeros((3, 2, 3))
    jet = _mixture(logs, scores, hessians, order=order)
    np.testing.assert_allclose(jet[0], common + np.log(2 * np.cosh(t)), rtol=2e-15)
    if order >= 1:
        np.testing.assert_allclose(jet[1], np.tanh(t)[:, None] * [1.0, -2.0], atol=1e-14)
    if order == 2:
        np.testing.assert_allclose(
            jet[2], (1 / np.cosh(t) ** 2)[:, None] * [1.0, -2.0, 4.0], rtol=3e-14, atol=1e-14
        )
    acc = _stream(2, order)
    for column in range(2):
        acc.add((logs[:, column], scores[:, column], hessians[:, column])[: order + 1])
    for batch, streaming in zip(jet, acc.finish(), strict=True):
        np.testing.assert_array_equal(batch, streaming)


def test_mixture_centering_preserves_small_variance_under_large_common_score():
    # The variance of equally likely -1, 0, 1 is 2/3, also after translating by 2**45.
    score = (2.0**45 + np.array([-1.0, 0.0, 1.0]))[None, :, None]
    logs = np.zeros((1, 3))
    hessian = np.full((1, 3, 1), 0.25)
    for permutation in itertools.permutations(range(3)):
        jet = _mixture(
            logs[:, permutation], score[:, permutation], hessian[:, permutation], order=2
        )
        np.testing.assert_allclose(jet[2], [[11 / 12]], rtol=2e-15)
        np.testing.assert_array_equal(jet[1], [[2.0**45]])


def test_mixture_matches_independent_differentiation_of_positive_polynomials():
    import mpmath as mp

    # Positive terms, not exponentials of linear forms: nonzero component Hessians.
    x, y = 0.3, -0.4
    with mp.workdps(60):
        terms = (
            lambda a, b: 1 + a * a + b * b,
            lambda a, b: 2 + (a - b) ** 2,
            lambda a, b: 3 + (2 * a + b) ** 2,
        )
        functions = [lambda a, b, f=f: mp.log(f(a, b)) for f in terms]

        def total(a, b):
            return mp.log(sum(f(a, b) for f in terms))

        orders = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
        component = np.array([[float(mp.diff(f, (x, y), o)) for o in orders] for f in functions])
        oracle = np.array([float(mp.diff(total, (x, y), o)) for o in orders])
    jet = _mixture(component[None, :, 0], component[None, :, 1:3], component[None, :, 3:], order=2)
    np.testing.assert_allclose(
        np.concatenate([j.ravel() for j in jet]), oracle, rtol=3e-14, atol=2e-15
    )


def test_absent_terms_empty_sums_and_unresolved_components_are_distinct():
    logs = np.array(
        [[0.0, -np.inf], [-np.inf, -np.inf], [0.0, np.nan], [0.0, np.inf], [0.0, -1000.0]]
    )
    score = np.zeros((5, 2, 1))
    score[:, 1] = np.nan
    hessian = score.copy()
    jet = _mixture(logs, score, hessian, order=2)
    assert jet[0][0] == 0.0 and jet[1][0, 0] == jet[2][0, 0] == 0.0
    assert jet[0][1] == -np.inf
    assert np.isnan(jet[1][1, 0]) and np.isnan(jet[2][1, 0])
    assert all(np.all(np.isnan(j[2:])) for j in jet)
    empty = _mixture(np.empty((3, 0)), np.empty((3, 0, 2)), np.empty((3, 0, 3)), order=2)
    assert np.all(empty[0] == -np.inf) and np.all(np.isnan(empty[1]))
    with pytest.raises(ValueError, match="component|empty"):
        _stream(2, 2).finish()


def test_constant_mixture_and_single_component_preserve_all_packed_channels():
    logs = np.log([[0.125, 0.375, 0.5], [0.25, 0.25, 0.5]])
    jet = _mixture(logs, np.zeros((2, 3, 3)), np.zeros((2, 3, 6)), order=2)
    np.testing.assert_allclose(jet[0], 0.0, atol=3e-16)
    np.testing.assert_array_equal(jet[1], np.zeros((2, 3)))
    np.testing.assert_array_equal(jet[2], np.zeros((2, 6)))
    single = (
        np.array([-3.0, 2.0]),
        np.arange(6.0).reshape(2, 3),
        np.arange(-6.0, 6.0).reshape(2, 6),
    )
    acc = _stream(3, 2)
    acc.add(single)
    snapshot = acc.finish()
    for got, want in zip(snapshot, single, strict=True):
        np.testing.assert_array_equal(got, want)
        assert not np.shares_memory(got, want)
    # Output mutation and adding an absent component cannot alter accumulated state.
    snapshot[1][:] = 99.0
    acc.add((np.full(2, -np.inf), np.full((2, 3), np.nan), np.full((2, 6), np.nan)))
    for got, want in zip(acc.finish(), single, strict=True):
        np.testing.assert_array_equal(got, want)


def test_small_mixture_share_scales_covariance_before_squaring():
    # A Bernoulli-distributed score 0 or 1e200 with probability ~1e-300
    # has variance ~1e100, even though the unweighted square overflows.
    logs = np.array([[0.0, np.log(1e-300)]])
    jet = _mixture(logs, np.array([[[0.0], [1e200]]]), np.zeros((1, 2, 1)), order=2)
    np.testing.assert_allclose(jet[1], [[1e-100]], rtol=1e-13)
    np.testing.assert_allclose(jet[2], [[1e100]], rtol=1e-13)


@pytest.mark.parametrize("log_share,large_score", [(-800.0, 1e200), (-1500.0, 1e308)])
def test_underflowed_finite_share_can_still_have_representable_derivatives(log_share, large_score):
    import mpmath as mp

    # The probability underflows, but p*g*g does not. Treating the term as
    # absent loses the Hessian; replacing the anchor also must retain its score.
    with mp.workdps(80):
        p = mp.exp(log_share) / (1 + mp.exp(log_share))
        g = mp.mpf(large_score)
        expected_score, expected_hessian = float(p * g), float(p * (1 - p) * g * g)
    for permutation in ((0, 1), (1, 0)):
        logs = np.array([[0.0, log_share]])[:, permutation]
        scores = np.array([[[0.0], [large_score]]])[:, permutation]
        jet = _mixture(logs, scores, np.zeros((1, 2, 1)), order=2)
        np.testing.assert_allclose(jet[1], [[expected_score]], rtol=3e-13, atol=0)
        np.testing.assert_allclose(jet[2], [[expected_hessian]], rtol=3e-13, atol=0)


def test_all_censored_limits_do_not_enter_exact_family_validation():
    def no_realizations(y):
        raise AssertionError("no exact responses exist")

    got = _base.validated_observation_response(
        np.array([-2.0, 0.0, 1.5]),
        censored=np.ones(3, dtype=bool),
        exact_validator=no_realizations,
        input_positions=np.array([8, 3, 5]),
    )
    np.testing.assert_array_equal(got, [-2.0, 0.0, 1.5])
    with pytest.raises(ValueError, match="input row 3"):
        _base.validated_observation_response(
            np.array([-2.0, np.inf, 1.5]),
            censored=np.ones(3, dtype=bool),
            exact_validator=no_realizations,
            input_positions=np.array([8, 3, 5]),
        )


def test_scalar_observation_type_refusals_keep_source_positions():
    def real_values(raw):
        if any(isinstance(value, str) for value in raw):
            raise TypeError("real observations required")
        return np.asarray(raw, dtype=float)

    with pytest.raises(ValueError, match="real observations required.*input row 9"):
        _base.validated_observation_response(
            np.array([1.0, "bad"], dtype=object),
            censored=None,
            exact_validator=real_values,
            input_positions=np.array([3, 9]),
        )


@pytest.mark.parametrize("order", [-1, 3, 4, True, 1.0])
def test_new_mixture_helpers_accept_only_core_integer_orders(order):
    with pytest.raises((ValueError, TypeError), match="order"):
        _mixture(np.zeros((1, 1)), order=order)
    with pytest.raises((ValueError, TypeError), match="order"):
        _stream(2, order)


def test_mixture_shape_guards_prevent_accidental_terms_first_broadcasting():
    with pytest.raises(ValueError, match="shape|two-dimensional"):
        _mixture(np.zeros(3), order=0)
    with pytest.raises(ValueError, match="shape|score"):
        _mixture(np.zeros((3, 2)), np.zeros((2, 3, 1)), order=1)
    with pytest.raises(ValueError, match="shape|Hessian|hessian"):
        _mixture(np.zeros((3, 2)), np.zeros((3, 2, 2)), np.zeros((3, 2, 4)), order=2)
    acc = _stream(1, 0)
    acc.add((np.zeros(2),))
    with pytest.raises(ValueError, match="shape|row"):
        acc.add((np.zeros(3),))


@pytest.mark.parametrize("order", [0, 1, 2])
def test_normalized_override_is_unit_mass_owns_validity_and_skips_both_windows(order):
    bounds = _bounds([0.0, 0.0], [2.0, 2.0], [LEFT_CENSORED, 0])

    def ordinary(lo, hi, rows):
        assert 0 not in rows, "normalized row reached ordinary subtraction"
        return (
            np.full(len(rows), -2.0),
            np.full((len(rows), 1), 0.5),
            np.full((len(rows), 1), -0.25),
        )

    def normalized(rows, requested):
        np.testing.assert_array_equal(rows, [0])
        assert requested == order
        return (
            np.array([-np.log(2.0)]),
            np.array([[0.5]]) if order >= 1 else None,
            np.array([[-0.25]]) if order == 2 else None,
        )

    result = bounded.bounded_channels(
        np.array([np.nan, -6.0]),
        np.array([[np.nan], [2.0]]) if order >= 1 else None,
        np.array([[np.nan], [-1.0]]) if order == 2 else None,
        np.array([False, True]),
        response=np.array([1.0, 1.0]),
        bounds=bounds,
        multiplier=np.array([3.0, 2.0]),
        window=ordinary,
        derivative_order=order,
        normalized_rows=np.array([0]),
        normalized=normalized,
        carrier=np.zeros(2),
    )
    np.testing.assert_allclose(result[0], [-3 * np.log(2), -2])
    np.testing.assert_array_equal(result[3], [True, True])
    if order >= 1:
        np.testing.assert_array_equal(result[1], [[1.5], [1.0]])
    if order == 2:
        np.testing.assert_array_equal(result[2], [[-0.75], [-0.5]])


def test_centered_normalization_survives_unresolvable_separate_channels():
    # A singleton event among two equal huge log masses has log probability -log(2).
    anchor = -1e200
    assert anchor - (anchor + np.log(2)) == 0.0  # unfixed separate subtraction loses it.
    bounds = _bounds([0.0], [2.0], [LEFT_CENSORED])

    def unresolved(*args):
        raise AssertionError("separate numerator/denominator must not be evaluated")

    result = bounded.bounded_channels(
        np.array([np.nan]),
        np.array([[np.nan]]),
        np.array([[np.nan]]),
        np.array([False]),
        response=np.ones(1),
        bounds=bounds,
        multiplier=np.array([3.0]),
        window=unresolved,
        derivative_order=2,
        normalized_rows=np.array([True]),
        carrier=np.zeros(1),
        normalized=lambda rows, order: (
            np.array([-np.log(2)]),
            np.array([[0.5]]),
            np.array([[-0.25]]),
        ),
    )
    np.testing.assert_allclose(result[0], [-3 * np.log(2)])
    assert result[3][0]


@pytest.mark.parametrize(
    "carrier", [None, np.array([1e200]), np.array([np.nan]), np.array([1e-300])]
)
def test_normalized_override_requires_explicit_exact_zero_carrier(carrier):
    with pytest.raises(ValueError, match="carrier"):
        bounded.bounded_channels(
            np.zeros(1),
            None,
            None,
            np.ones(1, dtype=bool),
            response=np.ones(1),
            bounds=_bounds([0.0], [2.0], [0]),
            multiplier=np.ones(1),
            window=lambda *args: (np.zeros(1), None, None),
            derivative_order=0,
            normalized_rows=np.array([0]),
            normalized=lambda *args: (np.zeros(1), None, None),
            carrier=carrier,
        )


def test_unresolved_normalized_derivative_is_not_authorized_by_finite_value():
    result = bounded.bounded_channels(
        np.zeros(1),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.ones(1, dtype=bool),
        response=np.ones(1),
        bounds=_bounds([0.0], [2.0], [0]),
        multiplier=np.ones(1),
        window=lambda *args: (np.zeros(1), None, None),
        derivative_order=2,
        normalized_rows=np.array([0]),
        carrier=np.zeros(1),
        normalized=lambda rows, order: (np.array([-0.5]), np.array([[0.0]]), np.array([[np.nan]])),
    )
    assert not result[3][0]
    assert all(np.all(np.isfinite(j)) for j in result[:3])


@pytest.mark.parametrize(
    "rows",
    [
        np.array([0, 0]),
        np.array([-1]),
        np.array([1]),
        np.array([[0]]),
        np.array([0.0]),
        np.array([True, False]),
    ],
)
def test_normalized_row_selection_rejects_ambiguous_or_out_of_range_rows(rows):
    with pytest.raises(ValueError, match="normalized_rows"):
        bounded.bounded_channels(
            np.zeros(1),
            None,
            None,
            np.ones(1, dtype=bool),
            response=np.ones(1),
            bounds=_bounds([0.0], [2.0], [0]),
            multiplier=np.ones(1),
            window=lambda *args: (np.zeros(1), None, None),
            derivative_order=0,
            normalized_rows=rows,
            normalized=lambda *args: (np.zeros(1), None, None),
            carrier=np.zeros(1),
        )


def test_normalized_callback_must_supply_requested_channels():
    with pytest.raises(ValueError, match="order-2"):
        bounded.bounded_channels(
            np.zeros(1),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.ones(1, dtype=bool),
            response=np.ones(1),
            bounds=_bounds([0.0], [2.0], [0]),
            multiplier=np.ones(1),
            window=lambda *args: (np.zeros(1), None, None),
            derivative_order=2,
            normalized_rows=np.array([0]),
            carrier=np.zeros(1),
            normalized=lambda rows, order: (np.array([-0.5]), np.array([[0.0]]), None),
        )
