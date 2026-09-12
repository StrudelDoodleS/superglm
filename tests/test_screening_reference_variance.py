"""Gaussian quadratic-form moments used by the interaction ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

import superglm.model.screening_ops as ops
import superglm.screening._structured as st
from superglm import SuperGLM
from superglm.features import Spline
from superglm.screening._pair_factor import PairFactor
from superglm.screening._score_stat import (
    _EDF_TOL,
    _lambda_bracket,
    _pair_pencil,
    _pencil_edf,
    _pencil_reference_variance,
    _pencil_stat,
    penalized_score_statistic_ladder,
)
from superglm.screening._structured import spline_cat_moments, structured_ladder


def _identity_pair(width: int) -> PairFactor:
    joint = np.eye(width + 1)
    joint[:width, -1] = np.arange(1.0, width + 1.0)
    return PairFactor(joint=joint, overlap_width=0, tensor_width=width)


@pytest.mark.parametrize(
    ("penalty", "variance"),
    [([1.0, 1.0, 1.0, 1.0], 2.0), ([0.0, 1.0, 3.0, 3.0], 2.75)],
)
def test_equal_edf_candidates_retain_their_distinct_reference_variances(penalty, variance):
    """Using 2*EDF misses both exact moments and their geometric difference."""
    result = penalized_score_statistic_ladder(
        _identity_pair(4), np.diag(np.sqrt(penalty)), budgets=(2.0,)
    )[0]
    # At lambda=1 the filters are either (1/2, 1/2, 1/2, 1/2) or
    # (1, 1/2, 1/4, 1/4). Both sum to 2; twice the sum of their squares differs.
    # Along the monotone ladder |d(2 sum a^2)/d(sum a)| <= 4. This transfers
    # the achieved-EDF error into a variance bound without a fitted tolerance.
    bound = 4.0 * abs(result.edf0 - 2.0) + 64 * np.finfo(float).eps
    assert getattr(result, "reference_variance", None) == pytest.approx(variance, abs=bound)


@pytest.mark.parametrize("width", [1, 4, 9])
def test_unpenalized_reference_variance_is_twice_the_identified_rank(width):
    result = penalized_score_statistic_ladder(_identity_pair(width), None)[0]
    assert result.edf0 == width
    assert getattr(result, "reference_variance", None) == 2.0 * width


@pytest.mark.parametrize("exponent", [-500, -300, 0, 300, 500])
def test_reference_variance_does_not_depend_on_penalty_units(exponent):
    result = penalized_score_statistic_ladder(
        _identity_pair(4), np.ldexp(np.eye(4), exponent), budgets=(2.0,)
    )[0]
    assert abs(result.edf0 - 2.0) <= _EDF_TOL
    bound = 4.0 * abs(result.edf0 - 2.0) + 64 * np.finfo(float).eps
    assert getattr(result, "reference_variance", None) == pytest.approx(2.0, abs=bound)


@pytest.mark.parametrize("exponent", [-537, -700])
def test_unrepresentable_penalty_target_clamps_to_a_finite_endpoint(exponent):
    """An infinite endpoint must not turn four identified directions into zero.

    For this root, lambda=2**(-2*exponent) would be needed to attain EDF 2,
    beyond float64's range. Even the largest finite lambda leaves the
    augmented identity well conditioned and its smoother near identity.
    """
    root = np.ldexp(np.eye(4), exponent)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = penalized_score_statistic_ladder(_identity_pair(4), root, budgets=(2.0,))[0]
    assert result.lambda0 == np.finfo(float).max
    # Compute the tiny perturbation without squaring the original root.
    penalty = np.sqrt(result.lambda0) * root
    filter_value = 1 / (1 + penalty[0, 0] ** 2)
    bound = 128 * np.finfo(float).eps
    assert result.edf0 == pytest.approx(4 * filter_value, abs=4 * bound, rel=0)
    assert result.reference_variance == pytest.approx(8 * filter_value**2, abs=8 * bound, rel=0)
    assert result.statistic == pytest.approx(30 * filter_value, abs=30 * bound, rel=0)


@pytest.mark.parametrize("root_diagonal", [[1.0, 1.0, 1.0, 1.0], [0.0, 0.5, 1.0, 2.0]])
def test_finite_large_penalty_root_retains_information_at_the_subnormal_edge(root_diagonal):
    """Squaring a finite factor must not turn an identified candidate into zero.

    At lambda=2**-1074, roots scaled by 2**538 give penalty eigenvalues
    4*root_diagonal**2. These small products provide an independent oracle.
    The mixed case also preserves the penalty's exact null direction.
    """
    root = np.ldexp(np.diag(root_diagonal), 538)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = penalized_score_statistic_ladder(_identity_pair(4), root, budgets=(2.0,))[0]
    filters = 1 / (1 + 4 * np.square(root_diagonal))
    assert result.lambda0 == np.finfo(float).smallest_subnormal
    bound = 128 * 4 * np.finfo(float).eps
    assert result.edf0 == pytest.approx(float(np.sum(filters)), abs=bound, rel=0)
    assert result.reference_variance == pytest.approx(
        2 * float(np.sum(filters**2)), abs=bound, rel=0
    )
    assert result.statistic == pytest.approx(
        float(np.arange(1, 5) ** 2 @ filters), abs=30 * bound, rel=0
    )


def test_unresolved_candidate_has_zero_reference_variance():
    pair = PairFactor(joint=np.zeros((3, 3)), overlap_width=0, tensor_width=2)
    result = penalized_score_statistic_ladder(pair, None)[0]
    assert result.edf0 == 0.0
    assert getattr(result, "reference_variance", None) == 0.0


def test_large_root_preserves_a_smaller_penalty_direction():
    """A global rescaling must not erase a sine before its square is restored."""
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        pencil = _pair_pencil(_identity_pair(2), np.diag([np.ldexp(1.0, 538), 1.0]))
        moments = (_pencil_edf(pencil, 1.0), _pencil_reference_variance(pencil, 1.0))
        statistic = _pencil_stat(pencil, 1.0)
    bound = 128 * 2 * np.finfo(float).eps
    assert moments == pytest.approx((0.5, 0.5), abs=bound, rel=0)
    assert statistic == pytest.approx(2.0, abs=4 * bound, rel=0)


def test_extreme_pencil_sums_before_rounding_subnormal_terms():
    """Individually unrepresentable filters can have a representable total."""
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        pencil = _pair_pencil(_identity_pair(4), np.ldexp(np.eye(4), 538))
        edf = _pencil_edf(pencil, 1.0)
        variance = _pencil_reference_variance(pencil, np.ldexp(1.0, -538))
    # Four filters near 2**-1076 sum to 2**-1074. At the second lambda,
    # 2*sum(a**2) is near 8*2**-1076. Both round to these exact floats.
    assert edf == np.finfo(float).smallest_subnormal
    assert variance == 2 * np.finfo(float).smallest_subnormal


def test_tiny_root_preserves_a_small_curvature_and_score_direction():
    """The balanced stack is well conditioned even though c**2 underflows."""
    tiny = np.ldexp(1.0, -540)
    joint = np.eye(3)
    joint[1, 1] = tiny
    joint[:2, -1] = 1.0
    pair = PairFactor(joint=joint, overlap_width=0, tensor_width=2)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        pencil = _pair_pencil(pair, tiny * np.eye(2))
        moments = (
            _pencil_edf(pencil, 1.0),
            _pencil_stat(pencil, 1.0),
            _pencil_reference_variance(pencil, 1.0),
        )
    # The diagonal filters round to (1, 1/2); the row score is (1, 1).
    assert moments == pytest.approx((1.5, 1.5, 2.5), abs=128 * 2 * np.finfo(float).eps, rel=0)


@pytest.mark.parametrize("target, multiple", [(0.7, 1), (0.6, 2)])
def test_unattainable_subnormal_target_chooses_the_closest_endpoint(target, multiple):
    result = penalized_score_statistic_ladder(
        _identity_pair(4), np.ldexp(np.eye(4), 538), budgets=(target,)
    )[0]
    assert result.lambda0 == multiple * np.finfo(float).smallest_subnormal
    assert result.edf0 == pytest.approx(4 / (1 + 4 * multiple), abs=128 * np.finfo(float).eps)


def test_scaled_lambda_bracket_recovers_an_underflowing_trace_quotient():
    # (2**-1074 / 4) * 2**1076 = 1, although the quotient alone is zero.
    bracket = _lambda_bracket(np.finfo(float).smallest_subnormal, denominator=4.0, exponent=1076)
    assert bracket == pytest.approx((1e-10, 1e10), rel=4 * np.finfo(float).eps, abs=0)


def test_public_z_uses_the_candidate_quadratic_variance(monkeypatch):
    """Restoring sqrt(2*EDF) must fail on a real penalized screen."""
    rng = np.random.default_rng(512)
    x, z = np.meshgrid(np.linspace(-1.0, 1.0, 21), np.linspace(-1.0, 1.0, 19))
    frame = pd.DataFrame({"x": x.ravel(), "z": z.ravel()})
    y = np.sin(2 * frame.x) * frame.z + rng.normal(size=len(frame))
    model = SuperGLM(
        family="gaussian", features={name: Spline(kind="ps", k=6) for name in frame}
    ).fit_reml(frame, y)
    captured = []
    real = ops.penalized_score_statistic_ladder

    def record(pair, penalty_root, **kwargs):
        captured.append((pair, penalty_root))
        return real(pair, penalty_root, **kwargs)

    monkeypatch.setattr(ops, "penalized_score_statistic_ladder", record)
    row = model.screen_interactions(frame, y, edf0=4.0, phi=1.0).iloc[0]
    assert len(captured) == 1
    pair, root = captured[0]
    overlap = scipy.linalg.orth(pair.joint[:, : pair.overlap_width])
    candidate = pair.joint[:, pair.overlap_width : -1]
    candidate = candidate - overlap @ (overlap.T @ candidate)
    stack = np.vstack((candidate, np.sqrt(row.lambda0) * root))
    triangular = scipy.linalg.qr(stack, mode="economic")[1]
    carried = scipy.linalg.solve_triangular(triangular.T, candidate.T, lower=True).T
    smoother = carried.T @ carried
    variance = 2 * np.sum(smoother**2)
    expected = (row.statistic - row.edf0) / np.sqrt(variance)
    condition = np.linalg.cond(stack)
    assert condition < 1e3, "This forward-accuracy fixture must be well conditioned"
    bound = 64 * max(stack.shape) * np.finfo(float).eps * condition**2 * max(1, abs(expected))
    assert row.z == pytest.approx(expected, abs=bound, rel=0)


def _small_spline_cat(penalty_scale=1.0, *, base_mass=1.0, reverse=False, n_levels=4):
    x = np.linspace(-1.0, 1.0, 9)
    basis = np.column_stack((x, x**2))
    weights = np.ones((len(x), n_levels))
    weights[:, 0] *= base_mass
    emitted = np.arange(1, n_levels)
    if reverse:
        weights = weights[:, ::-1]
        emitted = n_levels - 1 - emitted
    pair = spline_cat_moments(
        basis, penalty_scale * np.eye(2), np.zeros_like(weights), weights, emitted
    )
    # Assemble the weighted observation design independently of the arrow
    # factors. The emitted interaction columns are ordered by level.
    columns = np.vstack([basis for _ in range(n_levels)])
    levels = np.repeat(np.eye(n_levels), len(x), axis=0)
    overlap = np.column_stack((columns, levels))
    candidate = np.column_stack([columns * levels[:, j, None] for j in emitted])
    root_w = np.sqrt(weights.T.ravel())[:, None]
    overlap = scipy.linalg.orth(overlap * root_w)
    candidate = candidate * root_w
    candidate -= overlap @ (overlap.T @ candidate)
    return pair, candidate


@pytest.mark.parametrize("budgets", [(2.0,), (0.1, 100.0), (2.0, 2.0, 100.0, 100.0)])
@pytest.mark.parametrize("base_mass", [1.0, 0.0])
@pytest.mark.parametrize("n_levels, chunk_width", [(4, None), (10, None), (10, 2)])
def test_structured_variance_matches_the_observation_space_quadratic(
    budgets, base_mass, n_levels, chunk_width, monkeypatch
):
    """Cross-level norms must survive QR compression and later chunk merges.

    Four leaves only produce a compressed factor; ten also consume it in
    later merges. Truncating rows in place of QR passes the four-leaf case
    and fails the larger case against this observation-space oracle.
    """
    if chunk_width is not None:
        monkeypatch.setattr(st, "_trace_chunk_width", lambda *args: chunk_width)
    pair, candidate = _small_spline_cat(base_mass=base_mass, n_levels=n_levels)
    results = structured_ladder(pair, budgets=budgets)
    assert results is not None
    for result in results:
        stacked = np.vstack((candidate, np.sqrt(result.lambda0) * np.eye(candidate.shape[1])))
        _, singular, right = scipy.linalg.svd(stacked, full_matrices=False)
        carried = (candidate @ right.T) / singular
        smoother = carried.T @ carried
        variance = 2.0 * np.sum(smoother**2)
        # The variance is a fourth-degree norm of the carried design. Bound
        # its forward error through the augmented factor's condition number.
        condition = singular[0] / singular[-1]
        bound = 128 * max(stacked.shape) * np.finfo(float).eps * condition**2
        assert result.reference_variance == pytest.approx(variance, abs=bound, rel=0)


def test_structured_variance_is_invariant_to_level_order():
    forward = structured_ladder(_small_spline_cat()[0], budgets=(2.0,))[0]
    reverse = structured_ladder(_small_spline_cat(reverse=True)[0], budgets=(2.0,))[0]
    bound = 4.0 * abs(forward.edf0 - reverse.edf0) + 256 * np.finfo(float).eps
    assert reverse.reference_variance == pytest.approx(forward.reference_variance, abs=bound)


@pytest.mark.parametrize("reverse", [False, True])
def test_absorbed_border_preserves_the_unpenalized_projector(reverse):
    """An absent base level leaves four identified interaction directions."""
    pair, candidate = _small_spline_cat(penalty_scale=0.0, base_mass=0.0, reverse=reverse)
    singular = scipy.linalg.svdvals(candidate)
    cutoff = max(candidate.shape) * np.finfo(float).eps * singular[0]
    kept = singular[singular > cutoff]
    assert kept.size == 4
    condition = kept[0] / kept[-1]
    assert condition < 10
    result = structured_ladder(pair, budgets=(2.0,))[0]
    bound = 128 * max(candidate.shape) * np.finfo(float).eps * condition**2 * kept.size
    assert result.edf0 == pytest.approx(kept.size, abs=bound, rel=0)
    assert result.reference_variance == pytest.approx(2 * kept.size, abs=bound, rel=0)
