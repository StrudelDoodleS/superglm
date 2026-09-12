"""Gaussian quadratic-form moments used by the interaction ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

import superglm.model.screening_ops as ops
from superglm import SuperGLM
from superglm.features import Spline
from superglm.screening._pair_factor import PairFactor
from superglm.screening._score_stat import _EDF_TOL, penalized_score_statistic_ladder
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


@pytest.mark.parametrize("exponent", [-300, 0, 300])
def test_reference_variance_does_not_depend_on_penalty_units(exponent):
    result = penalized_score_statistic_ladder(
        _identity_pair(4), np.ldexp(np.eye(4), exponent), budgets=(2.0,)
    )[0]
    assert abs(result.edf0 - 2.0) <= _EDF_TOL
    bound = 4.0 * abs(result.edf0 - 2.0) + 64 * np.finfo(float).eps
    assert getattr(result, "reference_variance", None) == pytest.approx(2.0, abs=bound)


def test_unresolved_candidate_has_zero_reference_variance():
    pair = PairFactor(joint=np.zeros((3, 3)), overlap_width=0, tensor_width=2)
    result = penalized_score_statistic_ladder(pair, None)[0]
    assert result.edf0 == 0.0
    assert getattr(result, "reference_variance", None) == 0.0


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


def _small_spline_cat(penalty_scale=1.0, *, base_mass=1.0, reverse=False):
    x = np.linspace(-1.0, 1.0, 9)
    basis = np.column_stack((x, x**2))
    weights = np.ones((len(x), 4))
    weights[:, 0] *= base_mass
    emitted = np.arange(1, 4)
    if reverse:
        weights = weights[:, ::-1]
        emitted = 3 - emitted
    pair = spline_cat_moments(
        basis, penalty_scale * np.eye(2), np.zeros_like(weights), weights, emitted
    )
    # Assemble the weighted observation design independently of the arrow
    # factors. The emitted interaction columns are ordered by level.
    columns = np.vstack([basis for _ in range(4)])
    levels = np.repeat(np.eye(4), len(x), axis=0)
    overlap = np.column_stack((columns, levels))
    candidate = np.column_stack([columns * levels[:, j, None] for j in emitted])
    root_w = np.sqrt(weights.T.ravel())[:, None]
    overlap = scipy.linalg.orth(overlap * root_w)
    candidate = candidate * root_w
    candidate -= overlap @ (overlap.T @ candidate)
    return pair, candidate


@pytest.mark.parametrize("budgets", [(2.0,), (0.1, 100.0), (2.0, 2.0, 100.0, 100.0)])
@pytest.mark.parametrize("base_mass", [1.0, 0.0])
def test_structured_variance_matches_the_observation_space_quadratic(budgets, base_mass):
    """The arrow route must retain off-diagonal smoother contributions."""
    pair, candidate = _small_spline_cat(base_mass=base_mass)
    results = structured_ladder(pair, budgets=budgets)
    assert results is not None
    for result in results:
        stacked = np.vstack((candidate, np.sqrt(result.lambda0) * np.eye(6)))
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
