"""Analytic support regressions for scalar and compact penalty consumers."""

from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest

from superglm.reml import penalty_algebra as algebra
from superglm.reml.result import PenaltyCache
from superglm.types import LambdaPolicy, PenaltyComponent


def _component(name, matrix, **kwargs):
    matrix = np.asarray(matrix, dtype=float)
    return PenaltyComponent(
        name=name,
        group_name="shared",
        group_index=0,
        group_sl=slice(0, matrix.shape[0]),
        omega_raw=matrix,
        omega_ssp=matrix,
        **kwargs,
    )


def _tensor(a, b, left=1.0, right=1.0):
    summary = algebra.TensorPairLogdetSummary(
        "tensor", 7, ("left", "right"), np.array([0.0, left]), np.array([0.0, right])
    )
    return algebra.evaluate_tensor_pair_logdet_summaries(
        {"tensor": summary}, {"left": a, "right": b}
    )["tensor"]


@pytest.mark.parametrize(
    "a,b,left,right",
    [(1e12, 3.0, 1.0, 1.0), (1e200, 3e200, 1.0, 1.0), (1e300, 3.0, 1e50, 1.0)],
)
def test_tensor_support_and_log_domain_derivatives(a, b, left, right):
    """Kills the weighted rank cutoff and unscaled cross-product overflow."""
    result = _tensor(a, b, left, right)
    with localcontext() as context:
        context.prec = 80
        x = Decimal.from_float(a) * Decimal.from_float(left)
        y = Decimal.from_float(b) * Decimal.from_float(right)
        expected = float(x.ln() + y.ln() + (x + y).ln())
        gradient = np.array([float(1 + x / (x + y)), float(1 + y / (x + y))])
        cross = float(x * y / (x + y) ** 2)
    assert result.rank == 3
    reference_error = 2 * np.finfo(float).eps * max(abs(expected), 1.0)
    assert abs(result.logdet_s_plus - expected) <= result.logdet_error + reference_error
    for index, name in enumerate(("left", "right")):
        assert abs(result.gradient[name] - gradient[index]) <= (
            result.gradient_error[name] + 2 * np.finfo(float).eps * gradient[index]
        )
    assert abs(result.hessian["left", "left"] - cross) <= (
        result.hessian_error["left", "left"] + 2 * np.finfo(float).eps * cross
    )
    assert result.hessian["left", "right"] == -result.hessian["left", "left"]
    # The local bounds must resolve the analytic observables.
    assert result.logdet_error < 1e-10 * max(abs(expected), 1.0)
    assert max(result.gradient_error.values()) < 1e-10


@pytest.mark.parametrize("weights,rank", [((0.0, 0.0), 0), ((0.0, 3.0), 2), ((2.0, 0.0), 2)])
def test_tensor_exact_zero_activity(weights, rank):
    result = _tensor(*weights)
    assert result.rank == rank
    assert sum(result.gradient.values()) == pytest.approx(rank, abs=8 * np.finfo(float).eps)
    assert all(value == 0 for value in result.hessian.values())
    for name, weight in zip(("left", "right"), weights, strict=True):
        if weight == 0:
            assert result.gradient[name] == 0


def test_tensor_inverse_component_unit_change():
    reference = _tensor(2.0, 3.0)
    changed = _tensor(2.0**-500 * 2.0, 3.0, 2.0**500)
    assert changed.rank == reference.rank
    assert abs(changed.logdet_s_plus - reference.logdet_s_plus) <= (
        changed.logdet_error + reference.logdet_error
    )
    for name in reference.gradient:
        assert abs(changed.gradient[name] - reference.gradient[name]) <= (
            changed.gradient_error[name] + reference.gradient_error[name]
        )


def test_total_rank_treats_supplied_tensor_components_as_active():
    left = np.diag([0.0, 0.0, 1.0, 1.0])
    right = np.diag([0.0, 1.0, 0.0, 1.0])
    penalties = [_component("left", left), _component("right", right)]
    evaluation = _tensor(0.0, 3.0)
    assert evaluation.rank == 2
    assert algebra.compute_total_penalty_rank(penalties, {"shared": evaluation}) == 3


@pytest.mark.parametrize("bad", [-1.0, np.inf, np.nan])
def test_tensor_rejects_invalid_weights(bad):
    with pytest.raises(ValueError, match="finite.*non.?negative"):
        _tensor(bad, 1.0)


def test_scalar_consumers_keep_all_positive_weight_directions():
    penalties = [
        _component("a", np.diag([1.0, 0.0]), rank=1),
        _component("b", np.diag([0.0, 1.0]), rank=1),
    ]
    lambdas = {"a": 1e12, "b": 3.0}
    expected = np.log(1e12) + np.log(3.0)
    assert algebra.compute_logdet_s_plus(lambdas, penalties) == pytest.approx(
        expected, rel=0, abs=128 * np.finfo(float).eps * expected
    )
    gradient, hessian = algebra.compute_logdet_s_derivatives(lambdas, penalties)
    np.testing.assert_allclose(list(gradient.values()), [1, 1], atol=128 * np.finfo(float).eps)
    assert max(abs(value) for value in hessian.values()) < 128 * np.finfo(float).eps
    assert algebra.compute_total_penalty_rank(penalties) == 2
    assert (
        algebra.compute_penalty_nullity(
            hessian_rank=3, penalties=penalties, lambdas=lambdas, coefficient_width=2
        )
        == 1
    )


def test_scalar_rank_is_invariant_to_component_units():
    penalties = [
        _component("a", np.diag([1e-20, 0.0]), rank=1),
        _component("b", np.diag([0.0, 1.0]), rank=1),
    ]
    assert algebra.compute_total_penalty_rank(penalties) == 2
    assert (
        algebra.compute_penalty_nullity(
            hessian_rank=3, penalties=penalties, lambdas={"a": 1e20, "b": 1.0}
        )
        == 1
    )


def test_singleton_uses_checked_geometry_and_zero_weight_derivative():
    penalty = _component("a", np.diag([2.0, 0.0]), rank=2, log_det_omega_plus=123.0)
    assert algebra.compute_total_penalty_rank([penalty]) == 1
    assert algebra.compute_logdet_s_plus({"a": 3.0}, [penalty]) == pytest.approx(np.log(6))
    gradient, hessian = algebra.compute_logdet_s_derivatives({"a": 0.0}, [penalty])
    assert gradient == {"a": 0.0}
    assert hessian == {("a", "a"): 0.0}
    cache = PenaltyCache(
        omega_ssp=penalty.omega_ssp,
        rank=2,
        log_det_omega_plus=123.0,
        eigvals_omega=np.array([2.0]),
    )
    assert algebra.cached_logdet_s_plus({"a": 3.0}, {"a": cache}) == pytest.approx(np.log(6))


def test_legacy_identity_cache_keeps_its_analytic_geometry():
    cache = PenaltyCache(omega_ssp=None, rank=3.0, log_det_omega_plus=0.0, eigvals_omega=None)
    assert algebra.cached_logdet_s_plus({"identity": 2.0}, {"identity": cache}) == pytest.approx(
        3 * np.log(2.0)
    )


def test_legacy_uncached_singletons_share_the_canonical_cached_support():
    """Internally rounded SSP congruences must use the established producer."""
    from superglm.reml.gradient import reml_direct_gradient
    from superglm.reml.penalty_support import _penalty_support
    from tests.test_hessian_ift import _setup

    model, _, _, _, weights, groups, _, caches, fit, inverse, *_ = _setup("poisson")
    matrices = model._dm.group_matrices
    uncached = algebra.coerce_reml_penalties(reml_groups=groups, group_matrices=matrices)
    cached = algebra.coerce_reml_penalties(
        reml_groups=groups, group_matrices=matrices, penalty_caches=caches
    )
    assert [component.name for component in uncached] == [group.name for _, group in groups]
    for ordinary, reference in zip(uncached, cached, strict=True):
        support = _penalty_support([ordinary.omega_ssp])
        expected = _penalty_support([reference.omega_ssp])
        assert support.rank == expected.rank == ordinary.rank == reference.rank == 8
        root, reference_root = support.component_roots[0], expected.component_roots[0]
        reference_gram = reference_root.T @ reference_root
        allowance = 32 * root.shape[1] * np.finfo(float).eps * np.linalg.norm(reference_gram)
        assert np.linalg.norm(root.T @ root - reference_gram) <= allowance
        assert ordinary.group_sl == reference.group_sl
    actual = algebra._compute_penalty_logdet_evaluation(weights, uncached)
    reference = algebra._compute_penalty_logdet_evaluation(weights, cached)
    assert actual.rank == reference.rank == 16
    assert abs(actual.logdet - reference.logdet) <= actual.logdet_error + reference.logdet_error
    assert actual.gradient == reference.gradient
    assert actual.hessian == reference.hessian
    options = dict(reml_groups=groups)
    gradient = reml_direct_gradient(matrices, fit, inverse, weights, **options)
    cached_gradient = reml_direct_gradient(
        matrices, fit, inverse, weights, penalty_caches=caches, **options
    )
    allowance = 64 * len(fit.beta) * np.finfo(float).eps * (1 + np.linalg.norm(cached_gradient))
    assert np.linalg.norm(gradient - cached_gradient) <= allowance


def test_singleton_unit_compensation_does_not_require_the_unweighted_inverse():
    penalty = _component("a", [[1e-310]])
    weight = 1e300
    with localcontext() as context:
        context.prec = 80
        expected = float((Decimal.from_float(1e-310) * Decimal.from_float(weight)).ln())
    evaluation = algebra._compute_penalty_logdet_evaluation({"a": weight}, [penalty])
    assert evaluation.rank == 1
    assert abs(evaluation.logdet - expected) <= evaluation.logdet_error
    assert evaluation.gradient == {"a": 1.0}


def test_mixed_compact_group_uses_the_dense_expander():
    repeated = replace(
        _component("a", [[2.0]], rank=2),
        penalty_kind="repeated",
        repeat_count=2,
        block_width=1,
        group_sl=slice(0, 2),
    )
    identity = replace(repeated, name="b", penalty_kind="identity", omega_ssp=None)
    penalties = [repeated, identity]
    assert algebra.compute_logdet_s_plus({"a": 3.0, "b": 4.0}, penalties) == pytest.approx(
        2 * np.log(10)
    )
    assert algebra.compute_total_penalty_rank(penalties) == 2


def test_repeated_group_stays_compact_and_includes_fixed_components(monkeypatch):
    penalties = [
        replace(
            _component("a", np.diag([1.0, 0.0])),
            penalty_kind="repeated",
            repeat_count=5,
            block_width=2,
            group_sl=slice(0, 10),
        ),
        replace(
            _component("b", np.diag([1.0, 1.0])),
            penalty_kind="repeated",
            repeat_count=5,
            block_width=2,
            group_sl=slice(0, 10),
            lambda_policy=LambdaPolicy.fixed(3.0),
        ),
    ]

    def no_expansion(*_args, **_kwargs):
        pytest.fail("all-repeated penalty groups must evaluate their local blocks")

    monkeypatch.setattr(algebra, "penalty_component_dense_matrix", no_expansion)
    evaluation = algebra._compute_penalty_logdet_evaluation({"a": 2.0, "b": 3.0}, penalties)
    assert evaluation.rank == 10
    assert evaluation.logdet == pytest.approx(5 * np.log(15))
    assert evaluation.gradient["a"] == pytest.approx(2)
    assert evaluation.gradient["b"] == pytest.approx(8)
    assert evaluation.hessian["a", "b"] == pytest.approx(-1.2)


def test_public_discrete_tensor_only_factors_singleton_marginals(monkeypatch):
    """Check actual root dimensions as well as use of the tensor spectral path."""
    import pandas as pd

    import superglm.model.reml_finalize as finalize
    import superglm.reml.discrete as discrete
    import superglm.reml.multi_penalty as kernel
    from superglm import SuperGLM
    from superglm.features import Spline

    singleton_dimensions = []
    tensor_dimensions = {}
    evaluate_support = kernel._evaluate_penalty_geometry
    evaluate_tensor = algebra.evaluate_tensor_pair_logdet_summaries

    def capture_support(support, *args, **kwargs):
        singleton_dimensions.append(
            (len(support.component_roots), support.Q_plus.shape[0], support.rank)
        )
        return evaluate_support(support, *args, **kwargs)

    def capture_tensor(summaries, *args, **kwargs):
        for name, summary in summaries.items():
            tensor_dimensions[name] = (len(summary.eigvals_left), len(summary.eigvals_right))
        return evaluate_tensor(summaries, *args, **kwargs)

    monkeypatch.setattr(kernel, "_evaluate_penalty_geometry", capture_support)
    monkeypatch.setattr(discrete, "evaluate_tensor_pair_logdet_summaries", capture_tensor)
    monkeypatch.setattr(finalize, "evaluate_tensor_pair_logdet_summaries", capture_tensor)
    rng = np.random.default_rng(77)
    x1, x2 = rng.uniform(size=(2, 260))
    y = rng.poisson(np.exp(0.2 + np.sin(2 * np.pi * x1) + 0.3 * np.cos(2 * np.pi * x2)))
    model = SuperGLM(
        family="poisson",
        selection_penalty=0,
        discrete=True,
        features={"x1": Spline(n_knots=5), "x2": Spline(n_knots=5)},
        interactions=[("x1", "x2")],
    ).fit_reml(pd.DataFrame({"x1": x1, "x2": x2}), y, max_reml_iter=3, reml_tol=1e-12)
    assert tensor_dimensions and singleton_dimensions
    singleton_widths = {
        component.group_sl.stop - component.group_sl.start
        for component in model._reml_penalties
        if component.group_name not in tensor_dimensions
    }
    tensor_widths = {left * right for left, right in tensor_dimensions.values()}
    for count, width, rank in singleton_dimensions:
        assert count == 1
        assert width in singleton_widths
        assert 0 < rank <= width < min(tensor_widths)


@pytest.mark.parametrize("bad", [-1.0, np.inf, np.nan])
def test_scalar_rejects_invalid_weights(bad):
    penalty = _component("a", np.eye(2), rank=2)
    with pytest.raises(ValueError, match="finite.*non.?negative"):
        algebra.compute_logdet_s_plus({"a": bad}, [penalty])
