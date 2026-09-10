"""Fixed latent SCOP penalties retain their checked target across mode fits."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import superglm.reml.multi_penalty as kernel
import superglm.reml.penalty_algebra as algebra
import superglm.reml.penalty_support as support_module
import superglm.reml.scop_efs as scop
from superglm.types import PenaltyComponent


def _states(width=11, groups=2):
    difference = np.diff(np.eye(width - 1), axis=0)
    matrix = np.zeros((width, width))
    matrix[1:, 1:] = difference.T @ difference
    return {
        index: {
            "S_scop": matrix.copy(),
            "group_sl": slice(index * width, (index + 1) * width),
            "group_name": f"x{index}",
            "beta_eff": np.zeros(width),
            "reparam": object(),
        }
        for index in range(groups)
    }


def _build(states, cache, weights=None):
    return scop.build_scop_penalty_components(states, _cache=cache, _lambdas=weights)


def test_fixed_two_group_geometry_is_built_once_for_all_three_consumers(monkeypatch):
    states, cache = _states(), {}
    builds, summaries, full = [], [], []
    real_support = support_module._penalty_support
    real_summary = kernel._evaluate_penalty_summary
    real_full = kernel._evaluate_penalty_support

    def support(matrices):
        builds.append(tuple(matrix.shape for matrix in matrices))
        return real_support(matrices)

    def summary(selected, weights, *args, **kwargs):
        summaries.append((selected.Q_plus.shape, tuple(weights)))
        return real_summary(selected, weights, *args, **kwargs)

    def full_result(*args, **kwargs):
        full.append(True)
        return real_full(*args, **kwargs)

    monkeypatch.setattr(support_module, "_penalty_support", support)
    monkeypatch.setattr(kernel, "_evaluate_penalty_summary", summary)
    monkeypatch.setattr(kernel, "_evaluate_penalty_support", full_result)
    previous = None
    for mode in range(42):
        states = {
            index: {**state, "beta_eff": np.full(11, mode / 100)} for index, state in states.items()
        }
        components = _build(states, cache)
        if previous is not None:
            assert all(left is right for left, right in zip(components, previous, strict=True))
        values = {item.name: (mode + 1) / 100 for item in components}
        evaluation = algebra._compute_penalty_logdet_evaluation(values, components)
        assert evaluation.rank == 18
        if mode < 41:
            gradient, hessian = algebra.compute_logdet_s_derivatives(values, components)
            assert gradient == {"x0": 9.0, "x1": 9.0}
            assert not any(hessian.values())
        assert (
            algebra.compute_penalty_nullity(
                penalties=components, lambdas=values, coefficient_width=22, hessian_rank=22
            )
            == 4
        )
        previous = components
    assert builds == [((11, 11),), ((11, 11),)]
    assert summaries == [((11, 9), (1.0,)), ((11, 9), (1.0,))]
    assert full == []


@pytest.mark.parametrize("weights", [(2.0, 3.0), (0.0, 3.0), (2.0, 0.0), (0.0, 0.0)])
def test_cached_latent_target_preserves_rank_values_derivatives_and_bounds(weights):
    states = _states(width=5)
    original = scop.build_scop_penalty_components(states)
    retained = _build(states, {})
    values = dict(zip((item.name for item in original), weights, strict=True))
    before = algebra._compute_penalty_logdet_evaluation(values, original)
    after = algebra._compute_penalty_logdet_evaluation(values, retained)
    assert after.rank == before.rank == 3 * sum(value > 0 for value in weights)
    assert abs(after.logdet - before.logdet) <= after.logdet_error + before.logdet_error
    assert after.gradient == before.gradient
    assert after.hessian == before.hessian
    assert after.gradient_error == before.gradient_error
    assert after.hessian_error == before.hessian_error
    for old, new in zip(original, retained, strict=True):
        np.testing.assert_array_equal(new.omega_ssp, old.omega_ssp)
        assert algebra._context_geometry([new]) is not None


@pytest.mark.parametrize(
    "field",
    [
        "name",
        "group_name",
        "group_index",
        "group_sl",
        "omega_raw",
        "omega_ssp",
        "rank",
        "log_det_omega_plus",
        "eigvals_omega",
        "component_type",
        "lambda_policy",
        "penalty_kind",
        "repeat_count",
        "block_width",
    ],
)
def test_every_returned_descriptor_field_is_authenticated(field):
    states, cache = _states(width=5, groups=1), {}
    first = _build(states, cache)[0]
    value = getattr(first, field)
    if isinstance(value, np.ndarray):
        changed = value.copy()
    elif isinstance(value, slice):
        changed = slice(value.start + 1, value.stop + 1)
    elif isinstance(value, str):
        changed = "changed"
    elif value is None:
        changed = "changed"
    else:
        changed = value + 1
    setattr(first, field, changed)
    repaired = _build(states, cache)[0]
    assert repaired is not first
    assert algebra._context_geometry([repaired]) is not None
    reference = scop.build_scop_penalty_components(states)[0]
    for name in PenaltyComponent.__dataclass_fields__:
        actual, expected = getattr(repaired, name), getattr(reference, name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


@pytest.mark.parametrize(
    "change", ["matrix", "name", "slice", "reparam", "rank", "logdet", "eigenvalues"]
)
def test_changed_source_target_or_metadata_never_reuses_an_old_entry(change):
    states, cache = _states(width=5, groups=1), {}
    first = _build(states, cache)[0]
    source = states[0]
    if change == "matrix":
        source["S_scop"] *= 2
    elif change == "name":
        source["group_name"] = "new"
    elif change == "slice":
        source["group_sl"] = slice(2, 7)
    elif change == "reparam":
        source["reparam"] = object()
    elif change == "rank":
        source["penalty_rank"] += 1
    elif change == "logdet":
        source["penalty_log_det_omega_plus"] += 1
    else:
        source["penalty_eigvals_omega"] = source["penalty_eigvals_omega"] * 2
    second = _build(states, cache)[0]
    assert second is not first
    np.testing.assert_array_equal(second.omega_ssp, source["S_scop"])
    assert second.rank == 3


def test_failed_changed_target_leaves_the_previous_valid_entry_available():
    states, cache = _states(width=5, groups=1), {}
    first = _build(states, cache)[0]
    previous = dict(cache)
    old_matrix = states[0]["S_scop"].copy()
    states[0]["S_scop"][0, 1] = 1.0
    with pytest.raises((ValueError, support_module.PenaltyNumericalError)):
        _build(states, cache)
    assert cache == previous
    states[0]["S_scop"][:] = old_matrix
    assert _build(states, cache)[0] is first


def test_distinct_fit_contexts_do_not_share_mutable_penalty_owners():
    states = _states(width=5, groups=1)
    first = _build(states, {})[0]
    second = _build(states, {})[0]
    assert first is not second
    assert algebra._context_geometry([first]) is not algebra._context_geometry([second])


def test_replacing_a_fit_context_starts_a_new_penalty_cache():
    context = scop._SCOPREMLFitContext(
        dm=None,
        distribution=None,
        link=None,
        groups=[],
        y=np.empty(0),
        sample_weight=np.empty(0),
        offset_arr=np.empty(0),
        pirls_tol=1e-9,
        max_pirls_iter=10,
        reml_penalties=None,
        convergence="coefficients",
        scop_joint=True,
        debug_recorder=None,
        likelihood_size=0,
        weight_semantics="frequency",
        gamma_scale_data=None,
    )
    component = _build(_states(width=5, groups=1), context._penalty_context_cache)[0]
    copied = replace(context)
    assert context._penalty_context_cache
    assert copied._penalty_context_cache == {}
    assert _build(_states(width=5, groups=1), copied._penalty_context_cache)[0] is not component


def test_an_unresolved_new_zero_target_is_not_evaluated_or_published(monkeypatch):
    states, cache = _states(width=5, groups=1), {}

    def refusal(*_args, **_kwargs):
        raise support_module.PenaltyNumericalError("unit geometry accuracy refusal")

    monkeypatch.setattr(kernel, "_evaluate_penalty_summary", refusal)
    component = _build(states, cache, {"x0": 0.0})
    assert cache == {}
    evaluation = algebra._compute_penalty_logdet_evaluation({"x0": 0.0}, component)
    assert evaluation.rank == 0 and evaluation.logdet == 0
    with pytest.raises(
        support_module.PenaltyNumericalError, match="unit geometry accuracy refusal"
    ):
        _build(states, cache, {"x0": 1.0})
    assert cache == {}


def test_refused_changed_unit_geometry_does_not_replace_the_previous_entry(monkeypatch):
    states, cache = _states(width=5, groups=1), {}
    first = _build(states, cache)[0]
    previous = dict(cache)
    states[0]["S_scop"] *= 2

    def refusal(*_args, **_kwargs):
        raise support_module.PenaltyNumericalError("unit geometry accuracy refusal")

    monkeypatch.setattr(kernel, "_evaluate_penalty_summary", refusal)
    zero = _build(states, cache, {"x0": 0.0})[0]
    assert zero is not first and cache == previous
    with pytest.raises(
        support_module.PenaltyNumericalError, match="unit geometry accuracy refusal"
    ):
        _build(states, cache, {"x0": 1.0})
    assert cache == previous
    states[0]["S_scop"] /= 2
    assert _build(states, cache)[0] is first


def test_mixed_ordinary_family_keeps_its_existing_complete_owner():
    roots = [np.array([[1.0, 0.0]]), np.array([[1.0, 1.0]]), np.array([[0.0, 1.0]])]
    ordinary = [
        PenaltyComponent(
            name=f"ordinary:{i}",
            group_name="ordinary",
            group_index=2,
            group_sl=slice(10, 12),
            omega_raw=root.T @ root,
            omega_ssp=root.T @ root,
        )
        for i, root in enumerate(roots)
    ]
    algebra._attach_context_geometry(ordinary)
    owner = algebra._context_geometry(ordinary)
    components = scop._merge_scop_penalty_components(ordinary, _build(_states(width=5), {}))
    assert all(left is right for left, right in zip(components[:3], ordinary, strict=True))
    assert algebra._context_geometry(components[:3]) is owner
    weights = {"ordinary:0": 2.0, "ordinary:1": 3.0, "ordinary:2": 5.0, "x0": 2.0, "x1": 3.0}
    expected = algebra._compute_penalty_logdet_evaluation(weights, ordinary)
    actual = algebra._compute_penalty_logdet_evaluation(weights, components)
    assert actual.rank == expected.rank + 6
    for name in weights:
        if name.startswith("ordinary"):
            assert actual.gradient[name] == expected.gradient[name]
    for pair, value in expected.hessian.items():
        assert actual.hessian[pair] == value


def test_beta_and_hessian_dependent_efs_reductions_stay_fresh():
    states, cache = _states(width=5, groups=1), {}
    component = _build(states, cache)[0]
    results = []
    for scale in (1.0, 2.0):
        beta = scale * np.array([0.0, 0.0, 1.0, 0.0, -1.0])
        inverse = scale * 0.1 * np.eye(5)
        states[0]["beta_eff"] = beta
        assert _build(states, cache)[0] is component
        updated, _, _ = scop._joint_efs_lambda_step(
            [component],
            np.exp(beta),
            inverse,
            1.0,
            {"x0": 0.1},
            {"x0"},
            states,
            {"x0": 1.0},
            {},
        )
        quadratic = float(beta @ states[0]["S_scop"] @ beta)
        trace = float(np.trace(inverse @ states[0]["S_scop"]))
        expected = (3.0 - 0.1 * trace) / quadratic
        assert abs(updated["x0"] - expected) <= 16 * np.finfo(float).eps * max(1.0, expected)
        results.append(updated["x0"])
    assert results[0] != results[1]


def test_actual_mode_builders_share_one_latent_owner(monkeypatch):
    from .test_scop_efs import scop_model_inputs

    model, y, weights, offset = scop_model_inputs.__wrapped__()
    context = scop._SCOPREMLFitContext(
        dm=model._dm,
        distribution=model._distribution,
        link=model._link,
        groups=model._groups,
        y=y,
        sample_weight=np.asarray(weights),
        offset_arr=np.zeros_like(y) if offset is None else np.asarray(offset),
        pirls_tol=1e-6,
        max_pirls_iter=100,
        reml_penalties=None,
        convergence="coefficients",
        scop_joint=True,
        debug_recorder=None,
        likelihood_size=float(np.sum(weights)),
        weight_semantics="frequency",
        gamma_scale_data=None,
    )
    summaries = []
    original = kernel._evaluate_penalty_summary

    def counted(support, values, *args, **kwargs):
        summaries.append((support.Q_plus.shape, tuple(values)))
        return original(support, values, *args, **kwargs)

    monkeypatch.setattr(kernel, "_evaluate_penalty_summary", counted)
    first = scop._fit_scop_reml_mode(
        context,
        {"x": 1.0},
        beta_init=None,
        intercept_init=None,
        scop_state_init=None,
        phase="candidate",
        reml_iteration=0,
        require_converged=True,
    )
    assert first is not None
    second = scop._fit_scop_reml_mode(
        context,
        {"x": 2.0},
        beta_init=first.result.beta,
        intercept_init=first.result.intercept,
        scop_state_init=first.scop_states,
        phase="candidate",
        reml_iteration=1,
        require_converged=True,
    )
    assert second is not None
    fallback = scop._evaluate_scop_reml_mode(
        context,
        first.lambdas,
        result=first.result,
        xtwx=first.xtwx,
        centered_xtwx=first.centered_xtwx,
        fisher_mean_x=first.fisher_mean_x,
        fisher_sum_w=first.fisher_sum_w,
        scop_states=first.scop_states,
        mode_score=first.mode_score,
    )
    assert first.penalty_components[0] is second.penalty_components[0]
    assert fallback.penalty_components[0] is first.penalty_components[0]
    assert not np.array_equal(first.penalty, second.penalty)
    assert len(summaries) == 1 and summaries[0][1] == (1.0,)


def test_unused_inverse_overflow_does_not_block_a_finite_latent_summary():
    states = _states(width=2, groups=1)
    tiny = np.ldexp(1.0, -1060)
    states[0]["S_scop"] = tiny * np.eye(2)
    weight = np.ldexp(1.0, 1000)
    with pytest.raises(
        support_module.PenaltyNumericalError, match="required dense penalty inverse"
    ):
        kernel.similarity_transform_logdet([states[0]["S_scop"]], np.ones(1))
    weighted = kernel.similarity_transform_logdet([states[0]["S_scop"]], np.array([weight]))
    components = _build(states, {})
    result = algebra._compute_penalty_logdet_evaluation({"x0": weight}, components)
    assert result.rank == weighted.rank == 2
    assert abs(result.logdet - weighted.logdet_s_plus) <= (
        result.logdet_error + weighted._certificate.logdet_error
    )
    assert result.gradient == {"x0": 2.0}
    assert result.hessian == {("x0", "x0"): 0.0}
