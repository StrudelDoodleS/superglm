"""Independent stored-design contracts for streamed global moments.

Numerical references use literal stored rows and absolute raw-product bounds;
no grouped algebra is used to construct expected values.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import pytest

from superglm.distributional.layout import PredictorState, StackedLayout
from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
)
from superglm.links import IdentityLink
from superglm.types import GroupSlice

_MODULE = "superglm.distributional.solver._global_moments"
_FIELDS = ("score_data", "score_penalized", "data_curvature", "penalty", "penalized_curvature")


@pytest.fixture
def api():
    return importlib.import_module(_MODULE)


@dataclass
class Fixture:
    layout: StackedLayout
    coefficients: np.ndarray
    penalty: np.ndarray
    score: np.ndarray
    curvature: np.ndarray
    # Full-row bins for intentionally removing an activity mask in a mutation.
    full_bins: dict[tuple[int, int], np.ndarray]


def make_fixture(*, right_intercept=True, cancellation=False):
    """Small, distinct rectangular layouts; every buffer is public synthetic."""
    rng = np.random.default_rng(739204)
    n = 38 if cancellation else 37
    seed_rows = n // 2 if cancellation else n
    row = np.arange(seed_rows, dtype=np.intp)
    full_bins = {}

    def repeat(values):
        return np.repeat(values, 2, axis=0) if cancellation else values

    def support(bins, raw, width, stride, shift):
        basis = rng.uniform(-0.8, 1.1, (bins, raw))
        natural = rng.uniform(-0.7, 0.9, (raw, width))
        # Keep stored support unused at the final support point.
        indices = repeat((stride * row + shift) % (bins - 1))
        return basis, natural, indices

    def masked(predictor, group_index, bins, raw, width, modulus, empty=False):
        basis, natural, indices = support(bins, raw, width, 3, predictor + 1)
        activity = repeat((row % modulus) == 1)
        rows = np.flatnonzero(activity) if not empty else np.empty(0, dtype=np.intp)
        full_bins[predictor, group_index] = indices.copy()
        return DiscretizedSplineCategoricalGroupMatrix(basis, natural, indices, rows, n_rows=n)

    left = [
        DenseGroupMatrix(repeat(rng.uniform(-1, 1, (seed_rows, 2)))),
        CategoricalGroupMatrix(repeat(np.array([-1, 0, 1])[row % 3]), 3),
        DiscretizedSSPGroupMatrix(*support(7, 4, 3, 2, 0)),
        masked(0, 3, 5, 3, 2, 3),
        masked(0, 4, 4, 2, 1, 2, empty=True),
    ]
    right = [
        CategoricalGroupMatrix(repeat(np.array([-1, 0, 1])[(2 * row + 1) % 3]), 2),
        masked(1, 1, 6, 4, 3, 4),
        # Noncontiguous ordinary data is a separate rendering contract.
        DenseGroupMatrix(repeat(rng.uniform(-1, 1, (seed_rows, 2)))[:, ::2]),
        DiscretizedSSPGroupMatrix(*support(9, 3, 2, 5, 1)),
    ]
    states, names, terms = [], [], {}
    start = 0
    for a, (matrices, intercept) in enumerate(((left, True), (right, right_intercept))):
        groups, local_start = [], 0
        label = f"parameter_{a}"
        if intercept:
            names.append(f"{label}:(intercept)")
        for g, matrix in enumerate(matrices):
            width = matrix.shape[1]
            name = f"group_{g}"
            groups.append(GroupSlice(name, local_start, local_start + width, penalized=False))
            terms[f"{label}:{name}"] = slice(
                start + int(intercept) + local_start,
                start + int(intercept) + local_start + width,
            )
            names.extend(f"{label}:{name}[{j}]" for j in range(width))
            local_start += width
        stop = start + int(intercept) + local_start
        states.append(
            PredictorState(
                name=label,
                parameter_index=a,
                link=IdentityLink(),
                design=DesignMatrix(matrices, n, local_start),
                groups=tuple(groups),
                coefficient_slice=slice(start, stop),
                intercept_index=start if intercept else None,
                offset=np.zeros(n),
                penalties=(),
            )
        )
        start = stop
    layout = StackedLayout(tuple(states), start, tuple(names), terms, ())
    coefficients = rng.uniform(-0.2, 0.2, start)
    penalty = np.diag(np.linspace(0.01, 0.1, start))
    score = rng.uniform(-1.3, 0.8, (n, 2))
    curvature = rng.uniform(-1.1, 1.4, (n, 3))
    curvature[::5] = 0.0
    score[::7] = 0.0
    if cancellation:
        # Identical row pairs with opposite channels. Assess absolute backward
        # error, never relative error in the almost-zero final answer.
        score[1::2] = -score[::2]
        curvature[1::2] = -curvature[::2]
        score[-1] += np.finfo(np.float64).eps
        curvature[-1] += np.finfo(np.float64).eps
    return Fixture(layout, coefficients, penalty, score, curvature, full_bins)


def stored_matrices(fixture, *, remove_masks=False):
    """Materialize stored rows independently, with absolute-product envelopes."""
    matrices, envelopes = [], []
    for a, state in enumerate(fixture.layout.predictors):
        blocks, bounds = [], []
        if state.intercept_index is not None:
            blocks.append(np.ones((state.design.n, 1)))
            bounds.append(np.ones((state.design.n, 1)))
        for g, group in enumerate(state.design.group_matrices):
            if type(group) is DenseGroupMatrix:
                block = group.M.copy()
                bound = np.abs(block)
            elif type(group) is CategoricalGroupMatrix:
                block = np.zeros(group.shape)
                for row, code in enumerate(group.codes):
                    if 0 <= code < group.n_levels:
                        block[row, code] = 1.0
                bound = block.copy()
            elif type(group) is DiscretizedSSPGroupMatrix:
                block = group.B_unique[group.bin_idx] @ group.R_inv
                bound = np.abs(group.B_unique[group.bin_idx]) @ np.abs(group.R_inv)
            elif type(group) is DiscretizedSplineCategoricalGroupMatrix:
                block = np.zeros(group.shape)
                bound = np.zeros(group.shape)
                rows = np.arange(group.shape[0]) if remove_masks else group.row_idx
                bins = fixture.full_bins[a, g] if remove_masks else group.bin_idx_level
                block[rows] = group.B_unique[bins] @ group.R_inv
                bound[rows] = np.abs(group.B_unique[bins]) @ np.abs(group.R_inv)
            else:
                raise AssertionError(f"oracle has no rule for {type(group).__name__}")
            blocks.append(block)
            bounds.append(bound)
        matrices.append(np.column_stack(blocks))
        envelopes.append(np.column_stack(bounds))
    return tuple(matrices), tuple(envelopes)


def row_reference(fixture, *, remove_masks=False, unsigned=False):
    matrices, envelopes = stored_matrices(fixture, remove_masks=remove_masks)
    p = fixture.layout.n_coefficients
    score, score_scale = np.zeros(p), np.zeros(p)
    curvature, curvature_scale = np.zeros((p, p)), np.zeros((p, p))
    channel = 0
    for a, left in enumerate(matrices):
        left_slice = fixture.layout.predictors[a].coefficient_slice
        score[left_slice] = left.T @ fixture.score[:, a]
        score_scale[left_slice] = envelopes[a].T @ np.abs(fixture.score[:, a])
        for b in range(a, len(matrices)):
            right_slice = fixture.layout.predictors[b].coefficient_slice
            weights = fixture.curvature[:, channel]
            if unsigned:
                weights = np.abs(weights)
            value = left.T @ (weights[:, None] * matrices[b])
            bound = envelopes[a].T @ (np.abs(weights)[:, None] * envelopes[b])
            curvature[left_slice, right_slice] = value
            curvature_scale[left_slice, right_slice] = bound
            if a != b:
                curvature[right_slice, left_slice] = value.T
                curvature_scale[right_slice, left_slice] = bound.T
            channel += 1
    penalty_score = fixture.penalty @ fixture.coefficients
    reference = dict(
        score_data=score,
        score_penalized=score - penalty_score,
        data_curvature=curvature,
        penalty=fixture.penalty.copy(),
        penalized_curvature=curvature + fixture.penalty,
    )
    scales = dict(
        score_data=score_scale,
        score_penalized=score_scale + np.abs(fixture.penalty) @ np.abs(fixture.coefficients),
        data_curvature=curvature_scale,
        penalty=np.abs(fixture.penalty),
        penalized_curvature=curvature_scale + np.abs(fixture.penalty),
    )
    return reference, scales


def assert_geometry(actual, fixture):
    reference, scales = row_reference(fixture)
    # Bound includes row reductions, raw-basis and map contractions, ordinary
    # blocks, and final assembly.  Use absolute-product norms even under severe
    # cancellation in either maps or signed channels.
    dimension = (
        fixture.score.shape[0]
        + fixture.layout.n_coefficients
        + sum(
            group.B_unique.shape[1] + group.B_unique.shape[0]
            for state in fixture.layout.predictors
            for group in state.design.group_matrices
            if hasattr(group, "B_unique")
        )
    )
    epsilon = np.finfo(np.float64).eps
    gamma = dimension * epsilon / (1 - dimension * epsilon)
    errors = {}
    for field, expected in reference.items():
        obtained = actual[field] if isinstance(actual, dict) else getattr(actual, field)
        assert obtained.shape == expected.shape, field
        assert np.all(np.isfinite(obtained)), field
        error = float(np.linalg.norm(obtained - expected))
        scale = float(np.linalg.norm(scales[field]))
        bound = 32 * gamma * scale
        assert error <= bound, (field, error, bound)
        errors[field] = dict(error_norm=error, bound=bound, fraction=error / bound if bound else 0)
    return errors


def chunks(fixture, chunk_size):
    for start in range(0, fixture.score.shape[0], chunk_size):
        stop = min(start + chunk_size, fixture.score.shape[0])
        rows = np.arange(start, stop, dtype=np.intp)
        plans = []
        for state in fixture.layout.predictors:
            matrices = [group.row_subset(rows) for group in state.design.group_matrices]
            design = DesignMatrix(matrices, len(rows), state.design.p)
            plans.append(PredictorExecutionPlan(design, state.intercept_index is not None))
        yield tuple(plans), fixture.score[start:stop], fixture.curvature[start:stop]


def accumulate(plan, fixture, chunk_size):
    plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
    for plans, score, curvature in chunks(fixture, chunk_size):
        plan.add_chunk(plans, score, curvature)
    return plan.finish()


@contextmanager
def accepted_plan(api, fixture, chunk_size=8, **kwargs):
    build = api.build_global_moment_plan(fixture.layout, chunk_size=chunk_size, **kwargs)
    assert build.plan is not None, build.reason
    try:
        yield build.plan
    finally:
        build.plan.close()


@pytest.mark.parametrize("right_intercept", [False, True])
@pytest.mark.parametrize("cancellation", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 8, 38])
def test_signed_rectangular_geometry_matches_stored_rows(
    api, right_intercept, cancellation, chunk_size
):
    fixture = make_fixture(right_intercept=right_intercept, cancellation=cancellation)
    with accepted_plan(api, fixture, chunk_size) as plan:
        result = accumulate(plan, fixture, chunk_size)
    assert_geometry(result, fixture)
    np.testing.assert_array_equal(result.data_curvature, result.data_curvature.T)
    for field in _FIELDS:
        assert not getattr(result, field).flags.writeable


def test_zero_channels_preserve_only_the_penalty(api):
    fixture = make_fixture()
    fixture.score.fill(0)
    fixture.curvature.fill(0)
    with accepted_plan(api, fixture) as plan:
        result = accumulate(plan, fixture, 8)
    assert_geometry(result, fixture)
    np.testing.assert_array_equal(result.score_data, np.zeros(fixture.layout.n_coefficients))
    np.testing.assert_array_equal(result.data_curvature, np.zeros_like(fixture.penalty))


def test_reset_clears_all_moments_without_mutating_published_geometry(api):
    fixture = make_fixture()
    with accepted_plan(api, fixture) as plan:
        first = accumulate(plan, fixture, 8)
        saved = {field: getattr(first, field).copy() for field in _FIELDS}
        second_fixture = replace(
            fixture,
            score=-0.75 * fixture.score,
            curvature=0.25 * fixture.curvature,
            coefficients=-fixture.coefficients,
            penalty=2 * fixture.penalty,
        )
        second = accumulate(plan, second_fixture, 8)
        assert_geometry(second, second_fixture)
        assert plan.stats["reset_count"] == 2
    for field, expected in saved.items():
        np.testing.assert_array_equal(getattr(first, field), expected)


@pytest.mark.parametrize("mutation", ["unsigned", "unmasked"])
def test_independent_oracle_detects_wrong_signed_or_masked_accumulation(api, mutation):
    correct = make_fixture()
    mutant = make_fixture()
    if mutation == "unsigned":
        mutant.curvature[:] = np.abs(mutant.curvature)
    else:
        states = []
        for a, state in enumerate(mutant.layout.predictors):
            matrices = []
            for g, group in enumerate(state.design.group_matrices):
                if type(group) is DiscretizedSplineCategoricalGroupMatrix:
                    group = DiscretizedSplineCategoricalGroupMatrix(
                        group.B_unique,
                        group.R_inv,
                        mutant.full_bins[a, g],
                        np.arange(state.design.n),
                        n_rows=state.design.n,
                    )
                matrices.append(group)
            states.append(
                replace(state, design=DesignMatrix(matrices, state.design.n, state.design.p))
            )
        mutant = replace(mutant, layout=replace(mutant.layout, predictors=tuple(states)))
    with accepted_plan(api, mutant) as plan:
        result = accumulate(plan, mutant, 8)
    with pytest.raises(AssertionError):
        assert_geometry(result, correct)


def test_global_dispatch_does_not_materialize_or_use_grouped_moments(api, monkeypatch):
    fixture = make_fixture()

    def forbidden(*args, **kwargs):
        pytest.fail("global moments fell back to row materialization or grouped products")

    for cls in (
        DenseGroupMatrix,
        CategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
    ):
        monkeypatch.setattr(cls, "toarray", forbidden)
    monkeypatch.setattr(PredictorExecutionPlan, "diagonal_moment", forbidden)
    monkeypatch.setattr(PredictorExecutionPlan, "cross_moment", forbidden)
    with accepted_plan(api, fixture) as plan:
        accumulate(plan, fixture, 8)
        assert plan.stats["chunks"] == 5
        assert plan.stats["rows"] == 37
        assert plan.stats["histogram_update_calls"] > 0
        assert plan.stats["directional_update_calls"] > 0
        assert plan.stats["ordinary_curvature_products"] > 0


def test_budget_accounts_for_owned_solver_support_and_chunk_scratch(api):
    fixture = make_fixture()
    estimates = []
    for chunk_size in (1, 37):
        build = api.build_global_moment_plan(fixture.layout, chunk_size=chunk_size)
        assert build.plan is not None, build.reason
        try:
            stats = build.plan.stats
            support_bytes = sum(
                group.B_unique.shape[0] * group.R_inv.shape[1] * 8
                for state in fixture.layout.predictors
                for group in state.design.group_matrices
                if hasattr(group, "B_unique")
            )
            assert stats["solver_support_bytes"] == support_bytes
            assert stats["persistent_allocated_bytes"] >= (
                stats["accumulator_bytes"] + stats["support_authority_bytes"] + support_bytes
            )
            assert stats["persistent_allocated_bytes"] <= build.estimated_peak_bytes
            estimates.append(build.estimated_peak_bytes)
        finally:
            build.plan.close()
    assert estimates[1] > estimates[0]
    accepted = api.build_global_moment_plan(fixture.layout, chunk_size=37, byte_budget=estimates[1])
    assert accepted.plan is not None, accepted.reason
    accepted.plan.close()
    for budget in (1, estimates[1] - 1):
        refused = api.build_global_moment_plan(fixture.layout, chunk_size=37, byte_budget=budget)
        assert refused.plan is None and refused.reason


@pytest.mark.parametrize("argument", ["byte_budget", "chunk_size"])
def test_boolean_configuration_refuses(api, argument):
    options = {"chunk_size": 8, argument: True}
    refused = api.build_global_moment_plan(make_fixture().layout, **options)
    assert refused.plan is None and refused.reason


def test_explicit_chunk_size_is_required(api):
    with pytest.raises(TypeError, match="chunk_size"):
        api.build_global_moment_plan(make_fixture().layout)


def test_tiny_raw_basis_rescaled_to_moderate_solver_support_is_admitted(api):
    fixture = make_fixture()
    group = fixture.layout.predictors[0].design.group_matrices[2]
    group.B_unique.fill(2.0**-400)
    group.R_inv.fill(2.0**400)
    with accepted_plan(api, fixture) as plan:
        result = accumulate(plan, fixture, 8)
    assert_geometry(result, fixture)


def test_nonzero_transformed_support_below_domain_refuses_underflow_route(api):
    fixture = make_fixture()
    group = fixture.layout.predictors[0].design.group_matrices[2]
    group.B_unique.fill(2.0**-700)
    group.R_inv.fill(2.0**128)
    fixture.curvature.fill(2.0**128)
    refused = api.build_global_moment_plan(fixture.layout, chunk_size=8)
    assert refused.plan is None and refused.reason
    # This establishes dispatch refusal only. It does not claim the inherited
    # fallback's B.T @ diag(w) @ B association solves extreme underflow.


def test_signed_cancellation_does_not_apply_input_cutoff_to_moments(api):
    fixture = make_fixture(cancellation=True)
    fixture.score.fill(0)
    fixture.curvature.fill(0)
    large, residual = 2.0**-100, 2.0**-150
    fixture.score[0] = large
    fixture.score[1] = -large + residual
    fixture.curvature[0] = large
    fixture.curvature[1] = -large + residual
    with accepted_plan(api, fixture) as plan:
        result = accumulate(plan, fixture, 8)
    assert_geometry(result, fixture)
    # Only two nonzero values multiply literal ones. Their binary sum is exact
    # in any reduction order; residual is defined input signal, not roundoff.
    intercept = fixture.layout.predictors[0].intercept_index
    assert result.score_data[intercept] == residual
    assert result.data_curvature[intercept, intercept] == residual


@pytest.mark.parametrize("field", ["B_unique", "R_inv"])
def test_live_support_authority_change_discards_partial_state(api, field):
    fixture = make_fixture()
    with accepted_plan(api, fixture) as plan:
        plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
        stream = iter(chunks(fixture, 8))
        plan.add_chunk(*next(stream))
        group = fixture.layout.predictors[0].design.group_matrices[2]
        source = getattr(group, field)
        original = source.copy()
        source[0, 0] += 0.125
        with pytest.raises(api.GlobalMomentRefusalError) as caught:
            plan.add_chunk(*next(stream))
        assert not caught.value.recoverable
        with pytest.raises(api.GlobalMomentRefusalError):
            plan.finish()
        source[:] = original
        assert_geometry(accumulate(plan, fixture, 8), fixture)


@pytest.mark.parametrize("failure", ["out_of_domain", "bad_index", "oversized_chunk"])
def test_midstream_refusal_invalidates_geometry_and_reset_recovers(api, failure):
    fixture = make_fixture()
    with accepted_plan(api, fixture) as plan:
        plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
        stream = iter(chunks(fixture, 8))
        plan.add_chunk(*next(stream))
        supplied, score, curvature = next(stream)
        if failure == "out_of_domain":
            curvature = curvature.copy()
            curvature[-1, -1] = 2.0**129
        elif failure == "bad_index":
            supplied[0].design.group_matrices[2].bin_idx[-1] = 999
        else:
            supplied, score, curvature = next(chunks(fixture, 9))
        with pytest.raises(api.GlobalMomentRefusalError) as caught:
            plan.add_chunk(supplied, score, curvature)
        assert caught.value.recoverable == (failure == "out_of_domain")
        with pytest.raises(api.GlobalMomentRefusalError):
            plan.finish()
        assert_geometry(accumulate(plan, fixture, 8), fixture)


def test_close_releases_state_and_cannot_be_revived(api):
    fixture = make_fixture()
    build = api.build_global_moment_plan(fixture.layout, chunk_size=8)
    assert build.plan is not None
    plan = build.plan
    plan.close()
    for _ in range(2):
        with pytest.raises(api.GlobalMomentRefusalError):
            plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
        assert plan.stats["state"] == "closed"


def _tripwire_array(values, calls):
    class ConcealedExtrema(np.ndarray):
        def min(self, *args, **kwargs):
            calls.append("min")
            return 0

        def max(self, *args, **kwargs):
            calls.append("max")
            return 0

        def __array__(self, *args, **kwargs):
            calls.append("array")
            raise AssertionError("custom array coercion ran before exact-type refusal")

        def __array_function__(self, *args, **kwargs):
            calls.append("array_function")
            raise AssertionError("custom NumPy function ran before exact-type refusal")

        def __array_ufunc__(self, *args, **kwargs):
            calls.append("array_ufunc")
            raise AssertionError("custom ufunc ran before exact-type refusal")

        def __getitem__(self, key):
            calls.append("getitem")
            raise AssertionError("custom indexing ran before exact-type refusal")

        def copy(self, *args, **kwargs):
            calls.append("copy")
            raise AssertionError("custom copy ran before exact-type refusal")

    return values.view(ConcealedExtrema)


@pytest.mark.parametrize("field", ["B_unique", "R_inv", "M"])
def test_borrowed_source_array_subclass_refuses_before_hooks(api, field):
    fixture = make_fixture()
    group = fixture.layout.predictors[0].design.group_matrices[0 if field == "M" else 2]
    calls = []
    setattr(group, field, _tripwire_array(getattr(group, field), calls))
    result = api.build_global_moment_plan(fixture.layout, chunk_size=8)
    assert result.plan is None and result.reason
    assert calls == []


@pytest.mark.parametrize(
    "field",
    [
        "bin_idx",
        "codes",
        "bin_idx_level",
        "row_idx",
        "B_unique",
        "R_inv",
        "M",
        "score",
        "curvature",
    ],
)
def test_live_array_subclass_refuses_before_hooks_or_native_updates(api, monkeypatch, field):
    fixture = make_fixture()
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("unchecked native writer ran for a malicious array subclass")

    with accepted_plan(api, fixture) as plan:
        plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
        supplied, score, curvature = next(chunks(fixture, 8))
        if field == "score":
            score = _tripwire_array(score, calls)
        elif field == "curvature":
            curvature = _tripwire_array(curvature, calls)
        else:
            index = {"M": 0, "codes": 1, "bin_idx_level": 3, "row_idx": 3}.get(field, 2)
            group = supplied[0].design.group_matrices[index]
            values = getattr(group, field).copy()
            if field in ("bin_idx", "codes", "bin_idx_level", "row_idx"):
                values[-1] = 999  # lying min/max must never admit this native index
            setattr(group, field, _tripwire_array(values, calls))
        # Guard tests intentionally stop native entry, making a removed guard
        # fail safely rather than exposing the test runner to invalid indices.
        for name in ("_accumulate_vector", "_accumulate_histogram", "_accumulate_directional"):
            monkeypatch.setattr(api, name, forbidden)
        if field == "codes":
            monkeypatch.setattr(api, "_pack_categorical", forbidden)
        with pytest.raises(api.GlobalMomentRefusalError) as caught:
            plan.add_chunk(supplied, score, curvature)
        assert not caught.value.recoverable
        assert calls == []
        assert plan.stats["chunks"] == 0
        with pytest.raises(api.GlobalMomentRefusalError):
            plan.finish()


@pytest.mark.parametrize("field", ["B_unique", "R_inv"])
def test_nonfinite_raw_source_refuses_construction(api, field):
    fixture = make_fixture()
    group = fixture.layout.predictors[0].design.group_matrices[2]
    getattr(group, field)[-1, -1] = np.nan
    build = api.build_global_moment_plan(fixture.layout, chunk_size=8)
    assert build.plan is None and build.reason


def test_group_metadata_cap_applies_to_zero_width_groups(api):
    fixture = make_fixture()
    state = fixture.layout.predictors[0]
    matrices, groups = list(state.design.group_matrices), list(state.groups)
    for index in range(65):
        matrices.append(DenseGroupMatrix(np.empty((state.design.n, 0))))
        groups.append(GroupSlice(f"empty_{index}", state.design.p, state.design.p))
    crowded = replace(
        state, design=DesignMatrix(matrices, state.design.n, state.design.p), groups=tuple(groups)
    )
    layout = replace(fixture.layout, predictors=(crowded, fixture.layout.predictors[1]))
    build = api.build_global_moment_plan(layout, chunk_size=8)
    assert build.plan is None and build.reason


@pytest.mark.parametrize("combination", ["channel_and_source", "ordinary_and_later_index"])
def test_structural_failure_takes_precedence_over_same_chunk_domain_refusal(
    api, monkeypatch, combination
):
    fixture = make_fixture()

    def forbidden(*args, **kwargs):
        pytest.fail("combined invalid chunk reached native moment accumulation")

    with accepted_plan(api, fixture) as plan:
        plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
        stream = iter(chunks(fixture, 8))
        plan.add_chunk(*next(stream))
        supplied, score, curvature = next(stream)
        if combination == "channel_and_source":
            curvature = curvature.copy()
            curvature[0, 0] = 2.0**129
            support = supplied[1].design.group_matrices[3]
            support.B_unique = support.B_unique.copy()
            support.B_unique[0, 0] += 0.125
        else:
            supplied[0].design.group_matrices[0].M[0, 0] = 2.0**129
            supplied[1].design.group_matrices[3].bin_idx[-1] = 999
        monkeypatch.setattr(api, "_accumulate_vector", forbidden)
        with pytest.raises(api.GlobalMomentRefusalError) as caught:
            plan.add_chunk(supplied, score, curvature)
        assert not caught.value.recoverable, (
            "a domain refusal must not hide a later change to the geometry target"
        )
        assert plan.stats["chunks"] == 1
        with pytest.raises(api.GlobalMomentRefusalError):
            plan.finish()


def test_live_activity_length_refuses_before_unbounded_comparison(api, monkeypatch):
    fixture = make_fixture()
    with accepted_plan(api, fixture) as plan:
        plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
        supplied, score, curvature = next(chunks(fixture, 8))
        n = len(score)
        support = supplied[0].design.group_matrices[3]
        # Exact ndarrays and individually valid indices: only the impossible
        # activity length reveals the structural error before sortedness work.
        support.row_idx = np.zeros(4 * n + 1, dtype=np.intp)
        support.bin_idx_level = np.zeros(4 * n + 1, dtype=np.intp)
        original_any = np.any

        def bounded_any(values, *args, **kwargs):
            if (
                type(values) is np.ndarray
                and values.ndim == 1
                and values.dtype == np.bool_
                and values.size > n
            ):
                pytest.fail("activity comparison allocated beyond the chunk row bound")
            return original_any(values, *args, **kwargs)

        def forbidden(*args, **kwargs):
            pytest.fail("oversized activity vector reached native moment accumulation")

        # Observe the comparison's temporary size rather than process RSS. The
        # old rows[1:] <= rows[:-1] expression exposes an oversized bool vector.
        monkeypatch.setattr(api.np, "any", bounded_any)
        monkeypatch.setattr(api, "_accumulate_vector", forbidden)
        with pytest.raises(api.GlobalMomentRefusalError) as caught:
            plan.add_chunk(supplied, score, curvature)
        assert not caught.value.recoverable
        assert "activity" in caught.value.reason
        assert any(word in caught.value.reason for word in ("bound", "length", "vector"))
        assert plan.stats["chunks"] == 0
