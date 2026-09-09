"""Direct source-range ingestion contracts, independent of child designs."""

import importlib
import weakref

import pytest

from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
)
from tests.test_distributional_global_moments import (
    _tripwire_array,
    accepted_plan,
    assert_geometry,
    make_fixture,
)

api = importlib.import_module("superglm.distributional.solver._global_moments")


def source_plans(fixture):
    return tuple(
        PredictorExecutionPlan(state.design, state.intercept_index is not None)
        for state in fixture.layout.predictors
    )


def accumulate_ranges(plan, fixture, supplied, chunk_size):
    plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
    for start in range(0, len(fixture.score), chunk_size):
        stop = min(start + chunk_size, len(fixture.score))
        plan.add_row_range(
            supplied, start, stop, fixture.score[start:stop], fixture.curvature[start:stop]
        )
    return plan.finish()


@pytest.mark.parametrize("chunk_size", [1, 8, 38])
@pytest.mark.parametrize("cancellation", [False, True])
@pytest.mark.parametrize("right_intercept", [False, True])
def test_ranges_match_signed_masked_stored_rows_without_child_construction(
    monkeypatch, chunk_size, cancellation, right_intercept
):
    fixture = make_fixture(cancellation=cancellation, right_intercept=right_intercept)
    supplied = source_plans(fixture)
    with accepted_plan(api, fixture, chunk_size) as plan:

        def forbidden(*args, **kwargs):
            pytest.fail("direct range ingestion constructed a child group/design/plan")

        for cls in (
            PredictorExecutionPlan,
            DesignMatrix,
            DenseGroupMatrix,
            CategoricalGroupMatrix,
            DiscretizedSSPGroupMatrix,
            DiscretizedSplineCategoricalGroupMatrix,
        ):
            monkeypatch.setattr(cls, "__init__", forbidden)
        result = accumulate_ranges(plan, fixture, supplied, chunk_size)
        assert_geometry(result, fixture)
        assert plan.stats["current_geometry_rows"] == len(fixture.score)
        assert plan.stats["estimated_peak_bytes"] <= 64 << 20


@pytest.mark.parametrize(
    "start,stop", [(1, 8), (0, 0), (-1, 7), (0, 9), (0, 38), (False, 8), (0, 8.0)]
)
def test_invalid_ranges_refuse_before_accumulation(start, stop):
    fixture = make_fixture()
    with accepted_plan(api, fixture) as plan:
        plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
        with pytest.raises(api.GlobalMomentRefusalError) as caught:
            plan.add_row_range(
                source_plans(fixture), start, stop, fixture.score[:8], fixture.curvature[:8]
            )
        assert not caught.value.recoverable
        assert plan.stats["chunks"] == 0
        assert plan.stats["state"] == "refused"


@pytest.mark.parametrize(
    "field", ["M", "codes", "bin_idx", "row_idx", "bin_idx_level", "B_unique", "R_inv"]
)
def test_range_source_subclasses_refuse_before_hooks(field):
    fixture = make_fixture()
    supplied = source_plans(fixture)
    with accepted_plan(api, fixture) as plan:
        plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
        group_index = {"M": 0, "codes": 1, "row_idx": 3, "bin_idx_level": 3}.get(field, 2)
        group = supplied[0].design.group_matrices[group_index]
        calls = []
        setattr(group, field, _tripwire_array(getattr(group, field), calls))
        with pytest.raises(api.GlobalMomentRefusalError) as caught:
            plan.add_row_range(supplied, 0, 8, fixture.score[:8], fixture.curvature[:8])
        assert not caught.value.recoverable
        assert not calls
        assert plan.stats["chunks"] == 0


@pytest.mark.parametrize("field", ["B_unique", "R_inv", "bin_idx", "source_shape"])
def test_range_later_hard_source_error_precedes_numeric_refusal(field):
    fixture = make_fixture()
    supplied = source_plans(fixture)
    with accepted_plan(api, fixture) as plan:
        plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
        plan.add_row_range(supplied, 0, 8, fixture.score[:8], fixture.curvature[:8])
        fixture.score[8, 0] = 2.0**129
        group = supplied[1].design.group_matrices[-1]
        if field == "source_shape":
            group.bin_idx = group.bin_idx[:-1]
        elif field == "bin_idx":
            group.bin_idx[9] = group.n_bins
        else:
            getattr(group, field)[0, 0] += 0.125
        with pytest.raises(api.GlobalMomentRefusalError) as caught:
            plan.add_row_range(supplied, 8, 16, fixture.score[8:16], fixture.curvature[8:16])
        assert not caught.value.recoverable
        assert plan.stats["chunks"] == 1
        with pytest.raises(api.GlobalMomentRefusalError):
            plan.finish()


def test_range_reads_live_bins_and_ordinary_rows_after_reset():
    fixture = make_fixture()
    supplied = source_plans(fixture)
    with accepted_plan(api, fixture) as plan:
        assert_geometry(accumulate_ranges(plan, fixture, supplied, 8), fixture)
        for state in fixture.layout.predictors:
            for group in state.design.group_matrices:
                if type(group) is DenseGroupMatrix:
                    group.M[:] *= -0.5
                elif type(group) is CategoricalGroupMatrix:
                    group.codes[:] = (group.codes + 1) % (group.n_levels + 1)
                elif type(group) is DiscretizedSSPGroupMatrix:
                    group.bin_idx[:] = (group.bin_idx + 1) % group.n_bins
                else:
                    group.bin_idx_level[:] = (group.bin_idx_level + 1) % group.n_bins
        assert_geometry(accumulate_ranges(plan, fixture, supplied, 8), fixture)
        assert plan.stats["reset_count"] == 2
        plan.close()
        assert plan.stats["current_owned_bytes"] == 0
        with pytest.raises(api.GlobalMomentRefusalError):
            plan.add_row_range(supplied, 0, 8, fixture.score[:8], fixture.curvature[:8])
        assert plan.stats["state"] == "closed"


@pytest.mark.parametrize("hard_error", [False, True])
@pytest.mark.parametrize("lookup", ["_row_order", "_sorted_rows"])
def test_uncertified_lookup_defers_replay_refusal_until_later_sources_checked(
    monkeypatch, hard_error, lookup
):
    fixture = make_fixture()
    supplied = source_plans(fixture)
    with accepted_plan(api, fixture) as plan:
        plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
        plan.add_row_range(supplied, 0, 8, fixture.score[:8], fixture.curvature[:8])
        group = supplied[0].design.group_matrices[3]
        # Even equal replacement lookup contents have no immutable certificate.
        setattr(group, lookup, getattr(group, lookup).copy())
        if hard_error:
            supplied[1].design.group_matrices[-1].bin_idx[9] = 999

        def forbidden(*args, **kwargs):
            pytest.fail("uncertified lookup constructed an unaccounted fallback child")

        monkeypatch.setattr(DiscretizedSplineCategoricalGroupMatrix, "row_subset", forbidden)
        with pytest.raises(api.GlobalMomentRefusalError) as caught:
            plan.add_row_range(supplied, 8, 16, fixture.score[8:16], fixture.curvature[8:16])
        assert caught.value.recoverable is not hard_error
        assert plan.stats["chunks"] == 1
        assert plan.stats["current_geometry_rows"] == 8
        assert plan.stats["refusal_count"] == 1
        plan.close()
        assert plan.stats["current_owned_bytes"] == 0


def test_range_validation_only_scans_requested_rows(monkeypatch):
    fixture = make_fixture()
    supplied = source_plans(fixture)
    with accepted_plan(api, fixture) as plan:
        original = api._index_values
        inspected = []

        def bounded(values, count, upper, label):
            inspected.append(count)
            assert count <= 8, "range validation scanned a full source row map"
            return original(values, count, upper, label)

        monkeypatch.setattr(api, "_index_values", bounded)
        assert_geometry(accumulate_ranges(plan, fixture, supplied, 8), fixture)
        assert inspected


@pytest.mark.parametrize("field", ["M", "bin_idx"])
def test_outside_range_numeric_or_index_failure_is_checked_when_ingested(field):
    fixture = make_fixture()
    supplied = source_plans(fixture)
    with accepted_plan(api, fixture) as plan:
        plan.reset(coefficients=fixture.coefficients, penalty=fixture.penalty)
        group = supplied[0].design.group_matrices[0 if field == "M" else 2]
        if field == "M":
            group.M[-1, 0] = 2.0**129
        else:
            group.bin_idx[-1] = 999
        for start in range(0, 32, 8):
            stop = start + 8
            plan.add_row_range(
                supplied, start, stop, fixture.score[start:stop], fixture.curvature[start:stop]
            )
        with pytest.raises(api.GlobalMomentRefusalError) as caught:
            plan.add_row_range(supplied, 32, 37, fixture.score[32:], fixture.curvature[32:])
        assert caught.value.recoverable == (field == "M")
        assert plan.stats["chunks"] == 4


def test_range_reads_rebuilt_live_activity_mapping():
    fixture = make_fixture()
    supplied = source_plans(fixture)
    with accepted_plan(api, fixture) as plan:
        assert_geometry(accumulate_ranges(plan, fixture, supplied, 8), fixture)
        group = supplied[0].design.group_matrices[3]
        group.row_idx = group.row_idx[::-1].copy()
        group._sorted_rows = None
        group._row_order = None
        group._row_lookup_certificate = None
        assert_geometry(accumulate_ranges(plan, fixture, supplied, 8), fixture)


def test_range_category_temporaries_release_before_next_extraction(monkeypatch):
    fixture = make_fixture()
    supplied = source_plans(fixture)
    original = api._category_rows
    previous = []

    def bounded(group, start, stop):
        assert all(reference() is None for reference in previous), (
            "previous category's row-sized temporaries exceed the validation scratch budget"
        )
        result = original(group, start, stop)
        previous[:] = [weakref.ref(value) for value in result]
        return result

    monkeypatch.setattr(api, "_category_rows", bounded)
    with accepted_plan(api, fixture) as plan:
        assert_geometry(accumulate_ranges(plan, fixture, supplied, 8), fixture)
