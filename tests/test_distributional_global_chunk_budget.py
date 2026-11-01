"""Automatic row bounds use the assembler's exact owned-workspace estimate."""

import importlib

import numpy as np
import pytest

from tests.test_distributional_global_moments import make_fixture

api = importlib.import_module("superglm.distributional.solver._global_moments")


def estimate(layout, chunk_size):
    built = api.build_global_moment_plan(layout, chunk_size=chunk_size)
    assert built.plan is not None, built.reason
    try:
        assert built.estimated_peak_bytes == built.plan.stats["estimated_peak_bytes"]
        return built.estimated_peak_bytes
    finally:
        built.plan.close()


@pytest.mark.parametrize("boundary", [9, 17, 31])
@pytest.mark.parametrize("extra_byte", [-1, 0, 1])
def test_selector_chooses_largest_row_bound_proved_by_builder(boundary, extra_byte):
    fixture = make_fixture()
    budget = estimate(fixture.layout, boundary) + extra_byte
    selected = api.global_moment_chunk_size(
        fixture.layout, byte_budget=budget, minimum_chunk_size=8, maximum_chunk_size=32
    )
    assert selected == boundary - (extra_byte < 0)
    accepted = api.build_global_moment_plan(fixture.layout, chunk_size=selected, byte_budget=budget)
    assert accepted.plan is not None, accepted.reason
    try:
        assert accepted.plan.stats["estimated_peak_bytes"] <= budget
    finally:
        accepted.plan.close()
    refused = api.build_global_moment_plan(
        fixture.layout, chunk_size=selected + 1, byte_budget=budget
    )
    assert refused.plan is None
    assert refused.estimated_peak_bytes > budget


@pytest.mark.parametrize("maximum,expected", [(16, 16), (37, 37), (65536, 37)])
def test_selector_respects_maximum_and_observation_count(maximum, expected):
    fixture = make_fixture()
    assert (
        api.global_moment_chunk_size(
            fixture.layout, byte_budget=64 << 20, minimum_chunk_size=8, maximum_chunk_size=maximum
        )
        == expected
    )


@pytest.mark.parametrize("reason", ["budget", "source_shape", "source_type", "smaller_maximum"])
def test_selector_preserves_previous_minimum_on_refusal(reason):
    fixture = make_fixture()
    budget = 1 if reason == "budget" else 64 << 20
    maximum = 4 if reason == "smaller_maximum" else 32
    group = fixture.layout.predictors[0].design.group_matrices[2]
    if reason == "source_shape":
        group.bin_idx = group.bin_idx[:-1]
    elif reason == "source_type":

        class CustomArray(np.ndarray):
            pass

        group.B_unique = group.B_unique.view(CustomArray)
    assert (
        api.global_moment_chunk_size(
            fixture.layout, byte_budget=budget, minimum_chunk_size=8, maximum_chunk_size=maximum
        )
        == 8
    )


def test_selector_constructs_no_plan_and_does_not_scan_numeric_arrays(monkeypatch):
    fixture = make_fixture()

    def forbidden(*args, **kwargs):
        pytest.fail("metadata-only selector allocated a plan or scanned numeric source values")

    monkeypatch.setattr(api, "GlobalMomentPlan", forbidden)
    monkeypatch.setattr(api, "_small_finite", forbidden)
    monkeypatch.setattr(api, "_finite_bounded_2d", forbidden)
    monkeypatch.setattr(api.np, "empty", forbidden)
    assert (
        api.global_moment_chunk_size(
            fixture.layout, byte_budget=64 << 20, minimum_chunk_size=8, maximum_chunk_size=32
        )
        == 32
    )


def test_builder_revalidates_numeric_authority_after_metadata_selection():
    fixture = make_fixture()
    selected = api.global_moment_chunk_size(
        fixture.layout, byte_budget=64 << 20, minimum_chunk_size=8, maximum_chunk_size=32
    )
    fixture.layout.predictors[0].design.group_matrices[2].B_unique[0, 0] = np.nan
    refused = api.build_global_moment_plan(fixture.layout, chunk_size=selected)
    assert refused.plan is None
    assert "finite" in refused.reason


def test_default_upper_bound_is_65536_without_source_sized_allocations():
    fixture = make_fixture()
    n = 1_000_000
    # Broadcast views provide exact full-row metadata without a large fixture.
    for state in fixture.layout.predictors:
        state.design.n = n
        for group in state.design.group_matrices:
            group.shape = (n, group.shape[1])
            if type(group) is api.DenseGroupMatrix:
                group.M = np.broadcast_to(group.M[:1], group.shape)
            elif type(group) is api.CategoricalGroupMatrix:
                group.codes = np.broadcast_to(group.codes[:1], (n,))
            elif type(group) is api.DiscretizedSSPGroupMatrix:
                group.bin_idx = np.broadcast_to(group.bin_idx[:1], (n,))
            else:
                group.n_rows = n
    selected = api.global_moment_chunk_size(
        fixture.layout, byte_budget=64 << 20, minimum_chunk_size=8065
    )
    assert selected == 65536
    assert estimate(fixture.layout, selected) <= 64 << 20
