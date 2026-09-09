"""Admitted global likelihood streams consume original plans and bounded rows."""

import weakref

import numpy as np
import pytest

from superglm.distributional.solver import chunks
from superglm.distributional.solver._global_moments import GlobalMomentRefusalError

from .test_distributional_support_predictions import _support_problem


@pytest.mark.parametrize("category", [False, True])
def test_admitted_global_stream_does_not_construct_child_geometry(monkeypatch, category):
    problem = _support_problem(category)
    layout = problem[1]
    p = layout.n_coefficients
    options = dict(penalty=np.eye(p), chunk_size=7, curvature_source="observed")
    baseline = chunks.assemble_chunked_geometry(
        *problem, small_group_panel_byte_budget=None, **options
    )
    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: 64 << 20)
    original = chunks.PredictorExecutionPlan
    plans = []

    def counted(design, intercept):
        plans.append(design)
        return original(design, intercept)

    def forbidden(*args, **kwargs):
        pytest.fail("admitted global stream constructed child geometry")

    monkeypatch.setattr(chunks, "PredictorExecutionPlan", counted)
    monkeypatch.setattr(chunks, "group_row_range", forbidden)
    monkeypatch.setattr(chunks, "GroupedGeometryAccumulator", forbidden)
    result = chunks.assemble_chunked_geometry(*problem, **options)
    assert plans == [state.design for state in layout.predictors]
    tolerance = 64 * p * np.finfo(float).eps
    for field in ("score_data", "data_curvature", "score_penalized", "penalized_curvature"):
        np.testing.assert_allclose(
            getattr(result, field), getattr(baseline, field), rtol=tolerance, atol=tolerance
        )


def test_range_geometry_preserves_sum_before_intercept_and_default_child_plans():
    problem = _support_problem(False)
    _, layout, _, _, coefficients = problem
    state = layout.predictors[0]
    numeric, group = state.design.group_matrices
    coefficients[:] = 0
    start = state.coefficient_slice.start
    coefficients[start : start + 4] = [1e16, -1e16, 1, -1]
    numeric.M = np.ones_like(numeric.M)
    group.B_unique[:] = 0
    group.B_unique[:, :2] = [1e16, 1]
    group.R_inv[:] = 0
    group.R_inv[:2, :2] = [[1, 1], [1, 0]]
    state.offset.setflags(write=True)
    state.offset[:] = 0
    baseline = chunks.iter_likelihood_chunks(*problem, chunk_size=7, curvature_source="observed")
    ranged = chunks.iter_likelihood_chunks(
        *problem, chunk_size=7, curvature_source="observed", _range_geometry=True
    )
    for child, direct in zip(baseline, ranged, strict=True):
        assert all(plan.design.n <= 7 for plan in child.plans)
        assert all(
            plan.design is state.design
            for plan, state in zip(direct.plans, layout.predictors, strict=True)
        )
        np.testing.assert_array_equal(direct.eta[:, 0], 0)
        for field in ("eta", "theta", "score_eta", "curvature_packed"):
            np.testing.assert_array_equal(getattr(direct, field), getattr(child, field))


def test_range_refusal_releases_stream_before_complete_child_replay(monkeypatch):
    problem = _support_problem(True)
    p = problem[1].n_coefficients
    options = dict(penalty=np.eye(p), chunk_size=7, curvature_source="observed")
    expected = chunks.assemble_chunked_geometry(
        *problem, small_group_panel_byte_budget=None, **options
    )
    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: 64 << 20)
    original_build = chunks.build_global_moment_plan
    original_stream = chunks.iter_likelihood_chunks
    borrowed = []
    passes = []

    def build(*args, **kwargs):
        built = original_build(*args, **kwargs)
        kind = type(built.plan)
        original_add = kind.add_row_range

        def fail_second(self, plans, start, stop, score, curvature):
            if start == 7:
                raise GlobalMomentRefusalError("numerical-domain", recoverable=True)
            return original_add(self, plans, start, stop, score, curvature)

        monkeypatch.setattr(kind, "add_row_range", fail_second)
        return built

    def stream(*args, **kwargs):
        if passes:
            assert all(reference() is None for reference in borrowed)
        passes.append([])
        for chunk in original_stream(*args, **kwargs):
            passes[-1].append(chunk.rows.start)
            if len(passes) == 1:
                borrowed.append(weakref.ref(chunk.plans[0]))
                borrowed.append(weakref.ref(chunk.score_eta))
            yield chunk

    monkeypatch.setattr(chunks, "build_global_moment_plan", build)
    monkeypatch.setattr(chunks, "iter_likelihood_chunks", stream)
    result = chunks.assemble_chunked_geometry(*problem, **options)
    assert passes == [[0, 7], [0, 7, 14, 21]]
    tolerance = 64 * p * np.finfo(float).eps
    np.testing.assert_allclose(
        result.data_curvature, expected.data_curvature, rtol=tolerance, atol=tolerance
    )
