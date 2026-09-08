"""Chunk assembly contracts for the internal bounded panel experiment."""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, PredictorExecutionPlan, compile_predictors
from superglm.distributional.solver import chunks
from superglm.features import Categorical, Numeric

from ._distributional_weights import resolved_prior


def _problem():
    n = 29
    x = np.linspace(-1.1, 0.8, n)
    z = np.cos(np.arange(n) * 0.3)
    category = np.resize(["a", "b", "c"], n)
    family = GaussianLS()
    frame = as_eager_frame(pd.DataFrame({"x": x, "z": z, "g": category}))
    compiled = compile_predictors(
        frame,
        resolved_prior(np.ones(n)),
        family.parameters,
        (
            Predictor("location", {"x": Numeric(), "g": Categorical(base="c")}),
            Predictor("scale", {"z": Numeric()}, intercept=False),
        ),
    )
    layout = build_stacked_layout(compiled)
    # Literal model columns are independent of any stored-design renderer.
    literal = (
        np.column_stack((np.ones(n), x, category == "a", category == "b")),
        z[:, None],
    )
    score = np.column_stack((np.sin(np.arange(n)), np.linspace(-0.3, 0.8, n)))
    curvature = np.column_stack(
        (np.linspace(-0.7, 1.1, n), np.cos(np.arange(n)), np.linspace(1.3, -0.2, n))
    )
    curvature[::4] = 0
    return layout, literal, score, curvature


def _stream(layout, score, curvature, *, before_advance=None):
    for start in range(0, len(score), 8):
        if before_advance is not None:
            before_advance()
        stop = min(start + 8, len(score))
        rows = np.arange(start, stop, dtype=np.intp)
        yield SimpleNamespace(
            plans=tuple(
                PredictorExecutionPlan(
                    state.design.row_subset(rows), state.intercept_index is not None
                )
                for state in layout.predictors
            ),
            score_eta=score[start:stop],
            curvature_packed=curvature[start:stop],
        )
    if before_advance is not None:
        before_advance()


def _assemble(monkeypatch, *, byte_budget=2**20, before_advance=None, tiny_cross=False):
    layout, literal, score, curvature = _problem()
    if tiny_cross:
        curvature[:, 1] *= 2.0**-200
    monkeypatch.setattr(
        chunks,
        "iter_likelihood_chunks",
        lambda *args, **kwargs: _stream(layout, score, curvature, before_advance=before_advance),
    )
    coefficients = np.linspace(-0.2, 0.1, layout.n_coefficients)
    penalty = np.diag(np.linspace(0, 0.3, layout.n_coefficients))
    result = chunks.assemble_chunked_geometry(
        None,
        layout,
        None,
        None,
        coefficients,
        penalty=penalty,
        chunk_size=8,
        curvature_source="observed",
        small_group_panel_byte_budget=byte_budget,
    )
    return result, layout, literal, score, curvature, penalty, coefficients


def _assert_geometry(result, layout, literal, score, curvature, penalty, coefficients):
    expected_score = np.concatenate([matrix.T @ score[:, j] for j, matrix in enumerate(literal)])
    expected = np.zeros_like(penalty)
    for channel, (left, right) in enumerate(((0, 0), (0, 1), (1, 1))):
        block = literal[left].T @ (curvature[:, channel, None] * literal[right])
        ls = layout.predictors[left].coefficient_slice
        rs = layout.predictors[right].coefficient_slice
        expected[ls, rs] = block
        expected[rs, ls] = block.T
    absolute_design = np.column_stack([np.abs(matrix) for matrix in literal])
    scale = np.linalg.norm(absolute_design, "fro") ** 2 * np.max(np.abs(curvature))
    tolerance = 32 * len(score) * np.finfo(float).eps * max(1, scale)
    np.testing.assert_allclose(result.score_data, expected_score, rtol=0, atol=tolerance)
    np.testing.assert_allclose(result.data_curvature, expected, rtol=0, atol=tolerance)
    np.testing.assert_allclose(
        result.score_penalized, expected_score - penalty @ coefficients, rtol=0, atol=tolerance
    )
    np.testing.assert_allclose(
        result.penalized_curvature, expected + penalty, rtol=0, atol=tolerance
    )
    np.testing.assert_array_equal(result.data_curvature, result.data_curvature.T)


def test_signed_rectangular_chunk_panels_match_independent_literal_design(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("eligible curvature must use panels")

    monkeypatch.setattr(PredictorExecutionPlan, "diagonal_moment", forbidden)
    monkeypatch.setattr(PredictorExecutionPlan, "cross_moment", forbidden)
    _assert_geometry(*_assemble(monkeypatch))


@pytest.mark.parametrize("byte_budget", [None, 1])
def test_default_and_budget_refusal_keep_grouped_assembly(monkeypatch, byte_budget):
    _assert_geometry(*_assemble(monkeypatch, byte_budget=byte_budget))


def test_default_does_not_build_panels(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("default execution must not prepare panels")

    monkeypatch.setattr(chunks, "build_small_group_panels", forbidden)
    _assert_geometry(*_assemble(monkeypatch, byte_budget=None))


@pytest.mark.parametrize("reason", ["numerical-domain", "specialized-group", "unsupported-group"])
def test_builder_refusal_keeps_complete_grouped_channels(monkeypatch, reason):
    monkeypatch.setattr(
        chunks,
        "build_small_group_panels",
        lambda *args, **kwargs: SimpleNamespace(workspace=None, reason=reason),
    )
    _assert_geometry(*_assemble(monkeypatch))


def test_workspace_is_reused_then_released_before_next_chunk(monkeypatch):
    original = chunks.build_small_group_panels
    references = []
    calls = []

    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        workspace = result.workspace
        assert workspace is not None
        references.extend(weakref.ref(panel) for panel in workspace.panels)
        identities = tuple(map(id, workspace.panels))
        cross = workspace.cross_moment

        def counted(left, right, weights):
            assert tuple(map(id, workspace.panels)) == identities
            calls.append((left, right))
            return cross(left, right, weights)

        workspace.cross_moment = counted
        return result

    def released():
        assert all(reference() is None for reference in references)

    monkeypatch.setattr(chunks, "build_small_group_panels", build)
    _assert_geometry(*_assemble(monkeypatch, before_advance=released))
    assert calls == [(0, 0), (0, 1), (1, 1)] * 4


def test_late_channel_refusal_falls_back_without_duplicate_accumulation(monkeypatch):
    original = chunks.build_small_group_panels
    fallbacks = []
    cross = PredictorExecutionPlan.cross_moment

    def counted_cross(self, right, weights):
        fallbacks.append(len(weights))
        return cross(self, right, weights)

    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        workspace = result.workspace
        assert workspace is not None
        moment = workspace.cross_moment
        workspace.cross_moment = lambda left, right, weights: (
            None if left != right else moment(left, right, weights)
        )
        return result

    monkeypatch.setattr(chunks, "build_small_group_panels", build)
    monkeypatch.setattr(PredictorExecutionPlan, "cross_moment", counted_cross)
    _assert_geometry(*_assemble(monkeypatch))
    assert fallbacks == [8, 8, 8, 5]


def test_actual_weight_domain_refusal_preserves_signed_cross_channel(monkeypatch):
    cross = PredictorExecutionPlan.cross_moment
    fallbacks = []

    def counted_cross(self, right, weights):
        fallbacks.append(len(weights))
        return cross(self, right, weights)

    monkeypatch.setattr(PredictorExecutionPlan, "cross_moment", counted_cross)
    result, layout, literal, _, curvature, _, _ = _assemble(monkeypatch, tiny_cross=True)
    expected = literal[0].T @ (curvature[:, 1, None] * literal[1])
    absolute_sum = np.abs(literal[0]).T @ (np.abs(curvature[:, 1, None]) * np.abs(literal[1]))
    tolerance = 32 * len(curvature) * np.finfo(float).eps * np.max(absolute_sum)
    np.testing.assert_allclose(
        result.data_curvature[
            layout.predictors[0].coefficient_slice, layout.predictors[1].coefficient_slice
        ],
        expected,
        rtol=0,
        atol=tolerance,
    )
    assert fallbacks == [8, 8, 8, 5]


@pytest.mark.parametrize("phase", ["score", "curvature"])
def test_workspace_closes_when_assembly_raises(monkeypatch, phase):
    original = chunks.build_small_group_panels
    references = []

    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        workspace = result.workspace
        assert workspace is not None
        references.extend(weakref.ref(panel) for panel in workspace.panels)

        def raises(*args):
            raise RuntimeError("injected assembly failure")

        if phase == "curvature":
            workspace.cross_moment = raises
        else:
            monkeypatch.setattr(chunks.GroupedGeometryAccumulator, "add_score", raises)
        return result

    monkeypatch.setattr(chunks, "build_small_group_panels", build)
    with pytest.raises(RuntimeError, match="injected assembly failure"):
        _assemble(monkeypatch)
    assert references and all(reference() is None for reference in references)


def test_real_likelihood_chunk_geometry_preserves_score_and_signed_curvature():
    layout, literal, _, _ = _problem()
    family = GaussianLS()
    response = 0.3 + 0.8 * literal[0][:, 1] + np.sin(np.arange(29) * 0.7)
    plan = family.bind_likelihood(response, resolved_prior(np.ones(29)), COMPLETE_OBSERVATION)
    coefficients = np.linspace(-0.1, 0.15, layout.n_coefficients)
    penalty = np.diag(np.linspace(0, 0.4, layout.n_coefficients))
    arguments = (family, layout, response, plan, coefficients)
    options = {"penalty": penalty, "chunk_size": 8, "curvature_source": "observed"}
    expected = chunks.assemble_chunked_geometry(*arguments, **options)
    actual = chunks.assemble_chunked_geometry(
        *arguments, **options, small_group_panel_byte_budget=2**20
    )
    # Row-dot-product backward error, in coefficient-space geometry coordinates.
    tolerance = (
        32
        * len(response)
        * np.finfo(float).eps
        * max(1, np.linalg.norm(expected.data_curvature, ord=np.inf))
    )
    np.testing.assert_array_equal(actual.score_data, expected.score_data)
    np.testing.assert_allclose(
        actual.data_curvature, expected.data_curvature, rtol=0, atol=tolerance
    )
