"""Grouped chunk assembly matches an independent literal design."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from superglm._frame import as_eager_frame
from superglm.distributional.families.gaussian import GaussianLS
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


def _stream(layout, score, curvature):
    for start in range(0, len(score), 8):
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


def _assemble(monkeypatch, *, tiny_cross=False):
    layout, literal, score, curvature = _problem()
    if tiny_cross:
        curvature[:, 1] *= 2.0**-200
    monkeypatch.setattr(
        chunks,
        "iter_likelihood_chunks",
        lambda *args, **kwargs: _stream(layout, score, curvature),
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


def test_default_geometry_matches_independent_literal_design(monkeypatch):
    _assert_geometry(*_assemble(monkeypatch))


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
    # Four 8-row chunks stream in; the cross channel is contracted once over all rows.
    assert fallbacks == [len(curvature)]
