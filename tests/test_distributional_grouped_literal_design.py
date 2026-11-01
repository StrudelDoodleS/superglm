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
from superglm.features import Categorical, Numeric, Spline

from ._distributional_weights import resolved_prior
from .test_distributional_fused_trial import _literal_design


def _problem(*, mirror=False):
    n = 29
    x = np.linspace(-1.1, 0.8, n)
    if mirror:
        # Rows 14-27 repeat rows 0-13, so opposite scores on the halves cancel.
        x[14:28] = x[:14]
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


def _assemble(monkeypatch, *, tiny_cross=False, problem=None):
    layout, literal, score, curvature = _problem() if problem is None else problem
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


def test_tiny_signed_cross_channel_matches_the_literal_design(monkeypatch):
    cross = PredictorExecutionPlan.cross_moment
    contracted_rows = []

    def counted_cross(self, right, weights):
        contracted_rows.append(len(weights))
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
    assert contracted_rows == [len(curvature)]


def _mixed_problem():
    n = 29
    rows = np.arange(n)
    frame = pd.DataFrame(
        {
            "x": np.linspace(-1.1, 0.8, n),
            "s": np.cos(rows * 0.7),
            "g": np.resize(list("abc"), n),
        }
    )
    family = GaussianLS()
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved_prior(np.ones(n)),
        family.parameters,
        (
            Predictor(
                "location",
                {"x": Numeric(), "s": Spline(kind="cr", k=4), "g": Categorical()},
                interactions=(("s", "g"),),
            ),
            Predictor("scale", {"s": Spline(kind="cr", k=4)}),
        ),
        model_discrete=True,
        n_bins_config=9,
    )
    layout = build_stacked_layout(compiled)
    score = np.column_stack((np.sin(rows), np.linspace(-0.3, 0.8, n)))
    curvature = np.column_stack(
        (np.linspace(-0.7, 1.1, n), np.cos(rows), np.linspace(1.3, -0.2, n))
    )
    return layout, _literal_design(layout), score, curvature


def test_two_chunk_blocks_match_the_literal_mixed_design(monkeypatch):
    problem = _mixed_problem()
    kinds = {type(g).__name__ for g in problem[0].predictors[0].design.group_matrices}
    assert "DiscretizedSplineCategoricalGroupMatrix" in kinds
    monkeypatch.setattr(chunks, "contraction_block_rows", lambda n, row_bytes, rows: 2 * rows)
    moments = []
    for name in ("diagonal_moment", "cross_moment"):
        original = getattr(PredictorExecutionPlan, name)

        def counted(self, *args, _original=original, **kwargs):
            moments.append(self.design.n)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(PredictorExecutionPlan, name, counted)
    _assert_geometry(*_assemble(monkeypatch, problem=problem))
    # Four 8-row chunks: blocks of 16 and 13 rows, three channels each.
    assert moments == [16] * 3 + [13] * 3


def test_cancelling_scores_beyond_the_float_range_aggregate_finitely(monkeypatch):
    layout, literal, score, curvature = _problem(mirror=True)
    n = len(score)
    # A category level's five same-sign rows sum past float64's range in a
    # sequential bincount; the true scores are at most one row's value.
    score[:, 0] = 0.0
    score[:14, 0] = 2.0**1022
    score[14:28, 0] = -(2.0**1022)
    result, *_ = _assemble(monkeypatch, problem=(layout, literal, score, curvature))
    assert np.all(np.isfinite(result.score_data))
    scaled = 2.0**-30
    expected = literal[0].T @ (score[:, 0] * scaled)
    tolerance = 32 * n * np.finfo(float).eps * (np.abs(literal[0]).T @ np.abs(score[:, 0] * scaled))
    actual = result.score_data[layout.predictors[0].coefficient_slice] * scaled
    assert np.all(np.abs(actual - expected) <= tolerance), (actual, expected, tolerance)
