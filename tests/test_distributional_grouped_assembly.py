from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm._group_matrix._cross_matrix_execution import CrossMatrixExecutionPlan
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.distributional.assembly import assemble_grouped_geometry
from superglm.distributional.family import ParameterSpec, ParameterSupport
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.features import Numeric, Spline
from superglm.group_matrix import DiscretizedSSPGroupMatrix

from ._distributional_weights import resolved_prior


def _parameter(name: str, link: str = "identity") -> ParameterSpec:
    return ParameterSpec(name, link, name, ParameterSupport())


def _layout(*, discrete: bool, n: int = 180):
    x = np.linspace(-1.0, 1.0, n)
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "x": x,
                "z": np.sin(2.5 * np.pi * x) + 0.15 * np.cos(7.0 * x),
                "d": np.resize(np.array([-0.4, 0.2, 0.8, -0.1, 0.5]), n),
            }
        )
    )
    builds = compile_predictors(
        frame,
        resolved_prior(np.linspace(0.7, 1.4, n)),
        (_parameter("location"), _parameter("scale", "log")),
        (
            Predictor(
                "location",
                {
                    "x": Spline(kind="cr", n_knots=7, discrete=discrete),
                    "d": Numeric(),
                },
            ),
            Predictor(
                "scale",
                {
                    "z": Spline(kind="cr", n_knots=6, discrete=discrete),
                    "d": Numeric(),
                },
                link="log",
            ),
        ),
        offsets={
            "location": np.linspace(-0.08, 0.11, n),
            "scale": 0.04 * np.cos(np.linspace(0.0, 3.0 * np.pi, n)),
        },
        model_discrete=discrete,
        n_bins_config=31,
    )
    return build_stacked_layout(builds)


def _channels(n: int) -> tuple[np.ndarray, np.ndarray]:
    score = np.column_stack(
        (
            np.linspace(-1.1, 0.9, n),
            np.sin(np.linspace(0.0, 4.0 * np.pi, n)),
        )
    )
    curvature = np.column_stack(
        (
            0.8 + 0.3 * np.cos(np.linspace(0.0, 2.0 * np.pi, n)),
            np.linspace(-0.7, 0.9, n),
            1.2 + 0.25 * np.sin(np.linspace(0.0, 3.0 * np.pi, n)),
        )
    )
    curvature[::13, 1] = 0.0
    return score, curvature


def test_grouped_assembly_uses_two_symmetric_and_one_rectangular_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(discrete=True, n=120)
    score, curvature = _channels(120)
    coefficients = np.zeros(layout.n_coefficients)
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients))
    calls = {"symmetric": 0, "rectangular": 0}
    original_symmetric = MatrixExecutionPlan.moments
    original_rectangular = CrossMatrixExecutionPlan.cross_moment

    def counted_symmetric(self, *args, **kwargs):
        calls["symmetric"] += 1
        return original_symmetric(self, *args, **kwargs)

    def counted_rectangular(self, *args, **kwargs):
        calls["rectangular"] += 1
        return original_rectangular(self, *args, **kwargs)

    monkeypatch.setattr(MatrixExecutionPlan, "moments", counted_symmetric)
    monkeypatch.setattr(CrossMatrixExecutionPlan, "cross_moment", counted_rectangular)

    result = assemble_grouped_geometry(
        layout,
        score,
        curvature,
        penalty=penalty,
        coefficients=coefficients,
    )

    assert calls == {"symmetric": 2, "rectangular": 1}
    np.testing.assert_array_equal(result.data_curvature, result.data_curvature.T)


def test_compressed_grouped_assembly_does_not_materialize_discrete_slopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(discrete=True, n=240)
    score, curvature = _channels(240)
    coefficients = np.linspace(-0.15, 0.2, layout.n_coefficients)
    penalty = np.diag(np.linspace(0.0, 0.3, layout.n_coefficients))
    expected = assemble_grouped_geometry(
        layout,
        score,
        curvature,
        penalty=penalty,
        coefficients=coefficients,
    )

    def forbid_toarray(*_args, **_kwargs):
        raise AssertionError("discrete slope design was materialized")

    monkeypatch.setattr(DiscretizedSSPGroupMatrix, "toarray", forbid_toarray)
    actual = assemble_grouped_geometry(
        layout,
        score,
        curvature,
        penalty=penalty,
        coefficients=coefficients,
    )

    np.testing.assert_allclose(actual.score_data, expected.score_data, rtol=3e-13, atol=3e-12)
    np.testing.assert_allclose(
        actual.data_curvature,
        expected.data_curvature,
        rtol=4e-13,
        atol=4e-12,
    )
