from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.solver.assembly as assembly_module
from superglm._frame import as_eager_frame
from superglm.distributional.assembly import (
    assemble_grouped_geometry,
    dense_predictor_matrices,
    evaluate_predictors_dense,
)
from superglm.distributional.family import ParameterSpec, ParameterSupport
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.features import Numeric

from ._distributional_weights import resolved_prior


def _parameter(name: str, link: str = "identity") -> ParameterSpec:
    return ParameterSpec(name, link, name, ParameterSupport())


def _layout(n: int = 7):
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "x": np.linspace(-1.0, 1.0, n),
                "z": np.array([0.3, -0.2, 0.7, 1.1, -0.8, 0.4, 0.9])[:n],
                "w": np.linspace(0.2, 1.4, n),
            }
        )
    )
    builds = compile_predictors(
        frame,
        resolved_prior(np.ones(n)),
        (_parameter("location"), _parameter("scale", "log")),
        (
            Predictor("location", {"x": Numeric(), "z": Numeric()}),
            Predictor("scale", {"w": Numeric()}, link="log"),
        ),
        offsets={
            "location": np.linspace(-0.1, 0.2, n),
            "scale": np.linspace(0.3, -0.2, n),
        },
    )
    return build_stacked_layout(builds), frame


def _gamma(operation_count: int) -> float:
    epsilon = np.finfo(np.float64).eps
    return operation_count * epsilon / (1.0 - operation_count * epsilon)


@pytest.mark.parametrize(
    "cross_weights",
    [
        np.array([-0.4, 0.0, -0.1, 0.7, -0.3, 0.5, 0.0]),
        np.zeros(7),
    ],
    ids=["signed-with-zeros", "zero-channel"],
)
def test_retained_dense_geometry_matches_literal_augmented_products(
    cross_weights: np.ndarray,
) -> None:
    """Kills rematerialization or a wrong signed/zero dense block product."""

    layout, frame = _layout()
    n = len(frame)
    location_design = np.column_stack(
        (
            np.ones(n),
            frame.column_array("x"),
            frame.column_array("z"),
        )
    )
    scale_design = np.column_stack((np.ones(n), frame.column_array("w")))
    score_eta = np.column_stack(
        (
            np.linspace(-0.8, 1.1, n),
            np.array([0.7, -0.3, 0.4, 1.2, -0.9, 0.6, -0.2]),
        )
    )
    curvature_packed = np.column_stack(
        (
            np.linspace(0.5, 1.3, n),
            cross_weights,
            np.linspace(1.7, 0.6, n),
        )
    )
    coefficients = np.array([0.2, -0.3, 0.5, -0.4, 0.8])
    penalty = np.diag([0.0, 0.4, 0.7, 0.0, 1.2])

    results = (
        assembly_module._assemble_dense_geometry_from_matrices(
            layout,
            dense_predictor_matrices(layout),
            score_eta,
            curvature_packed,
            penalty=penalty,
            coefficients=coefficients,
        ),
        assemble_grouped_geometry(
            layout,
            score_eta,
            curvature_packed,
            penalty=penalty,
            coefficients=coefficients,
        ),
    )

    expected_score_data = np.concatenate(
        (
            location_design.T @ score_eta[:, 0],
            scale_design.T @ score_eta[:, 1],
        )
    )
    expected_h00 = location_design.T @ (curvature_packed[:, 0, None] * location_design)
    expected_h01 = location_design.T @ (curvature_packed[:, 1, None] * scale_design)
    expected_h11 = scale_design.T @ (curvature_packed[:, 2, None] * scale_design)
    expected_h_data = np.block([[expected_h00, expected_h01], [expected_h01.T, expected_h11]])
    score_scale = max(
        1.0,
        float(np.max(np.abs(location_design).T @ np.abs(score_eta[:, 0]))),
        float(np.max(np.abs(scale_design).T @ np.abs(score_eta[:, 1]))),
    )
    curvature_absolute_product = max(
        1.0,
        float(
            np.max(
                np.abs(location_design).T
                @ (np.abs(curvature_packed[:, 0, None]) * np.abs(location_design))
            )
        ),
        float(
            np.max(
                np.abs(location_design).T
                @ (np.abs(curvature_packed[:, 1, None]) * np.abs(scale_design))
            )
        ),
        float(
            np.max(
                np.abs(scale_design).T
                @ (np.abs(curvature_packed[:, 2, None]) * np.abs(scale_design))
            )
        ),
    )
    score_bound = _gamma(4 * n + 16) * score_scale
    curvature_bound = _gamma(8 * n + 32) * curvature_absolute_product

    for result in results:
        np.testing.assert_allclose(
            result.score_data,
            expected_score_data,
            rtol=0.0,
            atol=score_bound,
        )
        np.testing.assert_allclose(
            result.score_penalized,
            expected_score_data - penalty @ coefficients,
            rtol=0.0,
            atol=score_bound + _gamma(16) * float(np.linalg.norm(penalty, ord=np.inf)),
        )
        np.testing.assert_allclose(
            result.data_curvature,
            expected_h_data,
            rtol=0.0,
            atol=curvature_bound,
        )
        np.testing.assert_allclose(
            result.penalized_curvature,
            expected_h_data + penalty,
            rtol=0.0,
            atol=curvature_bound + _gamma(2) * float(np.linalg.norm(penalty, ord=np.inf)),
        )
        np.testing.assert_array_equal(
            result.penalized_curvature,
            result.penalized_curvature.T,
        )
        if not np.any(cross_weights):
            left = layout.predictors[0].coefficient_slice
            right = layout.predictors[1].coefficient_slice
            np.testing.assert_array_equal(result.data_curvature[left, right], 0.0)
        assert not result.data_curvature.flags.writeable


def test_dense_predictor_evaluation_includes_intercepts_slopes_and_offsets() -> None:
    layout, frame = _layout()
    coefficients = np.array([0.2, -0.3, 0.5, -0.4, 0.8])

    eta = evaluate_predictors_dense(layout, coefficients)

    expected_location = (
        0.2
        - 0.3 * frame.column_array("x")
        + 0.5 * frame.column_array("z")
        + layout.predictors[0].offset
    )
    expected_scale = -0.4 + 0.8 * frame.column_array("w") + layout.predictors[1].offset
    np.testing.assert_allclose(eta[:, 0], expected_location)
    np.testing.assert_allclose(eta[:, 1], expected_scale)
    assert not eta.flags.writeable


def test_zero_width_slope_and_no_intercept_predictors_are_supported() -> None:
    n = 5
    frame = as_eager_frame(pd.DataFrame({"row": np.arange(n)}))
    offsets = {
        "location": np.linspace(-0.2, 0.2, n),
        "scale": np.linspace(-1.0, 1.0, n),
    }
    builds = compile_predictors(
        frame,
        resolved_prior(np.ones(n)),
        (_parameter("location"), _parameter("scale", "log")),
        (
            Predictor("location", {}, intercept=True),
            Predictor("scale", {}, link="log", intercept=False),
        ),
        offsets=offsets,
    )
    layout = build_stacked_layout(builds)

    eta = evaluate_predictors_dense(layout, np.array([1.25]))
    geometry = assemble_grouped_geometry(
        layout,
        np.column_stack((np.arange(n), -np.arange(n))),
        np.column_stack((np.ones(n), -np.ones(n), 2.0 * np.ones(n))),
        penalty=np.zeros((1, 1)),
        coefficients=np.array([1.25]),
    )

    np.testing.assert_allclose(eta[:, 0], 1.25 + offsets["location"])
    np.testing.assert_allclose(eta[:, 1], offsets["scale"])
    assert geometry.score_data.shape == (1,)
    assert geometry.data_curvature.shape == (1, 1)


@pytest.mark.parametrize(
    ("penalty", "message"),
    [
        (np.zeros((4, 4)), "penalty.*shape"),
        (
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "penalty.*symmetric",
        ),
    ],
)
def test_assembler_rejects_mismatched_or_nonsymmetric_penalties(
    penalty: np.ndarray, message: str
) -> None:
    layout, frame = _layout()
    with pytest.raises(ValueError, match=message):
        assemble_grouped_geometry(
            layout,
            np.zeros((len(frame), 2)),
            np.zeros((len(frame), 3)),
            penalty=penalty,
            coefficients=np.zeros(layout.n_coefficients),
        )


def test_assembler_rejects_channel_coefficient_and_layout_shape_mismatches() -> None:
    layout, frame = _layout()
    n = len(frame)

    with pytest.raises(ValueError, match="score_eta.*shape"):
        assemble_grouped_geometry(
            layout,
            np.zeros((n, 1)),
            np.zeros((n, 3)),
            penalty=np.zeros((5, 5)),
            coefficients=np.zeros(5),
        )
    with pytest.raises(ValueError, match="curvature_packed.*shape"):
        assemble_grouped_geometry(
            layout,
            np.zeros((n, 2)),
            np.zeros((n, 2)),
            penalty=np.zeros((5, 5)),
            coefficients=np.zeros(5),
        )
    with pytest.raises(ValueError, match="coefficients.*shape"):
        evaluate_predictors_dense(layout, np.zeros(4))

    broken_scale = replace(layout.predictors[1], offset=np.zeros(n - 1))
    broken_layout = object.__new__(type(layout))
    object.__setattr__(broken_layout, "predictors", (layout.predictors[0], broken_scale))
    object.__setattr__(broken_layout, "n_coefficients", layout.n_coefficients)
    object.__setattr__(broken_layout, "coefficient_names", layout.coefficient_names)
    object.__setattr__(broken_layout, "term_slices", layout.term_slices)
    object.__setattr__(broken_layout, "penalties", layout.penalties)
    with pytest.raises(ValueError, match="predictor row counts"):
        evaluate_predictors_dense(broken_layout, np.zeros(5))
