from __future__ import annotations

import tracemalloc
import weakref
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.solver.assembly as assembly_module
import superglm.distributional.solver.solver as solver_module
from superglm._frame import as_eager_frame
from superglm._group_matrix._block_queue import block_queue_config
from superglm.distributional.family import ParameterSpec, ParameterSupport
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.solver.assembly import (
    _dense_channel_blocks,
    _evaluate_predictors_from_matrices,
    assemble_grouped_geometry,
    dense_predictor_matrices,
)
from superglm.features import Numeric

from ._distributional_weights import resolved_prior
from .test_distributional_support_predictions import _support_problem


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


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("slope_width", [0, 2])
@pytest.mark.parametrize("borrowed", [False, True])
def test_dense_rendering_owns_readonly_storage(
    intercept: bool, slope_width: int, borrowed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Freezing or retaining mutable group storage must not publish aliases."""
    n = 5
    frame = as_eager_frame(pd.DataFrame({"x": np.arange(n), "z": -np.arange(n)}))
    builds = compile_predictors(
        frame,
        resolved_prior(np.ones(n)),
        (_parameter("location"),),
        (
            Predictor(
                "location",
                {name: Numeric() for name in ("x", "z")[:slope_width]},
                intercept=intercept,
            ),
        ),
    )
    layout = build_stacked_layout(builds)
    sources = [group.toarray() for group in layout.predictors[0].design.group_matrices]
    if borrowed and slope_width:
        # A dense rendering provider need not transfer ownership of its buffer.
        # Exercise that boundary without changing its values or design metadata.
        slopes = layout.predictors[0].design.toarray()
        monkeypatch.setattr(layout.predictors[0].design, "toarray", lambda: slopes)
        sources.append(slopes)
    writeability = [source.flags.writeable for source in sources]
    assert all(writeability)
    expected = np.empty((n, slope_width + int(intercept)))
    if intercept:
        expected[:, 0] = 1.0
    if slope_width:
        expected[:, int(intercept) :] = np.column_stack((np.arange(n), -np.arange(n)))

    (matrix,) = dense_predictor_matrices(layout)

    assert matrix.dtype == np.float64
    assert matrix.flags.owndata
    assert not matrix.flags.writeable
    np.testing.assert_array_equal(matrix, expected)
    for source, was_writeable in zip(sources, writeability, strict=True):
        assert source.flags.writeable == was_writeable
        assert not np.shares_memory(matrix, source)
        if was_writeable:
            source[:] = 73.0
    np.testing.assert_array_equal(matrix, expected)


def test_dense_rendering_peak_allocation_excludes_duplicate_augmented_matrix() -> None:
    """An extra full augmented copy exceeds the slopes-plus-output memory budget."""
    n, slope_width = 32768, 8
    frame = as_eager_frame(
        pd.DataFrame({f"x{i}": np.arange(n, dtype=float) for i in range(slope_width)})
    )
    builds = compile_predictors(
        frame,
        resolved_prior(np.ones(n)),
        (_parameter("location"),),
        (Predictor("location", {f"x{i}": Numeric() for i in range(slope_width)}),),
    )
    layout = build_stacked_layout(builds)
    # Allow the final matrix, slope rendering, intercept vector and 256 KiB
    # of Python/validation overhead. A second output is 2.25 MiB here.
    itemsize = np.dtype(np.float64).itemsize
    budget = n * (2 * slope_width + 2) * itemsize + 256 * 1024
    tracemalloc.start()
    try:
        (matrix,) = dense_predictor_matrices(layout)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert matrix.shape == (n, slope_width + 1)
    assert peak <= budget, f"dense rendering allocated {peak} bytes; budget is {budget}"


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


def test_a_dense_assembly_forms_its_formation_statistics_only_when_read(monkeypatch) -> None:
    """Kills forming the certificate's statistics for every geometry, read or not."""
    layout, frame = _layout()
    n = len(frame)
    rng = np.random.default_rng(3)
    score_eta = rng.normal(size=(n, 2))
    curvature = rng.uniform(0.5, 1.5, size=(n, 3))
    matrices = dense_predictor_matrices(layout)
    formed = []
    original = assembly_module.dense_formation

    def counted(*args):
        formed.append(1)
        return original(*args)

    monkeypatch.setattr(assembly_module, "dense_formation", counted)
    geometry = assembly_module._assemble_dense_geometry_from_matrices(
        layout,
        matrices,
        score_eta,
        curvature,
        penalty=np.diag([0.0, 0.4, 0.7, 0.0, 1.2]),
        coefficients=rng.normal(size=5),
    )
    assert formed == []

    formation = geometry.formed()
    assert geometry.formed() is formation and len(formed) == 1
    assert formation == original(layout, matrices, curvature)


def test_a_dense_fit_validates_its_penalty_once(monkeypatch) -> None:
    """Kills validating the context's already-validated penalty again at every geometry."""
    family, layout, response, plan, initial = _support_problem(False)
    calls = []
    original = assembly_module.validated_dense_penalty

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(assembly_module, "validated_dense_penalty", counted)
    monkeypatch.setattr(solver_module, "validated_dense_penalty", counted)
    penalty = np.eye(layout.n_coefficients)
    fitted = solver_module.fit_dense_fixed_lambda(
        family, layout, response, plan, penalty, initial=initial, chunk_size=None
    )
    assert fitted.iterations > 1 and len(calls) == 1

    # Unvalidated input still gets the checks.
    asymmetric = penalty.copy()
    asymmetric[0, 1] = 1.0
    with pytest.raises(ValueError, match="penalty must be symmetric"):
        assembly_module._assemble_dense_geometry_from_matrices(
            layout,
            dense_predictor_matrices(layout),
            np.zeros((layout.predictors[0].design.n, 2)),
            np.ones((layout.predictors[0].design.n, 3)),
            penalty=asymmetric,
            coefficients=np.zeros(layout.n_coefficients),
        )


def test_dense_predictor_evaluation_includes_intercepts_slopes_and_offsets() -> None:
    layout, frame = _layout()
    coefficients = np.array([0.2, -0.3, 0.5, -0.4, 0.8])

    eta = _evaluate_predictors_from_matrices(layout, coefficients, dense_predictor_matrices(layout))

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

    eta = _evaluate_predictors_from_matrices(
        layout, np.array([1.25]), dense_predictor_matrices(layout)
    )
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
        _evaluate_predictors_from_matrices(layout, np.zeros(4), dense_predictor_matrices(layout))

    broken_scale = replace(layout.predictors[1], offset=np.zeros(n - 1))
    broken_layout = object.__new__(type(layout))
    object.__setattr__(broken_layout, "predictors", (layout.predictors[0], broken_scale))
    object.__setattr__(broken_layout, "n_coefficients", layout.n_coefficients)
    object.__setattr__(broken_layout, "coefficient_names", layout.coefficient_names)
    object.__setattr__(broken_layout, "term_slices", layout.term_slices)
    object.__setattr__(broken_layout, "penalties", layout.penalties)
    with pytest.raises(ValueError, match="predictor row counts"):
        _evaluate_predictors_from_matrices(
            broken_layout, np.zeros(5), dense_predictor_matrices(broken_layout)
        )


@pytest.mark.parametrize("weights", ["nonnegative", "mixed"])
def test_wide_dense_gram_is_the_same_at_every_worker_count(weights: str) -> None:
    rng = np.random.default_rng(19)
    n_rows = 2 * assembly_module._GRAM_BLOCK_ROWS + 5
    matrices = (rng.standard_normal((n_rows, 130)), rng.standard_normal((n_rows, 40)))
    curvature = rng.standard_normal((n_rows, 3))
    if weights == "nonnegative":
        curvature[:, 0] = np.abs(curvature[:, 0])
        curvature[::7, 0] = 0.0
    channels = [(0, 0), (0, 1), (1, 1)]

    with block_queue_config(workers=1):
        serial = _dense_channel_blocks(matrices, curvature, channels)
    with block_queue_config(workers=4):
        pooled = _dense_channel_blocks(matrices, curvature, channels)

    for channel, (left, right) in enumerate(channels):
        np.testing.assert_array_equal(pooled[channel], serial[channel])
        channel_weights = curvature[:, channel, None]
        literal = matrices[left].T @ (channel_weights * matrices[right])
        magnitude = np.abs(matrices[left]).T @ (np.abs(channel_weights) * np.abs(matrices[right]))
        assert np.all(np.abs(pooled[channel] - literal) <= _gamma(n_rows) * magnitude)


def test_the_dense_gram_charges_its_dispatch_in_multiply_adds(monkeypatch) -> None:
    """Its task costs are multiply-adds, so it stays off the row-visit overhead."""
    seen = []
    run = assembly_module.run_block_tasks

    def spy(tasks, cache, profile, *, dispatch_bound=False, row_visits=False):
        seen.append((dispatch_bound, row_visits))
        run(tasks, cache, profile, dispatch_bound=dispatch_bound, row_visits=row_visits)

    monkeypatch.setattr(assembly_module, "run_block_tasks", spy)
    n_rows = 2 * assembly_module._GRAM_BLOCK_ROWS + 5
    rows = np.ones((n_rows, 130))
    with block_queue_config(workers=1):
        _dense_channel_blocks((rows,), np.ones((n_rows, 1)), [(0, 0)])
    assert seen == [(True, False)]


def test_the_dense_gram_keeps_a_running_total_rather_than_every_block(monkeypatch) -> None:
    real_rows = assembly_module._gram_rows
    formed = []
    most_alive = 0

    def spy(*args):
        nonlocal most_alive
        most_alive = max(most_alive, sum(block() is not None for block in formed))
        block = real_rows(*args)
        formed.append(weakref.ref(block))
        return block

    monkeypatch.setattr(assembly_module, "_gram_rows", spy)
    rng = np.random.default_rng(29)
    matrix = rng.standard_normal((6 * assembly_module._GRAM_BLOCK_ROWS, 130))
    weights = rng.standard_normal((matrix.shape[0], 1))

    with block_queue_config(workers=1):
        (block,) = _dense_channel_blocks((matrix,), weights, [(0, 0)])

    assert len(formed) == 6 and most_alive == 1
    assert block.shape == (130, 130)


def test_a_narrow_dense_channel_stays_one_general_product(monkeypatch) -> None:
    def no_queue(*_args, **_kwargs):
        raise AssertionError("a narrow channel uses no block queue")

    monkeypatch.setattr(assembly_module, "run_block_tasks", no_queue)
    rng = np.random.default_rng(23)
    matrix = rng.standard_normal((3 * assembly_module._GRAM_BLOCK_ROWS, 5))
    weights = rng.standard_normal((matrix.shape[0], 1))

    (block,) = _dense_channel_blocks((matrix,), weights, [(0, 0)])

    np.testing.assert_array_equal(block, matrix.T @ (weights * matrix))


@pytest.mark.parametrize(
    ("n_rows", "blocks"),
    [(300_000, assembly_module._GRAM_MAX_BLOCKS), (2 * assembly_module._GRAM_BLOCK_ROWS + 5, 3)],
)
def test_the_dense_gram_partition_follows_the_row_count(monkeypatch, n_rows, blocks) -> None:
    run = assembly_module.run_block_tasks
    seen = []

    def counted(tasks, *args, **kwargs):
        seen.append(len(tasks))
        run(tasks, *args, **kwargs)

    monkeypatch.setattr(assembly_module, "run_block_tasks", counted)
    monkeypatch.setattr(assembly_module, "_GRAM_POOL_COST", 1)
    matrix = np.ones((n_rows, 2))

    (block,) = _dense_channel_blocks((matrix,), np.ones((n_rows, 1)), [(0, 0)])

    assert seen == [blocks]
    np.testing.assert_array_equal(block, np.full((2, 2), float(n_rows)))
