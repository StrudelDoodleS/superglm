from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from superglm._frame import as_eager_frame
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.inference import compute_joint_inference
from superglm.distributional.layout import StackedLayout, build_stacked_layout
from superglm.distributional.penalty_face import build_penalty_face
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.result import DenseSolverConfig, DenseSolverResult
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.features import Numeric
from superglm.solvers.rank import decompose_gram
from superglm.types import PenaltyComponent

from ._distributional_weights import resolved_prior


def _layout() -> StackedLayout:
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "x": [-1.0, -0.25, 0.5, 1.25],
                "z": [0.4, -0.8, 0.3, 1.1],
                "q": [1.0, 0.2, -0.5, 0.7],
            }
        )
    )
    predictors = (
        Predictor("location", {"x": Numeric(), "z": Numeric()}),
        Predictor("scale", {"q": Numeric()}),
    )
    compiled = compile_predictors(
        frame,
        resolved_prior(np.ones(len(frame))),
        GaussianLS().parameters,
        predictors,
    )
    layout = build_stacked_layout(compiled)
    assert layout.coefficient_names == (
        "location:(intercept)",
        "location:x",
        "location:z",
        "scale:(intercept)",
        "scale:q",
    )
    return layout


def _result(
    layout: StackedLayout,
    data_curvature: np.ndarray,
    penalty: np.ndarray,
    *,
    source: str = "observed",
) -> DenseSolverResult:
    penalized = np.asarray(data_curvature + penalty, dtype=np.float64)
    decomposition = decompose_gram(penalized)
    eigenvalues = np.linalg.eigvalsh(data_curvature)
    nonzero = np.abs(eigenvalues) > np.finfo(float).eps
    condition = (
        float(np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues[nonzero])))
        if np.any(nonzero)
        else None
    )
    telemetry = CurvatureTelemetry(
        requested_source="observed",
        actual_source=source,
        reason=None if source == "observed" else "material_indefiniteness_after_retry",
        minimum_eigenvalue=float(eigenvalues[0]),
        rank=int(np.linalg.matrix_rank(data_curvature)),
        condition_estimate=condition,
        fallback_count=0 if source == "observed" else 1,
    )
    n_observations = layout.predictors[0].design.n
    return DenseSolverResult(
        config=DenseSolverConfig(residual_tolerance=1.0e-10),
        family_likelihood_plan_identifier="test-plan:v1",
        resolved_chunk_size=None,
        execution_backend_identifier="distributional-dense-v1",
        coefficients=np.zeros(layout.n_coefficients),
        eta=np.zeros((n_observations, len(layout.predictors))),
        theta=np.column_stack((np.zeros(n_observations), np.ones(n_observations))),
        penalty=penalty,
        initial_penalized_log_likelihood=0.0,
        initial_penalized_optimizing_log_likelihood=0.0,
        log_likelihood=0.0,
        penalty_value=0.0,
        penalized_log_likelihood=0.0,
        terminal_score=np.zeros(layout.n_coefficients),
        score_relative=0.0,
        objective_relative_change=0.0,
        step_relative=0.0,
        converged=True,
        convergence_reason="score",
        iterations=0,
        history=(),
        backtracking_steps=0,
        terminal_data_curvature=data_curvature,
        terminal_penalized_curvature=penalized,
        terminal_rank=decomposition,
        terminal_curvature=telemetry,
    )


def _full_rank_fixture() -> tuple[StackedLayout, DenseSolverResult]:
    layout = _layout()
    penalized = np.array(
        [
            [4.0, 0.5, 0.0, 0.3, 0.0],
            [0.5, 3.0, 0.2, 0.0, 0.1],
            [0.0, 0.2, 2.0, 0.4, 0.0],
            [0.3, 0.0, 0.4, 2.5, 0.5],
            [0.0, 0.1, 0.0, 0.5, 1.5],
        ]
    )
    penalty = np.diag([0.0, 0.8, 2.5, 0.0, 0.4])
    return layout, _result(layout, penalized - penalty, penalty)


def test_joint_covariance_and_edf_match_dense_solve_without_explicit_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, result = _full_rank_fixture()
    expected_covariance = scipy.linalg.solve(
        result.terminal_penalized_curvature,
        np.eye(layout.n_coefficients),
        assume_a="sym",
    )
    expected_influence = scipy.linalg.solve(
        result.terminal_penalized_curvature,
        result.terminal_data_curvature,
        assume_a="sym",
    )

    def forbid_inverse(*args, **kwargs):
        raise AssertionError("joint inference must not form an explicit inverse")

    monkeypatch.setattr(np.linalg, "inv", forbid_inverse)
    inference = compute_joint_inference(layout, result)

    np.testing.assert_allclose(inference.covariance, expected_covariance)
    np.testing.assert_allclose(inference.influence, expected_influence, atol=1.0e-14)
    np.testing.assert_allclose(inference.coefficient_edf, np.diag(expected_influence))
    assert inference.total_edf == pytest.approx(float(np.trace(expected_influence)))
    np.testing.assert_allclose(
        inference.covariance[
            layout.predictor("location").coefficient_slice,
            layout.predictor("scale").coefficient_slice,
        ],
        expected_covariance[:3, 3:],
    )


def test_edf_attribution_uses_qualified_complete_slices_and_retains_negatives() -> None:
    layout, result = _full_rank_fixture()
    inference = compute_joint_inference(layout, result)
    diagonal = np.asarray(inference.coefficient_edf)

    assert inference.predictor_edf == {
        "location": pytest.approx(float(np.sum(diagonal[:3]))),
        "scale": pytest.approx(float(np.sum(diagonal[3:]))),
    }
    assert inference.intercept_edf == {
        "location:(intercept)": pytest.approx(float(diagonal[0])),
        "scale:(intercept)": pytest.approx(float(diagonal[3])),
    }
    assert inference.term_edf == {
        "location:x": pytest.approx(float(diagonal[1])),
        "location:z": pytest.approx(float(diagonal[2])),
        "scale:q": pytest.approx(float(diagonal[4])),
    }
    assert inference.term_edf["location:z"] < 0.0
    assert inference.negative_coefficient_edf == {"location:z": pytest.approx(float(diagonal[2]))}
    assert inference.slice_reconciliation_error <= inference.reconciliation_tolerance
    assert inference.predictor_reconciliation_error <= inference.reconciliation_tolerance
    assert sum(inference.predictor_edf.values()) == pytest.approx(inference.total_edf)
    assert sum(inference.intercept_edf.values()) + sum(
        inference.term_edf.values()
    ) == pytest.approx(inference.total_edf)


def test_rank_deficient_inference_uses_shared_retained_subspace_solves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout()
    data_curvature = np.diag([2.0, 1.0, 0.0, 1.0, 0.0])
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients))
    result = _result(layout, data_curvature, penalty, source="fisher")

    def forbid_generic_solve(*args, **kwargs):
        raise AssertionError("rank-deficient inference must use RankDecomposition.solve")

    monkeypatch.setattr(scipy.linalg, "solve", forbid_generic_solve)
    monkeypatch.setattr(np.linalg, "inv", forbid_generic_solve)
    inference = compute_joint_inference(layout, result)

    np.testing.assert_allclose(
        inference.covariance,
        np.diag([0.5, 1.0, 0.0, 1.0, 0.0]),
    )
    np.testing.assert_allclose(inference.coefficient_edf, [1.0, 1.0, 0.0, 1.0, 0.0])
    assert inference.rank == 3
    assert inference.curvature_source == "fisher"


def test_exact_face_inference_uses_reduced_covariance_and_solves() -> None:
    """Kills inference through the redundant full-coordinate rank decomposition."""
    layout, ordinary = _full_rank_fixture()
    constrained_slice = layout.term_slices["location:x"]
    component = PenaltyComponent(
        name="location:x#identity",
        group_name="location:x",
        group_index=0,
        group_sl=constrained_slice,
        omega_raw=np.ones((1, 1)),
        omega_ssp=np.ones((1, 1)),
        rank=1.0,
        eigvals_omega=np.ones(1),
    )
    layout = replace(layout, penalties=(component,))
    face = build_penalty_face(layout, (component.name,))
    reduced_rank = decompose_gram(face.reduce_matrix(ordinary.terminal_penalized_curvature))
    result = replace(
        ordinary,
        coefficient_face=face,
        terminal_reduced_rank=reduced_rank,
    )

    basis = face.null_basis
    expected_covariance = basis @ reduced_rank.pseudo_inverse() @ basis.T
    expected_influence = np.column_stack(
        tuple(
            basis @ reduced_rank.solve(basis.T @ result.terminal_data_curvature[:, column])
            for column in range(layout.n_coefficients)
        )
    )
    inference = compute_joint_inference(layout, result)

    np.testing.assert_allclose(inference.covariance, expected_covariance)
    np.testing.assert_allclose(inference.influence, expected_influence)
    constrained_index = constrained_slice.start
    np.testing.assert_array_equal(
        inference.covariance[constrained_index],
        np.zeros(layout.n_coefficients),
    )
    assert inference.coefficient_edf[constrained_index] == 0.0
    assert inference.rank == reduced_rank.rank


def test_inference_rejects_incomplete_or_overlapping_global_slice_partitions() -> None:
    layout, result = _full_rank_fixture()
    missing = replace(
        layout,
        term_slices=MappingProxyType(
            {name: value for name, value in layout.term_slices.items() if name != "location:z"}
        ),
    )
    with pytest.raises(ValueError, match="complete non-overlapping partition"):
        compute_joint_inference(missing, result)

    overlapping_terms = dict(layout.term_slices)
    overlapping_terms["location:x"] = slice(1, 3)
    overlapping = replace(layout, term_slices=MappingProxyType(overlapping_terms))
    with pytest.raises(ValueError, match="complete non-overlapping partition"):
        compute_joint_inference(overlapping, result)


def test_inference_state_rejects_mixed_covariance_and_edf_curvature_sources() -> None:
    layout, result = _full_rank_fixture()
    inference = compute_joint_inference(layout, result)

    with pytest.raises(ValueError, match="same terminal curvature source"):
        replace(inference, edf_curvature_source="fisher")
    with pytest.raises(ValueError, match="recorded actual terminal curvature"):
        replace(
            inference,
            covariance_curvature_source="fisher",
            edf_curvature_source="fisher",
        )


def test_inference_publishes_defensive_arrays_and_mappings() -> None:
    layout, result = _full_rank_fixture()
    inference = compute_joint_inference(layout, result)

    with pytest.raises(ValueError):
        inference.covariance[0, 0] = -1.0
    with pytest.raises(ValueError):
        inference.coefficient_edf[0] = -1.0
    with pytest.raises(TypeError):
        inference.term_edf["location:x"] = -1.0
