"""Result assessment authenticates the grouped target, not a re-rooted sum."""

import pickle
from dataclasses import replace
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from superglm.distributional.results import solver
from superglm.distributional.smoothing import endpoint_laml
from superglm.reml import penalty_algebra as algebra
from tests.test_distributional_endpoint_context import _retained_face_problem
from tests.test_distributional_endpoint_laml import _projected_penalty_problem


def _assessment_problem(weak=3.0):
    layout, face, values, _, _ = _projected_penalty_problem()
    selected, left, right = layout.penalties
    matrices = (
        np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    finite = [
        replace(item, omega_raw=matrix, omega_ssp=matrix, rank=1.0)
        for item, matrix in zip((left, right), matrices, strict=True)
    ]
    algebra._attach_context_geometry(finite)
    layout = replace(layout, penalties=(selected, *finite))
    values = dict(zip(layout.penalty_names, (0.0, 1e12, weak), strict=True))
    terminal = SimpleNamespace(log_pdet=0.0)
    result = SimpleNamespace(
        coefficient_face=None,
        penalty=layout.penalty_matrix(values),
        terminal_rank=terminal,
        terminal_reduced_rank=terminal,
        penalized_optimizing_log_likelihood=0.0,
    )
    return layout, face, values, result


@pytest.mark.parametrize("weak", [3.0, 5.0])
@pytest.mark.parametrize("on_face", [False, True])
def test_assessment_replays_the_grouped_logdet_and_derived_error(weak, on_face):
    layout, face, values, result = _assessment_problem(weak)
    context = solver._PenaltyAssessmentContext.from_layout(layout, faces=(face,))
    result.coefficient_face = face if on_face else None
    evaluate = (
        solver._assessment_exact_face_objective if on_face else solver._assessment_finite_objective
    )
    objective, logdet, error, rank = evaluate(result, context=context, lambdas=values)
    assert rank == 2
    with localcontext() as decimal_context:
        decimal_context.prec = 100
        exact = Decimal(4).ln() + Decimal(10**12).ln() + Decimal.from_float(weak).ln()
        assert abs(Decimal.from_float(logdet) - exact) <= Decimal.from_float(error)
        scalar = solver._assessment_scalar_error_bound(
            objective, -0.5 * float(exact), width=len(result.penalty), calculation_scale=abs(logdet)
        )
        assert abs(Decimal.from_float(objective) + exact / 2) <= Decimal.from_float(
            scalar + 0.5 * error
        )


@pytest.mark.parametrize("change", ["weight", "matrix", "face"])
def test_assessment_context_refuses_mismatched_fit_provenance(change):
    layout, face, values, result = _assessment_problem()
    context = solver._PenaltyAssessmentContext.from_layout(layout, faces=(face,))
    if change == "weight":
        values = dict(values)
        values[layout.penalty_names[-1]] = 7.0
    elif change == "matrix":
        result.penalty = result.penalty + np.eye(len(result.penalty))
    else:
        rotation = np.eye(face.reduced_width)
        rotation[:2, :2] = [[0.0, 1.0], [-1.0, 0.0]]
        result.coefficient_face = replace(face, null_basis=face.null_basis @ rotation)
    evaluate = (
        solver._assessment_finite_objective
        if result.coefficient_face is None
        else solver._assessment_exact_face_objective
    )
    with pytest.raises(ValueError, match="penalty|face|context"):
        evaluate(result, context=context, lambdas=values)


def test_assessment_snapshot_preserves_the_fixed_ssp_target_and_ignores_source_mutation():
    source, _, layout, face = _retained_face_problem()
    values = dict(zip(layout.penalty_names, (0.0, 2.0, 0.0), strict=True))
    reference = endpoint_laml._projected_finite_penalty_evaluation(
        layout=layout, lambdas=values, face=face
    )
    context = solver._PenaltyAssessmentContext.from_layout(layout, faces=(face,))
    result = SimpleNamespace(
        coefficient_face=face,
        penalty=layout.penalty_matrix(values),
        terminal_rank=SimpleNamespace(log_pdet=0.0),
        terminal_reduced_rank=SimpleNamespace(log_pdet=0.0),
        penalized_optimizing_log_likelihood=0.0,
    )
    layout.penalties[1].omega_ssp = np.eye(5)
    objective, logdet, error, rank = solver._assessment_exact_face_objective(
        result, context=context, lambdas=values
    )
    assert logdet == reference.logdet
    assert error == reference.logdet_error
    assert objective == -0.5 * reference.logdet
    assert rank == reference.rank
    assert algebra._context_geometry(source) is not None


def test_assessment_snapshot_does_not_accept_changed_owned_component_metadata():
    layout, face, values, result = _assessment_problem()
    context = solver._PenaltyAssessmentContext.from_layout(layout, faces=(face,))
    context.penalties[1].group_sl = slice(0, 1)
    with pytest.raises(ValueError, match="context"):
        solver._assessment_finite_objective(result, context=context, lambdas=values)


def test_assessment_snapshot_can_rebind_an_exact_already_qualified_family():
    _, _, layout, _ = _retained_face_problem()
    copied = tuple(replace(item) for item in layout.penalties)
    algebra._rebind_penalty_context(layout.penalties, copied)
    source = algebra._context_geometry(list(layout.penalties[1:]))
    held = algebra._context_geometry(list(copied[1:]))
    assert source is not None and held is not None and source is not held
    assert source.support is held.support
    assert source.coordinate_map is held.coordinate_map
    assert source.keys == held.keys


def _issue_receipt(layout, result, values, evaluation=None):
    if evaluation is None:
        evaluation = algebra._compute_penalty_logdet_evaluation(values, list(layout.penalties))
    rank = result.terminal_rank if result.coefficient_face is None else result.terminal_reduced_rank
    objective = -float(result.penalized_optimizing_log_likelihood) + 0.5 * (
        float(rank.log_pdet) - evaluation.logdet
    )
    solver._record_penalty_objective(
        layout, result, lambdas=values, evaluation=evaluation, objective=objective
    )
    return objective, evaluation


def test_assessment_retains_larger_original_error_after_a_smaller_duplicate_receipt():
    layout, face, values, result = _assessment_problem()
    evaluation = algebra._compute_penalty_logdet_evaluation(values, list(layout.penalties))
    larger = np.nextafter(evaluation.logdet_error, np.inf)
    original, _ = _issue_receipt(layout, result, values, replace(evaluation, logdet_error=larger))
    _issue_receipt(layout, result, values, evaluation)
    assert len(layout._penalty_objective_receipts) == 1
    context = solver._PenaltyAssessmentContext.from_layout(layout, faces=(face,))
    assert (
        context.original_error(result, lambdas=values, rank=evaluation.rank, objective=original)
        == larger
    )
    replay = solver._assessment_finite_objective(result, context=context, lambdas=values)
    bound = solver._assessment_objective_error_bound(
        replay, result=result, original_objective=original, context=context, lambdas=values
    )
    scalar = solver._assessment_scalar_error_bound(
        replay[0], original, width=len(result.penalty), calculation_scale=abs(replay[1])
    )
    assert bound >= scalar + 0.5 * (larger + replay[2])


@pytest.mark.parametrize("lost", ["missing", "layout_copy", "value", "activity", "target"])
def test_assessment_refuses_missing_or_mismatched_original_evidence(lost):
    layout, face, values, result = _assessment_problem()
    original, evaluation = _issue_receipt(layout, result, values)
    if lost == "missing":
        layout._penalty_objective_receipts.clear()
    if lost == "layout_copy":
        layout = replace(layout)
        assert not layout._penalty_objective_receipts
    context = solver._PenaltyAssessmentContext.from_layout(layout, faces=(face,))
    if lost == "value":
        original = np.nextafter(original, np.inf)
    elif lost == "activity":
        values = dict(values)
        values[layout.penalty_names[-1]] = 0.0
    elif lost == "target":
        # Matrix binding alone is insufficient: a different retained range is
        # a different target even when its rounded SSP matrices are unchanged.
        geometry = algebra._context_geometry(list(context.penalties[1:]))
        geometry.support = replace(geometry.support, Q_plus=geometry.support.Q_plus[:, :1])
    with pytest.raises(ValueError, match="original objective evidence"):
        context.original_error(result, lambdas=values, rank=evaluation.rank, objective=original)


def test_assessment_refuses_an_original_receipt_after_changing_the_common_map():
    _, _, layout, face = _retained_face_problem()
    values = dict(zip(layout.penalty_names, (0.0, 2.0, 3.0), strict=True))
    result = SimpleNamespace(
        coefficient_face=None,
        penalty=layout.penalty_matrix(values),
        terminal_rank=SimpleNamespace(log_pdet=0.0),
        penalized_optimizing_log_likelihood=0.0,
    )
    original, evaluation = _issue_receipt(layout, result, values)
    context = solver._PenaltyAssessmentContext.from_layout(layout, faces=(face,))
    geometry = algebra._context_geometry(list(context.penalties[1:]))
    geometry.coordinate_map = algebra._frozen_array(np.eye(5))
    with pytest.raises(ValueError, match="original objective evidence"):
        context.original_error(result, lambdas=values, rank=evaluation.rank, objective=original)


def test_re_rooting_the_weighted_matrix_still_fails_the_original_objective_comparison():
    layout, face, values, result = _assessment_problem()
    original, _ = _issue_receipt(layout, result, values)
    context = solver._PenaltyAssessmentContext.from_layout(layout, faces=(face,))
    replay = solver._assessment_finite_objective(result, context=context, lambdas=values)
    bound = solver._assessment_objective_error_bound(
        replay, result=result, original_objective=original, context=context, lambdas=values
    )
    assert abs(replay[0] - original) <= bound
    # Restoring the no-context weighted-Gram path must expose the old defect.
    old = solver._assessment_finite_objective(result)
    assert abs(old[0] - original) > bound


def test_scalar_receipts_authenticate_a_rejected_objective_without_re_evaluating_penalties(
    monkeypatch,
):
    layout, face, values, result = _assessment_problem()
    original, _ = _issue_receipt(layout, result, values)
    context = solver._PenaltyAssessmentContext.from_layout(layout, faces=(face,))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("historical literal decision re-evaluated its penalty")

    monkeypatch.setattr(algebra, "_compute_penalty_logdet_evaluation", forbidden)
    assert context.original_objectives(result, lambdas=values) == (original,)
    assert all(
        value > original - 1 for value in context.original_objectives(result, lambdas=values)
    )


def test_assessment_context_preserves_its_bindings_and_receipts_in_protocol_five_serialization():
    from superglm.distributional.serialization import _pickle_model

    layout, face, values, result = _assessment_problem()
    original, evaluation = _issue_receipt(layout, result, values)
    context = solver._PenaltyAssessmentContext.from_layout(layout, faces=(face,))
    restored = pickle.loads(_pickle_model(context))
    restored._validate()
    assert restored.original_error(
        result, lambdas=values, rank=evaluation.rank, objective=original
    ) == context.original_error(result, lambdas=values, rank=evaluation.rank, objective=original)
    assert restored.evaluate(result, values) == context.evaluate(result, values)


def test_actual_endpoint_history_requires_its_producer_receipts(monkeypatch):
    from superglm.distributional.smoothing import objective as objective_module
    from tests.test_distributional_endpoint_laml import _independent_penalty_efs_fit

    layout, smoothing, _ = _independent_penalty_efs_fit(monkeypatch, (0.5, 0.5))
    context = smoothing._penalty_assessment_context
    assert context is not None and context._receipts
    assert context._receipts == layout._penalty_objective_receipts
    assert context._receipts is not layout._penalty_objective_receipts
    assert any(item.activated_face_components for item in smoothing.history)

    # Dropping only the producer evidence must prevent the same history from
    # acquiring an authenticated assessment through a current-only error.
    monkeypatch.setattr(objective_module, "_record_penalty_objective", lambda *_a, **_k: None)
    monkeypatch.setattr(endpoint_laml, "_record_penalty_objective", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="invalid original evidence"):
        _independent_penalty_efs_fit(monkeypatch, (0.5, 0.5))


def test_float32_manual_target_has_the_same_canonical_receipt_after_snapshot():
    layout, face, values, result = _assessment_problem()
    copied = tuple(
        replace(
            item,
            omega_raw=None if item.omega_raw is None else item.omega_raw.astype(np.float32),
            omega_ssp=None if item.omega_ssp is None else item.omega_ssp.astype(np.float32),
        )
        for item in layout.penalties
    )
    layout = replace(layout, penalties=copied)
    original, evaluation = _issue_receipt(layout, result, values)
    context = solver._PenaltyAssessmentContext.from_layout(layout, faces=(face,))
    assert (
        context.original_error(result, lambdas=values, rank=evaluation.rank, objective=original)
        == evaluation.logdet_error
    )
