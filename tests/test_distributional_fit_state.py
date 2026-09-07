from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.fit_state as fit_state_module
import superglm.distributional.model as model_module
import superglm.distributional.solver.chunks as chunking
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.model import (
    fit_dense_distributional,
    refit_dense_distributional,
)
from superglm.distributional.predictor import Predictor
from superglm.distributional.weights import WeightContract
from superglm.features import Numeric, Spline


def _fixture() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, tuple[Predictor, ...]]:
    rng = np.random.default_rng(9103)
    x = np.linspace(-1.0, 1.0, 96)
    z = np.mod(0.2 + 1.4 * x, 1.0)
    sigma = 0.22 + np.exp(-1.25 + 0.25 * np.cos(2.0 * np.pi * z))
    response = 0.7 + 0.55 * x + rng.normal(scale=sigma)
    frame = pd.DataFrame({"x": x, "z": z})
    weights = np.linspace(0.6, 1.8, len(frame))
    predictors = (
        Predictor("location", {"x": Numeric()}),
        Predictor("scale", {"z": Spline(kind="cr", n_knots=5)}),
    )
    return frame, response, weights, predictors


def _fit(*, retain_rows: bool = True):
    frame, response, weights, predictors = _fixture()
    offsets = {"location": np.linspace(-0.08, 0.11, len(frame))}
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.025),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        sample_weight=weights,
        offsets=offsets,
        lambdas={"scale:z#wiggle": 0.4},
        retain_rows=retain_rows,
    )
    return model, frame, response, weights


def test_complete_compact_fit_state_is_published_from_one_terminal_revision() -> None:
    model, _, _, _ = _fit()
    state = model.fit_state
    fitted = model.fitted_result

    assert state.revision == 1
    assert fitted is state.result
    np.testing.assert_array_equal(fitted.coefficients, model.result.coefficients)
    assert fitted.coefficient_names == model.layout.coefficient_names
    assert fitted.parameter_names == ("location", "scale")
    assert tuple(fitted.predictor_coefficients) == fitted.parameter_names
    for predictor in model.layout.predictors:
        np.testing.assert_array_equal(
            fitted.predictor_coefficients[predictor.name],
            fitted.coefficients[predictor.coefficient_slice],
        )
    assert fitted.smoothing_parameters == model.lambdas
    np.testing.assert_array_equal(fitted.covariance, state.inference.covariance)
    assert fitted.total_effective_df == pytest.approx(state.inference.total_edf)
    assert fitted.predictor_edf == state.inference.predictor_edf
    assert fitted.term_edf == state.inference.term_edf
    assert fitted.null_objective == pytest.approx(state.null_model.objective)
    assert fitted.coefficient_converged is model.result.converged
    assert fitted.smoothing_converged is None
    assert fitted.curvature_telemetry is model.result.terminal_curvature
    assert state.weight_contract == WeightContract(semantics="prior")
    assert state.weight_provenance.contract == state.weight_contract
    assert state.family_likelihood_plan_identifier == (
        model.result.family_likelihood_plan_identifier
    )
    assert state.null_model.family_likelihood_plan_identifier == (
        state.family_likelihood_plan_identifier
    )
    assert state.null_model.weight_contract == state.weight_contract
    assert state.null_model.weight_provenance == state.weight_provenance
    assert state.null_model.weight_semantics == "prior"
    assert state.requested_discrete is False
    assert state.requested_n_bins == 256
    assert state.requested_chunk_size is None
    assert state.exact_face_components == ()


def test_compact_fit_state_refuses_a_face_claim_without_an_accepted_solver_face() -> None:
    model, _, _, _ = _fit()
    state = model.fit_state
    component_name = state.layout.penalty_names[0]

    with pytest.raises(ValueError, match="exact face.*accepted terminal solver face"):
        replace(state, exact_face_components=(component_name,))


def test_compact_rank_is_derived_from_and_validated_against_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_inference = fit_state_module.compute_joint_inference

    def lower_inference_rank(*args, **kwargs):
        inference = real_inference(*args, **kwargs)
        assert inference.rank > 0
        return replace(inference, rank=inference.rank - 1)

    monkeypatch.setattr(fit_state_module, "compute_joint_inference", lower_inference_rank)
    model, _, _, _ = _fit()
    state = model.fit_state

    assert state.result.rank == state.inference.rank
    with pytest.raises(ValueError, match="compact and inference rank"):
        replace(state, result=replace(state.result, rank=state.result.rank + 1))


def test_published_coefficient_covariance_predictor_and_telemetry_views_are_immutable() -> None:
    model, _, _, _ = _fit()

    with pytest.raises(ValueError):
        model.coefficients[0] = 0.0
    with pytest.raises(ValueError):
        model.covariance[0, 0] = 0.0
    with pytest.raises(ValueError):
        model.predictor_coefficients["location"][0] = 0.0
    with pytest.raises(TypeError):
        model.predictor_coefficients["location"] = np.zeros(2)
    with pytest.raises(TypeError):
        model.smoothing_parameters["scale:z#wiggle"] = 2.0
    with pytest.raises(FrozenInstanceError):
        model.telemetry.fallback_count = 100


def test_failed_refit_leaves_the_previous_accepted_revision_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, frame, response, weights = _fit()
    accepted_state = model.fit_state
    accepted_result = model.result
    accepted_prediction = model.predict(frame).copy()

    def fail_terminal_inference(*args, **kwargs):
        raise RuntimeError("terminal inference sentinel")

    monkeypatch.setattr(fit_state_module, "compute_joint_inference", fail_terminal_inference)
    with pytest.raises(RuntimeError, match="terminal inference sentinel"):
        refit_dense_distributional(
            model,
            frame,
            response + 0.15,
            sample_weight=weights,
        )

    assert model.fit_state is accepted_state
    assert model.result is accepted_result
    np.testing.assert_array_equal(model.predict(frame), accepted_prediction)


def test_successful_refit_swaps_one_complete_revision_after_validation() -> None:
    model, frame, response, weights = _fit()
    previous = model.fit_state
    previous_coefficients = model.coefficients.copy()
    assert previous.retained_rows is not None
    previous_carrier = previous.retained_rows.likelihood_weights

    returned = refit_dense_distributional(
        model,
        frame,
        response + 0.35 + 0.1 * frame["x"].to_numpy(),
        sample_weight=weights,
    )

    assert returned is model
    assert model.fit_state is not previous
    assert model.fit_state.revision == previous.revision + 1
    assert not np.allclose(model.coefficients, previous_coefficients)
    np.testing.assert_array_equal(previous.result.coefficients, previous_coefficients)
    assert model.fitted_result.null_objective == pytest.approx(model.null_model.objective)
    assert model.fit_state.retained_rows is not None
    assert model.fit_state.retained_rows.likelihood_weights is not previous_carrier


def test_retained_row_state_is_separate_from_compact_inference_state() -> None:
    retained, _, _, _ = _fit(retain_rows=True)
    compact_only, _, _, _ = _fit(retain_rows=False)

    rows = retained.fit_state.retained_rows
    assert rows is not None
    assert rows.response.shape == (96,)
    assert rows.likelihood_weights.values.shape == (96,)
    assert rows.likelihood_weights.provenance == retained.fit_state.weight_provenance
    assert rows.likelihood_weights.digest == retained.fit_state.weight_provenance.root_digest
    assert rows.fitted_eta.shape == (96, 2)
    assert rows.fitted_parameters.shape == (96, 2)
    assert rows.null_eta.shape == (96, 2)
    assert rows.null_parameters.shape == (96, 2)
    assert compact_only.fit_state.retained_rows is None
    assert compact_only.fit_state.weight_provenance.original_count == 96
    assert compact_only.fit_state.weight_provenance.retained_count == 96
    assert compact_only.fit_state.weight_provenance.root_digest
    assert not hasattr(compact_only.fitted_result, "eta")
    assert not hasattr(compact_only.null_model, "offsets")
    assert not hasattr(compact_only.null_model, "sample_weight")
    np.testing.assert_allclose(compact_only.coefficients, retained.coefficients)
    np.testing.assert_allclose(compact_only.covariance, retained.covariance)
    assert compact_only.null_model.offset_semantics == {
        "location": "nonzero_training_offset",
        "scale": "zero_offset",
    }


def _linear_problem(n: int) -> tuple[pd.DataFrame, np.ndarray, tuple[Predictor, ...]]:
    x = np.linspace(-1.0, 1.0, n)
    response = 0.4 + 0.65 * x + 0.17 * np.sin(np.arange(n, dtype=np.float64))
    return (
        pd.DataFrame({"x": x}),
        response,
        (
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {}),
        ),
    )


def test_frequency_zero_drop_publishes_compact_root_provenance_and_retained_carrier() -> None:
    frame, response, predictors = _linear_problem(12)
    counts = np.array([2, 0, 1, 3, 0, 2, 1, 4, 1, 2, 0, 3])

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.01),
        predictors=predictors,
        weight_contract=WeightContract(semantics="frequency"),
        sample_weight=counts,
        lambdas={},
    )
    state = model.fit_state
    provenance = state.weight_provenance
    rows = state.retained_rows

    assert provenance.contract == state.weight_contract == WeightContract("frequency")
    assert provenance.original_count == len(frame)
    assert provenance.retained_count == int(np.count_nonzero(counts))
    assert provenance.dropped_count == int(np.count_nonzero(counts == 0))
    assert provenance.likelihood_count == int(np.sum(counts))
    assert provenance.dropped_positions_digest
    assert rows is not None
    np.testing.assert_array_equal(rows.likelihood_weights.input_positions, np.flatnonzero(counts))
    np.testing.assert_array_equal(rows.likelihood_weights.values, counts[counts > 0])
    assert rows.likelihood_weights.provenance is provenance
    assert state.family_likelihood_plan_identifier == (
        state.solver_result.family_likelihood_plan_identifier
    )
    assert state.null_model.family_likelihood_plan_identifier == (
        state.family_likelihood_plan_identifier
    )
    assert state.null_model.weight_contract is state.weight_contract
    assert state.null_model.weight_provenance is provenance


def test_refit_inherits_requested_policy_but_recomputes_auto_chunk_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _linear_problem(18)
    counts = np.array([2, 1, 3, 1, 2, 4, 1, 1, 2, 3, 1, 2, 1, 4, 2, 1, 3, 2])
    monkeypatch.setattr(chunking, "AUTO_CHUNK_MEMORY_BYTES", 8 * 1024 * 1024)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.01),
        predictors=predictors,
        weight_contract=WeightContract(semantics="frequency"),
        sample_weight=counts,
        lambdas={},
        discrete=True,
        n_bins={"x": 9},
        chunk_size="auto",
    )
    previous = model.fit_state
    previous_root = previous.weight_provenance.root_digest
    previous_plan = previous.family_likelihood_plan_identifier
    assert previous.requested_chunk_size == "auto"
    assert previous.solver_result.resolved_chunk_size == len(frame)

    new_frame, new_response, _ = _linear_problem(27)
    main_chunk_requests: list[object] = []
    chunk_calls = 0
    original_fit = model_module.fit_dense_fixed_lambda
    original_chunked = chunking.evaluate_chunked_log_likelihood

    def record_fit(*args, **kwargs):
        main_chunk_requests.append(kwargs.get("chunk_size"))
        return original_fit(*args, **kwargs)

    def count_chunked(*args, **kwargs):
        nonlocal chunk_calls
        chunk_calls += 1
        return original_chunked(*args, **kwargs)

    monkeypatch.setattr(model_module, "fit_dense_fixed_lambda", record_fit)
    monkeypatch.setattr(chunking, "evaluate_chunked_log_likelihood", count_chunked)
    monkeypatch.setattr(chunking, "AUTO_CHUNK_MEMORY_BYTES", 256)

    refit_dense_distributional(model, new_frame, new_response)
    current = model.fit_state

    assert main_chunk_requests == ["auto"]
    assert chunk_calls > 0
    assert current.weight_contract == WeightContract("frequency")
    assert current.weight_provenance.all_unit is True
    assert current.weight_provenance.root_digest != previous_root
    assert current.family_likelihood_plan_identifier != previous_plan
    assert current.retained_rows is not None
    assert previous.retained_rows is not None
    assert current.retained_rows.likelihood_weights is not previous.retained_rows.likelihood_weights
    assert current.requested_discrete is True
    assert current.requested_n_bins == {"x": 9}
    assert current.requested_chunk_size == "auto"
    assert current.solver_result.resolved_chunk_size == 1
    assert current.solver_result.execution_backend_identifier == ("distributional-chunked-v1")
    assert previous.solver_result.resolved_chunk_size == len(frame)


def test_frequency_refit_matches_fresh_frequency_and_literal_row_expansion() -> None:
    frame, response, predictors = _linear_problem(10)
    initial_counts = np.array([1, 2, 1, 3, 2, 1, 4, 1, 2, 1])
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.01),
        predictors=predictors,
        weight_contract=WeightContract("frequency"),
        sample_weight=initial_counts,
        lambdas={},
    )

    new_frame = frame.iloc[::-1].reset_index(drop=True)
    new_response = response[::-1] + np.linspace(-0.05, 0.08, len(response))
    new_counts = np.array([3, 1, 2, 4, 1, 2, 1, 3, 2, 1])
    refit_dense_distributional(
        model,
        new_frame,
        new_response,
        sample_weight=new_counts,
    )
    fresh = fit_dense_distributional(
        new_frame,
        new_response,
        family=GaussianLS(scale_floor=0.01),
        predictors=predictors,
        weight_contract=WeightContract("frequency"),
        sample_weight=new_counts,
        lambdas={},
    )
    positions = np.repeat(np.arange(len(new_frame)), new_counts)
    expanded = fit_dense_distributional(
        new_frame.iloc[positions].reset_index(drop=True),
        new_response[positions],
        family=GaussianLS(scale_floor=0.01),
        predictors=predictors,
        weight_contract=WeightContract("prior"),
        lambdas={},
    )

    assert model.fit_state.weight_contract == WeightContract("frequency")
    np.testing.assert_allclose(model.coefficients, fresh.coefficients, rtol=0.0, atol=2e-11)
    np.testing.assert_allclose(model.coefficients, expanded.coefficients, rtol=0.0, atol=2e-11)


def test_equal_row_count_refit_never_reuses_training_offsets() -> None:
    frame, response, predictors = _linear_problem(20)
    training_offsets = {"location": np.linspace(-0.8, 0.9, len(frame))}
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.01),
        predictors=predictors,
        weight_contract=WeightContract("prior"),
        offsets=training_offsets,
        lambdas={},
    )
    new_frame = frame.iloc[::-1].reset_index(drop=True)
    new_response = 1.1 - 0.3 * new_frame["x"].to_numpy() + 0.1 * np.cos(np.arange(len(frame)))

    refit_dense_distributional(model, new_frame, new_response)
    fresh = fit_dense_distributional(
        new_frame,
        new_response,
        family=GaussianLS(scale_floor=0.01),
        predictors=predictors,
        weight_contract=WeightContract("prior"),
        lambdas={},
    )

    np.testing.assert_allclose(model.coefficients, fresh.coefficients, rtol=0.0, atol=2e-11)
    assert model.fit_state.retained_rows is not None
    np.testing.assert_array_equal(
        model.fit_state.retained_rows.offsets["location"],
        np.zeros(len(new_frame)),
    )


def test_equal_row_count_refit_with_omitted_weights_builds_a_fresh_unit_root() -> None:
    frame, response, predictors = _linear_problem(12)
    initial_counts = np.array([2, 1, 4, 2, 3, 1, 2, 5, 1, 3, 2, 4])
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.01),
        predictors=predictors,
        weight_contract=WeightContract("frequency"),
        sample_weight=initial_counts,
        lambdas={},
    )
    previous = model.fit_state
    assert previous.retained_rows is not None
    previous_carrier = previous.retained_rows.likelihood_weights

    new_frame = frame.iloc[::-1].reset_index(drop=True)
    new_response = response[::-1] + np.linspace(-0.07, 0.09, len(response))
    refit_dense_distributional(model, new_frame, new_response)
    fresh = fit_dense_distributional(
        new_frame,
        new_response,
        family=GaussianLS(scale_floor=0.01),
        predictors=predictors,
        weight_contract=WeightContract("frequency"),
        lambdas={},
    )

    current = model.fit_state
    assert current.retained_rows is not None
    assert fresh.fit_state.retained_rows is not None
    current_carrier = current.retained_rows.likelihood_weights
    fresh_carrier = fresh.fit_state.retained_rows.likelihood_weights
    assert current_carrier is not previous_carrier
    assert current_carrier is not fresh_carrier
    np.testing.assert_array_equal(current_carrier.values, np.ones(len(new_frame)))
    assert current_carrier.provenance.all_unit is True
    assert current_carrier.provenance.likelihood_count == len(new_frame)
    assert current_carrier.provenance.contract == WeightContract("frequency")
    assert current_carrier.digest != previous_carrier.digest
    assert current_carrier.digest == fresh_carrier.digest
    np.testing.assert_allclose(model.coefficients, fresh.coefficients, rtol=0.0, atol=2e-11)
