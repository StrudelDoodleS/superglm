from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.timing as timing
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.predictor import Predictor
from superglm.distributional.result import DenseSolverConfig, DistributionalEFSConfig
from superglm.distributional.timing import FitPhaseRecorder
from superglm.distributional.weights import WeightContract
from superglm.features import Numeric, Spline
from superglm.types import LambdaPolicy


def test_no_recorder_path_never_reads_phase_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_clock() -> float:
        raise AssertionError("disabled phase timing must not read the clock")

    monkeypatch.setattr(timing, "_clock", forbidden_clock)
    with timing.measure_phase(None, "coefficient_decomposition_solve"):
        value = 1 + 1

    assert value == 2


def test_nested_phase_measurements_accumulate_counts_and_inclusive_seconds() -> None:
    readings = iter((0.0, 1.0, 2.0, 4.0))
    recorder = FitPhaseRecorder(clock=lambda: next(readings))

    with recorder.measure("fit_total"):
        with recorder.measure("predictor_compilation"):
            pass

    snapshot = recorder.snapshot()
    assert snapshot.counts["fit_total"] == 1
    assert snapshot.counts["predictor_compilation"] == 1
    assert snapshot.seconds["fit_total"] == 4.0
    assert snapshot.seconds["predictor_compilation"] == 1.0
    assert snapshot.seconds["fit_total"] >= snapshot.seconds["predictor_compilation"]


def test_snapshot_is_immutable_owned_and_records_manual_samples() -> None:
    recorder = FitPhaseRecorder(clock=lambda: 0.0)
    recorder.add("serialization", 0.25)
    snapshot = recorder.snapshot()
    recorder.add("serialization", 0.75)

    assert snapshot.seconds["serialization"] == 0.25
    assert snapshot.counts["serialization"] == 1
    with pytest.raises(TypeError):
        snapshot.seconds["serialization"] = 2.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        snapshot.seconds = {}  # type: ignore[misc]


def _fixed_fixture() -> tuple[pd.DataFrame, np.ndarray, tuple[Predictor, Predictor]]:
    x = np.linspace(-1.0, 1.0, 48)
    response = 0.4 + 0.7 * x + 0.15 * np.sin(7.0 * x)
    return (
        pd.DataFrame({"x": x}),
        response,
        (
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {}),
        ),
    )


def _efs_fixture() -> tuple[pd.DataFrame, np.ndarray, tuple[Predictor, Predictor]]:
    rng = np.random.default_rng(314159)
    x = np.linspace(0.0, 1.0, 80)
    response = 0.3 + np.sin(2.0 * np.pi * x) + rng.normal(scale=0.25, size=len(x))
    return (
        pd.DataFrame({"x": x}),
        response,
        (
            Predictor(
                "location",
                {
                    "x": Spline(
                        kind="cr",
                        n_knots=6,
                        lambda_policy={"wiggle": LambdaPolicy.estimate()},
                    )
                },
            ),
            Predictor("scale", {}),
        ),
    )


def test_fixed_fit_records_stable_phase_boundaries_without_changing_result() -> None:
    frame, response, predictors = _fixed_fixture()
    baseline = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={},
        config=DenseSolverConfig(tolerance=1.0e-9),
    )
    recorder = FitPhaseRecorder()

    measured = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={},
        config=DenseSolverConfig(tolerance=1.0e-9),
        phase_recorder=recorder,
    )

    np.testing.assert_array_equal(measured.coefficients, baseline.coefficients)
    assert measured.result.history == baseline.result.history
    snapshot = recorder.snapshot()
    for phase in (
        "fit_total",
        "frame_normalization",
        "predictor_compilation",
        "layout_penalty_assembly",
        "dense_predictor_matrices",
        "initialization",
        "likelihood_evaluation",
        "curvature_gradient_assembly",
        "coefficient_decomposition_solve",
        "terminal_observed_retry_fallback",
        "inference_edf",
    ):
        assert snapshot.counts[phase] > 0
        assert snapshot.seconds[phase] >= 0.0
    assert snapshot.counts["efs_update_backtracking"] == 0


def test_complete_disabled_fit_never_reads_phase_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _fixed_fixture()

    def forbidden_clock() -> float:
        raise AssertionError("disabled fit timing must not read the phase clock")

    monkeypatch.setattr(timing, "_clock", forbidden_clock)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={},
    )

    assert model.result.converged is True


def test_efs_fit_threads_one_recorder_through_every_coefficient_refit() -> None:
    frame, response, predictors = _efs_fixture()
    recorder = FitPhaseRecorder()

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        efs_config=DistributionalEFSConfig(
            max_iterations=2,
            tolerance=1.0e-12,
        ),
        phase_recorder=recorder,
    )

    assert model.smoothing is not None
    snapshot = recorder.snapshot()
    assert snapshot.counts["efs_update_backtracking"] > 0
    assert snapshot.counts["initialization"] == len(model.smoothing.coefficient_fits)
    assert snapshot.counts["terminal_observed_retry_fallback"] == len(
        model.smoothing.coefficient_fits
    )
