"""Contract tests for ``SuperLSS.diagnose()`` and its fit work profile."""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm
from superglm import SuperLSS
from superglm.diagnostics.fit_report import (
    FitDiagnosticReport,
    FitPhaseProfile,
    FitWorkProfile,
    SmoothingComponentProfile,
)
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.fit_diagnostics import diagnose_distributional_fit
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.predictor import Predictor
from superglm.distributional.result import DenseSolverConfig, DistributionalEFSConfig
from superglm.distributional.timing import FitPhaseRecorder, FitPhaseSnapshot
from superglm.distributional.weights import WeightContract
from superglm.features import Numeric, RandomEffect, Spline
from superglm.types import LambdaPolicy


@pytest.fixture(scope="module")
def profiled_face_fit() -> tuple[SuperLSS, FitPhaseSnapshot]:
    levels = np.array([-100.0, 0.0, 100.0])
    x = np.repeat(levels, 6)
    z = np.tile(np.repeat(levels, 2), 3)
    sign = np.tile(np.array([-1.0, 1.0]), 9)
    response = 0.4 + 0.006 * x + np.exp(-1.2 + 0.003 * z) * sign
    ticks = itertools.count()
    recorder = FitPhaseRecorder(clock=lambda: next(ticks) * 0.001)
    model = SuperLSS(
        family=GaussianLS(scale_floor=0.0),
        predictors=(
            Predictor("location", {"x": Spline(kind="cr", k=3)}),
            Predictor("scale", {"z": Spline(kind="cr", k=3)}),
        ),
    ).fit_reml(
        pd.DataFrame({"x": x, "z": z}),
        response,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.3},
        max_reml_iter=60,
        reml_tol=1.0e-8,
        inner_tol=1.0e-10,
        retain_rows=False,
        phase_recorder=recorder,
    )
    return model, recorder.snapshot()


def _fixed_model() -> SuperLSS:
    return SuperLSS(
        family=GaussianLS(scale_floor=0.01),
        predictors=(
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {}),
        ),
    )


def _phase(name: str, seconds: float, fit_seconds: float, calls: int) -> FitPhaseProfile:
    return FitPhaseProfile(
        name=name,
        seconds=seconds,
        fit_share=seconds / fit_seconds,
        calls=calls,
    )


def _component(**overrides) -> SmoothingComponentProfile:
    values = dict(
        name="location:x#wiggle",
        predictor="location",
        term="x",
        initial_lambda=0.3,
        final_lambda=1.0e10,
        accepted_moves=5,
        dominant_update_share=0.5,
        terminal_term_edf=1.0,
        null_space_dimension=1,
        outcome="exact_face",
        exact_face_iteration=6,
        exact_face_effect="linear_only",
        upper_bound_iterations=1,
    )
    values.update(overrides)
    return SmoothingComponentProfile(**values)


def _profile(**overrides) -> FitWorkProfile:
    fit_seconds = 0.25
    values = dict(
        n_observations=18,
        n_coefficients=6,
        fit_seconds=fit_seconds,
        outer_iterations=7,
        coefficient_fits=8,
        inner_iterations=8,
        rejected_proposals=0,
        backtracked_proposals=0,
        phases=(
            _phase("likelihood_evaluation", 0.1, fit_seconds, 24),
            _phase("coefficient_decomposition_solve", 0.05, fit_seconds, 16),
            _phase("orchestration_and_unmeasured", 0.1, fit_seconds, 0),
        ),
        smoothing_components=(_component(),),
    )
    values.update(overrides)
    return FitWorkProfile(**values)


def _report(**overrides) -> FitDiagnosticReport:
    values = dict(
        schema_version=2,
        rule_set_version=1,
        model_type="SuperLSS",
        family="GaussianLS",
        fit_revision=1,
        scope="fit",
        fit_status="converged_uncertified",
        findings=(),
        coverage=("Retained fit telemetry was examined.",),
        limitations=("No row evidence was examined.",),
        profile=_profile(),
    )
    values.update(overrides)
    return FitDiagnosticReport(**values)


def test_profile_rejects_inconsistent_phase_shares() -> None:
    profile = _profile()
    likelihood = next(item for item in profile.phases if item.name == "likelihood_evaluation")

    with pytest.raises(ValueError, match="fit_share must not exceed one"):
        replace(likelihood, fit_share=1.01)

    inconsistent = replace(likelihood, fit_share=likelihood.fit_share / 2.0)
    phases = tuple(
        inconsistent if item.name == likelihood.name else item for item in profile.phases
    )
    with pytest.raises(ValueError, match="phase fit_share"):
        replace(profile, phases=phases)


def test_profile_accepts_signed_term_edf_and_renders_shared_scope_once() -> None:
    report = _report()
    assert report.profile is not None
    component = report.profile.smoothing_components[0]
    first = replace(
        component,
        terminal_term_edf=-0.125,
        outcome="finite",
        exact_face_iteration=None,
        exact_face_effect=None,
    )
    second = replace(first, name=f"{first.name}-second")
    profiled = replace(
        report,
        profile=replace(report.profile, smoothing_components=(first, second)),
    )

    rendered = profiled.render()
    assert rendered.count("-0.125") == 1
    assert "Term EDF is shared" in rendered
    with pytest.raises(ValueError, match="schema_version=1"):
        replace(profiled, schema_version=1)


def test_profile_does_not_call_a_fixed_lambda_a_search_boundary() -> None:
    levels = np.array([-100.0, 0.0, 100.0])
    x = np.repeat(levels, 6)
    z = np.tile(np.repeat(levels, 2), 3)
    sign = np.tile(np.array([-1.0, 1.0]), 9)
    response = 0.4 + 0.006 * x + np.exp(-1.2 + 0.003 * z) * sign
    model = SuperLSS(
        family=GaussianLS(scale_floor=0.0),
        predictors=(
            Predictor(
                "location",
                {
                    "x": Spline(
                        kind="cr",
                        k=3,
                        lambda_policy=LambdaPolicy.fixed(1.0e10),
                    )
                },
            ),
            Predictor("scale", {"z": Spline(kind="cr", k=3)}),
        ),
    ).fit_reml(
        pd.DataFrame({"x": x, "z": z}),
        response,
        lambdas={"scale:z#wiggle": 0.3},
        max_reml_iter=4,
        retain_rows=False,
    )

    report = model.diagnose()
    assert report.profile is not None
    fixed = next(
        component
        for component in report.profile.smoothing_components
        if component.name == "location:x#wiggle"
    )

    assert fixed.final_lambda == 1.0e10
    assert fixed.outcome == "fixed"
    assert fixed.accepted_moves == 0
    assert fixed.upper_bound_iterations == 0
    rendered = report.render()
    assert "fixed by caller" in rendered
    assert "at upper bound" not in rendered


def test_diagnose_leads_with_work_timing_and_smoothing_metrics(
    profiled_face_fit,
) -> None:
    model, measured = profiled_face_fit
    fitted = model._require_fitted()

    report = model.diagnose()
    profile = report.profile

    assert isinstance(report, FitDiagnosticReport)
    assert report == diagnose_distributional_fit(fitted, phase_snapshot=measured)
    assert report.schema_version == 2
    assert profile is not None
    assert profile.n_observations == 18
    assert profile.n_coefficients == len(model.coef_)
    assert profile.fit_seconds == measured.seconds["fit_total"]
    assert profile.outer_iterations == model.result_.n_smoothing_iter
    assert profile.inner_iterations == model.result_.n_inner_iter
    assert fitted.smoothing is not None
    assert profile.coefficient_fits == len(fitted.smoothing.coefficient_fits)
    assert profile.backtracked_proposals >= profile.rejected_proposals
    likelihood = next(item for item in profile.phases if item.name == "likelihood_evaluation")
    assert likelihood.calls == measured.counts["likelihood_evaluation"]
    assert likelihood.seconds == measured.seconds["likelihood_evaluation"]
    assert likelihood.fit_share == pytest.approx(likelihood.seconds / profile.fit_seconds)
    assert sum(item.fit_share for item in profile.phases) == pytest.approx(1.0)
    assert {item.name for item in profile.phases}.isdisjoint(
        {"initialization", "terminal_observed_retry_fallback"}
    )
    assert any(item.name == "orchestration_and_unmeasured" for item in profile.phases)
    terminal = next(
        item for item in profile.phases if item.name == "terminal_inference_and_null_fit"
    )
    assert terminal.seconds == measured.seconds["inference_edf"]
    assert all(item.name != "inference_edf" for item in profile.phases)
    dense = next(item for item in profile.phases if item.name == "dense_predictor_matrices")
    assert dense.calls == measured.counts["dense_predictor_matrices"]
    assert dense.seconds == measured.seconds["dense_predictor_matrices"]
    assert report.findings == diagnose_distributional_fit(fitted).findings

    rendered = report.render()
    assert rendered.startswith("SuperLSS fit diagnosis")
    assert "Fit time:" in rendered
    assert "Work" in rendered
    assert "Outer EFS iterations" in rendered
    assert "Coefficient fits" in rendered
    assert "Time distribution" in rendered
    assert "Smoothing parameters" in rendered


def test_exact_face_profile_names_the_driving_terms_without_contradicting_itself(
    profiled_face_fit,
) -> None:
    model, _measured = profiled_face_fit

    report = model.diagnose()
    assert report.profile is not None
    components = {item.name: item for item in report.profile.smoothing_components}

    assert set(components) == {"location:x#wiggle", "scale:z#wiggle"}
    for name, component in components.items():
        assert component.initial_lambda == 0.3
        assert component.final_lambda == model.result_.smoothing_parameters[name]
        assert component.outcome == "exact_face"
        assert component.exact_face_iteration is not None
        assert component.null_space_dimension == 1
        assert component.terminal_term_edf == model.result_.term_edf[name.removesuffix("#wiggle")]
        assert component.exact_face_effect == "linear_only"
        assert 0.0 <= component.dominant_update_share <= 1.0

    codes = {finding.code for finding in report.findings}
    assert "smoothing.penalized_subspace_suppressed" in codes
    rendered = report.render()
    assert "linear only" in rendered
    assert "Lead share" in rendered
    assert "Cap iters" in rendered
    assert "Subject:" not in rendered
    assert "Code:" not in rendered


def test_exact_face_profile_distinguishes_fully_penalized_random_effects() -> None:
    groups = np.tile(np.array(["a", "b", "c"]), 8)
    response = np.repeat(np.array([0.7, 0.9, 1.1, 1.3, 0.8, 1.2, 1.0, 1.0]), 3)
    model = fit_dense_distributional(
        pd.DataFrame({"group": groups}),
        response,
        family=GaussianLS(),
        predictors=(
            Predictor("location", {"group": RandomEffect()}),
            Predictor("scale", {}),
        ),
        weight_contract=WeightContract("prior"),
        lambdas={"location:group#wiggle": 10.0},
        config=DenseSolverConfig(tolerance=1.0e-10, max_iterations=150),
        efs_config=DistributionalEFSConfig(
            max_iterations=8,
            tolerance=1.0e-8,
            maximum_lambda=10.0,
        ),
        retain_rows=False,
    )

    report = diagnose_distributional_fit(model)
    assert report.profile is not None
    component = report.profile.smoothing_components[0]

    assert component.null_space_dimension == 0
    assert component.exact_face_effect == "fully_suppressed"
    assert "fully suppressed" in report.render()
    assert "linear only" not in report.render()
    assert all("loaded artifact" not in limitation for limitation in report.limitations)


def test_loaded_artifact_is_honest_when_machine_timing_is_unavailable(
    profiled_face_fit,
) -> None:
    model, _measured = profiled_face_fit

    restored = SuperLSS.from_bytes(model.to_bytes())
    report = restored.diagnose()

    assert report.profile is not None
    assert report.profile.fit_seconds is None
    assert report.profile.phases == ()
    assert "Timing unavailable" in report.render()


def test_diagnose_tracks_successive_public_fits() -> None:
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 8)})
    first_response = 0.5 + frame["x"].to_numpy()
    second_response = 2.0 - 2.0 * frame["x"].to_numpy()
    model = _fixed_model().fit(frame, first_response)
    first = model.diagnose()

    model.fit(frame, second_response)
    second = model.diagnose()

    assert first.fit_revision == 1
    assert second.fit_revision == 2
    assert first != second
    assert SuperLSS.from_bytes(model.to_bytes()).diagnose().fit_revision == 2
    model._fit_revision = 999
    assert model.diagnose().fit_revision == 2


def test_dense_fit_constructs_the_requested_authoritative_revision() -> None:
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 8)})
    model = fit_dense_distributional(
        frame,
        0.5 + frame["x"].to_numpy(),
        family=GaussianLS(scale_floor=0.01),
        predictors=(
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {}),
        ),
        weight_contract=WeightContract("prior"),
        lambdas={},
        retain_rows=False,
        revision=7,
    )

    assert model.fit_state.revision == 7


@pytest.mark.parametrize("revision", [0, -1, True, 1.5])
def test_dense_fit_rejects_invalid_revision_before_normalizing_input(revision) -> None:
    with pytest.raises(ValueError, match="revision must be a positive integer"):
        fit_dense_distributional(
            object(),
            np.array([1.0]),
            family=GaussianLS(),
            predictors=(),
            weight_contract=WeightContract("prior"),
            revision=revision,
        )


def test_diagnose_requires_a_fit_and_exports_only_the_report_container() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        _fixed_model().diagnose()

    assert superglm.FitDiagnosticReport is FitDiagnosticReport
    assert "FitDiagnosticReport" in superglm.__all__
    assert {
        "DiagnosticAction",
        "DiagnosticEvidence",
        "DiagnosticFinding",
        "DiagnosticSubject",
    }.isdisjoint(superglm.__all__)
