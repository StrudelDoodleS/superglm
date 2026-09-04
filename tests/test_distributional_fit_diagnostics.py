from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm.diagnostics import DiagnosticEvidence
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.fit_diagnostics import (
    _analyze_exact_faces,
    _analyze_fit_status,
    _fit_status,
    _LSSDiagnosticContext,
    _ordinary_updates,
    _penalty_subject,
    _rank_findings,
    diagnose_distributional_fit,
)
from superglm.distributional.model import DenseDistributionalModel, fit_dense_distributional
from superglm.distributional.predictor import Predictor
from superglm.distributional.result import DenseSolverConfig, DistributionalEFSConfig
from superglm.distributional.weights import WeightContract
from superglm.features import Numeric, RandomEffect, Spline


def _fixed_model(*, retain_rows: bool = False):
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 8)})
    response = np.array([-0.8, -0.5, -0.2, 0.1, 0.4, 0.8, 1.0, 1.2])
    return fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.01),
        predictors=(
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {}),
        ),
        weight_contract=WeightContract("prior"),
        lambdas={},
        retain_rows=retain_rows,
    )


def _smooth_fixture():
    rng = np.random.default_rng(23)
    x = np.linspace(-1.0, 1.0, 48)
    z = np.mod(np.arange(len(x)) * 0.37, 1.0)
    response = 0.4 + 0.6 * x + rng.normal(scale=0.25 + 0.08 * z)
    frame = pd.DataFrame({"x": x, "z": z})
    predictors = (
        Predictor("location", {"x": Numeric()}),
        Predictor("scale", {"z": Spline(kind="cr", n_knots=5)}),
    )
    return frame, response, predictors


def _fit_smooth(*, inner_iterations: int, outer_iterations: int, tolerance: float):
    frame, response, predictors = _smooth_fixture()
    return fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.01),
        predictors=predictors,
        weight_contract=WeightContract("prior"),
        lambdas={"scale:z#wiggle": 0.3},
        config=DenseSolverConfig(tolerance=1.0e-9, max_iterations=inner_iterations),
        efs_config=DistributionalEFSConfig(
            outer="efs",
            tolerance=tolerance,
            max_iterations=outer_iterations,
            plateau_tolerance=0.0,
            plateau_iterations=outer_iterations,
        ),
        retain_rows=False,
    )


def _with_curvature_fallbacks(model, counts: dict[int, int]):
    smoothing = model.smoothing
    assert smoothing is not None
    coefficient_fits = []
    for index, fit in enumerate(smoothing.coefficient_fits):
        count = counts.get(index, 0)
        telemetry = fit.terminal_curvature
        if count:
            telemetry = replace(
                telemetry,
                requested_source="observed",
                actual_source="fisher",
                reason="observed_curvature_not_usable",
                fallback_count=count,
            )
        coefficient_fits.append(replace(fit, terminal_curvature=telemetry))
    smoothing = replace(smoothing, coefficient_fits=tuple(coefficient_fits))
    state = model.fit_state
    terminal_fit = smoothing.terminal_fit
    state = replace(
        state,
        solver_result=terminal_fit,
        smoothing=smoothing,
        result=replace(state.result, curvature_telemetry=terminal_fit.terminal_curvature),
    )
    return DenseDistributionalModel(family=model.family, _fit_state=state)


def _with_terminal_condition(model, condition_estimate: float):
    smoothing = model.smoothing
    assert smoothing is not None
    fits = list(smoothing.coefficient_fits)
    terminal_index = smoothing.terminal_fit_index
    terminal = fits[terminal_index]
    terminal = replace(
        terminal,
        terminal_curvature=replace(
            terminal.terminal_curvature,
            condition_estimate=condition_estimate,
        ),
    )
    fits[terminal_index] = terminal
    smoothing = replace(smoothing, coefficient_fits=tuple(fits))
    state = replace(
        model.fit_state,
        solver_result=terminal,
        smoothing=smoothing,
        result=replace(model.fit_state.result, curvature_telemetry=terminal.terminal_curvature),
    )
    return DenseDistributionalModel(family=model.family, _fit_state=state)


def _replace_smoothing(model, smoothing):
    state = replace(
        model.fit_state,
        solver_result=smoothing.terminal_fit,
        smoothing=smoothing,
    )
    return DenseDistributionalModel(family=model.family, _fit_state=state)


def _with_accepted_steps(model, steps: tuple[dict[str, float], ...]):
    smoothing = model.smoothing
    assert smoothing is not None
    assert len(steps) == len(smoothing.history)
    history = tuple(
        replace(iteration, accepted_log_steps=accepted_steps)
        for iteration, accepted_steps in zip(smoothing.history, steps, strict=True)
    )
    return _replace_smoothing(model, replace(smoothing, history=history))


@pytest.fixture(scope="module")
def certified_model():
    model = _fit_smooth(inner_iterations=100, outer_iterations=30, tolerance=1.0e-3)
    assert model.smoothing is not None
    assert model.smoothing.matched_certified is True
    return model


@pytest.fixture(scope="module")
def inner_nonconverged_model():
    model = _fit_smooth(inner_iterations=1, outer_iterations=5, tolerance=1.0e-12)
    assert model.smoothing is not None
    assert model.smoothing.convergence_reason == "coefficient_not_converged"
    return model


@pytest.fixture(scope="module")
def outer_nonconverged_model():
    model = _fit_smooth(inner_iterations=100, outer_iterations=1, tolerance=1.0e-12)
    assert model.smoothing is not None
    assert model.smoothing.convergence_reason == "max_iterations"
    assert model.smoothing.terminal_fit.converged is True
    return model


@pytest.fixture(scope="module")
def unresolved_cap_model():
    rng = np.random.default_rng(11)
    z = np.linspace(0.0, 1.0, 40)
    response = (
        0.5 + 0.2 * np.sin(2.0 * np.pi * z) + rng.normal(scale=0.2 + 0.06 * np.cos(2.0 * np.pi * z))
    )
    model = fit_dense_distributional(
        pd.DataFrame({"z": z}),
        response,
        family=GaussianLS(scale_floor=0.01),
        predictors=(
            Predictor("location", {"z": Spline(kind="cr", n_knots=5)}),
            Predictor("scale", {"z": Spline(kind="cr", n_knots=5)}),
        ),
        weight_contract=WeightContract("prior"),
        lambdas={"location:z#wiggle": 0.3, "scale:z#wiggle": 0.3},
        config=DenseSolverConfig(tolerance=1.0e-9, max_iterations=100),
        efs_config=DistributionalEFSConfig(
            outer="efs",
            tolerance=1.0e-8,
            max_iterations=8,
            initial_lambda=0.1,
            maximum_lambda=0.3,
        ),
        retain_rows=False,
    )
    smoothing = model.smoothing
    assert smoothing is not None
    assert smoothing.convergence_reason == "lambda_cap_unresolved"
    assert smoothing.unresolved_upper_bound == ("scale:z#wiggle",)
    smoothing = replace(
        smoothing,
        unresolved_upper_bound=("location:z#wiggle", "scale:z#wiggle"),
    )
    return _replace_smoothing(model, smoothing)


@pytest.fixture(scope="module")
def rejected_step_model():
    rng = np.random.default_rng(0)
    z = np.linspace(0.0, 1.0, 36)
    response = 0.3 + 0.5 * np.sin(2.0 * np.pi * z) + rng.normal(scale=0.1 + 0.25 * z)
    model = fit_dense_distributional(
        pd.DataFrame({"z": z}),
        response,
        family=GaussianLS(scale_floor=0.01),
        predictors=(
            Predictor("location", {"z": Spline(kind="cr", n_knots=5)}),
            Predictor("scale", {"z": Spline(kind="cr", n_knots=5)}),
        ),
        weight_contract=WeightContract("prior"),
        lambdas={"location:z#wiggle": 0.1, "scale:z#wiggle": 0.1},
        config=DenseSolverConfig(tolerance=1.0e-9, max_iterations=100),
        efs_config=DistributionalEFSConfig(
            outer="efs",
            tolerance=1.0e-6,
            max_iterations=10,
            max_backtracks=2,
            objective_tolerance=0.0,
            plateau_tolerance=0.0,
            plateau_iterations=10,
        ),
        retain_rows=False,
    )
    smoothing = model.smoothing
    assert smoothing is not None
    assert smoothing.convergence_reason == "objective_rejected"
    assert [(item.accepted, item.backtracks) for item in smoothing.history] == [
        (True, 0),
        (True, 0),
        (False, 2),
    ]
    return model


@pytest.fixture(scope="module")
def two_component_model():
    rng = np.random.default_rng(11)
    z = np.linspace(0.0, 1.0, 60)
    response = (
        0.5 + 0.2 * np.sin(2.0 * np.pi * z) + rng.normal(scale=0.2 + 0.06 * np.cos(2.0 * np.pi * z))
    )
    model = fit_dense_distributional(
        pd.DataFrame({"z": z}),
        response,
        family=GaussianLS(scale_floor=0.01),
        predictors=(
            Predictor("location", {"z": Spline(kind="cr", n_knots=5)}),
            Predictor("scale", {"z": Spline(kind="cr", n_knots=5)}),
        ),
        weight_contract=WeightContract("prior"),
        lambdas={"location:z#wiggle": 0.3, "scale:z#wiggle": 0.3},
        config=DenseSolverConfig(tolerance=1.0e-9, max_iterations=100),
        efs_config=DistributionalEFSConfig(
            outer="efs",
            tolerance=1.0e-3,
            max_iterations=20,
            plateau_tolerance=0.0,
            plateau_iterations=20,
        ),
        retain_rows=False,
    )
    assert model.smoothing is not None
    assert model.smoothing.matched_certified is True
    assert model.smoothing.iterations == 7
    return model


@pytest.fixture(scope="module")
def upward_model():
    model = _fit_smooth(inner_iterations=100, outer_iterations=20, tolerance=1.0e-4)
    assert model.smoothing is not None
    assert model.smoothing.matched_certified is True
    assert model.smoothing.iterations == 4
    return model


@pytest.fixture(scope="module")
def rank_deficient_model():
    x = np.linspace(-1.0, 1.0, 24)
    response = 0.4 + 0.7 * x + 0.05 * np.sin(np.arange(len(x)))
    model = fit_dense_distributional(
        pd.DataFrame({"x": x, "x_copy": x}),
        response,
        family=GaussianLS(scale_floor=0.01),
        predictors=(
            Predictor("location", {"x": Numeric(), "x_copy": Numeric()}),
            Predictor("scale", {}),
        ),
        weight_contract=WeightContract("prior"),
        lambdas={},
        retain_rows=False,
    )
    assert model.result.terminal_rank.rank < model.layout.n_coefficients
    return model


@pytest.fixture(scope="module")
def face_model():
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
            outer="efs",
            max_iterations=8,
            tolerance=1.0e-8,
            maximum_lambda=10.0,
        ),
        retain_rows=False,
    )
    assert model.fit_state.exact_face_components == ("location:group#wiggle",)
    return model


def test_fixed_lambda_fit_has_fixed_status_without_smoothing_findings() -> None:
    report = diagnose_distributional_fit(_fixed_model())

    assert report.fit_status == "fixed_fit"
    assert all(not finding.code.startswith("smoothing.") for finding in report.findings)


def test_smoothed_convergence_sources_independently_control_status_and_findings(
    certified_model,
) -> None:
    context = _LSSDiagnosticContext.from_model(certified_model)
    smoothing = context.smoothing
    assert smoothing is not None
    assert context.fitted_result.converged
    assert context.terminal_fit.converged
    assert smoothing.converged

    terminal_failure = replace(
        context,
        terminal_fit=replace(
            context.terminal_fit,
            converged=False,
            convergence_reason="max_iterations",
        ),
    )
    smoothing_failure = replace(
        context,
        smoothing=replace(
            smoothing,
            converged=False,
            convergence_reason="max_iterations",
        ),
    )
    compact_failure = replace(
        context,
        fitted_result=replace(
            context.fitted_result,
            coefficient_converged=False,
            converged=False,
        ),
    )

    for failed, expected_loop in (
        (terminal_failure, "coefficient"),
        (smoothing_failure, "smoothing"),
    ):
        assert _fit_status(failed) == "not_converged"
        (finding,) = _analyze_fit_status(failed)
        evidence = {item.metric: item.value for item in finding.evidence}
        assert evidence["loop"] == expected_loop
        assert evidence["terminal_reason"] == "max_iterations"

    assert _fit_status(compact_failure) == "not_converged"
    (compact_finding,) = _analyze_fit_status(compact_failure)
    compact_evidence = {item.metric: item.value for item in compact_finding.evidence}
    assert compact_evidence["source"] == "compact_result"
    assert compact_evidence["compact_converged"] is False
    assert compact_evidence["coefficient_converged"] is False
    assert "terminal_reason" not in compact_evidence


def test_fixed_fit_requires_compact_and_terminal_convergence_without_borrowed_reason() -> None:
    context = _LSSDiagnosticContext.from_model(_fixed_model())
    terminal_failure = replace(
        context,
        terminal_fit=replace(
            context.terminal_fit,
            converged=False,
            convergence_reason="max_iterations",
        ),
    )
    compact_failure = replace(
        context,
        fitted_result=replace(
            context.fitted_result,
            coefficient_converged=False,
            converged=False,
        ),
    )

    assert _fit_status(terminal_failure) == "not_converged"
    (terminal_finding,) = _analyze_fit_status(terminal_failure)
    terminal_evidence = {item.metric: item.value for item in terminal_finding.evidence}
    assert terminal_evidence["loop"] == "coefficient"
    assert terminal_evidence["terminal_reason"] == "max_iterations"

    assert _fit_status(compact_failure) == "not_converged"
    (compact_finding,) = _analyze_fit_status(compact_failure)
    compact_evidence = {item.metric: item.value for item in compact_finding.evidence}
    assert compact_evidence["source"] == "compact_result"
    assert "terminal_reason" not in compact_evidence


def test_certified_reml_fit_reports_bounded_no_finding_result(certified_model) -> None:
    report = diagnose_distributional_fit(certified_model)

    assert report.fit_status == "converged_certified"
    assert report.findings == ()
    rendered = report.render(detail="full")
    assert "No solver pathology was detected in the available evidence" in rendered
    assert "No row, response, weight, exposure, influence, or per-level evidence" in rendered
    assert "No fit timing snapshot was retained" in rendered


@pytest.mark.parametrize(
    ("model_fixture", "expected_loops", "expected_reasons"),
    [
        (
            "inner_nonconverged_model",
            {"coefficient", "smoothing"},
            {"max_iterations", "coefficient_not_converged"},
        ),
        ("outer_nonconverged_model", {"smoothing"}, {"max_iterations"}),
    ],
)
def test_nonconverged_findings_name_authoritative_loop_and_reason(
    request: pytest.FixtureRequest,
    model_fixture: str,
    expected_loops: set[str],
    expected_reasons: set[str],
) -> None:
    report = diagnose_distributional_fit(request.getfixturevalue(model_fixture))

    findings = tuple(f for f in report.findings if f.code == "fit.not_converged")
    evidence = [{item.metric: item.value for item in finding.evidence} for finding in findings]
    assert report.fit_status == "not_converged"
    assert {item["loop"] for item in evidence} == expected_loops
    assert {item["terminal_reason"] for item in evidence} == expected_reasons


@pytest.mark.parametrize(
    ("counts", "expected_count", "fallback_word"),
    [({-1: 1}, 1, "fallback"), ({0: 1, -1: 2}, 3, "fallbacks")],
)
def test_curvature_fallbacks_make_converged_result_uncertified_and_report_sources(
    certified_model,
    counts: dict[int, int],
    expected_count: int,
    fallback_word: str,
) -> None:
    smoothing = certified_model.smoothing
    assert smoothing is not None
    resolved_counts = {
        (smoothing.terminal_fit_index if index == -1 else index): count
        for index, count in counts.items()
    }
    model = _with_curvature_fallbacks(certified_model, resolved_counts)

    report = diagnose_distributional_fit(model)

    assert report.fit_status == "converged_uncertified"
    uncertified = next(
        finding for finding in report.findings if finding.code == "fit.termination_uncertified"
    )
    fallback = next(
        finding
        for finding in report.findings
        if finding.code == "optimization.curvature_fallback_repeated"
    )
    evidence = {item.metric: item.value for item in fallback.evidence}
    assert {item.metric: item.value for item in uncertified.evidence}["matched_certified"] is False
    assert evidence["fallback_count"] == expected_count
    assert evidence["requested_sources"] == ("observed",)
    assert evidence["actual_sources"] == ("fisher",)
    assert f"{expected_count} curvature {fallback_word}" in fallback.observed


def test_lambda_cap_unresolved_is_per_qualified_component_with_refusal_evidence(
    unresolved_cap_model,
) -> None:
    report = diagnose_distributional_fit(unresolved_cap_model)

    findings = tuple(
        finding for finding in report.findings if finding.code == "fit.lambda_cap_unresolved"
    )
    assert tuple(finding.subject.identifier for finding in findings) == (
        "penalty:location:z#wiggle",
        "penalty:scale:z#wiggle",
    )
    assert tuple(finding.subject.predictor for finding in findings) == (
        "location",
        "scale",
    )
    assert all(finding.subject.term == "z" for finding in findings)
    by_subject = {
        finding.subject.identifier: {item.metric: item.value for item in finding.evidence}
        for finding in findings
    }
    for evidence in by_subject.values():
        assert evidence["terminal_lambda"] == 0.3
        assert evidence["configured_maximum_lambda"] == 0.3
        assert evidence["terminal_stationarity_evidence"] > 0.0
    assert {name: evidence["endpoint_refusal_reason"] for name, evidence in by_subject.items()} == {
        "penalty:location:z#wiggle": "joint_objective_rejected",
        "penalty:scale:z#wiggle": "joint_objective_rejected",
    }


def test_upper_boundary_without_exact_face_stays_a_boundary_and_trajectory_claim(
    unresolved_cap_model,
) -> None:
    assert unresolved_cap_model.fit_state.exact_face_components == ()

    report = diagnose_distributional_fit(unresolved_cap_model)

    boundary = tuple(
        finding
        for finding in report.findings
        if finding.code == "smoothing.penalty_at_upper_boundary"
    )
    unsettled = next(
        finding for finding in report.findings if finding.code == "smoothing.trajectory_unsettled"
    )
    assert {finding.subject.identifier for finding in boundary} == {
        "penalty:location:z#wiggle",
        "penalty:scale:z#wiggle",
    }
    assert all("endpoint" in " ".join(finding.caveats).lower() for finding in boundary)
    assert all(
        finding.code != "smoothing.penalized_subspace_suppressed" for finding in report.findings
    )
    unsettled_evidence = {item.metric: item for item in unsettled.evidence}
    assert unsettled_evidence["terminal_stationarity_evidence"].value > 0.0
    assert unsettled_evidence["terminal_stationarity_evidence"].threshold == 1.0e-8


def test_exhausted_outer_backtracking_reports_work_counts_not_wall_time(
    rejected_step_model,
) -> None:
    smoothing = rejected_step_model.smoothing
    assert smoothing is not None
    report = diagnose_distributional_fit(rejected_step_model)

    finding = next(
        finding
        for finding in report.findings
        if finding.code == "optimization.repeated_step_rejection"
    )
    evidence = {item.metric: item.value for item in finding.evidence}
    assert evidence == {
        "outer_iterations": 3,
        "accepted_ordinary_updates": 2,
        "rejected_iterations": 1,
        "backtracked_outer_iterations": 1,
        "total_backtracks": 2,
        "configured_backtrack_budget": 2,
        "terminal_stationarity_evidence": smoothing.terminal_convergence_max_log_residual,
    }
    stationarity = next(
        item for item in finding.evidence if item.metric == "terminal_stationarity_evidence"
    )
    assert stationarity.comparator == "<="
    assert stationarity.threshold == smoothing.config.tolerance
    assert "coefficient refits" in finding.observed
    assert "wall-time" in " ".join(finding.caveats)


def test_zero_backtrack_refused_face_is_not_an_ordinary_step_rejection(
    unresolved_cap_model,
) -> None:
    smoothing = unresolved_cap_model.smoothing
    assert smoothing is not None
    (refusal,) = smoothing.history
    assert refusal.accepted is False
    assert refusal.backtracks == 0
    assert refusal.refused_face_components
    smoothing = replace(
        smoothing,
        config=replace(smoothing.config, max_backtracks=0),
    )
    model = _replace_smoothing(unresolved_cap_model, smoothing)

    report = diagnose_distributional_fit(model)

    assert all(
        finding.code != "optimization.repeated_step_rejection" for finding in report.findings
    )


def test_update_dominance_divides_exact_ties_and_reports_eligible_denominator(
    two_component_model,
) -> None:
    location = "location:z#wiggle"
    scale = "scale:z#wiggle"
    model = _with_accepted_steps(
        two_component_model,
        (
            {location: 2.0, scale: 1.0},
            {location: 2.0, scale: 2.0},
            {location: 1.0, scale: 2.0},
            {location: 3.0, scale: 3.0},
            {location: 2.0, scale: 1.0},
            {location: 2.0, scale: 1.0},
            {location: 0.0, scale: 0.0},
        ),
    )

    report = diagnose_distributional_fit(model)

    finding = next(item for item in report.findings if item.code == "smoothing.update_dominance")
    evidence = {item.metric: item.value for item in finding.evidence}
    assert finding.subject.identifier == "penalty:location:z#wiggle"
    assert evidence["equivalent_wins"] == 4.0
    assert evidence["work_share"] == pytest.approx(2.0 / 3.0)
    assert evidence["tie_updates"] == 2
    assert evidence["eligible_updates"] == 6
    smoothing = model.smoothing
    assert smoothing is not None
    stationarity = next(
        item for item in finding.evidence if item.metric == "terminal_stationarity_evidence"
    )
    assert stationarity.value == smoothing.terminal_convergence_max_log_residual
    assert stationarity.comparator == "<="
    assert stationarity.threshold == smoothing.config.tolerance
    assert stationarity.value <= stationarity.threshold


def test_upward_drift_requires_four_nonzero_moves_and_never_claims_endpoint(
    certified_model,
    upward_model,
) -> None:
    below_threshold = diagnose_distributional_fit(certified_model)
    report = diagnose_distributional_fit(upward_model)

    assert all(
        finding.code != "smoothing.persistent_upward_drift" for finding in below_threshold.findings
    )
    finding = next(
        item for item in report.findings if item.code == "smoothing.persistent_upward_drift"
    )
    evidence = {item.metric: item.value for item in finding.evidence}
    assert evidence["positive_accepted_moves"] == 4
    assert evidence["negative_accepted_moves"] == 0
    assert evidence["zero_accepted_moves"] == 0
    smoothing = upward_model.smoothing
    assert smoothing is not None
    stationarity = next(
        item for item in finding.evidence if item.metric == "terminal_stationarity_evidence"
    )
    assert stationarity.value == smoothing.terminal_convergence_max_log_residual
    assert stationarity.comparator == "<="
    assert stationarity.threshold == smoothing.config.tolerance
    assert stationarity.value <= stationarity.threshold
    assert "does not establish an endpoint" in " ".join(finding.caveats).lower()


def test_estimable_rank_below_active_full_coordinate_dimension_reports_rank_loss(
    rank_deficient_model,
) -> None:
    report = diagnose_distributional_fit(rank_deficient_model)

    finding = next(item for item in report.findings if item.code == "numerics.rank_loss")
    evidence = {item.metric: item.value for item in finding.evidence}
    assert {item.provenance for item in finding.evidence} == {"fit_result"}
    assert evidence["estimable_rank"] == rank_deficient_model.result.terminal_rank.rank
    assert evidence["active_coordinate_dimension"] == (rank_deficient_model.layout.n_coefficients)
    assert evidence["coordinate_space"] == "full"


def test_conditioning_warning_uses_retained_terminal_estimate_and_exact_threshold(
    certified_model,
) -> None:
    threshold = 1.0 / np.sqrt(np.finfo(np.float64).eps)
    model = _with_terminal_condition(certified_model, threshold)

    report = diagnose_distributional_fit(model)

    finding = next(item for item in report.findings if item.code == "numerics.conditioning_warning")
    evidence = {item.metric: item for item in finding.evidence}
    assert evidence["condition_estimate"].value == threshold
    assert evidence["condition_estimate"].threshold == threshold
    assert evidence["condition_estimate"].provenance == "curvature_telemetry"
    wording = f"{finding.headline} {finding.observed} {finding.interpretation}".lower()
    assert "certified condition estimate" not in wording
    assert "retained condition estimate" in wording or "reported condition estimate" in wording


def test_exact_faced_spline_reports_suppressed_penalized_subspace_and_retained_null_space(
    certified_model,
) -> None:
    context = _LSSDiagnosticContext.from_model(certified_model)
    component = context.layout.penalties[0]
    width = component.group_sl.stop - component.group_sl.start
    assert component.rank == width - 1
    context = replace(context, exact_face_components=(component.name,))

    (finding,) = _analyze_exact_faces(context)

    evidence = {item.metric: item.value for item in finding.evidence}
    assert finding.code == "smoothing.penalized_subspace_suppressed"
    assert finding.confidence == "certified"
    assert evidence["component_rank"] == width - 1
    assert evidence["block_width"] == width
    assert evidence["retained_null_space_dimension"] == 1
    face_claim = f"{finding.headline} {finding.observed}".lower()
    assert "penalized subspace" in face_claim
    assert "removed penalty component" not in face_claim
    assert "removed component" not in face_claim
    assert "removed feature" not in face_claim
    caveat = " ".join(finding.caveats).lower()
    assert "retained null-space" in caveat
    assert "feature" in caveat and "useless" in caveat


def test_fully_penalized_exact_face_omits_retained_null_space_caveat(face_model) -> None:
    report = diagnose_distributional_fit(face_model)

    finding = next(
        item for item in report.findings if item.code == "smoothing.penalized_subspace_suppressed"
    )
    evidence = {item.metric: item.value for item in finding.evidence}
    assert evidence["retained_null_space_dimension"] == 0
    assert "retained null-space" not in " ".join(finding.caveats).lower()


@pytest.mark.parametrize("rank", [np.nan, 0.25, "unknown"])
def test_unresolved_component_rank_omits_null_space_claim(certified_model, rank) -> None:
    context = _LSSDiagnosticContext.from_model(certified_model)
    component = replace(context.layout.penalties[0], rank=rank)
    context = replace(
        context,
        layout=replace(context.layout, penalties=(component,)),
        exact_face_components=(component.name,),
    )

    (finding,) = _analyze_exact_faces(context)

    assert finding.confidence == "unresolved"
    assert all(item.metric != "retained_null_space_dimension" for item in finding.evidence)
    assert "retained null-space" not in " ".join(finding.caveats).lower()


def test_face_assessments_are_not_ordinary_updates_or_false_rank_loss(face_model) -> None:
    smoothing = face_model.smoothing
    assert smoothing is not None
    ordinary = _ordinary_updates(smoothing)
    accepted = tuple(item for item in smoothing.history if item.accepted)

    assert len(ordinary) < len(accepted)
    assert all(
        not (
            item.activated_face_components
            or item.deactivated_face_components
            or item.revalidated_face_components
            or item.refused_face_components
        )
        for item in ordinary
    )
    assert face_model.result.terminal_rank.rank < face_model.layout.n_coefficients
    report = diagnose_distributional_fit(face_model)
    assert all(finding.code != "numerics.rank_loss" for finding in report.findings)


def test_adapter_ranks_by_tier_scope_and_stable_component_identity(
    unresolved_cap_model,
) -> None:
    report = diagnose_distributional_fit(unresolved_cap_model)

    assert tuple(finding.priority_tier for finding in report.findings) == tuple(
        sorted(finding.priority_tier for finding in report.findings)
    )
    tier_one = tuple(
        finding.identifier for finding in report.findings if finding.priority_tier == 1
    )
    assert tier_one == (
        "fit.not_converged:smoothing",
        "fit.lambda_cap_unresolved:location:z#wiggle",
        "fit.lambda_cap_unresolved:scale:z#wiggle",
    )


def test_private_ranking_uses_confidence_before_descending_work_share(
    unresolved_cap_model,
) -> None:
    base = next(
        finding
        for finding in diagnose_distributional_fit(unresolved_cap_model).findings
        if finding.code == "smoothing.penalty_at_upper_boundary"
    )

    def ranked_fixture(identifier: str, confidence: str, share: float):
        return replace(
            base,
            identifier=identifier,
            confidence=confidence,
            subject=replace(base.subject, identifier=f"penalty:{identifier}"),
            evidence=(
                DiagnosticEvidence(
                    metric="work_share",
                    value=share,
                    unit=None,
                    window="accepted ordinary updates",
                    provenance="smoothing_history",
                ),
            ),
            priority_tier=3,
        )

    certified_low = ranked_fixture("certified-low", "certified", 0.1)
    strong_low = ranked_fixture("strong-low", "strong", 0.2)
    strong_high = ranked_fixture("strong-high", "strong", 0.8)

    ranked = _rank_findings((strong_low, certified_low, strong_high))

    assert tuple(finding.identifier for finding in ranked) == (
        "certified-low",
        "strong-high",
        "strong-low",
    )


def test_retained_rows_are_neither_read_nor_reflected_in_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = _fixed_model(retain_rows=True)
    absent = DenseDistributionalModel(
        family=retained.family,
        _fit_state=replace(retained.fit_state, retained_rows=None),
    )
    expected = diagnose_distributional_fit(absent)

    def refuse_row_access(_state):
        raise AssertionError("diagnosis must not read retained_rows")

    monkeypatch.setattr(
        type(retained.fit_state),
        "retained_rows",
        property(refuse_row_access),
        raising=False,
    )

    assert diagnose_distributional_fit(retained) == expected


def test_inconsistent_component_qualification_is_rejected_instead_of_guessed(
    certified_model,
) -> None:
    context = _LSSDiagnosticContext.from_model(certified_model)
    component = replace(context.layout.penalties[0], group_name="scale:other")

    with pytest.raises(ValueError, match="inconsistent qualified names"):
        _penalty_subject(context.layout, component)


def test_report_identifies_family_revision_and_examined_evidence_classes() -> None:
    model = _fixed_model()
    report = diagnose_distributional_fit(model)

    assert report.model_type == "SuperLSS"
    assert report.family == "GaussianLS"
    assert report.fit_revision == 1
    assert not hasattr(_LSSDiagnosticContext.from_model(model), "inference")
    assert report.coverage == (
        "Accepted compact fit status and revision metadata were examined.",
        "Terminal coefficient-solver convergence, rank, and curvature telemetry were examined.",
        "No smoothing result or smoothing history exists for this fixed fit.",
    )


def _stationary_model():
    """The smooth fixture under the Newton endgame: a ``stationary`` stop."""
    frame, response, predictors = _smooth_fixture()
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.01),
        predictors=predictors,
        weight_contract=WeightContract("prior"),
        lambdas={"scale:z#wiggle": 0.3},
        config=DenseSolverConfig(tolerance=1.0e-9, max_iterations=100),
        efs_config=DistributionalEFSConfig(
            tolerance=1.0e-3,
            max_iterations=30,
            plateau_tolerance=0.0,
            plateau_iterations=30,
            outer="efs+newton",
        ),
        retain_rows=False,
    )
    assert model.smoothing is not None
    assert model.smoothing.convergence_reason == "stationary"
    return model


def test_stationary_stop_is_described_as_the_reml_optimum() -> None:
    report = diagnose_distributional_fit(_stationary_model())
    text = report.render(detail="full")
    assert "REML optimum" in text and "projected gradient" in text
    assert not [f for f in report.findings if f.code == "smoothing.persistent_upward_drift"]
