"""Read-only fit diagnostics for one accepted distributional revision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from superglm.diagnostics import (
    DiagnosticAction,
    DiagnosticEvidence,
    DiagnosticFinding,
    DiagnosticSubject,
    FitDiagnosticReport,
)
from superglm.diagnostics.fit_report import (
    FitPhaseProfile,
    FitWorkProfile,
    JsonValue,
    SmoothingComponentProfile,
)
from superglm.distributional.family import ExpectedInformationFamily
from superglm.distributional.layout import StackedLayout
from superglm.distributional.model import DenseDistributionalModel
from superglm.distributional.result import (
    DenseSolverResult,
    DistributionalEFSIteration,
    DistributionalEFSResult,
    DistributionalFitResult,
)
from superglm.distributional.timing import FitPhaseSnapshot
from superglm.features.spline import CubicRegressionSpline
from superglm.solvers.rank import SHARED_RANK_POLICY, _eigensolver_relative_bar
from superglm.types import PenaltyComponent

_MIN_DOMINANCE_UPDATES = 4
_MIN_DOMINANCE_EQUIVALENT_WINS = 3.0
_DOMINANCE_SHARE = 0.5
_MIN_UPWARD_DRIFT_UPDATES = 4
_CONDITION_WARNING_THRESHOLD = 1.0 / np.sqrt(np.finfo(np.float64).eps)
_SMOOTHING_HISTORY = "smoothing_history"
_CURVATURE_TELEMETRY = "curvature_telemetry"
_ALL_FITTED_PARAMETERS = "all fitted distributional parameters"
_STATIONARITY_METRIC = "terminal stationarity evidence"
type _EvidenceProvenance = Literal[
    "fit_result",
    "solver_history",
    "smoothing_history",
    "curvature_telemetry",
    "inference",
    "counterfactual_fit",
]
type _FindingSeverity = Literal["error", "warning", "info"]
type _FindingConfidence = Literal["certified", "strong", "suggestive", "unresolved"]
type _CurvatureMatrixKind = Literal["data", "penalized"]
# Recorded phases that never enclose another recorded phase, so their seconds
# add up without double counting; ``initialization`` and
# ``terminal_observed_retry_fallback`` wrap leaf phases and are left out.  The
# left name is the recorder's, the right name is the profile's.
_PROFILE_LEAF_PHASES = (
    ("frame_normalization", "frame_normalization"),
    ("predictor_compilation", "predictor_compilation"),
    ("layout_penalty_assembly", "layout_penalty_assembly"),
    ("dense_predictor_matrices", "dense_predictor_matrices"),
    ("likelihood_evaluation", "likelihood_evaluation"),
    ("curvature_gradient_assembly", "curvature_gradient_assembly"),
    ("coefficient_decomposition_solve", "coefficient_decomposition_solve"),
    ("efs_update_backtracking", "efs_update_backtracking"),
    ("inference_edf", "terminal_inference_and_null_fit"),
)


@dataclass(frozen=True, slots=True)
class _LSSDiagnosticContext:
    """One call-scoped view of bounded evidence from an accepted revision."""

    revision: int
    n_observations: int
    family_name: str
    terminal_policy_matrix_kind: _CurvatureMatrixKind
    layout: StackedLayout
    linear_spline_terms: tuple[str, ...]
    fitted_result: DistributionalFitResult
    terminal_fit: DenseSolverResult
    smoothing: DistributionalEFSResult | None
    exact_face_components: tuple[str, ...]

    @classmethod
    def from_model(cls, model: DenseDistributionalModel) -> _LSSDiagnosticContext:
        if not isinstance(model, DenseDistributionalModel):
            raise TypeError("model must be a DenseDistributionalModel")
        state = model.fit_state
        return cls(
            revision=state.revision,
            n_observations=state.null_model.n_observations,
            family_name=type(model.family).__name__,
            terminal_policy_matrix_kind=(
                state.solver_result.terminal_curvature.matrix_kind
                or ("data" if isinstance(model.family, ExpectedInformationFamily) else "penalized")
            ),
            layout=state.layout,
            linear_spline_terms=tuple(
                f"{predictor.name}:{term}"
                for predictor in state.predictor_templates
                for term, feature in predictor.features.items()
                if type(feature) is CubicRegressionSpline and not bool(getattr(feature, "select"))
            ),
            fitted_result=state.result,
            terminal_fit=state.solver_result,
            smoothing=state.smoothing,
            exact_face_components=state.exact_face_components,
        )


_WHOLE_FIT_SUBJECT = DiagnosticSubject(
    identifier="fit",
    display_name="complete fitted model",
    scope="whole_fit",
)


def _penalty_subject(layout: StackedLayout, component: PenaltyComponent) -> DiagnosticSubject:
    predictor, separator, term = component.group_name.partition(":")
    component_term, marker, suffix = component.name.rpartition("#")
    predictor_names = {state.name for state in layout.predictors}
    if (
        not separator
        or predictor not in predictor_names
        or not term
        or not marker
        or not suffix
        or component_term != component.group_name
    ):
        raise ValueError(f"penalty component {component.name!r} has inconsistent qualified names")
    return DiagnosticSubject(
        identifier=f"penalty:{component.name}",
        display_name=(f"{predictor} predictor term {term!r}, penalty component {component.name!r}"),
        scope="penalized_subspace",
        predictor=predictor,
        term=term,
        component=component.name,
    )


def _component(context: _LSSDiagnosticContext, name: str) -> PenaltyComponent:
    matches = tuple(item for item in context.layout.penalties if item.name == name)
    if len(matches) != 1:
        raise ValueError(f"accepted layout does not uniquely identify penalty {name!r}")
    return matches[0]


def _component_subject(context: _LSSDiagnosticContext, name: str) -> DiagnosticSubject:
    return _penalty_subject(context.layout, _component(context, name))


def _evidence_set(
    provenance: _EvidenceProvenance,
    *items: tuple[str, JsonValue],
) -> tuple[DiagnosticEvidence, ...]:
    return tuple(
        DiagnosticEvidence(
            metric=metric,
            value=value,
            unit=None,
            window="accepted terminal revision",
            provenance=provenance,
        )
        for metric, value in items
    )


StopKind = Literal["practical", "stationary", "strict"]


def _stop_kind(smoothing: DistributionalEFSResult) -> StopKind:
    if smoothing.convergence_reason == "practical_plateau":
        return "practical"
    if smoothing.convergence_reason == "stationary":
        return "stationary"
    return "strict"


def _stationary_statement(smoothing: DistributionalEFSResult) -> str:
    """The Newton endgame's stop in one sentence, with its numbers."""
    norm = smoothing.terminal_projected_gradient_norm
    return (
        "The smoothing search ended at the REML optimum, a stationary point of the LAML "
        "criterion in log lambda: the largest projected gradient component is "
        f"{0.0 if norm is None else norm:.3g} against a tolerance of "
        f"{smoothing.stationarity_bar:.3g}."
    )


def _stationarity_evidence(smoothing: DistributionalEFSResult) -> DiagnosticEvidence:
    """The stop's own authority: the projected exact gradient at a stationary stop,
    the Fellner--Schall residual otherwise."""
    if smoothing.convergence_reason == "stationary":
        norm = smoothing.terminal_projected_gradient_norm
        return DiagnosticEvidence(
            metric="terminal_projected_gradient",
            value=0.0 if norm is None else norm,
            unit="objective scale",
            window="accepted terminal smoothing state",
            provenance=_SMOOTHING_HISTORY,
            comparator="<=",
            threshold=smoothing.stationarity_bar,
        )
    return DiagnosticEvidence(
        metric="terminal_stationarity_evidence",
        value=smoothing.terminal_convergence_max_log_residual,
        unit="normalized residual",
        window="accepted terminal smoothing state",
        provenance=_SMOOTHING_HISTORY,
        comparator="<=",
        threshold=smoothing.config.tolerance,
    )


def _gradient_certificate_evidence(smoothing: DistributionalEFSResult) -> DiagnosticEvidence:
    certificate = smoothing.terminal_gradient_certificate
    largest = 0.0 if not certificate else max(certificate.values())
    return DiagnosticEvidence(
        metric="terminal_gradient_certificate",
        value=largest,
        unit="objective scale",
        window="accepted terminal smoothing state",
        provenance=_SMOOTHING_HISTORY,
        comparator="<=",
        threshold=smoothing.stationarity_bar,
    )


def _terminal_lambda_evidence(
    smoothing: DistributionalEFSResult,
    name: str,
) -> tuple[DiagnosticEvidence, ...]:
    return _evidence_set(
        _SMOOTHING_HISTORY,
        ("terminal_lambda", float(smoothing.lambdas[name])),
        ("configured_maximum_lambda", float(smoothing.config.maximum_lambda)),
    )


def _residual_severity(residual: float, tolerance: float) -> Literal["info", "warning", "error"]:
    """Grade a terminal raw log-step residual on its magnitude alone.

    Below one percent of a log unit the smoothing parameters are settled for
    every downstream purpose; up to one log unit the fit is usable but the
    selection is loose; above that the trajectory was still moving.
    """
    if residual <= max(1.0e-2, tolerance):
        return "info"
    if residual <= 1.0:
        return "warning"
    return "error"


def _finding(
    code: str,
    subject: DiagnosticSubject = _WHOLE_FIT_SUBJECT,
    *,
    priority: int,
    headline: str,
    observed: str,
    interpretation: str,
    evidence: tuple[DiagnosticEvidence, ...],
    action_kind: str,
    action_question: str,
    action_metrics: tuple[str, ...],
    identifier: str | None = None,
    severity: _FindingSeverity = "warning",
    confidence: _FindingConfidence = "certified",
    impacts: tuple[str, ...] = (),
    caveats: tuple[str, ...] = (),
) -> DiagnosticFinding:
    """Construct one validated finding without adding a mutable builder surface."""
    return DiagnosticFinding(
        identifier=f"{code}:{subject.identifier}" if identifier is None else identifier,
        code=code,
        category="fit_status" if code.startswith("fit.") else code.partition(".")[0],
        severity=severity,
        confidence=confidence,
        impacts=impacts,
        subject=subject,
        headline=headline,
        observed=observed,
        interpretation=interpretation,
        caveats=caveats,
        evidence=evidence,
        actions=(
            DiagnosticAction(
                kind=action_kind,
                question=action_question,
                requires_data=True,
                comparison_metrics=action_metrics,
            ),
        ),
        priority_tier=priority,
    )


def _not_converged_finding(
    *,
    loop: str,
    reason: str,
    iterations: int,
    provenance: _EvidenceProvenance,
) -> DiagnosticFinding:
    return _finding(
        "fit.not_converged",
        priority=1,
        identifier=f"fit.not_converged:{loop}",
        severity="error",
        impacts=("fit_validity",),
        headline=f"The accepted {loop} loop did not converge.",
        observed=f"The authoritative {loop} result stopped with reason {reason!r}.",
        interpretation="The reported parameter state did not satisfy its governing fit contract.",
        caveats=("This evidence does not identify a statistical cause for the failure.",),
        evidence=_evidence_set(
            provenance,
            ("loop", loop),
            ("terminal_reason", reason),
            ("iterations", iterations),
        ),
        action_kind="inspect_fit_configuration",
        action_question=(
            f"Which data or configuration change lets the {loop} loop satisfy its declared "
            "convergence contract?"
        ),
        action_metrics=("convergence reason", "iterations", "fit objective"),
    )


def _compact_not_converged_finding(
    fitted_result: DistributionalFitResult,
) -> DiagnosticFinding:
    return _finding(
        "fit.not_converged",
        priority=1,
        identifier="fit.not_converged:compact_result",
        severity="error",
        impacts=("fit_validity",),
        headline="The accepted compact fit result reports nonconvergence.",
        observed="Its retained overall convergence flag is false.",
        interpretation="The accepted complete-fit summary did not satisfy its fit contract.",
        caveats=("The compact result does not retain a terminal convergence reason.",),
        evidence=_evidence_set(
            "fit_result",
            ("source", "compact_result"),
            ("compact_converged", fitted_result.converged),
            ("coefficient_converged", fitted_result.coefficient_converged),
            ("smoothing_converged", fitted_result.smoothing_converged),
        ),
        action_kind="inspect_fit_configuration",
        action_question="Which controlled change makes the complete-fit summary converge?",
        action_metrics=("compact convergence flags", "fit objective"),
    )


def _curvature_coordinates(fit: DenseSolverResult) -> tuple[int, str]:
    face = fit.coefficient_face
    if face is None:
        return len(fit.coefficients), "full"
    return face.reduced_width, "reduced_exact_face"


def _retained_policy_curvature(context: _LSSDiagnosticContext) -> np.ndarray:
    fit = context.terminal_fit
    matrix = (
        fit.terminal_data_curvature
        if context.terminal_policy_matrix_kind == "data"
        else fit.terminal_penalized_curvature
    )
    face = fit.coefficient_face
    return np.asarray(matrix, dtype=np.float64) if face is None else face.reduce_matrix(matrix)


def _negative_eigenvalue_resolution(matrix: np.ndarray) -> float:
    width = matrix.shape[0]
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    relative_bar = max(
        100.0 * SHARED_RANK_POLICY.gram_rcond,
        _eigensolver_relative_bar(width),
    )
    return relative_bar * scale


def _curvature_indefinite_finding(
    context: _LSSDiagnosticContext,
) -> DiagnosticFinding | None:
    fit = context.terminal_fit
    telemetry = fit.terminal_curvature
    minimum = float(telemetry.minimum_eigenvalue)
    active_width, coordinate_space = _curvature_coordinates(fit)
    if active_width == 0:
        return None
    used_fallback = telemetry.actual_source != telemetry.requested_source
    material_fallback = used_fallback and telemetry.reason == "material_indefiniteness_after_retry"
    resolution: float | None = None
    if used_fallback:
        # The accepted terminal matrix is Fisher curvature, but the retained
        # minimum is from the rejected requested spectrum.  Its matrix scale is
        # not retained, so the solver's materiality decision is the only sound
        # authority; do not compare that minimum with the accepted Fisher norm.
        if not material_fallback:
            return None
    else:
        if minimum >= 0.0:
            return None
        resolution = _negative_eigenvalue_resolution(_retained_policy_curvature(context))
        if minimum >= -resolution:
            return None

    matrix_kind = context.terminal_policy_matrix_kind
    coordinate_phrase = (
        "full coefficient coordinates"
        if coordinate_space == "full"
        else "retained exact-face coordinates"
    )
    source_label = f"requested {telemetry.requested_source} {matrix_kind} curvature"
    if resolution is None:
        headline = f"The terminal policy found the {source_label} materially indefinite."
        observed = (
            f"The {source_label} in {coordinate_phrase} recorded raw minimum eigenvalue "
            f"{minimum:.3e}; the terminal policy's diagonally equilibrated analysis classified "
            f"the requested matrix as materially indefinite before accepting "
            f"{telemetry.actual_source} curvature."
        )
        minimum_evidence = DiagnosticEvidence(
            metric="minimum_eigenvalue",
            value=minimum,
            unit=None,
            window="requested terminal coefficient curvature",
            provenance=_CURVATURE_TELEMETRY,
        )
        resolution_caveat = (
            "The raw minimum eigenvalue and the policy's diagonally equilibrated rank analysis "
            "are different diagnostics. The requested matrix norm and numerical threshold are "
            "not retained after fallback; materiality comes from the terminal curvature-policy "
            f"decision, not from the accepted {telemetry.actual_source} rank or condition estimate."
        )
    else:
        headline = f"The {source_label} has a resolved negative eigenvalue."
        observed = (
            f"The {source_label} in {coordinate_phrase} has minimum eigenvalue {minimum:.3e}, "
            f"below the sign-resolution boundary {-resolution:.3e}."
        )
        minimum_evidence = DiagnosticEvidence(
            metric="minimum_eigenvalue",
            value=minimum,
            unit=None,
            window="requested terminal coefficient curvature",
            provenance=_CURVATURE_TELEMETRY,
            comparator="<",
            threshold=-resolution,
        )
        resolution_caveat = (
            "The sign boundary uses the shared Gram-rank policy, retained coordinate dimension, "
            "floating-point epsilon, and the retained matrix spectral norm."
        )

    if telemetry.requested_source == "observed":
        interpretation = (
            "This is resolved local nonconcavity of the unpenalized likelihood geometry in a "
            "active coefficient direction."
            if matrix_kind == "data"
            else (
                "This is resolved local nonconcavity of the penalized objective geometry in a "
                "active coefficient direction."
            )
        )
        action_question = (
            "Does comparing requested and accepted curvature, terminal score, and held-out "
            "objective isolate a consequential nonconcave direction?"
        )
    else:
        interpretation = (
            f"The {source_label} matrix has a resolved negative direction in the assessed "
            "coefficient coordinates."
        )
        action_question = (
            "Does comparing requested and accepted curvature, terminal score, and held-out "
            "objective isolate a consequential indefinite curvature direction?"
        )
    return _finding(
        "fit.curvature_indefinite",
        priority=1,
        severity="warning",
        confidence="strong",
        impacts=("fit_reliability",),
        headline=headline,
        observed=observed,
        interpretation=interpretation,
        caveats=(
            "Curvature alone does not show that the likelihood is increasing in this direction, "
            "identify its statistical cause, or establish that a different accepted curvature "
            "matrix is indefinite.",
            resolution_caveat,
        ),
        evidence=(
            minimum_evidence,
            *_evidence_set(
                _CURVATURE_TELEMETRY,
                ("requested_source", telemetry.requested_source),
                ("accepted_source", telemetry.actual_source),
                ("curvature_policy_reason", telemetry.reason),
            ),
            *_evidence_set(
                "solver_history",
                ("curvature_matrix", matrix_kind),
                ("coordinate_space", coordinate_space),
                ("active_coordinate_dimension", active_width),
            ),
        ),
        action_kind="compare_curvature_policy",
        action_question=action_question,
        action_metrics=("minimum_eigenvalue", "terminal score", "holdout log-likelihood"),
    )


def _analyze_fit_status(context: _LSSDiagnosticContext) -> tuple[DiagnosticFinding, ...]:
    smoothing = context.smoothing
    findings: list[DiagnosticFinding] = []
    terminal_failed = not context.terminal_fit.converged
    smoothing_failed = smoothing is not None and not smoothing.converged
    if terminal_failed:
        findings.append(
            _not_converged_finding(
                loop="coefficient",
                reason=context.terminal_fit.convergence_reason,
                iterations=context.terminal_fit.iterations,
                provenance="solver_history",
            )
        )
    if smoothing_failed:
        findings.append(
            _not_converged_finding(
                loop="smoothing",
                reason=smoothing.convergence_reason,
                iterations=smoothing.iterations,
                provenance=_SMOOTHING_HISTORY,
            )
        )
    if smoothing_failed and smoothing.convergence_reason == "gradient_unresolved":
        findings.extend(_gradient_unresolved_findings(context, smoothing))
    if not context.fitted_result.converged and not terminal_failed and not smoothing_failed:
        findings.append(_compact_not_converged_finding(context.fitted_result))
    curvature = _curvature_indefinite_finding(context)
    if curvature is not None:
        findings.append(curvature)
    return tuple(findings)


def _gradient_unresolved_findings(
    context: _LSSDiagnosticContext,
    smoothing: DistributionalEFSResult,
) -> tuple[DiagnosticFinding, ...]:
    """One error per component whose gradient certificate exceeds the stationarity bar."""
    gradient = smoothing.terminal_gradient or {}
    certificate = smoothing.terminal_gradient_certificate or {}
    bar = smoothing.stationarity_bar
    findings: list[DiagnosticFinding] = []
    for name, value in certificate.items():
        if value <= bar:
            continue
        findings.append(
            _finding(
                "fit.gradient_unresolved",
                _component_subject(context, name),
                priority=1,
                identifier=f"fit.gradient_unresolved:{name}",
                severity="error",
                impacts=("fit_validity", "smoothing_selection"),
                headline=(
                    f"The LAML gradient of {name!r} could not be certified at its finite-difference "
                    "step."
                ),
                observed=(
                    f"Its gradient {gradient.get(name, 0.0):.3g} carries a certificate of "
                    f"{value:.3g} against a stationarity bar of {bar:.3g}."
                ),
                interpretation=(
                    "The Newton endgame cannot tell this component's gradient from zero, so the "
                    "REML optimum is not certified along it."
                ),
                caveats=(
                    "The certificate bounds finite-difference error; it does not locate the "
                    "optimum.",
                ),
                evidence=(
                    *_evidence_set(
                        _SMOOTHING_HISTORY,
                        ("terminal_gradient", float(gradient.get(name, 0.0))),
                        ("terminal_gradient_certificate", float(value)),
                        ("derivative_step", float(smoothing.config.derivative_step)),
                    ),
                    _stationarity_evidence(smoothing),
                    _gradient_certificate_evidence(smoothing),
                ),
                action_kind="investigate_smoothing_convergence",
                action_question=(
                    "Does a larger finite-difference step or an analytic third derivative "
                    "certify this component's gradient?"
                ),
                action_metrics=(_STATIONARITY_METRIC, _ALL_FITTED_PARAMETERS),
            )
        )
    return tuple(findings)


def _analyze_authority(context: _LSSDiagnosticContext) -> tuple[DiagnosticFinding, ...]:
    smoothing = context.smoothing
    if (
        smoothing is None
        or not context.fitted_result.converged
        or not context.terminal_fit.converged
        or not smoothing.converged
        or smoothing.matched_certified
    ):
        return ()
    source_mismatches = sum(
        fit.terminal_curvature.requested_source != fit.terminal_curvature.actual_source
        for fit in smoothing.coefficient_fits
    )
    kind = _stop_kind(smoothing)
    wording = _stop_wording(kind)
    stationary_evidence = (
        (_gradient_certificate_evidence(smoothing),) if kind == "stationary" else ()
    )
    return (
        _finding(
            "fit.termination_uncertified",
            priority=1,
            impacts=("fit_reliability", "inference_reliability"),
            severity=cast(_FindingSeverity, wording["termination.severity"]),
            headline=wording["termination.headline"],
            observed=wording["termination.observed"],
            interpretation=wording["termination.interpretation"],
            caveats=(
                "This finding does not change or reinterpret either authoritative convergence "
                "flag.",
            ),
            evidence=(
                *_evidence_set(_SMOOTHING_HISTORY, ("matched_certified", False)),
                *_evidence_set(
                    _SMOOTHING_HISTORY,
                    ("terminal_reason", smoothing.convergence_reason),
                ),
                *_evidence_set(
                    _CURVATURE_TELEMETRY,
                    ("curvature_fallback_count", smoothing.fallback_count),
                    ("curvature_source_mismatch_fits", source_mismatches),
                ),
                _stationarity_evidence(smoothing),
                *stationary_evidence,
            ),
            action_kind=wording["termination.action_kind"],
            action_question=wording["termination.action_question"],
            action_metrics=_TERMINATION_ACTION_METRICS[kind],
        ),
    )


_TERMINATION_ACTION_METRICS = {
    "practical": (_ALL_FITTED_PARAMETERS, "standard errors", "effective degrees of freedom"),
    "strict": (_ALL_FITTED_PARAMETERS, "curvature source", "standard errors"),
    "stationary": (_ALL_FITTED_PARAMETERS, "curvature source", "standard errors"),
}


def _stop_wording(kind: StopKind) -> dict[str, str]:
    """The practical-plateau, stationary or strict wording of every stop-related finding.

    One table, so no finding phrases the same stop two ways.  The
    ``lambda_cap_unresolved`` finding has no practical variant: a practical
    plateau is an interior stop and cannot coexist with an unresolved cap.
    A stationary stop is the Newton endgame's: its authority is the projected
    exact LAML gradient, not the Fellner--Schall residual.
    """
    if kind == "practical":
        return {
            "termination.severity": "info",
            "termination.headline": (
                "The fit used practical REML convergence, not strict stationarity."
            ),
            "termination.observed": (
                "The accepted objective and fitted parameters met the configured practical "
                "plateau, while strict matched certification remains false."
            ),
            "termination.interpretation": (
                "This is the requested fast stopping policy, not a fit failure. Use strict REML "
                "when matched stationarity evidence is required."
            ),
            "termination.action_kind": "compare_strict_reml",
            "termination.action_question": (
                "Does strict REML materially change fitted parameters or uncertainty?"
            ),
            "trajectory.headline": "Practical REML stopped before strict smoothing stationarity.",
            "trajectory.observed": (
                "The practical plateau passed, but the retained strict stationarity residual "
                "still exceeds its tolerance."
            ),
            "trajectory.interpretation_info": (
                "This is expected under practical convergence and does not mean the fit failed."
            ),
            "trajectory.interpretation": (
                "The practical plateau accepted the stop while the strict stationarity "
                "residual was still material, so the smoothing selection is loose."
            ),
            "trajectory.caveat": (
                "Compare with strict REML when small smoothing or uncertainty differences matter."
            ),
        }
    if kind == "stationary":
        return {
            "termination.severity": "warning",
            "termination.headline": (
                "The stationary smoothing result lacks matched certification."
            ),
            "termination.observed": (
                "The smoothing search ended at the REML optimum (a stationary point of the LAML "
                "criterion under its projected gradient), but the retained algorithm-matched "
                "certification predicate is false."
            ),
            "termination.interpretation": (
                "The optimum was located, but a curvature fallback, a refused endpoint "
                "assessment, a non-analytic exact face or an uncertified gradient component "
                "keeps the stronger certification contract unsatisfied."
            ),
            "termination.action_kind": "compare_certified_configuration",
            "termination.action_question": (
                "Does a configuration that preserves the requested curvature authority "
                "produce materially different fitted parameters or uncertainty?"
            ),
            "trajectory.headline": (
                "The Fellner--Schall residual is not the authority at a stationary stop."
            ),
            "trajectory.observed": (
                "The retained Fellner--Schall residual exceeds the outer tolerance at the "
                "REML optimum, where that fixed point does not sit."
            ),
            "trajectory.interpretation_info": (
                "This is expected at a stationary stop and does not mean the fit failed."
            ),
            "trajectory.interpretation": (
                "This is expected at a stationary stop and does not mean the fit failed."
            ),
            "trajectory.caveat": "The projected exact gradient is the stop's certificate.",
        }
    return {
        "termination.severity": "warning",
        "termination.headline": "The converged smoothing result lacks matched certification.",
        "termination.observed": (
            "The accepted inner and outer results report convergence, but the retained "
            "algorithm-matched certification predicate is false."
        ),
        "termination.interpretation": (
            "The fitted values exist, but the stronger accepted certification contract "
            "was not satisfied."
        ),
        "termination.action_kind": "compare_certified_configuration",
        "termination.action_question": (
            "Does a configuration that preserves the requested curvature authority "
            "produce materially different fitted parameters or uncertainty?"
        ),
        "trajectory.headline": "The terminal smoothing evidence remains outside its tolerance.",
        "trajectory.observed": (
            "The retained raw terminal stationarity evidence exceeds the configured "
            "outer tolerance."
        ),
        "trajectory.interpretation_info": (
            "The accepted smoothing trajectory did not settle under its rule."
        ),
        "trajectory.interpretation": "The accepted smoothing trajectory did not settle under its rule.",
        "trajectory.caveat": "This aggregate terminal residual does not identify a causal component.",
    }


def _endpoint_refusal_reason(smoothing: DistributionalEFSResult, component_name: str) -> str | None:
    for iteration in reversed(smoothing.history):
        if component_name not in iteration.refused_face_components:
            continue
        if iteration.endpoint_assessment_failure_reason is not None:
            return iteration.endpoint_assessment_failure_reason
        if iteration.endpoint_direction_evidence is not None:
            return f"direction_{iteration.endpoint_direction_evidence.decision}"
    return None


def _analyze_lambda_caps(context: _LSSDiagnosticContext) -> tuple[DiagnosticFinding, ...]:
    smoothing = context.smoothing
    if smoothing is None or not smoothing.unresolved_upper_bound:
        return ()
    findings: list[DiagnosticFinding] = []
    for name in smoothing.unresolved_upper_bound:
        refusal_reason = _endpoint_refusal_reason(smoothing, name)
        evidence = [
            *_terminal_lambda_evidence(smoothing, name),
            _stationarity_evidence(smoothing),
        ]
        if refusal_reason is not None:
            evidence.extend(
                _evidence_set(
                    _SMOOTHING_HISTORY,
                    ("endpoint_refusal_reason", refusal_reason),
                )
            )
        findings.append(
            _finding(
                "fit.lambda_cap_unresolved",
                _component_subject(context, name),
                priority=1,
                identifier=f"fit.lambda_cap_unresolved:{name}",
                severity="error",
                impacts=("fit_validity", "smoothing_selection"),
                headline=f"Penalty component {name!r} has unresolved pressure at its upper cap.",
                observed=(
                    "The accepted smoothing result retained this component in its unresolved "
                    "upper-bound set."
                ),
                interpretation=(
                    "The finite configured cap did not supply accepted convergence or "
                    "exact-face authority for this component."
                ),
                caveats=(
                    "Reaching the cap is not convergence and does not establish an "
                    "infinite optimum.",
                ),
                evidence=tuple(evidence),
                action_kind="investigate_lambda_boundary",
                action_question=(
                    "Does a wider certified boundary or an alternative term representation "
                    "resolve the terminal smoothing evidence?"
                ),
                action_metrics=(_STATIONARITY_METRIC, _ALL_FITTED_PARAMETERS, "predictions"),
            )
        )
    return tuple(findings)


def _analyze_smoothing_state(context: _LSSDiagnosticContext) -> tuple[DiagnosticFinding, ...]:
    smoothing = context.smoothing
    if smoothing is None:
        return ()
    findings: list[DiagnosticFinding] = []
    for name, terminal_lambda in smoothing.lambdas.items():
        if terminal_lambda != smoothing.config.maximum_lambda:
            continue
        findings.append(
            _finding(
                "smoothing.penalty_at_upper_boundary",
                _component_subject(context, name),
                priority=6,
                identifier=f"smoothing.penalty_at_upper_boundary:{name}",
                severity="info",
                impacts=("smoothing_selection",),
                headline=f"Penalty component {name!r} ended at its configured upper bound.",
                observed="Its accepted terminal lambda equals the configured finite maximum.",
                interpretation="The smoothing trajectory reached the boundary of its search box.",
                caveats=(
                    "A finite upper-bound lambda alone does not establish an endpoint, exact "
                    "suppression, or an infinite optimum.",
                ),
                evidence=_terminal_lambda_evidence(smoothing, name),
                action_kind="inspect_smoothing_boundary",
                action_question=(
                    "Does changing the finite search boundary alter the accepted stationarity "
                    "evidence or fitted parameters?"
                ),
                action_metrics=(_STATIONARITY_METRIC, _ALL_FITTED_PARAMETERS),
            )
        )
    residual = smoothing.terminal_convergence_max_log_residual
    # At a stationary stop the Fellner--Schall residual is not the authority:
    # the exact gradient is, and it is within its bar by construction.
    if residual > smoothing.config.tolerance and _stop_kind(smoothing) != "stationary":
        wording = _stop_wording(_stop_kind(smoothing))
        severity = _residual_severity(residual, smoothing.config.tolerance)
        findings.append(
            _finding(
                "smoothing.trajectory_unsettled",
                priority=3,
                severity=severity,
                impacts=(
                    ("smoothing_selection",)
                    if severity != "error"
                    else ("fit_validity", "smoothing_selection")
                ),
                headline=wording["trajectory.headline"],
                observed=wording["trajectory.observed"],
                interpretation=wording[
                    "trajectory.interpretation_info"
                    if severity == "info"
                    else "trajectory.interpretation"
                ],
                caveats=(wording["trajectory.caveat"],),
                evidence=(
                    _stationarity_evidence(smoothing),
                    *_evidence_set(
                        _SMOOTHING_HISTORY,
                        ("terminal_reason", smoothing.convergence_reason),
                    ),
                ),
                action_kind="investigate_smoothing_convergence",
                action_question="Which controlled change brings the outer residual within tolerance?",
                action_metrics=(_STATIONARITY_METRIC, "outer iterations", _ALL_FITTED_PARAMETERS),
            )
        )
    return tuple(findings)


def _finite_component_rank(component: PenaltyComponent) -> float | None:
    try:
        rank = float(component.rank)
    except (TypeError, ValueError, OverflowError):
        return None
    return rank if np.isfinite(rank) else None


def _component_null_dimension(component: PenaltyComponent) -> int | None:
    width = component.group_sl.stop - component.group_sl.start
    rank = _finite_component_rank(component)
    if rank is None:
        return None
    rounded = round(rank)
    tolerance = 32.0 * max(width, 1) * np.finfo(np.float64).eps * max(1.0, abs(rank))
    if abs(rank - rounded) > tolerance or not 0 <= rounded <= width:
        return None
    return width - int(rounded)


def _supports_linear_spline_counterfactual(
    context: _LSSDiagnosticContext,
    component: PenaltyComponent,
    null_dimension: int | None,
) -> bool:
    """Whether an exact face on this component leaves exactly a linear effect.

    True only for the wiggle penalty of a plain cubic regression spline whose
    penalty null space is one-dimensional: the surviving subspace is then the
    linear trend and a linear counterfactual is the natural comparison.
    """
    return bool(
        null_dimension == 1
        and component.group_name in context.linear_spline_terms
        and component.name.endswith("#wiggle")
    )


def _exact_face_effect(
    context: _LSSDiagnosticContext,
    component: PenaltyComponent,
    null_dimension: int | None,
) -> Literal["fully_suppressed", "linear_only", "null_space_only", "unresolved"]:
    if null_dimension is None:
        return "unresolved"
    if null_dimension == 0:
        return "fully_suppressed"
    if _supports_linear_spline_counterfactual(context, component, null_dimension):
        return "linear_only"
    return "null_space_only"


def _analyze_exact_faces(context: _LSSDiagnosticContext) -> tuple[DiagnosticFinding, ...]:
    findings: list[DiagnosticFinding] = []
    for name in context.exact_face_components:
        component = _component(context, name)
        width = component.group_sl.stop - component.group_sl.start
        null_dimension = _component_null_dimension(component)
        evidence = [
            *_evidence_set(_SMOOTHING_HISTORY, ("exact_face", True)),
            *_evidence_set("fit_result", ("block_width", width)),
        ]
        rank = _finite_component_rank(component)
        if rank is not None:
            evidence.extend(_evidence_set("fit_result", ("component_rank", rank)))
        caveats: tuple[str, ...] = ()
        if null_dimension is not None:
            evidence.extend(
                _evidence_set(
                    "fit_result",
                    ("retained_null_space_dimension", null_dimension),
                )
            )
            if null_dimension > 0:
                caveats = (
                    "A retained null-space effect remains; this evidence does not show that the "
                    "feature is useless or should be removed.",
                )
        else:
            caveats = (
                "The component rank metadata is unresolved, so no null-space dimension is claimed.",
            )
        findings.append(
            _finding(
                "smoothing.penalized_subspace_suppressed",
                _penalty_subject(context.layout, component),
                priority=4,
                severity="info",
                confidence=("certified" if null_dimension is not None else "unresolved"),
                impacts=("model_complexity", "smoothing_selection"),
                headline=f"The exact face suppressed the penalized subspace governed by {name!r}.",
                observed=(
                    "Accepted exact-face evidence removed only the penalized coefficient subspace "
                    "governed by this component."
                ),
                interpretation="The fitted model retained no flexibility in those penalized directions.",
                caveats=caveats,
                evidence=tuple(evidence),
                action_kind="compare_simplified_representation",
                action_question=(
                    "Does a representation matching the surviving subspace preserve fitted "
                    "outputs and held-out behavior?"
                ),
                action_metrics=(
                    _ALL_FITTED_PARAMETERS,
                    "predictions",
                    "held-out performance",
                    "fit work",
                ),
            )
        )
    return tuple(findings)


def _is_face_assessment(iteration: DistributionalEFSIteration) -> bool:
    return bool(
        iteration.activated_face_components
        or iteration.deactivated_face_components
        or iteration.revalidated_face_components
        or iteration.refused_face_components
    )


def _ordinary_updates(smoothing: DistributionalEFSResult) -> tuple[DistributionalEFSIteration, ...]:
    return tuple(
        item for item in smoothing.history if item.accepted and not _is_face_assessment(item)
    )


def _dominant_update_summary(
    smoothing: DistributionalEFSResult,
) -> tuple[dict[str, float], dict[str, int], int, dict[str, float]]:
    """Per component: equivalent wins, tie counts, the eligible-update count, and win shares.

    An accepted ordinary update is eligible when some component moved; the
    components with the largest absolute log step share one win equally.
    """
    names = tuple(smoothing.lambdas)
    wins = dict.fromkeys(names, 0.0)
    tie_updates = dict.fromkeys(names, 0)
    eligible = 0
    for item in _ordinary_updates(smoothing):
        magnitudes = {name: abs(item.accepted_log_steps[name]) for name in names}
        largest = max(magnitudes.values(), default=0.0)
        if largest == 0.0:
            continue
        leaders = tuple(name for name in names if magnitudes[name] == largest)
        eligible += 1
        for name in leaders:
            wins[name] += 1.0 / len(leaders)
            tie_updates[name] += len(leaders) > 1
    shares = (
        dict.fromkeys(names, 0.0)
        if eligible == 0
        else {name: wins[name] / eligible for name in names}
    )
    return wins, tie_updates, eligible, shares


def _build_profile(
    context: _LSSDiagnosticContext,
    phase_snapshot: FitPhaseSnapshot | None,
) -> FitWorkProfile:
    """Summarise the accepted fit's work, timing shares, and smoothing outcomes.

    Every published number is one the profile records validate: shares lie in
    [0, 1], leaf phase seconds plus the unmeasured remainder sum to the fit
    total, and an exact-face outcome names the iteration that activated it.
    """
    smoothing = context.smoothing
    if smoothing is None:
        ordinary_iterations: tuple[DistributionalEFSIteration, ...] = ()
        shares: dict[str, float] = {}
        initial_lambdas = context.fitted_result.smoothing_parameters
        coefficient_fits = 1
        outer_iterations = 0
        unresolved: set[str] = set()
        maximum_lambda = None
    else:
        ordinary_iterations = tuple(
            item for item in smoothing.history if not _is_face_assessment(item)
        )
        _wins, _ties, _eligible, shares = _dominant_update_summary(smoothing)
        initial_lambdas = smoothing.initial_lambdas
        coefficient_fits = len(smoothing.coefficient_fits)
        outer_iterations = smoothing.iterations
        unresolved = set(smoothing.unresolved_upper_bound)
        maximum_lambda = smoothing.config.maximum_lambda

    exact_faces = set(context.exact_face_components)
    smoothing_components: list[SmoothingComponentProfile] = []
    for component in context.layout.penalties:
        name = component.name
        policy = component.lambda_policy
        # Without a smoothing result every lambda was fixed by the caller.
        fixed = smoothing is None or (policy is not None and policy.mode == "fixed")
        final_lambda = float(context.fitted_result.smoothing_parameters[name])
        null_dimension = _component_null_dimension(component)
        exact_face_iteration = None
        if smoothing is not None and name in exact_faces:
            exact_face_iteration = next(
                (
                    item.iteration
                    for item in smoothing.history
                    if name in item.activated_face_components
                ),
                None,
            )
            if exact_face_iteration is None:
                raise ValueError(
                    f"accepted exact face {name!r} has no activating smoothing iteration"
                )
        if fixed:
            outcome = "fixed"
        elif name in exact_faces:
            outcome = "exact_face"
        elif name in unresolved:
            outcome = "unresolved_cap"
        elif maximum_lambda is not None and final_lambda == maximum_lambda:
            outcome = "upper_bound"
        else:
            outcome = "finite"
        accepted_moves = (
            0
            if fixed
            else sum(
                item.accepted_log_steps[name] != 0.0
                for item in ordinary_iterations
                if item.accepted
            )
        )
        upper_bound_iterations = (
            0
            if fixed or maximum_lambda is None
            else sum(item.lambdas_after[name] == maximum_lambda for item in ordinary_iterations)
        )
        smoothing_components.append(
            SmoothingComponentProfile(
                name=name,
                predictor=component.group_name.partition(":")[0],
                term=component.group_name.partition(":")[2],
                initial_lambda=float(initial_lambdas[name]),
                final_lambda=final_lambda,
                accepted_moves=accepted_moves,
                dominant_update_share=shares.get(name, 0.0),
                terminal_term_edf=context.fitted_result.term_edf.get(component.group_name),
                null_space_dimension=null_dimension,
                outcome=outcome,
                exact_face_iteration=exact_face_iteration,
                exact_face_effect=(
                    _exact_face_effect(context, component, null_dimension)
                    if outcome == "exact_face"
                    else None
                ),
                upper_bound_iterations=upper_bound_iterations,
            )
        )

    if phase_snapshot is None:
        fit_seconds = None
        phases: tuple[FitPhaseProfile, ...] = ()
    else:
        fit_seconds = float(phase_snapshot.seconds["fit_total"])
        leaf_phases = tuple(
            FitPhaseProfile(
                name=profile_name,
                seconds=float(phase_snapshot.seconds[source_name]),
                fit_share=(
                    float(phase_snapshot.seconds[source_name]) / fit_seconds
                    if fit_seconds > 0.0
                    else 0.0
                ),
                calls=phase_snapshot.counts[source_name],
            )
            for source_name, profile_name in _PROFILE_LEAF_PHASES
            if phase_snapshot.counts[source_name] > 0
        )
        measured_seconds = sum(item.seconds for item in leaf_phases)
        timing_error = 64.0 * np.finfo(np.float64).eps * max(fit_seconds, measured_seconds, 1.0)
        if measured_seconds > fit_seconds + timing_error:
            raise ValueError("leaf phase timings cannot exceed the total fit time")
        unmeasured_seconds = max(fit_seconds - measured_seconds, 0.0)
        phases = (
            *leaf_phases,
            FitPhaseProfile(
                name="orchestration_and_unmeasured",
                seconds=unmeasured_seconds,
                fit_share=(unmeasured_seconds / fit_seconds if fit_seconds > 0.0 else 0.0),
                calls=0,
            ),
        )

    return FitWorkProfile(
        n_observations=context.n_observations,
        n_coefficients=context.layout.n_coefficients,
        fit_seconds=fit_seconds,
        outer_iterations=outer_iterations,
        coefficient_fits=coefficient_fits,
        inner_iterations=context.fitted_result.n_inner_iter,
        rejected_proposals=sum(not item.accepted for item in ordinary_iterations),
        backtracked_proposals=sum(item.backtracks > 0 for item in ordinary_iterations),
        phases=phases,
        smoothing_components=tuple(smoothing_components),
    )


def _analyze_accepted_trajectory(
    context: _LSSDiagnosticContext,
) -> tuple[DiagnosticFinding, ...]:
    smoothing = context.smoothing
    if smoothing is None or smoothing.convergence_reason == "stationary":
        # The dominance and drift readings describe a Fellner--Schall trajectory
        # judged by its residual; a stationary stop's authority is the exact
        # gradient, and its Newton steps are not that trajectory.
        return ()
    updates = _ordinary_updates(smoothing)
    names = tuple(smoothing.lambdas)
    eligible: list[tuple[str, ...]] = []
    for item in updates:
        magnitudes = {name: abs(item.accepted_log_steps[name]) for name in names}
        largest = max(magnitudes.values(), default=0.0)
        if largest > 0.0:
            eligible.append(tuple(name for name in names if magnitudes[name] == largest))

    wins = dict.fromkeys(names, 0.0)
    tie_updates = dict.fromkeys(names, 0)
    for tied in eligible:
        equivalent_win = 1.0 / len(tied)
        for name in tied:
            wins[name] += equivalent_win
            tie_updates[name] += len(tied) > 1

    findings: list[DiagnosticFinding] = []
    denominator = len(eligible)
    if denominator >= _MIN_DOMINANCE_UPDATES:
        for name in names:
            share = wins[name] / denominator
            if wins[name] < _MIN_DOMINANCE_EQUIVALENT_WINS or share < _DOMINANCE_SHARE:
                continue
            findings.append(
                _finding(
                    "smoothing.update_dominance",
                    _component_subject(context, name),
                    priority=5,
                    confidence="strong",
                    headline=f"Penalty component {name!r} dominated accepted smoothing movement.",
                    observed=(
                        f"It accumulated {wins[name]:g} equivalent wins across {denominator} "
                        "eligible ordinary updates."
                    ),
                    interpretation="This component accounted for a substantial share of movement.",
                    caveats=(
                        "Accepted movement is not wall-time attribution or evidence of causality.",
                    ),
                    evidence=(
                        *_evidence_set(
                            _SMOOTHING_HISTORY,
                            ("equivalent_wins", wins[name]),
                            ("work_share", share),
                            ("tie_updates", tie_updates[name]),
                            ("eligible_updates", denominator),
                        ),
                        *_terminal_lambda_evidence(smoothing, name),
                        _stationarity_evidence(smoothing),
                    ),
                    action_kind="compare_smoothing_representation",
                    action_question=(
                        "Does an alternative representation reduce accepted smoothing movement "
                        "without degrading fitted outputs?"
                    ),
                    action_metrics=("ordinary updates", _ALL_FITTED_PARAMETERS, "predictions"),
                )
            )

    for name in names:
        moves = tuple(item.accepted_log_steps[name] for item in updates)
        positive = sum(value > 0.0 for value in moves)
        negative = sum(value < 0.0 for value in moves)
        zero = len(moves) - positive - negative
        if positive < _MIN_UPWARD_DRIFT_UPDATES or negative:
            continue
        findings.append(
            _finding(
                "smoothing.persistent_upward_drift",
                _component_subject(context, name),
                priority=3,
                confidence="strong",
                headline=f"Penalty component {name!r} moved upward without an accepted reversal.",
                observed=f"It had {positive} positive nonzero moves and no negative moves.",
                interpretation="The accepted ordinary trajectory continued toward stronger penalty.",
                caveats=(
                    "Upward drift does not establish an endpoint, exact suppression, or an "
                    "infinite optimum.",
                ),
                evidence=(
                    *_evidence_set(
                        _SMOOTHING_HISTORY,
                        ("positive_accepted_moves", positive),
                        ("negative_accepted_moves", negative),
                        ("zero_accepted_moves", zero),
                        (
                            "persistence_share",
                            positive / max(positive + negative, 1),
                        ),
                    ),
                    *_terminal_lambda_evidence(smoothing, name),
                    _stationarity_evidence(smoothing),
                ),
                action_kind="inspect_upward_trajectory",
                action_question="Does a controlled alternative settle this upward trajectory?",
                action_metrics=(
                    "accepted smoothing moves",
                    _STATIONARITY_METRIC,
                    _ALL_FITTED_PARAMETERS,
                ),
            )
        )
    return tuple(findings)


def _analyze_work_pattern(context: _LSSDiagnosticContext) -> tuple[DiagnosticFinding, ...]:
    smoothing = context.smoothing
    if smoothing is None:
        return ()
    ordinary_iterations = tuple(item for item in smoothing.history if not _is_face_assessment(item))
    backtracked = sum(item.backtracks > 0 for item in ordinary_iterations)
    rejected = sum(not item.accepted for item in ordinary_iterations)
    exhausted = any(
        not item.accepted and item.backtracks >= smoothing.config.max_backtracks
        for item in ordinary_iterations
    )
    if backtracked < 2 and not exhausted:
        return ()
    total_backtracks = sum(item.backtracks for item in ordinary_iterations)
    evidence = (
        *_evidence_set(
            _SMOOTHING_HISTORY,
            ("outer_iterations", smoothing.iterations),
            ("accepted_ordinary_updates", len(_ordinary_updates(smoothing))),
            ("rejected_iterations", rejected),
            ("backtracked_outer_iterations", backtracked),
            ("total_backtracks", total_backtracks),
            ("configured_backtrack_budget", smoothing.config.max_backtracks),
        ),
        _stationarity_evidence(smoothing),
    )
    return (
        _finding(
            "optimization.repeated_step_rejection",
            priority=5,
            confidence="strong",
            headline="Outer smoothing proposals repeatedly required rejection safeguards.",
            observed=(
                f"Backtracking and rejection added coefficient refits across "
                f"{smoothing.iterations} outer iterations."
            ),
            interpretation="The retained work counts show extra proposal-assessment work.",
            caveats=("No wall-time attribution was retained for those coefficient refits.",),
            evidence=evidence,
            action_kind="compare_smoothing_initialization",
            action_question="Does a controlled initialization change reduce rejected proposals?",
            action_metrics=("coefficient refits", "outer iterations", _ALL_FITTED_PARAMETERS),
        ),
    )


def _unique_sources(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _analyze_curvature(context: _LSSDiagnosticContext) -> tuple[DiagnosticFinding, ...]:
    smoothing = context.smoothing
    if smoothing is None or smoothing.fallback_count <= 0:
        return ()
    fallback_telemetry = tuple(
        fit.terminal_curvature
        for fit in smoothing.coefficient_fits
        if fit.terminal_curvature.fallback_count > 0
    )
    fallback_count = smoothing.fallback_count
    noun = "fallback" if fallback_count == 1 else "fallbacks"
    return (
        _finding(
            "optimization.curvature_fallback_repeated",
            priority=2,
            confidence="strong",
            impacts=("fit_reliability", "inference_reliability"),
            headline="Coefficient fitting used curvature fallback authority.",
            observed=(
                f"The retained coefficient-fit chronology recorded {fallback_count} curvature "
                f"{noun} across {len(fallback_telemetry)} coefficient fits."
            ),
            interpretation=(
                "One or more requested curvature sources could not be used for their accepted "
                "coefficient fit."
            ),
            caveats=(
                "Fallback counts are numerical work and authority evidence, not wall-time "
                "attribution.",
            ),
            evidence=_evidence_set(
                _CURVATURE_TELEMETRY,
                ("fallback_count", fallback_count),
                ("coefficient_fits_with_fallback", len(fallback_telemetry)),
                (
                    "requested_sources",
                    _unique_sources(tuple(item.requested_source for item in fallback_telemetry)),
                ),
                (
                    "actual_sources",
                    _unique_sources(tuple(item.actual_source for item in fallback_telemetry)),
                ),
            ),
            action_kind="compare_curvature_policy",
            action_question=(
                "Does a fit that retains its requested curvature source change the parameters, "
                "uncertainty, or convergence evidence?"
            ),
            action_metrics=(_ALL_FITTED_PARAMETERS, "standard errors", "convergence evidence"),
        ),
    )


def _analyze_numerics(context: _LSSDiagnosticContext) -> tuple[DiagnosticFinding, ...]:
    fit = context.terminal_fit
    if fit.coefficient_face is None:
        rank = fit.terminal_rank.rank
        width = context.layout.n_coefficients
        coordinate_space = "full"
    else:
        reduced_rank = fit.terminal_reduced_rank
        if reduced_rank is None:
            raise ValueError("accepted coefficient face lacks reduced-rank evidence")
        rank = reduced_rank.rank
        width = fit.coefficient_face.reduced_width
        coordinate_space = "reduced_exact_face"

    findings: list[DiagnosticFinding] = []
    if rank < width:
        findings.append(
            _finding(
                "numerics.rank_loss",
                priority=2,
                headline="The estimable rank is below the active coordinate dimension.",
                observed=f"The retained rank is {rank} in an active width of {width}.",
                interpretation="At least one active fitted direction is not numerically estimable.",
                caveats=(
                    "An intentionally removed exact-face direction is not counted as rank loss.",
                ),
                evidence=_evidence_set(
                    "fit_result",
                    ("estimable_rank", rank),
                    ("active_coordinate_dimension", width),
                    ("coordinate_space", coordinate_space),
                ),
                action_kind="investigate_rank_loss",
                action_question="Which representation restores estimability of the active fit?",
                action_metrics=("estimable rank", "predictions", _ALL_FITTED_PARAMETERS),
            )
        )

    condition = fit.terminal_curvature.condition_estimate
    if condition is not None and condition >= _CONDITION_WARNING_THRESHOLD:
        accepted_source = fit.terminal_curvature.actual_source
        matrix_kind = context.terminal_policy_matrix_kind
        source_label = f"accepted {accepted_source} {matrix_kind} curvature"
        findings.append(
            _finding(
                "numerics.conditioning_warning",
                priority=2,
                confidence="strong",
                headline=f"The {source_label} has a large pre-truncation condition estimate.",
                observed="Its pre-truncation condition estimate meets the warning threshold.",
                interpretation=(
                    "This is a factor-scale conditioning estimate for the diagonally equilibrated "
                    "policy matrix before numerical rank truncation."
                ),
                caveats=(
                    "The estimator and its norm depend on the accepted decomposition path; it is "
                    "not the condition number of the unscaled curvature matrix.",
                    "A post-truncation condition estimate is not retained, so this does not "
                    "establish amplification among retained fitted directions.",
                ),
                evidence=(
                    DiagnosticEvidence(
                        metric="condition_estimate",
                        value=float(condition),
                        unit=None,
                        window="accepted terminal coefficient fit",
                        provenance=_CURVATURE_TELEMETRY,
                        comparator=">=",
                        threshold=float(_CONDITION_WARNING_THRESHOLD),
                    ),
                    *_evidence_set(
                        _CURVATURE_TELEMETRY,
                        ("accepted_source", accepted_source),
                        ("condition_scope", "pre_truncation"),
                    ),
                    *_evidence_set(
                        "solver_history",
                        ("curvature_matrix", matrix_kind),
                        ("coordinate_space", coordinate_space),
                        ("active_coordinate_dimension", width),
                    ),
                ),
                action_kind="compare_conditioning",
                action_question="Does a better-conditioned representation preserve fitted outputs?",
                action_metrics=("condition estimate", "estimable rank", _ALL_FITTED_PARAMETERS),
            )
        )
    return tuple(findings)


_ANALYZERS = (
    _analyze_fit_status,
    _analyze_lambda_caps,
    _analyze_authority,
    _analyze_exact_faces,
    _analyze_smoothing_state,
    _analyze_accepted_trajectory,
    _analyze_work_pattern,
    _analyze_curvature,
    _analyze_numerics,
)


_SCOPE_ORDER = {
    "whole_fit": 0,
    "predictor": 1,
    "term": 2,
    "penalized_subspace": 3,
    "level": 4,
}
_CONFIDENCE_ORDER = {"certified": 0, "strong": 1, "suggestive": 2, "unresolved": 3}


def _persistence_or_work_share(finding: DiagnosticFinding) -> float:
    values = (
        float(evidence.value)
        for evidence in finding.evidence
        if evidence.metric in {"persistence_share", "work_share"}
        and not isinstance(evidence.value, bool)
        and isinstance(evidence.value, int | float)
    )
    return max(values, default=0.0)


def _rank_findings(
    findings: tuple[DiagnosticFinding, ...] | list[DiagnosticFinding],
) -> tuple[DiagnosticFinding, ...]:
    """Rank findings by deterministic urgency, scope, authority, and persistence."""
    return tuple(
        sorted(
            findings,
            key=lambda finding: (
                finding.priority_tier,
                _SCOPE_ORDER[finding.subject.scope],
                _CONFIDENCE_ORDER[finding.confidence],
                -_persistence_or_work_share(finding),
                finding.code,
                finding.subject.identifier,
            ),
        )
    )


def _fit_status(context: _LSSDiagnosticContext) -> str:
    smoothing = context.smoothing
    if smoothing is None:
        if context.fitted_result.converged and context.terminal_fit.converged:
            return "fixed_fit"
        return "not_converged"
    if not all(
        (context.fitted_result.converged, context.terminal_fit.converged, smoothing.converged)
    ):
        return "not_converged"
    return "converged_certified" if smoothing.matched_certified else "converged_uncertified"


def diagnose_distributional_fit(
    model: DenseDistributionalModel,
    *,
    phase_snapshot: FitPhaseSnapshot | None = None,
) -> FitDiagnosticReport:
    """Diagnose one accepted dense distributional fit without reading row data.

    ``phase_snapshot`` is the phase timing captured around this fit alone;
    without it the work profile carries counts only and says so.
    """
    if phase_snapshot is not None and not isinstance(phase_snapshot, FitPhaseSnapshot):
        raise TypeError("phase_snapshot must be a FitPhaseSnapshot or None")
    context = _LSSDiagnosticContext.from_model(model)
    smoothing_coverage = (
        "No smoothing result or smoothing history exists for this fixed fit."
        if context.smoothing is None
        else (
            "Smoothing convergence, trajectory, coefficient-fit summaries, and exact-face "
            "evidence were examined."
        )
    )
    stationary_coverage = (
        (_stationary_statement(context.smoothing),)
        if context.smoothing is not None and context.smoothing.convergence_reason == "stationary"
        else ()
    )
    return FitDiagnosticReport(
        schema_version=2,
        rule_set_version=1,
        model_type="SuperLSS",
        family=context.family_name,
        fit_revision=context.revision,
        scope="fit",
        fit_status=_fit_status(context),
        findings=_rank_findings(
            tuple(finding for analyzer in _ANALYZERS for finding in analyzer(context))
        ),
        coverage=(
            "Accepted compact fit status and revision metadata were examined.",
            "Terminal coefficient-solver convergence, rank, and curvature telemetry were examined.",
            smoothing_coverage,
            *stationary_coverage,
        ),
        limitations=(
            "No row, response, weight, exposure, influence, or per-level evidence was examined.",
            (
                "No fit timing snapshot was retained; wall-time attribution is unavailable."
                if phase_snapshot is None
                else (
                    "Phase timings are measurements of this fit on this machine, not portable "
                    "benchmark claims or per-feature runtime attribution."
                )
            ),
            "No counterfactual refit recipe or training data was examined; this report cannot "
            "establish feature usefulness or causal effects.",
        ),
        profile=_build_profile(context, phase_snapshot),
    )
