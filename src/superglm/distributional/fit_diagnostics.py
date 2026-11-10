"""Read-only fit diagnostics for one accepted distributional revision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from superglm._roundoff import EPS
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
from superglm.distributional.results.fit import DistributionalFitResult
from superglm.distributional.results.iteration import DistributionalEFSIteration
from superglm.distributional.results.shape import (
    ShapeFit,
    shape_smoothing_sentence,
    shape_term_sentence,
)
from superglm.distributional.results.smoothing import DistributionalEFSResult
from superglm.distributional.results.solver import DenseSolverResult
from superglm.distributional.timing import FitPhaseSnapshot
from superglm.features.spline import CubicRegressionSpline
from superglm.solvers.rank import SHARED_RANK_POLICY, _eigensolver_relative_bar
from superglm.types import PenaltyComponent

_MIN_DOMINANCE_UPDATES = 4
_MIN_DOMINANCE_EQUIVALENT_WINS = 3.0
_DOMINANCE_SHARE = 0.5
_MIN_UPWARD_DRIFT_UPDATES = 4
_CONDITION_WARNING_THRESHOLD = 1.0 / np.sqrt(EPS)
_SMOOTHING_HISTORY = "smoothing_history"
_CURVATURE_TELEMETRY = "curvature_telemetry"
_ALL_FITTED_PARAMETERS = "all fitted distributional parameters"
_STATIONARITY_METRIC = "how settled the smoothing is"
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
# Phases the profile reports by name (recorder name -> profile name). Each clock
# interval is charged to the innermost open phase only, so nothing double
# counts; every other phase's exclusive time forms the orchestration row.
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
    shape_fit: ShapeFit | None
    null_converged: bool
    null_convergence_reason: str

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
                or _legacy_policy_matrix_kind(model)
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
            shape_fit=state.shape_fit,
            null_converged=state.null_model.converged,
            null_convergence_reason=str(state.null_model.convergence_reason),
        )


def _legacy_policy_matrix_kind(model: DenseDistributionalModel) -> _CurvatureMatrixKind:
    """The curvature scope schema-8 telemetry left unrecorded.

    Before schema 9 the policy assessed data curvature when the family had
    expected information, else penalized curvature; this reads what the
    artifact recorded about the family, not what it declares today.
    """
    recorded = model.recorded_expected_information
    expected = isinstance(model.family, ExpectedInformationFamily) if recorded is None else recorded
    return "data" if expected else "penalized"


_WHOLE_FIT_SUBJECT = DiagnosticSubject(
    identifier="fit",
    display_name="complete fitted model",
    scope="whole_fit",
)


def _part_phrase(component_name: str, *, sentence_start: bool = False) -> str:
    """A penalty component named the way the user wrote its term, and the part of it meant.

    The exact component identifier stays in the structured fields.
    """
    term, marker, suffix = component_name.rpartition("#")
    if not marker:
        phrase = f"the penalty {component_name!r}"
    elif suffix == "wiggle":
        phrase = f"the curved part of {term!r}"
    else:
        phrase = f"the {suffix!r} part of {term!r}"
    return phrase[0].upper() + phrase[1:] if sentence_start else phrase


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
        display_name=_part_phrase(component.name),
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
        "The smoothing search ended at an optimum: the REML score no longer improves in any "
        f"direction. The largest remaining slope is {0.0 if norm is None else norm:.3g}, against "
        f"an allowed {smoothing.stationarity_bar:.3g}."
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


_LOOP_NAMES = {"smoothing": "smoothing search", "coefficient": "coefficient fit"}
_LOOP_UNITS = {"smoothing": "smoothing round", "coefficient": "iteration"}
_LOOP_BUDGETS = {"smoothing": "max_reml_iter", "coefficient": "max_inner_iter"}


def _plural(count: int, unit: str) -> str:
    return f"{count} {unit}{'' if count == 1 else 's'}"


def _not_converged_finding(
    *,
    loop: str,
    reason: str,
    iterations: int,
    provenance: _EvidenceProvenance,
) -> DiagnosticFinding:
    name = _LOOP_NAMES.get(loop, f"{loop} loop")
    budget = _LOOP_BUDGETS.get(loop) if reason == "max_iterations" else None
    stopped = f"It stopped after {_plural(iterations, _LOOP_UNITS.get(loop, 'iteration'))}"
    return _finding(
        "fit.not_converged",
        priority=1,
        identifier=f"fit.not_converged:{loop}",
        severity="error",
        impacts=("fit_validity",),
        headline=f"The {name} did not converge.",
        observed=(
            f"{stopped} because it reached its limit ({budget}; reason {reason!r})."
            if budget
            else f"{stopped} with reason {reason!r}."
        ),
        interpretation=(
            "The solver could not find any step that improved the fit, so it stopped at the last "
            "coefficients it reached. The fitted numbers are not at a maximum."
            if reason == "no_ascent_direction"
            else "Two checks disagreed about which combinations of coefficients the data can pin "
            "down, so the stop could not be confirmed. The fitted values are from the last step "
            "the solver took."
            if reason == "rank_reduced_at_stop"
            else f"The fitted numbers are from the last step the {name} took, not from a "
            "finished fit, so do not rely on the standard errors or the effective degrees of "
            "freedom."
        ),
        caveats=("This does not say why the fit failed.",),
        evidence=_evidence_set(
            provenance,
            ("loop", loop),
            ("terminal_reason", reason),
            ("iterations", iterations),
        ),
        action_kind="inspect_fit_configuration",
        action_question=(
            "Try larger starting smoothing values (initial_lambda), simpler terms or, where the "
            "family supplies expected information, coefficient_curvature='fisher'."
            if reason == "no_ascent_direction"
            else "Check whether two or more coefficients are aliased (one is a combination of "
            "others). Dropping or merging those terms, or adding a penalty on them, may let the "
            "stop be confirmed."
            if reason == "rank_reduced_at_stop"
            else "Try "
            + (f"a larger {budget}, " if budget else "")
            + "other starting smoothing values (initial_lambda), or simpler terms."
        ),
        action_metrics=("convergence reason", "iterations", "fit objective"),
    )


def _null_model_not_converged_finding(reason: str) -> DiagnosticFinding:
    return _finding(
        "fit.null_model_not_converged",
        priority=2,
        severity="warning",
        impacts=("null_comparison",),
        headline=(
            "The comparison model with no terms (the intercept-only null model) did not converge."
        ),
        observed=f"Its coefficient fit stopped with reason {reason!r}.",
        interpretation=(
            "Your fitted model is unaffected. The null objective and every comparison with the "
            "null model use the last step the null fit reached."
        ),
        evidence=_evidence_set(
            "fit_result",
            ("loop", "null_model"),
            ("terminal_reason", reason),
        ),
        action_kind="inspect_fit_configuration",
        action_question="Try a larger max_inner_iter.",
        action_metrics=("null convergence reason", "null objective"),
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
        headline="The fit's summary says it did not converge, but it did not record why.",
        observed="Its overall 'converged' flag is false.",
        interpretation=(
            "The fitted numbers may not be at an optimum, so do not rely on the standard errors "
            "or the effective degrees of freedom."
        ),
        caveats=("The summary does not record why the fit stopped.",),
        evidence=_evidence_set(
            "fit_result",
            ("source", "compact_result"),
            ("compact_converged", fitted_result.converged),
            ("coefficient_converged", fitted_result.coefficient_converged),
            ("smoothing_converged", fitted_result.smoothing_converged),
        ),
        action_kind="inspect_fit_configuration",
        action_question="Try a larger max_reml_iter or max_inner_iter, or simpler terms.",
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


def _accepted_source_label(source: str) -> str:
    """Name a curvature source in prose."""
    return {
        "nearest_semidefinite": "adjusted observed (nearest positive semidefinite)",
        "fisher": "expected (Fisher)",
    }.get(source, source)


def _curvature_phrase(source: str, matrix_kind: str) -> str:
    """A curvature matrix in prose: where it comes from and what it is the curvature of."""
    target = "likelihood" if matrix_kind == "data" else "penalised likelihood"
    return f"{_accepted_source_label(source)} curvature of the {target}"


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
        # Fisher was accepted but the retained minimum is from the rejected
        # spectrum, whose scale is not kept: trust the solver's materiality decision.
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
        "all the coefficients"
        if coordinate_space == "full"
        else "the coefficients left after smoothed-away directions were removed"
    )
    source_label = _curvature_phrase(telemetry.requested_source, matrix_kind)
    if resolution is None:
        accepted = _accepted_source_label(telemetry.actual_source)
        headline = (
            f"At the final coefficients the {source_label} curved the wrong way in at least "
            "one direction."
        )
        observed = (
            f"Its lowest curvature value was {minimum:.3e}, over {coordinate_phrase}. The solver "
            f"judged that large enough to matter and switched to the {accepted} curvature."
        )
        minimum_evidence = DiagnosticEvidence(
            metric="minimum_eigenvalue",
            value=minimum,
            unit=None,
            window="requested terminal coefficient curvature",
            provenance=_CURVATURE_TELEMETRY,
        )
        resolution_caveats = (
            "The lowest curvature value and the solver's own check measure this on different "
            "scales, so they can disagree.",
            "The size of the original matrix and the cutoff were not kept after the switch.",
            f"The switch rests on the solver's decision, not on the rank or condition of the "
            f"{accepted} curvature.",
        )
    else:
        headline = f"The {source_label} has a clearly negative value."
        observed = (
            f"Its lowest curvature value, over {coordinate_phrase}, is {minimum:.3e}, below the "
            f"cutoff of {-resolution:.3e} that separates a real negative from rounding error."
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
        resolution_caveats = (
            "The cutoff comes from the shared rank rule, the number of coefficients assessed, "
            "floating-point precision and the size of the curvature matrix.",
        )

    if telemetry.requested_source == "observed":
        interpretation = (
            "In at least one direction the "
            + ("likelihood" if matrix_kind == "data" else "penalised likelihood")
            + " curves downward near the final coefficients, so the point may not be a maximum."
        )
    else:
        interpretation = (
            f"The {source_label} has a negative value in at least one direction of the "
            "coefficients assessed. That alone does not show that the fit is not at a maximum."
        )
    if used_fallback:
        interpretation += (
            " The standard errors, effective degrees of freedom and smoothing updates use the "
            "substitute curvature."
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
            "Curvature alone does not show that the likelihood is increasing in this direction.",
            "It does not identify the statistical cause.",
            "It does not show that a different curvature matrix also curves the wrong way.",
            *resolution_caveats,
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
        action_question=(
            "Compare fits with each curvature choice (coefficient_curvature): do the "
            "coefficients, standard errors or held-out score differ?"
        ),
        action_metrics=("minimum_eigenvalue", "final score", "holdout log-likelihood"),
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
    if not context.null_converged:
        findings.append(_null_model_not_converged_finding(context.null_convergence_reason))
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
                    f"The search could not confirm that the smoothing of {_part_phrase(name)} is "
                    "at its optimum."
                ),
                observed=(
                    f"The slope of the REML score is {gradient.get(name, 0.0):.3g}, but it is only "
                    f"measured to within {value:.3g}, and the limit is {bar:.3g}."
                ),
                interpretation=(
                    "The slope cannot be told from zero, so this term's smoothing is not confirmed "
                    "as the best choice. That can affect its effective degrees of freedom and "
                    "standard errors."
                ),
                caveats=(
                    "The accuracy bound is for the slope measurement. It does not say where the "
                    "optimum is.",
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
                    "Try a larger finite-difference step, or an analytic third derivative where "
                    "the family offers one."
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
            caveats=("This finding does not change the fit's converged flags.",),
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


_EXTRA_CHECK_CAUSES = (
    "the solver fell back to another curvature, a term's edge-of-range check was refused, a "
    "smoothed-away term could not be checked analytically, or a slope could not be measured "
    "accurately"
)
_SETTLE_OBSERVED = (
    "At the end, the largest change to a smoothing parameter was {residual} on the log scale "
    "({change}), above the set limit of {tolerance}."
)
_LOOSE_SMOOTHING = (
    "so the chosen smoothing parameters, and the effective degrees of freedom and standard "
    "errors that follow from them, are not settled."
)


def _relative_change(log_residual: float) -> str:
    """A log-scale change of a smoothing parameter as a share of its value."""
    if log_residual >= 4.6:
        return "more than 100 times its value"
    return f"about {100.0 * np.expm1(log_residual):.2g}% of its value"


def _stop_wording(kind: StopKind) -> dict[str, str]:
    """The quick-rule, optimum or strict wording of every stop-related finding.

    One table, so no finding phrases a stop two ways. ``lambda_cap_unresolved``
    has no quick-rule variant (a plateau is an interior stop); a stop at an
    optimum is judged by the exact slope of the REML score. The
    ``trajectory.observed`` entry reads ``residual``, ``change`` and ``tolerance``.
    """
    if kind == "practical":
        return {
            "termination.severity": "info",
            "termination.headline": (
                "The search stopped under the quick stopping rule, not the strict one."
            ),
            "termination.observed": (
                "The REML score and the smoothing parameters stopped changing by more than the "
                "quick rule's threshold. The strict check was not passed."
            ),
            "termination.interpretation": (
                "This is the fast rule you asked for, not a failure, and the fitted numbers are "
                "usable. Refit with practical_reml=False if you need the stronger check."
            ),
            "termination.action_kind": "compare_strict_reml",
            "termination.action_question": (
                "Do the fitted numbers or their uncertainty change materially under strict REML "
                "(practical_reml=False)?"
            ),
            "trajectory.headline": (
                "The smoothing parameters were still changing when the quick rule stopped the "
                "search."
            ),
            "trajectory.observed": _SETTLE_OBSERVED,
            "trajectory.interpretation_info": (
                "This is expected under the quick rule and does not mean the fit failed. The "
                "fitted numbers are usable."
            ),
            "trajectory.interpretation": (
                "The quick rule stopped the search while the smoothing was still moving, "
                + _LOOSE_SMOOTHING
            ),
            "trajectory.caveat": (
                "This is one overall figure. It does not say which term caused it."
            ),
            "trajectory.action_question": (
                "Try practical_reml=False, a larger max_reml_iter or other starting smoothing "
                "values (initial_lambda) to bring the change within the limit."
            ),
        }
    if kind == "stationary":
        return {
            "termination.severity": "warning",
            "termination.headline": (
                "The search reached an optimum, but one extra check was not passed."
            ),
            "termination.observed": (
                "The smoothing search ended at an optimum: the REML score no longer improves in "
                "any direction. The extra check that confirms this at every term did not pass."
            ),
            "termination.interpretation": (
                "The optimum was found, and the fitted values are usable. One of these stopped "
                f"the stronger check from passing: {_EXTRA_CHECK_CAUSES}."
            ),
            "termination.action_kind": "compare_certified_configuration",
            "termination.action_question": (
                "Do the fitted numbers or their standard errors change materially in a fit that "
                "keeps the curvature you asked for?"
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
            "trajectory.action_question": (
                "Which controlled change brings the outer residual within tolerance?"
            ),
        }
    return {
        "termination.severity": "warning",
        "termination.headline": "The search converged, but one extra check was not passed.",
        "termination.observed": (
            "Both the coefficient fit and the smoothing search report convergence, but the extra "
            "check that the stop was at a true optimum failed."
        ),
        "termination.interpretation": (
            "The fitted values are usable, but the stronger guarantee is not available. "
            f"Possible causes: {_EXTRA_CHECK_CAUSES}."
        ),
        "termination.action_kind": "compare_certified_configuration",
        "termination.action_question": (
            "Do the fitted numbers or their standard errors change materially in a fit that "
            "keeps the curvature you asked for?"
        ),
        "trajectory.headline": (
            "The smoothing parameters were still changing when the search stopped."
        ),
        "trajectory.observed": _SETTLE_OBSERVED,
        "trajectory.interpretation_info": (
            "The smoothing was not fully settled, but the change is small."
        ),
        "trajectory.interpretation": (
            "The smoothing was still moving when the search stopped, " + _LOOSE_SMOOTHING
        ),
        "trajectory.caveat": "This is one overall figure. It does not say which term caused it.",
        "trajectory.action_question": (
            "Try a larger max_reml_iter or other starting smoothing values (initial_lambda) to "
            "bring the change within the limit."
        ),
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
                headline=(
                    f"The smoothing of {_part_phrase(name)} ran into the largest value allowed, "
                    "and the search could not settle whether to go higher."
                ),
                observed=(
                    f"Its smoothing parameter is {smoothing.lambdas[name]:.3g}, the maximum "
                    f"(max_lambda) is {smoothing.config.maximum_lambda:.3g}, and the fit lists "
                    "this term as unresolved at the maximum."
                ),
                interpretation=(
                    "Hitting the limit gave neither a converged answer nor a reason to remove the "
                    "term. Its fitted shape, effective degrees of freedom and standard errors "
                    "rest on a smoothing value that was not settled."
                ),
                caveats=(
                    "Reaching the limit is not convergence, and it does not show that the best "
                    "smoothing is infinite.",
                ),
                evidence=tuple(evidence),
                action_kind="investigate_lambda_boundary",
                action_question=(
                    "Try a larger max_lambda, or a simpler version of this term, and check "
                    "whether the fitted numbers change."
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
                headline=(
                    f"{_part_phrase(name, sentence_start=True)} ended at the largest smoothing "
                    "the search allows."
                ),
                observed=(
                    f"Its final smoothing parameter, {smoothing.lambdas[name]:.3g}, equals the "
                    "maximum (max_lambda)."
                ),
                interpretation="The search ran into the edge of the range it may try.",
                caveats=(
                    "Reaching the maximum alone does not show that the term was smoothed away "
                    "completely or that the best smoothing is infinite.",
                ),
                evidence=_terminal_lambda_evidence(smoothing, name),
                action_kind="inspect_smoothing_boundary",
                action_question=(
                    "Try a larger max_lambda and check whether the fitted numbers change."
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
        observed = wording["trajectory.observed"].format(
            residual=f"{residual:.2g}",
            change=_relative_change(residual),
            tolerance=f"{smoothing.config.tolerance:.2g}",
        )
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
                observed=observed,
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
                action_question=wording["trajectory.action_question"],
                action_metrics=(_STATIONARITY_METRIC, "smoothing rounds", _ALL_FITTED_PARAMETERS),
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
        if null_dimension is None:
            remaining = "What is left of it could not be worked out."
        elif null_dimension == 0:
            remaining = "Nothing of it is left: it was smoothed away completely."
        else:
            remaining = (
                "Only the part the penalty cannot touch is left, for example a straight line."
            )
        if null_dimension is not None:
            evidence.extend(
                _evidence_set(
                    "fit_result",
                    ("retained_null_space_dimension", null_dimension),
                )
            )
            if null_dimension > 0:
                caveats = (
                    "A part the penalty cannot touch remains. This does not show that the feature "
                    "is useless or should be removed.",
                )
        else:
            caveats = (
                "The penalty's rank was not recorded, so no claim is made about what remains.",
            )
        findings.append(
            _finding(
                "smoothing.penalized_subspace_suppressed",
                _penalty_subject(context.layout, component),
                priority=4,
                severity="info",
                confidence=("certified" if null_dimension is not None else "unresolved"),
                impacts=("model_complexity", "smoothing_selection"),
                headline=f"{_part_phrase(name, sentence_start=True)} was smoothed away.",
                observed=remaining,
                interpretation=(
                    "The fitted model has no flexibility left in the directions this penalty "
                    "controls."
                ),
                caveats=caveats,
                evidence=tuple(evidence),
                action_kind="compare_simplified_representation",
                action_question=(
                    "Try a simpler version of this term that matches what is left, and check "
                    "whether the fitted numbers and held-out performance change."
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

    Shares lie in [0, 1], phases are disjoint exclusive times whose sum is the
    fit time, and an exact-face outcome names the iteration that activated it.
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
        exclusive = phase_snapshot.exclusive_seconds
        reported = {source_name for source_name, _ in _PROFILE_LEAF_PHASES}
        parts = [
            (profile_name, float(exclusive[source_name]), phase_snapshot.counts[source_name])
            for source_name, profile_name in _PROFILE_LEAF_PHASES
            if phase_snapshot.counts[source_name] > 0
        ]
        parts.append(
            (
                "orchestration_and_unmeasured",
                sum(float(seconds) for name, seconds in exclusive.items() if name not in reported),
                0,
            )
        )
        # Sum the disjoint intervals in the profile check's own order, so the two
        # totals are identical without a rounding allowance.
        fit_seconds = sum(seconds for _, seconds, _ in parts)
        phases = tuple(
            FitPhaseProfile(
                name=name,
                seconds=seconds,
                fit_share=seconds / fit_seconds if fit_seconds > 0.0 else 0.0,
                calls=calls,
            )
            for name, seconds, calls in parts
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
                    severity="info",
                    confidence="strong",
                    headline=(
                        f"The smoothing of {_part_phrase(name)} moved the most in most rounds."
                    ),
                    observed=(
                        f"In {wins[name]:g} of the {denominator} smoothing rounds that changed "
                        "anything, it moved the most"
                        + (
                            " (a round where several terms tied counts as shared)."
                            if tie_updates[name]
                            else "."
                        )
                    ),
                    interpretation=(
                        "Most of the search effort went into this one term. This describes how "
                        "the search went. It does not mean the numbers are wrong."
                    ),
                    caveats=("Which term moved the most is not a measure of fit time or a cause.",),
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
                        "Try a simpler or smaller basis for this term, and check whether it "
                        "settles faster without changing the fitted numbers."
                    ),
                    action_metrics=("smoothing rounds", _ALL_FITTED_PARAMETERS, "predictions"),
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
                severity="info",
                confidence="strong",
                headline=(
                    f"The smoothing of {_part_phrase(name)} was raised every round and never "
                    "lowered."
                ),
                observed=f"It went up in all {positive} of the rounds in which it changed.",
                interpretation="The search kept pushing this term toward a flatter curve.",
                caveats=(
                    "Rising every round does not show that the term was smoothed away or that "
                    "the best smoothing is infinite.",
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
                action_question=(
                    "Try a simpler basis for this term, or a larger starting smoothing value "
                    "(initial_lambda), and check whether the search settles."
                ),
                action_metrics=(
                    "smoothing moves",
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
            severity="info",
            confidence="strong",
            headline="Several smoothing steps had to be shortened or rejected before one was accepted.",
            observed=(
                f"{_plural(backtracked, 'smoothing step')} "
                f"{'was' if backtracked == 1 else 'were'} shortened and retried, and {rejected} "
                f"{'was' if rejected == 1 else 'were'} rejected. That added extra coefficient "
                f"fits over {_plural(smoothing.iterations, 'smoothing round')}."
            ),
            interpretation=("This cost fit time. It does not by itself mean the result is wrong."),
            caveats=("No timing was recorded for those extra coefficient fits.",),
            evidence=evidence,
            action_kind="compare_smoothing_initialization",
            action_question=(
                "Try other starting smoothing values (initial_lambda) and check whether fewer "
                "steps are rejected."
            ),
            action_metrics=("coefficient fits", "smoothing rounds", _ALL_FITTED_PARAMETERS),
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
    return (
        _finding(
            "optimization.curvature_fallback_repeated",
            priority=2,
            confidence="strong",
            impacts=("fit_reliability", "inference_reliability"),
            headline="The coefficient solver had to use a backup way of measuring curvature.",
            observed=(
                f"The backup was used {_plural(fallback_count, 'time')}, in "
                f"{len(fallback_telemetry)} of the coefficient fits."
            ),
            interpretation=(
                "The curvature you asked for could not be used in those fits, so the results "
                "that depend on them rest on the backup curvature."
            ),
            caveats=("The counts describe the numerical work, not the time it took.",),
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
                "Check whether a fit that keeps the curvature you asked for changes the "
                "parameters, standard errors or convergence."
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
                headline="The data cannot pin down every coefficient.",
                observed=f"Only {rank} of {width} coefficient combinations are determined.",
                interpretation=(
                    "At least one combination is not determined by the data, so the standard "
                    "errors cover only the combinations that are."
                ),
                caveats=("A direction removed on purpose (a term smoothed away) is not counted.",),
                evidence=_evidence_set(
                    "fit_result",
                    ("estimable_rank", rank),
                    ("active_coordinate_dimension", width),
                    ("coordinate_space", coordinate_space),
                ),
                action_kind="investigate_rank_loss",
                action_question=(
                    "Check which terms overlap. Dropping or merging them, or using a smaller "
                    "basis (k), may restore the missing combinations."
                ),
                action_metrics=("estimable rank", "predictions", _ALL_FITTED_PARAMETERS),
            )
        )

    condition = fit.terminal_curvature.condition_estimate
    if condition is not None and condition >= _CONDITION_WARNING_THRESHOLD:
        accepted_source = fit.terminal_curvature.actual_source
        matrix_kind = context.terminal_policy_matrix_kind
        findings.append(
            _finding(
                "numerics.conditioning_warning",
                priority=2,
                confidence="strong",
                headline=(
                    f"The {_curvature_phrase(accepted_source, matrix_kind)} is close to singular "
                    f"(condition estimate {condition:.1e})."
                ),
                observed=(
                    f"The condition estimate, {condition:.1e}, is at or above the warning level of "
                    f"{_CONDITION_WARNING_THRESHOLD:.1e}."
                ),
                interpretation=(
                    "Some coefficients may be poorly determined. This is a warning, not proof "
                    "that they are."
                ),
                caveats=(
                    "The estimate is made after rescaling the matrix and before its smallest "
                    "directions are dropped.",
                    "How it is estimated depends on how the matrix was decomposed, and it is not "
                    "the condition number of the unscaled matrix.",
                    "No estimate was kept for the directions that remain after dropping, so this "
                    "does not show that they amplify error.",
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
                action_question=(
                    "Centre numeric columns that sit far from zero, or drop columns that nearly "
                    "duplicate one another, and check whether the fitted numbers stay the same."
                ),
                action_metrics=("condition estimate", "estimable rank", _ALL_FITTED_PARAMETERS),
            )
        )
    return tuple(findings)


def _analyze_shape_rules(context: _LSSDiagnosticContext) -> tuple[DiagnosticFinding, ...]:
    """One plain sentence per term that carried a ``Constraint.fit`` rule: what the rule did."""
    shape = context.shape_fit
    if shape is None:
        return ()
    findings = []
    for term in shape.terms:
        predictor, _, name = term.term.partition(":")
        subject = DiagnosticSubject(
            identifier=f"term:{term.term}",
            display_name=f"the term {term.term!r}",
            scope="term",
            predictor=predictor,
            term=name,
        )
        sentence = shape_term_sentence(term, shape)
        smoothing_sentence = shape_smoothing_sentence(shape)
        settled = shape.converged and (shape.smoothing is None or shape.smoothing.converged)
        findings.append(
            _finding(
                "shape.rule",
                subject,
                priority=3,
                severity="info" if settled else "warning",
                headline=sentence
                if smoothing_sentence is None
                else f"{sentence} {smoothing_sentence}",
                observed=(
                    f"{term.n_binding} of {term.n_rows} rows bind; the curve moved by at most "
                    f"{term.largest_change:.3g} on the linear-predictor scale."
                ),
                interpretation=(
                    "The coefficients are the fit without the rule, so the curve may break it; "
                    "refit with a larger max_inner_iter, or relax the rule."
                    if shape.unruled
                    else "The smoothing was chosen with the rule held at the rows that bind, so a "
                    "binding rule costs likelihood, fixes the curve at those rows and moves the "
                    "smoothing to suit the curve it allows."
                    if shape.smoothing is not None
                    else "The smoothing the search chose without the rule was kept and the "
                    "coefficients were refitted under it, so a binding rule costs likelihood and "
                    "fixes the curve at the rows that bind."
                    if shape.applied
                    else "The fit without the rule already obeyed it, so it is the fit."
                ),
                evidence=_evidence_set(
                    "fit_result",
                    ("rule_rows", term.n_rows),
                    ("rule_rows_binding", term.n_binding),
                    ("largest_change_in_effect", term.largest_change),
                    ("refit_converged", shape.converged),
                    (
                        "smoothing_converged",
                        True if shape.smoothing is None else shape.smoothing.converged,
                    ),
                ),
                action_kind="compare_unconstrained_fit",
                action_question=(
                    "Refit without the rule and compare the curve, the standard errors and "
                    "held-out accuracy."
                ),
                action_metrics=("fitted curve", "standard errors", "held-out score"),
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
    _analyze_shape_rules,
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
        "Smoothing was fixed, so there was no search to check."
        if context.smoothing is None
        else "Checked: how the smoothing search went, and which terms were smoothed away."
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
            "Checked: whether the fit converged, and which revision it is.",
            "Checked: whether the final coefficient fit converged, how many coefficient "
            "combinations the data pins down, and the curvature.",
            smoothing_coverage,
            *stationary_coverage,
        ),
        limitations=(
            "This check did not look at your rows, responses, weights, exposures or levels.",
            (
                "No timing was recorded for this fit, so its time cannot be split by phase."
                if phase_snapshot is None
                else "Timings are for this fit on this machine. They are not benchmarks and are "
                "not split by term."
            ),
            "It did not refit your model, so it cannot say whether a term is useful or what "
            "causes an effect.",
        ),
        profile=_build_profile(context, phase_snapshot),
    )
