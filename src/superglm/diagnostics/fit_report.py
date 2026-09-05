"""Immutable, model-agnostic values for fit diagnostic reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | tuple[JsonValue, ...] | Mapping[str, JsonValue]
type _MutableJsonValue = JsonScalar | list[_MutableJsonValue] | dict[str, _MutableJsonValue]

_SUBJECT_SCOPES = frozenset({"whole_fit", "predictor", "term", "penalized_subspace", "level"})
_EVIDENCE_PROVENANCE = frozenset(
    {
        "fit_result",
        "solver_history",
        "smoothing_history",
        "curvature_telemetry",
        "inference",
        "counterfactual_fit",
    }
)
_SEVERITIES = frozenset({"error", "warning", "info"})
_CONFIDENCES = frozenset({"certified", "strong", "suggestive", "unresolved"})
_SUBJECT_SCOPE_ORDER = {
    "whole_fit": 0,
    "predictor": 1,
    "term": 2,
    "penalized_subspace": 3,
    "level": 4,
}
_CONFIDENCE_ORDER = {"certified": 0, "strong": 1, "suggestive": 2, "unresolved": 3}
_RANK_SHARE_METRICS = frozenset({"persistence_share", "work_share", "dominant_update_share"})
_REPORT_SCOPES = frozenset({"fit", "deep"})
_FIT_STATUSES = frozenset(
    {
        "converged_certified",
        "converged_exact_face",
        "converged_uncertified",
        "not_converged",
        "fixed_fit",
    }
)
_DETAIL_LEVELS = frozenset({"compact", "full"})
_SMOOTHING_OUTCOMES = frozenset({"fixed", "finite", "upper_bound", "unresolved_cap", "exact_face"})
_EXACT_FACE_EFFECTS = frozenset(
    {"fully_suppressed", "linear_only", "null_space_only", "unresolved"}
)
_MAX_JSON_CONTAINER_ITEMS = 64
_MAX_JSON_CONTAINER_LEVELS = 8
_MAX_JSON_NODES = 256


@dataclass(slots=True)
class _JsonBudget:
    nodes: int = 0


def _required_string(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _optional_string(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _required_string(name, value)


def _literal_string(name: str, value: object, allowed: frozenset[str]) -> str:
    value = _required_string(name, value)
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {options}")
    return value


def _string_tuple(name: str, values: object) -> tuple[str, ...]:
    if not isinstance(values, list | tuple):
        raise TypeError(f"{name} must be a list or tuple")
    return tuple(_required_string(f"{name}[{index}]", value) for index, value in enumerate(values))


def _instance_tuple[T](name: str, values: object, expected_type: type[T]) -> tuple[T, ...]:
    if not isinstance(values, list | tuple):
        raise TypeError(f"{name} must be a list or tuple")
    owned = tuple(values)
    for index, value in enumerate(owned):
        if not isinstance(value, expected_type):
            raise TypeError(f"{name}[{index}] must be {expected_type.__name__}")
    return owned


def _own_json(
    value: object,
    *,
    name: str,
    active: set[int] | None = None,
    budget: _JsonBudget | None = None,
    container_level: int = 0,
) -> JsonValue:
    if budget is None:
        budget = _JsonBudget()
    budget.nodes += 1
    if budget.nodes > _MAX_JSON_NODES:
        raise ValueError(f"{name} must contain at most {_MAX_JSON_NODES} nodes")

    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return cast(JsonScalar, value)
    if value_type is float:
        scalar = cast(float, value)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must contain only finite floats")
        return scalar

    if active is None:
        active = set()
    if value_type in {list, tuple}:
        sequence = cast("list[object] | tuple[object, ...]", value)
        _validate_container_budget(name, sequence, container_level)
        container_id = id(value)
        if container_id in active:
            raise ValueError(f"{name} must not contain recursive containers")
        active.add(container_id)
        try:
            return tuple(
                _own_json(
                    item,
                    name=f"{name}[{index}]",
                    active=active,
                    budget=budget,
                    container_level=container_level + 1,
                )
                for index, item in enumerate(sequence)
            )
        finally:
            active.remove(container_id)

    if isinstance(value, Mapping):
        _validate_container_budget(name, value, container_level)
        container_id = id(value)
        if container_id in active:
            raise ValueError(f"{name} must not contain recursive containers")
        active.add(container_id)
        try:
            owned: dict[str, JsonValue] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"{name} mappings must have string keys")
                owned[key] = _own_json(
                    item,
                    name=f"{name}[{key!r}]",
                    active=active,
                    budget=budget,
                    container_level=container_level + 1,
                )
            return MappingProxyType(owned)
        finally:
            active.remove(container_id)

    raise TypeError(
        f"{name} must contain only None, bool, int, finite float, str, list, tuple, or mapping"
    )


def _validate_container_budget(name: str, value: Sized, container_level: int) -> None:
    if container_level >= _MAX_JSON_CONTAINER_LEVELS:
        raise ValueError(
            f"{name} must contain at most {_MAX_JSON_CONTAINER_LEVELS} container levels"
        )
    if len(value) > _MAX_JSON_CONTAINER_ITEMS:
        raise ValueError(
            f"{name} containers must contain at most {_MAX_JSON_CONTAINER_ITEMS} items"
        )


def _mutable_json(value: JsonValue) -> _MutableJsonValue:
    if isinstance(value, tuple):
        return [_mutable_json(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _mutable_json(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class DiagnosticSubject:
    """The fitted scope to which one diagnostic finding applies."""

    identifier: str
    display_name: str
    scope: Literal["whole_fit", "predictor", "term", "penalized_subspace", "level"]
    predictor: str | None = None
    term: str | None = None
    component: str | None = None

    def __post_init__(self) -> None:
        _required_string("identifier", self.identifier)
        _required_string("display_name", self.display_name)
        _literal_string("scope", self.scope, _SUBJECT_SCOPES)
        _optional_string("predictor", self.predictor)
        _optional_string("term", self.term)
        _optional_string("component", self.component)


@dataclass(frozen=True, slots=True)
class DiagnosticEvidence:
    """One bounded, JSON-safe piece of retained diagnostic evidence."""

    metric: str
    value: JsonValue
    unit: str | None
    window: str
    provenance: Literal[
        "fit_result",
        "solver_history",
        "smoothing_history",
        "curvature_telemetry",
        "inference",
        "counterfactual_fit",
    ]
    comparator: str | None = None
    threshold: JsonValue | None = None

    def __post_init__(self) -> None:
        _required_string("metric", self.metric)
        object.__setattr__(self, "value", _own_json(self.value, name="value"))
        _optional_string("unit", self.unit)
        _required_string("window", self.window)
        _literal_string("provenance", self.provenance, _EVIDENCE_PROVENANCE)
        _optional_string("comparator", self.comparator)
        object.__setattr__(self, "threshold", _own_json(self.threshold, name="threshold"))


@dataclass(frozen=True, slots=True)
class DiagnosticAction:
    """A concrete follow-up question or comparison suggested by a finding."""

    kind: str
    question: str
    requires_data: bool
    comparison_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_string("kind", self.kind)
        _required_string("question", self.question)
        if type(self.requires_data) is not bool:
            raise TypeError("requires_data must be a bool")
        object.__setattr__(
            self,
            "comparison_metrics",
            _string_tuple("comparison_metrics", self.comparison_metrics),
        )


@dataclass(frozen=True, slots=True)
class DiagnosticFinding:
    """A structured observation, interpretation, caveat, and next action."""

    identifier: str
    code: str
    category: str
    severity: Literal["error", "warning", "info"]
    confidence: Literal["certified", "strong", "suggestive", "unresolved"]
    impacts: tuple[str, ...]
    subject: DiagnosticSubject
    headline: str
    observed: str
    interpretation: str
    caveats: tuple[str, ...]
    evidence: tuple[DiagnosticEvidence, ...]
    actions: tuple[DiagnosticAction, ...]
    priority_tier: int

    def __post_init__(self) -> None:
        _required_string("identifier", self.identifier)
        _required_string("code", self.code)
        _required_string("category", self.category)
        _literal_string("severity", self.severity, _SEVERITIES)
        _literal_string("confidence", self.confidence, _CONFIDENCES)
        object.__setattr__(self, "impacts", _string_tuple("impacts", self.impacts))
        if not isinstance(self.subject, DiagnosticSubject):
            raise TypeError("subject must be DiagnosticSubject")
        _required_string("headline", self.headline)
        _required_string("observed", self.observed)
        _required_string("interpretation", self.interpretation)
        object.__setattr__(self, "caveats", _string_tuple("caveats", self.caveats))
        object.__setattr__(
            self,
            "evidence",
            _instance_tuple("evidence", self.evidence, DiagnosticEvidence),
        )
        object.__setattr__(
            self,
            "actions",
            _instance_tuple("actions", self.actions, DiagnosticAction),
        )
        if not self.actions:
            raise ValueError("actions must contain at least one DiagnosticAction")
        if type(self.priority_tier) is not int:
            raise TypeError("priority_tier must be an integer")
        if not 1 <= self.priority_tier <= 6:
            raise ValueError("priority_tier must be between 1 and 6")


def _ranking_share(finding: DiagnosticFinding) -> int | float:
    values = (
        evidence.value
        for evidence in finding.evidence
        if evidence.metric in _RANK_SHARE_METRICS
        and not isinstance(evidence.value, bool)
        and isinstance(evidence.value, int | float)
    )
    return max(values, default=0.0)


def _finding_rank_key(
    finding: DiagnosticFinding,
) -> tuple[int, int, int, int | float, str, str, str]:
    return (
        finding.priority_tier,
        _SUBJECT_SCOPE_ORDER[finding.subject.scope],
        _CONFIDENCE_ORDER[finding.confidence],
        -_ranking_share(finding),
        finding.code,
        finding.subject.identifier,
        finding.identifier,
    )


def _rank_findings(
    findings: tuple[DiagnosticFinding, ...] | list[DiagnosticFinding],
) -> tuple[DiagnosticFinding, ...]:
    """Return findings in their version-one canonical diagnostic order."""
    return tuple(sorted(findings, key=_finding_rank_key))


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonnegative_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    result = float(value)  # ty: ignore[invalid-argument-type] -- validated conversion boundary
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)  # ty: ignore[invalid-argument-type] -- validated conversion boundary
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class FitPhaseProfile:
    """Cumulative timing for one measured fit phase."""

    name: str
    seconds: float
    fit_share: float
    calls: int

    def __post_init__(self) -> None:
        _required_string("name", self.name)
        object.__setattr__(self, "seconds", _nonnegative_float("seconds", self.seconds))
        share = _nonnegative_float("fit_share", self.fit_share)
        if share > 1.0:
            raise ValueError("fit_share must not exceed one")
        object.__setattr__(self, "fit_share", share)
        _nonnegative_int("calls", self.calls)


@dataclass(frozen=True, slots=True)
class SmoothingComponentProfile:
    """Terminal state and retained EFS movement for one smoothing component."""

    name: str
    predictor: str
    term: str
    initial_lambda: float
    final_lambda: float
    accepted_moves: int
    dominant_update_share: float
    terminal_term_edf: float | None
    null_space_dimension: int | None
    outcome: Literal["fixed", "finite", "upper_bound", "unresolved_cap", "exact_face"]
    exact_face_iteration: int | None
    exact_face_effect: (
        Literal["fully_suppressed", "linear_only", "null_space_only", "unresolved"] | None
    )
    upper_bound_iterations: int

    def __post_init__(self) -> None:
        _required_string("name", self.name)
        _required_string("predictor", self.predictor)
        _required_string("term", self.term)
        object.__setattr__(
            self,
            "initial_lambda",
            _nonnegative_float("initial_lambda", self.initial_lambda),
        )
        object.__setattr__(
            self,
            "final_lambda",
            _nonnegative_float("final_lambda", self.final_lambda),
        )
        _nonnegative_int("accepted_moves", self.accepted_moves)
        share = _nonnegative_float("dominant_update_share", self.dominant_update_share)
        if share > 1.0:
            raise ValueError("dominant_update_share must not exceed one")
        object.__setattr__(self, "dominant_update_share", share)
        if self.terminal_term_edf is not None:
            object.__setattr__(
                self,
                "terminal_term_edf",
                _finite_float("terminal_term_edf", self.terminal_term_edf),
            )
        if self.null_space_dimension is not None:
            _nonnegative_int("null_space_dimension", self.null_space_dimension)
        _literal_string("outcome", self.outcome, _SMOOTHING_OUTCOMES)
        if self.exact_face_iteration is not None:
            if (
                isinstance(self.exact_face_iteration, bool)
                or not isinstance(self.exact_face_iteration, int)
                or self.exact_face_iteration < 1
            ):
                raise ValueError("exact_face_iteration must be a positive integer or None")
        if (self.outcome == "exact_face") != (self.exact_face_iteration is not None):
            raise ValueError("only an exact-face outcome identifies an exact-face iteration")
        if self.exact_face_effect is not None:
            _literal_string("exact_face_effect", self.exact_face_effect, _EXACT_FACE_EFFECTS)
        if (self.outcome == "exact_face") != (self.exact_face_effect is not None):
            raise ValueError("only an exact-face outcome identifies its retained effect")
        if self.exact_face_effect == "fully_suppressed" and self.null_space_dimension != 0:
            raise ValueError("a fully suppressed face must have zero null-space dimension")
        if self.exact_face_effect in {"linear_only", "null_space_only"} and (
            self.null_space_dimension is None or self.null_space_dimension < 1
        ):
            raise ValueError("a retained exact-face effect requires a positive null space")
        if self.exact_face_effect == "unresolved" and self.null_space_dimension is not None:
            raise ValueError("an unresolved exact-face effect cannot claim a null-space dimension")
        _nonnegative_int("upper_bound_iterations", self.upper_bound_iterations)


@dataclass(frozen=True, slots=True)
class FitWorkProfile:
    """Compact operational profile retained for one accepted fit."""

    n_observations: int
    n_coefficients: int
    fit_seconds: float | None
    outer_iterations: int
    coefficient_fits: int
    inner_iterations: int
    rejected_proposals: int
    backtracked_proposals: int
    phases: tuple[FitPhaseProfile, ...]
    smoothing_components: tuple[SmoothingComponentProfile, ...]

    def __post_init__(self) -> None:
        _nonnegative_int("n_observations", self.n_observations)
        _nonnegative_int("n_coefficients", self.n_coefficients)
        if self.fit_seconds is not None:
            object.__setattr__(
                self,
                "fit_seconds",
                _nonnegative_float("fit_seconds", self.fit_seconds),
            )
        phases = _instance_tuple("phases", self.phases, FitPhaseProfile)
        components = _instance_tuple(
            "smoothing_components",
            self.smoothing_components,
            SmoothingComponentProfile,
        )
        if len({item.name for item in phases}) != len(phases):
            raise ValueError("phase names must be unique")
        if len({item.name for item in components}) != len(components):
            raise ValueError("smoothing component names must be unique")
        if (self.fit_seconds is None) != (not phases):
            raise ValueError("phase timings require a measured fit duration")
        if self.fit_seconds is not None:
            total_seconds = sum(item.seconds for item in phases)
            scale = max(self.fit_seconds, total_seconds, 1.0)
            tolerance = 64.0 * math.ulp(scale)
            if abs(total_seconds - self.fit_seconds) > tolerance:
                raise ValueError("phase seconds must sum to fit_seconds")
            if self.fit_seconds > 0.0:
                for phase in phases:
                    expected_share = phase.seconds / self.fit_seconds
                    if abs(phase.fit_share - expected_share) > 64.0 * math.ulp(1.0):
                        raise ValueError(
                            "phase fit_share must equal seconds divided by fit_seconds"
                        )
                if abs(sum(item.fit_share for item in phases) - 1.0) > 64.0 * math.ulp(1.0):
                    raise ValueError("phase fit_share values must sum to one")
            elif any(item.seconds != 0.0 or item.fit_share != 0.0 for item in phases):
                raise ValueError("zero fit_seconds requires zero phase timings")
        for name in (
            "outer_iterations",
            "coefficient_fits",
            "inner_iterations",
            "rejected_proposals",
            "backtracked_proposals",
        ):
            _nonnegative_int(name, getattr(self, name))
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "smoothing_components", components)


@dataclass(frozen=True, slots=True)
class FitDiagnosticReport:
    """An immutable structured report with compact and full text renderings."""

    schema_version: int
    rule_set_version: int
    model_type: str
    family: str
    fit_revision: int
    scope: Literal["fit", "deep"]
    fit_status: str
    findings: tuple[DiagnosticFinding, ...]
    coverage: tuple[str, ...]
    limitations: tuple[str, ...]
    profile: FitWorkProfile | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {1, 2}:
            raise ValueError("schema_version must be 1 or 2")
        _exact_version("rule_set_version", self.rule_set_version)
        _required_string("model_type", self.model_type)
        _required_string("family", self.family)
        if type(self.fit_revision) is not int:
            raise TypeError("fit_revision must be an integer")
        if self.fit_revision < 0:
            raise ValueError("fit_revision must be nonnegative")
        _literal_string("scope", self.scope, _REPORT_SCOPES)
        _literal_string("fit_status", self.fit_status, _FIT_STATUSES)
        findings = _instance_tuple("findings", self.findings, DiagnosticFinding)
        identifiers: set[str] = set()
        for finding in findings:
            if finding.identifier in identifiers:
                raise ValueError(f"duplicate finding identifier: {finding.identifier!r}")
            identifiers.add(finding.identifier)
        object.__setattr__(self, "findings", _rank_findings(findings))
        object.__setattr__(self, "coverage", _string_tuple("coverage", self.coverage))
        object.__setattr__(self, "limitations", _string_tuple("limitations", self.limitations))
        if self.profile is not None and not isinstance(self.profile, FitWorkProfile):
            raise TypeError("profile must be FitWorkProfile or None")
        if self.schema_version == 1 and self.profile is not None:
            raise ValueError("schema_version=1 cannot carry a fit profile")
        if self.schema_version == 2 and self.profile is None:
            raise ValueError("schema_version=2 requires a fit profile")

    def render(self, *, detail: Literal["compact", "full"] = "compact") -> str:
        """Render the report as stable semantic sections for human inspection."""
        detail = cast(
            'Literal["compact", "full"]',
            _literal_string("detail", detail, _DETAIL_LEVELS),
        )
        if detail == "compact":
            return self._render_compact()
        return self._render_full()

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a fresh tree of plain JSON containers owned by the caller."""
        payload: dict[str, _MutableJsonValue] = {
            "schema_version": self.schema_version,
            "rule_set_version": self.rule_set_version,
            "model_type": self.model_type,
            "family": self.family,
            "fit_revision": self.fit_revision,
            "scope": self.scope,
            "fit_status": self.fit_status,
            "findings": [_finding_dict(finding) for finding in self.findings],
            "coverage": list(self.coverage),
            "limitations": list(self.limitations),
        }
        if self.schema_version == 2:
            assert self.profile is not None
            payload["profile"] = _profile_dict(self.profile)
        return cast(dict[str, JsonValue], payload)

    def __str__(self) -> str:
        return self.render(detail="compact")

    def _render_compact(self) -> str:
        lines = self._render_profile() if self.profile is not None else [self._render_header()]
        if self.profile is not None:
            if self.findings:
                lines.extend(["", "Findings"])
                for finding in self.findings:
                    lines.append(f"- {finding.headline}")
                    if finding.actions:
                        lines.append(f"  Next: {finding.actions[0].question}")
            else:
                lines.extend(["", "No solver pathology was detected in the available evidence."])
            lines.extend(
                [
                    "",
                    "Scope: retained fit telemetry only; use detail='full' for evidence, "
                    "caveats and limitations.",
                ]
            )
            return "\n".join(lines)
        if not self.findings:
            lines.extend(
                [
                    "",
                    "No solver pathology was detected in the available evidence.",
                ]
            )
        for index, finding in enumerate(self.findings, start=1):
            lines.extend(
                [
                    "",
                    f"{index}. {finding.headline}",
                    _subject_line(finding.subject),
                    f"   Code: {finding.code}",
                ]
            )
            if finding.actions:
                lines.append(f"   Next question: {finding.actions[0].question}")
        _append_coverage(lines, self.coverage, self.limitations)
        return "\n".join(lines)

    def _render_full(self) -> str:
        lines = self._render_profile() if self.profile is not None else [self._render_header()]
        if not self.findings:
            lines.extend(
                [
                    "",
                    "No solver pathology was detected in the available evidence.",
                ]
            )
        for index, finding in enumerate(self.findings, start=1):
            lines.extend(
                [
                    "",
                    f"{index}. {finding.headline}",
                    _subject_line(finding.subject),
                    f"   Code: {finding.code}",
                    f"   Severity: {finding.severity}; confidence: {finding.confidence}",
                    "",
                    f"What we observed: {finding.observed}",
                    "",
                    f"What this may mean: {finding.interpretation}",
                ]
            )
            if finding.caveats:
                lines.extend(["", "What this does not establish:"])
                lines.extend(f"- {caveat}" for caveat in finding.caveats)
            for action_index, action in enumerate(finding.actions):
                heading = (
                    "Suggested experiment"
                    if action_index == 0
                    else "Additional suggested experiment"
                )
                lines.extend(["", f"{heading}: {action.question}"])
                if action.comparison_metrics:
                    lines.append(f"Compare: {', '.join(action.comparison_metrics)}")
                lines.append(f"Requires data: {'yes' if action.requires_data else 'no'}")
            lines.extend(["", "Technical evidence:"])
            if finding.evidence:
                lines.extend(f"- {_render_evidence(evidence)}" for evidence in finding.evidence)
            else:
                lines.append("- No additional technical evidence was retained.")
        _append_coverage(lines, self.coverage, self.limitations)
        return "\n".join(lines)

    def _render_header(self) -> str:
        return (
            f"Fit diagnostic report — model: {self.model_type}; family: {self.family}; "
            f"status: {self.fit_status}; revision: {self.fit_revision}"
        )

    def _render_profile(self) -> list[str]:
        assert self.profile is not None
        profile = self.profile
        lines = [
            f"{self.model_type} fit diagnosis — family: {self.family}; "
            f"status: {self.fit_status}; revision: {self.fit_revision}",
            "",
        ]
        if profile.fit_seconds is None:
            timing = "Timing unavailable (no fit timing snapshot was retained)."
        else:
            timing = f"Fit time: {_format_seconds(profile.fit_seconds)}"
        lines.append(
            f"Rows: {profile.n_observations:,}    "
            f"Coefficients: {profile.n_coefficients:,}    {timing}"
        )
        lines.extend(
            [
                "",
                "Work",
                f"Outer EFS iterations       {profile.outer_iterations}",
                f"Coefficient fits           {profile.coefficient_fits}",
                f"Total inner iterations     {profile.inner_iterations}",
                f"Ordinary outer proposals rejected   {profile.rejected_proposals}",
                f"Ordinary outer proposals backtracked {profile.backtracked_proposals}",
            ]
        )
        if profile.phases:
            lines.extend(
                ["", "Time distribution", "Phase                         Time    Share   Calls"]
            )
            for phase in sorted(profile.phases, key=lambda item: (-item.seconds, item.name)):
                label = phase.name.replace("_", " ")
                lines.append(
                    f"{label:<29} {_format_seconds(phase.seconds):>8} "
                    f"{phase.fit_share:>7.1%} {phase.calls:>7}"
                )
        if profile.smoothing_components:
            lines.extend(
                [
                    "",
                    "Smoothing parameters",
                    "Component                         Initial λ    Final λ  Moves  "
                    "Lead share  Cap iters  Term EDF  Result",
                ]
            )
            rendered_term_edf: set[tuple[str, str]] = set()
            shared_term_edf = False
            for component in profile.smoothing_components:
                final_lambda = (
                    "∞"
                    if component.outcome == "exact_face"
                    else _format_number(component.final_lambda)
                )
                term_key = (component.predictor, component.term)
                if component.terminal_term_edf is None:
                    edf = "—"
                elif term_key in rendered_term_edf:
                    edf = "shared"
                    shared_term_edf = True
                else:
                    edf = f"{component.terminal_term_edf:.3f}"
                    rendered_term_edf.add(term_key)
                if component.outcome == "exact_face":
                    assert component.exact_face_effect is not None
                    outcome = {
                        "fully_suppressed": "fully suppressed",
                        "linear_only": "linear only",
                        "null_space_only": "null-space only",
                        "unresolved": "exact face; retained effect unresolved",
                    }[component.exact_face_effect]
                else:
                    outcome = {
                        "fixed": "fixed by caller",
                        "finite": "finite",
                        "upper_bound": "at upper bound",
                        "unresolved_cap": "cap unresolved",
                    }[component.outcome]
                if component.exact_face_iteration is not None:
                    outcome += f" @ iter {component.exact_face_iteration}"
                lines.append(
                    f"{component.name:<33} {_format_number(component.initial_lambda):>10} "
                    f"{final_lambda:>10} {component.accepted_moves:>6} "
                    f"{component.dominant_update_share:>10.1%} "
                    f"{component.upper_bound_iterations:>9} {edf:>8}  {outcome}"
                )
            if shared_term_edf:
                lines.append(
                    "Term EDF is shared by penalty components of the same predictor term; "
                    "do not add repeated component rows."
                )
        return lines


def _exact_version(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value != 1:
        raise ValueError(f"{name} must be 1")


def _subject_line(subject: DiagnosticSubject) -> str:
    return f"   Subject: {subject.display_name} [{subject.scope}; identifier: {subject.identifier}]"


def _render_evidence(evidence: DiagnosticEvidence) -> str:
    rendered = [
        f"{evidence.metric} = {_render_json(evidence.value)}",
        f"window: {evidence.window}",
        f"provenance: {evidence.provenance}",
    ]
    if evidence.unit is not None:
        rendered.append(f"unit: {evidence.unit}")
    if evidence.comparator is not None:
        comparison = f"comparator: {evidence.comparator}"
        if evidence.threshold is not None:
            comparison += f" {_render_json(evidence.threshold)}"
        rendered.append(comparison)
    elif evidence.threshold is not None:
        rendered.append(f"threshold: {_render_json(evidence.threshold)}")
    return "; ".join(rendered)


def _render_json(value: JsonValue) -> str:
    return json.dumps(
        _mutable_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _format_seconds(value: float) -> str:
    if value < 0.001:
        return f"{value * 1_000_000.0:.0f} µs"
    if value < 1.0:
        return f"{value * 1_000.0:.1f} ms"
    return f"{value:.2f} s"


def _format_number(value: float) -> str:
    if value == 0.0:
        return "0"
    if 1.0e-3 <= abs(value) < 1.0e5:
        return f"{value:.4g}"
    return f"{value:.3e}"


def _append_coverage(
    lines: list[str],
    coverage: tuple[str, ...],
    limitations: tuple[str, ...],
) -> None:
    lines.extend(["", "Coverage:"])
    if coverage:
        lines.extend(f"- {item}" for item in coverage)
    else:
        lines.append("- No evidence classes were declared.")
    lines.extend(["", "Limitations:"])
    if limitations:
        lines.extend(f"- {item}" for item in limitations)
    else:
        lines.append("- No additional limitations were declared.")


def _subject_dict(subject: DiagnosticSubject) -> dict[str, _MutableJsonValue]:
    return {
        "identifier": subject.identifier,
        "display_name": subject.display_name,
        "scope": subject.scope,
        "predictor": subject.predictor,
        "term": subject.term,
        "component": subject.component,
    }


def _evidence_dict(evidence: DiagnosticEvidence) -> dict[str, _MutableJsonValue]:
    return {
        "metric": evidence.metric,
        "value": _mutable_json(evidence.value),
        "unit": evidence.unit,
        "window": evidence.window,
        "provenance": evidence.provenance,
        "comparator": evidence.comparator,
        "threshold": _mutable_json(evidence.threshold),
    }


def _action_dict(action: DiagnosticAction) -> dict[str, _MutableJsonValue]:
    return {
        "kind": action.kind,
        "question": action.question,
        "requires_data": action.requires_data,
        "comparison_metrics": list(action.comparison_metrics),
    }


def _finding_dict(finding: DiagnosticFinding) -> dict[str, _MutableJsonValue]:
    return {
        "identifier": finding.identifier,
        "code": finding.code,
        "category": finding.category,
        "severity": finding.severity,
        "confidence": finding.confidence,
        "impacts": list(finding.impacts),
        "subject": _subject_dict(finding.subject),
        "headline": finding.headline,
        "observed": finding.observed,
        "interpretation": finding.interpretation,
        "caveats": list(finding.caveats),
        "evidence": [_evidence_dict(evidence) for evidence in finding.evidence],
        "actions": [_action_dict(action) for action in finding.actions],
        "priority_tier": finding.priority_tier,
    }


def _profile_dict(profile: FitWorkProfile) -> dict[str, _MutableJsonValue]:
    return {
        "n_observations": profile.n_observations,
        "n_coefficients": profile.n_coefficients,
        "fit_seconds": profile.fit_seconds,
        "outer_iterations": profile.outer_iterations,
        "coefficient_fits": profile.coefficient_fits,
        "inner_iterations": profile.inner_iterations,
        "rejected_proposals": profile.rejected_proposals,
        "backtracked_proposals": profile.backtracked_proposals,
        "phases": [
            {
                "name": phase.name,
                "seconds": phase.seconds,
                "fit_share": phase.fit_share,
                "calls": phase.calls,
            }
            for phase in profile.phases
        ],
        "smoothing_components": [
            {
                "name": component.name,
                "predictor": component.predictor,
                "term": component.term,
                "initial_lambda": component.initial_lambda,
                "final_lambda": component.final_lambda,
                "accepted_moves": component.accepted_moves,
                "dominant_update_share": component.dominant_update_share,
                "terminal_term_edf": component.terminal_term_edf,
                "null_space_dimension": component.null_space_dimension,
                "outcome": component.outcome,
                "exact_face_iteration": component.exact_face_iteration,
                "exact_face_effect": component.exact_face_effect,
                "upper_bound_iterations": component.upper_bound_iterations,
            }
            for component in profile.smoothing_components
        ],
    }


__all__ = [
    "DiagnosticAction",
    "DiagnosticEvidence",
    "DiagnosticFinding",
    "DiagnosticSubject",
    "FitDiagnosticReport",
    "FitPhaseProfile",
    "FitWorkProfile",
    "JsonScalar",
    "JsonValue",
    "SmoothingComponentProfile",
]
