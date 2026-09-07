"""Contract tests for the model-agnostic fit diagnostic report values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import numpy as np
import pytest

from superglm.diagnostics import (
    DiagnosticAction,
    DiagnosticEvidence,
    DiagnosticFinding,
    DiagnosticSubject,
    FitDiagnosticReport,
)


def _subject(*, identifier: str = "penalty:mean:ordered_age") -> DiagnosticSubject:
    return DiagnosticSubject(
        identifier=identifier,
        display_name='mean predictor term "ordered_age"',
        scope="penalized_subspace",
        predictor="mean",
        term="ordered_age",
        component="mean:ordered_age:wiggle",
    )


def _evidence(*, value: object | None = None) -> DiagnosticEvidence:
    if value is None:
        value = {
            "ordinary_updates": [18, 30],
            "terminal": {"at_upper_bound": True},
        }
    return DiagnosticEvidence(
        metric="dominant_update_share",
        value=value,
        unit="fraction",
        window="ordinary accepted smoothing updates",
        provenance="smoothing_history",
        comparator=">=",
        threshold={"minimum_updates": 4, "share": 0.5},
    )


def _action(*, question: str = "Would a linear representation retain predictive quality?"):
    return DiagnosticAction(
        kind="compare_linear_representation",
        question=question,
        requires_data=True,
        comparison_metrics=[
            "all distributional parameters",
            "fit work",
            "predictions",
            "held-out performance",
        ],
    )


def _finding(
    *,
    identifier: str = "mean:ordered_age:dominance",
    code: str = "smoothing.update_dominance",
    headline: str = "The ordered_age smooth dominated the remaining smoothing movement.",
    priority_tier: int = 3,
) -> DiagnosticFinding:
    return DiagnosticFinding(
        identifier=identifier,
        code=code,
        category="optimization_trajectory",
        severity="warning",
        confidence="strong",
        impacts=["fit_work", "term_complexity"],
        subject=_subject(identifier=f"penalty:{identifier}"),
        headline=headline,
        observed="It led 18 of 30 ordinary accepted smoothing updates.",
        interpretation="The penalized nonlinear directions may be strongly suppressed.",
        caveats=[
            "This does not establish that the predictor should be removed.",
            "A retained null-space effect may still be useful.",
        ],
        evidence=[_evidence()],
        actions=[_action()],
        priority_tier=priority_tier,
    )


def _report(*, findings: object | None = None, **overrides: object) -> FitDiagnosticReport:
    if findings is None:
        findings = [_finding()]
    values = {
        "schema_version": 1,
        "rule_set_version": 1,
        "model_type": "SuperLSS",
        "family": "GaussianLS",
        "fit_revision": 7,
        "scope": "fit",
        "fit_status": "converged_certified",
        "findings": findings,
        "coverage": ["solver history", "smoothing trajectory"],
        "limitations": ["Training rows and influence diagnostics were not examined."],
    }
    values.update(overrides)
    return FitDiagnosticReport(**values)


def test_construction_recursively_owns_containers_and_canonicalizes_reversed_inputs():
    """Aliasing or caller order must not change the canonical report value."""
    evidence_value = {
        "ordinary_updates": [18, 30],
        "terminal": {"at_upper_bound": True},
    }
    impacts = ["fit_work", "term_complexity"]
    evidence = _evidence(value=evidence_value)
    first = replace(_finding(), impacts=impacts, evidence=[evidence])
    second = _finding(
        identifier="fit:termination",
        code="fit.termination_uncertified",
        headline="The reported convergence was not certified.",
        priority_tier=1,
    )

    forward_report = _report(findings=[first, second])
    reversed_report = _report(findings=[second, first])
    evidence_value["ordinary_updates"].append(99)
    evidence_value["terminal"]["at_upper_bound"] = False
    impacts.append("predictions")

    assert forward_report.findings == (second, first)
    assert forward_report == reversed_report
    assert forward_report.to_dict() == reversed_report.to_dict()
    assert forward_report.render() == reversed_report.render()
    assert forward_report.render(detail="full") == reversed_report.render(detail="full")
    assert forward_report.coverage == ("solver history", "smoothing trajectory")
    assert forward_report.findings[1].impacts == ("fit_work", "term_complexity")
    assert forward_report.findings[1].evidence == (evidence,)
    assert forward_report.findings[1].actions[0].comparison_metrics == (
        "all distributional parameters",
        "fit work",
        "predictions",
        "held-out performance",
    )
    assert isinstance(evidence.value, MappingProxyType)
    assert evidence.value["ordinary_updates"] == (18, 30)
    assert isinstance(evidence.value["terminal"], MappingProxyType)
    assert evidence.value["terminal"]["at_upper_bound"] is True
    assert isinstance(evidence.threshold, MappingProxyType)
    with pytest.raises(TypeError):
        evidence.value["ordinary_updates"] = ()


def test_finding_ranking_uses_the_complete_canonical_key():
    """Every approved key component must defeat reversed caller insertion order."""

    def ranked_finding(
        identifier,
        *,
        priority,
        scope="whole_fit",
        confidence="strong",
        metric="unranked_count",
        share=0.0,
        code="same.code",
        subject_identifier="same:subject",
    ):
        evidence = replace(
            _evidence(value=share),
            metric=metric,
            comparator=None,
            threshold=None,
        )
        subject = replace(
            _subject(identifier=subject_identifier),
            scope=scope,
        )
        return replace(
            _finding(identifier=identifier, priority_tier=priority),
            code=code,
            confidence=confidence,
            subject=subject,
            evidence=[evidence],
        )

    canonical = [
        ranked_finding(
            "scope:z-whole-fit",
            priority=1,
            scope="whole_fit",
            subject_identifier="scope:z-whole-fit",
        ),
        ranked_finding(
            "scope:y-predictor",
            priority=1,
            scope="predictor",
            subject_identifier="scope:y-predictor",
        ),
        ranked_finding(
            "scope:x-term",
            priority=1,
            scope="term",
            subject_identifier="scope:x-term",
        ),
        ranked_finding(
            "scope:b-penalized-subspace",
            priority=1,
            scope="penalized_subspace",
            subject_identifier="scope:b-penalized-subspace",
        ),
        ranked_finding(
            "scope:a-level",
            priority=1,
            scope="level",
            subject_identifier="scope:a-level",
        ),
        ranked_finding("confidence:z-certified", priority=2, confidence="certified"),
        ranked_finding("confidence:y-strong", priority=2, confidence="strong"),
        ranked_finding("confidence:b-suggestive", priority=2, confidence="suggestive"),
        ranked_finding("confidence:a-unresolved", priority=2, confidence="unresolved"),
        ranked_finding(
            "share:z-dominant",
            priority=3,
            scope="term",
            metric="dominant_update_share",
            share=0.9,
        ),
        ranked_finding(
            "share:m-persistence",
            priority=3,
            scope="term",
            metric="persistence_share",
            share=0.8,
        ),
        ranked_finding(
            "share:a-work",
            priority=3,
            scope="term",
            metric="work_share",
            share=0.7,
        ),
        ranked_finding("code:z-identifier", priority=4, code="a.code"),
        ranked_finding("code:a-identifier", priority=4, code="z.code"),
        ranked_finding(
            "subject:z-identifier",
            priority=5,
            code="subject.code",
            subject_identifier="subject:a",
        ),
        ranked_finding(
            "subject:a-identifier",
            priority=5,
            code="subject.code",
            subject_identifier="subject:z",
        ),
        ranked_finding(
            "identifier:a",
            priority=6,
            code="identifier.code",
            subject_identifier="identifier:subject",
        ),
        ranked_finding(
            "identifier:z",
            priority=6,
            code="identifier.code",
            subject_identifier="identifier:subject",
        ),
    ]

    canonical_report = _report(findings=canonical)
    reversed_report = _report(findings=list(reversed(canonical)))

    assert tuple(finding.identifier for finding in reversed_report.findings) == (
        "scope:z-whole-fit",
        "scope:y-predictor",
        "scope:x-term",
        "scope:b-penalized-subspace",
        "scope:a-level",
        "confidence:z-certified",
        "confidence:y-strong",
        "confidence:b-suggestive",
        "confidence:a-unresolved",
        "share:z-dominant",
        "share:m-persistence",
        "share:a-work",
        "code:z-identifier",
        "code:a-identifier",
        "subject:z-identifier",
        "subject:a-identifier",
        "identifier:a",
        "identifier:z",
    )
    assert reversed_report == canonical_report
    assert reversed_report.to_dict() == canonical_report.to_dict()
    assert reversed_report.render() == canonical_report.render()


def test_to_dict_returns_a_fresh_mutable_json_tree_every_time():
    """Mutating one payload must not mutate the report or a later payload."""
    report = _report()

    first_payload = report.to_dict()
    first_payload["coverage"].append("caller mutation")
    first_payload["findings"][0]["impacts"].append("caller mutation")
    first_payload["findings"][0]["evidence"][0]["value"]["ordinary_updates"].append(99)

    second_payload = report.to_dict()

    assert "profile" not in second_payload
    assert second_payload["coverage"] == ["solver history", "smoothing trajectory"]
    assert second_payload["findings"][0]["impacts"] == ["fit_work", "term_complexity"]
    assert second_payload["findings"][0]["evidence"][0]["value"] == {
        "ordinary_updates": [18, 30],
        "terminal": {"at_upper_bound": True},
    }
    assert set(second_payload["findings"][0]) == {
        "identifier",
        "code",
        "category",
        "severity",
        "confidence",
        "impacts",
        "subject",
        "headline",
        "observed",
        "interpretation",
        "caveats",
        "evidence",
        "actions",
        "priority_tier",
    }
    assert set(second_payload["findings"][0]["subject"]) == {
        "identifier",
        "display_name",
        "scope",
        "predictor",
        "term",
        "component",
    }


@pytest.mark.parametrize(
    "value",
    [
        np.array([1.0]),
        object(),
        {1, 2},
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_evidence_rejects_non_json_or_nonfinite_values(value):
    """Raw arrays, objects, sets, and nonfinite numbers must not escape in evidence."""
    with pytest.raises((TypeError, ValueError)):
        _evidence(value=value)


def test_evidence_rejects_empty_metric_and_unknown_provenance():
    """Unlabelled or unauthoritative evidence would be uninterpretable downstream."""
    with pytest.raises(ValueError, match="metric"):
        replace(_evidence(), metric="")
    with pytest.raises(ValueError, match="provenance"):
        replace(_evidence(), provenance="training_rows")


def test_nested_json_mappings_require_string_keys_and_are_recursively_copied():
    """A nested non-string key is not JSON, and later nested mutation must be isolated."""
    with pytest.raises(TypeError, match="string keys"):
        _evidence(value={"outer": {1: "not JSON"}})

    nested = {"outer": [{"value": 3.0}]}
    evidence = _evidence(value=nested)
    nested["outer"][0]["value"] = 9.0

    assert evidence.value["outer"][0]["value"] == 3.0
    with pytest.raises(TypeError):
        evidence.value["outer"][0]["value"] = 5.0


@pytest.mark.parametrize(
    "value",
    [
        list(range(65)),
        tuple(range(65)),
        {f"metric_{index}": index for index in range(65)},
    ],
)
def test_evidence_rejects_containers_over_the_v1_item_budget(value):
    """One oversized sequence or mapping must not carry row-scale evidence."""
    with pytest.raises(ValueError, match="at most 64 items"):
        _evidence(value=value)


def test_evidence_rejects_trees_over_the_v1_total_node_budget():
    """Many individually small containers must not evade the total evidence bound."""
    value = [list(range(16)) for _ in range(16)]

    with pytest.raises(ValueError, match="at most 256 nodes"):
        _evidence(value=value)


def test_evidence_rejects_excessive_depth_without_leaking_recursion_error():
    """Adversarial nesting must fail at the contract boundary before Python recursion."""
    value: object = 0
    for _ in range(2048):
        value = [value]

    with pytest.raises(ValueError, match="at most 8 container levels"):
        _evidence(value=value)


@pytest.mark.parametrize("scope", ["model", "", None])
def test_subject_rejects_invalid_scope(scope):
    """A subject outside the stable scope vocabulary would break ranking consumers."""
    with pytest.raises((TypeError, ValueError), match="scope"):
        replace(_subject(), scope=scope)


@pytest.mark.parametrize("severity", ["fatal", "", None])
def test_finding_rejects_invalid_severity(severity):
    """Severity must stay inside the public three-level vocabulary."""
    with pytest.raises((TypeError, ValueError), match="severity"):
        replace(_finding(), severity=severity)


@pytest.mark.parametrize("confidence", ["certain", "", None])
def test_finding_rejects_invalid_confidence(confidence):
    """Confidence must preserve the stable evidence-authority vocabulary."""
    with pytest.raises((TypeError, ValueError), match="confidence"):
        replace(_finding(), confidence=confidence)


@pytest.mark.parametrize("priority_tier", [0, 7, 1.5, True])
def test_finding_rejects_invalid_priority_tier(priority_tier):
    """Only integer urgency tiers one through six can participate in ranking."""
    with pytest.raises((TypeError, ValueError), match="priority_tier"):
        replace(_finding(), priority_tier=priority_tier)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 3),
        ("rule_set_version", 2),
        ("scope", "model"),
        ("fit_status", "converged"),
    ],
)
def test_report_rejects_unknown_versions_scope_or_status(field, value):
    """Unknown report contracts must fail rather than masquerade as version one."""
    with pytest.raises((TypeError, ValueError), match=field):
        _report(**{field: value})


def test_report_rejects_duplicate_finding_identifiers():
    """Duplicate identifiers would make structured findings ambiguous to consumers."""
    first = _finding()
    duplicate = replace(first, code="smoothing.penalty_at_upper_boundary")

    with pytest.raises(ValueError, match="duplicate finding identifier"):
        _report(findings=[first, duplicate])


def test_finding_rejects_an_empty_suggested_action_list():
    """Every valid finding must render at least one concrete suggested experiment."""
    with pytest.raises(ValueError, match="actions must contain at least one"):
        replace(_finding(), actions=[])


@pytest.mark.parametrize(
    ("instance", "field", "replacement"),
    [
        (_subject(), "identifier", "changed"),
        (_evidence(), "metric", "changed"),
        (_action(), "question", "changed"),
        (_finding(), "headline", "changed"),
        (_report(), "fit_status", "not_converged"),
    ],
)
def test_report_values_are_frozen_slot_instances(instance, field, replacement):
    """Public report values must permit neither field reassignment nor ad-hoc state."""
    assert not hasattr(instance, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field, replacement)
    with pytest.raises((AttributeError, TypeError)):
        setattr(instance, "undeclared_attribute", replacement)


def test_compact_and_full_rendering_expose_semantic_sections():
    """Rendering must expose the stable semantic contract without freezing paragraphs."""
    report = _report()

    compact = report.render()
    full = report.render(detail="full")

    for expected in ("SuperLSS", "GaussianLS", "converged_certified"):
        assert expected in compact
    assert "1." in compact
    assert "smoothing.update_dominance" in compact
    assert 'mean predictor term "ordered_age"' in compact
    assert "Would a linear representation retain predictive quality?" in compact
    assert "Coverage" in compact
    assert "Limitations" in compact
    assert str(report) == compact

    assert "What we observed" in full
    assert "It led 18 of 30 ordinary accepted smoothing updates." in full
    assert "What this may mean" in full
    assert "What this does not establish" in full
    assert "This does not establish that the predictor should be removed." in full
    assert "Suggested experiment" in full
    assert "Would a linear representation retain predictive quality?" in full
    assert "Technical evidence" in full
    assert "dominant_update_share" in full
    assert "$" not in compact
    assert "$" not in full

    with pytest.raises(ValueError, match="detail"):
        report.render(detail="verbose")


def test_empty_report_states_the_bounded_claim_and_still_renders_limitations():
    """No findings must not be misreported as proof that the statistical model is healthy."""
    report = _report(
        findings=[],
        limitations=["Row-scale residual and influence evidence was not examined."],
    )

    compact = report.render()
    full = report.render(detail="full")

    message = "No solver pathology was detected in the available evidence"
    assert message in compact
    assert message in full
    assert "Limitations" in compact
    assert "Row-scale residual and influence evidence was not examined." in compact
    assert "Limitations" in full
    assert "Row-scale residual and influence evidence was not examined." in full
