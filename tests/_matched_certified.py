"""Raising form of the strict smoothing certification contract.

``SmoothingResult.matched_certified`` states the contract as a property; this
helper checks a subset of its clauses independently, one named refusal at a
time. It leaves out two clauses of the property: the check that
``tuple(terminal_endpoint_directions) == face_names``, and the refusal of a
stationary outcome with a component resting at the maximum lambda.
"""

from __future__ import annotations

from superglm.distributional.results.smoothing import _revalidated_endpoint_directions


def assert_matched_certified(smoothing) -> None:
    if smoothing.convergence_reason == "practical_plateau":
        raise RuntimeError("algorithm-matched certification requires strict smoothing convergence")
    face = smoothing.terminal_fit.coefficient_face
    if any(item.deactivated_face_components for item in smoothing.history):
        raise RuntimeError(
            "algorithm-matched certification forbids a terminal exact-face revalidation failure"
        )
    if face is not None:
        raise RuntimeError("the exact coefficient face is numerically supported but not certified")
    if any(
        item.refused_face_components or item.endpoint_assessment_failure_reason is not None
        for item in smoothing.history
    ):
        raise RuntimeError("algorithm-matched certification forbids a refused endpoint assessment")
    if any(
        not evidence._derived_authority_matches()
        or evidence.authority_identifier != "analytic-observed-curvature-direction/v1"
        for evidence in smoothing.terminal_endpoint_directions.values()
    ):
        raise RuntimeError(
            "algorithm-matched certification requires self-consistent endpoint evidence"
        )
    for item in smoothing.history:
        evidence = item.endpoint_direction_evidence
        if evidence is None:
            continue
        if not evidence.fit_indices or any(
            index >= len(smoothing.coefficient_fits)
            or not smoothing.coefficient_fits[index].converged
            or smoothing.coefficient_fits[index].config.tolerance != evidence.coefficient_tolerance
            for index in evidence.fit_indices
        ):
            raise RuntimeError(
                "algorithm-matched certification requires converged endpoint assessment fits"
            )
    final_revalidations = {
        name: direction
        for item in smoothing.history
        for name, direction in _revalidated_endpoint_directions(item)
    }
    if dict(smoothing.terminal_endpoint_directions) != final_revalidations:
        raise RuntimeError("algorithm-matched certification requires the final face revalidation")
    if smoothing.fallback_count or any(
        fit.terminal_curvature.actual_source != fit.terminal_curvature.requested_source
        for fit in smoothing.coefficient_fits
    ):
        raise RuntimeError("algorithm-matched certification forbids curvature fallback")
    if smoothing.unresolved_upper_bound:
        raise RuntimeError("algorithm-matched certification forbids an unresolved upper bound")
    if not smoothing.terminal_evidence_fresh:
        raise RuntimeError("algorithm-matched certification requires fresh convergence evidence")
    if not smoothing.converged or not smoothing.coefficient_converged:
        raise RuntimeError("algorithm-matched certification requires converged inner and EFS fits")
