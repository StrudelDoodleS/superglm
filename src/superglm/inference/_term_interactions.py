"""Internal interaction-term inference helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from superglm.inference._term_types import InteractionInference

if TYPE_CHECKING:
    from superglm.solvers.pirls import PIRLSResult
    from superglm.types import GroupSlice


def _interaction_inference(
    name: str,
    *,
    result: PIRLSResult,
    groups: list[GroupSlice],
    interaction_specs: dict[str, Any],
) -> InteractionInference:
    """Build an InteractionInference from a fitted interaction."""
    ispec = interaction_specs[name]
    feature_groups = [g for g in groups if g.feature_name == name]
    beta_combined = np.concatenate([result.beta[g.sl] for g in feature_groups])
    active = bool(np.linalg.norm(beta_combined) > 1e-12)

    raw = ispec.reconstruct(beta_combined)

    if "per_level" in raw and "x" in raw:
        return InteractionInference(
            name=name,
            kind="spline_categorical",
            active=active,
            x=raw["x"],
            levels=raw["levels"],
            per_level=raw["per_level"],
        )

    if "pairs" in raw:
        return InteractionInference(
            name=name,
            kind="categorical",
            active=active,
            pairs=raw["pairs"],
            log_relativity=raw["log_relativities"],
            relativity=raw["relativities"],
        )

    if "relativities_per_unit" in raw:
        return InteractionInference(
            name=name,
            kind="numeric_categorical",
            active=active,
            levels=raw["levels"],
            relativities_per_unit=raw["relativities_per_unit"],
            log_relativities_per_unit=raw["log_relativities_per_unit"],
        )

    if "relativity_per_unit_unit" in raw:
        return InteractionInference(
            name=name,
            kind="numeric",
            active=active,
            relativity_per_unit_unit=raw["relativity_per_unit_unit"],
            coef=raw["coef"],
        )

    return InteractionInference(
        name=name,
        kind="surface",
        active=active,
    )


__all__ = ["_interaction_inference"]
