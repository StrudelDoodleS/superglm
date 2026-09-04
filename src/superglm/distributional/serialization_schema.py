"""The artifact schema this build writes, the older majors it reads, and how the
manifest names a type.

Policy lives with :mod:`superglm.distributional.serialization`; this module
holds the data that policy consults, so the reader and the writer agree on it.
"""

from __future__ import annotations

# 8.0.0 is the first public schema.  It advertises distribution-function
# capabilities consistently with the methods implemented by each family.
SCHEMA_VERSION = "8.0.0"

# No earlier distributional artifact schema was released publicly.  Treat all
# other majors as incompatible instead of carrying migration promises for
# development-only artifacts.
READABLE_PREVIOUS_MAJORS: frozenset[int] = frozenset()

#: Manifest type names are the classes' canonical compatibility paths, held
#: fixed across the package reorganisation so the manifest authenticates fitted
#: state, not module layout. Every canonical path still resolves through its
#: compatibility shim.
CANONICAL_TYPE_MODULES = {
    "superglm.distributional.results.endpoint_evidence": "superglm.distributional.result",
    "superglm.distributional.results.fit": "superglm.distributional.result",
    "superglm.distributional.results.iteration": "superglm.distributional.result",
    "superglm.distributional.results.smoothing": "superglm.distributional.result",
    "superglm.distributional.results.solver": "superglm.distributional.result",
    "superglm.distributional.smoothing.acceleration": "superglm.distributional.efs_acceleration",
    "superglm.distributional.smoothing.authority": "superglm.distributional.efs",
    "superglm.distributional.smoothing.endpoint_direction": (
        "superglm.distributional.endpoint_direction"
    ),
    "superglm.distributional.smoothing.endpoint_laml": "superglm.distributional.endpoint_laml",
    "superglm.distributional.smoothing.evidence": "superglm.distributional.efs",
    "superglm.distributional.smoothing.face_efs": "superglm.distributional.face_efs",
    "superglm.distributional.smoothing.faces": "superglm.distributional.efs",
    "superglm.distributional.smoothing.loop": "superglm.distributional.efs",
    "superglm.distributional.smoothing.objective": "superglm.distributional.efs",
    "superglm.distributional.smoothing.penalty_face": "superglm.distributional.penalty_face",
    "superglm.distributional.smoothing.proposals": "superglm.distributional.efs",
    "superglm.distributional.solver.solver": "superglm.distributional.solver",
    "superglm.distributional.solver.curvature": "superglm.distributional.curvature",
    "superglm.distributional.solver.assembly": "superglm.distributional.assembly",
    "superglm.distributional.solver.chunks": "superglm.distributional.chunks",
    "superglm.distributional.solver.derivatives": "superglm.distributional.derivatives",
    "superglm.distributional.solver.packing": "superglm.distributional.packing",
}


def qualified_type(value: object) -> str:
    """The manifest name of ``value``'s type: its canonical module and qualname."""
    value_type = type(value)
    module = CANONICAL_TYPE_MODULES.get(value_type.__module__, value_type.__module__)
    return f"{module}.{value_type.__qualname__}"


__all__ = [
    "CANONICAL_TYPE_MODULES",
    "READABLE_PREVIOUS_MAJORS",
    "SCHEMA_VERSION",
    "qualified_type",
]
