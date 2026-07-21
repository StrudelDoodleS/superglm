"""Read-only descriptions of fitted design storage and planned matrix routes."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import pandas as pd

from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)

_StorageRows = Literal[
    "matrix",
    "codes",
    "sparse-basis",
    "spline-level-support",
    "unique-basis",
    "unique-scop-basis",
]


@dataclass(frozen=True)
class _RepresentationMetadata:
    representation: str
    compressed: bool
    storage_rows: _StorageRows
    ordinary_candidate: bool = False
    specialised_discrete_route: str | None = None


_REPRESENTATION_BY_TYPE = MappingProxyType(
    {
        DenseGroupMatrix: _RepresentationMetadata(
            "dense",
            False,
            "matrix",
            ordinary_candidate=True,
        ),
        SparseGroupMatrix: _RepresentationMetadata(
            "sparse-csr",
            False,
            "matrix",
            ordinary_candidate=True,
        ),
        CategoricalGroupMatrix: _RepresentationMetadata(
            "categorical-codes",
            False,
            "codes",
            ordinary_candidate=True,
        ),
        SparseSSPGroupMatrix: _RepresentationMetadata(
            "sparse-ssp",
            False,
            "sparse-basis",
        ),
        SplineCategoricalGroupMatrix: _RepresentationMetadata(
            "spline-categorical",
            False,
            "spline-level-support",
        ),
        DiscretizedSSPGroupMatrix: _RepresentationMetadata(
            "discretized-ssp",
            True,
            "unique-basis",
            specialised_discrete_route="binned-ssp",
        ),
        DiscretizedSCOPGroupMatrix: _RepresentationMetadata(
            "discretized-scop",
            True,
            "unique-scop-basis",
            specialised_discrete_route="binned-scop",
        ),
        DiscretizedSplineCategoricalGroupMatrix: _RepresentationMetadata(
            "discretized-spline-categorical",
            True,
            "unique-basis",
            specialised_discrete_route="binned-spline-categorical",
        ),
        DiscretizedTensorGroupMatrix: _RepresentationMetadata(
            "discretized-tensor",
            True,
            "unique-basis",
            specialised_discrete_route="observed-tensor-support",
        ),
    }
)

_SUMMARY_COLUMNS = [
    "term",
    "feature",
    "solver_start",
    "solver_end",
    "n_columns",
    "representation",
    "compressed",
    "storage_rows",
    "ordinary_tabmat_partition",
    "specialised_discrete_route",
    "route_reason",
]


def _representation_metadata(matrix) -> _RepresentationMetadata:
    matrix_type = type(matrix)
    metadata = _REPRESENTATION_BY_TYPE.get(matrix_type)
    if metadata is not None:
        return metadata
    for base in matrix_type.__mro__[1:]:
        metadata = _REPRESENTATION_BY_TYPE.get(base)
        if metadata is not None:
            return metadata
    raise TypeError(f"Unsupported fitted group-matrix representation: {matrix_type.__name__}")


def _storage_row_count(matrix, kind: _StorageRows) -> int:
    if kind == "matrix":
        return int(matrix.M.shape[0])
    if kind == "codes":
        return int(matrix.codes.shape[0])
    if kind == "sparse-basis":
        return int(matrix.B.shape[0])
    if kind == "spline-level-support":
        return int(matrix.B_level.shape[0])
    if kind == "unique-scop-basis":
        return int(matrix.B_scop_unique.shape[0])
    return int(matrix.B_unique.shape[0])


def build_design_summary(model) -> pd.DataFrame:
    """Describe fitted storage and static route eligibility without executing a route.

    ``ordinary_tabmat_partition`` reports a construction-time eligibility
    decision. It does not prove that a SplitMatrix was built or that a Tabmat
    kernel was called; fit and REML traces remain authoritative for execution.
    """
    if getattr(model, "_result", None) is None:
        raise RuntimeError("Model must be fitted before calling design_summary().")
    design = getattr(model, "_dm", None)
    if design is None:
        raise RuntimeError(
            "retain_fit_state=False discarded the fitted design; refit with "
            "retain_fit_state=True before calling design_summary()."
        )

    plan = design.execution_plan
    ordinary_indices = frozenset(plan.ordinary_indices)
    rows: list[dict[str, object]] = []
    for index, (group, matrix, span) in enumerate(
        zip(model._groups, design.group_matrices, plan.group_spans, strict=True)
    ):
        metadata = _representation_metadata(matrix)
        if (group.start, group.end) != (span.start, span.end):
            raise ValueError("fitted group slices do not match the design execution plan")
        if metadata.ordinary_candidate:
            route_reason = plan.ordinary_partition_reason
        elif metadata.compressed:
            route_reason = "contains-compressed-group"
        else:
            route_reason = "specialised-group-representation"
        rows.append(
            {
                "term": group.name,
                "feature": group.feature_name,
                "solver_start": span.start,
                "solver_end": span.end,
                "n_columns": span.end - span.start,
                "representation": metadata.representation,
                "compressed": metadata.compressed,
                "storage_rows": _storage_row_count(matrix, metadata.storage_rows),
                "ordinary_tabmat_partition": index in ordinary_indices,
                "specialised_discrete_route": metadata.specialised_discrete_route,
                "route_reason": route_reason,
            }
        )
    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


__all__ = ["build_design_summary"]
