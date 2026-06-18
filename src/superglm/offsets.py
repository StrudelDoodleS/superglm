"""Deployable fixed offsets for SuperGLM models.

A plain offset array preserves its row values but not the expression that created
it.  This module adds structured offsets that can be reproduced by deployment
runtimes such as SQL.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from functools import wraps
from html import escape
from typing import Any, Sequence, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class LogRatioOffset:
    """Fixed offset ``coefficient * log(source_feature / reference)``.

    For a log-link model, the response-scale multiplier is
    ``(source_feature / reference) ** coefficient``.  The coefficient is fixed,
    not estimated, so it has no standard error or p-value.

    Parameters
    ----------
    name : str
        Stable deployment name for the offset term.
    source_feature : str
        Input column used by the transformation.
    reference : float
        Positive reference value in the ratio.  For policy term measured in
        months, ``reference=12`` creates an annual-term multiplier.
    coefficient : float
        Fixed link-scale coefficient.  Defaults to 1.
    """

    name: str
    source_feature: str
    reference: float
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("name", self.name),
            ("source_feature", self.source_feature),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            if _SAFE_NAME.fullmatch(value.strip()) is None:
                raise ValueError(
                    f"{field_name} must be a SQL/JSON-safe identifier containing only "
                    "letters, digits, and underscores"
                )

        reference = float(self.reference)
        coefficient = float(self.coefficient)
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("reference must be a positive finite number")
        if not np.isfinite(coefficient):
            raise ValueError("coefficient must be finite")

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "source_feature", self.source_feature.strip())
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "coefficient", coefficient)

    @property
    def transform_type(self) -> str:
        return "LOG_RATIO"

    @property
    def term_type(self) -> str:
        return "FIXED_OFFSET"

    @property
    def link_expression(self) -> str:
        return (
            f"{self.coefficient:.15g} * "
            f"log({self.source_feature} / {self.reference:.15g})"
        )

    @property
    def response_multiplier_expression(self) -> str:
        return (
            f"({self.source_feature} / {self.reference:.15g})"
            f" ^ {self.coefficient:.15g}"
        )

    def evaluate(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Evaluate the link-scale offset for a model frame."""
        if self.source_feature not in X.columns:
            raise KeyError(
                f"Offset {self.name!r} requires source feature "
                f"{self.source_feature!r}"
            )
        try:
            values = np.asarray(X[self.source_feature], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Offset source feature {self.source_feature!r} must be numeric"
            ) from exc
        if values.ndim != 1:
            raise ValueError(
                f"Offset source feature {self.source_feature!r} must be one-dimensional"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"Offset source feature {self.source_feature!r} must contain only finite values"
            )
        if np.any(values <= 0):
            raise ValueError(
                f"Offset source feature {self.source_feature!r} must be strictly positive "
                "for a log-ratio offset"
            )
        return self.coefficient * np.log(values / self.reference)

    def to_record(self, *, sequence_no: int) -> dict[str, str | float | int]:
        """Return the stable workbook and deployment representation."""
        return {
            "Term": self.name,
            "Term Type": self.term_type,
            "Source Feature": self.source_feature,
            "Transform": self.transform_type,
            "Reference Value": self.reference,
            "Coefficient": self.coefficient,
            "Sequence": int(sequence_no),
            "Link Expression": self.link_expression,
            "Response Multiplier": self.response_multiplier_expression,
        }


DeployableOffset: TypeAlias = LogRatioOffset
DeployableOffsetInput: TypeAlias = (
    NDArray | Sequence[float] | DeployableOffset | Sequence[DeployableOffset] | None
)


def _as_deployable_terms(offset: Any) -> tuple[DeployableOffset, ...] | None:
    if isinstance(offset, LogRatioOffset):
        return (offset,)
    if isinstance(offset, (list, tuple)) and offset and all(
        isinstance(term, LogRatioOffset) for term in offset
    ):
        names = [term.name for term in offset]
        if len(names) != len(set(names)):
            raise ValueError("Deployable offset names must be unique")
        return tuple(offset)
    return None


def evaluate_deployable_offsets(
    terms: Sequence[DeployableOffset],
    X: pd.DataFrame,
) -> NDArray[np.float64] | None:
    """Evaluate and add one or more fixed offsets on the link scale."""
    if not terms:
        return None
    values = np.zeros(len(X), dtype=np.float64)
    for term in terms:
        values += term.evaluate(X)
    return values


def resolve_offset_input(
    offset: DeployableOffsetInput,
    X: pd.DataFrame,
) -> tuple[Any, tuple[DeployableOffset, ...]]:
    """Resolve structured metadata to values while retaining the expression."""
    terms = _as_deployable_terms(offset)
    if terms is None:
        return offset, ()
    return evaluate_deployable_offsets(terms, X), terms


def deployable_offsets(model: Any) -> tuple[DeployableOffset, ...]:
    """Return fixed deployment offsets attached by the most recent fit."""
    return tuple(getattr(model, "_deployable_offsets", ()) or ())


def fixed_offset_frame(model: Any) -> pd.DataFrame:
    """Return workbook-ready fixed-offset metadata for a fitted model."""
    columns = [
        "Term",
        "Term Type",
        "Source Feature",
        "Transform",
        "Reference Value",
        "Coefficient",
        "Sequence",
        "Link Expression",
        "Response Multiplier",
    ]
    rows = [
        term.to_record(sequence_no=sequence_no)
        for sequence_no, term in enumerate(deployable_offsets(model), start=1)
    ]
    return pd.DataFrame(rows, columns=columns)


class FixedOffsetModelSummary:
    """ModelSummary proxy that reports offsets outside fitted coefficients."""

    def __init__(self, base_summary: Any, terms: Sequence[DeployableOffset]):
        self._base_summary = base_summary
        self._terms = tuple(terms)

    def _records(self) -> list[dict[str, str | float | int]]:
        return [
            term.to_record(sequence_no=sequence_no)
            for sequence_no, term in enumerate(self._terms, start=1)
        ]

    def to_dict(self) -> dict[str, Any]:
        data = dict(self._base_summary.to_dict())
        data["fixed_offsets"] = self._records()
        return data

    def __contains__(self, key: str) -> bool:
        return key == "fixed_offsets" or key in self._base_summary

    def __getitem__(self, key: str) -> Any:
        if key == "fixed_offsets":
            return self._records()
        return self._base_summary[key]

    def items(self):
        return self.to_dict().items()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_summary, name)

    def __str__(self) -> str:
        lines = [str(self._base_summary), "", "Fixed offsets (not estimated):"]
        for term in self._terms:
            lines.append(
                f"  {term.name}: eta += {term.link_expression}; "
                f"response multiplier = {term.response_multiplier_expression}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def _repr_html_(self) -> str:
        base_html = self._base_summary._repr_html_()
        rows = []
        for term in self._terms:
            rows.append(
                "<tr>"
                f"<td style='padding:3px 8px;text-align:left'>{escape(term.name)}</td>"
                f"<td style='padding:3px 8px;text-align:left'>{escape(term.link_expression)}</td>"
                f"<td style='padding:3px 8px;text-align:left'>"
                f"{escape(term.response_multiplier_expression)}</td>"
                "<td style='padding:3px 8px;text-align:left'>fixed; not estimated</td>"
                "</tr>"
            )
        table = (
            "<table style='border-collapse:collapse;font-family:monospace;font-size:13px;"
            "margin:8px 0'>"
            "<tr><th colspan='4' style='text-align:left;padding:5px 8px'>"
            "Fixed offsets (not estimated)</th></tr>"
            "<tr><th style='text-align:left;padding:3px 8px'>Term</th>"
            "<th style='text-align:left;padding:3px 8px'>Link expression</th>"
            "<th style='text-align:left;padding:3px 8px'>Response multiplier</th>"
            "<th style='text-align:left;padding:3px 8px'>Inference</th></tr>"
            + "".join(rows)
            + "</table>"
        )
        return base_html + table


def _prediction_offset(model: Any, X: pd.DataFrame, offset: Any) -> Any:
    resolved, explicit_terms = resolve_offset_input(offset, X)
    if explicit_terms:
        return resolved
    if offset is None:
        return evaluate_deployable_offsets(deployable_offsets(model), X)
    return resolved


def _install_fit_wrapper(model_cls: type, method_name: str) -> None:
    original = getattr(model_cls, method_name)

    @wraps(original)
    def wrapped(self, X, y, sample_weight=None, offset=None, **kwargs):
        resolved_offset, terms = resolve_offset_input(offset, X)
        previous_terms = deployable_offsets(self)
        self._deployable_offsets = ()
        try:
            result = original(self, X, y, sample_weight, resolved_offset, **kwargs)
        except Exception:
            self._deployable_offsets = previous_terms
            raise

        if terms and type(self._link).__name__ != "LogLink":
            self._deployable_offsets = ()
            raise ValueError("LogRatioOffset deployment metadata requires a log-link model")
        self._deployable_offsets = terms
        return result

    setattr(model_cls, method_name, wrapped)


def install_deployable_offset_support(model_cls: type) -> None:
    """Install structured-offset fit, prediction, summary, and export support."""
    if getattr(model_cls, "_deployable_offset_support_installed", False):
        return

    for method_name in ("fit", "fit_path", "fit_reml"):
        _install_fit_wrapper(model_cls, method_name)

    original_summary = model_cls.summary

    @wraps(original_summary)
    def summary(self, *args, **kwargs):
        result = original_summary(self, *args, **kwargs)
        terms = deployable_offsets(self)
        return FixedOffsetModelSummary(result, terms) if terms else result

    model_cls.summary = summary

    original_metrics = model_cls.metrics

    @wraps(original_metrics)
    def metrics(self, X, y, sample_weight=None, offset=None):
        resolved_offset = _prediction_offset(self, X, offset)
        return original_metrics(self, X, y, sample_weight, resolved_offset)

    model_cls.metrics = metrics

    base_module = importlib.import_module("superglm.model.base")
    original_predict_eta_exact = base_module.predict_eta_exact
    original_predict_eta_fast = base_module.predict_eta_fast_discrete

    @wraps(original_predict_eta_exact)
    def predict_eta_exact(model, X, offset=None):
        return original_predict_eta_exact(model, X, _prediction_offset(model, X, offset))

    @wraps(original_predict_eta_fast)
    def predict_eta_fast_discrete(model, X, offset=None):
        return original_predict_eta_fast(model, X, _prediction_offset(model, X, offset))

    base_module.predict_eta_exact = predict_eta_exact
    base_module.predict_eta_fast_discrete = predict_eta_fast_discrete

    rating_tables_module = importlib.import_module("superglm.export.rating_tables")
    offset_block = getattr(rating_tables_module, "_offset_multiplier_block", None)
    if offset_block is not None:

        @wraps(offset_block)
        def offset_multiplier_block(model, *args, **kwargs):
            if deployable_offsets(model):
                return None
            return offset_block(model, *args, **kwargs)

        rating_tables_module._offset_multiplier_block = offset_multiplier_block

    model_cls._deployable_offset_support_installed = True


__all__ = [
    "DeployableOffset",
    "DeployableOffsetInput",
    "FixedOffsetModelSummary",
    "LogRatioOffset",
    "deployable_offsets",
    "evaluate_deployable_offsets",
    "fixed_offset_frame",
    "install_deployable_offset_support",
    "resolve_offset_input",
]
