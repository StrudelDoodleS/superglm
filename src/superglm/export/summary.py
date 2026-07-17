"""Renderer-independent model-summary export payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from superglm.features.categorical import Categorical
from superglm.features.interaction import SplineCategorical, TensorInteraction
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.spline import _SplineBase
from superglm.solvers.rank import selected_group_name_set

if TYPE_CHECKING:
    from superglm.model import SuperGLM


@dataclass(frozen=True)
class SummaryOverviewRow:
    """One typed key-value row in the model overview."""

    section: str
    metric: str
    value: str | int | float | bool | None


@dataclass(frozen=True)
class SummaryTermRow:
    """One renderer-independent coefficient or smooth-inference row."""

    term: str
    group: str
    kind: str
    estimate: float | None
    std_error: float | None
    statistic: float | None
    statistic_type: str
    p_value: float | None
    ci_lower: float | None
    ci_upper: float | None
    edf: float | None
    smoothing_lambda: float | None
    active: bool
    significance: str
    warning: str


@dataclass(frozen=True)
class SummaryExportPayload:
    """Structured model summary ready for a renderer such as Excel."""

    overview: tuple[SummaryOverviewRow, ...]
    terms: tuple[SummaryTermRow, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _CompactSummarySource:
    """Private compact-summary storage isolated at the export boundary."""

    data: dict[str, Any]
    info: dict[str, Any]
    rows: tuple[Any, ...]


_PARAMETRIC_WALD_NOTE = (
    "Parametric p-values are Wald approximations.\n"
    "For borderline significance, use a likelihood ratio test."
)
_SMOOTH_WOOD_NOTE = "Smooth p-values use Wood (2013) Bayesian tests."
_GROUP_WALD_NOTE = "Group chi-square p-values are Wald approximations."
_EDITOR_STALE_NOTE = (
    "Editor edits applied: coefficient standard errors, confidence intervals, "
    "and p-values are suppressed because they belong to the original fitted "
    "model, not the manually edited coefficients."
)
_EDITOR_OFFSET_NOTE = (
    "Editor offset refit: listed editor terms are fixed offset factors. "
    "Inference is conditional on those fixed offsets."
)
_QS_NOTE = (
    "QS: quasi-complete separation — a predictor perfectly or nearly predicts\n"
    "zero response, so the log-link coefficient diverges to -∞ and no finite\n"
    "MLE exists. Flagged levels have <20 obs or <0.05% exposure."
)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _finite_int(value: Any) -> int | None:
    number = _finite_float(value)
    return int(number) if number is not None else None


def _display_method(value: Any) -> str:
    method = str(value or "")
    return "MLE" if method == "ML" else method


def _profile_ci(info: dict[str, Any], key: str) -> tuple[float | None, float | None]:
    ci = info.get(key)
    if ci is None or len(ci) < 2:
        return None, None
    return _finite_float(ci[0]), _finite_float(ci[1])


def _adapt_compact_summary(summary) -> _CompactSummarySource:
    """Adapt the current compact ``ModelSummary`` internals in one place.

    ``ModelSummary`` has no public row iterator yet. Keeping its private info
    and coefficient-row access here prevents renderer details from leaking
    into the typed export contract.
    """
    return _CompactSummarySource(
        data=summary.to_dict(),
        info=summary._info,
        rows=tuple(summary._coef_rows),
    )


def _overview_rows(source: _CompactSummarySource) -> tuple[SummaryOverviewRow, ...]:
    info = source.info
    data = source.data
    fit = data.get("fit", {})
    deviance = data.get("deviance", {})
    criteria = data.get("information_criteria", {})

    rows = [
        SummaryOverviewRow("Model", "Family", str(info.get("family", ""))),
        SummaryOverviewRow("Model", "Link", str(info.get("link", ""))),
        SummaryOverviewRow("Model", "Method", _display_method(info.get("method", "ML"))),
        SummaryOverviewRow("Model", "Penalty", str(info.get("penalty", ""))),
        SummaryOverviewRow("Fit", "Observations", _finite_int(info.get("n_obs", fit.get("n_obs")))),
        SummaryOverviewRow(
            "Fit",
            "Effective DF",
            _finite_float(info.get("effective_df", fit.get("effective_df"))),
        ),
        SummaryOverviewRow("Fit", "Scale (phi)", _finite_float(info.get("phi", fit.get("phi")))),
        SummaryOverviewRow(
            "Fit", "Deviance", _finite_float(info.get("deviance", deviance.get("deviance")))
        ),
        SummaryOverviewRow("Fit", "Null Deviance", _finite_float(deviance.get("null_deviance"))),
        SummaryOverviewRow(
            "Fit", "Explained Deviance", _finite_float(deviance.get("explained_deviance"))
        ),
        SummaryOverviewRow("Fit", "Converged", bool(info.get("converged", False))),
        SummaryOverviewRow("Fit", "Iterations", _finite_int(info.get("n_iter"))),
        SummaryOverviewRow(
            "Information Criteria",
            "Log-Likelihood",
            _finite_float(info.get("log_likelihood", criteria.get("log_likelihood"))),
        ),
        SummaryOverviewRow(
            "Information Criteria", "AIC", _finite_float(info.get("aic", criteria.get("aic")))
        ),
        SummaryOverviewRow(
            "Information Criteria",
            "AICc",
            _finite_float(info.get("aicc", criteria.get("aicc"))),
        ),
        SummaryOverviewRow(
            "Information Criteria", "BIC", _finite_float(info.get("bic", criteria.get("bic")))
        ),
        SummaryOverviewRow(
            "Information Criteria",
            "EBIC",
            _finite_float(info.get("ebic", criteria.get("ebic"))),
        ),
    ]

    if "nb_theta" in info:
        ci_lower, ci_upper = _profile_ci(info, "nb_theta_ci")
        rows.extend(
            [
                SummaryOverviewRow(
                    "Distribution Profile", "NB2 Theta", _finite_float(info["nb_theta"])
                ),
                SummaryOverviewRow("Distribution Profile", "NB2 Theta CI Lower", ci_lower),
                SummaryOverviewRow("Distribution Profile", "NB2 Theta CI Upper", ci_upper),
                SummaryOverviewRow(
                    "Distribution Profile",
                    "NB2 Theta Method",
                    str(info.get("nb_theta_method", "")),
                ),
            ]
        )

    if "tweedie_p" in info:
        ci_lower, ci_upper = _profile_ci(info, "tweedie_p_ci")
        rows.extend(
            [
                SummaryOverviewRow(
                    "Distribution Profile", "Tweedie p", _finite_float(info["tweedie_p"])
                ),
                SummaryOverviewRow("Distribution Profile", "Tweedie p CI Lower", ci_lower),
                SummaryOverviewRow("Distribution Profile", "Tweedie p CI Upper", ci_upper),
                SummaryOverviewRow(
                    "Distribution Profile",
                    "Tweedie p CI Status",
                    str(info.get("tweedie_p_ci_status", "")),
                ),
                SummaryOverviewRow(
                    "Distribution Profile", "Tweedie phi", _finite_float(info.get("tweedie_phi"))
                ),
                SummaryOverviewRow(
                    "Distribution Profile",
                    "Tweedie p Method",
                    str(info.get("tweedie_p_method", "")),
                ),
            ]
        )

    return tuple(rows)


def _canonical_level_row_names(model: SuperGLM) -> set[str]:
    names: set[str] = set()
    for feature_name, spec in model._specs.items():
        groups = [group for group in model._groups if group.feature_name == feature_name]
        if isinstance(spec, OrderedCategorical):
            levels = (
                spec._ordered_levels
                if spec.basis == "spline"
                else [level for level in spec._ordered_levels if level != spec._base_level]
            )
            names.update(f"{feature_name}[{level}]" for level in levels)
        elif isinstance(spec, Categorical):
            for group in groups:
                names.update(f"{group.name}[{level}]" for level in spec._non_base)
    return names


def _source_groups_for_row(model: SuperGLM, row) -> tuple[Any, ...]:
    term = str(row.name)
    for feature_name, spec in model._specs.items():
        if isinstance(spec, OrderedCategorical) and (
            term == feature_name or term.startswith(f"{feature_name}[")
        ):
            return tuple(group for group in model._groups if group.feature_name == feature_name)
    return tuple(
        group for group in model._groups if term == group.name or term.startswith(f"{group.name}[")
    )


def _source_spec(model: SuperGLM, groups: tuple[Any, ...]) -> Any:
    if not groups:
        return None
    feature_name = groups[0].feature_name
    if feature_name in model._specs:
        return model._specs[feature_name]
    return model._interaction_specs.get(feature_name)


def _group_test_kind(model: SuperGLM, row, groups: tuple[Any, ...]) -> str:
    spec = _source_spec(model, groups)
    if isinstance(spec, OrderedCategorical) and spec.basis == "spline":
        return "smooth"
    if isinstance(spec, _SplineBase):
        return "group" if row.subgroup_type == "linear" else "smooth"
    if isinstance(spec, SplineCategorical | TensorInteraction):
        return "smooth"
    return "group"


def _significance(p_value: float | None, quasi_separated: bool) -> str:
    if quasi_separated:
        return "QS"
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return ""


def _term_rows(model: SuperGLM, source: _CompactSummarySource) -> tuple[SummaryTermRow, ...]:
    level_names = _canonical_level_row_names(model)
    selected_names = selected_group_name_set(
        model.result,
        model._groups,
        penalty=model.penalty,
    )
    terms: list[SummaryTermRow] = []
    for row in source.rows:
        source_groups = _source_groups_for_row(model, row)
        is_group_test = bool(row.is_spline)
        kind = (
            _group_test_kind(model, row, source_groups)
            if is_group_test
            else ("level" if row.name in level_names else "coefficient")
        )
        statistic = _finite_float(row.wald_chi2 if is_group_test else row.z)
        p_value = _finite_float(row.wald_p if is_group_test else row.p)
        quasi_separated = bool(row.quasi_separated)
        terms.append(
            SummaryTermRow(
                term=str(row.name),
                group=str(row.group or ""),
                kind=kind,
                estimate=None if is_group_test else _finite_float(row.coef),
                std_error=None if is_group_test else _finite_float(row.se),
                statistic=statistic,
                statistic_type=(
                    "chi2" if is_group_test else ("z" if statistic is not None else "")
                ),
                p_value=p_value,
                ci_lower=None if is_group_test else _finite_float(row.ci_low),
                ci_upper=None if is_group_test else _finite_float(row.ci_high),
                edf=_finite_float(row.edf),
                smoothing_lambda=_finite_float(row.smoothing_lambda),
                active=(
                    str(row.name) == "Intercept"
                    or any(group.name in selected_names for group in source_groups)
                ),
                significance=_significance(p_value, quasi_separated),
                warning="Quasi-separated" if quasi_separated else "",
            )
        )
    return tuple(terms)


def _summary_notes(
    source: _CompactSummarySource,
    terms: tuple[SummaryTermRow, ...],
) -> tuple[str, ...]:
    info = source.info
    notes: list[str] = []
    inference_stale = bool(info.get("editor_inference_stale", False))
    if inference_stale:
        edited_terms = ", ".join(info.get("editor_edited_terms") or [])
        suffix = f" Edited terms: {edited_terms}." if edited_terms else ""
        notes.append(_EDITOR_STALE_NOTE + suffix)
    if info.get("editor_offset_terms"):
        offset_terms = ", ".join(info.get("editor_offset_terms") or [])
        notes.append(f"{_EDITOR_OFFSET_NOTE} Offset terms: {offset_terms}.")
    if not inference_stale:
        if any(row.kind == "smooth" for row in terms):
            notes.append(_SMOOTH_WOOD_NOTE)
        if any(row.kind == "group" for row in terms):
            notes.append(_GROUP_WALD_NOTE)
        notes.append(_PARAMETRIC_WALD_NOTE)
    if any(row.warning for row in terms):
        notes.append(_QS_NOTE)
    return tuple(notes)


def build_summary_export_payload(model: SuperGLM) -> SummaryExportPayload:
    """Build a typed summary payload from a fitted model's compact summary."""
    source = _adapt_compact_summary(model.summary(detail="compact"))
    terms = _term_rows(model, source)
    return SummaryExportPayload(
        overview=_overview_rows(source),
        terms=terms,
        notes=_summary_notes(source, terms),
    )


__all__ = [
    "SummaryExportPayload",
    "SummaryOverviewRow",
    "SummaryTermRow",
    "build_summary_export_payload",
]
