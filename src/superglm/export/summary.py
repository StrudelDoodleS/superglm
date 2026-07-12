"""Renderer-independent model-summary export payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from superglm.features.categorical import Categorical
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.inference.summary import _EDITOR_STALE_NOTE, _QS_NOTE, _WALD_NOTE

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


_PARAMETRIC_WALD_NOTE = (
    "Parametric p-values are Wald approximations.\n"
    "For borderline significance, use a likelihood ratio test."
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


def _overview_rows(summary) -> tuple[SummaryOverviewRow, ...]:
    info = summary._info
    data = summary.to_dict()
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


def _term_rows(model: SuperGLM, summary) -> tuple[SummaryTermRow, ...]:
    level_names = _canonical_level_row_names(model)
    terms: list[SummaryTermRow] = []
    for row in summary._coef_rows:
        is_smooth = bool(row.is_spline)
        kind = "smooth" if is_smooth else ("level" if row.name in level_names else "coefficient")
        statistic = _finite_float(row.wald_chi2 if is_smooth else row.z)
        p_value = _finite_float(row.wald_p if is_smooth else row.p)
        quasi_separated = bool(row.quasi_separated)
        terms.append(
            SummaryTermRow(
                term=str(row.name),
                group=str(row.group or ""),
                kind=kind,
                estimate=None if is_smooth else _finite_float(row.coef),
                std_error=None if is_smooth else _finite_float(row.se),
                statistic=statistic,
                statistic_type="chi2" if is_smooth else ("z" if statistic is not None else ""),
                p_value=p_value,
                ci_lower=None if is_smooth else _finite_float(row.ci_low),
                ci_upper=None if is_smooth else _finite_float(row.ci_high),
                edf=_finite_float(row.edf),
                smoothing_lambda=_finite_float(row.smoothing_lambda),
                active=bool(row.active or (not is_smooth and row.coef is not None)),
                significance=_significance(p_value, quasi_separated),
                warning="Quasi-separated" if quasi_separated else "",
            )
        )
    return tuple(terms)


def _summary_notes(summary, terms: tuple[SummaryTermRow, ...]) -> tuple[str, ...]:
    info = summary._info
    notes: list[str] = []
    if info.get("editor_inference_stale", False):
        edited_terms = ", ".join(info.get("editor_edited_terms") or [])
        suffix = f" Edited terms: {edited_terms}." if edited_terms else ""
        notes.append(_EDITOR_STALE_NOTE + suffix)
    elif any(row.kind == "smooth" for row in terms):
        notes.append(_WALD_NOTE)
    else:
        notes.append(_PARAMETRIC_WALD_NOTE)
    if any(row.warning for row in terms):
        notes.append(_QS_NOTE)
    return tuple(notes)


def build_summary_export_payload(model: SuperGLM) -> SummaryExportPayload:
    """Build a typed summary payload from a fitted model's compact summary."""
    summary = model.summary(detail="compact")
    terms = _term_rows(model, summary)
    return SummaryExportPayload(
        overview=_overview_rows(summary),
        terms=terms,
        notes=_summary_notes(summary, terms),
    )


__all__ = [
    "SummaryExportPayload",
    "SummaryOverviewRow",
    "SummaryTermRow",
    "build_summary_export_payload",
]
