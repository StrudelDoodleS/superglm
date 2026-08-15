"""Renderer-independent model-summary export payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from superglm.features.categorical import Categorical
from superglm.features.interaction import SplineCategorical, TensorInteraction
from superglm.features.ordered_categorical import (
    _STEP_MODE_REMOVED_MESSAGE,
    OrderedCategorical,
)
from superglm.features.spline import _SplineBase
from superglm.inference._term_helpers import spline_groups

# The legend's wording is imported rather than restated: it was a second string
# literal here, and that is how it came to describe a rule neither module
# implements (issue #239).
from superglm.inference.summary import _LOW_CREDIBILITY_NOTE_BODY
from superglm.model.fit_state import fitted_penalty
from superglm.solvers.rank import selected_group_name_set

if TYPE_CHECKING:
    from superglm.inference.summary import _CoefRow
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
# The workbook's short cell values, one per trigger.  The two are deliberately
# distinct because this cell travels WITHOUT the note: it lands in a
# spreadsheet column that a downstream consumer reads on its own, so it has to
# be a true row-level statement rather than the union of two.  Calling an
# SE-flagged row "Low credibility" would assert a shortage of data about a row
# that may have none -- that branch's threshold is not scale-invariant, so a
# predictor in very small units trips it with the whole sample behind it and an
# unchanged z and p-value.
#
# WHICH one fired is read off the row's ``advisory_trigger``, recorded where
# the flag is set; ``_advisory_warning`` says why it is not re-derived here.
# An unrecognised value gets the neutral third value rather than either claim.
_THIN_LEVEL_WARNING = "Low credibility"
_OUTSIZED_SE_WARNING = "Outsized standard error"
_UNKNOWN_ADVISORY_WARNING = "Advisory"


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
            if spec.basis != "spline":
                # The dropped-base row set this used to build is step geometry.
                # The 0.24.0 removal took the coverage for it with the mode, so
                # leaving the arm live meant a restored step artifact silently
                # exported a wrong-shaped row set. Refuse with the migration
                # sentence, as every other surviving step path does.
                raise AttributeError(
                    f"Cannot canonicalise level row names for {feature_name!r}: "
                    f"{_STEP_MODE_REMOVED_MESSAGE}"
                )
            names.update(f"{feature_name}[{level}]" for level in spec._ordered_levels)
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
    if isinstance(spec, OrderedCategorical) and spec.basis_kind == "spline":
        # Per-term marker: the term's whole-smooth row records that the term
        # contains free levels.  Which ones is recorded on the level rows.
        return "smooth+free" if spec.has_specials else "smooth"
    if isinstance(spec, OrderedCategorical):
        # Piecewise/Polynomial inner basis: an unpenalized parametric block
        # under a plain Wald test -- "group", never "smooth", with the same
        # free-level marker when specials ride along.
        return "group+free" if spec.has_specials else "group"
    if isinstance(spec, _SplineBase):
        return "group" if row.subgroup_type == "linear" else "smooth"
    if isinstance(spec, SplineCategorical | TensorInteraction):
        return "smooth"
    return "group"


def _level_row_kind(row: _CoefRow) -> str:
    """Per-level provenance for a level row.

    The Summary sheet emits one row per coefficient row, so an
    OrderedCategorical special already has its own row here; it only needs a
    kind that says so.  ``level_fit`` is set by both level-row builders
    (``coef_tables.build_coef_rows`` and
    ``report_ops._build_editor_stale_coef_rows``), so the edited-model path
    carries the marker too.

    Read the field directly: every row here is a ``_CoefRow``, so a defaulting
    ``getattr`` would only turn a rename or a wrong row type into every special
    silently reverting to ``"level"`` in the exported workbook.
    """
    return "free level" if row.level_fit == "free" else "level"


def _significance(p_value: float | None) -> str:
    """The p-value's own code, and nothing else.

    The low-credibility advisory used to be returned from here instead of the
    stars, which made the exported workbook say that a flagged row had no
    significance to report.  It is not a verdict on the estimate -- the rule
    behind it reads a level's observation count and exposure share, or a
    standard error's size against the model's typical one, and never the
    p-value -- so a level can legitimately be both ``***`` and flagged.  The
    console has always shown them as independent columns; the payload now
    does the same, carrying the advisory in ``warning`` where it already had a
    column of its own (issue #239).
    """
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


_ADVISORY_WARNINGS = {"thin_level": _THIN_LEVEL_WARNING, "outsized_se": _OUTSIZED_SE_WARNING}


def _advisory_warning(row: _CoefRow, quasi_separated: bool) -> str:
    """Name the trigger that fired, not the union of the two.

    Read off ``advisory_trigger``, which is recorded where the flag is set,
    rather than re-derived from ``level_n_obs`` here: the level display strips
    those diagnostics from all but the first member of a grouped level, so a
    renderer that re-derives disagrees with this one about the same row.
    """
    if not quasi_separated:
        return ""
    return _ADVISORY_WARNINGS.get(row.advisory_trigger, _UNKNOWN_ADVISORY_WARNING)


def _term_rows(model: SuperGLM, source: _CompactSummarySource) -> tuple[SummaryTermRow, ...]:
    level_names = _canonical_level_row_names(model)
    selected_names = selected_group_name_set(
        model.result,
        model._groups,
        penalty=fitted_penalty(model),
    )
    terms: list[SummaryTermRow] = []
    for row in source.rows:
        source_groups = _source_groups_for_row(model, row)
        is_group_test = bool(row.is_spline)
        is_structured_metadata = row.structured_kind is not None
        is_group_row = is_group_test or is_structured_metadata
        kind = (
            _group_test_kind(model, row, source_groups)
            if is_group_row
            else (_level_row_kind(row) if row.name in level_names else "coefficient")
        )
        statistic = _finite_float(
            row.wald_chi2 if is_group_test else (None if is_structured_metadata else row.z)
        )
        p_value = _finite_float(
            row.wald_p if is_group_test else (None if is_structured_metadata else row.p)
        )
        quasi_separated = bool(row.quasi_separated)
        # An OrderedCategorical specials term contributes two source groups: the
        # penalized spline block and an unpenalized special block that selection
        # can never drop. Asking whether ANY of them survived would let the
        # special keep the rest of the term alive:
        #
        #   - the "smooth+free" row describes the SPLINE block (every statistic on
        #     it was scoped there), so it would report active while showing an
        #     empty smooth, contradicting model.summary();
        #   - a SMOOTHED level row is no longer fitted once the spline is dropped,
        #     so it would inherit activity from a block it has no coefficient in.
        #
        # Only a level actually fitted free may take its activity from the special
        # block -- that one really is still estimated when the curve is gone.
        #
        # Through ``spline_groups`` rather than a local
        # ``getattr(group, "subgroup_type", None) != "special"``: the defaulting
        # form fails OPEN under a rename (``None != "special"`` keeps every
        # group) and the filter would silently become a no-op here, which is the
        # regression this comment describes.
        inherits_from_special = kind == "free level"
        active_groups = source_groups
        if not inherits_from_special:
            active_groups = tuple(spline_groups(source_groups))
        terms.append(
            SummaryTermRow(
                term=str(row.name),
                group=str(row.group or ""),
                kind=kind,
                estimate=None if is_group_row else _finite_float(row.coef),
                std_error=None if is_group_row else _finite_float(row.se),
                statistic=statistic,
                statistic_type=(
                    "chi2" if is_group_test else ("z" if statistic is not None else "")
                ),
                p_value=p_value,
                ci_lower=None if is_group_row else _finite_float(row.ci_low),
                ci_upper=None if is_group_row else _finite_float(row.ci_high),
                edf=_finite_float(row.edf),
                smoothing_lambda=_finite_float(row.smoothing_lambda),
                active=(
                    str(row.name) == "Intercept"
                    or any(group.name in selected_names for group in active_groups)
                ),
                significance=_significance(p_value),
                warning=_advisory_warning(row, quasi_separated),
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
        if any(row.kind.startswith("smooth") for row in terms):
            notes.append(_SMOOTH_WOOD_NOTE)
        if any(row.kind == "group" and row.statistic_type == "chi2" for row in terms):
            notes.append(_GROUP_WALD_NOTE)
        notes.append(_PARAMETRIC_WALD_NOTE)
    if any(row.warning for row in terms):
        notes.append(_LOW_CREDIBILITY_NOTE_BODY)
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
