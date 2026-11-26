"""Formatted model summary with ASCII and HTML output (statsmodels-style)."""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html import escape as html_escape
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np

if TYPE_CHECKING:
    from superglm.inference.summary_levels import SummaryLevelDisplay


def _scale_cell(info, fmt) -> str:
    """``phi`` with the estimator standard errors read it from."""
    from superglm.solvers.dispersion import KNOWN, SCALE_METHOD_LABELS

    method = info.get("phi_method")
    if method is None or method == KNOWN:
        return fmt(info["phi"])
    return f"{fmt(info['phi'])} ({SCALE_METHOD_LABELS.get(method, method)})"


def _likelihood_scale_rows(info, fmt) -> list[tuple[str, str, str, str]]:
    """The scale the log-likelihood and AIC read, where it is a separate estimate."""
    from superglm.solvers.dispersion import KNOWN, LEGACY, SCALE_METHOD_LABELS

    method = info.get("phi_likelihood_method")
    if method in (None, KNOWN, LEGACY):
        # A saved model's one scale serves both readers; ``Scale (phi)`` already shows it.
        return []
    label = SCALE_METHOD_LABELS.get(method, method)
    return [("Likelihood phi", f"{fmt(info['phi_likelihood'])} ({label})", "", "")]


def _robust_covariance_header_rows(info, fmt) -> list[tuple[str, str, str, str]]:
    """Header row displaying robust sandwich covariance method and sample/cluster count."""
    method = info.get("covariance_method")
    if not method:
        return []
    if info.get("covariance_clustered") and info.get("covariance_n_clusters") is not None:
        return [("Covariance", str(method), "No. Clusters", str(info["covariance_n_clusters"]))]
    n_units = info.get("covariance_n_units")
    units_text = (
        ""
        if n_units is None
        else (str(int(n_units)) if float(n_units).is_integer() else fmt(float(n_units)))
    )
    return [("Covariance", str(method), "Cov. Units", units_text)]


def _robust_covariance_note(info: dict[str, Any], *, has_smooth: bool) -> str:
    """Footnote describing robust sandwich covariance assumptions and omitted tests."""
    method = str(info["covariance_method"])
    parts = [
        f"Note: standard errors use robust {method} covariance at fixed basis, "
        "penalty, and smoothing parameters."
    ]
    if has_smooth:
        parts.append("Smooth p-values (Wood 2013) are Bayesian and omitted from robust output.")
    if info.get("covariance_clustered"):
        parts.append(
            "Parametric tests and intervals use a large-sample normal Wald reference; "
            "few or dominant clusters can make cluster-robust Wald tests anti-conservative."
        )
    else:
        parts.append("Parametric tests and intervals use a large-sample normal Wald reference.")
    return "\n".join(parts)


@dataclass
class _CoefRow:
    """One row of the coefficient table."""

    name: str
    group: str = ""  # feature group name for separator logic
    coef: float | None = None
    se: float | None = None
    z: float | None = None
    p: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    estimable: bool = True
    # Structured multi-column term summary (RE / factor smooth).
    structured_kind: str | None = None
    n_levels: int | None = None
    smoothing_lambdas: tuple[tuple[str, float], ...] = ()
    # Spline summary row (group-level Wald test)
    is_spline: bool = False
    n_params: int = 0
    active: bool = False
    group_norm: float = 0.0
    wald_chi2: float | None = None
    wald_p: float | None = None
    ref_df: float | None = None
    curve_se_min: float | None = None
    curve_se_max: float | None = None
    subgroup_type: str | None = None  # "linear", "spline", or None
    # Enriched spline metadata
    edf: float | None = None
    smoothing_lambda: float | None = None
    spline_kind: str | None = None  # "PSpline", "NaturalSpline", etc.
    knot_strategy: str | None = None
    boundary: tuple[float, float] | None = None
    difference_penalty: str | None = None  # "standard", "general" or "projected"
    # Monotonicity
    monotone: str | None = None  # the term's rules as written, e.g. "increasing & concave", or None
    monotone_engine: str | None = None  # "qp" or "scop"
    monotone_repaired: bool = False
    # Low-credibility advisory: a thin categorical level, or a standard error far above the
    # model's typical one. The name predates issue #239 (renamed under issue #304); renderers no
    # longer report it as a diagnosis.
    quasi_separated: bool = False
    # Which trigger fired, recorded beside the flag: the level display strips diagnostics from
    # all but the first member of a grouped level, so a renderer re-deriving it from a display
    # row would call a pooled thin level an outsized SE.
    advisory_trigger: str = ""
    level_n_obs: int | None = None
    level_exposure_share: float | None = None
    # A level or crossed cell whose effect moves along a separating direction
    # (``separated_levels.separated_coefficients``): its estimate is where the fit
    # stopped, and its Wald standard error, z, p and interval are left blank.
    separated: bool = False
    # Per-level fit provenance: "smooth" for a level carried by the spline,
    # "free" for an OrderedCategorical special, None when the term has no
    # specials. Drives the optional `fit` column in both renderers.
    level_fit: str | None = None
    # Summary presentation only. Canonical coefficient builders leave these
    # fields at their defaults.
    level_group: str = ""
    is_reference: bool = False


@dataclass
class _BasisDetailRow:
    """One per-coefficient detail row for a spline group."""

    parent_name: str  # group name, e.g. "x" or "x:spline"
    basis_index: int  # 0-based within group
    coef: float
    se: float
    z: float | None
    p: float | None
    ci_low: float | None
    ci_high: float | None


def _compute_coef_stats(
    coef: float,
    se: float,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Compute z-value, p-value, and confidence interval."""
    if se <= 0:
        return np.nan, np.nan, np.nan, np.nan
    import scipy.spatial
    from scipy.stats import norm

    z = coef / se
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    q = norm.ppf(1.0 - alpha / 2.0)
    return z, p, coef - q * se, coef + q * se


def _display_method(method: Any) -> str:
    """Return the presentation label for the fitting method."""
    method_str = str(method)
    return "MLE" if method_str == "ML" else method_str


_MARKED_CI_STATUSES = ("censored", "caution", "censored with caution")


def _format_profile_estimate(
    estimate: Any,
    ci: Any,
    ci_status: Any,
) -> str:
    """Format a profile estimate without assuming an interval was computed."""
    if isinstance(ci, tuple | list) and len(ci) >= 2:
        # A censored side is where the search stopped, not a likelihood-ratio
        # crossing; a caution says the interval is not about the estimate as it
        # stands (``warnings`` says why).
        censored = f" {ci_status}" if ci_status in _MARKED_CI_STATUSES else ""
        try:
            return f"{float(estimate):.3f} [{float(ci[0]):.3f}, {float(ci[1]):.3f}]{censored}"
        except (TypeError, ValueError, OverflowError):
            pass
    return f"{float(estimate):.3f} [CI not computed]"


def _profile_nll_cell(info: dict[str, Any], key: str) -> tuple[str, str]:
    """The right-hand header cell of a profile row: its NLL, when the summary carries one."""
    return ("Profile NLL", f"{info[key]:.4f}") if key in info else ("", "")


#: The engine tag of a rule line whose rules a ``fit_path`` fit did not enforce.
NOT_ENFORCED_ON_PATH = "not enforced: fit_path"


def unenforced_rules_tag(rules: Any) -> str:
    """The engine tag for ``rules`` a ``fit_path`` fit left unenforced.

    A flat stretch is part of the term's basis and still holds on the path; the
    tag says so when the rules carry one.
    """
    flat = any(rule.kind == "flat" for rule in rules)
    return NOT_ENFORCED_ON_PATH + ("; a flat stretch still holds" if flat else "")


def _rules_text(row: _CoefRow) -> str:
    """A term's rule line: its rules as written, the engine, and a repair marker.

    The term test, ref_df and curve SE band are withheld for a repaired term rather
    than printed against the unconstrained fit's reference distribution; the marker
    says so, or the row reads as a term that simply had no test.
    """
    text = f"mono={row.monotone}"
    if row.monotone_engine is not None:
        text += f" ({row.monotone_engine})"
    if row.monotone_repaired:
        text += ", repaired (inference withheld)"
    return text


def _level_rule_notes(rows: list[_CoefRow]) -> dict[str, str]:
    """The rule line of each term shown as level rows, which have no spline row to carry it."""
    notes: dict[str, str] = {}
    for row in rows:
        if row.monotone is not None and not row.is_spline:
            notes.setdefault(row.group, _rules_text(row))
    return notes


_SIG_LEGEND = "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
# The column tag and the note that defines it, shared with ``superglm.export.summary``.
# The note states what the rule establishes. The first branch is a volume test, a
# limited-fluctuation credibility question (Mowbray, *PCAS* 1, 1914; ASOP No. 25): the
# sparse-cell estimate is finite but unstable and biased away from the null (Firth,
# *Biometrika* 80(1), 1993). The second is an SE screen with threshold
# ``max(50 * median_se, 10.0)``: it is not scale-invariant (rescaling one predictor trips it
# on units alone) and the absolute 10.0 binds whenever the median SE is under 0.2, so it is a
# screen, not a ranking (splitting the flags is issue #304). Neither branch looks at the
# response, so neither can establish separation (Albert and Anderson, *Biometrika* 71(1),
# 1984; Konis, DPhil thesis, Oxford, 2007), which is defined only where the response has a
# boundary the predictor can approach.
_LOW_CREDIBILITY_TAG = "LC"
#: A level is thin when it holds fewer rows than this and less weight than this
#: many rows of average weight (``coef_tables._thin_levels``).
_THIN_LEVEL_ROWS = 20
# Split so the workbook can carry the note without the column tag: ``LC`` is a
# column on the console and in the editor and does not exist in the workbook,
# whose advisory lives in a Warning cell.
_LOW_CREDIBILITY_NOTE_BODY = (
    "Low credibility — flagged when a categorical level carries fewer than\n"
    f"{_THIN_LEVEL_ROWS} rows and less weight than {_THIN_LEVEL_ROWS} rows of average weight (under\n"
    "frequency weights a row counts once per replication; otherwise only a row\n"
    "of positive weight counts), or, for a row that has no per-level counts,\n"
    "when its standard error is both far above this model's typical one and\n"
    "above a fixed floor. The estimate, interval and p-value\n"
    "beside it are unchanged and are the ones the fit produced. A flagged LEVEL\n"
    "is thin: treat it as partially credible and pool it, or blend it toward the\n"
    "portfolio, before pricing on it. A flagged COEFFICIENT is wide in absolute\n"
    "terms — a direction the data does not identify does that, and so does a\n"
    "predictor on a much smaller scale than the rest, so check its units before\n"
    "reading it as a shortage of data. The floor makes this column a screen\n"
    "rather than a ranking: a coefficient can be many times the typical width\n"
    "and go unflagged."
)
_LOW_CREDIBILITY_NOTE = (
    "LC: " + _LOW_CREDIBILITY_NOTE_BODY[0].lower() + _LOW_CREDIBILITY_NOTE_BODY[1:]
)
# The advisory column's tag for a separated level, and what it means; the workbook
# carries the note's body beside its own Warning cell.
_SEPARATED_TAG = "SEP"
_SEPARATED_NOTE_BODY = (
    "Separated — every response on this level's, or cell's, rows is 0, or every one\n"
    "is 1, so the data give its effect no finite estimate. The estimate shown is where\n"
    "the fit stopped, and its standard error, z, p-value and interval are left blank:\n"
    "a Wald statistic has no finite reference there (Hauck and Donner, JASA 72, 1977).\n"
    "A coefficient the other rows do not determine moves with it and is flagged too:\n"
    "every level against a separated base, the intercept, a parent's main effect\n"
    "beside a separated cell. Group the level with similar levels, or fit the term\n"
    "as a RandomEffect, to price it."
)
_SEPARATED_NOTE = f"{_SEPARATED_TAG}: " + _SEPARATED_NOTE_BODY[0].lower() + _SEPARATED_NOTE_BODY[1:]
_PARAMETRIC_WALD_NOTE = (
    "Parametric p-values are Wald approximations.\n"
    "For borderline significance, use a likelihood ratio test."
)
_WALD_NOTE = "Note: smooth p-values use Wood (2013) Bayesian test.\n" + _PARAMETRIC_WALD_NOTE
_EDITOR_STALE_NOTE = (
    "Editor edits applied: coefficient standard errors, confidence intervals, "
    "and p-values are suppressed because they belong to the original fitted "
    "model, not the manually edited coefficients."
)


# Group-test rows whose test is a plain Wald chi-square on an unpenalized
# block, not Wood's smooth test: the numeric Piecewise whole-term row, the two
# ordered parametric whole-term rows, and a curvature family inside one.
_PARAMETRIC_GROUP_TEST_TYPES = frozenset(
    {"piecewise", "ordered_piecewise", "ordered_polynomial", "curvature"}
)

# Display label per group-test subgroup type; anything unrecognised renders as
# the historical "spline".
_GROUP_TEST_KIND_LABELS = {
    "linear": "linear",
    "ordered_spline": "ordered spline",
    "piecewise": "piecewise",
    "ordered_piecewise": "ordered piecewise",
    "ordered_polynomial": "ordered polynomial",
    "curvature": "curvature",
}


def _is_smooth_group_row(row: _CoefRow) -> bool:
    """Whether a group row's df is smooth df and its p-value a Wood (2013) test.

    ``is_spline`` is the flag that routes a row through the group-test renderer
    -- the ``[kind, n params, chi2(df)]`` line -- and setting it is not a claim
    that anything was smoothed.  A ``Piecewise`` whole-term row uses that
    renderer but has no penalty, no estimated smoothing parameter and a df fixed
    at ``J + 1``; its test is a plain Wald chi-square.  Counting it as smooth put
    its parametric df in the header's smooth bucket and printed the Wood
    footnote over a model containing no smooth at all.  The ordered
    piecewise/polynomial whole-term rows and curvature family rows are the same
    shape: unpenalized blocks under plain Wald tests.
    """
    return bool(row.is_spline) and row.subgroup_type not in _PARAMETRIC_GROUP_TEST_TYPES


def _low_credibility_label(row: _CoefRow) -> str:
    """Label fitted-group diagnostics without attributing them to one member."""
    if row.level_group:
        return f"{row.group or row.name} {row.level_group}"
    return row.name


_EDITOR_OFFSET_NOTE = (
    "Editor offset refit: listed editor terms are fixed offset factors. "
    "Inference is conditional on those fixed offsets."
)


def _sig_stars(p: float | None) -> str:
    """R-style significance stars for a p-value."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def _format_ascii_number(value: float | None, *, decimals: int, width: int) -> str:
    """Format one finite table value without exceeding its declared column width."""
    if value is None or not np.isfinite(value):
        return "---"
    numeric = float(value)

    def _is_safe(candidate: str) -> bool:
        parsed = float(candidate)
        return np.isfinite(parsed) and (numeric == 0.0 or parsed != 0.0)

    # Shed fixed decimals before switching notation: a width-limited scientific field keeps
    # only ``width - 5`` significant figures and would collapse a monetary-scale interval to
    # ``2.500e+05 2.500e+05``.
    for fixed_decimals in range(decimals, -1, -1):
        fixed = f"{numeric:.{fixed_decimals}f}"
        if len(fixed) <= width and _is_safe(fixed):
            return fixed
    for scientific_decimals in range(decimals, -1, -1):
        scientific = f"{numeric:.{scientific_decimals}e}"
        if len(scientific) <= width and _is_safe(scientific):
            return scientific
    # Rounding the largest finite float to a width-limited mantissa can produce
    # ``1.8e+308``, which parses as infinity. A one-digit, toward-zero order of
    # magnitude remains finite and bounded while preserving sign and scale.
    exponent = int(np.floor(np.log10(abs(numeric))))
    fallback = f"{'-' if numeric < 0.0 else ''}1e{exponent:+d}"
    if len(fallback) <= width and _is_safe(fallback):
        return fallback
    # Every coefficient column is wide enough for the shortest finite float64
    # scientific form (including sign and a three-digit exponent).
    raise ValueError(f"{numeric!r} cannot be represented in an ASCII field of width {width}")


#: The coefficient columns' minimum widths: six numbers, then Sig and the advisory tag.
_COEF_FIELD_WIDTHS = (10, 10, 8, 8, 9, 9, 3, 3)
_STRUCTURED_KIND_LABELS = {
    "random_effect": "random effect",
    "factor_smooth_fs": "factor smooth (fs)",
    "factor_smooth_sz": "factor smooth (sz)",
}
_LOW_CREDIBILITY_HEADING = "? Low-credibility levels, and the experience behind each:"


@dataclass(frozen=True)
class _Markup:
    """How a renderer spells what the text and HTML summaries share."""

    escape: Callable[[str], str]
    lam: str
    dash: str
    less: str
    chi2: str


_TEXT = _Markup(escape=str, lam="lam", dash="-", less="<", chi2="chi2")
_HTML = _Markup(escape=html_escape, lam="&lambda;", dash="&ndash;", less="&lt;", chi2="&chi;&sup2;")


class _GroupText(NamedTuple):
    """A group-test row in one renderer's markup."""

    kind: str
    text: str
    has_test: bool
    stars: str
    detail: str


def _row_kind(row: _CoefRow) -> str:
    """Which of the table's row shapes a row takes."""
    if row.structured_kind is not None:
        return "structured"
    if row.is_spline:
        return "group"
    if row.is_reference:
        return "reference"
    if (
        row.coef is not None
        and row.se is not None
        and (row.se > 0 or (row.p is None and row.ci_low is not None and row.ci_high is not None))
    ):
        return "estimated"
    return "blank"


def _structured_text(row: _CoefRow, markup: _Markup) -> str:
    """A structured term's one-line description; its levels have a dedicated report."""
    structured_kind = cast(str, row.structured_kind)
    kind = _STRUCTURED_KIND_LABELS.get(structured_kind, structured_kind.replace("_", " "))
    detail = [f"{row.n_params} params"]
    if row.n_levels is not None:
        detail.insert(0, f"{row.n_levels} levels")
    if row.edf is not None:
        detail.append(f"edf={row.edf:.1f}")
    detail.extend(f"{markup.escape(name)}={value:.2g}" for name, value in row.smoothing_lambdas)
    return f"[{markup.escape(kind)}, {', '.join(detail)}; use dedicated term report]"


def _group_text(row: _CoefRow, markup: _Markup) -> _GroupText:
    """A group-test row's bracketed test and its detail line (edf, lambda, curve SE, rules)."""
    has_test = bool(
        row.active
        and row.wald_chi2 is not None
        and np.isfinite(row.wald_chi2)
        and row.wald_p is not None
        and np.isfinite(row.wald_p)
    )
    kind = _GROUP_TEST_KIND_LABELS.get(row.subgroup_type or "", "spline")
    params = f"{row.n_params} params"
    if not row.active:
        return _GroupText(kind, f"[{kind}, {params}, inactive]", False, "", "")
    detail = [f"rank={row.n_params}"]
    if row.edf is not None:
        detail.append(f"edf={row.edf:.1f}")
    if row.smoothing_lambda is not None:
        detail.append(f"{markup.lam}={row.smoothing_lambda:.2g}")
    if row.curve_se_min is not None and not np.isnan(row.curve_se_min):
        detail.append(f"curve SE: {row.curve_se_min:.2f}{markup.dash}{row.curve_se_max:.2f}")
    if row.monotone is not None:
        detail.append(markup.escape(_rules_text(row)))
    if not has_test:
        return _GroupText(kind, f"[{kind}, {params}, active]", False, "", ", ".join(detail))
    assert row.wald_p is not None
    p_text = f"{row.wald_p:.3f}" if row.wald_p >= 0.001 else f"{markup.less}0.001"
    df_text = f"{row.ref_df:.1f}" if row.ref_df is not None else str(row.n_params)
    text = f"[{kind}, {params}, {markup.chi2}({df_text})={row.wald_chi2:.1f}, p={p_text}]"
    return _GroupText(kind, text, True, _sig_stars(row.wald_p), ", ".join(detail))


def _ascii_numbers(row: _CoefRow | _BasisDetailRow) -> tuple[str, ...]:
    """The six numeric fields of a coefficient or basis-detail row, each within its width."""
    z_decimals = 1 if row.z is not None and np.isfinite(row.z) and abs(row.z) >= 100 else 3
    values = (row.coef, row.se, row.z, row.p, row.ci_low, row.ci_high)
    decimals = (4, 4, z_decimals, 3, 3, 3)
    return tuple(
        _format_ascii_number(value, decimals=places, width=width)
        for value, places, width in zip(values, decimals, _COEF_FIELD_WIDTHS[:6], strict=True)
    )


def _html_number(value: float | None, decimals: int = 3) -> str:
    """A finite value at ``decimals`` places, else ``---``."""
    return f"{value:.{decimals}f}" if value is not None and np.isfinite(value) else "---"


def _plain(text: str) -> str:
    return text


def _with_breaks(text: str) -> str:
    return text.replace("\n", "<br>")


def _escaped_with_breaks(text: str) -> str:
    return html_escape(text).replace("\n", "<br>")


# Each footer note's style: the text summary wraps the long free-text ones; the HTML
# summary gives each its colour and escaping.  HTML shows no penalty abbreviations.
_TEXT_WRAPPED_NOTES = frozenset({"warning", "shared_cell"})
_HTML_NOTES: dict[str, tuple[str, Callable[[str], str]] | None] = {
    "legend": ("color:#666;border:none;", _plain),
    "low_credibility": ("color:#c60;border:none;", _with_breaks),
    "separated": ("color:#c60;border:none;", _escaped_with_breaks),
    "abbrevs": None,
    "warning": ("color:#c60;white-space:normal;border:none;", html_escape),
    "editor": ("color:#8a4b00;font-style:italic;border:none;", _plain),
    "shared_cell": ("white-space:normal;border:none;", html_escape),
    "method": ("color:#888;font-style:italic;border:none;", _with_breaks),
}


class ModelSummary:
    """Formatted model summary with ASCII and HTML output.

    Returned by ``model.summary()`` and ``ModelMetrics.summary()``. Supports:

    - ``print(summary)`` — ASCII table for terminals
    - Jupyter ``_repr_html_`` — HTML table for notebooks
    - ``summary['fit']`` — dict access (backward compat)
    - ``'fit' in summary`` — membership test (backward compat)
    - ``summary.to_dict()`` — full dict

    Both renderers read one layout: the header rows (``_header_rows``), each row's shape
    (``_row_kind``) and text, and the footer notes (``_notes``); each adds only markup.
    """

    def __init__(
        self,
        data: dict[str, Any],
        model_info: dict[str, Any],
        coef_rows: list[_CoefRow],
        alpha: float = 0.05,
        detail: str = "compact",
        basis_detail: dict[str, list[_BasisDetailRow]] | None = None,
        level_presentation: SummaryLevelDisplay | None = None,
    ):
        _VALID_DETAIL = {"compact", "full"}
        if detail not in _VALID_DETAIL:
            raise ValueError(
                f"detail={detail!r} is not valid. Expected one of {sorted(_VALID_DETAIL)}."
            )
        self._data = data
        self._info = model_info
        self._coef_rows = coef_rows
        self._display_rows = (
            list(level_presentation.rows) if level_presentation is not None else list(coef_rows)
        )
        self._level_display = (
            level_presentation.level_display if level_presentation is not None else "expanded"
        )
        self._level_groups = (
            level_presentation.level_groups if level_presentation is not None else ()
        )
        self._alpha = alpha
        self._detail = detail
        self._basis_detail: dict[str, list[_BasisDetailRow]] = basis_detail or {}

    # ── Backward-compat dict interface ────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return the raw summary dict (backward compatibility)."""
        return self._data

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def items(self):
        return self._data.items()

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _fmt_scalar(v: Any) -> str:
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if abs(v) >= 1000:
                return f"{v:.1f}"
            if abs(v) >= 10:
                return f"{v:.2f}"
            if abs(v) >= 1:
                return f"{v:.3f}"
            return f"{v:.4f}"
        return str(v)

    # ── The layout both renderers read ────────────────────────────

    def _header_rows(self, *, suffix_on_parametric: bool) -> list[tuple[str, str, str, str]]:
        """The header's (label, value, label, value) rows.

        The effective-df cell names its smooth share when some EDF is smooth; the text
        summary also does when the EDF is all parametric (``suffix_on_parametric``), the
        HTML summary does not.
        """
        info, fmt = self._info, self._fmt_scalar
        smooth_edf = sum(
            r.edf
            for r in self._coef_rows
            if (_is_smooth_group_row(r) or r.structured_kind is not None) and r.edf is not None
        )
        parametric_edf = sum(
            r.edf
            for r in self._coef_rows
            if not _is_smooth_group_row(r) and r.structured_kind is None and r.edf is not None
        )
        edf_text = fmt(info["effective_df"])
        if smooth_edf > 0 or (suffix_on_parametric and parametric_edf > 0):
            edf_text = f"{fmt(info['effective_df'])} ({fmt(smooth_edf)} smooth)"
        rows = [
            ("Family", info["family"], "No. Observations", str(info["n_obs"])),
            ("Link", info["link"], "Df (effective)", edf_text),
            ("Method", _display_method(info.get("method", "ML")), "Penalty", info["penalty"]),
            (
                "Scale (phi)",
                _scale_cell(info, fmt),
                "Pearson chi2",
                fmt(info.get("pearson_chi2", "")),
            ),
            ("Log-Likelihood", fmt(info["log_likelihood"]), "AIC", fmt(info["aic"])),
            ("AICc", fmt(info["aicc"]), "BIC", fmt(info["bic"])),
            (
                "EBIC",
                fmt(info["ebic"]),
                "Converged",
                f"{info['converged']} ({info['n_iter']} iter)",
            ),
            ("Deviance", fmt(info["deviance"]), "", ""),
        ]
        if info.get("weight_semantics"):
            rows.append(("Weights", f"{info['weight_semantics']} weights", "", ""))
        rows.extend(_robust_covariance_header_rows(info, fmt))
        rows.extend(_likelihood_scale_rows(info, fmt))
        if "nb_theta" in info:
            theta = _format_profile_estimate(
                info["nb_theta"], info["nb_theta_ci"], info["nb_theta_ci_status"]
            )
            rows.append(("Theta", theta, *_profile_nll_cell(info, "nb_profile_nll")))
        if "tweedie_p" in info:
            p_text = _format_profile_estimate(
                info["tweedie_p"], info.get("tweedie_p_ci"), info.get("tweedie_p_ci_status")
            )
            rows.append(("Tweedie p", p_text, *_profile_nll_cell(info, "tweedie_profile_nll")))
        return rows

    def _column_names(self) -> tuple[str, ...]:
        half = self._alpha / 2.0
        return ("coef", "std err", "z", "P>|z|", f"[{half:.3f}", f"{1 - half:.3f}]", "Sig")

    def _level_group_legends(self) -> dict[str, list[Any]]:
        """Each grouped feature's fitted level groups, when levels are shown grouped."""
        legends: dict[str, list[Any]] = {}
        if self._level_display == "grouped":
            for item in self._level_groups:
                legends.setdefault(item.feature, []).append(item)
        return legends

    def _notes(self) -> list[tuple[str, str]]:
        """The footer notes, each with its style, in the order both renderers print them."""
        info, rows = self._info, self._display_rows
        notes = [("legend", _SIG_LEGEND)]
        if any(r.quasi_separated for r in rows):
            notes.append(("low_credibility", _LOW_CREDIBILITY_NOTE))
        if any(r.separated for r in rows):
            notes.append(("separated", _SEPARATED_NOTE))
        abbrevs = info.get("penalty_abbrevs", {})
        if abbrevs:
            notes.append(("abbrevs", "; ".join(f"{k}: {v}" for k, v in abbrevs.items())))
        for note in (info.get("convergence_note"), info.get("standard_error_note")):
            if note:
                notes.append(("warning", note))
        notes.extend(("editor", note) for note in _editor_notes(info))
        notes.extend(("shared_cell", note) for note in info.get("shared_cell_rows") or ())
        if not info.get("editor_inference_stale", False):
            notes.append(("method", self._method_note()))
        return notes

    def _method_note(self) -> str:
        """What the p-values are: robust, Wood's test beside Wald, or Wald alone."""
        has_smooth = any(_is_smooth_group_row(r) for r in self._coef_rows)
        if self._info.get("covariance_method"):
            return _robust_covariance_note(self._info, has_smooth=has_smooth)
        return _WALD_NOTE if has_smooth else _PARAMETRIC_WALD_NOTE

    def _low_credibility_levels(self) -> list[tuple[str, int, float]]:
        """Each flagged level's label, row count and exposure percentage."""
        return [
            (
                _low_credibility_label(r),
                r.level_n_obs,
                r.level_exposure_share * 100 if r.level_exposure_share is not None else 0,
            )
            for r in self._display_rows
            if r.quasi_separated and r.level_n_obs is not None
        ]

    # ── ASCII output ──────────────────────────────────────────────

    def __str__(self) -> str:
        header = self._header_rows(suffix_on_parametric=True)
        box = _TextBox(self, header)
        lines = [box.top(" SuperGLM Results ")]
        lines.extend(box.header_row(*cells) for cells in header)
        lines.append(box.mid())
        columns = box.prefix("Term", "Level group", "Fit")
        names = self._column_names()
        lines.append(box.row(columns + box.fields(names[:6], names[6], _LOW_CREDIBILITY_TAG)))
        lines.append(box.thin())
        lines.extend(self._text_body(box))
        lines.append(box.bot())
        lines.extend(self._text_footer(box.W))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def _text_body(self, box: _TextBox) -> list[str]:
        """The coefficient rows, under a separator wherever the group changes."""
        lines: list[str] = []
        level_rules = _level_rule_notes(self._display_rows)
        prev_group = None
        for row in self._display_rows:
            if row.group and row.group != prev_group:
                if prev_group is not None:
                    lines.append(box.row(""))
                lines.extend((box.group_sep(row.group), box.row("")))
                if row.group in level_rules:
                    lines.append(box.row(f"  {level_rules[row.group]}"))
            prev_group = row.group
            lines.extend(self._text_row(box, row))
        return lines

    def _text_row(self, box: _TextBox, row: _CoefRow) -> list[str]:
        prefix = box.prefix(row.name, row.level_group, row.level_fit or "")
        kind = _row_kind(row)
        if kind == "structured":
            return [box.row(f"{prefix}  {_structured_text(row, _TEXT)}")]
        if kind == "group":
            return self._text_group_row(box, row, prefix)
        if kind == "reference":
            return [box.row(prefix + box.fields(("0.0000", "ref", "---", "---", "---", "---")))]
        if kind == "estimated":
            advisory = "?" if row.quasi_separated else ""
            return [box.row(prefix + box.fields(_ascii_numbers(row), _sig_stars(row.p), advisory))]
        coef = _format_ascii_number(row.coef, decimals=4, width=_COEF_FIELD_WIDTHS[0])
        advisory = _SEPARATED_TAG if row.separated else ""
        return [box.row(prefix + box.fields((coef, *("---",) * 5), advisory=advisory))]

    def _text_group_row(self, box: _TextBox, row: _CoefRow, prefix: str) -> list[str]:
        group = _group_text(row, _TEXT)
        if group.has_test:
            lines = [box.row(box.sig_aligned(prefix, group))]
        else:
            lines = [box.row(f"{prefix}  {group.text}")]
        if group.detail:
            lines.append(box.row(f"{box.prefix('')}    {group.detail}"))
        if self._detail == "full":
            for br in self._basis_detail.get(row.name, ()):
                label = box.prefix(f"  Coef {br.basis_index + 1}")
                lines.append(box.row(label + box.fields(_ascii_numbers(br), _sig_stars(br.p))))
        return lines

    def _text_footer(self, width: int) -> list[str]:
        wrap_width = max(60, min(width + 2, 100))
        lines: list[str] = []
        for feature, items in self._level_group_legends().items():
            mapping = "; ".join(f"{item.group_id} = {', '.join(item.members)}" for item in items)
            legend = f"Level groups ({feature}): {mapping}"
            lines.extend(textwrap.wrap(legend, width=wrap_width, subsequent_indent="  "))
        for style, note in self._notes():
            if style in _TEXT_WRAPPED_NOTES:
                lines.extend(textwrap.wrap(note, width=wrap_width))
            else:
                lines.append(note)
        levels = self._low_credibility_levels()
        if levels:
            lines.extend(("", _LOW_CREDIBILITY_HEADING))
            lines.extend(
                f"    {label}: {n_obs} obs ({share:.2f}% exposure)"
                for label, n_obs, share in levels
            )
        return lines

    # ── HTML output ───────────────────────────────────────────────

    def _repr_html_(self) -> str:
        page = _HtmlTable(self)
        parts = [
            f'<table style="{_HTML_TABLE_STYLE}">',
            f'<tr><td colspan="{page.ncols}" style="text-align:center;font-weight:bold;'
            f'padding:8px;font-size:15px;border-bottom:2px solid #333;">'
            f"SuperGLM Results</td></tr>",
        ]
        parts.extend(
            page.header_row(*cells) for cells in self._header_rows(suffix_on_parametric=False)
        )
        parts.append(page.separator())
        parts.extend(page.column_header(self._column_names()))
        parts.append(page.separator())
        parts.extend(self._html_body(page))
        parts.extend(
            page.level_group_legend(feature, items)
            for feature, items in self._level_group_legends().items()
        )
        parts.append(
            f'<tr><td colspan="{page.ncols}" style="border-bottom:2px solid #333;"></td></tr>'
        )
        for style, note in self._notes():
            rendering = _HTML_NOTES[style]
            if rendering is not None:
                parts.append(page.note(rendering[0], rendering[1](note)))
        levels = self._low_credibility_levels()
        if levels:
            lines = [_LOW_CREDIBILITY_HEADING]
            lines.extend(
                f"&nbsp;&nbsp;&nbsp;&nbsp;{html_escape(label)}: {n_obs} obs ({share:.2f}% exposure)"
                for label, n_obs, share in levels
            )
            parts.append(page.note("color:#c60;border:none;", "<br>".join(lines)))
        parts.append("</table>")
        return "\n".join(parts)

    def _html_body(self, page: _HtmlTable) -> list[str]:
        parts: list[str] = []
        level_rules = _level_rule_notes(self._display_rows)
        prev_group = None
        for row in self._display_rows:
            if row.group and row.group != prev_group:
                parts.append(page.group_separator(row.group))
                if row.group in level_rules:
                    parts.append(page.rule_line(level_rules[row.group]))
            prev_group = row.group
            parts.extend(self._html_row(page, row))
        return parts

    def _html_row(self, page: _HtmlTable, row: _CoefRow) -> list[str]:
        kind = _row_kind(row)
        if kind == "structured":
            return [page.text_row(row, _structured_text(row, _HTML))]
        if kind == "group":
            return self._html_group_row(page, row)
        if kind == "reference":
            return [page.numeric_row(row, ("0.0000", "ref", "---", "---", "---", "---"))]
        if kind == "estimated":
            numbers = (
                f"{row.coef:.4f}",
                f"{row.se:.4f}",
                *(_html_number(value) for value in (row.z, row.p, row.ci_low, row.ci_high)),
            )
            advisory = "?" if row.quasi_separated else ""
            return [page.numeric_row(row, numbers, _sig_stars(row.p), advisory)]
        coef = f"{row.coef:.4f}" if row.coef is not None else "---"
        advisory = _SEPARATED_TAG if row.separated else ""
        return [page.numeric_row(row, (coef, *("---",) * 5), advisory=advisory)]

    def _html_group_row(self, page: _HtmlTable, row: _CoefRow) -> list[str]:
        group = _group_text(row, _HTML)
        detail = f"<br><span style='font-size:11px;'>{group.detail}</span>" if group.detail else ""
        if group.has_test:
            parts = [page.text_row(row, group.text + detail, stars=group.stars)]
        else:
            parts = [page.text_row(row, group.text + detail)]
        if row.name in self._basis_detail:
            parts.append(
                page.basis_detail(self._basis_detail[row.name], open_=self._detail == "full")
            )
        return parts


class _TextBox:
    """The text summary's box: its column widths and the lines drawn with them."""

    def __init__(self, summary: ModelSummary, header: list[tuple[str, str, str, str]]):
        rows = summary._display_rows
        basis_name_w = max(
            (
                len(f"  Coef {basis_row.basis_index + 1}")
                for basis_rows in summary._basis_detail.values()
                for basis_row in basis_rows
            ),
            default=0,
        )
        name_w = max(len(r.name) for r in rows) if rows else 10
        self.name_w = max(name_w, basis_name_w, len("Term"), 10)
        self.level_group_w = (
            max(len("Level group"), *(len(row.level_group) for row in rows))
            if summary._level_groups
            else 0
        )
        self.level_fit_w = (
            max(len("Fit"), *(len(row.level_fit or "") for row in rows)) + 2
            if any(row.level_fit is not None for row in rows)
            else 0
        )
        coef_W = (
            self.name_w
            + self.level_group_w
            + self.level_fit_w
            + sum(_COEF_FIELD_WIDTHS)
            + len(_COEF_FIELD_WIDTHS)
        )
        # Each header half is a 20-wide label and a value: 42 + 2 * the longest value.
        max_val = max(max(len(v1), len(v2)) for _, v1, _, v2 in header if v2) if header else 0
        self.W = max(coef_W, 42 + 2 * max_val)  # content width
        self.F = self.W + 2  # fill width between the border characters
        self.val_l = (self.W - 42) // 2
        self.val_r = self.W - 42 - self.val_l  # absorbs an odd remainder

    def top(self, text: str) -> str:
        pad = self.F - len(text)
        left = pad // 2
        return f"╔{'═' * left}{text}{'═' * (pad - left)}╗"

    def mid(self) -> str:
        return f"╠{'═' * self.F}╣"

    def thin(self) -> str:
        return f"╟{'─' * self.F}╢"

    def bot(self) -> str:
        return f"╚{'═' * self.F}╝"

    def group_sep(self, name: str) -> str:
        pad = self.F - (len(name) + 4)
        left = pad // 2
        return f"╠{'═' * left}╡ {name} ╞{'═' * (pad - left)}╣"

    def row(self, text: str) -> str:
        return f"║ {text:<{self.W}s} ║"

    def header_row(self, k1: str, v1: str, k2: str, v2: str) -> str:
        left = f"{k1 + ':':<20s}{v1:>{self.val_l}s}"
        right_label = f"{k2 + ':':<20s}" if k2 else " " * 20
        return self.row(f"{left}  {right_label}{v2:>{self.val_r}s}")

    def prefix(self, name: str, level_group: str = "", fit: str = "") -> str:
        """The term-name column, then the level-group and Fit columns when shown."""
        text = f"{name:<{self.name_w}s}"
        if self.level_group_w:
            text += f"{level_group:>{self.level_group_w}s}"
        if self.level_fit_w:
            text += f"{fit:>{self.level_fit_w}s}"
        return text

    def fields(self, numbers: Sequence[str], sig: str = "", advisory: str = "") -> str:
        """The six numeric fields, Sig and the advisory tag, each after its own space."""
        rendered = "".join(
            f" {value:>{width}s}"
            for value, width in zip(numbers, _COEF_FIELD_WIDTHS[:6], strict=True)
        )
        return f"{rendered} {sig or '---':<3s} {advisory or '---':<3s}"

    def sig_aligned(self, prefix: str, group: _GroupText) -> str:
        """A group-test line with its stars at the header's own Sig offset.

        The text is elided rather than allowed to push the stars right, where a clamped
        pad would slide them out of Sig and overrun the box.  ``room`` is the space
        before Sig, less the two-character gap after the term name.  The kind is elided,
        not the ``p=`` tail: the console is the only place the whole-term p-value
        appears, and the HTML row prints it in full.
        """
        sig_col = len(prefix) + sum(_COEF_FIELD_WIDTHS[:6]) + 7
        room = sig_col - len(prefix) - 2
        text = group.text
        overflow = len(text) - room
        if overflow > 0:
            keep = len(group.kind) - overflow - 1
            if keep >= 1:
                text = text.replace(f"[{group.kind},", f"[{group.kind[:keep]}…,", 1)
            else:
                text = text[: max(room - 1, 0)] + "…"
        line = f"{prefix}  {text}"
        pad = max(sig_col - len(line), 0)
        return f"{line}{'':<{pad}s}{group.stars:<3s} {'':<3s}"


_HTML_TABLE_STYLE = "border-collapse:collapse;font-family:monospace;font-size:13px;margin:8px 0;"
_HTML_CELL = "padding:3px 8px;text-align:right;border:none;"
_HTML_CELL_LEFT = "padding:3px 8px;text-align:left;border:none;"
_HTML_HEADER_CELL = "padding:3px 8px;text-align:right;font-weight:bold;border:none;"
_HTML_HEADER_CELL_LEFT = "padding:3px 8px;text-align:left;font-weight:bold;border:none;"
_HTML_LABEL = "padding:3px 8px;text-align:left;font-weight:bold;color:#555;border:none;"
_HTML_SIG_CELL = "padding:3px 4px;text-align:left;border:none;"
_HTML_GROUP_SEPARATOR = (
    "padding:2px 8px;text-align:left;font-weight:bold;color:#555;"
    "border-top:1px solid #bbb;border-bottom:none;font-size:12px;"
)
_HTML_DETAIL_CELL = "<td style='padding:1px 6px;'>{}</td>"


class _HtmlTable:
    """The HTML summary's table: its column count and the rows drawn with it."""

    def __init__(self, summary: ModelSummary):
        rows = summary._display_rows
        self.has_level_groups = bool(summary._level_groups)
        self.has_level_fit = any(row.level_fit is not None for row in rows)
        self.extra_cols = int(self.has_level_groups) + int(self.has_level_fit)
        self.ncols = 9 + self.extra_cols

    def header_row(self, k1: str, v1: str, k2: str, v2: str) -> str:
        right_label = f"{k2}:" if k2 else ""
        return (
            f"<tr>"
            f'<td style="{_HTML_LABEL}">{k1}:</td>'
            f'<td style="{_HTML_CELL}">{v1}</td>'
            f'<td style="{_HTML_CELL}"></td>'
            f'<td style="{_HTML_LABEL}">{right_label}</td>'
            f'<td colspan="{self.ncols - 4}" style="{_HTML_CELL}">{v2}</td>'
            f"</tr>"
        )

    def separator(self) -> str:
        return f'<tr><td colspan="{self.ncols}" style="border-bottom:1px solid #999;"></td></tr>'

    def column_header(self, numeric: tuple[str, ...]) -> list[str]:
        """The column labels: the term and level columns left-aligned, the numbers right."""
        left = [""]
        if self.has_level_groups:
            left.append("Level group")
        if self.has_level_fit:
            left.append("Fit")
        return [
            "<tr>",
            *(f'<td style="{_HTML_HEADER_CELL_LEFT}">{name}</td>' for name in left),
            *(f'<td style="{_HTML_HEADER_CELL}">{name}</td>' for name in numeric),
            f'<td style="{_HTML_HEADER_CELL_LEFT}">{_LOW_CREDIBILITY_TAG}</td>',
            "</tr>",
        ]

    def group_separator(self, group: str) -> str:
        return (
            f'<tr><td colspan="{self.ncols}" style="{_HTML_GROUP_SEPARATOR}">'
            f"{html_escape(group)}</td></tr>"
        )

    def rule_line(self, text: str) -> str:
        return (
            f'<tr><td colspan="{self.ncols}" style="padding:0 8px 2px;'
            f'font-size:11px;color:#555;">{html_escape(text)}</td></tr>'
        )

    def prefix(self, row: _CoefRow) -> str:
        """The term-name cell, then the level-group and Fit cells when shown."""
        cells = f'<td style="{_HTML_CELL_LEFT}">{html_escape(row.name)}</td>'
        if self.has_level_groups:
            cells += f'<td style="{_HTML_CELL_LEFT}">{html_escape(row.level_group)}</td>'
        if self.has_level_fit:
            cells += f'<td style="{_HTML_CELL_LEFT}">{html_escape(row.level_fit or "")}</td>'
        return cells

    def text_row(self, row: _CoefRow, text: str, *, stars: str | None = None) -> str:
        """A row whose description spans the numeric columns; ``stars`` keeps Sig and LC."""
        span = self.ncols - 1 - self.extra_cols - (0 if stars is None else 2)
        cell = (
            f'<td colspan="{span}" style="{_HTML_CELL_LEFT};color:#666;font-style:italic;">'
            f"{text}</td>"
        )
        if stars is None:
            return f"<tr>{self.prefix(row)}{cell}</tr>"
        return (
            f"<tr>{self.prefix(row)}{cell}"
            f'<td style="{_HTML_SIG_CELL}">{stars}</td><td style="{_HTML_SIG_CELL}"></td></tr>'
        )

    def numeric_row(
        self, row: _CoefRow, numbers: tuple[str, ...], sig: str = "", advisory: str = ""
    ) -> str:
        cells = "".join(f'<td style="{_HTML_CELL}">{value}</td>' for value in numbers)
        return (
            f"<tr>{self.prefix(row)}{cells}"
            f'<td style="{_HTML_SIG_CELL}">{sig}</td><td style="{_HTML_SIG_CELL}">{advisory}</td>'
            f"</tr>"
        )

    def basis_detail(self, rows: list[_BasisDetailRow], *, open_: bool) -> str:
        """A spline row's per-coefficient detail, as a disclosure open under ``detail="full"``."""
        inner_rows = [
            "<tr>"
            + "".join(
                _HTML_DETAIL_CELL.format(text)
                for text in (
                    br.basis_index + 1,
                    f"{br.coef:.4f}",
                    f"{br.se:.4f}" if np.isfinite(br.se) else "---",
                    _html_number(br.z),
                    _html_number(br.p),
                    _html_number(br.ci_low),
                    _html_number(br.ci_high),
                    _sig_stars(br.p),
                    "",
                )
            )
            + "</tr>"
            for br in rows
        ]
        headers = (
            "coef",
            "std err",
            "z",
            "P&gt;|z|",
            "[ci_lo",
            "ci_hi]",
            "Sig",
            _LOW_CREDIBILITY_TAG,
        )
        inner_table = (
            "<table style='width:100%;font-size:11px;color:#555;"
            "border-collapse:collapse;margin:2px 0;'>"
            "<tr><th style='padding:1px 6px;text-align:left;'>#</th>"
            + "".join(f"<th style='padding:1px 6px;'>{name}</th>" for name in headers)
            + "</tr>"
            + "".join(inner_rows)
            + "</table>"
        )
        return (
            f'<tr><td colspan="{self.ncols}" style="padding:0;border:none;">'
            f"<details{' open' if open_ else ''}>"
            f"<summary style='cursor:pointer;font-size:11px;color:#888;"
            f"padding:2px 8px;'>&#x25B6; coefficient detail</summary>"
            f"{inner_table}"
            f"</details></td></tr>"
        )

    def level_group_legend(self, feature: str, items: list[Any]) -> str:
        mapping = "; ".join(
            f"<strong>{html_escape(item.group_id)}</strong> = "
            f"{', '.join(html_escape(member) for member in item.members)}"
            for item in items
        )
        return (
            f'<tr><td colspan="{self.ncols}" style="padding:4px 8px;'
            f'white-space:normal;overflow-wrap:anywhere;border:none;" '
            f'aria-label="Level groups for {html_escape(feature, quote=True)}">'
            f"<strong>Level groups ({html_escape(feature)}):</strong> "
            f"{mapping}</td></tr>"
        )

    def note(self, style: str, content: str) -> str:
        return (
            f'<tr><td colspan="{self.ncols}" style="padding:4px 8px;font-size:11px;'
            f'{style}">{content}</td></tr>'
        )


_EDITOR_SHAPE_NOTE = (
    "were chosen in the editor from this data. Tests are conditional on them; "
    "judge them on validation deviance."
)


def editor_shape_notes(info: dict[str, Any]) -> list[str]:
    """The note for terms whose shaped ranges were chosen in the editor."""
    terms = info.get("editor_shape_terms") or []
    return [f"Shaped ranges for {', '.join(terms)} {_EDITOR_SHAPE_NOTE}"] if terms else []


def shared_cell_row_notes(interaction_specs: dict[str, Any]) -> list[str]:
    """For each categorical interaction whose row names other cells share, the cells.

    A row is named ``term[level1:level2]``, a label other cells of the full grid share once a
    level holds a colon: another row's cell, or a base or unestimated cell that has no row.
    """
    from superglm.features.interaction import CategoricalInteraction, shared_cell_labels

    notes = []
    for name, spec in interaction_specs.items():
        if not isinstance(spec, CategoricalInteraction):
            continue
        grid = shared_cell_labels([spec._base1, *spec._non_base1], [spec._base2, *spec._non_base2])
        rows_by_label: dict[str, list[tuple]] = {}
        for pair in spec._pairs:
            label = f"{pair[0]}:{pair[1]}"
            if label in grid:
                rows_by_label.setdefault(label, []).append(pair)
        first, second = spec.parent_names

        def cell(pair: tuple) -> str:
            return f"({first}={pair[0]}, {second}={pair[1]})"

        shared = []
        for label, rows in rows_by_label.items():
            text = f"{name}[{label}] is " + " then ".join(map(cell, rows))
            rowless = [pair for pair in grid[label] if pair not in rows]
            if rowless:
                text += (
                    f", not {' or '.join(map(cell, rowless))}, which has no row "
                    "(a base or unestimated cell)"
                )
            shared.append(text)
        if shared:
            notes.append(
                f"{name}: levels containing ':' give several cells one row name. In row order, "
                f"{'; '.join(shared)}. relativities()[{name!r}] names each row's cell in its "
                "level1 and level2 columns; renaming the levels so none contains ':' gives every "
                "row its own name."
            )
    return notes


def _editor_notes(info: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if info.get("editor_statistics_note"):
        notes.append(info["editor_statistics_note"])
    if info.get("editor_inference_stale", False):
        terms = ", ".join(info.get("editor_edited_terms") or [])
        suffix = f" Edited terms: {terms}." if terms else ""
        notes.append(_EDITOR_STALE_NOTE + suffix)
    if info.get("editor_offset_terms"):
        terms = ", ".join(info.get("editor_offset_terms") or [])
        notes.append(f"{_EDITOR_OFFSET_NOTE} Offset terms: {terms}.")
    notes.extend(editor_shape_notes(info))
    return notes
