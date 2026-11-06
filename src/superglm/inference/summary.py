"""Formatted model summary with ASCII and HTML output (statsmodels-style)."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from html import escape as html_escape
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from superglm.inference.summary_levels import SummaryLevelDisplay


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
    z: float
    p: float
    ci_low: float
    ci_high: float


def _compute_coef_stats(
    coef: float,
    se: float,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Compute z-value, p-value, and confidence interval."""
    if se <= 0:
        return np.nan, np.nan, np.nan, np.nan
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
_WALD_NOTE = (
    "Note: smooth p-values use Wood (2013) Bayesian test.\n"
    "Parametric p-values are Wald approximations.\n"
    "For borderline significance, use a likelihood ratio test."
)
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


class ModelSummary:
    """Formatted model summary with ASCII and HTML output.

    Returned by ``model.summary()`` and ``ModelMetrics.summary()``. Supports:

    - ``print(summary)`` — ASCII table for terminals
    - Jupyter ``_repr_html_`` — HTML table for notebooks
    - ``summary['fit']`` — dict access (backward compat)
    - ``'fit' in summary`` — membership test (backward compat)
    - ``summary.to_dict()`` — full dict
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

    # ── ASCII output ──────────────────────────────────────────────

    def __str__(self) -> str:
        info = self._info
        half = self._alpha / 2.0
        _fmt = self._fmt_scalar
        display_rows = self._display_rows
        has_level_groups = bool(self._level_groups)
        has_level_fit = any(row.level_fit is not None for row in display_rows)

        # Compute EDF breakdown from coef rows
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
        total_edf = info["effective_df"]
        edf_str = _fmt(total_edf)
        if smooth_edf > 0 or parametric_edf > 0:
            edf_str = f"{_fmt(total_edf)} ({_fmt(smooth_edf)} smooth)"

        # Build header rows first (needed to compute minimum width)
        conv_str = f"{info['converged']} ({info['n_iter']} iter)"
        rows = [
            ("Family", info["family"], "No. Observations", str(info["n_obs"])),
            ("Link", info["link"], "Df (effective)", edf_str),
            ("Method", _display_method(info.get("method", "ML")), "Penalty", info["penalty"]),
            ("Scale (phi)", _fmt(info["phi"]), "Pearson chi2", _fmt(info.get("pearson_chi2", ""))),
            ("Log-Likelihood", _fmt(info["log_likelihood"]), "AIC", _fmt(info["aic"])),
            ("AICc", _fmt(info["aicc"]), "BIC", _fmt(info["bic"])),
            ("EBIC", _fmt(info["ebic"]), "Converged", conv_str),
            ("Deviance", _fmt(info["deviance"]), "", ""),
        ]
        if info.get("weight_semantics"):
            rows.append(("Weights", f"{info['weight_semantics']} weights", "", ""))

        # NB theta profile row
        if "nb_theta" in info:
            theta_str = _format_profile_estimate(
                info["nb_theta"], info["nb_theta_ci"], info["nb_theta_ci_status"]
            )
            rows.append(("Theta", theta_str, *_profile_nll_cell(info, "nb_profile_nll")))

        # Tweedie p profile row
        if "tweedie_p" in info:
            p_str = _format_profile_estimate(
                info["tweedie_p"],
                info.get("tweedie_p_ci"),
                info.get("tweedie_p_ci_status"),
            )
            rows.append(("Tweedie p", p_str, *_profile_nll_cell(info, "tweedie_profile_nll")))

        # Compute content width from coefficient columns AND header values
        # Every coefficient field has an explicit leading separator.  Widths
        # remain minimum alignment widths, never implicit column boundaries.
        coef_field_widths = (10, 10, 8, 8, 9, 9, 3, 3)
        name_w = max(len(r.name) for r in display_rows) if display_rows else 10
        basis_name_w = max(
            (
                len(f"  Coef {basis_row.basis_index + 1}")
                for basis_rows in self._basis_detail.values()
                for basis_row in basis_rows
            ),
            default=0,
        )
        name_w = max(name_w, basis_name_w, len("Term"), 10)
        level_group_w = (
            max(len("Level group"), *(len(row.level_group) for row in display_rows))
            if has_level_groups
            else 0
        )
        level_fit_w = (
            max(len("Fit"), *(len(row.level_fit or "") for row in display_rows)) + 2
            if has_level_fit
            else 0
        )
        coef_W = (
            name_w + level_group_w + level_fit_w + sum(coef_field_widths) + len(coef_field_widths)
        )

        # Header layout: "{k1:20}{v1:>val}  {k2:20}{v2:>val}" → need val >= max value len
        # Each half = 20 (key) + val; total = 20 + val + 2 + 20 + val = 42 + 2*val
        max_val = max(max(len(v1), len(v2)) for _, v1, _, v2 in rows if v2) if rows else 0
        header_W = 42 + 2 * max_val

        W = max(coef_W, header_W)  # content width
        F = W + 2  # fill width (between border chars, includes padding spaces)

        # Box-drawing characters (avoid backslash in f-strings for Python <3.12)
        _D = "\u2550"  # ═ double horizontal
        _S = "\u2500"  # ─ single horizontal
        _TL = "\u2554"  # ╔
        _TR = "\u2557"  # ╗
        _BL = "\u255a"  # ╚
        _BR = "\u255d"  # ╝
        _V = "\u2551"  # ║
        _ML = "\u2560"  # ╠
        _MR = "\u2563"  # ╣
        _SL = "\u255f"  # ╟
        _SR = "\u2562"  # ╢
        _LT = "\u2561"  # ╡
        _RT = "\u255e"  # ╞

        # Box-drawing helpers
        def _top(text: str = "") -> str:
            if text:
                pad = F - len(text)
                left = pad // 2
                right = pad - left
                return f"{_TL}{_D * left}{text}{_D * right}{_TR}"
            return f"{_TL}{_D * F}{_TR}"

        def _mid() -> str:
            return f"{_ML}{_D * F}{_MR}"

        def _thin() -> str:
            return f"{_SL}{_S * F}{_SR}"

        def _group_sep(name: str) -> str:
            label = f"{_LT} {name} {_RT}"
            label_cols = len(name) + 4
            pad = F - label_cols
            left = pad // 2
            right = pad - left
            return f"{_ML}{_D * left}{label}{_D * right}{_MR}"

        def _row(text: str) -> str:
            return f"{_V} {text:<{W}s} {_V}"

        def _bot() -> str:
            return f"{_BL}{_D * F}{_BR}"

        def _coef_prefix(row: _CoefRow, *, name: str | None = None) -> str:
            prefix = f"{row.name if name is None else name:<{name_w}s}"
            if has_level_groups:
                prefix += f"{row.level_group if name is None else '':>{level_group_w}s}"
            if has_level_fit:
                fit = (row.level_fit or "") if name is None else ""
                prefix += f"{fit:>{level_fit_w}s}"
            return prefix

        def _coef_fields(
            coef: str,
            se: str,
            z: str,
            p: str,
            ci_low: str,
            ci_high: str,
            sig: str = "",
            advisory: str = "",
        ) -> str:
            numeric = (coef, se, z, p, ci_low, ci_high)
            rendered = "".join(
                f" {value:>{width}s}"
                for value, width in zip(numeric, coef_field_widths[:6], strict=True)
            )
            return f"{rendered} {sig or '---':<3s} {advisory or '---':<3s}"

        lines: list[str] = []

        # Title
        lines.append(_top(" SuperGLM Results "))

        # Header key-value pairs
        val_w = (W - 42) // 2
        val_l = val_w
        val_r = W - 42 - val_w  # absorb odd remainder

        def _header_row(k1: str, v1: str, k2: str, v2: str) -> str:
            left = f"{k1 + ':':<20s}{v1:>{val_l}s}"
            right_label = f"{k2 + ':':<20s}" if k2 else " " * 20
            right = f"{right_label}{v2:>{val_r}s}"
            return _row(f"{left}  {right}")

        for k1, v1, k2, v2 in rows:
            lines.append(_header_row(k1, v1, k2, v2))
        lines.append(_mid())

        # Coefficient table header
        hdr_prefix = f"{'Term':<{name_w}s}"
        if has_level_groups:
            hdr_prefix += f"{'Level group':>{level_group_w}s}"
        if has_level_fit:
            hdr_prefix += f"{'Fit':>{level_fit_w}s}"
        hdr = hdr_prefix + _coef_fields(
            "coef",
            "std err",
            "z",
            "P>|z|",
            "[" + f"{half:.3f}",
            f"{1 - half:.3f}" + "]",
            "Sig",
            _LOW_CREDIBILITY_TAG,
        )
        lines.append(_row(hdr))
        lines.append(_thin())

        # Coefficient rows with group separators
        level_rules = _level_rule_notes(display_rows)
        prev_group = None
        for row in display_rows:
            # Emit group separator when the group changes (blank rows for breathing room)
            if row.group and row.group != prev_group:
                if prev_group is not None:
                    lines.append(_row(""))
                lines.append(_group_sep(row.group))
                lines.append(_row(""))
                if row.group in level_rules:
                    lines.append(_row(f"  {level_rules[row.group]}"))
            prev_group = row.group

            if row.structured_kind is not None:
                kind = {
                    "random_effect": "random effect",
                    "factor_smooth_fs": "factor smooth (fs)",
                    "factor_smooth_sz": "factor smooth (sz)",
                }.get(row.structured_kind, row.structured_kind.replace("_", " "))
                detail = [f"{row.n_params} params"]
                if row.n_levels is not None:
                    detail.insert(0, f"{row.n_levels} levels")
                if row.edf is not None:
                    detail.append(f"edf={row.edf:.1f}")
                detail.extend(f"{name}={value:.2g}" for name, value in row.smoothing_lambdas)
                lines.append(
                    _row(
                        f"{_coef_prefix(row)}  "
                        f"[{kind}, {', '.join(detail)}; use dedicated term report]"
                    )
                )
                continue

            if row.is_spline:
                has_test = (
                    row.active
                    and row.wald_chi2 is not None
                    and np.isfinite(row.wald_chi2)
                    and row.wald_p is not None
                    and np.isfinite(row.wald_p)
                )
                kind = _GROUP_TEST_KIND_LABELS.get(row.subgroup_type or "", "spline")
                param_label = f"{row.n_params} params"
                # Build detail line: edf, lambda, curve SE, monotone
                detail_parts = []
                detail_parts.append(f"rank={row.n_params}")
                if row.edf is not None:
                    detail_parts.append(f"edf={row.edf:.1f}")
                if row.smoothing_lambda is not None:
                    detail_parts.append(f"lam={row.smoothing_lambda:.2g}")
                if has_test and row.curve_se_min is not None and not np.isnan(row.curve_se_min):
                    detail_parts.append(f"curve SE: {row.curve_se_min:.2f}-{row.curve_se_max:.2f}")
                if row.monotone is not None:
                    detail_parts.append(_rules_text(row))
                detail_str = ", ".join(detail_parts)

                if has_test:
                    assert row.wald_p is not None
                    p_str = f"{row.wald_p:.3f}" if row.wald_p >= 0.001 else "<0.001"
                    stars = _sig_stars(row.wald_p)
                    if row.ref_df is not None:
                        df_str = f"{row.ref_df:.1f}"
                    else:
                        df_str = str(row.n_params)
                    spline_text = (
                        f"[{kind}, {param_label}, chi2({df_str})={row.wald_chi2:.1f}, p={p_str}]"
                    )
                    # Placed at the header's own Sig offset: a group row hand-builds its line, and padding to
                    # the box edge put stars in the advisory field. ``_coef_fields`` emits six numeric fields
                    # each preceded by a space, then a space and the 3-wide Sig.
                    row_prefix = _coef_prefix(row)
                    sig_col = len(row_prefix) + sum(coef_field_widths[:6]) + 7
                    # Elided rather than allowed to push the stars right: a clamped pad slides them out of Sig
                    # on a long group text and overruns the box. ``room`` is exactly the space before Sig (two
                    # characters go to the gap after the term name).
                    room = sig_col - len(row_prefix) - 2
                    overflow = len(spline_text) - room
                    if overflow > 0:
                        # The kind is elided, not the ``p=`` tail: the console is the only place the whole-term
                        # p-value appears, and the HTML row prints it in full, so cutting it would make the two
                        # renderers disagree.
                        keep = len(kind) - overflow - 1
                        if keep >= 1:
                            spline_text = spline_text.replace(f"[{kind},", f"[{kind[:keep]}…,", 1)
                        else:
                            spline_text = spline_text[: max(room - 1, 0)] + "…"
                    prefix = f"{row_prefix}  {spline_text}"
                    pad = max(sig_col - len(prefix), 0)
                    lines.append(_row(f"{prefix}{'':<{pad}s}{stars:<3s} {'':<3s}"))
                    if detail_str:
                        lines.append(
                            _row(f"{'':<{name_w + level_group_w + level_fit_w}s}    {detail_str}")
                        )
                elif row.active:
                    spline_text = f"[{kind}, {param_label}, active]"
                    lines.append(_row(f"{_coef_prefix(row)}  {spline_text}"))
                    if detail_str:
                        lines.append(
                            _row(f"{'':<{name_w + level_group_w + level_fit_w}s}    {detail_str}")
                        )
                else:
                    spline_text = f"[{kind}, {param_label}, inactive]"
                    lines.append(_row(f"{_coef_prefix(row)}  {spline_text}"))

                # Coefficient detail rows (only for detail="full")
                if self._detail == "full" and row.name in self._basis_detail:
                    for br in self._basis_detail[row.name]:
                        b_stars = _sig_stars(br.p)
                        b_label = f"  Coef {br.basis_index + 1}"
                        bz_decimals = 1 if abs(br.z) >= 100 else 3
                        lines.append(
                            _row(
                                f"{_coef_prefix(row, name=b_label)}"
                                + _coef_fields(
                                    _format_ascii_number(
                                        br.coef,
                                        decimals=4,
                                        width=coef_field_widths[0],
                                    ),
                                    _format_ascii_number(
                                        br.se,
                                        decimals=4,
                                        width=coef_field_widths[1],
                                    ),
                                    _format_ascii_number(
                                        br.z,
                                        decimals=bz_decimals,
                                        width=coef_field_widths[2],
                                    ),
                                    _format_ascii_number(
                                        br.p,
                                        decimals=3,
                                        width=coef_field_widths[3],
                                    ),
                                    _format_ascii_number(
                                        br.ci_low,
                                        decimals=3,
                                        width=coef_field_widths[4],
                                    ),
                                    _format_ascii_number(
                                        br.ci_high,
                                        decimals=3,
                                        width=coef_field_widths[5],
                                    ),
                                    b_stars,
                                )
                            )
                        )

            elif row.is_reference:
                lines.append(
                    _row(
                        f"{_coef_prefix(row)}"
                        + _coef_fields("0.0000", "ref", "---", "---", "---", "---")
                    )
                )
            elif (
                row.coef is not None
                and row.se is not None
                and (
                    row.se > 0
                    or (row.p is None and row.ci_low is not None and row.ci_high is not None)
                )
            ):
                stars = _sig_stars(row.p)
                advisory = "?" if row.quasi_separated else ""
                z_decimals = (
                    1 if row.z is not None and np.isfinite(row.z) and abs(row.z) >= 100 else 3
                )
                lines.append(
                    _row(
                        f"{_coef_prefix(row)}"
                        + _coef_fields(
                            _format_ascii_number(
                                row.coef,
                                decimals=4,
                                width=coef_field_widths[0],
                            ),
                            _format_ascii_number(
                                row.se,
                                decimals=4,
                                width=coef_field_widths[1],
                            ),
                            _format_ascii_number(
                                row.z,
                                decimals=z_decimals,
                                width=coef_field_widths[2],
                            ),
                            _format_ascii_number(
                                row.p,
                                decimals=3,
                                width=coef_field_widths[3],
                            ),
                            _format_ascii_number(
                                row.ci_low,
                                decimals=3,
                                width=coef_field_widths[4],
                            ),
                            _format_ascii_number(
                                row.ci_high,
                                decimals=3,
                                width=coef_field_widths[5],
                            ),
                            stars,
                            advisory,
                        )
                    )
                )
            else:
                coef_str = _format_ascii_number(
                    row.coef,
                    decimals=4,
                    width=coef_field_widths[0],
                )
                lines.append(
                    _row(
                        f"{_coef_prefix(row)}"
                        + _coef_fields(coef_str, "---", "---", "---", "---", "---")
                    )
                )

        lines.append(_bot())
        if self._level_display == "grouped" and self._level_groups:
            groups_by_feature: dict[str, list[Any]] = {}
            for item in self._level_groups:
                groups_by_feature.setdefault(item.feature, []).append(item)
            for feature, feature_groups in groups_by_feature.items():
                mapping = "; ".join(
                    f"{item.group_id} = {', '.join(item.members)}" for item in feature_groups
                )
                legend = f"Level groups ({feature}): {mapping}"
                lines.extend(
                    textwrap.wrap(
                        legend,
                        width=max(60, min(W + 2, 100)),
                        subsequent_indent="  ",
                    )
                )
        lines.append(_SIG_LEGEND)
        has_low_credibility = any(r.quasi_separated for r in display_rows)
        if has_low_credibility:
            lines.append(_LOW_CREDIBILITY_NOTE)
        abbrevs = info.get("penalty_abbrevs", {})
        if abbrevs:
            lines.append("; ".join(f"{k}: {v}" for k, v in abbrevs.items()))
        if info.get("convergence_note"):
            lines.extend(textwrap.wrap(info["convergence_note"], width=max(60, min(W + 2, 100))))
        for note in _editor_notes(info):
            lines.append(note)
        has_smooth = any(_is_smooth_group_row(r) for r in self._coef_rows)
        if not info.get("editor_inference_stale", False):
            if has_smooth:
                lines.append(_WALD_NOTE)
            else:
                lines.append(
                    "Parametric p-values are Wald approximations.\n"
                    "For borderline significance, use a likelihood ratio test."
                )

        # Low-credibility footnote
        low_credibility_rows = [
            r for r in display_rows if r.quasi_separated and r.level_n_obs is not None
        ]
        if low_credibility_rows:
            lines.append("")
            lines.append("? Low-credibility levels, and the experience behind each:")
            for r in low_credibility_rows:
                exp_pct = r.level_exposure_share * 100 if r.level_exposure_share is not None else 0
                lines.append(
                    f"    {_low_credibility_label(r)}: {r.level_n_obs} obs ({exp_pct:.2f}% exposure)"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    # ── HTML output ───────────────────────────────────────────────

    def _repr_html_(self) -> str:
        info = self._info
        half = self._alpha / 2.0
        _fmt = self._fmt_scalar
        display_rows = self._display_rows
        has_level_groups = bool(self._level_groups)
        has_level_fit = any(row.level_fit is not None for row in display_rows)
        extra_cols = int(has_level_groups) + int(has_level_fit)
        ncols = 9 + extra_cols

        css = "border-collapse:collapse;font-family:monospace;font-size:13px;margin:8px 0;"
        cell = "padding:3px 8px;text-align:right;border:none;"
        cell_l = "padding:3px 8px;text-align:left;border:none;"
        hdr_cell = "padding:3px 8px;text-align:right;font-weight:bold;border:none;"
        hdr_cell_l = "padding:3px 8px;text-align:left;font-weight:bold;border:none;"
        sep_style = "border-bottom:1px solid #999;"
        label_style = "padding:3px 8px;text-align:left;font-weight:bold;color:#555;border:none;"
        sig_cell = "padding:3px 4px;text-align:left;border:none;"

        def _level_group_cell(row: _CoefRow) -> str:
            if not has_level_groups:
                return ""
            return f'<td style="{cell_l}">{html_escape(row.level_group)}</td>'

        def _level_fit_cell(row: _CoefRow) -> str:
            if not has_level_fit:
                return ""
            return f'<td style="{cell_l}">{html_escape(row.level_fit or "")}</td>'

        parts: list[str] = []
        parts.append(f'<table style="{css}">')

        # Title
        parts.append(
            f'<tr><td colspan="{ncols}" style="text-align:center;font-weight:bold;'
            f'padding:8px;font-size:15px;border-bottom:2px solid #333;">'
            f"SuperGLM Results</td></tr>"
        )

        # Compute EDF breakdown (same as ASCII path)
        smooth_edf_html = sum(
            r.edf
            for r in self._coef_rows
            if (_is_smooth_group_row(r) or r.structured_kind is not None) and r.edf is not None
        )
        total_edf_html = info["effective_df"]
        edf_str_html = _fmt(total_edf_html)
        if smooth_edf_html > 0:
            edf_str_html = f"{_fmt(total_edf_html)} ({_fmt(smooth_edf_html)} smooth)"

        # Header rows
        conv_str = f"{info['converged']} ({info['n_iter']} iter)"
        header_rows = [
            ("Family", info["family"], "No. Observations", str(info["n_obs"])),
            ("Link", info["link"], "Df (effective)", edf_str_html),
            ("Method", _display_method(info.get("method", "ML")), "Penalty", info["penalty"]),
            ("Scale (phi)", _fmt(info["phi"]), "Pearson chi2", _fmt(info.get("pearson_chi2", ""))),
            ("Log-Likelihood", _fmt(info["log_likelihood"]), "AIC", _fmt(info["aic"])),
            ("AICc", _fmt(info["aicc"]), "BIC", _fmt(info["bic"])),
            ("EBIC", _fmt(info["ebic"]), "Converged", conv_str),
            ("Deviance", _fmt(info["deviance"]), "", ""),
        ]
        if info.get("weight_semantics"):
            header_rows.append(("Weights", f"{info['weight_semantics']} weights", "", ""))

        # NB theta profile row
        if "nb_theta" in info:
            theta_str = _format_profile_estimate(
                info["nb_theta"], info["nb_theta_ci"], info["nb_theta_ci_status"]
            )
            header_rows.append(("Theta", theta_str, *_profile_nll_cell(info, "nb_profile_nll")))

        # Tweedie p profile row
        if "tweedie_p" in info:
            p_str = _format_profile_estimate(
                info["tweedie_p"],
                info.get("tweedie_p_ci"),
                info.get("tweedie_p_ci_status"),
            )
            header_rows.append(
                ("Tweedie p", p_str, *_profile_nll_cell(info, "tweedie_profile_nll"))
            )
        for k1, v1, k2, v2 in header_rows:
            right_label = f"{k2}:" if k2 else ""
            parts.append(
                f"<tr>"
                f'<td style="{label_style}">{k1}:</td>'
                f'<td style="{cell}">{v1}</td>'
                f'<td style="{cell}"></td>'
                f'<td style="{label_style}">{right_label}</td>'
                f'<td colspan="{ncols - 4}" style="{cell}">{v2}</td>'
                f"</tr>"
            )

        # Separator
        parts.append(f'<tr><td colspan="{ncols}" style="{sep_style}"></td></tr>')

        # Coefficient table header
        col_names = [""]
        if has_level_groups:
            col_names.append("Level group")
        if has_level_fit:
            col_names.append("Fit")
        col_names.extend(
            [
                "coef",
                "std err",
                "z",
                "P>|z|",
                f"[{half:.3f}",
                f"{1 - half:.3f}]",
                "Sig",
                _LOW_CREDIBILITY_TAG,
            ]
        )
        parts.append("<tr>")
        parts.append(f'<td style="{hdr_cell_l}">{col_names[0]}</td>')
        first_numeric = 1 + extra_cols
        for cn in col_names[1:first_numeric]:
            parts.append(f'<td style="{hdr_cell_l}">{cn}</td>')
        for cn in col_names[first_numeric:-1]:
            parts.append(f'<td style="{hdr_cell}">{cn}</td>')
        parts.append(f'<td style="{hdr_cell_l}">{col_names[-1]}</td>')
        parts.append("</tr>")
        parts.append(f'<tr><td colspan="{ncols}" style="{sep_style}"></td></tr>')

        # Coefficient rows with group separators
        group_sep_style = (
            "padding:2px 8px;text-align:left;font-weight:bold;color:#555;"
            "border-top:1px solid #bbb;border-bottom:none;font-size:12px;"
        )
        level_rules = _level_rule_notes(display_rows)
        prev_group = None
        for row in display_rows:
            if row.group and row.group != prev_group:
                parts.append(
                    f'<tr><td colspan="{ncols}" style="{group_sep_style}">'
                    f"{html_escape(row.group)}</td></tr>"
                )
                if row.group in level_rules:
                    parts.append(
                        f'<tr><td colspan="{ncols}" style="padding:0 8px 2px;'
                        f'font-size:11px;color:#555;">{html_escape(level_rules[row.group])}</td></tr>'
                    )
            prev_group = row.group

            if row.structured_kind is not None:
                kind = {
                    "random_effect": "random effect",
                    "factor_smooth_fs": "factor smooth (fs)",
                    "factor_smooth_sz": "factor smooth (sz)",
                }.get(row.structured_kind, row.structured_kind.replace("_", " "))
                detail = [f"{row.n_params} params"]
                if row.n_levels is not None:
                    detail.insert(0, f"{row.n_levels} levels")
                if row.edf is not None:
                    detail.append(f"edf={row.edf:.1f}")
                detail.extend(
                    f"{html_escape(name)}={value:.2g}" for name, value in row.smoothing_lambdas
                )
                text = f"[{html_escape(kind)}, {', '.join(detail)}; use dedicated term report]"
                parts.append(
                    f"<tr>"
                    f'<td style="{cell_l}">{html_escape(row.name)}</td>'
                    f"{_level_group_cell(row)}"
                    f"{_level_fit_cell(row)}"
                    f'<td colspan="{ncols - 1 - extra_cols}" '
                    f'style="{cell_l};color:#666;font-style:italic;">{text}</td>'
                    f"</tr>"
                )
                continue

            if row.is_spline:
                has_test = (
                    row.active
                    and row.wald_chi2 is not None
                    and np.isfinite(row.wald_chi2)
                    and row.wald_p is not None
                    and np.isfinite(row.wald_p)
                )
                kind = _GROUP_TEST_KIND_LABELS.get(row.subgroup_type or "", "spline")
                param_label = f"{row.n_params} params"
                # Build detail suffix: edf, lambda, curve SE, monotone
                detail_parts = []
                detail_parts.append(f"rank={row.n_params}")
                if row.edf is not None:
                    detail_parts.append(f"edf={row.edf:.1f}")
                if row.smoothing_lambda is not None:
                    detail_parts.append(f"&lambda;={row.smoothing_lambda:.2g}")
                if has_test and row.curve_se_min is not None and not np.isnan(row.curve_se_min):
                    detail_parts.append(
                        f"curve SE: {row.curve_se_min:.2f}&ndash;{row.curve_se_max:.2f}"
                    )
                if row.monotone is not None:
                    detail_parts.append(html_escape(_rules_text(row)))
                detail_str = ", ".join(detail_parts)
                detail_html = (
                    f"<br><span style='font-size:11px;'>{detail_str}</span>" if detail_str else ""
                )

                if has_test:
                    assert row.wald_p is not None
                    p_str = f"{row.wald_p:.3f}" if row.wald_p >= 0.001 else "&lt;0.001"
                    stars = _sig_stars(row.wald_p)
                    if row.ref_df is not None:
                        df_str = f"{row.ref_df:.1f}"
                    else:
                        df_str = str(row.n_params)
                    text = (
                        f"[{kind}, "
                        f"{param_label}, "
                        f"&chi;&sup2;({df_str})={row.wald_chi2:.1f}, "
                        f"p={p_str}]{detail_html}"
                    )
                    parts.append(
                        f"<tr>"
                        f'<td style="{cell_l}">{html_escape(row.name)}</td>'
                        f"{_level_group_cell(row)}"
                        f"{_level_fit_cell(row)}"
                        f'<td colspan="{ncols - 3 - extra_cols}" '
                        f'style="{cell_l};color:#666;'
                        f'font-style:italic;">{text}</td>'
                        f'<td style="{sig_cell}">{stars}</td>'
                        f'<td style="{sig_cell}"></td>'
                        f"</tr>"
                    )
                elif row.active:
                    text = f"[{kind}, {param_label}, active]{detail_html}"
                    parts.append(
                        f"<tr>"
                        f'<td style="{cell_l}">{html_escape(row.name)}</td>'
                        f"{_level_group_cell(row)}"
                        f"{_level_fit_cell(row)}"
                        f'<td colspan="{ncols - 1 - extra_cols}" '
                        f'style="{cell_l};color:#666;'
                        f'font-style:italic;">{text}</td></tr>'
                    )
                else:
                    text = f"[{kind}, {param_label}, inactive]"
                    parts.append(
                        f"<tr>"
                        f'<td style="{cell_l}">{html_escape(row.name)}</td>'
                        f"{_level_group_cell(row)}"
                        f"{_level_fit_cell(row)}"
                        f'<td colspan="{ncols - 1 - extra_cols}" '
                        f'style="{cell_l};color:#666;'
                        f'font-style:italic;">{text}</td></tr>'
                    )

                # HTML coefficient-detail disclosure
                if row.name in self._basis_detail:
                    open_attr = " open" if self._detail == "full" else ""
                    inner_rows = []
                    for br in self._basis_detail[row.name]:
                        b_stars = _sig_stars(br.p)
                        inner_rows.append(
                            f"<tr>"
                            f"<td style='padding:1px 6px;'>{br.basis_index + 1}</td>"
                            f"<td style='padding:1px 6px;'>{br.coef:.4f}</td>"
                            f"<td style='padding:1px 6px;'>{br.se:.4f}</td>"
                            f"<td style='padding:1px 6px;'>{br.z:.3f}</td>"
                            f"<td style='padding:1px 6px;'>{br.p:.3f}</td>"
                            f"<td style='padding:1px 6px;'>{br.ci_low:.3f}</td>"
                            f"<td style='padding:1px 6px;'>{br.ci_high:.3f}</td>"
                            f"<td style='padding:1px 6px;'>{b_stars}</td>"
                            f"<td style='padding:1px 6px;'></td>"
                            f"</tr>"
                        )
                    inner_table = (
                        "<table style='width:100%;font-size:11px;color:#555;"
                        "border-collapse:collapse;margin:2px 0;'>"
                        "<tr>"
                        "<th style='padding:1px 6px;text-align:left;'>#</th>"
                        "<th style='padding:1px 6px;'>coef</th>"
                        "<th style='padding:1px 6px;'>std err</th>"
                        "<th style='padding:1px 6px;'>z</th>"
                        "<th style='padding:1px 6px;'>P&gt;|z|</th>"
                        "<th style='padding:1px 6px;'>[ci_lo</th>"
                        "<th style='padding:1px 6px;'>ci_hi]</th>"
                        "<th style='padding:1px 6px;'>Sig</th>"
                        f"<th style='padding:1px 6px;'>{_LOW_CREDIBILITY_TAG}</th>"
                        "</tr>" + "".join(inner_rows) + "</table>"
                    )
                    parts.append(
                        f'<tr><td colspan="{ncols}" style="padding:0;border:none;">'
                        f"<details{open_attr}>"
                        f"<summary style='cursor:pointer;font-size:11px;color:#888;"
                        f"padding:2px 8px;'>&#x25B6; coefficient detail</summary>"
                        f"{inner_table}"
                        f"</details></td></tr>"
                    )

            elif row.is_reference:
                parts.append(
                    f"<tr>"
                    f'<td style="{cell_l}">{html_escape(row.name)}</td>'
                    f"{_level_group_cell(row)}"
                    f"{_level_fit_cell(row)}"
                    f'<td style="{cell}">0.0000</td>'
                    f'<td style="{cell}">ref</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{sig_cell}"></td>'
                    f'<td style="{sig_cell}"></td>'
                    f"</tr>"
                )
            elif (
                row.coef is not None
                and row.se is not None
                and (
                    row.se > 0
                    or (row.p is None and row.ci_low is not None and row.ci_high is not None)
                )
            ):
                stars = _sig_stars(row.p)
                advisory = "?" if row.quasi_separated else ""
                z_text = f"{row.z:.3f}" if row.z is not None and np.isfinite(row.z) else "---"
                p_text = f"{row.p:.3f}" if row.p is not None and np.isfinite(row.p) else "---"
                ci_low_text = (
                    f"{row.ci_low:.3f}"
                    if row.ci_low is not None and np.isfinite(row.ci_low)
                    else "---"
                )
                ci_high_text = (
                    f"{row.ci_high:.3f}"
                    if row.ci_high is not None and np.isfinite(row.ci_high)
                    else "---"
                )
                parts.append(
                    f"<tr>"
                    f'<td style="{cell_l}">{html_escape(row.name)}</td>'
                    f"{_level_group_cell(row)}"
                    f"{_level_fit_cell(row)}"
                    f'<td style="{cell}">{row.coef:.4f}</td>'
                    f'<td style="{cell}">{row.se:.4f}</td>'
                    f'<td style="{cell}">{z_text}</td>'
                    f'<td style="{cell}">{p_text}</td>'
                    f'<td style="{cell}">{ci_low_text}</td>'
                    f'<td style="{cell}">{ci_high_text}</td>'
                    f'<td style="{sig_cell}">{stars}</td>'
                    f'<td style="{sig_cell}">{advisory}</td>'
                    f"</tr>"
                )
            else:
                coef_str = f"{row.coef:.4f}" if row.coef is not None else "---"
                parts.append(
                    f"<tr>"
                    f'<td style="{cell_l}">{html_escape(row.name)}</td>'
                    f"{_level_group_cell(row)}"
                    f"{_level_fit_cell(row)}"
                    f'<td style="{cell}">{coef_str}</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{sig_cell}"></td>'
                    f'<td style="{sig_cell}"></td>'
                    f"</tr>"
                )

        if self._level_display == "grouped" and self._level_groups:
            groups_by_feature: dict[str, list[Any]] = {}
            for item in self._level_groups:
                groups_by_feature.setdefault(item.feature, []).append(item)
            for feature, feature_groups in groups_by_feature.items():
                mapping_html = "; ".join(
                    f"<strong>{html_escape(item.group_id)}</strong> = "
                    f"{', '.join(html_escape(member) for member in item.members)}"
                    for item in feature_groups
                )
                parts.append(
                    f'<tr><td colspan="{ncols}" style="padding:4px 8px;'
                    f'white-space:normal;overflow-wrap:anywhere;border:none;" '
                    f'aria-label="Level groups for {html_escape(feature, quote=True)}">'
                    f"<strong>Level groups ({html_escape(feature)}):</strong> "
                    f"{mapping_html}</td></tr>"
                )

        # Bottom border + legend
        parts.append(f'<tr><td colspan="{ncols}" style="border-bottom:2px solid #333;"></td></tr>')
        parts.append(
            f'<tr><td colspan="{ncols}" style="padding:4px 8px;font-size:11px;'
            f'color:#666;border:none;">{_SIG_LEGEND}</td></tr>'
        )
        has_low_credibility = any(r.quasi_separated for r in display_rows)
        if has_low_credibility:
            # The only multi-line note in this renderer that used to keep its
            # source newlines, which HTML collapses into one paragraph.  Hoisted
            # so the conversion reads the same way as the Wald note's below.
            advisory_note_html = _LOW_CREDIBILITY_NOTE.replace("\n", "<br>")
            parts.append(
                f'<tr><td colspan="{ncols}" style="padding:4px 8px;font-size:11px;'
                f'color:#c60;border:none;">{advisory_note_html}</td></tr>'
            )
        if info.get("convergence_note"):
            parts.append(
                f'<tr><td colspan="{ncols}" style="padding:4px 8px;font-size:11px;'
                f'color:#c60;white-space:normal;border:none;">'
                f"{html_escape(info['convergence_note'])}</td></tr>"
            )
        for note in _editor_notes(info):
            parts.append(
                f'<tr><td colspan="{ncols}" style="padding:4px 8px;font-size:11px;'
                f'color:#8a4b00;font-style:italic;border:none;">{note}</td></tr>'
            )
        has_smooth = any(_is_smooth_group_row(r) for r in self._coef_rows)
        if not info.get("editor_inference_stale", False):
            if has_smooth:
                note_text = _WALD_NOTE
            else:
                note_text = (
                    "Parametric p-values are Wald approximations.\n"
                    "For borderline significance, use a likelihood ratio test."
                )
            note_html = note_text.replace("\n", "<br>")
            parts.append(
                f'<tr><td colspan="{ncols}" style="padding:4px 8px;font-size:11px;'
                f'color:#888;font-style:italic;border:none;">{note_html}</td></tr>'
            )
        # Low-credibility footnote
        low_credibility_rows = [
            r for r in display_rows if r.quasi_separated and r.level_n_obs is not None
        ]
        if low_credibility_rows:
            advisory_lines = ["? Low-credibility levels, and the experience behind each:"]
            for r in low_credibility_rows:
                exp_pct = r.level_exposure_share * 100 if r.level_exposure_share is not None else 0
                advisory_lines.append(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;{html_escape(_low_credibility_label(r))}: "
                    f"{r.level_n_obs} obs ({exp_pct:.2f}% exposure)"
                )
            advisory_html = "<br>".join(advisory_lines)
            parts.append(
                f'<tr><td colspan="{ncols}" style="padding:4px 8px;font-size:11px;'
                f'color:#c60;border:none;">{advisory_html}</td></tr>'
            )

        parts.append("</table>")
        return "\n".join(parts)


_EDITOR_SHAPE_NOTE = (
    "were chosen in the editor from this data. Tests are conditional on them; "
    "judge them on validation deviance."
)


def editor_shape_notes(info: dict[str, Any]) -> list[str]:
    """The note for terms whose shaped ranges were chosen in the editor."""
    terms = info.get("editor_shape_terms") or []
    return [f"Shaped ranges for {', '.join(terms)} {_EDITOR_SHAPE_NOTE}"] if terms else []


def _editor_notes(info: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if info.get("editor_inference_stale", False):
        terms = ", ".join(info.get("editor_edited_terms") or [])
        suffix = f" Edited terms: {terms}." if terms else ""
        notes.append(_EDITOR_STALE_NOTE + suffix)
    if info.get("editor_offset_terms"):
        terms = ", ".join(info.get("editor_offset_terms") or [])
        notes.append(f"{_EDITOR_OFFSET_NOTE} Offset terms: {terms}.")
    notes.extend(editor_shape_notes(info))
    return notes
