"""Formatted model summary with ASCII and HTML output (statsmodels-style)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm


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
    spline_kind: str | None = None  # "BasisSpline", "NaturalSpline", etc.
    knot_strategy: str | None = None
    boundary: tuple[float, float] | None = None
    # Monotonicity
    monotone: str | None = None  # "increasing", "decreasing", or None
    monotone_repaired: bool = False


def _compute_coef_stats(
    coef: float,
    se: float,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Compute z-value, p-value, and confidence interval."""
    if se <= 0:
        return np.nan, np.nan, np.nan, np.nan
    z = coef / se
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    q = norm.ppf(1.0 - alpha / 2.0)
    return z, p, coef - q * se, coef + q * se


def _camel_to_spaced(name: str) -> str:
    """Convert CamelCase to spaced: 'GroupLasso' -> 'Group Lasso'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


_SIG_LEGEND = "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
_WALD_NOTE = (
    "Note: smooth p-values use Wood (2013) Bayesian test.\n"
    "Parametric p-values are Wald approximations.\n"
    "For borderline significance, use a likelihood ratio test."
)


def _sig_stars(p: float | None) -> str:
    """R-style significance stars for a p-value."""
    if p is None or np.isnan(p):
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
    ):
        self._data = data
        self._info = model_info
        self._coef_rows = coef_rows
        self._alpha = alpha

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

        # Build header rows first (needed to compute minimum width)
        conv_str = f"{info['converged']} ({info['n_iter']} iter)"
        rows = [
            ("Family", info["family"], "No. Observations", str(info["n_obs"])),
            ("Link", info["link"], "Df (effective)", _fmt(info["effective_df"])),
            ("Method", info.get("method", "ML"), "Penalty", info["penalty"]),
            ("Scale (phi)", _fmt(info["phi"]), "Lambda1", _fmt(info["lambda1"])),
            ("Log-Likelihood", _fmt(info["log_likelihood"]), "AIC", _fmt(info["aic"])),
            ("AICc", _fmt(info["aicc"]), "BIC", _fmt(info["bic"])),
            ("EBIC", _fmt(info["ebic"]), "Converged", conv_str),
            ("Deviance", _fmt(info["deviance"]), "", ""),
        ]

        # NB theta profile row
        if "nb_theta" in info:
            ci = info["nb_theta_ci"]
            theta_str = f"{info['nb_theta']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            rows.append(("Theta", theta_str, "Method", info["nb_theta_method"]))

        # Tweedie p profile row
        if "tweedie_p" in info:
            ci = info["tweedie_p_ci"]
            p_str = f"{info['tweedie_p']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            rows.append(("Tweedie p", p_str, "Method", info["tweedie_p_method"]))

        # Compute content width from coefficient columns AND header values
        #   coef table: name_w + coef(10) + se(10) + z(8) + p(8) + ci_lo(9) + ci_hi(9) + sig(4)
        name_w = max(len(r.name) for r in self._coef_rows) if self._coef_rows else 10
        name_w = max(name_w, 10)
        coef_W = name_w + 10 + 10 + 8 + 8 + 9 + 9 + 4

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
        hdr = (
            f"{'':>{name_w}s}"
            f"{'coef':>10s}"
            f"{'std err':>10s}"
            f"{'z':>6s}  "
            f"{'P>|z|':>8s}"
            f"{'[' + f'{half:.3f}':>9s}"
            f"{f'{1 - half:.3f}' + ']':>9s}"
            f"{'':>4s}"
        )
        lines.append(_row(hdr))
        lines.append(_thin())

        # Coefficient rows with group separators
        prev_group = None
        for row in self._coef_rows:
            # Emit group separator when the group changes (blank rows for breathing room)
            if row.group and row.group != prev_group:
                if prev_group is not None:
                    lines.append(_row(""))
                lines.append(_group_sep(row.group))
                lines.append(_row(""))
            prev_group = row.group

            if row.is_spline:
                has_test = row.active and row.wald_chi2 is not None and not np.isnan(row.wald_chi2)
                kind = "linear" if row.subgroup_type == "linear" else "spline"
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
                    mono_str = f"mono={row.monotone}"
                    if row.monotone_repaired:
                        mono_str += ", repaired"
                    detail_parts.append(mono_str)
                detail_str = ", ".join(detail_parts)

                if has_test:
                    p_str = f"{row.wald_p:.3f}" if row.wald_p >= 0.001 else "<0.001"
                    stars = _sig_stars(row.wald_p)
                    if row.ref_df is not None:
                        df_str = f"{row.ref_df:.1f}"
                    else:
                        df_str = str(row.n_params)
                    spline_text = (
                        f"[{kind}, {param_label}, chi2({df_str})={row.wald_chi2:.1f}, p={p_str}]"
                    )
                    prefix = f"{row.name:<{name_w}s}  {spline_text} "
                    pad = max(W - len(prefix) - 3, 0)
                    lines.append(_row(f"{prefix}{'':<{pad}s}{stars:<3s}"))
                    if detail_str:
                        lines.append(_row(f"{'':<{name_w}s}    {detail_str}"))
                elif row.active:
                    spline_text = f"[{kind}, {param_label}, active]"
                    lines.append(_row(f"{row.name:<{name_w}s}  {spline_text}"))
                    if detail_str:
                        lines.append(_row(f"{'':<{name_w}s}    {detail_str}"))
                else:
                    spline_text = f"[{kind}, {param_label}, inactive]"
                    lines.append(_row(f"{row.name:<{name_w}s}  {spline_text}"))
            elif row.coef is not None and row.se is not None and row.se > 0:
                stars = _sig_stars(row.p)
                if abs(row.z) >= 100:
                    z_str = f"{row.z:>8.1f}"
                else:
                    z_str = f"{row.z:>8.3f}"
                lines.append(
                    _row(
                        f"{row.name:<{name_w}s}"
                        f"{row.coef:>10.4f}"
                        f"{row.se:>10.4f}"
                        f"{z_str}"
                        f"{row.p:>8.3f}"
                        f"{row.ci_low:>9.3f}"
                        f"{row.ci_high:>9.3f}"
                        f" {stars:<3s}"
                    )
                )
            else:
                coef_str = f"{row.coef:>10.4f}" if row.coef is not None else f"{'---':>10s}"
                lines.append(
                    _row(
                        f"{row.name:<{name_w}s}"
                        f"{coef_str}"
                        f"{'---':>10s}"
                        f"{'---':>8s}"
                        f"{'---':>8s}"
                        f"{'---':>9s}"
                        f"{'---':>9s}"
                        f"{'':>4s}"
                    )
                )

        lines.append(_bot())
        lines.append(_SIG_LEGEND)
        abbrevs = info.get("penalty_abbrevs", {})
        if abbrevs:
            lines.append("; ".join(f"{k}: {v}" for k, v in abbrevs.items()))
        lines.append(_WALD_NOTE)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    # ── HTML output ───────────────────────────────────────────────

    def _repr_html_(self) -> str:
        info = self._info
        half = self._alpha / 2.0
        _fmt = self._fmt_scalar
        ncols = 8  # name + coef + se + z + p + ci_lo + ci_hi + sig

        css = "border-collapse:collapse;font-family:monospace;font-size:13px;margin:8px 0;"
        cell = "padding:3px 8px;text-align:right;border:none;"
        cell_l = "padding:3px 8px;text-align:left;border:none;"
        hdr_cell = "padding:3px 8px;text-align:right;font-weight:bold;border:none;"
        hdr_cell_l = "padding:3px 8px;text-align:left;font-weight:bold;border:none;"
        sep_style = "border-bottom:1px solid #999;"
        label_style = "padding:3px 8px;text-align:left;font-weight:bold;color:#555;border:none;"
        sig_cell = "padding:3px 4px;text-align:left;border:none;"

        parts: list[str] = []
        parts.append(f'<table style="{css}">')

        # Title
        parts.append(
            f'<tr><td colspan="{ncols}" style="text-align:center;font-weight:bold;'
            f'padding:8px;font-size:15px;border-bottom:2px solid #333;">'
            f"SuperGLM Results</td></tr>"
        )

        # Header rows
        conv_str = f"{info['converged']} ({info['n_iter']} iter)"
        header_rows = [
            ("Family", info["family"], "No. Observations", str(info["n_obs"])),
            ("Link", info["link"], "Df (effective)", _fmt(info["effective_df"])),
            ("Method", info.get("method", "ML"), "Penalty", info["penalty"]),
            ("Scale (phi)", _fmt(info["phi"]), "Lambda1", _fmt(info["lambda1"])),
            ("Log-Likelihood", _fmt(info["log_likelihood"]), "AIC", _fmt(info["aic"])),
            ("AICc", _fmt(info["aicc"]), "BIC", _fmt(info["bic"])),
            ("EBIC", _fmt(info["ebic"]), "Converged", conv_str),
            ("Deviance", _fmt(info["deviance"]), "", ""),
        ]

        # NB theta profile row
        if "nb_theta" in info:
            ci = info["nb_theta_ci"]
            theta_str = f"{info['nb_theta']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            header_rows.append(("Theta", theta_str, "Method", info["nb_theta_method"]))

        # Tweedie p profile row
        if "tweedie_p" in info:
            ci = info["tweedie_p_ci"]
            p_str = f"{info['tweedie_p']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            header_rows.append(("Tweedie p", p_str, "Method", info["tweedie_p_method"]))
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
        col_names = [
            "",
            "coef",
            "std err",
            "z",
            "P>|z|",
            f"[{half:.3f}",
            f"{1 - half:.3f}]",
            "",
        ]
        parts.append("<tr>")
        parts.append(f'<td style="{hdr_cell_l}">{col_names[0]}</td>')
        for cn in col_names[1:-1]:
            parts.append(f'<td style="{hdr_cell}">{cn}</td>')
        parts.append(f'<td style="{hdr_cell_l}">{col_names[-1]}</td>')
        parts.append("</tr>")
        parts.append(f'<tr><td colspan="{ncols}" style="{sep_style}"></td></tr>')

        # Coefficient rows with group separators
        group_sep_style = (
            "padding:2px 8px;text-align:left;font-weight:bold;color:#555;"
            "border-top:1px solid #bbb;border-bottom:none;font-size:12px;"
        )
        prev_group = None
        for row in self._coef_rows:
            if row.group and row.group != prev_group:
                parts.append(
                    f'<tr><td colspan="{ncols}" style="{group_sep_style}">{row.group}</td></tr>'
                )
            prev_group = row.group

            if row.is_spline:
                has_test = row.active and row.wald_chi2 is not None and not np.isnan(row.wald_chi2)
                kind = "linear" if row.subgroup_type == "linear" else "spline"
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
                    mono_str = f"mono={row.monotone}"
                    if row.monotone_repaired:
                        mono_str += ", repaired"
                    detail_parts.append(mono_str)
                detail_str = ", ".join(detail_parts)
                detail_html = (
                    f"<br><span style='font-size:11px;'>{detail_str}</span>" if detail_str else ""
                )

                if has_test:
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
                        f'<td style="{cell_l}">{row.name}</td>'
                        f'<td colspan="{ncols - 2}" style="{cell_l};color:#666;'
                        f'font-style:italic;">{text}</td>'
                        f'<td style="{sig_cell}">{stars}</td>'
                        f"</tr>"
                    )
                elif row.active:
                    text = f"[{kind}, {param_label}, active]{detail_html}"
                    parts.append(
                        f"<tr>"
                        f'<td style="{cell_l}">{row.name}</td>'
                        f'<td colspan="{ncols - 1}" style="{cell_l};color:#666;'
                        f'font-style:italic;">{text}</td></tr>'
                    )
                else:
                    text = f"[{kind}, {param_label}, inactive]"
                    parts.append(
                        f"<tr>"
                        f'<td style="{cell_l}">{row.name}</td>'
                        f'<td colspan="{ncols - 1}" style="{cell_l};color:#666;'
                        f'font-style:italic;">{text}</td></tr>'
                    )
            elif row.coef is not None and row.se is not None and row.se > 0:
                stars = _sig_stars(row.p)
                parts.append(
                    f"<tr>"
                    f'<td style="{cell_l}">{row.name}</td>'
                    f'<td style="{cell}">{row.coef:.4f}</td>'
                    f'<td style="{cell}">{row.se:.4f}</td>'
                    f'<td style="{cell}">{row.z:.3f}</td>'
                    f'<td style="{cell}">{row.p:.3f}</td>'
                    f'<td style="{cell}">{row.ci_low:.3f}</td>'
                    f'<td style="{cell}">{row.ci_high:.3f}</td>'
                    f'<td style="{sig_cell}">{stars}</td>'
                    f"</tr>"
                )
            else:
                coef_str = f"{row.coef:.4f}" if row.coef is not None else "---"
                parts.append(
                    f"<tr>"
                    f'<td style="{cell_l}">{row.name}</td>'
                    f'<td style="{cell}">{coef_str}</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{cell}">---</td>'
                    f'<td style="{sig_cell}"></td>'
                    f"</tr>"
                )

        # Bottom border + legend
        parts.append(f'<tr><td colspan="{ncols}" style="border-bottom:2px solid #333;"></td></tr>')
        parts.append(
            f'<tr><td colspan="{ncols}" style="padding:4px 8px;font-size:11px;'
            f'color:#666;border:none;">{_SIG_LEGEND}</td></tr>'
        )
        wald_html = _WALD_NOTE.replace("\n", "<br>")
        parts.append(
            f'<tr><td colspan="{ncols}" style="padding:4px 8px;font-size:11px;'
            f'color:#888;font-style:italic;border:none;">{wald_html}</td></tr>'
        )
        parts.append("</table>")
        return "\n".join(parts)
