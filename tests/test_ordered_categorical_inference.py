"""Inference semantics for spline-backed ordered categorical terms."""

import numpy as np
import pandas as pd
import pytest

from superglm import OrderedCategorical, Spline, SuperGLM
from superglm.editor.summaries import _compact_summary_row


def _ordered_trend_data():
    rng = np.random.default_rng(20260710)
    levels = [f"L{i}" for i in range(7)]
    codes = np.tile(np.arange(len(levels)), 180)
    rng.shuffle(codes)
    x = np.asarray(levels, dtype=object)[codes]
    x_numeric = codes / (len(levels) - 1)
    weights = rng.uniform(0.6, 1.8, len(codes))
    eta = -0.8 + 0.9 * x_numeric + 0.15 * np.sin(2.0 * np.pi * x_numeric)
    y = rng.poisson(np.exp(eta) * weights).astype(float)
    return pd.DataFrame({"band": x}), pd.DataFrame({"band": x_numeric}), y, weights, levels


def _fit_ordered_and_direct_spline():
    X_ordered, X_numeric, y, weights, levels = _ordered_trend_data()
    ordered = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=levels,
                base="first",
                basis=Spline(kind="ps", k=7),
            )
        },
    )
    direct = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"band": Spline(kind="ps", k=7)},
    )
    ordered.fit(X_ordered, y, sample_weight=weights)
    direct.fit(X_numeric, y, sample_weight=weights)
    return ordered, direct, X_ordered, y, weights, levels


def _smooth_row(summary, name="band"):
    rows = [row for row in summary._coef_rows if row.name == name and row.is_spline]
    assert len(rows) == 1
    return rows[0]


def _level_rows(summary, name="band"):
    return [row for row in summary._coef_rows if row.group == name and not row.is_spline]


def test_ordered_spline_uses_same_global_wood_test_as_direct_spline():
    ordered, direct, _, _, _, _ = _fit_ordered_and_direct_spline()

    ordered_row = _smooth_row(ordered.summary())
    direct_row = _smooth_row(direct.summary())

    assert ordered_row.active
    assert ordered_row.subgroup_type == "ordered_spline"
    assert np.isfinite(ordered_row.wald_chi2)
    assert 0.0 <= ordered_row.wald_p <= 1.0
    assert ordered_row.edf > 0.0
    assert ordered_row.wald_chi2 == pytest.approx(direct_row.wald_chi2, rel=1e-10)
    assert ordered_row.wald_p == pytest.approx(direct_row.wald_p, rel=1e-10)
    assert ordered_row.ref_df == pytest.approx(direct_row.ref_df, rel=1e-10)
    assert ordered_row.edf == pytest.approx(direct_row.edf, rel=1e-10)


def test_ordered_spline_summary_propagates_wood_programming_errors(monkeypatch):
    ordered, _, _, _, _, _ = _fit_ordered_and_direct_spline()
    from superglm.stats import wood_pvalue

    def fail_wood_test(*_args, **_kwargs):
        raise RuntimeError("ordered Wood sentinel defect")

    monkeypatch.setattr(wood_pvalue, "wood_test_smooth", fail_wood_test)

    with pytest.raises(RuntimeError, match="ordered Wood sentinel defect"):
        ordered.summary()


def test_ordered_level_rows_are_effect_estimates_not_hypothesis_tests():
    ordered, _, _, _, _, levels = _fit_ordered_and_direct_spline()
    summary = ordered.summary()
    rows = _level_rows(summary)

    assert len(rows) == len(levels)
    assert all(row.z is None and row.p is None for row in rows)
    assert all(row.edf is None for row in rows)
    assert all(row.se is not None and np.isfinite(row.se) and row.se >= 0.0 for row in rows)
    assert all(np.isfinite(row.ci_low) and np.isfinite(row.ci_high) for row in rows)

    text = str(summary)
    html = summary._repr_html_()
    assert "Wood (2013)" in text
    for level in levels:
        row_name = f"band[{level}]"
        text_row = next(line for line in text.splitlines() if row_name in line)
        assert "---" in text_row
        assert "*" not in text_row
        html_row = html.split(f">{row_name}<", 1)[1].split("</tr>", 1)[0]
        assert "None" not in html_row
        assert "***" not in html_row


def test_editor_payload_reserves_significance_for_global_ordered_smooth():
    ordered, _, _, _, _, _ = _fit_ordered_and_direct_spline()
    summary = ordered.summary()

    smooth = _compact_summary_row(_smooth_row(summary))
    level = _compact_summary_row(_level_rows(summary)[1])

    assert smooth["kind"] == "spline"
    assert smooth["stat_label"] == "chi2"
    assert smooth["p_value"] is not None
    assert level["kind"] == "coef"
    assert level["se"] is not None
    assert level["stat"] is None
    assert level["stat_label"] == ""
    assert level["p_value"] is None
    assert level["sig_code"] == ""


def test_metrics_feature_se_reports_ordered_level_contrasts():
    ordered, _, X, y, weights, levels = _fit_ordered_and_direct_spline()

    metrics = ordered.metrics(X, y, sample_weight=weights)
    result = metrics.feature_se("band")

    assert set(result) == {"levels", "base_level", "se_log_relativity"}
    assert list(result["levels"]) == levels
    assert result["base_level"] == ordered._specs["band"]._base_level
    se = np.asarray(result["se_log_relativity"])
    assert se.shape == (len(levels),)
    assert np.all(np.isfinite(se))
    assert np.all(se >= 0.0)

    summary_se = np.asarray([row.se for row in _level_rows(ordered.summary())])
    np.testing.assert_allclose(
        se,
        np.sqrt(metrics._coefficient_dispersion) * summary_se,
        rtol=1e-12,
        atol=1e-12,
    )


def test_model_and_metrics_summaries_agree_on_ordered_smooth_test():
    ordered, _, X, y, weights, _ = _fit_ordered_and_direct_spline()

    model_summary = ordered.summary()
    metrics_summary = ordered.metrics(X, y, sample_weight=weights).summary()
    model_row = _smooth_row(model_summary)
    metrics_row = _smooth_row(metrics_summary)

    assert metrics_row.wald_chi2 == pytest.approx(model_row.wald_chi2, rel=1e-10)
    assert metrics_row.wald_p == pytest.approx(model_row.wald_p, rel=1e-10)
    assert metrics_row.ref_df == pytest.approx(model_row.ref_df, rel=1e-10)
    model_intercept = next(row for row in model_summary._coef_rows if row.name == "Intercept")
    metrics_intercept = next(row for row in metrics_summary._coef_rows if row.name == "Intercept")
    assert metrics_intercept.coef == pytest.approx(model_intercept.coef, rel=1e-10)
    assert metrics_intercept.se == pytest.approx(model_intercept.se, rel=1e-10)


def test_global_ordered_smooth_test_is_invariant_to_display_base():
    X, _, y, weights, levels = _ordered_trend_data()
    rows = []
    for base in (levels[0], levels[-1]):
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "band": OrderedCategorical(
                    order=levels,
                    base=base,
                    basis=Spline(kind="ps", k=7),
                )
            },
        )
        model.fit(X, y, sample_weight=weights)
        rows.append(_smooth_row(model.summary()))

    assert rows[0].wald_chi2 == pytest.approx(rows[1].wald_chi2, rel=1e-12)
    assert rows[0].wald_p == pytest.approx(rows[1].wald_p, rel=1e-12)
    assert rows[0].ref_df == pytest.approx(rows[1].ref_df, rel=1e-12)


def test_summary_intercept_matches_base_relative_level_parameterization():
    ordered, _, _, _, _, levels = _fit_ordered_and_direct_spline()
    summary = ordered.summary()
    spec = ordered._specs["band"]
    feature_groups = [group for group in ordered._groups if group.feature_name == "band"]
    beta = np.concatenate([ordered.result.beta[group.sl] for group in feature_groups])
    base_effect = float(spec.score(np.array([spec._base_level], dtype=object), beta)[0])
    intercept_row = next(row for row in summary._coef_rows if row.name == "Intercept")

    assert intercept_row.coef == pytest.approx(ordered.result.intercept + base_effect)

    active_group = next(
        group
        for group in ordered._fit_inference_info["active_groups"]
        if group.feature_name == "band"
    )
    base_design = np.asarray(
        spec.transform(np.array([spec._base_level], dtype=object)),
        dtype=float,
    ).ravel()
    contrast = np.zeros(ordered._fit_inference_info["XtWX_inv_aug"].shape[0])
    contrast[0] = 1.0
    contrast[1 + active_group.start : 1 + active_group.end] = base_design
    covariance = ordered._fit_inference_info["XtWX_inv_aug"]
    expected_se = np.sqrt(contrast @ covariance @ contrast)
    assert intercept_row.se == pytest.approx(expected_se)

    prediction_frame = pd.DataFrame({"band": levels})
    fitted_eta = np.log(ordered.predict(prediction_frame))
    displayed_eta = intercept_row.coef + np.array(
        [
            next(row.coef for row in _level_rows(summary) if row.name == f"band[{level}]")
            for level in levels
        ]
    )
    np.testing.assert_allclose(displayed_eta, fitted_eta, rtol=1e-11, atol=1e-11)


def test_select_ordered_spline_has_one_global_test_and_no_level_tests():
    X, _, y, weights, levels = _ordered_trend_data()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=levels,
                basis=Spline(kind="ps", k=7, select=True),
            )
        },
    )
    model.fit(X, y, sample_weight=weights)
    summary = model.summary()

    assert np.isfinite(_smooth_row(summary).wald_p)
    assert all(row.p is None for row in _level_rows(summary))


def test_inactive_ordered_spline_suppresses_level_uncertainty() -> None:
    rng = np.random.default_rng(919)
    levels = [f"L{i}" for i in range(7)]
    X = pd.DataFrame({"band": np.tile(levels, 80)})
    y = rng.poisson(1.0, len(X)).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=1e6,
        features={
            "band": OrderedCategorical(
                order=levels,
                basis=Spline(kind="ps", k=7, select=True),
            )
        },
    )
    model.fit(X, y)
    summary = model.summary()

    assert not _smooth_row(summary).active
    level_rows = _level_rows(summary)
    assert all(row.se is None for row in level_rows)
    assert all(row.ci_low is None and row.ci_high is None for row in level_rows)
    assert all(row.z is None and row.p is None for row in level_rows)

    text = str(summary)
    html = summary._repr_html_()
    assert "nan" not in text.lower()
    assert "nan" not in html.lower()
    compact = [_compact_summary_row(row) for row in level_rows]
    assert all(row["se"] is None and row["p_value"] is None for row in compact)


def test_inactive_metrics_feature_se_keeps_ordered_level_schema() -> None:
    rng = np.random.default_rng(919)
    levels = [f"L{i}" for i in range(7)]
    X = pd.DataFrame({"band": np.tile(levels, 80)})
    y = rng.poisson(1.0, len(X)).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=1e6,
        features={
            "band": OrderedCategorical(
                order=levels,
                basis=Spline(kind="ps", k=7, select=True),
            )
        },
    )
    model.fit(X, y)

    result = model.metrics(X, y).feature_se("band")

    assert set(result) == {"levels", "base_level", "se_log_relativity"}
    assert list(result["levels"]) == levels
    assert result["base_level"] == model._specs["band"]._base_level
    np.testing.assert_array_equal(result["se_log_relativity"], np.zeros(len(levels)))
