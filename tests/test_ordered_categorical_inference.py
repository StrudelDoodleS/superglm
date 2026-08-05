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


def _special_band_data():
    """Ordered bands on a smooth trend plus a structurally different MISSING band."""
    rng = np.random.default_rng(20260805)
    levels = [f"L{i}" for i in range(7)]
    codes = np.tile(np.arange(len(levels)), 180)
    rng.shuffle(codes)
    x_ordered = np.asarray(levels, dtype=object)[codes]
    x_numeric = codes / (len(levels) - 1)
    w_ordered = rng.uniform(0.6, 1.8, len(codes))
    eta_ordered = -0.8 + 0.9 * x_numeric + 0.15 * np.sin(2.0 * np.pi * x_numeric)

    n_special = 90
    w_special = rng.uniform(0.6, 1.8, n_special)
    eta_special = np.full(n_special, 0.9)

    band = np.concatenate([x_ordered, np.full(n_special, "MISSING", dtype=object)])
    weights = np.concatenate([w_ordered, w_special])
    eta = np.concatenate([eta_ordered, eta_special])
    y = rng.poisson(np.exp(eta) * weights).astype(float)
    ordered_mask = np.asarray(band != "MISSING", dtype=bool)
    return pd.DataFrame({"band": band}), y, weights, ordered_mask, levels


def _fit_special_band_model():
    """Fit the specials term. Poisson keeps the scale known, so the Wood test
    uses res_df = -1 and no dispersion estimate enters the comparison."""
    frame, y, weights, _, levels = _special_band_data()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        tol=1e-10,
        max_iter=200,
        features={
            "band": OrderedCategorical(
                order=levels,
                specials=["MISSING"],
                base="first",
                basis=Spline(kind="ps", k=7),
            )
        },
    )
    model.fit(frame, y, sample_weight=weights)
    return model, levels


def test_whole_smooth_test_sees_only_the_spline_block(monkeypatch):
    model, _ = _fit_special_band_model()
    spline_group = next(g for g in model._groups if g.name == "band")
    special_group = next(g for g in model._groups if g.name == "band:special")
    assert special_group.size == 1
    assert special_group.subgroup_type == "special"

    from superglm.stats import wood_pvalue

    real_wood_test = wood_pvalue.wood_test_smooth
    calls = []

    def recording_wood_test(beta_j, X_j, V_b_j, edf1_j, res_df=-1.0):
        calls.append((np.shape(beta_j), np.shape(X_j), np.shape(V_b_j)))
        return real_wood_test(beta_j, X_j, V_b_j, edf1_j, res_df)

    monkeypatch.setattr(wood_pvalue, "wood_test_smooth", recording_wood_test)
    summary = model.summary()

    # FAILS TODAY: coef_tables.py:340 selects every GroupSlice whose
    # feature_name is "band", so active_indices (386-388) spans the spline
    # block AND the special column.  beta_j / X_j / V_b_j therefore come in
    # one column too wide and the p-value tests "curve is flat AND the
    # MISSING offset is zero".
    assert len(calls) == 1
    beta_shape, x_shape, v_shape = calls[0]
    assert beta_shape == (spline_group.size,)
    assert x_shape[1] == spline_group.size
    assert v_shape == (spline_group.size, spline_group.size)

    # The specials block gets no sibling smooth row: one group row per term.
    smooth_rows = [row for row in summary._coef_rows if row.is_spline and row.group == "band"]
    assert [row.name for row in smooth_rows] == ["band"]
    # FAILS TODAY: n_params is len(beta_combined) at coef_tables.py:421.
    assert smooth_rows[0].n_params == spline_group.size


def _fit_ordered_only_reference():
    """The same model with the special rows removed instead of held out."""
    frame, y, weights, ordered_mask, levels = _special_band_data()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        tol=1e-10,
        max_iter=200,
        features={
            "band": OrderedCategorical(
                order=levels,
                base="first",
                basis=Spline(kind="ps", k=7),
            )
        },
    )
    model.fit(
        frame.loc[ordered_mask].reset_index(drop=True),
        y[ordered_mask],
        sample_weight=weights[ordered_mask],
    )
    return model


def test_special_level_does_not_change_the_reported_smooth_statistics():
    with_special, _ = _fit_special_band_model()
    reference = _fit_ordered_only_reference()

    row = _smooth_row(with_special.summary())
    ref_row = _smooth_row(reference.summary())

    assert row.subgroup_type == "ordered_spline"
    assert row.active

    # FAILS TODAY, all four:
    #   n_params  -> spline size + 1   (coef_tables.py:421)
    #   edf       -> spline edf + ~1.0 (coef_tables.py:343-347)
    #   wald_chi2 -> joint test including the large MISSING offset
    #   ref_df    -> driven by edf1 summed over the special column too
    assert row.n_params == ref_row.n_params
    assert row.edf == pytest.approx(ref_row.edf, rel=1e-3)
    assert row.wald_chi2 == pytest.approx(ref_row.wald_chi2, rel=1e-2)
    assert row.ref_df == pytest.approx(ref_row.ref_df, rel=1e-2)
    assert row.wald_p == pytest.approx(ref_row.wald_p, rel=1e-2, abs=1e-6)

    # Lambda / knot metadata are read from the spline block by subgroup_type
    # rather than positionally (coef_tables.py:412-415).  These two assertions
    # do NOT discriminate that read: while the documented block order holds,
    # smooth_groups[0] and feature_groups[0] are the same GroupSlice, so the
    # positional read returns the same metadata.  They pin the values, not the
    # selection rule; the rule is defensive against a reversed build order,
    # which _split_beta and transform would reject before this code runs.
    assert row.spline_kind == ref_row.spline_kind
    assert row.smoothing_lambda == pytest.approx(ref_row.smoothing_lambda, rel=1e-12)


def test_ordered_spline_ref_df_falls_back_to_the_spline_block_width(monkeypatch):
    """When the Wood test cannot factorize, ref_df must still exclude the special."""
    model, _ = _fit_special_band_model()
    spline_group = next(g for g in model._groups if g.name == "band")
    special_group = next(g for g in model._groups if g.name == "band:special")

    from superglm.stats import wood_pvalue

    def singular_wood_test(*_args, **_kwargs):
        raise np.linalg.LinAlgError("singular Wood pseudo-inverse")

    monkeypatch.setattr(wood_pvalue, "wood_test_smooth", singular_wood_test)
    row = _smooth_row(model.summary())

    assert row.active
    assert np.isnan(row.wald_chi2)
    assert row.ref_df == pytest.approx(float(spline_group.size))
    assert row.ref_df != pytest.approx(float(spline_group.size + special_group.size))


def _fit_deselected_spline_with_live_special():
    """Flat ordered trend beside a large MISSING offset: group selection drops
    the spline block, while the unpenalized special block always survives."""
    rng = np.random.default_rng(20260805)
    levels = [f"L{i}" for i in range(7)]
    codes = np.tile(np.arange(len(levels)), 180)
    rng.shuffle(codes)
    x_ordered = np.asarray(levels, dtype=object)[codes]
    w_ordered = rng.uniform(0.6, 1.8, len(codes))
    eta_ordered = np.full(len(codes), -0.8)

    n_special = 200
    w_special = rng.uniform(0.6, 1.8, n_special)
    eta_special = np.full(n_special, 1.2)

    band = np.concatenate([x_ordered, np.full(n_special, "MISSING", dtype=object)])
    weights = np.concatenate([w_ordered, w_special])
    eta = np.concatenate([eta_ordered, eta_special])
    y = rng.poisson(np.exp(eta) * weights).astype(float)

    model = SuperGLM(
        family="poisson",
        selection_penalty=200.0,
        features={
            "band": OrderedCategorical(
                order=levels,
                specials=["MISSING"],
                base="first",
                basis=Spline(kind="ps", k=7, select=True),
            )
        },
    )
    model.fit(pd.DataFrame({"band": band}), y, sample_weight=weights)
    return model


def test_deselected_spline_reports_inactive_even_though_the_special_survives():
    model = _fit_deselected_spline_with_live_special()
    assert set(model.result.rank_info.selected_group_names) == {"band:special"}

    summary = model.summary()
    row = _smooth_row(summary)

    # The specials block is built unpenalized, so it is never deselected.
    # Asking "is any block of this feature selected?" would therefore make the
    # smooth row permanently active while every statistic on it — all of which
    # now describe the spline block — reports an empty smooth.
    assert row.active is False
    assert row.edf == 0.0
    assert row.group_norm == 0.0
    assert row.wald_chi2 is None
    assert row.wald_p is None
    assert row.ref_df is None
    assert _compact_summary_row(row)["ref_df"] is None
    assert "inactive" in str(summary)

    # The free level is still fitted and still has a real standard error: the
    # smooth row going inactive must not suppress the level table.
    special_row = next(r for r in _level_rows(summary) if r.name == "band[MISSING]")
    assert special_row.coef > 1.0
    assert special_row.se is not None and np.isfinite(special_row.se) and special_row.se > 0.0


def test_editor_stale_rows_agree_that_a_deselected_spline_is_inactive():
    from superglm.model.report_ops import _build_editor_stale_coef_rows

    model = _fit_deselected_spline_with_live_special()
    rows = _build_editor_stale_coef_rows(model)
    smooth_row = next(row for row in rows if row.is_spline and row.name == "band")

    assert smooth_row.active is False
    assert smooth_row.group_norm == pytest.approx(0.0, abs=1e-12)


def test_editor_stale_rows_report_the_spline_block_only():
    from superglm.model.report_ops import _build_editor_stale_coef_rows

    model, _ = _fit_special_band_model()
    spline_group = next(g for g in model._groups if g.name == "band")
    # The free level really does cost a whole degree of freedom.
    assert model._group_edf["band:special"] == pytest.approx(1.0, abs=0.05)

    rows = _build_editor_stale_coef_rows(model)
    smooth_row = next(row for row in rows if row.is_spline and row.name == "band")

    # FAILS TODAY: report_ops.py:399-401 sums group_edf over both blocks,
    # :417 counts len(beta_combined), :419 norms the full-width vector —
    # so the stale summary shows the free level's edf and its coefficient
    # inside the smooth row.
    assert smooth_row.n_params == spline_group.size
    assert smooth_row.edf == pytest.approx(model._group_edf["band"], rel=1e-12)
    assert smooth_row.group_norm == pytest.approx(
        float(np.linalg.norm(model.result.beta[spline_group.sl])), rel=1e-12
    )


def test_term_inference_edf_agrees_with_the_summary_smooth_row():
    # False today. `_compute_term_edf` sums per-group edf over EVERY group of the
    # feature, including the unpenalized `band:special` block (~1.0 edf per
    # special). `coef_tables` and `report_ops` both scope the same quantity to the
    # spline block, so one fitted term reports two different edfs depending on
    # which surface you read -- and TermInference.edf is the one that reaches the
    # editor's context bar and the plot data.
    #
    # This PR's contract is that everything describing the smooth is the spline
    # block, and the edf sits beside a curve that IS the spline.
    model = _fit_special_band_model()[0]
    summary_edf = _smooth_row(model.summary()).edf
    assert summary_edf is not None

    assert model.term_inference("band").edf == pytest.approx(summary_edf)


def test_a_deselected_spline_does_not_present_an_active_smooth_curve():
    # False today. When selection drops the spline but keeps the unpenalized
    # special, the outer `active` flag stays true (correctly -- the free level IS
    # still fitted), and the curve branch then runs with every spline group
    # filtered out. `_spline_se` returns all-zero SEs and a SmoothCurve is emitted
    # anyway, so a plot renders an active-looking smooth with zero-width bands for
    # a block that was deselected.
    model = _fit_deselected_spline_with_live_special()
    assert set(model.result.rank_info.selected_group_names) == {"band:special"}

    ti = model.term_inference("band")
    assert ti.smooth_curve is None, "no spline block survived, so there is no curve"
