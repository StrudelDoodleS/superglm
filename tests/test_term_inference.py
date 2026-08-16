"""Tests for term_inference API and enriched summary."""

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    InteractionInference,
    Numeric,
    OrderedCategorical,
    Polynomial,
    Spline,
    SplineMetadata,
    SuperGLM,
    TermInference,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 500
    age = rng.uniform(18, 85, n)
    region = rng.choice(["A", "B", "C"], n, p=[0.3, 0.3, 0.4])
    density = rng.normal(5, 2, n)
    sample_weight = rng.uniform(0.3, 1.0, n)
    mu = np.exp(-2.0 + 0.01 * (age - 50) ** 2 / 100 + (region == "A") * 0.3)
    y = rng.poisson(mu * sample_weight).astype(float)
    X = pd.DataFrame({"age": age, "region": region, "density": density})
    return X, y, sample_weight


@pytest.fixture
def fitted_model(sample_data):
    X, y, sample_weight = sample_data
    model = SuperGLM(
        penalty="group_lasso",
        selection_penalty=0.01,
        features={
            "age": Spline(n_knots=10, penalty="ssp"),
            "region": Categorical(base="first"),
            "density": Numeric(),
        },
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


# ── Phase 1-2: TermInference shape and metadata ─────────────────


class TestTermInferenceSpline:
    def test_returns_term_inference(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert isinstance(ti, TermInference)

    def test_kind_is_spline(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.kind == "spline"

    def test_active(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.active is True

    def test_x_shape(self, fitted_model):
        ti = fitted_model.term_inference("age", n_points=100)
        assert ti.x.shape == (100,)

    def test_log_relativity_shape(self, fitted_model):
        ti = fitted_model.term_inference("age", n_points=100)
        assert ti.log_relativity.shape == (100,)
        assert ti.relativity.shape == (100,)

    def test_se_finite(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.se_log_relativity is not None
        assert np.all(np.isfinite(ti.se_log_relativity))
        assert np.all(ti.se_log_relativity >= 0)

    def test_pointwise_ci(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.ci_lower is not None
        assert ti.ci_upper is not None
        assert np.all(np.isfinite(ti.ci_lower))
        assert np.all(np.isfinite(ti.ci_upper))
        assert np.all(ti.ci_lower <= ti.relativity)
        assert np.all(ti.ci_upper >= ti.relativity)

    def test_no_simultaneous_by_default(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.ci_lower_simultaneous is None
        assert ti.ci_upper_simultaneous is None
        assert ti.critical_value_simultaneous is None

    def test_simultaneous_when_requested(self, fitted_model):
        ti = fitted_model.term_inference("age", simultaneous=True)
        assert ti.ci_lower_simultaneous is not None
        assert ti.ci_upper_simultaneous is not None
        assert ti.critical_value_simultaneous is not None
        assert ti.critical_value_simultaneous > 0

    def test_simultaneous_wider_than_pointwise(self, fitted_model):
        ti = fitted_model.term_inference("age", simultaneous=True)
        # Simultaneous bands should be at least as wide as pointwise
        assert np.all(ti.ci_lower_simultaneous <= ti.ci_lower + 1e-10)
        assert np.all(ti.ci_upper_simultaneous >= ti.ci_upper - 1e-10)

    def test_alpha_propagated(self, fitted_model):
        ti = fitted_model.term_inference("age", alpha=0.10)
        assert ti.alpha == 0.10

    def test_edf_present_and_positive(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.edf is not None
        assert ti.edf > 0

    def test_smoothing_lambda_present(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.smoothing_lambda is not None

    def test_absorbs_intercept(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.absorbs_intercept is True

    def test_centering_mode_default_native(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.centering_mode == "training_mean_zero_unweighted"

    def test_centering_mode_explicit_mean(self, fitted_model):
        ti = fitted_model.term_inference("age", centering="mean")
        assert ti.centering_mode == "mean"


class TestTermInferenceSplineMetadata:
    def test_spline_metadata_present(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.spline is not None
        assert isinstance(ti.spline, SplineMetadata)

    def test_spline_kind(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.spline.kind == "PSpline"

    def test_knot_strategy(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.spline.knot_strategy == "uniform"

    def test_interior_knots(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert len(ti.spline.interior_knots) == 10
        assert np.all(np.diff(ti.spline.interior_knots) > 0)

    def test_boundary(self, fitted_model):
        ti = fitted_model.term_inference("age")
        lo, hi = ti.spline.boundary
        assert lo < hi
        assert lo >= 18
        assert hi <= 85

    def test_n_basis(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.spline.n_basis == 14  # n_knots(10) + degree(3) + 1

    def test_degree(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.spline.degree == 3

    def test_extrapolation(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.spline.extrapolation == "clip"

    def test_knot_alpha_none_for_uniform(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.spline.knot_alpha is None

    def test_no_spline_metadata_for_categorical(self, fitted_model):
        ti = fitted_model.term_inference("region")
        assert ti.spline is None

    def test_no_spline_metadata_for_numeric(self, fitted_model):
        ti = fitted_model.term_inference("density")
        assert ti.spline is None


class TestTermInferenceCategorical:
    def test_kind(self, fitted_model):
        ti = fitted_model.term_inference("region")
        assert ti.kind == "categorical"

    def test_levels(self, fitted_model):
        ti = fitted_model.term_inference("region")
        assert ti.levels is not None
        assert set(ti.levels) == {"A", "B", "C"}

    def test_relativity_shape(self, fitted_model):
        ti = fitted_model.term_inference("region")
        assert len(ti.relativity) == 3
        assert len(ti.log_relativity) == 3

    def test_se_finite(self, fitted_model):
        ti = fitted_model.term_inference("region")
        assert ti.se_log_relativity is not None
        assert np.all(np.isfinite(ti.se_log_relativity))

    def test_ci_present(self, fitted_model):
        ti = fitted_model.term_inference("region")
        assert ti.ci_lower is not None
        assert ti.ci_upper is not None

    def test_x_is_none(self, fitted_model):
        ti = fitted_model.term_inference("region")
        assert ti.x is None

    def test_centering_mode_default_native(self, fitted_model):
        ti = fitted_model.term_inference("region")
        assert ti.centering_mode == "base_level"

    def test_centering_mode_explicit_mean(self, fitted_model):
        ti = fitted_model.term_inference("region", centering="mean")
        assert ti.centering_mode == "mean"


class TestTermInferenceNumeric:
    def test_kind(self, fitted_model):
        ti = fitted_model.term_inference("density")
        assert ti.kind == "numeric"

    def test_relativity_scalar(self, fitted_model):
        ti = fitted_model.term_inference("density")
        assert len(ti.relativity) == 1
        assert len(ti.log_relativity) == 1

    def test_se_finite(self, fitted_model):
        ti = fitted_model.term_inference("density")
        assert ti.se_log_relativity is not None
        assert np.all(np.isfinite(ti.se_log_relativity))


class TestTermInferencePolynomialResolution:
    """``n_points`` has to reach the grid, not only the standard errors.

    ``feature_se_from_cov`` honoured the caller's ``n_points`` while the
    polynomial branch reconstructed at the default 200, and ``term_inference``
    zipped the two: measured on a pristine checkout, ``n_points=50`` raised
    ``operands could not be broadcast together with shapes (200,) (50,)`` from
    ``ci_lo = _safe_exp(log_rel - z_alpha * se)``.  So the term, and every
    surface that forwards ``n_points`` -- ``plot``, ``plot_data``, the editor
    session -- was unusable on a ``Polynomial`` at any resolution but 200.
    """

    @pytest.fixture
    def poly_model(self, sample_data):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            selection_penalty=0.0,
            features={"age": Polynomial(degree=3), "region": Categorical(base="first")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        return model

    @pytest.mark.parametrize("n_points", [7, 50, 200, 401])
    def test_every_reported_vector_is_at_the_requested_resolution(self, poly_model, n_points):
        ti = poly_model.term_inference("age", n_points=n_points)
        for field in ("x", "log_relativity", "relativity", "se_log_relativity"):
            assert getattr(ti, field).shape == (n_points,), field
        assert ti.ci_lower.shape == (n_points,)
        assert ti.ci_upper.shape == (n_points,)

    def test_a_coarser_grid_is_the_same_curve_read_at_fewer_points(self, poly_model):
        """Resolution changes where the curve is sampled, not what it is.

        397 and 100 are chosen so the coarse grid is an exact SUBSET of the
        fine one: ``linspace`` spans the same fitted range in both, and
        ``396 / 99 == 4``, so ``fine.x[::4]`` and ``coarse.x`` are the same
        points.  (200 and 50 are not such a pair -- 199 is prime -- which is
        why the sizes here are not the defaults.)  The two spellings of each
        point differ only by the float64 rounding of ``lo + i*(hi-lo)/n``, so
        one cubic on one coefficient vector must agree to round-off; the
        tolerance is that, not headroom.
        """
        fine = poly_model.term_inference("age", n_points=397)
        coarse = poly_model.term_inference("age", n_points=100)

        np.testing.assert_allclose(coarse.x, fine.x[::4], rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(
            coarse.log_relativity, fine.log_relativity[::4], rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            coarse.se_log_relativity, fine.se_log_relativity[::4], rtol=1e-12, atol=1e-14
        )
        # A band, not a translated one: every point's interval brackets it.
        assert np.all(coarse.ci_lower <= coarse.relativity)
        assert np.all(coarse.relativity <= coarse.ci_upper)


class TestTermInferenceWithoutSE:
    def test_no_se_when_disabled(self, fitted_model):
        ti = fitted_model.term_inference("age", with_se=False)
        assert ti.se_log_relativity is None
        assert ti.ci_lower is None
        assert ti.ci_upper is None


# ── Phase 3: to_dataframe ────────────────────────────────────────


class TestToDataFrame:
    def test_spline_df_columns(self, fitted_model):
        ti = fitted_model.term_inference("age")
        df = ti.to_dataframe()
        assert "x" in df.columns
        assert "log_relativity" in df.columns
        assert "relativity" in df.columns
        assert "se_log_relativity" in df.columns
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns

    def test_spline_df_with_simultaneous(self, fitted_model):
        ti = fitted_model.term_inference("age", simultaneous=True)
        df = ti.to_dataframe()
        assert "ci_lower_simultaneous" in df.columns
        assert "ci_upper_simultaneous" in df.columns

    def test_categorical_df_columns(self, fitted_model):
        ti = fitted_model.term_inference("region")
        df = ti.to_dataframe()
        assert "level" in df.columns
        assert "log_relativity" in df.columns
        assert "relativity" in df.columns

    def test_numeric_df_columns(self, fitted_model):
        ti = fitted_model.term_inference("density")
        df = ti.to_dataframe()
        assert "label" in df.columns
        assert df["label"].iloc[0] == "per_unit"


# ── Phase 4: Enriched summary ───────────────────────────────────


class TestEnrichedSummary:
    def test_spline_row_has_edf(self, sample_data):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        m = model.metrics(X, y, sample_weight=sample_weight)
        s = m.summary()
        text = str(s)
        assert "edf=" in text

    def test_spline_row_has_lambda(self, sample_data):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        m = model.metrics(X, y, sample_weight=sample_weight)
        s = m.summary()
        text = str(s)
        assert "lam=" in text

    def test_spline_row_has_rank(self, sample_data):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        text = str(model.metrics(X, y, sample_weight=sample_weight).summary())
        assert "rank=" in text

    def test_non_spline_terms_unaffected(self, sample_data):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "region": Categorical(base="first"),
                "density": Numeric(),
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        m = model.metrics(X, y, sample_weight=sample_weight)
        s = m.summary()
        # Check that non-spline rows still have normal coef/se/z/p format
        text = str(s)
        assert "region[B]" in text
        assert "density" in text

    def test_coef_rows_have_metadata(self, sample_data):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={"age": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        m = model.metrics(X, y, sample_weight=sample_weight)
        s = m.summary()
        spline_rows = [r for r in s._coef_rows if r.is_spline]
        assert len(spline_rows) >= 1
        row = spline_rows[0]
        assert row.spline_kind == "PSpline"
        assert row.knot_strategy == "uniform"
        assert row.boundary is not None
        assert row.edf is not None
        assert row.smoothing_lambda is not None


class TestModelDiagnosticsSplineKeys:
    """model.diagnostics() dict includes spline metadata for spline groups."""

    def test_spline_group_has_enriched_keys(self, fitted_model):
        s = fitted_model.diagnostics()
        # "age" is the spline group name
        age_entry = s["age"]
        assert "edf" in age_entry
        assert "smoothing_lambda" in age_entry
        assert "spline_kind" in age_entry
        assert "knot_strategy" in age_entry
        assert "boundary" in age_entry

    def test_spline_group_values(self, fitted_model):
        s = fitted_model.diagnostics()
        age_entry = s["age"]
        assert age_entry["spline_kind"] == "PSpline"
        assert age_entry["knot_strategy"] == "uniform"
        assert age_entry["edf"] is not None
        assert age_entry["edf"] > 0
        assert age_entry["smoothing_lambda"] is not None
        lo, hi = age_entry["boundary"]
        assert lo < hi

    def test_non_spline_group_no_extra_keys(self, fitted_model):
        s = fitted_model.diagnostics()
        region_entry = s["region"]
        assert "edf" not in region_entry
        assert "spline_kind" not in region_entry

    def test_numeric_group_no_extra_keys(self, fitted_model):
        s = fitted_model.diagnostics()
        density_entry = s["density"]
        assert "edf" not in density_entry
        assert "spline_kind" not in density_entry

    def test_backward_compat_keys_preserved(self, fitted_model):
        s = fitted_model.diagnostics()
        for name in ["age", "region", "density"]:
            assert "active" in s[name]
            assert "group_norm" in s[name]
            assert "n_params" in s[name]
        assert "_model" in s

    def test_diagnostics_and_metrics_summary_agree(self, sample_data):
        """Both report the same edf/lambda for the same spline group."""
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={"age": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, sample_weight=sample_weight)

        diag = model.diagnostics()
        rich = model.metrics(X, y, sample_weight=sample_weight).summary()
        rich_row = next(r for r in rich._coef_rows if r.is_spline)

        assert np.isclose(diag["age"]["edf"], rich_row.edf, rtol=1e-10)
        assert diag["age"]["smoothing_lambda"] == rich_row.smoothing_lambda
        assert diag["age"]["spline_kind"] == rich_row.spline_kind

    def test_inactive_groups_report_zero_edf(self):
        """Groups zeroed by group lasso must report edf=0, not n_levels or 1."""
        from superglm.features.categorical import Categorical
        from superglm.features.numeric import Numeric

        rng = np.random.default_rng(99)
        n = 500
        X = pd.DataFrame(
            {
                "x": rng.standard_normal(n),
                "cat": rng.choice(["a", "b", "c"], n),
            }
        )
        y = rng.poisson(1.0, n).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=100.0,  # heavy penalty to zero out groups
            features={"x": Numeric(), "cat": Categorical(base="first")},
        )
        model.fit(X, y)

        s = model.summary()
        for row in s._coef_rows:
            if row.name == "x" or (row.group == "cat" and row.edf is not None):
                assert row.edf == 0.0, f"{row.name} should have edf=0 when inactive, got {row.edf}"

    def test_edf_header_ascii_html_parity(self):
        """ASCII and HTML summaries must show the same EDF breakdown."""
        rng = np.random.default_rng(42)
        n = 300
        X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
        y = rng.poisson(np.exp(0.5 + 0.3 * np.sin(X["x"].values))).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit_reml(X, y)
        s = model.summary()

        ascii_out = str(s)
        html_out = s._repr_html_()

        # Both should contain the smooth EDF breakdown
        import re

        ascii_match = re.search(r"Df \(effective\).*?(\d+\.\d+).*?\((\d+\.\d+) smooth\)", ascii_out)
        html_match = re.search(r"Df \(effective\).*?(\d+\.\d+).*?\((\d+\.\d+) smooth\)", html_out)

        assert ascii_match is not None, "ASCII summary missing EDF breakdown"
        assert html_match is not None, "HTML summary missing EDF breakdown"
        assert ascii_match.group(1) == html_match.group(1), "Total EDF mismatch"
        assert ascii_match.group(2) == html_match.group(2), "Smooth EDF mismatch"


# ── Phase 5: Spline kinds ───────────────────────────────────────


class TestSplineKinds:
    @pytest.mark.parametrize(
        "kind,expected_class",
        [("bs", "BSplineSmooth"), ("ns", "NaturalSpline"), ("cr", "CubicRegressionSpline")],
    )
    def test_term_inference_spline_kind(self, sample_data, kind, expected_class):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={"age": Spline(kind=kind, n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("age")
        assert ti.spline.kind == expected_class

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr"])
    def test_se_finite_all_kinds(self, sample_data, kind):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={"age": Spline(kind=kind, n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("age")
        assert np.all(np.isfinite(ti.se_log_relativity))


# ── Phase 6: Interaction inference ───────────────────────────────


class TestInteractionInference:
    def test_spline_categorical_interaction(self, sample_data):
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.001,
            features={
                "age": Spline(n_knots=8, penalty="ssp"),
                "region": Categorical(base="first"),
            },
            interactions=[("age", "region")],
        )
        model.fit(X, y, sample_weight=sample_weight)
        ii = model.term_inference("age:region")
        assert isinstance(ii, InteractionInference)
        assert ii.kind == "spline_categorical"

    def test_unknown_feature_raises(self, fitted_model):
        with pytest.raises(KeyError, match="Feature not found"):
            fitted_model.term_inference("nonexistent")


# ── Phase 7: Centering validation ────────────────────────────────


class TestCenteringMetadata:
    def test_spline_default_native_centering(self, sample_data):
        """Default centering='native' returns the canonical fitted term."""
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={"age": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("age")
        assert ti.centering_mode == "training_mean_zero_unweighted"

    def test_spline_explicit_mean_centering(self, sample_data):
        """centering='mean' shifts so geometric mean of relativities = 1."""
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={"age": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        ti = model.term_inference("age", centering="mean")
        assert ti.centering_mode == "mean"
        assert abs(np.mean(ti.log_relativity)) < 1e-10

    def test_reconstruct_matches_native_term_inference(self, sample_data):
        """reconstruct_feature() and term_inference(centering='native') agree."""
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={"age": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, sample_weight=sample_weight)
        raw = model.reconstruct_feature("age")
        ti = model.term_inference("age", centering="native")
        np.testing.assert_allclose(raw["log_relativity"], ti.log_relativity, atol=1e-12)

    def test_the_recorded_shift_is_the_constant_that_was_subtracted(self, sample_data):
        """``centering_shift`` is not recoverable from the values it produced.

        A mean-centered term reports values whose mean is zero, so the removed
        constant leaves no trace in ``log_relativity``.  Anything that has to
        add it back -- the rating-table export folds it into the exported base
        relativity -- needs it recorded, and needs it to be the number that was
        subtracted, exactly, on every element.
        """
        X, y, sample_weight = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            selection_penalty=0.01,
            features={"age": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, sample_weight=sample_weight)

        native = model.term_inference("age", centering="native")
        mean = model.term_inference("age", centering="mean")

        assert native.centering_shift == 0.0
        assert mean.centering_shift != 0.0
        np.testing.assert_array_equal(
            np.asarray(native.log_relativity) - mean.centering_shift,
            np.asarray(mean.log_relativity),
        )

    def test_a_term_the_centering_left_alone_records_no_shift(self, sample_data):
        """Zero is a claim about the values, not a default nobody set.

        A ``Numeric`` has one value and no mean to center on, and an
        ``OrderedCategorical`` is already anchored on its base level, so
        ``centering="mean"`` returns both untouched.  A consumer that adds a
        constant back for them would move every prediction, so the zero has to
        be the truth rather than an omission.
        """
        X, y, sample_weight = sample_data
        band = np.asarray(["b1", "b2", "b3", "b4"])[np.arange(len(X)) % 4]
        frame = X.copy()
        frame["band"] = band
        model = SuperGLM(
            selection_penalty=0.0,
            features={
                "density": Numeric(),
                "band": OrderedCategorical(
                    order=["b1", "b2", "b3", "b4"], basis=Spline(kind="ps", n_knots=3)
                ),
            },
        )
        model.fit(frame, y, sample_weight=sample_weight)

        for name in ("density", "band"):
            native = model.term_inference(name, centering="native")
            mean = model.term_inference(name, centering="mean")
            assert mean.centering_shift == 0.0, name
            np.testing.assert_array_equal(native.log_relativity, mean.log_relativity)

        # Both would have contributed a non-zero constant to a re-derivation,
        # so the zeros above are constraints rather than coincidences.
        for name in ("density", "band"):
            log_rel = np.asarray(model.term_inference(name).log_relativity)
            assert abs(float(np.mean(log_rel))) > 1e-3, name


class TestTermInferenceConstructorContract:
    """``TermInference`` is public, so its POSITIONAL order is a contract.

    Every field after ``active`` is optional with a default, so a field
    inserted in the middle does not raise: it renumbers every positional
    argument after it and the call still succeeds, with each one landing a
    field to the left of where its author wrote it.  That is the one shape of
    break a type checker, a test suite and a reviewer all read straight past.

    It is not hypothetical.  ``centering_shift`` was first added between
    ``centering_mode`` and ``edf``, where a caller's sixteenth positional
    argument -- written as ``edf`` -- became a centering shift instead, with
    ``edf`` silently ``None``.  ``centering_shift`` is added into the exported
    base relativity, so that caller's rating tables would have priced every
    risk at ``exp(edf)`` of the model's premium.
    """

    # v0.26.0's order, which is the released constructor contract.  Extend this
    # list by APPENDING; changing any existing position is the break.
    _RELEASED_FIELD_ORDER = (
        "name",
        "kind",
        "active",
        "x",
        "levels",
        "log_relativity",
        "relativity",
        "se_log_relativity",
        "ci_lower",
        "ci_upper",
        "ci_lower_simultaneous",
        "ci_upper_simultaneous",
        "critical_value_simultaneous",
        "absorbs_intercept",
        "centering_mode",
        "edf",
        "smoothing_lambda",
        "spline",
        "knot_covariance",
        "smooth_curve",
        "level_is_special",
        "monotone",
        "monotone_repaired",
        "alpha",
    )

    def test_term_inference_field_order_is_append_only(self):
        """New fields go on the end; the released prefix never moves."""
        import dataclasses

        order = tuple(f.name for f in dataclasses.fields(TermInference))
        assert order[: len(self._RELEASED_FIELD_ORDER)] == self._RELEASED_FIELD_ORDER
        # Everything added since is genuinely new, not a rename of a released
        # field that was moved to the end to satisfy the check above.
        assert set(order[len(self._RELEASED_FIELD_ORDER) :]).isdisjoint(self._RELEASED_FIELD_ORDER)

    def test_a_released_positional_call_still_lands_its_arguments(self):
        """The failure the ordering check exists to prevent, spelled out.

        Written against v0.26.0: sixteen positional arguments, the sixteenth
        being ``edf``.  If a field is ever inserted ahead of ``edf`` again this
        assertion reports ``edf is None`` and a non-zero ``centering_shift``,
        which is the exact corruption rather than a field-name diff.
        """
        ti = TermInference(
            "f",  # name
            "categorical",  # kind
            True,  # active
            None,  # x
            ["a", "b"],  # levels
            np.array([0.0, 0.5]),  # log_relativity
            np.exp(np.array([0.0, 0.5])),  # relativity
            None,  # se_log_relativity
            None,  # ci_lower
            None,  # ci_upper
            None,  # ci_lower_simultaneous
            None,  # ci_upper_simultaneous
            None,  # critical_value_simultaneous
            True,  # absorbs_intercept
            "training_mean_zero_unweighted",  # centering_mode
            3.0,  # edf
        )
        assert ti.edf == 3.0
        assert ti.centering_shift == 0.0

    def test_the_appended_field_defaults_to_no_shift(self):
        """A caller that predates the field gets the value that means "untouched"."""
        ti = TermInference("f", "numeric", True)
        assert ti.centering_shift == 0.0
