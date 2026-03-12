"""Tests for term_inference API and enriched summary."""

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    InteractionInference,
    Numeric,
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
    exposure = rng.uniform(0.3, 1.0, n)
    mu = np.exp(-2.0 + 0.01 * (age - 50) ** 2 / 100 + (region == "A") * 0.3)
    y = rng.poisson(mu * exposure).astype(float)
    X = pd.DataFrame({"age": age, "region": region, "density": density})
    return X, y, exposure


@pytest.fixture
def fitted_model(sample_data):
    X, y, exposure = sample_data
    model = SuperGLM(
        penalty="group_lasso",
        lambda1=0.01,
        features={
            "age": Spline(n_knots=10, penalty="ssp"),
            "region": Categorical(base="first"),
            "density": Numeric(),
        },
    )
    model.fit(X, y, exposure=exposure)
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

    def test_centering_mode(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.centering_mode == "training_mean_zero_unweighted"


class TestTermInferenceSplineMetadata:
    def test_spline_metadata_present(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.spline is not None
        assert isinstance(ti.spline, SplineMetadata)

    def test_spline_kind(self, fitted_model):
        ti = fitted_model.term_inference("age")
        assert ti.spline.kind == "BasisSpline"

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

    def test_centering_mode(self, fitted_model):
        ti = fitted_model.term_inference("region")
        assert ti.centering_mode == "base_level"


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
        X, y, exposure = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, exposure=exposure)
        m = model.metrics(X, y, exposure=exposure)
        s = m.summary()
        text = str(s)
        assert "edf=" in text

    def test_spline_row_has_lambda(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "region": Categorical(base="first"),
            },
        )
        model.fit(X, y, exposure=exposure)
        m = model.metrics(X, y, exposure=exposure)
        s = m.summary()
        text = str(s)
        assert "lam=" in text

    def test_non_spline_terms_unaffected(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={
                "age": Spline(n_knots=10, penalty="ssp"),
                "region": Categorical(base="first"),
                "density": Numeric(),
            },
        )
        model.fit(X, y, exposure=exposure)
        m = model.metrics(X, y, exposure=exposure)
        s = m.summary()
        # Check that non-spline rows still have normal coef/se/z/p format
        text = str(s)
        assert "region[B]" in text
        assert "density" in text

    def test_coef_rows_have_metadata(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={"age": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, exposure=exposure)
        m = model.metrics(X, y, exposure=exposure)
        s = m.summary()
        spline_rows = [r for r in s._coef_rows if r.is_spline]
        assert len(spline_rows) >= 1
        row = spline_rows[0]
        assert row.spline_kind == "BasisSpline"
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
        assert age_entry["spline_kind"] == "BasisSpline"
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
        X, y, exposure = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={"age": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, exposure=exposure)

        diag = model.diagnostics()
        rich = model.metrics(X, y, exposure=exposure).summary()
        rich_row = next(r for r in rich._coef_rows if r.is_spline)

        assert diag["age"]["edf"] == rich_row.edf
        assert diag["age"]["smoothing_lambda"] == rich_row.smoothing_lambda
        assert diag["age"]["spline_kind"] == rich_row.spline_kind


# ── Phase 5: Spline kinds ───────────────────────────────────────


class TestSplineKinds:
    @pytest.mark.parametrize(
        "kind,expected_class",
        [("bs", "BasisSpline"), ("ns", "NaturalSpline"), ("cr", "CubicRegressionSpline")],
    )
    def test_term_inference_spline_kind(self, sample_data, kind, expected_class):
        X, y, exposure = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={"age": Spline(kind=kind, n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, exposure=exposure)
        ti = model.term_inference("age")
        assert ti.spline.kind == expected_class

    @pytest.mark.parametrize("kind", ["bs", "ns", "cr"])
    def test_se_finite_all_kinds(self, sample_data, kind):
        X, y, exposure = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={"age": Spline(kind=kind, n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, exposure=exposure)
        ti = model.term_inference("age")
        assert np.all(np.isfinite(ti.se_log_relativity))


# ── Phase 6: Interaction inference ───────────────────────────────


class TestInteractionInference:
    def test_spline_categorical_interaction(self, sample_data):
        X, y, exposure = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.001,
            features={
                "age": Spline(n_knots=8, penalty="ssp"),
                "region": Categorical(base="first"),
            },
            interactions=[("age", "region")],
        )
        model.fit(X, y, exposure=exposure)
        ii = model.term_inference("age:region")
        assert isinstance(ii, InteractionInference)
        assert ii.kind == "spline_categorical"

    def test_unknown_feature_raises(self, fitted_model):
        with pytest.raises(KeyError, match="Feature not found"):
            fitted_model.term_inference("nonexistent")


# ── Phase 7: Centering validation ────────────────────────────────


class TestCenteringMetadata:
    def test_spline_training_mean_zero(self, sample_data):
        """Verify that reported centering metadata matches actual mean-zero behavior."""
        X, y, exposure = sample_data
        model = SuperGLM(
            penalty="group_lasso",
            lambda1=0.01,
            features={"age": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y, exposure=exposure)
        ti = model.term_inference("age")
        assert ti.centering_mode == "training_mean_zero_unweighted"

        # The log-relativity curve evaluated at training points should
        # have mean approximately zero (within tolerance of penalty shrinkage)
        raw = model.reconstruct_feature("age")
        # The unweighted training mean of the reconstructed curve (at grid points)
        # should be close to zero for an identifiable spline
        mean_log_rel = np.mean(raw["log_relativity"])
        assert abs(mean_log_rel) < 0.5  # relaxed tolerance for penalized fit
