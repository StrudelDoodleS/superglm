"""Tests for all interaction features."""

import numpy as np
import pandas as pd
import pytest

from superglm.features.categorical import Categorical
from superglm.features.interaction import (
    CategoricalInteraction,
    NumericCategorical,
    NumericInteraction,
    PolynomialCategorical,
    PolynomialInteraction,
    SplineCategorical,
    TensorInteraction,
)
from superglm.features.numeric import Numeric
from superglm.features.polynomial import Polynomial
from superglm.features.spline import CubicRegressionSpline, NaturalSpline, Spline
from superglm.model import SuperGLM
from superglm.types import GroupInfo


# ── Fixture data ───────────────────────────────────────────────
@pytest.fixture
def interaction_data():
    """Synthetic dataset with multiple feature types."""
    rng = np.random.default_rng(42)
    n = 2000
    x_cont = rng.uniform(18, 80, n)
    x_num1 = rng.normal(100, 15, n)
    x_num2 = rng.normal(50, 10, n)
    x_cat = rng.choice(["A", "B", "C"], n)
    x_cat2 = rng.choice(["X", "Y", "Z"], n)

    mu = np.exp(
        -1.0
        + 0.01 * x_cont / 10
        + 0.3 * (x_cat == "B")
        + 0.5 * (x_cat == "C")
        + 0.2 * (x_cat == "B") * np.sin(x_cont / 20)
        + 0.1 * (x_cat2 == "Y") * (x_cat == "B")
        + 0.001 * x_num1
    )
    y = rng.poisson(mu)
    exposure = np.ones(n)

    X = pd.DataFrame(
        {
            "age": x_cont,
            "bm": x_num1,
            "density": x_num2,
            "region": x_cat,
            "type": x_cat2,
        }
    )
    return X, y, exposure


@pytest.fixture
def simple_model_data():
    """Minimal dataset for API tests."""
    rng = np.random.default_rng(123)
    n = 500
    x_cont = rng.uniform(0, 100, n)
    x_cat = rng.choice(["A", "B"], n)
    mu = np.exp(-0.5 + 0.005 * x_cont + 0.2 * (x_cat == "B"))
    y = rng.poisson(mu)

    X = pd.DataFrame({"x": x_cont, "cat": x_cat})
    return X, y


# ── SplineCategorical build tests ─────────────────────────────


class TestSplineCategoricalBuild:
    def test_correct_group_count(self):
        """One group per non-base level."""
        spline_spec = Spline(n_knots=10)
        cat_spec = Categorical(base="first")
        x_spline = np.linspace(0, 100, 500)
        x_cat = np.array(["A", "B", "C"] * 166 + ["A", "B"])

        spline_spec.build(x_spline)
        cat_spec.build(x_cat)

        sc = SplineCategorical("spline", "cat")
        groups = sc.build(x_spline, x_cat, {"spline": spline_spec, "cat": cat_spec})
        assert len(groups) == 2

    def test_group_names_and_n_cols(self):
        spline_spec = Spline(n_knots=5)
        cat_spec = Categorical(base="first")
        x_spline = np.linspace(0, 100, 300)
        x_cat = np.array(["A", "B", "C"] * 100)

        spline_spec.build(x_spline)
        cat_spec.build(x_cat)

        sc = SplineCategorical("spline", "cat")
        groups = sc.build(x_spline, x_cat, {"spline": spline_spec, "cat": cat_spec})
        for g in groups:
            assert g.n_cols == spline_spec._n_basis
            assert g.reparametrize is True
            assert g.penalty_matrix is not None

    def test_sparse_columns(self):
        import scipy.sparse as sp

        spline_spec = Spline(n_knots=5)
        cat_spec = Categorical(base="first")
        x_spline = np.linspace(0, 100, 300)
        x_cat = np.array(["A", "B", "C"] * 100)

        spline_spec.build(x_spline)
        cat_spec.build(x_cat)

        sc = SplineCategorical("spline", "cat")
        groups = sc.build(x_spline, x_cat, {"spline": spline_spec, "cat": cat_spec})

        B_level = groups[0].columns
        assert sp.issparse(B_level)
        mask_not_b = x_cat != "B"
        arr = B_level.toarray()
        np.testing.assert_array_equal(arr[mask_not_b], 0.0)

    def test_penalty_matches_parent(self):
        spline_spec = Spline(n_knots=8)
        cat_spec = Categorical(base="first")
        x = np.linspace(0, 100, 200)
        x_cat = np.array(["A", "B"] * 100)

        parent_info = spline_spec.build(x)
        cat_spec.build(x_cat)

        sc = SplineCategorical("spline", "cat")
        groups = sc.build(x, x_cat, {"spline": spline_spec, "cat": cat_spec})
        np.testing.assert_allclose(groups[0].penalty_matrix, parent_info.penalty_matrix, atol=1e-12)

    def test_parent_names(self):
        sc = SplineCategorical("age", "region")
        assert sc.parent_names == ("age", "region")


# ── SplineCategorical + natural build tests ───────────────────


class TestSplineCategoricalNaturalBuild:
    def test_natural_group_n_cols(self):
        """Per-level groups have K-2 columns when parent is NaturalSpline."""
        spline_spec = NaturalSpline(n_knots=10)
        cat_spec = Categorical(base="first")
        x_spline = np.linspace(0, 100, 500)
        x_cat = np.array(["A", "B", "C"] * 166 + ["A", "B"])

        spline_spec.build(x_spline)
        cat_spec.build(x_cat)

        sc = SplineCategorical("spline", "cat")
        groups = sc.build(x_spline, x_cat, {"spline": spline_spec, "cat": cat_spec})
        assert len(groups) == 2
        for g in groups:
            assert g.n_cols == spline_spec._n_basis - 2

    def test_natural_penalty_projected(self):
        """Per-level penalty is the projected (K-2, K-2) penalty."""
        spline_spec = NaturalSpline(n_knots=8)
        cat_spec = Categorical(base="first")
        x = np.linspace(0, 100, 200)
        x_cat = np.array(["A", "B"] * 100)

        parent_info = spline_spec.build(x)
        cat_spec.build(x_cat)

        sc = SplineCategorical("spline", "cat")
        groups = sc.build(x, x_cat, {"spline": spline_spec, "cat": cat_spec})
        np.testing.assert_allclose(groups[0].penalty_matrix, parent_info.penalty_matrix, atol=1e-12)


# ── PolynomialCategorical build tests ─────────────────────────


class TestPolynomialCategoricalBuild:
    def test_correct_group_count(self):
        poly_spec = Polynomial(degree=3)
        cat_spec = Categorical(base="first")
        x = np.linspace(0, 100, 300)
        x_cat = np.array(["A", "B", "C"] * 100)

        poly_spec.build(x)
        cat_spec.build(x_cat)

        pc = PolynomialCategorical("poly", "cat")
        groups = pc.build(x, x_cat, {"poly": poly_spec, "cat": cat_spec})
        assert len(groups) == 2  # B and C (A is base)

    def test_n_cols_equals_degree(self):
        poly_spec = Polynomial(degree=2)
        cat_spec = Categorical(base="first")
        x = np.linspace(0, 100, 200)
        x_cat = np.array(["A", "B"] * 100)

        poly_spec.build(x)
        cat_spec.build(x_cat)

        pc = PolynomialCategorical("poly", "cat")
        groups = pc.build(x, x_cat, {"poly": poly_spec, "cat": cat_spec})
        assert groups[0].n_cols == 2
        assert groups[0].penalty_matrix is None  # no penalty for polynomial

    def test_masking(self):
        poly_spec = Polynomial(degree=2)
        cat_spec = Categorical(base="first")
        x = np.linspace(0, 100, 200)
        x_cat = np.array(["A", "B"] * 100)

        poly_spec.build(x)
        cat_spec.build(x_cat)

        pc = PolynomialCategorical("poly", "cat")
        groups = pc.build(x, x_cat, {"poly": poly_spec, "cat": cat_spec})
        arr = groups[0].columns
        # Rows where cat != "B" should be zero
        mask_not_b = x_cat != "B"
        np.testing.assert_array_equal(arr[mask_not_b], 0.0)

    def test_parent_names(self):
        pc = PolynomialCategorical("age", "region")
        assert pc.parent_names == ("age", "region")


# ── NumericCategorical build tests ────────────────────────────


class TestNumericCategoricalBuild:
    def test_single_group(self):
        num_spec = Numeric()
        cat_spec = Categorical(base="first")
        x_num = np.random.default_rng(42).normal(0, 1, 300)
        x_cat = np.array(["A", "B", "C"] * 100)

        num_spec.build(x_num)
        cat_spec.build(x_cat)

        nc = NumericCategorical("num", "cat")
        info = nc.build(x_num, x_cat, {"num": num_spec, "cat": cat_spec})
        assert isinstance(info, GroupInfo)
        assert info.n_cols == 2  # B and C (A is base)

    def test_masking(self):
        num_spec = Numeric(standardize=False)
        cat_spec = Categorical(base="first")
        x_num = np.ones(200)
        x_cat = np.array(["A", "B"] * 100)

        num_spec.build(x_num)
        cat_spec.build(x_cat)

        nc = NumericCategorical("num", "cat")
        info = nc.build(x_num, x_cat, {"num": num_spec, "cat": cat_spec})
        arr = info.columns
        # For level B: column should be 1 where cat=="B", 0 elsewhere
        expected = (x_cat == "B").astype(float)
        np.testing.assert_array_equal(arr[:, 0], expected)

    def test_parent_names(self):
        nc = NumericCategorical("bm", "region")
        assert nc.parent_names == ("bm", "region")


# ── CategoricalInteraction build tests ────────────────────────


class TestCategoricalInteractionBuild:
    def test_single_group(self):
        cat1 = Categorical(base="first")
        cat2 = Categorical(base="first")
        x1 = np.array(["A", "B", "C"] * 100)
        x2 = np.array(["X", "Y"] * 150)

        cat1.build(x1)
        cat2.build(x2)

        ci = CategoricalInteraction("c1", "c2")
        info = ci.build(x1, x2, {"c1": cat1, "c2": cat2})
        assert isinstance(info, GroupInfo)

    def test_correct_n_cols(self):
        cat1 = Categorical(base="first")
        cat2 = Categorical(base="first")
        x1 = np.array(["A", "B", "C"] * 100)
        x2 = np.array(["X", "Y", "Z"] * 100)

        cat1.build(x1)
        cat2.build(x2)

        ci = CategoricalInteraction("c1", "c2")
        info = ci.build(x1, x2, {"c1": cat1, "c2": cat2})
        assert info.n_cols == 4  # 2 * 2

    def test_sparse_indicators(self):
        import scipy.sparse as sp

        cat1 = Categorical(base="first")
        cat2 = Categorical(base="first")
        x1 = np.array(["A", "B"] * 50)
        x2 = np.array(["X", "Y"] * 50)

        cat1.build(x1)
        cat2.build(x2)

        ci = CategoricalInteraction("c1", "c2")
        info = ci.build(x1, x2, {"c1": cat1, "c2": cat2})
        assert sp.issparse(info.columns)

        arr = info.columns.toarray()
        assert set(np.unique(arr)).issubset({0.0, 1.0})
        assert np.all(arr.sum(axis=1) <= 1)

    def test_pairs_correct(self):
        cat1 = Categorical(base="first")
        cat2 = Categorical(base="first")
        x1 = np.array(["A", "B", "C"] * 100)
        x2 = np.array(["X", "Y"] * 150)

        cat1.build(x1)
        cat2.build(x2)

        ci = CategoricalInteraction("c1", "c2")
        ci.build(x1, x2, {"c1": cat1, "c2": cat2})
        assert ci._pairs == [("B", "Y"), ("C", "Y")]

    def test_parent_names(self):
        ci = CategoricalInteraction("region", "type")
        assert ci.parent_names == ("region", "type")


# ── NumericInteraction build tests ────────────────────────────


class TestNumericInteractionBuild:
    def test_single_column(self):
        s1 = Numeric(standardize=False)
        s2 = Numeric(standardize=False)
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([4.0, 5.0, 6.0])

        s1.build(x1)
        s2.build(x2)

        ni = NumericInteraction("a", "b")
        info = ni.build(x1, x2, {"a": s1, "b": s2})
        assert info.n_cols == 1
        np.testing.assert_allclose(info.columns.ravel(), [4.0, 10.0, 18.0])

    def test_standardized_product(self):
        s1 = Numeric(standardize=True)
        s2 = Numeric(standardize=True)
        rng = np.random.default_rng(42)
        x1 = rng.normal(10, 2, 200)
        x2 = rng.normal(5, 1, 200)

        s1.build(x1)
        s2.build(x2)

        ni = NumericInteraction("a", "b")
        info = ni.build(x1, x2, {"a": s1, "b": s2})
        # Product of standardized values
        x1_s = (x1 - s1._mean) / s1._std
        x2_s = (x2 - s2._mean) / s2._std
        np.testing.assert_allclose(info.columns.ravel(), x1_s * x2_s)

    def test_parent_names(self):
        ni = NumericInteraction("bm", "density")
        assert ni.parent_names == ("bm", "density")


# ── PolynomialInteraction build tests ─────────────────────────


class TestPolynomialInteractionBuild:
    def test_n_cols(self):
        p1 = Polynomial(degree=2)
        p2 = Polynomial(degree=3)
        x1 = np.linspace(0, 100, 200)
        x2 = np.linspace(0, 50, 200)

        p1.build(x1)
        p2.build(x2)

        pi = PolynomialInteraction("a", "b")
        info = pi.build(x1, x2, {"a": p1, "b": p2})
        assert info.n_cols == 6  # 2 * 3

    def test_reconstruct_2d(self):
        p1 = Polynomial(degree=2)
        p2 = Polynomial(degree=2)
        x1 = np.linspace(0, 100, 200)
        x2 = np.linspace(0, 50, 200)

        p1.build(x1)
        p2.build(x2)

        pi = PolynomialInteraction("a", "b")
        pi.build(x1, x2, {"a": p1, "b": p2})

        beta = np.array([0.1, -0.05, 0.02, -0.01])
        raw = pi.reconstruct(beta, n_points=20)
        assert "x1" in raw
        assert "x2" in raw
        assert raw["log_relativity"].shape == (20, 20)
        assert raw["interaction"] is True

    def test_parent_names(self):
        pi = PolynomialInteraction("age", "vehage")
        assert pi.parent_names == ("age", "vehage")


# ── Model-level tests: SplineCategorical ──────────────────────


class TestSplineCategoricalModel:
    def test_fit_predict_roundtrip(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "region": Categorical()},
            interactions=[("age", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert np.all(pred > 0)

    def test_groups_created(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "region": Categorical()},
            interactions=[("age", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        igroups = [g for g in model._groups if g.feature_name == "age:region"]
        assert len(igroups) >= 1

    def test_high_lambda_zeros_interactions(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "region": Categorical()},
            interactions=[("age", "region")],
            lambda1=1e4,
        )
        model.fit(X, y, exposure=exposure)
        igroups = [g for g in model._groups if g.feature_name == "age:region"]
        for g in igroups:
            assert np.linalg.norm(model.result.beta[g.sl]) < 1e-10

    def test_reconstruct_per_level(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "region": Categorical()},
            interactions=[("age", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        raw = model.reconstruct_feature("age:region")
        assert raw["interaction"] is True
        assert "per_level" in raw
        assert "x" in raw

    def test_relativities_include_interaction(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "region": Categorical()},
            interactions=[("age", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        rels = model.relativities()
        assert "age" in rels
        assert "region" in rels
        interaction_keys = [k for k in rels if k.startswith("age:region")]
        assert len(interaction_keys) >= 1


# ── Model-level tests: Natural splines ────────────────────────


class TestNaturalSplineModel:
    def test_natural_spline_fit_predict(self, interaction_data):
        """Full model with NaturalSpline fits and predicts."""
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": NaturalSpline(n_knots=10), "region": Categorical()},
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert np.all(pred > 0)

    def test_natural_spline_reconstruct(self, interaction_data):
        """Relativity curve is reasonable with NaturalSpline."""
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": NaturalSpline(n_knots=10), "region": Categorical()},
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        raw = model.reconstruct_feature("age")
        assert "x" in raw
        assert "relativity" in raw
        # Relativities should be finite and positive
        assert np.all(np.isfinite(raw["relativity"]))
        assert np.all(raw["relativity"] > 0)

    def test_spline_cat_natural_fit_predict(self, interaction_data):
        """SplineCategorical with NaturalSpline parent works end-to-end."""
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": NaturalSpline(n_knots=10), "region": Categorical()},
            interactions=[("age", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert np.all(pred > 0)

    def test_spline_cat_natural_reconstruct(self, interaction_data):
        """Per-level reconstruct with NaturalSpline produces sensible curves."""
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": NaturalSpline(n_knots=10), "region": Categorical()},
            interactions=[("age", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        raw = model.reconstruct_feature("age:region")
        assert raw["interaction"] is True
        assert "per_level" in raw
        for level_data in raw["per_level"].values():
            assert np.all(np.isfinite(level_data["relativity"]))
            assert np.all(level_data["relativity"] > 0)

    def test_spline_cat_natural_spline_class(self, interaction_data):
        """SplineCategorical accepts NaturalSpline as parent."""
        spline_spec = NaturalSpline(n_knots=10)
        cat_spec = Categorical(base="first")
        x_spline = np.linspace(0, 100, 500)
        x_cat = np.array(["A", "B", "C"] * 166 + ["A", "B"])

        spline_spec.build(x_spline)
        cat_spec.build(x_cat)

        sc = SplineCategorical("spline", "cat")
        groups = sc.build(x_spline, x_cat, {"spline": spline_spec, "cat": cat_spec})
        assert len(groups) == 2


# ── Model-level tests: CubicRegressionSpline ──────────────────


class TestCubicRegressionSplineModel:
    def test_cr_spline_fit_predict(self, interaction_data):
        """Full model with CubicRegressionSpline fits and predicts."""
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": CubicRegressionSpline(n_knots=10), "region": Categorical()},
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert np.all(pred > 0)

    def test_cr_spline_reconstruct(self, interaction_data):
        """Relativity curve is finite and positive."""
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": CubicRegressionSpline(n_knots=10), "region": Categorical()},
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        raw = model.reconstruct_feature("age")
        assert "x" in raw
        assert "relativity" in raw
        assert np.all(np.isfinite(raw["relativity"]))
        assert np.all(raw["relativity"] > 0)


# ── Model-level tests: PolynomialCategorical ──────────────────


class TestPolynomialCategoricalModel:
    def test_fit_predict_roundtrip(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Polynomial(degree=3), "region": Categorical()},
            interactions=[("age", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert np.all(pred > 0)

    def test_per_level_groups(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Polynomial(degree=3), "region": Categorical()},
            interactions=[("age", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        igroups = [g for g in model._groups if g.feature_name == "age:region"]
        assert len(igroups) >= 1

    def test_reconstruct_per_level(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Polynomial(degree=3), "region": Categorical()},
            interactions=[("age", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        raw = model.reconstruct_feature("age:region")
        assert raw["interaction"] is True
        assert "per_level" in raw


# ── Model-level tests: NumericCategorical ─────────────────────


class TestNumericCategoricalModel:
    def test_fit_predict_roundtrip(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"bm": Numeric(), "region": Categorical()},
            interactions=[("bm", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert np.all(pred > 0)

    def test_reconstruct_per_level_slopes(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"bm": Numeric(), "region": Categorical()},
            interactions=[("bm", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        raw = model.reconstruct_feature("bm:region")
        assert raw["interaction"] is True
        assert "relativities_per_unit" in raw

    def test_relativities_include_interaction(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"bm": Numeric(), "region": Categorical()},
            interactions=[("bm", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        rels = model.relativities()
        assert "bm:region" in rels

    def test_auto_detect_swaps_order(self, interaction_data):
        """Categorical first, Numeric second → still puts numeric first."""
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"bm": Numeric(), "region": Categorical()},
            interactions=[("region", "bm")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        assert "bm:region" in model._interaction_specs
        assert isinstance(model._interaction_specs["bm:region"], NumericCategorical)


# ── Model-level tests: CategoricalInteraction ─────────────────


class TestCategoricalInteractionModel:
    def test_fit_predict_roundtrip(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"region": Categorical(), "type": Categorical()},
            interactions=[("region", "type")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert np.all(pred > 0)

    def test_high_lambda_zeros_interaction(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"region": Categorical(), "type": Categorical()},
            interactions=[("region", "type")],
            lambda1=100.0,
        )
        model.fit(X, y, exposure=exposure)
        igroups = [g for g in model._groups if g.feature_name == "region:type"]
        for g in igroups:
            assert np.linalg.norm(model.result.beta[g.sl]) < 1e-10

    def test_reconstruct_pairs(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"region": Categorical(), "type": Categorical()},
            interactions=[("region", "type")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        raw = model.reconstruct_feature("region:type")
        assert raw["interaction"] is True
        assert "pairs" in raw

    def test_relativities_include_interaction(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"region": Categorical(), "type": Categorical()},
            interactions=[("region", "type")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        rels = model.relativities()
        assert "region:type" in rels


# ── Model-level tests: NumericInteraction ─────────────────────


class TestNumericInteractionModel:
    def test_fit_predict_roundtrip(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"bm": Numeric(), "density": Numeric()},
            interactions=[("bm", "density")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert np.all(pred > 0)

    def test_reconstruct(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"bm": Numeric(), "density": Numeric()},
            interactions=[("bm", "density")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        raw = model.reconstruct_feature("bm:density")
        assert raw["interaction"] is True
        assert "coef_original" in raw

    def test_relativities_include_interaction(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"bm": Numeric(), "density": Numeric()},
            interactions=[("bm", "density")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        rels = model.relativities()
        assert "bm:density" in rels


# ── Model-level tests: PolynomialInteraction ──────────────────


class TestPolynomialInteractionModel:
    def test_fit_predict_roundtrip(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Polynomial(degree=2), "bm": Polynomial(degree=2)},
            interactions=[("age", "bm")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert np.all(pred > 0)

    def test_reconstruct_2d(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Polynomial(degree=2), "bm": Polynomial(degree=2)},
            interactions=[("age", "bm")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        raw = model.reconstruct_feature("age:bm")
        assert raw["interaction"] is True
        assert "x1" in raw
        assert "x2" in raw

    def test_relativities_skips_2d(self, interaction_data):
        """2D surface interactions don't appear in relativities()."""
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Polynomial(degree=2), "bm": Polynomial(degree=2)},
            interactions=[("age", "bm")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        rels = model.relativities()
        # 2D surface not representable as DataFrame
        assert "age:bm" not in rels


# ── API tests ─────────────────────────────────────────────────


class TestInteractionAPI:
    def test_add_interaction_via_constructor(self, simple_model_data):
        X, y = simple_model_data
        model = SuperGLM(
            features={"x": Spline(n_knots=5), "cat": Categorical()},
            interactions=[("x", "cat")],
        )
        model.fit(X, y)
        assert "x:cat" in model._interaction_specs

    def test_constructor_param(self, simple_model_data):
        X, y = simple_model_data
        model = SuperGLM(
            features={"x": Spline(n_knots=5), "cat": Categorical()},
            interactions=[("x", "cat")],
        )
        model.fit(X, y)
        assert "x:cat" in model._interaction_specs

    def test_auto_detect_spline_categorical(self, simple_model_data):
        X, y = simple_model_data
        model = SuperGLM(
            features={"x": Spline(n_knots=5), "cat": Categorical()},
            interactions=[("x", "cat")],
        )
        model.fit(X, y)
        assert isinstance(model._interaction_specs["x:cat"], SplineCategorical)

    def test_auto_detect_swaps_spline_cat(self, simple_model_data):
        X, y = simple_model_data
        model = SuperGLM(
            features={"x": Spline(n_knots=5), "cat": Categorical()},
            interactions=[("cat", "x")],
        )
        model.fit(X, y)
        assert "x:cat" in model._interaction_specs

    def test_auto_detect_poly_cat(self, simple_model_data):
        X, y = simple_model_data
        X = X.rename(columns={"x": "p", "cat": "c"})
        model = SuperGLM(
            features={"p": Polynomial(degree=2), "c": Categorical()},
            interactions=[("p", "c")],
        )
        model.fit(X, y)
        assert isinstance(model._interaction_specs["p:c"], PolynomialCategorical)

    def test_auto_detect_swaps_poly_cat(self, simple_model_data):
        X, y = simple_model_data
        X = X.rename(columns={"x": "p", "cat": "c"})
        model = SuperGLM(
            features={"p": Polynomial(degree=2), "c": Categorical()},
            interactions=[("c", "p")],
        )
        model.fit(X, y)
        assert "p:c" in model._interaction_specs

    def test_auto_detect_num_cat(self, simple_model_data):
        X, y = simple_model_data
        X = X.rename(columns={"x": "n", "cat": "c"})
        model = SuperGLM(
            features={"n": Numeric(), "c": Categorical()},
            interactions=[("n", "c")],
        )
        model.fit(X, y)
        assert isinstance(model._interaction_specs["n:c"], NumericCategorical)

    def test_auto_detect_swaps_num_cat(self, simple_model_data):
        X, y = simple_model_data
        X = X.rename(columns={"x": "n", "cat": "c"})
        model = SuperGLM(
            features={"n": Numeric(), "c": Categorical()},
            interactions=[("c", "n")],
        )
        model.fit(X, y)
        assert "n:c" in model._interaction_specs

    def test_auto_detect_cat_cat(self, simple_model_data):
        X, y = simple_model_data
        X["a"] = X["cat"]
        X["b"] = np.where(X["cat"] == "A", "X", "Y")
        model = SuperGLM(
            features={"a": Categorical(), "b": Categorical()},
            interactions=[("a", "b")],
        )
        model.fit(X, y)
        assert isinstance(model._interaction_specs["a:b"], CategoricalInteraction)

    def test_auto_detect_num_num(self, simple_model_data):
        X, y = simple_model_data
        X["a"] = X["x"]
        X["b"] = X["x"] * 2.0
        model = SuperGLM(
            features={"a": Numeric(), "b": Numeric()},
            interactions=[("a", "b")],
        )
        model.fit(X, y)
        assert isinstance(model._interaction_specs["a:b"], NumericInteraction)

    def test_auto_detect_poly_poly(self, simple_model_data):
        X, y = simple_model_data
        X["a"] = X["x"]
        X["b"] = X["x"] * 2.0
        model = SuperGLM(
            features={"a": Polynomial(degree=2), "b": Polynomial(degree=3)},
            interactions=[("a", "b")],
        )
        model.fit(X, y)
        assert isinstance(model._interaction_specs["a:b"], PolynomialInteraction)

    def test_error_unknown_parent(self, simple_model_data):
        X, y = simple_model_data
        X = X.rename(columns={"cat": "a"})
        model = SuperGLM(
            features={"a": Categorical()},
            interactions=[("a", "nonexistent")],
        )
        with pytest.raises(ValueError, match="Parent feature not found"):
            model.fit(X, y)

    def test_auto_detect_spline_spline(self, simple_model_data):
        X, y = simple_model_data
        X["a"] = X["x"]
        X["b"] = X["x"] * 2.0
        model = SuperGLM(
            features={"a": Spline(n_knots=5), "b": Spline(n_knots=5)},
            interactions=[("a", "b")],
        )
        model.fit(X, y)
        assert isinstance(model._interaction_specs["a:b"], TensorInteraction)

    def test_error_spline_numeric(self, simple_model_data):
        X, y = simple_model_data
        X["a"] = X["x"]
        X["b"] = X["x"] * 2.0
        model = SuperGLM(
            features={"a": Spline(n_knots=5), "b": Numeric()},
            interactions=[("a", "b")],
        )
        with pytest.raises(TypeError, match="Cannot create interaction"):
            model.fit(X, y)

    def test_duplicate_interaction_in_constructor(self, simple_model_data):
        """Duplicate tuples in interactions list are silently deduplicated at fit time."""
        X, y = simple_model_data
        model = SuperGLM(
            features={"x": Spline(n_knots=5), "cat": Categorical()},
            interactions=[("x", "cat"), ("x", "cat")],
        )
        model.fit(X, y)
        assert "x:cat" in model._interaction_specs


# ── Summary tests ─────────────────────────────────────────────


class TestInteractionSummary:
    def test_spline_cat_summary(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "region": Categorical()},
            interactions=[("age", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        m = model.metrics(X, y, exposure=exposure)
        s = m.summary()
        assert "age:region" in str(s)

    def test_cat_cat_summary(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"region": Categorical(), "type": Categorical()},
            interactions=[("region", "type")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        m = model.metrics(X, y, exposure=exposure)
        s = m.summary()
        assert "region:type" in str(s)

    def test_num_cat_summary(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"bm": Numeric(), "region": Categorical()},
            interactions=[("bm", "region")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        m = model.metrics(X, y, exposure=exposure)
        s = m.summary()
        assert "bm:region" in str(s)

    def test_num_num_summary(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"bm": Numeric(), "density": Numeric()},
            interactions=[("bm", "density")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        m = model.metrics(X, y, exposure=exposure)
        s = m.summary()
        assert "bm:density" in str(s)

    def test_poly_poly_summary(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Polynomial(degree=2), "bm": Polynomial(degree=2)},
            interactions=[("age", "bm")],
            lambda1=0.01,
        )
        model.fit(X, y, exposure=exposure)
        m = model.metrics(X, y, exposure=exposure)
        s = m.summary()
        assert "age:bm" in str(s)


# ── No-overhead test ──────────────────────────────────────────


class TestNoOverhead:
    def test_model_without_interactions_unchanged(self, simple_model_data):
        X, y = simple_model_data
        model = SuperGLM(
            features={"x": Spline(n_knots=5), "cat": Categorical()},
            lambda1=0.01,
        )
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert model._interaction_order == []
        assert model._interaction_specs == {}


# ── TensorInteraction build tests ─────────────────────────────


class TestTensorInteractionBuild:
    def test_build_shape(self):
        """GroupInfo columns have shape (n, k1*k2)."""
        s1 = Spline(n_knots=5)
        s2 = Spline(n_knots=5)
        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 100, 500)
        x2 = rng.uniform(0, 50, 500)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b", n_knots=(5, 5))
        info = ti.build(x1, x2, {"a": s1, "b": s2})

        # k = n_knots + degree + 1 = 5 + 3 + 1 = 9 per margin
        assert info.columns.shape == (500, 9 * 9)
        assert info.n_cols == 81

    def test_penalty_full_rank(self):
        """ti-style penalty must be full-rank (no null space)."""
        s1 = Spline(n_knots=5)
        s2 = Spline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b", n_knots=(5, 5))
        info = ti.build(x1, x2, {"a": s1, "b": s2})

        eigvals = np.linalg.eigvalsh(info.penalty_matrix)
        assert np.all(eigvals > 1e-10), "ti penalty must be full-rank"

    def test_kronecker_values(self):
        """T[i,:] = B1[i,:] ⊗ B2[i,:] for random rows."""
        s1 = Spline(n_knots=3)
        s2 = Spline(n_knots=3)
        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 100, 200)
        x2 = rng.uniform(0, 50, 200)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b", n_knots=(3, 3))
        info = ti.build(x1, x2, {"a": s1, "b": s2})
        T = info.columns.toarray()

        # Verify Kronecker structure for a few rows
        from scipy.interpolate import BSpline as BSpl

        x1_clip = np.clip(x1, ti._knots1[0], ti._knots1[-1])
        x2_clip = np.clip(x2, ti._knots2[0], ti._knots2[-1])
        B1 = BSpl.design_matrix(x1_clip, ti._knots1, ti._degree).toarray()
        B2 = BSpl.design_matrix(x2_clip, ti._knots2, ti._degree).toarray()

        for i in [0, 50, 150]:
            expected = np.kron(B1[i], B2[i])
            np.testing.assert_allclose(T[i], expected, atol=1e-12)

    def test_sparse_output(self):
        """Build returns sparse columns."""
        import scipy.sparse as sp

        s1 = Spline(n_knots=5)
        s2 = Spline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b")
        info = ti.build(x1, x2, {"a": s1, "b": s2})
        assert sp.issparse(info.columns)

    def test_parent_names(self):
        ti = TensorInteraction("age", "vehage")
        assert ti.parent_names == ("age", "vehage")

    def test_custom_knots(self):
        """n_knots parameter controls resolution."""
        s1 = Spline(n_knots=5)
        s2 = Spline(n_knots=5)
        x1 = np.linspace(0, 100, 200)
        x2 = np.linspace(0, 50, 200)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b", n_knots=(3, 4))
        info = ti.build(x1, x2, {"a": s1, "b": s2})
        # k1 = 3+3+1 = 7, k2 = 4+3+1 = 8
        assert info.n_cols == 7 * 8

    def test_reconstruct_2d(self):
        """Reconstruct returns 2D surface grid."""
        s1 = Spline(n_knots=5)
        s2 = Spline(n_knots=5)
        x1 = np.linspace(0, 100, 200)
        x2 = np.linspace(0, 50, 200)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b", n_knots=(5, 5))
        info = ti.build(x1, x2, {"a": s1, "b": s2})

        beta = np.zeros(info.n_cols)
        beta[0] = 0.1
        raw = ti.reconstruct(beta, n_points=20)

        assert "x1" in raw
        assert "x2" in raw
        assert raw["log_relativity"].shape == (20, 20)
        assert raw["relativity"].shape == (20, 20)
        assert raw["interaction"] is True


# ── Model-level tests: TensorInteraction ──────────────────────


class TestTensorInteractionModel:
    def test_fit_predict_roundtrip(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "bm": Spline(n_knots=5)},
            interactions=[("age", "bm")],
            lambda1=0.1,
        )
        model.fit(X, y, exposure=exposure)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert np.all(pred > 0)

    def test_groups_created(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "bm": Spline(n_knots=5)},
            interactions=[("age", "bm")],
            lambda1=0.1,
        )
        model.fit(X, y, exposure=exposure)
        igroups = [g for g in model._groups if g.feature_name == "age:bm"]
        assert len(igroups) == 1

    def test_high_lambda_zeros_interaction(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "bm": Spline(n_knots=5)},
            interactions=[("age", "bm")],
            lambda1=1e4,
        )
        model.fit(X, y, exposure=exposure)
        igroups = [g for g in model._groups if g.feature_name == "age:bm"]
        for g in igroups:
            assert np.linalg.norm(model.result.beta[g.sl]) < 1e-10

    def test_reconstruct_2d_surface(self, interaction_data):
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "bm": Spline(n_knots=5)},
            interactions=[("age", "bm")],
            lambda1=0.1,
        )
        model.fit(X, y, exposure=exposure)
        raw = model.reconstruct_feature("age:bm")
        assert raw["interaction"] is True
        assert "x1" in raw
        assert "x2" in raw
        assert raw["log_relativity"].ndim == 2

    def test_auto_dispatch_spline_spline(self, interaction_data):
        X, y, exposure = interaction_data
        X["a"] = X["age"]
        X["b"] = X["bm"]
        model = SuperGLM(
            features={"a": Spline(n_knots=5), "b": Spline(n_knots=5)},
            interactions=[("a", "b")],
            lambda1=0.1,
        )
        model.fit(X, y, exposure=exposure)
        assert isinstance(model._interaction_specs["a:b"], TensorInteraction)

    def test_auto_dispatch_with_kwargs(self):
        model = SuperGLM(features={"a": Spline(n_knots=5), "b": Spline(n_knots=5)})
        model._add_interaction("a", "b", n_knots=(3, 4))
        ispec = model._interaction_specs["a:b"]
        assert isinstance(ispec, TensorInteraction)
        assert ispec._n_knots == (3, 4)

    def test_relativities_skips_2d(self, interaction_data):
        """2D tensor interaction doesn't appear in relativities() (like PolynomialInteraction)."""
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "bm": Spline(n_knots=5)},
            interactions=[("age", "bm")],
            lambda1=0.1,
        )
        model.fit(X, y, exposure=exposure)
        rels = model.relativities()
        assert "age:bm" not in rels

    def test_natural_spline_parents(self, interaction_data):
        """TensorInteraction works with NaturalSpline parents."""
        X, y, exposure = interaction_data
        model = SuperGLM(
            features={"age": NaturalSpline(n_knots=10), "bm": NaturalSpline(n_knots=5)},
            interactions=[("age", "bm")],
            lambda1=0.1,
        )
        model.fit(X, y, exposure=exposure)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert np.all(pred > 0)
