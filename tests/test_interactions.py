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
    n = 1000
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
    sample_weight = np.ones(n)

    X = pd.DataFrame(
        {
            "age": x_cont,
            "bm": x_num1,
            "density": x_num2,
            "region": x_cat,
            "type": x_cat2,
        }
    )
    return X, y, sample_weight


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
            assert g.n_cols == spline_spec._n_basis - 1  # identifiability
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
        # B_level stays sparse (projection is on GroupInfo, not applied to columns)
        assert sp.issparse(B_level)
        # Projection is set for dm_builder to fold into R_inv
        assert groups[0].projection is not None
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


class TestSplineCategoricalNaturalBuild:
    def test_natural_group_n_cols(self):
        """Per-level groups have K-3 columns when parent is NaturalSpline."""
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
            assert g.n_cols == spline_spec._n_basis - 3

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
        num_spec = Numeric()
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


class TestNumericInteractionBuild:
    def test_single_column(self):
        s1 = Numeric()
        s2 = Numeric()
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([4.0, 5.0, 6.0])

        s1.build(x1)
        s2.build(x2)

        ni = NumericInteraction("a", "b")
        info = ni.build(x1, x2, {"a": s1, "b": s2})
        assert info.n_cols == 1
        np.testing.assert_allclose(info.columns.ravel(), [4.0, 10.0, 18.0])

    def test_standardized_product(self):
        s1 = Numeric()
        s2 = Numeric()
        rng = np.random.default_rng(42)
        x1 = rng.normal(10, 2, 200)
        x2 = rng.normal(5, 1, 200)

        s1.build(x1)
        s2.build(x2)

        ni = NumericInteraction("a", "b")
        info = ni.build(x1, x2, {"a": s1, "b": s2})
        # Product of raw values (no standardization)
        np.testing.assert_allclose(info.columns.ravel(), x1 * x2)

    def test_parent_names(self):
        ni = NumericInteraction("bm", "density")
        assert ni.parent_names == ("bm", "density")


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


# ══════════════════════════════════════════════════════════════
# Parameterized model integration tests
# ══════════════════════════════════════════════════════════════
# Replace 8 per-type model test classes that had identical
# fit/predict/groups/reconstruct patterns.

# Factory approach — specs are mutable and interactions list is mutated
# by build_design_matrix, so each test needs fresh objects.


def _make_case(case_id: str):
    """Return (features_dict, interactions_list, iname) with fresh objects."""
    cases = {
        "spline_cat": (
            {"age": Spline(n_knots=10), "region": Categorical()},
            [("age", "region")],
            "age:region",
        ),
        "natural_spline_cat": (
            {"age": NaturalSpline(n_knots=10), "region": Categorical()},
            [("age", "region")],
            "age:region",
        ),
        "crs_cat": (
            {"age": CubicRegressionSpline(n_knots=10), "region": Categorical()},
            [("age", "region")],
            "age:region",
        ),
        "poly_cat": (
            {"age": Polynomial(degree=3), "region": Categorical()},
            [("age", "region")],
            "age:region",
        ),
        "num_cat": (
            {"bm": Numeric(), "region": Categorical()},
            [("bm", "region")],
            "bm:region",
        ),
        "cat_cat": (
            {"region": Categorical(), "type": Categorical()},
            [("region", "type")],
            "region:type",
        ),
        "num_num": (
            {"bm": Numeric(), "density": Numeric()},
            [("bm", "density")],
            "bm:density",
        ),
        "poly_poly": (
            {"age": Polynomial(degree=2), "bm": Polynomial(degree=2)},
            [("age", "bm")],
            "age:bm",
        ),
        "tensor": (
            {"age": Spline(n_knots=10), "bm": Spline(n_knots=5)},
            [("age", "bm")],
            "age:bm",
        ),
    }
    return cases[case_id]


_ALL_CASE_IDS = [
    "spline_cat",
    "natural_spline_cat",
    "crs_cat",
    "poly_cat",
    "num_cat",
    "cat_cat",
    "num_num",
    "poly_poly",
    "tensor",
]

_MODEL_CASES = [pytest.param(k, id=k) for k in _ALL_CASE_IDS]


class TestInteractionModelFitPredict:
    @pytest.mark.parametrize("case_id", _MODEL_CASES)
    def test_fit_predict_roundtrip(self, interaction_data, case_id):
        features, interactions, iname = _make_case(case_id)
        X, y, sample_weight = interaction_data
        model = SuperGLM(features=features, interactions=interactions, selection_penalty=0.01)
        model.fit(X, y, sample_weight=sample_weight)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert np.all(pred > 0)

    @pytest.mark.parametrize("case_id", _MODEL_CASES)
    def test_groups_created(self, interaction_data, case_id):
        features, interactions, iname = _make_case(case_id)
        X, y, sample_weight = interaction_data
        model = SuperGLM(features=features, interactions=interactions, selection_penalty=0.01)
        model.fit(X, y, sample_weight=sample_weight)
        igroups = [g for g in model._groups if g.feature_name == iname]
        assert len(igroups) >= 1

    @pytest.mark.parametrize("case_id", _MODEL_CASES)
    def test_reconstruct_interaction(self, interaction_data, case_id):
        features, interactions, iname = _make_case(case_id)
        X, y, sample_weight = interaction_data
        model = SuperGLM(features=features, interactions=interactions, selection_penalty=0.01)
        model.fit(X, y, sample_weight=sample_weight)
        raw = model.reconstruct_feature(iname)
        assert raw["interaction"] is True


# High lambda should zero out interaction groups
_HIGH_LAMBDA_CASES = [pytest.param(k, id=k) for k in ("spline_cat", "cat_cat", "tensor")]


class TestInteractionHighLambda:
    @pytest.mark.parametrize("case_id", _HIGH_LAMBDA_CASES)
    def test_high_lambda_zeros_interaction(self, interaction_data, case_id):
        features, interactions, iname = _make_case(case_id)
        X, y, sample_weight = interaction_data
        model = SuperGLM(features=features, interactions=interactions, selection_penalty=1e4)
        model.fit(X, y, sample_weight=sample_weight)
        igroups = [g for g in model._groups if g.feature_name == iname]
        for g in igroups:
            assert np.linalg.norm(model.result.beta[g.sl]) < 1e-10


# Interactions that appear in relativities (1D per-level or scalar)
_RELATIVITIES_INCLUDE = [
    pytest.param(k, id=k) for k in ("spline_cat", "num_cat", "cat_cat", "num_num")
]
_RELATIVITIES_SKIP = [pytest.param(k, id=k) for k in ("poly_poly", "tensor")]


class TestInteractionRelativities:
    @pytest.mark.parametrize("case_id", _RELATIVITIES_INCLUDE)
    def test_relativities_include_interaction(self, interaction_data, case_id):
        features, interactions, iname = _make_case(case_id)
        X, y, sample_weight = interaction_data
        model = SuperGLM(features=features, interactions=interactions, selection_penalty=0.01)
        model.fit(X, y, sample_weight=sample_weight)
        rels = model.relativities()
        interaction_keys = [k for k in rels if k.startswith(iname)]
        assert len(interaction_keys) >= 1

    @pytest.mark.parametrize("case_id", _RELATIVITIES_SKIP)
    def test_relativities_skip_2d(self, interaction_data, case_id):
        features, interactions, iname = _make_case(case_id)
        X, y, sample_weight = interaction_data
        model = SuperGLM(features=features, interactions=interactions, selection_penalty=0.01)
        model.fit(X, y, sample_weight=sample_weight)
        rels = model.relativities()
        assert iname not in rels


# ── Type-specific reconstruct assertions ──────────────────────
# These test assertions unique to each interaction type that
# can't be parameterized (different reconstruct dict keys).


class TestSplineCategoricalPerLevel:
    def test_reconstruct_per_level(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "region": Categorical()},
            interactions=[("age", "region")],
            selection_penalty=0.01,
        )
        model.fit(X, y, sample_weight=sample_weight)
        raw = model.reconstruct_feature("age:region")
        assert "per_level" in raw
        assert "x" in raw


class TestNaturalSplineModelSpecifics:
    def test_natural_spline_reconstruct(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"age": NaturalSpline(n_knots=10), "region": Categorical()},
            selection_penalty=0.01,
        )
        model.fit(X, y, sample_weight=sample_weight)
        raw = model.reconstruct_feature("age")
        assert "x" in raw and "relativity" in raw
        assert np.all(np.isfinite(raw["relativity"]))
        assert np.all(raw["relativity"] > 0)

    def test_spline_cat_natural_fit_predict(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"age": NaturalSpline(n_knots=10), "region": Categorical()},
            interactions=[("age", "region")],
            selection_penalty=0.01,
        )
        model.fit(X, y, sample_weight=sample_weight)
        pred = model.predict(X)
        assert pred.shape == (len(y),) and np.all(pred > 0)

    def test_spline_cat_natural_reconstruct(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"age": NaturalSpline(n_knots=10), "region": Categorical()},
            interactions=[("age", "region")],
            selection_penalty=0.01,
        )
        model.fit(X, y, sample_weight=sample_weight)
        raw = model.reconstruct_feature("age:region")
        assert raw["interaction"] is True
        assert "per_level" in raw

    def test_spline_cat_natural_spline_class(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"age": NaturalSpline(n_knots=10), "region": Categorical()},
            interactions=[("age", "region")],
            selection_penalty=0.01,
        )
        model.fit(X, y, sample_weight=sample_weight)
        assert isinstance(model._interaction_specs["age:region"], SplineCategorical)
        assert isinstance(model._specs["age"], NaturalSpline)


class TestCubicRegressionSplineModelSpecifics:
    def test_cr_spline_reconstruct(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"age": CubicRegressionSpline(n_knots=10), "region": Categorical()},
            selection_penalty=0.01,
        )
        model.fit(X, y, sample_weight=sample_weight)
        raw = model.reconstruct_feature("age")
        assert "x" in raw and "relativity" in raw
        assert np.all(np.isfinite(raw["relativity"]))
        assert np.all(raw["relativity"] > 0)


class TestNumericCategoricalModelSpecifics:
    def test_reconstruct_per_level_slopes(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"bm": Numeric(), "region": Categorical()},
            interactions=[("bm", "region")],
            selection_penalty=0.01,
        )
        model.fit(X, y, sample_weight=sample_weight)
        raw = model.reconstruct_feature("bm:region")
        assert "relativities_per_unit" in raw

    def test_auto_detect_swaps_order(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"bm": Numeric(), "region": Categorical()},
            interactions=[("region", "bm")],
            selection_penalty=0.01,
        )
        model.fit(X, y, sample_weight=sample_weight)
        assert "bm:region" in model._interaction_specs
        assert isinstance(model._interaction_specs["bm:region"], NumericCategorical)


class TestCategoricalInteractionModelSpecifics:
    def test_reconstruct_pairs(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"region": Categorical(), "type": Categorical()},
            interactions=[("region", "type")],
            selection_penalty=0.01,
        )
        model.fit(X, y, sample_weight=sample_weight)
        raw = model.reconstruct_feature("region:type")
        assert "pairs" in raw


class TestNumericInteractionModelSpecifics:
    def test_reconstruct(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"bm": Numeric(), "density": Numeric()},
            interactions=[("bm", "density")],
            selection_penalty=0.01,
        )
        model.fit(X, y, sample_weight=sample_weight)
        raw = model.reconstruct_feature("bm:density")
        assert "coef" in raw


class TestPolynomialInteractionModelSpecifics:
    def test_reconstruct_2d(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"age": Polynomial(degree=2), "bm": Polynomial(degree=2)},
            interactions=[("age", "bm")],
            selection_penalty=0.01,
        )
        model.fit(X, y, sample_weight=sample_weight)
        raw = model.reconstruct_feature("age:bm")
        assert raw["log_relativity"].ndim == 2


# ── Interaction API tests ─────────────────────────────────────


class TestInteractionAPI:
    def test_add_interaction_via_constructor(self, simple_model_data):
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
        X, y = simple_model_data
        model = SuperGLM(
            features={"x": Spline(n_knots=5), "cat": Categorical()},
            interactions=[("x", "cat"), ("x", "cat")],
        )
        model.fit(X, y)
        assert "x:cat" in model._interaction_specs


# ── Summary tests (parameterized) ────────────────────────────

_SUMMARY_CASES = [
    pytest.param(k, id=k) for k in ("spline_cat", "cat_cat", "num_cat", "num_num", "poly_poly")
]


class TestInteractionSummary:
    @pytest.mark.parametrize("case_id", _SUMMARY_CASES)
    def test_interaction_in_summary(self, interaction_data, case_id):
        features, interactions, iname = _make_case(case_id)
        X, y, sample_weight = interaction_data
        model = SuperGLM(features=features, interactions=interactions, selection_penalty=0.01)
        model.fit(X, y, sample_weight=sample_weight)
        m = model.metrics(X, y, sample_weight=sample_weight)
        assert iname in str(m.summary())


# ── No-overhead test ──────────────────────────────────────────


class TestNoOverhead:
    def test_model_without_interactions_unchanged(self, simple_model_data):
        X, y = simple_model_data
        model = SuperGLM(
            features={"x": Spline(n_knots=5), "cat": Categorical()},
            selection_penalty=0.01,
        )
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert model._interaction_order == []
        assert model._interaction_specs == {}


# ── TensorInteraction build tests ─────────────────────────────


class TestTensorInteractionBuild:
    def test_build_shape(self):
        s1 = Spline(n_knots=5)
        s2 = Spline(n_knots=5)
        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 100, 500)
        x2 = rng.uniform(0, 50, 500)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b", n_knots=(5, 5))
        info = ti.build(x1, x2, {"a": s1, "b": s2})
        assert info.columns.shape == (500, 8 * 8)
        assert info.n_cols == 64

    def test_penalty_leaves_only_bilinear_null_space(self):
        s1 = Spline(n_knots=5)
        s2 = Spline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b", n_knots=(5, 5))
        info = ti.build(x1, x2, {"a": s1, "b": s2})
        eigvals = np.linalg.eigvalsh(info.penalty_matrix)
        assert np.sum(eigvals < 1e-10) == 1

    def test_kronecker_values(self):
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

        from scipy.interpolate import BSpline as BSpl

        m1, m2 = ti._marginal1, ti._marginal2
        x1_clip = np.clip(x1, m1.lo, m1.hi)
        x2_clip = np.clip(x2, m2.lo, m2.hi)
        B1 = BSpl.design_matrix(x1_clip, m1.knots, m1.degree).toarray() @ m1.projection
        B2 = BSpl.design_matrix(x2_clip, m2.knots, m2.degree).toarray() @ m2.projection

        for i in [0, 50, 150]:
            expected = np.kron(B1[i], B2[i])
            np.testing.assert_allclose(T[i], expected, atol=1e-12)

    def test_sparse_output(self):
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
        s1 = Spline(n_knots=5)
        s2 = Spline(n_knots=5)
        x1 = np.linspace(0, 100, 200)
        x2 = np.linspace(0, 50, 200)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b", n_knots=(3, 4))
        info = ti.build(x1, x2, {"a": s1, "b": s2})
        assert info.n_cols == 6 * 7

    def test_decompose_returns_bilinear_and_wiggly_groups(self):
        s1 = Spline(n_knots=5)
        s2 = Spline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b", n_knots=(5, 5), decompose=True)
        infos = ti.build(x1, x2, {"a": s1, "b": s2})

        assert isinstance(infos, list)
        assert [info.subgroup_name for info in infos] == ["bilinear", "wiggly"]
        assert infos[0].n_cols == 1
        assert infos[1].n_cols == (8 * 8 - 1)
        assert infos[0].projection is not None
        assert infos[1].projection is not None

    def test_reconstruct_2d(self):
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

        assert "x1" in raw and "x2" in raw
        assert raw["log_relativity"].shape == (20, 20)
        assert raw["relativity"].shape == (20, 20)
        assert raw["interaction"] is True


# ── TensorInteraction model-level specifics ───────────────────


class TestTensorInteractionModelSpecifics:
    def test_reconstruct_2d_surface(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "bm": Spline(n_knots=5)},
            interactions=[("age", "bm")],
            selection_penalty=0.1,
        )
        model.fit(X, y, sample_weight=sample_weight)
        raw = model.reconstruct_feature("age:bm")
        assert "x1" in raw and "x2" in raw
        assert raw["log_relativity"].ndim == 2

    def test_auto_dispatch_with_kwargs(self):
        model = SuperGLM(features={"a": Spline(n_knots=5), "b": Spline(n_knots=5)})
        model._add_interaction("a", "b", n_knots=(3, 4), decompose=True)
        ispec = model._interaction_specs["a:b"]
        assert isinstance(ispec, TensorInteraction)
        assert ispec._n_knots == (3, 4)
        assert ispec._decompose is True

    def test_natural_spline_parents(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"age": NaturalSpline(n_knots=10), "bm": NaturalSpline(n_knots=5)},
            interactions=[("age", "bm")],
            selection_penalty=0.1,
        )
        model.fit(X, y, sample_weight=sample_weight)
        pred = model.predict(X)
        assert pred.shape == (len(y),) and np.all(pred > 0)

    def test_decomposed_fit_predict_roundtrip(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "bm": Spline(n_knots=5)},
            selection_penalty=0.1,
        )
        model._add_interaction("age", "bm", decompose=True)
        model.fit(X, y, sample_weight=sample_weight)
        pred = model.predict(X)
        assert pred.shape == (len(y),) and np.all(pred > 0)
        names = [g.name for g in model._groups if g.feature_name == "age:bm"]
        assert names == ["age:bm:bilinear", "age:bm:wiggly"]
        raw = model.reconstruct_feature("age:bm")
        assert raw["log_relativity"].ndim == 2

    def test_decomposed_fit_reml_updates_both_subgroups(self, interaction_data):
        X, y, sample_weight = interaction_data
        model = SuperGLM(
            features={"age": Spline(n_knots=10), "bm": Spline(n_knots=5)},
            selection_penalty=0.0,
        )
        model._add_interaction("age", "bm", decompose=True)
        model.fit_reml(X, y, sample_weight=sample_weight, max_reml_iter=3)
        names = [g.name for g in model._groups if g.feature_name == "age:bm"]
        assert names == ["age:bm:bilinear", "age:bm:wiggly"]
        assert "age:bm:bilinear" in model._reml_lambdas
        assert "age:bm:wiggly" in model._reml_lambdas


# ── Tensor marginal parent geometry tests ──────────────────────


class TestTensorMarginalParentGeometry:
    """Verify that TensorInteraction consumes parent spline geometry."""

    def test_bs_edge_padding_flows_through(self):
        """BasisSpline parent's open knot vector is used in tensor marginals."""
        s1 = Spline(n_knots=5)
        s2 = Spline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b", n_knots=(5, 5))
        ti.build(x1, x2, {"a": s1, "b": s2})

        # BasisSpline uses open knot vector with 0.001*range padding
        # The tensor marginal should inherit the same knot vector
        np.testing.assert_array_equal(ti._marginal1.knots, s1._knots)
        np.testing.assert_array_equal(ti._marginal2.knots, s2._knots)

        # Open knots extend beyond the data range
        assert ti._marginal1.knots[0] < 0.0
        assert ti._marginal1.knots[-1] > 100.0

    def test_natural_spline_marginals_build(self):
        """ns+ns tensor: column count reflects natural constraint reduction."""
        s1 = NaturalSpline(n_knots=5)
        s2 = NaturalSpline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b")
        info = ti.build(x1, x2, {"a": s1, "b": s2})

        # NaturalSpline: K_raw = n_knots + degree + 1 = 9
        # Natural constraints remove 2: K_constrained = 7
        # Centering removes 1: K_eff = 6
        assert ti._marginal1.K_eff == 6
        assert ti._marginal2.K_eff == 6
        assert info.n_cols == 6 * 6

    def test_crs_marginals_build(self):
        """cr+cr tensor: uses integrated-f'' penalty, correct column count."""
        s1 = CubicRegressionSpline(n_knots=5)
        s2 = CubicRegressionSpline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b")
        info = ti.build(x1, x2, {"a": s1, "b": s2})

        # CRS: K_raw = 9, natural constraints remove 2, centering removes 1 → K_eff = 6
        assert ti._marginal1.K_eff == 6
        assert info.n_cols == 6 * 6

        # Penalty should differ from D2.T@D2 (it's integrated f'')
        # Verify it's not all zeros
        assert np.linalg.norm(ti._marginal1.penalty) > 0

    def test_mixed_bs_ns_build(self):
        """bs+ns tensor builds correctly with different marginal types."""
        s1 = Spline(n_knots=5)
        s2 = NaturalSpline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b")
        info = ti.build(x1, x2, {"a": s1, "b": s2})

        # BasisSpline: K_raw=9, centering removes 1 → K_eff=8
        # NaturalSpline: K_raw=9, natural removes 2, centering removes 1 → K_eff=6
        assert ti._marginal1.K_eff == 8
        assert ti._marginal2.K_eff == 6
        assert info.n_cols == 8 * 6

    def test_mixed_bs_cr_build(self):
        """bs+cr tensor builds correctly with different marginal types."""
        s1 = Spline(n_knots=5)
        s2 = CubicRegressionSpline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b")
        info = ti.build(x1, x2, {"a": s1, "b": s2})

        assert ti._marginal1.K_eff == 8
        assert ti._marginal2.K_eff == 6
        assert info.n_cols == 8 * 6

    def test_null_eigenvalue_invariant_bs(self):
        """bs tensor penalty has exactly 1 null eigenvalue (bilinear)."""
        s1 = Spline(n_knots=5)
        s2 = Spline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b")
        info = ti.build(x1, x2, {"a": s1, "b": s2})
        eigvals = np.linalg.eigvalsh(info.penalty_matrix)
        assert np.sum(eigvals < 1e-10) == 1

    def test_ns_penalty_is_positive_definite(self):
        """ns tensor penalty has no null eigenvalue.

        The D2 penalty's 'linear in index' null direction is NOT in the
        natural constraint subspace (Z), so after Z-projection + centering
        the penalty is full rank.
        """
        s1 = NaturalSpline(n_knots=5)
        s2 = NaturalSpline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b")
        info = ti.build(x1, x2, {"a": s1, "b": s2})
        eigvals = np.linalg.eigvalsh(info.penalty_matrix)
        assert np.all(eigvals > 1e-12), "NS tensor penalty should be positive definite"

    def test_null_eigenvalue_invariant_cr(self):
        """cr tensor penalty has exactly 1 null eigenvalue (bilinear).

        The integrated f'' penalty has null space {constant, linear in x},
        both of which satisfy f''=0 at boundaries. After Z-projection (2
        null eigvals) and centering (removes constant) → 1 null eigenvalue.
        """
        s1 = CubicRegressionSpline(n_knots=5)
        s2 = CubicRegressionSpline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b")
        info = ti.build(x1, x2, {"a": s1, "b": s2})
        eigvals = np.linalg.eigvalsh(info.penalty_matrix)
        assert np.sum(eigvals < 1e-10) == 1

    def test_n_knots_none_uses_parent_geometry(self):
        """Default n_knots=None matches parent knot count."""
        s1 = Spline(n_knots=7)
        s2 = Spline(n_knots=4)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b")  # n_knots=None
        info = ti.build(x1, x2, {"a": s1, "b": s2})

        # BasisSpline(n_knots=7): K=11, centered=10
        # BasisSpline(n_knots=4): K=8, centered=7
        assert ti._marginal1.K_eff == 10
        assert ti._marginal2.K_eff == 7
        assert info.n_cols == 10 * 7

    def test_cardinal_cr_raises(self):
        """CardinalCRSpline rejected via tensor_marginal_ingredients."""
        from superglm.features.spline import CardinalCRSpline

        s1 = CardinalCRSpline(n_knots=5)
        s2 = Spline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b")
        with pytest.raises(TypeError, match="CardinalCRSpline"):
            ti.build(x1, x2, {"a": s1, "b": s2})

    def test_cr_n_knots_override(self):
        """CRS parents with n_knots override should not raise TypeError."""
        s1 = CubicRegressionSpline(n_knots=5)
        s2 = CubicRegressionSpline(n_knots=5)
        x1 = np.linspace(0, 100, 300)
        x2 = np.linspace(0, 50, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b", n_knots=(7, 7))
        info = ti.build(x1, x2, {"a": s1, "b": s2})
        # CRS(n_knots=7): K=11, natural removes 2, centering removes 1 → K_eff=8
        assert ti._marginal1.K_eff == 8
        assert info.n_cols == 8 * 8

    def test_n_knots_override_preserves_knot_alpha(self):
        """Overridden marginals should use the parent's knot_alpha."""
        s1 = Spline(n_knots=10, knot_strategy="quantile_tempered", knot_alpha=0.14)
        s2 = Spline(n_knots=10, knot_strategy="quantile_tempered", knot_alpha=0.14)
        rng = np.random.default_rng(42)
        x1 = rng.exponential(50, 300)
        x2 = rng.exponential(25, 300)
        s1.build(x1)
        s2.build(x2)

        ti = TensorInteraction("a", "b", n_knots=(5, 5))
        ti.build(x1, x2, {"a": s1, "b": s2})

        # Build reference with the correct knot_alpha
        from superglm.features.spline import BasisSpline

        ref = BasisSpline(
            n_knots=5,
            knot_strategy="quantile_tempered",
            knot_alpha=0.14,
            boundary=(s1._lo, s1._hi),
        )
        ref._place_knots(x1)
        ref_info = ref.tensor_marginal_ingredients(x1)
        np.testing.assert_array_equal(ti._marginal1.knots, ref_info.knots)
