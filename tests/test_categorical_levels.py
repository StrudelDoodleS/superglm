"""Bound level universes on Categorical (spec 2026-08-11)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm.features.categorical import Categorical
from superglm.features.grouping import collapse_levels


def _build(spec, x, w=None):
    return spec.build(np.asarray(x, dtype=object), sample_weight=w)


class TestDeclaredUniverse:
    def test_declared_unobserved_level_is_known_and_pinned(self):
        spec = Categorical(base="first", levels=["a", "b", "c"])
        with pytest.warns(UserWarning, match=r"pinned to base.*'c'"):
            info = _build(spec, ["a", "b", "a"])
        assert spec._levels == ["a", "b", "c"]
        assert spec._pinned_levels == ["c"]
        assert spec._non_base == ["b"]  # no column for 'c'
        assert info.n_cols == 1

    def test_pinned_level_scores_as_base(self):
        spec = Categorical(base="first", levels=["a", "b", "c"])
        with pytest.warns(UserWarning):
            _build(spec, ["a", "b", "a"])
        eta = spec.score(np.asarray(["a", "c", "b"], dtype=object), np.array([0.7]))
        assert eta == pytest.approx([0.0, 0.0, 0.7])

    def test_fit_data_outside_universe_errors(self):
        spec = Categorical(base="first", levels=["a", "b"])
        with pytest.raises(ValueError, match="outside the declared level universe"):
            _build(spec, ["a", "b", "ROGUE"])

    def test_missing_values_still_error_before_universe_check(self):
        spec = Categorical(base="first", levels=["a", "b"])
        with pytest.raises(ValueError, match="missing values"):
            _build(spec, ["a", None, "b"])

    def test_declared_order_defines_base_first(self):
        spec = Categorical(base="first", levels=["z", "a"])
        _build(spec, ["a", "z"])
        assert spec._base_level == "z"

    def test_no_universe_is_status_quo(self):
        spec = Categorical(base="first")
        info = _build(spec, ["b", "a", "b"])
        assert spec._levels == ["a", "b"]
        assert spec._level_source == "inferred"
        assert info.n_cols == 1

    def test_levels_source_series(self):
        spec = Categorical(base="first", levels=pd.Series(["b", "a", "b"]))
        assert spec._declared_levels == ["a", "b"]
        assert spec._level_source == "declared"


class TestZeroWeightAndBaseFallback:
    def test_zero_weight_level_is_pinned(self):
        spec = Categorical(base="first", levels=["a", "b", "c"])
        with pytest.warns(UserWarning, match=r"pinned to base.*'c'"):
            _build(spec, ["a", "b", "c"], w=np.array([1.0, 1.0, 0.0]))
        assert spec._pinned_levels == ["c"]

    def test_empty_declared_base_falls_back_deterministically(self):
        spec = Categorical(base="c", levels=["a", "b", "c"])
        with pytest.warns(UserWarning, match="fall"):
            _build(spec, ["a", "b", "b"], w=np.array([1.0, 2.0, 2.0]))
        assert spec._base_level == "b"  # most exposed observed
        assert spec._base_fallback == ("c", "b")

    def test_empty_base_fallback_unweighted_first_observed(self):
        spec = Categorical(base="c", levels=["a", "b", "c"])
        with pytest.warns(UserWarning, match="fall"):
            _build(spec, ["b", "a", "b"])
        assert spec._base_level == "a"  # first observed in universe order

    def test_most_exposed_ignores_pinned_levels(self):
        spec = Categorical(levels=["a", "b", "c"])
        with pytest.warns(UserWarning):
            _build(spec, ["a", "b"], w=np.array([1.0, 5.0]))
        assert spec._base_level == "b"


class TestUnseenPolicy:
    def _fitted(self):
        spec = Categorical(base="first", unseen="base")
        _build(spec, ["a", "b", "a"])
        return spec

    def test_unseen_base_routes_to_zero_with_warning(self):
        spec = self._fitted()
        with pytest.warns(UserWarning, match=r"NOVEL.*2 row"):
            eta = spec.score(np.asarray(["a", "NOVEL", "NOVEL"], dtype=object), np.array([0.5]))
        assert eta == pytest.approx([0.0, 0.0, 0.0])

    def test_unseen_base_transform_zero_rows(self):
        spec = self._fitted()
        with pytest.warns(UserWarning):
            T = spec.transform(np.asarray(["b", "NOVEL"], dtype=object))
        assert T.tolist() == [[1.0], [0.0]]

    def test_unseen_error_is_default_and_unchanged(self):
        spec = Categorical(base="first")
        _build(spec, ["a", "b"])
        with pytest.raises(ValueError, match="unseen categorical levels"):
            spec.score(np.asarray(["NOVEL"], dtype=object), np.array([0.5]))

    def test_unseen_base_missing_values_still_error(self):
        spec = self._fitted()
        with pytest.raises(ValueError, match="missing values"):
            spec.score(np.asarray(["a", None], dtype=object), np.array([0.5]))

    def test_invalid_unseen_rejected(self):
        with pytest.raises(ValueError, match="unseen"):
            Categorical(unseen="ignore")

    def test_unseen_routing_uses_a_supported_pandas_lookup(self):
        # pandas deprecates -- and will eventually reject -- constructing a
        # Categorical with values outside the declared categories, which is the
        # one call the unseen route cannot do without. Pin the lookup, or the
        # feature stops working on a pandas upgrade rather than in a test.
        spec = self._fitted()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spec.score(np.asarray(["a", "NOVEL"], dtype=object), np.array([0.5]))
        assert [
            w for w in caught if issubclass(w.category, DeprecationWarning | FutureWarning)
        ] == []

    def test_wraparound_guard_no_negative_indexing(self):
        # base='b' is the LAST level: a wrapped -1 would grab beta of the
        # last non-base level instead of 0. Assert exact zeros.
        spec = Categorical(base="b", unseen="base")
        _build(spec, ["a", "b", "a"])
        with pytest.warns(UserWarning):
            eta = spec.score(np.asarray(["NOVEL"], dtype=object), np.array([3.14]))
        assert eta == pytest.approx([0.0])


class TestGroupedDeclared:
    def test_grouping_must_cover_declared_universe(self):
        grouping = collapse_levels(["a", "b"], groups={"grp": ["a", "b"]})
        with pytest.raises(ValueError, match="not covered by the grouping"):
            Categorical(levels=["a", "b", "c"], grouping=grouping)

    def test_declared_raws_become_the_grouping_image(self):
        grouping = collapse_levels(["a", "b", "c"], groups={"grp": ["a", "b"]})
        spec = Categorical(base="first", levels=["a", "b", "c"], grouping=grouping)
        with pytest.warns(UserWarning, match=r"pinned to base.*'c'"):
            _build(spec, ["a", "b"])
        assert spec._levels == ["grp", "c"]  # first-occurrence order of the image
        assert spec._pinned_levels == ["c"]

    def test_novel_raw_label_still_errors_under_unseen_base(self):
        # Documented v1 limitation: unseen='base' routes novel WORKING levels,
        # but a grouped spec validates raw labels against the grouping domain
        # first, so a novel RAW label still errors.
        grouping = collapse_levels(["a", "b", "c"], groups={"grp": ["a", "b"]})
        spec = Categorical(base="first", unseen="base", grouping=grouping)
        _build(spec, ["a", "b", "c"])
        with pytest.raises(ValueError, match="unseen categorical levels"):
            spec.score(np.asarray(["ROGUE"], dtype=object), np.array([0.5]))


class TestAdoptionHooks:
    def test_adopt_dtype_categories_when_unset(self):
        spec = Categorical(base="first")
        spec.adopt_dtype_categories(["a", "b", "c"])
        assert spec._declared_levels == ["a", "b", "c"]
        assert spec._level_source == "dtype"

    def test_adopt_does_not_override_declared(self):
        spec = Categorical(base="first", levels=["x", "y"])
        spec.adopt_dtype_categories(["a", "b"])
        assert spec._declared_levels == ["x", "y"]
        assert spec._level_source == "declared"

    def test_apply_level_binding_levels_and_base(self):
        from superglm.types import LevelBinding

        spec = Categorical()  # most_exposed, no universe
        spec.apply_level_binding(LevelBinding(levels=("a", "b"), base="b"))
        assert spec._declared_levels == ["a", "b"]
        assert spec._level_source == "full-frame"
        _build(spec, ["a", "b"], w=np.array([9.0, 1.0]))
        assert spec._base_level == "b"  # pinned wins over fold exposure

    def test_binding_base_ignored_for_explicit_base(self):
        from superglm.types import LevelBinding

        spec = Categorical(base="a", levels=["a", "b"])
        spec.apply_level_binding(LevelBinding(levels=("a", "b"), base="b"))
        _build(spec, ["a", "b"])
        assert spec._base_level == "a"

    def test_resolve_binding_pure(self):
        spec = Categorical()
        binding = spec.resolve_binding(
            np.asarray(["a", "b", "b"], dtype=object), np.array([1.0, 3.0, 3.0])
        )
        assert list(binding.levels) == ["a", "b"] and binding.base == "b"
        assert spec._levels == [] and spec._declared_levels is None  # untouched


class TestEndToEndDtypeUniverse:
    def test_categorical_dtype_column_declares_universe_through_fit(self):
        from superglm import SuperGLM

        rng = np.random.default_rng(0)
        g = pd.Categorical(rng.choice(["a", "b"], size=200), categories=["a", "b", "c"])
        X = pd.DataFrame({"g": g, "x": rng.normal(size=200)})
        y = rng.poisson(1.0, size=200).astype(float)
        model = SuperGLM(family="poisson", features={"g": Categorical(base="first")})
        with pytest.warns(UserWarning, match="pinned to base"):
            model.fit(X, y)
        spec = model._specs["g"]
        assert spec._levels == ["a", "b", "c"]
        assert spec._pinned_levels == ["c"]
        assert spec._level_source == "dtype"
        # predict on a frame containing the declared-but-unfitted level: no error
        Xp = pd.DataFrame(
            {
                "g": pd.Categorical(["c", "a"], categories=["a", "b", "c"]),
                "x": [0.0, 0.0],
            }
        )
        mu = model.predict(Xp)
        assert np.isfinite(mu).all()

    def test_level_bindings_flow_through_config(self):
        from superglm import SuperGLM
        from superglm.types import LevelBinding

        rng = np.random.default_rng(1)
        X = pd.DataFrame({"g": rng.choice(["a", "b"], size=100).astype(object)})
        y = rng.poisson(1.0, size=100).astype(float)
        model = SuperGLM(family="poisson", features={"g": Categorical(base="first")})
        model._config = model._config.with_value(
            level_bindings=(("g", LevelBinding(levels=("a", "b", "z"), base=None)),)
        )
        with pytest.warns(UserWarning, match="pinned to base"):
            model.fit(X, y)
        assert model._specs["g"]._levels == ["a", "b", "z"]
        assert model._specs["g"]._level_source == "full-frame"

    def test_config_pickle_roundtrip_without_bindings(self):
        import pickle

        from superglm import SuperGLM

        model = SuperGLM(family="poisson")
        state = pickle.loads(pickle.dumps(model._config))
        assert state.level_bindings is None


class TestDerivedTerms:
    """Interactions inherit the parent's universe, pins and unseen policy."""

    def _parent(self, x, unseen="error"):
        spec = Categorical(base="first", levels=["a", "b", "c"], unseen=unseen)
        with pytest.warns(UserWarning, match="pinned to base"):
            _build(spec, x)
        return spec

    def _spline_by_cat_model(self, unseen="error"):
        from superglm import SuperGLM
        from superglm.features.spline import Spline

        rng = np.random.default_rng(3)
        n = 240
        X = pd.DataFrame(
            {
                "x": rng.uniform(0.0, 10.0, n),
                "g": np.asarray(rng.choice(["a", "b"], size=n), dtype=object),
            }
        )
        y = rng.poisson(1.0, size=n).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "x": Spline(n_knots=5),
                "g": Categorical(base="first", levels=["a", "b", "c"], unseen=unseen),
            },
            interactions=[("x", "g")],
        )
        with pytest.warns(UserWarning, match="pinned to base"):
            model.fit(X, y)
        return model

    def test_spline_categorical_inherits_universe_and_pins(self):
        model = self._spline_by_cat_model()
        ispec = model._interaction_specs["x:g"]
        assert ispec._non_base == ["b"]  # no block for the pinned 'c'
        assert ispec._cat_levels == ["a", "b", "c"]
        assert ispec._cat_unseen == "error"

    def test_pinned_level_predicts_as_base_through_the_interaction(self):
        model = self._spline_by_cat_model()
        Xp = pd.DataFrame({"x": [2.5, 2.5], "g": np.asarray(["c", "a"], dtype=object)})
        mu = model.predict(Xp)
        assert np.isfinite(mu).all()
        assert mu[0] == pytest.approx(mu[1])

    def test_pinned_level_contributes_zero_to_the_interaction(self):
        ispec = self._spline_by_cat_model()._interaction_specs["x:g"]
        x_num = np.array([1.0, 5.0])
        x_cat = np.asarray(["c", "c"], dtype=object)
        blocks = ispec.transform(x_num, x_cat)
        assert np.count_nonzero(blocks) == 0
        eta = ispec.score(x_num, x_cat, np.ones(blocks.shape[1]))
        assert eta == pytest.approx([0.0, 0.0])

    def test_unseen_base_routes_novel_levels_through_the_interaction(self):
        model = self._spline_by_cat_model(unseen="base")
        Xp = pd.DataFrame({"x": [2.5, 2.5], "g": np.asarray(["NOVEL", "a"], dtype=object)})
        with pytest.warns(UserWarning, match="NOVEL"):
            mu = model.predict(Xp)
        assert mu[0] == pytest.approx(mu[1])

    def test_unseen_error_still_raises_through_the_interaction(self):
        ispec = self._spline_by_cat_model()._interaction_specs["x:g"]
        with pytest.raises(ValueError, match="unseen categorical levels"):
            ispec.score(np.array([2.5]), np.asarray(["NOVEL"], dtype=object), np.zeros(1))

    def test_numeric_categorical_accepts_pins_and_routes_novel(self):
        from superglm.features.interaction import NumericCategorical
        from superglm.features.numeric import Numeric

        x_cat = np.asarray(["a", "b"] * 6, dtype=object)
        x_num = np.arange(12, dtype=float)
        num = Numeric()
        num.build(x_num)
        nc = NumericCategorical("v", "g")
        nc.build(x_num, x_cat, {"v": num, "g": self._parent(x_cat, unseen="base")})

        assert nc.transform(np.array([1.0]), np.asarray(["c"], dtype=object)).tolist() == [[0.0]]
        with pytest.warns(UserWarning, match="NOVEL"):
            eta = nc.score(np.array([1.0]), np.asarray(["NOVEL"], dtype=object), np.array([2.0]))
        assert eta == pytest.approx([0.0])

    def test_polynomial_categorical_accepts_pinned_levels(self):
        from superglm.features.interaction import PolynomialCategorical
        from superglm.features.polynomial import Polynomial

        x_cat = np.asarray(["a", "b"] * 6, dtype=object)
        x_poly = np.linspace(0.0, 1.0, 12)
        poly = Polynomial(degree=2)
        poly.build(x_poly)
        pc = PolynomialCategorical("p", "g")
        pc.build(x_poly, x_cat, {"p": poly, "g": self._parent(x_cat)})

        blocks = pc.transform(np.array([0.4]), np.asarray(["c"], dtype=object))
        assert np.count_nonzero(blocks) == 0

    def test_categorical_interaction_accepts_pins_and_routes_novel(self):
        from superglm.features.interaction import CategoricalInteraction

        left_x = np.asarray(["a", "b"] * 6, dtype=object)
        right_x = np.asarray(["X", "Y"] * 6, dtype=object)
        right = Categorical(base="first")
        _build(right, right_x)
        ci = CategoricalInteraction("g", "h")
        ci.build(left_x, right_x, {"g": self._parent(left_x, unseen="base"), "h": right})

        eta = ci.score(
            np.asarray(["c"], dtype=object), np.asarray(["Y"], dtype=object), np.array([1.5])
        )
        assert eta == pytest.approx([0.0])
        with pytest.warns(UserWarning, match="NOVEL"):
            eta = ci.score(
                np.asarray(["NOVEL"], dtype=object),
                np.asarray(["Y"], dtype=object),
                np.array([1.5]),
            )
        assert eta == pytest.approx([0.0])

    def test_screening_codes_accept_declared_but_pinned_levels(self):
        from superglm.model.screening_ops import _categorical_codes

        spec = self._parent(["a", "b", "a"])
        codes, n_levels = _categorical_codes(spec, np.asarray(["c", "a"], dtype=object))
        assert n_levels == 3
        assert codes.tolist() == [2, 0]

    def test_screening_guard_names_the_declared_universe(self, monkeypatch):
        # The guard is defensive: label resolution rejects out-of-universe data
        # first. Disable that to reach the branch and pin what it says.
        from superglm.model import screening_ops

        monkeypatch.setattr(
            screening_ops,
            "_resolve_categorical_labels",
            lambda x, grouping, **kwargs: np.asarray(x),
        )
        spec = self._parent(["a", "b", "a"])
        with pytest.raises(ValueError, match="declared level universe"):
            screening_ops._categorical_codes(spec, np.asarray(["a", "ROGUE"], dtype=object))


class TestReconstruct:
    def test_reconstruct_reports_pins_and_source(self):
        spec = Categorical(base="first", levels=["a", "b", "c"])
        with pytest.warns(UserWarning):
            _build(spec, ["a", "b", "a"])
        rec = spec.reconstruct(np.array([0.7]))
        assert rec["pinned_levels"] == ["c"]
        assert rec["level_source"] == "declared"
        assert rec["base_fallback"] is None
        assert rec["relativities"]["c"] == pytest.approx(1.0)
        assert rec["log_relativities"]["c"] == pytest.approx(0.0)
