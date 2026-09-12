import numpy as np
import pandas as pd
import pytest


def test_bound_terms_keep_numeric_and_categorical_semantics():
    from importlib.util import find_spec

    assert find_spec("superglm.terms") is not None, "bound terms are not implemented"
    from superglm.features import Categorical, Numeric
    from superglm.terms import cat, normalize_terms, s

    result = normalize_terms(("age", s("value", kind="cr", k=6), cat("area")))
    assert tuple(result.features) == ("age", "value", "area")
    assert isinstance(result.features["age"], Numeric)
    assert isinstance(result.features["area"], Categorical)


def _compile(terms, data):
    from superglm._frame import as_eager_frame
    from superglm._predictor_compiler import compile_predictor_design
    from superglm.terms import normalize_terms

    normalized = normalize_terms(terms)
    weights = np.ones(len(data))
    return compile_predictor_design(
        as_eager_frame(data),
        weights,
        geometry_weight=weights,
        polynomial_weight=weights,
        categorical_reporting_weight=weights,
        ordered_reporting_weight=weights,
        specs=normalized.features,
        feature_order=tuple(normalized.features),
        interaction_specs=normalized.interaction_specs,
        interaction_order=normalized.interaction_order,
        pending_interactions=(),
        model_discrete=False,
        n_bins_config=32,
        lambda2=1.0,
    )


def test_interactions_resolve_later_parents_and_keep_partition_order():
    from superglm.features.interaction import NumericInteraction
    from superglm.terms import interaction, normalize_terms, s, ti

    terms = (
        ti("x", "z"),
        "a",
        s("z", k=5),
        interaction(NumericInteraction("a", "b"), name="product"),
        "b",
        s("x", k=5),
    )
    normalized = normalize_terms(terms)
    assert tuple(normalized.features) == ("a", "z", "b", "x")
    assert normalized.interaction_order == ("x:z", "product")
    rng = np.random.default_rng(42)
    built = _compile(terms, pd.DataFrame(rng.normal(size=(100, 4)), columns=["a", "b", "x", "z"]))
    assert built.feature_order == ("a", "z", "b", "x")
    assert built.interaction_order == ("x:z", "product")
    assert built.interaction_specs["x:z"]._p1 > 0
    assert built.interaction_specs["x:z"]._p2 > 0


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "collision",
        "missing",
        "numeric_parent",
        "invalid",
        "empty",
        "duplicate_interaction",
    ],
)
def test_invalid_declarations_are_refused(case):
    from superglm.terms import cat, normalize_terms, s, ti

    cases = {
        "duplicate": ("x", cat("x")),
        "collision": ("x:z", s("x"), s("z"), ti("x", "z")),
        "missing": (s("x"), ti("x", "z")),
        "numeric_parent": ("x", s("z"), ti("x", "z")),
        "invalid": (42,),
        "empty": ("",),
        "duplicate_interaction": (s("x"), s("z"), ti("x", "z"), ti("x", "z")),
    }
    with pytest.raises((ValueError, TypeError)):
        normalize_terms(cases[case])


def test_declarations_own_mutable_settings_and_accessors():
    from superglm.features import Categorical
    from superglm.terms import normalize_terms, term

    spec = Categorical(base="first", levels=["a", "b", "c"])
    declaration = term("area", spec)
    spec._declared_levels.append("d")
    declaration.spec._declared_levels.append("e")
    normalized = normalize_terms((declaration,))
    normalized.features["area"]._declared_levels.append("f")
    built = _compile((declaration,), pd.DataFrame({"area": ["a", "b", "c", "a"]}))
    assert built.specs["area"].transform(np.array(["a", "b", "c"])).shape == (3, 2)
    with pytest.raises(ValueError):
        built.specs["area"].transform(np.array(["d"]))


def test_spline_and_random_effect_options_reach_build():
    from superglm.terms import re, s

    built = _compile(
        (s("x", kind="cr", k=6, select=True), re("group", levels=["a", "b", "c"], unseen="error")),
        pd.DataFrame({"x": np.linspace(0, 1, 30), "group": ["a", "b", "c"] * 10}),
    )
    assert built.specs["x"].transform(np.linspace(0, 1, 7)).shape == (7, 5)
    info = built.specs["x"].build(np.linspace(0, 1, 30))
    assert tuple(name for name, _ in info.penalty_components) == ("null", "wiggle")
    assert built.specs["group"].transform(np.array(["a", "b", "c"])).shape == (3, 3)
    with pytest.raises(ValueError):
        built.specs["group"].transform(np.array(["unknown"]))


def test_string_data_requires_explicit_categorical_encoding():
    from superglm.terms import cat

    data = pd.DataFrame({"area": ["a", "b", "a", "b"]})
    with pytest.raises(ValueError, match="cat\\("):
        _compile(("area",), data)
    built = _compile((cat("area"),), data)
    assert built.specs["area"].transform(np.array(["a", "b"])).shape == (2, 1)


def test_explicit_factor_smooth_retains_its_configured_name():
    from superglm.features import FactorSmooth
    from superglm.terms import cat, interaction, normalize_terms

    normalized = normalize_terms(
        (
            "x",
            cat("group"),
            interaction(FactorSmooth("x", group="group", name="by_group")),
        )
    )
    assert normalized.interaction_order == ("by_group",)


def test_generic_main_term_refuses_an_interaction_spec():
    from superglm.features.interaction import NumericInteraction
    from superglm.terms import term

    with pytest.raises(TypeError, match="interaction"):
        term("x", NumericInteraction("x", "y"))


def test_custom_configuration_graph_is_copied_without_building():
    from superglm.features import Numeric
    from superglm.terms import normalize_terms, term

    class StatefulFeature(Numeric):
        def __init__(self):
            self.left = [1.0, 2.0]
            self.right = self.left

        def build(self, x, sample_weight=None):
            raise AssertionError("Declaration must not build row arrays")

    original = StatefulFeature()
    declaration = term("x", original)
    original.left.append(3.0)
    owned = normalize_terms((declaration,)).features["x"]
    assert owned.left == [1.0, 2.0]
    assert owned.left is owned.right
    owned.left.append(4.0)
    assert declaration.spec.left == [1.0, 2.0]


def test_invalid_helper_names_and_types_fail_at_declaration():
    from superglm.terms import cat, interaction, re, s, term, ti

    for helper in (cat, re, s):
        with pytest.raises(ValueError):
            helper(" ")
    with pytest.raises(ValueError):
        ti("x", "")
    with pytest.raises(TypeError):
        term("x", object())
    with pytest.raises(TypeError):
        interaction(object())


def test_numeric_conversion_preserves_convertible_strings():
    from superglm.features import Numeric

    numeric = Numeric()
    values = np.array(["1", "2.5", "-3"])
    np.testing.assert_array_equal(numeric.build(values).columns[:, 0], [1, 2.5, -3])
    np.testing.assert_array_equal(numeric.transform(values)[:, 0], [1, 2.5, -3])
    np.testing.assert_array_equal(numeric.score(values, np.array([2.0])), [2, 5, -6])
    for operation in (numeric.build, numeric.transform):
        with pytest.raises(ValueError, match="cat\\("):
            operation(np.array(["area_a"]))
    with pytest.raises(ValueError, match="cat\\("):
        numeric.score(np.array(["area_a"]), np.array([2.0]))


def test_explicit_tensor_preserves_ordered_spline_parent_support():
    from superglm.features import OrderedCategorical, Spline
    from superglm.features.interaction import TensorInteraction
    from superglm.terms import interaction, normalize_terms, s, term, ti

    parents = (
        term("band", OrderedCategorical(order=["a", "b", "c", "d", "e"], basis=Spline(k=5))),
        s("x", k=5),
    )
    data = pd.DataFrame(
        {"band": ["a", "b", "c", "d", "e"] * 20, "x": np.random.default_rng(1).uniform(size=100)}
    )
    built = _compile((*parents, interaction(TensorInteraction("band", "x"))), data)
    assert built.interaction_specs["band:x"]._p1 > 0
    with pytest.raises(TypeError, match="spline"):
        normalize_terms((*parents, ti("band", "x")))
