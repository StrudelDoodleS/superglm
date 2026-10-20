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
        "same_text",
        "duplicate_interaction",
    ],
)
def test_invalid_declarations_are_refused(case):
    from superglm.features import Numeric
    from superglm.terms import cat, normalize_terms, s, term, ti

    cases = {
        "duplicate": ("x", cat("x")),
        "collision": ("x:z", s("x"), s("z"), ti("x", "z")),
        "missing": (s("x"), ti("x", "z")),
        "numeric_parent": ("x", s("z"), ti("x", "z")),
        "invalid": (42,),
        "same_text": (term(0, Numeric()), "0"),
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


def test_a_factor_smooth_needs_no_parent_terms():
    """It reads its own columns, as under interactions=; master required both as terms."""
    from superglm.features import FactorSmooth
    from superglm.terms import interaction, normalize_terms

    normalized = normalize_terms([interaction(FactorSmooth("x", group="group"))])
    assert normalized.features == {}
    assert normalized.interaction_order == ("x:group:fs",)


def test_superlss_fits_a_factor_smooth_without_a_term_on_its_group():
    """A SuperLSS factor smooth no longer needs a main effect on its group.

    The factor smooth already spans every level's intercept, so the main
    effect master required duplicated it (a random effect there pushes its
    smoothing parameter to the cap). Master refused this declaration.
    """
    from superglm import FactorSmooth, GaussianLS, SuperLSS
    from superglm.terms import interaction, s

    rng = np.random.default_rng(5)
    n = 600
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, n), "group": rng.choice(list("abcd"), n)})
    shift = frame["group"].map({"a": 0.4, "b": -0.3, "c": 0.1, "d": 0.0}).to_numpy()
    y = np.sin(2.0 * frame["x"].to_numpy()) + shift + 0.3 * rng.standard_normal(n)
    family = GaussianLS()
    smooth = interaction(FactorSmooth("x", group="group", k=6))
    model = SuperLSS(family, family.location(s("x", kind="ps", k=7), smooth), family.scale())
    model.fit_reml(frame, y)
    assert "location:x:group:fs#wiggle" in model.smoothing_parameters_
    predicted = model.predict(frame.head(5))
    assert np.all(np.isfinite(predicted))
    restored = SuperLSS.from_bytes(model.to_bytes())
    assert np.array_equal(restored.predict(frame.head(5)), predicted)


def test_superlss_sz_smooth_still_needs_a_term_on_its_variable():
    """As on master, now with the term to add; the group needs none."""
    from superglm import FactorSmooth, GaussianLS
    from superglm.terms import interaction

    family = GaussianLS()
    smooth = interaction(FactorSmooth("x", group="group", basis="sz"))
    with pytest.raises(ValueError, match=r"needs a term on 'x'.*add s\('x'\)"):
        family.location(smooth)
    assert family.location("x", smooth).template.interaction_order == ("x:group:sz",)


@pytest.mark.parametrize(
    "second", ["transposed", "renamed"], ids=["ti(z, x)", "ti(x, z, name=...)"]
)
def test_one_interaction_of_a_class_per_column_pair(second):
    from superglm.features.interaction import TensorInteraction
    from superglm.terms import interaction, normalize_terms, s, ti

    other = (
        ti("z", "x")
        if second == "transposed"
        else interaction(TensorInteraction("x", "z"), name="x_by_z")
    )
    with pytest.raises(ValueError, match="same two columns"):
        normalize_terms([s("x"), s("z"), ti("x", "z"), other])


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
        with pytest.raises(TypeError, match="hashable"):
            helper(["x"])
    with pytest.raises(ValueError, match="non-empty strings"):
        interaction(ti("x", "z").spec, name="")
    with pytest.raises(TypeError, match="hashable"):
        ti("x", ["z"])
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
    from superglm import Polynomial
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
    with pytest.raises(TypeError, match=r"interaction\(TensorInteraction\('band', 'x'\)\)"):
        normalize_terms((*parents, ti("band", "x")))
    stepped = term("band", OrderedCategorical(order=["a", "b", "c"], basis=Polynomial(degree=1)))
    with pytest.raises(TypeError, match="without a spline basis"):
        normalize_terms((stepped, s("x", k=5), ti("band", "x")))
    with pytest.raises(TypeError, match=r"use s\('x'\)"):
        normalize_terms(("x", s("z", k=5), ti("x", "z")))


def test_a_feature_mapping_expands_to_terms_in_its_order():
    from superglm.features import Categorical, Numeric, Spline
    from superglm.terms import normalize_terms, s, ti

    shared = Spline(kind="cr", k=6)
    mapping = {"x": shared, "z": shared, "area": Categorical()}
    result = normalize_terms(("w", mapping, ti("x", "z"), s("v", k=5)))
    assert tuple(result.features) == ("w", "x", "z", "area", "v")
    assert result.interaction_order == ("x:z",)
    assert isinstance(result.features["w"], Numeric)
    # One spec object under two columns gives two independent terms.
    assert result.features["x"] is not result.features["z"]
    assert result.features["x"] is not shared
    declared = shared.n_knots
    shared.n_knots = declared + 5
    assert result.features["x"].n_knots == result.features["z"].n_knots == declared


@pytest.mark.parametrize("case", ["duplicate", "repeated", "interaction", "column", "bare"])
def test_invalid_feature_mappings_are_refused(case):
    from superglm.features import Numeric, Spline
    from superglm.features.interaction import TensorInteraction
    from superglm.terms import normalize_terms, s

    cases = {
        "duplicate": (({"x": Spline()}, s("x")), ValueError, "Duplicate main term"),
        "repeated": (({"x": Numeric()}, {"x": Numeric()}), ValueError, "Duplicate main term"),
        "interaction": (
            ({"x": Spline(), "z": Spline(), "x:z": TensorInteraction("x", "z")},),
            TypeError,
            "interaction",
        ),
        "column": (({0: Numeric(), "0": Numeric()},), ValueError, "both read as '0'"),
        "bare": ({"x": Numeric()}, TypeError, "inside"),
    }
    terms, error, message = cases[case]
    with pytest.raises(error, match=message):
        normalize_terms(terms)


def test_superlss_takes_integer_column_labels_end_to_end():
    """A SuperLSS predictor takes the labels SuperGLM's features= takes; master refused them.

    Results name the term by its text, as in ``"location:0#wiggle"``, and a
    term is found by its label or that text.
    """
    from superglm import Categorical, GaussianLS, Numeric, Spline, SuperLSS

    rng = np.random.default_rng(11)
    n = 500
    frame = pd.DataFrame(
        {0: rng.uniform(size=n), 1: rng.choice(list("abc"), n), 2: rng.uniform(size=n)}
    )
    y = (
        np.sin(3.0 * frame[0].to_numpy())
        + 0.3 * (frame[1].to_numpy() == "a")
        + (0.3 + 0.2 * frame[2].to_numpy()) * rng.standard_normal(n)
    )
    family = GaussianLS()
    model = SuperLSS(
        family,
        family.location({0: Spline(k=6), 1: Categorical()}),
        family.scale({2: Numeric()}),
        discrete=True,
        n_bins={0: 32},
    )
    model.fit_reml(frame, y)
    assert list(model.smoothing_parameters_) == ["location:0#wiggle"]
    assert any(name.startswith("location:1[") for name in model.coef_)
    predicted = model.predict(frame.head(5))
    assert np.all(np.isfinite(predicted))
    assert np.array_equal(SuperLSS.from_bytes(model.to_bytes()).predict(frame.head(5)), predicted)
    summary = model.summary()
    assert {"0", "1"} <= {str(term) for term in summary["term"]}
    by_label = model.term_inference("location", 0)
    by_text = model.term_inference("location", "0")
    assert np.array_equal(by_label.effect, by_text.effect)
    assert model.term_test("location", 0).p_value == model.term_test("location", "0").p_value
    assert model.term_test("location", 1).rank > 0
    # Diagnostics name a covariate by its column label.
    for check in (model.actual_expected, model.check):
        assert check(frame, y, 1) is not None
    # The label's text names the column too, as it names the term.
    by_text, by_label = (model.actual_expected(frame, y, key) for key in ("1", 1))
    assert by_text.levels == by_label.levels
    assert np.array_equal(by_text.ratio, by_label.ratio)
    assert model.plot_data("term", parameter="location", term=0) is not None


def test_superlss_refuses_two_labels_with_one_text():
    from superglm import GaussianLS, Numeric

    family = GaussianLS()
    with pytest.raises(ValueError, match=r"Columns 0 and '0' both read as '0'"):
        family.location({0: Numeric(), "0": Numeric()})


@pytest.mark.parametrize("basis", ["fs", "sz"])
@pytest.mark.parametrize("helper", ["cat", "re"])
def test_superlss_warns_about_a_term_on_a_factor_smooth_group(basis, helper):
    """0.40 required the term, so it keeps fitting; the warning names it as redundant.

    Master gave no warning. A random effect there also gets its measured
    convergence note.
    """
    from superglm import FactorSmooth, GaussianLS
    from superglm import terms as term_helpers
    from superglm.features.factor_smooth import FactorSmoothGroupTermWarning

    family = GaussianLS()
    declared = getattr(term_helpers, helper)("group")
    smooth = term_helpers.interaction(FactorSmooth("x", group="group", basis=basis))
    with pytest.warns(
        FactorSmoothGroupTermWarning,
        match=rf"already carries each level's intercept.*Drop {helper}\('group'\) "
        r"from the 'location' predictor",
    ) as caught:
        family.location(term_helpers.s("x"), declared, smooth)
    converging = "smoothing selection can fail to converge"
    assert (converging in str(caught[0].message)) is (helper == "re")


def test_superlss_pickles_deepcopies_and_dumps_with_joblib():
    """Plain pickle refused SuperLSS's read-only mappings on master, fitted or not."""
    import copy
    import io
    import pickle

    import joblib

    from superglm import GaussianLS, SuperLSS
    from superglm.terms import cat, s

    rng = np.random.default_rng(13)
    n = 400
    frame = pd.DataFrame({"x": rng.uniform(size=n), "g": rng.choice(list("abc"), n)})
    y = np.sin(3.0 * frame["x"].to_numpy()) + 0.3 * rng.standard_normal(n)
    family = GaussianLS()
    model = SuperLSS(family, family.location(s("x", k=6), cat("g")), family.scale())
    unfitted = pickle.loads(pickle.dumps(model))
    model.fit_reml(frame, y)
    expected = model.predict(frame)
    stream = io.BytesIO()
    joblib.dump(model, stream)
    stream.seek(0)
    for restored in (
        pickle.loads(pickle.dumps(model)),
        copy.deepcopy(model),
        joblib.load(stream),
        unfitted.fit_reml(frame, y),
    ):
        assert np.array_equal(restored.predict(frame), expected)
    assert np.array_equal(SuperLSS.from_bytes(model.to_bytes()).predict(frame), expected)

    # As to_bytes, a pickle leaves the training frame out (its parent pickled
    # every row and column of it): a restored model takes X_train=. A deep
    # copy keeps it, by reference; a shallow copy shares every member.
    restored = pickle.loads(pickle.dumps(model))
    assert restored._training_frame is None
    with pytest.raises(RuntimeError, match="X_train"):
        restored.summary()
    assert restored.summary(X_train=frame).equals(model.summary())
    assert copy.deepcopy(model)._training_frame is model._training_frame
    assert copy.deepcopy(model).summary().equals(model.summary())
    shallow = copy.copy(model)
    assert shallow._model is model._model and shallow._training_frame is model._training_frame


def test_superlss_interactions_take_integer_column_labels():
    """Master refused a non-string interaction parent in a SuperLSS predictor."""
    import pickle

    from superglm import GaussianLS, SuperLSS
    from superglm.features.interaction import SplineCategorical
    from superglm.terms import cat, interaction, s, ti

    rng = np.random.default_rng(17)
    n = 600
    frame = pd.DataFrame(
        {0: rng.uniform(size=n), 1: rng.choice(list("abc"), n), 2: rng.uniform(size=n)}
    )
    y = np.sin(3.0 * frame[0].to_numpy()) * frame[2].to_numpy() + 0.3 * rng.standard_normal(n)
    family = GaussianLS()
    location = family.location(
        s(0, k=6), cat(1), s(2, k=6), ti(0, 2), interaction(SplineCategorical(0, 1))
    )
    model = SuperLSS(family, location, family.scale()).fit_reml(frame, y)
    assert {"0:2", "0:1[a]"} <= set(model.summary()["term"])
    assert model.term_test("location", "0:2").rank > 0
    expected = model.predict(frame.head(5))
    assert np.array_equal(SuperLSS.from_bytes(model.to_bytes()).predict(frame.head(5)), expected)
    assert np.array_equal(pickle.loads(pickle.dumps(model)).predict(frame.head(5)), expected)


def test_a_saved_superlss_model_with_a_term_on_its_factor_smooth_group_still_loads():
    """The refusal is the declaration's; a template restored from an artifact is not re-refused."""
    from superglm import Categorical, FactorSmooth, GaussianLS, SuperLSS
    from superglm.distributional import Predictor
    from superglm.features import Spline
    from tests.bound_predictor_fixtures import model_from_templates

    rng = np.random.default_rng(23)
    n = 400
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, n), "g": rng.choice(list("abc"), n)})
    y = np.sin(2.0 * frame["x"].to_numpy()) + 0.3 * rng.standard_normal(n)
    legacy = model_from_templates(
        family=GaussianLS(),
        predictors=[
            Predictor(
                "location",
                {"x": Spline("cr", k=6), "g": Categorical()},
                interaction_specs={"x:g:fs": FactorSmooth("x", group="g", k=6)},
            ),
            Predictor("scale", {}),
        ],
    )
    lambdas = {name: 1.0 for name in ("location:x#wiggle", "location:x:g:fs#wiggle")}
    lambdas |= {"location:x:g:fs#null_0": 1.0, "location:x:g:fs#null_1": 1.0}
    fitted = legacy.fit(frame, y, lambdas=lambdas)
    restored = SuperLSS.from_bytes(fitted.to_bytes())
    assert np.array_equal(restored.predict(frame), fitted.predict(frame))


@pytest.mark.parametrize(("basis", "intercept"), [("sz", False), ("sz", True), ("fs", False)])
def test_the_group_warning_names_intercept_true_for_sz_without_an_intercept(basis, intercept):
    """Only sz without an intercept leaves the overall level to the group term."""
    from superglm import FactorSmooth, GaussianLS
    from superglm.features.factor_smooth import FactorSmoothGroupTermWarning
    from superglm.terms import cat, interaction, s

    family = GaussianLS()
    smooth = interaction(FactorSmooth("x", group="g", basis=basis))
    with pytest.warns(FactorSmoothGroupTermWarning, match=r"Drop cat\('g'\)") as caught:
        family.location(s("x"), cat("g"), smooth, intercept=intercept)
    sentence = "With intercept=False, the same model is intercept=True without the term on 'g'."
    assert (sentence in str(caught[0].message)) is (basis == "sz" and not intercept)


@pytest.mark.parametrize(
    "second", ["transposed", "renamed"], ids=["ti(z, x)", "ti(x, z, name=...)"]
)
def test_superlss_keeps_a_repeated_interaction_pair_with_a_warning(second):
    """0.40 fitted both; #503 refused them. SuperLSS keeps both, and warns (bit for bit with 0.40)."""
    from superglm import GaussianLS
    from superglm.features.interaction import TensorInteraction
    from superglm.terms import RedundantInteractionWarning, interaction, s, ti

    other = (
        ti("z", "x")
        if second == "transposed"
        else interaction(TensorInteraction("x", "z"), name="x_by_z")
    )
    family = GaussianLS()
    with pytest.warns(RedundantInteractionWarning, match="the second is redundant"):
        location = family.location(s("x"), s("z"), ti("x", "z"), other)
    assert len(location.template.interaction_order) == 2


def test_superlss_saves_a_model_whose_labels_are_not_json_values():
    """A date or NumPy integer label fits and survives to_bytes; the parent refused saving it."""
    import datetime

    from superglm import Categorical, GaussianLS, Numeric, Spline, SuperLSS
    from superglm.terms import ti

    rng = np.random.default_rng(31)
    n = 400
    day, code = datetime.date(2026, 1, 1), np.int64(7)
    frame = pd.DataFrame(
        {day: rng.uniform(size=n), code: rng.uniform(size=n), "g": rng.choice(list("abc"), n)}
    )
    y = np.sin(3.0 * frame[day].to_numpy()) + 0.3 * rng.standard_normal(n)
    family = GaussianLS()
    model = SuperLSS(
        family,
        family.location({day: Spline(k=6), code: Spline(k=6), "g": Categorical()}, ti(day, code)),
        family.scale({code: Numeric()}),
    ).fit_reml(frame, y)
    restored = SuperLSS.from_bytes(model.to_bytes())
    assert np.array_equal(restored.predict(frame), model.predict(frame))
