"""``SuperGLM(terms=[...])``: the list form of ``features=`` and ``interactions=``."""

import inspect
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

import superglm.model.base as model_base
from superglm import (
    Categorical,
    FactorSmooth,
    RandomEffect,
    Spline,
    Structure,
    SuperGLM,
    cat,
    cross_validate,
    interaction,
    re,
    s,
    ti,
)
from superglm.editor import EditorSession
from superglm.features.interaction import SplineCategorical
from superglm.terms import NormalizedTerms


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20261019)
    n = 600
    X = pd.DataFrame(
        {
            "x": rng.uniform(size=n),
            "z": rng.uniform(size=n),
            "g": rng.choice(["a", "b", "c"], n),
            "b": rng.choice([f"b{i}" for i in range(8)], n),
        }
    )
    eta = 0.3 * np.sin(3.0 * X["x"]) + 0.2 * X["z"] + 0.2 * (X["g"] == "a")
    return X, rng.poisson(np.exp(eta)).astype(float)


def _pair(method: str, discrete: bool) -> tuple[SuperGLM, SuperGLM]:
    """The same model declared as a features dict and as a terms list."""
    features = {"x": Spline(k=6), "z": Spline(k=6), "g": Categorical()}
    terms = [s("x", k=6), s("z", k=6), cat("g")]
    if method == "fit_reml":  # a random effect is fitted only by fit_reml
        features["b"] = RandomEffect()
        terms.append(re("b"))
    options = {"discrete": discrete, "selection_penalty": 0.0}
    as_dict = SuperGLM(features=features, interactions=[("x", "z")], **options)
    as_list = SuperGLM(terms=[*terms, ti("x", "z")], **options)
    return as_dict, as_list


def _assert_same_fit(left: SuperGLM, right: SuperGLM, X: pd.DataFrame, *, reml: bool) -> None:
    assert repr(left) == repr(right)
    assert [g.name for g in left._groups] == [g.name for g in right._groups]
    assert np.array_equal(left.result.beta, right.result.beta)
    assert left.result.intercept == right.result.intercept
    assert left.result.deviance == right.result.deviance
    assert np.array_equal(left.predict(X), right.predict(X))
    if reml:
        assert left.reml_diagnostics()["lambdas"] == right.reml_diagnostics()["lambdas"]


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
@pytest.mark.parametrize("method", ["fit", "fit_reml"])
def test_terms_list_fits_bit_identically_to_the_features_dict(data, method, discrete):
    X, y = data
    as_dict, as_list = _pair(method, discrete)
    assert repr(as_dict) == repr(as_list)
    getattr(as_dict, method)(X, y)
    getattr(as_list, method)(X, y)
    _assert_same_fit(as_dict, as_list, X, reml=method == "fit_reml")


@pytest.mark.parametrize("mutation", ["reorder", "drop"])
def test_bit_identity_check_catches_a_normalizer_that_reorders_or_drops(
    data, monkeypatch, mutation
):
    """Mutation check: the comparison above fails when ``normalize_terms`` is wrong."""
    original = model_base.normalize_terms

    def mutated(terms):
        normalized = original(terms)
        items = list(normalized.features.items())
        items = items[::-1] if mutation == "reorder" else [i for i in items if i[0] != "g"]
        return NormalizedTerms(
            dict(items), normalized.interaction_specs, normalized.interaction_order
        )

    monkeypatch.setattr(model_base, "normalize_terms", mutated)
    X, y = data
    as_dict, as_list = _pair("fit", discrete=False)
    as_dict.fit(X, y)
    as_list.fit(X, y)
    with pytest.raises(AssertionError):
        _assert_same_fit(as_dict, as_list, X, reml=False)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "conflict",
    [{"features": {"x": Spline()}}, {"splines": ["x"]}, {"interactions": [("x", "z")]}],
    ids=["features", "splines", "interactions"],
)
def test_terms_refuses_the_arguments_it_replaces(conflict):
    (name,) = conflict
    with pytest.raises(ValueError, match=f"'terms' and '{name}'"):
        SuperGLM(terms=[s("x"), s("z")], **conflict)


@pytest.mark.parametrize(
    "shorthand", [{"n_knots": 8}, {"degree": 2}, {"categorical_base": "first"}]
)
def test_terms_refuses_shorthand_defaults_it_would_not_read(shorthand):
    ((name, value),) = shorthand.items()
    with pytest.raises(ValueError, match=f"'terms' and {name}={value!r}"):
        SuperGLM(terms=["x"], **shorthand)
    SuperGLM(terms=["x"], n_knots=10, degree=3, categorical_base="most_exposed")


@pytest.mark.parametrize("declarations", [["x"], (s("x"), cat("g"))], ids=["list", "tuple"])
def test_features_refuses_term_declarations_and_names_terms(declarations):
    with pytest.raises(TypeError, match=r"terms=\[\.\.\.\]"):
        SuperGLM(features=declarations)


def test_terms_is_keyword_only_and_takes_a_list_holding_any_mapping():
    parameter = inspect.signature(SuperGLM).parameters["terms"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="inside"):
        SuperGLM(terms={"x": Spline()})
    model = SuperGLM(terms=[{"x": Spline(k=6), "g": Categorical()}, "z"])
    assert list(model.features) == ["x", "g", "z"]


def test_a_factor_smooth_group_declared_as_a_term_is_refused():
    """The docs send factor smooths to features= and interactions= for this reason."""
    with pytest.raises(ValueError, match="FactorSmooth"):
        SuperGLM(terms=[s("x"), cat("g"), interaction(FactorSmooth("x", group="g"))])


def test_clones_keep_declared_interaction_options_and_names(data):
    """drop1, refit_unpenalised, Structure.apply and the editor refit clone the model.

    Rebuilt from a ``(left, right)`` pair, the tensor lost its margins and
    ``decompose`` and the named interaction its name, so drop1 tested a
    different reduced model and Structure.apply refitted a different model.
    """
    X, y = data
    # Margins smaller than the parents' four knots, so the options move the fit.
    tensor = ti("x", "z", n_knots=(2, 2), decompose=True)
    model = SuperGLM(
        terms=[
            s("x", k=6),
            s("z", k=6),
            cat("g"),
            tensor,
            interaction(SplineCategorical("x", "g"), name="x_by_g"),
        ]
    ).fit(X, y)
    reduced = SuperGLM(terms=[s("x", k=6), s("z", k=6), tensor]).fit(X, y)
    table = model.drop1(X, y).set_index("feature")
    assert table.loc["g", "deviance_reduced"] == reduced.result.deviance
    applied = Structure(features={}).apply(model).fit(X, y)
    assert np.array_equal(applied.result.beta, model.result.beta)
    assert applied.result.deviance == model.result.deviance


def test_terms_model_round_trips_and_reports_like_the_dict_model(data):
    X, y = data
    as_dict, as_list = _pair("fit", discrete=False)
    as_dict.fit(X, y)
    as_list.fit(X, y)

    restored = pickle.loads(pickle.dumps(as_list))
    assert np.array_equal(restored.predict(X), as_list.predict(X))
    _assert_same_fit(as_list.clone_unfitted().fit(X, y), as_dict, X, reml=False)
    assert Structure.from_model(as_list).to_json() == Structure.from_model(as_dict).to_json()
    assert str(as_list.summary()) == str(as_dict.summary())
    pd.testing.assert_frame_equal(as_list.drop1(X, y), as_dict.drop1(X, y))
    for name, table in as_dict.relativities().items():
        pd.testing.assert_frame_equal(as_list.relativities()[name], table)
    for name in ("x", "g", "x:z"):
        plots = [m.plot_data(name, ci=None) for m in (as_list, as_dict)]
        effects = [p["effect"] if "effect" in p else p["terms"][0]["effect"] for p in plots]
        pd.testing.assert_frame_equal(*effects)
    sessions = [EditorSession.from_model(m) for m in (as_list, as_dict)]
    assert list(sessions[0].terms) == list(sessions[1].terms)
    tables = [as_list.rating_table_payload(X, y), as_dict.rating_table_payload(X, y)]
    blocks = [[*payload.main_effects, *payload.interactions] for payload in tables]
    assert [b.name for b in blocks[0]] == [b.name for b in blocks[1]]
    for left, right in zip(*blocks, strict=True):
        pd.testing.assert_frame_equal(left.table, right.table)
    folds = KFold(n_splits=2, shuffle=True, random_state=0)
    scores = [
        cross_validate(m, X, y, cv=folds).fold_scores.filter(regex="^(?!.*time)")
        for m in (as_list, as_dict)
    ]
    pd.testing.assert_frame_equal(*scores)
