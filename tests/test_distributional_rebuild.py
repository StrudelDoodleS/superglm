"""SuperLSS rebuilds with replaced terms and replays the fit it was made with."""

from __future__ import annotations

import types

import joblib
import numpy as np
import pandas as pd
import pytest

from superglm import GammaLS, LogNormalLS, SuperLSS, s, ti
from tests._lss_editor_fixtures import EPS, lognormal_case, nb_case


def _vector(model):
    return np.array(list(model.coef_.values()))


def _block_length(model, term):
    return sum(1 for name in model.coef_ if name.startswith(f"{term}["))


def _refit(case, clone):
    return clone._replay_fit(case.X, case.y, case.sample_weight, case.offsets)


def test_rebuild_with_no_replacements_refits_to_the_same_fit():
    case = lognormal_case()
    m = case.model
    clone = m._clone_with_replaced_terms({})
    assert clone._model is None
    again = _refit(case, clone)
    assert again is clone
    before = _vector(m)
    scale = np.linalg.norm(before)
    np.testing.assert_allclose(_vector(again), before, rtol=0, atol=1e3 * EPS * before.size * scale)
    assert again.smoothing_parameters_.keys() == m.smoothing_parameters_.keys()


def test_replaced_term_takes_the_new_spec():
    case = lognormal_case()
    m = case.model
    clone = m._clone_with_replaced_terms({"mean:age": s("age", k=5)})
    again = _refit(case, clone)
    assert _block_length(again, "mean:age") == 4
    assert _block_length(again, "scale:age") == _block_length(m, "scale:age") == 5
    by_name = {p.name: p for p in again.predictors}
    assert by_name["mean"].features["age"].n_knots == 3
    assert by_name["scale"].features["age"].n_knots == 4
    # The source model is untouched.
    assert {p.name: p for p in m.predictors}["mean"].features["age"].n_knots == 6
    assert _block_length(m, "mean:age") == 7


def test_settings_replay_the_estimator():
    case = lognormal_case()
    m = case.model
    clone = m._clone_with_replaced_terms({})
    assert m._fit_settings["method"] == "fit_reml"
    assert clone._fit_settings["outer"] == "efs+newton"
    assert clone._fit_settings == m._fit_settings

    family = LogNormalLS()
    fixed = SuperLSS(
        family,
        family.mean(s("age", k=6), s("power", k=5)),
        family.scale(s("age", k=5)),
    ).fit(
        case.X,
        case.y,
        lambdas={"mean:age#wiggle": 3.0, "mean:power#wiggle": 4.0, "scale:age#wiggle": 5.0},
    )
    assert fixed._fit_settings["method"] == "fit"
    dropped = fixed._clone_with_replaced_terms({"mean:age": s("age", k=7)})
    assert dropped._fit_settings["lambdas"] == {
        "mean:power#wiggle": 4.0,
        "scale:age#wiggle": 5.0,
    }
    assert fixed._fit_settings["lambdas"]["mean:age#wiggle"] == 3.0


def test_settings_exclude_data_and_row_options():
    case = lognormal_case()
    assert {
        "X",
        "y",
        "sample_weight",
        "offsets",
        "truncation",
        "censored",
        "phase_recorder",
    }.isdisjoint(case.model._fit_settings)


def test_initial_lambda_is_not_dropped_by_an_unrelated_replacement():
    case = lognormal_case()
    family = LogNormalLS()
    model = SuperLSS(family, family.mean(s("age", k=6)), family.scale(s("age", k=5))).fit_reml(
        case.X, case.y, lambdas={"scale:age#wiggle": 2.0}, initial_lambda=0.5
    )
    clone = model._clone_with_replaced_terms({"scale:age": s("age", k=4)})
    assert clone._fit_settings["initial_lambda"] == 0.5
    assert clone._fit_settings["lambdas"] == {}
    # An empty mapping replays: the replaced term's strength is estimated again.
    again = _refit(case, clone)
    assert again._fit_settings["lambdas"] == {}
    assert again.smoothing_parameters_["scale:age#wiggle"] != 2.0


def test_interaction_terms_cannot_be_replaced():
    case = lognormal_case()
    family = LogNormalLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=6), s("power", k=5), ti("age", "power")),
        family.scale(s("age", k=5)),
    ).fit_reml(case.X, case.y, outer="efs+newton")
    with pytest.raises(ValueError, match="Interaction terms cannot be replaced: mean:age:power"):
        model._clone_with_replaced_terms({"mean:age:power": s("age", k=5)})
    # The interaction survives a clone that replaces something else.
    clone = model._clone_with_replaced_terms({"mean:age": s("age", k=5)})
    assert tuple(dict({p.name: p for p in clone.predictors}["mean"].interaction_specs)) == (
        "age:power",
    )
    _refit(case, clone)


@pytest.mark.parametrize("key", ["mean:nope", "nope:age", "age"])
def test_unknown_key_raises(key):
    with pytest.raises(KeyError):
        lognormal_case().model._clone_with_replaced_terms({key: s("age", k=5)})


def test_constructor_settings_survive():
    case = lognormal_case()
    family = LogNormalLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=6)),
        family.scale(s("age", k=5)),
        discrete=True,
        n_bins=64,
        weight_semantics="frequency",
        separation="ignore",
        coefficient_curvature="fisher",
    )
    clone = model._clone_with_replaced_terms({"mean:age": s("age", k=7)})
    assert clone.discrete is True
    assert clone.n_bins == 64
    assert clone._weight_contract.semantics == "frequency"
    assert clone._separation == "ignore"
    assert clone._coefficient_curvature == "fisher"
    assert type(clone._family) is LogNormalLS
    assert clone._model is None
    assert case.model._fit_settings is not None


def test_offsets_replay_on_a_count_model():
    case = nb_case()
    again = _refit(case, case.model._clone_with_replaced_terms({}))
    before = _vector(case.model)
    scale = np.linalg.norm(before)
    np.testing.assert_allclose(_vector(again), before, rtol=0, atol=1e3 * EPS * before.size * scale)


def test_a_bounded_fit_is_not_replayed():
    rng = np.random.default_rng(20261103)
    X = pd.DataFrame({"age": rng.uniform(18.0, 90.0, 150)})
    y = rng.gamma(4.0, 100.0, 150)
    censored = np.zeros(150, dtype=bool)
    censored[:6] = True
    family = GammaLS()
    model = SuperLSS(family, family.mean(s("age", k=5)), family.scale())
    model.fit_reml(X, y, truncation=(1.0, None), censored=censored)
    with pytest.raises(ValueError, match="fitted with truncated or censored rows"):
        model._clone_with_replaced_terms({})


def test_replay_without_settings_raises():
    family = LogNormalLS()
    model = SuperLSS(family, family.mean(s("age")), family.scale(s("age")))
    with pytest.raises(RuntimeError, match="no recorded fit settings"):
        model._replay_fit(pd.DataFrame({"age": [1.0]}), np.array([1.0]))


def test_fit_settings_survive_pickle(tmp_path):
    case = lognormal_case()
    path = tmp_path / "m.joblib"
    joblib.dump(case.model, path)
    restored = joblib.load(path)
    assert restored._fit_settings == case.model._fit_settings
    assert restored._clone_with_replaced_terms({})._fit_settings["outer"] == "efs+newton"


def test_fit_takes_the_last_fits_smoothing_parameters():
    # smoothing_parameters_ is a read-only mapping view; recording it must not copy it deeply.
    case = lognormal_case()
    lambdas = case.model.smoothing_parameters_
    fitted = case.model._clone_with_replaced_terms({}).fit(case.X, case.y, lambdas=lambdas)
    assert fitted._fit_settings["lambdas"] == dict(lambdas)
    assert type(fitted._fit_settings["lambdas"]) is dict
    again = _refit(case, fitted._clone_with_replaced_terms({}))
    before = _vector(fitted)
    scale = np.linalg.norm(before)
    np.testing.assert_allclose(_vector(again), before, rtol=0, atol=1e3 * EPS * before.size * scale)


def test_fit_reml_takes_a_mapping_proxy_of_lambdas():
    case = lognormal_case()
    family = LogNormalLS()
    fixed = types.MappingProxyType({"scale:age#wiggle": 2.0})
    model = SuperLSS(family, family.mean(s("age", k=6)), family.scale(s("age", k=5))).fit_reml(
        case.X, case.y, lambdas=fixed
    )
    assert model._fit_settings["lambdas"] == {"scale:age#wiggle": 2.0}
