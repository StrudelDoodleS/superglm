"""Ownership and configuration-intent tests for fit workspaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import GroupLasso, LogLink, NegativeBinomial, Numeric, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.model.fit_state import fitted_lambda2


def _poisson_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260717)
    x = np.linspace(-1.0, 1.0, 120)
    z = rng.normal(size=len(x))
    weights = rng.uniform(0.5, 1.5, size=len(x))
    y = rng.poisson(np.exp(0.15 + 0.25 * x - 0.1 * z)).astype(np.float64)
    return pd.DataFrame({"x": x, "z": z}), y, weights


def test_constructor_defensively_owns_mutable_inputs():
    feature = Spline(n_knots=5)
    penalty = GroupLasso(lambda1=None)
    family = NegativeBinomial(theta="auto")
    link = LogLink()
    interactions = [("x", "z")]

    model = SuperGLM(
        family=family,
        link=link,
        penalty=penalty,
        features={"x": feature, "z": Numeric()},
        interactions=interactions,
    )

    assert model.penalty is not penalty
    assert model.family is not family
    assert model.link is not link
    assert model.features["x"] is not feature
    interactions.clear()
    penalty.lambda1 = 99.0
    family.theta = 99.0
    feature.n_knots = 99
    assert model.penalty.lambda1 is None
    assert model.family.theta == "auto"
    assert model.features["x"].n_knots == 5
    assert model._pending_interactions == (("x", "z"),)


def test_features_property_returns_a_defensive_copy():
    model = SuperGLM(features={"x": Spline(n_knots=5)})

    exposed = model.features
    exposed["x"].n_knots = 99
    exposed["new"] = Numeric()

    assert model.features["x"].n_knots == 5
    assert "new" not in model.features


def test_auto_selection_penalty_intent_survives_successful_fit():
    X, y, weights = _poisson_data()
    model = SuperGLM(
        selection_penalty=None,
        features={"x": Numeric(), "z": Numeric()},
    )

    model.fit(X, y, sample_weight=weights)

    assert model.penalty.lambda1 is None
    assert model.selection_penalty_ > 0.0


def test_supported_selection_penalty_assignment_replaces_configuration_revision():
    model = SuperGLM(selection_penalty=None)
    before = model._config_revision

    model.selection_penalty = 0.25

    assert model._config_revision == before + 1
    assert model.penalty.lambda1 == pytest.approx(0.25)


def test_two_models_do_not_share_constructor_templates():
    penalty = GroupLasso(lambda1=0.1)
    feature = Spline(n_knots=5)

    left = SuperGLM(penalty=penalty, features={"x": feature})
    right = SuperGLM(penalty=penalty, features={"x": feature})

    left._specs["x"].n_knots = 7
    left.penalty.lambda1 = 0.2
    assert right.features["x"].n_knots == 5
    assert right.penalty.lambda1 == pytest.approx(0.1)


def test_constructor_owns_auxiliary_sequence_and_mapping_controls():
    splines = ["x"]
    n_knots = [5]
    n_bins = {"x": 32}
    model = SuperGLM(splines=splines, n_knots=n_knots, n_bins=n_bins)

    splines.append("z")
    n_knots[0] = 99
    n_bins["x"] = 2

    assert model._splines == ["x"]
    assert model._n_knots == [5]
    assert model._n_bins == {"x": 32}


def test_auto_nb_family_intent_survives_ordinary_fit():
    rng = np.random.default_rng(4815)
    n = 180
    x = rng.normal(size=n)
    mu = np.exp(0.2 + 0.25 * x)
    theta = 3.0
    rate = rng.gamma(shape=theta, scale=mu / theta)
    y = rng.poisson(rate).astype(np.float64)
    X = pd.DataFrame({"x": x})
    model = SuperGLM(
        family=NegativeBinomial(theta="auto"),
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    configured_family = model._family_config

    model.fit(X, y)

    assert model._family_config is configured_family
    assert model.family.theta == "auto"
    assert model.theta_ > 0.0
    assert model.distribution_.theta == pytest.approx(model.theta_)


def test_fitted_distribution_getter_is_defensive():
    rng = np.random.default_rng(4816)
    x = rng.normal(size=120)
    y = rng.poisson(np.exp(0.1 + 0.15 * x)).astype(np.float64)
    model = SuperGLM(
        family=NegativeBinomial(theta=2.0),
        selection_penalty=0.0,
        features={"x": Numeric()},
    ).fit(pd.DataFrame({"x": x}), y)
    fitted_theta = model.theta_

    exposed = model.distribution_
    exposed.theta = 99.0

    assert model.distribution_ is not exposed
    assert model.theta_ == pytest.approx(fitted_theta)
    assert model.distribution_.theta == pytest.approx(fitted_theta)


def test_successful_fit_preserves_smoothing_configuration_identity():
    X, y, weights = _poisson_data()
    model = SuperGLM(
        selection_penalty=0.0,
        features={"x": Numeric(), "z": Numeric()},
    )
    model.lambda2 = {"x": 0.2, "z": 0.4}
    configured_lambda2 = model._lambda2_config

    model.fit(X, y, sample_weight=weights)

    assert model._lambda2_config is configured_lambda2
    assert model.lambda2 == {"x": 0.2, "z": 0.4}


def test_post_fit_lambda2_assignment_does_not_reinterpret_installed_state():
    X, y, weights = _poisson_data()

    def fitted_model() -> SuperGLM:
        return SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            spline_penalty=0.25,
            features={"x": Spline(n_knots=6, penalty="ssp"), "z": Numeric()},
        ).fit(X, y, sample_weight=weights)

    reference = fitted_model()
    staged = fitted_model()
    installed_state = staged._fit_state
    installed_revision = staged._fit_revision

    staged.lambda2 = 9.0

    assert staged._fit_state is installed_state
    assert staged._fit_revision == installed_revision
    assert staged.lambda2 == pytest.approx(9.0)
    assert fitted_lambda2(staged) == pytest.approx(0.25)
    assert staged._fit_state.resolved_lambda2 == pytest.approx(0.25)
    np.testing.assert_allclose(staged.predict(X), reference.predict(X), rtol=0.0, atol=0.0)
    assert str(staged.summary(alpha=0.037)) == str(reference.summary(alpha=0.037))
    staged_covariance, staged_groups = staged._coef_covariance
    reference_covariance, reference_groups = reference._coef_covariance
    np.testing.assert_allclose(staged_covariance, reference_covariance, rtol=0.0, atol=0.0)
    assert staged_groups == reference_groups

    staged.fit(X, y, sample_weight=weights)

    assert fitted_lambda2(staged) == pytest.approx(9.0)
    assert staged._fit_state.resolved_lambda2 == pytest.approx(9.0)


def test_post_construction_interaction_is_rebuilt_on_every_fit():
    first = _poisson_data()
    X_a, y_a, weights_a = first
    X_b = X_a.assign(x=-X_a["x"], z=X_a["z"] + 0.5)
    rng = np.random.default_rng(8128)
    y_b = rng.poisson(np.exp(0.1 - 0.2 * X_b["x"] + 0.1 * X_b["z"])).astype(np.float64)

    sequential = SuperGLM(
        selection_penalty=0.0,
        features={"x": Numeric(), "z": Numeric()},
    )
    sequential._add_interaction("x", "z")
    sequential.fit(X_a, y_a, sample_weight=weights_a)
    sequential.fit(X_b, y_b, sample_weight=weights_a)

    fresh = SuperGLM(
        selection_penalty=0.0,
        features={"x": Numeric(), "z": Numeric()},
    )
    fresh._add_interaction("x", "z")
    fresh.fit(X_b, y_b, sample_weight=weights_a)

    assert sequential._interaction_order == ["x:z"]
    assert [group.feature_name for group in sequential._groups].count("x:z") == 1
    np.testing.assert_allclose(
        sequential.predict(X_b),
        fresh.predict(X_b),
        rtol=1e-12,
        atol=1e-12,
    )


def test_post_fit_interaction_update_does_not_requeue_constructor_interactions():
    X, y, weights = _poisson_data()
    X = X.assign(w=X["x"] * X["z"])
    model = SuperGLM(
        selection_penalty=0.0,
        features={"x": Numeric(), "z": Numeric(), "w": Numeric()},
        interactions=[("x", "z")],
    ).fit(X, y, sample_weight=weights)

    model._add_interaction("x", "w")
    cloned = model.clone_unfitted()

    assert model._pending_interactions == ()
    assert model._config.interactions == ()
    assert cloned._pending_interactions == ()
    assert cloned._interaction_order == ["x:z", "x:w"]


def test_public_mutable_configuration_getters_are_defensive():
    model = SuperGLM(
        family=NegativeBinomial(theta="auto"),
        link=LogLink(),
        penalty=GroupLasso(lambda1=0.1),
        features={"x": Numeric()},
    )
    model.lambda2 = {"x": 0.2}
    before_revision = model._config_revision

    exposed_family = model.family
    exposed_link = model.link
    exposed_penalty = model.penalty
    exposed_lambda2 = model.lambda2
    exposed_family.theta = 99.0
    exposed_link.external_marker = True
    exposed_penalty.lambda1 = 99.0
    exposed_lambda2["x"] = 99.0

    assert model.family is not exposed_family
    assert model.link is not exposed_link
    assert model.penalty is not exposed_penalty
    assert model.lambda2 is not exposed_lambda2
    assert model.family.theta == "auto"
    assert not hasattr(model.link, "external_marker")
    assert model.penalty.lambda1 == pytest.approx(0.1)
    assert model.lambda2 == {"x": 0.2}
    assert model._config.family.theta == "auto"
    assert model._config.penalty.lambda1 == pytest.approx(0.1)
    assert model._config.lambda2 == {"x": 0.2}
    assert model._config_revision == before_revision


def test_editor_implicit_refresh_rejects_mutated_retained_response():
    x = np.linspace(0.0, 1.0, 180)
    X = pd.DataFrame({"x": x})
    y = 0.2 + np.sin(5.0 * x) + 0.01 * np.cos(17.0 * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Spline(n_knots=7)},
    ).fit(X, y)
    session = EditorSession.from_model(model, terms=["x"])
    session.select_x("x", 0.25, 0.75)
    session.shift("x", 0.2)
    installed_dict = model.__dict__
    installed_result = model.result
    y[0] += 10.0

    with pytest.raises(RuntimeError, match="retained fit data.*mutated"):
        session.to_model()

    assert model.__dict__ is installed_dict
    assert model.result is installed_result


@pytest.mark.parametrize("mutation_source", ["exported_view", "constructor_array"])
def test_editor_implicit_refresh_rejects_numpy_alias_mutation_of_retained_frame(
    mutation_source,
):
    x = np.linspace(0.0, 1.0, 180)
    caller_array = x.reshape(-1, 1).copy()
    if mutation_source == "constructor_array":
        X = pd.DataFrame(caller_array, columns=["x"], copy=False)
        alias = caller_array
    else:
        X = pd.DataFrame({"x": x})
        alias = None
    y = 0.2 + np.sin(5.0 * x) + 0.01 * np.cos(17.0 * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Spline(n_knots=7)},
    ).fit(X, y)
    session = EditorSession.from_model(model, terms=["x"])
    session.select_x("x", 0.25, 0.75)
    session.shift("x", 0.2)
    installed_dict = model.__dict__
    installed_result = model.result

    if alias is None:
        alias = X["x"].to_numpy(copy=False)
        alias.setflags(write=True)
    assert np.shares_memory(alias, X["x"].to_numpy(copy=False))

    if alias.ndim == 2:
        alias[0, 0] += 0.25
    else:
        alias[0] += 0.25

    with pytest.raises(RuntimeError, match="retained fit data.*mutated"):
        session.to_model()

    assert model.__dict__ is installed_dict
    assert model.result is installed_result
