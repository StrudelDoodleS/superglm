"""OrderedCategorical as an interaction parent: build-side enabler."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, SuperGLM
from superglm.dm_builder import (
    should_discretize_spline_categorical_interaction,
    should_discretize_tensor_interaction,
)
from superglm.features.interaction import SplineCategorical, TensorInteraction
from superglm.features.ordered_categorical import (
    OrderedCategorical,
    resolve_interaction_parent,
)
from superglm.features.spline import Spline, _SplineBase

BANDS = ["18-25", "26-35", "36-45", "46-55", "56+"]


def _frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age_band": rng.choice(BANDS, n),
            "region": rng.choice(list("ABCD"), n),
            "power": rng.uniform(20.0, 200.0, n),
        }
    )
    band_effect = df["age_band"].map({b: v for b, v in zip(BANDS, [0.4, 0.1, 0.0, 0.1, 0.3])})
    eta = -1.5 + band_effect + 0.002 * df["power"]
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    return df, y


def _oc():
    return OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=4))


def test_resolver_is_identity_for_non_oc_and_none():
    x = np.array([1.0, 2.0])
    spec = Spline(kind="ps", n_knots=4)
    assert resolve_interaction_parent(spec, x) == (spec, x)
    assert resolve_interaction_parent(None, x) == (None, x)


def test_resolver_maps_oc_to_inner_spline_and_scores():
    spec = _oc()
    labels = np.array(["18-25", "56+", "36-45"], dtype=object)
    eff_spec, eff_x = resolve_interaction_parent(spec, labels)
    assert isinstance(eff_spec, _SplineBase)
    assert eff_spec is spec._spline
    expected = [spec._level_to_value[v] for v in labels]
    np.testing.assert_allclose(eff_x, expected)


def test_resolver_rejects_step_mode():
    with pytest.warns(FutureWarning):
        spec = OrderedCategorical(order=BANDS, basis="step")
    with pytest.raises(NotImplementedError, match="step"):
        resolve_interaction_parent(spec, np.array(BANDS, dtype=object))


def test_resolver_rejects_unseen_levels():
    spec = _oc()
    with pytest.raises(ValueError, match="unseen|levels"):
        resolve_interaction_parent(spec, np.array(["99+"], dtype=object))


def test_oc_categorical_interaction_fits():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "region": Categorical()},
        interactions=[("age_band", "region")],
    )
    model.fit_reml(df, y)
    assert np.isfinite(model._result.effective_df)


def test_oc_spline_tensor_interaction_fits():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model.fit_reml(df, y)
    assert np.isfinite(model._result.effective_df)


def test_oc_categorical_interaction_fits_discrete():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "region": Categorical()},
        interactions=[("age_band", "region")],
        discrete=True,
    )
    model.fit_reml(df, y)
    assert np.isfinite(model._result.effective_df)


def test_oc_spline_tensor_interaction_fits_discrete():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
        discrete=True,
    )
    model.fit_reml(df, y)
    assert np.isfinite(model._result.effective_df)


def test_discretize_gates_refuse_oc_parents_under_discrete_mode():
    # Both gates receive the ORIGINAL specs, so they see the OrderedCategorical
    # itself. Under discrete=True each must refuse, while the same pair with a
    # plain Spline in place of the OC is accepted — so it is the OC parent, not
    # the model flag or the interaction class, that produces the refusal.
    power = Spline(kind="ps", n_knots=5)
    region = Categorical()
    plain = Spline(kind="ps", n_knots=4)

    tensor = TensorInteraction("age_band", "power")
    assert (
        should_discretize_tensor_interaction(tensor, {"age_band": _oc(), "power": power}, True)
        is False
    )
    assert (
        should_discretize_tensor_interaction(tensor, {"age_band": plain, "power": power}, True)
        is True
    )

    spline_cat = SplineCategorical("age_band", "region")
    assert (
        should_discretize_spline_categorical_interaction(
            spline_cat, {"age_band": _oc(), "region": region}, True
        )
        is False
    )
    assert (
        should_discretize_spline_categorical_interaction(
            spline_cat, {"age_band": plain, "region": region}, True
        )
        is True
    )


class _SplineBackedOC(OrderedCategorical, _SplineBase):
    """An OrderedCategorical that also passes the spline-spec duck type."""

    discrete = None


def test_discretize_gates_refuse_oc_parents_that_pass_the_spline_duck_type():
    # The guards in both gates are a regression pin, not a behaviour change:
    # today `should_discretize` already refuses an OrderedCategorical because it
    # is not a `_SplineBase`, so removing the guards changes no outcome. This
    # double is the state the pin exists for. If OC ever gains the spline-spec
    # duck type, the discrete paths must still refuse it, because
    # `_compile_interaction_fast_discrete_metadata` reads the raw parent columns
    # as float64 and an OC column holds labels, not numbers.
    oc = _SplineBackedOC(order=BANDS, basis=Spline(kind="ps", n_knots=4))
    from superglm.dm_builder import should_discretize

    assert should_discretize(oc, True) is True  # the guard is the only refusal left

    power = Spline(kind="ps", n_knots=5)
    tensor = TensorInteraction("age_band", "power")
    assert (
        should_discretize_tensor_interaction(tensor, {"age_band": oc, "power": power}, True)
        is False
    )
    spline_cat = SplineCategorical("age_band", "region")
    assert (
        should_discretize_spline_categorical_interaction(
            spline_cat, {"age_band": oc, "region": Categorical()}, True
        )
        is False
    )


def test_oc_tensor_fit_matches_manual_score_mapping():
    # The OC×spline fit must equal the same model fitted on the scores directly.
    df, y = _frame()
    oc = _oc()
    model_oc = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model_oc.fit_reml(df, y)

    df_num = df.copy()
    df_num["age_band"] = df_num["age_band"].map(oc._level_to_value)
    model_num = SuperGLM(
        family="poisson",
        features={"age_band": Spline(kind="ps", n_knots=4), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model_num.fit_reml(df_num, y)
    np.testing.assert_allclose(model_oc._result.deviance, model_num._result.deviance, rtol=1e-6)

    # Deviance alone is invariant to how a term is recentered, and the two
    # models take different canonicalization paths: OrderedCategorical has no
    # ``_basis_matrix``, so neither the OC main effect nor the OC×power
    # interaction is spline-backed, while every term of the numeric reference
    # is. Assert on the fitted values too, which that asymmetry would move.
    eta_oc = model_oc._dm.matvec(model_oc._result.beta) + model_oc._result.intercept
    eta_num = model_num._dm.matvec(model_num._result.beta) + model_num._result.intercept
    np.testing.assert_allclose(eta_oc, eta_num, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        model_oc._link.inverse(eta_oc), model_num._link.inverse(eta_num), rtol=1e-10, atol=1e-12
    )


def test_oc_interaction_predict_round_trips_training_frame():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "region": Categorical()},
        interactions=[("age_band", "region")],
    )
    model.fit_reml(df, y)
    mu_train = model.predict(df)
    assert mu_train.shape == (len(df),)
    assert np.all(np.isfinite(mu_train)) and np.all(mu_train > 0)


def test_oc_tensor_predict_matches_manual_score_mapping():
    df, y = _frame()
    oc = _oc()
    model_oc = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model_oc.fit_reml(df, y)
    new = df.iloc[:200].copy()
    pred_oc = model_oc.predict(new)

    df_num = df.copy()
    df_num["age_band"] = df_num["age_band"].map(oc._level_to_value)
    model_num = SuperGLM(
        family="poisson",
        features={"age_band": Spline(kind="ps", n_knots=4), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model_num.fit_reml(df_num, y)
    new_num = new.copy()
    new_num["age_band"] = new_num["age_band"].map(oc._level_to_value)
    np.testing.assert_allclose(pred_oc, model_num.predict(new_num), rtol=1e-8)


def test_oc_interaction_predict_rejects_unseen_level():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "region": Categorical()},
        interactions=[("age_band", "region")],
    )
    model.fit_reml(df, y)
    bad = df.iloc[:5].copy()
    bad.loc[bad.index[0], "age_band"] = "99+"
    with pytest.raises(ValueError, match="unseen|levels"):
        model.predict(bad)


def test_oc_interaction_added_post_hoc_refits():
    # Exercises the config-template deepcopy path (the editor-clone contract):
    # add_interaction stores a deep-copied template that the next fit rebuilds.
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "region": Categorical()},
    )
    model.fit_reml(df, y)
    model._add_interaction("age_band", "region")
    model.fit_reml(df, y)
    mu = model.predict(df.iloc[:50])
    assert np.all(np.isfinite(mu)) and np.all(mu > 0)


def test_oc_interaction_metrics_on_evaluation_rows_resolve_the_parent():
    # The evaluation-design path re-evaluates the frozen prediction plan on the
    # requested rows, feeding the PARENT columns to the interaction transform.
    # An OC parent's column holds labels, so it must be resolved to its mapped
    # scores there exactly as the fit and predict paths resolve it.  Reachable
    # from public metrics whenever the rows (or offset) differ from the fit's.
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "region": Categorical()},
        interactions=[("age_band", "region")],
    )
    model.fit_reml(df, y)

    fit_metrics = model.metrics(df, y)
    assert fit_metrics._uses_fit_design
    # Same rows, same offset -- but not the fit's frame OBJECT, which is what
    # routes the diagnostics through EvaluationDesign instead of the fit design.
    eval_metrics = model.metrics(df.copy(), y)
    assert not eval_metrics._uses_fit_design
    np.testing.assert_allclose(eval_metrics.leverage, fit_metrics.leverage, rtol=1e-8, atol=1e-10)

    holdout = df.iloc[:200].copy()
    holdout_leverage = model.metrics(holdout, y[:200]).leverage
    assert holdout_leverage.shape == (len(holdout),)
    assert np.all(np.isfinite(holdout_leverage))


def test_oc_interaction_survives_discrete_mode():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
        discrete=True,
    )
    model.fit_reml(df, y)
    mu = model.predict(df.iloc[:100])
    assert np.all(np.isfinite(mu))
