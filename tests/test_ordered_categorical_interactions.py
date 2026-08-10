"""OrderedCategorical as an interaction parent: build-side enabler."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, SuperGLM
from superglm.dm_builder import (
    should_discretize_spline_categorical_interaction,
    should_discretize_tensor_interaction,
)
from superglm.features.factor_smooth import FactorSmooth
from superglm.features.interaction import SplineCategorical, TensorInteraction
from superglm.features.ordered_categorical import (
    OrderedCategorical,
    resolve_interaction_parent,
    resolve_interaction_parent_of,
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


def _pre_024_step_pickle() -> OrderedCategorical:
    """The shape a pre-0.24 step-mode pickle restores: no inner spline.

    Step mode was removed in 0.24.0, so the only way such a spec reaches the
    interaction machinery is a pickle, whose ``__dict__`` restores without
    ``__init__`` running.
    """
    spec = OrderedCategorical.__new__(OrderedCategorical)
    spec.__dict__.update({"basis": "step", "_spline": None, "_spline_obj": None})
    return spec


def test_pre_024_step_pickle_parent_is_rejected_when_the_interaction_is_added():
    """A spec without an inner spline cannot parent an interaction, and the
    refusal happens where the pair is declared -- naming both features and
    the removal -- rather than mid design-matrix build after the caller has
    committed a fit."""
    step = _pre_024_step_pickle()
    model = SuperGLM(
        family="poisson",
        features={"age_band": step, "power": Spline(kind="ps", n_knots=4)},
    )
    with pytest.raises(NotImplementedError, match=r"\('power', 'age_band'\).*inner spline"):
        model._add_interaction("power", "age_band")
    assert model._interaction_specs == {}  # nothing was half-registered


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


def test_factor_smooth_keeps_label_levels_over_an_oc_group_main():
    """A FactorSmooth's second parent is a GROUPING column, not a margin.

    The build loop resolves interaction parents so an OC margin contributes its
    mapped level scores -- but a FactorSmooth factorizes its group column into
    the term's own level set, so resolving it would silently re-key that
    identity to ``[0.0, 0.25, ...]`` and every by-label lookup its inference
    exposes would fail on the fitted labels.  Reachable from public API: an OC
    main is the ONLY main a FactorSmooth's group column may carry (a
    Categorical there is rejected as duplicated group-intercept geometry).
    """
    from superglm import FactorSmooth

    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)},
        interactions=[FactorSmooth("power", group="age_band", basis="fs", kind="ps", k=5)],
        selection_penalty=0.0,
    )
    model.fit_reml(df, y)

    spec = model._interaction_specs["power:age_band:fs"]
    assert spec._levels == BANDS
    assert list(spec._level_to_code) == BANDS

    # ... and the public per-level accessor resolves a fitted label rather than
    # raising KeyError on it (the level table always carries every level; the
    # curves are what `levels=` selects).
    result = model.factor_smooth("power:age_band:fs", levels=["18-25"], grid=8)
    assert set(result.curves["level"]) == {"18-25"}
    assert list(result.table["level"]) == BANDS

    # The predict and evaluation-design paths resolve parents from the same
    # fitted specs, so they have to make the SAME exception or the term goes
    # silent: mapped scores against label levels index to -1 everywhere, and
    # `unseen="population"` then serves a zero block without raising.  Pin it
    # against the fit itself -- predictions on the training frame must
    # reproduce the fitted deviance, which a zeroed block would not.
    mu = model.predict(df)
    deviance = float(np.sum(model.distribution_.deviance_unit(y, mu)))
    assert deviance == pytest.approx(model._result.deviance, rel=1e-9)

    # `df.copy()` is not the fit's frame OBJECT, which routes the diagnostics
    # through EvaluationDesign -- the third path that resolves parents.
    fit_metrics = model.metrics(df, y)
    eval_metrics = model.metrics(df.copy(), y)
    assert fit_metrics._uses_fit_design and not eval_metrics._uses_fit_design
    np.testing.assert_allclose(eval_metrics.leverage, fit_metrics.leverage, rtol=1e-8, atol=1e-10)


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


def _poisson_deviance(y, mu):
    term = np.where(y > 0, y * np.log(np.maximum(y, 1e-300) / mu), 0.0)
    return 2.0 * float(np.sum(term - (y - mu)))


@pytest.mark.parametrize("level_dtype", ["int", "str"])
def test_predict_reproduces_the_fit_for_non_string_partner_levels(level_dtype):
    """An OC x Categorical fit must predict what it fitted, whatever the level type.

    ``update_reml_r_inv`` recovers each level from a group's DISPLAY name, so a
    non-string level arrives stringified -- integer ``2`` as ``"2"``.
    ``SplineCategorical.score()`` looks the ORIGINAL key up, so keying the
    reparametrisation dict on the parsed text drops it for every non-string
    level and ``predict()`` silently diverges from the design it was fitted on.
    The string case is the control: it was always correct and must stay so.
    """
    rng = np.random.default_rng(0)
    n = 4000
    codes = rng.integers(0, 4, n)
    df = pd.DataFrame(
        {
            "age_band": pd.Categorical(rng.choice(BANDS, n), categories=BANDS, ordered=True),
            "grp": codes if level_dtype == "int" else codes.astype(str),
        }
    )
    band_codes = pd.Categorical(df["age_band"], categories=BANDS).codes
    y = rng.poisson(np.exp(0.1 + 0.09 * band_codes + 0.06 * codes)).astype(float)

    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "grp": Categorical()},
        interactions=[("age_band", "grp")],
    )
    model.fit_reml(df, y)

    # Not a value pin: predict() on the training frame must reproduce the
    # fitted deviance exactly, because it is the same design.
    assert _poisson_deviance(y, model.predict(df)) == pytest.approx(
        model._result.deviance, rel=1e-9, abs=1e-9
    )


def test_classifier_decision_function_resolves_oc_interaction_parents():
    """``decision_function`` must resolve parents like every other predict path.

    It builds eta itself rather than routing through ``model.predict()``, so it
    needs the same ``resolve_interaction_parent_of`` step: a spline-mode OC
    enters an interaction through its mapped level scores.  Handing the raw
    labels to ``ispec.transform`` raises on string levels and would score a
    different linear predictor on numeric ones.
    """
    pytest.importorskip("sklearn")
    from superglm.sklearn import SuperGLMClassifier

    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame(
        {
            "age_band": pd.Categorical(rng.choice(BANDS, n), categories=BANDS, ordered=True),
            "power": rng.uniform(20.0, 200.0, n),
        }
    )
    lin = (
        0.012 * df["power"].to_numpy()
        + 0.2 * pd.Categorical(df["age_band"], categories=BANDS).codes
    )
    yb = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(lin - 1.4)))).astype(int)

    clf = SuperGLMClassifier(features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)})
    clf.fit(df, yb)
    clf._model._add_interaction("age_band", "power")
    clf._model.fit_reml(df, yb.astype(float))

    # predict_proba goes through model.predict(), which already resolves;
    # decision_function builds eta itself.  They must agree.
    proba = clf.predict_proba(df)[:, 1]
    np.testing.assert_allclose(
        clf.decision_function(df), np.log(proba / (1.0 - proba)), rtol=1e-9, atol=1e-9
    )


def _specials_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    band = rng.choice(BANDS, n)
    band = np.where(rng.random(n) < 0.18, "MISSING", band)
    df = pd.DataFrame(
        {
            "age_band": band,
            "region": rng.choice(list("ABCD"), n),
            "power": rng.uniform(20.0, 200.0, n),
        }
    )
    y = rng.poisson(np.exp(-1.5 + 0.002 * df["power"])).astype(np.float64)
    return df, y


def _oc_specials():
    return OrderedCategorical(
        order=BANDS,
        specials=["MISSING"],
        basis=Spline(kind="ps", n_knots=4),
    )


def test_resolver_rejects_a_specials_parent():
    """FALSE TODAY: the resolver has no specials rule, so it validates MISSING
    as a known level and returns NaN scores from _map_to_numeric instead."""
    spec = _oc_specials()
    with pytest.raises(NotImplementedError, match="specials"):
        resolve_interaction_parent(spec, np.array(["18-25", "MISSING"], dtype=object))


def test_specials_parent_is_rejected_where_the_interaction_is_declared():
    """FALSE TODAY: the declaration guard only knows about basis='step', so a
    specials pair registers and fails much later, mid design-matrix build."""
    df, y = _specials_frame()

    with pytest.raises(NotImplementedError, match="age_band.*specials"):
        SuperGLM(
            family="poisson",
            features={"age_band": _oc_specials(), "region": Categorical()},
            interactions=[("age_band", "region")],
        ).fit_reml(df, y)

    # ... and through the incremental API, with the OC in either position
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc_specials(), "power": Spline(kind="ps", n_knots=4)},
    )
    with pytest.raises(NotImplementedError, match=r"\('power', 'age_band'\)"):
        model._add_interaction("power", "age_band")
    assert model._interaction_specs == {}  # nothing was half-registered


def test_an_explicit_interaction_spec_cannot_smuggle_a_specials_parent_past_it():
    """FALSE TODAY: base.py:805-811 deep-copies anything carrying .parent_names
    and .name straight into _interaction_specs without calling add_interaction,
    so the declaration guard never runs for an explicit spec object.  Nothing
    then refuses the pair -- it reaches the build with the special's score NaN.
    The resolver guard at dm_builder.py:1071 is what stops this form."""
    df, y = _specials_frame()
    ti = TensorInteraction("age_band", "power")
    ti.name = "age_band:power"
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc_specials(), "power": Spline(kind="ps", n_knots=4)},
        interactions=[ti],
    )
    with pytest.raises(NotImplementedError, match="specials"):
        model.fit_reml(df, y)


def test_a_specials_term_may_still_be_a_factor_smooth_group():
    """Pins the EXEMPTION AT ITS EXACT WIDTH, both edges asserted at the one
    seam that decides it.  The refusal lives in resolve_interaction_parent, so
    resolve_interaction_parent_of hands a FactorSmooth its group column
    untouched -- labels, never scores -- while refusing the SAME spec and the
    SAME column for an interaction that reads both parents marginally.
    Asserting only that the FactorSmooth fit runs would pass against unguarded
    code too, since unguarded code refuses nothing anywhere."""
    spec = _oc_specials()
    labels = np.array(["18-25", "MISSING", "56+"], dtype=object)

    eff_spec, eff_x = resolve_interaction_parent_of(
        FactorSmooth(variable="power", group="age_band"), spec, labels
    )
    assert eff_spec is spec  # not swapped for spec._spline
    np.testing.assert_array_equal(eff_x, labels)  # not mapped to level scores

    # ... and the exemption is the interaction's, not the spec's: the same
    # parent under a marginal-reading interaction is refused.
    with pytest.raises(NotImplementedError, match="specials"):
        resolve_interaction_parent_of(TensorInteraction("age_band", "power"), spec, labels)

    df, y = _specials_frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc_specials(), "power": Spline(kind="ps", n_knots=4)},
        interactions=[FactorSmooth(variable="power", group="age_band")],
    )
    model.fit_reml(df, y)
    assert np.isfinite(model._result.effective_df)
    # The group kept its label identity, the special included as a level of it.
    fs = model._interaction_specs["power:age_band:fs"]
    assert set(fs._levels) == set(BANDS) | {"MISSING"}
