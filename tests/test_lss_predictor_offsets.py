"""Predictor offsets are fixed link-scale contributions, never fitted terms."""

from __future__ import annotations

import copy
import inspect
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from superglm import GaussianLS, SuperLSS, bind_predictor, cross_validate, s
from superglm.distributional.families.dirichlet import DirichletLSS
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.predictor import Predictor
from superglm.distributional.weights import WeightContract
from superglm.features import Numeric


def _data(n=64):
    rng = np.random.default_rng(953)
    x = np.linspace(-1, 1, n)
    a, b = 0.4 * np.sin(5 * x), 0.2 * np.cos(7 * x)
    frame = pd.DataFrame({"x": x, "a": a, "b": b, "scale_offset": 0.1 * x})
    y = 2 + 0.5 * x + a + b + rng.normal(size=n) * np.exp(-1 + 0.1 * x)
    return frame, y


def _model(location, scale="scale_offset", *, smooth=False, discrete=False):
    family = GaussianLS()
    return SuperLSS(
        family,
        family.location(s("x", k=5) if smooth else "x", offset=location),
        family.scale(offset=scale),
        discrete=discrete,
    )


@pytest.mark.parametrize("method", ["fit", "fit_reml"])
def test_fitting_has_no_offset_argument_or_positional_bridge(method):
    fit = getattr(SuperLSS, method)
    assert "offsets" not in inspect.signature(fit).parameters
    family = GaussianLS()
    model = SuperLSS(family, family.location(), family.scale())
    X, y = _data()
    with pytest.raises(TypeError, match="offsets"):
        getattr(model, method)(X, y, offsets={})
    with pytest.raises(TypeError, match="positional"):
        getattr(model, method)(X, y, None, {})


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_columns_arrays_and_summed_sources_match_complete_fits(discrete, reml):
    X, y = _data()
    sources = [["a", "b"], [X.a.to_numpy(), X.b.to_numpy()], (X.a + X.b).tolist()]
    models = [_model(source, smooth=reml, discrete=discrete) for source in sources]
    for model in models:
        if reml:
            model.fit_reml(X, y, max_reml_iter=4, outer="efs")
        else:
            model.fit(X, y)
    first = models[0]
    for other in models[1:]:
        np.testing.assert_allclose(list(other.coef_.values()), list(first.coef_.values()))
        np.testing.assert_allclose(other.covariance_, first.covariance_)
        assert other.result_.total_effective_df == pytest.approx(first.result_.total_effective_df)
        assert other.result_.log_likelihood == pytest.approx(first.result_.log_likelihood)
        np.testing.assert_allclose(
            other.predict_parameters(X, offsets={"location": X.a + X.b}),
            first.predict_parameters(X),
        )
    assert not any(name.endswith(":a") or name.endswith(":b") for name in first.coef_)
    assert set(first.predictors[0].features) == {"x"}


def test_fixed_coefficients_match_independent_gaussian_solution_and_internal_control():
    X, y = _data()
    model = _model(["a", "b"], scale=None).fit(X, y)
    design = np.column_stack([np.ones(len(X)), X.x])
    beta = np.linalg.lstsq(design, y - X.a - X.b, rcond=None)[0]
    sigma = np.sqrt(np.mean((y - X.a - X.b - design @ beta) ** 2))
    np.testing.assert_allclose(model.predict_parameters(X)["location"], design @ beta + X.a + X.b)
    np.testing.assert_allclose(model.predict_parameters(X)["scale"], sigma, rtol=1e-6)
    control = fit_dense_distributional(
        X,
        y,
        family=GaussianLS(),
        predictors=(Predictor("location", {"x": Numeric()}), Predictor("scale", {})),
        weight_contract=WeightContract("prior"),
        offsets={"location": X.a + X.b},
        lambdas={},
    )
    np.testing.assert_allclose(model.covariance_, control.covariance)
    rows = model._require_fitted().fit_state.retained_rows
    np.testing.assert_allclose(
        rows.null_eta[:, 0] - rows.offsets["location"],
        model._require_fitted().fit_state.null_model.coefficients[0],
    )


def test_mixed_sources_replace_only_array_contributions_on_new_rows():
    X, y = _data()
    model = _model(["a", X.b.to_numpy()], scale=X.scale_offset.to_numpy()).fit(X, y)
    new = X.iloc[[8, 2, 17]].copy()
    with pytest.raises(ValueError, match="location.*replacement"):
        model.predict_parameters(new)
    replacement = {"location": new.b.to_numpy() + 0.7, "scale": np.zeros(3)}
    eta = model.predict_link(new, offsets=replacement).to_numpy()
    coef = model.coef_
    np.testing.assert_allclose(
        eta[:, 0], coef["location:(intercept)"] + coef["location:x"] * new.x + new.a + new.b + 0.7
    )
    np.testing.assert_allclose(eta[:, 1], coef["scale:(intercept)"])
    columns = _model(["a", "b"]).fit(X, y)
    with pytest.raises(ValueError, match="location.*(duplicate|column)"):
        columns.predict_parameters(new, offsets={"location": new.a + new.b})


def test_template_copying_freezes_arrays_and_keeps_declarations(tmp_path):
    X, y = _data()
    original = X.a.to_numpy(copy=True)
    family = GaussianLS()
    bound = bind_predictor(family, "location", "x", offset=[original, "b"])
    original[:] = 999
    model = SuperLSS(family, bound, family.scale()).fit(X, y)
    expected = model.predict_parameters(X, offsets={"location": X.a})
    for template in (bound.template, model.predictors[0], model.predictors_[0]):
        assert not template.offset[0].flags.writeable
        assert template.offset[1] == "b"
        np.testing.assert_array_equal(template.offset[0], X.a)
    clone = model._clone_with_replaced_terms({})
    clone._replay_fit(X, y)
    np.testing.assert_allclose(clone.predict_parameters(X, offsets={"location": X.a}), expected)
    path = tmp_path / "offset.joblib"
    joblib.dump(model, path)
    for restored in (copy.deepcopy(model), pickle.loads(pickle.dumps(model)), joblib.load(path)):
        np.testing.assert_array_equal(
            restored.predict_parameters(X, offsets={"location": X.a}), expected
        )
        assert not restored.predictors[0].offset[0].flags.writeable


def test_shape_and_label_validation_precedes_take_but_values_follow_it():
    X, y = _data()
    weights = np.ones(len(X))
    weights[5] = 0
    bad = X.a.astype(object).to_numpy(copy=True)
    bad[5] = "ignored"
    X.loc[5, "b"] = np.nan
    model = _model([bad, "b"]).fit(X, y, sample_weight=weights)
    keep = np.flatnonzero(weights)
    np.testing.assert_allclose(
        model._require_fitted().layout.predictors[0].offset, X.a.iloc[keep] + X.b.iloc[keep]
    )
    with pytest.raises(ValueError, match="length"):
        _model(bad[:-1]).fit(X, y, sample_weight=weights)
    with pytest.raises(ValueError, match="original.*5"):
        _model(bad).fit(X, y)
    with pytest.raises(ValueError, match="index"):
        _model(pd.Series(np.zeros(len(X)), index=X.index[::-1])).fit(X, y, sample_weight=weights)
    with pytest.raises(ValueError, match="missing.*missing"):
        _model("missing").fit(X, y, sample_weight=weights)
    with pytest.raises(ValueError, match="unique"):
        _model("a").fit(pd.concat([X, X[["a"]]], axis=1), y)


@pytest.mark.parametrize("bad", [[], [[1, 2], [3]], ["a", 1.0], lambda x: x, np.ones((2, 3))])
def test_malformed_declarations_are_refused(bad):
    with pytest.raises((TypeError, ValueError), match="offset"):
        GaussianLS().location(offset=bad)


def test_columns_resolve_again_at_fit_and_prediction_and_polars_works():
    import polars as pl

    X, y = _data()
    model = _model(["a", "b"]).fit(pl.from_pandas(X), y)
    before = model.predict_link(pl.from_pandas(X)).to_numpy()
    X["a"] += 0.5
    np.testing.assert_allclose(model.predict_link(X).to_numpy()[:, 0], before[:, 0] + 0.5)
    model.fit(X, y)
    np.testing.assert_allclose(model.predict_link(X).to_numpy(), before, rtol=1e-6, atol=1e-6)


def test_scalar_cv_slices_array_declarations_and_scores_column_offsets():
    X, y = _data()
    arrays = _model([X.a.to_numpy(), "b"])
    columns = _model(["a", "b"])
    kwargs = dict(
        cv=KFold(2, shuffle=True, random_state=1),
        fit_kwargs={"lambdas": {}},
        return_oof=True,
        return_estimators=True,
    )
    a = cross_validate(arrays, X, y, **kwargs)
    b = cross_validate(columns, X, y, **kwargs)
    np.testing.assert_allclose(a.oof_predictions, b.oof_predictions)
    for score in ("nll", "crps", "actual_expected"):
        assert a.mean_scores[score] == pytest.approx(b.mean_scores[score])


def test_dirichlet_component_offsets_survive_family_order_and_response_labels():
    rng = np.random.default_rng(18)
    X, _ = _data(48)
    y = pd.DataFrame(rng.dirichlet([2.0, 3.0, 4.0], len(X)), columns=["a", "b", "c"])
    family = DirichletLSS(components=("a", "b", "c"))
    model = SuperLSS(
        family,
        family.alpha("c"),
        family.alpha("a", offset=["a", "b"]),
        family.alpha("b", offset=X.a.to_numpy()),
    ).fit(X, y[["c", "a", "b"]])
    control = SuperLSS(
        family,
        family.alpha("a", offset=(X.a + X.b).to_numpy()),
        family.alpha("b", offset="a"),
        family.alpha("c"),
    ).fit(X, y)
    a = model.predict_parameters(X, offsets={"alpha_1": X.a}).to_numpy()
    b = control.predict_parameters(X, offsets={"alpha_0": X.a + X.b}).to_numpy()
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a / a.sum(axis=1, keepdims=True), b / b.sum(axis=1, keepdims=True))
    permuted_family = DirichletLSS(components=("c", "a", "b"))
    permuted = SuperLSS(
        permuted_family,
        permuted_family.alpha("b", offset="a"),
        permuted_family.alpha("a", offset=["a", "b"]),
        permuted_family.alpha("c"),
    ).fit(X, y)
    by_component = permuted.predict_parameters(X).to_numpy()[:, [1, 2, 0]]
    np.testing.assert_allclose(by_component, a, rtol=1e-6)


def test_legacy_templates_without_declaration_keep_prediction_mapping():
    X, y = _data()
    family = GaussianLS()
    model = SuperLSS(family, family.location("x"), family.scale()).fit(X, y)
    for template in (*model._predictors, *model._require_fitted().fit_state.predictor_templates):
        template.__dict__.pop("offset", None)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(
        restored.predict_link(X, offsets={"location": X.a})["location"],
        restored.predict_link(X)["location"] + X.a,
    )


def test_scoring_residuals_and_posterior_resolve_offsets_once():
    from superglm.distributional.posterior import posterior_parameters

    X, y = _data()
    columns = _model(["a", "b"]).fit(X, y)
    arrays = _model((X.a + X.b).to_numpy()).fit(X, y)
    weights = np.ones(len(X))
    weights[5] = 0
    new = X.copy()
    new.loc[5, "a"] = np.nan
    offsets = {"location": (new.a + new.b).to_numpy()}
    np.testing.assert_allclose(
        columns.scores(new, y, sample_weight=weights)[["log", "crps"]],
        arrays.scores(new, y, sample_weight=weights, offsets=offsets)[["log", "crps"]],
    )
    for model, replacement in [(columns, None), (arrays, offsets)]:
        fitted = model._require_fitted()
        draws = model.posterior_draws(8, seed=12)
        clean = X.iloc[[1, 3, 8]]
        rep = None if replacement is None else {"location": (clean.a + clean.b).to_numpy()}
        # Fixed offsets shift predictions, and have no uncertainty of their own.
        link, se = model.predict_link(clean, offsets=rep, se_fit=True)
        assert np.all(np.isfinite(se))
        chunks = list(posterior_parameters(fitted, clean, draws, offsets=rep, chunk_rows=2))
        assert sum(block.shape[1] for _, block in chunks) == 3
        np.testing.assert_allclose(link, columns.predict_link(clean))
    np.testing.assert_allclose(
        columns.predict_link(X, se_fit=True)[1],
        arrays.predict_link(X, offsets={"location": X.a + X.b}, se_fit=True)[1],
    )


@pytest.mark.parametrize("invalid", [True, 1j, "1.0", np.inf, np.nan])
def test_retained_sources_refuse_nonnumeric_and_nonfinite_values(invalid):
    X, y = _data()
    values = X.a.to_list()
    values[2] = invalid
    # Nested explicitly: a list of strings alone means column sources.
    with pytest.raises(ValueError, match="offset.*original.*2"):
        _model([values]).fit(X, y)
    weights = np.ones(len(X))
    weights[2] = 0
    _model([values]).fit(X, y, sample_weight=weights)


def test_declared_offsets_apply_to_retained_diagnostics_and_hand_edits():
    X, y = _data()
    model = _model(["a", "b"]).fit(X, y)
    control = _model((X.a + X.b).to_numpy()).fit(X, y)
    weights = np.ones(len(X))
    weights[2] = 0
    arrays = {"location": X.a + X.b}
    np.testing.assert_allclose(
        model.residuals(X, y, sample_weight=weights),
        control.residuals(X, y, sample_weight=weights, offsets=arrays),
    )
    a = model.actual_expected(X, y, "x", sample_weight=weights)
    b = control.actual_expected(X, y, "x", sample_weight=weights, offsets=arrays)
    np.testing.assert_allclose(a.expected, b.expected)
    coef = np.array(list(model.coef_.values()))
    coef[1] += 0.2
    edited = model._with_revised_coefficients(coef, edited_terms=("location:x",))
    np.testing.assert_allclose(
        edited.predict_link(X)["location"], model.predict_link(X)["location"] + 0.2 * X.x
    )


def test_prediction_replacements_validate_original_labels_and_retained_raw_values():
    X, y = _data()
    model = _model(X.a.to_numpy(), scale=None).fit(X, y)
    values = X.a.to_list()
    values[2] = "dropped"
    weights = np.ones(len(X))
    weights[2] = 0
    result = model.scores(X, y, which=("log",), sample_weight=weights, offsets={"location": values})
    assert np.isfinite(result["log"].iloc[weights > 0]).all()
    actual_expected = model.actual_expected(
        X, y, "x", sample_weight=weights, offsets={"location": values}
    )
    assert np.isfinite(actual_expected.expected).all()
    with pytest.raises(ValueError, match="index"):
        model.scores(
            X,
            y,
            which=("log",),
            sample_weight=weights,
            offsets={"location": pd.Series(X.a.to_numpy(), index=X.index[::-1])},
        )
    with pytest.raises(ValueError, match="index"):
        model.residuals(
            X,
            y,
            sample_weight=weights,
            offsets={"location": pd.Series(X.a.to_numpy(), index=X.index[::-1])},
        )
    with pytest.raises(ValueError, match="offset"):
        model.predict_parameters(X, offsets={"location": "a"})


def test_column_declarations_do_not_block_editor_refit_without_replacement_data():
    from superglm.editor.lss.availability import refit_reason

    X, y = _data()
    model = _model(["a", "b"]).fit(X, y)
    assert refit_reason(model, None, None, y) is None


def _editor_offset_dataset(model, X, y, values, weights=None):
    from superglm.editor.evaluation import coerce_dataset

    return coerce_dataset(
        "validation",
        (X, y, weights, {"location": values}),
        family=model.family,
        weight_semantics="prior",
        offset_parameters=model.parameter_names_,
    )


def test_editor_replacements_check_original_series_index_before_normalizing():
    from superglm.editor.lss.validation import lss_dataset_metrics

    X, y = _data()
    X.index = np.arange(100, 100 + len(X))
    model = _model(X.a.to_numpy(), scale=None).fit(X, y)
    values = pd.Series(X.a.to_numpy(), index=X.index)
    dataset = _editor_offset_dataset(model, X, y, values)
    metrics = lss_dataset_metrics(model, dataset)
    direct = model.scores(X, y, which=("log", "crps"), offsets={"location": values})
    assert metrics["log_score"] == pytest.approx(direct["log"].mean())
    assert metrics["crps"] == pytest.approx(direct["crps"].mean())
    with pytest.raises(ValueError, match="index.*original"):
        _editor_offset_dataset(model, X, y, values.iloc[::-1])


@pytest.mark.parametrize("consumer", ["metrics", "export", "refit"])
def test_editor_offsets_preserve_raw_values_until_zero_weight_selection(consumer):
    from superglm.editor.lss.persistence import lss_prediction_rows
    from superglm.editor.lss.validation import lss_dataset_metrics, stack_offsets

    X, y = _data()
    X.index = np.arange(100, 100 + len(X))
    model = _model(X.a.to_numpy(), scale=None).fit(X, y)
    weights = np.ones(len(X))
    weights[2] = 0
    values = pd.Series(X.a.to_numpy(), index=X.index, dtype=object)
    values.iloc[2] = "dropped"
    dataset = _editor_offset_dataset(model, X, y, values, weights)
    positions = np.flatnonzero(weights)
    if consumer == "metrics":
        got = lss_dataset_metrics(model, dataset)
        expected = model.scores(
            X, y, which=("log",), sample_weight=weights, offsets={"location": values}
        )
        assert got["log_score"] == pytest.approx(expected["log"].sum() / len(positions))
    elif consumer == "export":
        got = lss_prediction_rows(model, dataset, max_rows=len(X))
        expected = model.predict_parameters(
            X.iloc[positions], offsets={"location": X.a.iloc[positions]}
        )
        np.testing.assert_allclose(got, expected)
    else:
        stacked = stack_offsets([dataset, dataset])
        refit = model._clone_with_replaced_terms({})._replay_fit(
            pd.concat([X, X], ignore_index=True),
            np.tile(y, 2),
            np.tile(weights, 2),
            stacked,
        )
        control = _model(X.a.to_numpy(), scale=None).fit(X, y, sample_weight=weights)
        np.testing.assert_allclose(list(refit.coef_.values()), list(control.coef_.values()))


@pytest.mark.parametrize("source", ["array", "column"])
def test_taken_diagnostic_offsets_report_invalid_original_position(source):
    X, y = _data()
    model = _model(X.a.to_numpy() if source == "array" else "a", scale=None).fit(X, y)
    weights = np.ones(len(X))
    weights[2] = 0
    X.loc[6, "a"] = np.nan
    offsets = {"location": X.a} if source == "array" else None
    with pytest.raises(ValueError, match=r"original rows \[6\]"):
        model.actual_expected(X, y, "x", sample_weight=weights, offsets=offsets)


def test_editor_final_fit_keeps_dropped_raw_offsets_out_of_value_checks():
    from types import SimpleNamespace

    from superglm import GammaLS
    from superglm.editor import edit
    from superglm.editor.cv import capture_final_fit, run_final_fit

    X, y = _data()
    X.index = np.arange(100, 100 + len(X))
    y = np.abs(y) + 0.5
    family = GammaLS()

    def build():
        return SuperLSS(family, family.mean("x", offset=X.a.to_numpy()), family.scale())

    weights = np.ones(len(X))
    weights[2] = 0
    model = build().fit(X, y, sample_weight=weights)
    values = pd.Series(X.a.to_numpy(), index=X.index, dtype=object)
    values.iloc[2] = "dropped"
    split = (X, y, weights, {"mean": values})
    session = edit(model, train_data=split, validation_data=split)
    context = SimpleNamespace(check=lambda: None, progress=lambda *args, **kwargs: None)
    final = run_final_fit(capture_final_fit(session), context)
    control = build().fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(list(final.model.coef_.values()), list(control.coef_.values()))


def test_editor_missing_split_offsets_compare_retained_values_only():
    import dataclasses

    from superglm.editor.cv import _unstackable_reason

    X, y = _data()
    model = _model(np.zeros(len(X)), scale=None).fit(X, y)
    weights = np.ones(len(X))
    weights[2] = 0
    values = np.zeros(len(X), dtype=object)
    values[2] = "dropped"
    dataset = _editor_offset_dataset(model, X, y, values, weights)
    missing = dataclasses.replace(dataset, name="test", offset=None)
    assert _unstackable_reason([dataset, missing]) is None


@pytest.mark.parametrize("kind", ["spread", "portfolio", "qq"])
def test_retained_surfaces_and_qq_preserve_raw_replacement_values_and_labels(kind):
    X, y = _data()
    model = _model(X.a.to_numpy(), scale=None).fit(X, y)
    values = X.a.to_list()
    values[2] = "dropped"
    weights = np.ones(len(X))
    weights[2] = 0
    kwargs = {
        "spread": {"threshold": 2.0, "n_bins": 3},
        "portfolio": {"n_draws": 8, "seed": 4},
        "qq": {"y": y, "n_sim": 4, "seed": 4},
    }[kind]
    full = model.plot_data(kind, X=X, sample_weight=weights, offsets={"location": values}, **kwargs)
    if kind == "qq":
        kwargs["y"] = y[weights > 0]
    retained = model.plot_data(
        kind, X=X.loc[weights > 0], offsets={"location": X.a.loc[weights > 0]}, **kwargs
    )
    assert full == retained
    if kind != "qq":
        with pytest.raises(ValueError, match="index"):
            model.plot_data(
                kind,
                X=X,
                sample_weight=weights,
                offsets={"location": pd.Series(X.a.to_numpy(), index=X.index[::-1])},
                **kwargs,
            )


@pytest.mark.parametrize("rows", [9, 17000])
def test_prediction_sizes_do_not_reuse_training_arrays_or_run_an_optimizer(rows, monkeypatch):
    import superglm.distributional.model as implementation

    X, y = _data()
    model = _model(["a", "b"]).fit(X, y, retain_rows=False)

    def refuse(*args, **kwargs):
        raise AssertionError("prediction called the optimizer")

    monkeypatch.setattr(implementation, "fit_dense_fixed_lambda", refuse)
    new = pd.DataFrame(
        {
            "x": np.linspace(-1, 1, rows),
            "a": np.full(rows, 0.3),
            "b": np.full(rows, -0.2),
            "scale_offset": np.zeros(rows),
        }
    )
    eta = model.predict_link(new)
    np.testing.assert_allclose(
        eta["location"],
        model.coef_["location:(intercept)"] + model.coef_["location:x"] * new.x + 0.1,
    )


def test_prediction_surfaces_keep_offset_only_columns_in_the_reference():
    X, y = _data()
    model = _model(["a", "b"]).fit(X, y)
    curves = model.risk_curves(
        {"a": 0.7, "b": -0.2, "scale_offset": 0.0}, "x", n_points=7, n_draws=8, quantiles=(0.5,)
    )
    np.testing.assert_allclose(
        curves.values[0],
        model.coef_["location:(intercept)"] + model.coef_["location:x"] * curves.x + 0.5,
    )


def test_intercept_only_offset_has_one_fixed_contribution_and_no_edf():
    X, y = _data()
    family = GaussianLS()
    model = SuperLSS(family, family.location(offset=["a", "b"]), family.scale()).fit(X, y)
    expected = np.mean(y - X.a - X.b) + X.a + X.b
    np.testing.assert_allclose(model.predict(X), expected)
    assert len(model.coef_) == 2
    assert model.result_.total_effective_df == pytest.approx(2.0)


def test_every_parameter_helper_carries_both_source_list_forms():
    from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
    from superglm.distributional.families.negative_binomial import NegativeBinomialLS
    from superglm.distributional.families.tweedie import TweedieLSS
    from superglm.distributional.families.two_piece import TwoPieceNormalLSS

    a, b = np.array([0.1, 0.2]), np.array([0.3, -0.2])
    for family, helpers in (
        (GaussianLS(), ("location", "scale")),
        (NegativeBinomialLS(), ("mean", "theta")),
        (GeneralizedGammaLSS(parametrisation="location"), ("location", "scale", "shape")),
        (TwoPieceNormalLSS(), ("location", "scale", "skew")),
        (TweedieLSS(), ("mu", "phi", "p")),
    ):
        for helper in helpers:
            for source in (["a", "b"], [a, b]):
                template = getattr(family, helper)(offset=source).template
                assert not template.features
                assert len(template.offset) == 2
                for original, kept in zip(source, template.offset, strict=True):
                    np.testing.assert_array_equal(original, kept)
