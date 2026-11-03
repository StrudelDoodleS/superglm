"""Reference rows and level sweeps honour every term that reads a column with levels (#572)."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from superglm import Categorical, LambdaPolicy, Numeric, OrderedCategorical, Polynomial
from superglm._frame import as_eager_frame
from superglm.distributional import Predictor, terms
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.surfaces import density_fan, risk_curves
from superglm.features.factor_smooth import FactorSmooth
from superglm.features.grouping import collapse_levels
from tests.bound_predictor_fixtures import model_from_templates


def _fit(predictors, frame, response, sample_weight=None):
    model = model_from_templates(family=GaussianLS(), predictors=predictors)
    model.fit_reml(frame, response, sample_weight=sample_weight, max_reml_iter=20)
    assert model.result_.converged
    return model, model._require_fitted()


def _curve(fitted, frame, covariate="x", reference=None):
    return risk_curves(
        fitted,
        frame,
        {} if reference is None else reference,
        covariate,
        quantiles=(0.5,),
        n_points=5,
        n_draws=32,
        seed=4,
    )


def _fan(fitted, frame, covariate="x", reference=None):
    return density_fan(
        fitted,
        frame,
        {} if reference is None else reference,
        covariate,
        quantiles=(0.5,),
        n_points=5,
        n_y=40,
    )


def _assert_profile(fitted, curves, fan, manual):
    """Independent oracle: predict a manually specified policy, no ref helpers."""
    theta = fitted.predict_parameters(manual)
    np.testing.assert_allclose(curves.values[0], theta[:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fan.quantiles[0], theta[:, 0], rtol=1e-12, atol=1e-12)
    expected_cdf = ndtr((fan.y_grid[None, :] - theta[:, 0, None]) / theta[:, 1, None])
    expected_density = np.gradient(expected_cdf, fan.y_grid, axis=1)
    np.testing.assert_allclose(fan.density, expected_density, rtol=1e-10, atol=1e-12)


def _factor_fit(levels, unseen="error", main_scale=None):
    count = 60
    frame = pd.DataFrame(
        {
            "g": np.repeat(levels, count),
            "x": np.tile(np.linspace(-1, 1, count), len(levels)),
        }
    )
    code = np.repeat(np.arange(len(levels)), count)
    response = (
        2 * code
        + (0.4 + 0.3 * code) * frame.x.to_numpy()
        + np.random.default_rng(572).normal(0, 0.3, len(frame))
    )
    interaction = FactorSmooth(
        "x", group="g", levels=levels, k=5, unseen=unseen, lambda_policy=LambdaPolicy.fixed(1)
    )
    model, fitted = _fit(
        [
            Predictor("location", {"x": Numeric()}, interaction_specs={"x:g:fs": interaction}),
            Predictor("scale", {} if main_scale is None else {"g": main_scale}),
        ],
        frame,
        response,
    )
    return model, fitted, frame


@pytest.mark.parametrize("size,unseen", [(4, "error"), (4, "population"), (5, "population")])
def test_factor_integer_group_uses_fitted_level_for_all_reference_builders(size, unseen):
    model, fitted, frame = _factor_fit(list(range(1, size + 1)), unseen)
    beta = fitted.coefficients.copy()
    curves = _curve(fitted, frame)
    fan = _fan(fitted, frame)
    assert curves.reference == fan.reference == {"g": 1}
    manual = pd.DataFrame({"x": np.linspace(-1, 1, 5), "g": [1] * 5})
    _assert_profile(fitted, curves, fan, manual)
    effect = terms.term_effect(fitted, frame, "location", "x", n_points=5, simultaneous=False)
    public = model.term_inference("location", "x", n_points=5, simultaneous=False)
    block = fitted.layout.term_slices["location:x"]
    np.testing.assert_allclose(effect.effect, manual.x.to_numpy() * beta[block][0])
    np.testing.assert_allclose(public.effect, effect.effect)
    np.testing.assert_array_equal(fitted.coefficients, beta)


def test_factor_group_sweeps_fitted_order_and_keeps_numeric_marginal_median():
    order = ["d", "c", "b", "a"]
    _, fitted, frame = _factor_fit(order)
    assert _curve(fitted, frame).reference == {"g": "d"}
    curves = _curve(fitted, frame, "g")
    fan = _fan(fitted, frame, "g")
    assert curves.levels == fan.levels == tuple(order)
    # Zero is an actual numeric median between two x observations.
    assert 0.0 not in frame.x.to_numpy()
    assert curves.reference == fan.reference == {"x": pytest.approx(0.0)}
    manual = pd.DataFrame({"x": np.zeros(4), "g": order})
    _assert_profile(fitted, curves, fan, manual)


def test_integer_factor_group_sweeps_no_interpolated_ids():
    _, fitted, frame = _factor_fit([4, 3, 2, 1])
    curves, fan = _curve(fitted, frame, "g"), _fan(fitted, frame, "g")
    assert curves.levels == fan.levels == ("4", "3", "2", "1")
    manual = pd.DataFrame({"x": np.zeros(4), "g": [4, 3, 2, 1]})
    _assert_profile(fitted, curves, fan, manual)


def test_large_integer_group_reference_survives_surface_frame_construction():
    order = [2**53 + step for step in (3, 4, 1, 2)]
    _, fitted, frame = _factor_fit(order)
    beta = fitted.coefficients.copy()
    curves, fan = _curve(fitted, frame), _fan(fitted, frame)
    assert curves.reference == fan.reference == {"g": order[0]}
    manual = pd.DataFrame({"x": np.linspace(-1, 1, 5), "g": [order[0]] * 5})
    _assert_profile(fitted, curves, fan, manual)
    np.testing.assert_array_equal(fitted.coefficients, beta)


@pytest.mark.parametrize("unseen", ["error", "population"])
def test_ordered_group_sweep_preserves_jointly_readable_numeric_labels(unseen):
    order = [4.0, 3.0, 2.0, 1.0]
    model, fitted, frame = _factor_fit(
        order,
        unseen,
        main_scale=OrderedCategorical(order=["4", "3", "2", "1"], basis=Polynomial(powers=[1])),
    )
    beta = fitted.coefficients.copy()
    curves, fan = _curve(fitted, frame, "g"), _fan(fitted, frame, "g")
    assert curves.levels == fan.levels == ("4", "3", "2", "1")
    manual = pd.DataFrame({"x": np.zeros(4), "g": order})
    _assert_profile(fitted, curves, fan, manual)
    effect = terms.term_effect(fitted, frame, "scale", "g", simultaneous=False)
    public = model.term_inference("scale", "g", simultaneous=False)
    assert np.all(np.isfinite(effect.effect))
    np.testing.assert_array_equal(public.effect, effect.effect)
    np.testing.assert_array_equal(fitted.coefficients, beta)


def test_main_term_order_keeps_precedence_over_an_earlier_predictors_factor_group():
    _, fitted, frame = _factor_fit(
        ["d", "c", "b", "a"], main_scale=Categorical(levels=["b", "a", "d", "c"])
    )
    assert _curve(fitted, frame).reference == {"g": "b"}
    assert _curve(fitted, frame, "g").levels == ("b", "a", "d", "c")


def _grouped_fit(*, zero="d", seen="c", routed=False, secondary="categorical"):
    group = np.repeat(["a", "b", seen, zero], [200, 200, 150, 300])
    frame = pd.DataFrame({"g": group, "x": np.random.default_rng(4).uniform(-1, 1, len(group))})
    weights = (group != zero).astype(float)
    response = (
        0.8 * frame.x.to_numpy()
        + np.select([group == "a", group == "b"], [0.4, -0.4], 1.2)
        + np.random.default_rng(5).normal(0, 0.3, len(group))
    )
    basis = frame.loc[weights > 0, "g"] if routed else frame.g
    grouping = collapse_levels(basis, groups={"joined": [seen] if routed else [zero, seen]})
    first = Categorical(grouping=grouping, unseen="joined" if routed else "error")
    if secondary == "factor":
        scale = Predictor(
            "scale",
            {},
            interaction_specs={
                "x:g:fs": FactorSmooth(
                    "x", group="g", k=5, unseen="error", lambda_policy=LambdaPolicy.fixed(1)
                ),
            },
        )
    else:
        scale = Predictor("scale", {"g": Categorical()})
    model, fitted = _fit(
        [
            Predictor("location", {"x": Numeric(), "g": first}),
            scale,
        ],
        frame,
        response,
        weights,
    )
    return model, fitted, frame, seen


@pytest.mark.parametrize("secondary", ["categorical", "factor"])
@pytest.mark.parametrize("routed", [False, True])
def test_group_reference_member_must_be_fitted_for_every_reader(secondary, routed):
    model, fitted, frame, seen = _grouped_fit(secondary=secondary, routed=routed)
    curves, fan = _curve(fitted, frame), _fan(fitted, frame)
    # Preserve the grouped vote: joined holds 450, then choose c, not d.
    assert curves.reference == fan.reference == {"g": seen}
    manual = pd.DataFrame({"x": np.linspace(frame.x.min(), frame.x.max(), 5), "g": [seen] * 5})
    _assert_profile(fitted, curves, fan, manual)
    terms.term_effect(fitted, frame, "location", "x", n_points=5, simultaneous=False)
    public = model.risk_curves({}, "x", X_train=frame, quantiles=(0.5,), n_points=5, n_draws=32)
    assert public.reference == {"g": seen}
    public_effect = model.term_inference(
        "location", "x", X_train=frame, n_points=5, simultaneous=False
    )
    assert np.all(np.isfinite(public_effect.effect))


@pytest.mark.parametrize("zero,seen,routed", [("d", "z", False), ("0-unseen", "c", True)])
def test_group_sweep_skips_a_sorted_first_label_another_parameter_did_not_fit(zero, seen, routed):
    _, fitted, frame, expected = _grouped_fit(zero=zero, seen=seen, routed=routed)
    curves, fan = _curve(fitted, frame, "g"), _fan(fitted, frame, "g")
    effect = terms.term_effect(fitted, frame, "location", "g", simultaneous=False)
    assert curves.levels == fan.levels == effect.levels == ("a", "b", "joined")
    manual = pd.DataFrame({"x": [np.median(frame.x)] * 3, "g": ["a", "b", expected]})
    _assert_profile(fitted, curves, fan, manual)
    # The term's grouped contrast also agrees with the manually predicted location.
    location = fitted.predict_parameters(manual)[:, 0]
    np.testing.assert_allclose(effect.effect - effect.effect[0], location - location[0], atol=1e-12)


def test_group_member_fallback_and_explicit_postfit_frame_filtering():
    _, fitted, frame, seen = _grouped_fit(zero="d", seen="z")
    # A post-fit frame lacks every label of joined. Its first declared member d
    # is invalid to scale, but z is a compatible member fitted by both terms.
    subset = frame.loc[frame.g == "a"].copy()
    curves, fan = _curve(fitted, subset, "g"), _fan(fitted, subset, "g")
    manual = pd.DataFrame({"x": [np.median(subset.x)] * 3, "g": ["a", "b", seen]})
    _assert_profile(fitted, curves, fan, manual)
    # With the whole caller-supplied reference frame, joined has all 450 votes.
    assert _curve(fitted, frame).reference == {"g": seen}
    assert _curve(fitted, subset).reference == {"g": "a"}


def test_all_unseen_reference_retains_native_prediction_failure():
    _, fitted, frame, _ = _grouped_fit()
    unseen = frame.iloc[:4].assign(g="outside-every-domain")
    # The fix must not invent a training label when the supplied frame has none.
    assert terms._reference_value(fitted, as_eager_frame(unseen), "g") == "outside-every-domain"
    with pytest.raises(ValueError, match="unseen categorical"):
        _curve(fitted, unseen)
