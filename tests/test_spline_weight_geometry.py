"""Family-aware sample-weight contracts for learned spline geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Piecewise, PSpline, Spline, SuperGLM, Tweedie

_X = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
_WEIGHT = np.array([1, 4, 2, 6, 1, 3, 5, 1, 4, 2, 3, 1], dtype=np.float64)


@pytest.mark.parametrize("strategy", ["quantile_rows", "quantile_tempered"])
@pytest.mark.parametrize("kind", ["ps", "cr_cardinal"])
def test_integer_frequency_knots_match_literal_row_replication(strategy, kind):
    """Compact frequency mass reproduces the existing expanded-row algorithms."""
    repeated = np.repeat(np.arange(len(_X)), _WEIGHT.astype(np.intp))

    weighted = Spline(kind=kind, n_knots=4, knot_strategy=strategy)
    replicated = Spline(kind=kind, n_knots=4, knot_strategy=strategy)
    weighted.build(_X, sample_weight=_WEIGHT)
    replicated.build(_X[repeated])

    assert weighted.fitted_boundary == replicated.fitted_boundary
    np.testing.assert_allclose(weighted.fitted_knots, replicated.fitted_knots, rtol=0.0)


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_positive_frequency_magnitude_does_not_move_control_strategies(strategy):
    """Uniform and unique-value quantiles are replication-invariant controls."""
    repeated = np.repeat(np.arange(len(_X)), _WEIGHT.astype(np.intp))

    weighted = Spline(kind="ps", n_knots=4, knot_strategy=strategy)
    replicated = Spline(kind="ps", n_knots=4, knot_strategy=strategy)
    weighted.build(_X, sample_weight=_WEIGHT)
    replicated.build(_X[repeated])

    assert weighted.fitted_boundary == replicated.fitted_boundary
    np.testing.assert_allclose(weighted.fitted_knots, replicated.fitted_knots, rtol=0.0)


@pytest.mark.parametrize(
    "strategy",
    ["uniform", "quantile", "quantile_rows", "quantile_tempered"],
)
def test_zero_frequency_row_is_absent_from_all_main_spline_geometry(strategy):
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 100.0])
    weight = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    active = weight > 0.0

    weighted = Spline(kind="ps", n_knots=2, knot_strategy=strategy)
    omitted = Spline(kind="ps", n_knots=2, knot_strategy=strategy)
    weighted.build(x, sample_weight=weight)
    omitted.build(x[active])

    assert weighted.fitted_boundary == (-1.0, 1.0)
    assert weighted.fitted_boundary == omitted.fitted_boundary
    np.testing.assert_allclose(weighted.fitted_knots, omitted.fitted_knots, rtol=0.0)


def _gaussian_model(strategy: str, *, discrete: bool) -> SuperGLM:
    return SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.6,
        discrete=discrete,
        n_bins=5,
        features={"x": PSpline(n_knots=4, knot_strategy=strategy)},
    )


@pytest.mark.parametrize("strategy", ["quantile_rows", "quantile_tempered"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("fit_method", ["fit", "fit_reml"])
def test_frequency_geometry_and_fit_match_replication(strategy, discrete, fit_method):
    """The family contract reaches both fit entry points and spline build paths."""
    y = 1.0 + 0.3 * _X + 0.05 * _X**2
    repeated = np.repeat(np.arange(len(_X)), _WEIGHT.astype(np.intp))
    frame = pd.DataFrame({"x": _X})
    repeated_frame = frame.iloc[repeated].reset_index(drop=True)

    weighted = _gaussian_model(strategy, discrete=discrete)
    replicated = _gaussian_model(strategy, discrete=discrete)
    if fit_method == "fit_reml":
        weighted.fit_reml(frame, y, sample_weight=_WEIGHT, max_reml_iter=2)
        replicated.fit_reml(repeated_frame, y[repeated], max_reml_iter=2)
    else:
        weighted.fit(frame, y, sample_weight=_WEIGHT)
        replicated.fit(repeated_frame, y[repeated])

    weighted_knots = weighted.knot_summary()["x"]
    replicated_knots = replicated.knot_summary()["x"]
    assert weighted_knots["boundary"] == replicated_knots["boundary"]
    np.testing.assert_allclose(
        weighted_knots["interior_knots"],
        replicated_knots["interior_knots"],
        rtol=0.0,
    )
    np.testing.assert_allclose(
        weighted.predict(frame),
        replicated.predict(frame),
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_zero_frequency_uniform_outlier_matches_literal_omission(discrete):
    """Regression: x=100, w=0 must not widen the active [-1, 1] spline."""
    x = np.array([-1.0, 0.0, 1.0, 100.0])
    y = np.array([0.0, 1.0, 0.0, 50.0])
    weight = np.array([1.0, 1.0, 1.0, 0.0])
    active = weight > 0.0
    frame = pd.DataFrame({"x": x})
    active_frame = frame.loc[active].reset_index(drop=True)

    def model() -> SuperGLM:
        return SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            spline_penalty=0.8,
            discrete=discrete,
            n_bins=2,
            features={"x": PSpline(n_knots=2, knot_strategy="uniform")},
        )

    weighted = model().fit(frame, y, sample_weight=weight)
    omitted = model().fit(active_frame, y[active])

    assert weighted.knot_summary()["x"]["boundary"] == (-1.0, 1.0)
    np.testing.assert_allclose(
        weighted.predict(active_frame),
        omitted.predict(active_frame),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("strategy", ["quantile_rows", "quantile_tempered"])
def test_tweedie_prior_weights_keep_physical_row_knot_geometry(strategy):
    """EDM prior weights change likelihood precision, not row geometry."""
    frame = pd.DataFrame({"x": _X})
    y = np.exp(0.1 + 0.05 * _X)

    def model() -> SuperGLM:
        return SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            spline_penalty=0.6,
            features={"x": PSpline(n_knots=4, knot_strategy=strategy)},
        )

    weighted = model().fit(frame, y, sample_weight=_WEIGHT)
    physical_rows = model().fit(frame, y, sample_weight=np.ones_like(_WEIGHT))

    weighted_knots = weighted.knot_summary()["x"]
    physical_knots = physical_rows.knot_summary()["x"]
    assert weighted_knots["boundary"] == physical_knots["boundary"]
    np.testing.assert_allclose(
        weighted_knots["interior_knots"],
        physical_knots["interior_knots"],
        rtol=0.0,
    )


def test_tweedie_prior_weights_do_not_move_piecewise_placement_or_base():
    """Piecewise int-mode placement and base selection are model geometry.

    Under Tweedie the weights are EDM prior weights, not frequency mass, so
    knot placement and ``base='most_exposed'`` follow physical rows -- the
    same rule the spline strategies above follow (explicit-breaks mode has no
    learned placement to move).  The Poisson control proves the weights would
    otherwise have moved the knots, so the equality is not vacuous.
    """
    rng = np.random.default_rng(7)
    x = np.round(rng.uniform(0.0, 40.0, 400), 0)
    weight = np.where(x > 30.0, 9.0, 0.5)
    y = rng.poisson(2.0, 400).astype(np.float64)
    frame = pd.DataFrame({"x": x})

    def fitted_spec(family, sample_weight):
        model = SuperGLM(family=family, features={"x": Piecewise(3, base="most_exposed")})
        model.fit(frame, y, sample_weight=sample_weight)
        return model._specs["x"]

    tweedie_weighted = fitted_spec(Tweedie(p=1.5), weight)
    tweedie_unweighted = fitted_spec(Tweedie(p=1.5), None)
    np.testing.assert_array_equal(tweedie_weighted._knots, tweedie_unweighted._knots)
    assert tweedie_weighted._base_index == tweedie_unweighted._base_index

    poisson_weighted = fitted_spec("poisson", weight)
    assert not np.array_equal(poisson_weighted._knots, tweedie_unweighted._knots)


def test_tweedie_prior_weights_do_not_move_hosted_piecewise_geometry():
    """The physical-rows rule reaches an OrderedCategorical's inner Piecewise.

    Hosted int-mode placement and ``base='most_exposed'`` are the same MODEL
    geometry as the numeric term's, so under Tweedie they must match the
    no-prior-weights result. Observed before the fix: prior weights heaped on
    the upper bands pulled the placement to [0, 5, 6, 7] with base index 3,
    against the physical-rows [0, 2, 4, 5, 7] with base index 1 -- identical
    to the Poisson weighted fit, which is the control proving the equality
    below is not vacuous. Polynomial standardization deliberately keeps
    following ``sample_weight`` under every family (the inference-geometry
    rule), so only the Piecewise inner is pinned here.
    """
    from superglm import OrderedCategorical

    levels = [f"Mi{i:03d}" for i in range(8)]
    rng = np.random.default_rng(7)
    bands = rng.choice(levels, 400)
    positions = np.array([levels.index(band) for band in bands], dtype=np.float64)
    weight = np.where(positions > 5.0, 9.0, 0.5)
    y = rng.poisson(2.0, 400).astype(np.float64)
    frame = pd.DataFrame({"band": bands})

    def fitted_inner(family, sample_weight):
        model = SuperGLM(
            family=family,
            features={
                "band": OrderedCategorical(
                    order=levels, basis=Piecewise(3, base="most_exposed")
                )
            },
        )
        model.fit(frame, y, sample_weight=sample_weight)
        return model._specs["band"]._spline

    tweedie_weighted = fitted_inner(Tweedie(p=1.5), weight)
    tweedie_unweighted = fitted_inner(Tweedie(p=1.5), None)
    np.testing.assert_array_equal(tweedie_weighted._knots, tweedie_unweighted._knots)
    assert tweedie_weighted._base_index == tweedie_unweighted._base_index

    poisson_weighted = fitted_inner("poisson", weight)
    assert not np.array_equal(poisson_weighted._knots, tweedie_unweighted._knots)
