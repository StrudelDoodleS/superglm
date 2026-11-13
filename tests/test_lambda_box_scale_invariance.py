"""The smoothing-parameter box moves with the data's own scale (``_lambda_box``).

Multiplying every prior weight by ``c`` multiplies the penalized
log-likelihood, the Fisher weights and the optimal smoothing parameters by
``c`` in every family whose dispersion is estimated (the scale absorbs ``c``),
so the REML optimum, its predictions and its EDF are invariant (Wood, Pya &
Saefken 2016, sec. 3.1). In a SuperLSS Gaussian location, rescaling the
response by ``a`` multiplies the location's information, and its optimal
smoothing parameters, by ``a**-2``. A box fixed in absolute units held those
optima back: at ``c = 2**30`` a SuperGLM term reached the ceiling, at
``c = 2**-20`` the floor, and at ``a = 2**-13`` the SuperLSS location terms
reached the cap. Powers of two make the rescaled data exact.

Tolerance. Both fits stop on Wood's compound test, the objective change and
the projected gradient each below ``tol * (1 + |V|)`` (``reml_tol``: 1e-9 on
SuperGLM, 1e-6 on SuperLSS). Two such stops on the same optimum lie
``O(sqrt(tol))`` apart along a direction of unit curvature, which bounds the
relative movement of a prediction or the EDF: ``sqrt(tol)``. A box that bites
moves them by 1e-3 to 1e-1 here.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import superglm._lambda_box as lambda_box
from superglm import Categorical, GammaLS, GaussianLS, PSpline, SuperGLM, SuperLSS, cat, s
from superglm.distributions import Tweedie
from superglm.model.reml_execute import NEWTON_REML_TOL_DEFAULT

_GLM_TOL = math.sqrt(NEWTON_REML_TOL_DEFAULT)
_LSS_TOL = math.sqrt(1.0e-6)


def _rows(n: int = 2500, seed: int = 20261113):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x": rng.uniform(0.0, 1.0, n),
            "z": rng.uniform(0.0, 1.0, n),
            "g": rng.choice(list("abcde"), n),
        }
    )
    eta = 0.6 + 0.5 * np.sin(4.0 * np.pi * X["x"].to_numpy()) + 0.2 * X["z"].to_numpy()
    return X, eta, rng.uniform(0.5, 1.5, n), rng


def _glm_response(family: str, eta, w, rng):
    mu = np.exp(eta)
    if family == "gaussian":
        return mu + rng.normal(0.0, 0.05, len(mu)) / np.sqrt(w)
    if family == "gamma":
        shape = 20.0 * w
        return rng.gamma(shape, mu / shape)
    counts = rng.poisson(mu * w)
    return np.array([rng.gamma(2.0, 0.5, k).sum() for k in counts]) / w


def _glm_fit(family: str, X, y, w):
    model = SuperGLM(
        family=Tweedie(p=1.5) if family == "tweedie" else family,
        link=None if family == "gaussian" else "log",
        features={"x": PSpline(n_knots=12), "z": PSpline(n_knots=8), "g": Categorical()},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y, sample_weight=w)
    return np.asarray(model.predict(X), dtype=np.float64), model


def _assert_same(reference, rescaled, tol):
    np.testing.assert_allclose(rescaled, reference, rtol=tol, atol=0.0)


@pytest.mark.parametrize("family", ["gaussian", "gamma", "tweedie"])
@pytest.mark.parametrize("exponent", [-20, 30])
def test_superglm_fit_is_invariant_to_rescaled_prior_weights(family, exponent):
    X, eta, w, rng = _rows()
    y = _glm_response(family, eta, w, rng)
    reference, base = _glm_fit(family, X, y, w)
    rescaled, scaled = _glm_fit(family, X, y, w * 2.0**exponent)
    _assert_same(reference, rescaled, _GLM_TOL)
    edf = np.array([base._group_edf[name] for name in base._group_edf])
    scaled_edf = np.array([scaled._group_edf[name] for name in base._group_edf])
    np.testing.assert_allclose(scaled_edf, edf, rtol=0.0, atol=_GLM_TOL * (1.0 + edf.sum()))
    # The box moved with the weights by exactly 2**exponent.
    assert scaled._reml_result.lambda_box.unit == base._reml_result.lambda_box.unit * 2.0**exponent


def _lss_gaussian(X, y, w):
    family = GaussianLS(scale_floor=0.0)
    model = SuperLSS(
        family,
        family.location(s("x", k=12), s("z", k=8), cat("g")),
        family.scale(s("z", k=6)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y, sample_weight=w)
    return model


def _lss_gamma(X, y, w):
    family = GammaLS()
    model = SuperLSS(family, family.mean(s("x", k=12), s("z", k=8), cat("g")), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y, sample_weight=w)
    return model


def _location(model, X, a):
    return np.asarray(model.predict_parameters(X).iloc[:, 0], dtype=np.float64) / a


@pytest.mark.parametrize(("weight_exponent", "response_exponent"), [(-20, 0), (30, 0), (0, -13)])
def test_superlss_gaussian_location_is_invariant_to_rescaled_weights_and_response(
    weight_exponent, response_exponent
):
    X, eta, w, rng = _rows()
    y = eta + rng.normal(0.0, 0.3, len(eta)) / np.sqrt(w)
    c, a = 2.0**weight_exponent, 2.0**response_exponent
    reference = _location(_lss_gaussian(X, y, w), X, 1.0)
    rescaled = _lss_gaussian(X, y * a, w * c)
    assert rescaled.smoothing_convergence_reason_ != "lambda_cap_unresolved"
    _assert_same(reference, _location(rescaled, X, a), _LSS_TOL)


@pytest.mark.parametrize("exponent", [-20, 30])
def test_superlss_gamma_mean_is_invariant_to_rescaled_weights(exponent):
    X, eta, w, rng = _rows()
    shape = 5.0 * w
    y = rng.gamma(shape, np.exp(eta) / shape)
    reference = _location(_lss_gamma(X, y, w), X, 1.0)
    rescaled = _location(_lss_gamma(X, y, w * 2.0**exponent), X, 1.0)
    _assert_same(reference, rescaled, _LSS_TOL)


def test_an_unweighted_gaussian_fit_keeps_the_unit_box():
    X, eta, _, rng = _rows()
    _, model = _glm_fit("gaussian", X, eta + rng.normal(0.0, 0.05, len(eta)), None)
    box = model._reml_result.lambda_box
    assert (box.floor, box.ceiling) == (lambda_box.LAMBDA_FLOOR, lambda_box.LAMBDA_CEILING)


@pytest.mark.parametrize(
    ("bound", "value", "side", "word"),
    [
        ("LAMBDA_CEILING", 1.0e-4, "ceiling", "largest"),
        ("LAMBDA_FLOOR", 1.0e8, "floor", "smallest"),
    ],
)
def test_a_superglm_term_held_at_a_bound_is_named_with_its_limit(
    monkeypatch, bound, value, side, word
):
    X, eta, _, rng = _rows()
    y = np.exp(eta) + rng.normal(0.0, 0.05, len(eta))
    # Force the event: a box whose one side cuts through every term's optimum.
    if side == "floor":
        monkeypatch.setattr(lambda_box, "LAMBDA_CEILING", 1.0e12)
    monkeypatch.setattr(lambda_box, bound, value)
    model = SuperGLM(family="gaussian", features={"x": PSpline(n_knots=12)})
    with pytest.warns(UserWarning, match=f"'x' at the {word} smoothing parameter") as record:
        model.fit_reml(X, y)
    assert model._reml_result.bound_pressure == (("x", side),)
    message = next(str(item.message) for item in record if "'x' at the" in str(item.message))
    assert ("fewer basis functions" in message) == (side == "floor")
    assert ("unpenalized part" in message) == (side == "ceiling")


def test_a_superglm_term_inside_the_box_gives_no_bound_warning():
    X, eta, _, rng = _rows()
    model = SuperGLM(family="gaussian", features={"x": PSpline(n_knots=12)})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.fit_reml(X, np.exp(eta) + rng.normal(0.0, 0.05, len(eta)))
    assert model._reml_result.bound_pressure == ()
    assert not any("smoothing parameter the search allows" in str(r.message) for r in record)


def test_a_superlss_term_held_at_its_cap_is_named():
    X, eta, w, rng = _rows()
    y = eta + rng.normal(0.0, 0.3, len(eta))
    family = GaussianLS(scale_floor=0.0)
    model = SuperLSS(family, family.location(s("x", k=12)), family.scale())
    with pytest.warns(UserWarning, match="'location:x#wiggle' at the largest smoothing parameter"):
        model.fit_reml(X, y, max_lambda=1.0e-3, initial_lambda=1.0e-4, outer="efs")


def test_a_term_reaching_the_floor_through_an_accepted_line_search_step_is_named():
    # A response with almost no noise asks for less smoothing than the floor
    # allows. The search reaches the floor through an accepted line-search
    # trial, whose carried-forward evaluation is no strict improvement on it;
    # the bound pressure must still be read there.
    x = np.linspace(0.0, 1.0, 500)
    y = 2.0 + np.sin(4.0 * np.pi * x) + 1.0e-4 * np.random.default_rng(12).normal(size=500)
    model = SuperGLM(family="gaussian", features={"x": PSpline(n_knots=30)})
    with pytest.warns(UserWarning, match="'x' at the smallest smoothing parameter"):
        model.fit_reml(pd.DataFrame({"x": x}), y)
    assert model._reml_result.lambdas["x"] == pytest.approx(lambda_box.LAMBDA_FLOOR)
    assert model._reml_result.bound_pressure == (("x", "floor"),)


def test_an_explicit_max_lambda_is_described_as_the_cap_set_for_the_fit():
    X, eta, _, rng = _rows()
    y = eta + rng.normal(0.0, 0.3, len(eta))
    family = GaussianLS(scale_floor=0.0)
    model = SuperLSS(family, family.location(s("x", k=12)), family.scale())
    with pytest.warns(UserWarning, match="'location:x#wiggle' at the largest") as record:
        model.fit_reml(X, y, max_lambda=1.0e-3, initial_lambda=1.0e-4, outer="efs")
    message = next(str(r.message) for r in record if "'location:x#wiggle' at the" in str(r.message))
    assert "the max_lambda cap set for this fit, 0.001" in message
    assert "information per row" not in message
    assert "unpenalized part" not in message


def _small_gaussian_lss(y, **kwargs):
    family = GaussianLS(scale_floor=0.0)
    model = SuperLSS(family, family.location(s("x", k=8)), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(pd.DataFrame({"x": np.linspace(0.0, 1.0, len(y))}), y, **kwargs)
    return model._model.smoothing


@pytest.mark.parametrize(("response_scale", "start"), [(32.0, 1.0e-8), (2.0**-14, 1.0e12)])
def test_a_scalar_initial_lambda_meets_the_relative_box_like_a_named_one(response_scale, start):
    # Each start lies inside the location's relative box but outside the
    # absolute [1e-6, 1e10]: below it on a response in large units, above it
    # on one in small units.
    x = np.linspace(0.0, 1.0, 160)
    y = response_scale * (np.sin(6.0 * x) + 0.3 * np.random.default_rng(8).normal(size=160))
    name = "location:x#wiggle"
    scalar = _small_gaussian_lss(y, initial_lambda=start)
    named = _small_gaussian_lss(y, lambdas={name: start})
    lower, upper = scalar.config.lower(name), scalar.config.upper(name)
    assert lower < start < upper
    assert scalar.initial_lambdas[name] == named.initial_lambdas[name] == start
    assert scalar.lambdas[name] == named.lambdas[name]


def test_a_finite_start_far_above_the_box_is_clipped_not_refused():
    # 1e308 times a penalty overflows. The start must reach the box's clip before
    # any penalty is formed from it, as it did when the box was absolute.
    x = np.linspace(0.0, 1.0, 160)
    y = np.sin(6.0 * x) + 0.3 * np.random.default_rng(8).normal(size=160)
    name = "location:x#wiggle"
    smoothing = _small_gaussian_lss(y, initial_lambda=1.0e308)
    assert smoothing.initial_lambdas[name] == smoothing.config.upper(name)
    assert np.isfinite(smoothing.lambdas[name])


def test_a_predictor_at_its_link_floor_keeps_the_absolute_ceiling():
    # A generalized gamma whose scale offset holds it on its link's floor
    # (``scale_floor + exp(eta)``, eta near -35): its rows carry about 2**-38
    # of information each, and the REML criterion is flat in its smoothing
    # parameter. A box of 1e10 times that unit put the ceiling at 0.036, far
    # below the 7.2e4 the absolute box reached, and the search stopped there
    # unconverged ('endpoint_revalidation_failed'); the post-fit repairs
    # started from that fit, whose scale penalty no longer dominated.
    from superglm import Constraint
    from superglm.distributional.families import GeneralizedGammaLSS

    x = np.linspace(0.0, 1.0, 200)
    X = pd.DataFrame({"x": x, "off": -35.0 + 30.0 * x})
    y = 1.0 + 0.005 * np.random.default_rng(0).normal(size=200)
    family = GeneralizedGammaLSS()
    model = SuperLSS(
        family,
        family.mean(s("x", kind="bs", k=6)),
        family.scale(
            s("x", kind="bs", k=6, constraint=Constraint.postfit.increasing), offset="off"
        ),
        family.shape(s("x", kind="bs", k=6)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, y)
    smoothing = model._model.smoothing
    name = "scale:x#wiggle"
    assert smoothing.config.unit(name) < 2.0**-30
    assert smoothing.config.upper(name) == lambda_box.LAMBDA_CEILING
    assert smoothing.converged
    assert smoothing.lambdas[name] > 1.0e3
