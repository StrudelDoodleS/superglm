from __future__ import annotations

import contextlib
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm import SuperLSS
from superglm.diagnostics.separation import SeparationError, SeparationWarning
from superglm.distributional import GammaLS, GaussianLS, NegativeBinomialLS, Predictor, TweedieLSS
from superglm.distributional.curvature import RepeatedCurvatureIndefinitenessError
from superglm.distributional.family import ResponseBoundaryFamily
from superglm.distributional.fit_diagnostics import diagnose_distributional_fit
from superglm.features import Categorical, Spline
from superglm.links import IdentityLink, LogLink


def test_tweedie_and_nb2_declare_their_zero_boundary() -> None:
    tweedie = TweedieLSS()
    assert isinstance(tweedie, ResponseBoundaryFamily)
    links = tuple(p.default_link for p in tweedie.parameters)
    assert tweedie.response_boundaries(links) == (("zero",), ("zero",), ())
    nb2 = NegativeBinomialLS()
    assert isinstance(nb2, ResponseBoundaryFamily)
    assert nb2.response_boundaries((LogLink(), LogLink())) == (("zero",), ("zero",))


def test_a_finite_boundary_link_disables_the_scan() -> None:
    nb2 = NegativeBinomialLS()
    assert nb2.response_boundaries((IdentityLink(), LogLink())) == ((), ("zero",))


def test_gaussian_and_gamma_do_not_declare_boundaries() -> None:
    assert not isinstance(GaussianLS(), ResponseBoundaryFamily)
    assert not isinstance(GammaLS(), ResponseBoundaryFamily)


def _separated_tweedie_fixture(n: int = 600, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    region = rng.choice(np.array(["a", "b", "c", "d"]), size=n)
    mu = np.exp(0.6 + 0.3 * x)
    lam = mu**0.5 / (0.8 * 0.5)
    counts = rng.poisson(lam)
    y = np.where(counts > 0, rng.gamma(np.maximum(counts, 1) * 1.0, 0.8 * 0.5 * mu**0.5), 0.0)
    exposure = rng.uniform(0.2, 1.0, n)
    # level "d" keeps its exposure but loses every claim: a separated cell
    y = np.where(region == "d", 0.0, y)
    frame = pd.DataFrame({"x": x, "region": region})
    return frame, y, exposure


def _tweedie_model(**kwargs):
    return SuperLSS(
        family=TweedieLSS(),
        predictors=(
            Predictor("mean", {"x": Spline(kind="cr", k=6), "region": Categorical()}),
            Predictor("dispersion", {"region": Categorical()}),
            Predictor("power", {}),
        ),
        **kwargs,
    )


# With the all-zero level on both the mean and the dispersion predictor its
# rows share one direction in (log mu, log phi): an exact null vector of the
# unpenalised curvature, which the coefficient solver's rank policy refuses
# as materially indefinite.  The scan runs before that solver, and the
# refusal is not what these tests assert.
_SOLVER_REFUSAL = RepeatedCurvatureIndefinitenessError


def test_separated_level_warns_before_fitting_and_names_both_predictors() -> None:
    frame, y, exposure = _separated_tweedie_fixture()
    with pytest.warns(SeparationWarning) as record, contextlib.suppress(_SOLVER_REFUSAL):
        _tweedie_model().fit(frame, y, sample_weight=exposure, lambdas={"mean:x#wiggle": 1.0})
    message = str(record[0].message)
    assert "'mean:region'" in message and "'dispersion:region'" in message
    assert "'d'" in message
    assert "collapse" in message


def test_separation_error_refuses_the_design() -> None:
    frame, y, exposure = _separated_tweedie_fixture()
    with pytest.raises(SeparationError, match="Separation detected"):
        _tweedie_model(separation="error").fit(
            frame, y, sample_weight=exposure, lambdas={"mean:x#wiggle": 1.0}
        )


def test_separation_ignore_is_silent() -> None:
    frame, y, exposure = _separated_tweedie_fixture()
    with warnings.catch_warnings(), contextlib.suppress(_SOLVER_REFUSAL):
        warnings.simplefilter("error", SeparationWarning)
        _tweedie_model(separation="ignore").fit(
            frame, y, sample_weight=exposure, lambdas={"mean:x#wiggle": 1.0}
        )


def test_a_level_with_a_claim_does_not_warn() -> None:
    frame, y, exposure = _separated_tweedie_fixture()
    y = y.copy()
    y[np.flatnonzero(frame["region"].to_numpy() == "d")[:3]] = 5.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        _tweedie_model().fit(frame, y, sample_weight=exposure, lambdas={"mean:x#wiggle": 1.0})


def test_families_without_a_boundary_are_never_scanned() -> None:
    frame, y, exposure = _separated_tweedie_fixture()
    y = np.abs(y) + 0.1
    model = SuperLSS(
        family=GammaLS(),
        predictors=(
            Predictor("mean", {"x": Spline(kind="cr", k=6), "region": Categorical()}),
            Predictor("scale", {}),
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        model.fit(frame, y, sample_weight=exposure, lambdas={"mean:x#wiggle": 1.0})


def test_invalid_policy_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="separation"):
        _tweedie_model(separation="maybe")


def test_roundoff_negative_curvature_does_not_infer_separation_through_fit_read_path() -> None:
    """A telemetry sign beneath numerical resolution is not separation evidence."""
    frame, y, exposure = _separated_tweedie_fixture()
    y = y.copy()
    y[np.flatnonzero(frame["region"].to_numpy() == "d")[:5]] = 4.0
    model = _tweedie_model()
    model.fit(frame, y, sample_weight=exposure, lambdas={"mean:x#wiggle": 1.0})
    fitted = model._require_fitted()
    telemetry = fitted.fit_state.solver_result.terminal_curvature
    negative = replace(telemetry, minimum_eigenvalue=-np.finfo(np.float64).eps)
    solver_result = replace(fitted.fit_state.solver_result, terminal_curvature=negative)
    fit_state = replace(fitted.fit_state, solver_result=solver_result)
    injected = replace(fitted, _fit_state=fit_state)
    report = diagnose_distributional_fit(injected)
    assert not [f for f in report.findings if f.code == "fit.curvature_indefinite"]
    clean = diagnose_distributional_fit(fitted)
    assert not [f for f in clean.findings if f.code == "fit.curvature_indefinite"]


def test_a_clean_fit_has_no_curvature_finding() -> None:
    frame, y, exposure = _separated_tweedie_fixture()
    y = y.copy()
    y[np.flatnonzero(frame["region"].to_numpy() == "d")[:5]] = 4.0
    model = _tweedie_model()
    model.fit(frame, y, sample_weight=exposure, lambdas={"mean:x#wiggle": 1.0})
    report = diagnose_distributional_fit(model._require_fitted())
    assert not [f for f in report.findings if f.code == "fit.curvature_indefinite"]
