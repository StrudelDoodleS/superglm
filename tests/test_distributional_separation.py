from __future__ import annotations

import pickle
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm.diagnostics.separation import SeparationError, SeparationWarning
from superglm.distributional import GammaLS, GaussianLS, NegativeBinomialLS, Predictor, TweedieLSS
from superglm.distributional.family import ResponseBoundaryFamily
from superglm.distributional.fit_diagnostics import diagnose_distributional_fit
from superglm.features import Categorical, Spline
from superglm.links import IdentityLink, LogLink
from tests.bound_predictor_fixtures import model_from_templates


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
    return model_from_templates(
        family=TweedieLSS(),
        predictors=(
            Predictor("mean", {"x": Spline(kind="cr", k=6), "region": Categorical()}),
            Predictor("dispersion", {"region": Categorical()}),
            Predictor("power", {}),
        ),
        **kwargs,
    )


# With the all-zero level on both the mean and the dispersion predictor its
# rows share one direction in (log mu, log phi), along which the observed
# curvature stays materially indefinite.  TweedieLSS has no expected
# information, so the solver used to refuse the fit here; it now accepts the
# nearest semidefinite curvature and discloses it.


def test_separated_level_warns_before_fitting_and_names_both_predictors() -> None:
    frame, y, exposure = _separated_tweedie_fixture()
    with pytest.warns(SeparationWarning) as record:
        _tweedie_model().fit(frame, y, sample_weight=exposure, lambdas={"mean:x#wiggle": 1.0})
    message = str(record[0].message)
    assert "'mean:region'" in message and "'dispersion:region'" in message
    assert "'d'" in message
    assert "collapse" in message


def test_a_fitted_model_keeps_no_per_row_level_codes_once_the_scan_has_read_them() -> None:
    """The compiled predictors outlive the scan, and a pickled model carries them.

    The scan pops each term's build codes, and a compiled design never pickles
    them, so a saved model holds neither the integers nor a class for them.
    Mutation: a scan that only reads them leaves an integer per row per term on
    the model.
    """
    from superglm.distributional.serialization import pickle_state

    frame, y, exposure = _separated_tweedie_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeparationWarning)
        model = _tweedie_model().fit(
            frame, y, sample_weight=exposure, lambdas={"mean:x#wiggle": 1.0}
        )
    designs = [predictor.compiled for predictor in model._model.compiled_predictors]
    assert [design.scan_codes for design in designs] == [{}, {}, {}]
    designs[0].scan_codes["region"] = np.zeros(len(y), dtype=np.intp)
    assert pickle.loads(pickle_state(designs[0])).scan_codes == {}
    assert b"BuildCodes" not in pickle.dumps(model)


def test_the_codes_held_until_the_scan_take_a_byte_a_row() -> None:
    """Every scanned term's codes are held across the other builds, so they are stored narrow."""
    from superglm._frame import as_eager_frame
    from superglm.distributional.predictor import compile_predictors
    from superglm.distributional.weights import WeightContract, resolve_likelihood_weights

    frame, y, exposure = _separated_tweedie_fixture()
    family = TweedieLSS()
    templates = (
        Predictor("mean", {"region": Categorical()}),
        Predictor("dispersion", {}),
        Predictor("power", {}),
    )
    weights = resolve_likelihood_weights(
        exposure, n_observations=len(y), contract=WeightContract(semantics="prior")
    )
    (mean, *_) = compile_predictors(
        as_eager_frame(frame),
        weights,
        family.parameters,
        templates,
        separation_boundaries=(("zero",), ("zero",), ()),
        separation_response=y,
    )
    assert {k: v.dtype for k, v in mean.compiled.scan_codes.items()} == {"region": np.uint8}


def test_separated_tweedie_fit_finishes_on_the_nearest_semidefinite_curvature() -> None:
    """Kills the refusal for a family without expected information (master raised here)."""
    frame, y, exposure = _separated_tweedie_fixture()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _tweedie_model().fit(
            frame, y, sample_weight=exposure, lambdas={"mean:x#wiggle": 1.0}
        )
    telemetry = model._require_fitted().fit_state.solver_result.terminal_curvature
    assert telemetry.actual_source == "nearest_semidefinite"
    assert telemetry.fallback_count == 1
    assert len(model.result_.warnings) == 1
    assert "nearest positive semidefinite" in model.result_.warnings[0]
    assert any(str(item.message) == model.result_.warnings[0] for item in caught)
    assert np.all(np.isfinite(model.covariance_))
    restored = pickle.loads(pickle.dumps(model))
    assert restored.result_.warnings == model.result_.warnings
    assert restored.result_.curvature_telemetry == model.result_.curvature_telemetry


def test_separation_error_refuses_the_design() -> None:
    frame, y, exposure = _separated_tweedie_fixture()
    with pytest.raises(SeparationError, match="Separation detected"):
        _tweedie_model(separation="error").fit(
            frame, y, sample_weight=exposure, lambdas={"mean:x#wiggle": 1.0}
        )


def test_separation_ignore_is_silent() -> None:
    frame, y, exposure = _separated_tweedie_fixture()
    with warnings.catch_warnings():
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
    model = model_from_templates(
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
