from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest

from superglm import SuperLSS
from superglm.distributional import GammaLS, GaussianLS, Predictor
from superglm.distributional import TweedieLSS as _TweedieLSS
from superglm.distributional import api as api_module
from superglm.distributional import model as model_module
from superglm.distributional.families import TweedieLSS as _FamiliesTweedieLSS
from superglm.distributional.families.negative_binomial import (
    NegativeBinomialLS as _NegativeBinomialLS,
)
from superglm.distributional.family import DistributionalFamily, ObservationContract
from superglm.distributional.fit_diagnostics import diagnose_distributional_fit
from superglm.distributional.model import fit_dense_distributional, refit_dense_distributional
from superglm.distributional.result import DistributionalFitResult
from superglm.distributional.serialization import (
    SCHEMA_VERSION,
    DistributionalSerializationError,
)
from superglm.distributional.timing import FitPhaseRecorder
from superglm.distributional.weights import (
    LegacyPowerWeightArtifactError,
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.features import Numeric, RandomEffect, Spline
from superglm.types import LambdaPolicy


def _fixture(
    n: int = 88,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(20260723)
    x = np.linspace(-1.0, 1.0, n)
    z = np.cos(np.linspace(0.0, 2.0 * np.pi, n))
    sigma = 0.18 + np.exp(-1.45 + 0.25 * z)
    response = 0.65 + 0.7 * x + rng.normal(scale=sigma)
    frame = pd.DataFrame(
        {"x": x, "z": z},
        index=pd.Index(np.arange(n) + 100, name="row"),
    )
    weights = np.linspace(0.7, 1.4, n)
    offsets = {
        "location": np.linspace(-0.08, 0.11, n),
        "scale": 0.03 * np.sin(np.linspace(0.0, 2.0 * np.pi, n)),
    }
    return frame, response, weights, offsets


def _tweedie_fixture(
    n: int = 36,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a compact nonnegative sample containing structural zeros."""
    axis = np.linspace(-1.0, 1.0, n)
    frame = pd.DataFrame({"x_mean": np.roll(axis, 5)})
    response = np.exp(0.35 + 0.30 * frame["x_mean"]) * (1.0 + 0.18 * np.sin(1.7 * np.arange(n)))
    response.iloc[::11] = 0.0
    return frame, response.to_numpy()


def _negative_binomial_fixture(
    n: int = 240,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Generate counts from NB2(exposure * mean, exposure * theta)."""
    rng = np.random.default_rng(2026083115)
    x_mean = rng.permutation(np.linspace(-1.0, 1.0, n))
    x_theta = rng.permutation(np.linspace(-1.0, 1.0, n))
    exposure = np.resize(np.array([0.5, 1.0, 1.5, 2.0]), n)
    mean_offset = 0.08 * np.sin(np.pi * x_mean)
    theta_offset = -0.06 * np.cos(np.pi * x_theta)
    mean = np.exp(0.55 + 0.35 * x_mean + mean_offset)
    theta = np.exp(0.20 - 0.25 * x_theta + theta_offset)
    probability = theta / (mean + theta)
    count = rng.negative_binomial(exposure * theta, probability).astype(np.float64)
    return (
        pd.DataFrame({"x_mean": x_mean, "x_theta": x_theta}),
        count / exposure,
        exposure,
        {"mean": mean_offset, "theta": theta_offset},
    )


def _certified_automatic_tweedie_fixture() -> tuple[pd.DataFrame, np.ndarray]:
    """Generate the certified CPG outcome fixture independently of fitting."""
    rng = np.random.default_rng(2026082804)
    n = 720
    axis = np.linspace(-1.0, 1.0, n)
    x_mean = rng.permutation(axis)
    x_dispersion = rng.permutation(axis)
    x_power = rng.permutation(axis)
    mean = np.exp(0.55 + 0.45 * np.sin(np.pi * x_mean) + 0.15 * x_mean)
    dispersion = np.exp(-0.05 + 0.28 * np.cos(np.pi * x_dispersion) - 0.12 * x_dispersion)
    power = 1.5 + 0.16 * np.sin(np.pi * x_power) + 0.04 * x_power
    r = power - 1.0
    s = 2.0 - power
    rate = mean**s / (dispersion * s)
    jump_shape = s / r
    jump_scale = dispersion * r * mean**r
    counts = rng.poisson(rate)
    response = np.zeros(n, dtype=np.float64)
    positive = counts > 0
    response[positive] = rng.gamma(
        jump_shape[positive] * counts[positive],
        jump_scale[positive],
    )
    return (
        pd.DataFrame(
            {
                "x_mean": x_mean,
                "x_dispersion": x_dispersion,
                "x_power": x_power,
            }
        ),
        response,
    )


def _mean_smooth_tweedie_predictors() -> tuple[Predictor, ...]:
    return (
        Predictor(
            "mean",
            {
                "x_mean": Spline(
                    kind="cr",
                    n_knots=4,
                    knot_strategy="quantile_rows",
                    lambda_policy=LambdaPolicy.estimate(),
                )
            },
        ),
        Predictor("dispersion", {}),
        Predictor("power", {}),
    )


def _all_smooth_tweedie_predictors() -> tuple[Predictor, ...]:
    return tuple(
        Predictor(
            name,
            {
                f"x_{name}": Spline(
                    kind="cr",
                    n_knots=5,
                    knot_strategy="quantile_rows",
                    lambda_policy=LambdaPolicy.estimate(),
                )
            },
        )
        for name in ("mean", "dispersion", "power")
    )


def _intercept_only_tweedie_predictors() -> tuple[Predictor, ...]:
    return tuple(Predictor(name, {}) for name in ("mean", "dispersion", "power"))


def _roundoff_factor(*values: np.ndarray) -> float:
    operation_count = 128 * sum(np.asarray(value).size for value in values) + 1
    product = operation_count * np.finfo(np.float64).eps
    assert product < 1.0
    return float(np.nextafter(product / (1.0 - product), np.inf))


def _roundoff_envelope(*values: np.ndarray) -> float:
    scale = max(
        1.0,
        *(float(np.linalg.norm(np.asarray(value), ord=np.inf)) for value in values),
    )
    return _roundoff_factor(*values) * scale


def _tweedie_model(
    *,
    automatic: bool = False,
    frequency: bool = False,
) -> SuperLSS:
    kwargs = {"weight_semantics": "frequency"} if frequency else {}
    return SuperLSS(
        family=_TweedieLSS(power_lower=1.08, power_upper=1.92),
        predictors=(
            _intercept_only_tweedie_predictors()
            if not automatic
            else _mean_smooth_tweedie_predictors()
        ),
        **kwargs,
    )


def _assert_observed_coefficient_fits(model: SuperLSS) -> None:
    fitted = model._require_fitted()
    smoothing = fitted.smoothing
    coefficient_fits = (fitted.result,) if smoothing is None else smoothing.coefficient_fits
    assert all(
        fit.config.coefficient_curvature == "observed"
        and fit.terminal_curvature.requested_source == "observed"
        and fit.terminal_curvature.actual_source == "observed"
        and fit.terminal_curvature.fallback_count == 0
        for fit in coefficient_fits
    )
    null_curvature = fitted.null_model.curvature_telemetry
    assert (
        null_curvature.requested_source == "observed"
        and null_curvature.actual_source == "observed"
        and null_curvature.fallback_count == 0
    )


def _assert_public_tweedie_round_trip(model: SuperLSS, frame: pd.DataFrame) -> None:
    restored = SuperLSS.from_bytes(model.to_bytes())
    source = model._require_fitted()
    target = restored._require_fitted()

    assert type(restored.family_) is _TweedieLSS
    assert restored.family_.to_config() == model.family_.to_config()
    assert restored.weight_semantics == model.weight_semantics
    assert restored.training_telemetry() == model.training_telemetry()
    assert restored.coef_.keys() == model.coef_.keys()
    np.testing.assert_array_equal(tuple(restored.coef_.values()), tuple(model.coef_.values()))
    np.testing.assert_array_equal(restored.covariance_, model.covariance_)
    assert restored.smoothing_parameters_ == model.smoothing_parameters_
    assert target.smoothing is not None and source.smoothing is not None
    assert target.smoothing.config == source.smoothing.config
    assert target.null_model.link_types == source.null_model.link_types
    np.testing.assert_array_equal(target.null_model.coefficients, source.null_model.coefficients)
    assert target.null_model.objective == source.null_model.objective
    for predict in (SuperLSS.predict_link, SuperLSS.predict_parameters, SuperLSS.predict):
        np.testing.assert_array_equal(predict(restored, frame), predict(model, frame))
    source_rows, target_rows = source.fit_state.retained_rows, target.fit_state.retained_rows
    assert (target_rows is not None) is (source_rows is not None)
    if source_rows is not None and target_rows is not None:
        np.testing.assert_array_equal(target_rows.response, source_rows.response)
        np.testing.assert_array_equal(
            target_rows.likelihood_weights.values,
            source_rows.likelihood_weights.values,
        )


def _linear_model() -> SuperLSS:
    return SuperLSS(
        family=GaussianLS(scale_floor=0.02),
        predictors=(
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {"z": Numeric()}),
        ),
    )


class _FailIfCompiledNumeric(Numeric):
    def build(self, *args, **kwargs):
        raise AssertionError("predictor geometry was compiled before likelihood-plan refusal")


class _SubstitutingLikelihoodFamily:
    def __init__(self, *, scale_floor: float) -> None:
        self.base = GaussianLS(scale_floor=scale_floor)

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def default_prediction_name(self):
        return self.base.default_prediction_name

    def to_config(self):
        return self.base.to_config()

    def bind_likelihood(
        self,
        y: np.ndarray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ):
        substitute = resolve_likelihood_weights(
            None,
            n_observations=len(y),
            contract=WeightContract("frequency"),
        )
        return self.base.bind_likelihood(y, substitute, observation)

    def initialize(self, y, plan):
        return self.base.initialize(y, plan)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        return self.base.evaluate_natural(
            y,
            theta,
            plan,
            derivative_order=derivative_order,
        )

    def expected_information_natural(self, theta, plan):
        return self.base.expected_information_natural(theta, plan)

    def default_prediction(self, theta):
        return self.base.default_prediction(theta)


class _GammaLSSubclass(GammaLS):
    pass


class _GammaLookalike:
    base = GammaLS()
    parameters = base.parameters
    default_prediction_name = base.default_prediction_name
    to_config = base.to_config

    def bind_likelihood(self, y, weights, observation):
        return self.base.bind_likelihood(y, weights, observation)

    def initialize(self, y, plan):
        return self.base.initialize(y, plan)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        return self.base.evaluate_natural(
            y,
            theta,
            plan,
            derivative_order=derivative_order,
        )

    def default_prediction(self, theta):
        return self.base.default_prediction(theta)


class _TweedieLSSSubclass(_TweedieLSS):
    pass


class _TweedieLookalike:
    _base = _TweedieLSS()
    parameters = _base.parameters
    default_prediction_name = _base.default_prediction_name
    to_config = _base.to_config
    bind_likelihood = _base.bind_likelihood
    initialize = _base.initialize
    evaluate_natural = _base.evaluate_natural
    default_prediction = _base.default_prediction


class _NegativeBinomialLSSubclass(_NegativeBinomialLS):
    pass


class _NegativeBinomialLookalike:
    _base = _NegativeBinomialLS()
    parameters = _base.parameters
    default_prediction_name = _base.default_prediction_name
    to_config = _base.to_config
    bind_likelihood = _base.bind_likelihood
    initialize = _base.initialize
    evaluate_natural = _base.evaluate_natural
    default_prediction = _base.default_prediction


class _UnconfiguredGammaLookalike:
    base = GammaLS()
    parameters = base.parameters

    def bind_likelihood(self, y, weights, observation):
        raise AssertionError("configuration refusal must precede likelihood binding")

    def initialize(self, y, plan):
        raise AssertionError("configuration refusal must precede initialization")

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        raise AssertionError("configuration refusal must precede family evaluation")


def _resign_public_config(artifact: dict) -> None:
    config = artifact["public_api"]["config"]
    canonical = json.dumps(
        config,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(artifact["payload"]["sha256"].lower().encode("ascii"))
    digest.update(b"\n")
    digest.update(canonical)
    artifact["public_api"]["sha256"] = digest.hexdigest()


def test_public_imports_and_fixed_fit_publish_named_immutable_views() -> None:
    frame, response, weights, offsets = _fixture()
    model = _linear_model()

    returned = model.fit(
        frame,
        response,
        sample_weight=weights,
        offsets=offsets,
    )

    assert returned is model
    assert isinstance(model.family_, GaussianLS)
    assert tuple(predictor.name for predictor in model.predictors_) == (
        "location",
        "scale",
    )
    assert model.parameter_names_ == ("location", "scale")
    assert isinstance(model.coef_, Mapping)
    assert tuple(model.coef_) == model.result_.coefficient_names
    assert tuple(model.coef_by_predictor_) == model.parameter_names_
    assert isinstance(model.result_, DistributionalFitResult)
    assert model.result_.smoothing_converged is None
    assert model.result_.coefficient_converged is True
    assert model.smoothing_convergence_reason_ is None
    assert model.smoothing_certified_ is None
    assert model.smoothing_unresolved_upper_bound_ is None
    assert model._require_fitted().result.config.coefficient_curvature == "observed"
    assert model.smoothing_parameters_ == {}
    assert not model.covariance_.flags.writeable
    assert all(not values.flags.writeable for values in model.coef_by_predictor_.values())
    with pytest.raises(TypeError):
        model.coef_["location:(intercept)"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError):
        model.covariance_[0, 0] = 0.0

    telemetry = model.training_telemetry()
    assert telemetry.model_type == "SuperLSS"
    assert telemetry.family == "GaussianLS"
    assert telemetry.parameter_names == ("location", "scale")
    assert telemetry.n_observations == len(frame)
    assert telemetry.predictor_dimensions == {"location": 2, "scale": 2}
    assert telemetry.total_dimension == 4
    assert telemetry.curvature_channels == 3
    assert telemetry.discrete is False
    assert telemetry.inner_iterations == model.result_.n_inner_iter
    assert telemetry.smoothing_iterations == 0
    assert telemetry.converged is True
    assert telemetry.smoothing_convergence_reason is None
    assert telemetry.smoothing_certified is None
    assert telemetry.smoothing_unresolved_upper_bound is None
    assert telemetry.curvature is model.result_.curvature_telemetry


def test_public_superlss_refuses_unfinished_discrete_fitting() -> None:
    with pytest.raises(
        NotImplementedError,
        match="Discrete SuperLSS fitting is not implemented",
    ):
        SuperLSS(
            family=GaussianLS(),
            predictors=(Predictor("location", {}), Predictor("scale", {})),
            discrete=True,
        )


def test_public_reml_defaults_to_practical_convergence_with_a_strict_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []

    def capture_fit(self, *args, **kwargs):
        captured.append(kwargs["efs_config"])
        return self

    monkeypatch.setattr(SuperLSS, "_fit", capture_fit)
    frame, response, _weights, _offsets = _fixture(12)
    model = SuperLSS(
        family=GaussianLS(),
        predictors=(Predictor("location", {}), Predictor("scale", {})),
    )

    model.fit_reml(frame, response)
    model.fit_reml(
        frame,
        response,
        practical_reml=False,
        practical_reml_parameter_tol=2.5e-4,
        reml_plateau_tol=7.5e-6,
        max_lambda=123.0,
    )

    assert captured[0].practical_convergence is True
    assert captured[0].practical_parameter_tolerance == 1.0e-3
    assert captured[1].practical_convergence is False
    assert captured[1].practical_parameter_tolerance == 2.5e-4
    assert captured[1].plateau_tolerance == 7.5e-6
    assert captured[1].maximum_lambda == 123.0


def test_public_reml_places_its_implicit_start_inside_a_small_lambda_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []

    def capture_fit(self, *args, **kwargs):
        captured.append(kwargs["efs_config"])
        return self

    monkeypatch.setattr(SuperLSS, "_fit", capture_fit)
    frame, response, _weights, _offsets = _fixture(12)
    model = SuperLSS(
        family=GaussianLS(),
        predictors=(Predictor("location", {}), Predictor("scale", {})),
    )

    model.fit_reml(frame, response, max_lambda=0.05)

    assert captured[0].initial_lambda == 0.05
    assert captured[0].maximum_lambda == 0.05


@pytest.mark.parametrize(
    ("family", "predictors", "names"),
    [
        (GammaLS(), (Predictor("mean", {}), Predictor("scale", {})), ("mean", "scale")),
        (_TweedieLSS(), _intercept_only_tweedie_predictors(), ("mean", "dispersion", "power")),
        (
            _NegativeBinomialLS(),
            (Predictor("mean", {}), Predictor("theta", {})),
            ("mean", "theta"),
        ),
    ],
)
def test_constructor_admits_exact_public_distributional_families(
    family: object,
    predictors: tuple[Predictor, ...],
    names: tuple[str, ...],
) -> None:
    model = SuperLSS(
        family=family,  # type: ignore[arg-type]
        predictors=predictors,
    )

    assert type(model.family) is type(family)
    assert tuple(parameter.name for parameter in model.family.parameters) == names
    if type(family) is _TweedieLSS:
        assert _FamiliesTweedieLSS is _TweedieLSS


def test_public_tweedie_fixed_fit_requests_and_uses_observed_curvature() -> None:
    frame, response = _tweedie_fixture()
    assert 0 < np.count_nonzero(response == 0.0) < len(response)
    model = _tweedie_model().fit(frame, response)
    fitted = model._require_fitted()

    _assert_observed_coefficient_fits(model)
    assert fitted.result.execution_backend_identifier == "distributional-dense-v1"
    assert model.training_telemetry().curvature_policy == "observed"


def test_public_tweedie_configured_prior_reml_publishes_certified_observed_state() -> None:
    frame, response = _certified_automatic_tweedie_fixture()
    penalty_names = tuple(f"{name}:x_{name}#wiggle" for name in ("mean", "dispersion", "power"))
    model = SuperLSS(
        family=_TweedieLSS(power_lower=1.08, power_upper=1.92),
        predictors=_all_smooth_tweedie_predictors(),
    ).fit_reml(
        frame,
        response,
        lambdas=dict.fromkeys(penalty_names, 1.0),
        max_reml_iter=120,
        reml_tol=1.0e-4,
        max_log_step=1.0,
        acceleration="none",
        practical_reml=False,
        outer="efs",
    )
    fitted = model._require_fitted()
    smoothing = fitted.smoothing
    assert smoothing is not None

    assert model.weight_semantics == "prior"
    assert fitted.fit_state.weight_contract == WeightContract("prior")
    assert tuple(predictor.name for predictor in model.predictors_) == (
        "mean",
        "dispersion",
        "power",
    )
    assert model.parameter_names_ == ("mean", "dispersion", "power")
    assert all(
        component.lambda_policy == LambdaPolicy.estimate() for component in fitted.layout.penalties
    )
    assert smoothing.config.max_iterations == 120
    assert smoothing.config.tolerance == 1.0e-4
    assert smoothing.config.max_log_step == 1.0
    assert smoothing.config.acceleration == "none"
    assert dict(smoothing.initial_lambdas) == dict.fromkeys(penalty_names, 1.0)
    assert 0 < smoothing.iterations < smoothing.config.max_iterations
    assert smoothing.terminal_raw_max_log_step <= smoothing.config.tolerance
    assert smoothing.matched_certified is True
    _assert_observed_coefficient_fits(model)
    parameters = model.predict_parameters(frame)
    links = model.predict_link(frame)
    assert tuple(parameters.columns) == model.parameter_names_
    assert tuple(links.columns) == model.parameter_names_
    assert np.all(np.isfinite(parameters.to_numpy()))
    assert np.all(parameters["mean"] > 0.0)
    assert np.all(parameters["dispersion"] > 0.0)
    assert np.all(parameters["power"] > model.family_.power_lower)
    assert np.all(parameters["power"] < model.family_.power_upper)
    np.testing.assert_array_equal(model.predict(frame), parameters["mean"].to_numpy())
    assert np.all(np.isfinite(model.covariance_))
    assert np.isfinite(model.result_.total_effective_df)
    assert all(np.isfinite(tuple(model.result_.predictor_edf.values())))
    assert fitted.null_model.parameter_names == model.parameter_names_
    assert np.all(np.isfinite(fitted.null_model.coefficients))
    assert fitted.fit_state.retained_rows is not None
    _assert_public_tweedie_round_trip(model, frame)


def test_public_tweedie_reml_uses_an_honest_practical_plateau_by_default() -> None:
    frame, response = _certified_automatic_tweedie_fixture()
    model = SuperLSS(
        family=_TweedieLSS(power_lower=1.08, power_upper=1.92),
        predictors=_all_smooth_tweedie_predictors(),
    ).fit_reml(
        frame,
        response,
        max_reml_iter=120,
        reml_tol=1.0e-4,
        max_log_step=1.0,
        acceleration="none",
        outer="efs",
    )
    smoothing = model._require_fitted().smoothing

    assert smoothing is not None
    assert smoothing.converged is True
    assert smoothing.convergence_reason == "practical_plateau"
    assert smoothing.terminal_raw_max_log_step > smoothing.config.tolerance
    assert smoothing.matched_certified is False
    assert model.smoothing_convergence_reason_ == "practical_plateau"
    assert model.smoothing_certified_ is False
    assert model.smoothing_unresolved_upper_bound_ == ()
    telemetry = model.training_telemetry()
    assert telemetry.smoothing_convergence_reason == "practical_plateau"
    assert telemetry.smoothing_certified is False
    assert telemetry.smoothing_unresolved_upper_bound == ()
    report = diagnose_distributional_fit(model._require_fitted())
    uncertified = next(
        finding for finding in report.findings if finding.code == "fit.termination_uncertified"
    )
    unsettled = next(
        finding for finding in report.findings if finding.code == "smoothing.trajectory_unsettled"
    )
    assert "practical" in uncertified.headline.lower()
    assert "practical" in unsettled.headline.lower()
    assert "did not settle" not in unsettled.interpretation.lower()
    assert unsettled.impacts == ("smoothing_selection",)
    _assert_public_tweedie_round_trip(model, frame)


def test_public_practical_fit_defers_cap_pressure_to_the_exact_face() -> None:
    rng = np.random.default_rng(7)
    levels = np.repeat(np.array(["a", "b", "c", "d"]), 10)
    response = rng.normal(size=len(levels))
    model = SuperLSS(
        family=GaussianLS(scale_floor=1.0e-4),
        predictors=(
            Predictor("location", {"effect": RandomEffect()}),
            Predictor("scale", {}),
        ),
    ).fit_reml(
        pd.DataFrame({"effect": levels}),
        response,
        lambdas={"location:effect#wiggle": 1000.0},
        max_lambda=1002.5,
        max_log_step=1.0e-3,
        max_reml_iter=10,
        reml_tol=1.0e-8,
        inner_tol=1.0e-10,
        reml_plateau_tol=1.0e-6,
    )

    # The practical stop never publishes a point that still carries upper-cap
    # pressure: the exact-face assessment runs first and resolves it.
    assert model.smoothing_convergence_reason_ != "practical_plateau"
    assert model.smoothing_convergence_reason_ == "lambda_change"
    assert model.smoothing_certified_ is False
    assert model.smoothing_unresolved_upper_bound_ == ()
    assert model.exact_face_components_ == ("location:effect#wiggle",)
    assert model.training_telemetry().smoothing_unresolved_upper_bound == ()
    report = diagnose_distributional_fit(model._require_fitted())
    assert not [
        finding for finding in report.findings if finding.code == "fit.lambda_cap_unresolved"
    ]


def test_public_tweedie_nonunit_prior_weights_scale_dispersion_not_replication() -> None:
    frame, response = _tweedie_fixture()
    weight = 2.5
    unit = _tweedie_model().fit(frame, response)
    weighted = _tweedie_model().fit(frame, response, sample_weight=np.full(len(frame), weight))
    unit_parameters = unit.predict_parameters(frame).to_numpy()
    weighted_parameters = weighted.predict_parameters(frame).to_numpy()
    envelope = _roundoff_envelope(unit_parameters, weighted_parameters)

    assert weighted.weight_semantics == "prior"
    np.testing.assert_allclose(
        weighted_parameters[:, (0, 2)],
        unit_parameters[:, (0, 2)],
        rtol=0.0,
        atol=envelope,
    )
    np.testing.assert_allclose(
        weighted_parameters[:, 1],
        weight * unit_parameters[:, 1],
        rtol=0.0,
        atol=weight * envelope,
    )
    assert abs(weighted.result_.log_likelihood - unit.result_.log_likelihood) <= envelope


def test_public_tweedie_configured_frequency_contract_matches_literal_replication() -> None:
    frame, response = _tweedie_fixture()
    counts = np.tile(np.array([1, 2, 1, 3], dtype=np.int64), len(frame) // 4)
    positions = np.repeat(np.arange(len(frame)), counts)
    # The shared loose policy isolates frequency semantics; the configured
    # prior test above owns optimizer-accuracy certification.
    frequency = _tweedie_model(automatic=True, frequency=True).fit_reml(
        frame,
        response,
        sample_weight=counts,
        max_reml_iter=12,
        reml_tol=0.25,
        retain_rows=False,
        outer="efs",
    )
    replicated = _tweedie_model(automatic=True).fit_reml(
        frame.iloc[positions].reset_index(drop=True),
        response[positions],
        max_reml_iter=12,
        reml_tol=0.25,
        outer="efs",
    )
    frequency_state = frequency._require_fitted().fit_state
    frequency_coefficients = np.asarray(tuple(frequency.coef_.values()))
    replicated_coefficients = np.asarray(tuple(replicated.coef_.values()))
    coefficient_envelope = _roundoff_envelope(
        frequency_coefficients,
        replicated_coefficients,
    )
    covariance_envelope = _roundoff_factor(
        frequency.covariance_, replicated.covariance_
    ) * np.maximum(
        1.0,
        np.maximum(np.abs(frequency.covariance_), np.abs(replicated.covariance_)),
    )
    frequency_parameters = frequency.predict_parameters(frame).to_numpy()
    replicated_parameters = replicated.predict_parameters(frame).to_numpy()
    parameter_envelope = _roundoff_envelope(frequency_parameters, replicated_parameters)
    frequency_lambda = frequency.smoothing_parameters_["mean:x_mean#wiggle"]
    replicated_lambda = replicated.smoothing_parameters_["mean:x_mean#wiggle"]
    lambda_envelope = _roundoff_envelope(
        np.array([frequency_lambda]),
        np.array([replicated_lambda]),
    )

    assert frequency.weight_semantics == "frequency"
    assert replicated.weight_semantics == "prior"
    assert frequency_state.weight_provenance.likelihood_count == len(positions)
    np.testing.assert_allclose(
        frequency_coefficients,
        replicated_coefficients,
        rtol=0.0,
        atol=coefficient_envelope,
    )
    assert np.all(np.abs(frequency.covariance_ - replicated.covariance_) <= covariance_envelope)
    np.testing.assert_allclose(
        frequency_parameters,
        replicated_parameters,
        rtol=0.0,
        atol=parameter_envelope,
    )
    assert set(frequency.smoothing_parameters_) == {"mean:x_mean#wiggle"}
    assert abs(frequency_lambda - replicated_lambda) <= lambda_envelope
    assert frequency.result_.smoothing_converged is True
    assert replicated.result_.smoothing_converged is True
    _assert_observed_coefficient_fits(frequency)
    _assert_public_tweedie_round_trip(frequency, frame)


@pytest.mark.parametrize(
    ("family", "predictors"),
    [
        (_GammaLSSubclass(), (Predictor("mean", {}), Predictor("scale", {}))),
        (_GammaLookalike(), (Predictor("mean", {}), Predictor("scale", {}))),
        (_TweedieLSSSubclass(), _intercept_only_tweedie_predictors()),
        (_TweedieLookalike(), _intercept_only_tweedie_predictors()),
        (
            _NegativeBinomialLSSubclass(),
            (Predictor("mean", {}), Predictor("theta", {})),
        ),
        (
            _NegativeBinomialLookalike(),
            (Predictor("mean", {}), Predictor("theta", {})),
        ),
    ],
)
def test_constructor_admits_structural_family_subclasses_and_lookalikes(
    family: DistributionalFamily,
    predictors: tuple[Predictor, ...],
) -> None:
    model = SuperLSS(family=family, predictors=predictors)

    assert model.family is family


def test_complete_fit_admission_requires_declared_family_configuration() -> None:
    with pytest.raises(
        TypeError,
        match="complete-fit.*ConfigurableDistributionalFamily.*to_config",
    ):
        SuperLSS(
            family=_UnconfiguredGammaLookalike(),
            predictors=(Predictor("mean", {}), Predictor("scale", {})),
        )


@pytest.mark.parametrize(
    ("family", "names", "message"),
    [
        (GammaLS(), ("mean",), "missing.*scale"),
        (GammaLS(), ("mean", "mean"), "duplicate.*mean"),
        (GammaLS(), ("mean", "shape"), "unknown.*shape"),
        (GammaLS(), ("scale", "mean"), "order.*mean.*scale"),
        (_TweedieLSS(), ("mean", "dispersion"), "missing.*power"),
        (_TweedieLSS(), ("mean", "dispersion", "dispersion"), "duplicate.*dispersion"),
        (_TweedieLSS(), ("mean", "dispersion", "shape"), "unknown.*shape"),
        (_TweedieLSS(), ("dispersion", "mean", "power"), "order.*mean.*dispersion.*power"),
        (_NegativeBinomialLS(), ("mean",), "missing.*theta"),
        (_NegativeBinomialLS(), ("mean", "mean"), "duplicate.*mean"),
        (_NegativeBinomialLS(), ("mean", "shape"), "unknown.*shape"),
        (_NegativeBinomialLS(), ("theta", "mean"), "order.*mean.*theta"),
    ],
)
def test_constructor_rejects_public_predictors_that_do_not_match_family_order(
    family: object,
    names: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SuperLSS(
            family=family,  # type: ignore[arg-type]
            predictors=tuple(Predictor(name, {}) for name in names),
        )


@pytest.mark.parametrize(
    ("predictors", "message"),
    [
        ((Predictor("location", {}),), "missing.*scale"),
        (
            (Predictor("location", {}), Predictor("location", {})),
            "duplicate.*location",
        ),
        (
            (Predictor("location", {}), Predictor("shape", {})),
            "unknown.*shape",
        ),
        (
            (Predictor("scale", {}), Predictor("location", {})),
            "order.*location.*scale",
        ),
    ],
)
def test_constructor_rejects_predictors_that_do_not_match_family_order(
    predictors: tuple[Predictor, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SuperLSS(family=GaussianLS(), predictors=predictors)


@pytest.mark.parametrize(
    ("family", "predictors", "message"),
    [
        (
            GaussianLS(scale_floor=0.01),
            (Predictor("location", {}), Predictor("scale", {}, link="identity")),
            "link.*scale.*incompatible",
        ),
        (
            _TweedieLSS(),
            (
                Predictor("mean", {}),
                Predictor("dispersion", {}),
                Predictor("power", {}, link="identity"),
            ),
            "link.*power.*incompatible",
        ),
    ],
)
def test_constructor_rejects_public_incompatible_links_before_fit(
    family: object,
    predictors: tuple[Predictor, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SuperLSS(family=family, predictors=predictors)  # type: ignore[arg-type]


def test_model_fit_refuses_a_family_plan_bound_to_a_substituted_root_before_geometry() -> None:
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 6)})
    response = np.array([-0.8, -0.1, 0.4, 0.9, 1.3, 2.1])
    family = _SubstitutingLikelihoodFamily(scale_floor=0.01)

    with pytest.raises(
        UnsupportedLikelihoodContractError,
        match="changed the fitted likelihood weights",
    ):
        fit_dense_distributional(
            frame,
            response,
            family=family,
            predictors=(
                Predictor("location", {"x": _FailIfCompiledNumeric()}),
                Predictor("scale", {}),
            ),
            sample_weight=np.array([0.4, 0.7, 1.1, 1.6, 2.2, 3.0]),
            weight_contract=WeightContract("prior"),
        )


def test_public_fit_reml_exposes_only_efs_and_honours_fixed_policy() -> None:
    frame, response, _, _ = _fixture()
    model = SuperLSS(
        family=GaussianLS(),
        predictors=(
            Predictor(
                "location",
                {
                    "x": Spline(
                        kind="cr",
                        n_knots=6,
                        lambda_policy={"wiggle": LambdaPolicy.fixed(0.45)},
                    )
                },
            ),
            Predictor("scale", {}),
        ),
    )

    returned = model.fit_reml(
        frame,
        response,
        method="efs",
        max_reml_iter=4,
        reml_tol=1.0e-5,
        max_inner_iter=80,
        inner_tol=1.0e-8,
    )

    assert returned is model
    assert model.smoothing_parameters_ == {"location:x#wiggle": 0.45}
    assert model.result_.smoothing_converged is True
    assert model.result_.n_smoothing_iter == 0
    assert model._require_fitted().result.config.coefficient_curvature == "observed"

    fresh = _linear_model()
    with pytest.raises(NotImplementedError, match="only.*efs"):
        fresh.fit_reml(frame, response, method="laml")


def test_superlss_fit_uses_the_shared_solver_blas_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, _, _ = _fixture()
    model = _linear_model()
    events: list[str] = []
    fitted = object()

    @contextmanager
    def recorded_scope():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def fit_spy(*args, **kwargs):
        assert events == ["enter"]
        return fitted

    monkeypatch.setattr(api_module, "solver_blas_threads", recorded_scope)
    monkeypatch.setattr(api_module, "fit_dense_distributional", fit_spy)

    assert model.fit(frame, response) is model
    assert model._model is fitted
    assert events == ["enter", "exit"]


def test_superlss_reports_its_coefficient_dimension_to_the_blas_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, _, _ = _fixture()
    model = _linear_model()
    dimensions: list[int] = []

    monkeypatch.setattr(model_module, "allow_wide_design", dimensions.append)
    model.fit(frame, response)

    assert dimensions == [model._require_fitted().layout.n_coefficients]


def test_fit_reml_forwards_multisecant_config_and_phase_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, _, _ = _fixture()
    model = _linear_model()
    recorder = FitPhaseRecorder(clock=lambda: 0.0)
    calls: list[dict[str, object]] = []

    def fit_spy(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return self

    monkeypatch.setattr(SuperLSS, "_fit", fit_spy)

    assert model.fit_reml(frame, response) is model
    assert (
        model.fit_reml(
            frame,
            response,
            acceleration="multisecant",
            acceleration_history=3,
            acceleration_max_amplification=4.0,
            phase_recorder=recorder,
        )
        is model
    )

    default_config = calls[0]["efs_config"]
    accelerated_config = calls[1]["efs_config"]
    assert default_config.acceleration == "none"
    assert accelerated_config.acceleration == "multisecant"
    assert accelerated_config.acceleration_history == 3
    assert accelerated_config.acceleration_max_amplification == 4.0
    assert calls[0]["phase_recorder"] is None
    assert calls[1]["phase_recorder"] is recorder


def test_fixed_fit_honours_component_lambda_policy_without_duplicate_input() -> None:
    frame, response, _, _ = _fixture()
    model = SuperLSS(
        family=GaussianLS(),
        predictors=(
            Predictor(
                "location",
                {
                    "x": Spline(
                        kind="cr",
                        n_knots=6,
                        lambda_policy={"wiggle": LambdaPolicy.fixed(0.45)},
                    )
                },
            ),
            Predictor("scale", {}),
        ),
    )

    model.fit(frame, response)

    assert model.smoothing_parameters_ == {"location:x#wiggle": 0.45}
    assert model.result_.smoothing_converged is None


def test_scalar_offset_keyword_is_not_part_of_the_public_fit_contract() -> None:
    frame, response, _, _ = _fixture()
    model = _linear_model()

    with pytest.raises(TypeError, match="offset"):
        model.fit(  # type: ignore[call-arg]
            frame,
            response,
            offset=np.zeros(len(frame)),
        )


def test_failed_second_fit_keeps_the_previous_complete_revision() -> None:
    frame, response, _, _ = _fixture()
    model = _linear_model().fit(frame, response)
    accepted_result = model.result_
    accepted_prediction = model.predict(frame).copy()
    invalid_response = response.copy()
    invalid_response[5] = np.nan

    with pytest.raises(ValueError, match="finite"):
        model.fit(frame, invalid_response)

    assert model.result_ is accepted_result
    np.testing.assert_array_equal(model.predict(frame), accepted_prediction)


def test_public_artifact_round_trip_preserves_fitted_state_and_execution_config() -> None:
    frame, response, weights, offsets = _fixture()
    model = SuperLSS(
        family=GaussianLS(scale_floor=0.025),
        predictors=(
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {"z": Numeric()}),
        ),
        n_bins={"x": 128, "z": 96},
        separation="error",
    ).fit(
        frame,
        response,
        sample_weight=weights,
        offsets=offsets,
        retain_rows=False,
    )

    artifact = model.to_bytes()
    restored = SuperLSS.from_bytes(artifact)

    assert isinstance(artifact, bytes)
    assert restored.discrete is False
    assert restored.n_bins == {"x": 128, "z": 96}
    assert restored.weight_semantics == "prior"
    assert restored.separation == "error"
    assert restored._require_fitted().fit_state.requested_discrete is False
    assert restored._require_fitted().fit_state.requested_n_bins == {"x": 128, "z": 96}
    assert restored._require_fitted().fit_state.requested_chunk_size is None
    assert restored.family_.scale_floor == 0.025
    assert restored.result_.coefficient_names == model.result_.coefficient_names
    assert restored.training_telemetry() == model.training_telemetry()
    np.testing.assert_array_equal(
        restored.predict_link(frame, offsets=offsets),
        model.predict_link(frame, offsets=offsets),
    )
    np.testing.assert_array_equal(
        restored.predict_parameters(frame, offsets=offsets),
        model.predict_parameters(frame, offsets=offsets),
    )

    corrupted = json.loads(artifact)
    corrupted["public_api"]["config"]["discrete"] = True
    with pytest.raises(DistributionalSerializationError, match="digest does not match"):
        SuperLSS.from_bytes(json.dumps(corrupted))

    with pytest.raises(RuntimeError, match="not fitted"):
        _linear_model().to_bytes()


def test_public_schema_version_is_independent_of_the_envelope_version() -> None:
    """The two version numbers are separate concepts, and both must be present.

    ``_PUBLIC_SCHEMA_VERSION`` versions only the ``public_api`` block; the
    envelope beneath it carries ``serialization.SCHEMA_VERSION``.  They are
    allowed to differ and version their own key sets.  The artifact records both
    so whichever layer refuses a stale artifact can identify its own boundary.
    """
    frame, response, _, _ = _fixture()
    model = _linear_model().fit(frame, response)
    artifact = json.loads(model.to_bytes())

    public = artifact["public_api"]
    assert set(public) == {"artifact_type", "schema_version", "config", "sha256"}
    assert public["schema_version"] == "2.0.0"
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert public["schema_version"] != artifact["schema_version"]
    assert set(public["config"]) == {
        "discrete",
        "n_bins",
        "separation",
        "weight_contract",
    }
    assert public["config"]["separation"] == "warn"
    assert public["config"]["weight_contract"] == {
        "schema_version": 1,
        "semantics": "prior",
    }


def test_from_bytes_reports_a_precontract_envelope_as_typed_legacy() -> None:
    """A pre-contract envelope under a current public block is unknowable.

    The public block's digest covers the payload and config only, so an artifact
    with an older envelope clears every public check.  The internal schema must
    route it to the typed legacy-weight refusal before unpickling, because no
    old compact field proves whether its weights were counts or powers.
    """
    frame, response, _, _ = _fixture()
    model = _linear_model().fit(frame, response)
    artifact = json.loads(model.to_bytes())

    stale = "1.0.0"  # what every pre-penalty-geometry writer stamped
    artifact["schema_version"] = stale
    for component in artifact["manifest"]["penalties"]["components"]:
        for key in ("penalty_kind", "repeat_count", "block_width"):
            component.pop(key, None)

    with pytest.raises(LegacyPowerWeightArtifactError, match="legacy|pre-contract"):
        SuperLSS.from_bytes(json.dumps(artifact))


def test_frequency_facade_round_trip_and_refit_keep_the_canonical_contract() -> None:
    frame, response, _, offsets = _fixture(14)
    counts = np.array([1, 3, 2, 1, 4, 1, 2, 3, 1, 2, 1, 4, 2, 1])
    predictors = (
        Predictor("location", {"x": Numeric()}),
        Predictor("scale", {"z": Numeric()}),
    )
    model = SuperLSS(
        family=GaussianLS(scale_floor=0.02),
        predictors=predictors,
        weight_semantics="frequency",
    ).fit(frame, response, sample_weight=counts, offsets=offsets, retain_rows=False)
    restored = SuperLSS.from_bytes(model.to_bytes())
    state = restored._require_fitted().fit_state

    assert restored.weight_semantics == "frequency"
    assert state.weight_contract == WeightContract("frequency")
    assert state.null_model.weight_contract == state.weight_contract
    assert state.null_model.weight_provenance == state.weight_provenance
    assert state.family_likelihood_plan_identifier == (
        state.solver_result.family_likelihood_plan_identifier
    )
    assert state.null_model.family_likelihood_plan_identifier == (
        state.family_likelihood_plan_identifier
    )

    new_frame = frame.iloc[::-1].reset_index(drop=True)
    new_response = response[::-1] + np.linspace(-0.04, 0.06, len(response))
    new_offsets = {name: values[::-1].copy() for name, values in offsets.items()}
    new_counts = np.array([2, 1, 3, 2, 1, 4, 1, 2, 3, 1, 2, 1, 4, 2])
    refit_dense_distributional(
        restored._require_fitted(),
        new_frame,
        new_response,
        sample_weight=new_counts,
        offsets=new_offsets,
    )
    fresh = SuperLSS(
        family=GaussianLS(scale_floor=0.02),
        predictors=predictors,
        weight_semantics="frequency",
    ).fit(
        new_frame,
        new_response,
        sample_weight=new_counts,
        offsets=new_offsets,
    )
    positions = np.repeat(np.arange(len(new_frame)), new_counts)
    expanded = SuperLSS(
        family=GaussianLS(scale_floor=0.02),
        predictors=predictors,
    ).fit(
        new_frame.iloc[positions].reset_index(drop=True),
        new_response[positions],
        offsets={name: values[positions] for name, values in new_offsets.items()},
    )

    assert restored.weight_semantics == "frequency"
    assert restored._require_fitted().fit_state.weight_contract == WeightContract("frequency")
    restored_coefficients = np.array(tuple(restored.coef_.values()))
    np.testing.assert_allclose(
        restored_coefficients,
        tuple(fresh.coef_.values()),
        rtol=0.0,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        restored_coefficients,
        tuple(expanded.coef_.values()),
        rtol=0.0,
        atol=2e-11,
    )


def test_public_schema_routes_legacy_future_and_current_duplicate_separately() -> None:
    frame, response, _, _ = _fixture(12)
    model = SuperLSS(
        family=GaussianLS(),
        predictors=(Predictor("location", {}), Predictor("scale", {})),
        weight_semantics="frequency",
    ).fit(frame, response, sample_weight=np.ones(len(frame), dtype=np.int64))

    legacy = json.loads(model.to_bytes())
    legacy["public_api"]["schema_version"] = "1.0.0"
    legacy["public_api"]["config"].pop("weight_contract")
    legacy["public_api"]["config"].pop("separation")
    raw = b"validly hashed payload that must not be unpickled"
    legacy["payload"]["data"] = base64.b64encode(raw).decode("ascii")
    legacy["payload"]["sha256"] = hashlib.sha256(raw).hexdigest()
    _resign_public_config(legacy)
    with pytest.raises(LegacyPowerWeightArtifactError, match="legacy|pre-contract"):
        SuperLSS.from_bytes(json.dumps(legacy))

    future = json.loads(model.to_bytes())
    future["public_api"]["schema_version"] = "3.0.0"
    with pytest.raises(DistributionalSerializationError, match="unreadable by this build"):
        SuperLSS.from_bytes(json.dumps(future))

    mismatch = json.loads(model.to_bytes())
    mismatch["public_api"]["config"]["weight_contract"] = {
        "schema_version": 1,
        "semantics": "prior",
    }
    _resign_public_config(mismatch)
    with pytest.raises(DistributionalSerializationError, match="contract.*does not match"):
        SuperLSS.from_bytes(json.dumps(mismatch))


@pytest.mark.parametrize(
    ("accepted_n_bins", "aliased_n_bins"),
    [
        (1, True),
        (1, 1.0),
        ({"x": 2, "z": 1}, {"x": 2, "z": True}),
        ({"x": 2, "z": 1}, {"x": 2, "z": 1.0}),
    ],
    ids=["scalar-bool", "scalar-float", "mapping-bool", "mapping-float"],
)
def test_current_public_n_bins_rejects_type_aliases_after_valid_resigning(
    accepted_n_bins: int | dict[str, int],
    aliased_n_bins: object,
) -> None:
    frame, response, _, _ = _fixture(12)
    model = SuperLSS(
        family=GaussianLS(),
        predictors=(Predictor("location", {}), Predictor("scale", {})),
        n_bins=accepted_n_bins,
    ).fit(frame, response)
    artifact = json.loads(model.to_bytes())
    artifact["public_api"]["config"]["n_bins"] = aliased_n_bins
    _resign_public_config(artifact)

    with pytest.raises(DistributionalSerializationError, match="n_bins.*invalid"):
        SuperLSS.from_bytes(json.dumps(artifact))


def test_current_public_weight_schema_rejects_float_alias_after_valid_resigning() -> None:
    frame, response, _, _ = _fixture(12)
    model = _linear_model().fit(frame, response)
    artifact = json.loads(model.to_bytes())
    artifact["public_api"]["config"]["weight_contract"]["schema_version"] = 1.0
    _resign_public_config(artifact)

    with pytest.raises(
        DistributionalSerializationError,
        match="contract configuration is invalid",
    ):
        SuperLSS.from_bytes(json.dumps(artifact))


@pytest.mark.parametrize("invalid", ["silent", 1, True, None])
def test_current_public_separation_policy_rejects_invalid_values_after_valid_resigning(
    invalid: object,
) -> None:
    frame, response, _, _ = _fixture(12)
    model = _linear_model().fit(frame, response)
    artifact = json.loads(model.to_bytes())
    artifact["public_api"]["config"]["separation"] = invalid
    _resign_public_config(artifact)

    with pytest.raises(
        DistributionalSerializationError,
        match="separation configuration is invalid",
    ):
        SuperLSS.from_bytes(json.dumps(artifact))


def _gamma_curvature_fixture(n: int = 400) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(2026090201)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.uniform(-1.0, 1.0, n)
    mean = np.exp(0.4 + 0.6 * np.sin(np.pi * x))
    shape = 1.0 / np.exp(-0.9 + 0.3 * z) ** 2
    return pd.DataFrame({"x": x, "z": z}), rng.gamma(shape, mean / shape)


def _gamma_curvature_predictors() -> tuple[Predictor, Predictor]:
    return (
        Predictor("mean", {"x": Spline(kind="cr", k=8)}),
        Predictor("scale", {"z": Spline(kind="cr", k=6)}),
    )


_GAMMA_CURVATURE_LAMBDAS = {"mean:x#wiggle": 1.0, "scale:z#wiggle": 1.0}


class _ExpectedInformationSpy:
    """Delegate a complete family and count its expected-information requests."""

    def __init__(self, base: DistributionalFamily) -> None:
        self.base = base
        self.expected_information_calls = 0

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def capabilities(self):
        return self.base.capabilities

    @property
    def default_prediction_name(self):
        return self.base.default_prediction_name

    def to_config(self):
        return {"type": "spy", "base": dict(self.base.to_config())}

    def bind_likelihood(self, y, weights, observation):
        return self.base.bind_likelihood(y, weights, observation)

    def initialize(self, y, plan):
        return self.base.initialize(y, plan)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        return self.base.evaluate_natural(y, theta, plan, derivative_order=derivative_order)

    def expected_information_natural(self, theta, plan):
        self.expected_information_calls += 1
        return self.base.expected_information_natural(theta, plan)

    def default_prediction(self, theta):
        return self.base.default_prediction(theta)


@dataclass(frozen=True)
class _IndefiniteObservedGaussianLS:
    """GaussianLS whose observed curvature is materially indefinite at every state.

    The scale-scale Hessian channel is negated, so the observed coefficient curvature
    always carries a negative direction while the family's expected information stays
    positive definite: the terminal curvature policy has to fall back, never refuse.
    """

    inner: GaussianLS = GaussianLS()

    @property
    def parameters(self):
        return self.inner.parameters

    @property
    def capabilities(self):
        return self.inner.capabilities

    @property
    def default_prediction_name(self):
        return self.inner.default_prediction_name

    def to_config(self):
        return {"type": "IndefiniteObservedGaussianLS"}

    def bind_likelihood(self, y, weights, observation):
        return self.inner.bind_likelihood(y, weights, observation)

    def initialize(self, y, plan):
        return self.inner.initialize(y, plan)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        evaluation = self.inner.evaluate_natural(y, theta, plan, derivative_order=derivative_order)
        if evaluation.hessian_packed is None:
            return evaluation
        hessian = np.array(evaluation.hessian_packed, dtype=np.float64)
        hessian[:, 2] = -hessian[:, 2]
        return replace(evaluation, hessian_packed=hessian)

    def expected_information_natural(self, theta, plan):
        return self.inner.expected_information_natural(theta, plan)

    def default_prediction(self, theta):
        return self.inner.default_prediction(theta)


def test_expected_information_is_a_fallback_not_a_switch() -> None:
    """A family with expected information (GammaLS) is solved with the observed Hessian by
    default; Fisher scoring is a request.  The solve policy lives in the requested solver
    config, while the terminal telemetry always requests observed curvature for inference."""
    frame, y = _gamma_curvature_fixture()
    observed_spy = _ExpectedInformationSpy(GammaLS())
    model = SuperLSS(family=observed_spy, predictors=_gamma_curvature_predictors())
    assert model.coefficient_curvature == "observed"
    fitted = model.fit(frame, y, lambdas=_GAMMA_CURVATURE_LAMBDAS)._require_fitted()
    assert fitted.fit_state.requested_solver_config.coefficient_curvature == "observed"
    assert model.training_telemetry().curvature_policy == "observed"
    telemetry = fitted.fit_state.solver_result.terminal_curvature
    assert telemetry.requested_source == "observed"
    assert telemetry.actual_source == "observed"
    assert telemetry.fallback_count == 0
    assert observed_spy.expected_information_calls == 0
    assert fitted.fit_state.solver_result.converged
    observed_iterations = fitted.fit_state.solver_result.iterations

    fisher_spy = _ExpectedInformationSpy(GammaLS())
    fisher = SuperLSS(
        family=fisher_spy,
        predictors=_gamma_curvature_predictors(),
        coefficient_curvature="fisher",
    )
    assert fisher.coefficient_curvature == "fisher"
    fitted_fisher = fisher.fit(frame, y, lambdas=_GAMMA_CURVATURE_LAMBDAS)._require_fitted()
    assert fitted_fisher.fit_state.requested_solver_config.coefficient_curvature == "fisher"
    assert fisher.training_telemetry().curvature_policy == "fisher"
    assert fitted_fisher.fit_state.solver_result.converged
    assert fisher_spy.expected_information_calls >= fitted_fisher.fit_state.solver_result.iterations
    assert observed_iterations <= fitted_fisher.fit_state.solver_result.iterations
    observed_objective = fitted.fit_state.solver_result.penalized_log_likelihood
    fisher_objective = fitted_fisher.fit_state.solver_result.penalized_log_likelihood
    assert abs(observed_objective - fisher_objective) < 1.0e-8 * (1.0 + abs(observed_objective))


def test_fisher_request_requires_the_capability() -> None:
    with pytest.raises(ValueError, match="expected information"):
        SuperLSS(
            family=_TweedieLSS(),
            predictors=_mean_smooth_tweedie_predictors(),
            coefficient_curvature="fisher",
        )
    with pytest.raises(ValueError, match="coefficient_curvature"):
        SuperLSS(
            family=GammaLS(),
            predictors=_gamma_curvature_predictors(),
            coefficient_curvature="newton",  # type: ignore[arg-type]
        )


def test_material_indefiniteness_still_falls_back_to_fisher() -> None:
    """The solver-level fixture in test_distributional_solver drives the policy directly; this
    is the public path.  A hostile start alone cannot reach the fallback (Levenberg damping
    absorbs it inside the iteration loop), so the family's observed curvature is made
    indefinite at every state instead; the accepted terminal point then lands on Fisher with
    the fallback recorded, while the requested policy stays observed."""
    frame, response, _, _ = _fixture()
    model = SuperLSS(
        family=_IndefiniteObservedGaussianLS(),
        predictors=(
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {"z": Numeric()}),
        ),
    )
    model.fit(frame, response, lambdas={}, max_inner_iter=5)
    telemetry = model._require_fitted().fit_state.solver_result.terminal_curvature
    assert telemetry.requested_source == "observed"
    assert telemetry.actual_source == "fisher"
    assert telemetry.reason == "material_indefiniteness_after_retry"
    assert telemetry.fallback_count >= 1
    assert model.training_telemetry().curvature_policy == "observed"
    assert model.training_telemetry().curvature.fallback_count >= 1


def _small_gaussian_reml_model() -> tuple[SuperLSS, pd.DataFrame, np.ndarray]:
    frame, response, _, _ = _fixture()
    model = SuperLSS(
        family=GaussianLS(scale_floor=0.02),
        predictors=(
            Predictor("location", {"x": Spline(kind="cr", n_knots=6)}),
            Predictor("scale", {"z": Numeric()}),
        ),
    )
    return model, frame, response


def test_fit_reml_outer_defaults_to_the_newton_endgame_and_is_validated() -> None:
    model, frame, response = _small_gaussian_reml_model()
    model.fit_reml(frame, response)
    smoothing = model._require_fitted().smoothing
    assert smoothing is not None
    assert smoothing.config.outer == "efs+newton"
    assert smoothing.convergence_reason in {"stationary", "fixed_only"}
    with pytest.raises(ValueError, match="outer"):
        model.fit_reml(frame, response, outer="newton")
    efs_model, _, _ = _small_gaussian_reml_model()
    efs_model.fit_reml(frame, response, outer="efs")
    efs = efs_model._require_fitted().smoothing
    assert efs is not None
    assert efs.config.outer == "efs"
