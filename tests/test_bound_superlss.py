"""Regression coverage for family-bound public construction."""

import inspect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import (
    BoundPredictor,
    GammaLS,
    GaussianLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    LogNormalLS,
    NegativeBinomialLS,
    Numeric,
    Predictor,
    Spline,
    SuperLSS,
    TensorInteraction,
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
    bind_predictor,
    s,
    term,
    ti,
)
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.family import ParameterSpec, ParameterSupport
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.result import DenseSolverConfig, DistributionalEFSConfig
from superglm.distributional.weights import WeightContract
from superglm.links import IdentityLink
from tests.test_bound_predictors import _FourParameterFamily
from tests.test_superlss_api import _ExpectedInformationSpy, _fixture, _roundoff_factor


def test_family_first_constructor_orders_named_predictors():
    family = TweedieLSS(power_lower=1.08, power_upper=1.92)
    model = SuperLSS(family, family.p(), family.mu("x"), family.phi())
    assert tuple(p.name for p in model.predictors) == ("mean", "dispersion", "power")
    assert model.family.to_config() == family.to_config()
    assert model.family is not family


@pytest.mark.parametrize(
    "family,helpers",
    [
        (GaussianLS(), ("location", "scale")),
        (GammaLS(), ("mean", "scale")),
        (TweedieLSS(), ("mu", "phi", "p")),
        (NegativeBinomialLS(), ("mean", "theta")),
        (GeneralizedParetoLSS(), ("scale", "shape")),
        (GeneralizedGammaLSS(), ("mean", "scale", "shape")),
        (GeneralizedGammaLSS(parametrisation="location"), ("location", "scale", "shape")),
        (LogNormalLS(), ("mean", "scale")),
        (LogNormalLS(parametrisation="location"), ("location", "scale")),
        (TwoPieceLogNormalLSS(), ("mean", "scale", "skew")),
        (TwoPieceLogNormalLSS(parametrisation="location"), ("location", "scale", "skew")),
        (TwoPieceNormalLSS(), ("location", "scale", "skew")),
    ],
)
def test_all_public_family_modes_normalize_helper_order(family, helpers):
    model = SuperLSS(family, *(getattr(family, helper)() for helper in reversed(helpers)))
    assert tuple(p.name for p in model.predictors) == tuple(p.name for p in family.parameters)
    assert all(p.intercept and not p.features for p in model.predictors)


class _ConfiguredFourParameterFamily(_FourParameterFamily):
    def to_config(self):
        return {"type": "configured_four", "lower": self.config["lower"][0]}


def test_public_constructor_accepts_custom_four_parameter_family():
    family = _ConfiguredFourParameterFamily()
    model = SuperLSS(family, *(bind_predictor(family, name) for name in ("d", "b", "a", "c")))
    family.config["lower"][0] = 0.5
    model.family.config["lower"][0] = 0.7
    assert tuple(p.name for p in model.predictors) == ("a", "b", "c", "d")
    assert model.family.parameters[0].support.lower == 0.1


def test_numeric_string_declaration_points_to_explicit_categorical_encoding():
    frame, response, _, _ = _fixture()
    frame["area"] = "north"
    family = GaussianLS()
    with pytest.raises((TypeError, ValueError), match="cat\\("):
        SuperLSS(family, family.location("area"), family.scale()).fit(frame, response)


def test_public_constructor_signature_is_explicit_and_family_first():
    parameters = inspect.signature(SuperLSS).parameters
    assert parameters["family"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert parameters["predictors"].kind is inspect.Parameter.VAR_POSITIONAL
    assert tuple(parameters) == (
        "family",
        "predictors",
        "weight_semantics",
        "discrete",
        "n_bins",
        "separation",
        "coefficient_curvature",
    )
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in list(parameters.values())[2:])


@pytest.mark.parametrize("kind", ["dict", "tuple", "template", "method", "duplicate", "foreign"])
def test_public_constructor_rejects_invalid_bound_arguments(kind):
    family = GaussianLS()
    arguments, error, message = {
        "dict": (({},), TypeError, "BoundPredictor"),
        "tuple": (((family.location(), family.scale()),), TypeError, "BoundPredictor"),
        "template": ((Predictor("location", {}),), TypeError, "BoundPredictor"),
        "method": ((family.location,), TypeError, "bare method.*location"),
        "duplicate": ((family.location(), family.location()), ValueError, "Duplicate.*location"),
        "foreign": ((GaussianLS().location(),), ValueError, "different family instance"),
    }[kind]
    with pytest.raises(error, match=message):
        SuperLSS(family, *arguments)


def test_removed_keyword_constructor_is_rejected():
    family = GaussianLS()
    with pytest.raises(TypeError):
        SuperLSS(family=family, predictors=(family.location(), family.scale()))
    with pytest.raises(TypeError, match="predictors"):
        SuperLSS(family, predictors=(family.location(), family.scale()))


def test_missing_predictor_points_to_the_required_tweedie_helper():
    family = TweedieLSS()
    with pytest.raises(ValueError) as error:
        SuperLSS(family, family.mu("x"), family.phi())
    message = str(error.value)
    assert "power" in message and "SuperLSS(" in message
    marked = [line for line in message.splitlines() if "<---" in line]
    assert len(marked) == 1 and "family.p(...)" in marked[0]
    assert "constant" not in message


def test_family_and_predictor_accessors_isolate_mutable_configuration():
    family = _ExpectedInformationSpy(GaussianLS(scale_floor=0.02))
    feature = Numeric()
    location = bind_predictor(family, "location", term("x", feature))
    scale = bind_predictor(family, "scale", "z")
    model = SuperLSS(family, location, scale)
    family.base = GaussianLS(scale_floor=0.3)
    model.family.base = GaussianLS(scale_floor=0.4)
    assert model.family.base.scale_floor == 0.02
    model.predictors[0].features["x"].name = "mutated"
    assert getattr(model.predictors[0].features["x"], "name", None) != "mutated"
    frame, response, _, _ = _fixture()
    model.fit(frame, response)
    expected = model.predict(frame)
    model.family_.base = GaussianLS(scale_floor=0.5)
    assert model.family_.base.scale_floor == 0.02
    np.testing.assert_array_equal(model.predict(frame), expected)
    assert isinstance(location, BoundPredictor)


class _ShiftedIdentityLink(IdentityLink):
    def __init__(self, shift=0.0):
        self.shift = shift

    def link(self, value):
        return super().link(value) - self.shift

    def inverse(self, value):
        return super().inverse(value) + self.shift


def _shifted_parameters(link=None):
    return (
        ParameterSpec(
            "location",
            _ShiftedIdentityLink() if link is None else link,
            "location",
            ParameterSupport(),
        ),
        GaussianLS().parameters[1],
    )


@pytest.fixture
def class_metadata_family():
    # Each test owns its class too, so caller mutations cannot leak between tests.
    class CustomGaussian(GaussianLS):
        parameters = _shifted_parameters()

    return CustomGaussian()


@pytest.mark.parametrize("source", ["caller", "family", "family_"])
def test_class_metadata_mutation_does_not_change_fitted_predictions(class_metadata_family, source):
    family = class_metadata_family
    model = SuperLSS(family, family.location("x"), family.scale())
    x = np.linspace(-1, 1, 30)
    frame = pd.DataFrame({"x": x})
    response = 1.0 + 2.0 * x + np.random.default_rng(1).normal(scale=0.3, size=len(x))
    model.fit(frame, response, lambdas={})
    expected = model.predict(frame)
    target = family if source == "caller" else getattr(model, source)
    target.parameters[0].default_link.shift = 7.0
    np.testing.assert_array_equal(model.predict(frame), expected)


@pytest.mark.parametrize("source", ["caller", "family"])
def test_class_metadata_is_owned_before_fit(class_metadata_family, source):
    family = class_metadata_family
    model = SuperLSS(family, family.location("x"), family.scale())
    target = family if source == "caller" else model.family
    target.parameters[0].default_link.shift = 7.0
    assert model.family.parameters[0].default_link.shift == 0.0


def test_snapshot_preserves_link_aliases_within_its_own_configuration(class_metadata_family):
    family = class_metadata_family
    family.location_link = family.parameters[0].default_link
    model = SuperLSS(family, family.location(), family.scale())
    owned = model.family
    owned.location_link.shift = 3.0
    assert owned.parameters[0].default_link.shift == 3.0
    assert family.location_link.shift == 0.0
    assert model.family.parameters[0].default_link.shift == 0.0


def test_read_only_shared_parameter_metadata_is_refused():
    parameters = _shifted_parameters()

    class SharedMetadataGaussian(GaussianLS):
        @property
        def parameters(self):
            return parameters

    family = SharedMetadataGaussian()
    with pytest.raises(TypeError, match="family.*snapshot"):
        SuperLSS(family, family.location(), family.scale())


def test_link_that_returns_itself_from_deepcopy_is_refused():
    class SharedLink(_ShiftedIdentityLink):
        def __deepcopy__(self, memo):
            return self

    class SharedLinkGaussian(GaussianLS):
        parameters = _shifted_parameters(SharedLink())

    family = SharedLinkGaussian()
    with pytest.raises(TypeError, match="family.*snapshot"):
        SuperLSS(family, family.location(), family.scale())


@pytest.mark.parametrize("storage", ["read_only", "frozen"])
def test_independent_read_only_and_frozen_metadata_remains_supported(storage):
    if storage == "read_only":

        class IndependentGaussian(GaussianLS):
            def __init__(self):
                super().__init__()
                self._parameters = _shifted_parameters()

            @property
            def parameters(self):
                return self._parameters

    else:

        @dataclass(frozen=True)
        class IndependentGaussian(GaussianLS):
            _parameters: tuple = field(default_factory=_shifted_parameters)

            @property
            def parameters(self):
                return self._parameters

    family = IndependentGaussian()
    model = SuperLSS(family, family.location(), family.scale())
    family.parameters[0].default_link.shift = 7.0
    model.family.parameters[0].default_link.shift = 9.0
    assert model.family.parameters[0].default_link.shift == 0.0


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("reml", [False, True])
def test_bound_smooth_interaction_matches_internal_fit(discrete, reml):
    frame, response, weights, offsets = _fixture(n=160)
    family = GaussianLS(scale_floor=0.02)
    model = SuperLSS(
        family,
        family.scale("z"),
        family.location(ti("x", "z", n_knots=(4, 4)), s("x", n_knots=4), s("z", n_knots=4)),
        discrete=discrete,
        n_bins=32,
    )
    templates = (
        Predictor(
            "location",
            {"x": Spline(n_knots=4), "z": Spline(n_knots=4)},
            interaction_specs={"x:z": TensorInteraction("x", "z", n_knots=(4, 4))},
        ),
        Predictor("scale", {"z": Numeric()}),
    )
    lambdas = (
        None
        if reml
        else {
            "location:x#wiggle": 0.7,
            "location:z#wiggle": 0.9,
            "location:x:z#margin_x": 1.1,
            "location:x:z#margin_z": 1.3,
        }
    )
    baseline = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.02),
        predictors=templates,
        weight_contract=WeightContract("prior"),
        sample_weight=weights,
        offsets=offsets,
        config=DenseSolverConfig(max_iterations=100, tolerance=1.0e-7),
        lambdas=lambdas,
        efs_config=(
            DistributionalEFSConfig(
                max_iterations=100, initial_lambda=None, practical_convergence=True
            )
            if reml
            else None
        ),
        discrete=discrete,
        n_bins=32,
        chunk_size="auto" if discrete else None,
    )
    fit = model.fit_reml if reml else model.fit
    fit(frame, response, sample_weight=weights, offsets=offsets, lambdas=lambdas)
    fitted = model._require_fitted()
    assert model.parameter_names_ == baseline.parameter_names
    assert fitted.result.converged == baseline.result.converged
    assert (
        fitted.result.execution_backend_identifier == baseline.result.execution_backend_identifier
    )
    for actual, expected in zip(
        fitted.fit_state.compiled_predictors, baseline.fit_state.compiled_predictors, strict=True
    ):
        np.testing.assert_array_equal(
            actual.compiled.design.toarray(), expected.compiled.design.toarray()
        )
        assert tuple(p.name for p in actual.penalties) == tuple(p.name for p in expected.penalties)
        for a, b in zip(actual.penalties, expected.penalties, strict=True):
            assert a.rank == b.rank
            # Independent penalty factorizations can differ at roundoff. Use
            # the dimension/epsilon budget and matrix norm, including at zeros.
            tolerance = _roundoff_factor(a.omega_ssp, b.omega_ssp)
            scale = max(1.0, np.linalg.norm(b.omega_ssp, ord=np.inf))
            np.testing.assert_allclose(a.omega_ssp, b.omega_ssp, rtol=0.0, atol=tolerance * scale)
    for actual, expected in (
        (model.predict(frame, offsets=offsets), baseline.predict(frame, offsets=offsets)),
        (model.covariance_, baseline.covariance),
    ):
        tolerance = _roundoff_factor(actual, expected)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance * max(1.0, np.linalg.norm(expected, ord=np.inf)),
        )
    restored = SuperLSS.from_bytes(model.to_bytes())
    np.testing.assert_array_equal(
        restored.predict(frame, offsets=offsets), model.predict(frame, offsets=offsets)
    )
    np.testing.assert_array_equal(restored.covariance_, model.covariance_)
    assert restored.predictors[0].interaction_order == ("x:z",)


def test_trusted_032_artifact_preserves_predictions_and_covariance():
    artifact = Path(__file__).with_name("fixtures") / "superlss-v0.32.0.json"
    restored = SuperLSS.from_bytes(artifact.read_bytes())
    frame, response, weights, offsets = _fixture()
    family = GaussianLS(scale_floor=0.02)
    fresh = SuperLSS(family, family.location("x"), family.scale("z")).fit(
        frame,
        response,
        sample_weight=weights,
        offsets=offsets,
        lambdas={},
    )
    # Independent fits can differ by platform-dependent numerical roundoff;
    # exact equality remains required for the same-run round trips above.
    for historical, current in (
        (restored.predict(frame, offsets=offsets), fresh.predict(frame, offsets=offsets)),
        (restored.covariance_, fresh.covariance_),
        (np.asarray(tuple(restored.coef_.values())), np.asarray(tuple(fresh.coef_.values()))),
    ):
        tolerance = _roundoff_factor(historical, current)
        scale = max(1.0, np.linalg.norm(current, ord=np.inf))
        np.testing.assert_allclose(historical, current, rtol=tolerance, atol=tolerance * scale)
