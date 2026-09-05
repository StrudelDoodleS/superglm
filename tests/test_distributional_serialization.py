from __future__ import annotations

import base64
import copyreg
import hashlib
import io
import json
import pickle
from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.efs as efs_module
import superglm.distributional.serialization as serialization_module
import superglm.distributional.smoothing.authority as smoothing_authority
import superglm.distributional.smoothing.loop as smoothing_loop
import superglm.distributional.smoothing.objective as smoothing_objective
from superglm import NegativeBinomial
from superglm.distributional.efs_acceleration import (
    MultisecantDecision,
    MultisecantProposal,
    WindowedTypeIIAnderson,
)
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.predictor import Predictor
from superglm.distributional.result import (
    DenseSolverConfig,
    DistributionalEFSConfig,
    JointEndpointDirectionEvidence,
)
from superglm.distributional.serialization import (
    SCHEMA_VERSION,
    DistributionalSerializationError,
    deserialize_distributional_model,
    distributional_manifest,
    serialize_distributional_model,
)
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.distributional.weights import (
    LegacyPowerWeightArtifactError,
    ResolvedLikelihoodWeights,
    WeightContract,
)
from superglm.features import Numeric, RandomEffect, Spline
from superglm.types import LambdaPolicy


def _data() -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(4301)
    x = np.linspace(-1.0, 1.0, 84)
    d = np.sin(np.linspace(0.0, 3.0 * np.pi, len(x)))
    sigma = 0.18 + np.exp(-1.35 + 0.2 * np.cos(2.0 * np.pi * x))
    response = 0.45 + 0.7 * np.sin(np.pi * x) + 0.15 * d + rng.normal(scale=sigma)
    offsets = {
        "location": np.linspace(-0.06, 0.09, len(x)),
        "scale": 0.04 * np.cos(np.linspace(0.0, 2.0 * np.pi, len(x))),
    }
    return pd.DataFrame({"x": x, "d": d}), response, offsets


def _fixed_model(*, retain_rows: bool = True):
    frame, response, offsets = _data()
    shared = Spline(kind="cr", n_knots=6)
    predictors = (
        Predictor("location", {"x": shared, "d": Numeric()}),
        Predictor("scale", {"x": shared}),
    )
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.04),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        offsets=offsets,
        lambdas={"location:x#wiggle": 0.35, "scale:x#wiggle": 0.8},
        retain_rows=retain_rows,
    )
    return model, frame, offsets, shared


def _gamma_model(*, retain_rows: bool = True):
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 12)})
    response = np.array([0.7, 1.0, 1.4, 2.1, 1.8, 2.6, 3.2, 2.9, 3.8, 4.5, 4.0, 5.1])
    weights = np.array([0.6, 1.1, 1.7, 0.8, 1.4, 2.0, 0.9, 1.3, 2.2, 0.7, 1.6, 1.2])
    model = fit_dense_distributional(
        frame,
        response,
        family=GammaLS(),
        weight_contract=WeightContract("prior"),
        predictors=(Predictor("mean", {"x": Numeric()}), Predictor("scale", {})),
        sample_weight=weights,
        lambdas={},
        retain_rows=retain_rows,
    )
    return model, frame


def _generalized_gamma_model(*, retain_rows: bool = True):
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 12)})
    response = np.array([0.7, 1.0, 1.4, 2.1, 1.8, 2.6, 3.2, 2.9, 3.8, 4.5, 4.0, 5.1])
    model = fit_dense_distributional(
        frame,
        response,
        family=GeneralizedGammaLSS(scale_floor=0.04),
        weight_contract=WeightContract("prior"),
        predictors=(
            Predictor("mean", {"x": Numeric()}),
            Predictor("scale", {}),
            Predictor("shape", {}),
        ),
        lambdas={},
        retain_rows=retain_rows,
    )
    return model, frame


def _negative_binomial_model(*, retain_rows: bool = True):
    rng = np.random.default_rng(2026083115)
    n = 240
    x_mean = rng.permutation(np.linspace(-1.0, 1.0, n))
    x_theta = rng.permutation(np.linspace(-1.0, 1.0, n))
    exposure = np.resize(np.array([0.5, 1.0, 1.5, 2.0]), n)
    mean_offset = 0.08 * np.sin(np.pi * x_mean)
    theta_offset = -0.06 * np.cos(np.pi * x_theta)
    mean = np.exp(0.55 + 0.35 * x_mean + mean_offset)
    theta = np.exp(0.20 - 0.25 * x_theta + theta_offset)
    count = rng.negative_binomial(exposure * theta, theta / (mean + theta)).astype(np.float64)
    frame = pd.DataFrame({"x_mean": x_mean, "x_theta": x_theta})
    offsets = {"mean": mean_offset, "theta": theta_offset}
    model = fit_dense_distributional(
        frame,
        count / exposure,
        family=NegativeBinomialLS(),
        weight_contract=WeightContract("prior"),
        predictors=(
            Predictor("mean", {"x_mean": Numeric()}),
            Predictor("theta", {"x_theta": Numeric()}),
        ),
        sample_weight=exposure,
        offsets=offsets,
        lambdas={},
        config=DenseSolverConfig(
            coefficient_curvature="observed",
            tolerance=1.0e-8,
            max_iterations=100,
        ),
        retain_rows=retain_rows,
    )
    return model, frame, offsets


def _gamma_artifact_model(
    *,
    semantics: str,
    fit_mode: str,
    retain_rows: bool,
):
    x = np.linspace(-1.0, 1.0, 36)
    frame = pd.DataFrame({"x": x})
    mean = np.exp(0.6 + 0.35 * x)
    scale = np.exp(-0.9 + 0.2 * np.cos(np.pi * x))
    response = np.random.default_rng(20260828).gamma(
        shape=1.0 / scale**2,
        scale=mean * scale**2,
    )
    weights = (
        np.linspace(0.55, 1.85, len(frame))
        if semantics == "prior"
        else np.tile(np.array([1.0, 3.0, 2.0, 4.0]), len(frame) // 4)
    )
    predictors = (
        Predictor("mean", {"x": Numeric()}),
        Predictor("scale", {"x": Spline(kind="cr", n_knots=4)}),
    )
    kwargs = (
        {"lambdas": {"scale:x#wiggle": 0.4}}
        if fit_mode == "fixed"
        else {
            "lambdas": {"scale:x#wiggle": 0.4},
            "efs_config": DistributionalEFSConfig(outer="efs", max_iterations=2, tolerance=10.0),
        }
    )
    model = fit_dense_distributional(
        frame,
        response,
        family=GammaLS(),
        weight_contract=WeightContract(semantics),  # type: ignore[arg-type]
        predictors=predictors,
        sample_weight=weights,
        retain_rows=retain_rows,
        **kwargs,
    )
    return model, frame, response, weights


def _tweedie_contract_artifact_model(*, semantics: str, retain_rows: bool):
    n = 120
    rng = np.random.default_rng(2026082804)
    x = rng.permutation(np.linspace(-1.0, 1.0, n))
    frame = pd.DataFrame({"x": x})
    mean = np.exp(0.55 + 0.45 * np.sin(np.pi * x) + 0.15 * x)
    power = 1.5
    r = power - 1.0
    s = 2.0 - power
    counts = rng.poisson(mean**s / s)
    response = np.zeros(n, dtype=np.float64)
    positive = counts > 0
    response[positive] = rng.gamma(
        counts[positive] * s / r,
        r * mean[positive] ** r,
    )
    weights = (
        np.linspace(0.7, 1.5, n)
        if semantics == "prior"
        else np.resize(np.array([1, 2, 3], dtype=np.int64), n)
    )
    model = fit_dense_distributional(
        frame,
        response,
        family=TweedieLSS(power_lower=1.08, power_upper=1.92),
        predictors=(
            Predictor(
                "mean",
                {
                    "x": Spline(
                        kind="cr",
                        n_knots=4,
                        lambda_policy=LambdaPolicy.estimate(),
                    )
                },
            ),
            Predictor("dispersion", {}),
            Predictor("power", {}),
        ),
        weight_contract=WeightContract(semantics),  # type: ignore[arg-type]
        sample_weight=weights,
        config=DenseSolverConfig(coefficient_curvature="observed"),
        # This loose configured fit isolates the persistence contract; it is
        # not optimizer-accuracy evidence.
        efs_config=DistributionalEFSConfig(max_iterations=8, tolerance=0.5, outer="efs"),
        retain_rows=retain_rows,
    )
    return model, frame


def _current_major() -> int:
    return int(SCHEMA_VERSION.split(".", 1)[0])


def _walk_object_graph(root: object) -> Iterator[object]:
    pending = [root]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        yield value
        if isinstance(value, np.ndarray | str | bytes | bytearray | memoryview):
            continue
        if isinstance(value, Mapping):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, tuple | list | set | frozenset):
            pending.extend(value)
        elif hasattr(value, "__dict__"):
            pending.extend(vars(value).values())


#: Development-only schema shapes used to verify typed refusal at the first
#: public read boundary. Anchored as literals rather than derived from
#: ``SCHEMA_VERSION`` so the tests cannot silently follow the current version.
_SCHEMA_BEFORE_PENALTY_GEOMETRY = "1.0.0"
_SCHEMA_BEFORE_WEIGHT_CONTRACT = "2.0.0"
_SCHEMA_BEFORE_ACCELERATION_TELEMETRY = "3.0.0"
_SCHEMA_BEFORE_TERMINAL_RAW_EVIDENCE = "4.0.0"
_SCHEMA_BEFORE_JOINT_ENDPOINT_EVIDENCE = "5.0.0"

_LIKELIHOOD_DECOMPOSITION_FIELDS = (
    "optimizing_log_likelihood",
    "parameter_independent_carrier",
    "log_likelihood",
    "penalized_optimizing_log_likelihood",
    "penalized_log_likelihood",
    "initial_penalized_optimizing_log_likelihood",
    "initial_penalized_log_likelihood",
)


def _rehash_pickled_artifact(artifact: dict, mutate) -> bytes:
    """Apply a trusted-payload mutation and recompute only its transport digest."""

    payload = artifact["payload"]
    restored = pickle.loads(base64.b64decode(payload["data"]))
    mutate(restored)
    raw = serialization_module._pickle_model(restored)
    payload["data"] = base64.b64encode(raw).decode("ascii")
    payload["sha256"] = hashlib.sha256(raw).hexdigest()
    return json.dumps(artifact).encode()


def _with_unpickle_sentinel(artifact: dict) -> dict:
    raw = b"digest-valid bytes that must not reach pickle.loads"
    artifact["payload"]["data"] = base64.b64encode(raw).decode("ascii")
    artifact["payload"]["sha256"] = hashlib.sha256(raw).hexdigest()
    return artifact


def _independent_array_descriptor(values: np.ndarray) -> dict[str, object]:
    """Describe an array without using the serializer's manifest helper."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": digest.hexdigest()}


@pytest.fixture(scope="module")
def _decomposition_models():
    fixed = _fixed_model()[0]
    frame, response, offsets = _data()
    selected = Spline(
        kind="cr",
        n_knots=5,
        lambda_policy=LambdaPolicy.estimate(),
    )
    efs = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.015),
        weight_contract=WeightContract(semantics="prior"),
        predictors=(
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {"x": selected}),
        ),
        offsets=offsets,
        lambdas={"scale:x#wiggle": 0.3},
        efs_config=DistributionalEFSConfig(outer="efs", max_iterations=2, tolerance=10.0),
    )
    assert efs.smoothing is not None
    return fixed, efs


@pytest.fixture(scope="module")
def _exact_face_model():
    groups = np.tile(np.array(["a", "b", "c"]), 8)
    response = np.repeat(np.array([0.7, 0.9, 1.1, 1.3, 0.8, 1.2, 1.0, 1.0]), 3)
    model = fit_dense_distributional(
        pd.DataFrame({"group": groups}),
        response,
        family=GaussianLS(),
        weight_contract=WeightContract("prior"),
        predictors=(
            Predictor("location", {"group": RandomEffect()}),
            Predictor("scale", {}),
        ),
        lambdas={"location:group#wiggle": 10.0},
        config=DenseSolverConfig(tolerance=1.0e-10, max_iterations=150),
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=8,
            tolerance=1.0e-8,
            maximum_lambda=10.0,
        ),
        retain_rows=True,
    )
    assert model.smoothing is not None
    assert model.smoothing.matched_certified is False
    assert model.smoothing.terminal_fit.coefficient_face is not None
    return model


@pytest.fixture(scope="module")
def _joint_exact_face_model():
    levels = np.array([-100.0, 0.0, 100.0])
    x = np.repeat(levels, 6)
    z = np.tile(np.repeat(levels, 2), 3)
    sign = np.tile(np.array([-1.0, 1.0]), 9)
    frame = pd.DataFrame({"x": x, "z": z})
    response = 0.4 + 0.006 * x + np.exp(-1.2 + 0.003 * z) * sign
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.0),
        weight_contract=WeightContract("prior"),
        predictors=(
            Predictor("location", {"x": Spline(kind="cr", k=3)}),
            Predictor("scale", {"z": Spline(kind="cr", k=3)}),
        ),
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.3},
        config=DenseSolverConfig(tolerance=1.0e-9, max_iterations=200),
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=60,
            tolerance=1.0e-8,
        ),
        retain_rows=True,
    )
    smoothing = model.smoothing
    assert smoothing is not None
    assert smoothing.converged is True
    face = smoothing.terminal_fit.coefficient_face
    assert face is not None
    assert face.component_names == (
        "location:x#wiggle",
        "scale:z#wiggle",
    )
    activation = next(item for item in smoothing.history if item.activated_face_components)
    assert activation.activated_face_components == face.component_names
    assert len(activation.coefficient_fit_indices) == 1
    assert activation.accepted_fit_index == activation.coefficient_fit_indices[0]
    assert isinstance(activation.endpoint_direction_evidence, JointEndpointDirectionEvidence)
    final = smoothing.history[-1]
    assert final.revalidated_face_components == face.component_names
    assert isinstance(final.endpoint_direction_evidence, JointEndpointDirectionEvidence)
    return model, frame


def _as_schema_1_0_0(artifact: dict) -> dict:
    """Rewrite a current artifact into what a schema-1.0.0 writer produced.

    1.0.0 had no ``penalty_kind`` / ``repeat_count`` / ``block_width`` in its
    penalty components.  Those three keys are the whole of the change, and are
    why such an artifact cannot be read by a build that recomputes the manifest
    and compares it for equality.
    """
    artifact["schema_version"] = _SCHEMA_BEFORE_PENALTY_GEOMETRY
    for component in artifact["manifest"]["penalties"]["components"]:
        for key in ("penalty_kind", "repeat_count", "block_width"):
            component.pop(key, None)
    return artifact


def _assert_round_trip(model, restored, frame, offsets) -> None:
    np.testing.assert_array_equal(
        restored.predict_eta(frame, offsets=offsets),
        model.predict_eta(frame, offsets=offsets),
    )
    np.testing.assert_array_equal(
        restored.predict_parameters(frame, offsets=offsets),
        model.predict_parameters(frame, offsets=offsets),
    )
    np.testing.assert_array_equal(
        restored.predict(frame, offsets=offsets),
        model.predict(frame, offsets=offsets),
    )
    np.testing.assert_array_equal(restored.coefficients, model.coefficients)
    np.testing.assert_array_equal(restored.covariance, model.covariance)
    assert restored.fitted_result.total_effective_df == model.fitted_result.total_effective_df
    assert restored.fitted_result.predictor_edf == model.fitted_result.predictor_edf
    assert restored.fitted_result.intercept_edf == model.fitted_result.intercept_edf
    assert restored.fitted_result.term_edf == model.fitted_result.term_edf
    assert restored.smoothing_parameters == model.smoothing_parameters
    assert restored.telemetry.to_dict() == model.telemetry.to_dict()
    np.testing.assert_array_equal(
        restored.null_model.coefficients,
        model.null_model.coefficients,
    )
    assert dict(restored.null_model.family_config) == dict(model.null_model.family_config)
    assert restored.null_model.parameter_names == model.null_model.parameter_names
    assert dict(restored.null_model.link_types) == dict(model.null_model.link_types)
    assert dict(restored.null_model.offset_semantics) == dict(model.null_model.offset_semantics)
    assert restored.null_model.objective == model.null_model.objective
    assert restored.null_model.log_likelihood == model.null_model.log_likelihood
    assert restored.null_model.curvature_telemetry.to_dict() == (
        model.null_model.curvature_telemetry.to_dict()
    )
    assert restored.fit_state.weight_contract == model.fit_state.weight_contract
    assert restored.fit_state.weight_provenance == model.fit_state.weight_provenance
    assert restored.fit_state.family_likelihood_plan_identifier == (
        model.fit_state.family_likelihood_plan_identifier
    )
    assert restored.result.family_likelihood_plan_identifier == (
        restored.fit_state.family_likelihood_plan_identifier
    )
    assert restored.null_model.family_likelihood_plan_identifier == (
        restored.fit_state.family_likelihood_plan_identifier
    )
    assert restored.fit_state.requested_discrete is model.fit_state.requested_discrete
    assert restored.fit_state.requested_n_bins == model.fit_state.requested_n_bins
    assert restored.fit_state.requested_chunk_size == model.fit_state.requested_chunk_size


def _expected_family_manifest(
    python_type: str,
    config: dict[str, object],
    *,
    expected_information: bool,
    distribution_functions: bool,
) -> dict[str, object]:
    return {
        "python_type": python_type,
        "config": config,
        "default_prediction_name": "conditional_mean",
        "capabilities": {
            "max_derivative_order": 2,
            "expected_information": expected_information,
            "cdf": distribution_functions,
            "quantile": distribution_functions,
            "random": False,
            "response_mean": True,
            "censored_response": False,
        },
    }


def _protocol_five_mapping_proxy_reduction(value: Mapping) -> tuple[object, tuple[dict]]:
    return serialization_module._restore_mapping_proxy, (dict(value),)


class _ProtocolFiveDistributionalPickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[type(MappingProxyType({}))] = _protocol_five_mapping_proxy_reduction

    def __init__(self, stream: io.BytesIO) -> None:
        super().__init__(stream, protocol=5)


def _assert_canonical_builtin_bytes(
    model,
    serialized: bytes,
    family_manifest: dict[str, object],
) -> None:
    """Pin the existing envelope bytes independently of the serializer entry point."""
    stream = io.BytesIO()
    _ProtocolFiveDistributionalPickler(stream).dump(model)
    payload = stream.getvalue()
    artifact = json.loads(serialized)
    assert artifact["manifest"]["family"] == family_manifest
    expected = {
        "artifact_type": "superglm.DenseDistributionalModel",
        "manifest": artifact["manifest"],
        "payload": {
            "data": base64.b64encode(payload).decode("ascii"),
            "encoding": "python-pickle-v5-base64",
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "schema_version": SCHEMA_VERSION,
    }
    assert artifact == expected
    assert (
        serialized
        == json.dumps(
            expected,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )


class _GaussianPersistenceSubclass(GaussianLS):
    def to_config(self) -> dict[str, object]:
        return {"type": "GaussianLS", "scale_floor": self.scale_floor}


def test_fixed_lambda_round_trip_preserves_compiled_ownership_and_complete_state() -> None:
    model, frame, offsets, caller_owned = _fixed_model()
    manifest = distributional_manifest(model)
    restored = deserialize_distributional_model(serialize_distributional_model(model))

    assert manifest["family"]["config"] == {"type": "GaussianLS", "scale_floor": 0.04}
    assert manifest["parameter_order"] == ["location", "scale"]
    assert manifest["layout"]["coefficient_names"] == list(model.layout.coefficient_names)
    assert manifest["fit"]["smoothing_parameters"] == dict(model.lambdas)
    assert manifest["fit"]["covariance"] == model.covariance.tolist()
    assert set(manifest["penalties"]) == {"components", "lambdas"}
    assert manifest["null_model"]["weight_semantics"] == "prior"
    assert restored.family.scale_floor == 0.04
    assert restored.compiled_predictors[0].compiled.specs["x"] is not caller_owned
    assert restored.compiled_predictors[1].compiled.specs["x"] is not caller_owned
    assert (
        restored.compiled_predictors[0].compiled.specs["x"]
        is not restored.compiled_predictors[1].compiled.specs["x"]
    )
    _assert_round_trip(model, restored, frame, offsets)


def test_first_public_efs_manifest_records_the_complete_current_schema() -> None:
    model, _, _, _ = _gamma_artifact_model(
        semantics="prior",
        fit_mode="efs",
        retain_rows=True,
    )
    assert model.smoothing is not None
    manifest = distributional_manifest(model)["smoothing"]
    assert manifest is not None

    config_fields = {field.name for field in fields(model.smoothing.config)}
    assert config_fields <= manifest["config"].keys()

    current_iteration_fields = {
        "stage",
        "step_source",
        "gradient",
        "gradient_certificate",
        "hessian_certificate",
        "projected_gradient_norm",
        "newton_ridge",
    }
    assert all(current_iteration_fields <= iteration.keys() for iteration in manifest["history"])


@pytest.mark.parametrize(
    ("model_factory", "python_type", "config"),
    [
        (
            lambda: _fixed_model()[0],
            "superglm.distributional.families.gaussian.GaussianLS",
            {"type": "GaussianLS", "scale_floor": 0.04},
        ),
        (
            lambda: _gamma_model()[0],
            "superglm.distributional.families.gamma.GammaLS",
            {"type": "GammaLS", "parameterization": "mean_cv"},
        ),
    ],
    ids=["GaussianLS", "GammaLS"],
)
def test_deserializer_admits_only_the_pinned_builtin_family_manifest_pairs(
    model_factory,
    python_type: str,
    config: dict[str, object],
) -> None:
    model = model_factory()
    serialized = serialize_distributional_model(model)
    artifact = json.loads(serialized)

    assert artifact["manifest"]["family"]["python_type"] == python_type
    assert artifact["manifest"]["family"]["config"] == config
    expected_family = _expected_family_manifest(
        python_type,
        config,
        expected_information=True,
        distribution_functions=True,
    )
    _assert_canonical_builtin_bytes(
        model,
        serialized,
        expected_family,
    )
    restored = deserialize_distributional_model(json.dumps(artifact))
    assert type(restored.family) is type(model.family)
    assert restored.family.to_config() == config


def test_unregistered_family_refuses_before_metadata_or_pickle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = replace(
        _fixed_model(retain_rows=False)[0],
        family=_GaussianPersistenceSubclass(scale_floor=0.04),
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("unregistered family reached metadata or pickle")

    monkeypatch.setattr(_GaussianPersistenceSubclass, "to_config", unexpected_call)
    monkeypatch.setattr(serialization_module, "_pickle_model", unexpected_call)

    with pytest.raises(DistributionalSerializationError, match="built-in"):
        serialize_distributional_model(model)


def test_builtin_family_config_type_is_validated_before_pickle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _fixed_model(retain_rows=False)[0]

    def wrong_config(family: GaussianLS) -> dict[str, object]:
        return {"type": "GammaLS", "scale_floor": family.scale_floor}

    def unexpected_pickle(*args, **kwargs):
        raise AssertionError("mismatched built-in config reached pickle")

    monkeypatch.setattr(GaussianLS, "to_config", wrong_config)
    monkeypatch.setattr(serialization_module, "_pickle_model", unexpected_pickle)

    with pytest.raises(DistributionalSerializationError, match="family configuration"):
        serialize_distributional_model(model)


@pytest.fixture(scope="module")
def negative_binomial_artifact_model():
    return _negative_binomial_model()


def test_negative_binomial_manifest_pair_and_round_trip_preserve_complete_state(
    negative_binomial_artifact_model,
) -> None:
    """Kills a non-exact NB2 family pair or a partial fitted-state restore."""
    model, frame, offsets = negative_binomial_artifact_model
    serialized = serialize_distributional_model(model)
    manifest = json.loads(serialized)["manifest"]
    restored = deserialize_distributional_model(serialized)

    assert manifest["family"]["python_type"] == (
        "superglm.distributional.families.negative_binomial.NegativeBinomialLS"
    )
    assert manifest["family"]["config"] == {
        "type": "NegativeBinomialLS",
        "parameterization": "nb2_mean_theta",
    }
    _assert_canonical_builtin_bytes(
        model,
        serialized,
        _expected_family_manifest(
            "superglm.distributional.families.negative_binomial.NegativeBinomialLS",
            {"type": "NegativeBinomialLS", "parameterization": "nb2_mean_theta"},
            expected_information=False,
            distribution_functions=False,
        ),
    )
    assert type(restored.family) is NegativeBinomialLS
    assert restored.parameter_names == ("mean", "theta")
    _assert_round_trip(model, restored, frame, offsets)
    source_rows = model.fit_state.retained_rows
    target_rows = restored.fit_state.retained_rows
    assert source_rows is not None and target_rows is not None
    for name in ("mean", "theta"):
        np.testing.assert_array_equal(target_rows.offsets[name], source_rows.offsets[name])
    np.testing.assert_array_equal(target_rows.fitted_parameters, source_rows.fitted_parameters)


@pytest.fixture(scope="module")
def negative_binomial_artifact_bytes(negative_binomial_artifact_model) -> bytes:
    model, _, _ = negative_binomial_artifact_model
    return serialize_distributional_model(model)


@pytest.mark.parametrize(
    ("target", "field", "operation", "value", "message"),
    [
        ("family", "python_type", "pop", None, "NegativeBinomialLS.*exact pair"),
        (
            "family",
            "python_type",
            "set",
            "superglm.distributions.NegativeBinomial",
            "NegativeBinomialLS.*exact pair",
        ),
        ("config", "type", "pop", None, "NegativeBinomialLS.*exact pair"),
        ("config", "type", "set", "NegativeBinomial", "NegativeBinomialLS.*exact pair"),
        (
            "config",
            "parameterization",
            "pop",
            None,
            "NegativeBinomialLS.*configuration",
        ),
        (
            "config",
            "parameterization",
            "set",
            "mean_dispersion",
            "NegativeBinomialLS.*configuration",
        ),
        ("config", "extra", "set", True, "NegativeBinomialLS.*configuration"),
    ],
)
def test_negative_binomial_manifest_refuses_tampering_before_payload_unpickling(
    negative_binomial_artifact_bytes: bytes,
    target: str,
    field: str,
    operation: str,
    value: object,
    message: str,
) -> None:
    artifact = _with_unpickle_sentinel(json.loads(negative_binomial_artifact_bytes))
    family = artifact["manifest"]["family"]
    mutated = family if target == "family" else family["config"]
    if operation == "pop":
        mutated.pop(field)
    else:
        mutated[field] = value

    with pytest.raises(DistributionalSerializationError, match=message):
        deserialize_distributional_model(json.dumps(artifact))


def test_scalar_negative_binomial_pickle_persistence_remains_unchanged() -> None:
    """Guards the scalar NB2 family from the new distributional manifest entry."""
    restored = pickle.loads(pickle.dumps(NegativeBinomial(theta=3.5)))
    mean = np.array([0.5, 2.0, 7.0])

    assert type(restored) is NegativeBinomial
    assert restored.theta == 3.5
    np.testing.assert_allclose(
        restored.variance(mean),
        np.array([0.5714285714285714, 3.142857142857143, 21.0]),
        rtol=0.0,
        atol=np.finfo(np.float64).eps,
    )


@pytest.mark.parametrize(
    ("semantics", "retain_rows"),
    [("prior", True), ("frequency", False)],
)
def test_tweedie_configured_contract_artifact_preserves_observed_fitted_state(
    semantics: str,
    retain_rows: bool,
) -> None:
    model, frame = _tweedie_contract_artifact_model(
        semantics=semantics,
        retain_rows=retain_rows,
    )
    serialized = serialize_distributional_model(model)
    manifest = json.loads(serialized)["manifest"]
    restored = deserialize_distributional_model(serialized)

    assert manifest["family"]["python_type"] == (
        "superglm.distributional.families.tweedie.TweedieLSS"
    )
    assert manifest["family"]["config"] == {
        "type": "TweedieLSS",
        "power_lower": 1.08,
        "power_upper": 1.92,
    }
    _assert_canonical_builtin_bytes(
        model,
        serialized,
        _expected_family_manifest(
            "superglm.distributional.families.tweedie.TweedieLSS",
            {"type": "TweedieLSS", "power_lower": 1.08, "power_upper": 1.92},
            expected_information=False,
            distribution_functions=True,
        ),
    )
    assert type(restored.family) is TweedieLSS
    assert restored.family.to_config() == model.family.to_config()
    power = model.predict_parameters(frame)[:, 2]
    wall_resolution = np.sqrt(np.finfo(np.float64).eps) * (
        model.family.power_upper - model.family.power_lower
    )
    assert np.all(power - model.family.power_lower > wall_resolution)
    assert np.all(model.family.power_upper - power > wall_resolution)
    _assert_round_trip(model, restored, frame, None)
    assert restored.smoothing is not None and model.smoothing is not None
    assert restored.smoothing.config == model.smoothing.config
    assert all(
        fit.config.coefficient_curvature == "observed"
        and fit.terminal_curvature.requested_source == "observed"
        and fit.terminal_curvature.actual_source == "observed"
        and fit.terminal_curvature.fallback_count == 0
        for fit in restored.smoothing.coefficient_fits
    )
    assert (restored.fit_state.retained_rows is not None) is retain_rows


@pytest.fixture(scope="module")
def tweedie_contract_artifact_bytes() -> bytes:
    model, _ = _tweedie_contract_artifact_model(semantics="prior", retain_rows=False)
    return serialize_distributional_model(model)


@pytest.mark.parametrize(
    ("target", "field", "operation", "value", "message"),
    [
        ("config", "power_lower", "pop", None, "TweedieLSS.*configuration"),
        ("config", "power_upper", "pop", None, "TweedieLSS.*configuration"),
        ("config", "type", "pop", None, "TweedieLSS.*pair"),
        ("config", "extra", "set", 0, "TweedieLSS.*configuration"),
        ("config", "power_lower", "set", True, "TweedieLSS.*power walls"),
        ("config", "power_upper", "set", "1.92", "TweedieLSS.*power walls"),
        ("config", "power_lower", "set", float("nan"), "non-finite JSON"),
        ("config", "power_upper", "set", float("inf"), "non-finite JSON"),
        ("config", "power_lower", "set", 1.0, "TweedieLSS.*power walls"),
        ("config", "power_upper", "set", 2.0, "TweedieLSS.*power walls"),
        ("config", "power_lower", "set", 1.92, "TweedieLSS.*power walls"),
        ("family", "python_type", "set", "tests.lookalikes.TweedieLSS", "TweedieLSS.*pair"),
        ("config", "type", "set", "GammaLS", "TweedieLSS.*pair"),
    ],
)
def test_tweedie_manifest_refusals_happen_before_payload_unpickling(
    tweedie_contract_artifact_bytes: bytes,
    target: str,
    field: str,
    operation: str,
    value: object,
    message: str,
) -> None:
    artifact = _with_unpickle_sentinel(json.loads(tweedie_contract_artifact_bytes))
    family = artifact["manifest"]["family"]
    mutated = family if target == "family" else family["config"]
    if operation == "pop":
        mutated.pop(field)
    else:
        mutated[field] = value

    with pytest.raises(DistributionalSerializationError, match=message):
        deserialize_distributional_model(json.dumps(artifact))


@pytest.mark.parametrize(
    ("model_factory", "replacement"),
    [
        (lambda: _fixed_model()[0], None),
        (lambda: _fixed_model()[0], 7),
        (lambda: _fixed_model()[0], "tests.lookalikes.GaussianLS"),
        (lambda: _gamma_model()[0], None),
        (lambda: _gamma_model()[0], 7),
        (lambda: _gamma_model()[0], "tests.lookalikes.GammaLS"),
    ],
    ids=[
        "gaussian-missing",
        "gaussian-non-string",
        "gaussian-lookalike",
        "gamma-missing",
        "gamma-non-string",
        "gamma-lookalike",
    ],
)
def test_qualified_family_type_is_exact_before_payload_unpickling(
    model_factory,
    replacement: object,
) -> None:
    artifact = _with_unpickle_sentinel(json.loads(serialize_distributional_model(model_factory())))
    family = artifact["manifest"]["family"]
    if replacement is None:
        family.pop("python_type")
    else:
        family["python_type"] = replacement

    with pytest.raises(DistributionalSerializationError, match="family configuration"):
        deserialize_distributional_model(json.dumps(artifact))


@pytest.mark.parametrize(
    ("model_factory", "wrong_type"),
    [
        (lambda: _fixed_model()[0], "GammaLS"),
        (lambda: _gamma_model()[0], "GaussianLS"),
    ],
    ids=["gaussian-qualified", "gamma-qualified"],
)
def test_qualified_type_and_config_type_must_be_the_exact_pair_before_unpickling(
    model_factory,
    wrong_type: str,
) -> None:
    artifact = _with_unpickle_sentinel(json.loads(serialize_distributional_model(model_factory())))
    artifact["manifest"]["family"]["config"]["type"] = wrong_type

    with pytest.raises(DistributionalSerializationError, match="family configuration"):
        deserialize_distributional_model(json.dumps(artifact))


@pytest.mark.parametrize("field", ["python_type", "config_type"])
def test_unhashable_manifest_pair_refuses_before_decode_or_unpickle(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = json.loads(serialize_distributional_model(_fixed_model()[0]))
    family = artifact["manifest"]["family"]
    if field == "python_type":
        family["python_type"] = []
    else:
        family["config"]["type"] = []

    def unexpected_call(*args, **kwargs):
        raise AssertionError("invalid family pair reached payload decoding")

    monkeypatch.setattr(serialization_module, "_decode_payload", unexpected_call)
    monkeypatch.setattr(serialization_module.pickle, "loads", unexpected_call)

    with pytest.raises(DistributionalSerializationError, match="family configuration"):
        deserialize_distributional_model(json.dumps(artifact))


@pytest.mark.parametrize(
    ("model_factory", "field", "add_extra"),
    [
        (lambda: _fixed_model()[0], "scale_floor", False),
        (lambda: _fixed_model()[0], "extra", True),
        (lambda: _gamma_model()[0], "parameterization", False),
        (lambda: _gamma_model()[0], "extra", True),
        (lambda: _generalized_gamma_model()[0], "parametrisation", False),
        (lambda: _generalized_gamma_model()[0], "scale_floor", False),
        (lambda: _generalized_gamma_model()[0], "extra", True),
    ],
    ids=[
        "gaussian-missing",
        "gaussian-extra",
        "gamma-missing",
        "gamma-extra",
        "gengamma-missing-parametrisation",
        "gengamma-missing-scale-floor",
        "gengamma-extra",
    ],
)
def test_family_config_key_set_is_exact_before_payload_unpickling(
    model_factory,
    field: str,
    add_extra: bool,
) -> None:
    artifact = _with_unpickle_sentinel(json.loads(serialize_distributional_model(model_factory())))
    config = artifact["manifest"]["family"]["config"]
    if add_extra:
        config[field] = True
    else:
        config.pop(field)

    with pytest.raises(
        DistributionalSerializationError,
        match="family configuration|scale_floor",
    ):
        deserialize_distributional_model(json.dumps(artifact))


@pytest.mark.parametrize(
    "parameterization",
    [1, "shape_scale"],
    ids=["non-string", "altered"],
)
def test_gamma_parameterization_is_exact_before_payload_unpickling(
    parameterization: object,
) -> None:
    artifact = _with_unpickle_sentinel(
        json.loads(serialize_distributional_model(_gamma_model()[0]))
    )
    artifact["manifest"]["family"]["config"]["parameterization"] = parameterization

    with pytest.raises(DistributionalSerializationError, match="family configuration"):
        deserialize_distributional_model(json.dumps(artifact))


@pytest.mark.parametrize(
    "scale_floor",
    [True, "0.04", -0.04, 10**400],
    ids=["bool", "numeric-string", "negative", "oversized-integer"],
)
def test_gaussian_scale_floor_is_a_finite_nonnegative_json_number_before_unpickling(
    scale_floor: object,
) -> None:
    artifact = _with_unpickle_sentinel(
        json.loads(serialize_distributional_model(_fixed_model()[0]))
    )
    artifact["manifest"]["family"]["config"]["scale_floor"] = scale_floor

    with pytest.raises(DistributionalSerializationError, match="scale_floor"):
        deserialize_distributional_model(json.dumps(artifact))


@pytest.mark.parametrize(
    "parametrisation",
    [1, "shape_scale", None],
    ids=["non-string", "altered", "null"],
)
def test_generalized_gamma_parametrisation_is_exact_before_payload_unpickling(
    parametrisation: object,
) -> None:
    artifact = _with_unpickle_sentinel(
        json.loads(serialize_distributional_model(_generalized_gamma_model()[0]))
    )
    artifact["manifest"]["family"]["config"]["parametrisation"] = parametrisation

    with pytest.raises(DistributionalSerializationError, match="parametrisation"):
        deserialize_distributional_model(json.dumps(artifact))


@pytest.mark.parametrize(
    "scale_floor",
    [True, "0.04", -0.04, 10**400],
    ids=["bool", "numeric-string", "negative", "oversized-integer"],
)
def test_generalized_gamma_scale_floor_is_a_finite_nonnegative_json_number_before_unpickling(
    scale_floor: object,
) -> None:
    artifact = _with_unpickle_sentinel(
        json.loads(serialize_distributional_model(_generalized_gamma_model()[0]))
    )
    artifact["manifest"]["family"]["config"]["scale_floor"] = scale_floor

    with pytest.raises(DistributionalSerializationError, match="scale_floor"):
        deserialize_distributional_model(json.dumps(artifact))


def test_gaussian_scale_floor_rejects_a_json_number_that_overflows_to_infinity() -> None:
    artifact = _with_unpickle_sentinel(
        json.loads(serialize_distributional_model(_fixed_model()[0]))
    )
    artifact["manifest"]["family"]["config"]["scale_floor"] = "overflow-placeholder"
    serialized = json.dumps(artifact).replace('"overflow-placeholder"', "1e309", 1)

    with pytest.raises(DistributionalSerializationError, match="scale_floor"):
        deserialize_distributional_model(serialized)


@pytest.mark.parametrize(
    ("manifest_factory", "payload_factory"),
    [
        (
            lambda: _fixed_model(retain_rows=False)[0],
            lambda: _gamma_model(retain_rows=False)[0],
        ),
        (
            lambda: _gamma_model(retain_rows=False)[0],
            lambda: _fixed_model(retain_rows=False)[0],
        ),
    ],
    ids=["gaussian-manifest-gamma-payload", "gamma-manifest-gaussian-payload"],
)
def test_digest_valid_cross_family_payload_is_rejected_after_the_pair_gate(
    manifest_factory,
    payload_factory,
) -> None:
    artifact = json.loads(serialize_distributional_model(manifest_factory()))
    payload_artifact = json.loads(serialize_distributional_model(payload_factory()))
    artifact["payload"] = payload_artifact["payload"]

    with pytest.raises(
        DistributionalSerializationError,
        match="payload family class does not match family manifest",
    ):
        deserialize_distributional_model(json.dumps(artifact))


def test_restored_builtin_subclass_is_rejected_by_exact_class_after_unpickling() -> None:
    model = _fixed_model(retain_rows=False)[0]
    artifact = json.loads(serialize_distributional_model(model))

    def replace_family(restored) -> None:
        object.__setattr__(
            restored,
            "family",
            _GaussianPersistenceSubclass(scale_floor=model.family.scale_floor),
        )

    forged = _rehash_pickled_artifact(artifact, replace_family)
    with pytest.raises(
        DistributionalSerializationError,
        match="payload family class does not match family manifest",
    ):
        deserialize_distributional_model(forged)


@pytest.mark.parametrize("retain_rows", [False, True], ids=["compact", "retained"])
@pytest.mark.parametrize("fit_mode", ["fixed", "efs"])
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_gamma_artifact_matrix_preserves_exact_state_and_provenance(
    semantics: str,
    fit_mode: str,
    retain_rows: bool,
) -> None:
    model, frame, response, weights = _gamma_artifact_model(
        semantics=semantics,
        fit_mode=fit_mode,
        retain_rows=retain_rows,
    )
    serialized = serialize_distributional_model(model)
    restored = deserialize_distributional_model(serialized)
    source_state = model.fit_state
    restored_state = restored.fit_state

    assert type(restored.family) is GammaLS
    assert restored.family.to_config() == {"type": "GammaLS", "parameterization": "mean_cv"}
    assert restored.parameter_names == model.parameter_names == ("mean", "scale")
    assert tuple(predictor.name for predictor in restored_state.predictor_templates) == (
        "mean",
        "scale",
    )
    assert tuple(type(state.link) for state in restored.layout.predictors) == tuple(
        type(state.link) for state in model.layout.predictors
    )
    _assert_round_trip(model, restored, frame, None)
    np.testing.assert_array_equal(restored.coefficients, model.coefficients)
    np.testing.assert_array_equal(restored.covariance, model.covariance)
    assert restored.result.terminal_rank.rank == model.result.terminal_rank.rank
    assert restored.inference.covariance_curvature_source == (
        model.inference.covariance_curvature_source
    )
    assert restored.inference.edf_curvature_source == model.inference.edf_curvature_source
    assert restored.inference.reconciliation_tolerance == model.inference.reconciliation_tolerance
    assert restored_state.weight_contract == WeightContract(semantics)  # type: ignore[arg-type]
    assert restored_state.weight_provenance == source_state.weight_provenance
    plan_identifier = source_state.family_likelihood_plan_identifier
    assert restored_state.family_likelihood_plan_identifier == plan_identifier
    assert restored.result.family_likelihood_plan_identifier == plan_identifier
    assert restored.null_model.family_likelihood_plan_identifier == plan_identifier

    if fit_mode == "fixed":
        assert restored.smoothing is model.smoothing is None
        assert dict(restored.lambdas) == {"scale:x#wiggle": 0.4}
    else:
        source_smoothing = model.smoothing
        restored_smoothing = restored.smoothing
        assert source_smoothing is not None and restored_smoothing is not None
        assert dict(restored_smoothing.initial_lambdas) == dict(source_smoothing.initial_lambdas)
        assert dict(restored_smoothing.lambdas) == dict(source_smoothing.lambdas)
        assert restored_smoothing.terminal_fit_index == source_smoothing.terminal_fit_index
        assert restored_smoothing.matched_certified is source_smoothing.matched_certified
        assert restored_smoothing.objective == source_smoothing.objective
        assert [
            fit.terminal_curvature.to_dict() for fit in restored_smoothing.coefficient_fits
        ] == [fit.terminal_curvature.to_dict() for fit in source_smoothing.coefficient_fits]
        assert all(
            fit.family_likelihood_plan_identifier == plan_identifier
            for fit in restored_smoothing.coefficient_fits
        )

    rows = restored_state.retained_rows
    if retain_rows:
        assert rows is not None
        np.testing.assert_array_equal(rows.response, response)
        np.testing.assert_array_equal(rows.likelihood_weights.values, weights)
        assert rows.likelihood_weights.provenance is restored_state.weight_provenance
        rebound = restored.family.bind_likelihood(
            rows.response,
            rows.likelihood_weights,
            COMPLETE_OBSERVATION,
        )
        assert rebound.plan_identifier == plan_identifier
    else:
        assert rows is None
        artifact = json.loads(serialized)
        decoded = pickle.loads(base64.b64decode(artifact["payload"]["data"], validate=True))
        graph = tuple(_walk_object_graph(decoded))
        assert not any(isinstance(value, ResolvedLikelihoodWeights) for value in graph)
        assert not any(
            isinstance(value, np.ndarray)
            and value.shape == response.shape
            and np.array_equal(value, response)
            for value in graph
        )
        assert not any(
            isinstance(value, np.ndarray)
            and value.shape == weights.shape
            and np.array_equal(value, weights)
            for value in graph
        )


def test_efs_double_penalty_and_fallback_telemetry_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, offsets = _data()
    original_solver = efs_module.fit_dense_fixed_lambda

    def fallback_solver(*args, **kwargs):
        result = original_solver(*args, **kwargs)
        observed = result.terminal_curvature
        return replace(
            result,
            terminal_curvature=CurvatureTelemetry(
                requested_source="observed",
                actual_source="fisher",
                reason="material_indefiniteness_after_retry",
                minimum_eigenvalue=observed.minimum_eigenvalue,
                rank=observed.rank,
                condition_estimate=observed.condition_estimate,
                fallback_count=1,
            ),
        )

    monkeypatch.setattr(smoothing_authority, "fit_dense_fixed_lambda", fallback_solver)
    monkeypatch.setattr(smoothing_loop, "fit_dense_fixed_lambda", fallback_solver)
    monkeypatch.setattr(efs_module, "fit_dense_fixed_lambda", fallback_solver)
    selected = Spline(
        kind="cr",
        n_knots=5,
        select=True,
        lambda_policy={
            "null": LambdaPolicy.estimate(),
            "wiggle": LambdaPolicy.estimate(),
        },
    )
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.015),
        weight_contract=WeightContract(semantics="prior"),
        predictors=(
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {"x": selected}),
        ),
        offsets=offsets,
        lambdas={"scale:x#null": 0.2, "scale:x#wiggle": 0.3},
        efs_config=DistributionalEFSConfig(outer="efs", max_iterations=3, tolerance=10.0),
    )
    restored = deserialize_distributional_model(serialize_distributional_model(model))

    assert set(restored.smoothing_parameters) == {"scale:x#null", "scale:x#wiggle"}
    assert restored.smoothing is not None
    assert restored.smoothing.fallback_count == model.smoothing.fallback_count == 1
    assert restored.telemetry.actual_source == "fisher"
    assert restored.telemetry.reason == "material_indefiniteness_after_retry"
    assert restored.telemetry.fallback_count == 1
    assert restored.smoothing.matched_certified is False
    assert all(
        fit.family_likelihood_plan_identifier == model.fit_state.family_likelihood_plan_identifier
        for fit in restored.smoothing.coefficient_fits
    )
    assert [component.lambda_policy for component in restored.layout.penalties] == [
        LambdaPolicy.estimate(),
        LambdaPolicy.estimate(),
    ]
    _assert_round_trip(model, restored, frame, offsets)


def test_accelerated_efs_manifest_and_round_trip_preserve_compact_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, offsets = _data()
    objectives = iter((0.0, -1.0, -2.0))
    proposal_calls = 0

    def objective(*args, **kwargs) -> float:
        return next(objectives)

    def proposal(self, **kwargs) -> MultisecantDecision:
        nonlocal proposal_calls
        del kwargs
        proposal_calls += 1
        if proposal_calls == 1:
            return MultisecantDecision(proposal=None, refusal_reason="warming")
        current = self._pairs[-1].log_lambdas
        log_step = np.full(current.shape, 0.1)
        return MultisecantDecision(
            proposal=MultisecantProposal(
                log_lambdas=current + log_step,
                log_step=log_step,
                raw_residual_norm=1.0,
                model_residual_norm=0.5,
                numerical_rank=1,
                secant_depth=1,
            ),
            refusal_reason=None,
        )

    monkeypatch.setattr(smoothing_objective, "joint_laplace_objective", objective)
    monkeypatch.setattr(efs_module, "joint_laplace_objective", objective)
    monkeypatch.setattr(WindowedTypeIIAnderson, "propose", proposal)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.015),
        weight_contract=WeightContract(semantics="prior"),
        predictors=(
            Predictor("location", {"x": Numeric()}),
            Predictor("scale", {"x": Spline(kind="cr", n_knots=5)}),
        ),
        offsets=offsets,
        lambdas={"scale:x#wiggle": 0.3},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=2,
            tolerance=1.0e-12,
            acceleration="multisecant",
            acceleration_history=3,
            acceleration_max_amplification=4.0,
        ),
    )

    manifest = distributional_manifest(model)
    smoothing_manifest = manifest["smoothing"]
    assert smoothing_manifest is not None
    assert {
        "acceleration": "multisecant",
        "acceleration_history": 3,
        "acceleration_max_amplification": 4.0,
    }.items() <= smoothing_manifest["config"].items()
    assert {
        "acceleration_outcome",
        "acceleration_refusal_reason",
        "accelerated_fit_index",
        "raw_backtracks",
        "boundary_nominations",
    } <= smoothing_manifest["history"][1].keys()
    assert smoothing_manifest["accelerated_trial_count"] == 1
    assert smoothing_manifest["accelerated_accept_count"] == 1
    assert smoothing_manifest["raw_fallback_count"] == 0

    restored = deserialize_distributional_model(serialize_distributional_model(model))
    assert restored.smoothing is not None and model.smoothing is not None
    assert restored.smoothing.config == model.smoothing.config
    assert restored.smoothing.history == model.smoothing.history
    assert restored.smoothing.terminal_raw_max_log_step == (
        model.smoothing.terminal_raw_max_log_step
    )
    assert restored.smoothing.unresolved_upper_bound == model.smoothing.unresolved_upper_bound
    assert restored.smoothing.accelerated_trial_count == 1
    assert restored.smoothing.accelerated_accept_count == 1
    assert restored.smoothing.raw_fallback_count == 0
    _assert_round_trip(model, restored, frame, offsets)


def test_exact_face_manifest_and_round_trip_preserve_terminal_authority(
    _exact_face_model,
) -> None:
    model = _exact_face_model
    smoothing = model.smoothing
    assert smoothing is not None
    terminal = smoothing.terminal_fit
    face = terminal.coefficient_face
    reduced_rank = terminal.terminal_reduced_rank
    assert face is not None
    assert reduced_rank is not None

    manifest = distributional_manifest(model)
    smoothing_manifest = manifest["smoothing"]
    assert smoothing_manifest is not None
    assert smoothing_manifest["matched_certified"] is False
    terminal_manifest = smoothing_manifest["coefficient_fits"][smoothing.terminal_fit_index]
    face_manifest = terminal_manifest["coefficient_face"]
    assert face_manifest["component_names"] == list(face.component_names)
    assert face_manifest["null_basis"] == _independent_array_descriptor(face.null_basis)
    assert terminal_manifest["terminal_reduced_rank"]["policy_version"] == (
        reduced_rank.policy_version
    )
    assert terminal_manifest["terminal_reduced_rank"]["rank"] == reduced_rank.rank
    endpoint_manifest = smoothing_manifest["terminal_endpoint_directions"]
    assert set(endpoint_manifest) == set(face.component_names)
    assert all(value["decision"] == "endpoint" for value in endpoint_manifest.values())

    restored = deserialize_distributional_model(serialize_distributional_model(model))
    assert restored.smoothing is not None
    restored_face = restored.smoothing.terminal_fit.coefficient_face
    assert restored_face is not None
    np.testing.assert_array_equal(restored_face.null_basis, face.null_basis)
    assert restored.smoothing.terminal_endpoint_directions == (
        smoothing.terminal_endpoint_directions
    )
    assert smoothing.matched_certified is False
    assert restored.smoothing.matched_certified is False


def test_resolution_limited_exact_face_reason_round_trips_without_certification(
    _exact_face_model,
) -> None:
    """Kills dropping the new reason or treating it as matched certification."""
    model = _exact_face_model
    smoothing = model.smoothing
    assert smoothing is not None
    terminal = smoothing.terminal_fit
    face = terminal.coefficient_face
    assert face is not None

    retained_score = np.zeros(face.reduced_width, dtype=np.float64)
    retained_score[0] = (
        2.0 * terminal.config.tolerance * (1.0 + abs(terminal.penalized_optimizing_log_likelihood))
    )
    resolution_limited_terminal = replace(
        terminal,
        terminal_score=face.lift_vector(retained_score),
        convergence_reason="resolution_limited_stationarity",
    )
    coefficient_fits = list(smoothing.coefficient_fits)
    coefficient_fits[smoothing.terminal_fit_index] = resolution_limited_terminal
    resolution_limited_smoothing = replace(
        smoothing,
        coefficient_fits=tuple(coefficient_fits),
    )
    resolution_limited_state = replace(
        model.fit_state,
        solver_result=resolution_limited_terminal,
        smoothing=resolution_limited_smoothing,
    )
    resolution_limited_model = replace(model, _fit_state=resolution_limited_state)

    serialized = serialize_distributional_model(resolution_limited_model)
    restored = deserialize_distributional_model(serialized)
    restored_smoothing = restored.smoothing
    assert json.loads(serialized)["schema_version"] == SCHEMA_VERSION == "9.0.0"
    assert restored_smoothing is not None
    assert restored_smoothing.terminal_fit.convergence_reason == "resolution_limited_stationarity"
    np.testing.assert_array_equal(
        restored_smoothing.terminal_fit.terminal_score,
        resolution_limited_terminal.terminal_score,
    )
    assert restored_smoothing.matched_certified is False
    with pytest.raises(
        RuntimeError,
        match="exact coefficient face is numerically supported but not certified",
    ):
        restored_smoothing.assert_matched_certified()


def test_joint_exact_face_manifest_and_round_trip_preserve_atomic_authority(
    _joint_exact_face_model,
) -> None:
    model, frame = _joint_exact_face_model
    smoothing = model.smoothing
    assert smoothing is not None
    revalidation = smoothing.history[-1]
    evidence = revalidation.endpoint_direction_evidence
    assert isinstance(evidence, JointEndpointDirectionEvidence)

    manifest = distributional_manifest(model)
    smoothing_manifest = manifest["smoothing"]
    assert smoothing_manifest is not None
    receipt = smoothing_manifest["history"][-1]["endpoint_direction_evidence"]
    fields = receipt["fields"]
    named_directions = fields["component_directions"]
    assert receipt["type"] == ("superglm.distributional.result.JointEndpointDirectionEvidence")
    assert [item[0] for item in named_directions] == list(evidence.component_names)
    assert fields["endpoint_fit_index"] == revalidation.accepted_fit_index
    assert fields["endpoint_fit_index"] == evidence.endpoint_fit_index
    assert fields["coefficient_tolerance"] == evidence.coefficient_tolerance
    assert revalidation.coefficient_tolerances == (fields["coefficient_tolerance"],)
    assert fields["coefficient_tolerance"] == 1.0e-12
    for manifest_item, (name, direction) in zip(
        named_directions,
        evidence.component_directions,
        strict=True,
    ):
        assert manifest_item[0] == name
        direction_fields = manifest_item[1]["fields"]
        assert direction_fields["decision"] == direction.decision == "endpoint"
        assert direction_fields["lower_bound"] == direction.lower_bound
        assert direction_fields["upper_bound"] == direction.upper_bound
        assert direction_fields["endpoint_objective"] == evidence.endpoint_objective

    restored = deserialize_distributional_model(serialize_distributional_model(model))
    _assert_round_trip(model, restored, frame, None)
    assert restored.smoothing is not None
    assert restored.smoothing.history == smoothing.history
    assert restored.smoothing.terminal_endpoint_directions == (
        smoothing.terminal_endpoint_directions
    )
    restored_evidence = restored.smoothing.history[-1].endpoint_direction_evidence
    assert isinstance(restored_evidence, JointEndpointDirectionEvidence)
    assert restored_evidence == evidence


def test_restore_reconstructs_joint_endpoint_receipts_through_their_invariants(
    _joint_exact_face_model,
) -> None:
    model, _frame = _joint_exact_face_model
    artifact = json.loads(serialize_distributional_model(model))

    def replace_canonical_tuples_with_manifest_equivalent_lists(restored) -> None:
        assert restored.smoothing is not None
        for iteration in restored.smoothing.history:
            evidence = iteration.endpoint_direction_evidence
            if isinstance(evidence, JointEndpointDirectionEvidence):
                object.__setattr__(
                    evidence,
                    "component_directions",
                    list(evidence.component_directions),
                )

    noncanonical = _rehash_pickled_artifact(
        artifact,
        replace_canonical_tuples_with_manifest_equivalent_lists,
    )
    restored = deserialize_distributional_model(noncanonical)

    assert restored.smoothing is not None
    restored_joint_receipts = tuple(
        iteration.endpoint_direction_evidence
        for iteration in restored.smoothing.history
        if isinstance(
            iteration.endpoint_direction_evidence,
            JointEndpointDirectionEvidence,
        )
    )
    assert len(restored_joint_receipts) == 2
    assert all(type(evidence.component_directions) is tuple for evidence in restored_joint_receipts)


def test_same_major_finite_payload_without_optional_endpoint_fields_is_migrated(
    _decomposition_models,
) -> None:
    """Optional finite state gains defaults without changing its current manifest."""

    _, model = _decomposition_models
    artifact = json.loads(serialize_distributional_model(model))
    smoothing_manifest = artifact["manifest"]["smoothing"]
    assert smoothing_manifest is not None
    assert "terminal_endpoint_directions" not in smoothing_manifest

    def remove_optional_endpoint_fields(restored) -> None:
        assert restored.smoothing is not None
        vars(restored.smoothing).pop("terminal_endpoint_directions")
        for iteration in restored.smoothing.history:
            vars(iteration).pop("endpoint_assessment_failure_reason")

    same_major = _rehash_pickled_artifact(artifact, remove_optional_endpoint_fields)
    same_major_artifact = json.loads(same_major)
    decoded = pickle.loads(base64.b64decode(same_major_artifact["payload"]["data"], validate=True))
    assert decoded.smoothing is not None
    assert "terminal_endpoint_directions" not in vars(decoded.smoothing)
    with pytest.raises(AttributeError):
        decoded.smoothing.terminal_endpoint_directions
    assert all(
        "endpoint_assessment_failure_reason" not in vars(iteration)
        for iteration in decoded.smoothing.history
    )

    restored = deserialize_distributional_model(same_major)

    assert restored.smoothing is not None
    assert "terminal_endpoint_directions" in vars(restored.smoothing)
    assert dict(restored.smoothing.terminal_endpoint_directions) == {}
    assert all(
        iteration.endpoint_assessment_failure_reason is None
        for iteration in restored.smoothing.history
    )
    assert distributional_manifest(restored) == same_major_artifact["manifest"]


def test_same_major_finite_payload_without_optional_newton_decrement_is_migrated(
    _decomposition_models,
) -> None:
    """An optional solver setting retains its authenticated current manifest."""

    _, model = _decomposition_models
    artifact = json.loads(serialize_distributional_model(model))
    artifact["manifest"]["solver"]["config"].pop(
        "newton_decrement_tolerance",
        None,
    )
    smoothing_manifest = artifact["manifest"]["smoothing"]
    assert smoothing_manifest is not None
    for fit_manifest in smoothing_manifest["coefficient_fits"]:
        fit_manifest["config"].pop("newton_decrement_tolerance", None)

    def remove_optional_config_field(restored) -> None:
        configs = {id(restored.result.config): restored.result.config}
        assert restored.smoothing is not None
        configs.update({id(fit.config): fit.config for fit in restored.smoothing.coefficient_fits})
        for config in configs.values():
            vars(config).pop("newton_decrement_tolerance")

    same_major = _rehash_pickled_artifact(artifact, remove_optional_config_field)
    restored = deserialize_distributional_model(same_major)

    assert restored.result.config.newton_decrement_tolerance is None
    assert restored.smoothing is not None
    assert all(
        fit.config.newton_decrement_tolerance is None for fit in restored.smoothing.coefficient_fits
    )
    assert distributional_manifest(restored) == artifact["manifest"]


def test_rehashed_payload_cannot_remove_exact_face_terminal_evidence(
    _exact_face_model,
) -> None:
    artifact = json.loads(serialize_distributional_model(_exact_face_model))

    def remove_terminal_evidence(restored) -> None:
        assert restored.smoothing is not None
        object.__setattr__(restored.smoothing, "terminal_endpoint_directions", {})

    forged = _rehash_pickled_artifact(artifact, remove_terminal_evidence)
    with pytest.raises(DistributionalSerializationError, match="manifest does not match"):
        deserialize_distributional_model(forged)


def test_rehashed_payload_cannot_mutate_exact_face_geometry(_exact_face_model) -> None:
    artifact = json.loads(serialize_distributional_model(_exact_face_model))

    def mutate_null_basis(restored) -> None:
        assert restored.smoothing is not None
        face = restored.smoothing.terminal_fit.coefficient_face
        assert face is not None
        forged = np.array(face.null_basis, copy=True)
        forged[0, 0] = np.nextafter(forged[0, 0], np.inf)
        forged.setflags(write=False)
        object.__setattr__(face, "null_basis", forged)

    forged = _rehash_pickled_artifact(artifact, mutate_null_basis)
    with pytest.raises(DistributionalSerializationError, match="manifest does not match"):
        deserialize_distributional_model(forged)


def test_rehashed_payload_cannot_shift_every_terminal_face_penalty(
    _exact_face_model,
) -> None:
    """Kills a common penalty offset that preserves pairwise fitted differences."""
    artifact = json.loads(serialize_distributional_model(_exact_face_model))

    def shift_final_assessment(restored) -> None:
        smoothing = restored.smoothing
        assert smoothing is not None
        indices = {
            index
            for evidence in smoothing.terminal_endpoint_directions.values()
            for index in evidence.fit_indices
        }
        assert indices
        for index in indices:
            fit = smoothing.coefficient_fits[index]
            offset = 0.125 * np.eye(len(fit.coefficients))
            object.__setattr__(fit, "penalty", fit.penalty + offset)
            object.__setattr__(
                fit,
                "terminal_data_curvature",
                fit.terminal_data_curvature - offset,
            )
            forged_penalty_value = 0.5 * float(fit.coefficients @ (fit.penalty) @ fit.coefficients)
            object.__setattr__(fit, "penalty_value", forged_penalty_value)
            object.__setattr__(
                fit,
                "penalized_optimizing_log_likelihood",
                fit.optimizing_log_likelihood - forged_penalty_value,
            )
            object.__setattr__(
                fit,
                "penalized_log_likelihood",
                fit.log_likelihood - forged_penalty_value,
            )

    forged_artifact = json.loads(_rehash_pickled_artifact(artifact, shift_final_assessment))
    forged_model = pickle.loads(base64.b64decode(forged_artifact["payload"]["data"], validate=True))
    forged_artifact["manifest"] = distributional_manifest(forged_model)
    forged = json.dumps(forged_artifact).encode()
    with pytest.raises(DistributionalSerializationError, match="layout and fitted lambdas"):
        deserialize_distributional_model(forged)


def test_artifact_binds_the_curvature_matrix_scope() -> None:
    model, _, _, _ = _fixed_model()
    manifest = distributional_manifest(model)
    assert manifest["solver"]["curvature_telemetry"]["matrix_kind"] == "penalized"
    restored = deserialize_distributional_model(serialize_distributional_model(model))
    assert restored.result.terminal_curvature.matrix_kind == "penalized"


def test_schema_eight_missing_curvature_scope_preserves_legacy_manifest() -> None:
    model, _, _, _ = _fixed_model()
    artifact = json.loads(serialize_distributional_model(model))

    def remove_scope(restored):
        seen = set()

        def visit(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, CurvatureTelemetry):
                if "matrix_kind" in vars(value):
                    object.__delattr__(value, "matrix_kind")
            elif is_dataclass(value):
                for field in fields(value):
                    visit(getattr(value, field.name))
            elif isinstance(value, Mapping):
                for item in value.values():
                    visit(item)
            elif isinstance(value, tuple):
                for item in value:
                    visit(item)

        visit(restored)

    artifact = json.loads(_rehash_pickled_artifact(artifact, remove_scope))
    artifact["schema_version"] = "8.0.0"

    def strip_manifest(value):
        if isinstance(value, dict):
            value.pop("matrix_kind", None)
            for item in value.values():
                strip_manifest(item)
        elif isinstance(value, list):
            for item in value:
                strip_manifest(item)

    strip_manifest(artifact["manifest"])
    restored = deserialize_distributional_model(json.dumps(artifact).encode())
    assert restored.result.terminal_curvature.matrix_kind is None
    assert distributional_manifest(restored) == artifact["manifest"]
    np.testing.assert_array_equal(restored.result.coefficients, model.result.coefficients)


def test_unreleased_previous_major_is_refused_before_unpickling() -> None:
    model, _, _, _ = _fixed_model()
    artifact = _with_unpickle_sentinel(json.loads(serialize_distributional_model(model)))
    artifact["schema_version"] = "7.3.0"

    with pytest.raises(DistributionalSerializationError, match="unreadable by this build"):
        deserialize_distributional_model(json.dumps(artifact).encode())


def test_deserializer_rejects_unknown_major_and_missing_gaussian_scale_floor() -> None:
    model, _, _, _ = _fixed_model()
    artifact = json.loads(serialize_distributional_model(model))

    future = f"{_current_major() + 1}.0.0"
    artifact["schema_version"] = future
    with pytest.raises(DistributionalSerializationError, match="unreadable by this build"):
        deserialize_distributional_model(json.dumps(artifact).encode())

    artifact["schema_version"] = SCHEMA_VERSION
    del artifact["manifest"]["family"]["config"]["scale_floor"]
    with pytest.raises(DistributionalSerializationError, match="scale_floor"):
        deserialize_distributional_model(json.dumps(artifact).encode())


def test_deserializer_rejects_payload_or_manifest_tampering() -> None:
    model, _, _, _ = _fixed_model()
    artifact = json.loads(serialize_distributional_model(model))
    artifact["payload"]["sha256"] = "0" * 64
    with pytest.raises(DistributionalSerializationError, match="payload digest"):
        deserialize_distributional_model(json.dumps(artifact).encode())

    artifact = json.loads(serialize_distributional_model(model))
    artifact["manifest"]["fit"]["rank"] += 1
    with pytest.raises(DistributionalSerializationError, match="manifest does not match"):
        deserialize_distributional_model(json.dumps(artifact).encode())


def test_identity_penalty_component_round_trips_with_its_compact_geometry() -> None:
    """A random effect carries ``penalty_kind='identity'`` and no raw omega.

    The manifest has to describe the compact geometry master's PenaltyComponent
    gained (``penalty_kind`` / ``repeat_count`` / ``block_width``), and it has
    to tolerate ``omega_raw is None`` — describing an absent array is what an
    identity component always presents.
    """
    rng = np.random.default_rng(90211)
    rows = 96
    x = np.linspace(-1.0, 1.0, rows)
    frame = pd.DataFrame({"x": x, "site": [f"s{index % 4}" for index in range(rows)]})
    response = 0.5 * np.sin(np.pi * x) + rng.normal(scale=0.25, size=rows)

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.04),
        weight_contract=WeightContract(semantics="prior"),
        predictors=(
            Predictor("location", {"x": Spline(kind="cr", n_knots=6), "site": RandomEffect()}),
            Predictor("scale", {"x": Numeric()}),
        ),
        lambdas={"location:x#wiggle": 0.4, "location:site#wiggle": 1.7},
    )

    identity = next(
        component
        for component in model.layout.penalties
        if component.name == "location:site#wiggle"
    )
    assert identity.penalty_kind == "identity"
    assert identity.omega_raw is None

    manifest = distributional_manifest(model)
    described = {component["name"]: component for component in manifest["penalties"]["components"]}
    assert described["location:site#wiggle"]["omega_raw"] is None
    assert described["location:site#wiggle"]["penalty_kind"] == "identity"
    assert described["location:x#wiggle"]["penalty_kind"] == "dense"
    for component in model.layout.penalties:
        entry = described[component.name]
        assert entry["repeat_count"] == component.repeat_count
        assert entry["block_width"] == component.block_width

    restored = deserialize_distributional_model(serialize_distributional_model(model))
    np.testing.assert_array_equal(restored.coefficients, model.coefficients)
    np.testing.assert_array_equal(
        restored.predict(frame),
        model.predict(frame),
    )
    assert [component.penalty_kind for component in restored.layout.penalties] == [
        component.penalty_kind for component in model.layout.penalties
    ]


@pytest.mark.parametrize(
    "legacy_version",
    [_SCHEMA_BEFORE_PENALTY_GEOMETRY, _SCHEMA_BEFORE_WEIGHT_CONTRACT],
)
def test_precontract_artifact_raises_typed_legacy_refusal_before_unpickling(
    legacy_version: str,
) -> None:
    """A pre-contract artifact is semantically unknowable and cannot be loaded.

    The payload is digest-valid but deliberately not a pickle.  Seeing the
    typed legacy refusal proves schema/manifest evidence routes the artifact
    before payload unpickling; old compact summaries cannot prove whether an
    integral-looking value was a count or an historical power weight.
    """
    model, _, _, _ = _fixed_model()
    artifact = json.loads(serialize_distributional_model(model))
    if legacy_version == _SCHEMA_BEFORE_PENALTY_GEOMETRY:
        artifact = _as_schema_1_0_0(artifact)
    else:
        artifact["schema_version"] = legacy_version
    artifact["manifest"].pop("weights", None)
    artifact["manifest"].pop("execution", None)
    raw = b"digest-valid bytes that must never reach pickle.loads"
    artifact["payload"]["data"] = base64.b64encode(raw).decode("ascii")
    artifact["payload"]["sha256"] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(LegacyPowerWeightArtifactError, match="legacy|pre-contract"):
        deserialize_distributional_model(json.dumps(artifact).encode())


def test_current_duplicate_mismatch_is_corruption_not_legacy() -> None:
    model, _, _, _ = _fixed_model()
    artifact = json.loads(serialize_distributional_model(model))
    artifact["manifest"]["null_model"]["weight_root_digest"] = "0" * 64

    with pytest.raises(DistributionalSerializationError, match="manifest does not match"):
        deserialize_distributional_model(json.dumps(artifact).encode())


@pytest.mark.parametrize("schema_alias", [True, 1.0], ids=["bool", "float"])
def test_current_weight_schema_type_alias_is_manifest_corruption(schema_alias: object) -> None:
    model, _, _, _ = _fixed_model()
    artifact = json.loads(serialize_distributional_model(model))
    artifact["manifest"]["weights"]["contract"]["schema_version"] = schema_alias

    with pytest.raises(DistributionalSerializationError, match="manifest does not match"):
        deserialize_distributional_model(json.dumps(artifact).encode())


def test_current_version_round_trip_states_the_version_it_wrote() -> None:
    model, frame, offsets, _ = _fixed_model()
    serialized = serialize_distributional_model(model)

    assert json.loads(serialized)["schema_version"] == SCHEMA_VERSION
    _assert_round_trip(model, deserialize_distributional_model(serialized), frame, offsets)


def test_schema_before_acceleration_telemetry_is_an_ordinary_read_barrier() -> None:
    model, _, _, _ = _fixed_model()
    artifact = json.loads(serialize_distributional_model(model))
    artifact["schema_version"] = _SCHEMA_BEFORE_ACCELERATION_TELEMETRY
    raw = b"digest-valid bytes that an older schema must refuse before unpickling"
    artifact["payload"]["data"] = base64.b64encode(raw).decode("ascii")
    artifact["payload"]["sha256"] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(DistributionalSerializationError, match="unreadable by this build"):
        deserialize_distributional_model(json.dumps(artifact).encode())


def test_schema_before_terminal_raw_evidence_is_an_ordinary_read_barrier() -> None:
    model, _, _, _ = _fixed_model()
    artifact = json.loads(serialize_distributional_model(model))
    artifact["schema_version"] = _SCHEMA_BEFORE_TERMINAL_RAW_EVIDENCE
    raw = b"digest-valid bytes that an older schema must refuse before unpickling"
    artifact["payload"]["data"] = base64.b64encode(raw).decode("ascii")
    artifact["payload"]["sha256"] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(DistributionalSerializationError, match="unreadable by this build"):
        deserialize_distributional_model(json.dumps(artifact).encode())


def test_schema_before_joint_endpoint_evidence_is_refused_before_unpickling() -> None:
    """Schema 5 cannot be upgraded by inventing joint endpoint authority."""
    model, _, _, _ = _fixed_model()
    artifact = _with_unpickle_sentinel(json.loads(serialize_distributional_model(model)))
    artifact["schema_version"] = _SCHEMA_BEFORE_JOINT_ENDPOINT_EVIDENCE

    with pytest.raises(
        DistributionalSerializationError,
        match="unreadable by this build.*major versions differ",
    ):
        deserialize_distributional_model(json.dumps(artifact).encode())


def test_manifest_key_set_is_pinned_to_the_current_major(
    _decomposition_models,
    _joint_exact_face_model,
) -> None:
    """Pin the schema-9 manifest, including terminal curvature scope.

    ``deserialize_distributional_model`` compares the recomputed manifest to the
    stored one for equality, so omitting a behavior-driving key leaves that state
    unauthenticated. Deliberate manifest changes must bump the major component
    and update these key sets together.
    """
    assert SCHEMA_VERSION == "9.0.0"

    manifest = distributional_manifest(_fixed_model()[0])
    assert set(manifest) == {
        "family",
        "fit",
        "inference",
        "layout",
        "null_model",
        "parameter_order",
        "parameters",
        "penalties",
        "predictors",
        "retained_rows",
        "smoothing",
        "solver",
        "execution",
        "weights",
    }
    assert set(manifest["penalties"]) == {"components", "lambdas"}
    solver_fields = {
        "backtracking_steps",
        "config",
        "converged",
        "convergence_reason",
        "curvature_telemetry",
        "execution_backend_identifier",
        "family_likelihood_plan_identifier",
        "history",
        "iterations",
        "objective_relative_change",
        "penalty_value",
        "rank",
        "resolved_chunk_size",
        "score_relative",
        "step_relative",
        *_LIKELIHOOD_DECOMPOSITION_FIELDS,
    }
    assert set(manifest["solver"]) == solver_fields
    assert set(manifest["solver"]["curvature_telemetry"]) == {
        "requested_source",
        "actual_source",
        "reason",
        "minimum_eigenvalue",
        "rank",
        "condition_estimate",
        "fallback_count",
        "matrix_kind",
    }
    smoothing_model = _decomposition_models[1]
    smoothing = smoothing_model.smoothing
    assert smoothing is not None
    smoothing_manifest = distributional_manifest(smoothing_model)["smoothing"]
    assert smoothing_manifest is not None
    assert set(smoothing_manifest) == {
        "accelerated_accept_count",
        "accelerated_trial_count",
        "coefficient_fits",
        "config",
        "converged",
        "convergence_reason",
        "fallback_count",
        "history",
        "initial_lambdas",
        "initial_objective",
        "iterations",
        "lambdas",
        "matched_certified",
        "objective",
        "raw_fallback_count",
        "terminal_fit_index",
        "terminal_raw_max_log_step",
        "unresolved_upper_bound",
    }
    efs_iteration_fields = {
        "acceleration_outcome",
        "acceleration_refusal_reason",
        "accelerated_fit_index",
        "accepted",
        "accepted_curvature",
        "accepted_fit_index",
        "accepted_log_steps",
        "backtracks",
        "boundary_nominations",
        "coefficient_fit_indices",
        "coefficient_tolerances",
        "iteration",
        "lambdas_after",
        "lambdas_before",
        "max_accepted_log_step",
        "max_proposed_log_step",
        "objective_after",
        "objective_before",
        "objective_relative_change",
        "proposed_lambdas",
        "proposed_log_steps",
        "quadratic_forms",
        "raw_backtracks",
        "source_fit_index",
        "stage",
        "step_source",
        "gradient",
        "gradient_certificate",
        "hessian_certificate",
        "projected_gradient_norm",
        "newton_ridge",
        "trace_terms",
        "update_curvature",
    }
    assert all(
        set(iteration) == efs_iteration_fields for iteration in smoothing_manifest["history"]
    )
    assert all(
        set(coefficient_fit) == solver_fields
        for coefficient_fit in smoothing_manifest["coefficient_fits"]
    )
    for component in manifest["penalties"]["components"]:
        assert set(component) == {
            "block_width",
            "component_type",
            "group_index",
            "group_name",
            "group_slice",
            "lambda",
            "lambda_policy",
            "log_det_omega_plus",
            "name",
            "omega_raw",
            "omega_ssp",
            "penalty_kind",
            "positive_eigenvalues",
            "rank",
            "repeat_count",
        }

    joint_model, _frame = _joint_exact_face_model
    joint_smoothing = joint_model.smoothing
    assert joint_smoothing is not None
    joint_smoothing_manifest = distributional_manifest(joint_model)["smoothing"]
    assert joint_smoothing_manifest is not None
    assert set(joint_smoothing_manifest) == {
        *set(smoothing_manifest),
        "terminal_endpoint_directions",
    }
    joint_iteration = joint_smoothing_manifest["history"][-1]
    assert set(joint_iteration) == {
        *efs_iteration_fields,
        "endpoint_direction_evidence",
        "revalidated_face_components",
    }
    receipt = joint_iteration["endpoint_direction_evidence"]
    assert set(receipt) == {"fields", "type"}
    assert receipt["type"] == ("superglm.distributional.result.JointEndpointDirectionEvidence")
    assert set(receipt["fields"]) == {
        "authority_identifier",
        "coefficient_tolerance",
        "component_directions",
        "endpoint_fit_index",
    }
    assert receipt["fields"]["authority_identifier"] == (
        "joint-analytic-observed-curvature-direction/v1"
    )
    assert type(receipt["fields"]["endpoint_fit_index"]) is int
    assert type(receipt["fields"]["coefficient_tolerance"]) is float
    component_directions = receipt["fields"]["component_directions"]
    assert type(component_directions) is list
    assert [item[0] for item in component_directions] == list(
        joint_smoothing.terminal_fit.coefficient_face.component_names
    )
    endpoint_fields = {
        "analytic_derivative",
        "authority_identifier",
        "coefficient_tolerance",
        "curvature_drift_term",
        "curvature_schur_term",
        "decision",
        "endpoint_objective",
        "fit_indices",
        "lower_bound",
        "numerical_error",
        "profile_score_term",
        "upper_bound",
    }
    assert all(
        type(item) is list
        and len(item) == 2
        and type(item[0]) is str
        and set(item[1]) == {"fields", "type"}
        and item[1]["type"] == "superglm.distributional.result.EndpointDirectionEvidence"
        and set(item[1]["fields"]) == endpoint_fields
        for item in component_directions
    )


@pytest.mark.parametrize(
    ("field", "mutated"),
    [
        ("terminal_raw_max_log_step", 0.125),
        ("unresolved_upper_bound", ("forged:upper",)),
    ],
)
def test_rehashed_payload_cannot_mutate_terminal_smoothing_evidence(
    _decomposition_models,
    field: str,
    mutated: object,
) -> None:
    """Kills omitting either terminal authority field from the manifest."""

    _, model = _decomposition_models
    artifact = json.loads(serialize_distributional_model(model))

    def mutate(restored) -> None:
        assert restored.smoothing is not None
        object.__setattr__(restored.smoothing, field, mutated)

    corrupted = _rehash_pickled_artifact(artifact, mutate)
    with pytest.raises(DistributionalSerializationError, match="manifest does not match"):
        deserialize_distributional_model(corrupted)


@pytest.mark.parametrize("field", _LIKELIHOOD_DECOMPOSITION_FIELDS)
@pytest.mark.parametrize("target", ["terminal", "efs"], ids=["terminal", "efs-fit"])
def test_rehashed_payload_cannot_mutate_any_likelihood_decomposition_channel(
    _decomposition_models,
    target: str,
    field: str,
) -> None:
    """Kills payload changes hidden behind omitted solver-manifest fields."""

    fixed, efs = _decomposition_models
    model = fixed if target == "terminal" else efs
    artifact = json.loads(serialize_distributional_model(model))

    def mutate(restored) -> None:
        fit = restored.result
        if target == "efs":
            assert restored.smoothing is not None
            fit = restored.smoothing.coefficient_fits[0]
        baseline = getattr(fit, field, fit.initial_penalized_log_likelihood)
        object.__setattr__(fit, field, float(baseline) + 0.125)

    corrupted = _rehash_pickled_artifact(artifact, mutate)
    with pytest.raises(
        DistributionalSerializationError,
        match="decomposition|manifest does not match",
    ):
        deserialize_distributional_model(corrupted)


@pytest.mark.parametrize("target", ["terminal", "efs"], ids=["terminal", "efs-fit"])
def test_restore_revalidates_live_likelihood_decomposition_before_manifest_comparison(
    _decomposition_models,
    target: str,
) -> None:
    """Kills trusting a rehashed pickle whose omitted channels violate live identities."""

    fixed, efs = _decomposition_models
    model = fixed if target == "terminal" else efs
    artifact = json.loads(serialize_distributional_model(model))

    def mutate(restored) -> None:
        fit = restored.result
        if target == "efs":
            assert restored.smoothing is not None
            fit = restored.smoothing.coefficient_fits[0]
        object.__setattr__(
            fit,
            "optimizing_log_likelihood",
            float(fit.optimizing_log_likelihood) + 0.25,
        )
        object.__setattr__(
            fit,
            "parameter_independent_carrier",
            float(fit.parameter_independent_carrier) + 0.5,
        )

    corrupted = _rehash_pickled_artifact(artifact, mutate)
    with pytest.raises(DistributionalSerializationError, match="decomposition"):
        deserialize_distributional_model(corrupted)


@pytest.mark.parametrize("retain_rows", [False, True], ids=["compact", "retained"])
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_weight_contract_round_trip_preserves_compact_or_executable_state(
    semantics: str,
    retain_rows: bool,
) -> None:
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 9)})
    response = np.array([-0.8, -0.1, 0.2, 0.9, 0.4, 1.2, 1.7, 1.4, 2.1])
    weights = (
        np.array([0.4, 0.0, 1.1, 2.3, 0.7, 1.8, 0.9, 1.4, 2.0])
        if semantics == "prior"
        else np.array([1, 0, 2, 4, 1, 3, 2, 1, 3])
    )
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.01),
        weight_contract=WeightContract(semantics),  # type: ignore[arg-type]
        predictors=(Predictor("location", {}), Predictor("scale", {})),
        sample_weight=weights,
        lambdas={},
        retain_rows=retain_rows,
        discrete=True,
        n_bins={"x": 7},
        chunk_size="auto",
    )
    manifest = distributional_manifest(model)
    restored = deserialize_distributional_model(serialize_distributional_model(model))
    state = restored.fit_state

    assert state.weight_contract == WeightContract(semantics)  # type: ignore[arg-type]
    assert state.weight_provenance.original_count == len(frame)
    assert state.weight_provenance.dropped_count == 1
    assert state.weight_provenance.root_digest == manifest["weights"]["provenance"]["root_digest"]
    assert "values" not in manifest["weights"]
    assert (
        state.family_likelihood_plan_identifier
        == manifest["weights"]["family_likelihood_plan_identifier"]
    )
    assert state.null_model.weight_contract == state.weight_contract
    assert state.null_model.weight_provenance == state.weight_provenance
    assert state.null_model.weight_provenance is state.weight_provenance
    assert state.null_model.family_likelihood_plan_identifier == (
        state.family_likelihood_plan_identifier
    )
    assert state.requested_discrete is True
    assert state.requested_n_bins == {"x": 7}
    assert state.requested_chunk_size == "auto"
    assert state.solver_result.resolved_chunk_size == int(np.count_nonzero(weights))
    assert state.solver_result.execution_backend_identifier == ("distributional-chunked-v1")

    if retain_rows:
        assert state.retained_rows is not None
        assert state.retained_rows.likelihood_weights.provenance is state.weight_provenance
        np.testing.assert_array_equal(
            state.retained_rows.likelihood_weights.values,
            weights[weights > 0],
        )
    else:
        assert state.retained_rows is None

    retained = weights > 0
    expected_mu = np.average(response[retained], weights=weights[retained])
    denominator = (
        int(np.sum(weights)) if semantics == "frequency" else int(np.count_nonzero(retained))
    )
    expected_sigma = np.sqrt(
        np.dot(weights[retained], (response[retained] - expected_mu) ** 2) / denominator
    )
    np.testing.assert_allclose(
        restored.coefficients,
        [expected_mu, np.log(expected_sigma - restored.family.scale_floor)],
        rtol=0.0,
        atol=3e-9,
    )


def test_restore_reconstructs_retained_gamma_plan_from_the_live_response() -> None:
    """Kills accepting a forged response whose stored plan ID names the old response."""

    model, _ = _gamma_model(retain_rows=True)
    artifact = json.loads(serialize_distributional_model(model))
    payload = artifact["payload"]
    restored = pickle.loads(base64.b64decode(payload["data"], validate=True))
    rows = restored.fit_state.retained_rows
    assert rows is not None
    forged_response = np.ascontiguousarray(
        rows.response * np.linspace(1.01, 1.12, len(rows.response)),
        dtype=np.float64,
    )
    object.__setattr__(
        restored.fit_state,
        "retained_rows",
        replace(rows, response=forged_response),
    )
    raw = serialization_module._pickle_model(restored)
    payload["data"] = base64.b64encode(raw).decode("ascii")
    payload["sha256"] = hashlib.sha256(raw).hexdigest()
    artifact["manifest"]["retained_rows"]["response"] = _independent_array_descriptor(
        forged_response
    )

    with pytest.raises(
        DistributionalSerializationError,
        match="retained.*plan|live response|root likelihood plan",
    ):
        deserialize_distributional_model(json.dumps(artifact))


def test_compact_payload_omits_resolved_carriers_and_the_raw_weight_vector() -> None:
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 7)})
    response = np.array([-0.9, -0.2, 0.3, 1.0, 0.6, 1.5, 2.2])
    sentinel_weights = np.array(
        [
            0.1234567890123,
            0.2718281828459,
            0.3141592653589,
            0.5772156649015,
            1.4142135623731,
            1.6180339887499,
            2.7182818284590,
        ]
    )
    compact = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.01),
        weight_contract=WeightContract("prior"),
        predictors=(Predictor("location", {}), Predictor("scale", {})),
        sample_weight=sentinel_weights,
        lambdas={},
        retain_rows=False,
    )
    artifact = json.loads(serialize_distributional_model(compact))
    raw_payload = base64.b64decode(artifact["payload"]["data"], validate=True)
    decoded = pickle.loads(raw_payload)
    graph = tuple(_walk_object_graph(decoded))

    assert not any(isinstance(value, ResolvedLikelihoodWeights) for value in graph)
    assert not any(
        isinstance(value, np.ndarray)
        and value.shape == sentinel_weights.shape
        and np.array_equal(value, sentinel_weights)
        for value in graph
    )
    assert sentinel_weights.astype(np.float64).tobytes(order="C") not in raw_payload


def test_frequency_round_trip_matches_literal_expansion() -> None:
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 8)})
    response = np.array([-0.7, -0.1, 0.5, 0.2, 1.1, 1.4, 1.2, 2.0])
    counts = np.array([1, 3, 0, 2, 4, 1, 2, 3])
    predictors = (Predictor("location", {"x": Numeric()}), Predictor("scale", {}))
    compact = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.01),
        weight_contract=WeightContract("frequency"),
        predictors=predictors,
        sample_weight=counts,
        lambdas={},
        retain_rows=False,
    )
    restored = deserialize_distributional_model(serialize_distributional_model(compact))
    positions = np.repeat(np.arange(len(frame)), counts)
    expanded = fit_dense_distributional(
        frame.iloc[positions].reset_index(drop=True),
        response[positions],
        family=GaussianLS(scale_floor=0.01),
        weight_contract=WeightContract("prior"),
        predictors=predictors,
        lambdas={},
    )

    np.testing.assert_allclose(restored.coefficients, expanded.coefficients, rtol=0.0, atol=2e-11)
    assert restored.fit_state.weight_contract == WeightContract("frequency")


def test_two_piece_configs_are_validated_key_by_key():
    from superglm.distributional.serialization import (
        _validate_two_piece_log_normal_config,
        _validate_two_piece_normal_config,
    )

    good_log_normal = {
        "type": "TwoPieceLogNormalLSS",
        "parametrisation": "location",
        "scale_floor": 0.01,
        "skew_bound": 0.9,
    }
    _validate_two_piece_log_normal_config(good_log_normal)
    for mutation in (
        {"parametrisation": "median"},
        {"scale_floor": -1.0},
        {"skew_bound": 1.0},
        {"skew_bound": "0.9"},
    ):
        with pytest.raises(ValueError):
            _validate_two_piece_log_normal_config({**good_log_normal, **mutation})
    with pytest.raises(ValueError):
        _validate_two_piece_log_normal_config(
            {k: v for k, v in good_log_normal.items() if k != "skew_bound"}
        )
    good_normal = {"type": "TwoPieceNormalLSS", "scale_floor": 0.01, "skew_bound": 0.9}
    _validate_two_piece_normal_config(good_normal)
    with pytest.raises(ValueError):
        _validate_two_piece_normal_config({**good_normal, "parametrisation": "mean"})


def test_log_normal_config_is_validated_key_by_key():
    from superglm.distributional.serialization import _validate_log_normal_config

    good = {"type": "LogNormalLS", "parametrisation": "location", "scale_floor": 0.01}
    _validate_log_normal_config(good)
    _validate_log_normal_config({**good, "parametrisation": "mean"})
    with pytest.raises(DistributionalSerializationError, match="exactly type"):
        _validate_log_normal_config({**good, "skew_bound": 0.9})
    for key in ("type", "parametrisation", "scale_floor"):
        with pytest.raises(DistributionalSerializationError, match="exactly type"):
            _validate_log_normal_config({k: v for k, v in good.items() if k != key})
    with pytest.raises(DistributionalSerializationError, match="parametrisation must be"):
        _validate_log_normal_config({**good, "parametrisation": "median"})
    for floor in (True, "0.01", -1.0, float("nan"), float("inf")):
        with pytest.raises(DistributionalSerializationError, match="scale_floor must be"):
            _validate_log_normal_config({**good, "scale_floor": floor})


@pytest.mark.parametrize(
    ("validator", "config"),
    [
        (
            "_validate_generalized_pareto_config",
            {"type": "GeneralizedParetoLSS", "shape_lower": 0.0, "shape_upper": 10**400},
        ),
        (
            "_validate_two_piece_skew_bound",
            10**400,
        ),
        (
            "_validate_two_piece_scale_floor",
            10**400,
        ),
    ],
)
def test_new_family_validators_reject_oversized_json_integers(validator, config):
    from superglm.distributional import serialization

    function = getattr(serialization, validator)
    with pytest.raises(serialization.DistributionalSerializationError):
        if isinstance(config, dict):
            function(config)
        else:
            function(config, "TwoPieceLogNormalLSS")
