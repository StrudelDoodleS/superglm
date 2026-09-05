from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import superglm.distributional as distributional
import superglm.distributional.family as family_contracts
from superglm.distributional import (
    ConfigurableDistributionalFamily,
    GammaLS,
    GaussianLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    LikelihoodPlanValidatingFamily,
    LogNormalLS,
    NegativeBinomialLS,
    TweedieLSS,
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
)
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    DefaultPredictionFamily,
    DistributionalFamily,
    ExpectedInformationFamily,
    FamilyCapabilities,
    FamilyLikelihoodPlan,
    InitialParameterState,
    NaturalLikelihoodEvaluation,
    ObservationContract,
    ParameterSpec,
    ParameterSupport,
    _validated_complete_fit_configuration,
    _validated_derivative_order,
    _validated_parameter_matrix,
    validate_family,
)
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.predictor import Predictor
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.links import IdentityLink
from superglm.types import GroupInfo


def _capabilities(**overrides: object) -> FamilyCapabilities:
    values: dict[str, object] = {
        "max_derivative_order": 2,
        "expected_information": True,
        "cdf": True,
        "quantile": True,
        "random": True,
        "response_mean": True,
        "censored_response": False,
    }
    values.update(overrides)
    return FamilyCapabilities(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("family", "point"),
    [
        (GaussianLS(), [2.0, 0.9]),
        (GammaLS(), [2.0, 0.9]),
        (TweedieLSS(), [2.0, 0.9, 1.6]),
        (NegativeBinomialLS(), [2.0, 1.5]),
        (GeneralizedGammaLSS(), [2.0, 0.9, 0.3]),
        (GeneralizedParetoLSS(), [2.0, 0.25]),
        (LogNormalLS(), [2.0, 0.9]),
        (TwoPieceNormalLSS(), [2.0, 0.9, 0.2]),
        (TwoPieceLogNormalLSS(), [2.0, 0.9, 0.2]),
    ],
)
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_value_screen_preserves_each_builtin_likelihood(family, point, semantics) -> None:
    """A value-only trial must not reject a different target than its full trial."""
    y = np.array([1.0, 2.0, 1.0, 4.0, 3.0, 2.0, 7.0, 1.0])
    weights = resolve_likelihood_weights(
        np.ones(len(y)) if semantics == "prior" else np.arange(1.0, len(y) + 1.0),
        n_observations=len(y),
        contract=WeightContract(semantics),
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    theta = np.tile(point, (len(y), 1)) * np.linspace(0.8, 1.2, len(y))[:, None]
    value = family.evaluate_natural(y, theta, plan, derivative_order=0)
    full = family.evaluate_natural(y, theta, plan, derivative_order=2)

    assert np.all(full.valid)
    np.testing.assert_array_equal(value.valid, full.valid)
    np.testing.assert_array_equal(value.optimizing_log_likelihood, full.optimizing_log_likelihood)
    np.testing.assert_array_equal(
        value.parameter_independent_carrier, full.parameter_independent_carrier
    )


def _parameter(name: str) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        default_link=IdentityLink(),
        role="location",
        support=ParameterSupport(),
        curvature="observed",
    )


@dataclass(frozen=True)
class _FakePlan:
    weights: ResolvedLikelihoodWeights

    @property
    def plan_identifier(self) -> str:
        return f"fake-plan/v1:{self.weights.digest}"

    def take(self, indices: np.ndarray) -> _FakePlan:
        return _FakePlan(self.weights.take(indices))


@dataclass(frozen=True)
class _FakeFamily:
    parameters: tuple[ParameterSpec, ...]

    def bind_likelihood(
        self,
        y: np.ndarray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> _FakePlan:
        del y, observation
        return _FakePlan(weights)

    def initialize(
        self,
        y: np.ndarray,
        plan: FamilyLikelihoodPlan,
    ) -> InitialParameterState:
        assert isinstance(plan, _FakePlan)
        return InitialParameterState(np.zeros((len(y), len(self.parameters))))

    def evaluate_natural(
        self,
        y: np.ndarray,
        theta: np.ndarray,
        plan: FamilyLikelihoodPlan,
        *,
        derivative_order: int = 2,
    ) -> NaturalLikelihoodEvaluation:
        assert isinstance(plan, _FakePlan)
        order = _validated_derivative_order(derivative_order)
        n_rows, n_parameters = theta.shape
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=np.zeros(n_rows),
            parameter_independent_carrier=np.zeros(n_rows),
            score=None if order == 0 else np.zeros((n_rows, n_parameters)),
            hessian_packed=(
                None if order < 2 else np.zeros((n_rows, n_parameters * (n_parameters + 1) // 2))
            ),
            valid=np.ones(n_rows, dtype=bool),
        )


@dataclass(frozen=True)
class _FrequencyOnlyPlan:
    weights: ResolvedLikelihoodWeights
    row_law: str

    @property
    def plan_identifier(self) -> str:
        return f"frequency-only/v1:{self.row_law}:{self.weights.digest}"

    def take(self, indices: np.ndarray) -> _FrequencyOnlyPlan:
        return _FrequencyOnlyPlan(self.weights.take(indices), self.row_law)


class _FrequencyOnlyFamily:
    """Minimal literal-replication family with an explicit unit-prior bridge."""

    def __init__(self) -> None:
        self.initialize_calls = 0
        self.evaluate_calls = 0
        self.last_bound_row_law: str | None = None

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        return (_parameter("location"),)

    def to_config(self) -> dict[str, str]:
        return {"type": "FrequencyOnlyTestFamily"}

    def bind_likelihood(
        self,
        y: np.ndarray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> _FrequencyOnlyPlan:
        if observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError("complete observations are required")
        if not isinstance(weights, ResolvedLikelihoodWeights):
            raise UnsupportedLikelihoodContractError("resolved weights are required")
        response = np.asarray(y, dtype=np.float64)
        if response.shape != weights.values.shape or not np.all(np.isfinite(response)):
            raise ValueError("response must be a finite vector matching the weights")
        semantics = weights.provenance.contract.semantics
        if semantics == "prior" and not weights.provenance.all_unit:
            raise UnsupportedLikelihoodContractError(
                "FrequencyOnlyTestFamily cannot implement non-unit prior weights"
            )
        row_law = (
            "unit-prior-explicitly-equals-frequency/v1"
            if semantics == "prior"
            else "literal-frequency-replication/v1"
        )
        self.last_bound_row_law = row_law
        return _FrequencyOnlyPlan(weights, row_law)

    def initialize(
        self,
        y: np.ndarray,
        plan: FamilyLikelihoodPlan,
    ) -> InitialParameterState:
        self.initialize_calls += 1
        weights = plan.weights.values
        location = float(np.dot(weights, y) / np.sum(weights, dtype=np.float64))
        return InitialParameterState(np.full((len(y), 1), location))

    def evaluate_natural(
        self,
        y: np.ndarray,
        theta: np.ndarray,
        plan: FamilyLikelihoodPlan,
        *,
        derivative_order: int = 2,
    ) -> NaturalLikelihoodEvaluation:
        del derivative_order
        self.evaluate_calls += 1
        residual = np.asarray(y, dtype=np.float64) - theta[:, 0]
        weights = plan.weights.values
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=-0.5 * weights * residual**2,
            parameter_independent_carrier=np.zeros(len(y)),
            score=(weights * residual)[:, None],
            hessian_packed=(-weights)[:, None],
            valid=np.ones(len(y), dtype=np.bool_),
        )

    def expected_information_natural(
        self,
        theta: np.ndarray,
        plan: FamilyLikelihoodPlan,
    ) -> np.ndarray:
        if theta.shape != (len(plan.weights.values), 1):
            raise ValueError("theta must match the likelihood rows")
        return plan.weights.values[:, None]


class _ExplodingFeature:
    def build(self, x: np.ndarray, sample_weight: np.ndarray | None = None) -> GroupInfo:
        del x, sample_weight
        raise AssertionError("predictor geometry compiled before likelihood binding")

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)[:, None]

    def reconstruct(self, beta: np.ndarray) -> dict[str, float]:
        return {"coefficient": float(beta[0])}


@pytest.mark.parametrize(
    "name",
    [
        "ConfigurableDistributionalFamily",
        "DefaultPredictionFamily",
        "FitFailureDiagnosingFamily",
        "_validated_complete_fit_configuration",
        "_validated_derivative_order",
        "_validated_parameter_matrix",
    ],
)
def test_family_contract_module_declares_explicit_extension_boundaries(name: str) -> None:
    assert getattr(family_contracts, name, None) is not None


def test_public_package_exports_complete_fit_and_plan_validation_protocols() -> None:
    """Missing public protocols or structurally misclassified built-ins fail here."""

    assert "ConfigurableDistributionalFamily" in distributional.__all__
    assert "LikelihoodPlanValidatingFamily" in distributional.__all__
    built_ins = (
        GammaLS(),
        GaussianLS(),
        NegativeBinomialLS(),
        TweedieLSS(),
    )

    assert all(isinstance(family, ConfigurableDistributionalFamily) for family in built_ins)
    assert isinstance(built_ins[0], LikelihoodPlanValidatingFamily)
    assert not isinstance(built_ins[1], LikelihoodPlanValidatingFamily)
    assert isinstance(built_ins[2], LikelihoodPlanValidatingFamily)
    assert not isinstance(built_ins[3], LikelihoodPlanValidatingFamily)


def test_mandatory_family_protocol_does_not_require_prediction_or_capability_bag() -> None:
    family = _FakeFamily((_parameter("location"), _parameter("scale")))
    weights = resolve_likelihood_weights(
        np.ones(3),
        n_observations=3,
        contract=WeightContract(semantics="prior"),
    )
    plan = family.bind_likelihood(np.zeros(3), weights, COMPLETE_OBSERVATION)

    assert isinstance(plan, _FakePlan)
    assert isinstance(family, DistributionalFamily)
    assert not hasattr(family, "default_prediction")
    assert not hasattr(family, "capabilities")
    assert not isinstance(family, DefaultPredictionFamily)
    assert not isinstance(family, ExpectedInformationFamily)
    assert validate_family(family) == family.parameters


def test_four_method_solver_family_is_not_silently_a_complete_fit_family() -> None:
    family = _FakeFamily((_parameter("location"), _parameter("scale")))

    assert isinstance(family, DistributionalFamily)
    assert not isinstance(family, ConfigurableDistributionalFamily)
    with pytest.raises(
        TypeError,
        match="complete-fit.*ConfigurableDistributionalFamily.*to_config",
    ):
        _validated_complete_fit_configuration(family)


def test_complete_fit_configuration_requires_a_callable_method() -> None:
    class NonCallableConfiguration:
        to_config = {"type": "not-callable"}

    with pytest.raises(
        TypeError,
        match="complete-fit.*ConfigurableDistributionalFamily.*to_config",
    ):
        _validated_complete_fit_configuration(NonCallableConfiguration())


@pytest.mark.parametrize("raw", [{}, {1: "not-a-string-key"}, []])
def test_complete_fit_configuration_must_be_a_nonempty_string_keyed_mapping(
    raw: object,
) -> None:
    class Configured:
        def to_config(self) -> object:
            return raw

    with pytest.raises((TypeError, ValueError), match="to_config|configuration"):
        _validated_complete_fit_configuration(Configured())


def test_complete_fit_configuration_is_an_owned_read_only_snapshot() -> None:
    raw = {"type": "Configured", "nested": {"value": 1}}

    class Configured:
        def to_config(self) -> object:
            return raw

    snapshot = _validated_complete_fit_configuration(Configured())
    raw["nested"]["value"] = 2

    assert snapshot == {"type": "Configured", "nested": {"value": 1}}
    with pytest.raises(TypeError):
        snapshot["type"] = "Changed"


@pytest.mark.parametrize("value", [True, np.bool_(False), -1, 3, 1.5, "2"])
def test_derivative_order_validator_accepts_only_exact_orders_zero_through_two(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="zero through two"):
        _validated_derivative_order(value)


def test_parameter_matrix_validator_applies_every_declared_support() -> None:
    parameters = (
        _parameter("location"),
        ParameterSpec(
            "scale",
            IdentityLink(),
            "scale",
            ParameterSupport(lower=0.0),
            "observed",
        ),
    )

    with pytest.raises(ValueError, match="TestFamily.*scale.*support"):
        _validated_parameter_matrix(
            np.array([[0.0, 1.0], [2.0, 0.0]]),
            n_observations=2,
            parameters=parameters,
            family_name="TestFamily",
        )


def test_complete_observation_and_family_plan_are_explicit_immutable_contracts() -> None:
    weights = resolve_likelihood_weights(
        np.ones(3),
        n_observations=3,
        contract=WeightContract(semantics="prior"),
    )
    plan = _FakePlan(weights)

    assert COMPLETE_OBSERVATION == ObservationContract(kind="complete", schema_version=1)
    assert isinstance(plan, FamilyLikelihoodPlan)
    child = plan.take(np.array([2, 0], dtype=np.intp))
    assert child.weights.root_digest == plan.weights.root_digest
    assert child.plan_identifier != plan.plan_identifier


def test_nonunit_prior_refusal_precedes_geometry_initialization_and_evaluation() -> None:
    family = _FrequencyOnlyFamily()
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 6)})

    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior"):
        fit_dense_distributional(
            frame,
            np.linspace(0.0, 1.0, len(frame)),
            family=family,
            predictors=(Predictor("location", {"x": _ExplodingFeature()}),),
            sample_weight=np.array([0.5, 1.0, 1.5, 2.0, 0.75, 1.25]),
            weight_contract=WeightContract(semantics="prior"),
        )

    assert family.initialize_calls == 0
    assert family.evaluate_calls == 0


def test_frequency_only_family_accepts_unit_prior_through_explicit_equivalence() -> None:
    family = _FrequencyOnlyFamily()
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 6)})

    fitted = fit_dense_distributional(
        frame,
        np.array([-0.3, 0.1, 0.4, 0.8, 1.2, 1.6]),
        family=family,
        predictors=(Predictor("location", {}),),
        sample_weight=np.ones(len(frame)),
        weight_contract=WeightContract(semantics="prior"),
    )

    assert fitted.result.converged
    assert family.last_bound_row_law == "unit-prior-explicitly-equals-frequency/v1"
    assert family.initialize_calls > 0
    assert family.evaluate_calls > 0
    assert fitted.predict_parameters(frame).shape == (len(frame), 1)
    with pytest.raises(NotImplementedError, match="default prediction"):
        fitted.predict(frame)


def test_family_validation_rejects_duplicate_parameter_names() -> None:
    family = _FakeFamily((_parameter("location"), _parameter("location")))

    with pytest.raises(ValueError, match="duplicate.*location"):
        validate_family(family)


@pytest.mark.parametrize("name", ["", "has space", "location:scale", "scale#null"])
def test_parameter_name_must_be_nonempty_identifier(name: str) -> None:
    with pytest.raises(ValueError, match="parameter name"):
        _parameter(name)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_derivative_order": True}, "max_derivative_order"),
        ({"max_derivative_order": -1}, "max_derivative_order"),
        ({"expected_information": 1}, "expected_information"),
        ({"cdf": "yes"}, "cdf"),
    ],
)
def test_capabilities_reject_invalid_declarations(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _capabilities(**overrides)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower": np.nan},
        {"upper": np.inf},
        {"lower": 1.0, "upper": 1.0},
        {"lower_inclusive": 1},
    ],
)
def test_parameter_support_rejects_invalid_bounds(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ParameterSupport(**cast(Any, kwargs))


def test_parameter_support_returns_row_mask() -> None:
    support = ParameterSupport(lower=0.0, upper=1.0, lower_inclusive=False)

    np.testing.assert_array_equal(
        support.contains(np.array([-0.1, 0.0, 0.4, 1.0, 1.1])),
        np.array([False, False, True, False, False]),
    )


def test_initial_state_validates_shape_finiteness_and_owns_read_only_array() -> None:
    source = np.arange(8.0).reshape(4, 2)
    state = InitialParameterState(source)
    source[0, 0] = -100.0

    state.validate_shape(n_observations=4, k_parameters=2)
    assert state.theta[0, 0] == 0.0
    assert not state.theta.flags.writeable

    with pytest.raises(ValueError, match="shape"):
        state.validate_shape(n_observations=3, k_parameters=2)
    with pytest.raises(ValueError, match="two-dimensional"):
        InitialParameterState(np.ones(4))
    with pytest.raises(ValueError, match="finite"):
        InitialParameterState(np.array([[0.0, np.nan]]))


def test_natural_evaluation_owns_validated_raw_signed_hessian_contract() -> None:
    optimizing = np.array([-1.0, -2.0, -3.0])
    carrier = np.array([0.1, 0.2, 0.3])
    score = np.arange(6.0).reshape(3, 2)
    raw_signed_hessian = -np.arange(9.0).reshape(3, 3)
    valid = np.array([True, False, True])

    evaluation = NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=optimizing,
        parameter_independent_carrier=carrier,
        score=score,
        hessian_packed=raw_signed_hessian,
        valid=valid,
    )
    optimizing[:] = -100.0
    carrier[:] = 100.0
    score[0, 0] = -100.0
    raw_signed_hessian[0, 0] = 100.0

    assert evaluation.score[0, 0] == 0.0
    assert evaluation.hessian_packed[0, 0] == 0.0
    np.testing.assert_array_equal(evaluation.optimizing_log_likelihood, [-1.0, -2.0, -3.0])
    np.testing.assert_array_equal(evaluation.parameter_independent_carrier, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(evaluation.reported_log_likelihood, [-0.9, -1.8, -2.7])
    assert not evaluation.optimizing_log_likelihood.flags.writeable
    assert not evaluation.parameter_independent_carrier.flags.writeable
    assert not evaluation.reported_log_likelihood.flags.writeable
    assert not evaluation.hessian_packed.flags.writeable
    assert "raw signed Hessian" in (NaturalLikelihoodEvaluation.__doc__ or "")


@pytest.mark.parametrize("derivative_order", [0, 1, 2])
def test_natural_evaluation_owns_exact_optional_derivative_orders(
    derivative_order: int,
) -> None:
    """Kills dummy arrays, inferred wrong orders, and borrowed optional channels."""

    optimizing = np.array([-1.0, -2.0])
    carrier = np.array([0.25, 0.5])
    score = np.array([[1.0, 2.0], [3.0, 4.0]]) if derivative_order >= 1 else None
    hessian = np.arange(6.0).reshape(2, 3) if derivative_order == 2 else None
    valid = np.array([True, False])
    evaluation = NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=optimizing,
        parameter_independent_carrier=carrier,
        score=score,
        hessian_packed=hessian,
        valid=valid,
    )

    optimizing[:] = 100.0
    carrier[:] = 100.0
    valid[:] = True
    if score is not None:
        score[:] = 100.0
    if hessian is not None:
        hessian[:] = 100.0

    assert evaluation.derivative_order == derivative_order
    np.testing.assert_array_equal(evaluation.reported_log_likelihood, [-0.75, -1.5])
    np.testing.assert_array_equal(evaluation.valid, [True, False])
    assert not evaluation.optimizing_log_likelihood.flags.writeable
    assert not evaluation.parameter_independent_carrier.flags.writeable
    assert evaluation.score is None or not evaluation.score.flags.writeable
    assert evaluation.hessian_packed is None or not evaluation.hessian_packed.flags.writeable


def test_likelihood_delta_is_independent_of_available_derivative_order() -> None:
    """Kills value helpers that accidentally dereference absent derivatives."""

    reference = NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=np.array([1.0e16, -1.0e16]),
        parameter_independent_carrier=np.array([1.0e18, 1.0e18]),
        score=None,
        hessian_packed=None,
    )
    candidate = NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=np.array([1.0e16 + 4.0, -1.0e16 + 2.0]),
        parameter_independent_carrier=np.array([-1.0e18, -1.0e18]),
        score=np.zeros((2, 1)),
        hessian_packed=np.zeros((2, 1)),
    )

    assert candidate.log_likelihood_delta(reference) == 6.0


def test_likelihood_delta_uses_rowwise_optimizing_values_only() -> None:
    score = np.zeros((2, 1))
    hessian = np.zeros((2, 1))
    reference = NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=np.array([1.0e16, -1.0e16]),
        parameter_independent_carrier=np.array([1.0e18, 1.0e18]),
        score=score,
        hessian_packed=hessian,
    )
    candidate = NaturalLikelihoodEvaluation(
        optimizing_log_likelihood=np.array([1.0e16 + 4.0, -1.0e16 + 2.0]),
        parameter_independent_carrier=np.array([-1.0e18, -1.0e18]),
        score=score,
        hessian_packed=hessian,
    )

    assert candidate.log_likelihood_delta(reference) == 6.0


@pytest.mark.parametrize(
    ("optimizing", "carrier", "score", "hessian", "valid", "message"),
    [
        (np.zeros((2, 1)), np.zeros(2), np.zeros((2, 2)), np.zeros((2, 3)), None, "optimizing"),
        (np.zeros(2), np.zeros((2, 1)), np.zeros((2, 2)), np.zeros((2, 3)), None, "carrier"),
        (np.zeros(2), np.zeros(2), np.zeros(2), np.zeros((2, 1)), None, "score"),
        (np.zeros(2), np.zeros(2), np.zeros((3, 2)), np.zeros((2, 3)), None, "row"),
        (np.zeros(2), np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2)), None, "packed"),
        (np.zeros(2), np.zeros(2), np.zeros((2, 3)), np.zeros((2, 5)), None, "packed"),
        (np.zeros(2), np.zeros(2), np.zeros((2, 2)), np.zeros((2, 3)), np.ones(3, bool), "valid"),
        (np.array([0.0, np.nan]), np.zeros(2), np.zeros((2, 2)), np.zeros((2, 3)), None, "finite"),
        (
            np.zeros(2),
            np.zeros(2),
            np.array([[0.0, np.inf], [0.0, 0.0]]),
            np.zeros((2, 3)),
            None,
            "finite",
        ),
        (
            np.zeros(2),
            np.zeros(2),
            np.zeros((2, 1)),
            np.array([[np.nan], [0.0]]),
            None,
            "finite",
        ),
        (np.zeros(2), np.zeros(3), None, None, None, "row"),
        (np.zeros(2), np.zeros(2), None, np.zeros((2, 1)), None, "score"),
        (np.zeros(2), np.zeros(2), np.zeros((3, 1)), None, None, "row"),
    ],
)
def test_natural_evaluation_rejects_malformed_or_nonfinite_arrays(
    optimizing: np.ndarray,
    carrier: np.ndarray,
    score: np.ndarray | None,
    hessian: np.ndarray | None,
    valid: np.ndarray | None,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        NaturalLikelihoodEvaluation(optimizing, carrier, score, hessian, valid)
