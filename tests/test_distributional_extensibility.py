from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from superglm import SuperLSS
from superglm.distributional import (
    InitialParameterState,
    NaturalLikelihoodEvaluation,
    ParameterSpec,
    ParameterSupport,
    Predictor,
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import IdentityLink


def _spec(name: str) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        default_link=IdentityLink(),
        role=name,
        support=ParameterSupport(),
        curvature="observed",
    )


@dataclass(frozen=True)
class _FourParameterPlan:
    weights: ResolvedLikelihoodWeights

    @property
    def plan_identifier(self) -> str:
        return f"synthetic-four/v1:{self.weights.digest}"

    def take(self, indices: np.ndarray) -> _FourParameterPlan:
        return _FourParameterPlan(self.weights.take(indices))


@dataclass(frozen=True)
class _FourParameterFamily:
    parameters = tuple(_spec(name) for name in ("a", "b", "c", "d"))

    def to_config(self) -> dict[str, object]:
        return {"type": "SyntheticFourParameter", "parameters": 4}

    def bind_likelihood(
        self,
        y: np.ndarray,
        weights: ResolvedLikelihoodWeights,
        observation: object,
    ) -> _FourParameterPlan:
        response = np.asarray(y, dtype=np.float64)
        if (
            getattr(observation, "kind", None) != "complete"
            or getattr(observation, "schema_version", None) != 1
        ):
            raise UnsupportedLikelihoodContractError("complete observations are required")
        if response.shape != weights.values.shape or not np.all(np.isfinite(response)):
            raise ValueError("response and resolved weights must contain the same finite rows")
        return _FourParameterPlan(weights)

    @staticmethod
    def _targets(y: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                y,
                np.full(len(y), 0.25),
                np.full(len(y), -0.5),
                np.full(len(y), 1.0),
            )
        )

    def initialize(
        self,
        y: np.ndarray,
        plan: object,
    ) -> InitialParameterState:
        assert isinstance(plan, _FourParameterPlan)
        return InitialParameterState(self._targets(np.asarray(y, dtype=np.float64)))

    def evaluate_natural(
        self,
        y: np.ndarray,
        theta: np.ndarray,
        plan: object,
        *,
        derivative_order: int = 2,
    ) -> NaturalLikelihoodEvaluation:
        assert isinstance(plan, _FourParameterPlan)
        if (
            isinstance(derivative_order, bool | np.bool_)
            or not isinstance(derivative_order, int | np.integer)
            or int(derivative_order) not in (0, 1, 2)
        ):
            raise ValueError("derivative_order must be an integer from zero through two")
        order = int(derivative_order)
        targets = self._targets(np.asarray(y, dtype=np.float64))
        residual = targets - np.asarray(theta, dtype=np.float64)
        weights = plan.weights.values
        score = None if order == 0 else weights[:, None] * residual
        hessian = None
        if order == 2:
            hessian = np.zeros((len(targets), 10), dtype=np.float64)
            hessian[:, (0, 4, 7, 9)] = -weights[:, None]
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=-0.5 * weights * np.sum(residual * residual, axis=1),
            parameter_independent_carrier=np.zeros(len(targets)),
            score=score,
            hessian_packed=hessian,
            valid=np.ones(len(targets), dtype=bool),
        )


def _fitted_four_parameter_model() -> tuple[SuperLSS, pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame({"row": np.linspace(-1.0, 1.0, 12)})
    response = np.linspace(-0.8, 1.2, len(frame))
    model = SuperLSS(
        family=_FourParameterFamily(),
        predictors=tuple(Predictor(name, {}) for name in ("a", "b", "c", "d")),
    ).fit(frame, response, lambdas={})
    return model, frame, response


def test_public_fixed_fit_accepts_four_parameters_and_ten_curvature_channels() -> None:
    model, frame, response = _fitted_four_parameter_model()

    parameters = model.predict_parameters(frame).to_numpy()
    assert parameters.shape == (len(frame), 4)
    assert model.training_telemetry().curvature_channels == 10
    np.testing.assert_allclose(parameters[:, 0], np.mean(response), rtol=0.0, atol=1.0e-8)
    expected_constants = np.tile([0.25, -0.5, 1.0], (len(frame), 1))
    np.testing.assert_allclose(parameters[:, 1:], expected_constants, rtol=0.0, atol=1.0e-8)


def test_optional_prediction_does_not_block_structural_fixed_fit() -> None:
    model, frame, _ = _fitted_four_parameter_model()

    with pytest.raises(NotImplementedError, match="default prediction"):
        model.predict(frame)


def test_custom_family_fit_predicts_but_public_save_refuses() -> None:
    """A fitted extension remains usable, but is not a native persistence type."""
    model, frame, response = _fitted_four_parameter_model()
    parameters = model.predict_parameters(frame).to_numpy()
    np.testing.assert_allclose(parameters[:, 0], np.mean(response), rtol=0.0, atol=1.0e-8)

    with pytest.raises(ValueError, match="built-in"):
        model.to_bytes()
