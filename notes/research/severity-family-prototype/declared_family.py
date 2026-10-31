"""One generic SuperLSS family adapter for every declared family.

Prototype for notes/research/2026-10-31-severity-family-framework.md; not production code.

``DeclaredFamily(name)`` reads its declaration and generated kernel from
:data:`REGISTRY`, so the family object itself stays a small frozen dataclass
that SuperLSS can snapshot.  Every parameter is positive with a log link; the
law is not reproductive, so non-unit prior weights are refused and frequency
weights replicate rows, as ``LogNormalLS`` does.  The carrier rides in the
optimizing value (the plan reports a zero carrier), which keeps the plan's
identifier to exactly stored inputs.  Distribution functions and the mean come
from scipy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from superglm.distributional import (
    COMPLETE_OBSERVATION,
    InitialParameterState,
    NaturalLikelihoodEvaluation,
    ParameterSpec,
    ParameterSupport,
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import LogLink

REGISTRY: dict[str, tuple] = {}


def register(declaration, kernel, scipy_fit: Callable[[np.ndarray], tuple[float, ...]]) -> None:
    REGISTRY[declaration.name] = (declaration, kernel, scipy_fit)


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


@dataclass(frozen=True, eq=False)
class DeclaredPlan:
    weights: ResolvedLikelihoodWeights
    response: np.ndarray
    carrier: np.ndarray
    tag: str

    @property
    def plan_identifier(self) -> str:
        return f"{self.tag}:{self.weights.digest}:{_digest(self.response)}"

    @property
    def parameter_independent_carrier(self) -> np.ndarray:
        return np.zeros(len(self.response))

    def take(self, indices) -> DeclaredPlan:
        return DeclaredPlan(
            self.weights.take(indices), self.response[indices], self.carrier[indices], self.tag
        )


@dataclass(frozen=True)
class DeclaredFamily:
    name: str

    @property
    def _entry(self):
        return REGISTRY[self.name]

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        declaration = self._entry[0]
        return tuple(
            ParameterSpec(
                name=parameter,
                default_link=LogLink(),
                role=parameter,
                support=ParameterSupport(lower=0.0),
                curvature="observed",
            )
            for parameter in declaration.parameters
        )

    def to_config(self) -> dict[str, object]:
        return {"type": "DeclaredFamily", "declaration": self.name}

    def bind_likelihood(self, y, weights, observation) -> DeclaredPlan:
        response = np.asarray(y, dtype=np.float64)
        if observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError("complete observations only in this prototype")
        if not np.all(np.isfinite(response)) or np.any(response <= 0.0):
            raise ValueError("the response must be finite and positive")
        semantics = weights.provenance.contract.semantics
        if semantics == "prior" and np.any(weights.values != 1.0):
            raise UnsupportedLikelihoodContractError(
                f"{self.name} is not reproductive; non-unit prior weights have no row law"
            )
        carrier = self._entry[1].carrier(response, weights.values)
        return DeclaredPlan(weights, response, carrier, f"Declared/{self.name}/v0")

    def initialize(self, y, plan) -> InitialParameterState:
        response = np.asarray(y, dtype=np.float64)
        sample = response[:: max(1, len(response) // 20_000)]
        start = np.asarray(self._entry[2](sample), dtype=np.float64)
        return InitialParameterState(np.tile(start, (len(response), 1)))

    def evaluate_natural(self, y, theta, plan, *, derivative_order: int = 2):
        order = int(derivative_order)
        value, score, hessian = self._entry[1].evaluate(y, theta, plan.weights.values, order)
        value = value + plan.carrier
        valid = np.isfinite(value)
        for channel in (score, hessian):
            if channel is not None:
                valid &= np.all(np.isfinite(channel), axis=1)
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=value,
            parameter_independent_carrier=plan.parameter_independent_carrier,
            score=score,
            hessian_packed=hessian,
            valid=valid,
        )

    @property
    def default_prediction_name(self) -> str:
        return "mean"

    def _law(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        return self._entry[0].scipy_law(*(theta[:, j] for j in range(theta.shape[1])))

    def default_prediction(self, theta) -> np.ndarray:
        return np.asarray(self._law(theta).mean(), dtype=np.float64)

    def cdf(self, y, theta) -> np.ndarray:
        return np.asarray(self._law(theta).cdf(y), dtype=np.float64)

    def quantile(self, p, theta) -> np.ndarray:
        return np.asarray(self._law(theta).ppf(p), dtype=np.float64)
