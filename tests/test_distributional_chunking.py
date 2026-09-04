from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.family as family_contracts
import superglm.distributional.solver.chunks as chunking
from superglm import SuperLSS
from superglm._frame import as_eager_frame
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.negative_binomial import (
    NegativeBinomialLikelihoodPlan,
    NegativeBinomialLS,
)
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    FamilyLikelihoodPlan,
    NaturalLikelihoodEvaluation,
)
from superglm.distributional.inference import compute_joint_inference
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.packing import packed_pairs
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.result import DenseSolverConfig
from superglm.distributional.solver import fit_dense_fixed_lambda
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.features import Numeric, Spline

from ._distributional_weights import resolved_prior
from ._gaussian_lss_oracles import (
    GaussianFitCertificate,
    assert_gaussian_fit_parity,
    certify_gaussian_result,
    fixed_route_fixture,
)

_SEMANTIC_SOLVER_TOLERANCE = float(np.sqrt(np.finfo(np.float64).eps))


def _problem(*, n: int = 43, discrete: bool = True):
    rng = np.random.default_rng(6621)
    x = np.linspace(-1.0, 1.0, n)
    z = np.mod(0.17 + 1.31 * x, 1.0)
    frame = as_eager_frame(pd.DataFrame({"x": x, "z": z}))
    family = GaussianLS(scale_floor=0.02)
    weights = np.linspace(0.55, 1.75, n)
    location = 0.35 + 0.7 * x + 0.18 * np.sin(2.0 * np.pi * z)
    scale = family.scale_floor + np.exp(-1.25 + 0.3 * np.cos(2.0 * np.pi * z))
    response = location + rng.normal(scale=scale)
    compiled = compile_predictors(
        frame,
        resolved_prior(weights),
        family.parameters,
        (
            Predictor(
                "location",
                {
                    "x": Numeric(),
                    "z": Spline(kind="cr", n_knots=5, discrete=discrete),
                },
            ),
            Predictor(
                "scale",
                {"z": Spline(kind="cr", n_knots=4, discrete=discrete)},
            ),
        ),
        offsets={"location": 0.04 * np.sin(np.linspace(0.0, np.pi, n))},
        model_discrete=discrete,
        n_bins_config=17,
    )
    layout = build_stacked_layout(compiled)
    lambdas = {name: 0.35 + 0.1 * index for index, name in enumerate(layout.penalty_names)}
    penalty = layout.penalty_matrix(lambdas)
    initial = np.zeros(layout.n_coefficients)
    config = DenseSolverConfig(max_iterations=60, tolerance=1.0e-7)
    return family, layout, response, weights, penalty, initial, config


def _plan(family, response: np.ndarray, weights: np.ndarray):
    resolved = resolve_likelihood_weights(
        weights,
        n_observations=len(response),
        contract=WeightContract(semantics="prior"),
    )
    return family.bind_likelihood(response, resolved, COMPLETE_OBSERVATION)


class _PlanValidationAuditedFamily:
    """Delegate a real fit while counting its optional root-plan validation."""

    def __init__(self, base: GaussianLS, invalid_validation: str | None = None) -> None:
        self.base = base
        self.invalid_validation = invalid_validation
        self.validation_calls = 0
        self.evaluation_calls = 0

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def default_prediction_name(self):
        return self.base.default_prediction_name

    def bind_likelihood(self, y, weights, observation):
        return self.base.bind_likelihood(y, weights, observation)

    def validate_likelihood_plan(self, y, plan):
        self.validation_calls += 1
        response = np.array(y, dtype=np.float64, copy=True)
        response.setflags(write=False)
        if self.invalid_validation == "list":
            return response.tolist()
        if self.invalid_validation == "writable":
            return response.copy()
        if self.invalid_validation == "float32":
            return response.astype(np.float32)
        if self.invalid_validation == "wrong-shape":
            return response[:-1]
        if self.invalid_validation == "nonfinite":
            invalid = response.copy()
            invalid[0] = np.nan
            invalid.setflags(write=False)
            return invalid
        if self.invalid_validation == "raises":
            raise ValueError("malformed custom-family validation hook")
        return response

    def initialize(self, y, plan):
        return self.base.initialize(y, plan)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        self.evaluation_calls += 1
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


def _nb2_authority_problem():
    family = NegativeBinomialLS()
    weights = np.array([0.5, 0.75, 1.25, 1.5])
    counts = np.array([0.0, 3.0, 10.0, 18.0])
    response = counts / weights
    resolved = resolve_likelihood_weights(
        weights,
        n_observations=len(response),
        contract=WeightContract(semantics="prior"),
    )
    plan = family.bind_likelihood(response, resolved, COMPLETE_OBSERVATION)
    return family, response, plan


@dataclass(frozen=True)
class _CarrierSubstitutingChildPlan:
    """Keep exact positional lineage while changing only the hidden child carrier."""

    base: FamilyLikelihoodPlan
    child_carrier: float
    active_carrier: float = 0.0

    @property
    def weights(self) -> ResolvedLikelihoodWeights:
        return self.base.weights

    @property
    def plan_identifier(self) -> str:
        return self.base.plan_identifier

    def take(self, indices: np.ndarray) -> _CarrierSubstitutingChildPlan:
        return _CarrierSubstitutingChildPlan(
            base=self.base.take(indices),
            child_carrier=self.child_carrier,
            active_carrier=self.child_carrier,
        )


class _CarrierSubstitutingGaussian(GaussianLS):
    """Gaussian subtype whose opaque child executes an unreported fixed carrier."""

    def __init__(self, child_carrier: float = 5.0) -> None:
        super().__init__(scale_floor=0.02)
        object.__setattr__(self, "child_carrier", child_carrier)
        object.__setattr__(self, "bind_calls", 0)
        object.__setattr__(self, "kernel_calls", 0)

    def bind_likelihood(self, y, weights, observation):
        object.__setattr__(self, "bind_calls", self.bind_calls + 1)
        return _CarrierSubstitutingChildPlan(
            base=super().bind_likelihood(y, weights, observation),
            child_carrier=self.child_carrier,
        )

    def initialize(self, y, plan):
        assert isinstance(plan, _CarrierSubstitutingChildPlan)
        return super().initialize(y, plan.base)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        assert isinstance(plan, _CarrierSubstitutingChildPlan)
        object.__setattr__(self, "kernel_calls", self.kernel_calls + 1)
        evaluation = super().evaluate_natural(
            y,
            theta,
            plan.base,
            derivative_order=derivative_order,
        )
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=evaluation.optimizing_log_likelihood,
            parameter_independent_carrier=(
                evaluation.parameter_independent_carrier + plan.active_carrier
            ),
            score=evaluation.score,
            hessian_packed=evaluation.hessian_packed,
            valid=evaluation.valid,
        )

    def expected_information_natural(self, theta, plan):
        assert isinstance(plan, _CarrierSubstitutingChildPlan)
        return super().expected_information_natural(theta, plan.base)


def _history_values(result, name: str) -> np.ndarray:
    return np.asarray([getattr(item, name) for item in result.history], dtype=np.float64)


def test_auto_chunk_selector_has_stable_release_identity_and_budget() -> None:
    assert chunking.AUTO_CHUNK_SELECTOR == "distributional-auto-v1"
    assert chunking.AUTO_CHUNK_MEMORY_BYTES == 8_388_608
    assert (
        chunking.resolve_chunk_size(
            100_000,
            2,
            "auto",
            p_coefficients=25,
        )
        == 19_784
    )


@pytest.mark.parametrize(
    "chunk_size",
    [1, 11, 43, "auto"],
    ids=["one", "non-divisor", "all-rows", "auto"],
)
def test_chunk_sizes_match_dense_objective_geometry_steps_and_predictions(
    chunk_size: int | str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family, layout, response, weights, penalty, initial, config = _problem()
    reference = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, weights),
        penalty,
        initial=initial,
        config=config,
    )
    if chunk_size == "auto":
        monkeypatch.setattr(chunking, "AUTO_CHUNK_MEMORY_BYTES", 1_024)

    actual = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, weights),
        penalty,
        initial=initial,
        config=config,
        chunk_size=chunk_size,
    )

    assert actual.converged is reference.converged
    assert actual.convergence_reason == reference.convergence_reason
    assert actual.iterations == reference.iterations
    assert [item.backtracks for item in actual.history] == [
        item.backtracks for item in reference.history
    ]
    assert [item.rank for item in actual.history] == [item.rank for item in reference.history]
    np.testing.assert_allclose(
        actual.penalized_log_likelihood,
        reference.penalized_log_likelihood,
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        actual.terminal_score, reference.terminal_score, rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(actual.coefficients, reference.coefficients, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(actual.eta, reference.eta, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(actual.theta, reference.theta, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(
        _history_values(actual, "objective_after"),
        _history_values(reference, "objective_after"),
        rtol=2e-9,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        _history_values(actual, "step_scale"),
        _history_values(reference, "step_scale"),
        rtol=2e-8,
        atol=2e-9,
    )
    for left_index, right_index in packed_pairs(len(layout.predictors)):
        left = layout.predictors[left_index].coefficient_slice
        right = layout.predictors[right_index].coefficient_slice
        np.testing.assert_allclose(
            actual.terminal_data_curvature[left, right],
            reference.terminal_data_curvature[left, right],
            rtol=2e-8,
            atol=2e-9,
        )


def test_row_chunk_iterator_covers_rows_once_and_validates_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert [(item.start, item.stop) for item in chunking.iter_row_chunks(8, 3)] == [
        (0, 3),
        (3, 6),
        (6, 8),
    ]
    np.testing.assert_array_equal(
        np.concatenate([item.indices for item in chunking.iter_row_chunks(8, 3)]),
        np.arange(8),
    )
    monkeypatch.setattr(chunking, "AUTO_CHUNK_MEMORY_BYTES", 640)
    automatic = chunking.resolve_chunk_size(100, 2, "auto")
    assert 1 <= automatic < 100
    with pytest.raises(ValueError, match="chunk_size"):
        tuple(chunking.iter_row_chunks(8, 0))
    with pytest.raises(TypeError, match="chunk_size"):
        tuple(chunking.iter_row_chunks(8, True))


def test_public_superlss_admits_a_structural_gaussian_subclass_without_eager_binding() -> None:
    family = _CarrierSubstitutingGaussian()

    model = SuperLSS(
        family=family,
        predictors=(Predictor("location", {}), Predictor("scale", {})),
    )

    assert model.family is family
    assert family.bind_calls == 0
    assert family.kernel_calls == 0


def test_nb2_bound_likelihood_rejects_a_different_supplied_response() -> None:
    family, response, plan = _nb2_authority_problem()
    supplied_response = response.copy()
    supplied_response[1] += 1.0

    with pytest.raises(
        UnsupportedLikelihoodContractError,
        match="[Nn]egative-binomial|NegativeBinomial|NB2",
    ):
        chunking._validate_bound_likelihood(
            family,
            plan,
            supplied_response,
        )


@pytest.mark.parametrize(
    "indices",
    [
        np.array([0, 3], dtype=np.intp),
        np.array([3, 1], dtype=np.intp),
    ],
)
def test_nb2_take_is_the_exact_requested_positional_slice(indices: np.ndarray) -> None:
    _, response, plan = _nb2_authority_problem()

    child = plan.take(indices)

    assert type(child) is NegativeBinomialLikelihoodPlan
    np.testing.assert_array_equal(child.weights.values, plan.weights.values[indices])
    np.testing.assert_array_equal(child.weights.root_take_map, indices)
    np.testing.assert_array_equal(child.exact_response, response[indices])
    np.testing.assert_array_equal(child.exact_count, plan.exact_count[indices])
    np.testing.assert_array_equal(
        child.parameter_independent_carrier,
        plan.parameter_independent_carrier[indices],
    )
    assert child.plan_identifier != plan.plan_identifier


def test_both_chunk_routes_accept_ordinary_ordered_children() -> None:
    """The child gate must preserve normal ordered positional chunks."""

    family, layout, response, weights, penalty, coefficients, _ = _problem(n=7)
    plan = _plan(family, response, weights)

    likelihood = chunking.evaluate_chunked_log_likelihood(
        family,
        layout,
        response,
        plan,
        coefficients,
        chunk_size=3,
    )
    geometry = chunking.assemble_chunked_geometry(
        family,
        layout,
        response,
        plan,
        coefficients,
        penalty=penalty,
        chunk_size=3,
        curvature_source="fisher",
    )

    assert np.isfinite(likelihood.log_likelihood)
    assert geometry.data_curvature.shape == (layout.n_coefficients, layout.n_coefficients)
    np.testing.assert_array_equal(geometry.data_curvature, geometry.data_curvature.T)


def test_nb2_discrete_fit_refuses_through_the_generic_missing_information_gate() -> None:
    """Kills bypassing the generic chunking capability gate for one observed-only family."""

    family = NegativeBinomialLS()
    response = np.array([0.0, 1.0, 4.0, 2.0, 7.0, 3.0, 8.0, 5.0])
    weights = np.ones(len(response))
    x = np.linspace(-1.0, 1.0, len(response))
    frame = as_eager_frame(pd.DataFrame({"x": x}))
    compiled = compile_predictors(
        frame,
        resolved_prior(weights),
        family.parameters,
        (
            Predictor("mean", {"x": Spline(kind="cr", n_knots=4, discrete=True)}),
            Predictor("theta", {"x": Spline(kind="cr", n_knots=4, discrete=True)}),
        ),
        model_discrete=True,
        n_bins_config=8,
    )
    layout = build_stacked_layout(compiled)
    lambdas = {name: 0.5 for name in layout.penalty_names}
    penalty = layout.penalty_matrix(lambdas)
    plan = family.bind_likelihood(
        response,
        resolve_likelihood_weights(
            weights,
            n_observations=len(response),
            contract=WeightContract("prior"),
        ),
        COMPLETE_OBSERVATION,
    )

    with pytest.raises(
        ValueError,
        match="chunked fitting requires expected information capability",
    ):
        fit_dense_fixed_lambda(
            family,
            layout,
            response,
            plan,
            penalty,
            initial=np.zeros(layout.n_coefficients),
            config=DenseSolverConfig(
                max_iterations=2,
                tolerance=1.0e-7,
                coefficient_curvature="observed",
            ),
            chunk_size=3,
        )


@pytest.mark.parametrize("chunk_size", [None, 3], ids=["dense", "chunked"])
def test_fit_validates_bound_likelihood_once_before_evaluations(
    chunk_size: int | None,
) -> None:
    base, layout, response, weights, penalty, coefficients, _ = _problem(n=7)
    family = _PlanValidationAuditedFamily(base)
    plan = _plan(family, response, weights)

    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        penalty,
        initial=coefficients,
        config=DenseSolverConfig(
            max_iterations=2,
            tolerance=1.0e-7,
            coefficient_curvature="observed",
        ),
        chunk_size=chunk_size,
    )

    assert np.isfinite(result.penalized_log_likelihood)
    assert family.validation_calls == 1
    assert family.evaluation_calls > 1
    assert isinstance(family, family_contracts.LikelihoodPlanValidatingFamily)


@pytest.mark.parametrize("chunk_size", [None, 3], ids=["dense", "chunked"])
def test_fit_normalizes_invalid_family_likelihood_validation(
    chunk_size: int | None,
) -> None:
    base, layout, response, weights, penalty, coefficients, _ = _problem(n=7)
    messages: list[str] = []

    for invalid_validation in (
        "list",
        "writable",
        "float32",
        "wrong-shape",
        "nonfinite",
        "raises",
    ):
        family = _PlanValidationAuditedFamily(base, invalid_validation)
        plan = _plan(family, response, weights)

        with pytest.raises(UnsupportedLikelihoodContractError) as exc_info:
            fit_dense_fixed_lambda(
                family,
                layout,
                response,
                plan,
                penalty,
                initial=coefficients,
                config=DenseSolverConfig(
                    max_iterations=2,
                    tolerance=1.0e-7,
                    coefficient_curvature="observed",
                ),
                chunk_size=chunk_size,
            )

        messages.append(str(exc_info.value))
        assert family.validation_calls == 1
        assert family.evaluation_calls == 0

    assert set(messages) == {"family likelihood validation returned an invalid response"}


def test_chunk_route_leaves_the_gaussian_plan_contract_to_family_evaluation() -> None:

    family, layout, response, weights, _, coefficients, _ = _problem(n=7)
    plan = _plan(family, response, weights)
    object.__setattr__(
        plan,
        "family_config",
        ("GaussianLS/v1", family.scale_floor + 0.5),
    )

    with pytest.raises(UnsupportedLikelihoodContractError, match="configuration"):
        chunking.evaluate_chunked_log_likelihood(
            family,
            layout,
            response,
            plan,
            coefficients,
            chunk_size=3,
        )


def test_closing_chunk_iterator_releases_current_temporary_arrays() -> None:
    family, layout, response, weights, _, initial, _ = _problem(n=19)
    iterator = chunking.iter_likelihood_chunks(
        family,
        layout,
        response,
        _plan(family, response, weights),
        initial,
        chunk_size=5,
        curvature_source="observed",
    )
    current = next(iterator)
    references = tuple(
        weakref.ref(values)
        for values in (
            current.eta,
            current.theta,
            current.score_eta,
            current.curvature_packed,
        )
    )

    del current
    iterator.close()
    del iterator
    gc.collect()

    assert all(reference() is None for reference in references)


class _ExplodingGaussian:
    def __init__(self, *, explode_on: int) -> None:
        self.base = GaussianLS(scale_floor=0.02)
        self.explode_on = explode_on
        self.evaluations = 0

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def default_prediction_name(self):
        return self.base.default_prediction_name

    def bind_likelihood(self, y, weights, observation):
        return self.base.bind_likelihood(y, weights, observation)

    def initialize(self, y, plan):
        return self.base.initialize(y, plan)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        self.evaluations += 1
        if self.evaluations == self.explode_on:
            raise RuntimeError("chunk failure sentinel")
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


def test_chunk_exception_aborts_without_returning_a_partial_fit() -> None:
    _, layout, response, weights, penalty, initial, config = _problem(n=19)
    family = _ExplodingGaussian(explode_on=2)

    with pytest.raises(RuntimeError, match="chunk failure sentinel"):
        fit_dense_fixed_lambda(
            family,
            layout,
            response,
            _plan(family, response, weights),
            penalty,
            initial=initial,
            config=config,
            chunk_size=5,
        )

    assert family.evaluations == 2


class _ChildPlanAuditedGaussian:
    """Record only likelihood/information plans; all Gaussian work stays real."""

    def __init__(self) -> None:
        self.base = GaussianLS(scale_floor=0.0)
        self.evaluation_plans: list[tuple[object, int]] = []
        self.information_plans: list[tuple[object, int]] = []

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def default_prediction_name(self):
        return self.base.default_prediction_name

    def bind_likelihood(self, y, weights, observation):
        return self.base.bind_likelihood(y, weights, observation)

    def initialize(self, y, plan):
        return self.base.initialize(y, plan)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        self.evaluation_plans.append((plan, len(y)))
        return self.base.evaluate_natural(
            y,
            theta,
            plan,
            derivative_order=derivative_order,
        )

    def expected_information_natural(self, theta, plan):
        self.information_plans.append((plan, len(theta)))
        return self.base.expected_information_natural(theta, plan)

    def default_prediction(self, theta):
        return self.base.default_prediction(theta)


class _WrongDerivativeOrderGaussian(_ChildPlanAuditedGaussian):
    """Return a valid result at the opposite order from the chunk request."""

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        wrong_order = 2 if derivative_order == 0 else 0
        return self.base.evaluate_natural(
            y,
            theta,
            plan,
            derivative_order=wrong_order,
        )


@pytest.mark.parametrize(("route", "expected_order"), [("value", 0), ("geometry", 2)])
def test_chunk_consumers_refuse_a_family_returning_the_wrong_exact_order(
    route: str,
    expected_order: int,
) -> None:
    """Kills consumers that accept dummy or incomplete derivative levels."""

    base, layout, response, weights, penalty, coefficients, _ = _problem(n=7)
    del base
    family = _WrongDerivativeOrderGaussian()
    plan = _plan(family, response, weights)

    with pytest.raises(ValueError, match=rf"exact derivative order {expected_order}"):
        if route == "value":
            chunking.evaluate_chunked_log_likelihood(
                family,
                layout,
                response,
                plan,
                coefficients,
                chunk_size=3,
            )
        else:
            chunking.assemble_chunked_geometry(
                family,
                layout,
                response,
                plan,
                coefficients,
                penalty=penalty,
                chunk_size=3,
                curvature_source="observed",
            )


def _semantic_chunk_problem(*, semantics: str, expanded: bool):
    fixture = fixed_route_fixture()
    frame = pd.DataFrame({"x": fixture.x, "z": fixture.z})
    response = fixture.y
    weights = fixture.prior_weights if semantics == "prior" else fixture.frequency_counts
    if expanded:
        if semantics != "frequency":
            raise ValueError("only frequency rows have a literal expansion")
        take = np.repeat(np.arange(len(response)), weights.astype(np.intp))
        frame = frame.iloc[take].reset_index(drop=True)
        response = response[take]
        weights = np.ones(len(take), dtype=np.float64)
    resolved = resolve_likelihood_weights(
        weights,
        n_observations=len(response),
        contract=WeightContract(semantics=semantics),  # type: ignore[arg-type]
    )
    family = _ChildPlanAuditedGaussian()
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved,
        family.parameters,
        (
            Predictor(
                "location",
                {
                    "x": Spline(
                        kind="cr",
                        n_knots=4,
                        knot_strategy="quantile_rows",
                    )
                },
            ),
            Predictor("scale", {"z": Numeric()}),
        ),
    )
    layout = build_stacked_layout(compiled)
    penalty = layout.penalty_matrix({"location:x#wiggle": 0.6})
    plan = family.bind_likelihood(response, resolved, COMPLETE_OBSERVATION)
    config = DenseSolverConfig(tolerance=_SEMANTIC_SOLVER_TOLERANCE, coefficient_curvature="fisher")
    return family, layout, response, plan, penalty, config


def _certify_semantic_result(
    family,
    layout,
    response,
    plan,
    result,
) -> GaussianFitCertificate:
    inference = compute_joint_inference(layout, result)
    return certify_gaussian_result(
        layout,
        result,
        response,
        plan.weights.values,
        semantics=plan.weights.provenance.contract.semantics,
        covariance=inference.covariance,
        total_edf=inference.total_edf,
        inference_rank=inference.rank,
        scale_floor=family.base.scale_floor if hasattr(family, "base") else family.scale_floor,
    )


def _assert_only_ordered_positional_children(family, root_plan, *, chunk_size: int) -> None:
    root = root_plan.weights
    expected = np.arange(len(root.values), dtype=np.intp)
    chunks_per_pass = (len(expected) + chunk_size - 1) // chunk_size
    for records in (family.evaluation_plans, family.information_plans):
        assert records
        assert len(records) % chunks_per_pass == 0
        for plan, row_count in records:
            child = plan.weights
            assert child.root_digest == root.root_digest
            assert child.digest != root.digest
            assert plan.plan_identifier != root_plan.plan_identifier
            assert len(child.root_take_map) == row_count
            assert len(np.unique(child.root_take_map)) == row_count
            assert np.all(np.diff(child.root_take_map) > 0)
            np.testing.assert_array_equal(
                child.input_positions,
                root.input_positions[child.root_take_map],
            )
            np.testing.assert_array_equal(
                plan.parameter_independent_carrier,
                root_plan.parameter_independent_carrier[child.root_take_map],
            )
        for start in range(0, len(records), chunks_per_pass):
            batch = records[start : start + chunks_per_pass]
            np.testing.assert_array_equal(
                np.concatenate([plan.weights.root_take_map for plan, _ in batch]),
                expected,
            )


def _assert_semantic_fit_parity(
    left_result,
    right_result,
    left_certificate: GaussianFitCertificate,
    right_certificate: GaussianFitCertificate,
    *,
    left_take: np.ndarray | None = None,
    right_take: np.ndarray | None = None,
) -> None:
    left_positions = slice(None) if left_take is None else left_take
    right_positions = slice(None) if right_take is None else right_take
    assert_gaussian_fit_parity(
        left_result,
        right_result,
        left_certificate,
        right_certificate,
        left_eta=left_result.eta[left_positions],
        right_eta=right_result.eta[right_positions],
        left_prediction=left_result.theta[left_positions],
        right_prediction=right_result.theta[right_positions],
    )


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_chunked_fixed_lambda_matches_literal_oracle_before_dense_parity(
    semantics: str,
) -> None:
    """Kills a chunk evaluator that passes a raw root instead of positional children."""

    chunk_size = 4
    family, layout, response, plan, penalty, config = _semantic_chunk_problem(
        semantics=semantics,
        expanded=False,
    )
    chunked = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        penalty,
        config=config,
        chunk_size=chunk_size,
    )
    assert chunked.execution_backend_identifier == "distributional-chunked-v1"
    assert chunked.resolved_chunk_size == chunk_size
    assert chunked.family_likelihood_plan_identifier == plan.plan_identifier
    chunk_certificate = _certify_semantic_result(
        family,
        layout,
        response,
        plan,
        chunked,
    )
    _assert_only_ordered_positional_children(family, plan, chunk_size=chunk_size)

    dense_family = GaussianLS(scale_floor=0.0)
    dense = fit_dense_fixed_lambda(
        dense_family,
        layout,
        response,
        plan,
        penalty,
        config=config,
    )
    dense_certificate = _certify_semantic_result(
        dense_family,
        layout,
        response,
        plan,
        dense,
    )
    _assert_semantic_fit_parity(
        chunked,
        dense,
        chunk_certificate,
        dense_certificate,
    )

    if semantics == "frequency":
        expanded_family, expanded_layout, expanded_y, expanded_plan, expanded_penalty, _ = (
            _semantic_chunk_problem(semantics="frequency", expanded=True)
        )
        expanded = fit_dense_fixed_lambda(
            expanded_family,
            expanded_layout,
            expanded_y,
            expanded_plan,
            expanded_penalty,
            config=config,
            chunk_size=chunk_size,
        )
        expanded_certificate = _certify_semantic_result(
            expanded_family,
            expanded_layout,
            expanded_y,
            expanded_plan,
            expanded,
        )
        _assert_only_ordered_positional_children(
            expanded_family,
            expanded_plan,
            chunk_size=chunk_size,
        )
        take = np.repeat(
            np.arange(len(response)),
            plan.weights.values.astype(np.intp),
        )
        first_expanded = np.flatnonzero(np.r_[True, np.diff(take) != 0])
        np.testing.assert_array_equal(
            first_expanded,
            np.unique(take, return_index=True)[1],
        )
        _assert_semantic_fit_parity(
            chunked,
            expanded,
            chunk_certificate,
            expanded_certificate,
            right_take=first_expanded,
        )
