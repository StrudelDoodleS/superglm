from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import superglm.distributional.efs as efs_module
import superglm.distributional.smoothing.authority as smoothing_authority
import superglm.distributional.smoothing.evidence as smoothing_evidence
import superglm.distributional.smoothing.loop as smoothing_loop
import superglm.distributional.smoothing.objective as smoothing_objective
import superglm.reml.efs as scalar_efs
import superglm.solvers.rank as rank_module
from superglm._predictor_compiler import CompiledPredictorDesign
from superglm.distributional.efs_acceleration import (
    MultisecantDecision,
    MultisecantProposal,
    WindowedTypeIIAnderson,
)
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import (
    FamilyLikelihoodPlan,
    NaturalLikelihoodEvaluation,
    ObservationContract,
)
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.predictor import CompiledPredictor, Predictor
from superglm.distributional.result import (
    DenseSolverConfig,
    DistributionalEFSConfig,
    DistributionalEFSResult,
    EndpointDirectionEvidence,
)
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.distributional.weights import ResolvedLikelihoodWeights, WeightContract
from superglm.features import Numeric, RandomEffect, Spline
from superglm.group_matrix import (
    DesignMatrix,
    FactorSmoothGroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.links import IdentityLink
from superglm.model.reml_setup import collect_reml_groups
from superglm.reml.efs_update import (
    EFSComponentState,
    EFSUpdateResult,
    wood_fasiolo_update,
)
from superglm.reml.penalty_algebra import (
    build_penalty_components,
    compute_logdet_s_derivatives,
    compute_logdet_s_plus,
)
from superglm.types import GroupSlice, LambdaPolicy

from ._gaussian_lss_oracles import (
    GaussianFitCertificate,
    _hessian_drift_bound,
    certify_gaussian_result,
    coefficient_oracle,
    gamma,
    local_root_certificate,
    oracle_bounds,
)

# The local-mode fixed-point enclosure transports coefficient error through H
# and its inverse; eps**(2/3) keeps that channel subordinate to the outer rule.
_EFS_INNER_TOLERANCE = float(np.finfo(np.float64).eps ** (2.0 / 3.0))
_EFS_REPLICATION_INNER_TOLERANCE = float(128.0 * np.finfo(np.float64).eps)
_EFS_CARRIER_PER_ROW = 1.0e8
_EFS_LAMBDA_NAME = "scale:z#wiggle"


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("analytic_derivative", 0.0),
        ("profile_score_term", 0.0),
        ("numerical_error", 0.0),
        ("lower_bound", 1.0),
        ("upper_bound", 2.0),
    ],
)
def test_endpoint_direction_evidence_rejects_forged_derived_authority(
    field_name: str,
    replacement: object,
) -> None:
    """Kills accepting endpoint bounds inconsistent with their analytic terms."""
    evidence = EndpointDirectionEvidence(
        authority_identifier="analytic-observed-curvature-direction/v1",
        decision="endpoint",
        endpoint_objective=0.125,
        analytic_derivative=1.5,
        profile_score_term=1.0,
        curvature_schur_term=2.0,
        curvature_drift_term=0.0,
        numerical_error=0.1,
        lower_bound=1.4,
        upper_bound=1.6,
    )

    with pytest.raises(ValueError, match="derived endpoint direction evidence"):
        replace(evidence, **{field_name: replacement})


@dataclass(frozen=True)
class _SemanticEFSFixture:
    frame: pd.DataFrame
    response: np.ndarray
    prior_weights: np.ndarray
    frequency_counts: np.ndarray


@dataclass(frozen=True)
class _EFSStateCertificate:
    fit_index: int
    lambda_value: float
    fixed: GaussianFitCertificate
    laml: float
    laml_bound: float
    penalty_rank: int
    fixed_point_log_residual: float
    fixed_point_bound: float
    fixed_point_map_bound: float
    returned_fixed_point_map_bound: float
    recovered_penalty: _RecoveredPenalty


@dataclass(frozen=True)
class _RecoveredPenalty:
    matrix: np.ndarray
    elementwise_error: np.ndarray
    spectral_error: float


@dataclass(frozen=True)
class _SpectralLogdetCertificate:
    rank: int
    logdet: float
    logdet_error: float
    eigensolver_error: float
    positive_lower: np.ndarray
    null_upper: float


@dataclass(frozen=True)
class _LogdetPairCertificate:
    candidate: _SpectralLogdetCertificate
    reference: _SpectralLogdetCertificate
    total_bound: float


@dataclass(frozen=True)
class _ProductionLogdetCertificate:
    rank: int
    logdet: float
    total_bound: float


@dataclass(frozen=True)
class _CarrierPlan:
    base: FamilyLikelihoodPlan
    carrier_per_row: float

    @property
    def weights(self) -> ResolvedLikelihoodWeights:
        return self.base.weights

    @property
    def plan_identifier(self) -> str:
        return f"efs-carrier:{self.carrier_per_row}:{self.base.plan_identifier}"

    def take(self, indices: np.ndarray) -> _CarrierPlan:
        return _CarrierPlan(self.base.take(indices), self.carrier_per_row)


class _ConstantCarrierGaussian:
    """Test-only family wrapper changing only the fixed row carrier."""

    def __init__(self, carrier_per_row: float) -> None:
        self.base = GaussianLS(scale_floor=0.0)
        self.carrier_per_row = float(carrier_per_row)

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def default_prediction_name(self):
        return self.base.default_prediction_name

    @property
    def scale_floor(self) -> float:
        return self.base.scale_floor

    def to_config(self) -> dict[str, object]:
        return self.base.to_config()

    def bind_likelihood(
        self,
        y: np.ndarray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> _CarrierPlan:
        return _CarrierPlan(
            self.base.bind_likelihood(y, weights, observation),
            self.carrier_per_row,
        )

    def initialize(self, y, plan):
        assert isinstance(plan, _CarrierPlan)
        return self.base.initialize(y, plan.base)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        assert isinstance(plan, _CarrierPlan)
        evaluation = self.base.evaluate_natural(
            y,
            theta,
            plan.base,
            derivative_order=derivative_order,
        )
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=evaluation.optimizing_log_likelihood,
            parameter_independent_carrier=(
                evaluation.parameter_independent_carrier + plan.carrier_per_row
            ),
            score=evaluation.score,
            hessian_packed=evaluation.hessian_packed,
            valid=evaluation.valid,
        )

    def expected_information_natural(self, theta, plan):
        assert isinstance(plan, _CarrierPlan)
        return self.base.expected_information_natural(theta, plan.base)

    def default_prediction(self, theta):
        return self.base.default_prediction(theta)


def _semantic_efs_fixture() -> _SemanticEFSFixture:
    rng = np.random.default_rng(1823)
    x = np.linspace(0.0, 1.0, 72)
    z = np.mod(0.37 + 1.7 * x, 1.0)
    sigma = 0.22 + np.exp(-1.2 + 0.35 * np.cos(2.0 * np.pi * z))
    response = 0.35 + 0.8 * np.sin(2.0 * np.pi * x) + rng.normal(scale=sigma)
    prior_weights = 0.35 + 2.0 * np.mod(
        np.arange(len(x), dtype=np.float64) * 0.6180339887498949,
        1.0,
    )
    return _SemanticEFSFixture(
        frame=pd.DataFrame({"x": x, "z": z}),
        response=response,
        prior_weights=prior_weights,
        frequency_counts=np.resize(np.array([1.0, 2.0]), len(x)),
    )


def _semantic_efs_predictors() -> tuple[Predictor, Predictor]:
    return (
        Predictor("location", {"x": Numeric()}),
        Predictor(
            "scale",
            {
                "z": Spline(
                    kind="cr",
                    n_knots=5,
                    knot_strategy="quantile_rows",
                )
            },
        ),
    )


def _semantic_efs_config(
    *,
    tolerance: float,
    objective_tolerance: float,
) -> DistributionalEFSConfig:
    return DistributionalEFSConfig(
        outer="efs",
        max_iterations=120,
        tolerance=tolerance,
        max_log_step=1.0,
        objective_tolerance=objective_tolerance,
        plateau_tolerance=0.0,
        plateau_iterations=120,
    )


def _fit_semantic_efs(
    *,
    family,
    semantics: str,
    frame: pd.DataFrame,
    response: np.ndarray,
    weights: np.ndarray,
    efs_config: DistributionalEFSConfig,
    inner_tolerance: float = _EFS_INNER_TOLERANCE,
):
    return fit_dense_distributional(
        frame,
        response,
        family=family,
        weight_contract=WeightContract(semantics=semantics),  # type: ignore[arg-type]
        predictors=_semantic_efs_predictors(),
        sample_weight=weights,
        lambdas={_EFS_LAMBDA_NAME: 0.3},
        # The replication and all-one contract scenarios were derived on the Fisher
        # path; observed Newton closes the optimizing-log-likelihood gap they rely on.
        config=DenseSolverConfig(
            tolerance=inner_tolerance,
            max_iterations=100,
            coefficient_curvature="fisher",
        ),
        efs_config=efs_config,
    )


def _semantic_efs_penalty_group_matrix(model) -> SparseSSPGroupMatrix:
    groups = tuple(
        group
        for predictor in model.compiled_predictors
        for group in predictor.compiled.design.group_matrices
        if isinstance(group, SparseSSPGroupMatrix)
    )
    assert len(groups) == 1
    return groups[0]


def test_fresh_efs_evidence_reuses_one_terminal_inverse(monkeypatch) -> None:
    """Kills repeating the q-by-q inverse solve in one ordinary EFS update."""
    fixture = _semantic_efs_fixture()
    model = fit_dense_distributional(
        fixture.frame,
        fixture.response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=_semantic_efs_predictors(),
        sample_weight=fixture.prior_weights,
        lambdas={_EFS_LAMBDA_NAME: 0.3},
        config=DenseSolverConfig(tolerance=1.0e-10, max_iterations=80),
    )
    calls = 0
    original = rank_module.RankDecomposition.pseudo_inverse

    def counted(decomposition):
        nonlocal calls
        calls += 1
        return original(decomposition)

    monkeypatch.setattr(rank_module.RankDecomposition, "pseudo_inverse", counted)
    evidence = efs_module._fresh_raw_evidence(
        model.layout,
        model.lambdas,
        model.result,
        DistributionalEFSConfig(outer="efs"),
    )

    assert evidence.update is not None
    assert calls == 1


def _digest_field(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


def _digest_array(digest, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    _digest_field(digest, array.dtype.str.encode("ascii"))
    _digest_field(digest, repr(array.shape).encode("ascii"))
    _digest_field(digest, array.tobytes(order="C"))


def _expected_gaussian_plan_identifier(
    weights: np.ndarray,
    *,
    semantics: str,
    carrier_wrapper: float | None = None,
) -> str:
    """Fixture-owned root identity, independently hashed from literal inputs."""

    values = np.ascontiguousarray(weights, dtype=np.float64)
    positions = np.arange(len(values), dtype=np.intp)
    dropped = np.array([], dtype=np.intp)
    root = hashlib.sha256()
    for field in (
        b"superglm-likelihood-weights-root",
        b"1",
        semantics.encode("ascii"),
        str(len(values)).encode("ascii"),
    ):
        _digest_field(root, field)
    _digest_array(root, positions)
    _digest_array(root, dropped)
    _digest_array(root, values)
    weight_digest = root.hexdigest()

    carrier = 0.5 * np.log(values) if semantics == "prior" else np.zeros(len(values))
    carrier = np.ascontiguousarray(carrier, dtype=np.float64)
    carrier_hash = hashlib.sha256()
    carrier_hash.update(b"GaussianLS/carrier/v1\0")
    carrier_hash.update(carrier.dtype.str.encode("ascii"))
    carrier_hash.update(repr(carrier.shape).encode("ascii"))
    carrier_hash.update(memoryview(carrier).cast("B"))
    row_law, invariant = (
        ("normal-variance-sigma2-over-w/v1", "conditional-location")
        if semantics == "prior"
        else ("normal-literal-replication/v1", "literal-row-replication")
    )
    payload = "\0".join(
        (
            "GaussianLS/v1",
            row_law,
            invariant,
            "GaussianLS/v1",
            repr(0.0),
            "complete",
            "1",
            weight_digest,
            carrier_hash.hexdigest(),
        )
    ).encode("utf-8")
    base = f"GaussianLS/v1:{hashlib.sha256(payload).hexdigest()}"
    return base if carrier_wrapper is None else f"efs-carrier:{carrier_wrapper}:{base}"


def _up(value: float) -> float:
    numeric = float(value)
    assert math.isfinite(numeric) and numeric >= 0.0
    return float(np.nextafter(numeric, np.inf))


def _down(value: float) -> float:
    numeric = float(value)
    assert math.isfinite(numeric)
    return float(np.nextafter(numeric, -np.inf))


def _bound_sum(*terms: float, operations: int = 32) -> float:
    values = np.asarray(terms, dtype=np.float64)
    assert np.all(np.isfinite(values)) and np.all(values >= 0.0)
    raw = float(np.sum(values, dtype=np.float64))
    return _up(raw + gamma(max(operations, 1)) * max(1.0, raw))


def _array_bound_sum(*terms: np.ndarray, operations: int) -> np.ndarray:
    arrays = tuple(np.asarray(term, dtype=np.float64) for term in terms)
    raw = np.sum(np.stack(np.broadcast_arrays(*arrays)), axis=0, dtype=np.float64)
    assert np.all(np.isfinite(raw)) and np.all(raw >= 0.0)
    upward = raw + gamma(max(operations, 1)) * np.maximum(1.0, raw)
    return np.nextafter(upward, np.inf)


def _frobenius_up(matrix: np.ndarray, *, operations: int) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    raw = float(np.linalg.norm(values, ord="fro"))
    scale = max(1.0, raw, float(np.sum(np.abs(values), dtype=np.float64)))
    return _up(raw + gamma(max(operations, 1)) * scale)


def _spectral_upper(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    return _frobenius_up(values, operations=max(64 * values.size, 1))


def _absolute_product_upper(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    inner = left_values.shape[-1]
    denominator = 1.0 - gamma(max(inner, 1))
    assert denominator > 0.0
    product = np.abs(left_values) @ np.abs(right_values)
    upper = np.nextafter(product / denominator, np.inf)
    assert np.all(np.isfinite(upper))
    return upper


def _covariance_residual_bound(curvature: np.ndarray, covariance: np.ndarray) -> float:
    """Frobenius backward residual including the rounded matrix product."""

    hessian = np.asarray(curvature, dtype=np.float64)
    candidate = np.asarray(covariance, dtype=np.float64)
    width = len(hessian)
    residual = float(np.linalg.norm(hessian @ candidate - np.eye(width), ord="fro"))
    product_scale = _spectral_upper(np.abs(hessian) @ np.abs(candidate))
    return _bound_sum(
        residual,
        gamma(max(64 * width * width, 1)) * max(1.0, product_scale),
        operations=max(64 * width * width, 1),
    )


def _spectral_logdet(
    matrix: np.ndarray,
    *,
    full_rank: bool,
) -> _SpectralLogdetCertificate:
    """Return a threshold-separated positive-spectrum log determinant."""

    values = np.asarray(matrix, dtype=np.float64)
    symmetric = 0.5 * (values + values.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    width = len(eigenvalues)
    scale = max(1.0, _spectral_upper(symmetric))
    resolution = _up(gamma(max(256 * width, 1)) * scale)
    assert float(np.min(eigenvalues, initial=0.0)) >= -resolution
    positive = eigenvalues[eigenvalues > resolution]
    if full_rank:
        assert len(positive) == width
    else:
        assert 0 < len(positive) < width
    smallest = float(np.min(positive))
    assert smallest > 1024.0 * resolution
    positive_lower = np.nextafter(positive - resolution, -np.inf)
    assert np.all(positive_lower > 0.0)
    null = eigenvalues[eigenvalues <= resolution]
    null_upper = float(np.max(np.abs(null), initial=0.0)) + resolution
    logs = np.log(positive)
    eigen_log_error = float(np.sum(-np.log1p(-resolution / positive_lower)))
    summation_error = gamma(max(32 * len(positive), 1)) * max(
        1.0,
        float(np.sum(np.abs(logs), dtype=np.float64)),
    )
    return _SpectralLogdetCertificate(
        rank=len(positive),
        logdet=float(np.sum(logs, dtype=np.float64)),
        logdet_error=_bound_sum(eigen_log_error, summation_error),
        eigensolver_error=resolution,
        positive_lower=np.array(positive_lower, copy=True),
        null_upper=_up(null_upper),
    )


def _certify_logdet_pair(
    candidate_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    *,
    matrix_bound: float,
    full_rank: bool,
    label: str,
) -> _LogdetPairCertificate:
    """Propagate a derived Frobenius bound through Weyl and both eigensolvers."""

    candidate_values = np.asarray(candidate_matrix, dtype=np.float64)
    reference_values = np.asarray(reference_matrix, dtype=np.float64)
    assert candidate_values.shape == reference_values.shape
    width = len(reference_values)
    observed = _frobenius_up(
        candidate_values - reference_values,
        operations=max(64 * width * width, 1),
    )
    if observed > matrix_bound:
        raise AssertionError(
            f"{label} matrix perturbation exceeds its derived Frobenius bound: "
            f"observed={observed}, bound={matrix_bound}"
        )
    candidate = _spectral_logdet(candidate_values, full_rank=full_rank)
    reference = _spectral_logdet(reference_values, full_rank=full_rank)
    assert candidate.rank == reference.rank
    delta = _bound_sum(
        matrix_bound,
        candidate.eigensolver_error,
        reference.eigensolver_error,
        operations=max(64 * width, 1),
    )
    smallest = float(np.min(reference.positive_lower))
    assert delta < smallest, f"{label} positive spectrum reaches zero"
    if not full_rank:
        assert max(candidate.null_upper, reference.null_upper) + delta < smallest, (
            f"{label} positive and null penalty spectra are not separated"
        )
    weyl_log = float(np.sum(-np.log1p(-delta / reference.positive_lower)))
    total = _bound_sum(
        weyl_log,
        candidate.logdet_error,
        reference.logdet_error,
        gamma(max(32 * reference.rank, 1)) * max(1.0, abs(candidate.logdet), abs(reference.logdet)),
        operations=max(64 * reference.rank, 1),
    )
    return _LogdetPairCertificate(
        candidate=candidate,
        reference=reference,
        total_bound=total,
    )


def _certify_cholesky_logdet(
    matrix: np.ndarray,
    decomposition,
) -> _ProductionLogdetCertificate:
    """Certify the stored equilibrated-Cholesky log determinant."""

    values = np.asarray(matrix, dtype=np.float64)
    width = len(values)
    assert values.shape == (width, width)
    assert decomposition.method == "cholesky"
    assert decomposition.rank == width
    np.testing.assert_array_equal(
        np.asarray(decomposition.active_columns, dtype=np.intp),
        np.arange(width, dtype=np.intp),
    )
    assert decomposition.rank_truncated is False
    assert not np.any(decomposition.structural_aliases)
    factor = np.tril(np.asarray(decomposition.cholesky_factor, dtype=np.float64))
    scales = np.asarray(decomposition.column_scale, dtype=np.float64)
    assert factor.shape == values.shape
    assert scales.shape == (width,)
    assert np.all(scales > 0.0)
    assert np.all(np.diag(factor) > 0.0)
    symmetric = 0.5 * (values + values.T)
    np.testing.assert_array_equal(scales, np.sqrt(np.diag(symmetric)))

    # Reconstruct in original coordinates so the a-posteriori residual covers
    # both equilibration and factorization.  Its outward evaluation terms are
    # kept separate from the observed residual rather than fitted to it.
    scaled_factor = scales[:, None] * factor
    reconstructed = scaled_factor @ scaled_factor.T
    absolute_product = _absolute_product_upper(scaled_factor, scaled_factor.T)
    eta = np.finfo(np.float64).eps / (1.0 - np.finfo(np.float64).eps)
    reconstruction_error = np.nextafter(
        (2.0 * eta + eta**2 + gamma(max(width, 1))) * absolute_product
        + eta * (np.abs(symmetric) + np.abs(reconstructed)),
        np.inf,
    )
    observed_factor_residual = _frobenius_up(
        symmetric - reconstructed,
        operations=max(64 * width * width, 1),
    )
    factor_matrix_residual = _bound_sum(
        observed_factor_residual,
        _frobenius_up(
            reconstruction_error,
            operations=max(64 * width * width, 1),
        ),
        operations=max(64 * width * width, 1),
    )

    factor_logs = np.log(np.diag(factor))
    scale_logs = np.log(scales)
    recomputed_logdet = 2.0 * float(np.sum(factor_logs, dtype=np.float64)) + 2.0 * float(
        np.sum(scale_logs, dtype=np.float64)
    )
    if float(decomposition.log_pdet) != recomputed_logdet:
        raise AssertionError("production Cholesky log-sum replay changed")
    scalar_bound = _up(
        gamma(max(32 * width, 1))
        * max(
            1.0,
            abs(recomputed_logdet),
            2.0
            * float(
                np.sum(np.abs(factor_logs), dtype=np.float64)
                + np.sum(np.abs(scale_logs), dtype=np.float64)
            ),
        )
    )

    candidate = _spectral_logdet(values, full_rank=True)
    delta = _bound_sum(
        factor_matrix_residual,
        operations=max(64 * width, 1),
    )
    smallest = float(np.min(candidate.positive_lower))
    assert delta < smallest, "production Cholesky reaches a non-positive eigenvalue"
    weyl_log = float(np.sum(-np.log1p(-delta / candidate.positive_lower)))
    total = _bound_sum(
        scalar_bound,
        weyl_log,
        candidate.logdet_error,
        operations=max(64 * width, 1),
    )
    _assert_zero_centered(
        float(decomposition.log_pdet) - candidate.logdet,
        total,
        label="production Cholesky log-pdet",
    )
    return _ProductionLogdetCertificate(
        rank=width,
        logdet=candidate.logdet,
        total_bound=total,
    )


def _certify_structural_penalty_logdet(
    matrix: np.ndarray,
    recovered: _RecoveredPenalty,
    components,
    group_matrix,
    *,
    lambda_value: float,
    scaled_penalty_bound: float,
) -> _ProductionLogdetCertificate:
    """Certify the stored one-component structural-S log determinant."""

    assert len(components) == 1
    component = components[0]
    assert component.name == _EFS_LAMBDA_NAME
    assert component.penalty_kind == "dense"
    assert lambda_value > 0.0
    declared_rank = int(component.rank)
    assert float(declared_rank) == float(component.rank)
    assert component.omega_raw is not None
    assert component.omega_ssp is not None
    assert component.eigvals_omega is not None
    omega = np.asarray(component.omega_ssp, dtype=np.float64)
    raw_omega = np.asarray(component.omega_raw, dtype=np.float64)
    local_width = len(omega)
    assert omega.shape == (local_width, local_width)
    assert 0 < declared_rank < local_width
    assert isinstance(group_matrix, SparseSSPGroupMatrix)
    np.testing.assert_array_equal(raw_omega, group_matrix.omega)

    # eigvals_omega precedes omega_ssp canonicalization in production. Replay
    # that exact intervening route before applying a numerical residual bound.
    transform = group_matrix.R_inv.T @ raw_omega @ group_matrix.R_inv
    replay_values, replay_vectors = np.linalg.eigh(transform)
    retained_values = replay_values[-declared_rank:]
    retained_vectors = replay_vectors[:, -declared_rank:]
    stored_eigenvalues = np.asarray(component.eigvals_omega, dtype=np.float64)
    replay_eigenvalues = np.sort(replay_values)[::-1][:declared_rank]
    if not np.array_equal(stored_eigenvalues, replay_eigenvalues):
        raise AssertionError("production structural-S eigensystem replay changed")
    if declared_rank == local_width:
        replay_omega = 0.5 * (transform + transform.T)
    else:
        replay_omega = (retained_vectors * retained_values) @ retained_vectors.T
        replay_omega = 0.5 * (replay_omega + replay_omega.T)
    if not np.array_equal(omega, replay_omega):
        raise AssertionError("production structural-S canonical replay changed")
    assert stored_eigenvalues.shape == (declared_rank,)
    assert np.all(stored_eigenvalues > 0.0)

    matrix_operations = max(64 * local_width * local_width, 1)
    retained_gram = retained_vectors.T @ retained_vectors
    orthogonality_residual = _frobenius_up(
        retained_gram - np.eye(declared_rank),
        operations=matrix_operations,
    )
    tau = _bound_sum(
        orthogonality_residual,
        _frobenius_up(
            gamma(matrix_operations)
            * (
                _absolute_product_upper(retained_vectors.T, retained_vectors)
                + np.eye(declared_rank)
            ),
            operations=matrix_operations,
        ),
        operations=matrix_operations,
    )
    assert tau < 1.0

    weighted_vectors = retained_vectors * retained_values
    retained_product = weighted_vectors @ retained_vectors.T
    retained_reconstruction = 0.5 * (retained_product + retained_product.T)
    canonicalization_residual = _bound_sum(
        _frobenius_up(
            omega - retained_reconstruction,
            operations=matrix_operations,
        ),
        _frobenius_up(
            gamma(matrix_operations)
            * (
                _absolute_product_upper(weighted_vectors, retained_vectors.T)
                + np.abs(omega)
                + np.abs(retained_reconstruction)
            ),
            operations=matrix_operations,
        ),
        operations=matrix_operations,
    )
    canonical_pair = _certify_logdet_pair(
        omega,
        retained_reconstruction,
        matrix_bound=canonicalization_residual,
        full_rank=False,
        label="production structural-S canonicalization",
    )
    assert canonical_pair.candidate.rank == canonical_pair.reference.rank == declared_rank

    stored_logs = np.log(np.maximum(stored_eigenvalues, 1.0e-300))
    stored_logdet = float(np.sum(stored_logs, dtype=np.float64))
    if float(component.log_det_omega_plus) != stored_logdet:
        raise AssertionError("production structural-S base logdet replay changed")
    omega_scalar_bound = _up(
        gamma(max(32 * declared_rank, 1))
        * max(
            1.0,
            abs(stored_logdet),
            float(np.sum(np.abs(stored_logs), dtype=np.float64)),
        )
    )
    omega_logdet_bound = _bound_sum(
        omega_scalar_bound,
        declared_rank * -math.log1p(-tau),
        canonical_pair.total_bound,
        operations=max(64 * declared_rank, 1),
    )
    _assert_zero_centered(
        float(component.log_det_omega_plus) - canonical_pair.candidate.logdet,
        omega_logdet_bound,
        label="production structural-S spectral logdet",
    )

    width = len(matrix)
    structural_unscaled = np.zeros((width, width), dtype=np.float64)
    group_slice = component.group_sl
    assert group_slice.stop - group_slice.start == local_width
    structural_unscaled[group_slice, group_slice] = omega
    structural_spectrum = _spectral_logdet(structural_unscaled, full_rank=False)
    assert structural_spectrum.rank == declared_rank
    _componentwise_budget(
        recovered.matrix,
        structural_unscaled,
        recovered.elementwise_error,
        label="production structural-S P/lambda extraction",
    )

    scaled_structural = lambda_value * structural_unscaled
    structural_scaling_roundoff = _frobenius_up(
        gamma(max(64 * width, 1)) * np.abs(scaled_structural),
        operations=max(64 * width * width, 1),
    )
    matrix_bound = _bound_sum(
        scaled_penalty_bound,
        abs(lambda_value) * recovered.spectral_error,
        structural_scaling_roundoff,
        operations=max(64 * width * width, 1),
    )
    matrix_logdet = _certify_logdet_pair(
        matrix,
        scaled_structural,
        matrix_bound=matrix_bound,
        full_rank=False,
        label="production structural-S scaled matrix",
    )
    assert matrix_logdet.candidate.rank == matrix_logdet.reference.rank == declared_rank

    production_logdet = compute_logdet_s_plus(
        {_EFS_LAMBDA_NAME: lambda_value},
        list(components),
    )
    log_lambda = np.log(lambda_value)
    recomputed_logdet = component.rank * log_lambda + component.log_det_omega_plus
    if production_logdet != recomputed_logdet:
        raise AssertionError("production structural-S formula replay changed")
    formula_bound = _up(
        gamma(max(32 * declared_rank, 1))
        * max(
            1.0,
            declared_rank * abs(float(log_lambda)),
            abs(float(component.log_det_omega_plus)),
            abs(float(recomputed_logdet)),
        )
    )
    total = _bound_sum(
        formula_bound,
        omega_logdet_bound,
        matrix_logdet.total_bound,
        operations=max(64 * width, 1),
    )
    _assert_zero_centered(
        production_logdet - matrix_logdet.candidate.logdet,
        total,
        label="production structural-S log-pdet",
    )
    return _ProductionLogdetCertificate(
        rank=declared_rank,
        logdet=matrix_logdet.candidate.logdet,
        total_bound=total,
    )


def _componentwise_budget(
    actual: np.ndarray,
    expected: np.ndarray,
    budget: np.ndarray,
    *,
    label: str,
) -> None:
    observed = np.asarray(actual, dtype=np.float64)
    target = np.asarray(expected, dtype=np.float64)
    allowance = np.broadcast_to(np.asarray(budget, dtype=np.float64), observed.shape)
    assert observed.shape == target.shape
    difference = np.abs(observed - target)
    if np.any(difference > allowance):
        excess = np.where(difference > allowance, difference - allowance, -np.inf)
        index = np.unravel_index(int(np.argmax(excess)), observed.shape)
        raise AssertionError(
            f"{label} exceeds its arithmetic budget at {index}: "
            f"difference={difference[index]}, allowance={allowance[index]}"
        )


def _accepted_lambda_states(smoothing) -> dict[int, float]:
    assert tuple(smoothing.initial_lambdas) == (_EFS_LAMBDA_NAME,)
    states = {0: float(smoothing.initial_lambdas[_EFS_LAMBDA_NAME])}
    for iteration in smoothing.history:
        if iteration.accepted:
            assert iteration.accepted_fit_index is not None
            states[iteration.accepted_fit_index] = float(iteration.lambdas_after[_EFS_LAMBDA_NAME])
    return states


def _componentwise_matrix_bound(
    allowance: float,
    *matrices: np.ndarray,
) -> float:
    width = len(matrices[0])
    scale = max(1.0, *(_spectral_upper(matrix) for matrix in matrices))
    return _bound_sum(
        width * allowance,
        gamma(max(64 * width * width, 1)) * scale,
        operations=max(64 * width, 1),
    )


def _scaled_penalty_bound(recovered: _RecoveredPenalty, lambda_value: float) -> float:
    expected = lambda_value * recovered.matrix
    width = len(expected)
    scalar_error = gamma(max(64 * width, 1)) * np.abs(expected)
    componentwise = abs(lambda_value) * recovered.elementwise_error + scalar_error
    return _bound_sum(
        _frobenius_up(componentwise, operations=max(64 * width * width, 1)),
        gamma(max(64 * width * width, 1)) * max(1.0, _spectral_upper(expected)),
        operations=max(64 * width, 1),
    )


def _fixed_point_certificate(
    fixed: GaussianFitCertificate,
    recovered: _RecoveredPenalty,
    *,
    lambda_value: float,
    penalty_rank: int,
    tolerance: float,
) -> tuple[float, float, float, float]:
    """Certify the one-penalty map at the actual-P strict local coefficient mode.

    The fixed certificate covers the mode of the matrix stored in
    ``oracle.penalty``. The recovered unscaled penalty is uncertain only in the
    smoothing map here; it is deliberately not relabelled as the coefficient
    objective's exact P.
    """

    oracle = fixed.oracle
    beta = oracle.coefficients
    width = len(beta)
    penalty = recovered.matrix
    penalty_norm = _spectral_upper(penalty)
    beta_norm = float(np.linalg.norm(beta, ord=2))
    mode_error = _up(float(fixed.local_root.candidate_errors[0]))
    beta_ball = _bound_sum(beta_norm, mode_error)

    quadratic = float(beta @ penalty @ beta)
    quadratic_scale = max(
        1.0,
        abs(quadratic),
        float(np.abs(beta) @ np.abs(penalty) @ np.abs(beta)),
    )
    quadratic_roundoff = _up(gamma(max(64 * width, 1)) * quadratic_scale)
    quadratic_mode = _up(penalty_norm * mode_error * (2.0 * beta_norm + mode_error))
    quadratic_recovery = _up(recovered.spectral_error * beta_ball**2)

    curvature_arithmetic = _componentwise_matrix_bound(
        fixed.bounds.curvature,
        oracle.penalized_curvature,
    )
    curvature_mode = _hessian_drift_bound(oracle, mode_error)
    hessian_mode_error = _bound_sum(curvature_mode, curvature_arithmetic)
    eigen_lower = _down(
        oracle.spectral_gap.smallest_eigenvalue
        - oracle.spectral_gap.resolution_bar
        - curvature_arithmetic
    )
    assert eigen_lower > 0.0
    assert hessian_mode_error < eigen_lower
    mode_eigen_lower = _down(eigen_lower - hessian_mode_error)
    assert mode_eigen_lower > 0.0

    covariance = oracle.covariance
    identity_residual = _covariance_residual_bound(
        oracle.penalized_curvature,
        covariance,
    )
    covariance_mode = _up(
        hessian_mode_error / (eigen_lower * mode_eigen_lower) + identity_residual / eigen_lower
    )
    covariance_norm = _spectral_upper(covariance)
    trace_term = float(np.trace(covariance @ penalty))
    trace_scale = max(
        1.0,
        abs(trace_term),
        float(np.sum(np.abs(covariance) * np.abs(penalty.T), dtype=np.float64)),
    )
    trace_roundoff = _up(gamma(max(64 * width * width, 1)) * trace_scale)
    trace_mode = _up(
        width
        * (
            covariance_mode * (penalty_norm + recovered.spectral_error)
            + covariance_norm * recovered.spectral_error
        )
    )
    denominator = quadratic + trace_term
    denominator_error = _bound_sum(
        quadratic_mode,
        quadratic_recovery,
        quadratic_roundoff,
        trace_mode,
        trace_roundoff,
        operations=max(64 * width * width, 1),
    )
    assert denominator > denominator_error
    log_map_bound = _bound_sum(
        -math.log1p(-denominator_error / denominator),
        gamma(max(32 * width, 1)),
        operations=max(32 * width, 1),
    )

    # Bridge the production stopping map to this independently evaluated map.
    # It has no coefficient-mode movement, but it must retain the published
    # inverse's backward residual and recovered-S uncertainty.
    published_covariance_error = _up(
        _covariance_residual_bound(
            oracle.penalized_curvature,
            fixed.covariance,
        )
        / eigen_lower
    )
    published_covariance_norm = _spectral_upper(fixed.covariance)
    returned_denominator_error = _bound_sum(
        _up(recovered.spectral_error * beta_norm**2),
        quadratic_roundoff,
        _up(
            width
            * (
                published_covariance_error * (penalty_norm + recovered.spectral_error)
                + published_covariance_norm * recovered.spectral_error
            )
        ),
        trace_roundoff,
        operations=max(64 * width * width, 1),
    )
    assert denominator > returned_denominator_error
    returned_map_bound = _bound_sum(
        -math.log1p(-returned_denominator_error / denominator),
        gamma(max(32 * width, 1)),
        operations=max(32 * width, 1),
    )
    fixed_lambda = penalty_rank / denominator
    residual = abs(math.log(fixed_lambda) - math.log(lambda_value))
    propagation_bound = _bound_sum(
        returned_map_bound,
        log_map_bound,
        operations=max(32 * width, 1),
    )
    local_mode_bound = _bound_sum(
        tolerance,
        propagation_bound,
        gamma(max(32 * width, 1)),
        operations=max(32 * width, 1),
    )
    return residual, local_mode_bound, propagation_bound, returned_map_bound


def _certify_efs_state(
    model,
    *,
    fit_index: int,
    lambda_value: float,
    recovered_penalty: _RecoveredPenalty,
    expected_semantics: str,
    expected_scale_floor: float,
    expected_plan_identifier: str,
    prediction_frame: pd.DataFrame | None = None,
) -> _EFSStateCertificate:
    smoothing = model.smoothing
    retained = model.fit_state.retained_rows
    assert smoothing is not None and retained is not None
    fit = smoothing.coefficient_fits[fit_index]
    assert model.fit_state.weight_contract == WeightContract(expected_semantics)  # type: ignore[arg-type]
    assert retained.likelihood_weights.provenance.contract == WeightContract(  # type: ignore[arg-type]
        expected_semantics
    )
    assert model.family.scale_floor == expected_scale_floor
    assert model.fit_state.family_likelihood_plan_identifier == expected_plan_identifier
    assert fit.family_likelihood_plan_identifier == (
        model.fit_state.family_likelihood_plan_identifier
    )
    assert fit.terminal_curvature.requested_source == "observed"
    assert fit.terminal_curvature.actual_source == "observed"
    assert fit.terminal_curvature.fallback_count == 0
    expected_penalty = lambda_value * recovered_penalty.matrix
    penalty_matrix_bound = _scaled_penalty_bound(recovered_penalty, lambda_value)
    _componentwise_budget(
        fit.penalty,
        expected_penalty,
        abs(lambda_value) * recovered_penalty.elementwise_error
        + gamma(max(64 * len(fit.coefficients), 1)) * np.abs(expected_penalty),
        label="accepted-state scaled penalty",
    )
    terminal = fit_index == smoothing.terminal_fit_index
    covariance = model.covariance if terminal else fit.terminal_rank.pseudo_inverse()
    total_edf = (
        model.inference.total_edf
        if terminal
        else float(np.trace(covariance @ fit.terminal_data_curvature))
    )
    fixed = certify_gaussian_result(
        model.layout,
        fit,
        retained.response,
        retained.likelihood_weights.values,
        semantics=expected_semantics,  # type: ignore[arg-type]
        covariance=covariance,
        total_edf=total_edf,
        inference_rank=fit.terminal_rank.rank,
        scale_floor=expected_scale_floor,
        prediction_parameters=(
            None if prediction_frame is None else model.predict_parameters(prediction_frame)
        ),
        default_prediction=None if prediction_frame is None else model.predict(prediction_frame),
    )
    width = len(fit.coefficients)
    h_matrix_bound = _componentwise_matrix_bound(
        fixed.bounds.curvature,
        fit.terminal_penalized_curvature,
        fixed.oracle.penalized_curvature,
    )
    h_production_logdet = _certify_cholesky_logdet(
        fit.terminal_penalized_curvature,
        fit.terminal_rank,
    )
    h_logdet = _certify_logdet_pair(
        fit.terminal_penalized_curvature,
        fixed.oracle.penalized_curvature,
        matrix_bound=h_matrix_bound,
        full_rank=True,
        label="penalized curvature",
    )
    production_objective = efs_module.joint_laplace_objective(
        fit,
        layout=model.layout,
        lambdas={_EFS_LAMBDA_NAME: lambda_value},
    )
    assert fit.penalized_optimizing_log_likelihood is not None
    penalty_production_logdet = _certify_structural_penalty_logdet(
        fit.penalty,
        recovered_penalty,
        model.layout.penalties,
        _semantic_efs_penalty_group_matrix(model),
        lambda_value=lambda_value,
        scaled_penalty_bound=penalty_matrix_bound,
    )
    assert fit.terminal_rank.rank == h_production_logdet.rank == h_logdet.reference.rank == width
    laml = -fixed.oracle.penalized_optimizing_log_likelihood + 0.5 * (
        h_logdet.reference.logdet - penalty_production_logdet.logdet
    )
    laml_bound = _bound_sum(
        fixed.bounds.likelihood_sum,
        0.5
        * (
            h_production_logdet.total_bound
            + h_logdet.total_bound
            + penalty_production_logdet.total_bound
        ),
        gamma(max(64 * width, 1)) * max(1.0, abs(laml)),
        operations=max(64 * width, 1),
    )
    np.testing.assert_allclose(
        production_objective,
        laml,
        rtol=0.0,
        atol=laml_bound,
    )

    residual, residual_bound, map_bound, returned_map_bound = _fixed_point_certificate(
        fixed,
        recovered_penalty,
        lambda_value=lambda_value,
        penalty_rank=penalty_production_logdet.rank,
        tolerance=smoothing.config.tolerance,
    )
    return _EFSStateCertificate(
        fit_index=fit_index,
        lambda_value=lambda_value,
        fixed=fixed,
        laml=laml,
        laml_bound=laml_bound,
        penalty_rank=penalty_production_logdet.rank,
        fixed_point_log_residual=residual,
        fixed_point_bound=float(residual_bound),
        fixed_point_map_bound=map_bound,
        returned_fixed_point_map_bound=returned_map_bound,
        recovered_penalty=recovered_penalty,
    )


def _recover_unscaled_penalty(
    initial_penalty: np.ndarray,
    initial_lambda: float,
) -> _RecoveredPenalty:
    matrix = np.asarray(initial_penalty, dtype=np.float64) / initial_lambda
    width = len(matrix)
    recovery_gamma = gamma(max(64 * width, 1))
    assert recovery_gamma < 1.0
    elementwise_error = np.nextafter(
        recovery_gamma / (1.0 - recovery_gamma) * np.abs(matrix),
        np.inf,
    )
    spectral_error = _frobenius_up(
        elementwise_error,
        operations=max(64 * width * width, 1),
    )
    return _RecoveredPenalty(
        matrix=np.array(matrix, copy=True),
        elementwise_error=np.array(elementwise_error, copy=True),
        spectral_error=spectral_error,
    )


def _unscaled_penalty(model) -> _RecoveredPenalty:
    smoothing = model.smoothing
    assert smoothing is not None
    return _recover_unscaled_penalty(
        smoothing.coefficient_fits[0].penalty,
        float(smoothing.initial_lambdas[_EFS_LAMBDA_NAME]),
    )


def _certify_history(
    model,
    *,
    training_frame: pd.DataFrame,
    expected_semantics: str,
    expected_scale_floor: float,
    expected_plan_identifier: str,
) -> dict[int, _EFSStateCertificate]:
    smoothing = model.smoothing
    retained = model.fit_state.retained_rows
    assert smoothing is not None and retained is not None
    recovered = _unscaled_penalty(model)
    certificates = {
        index: _certify_efs_state(
            model,
            fit_index=index,
            lambda_value=value,
            recovered_penalty=recovered,
            expected_semantics=expected_semantics,
            expected_scale_floor=expected_scale_floor,
            expected_plan_identifier=expected_plan_identifier,
            prediction_frame=(training_frame if index == smoothing.terminal_fit_index else None),
        )
        for index, value in _accepted_lambda_states(smoothing).items()
    }
    np.testing.assert_allclose(
        smoothing.initial_objective,
        certificates[0].laml,
        rtol=0.0,
        atol=certificates[0].laml_bound,
    )
    for iteration in smoothing.history:
        source = certificates[iteration.source_fit_index]
        np.testing.assert_allclose(
            iteration.objective_before,
            source.laml,
            rtol=0.0,
            atol=source.laml_bound,
        )
        if not iteration.accepted:
            assert iteration.objective_after == iteration.objective_before
            continue
        assert iteration.accepted_fit_index is not None
        accepted = certificates[iteration.accepted_fit_index]
        np.testing.assert_allclose(
            iteration.objective_after,
            accepted.laml,
            rtol=0.0,
            atol=accepted.laml_bound,
        )
        signed_change = iteration.objective_after - iteration.objective_before
        ceiling_change = smoothing.config.objective_tolerance * (
            1.0 + abs(iteration.objective_before)
        )
        assert signed_change <= ceiling_change
        assert iteration.accepted_curvature == (
            smoothing.coefficient_fits[iteration.accepted_fit_index].terminal_curvature
        )
    terminal = certificates[smoothing.terminal_fit_index]
    np.testing.assert_allclose(
        smoothing.objective,
        terminal.laml,
        rtol=0.0,
        atol=terminal.laml_bound,
    )
    assert smoothing.converged is True
    assert smoothing.convergence_reason == "lambda_change"
    assert terminal.fixed_point_log_residual <= (
        smoothing.config.tolerance + terminal.returned_fixed_point_map_bound
    )
    assert terminal.fixed_point_log_residual <= terminal.fixed_point_bound
    # Algorithm-matched EFS is defined at the accepted coefficient fit.  The
    # returned-map channel encloses the independently reconstructed update at
    # that same fit; the larger ``fixed_point_map_bound`` additionally
    # transports to an exact coefficient mode and therefore belongs to a
    # stronger stationarity claim that EFS does not make.
    assert terminal.returned_fixed_point_map_bound < (
        len(terminal.fixed.oracle.coefficients) * smoothing.config.tolerance
    )
    separation = gamma(64) * max(
        1.0,
        abs(terminal.lambda_value),
        abs(smoothing.config.minimum_lambda),
        abs(smoothing.config.maximum_lambda),
    )
    assert terminal.lambda_value > smoothing.config.minimum_lambda + separation
    assert terminal.lambda_value < smoothing.config.maximum_lambda - separation
    return certificates


def _certify_terminal(
    model,
    *,
    training_frame: pd.DataFrame,
    expected_semantics: str,
    expected_scale_floor: float,
    expected_plan_identifier: str,
) -> _EFSStateCertificate:
    smoothing = model.smoothing
    assert smoothing is not None
    certificate = _certify_efs_state(
        model,
        fit_index=smoothing.terminal_fit_index,
        lambda_value=float(smoothing.lambdas[_EFS_LAMBDA_NAME]),
        recovered_penalty=_unscaled_penalty(model),
        expected_semantics=expected_semantics,
        expected_scale_floor=expected_scale_floor,
        expected_plan_identifier=expected_plan_identifier,
        prediction_frame=training_frame,
    )
    assert smoothing.converged is True
    assert smoothing.convergence_reason == "lambda_change"
    assert smoothing.fallback_count == 0
    assert certificate.fixed_point_log_residual <= (
        smoothing.config.tolerance + certificate.returned_fixed_point_map_bound
    )
    assert certificate.fixed_point_log_residual <= certificate.fixed_point_bound
    assert certificate.returned_fixed_point_map_bound < (
        len(certificate.fixed.oracle.coefficients) * smoothing.config.tolerance
    )
    separation = gamma(64) * max(
        1.0,
        abs(certificate.lambda_value),
        abs(smoothing.config.minimum_lambda),
        abs(smoothing.config.maximum_lambda),
    )
    assert certificate.lambda_value > smoothing.config.minimum_lambda + separation
    assert certificate.lambda_value < smoothing.config.maximum_lambda - separation
    np.testing.assert_allclose(
        smoothing.objective,
        certificate.laml,
        rtol=0.0,
        atol=certificate.laml_bound,
    )
    return certificate


def _carrier_signature(model) -> tuple[object, ...]:
    smoothing = model.smoothing
    assert smoothing is not None
    return tuple(
        (
            item.accepted,
            item.backtracks,
            tuple(item.lambdas_before.items()),
            tuple(item.proposed_lambdas.items()),
            tuple(item.lambdas_after.items()),
            tuple(item.proposed_log_steps.items()),
            tuple(item.accepted_log_steps.items()),
            item.objective_before,
            item.objective_after,
            item.source_fit_index,
            item.accepted_fit_index,
            item.coefficient_fit_indices,
        )
        for item in smoothing.history
    )


def _assert_carrier_noninterference(reference, shifted) -> None:
    left = reference.smoothing
    right = shifted.smoothing
    assert left is not None and right is not None
    assert _carrier_signature(shifted) == _carrier_signature(reference)
    assert dict(right.lambdas) == dict(left.lambdas)
    assert right.converged == left.converged
    assert right.convergence_reason == left.convergence_reason
    assert right.initial_objective == left.initial_objective
    assert right.objective == left.objective
    assert right.terminal_fit_index == left.terminal_fit_index
    assert len(right.coefficient_fits) == len(left.coefficient_fits)
    assert {fit.family_likelihood_plan_identifier for fit in left.coefficient_fits} == {
        reference.fit_state.family_likelihood_plan_identifier
    }
    assert {fit.family_likelihood_plan_identifier for fit in right.coefficient_fits} == {
        shifted.fit_state.family_likelihood_plan_identifier
    }

    retained = reference.fit_state.retained_rows
    assert retained is not None
    carrier_rows = 0.5 * np.log(retained.likelihood_weights.values)
    expected_shift = _EFS_CARRIER_PER_ROW * len(carrier_rows)
    carrier_bound = gamma(max(32 * len(carrier_rows), 1)) * max(
        1.0,
        expected_shift,
        float(
            np.sum(
                np.abs(carrier_rows + _EFS_CARRIER_PER_ROW),
                dtype=np.float64,
            )
        ),
    )
    for left_fit, right_fit in zip(
        left.coefficient_fits,
        right.coefficient_fits,
        strict=True,
    ):
        for name in (
            "coefficients",
            "eta",
            "theta",
            "terminal_data_curvature",
            "terminal_penalized_curvature",
        ):
            np.testing.assert_array_equal(getattr(right_fit, name), getattr(left_fit, name))
        assert right_fit.history == left_fit.history
        assert right_fit.convergence_reason == left_fit.convergence_reason
        assert right_fit.terminal_curvature == left_fit.terminal_curvature
        assert right_fit.terminal_rank.rank == left_fit.terminal_rank.rank
        assert right_fit.terminal_rank.log_pdet == left_fit.terminal_rank.log_pdet
        np.testing.assert_array_equal(
            right_fit.terminal_rank.pseudo_inverse(),
            left_fit.terminal_rank.pseudo_inverse(),
        )
        assert right_fit.optimizing_log_likelihood == left_fit.optimizing_log_likelihood
        assert right_fit.penalized_optimizing_log_likelihood == (
            left_fit.penalized_optimizing_log_likelihood
        )
        assert right_fit.objective == left_fit.objective
        for name in (
            "parameter_independent_carrier",
            "log_likelihood",
            "penalized_log_likelihood",
        ):
            np.testing.assert_allclose(
                float(getattr(right_fit, name)) - float(getattr(left_fit, name)),
                expected_shift,
                rtol=0.0,
                atol=carrier_bound,
            )
    np.testing.assert_array_equal(shifted.covariance, reference.covariance)
    assert shifted.inference.total_edf == reference.inference.total_edf
    for fit_index, lambda_value in _accepted_lambda_states(left).items():
        left_objective = efs_module.joint_laplace_objective(
            left.coefficient_fits[fit_index],
            layout=reference.layout,
            lambdas={_EFS_LAMBDA_NAME: lambda_value},
        )
        right_objective = efs_module.joint_laplace_objective(
            right.coefficient_fits[fit_index],
            layout=shifted.layout,
            lambdas={_EFS_LAMBDA_NAME: lambda_value},
        )
        assert right_objective == left_objective, (
            "authoritative outer objective changed with a fixed carrier"
        )


def _assert_state_bound_mutations(
    certificate: _EFSStateCertificate,
    model,
    *,
    tolerance: float,
) -> None:
    """Kill omitted H/S spectral and fixed-point propagation channels."""

    smoothing = model.smoothing
    assert smoothing is not None
    fit = smoothing.coefficient_fits[certificate.fit_index]
    layout = model.layout
    group_matrix = _semantic_efs_penalty_group_matrix(model)
    oracle = certificate.fixed.oracle
    width = len(oracle.coefficients)
    component_allowance = certificate.fixed.bounds.curvature
    h_mutation = np.array(oracle.penalized_curvature, copy=True)
    h_mutation[0, 0] += 0.5 * width * component_allowance
    with pytest.raises(AssertionError, match="matrix perturbation"):
        _certify_logdet_pair(
            h_mutation,
            oracle.penalized_curvature,
            matrix_bound=_bound_sum(component_allowance, gamma(64 * width * width)),
            full_rank=True,
            label="missing componentwise-to-Frobenius H mutant",
        )
    _certify_logdet_pair(
        h_mutation,
        oracle.penalized_curvature,
        matrix_bound=_componentwise_matrix_bound(
            component_allowance,
            h_mutation,
            oracle.penalized_curvature,
        ),
        full_rank=True,
        label="complete H perturbation mutant",
    )

    assert fit.terminal_rank.cholesky_factor is not None
    mutated_factor = np.array(fit.terminal_rank.cholesky_factor, copy=True)
    mutated_factor[1, 0] += 1.0
    mutated_rank = replace(fit.terminal_rank, cholesky_factor=mutated_factor)
    with pytest.raises(AssertionError, match="production Cholesky"):
        _certify_cholesky_logdet(
            fit.terminal_penalized_curvature,
            mutated_rank,
        )
    with pytest.raises(AssertionError, match="production Cholesky log-sum"):
        _certify_cholesky_logdet(
            fit.terminal_penalized_curvature,
            replace(fit.terminal_rank, log_pdet=fit.terminal_rank.log_pdet + 1.0),
        )

    recovered = certificate.recovered_penalty
    values, vectors = np.linalg.eigh(recovered.matrix)
    direction = vectors[:, int(np.argmax(values))]
    s_mutation = np.array(recovered.matrix, copy=True)
    s_mutation += 0.5 * recovered.spectral_error * np.outer(direction, direction)
    with pytest.raises(AssertionError, match="matrix perturbation"):
        _certify_logdet_pair(
            s_mutation,
            recovered.matrix,
            matrix_bound=0.25 * recovered.spectral_error,
            full_rank=False,
            label="missing recovered-S mutant",
        )
    _certify_logdet_pair(
        s_mutation,
        recovered.matrix,
        matrix_bound=_bound_sum(2.0 * recovered.spectral_error),
        full_rank=False,
        label="complete recovered-S mutant",
    )

    component = layout.penalties[0]

    def certify_structural(mutated_component=component, mutated_recovered=recovered):
        return _certify_structural_penalty_logdet(
            fit.penalty,
            mutated_recovered,
            (mutated_component,),
            group_matrix,
            lambda_value=certificate.lambda_value,
            scaled_penalty_bound=_scaled_penalty_bound(
                mutated_recovered,
                certificate.lambda_value,
            ),
        )

    mutated_component = replace(
        component,
        log_det_omega_plus=component.log_det_omega_plus + 1.0,
    )
    assert mutated_component.log_det_omega_plus != component.log_det_omega_plus
    with pytest.raises(AssertionError, match="production structural-S"):
        certify_structural(mutated_component)
    mutated_eigenvalues = np.array(component.eigvals_omega, copy=True)
    mutated_eigenvalues[0] *= 2.0
    coherent_eigen_mutation = replace(
        component,
        eigvals_omega=mutated_eigenvalues,
        log_det_omega_plus=float(np.sum(np.log(np.maximum(mutated_eigenvalues, 1.0e-300)))),
    )
    with pytest.raises(AssertionError, match="eigensystem replay"):
        certify_structural(coherent_eigen_mutation)
    with pytest.raises(AssertionError):
        certify_structural(replace(component, rank=component.rank + 1.0))
    initial_penalty_mutation = np.array(smoothing.coefficient_fits[0].penalty, copy=True)
    initial_penalty_mutation[component.group_sl.start, component.group_sl.start] += 1.0e-4
    recovery_mutation = _recover_unscaled_penalty(
        initial_penalty_mutation,
        float(smoothing.initial_lambdas[_EFS_LAMBDA_NAME]),
    )
    with pytest.raises(AssertionError, match="P/lambda extraction"):
        certify_structural(mutated_recovered=recovery_mutation)

    no_recovery = replace(
        recovered,
        elementwise_error=np.zeros_like(recovered.elementwise_error),
        spectral_error=0.0,
    )
    _, _, without_recovery, _ = _fixed_point_certificate(
        certificate.fixed,
        no_recovery,
        lambda_value=certificate.lambda_value,
        penalty_rank=certificate.penalty_rank,
        tolerance=tolerance,
    )
    assert certificate.fixed_point_map_bound > without_recovery
    recovered_channel_mutation = 0.5 * (certificate.fixed_point_map_bound + without_recovery)
    with pytest.raises(AssertionError, match="zero-centered"):
        _assert_zero_centered(
            recovered_channel_mutation,
            without_recovery,
            label="missing recovered-S fixed-point mutant",
        )
    _assert_zero_centered(
        recovered_channel_mutation,
        certificate.fixed_point_map_bound,
        label="complete recovered-S fixed-point mutant",
    )

    assert certificate.fixed_point_map_bound > certificate.returned_fixed_point_map_bound
    coefficient_channel_mutation = 0.5 * (
        certificate.fixed_point_map_bound + certificate.returned_fixed_point_map_bound
    )
    with pytest.raises(AssertionError, match="zero-centered"):
        _assert_zero_centered(
            coefficient_channel_mutation,
            certificate.returned_fixed_point_map_bound,
            label="missing coefficient/H/covariance fixed-point mutant",
        )
    _assert_zero_centered(
        coefficient_channel_mutation,
        certificate.fixed_point_map_bound,
        label="complete coefficient/H/covariance fixed-point mutant",
    )


def test_prior_efs_reconstructs_every_accepted_state_and_ignores_fixed_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills plan rebinding, reported-likelihood LAML, and unsupported stop claims."""

    fixture = _semantic_efs_fixture()
    config = _semantic_efs_config(
        tolerance=1.0e-4,
        objective_tolerance=1.0e-10,
    )
    reference = _fit_semantic_efs(
        family=_ConstantCarrierGaussian(0.0),
        semantics="prior",
        frame=fixture.frame,
        response=fixture.response,
        weights=fixture.prior_weights,
        efs_config=config,
    )
    shifted = _fit_semantic_efs(
        family=_ConstantCarrierGaussian(_EFS_CARRIER_PER_ROW),
        semantics="prior",
        frame=fixture.frame,
        response=fixture.response,
        weights=fixture.prior_weights,
        efs_config=config,
    )
    certificates = _certify_history(
        reference,
        training_frame=fixture.frame,
        expected_semantics="prior",
        expected_scale_floor=0.0,
        expected_plan_identifier=_expected_gaussian_plan_identifier(
            fixture.prior_weights,
            semantics="prior",
            carrier_wrapper=0.0,
        ),
    )
    assert shifted.fit_state.weight_contract == WeightContract("prior")
    assert shifted.family.scale_floor == 0.0
    assert shifted.fit_state.family_likelihood_plan_identifier == (
        _expected_gaussian_plan_identifier(
            fixture.prior_weights,
            semantics="prior",
            carrier_wrapper=_EFS_CARRIER_PER_ROW,
        )
    )
    _assert_carrier_noninterference(reference, shifted)
    assert reference.smoothing is not None
    terminal = certificates[reference.smoothing.terminal_fit_index]
    assert terminal.penalty_rank == 5
    assert terminal.fixed_point_log_residual < config.tolerance
    _assert_state_bound_mutations(
        terminal,
        reference,
        tolerance=config.tolerance,
    )

    with pytest.raises(AssertionError):
        _certify_efs_state(
            reference,
            fit_index=reference.smoothing.terminal_fit_index,
            lambda_value=terminal.lambda_value,
            recovered_penalty=terminal.recovered_penalty,
            expected_semantics="frequency",
            expected_scale_floor=0.0,
            expected_plan_identifier=_expected_gaussian_plan_identifier(
                fixture.prior_weights,
                semantics="frequency",
            ),
        )

    original_objective = efs_module.joint_laplace_objective

    def reported_likelihood_mutation(result, *, layout, lambdas):
        return original_objective(result, layout=layout, lambdas=lambdas) - (
            result.parameter_independent_carrier
        )

    with monkeypatch.context() as mutation:
        mutation.setattr(
            smoothing_objective, "joint_laplace_objective", reported_likelihood_mutation
        )
        mutation.setattr(efs_module, "joint_laplace_objective", reported_likelihood_mutation)
        with pytest.raises(
            AssertionError,
            match="authoritative outer objective changed with a fixed carrier",
        ):
            _assert_carrier_noninterference(reference, shifted)


def _assert_zero_centered(value: float, bound: float, *, label: str) -> None:
    if abs(float(value)) > bound:
        raise AssertionError(
            f"{label} exceeds its zero-centered derived bound: "
            f"difference={abs(float(value))}, bound={bound}"
        )


def _assert_canonical_row_identity(left, right) -> None:
    """Establish exact replication/all-one premises before perturbation math."""

    if len(left.response) != len(right.response):
        assert left.semantics == right.semantics == "frequency"
        assert np.all(left.weights == np.floor(left.weights))
        take = np.repeat(np.arange(len(left.response), dtype=np.intp), left.weights.astype(np.intp))
        assert np.all(right.weights == 1.0)
    else:
        take = np.arange(len(left.response), dtype=np.intp)
        assert np.all(left.weights == 1.0)
        assert np.all(right.weights == 1.0)
    for left_values, right_values in (
        (left.response[take], right.response),
        (left.location_design[take], right.location_design),
        (left.scale_design[take], right.scale_design),
        (left.location_offset[take], right.location_offset),
        (left.scale_offset[take], right.scale_offset),
    ):
        np.testing.assert_array_equal(left_values, right_values)
    assert left.scale_floor == right.scale_floor == 0.0


def _quadratic_roundoff(beta: np.ndarray, penalty: np.ndarray) -> float:
    width = len(beta)
    value = float(beta @ penalty @ beta)
    scale = max(
        1.0,
        abs(value),
        float(np.abs(beta) @ np.abs(penalty) @ np.abs(beta)),
    )
    return _up(gamma(max(64 * width, 1)) * scale)


def _matvec_roundoff(matrix: np.ndarray, vector: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    probe = np.asarray(vector, dtype=np.float64)
    width = len(probe)
    assert values.shape == (width, width)
    componentwise_error = np.nextafter(
        gamma(max(width, 1)) * _absolute_product_upper(values, probe),
        np.inf,
    )
    raw = float(np.linalg.norm(componentwise_error, ord=2))
    return _up(
        raw
        + gamma(max(64 * width, 1))
        * max(1.0, raw, float(np.sum(componentwise_error, dtype=np.float64)))
    )


def _assert_two_matvec_roundoff_channels() -> None:
    probe = np.array([0.5, 1.0, 1.5])
    left = _matvec_roundoff(np.diag([1.0, 2.0, 3.0]), probe)
    right = _matvec_roundoff(np.diag([2.0, 3.0, 4.0]), probe)
    boundary = _down(left + right)
    with pytest.raises(AssertionError, match="zero-centered"):
        _assert_zero_centered(boundary, left, label="missing right covariance matvec")
    with pytest.raises(AssertionError, match="zero-centered"):
        _assert_zero_centered(boundary, right, label="missing left covariance matvec")
    _assert_zero_centered(
        boundary,
        _bound_sum(left, right),
        label="complete two-route covariance matvec",
    )


def _assert_all_one_exact_terminal_math(
    left_model,
    right_model,
    left: _EFSStateCertificate,
    right: _EFSStateCertificate,
    *,
    frame: pd.DataFrame,
) -> None:
    """Pin identical arithmetic after the two all-one contracts are bound."""

    left_smoothing = left_model.smoothing
    right_smoothing = right_model.smoothing
    assert left_smoothing is not None and right_smoothing is not None
    assert dict(left_smoothing.lambdas) == dict(right_smoothing.lambdas)
    assert left_smoothing.objective == right_smoothing.objective

    left_fit = left_model.result
    right_fit = right_model.result
    for name in (
        "coefficients",
        "eta",
        "theta",
        "penalty",
        "terminal_score",
        "terminal_data_curvature",
        "terminal_penalized_curvature",
    ):
        np.testing.assert_array_equal(getattr(left_fit, name), getattr(right_fit, name))
    np.testing.assert_array_equal(left_model.covariance, right_model.covariance)
    np.testing.assert_array_equal(
        left_model.predict_parameters(frame),
        right_model.predict_parameters(frame),
    )
    np.testing.assert_array_equal(left_model.predict(frame), right_model.predict(frame))

    for name in (
        "optimizing_log_likelihood",
        "parameter_independent_carrier",
        "log_likelihood",
        "penalty_value",
        "penalized_optimizing_log_likelihood",
        "penalized_log_likelihood",
        "objective",
    ):
        assert getattr(left_fit, name) == getattr(right_fit, name)
    assert left_fit.terminal_rank.rank == right_fit.terminal_rank.rank
    assert left_fit.terminal_rank.log_pdet == right_fit.terminal_rank.log_pdet
    assert left_model.inference.rank == right_model.inference.rank
    assert left_model.inference.total_edf == right_model.inference.total_edf
    assert left.lambda_value == right.lambda_value
    assert left.laml == right.laml
    assert left.fixed_point_log_residual == right.fixed_point_log_residual
    assert left.penalty_rank == right.penalty_rank
    np.testing.assert_array_equal(
        left.recovered_penalty.matrix,
        right.recovered_penalty.matrix,
    )


def _assert_terminal_parity(
    left_model,
    right_model,
    left: _EFSStateCertificate,
    right: _EFSStateCertificate,
    *,
    left_prediction: np.ndarray,
    right_prediction: np.ndarray,
) -> None:
    """Zero-centered terminal parity, conditional on algorithm resolution in λ."""

    left_oracle = left.fixed.oracle
    right_oracle = right.fixed.oracle
    width = len(left_oracle.coefficients)
    _assert_canonical_row_identity(left_oracle, right_oracle)

    left_recovered = left.recovered_penalty
    right_recovered = right.recovered_penalty
    unscaled_bound = _bound_sum(
        left_recovered.spectral_error,
        right_recovered.spectral_error,
        operations=max(64 * width, 1),
    )
    unscaled_difference = _frobenius_up(
        left_recovered.matrix - right_recovered.matrix,
        operations=max(64 * width * width, 1),
    )
    assert unscaled_difference <= unscaled_bound

    # No scalar fixed-point isolation theorem is claimed.  This is deliberately
    # an algorithm-resolution premise for the two terminal EFS stops.
    smoothing_tolerance = max(
        left_model.smoothing.config.tolerance,
        right_model.smoothing.config.tolerance,
    )
    log_lambda_bound = _bound_sum(
        2.0 * smoothing_tolerance,
        gamma(max(32 * width, 1)),
        operations=max(32 * width, 1),
    )
    _assert_zero_centered(
        math.log(left.lambda_value) - math.log(right.lambda_value),
        log_lambda_bound,
        label="terminal log-lambda algorithm-resolution agreement",
    )
    maximum_lambda = max(left.lambda_value, right.lambda_value)
    lambda_difference_bound = _up(maximum_lambda * math.expm1(log_lambda_bound))
    penalty_matrix_bound = _bound_sum(
        lambda_difference_bound * _spectral_upper(left_recovered.matrix),
        maximum_lambda * unscaled_bound,
        _scaled_penalty_bound(left_recovered, left.lambda_value),
        _scaled_penalty_bound(right_recovered, right.lambda_value),
        operations=max(64 * width * width, 1),
    )
    assert (
        _frobenius_up(
            left_oracle.penalty - right_oracle.penalty,
            operations=max(64 * width * width, 1),
        )
        <= penalty_matrix_bound
    )

    common = local_root_certificate(
        left_oracle,
        np.vstack((left_oracle.coefficients, right_oracle.coefficients)),
    )
    coefficient_bound = _up(float(np.sum(common.candidate_errors, dtype=np.float64)))
    _assert_zero_centered(
        float(np.linalg.norm(left_oracle.coefficients - right_oracle.coefficients, ord=2)),
        coefficient_bound,
        label="terminal coefficient local-mode agreement",
    )

    center_left = coefficient_oracle(
        left_oracle.response,
        left_oracle.weights,
        semantics=left_oracle.semantics,
        location_design=left_oracle.location_design,
        scale_design=left_oracle.scale_design,
        coefficients=common.center,
        penalty=left_oracle.penalty,
        location_offset=left_oracle.location_offset,
        scale_offset=left_oracle.scale_offset,
        scale_floor=0.0,
    )
    center_right = coefficient_oracle(
        right_oracle.response,
        right_oracle.weights,
        semantics=right_oracle.semantics,
        location_design=right_oracle.location_design,
        scale_design=right_oracle.scale_design,
        coefficients=common.center,
        penalty=right_oracle.penalty,
        location_offset=right_oracle.location_offset,
        scale_offset=right_oracle.scale_offset,
        scale_floor=0.0,
    )
    center_left_bounds = oracle_bounds(center_left)
    center_right_bounds = oracle_bounds(center_right)
    center_sum_bound = _bound_sum(
        center_left_bounds.likelihood_sum,
        center_right_bounds.likelihood_sum,
        operations=max(64 * len(right_oracle.response), 1),
    )
    for name in (
        "optimizing_log_likelihood",
        "reported_log_likelihood",
        "parameter_independent_carrier",
    ):
        _assert_zero_centered(
            float(getattr(center_left, name)) - float(getattr(center_right, name)),
            center_sum_bound,
            label=f"canonical-center {name}",
        )

    data_drift = _hessian_drift_bound(center_left, common.radius)
    data_norm_on_ball = _bound_sum(_spectral_upper(center_left.data_curvature), data_drift)
    score_norm_on_ball = _bound_sum(
        float(np.linalg.norm(center_left.score_data, ord=2)),
        data_norm_on_ball * common.radius,
        math.sqrt(width) * center_left_bounds.score_roundoff,
        operations=max(64 * len(left_oracle.response) + 32 * width, 1),
    )
    likelihood_bound = _bound_sum(
        score_norm_on_ball * coefficient_bound,
        left.fixed.bounds.likelihood_sum,
        right.fixed.bounds.likelihood_sum,
        center_sum_bound,
        operations=max(64 * len(right_oracle.response), 1),
    )
    for oracle_name, result_name in (
        ("optimizing_log_likelihood", "optimizing_log_likelihood"),
        ("reported_log_likelihood", "log_likelihood"),
    ):
        _assert_zero_centered(
            float(getattr(left_oracle, oracle_name)) - float(getattr(right_oracle, oracle_name)),
            likelihood_bound,
            label=f"oracle {oracle_name}",
        )
        _assert_zero_centered(
            float(getattr(left_model.result, result_name))
            - float(getattr(right_model.result, result_name)),
            likelihood_bound,
            label=f"published {result_name}",
        )

    data_curvature_bound = _bound_sum(
        _hessian_drift_bound(left_oracle, coefficient_bound),
        _componentwise_matrix_bound(
            left.fixed.bounds.curvature,
            left_oracle.data_curvature,
        ),
        _componentwise_matrix_bound(
            right.fixed.bounds.curvature,
            right_oracle.data_curvature,
        ),
        operations=max(96 * len(right_oracle.response) + 32 * width, 1),
    )
    assert (
        _frobenius_up(
            left_oracle.data_curvature - right_oracle.data_curvature,
            operations=max(64 * width * width, 1),
        )
        <= data_curvature_bound
    )
    penalized_curvature_bound = _bound_sum(
        data_curvature_bound,
        penalty_matrix_bound,
        operations=max(64 * width * width, 1),
    )
    assert (
        _frobenius_up(
            left_oracle.penalized_curvature - right_oracle.penalized_curvature,
            operations=max(64 * width * width, 1),
        )
        <= penalized_curvature_bound
    )

    eigen_lower = _down(
        left_oracle.spectral_gap.smallest_eigenvalue
        - left_oracle.spectral_gap.resolution_bar
        - _componentwise_matrix_bound(
            left.fixed.bounds.curvature,
            left_oracle.penalized_curvature,
        )
    )
    assert penalized_curvature_bound < eigen_lower
    right_eigen_lower = _down(eigen_lower - penalized_curvature_bound)
    covariance_movement = _up(penalized_curvature_bound / (eigen_lower * right_eigen_lower))
    left_covariance_error = _up(
        _covariance_residual_bound(left_oracle.penalized_curvature, left_model.covariance)
        / eigen_lower
    )
    right_covariance_error = _up(
        _covariance_residual_bound(right_oracle.penalized_curvature, right_model.covariance)
        / right_eigen_lower
    )
    probe = np.linspace(0.5, 1.5, width)
    probe_norm = float(np.linalg.norm(probe, ord=2))
    left_matvec_roundoff = _matvec_roundoff(left_model.covariance, probe)
    right_matvec_roundoff = _matvec_roundoff(right_model.covariance, probe)
    left_action = left_model.covariance @ probe
    right_action = right_model.covariance @ probe
    subtraction_roundoff = _up(
        gamma(max(64 * width, 1))
        * max(
            1.0,
            float(np.linalg.norm(np.abs(left_action) + np.abs(right_action), ord=2)),
        )
    )
    action_bound = _bound_sum(
        (covariance_movement + left_covariance_error + right_covariance_error) * probe_norm,
        left_matvec_roundoff,
        right_matvec_roundoff,
        subtraction_roundoff,
        operations=max(64 * width * width, 1),
    )
    _assert_zero_centered(
        float(np.linalg.norm(left_action - right_action, ord=2)),
        action_bound,
        label="published covariance action",
    )
    edf_movement = _up(
        width
        * (
            covariance_movement * _spectral_upper(left_oracle.data_curvature)
            + data_curvature_bound / right_eigen_lower
        )
    )
    edf_bound = _bound_sum(
        edf_movement,
        left.fixed.bounds.edf,
        right.fixed.bounds.edf,
        gamma(max(64 * width * width, 1))
        * max(1.0, abs(left.fixed.total_edf), abs(right.fixed.total_edf)),
        operations=max(64 * width * width, 1),
    )
    _assert_zero_centered(
        left_model.inference.total_edf - right_model.inference.total_edf,
        edf_bound,
        label="published EDF",
    )

    beta_ball = _bound_sum(float(np.linalg.norm(common.center, ord=2)), common.radius)
    penalty_value_bound = _bound_sum(
        _spectral_upper(left_oracle.penalty) * beta_ball * coefficient_bound,
        0.5 * _spectral_upper(left_oracle.penalty) * coefficient_bound**2,
        0.5 * penalty_matrix_bound * beta_ball**2,
        _quadratic_roundoff(left_oracle.coefficients, left_oracle.penalty),
        _quadratic_roundoff(right_oracle.coefficients, right_oracle.penalty),
        operations=max(64 * width, 1),
    )
    log_h = _certify_logdet_pair(
        right_oracle.penalized_curvature,
        left_oracle.penalized_curvature,
        matrix_bound=penalized_curvature_bound,
        full_rank=True,
        label="cross-route penalized curvature",
    )
    log_penalty = _certify_logdet_pair(
        right_oracle.penalty,
        left_oracle.penalty,
        matrix_bound=penalty_matrix_bound,
        full_rank=False,
        label="cross-route scaled penalty",
    )
    laml_bound = _bound_sum(
        likelihood_bound,
        penalty_value_bound,
        0.5 * (log_h.total_bound + log_penalty.total_bound),
        gamma(max(64 * width, 1)) * max(1.0, abs(left.laml), abs(right.laml)),
        operations=max(64 * width, 1),
    )
    exact_left_laml = -left_oracle.penalized_optimizing_log_likelihood + 0.5 * (
        _spectral_logdet(left_oracle.penalized_curvature, full_rank=True).logdet
        - _spectral_logdet(left_oracle.penalty, full_rank=False).logdet
    )
    exact_right_laml = -right_oracle.penalized_optimizing_log_likelihood + 0.5 * (
        _spectral_logdet(right_oracle.penalized_curvature, full_rank=True).logdet
        - _spectral_logdet(right_oracle.penalty, full_rank=False).logdet
    )
    _assert_zero_centered(
        exact_left_laml - exact_right_laml,
        laml_bound,
        label="independent LAML",
    )
    _assert_zero_centered(
        left_model.smoothing.objective - right_model.smoothing.objective,
        _bound_sum(laml_bound, left.laml_bound, right.laml_bound),
        label="published LAML",
    )

    right_on_left = coefficient_oracle(
        left_oracle.response,
        left_oracle.weights,
        semantics=left_oracle.semantics,
        location_design=left_oracle.location_design,
        scale_design=left_oracle.scale_design,
        coefficients=right_oracle.coefficients,
        penalty=left_oracle.penalty,
        location_offset=left_oracle.location_offset,
        scale_offset=left_oracle.scale_offset,
        scale_floor=0.0,
    )
    right_on_left_bounds = oracle_bounds(right_on_left)
    _componentwise_budget(
        left_prediction,
        left_oracle.theta,
        left.fixed.bounds.theta_evaluation,
        label="left terminal prediction",
    )
    _componentwise_budget(
        right_prediction,
        right_on_left.theta,
        right_on_left_bounds.theta_evaluation,
        label="right terminal prediction",
    )
    location_norm = np.linalg.norm(left_oracle.location_design, axis=1)
    scale_norm = np.linalg.norm(left_oracle.scale_design, axis=1)
    location_movement = _array_bound_sum(
        location_norm * coefficient_bound,
        operations=max(32 * width, 1),
    )
    exponent_upper = (
        center_left.eta[:, 1] + center_left_bounds.eta_evaluation[:, 1] + scale_norm * common.radius
    )
    maximum_scale = np.nextafter(
        np.exp(exponent_upper) * (1.0 + gamma(8)),
        np.inf,
    )
    assert np.all(np.isfinite(maximum_scale))
    scale_movement = _array_bound_sum(
        maximum_scale * np.expm1(scale_norm * coefficient_bound),
        operations=max(32 * width, 1),
    )
    prediction_movement = np.column_stack(
        (
            location_movement,
            scale_movement,
        )
    )
    prediction_bound = _array_bound_sum(
        prediction_movement,
        left.fixed.bounds.theta_evaluation,
        right_on_left_bounds.theta_evaluation,
        operations=max(64 * len(left_prediction), 1),
    )
    _componentwise_budget(
        left_prediction,
        right_prediction,
        prediction_bound,
        label="zero-centered terminal predictions",
    )

    _assert_zero_centered(
        left.fixed_point_log_residual - right.fixed_point_log_residual,
        _bound_sum(left.fixed_point_bound, right.fixed_point_bound),
        label="local-mode fixed-point residual",
    )
    assert left_model.inference.rank == right_model.inference.rank
    assert left.penalty_rank == right.penalty_rank


def test_frequency_efs_is_literal_replication_and_all_one_contracts_agree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills frequency rebinding and prior/frequency all-one branch drift."""

    _assert_two_matvec_roundoff_channels()
    fixture = _semantic_efs_fixture()
    frequency_config = _semantic_efs_config(
        tolerance=1.0e-3,
        objective_tolerance=1.0e-7,
    )
    counts = fixture.frequency_counts
    take = np.repeat(np.arange(len(counts), dtype=np.intp), counts.astype(np.intp))
    expanded_frame = fixture.frame.iloc[take].reset_index(drop=True)
    compressed = _fit_semantic_efs(
        family=GaussianLS(scale_floor=0.0),
        semantics="frequency",
        frame=fixture.frame,
        response=fixture.response,
        weights=counts,
        efs_config=frequency_config,
        inner_tolerance=_EFS_REPLICATION_INNER_TOLERANCE,
    )
    expanded = _fit_semantic_efs(
        family=GaussianLS(scale_floor=0.0),
        semantics="frequency",
        frame=expanded_frame,
        response=fixture.response[take],
        weights=np.ones(len(take), dtype=np.float64),
        efs_config=frequency_config,
        inner_tolerance=_EFS_REPLICATION_INNER_TOLERANCE,
    )
    compressed_certificate = _certify_terminal(
        compressed,
        training_frame=fixture.frame,
        expected_semantics="frequency",
        expected_scale_floor=0.0,
        expected_plan_identifier=_expected_gaussian_plan_identifier(
            counts,
            semantics="frequency",
        ),
    )
    expanded_certificate = _certify_terminal(
        expanded,
        training_frame=expanded_frame,
        expected_semantics="frequency",
        expected_scale_floor=0.0,
        expected_plan_identifier=_expected_gaussian_plan_identifier(
            np.ones(len(take), dtype=np.float64),
            semantics="frequency",
        ),
    )
    arithmetic_only = _bound_sum(
        compressed_certificate.fixed.bounds.likelihood_sum,
        expanded_certificate.fixed.bounds.likelihood_sum,
    )
    with pytest.raises(AssertionError, match="zero-centered"):
        _assert_zero_centered(
            compressed_certificate.fixed.oracle.optimizing_log_likelihood
            - expanded_certificate.fixed.oracle.optimizing_log_likelihood,
            arithmetic_only,
            label="missing common-local-root likelihood mutant",
        )
    compressed_prediction = compressed.predict_parameters(fixture.frame)
    expanded_prediction = expanded.predict_parameters(fixture.frame)

    def assert_replication_parity(right_prediction):
        _assert_terminal_parity(
            compressed,
            expanded,
            compressed_certificate,
            expanded_certificate,
            left_prediction=compressed_prediction,
            right_prediction=right_prediction,
        )

    assert_replication_parity(expanded_prediction)
    model_type = type(expanded)
    original_predict_parameters = model_type.predict_parameters

    def corrupted_expanded_prediction(self, X, *, offsets=None):
        values = original_predict_parameters(self, X, offsets=offsets)
        if self is not expanded:
            return values
        corrupted = np.array(values, copy=True)
        corrupted[0, 0] += 1.0
        return corrupted

    with monkeypatch.context() as mutation:
        mutation.setattr(
            model_type,
            "predict_parameters",
            corrupted_expanded_prediction,
        )
        with pytest.raises(AssertionError, match="right terminal prediction"):
            assert_replication_parity(expanded.predict_parameters(fixture.frame))
    np.testing.assert_array_equal(
        expanded.predict_parameters(fixture.frame),
        expanded_prediction,
    )
    assert_replication_parity(expanded_prediction)
    compressed_rows = compressed.fit_state.retained_rows
    expanded_rows = expanded.fit_state.retained_rows
    assert compressed_rows is not None and expanded_rows is not None
    assert compressed_rows.likelihood_weights.provenance.physical_count == len(counts)
    assert expanded_rows.likelihood_weights.provenance.physical_count == len(take)
    assert compressed_rows.likelihood_weights.provenance.likelihood_count == len(take)
    assert expanded_rows.likelihood_weights.provenance.likelihood_count == len(take)

    all_one_config = _semantic_efs_config(
        tolerance=5.0e-3,
        objective_tolerance=1.0e-9,
    )
    unit_weights = np.ones(len(fixture.response), dtype=np.float64)
    all_one_prior = _fit_semantic_efs(
        family=GaussianLS(scale_floor=0.0),
        semantics="prior",
        frame=fixture.frame,
        response=fixture.response,
        weights=unit_weights,
        efs_config=all_one_config,
        inner_tolerance=_EFS_REPLICATION_INNER_TOLERANCE,
    )
    all_one_frequency = _fit_semantic_efs(
        family=GaussianLS(scale_floor=0.0),
        semantics="frequency",
        frame=fixture.frame,
        response=fixture.response,
        weights=unit_weights,
        efs_config=all_one_config,
        inner_tolerance=_EFS_REPLICATION_INNER_TOLERANCE,
    )
    prior_certificate = _certify_terminal(
        all_one_prior,
        training_frame=fixture.frame,
        expected_semantics="prior",
        expected_scale_floor=0.0,
        expected_plan_identifier=_expected_gaussian_plan_identifier(
            unit_weights,
            semantics="prior",
        ),
    )
    frequency_certificate = _certify_terminal(
        all_one_frequency,
        training_frame=fixture.frame,
        expected_semantics="frequency",
        expected_scale_floor=0.0,
        expected_plan_identifier=_expected_gaussian_plan_identifier(
            unit_weights,
            semantics="frequency",
        ),
    )
    _assert_terminal_parity(
        all_one_prior,
        all_one_frequency,
        prior_certificate,
        frequency_certificate,
        left_prediction=all_one_prior.predict_parameters(fixture.frame),
        right_prediction=all_one_frequency.predict_parameters(fixture.frame),
    )
    _assert_all_one_exact_terminal_math(
        all_one_prior,
        all_one_frequency,
        prior_certificate,
        frequency_certificate,
        frame=fixture.frame,
    )
    assert all_one_prior.fit_state.weight_contract == WeightContract("prior")
    assert all_one_frequency.fit_state.weight_contract == WeightContract("frequency")
    assert (
        all_one_prior.fit_state.family_likelihood_plan_identifier
        != all_one_frequency.fit_state.family_likelihood_plan_identifier
    )


def _smooth_fixture(
    *,
    fixed_location: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, tuple[Predictor, Predictor]]:
    rng = np.random.default_rng(1823)
    x = np.linspace(0.0, 1.0, 72)
    z = np.mod(0.37 + 1.7 * x, 1.0)
    sigma = 0.22 + np.exp(-1.2 + 0.35 * np.cos(2.0 * np.pi * z))
    response = 0.35 + 0.8 * np.sin(2.0 * np.pi * x) + rng.normal(scale=sigma)
    location_policy = (
        {"wiggle": LambdaPolicy.fixed(0.75)}
        if fixed_location
        else {"wiggle": LambdaPolicy.estimate()}
    )
    predictors = (
        Predictor(
            "location",
            {
                "x": Spline(
                    kind="cr",
                    n_knots=6,
                    lambda_policy=location_policy,
                )
            },
        ),
        Predictor(
            "scale",
            {
                "z": Spline(
                    kind="cr",
                    n_knots=5,
                    lambda_policy={"wiggle": LambdaPolicy.estimate()},
                )
            },
        ),
    )
    return pd.DataFrame({"x": x, "z": z}), response, predictors


def _decreasing_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def objective(*args, **kwargs) -> float:
        nonlocal calls
        calls += 1
        return -float(calls)

    monkeypatch.setattr(smoothing_objective, "joint_laplace_objective", objective)
    monkeypatch.setattr(efs_module, "joint_laplace_objective", objective)


def _patch_second_iteration_acceleration(
    monkeypatch: pytest.MonkeyPatch,
    *,
    step: float = 0.25,
) -> list[str]:
    decisions: list[str] = []

    def deterministic_proposal(
        self: WindowedTypeIIAnderson,
        *,
        max_log_step: float,
        minimum_log_lambda: float,
        maximum_log_lambda: float,
    ) -> MultisecantDecision:
        del max_log_step, minimum_log_lambda, maximum_log_lambda
        if not decisions:
            decisions.append("warming")
            return MultisecantDecision(proposal=None, refusal_reason="warming")
        decisions.append("proposal")
        current = self._pairs[-1].log_lambdas
        log_step = np.full(current.shape, step)
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

    monkeypatch.setattr(WindowedTypeIIAnderson, "propose", deterministic_proposal)
    return decisions


def _patch_raw_residual_sequence(
    monkeypatch: pytest.MonkeyPatch,
    residuals: tuple[float | dict[str, float], ...],
) -> list[dict[str, float]]:
    """Script equal proposal and stationarity residuals for chronology tests."""

    original_update = efs_module.wood_fasiolo_update
    remaining = iter(residuals)
    observed: list[dict[str, float]] = []

    def sequenced_update(*args, **kwargs) -> EFSUpdateResult:
        update = original_update(*args, **kwargs)
        residual = next(remaining)
        raw = (
            {name: float(residual[name]) for name in update.raw_log_steps}
            if isinstance(residual, dict)
            else {name: float(residual) for name in update.raw_log_steps}
        )
        observed.append(raw)
        return replace(
            update,
            raw_log_steps=raw,
            stationarity_log_residuals=raw,
        )

    monkeypatch.setattr(smoothing_evidence, "wood_fasiolo_update", sequenced_update)
    monkeypatch.setattr(efs_module, "wood_fasiolo_update", sequenced_update)
    return observed


def test_multisecant_config_defaults_and_validation() -> None:
    config = DistributionalEFSConfig(outer="efs")
    assert config.acceleration == "none"
    assert config.acceleration_history == 5
    assert config.acceleration_max_amplification == 8.0

    for value in ("anderson", "bfgs", ""):
        with pytest.raises(ValueError, match="acceleration"):
            DistributionalEFSConfig(outer="efs", acceleration=value)  # type: ignore[arg-type]
    for value in (0, -1, True):
        with pytest.raises(ValueError, match="acceleration_history"):
            DistributionalEFSConfig(outer="efs", acceleration_history=value)  # type: ignore[arg-type]
    for value in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="acceleration_max_amplification"):
            DistributionalEFSConfig(
                outer="efs",
                acceleration_max_amplification=value,  # type: ignore[arg-type]
            )


def test_practical_convergence_config_is_explicit_and_validated() -> None:
    config = DistributionalEFSConfig(outer="efs")
    assert config.practical_convergence is False
    assert config.practical_parameter_tolerance == 1.0e-3

    with pytest.raises(TypeError, match="practical_convergence"):
        DistributionalEFSConfig(outer="efs", practical_convergence=1)  # type: ignore[arg-type]
    for value in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="practical_parameter_tolerance"):
            DistributionalEFSConfig(
                outer="efs",
                practical_parameter_tolerance=value,  # type: ignore[arg-type]
            )


def test_non_gfs_fallback_refuses_and_resets_multisecant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    _decreasing_objective(monkeypatch)
    original_update = efs_module.wood_fasiolo_update
    update_calls = 0

    def fallback_update(*args, **kwargs):
        nonlocal update_calls
        update = original_update(*args, **kwargs)
        update_calls += 1
        names = tuple(update.lambdas)
        kind = "fixed_point_fallback" if update_calls == 2 else "gfs"
        return replace(
            update,
            raw_log_steps=dict.fromkeys(names, 0.25),
            stationarity_log_residuals=dict.fromkeys(names, 0.25),
            proposal_kinds=dict.fromkeys(names, kind),
        )

    monkeypatch.setattr(smoothing_evidence, "wood_fasiolo_update", fallback_update)
    monkeypatch.setattr(efs_module, "wood_fasiolo_update", fallback_update)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs", max_iterations=3, acceleration="multisecant"
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert [iteration.acceleration_outcome for iteration in smoothing.history] == [
        "warming",
        "refused",
        "warming",
    ]
    assert smoothing.history[1].acceleration_refusal_reason == "non_gfs_proposal"


def test_working_infinity_direction_cannot_certify_from_a_small_finite_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    original_update = efs_module.wood_fasiolo_update

    def working_infinity_update(*args, **kwargs):
        update = original_update(*args, **kwargs)
        names = tuple(update.lambdas)
        return replace(
            update,
            raw_log_steps=dict.fromkeys(names, np.nextafter(0.0, math.inf)),
            stationarity_log_residuals=dict.fromkeys(names, 0.0),
            proposal_kinds=dict.fromkeys(names, "working_infinity"),
        )

    monkeypatch.setattr(smoothing_evidence, "wood_fasiolo_update", working_infinity_update)
    monkeypatch.setattr(efs_module, "wood_fasiolo_update", working_infinity_update)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=3,
            tolerance=1.0e-6,
            plateau_tolerance=1.0,
            practical_convergence=True,
            practical_parameter_tolerance=1.0,
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "max_iterations"
    assert smoothing.terminal_raw_max_log_step > 0.0


def test_accelerated_log_endpoint_preserves_the_exact_lambda_box() -> None:
    config = DistributionalEFSConfig(
        outer="efs",
        minimum_lambda=0.1,
        maximum_lambda=10.0,
    )
    current = {"smooth": 1.0}
    names = ("smooth",)
    upper_log = math.log(config.maximum_lambda)

    proposal = efs_module._accelerated_proposal(
        current,
        names,
        np.array([upper_log]),
        np.array([upper_log]),
        config,
    )

    assert proposal is not None
    lambdas, steps = proposal
    assert lambdas == {"smooth": 10.0}
    assert math.isfinite(lambdas["smooth"])
    assert config.minimum_lambda <= lambdas["smooth"] <= config.maximum_lambda
    assert steps == {"smooth": upper_log}
    assert (
        efs_module._accelerated_proposal(
            current,
            names,
            np.array([np.nextafter(upper_log, math.inf)]),
            np.array([upper_log]),
            config,
        )
        is None
    )


def test_default_efs_iterations_publish_acceleration_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    _decreasing_objective(monkeypatch)

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(outer="efs", max_iterations=1),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    iteration = smoothing.history[0]
    assert iteration.acceleration_outcome in {
        "disabled",
        "warming",
        "refused",
        "accepted",
        "rejected",
    }
    assert iteration.raw_backtracks >= 0
    assert isinstance(iteration.boundary_nominations, tuple)


def test_acceleration_telemetry_rejects_unknown_reasons_and_orphan_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    _decreasing_objective(monkeypatch)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(outer="efs", max_iterations=1),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    iteration = smoothing.history[0]
    with pytest.raises(ValueError, match="refusal reason"):
        replace(
            iteration,
            acceleration_outcome="refused",
            acceleration_refusal_reason="invented",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="chronology"):
        replace(
            smoothing,
            coefficient_fits=(*smoothing.coefficient_fits, smoothing.coefficient_fits[-1]),
        )


def test_rejected_acceleration_falls_back_to_raw_scale_one_from_same_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    objectives = iter((0.0, -1.0, 0.0, -2.0))
    objective_calls = 0

    def objective(*args, **kwargs) -> float:
        nonlocal objective_calls
        objective_calls += 1
        return next(objectives)

    decisions = _patch_second_iteration_acceleration(monkeypatch)
    recorded_states: list[np.ndarray] = []
    original_record = WindowedTypeIIAnderson.record_accepted
    original_reject = WindowedTypeIIAnderson.reject
    rejection_count = 0

    def record_spy(self, *, log_lambdas, raw_residual, provenance):
        recorded_states.append(np.array(log_lambdas, copy=True))
        return original_record(
            self,
            log_lambdas=log_lambdas,
            raw_residual=raw_residual,
            provenance=provenance,
        )

    def reject_spy(self) -> None:
        nonlocal rejection_count
        rejection_count += 1
        original_reject(self)

    warm_starts: list[np.ndarray | None] = []
    original_solver = efs_module.fit_dense_fixed_lambda

    def solver_spy(*args, **kwargs):
        initial = kwargs.get("initial")
        warm_starts.append(None if initial is None else np.array(initial, copy=True))
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(smoothing_objective, "joint_laplace_objective", objective)
    monkeypatch.setattr(efs_module, "joint_laplace_objective", objective)
    monkeypatch.setattr(WindowedTypeIIAnderson, "record_accepted", record_spy)
    monkeypatch.setattr(WindowedTypeIIAnderson, "reject", reject_spy)
    monkeypatch.setattr(smoothing_authority, "fit_dense_fixed_lambda", solver_spy)
    monkeypatch.setattr(smoothing_loop, "fit_dense_fixed_lambda", solver_spy)
    monkeypatch.setattr(efs_module, "fit_dense_fixed_lambda", solver_spy)

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=2,
            tolerance=1.0e-12,
            max_backtracks=2,
            acceleration="multisecant",
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert objective_calls == 4
    assert decisions == ["warming", "proposal"]
    assert len(recorded_states) == 2
    assert rejection_count == 1
    second = smoothing.history[1]
    assert second.coefficient_fit_indices == (2, 3)
    assert second.acceleration_outcome == "rejected"
    assert second.accelerated_fit_index == 2
    assert second.accepted_fit_index == 3
    assert second.raw_backtracks == 0
    assert smoothing.raw_fallback_count == 1
    source = smoothing.coefficient_fits[second.source_fit_index]
    np.testing.assert_array_equal(warm_starts[2], source.coefficients)
    np.testing.assert_array_equal(warm_starts[3], source.coefficients)
    assert smoothing.terminal_fit_index != second.accelerated_fit_index


def test_admissible_acceleration_is_the_only_trial_when_objective_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    objectives = iter((0.0, -1.0, -2.0))
    monkeypatch.setattr(
        smoothing_objective, "joint_laplace_objective", lambda *args, **kwargs: next(objectives)
    )
    monkeypatch.setattr(
        efs_module, "joint_laplace_objective", lambda *args, **kwargs: next(objectives)
    )
    decisions = _patch_second_iteration_acceleration(monkeypatch)

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=2,
            tolerance=1.0e-12,
            acceleration="multisecant",
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    second = smoothing.history[1]
    assert decisions == ["warming", "proposal"]
    assert second.coefficient_fit_indices == (2,)
    assert second.acceleration_outcome == "accepted"
    assert second.accelerated_fit_index == second.accepted_fit_index == 2
    assert second.raw_backtracks == 0
    assert smoothing.accelerated_trial_count == 1
    assert smoothing.accelerated_accept_count == 1
    assert smoothing.raw_fallback_count == 0
    with pytest.raises(ValueError, match="accelerated_fit_index"):
        replace(second, accelerated_fit_index=np.int64(2))


def test_acceleration_with_changed_fit_provenance_falls_back_to_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    objectives = iter((0.0, -1.0, -2.0, -3.0))
    monkeypatch.setattr(
        smoothing_objective, "joint_laplace_objective", lambda *args, **kwargs: next(objectives)
    )
    monkeypatch.setattr(
        efs_module, "joint_laplace_objective", lambda *args, **kwargs: next(objectives)
    )
    _patch_second_iteration_acceleration(monkeypatch)
    solver_calls = 0
    original_solver = efs_module.fit_dense_fixed_lambda

    def changed_accelerated_provenance(*args, **kwargs):
        nonlocal solver_calls
        solver_calls += 1
        result = original_solver(*args, **kwargs)
        if solver_calls != 3:
            return result
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

    monkeypatch.setattr(
        smoothing_authority, "fit_dense_fixed_lambda", changed_accelerated_provenance
    )
    monkeypatch.setattr(smoothing_loop, "fit_dense_fixed_lambda", changed_accelerated_provenance)
    monkeypatch.setattr(efs_module, "fit_dense_fixed_lambda", changed_accelerated_provenance)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=2,
            tolerance=1.0e-12,
            acceleration="multisecant",
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    second = smoothing.history[1]
    assert second.acceleration_outcome == "rejected"
    assert second.accelerated_fit_index == 2
    assert second.accepted_fit_index == 3
    assert second.raw_backtracks == 0
    assert smoothing.terminal_fit_index == 3


def test_warming_and_current_duplicate_refusal_add_no_coefficient_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    _decreasing_objective(monkeypatch)
    proposal_calls = 0

    def refuse_after_warming(self, **kwargs) -> MultisecantDecision:
        nonlocal proposal_calls
        del self, kwargs
        proposal_calls += 1
        reason = "warming" if proposal_calls == 1 else "current_duplicate"
        return MultisecantDecision(proposal=None, refusal_reason=reason)

    monkeypatch.setattr(WindowedTypeIIAnderson, "propose", refuse_after_warming)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=2,
            tolerance=1.0e-12,
            acceleration="multisecant",
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    first, second = smoothing.history
    assert first.acceleration_outcome == "warming"
    assert first.acceleration_refusal_reason is None
    assert first.coefficient_fit_indices == (1,)
    assert second.acceleration_outcome == "refused"
    assert second.acceleration_refusal_reason == "current_duplicate"
    assert second.coefficient_fit_indices == (2,)
    assert len(smoothing.coefficient_fits) == 3
    assert smoothing.accelerated_trial_count == 0


def test_real_multisecant_controller_warms_without_an_extra_coefficient_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    _decreasing_objective(monkeypatch)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=1,
            tolerance=1.0e-12,
            acceleration="multisecant",
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert len(smoothing.coefficient_fits) == 2
    assert smoothing.history[0].coefficient_fit_indices == (1,)
    assert smoothing.history[0].acceleration_outcome == "warming"
    assert smoothing.history[0].acceleration_refusal_reason is None
    assert smoothing.history[0].accelerated_fit_index is None


def test_rejected_acceleration_is_not_backtracked_before_raw_trials_all_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    calls = 0

    def objective(*args, **kwargs) -> float:
        nonlocal calls
        calls += 1
        return -1.0 if calls == 2 else 0.0

    monkeypatch.setattr(smoothing_objective, "joint_laplace_objective", objective)
    monkeypatch.setattr(efs_module, "joint_laplace_objective", objective)
    decisions = _patch_second_iteration_acceleration(monkeypatch)
    max_backtracks = 2
    raw_scales: list[float] = []
    original_scaled_proposal = efs_module._scaled_proposal

    def scaled_proposal_spy(current, steps, estimated_names, scale, config):
        raw_scales.append(scale)
        return original_scaled_proposal(current, steps, estimated_names, scale, config)

    monkeypatch.setattr(smoothing_loop, "_scaled_proposal", scaled_proposal_spy)
    monkeypatch.setattr(smoothing_objective, "_scaled_proposal", scaled_proposal_spy)
    monkeypatch.setattr(efs_module, "_scaled_proposal", scaled_proposal_spy)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=2,
            tolerance=1.0e-12,
            max_backtracks=max_backtracks,
            acceleration="multisecant",
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    second = smoothing.history[1]
    assert decisions == ["warming", "proposal"]
    assert raw_scales == [1.0, 1.0, 0.5, 0.25]
    assert second.acceleration_outcome == "rejected"
    assert second.coefficient_fit_indices == (2, 3, 4, 5)
    assert second.accelerated_fit_index == 2
    assert second.accepted is False
    assert second.raw_backtracks == max_backtracks
    assert second.backtracks == max_backtracks + 1
    assert smoothing.terminal_fit_index == 1
    assert smoothing.raw_fallback_count == 1


def test_small_accepted_acceleration_does_not_supply_terminal_step_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    objectives = iter((0.0, -1.0, -2.0))
    monkeypatch.setattr(
        smoothing_objective, "joint_laplace_objective", lambda *args, **kwargs: next(objectives)
    )
    monkeypatch.setattr(
        efs_module, "joint_laplace_objective", lambda *args, **kwargs: next(objectives)
    )
    _patch_second_iteration_acceleration(monkeypatch, step=1.0e-7)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=2,
            tolerance=1.0e-6,
            acceleration="multisecant",
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert smoothing.history[-1].acceleration_outcome == "accepted"
    assert smoothing.history[-1].max_accepted_log_step < smoothing.config.tolerance
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "max_iterations"


def test_tiny_acceleration_with_unsettled_fresh_raw_residual_checks_next_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills treating the accepted accelerated step as terminal authority."""

    frame, response, predictors = _smooth_fixture()
    _decreasing_objective(monkeypatch)
    decisions = _patch_second_iteration_acceleration(monkeypatch, step=1.0e-7)
    raw_updates = _patch_raw_residual_sequence(monkeypatch, (0.25, 0.25, 0.25, 0.25))

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=3,
            tolerance=1.0e-6,
            acceleration="multisecant",
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert decisions == ["warming", "proposal", "proposal"]
    assert smoothing.history[1].acceleration_outcome == "accepted"
    assert smoothing.history[1].max_accepted_log_step < smoothing.config.tolerance
    assert len(raw_updates) == 1 + len(smoothing.history)
    assert smoothing.terminal_raw_max_log_step == 0.25
    assert smoothing.unresolved_upper_bound == ()
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "max_iterations"


def test_last_acceleration_with_zero_fresh_raw_residual_converges_without_another_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills deferring accepted-state evidence until a budget-forbidden iteration."""

    frame, response, predictors = _smooth_fixture()
    monkeypatch.setattr(smoothing_objective, "joint_laplace_objective", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(efs_module, "joint_laplace_objective", lambda *args, **kwargs: 0.0)
    decisions = _patch_second_iteration_acceleration(monkeypatch, step=1.0e-7)
    raw_updates = _patch_raw_residual_sequence(monkeypatch, (0.25, 0.25, 0.0))

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=2,
            tolerance=1.0e-6,
            plateau_tolerance=0.0,
            plateau_iterations=1,
            acceleration="multisecant",
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert decisions == ["warming", "proposal"]
    assert len(raw_updates) == 3
    assert len(smoothing.coefficient_fits) == 3
    assert smoothing.terminal_fit_index == 2
    assert smoothing.terminal_raw_max_log_step == 0.0
    assert smoothing.unresolved_upper_bound == ()
    assert smoothing.converged is True
    assert smoothing.convergence_reason == "objective_plateau"


def test_last_iteration_publishes_recomputed_raw_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills reusing the source update when the accepted fit exhausts the budget."""

    frame, response, predictors = _smooth_fixture()
    _decreasing_objective(monkeypatch)
    raw_updates = _patch_raw_residual_sequence(monkeypatch, (0.25, 0.125))

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(outer="efs", max_iterations=1, tolerance=1.0e-6),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert len(raw_updates) == 2
    assert len(smoothing.coefficient_fits) == 2
    assert smoothing.terminal_raw_max_log_step == 0.125
    assert smoothing.unresolved_upper_bound == ()
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "max_iterations"


def test_accepted_acceleration_with_unsettled_raw_residual_is_not_a_plateau(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    objectives = iter((0.0, -1.0, -1.0))
    monkeypatch.setattr(
        smoothing_objective, "joint_laplace_objective", lambda *args, **kwargs: next(objectives)
    )
    monkeypatch.setattr(
        efs_module, "joint_laplace_objective", lambda *args, **kwargs: next(objectives)
    )
    _patch_second_iteration_acceleration(monkeypatch, step=1.0e-7)
    raw_residual_norms: list[float] = []
    original_update = efs_module.wood_fasiolo_update

    def update_spy(*args, **kwargs):
        update = original_update(*args, **kwargs)
        raw_residual_norms.append(max(abs(value) for value in update.raw_log_steps.values()))
        return update

    monkeypatch.setattr(smoothing_evidence, "wood_fasiolo_update", update_spy)
    monkeypatch.setattr(efs_module, "wood_fasiolo_update", update_spy)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=2,
            tolerance=1.0e-12,
            plateau_tolerance=0.0,
            plateau_iterations=1,
            acceleration="multisecant",
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert raw_residual_norms[-1] > smoothing.config.tolerance
    assert smoothing.history[-1].acceleration_outcome == "accepted"
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "max_iterations"


def test_flat_objective_with_unsettled_fresh_raw_residual_is_not_a_plateau(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills objective-plateau termination without fresh convergence evidence."""

    frame, response, predictors = _smooth_fixture()
    monkeypatch.setattr(smoothing_objective, "joint_laplace_objective", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(efs_module, "joint_laplace_objective", lambda *args, **kwargs: 0.0)
    raw_updates = _patch_raw_residual_sequence(monkeypatch, (0.25, 0.25, 0.25))

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=2,
            tolerance=1.0e-6,
            plateau_tolerance=0.0,
            plateau_iterations=1,
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert len(raw_updates) == 3
    assert len(smoothing.history) == 2
    assert smoothing.terminal_raw_max_log_step == 0.25
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "max_iterations"


def test_flat_objective_may_retain_plateau_reason_after_fresh_raw_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    monkeypatch.setattr(smoothing_objective, "joint_laplace_objective", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(efs_module, "joint_laplace_objective", lambda *args, **kwargs: 0.0)
    raw_updates = _patch_raw_residual_sequence(monkeypatch, (0.25, 0.0))

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=1,
            tolerance=1.0e-6,
            plateau_tolerance=0.0,
            plateau_iterations=1,
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert len(raw_updates) == 2
    assert smoothing.terminal_raw_max_log_step == 0.0
    assert smoothing.unresolved_upper_bound == ()
    assert smoothing.converged is True
    assert smoothing.convergence_reason == "objective_plateau"


def test_default_and_explicit_none_routes_are_bit_identical() -> None:
    frame, response, predictors = _smooth_fixture()
    fit_kwargs = {
        "family": GaussianLS(),
        "weight_contract": WeightContract(semantics="prior"),
        "predictors": predictors,
        "lambdas": {"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
    }
    default = fit_dense_distributional(
        frame,
        response,
        efs_config=DistributionalEFSConfig(outer="efs", max_iterations=2),
        **fit_kwargs,
    )
    explicit = fit_dense_distributional(
        frame,
        response,
        efs_config=DistributionalEFSConfig(outer="efs", max_iterations=2, acceleration="none"),
        **fit_kwargs,
    )

    assert default.smoothing is not None and explicit.smoothing is not None
    np.testing.assert_array_equal(default.coefficients, explicit.coefficients)
    assert default.lambdas == explicit.lambdas
    assert default.smoothing.initial_objective == explicit.smoothing.initial_objective
    assert default.smoothing.objective == explicit.smoothing.objective
    assert default.smoothing.history == explicit.smoothing.history
    np.testing.assert_array_equal(default.covariance, explicit.covariance)
    assert default.inference.total_edf == explicit.inference.total_edf
    assert default.inference.predictor_edf == explicit.inference.predictor_edf
    np.testing.assert_array_equal(default.predict(frame), explicit.predict(frame))


def test_fixed_components_never_move_and_fixed_only_fit_stops_cleanly() -> None:
    frame, response, predictors = _smooth_fixture(fixed_location=True)
    fixed_only = (predictors[0], Predictor("scale", {}))

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=fixed_only,
        lambdas={"location:x#wiggle": 99.0},
        efs_config=DistributionalEFSConfig(outer="efs", max_iterations=3),
    )

    assert model.lambdas == {"location:x#wiggle": 0.75}
    assert model.smoothing is not None
    assert model.smoothing.converged is True
    assert model.smoothing.convergence_reason == "fixed_only"
    assert model.smoothing.terminal_raw_max_log_step == 0.0
    assert model.smoothing.unresolved_upper_bound == ()
    assert model.smoothing.history == ()
    assert len(model.smoothing.coefficient_fits) == 1
    assert model.result.converged is True
    with pytest.raises(ValueError, match="fresh convergence"):
        replace(
            model.smoothing,
            terminal_raw_max_log_step=np.nextafter(model.smoothing.config.tolerance, math.inf),
        )
    with pytest.raises(ValueError, match="unresolved upper"):
        replace(
            model.smoothing,
            unresolved_upper_bound=("location:x#wiggle",),
        )

    tampered = replace(model.smoothing)
    object.__setattr__(
        tampered,
        "terminal_raw_max_log_step",
        np.nextafter(tampered.config.tolerance, math.inf),
    )
    assert tampered.matched_certified is False
    with pytest.raises(RuntimeError, match="fresh convergence"):
        tampered.assert_matched_certified()


def _upper_cap_refusal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, EFSUpdateResult, DistributionalEFSResult]:
    frame, response, predictors = _smooth_fixture(fixed_location=True)
    target = "scale:z#wiggle"
    maximum_lambda = 10.0
    raw_step = 2.0**-30
    synthetic_components = (
        EFSComponentState(
            name="location:x#wiggle",
            coefficient_slice=slice(0, 1),
            penalty=np.ones((1, 1)),
            rank=1.0,
            lambda_value=0.75,
            policy=LambdaPolicy.fixed(0.75),
        ),
        EFSComponentState(
            name=target,
            coefficient_slice=slice(1, 2),
            penalty=np.ones((1, 1)),
            rank=1.0,
            lambda_value=maximum_lambda,
            policy=LambdaPolicy.estimate(),
        ),
    )
    synthetic_update = wood_fasiolo_update(
        synthetic_components,
        np.array([0.0, math.sqrt(1.0 / (maximum_lambda * math.exp(raw_step)))]),
        np.zeros((2, 2)),
        max_log_step=2.0,
        maximum_lambda=maximum_lambda,
    )
    assert synthetic_update.lambdas[target] == maximum_lambda
    assert synthetic_update.log_steps[target] == 0.0
    assert 0.0 < synthetic_update.raw_log_steps[target] < 1.0e-6
    saturation_update = replace(
        synthetic_update,
        trace_terms={"location:x#wiggle": 0.0, target: 0.099},
    )
    assert efs_module._saturated_names(
        synthetic_components,
        saturation_update,
        (target,),
        0.95,
    ) == frozenset({target})
    monkeypatch.setattr(
        smoothing_evidence, "wood_fasiolo_update", lambda *args, **kwargs: synthetic_update
    )
    monkeypatch.setattr(efs_module, "wood_fasiolo_update", lambda *args, **kwargs: synthetic_update)

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={target: maximum_lambda},
        # the forged non-stationary cap score is only consistent with a Fisher-path fit
        config=DenseSolverConfig(coefficient_curvature="fisher"),
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=2,
            tolerance=1.0e-6,
            maximum_lambda=maximum_lambda,
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    return target, synthetic_update, smoothing


def test_positive_raw_pressure_at_the_exact_upper_cap_is_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills using the bounded zero step as convergence at the finite cap."""

    target, synthetic_update, smoothing = _upper_cap_refusal_result(monkeypatch)
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "lambda_cap_unresolved"
    assert smoothing.terminal_raw_max_log_step == pytest.approx(
        synthetic_update.raw_log_steps[target]
    )
    assert smoothing.unresolved_upper_bound == (target,)
    assert len(smoothing.history) == 1
    refusal = smoothing.history[0]
    assert refusal.accepted is False
    assert refusal.refused_face_components == (target,)
    assert refusal.endpoint_direction_evidence is None
    assert refusal.endpoint_assessment_failure_reason == "provenance_changed"
    assert len(smoothing.coefficient_fits) == 3
    with pytest.raises(ValueError, match="failure reason"):
        replace(refusal, endpoint_assessment_failure_reason=None)
    with pytest.raises(ValueError, match="provenance failure reason"):
        replace(
            smoothing,
            history=(replace(refusal, endpoint_assessment_failure_reason="analytic_unavailable"),),
        )
    assert smoothing.matched_certified is False
    with pytest.raises(RuntimeError, match="certification"):
        smoothing.assert_matched_certified()
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(smoothing, terminal_raw_max_log_step=-1.0)
    with pytest.raises(ValueError, match="positive fresh convergence"):
        replace(smoothing, terminal_raw_max_log_step=0.0)
    with pytest.raises(ValueError, match="positive fresh convergence"):
        replace(
            smoothing,
            convergence_reason="coefficient_not_converged",
            terminal_raw_max_log_step=0.0,
        )
    with pytest.raises(ValueError, match="nonempty unresolved"):
        replace(smoothing, unresolved_upper_bound=())
    with pytest.raises(ValueError, match="exact configured maximum"):
        replace(smoothing, unresolved_upper_bound=("location:x#wiggle",))
    with pytest.raises(ValueError, match="invalid EFS convergence reason"):
        replace(
            smoothing,
            convergence_reason="boundary_heuristic_stopped",  # type: ignore[arg-type]
        )


def test_cap_not_stationary_receipt_authenticates_retained_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills trusting the convergence flag or stored score telemetry at the cap."""

    _, _, smoothing = _upper_cap_refusal_result(monkeypatch)
    source_fit, cap_fit, endpoint_fit = smoothing.coefficient_fits
    refusal = smoothing.history[0]
    tolerance = refusal.coefficient_tolerances[0]
    cap_objective = cap_fit.penalized_optimizing_log_likelihood
    assert cap_objective is not None
    nonstationary_score = np.zeros_like(cap_fit.terminal_score)
    nonstationary_score[0] = 2.0 * tolerance * (1.0 + abs(cap_objective))
    nonstationary_cap = replace(
        cap_fit,
        terminal_score=nonstationary_score,
        score_relative=0.0,
    )
    receipt = replace(
        refusal,
        coefficient_fit_indices=(1,),
        coefficient_tolerances=(tolerance,),
        endpoint_assessment_failure_reason="cap_not_stationary",
    )

    authenticated = replace(
        smoothing,
        history=(receipt,),
        coefficient_fits=(source_fit, nonstationary_cap),
    )
    assert authenticated.history[0].endpoint_assessment_failure_reason == "cap_not_stationary"

    boundary_score = np.zeros_like(nonstationary_score)
    boundary_score[0] = tolerance * (1.0 + abs(cap_objective))
    assert np.linalg.norm(boundary_score, ord=np.inf) > tolerance
    stationary_cap_with_forged_telemetry = replace(
        nonstationary_cap,
        terminal_score=boundary_score,
        score_relative=1.0,
    )
    with pytest.raises(ValueError, match="cap stationarity failure reason"):
        replace(
            authenticated,
            coefficient_fits=(source_fit, stationary_cap_with_forged_telemetry),
        )

    nonconverged_cap = replace(
        nonstationary_cap,
        converged=False,
        convergence_reason="max_iterations",
    )
    with pytest.raises(ValueError, match="cap stationarity failure reason"):
        replace(authenticated, coefficient_fits=(source_fit, nonconverged_cap))

    wrong_config_cap = replace(
        nonstationary_cap,
        config=replace(nonstationary_cap.config, tolerance=2.0 * tolerance),
    )
    with pytest.raises(ValueError, match="tight tolerance"):
        replace(authenticated, coefficient_fits=(source_fit, wrong_config_cap))

    two_fit_receipt = replace(
        receipt,
        coefficient_fit_indices=(1, 2),
        coefficient_tolerances=(tolerance, tolerance),
    )
    with pytest.raises(ValueError, match="failure reason disagrees with its fits"):
        replace(
            smoothing,
            history=(two_fit_receipt,),
            coefficient_fits=(source_fit, nonstationary_cap, endpoint_fit),
        )

    legacy_failed_cap = replace(
        nonstationary_cap,
        converged=False,
        convergence_reason="max_iterations",
    )
    legacy_receipt = replace(
        receipt,
        endpoint_assessment_failure_reason="cap_not_converged",
    )
    legacy = replace(
        smoothing,
        history=(legacy_receipt,),
        coefficient_fits=(source_fit, legacy_failed_cap),
    )
    assert legacy.history[0].endpoint_assessment_failure_reason == "cap_not_converged"
    with pytest.raises(ValueError, match="endpoint cap failure reason"):
        replace(legacy, coefficient_fits=(source_fit, cap_fit))


def test_endpoint_not_stationary_receipt_authenticates_retained_face_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills using full-space or stored score telemetry for endpoint authority."""

    _, _, smoothing = _upper_cap_refusal_result(monkeypatch)
    source_fit, cap_fit, endpoint_fit = smoothing.coefficient_fits
    refusal = smoothing.history[0]
    tolerance = refusal.coefficient_tolerances[0]
    endpoint_objective = endpoint_fit.penalized_optimizing_log_likelihood
    endpoint_face = endpoint_fit.coefficient_face
    assert endpoint_objective is not None and endpoint_face is not None
    nonstationary_reduced_score = np.zeros(endpoint_face.reduced_width)
    nonstationary_reduced_score[0] = 2.0 * tolerance * (1.0 + abs(endpoint_objective))
    nonstationary_endpoint = replace(
        endpoint_fit,
        terminal_score=endpoint_face.lift_vector(nonstationary_reduced_score),
        score_relative=0.0,
        convergence_reason="objective_and_step",
    )
    receipt = replace(
        refusal,
        endpoint_assessment_failure_reason="endpoint_not_stationary",
    )
    cap_objective = cap_fit.penalized_optimizing_log_likelihood
    assert cap_objective is not None
    boundary_cap_score = np.zeros_like(cap_fit.terminal_score)
    boundary_cap_score[0] = tolerance * (1.0 + abs(cap_objective))
    boundary_cap = replace(
        cap_fit,
        terminal_score=boundary_cap_score,
        score_relative=0.0,
    )

    authenticated = replace(
        smoothing,
        history=(receipt,),
        coefficient_fits=(source_fit, boundary_cap, nonstationary_endpoint),
    )
    assert authenticated.history[0].endpoint_assessment_failure_reason == "endpoint_not_stationary"

    retained_direction = endpoint_face.null_basis[:, 0]
    retained_unit_norm = float(
        np.max(np.abs(endpoint_face.reduce_vector(retained_direction)), initial=0.0)
    )
    boundary_coefficient = tolerance * (1.0 + abs(endpoint_objective)) / retained_unit_norm
    for _ in range(64):
        endpoint_boundary_score = retained_direction * boundary_coefficient
        retained_boundary_ratio = float(
            np.max(
                np.abs(endpoint_face.reduce_vector(endpoint_boundary_score)),
                initial=0.0,
            )
            / (1.0 + abs(endpoint_objective))
        )
        if retained_boundary_ratio == tolerance:
            break
        boundary_coefficient = np.nextafter(
            boundary_coefficient,
            0.0 if retained_boundary_ratio > tolerance else math.inf,
        )
    else:
        raise AssertionError("could not construct the exact retained-score boundary")
    assert np.linalg.norm(endpoint_boundary_score, ord=np.inf) > tolerance
    boundary_endpoint_with_forged_telemetry = replace(
        endpoint_fit,
        terminal_score=endpoint_boundary_score,
        score_relative=1.0,
    )
    with pytest.raises(ValueError, match="endpoint stationarity failure reason"):
        replace(
            authenticated,
            coefficient_fits=(source_fit, boundary_cap, boundary_endpoint_with_forged_telemetry),
        )

    nonstationary_cap_score = np.zeros_like(cap_fit.terminal_score)
    nonstationary_cap_score[0] = 2.0 * tolerance * (1.0 + abs(cap_objective))
    nonstationary_cap = replace(
        cap_fit,
        terminal_score=nonstationary_cap_score,
        score_relative=0.0,
        convergence_reason="objective_and_step",
    )
    with pytest.raises(ValueError, match="stationary endpoint cap"):
        replace(
            authenticated,
            coefficient_fits=(source_fit, nonstationary_cap, nonstationary_endpoint),
        )

    constrained_score = (
        endpoint_face.constraint_basis[:, 0] * 100.0 * tolerance * (1.0 + abs(endpoint_objective))
    )
    stationary_endpoint_with_forged_telemetry = replace(
        endpoint_fit,
        terminal_score=constrained_score,
        score_relative=1.0,
    )
    with pytest.raises(ValueError, match="endpoint stationarity failure reason"):
        replace(
            authenticated,
            coefficient_fits=(
                source_fit,
                boundary_cap,
                stationary_endpoint_with_forged_telemetry,
            ),
        )

    nonconverged_endpoint = replace(
        nonstationary_endpoint,
        converged=False,
        convergence_reason="max_iterations",
    )
    with pytest.raises(ValueError, match="endpoint stationarity failure reason"):
        replace(
            authenticated,
            coefficient_fits=(source_fit, boundary_cap, nonconverged_endpoint),
        )

    wrong_tolerance_receipt = replace(
        receipt,
        coefficient_tolerances=(tolerance, 2.0 * tolerance),
    )
    with pytest.raises(ValueError, match="tight tolerance"):
        replace(
            smoothing,
            history=(wrong_tolerance_receipt,),
            coefficient_fits=(source_fit, boundary_cap, nonstationary_endpoint),
        )

    one_fit_receipt = replace(
        receipt,
        coefficient_fit_indices=(1,),
        coefficient_tolerances=(tolerance,),
    )
    with pytest.raises(ValueError, match="failure reason disagrees with its fits"):
        replace(
            smoothing,
            history=(one_fit_receipt,),
            coefficient_fits=(source_fit, boundary_cap),
        )

    with pytest.raises(ValueError, match="wrong exact coefficient face"):
        replace(authenticated, coefficient_fits=(source_fit, boundary_cap, cap_fit))

    legacy_failed_endpoint = replace(
        nonstationary_endpoint,
        converged=False,
        convergence_reason="max_iterations",
    )
    legacy_receipt = replace(
        receipt,
        endpoint_assessment_failure_reason="endpoint_not_converged",
    )
    legacy = replace(
        smoothing,
        history=(legacy_receipt,),
        coefficient_fits=(source_fit, nonstationary_cap, legacy_failed_endpoint),
    )
    assert legacy.history[0].endpoint_assessment_failure_reason == "endpoint_not_converged"
    with pytest.raises(ValueError, match="endpoint fit failure reason"):
        replace(
            legacy,
            coefficient_fits=(source_fit, nonstationary_cap, endpoint_fit),
        )


def test_each_accepted_lambda_update_refits_and_uses_fresh_observed_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture(fixed_location=True)
    _decreasing_objective(monkeypatch)
    scalar_calls = 0

    def forbid_scalar_efs(*args, **kwargs):
        nonlocal scalar_calls
        scalar_calls += 1
        raise AssertionError("distributional EFS must not call scalar EFS")

    monkeypatch.setattr(scalar_efs, "optimize_efs_reml", forbid_scalar_efs)
    solver_calls: list[tuple[np.ndarray | None, object | None, object | None, object]] = []
    original_solver = efs_module.fit_dense_fixed_lambda

    def solver_spy(*args, **kwargs):
        initial = kwargs.get("initial")
        reuse_session = kwargs.get("_reuse_session")
        reuse_source = kwargs.get("_reuse_source")
        result = original_solver(*args, **kwargs)
        solver_calls.append(
            (
                None if initial is None else np.array(initial, copy=True),
                reuse_session,
                reuse_source,
                result,
            )
        )
        return result

    inverse_inputs: list[np.ndarray] = []
    original_update = efs_module.wood_fasiolo_update

    def update_spy(*args, **kwargs):
        inverse_inputs.append(np.array(args[2], copy=True))
        assert kwargs["inverse_scale"] == 1.0
        return original_update(*args, **kwargs)

    monkeypatch.setattr(smoothing_authority, "fit_dense_fixed_lambda", solver_spy)
    monkeypatch.setattr(smoothing_loop, "fit_dense_fixed_lambda", solver_spy)
    monkeypatch.setattr(efs_module, "fit_dense_fixed_lambda", solver_spy)
    monkeypatch.setattr(smoothing_evidence, "wood_fasiolo_update", update_spy)
    monkeypatch.setattr(efs_module, "wood_fasiolo_update", update_spy)

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"scale:z#wiggle": 0.2},
        config=DenseSolverConfig(tolerance=1.0e-8),
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=2,
            tolerance=1.0e-12,
            max_log_step=1.0,
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert len(smoothing.history) == 2
    assert all(iteration.accepted for iteration in smoothing.history)
    assert scalar_calls == 0
    assert solver_calls[0][1] is not None
    assert solver_calls[0][2] is None
    assert len(inverse_inputs) == 1 + len(smoothing.history)
    for inverse, iteration in zip(inverse_inputs[:-1], smoothing.history, strict=True):
        source = smoothing.coefficient_fits[iteration.source_fit_index]
        np.testing.assert_allclose(inverse, source.terminal_rank.pseudo_inverse())
        assert iteration.update_curvature.requested_source == "observed"
        assert iteration.coefficient_tolerances == (1.0e-8,)
        assert iteration.accepted_fit_index is not None
        warm_start, reuse_session, reuse_source, accepted = solver_calls[
            iteration.accepted_fit_index
        ]
        assert warm_start is not None
        np.testing.assert_array_equal(warm_start, source.coefficients)
        assert reuse_session is solver_calls[0][1]
        assert reuse_source is source
        assert accepted is smoothing.coefficient_fits[iteration.accepted_fit_index]
    np.testing.assert_allclose(
        inverse_inputs[-1],
        smoothing.terminal_fit.terminal_rank.pseudo_inverse(),
    )
    assert model.lambdas["location:x#wiggle"] == 0.75
    assert model.lambdas["scale:z#wiggle"] != 0.2


def test_shared_block_efs_components_use_effective_logdet_ranks() -> None:
    frame, response, _ = _smooth_fixture()
    predictors = (
        Predictor(
            "location",
            {
                "x": Spline(kind="cr", n_knots=5),
                "z": Spline(kind="cr", n_knots=5),
            },
            interactions=(("x", "z"),),
        ),
        Predictor("scale", {}),
    )
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={
            "location:x#wiggle": 0.6,
            "location:z#wiggle": 0.4,
            "location:x:z#margin_x": 0.2,
            "location:x:z#margin_z": 1.7,
        },
    )
    effective_ranks, _ = compute_logdet_s_derivatives(
        dict(model.lambdas),
        list(model.layout.penalties),
    )

    states = efs_module._component_states(model.layout, model.lambdas)

    assert {state.name: state.rank for state in states} == pytest.approx(effective_ranks)
    tensor_states = [state for state in states if "#margin_" in state.name]
    static_ranks = {component.name: component.rank for component in model.layout.penalties}
    assert any(state.rank != pytest.approx(static_ranks[state.name]) for state in tensor_states)


def _spectral_log_pdet(penalty: np.ndarray) -> float:
    """Return log|S|₊ from an explicit symmetric eigendecomposition.

    Deliberately independent of ``superglm.solvers.rank``: this sums ``log``
    over the strictly positive spectrum and certifies its own rank cut by
    demanding the same eigenvalue count across five decades of relative
    threshold.  A genuine rank-deficient penalty leaves a wide plateau between
    its numerical zeros and its smallest real eigenvalue, so a count that moves
    with the threshold means the oracle itself cannot be trusted and the test
    must say so rather than pin whichever integer the eigensolver rounded to.
    """
    symmetric = 0.5 * (penalty + penalty.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = float(np.max(np.abs(eigenvalues)))
    assert scale > 0.0
    counts = {
        int(np.count_nonzero(eigenvalues > rcond * scale))
        for rcond in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7)
    }
    assert len(counts) == 1, f"penalty rank is threshold-dependent across decades: {counts}"
    positive = eigenvalues[eigenvalues > 1e-10 * scale]
    return float(np.sum(np.log(positive)))


def test_joint_objective_matches_the_laplace_criterion_over_the_penalty_spectrum() -> None:
    """The joint objective is the generic Laplace criterion, no dispersion profiled.

    ``-penalized_optimizing_log_likelihood + (log|H| - log|S|₊) / 2``, with ``log|S|₊``
    taken over the penalty's true positive spectrum.

    The oracle here is an explicit ``eigvalsh`` sum, NOT
    ``decompose_gram(result.penalty).log_pdet``.  Do not restore that one: on
    this fixture the penalty is 15x15 of true rank 11, with four numerical
    zeros of order 1e-15 and below, and ``decompose_gram`` returns rank 12 and
    ``log_pdet = -21.010`` where the spectrum gives ``+17.633`` identically at
    every relative threshold from 1e-12 to 1e-7.  That is the known
    over-ranking defect in ``superglm.solvers.rank`` (its cut sits a factor of
    ``m`` below what a backward-stable symmetric eigensolver resolves), and it
    put 38.643 nats -- 19.321 after the one half -- into the expectation.  The
    shipped objective was right and this test's old expectation was wrong; it
    is the acceptance criterion of the EFS backtracking line search, so letting
    the old oracle win would have made EFS accept wrong lambda steps.
    """
    frame, response, predictors = _smooth_fixture()
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.6, "scale:z#wiggle": 0.4},
    )

    expected = -model.result.penalized_optimizing_log_likelihood + 0.5 * (
        model.result.terminal_rank.log_pdet - _spectral_log_pdet(model.result.penalty)
    )

    assert efs_module.joint_laplace_objective(
        model.result,
        layout=model.layout,
        lambdas=model.lambdas,
    ) == pytest.approx(expected)


def test_joint_objective_uses_structural_penalty_logdet_not_a_fresh_matrix_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superglm.reml.penalty_algebra import compute_logdet_s_plus

    frame, response, predictors = _smooth_fixture()
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.6, "scale:z#wiggle": 0.4},
    )
    expected_logdet = compute_logdet_s_plus(
        dict(model.lambdas),
        list(model.layout.penalties),
    )
    expected = -model.result.penalized_optimizing_log_likelihood + 0.5 * (
        model.result.terminal_rank.log_pdet - expected_logdet
    )

    def unstable_full_matrix_rank(*args, **kwargs):
        raise AssertionError("the penalty rank must come from structural component metadata")

    monkeypatch.setattr(rank_module, "decompose_gram", unstable_full_matrix_rank)

    actual = efs_module.joint_laplace_objective(
        model.result,
        layout=model.layout,
        lambdas=model.lambdas,
    )

    assert actual == pytest.approx(expected)


def test_joint_objective_requires_structural_penalty_metadata() -> None:
    frame, response, predictors = _smooth_fixture()
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.6, "scale:z#wiggle": 0.4},
    )

    with pytest.raises(TypeError, match="layout"):
        efs_module.joint_laplace_objective(model.result)


@pytest.mark.slow
def test_live_joint_laplace_safeguard_never_accepts_a_worse_outer_state() -> None:
    frame, response, predictors = _smooth_fixture()
    config = DistributionalEFSConfig(
        outer="efs",
        max_iterations=3,
        max_backtracks=2,
        max_log_step=0.75,
    )
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.4, "scale:z#wiggle": 0.3},
        efs_config=config,
    )

    smoothing = model.smoothing
    assert smoothing is not None
    for iteration in smoothing.history:
        ceiling = iteration.objective_before + config.objective_tolerance * (
            1.0 + abs(iteration.objective_before)
        )
        if iteration.accepted:
            assert iteration.objective_after <= ceiling
        else:
            assert iteration.objective_after == iteration.objective_before
            assert iteration.lambdas_after == iteration.lambdas_before
    assert smoothing.objective == pytest.approx(
        efs_module.joint_laplace_objective(
            model.result,
            layout=model.layout,
            lambdas=model.lambdas,
        )
    )
    assert model.lambdas == smoothing.lambdas


def test_rejected_outer_step_rolls_back_lambda_and_coefficient_state_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    calls = 0

    def rejecting_objective(*args, **kwargs) -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 1.0

    monkeypatch.setattr(smoothing_objective, "joint_laplace_objective", rejecting_objective)
    monkeypatch.setattr(efs_module, "joint_laplace_objective", rejecting_objective)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.3, "scale:z#wiggle": 0.2},
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=3,
            max_backtracks=2,
            objective_tolerance=1.0e-12,
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "objective_rejected"
    assert smoothing.lambdas == smoothing.initial_lambdas
    assert model.lambdas == smoothing.initial_lambdas
    assert smoothing.terminal_fit_index == 0
    assert model.result is smoothing.coefficient_fits[0]
    assert len(smoothing.coefficient_fits) == 4
    assert len(smoothing.history) == 1
    rejected = smoothing.history[0]
    assert rejected.accepted is False
    assert rejected.accepted_fit_index is None
    assert rejected.coefficient_fit_indices == (1, 2, 3)
    assert rejected.objective_before == rejected.objective_after == 0.0


def test_curvature_fallback_is_auditable_and_blocks_matched_certification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    _decreasing_objective(monkeypatch)
    original_solver = efs_module.fit_dense_fixed_lambda

    def fallback_solver(*args, **kwargs):
        result = original_solver(*args, **kwargs)
        observed = result.terminal_curvature
        telemetry = CurvatureTelemetry(
            requested_source="observed",
            actual_source="fisher",
            reason="material_indefiniteness_after_retry",
            minimum_eigenvalue=observed.minimum_eigenvalue,
            rank=observed.rank,
            condition_estimate=observed.condition_estimate,
            fallback_count=1,
        )
        return replace(result, terminal_curvature=telemetry)

    monkeypatch.setattr(smoothing_authority, "fit_dense_fixed_lambda", fallback_solver)
    monkeypatch.setattr(smoothing_loop, "fit_dense_fixed_lambda", fallback_solver)
    monkeypatch.setattr(efs_module, "fit_dense_fixed_lambda", fallback_solver)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.5, "scale:z#wiggle": 0.5},
        efs_config=DistributionalEFSConfig(outer="efs", max_iterations=1, max_log_step=0.5),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert smoothing.fallback_count >= 1
    assert smoothing.matched_certified is False
    assert smoothing.history[0].update_curvature.actual_source == "fisher"
    with pytest.raises(RuntimeError, match="curvature fallback"):
        smoothing.assert_matched_certified()


def test_inner_and_smoothing_convergence_are_reported_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, predictors = _smooth_fixture()
    original_solver = efs_module.fit_dense_fixed_lambda
    original_update = efs_module.wood_fasiolo_update
    diagnostic_raw_maxima: list[float] = []

    def unconverged_solver(*args, **kwargs):
        result = original_solver(*args, **kwargs)
        return replace(result, converged=False, convergence_reason="max_iterations")

    def diagnostic_update(*args, **kwargs):
        update = original_update(*args, **kwargs)
        diagnostic_raw_maxima.append(
            max(abs(value) for value in update.stationarity_log_residuals.values())
        )
        return update

    monkeypatch.setattr(smoothing_authority, "fit_dense_fixed_lambda", unconverged_solver)
    monkeypatch.setattr(smoothing_loop, "fit_dense_fixed_lambda", unconverged_solver)
    monkeypatch.setattr(efs_module, "fit_dense_fixed_lambda", unconverged_solver)
    monkeypatch.setattr(smoothing_evidence, "wood_fasiolo_update", diagnostic_update)
    monkeypatch.setattr(efs_module, "wood_fasiolo_update", diagnostic_update)
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.5, "scale:z#wiggle": 0.5},
        efs_config=DistributionalEFSConfig(outer="efs", max_iterations=2),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert smoothing.coefficient_converged is False
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "coefficient_not_converged"
    assert smoothing.history == ()
    assert len(diagnostic_raw_maxima) == 1
    assert math.isfinite(smoothing.terminal_raw_max_log_step)
    assert smoothing.terminal_raw_max_log_step == diagnostic_raw_maxima[0]
    assert model.result.converged is False


# ── PenaltyComponent contract on the path that actually selects lambda ──
#
# ``_component_states`` hands ``wood_fasiolo_update`` the block it uses for
# ``beta.T @ S_j @ beta`` and ``trace(H^-1 S_j)``.  A ``PenaltyComponent``
# stores a COMPACT representation, so for three of master's four
# ``penalty_kind`` values the stored omega is not that block: ``identity``
# stores no array at all, while ``repeated`` and ``sum_to_zero`` store a single
# level block that expands by Kronecker product / sum-to-zero contrast.
# Reading ``omega_ssp`` (falling back to ``omega_raw``) therefore selected
# lambda against a penalty the solver never applied.

_EFS_ROWS = 24
_EFS_BLOCK = 3


def _efs_penalty_group_matrix(kind: str, n_levels: int):
    """Return ``(group_matrix, group_width)`` for one master ``penalty_kind``."""
    codes = np.arange(_EFS_ROWS, dtype=np.intp) % n_levels
    if kind == "identity":
        return RandomEffectGroupMatrix(codes, n_levels), n_levels
    if kind in ("repeated", "sum_to_zero"):
        basis = sp.csr_matrix(np.eye(_EFS_BLOCK)[np.arange(_EFS_ROWS) % _EFS_BLOCK])
        matrix = FactorSmoothGroupMatrix(
            basis,
            codes,
            n_levels,
            natural_map=np.eye(_EFS_BLOCK),
            levels=tuple(range(n_levels)),
            repeated_penalty_components=(("wiggle", np.diag([2.0, 0.8, 0.0])),),
            factor_basis="fs" if kind == "repeated" else "sz",
        )
        return matrix, matrix.coefficient_levels * _EFS_BLOCK
    if kind == "dense":
        width = n_levels * _EFS_BLOCK
        rng = np.random.default_rng(4471)
        matrix = SparseSSPGroupMatrix(
            sp.csr_matrix(rng.normal(size=(_EFS_ROWS, width))),
            np.triu(np.ones((width, width))) * 0.5 + np.eye(width),
        )
        difference = np.diff(np.eye(width), 2, axis=0)
        matrix.omega = difference.T @ difference
        return matrix, width
    raise AssertionError(f"unhandled penalty kind {kind!r}")


def _efs_single_group_layout(kind: str, n_levels: int):
    """Return ``(layout, group_matrix, width)`` around one real group of that kind."""
    group_matrix, width = _efs_penalty_group_matrix(kind, n_levels)
    group = GroupSlice(name="g", start=0, end=width)
    components = build_penalty_components(
        [group_matrix],
        collect_reml_groups([group], [group_matrix]),
    )
    compiled = CompiledPredictorDesign(
        design=DesignMatrix([group_matrix], _EFS_ROWS, width),
        groups=(group,),
        specs={},
        feature_order=(),
        interaction_specs={},
        interaction_order=(),
    )
    build = CompiledPredictor(
        name="location",
        parameter_index=0,
        link=IdentityLink(),
        compiled=compiled,
        intercept=False,
        offset=np.zeros(_EFS_ROWS),
        penalties=tuple(components),
    )
    return build_stacked_layout((build,)), group_matrix, width


# The level penalty ``_efs_penalty_group_matrix`` declares for the two
# expanding kinds.  Its rank is 2 of 3, so a misbuilt expansion shows up as a
# changed eigenvalue pattern and not only as a changed scale.
_EFS_LEVEL_PENALTY = np.diag([2.0, 0.8, 0.0])

# ``C.T @ C`` for the sum-to-zero contrast ``C = [I; -1]`` that maps ``L - 1``
# free level blocks onto ``L`` raw blocks summing to zero, so the Gram is
# ``I + J``.  Written out per level count so the expectation cannot re-derive
# itself from the code under test.
_EFS_SUM_TO_ZERO_LEVEL_GRAM = {
    2: np.array([[2.0]]),
    3: np.array([[2.0, 1.0], [1.0, 2.0]]),
}


def _efs_expected_block_from_definition(
    kind: str,
    n_levels: int,
    group_matrix,
    width: int,
) -> np.ndarray:
    """Return the unweighted block each kind is DEFINED to hand EFS.

    Written from each kind's own definition, never read back from
    ``penalty_component_dense_matrix``: that is the expander the EFS state and
    ``layout.penalty_matrix`` BOTH call, so comparing them to each other passes
    for any wrong expansion they happen to share.
    """
    if kind == "identity":
        # A random effect penalises every level coefficient at unit strength.
        return np.eye(n_levels)
    if kind == "repeated":
        # ``n_levels`` independent copies of the same level penalty.
        return np.kron(np.eye(n_levels), _EFS_LEVEL_PENALTY)
    if kind == "sum_to_zero":
        # The same level penalty seen through the sum-to-zero contrast.
        return np.kron(_EFS_SUM_TO_ZERO_LEVEL_GRAM[n_levels], _EFS_LEVEL_PENALTY)
    if kind == "dense":
        # The second-difference penalty this fixture assigns as ``omega``,
        # carried into solver space by the group's own root, ``R_inv.T Omega R_inv``.
        difference = np.diff(np.eye(width), 2, axis=0)
        r_inv = np.asarray(group_matrix.R_inv, dtype=np.float64)
        return r_inv.T @ (difference.T @ difference) @ r_inv
    raise AssertionError(f"no independent definition for kind {kind!r}")


@pytest.mark.parametrize("kind", ["identity", "sum_to_zero", "repeated", "dense"])
@pytest.mark.parametrize("n_levels", [2, 3])
def test_efs_component_penalty_matches_each_kinds_own_definition(
    kind: str,
    n_levels: int,
) -> None:
    """Every kind must reach EFS as the block its own definition prescribes.

    Two live failure modes ride on this: ``identity`` reached EFS as a 0-d
    object array and raised, while a two-level ``sum_to_zero`` reached it at
    half strength and silently selected the wrong lambda.

    The expectation is independent arithmetic -- ``np.eye``, an explicit
    ``kron`` against a hand-written level Gram, an explicit congruence by the
    group's root.  It is deliberately NOT ``layout.penalty_matrix``: that
    delegates to the same ``penalty_component_dense_matrix`` the EFS state
    reads, so an expander returning ``0.5 * block`` halves both sides and every
    parametrisation still passes -- measured, and precisely the silent-2x class
    this test exists to catch.  Agreement with the solver's own matrix is still
    asserted below, but as a consistency check on top of a correct oracle
    rather than in place of one.
    """
    layout, group_matrix, width = _efs_single_group_layout(kind, n_levels)
    assert {component.penalty_kind for component in layout.penalties} == {kind}

    lambdas = {name: 1.0 for name in layout.penalty_names}
    states = efs_module._component_states(layout, lambdas)
    assert len(states) == len(layout.penalties)

    expected = _efs_expected_block_from_definition(kind, n_levels, group_matrix, width)
    assert np.any(expected != 0.0)

    for component, state in zip(layout.penalties, states, strict=True):
        penalty = np.asarray(state.penalty, dtype=np.float64)
        assert penalty.shape == expected.shape
        np.testing.assert_allclose(penalty, expected, atol=1e-12)

        # And the block EFS holds is still the one the coefficient fit was
        # minimised against, which is what ``fit_dense_fixed_lambda`` uses.
        isolated = {name: 0.0 for name in layout.penalty_names}
        isolated[component.name] = 1.0
        embedded = layout.penalty_matrix(isolated)[component.group_sl, component.group_sl]
        np.testing.assert_allclose(penalty, embedded)


def test_a_random_effect_fits_under_efs_and_recovers_its_group_variance() -> None:
    """An identity penalty stores no array, so EFS never ran on one at all.

    ``component.omega_ssp`` and ``component.omega_raw`` are both ``None`` for a
    ``RandomEffect``, so the old read produced ``np.asarray(None)`` — a 0-d
    object array — and ``EFSComponentState`` rejected it.  A random effect fit
    under fixed lambdas but died the moment lambda was estimated.

    The solver subtracts ``0.5 * lambda * b.T @ b`` from the log-likelihood, so
    the MAP prior is ``b ~ N(0, 1/lambda)`` and a converged EFS lambda must come
    back near the reciprocal of the group-effect variance that was simulated.
    """
    rng = np.random.default_rng(101)
    n_sites, per_site = 40, 30
    standard = rng.normal(size=n_sites)
    site_codes = np.repeat(np.arange(n_sites), per_site)

    recovered: dict[float, float] = {}
    for group_sd in (0.3, 0.6):
        effects = group_sd * standard
        noise = np.random.default_rng(707).normal(scale=0.4, size=n_sites * per_site)
        response = 1.3 + effects[site_codes] + noise
        frame = pd.DataFrame({"site": [f"s{code}" for code in site_codes]})

        model = fit_dense_distributional(
            frame,
            response,
            family=GaussianLS(),
            weight_contract=WeightContract(semantics="prior"),
            predictors=(
                Predictor("location", {"site": RandomEffect()}),
                Predictor("scale", {}),
            ),
            efs_config=DistributionalEFSConfig(outer="efs", max_iterations=60),
        )

        smoothing = model.smoothing
        assert smoothing is not None
        assert smoothing.converged is True
        # Either converged reason is correct here: the objective plateau and the
        # lambda-step bound are satisfied on the same iteration for this fit, and
        # the recovered lambda below is identical whichever one reports it.
        assert smoothing.convergence_reason in {"lambda_change", "objective_plateau"}

        lam = model.lambdas["location:site#wiggle"]
        recovered[group_sd] = lam

        # Truth as simulated: the centred variance of the drawn effects, since
        # the predictor's intercept absorbs their mean.
        realized_variance = float(np.var(effects - effects.mean()))
        assert 0.85 <= lam * realized_variance <= 1.15
        # And still close to the nominal variance the draw was scaled to.
        assert 0.75 <= lam * group_sd**2 <= 1.25

    # Doubling the simulated group sd must quarter the recovered lambda.
    assert 3.6 <= recovered[0.3] / recovered[0.6] <= 4.4


def _null_space_fixture() -> tuple[pd.DataFrame, np.ndarray, tuple[Predictor, Predictor]]:
    """A location smooth on a covariate the response does not depend on.

    Its true effect is zero, which lies in the null space of a second-derivative
    penalty, so its REML-optimal lambda is ``+inf``.  The other two smooths are
    genuinely wiggly and have finite optima.
    """
    # Sized and seeded so the boundary component's saturation crossing is
    # SUSTAINED rather than transient: below roughly this size the null-space
    # term's lambda has not grown enough for the penalty to dominate its block,
    # and the plateau rule stops the fit first.
    rng = np.random.default_rng(2024)
    n = 400
    x = np.linspace(0.0, 1.0, n)
    z = np.mod(0.37 + 1.7 * x, 1.0)
    w = rng.uniform(0.0, 1.0, n)
    sigma = 0.22 + np.exp(-1.2 + 0.35 * np.cos(2.0 * np.pi * z))
    response = 0.35 + 0.8 * np.sin(2.0 * np.pi * x) + rng.normal(scale=sigma)
    estimate = {"wiggle": LambdaPolicy.estimate()}
    predictors = (
        Predictor(
            "location",
            {
                "x": Spline(kind="cr", n_knots=6, lambda_policy=estimate),
                "w": Spline(kind="cr", n_knots=6, lambda_policy=estimate),
            },
        ),
        Predictor("scale", {"z": Spline(kind="cr", n_knots=5, lambda_policy=estimate)}),
    )
    return pd.DataFrame({"x": x, "z": z, "w": w}), response, predictors


def test_boundary_nomination_is_telemetry_only_and_preserves_the_numerical_route() -> None:
    frame, response, predictors = _null_space_fixture()
    nominated = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        efs_config=DistributionalEFSConfig(
            outer="efs", max_iterations=200, boundary_saturation=0.95
        ),
    )
    disabled = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        efs_config=DistributionalEFSConfig(outer="efs", max_iterations=200),
    )

    left = nominated.smoothing
    right = disabled.smoothing
    assert left is not None and right is not None
    assert any(iteration.boundary_nominations for iteration in left.history)
    assert all(not iteration.boundary_nominations for iteration in right.history)
    assert left.iterations == right.iterations
    assert dict(left.lambdas) == dict(right.lambdas)
    assert left.objective == right.objective
    assert left.terminal_fit_index == right.terminal_fit_index
    np.testing.assert_array_equal(nominated.coefficients, disabled.coefficients)
    assert (
        tuple(replace(iteration, boundary_nominations=()) for iteration in left.history)
        == right.history
    )

    terminal_components = efs_module._component_states(nominated.layout, left.lambdas)
    estimated_names = efs_module._estimated_names(terminal_components)
    terminal_update = wood_fasiolo_update(
        terminal_components,
        left.terminal_fit.coefficients,
        left.terminal_fit.terminal_rank.pseudo_inverse(),
        inverse_scale=1.0,
        max_log_step=left.config.max_log_step,
        minimum_lambda=left.config.minimum_lambda,
        maximum_lambda=left.config.maximum_lambda,
    )
    proposal_by_name = {
        name: abs(float(terminal_update.raw_log_steps[name])) for name in estimated_names
    }
    stationarity_by_name = {
        name: abs(float(terminal_update.stationarity_log_residuals[name]))
        for name in estimated_names
    }
    largest_proposal_name = max(proposal_by_name, key=proposal_by_name.__getitem__)
    nominated_names = {
        name for iteration in left.history for name in iteration.boundary_nominations
    }
    assert largest_proposal_name in nominated_names
    assert proposal_by_name[largest_proposal_name] > 0.0
    assert left.terminal_raw_max_log_step == max(stationarity_by_name.values())
    assert left.converged == right.converged
    assert left.convergence_reason == right.convergence_reason
    assert left.matched_certified == right.matched_certified


def test_an_interior_boundary_nomination_never_grants_convergence() -> None:
    frame, response, predictors = _smooth_fixture()
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        efs_config=DistributionalEFSConfig(
            outer="efs", max_iterations=200, boundary_saturation=0.95
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    # This deliberately demonstrates why saturation is telemetry: a finite
    # interior fit can cross the same threshold and retain positive raw drift.
    assert any(iteration.boundary_nominations for iteration in smoothing.history)
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "max_iterations"
    assert smoothing.terminal_raw_max_log_step > smoothing.config.tolerance
    assert smoothing.unresolved_upper_bound == ()
    assert smoothing.matched_certified is False


def test_boundary_saturation_must_be_a_usable_fraction() -> None:
    for value in (0.0, -0.25, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="boundary_saturation"):
            DistributionalEFSConfig(outer="efs", boundary_saturation=value)
    # one disables the rule: the ratio it thresholds is bounded above by one,
    # and disabled is the default
    assert DistributionalEFSConfig(outer="efs").boundary_saturation == 1.0
    # A zero streak would make nomination or plateau qualification immediate;
    # fresh convergence evidence still independently controls convergence.
    for value in (0, -1):
        with pytest.raises(ValueError, match="boundary_iterations"):
            DistributionalEFSConfig(outer="efs", boundary_iterations=value)
        with pytest.raises(ValueError, match="plateau_iterations"):
            DistributionalEFSConfig(outer="efs", plateau_iterations=value)


def test_high_start_boundary_telemetry_never_changes_the_fit() -> None:
    """Boundary nominations remain telemetry even when their classifier fires.

    A fit started far above a finite optimum is saturated from iteration one
    while EFS initially walks lambda down. True GFS can later propose an upward
    correction, so neither saturation nor instantaneous direction separates a
    finite optimum from infinity. The telemetry must not change the route.
    """
    frame, response, predictors = _smooth_fixture()
    high_start = {"location:x#wiggle": 100.0, "scale:z#wiggle": 100.0}
    guarded = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas=high_start,
        efs_config=DistributionalEFSConfig(
            outer="efs", max_iterations=3000, boundary_saturation=0.95
        ),
    )
    reference = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas=high_start,
        efs_config=DistributionalEFSConfig(outer="efs", max_iterations=3000),
    )

    assert guarded.smoothing is not None and reference.smoothing is not None
    assert any(iteration.boundary_nominations for iteration in guarded.smoothing.history)
    assert guarded.smoothing.iterations == reference.smoothing.iterations
    assert guarded.smoothing.objective == reference.smoothing.objective
    assert guarded.smoothing.converged == reference.smoothing.converged
    assert guarded.smoothing.convergence_reason == reference.smoothing.convergence_reason
    assert guarded.smoothing.matched_certified == reference.smoothing.matched_certified
    np.testing.assert_array_equal(guarded.coefficients, reference.coefficients)
    for name, reference_value in dict(reference.smoothing.lambdas).items():
        assert dict(guarded.smoothing.lambdas)[name] == float(reference_value)
