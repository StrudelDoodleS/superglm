"""Versioned native artifacts for complete dense distributional fitted state.

The JSON envelope is inspectable and integrity checked.  Its payload uses the
Python pickle protocol because fitted feature specifications contain learned
Python and SciPy objects whose semantics cannot be reconstructed from a small
configuration dictionary.  Consequently, artifacts must only be loaded from
trusted sources; the SHA-256 digest detects corruption, not malicious input.

Schema versioning
-----------------
``SCHEMA_VERSION`` is a semantic version, and MAJOR is the only component the
reader consults:

* **MAJOR is a read barrier unless explicitly supported.** Schema 8 retains
  its original telemetry encoding and family-dependent curvature scope. Known
  pre-contract majors receive the typed legacy-weight refusal; other majors
  receive an ordinary version error.
* **MINOR and PATCH promise readability.**  A reader accepts any artifact that
  shares its major.

:func:`deserialize_distributional_model` recomputes the manifest from the
restored payload and compares it to the stored one *for equality*.  Any change
to :func:`distributional_manifest` output — a new key, a renamed key, a changed
value encoding — is therefore a read barrier by construction and **must** be a
MAJOR bump.  MINOR and PATCH are reserved for changes that leave manifest
output identical.  ``test_manifest_key_set_is_pinned_to_the_current_major``
pins the manifest key set so a manifest change that forgets the MAJOR bump
fails a test instead of shipping.

That is what buys the invariant the manifest check exists to provide: once the
version gate has passed, a manifest mismatch means the artifact was tampered
with or corrupted, and never merely that it is old.
"""

from __future__ import annotations

import base64
import copyreg
import dataclasses
import hashlib
import hmac
import io
import json
import math
import pickle
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import scipy.sparse as sp

from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.families.log_normal import LogNormalLS
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.families.two_piece import (
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
)
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    DefaultPredictionFamily,
    DistributionalFamily,
    FamilyCapabilities,
    ParameterSpec,
)
from superglm.distributional.fit_state import (
    CompactNullModel,
    DistributionalFitState,
    DistributionalRowState,
)
from superglm.distributional.inference import JointInference
from superglm.distributional.model import DenseDistributionalModel
from superglm.distributional.result import (
    CHUNKED_EXECUTION_BACKEND_IDENTIFIER,
    DENSE_EXECUTION_BACKEND_IDENTIFIER,
    DenseSolverConfig,
    DenseSolverResult,
    DistributionalEFSConfig,
    DistributionalEFSIteration,
    DistributionalEFSResult,
    DistributionalFitResult,
    EndpointDirectionEvidence,
    JointEndpointDirectionEvidence,
    validate_solver_likelihood_decomposition,
)
from superglm.distributional.serialization_schema import (
    READABLE_PREVIOUS_MAJORS,
    SCHEMA_VERSION,
    qualified_type,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.distributional.weights import (
    LegacyPowerWeightArtifactError,
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
    WeightContract,
    WeightProvenance,
)
from superglm.solvers.rank import RankDecomposition
from superglm.types import GroupSlice, LambdaPolicy, PenaltyComponent

_ARTIFACT_TYPE = "superglm.DenseDistributionalModel"
_PAYLOAD_ENCODING = "python-pickle-v5-base64"
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))


class DistributionalSerializationError(ValueError):
    """A distributional artifact is incomplete, corrupt, or incompatible."""


class _InvalidRestoredStateError(DistributionalSerializationError):
    """Internal marker for a payload rejected by live dataclass invariants."""


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise DistributionalSerializationError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DistributionalSerializationError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise DistributionalSerializationError(f"{name} must be a finite number")
    return number


def _slice_manifest(value: slice) -> list[int]:
    if (
        value.step not in (None, 1)
        or not isinstance(value.start, int)
        or not isinstance(value.stop, int)
    ):
        raise DistributionalSerializationError("only bounded contiguous slices are serializable")
    return [value.start, value.stop]


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_descriptor(values: Any) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_digest(array),
    }


def _metadata_value(value: Any) -> Any:
    """Return deterministic, bounded JSON metadata for learned Python objects."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float | np.floating):
        return _finite_float(value, name="metadata value")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return {"array": _array_descriptor(value)}
    if sp.issparse(value):
        matrix = value.tocsr()
        return {
            "sparse_array": {
                "format": "csr",
                "shape": list(matrix.shape),
                "data": _array_descriptor(matrix.data),
                "indices": _array_descriptor(matrix.indices),
                "indptr": _array_descriptor(matrix.indptr),
            }
        }
    if isinstance(value, slice):
        return {"slice": _slice_manifest(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_metadata_value(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "type": qualified_type(value),
            "fields": {
                field.name: _metadata_value(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    return {"type": qualified_type(value)}


def _object_state(value: object) -> dict[str, Any]:
    state = vars(value) if hasattr(value, "__dict__") else {}
    return {
        "type": qualified_type(value),
        "state": {name: _metadata_value(item) for name, item in sorted(state.items())},
    }


def _dataclass_config(value: object) -> dict[str, Any]:
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise DistributionalSerializationError(
            f"{qualified_type(value)} does not expose dataclass configuration"
        )
    result = {
        field.name: _metadata_value(getattr(value, field.name))
        for field in dataclasses.fields(value)
    }
    if isinstance(value, DenseSolverConfig) and result.get("newton_decrement_tolerance") is None:
        result.pop("newton_decrement_tolerance")
    return result


def _telemetry_manifest(value: CurvatureTelemetry) -> dict[str, Any]:
    if not isinstance(value, CurvatureTelemetry):
        raise DistributionalSerializationError("curvature telemetry is missing")
    return value.to_dict()


def _parameter_manifest(parameter: ParameterSpec, fitted_link: object) -> dict[str, Any]:
    support = parameter.support
    return {
        "name": parameter.name,
        "role": parameter.role,
        "curvature": parameter.curvature,
        "support": {
            "lower": support.lower,
            "upper": support.upper,
            "lower_inclusive": support.lower_inclusive,
            "upper_inclusive": support.upper_inclusive,
        },
        "default_link": _object_state(parameter.default_link)
        if not isinstance(parameter.default_link, str)
        else {"name": parameter.default_link},
        "fitted_link": _object_state(fitted_link),
    }


def _family_manifest(
    family: DistributionalFamily,
    saved_family: _SavedFamily,
) -> dict[str, Any]:
    to_config = getattr(family, "to_config", None)
    if not callable(to_config):
        raise DistributionalSerializationError("family must expose complete to_config() state")
    config = to_config()
    if not isinstance(config, Mapping):
        raise DistributionalSerializationError("family to_config() must return a mapping")
    saved_family = _saved_family_for_manifest_pair(
        saved_family.python_type,
        config.get("type"),
    )
    saved_family.validate_config(config)
    capabilities = cast(FamilyCapabilities, getattr(family, "capabilities"))
    prediction_family = cast(DefaultPredictionFamily, family)
    return {
        "python_type": saved_family.python_type,
        "config": _metadata_value(config),
        "default_prediction_name": prediction_family.default_prediction_name,
        "capabilities": _dataclass_config(capabilities),
    }


def _lambda_policy_manifest(policy: LambdaPolicy | None) -> dict[str, Any] | None:
    if policy is None:
        return None
    return {"mode": policy.mode, "value": policy.value}


def _group_manifest(group: GroupSlice) -> dict[str, Any]:
    return {
        "name": group.name,
        "slice": [group.start, group.end],
        "size": group.size,
        "feature_name": group.feature_name,
        "penalized": group.penalized,
        "subgroup_type": group.subgroup_type,
        "monotone_engine": group.monotone_engine,
    }


def _penalty_manifest(component: PenaltyComponent, lam: float) -> dict[str, Any]:
    return {
        "name": component.name,
        "group_name": component.group_name,
        "group_index": component.group_index,
        "group_slice": _slice_manifest(component.group_sl),
        "component_type": component.component_type,
        "lambda_policy": _lambda_policy_manifest(component.lambda_policy),
        "lambda": _finite_float(lam, name=f"lambda {component.name!r}"),
        "rank": _finite_float(component.rank, name=f"rank {component.name!r}"),
        "log_det_omega_plus": _finite_float(
            component.log_det_omega_plus,
            name=f"log determinant {component.name!r}",
        ),
        "omega_raw": (
            None if component.omega_raw is None else _array_descriptor(component.omega_raw)
        ),
        "omega_ssp": (
            None if component.omega_ssp is None else _array_descriptor(component.omega_ssp)
        ),
        "positive_eigenvalues": (
            None if component.eigvals_omega is None else _array_descriptor(component.eigvals_omega)
        ),
        "penalty_kind": component.penalty_kind,
        "repeat_count": int(component.repeat_count),
        "block_width": (None if component.block_width is None else int(component.block_width)),
    }


class _OwnershipRegistry:
    def __init__(self) -> None:
        self._identifiers: dict[int, int] = {}

    def describe(self, value: object) -> dict[str, Any]:
        identity = id(value)
        if identity not in self._identifiers:
            self._identifiers[identity] = len(self._identifiers)
        return {"ownership_id": self._identifiers[identity], **_object_state(value)}


def _predictors_manifest(model: DenseDistributionalModel) -> list[dict[str, Any]]:
    ownership = _OwnershipRegistry()
    result: list[dict[str, Any]] = []
    for template, compiled, state in zip(
        model.fit_state.predictor_templates,
        model.compiled_predictors,
        model.layout.predictors,
        strict=True,
    ):
        template_features = [
            {
                "name": name,
                **ownership.describe(template.features[name]),
            }
            for name in template.features
        ]
        compiled_features = [
            {
                "name": name,
                **ownership.describe(compiled.compiled.specs[name]),
            }
            for name in compiled.compiled.feature_order
        ]
        compiled_interactions = [
            {
                "name": name,
                **ownership.describe(compiled.compiled.interaction_specs[name]),
            }
            for name in compiled.compiled.interaction_order
        ]
        result.append(
            {
                "name": state.name,
                "parameter_index": state.parameter_index,
                "intercept": compiled.intercept,
                "interactions": [list(pair) for pair in template.interactions],
                "coefficient_slice": _slice_manifest(state.coefficient_slice),
                "intercept_index": state.intercept_index,
                "link": _object_state(state.link),
                "template_features": template_features,
                "compiled_features": compiled_features,
                "compiled_interactions": compiled_interactions,
                "feature_order": list(compiled.compiled.feature_order),
                "interaction_order": list(compiled.compiled.interaction_order),
                "groups": [_group_manifest(group) for group in state.groups],
                "design": {
                    "n_observations": compiled.compiled.design.n,
                    "n_coefficients": compiled.compiled.design.p,
                    "group_matrix_types": [
                        qualified_type(matrix) for matrix in compiled.compiled.design.group_matrices
                    ],
                },
                "training_offset": {
                    "nonzero": bool(np.any(state.offset != 0.0)),
                    **_array_descriptor(state.offset),
                },
            }
        )
    return result


def _rank_manifest(rank: RankDecomposition) -> dict[str, Any]:
    def optional_array(values: np.ndarray | None) -> dict[str, Any] | None:
        return None if values is None else _array_descriptor(values)

    condition = float(rank.pre_truncation_condition)
    return {
        "policy_version": rank.policy_version,
        "method": rank.method,
        "width": rank.width,
        "rank": rank.rank,
        "pre_truncation_condition": condition if math.isfinite(condition) else None,
        "pre_truncation_condition_infinite": bool(math.isinf(condition)),
        "cutoff": _finite_float(rank.cutoff, name="rank cutoff"),
        "rank_truncated": rank.rank_truncated,
        "used_svd_fallback": rank.used_svd_fallback,
        "resolution_limited": rank.resolution_limited,
        "log_pdet": _finite_float(rank.log_pdet, name="rank log pseudo-determinant"),
        "active_columns": np.asarray(rank.active_columns, dtype=np.int64).tolist(),
        "column_scale": _array_descriptor(rank.column_scale),
        "cholesky_factor": optional_array(rank.cholesky_factor),
        "pivots": optional_array(rank.pivots),
        "solution_basis": optional_array(rank.solution_basis),
        "parameter_null_basis": optional_array(rank.parameter_null_basis),
        "estimable_functional_basis": optional_array(rank.estimable_functional_basis),
        "structural_aliases": optional_array(rank.structural_aliases),
        "retained_values": optional_array(rank.retained_values),
    }


def _coefficient_face_manifest(result: DenseSolverResult) -> dict[str, Any] | None:
    face = result.coefficient_face
    if face is None:
        return None
    return {
        "component_names": list(face.component_names),
        "coefficient_names": list(face.coefficient_names),
        "constraint_rank": face.constraint_rank,
        "constraint_matrix": _array_descriptor(face.constraint_matrix),
        "null_basis": _array_descriptor(face.null_basis),
        "constraint_basis": _array_descriptor(face.constraint_basis),
        "rank_resolution": _finite_float(face.rank_resolution, name="face rank resolution"),
        "null_residual_bound": _finite_float(
            face.null_residual_bound,
            name="face null residual bound",
        ),
        "orthogonality_bound": _finite_float(
            face.orthogonality_bound,
            name="face orthogonality bound",
        ),
    }


def _solver_manifest(result: DenseSolverResult) -> dict[str, Any]:
    manifest = {
        "config": _dataclass_config(result.config),
        "family_likelihood_plan_identifier": (result.family_likelihood_plan_identifier),
        "resolved_chunk_size": result.resolved_chunk_size,
        "execution_backend_identifier": result.execution_backend_identifier,
        "converged": result.converged,
        "convergence_reason": result.convergence_reason,
        "iterations": result.iterations,
        "backtracking_steps": result.backtracking_steps,
        "optimizing_log_likelihood": result.optimizing_log_likelihood,
        "parameter_independent_carrier": result.parameter_independent_carrier,
        "log_likelihood": result.log_likelihood,
        "penalty_value": result.penalty_value,
        "penalized_optimizing_log_likelihood": result.penalized_optimizing_log_likelihood,
        "penalized_log_likelihood": result.penalized_log_likelihood,
        "initial_penalized_optimizing_log_likelihood": (
            result.initial_penalized_optimizing_log_likelihood
        ),
        "initial_penalized_log_likelihood": result.initial_penalized_log_likelihood,
        "score_relative": result.score_relative,
        "objective_relative_change": result.objective_relative_change,
        "step_relative": result.step_relative,
        "rank": _rank_manifest(result.terminal_rank),
        "curvature_telemetry": _telemetry_manifest(result.terminal_curvature),
        "history": [_dataclass_config(item) for item in result.history],
    }
    face = _coefficient_face_manifest(result)
    if face is not None:
        reduced_rank = result.terminal_reduced_rank
        if reduced_rank is None:
            raise DistributionalSerializationError(
                "an exact coefficient face requires reduced-rank provenance"
            )
        manifest["coefficient_face"] = face
        manifest["terminal_reduced_rank"] = _rank_manifest(reduced_rank)
    return manifest


def _efs_iteration_manifest(item: DistributionalEFSIteration) -> dict[str, Any]:
    result = _dataclass_config(item)
    if not item.activated_face_components:
        result.pop("activated_face_components")
    if not item.deactivated_face_components:
        result.pop("deactivated_face_components")
    if not item.revalidated_face_components:
        result.pop("revalidated_face_components")
    if not item.refused_face_components:
        result.pop("refused_face_components")
    if item.endpoint_direction_evidence is None:
        result.pop("endpoint_direction_evidence")
    if item.endpoint_assessment_failure_reason is None:
        result.pop("endpoint_assessment_failure_reason")
    if item.joint_rollback_penalty_fingerprint is None:
        result.pop("joint_rollback_penalty_fingerprint")
    result["update_curvature"] = _telemetry_manifest(item.update_curvature)
    result["accepted_curvature"] = (
        None if item.accepted_curvature is None else _telemetry_manifest(item.accepted_curvature)
    )
    return result


def _smoothing_manifest(smoothing: DistributionalEFSResult | None) -> dict[str, Any] | None:
    if smoothing is None:
        return None
    manifest = {
        "config": _dataclass_config(smoothing.config),
        "initial_lambdas": dict(smoothing.initial_lambdas),
        "lambdas": dict(smoothing.lambdas),
        "initial_objective": smoothing.initial_objective,
        "objective": smoothing.objective,
        "converged": smoothing.converged,
        "convergence_reason": smoothing.convergence_reason,
        "terminal_raw_max_log_step": smoothing.terminal_raw_max_log_step,
        "unresolved_upper_bound": list(smoothing.unresolved_upper_bound),
        "iterations": smoothing.iterations,
        "terminal_fit_index": smoothing.terminal_fit_index,
        "fallback_count": smoothing.fallback_count,
        "accelerated_trial_count": smoothing.accelerated_trial_count,
        "accelerated_accept_count": smoothing.accelerated_accept_count,
        "raw_fallback_count": smoothing.raw_fallback_count,
        "matched_certified": smoothing.matched_certified,
        "history": [_efs_iteration_manifest(item) for item in smoothing.history],
        "coefficient_fits": [_solver_manifest(item) for item in smoothing.coefficient_fits],
    }
    if smoothing.terminal_endpoint_directions:
        manifest["terminal_endpoint_directions"] = {
            name: _dataclass_config(evidence)
            for name, evidence in smoothing.terminal_endpoint_directions.items()
        }
    # Optional terminal diagnostics enter the manifest only when populated.
    # Same-major readers restore absent optional fields from current defaults.
    if smoothing.terminal_gradient is not None:
        assert smoothing.terminal_gradient_certificate is not None
        manifest["terminal_gradient"] = dict(smoothing.terminal_gradient)
        manifest["terminal_gradient_certificate"] = dict(smoothing.terminal_gradient_certificate)
    if smoothing.terminal_projected_gradient_norm is not None:
        manifest["terminal_projected_gradient_norm"] = smoothing.terminal_projected_gradient_norm
    if smoothing.smoothing_hessian is not None:
        manifest["smoothing_hessian"] = _metadata_value(smoothing.smoothing_hessian)
        manifest["smoothing_hessian_certificate"] = _metadata_value(
            smoothing.smoothing_hessian_certificate
        )
    if smoothing.newton_iterations:
        manifest["newton_iterations"] = smoothing.newton_iterations
    if smoothing.bfgs_fallback_iterations:
        manifest["bfgs_fallback_iterations"] = smoothing.bfgs_fallback_iterations
    if smoothing.beyond_cap_components:
        manifest["beyond_cap_components"] = list(smoothing.beyond_cap_components)
    return manifest


def _null_manifest(null: CompactNullModel) -> dict[str, Any]:
    return {
        "family_config": _metadata_value(null.family_config),
        "parameter_names": list(null.parameter_names),
        "link_types": dict(null.link_types),
        "offset_semantics": dict(null.offset_semantics),
        "weight_semantics": null.weight_semantics,
        "weight_contract_schema": null.weight_contract.schema_version,
        "weight_root_digest": null.weight_provenance.root_digest,
        "family_likelihood_plan_identifier": (null.family_likelihood_plan_identifier),
        "n_observations": null.n_observations,
        "weight_sum": null.weight_sum,
        "coefficients": null.coefficients.tolist(),
        "objective": null.objective,
        "log_likelihood": null.log_likelihood,
        "converged": null.converged,
        "convergence_reason": null.convergence_reason,
        "rank": null.rank,
        "curvature_telemetry": _telemetry_manifest(null.curvature_telemetry),
    }


def _retained_rows_manifest(rows: DistributionalRowState | None) -> dict[str, Any]:
    if rows is None:
        return {"retained": False}
    return {
        "retained": True,
        "n_observations": len(rows.response),
        "response": _array_descriptor(rows.response),
        "likelihood_weight_values": _array_descriptor(rows.likelihood_weights.values),
        "weight_root_digest": rows.likelihood_weights.provenance.root_digest,
        "weight_carrier_digest": rows.likelihood_weights.digest,
        "offsets": {name: _array_descriptor(value) for name, value in rows.offsets.items()},
        "fitted_eta": _array_descriptor(rows.fitted_eta),
        "fitted_parameters": _array_descriptor(rows.fitted_parameters),
        "null_eta": _array_descriptor(rows.null_eta),
        "null_parameters": _array_descriptor(rows.null_parameters),
    }


def _weight_contract_manifest(contract: WeightContract) -> dict[str, Any]:
    if not isinstance(contract, WeightContract):
        raise DistributionalSerializationError("weight contract is missing")
    return {
        "schema_version": contract.schema_version,
        "semantics": contract.semantics,
        "geometry_rule": contract.geometry_rule,
        "zero_row_rule": contract.zero_row_rule,
    }


def _weight_provenance_manifest(provenance: WeightProvenance) -> dict[str, Any]:
    if not isinstance(provenance, WeightProvenance):
        raise DistributionalSerializationError("weight provenance is missing")
    return {
        "original_count": provenance.original_count,
        "retained_count": provenance.retained_count,
        "dropped_count": provenance.dropped_count,
        "physical_count": provenance.physical_count,
        "likelihood_count": provenance.likelihood_count,
        "weight_sum": provenance.weight_sum,
        "log_weight_sum": provenance.log_weight_sum,
        "min_weight": provenance.min_weight,
        "max_weight": provenance.max_weight,
        "all_unit": provenance.all_unit,
        "root_digest": provenance.root_digest,
        "dropped_positions_digest": provenance.dropped_positions_digest,
    }


def _validated_likelihood_state(model: DenseDistributionalModel) -> None:
    state = model.fit_state
    contract = state.weight_contract
    provenance = state.weight_provenance
    plan_identifier = state.family_likelihood_plan_identifier
    if not isinstance(contract, WeightContract) or not isinstance(
        provenance,
        WeightProvenance,
    ):
        raise DistributionalSerializationError(
            "fitted likelihood contract or provenance is missing"
        )
    if provenance.contract != contract:
        raise DistributionalSerializationError(
            "weight provenance does not match the fitted contract"
        )
    if not isinstance(plan_identifier, str) or not plan_identifier:
        raise DistributionalSerializationError("fitted root likelihood plan is missing")
    if state.solver_result.family_likelihood_plan_identifier != plan_identifier:
        raise DistributionalSerializationError(
            "solver result does not match the fitted root likelihood plan"
        )
    fits = (state.solver_result,) if state.smoothing is None else state.smoothing.coefficient_fits
    try:
        for fit in fits:
            validate_solver_likelihood_decomposition(fit)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DistributionalSerializationError(
            "solver likelihood decomposition failed live validation"
        ) from exc
    expected_backend = (
        DENSE_EXECUTION_BACKEND_IDENTIFIER
        if state.solver_result.resolved_chunk_size is None
        else CHUNKED_EXECUTION_BACKEND_IDENTIFIER
    )
    if state.solver_result.execution_backend_identifier != expected_backend:
        raise DistributionalSerializationError(
            "solver execution backend does not match its resolved chunk route"
        )
    if state.smoothing is not None and any(
        fit.family_likelihood_plan_identifier != plan_identifier
        for fit in state.smoothing.coefficient_fits
    ):
        raise DistributionalSerializationError(
            "EFS coefficient result does not match the fitted root likelihood plan"
        )
    null = state.null_model
    if (
        null.weight_contract != contract
        or null.weight_provenance != provenance
        or null.family_likelihood_plan_identifier != plan_identifier
    ):
        raise DistributionalSerializationError(
            "null-model likelihood metadata does not match fitted state"
        )
    rows = state.retained_rows
    if rows is None:
        return
    weights = rows.likelihood_weights
    if not isinstance(weights, ResolvedLikelihoodWeights) or (
        weights.provenance != provenance or weights.digest != provenance.root_digest
    ):
        raise DistributionalSerializationError(
            "retained-row likelihood carrier does not match fitted state"
        )
    try:
        reconstructed_plan = model.family.bind_likelihood(
            rows.response,
            weights,
            COMPLETE_OBSERVATION,
        )
    except (TypeError, ValueError, UnsupportedLikelihoodContractError) as exc:
        raise DistributionalSerializationError(
            "retained-row family likelihood plan failed live response reconstruction"
        ) from exc
    if reconstructed_plan.plan_identifier != plan_identifier:
        raise DistributionalSerializationError(
            "retained-row family likelihood plan does not identify the live response"
        )


def distributional_manifest(model: DenseDistributionalModel) -> dict[str, Any]:
    """Return the deterministic, inspectable binding manifest for ``model``."""
    if not isinstance(model, DenseDistributionalModel):
        raise TypeError("model must be a DenseDistributionalModel")
    saved_family = _saved_family_for_class(model.family)
    _validated_likelihood_state(model)
    family = _family_manifest(model.family, saved_family)
    parameters = tuple(model.family.parameters)
    if len(parameters) != len(model.layout.predictors):
        raise DistributionalSerializationError("family and layout parameter counts disagree")
    fit = model.fitted_result
    penalties = [
        _penalty_manifest(component, model.lambdas[component.name])
        for component in model.layout.penalties
    ]
    return {
        "weights": {
            "contract": _weight_contract_manifest(model.fit_state.weight_contract),
            "provenance": _weight_provenance_manifest(model.fit_state.weight_provenance),
            "family_likelihood_plan_identifier": (
                model.fit_state.family_likelihood_plan_identifier
            ),
        },
        "execution": {
            "requested": {
                "discrete": model.fit_state.requested_discrete,
                "n_bins": _metadata_value(model.fit_state.requested_n_bins),
                "chunk_size": model.fit_state.requested_chunk_size,
            },
            "accepted": {
                "resolved_chunk_size": model.result.resolved_chunk_size,
                "backend_identifier": model.result.execution_backend_identifier,
            },
        },
        "family": family,
        "parameter_order": list(model.parameter_names),
        "parameters": [
            _parameter_manifest(parameter, state.link)
            for parameter, state in zip(parameters, model.layout.predictors, strict=True)
        ],
        "predictors": _predictors_manifest(model),
        "layout": {
            "n_coefficients": model.layout.n_coefficients,
            "coefficient_names": list(model.layout.coefficient_names),
            "term_slices": {
                name: _slice_manifest(value) for name, value in model.layout.term_slices.items()
            },
        },
        "penalties": {
            "components": penalties,
            "lambdas": dict(model.lambdas),
        },
        "fit": {
            "revision": model.fit_state.revision,
            "coefficients": fit.coefficients.tolist(),
            "predictor_coefficients": {
                name: values.tolist() for name, values in fit.predictor_coefficients.items()
            },
            "smoothing_parameters": dict(fit.smoothing_parameters),
            **(
                {"exact_face_components": list(fit.exact_face_components)}
                if fit.exact_face_components
                else {}
            ),
            "covariance": fit.covariance.tolist(),
            "total_effective_df": fit.total_effective_df,
            "predictor_edf": dict(fit.predictor_edf),
            "intercept_edf": dict(fit.intercept_edf),
            "term_edf": dict(fit.term_edf),
            "log_likelihood": fit.log_likelihood,
            "penalized_log_likelihood": fit.penalized_log_likelihood,
            "null_objective": fit.null_objective,
            "converged": fit.converged,
            "coefficient_converged": fit.coefficient_converged,
            "smoothing_converged": fit.smoothing_converged,
            "n_inner_iter": fit.n_inner_iter,
            "n_smoothing_iter": fit.n_smoothing_iter,
            "rank": fit.rank,
            "curvature_telemetry": _telemetry_manifest(fit.curvature_telemetry),
        },
        "solver": _solver_manifest(model.result),
        "smoothing": _smoothing_manifest(model.smoothing),
        "inference": {
            "curvature_source": model.inference.curvature_source,
            "rank": model.inference.rank,
            "total_edf": model.inference.total_edf,
            "coefficient_edf": model.inference.coefficient_edf.tolist(),
            "predictor_edf": dict(model.inference.predictor_edf),
            "intercept_edf": dict(model.inference.intercept_edf),
            "term_edf": dict(model.inference.term_edf),
            "reconciliation_tolerance": model.inference.reconciliation_tolerance,
            "slice_reconciliation_error": model.inference.slice_reconciliation_error,
            "predictor_reconciliation_error": model.inference.predictor_reconciliation_error,
        },
        "null_model": _null_manifest(model.null_model),
        "retained_rows": _retained_rows_manifest(model.fit_state.retained_rows),
    }


def _restore_mapping_proxy(values: Mapping[Any, Any]) -> MappingProxyType:
    return MappingProxyType(dict(values))


def _reduce_mapping_proxy(value: Mapping[Any, Any]) -> tuple[Any, tuple[dict[Any, Any]]]:
    return _restore_mapping_proxy, (dict(value),)


class _DistributionalPickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[_MAPPING_PROXY_TYPE] = _reduce_mapping_proxy


def _pickle_model(model: DenseDistributionalModel) -> bytes:
    stream = io.BytesIO()
    try:
        _DistributionalPickler(stream, protocol=5).dump(model)
    except Exception as exc:
        raise DistributionalSerializationError("could not encode fitted model payload") from exc
    return stream.getvalue()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise DistributionalSerializationError(
            "artifact metadata must be finite and JSON serializable"
        ) from exc


def serialize_distributional_model(model: DenseDistributionalModel) -> bytes:
    """Serialize one complete fitted revision into a canonical JSON envelope."""
    manifest = distributional_manifest(model)
    payload = _pickle_model(model)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "manifest": manifest,
        "payload": {
            "encoding": _PAYLOAD_ENCODING,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "data": base64.b64encode(payload).decode("ascii"),
        },
    }
    return _canonical_json(artifact)


def _reject_json_constant(value: str) -> None:
    raise DistributionalSerializationError(f"non-finite JSON constant {value!r} is forbidden")


def _parse_artifact(serialized: bytes | bytearray | memoryview | str) -> Mapping[str, Any]:
    if isinstance(serialized, str):
        text = serialized
    elif isinstance(serialized, bytes | bytearray | memoryview):
        try:
            text = bytes(serialized).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DistributionalSerializationError("artifact must be UTF-8 JSON") from exc
    else:
        raise TypeError("serialized artifact must be bytes or str")
    try:
        artifact = json.loads(text, parse_constant=_reject_json_constant)
    except DistributionalSerializationError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise DistributionalSerializationError("artifact is not valid JSON") from exc
    if not isinstance(artifact, Mapping):
        raise DistributionalSerializationError("artifact root must be a JSON object")
    return artifact


def _schema_version_text(version: Any) -> str:
    if not isinstance(version, str):
        raise DistributionalSerializationError("schema_version must be a semantic version")
    parts = version.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise DistributionalSerializationError("schema_version must be a semantic version")
    return version


def _schema_major(version: Any) -> int:
    return int(_schema_version_text(version).split(".", 1)[0])


def _required_mapping(parent: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        raise DistributionalSerializationError(f"artifact {name!r} must be an object")
    return value


def _validate_gaussian_config(config: Mapping[str, Any]) -> None:
    if set(config) != {"type", "scale_floor"}:
        if "scale_floor" not in config:
            raise DistributionalSerializationError(
                "GaussianLS family configuration must include scale_floor"
            )
        raise DistributionalSerializationError("unsupported or incomplete family configuration")
    floor = config["scale_floor"]
    if isinstance(floor, bool) or not isinstance(floor, int | float):
        raise DistributionalSerializationError(
            "GaussianLS scale_floor must be a finite non-negative number"
        )
    try:
        finite_floor = float(floor)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DistributionalSerializationError(
            "GaussianLS scale_floor must be a finite non-negative number"
        ) from exc
    if not math.isfinite(finite_floor) or finite_floor < 0.0:
        raise DistributionalSerializationError(
            "GaussianLS scale_floor must be a finite non-negative number"
        )


def _validate_gamma_config(config: Mapping[str, Any]) -> None:
    if set(config) != {"type", "parameterization"} or config.get("parameterization") != "mean_cv":
        raise DistributionalSerializationError("unsupported or incomplete family configuration")


def _validate_negative_binomial_config(config: Mapping[str, Any]) -> None:
    if (
        set(config) != {"type", "parameterization"}
        or config.get("parameterization") != "nb2_mean_theta"
    ):
        raise DistributionalSerializationError(
            "NegativeBinomialLS family configuration must contain exactly "
            "type='NegativeBinomialLS' and parameterization='nb2_mean_theta'"
        )


def _validate_tweedie_config(config: Mapping[str, Any]) -> None:
    if set(config) != {"type", "power_lower", "power_upper"}:
        raise DistributionalSerializationError(
            "TweedieLSS family configuration must contain exactly its two power walls"
        )
    lower = config["power_lower"]
    upper = config["power_upper"]
    if type(lower) not in (int, float) or type(upper) not in (int, float):
        raise DistributionalSerializationError(
            "TweedieLSS power walls must be finite JSON numbers with 1 < lower < upper < 2"
        )
    try:
        lower_float = float(lower)
        upper_float = float(upper)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DistributionalSerializationError(
            "TweedieLSS power walls must be finite JSON numbers with 1 < lower < upper < 2"
        ) from exc
    if not (
        math.isfinite(lower_float)
        and math.isfinite(upper_float)
        and 1.0 < lower_float < upper_float < 2.0
    ):
        raise DistributionalSerializationError(
            "TweedieLSS power walls must be finite JSON numbers with 1 < lower < upper < 2"
        )


def _validate_generalized_gamma_config(config: Mapping[str, Any]) -> None:
    if set(config) != {"type", "parametrisation", "scale_floor"}:
        raise DistributionalSerializationError(
            "GeneralizedGammaLSS family configuration must contain exactly type, "
            "parametrisation and scale_floor"
        )
    if config["parametrisation"] not in ("mean", "location"):
        raise DistributionalSerializationError(
            "GeneralizedGammaLSS parametrisation must be 'mean' or 'location'"
        )
    floor = config["scale_floor"]
    if isinstance(floor, bool) or not isinstance(floor, int | float):
        raise DistributionalSerializationError(
            "GeneralizedGammaLSS scale_floor must be a finite non-negative number"
        )
    try:
        # a JSON integer has no width limit, so float() is the raising step here
        finite_floor = float(floor)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DistributionalSerializationError(
            "GeneralizedGammaLSS scale_floor must be a finite non-negative number"
        ) from exc
    if not math.isfinite(finite_floor) or finite_floor < 0.0:
        raise DistributionalSerializationError(
            "GeneralizedGammaLSS scale_floor must be a finite non-negative number"
        )


def _validate_log_normal_config(config: Mapping[str, Any]) -> None:
    if set(config) != {"type", "parametrisation", "scale_floor"}:
        raise DistributionalSerializationError(
            "LogNormalLS family configuration must contain exactly type, "
            "parametrisation and scale_floor"
        )
    if config["parametrisation"] not in ("mean", "location"):
        raise DistributionalSerializationError(
            "LogNormalLS parametrisation must be 'mean' or 'location'"
        )
    floor = config["scale_floor"]
    if isinstance(floor, bool) or not isinstance(floor, int | float):
        raise DistributionalSerializationError(
            "LogNormalLS scale_floor must be a finite non-negative number"
        )
    try:
        # a JSON integer has no width limit, so float() is the raising step here
        finite_floor = float(floor)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DistributionalSerializationError(
            "LogNormalLS scale_floor must be a finite non-negative number"
        ) from exc
    if not math.isfinite(finite_floor) or finite_floor < 0.0:
        raise DistributionalSerializationError(
            "LogNormalLS scale_floor must be a finite non-negative number"
        )


def _config_number_is_finite(value: object) -> bool:
    """Whether a JSON number converts to a finite float (an oversized integer does not)."""
    try:
        return math.isfinite(
            float(value)  # ty: ignore[invalid-argument-type] -- validated JSON boundary
        )
    except (OverflowError, TypeError, ValueError):
        return False


def _validate_generalized_pareto_config(config: Mapping[str, Any]) -> None:
    if set(config) != {"type", "shape_lower", "shape_upper"}:
        raise DistributionalSerializationError(
            "GeneralizedParetoLSS family configuration must contain exactly type, shape_lower "
            "and shape_upper"
        )
    walls = []
    for key in ("shape_lower", "shape_upper"):
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not _config_number_is_finite(value)
        ):
            raise DistributionalSerializationError(
                f"GeneralizedParetoLSS {key} must be a finite real number"
            )
        walls.append(float(value))
    lower, upper = walls
    if not 0.0 <= lower < upper <= 1.0:
        raise DistributionalSerializationError(
            "GeneralizedParetoLSS shape walls must satisfy 0 <= shape_lower < shape_upper <= 1"
        )


def _validate_two_piece_skew_bound(value: object, family: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not _config_number_is_finite(value)
        or not 0.0 < float(value) < 1.0
    ):
        raise DistributionalSerializationError(
            f"{family} skew_bound must be a finite number strictly inside (0, 1)"
        )


def _validate_two_piece_scale_floor(value: object, family: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not _config_number_is_finite(value)
        or float(value) < 0.0
    ):
        raise DistributionalSerializationError(
            f"{family} scale_floor must be a finite non-negative number"
        )


def _validate_two_piece_log_normal_config(config: Mapping[str, Any]) -> None:
    if set(config) != {"type", "parametrisation", "scale_floor", "skew_bound"}:
        raise DistributionalSerializationError(
            "TwoPieceLogNormalLSS family configuration must contain exactly type, "
            "parametrisation, scale_floor and skew_bound"
        )
    if config["parametrisation"] not in ("mean", "location"):
        raise DistributionalSerializationError(
            "TwoPieceLogNormalLSS parametrisation must be 'mean' or 'location'"
        )
    _validate_two_piece_scale_floor(config["scale_floor"], "TwoPieceLogNormalLSS")
    _validate_two_piece_skew_bound(config["skew_bound"], "TwoPieceLogNormalLSS")


def _validate_two_piece_normal_config(config: Mapping[str, Any]) -> None:
    if set(config) != {"type", "scale_floor", "skew_bound"}:
        raise DistributionalSerializationError(
            "TwoPieceNormalLSS family configuration must contain exactly type, "
            "scale_floor and skew_bound"
        )
    _validate_two_piece_scale_floor(config["scale_floor"], "TwoPieceNormalLSS")
    _validate_two_piece_skew_bound(config["skew_bound"], "TwoPieceNormalLSS")


@dataclasses.dataclass(frozen=True, slots=True)
class _SavedFamily:
    family_class: type[object]
    python_type: str
    config_type: str
    validate_config: Callable[[Mapping[str, Any]], None]


_SAVED_FAMILIES = (
    _SavedFamily(
        GaussianLS,
        "superglm.distributional.families.gaussian.GaussianLS",
        "GaussianLS",
        _validate_gaussian_config,
    ),
    _SavedFamily(
        GammaLS,
        "superglm.distributional.families.gamma.GammaLS",
        "GammaLS",
        _validate_gamma_config,
    ),
    _SavedFamily(
        NegativeBinomialLS,
        "superglm.distributional.families.negative_binomial.NegativeBinomialLS",
        "NegativeBinomialLS",
        _validate_negative_binomial_config,
    ),
    _SavedFamily(
        TweedieLSS,
        "superglm.distributional.families.tweedie.TweedieLSS",
        "TweedieLSS",
        _validate_tweedie_config,
    ),
    _SavedFamily(
        GeneralizedGammaLSS,
        "superglm.distributional.families.generalized_gamma.GeneralizedGammaLSS",
        "GeneralizedGammaLSS",
        _validate_generalized_gamma_config,
    ),
    _SavedFamily(
        GeneralizedParetoLSS,
        "superglm.distributional.families.generalized_pareto.GeneralizedParetoLSS",
        "GeneralizedParetoLSS",
        _validate_generalized_pareto_config,
    ),
    _SavedFamily(
        TwoPieceLogNormalLSS,
        "superglm.distributional.families.two_piece.TwoPieceLogNormalLSS",
        "TwoPieceLogNormalLSS",
        _validate_two_piece_log_normal_config,
    ),
    _SavedFamily(
        TwoPieceNormalLSS,
        "superglm.distributional.families.two_piece.TwoPieceNormalLSS",
        "TwoPieceNormalLSS",
        _validate_two_piece_normal_config,
    ),
    _SavedFamily(
        LogNormalLS,
        "superglm.distributional.families.log_normal.LogNormalLS",
        "LogNormalLS",
        _validate_log_normal_config,
    ),
)


def _saved_family_for_class(family: object) -> _SavedFamily:
    for saved_family in _SAVED_FAMILIES:
        if type(family) is saved_family.family_class:
            return saved_family
    raise DistributionalSerializationError("only built-in families can be serialized")


def _saved_family_for_manifest_pair(python_type: Any, config_type: Any) -> _SavedFamily:
    for saved_family in _SAVED_FAMILIES:
        if python_type == saved_family.python_type and config_type == saved_family.config_type:
            return saved_family
    for saved_family in reversed(_SAVED_FAMILIES[2:]):
        if python_type == saved_family.python_type or config_type == saved_family.config_type:
            raise DistributionalSerializationError(
                f"{saved_family.config_type} qualified and configuration types "
                "must be an exact pair"
            )
    raise DistributionalSerializationError("unsupported or incomplete family configuration")


def _validate_family_manifest(manifest: Mapping[str, Any]) -> _SavedFamily:
    family = _required_mapping(manifest, "family")
    config = _required_mapping(family, "config")
    saved_family = _saved_family_for_manifest_pair(
        family.get("python_type"),
        config.get("type"),
    )
    saved_family.validate_config(config)
    return saved_family


def _decode_payload(artifact: Mapping[str, Any]) -> bytes:
    payload = _required_mapping(artifact, "payload")
    if payload.get("encoding") != _PAYLOAD_ENCODING:
        raise DistributionalSerializationError("unsupported fitted-state payload encoding")
    expected_digest = payload.get("sha256")
    encoded = payload.get("data")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise DistributionalSerializationError("payload digest must be a SHA-256 hexadecimal value")
    if not isinstance(encoded, str):
        raise DistributionalSerializationError("payload data must be base64 text")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise DistributionalSerializationError("payload data is not valid base64") from exc
    actual_digest = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest.lower()):
        raise DistributionalSerializationError("payload digest does not match payload data")
    return raw


def _build_from_state[T](state: Mapping[str, Any], cls: type[T]) -> T:
    """Construct ``cls`` from a pickled state, filling fields that build lacked."""
    fields = [
        field
        for field in dataclasses.fields(
            cls  # ty: ignore[invalid-argument-type] -- internal dataclass type
        )
        if field.init
    ]
    kwargs: dict[str, Any] = {}
    for field in fields:
        if field.name in state:
            kwargs[field.name] = state[field.name]
        elif field.default is not dataclasses.MISSING:
            kwargs[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            kwargs[field.name] = field.default_factory()
        else:
            raise DistributionalSerializationError(
                f"pickled {cls.__name__} lacks its required field {field.name!r}"
            )
    return cls(**kwargs)


def _with_absent_fields[T](value: T, cls: type[T]) -> T:
    """Rebuild a pickled frozen dataclass whose build predates some of its fields.

    A field the pickling build did not know takes its current schema-major
    default; a complete state is returned as is.
    """
    state = vars(value)
    if all(
        field.name in state
        for field in dataclasses.fields(
            cls  # ty: ignore[invalid-argument-type] -- internal dataclass type
        )
        if field.init
    ):
        return value
    return _build_from_state(state, cls)


def _migrate_unpickled_fit_state(model: DenseDistributionalModel):
    """Restore optional defaults absent from same-major pickle state."""
    smoothing = model.smoothing
    if smoothing is None:
        return model.fit_state
    state = dict(vars(smoothing))
    config = _with_absent_fields(state["config"], DistributionalEFSConfig)
    history = tuple(
        _with_absent_fields(item, DistributionalEFSIteration) for item in state["history"]
    )
    changed = config is not state["config"] or any(
        rebuilt is not item for rebuilt, item in zip(history, state["history"], strict=True)
    )
    terminal_fit = state["coefficient_fits"][state["terminal_fit_index"]]
    if "terminal_endpoint_directions" not in state and terminal_fit.coefficient_face is None:
        state["terminal_endpoint_directions"] = {}
        changed = True
    result_fields = {
        field.name for field in dataclasses.fields(DistributionalEFSResult) if field.init
    }
    if not changed and result_fields <= set(state):
        return model.fit_state
    state["config"] = config
    state["history"] = history
    migrated_smoothing = _build_from_state(state, DistributionalEFSResult)
    return dataclasses.replace(model.fit_state, smoothing=migrated_smoothing)


_INVARIANT_DATACLASS_TYPES = (
    WeightContract,
    CurvatureTelemetry,
    DenseSolverConfig,
    PenaltyFace,
    EndpointDirectionEvidence,
    JointEndpointDirectionEvidence,
    DenseSolverResult,
    DistributionalEFSConfig,
    DistributionalEFSIteration,
    JointInference,
    CompactNullModel,
    DistributionalFitResult,
    DistributionalRowState,
)


def _reconstruct_invariant_dataclass(value: Any, memo: dict[int, Any]) -> Any:
    """Re-run trusted dataclass constructors while preserving shared references."""

    if isinstance(value, _INVARIANT_DATACLASS_TYPES):
        cached = memo.get(id(value))
        if cached is not None:
            return cached
        changes = {
            field.name: _reconstruct_invariant_dataclass(getattr(value, field.name), memo)
            for field in dataclasses.fields(value)
            if field.init
        }
        rebuilt = dataclasses.replace(value, **changes)
        memo[id(value)] = rebuilt
        return rebuilt
    if isinstance(value, tuple):
        rebuilt = tuple(_reconstruct_invariant_dataclass(item, memo) for item in value)
        return (
            value
            if all(left is right for left, right in zip(value, rebuilt, strict=True))
            else rebuilt
        )
    if isinstance(value, Mapping):
        rebuilt = {key: _reconstruct_invariant_dataclass(item, memo) for key, item in value.items()}
        if all(rebuilt[key] is item for key, item in value.items()):
            return value
        return rebuilt
    return value


def _revalidate_unpickled_fit_state(fit_state: DistributionalFitState) -> DistributionalFitState:
    """Reconstruct the serialized authority graph in dependency order."""

    if not isinstance(fit_state, DistributionalFitState):
        raise TypeError("fit state must be DistributionalFitState")
    memo: dict[int, Any] = {}
    smoothing = fit_state.smoothing
    if smoothing is None:
        rebuilt_smoothing = None
    else:
        if not isinstance(smoothing, DistributionalEFSResult):
            raise TypeError("smoothing must be DistributionalEFSResult")
        coefficient_fits = tuple(
            _reconstruct_invariant_dataclass(item, memo) for item in smoothing.coefficient_fits
        )
        history = tuple(_reconstruct_invariant_dataclass(item, memo) for item in smoothing.history)
        endpoint_directions = {
            name: _reconstruct_invariant_dataclass(item, memo)
            for name, item in smoothing.terminal_endpoint_directions.items()
        }
        rebuilt_smoothing = dataclasses.replace(
            smoothing,
            config=_reconstruct_invariant_dataclass(smoothing.config, memo),
            history=history,
            coefficient_fits=coefficient_fits,
            terminal_endpoint_directions=endpoint_directions,
        )
        memo[id(smoothing)] = rebuilt_smoothing

    rebuilt_solver_result = _reconstruct_invariant_dataclass(fit_state.solver_result, memo)
    return dataclasses.replace(
        fit_state,
        weight_contract=_reconstruct_invariant_dataclass(fit_state.weight_contract, memo),
        solver_result=rebuilt_solver_result,
        smoothing=rebuilt_smoothing,
        inference=_reconstruct_invariant_dataclass(fit_state.inference, memo),
        null_model=_reconstruct_invariant_dataclass(fit_state.null_model, memo),
        result=_reconstruct_invariant_dataclass(fit_state.result, memo),
        retained_rows=_reconstruct_invariant_dataclass(fit_state.retained_rows, memo),
    )


def deserialize_distributional_model(
    serialized: bytes | bytearray | memoryview | str,
) -> DenseDistributionalModel:
    """Load a trusted native artifact after schema, digest, and manifest checks.

    Refuses known pre-contract majors with the typed legacy-weight error and
    unsupported majors with an ordinary version error. Per the module docstring,
    a manifest mismatch raised past that gate always means
    the artifact was tampered with or corrupted — never that it is merely old.
    """
    artifact = _parse_artifact(serialized)
    artifact_version = _schema_version_text(artifact.get("schema_version"))
    if artifact.get("artifact_type") != _ARTIFACT_TYPE:
        raise DistributionalSerializationError("artifact type is not a distributional model")
    manifest = _required_mapping(artifact, "manifest")
    artifact_major = _schema_major(artifact_version)
    current_major = _schema_major(SCHEMA_VERSION)
    if artifact_major in {1, 2}:
        _validate_family_manifest(manifest)
        _decode_payload(artifact)
        raise LegacyPowerWeightArtifactError(
            f"legacy pre-contract distributional artifact schema version "
            f"{artifact_version} cannot establish likelihood-weight semantics; "
            "refit and regenerate the artifact"
        )
    if artifact_major != current_major and artifact_major not in READABLE_PREVIOUS_MAJORS:
        raise DistributionalSerializationError(
            f"artifact schema version {artifact_version} is unreadable by this build, "
            f"which reads and writes schema version {SCHEMA_VERSION}; the major "
            "versions differ, so the artifact must be regenerated"
        )
    saved_family = _validate_family_manifest(manifest)
    payload = _decode_payload(artifact)
    try:
        restored = pickle.loads(payload)
    except Exception as exc:
        raise DistributionalSerializationError("could not decode fitted model payload") from exc
    if not isinstance(restored, DenseDistributionalModel):
        raise DistributionalSerializationError("payload is not a dense distributional model")
    if type(restored.family) is not saved_family.family_class:
        raise DistributionalSerializationError(
            "payload family class does not match family manifest"
        )
    try:
        fit_state = _revalidate_unpickled_fit_state(_migrate_unpickled_fit_state(restored))
        validated = DenseDistributionalModel(family=restored.family, _fit_state=fit_state)
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        raise _InvalidRestoredStateError(
            f"manifest does not match a valid fitted model payload: {detail}"
        ) from exc
    restored_manifest = distributional_manifest(validated)
    if not hmac.compare_digest(
        _canonical_json(restored_manifest),
        _canonical_json(manifest),
    ):
        raise DistributionalSerializationError("manifest does not match fitted model payload")
    return validated


__all__ = [
    "SCHEMA_VERSION",
    "DistributionalSerializationError",
    "deserialize_distributional_model",
    "distributional_manifest",
    "serialize_distributional_model",
]
