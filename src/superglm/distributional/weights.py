"""Immutable likelihood-weight contracts for distributional fitting."""

from __future__ import annotations

import hashlib
import operator
from dataclasses import dataclass
from typing import Literal

import numpy as np

WeightSemantics = Literal["prior", "frequency"]
WEIGHT_CONTRACT_SCHEMA = 1
MAX_EXACT_FREQUENCY_COUNT = min(np.iinfo(np.intp).max, 2**53)


class LikelihoodWeightError(Exception):
    """Raised when likelihood weights cannot be represented safely."""


class UnsupportedLikelihoodContractError(LikelihoodWeightError):
    """Raised when a likelihood-weight semantic is not supported."""


class LegacyPowerWeightArtifactError(LikelihoodWeightError):
    """Raised when a legacy power-weight artifact cannot be interpreted."""


@dataclass(frozen=True)
class WeightContract:
    """The semantic interpretation for likelihood weights."""

    semantics: WeightSemantics

    def __post_init__(self) -> None:
        if self.semantics not in {"prior", "frequency"}:
            raise UnsupportedLikelihoodContractError(
                f"unsupported likelihood-weight semantics: {self.semantics!r}"
            )

    @property
    def schema_version(self) -> int:
        return WEIGHT_CONTRACT_SCHEMA

    @property
    def geometry_rule(self) -> str:
        return "physical_rows" if self.semantics == "prior" else "replication_mass"

    @property
    def zero_row_rule(self) -> str:
        return "drop"


@dataclass(frozen=True)
class WeightProvenance:
    """Constant-size provenance for a resolved likelihood-weight root."""

    contract: WeightContract
    original_count: int
    retained_count: int
    dropped_count: int
    physical_count: int
    likelihood_count: int
    weight_sum: float
    log_weight_sum: float
    min_weight: float
    max_weight: float
    all_unit: bool
    root_digest: str
    dropped_positions_digest: str

    @property
    def min(self) -> float:
        """Smallest retained weight."""

        return self.min_weight

    @property
    def max(self) -> float:
        """Largest retained weight."""

        return self.max_weight


@dataclass(frozen=True)
class ResolvedLikelihoodWeights:
    """Read-only retained likelihood weights and their positional provenance."""

    provenance: WeightProvenance
    values: np.ndarray
    geometry_values: np.ndarray
    root_take_map: np.ndarray
    input_positions: np.ndarray
    dropped_input_positions: np.ndarray
    digest: str

    @property
    def root_digest(self) -> str:
        return self.provenance.root_digest

    @property
    def physical_count(self) -> int:
        return self.provenance.physical_count

    @property
    def likelihood_count(self) -> int:
        return self.provenance.likelihood_count

    @property
    def weight_sum(self) -> float:
        return self.provenance.weight_sum

    def take(self, indices: np.ndarray) -> ResolvedLikelihoodWeights:
        """Return a non-overlapping positional child of these retained rows."""

        take_map = _take_map(indices, len(self.values))
        root_take_map = _readonly_array(self.root_take_map[take_map], dtype=np.intp)
        values = _readonly_array(self.values[take_map])
        positions = _readonly_array(self.input_positions[take_map], dtype=np.intp)
        return ResolvedLikelihoodWeights(
            provenance=self.provenance,
            values=values,
            geometry_values=_readonly_array(self.geometry_values[take_map]),
            root_take_map=root_take_map,
            input_positions=positions,
            dropped_input_positions=self.dropped_input_positions,
            digest=_child_digest(self.root_digest, root_take_map, positions, values),
        )


def resolve_likelihood_weights(
    weights: np.ndarray | None,
    *,
    n_observations: int,
    contract: WeightContract,
) -> ResolvedLikelihoodWeights:
    """Validate, retain, and stamp likelihood weights with root provenance."""

    if not isinstance(contract, WeightContract):
        raise UnsupportedLikelihoodContractError("contract must be a WeightContract")
    original_count = _observation_count(n_observations)
    values = _weight_values(weights, original_count, contract)
    retained = values > 0.0
    if not np.any(retained):
        raise LikelihoodWeightError("likelihood weights must retain at least one row")

    retained_values = _readonly_array(values[retained])
    input_positions = _readonly_array(np.flatnonzero(retained), dtype=np.intp)
    dropped_positions = _readonly_array(np.flatnonzero(~retained), dtype=np.intp)
    geometry_values = (
        _readonly_array(np.ones(len(retained_values), dtype=np.float64))
        if contract.semantics == "prior"
        else retained_values
    )
    physical_count = len(retained_values)
    root_take_map = _readonly_array(np.arange(physical_count, dtype=np.intp), dtype=np.intp)
    with np.errstate(over="ignore", invalid="ignore"):
        weight_sum = float(np.sum(retained_values, dtype=np.float64))
    if not np.isfinite(weight_sum):
        raise UnsupportedLikelihoodContractError(
            "resolved likelihood weights require a finite weight_sum"
        )
    likelihood_count = (
        physical_count
        if contract.semantics == "prior"
        else sum(int(value) for value in retained_values)
    )
    root_digest = _root_digest(
        contract,
        original_count,
        input_positions,
        dropped_positions,
        retained_values,
    )
    provenance = WeightProvenance(
        contract=contract,
        original_count=original_count,
        retained_count=physical_count,
        dropped_count=len(dropped_positions),
        physical_count=physical_count,
        likelihood_count=likelihood_count,
        weight_sum=weight_sum,
        log_weight_sum=_log_weight_sum(retained_values),
        min_weight=float(np.min(retained_values)),
        max_weight=float(np.max(retained_values)),
        all_unit=bool(np.all(retained_values == 1.0)),
        root_digest=root_digest,
        dropped_positions_digest=_array_digest(dropped_positions),
    )
    return ResolvedLikelihoodWeights(
        provenance=provenance,
        values=retained_values,
        geometry_values=geometry_values,
        root_take_map=root_take_map,
        input_positions=input_positions,
        dropped_input_positions=dropped_positions,
        digest=root_digest,
    )


def _observation_count(value: int) -> int:
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise LikelihoodWeightError("n_observations must be a non-negative integer") from exc
    if count < 0:
        raise LikelihoodWeightError("n_observations must be a non-negative integer")
    return count


def _log_weight_sum(values: np.ndarray) -> float:
    # Use a distinct view so family-level instrumentation can distinguish this
    # provenance summary from Gaussian carrier preparation without a row copy.
    return float(np.sum(np.log(values.view()), dtype=np.float64))


def _weight_values(
    weights: np.ndarray | None,
    n_observations: int,
    contract: WeightContract,
) -> np.ndarray:
    if weights is None:
        if contract.semantics == "frequency" and n_observations > MAX_EXACT_FREQUENCY_COUNT:
            raise LikelihoodWeightError("frequency likelihood count exceeds the exact-count bound")
        return np.ones(n_observations, dtype=np.float64)
    try:
        source = np.asarray(weights)
    except (TypeError, ValueError) as exc:
        raise LikelihoodWeightError(
            "weights must be a one-dimensional value for each observation"
        ) from exc
    if source.ndim != 1 or len(source) != n_observations:
        raise LikelihoodWeightError("weights must be a one-dimensional value for each observation")
    if contract.semantics == "frequency":
        return np.asarray(_frequency_counts(source), dtype=np.float64)
    return _prior_values(source)


def _prior_values(source: np.ndarray) -> np.ndarray:
    _validate_real_weight_scalars(source)
    try:
        with np.errstate(over="raise", invalid="raise"):
            values = source.astype(np.float64, copy=True)
    except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
        raise LikelihoodWeightError("weights must be finite numeric values") from exc
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise LikelihoodWeightError("weights must be finite and non-negative")
    return values


def _frequency_counts(source: np.ndarray) -> list[int]:
    """Return validated replication counts without first narrowing to float64."""

    _validate_real_weight_scalars(source)
    counts: list[int] = []
    exact_count = 0
    for value in source:
        if isinstance(value, int | np.integer):
            count = int(value)
        else:
            if not np.isfinite(value) or value != np.floor(value):
                raise LikelihoodWeightError("frequency weights must be exact non-negative integers")
            count = int(value)
        if count < 0:
            raise LikelihoodWeightError("weights must be finite and non-negative")
        exact_count += count
        if exact_count > MAX_EXACT_FREQUENCY_COUNT:
            raise LikelihoodWeightError("frequency likelihood count exceeds the exact-count bound")
        counts.append(count)
    return counts


def _validate_real_weight_scalars(source: np.ndarray) -> None:
    """Refuse coercive or complex inputs before numeric conversion."""

    if source.dtype.kind not in {"f", "i", "u", "O"}:
        raise LikelihoodWeightError("weights must be finite real numeric values")
    if source.dtype.kind != "O":
        return
    for value in source:
        if isinstance(value, bool | np.bool_) or not isinstance(value, int | float | np.number):
            raise LikelihoodWeightError("weights must be finite real numeric values")
        if isinstance(value, complex | np.complexfloating):
            raise LikelihoodWeightError("weights must be finite real numeric values")


def _take_map(indices: np.ndarray, length: int) -> np.ndarray:
    source = np.asarray(indices)
    if source.ndim != 1 or not np.issubdtype(source.dtype, np.integer):
        raise LikelihoodWeightError("take indices must be a one-dimensional integer array")
    take_map = source.astype(np.intp, copy=False)
    if len(take_map) == 0:
        raise LikelihoodWeightError("take indices must retain at least one row")
    if np.any(take_map < 0) or np.any(take_map >= length):
        raise LikelihoodWeightError("take indices are out of range")
    if len(np.unique(take_map)) != len(take_map):
        raise LikelihoodWeightError("take indices must not contain duplicates")
    return _readonly_array(take_map, dtype=np.intp)


def _readonly_array(values: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    return immutable


def _update_field(digest: hashlib._Hash, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


def _update_array(digest: hashlib._Hash, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    _update_field(digest, array.dtype.str.encode("ascii"))
    _update_field(digest, repr(array.shape).encode("ascii"))
    _update_field(digest, array.tobytes(order="C"))


def _array_digest(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    _update_array(digest, values)
    return digest.hexdigest()


def _root_digest(
    contract: WeightContract,
    original_count: int,
    input_positions: np.ndarray,
    dropped_positions: np.ndarray,
    values: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    _update_field(digest, b"superglm-likelihood-weights-root")
    _update_field(digest, str(contract.schema_version).encode("ascii"))
    _update_field(digest, contract.semantics.encode("ascii"))
    _update_field(digest, str(original_count).encode("ascii"))
    _update_array(digest, input_positions)
    _update_array(digest, dropped_positions)
    _update_array(digest, values)
    return digest.hexdigest()


def _child_digest(
    root_digest: str,
    root_take_map: np.ndarray,
    input_positions: np.ndarray,
    values: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    _update_field(digest, b"superglm-likelihood-weights-child")
    _update_field(digest, root_digest.encode("ascii"))
    _update_array(digest, root_take_map)
    _update_array(digest, input_positions)
    _update_array(digest, values)
    return digest.hexdigest()


__all__ = [
    "LegacyPowerWeightArtifactError",
    "LikelihoodWeightError",
    "MAX_EXACT_FREQUENCY_COUNT",
    "ResolvedLikelihoodWeights",
    "UnsupportedLikelihoodContractError",
    "WEIGHT_CONTRACT_SCHEMA",
    "WeightContract",
    "WeightProvenance",
    "WeightSemantics",
    "resolve_likelihood_weights",
]
