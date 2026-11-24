"""Shared sandwich-covariance result contract and numerical primitives.

Used by ``SuperGLM`` (S1) and designed for direct reuse by ``SuperLSS`` (S2):
fit binding across round-trips/refits, explicit tagged lifecycle, cluster
alignment and cancellation-safe accumulation, small-sample multipliers,
power-of-two guarded Gram accumulation, and rank-guarded Wald block tests.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from superglm._roundoff import EPS as _EPS
from superglm.types import GroupSlice

SandwichStatus = Literal["available", "unavailable"]
SandwichCovType = Literal["HC0", "HC1", "CR0", "CR1"]
BreadCurvature = Literal["fisher", "observed"]
UnavailableReason = Literal[
    "missing_fit_state",
    "editor_stale",
    "shape_repaired",
    "not_converged",
    "selection_penalty",
    "active_constraints",
    "unsupported_curvature",
    "indefinite_bread",
    "non_finite_scores",
    "workspace_budget_exceeded",
]

# Explicit tagged lifecycle table (AGENTS.md: one declared table, one writer module).
SANDWICH_LIFECYCLE: dict[tuple[str, str], SandwichStatus] = {
    ("init", "build_available"): "available",
    ("init", "build_unavailable"): "unavailable",
}

UNAVAILABLE_REASONS: frozenset[str] = frozenset(
    {
        "missing_fit_state",
        "editor_stale",
        "shape_repaired",
        "not_converged",
        "selection_penalty",
        "active_constraints",
        "unsupported_curvature",
        "indefinite_bread",
        "non_finite_scores",
        "workspace_budget_exceeded",
    }
)


def _freeze(array: NDArray) -> NDArray:
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


def coef_digest(intercept: float, beta: NDArray) -> bytes:
    """Return a deterministic 16-byte BLAKE2b digest of published coefficients."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.array([float(intercept)], dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(beta, dtype="<f8").tobytes())
    return h.digest()


@dataclass(frozen=True)
class FitBinding:
    """Picklable identity tying a covariance result to one exact fitted state."""

    model_kind: Literal["SuperGLM", "SuperLSS"]
    fit_uid: str
    fit_revision: int
    repair_revision: int
    coef_digest: bytes

    def validate_match(self, current: FitBinding) -> None:
        """Raise ``ValueError`` if ``current`` does not identify the same fit."""
        if self.model_kind != current.model_kind:
            raise ValueError(
                f"SandwichCovariance was built for {self.model_kind}, not {current.model_kind}."
            )
        if (
            self.fit_uid != current.fit_uid
            or self.fit_revision != current.fit_revision
            or self.repair_revision != current.repair_revision
            or self.coef_digest != current.coef_digest
        ):
            raise ValueError(
                "SandwichCovariance is bound to a different fit or revision and cannot be "
                "reused on this model; recompute sandwich_covariance() on the current fit."
            )


@dataclass(frozen=True)
class SandwichTelemetry:
    """Work counts and workspace sizes for one sandwich-covariance computation."""

    scored_rows: int
    positive_weight_rows: int
    n_chunks: int
    n_factorizations: int
    backend: str
    bread_reused: bool
    peak_bytes: int
    retained_bytes: int


@dataclass(frozen=True)
class SandwichCovariance:
    """Opt-in, fit-bound robust (sandwich) covariance result.

    Constructed via :meth:`superglm.SuperGLM.sandwich_covariance` (and future
    ``SuperLSS`` equivalents) and passed as ``covariance=`` to ``summary()``,
    ``relativities()``, ``term_inference()``, and ``simultaneous_bands()``.
    """

    status: SandwichStatus
    unavailable_reason: UnavailableReason | None = None
    message: str | None = None
    binding: FitBinding | None = None
    cov_type: SandwichCovType | None = None
    clustered: bool = False
    correction: bool = False
    correction_factor: float = 1.0
    centered: bool = False
    penalized: bool = False
    weight_semantics: str | None = None
    bread_curvature: BreadCurvature | None = None
    n_obs: int = 0
    n_units: float = 0.0
    n_clusters: int | None = None
    n_params: int = 0
    bread_rank: int = 0
    meat_rank: int | None = None
    covariance: NDArray | None = None
    se: NDArray | None = None
    coefficient_estimable: NDArray[np.bool_] | None = None
    active_groups: tuple[GroupSlice, ...] = ()
    penalized_group_names: frozenset[str] = frozenset()
    telemetry: SandwichTelemetry | None = None

    @property
    def available(self) -> bool:
        """Whether robust covariance was computed and is available for inference."""
        return self.status == "available"

    @property
    def slope_covariance(self) -> NDArray:
        """Slope-parameter block ``covariance[1:, 1:]`` in active coordinates."""
        self.require_available()
        assert self.covariance is not None
        view = self.covariance[1:, 1:]
        return view

    @property
    def method_label(self) -> str:
        """Human-readable label describing the robust covariance estimator."""
        if not self.available:
            return f"unavailable ({self.unavailable_reason})"
        base = f"Cluster {self.cov_type}" if self.clustered else str(self.cov_type)
        suffix = ", centered" if self.centered else ""
        return f"{base} (sandwich{suffix})"

    def require_available(self, binding: FitBinding | None = None) -> SandwichCovariance:
        """Validate fit binding and availability before a reader consumes this covariance."""
        if binding is not None and self.binding is not None:
            self.binding.validate_match(binding)
        elif binding is not None and self.binding is None:
            raise ValueError("SandwichCovariance carries no fit binding and cannot be verified.")
        if not self.available:
            detail = f": {self.message}" if self.message else ""
            raise RuntimeError(
                f"SandwichCovariance is unavailable ({self.unavailable_reason}){detail}"
            )
        return self

    def __repr__(self) -> str:
        if not self.available:
            return (
                f"SandwichCovariance(status='unavailable', "
                f"unavailable_reason={self.unavailable_reason!r}, message={self.message!r})"
            )
        units = (
            f"n_clusters={self.n_clusters}, n_units={self.n_units:g}"
            if self.clustered
            else f"n_units={self.n_units:g}"
        )
        return (
            f"SandwichCovariance(status='available', cov_type={self.cov_type!r}, "
            f"centered={self.centered}, {units}, n_params={self.n_params}, "
            f"bread_curvature={self.bread_curvature!r})"
        )


def build_available_covariance(
    *,
    binding: FitBinding,
    cov_type: SandwichCovType,
    clustered: bool,
    correction: bool,
    correction_factor: float,
    centered: bool,
    penalized: bool,
    weight_semantics: str,
    bread_curvature: BreadCurvature,
    n_obs: int,
    n_units: float,
    n_clusters: int | None,
    n_params: int,
    bread_rank: int,
    meat_rank: int | None,
    covariance: NDArray,
    se: NDArray,
    coefficient_estimable: NDArray[np.bool_],
    active_groups: tuple[GroupSlice, ...],
    penalized_group_names: frozenset[str],
    telemetry: SandwichTelemetry,
) -> SandwichCovariance:
    """Sole lifecycle constructor for an available ``SandwichCovariance``."""
    status = SANDWICH_LIFECYCLE[("init", "build_available")]
    return SandwichCovariance(
        status=status,
        unavailable_reason=None,
        message=None,
        binding=binding,
        cov_type=cov_type,
        clustered=clustered,
        correction=correction,
        correction_factor=float(correction_factor),
        centered=centered,
        penalized=penalized,
        weight_semantics=weight_semantics,
        bread_curvature=bread_curvature,
        n_obs=int(n_obs),
        n_units=float(n_units),
        n_clusters=None if n_clusters is None else int(n_clusters),
        n_params=int(n_params),
        bread_rank=int(bread_rank),
        meat_rank=None if meat_rank is None else int(meat_rank),
        covariance=_freeze(np.asarray(covariance, dtype=np.float64)),
        se=_freeze(np.asarray(se, dtype=np.float64)),
        coefficient_estimable=_freeze(np.asarray(coefficient_estimable, dtype=bool)),
        active_groups=tuple(active_groups),
        penalized_group_names=frozenset(penalized_group_names),
        telemetry=telemetry,
    )


def build_unavailable_covariance(
    *,
    unavailable_reason: UnavailableReason,
    message: str,
    binding: FitBinding | None = None,
    n_obs: int = 0,
    weight_semantics: str | None = None,
) -> SandwichCovariance:
    """Sole lifecycle constructor for an unavailable ``SandwichCovariance``."""
    if unavailable_reason not in UNAVAILABLE_REASONS:
        raise ValueError(f"Unknown unavailable_reason: {unavailable_reason!r}")
    status = SANDWICH_LIFECYCLE[("init", "build_unavailable")]
    return SandwichCovariance(
        status=status,
        unavailable_reason=unavailable_reason,
        message=str(message),
        binding=binding,
        n_obs=int(n_obs),
        weight_semantics=weight_semantics,
    )


def resolve_center_mode(center: str | bool, *, penalized: bool) -> bool:
    """Resolve ``center`` option (``'auto'``, ``'mean'``, ``'none'``, or bool)."""
    if isinstance(center, bool):
        return center
    if isinstance(center, str):
        key = center.strip().lower()
        if key == "auto":
            return bool(penalized)
        if key == "mean":
            return True
        if key == "none":
            return False
    raise ValueError(f"center must be 'auto', 'mean', 'none', or bool; got {center!r}")


def small_sample_multiplier(
    *,
    clustered: bool,
    correction: bool,
    n_units: float,
    n_params: int,
    n_clusters: int | None,
    penalized: bool,
    full_rank: bool,
) -> tuple[SandwichCovType, float]:
    """Validate and compute the finite-sample multiplier and ``cov_type`` label.

    Small-sample corrections (HC1 and statsmodels-style CR1) apply only to
    unpenalised, full-rank fits with validated positive denominators.
    """
    if not isinstance(correction, bool):
        raise ValueError(f"correction must be a bool, got {correction!r}")
    if not correction:
        return ("CR0" if clustered else "HC0"), 1.0

    if penalized:
        raise ValueError(
            "Small-sample correction (correction=True) is only supported on unpenalised "
            "full-rank fits; pass correction=False for penalised fits."
        )
    if not full_rank:
        raise ValueError(
            "Small-sample correction (correction=True) is only supported on full-rank "
            "unpenalised fits; pass correction=False for rank-deficient fits."
        )
    if n_units <= n_params:
        raise ValueError(
            f"Small-sample correction requires effective sample size n > k "
            f"(got n={n_units:g}, k={n_params}); pass correction=False."
        )
    if clustered:
        if n_clusters is None or n_clusters < 2:
            raise ValueError(
                f"Cluster small-sample correction requires at least 2 clusters (got {n_clusters})."
            )
        factor = (float(n_clusters) / float(n_clusters - 1)) * (
            (float(n_units) - 1.0) / (float(n_units) - float(n_params))
        )
        return "CR1", factor

    factor = float(n_units) / (float(n_units) - float(n_params))
    return "HC1", factor


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))
    try:
        # Catches pandas.NA, pandas.NaT, Decimal('NaN'), etc.
        equal = value == value
        if isinstance(equal, (bool, np.bool_)):
            return not bool(equal)
        return True
    except Exception:
        return True


def normalize_clusters(cluster: Any, weights: NDArray) -> tuple[NDArray[np.intp], int]:
    """Validate and encode one-way cluster labels aligned by original fit row position.

    Zero-weight rows are ignored (code ``-1``) even if their label is missing.
    Positive-weight rows must carry non-null, non-NaN, hashable labels and
    produce at least ``G >= 2`` distinct clusters.
    """
    n_rows = len(weights)
    module_name = type(cluster).__module__
    if module_name.startswith("polars") and hasattr(cluster, "to_list"):
        raw_values: Any = cluster.to_list()
        arr = np.empty(len(raw_values), dtype=object)
        arr[:] = raw_values
    elif hasattr(cluster, "to_numpy") and hasattr(cluster, "isna"):
        arr = np.asarray(cluster.to_numpy())
    elif isinstance(cluster, np.ndarray):
        arr = cluster
    elif isinstance(cluster, (list, tuple)):
        arr = np.empty(len(cluster), dtype=object)
        arr[:] = cluster
    else:
        arr = np.asarray(cluster)

    if arr.ndim != 1:
        raise ValueError(f"cluster must be one-dimensional, got shape {arr.shape}.")
    if len(arr) != n_rows:
        raise ValueError(
            f"cluster length ({len(arr)}) must match the number of fitted input rows ({n_rows})."
        )

    pos = np.asarray(weights, dtype=np.float64) > 0.0
    codes = np.full(n_rows, -1, dtype=np.intp)
    if not np.any(pos):
        raise ValueError("Clustered sandwich covariance requires at least 2 positive-weight rows.")

    pos_vals = arr[pos]
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        uniq, inv = np.unique(pos_vals, return_inverse=True)
        n_clusters = int(len(uniq))
        codes[pos] = inv.astype(np.intp, copy=False)
    elif np.issubdtype(arr.dtype, np.str_) or np.issubdtype(arr.dtype, np.bytes_):
        uniq, inv = np.unique(pos_vals, return_inverse=True)
        n_clusters = int(len(uniq))
        codes[pos] = inv.astype(np.intp, copy=False)
    elif np.issubdtype(arr.dtype, np.floating):
        if np.any(np.isnan(pos_vals)):
            raise ValueError("cluster labels on positive-weight rows must not be NaN or missing.")
        if np.any(~np.isfinite(pos_vals)):
            raise ValueError("cluster labels on positive-weight rows must be finite.")
        cleaned = np.where(pos_vals == 0.0, 0.0, pos_vals)
        uniq, inv = np.unique(cleaned, return_inverse=True)
        n_clusters = int(len(uniq))
        codes[pos] = inv.astype(np.intp, copy=False)
    else:
        mapping: dict[Any, int] = {}
        encoded = np.empty(len(pos_vals), dtype=np.intp)
        for idx, value in enumerate(pos_vals):
            if _is_missing_scalar(value):
                raise ValueError(
                    "cluster labels on positive-weight rows must not be None, NaN, or missing."
                )
            if isinstance(value, (float, np.floating)):
                fval = float(value)
                if not math.isfinite(fval):
                    raise ValueError("cluster labels on positive-weight rows must be finite.")
                key: Any = 0.0 if fval == 0.0 else fval
            else:
                key = value
            try:
                cid = mapping.get(key)
            except TypeError as exc:
                raise ValueError(
                    f"cluster labels must be hashable; got {type(value).__name__}."
                ) from exc
            if cid is None:
                cid = len(mapping)
                mapping[key] = cid
            encoded[idx] = cid
        n_clusters = len(mapping)
        codes[pos] = encoded

    if n_clusters < 2:
        raise ValueError(
            f"Clustered sandwich covariance requires at least 2 positive-weight clusters, "
            f"got {n_clusters}."
        )
    return codes, n_clusters


def resolve_workspace_budget(
    max_memory: int | str | None,
    model_max_memory: int | str = "auto",
) -> int:
    """Resolve the byte budget for sandwich covariance workspaces."""
    from superglm._parallel import _parse_memory, resolve_max_memory, validate_max_memory

    if max_memory is not None:
        validated = validate_max_memory(max_memory)
        if validated == "auto":
            return resolve_max_memory()
        return _parse_memory(validated)
    if model_max_memory != "auto":
        validated = validate_max_memory(model_max_memory)
        if validated != "auto":
            return _parse_memory(validated)
    return resolve_max_memory()


class ScaledGramAccumulator:
    """Power-of-two guarded symmetric PSD Gram accumulator ``sum Z_b' Z_b``.

    Keeps intermediate squares well inside binary64's normal exponent range
    without changing mantissa bits, and optionally applies an exact binary
    exponent shift ``2**final_power_shift`` on finalization.
    """

    def __init__(self, width: int, *, final_power_shift: int = 0) -> None:
        self.width = int(width)
        self._final_power_shift = int(final_power_shift)
        self._scale_exp: int | None = None
        self._gram = np.zeros((self.width, self.width), dtype=np.float64)

    def add_rows(self, rows: NDArray) -> None:
        """Accumulate ``rows.T @ rows`` for a ``(m, width)`` chunk."""
        if rows.shape[0] == 0 or self.width == 0:
            return
        max_abs = float(np.max(np.abs(rows)))
        if max_abs == 0.0:
            return
        if not math.isfinite(max_abs):
            self._gram.fill(np.nan)
            return
        exp_chunk = math.frexp(max_abs)[1]
        if self._scale_exp is None:
            self._scale_exp = exp_chunk
            scaled = np.ldexp(rows, -exp_chunk)
            self._gram += scaled.T @ scaled
            return
        if exp_chunk > self._scale_exp:
            diff = self._scale_exp - exp_chunk
            # 2 * diff < -1074 underflows the old smaller contribution completely.
            self._gram = (
                np.ldexp(self._gram, 2 * diff) if 2 * diff >= -1074 else np.zeros_like(self._gram)
            )
            self._scale_exp = exp_chunk
            scaled = np.ldexp(rows, -exp_chunk)
            self._gram += scaled.T @ scaled
            return
        diff = exp_chunk - self._scale_exp
        if diff >= -500:
            scaled = np.ldexp(rows, -self._scale_exp)
            self._gram += scaled.T @ scaled

    def finalize(self, multiplier: float = 1.0) -> NDArray | None:
        """Return symmetric PSD ``multiplier * 2**final_power_shift * sum Z' Z``, or ``None``."""
        if not np.all(np.isfinite(self._gram)):
            return None
        sym = 0.5 * (self._gram + self._gram.T)
        if self._scale_exp is None or multiplier == 0.0:
            return np.zeros((self.width, self.width), dtype=np.float64)
        m_mantissa, m_exp = math.frexp(float(multiplier))
        total_exp = 2 * self._scale_exp + self._final_power_shift + m_exp
        with np.errstate(over="ignore", under="ignore"):
            out = np.ldexp(sym * m_mantissa, total_exp)
        if not np.all(np.isfinite(out)):
            return None
        return 0.5 * (out + out.T)


def _shewchuk_grow(partials: list[float], value: float) -> None:
    """In-place Shewchuk grow-expansion step for exact float64 summation."""
    if value == 0.0:
        return
    kept = 0
    for idx in range(len(partials)):
        other = partials[idx]
        if abs(value) < abs(other):
            value, other = other, value
        high = value + other
        low = other - (high - value)
        if low != 0.0:
            partials[kept] = low
            kept += 1
        value = high
    del partials[kept:]
    if value != 0.0:
        partials.append(value)


class ClusterScoreAccumulator:
    """Cluster score accumulator with C-speed SpMM and exact cancellation refinement.

    Accumulates per-cluster sums and absolute-value sums in chunks. Any cluster
    column whose condition number ``sum |g| / |sum g|`` exceeds 16 (within OR
    across chunks) is refined by exact Shewchuk summation (``math.fsum``) over
    all its contributing rows, preserving full binary64 accuracy under severe
    within-cluster cancellation (e.g. ``[+1e16, +1, -1e16]``).
    """

    def __init__(self, n_clusters: int, width: int, codes: NDArray[np.intp]) -> None:
        self.n_clusters = int(n_clusters)
        self.width = int(width)
        self._codes = codes
        self.totals = np.zeros((self.n_clusters, self.width), dtype=np.float64)
        self.abs_totals = np.zeros((self.n_clusters, self.width), dtype=np.float64)

    @property
    def buffer_bytes(self) -> int:
        return int(self.totals.nbytes + self.abs_totals.nbytes)

    def add_chunk(self, start: int, stop: int, chunk_scores: NDArray) -> None:
        chunk_codes = self._codes[start:stop]
        pos = chunk_codes >= 0
        if not np.any(pos):
            return
        c_pos = chunk_codes[pos]
        s_pos = chunk_scores[pos]
        unique_c, inv = np.unique(c_pos, return_inverse=True)
        m_pos = len(c_pos)
        indicator = sparse.csr_matrix(
            (
                np.ones(m_pos, dtype=np.float64),
                (inv, np.arange(m_pos, dtype=np.intp)),
            ),
            shape=(len(unique_c), m_pos),
        )
        self.totals[unique_c] += indicator @ s_pos
        self.abs_totals[unique_c] += indicator @ np.abs(s_pos)

    def cancelled_clusters(self) -> NDArray[np.intp]:
        """Return indices of clusters that experienced > 4 bits of cancellation."""
        cancelled = np.any(
            (self.abs_totals > 0.0) & (self.abs_totals > 16.0 * np.abs(self.totals)),
            axis=1,
        )
        return np.flatnonzero(cancelled)

    def refine_cancelled(
        self,
        cancelled: NDArray[np.intp],
        row_scores_fn: Callable[[NDArray[np.intp]], NDArray],
        *,
        max_chunk_rows: int = 8192,
    ) -> None:
        """Recompute cancelled clusters with exact float64 summation over all their rows."""
        if len(cancelled) == 0:
            return
        mask = np.isin(self._codes, cancelled)
        rows = np.flatnonzero(mask)
        if len(rows) == 0:
            return
        row_codes = self._codes[rows]
        order = np.argsort(row_codes, kind="mergesort")
        rows_sorted = rows[order]
        codes_sorted = row_codes[order]
        boundaries = np.flatnonzero(np.diff(codes_sorted) != 0) + 1
        starts = np.concatenate(([0], boundaries))
        stops = np.concatenate((boundaries, [len(rows_sorted)]))
        for seg_start, seg_stop in zip(starts, stops, strict=True):
            cid = int(codes_sorted[seg_start])
            c_rows = rows_sorted[seg_start:seg_stop]
            if len(c_rows) <= max_chunk_rows:
                block = row_scores_fn(c_rows)
                for col in range(self.width):
                    self.totals[cid, col] = math.fsum(block[:, col])
            else:
                partials: list[list[float]] = [[] for _ in range(self.width)]
                for pos in range(0, len(c_rows), max_chunk_rows):
                    block = row_scores_fn(c_rows[pos : pos + max_chunk_rows])
                    for col in range(self.width):
                        for val in block[:, col]:
                            _shewchuk_grow(partials[col], float(val))
                for col in range(self.width):
                    self.totals[cid, col] = math.fsum(partials[col])


def wald_block_statistic(
    beta_block: NDArray,
    cov_block: NDArray,
) -> tuple[float | None, float | None]:
    """Compute ordinary multi-df Wald ``b' V^-1 b`` only if ``V`` spans ``len(b)``.

    Returns ``(None, None)`` when ``cov_block`` is non-finite or rank-deficient
    (for example when cluster count ``G <= len(b)``).
    """
    from scipy.stats import chi2 as chi2_dist

    b = np.asarray(beta_block, dtype=np.float64).ravel()
    V = np.asarray(cov_block, dtype=np.float64)
    d = len(b)
    if d == 0 or V.shape != (d, d):
        return None, None
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(V))):
        return None, None
    V_sym = 0.5 * (V + V.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(V_sym)
    except np.linalg.LinAlgError:
        return None, None
    max_eig = float(eigvals[-1])
    if max_eig <= 0.0 or not math.isfinite(max_eig):
        return None, None
    # Require full numerical rank across the requested contrast dimension.
    cutoff = float(max(d, 1)) * _EPS * max_eig
    if float(eigvals[0]) <= cutoff:
        return None, None
    proj = eigvecs.T @ b
    stat = float(np.sum((proj * proj) / eigvals))
    if not math.isfinite(stat) or stat < 0.0:
        return None, None
    p_val = float(chi2_dist.sf(stat, float(d)))
    return stat, p_val


__all__ = [
    "SANDWICH_LIFECYCLE",
    "UNAVAILABLE_REASONS",
    "BreadCurvature",
    "ClusterScoreAccumulator",
    "FitBinding",
    "SandwichCovType",
    "SandwichCovariance",
    "SandwichStatus",
    "SandwichTelemetry",
    "ScaledGramAccumulator",
    "UnavailableReason",
    "build_available_covariance",
    "build_unavailable_covariance",
    "coef_digest",
    "normalize_clusters",
    "resolve_center_mode",
    "resolve_workspace_budget",
    "small_sample_multiplier",
    "wald_block_statistic",
]
