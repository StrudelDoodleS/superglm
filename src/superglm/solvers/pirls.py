"""PIRLS solver with pluggable penalty proximal operators."""

from __future__ import annotations

import copy
import logging
import math
import time
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Literal, cast, get_args

import numpy as np
import scipy.linalg
import scipy.optimize
from numpy.typing import NDArray

from superglm._fit_trace import TraceRun
from superglm.distributions import Distribution
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    GroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from superglm.links import Link
from superglm.penalties.base import Penalty, penalty_can_zero_groups, penalty_targets_group
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.solvers.centered_system import (
    build_centered_system,
    grouped_augmented_factor,
    grouped_weighted_factor,
)
from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom
from superglm.solvers.irls_state import (
    _evaluate_irls_state,
    _irls_objective_relative_change,
    _irls_objective_scale,
    _IRLSState,
    _poisson_sqrt_halving_budget,
    _select_irls_trial,
    _stable_penalized_deviance_delta,
)
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    RankInfo,
    decompose_factor,
    decompose_gram_if_authoritative,
)
from superglm.solvers.working_rows import (
    coefficient_initial_intercept,
    coefficient_working_rows,
    pearson_chi2,
)
from superglm.types import GroupSlice

logger = logging.getLogger(__name__)


def _lambda2_is_identically_zero(lambda2: float | dict[str, float]) -> bool:
    if isinstance(lambda2, dict):
        return all(float(value) == 0.0 for value in lambda2.values())
    return float(lambda2) == 0.0


def _has_structural_smoothing_penalty(
    gms: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
) -> bool:
    """Whether ordinary penalty assembly can contribute a nonzero S block."""
    if _lambda2_is_identically_zero(lambda2):
        return False
    for gm, group in zip(gms, groups, strict=True):
        if not group.penalized:
            continue
        if isinstance(
            gm,
            SparseSSPGroupMatrix
            | SplineCategoricalGroupMatrix
            | DiscretizedSplineCategoricalGroupMatrix
            | DiscretizedSSPGroupMatrix,
        ):
            omega_components = getattr(gm, "omega_components", None)
            if omega_components is not None:
                from superglm.reml.penalty_algebra import resolve_component_lambda

                for suffix, _omega_j in omega_components:
                    lam = resolve_component_lambda(lambda2, group.name, suffix)
                    if float(lam) != 0.0:
                        return True
                continue
            lam = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
            if float(lam) != 0.0 and gm.omega is not None:
                return True
        elif group.scop_reparameterization is not None:
            lam = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
            if float(lam) != 0.0:
                return True
    return False


def _positive_working_weight_stats(W: NDArray) -> tuple[float, float, float]:
    """Return positive W minimum, maximum, and ratio, excluding zero-weight rows."""
    positive = W[W > 0]
    if positive.size == 0:
        return float("nan"), float("nan"), float("inf")

    positive_min = float(np.min(positive))
    positive_max = float(np.max(positive))
    if not np.isfinite(positive_max):
        ratio = float("inf")
    else:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratio = float(np.divide(positive_max, positive_min))
    return positive_min, positive_max, ratio


def _extreme_weight_indices(
    W: NDArray[np.float64], k: int = 5
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Indices of the k largest and k smallest working weights, each ordered by W.

    ``np.argpartition`` requires ``-n <= kth < n``.  When ``W.size <= k`` the
    bottom partition must use ``k - 1``: it selects the same k smallest entries
    and stays in bounds for every k in [1, W.size].  The top partition is
    already safe because ``-k >= -W.size``.
    """
    k = min(k, W.size)
    top_idx = np.argpartition(W, -k)[-k:]
    bot_idx = np.argpartition(W, k - 1)[:k]
    return (
        top_idx[np.argsort(W[top_idx])[::-1]],
        bot_idx[np.argsort(W[bot_idx])],
    )


class _FrozenResultMapping(Mapping[object, object]):
    """Pickle-safe mapping used for recursively published result metadata."""

    __slots__ = ("__mapping",)

    def __init__(self, values: Mapping[object, object]) -> None:
        from types import MappingProxyType

        self.__mapping = MappingProxyType(dict(values))

    def __getitem__(self, key: object) -> object:
        return self.__mapping[key]

    def __iter__(self) -> Iterator[object]:
        return iter(self.__mapping)

    def __len__(self) -> int:
        return len(self.__mapping)

    def __reduce__(self):
        return (type(self), (dict(self.__mapping),))


@dataclass(frozen=True)
class REMLGeometrySummary:
    """Cheap centered-system moments retained for in-loop REML consumers.

    Carries exactly what the exact-REML gradient and objective otherwise read
    from ``rank_info``: the weighted column means, the working-weight sum, and
    the data-gram column scales feeding the signed-gram stability policy.
    Populating it is O(p) from quantities the solve already holds, which lets
    the optimizer loop skip per-fit rank certification without changing what
    either consumer sees.
    """

    mean_x: NDArray
    sum_w: float
    column_scale: NDArray


# Every value ``termination_reason`` may hold except the marker stamped after
# the fit, non-terminal sentinel included -- so the set is neither "what the
# loops write" nor "what the field holds", and reading it as either misleads.
# The vocabulary lives here because this module declares both fields that carry
# it, and both loops assign into a local of this type, so a reason invented
# inside a loop is a type error on the line that invents it rather than a novel
# string reaching the consumers that switch on the field. ``continue`` is the
# sentinel: it labels an iteration that ended without ending the loop, so it
# reaches the per-iteration diagnostics and the trace but never a result, which
# carries only a reason that actually ended the fit. ``mode_certified`` is the
# member neither loop writes -- the terminal REML refit stamps it when a mode
# that exhausted its iteration budget clears its KKT certificate anyway. The
# gap runs the other way exactly once, and deliberately: the fitted-state
# invalidation path stamps a synthetic marker of its own over the field through
# a dynamic ``setattr`` this annotation cannot see, and that records a revision
# rather than a reason a loop stopped, so it stays out of the vocabulary even
# though the field does hold it at runtime.
type TerminationReason = Literal[
    "curvature_fallback",
    "curvature_rescue",
    "constraint_infeasible",
    "constraint_kkt_incomplete",
    "step_rejected",
    "nonfinite_deviance",
    "converged",
    "max_iter",
    "continue",
    "mode_certified",
]

# The same vocabulary at runtime, for consumers that enumerate the reasons
# instead of annotating against them. A PEP 695 alias resolves lazily, so
# ``get_args`` has to be handed the value rather than the alias object.
PIRLS_TERMINATION_REASONS: frozenset[str] = frozenset(get_args(TerminationReason.__value__))


@dataclass(frozen=True)
class IterationDiagnostics:
    """Per-iteration IRLS diagnostics for debugging convergence issues."""

    iteration: int
    deviance: float
    w_min: float
    w_max: float
    w_ratio: float
    mu_min: float
    mu_max: float
    eta_min: float
    eta_max: float
    intercept: float
    step_halvings: int
    # Indices of the 5 observations with largest/smallest W
    top_w_indices: NDArray  # (5,) int
    bottom_w_indices: NDArray  # (5,) int
    # Condition estimate and SVD fallback flag (direct solver only)
    cond_estimate: float | None = None
    used_svd_fallback: bool | None = None
    raw_w_min: float | None = None
    raw_w_max: float | None = None
    raw_w_ratio: float | None = None
    eta_min_unclipped: float | None = None
    eta_max_unclipped: float | None = None
    eta_clipped: bool | None = None
    working_mu_min: float | None = None
    working_mu_max: float | None = None
    working_eta_min: float | None = None
    working_eta_max: float | None = None
    working_eta_min_unclipped: float | None = None
    working_eta_max_unclipped: float | None = None
    working_eta_clipped: bool | None = None
    step_rejected: bool = False
    rank_truncated: bool | None = None
    trials_attempted: int = 1
    accepted_alpha: float = 1.0
    base_state_id: int | None = None
    proposal_state_id: int | None = None
    committed_state_id: int | None = None
    evaluation_id: int | None = None
    state_space: str = "solver"
    basis_id: int | None = None
    convergence_criterion: str | None = None
    convergence_value: float | None = None
    convergence_tolerance: float | None = None
    termination_reason: TerminationReason | None = None


@dataclass
class PIRLSResult:
    beta: NDArray
    intercept: float
    n_iter: int
    deviance: float
    converged: bool
    phi: float
    effective_df: float
    iteration_log: list[IterationDiagnostics] | tuple[IterationDiagnostics, ...] | None = None
    # REML geometry after profiling the intercept. At full rank ``log_det_H``
    # is log|H_aug|. Under rank truncation it is the identified-coordinate
    # measure log(sum(W)) + log|H_c|_+, not the raw augmented matrix's
    # Moore-Penrose pseudo-determinant.
    log_det_H: float | None = None  # noqa: N815
    reml_hessian_rank: int | None = None
    reml_geometry: REMLGeometrySummary | None = None
    # Centered slope-system decomposition, retained only when the caller
    # opts in. Factor-L retention seam for RFC-2/RFC-7 consumers (route
    # solves through the factor instead of a materialized p x p
    # pseudo-inverse); RFC-12b, its original motivation, is retired per
    # audit 2026-07-28 §J.6. Dense exact path only.
    reml_slope_decomposition: RankDecomposition | None = None
    rank_info: RankInfo | None = None
    state_id: int | None = None
    evaluation_id: int | None = None
    state_space: str = "solver"
    basis_id: int | None = None
    termination_reason: TerminationReason | None = None
    direct_backend: str | None = None
    direct_fallback_reason: str | None = None
    # Terminal SCOP geometry is retained separately because covariance uses
    # expected Fisher curvature while EDF uses the full-Newton latent Hessian.
    scop_geometry: object | None = None
    scop_inference: object | None = None

    def __setattr__(self, name: str, value: object) -> None:
        if self.__dict__.get("_publication_locked", False):
            raise AttributeError(f"published PIRLSResult is immutable; cannot rebind {name!r}")
        object.__setattr__(self, name, value)

    def _mutable_copy(self, memo: dict[int, object] | None = None) -> PIRLSResult:
        """Return an unpublished private copy for a fitted-state revision."""
        if memo is None:
            memo = {}
        return self._copy_with_publication(published=False, memo=memo)

    def _publish(self, memo: dict[int, object] | None = None) -> PIRLSResult:
        """Seal this solver result at the fitted-state publication boundary."""
        if self.__dict__.get("_publication_locked", False):
            return self
        if memo is None:
            memo = {}
        for name, value in tuple(self.__dict__.items()):
            if name == "_publication_locked":
                continue
            object.__setattr__(self, name, _freeze_result_arrays(value, memo))
        object.__setattr__(self, "_publication_locked", True)
        return self

    def _copy_with_publication(
        self,
        *,
        published: bool,
        memo: dict[int, object],
    ) -> PIRLSResult:
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        object.__setattr__(clone, "_publication_locked", False)
        for name, value in self.__dict__.items():
            if name == "_publication_locked":
                continue
            object.__setattr__(clone, name, copy.deepcopy(value, memo))
        if published:
            clone._publish(memo)
        return clone

    def __deepcopy__(self, memo: dict[int, object]) -> PIRLSResult:
        existing = memo.get(id(self))
        if existing is not None:
            return existing  # type: ignore[return-value]
        return self._copy_with_publication(
            published=self.__dict__.get("_publication_locked", False),
            memo=memo,
        )

    def __getstate__(self) -> dict[str, object]:
        return dict(self.__dict__)

    def __setstate__(self, state: dict[str, object]) -> None:
        published = bool(state.get("_publication_locked", False))
        object.__setattr__(self, "_publication_locked", False)
        for name, value in state.items():
            if name != "_publication_locked":
                object.__setattr__(self, name, value)
        if published:
            self._publish()


def _immutable_array_copy(value: NDArray) -> NDArray:
    """Copy an array onto a bytes-backed buffer whose write flag cannot be restored."""
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _freeze_result_arrays(value: object, memo: dict[int, object]) -> object:
    """Replace coefficient-scale arrays in result metadata with immutable copies."""
    if isinstance(value, np.ndarray):
        frozen = _immutable_array_copy(value)
        memo[id(value)] = frozen
        return frozen

    existing = memo.get(id(value))
    if existing is not None:
        return existing

    if isinstance(value, RankInfo):
        # Rank geometry is intentionally shared by the public and solver
        # projections.  Freeze this metadata graph in place so deepcopy and
        # pickle round-trips preserve that identity instead of retaining two
        # potentially O(p²) decompositions.  This runs only at a publication
        # boundary or on a private copied/unpickled graph.
        memo[id(value)] = value
        for item in fields(value):
            object.__setattr__(
                value,
                item.name,
                _freeze_result_arrays(getattr(value, item.name), memo),
            )
        return value

    if is_dataclass(value) and not isinstance(value, type):
        clone = type(value).__new__(type(value))
        memo[id(value)] = clone
        for item in fields(value):
            object.__setattr__(
                clone,
                item.name,
                _freeze_result_arrays(getattr(value, item.name), memo),
            )
        if hasattr(value, "__dict__"):
            field_names = {item.name for item in fields(value)}
            for name, item_value in value.__dict__.items():
                if name not in field_names:
                    object.__setattr__(
                        clone,
                        name,
                        _freeze_result_arrays(item_value, memo),
                    )
        return clone

    if isinstance(value, list):
        clone_tuple = tuple(_freeze_result_arrays(item, memo) for item in value)
        memo[id(value)] = clone_tuple
        return clone_tuple

    if isinstance(value, tuple):
        clone_tuple = tuple(_freeze_result_arrays(item, memo) for item in value)
        memo[id(value)] = clone_tuple
        return clone_tuple

    if isinstance(value, Mapping):
        frozen_mapping = _FrozenResultMapping(
            {
                _freeze_result_arrays(key, memo): _freeze_result_arrays(item, memo)
                for key, item in value.items()
            }
        )
        memo[id(value)] = frozen_mapping
        return frozen_mapping

    return value


def _build_group_hessians(
    gms: list[GroupMatrix],
    W: NDArray,
    groups: list[GroupSlice] | None = None,
    S: NDArray | None = None,
) -> list[NDArray]:
    """Build the exact per-group Hessians of the smooth subproblem."""
    if S is not None and groups is None:
        raise ValueError("groups are required when a smooth penalty matrix is supplied")

    hessians: list[NDArray] = []
    for index, gm in enumerate(gms):
        hessian = gm.gram(W)
        if S is not None:
            assert groups is not None
            group_slice = groups[index].sl
            hessian = hessian + S[group_slice, group_slice]
        hessians.append(0.5 * (hessian + hessian.T))
    return hessians


def _compute_group_hessians(
    gms: list[GroupMatrix],
    W: NDArray,
    groups: list[GroupSlice] | None = None,
    S: NDArray | None = None,
) -> tuple[list[float], list[NDArray]]:
    """Return per-group smooth Hessians and their Lipschitz constants.

    The smooth block Hessian is ``X_g' diag(W) X_g + S_gg``.  No
    conditioning ridge is added: doing so changes the fitted objective.  The
    matrices are also reused by the exact Ridge block solver.

    For typical group sizes (p_g <= 20) this is trivially cheap.
    Total cost is O(n * p) across all groups.
    """
    hessians = _build_group_hessians(gms, W, groups, S)
    L_groups = [_block_lipschitz(hessian) for hessian in hessians]
    return L_groups, hessians


def _block_lipschitz(hessian: NDArray) -> float:
    """Bound represented positive curvature without introducing a unit floor."""
    scale = float(np.max(np.abs(hessian), initial=0.0))
    if not np.isfinite(scale):
        raise np.linalg.LinAlgError("nonfinite composite block Hessian")
    if scale == 0.0:
        return 0.0
    eigenvalues = scipy.linalg.eigvalsh(hessian / scale, check_finite=False)
    cutoff = np.finfo(float).eps * len(eigenvalues) * float(np.max(np.abs(eigenvalues)))
    if eigenvalues[0] < -cutoff:
        raise np.linalg.LinAlgError("composite block Hessian is not positive semidefinite")
    return float((max(float(eigenvalues[-1]), 0.0) + cutoff) * scale)


def _safe_norm(values: NDArray) -> float:
    """Use the scaled BLAS Euclidean norm instead of squaring raw coordinates."""
    return float(scipy.linalg.norm(values, check_finite=False))


def _positive_product_ratio(left: float, right: float, denominator: float) -> float:
    """Evaluate nonnegative ``left * right / denominator`` without an unsafe product."""
    if not all(np.isfinite(value) for value in (left, right, denominator)):
        return float("inf")
    if denominator <= 0.0 or left < 0.0 or right < 0.0:
        return float("inf")
    if left == 0.0 or right == 0.0:
        return 0.0
    lm, le = math.frexp(left)
    rm, re = math.frexp(right)
    dm, de = math.frexp(denominator)
    try:
        return math.ldexp(lm * rm / dm, le + re - de)
    except OverflowError:
        return float("inf")


def _roundoff_gamma(operations: int) -> float:
    eps_count = operations * np.finfo(float).eps
    if not 0.0 <= eps_count < 1.0:
        raise ValueError("dimensions do not admit the composite stationarity roundoff bound")
    return eps_count / (1.0 - eps_count)


def _stationarity_ratio(residual: float, scale: float, allowance: float, tol: float) -> float:
    """Encode ``residual <= tol * scale + allowance`` in the existing tol units."""
    if not all(np.isfinite(value) and value >= 0.0 for value in (residual, scale, allowance)):
        return float("inf")
    if residual == 0.0:
        return 0.0
    normalizer = max(residual, scale, allowance)
    denominator = tol * (scale / normalizer) + allowance / normalizer
    if denominator == 0.0:
        return float("inf")
    ratio = _positive_product_ratio(tol, residual / normalizer, denominator)
    # An underflowed diagnostic must not turn a nonzero computed residual into
    # an exact fixed point in the published trace.
    return max(ratio, float(np.nextafter(0.0, 1.0)))


def _factor_psd_block(matrix: NDArray) -> tuple[NDArray, bool]:
    """Factor a positive-semidefinite block without perturbing its objective."""
    try:
        return scipy.linalg.cholesky(matrix, lower=True, check_finite=False), True
    except scipy.linalg.LinAlgError:
        return scipy.linalg.pinvh(matrix, check_finite=False), False


def _solve_factored_block(factor: tuple[NDArray, bool], rhs: NDArray) -> NDArray:
    values, is_cholesky = factor
    if is_cholesky:
        return scipy.linalg.cho_solve((values, True), rhs, check_finite=False)
    return values @ rhs


def _ridge_block_factors(
    hessians: list[NDArray],
    groups: list[GroupSlice],
    penalty: Ridge,
) -> list[tuple[NDArray, bool]]:
    """Pre-factor the exact conditional Hessian for every Ridge block."""
    factors: list[tuple[NDArray, bool]] = []
    lam = float(penalty.lambda1 or 0.0)
    for hessian, group in zip(hessians, groups, strict=True):
        system = hessian.copy()
        if penalty_targets_group(penalty, group) and lam != 0.0:
            system[np.diag_indices_from(system)] += lam
        factors.append(_factor_psd_block(system))
    return factors


def _radial_block_eigensystems(
    hessians: list[NDArray],
    groups: list[GroupSlice],
    penalty: GroupLasso | GroupElasticNet,
) -> tuple[list[float], list[tuple[NDArray, NDArray, float, NDArray]]]:
    """Normalize each whole quadratic, preserving its Euclidean radial penalty."""
    systems: list[tuple[NDArray, NDArray, float, NDArray]] = []
    lipschitz: list[float] = []
    lam = float(penalty.lambda1 or 0.0)
    ridge_fraction = 0.0 if type(penalty) is GroupLasso else 1.0 - penalty.alpha
    for hessian, group in zip(hessians, groups, strict=True):
        system = hessian.copy()
        if penalty_targets_group(penalty, group) and ridge_fraction != 0.0:
            system[np.diag_indices_from(system)] += lam * ridge_fraction
        scale = float(np.max(np.abs(system), initial=0.0))
        if not np.isfinite(scale):
            raise np.linalg.LinAlgError("nonfinite composite block Hessian")
        if scale != 0.0:
            system /= scale
        if system.shape == (1, 1):
            eigenvalues = np.array([float(system[0, 0])])
            eigenvectors = np.ones((1, 1))
        else:
            eigenvalues, eigenvectors = scipy.linalg.eigh(system, check_finite=False)
        cutoff = np.finfo(float).eps * len(eigenvalues) * float(np.max(np.abs(eigenvalues)))
        eigenvalues[np.abs(eigenvalues) <= cutoff] = 0.0
        if np.any(eigenvalues < 0.0):
            raise np.linalg.LinAlgError("composite block Hessian is not positive semidefinite")
        systems.append((eigenvalues, eigenvectors, scale, system))
        lipschitz.append(float((eigenvalues[-1] + cutoff) * scale))
    return lipschitz, systems


def _radial_action_is_compatible(
    system: NDArray,
    beta: NDArray,
    rhs: NDArray,
    threshold: float,
) -> bool:
    """Check the original quadratic action, including eigensystem error.

    A componentwise residual bound cannot hide a score in an exactly zero
    matrix row behind the magnitude of unrelated, resolved rows.  The gamma
    count covers the matrix action, radial norm/division and residual assembly.
    """
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        force = np.zeros_like(beta)
        if threshold > 0.0:
            norm = _safe_norm(beta)
            if not np.isfinite(norm):
                return False
            if norm == 0.0:
                # At zero the radial subgradient is a ball.  Its nearest
                # point to rhs certifies the possibly rounded activity test.
                rhs_norm = _safe_norm(rhs)
                if not np.isfinite(rhs_norm):
                    return False
                if rhs_norm > 0.0:
                    force = min(threshold, rhs_norm) * (rhs / rhs_norm)
            else:
                force = threshold * (beta / norm)
        residual = system @ beta + force - rhs
        action_scale = np.abs(system) @ np.abs(beta) + np.abs(force) + np.abs(rhs)
        allowance = _roundoff_gamma(2 * len(beta) + 8) * action_scale
    return bool(
        np.all(np.isfinite(residual))
        and np.all(np.isfinite(allowance))
        and np.all(np.abs(residual) <= allowance)
    )


def _solve_radial_block(
    eigensystem: tuple[NDArray, NDArray, float, NDArray],
    rhs: NDArray,
    threshold: float,
) -> NDArray:
    """Solve the radial KKT equation in whole-quadratic normalized units."""
    eigenvalues, eigenvectors, scale, system = eigensystem
    if not np.isfinite(threshold) or threshold < 0.0 or not np.all(np.isfinite(rhs)):
        raise np.linalg.LinAlgError("nonfinite or invalid radial subproblem")
    if _safe_norm(rhs) <= threshold:
        return np.zeros_like(rhs)
    if scale == 0.0:
        raise np.linalg.LinAlgError("unbounded radial subproblem with zero curvature")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled_rhs = rhs / scale
        projected = eigenvectors.T @ scaled_rhs
        threshold = float(np.divide(threshold, scale))
    if not np.all(np.isfinite(projected)) or not np.isfinite(threshold):
        raise np.linalg.LinAlgError("nonfinite normalized radial subproblem")
    rhs_norm = _safe_norm(scaled_rhs)
    norm_allowance = _roundoff_gamma(2 * len(rhs) + 8) * rhs_norm
    if np.isfinite(rhs_norm) and rhs_norm - threshold <= norm_allowance:
        # Normalizing and projecting can erase the last ulp of an active
        # norm margin.  Certify zero against the original normalized RHS
        # instead of asking a rounded root equation for an infinite bracket.
        zero = np.zeros_like(rhs)
        if _radial_action_is_compatible(system, zero, scaled_rhs, threshold):
            return zero
    null = eigenvalues == 0.0
    null_norm = _safe_norm(projected[null])
    if null_norm != 0.0 and null_norm >= threshold:
        # Computed null eigenvectors need not annihilate a compatible RHS
        # exactly.  Certify against the original normalized matrix before
        # discarding any such component; a projection-dot error bound alone
        # would miss errors in the eigenvectors themselves.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            range_beta = eigenvectors[:, ~null] @ (projected[~null] / eigenvalues[~null])
        if _radial_action_is_compatible(system, range_beta, scaled_rhs, 0.0):
            projected[null] = 0.0
            null_norm = 0.0
    if null_norm > threshold:
        raise np.linalg.LinAlgError("unbounded radial subproblem in the quadratic null space")
    if threshold > 0.0 and null_norm == threshold:
        raise np.linalg.LinAlgError("radial subproblem has no finite minimizer")
    if len(eigenvalues) == 1:
        numerator = np.sign(projected[0]) * max(abs(float(projected[0])) - threshold, 0.0)
        with np.errstate(over="ignore"):
            result = eigenvectors[:, 0] * (numerator / eigenvalues[0])
    elif threshold == 0.0:
        inverse = np.divide(
            projected,
            eigenvalues,
            out=np.zeros_like(projected),
            where=eigenvalues > 0.0,
        )
        result = eigenvectors @ inverse
    else:

        def root(rho: float) -> float:
            if rho == 0.0:
                return null_norm - threshold
            # rho/(d+rho) lies in [0, 1].  Forming this ratio first avoids
            # overflowing the inverse or underflowing rho * norm(inverse).
            value = _safe_norm(projected * (rho / (eigenvalues + rho))) - threshold
            if not np.isfinite(value):
                raise np.linalg.LinAlgError("nonfinite radial root equation")
            return value

        upper = float(eigenvalues[-1])
        while root(upper) <= 0.0:
            if upper >= np.finfo(float).max / 2.0:
                raise np.linalg.LinAlgError("radial root has no finite bracket")
            upper *= 2.0
        rho = scipy.optimize.brentq(
            root,
            0.0,
            upper,
            xtol=np.nextafter(0.0, 1.0),
            rtol=4.0 * np.finfo(float).eps,
            maxiter=2048,
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            result = eigenvectors @ (projected / (eigenvalues + rho))
    if not np.all(np.isfinite(result)):
        raise np.linalg.LinAlgError("nonfinite radial block solution")
    if np.any(null) and not _radial_action_is_compatible(system, result, scaled_rhs, threshold):
        raise np.linalg.LinAlgError("radial solution cannot be certified against its quadratic")
    return result


def _composite_kkt_violation(
    *,
    dm: DesignMatrix,
    state: _IRLSState,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    offset: NDArray,
    groups: list[GroupSlice],
    penalty: Penalty,
    S: NDArray | None,
    has_smooth_penalty: bool,
    L_groups: list[float] | None = None,
    curvature_weights: NDArray | None = None,
    tol: float = 1e-6,
) -> float:
    """Return homogeneous proximal stationarity with represented-score roundoff.

    Zero is equivalent to the composite KKT equations for every penalty that
    implements the solver's proximal protocol.  The arithmetic allowance is
    for the represented working rows and smooth penalty, not uncertainty in
    likelihood derivatives.  Cached curvature must use these same weights.
    """
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("composite stationarity tolerance must be finite and positive")
    n, p = dm.shape
    score_gamma = _roundoff_gamma(n + p + 4)
    intercept_gamma = _roundoff_gamma(n + 2)
    working_rows = coefficient_working_rows(
        distribution=family,
        link=link,
        y=y,
        mu=state.mu,
        eta=state.eta,
        sample_weight=weights,
        prefer_observed=False,
    )
    W = working_rows.weights
    z = working_rows.response
    working_residual = z - state.eta
    loss_gradient = -dm.rmatvec(W * working_residual)
    if has_smooth_penalty:
        assert S is not None
        smooth_gradient = loss_gradient + S @ state.beta
    else:
        smooth_gradient = loss_gradient
    if L_groups is None or curvature_weights is None or not np.array_equal(W, curvature_weights):
        L_groups, _ = _compute_group_hessians(
            list(dm.group_matrices),
            W,
            groups if has_smooth_penalty else None,
            S if has_smooth_penalty else None,
        )

    with np.errstate(over="ignore", invalid="ignore"):
        intercept_residual = abs(float(np.sum(W * working_residual)))
        intercept_scale = float(np.sum(W * (np.abs(z) + np.abs(state.eta))))
        root_w = np.sqrt(W)
        response_norm = _safe_norm(root_w * z) + _safe_norm(root_w * state.eta)
    if not np.isfinite(response_norm) or not np.all(np.isfinite(smooth_gradient)):
        return float("inf")
    max_violation = _stationarity_ratio(
        intercept_residual,
        intercept_scale,
        intercept_gamma * intercept_scale,
        tol,
    )
    for group, L_g in zip(groups, L_groups, strict=True):
        if not np.isfinite(L_g) or L_g < 0.0:
            return float("inf")
        # An arbitrary positive step is valid for a structurally zero quadratic.
        # It must never replace a represented positive curvature.
        step_curvature = L_g if L_g > 0.0 else 1.0
        step = 1.0 / step_curvature
        if not np.isfinite(step):
            return float("inf")
        beta_g = state.beta[group.sl]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            u = smooth_gradient[group.sl] / step_curvature
            candidate = penalty.prox_group(beta_g - u, group, step)
            d = beta_g - candidate
            penalty_force = d - u
        if not all(np.all(np.isfinite(value)) for value in (u, candidate, d, penalty_force)):
            return float("inf")
        beta_norm, candidate_norm, u_norm = map(_safe_norm, (beta_g, candidate, u))
        scale = max(beta_norm, candidate_norm, u_norm, _safe_norm(penalty_force))
        # || |X_g|' W (|z| + |eta|) || <= sqrt(k_g L_g) R, since
        # each weighted column norm is bounded by sqrt(L_g).
        bound = (
            _positive_product_ratio(math.sqrt(group.size), response_norm, math.sqrt(L_g))
            if L_g > 0.0
            else 0.0
        )
        if has_smooth_penalty:
            assert S is not None
            with np.errstate(over="ignore", invalid="ignore"):
                bound += _safe_norm(np.abs(S[group.sl, :]) @ np.abs(state.beta)) / step_curvature
        allowance = score_gamma * bound
        allowance += _roundoff_gamma(group.size + 5) * (beta_norm + candidate_norm + u_norm)
        max_violation = max(
            max_violation,
            _stationarity_ratio(_safe_norm(d), scale, allowance, tol),
        )
    return max_violation


def _add_selection_local_curvature(
    *,
    curvature: NDArray,
    penalty: Penalty,
    beta: NDArray,
    original_groups: list[GroupSlice],
    active_groups: list[GroupSlice],
) -> None:
    """Add exact built-in selection curvature on the active manifold.

    Ridge contributes its ordinary quadratic Hessian.  For a nonzero group
    lasso block, the Hessian of ``lambda * weight * ||beta_g||`` is the
    tangential projector ``lambda * weight / ||beta_g|| * (I - uu')``.
    This is the local sensitivity geometry used by the general-design group
    lasso degrees-of-freedom formula.  Custom penalty subclasses retain the
    historical protocol fallback rather than inheriting built-in semantics.
    """
    width = sum(group.size for group in active_groups)
    if curvature.shape != (width, width):
        raise ValueError("selection curvature matrix does not match active groups")
    if type(penalty) not in (Ridge, GroupLasso, GroupElasticNet):
        return

    lam = float(penalty.lambda1 or 0.0)
    if lam == 0.0:
        return
    for original, active in zip(original_groups, active_groups, strict=True):
        if not penalty_targets_group(penalty, original):
            continue
        block = curvature[active.sl, active.sl]
        if type(penalty) is Ridge:
            block[np.diag_indices_from(block)] += lam
            continue

        if type(penalty) is GroupElasticNet:
            elastic_net = cast(GroupElasticNet, penalty)
            radial_fraction = float(elastic_net.alpha)
            ridge_fraction = 1.0 - radial_fraction
            if ridge_fraction != 0.0:
                block[np.diag_indices_from(block)] += lam * ridge_fraction
        else:
            radial_fraction = 1.0

        radial_scale = lam * radial_fraction * original.weight
        beta_group = np.asarray(beta[original.sl], dtype=np.float64)
        norm_group = float(np.linalg.norm(beta_group))
        if radial_scale == 0.0 or norm_group == 0.0:
            continue
        unit = beta_group / norm_group
        block += (radial_scale / norm_group) * (
            np.eye(original.size, dtype=np.float64) - np.outer(unit, unit)
        )


def _selection_local_curvature_depends_on_beta(penalty: Penalty) -> bool:
    """Whether exact built-in inference curvature changes with coefficients."""
    lam = float(penalty.lambda1 or 0.0)
    if lam == 0.0:
        return False
    if type(penalty) is GroupLasso:
        return True
    if type(penalty) is GroupElasticNet:
        return float(cast(GroupElasticNet, penalty).alpha) != 0.0
    return False


def _fit_pirls_inner(
    dm: DesignMatrix,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    groups: list[GroupSlice],
    penalty: Penalty,
    offset: NDArray,
    beta_init: NDArray | None = None,
    intercept_init: float | None = None,
    max_iter_outer: int = 100,
    max_iter_inner: int = 5,
    tol: float = 1e-6,
    active_set: bool = False,
    lambda2: float | dict[str, float] = 0.0,
    record_diagnostics: bool = False,
    convergence: str = "deviance",
    S_override: NDArray | None = None,
    trace_run: TraceRun | None = None,
    trace_basis_id: int | None = None,
    trace_purpose: str = "fit",
    separation: str = "warn",
    *,
    weight_semantics: str,
) -> PIRLSResult:
    """Single-pass PIRLS fit with a composite block-coordinate inner solver."""
    n, p = dm.shape
    beta = beta_init.copy() if beta_init is not None else np.zeros(p)
    # Always an empty list; only ``record_diagnostics`` decides whether rows are
    # appended and whether it is published on the result.
    iteration_log: list[IterationDiagnostics] = []
    objective_merit_scale = _irls_objective_scale(
        y=y,
        weights=weights,
        family=family,
        link=link,
    )

    # Initialize intercept
    if intercept_init is not None:
        intercept = intercept_init
    else:
        intercept = coefficient_initial_intercept(
            distribution=family,
            link=link,
            y=y,
            sample_weight=weights,
        )

    gms = list(dm.group_matrices)
    n_groups = len(groups)
    can_zero_groups = penalty_can_zero_groups(penalty)

    # The fitted composite objective is
    #   0.5 * D(beta) + 0.5 * beta' S beta + penalty.eval(beta).
    # Build S once so coefficient updates, line-search merit, and inference all
    # use the same matrix.  Keep a branch-free fast path for the common S=0
    # selection-only fit.
    if S_override is None and not _has_structural_smoothing_penalty(gms, groups, lambda2):
        S = None
    elif S_override is None:
        from superglm.reml.penalty_algebra import build_penalty_matrix

        S = build_penalty_matrix(gms, groups, lambda2, p)
    else:
        S = np.asarray(S_override, dtype=np.float64)
        if S.shape != (p, p):
            raise ValueError(f"S_override must have shape {(p, p)}, got {S.shape}")
    has_smooth_penalty = S is not None and bool(np.any(S))
    if not has_smooth_penalty:
        S = None

    t_total = time.perf_counter()
    t_lipschitz_total = 0.0
    t_inner_total = 0.0
    total_inner_iters = 0
    total_groups_skipped = 0

    trace_enabled = trace_run is not None and trace_run.enabled
    if trace_enabled and trace_basis_id is None:
        assert trace_run is not None
        trace_basis_id = trace_run.next_basis_id()
    resolved_lambdas: tuple[tuple[str, object], ...]
    if not trace_enabled:
        resolved_lambdas = ()
    elif isinstance(lambda2, dict):
        resolved_lambdas = tuple(
            [("selection", penalty.lambda1)]
            + [(f"smooth:{name}", float(value)) for name, value in sorted(lambda2.items())]
        )
    else:
        resolved_lambdas = (
            ("selection", penalty.lambda1),
            ("smooth", float(lambda2)),
        )

    def nonsmooth_merit(values: NDArray) -> float:
        """The selection penalty on the *merit* convention.

        The composite objective is ``0.5 * D + 0.5 * b' S b + penalty.eval(b)``
        and the merit tracked here is twice it, on the deviance scale, so the
        nonsmooth term enters at ``2 *``.  The recorded ``penalized_deviance``
        and the line search's merit delta must agree on that factor or the
        search would optimize a different objective than the one it records --
        the exact incoherence this shared closure exists to prevent.
        """
        return 2.0 * float(penalty.eval(values, groups))

    def evaluate_state(
        beta_values: NDArray,
        intercept_value: float,
        *,
        phase: str,
        outer_iteration: int,
        alpha: float | None = None,
        eta_unclipped: NDArray | None = None,
    ) -> _IRLSState:
        if trace_enabled:
            assert trace_run is not None
            state_id = trace_run.next_state_id()
            evaluation_id = trace_run.next_evaluation_id()
        else:
            state_id = None
            evaluation_id = None
        state = _evaluate_irls_state(
            dm,
            y,
            weights,
            family,
            link,
            offset,
            beta_values,
            intercept_value,
            eta_unclipped=eta_unclipped,
            state_id=state_id,
            evaluation_id=evaluation_id,
            basis_id=trace_basis_id,
            lambdas=resolved_lambdas,
        )
        state = replace(
            state,
            penalized_deviance=float(
                state.deviance
                + (
                    float(state.beta @ S @ state.beta)
                    if has_smooth_penalty and S is not None
                    else 0.0
                )
                + nonsmooth_merit(state.beta)
            ),
        )
        if trace_enabled:
            assert trace_run is not None
            trace_run.emit_lazy(
                "evaluation",
                lambda: {
                    "state_id": state.state_id,
                    "evaluation_id": state.evaluation_id,
                    "solver": "pirls",
                    "phase": phase,
                    "outer_iteration": outer_iteration,
                    "trial_alpha": alpha,
                    "state_space": state.state_space,
                    "basis_id": state.basis_id,
                    "lambdas": state.lambdas,
                    "dispersion": state.dispersion,
                    "intercept": state.intercept,
                    "deviance": state.deviance,
                    "penalized_deviance": state.penalized_deviance,
                },
                channel="pirls",
                purpose=trace_purpose,
                authoritative=False,
            )
        return state

    def emit_state_commit(
        state: _IRLSState,
        *,
        outer_iteration: int,
        fit_converged: bool,
        convergence_criterion: str | None,
        convergence_value: float | None,
        termination_reason: TerminationReason | None,
    ) -> None:
        if not trace_enabled:
            return
        assert trace_run is not None
        trace_run.emit_lazy(
            "state_commit",
            lambda: {
                "state_id": state.state_id,
                "evaluation_id": state.evaluation_id,
                "solver": "pirls",
                "phase": "initial" if outer_iteration == 0 else "outer",
                "outer_iteration": outer_iteration,
                "state_space": state.state_space,
                "basis_id": state.basis_id,
                "lambdas": state.lambdas,
                "dispersion": state.dispersion,
                "intercept": state.intercept,
                "deviance": state.deviance,
                "penalized_deviance": state.penalized_deviance,
                "fit_converged": fit_converged,
                "convergence_criterion": convergence_criterion,
                "convergence_value": convergence_value,
                "convergence_tolerance": tol,
                "termination_reason": termination_reason,
            },
            channel="pirls",
            purpose=trace_purpose,
        )

    # Freeze the fit-entry state so every trial is evaluated from fixed endpoints.
    committed = evaluate_state(beta, intercept, phase="initial", outer_iteration=0)
    emit_state_commit(
        committed,
        outer_iteration=0,
        fit_converged=False,
        convergence_criterion=None,
        convergence_value=None,
        termination_reason=None,
    )
    objective_prev = committed.penalized_deviance
    if objective_prev is None or not np.isfinite(objective_prev):
        objective_prev = np.inf
    converged = False
    max_halving = 20  # max step-halving attempts per outer iteration
    # Declared, not bound: the loop's reason chain is checked against the shared
    # vocabulary at each assignment, so a reason this module gains has to be
    # added to ``TerminationReason`` before it can leave the loop.
    termination_reason: TerminationReason
    for outer in range(max_iter_outer):
        t_outer_start = time.perf_counter()

        beta_prev = committed.beta
        intercept_prev = committed.intercept
        beta = committed.beta.copy()
        intercept = committed.intercept

        # Current predictions are the complete retained snapshot.
        eta_unclipped = committed.eta_unclipped
        eta = committed.eta
        mu = committed.mu

        # Working weights and response (PIRLS)
        working_rows = coefficient_working_rows(
            distribution=family,
            link=link,
            y=y,
            mu=mu,
            eta=eta,
            sample_weight=weights,
            prefer_observed=False,
        )
        W = working_rows.weights
        z = working_rows.response

        # Per-group Hessians and Lipschitz constants
        t0 = time.perf_counter()
        block_hessians = _build_group_hessians(
            gms,
            W,
            groups if has_smooth_penalty else None,
            S if has_smooth_penalty else None,
        )
        if type(penalty) in (GroupLasso, GroupElasticNet):
            radial_penalty = cast(GroupLasso | GroupElasticNet, penalty)
            L_groups, radial_eigensystems = _radial_block_eigensystems(
                block_hessians,
                groups,
                radial_penalty,
            )
        elif type(penalty) is Ridge:
            radial_penalty = None
            # The exact Ridge branch does not use a scalar-gradient step.  A
            # cheap positive upper bound is sufficient for its later KKT map.
            L_groups = [float(np.linalg.norm(hessian, ord=np.inf)) for hessian in block_hessians]
            radial_eigensystems = None
        else:
            radial_penalty = None
            L_groups = [_block_lipschitz(hessian) for hessian in block_hessians]
            radial_eigensystems = None
        ridge_factors = (
            _ridge_block_factors(block_hessians, groups, penalty)
            if type(penalty) is Ridge
            else None
        )
        t_lipschitz_total += time.perf_counter() - t0

        # Initialize residual
        r = z - dm.matvec(beta) - intercept - offset
        if has_smooth_penalty:
            assert S is not None
            S_beta = S @ beta
        else:
            S_beta = None

        # Active set: track which groups can be skipped.
        # A group is inactive if beta_g == 0 AND ||grad_g|| < lambda1 * w_g
        # (KKT optimality for zeroed group).  First inner iter is always
        # a full sweep; subsequent iters skip inactive groups.
        group_active = [True] * n_groups

        # Inner loop: proximal Newton block coordinate descent
        t_inner_start = time.perf_counter()
        for inner in range(max_iter_inner):
            # Periodic residual refresh to avoid float drift
            if inner > 0 and inner % 5 == 0:
                r = z - dm.matvec(beta) - intercept - offset
                if has_smooth_penalty:
                    assert S is not None
                    S_beta = S @ beta

            beta_before = beta.copy()

            # Update intercept (closed form, unpenalised)
            delta_int: float = float(np.sum(W * r) / np.sum(W))
            intercept += delta_int
            r -= delta_int

            # BCD cycle over groups.  Generic penalties use a mathematically
            # coherent Euclidean proximal-gradient step.  Ridge uses the exact
            # conditional quadratic solve in the same dispatch.
            for gi, (gm, g, L_g) in enumerate(zip(gms, groups, L_groups, strict=True)):
                # Active set: skip groups confirmed inactive on previous sweep
                if active_set and inner > 0 and not group_active[gi]:
                    total_groups_skipped += 1
                    continue

                bg_old = beta[g.sl].copy()
                step_curvature = L_g if L_g > 0.0 else 1.0

                grad_g = -gm.rmatvec(W * r)
                if S_beta is not None:
                    grad_g = grad_g + S_beta[g.sl]

                if ridge_factors is not None:
                    ridge_gradient = grad_g.copy()
                    if penalty_targets_group(penalty, g):
                        ridge_gradient += float(penalty.lambda1 or 0.0) * bg_old
                    bg_new = bg_old - _solve_factored_block(
                        ridge_factors[gi],
                        ridge_gradient,
                    )
                elif radial_eigensystems is not None:
                    assert radial_penalty is not None
                    lam = float(radial_penalty.lambda1 or 0.0)
                    if not penalty_targets_group(radial_penalty, g):
                        threshold = 0.0
                    elif type(radial_penalty) is GroupLasso:
                        threshold = lam * g.weight
                    else:
                        elastic_net = cast(GroupElasticNet, radial_penalty)
                        threshold = lam * elastic_net.alpha * g.weight
                    block_rhs = block_hessians[gi] @ bg_old - grad_g
                    bg_new = _solve_radial_block(
                        radial_eigensystems[gi],
                        block_rhs,
                        threshold,
                    )
                else:
                    step_g = 1.0 / step_curvature
                    bg_cand = bg_old - grad_g / step_curvature
                    bg_new = penalty.prox_group(bg_cand, g, step_g)

                d = bg_new - bg_old
                if np.any(d != 0):
                    r -= gm.matvec(d)
                    beta[g.sl] = bg_new
                    if S_beta is not None:
                        assert S is not None
                        S_beta += S[:, g.sl] @ d

                # Active set: check KKT for zeroed groups after the update
                if active_set:
                    if not np.any(bg_new != 0.0):
                        # A zero block is inactive exactly when it is a fixed
                        # point of the penalty's own proximal operator.  This
                        # works for group, sparse-group, elastic-net, and custom
                        # penalties without duplicating their subgradients.
                        grad_after = -gm.rmatvec(W * r)
                        if S_beta is not None:
                            grad_after = grad_after + S_beta[g.sl]
                        zero_probe = penalty.prox_group(
                            -grad_after / step_curvature,
                            g,
                            1.0 / step_curvature,
                        )
                        group_active[gi] = bool(np.any(zero_probe != 0.0))
                    else:
                        group_active[gi] = True

            # Check inner convergence
            change: float = float(np.max(np.abs(beta - beta_before)))
            if change < tol * 0.01:
                break

        inner_iters = inner + 1
        total_inner_iters += inner_iters
        t_inner_total += time.perf_counter() - t_inner_start

        proposal = evaluate_state(
            beta,
            intercept,
            phase="proposal",
            outer_iteration=outer + 1,
            alpha=1.0,
        )
        trial_cache: dict[float, _IRLSState] = {1.0: proposal}
        trial_directions: tuple[NDArray, float, NDArray] | None = None

        def evaluate_trial(alpha: float) -> _IRLSState:
            nonlocal trial_directions
            if trial_directions is None:
                trial_directions = (
                    proposal.beta - committed.beta,
                    proposal.intercept - committed.intercept,
                    proposal.eta_unclipped - committed.eta_unclipped,
                )
            beta_direction, intercept_direction, eta_direction = trial_directions
            beta_trial = committed.beta + alpha * beta_direction
            intercept_trial = committed.intercept + alpha * intercept_direction
            eta_trial = committed.eta_unclipped + alpha * eta_direction
            candidate = evaluate_state(
                beta_trial,
                intercept_trial,
                phase="line_search_trial",
                outer_iteration=outer + 1,
                alpha=alpha,
                eta_unclipped=eta_trial,
            )
            trial_cache[alpha] = candidate
            return candidate

        decision = _select_irls_trial(
            committed=committed,
            proposal=proposal,
            evaluate_state=evaluate_trial,
            max_halving=max_halving,
            extended_max_halving=lambda: _poisson_sqrt_halving_budget(
                committed=committed,
                proposal=proposal,
                y=y,
                weights=weights,
                family=family,
                link=link,
                default=max_halving,
            ),
            merit_scale=objective_merit_scale,
            merit_delta=lambda candidate, base: _stable_penalized_deviance_delta(
                candidate,
                base,
                S,
                nonsmooth_penalty=nonsmooth_merit,
            ),
        )
        retained = committed if decision.step_rejected else trial_cache[decision.alpha]
        beta = retained.beta.copy()
        intercept = retained.intercept
        eta_new_unclipped = retained.eta_unclipped
        eta_new = retained.eta
        mu_new = retained.mu
        dev = retained.deviance
        objective = retained.penalized_deviance
        if objective is None:  # pragma: no cover - PIRLS always attaches it
            objective = dev
        n_halvings = decision.step_halvings
        step_rejected = decision.step_rejected
        objective_relative_change = _irls_objective_relative_change(
            objective=objective,
            previous=objective_prev,
            objective_scale=objective_merit_scale,
        )

        convergence_criterion = convergence
        kkt_violation: float | None = None
        if step_rejected or not np.isfinite(dev):
            convergence_value = float("inf")
            iteration_converged = False
        elif convergence == "coefficients":
            coef_change = float(np.max(np.abs(beta - beta_prev) / np.maximum(1.0, np.abs(beta))))
            convergence_value = max(
                coef_change,
                abs(intercept - intercept_prev) / max(1.0, abs(intercept)),
            )
            iteration_converged = convergence_value < tol
        else:
            convergence_value = objective_relative_change
            iteration_converged = convergence_value < tol

        # Objective or coefficient stagnation is necessary but not sufficient
        # for a composite fit.  Before accepting convergence, enforce the
        # proximal fixed-point form of the KKT equations on the retained state.
        if iteration_converged:
            kkt_violation = _composite_kkt_violation(
                dm=dm,
                state=retained,
                y=y,
                weights=weights,
                family=family,
                link=link,
                offset=offset,
                groups=groups,
                penalty=penalty,
                S=S,
                has_smooth_penalty=has_smooth_penalty,
                L_groups=L_groups,
                curvature_weights=W,
                tol=tol,
            )
            convergence_value = max(convergence_value, kkt_violation)
            iteration_converged = convergence_value < tol

        if step_rejected:
            termination_reason = "step_rejected"
        elif not np.isfinite(dev):
            termination_reason = "nonfinite_deviance"
        elif iteration_converged:
            termination_reason = "converged"
        elif outer + 1 == max_iter_outer:
            termination_reason = "max_iter"
        else:
            termination_reason = "continue"

        if trace_enabled:
            assert trace_run is not None
            trace_run.emit_lazy(
                "step_decision",
                lambda: {
                    "solver": "pirls",
                    "outer_iteration": outer + 1,
                    "base_state_id": committed.state_id,
                    "proposal_state_id": proposal.state_id,
                    "committed_state_id": retained.state_id,
                    "accepted_alpha": decision.alpha,
                    "step_halvings": decision.step_halvings,
                    "trials_attempted": decision.trials_attempted,
                    "step_rejected": decision.step_rejected,
                    "fit_converged": iteration_converged,
                    "convergence_criterion": convergence_criterion,
                    "convergence_value": convergence_value,
                    "kkt_violation": kkt_violation,
                    "convergence_tolerance": tol,
                    "termination_reason": termination_reason,
                },
                channel="pirls",
                purpose=trace_purpose,
            )
        emit_state_commit(
            retained,
            outer_iteration=outer + 1,
            fit_converged=iteration_converged,
            convergence_criterion=convergence_criterion,
            convergence_value=convergence_value,
            termination_reason=termination_reason,
        )
        if n_halvings:
            logger.info(
                "  PIRLS outer=%d: accepted step fraction %.5g after %d halvings, dev=%.2e",
                outer + 1,
                decision.alpha,
                n_halvings,
                dev,
            )

        # Warn on extreme working weight range (helps diagnose bad data)
        positive_w_min, positive_w_max, w_ratio = _positive_working_weight_stats(W)
        if w_ratio > 1e12:
            logger.debug(
                f"PIRLS outer={outer + 1}: extreme W ratio {w_ratio:.1e} "
                f"(positive W range [{positive_w_min:.2e}, {positive_w_max:.2e}])"
            )

        # Record per-iteration diagnostics
        if record_diagnostics:
            top_idx, bot_idx = _extreme_weight_indices(W)
            # Each extremum below is reported under two field names and, for
            # the eta pairs, also decides the clipping flag.  Bind once: these
            # are O(n) passes over arrays nothing here mutates.
            w_min = float(W.min())
            w_max = float(W.max())
            eta_min = float(np.min(eta_new))
            eta_max = float(np.max(eta_new))
            eta_min_unclipped = float(np.min(eta_new_unclipped))
            eta_max_unclipped = float(np.max(eta_new_unclipped))
            working_eta_min = float(np.min(eta))
            working_eta_max = float(np.max(eta))
            working_eta_min_unclipped = float(np.min(eta_unclipped))
            working_eta_max_unclipped = float(np.max(eta_unclipped))
            working_eta_clipped = bool(
                working_eta_min_unclipped < working_eta_min
                or working_eta_max_unclipped > working_eta_max
            )
            eta_clipped = bool(eta_min_unclipped < eta_min or eta_max_unclipped > eta_max)
            iteration_log.append(
                IterationDiagnostics(
                    iteration=outer + 1,
                    deviance=dev,
                    w_min=w_min,
                    w_max=w_max,
                    w_ratio=w_ratio,
                    mu_min=float(mu_new.min()),
                    mu_max=float(mu_new.max()),
                    eta_min=eta_min,
                    eta_max=eta_max,
                    intercept=intercept,
                    step_halvings=n_halvings,
                    top_w_indices=top_idx,
                    bottom_w_indices=bot_idx,
                    raw_w_min=w_min,
                    raw_w_max=w_max,
                    raw_w_ratio=w_ratio,
                    eta_min_unclipped=eta_min_unclipped,
                    eta_max_unclipped=eta_max_unclipped,
                    eta_clipped=eta_clipped,
                    working_mu_min=float(mu.min()),
                    working_mu_max=float(mu.max()),
                    working_eta_min=working_eta_min,
                    working_eta_max=working_eta_max,
                    working_eta_min_unclipped=working_eta_min_unclipped,
                    working_eta_max_unclipped=working_eta_max_unclipped,
                    working_eta_clipped=working_eta_clipped,
                    step_rejected=step_rejected,
                    trials_attempted=decision.trials_attempted,
                    accepted_alpha=decision.alpha,
                    base_state_id=committed.state_id,
                    proposal_state_id=proposal.state_id,
                    committed_state_id=retained.state_id,
                    evaluation_id=retained.evaluation_id,
                    state_space=retained.state_space,
                    basis_id=retained.basis_id,
                    convergence_criterion=convergence_criterion,
                    convergence_value=convergence_value,
                    convergence_tolerance=tol,
                    termination_reason=termination_reason,
                )
            )

        t_outer_elapsed = time.perf_counter() - t_outer_start
        logger.info(
            f"  outer={outer + 1:3d}  bcd_cycles={inner_iters:4d}  "
            f"dev={dev:12.1f}  "
            f"pdev_delta={objective_relative_change:10.2e}  "
            f"time={t_outer_elapsed:.3f}s"
        )

        if step_rejected:
            logger.warning(
                "PIRLS rejected all trial steps at outer=%d; restored committed state "
                "(committed dev=%.6g, proposal dev=%.6g, committed pdev=%.6g, "
                "proposal pdev=%.6g, trials=%s)",
                outer + 1,
                committed.deviance,
                proposal.deviance,
                committed.penalized_deviance,
                proposal.penalized_deviance,
                {
                    alpha: (state.deviance, state.penalized_deviance)
                    for alpha, state in trial_cache.items()
                },
            )
            break

        if not np.isfinite(dev):
            logger.warning(f"PIRLS non-finite deviance at outer={outer + 1}: dev={dev:.2e}")
            break

        if iteration_converged:
            converged = True
            break
        committed = retained
        objective_prev = objective

    t_elapsed = time.perf_counter() - t_total
    logger.info(
        f"  PIRLS done: {outer + 1} outer iters, {total_inner_iters} total BCD cycles, "
        f"{t_elapsed:.2f}s total"
    )

    # Runtime separation backstop (issue #341), mirroring irls_direct: a
    # retained predictor pinned at the link's overflow guard, or an exhausted
    # budget with a frozen objective, both under an exploded working-weight
    # range, mark a coefficient drifting to +/-infinity.  Reachable here only
    # for groups the selection penalty does not bound (penalty_features
    # restrictions); cell separation in those groups is already named at
    # build time, so this catches what the design scan cannot see.
    if separation != "ignore" and np.isfinite(dev):
        from superglm.diagnostics.separation import (
            EXTREME_WEIGHT_RATIO,
            STAGNANT_DEVIANCE_DELTA,
            SeparationError,
            SeparationWarning,
            format_runtime_message,
        )

        if w_ratio > EXTREME_WEIGHT_RATIO:
            pinned = bool(np.any((retained.eta != retained.eta_unclipped) & (weights > 0)))
            exhausted_stagnant = (
                not converged
                # Only a real budget makes exhaustion-with-stagnation
                # evidence of a drift; warm-started micro-budget solves
                # exhaust theirs by construction (see irls_direct).
                and max_iter_outer >= 10
                and outer + 1 >= max_iter_outer
                and not step_rejected
                and objective_relative_change is not None
                and objective_relative_change < STAGNANT_DEVIANCE_DELTA
            )
            if pinned or exhausted_stagnant:
                max_abs = float(np.max(np.abs(beta))) if beta.size else 0.0
                drifting = [
                    g.name
                    for g in groups
                    if beta[g.sl].size and float(np.max(np.abs(beta[g.sl]))) >= max_abs - 2.0
                ][:5]
                message = format_runtime_message(w_ratio, outer + 1, drifting, pinned)
                if separation == "error":
                    raise SeparationError(message)
                warnings.warn(message, SeparationWarning, stacklevel=2)
    extra = ""
    if active_set:
        total_group_updates = total_inner_iters * n_groups
        extra = f"  groups_skipped={total_groups_skipped}/{total_group_updates}"
    logger.info(
        f"  Breakdown: group_lipschitz={t_lipschitz_total:.2f}s  bcd_cycles={t_inner_total:.2f}s"
        + extra
    )

    # Selection is derived from the final retained coefficients.  A proposal
    # may have been step-halved or rejected, so its proximal state is not an
    # accepted source of rank and inference metadata.
    group_selected = [
        not can_zero_groups
        or not penalty_targets_group(penalty, group)
        or bool(np.any(beta[group.sl] != 0.0))
        for group in groups
    ]
    selected_columns_list: list[int] = []
    selected_groups: list[GroupSlice] = []
    selected_original_groups: list[GroupSlice] = []
    selected_gms: list[GroupMatrix] = []
    selected_group_names: list[str] = []
    selected_offset = 0
    for is_selected, gm, group in zip(group_selected, gms, groups, strict=True):
        if not is_selected:
            continue
        selected_columns_list.extend(range(group.start, group.end))
        selected_group_names.append(group.name)
        selected_gms.append(gm)
        selected_original_groups.append(group)
        selected_groups.append(
            GroupSlice(
                name=group.name,
                start=selected_offset,
                end=selected_offset + group.size,
                weight=group.weight,
                penalty_dim=group.penalty_dim,
                penalized=group.penalized,
                feature_name=group.feature_name,
                subgroup_type=group.subgroup_type,
            )
        )
        selected_offset += group.size

    selected_columns = np.asarray(selected_columns_list, dtype=int)
    selected_dm = DesignMatrix(selected_gms, n=n, p=len(selected_columns))
    if has_smooth_penalty:
        assert S is not None
        selected_penalty = S[np.ix_(selected_columns, selected_columns)]
    else:
        selected_penalty = np.zeros((len(selected_columns), len(selected_columns)))
    _add_selection_local_curvature(
        curvature=selected_penalty,
        penalty=penalty,
        beta=beta,
        original_groups=selected_original_groups,
        active_groups=selected_groups,
    )
    has_inference_curvature = bool(np.any(selected_penalty))

    final_working_rows = coefficient_working_rows(
        distribution=family,
        link=link,
        y=y,
        mu=mu_new,
        eta=eta_new,
        sample_weight=weights,
        prefer_observed=False,
    )
    W_final = final_working_rows.weights
    z_final = final_working_rows.response
    centered = build_centered_system(
        dm=selected_dm,
        W=W_final,
        z_off=z_final - offset,
        penalty=selected_penalty,
    )
    data_rank = decompose_gram_if_authoritative(centered.data_gram)
    if data_rank is None:
        certified = decompose_factor(
            grouped_weighted_factor(
                selected_dm,
                W_final,
                center=centered.mean_x,
            )
        )
        data_rank = certified
    # Reusing ``data_rank`` cannot need certifying: it is either a factor
    # certificate or a Gram the predicate already accepted.
    augmented_rank = (
        data_rank
        if not np.any(selected_penalty)
        else decompose_gram_if_authoritative(centered.hessian)
    )
    if augmented_rank is None:
        certified = decompose_factor(
            grouped_augmented_factor(
                selected_dm,
                W_final,
                selected_penalty,
                center=centered.mean_x,
            )
        )
        augmented_rank = certified
    raw_gram, _, _, _ = centered.raw_weighted_moments()
    coefficient_rank = decompose_gram_if_authoritative(raw_gram + selected_penalty)
    if coefficient_rank is None:
        certified = decompose_factor(
            grouped_augmented_factor(selected_dm, W_final, selected_penalty)
        )
        coefficient_rank = certified
    feature_edf = np.zeros(p)
    group_edf = {group.name: 0.0 for group in groups}

    if has_inference_curvature:
        selected_edf = np.diag(augmented_rank.pseudo_inverse() @ centered.data_gram).copy()
        selected_edf[np.abs(selected_edf) < 100.0 * np.finfo(float).eps] = 0.0
        feature_edf[selected_columns] = selected_edf
        for selected_group, original_group in zip(
            selected_groups,
            selected_original_groups,
            strict=True,
        ):
            group_edf[original_group.name] = float(np.sum(selected_edf[selected_group.sl]))
    else:
        # Preserve Breheny-Huang (2009) group-lasso EDF allocation.
        #
        # The ``p_g`` of ``p_g - (p_g - 1) * shrink`` is the same group
        # dimension the ``sqrt(p_g)`` weight is built from, and both read
        # ``GroupSlice.penalty_size`` so they cannot disagree.  Whether that
        # dimension is the group's identifiable rank (the emitted width;
        # ``group_pricing="rank"``, the default) or the width the term spans
        # (``"spanned"``, the historical behaviour) is decided once at build
        # time by ``dm_builder._priced_group_dimension``; under ``"rank"``
        # the ``penalty_dim`` stamp is absent and ``penalty_size`` is simply
        # the emitted width.  The per-column allocation on the last line
        # divides by the EMITTED width on purpose: ``effective_df`` is a sum
        # over emitted columns, so only that denominator makes the columns
        # add back up to ``df_group``.
        lam = penalty.lambda1 if penalty.lambda1 is not None else 0.0
        for is_selected, group in zip(group_selected, groups, strict=True):
            if not is_selected:
                continue
            norm_g = float(np.linalg.norm(beta[group.sl]))
            priced = group.penalty_size
            if not penalty_targets_group(penalty, group) or not can_zero_groups:
                df_group = float(priced)
            else:
                shrink = min(1.0, lam * group.weight / max(norm_g, 1e-300))
                df_group = float(priced - (priced - 1) * shrink)
            group_edf[group.name] = df_group
            feature_edf[group.sl] = df_group / group.size

    mean_x = np.zeros(p)
    mean_x[selected_columns] = centered.mean_x
    selected_columns.setflags(write=False)
    mean_x.setflags(write=False)
    feature_edf.setflags(write=False)
    rank_info = RankInfo(
        policy_version=SHARED_RANK_POLICY.version,
        coordinate_space="solver",
        selected_columns=selected_columns,
        selected_group_names=tuple(selected_group_names),
        sum_w=centered.sum_w,
        mean_x=mean_x,
        intercept_edf=1.0,
        data=data_rank,
        augmented=augmented_rank,
        coefficient=coefficient_rank,
        feature_edf=feature_edf,
        group_edf=group_edf,
        objective_loss=None,
    )
    p_eff = rank_info.total_edf

    # The Pearson numerator has the same form under either weight contract --
    # both scale a row's squared residual by its weight -- so only the
    # denominator's likelihood size distinguishes them.
    pearson_sum = pearson_chi2(distribution=family, y=y, mu=mu_new, sample_weight=weights)
    df_resid = pearson_residual_degrees_of_freedom(
        weights,
        p_eff,
        weight_semantics=weight_semantics,
    )
    phi = pearson_sum / df_resid

    return PIRLSResult(
        beta=beta,
        intercept=intercept,
        n_iter=outer + 1,
        deviance=dev,
        converged=converged,
        phi=phi,
        effective_df=p_eff,
        iteration_log=iteration_log if record_diagnostics else None,
        rank_info=rank_info,
        state_id=retained.state_id,
        evaluation_id=retained.evaluation_id,
        state_space=retained.state_space,
        basis_id=retained.basis_id,
        termination_reason=termination_reason,
    )


def _wrap_dense_X(X: NDArray, groups: list[GroupSlice]) -> DesignMatrix:
    """Wrap a dense NDArray into a DesignMatrix for backward compatibility."""
    n, p = X.shape
    gms: list[GroupMatrix] = [DenseGroupMatrix(X[:, g.sl]) for g in groups]
    return DesignMatrix(gms, n, p)


def fit_pirls(
    X: NDArray | DesignMatrix,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    groups: list[GroupSlice],
    penalty: Penalty,
    offset: NDArray | None = None,
    beta_init: NDArray | None = None,
    intercept_init: float | None = None,
    max_iter_outer: int = 100,
    max_iter_inner: int = 5,
    tol: float = 1e-6,
    active_set: bool = False,
    lambda2: float | dict[str, float] = 0.0,
    record_diagnostics: bool = False,
    convergence: str = "deviance",
    S_override: NDArray | None = None,
    trace_run: TraceRun | None = None,
    separation: str = "warn",
    *,
    weight_semantics: str,
) -> PIRLSResult:
    """Fit a penalised GLM via PIRLS with proximal Newton BCD.

    If the penalty has a flavor (e.g. Adaptive), a two-stage fit is performed:
    1. Fit with uniform weights → beta_init
    2. Flavor adjusts group weights based on beta_init
    3. Refit with adjusted weights (warm started from stage 1)
    """
    if max_iter_outer < 1:
        raise ValueError(f"max_iter_outer must be at least 1, got {max_iter_outer}")
    if max_iter_inner < 1:
        raise ValueError(f"max_iter_inner must be at least 1, got {max_iter_inner}")
    if isinstance(X, DesignMatrix):
        dm = X
        n = dm.n
    else:
        dm = _wrap_dense_X(X, groups)
        n = X.shape[0]

    if offset is None:
        offset = np.zeros(n)

    trace_basis_id = (
        trace_run.next_basis_id() if trace_run is not None and trace_run.enabled else None
    )

    # Stage 1: initial fit
    result = _fit_pirls_inner(
        dm,
        y,
        weights,
        family,
        link,
        groups,
        penalty,
        offset,
        beta_init,
        intercept_init,
        max_iter_outer,
        max_iter_inner,
        tol,
        active_set,
        lambda2=lambda2,
        record_diagnostics=record_diagnostics,
        convergence=convergence,
        S_override=S_override,
        trace_run=trace_run,
        trace_basis_id=trace_basis_id,
        trace_purpose="initial_flavor_fit" if penalty.flavor is not None else "fit",
        # A flavored stage-1 fit is discarded scaffolding; only the final
        # returned fit carries the separation backstop.
        separation="ignore" if penalty.flavor is not None else separation,
        weight_semantics=weight_semantics,
    )

    # Stage 2: if flavor, adjust weights and refit (warm-start both beta and intercept)
    if penalty.flavor is not None:
        adjusted_groups = penalty.flavor.adjust_weights(
            groups, result.beta, group_matrices=list(dm.group_matrices)
        )
        result = _fit_pirls_inner(
            dm,
            y,
            weights,
            family,
            link,
            adjusted_groups,
            penalty,
            offset,
            beta_init=result.beta,
            intercept_init=result.intercept,
            max_iter_outer=max_iter_outer,
            max_iter_inner=max_iter_inner,
            tol=tol,
            active_set=active_set,
            lambda2=lambda2,
            record_diagnostics=record_diagnostics,
            convergence=convergence,
            S_override=S_override,
            trace_run=trace_run,
            trace_basis_id=trace_basis_id,
            trace_purpose="adjusted_flavor_fit",
            separation=separation,
            weight_semantics=weight_semantics,
        )

    return result
