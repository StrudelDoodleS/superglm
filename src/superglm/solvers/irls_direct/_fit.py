"""The direct IRLS fit: ``fit_irls_direct``, its method object ``_IRLSDirectFit`` and their helpers.

The package's ``__init__`` describes the solver and re-exports the names other modules use.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

import superglm.solvers.scop_exact_support as scop_exact_support
from superglm import _held_warnings as held_warnings
from superglm._fit_trace import TraceRun
from superglm._group_matrix._group_matrix_centered import _raw_centering_well_scaled
from superglm._group_matrix._group_matrix_tabmat import (
    _defer_raw_spline_tabmat_plan,
    _is_raw_spline_tabmat_centering_candidate,
)
from superglm._roundoff import EPS as _EPS_FLOAT
from superglm._roundoff import UNIT_ROUNDOFF as _UNIT_ROUNDOFF_FLOAT
from superglm._roundoff import gamma as _gamma
from superglm._utils import weighted_row_sum
from superglm.distributions import (
    Binomial,
    Distribution,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
    clip_mu,
)
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    GroupMatrix,
    SparseSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm.links import IdentityLink, Link, LogitLink, LogLink, stabilize_eta
from superglm.solvers.centered_system import (
    CenteredSystem,
    TabmatCenteringState,
    _FisherDataReuse,
    _InitialDataReuse,
    build_centered_system,
    grouped_augmented_factor,
    grouped_augmented_factor_rhs,
    grouped_projected_factor,
    grouped_structural_nulls,
    grouped_weighted_factor,
    refresh_centered_rhs,
)
from superglm.solvers.constrained_qp import (
    QPResult,
    _feasibility_slack,
    _is_feasible,
    solve_constrained_qp,
)
from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom
from superglm.solvers.irls_direct._centred import _system_offset_mean
from superglm.solvers.irls_direct._qr import _IRLSDirectQRStep
from superglm.solvers.irls_direct._recorder import _IRLSDirectRecorder
from superglm.solvers.irls_direct._structured import _IRLSDirectStructuredStep
from superglm.solvers.irls_state import (
    _evaluate_irls_state,
    _immutable_array,
    _irls_objective_relative_change,
    _irls_objective_scale,
    _IRLSState,
    _IRLSStepDecision,
    _mean_space_halving_budget,
    _poisson_sqrt_halving_budget,
    _select_irls_trial,
    _stable_penalized_deviance_delta,
    _state_is_finite,
    interior_start_intercept,
    link_domain_violation,
    mean_space_boundary_rows,
    mean_space_clipped_rows,
    mean_space_first_halving,
    mean_space_log_likelihood_rows,
    mean_space_newton_rows,
    mean_space_score_rows,
    mean_space_violation,
)
from superglm.solvers.mode_score import (
    MODE_CERTIFICATION_BAR,
    MODE_RESOLVE_CAP,
    ModeResidual,
    centred_data_score,
    centred_intercept_remainder,
    centred_matvec,
    null_basis_angle,
    penalized_mode_residual,
    prior_weighted_centre,
    rate_exhausts_budget,
    row_set_quadratics,
    row_set_residual,
    row_sets,
    stagnation_window,
    truncated_direction_ratio,
    weighted_column_centring,
)
from superglm.solvers.pirls import (
    PIRLSResult,
    REMLGeometrySummary,
    TerminationReason,
    _positive_working_weight_stats,
)
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    RankInfo,
    decide_data_rank,
    decompose_factor,
    decompose_gram_if_authoritative,
    decompose_symmetric,
    weights_fingerprint,
)
from superglm.solvers.scop_newton import (
    SCOPGroupSpec,
    SCOPJointNewton,
    _positive_quadratic_roundoff,
)
from superglm.solvers.structured import (
    BlockSymmetricOperator,
    NestedDataOperator,
    SumToZeroBlockOperator,
    resolve_structured_backend,
)
from superglm.solvers.working_rows import (
    coefficient_initial_intercept,
    coefficient_working_rows,
    concave_observed_rows,
    fisher_working_weights,
    pearson_chi2,
    supports_observed_newton,
)
from superglm.types import GroupSlice, LinearConstraintSet, PenaltyComponent

if TYPE_CHECKING:
    from superglm.solvers.irls_direct._step import _IRLSDirectStep

# the package's logger: its records keep the name they had when the solver was one module
logger = logging.getLogger(__package__)

_QP_FEASIBILITY_TOL = 1e-12

# The rung refusals a SCOP REML run carries from one coefficient fit to the next.
_CARRIED_REFUSALS = ("raw_moment_eligible", "tensor_raw_eligible")


@dataclass
class SCOPRunCentring:
    """Centring state one SCOP REML run carries across its coefficient fits.

    **Owner:** the run's fit context (``reml/scop_efs.py``,
    ``_SCOPREMLFitContext``), shared by the certification-retry contexts
    derived from it.  **Lifetime:** the run; it dies with the context.  It holds

    - the SCOP step's reduced design (the non-SCOP groups), which
      owns its compact-support caches (pattern plan, solver supports,
      execution plan).  These depend on the basis only;
    - each row-level SCOP group's exact support: its distinct rows and row
      index, or ``None`` where compressing does not pay
      (``scop_newton._exact_support``) or, for a fit's only SCOP group,
      where every row is distinct (``_lone_scop_support``).  It depends on
      the group's rows only, so it is found once per run, not once per fit;
    - the refusals (``_CARRIED_REFUSALS``) of the raw-moment and tensor raw
      rungs, on the full design and on the reduced one.

    **Invalidation:** a fit whose design, group matrices, groups, family or
    link differ by identity from the bound ones clears everything.  Weights
    change every fit and are not keyed: a carried refusal skips a rung, so a
    later fit whose weights would have passed it stays on the centre-first
    route every rejection already selects, at least as accurate: the
    anchor-centred compact supports, or, for a design they decline (a group
    with no compact support, such as a non-discretized spline, or an
    oversized one), the centred row chunks.  The raw-moment refusal is
    carried either way, as the Newton engine's ``raw_moment_policy`` carries
    it (``reml/direct.py``); only the tensor refusal waits for the anchor
    route to serve (``packed_centered_gram_rhs``), because retrying the
    tensor rungs is cheaper than the chunked pass a decline costs.  Where
    every carried refusal would recur the run is bitwise identical to one
    without the carry: measured on the freMTPL2 fits once certification
    retries start warm, and on non-discretized Poisson, Gamma and Tweedie
    SCOP fits, whose raw-moment rung passed at none of 20 to 42 per-fit
    attempts, so the carry saved those passes and changed no bit.  A cold
    retry restarts where the certificate can pass again, and then moves to
    the centre-first route, at the rounding level.
    Penalty target and precision never enter: a centring route reads the
    float64 data Gram only.
    """

    owners: tuple = ()
    reduced_key: tuple = ()
    reduced_dm: DesignMatrix | None = None
    exact_supports: dict = field(default_factory=dict)
    full_refusals: TabmatCenteringState = field(default_factory=TabmatCenteringState)
    reduced_refusals: TabmatCenteringState = field(default_factory=TabmatCenteringState)

    def bind(self, owners: tuple) -> None:
        """Keep the run's state for ``owners`` (compared by identity); any change clears it."""
        if len(owners) == len(self.owners) and all(
            current is bound for current, bound in zip(owners, self.owners, strict=True)
        ):
            return
        self.owners = owners
        self.reduced_key = ()
        self.reduced_dm = None
        self.exact_supports = {}
        self.full_refusals = TabmatCenteringState()
        self.reduced_refusals = TabmatCenteringState()

    def reduced_design(self, key: tuple[int, ...], matrices: list, n: int, p: int) -> DesignMatrix:
        """The reduced design over the groups ``key``, built once per bound run."""
        if self.reduced_dm is None or self.reduced_key != key:
            self.reduced_dm = DesignMatrix(matrices, n=n, p=p)
            self.reduced_key = key
            self.reduced_refusals = TabmatCenteringState()
        return self.reduced_dm


def _seed_refusals(state: TabmatCenteringState, carried: TabmatCenteringState) -> None:
    """Start a fit's centring state from the run's refusals."""
    for name in _CARRIED_REFUSALS:
        if getattr(carried, name) is False:
            setattr(state, name, False)


def _carry_refusals(carried: TabmatCenteringState, state: TabmatCenteringState) -> None:
    """Record a fit's refusals in the run's state."""
    for name in _CARRIED_REFUSALS:
        if getattr(state, name) is False:
            setattr(carried, name, False)


def _solve_constrained_qp_with_cold_retry(
    H: NDArray,
    g: NDArray,
    A: NDArray,
    b: NDArray,
    active_set_init: list[int] | None,
    *,
    degeneracy_retry: bool = True,
) -> QPResult:
    """Retry a failed warm active set without weakening the KKT contract.

    A warm active set is an acceleration hint, not part of the mathematical
    problem. Near an active boundary, platform-dependent roundoff can make the
    warm solve project a stationary candidate and correctly withhold its KKT
    certificate even though the identical problem converges from a cold active
    set. Retry only that failed warm case, and replace its best iterate only
    when the ordinary cold solve returns a complete certificate.

    If the ordinary solves end uncertified, one more cold solve runs with the
    degenerate-vertex safeguards (issue #472). That covers a failed warm
    solve followed by a failed cold one. It also covers a single failed cold
    solve, made when ``active_set_init`` is ``None`` or empty. The
    safeguards are for a flat monotone term at zero with more active rows
    than coefficients. The safeguarded result replaces the first one only if
    it passes the same complete certificate, so a solve that certifies
    without it is unchanged.

    ``degeneracy_retry=False`` skips that attempt. ``fit_irls_direct`` passes
    it while the previous solve in the fit was uncertified and re-arms it
    after any certified solve: a QP that keeps failing pays for the attempt
    once per run of failures, not once per IRLS iteration. With the default,
    an uncertified result means the attempt ran and failed.
    """
    result = solve_constrained_qp(
        H,
        g,
        A,
        b,
        active_set_init=active_set_init,
    )
    if active_set_init and not result.converged:
        cold_result = solve_constrained_qp(
            H,
            g,
            A,
            b,
            active_set_init=None,
        )
        if cold_result.converged:
            return cold_result
    if degeneracy_retry and not result.converged:
        safeguarded = solve_constrained_qp(
            H,
            g,
            A,
            b,
            active_set_init=None,
            _degeneracy_safeguards=True,
        )
        if safeguarded.converged:
            return safeguarded
    return result


def _lone_scop_support(gm, index: int, supports: dict | None) -> tuple[NDArray, NDArray] | None:
    """A fit's only row-level SCOP group as its distinct rows and row index; ``None`` if every row is distinct.

    With one SCOP group there are no pair tables, so it compresses whenever
    any row repeats. ``supports`` is the run's record by group
    (``SCOPRunCentring.exact_supports``): the rows never change within a
    run, so they are read and sorted once per run, not once per fit.
    """
    if supports is not None and index in supports:
        return supports[index]
    support = scop_exact_support.build_exact_scop_support(gm.toarray())
    compressed = None if support is None else (support.B_unique, support.row_to_support)
    if supports is not None:
        supports[index] = compressed
    return compressed


def _immutable_or_none(values: NDArray | None) -> NDArray | None:
    """Freeze an optional array, preserving ``None``."""
    return None if values is None else _immutable_array(values)


@dataclass(frozen=True)
class _SCOPGroupState:
    """Dynamic SCOP state committed atomically with an IRLS snapshot."""

    group_index: int
    beta_eff: NDArray
    gamma_eff: NDArray
    H_scop_penalized: NDArray | None
    last_step_norm: float
    last_fisher_fallback: bool
    discarded_directions: NDArray | None = None
    """This group's columns of the directions the last Newton step could not
    resolve. Mode certification measures stationarity on the complement."""


@dataclass(frozen=True)
class _SCOPTrialState:
    """One complete mixed ordinary/SCOP trial."""

    irls: _IRLSState
    groups: tuple[_SCOPGroupState, ...]
    level: float | None = None
    """The intercept about ``centre``, where the joint step formed it: the
    raw ``irls.intercept`` reads it at ``u |m' beta|``."""
    centre: tuple[NDArray, NDArray | None] | None = None
    """The ordinary system's centre pair ``(hi, lo)`` that ``level`` is about."""


@dataclass(frozen=True)
class _SCOPOrdinarySolve:
    """The ordinary block's solve at one iterate, which a damped trial re-solves at its latent coordinates.

    Given the latent coordinates the ordinary block is linear, so its
    working-model solve at a trial's mapped coefficients ``gamma`` is
    ``fixed - coupled (gamma - gamma_k)``, with centred level ``level -
    mean' (gamma - gamma_k)``: the linear block eliminated exactly at every
    value of the nonlinear one (variable projection, Golub & Pereyra 1973),
    not along the joint step's linearisation of ``gamma``. The part a
    trial does not owe to its latent change, ``fixed`` less the committed
    coefficients, is a scoring step and is damped along ``alpha`` with it.
    """

    columns: NDArray
    """The ordinary coefficients' positions in ``beta``."""
    committed: NDArray
    committed_level: float
    """The committed intercept profiled about ``mean_x``."""
    fixed: NDArray
    """The solve with the latent coordinates held."""
    level: float
    """Its profiled intercept about ``mean_x``."""
    coupled: NDArray
    mean: NDArray
    mean_x: NDArray
    centre: tuple[NDArray, NDArray | None]
    """The system's centre pair ``(hi, lo)``, which the levels are about."""
    centred_rows: Callable[[NDArray], NDArray] | None = None
    """``(X - 1 m') b`` about ``centre`` where the proposal formed its
    predictor so (``_raw_centering_well_scaled`` refused the raw one); a trial
    then forms its own the same way, never through its raw intercept."""


@dataclass(frozen=True)
class _CenteredFactorCertification:
    """One fit-local factor certificate tied to immutable centered geometry."""

    system: CenteredSystem
    factor: NDArray
    decomposition: RankDecomposition
    transformed_rhs: NDArray | None


def _evaluate_scop_trial(
    *,
    committed: _SCOPTrialState,
    proposed: _SCOPTrialState,
    alpha: float,
    specs: dict[int, SCOPGroupSpec],
    dm: DesignMatrix,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    offset: NDArray,
    state_id: int | None = None,
    evaluation_id: int | None = None,
    basis_id: int | None = None,
    lambdas: tuple[tuple[str, object], ...] = (),
    ordinary: _SCOPOrdinarySolve | None = None,
) -> _SCOPTrialState:
    """Evaluate a fixed-endpoint SCOP trial by interpolating latent state.

    With ``ordinary``, the trial's ordinary coefficients and intercept are
    that block's solve at the trial's mapped coefficients
    (``_SCOPOrdinarySolve``); otherwise they are interpolated as the latent
    coordinates are. The groups are in the joint step's latent order.
    """
    beta_trial = committed.irls.beta + alpha * (proposed.irls.beta - committed.irls.beta)
    intercept_trial = committed.irls.intercept + alpha * (
        proposed.irls.intercept - committed.irls.intercept
    )
    trial_groups: list[_SCOPGroupState] = []
    mapped_changes: list[NDArray] = []
    for committed_group, proposed_group in zip(committed.groups, proposed.groups, strict=True):
        if committed_group.group_index != proposed_group.group_index:
            raise ValueError("SCOP trial group ordering does not match")
        spec = specs[committed_group.group_index]
        beta_eff = committed_group.beta_eff + alpha * (
            proposed_group.beta_eff - committed_group.beta_eff
        )
        gamma_eff = spec.reparam.forward(beta_eff)
        beta_trial[spec.group.sl] = gamma_eff
        mapped_changes.append(gamma_eff - spec.reparam.forward(committed_group.beta_eff))
        trial_groups.append(
            _SCOPGroupState(
                group_index=committed_group.group_index,
                beta_eff=_immutable_array(beta_eff),
                gamma_eff=_immutable_array(gamma_eff),
                H_scop_penalized=None,
                last_step_norm=float(np.linalg.norm(beta_eff - committed_group.beta_eff)),
                last_fisher_fallback=proposed_group.last_fisher_fallback,
                discarded_directions=proposed_group.discarded_directions,
            )
        )

    # A latent step can map past the floating-point range (``exp`` of a
    # coordinate the Newton model sent far out); such a trial's objective is
    # not finite and the line search rejects it, without a warning.
    level, eta_unclipped = None, None
    with np.errstate(over="ignore", invalid="ignore"):
        if ordinary is not None:
            change = np.concatenate(mapped_changes)
            solved = (
                ordinary.committed
                + alpha * (ordinary.fixed - ordinary.committed)
                - ordinary.coupled @ change
            )
            level = (
                ordinary.committed_level
                + alpha * (ordinary.level - ordinary.committed_level)
                - float(ordinary.mean @ change)
            )
            beta_trial[ordinary.columns] = solved
            intercept_trial = level - float(ordinary.mean_x @ solved)
            if ordinary.centred_rows is not None:
                latent_columns = beta_trial.copy()
                latent_columns[ordinary.columns] = 0.0
                eta_unclipped = (
                    level + ordinary.centred_rows(solved) + dm.matvec(latent_columns) + offset
                )
        irls = _evaluate_irls_state(
            dm,
            y,
            weights,
            family,
            link,
            offset,
            beta_trial,
            intercept_trial,
            eta_unclipped=eta_unclipped,
            state_id=state_id,
            evaluation_id=evaluation_id,
            basis_id=basis_id,
            lambdas=lambdas,
        )
    return _SCOPTrialState(
        irls=irls,
        groups=tuple(trial_groups),
        level=level,
        centre=None if ordinary is None else ordinary.centre,
    )


def _level_about(
    level: float,
    centre: tuple[NDArray, NDArray | None],
    new_centre: tuple[NDArray, NDArray | None],
    beta: NDArray,
) -> float:
    """``level`` about ``new_centre``: the same predictor, ``level + (m_new - m)' beta``.

    Each centre is a pair ``(hi, lo)``; ``new_hi - hi`` is exact where the two
    lie within a factor two (Sterbenz), as successive working means of one
    column do, so the shift rounds at ``|m_new - m| |beta|``, not at ``|m' beta|``.
    """
    (hi, lo), (new_hi, new_lo) = centre, new_centre
    shift = np.asarray(new_hi, dtype=np.float64) - hi
    if lo is not None:
        shift = shift - lo
    if new_lo is not None:
        shift = shift + new_lo
    return float(level) + math.fsum(shift * beta)


_LATENT_MAX_STEP = 1000.0
"""The longest step a SCOP line search tries on any exponentiated latent coordinate.

Dennis & Schnabel's maximum step (1983, sec. 6.3): a step that would overflow
the objective or leave the region of interest is shortened to it before the
line search, which then backtracks from there with its usual budget. Here it
is the width of ``[-500, 500]``, the domain outside which
``SCOPSolverReparam.forward`` clips a coordinate: a longer step carries a
coordinate past the clip from anywhere inside it, where its mapped
coefficient no longer moves with the step, and halving through such trials
spends the budget on the clip's one value. The free coordinates, mapped by
the identity on the data's own scale, are not bounded.
"""


def _latent_step_depth(
    committed: _SCOPTrialState, proposed: _SCOPTrialState, specs: dict[int, SCOPGroupSpec]
) -> int:
    """The first halving depth whose step is within ``_LATENT_MAX_STEP`` on every exponentiated coordinate.

    ``0`` for a step within it already, and for a step that is not finite,
    which no halving can bring back and the line search refuses as it is.
    """
    largest = 0.0
    for committed_group, proposed_group in zip(committed.groups, proposed.groups, strict=True):
        free = specs[committed_group.group_index].reparam.free_dim
        change = proposed_group.beta_eff[free:] - committed_group.beta_eff[free:]
        largest = max(largest, float(np.max(np.abs(change), initial=0.0)))
    if not math.isfinite(largest):
        return 0
    depth = 0
    while math.ldexp(largest, -depth) > _LATENT_MAX_STEP:
        depth += 1
    return depth


def _feature_edf(
    centered: CenteredSystem,
    decomposition: RankDecomposition,
    factor: Callable[[int], NDArray],
) -> NDArray:
    """``diag((D + S)^+ D)``: each coefficient's effective degrees of freedom.

    ``D`` is the centred data Gram and ``S`` the penalty, ``D + S`` the
    Hessian ``decomposition`` factors.  The trace is invariant under scaling
    ``D`` and ``S`` jointly by one constant, but the pseudo-inverse is not: at
    a data scale ``s`` its entries are of order ``1 / s``, which overflows
    once ``s`` falls below ``1 / max float``, about 5.6e-309 (subnormal
    weights give ``s ~ 1e-317``).  The unscaled trace is formed first and,
    when every entry is finite, returned as it is: the unscaled computation,
    bit for bit, with no scan of the Hessian.  Otherwise the pseudo-inverse
    overflowed, and ``D`` and ``S`` are multiplied by the one power of two
    that brings the Hessian's largest entry into ``[1/2, 1)``, exact for
    normal and subnormal entries alike, and the scaled Hessian is factored
    again (``factor(exponent)`` when its Gram is not authoritative).  Both
    are positive semidefinite, so ``|S_ij| <= sqrt(S_ii S_jj) <= max diag(D
    + S)``: the scaled penalty is at most 1 and cannot overflow; a penalty
    that dominates the data leaves ``D``'s scaled entries small and the
    trace near its limit, 0.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        unscaled = np.diag(decomposition.pseudo_inverse() @ centered.data_gram).copy()
    if np.all(np.isfinite(unscaled)):
        return unscaled
    hessian = centered.hessian
    largest = float(np.max(np.abs(hessian), initial=0.0))
    if largest == 0.0 or not math.isfinite(largest):
        return unscaled
    exponent = math.frexp(largest)[1]
    scaled = decompose_gram_if_authoritative(np.ldexp(hessian, -exponent))
    if scaled is None:
        scaled = decompose_factor(factor(exponent))
    return np.diag(scaled.pseudo_inverse() @ np.ldexp(centered.data_gram, -exponent)).copy()


# The spacing of the subnormal numbers, 2^-1074, the smallest positive
# float64.  Gradual underflow adds at most half of it to one operation
# (Higham 2002, section 2.1); half is not representable (2.0**-1075 rounds to
# 0), so the bounds below count the whole spacing per operation.
_SUBNORMAL_SPACING = 2.0**-1074


def _underflow_allowance(rows: int) -> float:
    """The absolute error gradual underflow can add to a sum over ``rows`` rows and its two companions."""
    return (rows + 2) * _SUBNORMAL_SPACING


# The largest a coordinate's penalty terms may grow to in the certificate's
# units, short of float64's 2^1024 by enough for the sums that use them.
_PENALTY_HEADROOM_EXPONENT = 1000


def _even_exponent(value: float) -> int:
    """The even ``e`` with ``value * 2^-e`` in ``[1/4, 1)``; 0 for zero or non-finite ``value``."""
    if not (value > 0.0 and math.isfinite(value)):
        return 0
    exponent = math.frexp(value)[1]
    return exponent + (exponent % 2)


def _column_shift(penalty_size: NDArray, weight_exponent: int) -> NDArray:
    """Per coordinate, the even shift that keeps its penalty below ``2^1000`` in the weights' units.

    ``penalty_size`` in the penalty's own units; the shift is 0 wherever the
    penalty, divided by ``2^weight_exponent``, already fits.
    """
    size = np.asarray(penalty_size, dtype=np.float64)
    exponents = np.where(size > 0.0, np.frexp(size)[1], 0)
    excess = np.maximum(exponents - weight_exponent - _PENALTY_HEADROOM_EXPONENT, 0)
    return (excess + excess % 2).astype(np.int64)


def _centred_system_matvec(dm: DesignMatrix, beta: NDArray, system: CenteredSystem) -> NDArray:
    """``(X - 1 m') beta`` about the system's centre pair, through each block's own product.

    ``centred_matvec`` about ``hi``, less ``lo' beta`` (``lo`` is nonzero on
    dense columns only): a ``DenseGroupMatrix`` row is ``(x - hi) - lo`` before
    its product, so it rounds at ``|x - hi| |beta|`` rather than at its
    offset; every other block takes the product the native ``intercept + X
    beta`` takes, less ``m' beta``, so it rounds no worse than that route.  No
    row of the design is materialised.
    """
    centre, centre_lo = system.centre_pair()
    values = centred_matvec(dm, beta, centre)
    if centre_lo is not None:
        correction = math.fsum(np.asarray(centre_lo, dtype=np.float64) * beta)
        if correction != 0.0:
            values -= correction
    return values


def _has_constant_irls_weights(family: Distribution, link: Link) -> bool:
    """Return True when PIRLS weights are independent of ``mu``.

    The direct solver can reuse X'WX only when
    ``(dmu/deta)^2 / V(mu)`` is exactly constant.  Keep this deliberately
    conservative so performance never changes the fitted problem.
    """
    from superglm.distributions import Gamma, Gaussian, Poisson
    from superglm.links import IdentityLink, LogLink, SqrtLink

    return (
        (type(family) is Gaussian and type(link) is IdentityLink)
        or (type(family) is Gamma and type(link) is LogLink)
        or (type(family) is Poisson and type(link) is SqrtLink)
    )


def _separation_weight_ratio(
    curvature_source: str,
    w_ratio: float,
    *,
    family,
    link,
    mu: NDArray,
    eta: NDArray,
    weights: NDArray,
) -> float:
    """The working-weight ratio the separation bar was set on: expected curvature.

    Observed rows scale each Fisher weight by alpha = 1 + (y - mu)(V'/V + g''/g'),
    (2 - p) + (p - 1) y / mu for Tweedie/log, whose spread is not separation.
    """
    if curvature_source == "fisher":
        return w_ratio
    fisher = fisher_working_weights(
        distribution=family, link=link, mu=mu, eta=eta, sample_weight=weights
    )
    return _positive_working_weight_stats(fisher)[2]


def _build_penalty_matrix(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    p: int,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> NDArray:
    """Backward-compatible wrapper for the shared REML penalty builder."""
    from superglm.reml.penalty_algebra import build_penalty_matrix

    return build_penalty_matrix(
        group_matrices,
        groups,
        lambda2,
        p,
        reml_penalties=reml_penalties,
    )


def _safe_decompose_H(H: NDArray, residual_tol: float = 1e-6) -> tuple[NDArray, float, bool]:
    """Compatibility wrapper returning inverse, pseudo-logdet, and fast-path flag."""
    decomposition = decompose_symmetric(H, residual_tol=residual_tol)
    cholesky_ok = decomposition.method in {"cholesky", "pivoted_cholesky"}
    return decomposition.pseudo_inverse(), decomposition.log_pdet, cholesky_ok


def _refuse_scop_with_linear_constraints(groups: list[GroupSlice]) -> None:
    """Refuse a fit with both shape-constrained (SCOP) and linearly constrained groups.

    The SCOP step takes one Newton step on every coefficient and has no
    constrained QP, so a linear constraint beside it would be checked at the
    end but never enforced by a step.  Every fit entry point refuses the mix;
    the solver refuses it as well, so each fit has exactly one step method.
    """
    shaped = [group.name for group in groups if group.monotone_engine == "scop"]
    bounded = [group.name for group in groups if group.constraints is not None]
    if shaped and bounded:
        raise NotImplementedError(
            "fit_irls_direct cannot fit shape-constrained (SCOP) groups and linearly "
            f"constrained groups together: {', '.join(map(repr, shaped))} "
            f"{'is' if len(shaped) == 1 else 'are'} shape-constrained and "
            f"{', '.join(map(repr, bounded))} {'has' if len(bounded) == 1 else 'have'} "
            "linear constraints. Use one constraint method for every constrained term "
            "(all of them PSplines, which use SCOP, or none of them), or drop one of the "
            "constraints."
        )


def fit_irls_direct(
    X: NDArray | DesignMatrix,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    offset: NDArray | None = None,
    beta_init: NDArray | None = None,
    intercept_init: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    return_xtwx: bool = False,
    profile: dict | None = None,
    cache_out: dict | None = None,
    record_diagnostics: bool = False,
    direct_solve: str = "auto",
    convergence: str = "deviance",
    S_override: NDArray | None = None,
    reml_penalties: list[PenaltyComponent] | None = None,
    return_scop_state: bool = False,
    scop_state_init: dict[int, dict] | None = None,
    debug_recorder=None,
    debug_context: dict[str, object] | None = None,
    compute_rank_info: bool = True,
    _return_working_system: bool = False,
    _compute_fit_statistics: bool = True,
    _compute_reml_geometry: bool = True,
    _retain_reml_decomposition: bool = False,
    _use_observed_newton: bool = True,
    _working_newton: bool = False,
    _newton_latch: list[bool] | None = None,
    _deviance_init: float | None = None,
    trace_run: TraceRun | None = None,
    trace_purpose: str = "fit",
    _compute_scop_postfit_inference: bool = True,
    separation: str = "warn",
    *,
    weight_semantics: str,
    _initial_data_reuse: _InitialDataReuse | None = None,
    _raw_moment_policy: TabmatCenteringState | None = None,
    _fisher_data_reuse: _FisherDataReuse | None = None,
    _scop_run_centring: SCOPRunCentring | None = None,
    _laplace_excluded: tuple[int, ...] = (),
    _mode_bar: float | None = None,
    _compensate_centred_intercept: bool = True,
    _centred_init: tuple[float, NDArray] | None = None,
    _defer_rank_info: list[Callable[[], RankInfo]] | None = None,
) -> tuple[PIRLSResult, NDArray] | tuple[PIRLSResult, NDArray, NDArray]:
    """Fit by direct IRLS (see ``_fit_irls_direct_once``).

    A structured factor that cannot proceed raises ``StructuredSolverError``
    under every ``direct_solve`` (one-engine design §6, decision 6): no fit is
    rerun on another solver.  A Fisher-data reuse cache is cleared when the
    fit fails or ends unconverged.  Shape-constrained (SCOP) groups and
    linearly constrained groups are refused together
    (``_refuse_scop_with_linear_constraints``).
    """
    result = None
    try:
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")
        _refuse_scop_with_linear_constraints(groups)
        result = _fit_irls_direct_once(
            X=X,
            y=y,
            weights=weights,
            family=family,
            link=link,
            groups=groups,
            lambda2=lambda2,
            offset=offset,
            beta_init=beta_init,
            intercept_init=intercept_init,
            max_iter=max_iter,
            tol=tol,
            return_xtwx=return_xtwx,
            profile=profile,
            cache_out=cache_out,
            record_diagnostics=record_diagnostics,
            direct_solve=direct_solve,
            convergence=convergence,
            S_override=S_override,
            reml_penalties=reml_penalties,
            return_scop_state=return_scop_state,
            scop_state_init=scop_state_init,
            debug_recorder=debug_recorder,
            debug_context=debug_context,
            compute_rank_info=compute_rank_info,
            _return_working_system=_return_working_system,
            _compute_fit_statistics=_compute_fit_statistics,
            _compute_reml_geometry=_compute_reml_geometry,
            _retain_reml_decomposition=_retain_reml_decomposition,
            _use_observed_newton=_use_observed_newton,
            _working_newton=_working_newton,
            _newton_latch=_newton_latch,
            _deviance_init=_deviance_init,
            trace_run=trace_run,
            trace_purpose=trace_purpose,
            _compute_scop_postfit_inference=_compute_scop_postfit_inference,
            separation=separation,
            weight_semantics=weight_semantics,
            _initial_data_reuse=_initial_data_reuse,
            _raw_moment_policy=_raw_moment_policy,
            _fisher_data_reuse=_fisher_data_reuse,
            _scop_run_centring=_scop_run_centring,
            _laplace_excluded=_laplace_excluded,
            _mode_bar=_mode_bar,
            _compensate_centred_intercept=_compensate_centred_intercept,
            _centred_init=_centred_init,
            _defer_rank_info=_defer_rank_info,
        )
        return result
    finally:
        if _fisher_data_reuse is not None and (result is None or not result[0].converged):
            _fisher_data_reuse.clear()


def _certify_centered_factor(
    dm: DesignMatrix,
    _fisher_data_reuse: _FisherDataReuse | None,
    cache: list[_CenteredFactorCertification | None],
    system: CenteredSystem,
    W_current: NDArray,
    *,
    response: NDArray | None = None,
) -> _CenteredFactorCertification:
    """Return a factor certificate for one immutable centered geometry."""
    if _fisher_data_reuse is not None:
        _fisher_data_reuse.clear()
    cached = cache[0]
    same_geometry = bool(
        cached is not None
        and cached.system.data_gram is system.data_gram
        and cached.system.penalty is system.penalty
        and cached.system.hessian is system.hessian
        and cached.system.mean_x is system.mean_x
        and cached.system.sum_w == system.sum_w
    )
    # A refreshed RHS can share the exact weighted Gram while differing in
    # a factor-resolvable direction that normal equations round away. The
    # immutable CenteredSystem instance identifies one RHS generation, so
    # transformed RHS reuse needs only identity—not an O(n) response copy
    # and comparison. Geometry-only terminal consumers may reuse the same
    # compact factor across refreshed constant-weight systems.
    if (
        same_geometry
        and cached is not None
        and (response is None or (cached.system is system and cached.transformed_rhs is not None))
    ):
        return cached

    centre, centre_lo = system.centre_pair()
    if response is None:
        factor = grouped_augmented_factor(
            dm,
            W_current,
            system.penalty,
            center=centre,
            center_lo=centre_lo,
        )
        transformed_rhs = None
    else:
        factor, transformed_rhs = grouped_augmented_factor_rhs(
            dm,
            W_current,
            system.penalty,
            response=response,
            center=centre,
            center_lo=centre_lo,
        )
    factor_decomposition = decompose_factor(
        factor,
        retain_factor_solve=transformed_rhs is not None,
    )
    cached = _CenteredFactorCertification(
        system=system,
        factor=factor,
        decomposition=factor_decomposition,
        transformed_rhs=transformed_rhs,
    )
    cache[0] = cached
    return cached


def _retained_rank_info(
    centered_final: CenteredSystem,
    W: NDArray,
    augmented_rank: RankDecomposition,
    feature_edf: NDArray,
    *,
    dm: DesignMatrix,
    groups: list[GroupSlice],
    p: int,
    step_data_rank: Callable[[CenteredSystem, NDArray], RankDecomposition] | None,
    _return_working_system: bool,
    certify_centered_factor: Callable[..., _CenteredFactorCertification],
) -> RankInfo:
    """The data-subspace metadata only retained-fit inference reads.

    A pure function of this fit's final centred system and
    weights, so it is the same whether it runs here or, for a
    caller that does not yet know whether this fit is the one it
    publishes (``_defer_rank_info``), after the solve returns.

    Owner and lifetime: the caller's model, from the solve until
    ``fit_ops._install_deferred_rank_info`` runs it or the next
    REML attempt drops it. The centred system, weights, augmented
    decomposition and EDF are bound by ``partial`` with the design
    (``dm``, read by the data-rank factor route), which is read when it
    runs, so the invariant is that nothing mutates the design's group
    matrices in place in between (``update_reml_r_inv`` writes the
    specs; ``release_leaf_memo`` drops a memo only). A step's own data
    rank (``step_data_rank``, the QR step's on its dense design) is never
    deferred (``_final_geometry``), so no step is kept.  It is never a
    method of the fit object, which it would keep alive (#492).
    """
    M_beta = centered_final.hessian + centered_final.sum_w * np.outer(
        centered_final.mean_x, centered_final.mean_x
    )
    coefficient_rank = decompose_gram_if_authoritative(M_beta)
    if coefficient_rank is None:
        certification = certify_centered_factor(centered_final, W)
        raw_factor = np.vstack(
            (
                certification.factor,
                np.sqrt(centered_final.sum_w) * centered_final.mean_x,
            )
        )
        coefficient_rank = decompose_factor(raw_factor)
    if step_data_rank is not None:
        data_rank = step_data_rank(centered_final, W)
    else:
        # The export rows are Fisher (``prefer_observed=False``), whose
        # weights are nonnegative: the data Gram is PSD by construction,
        # so a negative eigenvalue at its null is formation rounding and
        # sends the rank to the factor below instead of raising.
        final_centre, final_centre_lo = centered_final.centre_pair()

        def data_factor_rank() -> RankDecomposition:
            if not np.any(centered_final.penalty):
                return certify_centered_factor(centered_final, W).decomposition
            return decompose_factor(
                grouped_weighted_factor(
                    dm,
                    W,
                    center=final_centre,
                    center_lo=final_centre_lo,
                )
            )

        # Unpenalized, the data Gram is the Hessian, whose factor
        # certificate the fit already holds for its standard
        # errors (``reml_slope_rank``); reading it costs nothing.
        data_rank = decide_data_rank(
            centered_final.data_gram,
            data_factor_rank,
            project=lambda basis: grouped_projected_factor(
                dm, W, basis, center=final_centre, center_lo=final_centre_lo
            ),
            structural=lambda: grouped_structural_nulls(
                dm, W, center=final_centre, center_lo=final_centre_lo
            ),
            rows=dm.n,
            psd_by_construction=not _return_working_system,
            certify=not np.any(centered_final.penalty),
        )
    group_edf = {g.name: float(np.sum(feature_edf[g.sl])) for g in groups}
    selected_columns = np.arange(p, dtype=int)
    selected_columns.setflags(write=False)
    feature_edf.setflags(write=False)
    return RankInfo(
        policy_version=SHARED_RANK_POLICY.version,
        coordinate_space="solver",
        selected_columns=selected_columns,
        selected_group_names=tuple(g.name for g in groups),
        sum_w=centered_final.sum_w,
        mean_x=centered_final.mean_x,
        intercept_edf=1.0,
        data=data_rank,
        augmented=augmented_rank,
        coefficient=coefficient_rank,
        feature_edf=feature_edf,
        group_edf=group_edf,
        objective_loss=None,
        data_gram=centered_final.data_gram,
        data_gram_weights=weights_fingerprint(W),
    )


def _fit_irls_direct_once(**kwargs):
    """Run one ``_IRLSDirectFit``: the module-level name callers and tests wrap."""
    return _IRLSDirectFit(**kwargs).run()


class _IRLSDirectGramStep:
    """The fit's step on the Gram path: SCOP, dense or the QP.

    Each iteration forms the weighted Gram and solves it.  Which of the three
    runs is still read from the fit's route flags, until each is a step of its
    own as ``_IRLSDirectQRStep`` and ``_IRLSDirectStructuredStep`` are.  It keeps
    nothing, and its rank metadata takes the module route, which may be deferred.
    """

    __slots__ = ()
    backend = "gram"
    data_rank = finalise = None

    def propose(self, fit: _IRLSDirectFit, it: int) -> None:
        # Gram path: form X'WX via per-group gram, solve (p+1)×(p+1).
        _t0 = time.perf_counter()
        z_off = fit.z - fit.offset
        if fit._has_scop:
            fit._step_scop(it, z_off, _t0)
        elif not fit.has_constraints:
            fit._step_dense(z_off, _t0)
        else:
            fit._step_qp(it, z_off, _t0)
        # A bounded factor pass can intentionally replace an uncertain Gram
        # rank.  Preserve that in iteration diagnostics, but do not report it
        # as a failed solve or recommend switching the whole fit to dense QR.
        _used_svd, used_rank_certification = fit._used_svd, fit.used_rank_certification
        _consecutive_svd = fit._consecutive_svd
        warnable_svd_fallback = _used_svd and not used_rank_certification
        if warnable_svd_fallback:
            _consecutive_svd += 1
        else:
            _consecutive_svd = 0
        fit._consecutive_svd = _consecutive_svd
        if fit.direct_solve == "auto" and _consecutive_svd == 3:
            logger.warning(
                "fit_irls_direct: %d consecutive SVD fallbacks (cond ~%.1e). "
                "Consider direct_solve='qr' for near-collinear data.",
                _consecutive_svd,
                fit._cond_est,
            )


class _IRLSDirectFitFields:
    """The fit's state beyond its parameters, by the method that first sets it.

    Slots, not an instance dict: a field is read and written at a fixed offset, not
    hashed into a dict of some two hundred entries, and a misspelt field is an error.
    """

    __slots__ = (
        # run
        "recorder"
        # _setup_design
        " dm n p gms objective_merit_scale structured_decision _use_structured step"
        " _direct_fallback_reason _state_center beta intercept _start_lowered S"
        " _penalty_curvature _row_set_curvature"
        # _setup_routes
        " prev_active_set _scop_run has_constraints A_all b_all abs_A_all _has_scop"
        " _scop_columns _scop_curvature _observed_newton_active _stagnation_newton_available"
        # _setup_scop
        " _S_reduced _reduced_to_full _reduced_dm _reduced_tabmat_state _scop_specs _scop_order"
        " _scop_newton _expose_exact_support_state _scop_state"
        # _setup_systems
        " _mean_space_newton_available _constant_centered_cache _constant_centered_z"
        " certify_centered_factor _initial_data_key _tabmat_split _tabmat_centering_state"
        " _can_reuse_weighted_gram fixed_owners _initial_data_pending"
        # _setup_loop
        " converged XtWX_beta _last_working_centered _last_working_offset_mean _score_centre"
        " _last_mode_residual"
        " _mode_ratios _stagnation_window score_stagnated _true_ratios _mean_space_newton"
        " _true_stop _true_confirmed _true_stop_ratio _last_true_residual _held_rank"
        " _judged_truncated _last_truncated _mean_space_invalid _link_domain_invalid"
        " objective_prev dev_prev eta_unclipped eta mu max_halving _consecutive_svd"
        " _reported_qp_nonconvergence _degeneracy_retry_failed retained_qp_converged"
        " termination_reason deviance_relative_change mode_bar committed"
        # _scop_initial_state
        " scop_outer_penalty scop_outer_penalty_abs scop_merit_errors merit_gamma"
        " gaussian_response_roundoff nb2_unit_roundoff scop_committed"
        # _working
        " beta_prev intercept_prev committed_active_set proposal_qp_converged rank_truncated"
        " used_rank_certification scop_proposal_eta_unclipped proposal_centred_intercept"
        " working_eta_unclipped working_eta working_mu working_rows W z newton_score"
        " newton_committed_rows w_ratio"
        # the step (``step.propose``)
        " _cond_est _used_svd"
        # _step_scop
        " scop_ordinary centred_level scop_results"
        # _search_scop
        " step_rejected dev n_halvings decision proposal_scop retained retained_scop"
        # _search
        " proposal"
        # _converge
        " proposal_state convergence_label convergence_tolerance dev_rel_change coef_change"
        " convergence_value converged_this_iter"
        # _finish
        " centred_intercept_lo _boundary_rows _published_truncated _truncated_boundary"
        # _terminal_rows
        " export_rows"
        # _final_system
        " centered_final XtWX XtW1 sum_W"
        # _final_geometry, or the step's own finaliser
        " structured_factor reml_geometry_summary XtWX_S_inv_beta log_det_H reml_hessian_rank"
        " retained_reml_decomposition p_eff rank_info"
    ).split()


@dataclass(eq=False, repr=False, slots=True)
class _IRLSDirectFit(_IRLSDirectFitFields):
    """Fit a penalised GLM via direct IRLS (no BCD).

    Solves β = (X'WX + S)⁻¹ X'Wz at each iteration.  Uses gram-based
    operations to form X'WX without materialising the full (n, p) dense
    matrix.  For discretized groups (DiscretizedSSPGroupMatrix), this
    reduces the per-iteration cost from O(n·p²) to O(n_bins·K²).

    Returns (PIRLSResult, XtWX_S_inv) where XtWX_S_inv is the (p, p)
    profiled-intercept slope inverse from the final iteration, reusable for
    REML trace terms.

    Parameters
    ----------
    X : DesignMatrix or ndarray
        Design matrix (per-group or dense).
    y : (n,) array
        Response variable.
    weights : (n,) array
        Fitting weights.  ``weight_semantics`` says what they mean, and
        decides only the likelihood size the published dispersion divides by;
        the working weights themselves are the same under either reading.
    family : Distribution
        GLM family (Poisson, Gamma, NB2, etc.).
    link : Link
        Link function.
    groups : list of GroupSlice
        Group structure.
    lambda2 : float or dict
        Smoothing penalty weight(s).
    offset : (n,) array, optional
        Offset term.
    beta_init : (p,) array, optional
        Warm-start coefficients.
    intercept_init : float, optional
        Warm-start intercept.
    max_iter : int
        Maximum IRLS iterations (default 100).
    tol : float
        Deviance convergence tolerance (default 1e-6).
    return_xtwx : bool
        If True, also return the final weighted Gram matrix X'WX. Used by the
        REML outer loop to avoid rebuilding X'WX in cheap iterations when W is
        held fixed.
    compute_rank_info : bool
        If False, skip data-subspace metadata used only by retained-fit
        inference. Intermediate REML fits still compute the coefficient and
        augmented decompositions needed for their objective and EDF.
    _defer_rank_info : list, optional
        With ``compute_rank_info=False``, a list to which a dense fit with
        statistics appends a zero-argument callable returning the
        ``RankInfo`` it skipped, bit-for-bit what ``compute_rank_info=True``
        attaches. For a caller that learns only after the fit whether it is
        the one published (NB2 auto-theta REML refits).
    _return_working_system : bool
        Internal fREML performance-iteration mode. Return the centered system
        used for the last coefficient update instead of rebuilding it at the
        proposed coefficients. The authoritative final fit must leave this
        False so exported inference remains tied to the retained model.
    _compute_fit_statistics : bool
        Internal optimization switch. If False, omit EDF and scale summaries
        that the fREML outer loop does not consume. The authoritative final
        fit must leave this True.
    _compensate_centred_intercept : bool
        Internal switch for in-loop REML fits that compute statistics (the
        bootstrap, and traced candidates and trials): False keeps their eta
        the solver's ``alpha + X~ beta`` bit for bit.  With it and
        ``_compute_fit_statistics`` True, a centred Gaussian identity fit
        publishes its intercept as the compensated pair ``(alpha, alpha_lo)``
        (``mode_score.centred_intercept_remainder``) and its deviance and
        scale at that predictor.
    _centred_init : (float, ndarray), optional
        The warm start's centred state ``(alpha, c)``: its predictor is
        ``alpha + (X - 1 c') beta_init + offset`` (``PIRLSResult.centred_intercept``
        and ``state_center``; ``centred_warm_start``).  The fit carries it to
        its own centre exactly instead of taking it back from
        ``intercept_init``, whose rounding cancels ``c' beta`` at a column's
        offset.  Ignored without ``beta_init`` (the state's own ``beta``) or
        when ``intercept_init`` is not the start.
    _compute_reml_geometry : bool
        Internal SCOP-candidate switch. If False, omit the generic profiled
        slope inverse, determinant, and rank because the caller replaces them
        with one joint latent-coordinate LAML geometry. This requires both
        retained rank metadata and fit statistics to be disabled. Public and
        terminal fits must leave this True.
    _newton_latch : list of one bool, optional
        Internal, shared by the PIRLS solves of one REML fit.  A Gamma/log
        solve that switched from Fisher to Newton sets it, and a later solve
        of the fit that finds it set starts on Newton steps.  ``None`` (the
        default) shares nothing.
    _working_newton : bool
        Internal, read only with ``_return_working_system``: the fREML
        performance iteration's update takes observed (Newton) rows instead
        of Fisher's, for a pair whose observed rows are never negative
        (``working_rows.concave_observed_rows``).  The discrete optimizer
        asks for it when the pair's binned REML curvature is observed
        (``classify_binned_reml_curvature``), so its working model is the
        criterion it publishes.
    _use_observed_newton : bool
        Internal curvature switch. When enabled, an ordinary Tweedie/log fit
        takes exact observed-Newton steps on every iteration, and a Gamma/log
        fit starts on Fisher scoring, switching to Newton only when its
        resolved score is measured to stall (``_observed_newton_active``) or
        when ``_newton_latch`` is set; no rejected step switches the curvature
        (one-engine design §3.11). Unsupported, constrained, SCOP, and
        cached-working-system routes retain Fisher scoring. Public callers
        should leave this True.
    _deviance_init : float, optional
        Previously evaluated deviance at ``beta_init``/``intercept_init``.
        Used by private fREML steps to avoid repeating a full response scan.
    _compute_scop_postfit_inference : bool
        Internal SCOP EFS switch. Candidate modes leave this False; the
        terminal/public mode installs covariance and EDF exactly once.
    record_diagnostics : bool
        If True, record per-iteration W/mu/eta stats on the result.
    S_override : (p, p) ndarray, optional
        Pre-built penalty matrix.  When provided, skips internal
        ``_build_penalty_matrix`` call entirely.
    reml_penalties : list of PenaltyComponent, optional
        Forwarded to ``_build_penalty_matrix`` for the multi-penalty path.

    Returns
    -------
    result : PIRLSResult
    XtWX_S_inv : (p, p) ndarray
        Slope block of the full augmented Hessian inverse after profiling the
        intercept: ``(X_c' W X_c + S)^+``.
    """

    X: NDArray | DesignMatrix
    y: NDArray
    weights: NDArray
    family: Distribution
    link: Link
    groups: list[GroupSlice]
    lambda2: float | dict[str, float]
    offset: NDArray | None = None
    beta_init: NDArray | None = None
    intercept_init: float | None = None
    max_iter: int = 100
    tol: float = 1e-8
    return_xtwx: bool = False
    profile: dict | None = None
    cache_out: dict | None = None
    record_diagnostics: bool = False
    direct_solve: str = "auto"
    convergence: str = "deviance"
    S_override: NDArray | None = None
    reml_penalties: list[PenaltyComponent] | None = None
    return_scop_state: bool = False
    scop_state_init: dict[int, dict] | None = None
    debug_recorder: Any = None
    debug_context: dict[str, object] | None = None
    compute_rank_info: bool = True
    _return_working_system: bool = False
    _compute_fit_statistics: bool = True
    _compute_reml_geometry: bool = True
    _retain_reml_decomposition: bool = False
    _use_observed_newton: bool = True
    _working_newton: bool = False
    _newton_latch: list[bool] | None = None
    _deviance_init: float | None = None
    trace_run: TraceRun | None = None
    trace_purpose: str = "fit"
    _compute_scop_postfit_inference: bool = True
    separation: str = "warn"
    _: KW_ONLY
    weight_semantics: str
    _initial_data_reuse: _InitialDataReuse | None = None
    _raw_moment_policy: TabmatCenteringState | None = None
    _fisher_data_reuse: _FisherDataReuse | None = None
    _scop_run_centring: SCOPRunCentring | None = None
    _laplace_excluded: tuple[int, ...] = ()
    _mode_bar: float | None = None
    _compensate_centred_intercept: bool = True
    _centred_init: tuple[float, NDArray] | None = None
    _defer_rank_info: list[Callable[[], RankInfo]] | None = None

    def run(self) -> tuple[PIRLSResult, NDArray] | tuple[PIRLSResult, NDArray, NDArray]:
        self._setup_design()
        self.recorder = recorder = _IRLSDirectRecorder(self)
        self._setup_routes()
        self._setup_scop()
        self._setup_systems()
        recorder.start()
        self._setup_loop()
        step = self.step
        for it in range(self.max_iter):
            self._working(it)
            step.propose(self, it)
            if self._has_scop:
                self._search_scop(it)
            else:
                self._search(it)
            self._converge(it)
            self._judge_mean_space()
            self._terminate(it)
            recorder.iteration(self, it)
            dev, dev_prev = self.dev, self.dev_prev
            deviance_relative_change = abs(dev - dev_prev) / (abs(dev_prev) + 1)
            self.deviance_relative_change = deviance_relative_change
            logger.info(
                f"  irls_direct iter={it + 1:3d}  "
                f"dev={dev:12.1f}  delta={deviance_relative_change:10.2e}"
            )

            if not np.isfinite(dev):
                logger.warning(f"IRLS direct non-finite deviance at iter={it + 1}: dev={dev:.2e}")
                break
            if recorder.record_debug_rows:
                recorder.debug_row(self, it)

            if self.step_rejected:
                logger.warning(
                    "IRLS direct rejected all trial steps at iter=%d; restored committed state",
                    it + 1,
                )
                break

            retained = self.retained
            self.committed = retained
            if self._has_scop:
                self.scop_committed = self.retained_scop
            if self.converged_this_iter:
                self.converged = True
                break
            if self.score_stagnated:
                break
            self.dev_prev = dev
            self.objective_prev = (
                retained.deviance
                if retained.penalized_deviance is None
                else retained.penalized_deviance
            )
        recorder.stop(it)
        self._finish(it)
        self._terminal_rows()
        recorder.profile_fit(self, it)
        if step.finalise is None:
            self._final_system()
            self._final_geometry()
        else:
            step.finalise(self)
        return self._publish(it)

    def _setup_design(self) -> None:
        X, family, link, groups, weights = self.X, self.family, self.link, self.groups, self.weights
        offset, S_override, intercept_init = self.offset, self.S_override, self.intercept_init
        if isinstance(X, DesignMatrix):
            dm = X
        else:
            from superglm.solvers.pirls import _wrap_dense_X

            dm = _wrap_dense_X(X, groups)

        n = dm.n
        p = dm.p
        gms = dm.group_matrices

        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n,):
            raise ValueError("weights must match the design row count")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("weights must contain at least one positive value")
        objective_merit_scale = _irls_objective_scale(
            y=self.y,
            weights=weights,
            family=family,
            link=link,
        )

        structured_decision = resolve_structured_backend(
            gms,
            groups,
            direct_solve=self.direct_solve,
            coefficient_width=p,
            lambda2=self.lambda2,
            S_override=S_override,
            nesting_cache=getattr(dm, "_structured_layout_cache", None),
        )
        if (
            structured_decision.use_structured
            and S_override is None
            and self.reml_penalties is None
        ):
            reason = (
                "the structured candidate requires compact reml_penalties "
                "when S_override is not supplied"
            )
            if self.direct_solve == "structured":
                raise ValueError(f"direct_solve='structured' is ineligible: {reason}.")
            structured_decision = replace(
                structured_decision,
                use_structured=False,
                fallback_reason=reason,
            )
        _use_structured = structured_decision.use_structured
        _direct_fallback_reason = structured_decision.fallback_reason
        # The structured step is chosen with its route, from the structure alone,
        # because its layout fixes the state's centre; QR or Gram is chosen in
        # ``_setup_systems``.
        _state_center: NDArray | None = None
        if _use_structured:
            self.step = step = _IRLSDirectStructuredStep(dm, groups, structured_decision)
            _state_center = step.state_centre(weights, p)

        if offset is None:
            offset = np.zeros(n)

        beta = self.beta_init.copy() if self.beta_init is not None else np.zeros(p)

        if intercept_init is not None:
            intercept = intercept_init
        else:
            intercept = coefficient_initial_intercept(
                distribution=family,
                link=link,
                y=self.y,
                sample_weight=weights,
            )
        # Whether the default start below was lowered into the family's mean space
        # (binomial/log; a start raised into an inverse link's domain is not).
        _start_lowered = False
        if intercept_init is None and (
            mean_space_violation(family, link) is not None
            or link_domain_violation(link) is not None
        ):
            # The domain guards below keep every accepted state inside the mean
            # space and the link's domain only from a start inside them, and the
            # default intercept is chosen before the offset: move it until the
            # start is inside (``irls_state.interior_start_intercept``).  A
            # caller's warm start is left as it is.
            start_intercept = interior_start_intercept(
                family,
                link,
                (dm.matvec(beta) if np.any(beta) else 0.0) + intercept + offset,
                weights,
                intercept,
                level=intercept,
            )
            if start_intercept != intercept:
                intercept = start_intercept
                self._deviance_init = None
                _start_lowered = mean_space_violation(family, link) is not None

        # Dense paths retain the existing p x p penalty oracle. Structured paths
        # add each penalty directly to A or d, unless a caller already supplied a
        # dense override (which remains authoritative).
        S: NDArray | None
        if _use_structured:
            S = None if S_override is None else np.asarray(S_override, dtype=np.float64)
        elif S_override is not None:
            S = S_override
        else:
            S = _build_penalty_matrix(
                gms, groups, self.lambda2, p, reml_penalties=self.reml_penalties
            )
        self.dm, self.n, self.p, self.gms, self.weights = dm, n, p, gms, weights
        self.offset, self.objective_merit_scale = offset, objective_merit_scale
        self.structured_decision, self._use_structured = structured_decision, _use_structured
        self._direct_fallback_reason, self._state_center = _direct_fallback_reason, _state_center
        self.beta, self.intercept, self._start_lowered, self.S = beta, intercept, _start_lowered, S
        # diag(S), formed once per call on first use by the mode score (S is fixed)
        self._penalty_curvature: list[NDArray] = []
        # d'Sd along each one-hot reference and joint-cell direction
        # (``mode_score.row_set_quadratics``): one penalty product each, formed
        # once per call on first use (S and the design are fixed within a call)
        self._row_set_curvature: list[NDArray] = []

    def penalty_matvec(self, beta_values: NDArray, *, magnitude: bool = False) -> NDArray:
        """Apply the fitted penalty without expanding an identity random-effect block.

        With ``magnitude``, return ``|S| |beta|``, the magnitude this product
        sums, which bounds its rounding componentwise.
        """
        S, lambda2 = self.S, self.lambda2
        values = np.asarray(beta_values, dtype=np.float64)
        if magnitude:
            values = np.abs(values)
        if S is not None:
            return (np.abs(S) if magnitude else S) @ values
        if self.reml_penalties is None:  # pragma: no cover - validated above
            raise RuntimeError("Structured penalty components are unavailable.")
        from superglm.reml.penalty_algebra import (
            penalty_component_magnitude_matvec,
            penalty_component_matvec,
        )

        apply = penalty_component_magnitude_matvec if magnitude else penalty_component_matvec
        product = np.zeros_like(values)
        for component in self.reml_penalties:
            lam = float(lambda2[component.name]) if isinstance(lambda2, dict) else float(lambda2)
            if lam == 0.0:
                continue
            product[component.group_sl] += lam * apply(
                component,
                values[component.group_sl],
                self.gms[component.group_index],
            )
        return product

    def penalty_rounding(self) -> int:
        """``k`` with ``fl(S v)`` within ``gamma_k fl(|S| |v|)`` (``penalty_product_rounding``)."""
        from superglm.reml.penalty_algebra import penalty_product_rounding

        return penalty_product_rounding(self.p, None if self.S is not None else self.reml_penalties)

    def penalty_curvature(self) -> NDArray:
        """``diag(S)``: the dense penalty's diagonal, or the components' without forming ``S``."""
        lambda2, reml_penalties = self.lambda2, self.reml_penalties
        if self.S is not None:
            return np.diag(self.S).astype(np.float64, copy=True)
        from superglm.reml.identified import penalty_diagonal

        lambdas = (
            lambda2
            if isinstance(lambda2, dict)
            else {component.name: float(lambda2) for component in reml_penalties or ()}
        )
        return penalty_diagonal(self.p, lambdas, reml_penalties)

    def penalty_quadratic(self, beta_values: NDArray) -> float:
        values = np.asarray(beta_values, dtype=np.float64)
        return float(values @ self.penalty_matvec(values))

    def mode_residual(
        self,
        beta_values: NDArray,
        intercept_value: float,
        mu_values: NDArray,
        eta_values: NDArray,
        centred_intercept: float | None = None,
        resolve_cap: float = MODE_RESOLVE_CAP,
    ) -> ModeResidual:
        """The observed-REML certificate's score at an iterate (``convergence="mode_score"``).

        The row score from Fisher rows (the score itself does not depend on
        the curvature the rows carry), centred on the system the last solve
        linearised about (``_score_centre``), its floors and weak tests
        resolved once every relative score is within ``MODE_RESOLVE_CAP``
        (``solvers.mode_score``).  The floors read the iterate's intercept
        about ``mean_x``; a centred state reads it from its own ``alpha`` and
        the offset of ``mean_x`` from the centre (``_system_offset_mean``), not
        from the raw intercept, which cancels ``c' beta`` at a column's offset.
        """
        _penalty_curvature = self._penalty_curvature
        assert self._score_centre is not None
        mean_x, sum_w, diagonal, excluded_indices, offset_mean = self._score_centre
        rows = coefficient_working_rows(
            distribution=self.family,
            link=self.link,
            y=self.y,
            mu=mu_values,
            eta=eta_values,
            sample_weight=self.weights,
            prefer_observed=False,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.sqrt(np.abs(diagonal) / sum_w)
        excluded = np.zeros(self.p, dtype=bool)
        excluded[list(excluded_indices)] = True
        # the slopes the REML fit's Laplace approximation leaves out
        # (``reml.identified``, design §3.9): flagged and kept, never gated
        excluded[list(self._laplace_excluded)] = True
        if centred_intercept is None or offset_mean is None:
            shift = float(mean_x @ beta_values)
            alpha = float(intercept_value) + shift
            eta_tilde = eta_values - self.offset - float(intercept_value) - shift
        else:
            shift = float(offset_mean @ beta_values)
            alpha = float(centred_intercept) + shift
            eta_tilde = eta_values - self.offset - float(centred_intercept) - shift
        if not _penalty_curvature:
            _penalty_curvature.append(self.penalty_curvature())
        return penalized_mode_residual(
            dm=self.dm,
            row_score=rows.weights * (rows.response - eta_values),
            fisher_weights=rows.weights,
            positive_prior=self.weights > 0.0,
            mean_x=mean_x,
            centered_scale=np.where(np.isfinite(scale), scale, 0.0),
            alpha=alpha,
            eta_tilde=eta_tilde,
            penalty_score=self.penalty_matvec(beta_values),
            penalty_magnitude=self.penalty_matvec(beta_values, magnitude=True),
            penalty_curvature=_penalty_curvature[0],
            sum_w=sum_w,
            bar=self.mode_bar,
            excluded=excluded,
            resolve_cap=resolve_cap,
            penalty_rounding=self.penalty_rounding(),
        )

    def true_mode_residual(
        self,
        beta_values: NDArray,
        intercept_value: float,
        eta_values: NDArray,
        active_rows: NDArray | None,
        scop_groups: tuple[_SCOPGroupState, ...] | None,
    ) -> tuple[float, bool]:
        """The mode certificate on the binomial/log score itself, for a stop the clip can fool.

        ``(ratio, resolved)``: the largest relative score over its bar
        (``ModeResidual.ratio``), ``inf`` when the arithmetic cannot resolve
        it, and whether every relative score is within ``MODE_RESOLVE_CAP``
        (near the mode).  ``mode_residual``'s relative penalized score, bar,
        floors and weak tests, evaluated on the score and Fisher weights of
        the unclipped ``eta_values`` (``irls_state.mean_space_score_rows``)
        and centred on those weights (``mode_score.weighted_column_centring``),
        not on the clipped solve's system.

        **Units.** Each relative score is a ratio of quantities of degree one
        in the weights and the penalty taken together, so it does not depend
        on their units; the arithmetic that forms it does.
        - The prior weights are first brought by an even power of two to a
          largest entry in ``[1/4, 1)``, before any row product, so no row
          product underflows that would not at unit scale.
        - Each coordinate then takes its own even power of two: the shift
          that keeps its penalty terms, in the weights' new units, below
          ``2^1000``, applied to its data terms too (``penalized_mode_residual``'s
          ``column_shift``).  A data term that underflows there lies more
          than ``2^-2000`` below the penalty beside it.
        - Powers of two are exact on normal and subnormal values, and an even
          one keeps ``zeta``'s square root exact.
        - Gradual underflow adds at most half the subnormal spacing
          ``2^-1074`` of absolute error to each operation (Higham 2002,
          section 2.1), so a sum over ``n`` rows carries at most ``(n + 2)
          2^-1074`` beyond its ``gamma_n`` relative error
          (``_underflow_allowance``, the whole spacing per operation since
          half of it is not representable).  Where that is not below the
          bar's allowance, ``bar`` times an identified coordinate's scale,
          the certificate cannot resolve the score and refuses: ``inf``,
          never a pass.
        - Slopes the weak test finds at rounding-level curvature are left out
          of the ratio only while the iterate is at the mode along them: half
          their block Newton decrement, over their penalised curvature block
          (``weak_penalty_block`` for the penalty's part), within the
          log-likelihood's own rounding over the rows the block touches,
          ``gamma_{|S| + 4} sum_{i in S} |l_i|`` (``penalized_mode_residual``'s
          ``decrement_noise``).  The penalty's rounding is left out of that
          noise, which can only refuse.

        **Constraints.** ``active_rows`` are the hard constraints ``a' beta >=
        b`` active at the iterate.  A constrained mode is stationary when ``G
        + A' m = 0`` with ``m >= 0``, so the residual tested is ``G + A' m``
        at the non-negative least-squares ``m`` of its Jacobi-scaled form.
        Each row is put in the coordinates' units, brought to a largest entry
        in ``[1/2, 1)`` by a power of two and then to unit length: this leaves
        the cone it spans unchanged, and its own units can neither overflow
        nor vanish in the norm.

        **SCOP.** A shape-constrained group's coefficients are its latent
        ``theta``, with ``gamma = forward(theta)``.  Its stationarity is the
        latent score ``J' B' s - lambda S theta`` (``J = d gamma / d theta``,
        diagonal).  The groups' latent scores are projected together off the
        joint directions their Newton step could not resolve, and tested in
        the same relative form, bar and floor, in the weights' units (a latent
        penalty that overflows there is refused).
        """
        dm, lambda2, p, y, weights = self.dm, self.lambda2, self.p, self.y, self.weights
        mode_bar, _penalty_curvature = self.mode_bar, self._penalty_curvature
        _row_set_curvature, penalty_matvec = self._row_set_curvature, self.penalty_matvec
        penalty_rounding = self.penalty_rounding
        self._last_true_residual[0] = None
        # this state is judged: no truncated direction unless one is found below
        self._judged_truncated[0] = (eta_values, ())
        positive = weights > 0.0
        largest_weight = float(np.max(weights[positive], initial=0.0)) if np.any(positive) else 0.0
        weight_exponent = _even_exponent(largest_weight)
        score, fisher = mean_space_score_rows(y, np.ldexp(weights, -weight_exponent), eta_values)
        if not (np.all(np.isfinite(score)) and np.all(np.isfinite(fisher))):
            return math.inf, False
        if not _penalty_curvature:
            _penalty_curvature.append(self.penalty_curvature())
        penalty_score = penalty_matvec(beta_values)
        penalty_magnitude = penalty_matvec(beta_values, magnitude=True)
        curvature = np.asarray(_penalty_curvature[0], dtype=np.float64)
        if not (
            np.all(np.isfinite(penalty_score))
            and np.all(np.isfinite(penalty_magnitude))
            and np.all(np.isfinite(curvature))
        ):
            return math.inf, False
        shift = _column_shift(
            np.maximum(np.abs(penalty_magnitude), np.abs(curvature)), weight_exponent
        )
        penalty_score = np.ldexp(penalty_score, -weight_exponent - shift)
        penalty_magnitude = np.ldexp(penalty_magnitude, -weight_exponent - shift)
        curvature = np.ldexp(curvature, -weight_exponent - shift)
        mean_x, sum_w, diagonal = weighted_column_centring(dm, fisher, positive)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.sqrt(np.maximum(diagonal, 0.0) / sum_w)
        excluded = np.zeros(p, dtype=bool)
        excluded[list(self._laplace_excluded)] = True
        if self._scop_columns is not None:
            excluded |= self._scop_columns
        tiny = float(np.finfo(np.float64).tiny)
        zeta = float(np.sum(np.abs(score))) / math.sqrt(max(sum_w, tiny))
        if active_rows is not None and active_rows.size:
            rows = np.ldexp(np.asarray(active_rows, dtype=np.float64), -shift[None, :])
            largest_entry = np.max(np.abs(rows), axis=1)
            rows = rows[largest_entry > 0.0]
            exponents = np.frexp(largest_entry[largest_entry > 0.0])[1]
            rows = np.ldexp(rows, -exponents[:, None])
            rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
            gradient = np.ldexp(centred_data_score(dm, score, mean_x), -shift) - penalty_score
            jacobi = np.ldexp(np.full(p, zeta), -(shift // 2)) * np.sqrt(
                np.ldexp(np.maximum(diagonal, 0.0), -shift) + np.maximum(curvature, 0.0)
            )
            jacobi = jacobi + np.abs(penalty_score)
            # any positive diagonal gives a valid m >= 0; keeping it within
            # eps of its largest entry keeps the scaled rows finite
            jacobi = np.maximum(jacobi, _EPS_FLOAT * max(float(np.max(jacobi, initial=0.0)), 1.0))
            if rows.size and np.all(np.isfinite(gradient)) and np.all(np.isfinite(jacobi)):
                from scipy.optimize import nnls

                try:
                    multipliers = nnls(rows.T / jacobi[:, None], -gradient / jacobi)[0]
                except RuntimeError:  # the active-set iteration's own limit
                    multipliers = np.zeros(rows.shape[0])
                penalty_score = penalty_score - rows.T @ multipliers
                penalty_magnitude = penalty_magnitude + np.abs(rows).T @ multipliers
        centre_shift = float(mean_x @ beta_values)

        likelihood_rows: list = []

        def likelihood_noise(support: NDArray) -> float:
            # the log-likelihood's rounding over the rows a weak block touches
            # (``penalized_mode_residual``): gamma for the terms summed there,
            # plus each row's own few roundings
            if not likelihood_rows:
                likelihood_rows.append(
                    np.abs(
                        mean_space_log_likelihood_rows(
                            y, np.ldexp(weights, -weight_exponent), eta_values
                        )
                    )
                )
            touched = np.asarray(support, dtype=bool) & positive
            noise = _gamma(int(np.count_nonzero(touched)) + 4) * float(
                np.sum(likelihood_rows[0][touched])
            )
            return noise if math.isfinite(noise) else 0.0

        def weak_penalty_block(columns: NDArray) -> NDArray:
            # the penalty's block on a few weakly identified slopes, in the
            # weights' units: one penalty product per slope
            block = np.empty((len(columns), len(columns)))
            for position, column in enumerate(columns):
                unit = np.zeros(p)
                unit[column] = 1.0
                block[:, position] = penalty_matvec(unit)[columns]
            with np.errstate(over="ignore"):
                return np.ldexp(0.5 * (block + block.T), -weight_exponent)

        residual = penalized_mode_residual(
            dm=dm,
            row_score=score,
            fisher_weights=fisher,
            positive_prior=positive,
            mean_x=mean_x,
            centered_scale=np.where(np.isfinite(scale), scale, 0.0),
            alpha=float(intercept_value) + centre_shift,
            eta_tilde=eta_values - self.offset - float(intercept_value) - centre_shift,
            penalty_score=penalty_score,
            penalty_magnitude=penalty_magnitude,
            penalty_curvature=curvature,
            sum_w=sum_w,
            bar=mode_bar,
            excluded=excluded,
            resolve_cap=MODE_RESOLVE_CAP,
            column_shift=shift,
            decrement_noise=likelihood_noise,
            penalty_block=weak_penalty_block,
            penalty_rounding=penalty_rounding(),
        )
        self._last_true_residual[0] = residual
        rows_n = int(np.count_nonzero(positive))
        underflow = _underflow_allowance(rows_n)
        assert residual.scale is not None
        identified = np.concatenate(([True], ~residual.excluded))
        if np.any(mode_bar * residual.scale[identified] <= underflow):
            return math.inf, False
        ratio = residual.ratio()
        # every set of rows the one-hot blocks move on their own, certified on
        # its own rows (``mode_score.row_set_residual``); the sets are the
        # design's, formed once per design (``mode_score.row_sets``)
        sets = row_sets(dm)
        if not _row_set_curvature:
            _row_set_curvature.append(
                row_set_quadratics(sets, p, penalty_matvec, penalty_rounding())
            )
        with np.errstate(over="ignore", invalid="ignore"):
            column_penalty = np.ldexp(penalty_score, shift)
            column_penalty_size = np.ldexp(penalty_magnitude, shift)
            column_curvature = np.ldexp(curvature, shift)
            set_curvature = np.ldexp(_row_set_curvature[0], -weight_exponent)
        ratio = max(
            ratio,
            row_set_residual(
                sets=sets,
                row_score=score,
                response=y,
                fisher_weights=fisher,
                positive_prior=positive,
                eta=eta_values,
                column_penalty=column_penalty,
                column_penalty_size=column_penalty_size,
                column_curvature=column_curvature,
                set_curvature=set_curvature,
                bar=mode_bar,
                underflow=underflow,
                penalty_rounding=penalty_rounding(),
            ),
        )
        # the directions the factorization this iteration holds truncates,
        # judged on the rows they move (``mode_score.truncated_direction_ratio``):
        # a light row set the relative score sums beside the heavy rows'
        # rounding.  The gram and qr routes hold one; unconstrained, no SCOP.
        held_rank = self._held_rank[0]
        if (
            held_rank is not None
            and held_rank.parameter_null_basis is not None
            and (active_rows is None or not active_rows.size)
            and not scop_groups
        ):
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                truncated_ratio, truncated = truncated_direction_ratio(
                    dm=dm,
                    null_basis=held_rank.parameter_null_basis,
                    angle=null_basis_angle(held_rank, self.n),
                    mean_x=mean_x,
                    row_score=score,
                    fisher_weights=fisher,
                    response=y,
                    positive_prior=positive,
                    penalty_gradient=np.ldexp(penalty_matvec(beta_values), -weight_exponent),
                    penalty_size=np.ldexp(
                        penalty_matvec(beta_values, magnitude=True), -weight_exponent
                    ),
                    penalty_apply=lambda v: np.ldexp(penalty_matvec(v), -weight_exponent),
                    bar=mode_bar,
                    underflow=underflow,
                    column_scale=held_rank.column_scale,
                    eta=eta_values,
                    penalty_size_apply=lambda v: np.ldexp(
                        penalty_matvec(v, magnitude=True), -weight_exponent
                    ),
                    penalty_rounding=penalty_rounding(),
                )
            ratio = max(ratio, truncated_ratio)
            self._judged_truncated[0] = (eta_values, truncated)
            if truncated:
                self._last_truncated[0] = (eta_values, truncated)
            if truncated_ratio == math.inf:
                # refused because the basis or the arithmetic cannot resolve
                # the truncated rows (an ``unresolved_basis`` record)
                self.recorder.count("irls_truncated_unresolved")
        resolved = bool(np.max(residual.relative, initial=0.0) <= MODE_RESOLVE_CAP)
        latent_parts = []
        for group_state in scop_groups or ():
            state = self._scop_state[group_state.group_index]
            group = self.groups[group_state.group_index]
            lam = float(lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2)
            theta = np.asarray(group_state.beta_eff, dtype=np.float64)
            penalty_matrix = np.asarray(state["S_scop"], dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                pull = np.ldexp(lam * (penalty_matrix @ theta), -weight_exponent)
                pull_size = np.ldexp(
                    lam * (np.abs(penalty_matrix) @ np.abs(theta)), -weight_exponent
                )
                pull_diagonal = np.ldexp(lam * np.diag(penalty_matrix), -weight_exponent)
            # a latent penalty past float64 beside the data's units: refused
            # (the shape-constrained groups are tested in the weights' units)
            if not (
                np.all(np.isfinite(pull))
                and np.all(np.isfinite(pull_size))
                and np.all(np.isfinite(pull_diagonal))
            ):
                return math.inf, False
            bins = state["bin_idx"]
            basis = np.asarray(state["B_scop"], dtype=np.float64)

            def by_bin(values: NDArray, bins=bins, basis=basis) -> NDArray:
                if bins is None:
                    return values
                return np.bincount(bins, weights=values, minlength=basis.shape[0])

            jacobian = state["reparam"].jacobian_diagonal(theta)
            latent_parts.append(
                (
                    group_state,
                    theta,
                    jacobian * (basis.T @ by_bin(score)) - pull,
                    jacobian**2 * ((basis**2).T @ by_bin(fisher)) + pull_diagonal,
                    np.abs(pull),
                    pull_size,
                    np.abs(jacobian) * (np.abs(basis).T @ by_bin(np.abs(score))),
                )
            )
        if latent_parts:
            # The joint step truncates one factor across every SCOP group and
            # hands each group its own columns of each discarded direction, so
            # the groups' blocks reassemble row by row into the joint
            # directions (a group absent here holds zeros in them).  The
            # concatenated latent score is projected once off their span,
            # which an SVD finds with its rank (a QR would invent a direction
            # for a zero block).
            widths = [part[1].size for part in latent_parts]
            offsets = np.concatenate(([0], np.cumsum(widths)))
            blocks = [
                (k, np.asarray(part[0].discarded_directions, dtype=np.float64))
                for k, part in enumerate(latent_parts)
                if part[0].discarded_directions is not None
                and np.asarray(part[0].discarded_directions).ndim == 2
                and np.asarray(part[0].discarded_directions).shape[0] > 0
            ]
            gradient = np.concatenate([part[2] for part in latent_parts])
            if blocks:
                counts = {rows.shape[0] for _, rows in blocks}
                if len(counts) != 1:  # pragma: no cover - joint-step invariant
                    raise RuntimeError(
                        "the joint SCOP step handed its groups different discarded directions"
                    )
                directions = np.zeros((counts.pop(), int(offsets[-1])))
                for k, rows in blocks:
                    directions[:, offsets[k] : offsets[k + 1]] = rows
                left, singular, _ = np.linalg.svd(directions.T, full_matrices=False)
                cutoff = float(np.max(singular, initial=0.0)) * directions.shape[1] * _EPS_FLOAT
                frozen = left[:, singular > cutoff]
                gradient = gradient - frozen @ (frozen.T @ gradient)
            curvature_all = np.concatenate([part[3] for part in latent_parts])
            pull_all = np.concatenate([part[4] for part in latent_parts])
            size_all = np.concatenate([part[5] for part in latent_parts])
            evaluation = np.concatenate([part[6] for part in latent_parts])
            latent_scale = np.maximum(
                tiny, zeta * np.sqrt(np.maximum(curvature_all, 0.0)) + pull_all
            )
            if np.any(mode_bar * latent_scale <= underflow):
                return math.inf, False
            relative = np.abs(gradient) / latent_scale
            floor = (
                _gamma(rows_n) * evaluation
                + _gamma(int(offsets[-1]) + 2) * size_all
                + _UNIT_ROUNDOFF_FLOAT * size_all
            ) / latent_scale
            ratio = max(ratio, float(np.max(relative / np.maximum(mode_bar, floor), initial=0.0)))
            resolved = resolved and bool(np.max(relative, initial=0.0) <= MODE_RESOLVE_CAP)
        return ratio, resolved

    def evaluate_state(
        self,
        beta_values: NDArray,
        intercept_value: float,
        *,
        phase: str,
        iteration: int,
        alpha: float | None = None,
        deviance: float | None = None,
        eta_unclipped: NDArray | None = None,
        enclosing_proposal_state_id: int | None = None,
        emit_trace: bool = True,
        centred_intercept: float | None = None,
    ) -> _IRLSState:
        recorder, _state_center = self.recorder, self._state_center
        if recorder.trace_enabled:
            trace_run = recorder.trace_run
            assert trace_run is not None
            state_id = trace_run.next_state_id()
            evaluation_id = trace_run.next_evaluation_id()
        else:
            state_id = None
            evaluation_id = None
        if _state_center is not None:
            # One-engine design §3.8: the state is (alpha, beta) about the fixed
            # prior-weighted centre, eta = alpha + X~ beta + offset, and the raw
            # intercept is only its reading; a state entering from raw
            # coordinates (the start, a warm start) is centred once.
            shift = math.fsum(_state_center * np.asarray(beta_values, dtype=np.float64))
            if centred_intercept is None:
                centred_intercept = float(intercept_value) + shift
            if eta_unclipped is None:
                eta_unclipped = (
                    centred_intercept
                    + centred_matvec(self.dm, beta_values, _state_center)
                    + self.offset
                )
            intercept_value = centred_intercept - shift
        state = _evaluate_irls_state(
            self.dm,
            self.y,
            self.weights,
            self.family,
            self.link,
            self.offset,
            beta_values,
            intercept_value,
            deviance=deviance,
            eta_unclipped=eta_unclipped,
            state_id=state_id,
            evaluation_id=evaluation_id,
            basis_id=recorder.trace_basis_id,
            lambdas=recorder.resolved_lambdas,
        )
        if not self._has_scop:
            state = replace(
                state,
                penalized_deviance=float(state.deviance + self.penalty_quadratic(state.beta)),
            )
        if centred_intercept is not None:
            state = replace(state, centred_intercept=float(centred_intercept))
        if emit_trace:
            recorder.emit_evaluation(
                state,
                phase=phase,
                iteration=iteration,
                alpha=alpha,
                enclosing_proposal_state_id=enclosing_proposal_state_id,
                deviance_reused=deviance is not None,
            )
        return state

    def _setup_routes(self) -> None:
        family, link, groups = self.family, self.link, self.groups
        _state_center, compute_rank_info = self._state_center, self.compute_rank_info
        _return_working_system = self._return_working_system
        # ── Constrained QP support (monotone splines) ──
        has_constraints = any(g.constraints is not None for g in groups)
        self.prev_active_set: list[int] | None = None
        A_all: NDArray | None = None
        b_all: NDArray | None = None
        # The feasibility scale needs ``|A_all|`` on every test, and the aggregate
        # system is fixed for the whole fit -- so it is built once here rather than
        # rebuilt two to three times per IRLS iteration plus once per line-search
        # halving.  See ``constrained_qp._feasibility_slack``.
        abs_A_all: NDArray | None = None
        if has_constraints:
            A_blocks: list[np.ndarray] = []
            b_blocks: list[np.ndarray] = []
            for g in groups:
                if g.constraints is not None:
                    A_model = np.zeros((g.constraints.n_constraints, self.p))
                    A_model[:, g.sl] = g.constraints.A
                    A_blocks.append(A_model)
                    b_blocks.append(g.constraints.b)
            A_all = np.vstack(A_blocks)
            b_all = np.concatenate(b_blocks)
            abs_A_all = np.abs(A_all)

        # ── SCOP monotone engine support ──
        _has_scop = any(g.monotone_engine == "scop" for g in groups)
        self._scop_run: SCOPRunCentring | None = None
        # the columns the binomial/log certificate tests in SCOP's latent
        # coordinates instead (``true_mode_residual``)
        _scop_columns: NDArray | None = None
        if _has_scop:
            _scop_columns = np.zeros(self.p, dtype=bool)
            for g in groups:
                if g.monotone_engine == "scop":
                    _scop_columns[g.sl] = True
        if (
            _state_center is None
            and not self._use_structured
            and not has_constraints
            and not _has_scop
            and hasattr(self.dm, "group_matrices")
        ):
            # The nested chain's centred state on gram's centred system too: eta
            # about the fixed prior-weighted centre, so a column at 1e8
            # contributes rows of its spread, not of its offset, and the
            # objective the line search compares keeps its digits (stage-1
            # verifier: a raw-1e8 column left the penalized deviance noisy at
            # 1e-11 relative, so no Newton step near the mode was accepted and
            # the score stalled at 1e-6).
            _state_center = prior_weighted_centre(self.dm, self.weights)
        _scop_curvature = "fisher"
        if _has_scop:
            from superglm.reml.observed_geometry import classify_scop_reml_curvature

            _scop_curvature = classify_scop_reml_curvature(family, link)
        if compute_rank_info and self._defer_rank_info is not None:
            raise ValueError("deferred rank metadata requires compute_rank_info=False")
        if (not self._compute_fit_statistics and compute_rank_info) or (
            _return_working_system and (compute_rank_info or has_constraints or _has_scop)
        ):
            raise ValueError("intermediate REML shortcuts require rank metadata to be disabled")
        if not self._compute_reml_geometry and (compute_rank_info or self._compute_fit_statistics):
            raise ValueError(
                "omitting generic REML geometry requires rank metadata and fit statistics "
                "to be disabled"
            )
        # Laplace-approximate REML is built on the observed Hessian, so its PIRLS
        # runs on full-Newton weights (Wood 2011, JRSSB 73(1), section 3; Wood, Pya
        # & Saefken 2016, JASA, section 3.3): Fisher scoring shares the mode but
        # converges only linearly under a non-canonical link. Newton starts only
        # where the Fisher weights vary, so no constant-weight Gram or Fisher-data
        # cache exists for it to invalidate. Gamma/log keeps Fisher: its constant
        # Fisher weights reuse one weighted Gram that Newton would rebuild every
        # iteration; the mode certificate's bar is set by the REML tolerance
        # (``mode_score.mode_certification_bar``), which Fisher's linear rate
        # reaches wherever Fisher scoring contracts.  It need not: Fisher's error
        # map ``I - F^-1 H`` has an eigenvalue below -1 where the observed
        # information ``w y / mu`` exceeds twice the Fisher information ``w``
        # along a direction, as heavy-tailed severities give, and there the score
        # stagnates above the bar; nearer 0 Fisher contracts too slowly to reach
        # it in the budget.  Such a solve switches to Newton once its resolved
        # score shows it (``mode_score.rate_exhausts_budget``; Green 1984, JRSSB
        # 46, section 2.3), and keeps it to the end; a solve whose Fisher score
        # reaches the bar in time never switches.
        # The exported geometry is Fisher either way (``export_rows``), and the
        # weighted-Gram cache is off while Newton runs.  Otherwise the curvature
        # is decided here, once, from the family, the link and the route
        # (one-engine design §3.11).
        # An observed iterate whose Hessian the structured factor refuses as not
        # positive definite takes a Levenberg shift inside the same curvature
        # (``_levenberg_shifted_operator``), and a trial whose observed rows are
        # not finite is rejected by the line search.
        # The fREML performance iteration (``_return_working_system``) takes
        # Newton rows when its optimizer asks (``_working_newton``): the discrete
        # route's working model then carries the observed curvature its REML
        # criterion is (``reml.discrete``).
        # A shape-constrained (SCOP) fit takes the same rows: its joint Newton
        # step is Pya & Wood's (2015, Appendix 1) full Newton on the observed
        # information, and the REML geometry it hands on reads Fisher rows at the
        # retained model either way.
        _observed_newton_active = bool(
            self._use_observed_newton
            and not has_constraints
            and (
                (self._working_newton and concave_observed_rows(family, link))
                if _return_working_system
                else supports_observed_newton(family, link)
                and not _has_constant_irls_weights(family, link)
            )
        )
        # A declared Newton pair kept on Fisher rows for its constant weights
        # (Gamma/log), which switches to Newton if its Fisher score stagnates,
        # and starts on Newton when an earlier solve of the same REML fit
        # switched (``_newton_latch``).
        _stagnation_newton_available = bool(
            self._use_observed_newton
            and not has_constraints
            and not _has_scop
            and not _return_working_system
            and supports_observed_newton(family, link)
            and not _observed_newton_active
        )
        if (
            _stagnation_newton_available
            and self._newton_latch is not None
            and self._newton_latch[0]
        ):
            _observed_newton_active = True
        self.has_constraints, self.A_all, self.b_all = has_constraints, A_all, b_all
        self.abs_A_all, self._has_scop, self._scop_columns = abs_A_all, _has_scop, _scop_columns
        self._state_center, self._scop_curvature = _state_center, _scop_curvature
        self._observed_newton_active = _observed_newton_active
        self._stagnation_newton_available = _stagnation_newton_available

    def observed_rows_invalid(self, candidate: _IRLSState) -> bool:
        """Whether a trial's observed rows are not finite, which rejects it.

        One-engine design §3.11: finiteness is the one condition the declared
        observed rows keep; a trial without finite rows is not a state the
        next Newton step can start from. Every line search of an observed fit
        reads it, the SCOP one included.
        """
        rows = coefficient_working_rows(
            distribution=self.family,
            link=self.link,
            y=self.y,
            mu=candidate.mu,
            eta=candidate.eta,
            sample_weight=self.weights,
            prefer_observed=True,
        )
        return rows.rejection_reason is not None

    def _setup_scop(self) -> None:
        groups, gms, n, lambda2 = self.groups, self.gms, self.n, self.lambda2
        _scop_run = self._scop_run
        _n_scop_groups = sum(g.monotone_engine == "scop" for g in groups)
        _expose_exact_support_state = False
        # group_idx -> {beta_scop, beta_scop_prev, reparam, B_scop, S_scop}
        _scop_state: dict[int, dict] = {}
        # A REML run's fits share one reduced design, their rung refusals and
        # their exact supports (``SCOPRunCentring``); a traced or debugged fit
        # stays fresh, as the Newton engine's carried policy does.
        if (
            self._has_scop
            and self._scop_run_centring is not None
            and self.debug_recorder is None
            and self.trace_run is None
        ):
            _scop_run = self._scop_run_centring
            _scop_run.bind((self.dm, self.family, self.link, *gms, *groups))
        if self._has_scop:
            for gi, g in enumerate(groups):
                if g.monotone_engine == "scop":
                    cached_scop_state = (
                        self.scop_state_init.get(gi) if self.scop_state_init is not None else None
                    )
                    reparam = g.scop_reparameterization
                    cached_S_scop = (
                        None if cached_scop_state is None else cached_scop_state.get("S_scop")
                    )
                    if isinstance(cached_S_scop, np.ndarray) and cached_S_scop.shape == (
                        g.size,
                        g.size,
                    ):
                        S_scop = cached_S_scop
                    else:
                        S_scop = reparam.penalty_matrix()
                    _gm = gms[gi]

                    # Warm-start beta_scop from previous outer EFS iteration if available
                    warm_beta_scop = None
                    if cached_scop_state is not None:
                        prev = cached_scop_state["beta_eff"]
                        q_eff = S_scop.shape[0]
                        if prev.shape == (q_eff,):
                            warm_beta_scop = prev.copy()

                    if isinstance(_gm, DiscretizedSCOPGroupMatrix):
                        state = {
                            "reparam": reparam,
                            "B_scop": _gm.B_scop_unique,
                            "S_scop": S_scop,
                            "bin_idx": _gm.bin_idx,
                            "beta_scop": warm_beta_scop,
                            "beta_scop_prev": None,
                        }
                    else:
                        compressed = (
                            _lone_scop_support(
                                _gm, gi, None if _scop_run is None else _scop_run.exact_supports
                            )
                            if _n_scop_groups == 1
                            else None
                        )
                        if compressed is not None:
                            _expose_exact_support_state = True
                            state = {
                                "reparam": reparam,
                                "B_scop": compressed[0],
                                "S_scop": S_scop,
                                "bin_idx": compressed[1],
                                "beta_scop": warm_beta_scop,
                                "beta_scop_prev": None,
                            }
                        else:
                            state = {
                                "reparam": reparam,
                                "B_scop": _gm.toarray(),
                                "S_scop": S_scop,
                                "bin_idx": None,
                                "beta_scop": warm_beta_scop,
                                "beta_scop_prev": None,
                            }
                    if cached_scop_state is not None:
                        for key in (
                            "H_scop_penalized",
                            "last_step_norm",
                            "last_fisher_fallback",
                            "discarded_directions",
                            "penalty_rank",
                            "penalty_log_det_omega_plus",
                            "penalty_eigvals_omega",
                        ):
                            if key in cached_scop_state:
                                state[key] = cached_scop_state[key]
                    _scop_state[gi] = state
            # Build mask of non-SCOP column indices for the reduced system
            _non_scop_cols = []
            _non_scop_groups_idx = []
            for gi, g in enumerate(groups):
                if gi not in _scop_state:
                    _non_scop_cols.extend(range(g.start, g.end))
                    _non_scop_groups_idx.append(gi)
            _non_scop_cols = np.array(_non_scop_cols, dtype=int)
            _p_reduced = len(_non_scop_cols)
            # Build reduced penalty matrix for non-SCOP groups
            _S_reduced = self.S[np.ix_(_non_scop_cols, _non_scop_cols)]
            # Build mapping from reduced beta index to full beta index
            _reduced_to_full = _non_scop_cols
            _reduced_gms = []
            for gi in _non_scop_groups_idx:
                _reduced_gms.append(gms[gi])
            if _scop_run is not None:
                _reduced_dm = _scop_run.reduced_design(
                    tuple(_non_scop_groups_idx), _reduced_gms, n, _p_reduced
                )
            else:
                _reduced_dm = DesignMatrix(_reduced_gms, n=n, p=_p_reduced)
            _reduced_tabmat_state = TabmatCenteringState()
            if _scop_run is not None:
                _seed_refusals(_reduced_tabmat_state, _scop_run.reduced_refusals)

            _scop_specs = {
                gi: SCOPGroupSpec(
                    group_index=gi,
                    group=groups[gi],
                    reparam=st["reparam"],
                    B_scop=st["B_scop"],
                    S_scop=st["S_scop"],
                    bin_idx=st["bin_idx"],
                )
                for gi, st in _scop_state.items()
            }
            _scop_order = sorted(_scop_specs)
            _scop_newton = SCOPJointNewton(
                [_scop_specs[gi] for gi in _scop_order],
                lambda2,
                _reduced_dm,
                n,
                supports=None if _scop_run is None else _scop_run.exact_supports,
            )

            # QP/warm initialization is part of the first committed state, not the
            # first proposal. This gives iteration one a coherent latent baseline.
            provisional = self.evaluate_state(
                self.beta,
                self.intercept,
                phase="scop_initialization",
                iteration=0,
            )
            initial_rows = coefficient_working_rows(
                distribution=self.family,
                link=self.link,
                y=self.y,
                mu=provisional.mu,
                eta=provisional.eta,
                sample_weight=self.weights,
                prefer_observed=False,
            )
            W_init = initial_rows.weights
            z_init = initial_rows.response
            z_off_init = z_init - self.offset
            for gi, st in _scop_state.items():
                if st["beta_scop"] is None:
                    g_i = groups[gi]
                    lam_scop = lambda2.get(g_i.name, 0.0) if isinstance(lambda2, dict) else lambda2
                    bin_idx = st["bin_idx"]
                    if bin_idx is not None:
                        n_bins = st["B_scop"].shape[0]
                        W_agg = np.bincount(bin_idx, weights=W_init, minlength=n_bins)
                        Wz_agg = np.bincount(
                            bin_idx,
                            weights=W_init * z_off_init,
                            minlength=n_bins,
                        )
                        with np.errstate(divide="ignore", invalid="ignore"):
                            z_bin = np.where(W_agg > 0, Wz_agg / W_agg, 0.0)
                        st["beta_scop"] = st["reparam"].qp_initialize(
                            st["B_scop"],
                            z_bin,
                            lambda_penalty=lam_scop,
                            weights=W_agg,
                        )
                    else:
                        st["beta_scop"] = st["reparam"].qp_initialize(
                            st["B_scop"],
                            z_off_init,
                            lambda_penalty=lam_scop,
                            weights=W_init,
                        )
                gamma_eff = st["reparam"].forward(st["beta_scop"])
                st["gamma_eff"] = gamma_eff.copy()
                self.beta[groups[gi].sl] = gamma_eff
            self._S_reduced, self._reduced_to_full = _S_reduced, _reduced_to_full
            self._reduced_dm, self._reduced_tabmat_state = _reduced_dm, _reduced_tabmat_state
            self._scop_specs, self._scop_order = _scop_specs, _scop_order
            self._scop_newton = _scop_newton
        self._expose_exact_support_state = _expose_exact_support_state
        self._scop_state, self._scop_run = _scop_state, _scop_run

    def _setup_systems(self) -> None:
        dm, family, link, groups, gms = self.dm, self.family, self.link, self.groups, self.gms
        has_constraints, _has_scop = self.has_constraints, self._has_scop
        _use_structured, _raw_moment_policy = self._use_structured, self._raw_moment_policy
        _fisher_data_reuse = self._fisher_data_reuse
        # The fit's one step method, chosen here unless it is structured
        # (``_setup_design``): QR materialises the full design once.
        # Constrained QP / SCOP requires Gram path — force it if constraints present
        _use_qr = self.direct_solve == "qr" and not has_constraints and not _has_scop
        self._mean_space_newton_available = bool(
            mean_space_violation(family, link) is not None
            and not _use_qr
            and not _has_scop
            and not self._return_working_system
        )
        if not _use_structured:
            self.step: _IRLSDirectStep = (
                _IRLSDirectQRStep(gms, self.S, self.p) if _use_qr else _IRLSDirectGramStep()
            )

        # Tabmat acceleration: the structured layout owns a pruned small-block
        # plan, so it must not construct a split containing the dominant factor.
        # Other non-discrete paths retain the shared full-design split behavior.
        if _use_structured:
            _tabmat_split = None
        elif _use_qr:
            _tabmat_split = None
        elif has_constraints:
            # Constrained QP uses the shared stable centered-system builder, while
            # retaining the execution-plan route instead of materializing a
            # separate tabmat copy of the design.
            _tabmat_split = None
        else:
            # Ordinary intercept profiling currently benefits only when the split
            # contains a native high-cardinality categorical component. Avoid
            # materializing an unused dense duplicate for numeric and low-cardinality fits.
            _tabmat_split = dm.tabmat_centering_split
        _can_reuse_weighted_gram = _has_constant_irls_weights(family, link) and not _has_scop
        # Observed rows are not the constant Fisher weights this cache assumes.
        _can_reuse_weighted_gram = (
            _can_reuse_weighted_gram and not _use_structured and not self._observed_newton_active
        )
        dm.execution_plan.validate_group_spans(groups)
        _defer_raw_spline = (
            not dm.raw_spline_tabmat_plan_built
            and _is_raw_spline_tabmat_centering_candidate(gms, n=self.n)
            and (
                _defer_raw_spline_tabmat_plan(
                    n=self.n,
                    raw_width=sum(int(group.B.shape[1]) for group in gms if group.shape[1] > 0),
                    constant_weights=_can_reuse_weighted_gram,
                    repeated_fit=self.trace_purpose
                    in {"reml_bootstrap", "reml_candidate", "reml_line_search"},
                )
            )
        )
        _tabmat_centering_state = TabmatCenteringState(
            raw_spline_eligible=False if _defer_raw_spline else None
        )
        if self._scop_run is not None:
            _seed_refusals(_tabmat_centering_state, self._scop_run.full_refusals)
        fixed_owners = ()
        if _raw_moment_policy is not None or _fisher_data_reuse is not None:
            # A negative route choice is safe for changed W; no accepted preflight,
            # weighted system or penalty survives. Fixed-coordinate QP is eligible.
            fixed_owners = (
                (dm, dm.execution_plan, family, link, *gms, *groups)
                if type(dm) is DesignMatrix
                and type(family) in (Poisson, Gaussian, Gamma, Binomial, NegativeBinomial, Tweedie)
                and type(link) in (LogLink, IdentityLink, LogitLink)
                and all(
                    type(gm)
                    in (
                        CategoricalGroupMatrix,
                        DenseGroupMatrix,
                        SparseSSPGroupMatrix,
                        SupportCompressedSSPGroupMatrix,
                    )
                    for gm in gms
                )
                and all(
                    type(group) is GroupSlice
                    and (
                        group.constraints is None or type(group.constraints) is LinearConstraintSet
                    )
                    for group in groups
                )
                and not (_use_qr or _use_structured or _has_scop)
                and self.debug_recorder is None
                and self.trace_run is None
                else ()
            )
            if _raw_moment_policy is not None:
                _tabmat_centering_state.raw_moment_eligible = _raw_moment_policy.seed_raw_rejection(
                    fixed_owners
                )
                if not fixed_owners:
                    _raw_moment_policy = None
        if _fisher_data_reuse is not None and not (
            fixed_owners and type(family) is Gamma and type(link) is LogLink and not has_constraints
        ):
            _fisher_data_reuse.clear()
            _fisher_data_reuse = None
        if _defer_raw_spline:
            self.recorder.count("centered_spline_tabmat_cold_policy_rejections")
        self._constant_centered_cache: CenteredSystem | None = None
        self._constant_centered_z: NDArray | None = None
        # the factor certificate and its one-entry cache (``_certify_centered_factor``)
        self.certify_centered_factor = partial(
            _certify_centered_factor, dm, _fisher_data_reuse, [None]
        )
        _initial_data_pending = (
            self._initial_data_reuse is not None
            and type(family) is Poisson
            and type(link) is LogLink
            and type(dm) is DesignMatrix
            and all(
                type(gm)
                in (
                    CategoricalGroupMatrix,
                    DenseGroupMatrix,
                    SparseSSPGroupMatrix,
                    SupportCompressedSSPGroupMatrix,
                )
                for gm in gms
            )
            and not (_use_qr or _use_structured or has_constraints or _has_scop)
            and self.debug_recorder is None
            and self.trace_run is None
        )
        # These objects stay live throughout the owner line search. A new design
        # or warm state cannot consume an entry from the previous coordinates.
        self._initial_data_key = (
            (
                id(dm),
                id(dm.execution_plan),
                id(family),
                id(link),
                self.beta.tobytes(),
                np.float64(self.intercept).tobytes(),
            )
            if _initial_data_pending
            else None
        )
        self._tabmat_split = _tabmat_split
        self._tabmat_centering_state = _tabmat_centering_state
        self._can_reuse_weighted_gram, self.fixed_owners = _can_reuse_weighted_gram, fixed_owners
        self._raw_moment_policy, self._fisher_data_reuse = _raw_moment_policy, _fisher_data_reuse
        self._initial_data_pending = _initial_data_pending

    def get_centered_system(self, W_current: NDArray, z_off_current: NDArray) -> CenteredSystem:
        _tabmat_centering_state = self._tabmat_centering_state
        # read on every call: the Gamma/log switch turns it off mid-fit
        _can_reuse_weighted_gram = self._can_reuse_weighted_gram
        _constant_centered_cache = self._constant_centered_cache
        reuse = self._initial_data_reuse if self._initial_data_pending else None
        self._initial_data_pending = False
        before = replace(_tabmat_centering_state) if reuse is not None else None
        if reuse is not None:
            reused = reuse.take(
                self._initial_data_key,
                W_current,
                z_off_current,
                _tabmat_centering_state,
                np.asarray(self.S),
            )
            if reused is not None:
                return reused
        if (
            _can_reuse_weighted_gram
            and _constant_centered_cache is not None
            and self._constant_centered_z is not None
        ):
            if np.array_equal(z_off_current, self._constant_centered_z):
                return _constant_centered_cache
            self._constant_centered_cache = refresh_centered_rhs(
                system=_constant_centered_cache,
                dm=self.dm,
                W=W_current,
                z_off=z_off_current,
            )
            self._constant_centered_z = z_off_current.copy()
            return self._constant_centered_cache
        fisher = (
            self._fisher_data_reuse
            if _can_reuse_weighted_gram and not self._observed_newton_active
            else None
        )
        data = (
            fisher.take((*self.fixed_owners, self.weight_semantics), W_current)
            if fisher is not None
            else None
        )
        system = build_centered_system(
            dm=self.dm,
            W=W_current,
            z_off=z_off_current,
            penalty=np.asarray(self.S),
            tabmat_split=self._tabmat_split,
            tabmat_state=_tabmat_centering_state,
            profile=self.profile,
            _data=data,
        )
        if fisher is not None and fisher.data is None:
            fisher.remember(W_current, system)
        if self._scop_run is not None:
            _carry_refusals(self._scop_run.full_refusals, _tabmat_centering_state)
        if (
            self._raw_moment_policy is not None
            and _tabmat_centering_state.raw_moment_eligible is False
        ):
            self._raw_moment_policy.raw_moment_eligible = False
        if reuse is not None:
            reuse.remember(
                self._initial_data_key,
                W_current,
                z_off_current,
                before,
                _tabmat_centering_state,
                system,
            )
        if _can_reuse_weighted_gram:
            self._constant_centered_cache = system
            self._constant_centered_z = z_off_current.copy()
        return system

    def _setup_loop(self) -> None:
        _state_center = self._state_center
        self.converged = False
        self.XtWX_beta: (
            NDArray | BlockSymmetricOperator | SumToZeroBlockOperator | NestedDataOperator | None
        ) = None
        self._last_working_centered: CenteredSystem | None = None
        # its mean_x less the state's centre (``_system_offset_mean``), None without one
        self._last_working_offset_mean: NDArray | None = None
        # convergence="mode_score": the centring of the certificate's score at the
        # iterate each solve linearised about -- (mean_x, sum_w, centred diagonal,
        # coefficients the factor truncated as weakly identified, mean_x less the
        # state's centre or None without one) -- or None on a
        # route that forms no centred system (SCOP, linear constraints), which
        # keeps the coefficient-step test.
        self._score_centre: tuple | None = None
        self._last_mode_residual: ModeResidual | None = None
        # the certificate ratio at every iterate of a mode_score solve, for the
        # stagnation stop
        self._mode_ratios: list[float] = []
        # the certificate's bar (``mode_score.mode_certification_bar``): the REML
        # fit passes the one its stopping tolerance needs
        mode_bar = MODE_CERTIFICATION_BAR if self._mode_bar is None else float(self._mode_bar)
        self._stagnation_window = stagnation_window(self.max_iter, mode_bar)
        self.score_stagnated = False
        # the binomial/log score's certificate ratio at every stop it refused
        # (``true_mode_residual``), for the same stagnation stop
        self._true_ratios: list[float] = []
        # Newton steps on the observed information once the certificate refused
        # a stop near the mode or its refused score stalled (below); never on the
        # QR route (whose least squares cannot carry a zero-curvature row's
        # score) or the SCOP route (which takes its own Newton steps)
        self._mean_space_newton = False
        # Whether the true-score certificate decided the last iteration under
        # Newton steps, its ratio there, and its last residual: the diagnostics
        # then publish that rule's numbers, and ``PIRLSResult.mean_space_true_mode``
        # tells a criterion read at the state that it is the model's own mode.
        self._true_stop = False
        self._true_confirmed = False
        self._true_stop_ratio = math.inf
        self._last_true_residual: list[ModeResidual | None] = [None]
        # the decomposition this iteration's solve holds (gram and qr routes); the
        # latest certificate's judgement of the truncated directions
        # (``truncated_direction_ratio``) with the state it judged (its eta), and
        # the latest judgement that found any.  Only a judgement of the returned
        # state decides the fit's verdict; an earlier one is published as history
        # (below)
        self._held_rank: list = [None]
        self._judged_truncated: list[tuple] = [(None, ())]
        self._last_truncated: list[tuple] = [(None, ())]

        # The family's mean space, when the link's inverse can leave it (declared
        # by the family and link, ``irls_state.mean_space_violation``).
        self._mean_space_invalid = mean_space_violation(self.family, self.link)
        # The link inverse's own domain (``irls_state.link_domain_violation``).
        self._link_domain_invalid = link_domain_violation(self.link)
        # A warm start's centred state carried to this fit's centre (one-engine
        # design §3.8): alpha + (X - 1 c_warm') beta = alpha_c + (X - 1 c') beta
        # gives alpha_c = alpha + fsum((c - c_warm) beta), exact on every column
        # the two centres share.  Taken back from the raw intercept instead it
        # errs by u |c' beta| in every row's eta, ~0.25 at a 1e16 offset.
        centred_start: float | None = None
        if (
            self._centred_init is not None
            and _state_center is not None
            and self.beta_init is not None
            and self.intercept_init is not None
            and self.intercept == self.intercept_init
        ):
            warm_alpha, warm_centre = self._centred_init
            warm_centre = np.asarray(warm_centre, dtype=np.float64)
            if warm_centre.shape == _state_center.shape:
                centred_start = float(warm_alpha) + math.fsum(
                    (_state_center - warm_centre) * self.beta
                )
        # Freeze the fit-entry state so iteration-one trial safety has a baseline.
        committed = self.evaluate_state(
            self.beta,
            self.intercept,
            phase="initial",
            iteration=0,
            deviance=self._deviance_init,
            emit_trace=not self._has_scop,
            centred_intercept=centred_start,
        )
        if self._has_scop:
            committed = self._scop_initial_state(committed)
        self.recorder.emit_state_commit(
            committed,
            iteration=0,
            fit_converged=False,
            convergence_value=None,
            termination_reason=None,
        )
        self.objective_prev = (
            committed.deviance
            if committed.penalized_deviance is None
            else committed.penalized_deviance
        )
        self.dev_prev = committed.deviance
        self.eta_unclipped = committed.eta_unclipped
        self.eta = committed.eta
        self.mu = committed.mu

        self.max_halving = 20  # max step-halving attempts per iteration
        self._consecutive_svd = 0  # for auto-mode warning
        self._reported_qp_nonconvergence = (
            False  # transient note once; terminal authority is separate
        )
        self._degeneracy_retry_failed = (
            False  # the #472 retry is off after a failure, until a solve certifies
        )
        # A constrained fit-entry state has not been certified by the inner QP.
        self.retained_qp_converged = not self.has_constraints
        # Declared, not bound: the chain below is the whole set of reasons this
        # solver can end an iteration on, and declaring it against the shared
        # vocabulary is what ties the two together. A tenth reason invented here is
        # a type error on its own assignment until ``TerminationReason`` carries
        # it, so the consumers that switch on the field cannot be handed a string
        # they have never been shown.
        self.termination_reason: TerminationReason

        # Captured inside the loop, BEFORE ``dev_prev`` advances, because the
        # post-loop separation backstop needs the final iteration's movement.
        # Reading ``abs(dev - dev_prev)`` after the loop cannot work: ``dev_prev``
        # is assigned ``dev`` as the loop's last statement, so budget exhaustion
        # makes that difference identically zero and any stagnation test on it
        # vacuous.  ``inf`` until an iteration completes, so a fit that never
        # finishes one is never called stagnant.
        self.deviance_relative_change = float("inf")
        self.mode_bar, self.committed = mode_bar, committed

    def _scop_initial_state(self, committed: _IRLSState) -> _IRLSState:
        family, y, _scop_state = self.family, self.y, self._scop_state
        scop_committed = _SCOPTrialState(
            irls=committed,
            groups=tuple(
                _SCOPGroupState(
                    group_index=gi,
                    beta_eff=_immutable_array(st["beta_scop"]),
                    gamma_eff=_immutable_array(st["gamma_eff"]),
                    H_scop_penalized=(
                        None
                        if st.get("H_scop_penalized") is None
                        else _immutable_array(st["H_scop_penalized"])
                    ),
                    last_step_norm=float(st.get("last_step_norm", 0.0)),
                    last_fisher_fallback=bool(st.get("last_fisher_fallback", False)),
                    discarded_directions=_immutable_or_none(st.get("discarded_directions")),
                )
                for gi, st in sorted(_scop_state.items())
            ),
        )

        # Remove mapped diagonal SCOP penalties before evaluating the merit.
        # Subtracting them after a full quadratic introduces avoidable
        # cancellation; cross-group override entries keep their existing role.
        scop_outer_penalty = self.S.copy()
        for spec in self._scop_specs.values():
            scop_outer_penalty[spec.group.sl, spec.group.sl] = 0.0
        scop_outer_penalty_abs = np.abs(scop_outer_penalty)
        scop_merit_errors: dict[int, float] = {}
        merit_gamma = _gamma(len(y) + 3 * len(self.beta) + 4 * len(self._scop_specs) + 16)
        # For Gaussian deviance, (|y|+|mu|)^2 <= 8*y^2 + 2*(y-mu)^2.
        # This encloses residual formation without an observation-level
        # calculation on every trial. Other families supply deviance-unit
        # values; the guard covers their weighted accumulation and penalty.
        # Include gamma before squaring. The response action itself may be
        # outside binary64 even when its roundoff allowance is finite.
        with np.errstate(over="ignore", invalid="ignore"):
            gaussian_response_roundoff = (
                np.sum((np.sqrt(merit_gamma) * np.sqrt(self.weights) * y) ** 2)
                if type(family) is Gaussian
                else 0.0
            )
        # NB2's unit deviance errs by u (y + theta) whatever its size: it
        # takes the log of (y + theta) / (mu + theta), or of (mu + theta) /
        # theta where y = 0, a ratio within gamma_3 of its own, and multiplies
        # it back by y + theta. That puts at most 2 gamma_4 (2y + theta) on
        # each row (8.9e-8 at theta = 1e8), beside the rounding of the two
        # terms Poisson's unit deviance also has, which the deviance's own
        # allowance covers. A backtracking search finds a step only when its
        # allowance bounds the merit's noise (Berahas, Byrd & Nocedal 2019,
        # Theorem 4.2).
        nb2_unit_roundoff = (
            2.0 * _gamma(4) * float(np.sum(np.abs(self.weights) * (2.0 * y + family.theta)))
            if type(family) is NegativeBinomial
            else 0.0
        )
        self.scop_outer_penalty = scop_outer_penalty
        self.scop_outer_penalty_abs = scop_outer_penalty_abs
        self.scop_merit_errors, self.merit_gamma = scop_merit_errors, merit_gamma
        self.gaussian_response_roundoff = gaussian_response_roundoff
        self.nb2_unit_roundoff = nb2_unit_roundoff

        scop_committed = self.with_scop_merit(scop_committed)
        committed = scop_committed.irls
        # Relative change uses the current objective alone. An initial-merit
        # reference would weaken late convergence after a poor starting fit.
        self.objective_merit_scale = 0.0
        self.recorder.emit_evaluation(
            committed,
            phase="initial",
            iteration=0,
            deviance_reused=self._deviance_init is not None,
        )
        self.scop_committed = scop_committed
        return committed

    def with_scop_merit(self, trial: _SCOPTrialState) -> _SCOPTrialState:
        """Attach deviance plus the latent-coordinate quadratic penalty."""
        merit_gamma, lambda2 = self.merit_gamma, self.lambda2
        # A trial whose mapped coefficients carry its ordinary block past
        # binary64's range has a merit that is not finite, which the line
        # search refuses, without a warning (as ``_evaluate_scop_trial``).
        with np.errstate(over="ignore", invalid="ignore"):
            penalty_quad = float(trial.irls.beta @ self.scop_outer_penalty @ trial.irls.beta)
        magnitude = np.abs(trial.irls.beta)
        penalty_roundoff = _positive_quadratic_roundoff(
            magnitude, self.scop_outer_penalty_abs, merit_gamma
        )
        for group_state in trial.groups:
            group = self.groups[group_state.group_index]
            lam_scop = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
            latent_penalty = self._scop_specs[group_state.group_index].S_scop
            penalty_quad += float(
                lam_scop * (group_state.beta_eff @ latent_penalty @ group_state.beta_eff)
            )
            magnitude = np.abs(group_state.beta_eff)
            penalty_roundoff += _positive_quadratic_roundoff(
                magnitude, latent_penalty, merit_gamma, abs(lam_scop)
            )
        retained = replace(
            trial,
            irls=replace(
                trial.irls,
                penalized_deviance=float(trial.irls.deviance + penalty_quad),
            ),
        )
        deviance_roundoff = merit_gamma * abs(trial.irls.deviance) + self.nb2_unit_roundoff
        if type(self.family) is Gaussian:
            deviance_roundoff = 8.0 * self.gaussian_response_roundoff + 2.0 * deviance_roundoff
        with np.errstate(over="ignore", invalid="ignore"):
            allowance = float((deviance_roundoff + penalty_roundoff) / (1.0 - merit_gamma))
        self.scop_merit_errors[id(retained.irls)] = allowance
        return retained

    def _working(self, it: int) -> None:
        committed, family, link, y = self.committed, self.family, self.link, self.y
        weights, recorder = self.weights, self.recorder
        self._held_rank[0] = None
        self.beta_prev = committed.beta
        self.intercept_prev = committed.intercept
        self.beta = committed.beta.copy()
        self.intercept = committed.intercept
        self.committed_active_set = (
            None if self.prev_active_set is None else list(self.prev_active_set)
        )
        self.proposal_qp_converged = not self.has_constraints
        # W/z are rebuilt below from the committed coefficients.  Any
        # constrained-QP certificate retained from the preceding iteration
        # therefore belongs to different working weights and response, not to
        # this iteration's H/g.  Only a certified full proposal from the
        # current working problem can restore the flag.
        self.retained_qp_converged = not self.has_constraints
        self.rank_truncated: bool | None = None
        self.used_rank_certification = False
        self.scop_proposal_eta_unclipped: NDArray | None = None
        self.proposal_centred_intercept: float | None = None

        # Working quantities from current eta/mu (already computed)
        _t0 = time.perf_counter()
        working_rows = coefficient_working_rows(
            distribution=family,
            link=link,
            y=y,
            mu=self.mu,
            eta=self.eta,
            sample_weight=weights,
            prefer_observed=self._observed_newton_active,
        )
        W = working_rows.weights
        z = working_rows.response
        # binomial/log Newton rows once engaged (refusal block below): the
        # observed curvature and the score of the unclipped eta, the step in
        # score form.  ``z`` is then the predictor itself, so ``W (z - eta)``
        # adds nothing and the score enters on its own; a state holding rows
        # at the boundary keeps Fisher's rows.
        newton_score: NDArray | None = None
        newton_committed_rows: NDArray | None = None
        if self._mean_space_newton and not mean_space_boundary_rows(
            family, link, committed.eta_unclipped, weights
        ):
            newton_rows = mean_space_newton_rows(y, weights, committed.eta_unclipped)
            if newton_rows is not None:
                W, newton_score = newton_rows
                z = np.array(committed.eta_unclipped, dtype=np.float64)
                newton_committed_rows = mean_space_log_likelihood_rows(
                    y, weights, committed.eta_unclipped
                )
                recorder.count("irls_mean_space_newton_iters")
        if working_rows.rejection_reason is not None:
            # Every accepted trial of an observed fit had finite rows (the
            # line search rejects the others), so only the entry state can
            # get here: the declared curvature has no step to take.
            raise ValueError(
                "IRLS direct cannot start: the observed-Newton rows of "
                f"{type(family).__name__} with a {type(link).__name__} are not finite at "
                "the starting coefficients."
            )
        if working_rows.curvature_source == "observed":
            recorder.count("irls_observed_newton_iters")
        self.working_eta_unclipped = self.eta_unclipped
        self.working_eta = self.eta
        self.working_mu = self.mu
        # the separation backstop and the diagnostics read Fisher's weights,
        # the curvature their bars were set on, on a Newton iteration too
        positive_w_min, positive_w_max, w_ratio = _positive_working_weight_stats(
            working_rows.weights
        )
        recorder.t_working += time.perf_counter() - _t0

        if w_ratio > 1e12:
            logger.debug(
                "IRLS direct iter=%d: extreme W ratio %.1e (positive W range [%.2e, %.2e])",
                it + 1,
                w_ratio,
                positive_w_min,
                positive_w_max,
            )
        self.working_rows, self.W, self.z, self.newton_score = working_rows, W, z, newton_score
        self.newton_committed_rows, self.w_ratio = newton_committed_rows, w_ratio

    def _step_scop(self, it: int, z_off: NDArray, _t0: float) -> None:
        W, committed, scop_committed = self.W, self.committed, self.scop_committed
        _scop_state, _scop_order = self._scop_state, self._scop_order
        _scop_newton, _reduced_dm = self._scop_newton, self._reduced_dm
        _reduced_to_full = self._reduced_to_full
        # ── SCOP: one Newton step on every coefficient jointly ──
        # The ordinary block is solved with the latent SCOP
        # coordinates held fixed, then ``SCOPJointNewton.step``
        # couples the two blocks through the Schur complement and
        # corrects that solve (Pya & Wood 2015, Appendix 1).
        scop_thetas = [_scop_state[gi]["beta_scop"] for gi in _scop_order]
        z_adj = z_off - _scop_newton.predictor(scop_thetas)

        # Profile the intercept in stable centered
        # coordinates.  The raw augmented Gram loses the ordinary
        # slope entirely after a large column translation, so the
        # system comes from ``build_centered_system``, the authority
        # every other route of this solver uses as it stands: it
        # subtracts raw moments only under ``_raw_centering_well_scaled``
        # and otherwise centres before it multiplies (anchor-centred
        # compact supports, centred row chunks, a dense column's
        # corrected two-pass pair).
        reduced_centered = build_centered_system(
            dm=_reduced_dm,
            W=W,
            z_off=z_adj,
            penalty=self._S_reduced,
            tabmat_split=_reduced_dm.tabmat_centering_split,
            tabmat_state=self._reduced_tabmat_state,
            profile=self.profile,
        )
        if self._scop_run is not None:
            _carry_refusals(self._scop_run.reduced_refusals, self._reduced_tabmat_state)
        # The certificate, read on the system's own location and
        # spread, chooses only how the proposal's predictor is formed.  It no longer
        # rebuilds the system: a rejection here can only fall on a
        # column the builder centred first (a raw rung's columns
        # passed this same test on these same numbers), so the rebuild
        # recomputed a centred Gram by materialising the whole design
        # twice per iteration.  A bounded column trips it whenever its
        # weighted mean exceeds its centred RMS, e.g. a 0/1 indicator
        # of weighted frequency above 1/2.
        reduced_scale = np.sqrt(
            np.maximum(np.diag(reduced_centered.data_gram), 0.0) / reduced_centered.sum_w
        )
        centred_prediction = not _raw_centering_well_scaled(
            reduced_centered.mean_x,
            reduced_scale,
        )
        self.recorder.t_gram += time.perf_counter() - _t0

        # The ordinary block's solve, the latent coordinates held fixed
        _t0 = time.perf_counter()
        reduced_rank = decompose_gram_if_authoritative(reduced_centered.hessian)
        reduced_factor_rhs = None
        if reduced_rank is None:
            reduced_factor, certified_rhs = grouped_augmented_factor_rhs(
                _reduced_dm,
                W,
                reduced_centered.penalty,
                response=z_adj - reduced_centered.mean_z,
                center=reduced_centered.mean_x,
            )
            certified = decompose_factor(
                reduced_factor,
                retain_factor_solve=True,
            )
            reduced_rank = certified
            reduced_factor_rhs = certified_rhs
            self.used_rank_certification = True
        beta_reduced = (
            reduced_rank.solve(reduced_centered.rhs)
            if reduced_factor_rhs is None
            else reduced_rank.solve_factor_rhs(reduced_factor_rhs)
        )
        self._cond_est = reduced_rank.pre_truncation_condition
        self._used_svd = reduced_rank.used_svd_fallback
        self.rank_truncated = reduced_rank.rank_truncated
        joint = _scop_newton.step(
            scop_thetas,
            W=W,
            z=self.z,
            eta=committed.eta_unclipped,
            reduced=reduced_centered,
            solve=reduced_rank.solve,
            ordinary_change=beta_reduced - committed.beta[_reduced_to_full],
            debug_recorder=self.recorder.debug_recorder,
            debug_context={**self.recorder.base_debug_context, "pirls_iteration": it + 1},
        )
        scop_results = joint.results
        committed_reduced = committed.beta[_reduced_to_full]
        centre = reduced_centered.centre_pair()
        committed_level = committed.intercept + float(reduced_centered.mean_x @ committed_reduced)
        if centred_prediction and scop_committed.level is not None:
            # The committed level carried to this centre: its raw
            # intercept reads it at u |m' b|, and the trials start there.
            committed_level = _level_about(
                scop_committed.level, scop_committed.centre, centre, committed_reduced
            )
        self.scop_ordinary = _SCOPOrdinarySolve(
            columns=_reduced_to_full,
            committed=committed_reduced,
            committed_level=committed_level,
            fixed=beta_reduced,
            level=reduced_centered.mean_z,
            coupled=joint.coupled,
            mean=joint.mean,
            mean_x=reduced_centered.mean_x,
            centre=centre,
            centred_rows=(
                partial(_centred_system_matvec, _reduced_dm, system=reduced_centered)
                if centred_prediction
                else None
            ),
        )
        beta_reduced = beta_reduced + joint.ordinary_correction
        centred_level = reduced_centered.mean_z - joint.intercept_shift
        intercept = centred_level - float(reduced_centered.mean_x @ beta_reduced)
        proposal_thetas = [scop_results[gi].beta_new for gi in _scop_order]
        beta = np.zeros(self.p)
        beta[_reduced_to_full] = beta_reduced
        for gi, theta in zip(_scop_order, proposal_thetas, strict=True):
            beta[self.groups[gi].sl] = self._scop_specs[gi].reparam.forward(theta)
        self.recorder.t_solve += time.perf_counter() - _t0

        if centred_prediction:
            eta_unconstrained = centred_level + _centred_system_matvec(
                _reduced_dm, beta_reduced, reduced_centered
            )
        else:
            eta_unconstrained = intercept + _reduced_dm.matvec(beta_reduced)
        self.scop_proposal_eta_unclipped = (
            eta_unconstrained + self.offset + _scop_newton.predictor(proposal_thetas)
        )
        self.beta, self.intercept, self.centred_level = beta, intercept, centred_level
        self.scop_results = scop_results

    def _step_dense(self, z_off: NDArray, _t0: float) -> None:
        W, dm, newton_score, _state_center = self.W, self.dm, self.newton_score, self._state_center
        proposal_centred_intercept = self.proposal_centred_intercept
        centered = self.get_centered_system(W, z_off)
        if newton_score is not None:
            # the working mean of ``z_off`` is the centred intercept;
            # the score moves it by sum(u) / sum(W) and the slopes'
            # right-hand side by X~' u (the step itself is the increment
            # below; the stored system is the one it solved)
            centered = replace(
                centered,
                mean_z=centered.mean_z + float(np.sum(newton_score)) / centered.sum_w,
                rhs=centered.rhs + centred_data_score(dm, newton_score, centered.mean_x),
            )
        self._last_working_centered = centered
        self.recorder.t_gram += time.perf_counter() - _t0
        _t0 = time.perf_counter()
        iteration_rank = decompose_gram_if_authoritative(centered.hessian)
        iteration_factor_rhs = None
        if iteration_rank is None:
            certification = self.certify_centered_factor(
                centered,
                W,
                response=None if newton_score is not None else z_off - centered.mean_z,
            )
            certified = certification.decomposition
            if certification.transformed_rhs is None and newton_score is None:
                raise RuntimeError(  # pragma: no cover - invariant
                    "factor certification omitted its transformed RHS"
                )
            iteration_rank = certified
            iteration_factor_rhs = certification.transformed_rhs
            self.used_rank_certification = True
        if newton_score is not None:
            # the increment, not the iterate: a direction the factor
            # truncates keeps the iterate's component instead of the
            # minimum-norm reset of solving for it (Pes & Rodriguez
            # 2021, eqs. 1.2-1.4)
            beta = self.committed.beta + iteration_rank.solve(
                centred_data_score(dm, newton_score, centered.mean_x)
                - self.penalty_matvec(self.committed.beta)
            )
        else:
            beta = (
                iteration_rank.solve(centered.rhs)
                if iteration_factor_rhs is None
                else iteration_rank.solve_factor_rhs(iteration_factor_rhs)
            )
        intercept = centered.mean_z - float(centered.mean_x @ beta)
        _last_working_offset_mean = None
        if _state_center is not None:
            # the intercept about the state's centre, from the offset
            # of the working mean to it, formed on centred rows (the
            # system's pair, else ``centre_offset_mean``): no
            # raw-scale cancellation
            _last_working_offset_mean = _system_offset_mean(dm, W, centered, _state_center)
            proposal_centred_intercept = centered.mean_z - math.fsum(
                _last_working_offset_mean * beta
            )
            intercept = proposal_centred_intercept - math.fsum(_state_center * beta)
        self._cond_est = iteration_rank.pre_truncation_condition
        self._held_rank[0] = iteration_rank
        self._used_svd = iteration_rank.used_svd_fallback
        self.rank_truncated = iteration_rank.rank_truncated
        if self.convergence == "mode_score":
            self._score_centre = (
                centered.mean_x,
                centered.sum_w,
                np.diag(centered.data_gram),
                (),
                _last_working_offset_mean,
            )
        self.recorder.t_solve += time.perf_counter() - _t0
        self.beta, self.intercept = beta, intercept
        self._last_working_offset_mean = _last_working_offset_mean
        self.proposal_centred_intercept = proposal_centred_intercept

    def _step_qp(self, it: int, z_off: NDArray, _t0: float) -> None:
        newton_score, prev_active_set = self.newton_score, self.prev_active_set
        _degeneracy_retry_failed = self._degeneracy_retry_failed
        _reported_qp_nonconvergence = self._reported_qp_nonconvergence
        centered = self.get_centered_system(self.W, z_off)
        if newton_score is not None:
            centered = replace(
                centered,
                mean_z=centered.mean_z + float(np.sum(newton_score)) / centered.sum_w,
                rhs=centered.rhs + centred_data_score(self.dm, newton_score, centered.mean_x),
            )
        self._last_working_centered = centered
        self.recorder.t_gram += time.perf_counter() - _t0

        # Solve the constrained system in its existing coordinate space.
        _t0 = time.perf_counter()
        qp_result = _solve_constrained_qp_with_cold_retry(
            centered.hessian,
            centered.rhs,
            self.A_all,
            self.b_all,
            prev_active_set,
            degeneracy_retry=not _degeneracy_retry_failed,
        )
        beta = qp_result.beta
        self.intercept = centered.mean_z - float(centered.mean_x @ beta)
        self.proposal_qp_converged = bool(qp_result.converged)
        prev_active_set = qp_result.active_set
        # An uncertified result means the degenerate-vertex retry ran
        # and failed (or was already latched off). It costs up to a
        # full active-set solve, so it is not repeated while the QP
        # keeps failing. A certified result re-arms it: while latched,
        # that can only be an ordinary solve, so the failure was
        # transient and a later #472 vertex keeps its rescue.
        _degeneracy_retry_failed = not qp_result.converged
        # Non-convergence usually persists for the rest of the fit, so
        # latch the report to the first occurrence rather than emitting
        # one identical line per IRLS iteration. A later solve may
        # recover; the retained state's terminal authority is checked
        # independently below.
        if not qp_result.converged and not _reported_qp_nonconvergence:
            _reported_qp_nonconvergence = True
            logger.info(
                "fit_irls_direct: constrained QP did not converge at "
                "iteration %d; its KKT certificate is incomplete. "
                "A later IRLS iteration may recover; subsequent inner "
                "failures in this fit are not reported here.",
                it + 1,
            )
        # Report what the QP actually inverted.  These were hardcoded
        # to False/0.0, so the constrained branch published "perfectly
        # conditioned" to the consumers below while the unconstrained
        # branch above read the real numbers off its decomposition.
        self._used_svd = qp_result.used_svd_fallback
        self._cond_est = qp_result.condition
        self.recorder.t_solve += time.perf_counter() - _t0
        self.beta, self.prev_active_set = beta, prev_active_set
        self._degeneracy_retry_failed = _degeneracy_retry_failed
        self._reported_qp_nonconvergence = _reported_qp_nonconvergence

    def _search_scop(self, it: int) -> None:
        beta, intercept, scop_results = self.beta, self.intercept, self.scop_results
        _scop_specs, scop_ordinary = self._scop_specs, self.scop_ordinary
        scop_committed, dm, y, weights = self.scop_committed, self.dm, self.y, self.weights
        family, link, offset = self.family, self.link, self.offset
        recorder = self.recorder
        trace_enabled, trace_run = recorder.trace_enabled, recorder.trace_run
        trace_basis_id, resolved_lambdas = recorder.trace_basis_id, recorder.resolved_lambdas
        max_halving, scop_merit_errors = self.max_halving, self.scop_merit_errors
        emit_evaluation, with_scop_merit = recorder.emit_evaluation, self.with_scop_merit
        _t0 = time.perf_counter()
        # as ``_evaluate_scop_trial``: a proposal past the floating-point
        # range is rejected by the line search, without a warning
        with np.errstate(over="ignore", invalid="ignore"):
            proposal_irls = self.evaluate_state(
                beta,
                intercept,
                phase="scop_proposal",
                iteration=it + 1,
                alpha=1.0,
                eta_unclipped=self.scop_proposal_eta_unclipped,
                emit_trace=False,
            )
        proposal_scop = _SCOPTrialState(
            irls=proposal_irls,
            groups=tuple(
                _SCOPGroupState(
                    group_index=gi,
                    beta_eff=_immutable_array(scop_results[gi].beta_new),
                    gamma_eff=_immutable_array(
                        _scop_specs[gi].reparam.forward(scop_results[gi].beta_new)
                    ),
                    H_scop_penalized=(
                        None
                        if scop_results[gi].H_penalized is None
                        else _immutable_array(scop_results[gi].H_penalized)
                    ),
                    last_step_norm=float(scop_results[gi].step_norm),
                    last_fisher_fallback=bool(scop_results[gi].used_fisher_fallback),
                    discarded_directions=_immutable_or_none(scop_results[gi].discarded_directions),
                )
                for gi in sorted(_scop_specs)
            ),
            level=self.centred_level,
            centre=scop_ordinary.centre,
        )
        proposal_scop = with_scop_merit(proposal_scop)
        proposal_irls = proposal_scop.irls
        emit_evaluation(
            proposal_irls,
            phase="scop_proposal",
            iteration=it + 1,
            alpha=1.0,
        )
        scop_trial_cache: dict[float, _SCOPTrialState] = {1.0: proposal_scop}

        def evaluate_scop_trial(alpha: float) -> _IRLSState:
            if trace_enabled:
                assert trace_run is not None
                state_id = trace_run.next_state_id()
                evaluation_id = trace_run.next_evaluation_id()
            else:
                state_id = None
                evaluation_id = None
            candidate = _evaluate_scop_trial(
                committed=scop_committed,
                proposed=proposal_scop,
                alpha=alpha,
                specs=_scop_specs,
                dm=dm,
                y=y,
                weights=weights,
                family=family,
                link=link,
                offset=offset,
                state_id=state_id,
                evaluation_id=evaluation_id,
                basis_id=trace_basis_id,
                lambdas=resolved_lambdas,
                ordinary=scop_ordinary,
            )
            candidate = with_scop_merit(candidate)
            emit_evaluation(
                candidate.irls,
                phase="scop_line_search_trial",
                iteration=it + 1,
                alpha=alpha,
                enclosing_proposal_state_id=proposal_scop.irls.state_id,
            )
            scop_trial_cache[alpha] = candidate
            return candidate.irls

        # A step past ``_LATENT_MAX_STEP`` is searched from its first
        # halving within it, on the same halving grid, with the full
        # budget from there; the full step is not tried.
        latent_depth = _latent_step_depth(scop_committed, proposal_scop, _scop_specs)
        shortened = math.ldexp(1.0, -latent_depth)
        search_proposal = evaluate_scop_trial(shortened) if latent_depth else proposal_scop.irls
        decision = _select_irls_trial(
            committed=scop_committed.irls,
            proposal=search_proposal,
            evaluate_state=lambda alpha: evaluate_scop_trial(alpha * shortened),
            invalid_state=self.observed_rows_invalid if self._observed_newton_active else None,
            max_halving=max_halving,
            extended_max_halving=lambda: _poisson_sqrt_halving_budget(
                committed=scop_committed.irls,
                proposal=search_proposal,
                y=y,
                weights=weights,
                family=family,
                link=link,
                default=max_halving,
            ),
            merit_scale=0.0,
            merit_roundoff=lambda candidate, current: (
                scop_merit_errors[id(candidate)] + scop_merit_errors[id(current)]
            ),
        )
        if latent_depth:
            # in fractions of the full step, which was evaluated but not tried
            decision = replace(
                decision,
                trials_attempted=decision.trials_attempted + 1,
                **(
                    {}
                    if decision.step_rejected
                    else {
                        "alpha": decision.alpha * shortened,
                        "step_halvings": decision.step_halvings + latent_depth,
                    }
                ),
            )
        retained_scop = (
            scop_committed if decision.step_rejected else scop_trial_cache[decision.alpha]
        )
        retained = retained_scop.irls
        evaluation_elapsed = time.perf_counter() - _t0
        recorder.t_deviance += evaluation_elapsed
        recorder.t_deviance_eval += evaluation_elapsed
        beta = retained.beta.copy()
        intercept = retained.intercept
        self.eta_unclipped = retained.eta_unclipped
        self.eta = retained.eta
        self.mu = retained.mu
        dev = retained.deviance
        n_halvings = decision.step_halvings
        self.step_rejected = decision.step_rejected
        committed_groups = {group.group_index: group for group in scop_committed.groups}
        for group_state in retained_scop.groups:
            st = self._scop_state[group_state.group_index]
            st["beta_scop_prev"] = committed_groups[group_state.group_index].beta_eff.copy()
            st["beta_scop"] = group_state.beta_eff.copy()
            st["gamma_eff"] = group_state.gamma_eff.copy()
            st["H_scop_penalized"] = (
                None
                if group_state.H_scop_penalized is None
                else group_state.H_scop_penalized.copy()
            )
            st["last_step_norm"] = group_state.last_step_norm
            st["last_fisher_fallback"] = group_state.last_fisher_fallback
            st["discarded_directions"] = (
                None
                if group_state.discarded_directions is None
                else group_state.discarded_directions.copy()
            )
        if n_halvings:
            logger.info(
                "  irls_direct SCOP iter=%d: accepted latent step fraction %.5g after "
                "%d halvings, dev=%.2e",
                it + 1,
                decision.alpha,
                n_halvings,
                dev,
            )
        self.beta, self.intercept, self.dev, self.n_halvings = beta, intercept, dev, n_halvings
        self.decision, self.proposal_scop, self.retained = decision, proposal_scop, retained
        self.retained_scop = retained_scop

    def _search(self, it: int) -> None:
        beta, intercept, committed = self.beta, self.intercept, self.committed
        has_constraints, A_all, b_all = self.has_constraints, self.A_all, self.b_all
        abs_A_all, newton_score = self.abs_A_all, self.newton_score
        newton_committed_rows, y, weights = self.newton_committed_rows, self.y, self.weights
        family, link, _mean_space_invalid = self.family, self.link, self._mean_space_invalid
        _link_domain_invalid, _start_lowered = self._link_domain_invalid, self._start_lowered
        max_halving, penalty_matvec = self.max_halving, self.penalty_matvec
        evaluate_state, recorder = self.evaluate_state, self.recorder
        _t0 = time.perf_counter()
        proposal = evaluate_state(
            beta,
            intercept,
            phase="proposal",
            iteration=it + 1,
            alpha=1.0,
            centred_intercept=self.proposal_centred_intercept,
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
            centred_trial = None
            if committed.centred_intercept is not None and proposal.centred_intercept is not None:
                centred_trial = committed.centred_intercept + alpha * (
                    proposal.centred_intercept - committed.centred_intercept
                )
            candidate = evaluate_state(
                beta_trial,
                intercept_trial,
                phase="line_search_trial",
                iteration=it + 1,
                alpha=alpha,
                eta_unclipped=eta_trial,
                enclosing_proposal_state_id=proposal.state_id,
                centred_intercept=centred_trial,
            )
            trial_cache[alpha] = candidate
            return candidate

        committed_constraints_feasible = True
        proposal_constraints_feasible = True
        constraint_trial_is_invalid = (
            self.observed_rows_invalid if self._observed_newton_active else None
        )
        if has_constraints:
            if A_all is None or b_all is None:  # pragma: no cover - construction invariant
                raise RuntimeError("constrained fit omitted its aggregate constraint system")
            committed_constraints_feasible = _is_feasible(
                A_all,
                committed.beta,
                b_all,
                _QP_FEASIBILITY_TOL,
                abs_A=abs_A_all,
            )
            proposal_constraints_feasible = _is_feasible(
                A_all,
                proposal.beta,
                b_all,
                _QP_FEASIBILITY_TOL,
                abs_A=abs_A_all,
            )

            def constraint_trial_is_invalid(candidate: _IRLSState) -> bool:
                return not _is_feasible(
                    A_all,
                    candidate.beta,
                    b_all,
                    _QP_FEASIBILITY_TOL,
                    abs_A=abs_A_all,
                )

        # the last trial's rows, read by both the decrease and its
        # round-off allowance: one log1mexp pass per trial
        _trial_rows: list = [None, None]

        def _newton_rows_at(state: _IRLSState) -> NDArray:
            if state is committed and newton_committed_rows is not None:
                return newton_committed_rows
            if _trial_rows[0] is not state:
                _trial_rows[0] = state
                _trial_rows[1] = mean_space_log_likelihood_rows(y, weights, state.eta_unclipped)
            return _trial_rows[1]

        def _newton_deviance_delta(candidate: _IRLSState, base: _IRLSState) -> float:
            # each row's change first, then numpy's pairwise sum
            return float(-2.0 * np.sum(_newton_rows_at(candidate) - _newton_rows_at(base)))

        def _newton_merit_roundoff(candidate: _IRLSState, base: _IRLSState) -> float:
            # The true deviance's own magnitudes: the rows' sizes (the
            # log1mexp and product roundings, and the pairwise sum's
            # gamma_{log2 n + 8}, are within 64 eps of them for any n
            # below 2^50), plus each state's penalty part.
            size = float(np.sum(np.abs(_newton_rows_at(candidate))))
            size += float(np.sum(np.abs(_newton_rows_at(base))))
            for state in (candidate, base):
                if state.penalized_deviance is not None:
                    size += abs(float(state.penalized_deviance) - float(state.deviance))
            return 64.0 * _EPS_FLOAT * 2.0 * size

        trial_is_invalid = constraint_trial_is_invalid
        if _mean_space_invalid is not None and not _mean_space_invalid(
            committed.eta_unclipped, weights
        ):
            # A feasible committed state keeps every accepted trial in the
            # family's mean space (``mean_space_violation``).
            def trial_is_invalid(candidate: _IRLSState, other=constraint_trial_is_invalid) -> bool:
                assert _mean_space_invalid is not None
                if _mean_space_invalid(candidate.eta_unclipped, weights):
                    return True
                return other is not None and other(candidate)

        if _link_domain_invalid is not None and not _link_domain_invalid(
            committed.eta_unclipped, weights
        ):
            # A committed state inside the link's domain keeps every
            # accepted trial inside it (``link_domain_violation``).
            def trial_is_invalid(candidate: _IRLSState, other=trial_is_invalid) -> bool:
                assert _link_domain_invalid is not None
                if _link_domain_invalid(candidate.eta_unclipped, weights):
                    return True
                return other is not None and other(candidate)

        decision = _select_irls_trial(
            committed=committed,
            proposal=proposal,
            evaluate_state=evaluate_trial,
            invalid_state=trial_is_invalid,
            max_halving=max_halving,
            extended_max_halving=lambda: max(
                _poisson_sqrt_halving_budget(
                    committed=committed,
                    proposal=proposal,
                    y=y,
                    weights=weights,
                    family=family,
                    link=link,
                    default=max_halving,
                ),
                # a lowered start can sit far inside the space with a
                # proposal far outside it, past the ordinary budget
                _mean_space_halving_budget(
                    committed=committed,
                    proposal=proposal,
                    weights=weights,
                    family=family,
                    link=link,
                    default=max_halving,
                )
                if _start_lowered
                else max_halving,
            ),
            merit_scale=self.objective_merit_scale,
            # a Newton step is judged on the model's own deviance, which
            # still sees the rows clip_mu holds; Fisher's on the clipped one
            merit_delta=lambda candidate, base: _stable_penalized_deviance_delta(
                candidate,
                base,
                penalty_matvec,
                deviance_delta=None
                if newton_score is None
                else _newton_deviance_delta(candidate, base),
            ),
            merit_roundoff=None if newton_score is None else _newton_merit_roundoff,
            # a proposal that leaves the space: the halvings above the
            # boundary are refused without a deviance pass each
            first_halving=lambda: mean_space_first_halving(
                committed=committed,
                proposal=proposal,
                weights=weights,
                family=family,
                link=link,
            ),
        )
        if decision.skipped:
            recorder.count("irls_halvings_skipped", decision.skipped)
        if (
            has_constraints
            and decision.step_rejected
            and not committed_constraints_feasible
            and proposal_constraints_feasible
            and _state_is_finite(proposal)
        ):
            # The objective comparison is not allowed to prefer an
            # inadmissible state over a finite feasible QP proposal. Once
            # this proposal is committed, convexity of A beta >= b and the
            # invalid-state predicate above preserve feasibility on every
            # subsequent accepted line-search segment.
            decision = _IRLSStepDecision(
                alpha=1.0,
                step_halvings=0,
                step_rejected=False,
                trials_attempted=decision.trials_attempted,
            )
            logger.info(
                "  irls_direct iter=%d: accepted finite feasible constrained "
                "proposal after objective line search found no admissible trial",
                it + 1,
            )
        retained = committed if decision.step_rejected else trial_cache[decision.alpha]
        if has_constraints:
            # Rejection retains coefficients from the preceding working
            # problem, while damping retains an interpolation.  Neither
            # state is the current H/g solution whose stationarity and
            # dual feasibility the inner QP certified.
            self.retained_qp_converged = bool(
                not decision.step_rejected and decision.alpha == 1.0 and self.proposal_qp_converged
            )
        evaluation_elapsed = time.perf_counter() - _t0
        recorder.t_deviance += evaluation_elapsed
        recorder.t_deviance_eval += evaluation_elapsed
        beta = retained.beta.copy()
        intercept = retained.intercept
        self.eta_unclipped = retained.eta_unclipped
        self.eta = retained.eta
        self.mu = retained.mu
        dev = retained.deviance
        n_halvings = decision.step_halvings
        step_rejected = decision.step_rejected
        if step_rejected:
            self.prev_active_set = self.committed_active_set
        elif n_halvings:
            logger.info(
                "  irls_direct iter=%d: accepted step fraction %.5g after %d halvings, dev=%.2e",
                it + 1,
                decision.alpha,
                n_halvings,
                dev,
            )
        self.proposal, self.decision, self.retained, self.beta = proposal, decision, retained, beta
        self.intercept, self.dev, self.n_halvings = intercept, dev, n_halvings
        self.step_rejected = step_rejected

    def _converge(self, it: int) -> None:
        _has_scop, convergence, tol, beta = self._has_scop, self.convergence, self.tol, self.beta
        beta_prev, intercept, retained = self.beta_prev, self.intercept, self.retained
        _mode_ratios, _stagnation_window = self._mode_ratios, self._stagnation_window
        _last_mode_residual = self._last_mode_residual
        score_stagnated = self.score_stagnated
        self.proposal_state = self.proposal_scop.irls if _has_scop else self.proposal
        dev_rel_change = None
        coef_change = None
        self.convergence_label: str = convergence
        self.convergence_tolerance = tol
        if np.isfinite(self.dev):
            if convergence == "mode_score" and self._score_centre is not None:
                # One-engine design §3.8: stop on the certificate's own score,
                # in centred coordinates over the identified coefficients,
                # once it meets the bar the certificate reads (no margin: the
                # certificate is this residual, ``mode_score``).  The step
                # length, with its raw intercept, is only reported.
                coef_change = float(
                    np.max(np.abs(beta - beta_prev) / np.maximum(1.0, np.abs(beta)), initial=0.0)
                )
                residual = self.mode_residual(
                    beta, intercept, self.mu, self.eta, centred_intercept=retained.centred_intercept
                )
                convergence_value = self.mode_bar * residual.ratio()
                converged_this_iter = residual.ratio() <= 1.0
                _last_mode_residual = residual
                _mode_ratios.append(residual.ratio())
                # The score stopped contracting (``stagnation_window``):
                # the iterate is at its limiting accuracy, so the solve ends
                # uncertified and the caller discloses it.
                score_stagnated = (
                    not converged_this_iter
                    and residual.resolved
                    and len(_mode_ratios) > _stagnation_window
                    and min(_mode_ratios[-_stagnation_window:])
                    > 0.5 * min(_mode_ratios[:-_stagnation_window])
                )
                switched_curvature = False
                if (
                    self._stagnation_newton_available
                    and not self._observed_newton_active
                    and not converged_this_iter
                    and (
                        score_stagnated
                        or (
                            residual.resolved
                            and rate_exhausts_budget(
                                _mode_ratios, _stagnation_window, it + 1, self.max_iter
                            )
                        )
                    )
                ):
                    # Fisher scoring near the mode contracts too slowly to
                    # meet the bar in the budget, or not at all (the comment
                    # at ``_observed_newton_active``): the iterations left
                    # take Newton steps on the observed information, positive
                    # for Gamma/log (``w y / mu``), so the penalized Hessian
                    # stays positive definite.
                    self._observed_newton_active = True
                    self._can_reuse_weighted_gram = False
                    score_stagnated = False
                    switched_curvature = True
                    _mode_ratios.clear()
                    if self._newton_latch is not None:
                        self._newton_latch[0] = True
                    self.recorder.count("irls_fisher_newton_switches")
                if (
                    not converged_this_iter
                    and not switched_curvature
                    and not self.step_rejected
                    and not _has_scop
                    and not self.has_constraints
                    and intercept == self.intercept_prev
                    and retained.centred_intercept == self.committed.centred_intercept
                    and np.array_equal(beta, beta_prev)
                ):
                    # The full step left the state bitwise where it was: the
                    # iteration is a deterministic map at a float64 fixed
                    # point, and every further iteration repeats this one
                    # (the step-tolerance stop of Dennis & Schnabel 1996,
                    # section 7.2, at its strictest).  A centred state is
                    # ``(alpha, beta)``, and the raw intercept only its
                    # reading, which rounds ``alpha - c' beta`` at ``|c' beta|``
                    # and can hold still while ``alpha`` moves, so ``alpha``
                    # is compared too (``None`` on both sides otherwise).  An
                    # iteration that has just changed the map (the Newton
                    # switch above) is not one: the next iteration takes the
                    # other step.  A fixed point ends the solve, with the
                    # floors resolved whatever the relative score: where
                    # the residuals are at the rounding of the fitted values
                    # (a model that fits its response exactly, as a linear
                    # model can) the score's own scale ``sum |s|`` is
                    # rounding too, the relative score sits near 1 and the
                    # resolve cap would never let the floor be read.  A score
                    # within its floor is certified; one above it is the
                    # limiting accuracy, published as not converged.
                    residual = self.mode_residual(
                        beta,
                        intercept,
                        self.mu,
                        self.eta,
                        centred_intercept=retained.centred_intercept,
                        resolve_cap=math.inf,
                    )
                    convergence_value = self.mode_bar * residual.ratio()
                    converged_this_iter = residual.ratio() <= 1.0
                    score_stagnated = not converged_this_iter
                    _last_mode_residual = residual
                    _mode_ratios[-1] = residual.ratio()
            elif convergence in ("coefficients", "mode_score"):
                coef_change = float(
                    np.max(np.abs(beta - beta_prev) / np.maximum(1.0, np.abs(beta)))
                )
                coef_change = max(
                    coef_change,
                    abs(intercept - self.intercept_prev) / max(1.0, abs(intercept)),
                )
                if _has_scop:
                    latent_change = max(
                        float(
                            np.max(
                                np.abs(retained_group.beta_eff - committed_group.beta_eff)
                                / np.maximum(1.0, np.abs(retained_group.beta_eff))
                            )
                        )
                        for retained_group, committed_group in zip(
                            self.retained_scop.groups,
                            self.scop_committed.groups,
                            strict=True,
                        )
                    )
                    coef_change = max(coef_change, latent_change)
                convergence_value = coef_change
                converged_this_iter = coef_change < tol
                if _has_scop and self.n_halvings:
                    # a damped step's change is a fraction of the full
                    # step's and bounds nothing either (below)
                    converged_this_iter = False
            else:
                objective = (
                    retained.deviance
                    if retained.penalized_deviance is None
                    else retained.penalized_deviance
                )
                if np.isfinite(self.objective_prev):
                    dev_rel_change = _irls_objective_relative_change(
                        objective=objective,
                        previous=self.objective_prev,
                        objective_scale=self.objective_merit_scale,
                    )
                converged_this_iter = dev_rel_change is not None and dev_rel_change < tol
                if _has_scop and self.n_halvings:
                    # The decrease of a full joint Newton step is its Newton
                    # decrement, which bounds the distance to the mode (Boyd &
                    # Vandenberghe 2004, sec. 9.5.1); a damped step's is a
                    # fraction of it and bounds nothing, so it never stops the
                    # fit: the next full step's does.
                    converged_this_iter = False
                convergence_value = dev_rel_change
        else:
            converged_this_iter = False
            convergence_value = None
        if self.step_rejected:
            converged_this_iter = False
        self.dev_rel_change, self.coef_change = dev_rel_change, coef_change
        self.convergence_value, self.converged_this_iter = convergence_value, converged_this_iter
        self.score_stagnated, self._last_mode_residual = score_stagnated, _last_mode_residual

    def _judge_mean_space(self) -> None:
        _mean_space_newton, _stagnation_window = self._mean_space_newton, self._stagnation_window
        _true_ratios, converged_this_iter = self._true_ratios, self.converged_this_iter
        family, link, retained, weights = self.family, self.link, self.retained, self.weights
        score_stagnated = self.score_stagnated
        # Once Newton steps are engaged the model's own certificate is the
        # stop, every iteration: the clipped rule a mode_score fit stops on
        # cannot pass at a mode it does not see (below).
        newton_stop = bool(_mean_space_newton and np.isfinite(self.dev) and not self.step_rejected)
        # A state holding rows at the boundary is never published as converged
        # (``mean_space_boundary`` below), so its stop stands as on master.
        at_boundary = bool(
            (converged_this_iter or newton_stop)
            and self._mean_space_invalid is not None
            and mean_space_boundary_rows(family, link, retained.eta_unclipped, weights)
        )
        # rows clip_mu holds, at the step's base or where it landed
        clipped_rows = (
            mean_space_clipped_rows(family, link, self.committed.eta_unclipped, weights)
            + mean_space_clipped_rows(family, link, retained.eta_unclipped, weights)
            if converged_this_iter and self._mean_space_invalid is not None and not at_boundary
            else 0
        )
        _true_confirmed = False
        if not at_boundary and (
            newton_stop or (converged_this_iter and (clipped_rows or self._start_lowered))
        ):
            # On a row whose mean leaves clip_mu's band the clipped deviance
            # is flat and the step's score is not the binomial/log score, and
            # a lowered start can sit far below that band: a stop read off
            # the clipped objective, or off a step taken from it, is accepted
            # only where the model's own penalized score certifies it
            # (``true_mode_residual``).  A refused stop goes on.  Each refused
            # stop sits where the iteration's own rule passed, so a score that
            # has stopped contracting across them (``stagnation_window``) will
            # not reach the bar: the fit ends uncertified, as under
            # ``mode_score``.  Under Newton steps the certificate alone
            # decides, at every iteration, and its own ratios alone stagnate.
            true_ratio, true_resolved = self.true_mode_residual(
                retained.beta,
                retained.intercept,
                retained.eta_unclipped,
                (
                    self.A_all[list(self.prev_active_set)]
                    if self.has_constraints and self.A_all is not None and self.prev_active_set
                    else None
                ),
                self.retained_scop.groups if self._has_scop else None,
            )
            # the state is certified on the model's own score, Newton or not,
            # where the clip holds a row (or Newton decided): a criterion read
            # at it reads the model's own likelihood and curvature
            _true_confirmed = bool(true_ratio <= 1.0 and (newton_stop or clipped_rows))
            if newton_stop:
                converged_this_iter = bool(true_ratio <= 1.0)
                score_stagnated = False
                # this rule decided: its numbers are the ones published
                self.convergence_label = "mean_space_mode_score"
                self.convergence_value = self.mode_bar * true_ratio
                self.convergence_tolerance = self.mode_bar
                self._true_stop_ratio = true_ratio
            if not true_ratio <= 1.0:
                converged_this_iter = False
                _true_ratios.append(true_ratio)
                score_stagnated = len(_true_ratios) > _stagnation_window and not (
                    min(_true_ratios[-_stagnation_window:])
                    < 0.5 * min(_true_ratios[:-_stagnation_window])
                )
                if (
                    self._mean_space_newton_available
                    and not _mean_space_newton
                    and (true_resolved or score_stagnated)
                ):
                    # Fisher scoring has carried the fit to where its own
                    # stop passes but the model's score does not: near the
                    # mode (resolved), or where scoring has stalled.  The
                    # iterations left take Newton steps on the observed
                    # information (``irls_state.mean_space_newton_rows``),
                    # which converge quadratically where scoring is linear;
                    # far from the mode Fisher's steps stay the safer ones
                    # (Green 1984, section 2.3 and its discussion).
                    _mean_space_newton = True
                    _true_ratios.clear()
                    score_stagnated = False

        self._true_stop = bool(newton_stop and not at_boundary)
        self.converged_this_iter, self.score_stagnated = converged_this_iter, score_stagnated
        self._true_confirmed, self._mean_space_newton = _true_confirmed, _mean_space_newton

    def _terminate(self, it: int) -> None:
        converged_this_iter, step_rejected = self.converged_this_iter, self.step_rejected
        dev, max_iter = self.dev, self.max_iter
        constraints_feasible_this_iter = True
        if self.has_constraints:
            if (
                self.A_all is None or self.b_all is None
            ):  # pragma: no cover - construction invariant
                raise RuntimeError("constrained fit omitted its aggregate constraint system")
            constraints_feasible_this_iter = _is_feasible(
                self.A_all,
                self.beta,
                self.b_all,
                _QP_FEASIBILITY_TOL,
                abs_A=self.abs_A_all,
            )
            # Objective or coefficient stagnation cannot certify a constrained
            # mode whose retained coefficients are outside the feasible set.
            # Keep iterating while budget remains: a later QP/line-search step
            # may still repair an infeasible warm start.
            if not constraints_feasible_this_iter:
                converged_this_iter = False
            # Outer coefficient/deviance stagnation cannot replace the inner
            # QP's stationarity and dual-feasibility certificate. Keep a
            # finite feasible iterate and continue: a later full QP proposal
            # may still obtain the missing certificate.
            if not self.retained_qp_converged:
                converged_this_iter = False

        terminal_constraint_infeasible = bool(
            not constraints_feasible_this_iter
            and (step_rejected or not np.isfinite(dev) or it + 1 == max_iter)
        )
        terminal_constraint_kkt_incomplete = bool(
            self.has_constraints
            and not self.retained_qp_converged
            and (step_rejected or not np.isfinite(dev) or it + 1 == max_iter)
        )

        if terminal_constraint_infeasible:
            termination_reason = "constraint_infeasible"
        elif terminal_constraint_kkt_incomplete:
            termination_reason = "constraint_kkt_incomplete"
        elif step_rejected:
            termination_reason = "step_rejected"
        elif not np.isfinite(dev):
            termination_reason = "nonfinite_deviance"
        elif converged_this_iter:
            termination_reason = "converged"
        elif self.score_stagnated:
            termination_reason = "score_stagnated"
        elif it + 1 == max_iter:
            termination_reason = "max_iter"
        else:
            termination_reason = "continue"
        self.converged_this_iter, self.termination_reason = converged_this_iter, termination_reason

    def _finish(self, it: int) -> None:
        retained, family, link, beta = self.retained, self.family, self.link, self.beta
        weights, eta_unclipped, eta, mu = self.weights, self.eta_unclipped, self.eta, self.mu
        dev, converged, A_all, b_all = self.dev, self.converged, self.A_all, self.b_all
        termination_reason, _last_truncated = self.termination_reason, self._last_truncated

        # A published Gaussian identity fit carries its centred intercept as the
        # compensated pair (alpha, alpha_lo): alpha's weighted mean rounds by a
        # kernel-dependent ulp or more, and alpha + (x - c) beta ties at every row
        # when alpha* is the midpoint of two adjacent floats, so a one-ulp signal
        # between two levels collapsed on some BLAS kernels.  The deviance and
        # scale published below are the compensated predictor's, the eta every
        # consumer reads (``mode_score.linear_predictor``).
        centred_intercept_lo: float | None = None
        if (
            self._compensate_centred_intercept
            and self._compute_fit_statistics
            and retained.centred_intercept is not None
            and self._state_center is not None
            and type(family) is Gaussian
            and type(link) is IdentityLink
        ):
            contribution = centred_matvec(self.dm, beta, self._state_center)
            centred_intercept_lo = centred_intercept_remainder(
                self.y, weights, self.offset, retained.centred_intercept, contribution
            )
            if centred_intercept_lo is not None:
                contribution += centred_intercept_lo
                eta_unclipped = (retained.centred_intercept + contribution) + self.offset
                eta = stabilize_eta(eta_unclipped, link)
                mu = clip_mu(link.inverse(eta), family)
                dev = weighted_row_sum(weights, family.deviance_unit, self.y, mu)

        # Runtime separation backstop (issue #341).  Two terminal signatures mark
        # a coefficient that walked toward +/-infinity instead of converging:
        #
        # * the retained linear predictor is pinned at the link's overflow guard
        #   on rows that carry weight -- the guard, not the likelihood, stopped
        #   the walk, and it also manufactures "convergence" by freezing the
        #   coefficient once eta saturates;
        # * the budget ran out with a frozen deviance while the coefficient
        #   criterion still failed -- the field trace of issue #340/#341.
        #
        # Both require the extreme working-weight range that a drifting cell
        # produces.  The build-time scan names cell separation before fitting
        # starts; this catches what the design scan cannot see (non-categorical
        # indicator structure), promoting a debug-level log line to a named
        # warning or, in strict mode, an error.
        if self.separation != "ignore" and np.isfinite(dev):
            from superglm.diagnostics.separation import (
                EXTREME_WEIGHT_RATIO,
                STAGNANT_DEVIANCE_DELTA,
                SeparationError,
                SeparationWarning,
                drifting_groups,
                format_runtime_message,
            )

            separation_ratio = _separation_weight_ratio(
                self.working_rows.curvature_source,
                self.w_ratio,
                family=family,
                link=link,
                mu=self.working_mu,
                eta=self.working_eta,
                weights=weights,
            )
            if separation_ratio > EXTREME_WEIGHT_RATIO:
                pinned = bool(np.any((eta != eta_unclipped) & (weights > 0)))
                exhausted_stagnant = (
                    not converged
                    # A warm-started micro-budget solve (REML performance
                    # iterations run max_iter=1) exhausts its budget by
                    # construction; only a real budget makes exhaustion-with-
                    # stagnation evidence of a drift (the field trace ran 100).
                    and self.max_iter >= 10
                    and it + 1 >= self.max_iter
                    and not self.step_rejected
                    and self.deviance_relative_change < STAGNANT_DEVIANCE_DELTA
                )
                if pinned or exhausted_stagnant:
                    drifting = drifting_groups(
                        self.dm, self.groups, beta, eta, eta_unclipped, weights
                    )
                    message = format_runtime_message(separation_ratio, it + 1, drifting, pinned)
                    if self.separation == "error":
                        raise SeparationError(message)
                    # fit_irls_direct's frame: _finish, run, _fit_irls_direct_once, it
                    held_warnings.warn(message, SeparationWarning, stacklevel=4)

        if self.has_constraints:
            if A_all is None or b_all is None:  # pragma: no cover - construction invariant
                raise RuntimeError("constrained fit omitted its aggregate constraint system")
            if not _is_feasible(A_all, beta, b_all, _QP_FEASIBILITY_TOL, abs_A=self.abs_A_all):
                minimum_scaled_slack = float(np.min(_feasibility_slack(A_all, beta, b_all)))
                converged = False
                termination_reason = "constraint_infeasible"
                logger.warning(
                    "fit_irls_direct: retained coefficient mode violates hard constraints "
                    "(minimum scaled slack %.3e, tolerance %.1e); fit is not converged.",
                    minimum_scaled_slack,
                    _QP_FEASIBILITY_TOL,
                )
            elif not self.retained_qp_converged:
                converged = False
                termination_reason = "constraint_kkt_incomplete"
                logger.warning(
                    "fit_irls_direct: retained coefficient mode has no complete "
                    "constrained-QP KKT certificate; fit is not converged."
                )

        # A returned state with rows at the mean-space boundary is not a mode of
        # the model: their capped mean makes the deviance flat there, and the
        # maximum it approaches is constrained, not stationary.  It is never
        # converged, whatever stopped the loop (``mean_space_boundary_rows``).
        _boundary_rows = (
            0
            if self._mean_space_invalid is None
            else mean_space_boundary_rows(family, link, eta_unclipped, weights)
        )
        if _boundary_rows:
            converged = False
            termination_reason = "mean_space_boundary"
            logger.info(
                "fit_irls_direct: %d row(s) at the boundary of the family's mean space; "
                "the penalized maximum is constrained, fit is not converged.",
                _boundary_rows,
            )
        # The truncated directions published.  A judgement of the returned state
        # (the certificate read its eta) is the fit's own: published as it is, and
        # the only one that may decide the verdict.  The returned state of a
        # rejected step is the committed one, so its judgement counts.  When the
        # returned state was never judged, a fit that is not converged publishes
        # the latest judgement that found any as history (``earlier``), so a
        # refusal stays disclosed, and it decides nothing.
        _judged_eta, _judged = self._judged_truncated[0]
        if _judged_eta is not None and _judged_eta is eta_unclipped:
            _published_truncated = _judged
        elif converged:
            _published_truncated = ()
        else:
            _published_truncated = tuple(
                replace(record, earlier=True) for record in _last_truncated[0][1]
            )
        _boundary_sets = [
            record for record in _published_truncated if record.boundary and not record.earlier
        ]
        # rows rising to the boundary along a direction the factorization
        # truncates relabel only an exit the loop ran out on, never a
        # non-finite deviance or a constraint's verdict
        _truncated_boundary = bool(
            not converged
            and not _boundary_rows
            and _boundary_sets
            and termination_reason in ("max_iter", "score_stagnated", "step_rejected")
        )
        if _truncated_boundary:
            # rows whose supremum along a direction the factorization truncates
            # is the boundary (``TruncatedDirection.boundary``): the same verdict
            # as rows held at it, though float64 cannot carry them there
            termination_reason = "mean_space_boundary"
            _boundary_rows = sum(record.row_count for record in _boundary_sets)
            logger.info(
                "fit_irls_direct: %d row(s) rise towards the boundary of the family's mean "
                "space along a direction float64 cannot resolve; fit is not converged.",
                _boundary_rows,
            )
        self.centred_intercept_lo = centred_intercept_lo
        self.eta_unclipped, self.eta, self.mu, self.dev = eta_unclipped, eta, mu, dev
        self.converged, self.termination_reason = converged, termination_reason
        self._boundary_rows, self._published_truncated = _boundary_rows, _published_truncated
        self._truncated_boundary = _truncated_boundary

    def _terminal_rows(self) -> None:
        W, z, _scop_state = self.W, self.z, self._scop_state
        if self._has_scop:
            # Final Gram and SCOP Hessian caches must describe the retained model,
            # not the working state or a discarded full proposal.
            final_rows = coefficient_working_rows(
                distribution=self.family,
                link=self.link,
                y=self.y,
                mu=self.mu,
                eta=self.eta,
                sample_weight=self.weights,
                prefer_observed=False,
            )
            W = final_rows.weights
            z = final_rows.response

            # ``block_curvature`` takes no step, so however the loop exited (a
            # rejected last step included) the curvature, fallback flag and
            # discarded directions are the retained model's. The working residual
            # is read off the exact committed predictor; rebuilding X beta +
            # intercept would lose score digits for a translated ordinary column.
            terminal = self._scop_newton.block_curvature(
                [_scop_state[gi]["beta_scop"] for gi in self._scop_order],
                W=W,
                z=z,
                eta=self.eta_unclipped,
            )
            for gi, refresh in terminal.items():
                _scop_state[gi]["H_scop_penalized"] = refresh.H_penalized
                _scop_state[gi]["last_fisher_fallback"] = refresh.used_fisher_fallback
                _scop_state[gi]["discarded_directions"] = refresh.discarded_directions

        # Every public/exported matrix and rank claim is evaluated at the retained
        # model. Private fREML performance iterations deliberately reuse the
        # working system used for their one coefficient update.
        if not self._return_working_system:
            export_rows = coefficient_working_rows(
                distribution=self.family,
                link=self.link,
                y=self.y,
                mu=self.mu,
                eta=self.eta,
                sample_weight=self.weights,
                prefer_observed=False,
            )
            W = export_rows.weights
            z = export_rows.response
            self.export_rows = export_rows
        self.W, self.z = W, z

    def _final_system(self) -> None:
        _return_working_system, W, cache_out = self._return_working_system, self.W, self.cache_out
        dm = self.dm
        # Preserve the raw coefficient-space payload used by REML separately from
        # the centered inference system.
        if _return_working_system:
            if self._last_working_centered is None:
                raise RuntimeError("working centered system was not computed")
            centered_final = self._last_working_centered
            offset_mean_final = self._last_working_offset_mean
        else:
            z_off = self.z - self.offset
            centered_final = self.get_centered_system(W, z_off)
            offset_mean_final = (
                None
                if self._state_center is None or cache_out is None
                else _system_offset_mean(dm, W, centered_final, self._state_center)
            )
        XtWX, XtW1, XtWz, sum_Wz = centered_final.raw_weighted_moments()
        sum_W = centered_final.sum_w

        # Cache final-iteration raw and stable centered quantities for the cached-W
        # fREML optimizer. These allow re-solving the profiled-intercept system with
        # a new penalty matrix S without any data passes (O(p³), not O(n·K²)).
        if cache_out is not None:
            cache_out["XtWX"] = XtWX
            cache_out["centered_XtWX"] = centered_final.data_gram
            # the pair the Gram was centred about (``(mean_x, None)`` without a dense column)
            cache_out["centre_pair"] = centered_final.centre_pair()
            # mean_x less the state's centre, formed on centred rows: a cached
            # trial's centred intercept (``reml.discrete``) reads it
            cache_out["centre_offset_mean"] = offset_mean_final
            # the matrix the slope decomposition was taken of (the identified
            # part of the Laplace approximation restricts it, ``reml.identified``)
            cache_out["centered_hessian"] = centered_final.hessian
            # the rows its Gram summed: its formation bound's length
            # (``reml.discrete._profiled_formation_error``)
            cache_out["n_rows"] = int(dm.n)
            cache_out["XtWz"] = XtWz
            cache_out["XtW1"] = XtW1
            cache_out["sum_W"] = sum_W
            cache_out["sum_Wz"] = sum_Wz
            cache_out["centered_rhs"] = centered_final.rhs
            cache_out["mean_x"] = centered_final.mean_x
            cache_out["mean_z"] = centered_final.mean_z
            if self._has_scop:
                # The SCOP LAML mode certificate must evaluate the exact retained
                # predictor. Reconstructing a huge translated column plus its
                # compensating intercept can otherwise manufacture a false KKT
                # residual several orders above tolerance.
                cache_out["eta_unclipped"] = self.eta_unclipped
        self.centered_final, self.XtWX, self.XtW1, self.sum_W = centered_final, XtWX, XtW1, sum_W

    def _final_geometry(self) -> None:
        centered_final, W, dm, step = self.centered_final, self.W, self.dm, self.step
        _defer_rank_info = self._defer_rank_info
        # REML works in the full (intercept, slopes) coefficient space.  Profiling
        # the unpenalized intercept yields the centered Schur complement H_c.  Its
        # inverse is the slope block of H_aug^{-1}, while the unit-determinant
        # centering transform gives log|H_aug| = log(sum(W)) + log|H_c| at full
        # rank. With aliases, the same expression is the retained centered-space
        # determinant measure, not the raw augmented pseudo-determinant.
        _t0 = time.perf_counter()
        reml_geometry_summary: REMLGeometrySummary | None = None
        XtWX_beta = self.XtWX
        reml_slope_rank: RankDecomposition | None
        if self._compute_reml_geometry:
            reml_slope_rank = decompose_gram_if_authoritative(centered_final.hessian)
            if reml_slope_rank is None:
                certification = self.certify_centered_factor(
                    centered_final,
                    W,
                )
                reml_slope_rank = certification.decomposition
            XtWX_S_inv_beta = reml_slope_rank.pseudo_inverse()
            log_det_H = float(np.log(centered_final.sum_w) + reml_slope_rank.log_pdet)
            reml_hessian_rank = 1 + reml_slope_rank.rank
            # O(p) summary of the centered system for the REML gradient and
            # objective, so the loop can run without per-fit rank metadata.
            # column_scale matches decompose_gram's definition bit-for-bit.
            reml_geometry_summary = REMLGeometrySummary(
                mean_x=np.asarray(centered_final.mean_x, dtype=np.float64),
                sum_w=float(centered_final.sum_w),
                column_scale=np.sqrt(np.maximum(np.diag(centered_final.data_gram), 0.0)),
                mean_hi=centered_final.mean_hi,
                mean_lo=centered_final.mean_lo,
            )
        else:
            reml_slope_rank = None
            XtWX_S_inv_beta = np.empty((0, 0), dtype=np.float64)
            log_det_H = None
            reml_hessian_rank = None
        retained_reml_decomposition = reml_slope_rank if self._retain_reml_decomposition else None

        if self._compute_fit_statistics:
            if reml_slope_rank is None:  # pragma: no cover - validated above
                raise RuntimeError("fit statistics require generic REML geometry")
            augmented_rank = reml_slope_rank
            final_system = centered_final
            feature_edf = _feature_edf(
                centered_final,
                augmented_rank,
                lambda exponent: grouped_augmented_factor(
                    dm,
                    np.ldexp(W, -exponent),
                    np.ldexp(final_system.penalty, -exponent),
                    center=final_system.mean_x,
                ),
            )
            feature_edf[np.abs(feature_edf) < 100.0 * np.finfo(float).eps] = 0.0
            p_eff = 1.0 + float(np.sum(feature_edf))

            step_data_rank = step.data_rank
            retained_rank_info = partial(
                _retained_rank_info,
                centered_final,
                W,
                augmented_rank,
                feature_edf,
                dm=dm,
                groups=self.groups,
                p=self.p,
                step_data_rank=step_data_rank,
                _return_working_system=self._return_working_system,
                certify_centered_factor=self.certify_centered_factor,
            )

            if self.compute_rank_info or (
                _defer_rank_info is not None and step_data_rank is not None
            ):
                # A deferred step's own data rank would keep the step, and the
                # QR step's dense (n, p) design, alive until install; it is
                # computed here.
                rank_info = retained_rank_info()
            else:
                rank_info = None
                if _defer_rank_info is not None:
                    _defer_rank_info.append(retained_rank_info)
        else:
            p_eff = 0.0
            rank_info = None
        self.recorder.finalize_time(_t0)
        self.structured_factor = None
        self.reml_geometry_summary, self.XtWX_beta = reml_geometry_summary, XtWX_beta
        self.XtWX_S_inv_beta, self.log_det_H = XtWX_S_inv_beta, log_det_H
        self.reml_hessian_rank = reml_hessian_rank
        self.retained_reml_decomposition, self.p_eff = retained_reml_decomposition, p_eff
        self.rank_info = rank_info

    def _publish(
        self, it: int
    ) -> tuple[PIRLSResult, NDArray] | tuple[PIRLSResult, NDArray, NDArray]:
        groups = self.groups
        _compute_fit_statistics, family, y = self._compute_fit_statistics, self.family, self.y
        weights, p_eff, retained, dm, S = self.weights, self.p_eff, self.retained, self.dm, self.S
        XtWX_S_inv_beta = self.XtWX_S_inv_beta
        _resolved_direct_backend = self.step.backend
        self.recorder.disclose_backend(self, _resolved_direct_backend)

        # Pearson-based phi for estimated-scale families.  The numerator is the
        # same under either weight contract; only the denominator's likelihood
        # size distinguishes them.
        if _compute_fit_statistics and not getattr(family, "scale_known", True):
            pearson_sum = pearson_chi2(distribution=family, y=y, mu=self.mu, sample_weight=weights)
            df_resid = pearson_residual_degrees_of_freedom(
                weights,
                p_eff,
                weight_semantics=self.weight_semantics,
            )
            phi = pearson_sum / df_resid
        else:
            phi = 1.0
        if not self._compute_reml_geometry or not _compute_fit_statistics:
            # Private SCOP candidates and stats-skipping in-loop REML fits have no
            # retained-fit statistics. NaN makes accidental publication fail
            # visibly instead of presenting 0 EDF or unit dispersion as if either
            # had been evaluated; published statistics come from the terminal
            # refit, which always computes them.
            p_eff = float("nan")
            phi = float("nan")

        result = PIRLSResult(
            beta=self.beta,
            intercept=self.intercept,
            n_iter=it + 1,
            deviance=self.dev,
            converged=self.converged,
            phi=phi,
            effective_df=p_eff,
            iteration_log=self.recorder.iteration_log if self.record_diagnostics else None,
            log_det_H=self.log_det_H,
            reml_hessian_rank=self.reml_hessian_rank,
            reml_geometry=self.reml_geometry_summary,
            reml_slope_decomposition=self.retained_reml_decomposition,
            rank_info=self.rank_info,
            state_id=retained.state_id,
            evaluation_id=retained.evaluation_id,
            state_space=retained.state_space,
            basis_id=retained.basis_id,
            termination_reason=self.termination_reason,
            direct_backend=_resolved_direct_backend,
            direct_fallback_reason=self._direct_fallback_reason,
            centred_intercept=retained.centred_intercept,
            state_center=None if retained.centred_intercept is None else self._state_center,
            centred_intercept_lo=self.centred_intercept_lo,
            mean_space_boundary_rows=self._boundary_rows,
            mean_space_true_mode=bool(self.converged and self._true_confirmed),
            truncated_directions=self._published_truncated,
            mean_space_boundary_unresolved=self._truncated_boundary,
        )

        # Collect converged SCOP state for EFS outer loop and fit results.
        if self._has_scop:
            scop_converged = {}
            for gi, st in self._scop_state.items():
                scop_converged[gi] = {
                    "beta_eff": st["beta_scop"].copy(),
                    "H_scop_penalized": st.get("H_scop_penalized"),
                    "S_scop": st["S_scop"],
                    "B_scop": st["B_scop"],
                    "reparam": st["reparam"],
                    "gamma_eff": st.get("gamma_eff"),
                    "bin_idx": st.get("bin_idx"),
                    "group_sl": groups[gi].sl,
                    "group_name": groups[gi].name,
                    "last_step_norm": st.get("last_step_norm", 0.0),
                    "last_fisher_fallback": st.get("last_fisher_fallback", False),
                    "discarded_directions": st.get("discarded_directions"),
                    "penalty_rank": st.get("penalty_rank"),
                    "penalty_log_det_omega_plus": st.get("penalty_log_det_omega_plus"),
                    "penalty_eigvals_omega": st.get("penalty_eigvals_omega"),
                }
        else:
            scop_converged = None

        if (
            self._has_scop
            and scop_converged is not None
            and _compute_fit_statistics
            and self._compute_scop_postfit_inference
        ):
            import superglm.reml.scop_geometry as scop_geometry

            XtWX, XtW1, sum_W = self.XtWX, self.XtW1, self.sum_W
            centered_final = self.centered_final
            if self._scop_curvature == "observed":
                joint_geometry = scop_geometry.build_observed_scop_joint_geometry(
                    dm=dm,
                    distribution=family,
                    link=self.link,
                    y=y,
                    sample_weight=weights,
                    offset_arr=self.offset,
                    result=result,
                    penalty=S,
                    scop_states=scop_converged,
                    fisher_XtWX=XtWX,
                    fisher_XtW1=XtW1,
                    fisher_sum_W=sum_W,
                    centered_fisher_gram=centered_final.data_gram,
                    fisher_mean_x=centered_final.mean_x,
                    eta_unclipped=self.eta_unclipped,
                )
            else:
                joint_geometry = scop_geometry.build_cached_scop_joint_geometry(
                    raw_fisher_gram=XtWX,
                    fisher_xtw=XtW1,
                    fisher_sum_w=sum_W,
                    latent_penalty=S,
                    scop_states=scop_converged,
                    centered_fisher_gram=centered_final.data_gram,
                    fisher_mean_x=centered_final.mean_x,
                    dm=dm,
                    fisher_weights=self.W,
                )
            inference = scop_geometry.install_scop_postfit_inference(
                result,
                raw_fisher_gram=XtWX,
                centered_fisher_gram=centered_final.data_gram,
                fisher_xtw=XtW1,
                fisher_mean_x=centered_final.mean_x,
                fisher_sum_w=sum_W,
                latent_penalty=S,
                scop_states=scop_converged,
                groups=groups,
                observed_geometry=joint_geometry,
                dm=dm,
                fisher_weights=self.W,
            )

            # Estimated dispersion must use the same terminal EDF that downstream
            # covariance and summaries expose.  Known-scale likelihoods retain
            # their defining phi=1 rather than profiling a Pearson scale.
            if not getattr(family, "scale_known", True):
                pearson_sum = pearson_chi2(
                    distribution=family, y=y, mu=self.mu, sample_weight=weights
                )
                result.phi = pearson_sum / pearson_residual_degrees_of_freedom(
                    weights,
                    inference.total_edf,
                    weight_semantics=self.weight_semantics,
                )
        if self._expose_exact_support_state:
            result.scop_states = scop_converged

        if self.return_xtwx:
            if self.return_scop_state and scop_converged is not None:
                return result, XtWX_S_inv_beta, self.XtWX_beta, scop_converged
            return result, XtWX_S_inv_beta, self.XtWX_beta

        if self.return_scop_state and scop_converged is not None:
            return result, XtWX_S_inv_beta, scop_converged
        return result, XtWX_S_inv_beta
