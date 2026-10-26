"""The SCOP step's state and trials: shape-constrained groups in their latent coordinates.

``SCOPRunCentring`` is what a SCOP REML run carries across its coefficient fits.
A trial (``_SCOPTrialState``) is an ordinary IRLS state with every SCOP group's
latent coordinates (``_SCOPGroupState``); the line search interpolates them
(``_evaluate_scop_trial``), re-solving the ordinary block at each trial's own
latent coordinates (``_SCOPOrdinarySolve``), from a step first shortened to
``_LATENT_MAX_STEP`` (``_latent_step_depth``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

import superglm.solvers.scop_exact_support as scop_exact_support
from superglm.distributions import Distribution
from superglm.group_matrix import DesignMatrix
from superglm.links import Link
from superglm.solvers.centered_system import CenteredSystem, TabmatCenteringState
from superglm.solvers.irls_state import _evaluate_irls_state, _immutable_array, _IRLSState
from superglm.solvers.mode_score import centred_matvec
from superglm.solvers.scop_newton import SCOPGroupSpec

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
