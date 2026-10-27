"""The SCOP step, its state and trials: shape-constrained groups in their latent coordinates.

``_IRLSDirectSCOPStep`` is the fit's step method when it has a SCOP group.
``SCOPRunCentring`` is what a SCOP REML run carries across its coefficient fits.
A trial (``_SCOPTrialState``) is an ordinary IRLS state with every SCOP group's
latent coordinates (``_SCOPGroupState``); the line search interpolates them
(``_evaluate_scop_trial``), re-solving the ordinary block at each trial's own
latent coordinates (``_SCOPOrdinarySolve``), from a step first shortened to
``_LATENT_MAX_STEP`` (``_latent_step_depth``).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import superglm.solvers.scop_exact_support as scop_exact_support
from superglm._group_matrix._group_matrix_centered import _raw_centering_well_scaled
from superglm._roundoff import gamma as _gamma
from superglm.distributions import Distribution, Gaussian, NegativeBinomial
from superglm.group_matrix import DesignMatrix, DiscretizedSCOPGroupMatrix
from superglm.links import Link
from superglm.solvers.centered_system import (
    CenteredSystem,
    TabmatCenteringState,
    build_centered_system,
    grouped_augmented_factor_rhs,
)
from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom
from superglm.solvers.irls_direct._finalise import _finalise_centred
from superglm.solvers.irls_direct._step import _IRLSDirectStep, _note_svd_fallback
from superglm.solvers.irls_state import (
    _evaluate_irls_state,
    _immutable_array,
    _IRLSState,
    _poisson_sqrt_halving_budget,
    _select_irls_trial,
)
from superglm.solvers.mode_score import centred_matvec
from superglm.solvers.rank import decompose_factor, decompose_gram_if_authoritative
from superglm.solvers.scop_newton import (
    SCOPGroupSpec,
    SCOPJointNewton,
    _positive_quadratic_roundoff,
)
from superglm.solvers.working_rows import coefficient_working_rows, pearson_chi2

if TYPE_CHECKING:
    from superglm.solvers.irls_direct._finalise import _IRLSDirectFinalGeometry
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit
    from superglm.solvers.pirls import PIRLSResult

# the package's logger: its records keep the name they had when the solver was one module
logger = logging.getLogger(__package__)

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
    eta_clock: Callable[[float], None] | None = None,
) -> _SCOPTrialState:
    """Evaluate a fixed-endpoint SCOP trial by interpolating latent state.

    With ``ordinary``, the trial's ordinary coefficients and intercept are
    that block's solve at the trial's mapped coefficients
    (``_SCOPOrdinarySolve``); otherwise they are interpolated as the latent
    coordinates are. The groups are in the joint step's latent order.
    ``eta_clock``, when given, is handed the seconds spent forming the
    trial's eta and mean (``_evaluate_irls_state``).
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
                started = time.perf_counter()
                latent_columns = beta_trial.copy()
                latent_columns[ordinary.columns] = 0.0
                eta_unclipped = (
                    level + ordinary.centred_rows(solved) + dm.matvec(latent_columns) + offset
                )
                if eta_clock is not None:
                    eta_clock(time.perf_counter() - started)
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
            eta_clock=eta_clock,
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


class _IRLSDirectSCOPStep(_IRLSDirectStep):
    """The fit's step when it has shape-constrained (SCOP) groups: one Newton step on every coefficient jointly.

    Chosen where the route is decided (``_IRLSDirectFit._setup_routes``): a fit
    with a SCOP group has no linear constraint (``fit_irls_direct`` refuses
    the mix) and takes neither the QR nor the structured route.  A SCOP
    group's coefficients are its latent ``theta``, mapped by the group's
    reparameterisation; the other groups form the reduced (ordinary) design.
    Each iteration solves the ordinary block with ``theta`` held, then
    ``SCOPJointNewton.step`` couples the two blocks through the Schur
    complement and corrects that solve (Pya & Wood 2015, Appendix 1).  Its own
    line search interpolates the latent coordinates on a merit with its own
    rounding allowance, re-solving the ordinary block at each damped trial's
    latent coordinates; the stop rules read its latent change, and a damped
    step never stops the fit.  It finalises on the centred route, with its own
    terminal curvature and post-fit inference.  It keeps the latent state, the
    reduced design and the Newton engine, the merit's allowances, and the
    iteration's committed and retained latent states, and no reference to the
    fit, whose state each method is handed.
    """

    __slots__ = (
        # setup
        "_run _state _specs _order _newton _reduced_dm _reduced_to_full _S_reduced"
        " _reduced_tabmat_state _expose_exact_support _curvature _lambda2 _gaussian _columns"
        # the merit (``initial_state``)
        " _outer_penalty _outer_penalty_abs _merit_errors _merit_gamma"
        " _gaussian_response_roundoff _nb2_unit_roundoff"
        # one iteration, from ``propose`` through ``search`` to the stop rules
        " _committed _retained _ordinary _centred_level _results _proposal_eta_unclipped"
    ).split()
    backend = "gram"
    finalise = _finalise_centred

    def __init__(self, fit: _IRLSDirectFit, curvature: str) -> None:
        """The latent state, the reduced design and the Newton engine, and the first latent coordinates.

        ``curvature`` is the SCOP REML curvature class
        (``classify_scop_reml_curvature``), which the post-fit inference reads.
        It writes two of the fit's fields: the first latent coordinates' mapped
        coefficients into ``beta``, and ``_scop_run``, the REML run's centring
        state where the fit shares it, whose full-design refusals the fit's
        centred system carries.
        """
        groups, gms, n, lambda2 = fit.groups, fit.gms, fit.n, fit.lambda2
        # the columns the binomial/log certificate tests in the latent
        # coordinates instead (``certificate_inputs``)
        columns = np.zeros(fit.p, dtype=bool)
        for g in groups:
            if g.monotone_engine == "scop":
                columns[g.sl] = True
        self._columns = columns
        _scop_run = None
        _n_scop_groups = sum(g.monotone_engine == "scop" for g in groups)
        _expose_exact_support_state = False
        # group_idx -> {beta_scop, beta_scop_prev, reparam, B_scop, S_scop}
        _scop_state: dict[int, dict] = {}
        # A REML run's fits share one reduced design, their rung refusals and
        # their exact supports (``SCOPRunCentring``); a traced or debugged fit
        # stays fresh, as the Newton engine's carried policy does.
        if (
            fit._scop_run_centring is not None
            and fit.debug_recorder is None
            and fit.trace_run is None
        ):
            _scop_run = fit._scop_run_centring
            _scop_run.bind((fit.dm, fit.family, fit.link, *gms, *groups))
        for gi, g in enumerate(groups):
            if g.monotone_engine == "scop":
                cached_scop_state = (
                    fit.scop_state_init.get(gi) if fit.scop_state_init is not None else None
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
        _S_reduced = fit.S[np.ix_(_non_scop_cols, _non_scop_cols)]
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
        provisional = fit.evaluate_state(
            fit.beta,
            fit.intercept,
            phase="scop_initialization",
            iteration=0,
        )
        initial_rows = coefficient_working_rows(
            distribution=fit.family,
            link=fit.link,
            y=fit.y,
            mu=provisional.mu,
            eta=provisional.eta,
            sample_weight=fit.weights,
            prefer_observed=False,
        )
        W_init = initial_rows.weights
        z_init = initial_rows.response
        z_off_init = z_init - fit.offset
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
            fit.beta[groups[gi].sl] = gamma_eff
        self._run, self._state, self._specs, self._order = (
            _scop_run,
            _scop_state,
            _scop_specs,
            _scop_order,
        )
        self._newton, self._reduced_dm, self._reduced_to_full = (
            _scop_newton,
            _reduced_dm,
            _reduced_to_full,
        )
        self._S_reduced, self._reduced_tabmat_state = _S_reduced, _reduced_tabmat_state
        self._expose_exact_support, self._curvature = _expose_exact_support_state, curvature
        self._lambda2, self._gaussian = lambda2, type(fit.family) is Gaussian
        fit._scop_run = _scop_run

    def initial_state(self, fit: _IRLSDirectFit, committed: _IRLSState) -> _IRLSState:
        """The fit-entry state with its latent bundle and merit, traced; the merit's allowances are fixed here."""
        family, y, _scop_state = fit.family, fit.y, self._state
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
        scop_outer_penalty = fit.S.copy()
        for spec in self._specs.values():
            scop_outer_penalty[spec.group.sl, spec.group.sl] = 0.0
        scop_outer_penalty_abs = np.abs(scop_outer_penalty)
        scop_merit_errors: dict[int, float] = {}
        merit_gamma = _gamma(len(y) + 3 * len(fit.beta) + 4 * len(self._specs) + 16)
        # For Gaussian deviance, (|y|+|mu|)^2 <= 8*y^2 + 2*(y-mu)^2.
        # This encloses residual formation without an observation-level
        # calculation on every trial. Other families supply deviance-unit
        # values; the guard covers their weighted accumulation and penalty.
        # Include gamma before squaring. The response action itself may be
        # outside binary64 even when its roundoff allowance is finite.
        with np.errstate(over="ignore", invalid="ignore"):
            gaussian_response_roundoff = (
                np.sum((np.sqrt(merit_gamma) * np.sqrt(fit.weights) * y) ** 2)
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
            2.0 * _gamma(4) * float(np.sum(np.abs(fit.weights) * (2.0 * y + family.theta)))
            if type(family) is NegativeBinomial
            else 0.0
        )
        self._outer_penalty, self._outer_penalty_abs = scop_outer_penalty, scop_outer_penalty_abs
        self._merit_errors, self._merit_gamma = scop_merit_errors, merit_gamma
        self._gaussian_response_roundoff = gaussian_response_roundoff
        self._nb2_unit_roundoff = nb2_unit_roundoff

        scop_committed = self._with_merit(scop_committed)
        committed = scop_committed.irls
        # Relative change uses the current objective alone. An initial-merit
        # reference would weaken late convergence after a poor starting fit.
        fit.objective_merit_scale = 0.0
        fit.recorder.emit_evaluation(
            committed,
            phase="initial",
            iteration=0,
            deviance_reused=fit._deviance_init is not None,
        )
        # the state the first iteration commits from (``propose``)
        self._retained = scop_committed
        return committed

    def _with_merit(self, trial: _SCOPTrialState) -> _SCOPTrialState:
        """Attach deviance plus the latent-coordinate quadratic penalty, and record the merit's rounding allowance by the state's ``id``."""
        merit_gamma, lambda2, specs = self._merit_gamma, self._lambda2, self._specs
        # A trial whose mapped coefficients carry its ordinary block past
        # binary64's range has a merit that is not finite, which the line
        # search refuses, without a warning (as ``_evaluate_scop_trial``).
        with np.errstate(over="ignore", invalid="ignore"):
            penalty_quad = float(trial.irls.beta @ self._outer_penalty @ trial.irls.beta)
        magnitude = np.abs(trial.irls.beta)
        penalty_roundoff = _positive_quadratic_roundoff(
            magnitude, self._outer_penalty_abs, merit_gamma
        )
        for group_state in trial.groups:
            spec = specs[group_state.group_index]
            group = spec.group
            lam_scop = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
            latent_penalty = spec.S_scop
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
        deviance_roundoff = merit_gamma * abs(trial.irls.deviance) + self._nb2_unit_roundoff
        if self._gaussian:
            deviance_roundoff = 8.0 * self._gaussian_response_roundoff + 2.0 * deviance_roundoff
        with np.errstate(over="ignore", invalid="ignore"):
            allowance = float((deviance_roundoff + penalty_roundoff) / (1.0 - merit_gamma))
        self._merit_errors[id(retained.irls)] = allowance
        return retained

    def propose(self, fit: _IRLSDirectFit, it: int) -> None:
        """The ordinary solve with the latent coordinates held, then the joint Newton step."""
        _t0 = time.perf_counter()
        z_off = fit.z - fit.offset
        W, committed, recorder = fit.W, fit.committed, fit.recorder
        # The loop commits the state the last search retained, or else ends
        # (``_IRLSDirectFit.run``), so that is this iteration's committed state.
        self._committed = scop_committed = self._retained
        _scop_state, _scop_order = self._state, self._order
        _scop_newton, _reduced_dm = self._newton, self._reduced_dm
        _reduced_to_full = self._reduced_to_full
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
            profile=fit.profile,
        )
        if self._run is not None:
            _carry_refusals(self._run.reduced_refusals, self._reduced_tabmat_state)
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
        recorder.t_gram += time.perf_counter() - _t0

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
            fit.used_rank_certification = True
        beta_reduced = (
            reduced_rank.solve(reduced_centered.rhs)
            if reduced_factor_rhs is None
            else reduced_rank.solve_factor_rhs(reduced_factor_rhs)
        )
        fit._cond_est = reduced_rank.pre_truncation_condition
        fit._used_svd = reduced_rank.used_svd_fallback
        fit.rank_truncated = reduced_rank.rank_truncated
        joint = _scop_newton.step(
            scop_thetas,
            W=W,
            z=fit.z,
            eta=committed.eta_unclipped,
            reduced=reduced_centered,
            solve=reduced_rank.solve,
            ordinary_change=beta_reduced - committed.beta[_reduced_to_full],
            debug_recorder=recorder.debug_recorder,
            debug_context={**recorder.base_debug_context, "pirls_iteration": it + 1},
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
        self._ordinary = _SCOPOrdinarySolve(
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
        beta = np.zeros(fit.p)
        beta[_reduced_to_full] = beta_reduced
        for gi, theta in zip(_scop_order, proposal_thetas, strict=True):
            beta[fit.groups[gi].sl] = self._specs[gi].reparam.forward(theta)
        recorder.t_solve += time.perf_counter() - _t0

        if centred_prediction:
            eta_unconstrained = centred_level + _centred_system_matvec(
                _reduced_dm, beta_reduced, reduced_centered
            )
        else:
            eta_unconstrained = intercept + _reduced_dm.matvec(beta_reduced)
        self._proposal_eta_unclipped = (
            eta_unconstrained + fit.offset + _scop_newton.predictor(proposal_thetas)
        )
        fit.beta, fit.intercept = beta, intercept
        self._centred_level, self._results = centred_level, scop_results
        _note_svd_fallback(fit)

    def search(self, fit: _IRLSDirectFit, it: int) -> None:
        """The latent line search: trials interpolate the latent coordinates from the committed state."""
        beta, intercept, scop_results = fit.beta, fit.intercept, self._results
        _scop_specs, scop_ordinary = self._specs, self._ordinary
        scop_committed, dm, y, weights = self._committed, fit.dm, fit.y, fit.weights
        family, link, offset = fit.family, fit.link, fit.offset
        recorder, curvature = fit.recorder, fit.curvature
        trace_enabled, trace_run = recorder.trace_enabled, recorder.trace_run
        trace_basis_id, resolved_lambdas = recorder.trace_basis_id, recorder.resolved_lambdas
        max_halving, scop_merit_errors = fit.max_halving, self._merit_errors
        emit_evaluation, with_scop_merit = recorder.emit_evaluation, self._with_merit
        _t0 = time.perf_counter()
        # as ``_evaluate_scop_trial``: a proposal past the floating-point
        # range is rejected by the line search, without a warning
        with np.errstate(over="ignore", invalid="ignore"):
            proposal_irls = fit.evaluate_state(
                beta,
                intercept,
                phase="scop_proposal",
                iteration=it + 1,
                alpha=1.0,
                eta_unclipped=self._proposal_eta_unclipped,
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
            level=self._centred_level,
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
                eta_clock=recorder.eta_formed,
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
            invalid_state=curvature.observed_rows_invalid if curvature.observed else None,
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
        fit.eta_unclipped = retained.eta_unclipped
        fit.eta = retained.eta
        fit.mu = retained.mu
        dev = retained.deviance
        n_halvings = decision.step_halvings
        fit.step_rejected = decision.step_rejected
        committed_groups = {group.group_index: group for group in scop_committed.groups}
        for group_state in retained_scop.groups:
            st = self._state[group_state.group_index]
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
        fit.beta, fit.intercept, fit.dev, fit.n_halvings = beta, intercept, dev, n_halvings
        fit.decision, fit.proposal_state, fit.retained = decision, proposal_scop.irls, retained
        self._retained = retained_scop

    def latent_change(self) -> float:
        """The largest relative change in any group's latent coordinates, retained against committed."""
        return max(
            float(
                np.max(
                    np.abs(retained_group.beta_eff - committed_group.beta_eff)
                    / np.maximum(1.0, np.abs(retained_group.beta_eff))
                )
            )
            for retained_group, committed_group in zip(
                self._retained.groups,
                self._committed.groups,
                strict=True,
            )
        )

    def damped(self, fit: _IRLSDirectFit) -> bool:
        """A damped step never stops the fit.

        The decrease of a full joint Newton step is its Newton decrement, which
        bounds the distance to the mode (Boyd & Vandenberghe 2004, sec.
        9.5.1); a damped step's decrease, and its change in the coefficients,
        is a fraction of the full step's and bounds nothing, so it never stops
        the fit: the next full step's does.
        """
        return fit.n_halvings > 0

    def certificate_inputs(self) -> tuple:
        """The SCOP columns, and the retained iterate's latent groups, each with its spec.

        The binomial/log certificate tests those columns in the groups' latent
        coordinates instead (``_IRLSDirectMeanSpaceOverlay.certify``).
        """
        specs = self._specs
        return (
            None,
            None,
            self._columns,
            tuple((group, specs[group.group_index]) for group in self._retained.groups),
        )

    def terminal(self, fit: _IRLSDirectFit) -> None:
        """The SCOP Hessian caches at the retained model, from its Fisher rows.

        They must describe the retained model, not the working state or a
        discarded full proposal.  ``block_curvature`` takes no step, so however
        the loop exited (a rejected last step included) the curvature,
        fallback flag and discarded directions are the retained model's. The
        working residual is read off the exact committed predictor; rebuilding
        X beta + intercept would lose score digits for a translated ordinary
        column.
        """
        _scop_state = self._state
        final_rows = coefficient_working_rows(
            distribution=fit.family,
            link=fit.link,
            y=fit.y,
            mu=fit.mu,
            eta=fit.eta,
            sample_weight=fit.weights,
            prefer_observed=False,
        )
        terminal = self._newton.block_curvature(
            [_scop_state[gi]["beta_scop"] for gi in self._order],
            W=final_rows.weights,
            z=final_rows.response,
            eta=fit.eta_unclipped,
        )
        for gi, refresh in terminal.items():
            _scop_state[gi]["H_scop_penalized"] = refresh.H_penalized
            _scop_state[gi]["last_fisher_fallback"] = refresh.used_fisher_fallback
            _scop_state[gi]["discarded_directions"] = refresh.discarded_directions

    def postfit(
        self, fit: _IRLSDirectFit, result: PIRLSResult, final: _IRLSDirectFinalGeometry
    ) -> dict[int, dict]:
        """The latent states the fit returns, the retained predictor, and the post-fit inference.

        The latent states are what the EFS outer loop and the fit's result
        read.  The SCOP LAML mode certificate must evaluate the exact retained
        predictor (``cache_out["eta_unclipped"]``): reconstructing a huge
        translated column plus its compensating intercept can otherwise
        manufacture a false KKT residual several orders above tolerance.  A
        fit with statistics and post-fit inference installs the covariance and
        EDF on ``result`` from the centred finaliser's moments, and an
        estimated dispersion from that EDF.
        """
        groups, family, y, weights = fit.groups, fit.family, fit.y, fit.weights
        cache_out = fit.cache_out
        if cache_out is not None:
            cache_out["eta_unclipped"] = fit.eta_unclipped
        scop_converged = {}
        for gi, st in self._state.items():
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

        if fit._compute_fit_statistics and fit._compute_scop_postfit_inference:
            import superglm.reml.scop_geometry as scop_geometry

            dm, S = fit.dm, fit.S
            centered_final = final.centered_final
            XtWX, XtW1, sum_W = final.XtWX_beta, final.XtW1, centered_final.sum_w
            if self._curvature == "observed":
                joint_geometry = scop_geometry.build_observed_scop_joint_geometry(
                    dm=dm,
                    distribution=family,
                    link=fit.link,
                    y=y,
                    sample_weight=weights,
                    offset_arr=fit.offset,
                    result=result,
                    penalty=S,
                    scop_states=scop_converged,
                    fisher_XtWX=XtWX,
                    fisher_XtW1=XtW1,
                    fisher_sum_W=sum_W,
                    centered_fisher_gram=centered_final.data_gram,
                    fisher_mean_x=centered_final.mean_x,
                    eta_unclipped=fit.eta_unclipped,
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
                    fisher_weights=fit.W,
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
                fisher_weights=fit.W,
            )

            # Estimated dispersion must use the same terminal EDF that downstream
            # covariance and summaries expose.  Known-scale likelihoods retain
            # their defining phi=1 rather than profiling a Pearson scale.
            if not getattr(family, "scale_known", True):
                pearson_sum = pearson_chi2(
                    distribution=family, y=y, mu=fit.mu, sample_weight=weights
                )
                result.phi = pearson_sum / pearson_residual_degrees_of_freedom(
                    weights,
                    inference.total_edf,
                    weight_semantics=fit.weight_semantics,
                )
        if self._expose_exact_support:
            result.scop_states = scop_converged
        return scop_converged
