"""A fit's existence decision: whether its penalised estimate is finite, decided exactly.

``decide_separation`` is the one decision (``diagnostics.separation`` holds its mathematics): the
fit's score proves existence where it can (``existence_certified``, Stiemke's alternative), and
otherwise the rows decide it (``separation.separating_direction``, the linear program of Correia,
Guimaraes & Zylkin 2019, App. A.2, verified on the rows exactly).  A certified separation is
published as the fit on the rows it keeps (Correia et al., Prop. 3; ``withheld_refit``), whose
own score must prove that the rows it keeps separate no further (``maximal``); a decision
float64 cannot make is ``"unresolved"``.  The record lives on the fit's identifiability record
(``Identifiability.separation``), so ``summary()``, ``metrics()`` and the runtime separation
disclosure read one decision (``published_separation``).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast, get_args

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import SQRT_EPS, gamma_up
from superglm.diagnostics.separation import (
    Separation,
    exact_null_basis,
    exact_row_signs,
    recession_excluded,
    response_boundaries,
    separating_direction,
)


def separation_boundaries(model) -> tuple[str, ...]:
    """The response boundaries the fit's family and link reach only at infinite ``eta``."""
    return response_boundaries(model._distribution, model._link)


def withheld_weights(model, weights: NDArray | None = None) -> NDArray:
    """The fit's prior weights with the rows a certified separation withholds at zero."""
    weights = np.asarray(model._fit_weights if weights is None else weights, dtype=np.float64)
    withheld = getattr(model, "_separated_rows", None)
    if withheld is None or not len(withheld):
        return weights
    kept = weights.copy()
    kept[np.asarray(withheld, dtype=np.intp)] = 0.0
    return kept


def published_separation(model) -> Separation | None:
    """The published fit's existence decision, or ``None`` when the fit made none."""
    from superglm.model.state_ops import retained_linear_state

    state = retained_linear_state(model)
    record = getattr(state, "identifiability", None)
    if record is None:
        rank_info = getattr(model._solver_pirls_result(), "rank_info", None)
        record = getattr(rank_info, "identifiability", None)
    return getattr(record, "separation", None)


def fisher_rows(model, solver, weights: NDArray) -> tuple[NDArray, NDArray]:
    """``(W, z - eta)``: the Fisher working rows at ``solver``'s fitted state, prior ``weights``."""
    from superglm._weight_units import solver_weights
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.solvers.mode_score import linear_predictor
    from superglm.solvers.working_rows import coefficient_working_rows

    eta = stabilize_eta(linear_predictor(model._dm, solver, model._fit_offset), model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    rows = coefficient_working_rows(
        distribution=model._distribution,
        link=model._link,
        y=np.asarray(model._fit_y_ref, dtype=np.float64),
        mu=mu,
        eta=eta,
        sample_weight=solver_weights(weights, solver),
        prefer_observed=False,
    )
    W = np.asarray(rows.weights, dtype=np.float64)
    return W, np.asarray(rows.response, dtype=np.float64) - eta


def existence_certified(
    model,
    *,
    solver,
    sides: NDArray,
    weights: NDArray,
    decomposition,
    columns: NDArray,
    design,
    penalty: Callable[[NDArray], tuple[NDArray, NDArray]],
    data_diagonal: NDArray,
    floor: float,
    weak_root: NDArray | None = None,
    rows: tuple[NDArray, NDArray] | None = None,
    centre: NDArray | None = None,
) -> bool:
    """Whether the fit's score proves its penalised estimate exists (``recession_excluded``).

    At the fitted state, with Fisher rows ``W``, ``z`` (one consistent set: the rows the score,
    the step and the defect all read, ``W`` the Hessian's own), the profiled score ``g = X_c' r
    - S beta``, ``r = W (z - eta)``, is tried as it stands (``v = 0``, whose defect is ``g``
    itself), and otherwise takes one Newton step ``v``: the decomposition's inverse on
    its certified part and the weak directions' ``U U' g`` (their covariance root,
    ``WeakDirections.root``), with the intercept's ``sum r / sum W - c'v``.  Its residual ``r' =
    r - W A v`` gives the boundary multipliers ``mu_i = s_i r'_i`` and the defect ``A'r' - S
    (beta + v)``, computed directly (so any ``v`` serves) and Jacobi-scaled: the intercept by
    ``sqrt(sum W)``, a slope by its Hessian diagonal's root.  The defect's own rounding is
    bounded by Cauchy-Schwarz, ``sum_i |x_ij r'_i| <= ||sqrt(W) x_j|| sqrt(sum r'^2 / W)``
    (``gamma_(n+p+4)`` per product sum, the penalty's ``gamma_(p+2) |S| |beta + v|``); the
    intercept's recursive sum ``sum r'`` errs by at most ``gamma_(n-1) sum |r'_i| <=
    gamma_(n-1) sqrt(sum W) sqrt(sum r'^2 / W)`` (Higham 2002, sec. 4.2), inside its
    ``gamma_(n+2)`` term once scaled.  ``floor`` is the record's certified curvature
    (``Identifiability.curvature_floor``); with the intercept's block of one, ``min(1,
    floor)`` bounds the augmented Hessian's Jacobi eigenvalues.  Any row on the boundary with
    ``W = 0`` (a mean rounded onto the boundary) leaves no multiplier: not certified, so the
    rows decide.  ``rows`` are the publication's own ``fisher_rows`` (the array its record was
    decided at), and ``centre`` the weighted column mean that record's Gram was centred at;
    either is formed here when not given.  Refinement stops once the defect is its own
    rounding, which no refined step can lower.
    """
    boundary = (sides == 1) | (sides == -1)
    if not np.any(boundary):
        return True
    if not floor > 0.0:
        return False
    W, residual = fisher_rows(model, solver, weights) if rows is None else rows
    live = sides != 2
    W = np.where(live, W, 0.0)
    if np.any(W[boundary] <= 0.0) or not np.all(np.isfinite(residual[live])):
        return False
    r = np.where(live, W * residual, 0.0)
    total = float(np.sum(W))
    if not total > 0.0:
        return False
    if centre is None:
        mean = np.asarray(design.rmatvec(W), dtype=np.float64) / total
    else:
        mean = np.asarray(centre, dtype=np.float64)
    root = np.sqrt(W[boundary])
    signs = sides[boundary]
    beta = np.asarray(solver.beta, dtype=np.float64)[columns]
    applied, _ = penalty(beta[:, None])

    def solve(vector: NDArray) -> NDArray:
        step = np.asarray(decomposition.solve(vector), dtype=np.float64)
        if weak_root is not None and weak_root.shape[1]:
            step = step + weak_root @ (weak_root.T @ vector)
        return step

    def residual_rows(step: NDArray, intercept: float) -> NDArray:
        delta = intercept + np.asarray(design.matvec(step), dtype=np.float64)
        return np.where(live, r - W * delta, 0.0)

    def defect(moved_rows: NDArray, step: NDArray) -> tuple[float, NDArray]:
        e0 = float(np.sum(moved_rows))
        moved, _ = penalty((beta + step)[:, None])
        es = np.asarray(design.rmatvec(moved_rows), np.float64) - mean * e0 - moved[:, 0]
        return e0, es

    rsum = float(np.sum(r))
    gradient = np.asarray(design.rmatvec(r), dtype=np.float64) - mean * rsum - applied[:, 0]
    step, intercept = np.zeros_like(gradient), 0.0
    scale = np.asarray(decomposition.column_scale, dtype=np.float64)
    safe = np.where(scale > 0.0, scale, 1.0)
    carried = W > 0.0
    uncentred = np.sqrt(np.maximum(np.asarray(data_diagonal, np.float64), 0.0) + total * mean**2)
    n, p = int(design.n), int(beta.size)
    curvature = min(1.0, float(floor))
    # the fitted state itself first (``v = 0``: its defect is the score, already formed, so a
    # converged fit costs no further pass over the rows), then the Newton step and its
    # iterative refinement (Wilkinson), each taken only when the certificate needs it: the step
    # solves ``H v = g`` only to the coupling's and the weak directions' accuracy; any ``v``
    # serves the certificate, and a refined one leaves a smaller defect
    for refinement in range(4):
        moved_rows = residual_rows(step, intercept) if refinement else r
        margin = float(np.min(signs * moved_rows[boundary] / root))
        e0, es = defect(moved_rows, step) if refinement else (rsum, gradient)
        _, magnitude = penalty((beta + step)[:, None])
        zeta = math.sqrt(float(np.sum(moved_rows[carried] ** 2 / W[carried])))
        rounding = float(
            np.linalg.norm(
                np.r_[
                    gamma_up(n + 2) * zeta,
                    (
                        2.0 * gamma_up(n + p + 4) * uncentred * zeta
                        + gamma_up(p + 2) * magnitude[:, 0]
                    )
                    / safe,
                ]
            )
        )
        scaled = float(np.linalg.norm(np.r_[e0 / math.sqrt(total), es / safe]))
        if margin > 0.0 and recession_excluded(margin, scaled + rounding, curvature):
            return True
        if refinement == 3 or scaled <= rounding:
            return False
        correction = solve(es)
        step = step + correction
        intercept = intercept + e0 / total - float(mean @ correction)
    return False


def structured_floor(factor, hessian_diagonal: NDArray) -> float:
    """A certified lower bound on a structured factor's Jacobi-scaled eigenvalues on its kept part.

    ``lambda_min(H~) >= 1 / tr(H~^+)``, ``tr(H~^+) = sum_j H_jj (H^+)_jj`` from the factor's
    selected inverse diagonal.  The computed inverse is that of ``H + dH`` with ``||dH~|| <=
    gamma_(16 p) p`` (backward stability of the block factor, Higham 2002, sec. 10.1, on the
    Jacobi-scaled ``H~`` whose norm is at most its trace ``p``); while ``8 p gamma_(16 p) tr``
    stays under one the true trace is at most twice the computed, so half its reciprocal is a
    bound.  Zero (no bound) otherwise.
    """
    size = int(np.size(hessian_diagonal))
    if size == 0:
        return 1.0
    inverse = np.asarray(factor.selected_inverse_diagonal(np.arange(size)), dtype=np.float64)
    trace = float(np.sum(np.maximum(np.asarray(hessian_diagonal, np.float64), 0.0) * inverse))
    if not (trace > 0.0 and math.isfinite(trace)) or 8 * size * gamma_up(16 * size) * trace >= 1:
        return 0.0
    return 0.5 / trace


def explicit_design(model) -> tuple[Any, NDArray]:
    """The rows' columns the exact test reads: the intercept and the unpenalised stored groups.

    A group enters when its construction is its own coordinates (``ConstructionColumns.unit``,
    no penalty reaches it) and its matrix holds its entries (one-hot codes, dense or sparse
    columns): every product with a direction is then exact to evaluate (``exact_row_signs``).
    Returns the CSR matrix (intercept first) and each column's fitted coordinate (``-1`` the
    intercept).  A smooth's penalty null space is not stored by its entries, so a separation
    along it cannot be verified exactly and is unresolved.
    """
    from scipy import sparse

    from superglm._group_matrix._group_matrix_core import (
        CategoricalGroupMatrix,
        DenseGroupMatrix,
        SparseGroupMatrix,
    )
    from superglm.model.state_ops import _construction_columns

    n = int(model._dm.n)
    construction = _construction_columns(model, list(model._groups))
    blocks = [sparse.csr_matrix(np.ones((n, 1)))]
    coordinates = [-1]
    for index, (group, matrix) in enumerate(
        zip(model._groups, model._dm.group_matrices, strict=True)
    ):
        own = construction(index)
        if own is None or own.blocks != 1 or not own.unit:
            continue
        if type(matrix) is CategoricalGroupMatrix:
            mask = matrix.codes < matrix.n_levels
            block = sparse.csr_matrix(
                (np.ones(int(mask.sum())), (np.flatnonzero(mask), matrix.codes[mask])),
                shape=(n, matrix.n_levels),
            )
        elif isinstance(matrix, DenseGroupMatrix):
            block = sparse.csr_matrix(np.asarray(matrix.M, dtype=np.float64))
        elif type(matrix) is SparseGroupMatrix:
            block = sparse.csr_matrix(matrix.M, dtype=np.float64)
        else:
            continue
        blocks.append(block)
        coordinates.extend(range(group.start, group.end))
    return sparse.hstack(blocks, format="csr"), np.asarray(coordinates, dtype=np.intp)


@dataclasses.dataclass
class WithheldFit:
    """The fit on the rows a certified separation keeps, before it is published."""

    result: Any
    state: Any
    rows: NDArray


def withheld_refit(model, rows: NDArray) -> WithheldFit:
    """The penalised fit at the fit's smoothing parameters with ``rows`` withheld (Prop. 3).

    The rows' prior weights are set to zero and the direct solver resumes from the fitted
    coefficients: its smoothing parameters, penalties and backend are the fit's, so a
    structured fit keeps its retained state (built from the refit's system, as the terminal
    REML refit builds it).  The coefficients along the separated span are restored by the
    caller (``publish_withheld``), since the refit cannot see them.
    """
    from superglm.model.fit_state import fitted_lambda2
    from superglm.model.reml_finalize import _build_structured_linear_system_state
    from superglm.solvers.dispersion import model_weight_semantics
    from superglm.solvers.irls_direct import fit_irls_direct

    solver = model._solver_pirls_result()
    weights = np.asarray(model._fit_weights, dtype=np.float64).copy()
    weights[np.asarray(rows, dtype=np.intp)] = 0.0
    cache: dict = {}
    output = fit_irls_direct(
        X=model._dm,
        y=np.asarray(model._fit_y_ref, dtype=np.float64),
        weights=weights,
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        lambda2=fitted_lambda2(model),
        offset=model._fit_offset,
        beta_init=np.asarray(solver.beta, dtype=np.float64),
        intercept_init=float(solver.intercept),
        max_iter=max(int(getattr(model, "_max_iter", 100)), 100),
        tol=min(float(getattr(model, "_tol", 1e-8)), 1e-10),
        return_xtwx=True,
        cache_out=cache,
        direct_solve=getattr(model, "_direct_solve", "auto"),
        reml_penalties=getattr(model, "_reml_penalties", None),
        weight_semantics=model_weight_semantics(model),
        separation="ignore",
        trace_purpose="separation_refit",
    )
    result, factor, xtwx = cast(tuple[Any, Any, Any], output)
    state = None
    if getattr(model, "_linear_system_state", None) is not None and "structured_system" in cache:
        reporting = getattr(model, "_reporting_support_state", None)
        state = _build_structured_linear_system_state(
            factor=factor,
            data_operator=xtwx,
            cache=cache,
            support_totals={} if reporting is None else dict(reporting.support_totals),
        )
    return WithheldFit(result=result, state=state, rows=np.asarray(rows, dtype=np.intp))


def aliases_exact(model, gauge, sides: NDArray) -> bool:
    """Whether every null direction of ``gauge`` on the explicit columns is exact lineality.

    A null direction (``Gauge.null`` with its intercept component) that lies on the explicit
    columns (``explicit_design``) must vanish on every live row exactly
    (``separation.exact_null_basis``): a direction the rows move by any amount, however small,
    may separate them (a near-duplicate one unit in the last place apart), so the certificate
    cannot set it aside.  A direction off the explicit columns (a smooth's) is 6a's alias at
    float64, whose rows cannot be evaluated exactly; the existence question along it is the
    alias's.
    """
    explicit, coordinates = explicit_design(model)
    null = np.vstack((np.asarray(gauge.intercept)[None, :], np.asarray(gauge.null)))
    if not null.shape[1]:
        return True
    rows = np.where(coordinates < 0, 0, coordinates + 1)
    off = np.ones(null.shape[0], dtype=bool)
    off[rows] = False
    largest = np.max(np.abs(null), axis=0)
    on_explicit = ~np.any(np.abs(null[off]) > SQRT_EPS * largest, axis=0)
    if not np.any(on_explicit):
        return True
    live = np.flatnonzero(sides != 2)
    return exact_null_basis(explicit, null[np.ix_(rows, on_explicit)], live) is not None


def separated_span(explicit, coordinates: NDArray, null: NDArray, rows: NDArray, kept: NDArray):
    """The exact span the withheld fit leaves null, in explicit columns, and its separated part.

    ``null`` (``(1 + width) x k``, ``[intercept; slopes]``) spans the withheld fit's null
    directions.  They must lie on the explicit columns (a share above the square root of eps
    off them leaves them unverifiable: ``None``) and vanish exactly on the ``kept`` rows
    (``separation.exact_null_basis``).  Returns ``(basis, separated)``: the exact basis in
    explicit columns, and the columns of it that are nonzero on a separated row exactly.
    """
    width = null.shape[0]
    explicit_rows = np.where(coordinates < 0, 0, coordinates + 1)
    off = np.ones(width, dtype=bool)
    off[explicit_rows] = False
    largest = np.max(np.abs(null), axis=0) if null.size else np.zeros(0)
    if np.any(np.abs(null[off]) > SQRT_EPS * largest):
        return None
    basis = exact_null_basis(explicit, null[explicit_rows], kept)
    if basis is None:
        return None
    moved = [
        j for j in range(basis.shape[1]) if np.any(exact_row_signs(explicit[rows], basis[:, j]))
    ]
    return basis, basis[:, moved]


def lift_span(span: NDArray, coordinates: NDArray, width: int) -> NDArray:
    """Explicit-column directions as ``[intercept; slopes]`` in fitted coordinates."""
    lifted = np.zeros((width + 1, span.shape[1]))
    lifted[np.where(coordinates < 0, 0, coordinates + 1)] = span
    return lifted


def separated_labels(
    model, rows: NDArray, sides: NDArray, coordinates: NDArray
) -> tuple[tuple[str, ...], int]:
    """The categorical levels (base levels included) whose live rows are all separated.

    One entry per factor, ``"level E of f"`` or ``"levels D, E of f"``, and the count of the
    separated rows those levels hold.
    """
    from superglm._group_matrix._group_matrix_core import CategoricalGroupMatrix

    separated = np.zeros(sides.size, dtype=bool)
    separated[rows] = True
    live = sides != 2
    specs = {**getattr(model, "_specs", {}), **getattr(model, "_interaction_specs", {})}
    labels = []
    held = np.zeros(sides.size, dtype=bool)
    explicit = set(int(c) for c in coordinates if c >= 0)
    for group, matrix in zip(model._groups, model._dm.group_matrices, strict=True):
        if type(matrix) is not CategoricalGroupMatrix or group.start not in explicit:
            continue
        spec = specs.get(group.feature_name)
        names = list(getattr(spec, "_non_base", ())) + [getattr(spec, "_base_level", "base")]
        codes = [
            int(code)
            for code in np.unique(matrix.codes[separated])
            if code < len(names) and np.all(separated[live & (matrix.codes == code)])
        ]
        if codes:
            held |= separated & np.isin(matrix.codes, codes)
            plural = "levels" if len(codes) > 1 else "level"
            labels.append(f"{plural} {', '.join(str(names[c]) for c in codes)} of {group.name}")
    return tuple(labels), int(np.count_nonzero(held))


def coefficient_names(model, span: NDArray) -> tuple[str, ...]:
    """The names of the coefficients ``span`` (``[intercept; slopes]``) moves, intercept first."""
    from superglm.solvers.identifiability import _coordinate_name

    specs = {**getattr(model, "_specs", {}), **getattr(model, "_interaction_specs", {})}
    names = ["Intercept"] if np.any(span[0] != 0.0) else []
    owner = {}
    for group in model._groups:
        for coordinate in range(group.start, group.end):
            owner[coordinate] = group
    for coordinate in np.flatnonzero(np.any(span[1:] != 0.0, axis=1)):
        names.append(_coordinate_name(int(coordinate), owner[int(coordinate)], specs))
    return tuple(names)


def proposal_span(direction: NDArray | None, coordinates: NDArray, width: int) -> NDArray:
    """An unverified proposal as a span whose entries below the square root of eps are cut."""
    if direction is None:
        return np.zeros((width + 1, 0))
    lifted = lift_span(direction[:, None], coordinates, width)
    largest = float(np.max(np.abs(lifted), initial=0.0))
    lifted[np.abs(lifted) <= SQRT_EPS * largest] = 0.0
    return lifted


def decide_separation(
    model, *, sides: NDArray, boundaries: tuple[str, ...], certified: bool
) -> tuple[Separation, NDArray | None, Any, NDArray]:
    """The decision before any refit: absent when ``certified``, else the rows' answer.

    Returns the decision (``"absent"``; ``"partial"`` for a verified direction, pending the
    withheld fit that would show it maximal; ``"unresolved"`` when none verifies or the rows'
    program exceeds its cap), the verified direction's strict rows (``None`` when none
    verified), the explicit design and its columns' fitted coordinates.
    """
    from superglm.links import CauchitLink

    unique = not isinstance(model._link, CauchitLink)
    if certified:
        return (
            Separation("absent", "certificate", boundaries=boundaries, unique=unique),
            None,
            None,
            np.zeros(0, dtype=np.intp),
        )
    explicit, coordinates = explicit_design(model)

    def priority() -> NDArray | None:
        # past its cap the program reads the rows the walk drove nearest their boundary first
        eta = _linear_predictor(model)
        return None if eta is None else sides * np.where(np.isfinite(eta), eta, 0.0)

    direction, strict, work = separating_direction(explicit, sides, priority)
    width = len(model._solver_pirls_result().beta)
    if direction is None:
        return (
            Separation(
                "unresolved",
                "linear program over its cap: no proposal verifies"
                if work.get("capped")
                else "no direction verifies",
                boundaries=boundaries,
                unique=unique,
                work=work,
            ),
            None,
            explicit,
            coordinates,
        )
    span = proposal_span(direction, coordinates, width)
    labels, labelled = separated_labels(model, strict, sides, coordinates)
    return (
        Separation(
            "partial",
            "linear program: maximality not shown",
            rows=strict,
            span=span,
            boundaries=boundaries,
            unique=unique,
            labels=labels,
            labelled_rows=labelled,
            names=coefficient_names(model, span),
            work=work,
        ),
        strict,
        explicit,
        coordinates,
    )


# ── What a published fit says about separation ──────────────────────────────

StatementTag = Literal["none", "absent", "present", "partial", "unresolved", "heuristic"]

#: The statement's lifecycle: for each tag, the events it takes and the tag each leads to.
#: Every publication of a solve writes a fresh statement (``publish <tag>``, from any tag:
#: the solve's own decision); a revision of the coefficients (an editor edit, Final fit's
#: carry, a shape repair) either keeps the tag (``revise``) or, for a statement whose ground
#: the revision removed, drops it (``revise away``).  ``"none"`` is a fit that says nothing.
STATEMENT_MOVES: Mapping[str, Mapping[str, str]] = {
    state: {
        **{f"publish {tag}": tag for tag in get_args(StatementTag)},
        **revisions,
    }
    for state, revisions in {
        "none": {"revise": "none"},
        "absent": {"revise": "absent"},
        "present": {"revise": "present", "revise away": "none"},
        "partial": {"revise": "partial"},
        "unresolved": {"revise": "unresolved"},
        "heuristic": {"revise": "heuristic", "revise away": "none"},
    }.items()
}


@dataclasses.dataclass(frozen=True)
class SeparationStatement:
    """What a published fit says about separation: one tagged value (``STATEMENT_MOVES``).

    ``tag`` is the fit's existence decision (``Separation.state``), or ``"heuristic"`` where no
    decision is built (a selection penalty, a constrained or SCOP fit, a family with no
    response boundary reached at infinity) and the solver's backstop found its signature
    (``PIRLSResult.separation_message``, which stays the raw finding).  ``text`` is the warning
    the fit gives, ``None`` when it says nothing at fit time.  ``reached`` are the solver slope
    coordinates and ``intercept`` whether the intercept still carries values from the
    separating direction, ``rows`` the separated rows, ``levels`` whether the decision names
    factor levels (whose remedy is grouping them), ``guard_rows`` the weighted rows at the
    link's guard when the evidence is the guard (zero otherwise: a decision's own proof, or a
    backstop's exhausted budget), and ``decision`` the record it was read from.  Written only
    by ``published_statement`` and ``revised_statement``.
    """

    tag: StatementTag
    text: str | None = None
    reached: NDArray = dataclasses.field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    intercept: bool = False
    rows: NDArray = dataclasses.field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    levels: bool = False
    guard_rows: int = 0
    decision: Separation | None = None

    @property
    def source(self) -> str:
        """``"decision"`` for the exact decision's statement, ``"heuristic"`` for the backstop's."""
        return "heuristic" if self.tag == "heuristic" else "decision"


def _tag(statement: SeparationStatement | None) -> str:
    return "none" if statement is None else statement.tag


def _moved(
    before: SeparationStatement | None, event: str, after: SeparationStatement | None
) -> SeparationStatement | None:
    """``after``, once the table says ``event`` takes ``before``'s tag to ``after``'s."""
    target = STATEMENT_MOVES[_tag(before)].get(event)
    if target != _tag(after):
        raise RuntimeError(
            f"separation statement: {_tag(before)} --{event}--> {_tag(after)} is not a move"
        )
    return after


def _linear_predictor(model) -> NDArray | None:
    """The published fit's linear predictor on its training rows, offset included."""
    from superglm.solvers.mode_score import linear_predictor

    if getattr(model, "_dm", None) is None:
        return None
    solver = model._solver_pirls_result()
    return np.asarray(linear_predictor(model._dm, solver, model._fit_offset), dtype=np.float64)


def _guard_rows(model, eta: NDArray | None) -> int:
    """The rows of positive prior weight whose linear predictor sits at the link's guard."""
    from superglm.links import stabilize_eta

    if eta is None:
        return 0
    weights = np.asarray(model._fit_weights, dtype=np.float64)
    return int(np.count_nonzero((stabilize_eta(eta, model._link) != eta) & (weights > 0.0)))


def published_statement(model) -> SeparationStatement | None:
    """The statement the published fit makes, from its decision or, without one, the backstop.

    The decision overrules the backstop: ``"absent"`` proves every estimate finite, so a row at
    the guard is not separation (an offset that alone holds a row there), and a decided
    separation states its coefficients, rows and where the fit stopped
    (``Separation.statement``).  An unresolved decision says so in its own words, with the
    backstop's evidence (rows at the guard, or an exhausted budget) as the reason to warn;
    without that evidence it says nothing at fit time.  With no decision built the backstop's
    finding is the statement, as before the decision existed.
    """
    solver = model._solver_pirls_result()
    raw = getattr(solver, "separation_message", None)
    decision = published_separation(model)
    if decision is None:
        if raw is None:
            return _moved(None, "publish none", None)
        eta = _linear_predictor(model)
        return _moved(
            None,
            "publish heuristic",
            SeparationStatement("heuristic", raw, guard_rows=_guard_rows(model, eta)),
        )
    if decision.state == "absent":
        return _moved(None, "publish absent", SeparationStatement("absent", decision=decision))
    eta = _linear_predictor(model)
    if decision.state == "unresolved":
        guard = None
        guard_rows = 0
        if raw is not None:
            guard_rows = _guard_rows(model, eta)
            guard = (
                f"The fit stopped with its linear predictor at the link's guard on {guard_rows} "
                f"{'row that carries' if guard_rows == 1 else 'rows that carry'} weight"
                if guard_rows
                else "The fit exhausted its iteration budget with a stagnant deviance and an "
                "extreme working-weight range"
            )
        statement = SeparationStatement(
            "unresolved",
            decision.statement(None, guard),
            guard_rows=guard_rows,
            decision=decision,
        )
        return _moved(None, "publish unresolved", statement)
    rows = np.asarray(decision.rows, dtype=np.intp)
    stops = None
    if eta is not None and rows.size:
        stops = (float(np.min(eta[rows])), float(np.max(eta[rows])))
    statement = SeparationStatement(
        decision.state,
        decision.statement(stops, None),
        reached=decision.reached,
        intercept=decision.intercept_reached,
        rows=rows,
        levels=bool(decision.labels),
        decision=decision,
    )
    return _moved(None, f"publish {decision.state}", statement)


def revised_statement(
    model, statement: SeparationStatement | None, replaced: NDArray
) -> SeparationStatement | None:
    """The statement a revision of ``model``'s coefficients leaves (``STATEMENT_MOVES``).

    ``replaced`` are the solver coordinates the revision set to values that read nothing of the
    fit (a single-coefficient term the edit sets outright); every other coordinate still holds,
    or reads (a level term's centring, a projection), the fitted values.  A decided separation
    survives while a reached coordinate, or the intercept, is not replaced, naming only those;
    a partial one, a lower bound whose full span may reach more, always does.  A statement
    whose evidence is the link's guard survives while a weighted row of the revised predictor
    is still there (one from an exhausted budget stands whatever the revision does), and an
    unresolved decision keeps its tag, its warning only so.  ``model`` holds the revised
    coefficients.
    """
    if statement is None or statement.tag == "absent":
        return _moved(statement, "revise", statement)
    if statement.tag == "partial":
        return _moved(statement, "revise", statement)
    if statement.tag == "present":
        decision = cast(Separation, statement.decision)
        kept = np.setdiff1d(statement.reached, np.asarray(replaced, dtype=np.intp))
        if not (statement.intercept or kept.size):
            return _moved(statement, "revise away", None)
        span = np.zeros((decision.span.shape[0], 1 + kept.size))
        span[0, 0] = 1.0 if statement.intercept else 0.0
        span[1 + kept, 1 + np.arange(kept.size)] = 1.0
        if not statement.intercept:
            span = span[:, 1:]
        names = coefficient_names(model, span)
        text = cast(str, dataclasses.replace(decision, names=names).statement(None, None))
        listed = ", ".join(names)
        revised = dataclasses.replace(
            statement,
            text=f"The revised coefficients keep the separated fit's values on {listed}: {text}",
            reached=kept,
        )
        return _moved(statement, "revise", revised)
    eta = _linear_predictor(model)
    standing = statement.guard_rows == 0 or eta is None or _guard_rows(model, eta) > 0
    if statement.tag == "unresolved":
        if standing:
            return _moved(statement, "revise", statement)
        return _moved(statement, "revise", dataclasses.replace(statement, text=None))
    if standing:
        return _moved(statement, "revise", statement)
    return _moved(statement, "revise away", None)


def disclosed_text(result) -> str | None:
    """What a fit handed to its caller warns about separation: its statement's text, once.

    A decided separation whose rows this call's build scan already named says nothing more
    (``separation.named_by_scan``).  A result published before statements existed, or never
    published, speaks with its solve's finding.
    """
    from superglm.diagnostics.separation import named_by_scan

    statement = getattr(result, "separation_statement", None)
    if statement is None:
        return getattr(result, "separation_message", None)
    if statement.text is None:
        return None
    if statement.tag in ("present", "partial") and named_by_scan(statement.rows):
        return None
    return statement.text
