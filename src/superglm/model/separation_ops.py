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
from collections.abc import Callable
from typing import Any, cast

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
) -> bool:
    """Whether the fit's score proves its penalised estimate exists (``recession_excluded``).

    At the fitted state, with Fisher rows ``W``, ``z`` (one consistent set: the rows the score,
    the step and the defect all read), the profiled score ``g = X_c' r - S beta``, ``r = W (z -
    eta)``, takes one Newton step ``v``: the decomposition's inverse on its certified part and
    the weak directions' ``U U' g`` (their covariance root, ``WeakDirections.root``), with the
    intercept's ``sum r / sum W - c'v``.  Its residual ``r' = r - W A v`` gives the boundary
    multipliers ``mu_i = s_i r'_i`` and the defect ``A'r' - S (beta + v)``, computed directly
    (so any ``v`` serves) and Jacobi-scaled: the intercept by ``sqrt(sum W)``, a slope by its
    Hessian diagonal's root.  The defect's own rounding is bounded by Cauchy-Schwarz,
    ``sum_i |x_ij r'_i| <= ||sqrt(W) x_j|| sqrt(sum r'^2 / W)`` (``gamma_(n+p+4)`` per
    product sum, the penalty's ``gamma_(p+2) |S| |beta + v|``).  ``floor`` is the record's
    certified curvature (``Identifiability.curvature_floor``); with the intercept's block of
    one, ``min(1, floor)`` bounds the augmented Hessian's Jacobi eigenvalues.  Any row on the
    boundary with ``W = 0`` (a mean rounded onto the boundary) leaves no multiplier: not
    certified, so the rows decide.
    """
    boundary = (sides == 1) | (sides == -1)
    if not np.any(boundary):
        return True
    if not floor > 0.0:
        return False
    W, residual = fisher_rows(model, solver, weights)
    live = sides != 2
    W = np.where(live, W, 0.0)
    if np.any(W[boundary] <= 0.0) or not np.all(np.isfinite(residual[live])):
        return False
    r = np.where(live, W * residual, 0.0)
    total = float(np.sum(W))
    if not total > 0.0:
        return False
    mean = np.asarray(design.rmatvec(W), dtype=np.float64) / total
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
        e0 = math.fsum(moved_rows.tolist())
        moved, _ = penalty((beta + step)[:, None])
        es = np.asarray(design.rmatvec(moved_rows), np.float64) - mean * e0 - moved[:, 0]
        return e0, es

    rsum = math.fsum(r.tolist())
    gradient = np.asarray(design.rmatvec(r), dtype=np.float64) - mean * rsum - applied[:, 0]
    step = solve(gradient)
    intercept = rsum / total - float(mean @ step)
    scale = np.asarray(decomposition.column_scale, dtype=np.float64)
    safe = np.where(scale > 0.0, scale, 1.0)
    carried = W > 0.0
    uncentred = np.sqrt(np.maximum(np.asarray(data_diagonal, np.float64), 0.0) + total * mean**2)
    n, p = int(design.n), int(beta.size)
    curvature, weight = min(1.0, float(floor)), float(np.max(W))
    # iterative refinement (Wilkinson), taken only when the certificate needs it: the step
    # solves ``H v = g`` only to the coupling's and the weak directions' accuracy; any ``v``
    # serves the certificate, and a refined one leaves a smaller defect
    for refinement in range(3):
        moved_rows = residual_rows(step, intercept)
        margin = float(np.min(sides[boundary] * moved_rows[boundary]))
        e0, es = defect(moved_rows, step)
        if margin > 0.0:
            _, magnitude = penalty((beta + step)[:, None])
            zeta = math.sqrt(float(np.sum(moved_rows[carried] ** 2 / W[carried])))
            rounding = np.r_[
                gamma_up(n + 2) * zeta,
                (2.0 * gamma_up(n + p + 4) * uncentred * zeta + gamma_up(p + 2) * magnitude[:, 0])
                / safe,
            ]
            scaled = np.r_[e0 / math.sqrt(total), es / safe]
            bound = float(np.linalg.norm(scaled)) + float(np.linalg.norm(rounding))
            if recession_excluded(margin, bound, curvature, weight):
                return True
        if refinement == 2:
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


def separated_labels(model, rows: NDArray, sides: NDArray, coordinates: NDArray) -> tuple:
    """The categorical levels (base levels included) whose live rows are all separated.

    One entry per factor, ``"level E of f"`` or ``"levels D, E of f"``.
    """
    from superglm._group_matrix._group_matrix_core import CategoricalGroupMatrix

    separated = np.zeros(sides.size, dtype=bool)
    separated[rows] = True
    live = sides != 2
    specs = {**getattr(model, "_specs", {}), **getattr(model, "_interaction_specs", {})}
    labels = []
    explicit = set(int(c) for c in coordinates if c >= 0)
    for group, matrix in zip(model._groups, model._dm.group_matrices, strict=True):
        if type(matrix) is not CategoricalGroupMatrix or group.start not in explicit:
            continue
        spec = specs.get(group.feature_name)
        names = list(getattr(spec, "_non_base", ())) + [getattr(spec, "_base_level", "base")]
        levels = [
            str(names[int(code)])
            for code in np.unique(matrix.codes[separated])
            if code < len(names) and np.all(separated[live & (matrix.codes == code)])
        ]
        if levels:
            plural = "levels" if len(levels) > 1 else "level"
            labels.append(f"{plural} {', '.join(levels)} of {group.name}")
    return tuple(labels)


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
    """The decision before any refit: absent when ``certified``, else the rows' proposal.

    Returns the decision (``"absent"``, or ``"unresolved"`` pending the withheld fit), the
    verified direction's strict rows (``None`` when none verified), the explicit design and its
    columns' fitted coordinates.
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
    direction, strict, work = separating_direction(explicit, sides)
    width = len(model._solver_pirls_result().beta)
    if direction is None:
        return (
            Separation(
                "unresolved",
                "no direction verifies",
                boundaries=boundaries,
                unique=unique,
                work=work,
            ),
            None,
            explicit,
            coordinates,
        )
    span = proposal_span(direction, coordinates, width)
    return (
        Separation(
            "unresolved",
            "maximality not shown",
            rows=strict,
            span=span,
            boundaries=boundaries,
            unique=unique,
            names=coefficient_names(model, span),
            work=work,
        ),
        strict,
        explicit,
        coordinates,
    )
