"""The directions a published SuperGLM fit does not determine, as its readers test them.

``diagnostics.estimability`` decides a row off support for a kind of direction when
``||r Q_k|| > 1e-6 ||r||`` in the record's dual Jacobi scale.  This module reads the kinds from
the published fit, never building anything (a fit without them discloses nothing):

- ``"alias"``: the gauge's moved directions (``identifiability.Gauge.null``, ``moved``), the
  directions of ``null(H)`` its constraint fixes;
- ``"separated"``: the published statement's span (``separation_ops.SeparationStatement``):
  the decision's verified span (``Separation.span``, ``"present"`` or ``"partial"``), or, on a
  model whose coefficients a revision replaced in part, the unit coordinates the revised
  statement still names (``reached``, and the intercept);
- ``"unresolved"``: the gauge's directions it holds without moving (``_design_null``), less a
  certified separation's span (the record's decision), which it also holds there;
- ``"unfitted"``: weak directions the fit identifies but kept where its solver stopped
  (``WeakDirections.fitted`` false).

**The metric.**  In the centred coordinates ``(alpha, beta)``, ``alpha = b0 + c' beta`` with ``c``
the fit's weighted column centre, the penalised Hessian of the augmented fit is
``diag(sum W, X_c' W X_c + S)``; its Jacobi scale is ``M = diag(sqrt(sum W), D)``,
``D = sqrt(diag(X_c' W X_c + S))`` (``TruncatedDecomposition.column_scale``; a coordinate with no
curvature reads one, as the record's own orthonormal bases do).  A direction ``[a0; a]`` in raw
coordinates is ``M [a0 + c'a; a]`` there, and a functional ``r`` of the raw coefficients is
``r M^-1`` in the dual scale.  Each kind's basis ``Q`` is orthonormal in that scale and is kept as
the raw coefficient vectors ``q = M^-1 Q`` (back in raw coordinates), so a reader's own linear
evaluation of a functional at the coefficients ``q`` gives ``r Q`` exactly: the prediction at
``q`` (``base``), or a term curve's standard error at the covariance ``q q'`` (``explain_ops``).

**Coordinates.**  The record is in the solver's coordinates.  A spline term whose public columns
lose their training means ``m`` (``runtime_canonicalize``) keeps its slopes, and its raw
intercept component moves by ``m' a``; the public centre is ``c - m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import SQRT_EPS


@dataclass(frozen=True)
class PublishedDirections:
    """Each kind's dual-Jacobi orthonormal basis of a published fit, in public raw coordinates.

    ``slopes[kind]`` (``p x k``) and ``intercepts[kind]`` (``k``) are the raw coefficient vectors
    ``q = M^-1 Q`` (module docstring).  ``scale`` is ``D`` (zero curvature read as one),
    ``centre`` the public weighted centre ``c - m`` and ``alpha_scale`` ``sqrt(sum W)`` (one when
    no row carries weight), which a reader needs for a row's norm in the dual scale.
    """

    slopes: dict[str, NDArray]
    intercepts: dict[str, NDArray]
    scale: NDArray
    centre: NDArray
    alpha_scale: float


def published_record(model):
    """The published fit's identifiability record, or ``None``; never built here.

    A dense fit keeps it on its rank record; a structured fit on its retained state, read only
    while that state is held (a retired one is rebuilt by inference, not by its readers).
    """
    rank_info = getattr(getattr(model, "_result", None), "rank_info", None)
    record = getattr(rank_info, "identifiability", None)
    if record is None:
        state = model.__dict__.get("_linear_system_state")
        record = getattr(state, "identifiability", None)
    return record


def _separated_span(statement, width: int) -> NDArray | None:
    """``[intercept; slopes]`` (solver coordinates) the published statement keeps separated."""
    if statement is None or statement.tag not in ("present", "partial"):
        return None
    decision = statement.decision
    if decision is None:
        return None
    reached = np.sort(np.asarray(statement.reached, dtype=np.intp))
    span = np.asarray(decision.span, dtype=np.float64)
    if np.array_equal(reached, np.sort(np.asarray(decision.reached, dtype=np.intp))) and bool(
        statement.intercept
    ) == bool(decision.intercept_reached):
        return span if span.shape[1] else None
    # a revision replaced part of the span: the coordinates still holding the walk's values
    units = np.zeros((1 + width, int(statement.intercept) + reached.size))
    if statement.intercept:
        units[0, 0] = 1.0
    units[1 + reached, int(statement.intercept) + np.arange(reached.size)] = 1.0
    return units if units.shape[1] else None


def _metric(model, record, width: int) -> tuple[NDArray, NDArray, float] | None:
    """``(D, c, sum W)`` in solver coordinates: the Jacobi scale the record was decided in."""
    rank_info = getattr(model._result, "rank_info", None)
    decomposition = getattr(rank_info, "augmented", None)
    if decomposition is not None:
        selected = np.asarray(rank_info.selected_columns, dtype=np.intp)
        column_scale = np.asarray(decomposition.column_scale, dtype=np.float64)
        centre = np.asarray(rank_info.mean_x, dtype=np.float64)
        if column_scale.shape != selected.shape or centre.shape != (width,):
            return None
        scale = np.zeros(width)
        scale[selected] = column_scale
        return scale, centre, float(rank_info.sum_w)
    state = model.__dict__.get("_linear_system_state")
    if state is None or record is None or getattr(state, "identifiability", None) is not record:
        return None
    from superglm.model.state_ops import _spread_centred_operator, _structured_penalty_diagonal
    from superglm.solvers._structured.operators import compact_operator_diagonal

    operator = _spread_centred_operator(state.centered_data_operator)
    diagonal = compact_operator_diagonal(operator) + _structured_penalty_diagonal(
        state.penalized_operator
    )
    if diagonal.shape != (width,):
        return None
    return np.sqrt(np.maximum(diagonal, 0.0)), np.asarray(operator.center), float(operator.total)


def _orthonormal(jacobi: NDArray) -> NDArray:
    """An orthonormal basis of ``jacobi``'s columns, cut below ``sqrt(eps)`` of the largest."""
    if jacobi.shape[1] == 0:
        return jacobi
    left, singular, _ = np.linalg.svd(jacobi, full_matrices=False)
    if singular.size == 0 or singular[0] == 0.0:
        return left[:, :0]
    return left[:, singular > SQRT_EPS * singular[0]]


def published_directions(model) -> PublishedDirections | None:
    """The published fit's directions by kind, or ``None`` when it has none to disclose.

    Reads the record and the published statement only; ``None`` (nothing to disclose, and no
    pass over any row) for a fit whose record holds no such direction, a model with no record (a
    legacy saved model) and a record whose metric is no longer held.
    """
    result = getattr(model, "_result", None)
    if result is None:
        return None
    width = int(np.asarray(result.beta).size)
    record = published_record(model)
    raw: dict[str, NDArray] = {}
    gauge = getattr(record, "gauge", None)
    if gauge is not None:
        moved = gauge._moved()
        augmented = np.vstack((np.asarray(gauge.intercept)[None, :], np.asarray(gauge.null)))
        if np.any(moved):
            raw["alias"] = augmented[:, moved]
        if not np.all(moved):
            raw["unresolved"] = augmented[:, ~moved]
    span = _separated_span(getattr(result, "separation_statement", None), width)
    if span is not None:
        raw["separated"] = span
    weak = getattr(record, "weak", None)
    if weak is not None and not weak.fitted:
        raw["unfitted"] = np.vstack((np.asarray(weak.intercept)[None, :], np.asarray(weak.root)))
    if not raw:
        return None
    metric = _metric(model, record, width)
    if metric is None:
        return None
    scale, centre, total = metric
    scale = np.where(scale > 0.0, scale, 1.0)
    alpha_scale = float(np.sqrt(total)) if total > 0.0 else 1.0

    def jacobi(directions: NDArray) -> NDArray:
        alpha = directions[0] + centre @ directions[1:]
        return np.vstack((alpha_scale * alpha[None, :], scale[:, None] * directions[1:]))

    bases = {kind: _orthonormal(jacobi(directions)) for kind, directions in raw.items()}
    # the gauge holds a certified separation's span unmoved as well; it is the statement's to
    # name (``"separated"``), or nobody's once a revision replaced it
    decision = getattr(record, "separation", None)
    decided = getattr(decision, "span", None)
    if "unresolved" in bases and decided is not None and np.shape(decided)[1]:
        held, span = bases["unresolved"], _orthonormal(jacobi(np.asarray(decided, np.float64)))
        bases["unresolved"] = _orthonormal(held - span @ (span.T @ held))
    terms = getattr(model, "_runtime_canonical_state", None) or {}
    from superglm.model.runtime_canonicalize import _public_column_shifts

    shifts = _public_column_shifts(terms.get("terms", {}), width)
    slopes: dict[str, NDArray] = {}
    intercepts: dict[str, NDArray] = {}
    for kind, basis in bases.items():
        if basis.shape[1] == 0:
            continue
        slope = basis[1:] / scale[:, None]
        slopes[kind] = slope
        # raw solver intercept ``alpha - c'a``, then the public columns' ``m'a``
        intercepts[kind] = basis[0] / alpha_scale - centre @ slope + shifts @ slope
    if not slopes:
        return None
    return PublishedDirections(slopes, intercepts, scale, centre - shifts, alpha_scale)


def term_coordinates(model, name) -> NDArray:
    """A term's coefficient positions, in the order its blocks concatenate them."""
    blocks = [
        np.arange(group.start, group.end, dtype=np.intp)
        for group in model._groups
        if group.feature_name == name
    ]
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.intp)


def point_labels(model, name, values) -> NDArray | None:
    """Each point a term reader reports, labelled by ``diagnostics.estimability.off_support``.

    ``values(beta_t)`` is the reader at the term's coefficients ``beta_t`` (``term_coordinates``):
    an array of the log values it reports, linear in ``beta_t`` (a curve, a level table, a cell
    table, a bin's weighted mean of the term).  Its point ``r`` is a functional of the term's
    coordinates alone, so its share along a kind is ``||r Q||``, read as ``values`` at each
    direction ``q_t`` (one call per direction), and its dual norm ``||r D^-1||``, read from
    ``values`` at the term's unit vectors.  ``None`` when the fit has no direction to disclose,
    and when none reaches the term enough to move a label: ``||r Q|| <= ||r D^-1|| ||D q_t||``
    (Cauchy-Schwarz), so a term with ``||D q_t|| <= 1e-6`` over every direction has no point off
    support, and nothing is read.
    """
    from superglm.diagnostics.estimability import PREDICTION_TOLERANCE, off_support

    directions = published_directions(model)
    if directions is None:
        return None
    index = term_coordinates(model, name)
    scale = directions.scale[index]
    reached = {
        kind: slopes[index]
        for kind, slopes in directions.slopes.items()
        if np.linalg.norm(scale[:, None] * slopes[index]) > PREDICTION_TOLERANCE
    }
    if not reached or index.size == 0:
        return None

    def read(beta: NDArray) -> NDArray:
        return np.asarray(values(np.asarray(beta, dtype=np.float64)), dtype=np.float64)

    units = np.eye(index.size)
    squares = sum((read(units[j]) / scale[j]) ** 2 for j in range(index.size))
    shares = {
        kind: np.stack([read(slopes[:, j]).ravel() for j in range(slopes.shape[1])], axis=1)
        for kind, slopes in reached.items()
    }
    norms = np.sqrt(np.asarray(squares, dtype=np.float64))
    return off_support(shares, norms.ravel()).reshape(norms.shape)
