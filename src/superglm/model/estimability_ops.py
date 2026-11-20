"""Predictions that lean on what the training data do not determine.

R's ``predict.lm(rankdeficient = "warnif")`` (R >= 4.3): a rank-deficient fit
determines ``x' beta`` only for rows ``x`` in the row space of its design, so
a prediction off that space is returned as computed and named in one warning.
A row ``r`` is off the space when its share along the design's null space
exceeds a tolerance relative to its own size,

    ||r N|| > tol ||r||,    tol = 1e-6,

with ``N`` an orthonormal basis of the null space: Lenth, *Estimability Tools
for Package Developers*, R Journal 7(1), 2015, whose test this is (rotation
invariant, so no choice of basis moves it), at ``predict.lm``'s default
``tol``.  Rows of the training design satisfy ``X N = 0``; their shares are
rounding, many orders below ``tol``.

**The directions.**  The fit's data rank (``RankInfo.data``, the decomposition
of the centred data Gram ``X_c' W X_c`` the fit already decided) records the
null space of the centred design as ``parameter_null_basis``.  The design with
its intercept, ``[1, X]``, has null vectors ``[-c' n; n]`` for each ``n`` of
it (``c`` the weighted column centre), so a scoring row ``[1, x]`` reads
``(x - c)' n`` on it.  Nothing here forms a decomposition or reads a
training row: a fit whose record holds no null direction is never tested.

**The scale.**  The record decided its rank on columns equilibrated by
``D = column_scale`` (the norms of the centred weighted columns); the
intercept's column has norm ``sqrt(sum W)``.  The test is read in that scale,
where the bases are orthonormal: a row is ``[1 / sqrt(sum W), (x - c) / D]``
and a null direction ``D n``, normalised.  A column the record holds
structurally null (``D = 0``: constant on the training rows) has unit scale.
So a row's size counts its intercept, and a row at the centre is not 0 / 0.

**Coordinates.**  The record is in the solver's columns.  A spline term whose
public columns lose their training means ``m`` (``runtime_canonicalize``)
keeps its coefficients, so a direction is the same coefficient vector in the
public model and its centre is ``c - m`` there.
"""

from __future__ import annotations

import contextlib
import contextvars
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ``predict.lm``'s ``tol`` (R >= 4.3, ``rankdeficient = "warnif"``), on the
# share ``||r N|| / ||r||`` of Lenth (2015).  Training rows read rounding,
# about ``u`` times the record's null-space error; a constructed row off the
# span reads order one.  The gap between the two is reported by the tests.
PREDICTION_TOLERANCE = 1e-6

# Rows per block when a term's design rows are formed for their norms: about
# 2**20 entries.
_NORM_BLOCK_ENTRIES = 2**20

_QUIET: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "superglm_non_estimable_prediction_quiet", default=False
)


class NonEstimablePredictionWarning(UserWarning):
    """Some predicted rows lean on directions the training data do not determine.

    ``rows`` holds their positions in the order predicted, as R's
    ``"non-estim"`` attribute does.  Their predictions are returned unchanged:
    the published fit's values, which its constraint fixes there rather than
    the data.
    """

    def __init__(self, message: str, rows: NDArray) -> None:
        super().__init__(message)
        self.rows = np.asarray(rows, dtype=np.intp)


@contextlib.contextmanager
def non_estimable_warning_quiet():
    """Inside, ``predict`` does not test its rows: a reader of its means, not ``predict``."""
    token = _QUIET.set(True)
    try:
        yield
    finally:
        _QUIET.reset(token)


@dataclass(frozen=True)
class DataNullDirections:
    """The record's null directions in public raw coordinates, with the scale they are read in.

    ``directions`` (``p x k``) are the coefficient vectors ``q = n``, orthonormal
    in the dual scale (``D q`` on the record's columns), and ``intercepts`` the
    raw intercept ``-c' q`` that makes ``[a; q]`` a null vector of the public
    design with its intercept.  ``scale`` is ``D`` (one on a structurally null
    column, zero on a column outside the record, which no direction reaches),
    ``centre`` the public weighted centre and ``alpha_scale`` ``sqrt(sum W)``.
    """

    directions: NDArray
    intercepts: NDArray
    scale: NDArray
    centre: NDArray
    alpha_scale: float


def data_null_directions(model) -> DataNullDirections | None:
    """The published fit's data-null directions, or ``None`` when it records none.

    Reads the data rank the fit decided (``RankInfo.data``) and nothing else;
    ``None`` for a fit without that record and for one whose record is full
    rank, so such a fit is never tested.
    """
    result = getattr(model, "_result", None)
    rank_info = getattr(result, "rank_info", None)
    record = getattr(rank_info, "data", None)
    if record is None:
        return None
    null = record.null_basis()
    if null.shape[1] == 0:
        return None
    width = int(np.asarray(result.beta).size)
    selected = np.asarray(rank_info.selected_columns, dtype=np.intp)
    mean_x = np.asarray(rank_info.mean_x, dtype=np.float64)
    if selected.size != record.width or mean_x.shape != (width,):
        return None
    column_scale = np.asarray(record.column_scale, dtype=np.float64)
    scaled = column_scale > 0.0
    dual = np.array(null, dtype=np.float64)
    if record.equilibrated_null_basis is not None:
        dual[scaled] = record.equilibrated_null_basis[scaled]
    else:
        dual[scaled] = null[scaled] * column_scale[scaled, None]
    norms = np.linalg.norm(dual, axis=0)
    kept = norms > np.finfo(np.float64).eps
    if not np.any(kept):
        return None
    dual = dual[:, kept] / norms[kept]
    local_scale = np.where(scaled, column_scale, 1.0)
    scale = np.zeros(width)
    scale[selected] = local_scale
    directions = np.zeros((width, dual.shape[1]))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        directions[selected] = dual / local_scale[:, None]
    from superglm.model.runtime_canonicalize import _public_column_shifts

    terms = (getattr(model, "_runtime_canonical_state", None) or {}).get("terms", {})
    centre = mean_x - _public_column_shifts(terms, width)
    sum_w = float(rank_info.sum_w)
    return DataNullDirections(
        directions=directions,
        intercepts=-(centre @ directions),
        scale=scale,
        centre=centre,
        alpha_scale=float(np.sqrt(sum_w)) if sum_w > 0.0 else 1.0,
    )


def _dense_rows(rows) -> NDArray:
    if hasattr(rows, "toarray"):  # a SciPy sparse block
        rows = rows.toarray()
    return np.asarray(rows, dtype=np.float64)


def _population_skipped(spec, random_effects: str) -> bool:
    """Whether a population prediction leaves the term out (``base._score_terms``)."""
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.random_effect import RandomEffect

    if random_effects != "population":
        return False
    if isinstance(spec, RandomEffect):
        return True
    return isinstance(spec, FactorSmooth) and not spec._has_population_offset


def _direction_products(model, frame, plan, null: DataNullDirections, random_effects: str):
    """``r q`` for every predicted row and direction: the predictor at ``[a; q]``, no offset.

    One coefficient pass per direction through ``predict``'s own terms
    (``base._score_terms``, linear in the coefficients), so the functional is
    the one ``predict`` evaluates, an ``sz`` term's held levels included.  It
    runs at the fit's centred state, so a dense column reads ``(x - c) q``
    rather than a difference of two large products.
    """
    from superglm.model.base import EtaSum, _score_terms, prediction_centred_state

    _, centre, _ = prediction_centred_state(model.result)
    n = len(frame)
    products = np.empty((n, null.directions.shape[1]))
    for j in range(null.directions.shape[1]):
        direction = null.directions[:, j]
        alpha = float(null.intercepts[j]) + (0.0 if centre is None else float(centre @ direction))
        eta = EtaSum(n, alpha, None)
        _score_terms(
            model,
            frame,
            plan,
            direction,
            eta,
            centre,
            random_effects=random_effects,
            fitted=False,
            warn=False,
            validate=False,
        )
        products[:, j] = eta.finish(consume=True)
    return products


def _row_norms(frame, plan, null: DataNullDirections, random_effects: str) -> NDArray:
    """``||[1 / sqrt(sum W), (x - c) / D]||`` for every predicted row, in row blocks.

    A term a population prediction leaves out has a zero design row there.
    """
    from superglm.model.base import _term_columns

    n = len(frame)
    squares = np.full(n, 1.0 / null.alpha_scale**2)
    for term in (*plan["features"], *plan["interactions"]):
        index = np.asarray(term["beta_idx"], dtype=np.intp)
        covered = null.scale[index] > 0.0
        if not np.any(covered):
            continue
        scale, centre = null.scale[index][covered], null.centre[index][covered]
        spec = term["spec"]
        if _population_skipped(spec, random_effects):
            squares += float(np.sum((centre / scale) ** 2))
            continue
        columns = _term_columns(term, frame)
        block = max(1, _NORM_BLOCK_ENTRIES // max(1, index.size))
        for start in range(0, n, block):
            rows = _dense_rows(
                spec.transform(*(column[start : start + block] for column in columns))
            )
            scaled = (rows[:, covered] - centre) / scale
            squares[start : start + block] += np.einsum("ij,ij->i", scaled, scaled)
    return np.sqrt(squares)


def prediction_shares(model, frame, plan, random_effects: str) -> NDArray | None:
    """Each predicted row's share ``||r N|| / ||r||`` along the data-null space, or ``None``.

    ``None``, with no pass over the rows, when the fit records no data-null
    direction (``data_null_directions``).
    """
    null = data_null_directions(model)
    if null is None:
        return None
    # ``predict``'s own pass has already said what scoring these rows says.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        products = _direction_products(model, frame, plan, null, random_effects)
        norms = _row_norms(frame, plan, null, random_effects)
    return np.linalg.norm(products, axis=1) / norms


def disclose_non_estimable(model, frame, random_effects: str) -> None:
    """Warn once, naming the predicted rows off the training design's row space.

    Nothing when the fit records no data-null direction, inside
    ``non_estimable_warning_quiet``, or when every row is on the space.
    """
    if _QUIET.get():
        return
    from superglm._warn import warn_caller
    from superglm.model.base import _prediction_plan

    shares = prediction_shares(model, frame, _prediction_plan(model), random_effects)
    if shares is None:
        return
    rows = np.flatnonzero(shares > PREDICTION_TOLERANCE)
    if rows.size == 0:
        return
    warn_caller(NonEstimablePredictionWarning(non_estimable_message(rows), rows))


def non_estimable_message(rows: NDArray) -> str:
    """The warning's text for the off-span rows ``rows``."""
    count = int(rows.size)
    first = ", ".join(str(int(row)) for row in rows[:5]) + (", ..." if count > 5 else "")
    subject = "1 predicted row combines" if count == 1 else f"{count} predicted rows combine"
    return (
        f"{subject} covariate values in a way the training data do not determine (rows "
        f"{first}): the fit cannot separate some of its terms on the training rows, and "
        "these rows tell them apart. Their predictions are the fitted model's, which rest on "
        "how the fit split those terms rather than on the data. The warning's `rows` "
        "attribute lists them; check those rows' levels and values against the training data, "
        "or merge the terms the fit cannot separate."
    )


def entry_shares(
    model,
    term_name: str,
    points: Mapping[str, Sequence],
    centring_points: Mapping[str, Sequence] | None = None,
) -> NDArray | None:
    """Each rating-table entry's share along the data-null space, or ``None``.

    An entry is the term's log relativity at the raw covariate values
    ``points`` (one entry per row of the mapping, keyed by the term's input
    columns), less the term's mean over ``centring_points`` when the table is
    mean-centred: the coefficient functional ``f = B(points) - mean
    B(centring_points)`` of the term's design rows ``B``.  It carries no
    intercept, so its share is ``||(f / D) N|| / ||f / D||`` on the term's
    columns, and an entry that reads no coefficient (a base level) has share
    zero.  ``None``, with no row formed, when the fit records no data-null
    direction.
    """
    null = data_null_directions(model)
    if null is None:
        return None
    from superglm.model.base import _prediction_plan

    plan = _prediction_plan(model)
    term = next(t for t in (*plan["features"], *plan["interactions"]) if t["name"] == term_name)
    functionals = _term_rows(term, points)
    if centring_points is not None:
        functionals = functionals - _term_rows(term, centring_points).mean(axis=0)
    index = np.asarray(term["beta_idx"], dtype=np.intp)
    covered = null.scale[index] > 0.0
    with np.errstate(all="ignore"):
        products = functionals[:, covered] @ null.directions[index][covered]
        dual = functionals[:, covered] / null.scale[index][covered]
        share = np.linalg.norm(products, axis=1)
        size = np.linalg.norm(dual, axis=1)
        return np.where(size > 0.0, share / np.where(size > 0.0, size, 1.0), 0.0)


def entry_marks(
    model,
    term_name: str,
    points: Mapping[str, Sequence],
    centring_points: Mapping[str, Sequence] | None = None,
) -> NDArray | None:
    """Which rating-table entries lean on the data-null space (``entry_shares``), or ``None``."""
    shares = entry_shares(model, term_name, points, centring_points)
    return None if shares is None else shares > PREDICTION_TOLERANCE


def _term_rows(term, points: Mapping[str, Sequence]) -> NDArray:
    """The term's design rows at the raw covariate values ``points``, as ``predict`` reads them."""
    import pandas as pd

    from superglm._frame import as_eager_frame
    from superglm.model.base import _term_columns

    frame = as_eager_frame(pd.DataFrame({key: list(values) for key, values in points.items()}))
    return _dense_rows(term["spec"].transform(*_term_columns(term, frame)))
