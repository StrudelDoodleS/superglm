"""Predictions that lean on what neither the training data nor the penalty determine.

R's ``predict.lm(rankdeficient = "warnif")`` (R >= 4.3): a rank-deficient fit
determines ``x' beta`` only for rows ``x`` in the row space of its design, so
a prediction off that space is returned as computed and named in one warning.
A row ``r`` is off the space when its share along the undetermined directions
exceeds a tolerance relative to its own size,

    ||r N|| > tol ||r||,    tol = 1e-6,

with ``N`` an orthonormal basis of those directions: Lenth, *Estimability
Tools for Package Developers*, R Journal 7(1), 2015, whose test this is
(rotation invariant, so no choice of basis moves it), at ``predict.lm``'s
default ``tol``.  A row of the training design reads rounding there.  A row
the record treats as determined only up to its rank cut (a near alias the
data rank truncated) can read a share near ``tol`` and warn, as it does in
``predict.lm``.  The warning is advisory: the price is unchanged.

**Which fits are tested.**  Only a dense, unbinned ``SuperGLM`` fit that
carries both rank records below.  Every other fit is silent, as before this
test existed: a binned fit (``discrete=True``, or a spline with
``discrete=True``), whose record is its binned design's and not the rows it
prices; a fit with a ``FactorSmooth`` term (``sz`` or ``fs``), whose held and
population levels are priced by their own rules; a structured fit, and a
compact or older saved model, that keeps no dense rank records; and a fit
whose centred columns sit so far from zero that float64 cannot resolve a row's
share at the tolerance (``_resolvable``).  ``SuperLSS`` has its own ``predict``
and is not tested.

**The directions.**  The fit's data rank (``RankInfo.data``, the decomposition
of the centred data Gram ``X_c' W X_c`` the fit already decided) records the
null space of the centred design.  A penalised fit determines a direction of
that space wherever its penalty ``S`` has curvature there (a random effect's
level sum under its zero-mean prior, a spline's wiggle): along a data-null
``n``, ``n' (X_c' W X_c + S) n = n' S n``, so the null space of the penalised
Hessian, which the fit also decided (``RankInfo.augmented``, at its own
scale-aware rank cut), is exactly the part no penalty curves.  Only the
intersection of the two recorded null spaces is tested: the principal
directions between them whose squared sine is at most ``FREE_SINE_CUT``.
Penalty null spaces are structural, so a shared direction's squared sine is
the bases' rounding and any penalised one's is one.  Nothing here reads the
design or a training row, and nothing runs at fit time.

The design with its intercept, ``[1, X]``, has null vectors ``[-c' n; n]`` for
each such ``n`` (``c`` the record's weighted column centre), so a scoring row
``[1, x]`` reads ``(x - c)' n``.

**The scale.**  The record decided its rank on columns equilibrated by
``D = column_scale`` (the norms of the centred weighted columns); the
intercept's column has norm ``sqrt(sum W)``.  The test is read in that scale,
where the bases are orthonormal: a row is ``[1 / sqrt(sum W), (x - c) / D]``
and a direction ``D n``, normalised.  A column the record holds structurally
null (``D = 0``: constant on the training rows) has unit scale.  Each row is
scaled by a power of two (exact) before it is divided by ``D``, projected or
its norm taken, LAPACK's ``xLASSQ`` scaled sum of squares, so no finite row
overflows.

**Coordinates.**  The record is in the solver's columns.  A spline term whose
public columns lose their training means ``m`` (``runtime_canonicalize``)
keeps its coefficients, so a direction is the same coefficient vector in the
public model and its centre is ``c - m`` there.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import SQRT_EPS, UNIT_ROUNDOFF

# ``predict.lm``'s ``tol`` (R >= 4.3, ``rankdeficient = "warnif"``), on the
# share ``||r N|| / ||r||`` of Lenth (2015).  Training rows read rounding,
# about ``u`` times the record's null-space error; a constructed row off the
# span reads order one.  The gap between the two is reported by the tests.
PREDICTION_TOLERANCE = 1e-6

# A data-null direction is free when its squared sine to the penalised Hessian's
# null space is at most this: a shared direction reads the two recorded bases'
# rounding (below 1e-15 on every sweep fixture), a penalised one reads one.
FREE_SINE_CUT = SQRT_EPS

# Rows per block when a term's design rows are formed: about 2**20 entries.
_BLOCK_ENTRIES = 2**20

_QUIET: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "superglm_non_estimable_prediction_quiet", default=False
)


class NonEstimablePredictionWarning(UserWarning):
    """Some predicted rows lean on directions the training data and the penalty do not determine.

    ``rows`` holds their positions in the order predicted, as R's
    ``"non-estim"`` attribute does.  Their predictions are returned unchanged:
    the published fit's values, which its constraint fixes there rather than
    the data.  Only dense, unbinned fits are tested
    (``superglm.model.estimability_ops``).
    """

    def __init__(self, message: str, rows: NDArray) -> None:
        super().__init__(message)
        self.rows = np.asarray(rows, dtype=np.intp)

    def __reduce__(self):
        # pickled with its rows, so a warning raised in a worker process arrives whole
        return (type(self), (str(self), self.rows))


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
    """The tested directions in public coordinates, with the scale they are read in.

    ``dual`` (``p x k``) is orthonormal: ``D n`` on the record's columns, zero
    elsewhere.  ``scale`` is ``D`` (one on a structurally null column, zero on
    a column outside the record, which no direction reaches) and ``centre``
    the public centre rows are read about.  ``alpha_scale`` is ``sqrt(sum W)``.
    ``outside`` holds, for every data-null direction, its squared sine to the
    penalised Hessian's null space, ascending; the kept directions are those
    at or below ``FREE_SINE_CUT`` (the restriction's evidence: its gap).
    """

    dual: NDArray
    scale: NDArray
    centre: NDArray
    alpha_scale: float
    outside: NDArray


def _tested(model) -> bool:
    """Whether the fit is one the test reads (module docstring): dense, unbinned, no factor smooth."""
    from superglm.dm_builder import should_discretize
    from superglm.features.factor_smooth import FactorSmooth

    discrete = bool(getattr(model, "_discrete", False))
    if discrete:
        return False
    if any(should_discretize(spec, discrete) for spec in getattr(model, "_specs", {}).values()):
        return False
    interactions = getattr(model, "_interaction_specs", {}) or {}
    return not any(isinstance(spec, FactorSmooth) for spec in interactions.values())


def _records(model):
    """The fit's result and its two dense rank records, or ``None`` when either is missing."""
    result = getattr(model, "_result", None)
    rank_info = getattr(result, "rank_info", None)
    data = getattr(rank_info, "data", None)
    augmented = getattr(rank_info, "augmented", None)
    if result is None or data is None or augmented is None:
        return None
    return result, rank_info, data


def _dual_null_basis(record) -> NDArray | None:
    """The record's null basis in its equilibrated scale, columns normalised, or ``None``."""
    null = record.null_basis()
    if null.shape[1] == 0:
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
    return dual[:, kept] / norms[kept]


def _orthonormal(basis: NDArray) -> NDArray:
    """An orthonormal basis of ``basis``'s columns (a thin QR of a few columns)."""
    if basis.shape[1] == 0:
        return basis
    q, _ = np.linalg.qr(basis)
    return q


def _penalty_free(rank_info, dual: NDArray, local_scale: NDArray):
    """The part of ``span(dual)`` the penalised Hessian leaves null, and every squared sine.

    ``dual`` is the data-null basis in the record's scale.  The penalised
    Hessian's recorded null basis, read in the same scale, is orthonormalised;
    the singular values of their overlap are the cosines of the principal
    angles between the two spaces.  A direction is kept when its squared sine
    is at most ``FREE_SINE_CUT``.
    """
    augmented = rank_info.augmented
    if augmented is rank_info.data:  # unpenalised: one decomposition
        return dual, np.zeros(dual.shape[1])
    if augmented.width != dual.shape[0]:
        return dual[:, :0], np.ones(dual.shape[1])
    held = augmented.null_basis()
    if held.shape[1] == 0:
        return dual[:, :0], np.ones(dual.shape[1])
    held = _orthonormal(held * local_scale[:, None])
    basis = _orthonormal(dual)
    left, cosines, _ = np.linalg.svd(basis.T @ held, full_matrices=True)
    sines = np.ones(dual.shape[1])
    sines[: cosines.size] = np.maximum(0.0, 1.0 - cosines**2)
    order = np.argsort(sines, kind="stable")
    sines, left = sines[order], left[:, order]
    return basis @ left[:, sines <= FREE_SINE_CUT], sines


def _resolvable(null: DataNullDirections) -> bool:
    """Whether float64 resolves a row's share at the tolerance, given the stored centre.

    The centre ``c`` is held to rounding, ``u |c_j|`` per column, which moves
    a row's dual entry by ``u |c_j| / D_j``; over a row of size at least
    ``1 / sqrt(sum W)`` and the unit directions that bounds the share's error
    by ``u sqrt(sum W) || (|c| / D)' |N| ||``.  A fit where that reaches a
    sixteenth of the tolerance (a column offset from zero by many times its
    spread, 2**40 against a spread of a few units) is not tested.
    """
    covered = null.scale > 0.0
    weight = np.abs(null.centre[covered]) / null.scale[covered]
    bound = UNIT_ROUNDOFF * null.alpha_scale * np.linalg.norm(weight @ np.abs(null.dual[covered]))
    return bool(bound < PREDICTION_TOLERANCE / 16.0)


def data_null_directions(model) -> DataNullDirections | None:
    """The tested directions of the published fit (module docstring), or ``None``.

    ``None`` for a fit the test does not read (``_tested``, ``_records``), for
    one with no direction the data and the penalty both leave free, and for
    one whose rows the centre cannot resolve (``_resolvable``): such a fit is
    never tested.
    """
    if not _tested(model):
        return None
    found = _records(model)
    if found is None:
        return None
    result, rank_info, record = found
    dual_local = _dual_null_basis(record)
    if dual_local is None:
        return None
    width = int(np.asarray(result.beta).size)
    selected = np.asarray(rank_info.selected_columns, dtype=np.intp)
    mean_x = np.asarray(rank_info.mean_x, dtype=np.float64)
    if selected.size != record.width or mean_x.shape != (width,):
        return None
    column_scale = np.asarray(record.column_scale, dtype=np.float64)
    local_scale = np.where(column_scale > 0.0, column_scale, 1.0)
    dual_local, outside = _penalty_free(rank_info, dual_local, local_scale)
    if dual_local.shape[1] == 0:
        return None
    from superglm.model.runtime_canonicalize import _public_column_shifts

    terms = (getattr(model, "_runtime_canonical_state", None) or {}).get("terms", {})
    scale = np.zeros(width)
    scale[selected] = local_scale
    dual = np.zeros((width, dual_local.shape[1]))
    dual[selected] = dual_local
    # Scoring uses the original indicators for held prices. In the free design
    # their contribution is an offset, and cannot leave the training span.
    if held_columns := getattr(result, "held_level_columns", ()):
        held = list(held_columns)
        scale[held] = 0.0
        dual[held] = 0.0
        if not np.any(dual):
            return None
    sum_w = float(rank_info.sum_w)
    null = DataNullDirections(
        dual=dual,
        scale=scale,
        centre=mean_x - _public_column_shifts(terms, width),
        alpha_scale=float(np.sqrt(sum_w)) if sum_w > 0.0 else 1.0,
        outside=outside,
    )
    return null if _resolvable(null) else None


def _dense_rows(rows) -> NDArray:
    if hasattr(rows, "toarray"):  # a SciPy sparse block
        rows = rows.toarray()
    return np.asarray(rows, dtype=np.float64)


class _ScaledRows:
    """Per row, the projections ``r N`` and ``||r||^2``, each held as ``value * 2**e``.

    A block of a row's entries is added at the row's running exponent, raised
    first to cover the block's largest entry (``xLASSQ``); every rescaling is
    by a power of two, so it is exact and nothing finite overflows.
    """

    def __init__(self, n: int, k: int) -> None:
        self.exponent = np.zeros(n, dtype=np.intc)
        self.projection = np.zeros((n, k))
        self.square = np.zeros(n)

    def add(self, rows: slice, entries: NDArray, null: NDArray, shift: NDArray) -> None:
        """Add a block of entries ``entries * 2**shift`` (``rows x m``) and their null rows.

        ``shift`` (per row) carries the power of two the caller scaled the
        entries by before dividing them by the column scale.
        """
        _, needed = np.frexp(np.max(np.abs(entries), axis=1, initial=0.0))
        needed = needed + shift
        current = self.exponent[rows]
        raised = np.flatnonzero(needed > current)
        if raised.size:
            at = np.arange(rows.start, rows.stop)[raised]
            change = current[raised] - needed[raised]
            self.projection[at] = np.ldexp(self.projection[at], change[:, None])
            self.square[at] = np.ldexp(self.square[at], 2 * change)
            self.exponent[at] = needed[raised]
        # ``ldexp`` on the entries themselves: exact, and no multiplier that
        # could overflow where the scaled entry is finite
        scaled = np.ldexp(entries, (shift - self.exponent[rows])[:, None])
        if np.any(null):
            self.projection[rows] += scaled @ null
        self.square[rows] += np.einsum("ij,ij->i", scaled, scaled)

    def shares(self) -> NDArray:
        projected = np.sqrt(np.einsum("ij,ij->i", self.projection, self.projection))
        size = np.sqrt(self.square)
        return np.divide(projected, size, out=np.zeros_like(size), where=size > 0.0)


def _population_skipped(spec, random_effects: str) -> bool:
    """Whether a population prediction leaves the term out (``base._predict_eta``)."""
    from superglm.features.random_effect import RandomEffect

    return random_effects == "population" and isinstance(spec, RandomEffect)


def _scaled_entries(rows: NDArray, centre: NDArray, scale: NDArray) -> tuple[NDArray, NDArray]:
    """``((rows - centre) * 2**-e) / scale`` and its per-row ``e``: the dual entries, unscaled later.

    Each row is scaled by a power of two to magnitude at most one before it is
    divided by the column scale, so a finite row cannot overflow there.
    """
    centred = rows - centre
    _, shift = np.frexp(np.max(np.abs(centred), axis=1, initial=0.0))
    centred = np.ldexp(centred, -shift[:, None])
    centred /= scale
    return centred, shift


def prediction_shares(model, frame, plan, random_effects: str) -> NDArray | None:
    """Each predicted row's share ``||r N|| / ||r||`` along the tested directions, or ``None``.

    ``None``, with no pass over the rows, when the fit has no tested direction
    (``data_null_directions``).  Otherwise one pass forms each term's design
    rows in blocks and accumulates both the projection and the norm; a term a
    population prediction leaves out has a zero row.  The rows' unseen-level
    notice was ``predict``'s own to give, so the pass gives none
    (``unseen_routing_quiet``, a context variable: no warning filter moves).
    """
    null = data_null_directions(model)
    if null is None:
        return None
    from superglm.features.categorical import unseen_routing_quiet
    from superglm.model.base import _term_columns

    n = len(frame)
    k = null.dual.shape[1]
    accumulated = _ScaledRows(n, k)
    intercept = np.full((n, 1), 1.0 / null.alpha_scale)
    accumulated.add(slice(0, n), intercept, np.zeros((1, k)), np.zeros(n, dtype=np.intc))
    # A rescaled part may underflow to zero, which the share cannot see; ``errstate``
    # is NumPy's per-context floating-point mode, not a warning filter.
    with unseen_routing_quiet(), np.errstate(under="ignore"):
        for term in (*plan["features"], *plan["interactions"]):
            index = np.asarray(term["beta_idx"], dtype=np.intp)
            covered = null.scale[index] > 0.0
            if not np.any(covered):
                continue
            scale, centre = null.scale[index][covered], null.centre[index][covered]
            term_null = null.dual[index][covered]
            spec = term["spec"]
            block = max(1, _BLOCK_ENTRIES // max(1, index.size))
            skipped = _population_skipped(spec, random_effects)
            columns = None if skipped else _term_columns(term, frame)
            for start in range(0, n, block):
                stop = min(n, start + block)
                rows = slice(start, stop)
                if columns is None:
                    design = np.zeros((stop - start, centre.size))
                else:
                    design = _dense_rows(spec.transform(*(column[rows] for column in columns)))
                    design = design if bool(np.all(covered)) else design[:, covered]
                entries, shift = _scaled_entries(design, centre, scale)
                accumulated.add(rows, entries, term_null, shift)
    return accumulated.shares()


def disclose_non_estimable(model, frame, random_effects: str) -> None:
    """Warn once, naming the predicted rows off the determined space.

    Nothing when the fit has no tested direction, inside
    ``non_estimable_warning_quiet``, or when every row is on the space.
    """
    if _QUIET.get():
        return
    from superglm.model.base import _prediction_plan

    shares = prediction_shares(model, frame, _prediction_plan(model), random_effects)
    if shares is None:
        return
    rows = np.flatnonzero(shares > PREDICTION_TOLERANCE)
    if rows.size == 0:
        return
    from superglm._warn import warn_caller

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
