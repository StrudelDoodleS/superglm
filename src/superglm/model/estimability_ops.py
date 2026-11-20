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
``predict.lm``.

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
the bases' rounding and any penalised one's is one.  Both records travel with
every fit, a compact one too, so nothing here reads the design.

The design with its intercept, ``[1, X]``, has null vectors ``[-c' n; n]`` for
each such ``n`` (``c`` a centre of the training rows), so a scoring row
``[1, x]`` reads ``(x - c)' n``.  Nothing here forms a decomposition of the
design or reads a training row: a fit with no such direction is not tested.

**The scale.**  The record decided its rank on columns equilibrated by
``D = column_scale`` (the norms of the centred weighted columns); the
intercept's column has norm ``sqrt(sum W)``.  The test is read in that scale,
where the bases are orthonormal: a row is ``[1 / sqrt(sum W), (x - c) / D]``
and a direction ``D n``, normalised.  A column the record holds structurally
null (``D = 0``: constant on the training rows) has unit scale.  Each row is
scaled by a power of two (exact) before it is projected or its norm taken,
LAPACK's ``xLASSQ`` scaled sum of squares, so no finite row overflows.

**Coordinates.**  The record is in the solver's columns.  A spline term whose
public columns lose their training means ``m`` (``runtime_canonicalize``)
keeps its coefficients, so a direction is the same coefficient vector in the
public model and its centre is ``c - m`` there.  Every column is read about
the one centre the record was decided at (its weighted mean ``c``): ``(x -
c)' n`` is the same constant on every training row only for a centre that is
one weighted mean of them, so centres are never mixed.  A binned fit (``discrete=True``) is read as it was fitted: each binned
column's value is replaced by its bin's support point (``_group_matrix_bins.fit_bin_lookup``).
"""

from __future__ import annotations

import contextlib
import contextvars
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import SQRT_EPS

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
    """The tested directions in public coordinates, with the scale they are read in.

    ``dual`` (``p x k``) is orthonormal: ``D n`` on the record's columns, zero
    elsewhere.  ``scale`` is ``D`` (one on a structurally null column, zero on
    a column outside the record, which no direction reaches) and ``centre``
    the public centre rows are read about, ``(x - centre) - centre_lo``.
    ``alpha_scale`` is ``sqrt(sum W)``.
    ``outside`` holds, for every data-null direction, its squared sine to the
    penalised Hessian's null space, ascending; the kept directions are those
    at or below ``FREE_SINE_CUT`` (the restriction's evidence: its gap).
    """

    dual: NDArray
    scale: NDArray
    centre: NDArray
    centre_lo: NDArray
    alpha_scale: float
    outside: NDArray

    @property
    def directions(self) -> NDArray:
        """The raw coefficient vectors ``n = D^-1 (D n)``."""
        safe = np.where(self.scale > 0.0, self.scale, 1.0)
        return self.dual / safe[:, None]


def _record(model):
    result = getattr(model, "_result", None)
    rank_info = getattr(result, "rank_info", None)
    record = getattr(rank_info, "data", None)
    if result is None or rank_info is None or record is None:
        return None
    return result, rank_info, record


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
    is at most ``FREE_SINE_CUT``.  A fit whose penalised Hessian has no record
    has none kept: its penalty cannot be read.
    """
    augmented = getattr(rank_info, "augmented", None)
    if augmented is rank_info.data:  # unpenalised: one decomposition
        return dual, np.zeros(dual.shape[1])
    if augmented is None or augmented.width != dual.shape[0]:
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


def data_null_directions(model) -> DataNullDirections | None:
    """The published fit's tested directions (module docstring), or ``None`` when it has none.

    Reads the data rank and the penalised Hessian's rank the fit decided
    (``RankInfo.data`` and ``RankInfo.augmented``);
    ``None`` for a fit without that record and for one with no direction the
    data and the penalty both leave free, so such a fit is never tested.
    """
    found = _record(model)
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
    pair = getattr(rank_info, "centre_pair", None)
    hi, lo = (mean_x, np.zeros(width)) if pair is None else pair
    centre = np.asarray(hi, dtype=np.float64) - _public_column_shifts(terms, width)
    scale = np.zeros(width)
    scale[selected] = local_scale
    dual = np.zeros((width, dual_local.shape[1]))
    dual[selected] = dual_local
    sum_w = float(rank_info.sum_w)
    return DataNullDirections(
        dual=dual,
        scale=scale,
        centre=centre,
        centre_lo=np.asarray(lo, dtype=np.float64),
        alpha_scale=float(np.sqrt(sum_w)) if sum_w > 0.0 else 1.0,
        outside=outside,
    )


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

    def _raise_to(self, rows: slice, largest: NDArray) -> NDArray:
        """Raise each row's exponent to cover ``largest``; the factor ``2**-e`` its entries take."""
        _, needed = np.frexp(largest)
        current = self.exponent[rows]
        raised = np.flatnonzero(needed > current)
        if raised.size:
            at = np.arange(rows.start, rows.stop)[raised]
            shift = current[raised] - needed[raised]
            self.projection[at] = np.ldexp(self.projection[at], shift[:, None])
            self.square[at] = np.ldexp(self.square[at], 2 * shift)
            self.exponent[at] = needed[raised]
        # a power of two, so multiplying by it is exact (``ldexp``)
        return np.ldexp(1.0, -self.exponent[rows])

    def add(self, rows: slice, entries: NDArray, null: NDArray) -> None:
        """Add a block of dual entries (``rows x m``) and their null rows (``m x k``)."""
        factor = self._raise_to(rows, np.max(np.abs(entries), axis=1, initial=0.0))
        scaled = entries * factor[:, None]
        if np.any(null):
            self.projection[rows] += scaled @ null
        self.square[rows] += np.einsum("ij,ij->i", scaled, scaled)

    def add_projection(self, rows: slice, values: NDArray) -> None:
        """Add products ``r q`` a term gave directly (``rows x k``), with no norm entry."""
        factor = self._raise_to(rows, np.max(np.abs(values), axis=1, initial=0.0))
        self.projection[rows] += values * factor[:, None]

    def shares(self) -> NDArray:
        projected = np.sqrt(np.einsum("ij,ij->i", self.projection, self.projection))
        size = np.sqrt(self.square)
        return np.divide(projected, size, out=np.zeros_like(size), where=size > 0.0)


def _term_inputs(term: dict[str, Any], frame) -> list[NDArray]:
    """The term's resolved input columns, each binned column at its bin's support point."""
    from superglm._group_matrix._group_matrix_bins import binned_values
    from superglm.model.base import _term_columns

    columns = list(_term_columns(term, frame))
    bins = getattr(term["spec"], "_fit_bins", None) or {}
    for axis, lookup in bins.items():
        columns[axis] = binned_values(lookup, columns[axis])
    return columns


def _held_frame(term: dict[str, Any], frame, rows: slice):
    """An ``sz`` term's two input columns on ``rows``, the numeric one binned as the fit read it."""
    import pandas as pd

    from superglm._frame import as_eager_frame
    from superglm._group_matrix._group_matrix_bins import binned_values

    left, right = term["parent_names"]
    values = np.asarray(frame.column_array(left), dtype=np.float64)[rows]
    lookup = (getattr(term["spec"], "_fit_bins", None) or {}).get(0)
    if lookup is not None:
        values = binned_values(lookup, values)
    return as_eager_frame(pd.DataFrame({left: values, right: frame.column_array(right)[rows]}))


def _population_skipped(spec, random_effects: str) -> bool:
    """Whether a population prediction leaves the term out (``base._predict_eta``)."""
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.random_effect import RandomEffect

    if random_effects != "population":
        return False
    if isinstance(spec, RandomEffect):
        return True
    return isinstance(spec, FactorSmooth) and not spec._has_population_offset


def prediction_shares(model, frame, plan, random_effects: str) -> NDArray | None:
    """Each predicted row's share ``||r N|| / ||r||`` along the tested directions, or ``None``.

    ``None``, with no pass over the rows, when the fit has no tested direction
    (``data_null_directions``).  Otherwise one pass forms each term's design
    rows in blocks and accumulates both the projection and the norm.  A term
    a population prediction leaves out has a zero row; an ``sz`` term with
    held levels gives its products as ``predict`` scores it, so a held level
    priced at the population curve reads what that curve reads.
    """
    null = data_null_directions(model)
    if null is None:
        return None
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.model.base import _score_unidentified_factor_smooth

    n = len(frame)
    accumulated = _ScaledRows(n, null.dual.shape[1])
    accumulated.add(
        slice(0, n), np.full((n, 1), 1.0 / null.alpha_scale), np.zeros((1, null.dual.shape[1]))
    )
    features = {feature["name"]: feature for feature in plan["features"]}
    directions = null.directions
    # ``predict``'s own pass has already said what scoring these rows says.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for term in (*plan["features"], *plan["interactions"]):
            index = np.asarray(term["beta_idx"], dtype=np.intp)
            covered = null.scale[index] > 0.0
            if not np.any(covered):
                continue
            every = bool(np.all(covered))
            scale, centre = null.scale[index][covered], null.centre[index][covered]
            centre_lo = null.centre_lo[index][covered]
            term_null = null.dual[index][covered]
            spec = term["spec"]
            block = max(1, _BLOCK_ENTRIES // max(1, index.size))
            held = isinstance(spec, FactorSmooth) and spec._has_population_offset
            skipped = _population_skipped(spec, random_effects)
            inputs = None if skipped else _term_inputs(term, frame)
            for start in range(0, n, block):
                stop = min(n, start + block)
                rows = slice(start, stop)
                if inputs is None:
                    entries = np.broadcast_to(
                        ((0.0 - centre) - centre_lo) / scale, (stop - start, centre.size)
                    )
                    accumulated.add(rows, entries, term_null)
                    continue
                design = _dense_rows(spec.transform(*(column[rows] for column in inputs)))
                entries = (design if every else design[:, covered]) - centre
                entries -= centre_lo
                entries /= scale
                if not held:
                    accumulated.add(rows, entries, term_null)
                    continue
                accumulated.add(rows, entries, np.zeros_like(term_null))
                sub = _held_frame(term, frame, rows)
                products = np.column_stack(
                    [
                        _score_unidentified_factor_smooth(
                            term,
                            sub,
                            directions[:, j],
                            population=random_effects == "population",
                            features=features,
                        )[0]
                        - (centre + centre_lo) @ directions[index][covered, j]
                        for j in range(directions.shape[1])
                    ]
                )
                accumulated.add_projection(rows, products)
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


def term_spec(model, term_name: str):
    """The term's spec as ``predict`` scores it."""
    return _term(model, term_name)["spec"]


def _term(model, term_name: str) -> dict[str, Any]:
    from superglm.model.base import _prediction_plan

    plan = _prediction_plan(model)
    return next(t for t in (*plan["features"], *plan["interactions"]) if t["name"] == term_name)


def term_rows(model, term_name: str, points: Mapping[str, Any] | None = None, *, columns=None):
    """The term's design rows at raw covariate ``points`` (keyed by input column), or at
    already resolved input ``columns`` (a tuple in the term's parent order)."""
    import pandas as pd

    from superglm._frame import as_eager_frame
    from superglm.model.base import _term_columns

    term = _term(model, term_name)
    if columns is None:
        frame = as_eager_frame(
            pd.DataFrame({key: list(values) for key, values in (points or {}).items()})
        )
        columns = _term_columns(term, frame)
    return _dense_rows(term["spec"].transform(*columns))


def functional_shares(model, term_name: str, functionals: NDArray) -> NDArray | None:
    """Each coefficient functional's share along the tested directions, or ``None``.

    ``functionals`` (``entries x p_t``, public coordinates) are rows of the
    term's coefficients, such as a rating-table entry's log relativity.  An
    entry carries no intercept, so its share is ``||(f / D) N|| / ||f / D||``
    on the term's columns (zero for an entry that reads no coefficient, such
    as a base level), scaled per row as the predicted rows are.  ``None``,
    with nothing formed, when the fit has no tested direction.
    """
    null = data_null_directions(model)
    if null is None:
        return None
    index = np.asarray(_term(model, term_name)["beta_idx"], dtype=np.intp)
    covered = null.scale[index] > 0.0
    functionals = np.asarray(functionals, dtype=np.float64)
    accumulated = _ScaledRows(functionals.shape[0], null.dual.shape[1])
    accumulated.add(
        slice(0, functionals.shape[0]),
        functionals[:, covered] / null.scale[index][covered],
        null.dual[index][covered],
    )
    return accumulated.shares()


def functional_marks(model, term_name: str, functionals: NDArray) -> NDArray | None:
    """Which functionals lean on the tested directions (``functional_shares``), or ``None``."""
    shares = functional_shares(model, term_name, functionals)
    return None if shares is None else shares > PREDICTION_TOLERANCE
