"""Predictions that lean on what a fit does not determine (R's ``predict.lm`` ``"non-estim"``).

A fit whose penalised Hessian is singular, or whose estimate is infinite along a separating
direction, determines its linear predictor only on rows orthogonal to those directions.  A row
``r`` (the scoring row, augmented with the intercept) is **off support** for a kind of direction
when its share there exceeds R's tolerance:

    || r Q_k ||_2 > tol || r ||_2,   tol = 1e-6,

with ``Q_k`` an orthonormal basis of that kind of direction in the record's coordinates
(``predict.lm`` with ``rankdeficient = "warnif"``, R >= 4.3, whose ``tol`` this is).  It is a
disclosure: the prediction is still returned (the published gauge's value, or the walk's value
along a separated direction), and no number moves.

The kinds, in the order a row is labelled when it leans on more than one:

- ``"alias"``: a direction of ``null(H)`` (``identifiability.Gauge``), along which the published
  constraint, not the data, fixes the value;
- ``"separated"``: a direction a certified separation leaves infinite
  (``diagnostics.separation.Separation.span``), along which the value is where the walk stopped;
- ``"unresolved"``: a direction the rows cannot resolve at float64, or a separation proposal
  not shown maximal;
- ``"unfitted"``: a direction the fit identifies but did not fit (its coefficients stay where
  the solver stopped).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

# ``predict.lm``'s ``tol`` (R >= 4.3, ``rankdeficient = "warnif"``)
PREDICTION_TOLERANCE = 1e-6

KINDS: tuple[str, ...] = ("alias", "separated", "unresolved", "unfitted")

_REASONS = {
    "alias": "a direction the data do not identify, fixed by the published constraint",
    "separated": "a direction with no finite estimate, at the value where the fit stopped",
    "unresolved": "a direction the fit cannot resolve at float64",
    "unfitted": "a direction the fit identifies but did not fit",
}


class NonEstimablePredictionWarning(UserWarning):
    """Some predicted rows lean on directions the fit does not determine.

    ``rows`` are the row positions (in the order predicted) and ``kinds`` each row's kind
    (``KINDS``), as R's ``"non-estim"`` attribute.  ``no_mean`` counts those rows that have no
    mean at the returned predictor (predicted NaN).  A term reader (a relativity or term curve)
    names its points by term in ``points`` (term -> each point's kind, ``""`` where the fit
    determines it), empty for a prediction.
    """

    def __init__(
        self,
        message: str,
        rows: NDArray,
        kinds: Sequence[str],
        no_mean: int = 0,
        points: Mapping[object, NDArray] | None = None,
    ) -> None:
        super().__init__(message)
        self.rows = np.asarray(rows, dtype=np.intp)
        self.kinds = tuple(kinds)
        self.no_mean = int(no_mean)
        self.points = dict(points or {})


def off_support(
    projections: Mapping[str, NDArray],
    norms: NDArray,
    tol: float = PREDICTION_TOLERANCE,
) -> NDArray:
    """Each row's kind of off-support direction, ``""`` for a row the fit determines.

    ``projections[kind]`` is ``r Q_k`` (``n x k``, or ``n`` for one direction) and ``norms`` is
    ``||r||_2`` per row, both in the record's coordinates.  A row is labelled with the first kind
    in ``KINDS`` whose share exceeds ``tol``.  A row of zero norm is on support.
    """
    norms = np.asarray(norms, dtype=np.float64)
    labels = np.full(norms.shape[0], "", dtype=object)
    for kind in KINDS:
        values = projections.get(kind)
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float64)
        share = np.abs(values) if values.ndim == 1 else np.linalg.norm(values, axis=1)
        labels[(labels == "") & (share > tol * norms)] = kind
    return labels


def reasons(labels: NDArray) -> str:
    """How many labelled rows lean on each kind, in words (``"2 on a direction ..."``)."""
    kinds = [str(kind) for kind in np.asarray(labels, dtype=object) if kind != ""]
    return "; ".join(f"{kinds.count(kind)} on {_REASONS[kind]}" for kind in KINDS if kind in kinds)


def disclosure(labels: NDArray, no_mean: NDArray | None = None) -> NonEstimablePredictionWarning:
    """The one warning a prediction call gives for its off-support rows (``off_support``).

    ``no_mean`` marks the rows with no mean at the returned predictor; those among the
    off-support rows are counted here, so each such row is named in this warning alone.
    """
    labels = np.asarray(labels, dtype=object)
    rows = np.flatnonzero(labels != "")
    kinds = [str(kind) for kind in labels[rows]]
    count = rows.size
    parts = reasons(labels)
    first = ", ".join(str(int(row)) for row in rows[:5]) + (", ..." if count > 5 else "")
    plural = "row leans" if count == 1 else "rows lean"
    message = (
        f"{count} predicted {plural} on what the fit does not determine ({parts}); rows {first}. "
        "Their predictions are returned at the published fit's values; the warning's `rows` "
        "and `kinds` name them."
    )
    absent = 0 if no_mean is None else int(np.count_nonzero(np.asarray(no_mean, bool)[rows]))
    if absent:
        message += (
            f" {absent} of them have no mean at the returned predictor, which lies outside the "
            "inverse link's domain there: predict returns NaN for them."
        )
    return NonEstimablePredictionWarning(message, rows, kinds, absent)
