"""Per-row truncation windows and censoring flags of a distributional response.

A truncated row was recorded only because its response fell inside its window
``(lower, upper]``, so it contributes ``log f(y) - log P(lower < Y <= upper)``.
A censored row's recorded response is its limit: a row right-censored at ``c``
means ``Y > c`` and contributes ``log P(c < Y <= upper)``; one left-censored at
``c`` means ``Y <= c`` and contributes ``log P(lower < Y <= c)``, each divided
by the row's truncation probability.  Which rows are censored, and on which
side, is the caller's flag, never read from the value.  The bounds describe how
rows were recorded, not the law: a fit with bounds estimates the parent,
untruncated law.

``ResponseBounds`` holds the resolved per-row arrays; ``resolve_response_bounds``
turns the public ``truncation=(lower, upper)`` and ``censored=`` arguments into
it, or into ``None`` when no row is truncated or censored.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

RIGHT_CENSORED = 1
LEFT_CENSORED = -1

BoundPair = tuple[object, object] | None
CensoredRows = ArrayLike | None

_CENSORED_LABELS = {"right": RIGHT_CENSORED, "left": LEFT_CENSORED, "none": 0}
_CENSORED_FORMS = (
    "pass True (right-censored) or False per row, or the labels 'right', 'left' and "
    "'none' (or None)"
)


def _frozen(values: NDArray, dtype: type) -> NDArray:
    """A read-only copy whose buffer is an immutable ``bytes`` object."""
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(array.shape)


def _digest(lower: NDArray, upper: NDArray, side: NDArray) -> str:
    digest = hashlib.sha256(b"ResponseBounds/v1\0")
    for values in (lower, upper, side):
        digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class ResponseBounds:
    """Resolved truncation windows and censoring sides, one entry per retained row.

    ``lower`` is ``-inf`` and ``upper`` is ``+inf`` where a row has no
    truncation bound on that side.  ``side`` is ``+1`` for a row right-censored
    at its response, ``-1`` for one left-censored at it and ``0`` otherwise.
    Equality is by content, through ``digest``.
    """

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    side: NDArray[np.intp]
    digest: str

    @classmethod
    def from_arrays(cls, lower: NDArray, upper: NDArray, side: NDArray) -> ResponseBounds:
        lo = _frozen(lower, np.float64)
        hi = _frozen(upper, np.float64)
        sides = _frozen(side, np.intp)
        if lo.ndim != 1 or lo.shape != hi.shape or lo.shape != sides.shape:
            raise ValueError("response bounds must be matching one-dimensional arrays")
        if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or not np.all(lo < hi):
            raise ValueError("each truncation window must satisfy lower < upper")
        if not np.all(np.isin(sides, (LEFT_CENSORED, 0, RIGHT_CENSORED))):
            raise ValueError("censoring sides must be -1, 0 or +1")
        return cls(lower=lo, upper=hi, side=sides, digest=_digest(lo, hi, sides))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResponseBounds):
            return NotImplemented
        return self.digest == other.digest and len(self.side) == len(other.side)

    def __hash__(self) -> int:
        return hash((self.digest, len(self.side)))

    @property
    def n_rows(self) -> int:
        return len(self.side)

    def truncated_rows(self) -> NDArray[np.intp]:
        return np.flatnonzero((self.lower > -np.inf) | (self.upper < np.inf))

    def censored_rows(self) -> NDArray[np.intp]:
        return np.flatnonzero(self.side != 0)

    def take(self, indices: NDArray[np.integer]) -> ResponseBounds:
        return ResponseBounds.from_arrays(
            self.lower[indices], self.upper[indices], self.side[indices]
        )

    def on_support(self, support_lower: float) -> ResponseBounds:
        """Read lower bounds at or below the family's support as absent."""
        below = self.lower <= support_lower
        if not np.any(below & (self.lower > -np.inf)):
            return self
        return ResponseBounds.from_arrays(
            np.where(below, -np.inf, self.lower), self.upper, self.side
        )

    def observed_window(
        self, response: NDArray, rows: NDArray[np.intp]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(low, high)`` of the observed interval of censored ``rows``."""
        values = np.asarray(response, dtype=np.float64)[rows]
        right = self.side[rows] == RIGHT_CENSORED
        low = np.where(right, values, self.lower[rows])
        high = np.where(right, self.upper[rows], values)
        return low, high


def _pair(value: BoundPair, name: str) -> tuple[object, object]:
    if value is None:
        return None, None
    if not isinstance(value, tuple | list) or len(value) != 2:
        raise TypeError(f"{name} must be a (lower, upper) pair; use None for an open side")
    return value[0], value[1]


def _side_array(value: object, n_rows: int, *, absent: float, name: str) -> NDArray[np.float64]:
    if value is None:
        return np.full(n_rows, absent, dtype=np.float64)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number or one number per row") from exc
    if array.ndim == 0:
        return np.full(n_rows, float(array), dtype=np.float64)
    if array.shape != (n_rows,):
        raise ValueError(f"{name} must be a number or one number per row ({n_rows} rows)")
    return np.array(array, dtype=np.float64, copy=True)


def _first_rows(mask: NDArray[np.bool_]) -> str:
    return f"{int(np.count_nonzero(mask))} row(s), first at input row {int(np.argmax(mask))}"


def _label_side(value: object) -> int | None:
    """The side of one ``censored=`` entry, or ``None`` when it is not a censoring flag."""
    if value is None:
        return 0
    if isinstance(value, bool | np.bool_):
        return RIGHT_CENSORED if value else 0
    if isinstance(value, str):
        return _CENSORED_LABELS.get(value)
    return None


def _refuse_missing_flags(mask: NDArray[np.bool_]) -> None:
    raise ValueError(
        f"censored= is missing (NaN) for {_first_rows(mask)}; give every row a flag, "
        "with False, 'none' or None for an uncensored row"
    )


def _censored_sides(censored: CensoredRows, n_rows: int) -> NDArray[np.intp]:
    """One censoring side per row from the public ``censored=`` flag.

    ``True`` and ``"right"`` mark a right-censored row, ``"left"`` a
    left-censored one, and ``False``, ``"none"`` and ``None`` an uncensored
    one.  A single value applies to every row.  Numbers other than booleans,
    missing entries and unknown labels are refused rather than guessed at.
    """
    if censored is None:
        return np.zeros(n_rows, dtype=np.intp)
    try:
        array = np.asarray(censored)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"censored= must be one flag per row; {_CENSORED_FORMS}") from exc
    if array.ndim == 0:
        array = np.full(n_rows, array.item(), dtype=array.dtype)
    if array.shape != (n_rows,):
        raise ValueError(
            f"censored= has shape {array.shape} for {n_rows} rows; pass one flag per row "
            "of X, or one value for every row"
        )
    if array.dtype == np.bool_:
        return np.where(array, RIGHT_CENSORED, 0).astype(np.intp)
    if array.dtype.kind in "fc" and np.any(np.isnan(array)):
        _refuse_missing_flags(np.isnan(array))
    if array.dtype.kind not in "OU":
        raise TypeError(
            f"censored= holds {array.dtype} values, which do not say which rows are censored; "
            f"{_CENSORED_FORMS}, for example censored=(flag == 1)"
        )
    sides = [_label_side(value) for value in array.tolist()]
    unknown = np.array([side is None for side in sides], dtype=bool)
    if np.any(unknown):
        missing = unknown & np.array([bool(pd.isna(value)) for value in array.tolist()], dtype=bool)
        if np.any(missing):
            _refuse_missing_flags(missing)
        first = array.tolist()[int(np.argmax(unknown))]
        raise ValueError(
            f"censored= has an unknown flag {first!r} for {_first_rows(unknown)}; {_CENSORED_FORMS}"
        )
    return np.asarray(sides, dtype=np.intp)


def unvalidated_bound_arrays(
    truncation: BoundPair, censored: CensoredRows, n_rows: int
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Shape-check and broadcast the public bounds to ``n_rows``; ``None`` when both are absent.

    Returns ``(truncation lower, truncation upper, censoring side)``, with
    infinite entries for open truncation sides, before any row selection.
    """
    if truncation is None and censored is None:
        return None
    t_lower, t_upper = _pair(truncation, "truncation")
    return (
        _side_array(t_lower, n_rows, absent=-np.inf, name="truncation lower bound"),
        _side_array(t_upper, n_rows, absent=np.inf, name="truncation upper bound"),
        _censored_sides(censored, n_rows),
    )


def _refuse_rows(mask: NDArray[np.bool_], positions: NDArray, message: str) -> None:
    if np.any(mask):
        first = int(positions[np.flatnonzero(mask)[0]])
        count = int(np.count_nonzero(mask))
        raise ValueError(f"{count} row(s) {message} (first at input row {first})")


def resolve_response_bounds(
    response: NDArray,
    arrays: tuple[NDArray, NDArray, NDArray] | None,
    positions: NDArray,
) -> ResponseBounds | None:
    """Validate retained-row bounds against the response; ``None`` when no row is bounded.

    ``arrays`` are :func:`unvalidated_bound_arrays` already taken at the
    retained ``positions``, which only name rows in error messages.  The
    censoring side is the caller's flag; a censored row's response is the
    limit it was censored at.  A response outside its truncation window, a
    censored row without a finite response, and a censored row that observed
    nothing inside its window are refused.
    """
    if arrays is None:
        return None
    values = np.asarray(response, dtype=np.float64)
    t_lower, t_upper, side = arrays
    for bound in (t_lower, t_upper):
        _refuse_rows(np.isnan(bound), positions, "have a missing (NaN) bound; use None or inf")
    _refuse_rows(
        ~(t_lower < t_upper), positions, "have a truncation lower bound at or above the upper one"
    )
    _refuse_rows(
        (values < t_lower) | (values > t_upper),
        positions,
        "have a response outside the truncation bounds; a truncated row's response must lie "
        "within them",
    )
    _refuse_rows(
        (side != 0) & ~np.isfinite(values),
        positions,
        "are censored but have no finite response; record a censored row's limit as its response",
    )
    _refuse_rows(
        (side == RIGHT_CENSORED) & (values >= t_upper),
        positions,
        "are right-censored at or above their truncation upper bound, so nothing was observed",
    )
    _refuse_rows(
        (side == LEFT_CENSORED) & (values <= t_lower),
        positions,
        "are left-censored at or below their truncation lower bound, so nothing was observed",
    )
    bounds = ResponseBounds.from_arrays(t_lower, t_upper, side)
    if len(bounds.truncated_rows()) == 0 and len(bounds.censored_rows()) == 0:
        return None
    return bounds


__all__ = [
    "LEFT_CENSORED",
    "RIGHT_CENSORED",
    "CensoredRows",
    "ResponseBounds",
    "resolve_response_bounds",
    "unvalidated_bound_arrays",
]
