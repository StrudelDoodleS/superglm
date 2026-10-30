"""Per-row truncation and censoring bounds of a distributional response.

A truncated row was recorded only because its response fell inside its window
``(lower, upper]``, so it contributes ``log f(y) - log P(lower < Y <= upper)``.
A censored row's response is its censoring bound: right-censored at ``c`` means
``Y > c`` and contributes ``log P(c < Y <= upper)``; left-censored at ``c``
means ``Y <= c`` and contributes ``log P(lower < Y <= c)``, each divided by the
row's truncation probability.  The bounds describe how rows were recorded, not
the law: a fit with bounds estimates the parent, untruncated law.

``ResponseBounds`` holds the resolved per-row arrays; ``resolve_response_bounds``
turns the public ``truncation=(lower, upper)`` and ``censoring=(lower, upper)``
arguments into it, or into ``None`` when no row is truncated or censored.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

RIGHT_CENSORED = 1
LEFT_CENSORED = -1

BoundPair = tuple[object, object] | None


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


def unvalidated_bound_arrays(
    truncation: BoundPair, censoring: BoundPair, n_rows: int
) -> tuple[NDArray[np.float64], ...] | None:
    """Shape-check and broadcast the public bounds to ``n_rows``; ``None`` when both are absent.

    Returns ``(truncation lower, truncation upper, censoring lower, censoring
    upper)`` with infinite entries for open sides, before any row selection.
    """
    if truncation is None and censoring is None:
        return None
    t_lower, t_upper = _pair(truncation, "truncation")
    c_lower, c_upper = _pair(censoring, "censoring")
    return (
        _side_array(t_lower, n_rows, absent=-np.inf, name="truncation lower bound"),
        _side_array(t_upper, n_rows, absent=np.inf, name="truncation upper bound"),
        _side_array(c_lower, n_rows, absent=-np.inf, name="censoring lower bound"),
        _side_array(c_upper, n_rows, absent=np.inf, name="censoring upper bound"),
    )


def _refuse_rows(mask: NDArray[np.bool_], positions: NDArray, message: str) -> None:
    if np.any(mask):
        first = int(positions[np.flatnonzero(mask)[0]])
        count = int(np.count_nonzero(mask))
        raise ValueError(f"{count} row(s) {message} (first at input row {first})")


def resolve_response_bounds(
    response: NDArray,
    arrays: tuple[NDArray[np.float64], ...] | None,
    positions: NDArray,
) -> ResponseBounds | None:
    """Validate retained-row bounds against the response; ``None`` when no row is bounded.

    ``arrays`` are :func:`unvalidated_bound_arrays` already taken at the
    retained ``positions``, which only name rows in error messages.  A
    response equal to a finite censoring bound is censored there; a response
    beyond its censoring bound, or outside its truncation window, is refused.
    """
    if arrays is None:
        return None
    values = np.asarray(response, dtype=np.float64)
    t_lower, t_upper, c_lower, c_upper = arrays
    for bound in arrays:
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
        ~(c_lower < c_upper), positions, "have a censoring lower bound at or above the upper one"
    )
    _refuse_rows(
        (values < c_lower) | (values > c_upper),
        positions,
        "have a response beyond a censoring bound; a censored row's response is its bound, "
        "so record the capped value (for example np.minimum(y, limit))",
    )
    side = np.zeros(len(values), dtype=np.intp)
    side[np.isfinite(c_upper) & (values == c_upper)] = RIGHT_CENSORED
    side[np.isfinite(c_lower) & (values == c_lower)] = LEFT_CENSORED
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
    "ResponseBounds",
    "resolve_response_bounds",
    "unvalidated_bound_arrays",
]
