"""Low-overhead mutation guards for retained caller fit inputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameBackend, FrameLike, as_eager_frame


def _same_numeric_vector(value, snapshot: NDArray[np.float64]) -> bool:
    try:
        current = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return False
    return current.shape == snapshot.shape and bool(np.array_equal(current, snapshot))


def _numeric_vector_digest(value) -> tuple[int, bytes]:
    """Return an exact, constant-size digest for one fitted numeric vector."""
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if array.ndim != 1:
        raise ValueError("fit geometry vectors must be one-dimensional")
    digest = hashlib.blake2b(array.data, digest_size=16).digest()
    return len(array), digest


@dataclass(frozen=True)
class FitGeometryGuard:
    """Constant-size fingerprint for inference geometry after row-state release."""

    x_backend: FrameBackend
    x_digest: bytes
    n_rows: int
    weight_digest: bytes
    offset_digest: bytes
    x_columns: tuple[object, ...] | None = None

    @classmethod
    def capture(
        cls,
        X: EagerFrame | FrameLike,
        sample_weight: NDArray,
        offset: NDArray,
        *,
        columns: tuple[object, ...] | None = None,
    ) -> FitGeometryGuard:
        frame = as_eager_frame(X)
        x_columns = None if columns is None else tuple(columns)
        n_weights, weight_digest = _numeric_vector_digest(sample_weight)
        n_offsets, offset_digest = _numeric_vector_digest(offset)
        if n_weights != len(frame) or n_offsets != len(frame):
            raise ValueError("fit geometry vectors must match the fitted frame")
        return cls(
            x_backend=frame.backend,
            x_digest=frame.digest(x_columns),
            n_rows=len(frame),
            weight_digest=weight_digest,
            offset_digest=offset_digest,
            x_columns=x_columns,
        )

    def matches(self, X, sample_weight, offset) -> bool:
        """Return whether evaluation rows, weights, and offset match the fit."""
        try:
            frame = as_eager_frame(X)
            n_weights, weight_digest = _numeric_vector_digest(sample_weight)
            n_offsets, offset_digest = _numeric_vector_digest(offset)
            return bool(
                len(frame) == self.n_rows
                and n_weights == self.n_rows
                and n_offsets == self.n_rows
                and frame.backend == self.x_backend
                and frame.digest(self.x_columns) == self.x_digest
                and weight_digest == self.weight_digest
                and offset_digest == self.offset_digest
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
            return False


@dataclass(frozen=True)
class FitDataGuard:
    """Fit-time snapshots sufficient to reject mutated identity-cache inputs."""

    x_backend: FrameBackend
    x_digest: bytes
    y_snapshot: NDArray[np.float64]
    x_columns: tuple[object, ...] | None = None

    @classmethod
    def capture(
        cls,
        X: EagerFrame | FrameLike,
        y: NDArray,
        *,
        columns: tuple[object, ...] | None = None,
    ) -> FitDataGuard:
        frame = as_eager_frame(X)
        x_columns = None if columns is None else tuple(columns)
        y_snapshot = np.array(y, dtype=np.float64, copy=True)
        y_snapshot.setflags(write=False)
        return cls(
            x_backend=frame.backend,
            x_digest=frame.digest(x_columns),
            y_snapshot=y_snapshot,
            x_columns=x_columns,
        )

    def _matches_frame(self, X: EagerFrame | FrameLike) -> bool:
        try:
            frame = as_eager_frame(X)
            return frame.backend == self.x_backend and frame.digest(self.x_columns) == self.x_digest
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
            return False

    def matches(
        self,
        X,
        y,
        sample_weight,
        offset,
        *,
        fit_weights: NDArray | None,
        fit_offset: NDArray | None,
    ) -> bool:
        """Return whether all identity-matched inputs retain their fit-time values."""
        if not _same_numeric_vector(y, self.y_snapshot):
            return False
        if sample_weight is not None and (
            fit_weights is None or not _same_numeric_vector(sample_weight, fit_weights)
        ):
            return False
        if offset is not None and (
            fit_offset is None or not _same_numeric_vector(offset, fit_offset)
        ):
            return False
        if not self._matches_frame(X):
            return False
        return True

    def matches_retained_values(self, X, y) -> bool:
        """Return whether retained feature/response values still match fit time.

        Unlike :meth:`matches`, this path is used before an operation consumes
        retained caller data.  It therefore accepts an equal independent frame,
        including after a model deepcopy or pickle round trip.  The content digest
        catches both pandas mutations and writable NumPy aliases into a frame's
        backing storage.
        """
        if not _same_numeric_vector(y, self.y_snapshot):
            return False
        if not self._matches_frame(X):
            return False
        return True


def require_unchanged_fit_data(model, X, y) -> None:
    """Reject use of retained caller data that changed after fitting."""
    guard = getattr(model, "_fit_data_guard", None)
    if guard is None or not guard.matches_retained_values(X, y):
        raise RuntimeError(
            "retained fit data were mutated after fitting or cannot be verified; "
            "refit before refreshing fitted state"
        )


__all__ = ["FitDataGuard", "FitGeometryGuard", "require_unchanged_fit_data"]
