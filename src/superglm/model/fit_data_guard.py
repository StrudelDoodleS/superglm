"""Low-overhead mutation guards for retained caller fit inputs."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from superglm._frame import (
    EagerFrame,
    FrameBackend,
    FrameLike,
    LegacyDigestUnavailableError,
    as_eager_frame,
)

_UNVERIFIABLE_WARNED = False


def _frame_digest_matches(
    frame: EagerFrame,
    columns: tuple[object, ...] | None,
    expected: bytes,
    scheme: str | None,
    *,
    include_index: bool,
    own_frame: bool = False,
) -> bool | None:
    """Compare a frame with a stored digest written under *scheme*.

    ``scheme`` is None for guards written before the tag existed.  Those used
    Polars' own row hash on Polars frames, which can be recomputed on Polars 1.x
    and not on 2.x.  A digest that cannot be recomputed says nothing about the
    frame.  For the model's own retained frame (``own_frame``) that is not a
    mismatch, so it warns once and lets the retained data stand; any other frame
    is refused with a warning that names the cause, as the data it holds cannot
    be shown to be the training data, and the verdict is None (cannot tell)
    rather than False (differs).
    """
    if scheme == frame.digest_scheme or (scheme is None and frame.backend == "pandas"):
        return frame.digest(columns, include_index=include_index) == expected
    if scheme is None:
        try:
            return frame.legacy_polars_digest(columns) == expected
        except LegacyDigestUnavailableError:
            cause = "was written by Polars 1.x and cannot be checked under this Polars"
    else:
        cause = f"was written under the digest scheme {scheme!r}, which this release cannot check"
    if not own_frame:
        warnings.warn(
            f"The model's fit-time digest {cause}, so a frame other than its retained "
            "one cannot be recognised as the training rows; refit to restore the check.",
            UserWarning,
            stacklevel=4,
        )
        return None
    global _UNVERIFIABLE_WARNED
    if not _UNVERIFIABLE_WARNED:
        _UNVERIFIABLE_WARNED = True
        warnings.warn(
            f"The retained data's fit-time digest {cause}, so the results assume the "
            "retained data are unchanged; refit to restore the check.",
            UserWarning,
            stacklevel=4,
        )
    return True


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
    y_digest: bytes
    weight_digest: bytes
    offset_digest: bytes
    x_columns: tuple[object, ...] | None = None
    x_digest_scheme: str | None = None

    @classmethod
    def capture(
        cls,
        X: EagerFrame | FrameLike,
        y: NDArray,
        sample_weight: NDArray,
        offset: NDArray,
        *,
        columns: tuple[object, ...] | None = None,
    ) -> FitGeometryGuard:
        frame = as_eager_frame(X)
        x_columns = None if columns is None else tuple(columns)
        n_y, y_digest = _numeric_vector_digest(y)
        n_weights, weight_digest = _numeric_vector_digest(sample_weight)
        n_offsets, offset_digest = _numeric_vector_digest(offset)
        if n_y != len(frame) or n_weights != len(frame) or n_offsets != len(frame):
            raise ValueError("fit geometry vectors must match the fitted frame")
        return cls(
            x_backend=frame.backend,
            x_digest=frame.digest(x_columns, include_index=False),
            n_rows=len(frame),
            y_digest=y_digest,
            weight_digest=weight_digest,
            offset_digest=offset_digest,
            x_columns=x_columns,
            x_digest_scheme=frame.digest_scheme,
        )

    def matches(self, X, sample_weight, offset, *, own_frame: bool = False) -> bool:
        """Return whether evaluation rows, weights, and offset match the fit.

        ``own_frame`` says ``X`` is the model's own retained frame, which a
        digest that cannot be recomputed under this Polars may not refuse.
        """
        try:
            frame = as_eager_frame(X)
            n_weights, weight_digest = _numeric_vector_digest(sample_weight)
            n_offsets, offset_digest = _numeric_vector_digest(offset)
            return bool(
                len(frame) == self.n_rows
                and n_weights == self.n_rows
                and n_offsets == self.n_rows
                and frame.backend == self.x_backend
                and weight_digest == self.weight_digest
                and offset_digest == self.offset_digest
                and _frame_digest_matches(
                    frame,
                    self.x_columns,
                    self.x_digest,
                    self.x_digest_scheme,
                    include_index=False,
                    own_frame=own_frame,
                )
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
            return False

    def records_same_fit(
        self, other: FitGeometryGuard, X, *, own_frame: bool = False
    ) -> bool | None:
        """Return whether ``other``, a guard captured from the frame ``X``, records this fit's inputs.

        Guards written under one digest scheme compare by value.  Across schemes
        (a guard saved before the tag) the response, weights and offset still
        compare by value, and this guard's frame digest is checked against ``X``
        as :meth:`matches` checks it; None when that digest cannot be checked
        against ``X`` at all.
        """
        if self.x_digest_scheme == other.x_digest_scheme:
            return self == other
        if replace(self, x_digest=other.x_digest, x_digest_scheme=other.x_digest_scheme) != other:
            return False
        try:
            return _frame_digest_matches(
                as_eager_frame(X),
                self.x_columns,
                self.x_digest,
                self.x_digest_scheme,
                include_index=False,
                own_frame=own_frame,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
            return False

    def matches_training(self, X, y, sample_weight, offset, *, own_frame: bool = False) -> bool:
        """Return whether all supplied rows reproduce the fitted training inputs."""
        try:
            n_y, y_digest = _numeric_vector_digest(y)
            return bool(
                n_y == self.n_rows
                and y_digest == self.y_digest
                and self.matches(X, sample_weight, offset, own_frame=own_frame)
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
    x_digest_scheme: str | None = None

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
            x_digest_scheme=frame.digest_scheme,
        )

    def _matches_frame(self, X: EagerFrame | FrameLike, *, own_frame: bool = False) -> bool:
        try:
            frame = as_eager_frame(X)
            return bool(
                frame.backend == self.x_backend
                and _frame_digest_matches(
                    frame,
                    self.x_columns,
                    self.x_digest,
                    self.x_digest_scheme,
                    include_index=True,
                    own_frame=own_frame,
                )
            )
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
        own_frame: bool = False,
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
        if not self._matches_frame(X, own_frame=own_frame):
            return False
        return True

    def matches_retained_values(self, X, y, *, own_frame: bool = False) -> bool:
        """Return whether retained feature/response values still match fit time.

        Unlike :meth:`matches`, this path is used before an operation consumes
        retained caller data.  It therefore accepts an equal independent frame,
        including after a model deepcopy or pickle round trip.  The content digest
        catches both pandas mutations and writable NumPy aliases into a frame's
        backing storage.  ``own_frame`` says ``X`` is the model's own retained
        frame (see :func:`_frame_digest_matches`).
        """
        if not _same_numeric_vector(y, self.y_snapshot):
            return False
        if not self._matches_frame(X, own_frame=own_frame):
            return False
        return True


def require_unchanged_fit_data(model, X, y) -> None:
    """Reject use of retained caller data that changed after fitting."""
    guard = getattr(model, "_fit_data_guard", None)
    own_frame = X is getattr(model, "_fit_X_ref", None)
    if guard is None or not guard.matches_retained_values(X, y, own_frame=own_frame):
        raise RuntimeError(
            "retained fit data were mutated after fitting or cannot be verified; "
            "refit before refreshing fitted state"
        )


__all__ = ["FitDataGuard", "FitGeometryGuard", "require_unchanged_fit_data"]
