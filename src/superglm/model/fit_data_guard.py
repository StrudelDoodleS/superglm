"""Low-overhead mutation guards for retained caller fit inputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _frame_digest(frame: pd.DataFrame) -> bytes:
    """Hash frame contents for a mutation-resistant fit-time fingerprint."""
    digest = hashlib.blake2b(digest_size=16, person=b"superglm-fit-v1")
    metadata = tuple((repr(name), str(dtype)) for name, dtype in frame.dtypes.items())
    digest.update(repr((frame.shape, metadata)).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(
        dtype=np.uint64,
        copy=False,
    )
    digest.update(np.ascontiguousarray(row_hashes).data)
    return digest.digest()


def _same_numeric_vector(value, snapshot: NDArray[np.float64]) -> bool:
    try:
        current = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return False
    return current.shape == snapshot.shape and bool(np.array_equal(current, snapshot))


@dataclass(frozen=True)
class FitDataGuard:
    """Fit-time snapshots sufficient to reject mutated identity-cache inputs."""

    x_digest: bytes
    y_snapshot: NDArray[np.float64]
    x_columns: tuple[object, ...] | None = None

    @classmethod
    def capture(
        cls,
        X: pd.DataFrame,
        y: NDArray,
        *,
        columns: tuple[object, ...] | None = None,
    ) -> FitDataGuard:
        x_columns = None if columns is None else tuple(columns)
        guarded_X = X if x_columns is None else X.loc[:, list(x_columns)]
        x_digest = _frame_digest(guarded_X)
        y_snapshot = np.array(y, dtype=np.float64, copy=True)
        y_snapshot.setflags(write=False)
        return cls(
            x_digest=x_digest,
            y_snapshot=y_snapshot,
            x_columns=x_columns,
        )

    def _matches_frame(self, X: pd.DataFrame) -> bool:
        try:
            guarded_X = X if self.x_columns is None else X.loc[:, list(self.x_columns)]
            return _frame_digest(guarded_X) == self.x_digest
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
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
        if not isinstance(X, pd.DataFrame):
            return False
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
        if not isinstance(X, pd.DataFrame):
            return False
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


__all__ = ["FitDataGuard", "require_unchanged_fit_data"]
