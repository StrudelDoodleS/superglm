"""Low-overhead mutation guards for retained caller fit inputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _pandas_copy_on_write_enabled() -> bool:
    """Return whether shallow frames are protected from peer mutation."""
    try:
        major = int(pd.__version__.split(".", maxsplit=1)[0])
    except (TypeError, ValueError):  # pragma: no cover - defensive version parsing
        major = 0
    if major >= 3:
        return True
    try:
        return bool(pd.options.mode.copy_on_write)
    except (AttributeError, TypeError):  # pragma: no cover - pandas < 2
        return False


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


def _same_cow_frame(frame: pd.DataFrame, snapshot: pd.DataFrame) -> bool:
    """Use shared block references to verify legacy copy-on-write guards."""
    if (
        frame.shape != snapshot.shape
        or not frame.columns.equals(snapshot.columns)
        or not frame.index.equals(snapshot.index)
    ):
        return False
    current_blocks = frame._mgr.blocks
    snapshot_blocks = snapshot._mgr.blocks
    if len(current_blocks) != len(snapshot_blocks):
        return False
    for current, retained in zip(current_blocks, snapshot_blocks, strict=True):
        if (
            current.dtype != retained.dtype
            or current.shape != retained.shape
            or not np.array_equal(current.mgr_locs.as_array, retained.mgr_locs.as_array)
            or getattr(current, "refs", None) is not getattr(retained, "refs", None)
        ):
            return False
    return True


@dataclass(frozen=True)
class FitDataGuard:
    """Fit-time snapshots sufficient to reject mutated identity-cache inputs."""

    x_snapshot: pd.DataFrame | None
    x_digest: bytes | None
    y_snapshot: NDArray[np.float64]

    @classmethod
    def capture(cls, X: pd.DataFrame, y: NDArray) -> FitDataGuard:
        x_digest = _frame_digest(X)
        if _pandas_copy_on_write_enabled():
            x_snapshot = X.copy(deep=False)
        else:
            x_snapshot = None
        y_snapshot = np.array(y, dtype=np.float64, copy=True)
        y_snapshot.setflags(write=False)
        return cls(
            x_snapshot=x_snapshot,
            x_digest=x_digest,
            y_snapshot=y_snapshot,
        )

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
        if self.x_digest is not None:
            if _frame_digest(X) != self.x_digest:
                return False
        elif self.x_snapshot is None or not _same_cow_frame(X, self.x_snapshot):
            return False
        return True

    def matches_retained_values(self, X, y) -> bool:
        """Return whether retained feature/response values still match fit time.

        Unlike :meth:`matches`, this path is used before an operation consumes
        retained caller data.  It therefore accepts an equal independent frame,
        including after a model deepcopy or pickle round trip.  A content digest
        remains necessary for copy-on-write frames because writable NumPy aliases
        can modify their blocks without changing pandas' shared-reference token.
        """
        if not isinstance(X, pd.DataFrame):
            return False
        if not _same_numeric_vector(y, self.y_snapshot):
            return False
        if self.x_digest is not None:
            if _frame_digest(X) != self.x_digest:
                return False
        elif self.x_snapshot is None or (
            not _same_cow_frame(X, self.x_snapshot)
            and _frame_digest(X) != _frame_digest(self.x_snapshot)
        ):
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
