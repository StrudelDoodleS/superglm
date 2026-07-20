"""Private eager-dataframe boundary for model-data operations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    import polars as pl

    FrameLike: TypeAlias = pd.DataFrame | pl.DataFrame
else:
    FrameLike: TypeAlias = object

FrameBackend: TypeAlias = Literal["pandas", "polars"]
ColumnKind: TypeAlias = Literal["numeric", "boolean", "categorical", "unsupported"]


@dataclass
class EagerFrame:
    """One operation-local view of a supported native eager dataframe."""

    native: FrameLike
    backend: FrameBackend
    _frame: nw.DataFrame
    _arrays: dict[object, NDArray] = field(default_factory=dict, repr=False)

    @property
    def columns(self) -> tuple[object, ...]:
        """Return column names in native order."""
        return tuple(self._frame.columns)

    def __len__(self) -> int:
        return len(self._frame)

    def require_columns(self, names: tuple[object, ...]) -> None:
        """Raise once with every required column absent from the frame."""
        required = tuple(dict.fromkeys(names))
        available = set(self.columns)
        missing = [name for name in required if name not in available]
        if missing:
            raise ValueError(f"X is missing required columns: {missing}")

    def column_kind(self, name: object) -> ColumnKind:
        """Classify one logical column for feature auto-detection."""
        if self.backend == "pandas":
            dtype = cast(pd.DataFrame, self.native)[cast(Any, name)].dtype
            if pd.api.types.is_bool_dtype(dtype):
                return "boolean"
            if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype):
                return "numeric"
            if (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
            ):
                return "categorical"
            return "unsupported"

        polars_name = cast(str, name)
        dtype = self._frame.schema[polars_name]
        if isinstance(dtype, nw.Boolean):
            return "boolean"
        if isinstance(dtype, nw.dtypes.NumericType):
            return "numeric"
        if isinstance(dtype, nw.String | nw.Categorical | nw.Enum):
            return "categorical"
        return "unsupported"

    def column_dtype(self, name: object) -> str:
        """Return a backend-neutral display name for one logical dtype."""
        if self.backend == "pandas":
            return str(cast(pd.DataFrame, self.native)[cast(Any, name)].dtype)
        return str(self._frame.schema[cast(str, name)])

    def _extract_column(self, name: object) -> NDArray:
        if self.backend == "pandas":
            return np.asarray(cast(pd.DataFrame, self.native)[cast(Any, name)].to_numpy(copy=False))
        return np.asarray(self._frame[cast(str, name)].to_numpy())

    def column_array(self, name: object, *, dtype=None) -> NDArray:
        """Return one logical column, extracting its native data at most once."""
        if name not in self._arrays:
            self._arrays[name] = self._extract_column(name)
        values = self._arrays[name]
        return values if dtype is None else np.asarray(values, dtype=dtype)

    def take_rows(self, indices: NDArray[np.integer]) -> FrameLike:
        """Return a positional native row selection in the same backend."""
        row_indices = np.asarray(indices)
        if row_indices.ndim != 1 or not np.issubdtype(row_indices.dtype, np.integer):
            raise ValueError("row indices must be a one-dimensional integer array")
        if self.backend == "pandas":
            return cast(pd.DataFrame, self.native).iloc[row_indices]
        return cast(Any, self.native)[row_indices.tolist()]

    def select_native(self, columns: tuple[object, ...]) -> FrameLike:
        """Return a non-empty native column selection in requested order."""
        if not columns:
            raise ValueError(
                "empty native column selection cannot preserve Polars row count; "
                "retain the original frame with an explicit empty feature configuration"
            )
        self.require_columns(columns)
        if self.backend == "pandas":
            return cast(pd.DataFrame, self.native).loc[:, list(columns)]
        return cast(Any, self.native).select(list(columns))

    def digest(self, columns: tuple[object, ...] | None = None) -> bytes:
        """Hash selected logical contents for retained fit-data verification."""
        if self.backend == "pandas":
            native = cast(pd.DataFrame, self.native)
            guarded = native if columns is None else native.loc[:, list(columns)]
            digest = hashlib.blake2b(digest_size=16, person=b"superglm-fit-v1")
            metadata = tuple((repr(name), str(dtype)) for name, dtype in guarded.dtypes.items())
            digest.update(repr((guarded.shape, metadata)).encode("utf-8"))
            row_hashes = pd.util.hash_pandas_object(
                guarded,
                index=True,
                categorize=True,
            ).to_numpy(dtype=np.uint64, copy=False)
            digest.update(np.ascontiguousarray(row_hashes).data)
            return digest.digest()

        selected = self.columns if columns is None else tuple(columns)
        self.require_columns(selected)
        digest = hashlib.blake2b(digest_size=16, person=b"superglm-fit-v1")
        metadata = tuple(
            (repr(name), str(self._frame.schema[cast(str, name)])) for name in selected
        )
        digest.update(repr(((len(self), len(selected)), metadata)).encode("utf-8"))
        if selected:
            native = cast(Any, self.native).select(list(selected))
            row_hashes = native.hash_rows(seed=0, seed_1=1, seed_2=2, seed_3=3).to_numpy()
            digest.update(np.ascontiguousarray(row_hashes, dtype=np.uint64).data)
        return digest.digest()


def as_eager_frame(value: Any) -> EagerFrame:
    """Return one adapter for a supported eager native dataframe."""
    if isinstance(value, EagerFrame):
        return value
    if isinstance(value, pd.DataFrame):
        if not value.columns.is_unique:
            raise ValueError("X columns must be unique")
        return EagerFrame(
            native=value,
            backend="pandas",
            _frame=nw.from_native(value, eager_only=True),
        )
    try:
        frame = cast(
            nw.DataFrame,
            nw.from_native(
                value,
                eager_only=True,
                pass_through=False,
                series_only=False,
                allow_series=False,
            ),
        )
    except TypeError as exc:
        try:
            maybe_lazy = nw.from_native(
                value,
                eager_only=False,
                pass_through=False,
                series_only=False,
                allow_series=False,
            )
        except TypeError:
            maybe_lazy = None
        if isinstance(maybe_lazy, nw.LazyFrame) and maybe_lazy.implementation.is_polars():
            raise ValueError(
                "X must be an eager Polars DataFrame; call collect() on the LazyFrame first"
            ) from exc
        raise ValueError("X must be a pandas or eager Polars DataFrame") from exc
    if frame.implementation.is_polars():
        return EagerFrame(native=cast(FrameLike, value), backend="polars", _frame=frame)
    raise ValueError("X must be a pandas or eager Polars DataFrame")


def is_supported_eager_frame(value: object) -> bool:
    """Return whether *value* is a supported eager native dataframe."""
    try:
        as_eager_frame(value)
    except ValueError:
        return False
    return True


__all__ = [
    "ColumnKind",
    "EagerFrame",
    "FrameBackend",
    "FrameLike",
    "as_eager_frame",
    "is_supported_eager_frame",
]
