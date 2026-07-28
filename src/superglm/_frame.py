"""Private eager-dataframe boundary for model-data operations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from narwhals.dependencies import get_polars, is_into_dataframe
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
    _frame: nw.DataFrame | None
    _arrays: dict[object, NDArray] = field(default_factory=dict, repr=False)
    _schema: Mapping[str, Any] | None = field(default=None, init=False, repr=False)

    @property
    def _polars_frame(self) -> nw.DataFrame:
        """Return the Narwhals view used only by the Polars backend."""
        if self.backend != "polars" or self._frame is None:
            raise RuntimeError("Narwhals dataframe state is available only for Polars inputs")
        return self._frame

    @property
    def _polars_schema(self) -> Mapping[str, Any]:
        """Return the operation-local Polars schema without repeated full scans."""
        if self._schema is None:
            self._schema = self._polars_frame.schema
        return self._schema

    @property
    def columns(self) -> tuple[object, ...]:
        """Return column names in native order."""
        if self.backend == "pandas":
            return tuple(cast(pd.DataFrame, self.native).columns)
        return tuple(self._polars_frame.columns)

    def __len__(self) -> int:
        if self.backend == "pandas":
            return len(cast(pd.DataFrame, self.native))
        return len(self._polars_frame)

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
        dtype = self._polars_schema[polars_name]
        if isinstance(dtype, nw.Boolean):
            return "boolean"
        if isinstance(dtype, nw.Decimal):
            # pandas stores Decimal values as object/categorical candidates;
            # preserve that established design structure across backends.
            return "categorical"
        if isinstance(dtype, nw.dtypes.NumericType):
            return "numeric"
        if isinstance(dtype, nw.String | nw.Categorical | nw.Enum):
            return "categorical"
        return "unsupported"

    def column_dtype(self, name: object) -> str:
        """Return a backend-neutral display name for one logical dtype."""
        if self.backend == "pandas":
            return str(cast(pd.DataFrame, self.native)[cast(Any, name)].dtype)
        return str(self._polars_schema[cast(str, name)])

    def _extract_column(self, name: object) -> NDArray:
        if self.backend == "pandas":
            return np.asarray(cast(pd.DataFrame, self.native)[cast(Any, name)])
        column = cast(Any, self.native).get_column(cast(str, name))
        return np.asarray(column.to_numpy())

    def column_array(self, name: object, *, dtype=None) -> NDArray:
        """Return one logical column, extracting its native data at most once."""
        if name not in self._arrays:
            if self.backend == "polars" and self.column_kind(name) == "unsupported":
                raise ValueError(
                    f"X column {name!r} has unsupported dtype {self.column_dtype(name)!r}; "
                    "convert it to numeric, boolean, string, categorical, or enum data"
                )
            values = self._extract_column(name)
            if values.ndim != 1:
                raise ValueError(
                    f"X column {name!r} must yield one-dimensional scalar values; "
                    f"got shape {values.shape}"
                )
            self._arrays[name] = values
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

    def digest(
        self,
        columns: tuple[object, ...] | None = None,
        *,
        include_index: bool = True,
    ) -> bytes:
        """Hash selected logical contents for retained fit-data verification."""
        if self.backend == "pandas":
            native = cast(pd.DataFrame, self.native)
            guarded = native if columns is None else native.loc[:, list(columns)]
            digest = hashlib.blake2b(digest_size=16, person=b"superglm-fit-v1")
            metadata = tuple((repr(name), str(dtype)) for name, dtype in guarded.dtypes.items())
            digest.update(repr((guarded.shape, metadata)).encode("utf-8"))
            if guarded.shape[1] or include_index:
                row_hashes = pd.util.hash_pandas_object(
                    guarded,
                    index=include_index,
                    categorize=True,
                ).to_numpy(dtype=np.uint64, copy=False)
                digest.update(np.ascontiguousarray(row_hashes).data)
            return digest.digest()

        selected = self.columns if columns is None else tuple(columns)
        self.require_columns(selected)
        digest = hashlib.blake2b(digest_size=16, person=b"superglm-fit-v1")
        metadata = tuple(
            (repr(name), str(self._polars_schema[cast(str, name)])) for name in selected
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
            _frame=None,
        )
    polars = get_polars()
    if polars is not None and isinstance(value, polars.LazyFrame):
        raise ValueError(
            "X must be an eager Polars DataFrame; call collect() on the LazyFrame first"
        )
    if polars is None or not isinstance(value, polars.DataFrame):
        raise ValueError("X must be a pandas or eager Polars DataFrame")
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
        raise ValueError("X must be a pandas or eager Polars DataFrame") from exc
    if frame.implementation.is_polars():
        return EagerFrame(native=cast(FrameLike, value), backend="polars", _frame=frame)
    raise ValueError("X must be a pandas or eager Polars DataFrame")


def _is_polars_lazy_frame(value: object) -> bool:
    """Identify a loaded Polars LazyFrame without importing Polars."""
    polars = get_polars()
    return polars is not None and isinstance(value, polars.LazyFrame)


def _is_recognized_dataframe(value: object) -> bool:
    """Identify dataframe objects understood by Narwhals, supported or not."""
    return is_into_dataframe(value)


def is_supported_eager_frame(value: object) -> bool:
    """Return whether *value* is a supported eager native dataframe."""
    if isinstance(value, EagerFrame | pd.DataFrame):
        return True
    polars = get_polars()
    return polars is not None and isinstance(value, polars.DataFrame)


__all__ = [
    "ColumnKind",
    "EagerFrame",
    "FrameBackend",
    "FrameLike",
    "as_eager_frame",
    "is_supported_eager_frame",
]
