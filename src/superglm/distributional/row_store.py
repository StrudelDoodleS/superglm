"""Persistent typed rows with bounded, content-verified snapshot reads.

Cached consumers must call ``verify_range`` before reusing a snapshot. The
generation authenticates metadata; only a verified range authenticates payloads.
"""

from __future__ import annotations

import errno
import hashlib
import json
import operator
import os
import stat
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from struct import Struct
from types import MappingProxyType
from typing import Any, SupportsIndex, cast

import numpy as np

_MAX_PAYLOAD = 8 << 20
_MAX_INDEX = 1 << 20
_MAX_MANIFEST = 64 << 10
_MAX_ROWS = 100_000_000
_MAX_BLOCK_ROWS = 65536
_DTYPES = {"<f8": 8, "<i8": 8, "<i4": 4, "|u1": 1}
_RECORD = Struct("<QQQ32s")
_FORMAT = "superglm-row-store/v1"
_DOMAIN = b"SuperGLM/row-store/v1\0"
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]


class RowStoreIntegrityError(ValueError):
    """Stored content no longer satisfies the pinned format or generation."""


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        return operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class RowColumn:
    name: str
    dtype: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("column names must be nonempty strings")
        if type(self.dtype) is not str or self.dtype not in _DTYPES:
            raise ValueError("column dtype must be <f8, <i8, <i4 or |u1")


@dataclass(frozen=True)
class RowRangeAuthority:
    generation: str
    start: int
    stop: int
    block_hashes: tuple[tuple[int, bytes], ...]


@dataclass(frozen=True)
class VerifiedRowBlock:
    start: int
    stop: int
    columns: Mapping[str, np.ndarray]
    authority: RowRangeAuthority

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", MappingProxyType(dict(self.columns)))


def _schema(schema: object) -> tuple[RowColumn, ...]:
    if type(schema) is not tuple:
        raise TypeError("schema must be a tuple of RowColumn records")
    if not 1 <= len(schema) <= 128 or any(type(column) is not RowColumn for column in schema):
        raise ValueError("schema must contain 1 to 128 RowColumn records")
    columns = cast(tuple[RowColumn, ...], schema)
    if len({column.name for column in columns}) != len(columns):
        raise ValueError("column names must be unique")
    if sum(len(column.name) for column in columns) > _MAX_MANIFEST:
        raise ValueError("schema exceeds the manifest byte limit")
    return columns


def _json_metadata(metadata: Mapping[str, JSONValue] | None) -> dict[str, JSONValue]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping with string keys")
    remaining = _MAX_MANIFEST

    def copy(value: Any) -> JSONValue:
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise ValueError("metadata exceeds the manifest byte limit")
        kind = type(value)
        if kind in (type(None), bool, int):
            return value
        if kind is float:
            if not np.isfinite(value):
                raise ValueError("JSON floats must be finite")
            return value
        if kind is str:
            remaining -= len(value)
            if remaining < 0:
                raise ValueError("metadata exceeds the manifest byte limit")
            return value
        if kind is list or kind is dict:
            if len(value) > remaining:
                raise ValueError("metadata exceeds the manifest byte limit")
            if kind is list:
                return [copy(item) for item in value]
            if any(type(key) is not str for key in value):
                raise TypeError("JSON object keys must be strings")
            return {cast(str, copy(key)): copy(item) for key, item in value.items()}
        raise TypeError("metadata must contain only JSON values")

    if len(metadata) > _MAX_MANIFEST:
        raise ValueError("metadata exceeds the manifest byte limit")
    try:
        return cast(dict[str, JSONValue], copy(dict(metadata)))
    except RecursionError as exc:
        raise ValueError("metadata is not a finite encodable JSON value") from exc


def _json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("metadata must be finite UTF-8 JSON") from exc
    if len(encoded) > _MAX_MANIFEST:
        raise ValueError("manifest exceeds 64 KiB")
    return encoded


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _directory(path: Path | str, *, dir_fd: int | None = None) -> int:
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
    except OSError as exc:
        raise RowStoreIntegrityError(f"row store directory is missing or unsafe: {path}") from exc


def _regular_fd(directory: int, name: str, limit: int, exact: int | None = None) -> tuple[int, int]:
    descriptor = None
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= limit:
            raise RowStoreIntegrityError(f"unsafe or oversized row store file: {name}")
        if exact is not None and info.st_size != exact:
            raise RowStoreIntegrityError(f"row store file length changed: {name}")
        return descriptor, info.st_size
    except (OSError, RowStoreIntegrityError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise RowStoreIntegrityError(f"invalid row store file: {name}") from exc


def _read_file(directory: int, name: str, limit: int, exact: int | None = None) -> bytes:
    descriptor, size = _regular_fd(directory, name, limit, exact)
    with os.fdopen(descriptor, "rb", buffering=0) as stream:
        payload = stream.read(size)
        if len(payload) != size or os.fstat(stream.fileno()).st_size != size:
            raise RowStoreIntegrityError(f"row store file changed while reading: {name}")
    return payload


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RowStoreIntegrityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise RowStoreIntegrityError(f"nonfinite JSON constant: {value}")


def _manifest(schema, metadata, n_rows, n_blocks, block_rows, index_sha256) -> bytes:
    return _json_bytes(
        {
            "format": _FORMAT,
            "schema": [{"name": column.name, "dtype": column.dtype} for column in schema],
            "metadata": metadata,
            "n_rows": n_rows,
            "n_blocks": n_blocks,
            "block_rows": block_rows,
            "index_sha256": index_sha256,
        }
    )


@dataclass(frozen=True)
class PreparedRowStore:
    _path: Path
    schema: tuple[RowColumn, ...]
    metadata: Mapping[str, JSONValue]
    n_rows: int
    block_rows: int
    generation: str
    _manifest_bytes: bytes
    _index_bytes: bytes

    @property
    def retained_index_bytes(self) -> int:
        return len(self._index_bytes)

    @classmethod
    def open(cls, path: Path, *, expected_generation: str | None = None) -> PreparedRowStore:
        if expected_generation is not None and type(expected_generation) is not str:
            raise TypeError("expected_generation must be a string or None")
        path = Path(path).absolute()
        directory = _directory(path)
        try:
            manifest = _read_file(directory, "manifest.json", _MAX_MANIFEST)
            try:
                data = json.loads(
                    manifest, object_pairs_hook=_no_duplicates, parse_constant=_bad_constant
                )
                if (
                    type(data) is not dict
                    or set(data)
                    != {
                        "format",
                        "schema",
                        "metadata",
                        "n_rows",
                        "n_blocks",
                        "block_rows",
                        "index_sha256",
                    }
                    or data["format"] != _FORMAT
                ):
                    raise ValueError("unsupported manifest schema")
                if type(data["schema"]) is not list or not 1 <= len(data["schema"]) <= 128:
                    raise ValueError("schema must be an ordered list")
                if any(
                    type(item) is not dict or set(item) != {"name", "dtype"}
                    for item in data["schema"]
                ):
                    raise ValueError("invalid column record")
                schema = _schema(tuple(RowColumn(**item) for item in data["schema"]))
                metadata = _json_metadata(data["metadata"])
                n_rows, count, block_rows = data["n_rows"], data["n_blocks"], data["block_rows"]
                if any(type(value) is not int for value in (n_rows, count, block_rows)):
                    raise ValueError("row counts must be integers")
                width = sum(_DTYPES[column.dtype] for column in schema)
                if not 0 <= n_rows <= _MAX_ROWS or not 1 <= block_rows <= min(
                    _MAX_BLOCK_ROWS, _MAX_PAYLOAD // width
                ):
                    raise ValueError("row or block limit exceeded")
                if (
                    count != (n_rows + block_rows - 1) // block_rows
                    or count * _RECORD.size > _MAX_INDEX
                ):
                    raise ValueError("invalid block/index count")
                if type(data["index_sha256"]) is not str or len(data["index_sha256"]) != 64:
                    raise ValueError("invalid index digest")
                if manifest != _manifest(
                    schema, metadata, n_rows, count, block_rows, data["index_sha256"]
                ):
                    raise ValueError("manifest is not canonical UTF-8 JSON")
            except (TypeError, ValueError, KeyError, RecursionError) as exc:
                raise RowStoreIntegrityError("invalid row store manifest") from exc
            index = _read_file(directory, "index.bin", _MAX_INDEX, count * _RECORD.size)
            if hashlib.sha256(index).hexdigest() != data["index_sha256"]:
                raise RowStoreIntegrityError("row store index digest changed")
            blocks = _directory("blocks", dir_fd=directory)
            try:
                for ordinal in range(count):
                    start, stop, size, _ = _RECORD.unpack_from(index, ordinal * _RECORD.size)
                    expected_start = ordinal * block_rows
                    expected_stop = min(n_rows, expected_start + block_rows)
                    if (start, stop, size) != (
                        expected_start,
                        expected_stop,
                        (expected_stop - expected_start) * width,
                    ):
                        raise RowStoreIntegrityError("index records do not cover the declared rows")
                    descriptor, _ = _regular_fd(blocks, f"{ordinal:08d}.bin", _MAX_PAYLOAD, size)
                    os.close(descriptor)
            finally:
                os.close(blocks)
        finally:
            os.close(directory)
        digest = hashlib.sha256(_DOMAIN + manifest)
        digest.update(index)
        generation = digest.hexdigest()
        if expected_generation is not None and generation != expected_generation:
            raise RowStoreIntegrityError("row store generation mismatch")
        store = cls(
            path, schema, _freeze_json(metadata), n_rows, block_rows, generation, manifest, index
        )
        store.validate_generation()
        return store

    def validate_generation(self) -> None:
        directory = _directory(self._path)
        try:
            manifest = _read_file(
                directory, "manifest.json", _MAX_MANIFEST, len(self._manifest_bytes)
            )
            index = _read_file(directory, "index.bin", _MAX_INDEX, len(self._index_bytes))
            if manifest != self._manifest_bytes or index != self._index_bytes:
                raise RowStoreIntegrityError("row store generation changed")
        finally:
            os.close(directory)

    def _range(self, start: int, stop: int, *, bounded: bool = True) -> tuple[int, int]:
        start, stop = _integer(start, "start"), _integer(stop, "stop")
        if not 0 <= start < stop <= self.n_rows:
            raise ValueError("range must be nonempty and within the stored rows")
        if bounded and stop - start > self.block_rows:
            raise ValueError("range exceeds effective block_rows")
        return start, stop

    def _verify(self, start: int, stop: int, *, assemble: bool):
        start, stop = self._range(start, stop)
        self.validate_generation()
        output = (
            {
                column.name: bytearray((stop - start) * _DTYPES[column.dtype])
                for column in self.schema
            }
            if assemble
            else None
        )
        hashes = []
        directory = _directory(self._path)
        try:
            blocks = _directory("blocks", dir_fd=directory)
            try:
                for ordinal in range(start // self.block_rows, (stop - 1) // self.block_rows + 1):
                    first, last, size, expected = _RECORD.unpack_from(
                        self._index_bytes, ordinal * _RECORD.size
                    )
                    payload = _read_file(blocks, f"{ordinal:08d}.bin", _MAX_PAYLOAD, size)
                    if hashlib.sha256(payload).digest() != expected:
                        raise RowStoreIntegrityError("row payload digest changed")
                    hashes.append((ordinal, expected))
                    if output is not None:
                        left, right = max(start, first), min(stop, last)
                        offset = 0
                        for column in self.schema:
                            itemsize = _DTYPES[column.dtype]
                            output[column.name][
                                (left - start) * itemsize : (right - start) * itemsize
                            ] = memoryview(payload)[
                                offset + (left - first) * itemsize : offset
                                + (right - first) * itemsize
                            ]
                            offset += (last - first) * itemsize
                    del payload
            finally:
                os.close(blocks)
        finally:
            os.close(directory)
        self.validate_generation()
        authority = RowRangeAuthority(self.generation, start, stop, tuple(hashes))
        if output is None:
            return authority
        columns = {
            column.name: np.frombuffer(bytes(output[column.name]), dtype=column.dtype)
            for column in self.schema
        }
        return VerifiedRowBlock(start, stop, columns, authority)

    def read_verified(self, start: int, stop: int) -> VerifiedRowBlock:
        return self._verify(start, stop, assemble=True)

    def verify_range(self, start: int, stop: int) -> RowRangeAuthority:
        return self._verify(start, stop, assemble=False)

    def iter_verified(
        self, *, start: int = 0, stop: int | None = None, batch_rows: int | None = None
    ) -> Iterator[VerifiedRowBlock]:
        start, stop = self._range(start, self.n_rows if stop is None else stop, bounded=False)
        batch_rows = self.block_rows if batch_rows is None else _integer(batch_rows, "batch_rows")
        if not 1 <= batch_rows <= self.block_rows:
            raise ValueError("batch_rows must be within effective block_rows")
        while start < stop:
            end = min(stop, start + batch_rows)
            yield self.read_verified(start, end)
            start = end


class RowStoreWriter:
    def __init__(
        self,
        path: Path,
        schema: tuple[RowColumn, ...],
        *,
        metadata: Mapping[str, JSONValue] | None = None,
        block_rows: int = 65536,
    ):
        self.schema = _schema(schema)
        requested = _integer(block_rows, "block_rows")
        if not 1 <= requested <= _MAX_BLOCK_ROWS:
            raise ValueError("block_rows must be between 1 and 65536")
        self._width = sum(_DTYPES[column.dtype] for column in schema)
        self.block_rows = min(requested, _MAX_PAYLOAD // self._width)
        self._metadata = _json_metadata(metadata)
        _manifest(schema, self._metadata, 0, 0, self.block_rows, "0" * 64)
        self._path = Path(path).absolute()
        try:
            self._path.mkdir()
        except FileExistsError as exc:
            raise ValueError("row store destination already exists") from exc
        self._identity = self._path.stat().st_ino, self._path.stat().st_dev
        (self._path / "blocks").mkdir()
        info = (self._path / "blocks").stat()
        self._blocks_identity = info.st_ino, info.st_dev
        self._index = (self._path / "index.bin").open("xb")
        self._index_hash = hashlib.sha256()
        self._pending = bytearray(self.block_rows * self._width)
        self._filled = self._n_rows = self._n_blocks = 0
        self._created_blocks = 0
        self._published = self._aborted = self._failed = False

    def _active(self) -> None:
        if self._published or self._aborted or self._failed:
            raise ValueError("row writer is no longer accepting data")

    def _owned_directory(self, path, identity, *, dir_fd=None) -> int:
        descriptor = _directory(path, dir_fd=dir_fd)
        info = os.fstat(descriptor)
        if (info.st_ino, info.st_dev) != identity:
            os.close(descriptor)
            raise RowStoreIntegrityError("unpublished writer directory was replaced")
        return descriptor

    def append(self, columns: Mapping[str, np.ndarray]) -> None:
        self._active()
        if not isinstance(columns, Mapping):
            raise TypeError("columns must be a mapping of exact ndarrays")
        if set(columns) != {column.name for column in self.schema}:
            raise ValueError("append columns must match the schema names")
        size = None
        for column in self.schema:
            array = columns[column.name]
            if type(array) is not np.ndarray:
                raise TypeError("append accepts exact ndarrays without coercion")
            if array.ndim != 1 or array.dtype.str != column.dtype:
                raise ValueError("append arrays must be one-dimensional with the schema dtype")
            if size is not None and len(array) != size:
                raise ValueError("append columns must have equal lengths")
            size = len(array)
        size = cast(int, size)  # A validated schema always has at least one column.
        pending = cast(bytearray, self._pending)  # _active excludes finish/abort.
        total = self._n_rows + size
        count = (total + self.block_rows - 1) // self.block_rows
        if total > _MAX_ROWS or count * _RECORD.size > _MAX_INDEX:
            raise ValueError("append exceeds the row or index byte budget")
        _manifest(self.schema, self._metadata, total, count, self.block_rows, "0" * 64)
        cursor = 0
        try:
            while cursor < size:
                take = min(size - cursor, self.block_rows - self._filled)
                offset = 0
                for column in self.schema:
                    itemsize = _DTYPES[column.dtype]
                    first = offset + self._filled * itemsize
                    pending[first : first + take * itemsize] = columns[column.name][
                        cursor : cursor + take
                    ].tobytes(order="C")
                    offset += self.block_rows * itemsize
                self._filled += take
                self._n_rows += take
                cursor += take
                if self._filled == self.block_rows:
                    self._flush()
        except BaseException:
            self._failed = True
            raise

    def _flush(self) -> None:
        if not self._filled:
            return
        digest = hashlib.sha256()
        directory = self._owned_directory(self._path, self._identity)
        try:
            blocks = self._owned_directory("blocks", self._blocks_identity, dir_fd=directory)
            try:
                descriptor = os.open(
                    f"{self._n_blocks:08d}.bin",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o666,
                    dir_fd=blocks,
                )
                with os.fdopen(descriptor, "wb") as stream:
                    self._created_blocks += 1
                    offset = 0
                    for column in self.schema:
                        itemsize = _DTYPES[column.dtype]
                        view = memoryview(cast(bytearray, self._pending))[
                            offset : offset + self._filled * itemsize
                        ]
                        stream.write(view)
                        digest.update(view)
                        offset += self.block_rows * itemsize
            finally:
                os.close(blocks)
        finally:
            os.close(directory)
        record = _RECORD.pack(
            self._n_rows - self._filled, self._n_rows, self._filled * self._width, digest.digest()
        )
        self._index.write(record)
        self._index_hash.update(record)
        self._filled = 0
        self._n_blocks += 1

    def finish(self) -> PreparedRowStore:
        self._active()
        try:
            self._flush()
            self._index.close()
            manifest = _manifest(
                self.schema,
                self._metadata,
                self._n_rows,
                self._n_blocks,
                self.block_rows,
                self._index_hash.hexdigest(),
            )
            directory = self._owned_directory(self._path, self._identity)
            try:
                descriptor = os.open(
                    ".manifest.pending",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o666,
                    dir_fd=directory,
                )
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(manifest)
                # Atomic publication without replacing any existing manifest.
                os.link(
                    ".manifest.pending",
                    "manifest.json",
                    src_dir_fd=directory,
                    dst_dir_fd=directory,
                    follow_symlinks=False,
                )
                self._published = True
                os.unlink(".manifest.pending", dir_fd=directory)
            finally:
                os.close(directory)
            self._pending = None
            digest = hashlib.sha256(_DOMAIN + manifest)
            # The index digest is stored in the manifest, but generation uses
            # its exact raw bytes as required by the format.
            directory = self._owned_directory(self._path, self._identity)
            try:
                index = _read_file(
                    directory, "index.bin", _MAX_INDEX, self._n_blocks * _RECORD.size
                )
                if hashlib.sha256(index).digest() != self._index_hash.digest():
                    raise RowStoreIntegrityError("unpublished writer index changed")
                digest.update(index)
            finally:
                os.close(directory)
            return PreparedRowStore.open(self._path, expected_generation=digest.hexdigest())
        except BaseException:
            self._failed = True
            raise

    def abort(self) -> None:
        if self._published or self._aborted:
            return
        self._index.close()
        directory = self._owned_directory(self._path, self._identity)
        try:
            blocks = self._owned_directory("blocks", self._blocks_identity, dir_fd=directory)
            try:
                for ordinal in range(self._created_blocks):
                    try:
                        os.unlink(f"{ordinal:08d}.bin", dir_fd=blocks)
                    except FileNotFoundError:
                        pass
            finally:
                os.close(blocks)
            for name in ("index.bin", ".manifest.pending"):
                try:
                    os.unlink(name, dir_fd=directory)
                except FileNotFoundError:
                    pass
        finally:
            os.close(directory)
        for directory in (self._path / "blocks", self._path):
            try:
                directory.rmdir()
            except OSError as exc:
                if exc.errno != errno.ENOTEMPTY:
                    raise
        self._pending = None
        self._aborted = True
