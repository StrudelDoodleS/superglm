"""Bounded, versioned digest tree for live chunk-reuse sources.

Metadata records preserve the caller's field/array order, including every
array occurrence. Separate array leaves permit independent hashing without
caching mutable bytes or making the certificate depend on worker scheduling.
"""

from __future__ import annotations

import hashlib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any, Self, cast

import numpy as np
from numba import get_num_threads
from numpy.typing import NDArray

_VERSION = b"superglm.chunk-reuse.v2\0"


def _array_digest(values: NDArray) -> bytes:
    header = repr((values.dtype.str, values.shape)).encode("utf-8")
    digest = hashlib.sha256(_VERSION + b"array\0")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    if values.flags.c_contiguous and not values.dtype.hasobject:
        # A zero-size multidimensional buffer cannot be cast to bytes. Its
        # shape and dtype still participate, with an empty payload.
        if values.size:
            digest.update(memoryview(values).cast("B"))
    else:
        # Require contiguous chunks so nditer owns the only scratch buffer.
        # Public replacements can have wider dtypes; bound bytes as well as
        # elements. If one element exceeds the budget, an unbuffered scalar
        # iterator exposes source views without allocating a copy.
        buffered = values.itemsize <= 64 * 1024
        # NumPy's stub currently spells the valid "contig" flag "config".
        op_flags = cast(Any, ["readonly", "contig"] if buffered else ["readonly"])
        with np.nditer(
            values,
            flags=["external_loop", "buffered", "zerosize_ok"] if buffered else ["zerosize_ok"],
            op_flags=op_flags,
            order="C",
            buffersize=min(8192, max(1, 64 * 1024 // max(1, values.itemsize))),
        ) as iterator:
            for block in iterator:
                # A single operand yields ndarrays, despite the tuple stub.
                digest.update(memoryview(cast(NDArray, block)).cast("B"))
    return digest.digest()


class _ReuseDigest:
    """Per-certificate worker pool with at most twice its width in flight."""

    def __init__(self) -> None:
        self._workers = 0
        self._executor: ThreadPoolExecutor | None = None
        self._pending: deque[Future[bytes]] = deque()
        self._metadata = hashlib.sha256(_VERSION + b"metadata\0")
        self._leaves = hashlib.sha256(_VERSION + b"leaves\0")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            # A schema refusal can return before hexdigest(). Still observe
            # earlier worker errors, just as a synchronous traversal would.
            if exc_type is None:
                self._drain()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            self._pending.clear()

    def field(self, value: object) -> None:
        encoded = repr((type(value).__module__, type(value).__qualname__, value)).encode("utf-8")
        self._metadata.update(b"F")
        self._metadata.update(len(encoded).to_bytes(8, "big"))
        self._metadata.update(encoded)

    def array(self, values: NDArray) -> None:
        values = np.asarray(values)
        if self._workers == 0:
            self._workers = get_num_threads()
            if self._workers > 1:
                self._executor = ThreadPoolExecutor(max_workers=self._workers)
        if self._executor is None:
            self._leaves.update(_array_digest(values))
        else:
            if len(self._pending) >= 2 * self._workers:
                self._consume_leaf()
            self._pending.append(self._executor.submit(_array_digest, values))
        self._metadata.update(b"A")

    def _consume_leaf(self) -> None:
        # Consume in traversal order, regardless of completion order. The
        # queue retains only bounded references and 32-byte results.
        self._leaves.update(self._pending.popleft().result())

    def _drain(self) -> None:
        while self._pending:
            self._consume_leaf()

    def hexdigest(self) -> str:
        self._drain()
        root = hashlib.sha256(_VERSION + b"root\0")
        root.update(self._metadata.digest())
        root.update(self._leaves.digest())
        return root.hexdigest()
