"""Parallelize independent global moments without changing their row order.

The plan supplies already validated bins and likelihood values in its owned
chunk buffers. Each target (a histogram, a directional moment or a paired
score/mass vector) is one task on the worker pool (``superglm._parallel``):
a serial kernel that releases the GIL updates the target's own accumulator,
visiting the rows in their input order, so signed floating-point additions
have exactly the serial reduction order at every worker count. Only integer
diagnostics are summed across tasks; fastmath is disabled.

Accumulators and ordinary panels are views of two owned flat buffers, and the
kernel addresses each through an element offset and a row width.  A
``numba.typed.List`` of the arrays would cost every process a compile:
building one from Python calls numba's own list helpers, which numba does not
cache (``numba/typed/typedlist.py`` declares them without ``cache=True``).
"""

from __future__ import annotations

from threading import Lock

import numpy as np
from numba import njit, types

from superglm._group_matrix._group_matrix_kernels import _native
from superglm._parallel import pool_workers, pooled_for

# Bytes one target's task holds while it runs: its loop state, not its rows.
_TASK_BYTES = 1 << 12


@njit(cache=True)
def _histogram_target(outputs, offset, columns, bins, curvature, g, h, channel, n):
    active = 0
    for i in range(n):
        left = bins[g, i]
        right = bins[h, i]
        if left >= 0 and right >= 0:
            outputs[offset + left * columns + right] += curvature[i, channel]
            active += 1
    return active


@njit(cache=True)
def _directional_target(
    outputs, offset, panels, panel_offset, columns, bins, curvature, g, channel, n
):
    active = 0
    for i in range(n):
        index = bins[g, i]
        if index >= 0:
            weight = curvature[i, channel]
            row = offset + index * columns
            source = panel_offset + i * columns
            # Row slices, not flat offsets: the inner loop then runs over
            # contiguous arrays and vectorises (4x on the global moments).
            target = outputs[row : row + columns]
            panel = panels[source : source + columns]
            for j in range(columns):
                target[j] += weight * panel[j]
            active += 1
    return active * columns


@njit(cache=True)
def _paired_vector_target(outputs, score_offset, mass_offset, bins, score, curvature, g, sc, cc, n):
    scores = outputs[score_offset:]
    mass = outputs[mass_offset:]
    active = 0
    for i in range(n):
        index = bins[g, i]
        if index >= 0:
            scores[index] += score[i, sc]
            mass[index] += curvature[i, cc]
            active += 1
    return active


@njit(cache=True, nogil=True)
def _accumulate_targets(
    metadata,
    panel_layout,
    outputs,
    panels,
    bins,
    score,
    curvature,
    n,
    n_histograms,
    n_matrices,
    first,
    stop,
):
    """Targets ``first..stop``, each over rows ``0..n`` in order: histogram visits,
    directional work and paired-vector visits."""
    histogram_active = 0
    directional_work = 0
    vector_active = 0
    for target in range(first, stop):
        g = metadata[target, 0]
        other = metadata[target, 1]
        channel = metadata[target, 2]
        offset = metadata[target, 3]
        if target < n_histograms:
            histogram_active += _histogram_target(
                outputs, offset, metadata[target, 4], bins, curvature, g, other, channel, n
            )
        elif target < n_matrices:
            directional_work += _directional_target(
                outputs,
                offset,
                panels,
                panel_layout[other, 0],
                panel_layout[other, 1],
                bins,
                curvature,
                g,
                channel,
                n,
            )
        else:
            vector_active += _paired_vector_target(
                outputs,
                offset,
                metadata[target, 4],
                bins,
                score,
                curvature,
                g,
                other,
                channel,
                n,
            )
    return histogram_active, directional_work, vector_active


# The kernel the worker pool runs: one target a task, ``nogil``, never ``prange``.
_POOLED_TARGET_KERNELS = (_accumulate_targets,)
_batched_initialization_lock = Lock()
_batched_initialized = False


def _initialize_batched_moments():
    """Compile the one native signature on warmup or first admitted call.

    ``accumulate`` hands the kernel read-only C-contiguous channels
    (``_native``: a view, or a copy of a strided or Fortran chunk), so one
    signature serves every channel layout a caller passes.
    """
    global _batched_initialized
    if _batched_initialized:
        return
    with _batched_initialization_lock:
        if _batched_initialized:
            return
        table = types.Array(types.intp, 2, "C", readonly=True)
        buffer = types.Array(types.float64, 1, "C")
        channels = types.Array(types.float64, 2, "C", readonly=True)
        _accumulate_targets.compile(
            (
                table,
                table,
                buffer,
                buffer,
                types.Array(types.intp, 2, "C"),
                channels,
                channels,
                types.intp,
                types.intp,
                types.intp,
                types.intp,
                types.intp,
            )
        )
        # A failed compile leaves readiness false and propagates before any
        # accumulation. Retain successful compilation for a subsequent retry;
        # dispatcher identities and Numba's own failure semantics stay intact.
        _accumulate_targets.disable_compile()
        _batched_initialized = True


def _batch_workspace_bytes(histogram_count, directional_count, ordinary_count, *, vector_count=0):
    """Return exact owned array bytes and a separate descriptor reserve.

    The two owned arrays contain five indices per target and an offset and a
    width per ordinary panel. The accumulators and panels are borrowed views
    of the caller's buffers.
    Python wrappers and tuples receive a bounded conservative reserve; global
    JIT/runtime state belongs to Numba itself. Both terms are independent of
    observation count and chunk length.
    """
    targets = histogram_count + directional_count + vector_count
    owned = np.dtype(np.intp).itemsize * (5 * targets + 2 * ordinary_count)
    descriptors = 2048 + 512 * (targets + vector_count + ordinary_count)
    return int(owned), int(descriptors)


def _owned_array(values, dtype, shape, label):
    if (
        type(values) is not np.ndarray
        or values.dtype != dtype
        or values.shape != shape
        or not values.flags.c_contiguous
        or not values.flags.owndata
        or not values.flags.writeable
        or not values.flags.aligned
    ):
        raise ValueError(f"{label} must be an owned writable aligned C array of shape {shape}")


def _view_offset(values, buffer):
    """Element offset of a C-contiguous view (``_carve``'s) inside its buffer."""
    if values.size == 0:
        return 0
    start = values.__array_interface__["data"][0] - buffer.__array_interface__["data"][0]
    return start // buffer.itemsize


def _index(value, upper, label):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= value < upper
    ):
        raise ValueError(f"{label} outside [0, {upper})")
    return int(value)


class _BatchedMomentReducers:
    """Bind certified plan buffers once for one native call per chunk.

    Every accumulator is a view of ``output_buffer`` and every ordinary panel a
    view of ``panel_buffer``, two flat arrays the plan allocates (``_carve``),
    which the kernel addresses by offset. Callers must not resize the buffers
    or mutate their shape/dtype. Before every
    ``accumulate``, the owning plan certifies bins in ``[-1, support_size)``
    and fills/validates the active rows of the ordinary and likelihood panels.
    This private prepared-buffer contract avoids revalidating every support
    index after the plan has already checked and copied it.
    """

    def __init__(
        self,
        histograms,
        directions,
        ordinary,
        bins,
        *,
        output_buffer,
        panel_buffer,
        support_sizes,
        n_channels,
        vectors=(),
        n_score_channels=None,
    ):
        if (
            isinstance(n_channels, (bool, np.bool_))
            or not isinstance(n_channels, (int, np.integer))
            or n_channels <= 0
        ):
            raise ValueError("a positive integer channel count is required")
        vector_specs = tuple(vectors)
        if (vector_specs or n_score_channels is not None) and (
            isinstance(n_score_channels, (bool, np.bool_))
            or not isinstance(n_score_channels, (int, np.integer))
            or n_score_channels <= 0
        ):
            raise ValueError("a positive integer score channel count is required")
        sizes = tuple(support_sizes)
        if any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size <= 0
            for size in sizes
        ):
            raise ValueError("positive integer support sizes are required")
        if type(bins) is not np.ndarray or bins.ndim != 2 or bins.shape[1] <= 0:
            raise ValueError("bins must be a matrix with positive chunk capacity")
        capacity = bins.shape[1]
        _owned_array(bins, np.dtype(np.intp), (len(sizes), capacity), "bins")
        panels = tuple(ordinary)
        panel_layout = np.empty((len(panels), 2), dtype=np.intp)
        for index, panel in enumerate(panels):
            if type(panel) is not np.ndarray or panel.ndim != 2:
                raise ValueError("ordinary panels must be matrices")
            panel_layout[index] = (_view_offset(panel, panel_buffer), panel.shape[1])
        specs = tuple(histograms) + tuple(directions)
        n_histograms = len(histograms)
        metadata = np.empty((len(specs) + len(vector_specs), 5), dtype=np.intp)
        for target, (g, other, channel, out) in enumerate(specs):
            g = _index(g, len(sizes), "support index")
            channel = _index(channel, n_channels, "channel index")
            if target < n_histograms:
                other = _index(other, len(sizes), "second support index")
                shape = (sizes[g], sizes[other])
            else:
                other = _index(other, len(panels), "ordinary panel index")
                shape = (sizes[g], panels[other].shape[1])
            metadata[target] = (g, other, channel, _view_offset(out, output_buffer), shape[1])
        for vector, (g, sc, cc, score_out, mass_out) in enumerate(vector_specs):
            g = _index(g, len(sizes), "support index")
            sc = _index(sc, n_score_channels, "score channel index")
            cc = _index(cc, n_channels, "curvature channel index")
            metadata[len(specs) + vector] = (
                g,
                sc,
                cc,
                _view_offset(score_out, output_buffer),
                _view_offset(mass_out, output_buffer),
            )
        metadata.flags.writeable = False
        panel_layout.flags.writeable = False
        self._output_buffer = output_buffer
        self._panel_buffer = panel_buffer
        self._panel_layout = panel_layout
        self._bins = bins
        self._metadata = metadata
        self._capacity = capacity
        self._n_channels = int(n_channels)
        self._n_score_channels = n_score_channels
        self._n_histograms = n_histograms
        self._n_matrices = len(specs)
        self._n_vectors = len(vector_specs)
        self.owned_arrays = (metadata, panel_layout)
        _, self.metadata_reserve_bytes = _batch_workspace_bytes(
            n_histograms, len(directions), len(panels), vector_count=len(vector_specs)
        )
        self.last_worker_count = 0
        self.last_vector_active_updates = 0

    def _check_channels(self, values, n, channels, label) -> np.ndarray:
        if (
            type(values) is not np.ndarray
            or values.dtype != np.dtype(np.float64)
            or values.shape != (n, channels)
            or not values.flags.aligned
        ):
            raise ValueError(f"{label} must be an aligned float64 matrix of active rows/channels")
        return values

    def accumulate(self, curvature, n, *, score=None):
        """Update prepared rows and return histogram visits and directional work.

        Paired targets require the score matrix too. Their two outputs are
        independent accumulators with identical masks, updated in row order.
        ``last_vector_active_updates`` counts paired active rows separately.
        """
        if (
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or not 1 <= n <= self._capacity
        ):
            raise ValueError("active rows must be within the owned chunk capacity")
        curvature = self._check_channels(curvature, n, self._n_channels, "curvature")
        if self._n_vectors or score is not None:
            score = self._check_channels(score, n, self._n_score_channels, "score")
        else:
            # No paired target reads this operand. Reuse a checked matrix so
            # legacy calls need neither an extra array nor an optional signature.
            score = curvature
        if self._metadata.shape[0] == 0:
            self.last_worker_count = 0
            self.last_vector_active_updates = 0
            return 0, 0
        _initialize_batched_moments()
        operands = (
            self._metadata,
            self._panel_layout,
            self._output_buffer,
            self._panel_buffer,
            self._bins,
            _native(score),
            _native(curvature),
            int(n),
            self._n_histograms,
            self._n_matrices,
        )
        targets = list(range(self._metadata.shape[0]))
        counts = pooled_for(
            lambda target: _accumulate_targets(*operands, target, target + 1),
            targets,
            _TASK_BYTES,
        )
        histogram_active = directional_work = vector_active = 0
        for histogram, directional, vector in counts:
            histogram_active += histogram
            directional_work += directional
            vector_active += vector
        self.last_worker_count = pool_workers(len(targets), _TASK_BYTES)
        self.last_vector_active_updates = int(vector_active)
        return int(histogram_active), int(directional_work)


def _carve(buffer, shapes):
    """Consecutive C-contiguous views of ``buffer`` with the given shapes, in order."""
    views, offset = [], 0
    for shape in shapes:
        size = int(np.prod(shape, dtype=np.int64))
        views.append(buffer[offset : offset + size].reshape(shape))
        offset += size
    return views


def _warmup_batched_moments():
    """Compile the native signature; nothing else compiles at a first fit."""
    _initialize_batched_moments()
