"""Parallelize independent global moments without changing their row order.

The plan supplies already validated bins and likelihood values in its owned
chunk buffers. Each native worker updates whole, disjoint accumulators, so
signed floating-point additions have exactly the serial reduction order.
Only integer diagnostics are reduced across workers; fastmath is disabled.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numba import get_num_threads, get_thread_id, njit, types
from numba.typed import List

if TYPE_CHECKING:
    from builtins import range as prange

    class _NumbaThreadConfig(Protocol):
        NUMBA_NUM_THREADS: int

    config: _NumbaThreadConfig
else:
    from numba import config, prange


@njit(cache=True)
def _histogram_target(out, bins, curvature, g, h, channel, n):
    active = 0
    for i in range(n):
        left = bins[g, i]
        right = bins[h, i]
        if left >= 0 and right >= 0:
            out[left, right] += curvature[i, channel]
            active += 1
    return active


@njit(cache=True)
def _directional_target(out, bins, panel, curvature, g, channel, n):
    active = 0
    for i in range(n):
        index = bins[g, i]
        if index >= 0:
            weight = curvature[i, channel]
            for j in range(panel.shape[1]):
                out[index, j] += weight * panel[i, j]
            active += 1
    return active * panel.shape[1]


@njit(cache=True)
def _paired_vector_target(score_out, mass_out, bins, score, curvature, g, sc, cc, n):
    active = 0
    for i in range(n):
        index = bins[g, i]
        if index >= 0:
            score_out[index] += score[i, sc]
            mass_out[index] += curvature[i, cc]
            active += 1
    return active


@njit(cache=True, parallel=True)
def _accumulate_batched(
    metadata,
    outputs,
    vector_outputs,
    ordinary,
    bins,
    score,
    curvature,
    n,
    n_histograms,
    n_matrices,
    activity,
    lanes,
):
    for worker in range(activity.size):
        activity[worker] = 0
    histogram_active = 0
    directional_work = 0
    vector_active = 0
    for lane_index in prange(lanes):
        activity[get_thread_id()] = 1
        # Round-robin lanes spread both histogram and wider directional targets
        # across the configured team. Cast prange's possibly unsigned index for
        # typed-list access; each target still visits rows in their input order.
        for target in range(np.intp(lane_index), metadata.shape[0], lanes):
            g = metadata[target, 0]
            other = metadata[target, 1]
            channel = metadata[target, 2]
            if target < n_histograms:
                out = outputs[target]
                histogram_active += _histogram_target(out, bins, curvature, g, other, channel, n)
            elif target < n_matrices:
                out = outputs[target]
                panel = ordinary[other]
                directional_work += _directional_target(out, bins, panel, curvature, g, channel, n)
            else:
                vector = 2 * (target - n_matrices)
                vector_active += _paired_vector_target(
                    vector_outputs[vector],
                    vector_outputs[vector + 1],
                    bins,
                    score,
                    curvature,
                    g,
                    other,
                    channel,
                    n,
                )
    workers = 0
    for worker in range(activity.size):
        workers += activity[worker]
    return histogram_active, directional_work, vector_active, workers


def _batch_workspace_bytes(histogram_count, directional_count, ordinary_count, *, vector_count=0):
    """Return exact owned array bytes and a separate descriptor reserve.

    The two owned arrays contain three indices per target and one activity
    flag per maximum Numba worker. Typed lists borrow existing arrays. Their
    native descriptors, Python wrappers and address tuples receive a bounded
    conservative reserve; global JIT/runtime state belongs to Numba itself.
    Both terms are independent of observation count and chunk length.
    """
    targets = histogram_count + directional_count + vector_count
    owned = np.dtype(np.intp).itemsize * (3 * targets + config.NUMBA_NUM_THREADS)
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

    Construction rejects shared or nonowned output storage before native code
    can write it. Borrowed buffers retain their original native descriptors;
    callers must not resize them or mutate their shape/dtype. Before every
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
        for panel in panels:
            if type(panel) is not np.ndarray or panel.ndim != 2:
                raise ValueError("ordinary panels must be matrices")
            _owned_array(panel, np.dtype(np.float64), (capacity, panel.shape[1]), "ordinary panel")
        specs = tuple(histograms) + tuple(directions)
        n_histograms = len(histograms)
        metadata = np.empty((len(specs) + len(vector_specs), 3), dtype=np.intp)
        # Independent owning allocations cannot overlap. Reject repeated owners
        # explicitly, including an output that is also an input panel.
        seen = {id(bins), *(id(panel) for panel in panels)}
        outputs, intervals = [], []

        def claim_output(out, shape):
            _owned_array(out, np.dtype(np.float64), shape, "moment accumulator")
            if id(out) in seen:
                raise ValueError("moment accumulators must have distinct nonaliasing owners")
            seen.add(id(out))
            if out.size:
                address = out.__array_interface__["data"][0]
                intervals.append((address, address + out.nbytes))

        for target, (g, other, channel, out) in enumerate(specs):
            g = _index(g, len(sizes), "support index")
            channel = _index(channel, n_channels, "channel index")
            if target < n_histograms:
                other = _index(other, len(sizes), "second support index")
                shape = (sizes[g], sizes[other])
            else:
                other = _index(other, len(panels), "ordinary panel index")
                shape = (sizes[g], panels[other].shape[1])
            claim_output(out, shape)
            metadata[target] = (g, other, channel)
            outputs.append(out)
        paired_outputs = []
        for vector, (g, sc, cc, score_out, mass_out) in enumerate(vector_specs):
            g = _index(g, len(sizes), "support index")
            sc = _index(sc, n_score_channels, "score channel index")
            cc = _index(cc, n_channels, "curvature channel index")
            claim_output(score_out, (sizes[g],))
            claim_output(mass_out, (sizes[g],))
            metadata[len(specs) + vector] = (g, sc, cc)
            paired_outputs.extend((score_out, mass_out))
        metadata.flags.writeable = False
        activity = np.empty(config.NUMBA_NUM_THREADS, dtype=np.intp)
        output_list = List.empty_list(types.float64[:, ::1])
        vector_list = List.empty_list(types.float64[::1])
        ordinary_list = List.empty_list(types.float64[:, ::1])
        for out in outputs:
            output_list.append(out)
        for panel in panels:
            ordinary_list.append(panel)
        for out in paired_outputs:
            vector_list.append(out)
        intervals.sort()
        self._output_starts = tuple(start for start, _ in intervals)
        self._output_stops = tuple(stop for _, stop in intervals)
        self._outputs = output_list
        self._vector_outputs = vector_list
        self._ordinary = ordinary_list
        self._bins = bins
        self._metadata = metadata
        self._activity = activity
        self._capacity = capacity
        self._n_channels = int(n_channels)
        self._n_score_channels = n_score_channels
        self._n_histograms = n_histograms
        self._n_matrices = len(specs)
        self._n_vectors = len(vector_specs)
        self.owned_arrays = (metadata, activity)
        _, self.metadata_reserve_bytes = _batch_workspace_bytes(
            n_histograms, len(directions), len(panels), vector_count=len(vector_specs)
        )
        self.last_worker_count = 0
        self.last_vector_active_updates = 0

    def _check_channels(self, values, n, channels, label):
        if (
            type(values) is not np.ndarray
            or values.dtype != np.dtype(np.float64)
            or values.shape != (n, channels)
            or not values.flags.aligned
        ):
            raise ValueError(f"{label} must be an aligned float64 matrix of active rows/channels")
        # A caller-owned derivative view must not alias a parallel output. The
        # enclosing byte interval is conservative even for negative strides;
        # the disjoint sorted outputs permit a logarithmic overlap check.
        lower = values.__array_interface__["data"][0]
        upper = lower + values.itemsize
        for length, stride in zip(values.shape, values.strides, strict=True):
            extent = (length - 1) * stride
            lower += min(0, extent)
            upper += max(0, extent)
        last = bisect_left(self._output_starts, upper) - 1
        if last >= 0 and self._output_stops[last] > lower:
            raise ValueError(f"{label} must not overlap a moment accumulator")

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
        self._check_channels(curvature, n, self._n_channels, "curvature")
        if self._n_vectors or score is not None:
            self._check_channels(score, n, self._n_score_channels, "score")
        else:
            # No paired target reads this operand. Reuse a checked matrix so
            # legacy calls need neither an extra array nor an optional signature.
            score = curvature
        if self._metadata.shape[0] == 0:
            self.last_worker_count = 0
            self.last_vector_active_updates = 0
            return 0, 0
        histogram_active, directional_work, vector_active, workers = _accumulate_batched(
            self._metadata,
            self._outputs,
            self._vector_outputs,
            self._ordinary,
            self._bins,
            score,
            curvature,
            int(n),
            self._n_histograms,
            self._n_matrices,
            self._activity,
            min(get_num_threads(), self._metadata.shape[0]),
        )
        self.last_worker_count = int(workers)
        self.last_vector_active_updates = int(vector_active)
        return int(histogram_active), int(directional_work)


def _warmup_batched_moments():
    """Warm typed lists and independent score/curvature storage signatures."""
    batch = _BatchedMomentReducers(
        [(0, 1, 0, np.zeros((2, 3)))],
        [(0, 0, 1, np.zeros((2, 2)))],
        [np.ones((4, 2))],
        np.zeros((2, 4), dtype=np.intp),
        support_sizes=(2, 3),
        n_channels=3,
        vectors=[(0, 0, 0, np.zeros(2), np.zeros(2))],
        n_score_channels=2,
    )

    def channels(width):
        result = []
        for layout in ("C", "F", "A"):
            for readonly in (False, True):
                values = (
                    np.ones((4, 2 * width))[:, ::2]
                    if layout == "A"
                    else np.ones((4, width), order=layout)
                )
                values.flags.writeable = not readonly
                result.append(values)
        return result

    for score in channels(2):
        for curvature in channels(3):
            batch.accumulate(curvature, 4, score=score)
