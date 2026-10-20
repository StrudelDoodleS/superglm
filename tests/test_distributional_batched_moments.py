"""Independent-target native reductions preserve the serial signed row order."""

from importlib import import_module

import numpy as np
import pytest
from numba import config, get_num_threads, set_num_threads


def _module():
    try:
        return import_module("superglm.distributional.solver._batched_moments")
    except ModuleNotFoundError:
        pytest.fail("the owned batched moment reducers have not been implemented")


def _buffered(values):
    """``values`` copied into consecutive views of one owned flat buffer."""
    buffer = np.empty(sum(value.size for value in values))
    views = _module()._carve(buffer, [value.shape for value in values])
    for view, value in zip(views, values, strict=True):
        view[...] = value
    return buffer, views


def _fixture(kind="both"):
    rng = np.random.default_rng(104)
    capacity = 11
    sizes = (3, 7, 2)
    panel_buffer, ordinary = _buffered([np.empty((capacity, width)) for width in (0, 2, 5)])
    bins = np.empty((len(sizes), capacity), dtype=np.intp)
    histograms = [
        (0, 1, 0, rng.normal(size=(3, 7))),
        (2, 0, 2, rng.normal(size=(2, 3))),
    ]
    directions = [
        (0, 1, 1, rng.normal(size=(3, 2))),
        (1, 2, 2, rng.normal(size=(7, 5))),
        (2, 1, 0, rng.normal(size=(2, 2))),
    ]
    if kind in ("directions", "empty"):
        histograms = []
    if kind in ("histograms", "empty"):
        directions = []
    output_buffer, outputs = _buffered([spec[-1] for spec in histograms + directions])
    histograms = [(*spec[:-1], out) for spec, out in zip(histograms, outputs, strict=False)]
    directions = [
        (*spec[:-1], out) for spec, out in zip(directions, outputs[len(histograms) :], strict=True)
    ]
    buffers = {"output_buffer": output_buffer, "panel_buffer": panel_buffer}
    return rng, sizes, ordinary, bins, histograms, directions, buffers


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("kind", ["both", "histograms", "directions", "empty"])
@pytest.mark.threads
def test_batch_matches_serial_signed_masked_multichunk_reductions(threads, kind):
    if threads > config.NUMBA_NUM_THREADS:
        pytest.skip("the configured Numba maximum does not permit four workers")
    module = _module()
    rng, sizes, ordinary, bins, histograms, directions, buffers = _fixture(kind)
    hist_expected = [spec[-1].copy() for spec in histograms]
    direction_expected = [spec[-1].copy() for spec in directions]
    batch = module._BatchedMomentReducers(
        histograms, directions, ordinary, bins, support_sizes=sizes, n_channels=3, **buffers
    )
    previous = get_num_threads()
    try:
        set_num_threads(threads)
        for chunk_index, n in enumerate((11, 8, 11, 1)):
            for g, size in enumerate(sizes):
                bins[g, :n] = rng.integers(-1, size, size=n)
            # Inactive rows, signed weights and unequal ordinary widths all
            # occur within each complete reduction; unused tails are poison.
            bins[:, n:] = 10_000
            for panel in ordinary:
                panel[:n] = rng.normal(size=(n, panel.shape[1]))
                panel[n:] = np.nan
            raw = rng.normal(size=(n, 6))
            weights = raw[:, ::2]
            if chunk_index % 2:
                weights = np.array(weights, order="F")
            weights.flags.writeable = False
            hist_active, direction_work = 0, 0
            for (g, h, channel, _), expected in zip(histograms, hist_expected, strict=True):
                for i in range(n):
                    left, right = bins[g, i], bins[h, i]
                    if left >= 0 and right >= 0:
                        expected[left, right] += weights[i, channel]
                        hist_active += 1
            for (g, a, channel, _), expected in zip(directions, direction_expected, strict=True):
                for i in range(n):
                    index = bins[g, i]
                    if index >= 0:
                        for j in range(ordinary[a].shape[1]):
                            expected[index, j] += weights[i, channel] * ordinary[a][i, j]
                        direction_work += ordinary[a].shape[1]
            assert batch.accumulate(weights, n) == (hist_active, direction_work)
            for spec, expected in zip(histograms, hist_expected, strict=True):
                np.testing.assert_array_equal(spec[-1], expected)
            for spec, expected in zip(directions, direction_expected, strict=True):
                np.testing.assert_array_equal(spec[-1], expected)
            assert batch.last_worker_count == min(threads, len(histograms) + len(directions))
            assert get_num_threads() == threads
    finally:
        set_num_threads(previous)


def test_row_order_fixture_detects_reassociation():
    module = _module()
    bins = np.zeros((1, 3), dtype=np.intp)
    panel_buffer, ordinary = _buffered([np.ones((3, 1))])
    output_buffer, (histogram, direction) = _buffered([np.zeros((1, 1)), np.zeros((1, 1))])
    batch = module._BatchedMomentReducers(
        [(0, 0, 0, histogram)],
        [(0, 0, 0, direction)],
        ordinary,
        bins,
        support_sizes=(1,),
        n_channels=1,
        output_buffer=output_buffer,
        panel_buffer=panel_buffer,
    )
    weights = np.array([[2.0**54], [1.0], [-(2.0**54)]])
    assert batch.accumulate(weights, 3) == (3, 3)
    assert histogram[0, 0] == direction[0, 0] == 0.0
    # A mutation that groups the first and last rows gives 1.0 instead.
    regrouped = (weights[0, 0] + weights[2, 0]) + weights[1, 0]
    assert regrouped == 1.0
    assert histogram[0, 0] != regrouped


@pytest.mark.parametrize(
    "mutation", ["duplicate", "outside", "overlap", "readonly", "dtype", "channel", "shared"]
)
def test_constructor_refuses_unsafe_output_or_metadata_before_native_writes(mutation):
    module = _module()
    _, sizes, ordinary, bins, histograms, directions, buffers = _fixture()
    if mutation == "duplicate":
        histograms.append(histograms[0])
    elif mutation == "shared":
        buffers["panel_buffer"] = buffers["output_buffer"]
    else:
        g, h, channel, out = histograms[0]
        if mutation == "outside":
            out = out.copy()
        elif mutation == "overlap":
            # The second histogram's view, moved back one element into the first's.
            *spec, second = histograms[1]
            start = (second.ctypes.data - buffers["output_buffer"].ctypes.data) // 8 - 1
            moved = buffers["output_buffer"][start : start + second.size].reshape(second.shape)
            histograms[1] = (*spec, moved)
        elif mutation == "readonly":
            out.flags.writeable = False
        elif mutation == "dtype":
            out = out.astype(np.float32)
        elif mutation == "channel":
            channel = 3
        histograms[0] = (g, h, channel, out)
    before = [spec[-1].copy() for spec in histograms + directions]
    with pytest.raises(ValueError):
        module._BatchedMomentReducers(
            histograms, directions, ordinary, bins, support_sizes=sizes, n_channels=3, **buffers
        )
    for spec, expected in zip(histograms + directions, before, strict=True):
        np.testing.assert_array_equal(spec[-1], expected)


def test_constructor_refuses_an_output_that_is_also_an_input_panel():
    module = _module()
    buffer, (panel,) = _buffered([np.ones((3, 3))])
    with pytest.raises(ValueError, match="overlap|alias|distinct|inside"):
        module._BatchedMomentReducers(
            [(0, 0, 0, panel)],
            [],
            [panel],
            np.zeros((1, 3), dtype=np.intp),
            support_sizes=(3,),
            n_channels=1,
            output_buffer=np.empty(9),
            panel_buffer=buffer,
        )


def test_curvature_alias_is_refused_without_mutating_outputs():
    module = _module()
    output_buffer, (output,) = _buffered([np.ones((3, 3))])
    panel_buffer, panels = _buffered([np.ones((3, 1))])
    batch = module._BatchedMomentReducers(
        [(0, 0, 0, output)],
        [],
        panels,
        np.zeros((1, 3), dtype=np.intp),
        support_sizes=(3,),
        n_channels=3,
        output_buffer=output_buffer,
        panel_buffer=panel_buffer,
    )
    with pytest.raises(ValueError, match="overlap|alias"):
        batch.accumulate(output, 3)
    np.testing.assert_array_equal(output, np.ones((3, 3)))
    assert batch.last_worker_count == 0


def test_metadata_accounting_and_warmup_cover_runtime_signatures():
    module = _module()
    module._warmup_batched_moments()
    kernels = (module._accumulate_batched, module._accumulate_batched_strided)
    compiled = tuple(tuple(kernel.nopython_signatures) for kernel in kernels)
    assert tuple(map(len, compiled)) == (1, 1)
    _, sizes, ordinary, bins, histograms, directions, buffers = _fixture()
    bins.fill(0)
    for panel in ordinary:
        panel.fill(1.0)
    batch = module._BatchedMomentReducers(
        histograms, directions, ordinary, bins, support_sizes=sizes, n_channels=3, **buffers
    )
    exact, reserve = module._batch_workspace_bytes(len(histograms), len(directions), len(ordinary))
    assert sum(array.nbytes for array in batch.owned_arrays) == exact
    assert len(batch.owned_arrays) == 3
    assert all(array.flags.owndata for array in batch.owned_arrays)
    assert not batch.owned_arrays[0].flags.writeable
    assert batch.metadata_reserve_bytes == reserve
    for layout in ("C", "F", "A"):
        for readonly in (False, True):
            weights = np.ones((11, 6))[:, ::2] if layout == "A" else np.ones((11, 3), order=layout)
            weights.flags.writeable = not readonly
            batch.accumulate(weights, 11)
    assert tuple(tuple(kernel.nopython_signatures) for kernel in kernels) == compiled
