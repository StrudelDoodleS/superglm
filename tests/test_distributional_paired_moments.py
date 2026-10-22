"""Paired native support scores and masses preserve the old vector reductions."""

import inspect

import numpy as np
import pytest

from superglm._parallel import parallel_config
from superglm.distributional.solver import _batched_moments as batch_api


# The serial reductions the batched kernel replaced, as row-order references:
# each adds a target's rows one at a time, in the order the kernel does.
def _accumulate_vector(out, bins, weights):
    active = 0
    for i in range(bins.shape[0]):
        if bins[i] >= 0:
            out[bins[i]] += weights[i]
            active += 1
    return active


def _accumulate_histogram(out, left_bins, right_bins, weights):
    active = 0
    for i in range(left_bins.shape[0]):
        if left_bins[i] >= 0 and right_bins[i] >= 0:
            out[left_bins[i], right_bins[i]] += weights[i]
            active += 1
    return active


def _accumulate_directional(out, bins, ordinary, weights):
    active = 0
    for i in range(bins.shape[0]):
        if bins[i] >= 0:
            for j in range(ordinary.shape[1]):
                out[bins[i], j] += weights[i] * ordinary[i, j]
            active += 1
    return active


def _buffered(values):
    """``values`` copied into consecutive views of one owned flat buffer."""
    buffer = np.empty(sum(value.size for value in values))
    views = batch_api._carve(buffer, [value.shape for value in values])
    for view, value in zip(views, values, strict=True):
        view[...] = value
    return buffer, views


def _fixture(mixed):
    rng = np.random.default_rng(519)
    sizes = tuple(3 + g % 5 for g in range(17))
    capacity = 11
    bins = np.zeros((len(sizes), capacity), dtype=np.intp)
    panel_buffer, ordinary = _buffered([np.ones((capacity, 2)), np.ones((capacity, 5))])
    matrices = [np.zeros((sizes[0], sizes[1])), np.zeros((sizes[2], 5))] if mixed else []
    pairs = [(rng.normal(size=size), rng.normal(size=size)) for size in sizes]
    output_buffer, outputs = _buffered(matrices + [out for pair in pairs for out in pair])
    histograms = [(0, 1, 0, outputs[0])] if mixed else []
    directions = [(2, 1, 2, outputs[1])] if mixed else []
    paired = outputs[len(matrices) :]
    vectors = [(g, g % 2, (g % 2) * 2, paired[2 * g], paired[2 * g + 1]) for g in range(len(sizes))]
    assert vectors[-1][-1].base is output_buffer and ordinary[-1].base is panel_buffer
    return rng, sizes, ordinary, bins, histograms, directions, vectors


def _buffers(ordinary, vectors):
    """The owned buffers behind a fixture's views (the mutations edit only the first vector)."""
    return {"output_buffer": vectors[-1][-1].base, "panel_buffer": ordinary[-1].base}


def _build(fixture):
    _, sizes, ordinary, bins, histograms, directions, vectors = fixture
    assert "vectors" in inspect.signature(batch_api._BatchedMomentReducers).parameters, (
        "the batch does not yet dispatch paired score and diagonal-mass targets"
    )
    return batch_api._BatchedMomentReducers(
        histograms,
        directions,
        ordinary,
        bins,
        support_sizes=sizes,
        n_channels=3,
        vectors=vectors,
        n_score_channels=2,
        **_buffers(ordinary, vectors),
    )


@pytest.mark.parametrize("threads", [1, 2, 16])
@pytest.mark.parametrize("mixed", [False, True])
def test_paired_targets_match_old_vectors_across_chunks_resets_and_threads(threads, mixed):
    fixture = _fixture(mixed)
    rng, sizes, ordinary, bins, histograms, directions, vectors = fixture
    batch = _build(fixture)
    targets = len(histograms) + len(directions) + len(vectors)
    with parallel_config(n_jobs=threads):
        for reset in (False, True):
            if reset:
                for spec in histograms + directions:
                    spec[-1].fill(0)
                for spec in vectors:
                    spec[-2].fill(0)
                    spec[-1].fill(0)
            expected_scores = [spec[-2].copy() for spec in vectors]
            expected_masses = [spec[-1].copy() for spec in vectors]
            expected_histograms = [spec[-1].copy() for spec in histograms]
            expected_directions = [spec[-1].copy() for spec in directions]
            for chunk_index, n in enumerate((11, 5, 11, 1)):
                for g, size in enumerate(sizes):
                    bins[g, :n] = rng.integers(-1, size, size=n)
                bins[-1, :n] = -1  # One completely inactive paired target.
                bins[:, n:] = 10_000  # Unused final rows must never be read.
                for panel in ordinary:
                    panel[:n] = rng.normal(size=(n, panel.shape[1]))
                    panel[n:] = np.nan
                score = rng.normal(size=(n, 4))[:, ::2]
                curvature = np.array(rng.normal(size=(n, 3)), order="F")
                if n >= 3:
                    bins[0, :3] = 0
                    score[:3, 0] = [2.0**54, 1.0, -(2.0**54)]
                    curvature[:3, 0] = [-(2.0**54), -1.0, 2.0**54]
                score.flags.writeable = not (chunk_index % 2)
                curvature.flags.writeable = not (chunk_index % 2)
                vector_active = 0
                for spec, expected_score, expected_mass in zip(
                    vectors, expected_scores, expected_masses, strict=True
                ):
                    g, score_channel, curvature_channel, _, _ = spec
                    vector_active += _accumulate_vector(
                        expected_score, bins[g, :n], score[:, score_channel]
                    )
                    _accumulate_vector(expected_mass, bins[g, :n], curvature[:, curvature_channel])
                hist_active, directional_work = 0, 0
                for (g, h, channel, _), expected in zip(
                    histograms, expected_histograms, strict=True
                ):
                    hist_active += _accumulate_histogram(
                        expected, bins[g, :n], bins[h, :n], curvature[:, channel]
                    )
                for (g, a, channel, _), expected in zip(
                    directions, expected_directions, strict=True
                ):
                    active = _accumulate_directional(
                        expected, bins[g, :n], ordinary[a][:n], curvature[:, channel]
                    )
                    directional_work += active * ordinary[a].shape[1]
                assert batch.accumulate(curvature, n, score=score) == (
                    hist_active,
                    directional_work,
                )
                assert batch.last_vector_active_updates == vector_active
                assert batch.last_worker_count == min(threads, targets)
                for spec, expected_score, expected_mass in zip(
                    vectors, expected_scores, expected_masses, strict=True
                ):
                    np.testing.assert_array_equal(
                        spec[-2].view(np.uint64), expected_score.view(np.uint64)
                    )
                    np.testing.assert_array_equal(
                        spec[-1].view(np.uint64), expected_mass.view(np.uint64)
                    )
                for spec, expected in zip(histograms, expected_histograms, strict=True):
                    np.testing.assert_array_equal(
                        spec[-1].view(np.uint64), expected.view(np.uint64)
                    )
                for spec, expected in zip(directions, expected_directions, strict=True):
                    np.testing.assert_array_equal(
                        spec[-1].view(np.uint64), expected.view(np.uint64)
                    )


@pytest.mark.parametrize("mutation", ["support", "score_channel", "curvature_channel"])
def test_paired_constructor_refuses_indices_outside_the_plan(mutation):
    fixture = _fixture(False)
    vectors = fixture[-1]
    g, sc, cc, score_out, mass_out = vectors[0]
    if mutation == "support":
        g = len(fixture[1])
    elif mutation == "score_channel":
        sc = 2
    elif mutation == "curvature_channel":
        cc = 3
    vectors[0] = (g, sc, cc, score_out, mass_out)
    with pytest.raises(ValueError, match="index"):
        _build(fixture)


@pytest.mark.parametrize("n_score_channels", [None, 0, -1, True, 2.0])
def test_paired_targets_require_a_positive_integer_score_channel_count(n_score_channels):
    _, sizes, ordinary, bins, histograms, directions, vectors = _fixture(False)
    with pytest.raises(ValueError, match="score.*channel"):
        batch_api._BatchedMomentReducers(
            histograms,
            directions,
            ordinary,
            bins,
            support_sizes=sizes,
            n_channels=3,
            vectors=vectors,
            n_score_channels=n_score_channels,
            **_buffers(ordinary, vectors),
        )


@pytest.mark.parametrize("mutation", ["missing", "dtype", "shape", "subclass", "unaligned"])
def test_invalid_score_channels_are_refused_before_any_native_write(mutation):
    fixture = _fixture(True)
    batch = _build(fixture)
    score = np.ones((11, 2))
    if mutation == "missing":
        score = None
    elif mutation == "dtype":
        score = score.astype(np.float32)
    elif mutation == "shape":
        score = score[:, :1]
    elif mutation == "subclass":

        class CustomArray(np.ndarray):
            pass

        score = score.view(CustomArray)
    elif mutation == "unaligned":
        score = np.ndarray((11, 2), dtype=np.float64, buffer=bytearray(11 * 2 * 8 + 1), offset=1)
    arrays = [spec[-1] for spec in fixture[4] + fixture[5]]
    arrays += [out for spec in fixture[-1] for out in spec[-2:]]
    expected = [out.copy() for out in arrays]
    with pytest.raises(ValueError, match="score"):
        batch.accumulate(np.ones((11, 3)), 11, score=score)
    for actual, before in zip(arrays, expected, strict=True):
        np.testing.assert_array_equal(actual, before)


def _channels(layout, readonly, width):
    values = (
        np.ones((11, 2 * width))[:, ::2] if layout == "A" else np.ones((11, width), order=layout)
    )
    values[:] = np.arange(values.size).reshape(values.shape) - 7.0
    values[:4, 0] = [-0.0, 2.0**54, 1.0, -(2.0**54)]
    values.flags.writeable = not readonly
    return values


def test_paired_memory_accounting_and_warmup_cover_independent_channel_layouts(monkeypatch):
    batch_api._warmup_batched_moments()
    kernel = batch_api._accumulate_targets
    signatures = tuple(kernel.nopython_signatures)
    assert len(signatures) == 1
    for channel in signatures[0].args[5:7]:
        assert channel.layout == "C"
        assert not channel.mutable
    calls = [0]

    def counted(*args):
        calls[0] += 1
        return kernel(*args)

    monkeypatch.setattr(batch_api, "_accumulate_targets", counted)
    fixture = _fixture(True)
    batch = _build(fixture)
    exact, reserve = batch_api._batch_workspace_bytes(1, 1, 2, vector_count=17)
    assert sum(array.nbytes for array in batch.owned_arrays) == exact
    assert len(batch.owned_arrays) == 2
    assert batch.metadata_reserve_bytes == reserve
    assert not batch.owned_arrays[0].flags.writeable
    _, _, ordinary, bins, histograms, directions, vectors = fixture
    bins[0, 4:7] = -1
    bins[-1] = -1
    outputs = [spec[-1] for spec in histograms + directions]
    outputs += [out for spec in vectors for out in spec[-2:]]
    for out in outputs:
        out.fill(-0.0)
    expected = [out.copy() for out in outputs]
    for score_layout in ("C", "F", "A"):
        for curvature_layout in ("C", "F", "A"):
            for score_readonly in (False, True):
                for curvature_readonly in (False, True):
                    score = _channels(score_layout, score_readonly, 2)
                    curvature = _channels(curvature_layout, curvature_readonly, 3)
                    before_score = score.tobytes()
                    before_curvature = curvature.tobytes()
                    for (g, h, channel, _), out in zip(histograms, expected[:1], strict=True):
                        _accumulate_histogram(out, bins[g], bins[h], curvature[:, channel])
                    for (g, a, channel, _), out in zip(directions, expected[1:2], strict=True):
                        _accumulate_directional(out, bins[g], ordinary[a], curvature[:, channel])
                    for vector, (g, sc, cc, _, _) in enumerate(vectors):
                        _accumulate_vector(expected[2 + 2 * vector], bins[g], score[:, sc])
                        _accumulate_vector(expected[3 + 2 * vector], bins[g], curvature[:, cc])
                    batch.accumulate(curvature, 11, score=score)
                    assert score.flags.writeable is not score_readonly
                    assert curvature.flags.writeable is not curvature_readonly
                    assert score.tobytes() == before_score
                    assert curvature.tobytes() == before_curvature
                    for actual, oracle in zip(outputs, expected, strict=True):
                        np.testing.assert_array_equal(
                            actual.view(np.uint64), oracle.view(np.uint64)
                        )
    # One call per target per accumulation, through the one signature, for all
    # 36 layout and writeability combinations of the two channel matrices.
    targets = len(histograms) + len(directions) + len(vectors)
    assert calls == [36 * targets]
    assert tuple(kernel.nopython_signatures) == signatures
