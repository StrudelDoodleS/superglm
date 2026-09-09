"""Paired native support scores and masses preserve the old vector reductions."""

import inspect

import numpy as np
import pytest
from numba import config, get_num_threads, set_num_threads

from superglm.distributional.solver import _batched_moments as batch_api
from superglm.distributional.solver._global_moments import (
    _accumulate_directional,
    _accumulate_histogram,
    _accumulate_vector,
)


def _fixture(mixed):
    rng = np.random.default_rng(519)
    sizes = tuple(3 + g % 5 for g in range(17))
    capacity = 11
    bins = np.zeros((len(sizes), capacity), dtype=np.intp)
    ordinary = [np.ones((capacity, 2)), np.ones((capacity, 5))]
    histograms = [(0, 1, 0, np.zeros((sizes[0], sizes[1])))] if mixed else []
    directions = [(2, 1, 2, np.zeros((sizes[2], 5)))] if mixed else []
    vectors = [
        (g, g % 2, (g % 2) * 2, rng.normal(size=size), rng.normal(size=size))
        for g, size in enumerate(sizes)
    ]
    return rng, sizes, ordinary, bins, histograms, directions, vectors


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
    )


@pytest.mark.parametrize("threads", [1, 2, 16])
@pytest.mark.parametrize("mixed", [False, True])
def test_paired_targets_match_old_vectors_across_chunks_resets_and_threads(threads, mixed):
    if threads > int(getattr(config, "NUMBA_NUM_THREADS")):
        pytest.skip("requested workers exceed the configured Numba maximum")
    fixture = _fixture(mixed)
    rng, sizes, ordinary, bins, histograms, directions, vectors = fixture
    batch = _build(fixture)
    previous = get_num_threads()
    try:
        set_num_threads(threads)
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
                assert batch.last_worker_count == threads
                assert get_num_threads() == threads
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
    finally:
        set_num_threads(previous)


@pytest.mark.parametrize(
    "mutation",
    [
        "support",
        "score_channel",
        "curvature_channel",
        "dtype",
        "view",
        "readonly",
        "shape",
        "nonarray",
        "alias",
        "duplicate",
    ],
)
def test_paired_constructor_refuses_invalid_buffers_and_metadata(mutation):
    fixture = _fixture(False)
    vectors = fixture[-1]
    g, sc, cc, score_out, mass_out = vectors[0]
    if mutation == "support":
        g = len(fixture[1])
    elif mutation == "score_channel":
        sc = 2
    elif mutation == "curvature_channel":
        cc = 3
    elif mutation == "dtype":
        score_out = score_out.astype(np.float32)
    elif mutation == "view":
        mass_out = mass_out.view()
    elif mutation == "readonly":
        mass_out.flags.writeable = False
    elif mutation == "shape":
        score_out = np.zeros(len(score_out) + 1)
    elif mutation == "nonarray":
        score_out = list(score_out)
    elif mutation == "alias":
        mass_out = score_out
    elif mutation == "duplicate":
        vectors.append(vectors[0])
    vectors[0] = (g, sc, cc, score_out, mass_out)
    with pytest.raises(ValueError):
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
        )


@pytest.mark.parametrize("input_kind", ["score", "curvature"])
@pytest.mark.parametrize("output_kind", ["score", "mass", "histogram"])
def test_both_derivative_inputs_refuse_overlap_with_every_output_kind(input_kind, output_kind):
    n = 3
    score_out, mass_out, histogram = np.ones(n), np.ones(n), np.ones((n, 1))
    batch = batch_api._BatchedMomentReducers(
        [(0, 1, 0, histogram)],
        [],
        [],
        np.zeros((2, n), dtype=np.intp),
        support_sizes=(n, 1),
        n_channels=1,
        vectors=[(0, 0, 0, score_out, mass_out)],
        n_score_channels=1,
    )
    derivatives = {"score": np.ones((n, 1)), "curvature": np.ones((n, 1))}
    derivatives[input_kind] = {
        "score": score_out,
        "mass": mass_out,
        "histogram": histogram,
    }[output_kind].reshape(n, 1)
    with pytest.raises(ValueError, match="overlap"):
        batch.accumulate(derivatives["curvature"], n, score=derivatives["score"])
    for output in (score_out, mass_out, histogram):
        np.testing.assert_array_equal(output, 1)
    assert batch.last_vector_active_updates == 0


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
    values.flags.writeable = not readonly
    return values


def test_paired_memory_accounting_and_warmup_cover_independent_channel_layouts():
    batch_api._warmup_batched_moments()
    signatures = tuple(batch_api._accumulate_batched.nopython_signatures)
    fixture = _fixture(True)
    batch = _build(fixture)
    exact, reserve = batch_api._batch_workspace_bytes(1, 1, 2, vector_count=17)
    assert sum(array.nbytes for array in batch.owned_arrays) == exact
    assert len(batch.owned_arrays) == 2
    assert batch.metadata_reserve_bytes == reserve
    assert not batch.owned_arrays[0].flags.writeable
    for score_layout in ("C", "F", "A"):
        for curvature_layout in ("C", "F", "A"):
            for score_readonly in (False, True):
                for curvature_readonly in (False, True):
                    batch.accumulate(
                        _channels(curvature_layout, curvature_readonly, 3),
                        11,
                        score=_channels(score_layout, score_readonly, 2),
                    )
    assert tuple(batch_api._accumulate_batched.nopython_signatures) == signatures
