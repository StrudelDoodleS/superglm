"""Public warmup covers global reducers before the first geometry attempt."""

import subprocess
import sys

import pytest


def test_public_warmup_covers_global_moments_in_a_fresh_process():
    script = r"""
import numpy as np
import superglm
from superglm.distributional.solver import _global_moments as moments
from superglm.distributional.solver._batched_moments import (
    _accumulate_batched, _accumulate_batched_strided,
)

kernels = (moments._finite_bounded_1d, moments._finite_bounded_2d,
           moments._copy_dense_checked, moments._pack_categorical, moments._accumulate_vector,
           moments._accumulate_histogram, moments._accumulate_directional,
           _accumulate_batched, _accumulate_batched_strided)
assert all(not kernel.nopython_signatures for kernel in kernels)
superglm.warmup()
compiled = tuple(tuple(kernel.nopython_signatures) for kernel in kernels)
assert all(compiled)
assert tuple(len(kernel.nopython_signatures) for kernel in kernels[-2:]) == (1, 1)
from tests.test_distributional_automatic_panels import _problem
from tests.test_distributional_global_moment_integration import _row_stream
for companion in ('mixed', 'intercept'):
    layout, score, curvature = _problem(('mixed', companion))
    for readonly in (False, True):
        score.flags.writeable = not readonly
        curvature.flags.writeable = not readonly
        built = moments.build_global_moment_plan(layout, chunk_size=8)
        assert built.plan is not None, built.reason
        plan = built.plan
        try:
            p = layout.n_coefficients
            plan.reset(coefficients=np.zeros(p), penalty=np.zeros((p, p)))
            for chunk in _row_stream(layout, score, curvature):
                plan.add_chunk(chunk.plans, chunk.score_eta, chunk.curvature_packed)
            plan.finish()
        finally:
            plan.close()
assert tuple(tuple(kernel.nopython_signatures) for kernel in kernels) == compiled
superglm.warmup()
assert tuple(tuple(kernel.nopython_signatures) for kernel in kernels) == compiled
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


@pytest.mark.parametrize("fail_first", [False, True])
def test_direct_first_use_initializes_once_and_retries_failed_compilation(fail_first):
    script = r"""
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
from superglm.distributional.solver import _batched_moments as module

kernels = (module._accumulate_batched, module._accumulate_batched_strided)
assert all(not kernel.nopython_signatures for kernel in kernels)
calls = [0, 0]
fail_first = sys.argv[1] == '1'
originals = [kernel.compile for kernel in kernels]
for index, kernel in enumerate(kernels):
    def compile_signature(signature, index=index):
        calls[index] += 1
        if fail_first and index == 1 and calls[index] == 1:
            raise RuntimeError('injected compile failure')
        return originals[index](signature)
    kernel.compile = compile_signature

def build():
    score_out, mass_out = np.zeros(3), np.zeros(3)
    batch = module._BatchedMomentReducers(
        [], [], [], np.array([[0, -1, 1, 0]], dtype=np.intp),
        support_sizes=(3,), n_channels=3,
        vectors=[(0, 0, 0, score_out, mass_out)], n_score_channels=2,
    )
    return batch, score_out, mass_out

first, second = build(), build()
assert all(not kernel.nopython_signatures for kernel in kernels)
score, curvature = np.ones((4, 2)), -np.ones((4, 3), order='F')
try:
    first[0].accumulate(curvature, 4, score=score.astype(np.float32))
except ValueError:
    pass
else:
    raise AssertionError('invalid channels must be refused before compilation')
assert calls == [0, 0]
assert all(not kernel.nopython_signatures for kernel in kernels)
if fail_first:
    try:
        first[0].accumulate(curvature, 4, score=score)
    except RuntimeError as error:
        assert str(error) == 'injected compile failure'
    else:
        raise AssertionError('the compile failure must propagate')
    assert not module._batched_initialized
    assert all(np.all(out == 0) for out in first[1:])

barrier = Barrier(2)
def accumulate(bound):
    barrier.wait()
    return bound[0].accumulate(curvature, 4, score=score)

with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(accumulate, (first, second)))
assert results == [(0, 0), (0, 0)]
assert calls == ([2, 2] if fail_first else [1, 1])
assert module._batched_initialized
assert tuple(len(kernel.nopython_signatures) for kernel in kernels) == (1, 1)
assert all(not kernel._can_compile for kernel in kernels)
assert score.flags.writeable and curvature.flags.writeable
for _, score_out, mass_out in (first, second):
    np.testing.assert_array_equal(score_out, [2., 1., 0.])
    np.testing.assert_array_equal(mass_out, [-2., -1., 0.])

def channels(width, layout, readonly):
    values = (np.ones((4, 2 * width))[:, ::2] if layout == 'A'
              else np.ones((4, width), order=layout))
    values[:, 0] = [-0., 99., -1., 2.]
    values.flags.writeable = not readonly
    return values

for score_layout in ('C', 'F', 'A'):
    for curvature_layout in ('C', 'F', 'A'):
        for score_readonly in (False, True):
            for curvature_readonly in (False, True):
                score = channels(2, score_layout, score_readonly)
                curvature = channels(3, curvature_layout, curvature_readonly)
                score_bytes, curvature_bytes = score.tobytes(), curvature.tobytes()
                for out in first[1:]:
                    out.fill(-0.)
                first[0].accumulate(curvature, 4, score=score)
                for out in first[1:]:
                    np.testing.assert_array_equal(
                        out.view(np.uint64), np.array([2., -1., -0.]).view(np.uint64)
                    )
                assert score.flags.writeable == (not score_readonly)
                assert curvature.flags.writeable == (not curvature_readonly)
                assert score.tobytes() == score_bytes
                assert curvature.tobytes() == curvature_bytes
module._warmup_batched_moments()
assert calls == ([2, 2] if fail_first else [1, 1])
assert tuple(len(kernel.nopython_signatures) for kernel in kernels) == (1, 1)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(int(fail_first))], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
