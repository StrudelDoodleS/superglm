"""Public warmup covers global reducers before the first geometry attempt."""

import subprocess
import sys


def test_public_warmup_covers_global_moments_in_a_fresh_process():
    script = r"""
import numpy as np
import superglm
from superglm.distributional.solver import _global_moments as moments
from superglm.distributional.solver._batched_moments import _accumulate_batched

kernels = (moments._finite_bounded_1d, moments._finite_bounded_2d,
           moments._pack_categorical, moments._accumulate_vector,
           moments._accumulate_histogram, moments._accumulate_directional,
           _accumulate_batched)
assert all(not kernel.nopython_signatures for kernel in kernels)
superglm.warmup()
compiled = tuple(tuple(kernel.nopython_signatures) for kernel in kernels)
assert all(compiled)
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
