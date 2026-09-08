"""Public warmup covers panel writers without import-time compilation."""

import subprocess
import sys


def test_public_warmup_covers_panel_writer_layouts_in_a_fresh_process():
    script = r"""
import numpy as np
import superglm
from superglm.distributional.solver import _small_group_panels as panels

writers = (panels._checked_copy, panels._checked_gather,
           panels._checked_scatter, panels._checked_factor_scatter)
assert all(not writer.nopython_signatures for writer in writers)
superglm.warmup()
compiled = tuple(tuple(writer.nopython_signatures) for writer in writers)
assert all(compiled)
rows = np.array([2, 0, 1], dtype=np.intp)
codes = np.array([0, 1, 2], dtype=np.intp)
for layout in ('C', 'F', 'A'):
    for readonly in (False, True):
        if layout == 'A':
            values = np.ones((3, 4))[::-1, ::2]
        else:
            values = np.ones((3, 2), order=layout)
        values.flags.writeable = not readonly
        for strided in (False, True):
            step = 2 if strided else 1
            out = np.empty((3, 2*step))[:, ::step]
            factor = np.empty((3, 4*step))[:, ::step]
            assert panels._checked_copy(values, out)
            assert panels._checked_gather(values, rows, out)
            assert panels._checked_scatter(values, rows, out)
            assert panels._checked_factor_scatter(values, codes, factor, 2, 2, True)
assert tuple(tuple(writer.nopython_signatures) for writer in writers) == compiled
superglm.warmup()
assert tuple(tuple(writer.nopython_signatures) for writer in writers) == compiled
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
