"""Bounded work/dispatch regressions, separate from numerical oracle tests."""

import sys
from collections import Counter

import numpy as np
import pytest

from superglm.distributional.kernels._tweedie_windows import tweedie_window_jets


def _python_work(path, name):
    path = path.replace("\\", "/")
    if path.endswith("/_tweedie_windows.py") and name in ("_up", "_add", "_mul"):
        return "bound"
    if path.endswith("/_response_tails.py") and name in ("log_window_jets", "log_mixture_jets"):
        return "gamma_composition"
    return None


@pytest.mark.parametrize("root", ["/tmp/superglm/", "C:\\superglm\\"])
@pytest.mark.parametrize(
    "filename,name,want",
    [
        ("_tweedie_windows.py", "_up", "bound"),
        ("_response_tails.py", "log_window_jets", "gamma_composition"),
        ("unrelated.py", "_up", None),
    ],
)
def test_work_guard_recognizes_native_profile_paths(root, filename, name, want):
    assert _python_work(root + filename, name) == want


def test_window_bounds_and_gamma_composition_avoid_scalar_python_arithmetic():
    """Catch reintroducing interpreted bound operations or generic gamma jets.

    Count work, not elapsed time. The independent value/derivative tests live
    in test_tweedie_response_bounds; this only protects the production route.
    """
    args = (np.full(32, 0.5), np.ones(32), np.tile([1.2, 1.1, 1.45], (32, 1)), np.ones(32))
    tweedie_window_jets(*args, order=2)  # exclude first compilation
    calls = Counter()

    def observe(frame, event, arg):
        if event == "call":
            path, name = frame.f_code.co_filename, frame.f_code.co_name
            kind = _python_work(path, name)
            if kind is not None:
                calls[kind] += 1

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        got = tweedie_window_jets(*args, order=2)
    finally:
        sys.setprofile(previous)
    assert all(np.isfinite(channel).all() for channel in got)
    assert calls == {}, calls
