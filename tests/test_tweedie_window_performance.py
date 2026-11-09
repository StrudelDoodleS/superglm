"""Bounded work/dispatch regressions, separate from numerical oracle tests."""

import sys
from collections import Counter

import numpy as np

from superglm.distributional.kernels._tweedie_windows import tweedie_window_jets


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
            if path.endswith("/_tweedie_windows.py") and name in ("_up", "_add", "_mul"):
                calls["bound"] += 1
            if path.endswith("/_response_tails.py") and name in (
                "log_window_jets",
                "log_mixture_jets",
            ):
                calls["gamma_composition"] += 1

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        got = tweedie_window_jets(*args, order=2)
    finally:
        sys.setprofile(previous)
    assert all(np.isfinite(channel).all() for channel in got)
    assert calls == {}, calls
