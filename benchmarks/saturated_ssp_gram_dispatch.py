"""Observe completed SSP Gram products in an untimed complete tensor fit.

Uses the arguments of ``benchmarks.multi_penalty_support``. Select the frozen
package through PYTHONPATH and omit --measure-time. The trace observes NumPy
matrix products after completion; it does not intercept native BLAS symbols.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from benchmarks import multi_penalty_support as benchmark

from superglm._group_matrix import _group_matrix_core as core


def _line(function, text):
    lines, start = inspect.getsourcelines(function)
    matches = [start + index for index, line in enumerate(lines) if text in line]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot identify the completed product in {function.__name__}")
    return matches[0]


@contextmanager
def _observe(sampler=None):
    gram = core.SparseSSPGroupMatrix.gram
    blocked = getattr(core, "_saturated_ssp_gram", None)
    product_function = gram if blocked is None else blocked
    product_line = _line(product_function, "lower =" if blocked is None else "del dense")
    calls = Counter()
    products = Counter()
    record = {
        "calls": calls,
        "completed_dense_products": products,
        "row_workspace_budget_bytes": getattr(core, "_MAX_SSP_GRAM_WORKSPACE_BYTES", None),
        "native_blas_symbols_intercepted": False,
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    codes = {
        gram.__code__: "ssp_gram",
        core._exact_ssp_moments.__code__: "exact_ssp_moments",
    }
    if blocked is not None:
        codes[blocked.__code__] = "saturated_ssp_gram"

    def trace(frame, event, arg):
        if event == "call":
            name = codes.get(frame.f_code)
            if name is not None:
                calls[name] += 1
            return trace if frame.f_code is product_function.__code__ else None
        if event == "line" and frame.f_lineno == product_line:
            dense = frame.f_locals["dense"]
            # This line follows the actual weighted matrix multiplication.
            # Record only metadata; never retain a basis or result buffer.
            shape = tuple(int(value) for value in dense.shape)
            products[f"shape={shape},dtype={dense.dtype},strides={dense.strides}"] += 1
            if sampler is not None:
                sampler.sample("ssp_gram:dense_product_completed")
        return trace

    original_csr = core._csr_weighted_gram

    def csr(*args, **kwargs):
        calls["csr_weighted_gram"] += 1
        return original_csr(*args, **kwargs)

    previous_trace = sys.gettrace()
    core._csr_weighted_gram = csr
    sys.settrace(trace)
    try:
        yield record
    finally:
        sys.settrace(previous_trace)
        core._csr_weighted_gram = original_csr


def main():
    if "--measure-time" in sys.argv:
        raise SystemExit("Dispatch observation must run separately from measured fits")
    original = benchmark._kernel_dispatch

    @contextmanager
    def combined(sampler=None):
        with original(sampler) as record, _observe(sampler) as gram_record:
            record["ssp_gram"] = gram_record
            yield record

    benchmark._kernel_dispatch = combined
    try:
        benchmark.main()
    finally:
        benchmark._kernel_dispatch = original


if __name__ == "__main__":
    main()
