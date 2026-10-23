"""Cold-compile budget: kernels restructured to compile less, with outputs unchanged to the bit.

A first fit on an empty Numba cache spends most of its time compiling.  These
tests hold the restructured kernels to what was measured when they were cut
(counts and code shape, never wall time) and pin their outputs to those of
the kernels they replaced (the master of 2026-10-19, f44b94c6), on
deterministic inputs that reach every edge route: zeros of both signs,
subnormals, products either side of ``2^-969``, overflow to the scaled
fallback, non-finite rows and exact cancellation.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
import os
import subprocess
import sys
import textwrap
from fractions import Fraction

import numpy as np
import pytest


def _stream(seed):
    """xorshift64*: inputs that do not depend on NumPy's generators."""
    state = seed & 0xFFFFFFFFFFFFFFFF
    while True:
        state ^= state >> 12
        state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 27
        yield (state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF


def _floats(seed, n, low, high, zeros=0.1):
    """Signed floats with exponents in ``[low, high)`` and a share of signed zeros."""
    out = []
    bits = _stream(seed)
    for _ in range(n):
        r = next(bits)
        if (r & 0xFF) < zeros * 256:
            out.append(-0.0 if r & 0x100 else 0.0)
            continue
        mantissa = 0.5 + ((r >> 11) & ((1 << 52) - 1)) / 2.0**53
        exponent = low + (r >> 8) % (high - low)
        out.append(math.copysign(math.ldexp(mantissa, exponent), -1.0 if r & 0x200 else 1.0))
    return np.array(out, dtype=np.float64)


def _sum_cases():
    cases = [
        ([1.0, 1.0, 1.0], [1e150, -1e150, 1e-200]),
        ([1.0, 1.0], [math.inf, 1.0]),
        ([1.0, math.nan], [1.0, 1.0]),
        ([5e-324, 5e-324, -0.0], [5e-324, -0.0, 1.0]),
        ([1e308, 1e308], [1e308, 1.0]),
        ([2.0**-969, -(2.0**-969), 2.0**-968], [1.0, 1.0, 1.0]),
        ([2.0**-500, 2.0**-469], [2.0**-470, 2.0**-500]),
        ([0.0, -0.0], [0.0, -0.0]),
        ([], []),
        ([0.1] * 200, [math.ldexp(1.0, e) for e in range(-1000, 1000, 10)]),
        ([3.0, 3.0, 3.0], [1.0 + 2.0**-52, 1.0, 1.0 - 2.0**-53]),
    ]
    for seed, n, (low, high) in (
        (1, 50, (-60, 60)),
        (2, 400, (-1074, 1024)),
        (3, 300, (-1074, -900)),
        (4, 300, (900, 1024)),
        (5, 2000, (-30, 30)),
    ):
        values = _floats(seed + 100, n, -1074 if low < -900 else low, high)
        cases.append((np.abs(_floats(seed, n, low, high)), values))
        cases.append((_floats(seed + 200, n, low, high), values))
    return [(np.array(w, dtype=np.float64), np.array(v, dtype=np.float64)) for w, v in cases]


def _formatted(result):
    return f"{float(result[0]).hex()},{int(result[1])},{bool(result[2])};".encode()


# The digests of the replaced kernels' outputs (master f44b94c6) on these inputs.
_EXACT_SUMS_DIGEST = "3e6461c7704bf1d4ee485612b7a3eb1b24a262159676eb8e39b7683497883138"
_DOT2_DIGEST = "af88a8e3cc2c44be911a60e974e66a0c4f6d9644b329bdd7b1b5802ed875abc5"


def test_exact_sums_match_the_replaced_kernels_to_the_bit():
    """``unscaled_exact_sum``, ``scaled_exact_sum`` and ``weighted_residual_sum``, every mode."""
    from superglm.solvers import _exact_sums as sums

    digest = hashlib.sha256()
    count = 0
    for w, v in _sum_cases():
        weights, values = sums.native_operand(w), sums.native_operand(v)
        for mode in (0, 1, 2):
            for shift in (0.0, 0.37, -1e300, 1e-310):
                for kernel in (sums.unscaled_exact_sum, sums.scaled_exact_sum):
                    digest.update(_formatted(kernel(weights, values, shift, mode)))
                    count += 1
        offset = sums.native_operand(v[::-1].copy())
        contribution = sums.native_operand(np.roll(v, 1))
        for has_offset in (False, True):
            for alpha in (0.0, 0.37, -1e300):
                digest.update(
                    _formatted(
                        sums.weighted_residual_sum(
                            weights, values, offset, has_offset, alpha, contribution
                        )
                    )
                )
                count += 1
    assert count == 630
    assert digest.hexdigest() == _EXACT_SUMS_DIGEST


def test_unscaled_exact_sum_stays_correctly_rounded():
    """Away from the subnormal merge, ``S`` is the exact sum rounded once, every mode."""
    from superglm.solvers import _exact_sums as sums

    checked = 0
    for w, v in _sum_cases():
        weights, values = sums.native_operand(w), sums.native_operand(v)
        for mode, shift in ((0, 0.0), (1, 0.0), (2, 0.37)):
            total, exponent, ok = sums.unscaled_exact_sum(weights, values, shift, mode)
            if not ok or exponent != 0 or not all(map(math.isfinite, (*w, *v))):
                continue
            exact_values = [Fraction(x) for x in w]
            if mode:
                exact_values = [
                    Fraction(a) * (Fraction(b) - Fraction(shift)) for a, b in zip(w, v, strict=True)
                ]
            if any(x and abs(x) < Fraction(2) ** -968 for x in exact_values):
                continue
            assert total == float(sum(exact_values, Fraction(0)))
            checked += 1
    assert checked >= 20


def test_dot2_kernels_match_the_replaced_kernels_to_the_bit():
    """``_dot2_value``, ``_dot2_selected`` (now given ``right`` transposed) and the quadratic form."""
    from superglm.reml import _compensated as compensated

    native = compensated._native_operand
    digest = hashlib.sha256()
    vectors = [
        ([1e16, 1.0, -1e16], [1.0, 1.0, 1.0]),
        ([1 + 2**-27, -1.0], [1 - 2**-27, 1.0]),
        ([5e-324, 1.0], [1.0, 1.0]),
        ([1e300, 1e300], [1e10, -1e10]),
        ([-0.0], [1.0]),
        ([0.0, -0.0], [-0.0, -0.0]),
        ([2.0**-600], [2.0**-500]),
    ]
    for seed in range(6):
        vectors.append(
            (_floats(10 + seed, 300, -40, 40, 0.05), _floats(20 + seed, 300, -40, 40, 0.05))
        )
    for x, y in vectors:
        value, ok = compensated._dot2_value(native(np.array(x, float)), native(np.array(y, float)))
        digest.update(f"{float(value).hex()},{bool(ok)};".encode())
    for seed in range(4):
        left = _floats(30 + seed, 7 * 40, -20, 20, 0.05).reshape(7, 40)
        right = _floats(40 + seed, 40 * 5, -20, 20, 0.05).reshape(40, 5)
        if seed == 3:
            left[2, 3] = 5e-324
        indices = np.array([[r, c] for r in range(7) for c in range(5)], dtype=np.int64)
        values, success = compensated._dot2_selected(native(left), native(right.T), indices)
        digest.update(np.asarray(values).tobytes() + np.asarray(success).tobytes())
    for seed in range(4):
        dense = _floats(50 + seed, 30 * 30, -10, 10, 0.6).reshape(30, 30)
        dense[5, :] = 0.0
        dense[7, :] = 0.0
        dense[7, 3] = 2.5
        if seed == 2:
            dense[9, 9] = 5e-324
        vector = native(_floats(60 + seed, 30, -10, 10, 0.1))
        value, ok = compensated._dot2_quadratic_form(*compensated._nonzero_entries(dense), vector)
        digest.update(f"{float(value).hex()},{bool(ok)};".encode())
    assert digest.hexdigest() == _DOT2_DIGEST


def _inline_expansions(module, kernel: str) -> dict[str, int]:
    """How many times Numba's IR inliner copies each ``inline="always"`` helper into ``kernel``.

    Every call site of an inlined helper is a full copy, and a copy of a
    helper that itself inlines carries that helper's copies too: the count
    is the code volume type inference and lowering work through.
    """
    tree = ast.parse(inspect.getsource(module))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def inlined(node) -> bool:
        return any(
            isinstance(decorator, ast.Call)
            and any(
                keyword.arg == "inline" and getattr(keyword.value, "value", None) == "always"
                for keyword in decorator.keywords
            )
            for decorator in node.decorator_list
        )

    def expand(name: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                callee = functions.get(node.func.id)
                if callee is not None and inlined(callee):
                    counts[callee.name] = counts.get(callee.name, 0) + 1
                    for nested, times in expand(callee.name).items():
                        counts[nested] = counts.get(nested, 0) + times
        return counts

    return expand(kernel)


def test_exact_sums_inline_the_expansion_step_once_per_set():
    """The kernels compiled 2.4 s cold with ten copies of the step; two compile in 1.1 s."""
    from superglm.reml import _compensated as compensated
    from superglm.solvers import _exact_sums as sums

    for kernel in ("unscaled_exact_sum", "weighted_residual_sum"):
        assert _inline_expansions(sums, kernel).get("_add_partial") == 2, kernel
    assert _inline_expansions(sums, "_merge").get("_add_partial") == 1
    assert _inline_expansions(compensated, "_dot2_value").get("_two_product_split") == 1
    assert _inline_expansions(compensated, "_dot2_quadratic_form").get("_two_product_split") == 3


# The lead's probe model: a 20,000-row Tweedie fit with a spline and a categorical.
_FIRST_FIT = """
import json
import numpy as np
import pandas as pd
from numba.core import event

compiled = {}

class Listener(event.Listener):
    def on_start(self, ev):
        function = getattr(ev.data.get("dispatcher"), "py_func", None)
        module = getattr(function, "__module__", "") or ""
        if module.startswith("superglm"):
            name = f"{module}.{function.__qualname__}"
            compiled[name] = compiled.get(name, 0) + 1

    def on_end(self, ev):
        pass

event.register("numba:compile", Listener())
import superglm

rng = np.random.default_rng(1)
n = 20000
X = pd.DataFrame({"a": rng.uniform(0, 1, n), "g": rng.choice(list("pqrs"), n)})
mu = np.exp(0.5 + np.sin(3 * X.a))
y = rng.gamma(1.5, mu / 1.5) * (rng.uniform(size=n) < 0.6)
model = superglm.SuperGLM(
    family=superglm.families.tweedie(p=1.5),
    features={"a": superglm.Spline(n_knots=8), "g": superglm.Categorical()},
)
model.fit_reml(X, y)
model.summary()
print(json.dumps(compiled))
"""


def _cold_run(script: str, tmp_path) -> dict:
    import json

    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba-cache"))
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_first_tweedie_fit_compiles_each_kernel_once(tmp_path):
    """On an empty cache, a first SuperGLM Tweedie fit and summary compile 14 superglm
    specialisations of 14 kernels: no kernel twice (master compiled 18, with ``_merge``,
    its two helpers and ``_csr_weighted_gram`` each compiled for a second argument type).

    The ``n != 1`` check is the regression guard. The cap of 14 is a budget: when
    a change puts a new kernel on this fit's path on purpose, raise it in the same
    change and say why.
    """
    compiled = _cold_run(_FIRST_FIT, tmp_path)
    assert compiled, "the compile listener saw no superglm kernel: nothing was measured"
    assert not {name: n for name, n in compiled.items() if n != 1}
    assert sum(compiled.values()) <= 14, compiled


@pytest.mark.slow
def test_tweedie_lss_kernels_compile_one_specialisation_each(tmp_path):
    """Status codes and seeds that reach a compiled helper are int64, not literals.

    Master compiled ``_row_failure`` and ``_series_failure`` once per status
    literal (16 specialisations), ``_log_adjacent_ratio`` three times and the
    parallel core's row kernel again for prange's unsigned index; the row
    kernel now runs serially on the worker pool, one signature above and below
    the pooled threshold.
    """
    script = """
    import json
    import numpy as np
    from superglm.distributional.kernels import _tweedie_numba as compiled
    from superglm.distributional.kernels.tweedie import evaluate_tweedie_rows

    n = 6000
    rng = np.random.default_rng(3)
    y = rng.gamma(1.5, 1.0, n) * (rng.uniform(size=n) < 0.6)
    ones = np.ones(n)
    evaluate_tweedie_rows(y, ones * 1.2, ones * 0.9, ones * 1.5, ones, "prior", derivative_order=2)
    # Below the pooled threshold the rows run as one block: the same kernel,
    # so a second signature would show here.
    head = slice(0, 100)
    evaluate_tweedie_rows(
        y[head], ones[head] * 1.2, ones[head] * 0.9, ones[head] * 1.5, ones[head], "prior",
        derivative_order=2,
    )
    print(json.dumps({
        name: len(value.signatures)
        for name, value in vars(compiled).items()
        if hasattr(value, "signatures") and value.signatures
    }))
    """
    signatures = _cold_run(script, tmp_path)
    assert signatures["_evaluate_tweedie_batch_rows"] == 1
    assert not {name: n for name, n in signatures.items() if n != 1}
