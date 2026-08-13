"""The cost instrument records what it claims to record.

An instrument that silently misses calls is worse than no instrument: every
complexity assertion built on it would pass. These tests pin the three ways it
could miss -- an aliased binding, a batched operand read as a single matrix,
and an allocation that never reaches LAPACK -- and each one is shown failing
against the weaker instrument that would miss it.
"""

from __future__ import annotations

import re
import sys
import tracemalloc
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

# A module-level alias, deliberately: this is the binding shape that a patch of
# the defining module alone does not reach, and superglm has real instances of
# it.  ``packages=("tests",)`` below makes the recorder cover this module.
from scipy.linalg import cho_factor

from ._linalg_cost import (
    FACTORIZATIONS,
    NUMPY_ROUTINES,
    SCIPY_ROUTINES,
    CostRecord,
    LinalgCall,
    Operand,
    assert_core_shapes_independent,
    assert_grows_linearly,
    record_linalg_calls,
    report,
)


def _spd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def test_it_records_the_routine_with_operand_shapes_and_dtypes():
    a = _spd(4)
    b = np.ones((4, 3), dtype=np.float32)

    with record_linalg_calls() as record:
        np.linalg.cholesky(a)
        scipy.linalg.solve_triangular(np.tril(a), b, lower=True)

    assert record.counts() == {
        "numpy.linalg.cholesky": 1,
        "scipy.linalg.solve_triangular": 1,
    }
    (cholesky,) = record.of("numpy.linalg.cholesky")
    assert cholesky.shapes == ((4, 4),)
    assert cholesky.dtypes == ("float64",)
    (triangular,) = record.of("scipy.linalg.solve_triangular")
    assert triangular.shapes == ((4, 4), (4, 3))
    assert triangular.dtypes == ("float64", "float32")


def test_it_records_a_call_made_through_a_module_level_alias():
    """And the recorder that patches only the defining module misses it.

    ``from scipy.linalg import cho_factor`` binds the function object into the
    importing module, so rebinding ``scipy.linalg`` leaves that binding
    pointing at the original.  Both halves are asserted here, because the
    passing half alone would pass just as happily against an instrument that
    never looked at aliases.
    """
    a = _spd(3)

    with record_linalg_calls(packages=()) as unaliased:
        cho_factor(a)
    assert unaliased.calls == [], "a defining-module patch should not see the alias"

    with record_linalg_calls(packages=("tests",)) as aliased:
        cho_factor(a)
    assert aliased.counts() == {"scipy.linalg.cho_factor": 1}

    assert cho_factor is scipy.linalg.cho_factor, "the alias must be put back"


def test_it_separates_the_batch_from_the_core_shape():
    """A stacked call is many factorizations of a small matrix, not one big one.

    Reading the whole shape instead would report a ``(7, 5, 5)`` stack as
    having a dimension of 7, which is the level count wearing the costume of a
    matrix size -- exactly the confusion that makes a dense temporary invisible.
    """
    stack = np.stack([_spd(5, seed) for seed in range(7)])

    with record_linalg_calls() as record:
        np.linalg.eigh(stack)

    (call,) = record.of("numpy.linalg.eigh")
    assert call.shapes == ((7, 5, 5),)
    assert call.core_shapes == ((5, 5),)
    assert call.batch == 7
    assert call.max_core_dim == 5
    assert record.elementary_factorizations() == 7
    assert record.core_signature() == {("numpy.linalg.eigh", ((5, 5),))}


def test_it_unpacks_a_packed_factor_argument():
    """``cho_solve`` takes its factor as a tuple; the operand is inside it."""
    a = _spd(4)
    factored = scipy.linalg.cho_factor(a)

    with record_linalg_calls() as record:
        scipy.linalg.cho_solve(factored, np.ones(4))

    (call,) = record.of("scipy.linalg.cho_solve")
    assert call.shapes == ((4, 4), (4,))


def test_it_reports_peak_allocation_including_numpy_buffers():
    """NumPy registers its data buffers with ``tracemalloc``, so they count.

    Without that a dense temporary would be invisible to the memory half of
    the instrument, since the array *object* is tiny however large its buffer.
    """
    with record_linalg_calls() as record:
        held = np.zeros((1024, 1024))  # 8 MiB
        del held

    assert record.peak_bytes >= 8 * 1024 * 1024


def test_it_restores_every_binding_even_when_the_block_raises():
    before = (np.linalg.eigh, scipy.linalg.qr, scipy.linalg.cho_factor, cho_factor)

    with pytest.raises(ZeroDivisionError):
        with record_linalg_calls(packages=("tests",)):
            np.linalg.eigh(_spd(2))
            raise ZeroDivisionError

    assert (np.linalg.eigh, scipy.linalg.qr, scipy.linalg.cho_factor, cho_factor) == before


def test_it_leaves_an_outer_tracemalloc_session_running():
    """A test that is itself being traced must not be switched off."""
    tracemalloc.start()
    try:
        with record_linalg_calls():
            np.linalg.cholesky(_spd(3))
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()

    with record_linalg_calls():
        np.linalg.cholesky(_spd(3))
    assert not tracemalloc.is_tracing(), "a session we started must be stopped"


def test_it_ignores_a_routine_it_does_not_register():
    """The registry is fixed and stated, so a gap is a known gap."""
    with record_linalg_calls() as record:
        np.dot(np.ones((3, 3)), np.ones((3, 3)))

    assert record.calls == []


def test_every_registered_routine_still_exists_upstream():
    """A rename must fail here rather than quietly stop recording.

    Routines are skipped when absent, so that the instrument keeps working
    across NumPy and SciPy versions.  The cost of that is a silent hole: a
    renamed routine would go unrecorded and every count assertion built on it
    would get weaker without failing.  This closes it.
    """
    missing = [
        f"{module_name}.{routine}"
        for module, module_name, routines in (
            (np.linalg, "numpy.linalg", NUMPY_ROUTINES),
            (scipy.linalg, "scipy.linalg", SCIPY_ROUTINES),
        )
        for routine in routines
        if not callable(getattr(module, routine, None))
    ]
    assert missing == [], f"registered routines that no longer exist: {missing}"

    registered = {f"numpy.linalg.{name}" for name in NUMPY_ROUTINES}
    registered |= {f"scipy.linalg.{name}" for name in SCIPY_ROUTINES}
    assert FACTORIZATIONS <= registered, sorted(FACTORIZATIONS - registered)


def test_the_registry_covers_every_routine_superglm_calls():
    """The registry is a denylist by omission; this ties it to actual usage.

    Guarding renames is not enough — nothing otherwise notices when superglm
    starts calling something unregistered, and an unregistered routine is
    silently uncounted rather than loudly missing.  Scanning the source makes
    coverage exactly as strong as this package needs, instead of as strong as
    upstream's API surface happens to be.
    """
    package = Path(__file__).resolve().parent.parent / "src" / "superglm"
    used: dict[str, set[str]] = {"numpy.linalg": set(), "scipy.linalg": set()}
    patterns = (
        (re.compile(r"\bnp\.linalg\.(\w+)"), "numpy.linalg"),
        (re.compile(r"\bnumpy\.linalg\.(\w+)"), "numpy.linalg"),
        (re.compile(r"\bscipy\.linalg\.(\w+)"), "scipy.linalg"),
        (re.compile(r"\bfrom scipy\.linalg import ([\w, ]+)"), "scipy.linalg"),
    )
    for source in package.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        for pattern, module_name in patterns:
            for match in pattern.findall(text):
                for name in (part.strip().split(" as ")[0] for part in match.split(",")):
                    used[module_name].add(name)

    # Not routines: the exception class, and the submodule that the rank
    # solver reaches through to call a LAPACK driver directly.  That call is
    # genuinely below the registry and no rebinding can see it.
    ignored = {"LinAlgError", "lapack", "blas", "interpolative"}
    registered = {
        "numpy.linalg": set(NUMPY_ROUTINES),
        "scipy.linalg": set(SCIPY_ROUTINES),
    }
    unregistered = {
        module_name: sorted(names - registered[module_name] - ignored)
        for module_name, names in used.items()
    }
    assert not any(unregistered.values()), (
        f"superglm calls routines the recorder would not see: {unregistered}"
    )
    assert used["numpy.linalg"], "the source scan matched nothing, so it proves nothing"


def test_a_routine_whose_leading_axes_are_not_a_batch_is_not_counted_as_work():
    """``tensorsolve`` reshapes, so the batch/core split does not describe it.

    Recorded, because a path reaching for it should be visible; excluded from
    the work metric, because counting it there would be counting a number that
    is wrong.  ``(2, 3, 4, 24)`` is one 24 x 24 solve, not six 4 x 24 ones.
    """
    identity = np.eye(24).reshape(2, 3, 4, 24)
    rhs = np.random.default_rng(0).normal(size=(2, 3, 4))

    with record_linalg_calls() as record:
        np.linalg.tensorsolve(identity, rhs)

    (call,) = record.of("numpy.linalg.tensorsolve")
    assert call.batch == 6, "the misreading this exclusion exists for"
    assert record.factorizations() == ()
    assert record.elementary_factorizations() == 0


def test_the_summary_names_every_routine_with_its_shapes():
    """``report`` only ever runs inside a failure message, so it needs its own.

    A helper that is exercised only when something else has already broken is
    a helper that is never exercised; if it raised, the failure a reader needs
    would be replaced by a traceback from the reporting.
    """
    with record_linalg_calls() as record:
        np.linalg.eigh(np.stack([_spd(4)] * 3))
        scipy.linalg.solve_triangular(np.tril(_spd(4)), np.ones((4, 2)), lower=True)

    summary = report(record)
    assert "numpy.linalg.eigh x1 shapes=[((3, 4, 4),)]" in summary
    assert "scipy.linalg.solve_triangular x1 shapes=[((4, 4), (4, 2))]" in summary
    assert summary.startswith(f"peak={record.peak_bytes} bytes")


def _record_of(*calls: LinalgCall) -> CostRecord:
    return CostRecord(calls=list(calls))


def _eigh(core: int, batch: int) -> LinalgCall:
    shape = (batch, core, core) if batch > 1 else (core, core)
    return LinalgCall("numpy.linalg.eigh", (Operand(shape, "float64"),))


def test_it_records_what_a_routine_returned_not_only_what_it_was_given():
    """``block_diag`` keeps every operand small and builds a large result.

    An assembly like that is invisible to any check that only reads operands,
    which is why the result is recorded and why ``max_elements`` spans both.
    """
    blocks = [np.eye(3) for _ in range(8)]

    with record_linalg_calls() as record:
        scipy.linalg.block_diag(*blocks)

    (call,) = record.of("scipy.linalg.block_diag")
    assert call.shapes == ((3, 3),) * 8
    assert call.outputs[0].shape == (24, 24)
    assert record.max_elements() == 24 * 24


def test_it_records_a_call_that_raised():
    """A routine that raised still did the work, and superglm falls back."""
    with record_linalg_calls() as record:
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(-np.eye(3))

    (call,) = record.of("numpy.linalg.cholesky")
    assert call.shapes == ((3, 3),)
    assert call.outputs == ()


def test_it_puts_back_an_alias_bound_by_a_module_imported_inside_the_block():
    """Otherwise that module holds a live wrapper for the rest of the session.

    The entry scan cannot see a module that does not exist yet, so the exit
    scan looks for wrappers rather than trusting the entry list.
    """
    module = types.ModuleType("superglm._costinst_probe")
    module.cho_factor = scipy.linalg.cho_factor

    with record_linalg_calls():
        sys.modules["superglm._costinst_probe"] = module

    try:
        assert module.cho_factor is scipy.linalg.cho_factor
    finally:
        del sys.modules["superglm._costinst_probe"]


def test_the_shape_assertion_refuses_an_empty_log():
    """Every arm of it is a tautology on nothing, so nothing must not pass."""
    with pytest.raises(AssertionError, match="no factorizations were recorded"):
        assert_core_shapes_independent((10, 20), [CostRecord(), CostRecord()])


def test_the_shape_assertion_rejects_an_operand_that_tracks_the_size():
    sizes = (10, 20)
    structured = [_record_of(_eigh(4, size)) for size in sizes]
    assert_core_shapes_independent(sizes, structured)

    densified = [_record_of(_eigh(4 * size, 1)) for size in sizes]
    with pytest.raises(AssertionError, match="core shapes depend on size"):
        assert_core_shapes_independent(sizes, densified)


def test_the_growth_assertion_separates_linear_from_quadratic():
    sizes = (100, 200, 400)
    assert_grows_linearly(sizes, [110, 210, 410], label="linear work")

    with pytest.raises(AssertionError, match="faster than linearly"):
        assert_grows_linearly(sizes, [100, 400, 1600], label="quadratic work")


def test_the_growth_assertion_rejects_the_exponents_its_default_claims_to():
    """Pin the default tolerance, which otherwise no test constrains.

    Without this, the default could be widened until the assertion admitted
    ``O(L*sqrt(L))`` and every test here would stay green — the loosening
    would be invisible, which is the failure mode this whole file exists to
    prevent.  The exponents come from the docstring: 1.25 allows 2.50 on a
    doubling, so anything from ``L^1.33`` up must fail, ``O(L*sqrt(L))``
    included.
    """
    sizes = (64, 128, 256)

    for exponent in (1.0, 1.1, 1.3):
        assert_grows_linearly(
            sizes, [float(size**exponent) for size in sizes], label=f"L^{exponent}"
        )

    for exponent in (1.33, 1.5, 2.0):
        with pytest.raises(AssertionError, match="faster than linearly"):
            assert_grows_linearly(
                sizes, [float(size**exponent) for size in sizes], label=f"L^{exponent}"
            )

    # And the tight value the largest-array channel is held to.
    assert_grows_linearly(sizes, [float(size) for size in sizes], label="exact", tolerance=1.05)
    with pytest.raises(AssertionError, match="faster than linearly"):
        assert_grows_linearly(
            sizes, [float(size**1.08) for size in sizes], label="L^1.08", tolerance=1.05
        )


def test_the_growth_assertion_needs_two_sizes_and_a_positive_baseline():
    with pytest.raises(ValueError, match="at least two sizes"):
        assert_grows_linearly((10,), [1], label="work")
    with pytest.raises(ValueError, match="cannot take a ratio"):
        assert_grows_linearly((10, 20), [0, 1], label="work")
