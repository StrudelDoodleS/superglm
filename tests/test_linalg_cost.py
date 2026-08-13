"""The cost instrument records what it claims to record.

An instrument that silently misses calls is worse than no instrument: every
complexity assertion built on it would pass. These tests pin the three ways it
could miss -- an aliased binding, a batched operand read as a single matrix,
and an allocation that never reaches LAPACK -- and each one is shown failing
against the weaker instrument that would miss it.
"""

from __future__ import annotations

import tracemalloc

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


def test_the_growth_assertion_needs_two_sizes_and_a_positive_baseline():
    with pytest.raises(ValueError, match="at least two sizes"):
        assert_grows_linearly((10,), [1], label="work")
    with pytest.raises(ValueError, match="cannot take a ratio"):
        assert_grows_linearly((10, 20), [0, 1], label="work")
