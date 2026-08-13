"""Count the linear algebra instead of timing it.

A wall clock cannot answer a complexity question on a shared machine.  Several
test sessions routinely run here at once, so the one-minute load average sits
well above the core count, and an A/B timing ratio for the same code has been
observed to span a factor of six.  Pinning thread pools does not rescue this:
it removes fan-out, not cache and memory-bandwidth contention.

Counting answers the same question exactly.  For a structured algorithm the
number of factorizations and the shapes handed to LAPACK *are* the complexity
-- a cost claim written in those terms is arithmetic over a call log rather
than a measurement, so it is reproducible on a busy machine and cheap enough
to run in CI, where a benchmark never could.  The specific risk it catches is
an accidental dense temporary: a path that is meant to touch ``L`` small
blocks, but assembles one ``L*k x L*k`` matrix instead, keeps its call count
and changes its operand shapes.

``record_linalg_calls`` rebinds a fixed registry of ``numpy.linalg`` and
``scipy.linalg`` entry points for the duration of a block, records each call
with its operand shapes and dtypes, and reports the ``tracemalloc`` peak.

The pattern is Django's ``assertNumQueries``, which has guarded against the
N+1 query for years by asserting an exact count of the expensive operation
rather than how long it took.  This is the linear-algebra transliteration.

Interception is by rebinding a fixed registry of *public* entry points.  Two
documented alternatives were considered and rejected:

* NumPy's ``__array_function__`` protocol (NEP 18) makes every public
  ``numpy.linalg`` function overridable, but by design it consults an override
  only when an argument is *not* a plain ``ndarray``.  It never fires for
  ordinary internal calls, and using it would mean threading an array subclass
  through the code under test, which changes what is being measured.  NEP 13
  (``__array_ufunc__``) excludes linear algebra; NEP 31 (uarray backends) was
  superseded and never adopted, and ``scipy.linalg`` has no dispatch mechanism
  at all.
* ``sys.monitoring`` (PEP 669) is binding-independent and does reach compiled
  entry points.  Its ``CALL`` event reports only ``arg0``, though, so it
  cannot see the second operand -- and a claim about a triangular solve is a
  claim about both its shapes.  It would also key on SciPy private names that
  move between releases, where the registry here names only public API.

Rebinding must therefore also cover module-level ``from scipy.linalg import
cho_factor`` aliases, which hold their own reference and are untouched by
patching the defining module -- the "where to patch" problem described in the
``unittest.mock`` documentation.  Aliases are rebound across ``packages``
(``superglm`` by default) by scanning module ``__dict__``s, which does not
trigger the lazy submodule loading that attribute access on SciPy would.

Three limits are worth stating rather than hiding.

Matrix multiplication is invisible.  ``A @ B`` is a bytecode operator, not a
registered call, so a dense temporary built that way never appears in the log
-- and no interception mechanism catches it, including ``sys.monitoring``.
This is exactly why ``peak_bytes`` is not optional: allocation is the
implementation-independent backstop that sees a dense temporary however it was
built.  Counts say which call did it; the peak says that it happened.

Calls below the registry are not individually recorded -- a raw LAPACK wrapper
obtained from ``scipy.linalg.get_lapack_funcs``, or a call from inside compiled
code.  ``get_lapack_funcs`` is registered so that *that* route is visible, but
it is not the only one: reaching straight into ``scipy.linalg.lapack`` bypasses
the registry with nothing in the log at all, and ``superglm`` does this once
today, in the rank solver.  The registry is therefore checked against the
routines this package actually uses, rather than trusted to cover them --
see ``test_the_registry_covers_every_routine_superglm_calls``.

The recorder is not thread-safe, so the recorded block must be single-threaded
at the Python level.  BLAS may still thread internally; that does not affect
counts, which is the point.
"""

from __future__ import annotations

import contextlib
import math
import sys
import tracemalloc
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy.linalg
import scipy.linalg

#: ``numpy.linalg`` entry points that are rebound while recording.
NUMPY_ROUTINES = (
    "cholesky",
    "cond",
    "det",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "inv",
    "lstsq",
    "matrix_power",
    "matrix_rank",
    "multi_dot",
    "norm",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "svd",
    "tensorinv",
    "tensorsolve",
)

#: ``scipy.linalg`` entry points that are rebound while recording.
SCIPY_ROUTINES = (
    "block_diag",
    "cho_factor",
    "cho_solve",
    "cho_solve_banded",
    "cholesky",
    "cholesky_banded",
    "det",
    "eig",
    "eigh",
    "eigh_tridiagonal",
    "eigvals",
    "eigvalsh",
    "eigvalsh_tridiagonal",
    "get_blas_funcs",
    "get_lapack_funcs",
    "inv",
    "ldl",
    "lstsq",
    "lu",
    "lu_factor",
    "lu_solve",
    "norm",
    "null_space",
    "orth",
    "pinv",
    "pinvh",
    "qr",
    "qr_multiply",
    "qz",
    "rq",
    "schur",
    "solve",
    "solve_banded",
    "solve_triangular",
    "solve_toeplitz",
    "solveh_banded",
    "subspace_angles",
    "svd",
    "svdvals",
)

#: Routines whose cost is at least quadratic in an operand dimension *and*
#: whose leading axes are a batch.  These are the calls a complexity claim is
#: made about; the cheap remainder -- ``norm``, ``block_diag`` and the
#: ``get_*_funcs`` lookups -- is still recorded, but counting it alongside a
#: Cholesky would compare unlike work.
#:
#: Three expensive routines are deliberately outside the set, all for the same
#: reason: their first operand's leading axes are not a batch, so the work
#: metric would sum a number that is wrong.  ``tensorsolve`` and ``tensorinv``
#: reshape, and would misread a ``(2, 3, 4, 24)`` operand as six
#: factorizations of a ``4 x 24`` when it is one of a ``24 x 24``;
#: ``multi_dot`` takes a *sequence* of matrices, so it has no batch axis at
#: all.  All three stay in the registry, so a path that starts using them is
#: visible in the log -- at which point this split needs revisiting rather
#: than trusting.  Everything else at least quadratic is in, including ``det``
#: beside ``slogdet``, which is the same factorization at the same cost.
FACTORIZATIONS = frozenset(
    f"{module}.{name}"
    for module, names in (
        (
            "numpy.linalg",
            (
                "cholesky",
                "cond",
                "eig",
                "eigh",
                "eigvals",
                "eigvalsh",
                "inv",
                "lstsq",
                "det",
                "matrix_power",
                "matrix_rank",
                "pinv",
                "qr",
                "slogdet",
                "solve",
                "svd",
            ),
        ),
        (
            "scipy.linalg",
            (
                "cho_factor",
                "cho_solve",
                "cho_solve_banded",
                "cholesky",
                "cholesky_banded",
                "eig",
                "eigh",
                "eigh_tridiagonal",
                "eigvals",
                "eigvalsh",
                "eigvalsh_tridiagonal",
                "inv",
                "ldl",
                "lstsq",
                "lu",
                "lu_factor",
                "lu_solve",
                "null_space",
                "orth",
                "pinv",
                "pinvh",
                "qr",
                "qr_multiply",
                "qz",
                "rq",
                "schur",
                "solve",
                "solve_banded",
                "solve_triangular",
                "solve_toeplitz",
                "solveh_banded",
                "subspace_angles",
                "svd",
                "svdvals",
            ),
        ),
    )
    for name in names
)


@dataclass(frozen=True)
class Operand:
    """Shape and dtype of one array argument, as passed.

    The split between *core shape* and *batch* is the one NumPy's linear
    algebra already makes: these routines operate on the last two axes and
    broadcast over everything in front, so one call on an ``(L, g, g)`` stack
    is ``L`` independent ``g x g`` factorizations.  Cost is
    ``batch * f(core_shape)``, and the two halves answer different questions
    -- a batch that tracks ``L`` is the algorithm working level by level,
    while a *core shape* that tracks ``L`` is the dense temporary.
    """

    shape: tuple[int, ...]
    dtype: str

    @property
    def core_shape(self) -> tuple[int, ...]:
        """The matrix LAPACK actually sees: the last two axes."""
        return self.shape[-2:]

    @property
    def batch(self) -> int:
        """How many independent matrices this operand stacks."""
        stacked = 1
        for dim in self.shape[:-2]:
            stacked *= dim
        return stacked


@dataclass(frozen=True)
class LinalgCall:
    """One recorded call: the routine, its array operands, and what it returned.

    Outputs are recorded because a routine can build something far larger than
    anything it was given -- ``block_diag`` turns ``L`` small blocks into one
    ``(L g, L g)`` array, and every *operand* stays small while it does.  They
    are kept out of :meth:`CostRecord.core_signature` deliberately: a batched
    ``eigh`` returns eigenvalues shaped ``(L, g)``, whose last two axes are
    genuinely size-dependent without anything being wrong, so folding outputs
    into the shape invariant would make it fail on correct code.
    :meth:`CostRecord.max_elements` is the assertion that uses them.
    """

    name: str
    operands: tuple[Operand, ...]
    outputs: tuple[Operand, ...] = ()

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        """Operand shapes in call order."""
        return tuple(operand.shape for operand in self.operands)

    @property
    def core_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Operand core shapes in call order."""
        return tuple(operand.core_shape for operand in self.operands)

    @property
    def dtypes(self) -> tuple[str, ...]:
        """Operand dtypes in call order."""
        return tuple(operand.dtype for operand in self.operands)

    @property
    def batch(self) -> int:
        """Independent factorizations this call performs.

        Read off the first operand, which is the matrix in every LAPACK-shaped
        signature here.  One for an unbatched call, and for a call with no
        array operands at all.
        """
        return self.operands[0].batch if self.operands else 1

    @property
    def max_core_dim(self) -> int:
        """Largest core dimension over all operands, 0 when there are none."""
        return max((max(shape, default=0) for shape in self.core_shapes), default=0)


@dataclass
class CostRecord:
    """The call log for one recorded block.

    ``calls`` fills in as the block runs; ``peak_bytes`` is only meaningful
    after the block exits.
    """

    calls: list[LinalgCall] = field(default_factory=list)
    peak_bytes: int = 0

    def counts(self) -> Counter[str]:
        """How many times each routine was called."""
        return Counter(call.name for call in self.calls)

    def of(self, *names: str) -> tuple[LinalgCall, ...]:
        """Recorded calls to *names*, in call order."""
        wanted = frozenset(names)
        return tuple(call for call in self.calls if call.name in wanted)

    def factorizations(self) -> tuple[LinalgCall, ...]:
        """Recorded calls whose cost is at least quadratic (see FACTORIZATIONS)."""
        return tuple(call for call in self.calls if call.name in FACTORIZATIONS)

    def elementary_factorizations(self) -> int:
        """Independent matrix factorizations performed, batches unrolled.

        The work metric a complexity claim is made in: unlike the number of
        Python-level calls it does not change when a loop is replaced by one
        batched call, and unlike a duration it does not change when the
        machine is busy.
        """
        return sum(call.batch for call in self.factorizations())

    def core_signature(self) -> frozenset[tuple[str, tuple[tuple[int, ...], ...]]]:
        """The distinct ``(routine, core shapes)`` pairs among factorizations.

        The multiset of calls grows with the problem; this set is what must
        not.  Operands only -- see :class:`LinalgCall` for why outputs are not
        in here.
        """
        return frozenset((call.name, call.core_shapes) for call in self.factorizations())

    def max_core_dim(self) -> int:
        """Largest core dimension handed to any factorization."""
        return max((call.max_core_dim for call in self.factorizations()), default=0)

    def max_elements(self) -> int:
        """Elements in the largest array seen, counting operands and outputs.

        The size-blind companion to :meth:`core_signature`, and the one that
        catches an assembly whose *result* is the dense object: an ``(L, g, g)``
        stack holds ``L g^2`` elements where the ``(L g, L g)`` matrix built
        from the same blocks holds ``L^2 g^2``.  Growth rate separates them
        without needing to know ``g``.
        """
        return max(
            (
                math.prod(operand.shape)
                for call in self.calls
                for operand in (*call.operands, *call.outputs)
            ),
            default=0,
        )


def _operands(value: Any, *, depth: int = 0) -> Iterator[Operand]:
    """Describe *value* as array operands, unpacking one level of sequence.

    One level is enough for the packed-factor convention -- ``cho_solve``
    takes its factor as a ``(c, lower)`` tuple -- and stops short of walking
    arbitrary nested structure.

    Zero-dimensional operands are dropped.  The ``lower`` in that same tuple
    arrives as a NumPy bool scalar, and a flag is not an operand; leaving it
    in would put an empty shape into every signature it touches.
    """
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            dims = tuple(int(dim) for dim in shape)
        except TypeError:  # pragma: no cover - exotic shape attribute
            return
        if dims:
            yield Operand(dims, str(dtype))
    elif depth == 0 and isinstance(value, (tuple, list)):
        for item in value:
            yield from _operands(item, depth=1)


def _describe(args: Sequence[Any], kwargs: Mapping[str, Any]) -> tuple[Operand, ...]:
    """Array operands of one call, positional first then keyword."""
    found: list[Operand] = []
    for value in (*args, *kwargs.values()):
        found.extend(_operands(value))
    return tuple(found)


def _alias_sites(originals: dict[int, str], packages: Sequence[str]) -> list[tuple[Any, str, Any]]:
    """Find ``from scipy.linalg import qr``-style bindings inside *packages*.

    Scans module ``__dict__``s rather than using ``getattr``, so a package
    with lazy submodule loading is not woken up by the scan.
    """
    prefixes = tuple(packages)
    sites: list[tuple[Any, str, Any]] = []
    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue
        if not any(
            module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefixes
        ):
            continue
        try:
            namespace = vars(module)
        except TypeError:  # pragma: no cover - module without __dict__
            continue
        for attr, value in list(namespace.items()):
            if id(value) in originals:
                sites.append((module, attr, value))
    return sites


@contextlib.contextmanager
def record_linalg_calls(
    *,
    packages: Sequence[str] = ("superglm",),
    trace_memory: bool = True,
) -> Iterator[CostRecord]:
    """Record ``numpy.linalg``/``scipy.linalg`` calls made inside the block.

    Deterministic and independent of machine load: the result is a function of
    the code path taken, not of how busy the box is.

    Parameters
    ----------
    packages
        Import-path prefixes whose module-level aliases are rebound as well as
        the defining modules.  Modules imported *after* the block starts are
        not covered, so import the code under test first.
    trace_memory
        Whether to measure peak allocation above the block's entry baseline.
        ``tracemalloc`` sees NumPy data buffers as well as Python objects, so
        it catches a dense temporary that leaves the call counts alone.  NumPy
        registers those buffers under its own allocation domain
        (``numpy.lib.tracemalloc_domain``, NEP 49) rather than domain 0;
        ``get_traced_memory`` aggregates every domain, so no filtering is
        needed here, but a ``DomainFilter`` on domain 0 elsewhere would report
        zero bytes and look green.  An outer tracer, if any, is left running.

    Yields
    ------
    CostRecord
        Populated as the block runs; ``peak_bytes`` is set on exit.
    """
    record = CostRecord()
    targets = [
        (numpy.linalg, "numpy.linalg", NUMPY_ROUTINES),
        (scipy.linalg, "scipy.linalg", SCIPY_ROUTINES),
    ]

    originals: dict[int, str] = {}
    replacements: dict[int, Any] = {}
    reverted: dict[int, Any] = {}
    restore: list[tuple[Any, str, Any]] = []

    for module, module_name, routines in targets:
        for routine in routines:
            original = getattr(module, routine, None)
            if original is None or not callable(original):
                continue
            qualified = f"{module_name}.{routine}"

            def wrapper(*args: Any, _f: Any = original, _n: str = qualified, **kwargs: Any):
                operands = _describe(args, kwargs)
                try:
                    result = _f(*args, **kwargs)
                except BaseException:
                    # A routine that raised still did its work, and superglm
                    # routinely catches LinAlgError and falls back.  Dropping
                    # the failed attempt would undercount the real cost.
                    record.calls.append(LinalgCall(_n, operands))
                    raise
                record.calls.append(LinalgCall(_n, operands, _describe((result,), {})))
                return result

            wrapper.__name__ = routine
            wrapper.__qualname__ = qualified
            wrapper.__doc__ = getattr(original, "__doc__", None)
            originals[id(original)] = qualified
            replacements[id(original)] = wrapper
            reverted[id(wrapper)] = original
            restore.append((module, routine, original))

    started_tracing = False
    baseline = 0
    try:
        # Inside the try: a setattr that fails partway through must still
        # unwind, or the process is left running against live wrappers that
        # append to a record nobody will ever read.
        restore.extend(_alias_sites(originals, packages))
        for module, attr, original in restore:
            setattr(module, attr, replacements[id(original)])

        if trace_memory:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                started_tracing = True
            tracemalloc.reset_peak()
            baseline = tracemalloc.get_traced_memory()[0]
        yield record
    finally:
        if trace_memory:
            record.peak_bytes = max(tracemalloc.get_traced_memory()[1] - baseline, 0)
            if started_tracing:
                tracemalloc.stop()
        for module, attr, original in reversed(restore):
            setattr(module, attr, original)
        # A module first imported *inside* the block bound the wrapper and was
        # not in the entry scan.  Left alone it would hold that wrapper for the
        # rest of the session, appending to a record nobody reads and keeping
        # it alive.  Scan once more, this time for wrappers.
        for module, attr, wrapped in _alias_sites(reverted, packages):
            setattr(module, attr, reverted[id(wrapped)])


def assert_core_shapes_independent(sizes: Sequence[int], records: Sequence[CostRecord]) -> None:
    """Assert the same problem at different *sizes* factors the same shapes.

    The exact half of a complexity claim, and the half that catches a dense
    temporary.  Running a structured path at several sizes multiplies how
    *often* each shape is factored; which shapes it factors is fixed by the
    block geometry and must not move at all.  No tolerance, because there is
    nothing here for a tolerance to absorb: a set of shapes either is the same
    set or is not.
    """
    signatures = [record.core_signature() for record in records]
    empty = [size for size, signature in zip(sizes, signatures, strict=True) if not signature]
    assert not empty, (
        f"no factorizations were recorded at sizes {empty}, so this assertion would "
        "pass on any code at all.  Either the path factors nothing, or the recorder "
        "did not see it -- check that the code under test was imported before the "
        "block and that `packages` covers it."
    )

    shared = signatures[0]
    offenders = [
        (size, sorted(signature ^ shared))
        for size, signature in zip(sizes, signatures, strict=True)
        if signature != shared
    ]
    assert not offenders, (
        "factorization core shapes depend on size; "
        f"at size {sizes[0]} they are {sorted(shared)}, and these differ: {offenders}"
    )


def assert_grows_linearly(
    sizes: Sequence[int],
    values: Sequence[float],
    *,
    label: str,
    tolerance: float = 1.25,
) -> None:
    """Assert *values* grow no faster than *sizes*, up to *tolerance*.

    Each consecutive pair must satisfy
    ``value_ratio <= size_ratio * tolerance``.  Comparing successive ratios
    rather than fitting a curve is the textbook separation of complexity
    classes, and it needs no constant.

    ``tolerance`` is *not* an allowance for a positive affine intercept, which
    is the intuitive reading and the wrong one: a cost of ``a * L + b`` with
    ``b > 0`` doubles to ``(2aL + b) / (aL + b) < 2``, so fixed overhead makes
    the ratio *smaller* and needs nothing.  Two things do push a linear cost
    above the size ratio.  A negative intercept -- a term like ``a(L - 1)``,
    one factorization per level with the first merged away -- doubles to
    ``2 + 1/(L - 1)``, or 2.016 at 64 levels.  And a constant that is itself
    mildly size-dependent: the screening ladder bisects a data-dependent
    number of times, so its factorizations-per-level runs between 113.0 and
    118.0 across the sweep, a 4.4% swing on top of a worst-case ratio of
    2.052.

    Hence 1.25, allowing 2.50 on a doubling: about 22% over what is observed,
    which covers that swing on another BLAS or a later NumPy without becoming
    meaningless.  It still rejects everything from ``O(L^1.33)`` up, including
    the ``O(L*sqrt(L))`` that a half-vectorised loop produces, which doubles
    at 2.83.

    Pass a tighter value on a channel that does not wobble.  ``max_elements``
    doubles at exactly 2.0000 on the screening path, because it is structural
    rather than search-dependent, and takes 1.05.

    It is emphatically not a noise allowance.  Counts reproduce exactly, so a
    failure is a real change in what the code does rather than a busy machine.
    Traced byte totals carry about 1% of run-to-run jitter from incidental
    Python object churn, an order of magnitude inside this bound.
    """
    if len(sizes) < 2:
        raise ValueError("need at least two sizes to compare growth")
    for (small, large), (low, high) in zip(
        zip(sizes, sizes[1:], strict=False),
        zip(values, values[1:], strict=False),
        strict=False,
    ):
        if low <= 0 or small <= 0:
            raise ValueError(f"{label}: cannot take a ratio against {low} at size {small}")
        allowed = (large / small) * tolerance
        observed = high / low
        assert observed <= allowed, (
            f"{label} grows faster than linearly: {small} -> {large} multiplied it by "
            f"{observed:.3f} ({low} -> {high}), above the {allowed:.3f} a linear cost "
            f"allows.  A quadratic cost would show {(large / small) ** 2:.3f}."
        )


def report(record: CostRecord) -> str:
    """One-line-per-routine summary, for pasting into a failure message."""
    lines = [f"peak={record.peak_bytes} bytes, largest array={record.max_elements()} elements"]
    for name, count in sorted(record.counts().items()):
        shapes = sorted({call.shapes for call in record.of(name)})
        lines.append(f"{name} x{count} shapes={shapes}")
    return "\n".join(lines)
