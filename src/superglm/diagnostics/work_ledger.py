"""A debug-only ledger of the expensive work a fit does.

    from superglm.diagnostics.work_ledger import work_ledger

    with work_ledger() as ledger:
        model.fit_reml(X, y)
    print(ledger.report())

For the duration of the ``with`` block the ledger replaces a declared list of
entry points (``TARGETS``: dense factorisations, equilibration, rank
decompositions, Gram builds and passes over the rows) with thin wrappers, and
puts every one back on exit, including after an exception.  Nothing in a
production code path changes, so a fit outside the block pays nothing.  A
name imported with ``from ... import`` is replaced where it is used: every
loaded ``superglm`` module is searched for a binding to the original.

Each wrapped call is recorded under ``(operation, caller)``: its count, the
largest input shape, a flop estimate, its own seconds (children subtracted)
and two flags:

* **repeated**: the input is bit-for-bit one that operation has seen before;
* **update candidate**: the input differs from the previous input at the same
  call site by one row or column (a row and its column, for a symmetric
  matrix), or by one bordering row and column.  Such a result could have been
  updated from the previous one for O(n^2) work instead of rebuilt for
  O(n^3) (Gill, Golub, Murray and Saunders 1974; Davis and Hager 2005).

Inputs are hashed and the last input per call site is kept only while the
ledger is on; the copies share a fixed byte budget, and an input too large for
it is hashed but not compared.  Flop counts follow LAPACK Working Note 41 for
the factorisations and are order-of-magnitude constants for the eigensolver and
SVD; a pass over the rows counts as dense, so it overstates a binned pass.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["LedgerRow", "WorkLedger", "work_ledger"]

_MIN_COMPARED_DIM = 4  # smaller inputs differ by "one line" by accident
_DIGESTS_PER_OPERATION = 8192
_INPUT_BYTES = 256 * 2**20  # copies kept for update detection, all call sites together
_HASH_BYTES = 16

# Library calls and passes over the rows count flops and hide the calls nested
# inside them (a library routine's own internals are one operation, a pass
# over the rows is one pass); a composite adds none and shows its children.
LIBRARY, PASS, COMPOSITE = "library", "pass", "composite"


def _dims(first) -> tuple[int, int, int]:
    """``(batch, rows, columns)`` of an input; a vector is one column."""
    if first is None:
        return 1, 0, 0
    shape = first.shape
    if len(shape) == 0:
        return 1, 1, 1
    if len(shape) == 1:
        return 1, shape[0], 1
    return int(np.prod(shape[:-2], dtype=float)), shape[-2], shape[-1]


def _cholesky(first, args, kwargs, result):
    batch, _, n = _dims(first)
    return batch * n**3 / 3.0


def _lu(first, args, kwargs, result):
    batch, m, n = _dims(first)
    lo, hi = min(m, n), max(m, n)
    return batch * (hi * lo**2 - lo**3 / 3.0)


def _qr(first, args, kwargs, result):
    batch, m, n = _dims(first)
    lo, hi = min(m, n), max(m, n)
    mode = kwargs.get("mode", args[3] if len(args) > 3 else None)
    forms_q = mode not in ("r", "raw")
    return batch * (2.0 * hi * lo**2 - 2.0 * lo**3 / 3.0) * (2.0 if forms_q else 1.0)


def _eigh(first, args, kwargs, result):
    _, _, n = _dims(first)
    values_only = bool(kwargs.get("eigvals_only") or kwargs.get("compute_v") is False)
    return _dims(first)[0] * (4.0 / 3.0 if values_only else 9.0) * n**3


def _eigvalsh(first, args, kwargs, result):
    batch, _, n = _dims(first)
    return batch * 4.0 / 3.0 * n**3


def _svd(first, args, kwargs, result):
    batch, m, n = _dims(first)
    lo, hi = min(m, n), max(m, n)
    vectors = kwargs.get("compute_uv", True)
    return batch * (
        4.0 * hi * lo**2 - 4.0 * lo**3 / 3.0 if not vectors else 4.0 * hi * lo**2 + 8.0 * lo**3
    )


def _solve(first, args, kwargs, result):
    batch, _, n = _dims(first)
    rhs = args[1] if len(args) > 1 else kwargs.get("b")
    columns = _dims(rhs)[2] if isinstance(rhs, np.ndarray) else 1
    return batch * (2.0 / 3.0 * n**3 + 2.0 * n**2 * columns)


def _inverse(first, args, kwargs, result):
    batch, _, n = _dims(first)
    return batch * 2.0 * n**3


def _equilibrate(first, args, kwargs, result):
    _, _, n = _dims(first)
    return 2.0 * n**2


def _triangular_inverse(first, args, kwargs, result):
    _, _, n = _dims(first)
    return n**3 / 3.0


def _factor_update(first, args, kwargs, result):
    _, m, n = _dims(first)
    return 4.0 * m * n


def _syrk(first, args, kwargs, result):
    _, rows, columns = _dims(first)
    inner, order = (
        (rows, columns) if kwargs.get("trans", args[5] if len(args) > 5 else 0) else (columns, rows)
    )
    return float(inner) * order * (order + 1)


def _moments_pass(first, args, kwargs, result):
    n = _dims(first)[1]
    p = result.gram.shape[0] if result is not None and hasattr(result, "gram") else 0
    return float(n) * p * (p + 1)


def _gram_pass(first, args, kwargs, result):
    design = kwargs.get("dm")
    channels = len(kwargs["weights"]) if "weights" in kwargs else 1
    if design is None:
        return 0.0
    return float(design.n) * design.p * (design.p + 1) * channels


def _lss_evaluation(first, args, kwargs, result):
    context, coefficients = args[0], args[1]
    return 2.0 * context.response.shape[0] * np.size(coefficients)


def _lss_geometry_at(score_position: int):
    def cost(first, args, kwargs, result):
        layout = args[0]
        score = args[score_position] if len(args) > score_position else kwargs["score_eta"]
        return float(np.shape(score)[0]) * layout.n_coefficients * (layout.n_coefficients + 1)

    return cost


def _signed_channels_pass(first, args, kwargs, result):
    n = _dims(first)[1]
    p = result[0].gram.shape[0] if result else 0
    return float(n) * p * (p + 1) * len(result)


def _column_sums_pass(first, args, kwargs, result):
    return 4.0 * kwargs["dm"].n * len(kwargs["columns"])


def _matvec_pass(first, args, kwargs, result):
    design = args[0]
    return 2.0 * design.n * design.p


def _no_flops(first, args, kwargs, result):
    return 0.0


@dataclass(frozen=True)
class _Target:
    owner: str  # "module" or "module:Class"
    name: str
    cost: Callable
    role: str = LIBRARY
    also: tuple[str, ...] = ()  # compiled twins of a scipy.linalg module, by short name
    input: str | None = None  # keyword whose array is hashed; default: the first array argument

    @property
    def kind(self) -> str:
        module, _, cls = self.owner.partition(":")
        local = f"{cls}.{self.name}" if cls else self.name
        return local if module.startswith("superglm") else f"{module}.{self.name}"


def _library(owner, names_costs, also=()):
    return tuple(_Target(owner, name, cost, LIBRARY, also) for name, cost in names_costs)


_NUMPY = (
    ("cholesky", _cholesky),
    ("qr", _qr),
    ("eigh", _eigh),
    ("eigvalsh", _eigvalsh),
    ("svd", _svd),
    ("solve", _solve),
    ("inv", _inverse),
    ("lstsq", _svd),
    ("pinv", _svd),
    ("matrix_rank", _svd),
    ("slogdet", _lu),
    ("det", _lu),
)
_SCIPY = (
    ("cholesky", _cholesky),
    ("cho_factor", _cholesky),
    ("lu_factor", _lu),
    ("lu", _lu),
    ("qr", _qr),
    ("eigh", _eigh),
    ("eigvalsh", _eigvalsh),
    ("svd", _svd),
    ("svdvals", _svd),
    ("solve", _solve),
    ("inv", _inverse),
    ("lstsq", _svd),
    ("pinv", _svd),
    ("pinvh", _eigh),
    ("orth", _svd),
    ("qr_insert", _factor_update),
    ("qr_delete", _factor_update),
)
_LAPACK = (
    ("dsyequb", _equilibrate),
    ("dpotrf", _cholesky),
    ("dpstrf", _cholesky),
    ("dpotri", _inverse),
    ("dtrtri", _triangular_inverse),
    ("dgetrf", _lu),
    ("dgeqrf", _qr),
    ("dgesdd", _svd),
    ("dgejsv", _svd),
    ("dsyevd", _eigh),
    ("dsyevr", _eigh),
)

_CENTERED = "superglm._group_matrix._group_matrix_centered"
_MODE_SCORE = "superglm.solvers.mode_score"
#: Stages of a fit, with no flops of their own.  Each shows the seconds it spent
#: outside the listed operations, so the table accounts for the whole fit.
PHASES = (
    ("superglm.dm_builder", "build_design_matrix"),
    ("superglm.reml.direct", "optimize_direct_reml"),
    ("superglm.reml.w_derivatives", "reml_w_correction"),
    ("superglm.solvers.irls_direct._fit", "fit_irls_direct"),
    ("superglm.solvers.centered_system", "build_centered_system"),
    ("superglm.solvers.mode_score", "penalized_mode_residual"),
    ("superglm.model.reml_finalize", "finalize_reml_fit"),
    ("superglm.distributional.predictor", "compile_predictors"),
    ("superglm.distributional.fit_state", "prepare_distributional_fit_state"),
    ("superglm.distributional.null_model", "fit_joint_null_model"),
    ("superglm.distributional.smoothing.loop", "fit_distributional_efs"),
    ("superglm.distributional.solver.solver", "_fit_dense_fixed_lambda_core"),
    ("superglm.distributional.solver.solver", "_certified_terminal_curvature"),
    ("superglm.distributional.solver.derivatives", "transform_natural_derivatives"),
)
_LSS_SOLVER = "superglm.distributional.solver.solver"
_LSS_ASSEMBLY = "superglm.distributional.solver.assembly"

#: Entry points the ledger wraps.  A fit that reaches a new expensive routine
#: gets it added here; ``test_work_ledger`` checks that every entry resolves.
TARGETS: tuple[_Target, ...] = (
    *_library("numpy.linalg", _NUMPY),
    *_library("scipy.linalg", _SCIPY),
    *_library("scipy.linalg.lapack", _LAPACK, also=("_flapack",)),
    *_library("scipy.linalg.blas", (("dsyrk", _syrk),), also=("_fblas",)),
    _Target("superglm.solvers.rank", "_leaf_qr", _qr),
    _Target("superglm.solvers.rank", "decompose_gram", _no_flops, COMPOSITE),
    _Target("superglm.solvers.rank", "decompose_gram_if_authoritative", _no_flops, COMPOSITE),
    _Target("superglm.solvers.rank", "decompose_gram_deflated", _no_flops, COMPOSITE),
    _Target("superglm.solvers.rank", "decompose_factor", _no_flops, COMPOSITE),
    _Target(
        "superglm._group_matrix._group_matrix_execution:MatrixExecutionPlan",
        "moments",
        _moments_pass,
        PASS,
    ),
    *(
        _Target(_CENTERED, name, _gram_pass, PASS, input=key)
        for name, key in (
            ("centered_gram_rhs", "W"),
            ("packed_centered_gram_rhs", "W"),
            ("anchor_support_centered_gram_rhs", "W"),
            ("centered_signed_grams", "weights"),
        )
    ),
    _Target(_LSS_SOLVER, "_evaluate_state_unmeasured", _lss_evaluation, PASS),
    _Target(_LSS_ASSEMBLY, "_assemble_dense_geometry_from_matrices", _lss_geometry_at(2), PASS),
    _Target(_LSS_ASSEMBLY, "assemble_grouped_geometry", _lss_geometry_at(1), PASS),
    _Target(
        "superglm._group_matrix._group_matrix_execution:MatrixExecutionPlan",
        "_signed_moments_channels",
        _signed_channels_pass,
        PASS,
    ),
    _Target(_MODE_SCORE, "column_sums", _column_sums_pass, PASS, input="magnitudes"),
    _Target(_MODE_SCORE, "centred_matvec", _matvec_pass, PASS),
    *(_Target(module, name, _no_flops, COMPOSITE) for module, name in PHASES),
    _Target("superglm.group_matrix:DesignMatrix", "matvec", _matvec_pass, PASS),
    _Target("superglm.group_matrix:DesignMatrix", "rmatvec", _matvec_pass, PASS),
)


@dataclass
class LedgerRow:
    """What one operation did at one call site."""

    operation: str
    caller: str
    via: str
    count: int = 0
    repeated: int = 0
    update_candidates: int = 0
    seconds: float = 0.0
    flops: float = 0.0
    largest_shape: tuple[int, ...] = ()
    untracked: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "caller": self.caller,
            "via": self.via,
            "count": self.count,
            "repeated": self.repeated,
            "update_candidates": self.update_candidates,
            "seconds": self.seconds,
            "flops": self.flops,
            "largest_shape": list(self.largest_shape),
            "untracked": self.untracked,
        }


@dataclass
class _Frame:
    child_seconds: float = 0.0


@dataclass
class WorkLedger:
    """The record of one ``work_ledger()`` block."""

    elapsed: float = 0.0
    bookkeeping: float = 0.0  # seconds spent hashing and recording, not fitting
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # ledger faults; the fit ran regardless
    _rows: dict[tuple[str, str], LedgerRow] = field(default_factory=dict, repr=False)
    _seen: dict[str, dict[bytes, None]] = field(default_factory=dict, repr=False)
    _last: dict[tuple[str, str], np.ndarray] = field(default_factory=dict, repr=False)
    _last_bytes: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def rows(self) -> list[LedgerRow]:
        """Rows ranked by the seconds each took on its own, then by estimated flops."""
        return sorted(self._rows.values(), key=lambda r: (-r.seconds, -r.flops))

    def total_flops(self) -> float:
        return sum(row.flops for row in self._rows.values())

    def find(self, operation: str) -> list[LedgerRow]:
        """Rows for one operation, e.g. ``"numpy.linalg.solve"``."""
        return [row for row in self.rows() if row.operation == operation]

    def count(self, operation: str) -> int:
        return sum(row.count for row in self.find(operation))

    def flagged(self) -> list[tuple[float, LedgerRow]]:
        """Rows with a repeat or update flag, with the seconds those calls took (an estimate).

        The estimate takes every call of a row to cost the same.
        """
        found = [
            (row.seconds * (row.repeated + row.update_candidates) / row.count, row)
            for row in self._rows.values()
            if row.repeated + row.update_candidates
        ]
        return sorted(found, key=lambda pair: -pair[0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed,
            "total_flops": self.total_flops(),
            "rows": [row.as_dict() for row in self.rows()],
        }

    def report(self, top: int = 25) -> str:
        """A table of the most expensive operations.

        ``repeat`` counts calls whose input an earlier call of the same
        operation already had; ``update`` counts calls whose input differs from
        the previous call at the same place by one row or column, where
        updating the earlier result would cost far less than redoing it.
        Seconds are summed over threads, so a share can pass 100%.  A row with
        no flops is a stage of the fit: its seconds leave out the listed
        operations called inside it.
        """
        total_flops = self.total_flops() or 1.0
        elapsed = (self.elapsed - self.bookkeeping) or 1.0
        lines = [
            f"{'operation':<34}{'called from':<46}{'count':>7}{'repeat':>8}{'update':>8}"
            f"{'largest input':>16}{'Gflop':>10}{'flop %':>8}{'time %':>8}"
        ]
        for row in self.rows()[:top]:
            shape = "x".join(str(d) for d in row.largest_shape)
            lines.append(
                f"{row.operation[:33]:<34}{row.caller[:45]:<46}{row.count:>7}{row.repeated:>8}"
                f"{row.update_candidates:>8}{shape:>16}{row.flops / 1e9:>10.2f}"
                f"{100 * row.flops / total_flops:>8.1f}{100 * row.seconds / elapsed:>8.1f}"
            )
        flagged = self.flagged()[:top]
        if flagged:
            lines += [
                "",
                "flagged calls, by estimated seconds they cost (repeat + update share of the row)",
            ]
            for seconds, row in flagged:
                lines.append(
                    f"{row.operation[:33]:<34}{row.caller[:45]:<46}{row.count:>7}{row.repeated:>8}"
                    f"{row.update_candidates:>8}{seconds:>10.3f} s{100 * seconds / elapsed:>7.1f}%"
                )
        listed = sum(row.seconds for row in self._rows.values())
        lines.append(
            f"fit {elapsed:.2f} s (ledger bookkeeping {self.bookkeeping:.2f} s not counted); "
            f"listed operations {listed:.2f} s; estimated {total_flops / 1e9:.1f} Gflop"
        )
        return "\n".join(lines)

    # -- recording ---------------------------------------------------------

    def _record(self, target, prepared, result, seconds):
        first, cost_input, tag, caller, via = prepared
        shape = tuple(first.shape) if first is not None else ()
        flops = target.cost(first, *cost_input, result)
        digest, snapshot, oversize = tag
        site = (target.kind, caller)
        with self._lock:
            row = self._rows.get(site)
            if row is None:
                row = self._rows[site] = LedgerRow(target.kind, caller, via)
            row.count += 1
            row.seconds += seconds
            row.flops += flops
            if np.prod(shape, dtype=float) >= np.prod(row.largest_shape, dtype=float):
                row.largest_shape = shape
            if digest is None:
                return
            seen = self._seen.setdefault(target.kind, {})
            previous = self._last.get(site)
            if digest in seen:
                row.repeated += 1
            elif previous is not None and snapshot is not None:
                row.update_candidates += _differs_by_one_line(previous, snapshot)
            row.untracked += oversize
            self._keep(site, snapshot)
            seen[digest] = None
            if len(seen) > _DIGESTS_PER_OPERATION:
                del seen[next(iter(seen))]

    def _keep(self, site, snapshot) -> None:
        old = self._last.pop(site, None)
        if old is not None:
            self._last_bytes -= old.nbytes
        if snapshot is None:
            return
        while self._last and self._last_bytes + snapshot.nbytes > _INPUT_BYTES:
            oldest = next(iter(self._last))
            self._last_bytes -= self._last.pop(oldest).nbytes
        self._last[site] = snapshot
        self._last_bytes += snapshot.nbytes


def _on_one_cross(rows: np.ndarray, cols: np.ndarray) -> bool:
    """Whether every changed entry lies in one row, one column or one row plus its column."""
    for line in (rows[0], cols[0]):
        if np.all((rows == line) | (cols == line)):
            return True
    return False


def _first_mismatch(big: np.ndarray, small: np.ndarray, axis: int) -> int:
    """First index along ``axis`` where ``big`` and ``small`` stop agreeing."""
    limit = small.shape[axis]
    window = big[:limit] if axis == 0 else big[:, :limit]
    unequal = window != small
    lines = unequal.any(axis=1 - axis)
    return int(np.argmax(lines)) if lines.any() else limit


def _differs_by_one_line(previous: np.ndarray, current: np.ndarray) -> bool:
    """Whether two matrices differ by one row/column, or by one bordering row and column."""
    if (
        previous.ndim != 2
        or current.ndim != 2
        or min(*previous.shape, *current.shape) < _MIN_COMPARED_DIM
    ):
        return False
    d_rows = current.shape[0] - previous.shape[0]
    d_cols = current.shape[1] - previous.shape[1]
    if (d_rows, d_cols) == (0, 0):
        rows, cols = np.nonzero(previous != current)
        return rows.size > 0 and _on_one_cross(rows, cols)
    if abs(d_rows) > 1 or abs(d_cols) > 1 or d_rows * d_cols < 0:
        return False
    big, small = (current, previous) if d_rows + d_cols > 0 else (previous, current)
    bordered = [axis for axis, d in enumerate((d_rows, d_cols)) if d]
    if len(bordered) == 2:
        diagonal = min(small.shape)
        mismatch = np.flatnonzero(np.diag(big)[:diagonal] != np.diag(small)[:diagonal])
        candidates = [int(mismatch[0])] if mismatch.size else []
        candidates.append(diagonal)
        return any(
            np.array_equal(np.delete(np.delete(big, i, axis=0), i, axis=1), small)
            for i in candidates
            if i < min(big.shape)
        )
    axis = bordered[0]
    index = _first_mismatch(big, small, axis)
    return np.array_equal(np.delete(big, index, axis=axis), small)


def _fingerprint(arrays, tokens, first) -> tuple[bytes | None, np.ndarray | None, int]:
    """Digest of every input, plus a private copy of ``first`` for update detection."""
    if not arrays:
        return None, None, 0
    hasher = hashlib.blake2b(digest_size=_HASH_BYTES)
    hasher.update("|".join(tokens).encode())
    for array in arrays:
        data = np.ascontiguousarray(array)
        hasher.update(f"{data.shape}{data.dtype.str}".encode())
        hasher.update(data.reshape(-1).view(np.uint8))
    if first is None or first.ndim != 2:
        return hasher.digest(), None, 0
    if first.nbytes > _INPUT_BYTES // 4:
        return hasher.digest(), None, 1
    return hasher.digest(), first.copy(), 0


_THIS_FILE = __file__
_LIBRARY_MARKERS = (f"{'site-packages'}/numpy", f"{'site-packages'}/scipy")


def _callers() -> tuple[str, str]:
    """The nearest two ``superglm`` (else nearest non-library) frames above the wrapper."""
    frame = sys._getframe(2)
    found: list[str] = []
    fallback = None
    while frame is not None and len(found) < 2:
        filename = frame.f_code.co_filename
        module = frame.f_globals.get("__name__", "")
        label = f"{module.removeprefix('superglm.')}:{frame.f_code.co_name}"
        if module.startswith("superglm") and filename != _THIS_FILE:
            if not found or found[-1] != label:
                found.append(label)
        elif (
            fallback is None
            and filename != _THIS_FILE
            and not any(m in filename for m in _LIBRARY_MARKERS)
        ):
            fallback = label
        frame = frame.f_back
    caller = found[0] if found else (fallback or "<unknown>")
    return caller, found[1] if len(found) > 1 else ""


class _State(threading.local):
    def __init__(self) -> None:
        self.stack: list[_Frame] = []
        self.hidden = 0


_ACTIVE: WorkLedger | None = None
_STATE = _State()


def _inputs(values, named: Any = None):
    """``(arrays, tokens, first)``: the arrays among ``values``, a token for every other value.

    A list or tuple contributes its arrays.  Numbers and strings enter the digest by value
    (``mode="r"`` is a different operation), other objects by identity.  ``first`` is the
    array that update detection compares: ``named`` when given, else the first array.
    """
    arrays: list[np.ndarray] = []
    tokens: list[str] = []
    for value in values:
        for item in value if isinstance(value, (list, tuple)) else (value,):
            if isinstance(item, np.ndarray):
                arrays.append(item)
            elif isinstance(item, (bool, int, float, str, type(None))):
                tokens.append(repr(item))
            else:
                tokens.append(f"{type(item).__name__}@{id(item)}")
    if isinstance(named, (list, tuple)) and named:
        named = named[0]
    first = named if isinstance(named, np.ndarray) else (arrays[0] if arrays else None)
    return arrays, tokens, first


def _prepare(target: _Target, signature, args, kwargs):
    """What one call needs recorded, read before the call (which may overwrite its inputs)."""
    if signature is None:
        cost_input = (args, kwargs)
        arrays, tokens, first = _inputs((*args, *kwargs.values()))
    else:
        bound = signature.bind_partial(*args, **kwargs).arguments
        cost_input = ((), bound)
        arrays, tokens, first = _inputs(bound.values(), bound.get(target.input))
    caller, via = _callers()
    return first, cost_input, _fingerprint(arrays, tokens, first), caller, via


def _wrap(target: _Target, original):
    signature = inspect.signature(original) if target.input is not None else None
    hides = target.role != COMPOSITE

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        ledger, state = _ACTIVE, _STATE
        if ledger is None or state.hidden:
            return original(*args, **kwargs)
        began = time.perf_counter()
        try:
            prepared = _prepare(target, signature, args, kwargs)
        except Exception as error:  # a ledger fault must never change the fit
            ledger.errors.append(f"{target.kind}: {error!r}")
            return original(*args, **kwargs)
        frame = _Frame()
        state.stack.append(frame)
        state.hidden += hides
        result = None
        started = time.perf_counter()
        preparing = started - began
        try:
            result = original(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - started
            state.hidden -= hides
            state.stack.pop()
            if state.stack:
                state.stack[-1].child_seconds += elapsed
            recording = time.perf_counter()
            try:
                ledger._record(target, prepared, result, elapsed - frame.child_seconds)
            except Exception as error:
                ledger.errors.append(f"{target.kind}: {error!r}")
            spent = preparing + time.perf_counter() - recording
            ledger.bookkeeping += spent
            if state.stack:
                state.stack[-1].child_seconds += spent

    return wrapper


def _resolve(owner: str):
    module_name, _, class_name = owner.partition(":")
    holder = importlib.import_module(module_name)
    return getattr(holder, class_name) if class_name else holder


@contextmanager
def work_ledger() -> Iterator[WorkLedger]:
    """Record the expensive work of every fit run inside the block."""
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("a work ledger is already recording; nest no ledgers")
    ledger = WorkLedger()
    undo: list[tuple[Any, str, Any]] = []
    superglm_modules = [
        module
        for name, module in list(sys.modules.items())
        if name.startswith("superglm") and name != __name__ and module is not None
    ]
    try:
        for target in TARGETS:
            try:
                holder = _resolve(target.owner)
                original = (
                    holder.__dict__[target.name]
                    if isinstance(holder, type)
                    else getattr(holder, target.name)
                )
            except (ImportError, AttributeError, KeyError):
                ledger.skipped.append(target.kind)
                continue
            wrapper = _wrap(target, original)
            places = [(holder, target.name)]
            for name in target.also:
                twin = importlib.import_module("scipy.linalg." + name)
                if getattr(twin, target.name, None) is original:
                    places.append((twin, target.name))
            if not isinstance(holder, type):
                places += [
                    (module, key)
                    for module in superglm_modules
                    for key, value in list(vars(module).items())
                    if value is original
                ]
            for place, name in places:
                undo.append((place, name, original))
                setattr(place, name, wrapper)
        _ACTIVE = ledger
        started = time.perf_counter()
        try:
            yield ledger
        finally:
            ledger.elapsed = time.perf_counter() - started
    finally:
        _ACTIVE = None
        for place, name, original in reversed(undo):
            setattr(place, name, original)
