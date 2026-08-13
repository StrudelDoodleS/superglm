"""The screening edf path costs O(L) in the level count, proved by counting.

``superglm.screening._arrow`` claims to factor a block-arrow matrix in
``O(L * (g^3 + g^2 r + g r^2))`` where densifying it would be
``O((L g + r)^3)`` and would allocate ``(L g + r)^2`` doubles.  The claim is
load-bearing -- it is the whole reason the path exists, since the dense route
is refused above about 124 levels and would need terabytes at fifty thousand
-- and until now it was untested, because the obvious test is a benchmark and
a benchmark cannot run here.  The machine that runs this suite routinely
carries several sessions at once, and an A/B timing ratio taken on it has
spanned a factor of six.

Counting settles it instead, and settles it better.  Cost for a batched
routine is ``batch * f(core shape)``: NumPy's linear algebra works on the last
two axes and broadcasts over the rest, so one call on an ``(L, g, g)`` stack
is ``L`` independent ``g x g`` factorizations.  Splitting the two apart turns
the complexity claim into two statements that are decidable from a call log:

* the *core shapes* are drawn from a fixed set that does not move with ``L``
  -- no matrix is ever ``L``-sized, which is the dense temporary;
* the *work* with batches unrolled, the largest array touched, and the peak
  allocation all grow linearly.

None of it can be corrupted by load, and all of it runs in CI.  The first two
reproduce exactly, being counts and geometry; the peak does not, since it picks
up incidental Python object churn, so it is asserted only as a growth bound and
never quoted as a figure.

Rows and spline width are held fixed across the sweep so that the level count
is the only thing that varies.  A generator that grew the row count alongside
``L`` -- the natural one, six rows per level -- would confound the two, and the
row dimension does legitimately appear in operand shapes.

What counting does *not* settle is stated here rather than left to be
discovered.  ``_arrow``'s cost has three terms; the ``g^3`` one is the
eigendecomposition and is counted, while ``g^2 r`` and ``g r^2`` live in
einsums and matrix products, which no recorder can see.  Allocation bounds
their space and nothing here bounds their time --
``test_quadratic_work_built_from_matrix_products_is_invisible_to_the_counter``
pins that hole open so it cannot close and reopen unnoticed.

Three of the tests below exist to prove the other three discriminate: they run
the same assertions against implementations lacking the property and require
them to fail.  A dense factorization keeps its counts small but grows its
shapes; a pairwise loop keeps its shapes small but does ``L^2`` work; a loop
written with ``@`` defeats both, on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

from superglm.screening._arrow import factor_arrow
from superglm.screening._structured import spline_cat_moments, structured_ladder

from ._linalg_cost import (
    assert_core_shapes_independent,
    assert_grows_linearly,
    record_linalg_calls,
    report,
)

BUDGETS = (2.0, 4.0, 8.0, 16.0)
LEVELS = (64, 128, 256, 512)
N_ROWS = 40
K_A = 5

# Small enough that the dense control's O(L^3) factorization stays quick; the
# assertions it has to fail do so at the first size that differs.
CONTROL_LEVELS = (32, 64, 128)


def _pair_inputs(n_levels: int, *, n_rows: int = N_ROWS, k_a: int = K_A, seed: int = 7):
    """Moment inputs for one spline x categorical pair with *n_levels* levels."""
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_rows, k_a)),
        np.eye(k_a),
        rng.normal(size=(n_rows, n_levels)),
        rng.uniform(0.1, 1.0, size=(n_rows, n_levels)),
        np.arange(1, n_levels, dtype=np.intp),
    )


def _record_ladder(n_levels: int):
    """Run the whole edf ladder for one pair and return its call log."""
    B_a, S_a, S_cell, W_cell, level_rows = _pair_inputs(n_levels)
    with record_linalg_calls() as record:
        pair = spline_cat_moments(B_a, S_a, S_cell, W_cell, level_rows)
        rungs = structured_ladder(pair, budgets=BUDGETS)
    assert rungs is not None and len(rungs) == len(BUDGETS), (
        f"the ladder refused {n_levels} levels, so nothing was measured"
    )
    return record


@pytest.fixture(scope="module")
def ladder_records():
    return [_record_ladder(n_levels) for n_levels in LEVELS]


def test_the_level_count_appears_only_as_a_batch_never_as_a_matrix_dimension(
    ladder_records,
):
    """The whole O(L) claim in one sentence, and it is decidable from the log."""
    # Before anything else: prove the recorder saw the path.  Every assertion
    # below degrades to a tautology on an empty log -- an empty shape set
    # matches an empty shape set, a maximum over nothing is zero, and the
    # empty batch set is a subset of everything -- so the guard is not
    # decoration.  Interception could stop working while the ladder keeps
    # running, and this test carries the headline claim.
    for n_levels, record in zip(LEVELS, ladder_records, strict=True):
        assert record.factorizations(), f"nothing recorded at {n_levels} levels"

    assert_core_shapes_independent(LEVELS, ladder_records)

    # The fixed set is small: arrow blocks are k_a square, the border is
    # (1 + k_a) square, and the row factors are n_rows x k_a.  Densifying
    # would put L * k_a -- at least 320 here -- in its place.
    widest = max(record.max_core_dim() for record in ladder_records)
    assert widest <= N_ROWS, report(ladder_records[-1])

    # Where L does appear, it appears as a count of small matrices.  A call
    # batching over anything else -- level pairs, or levels times spline
    # width -- would be super-linear work that the shape check cannot see.
    # Bounded rather than enumerated: `_trace_chunk_width` caps a batch by a
    # memory budget, so a larger sweep would legitimately split one batch of
    # L into several smaller ones.
    for n_levels, record in zip(LEVELS, ladder_records, strict=True):
        widest_batch = max(call.batch for call in record.factorizations())
        assert widest_batch <= n_levels, (
            f"at {n_levels} levels a factorization batched {widest_batch}, "
            f"which is more than one per level\n{report(record)}"
        )
        assert widest_batch >= n_levels - 1, (
            f"at {n_levels} levels nothing batched over the levels at all "
            f"(widest {widest_batch})\n{report(record)}"
        )


def test_the_edf_path_work_and_allocation_grow_linearly_in_the_level_count(
    ladder_records,
):
    """Factorizations and bytes both scale with L, not with L squared."""
    assert_grows_linearly(
        LEVELS,
        [record.elementary_factorizations() for record in ladder_records],
        label="elementary factorizations",
    )
    assert_grows_linearly(
        LEVELS,
        [record.peak_bytes for record in ladder_records],
        label="peak allocation",
    )
    # The largest single array, counting what routines return as well as what
    # they were given.  An assembly whose *result* is the dense object --
    # block_diag over L blocks, say -- keeps every operand small and would
    # slip past the shape invariant entirely.
    #
    # Held to a much tighter bound than the two above, because it earns one:
    # it is structural rather than search-dependent and doubles at exactly
    # 2.0000, where the factorization count's constant swings 4.4% with the
    # bisection.  At 1.05 this rejects anything from O(L^1.07) up.
    assert_grows_linearly(
        LEVELS,
        [record.max_elements() for record in ladder_records],
        label="largest array",
        tolerance=1.05,
    )


def _record_dense_arrow(n_levels: int):
    """Factor the same arrow matrix the way the dense path would.

    This is the implementation ``_arrow`` exists to replace, assembled from
    the same blocks: one ``(L g + r)`` square eigendecomposition.
    """
    g, r = K_A, 1 + K_A
    rng = np.random.default_rng(3)
    blocks = rng.normal(size=(n_levels, g, g))
    blocks = blocks @ np.swapaxes(blocks, -1, -2) + g * np.eye(g)
    coupling = rng.normal(size=(n_levels, r, g))
    border = np.eye(r) * (n_levels * g + r)

    with record_linalg_calls(packages=("tests",)) as record:
        width = n_levels * g
        dense = np.zeros((width + r, width + r))
        for q in range(n_levels):
            cell = slice(q * g, (q + 1) * g)
            dense[cell, cell] = blocks[q]
            dense[width:, cell] = coupling[q]
            dense[cell, width:] = coupling[q].T
        dense[width:, width:] = border
        np.linalg.eigh(dense)
    return record


def _record_pairwise_arrow(n_levels: int):
    """Factor every ordered pair of blocks: small shapes, quadratic work.

    Not a candidate implementation -- a control for the growth assertion, which
    the dense factorization above passes (it does one big factorization, not
    many small ones) and which therefore needs its own counterexample.
    """
    rng = np.random.default_rng(3)
    blocks = rng.normal(size=(n_levels, K_A, K_A))
    blocks = blocks @ np.swapaxes(blocks, -1, -2) + K_A * np.eye(K_A)

    with record_linalg_calls(packages=("tests",)) as record:
        for q in range(n_levels):
            np.linalg.eigh(blocks + blocks[q])
    return record


def test_a_dense_factorization_fails_the_shape_invariant():
    """The shape assertion is discriminating: it rejects the dense path."""
    records = [_record_dense_arrow(n_levels) for n_levels in CONTROL_LEVELS]

    with pytest.raises(AssertionError, match="core shapes depend on size"):
        assert_core_shapes_independent(CONTROL_LEVELS, records)

    # Its allocation is quadratic too, which the arrow path's is not.
    with pytest.raises(AssertionError, match="faster than linearly"):
        assert_grows_linearly(
            CONTROL_LEVELS,
            [record.peak_bytes for record in records],
            label="peak allocation",
        )


def test_a_quadratic_loop_fails_the_growth_bound():
    """The growth assertion is discriminating, and shapes alone would not be.

    Every core shape here is ``(k_a, k_a)`` and L-independent, so the shape
    invariant passes.  Only counting the work catches it.
    """
    records = [_record_pairwise_arrow(n_levels) for n_levels in CONTROL_LEVELS]

    assert_core_shapes_independent(CONTROL_LEVELS, records)

    with pytest.raises(AssertionError, match="faster than linearly"):
        assert_grows_linearly(
            CONTROL_LEVELS,
            [record.elementary_factorizations() for record in records],
            label="elementary factorizations",
        )


def test_quadratic_work_built_from_matrix_products_is_invisible_to_the_counter():
    """The instrument's blind spot, pinned rather than described.

    ``A @ B`` is a bytecode operator, not a call, so no recorder sees it --
    ``sys.monitoring`` included.  A quadratic loop written with ``@`` in
    bounded space therefore passes both assertions, and this test fails if
    that ever stops being true so the limits section cannot quietly rot.

    It also bounds what the counting half of this file proves.  ``_arrow``
    claims ``O(L (g^3 + g^2 r + g r^2))``.  The ``g^3`` term is the ``eigh``
    and is counted; the other two live in einsums and matrix products and are
    not.  Allocation covers their space, nothing here covers their time.
    """
    records = []
    for n_levels in CONTROL_LEVELS:
        rng = np.random.default_rng(5)
        blocks = rng.normal(size=(n_levels, K_A, K_A))
        with record_linalg_calls(packages=("tests",)) as record:
            total = np.zeros((K_A, K_A))
            for q in range(n_levels):  # L^2 work, O(1) extra space
                total += (blocks @ blocks[q]).sum(axis=0)
        records.append(record)

    assert all(not record.factorizations() for record in records)
    assert_grows_linearly(
        CONTROL_LEVELS,
        [record.peak_bytes + 1 for record in records],
        label="peak allocation",
    )


def test_one_arrow_factorization_costs_exactly_one_eigendecomposition_per_block():
    """The kernel's own claim, at its own level: L blocks plus one border.

    ``factor_arrow`` is where the linearity comes from, and here the count is
    exact rather than bounded -- there is no search above it to vary.

    Two calls, whatever ``L`` is, is the module's stated "nothing loops in
    Python over L".  Dropping to a per-block loop would leave the elementary
    count and the shapes untouched and only move this number, which is why it
    is asserted separately from them.
    """
    g, r = 4, 3
    for n_levels in (16, 64):
        rng = np.random.default_rng(11)
        blocks = rng.normal(size=(n_levels, g, g))
        blocks = blocks @ np.swapaxes(blocks, -1, -2) + g * np.eye(g)
        coupling = rng.normal(size=(n_levels, r, g))
        border = np.eye(r) * (n_levels * g + r)

        with record_linalg_calls() as record:
            factor_arrow(blocks, coupling, border)

        assert record.elementary_factorizations() == n_levels + 1
        assert record.core_signature() == {
            ("numpy.linalg.eigh", ((g, g),)),
            ("numpy.linalg.eigh", ((r, r),)),
        }
        assert len(record.factorizations()) == 2, (
            f"{n_levels} levels took {len(record.factorizations())} calls; the "
            f"block axis is meant to be batched, not looped\n{report(record)}"
        )
