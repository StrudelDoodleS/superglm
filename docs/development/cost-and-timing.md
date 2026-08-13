# Cost claims and the clock

**Counts and allocation settle complexity and cost claims. The clock answers
only absolute wall-time questions, only on a quiet machine.** When the machine
is not quiet, write "unmeasured" — a number quoted from a contended box is
worse than no number, because it looks like evidence.

## Why the clock is unusable here by default

This repository is normally worked on by several sessions at once, so the
normal operating condition is heavy oversubscription, not a quiet box. Observed
during ordinary work: 16 cores, one-minute load average between 30 and 56, with
concurrent pytest processes at 608%, 355% and 335% CPU.

What that costs in practice: an O(L) scaling claim on the screening `edf` path
was A/B tested at load 53 and the ratios came back **0.19, 0.63, 0.37, 1.10** —
a factor of six across repetitions of the same comparison. Every one of those
numbers had to be discarded, and the claim went unverified. Duration has misled
in the same way outside benchmarking: a suite was reported "stalled" when it was
contended, and a completed CI review was called "stuck" on the strength of how
long it had taken.

Three things that sound like they would fix it, and do not:

- **Pinning the thread pools.** `superglm._blas_threads` and the `OMP_NUM_THREADS`
  family remove fan-out, which is real and worth doing. They do nothing about
  cache and memory-bandwidth contention, so above roughly 2x oversubscription no
  clock is usable, pinned or not.
- **`taskset`.** Confining a process to some cores buys little when every core
  is contended.
- **Falling back to hardware counters.** There is no fallback on this machine:
  `perf` is not installed and `/proc/sys/kernel/perf_event_paranoid` is `3`,
  which denies perf events to unprivileged users even if it were; `valgrind`
  and `cachegrind` are not installed either.

## Check the load before timing anything

```bash
cat /proc/loadavg    # first field is the one-minute average
nproc
```

If the one-minute average is above about twice `nproc`, do not time. Take counts
instead, or record the question as unmeasured and move on.

## The instrument

`tests/_linalg_cost.py` records calls to a **fixed registry** of `numpy.linalg`
and `scipy.linalg` entry points made inside a block, with operand shapes and
dtypes, plus the `tracemalloc` peak. It is deterministic: the log is a function
of the code path taken, not of how busy the machine is.

The registry is a list, not the whole namespace — an unregistered routine
produces no entry at all, and a complexity assertion built on the log would
pass while that work stayed invisible. `NUMPY_ROUTINES` and `SCIPY_ROUTINES`
are the boundary, and `test_the_registry_covers_every_routine_superglm_calls`
scans `src/superglm` to keep them ahead of what this package actually calls.

```python
from ._linalg_cost import assert_grows_linearly, record_linalg_calls

with record_linalg_calls() as record:
    pair = spline_cat_moments(B_a, S_a, S_cell, W_cell, level_rows)
    structured_ladder(pair, budgets=budgets)

record.counts()                     # calls per routine
record.core_signature()             # the distinct (routine, core shapes) pairs
record.elementary_factorizations()  # work, batches unrolled
record.max_elements()               # largest array seen, operands and results
record.peak_bytes                   # peak allocation over the block
```

Assertions built on a call log have one failure mode worth guarding above all
others: **an empty log satisfies almost anything.** An empty set of shapes
matches an empty set of shapes, a maximum over nothing is zero, and an empty
set of batch sizes is a subset of every set. If interception silently stops
working, every such assertion goes green while measuring nothing at all.
`assert_core_shapes_independent` refuses an empty log for that reason, and any
new assertion should start by proving the recorder saw the path.

The split that makes this a cost model rather than a tally is **core shape
versus batch**. NumPy's linear algebra operates on the last two axes and
broadcasts over the rest, so one call on an `(L, g, g)` stack is `L`
independent `g x g` factorizations. Cost is `batch * f(core shape)`, and the two
halves answer different questions: a batch that tracks the level count is the
algorithm working level by level, while a *core shape* that tracks it is the
dense temporary. Counting them separately turns "this path is O(L)" into two
statements decidable from a call log.

Worked example, from `tests/test_screening_cost_scaling.py` on the screening
`edf` ladder with rows and spline width held fixed:

| levels | elementary factorizations | largest array | widest core dimension |
|---:|---:|---:|---:|
| 64 | 7,231 | 12,800 | 40 |
| 128 | 14,719 | 25,600 | 40 |
| 256 | 30,207 | 51,200 | 40 |
| 512 | 57,855 | 102,400 | 40 |

Eight times the levels, eight times the work, eight times the largest array,
and no matrix any larger — the arrays that grow do so in their *batch* axis,
which is the algorithm factoring more small blocks rather than one bigger one.
Peak allocation grows sub-linearly over the same span, between 5x and 7x
depending on the run. Densifying the arrow kernel — the implementation this
path replaces — moves the widest core dimension to 378, 762, 1,530 and 3,066,
and multiplies the peak by 3.76 per doubling against the 2.0 a linear path
shows.

**Counts repeat exactly.** The four factorization figures above were taken once
on an idle machine and again at load 15, and were identical to the digit — as
are the largest-array figures, which are pure geometry. The traced peak is the
one column that does not: it picks up incidental Python object churn, including
the recorder's own call log, and has been seen to move by half between runs.
Quote it as an order of magnitude, assert it only as a growth bound, and put
the weight of any claim on the two exact columns.

Exactly reproducible is not the same as constant in the size, though, and the
distinction decides how tightly each channel can be asserted. The ladder
bisects a data-dependent number of times, so its factorizations *per level*
runs between 113.0 and 118.0 across the sweep — a 4.4% swing that is perfectly
repeatable and still not flat. A quantity that is *structural* has no such
swing: the largest array the path touches doubles at exactly 2.0000. So put the
tight bound on the structural channel and a documented looser one on the
search-dependent channel, rather than one tolerance over both.

Sizing that tolerance has a trap worth naming, because the intuitive reason for
it is backwards. A positive fixed overhead does **not** need an allowance: for
`a*L + b` with `b > 0`, doubling gives `(2aL + b)/(aL + b) < 2`, so overhead
makes the ratio *smaller*. What pushes a linear cost above the size ratio is a
*negative* intercept — a term like `a(L-1)`, one factorization per level with
the first merged away, doubling at `2 + 1/(L-1)`.

## Which instrument answers which question

| question | instrument | why |
|---|---|---|
| Is this path O(L)? | call counts and operand core shapes | The count *is* the complexity, not a proxy for it. Exact, and load-invariant. |
| Did an accidental dense temporary appear? | `max_elements`, then `peak_bytes` | `max_elements` spans operands *and* results, so an assembly whose output is the dense object is visible even though every input stayed small. `peak_bytes` catches whatever built it. |
| Did we stop batching and start looping? | number of calls, at fixed work | Elementary count and shapes both survive that change; only the call count moves. |
| Is this fast enough in seconds? | the clock, on a quiet machine | Nothing else answers an absolute wall-time question. |

**Three blind spots, all real.** Matrix multiplication is a bytecode operator,
not a call, so `A @ B` never enters the log. (`sys.monitoring` can *detect* it
through the `INSTRUCTION` event, but that event carries no operands, so it
answers "a product happened" and never "how big".) Quadratic work written with
`@` in bounded space therefore passes every assertion here, and
`test_quadratic_work_built_from_matrix_products_is_invisible_to_the_counter`
pins that hole so it cannot quietly close and reopen. Allocation is the
backstop for a dense temporary's *space*; nothing here backstops its *time*.

Allocation is not unconditional either. `tracemalloc` sees NumPy buffers
because NEP 49 has NumPy register them, and sees whatever LAPACK workspace
SciPy allocates as an array — but a compiled extension using its own allocator
is invisible to the call log and the peak alike.

And the factorization count cannot separate `O(L)` from `O(L log L)`. Its
constant swings 4.4% with the ladder's bisection, which is wider than the gap
being looked for; `L log L` doubles at 2.25–2.33 where linear doubles at 2.05.
The largest-array channel *can* — it doubles at exactly 2.0000 and is held to a
bound that rejects anything from `O(L^1.07)` up.

That bounds what counting proves about the arrow path specifically. Its cost is
`O(L(g³ + g²r + gr²))`; the `g³` term is the eigendecomposition and is counted,
while the other two live in einsums and matrix products and are not.

And counts are necessary, not sufficient in a second sense: they pin the
*shape* of the work and say nothing about whether it is correct — a routine
blind to a weighting term passes every count assertion. Numerical-equivalence
tests stay; counting replaces the timing test only.

## When a wall-clock number is warranted

Timing belongs in `benchmarks/`, never in `tests/`. `benchmarks/local_perf_gate.py`
is the existing route: it refuses to certify wall time unless the baseline both
enables certification and names an operator-asserted local machine profile, and
it refuses hosted CI outright. That refusal is the correct default — leave it
refusing until someone has established a quiet, freshly calibrated machine.

Before taking such a measurement, check `/proc/loadavg` as above, and take it
exclusively: one agent at a time, or the numbers describe the other agents.

## Prior art

The pattern is Django's
[`assertNumQueries`](https://docs.djangoproject.com/en/stable/topics/testing/tools/),
which has guarded against the N+1 query for years by asserting an exact count of
the expensive operation instead of its duration. This is the linear-algebra
transliteration of it.

Stating cost as factorizations times operand shape is the textbook cost model —
Golub and Van Loan, *Matrix Computations*, 4th ed. (Johns Hopkins, 2013), and
the flop-count tables of the
[*LAPACK Users' Guide*](https://www.netlib.org/lapack/lug/), 3rd ed. (SIAM,
1999).

That timing on a shared machine misleads is itself established: Mytkowicz,
Diwan, Hauswirth and Sweeney, ["Producing Wrong Data Without Doing Anything
Obviously Wrong!"](https://dl.acm.org/doi/10.1145/1508244.1508275), ASPLOS 2009,
found measurement bias pervasive enough to reverse conclusions; and
[`pytest-benchmark`'s own FAQ](https://pytest-benchmark.readthedocs.io/en/latest/faq.html)
names "bad isolation" as its central failure and advises bare metal, which this
machine is not. The deterministic-benchmarking tradition — Cachegrind's
instruction counts, and the `iai-callgrind` line of Rust benchmark harnesses —
escapes the noise but keeps an assumption we do not need, that the count is a
*proxy* for time, valid only when instructions-per-cycle is stable. Counting
factorizations is not a proxy for the complexity claim; it is the claim.

Two nearby things that do not cover this. `asv`, which NumPy and SciPy use,
supports a counted metric through its `track_*` benchmarks, but it is a
historical dashboard rather than a pass/fail assertion.
[`pytest-memray`](https://pytest-memray.readthedocs.io/en/latest/usage.html)'s
`@pytest.mark.limit_memory` asserts an absolute allocation ceiling, which is a
reasonable alternative to `peak_bytes` when a fixed budget is wanted rather than
a growth rate. Neither counts operations.

A sweep of the literature for asserting flop counts, factorization counts or
operand shapes inside a numerical library's test suite found nothing: the
published work on algorithmic complexity in tests *infers* a complexity class by
fitting noisy measurements — Goldsmith, Aiken and Wilkerson, ["Measuring
Empirical Computational
Complexity"](https://cs.stanford.edu/~aiken/publications/papers/fse07.pdf),
ESEC/FSE 2007, and the input-sensitive profiling line that followed it — which
is the opposite direction from pinning an exact count. NumPy, SciPy and
scikit-learn do not assert complexity or allocation in their unit tests at all.
On the evidence of that search, this design is ours.
