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

`tests/_linalg_cost.py` records every `numpy.linalg` and `scipy.linalg` call
made inside a block, with operand shapes and dtypes, plus the `tracemalloc`
peak. It is deterministic: the log is a function of the code path taken, not of
how busy the machine is.

```python
from ._linalg_cost import assert_grows_linearly, record_linalg_calls

with record_linalg_calls() as record:
    pair = spline_cat_moments(B_a, S_a, S_cell, W_cell, level_rows)
    structured_ladder(pair, budgets=budgets)

record.counts()                     # calls per routine
record.core_signature()             # the distinct (routine, core shapes) pairs
record.elementary_factorizations()  # work, batches unrolled
record.peak_bytes                   # peak allocation over the block
```

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

| levels | elementary factorizations | peak | widest core dimension |
|---:|---:|---:|---:|
| 64 | 7,231 | ~370 KiB | 40 |
| 128 | 14,719 | ~645 KiB | 40 |
| 256 | 30,207 | ~1,275 KiB | 40 |
| 512 | 57,855 | ~2,530 KiB | 40 |

Eight times the levels, eight times the work and the bytes, and not one
operand any larger. Densifying the arrow kernel — the implementation this path
replaces — moves the widest core dimension to 378, 762, 1,530 and 3,066, and
multiplies the peak by 3.76 per doubling against the 2.0 a linear path shows.

The two columns are reproducible to different degrees, and the difference is
worth knowing. **Counts repeat exactly**: the four factorization figures above
were taken once on an idle machine and again at load 15, and were identical to
the digit. The traced peak moved by about 1% between those runs, because it
picks up incidental Python object churn along with the arrays. So assert counts
as equalities and allocation as a growth bound, never the other way round.

## Which instrument answers which question

| question | instrument | why |
|---|---|---|
| Is this path O(L)? | call counts and operand shapes | The count *is* the complexity, not a proxy for it. Exact, and load-invariant. |
| Did an accidental dense temporary appear? | `peak_bytes` | Implementation-independent. A `@` product is a bytecode operator, so no call recorder sees it — allocation does. |
| Did we stop batching and start looping? | number of calls, at fixed work | Elementary count and shapes both survive that change; only the call count moves. |
| Is this fast enough in seconds? | the clock, on a quiet machine | Nothing else answers an absolute wall-time question. |

Counts are necessary, not sufficient. They pin the *shape* of the work and say
nothing about whether it is correct — a routine blind to a weighting term
passes every count assertion. Numerical-equivalence tests stay; counting
replaces the timing test only.

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
