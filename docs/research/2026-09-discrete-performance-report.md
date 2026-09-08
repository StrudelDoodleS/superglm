# Discrete execution performance

Discrete execution improvements reduce complete-fit time on several public
fixtures, but the C1 performance gate remains open. The fragmented Gaussian
size sweep at `ed84669a` finds automatic chunking slower than exact dense fitting
from 65,536 to 1,048,576 rows. Subsequent range and renderer changes improve
complete fits at 262,144 rows. Those measurements support automatic bounded
panels for a narrow mixed ordinary layout. An experimental dense execution
override is faster but uses more memory; automatic dense selection is deferred.

The latest complete-fit measurements cover
`56d9507a8cb4f505092175624012a68b6d0021d1`. The subsequent automatic panel policy
has passed focused tests and independent review; final integration and default
route validation remain in progress. The
performance baseline is the frozen post-C3 source
`5f994c8f6ac0501606594e2f36bfc0cd24050ec1`; the plan checkpoint `0a15736e`
has the same production source. This is distinct from published v0.31.0 at
`8962c452`, the earlier C3/C1 release comparison base. Improvements below must
not be presented as measurements against that release.

The tracked [performance receipt](../../benchmarks/discrete_performance_receipt.json)
consolidates the size, thread, preparation and tabmat windows with raw-summary
hashes, source pins, ranges and qualifications.

## Execution changes at this checkpoint

- Discrete cross-products select support histograms or bounded row contractions
  using support size, row count, contraction work and setup costs. Tensor Gram
  and prediction kernels avoid repeated temporaries. Unsupported numeric domains
  retain fallback paths.
- Value-only passes avoid discarded geometry plans, and ordered row selections
  avoid repeated sorting. Categorical subsets now avoid redundant normalization
  and copying while retaining validation and owned storage.
- Gamma small-shape series run in compiled batches with truncation and rounding
  checks. Initialization evaluates identical shapes once while preserving the
  original row-term summation semantics.
- Chunked smoothing reuses raw likelihood scores and observed curvature only
  when the previous endpoint and live inputs pass a certificate. Penalty terms
  are rebuilt for each smoothing parameter update. Certificates retain O(p²)
  numeric state and use a bounded-memory input digest. Built-in family adapters
  preserve the solver/family boundary; coverage includes categorical, random
  effects, CSR, factor smooths and the built-in spline-by-categorical forms.
  Unsupported extensions refresh, and terminal evaluation rules are unchanged.
- `e23cd41c` adds bounded optional panels for ordinary grouped curvature and a
  guarded signed reduction route for factor-smooth/dense products with multiple
  right-hand sides. `50b5e8bb` compiles the panels' arithmetic-range checks and
  warms supported writable/readonly layouts. `ed84669a` adds the categorical
  subset improvement. Subsequent range and renderer changes are measured below.
  The automatic chunk-size policy is unchanged.
- Automatic panels now admit exact built-in mixed layouts containing numeric,
  categorical, stored spline and spline-by-category groups in every predictor
  with slopes. Each group has at most 32 columns; intercept-only companions are
  allowed. This is an initial tested scope, not a measured speed crossover.
  The additional panel workspace allowance is 64 MiB, separate from the existing
  8 MiB row chunk selector; neither bounds whole-process RSS. Explicit off and
  integer-budget overrides, numerical refusal and grouped fallback remain.
  Specialized, custom and unsupported layouts keep their existing routes.

## Representations and workloads

Stored-support execution and approximate binning are separate. A support budget
covering every supplied covariate value preserves those training values; a
continuous covariate exceeding the budget is binned. Agreement between execution
routes for one stored design does not prove agreement with the exact feature
basis.

| Public fixture | Training rows | Representation |
| --- | ---: | --- |
| Continuous synthetic | 20,000 | Continuous covariates binned |
| Support-32 synthetic | 100,000 | 32 support values per covariate; lossless supplied support |
| freMTPL2 Gamma severity | 89,800 | Four copies of 22,450 training policies; 73, 21 and 82 distinct covariate values fit within 256 bins |
| Fragmented Gaussian | 65,536–1,048,576 | Continuous covariates binned; fixed 102-coefficient layout |

All use four knots and a 256-bin budget. Severity has 2,494 holdout rows and
evaluates the learned basis directly, including an unseen marginal value of 119.
The fragmented fixture has 2,000 holdout rows and combines numeric, categorical,
spline and supported spline-by-categorical groups. It does not exercise the
separate `FactorSmooth` API. Each predictor compiles to 15 groups: six singleton
dense, four categorical, three discrete spline and two discrete
spline-by-categorical groups, totaling 50 columns plus an intercept. Its size
sweep holds the specification and generator fixed; training prefixes are not
asserted to be nested.

## Repeated historical comparisons

These earlier checkpoints have three fresh-process repetitions per arm. Times
are median complete-fit seconds with min–max ranges. RSS gives median process
high-water marks captured at fit completion, in MiB, ordered baseline discrete /
checkpoint discrete / checkpoint exact.

| Fixture / checkpoint | Baseline discrete | Checkpoint discrete | Checkpoint exact | RSS |
| --- | --- | --- | --- | --- |
| Continuous / `aef4ee7e` | 2.342 (2.309–2.372) | 1.521 (1.167–1.562) | 1.387 (1.343–1.603) | 434 / 432 / 514 |
| Support-32 / `aef4ee7e` | 3.753 (3.427–4.015) | 2.116 (2.035–3.157) | 4.331 (4.172–4.471) | 481 / 483 / 695 |
| Severity / `6e611349` | 13.383 (13.115–14.926) | 3.821 (2.976–4.214) | 2.962 (2.804–3.929) | 613 / 612 / 648 |

The complete candidate SHAs are `aef4ee7e8ddfe263fb44e760e4fc108adadad6c9`
and `6e6113499d5bbf3e4e2c7488c26217ed8983b4bf`. Maximum absolute changes
in saved training-parameter arrays between baseline and candidate discrete fits
are 2.1e-15, 6.7e-16 and zero respectively. Smoothing/inner iteration counts
remain 9/26, 7/20 and 7/18. Separate candidate exact/discrete differences are
0.0051 for the binned continuous fixture, 3.8e-15 for support-32 and 5.2e-10
for severity. The last two concern lossless training support.

Support-32 demonstrates that chunked discrete execution can be faster than exact
fitting on a favorable layout. It does not resolve the fragmented-layout cost.
Severity exact/discrete timing ranges overlap. These historical improvements do
not certify the final source or a universal speed ratio.

## Fragmented layout: panels and execution choice

The first opt-in panel pilot at `e23cd41c` was unfavorable: discrete fit time
increased from 6.068 to 6.423 seconds. Separate profiling confirmed the intended
grouped route but found substantial Python arithmetic-range checking. Retain
that result alongside the follow-up: three repetitions at `50b5e8bb`, after
compiled checks and warmup, reduced median discrete time from 6.847 to 5.475
seconds (about 20%). Off/on ranges are 6.086–7.218 and 5.416–5.627 seconds;
RSS medians are 502.9 and 501.7 MiB. Iterations agree and saved outputs differ
only at roundoff scale. Exact controls took about 2.4 seconds. This supports a
local panel benefit and leaves a substantial execution gap.

At `ed84669a`, an initial one-fit dense override reduced discrete time from
6.014 to 2.244 seconds at 65,536 rows, increasing RSS from about 503 to 582 MiB;
exact used about 625 MiB. A separate fixed-layout size sweep follows below.
Every entry is **one fresh fit**, not a median or crossover estimate. Cells give
worker wall seconds / process high-water RSS at fit completion in MiB.

| Rows | Exact dense | Discrete automatic | Discrete with 64 MiB panels | Discrete dense override |
| ---: | ---: | ---: | ---: | ---: |
| 65,536 | 2.734 / 625.0 | 7.728 / 503.5 | 5.755 / 502.2 | 2.184 / 582.4 |
| 262,144 | 8.609 / 1268.9 | 19.709 / 768.2 | 17.442 / 770.4 | 8.024 / 1094.4 |
| 1,048,576 | 30.014 / 3883.3 | 70.093 / 1831.6 | 62.747 / 1832.5 | 26.748 / 3177.7 |

The override selects the existing dense backend for the same stored discrete
basis; it does not rebuild the exact basis. Stored-representation hashes match
across the three discrete arms at each N. Backend receipts resolve automatic
and panel arms to `distributional-chunked-v1`, and exact/override arms to
`distributional-dense-v1`. The overrides are internal benchmark experiments;
they do not establish public API, serialization or default-policy behavior.

Each N has equal work counts across its four arms: inner/smoothing/geometry
counts are 22/9/23, 18/7/19 and 16/7/17 respectively. Work counts differ across
N, so the table is not a controlled per-iteration scaling law. For the same
stored representation, the largest saved training-parameter difference across
all three sizes is 4.5e-15; covariance, smoothing parameters, scores and terminal
curvature are saved separately for inspection. Exact/discrete maximum training
differences are approximately 0.00160, 0.000990 and 0.000507, reflecting the
separate binned-basis comparison. These are descriptive differences, not
certified bounds. Fits stop at `practical_plateau` with
`smoothing_certified=False`; they are not strictly certified REML benchmarks.

The sweep supports investigating execution selection independently of support
representation. Dense materialization buys time at a measurable memory cost in
this fixture. It does not justify an N-only crossover, a new automatic default,
or a claim that every discrete layout is slow. A panel budget bounds its
workspace, not whole-fit memory.

## BLAS-thread diagnostic

A separate frozen-source window at `ed84669a` compared one and four BLAS
threads on the 262,144-row fragmented fixture: two fresh fits for each of the
three stored-discrete execution arms, with the six-condition order reversed
for the second repetition. Runtime thread-pool inventories confirm the requested
limits at every recorded boundary; source, wrapper and activity checks passed.

| Execution | BLAS 1 wall seconds, median (range) | BLAS 4 wall seconds, median (range) | Process CPU seconds, median 1 / 4 | Fit RSS MiB, median 1 / 4 |
| --- | --- | --- | --- | --- |
| Automatic chunking | 18.241 (17.706–18.777) | 18.121 (17.333–18.908) | 18.222 / 36.518 | 769.8 / 782.9 |
| Optional panels | 16.394 (16.196–16.592) | 15.933 (15.558–16.308) | 16.370 / 64.402 | 766.6 / 784.9 |
| Dense override | 7.631 (7.477–7.785) | 5.721 (5.274–6.168) | 7.608 / 24.948 | 1094.2 / 1094.4 |

Four threads leave automatic chunking's wall time broadly unchanged, give a
small local panel improvement with overlapping ranges, and improve dense
execution more substantially while increasing CPU consumption. This diagnostic
does not explain away the fragmented chunking gap or establish a universal
thread recommendation.

All runs retain 18 inner and seven smoothing iterations and practical-plateau
stopping without strict certification. Stored representation hashes agree
across execution arms and repetitions **within each thread setting**. Between
settings, ten spline `R_inv` arrays have different hashes: the existing BLAS
scope includes compilation. Raw basis values, bin maps, shapes and group classes
match. Hashes alone do not quantify those transformation differences; the saved
fitted-output comparisons remain at roundoff scale, with maximum training
parameter difference 1.9e-15 and coefficient difference 6.7e-15. This is not a
bit-identical compiled-basis comparison across thread settings.

## What the historical profiles located

Separate one-thread cProfiles at `ed84669a` use the same 262,144-row public
fixture. Their seconds are diagnostic attribution, not uninstrumented fit-time
measurements. Immediate child edges partition one owner; cumulative parent and
descendant times must not be added together. NumPy matrix multiplication and
bin counting can appear in owner self time, so that time is not all Python
interpreter overhead.

Automatic chunk geometry's 17.356 profiled seconds divide into curvature-channel
work (14.269), likelihood-chunk iteration (2.655), score assembly (0.379), and
0.053 for owner self time and remaining direct children. With panels, the same
owner's 11.120 seconds divide into panel construction through its wrapper
(4.586), curvature channels (3.158), chunk iteration (2.854), scores (0.460),
and 0.062 for the remaining direct work. These are disjoint parts within each
owner, not additional costs on top of its total.

At this checkpoint, panel construction spends 3.557 profiled seconds in row rendering.
Across the entire panel profile, `_in_range` accounts for 1.469 self seconds
in the compiled range predicate. That is native checking cost, not the
old Python-loop defect corrected at `50b5e8bb`; it also overlaps the owner
partitions above. Dense geometry instead concentrates 4.152 self seconds in
matrix assembly. These findings support investigating repeated rendering,
row selection and range scans while preserving bounded workspace and numeric
refusals. They do not establish the performance of a proposed replacement.

## Range and renderer complete fits

Three fresh fits per arm compare frozen `ed84669a` with `56d9507a` on the
262,144-row fragmented workload. The five-arm order is forward, reverse, then
forward. Times are complete-fit medians in seconds; RSS is the process
high-water mark captured at fit completion, in MiB.

| Route | Old wall | New wall | Old CPU | New CPU | Old RSS | New RSS |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary automatic chunks | 22.648 | 17.906 | 22.618 | 17.889 | 771.46 | 777.34 |
| Explicit bounded panels | 17.172 | 12.207 | 17.158 | 12.196 | 769.29 | 777.30 |
| Stored discrete basis, dense control | — | 7.801 | — | 7.778 | — | 1100.81 |

Ordinary chunks improve by 20.9% and panels by 28.9% in this window. Old/new
wall ranges are 20.280–23.920 / 14.996–18.371 s for ordinary chunks, and
15.243–18.423 / 10.586–14.066 s for panels. The ranges do not overlap. Dense
controls span 7.626–8.041 s and retain the faster, higher-memory tradeoff.
The first old automatic fit has a 936.43 MiB high-water mark and a 19.50 s
warmup, versus 0.26–0.67 s warmup for the other workers. Warmup lies outside
fit timing; all observations, including this memory outlier, are retained.

All 15 stored representation hashes and iteration counts agree. Each fit has
18 coefficient iterations, seven smoothing iterations and 19 geometry builds.
Old/new same-route saved arrays agree exactly on this fixture; the largest
cross-route holdout difference is 1.11e-15. This is fixture evidence, not a
general bitwise-equivalence promise. Every new range call is accepted;
panel fits accept 627 builds and 1,881 curvature products, with a maximum
estimated workspace of 21,993,536 bytes. Timings retain disclosed integer-only
dispatch witnesses, with no call profiler or per-call clocks. Source, helper,
thread-pool and CPU-activity checks pass throughout.

## Raw-basis tabmat comparison

A separate prototype at `56d9507a` tests chunk-owned raw-basis tabmat matrices
on the same 262,144-row book, with 8,065-row chunks and one numerical thread. Construction,
conversion, channel copies, coefficient transforms and all geometry outputs
are included. Each method uses one fresh worker, one whole-geometry warmup and
three within-process repetitions: three workers in total. These give median
wall times of 0.791 s for grouped execution, 0.535 s
for panels and 1.467 s for raw-basis tabmat; corresponding process peaks are
517.74, 517.97 and 517.77 MiB. These are geometry-only diagnostics, not fits.

The prototype uses the actual stored support, including four nonzeros per
eight-column basis row in this fixture, and exercises native tabmat kernels.
Signed curvature and score results agree with norm-relative differences of
3.33e-16 and 2.62e-16. Representation, source and activity checks pass. This
constructor-inclusive result does not justify a full-fit adaptation. The
prototype and unfavorable receipts remain preserved outside production code;
they do not rule out other tabmat designs or workloads.

## Measurement and validation limits

Timed workers run serially in fresh interpreters with numerical threads fixed
to one except for the explicit BLAS-thread diagnostic, and public
`superglm.warmup()` before timing, retaining existing caches.
Additional JIT/cache misses may still occur during a fit. Arm order varies;
other numerical work is stopped and Headroom remains active. Worker wall time,
process CPU time and RSS receipts are primary; Headroom/Kompress tool wall time
is not benchmark time. Fit RSS includes runtime and compilation state and is
distinct from the later process peak during output collection.

All size-sweep fit endpoint activity screens passed and source/wrapper stability
checks passed. The first window stopped on a failed two-second preflight before
the million-row fits; its aborted manifest is retained. A continuation used
ten-second preflights with the same activity threshold. Guest CPU audits cannot
establish exclusive physical-host access. The size sweep retains lightweight
shared execution wrappers; expensive call-stack profiles are separate and their
costs are not complete-fit timings. Historical synthetic receipts predate fit
CPU recording; severity and the current sweep include it.

Full non-browser validation of production source `9f0e196c` records 12,350
passed and 109 skipped unique cases on Python 3.13, with `mpmath` and `pyarrow`
available. The four shards initially had one CI metadata failure: the test
duration manifest covered too few of the expanded suite. Adding 1,078 missing
entries from those measured JUnit durations, preserving existing entries, fixes
it; all eight CI contract tests pass on rerun. The counts use the latest outcome
per case and count repeated module-collection skips once. The three required
real-data suites contribute 84 passes and no skips. Ruff, formatting, dependency
and smoke checks pass; browser tests are outside this solver validation.

The subsequent range and panel-rendering checkpoint passes 602 combined focused
checks, with four inapplicable exact-category permutation cases skipped.
Contiguous value evaluation avoids ordinary group-subset construction while
preserving group reduction order and live numeric inputs. Immutable lookup
authority permits two-bound searches and releases obsolete backing storage on
refusal. Geometry retains owned chunk snapshots. Bounded support tables and
checked writers reduce repeated transformation and scanning work; public warmup
covers their compiled signatures. Independent reviews found no remaining
blocking issues. Their complete-fit comparison is recorded above; final actual
default complete-fit validation remains pending.

The automatic panel policy passes 172 focused tests, including 44 new dispatch,
override, refusal and lifetime regressions. Its default-dispatch regression
fails on the prior implementation. Independent review reports no remaining
findings. Final actual-default complete-fit checks remain.

The execution reviews found that tabmat supports signed weights, but a bounded
LSS route needs chunk-owned matrices, constructor/native-workspace accounting
and a strategy for rectangular curvature products. The constructor-inclusive
raw-basis comparison above does not support replacing the current kernel.
Further structural consolidation is deferred while the selected changes receive
final default-route, favorable support/tensor, dispatch and memory validation.
The [plan](2026-09-discrete-performance-plan.md) and
[roadmap](../ROADMAP.md) should continue to show an open performance gate.
