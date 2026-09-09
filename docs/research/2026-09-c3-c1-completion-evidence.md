# C3 and C1 completion evidence

This report compares the C1 implementation with frozen `origin/master`
`8962c4520cad948aa20c480a238b7bb1e276e9cd` (v0.31.0). It records the
supported execution path and the limits of the available complete-fit evidence.
The C3 diagnosis and independent references are in
[C3 stress evidence](2026-09-c3-stress-evidence.md). That work resolves the two
named stress cases through existing strict EFS plus Newton controls, while
preserving the unsuccessful EFS trajectories and unresolved reference probes.
It does not introduce a new optimizer or a GPD shape penalty.

Selected numerical summaries, timing audits and raw artifact checksums are
collected in the tracked [complete-fit receipt](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_c1_complete_fit_receipt.json).

The [pragmatic convergence follow-through](2026-09-pragmatic-convergence.md)
corrects the diagnosis of the NB2 stop recorded below. The old Newton recovery
could stop after an improving accepted fit and report stale derivative evidence.
The corrected finite-NB2 route reaches configured stationarity on the same book. Historical receipts
here remain unchanged; their rejected smoothing status did not establish that
the fitted model was unusable.

## Production changes

Public `SuperLSS(discrete=True)` now selects the existing grouped, chunked
execution path. Observed-curvature fitting accepts families without expected
information, including Tweedie and NB2. An explicit Fisher request still
requires that capability. Existing signed cross-predictor assembly, likelihood
weight contracts, coefficient safeguards, terminal curvature decisions and
smoothing stationarity rules remain authoritative.

The private [row-design adapter](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/src/superglm/distributional/_row_design.py)
expands bounded row blocks for smoothing derivatives. Newton, endpoint-related
validation and optional smoothing-corrected posterior replay can use this
adapter without retaining a full observation-by-coefficient predictor matrix.
Derivative cross-products and coefficient-direction contractions use bounded
row scratch. Ordinary conditional covariance already uses coefficient-space
matrices; it does not require a full dense training design.

Categorical spline row subsets now own immutable category indices and lazily
reuse a sorted lookup. Repeated small subsets no longer sort and copy all
parent category rows on every chunk. Restoration checks cover older serialized
objects and subclass metadata. The row-cache kernel evidence establishes that
specific allocation/work change; it is not a complete-fit speed claim.

Focused coverage is in
[public discrete tests](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/tests/test_superlss_discrete.py),
[bounded derivative tests](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/tests/test_distributional_bounded_derivatives.py)
and [categorical row-subset tests](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/tests/test_spline_categorical_row_subset.py).
These cover observed-family support, signed dispatch, weight semantics,
interactions, smoothing and serialization, same-design derivative agreement,
full-design materialization guards, and repeated-subset allocation behavior.

## Complete-fit receipts and provenance

The [complete-fit harness](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_c1_complete_fit.py) and its
[reproduction instructions](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_c1_complete_fit.md) describe
fixtures, thread controls, instrumentation and source binding. The summaries
below are in ignored `.benchmark-artifacts/c3-c1/`. Each identifies raw worker
JSON/NPZ receipts and hashes. A candidate Git SHA alone is insufficient for a
working-tree run: the source-tree and diff hashes bind the actual implementation.
The production changes are committed at `9277baef`; the production tree is
unchanged from the source-bound candidate receipts described here.

Only directly written raw worker evidence supports these results. Headroom or
Kompress summaries, transformed tool stdout and proxy clocks do not establish
numerical results or elapsed time. The initial numerical receipt clocks are explicitly **unmeasured** because
numerical probes and repository tests overlapped. Whole worker process RSS
includes imports, fixture preparation and retained histories; it is not a
measurement of solver scratch alone. Each initial comparison below contains
one fresh worker per arm. The later repeated timing series is reported separately.

All three comparisons below use 92 coefficients, 12 knots, one configured
numerical thread and strict EFS plus Newton smoothing. Baseline dispatch is
`distributional-dense-v1`; the discrete candidate reports
`distributional-chunked-v1`.

| Fixture and summary | Training rows | Smoothing outcome | Baseline / candidate peak RSS, bytes | Interpretation |
| --- | ---: | --- | ---: | --- |
| Real log-severity, `severity-k12-summary.json` | 22,450 | Both certified `stationary`; 11 smoothing iterations, 12 coefficient fits | 585,576,448 / 608,890,880 | No RSS saving on this smaller book. |
| Real log-severity rows replicated 20 times, `severity-k12-replicate20-summary.json` | 449,000 | Both certified `stationary`; 7 smoothing iterations, 9 coefficient fits | 1,588,465,664 / 1,126,014,976 | 29.1% lower complete-worker peak RSS in this comparison. |
| Real NB2 frequency, `nb2-k12-summary.json` | 610,212 | Both `objective_rejected`; 8 smoothing iterations, 9 coefficient fits | 2,098,151,424 / 1,608,032,256 | Stopping-result parity only; not positive evidence of a reliably solved smoothing fit. |

The severity holdout contains 2,494 policies. Replication occurs after the
holdout split: it adds no independent real observations and changes the
likelihood information content. The two replicated arms use the same
replication factor. Their maximum absolute coefficient difference is
`2.69e-12`, holdout natural-parameter difference `6.55e-13`, covariance relative
L2 difference `2.97e-12`, and maximum absolute log-lambda difference `2.67e-11`.
These are measured comparisons, not universal numerical acceptance tolerances.

The unreplicated severity instrumentation separately verifies chunked dispatch
and design expansions of at most 4,096 rows. Its whole-process RSS is not
substituted for the uninstrumented comparison. The NB2 numerical agreement and
lower observed RSS do not certify its smoothing optimum or smoothing-selected
uncertainty; both arms preserve the same refusal.

## Representation and grid sensitivity

Grouped products are checked against dense expansion of the **same compiled
design**. This checks execution algebra independently of approximation error.
For the real severity fixture, the three numerical supports contain 73, 21 and
82 values, below 256 bins. The design and penalty comparisons agree within the
recorded dimension-scaled tolerances; they are not byte-identical after basis
reparameterization.

That explicit saved compiled-design comparison belongs to the smaller severity
run. The large summaries establish equal inputs and coefficient names together
with numerical agreement. The replicated severity fixture preserves the same
covariate support, so its exact-support representation is an inference from the
fixture and compiler behavior, not a separately saved large-design comparison.

For continuous covariates exceeding the bin count, equal-width bin centers
change the design. A finer-grid comparison measures that approximation and
cannot be justified solely by machine-epsilon tolerances.

`continuous-gaussian-n5000-summary.json` compares a 5,000-row continuous
Gaussian fixture, 2,000 holdout rows and 128 coefficients across dense and
64/256/1024-bin fits. All four runs pass the existing stationarity contract in
six smoothing iterations. Held-out natural-parameter discrepancies decrease
with grid refinement in this fixture:

| Bins | Location RMS difference from dense | Scale RMS difference from dense | Negative-LAML difference from dense |
| ---: | ---: | ---: | ---: |
| 64 | 0.00381683 | 0.00218063 | +0.0390252 |
| 256 | 0.000820516 | 0.000643388 | -0.282976 |
| 1024 | 0.000215737 | 0.000135483 | -0.0302105 |

These are absolute RMS differences in natural-parameter units. The objective
does not vary monotonically with grid size. The fits use changed designs, so
the smaller prediction differences are quantization-sensitivity evidence,
not representation roundoff. Coefficients, smoothing parameters and covariance
arrays remain in the raw receipts; unaligned coefficient-space differences
are not used as coordinate-invariant grid-error measures.

`factor-smooth-n2000-support32-summary.json` covers Gaussian location and scale
with categorical-by-spline effects, 2,000 rows, 32 support values, four levels
and 56 coefficients. Both baseline and candidate pass stationarity after
19 smoothing iterations and 22 coefficient fits. Inputs and coefficient names
match. The maximum held-out natural-parameter difference is `6.05e-13`, maximum
absolute covariance difference `7.63e-14`, and maximum absolute log-lambda
difference `8.51e-10`.

The candidate factor fit is instrumented: it records 74 chunked geometry calls
and 1,476 discretized categorical-spline row-subset calls. This establishes
actual dispatch on a complete smoothing fit. Its instrumentation overhead makes
this pair ineligible for a factor-fit RSS or wall-time comparison. It also
does not replace the separate row-cache regression and kernel evidence.

The later uninstrumented factor comparison,
`factor-smooth-n2000-support32-uninstrumented-summary.json`, reuses that baseline
and runs one fresh candidate with the same production source-tree hash as the
instrumented replica. Both pass stationarity and preserve the numerical
agreement above. Whole-process peaks are 370,515,968 baseline bytes and
416,325,632 candidate bytes: this small case does not show a memory benefit.
These observed process peaks remain separate from the kernel allocation result.

## Accepted local complete-fit timing

The first six-run series, `severity-k12-replicate20-timing-summary.json`, retains
valid raw observations but is **excluded from quiet-window speed claims**.
Headroom activity rose from a quiet preflight to as much as 0.676 CPU core
averaged over a fit. Low load and a worker consuming nearly one core did not
exclude cache or memory-bandwidth interference.

The repeat, `severity-k12-replicate20-timing-idle-summary.json`, paused agent
work and root tool traffic during the fits. Each arm ran in three fresh
processes, one numerical thread, without warmup or instrumentation, in the
serial order baseline/candidate/candidate/baseline/baseline/candidate.
The fixture is the same **449,000 synthetic training rows from 22,450 real
severity policies**, with the original 2,494-policy holdout and 92 coefficients.

| Arm | Median public fit seconds [range] | Median whole-process peak RSS |
| --- | ---: | ---: |
| v0.31.0 dense baseline | 17.427 [17.076–17.844] | 1,514.84 MiB |
| Current discrete, 256 bins | 14.750 [14.098–15.320] | 1,015.93 MiB |

Median complete-fit time is **15.36% lower** and median process peak RSS is
**32.93% lower** in this local comparison. The clock surrounds public
`fit_reml`, including its compilation, smoothing iterations and terminal
checks; interpreter startup, data preparation and the subsequent held-out
comparison are outside it. Process RSS includes their resident allocations.

Independent review verified raw JSON/NPZ/log checksums, serial intervals,
identical inputs and numerical controls, stable sources and unchanged
within-arm output hashes. Every fit passes the existing stationarity checks
after seven smoothing iterations and nine coefficient fits. Candidate source
`9277baef` has production tree SHA256
`ee937668f83c46d8e6d0990f4d0781186a2d98531000afbb564eecd687b377af`,
matching the earlier candidate receipts.

During each fit, measured Headroom activity was 0.0029–0.0057 CPU core.
Other matched-process activity, excluding the worker, totaled 0.137–0.159 core;
the largest individual background process stayed below 0.030 core. The worker
used 0.993–0.998 core, and one-minute load was 0.742–1.097 on 16 available CPUs.
These observations support this qualified local comparison; they do not
establish a universal speedup. Heuristic process-category labels are audited
by PID so descriptive arguments cannot count the worker itself as a proxy.

## Validation and remaining limits

The integration run reports **11,135 passed, 470 skipped** for the full test
suite. The three real-data suites, run separately with
`SUPERGLM_REQUIRE_DATA=1`, report **84 passed, 7 warnings**. Skips remain skips;
the full-suite count does not claim that every optional suite ran. Ruff lint
and formatting checks pass, with 669 files already formatted. `uv lock --check`,
`uv pip check` and `uv run python run_test.py` pass. The last command's U-shape
`CHECK` diagnostic is identical on the baseline; it is not presented as a new
passing shape result.

The implementation removes full training-design storage from the promoted
chunked route, but is not constant-memory distributional fitting. Family
stencils and derivative channels still retain `O(n K² C)` values, where `K`
is the predictor count and `C = K(K+1)/2` the packed curvature channel count.
Retained coefficient-fit histories store full-row `eta` and `theta`, so their
memory grows with the number of retained fits. Coefficient factors remain
dense; coefficient-space derivative matrices and smoothing-pair directions
also consume memory. Exact support compression can itself be large when
support cardinality is high.

The results establish a bounded-design production route and a measured memory
benefit on one 449,000-row replicated-real-data workload. They do not establish
10-million-to-100-million-row scalability or a universal speed/RSS improvement.
“Certified” refers to the existing local stationarity contract and its
objective-scaled stopping bar. It is neither a rigorous enclosure of every
source of numerical error nor a global-minimum guarantee.
