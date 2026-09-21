# Cheap-interactions integration

## Status and scope

Integration branch: `codex/cheap-interactions-integration`, draft PR #406.
The results in this report describe the candidate before external PR review.
Complete-fit measurements use `7369564aa8bf922d51be89868f15baa1aba27664`.
The locally verified source is `9f760e5abb72dbc16068c2571a065681fbe2fa39`.
The only later production change adds three public-warmup compilation calls;
fitting arithmetic is unchanged. The full-suite rerun passes after that
correction. Published PR head `4b94232044d8981d7ba9c6274bbd448539f7f9a6` adds
the report and measurements. Original worktrees and historical receipts remain
intact. The performance costs below still need an integration decision; a green
local correctness suite does not make them disappear or establish CI portability.

This integrates the retained tensor execution work, finishes the Gram repair,
corrects gap-table reporting, and preserves the saved interaction-compression
pilot outside production code. ADMM/OSQP experiments, scalar-family optimization,
cold-start work and further solver research are unchanged. The original candidate
has no dependency, version or public solver-selection change. Subsequent review
fixes and CI remediation are recorded separately below.

The comparisons have two different purposes:

- `99ca0eddadedde08386634c01fd92a314f8e8a18` is fetched `origin/master` and
  version 0.34.0. It measures the combined interaction work.
- `97ba3b00c41da5d6ef3856c20b232cb4cb138a9c` already contains both retained
  tensor acceleration rounds and the descent repair. It isolates the cost of
  finishing Gram correctness and simplifying REML.

These are not ADMM/OSQP comparisons. The historical 4.4x to 72x range is not a
promised speedup for every complete fit or thread setting.

## Production changes and maintainability

At the published pre-review head, `src/` changes ten files against master:
1,103 lines added, 242 removed,
861 net. The distribution is:

| Area | Net lines |
| --- | ---: |
| Group-matrix assembly and kernels | 637 |
| Tensor metadata and design-building plumbing | 86 |
| Discrete REML | 124 |
| Public API documentation | 14 |

The retained pre-Gram interaction branch already accounted for 663 net lines.
This integration adds another 198 net source lines. In particular, removing
the redundant tensor sum-coordinate constraint and the unusable interior
step-length schedule removes 64 net lines from `reml/discrete.py` relative to
that branch. Descent safeguards, true-objective checks, stationarity checks,
and the full halving budget remain.

Projected supports and sparse diagonals reuse the existing per-assembly cache.
The cache owns references to its groups and dies after the assembly. A new
call therefore rebuilds results after changes to weights, basis, transforms,
subsets or reparameterization. There is no persistent model-wide cache or new
solver framework.

Compressed supports project before moment accumulation. Ordinary sparse blocks
keep the raw sparse route unless the projection-error screen flags cancellation.
For flagged mixed-sign weights, a sparse absolute-weight companion distinguishes
small signed moments from actual basis cancellation. The screen selects arithmetic;
it does not certify all preceding accumulation error or change rank tolerances.
Exceptional cross-product factors retain their previous association. Sensitive
sparse cross-blocks use compatible projected rows or columns.

## Review and regression evidence

The fresh `integration_review_gpt_6_astra_max` review rejected the first assembled
candidate, `bab957b1`, for four issues: signed derivative fallback cost, missing
cross-product range guards, inconsistent sensitive sparse/dense crosses, and
retained int64 CSR index copies. Nineteen new tests reproduced those failures
before the follow-up repair. A narrow re-review at `7369564a` found all four
addressed, with no open critical or important findings in that scope. The
reviewer withheld final merge readiness pending full tests and fit acceptance.
The review is separate from performance acceptance.

The coordinator independently reran 403 focused numerical, dispatch, memory,
centering and tensor-step tests on the revised source. All passed in 9.60 seconds.
Whole-Gram tests check error, rank, subspaces and residuals. They distinguish
rounding in represented solver rows from moment-assembly error. Existing rank
and Gram error tolerances are unchanged.

Reduced-cap int64 cross tests measured peaks of 250,101 and 944,085 bytes at
256 KiB and 1 MiB caps. Restoring the copying CSR constructors gives 740,109
and 1,476,109 bytes and fails both tests. These are kernel allocation checks,
not whole-fit RSS measurements.

The Gamma tensor-step test now directly forbids entry into the known-scale
surrogate step. It no longer demands bit-identical flat-direction lambdas after
Gram reassociation. Gamma numerical accuracy is checked separately by real-data
parity and the saved before/after fit outputs below.

## Complete-fit acceptance protocol

The reproducible worker is
`benchmarks/benchmark_cheap_interactions_acceptance.py`. Each worker starts a new
process, imports the selected source explicitly, and uses the same interpreter
and installed numerical libraries. Eight numerical-thread environment variables
are pinned to one or four; receipts also record actual BLAS and Numba settings.
Before timing, each source fits the first 2,000 rows for two REML iterations.
This is a warmed comparison, not a cold-start study.

Timing includes model construction and the complete public `fit_reml` call.
Prediction and compression of output artifacts happen afterward. Receipts record
source and driver hashes, fixture hashes, convergence, iterations, objectives,
deviance, EDF, smoothing parameters, backend dispatch, CPU time and RSS. Fit RSS
is the fresh process high-water immediately after fitting, including imports
and warmup but excluding the later prediction phase. It is not an isolated
incremental allocation measurement.

Ordinary Gaussian, Poisson, near-cutoff lossless support, and ten-pair interaction
fits use 100,000 rows. Candidate/pre-Gram timings use three serial interleaved
repeats at each thread count. Master supplies one ten-pair reference per thread
count. A 300,000-row ten-pair check uses four threads and one run per source.
The Gamma comparison has 2,000 rows. Profiling runs are separate and excluded
from timing ratios. This is a bounded regression sample, not a complete family,
dataset, hardware or threading survey.

The first campaign was stopped after detecting a 1.426x ordinary Poisson
slowdown at both thread counts. Its receipts remain unchanged. Profiling found
24 dense solver-row Gram fallbacks. The revised one-thread profile has none.
Fresh Poisson medians are 1.02188 versus 0.96537 seconds at one thread, and
1.13902 versus 1.04233 seconds at four threads. This leaves measured overhead
of 5.9% and 9.3%; it is not literal performance parity.

The final campaign completed 70 of 71 workers, including four separate profiling
runs. Every completed worker converged. The campaign process returned a nonzero
status because of the single timeout below; it did not pass without qualification.

### Complete-fit times

These are medians in seconds. Except for the marked single runs, the pre-Gram
and candidate columns each have three runs. The percentage compares the candidate
with the already-accelerated pre-Gram branch, not with master.

| Fixture | Rows | Threads | Pre-Gram | Candidate | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian, k=10 | 100,000 | 1 | 0.2465 | 0.2506 | +1.7% |
| Gaussian, k=10 | 100,000 | 4 | 0.2643 | 0.2665 | +0.8% |
| Gaussian, k=50 | 100,000 | 1 | 0.3526 | 0.3617 | +2.6% |
| Gaussian, k=50 | 100,000 | 4 | 0.5025 | 0.5009 | -0.3% |
| Poisson, two k=30 terms | 100,000 | 1 | 0.9654 | 1.0219 | +5.9% |
| Poisson, two k=30 terms | 100,000 | 4 | 1.0423 | 1.1390 | +9.3% |
| Lossless support, k=10 | 100,000 | 1 | 0.2403 | 0.2738 | +14.0% |
| Lossless support, k=10 | 100,000 | 4 | 0.2519 | 0.2540 | +0.9% |
| Poisson, ten interaction pairs | 100,000 | 1 | 15.2816 | 15.6021 | +2.1% |
| Poisson, ten interaction pairs | 100,000 | 4 | 15.8667 | 15.6535 | -1.3% |
| Poisson, ten pairs, one run | 300,000 | 4 | 22.1882 | 21.0624 | -5.1% |
| Gamma, one pair, one run | 2,000 | 1 | 0.2132 | 0.2045 | -4.1% |

Master's 100,000-row ten-pair references took 62.1312 seconds with one thread and
48.8347 seconds with four. The candidate medians are therefore 3.98x and 3.12x
faster than master on this fixture. All these fits converged in 12 outer
iterations. At 300,000 rows, candidate and pre-Gram both converged in 11.

Four threads do not improve the candidate's wall time on this fixture:
15.6021 seconds becomes 15.6535 seconds. Median CPU time rises from 15.58 to
71.73 seconds. The runs changed BLAS and Numba thread limits together, so these
measurements do not identify which pool causes the additional CPU use. The raw
channel histogram itself is serial. Independent thread-pool measurements are
needed before attributing the result or recommending a threading default.

The near-cutoff support fixture has 38,000 distinct rows and actually uses
`SupportCompressedSSPGroupMatrix`. Its one-thread increase and the Poisson
increases remain costs of this candidate. These observations do not support
a blanket ordinary-fit performance-parity claim. Three repetitions on one
machine do not establish a universal percentage or a statistical confidence
interval. No dispatch thresholds were retuned to fit these measurements.
In particular, the small tensor differences and the single-run 300,000-row
difference do not establish a performance improvement or regression. They are
observations within a small, noisy sample.

The 300,000-row master worker exceeded its 240-second process limit. Its log and
timeout record are preserved. Since that limit covers startup, warmup, fitting
and output generation, it is not a complete-fit timing or a valid speedup lower
bound. No 300,000-row master timing, memory or output comparison is claimed.

### Outputs and memory

Against pre-Gram, the ordinary Gaussian and Poisson outputs are bit-identical on
the saved fixtures. Maximum prediction differences are below 9.6e-13 for support
and tensor fixtures; relative L2 differences are below 3.2e-13. Gamma's maximum
prediction difference is 3.91e-10, its relative L2 difference is 1.84e-11, and
its REML objective difference is 3.71e-9. Both Gamma fits converge in 12 iterations.

Against master, the 100,000-row tensor predictions differ by relative L2
1.70e-4, with maximum absolute difference 5.96e-4. The candidate's objective is
0.00676 higher, deviance 0.02944 lower and EDF 0.02675 higher. The combined branch
includes a different REML step, so this is not bit-identical optimization output.

Candidate ten-pair fit RSS is 948.0 MiB and 956.5 MiB at 100,000 rows with one
and four threads, and 1,170.8 MiB at 300,000 rows with four threads. Relative to
pre-Gram, these are -0.05%, -0.05%, and +0.13%. The largest measured relative
RSS increase is the four-thread lossless-support fixture, +3.6%, reaching
422.5 MiB. Other candidate/pre-Gram comparisons stay within about 0.2%.
These process high-water measurements do not replace the scratch-allocation
regressions above.

The exact timings, source hashes, thread-library reports, fixture hashes,
comparison statistics and SHA-256 manifest for all 70 receipts are in
`2026-09-20-cheap-interactions-integration-measurements.json`. Raw receipts,
full prediction/coefficient arrays and profiles remain in the ignored
`.benchmark-artifacts/cheap-interactions-integration/acceptance-final/` directory.
The benchmark worker is committed and can recreate each case with its pinned
source and thread settings.

## Benchmark reporting and saved research

The gap-table amendment excludes closure percentages without positive
additive-to-ceiling headroom, makes incomplete R3 evidence undecided, and preserves
worker/source identity through atomic pre-fit checkpoints and launcher fallback.
It refuses to overwrite existing arm receipts. See
`2026-09-20-gap-table-reporting-amendment.md` for the protocol change.

Derived summaries change Kaggle R3 from true to undecided and remove Ames's
invalid closure percentage. All 216 historical input files were rehashed and
remain unchanged. Missing historical source hashes were not invented. The
benchmark and saved compression tests passed 138 tests in 8.21 seconds.

The saved compression pilot is diagnostic research, not a new cheaper fitting
algorithm or a measured training-time improvement. Its source stays under
`benchmarks/`; its original measurements and figures remain under research notes.

## Final checks

The first full run had 15,208 passes, 91 skips and four failures. All 84 tests
in the three required real-data modules passed without a data skip. The failures
were diagnosed separately:

- Public warmup did not compile the new exponent scanner. Three calls now cover
  its writable/read-only layouts and the new absolute-weight CSR specialization.
  The subprocess test verifies that subsequent calls do not compile new layouts.
  The acceptance driver does not invoke public warmup, so no timed fit function
  changed after the recorded campaign.
- The short-cross memory test required unprojected support identity. It now
  checks the real paired-panel byte budget and complete row coverage, allowing
  either valid representation. Bypassing chunk sizing makes the revised test
  fail. Numerical cross checks remain separate.
- The Numba worker-budget test explicitly requests two threads and cannot run
  under `NUMBA_NUM_THREADS=1`. It passes unchanged with the maximum set to four.
  The repeat full run uses that maximum while keeping BLAS at one thread.
- The duration manifest fell just below its 95% collection threshold. It now
  includes 128 missing nodeids from changed tests, with durations measured by
  the completed run. The renamed panel test also receives its current name and
  measured duration; other existing entries are unchanged.

The focused import, histogram-dispatch and reuse-digest run passed 157 tests;
all 12 CI-contract tests also passed after these corrections.
A separate nine-test run enabled 16 Numba workers and four BLAS threads, covering
the paired-moment and BLAS-release checks that the capped full run skips.

The repeat full suite passed 15,218 tests across four shards, with no failures
or errors. Its 85 reported skips comprise 69 opt-in browser tests, four missing
R/mgcv checks, four cases where the exact category lookup needs no permutation,
four capped-thread checks covered by the separate run, and the missing-jupytext
notebook module reported once per shard. All 84 required real-data tests passed:
29 parity, seven screening-guide, and 48 mixed-interaction screening tests.
Python was 3.13.14; this local run is not a substitute for the multi-version CI
matrix. The end-to-end script also exited successfully.

- Focused numerical/dispatch/memory tests: 403 passed.
- Benchmark/compression tests: 138 passed.
- Ruff check and formatting: passed across `src/`, `tests/` and the affected
  benchmark modules; 799 files already formatted.
- `uv lock --check` and `uv pip check`: passed; 72 installed packages compatible.
- Full suite with required freMTPL2 data: 15,218 passed, 85 reported skips, no
  failures or errors. All four shard processes exited 0.
- `run_test.py`: exit 0, `END-TO-END COMPLETE`.
- Final follow-up code review: no open critical or important findings. The final
  warmup/test correction received a separate narrow check; its one metadata
  comment was addressed by updating the renamed test's duration entry.
- Complete-fit campaign: 70 completed and converged; master 300,000-row worker
  timed out. Residual ordinary-fit costs are disclosed above.

Advisory impact: `release:patch`. This is corrective numerical, performance and
memory work without a new public solver API. No version bump, tag or publication
is authorized or performed.

## PR review follow-up

External review of `4b942320` found defects not caught by the first local suite.
The original CI run also failed three numerical tests on Python 3.12/3.13,
exceeded the type-check budget, and reported three advisories against the
existing locked AnyIO version. The earlier local results above are historical
evidence, not a claim that the original PR passed CI.

The corrective source changes are bounded to the reviewed paths:

- Cell-CSR reuse validates current tensor indices and stable ordering with an
  allocation-free scan. Public index mutation and replacement remain supported.
  Public warmup includes the new validator.
- Integer and float32 support moments keep the original weight-promoted
  multiplication order. Raw-channel admission checks its actual raw factors
  and projection, including compensating extreme factors.
- Channel workspace admission counts the active grid's CSR arrays, retained
  scratch and permuted weights, construction/gather temporaries, and the
  contraction/map/result peak. A newly cached weight vector remains counted
  during contraction. Reusable buffers are released before growth or fallback
  when needed. This bounds the operation's array workspace, not whole-model
  retained storage or process RSS.
- Structured random-effect crosses reuse the diagonal's existing per-assembly
  cache instead of computing a sparse Gram only to discard it.
- The tensor REML quadratic pre-filter and its unused diagnostics are removed.
  For the legitimate damped direction, its quadratic predicts a decrease at
  every positive trial length up to one, so it cannot reject a trial. The
  25-trial budget, true-objective acceptance, descent and stationarity checks
  remain. Each feasible trial requires a cached solve and objective evaluation.

REML loses 30 production lines relative to the reviewed head. The Gram fixes,
typed contracts and cache handoff add 102 net lines, including shorter contract
documentation. Together the follow-up adds 72 net source lines. Against master
`99ca0edd`, the resulting source diff is 1,200 additions and 267 deletions across
11 files, net 933. No solver framework or OSQP dependency is added.

The additive regressions now check generic dispatch independently and compare
stable objectives and predictions with conditioning-scaled error bounds. They
do not require bit-identical flat-direction lambdas. Gamma dispatch retains
the original fixture; convergence has a separate curved-margin fixture whose
smoothing directions are determined. Its original 12-iteration budget is
unchanged. The benchmark driver's Gamma fixture is also unchanged.

The backtracking regression retains the legitimate damped step and adds a
controlled quartic objective term with zero gradient and Hessian at the current
candidate. Restoring the old five-trial limit makes acceptance fail. Real-fit
descent, stall and optimum regressions remain separate. Source regressions
first reproduced stale indices, integer/float32 errors, exceptional raw factors,
workspace undercounting and duplicate sparse-Gram work. A second regression
caught a newly cached weight vector omitted from the first workspace repair.

Independent review found no production blocker in the corrective diff. Its
minor test finding was addressed by giving fused vector assertions explicit
binary64-epsilon tolerances; the old default allowed the float32 error the
fixture was meant to catch. The signed-loop duplication, redundant xtw-only
fusion, and own-margin identity-cache concerns were traced to master and left
for separate work. They are not silently treated as fixed by this PR.

AnyIO alone moves from 4.14.1 to 4.14.2 in `uv.lock`. The hash-checked audit of
all locked extras/groups then reports no known vulnerabilities. Project
dependency declarations and the package's own version remain unchanged.

The corrected source's type check reports 846 diagnostics, compared with 890
on master under the same checker/environment and 910 on the reviewed head.
The accepted CI ceiling remains 903. This is a passing backlog gate, not a
claim of a clean type-checking baseline.

Python 3.12 and 3.14 each pass 566 affected tests. After the review's stricter
assertions, the complete Gram-regression module passes another 22 tests on each
of Python 3.12, 3.13 and 3.14. Benchmark/reporting contracts pass 245 tests, and
the separate high-thread follow-up passes nine. Ruff, formatting, lock checks,
dependency consistency and the end-to-end smoke test pass.

The combined Python 3.13 full suite passes 15,251 tests, with 85 reported skips
and no failures or errors across four shard processes, all exiting zero. All
84 required freMTPL2 tests pass. The source tested and measured for this follow-up
is `52d330baaac96e39b07080fb1a7c8292098c6bec`, with source-tree SHA-256
`4238856cd27722bfd9823a8074d36388aaff8c3743247babd0ac19d643fd7746`.
The stricter assertion-only follow-up was separately rerun on all three Python
versions after review; production source did not change.

### Fresh complete-fit results after review

All 74 workers complete and converge: 72 unprofiled fits and two separate
profiles. The campaign exits zero. Source and fixture hashes remain stable.
The committed worker is unchanged. Timing still includes construction and the
complete fit after the same 2,000-row, two-iteration warmup. No other test jobs
run during timing. Candidate/pre-Gram variants use three serial, rotated
repetitions; master references, the 300,000-row check and Gamma use one run.
The earlier master timeout is not repeated or converted into a speedup bound.

The following matched-thread medians are seconds. Percentages compare the whole
correctness integration with pre-Gram `97ba3b00`, not just the review fixes.
The original PR head was not rerun as a separate control in this campaign.

| Fixture | Rows | BLAS/Numba threads | Pre-Gram | Candidate | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian, k=10 | 100,000 | 1/1 | 0.2677 | 0.2612 | -2.4% |
| Gaussian, k=10 | 100,000 | 4/4 | 0.3007 | 0.2734 | -9.1% |
| Gaussian, k=50 | 100,000 | 1/1 | 0.3694 | 0.3903 | +5.6% |
| Gaussian, k=50 | 100,000 | 4/4 | 0.5107 | 0.4922 | -3.6% |
| Poisson, two k=30 terms | 100,000 | 1/1 | 1.0064 | 1.1361 | +12.9% |
| Poisson, two k=30 terms | 100,000 | 4/4 | 1.0624 | 1.2090 | +13.8% |
| Lossless support, k=10 | 100,000 | 1/1 | 0.2747 | 0.2500 | -9.0% |
| Lossless support, k=10 | 100,000 | 4/4 | 0.2769 | 0.2683 | -3.1% |
| Poisson, ten pairs | 100,000 | 1/1 | 16.4005 | 17.9097 | +9.2% |
| Poisson, ten pairs | 100,000 | 4/4 | 15.5702 | 17.8764 | +14.8% |
| Poisson, ten pairs, single run | 300,000 | 4/4 | 22.3477 | 24.0599 | +7.7% |
| Gamma, one pair, single run | 2,000 | 1/1 | 0.1776 | 0.1950 | +9.8% |

Fresh master references take 61.3678 and 49.6566 seconds at 1/1 and 4/4.
The candidate is therefore 3.43x and 2.78x faster than master on this fixture.
That does not cancel the measured cost against the already-accelerated branch.
All 100,000-row tensor fits take 12 outer iterations; the larger pair takes 11.
The Gaussian/support timing ranges overlap, so their percentages do not
establish improvements. This remains a small sample on one host, not a general
performance guarantee. The individual samples are saved, not just the medians.

Candidate tensor fit RSS is 949.4/957.4 MiB at 100,000 rows with 1/1 and 4/4
threads, about 0.13%/0.10% above pre-Gram. The 300,000-row candidate reaches
1,161.1 MiB, about 0.74% below its single pre-Gram reference. The largest
ordinary-fit RSS increase is again the four-thread support fixture, 3.6%.
These are process high-water measurements, not exact live-allocation totals.

Gaussian and ordinary Poisson predictions remain bit-identical to pre-Gram.
Support/tensor prediction differences are below 9.6e-13. Gamma's maximum
difference is 3.91e-10 and relative L2 difference 1.84e-11. Against master,
tensor relative L2 difference remains about 1.70e-4; the combined REML change
is still not a bit-identical optimizer. Repeated candidate predictions are
identical within each setting. Changing BLAS thread limits changes tensor
predictions by at most 7.17e-13; changing Numba/OMP at fixed BLAS leaves them
identical.

### Independent thread limits

These are candidate medians for the 100,000-row ten-pair fit, with three runs
per cell. BLAS limits apply per library, not to total process CPU usage.
Receipts verify actual BLAS and Numba settings. OMP follows the Numba limit;
NumExpr stays at one thread.

| BLAS limit | Numba limit | Wall seconds | CPU seconds |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 17.9097 | 17.8848 |
| 1 | 4 | 18.1864 | 18.1654 |
| 4 | 1 | 16.7810 | 76.3829 |
| 4 | 4 | 17.8764 | 80.9019 |

The extra CPU follows the BLAS setting in this experiment. The higher Numba/OMP
limit does not supply a measured speedup here. BLAS 4/Numba 1 is about 6.3% faster
than 1/1 by the medians, while consuming over four times the CPU. Using four
for both barely changes median wall time. This does not establish why each
library thread is busy, or justify a universal threading default.

The separate one-thread profile records 945 channel contractions and 945
cell-CSR requests, confirming the raw-channel route remains active. Cell-CSR
checking/building takes 1.01 seconds cumulatively; cross-factor range checks
take 0.60 seconds across 2,835 calls. Workspace accounting itself takes 0.020
seconds across 945 calls. These are profiled costs, excluded from timing ratios.
Validating once within the existing weighted-assembly cache is a concrete
follow-up candidate, provided mutation between assemblies stays observable.
No such extra optimization or threading-policy change is included here.

The numerical results and CI repairs do not establish performance acceptance.
The PR remains a draft pending that decision and external re-review. The new
measurements, source identities, sample times, numerical comparisons, thread
libraries and hash manifest for all 74 receipts are recorded in
`2026-09-21-pr406-review-measurements.json`. Raw receipts and arrays stay under
the ignored `pr406-acceptance/` artifact directory. Earlier receipts and
measurements remain unchanged.

## Second review response

Claude and Codex reviewed `9609b0e7`. The hosted Python 3.12, 3.13 and 3.14
regression suites, real-data workflow, type budget and dependency audit all
passed on that head. Codex found a remaining retained-buffer overlap before
row fallback. Claude identified repeated scans and several smaller issues.
The second repair is `328373d4c0fbf3b424793d098a3e31a48bf036d4`.

- Channel scratch and permuted weights are now released before every channel
  decline reaches fallback, including early aggregate, dtype and range gates.
  The same overlap was reproduced on admitted same-id and shared-margin tensor
  routes and repaired there too. Shared-margin declines preserve the channel
  cache. This does not impose a universal workspace or RSS cap on older routes.
- Weight-range decisions and validated grid orders are reused inside the
  existing synchronous assembly cache. Entries retain their input/model owners;
  they do not retain aliases to released channel scratch. Both original weight
  predicates remain separate, preserving their inclusive power-of-two endpoints.
  New assemblies and uncached calls still observe mutations and replacements.
- An unaffordable raw stage no longer evicts buffers if eviction cannot admit
  it. The following dense stage may fit with those buffers retained.
- Claude's N2 claim that CSR `fill` was unbudgeted was not borne out by the code.
  `retained_index` includes pointers and row order; `stage1` separately reserves
  `8 * cells` for `fill`, and admission adds `max(stage1, stage2)` to the base.
  Adding that term again would double-count it. No accounting change was made
  for N2.
- N4 was valid. Conditioning of the coefficient Hessian does not certify a
  free-running outer optimizer's stopping error. The free additive fit now tests
  convergence and generic dispatch. Historical numerical comparisons remain in
  the three-iteration fixture, described as a conditioning-scaled regression
  allowance, not an optimizer error certificate. No exact free-iteration count
  or arbitrary tolerance padding was added. Two stale duration-cache names
  were corrected.
- N5's one-element loop was simplified.

The production repair changes one existing algebra file, adding 78 lines and
removing 14, net 64. Across the PR, source changes against master `99ca0edd` are
1,269 additions and 272 deletions across 11 files, net 997. There is no new
solver, public API, dependency, precision policy or threading default.

Eight regression cases failed before their repairs. At a reduced 262,144-byte
budget with 32,768 bytes of tracing allowance, the early-decline cases peaked
at about 432,900 bytes; admitted shared-margin and same-id cases reached
360,257 and 325,992 bytes. All pass after release. Separate dispatch checks
verify that four cross blocks over two grids use one legacy weight scan, one
exponent scan and two cell validations with the cache, versus four of each
without it. This is a reuse count, not a complete-fit speed claim.

Independent read-only review found no further correctness or maintainability
issue in the final four-file repair. Parent verification passed 15,266 tests
with 85 reported skips and no failures or errors across all four full-suite
shards. All 84 required real-data tests ran and passed. Python 3.12 and 3.14
each passed the affected 581-test selection. The separate 474-test numerical
selection, nine high-thread checks and end-to-end script also passed. Ruff,
formatting, lock and dependency checks passed. Type diagnostics remain 846,
below the unchanged budget of 903. Source-tree SHA-256 is
`2a767b68109d0a85f7b23db39fb84e9bfa9f3b9113de6ecc1bc73b3270cde58d`.

### Second review complete-fit measurements

All 78 fresh workers completed and converged: 74 unprofiled fits and four
separate profiles. The candidate is the exact source at `328373d4`; source
hashes match before and after every fit. The committed benchmark worker is
unchanged. Each worker runs the same 2,000-row, two-iteration warmup before
timing construction and the complete fit. No numerical test jobs ran alongside
the campaign. Receipts verify actual backend and thread settings.

The 100,000-row tensor fixture has three rotated serial repetitions per variant
and setting; ordinary Poisson has five. Ordinary Poisson also includes the
exact first PR head `4b942320`, not an old measurement of it. Medians below are
seconds. The controls are pre-Gram `97ba3b00`, first PR `4b942320` and reviewed
head `9609b0e7`.

| Fixture | BLAS/Numba | Pre-Gram | First PR | Reviewed head | Candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Poisson, two k=30 terms | 1/1 | 0.9982 | 1.1322 | 1.1574 | 1.1030 |
| Poisson, two k=30 terms | 4/4 | 1.1691 | 1.1813 | 1.2648 | 1.1108 |
| Poisson, ten pairs | 1/1 | 16.8144 | — | 20.0856 | 18.6687 |
| Poisson, ten pairs | 4/4 | 16.5688 | — | 17.3512 | 17.0831 |

Tensor medians are 7.1% and 1.5% lower than the reviewed head at 1/1 and 4/4,
but remain 11.0% and 3.1% above pre-Gram. Samples overlap substantially: at 1/1,
the reviewed-head tensor runs span 17.15–21.69 seconds and candidate runs span
17.80–18.78 seconds. These are small samples on one host, not guaranteed gains
or performance acceptance. No fresh master control was run; the earlier
3.43x/2.78x master comparisons remain historical and are not recomputed using
these new candidate times.

Ordinary Poisson remains 10.5% above pre-Gram at one thread by the medians. At
four threads its median is 5.0% below, but overlapping variable samples do not
establish parity or a general improvement. The exact-head controls do not
establish the previously suggested doubling of the ordinary-Poisson regression
as a source-caused change. The tensor cache does not run on this fixture, so
the lower candidate times must not be credited to that optimization. Profiles
retain 70 projected Gram checks and 35 sparse/SSP crosses on both heads.

The 300,000-row, four-thread tensor pair takes 24.3374 seconds on the reviewed
head and 22.1186 seconds on the candidate. The 2,000-row Gamma pair takes
0.2152 and 0.2001 seconds. These and the Gaussian/support fixtures are single
pairs, useful for dispatch, numerical and memory checks rather than precise
speed estimates. Every measured fit selects the Gram backend.

Candidate predictions, coefficients, objective and EDF are bit-identical to
the reviewed head in all saved fixture/setting comparisons. Repeated candidate
predictions are identical within each setting. Against pre-Gram, tensor
prediction differences are below 8.9e-13 in maximum absolute value and 3.2e-13
in relative L2 norm. All 100,000-row tensor fits take 12 outer iterations;
ordinary Poisson takes six, and the larger tensor pair takes 11.

Candidate tensor peak RSS is 948.7/956.8 MiB at 100,000 rows, versus
949.3/957.0 MiB on the reviewed head and about 0.08% above pre-Gram. At 300,000
rows it is 1,161.7 versus 1,158.7 MiB. These remain process high-water readings,
not a whole-model memory guarantee. Candidate tensor CPU medians are 18.65 and
78.09 seconds at 1/1 and 4/4: extra BLAS threads still consume much more CPU.

The separate one-thread tensor profiles confirm the intended reuse. Model
cell-order validation/build calls fall from 945 to 189, and their cumulative
time falls from 1.055 to 0.194 seconds. The new cache method is still called
945 times; its cumulative time includes those 189 model calls and must not be
added to them. Raw-channel contractions remain 945 on both heads. The unit
call-count regression separately proves weight-scan reuse; the profile does
not expose those compiled scanners as direct entries. No arithmetic or backend
change was needed for this reuse.

The [second-review measurements](2026-09-21-pr406-rereview-measurements.json)
record all individual timing samples, CPU and RSS, numerical comparisons,
source identities, thread libraries, filtered profiles and hashes for all 78
receipts. Raw receipts and arrays remain in the ignored `pr406-rereview-fits/`
artifact directory. Earlier evidence is unchanged. The PR remains a draft:
these results address the review findings but do not accept the remaining
performance cost or replace hosted checks on the pushed head.
