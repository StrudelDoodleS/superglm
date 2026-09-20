# Cheap-interactions integration

## Status and scope

Local integration branch: `codex/cheap-interactions-integration`.
Complete-fit measurements use `7369564aa8bf922d51be89868f15baa1aba27664`.
The final verified source is `9f760e5abb72dbc16068c2571a065681fbe2fa39`.
The only later production change adds three public-warmup compilation calls;
fitting arithmetic is unchanged. The full-suite rerun passes after that
correction. Nothing has been pushed or merged. Original worktrees and historical
receipts remain intact. The performance costs below still need an integration
decision; a green correctness suite does not make them disappear.

This integrates the retained tensor execution work, finishes the Gram repair,
corrects gap-table reporting, and preserves the saved interaction-compression
pilot outside production code. ADMM/OSQP experiments, scalar-family optimization,
cold-start work and further solver research are unchanged. No dependency,
version or public solver-selection change is included.

The comparisons have two different purposes:

- `99ca0eddadedde08386634c01fd92a314f8e8a18` is fetched `origin/master` and
  version 0.34.0. It measures the combined interaction work.
- `97ba3b00c41da5d6ef3856c20b232cb4cb138a9c` already contains both retained
  tensor acceleration rounds and the descent repair. It isolates the cost of
  finishing Gram correctness and simplifying REML.

These are not ADMM/OSQP comparisons. The historical 4.4x to 72x range is not a
promised speedup for every complete fit or thread setting.

## Production changes and maintainability

Against master, `src/` changes ten files: 1,103 lines added, 242 removed,
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

The near-cutoff support fixture has 38,000 distinct rows and actually uses
`SupportCompressedSSPGroupMatrix`. Its one-thread increase and the Poisson
increases remain costs of this candidate. These observations do not support
a blanket ordinary-fit performance-parity claim. Three repetitions on one
machine do not establish a universal percentage or a statistical confidence
interval. No dispatch thresholds were retuned to fit these measurements.

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
