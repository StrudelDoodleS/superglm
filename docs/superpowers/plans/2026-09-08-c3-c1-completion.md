# C3 + C1 Completion Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Track
> completed steps here; workers own disjoint files and preserve others' edits.

**Goal:** Complete the audited C3 convergence follow-through and production C1
grouped/discrete distributional capabilities with reproducible insurance evidence.

**Architecture:** Extend the existing grouped designs and coupled solver. Preserve
the current certification/penalty-face authorities. Bound design allocations in
derivative and inference replay instead of introducing a parallel optimizer.

**Tech Stack:** Python 3.12+, NumPy, SciPy, Numba, tabmat, pytest, uv.

**Spec:** [Design](../specs/2026-09-08-c3-c1-completion-design.md)

## Global constraints

Use apply_patch for edits. Preserve the source dossier worktree, unrelated changes,
statistical model and version files. Mathematical tests and performance/dispatch
tests are separate. No release action is authorized. The root coordinates serial
timing; workers may run focused correctness checks while developing.

## Stage 1: Establish facts and preserve baseline

- [x] Fetch origin/master; verify GitHub/PyPI v0.31.0 and exact baseline SHA.
- [x] Create `.worktrees/c3-c1-completion` and a detached frozen baseline worktree.
- [x] Read root/revised AGENTS, roadmap and dossier; copy without changing sources.
- [x] Audit public refusal, observed-chunk capability, grouped derivative gaps,
  existing C3 safeguards and original #376 workload provenance.
- [x] Install `uv sync --python 3.13 --extra dev`; fetch verified freMTPL2 data.
- [x] Finish baseline non-slow suite and classify any existing failures:
  frozen v0.31.0 passes 10,913 tests, with 470 optional/browser skips and 142
  slow tests deselected. The inherited PYTEST_ADDOPTS issue was environmental.

## Stage 2: Promote observed grouped fitting

**Ownership:** solver/solver.py, solver/chunks.py, api.py; public/chunk regression
tests. Coordinate any shared helper with the derivative worker.

- [x] First demonstrate the public discrete refusal and the observed Tweedie/NB2
  chunk rejection with focused failing tests.
- [x] Enable public discrete fitting with bounded automatic row chunks; retain
  explicit Fisher capability refusals and correct fallback semantics.
- [x] Prove same-design score/curvature, fixed-fit and complete smoothing
  equivalence, signed cross-blocks, weights/offsets, and actual grouped dispatch.
- [x] Validate API cloning, refits, serialization, inference and prediction.
  165 focused tests passed. Independent review found no blocking issue; removed
  its stale capability docstring. Actual backend/chunk fields are public telemetry.

Run targeted public, grouped assembly, chunking, memory and family tests.

## Stage 3: Bound certification and uncertainty design allocation

**Ownership:** smoothing/derivatives.py, smoothing/newton.py, posterior.py and new
focused derivative/allocation tests. Endpoint LAML already uses grouped matvec;
change it only if a regression exposes a gap.

- [x] Add a failing no-full-design allocation regression around grouped exact
  LAML gradient/Hessian and certified Newton completion.
- [x] Introduce a shared bounded design adapter/operations and remove forced
  dense predictor matrices in derivative and posterior smoothing replay.
- [x] Compare gradient/Hessian certificates, objective, covariance, finite and
  infinity-face behavior against identical dense representation.
- [x] Validate smoothing uncertainty and adversarial derivative/refusal tests.
  106 focused tests pass. Independent mathematical review found no actionable
  issue and independently passed all 17 new checks. Baseline and two process-local
  mutations demonstrate that allocation regressions detect restored dense paths.

## Stage 4: Remove repeated full-row work from chunk extraction

**Ownership:** underlying group matrix row-subset implementations and their tests.

- [x] Demonstrate repeated full-parent sorting in factor-by-smooth chunk access.
- [x] Reuse immutable row-order lookup or a contiguous slicing route while
  preserving arbitrary subset/repeated-row and original row-order semantics.
- [x] Test exact matrix reconstruction separately from allocation/dispatch work.
  417 initial focused tests passed; old v0.31 matrix pickles restore correctly.
  Independent review caught dropped subclass dictionaries during restoration;
  six failing regressions now pass and scoped re-review is clean.
- [x] Benchmark a complete categorical/smooth distributional fit against baseline.
  Both 2,000-row fits pass stationarity with numerical parity and actual discrete
  categorical-spline dispatch. The uninstrumented candidate uses more process
  RSS at this size; retain that result separately from the row-cache kernel gain.

## Stage 5: Resolve the named C3 cases

**Ownership:** C3 regression tests and solver/smoothing fixes identified by exact
reproduction; coordinate modifications with Stages 2–3 before editing.

- [x] Run the recovered correlated Tweedie and GPD fixtures on frozen baseline.
- [x] Classify inner versus outer failure, finite stationarity versus cap/face
  decisions, curvature/rank evidence, and sensitivity to starts/optimizer choice.
- [x] Add focused mathematical regressions and minimal justified corrections.
  Both exact fixtures certify using the existing strict Newton route; no solver
  correction or relaxed authority is justified. Two mathematical authority
  regressions pass baseline/current and reject forged projected-score evidence.
- [x] Repeat on the implementation; retain an explicit explained limitation for
  any case that cannot be certified without changing the statistical model.

## Stage 6: Reproducible practical evidence and integration

**Ownership:** benchmarks/c3_c1_complete_fit.py (and fixture helper), result JSON
and report; documentation/roadmap owned by root.

- [x] Implement a public-API subprocess harness with baseline/current source
  selection, pure recovered fixtures and real freMTPL2 workloads.
- [x] Record serial complete-fit timing and whole-process peak RSS; use a separate
  instrumented replica for work/dispatch and save numerical artifacts.
- [x] Separate same-design exact grouped equivalence from finite-grid sensitivity.
- [x] Establish practical reliable insurance fitting plus a meaningful memory,
  feasible-size or fitting/validation-time improvement; state limits honestly.
- [x] Run focused tests, full pytest, Ruff check/format, lock/environment checks,
  run_test.py, and the three `SUPERGLM_REQUIRE_DATA=1` real-data suites.
- [x] Obtain independent mathematical/code review; fix substantiated findings.
- [x] Update roadmap, design/plan facts and benchmark report; verify source hashes.

## Evidence log

- Starting baseline: `8962c4520cad948aa20c480a238b7bb1e276e9cd` / v0.31.0.
- The dossier's new-optimizer/new-assembler assumptions are stale; both already
  exist. Its GPD cap implies unbounded likelihood claim is not established.
- Retained coefficient-fit histories contain row parameter arrays needed for
  certification. Preserve them and report their memory scaling explicitly.
- `docs/ROADMAP.md` is ignored by the existing generic ROADMAP.md rule; explicitly
  include the scoped updated roadmap when committing this work.
- Baseline environment: inherited `PYTEST_ADDOPTS=-q` broke three output-parsing
  subprocess tests. All 50 targeted checks pass after removing that variable;
  no production or test changes were needed. The frozen baseline non-slow suite
  completed with 10,913 passing tests using one numerical thread.
- User measurement constraint: Headroom passthrough runs Kompress v2 and may
  transform tool output or contend for CPU. Raw in-process JSON/NPZ artifacts
  and checksums are authoritative; tool latency/output are not receipts. Record
  proxy activity and load, and mark elapsed-time evidence unmeasured if contention
  prevents a credible comparison.
- First real severity comparison: 22,450 training / 2,494 held-out policies,
  92 coefficients, all routes certified. All three marginal supports fit within
  256 bins, so this is exact representation evidence, not bin approximation.
  Discrete prediction/covariance differences are at rounding scale. Its process
  RSS is higher than baseline at this size (about 609 MB versus 586 MB); retain
  this negative result and assess larger books before claiming a memory benefit.

- Final validation: the full implementation suite passed 11,135 tests with 470
  skips; the separately required real-data suites passed all 84 tests with
  `SUPERGLM_REQUIRE_DATA=1` and optional mpmath available. Ruff lint/format,
  lock/environment checks and `run_test.py` pass. The latter's U-shape CHECK is
  byte-identical on the baseline and is not a new passing shape claim.
- Independent component and integration reviews are clear after correcting
  subclass state restoration and stale discrete-support documentation. The
  max-effort mathematical review qualified stationarity and derivative-error
  wording; it found no demonstrated C3 solver defect. Fresh C3 receipts verify
  terminal authority, source stability and numerical threads; forged-receipt
  mutations are rejected.
- The real 22,450-policy severity comparison saves no RSS. Synthetic replication
  to 449,000 rows passes the same stationarity checks and preserves held-out
  predictions/covariance. The separate 610,212-row real NB2 comparison stops
  `objective_rejected` in both versions; its memory reduction is not evidence of
  a solved smoothing optimum.
- The first serial timing window is excluded from speed claims because Headroom
  activity reached 0.676 CPU core during a fit. With all agents idle and root
  tool traffic paused, the second window had Headroom below 0.006 core.
  Three fresh processes per arm give median fit time 17.427 to 14.750 seconds
  (15.36% lower) and median peak process RSS 1,514.84 to 1,015.93 MiB (32.93%
  lower). These are qualified local results for the replicated severity
  fixture, with one numerical thread and no explicit warmup.
- Continuous-grid Gaussian fits at 64/256/1,024 bins all pass stationarity;
  held-out location RMS differences from dense are 0.003817/0.000821/0.000216.
  These are approximation differences in changed designs, separate from the
  finite-support representation checks.
- The [completion report](../../research/2026-09-c3-c1-completion-evidence.md),
  [C3 diagnosis](../../research/2026-09-c3-stress-evidence.md) and tracked benchmark
  receipts retain successful and unsuccessful outcomes. The roadmap retires the
  chosen C3/C1 scope, keeps C5 next, and records remaining NB2, memory and
  certification limits. Source document hashes match the original preserved
  copies; version files remain unchanged.
