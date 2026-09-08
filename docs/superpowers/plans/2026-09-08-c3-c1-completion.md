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
- [ ] Finish baseline non-slow suite and classify any existing failures.

## Stage 2: Promote observed grouped fitting

**Ownership:** solver/solver.py, solver/chunks.py, api.py; public/chunk regression
tests. Coordinate any shared helper with the derivative worker.

- [ ] First demonstrate the public discrete refusal and the observed Tweedie/NB2
  chunk rejection with focused failing tests.
- [ ] Enable public discrete fitting with bounded automatic row chunks; retain
  explicit Fisher capability refusals and correct fallback semantics.
- [ ] Prove same-design score/curvature, fixed-fit and complete smoothing
  equivalence, signed cross-blocks, weights/offsets, and actual grouped dispatch.
- [ ] Validate API cloning, refits, serialization, inference and prediction.

Run targeted public, grouped assembly, chunking, memory and family tests.

## Stage 3: Bound certification and uncertainty design allocation

**Ownership:** smoothing/derivatives.py, smoothing/newton.py, posterior.py and new
focused derivative/allocation tests. Endpoint LAML already uses grouped matvec;
change it only if a regression exposes a gap.

- [ ] Add a failing no-full-design allocation regression around grouped exact
  LAML gradient/Hessian and certified Newton completion.
- [ ] Introduce a shared bounded design adapter/operations and remove forced
  dense predictor matrices in derivative and posterior smoothing replay.
- [ ] Compare gradient/Hessian certificates, objective, covariance, finite and
  infinity-face behavior against identical dense representation.
- [ ] Validate smoothing uncertainty and adversarial derivative/refusal tests.

## Stage 4: Remove repeated full-row work from chunk extraction

**Ownership:** underlying group matrix row-subset implementations and their tests.

- [ ] Demonstrate repeated full-parent sorting in factor-by-smooth chunk access.
- [ ] Reuse immutable row-order lookup or a contiguous slicing route while
  preserving arbitrary subset/repeated-row and original row-order semantics.
- [ ] Test exact matrix reconstruction separately from allocation/dispatch work.
- [ ] Benchmark a complete categorical/smooth distributional fit against baseline.

## Stage 5: Resolve the named C3 cases

**Ownership:** C3 regression tests and solver/smoothing fixes identified by exact
reproduction; coordinate modifications with Stages 2–3 before editing.

- [ ] Run the recovered correlated Tweedie and GPD fixtures on frozen baseline.
- [ ] Classify inner versus outer failure, finite stationarity versus cap/face
  decisions, curvature/rank evidence, and sensitivity to starts/optimizer choice.
- [ ] Add focused mathematical regressions and minimal justified corrections.
- [ ] Repeat on the implementation; retain an explicit explained limitation for
  any case that cannot be certified without changing the statistical model.

## Stage 6: Reproducible practical evidence and integration

**Ownership:** benchmarks/c3_c1_complete_fit.py (and fixture helper), result JSON
and report; documentation/roadmap owned by root.

- [ ] Implement a public-API subprocess harness with baseline/current source
  selection, pure recovered fixtures and real freMTPL2 workloads.
- [ ] Record serial complete-fit timing and whole-process peak RSS; use a separate
  instrumented replica for work/dispatch and save numerical artifacts.
- [ ] Separate same-design exact grouped equivalence from finite-grid sensitivity.
- [ ] Establish practical reliable insurance fitting plus a meaningful memory,
  feasible-size or fitting/validation-time improvement; state limits honestly.
- [ ] Run focused tests, full pytest, Ruff check/format, lock/environment checks,
  run_test.py, and the three `SUPERGLM_REQUIRE_DATA=1` real-data suites.
- [ ] Obtain independent mathematical/code review; fix substantiated findings.
- [ ] Update roadmap, design/plan facts and benchmark report; verify source hashes.

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
  no production or test changes were needed. The full baseline suite now runs
  from the detached frozen checkout with one numerical thread.
- User measurement constraint: Headroom passthrough runs Kompress v2 and may
  transform tool output or contend for CPU. Raw in-process JSON/NPZ artifacts
  and checksums are authoritative; tool latency/output are not receipts. Record
  proxy activity and load, and mark elapsed-time evidence unmeasured if contention
  prevents a credible comparison.
