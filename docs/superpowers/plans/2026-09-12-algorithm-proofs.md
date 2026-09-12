# Distributional Algorithm Proofs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish reviewable mathematical claims about SuperGLM's assembly,
reuse and convergence rules, with explicit limits where a claim cannot be proved.

**Architecture:** Work from a pinned implementation and a source-to-claim
ledger. Prove local algebra and state invariants before composing error bounds
and a conditional convergence argument. Keep boundary cases in separate
claims; any runtime change needs a focused follow-up specification.

**Tech Stack:** Markdown/LaTeX, Python 3.12+, NumPy/SciPy/Numba/tabmat,
existing pytest fixtures and the strict MkDocs build.

**Spec:** [Proof programme design](../specs/2026-09-12-algorithm-proofs-design.md).
Read it alongside this plan; the equations and claim IDs below use its notation.

The [companion PSST plan](2026-09-12-psst-proofs.md) executes P7 in the same
proof programme. It covers the score's meaning, normalization and calibration,
and can proceed alongside Tasks 2–3 below. This file remains the LSS plan.

## Global Constraints

- Target Python 3.12+; retain the existing NumPy/SciPy/Numba/tabmat CPU stack.
- No package-version, release-tag, or publication changes.
- This scope plans proofs; it does not authorize a new solver or public API.
- Keep exact-arithmetic, floating-point, optimization and statistical claims separate.
- Numerical tests use dimension-, epsilon-, norm- and conditioning-based bounds.
- Near-rank and cancellation fixtures check certification, refusal or stable observables.
- Adversarial regressions require an unfixed demonstration or a mutation check.
- Performance changes require complete-fit timing, peak RSS, numerical outputs and actual dispatch.
- A failed proof obligation may produce a counterexample or a narrower supported claim.

---

## Delivery order and file responsibilities

This is a research execution plan, not a promise that every desired theorem is
true. Each task ends in a separately reviewable document or counterexample.
Derivations may require sustained work; they are not timed implementation steps.
Do not create empty proof files or an unimplemented certification API.

| Task | Output under `docs/research/proofs/` | Dependency |
| --- | --- | --- |
| 1 | `index.md`: baseline, ledger, terminology and reference-to-code map | None |
| 2 | `assembly.md`: P1 exact identities and P2 arithmetic obligations | Task 1 |
| 3 | `reuse.md`: P3 state invariants and freshness guarantees | Task 1 |
| 4 | `stopping.md`: P4 error budget and residual theorem | Tasks 2 and 3 |
| 5 | `controller.md`: P5 conditional convergence and code obligations | Task 4 |
| 6 | `boundaries.md`: P6 applicability and separate boundary arguments | Tasks 1 and 4 |
| 7 | Updated `index.md`, public guide and roadmap | Reviews of the preceding outputs |

The initial milestone is Tasks 1–3. It can land without claiming a convergence
theorem. Tasks 4–6 have explicit gates below; an unsupported prerequisite stays
visible and prevents promotion of a dependent implementation claim.

Use isolated work under `.worktrees/` and preserve unrelated changes. Install
the environment with `uv sync --python 3.13 --extra dev`. For documentation
validation, add `--group docs --extra plotting` to that sync command.
For numerical witnesses use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
NUMBA_NUM_THREADS=2` so the existing two-worker tests can run.

### Task 1: Pin the algorithm and define the claim ledger

**Files**

- Create: `docs/research/proofs/index.md`.
- Read: the design's source table, `docs/models/distributional.md`,
  `docs/research/2026-09-c3-stress-evidence.md`,
  `benchmarks/lss_newton_completion.md`, and the three primary papers linked
  by the design.

**Interface:** Produce records P1–P7 with fields `status`, `statement`,
`assumptions`, `source_symbols`, `argument`, `numerical_obligations`,
`regression_evidence`, and `review_disposition`. These are document fields,
not a proposed runtime type. Use only the five statuses defined in the design.

- [ ] Record `git rev-parse HEAD` and `git rev-parse HEAD:src/superglm`.
  Compare the source tree with the design's pinned source tree. Explain any source
  difference before using an earlier review or benchmark as current evidence.
- [ ] Transcribe the optimizing likelihood, penalty and Laplace objective from
  `joint_laplace_objective`; state the coefficient space, nullspaces, fixed
  parameters, smoothing box and objective constants. Record elimination of
  homogeneous equality constraints, exclusion of active inequalities, positive
  semidefinite penalties and the continuously selected smooth mode branch.
- [ ] Map Wood/Pya/Säfken's objective and derivatives, and Wood/Fasiolo's proposal
  assumptions, to named source functions. List the implementation's grouped
  assembly, reuse, mixed-step and stopping modifications separately. A departure
  from a paper is not automatically a research novelty.
- [ ] Create P1–P7 records with precise proposed statements and the unresolved
  obligations already identified in the design. For each assumption, distinguish
  runtime-checked, caller-supplied and analytically unestablished conditions.
  P7 points to the companion plan's four component statements.
- [ ] Review that `converged_`, `smoothing_certified_`, stationary point,
  local minimum, global minimum and interval coverage are distinct terms.
  Validate every source symbol and relative link.
- [ ] Commit the ledger as `docs: map distributional proof obligations to source`.

**Acceptance:** A reviewer can identify the exact algorithm, objective and
assumptions behind every proposed claim. No proof status is inferred from CI.

### Task 2: Derive assembly identities and arithmetic bounds

**Files**

- Create: `docs/research/proofs/assembly.md`.
- Update: P1/P2 in `docs/research/proofs/index.md`.
- Read: `src/superglm/distributional/solver/assembly.py`, `chunks.py`,
  `_global_moments.py`, `_batched_moments.py`, `curvature.py`, and
  `src/superglm/distributional/smoothing/penalty_geometry.py`.
- Inspect tests: `tests/test_distributional_chunking.py`,
  `tests/test_distributional_global_moments.py`,
  `tests/test_distributional_global_moment_ranges.py`,
  `tests/test_distributional_global_moment_integration.py`.

**Interface:** P1 supplies an exact identity for a fixed compiled model.
P2 supplies a stated arithmetic model and error inequalities, including their
unverified backend assumptions. Task 4 must not treat an unbounded term as zero.

- [ ] Derive the partition identity for scores and all joint curvature blocks,
  including signed cross-predictor weights. Express each grouped block as
  support-table contraction against sums over joint support indices. State how
  the indices and any retained row factors recover the actual compiled values;
  equal sparse support indices alone do not establish equal row values.
- [ ] Map intercepts, ordinary columns, spline/tensor supports, constraints,
  likelihood weights and penalties to that identity. Check every admission and
  fallback branch, especially that penalties and weights enter exactly once.
- [ ] Draw the actual accumulation stages: row products, histogram reductions,
  support contractions and final block assembly. Assign an error contribution
  to each stage under an explicit reduction and exceptional-range model.
- [ ] Derive absolute bounds using sums of absolute operands and \(\gamma_n\)
  where its assumptions hold. Carry product and histogram errors through the
  contractions. Explain why a small final signed sum is not a valid error scale.
- [ ] State what matrix residuals and conditioning can establish about solves,
  rank decisions and coefficient-forward error. Mark uncovered BLAS/native
  assumptions or range regimes `unsupported`; do not extrapolate a bound from
  a single observed roundoff residual.
- [ ] Run the four listed suites and record exact commands and outcomes. Match
  each relevant assertion to the claimed invariant. Identify any claim lacking
  a witness; specify its mathematical counterexample before scoping a new test.
- [ ] Obtain mathematical and source-mapping review, update P1/P2 independently,
  and commit as `docs: derive compiled-design assembly and error bounds`.

**Acceptance:** Exact compiled-design equivalence is separate from floating-point
accuracy, continuous-grid approximation, backend dispatch and fitting speed.
An incomplete P2 does not invalidate a correctly qualified P1.

### Task 3: Prove reuse invariants and the curvature-refresh property

**Files**

- Create: `docs/research/proofs/reuse.md`.
- Update: P3 in `docs/research/proofs/index.md`.
- Read: `src/superglm/distributional/solver/_reuse_digest.py`, `solver.py`,
  `src/superglm/distributional/smoothing/derivatives.py` and `newton.py`.
- Inspect tests: `tests/test_distributional_reuse_digest.py`,
  `tests/test_distributional_chunk_reuse.py`,
  `tests/test_distributional_fisher_geometry_reuse.py`,
  `tests/test_distributional_newton_curvature_refresh.py` and
  `tests/test_distributional_newton_practical_handoff.py`.

**Interface:** Produce a state-transition table for each cache and curvature
verdict: consumed state, admitted changes, invalidation guard and mathematical
quantity reused. This table is the provenance assumption used by P4 and P5.

- [ ] Define the model state and the dependency subset of each cached quantity.
  Include ownership and mutation rules, likelihood-plan contents, derivative
  step, rank/face state and changed penalties. State digest collision assumptions.
- [ ] Check each guard against that dependency set. Give a preservation argument
  for allowed changes and a concrete stale-value example for each excluded class.
- [ ] Trace accepted fit, derivative workspace, exact Hessian verdict and BFGS
  memory transitions. Prove only the fresh-evaluation property stated in P3,
  conditional on passing other gates and having sufficient iteration budget.
- [ ] Map the two completion regressions to the unfixed revision and the
  practical-handoff controls to their recorded mutation. Do not replace the
  mathematical state argument with those examples.
- [ ] Run the five listed suites with the declared thread limits. Review
  alias/mutation, changed penalty, rejected trial and unavailable-Hessian cases.
- [ ] Obtain source and mathematical review, record unresolved guard gaps, and
  commit as `docs: specify reuse and Newton curvature provenance`.

**Acceptance:** The claim explains when reuse is equivalent to recomputation,
under stated assumptions, and what the refresh repair guarantees. It does not
promise positive curvature or eventual convergence for every fit.

### Task 4: Build a defensible stopping error budget

**Files**

- Create: `docs/research/proofs/stopping.md`.
- Update: P4 in `docs/research/proofs/index.md`.
- Read: `src/superglm/distributional/smoothing/endpoint_direction.py`,
  `derivatives.py`, `penalty_geometry.py`, and
  `src/superglm/reml/convergence.py`.
- Inspect tests: `tests/test_distributional_endpoint_direction.py`,
  `tests/test_distributional_endpoint_roundoff.py`,
  `tests/test_distributional_bounded_derivatives.py`,
  `tests/test_distributional_endpoint_laml.py`.

**Interface:** Produce an inequality
\(\|\nabla F-\hat g\|_\infty\leq\delta\), with an explicit expression and
applicability conditions for every term, or record which missing bound prevents
it. P5 consumes this result; a refinement indicator is not a substitute.

- [ ] Start with an interior GaussianLS or GammaLS example on a fixed-rank
  branch and Newton's `stationary` outcome. Exclude EFS `lambda_change` and
  `objective_plateau` residuals from this full-gradient claim. Identify a
  neighborhood in which the coefficient solution is unique and its Hessian
  has a positive lower bound.
- [ ] Derive a coefficient-residual-to-mode bound in that neighborhood.
  Include the residual's own numerical error. If the neighborhood cannot be
  certified, retain it as an assumption and do not claim runtime certification.
- [ ] Differentiate the coefficient stationarity equation and the log determinants.
  Propagate P2's solve/assembly errors and family evaluation errors through the
  actual profile-gradient expression.
- [ ] Derive finite-difference truncation and roundoff bounds for the actual
  stencils, with the needed higher derivative bounds and valid perturbation
  domain. Compare those bounds with the current refinement indicators.
- [ ] Prove the design's interior and box residual inequalities. Audit objective
  constants, coordinate scaling, bound projection and heuristic freezing.
  Derive the bridge from the production masked score to \(\hat R\), including
  the exact-bound projection and frozen-coordinate recheck. Account for
  \(\max(\tau,\phi)+\delta\) with the design's definitions, rather than treating
  separate tolerance checks as one true-residual guarantee.
- [ ] Run the four listed suites. Use their exact polynomial/derivative,
  cancellation and endpoint cases as witnesses; record uncovered assumptions.
- [ ] Obtain mathematical and source review. If an enforceable bound is found,
  write a separate specification for the smallest runtime enforcement change,
  including a failing invariant test and its mutation. Otherwise record the
  obstruction precisely. Commit as `docs: bound strict smoothing residuals`.

**Acceptance:** P4 is either a qualified theorem with all terms accounted for
or an explicit unresolved error budget. Existing fit labels are not strengthened
by a document-only result.

### Task 5: Analyse the combined outer controller

**Files**

- Create: `docs/research/proofs/controller.md`.
- Update: P5 in `docs/research/proofs/index.md`.
- Read: `src/superglm/distributional/smoothing/newton.py`, `loop.py`,
  `objective.py` and `src/superglm/reml/convergence.py`.
- Inspect tests: `tests/test_distributional_newton_endgame.py`,
  `tests/test_distributional_practical_stop.py`,
  `tests/test_c3_stress_stationarity.py` and both completion suites in Task 3.

**Interface:** Consume P3 provenance and P4's qualified error budget. Produce a
conditional theorem with a line-by-line obligation map for the implemented
controller. A theorem for a modified controller must be labelled as such.

- [ ] Specify the strict, smooth, fixed-rank finite-box algorithm as a transition
  table: Newton proposal, BFGS reuse, EFS fallback/mixing, clipping, trial fit,
  acceptance, freezing, restart and refusal. Define the continued sequence with
  practical and finite-tolerance success exits disabled; analyse refusal
  separately from that sequence.
- [ ] Check feasible descent and spectral bounds for every proposal. Evaluate
  the directional product of the actual clipped step and the full objective
  gradient. Record a counterexample when an existing shortcut fails the claim.
- [ ] Use P4 to formulate objective/derivative inexactness conditions. Audit
  whether the current fixed tolerances, objective allowance and freezing floor
  satisfy them; do not assume a shrinking forcing sequence exists. Extend error
  control to all iterations and line-search trials used by the proof; a
  terminal enclosure alone is insufficient.
- [ ] Derive sufficient decrease of the form in the design from those conditions.
  Account for the slack in the bounded sublevel set. Telescope to obtain residual
  convergence, then establish accumulation-point first-order conditions.
- [ ] State separately what remains true with fixed tolerances, finite budgets
  and practical stopping. Identify any second-order condition needed before
  calling a stationary point a local minimum.
- [ ] Run the five listed suites. Attach existing stalled-trajectory witnesses
  without using them as evidence of universal convergence.
- [ ] Obtain mathematical and source review. If the implemented rule fails an
  obligation, produce a bounded corrective design with an adversarial witness;
  leave implementation and complete-fit performance validation to that follow-up.
  Commit as `docs: analyse safeguarded distributional convergence`.

**Acceptance:** The document distinguishes a theorem about production code,
a conditional theorem with unchecked assumptions, and a theorem requiring a
changed algorithm. It makes no global-optimality claim.

### Task 6: Determine applicability at faces and nonsmooth joins

**Files**

- Create: `docs/research/proofs/boundaries.md`.
- Update: P6 in `docs/research/proofs/index.md`.
- Read: `src/superglm/distributional/smoothing/penalty_face.py`,
  `endpoint_laml.py`, `faces.py`, `penalty_geometry.py` and
  `src/superglm/distributional/kernels/two_piece.py`.
- Inspect tests: `tests/test_distributional_endpoint_context.py`,
  `tests/test_distributional_endpoint_summary.py`,
  `tests/test_distributional_curvature_policy.py` and
  `tests/test_two_piece_lss_kernel.py`.

**Interface:** Produce an applicability table by branch/family mechanism,
identifying which of P1–P5 remain valid and which assumptions break.

- [ ] Derive the infinite-penalty nullspace limit and the cancellation of
  divergent Laplace terms under explicit rank and nullspace assumptions.
  Separate a limit, a fit on the limiting face and a finite-cap approximation.
- [ ] Trace numerical rank and face changes, including covariance context and
  invalidation. Identify assumptions needed for eventual branch stabilization;
  do not infer stabilization from memory resets.
- [ ] At a two-piece join, compute one-sided row derivatives and the resulting
  curvature/log-determinant behavior. State whether the profile objective is
  continuous and differentiable in each identified case.
- [ ] List family support restrictions and unresolved family evaluation bounds.
  Choose a separate restricted theorem or future nonsmooth analysis only after
  establishing the relevant regularity.
- [ ] Run the four listed suites. Use stable subspaces, reconstruction and
  one-sided identities; near-rank coefficient agreement is not an acceptance test.
- [ ] Review the applicability table, record counterexamples and bounded
  follow-up scopes, and commit as `docs: delimit boundary proof assumptions`.

**Acceptance:** No smooth fixed-rank theorem silently includes rank transitions,
exact infinite penalties or two-piece joins.

### Task 7: Publish only the claims established by review

**Files**

- Update: `docs/research/proofs/index.md`, `docs/models/distributional.md`,
  `docs/ROADMAP.md` and `mkdocs.yml` if adding the proof index to navigation.
- Review: P1–P6 and their referenced evidence; include the current P7 disposition
  from the companion PSST work without requiring that independent work to finish.

**Interface:** The public guide consumes reviewed claim statements and their
applicability limits. It does not change the meaning of a fitted attribute.

- [ ] Reconcile every claim's mathematical review with its source review.
  Preserve unresolved objections, failed obligations and counterexamples.
- [ ] Record exact source revisions and test receipts. Separate published-paper
  results, new derivations, conditional assumptions and empirical observations.
- [ ] Add concise links from the public smoothing guide to the reviewed claims.
  Explain any gap between a mathematical target and the existing numerical check.
- [ ] Update the roadmap with completed claims and the next bounded obligation.
  Keep new solver changes and statistical coverage research separately scoped.
- [ ] Run `uv run --no-sync mkdocs build --strict` and `git diff --check`.
  For any independently approved runtime follow-up, run its focused regressions,
  required repository checks and complete-fit comparison before claiming success.
- [ ] Open a docs-only review PR declaring exactly one `release:none` advisory
  impact and its rationale. Runtime fixes use their own impact declaration.

**Acceptance:** A reader can tell what is proved, under what assumptions,
which code revision it covers and what remains an empirical check.
