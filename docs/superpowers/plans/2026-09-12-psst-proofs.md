# PSST interpretation and calibration implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish what PSST's score measures, correct its Gaussian-reference
normalization through a scoped follow-up, and determine which calibration
claims can be supported for a fitted GAM.

**Architecture:** Treat PSST as P7 of the algorithm proof programme, with four
component claims. Start from the implemented working quadratic and both
execution paths. Separate fixed Gaussian reference results from fitted-model
sampling and predictive ranking.

**Tech Stack:** Python 3.12+, NumPy/SciPy/Numba, existing screening fixtures,
Markdown/LaTeX and MkDocs.

**Spec:** [Algorithm proof design, P7](../specs/2026-09-12-algorithm-proofs-design.md#p7-psst-interpretation-and-calibration).
The companion [LSS plan](2026-09-12-algorithm-proofs.md) owns P1–P6.

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

## Delivery and ownership

Work in an isolated `.worktrees/` checkout. Use
`uv sync --python 3.13 --extra dev`; add `--group docs --extra plotting`
for the documentation build. Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
NUMBA_NUM_THREADS=2` for numerical witnesses.

This plan owns `docs/research/proofs/psst.md` and P7's entries in
`docs/research/proofs/index.md`. Coordinate index edits with the LSS work.
Use the shared ledger fields and statuses from the design. A runtime repair
belongs in a separate specification and PR after its invariant and
cross-backend implementation are concrete.

### Task 1: Establish the score's local fitting interpretation

**Files**

- Create: `docs/research/proofs/psst.md`.
- Update: P7.S1 in `docs/research/proofs/index.md`.
- Read: `src/superglm/model/screening_ops.py`,
  `src/superglm/screening/_pair_factor.py`, `_score_stat.py`,
  `_structured.py`, `_overlap.py` and `_factor_kernels.py`.
- Inspect tests: `tests/test_pair_design_factor.py`,
  `tests/test_screening_penalty_scaling.py`,
  `tests/test_mixed_interaction_screening.py`.

**Interface:** Produce the exact definitions of \(U,V,S,\phi\), the retained
candidate space, the pair overlap space, and the working quadratic in P7.S1.
These definitions are the input to all subsequent claims.

- [ ] Pin the code revision and compare its source tree with the design.
  Trace prediction, working score, Fisher weights and dispersion from the
  fitted scalar model to each candidate result.
- [ ] Derive the identity \(\max q_\lambda=T_\lambda/2\) by completing the
  square on the identified space. Define the gain relative to the nuisance-only
  profiled optimum, allowing unpenalized changes to the pair overlap. State
  what the rank policy retains and when the maximizer exists.
- [ ] Map the profiled factor to the pair's intercept and margin columns.
  Identify all fitted nuisance columns outside that projection.
- [ ] Show how tensor, slope and categorical blocks represent the intended
  refit spaces. Separate exact support aggregation from quantization and
  compare the structured and dense representations of the same working model.
- [ ] Map each component to the primary references in P7. Distinguish the
  classical score form, penalized quadratic optimization, variance-component
  testing, adaptive bandwidth choice and FAST's screening approximation.
  Do not infer that their full sampling theories transfer unchanged.
- [ ] Run the three listed suites and record the available-data status of the
  real-book cases. If those cases support a claim, fetch freMTPL2 and require
  it with `SUPERGLM_REQUIRE_DATA=1`, as specified in AGENTS.md.
- [ ] Obtain mathematical and source review; commit as
  `docs: derive the PSST working quadratic`.

**Acceptance:** A reader can distinguish the exact algebraic identity from
an approximation to a complete refit and from a statistical significance claim.

### Task 2: Derive reference moments and scope the normalization repair

**Files**

- Update: `docs/research/proofs/psst.md` and P7.S2 in the index.
- Read: `src/superglm/screening/_score_stat.py`,
  `src/superglm/screening/_structured.py` and
  `src/superglm/model/screening_ops.py`.
- Inspect tests: `tests/test_screening_penalty_scaling.py`,
  `tests/test_structured_screening.py`, `tests/test_interaction_screening.py`.

**Interface:** Produce the Gaussian-reference mean and variance for each
candidate/rung, the shared-draw ladder law, and a concrete corrective design
covering both execution paths. Retain the fixed-geometry assumption.

- [ ] Derive \(E(T/\phi)=\sum a_j\) and
  \(\operatorname{Var}(T/\phi)=2\sum a_j^2\). Use \(V=S=I_4,\lambda=1\)
  as an analytic witness: EDF 2, variance 2, while the current denominator
  is based on variance 4. The unpenalized identity case supplies the control.
- [ ] Account for the stored pencil's balance in \(v_j,s_j\). Derive the
  structured route's equivalent trace-of-square quantity without imposing
  a dense global decomposition or substituting the mains model's `edf1`.
- [ ] Derive joint ladder simulation using \(u_j^*=\sqrt{\phi v_j}Z_j\)
  with a shared draw across rungs. Cover clamped/duplicate rungs, zero-rank
  refusal, unequal spectra with equal EDF, and changing penalty units.
- [ ] Specify a regression that fails on the current normalization and a
  mutation restoring that denominator. Use analytical moment bounds and
  same-model backend agreement; a finite simulation is a witness, not proof.
- [ ] Write the bounded runtime-fix specification from the reviewed formulas.
  Require paired old/new rankings, numerical outputs, complete-screen time,
  peak RSS and actual dense/structured dispatch. Preserve the ranking-only
  public contract unless Task 3 establishes a stronger one.
- [ ] Require the published null-floor battery to be rerun after changing
  normalization. Its old maxima do not transfer to the corrected score.
- [ ] Run the three listed suites, review the argument and repair specification,
  and commit as `docs: specify PSST reference-variance normalization`.

**Acceptance:** The correction's meaning, numerical guards and cost are
reviewable before implementation. A corrected variance is not described as
complete ladder or fitted-model calibration.

### Task 3: Determine the fitted-model null and ranking objective

**Files**

- Update: `docs/research/proofs/psst.md`, P7.S3/P7.S4 in the index,
  `docs/guide/screening.md`, `docs/guide/screening-evaluation.md`.
- Read: `benchmarks/screening_null_floors.py`,
  `benchmarks/screening_worth_gate.py`,
  `tests/test_screening_guide_numbers.py`,
  `tests/test_screening_worth_gate.py` and the recorded FAST comparison.

**Interface:** Produce an applicability table for reference versus actual
null laws, and a specified evaluation of evidence ranking versus predictive
return per refit. Unproved calibration conditions remain explicit.

- [ ] Derive the mean/covariance expressions involving \(I-H_0\) from P7.S3.
  Include unit nuisance and candidate directions \(x,a\), both orthogonal
  to the pair overlap, with \(x^\top a=0.5\). For \(y\sim N(0,I)\),
  \(H_0=xx^\top\), \(U=a^\top(I-H_0)y\), \(V=1\) and \(\lambda S=1\),
  show that \(E(T)=3/8\), versus the frozen reference mean \(1/2\).
  Full projection also changes curvature: \(A_{\rm full}=(I-H_0)a\) gives
  \(V_{\rm full}=3/4\) and the matching reference mean \(3/7\).
- [ ] Analyse estimated dispersion, smoothing selection, non-Gaussian fourth
  moments and sparse support. Do not treat fitted geometry as independent
  of the response merely because it is held fixed during simulation.
- [ ] Define the hypotheses for a global additive null and for individual
  pairs when other interactions exist. Specify which model-fitting steps a
  bootstrap repeats and which validity assumptions it still needs.
- [ ] Set Monte Carlo precision from the intended tail threshold. State the
  exchangeability condition for the plus-one p-value, the marginal validity
  needed by Bonferroni/Holm, and any dependence condition proposed for BH.
  Independent per-pair simulations do not supply a joint maximum null.
- [ ] Specify a paired comparison of legacy `z`, reference-variance `z`
  and FAST using the same baseline and all-candidate refits. Report
  held-out gains and costs separately from training gain and null evidence.
  Include wider candidate sets and replicated data/splits; account for
  dependence between overlapping pairs and uncertain holdout ordering.
- [ ] Separate historical timings from new measurements. Keep complete-screen
  costs, complete-refit costs, sample sizes and library versions explicit.
  Do not infer an updated speed ratio from changes to unrelated kernels.
- [ ] Update the walkthrough and comparison claims to the reviewed results.
  Preserve unsupported or refuted claims in the proof record.
- [ ] Run the documentation checks and relevant existing screening suites;
  require the real data for any newly claimed book measurement. Open a
  docs-only review PR with `release:none`; runtime fixes retain their own PR.

**Acceptance:** Users can tell what PSST estimates, how FAST differs, which
results are inherited from established theory, and which calibration or
predictive claims remain unproved.
