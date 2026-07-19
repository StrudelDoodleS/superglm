# Tweedie Exact Profile Acceleration Implementation Plan

> **Design:** `docs/superpowers/specs/2026-07-19-tweedie-joint-profile-design.md`

**Goal:** Replace repeated exact-density evaluation in ordinary Tweedie
profiling with one compiled sufficient-statistics sweep per Newton evaluation,
then use safeguarded analytic dispersion and power updates while retaining the
current globally defensive profiler as fallback.

**Architecture:** Add one private Numba numerical module. Keep orchestration,
records, diagnostics, and public API policy in `profiling/tweedie.py`. Integrate
in three separately testable layers: exact kernel, fixed-power dispersion
Newton, and eligible outer power solve.

**Constraints:** No ordinary-fit numerical changes, no saddlepoint/Pearson
substitution in the ordinary exact fast path, no unbounded work, no mgcv code,
and no broad optimizer abstraction.

---

## Task 1: Establish derivative and call-count contracts

**Files:**

- Modify: `tests/test_tweedie_numerics.py`
- Modify: `tests/test_tweedie_profile_performance.py`

1. Add independent finite-difference helpers for aggregate NLL derivatives in
   `(p, log(phi))` using only the existing public exact density.
2. Add deterministic fixtures covering positive-only data, zeros, prior
   weights, `p < 1.5`, `p == 1.5`, and `p > 1.5`.
3. Record the neutral 800-row profile reference and a stable current density
   pass count. Do not assert wall time in normal CI.
4. Run the new tests and confirm they fail because the fused-kernel API does not
   yet exist.

Run:

```bash
rtk pytest tests/test_tweedie_numerics.py -k "joint or sufficient"
rtk pytest tests/test_tweedie_profile_performance.py -k "joint_call_count"
```

## Task 2: Implement compiled special-function primitives

**Files:**

- Create: `src/superglm/_tweedie_profile_kernel.py`
- Modify: `tests/test_tweedie_numerics.py`

1. Write failing parameterized tests comparing private scalar digamma and
   trigamma helpers with `scipy.special.digamma` and `polygamma(1, x)` across
   the positive argument range used by the series.
2. Implement Numba-compatible recurrence-to-asymptotic digamma and trigamma
   functions under `@njit(cache=True)`, without `fastmath`.
3. Include explicit finite/positive input handling and deterministic status
   behavior.
4. Run the focused tests and lint the new module.

Run:

```bash
rtk pytest tests/test_tweedie_numerics.py -k "digamma or trigamma"
rtk ruff check src/superglm/_tweedie_profile_kernel.py tests/test_tweedie_numerics.py
```

Commit:

```bash
rtk git add src/superglm/_tweedie_profile_kernel.py tests/test_tweedie_numerics.py
rtk git commit -m "feat: add compiled Tweedie derivative primitives"
```

## Task 3: Implement the fused exact sufficient-statistics kernel

**Files:**

- Modify: `src/superglm/_tweedie_profile_kernel.py`
- Modify: `tests/test_tweedie_numerics.py`

1. Define a compact immutable Python result wrapper and integer kernel status
   codes for success, work limit, unsafe mode, non-finite term, and non-finite
   derivative.
2. Write failing tests for aggregate log likelihood, `p` score, `log(phi)`
   score, both Hessian diagonals, and the cross derivative. Compare likelihood
   to the existing exact route and derivatives to independent central
   differences.
3. Implement one mode-centered upward/downward compiled sweep that accumulates
   mass and derivative moments without row-sized result matrices.
4. Add analytic positive canonical and zero-mass contributions, weighted using
   effective dispersion `phi / w`.
5. Enforce the existing row/total work limits before unsafe allocation and
   return a failure status rather than throwing from compiled code.
6. Add row-order invariance, distant-mode, budget, and non-finite-status tests.
7. Measure gamma-term table reuse against scalar recomputation; keep only the
   faster bounded strategy and cover both contiguous and sparse-index paths.

Run:

```bash
rtk pytest tests/test_tweedie_numerics.py -k "sufficient or joint_kernel"
rtk ruff check src/superglm/_tweedie_profile_kernel.py tests/test_tweedie_numerics.py
```

Commit:

```bash
rtk git add src/superglm/_tweedie_profile_kernel.py tests/test_tweedie_numerics.py
rtk git commit -m "feat: fuse exact Tweedie profile derivatives"
```

## Task 4: Add one-time profile preparation

**Files:**

- Modify: `src/superglm/_tweedie_profile_kernel.py`
- Modify: `src/superglm/profiling/tweedie.py`
- Modify: `tests/test_tweedie_profile.py`

1. Write failing tests proving that `y`, `log(y)`, weights, masks, and immutable
   metadata are prepared once and owned independently of caller arrays.
2. Add a small `_ExactProfileData` holder built once by each profile context.
3. Route candidate `(p, phi, mu)` evaluations through the compiled aggregate
   kernel without constructing `_PreparedTweedieDensity`, masks, closures, or
   per-row density result arrays on every Newton step.
4. Add one final-validation adapter that evaluates the existing authoritative
   exact density once and compares NLL/score within binary64 tolerances.
5. Confirm preparation and validation failures leave the old path available.

Run:

```bash
rtk pytest tests/test_tweedie_profile.py -k "exact_profile_data or joint_validation"
rtk ruff check src/superglm/_tweedie_profile_kernel.py src/superglm/profiling/tweedie.py
```

## Task 5: Accelerate fixed-power MLE dispersion profiling

**Files:**

- Modify: `src/superglm/profiling/tweedie.py`
- Modify: `tests/test_tweedie_profile.py`
- Modify: `tests/test_tweedie_profile_performance.py`

1. Write failing tests comparing the fast fixed-power Newton result with the
   existing tight derivative-free references for routine weighted/zero cases.
2. Add failure-injection tests for work limits, bad curvature, rejected steps,
   branch changes, and final-validation disagreement. Require deterministic
   fallback reasons and the existing authoritative result.
3. Implement safeguarded Newton in `u = log(phi)` with previous/Pearson seed,
   step limits, improvement checks, halving, score tolerance, and hard iteration
   cap.
4. Materialize the existing `_PhiProfileResult` fields and exact density
   diagnostics without changing public trace schema.
5. Use the accelerator in fixed-power MLE evaluations made by all outer methods;
   retain the current global profiler whenever certification fails.
6. Add a call-count assertion showing routine fixed-power profiles use no more
   than four compiled sweeps after warm-up.

Run:

```bash
rtk pytest tests/test_tweedie_profile.py -k "fast_phi or phi_profile"
rtk pytest tests/test_tweedie_profile_performance.py -k "fast_phi_call_count"
```

Commit:

```bash
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_profile.py tests/test_tweedie_profile_performance.py
rtk git commit -m "perf: add safeguarded exact Tweedie phi Newton"
```

## Task 6: Implement eligible joint-ML power profiling

**Files:**

- Modify: `src/superglm/profiling/tweedie.py`
- Modify: `tests/test_tweedie_profile.py`
- Modify: `tests/test_tweedie_profile_performance.py`

1. Replace the old `joint_ml` not-implemented test with failing correctness,
   trace, callback, and convergence tests.
2. Add eligibility tests: ordinary ML takes the fast path; penalized fits must
   pass a two-sided exact-profile certificate, while constraints, Pearson
   dispersion, and REML use the trusted outer fallback.
3. Implement the exact outer score solve with Schur-curvature proposal,
   secant/bracket alternatives, strict bounds, warm coefficient/dispersion
   starts, candidate validation, and bounded iteration count.
4. Store only genuinely fixed-power-profiled records in the existing cache and
   finalize through `_finalize_profile_record`.
5. If the fast solve cannot certify a result, continue Brent using accumulated
   valid records and append a concise fallback reason to outer diagnostics.
6. Add deterministic neutral-reference and randomized comparisons against the
   existing Brent result on both sides of `p=1.5`.
7. Assert no more than 12 kernel calls on the neutral fixture initially; tighten
   toward eight after profiling.

Run:

```bash
rtk pytest tests/test_tweedie_profile.py -k "joint_ml"
rtk pytest tests/test_tweedie_profile_performance.py -k "joint_ml or joint_call_count"
```

Commit:

```bash
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_profile.py tests/test_tweedie_profile_performance.py
rtk git commit -m "feat: add exact joint Tweedie profile solver"
```

## Task 7: Make the adaptive method the default

**Files:**

- Modify: `src/superglm/profiling/tweedie.py`
- Modify: `src/superglm/model/profile_ops.py`
- Modify: `src/superglm/model/api.py`
- Modify: `docs/api/families.md`
- Modify: `docs/guide/fitting.md`
- Modify: `tests/test_tweedie_profile.py`

1. Write failing tests that omitted `method` selects `auto`, eligible ordinary
   MLE selects `joint_ml`, and unsupported cases preserve Brent behavior.
2. Change defaults to `method="auto"`, document dispatch, and retain explicit
   legacy method behavior.
3. Ensure public model publication, progress callbacks, final refit
   synchronization, lazy CI, and caller-state isolation work identically with
   the fast result.
4. Update public documentation with exact/fallback behavior and no performance
   promises tied to a particular machine.

Run:

```bash
rtk pytest tests/test_tweedie_profile.py -k "auto or joint_ml or publication or callback"
rtk ruff check src/superglm/profiling/tweedie.py src/superglm/model/profile_ops.py src/superglm/model/api.py
```

Commit:

```bash
rtk git add src/superglm/profiling/tweedie.py src/superglm/model/profile_ops.py src/superglm/model/api.py docs/api/families.md docs/guide/fitting.md tests/test_tweedie_profile.py
rtk git commit -m "perf: default to adaptive exact Tweedie profiling"
```

## Task 8: Pull out measured hot-path overhead

**Files:**

- Modify only files identified by profiles from Tasks 3--7
- Modify: `tests/test_tweedie_profile_performance.py`

1. Warm Numba once, then measure counterbalanced medians for current explicit
   Brent, new joint/auto, and the recorded mgcv fixture.
2. Profile the complete public call stack and record coefficient-fit count,
   kernel count, total terms, Python call count, preparation time, kernel time,
   and final-refit time.
3. Apply optimizations one at a time, retaining only measured wins:
   workspace reuse, fused validation, gamma-table reuse, fewer allocations,
   scalar result transport, cached transforms, or rejected-candidate short
   circuits.
4. Do not add concurrency, `fastmath`, approximate likelihood, relaxed cutoffs,
   or generalized abstractions to chase microbenchmarks.
5. Tighten stable call-count gates to eight or fewer if the fixtures support it.
6. Report whether internal and public timings reach, match, or exceed mgcv; do
   not conceal Numba cold-compilation time.

Run:

```bash
rtk pytest -s tests/test_tweedie_profile_performance.py -k "benchmark_report or joint"
rtk pytest tests/test_tweedie_profile_performance.py -k "call_count"
```

Commit:

```bash
rtk git add src tests/test_tweedie_profile_performance.py
rtk git commit -m "perf: remove Tweedie joint-profile overhead"
```

## Task 9: Full verification and branch handoff

1. Run focused numerical/profile suites, then the full non-browser suite.
2. Run formatting, Ruff, and mypy on changed code.
3. Re-run ordinary fit, pathological tiny-dispersion, neutral exact reference,
   joint profile, explicit Brent, REML, public publication, and performance
   reproductions in a clean process.
4. Inspect the complete diff for copied source, unrelated refactors, silent
   approximation, mutable caller state, and accidental API drift.
5. Record final warm/cold timings, call counts, parameter deltas, fallback
   coverage, and mgcv comparison in the PR description.

Run:

```bash
rtk pytest tests/test_tweedie_numerics.py tests/test_tweedie_profile.py tests/test_tweedie_profile_performance.py
rtk pytest -m "not browser"
rtk ruff check src tests
rtk mypy src/superglm/_tweedie_profile_kernel.py src/superglm/profiling/tweedie.py
rtk git diff --check
rtk git status --short
```

Use the verification-before-completion checklist before claiming the work is
ready or publishing an updated PR.
