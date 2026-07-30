# PR 165 Release Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every verified P1/P2 release-review finding on PR #165,
preserve compact structured performance and Python 3.10 support, and close the
remaining low-risk review cleanup.

**Architecture:** Keep unsupported singular or constrained geometry out of the
structured solver at dispatch boundaries, then add factor-level numerical
safety nets so invalid pseudo-geometry cannot be published. Reuse existing
compact penalty algebra for NB2 and post-fit repair, replace the one
version-incompatible SciPy batch call with NumPy batching, and add a minimum
Python pull-request gate without duplicating master CI.

**Tech Stack:** Python 3.10+, NumPy, SciPy, Tabmat-backed SuperGLM design
matrices, pytest, Ruff, uv, GitHub Actions.

---

## File Map

- `src/superglm/solvers/_structured/selection.py`: dynamic zero-penalty
  eligibility and forced-mode errors.
- `src/superglm/solvers/_structured/factors.py`: absolute Schur-pivot
  certification and coupled-null rejection.
- `src/superglm/solvers/irls_direct.py`: defensive auto fallback when compact
  penalties are unavailable.
- `src/superglm/profiling/nb.py`: build and reuse compact penalty components
  during automatic theta estimation.
- `src/superglm/solvers/sum_to_zero.py`: dependency-floor-compatible batched
  local eigendecomposition.
- `src/superglm/model/fit_ops.py`: public preflight for fit-time constraints
  combined with structured terms.
- `src/superglm/model/shape_ops.py`: compact post-fit penalty evaluation.
- `src/superglm/solvers/_structured/assembly.py`: authoritative scalar
  `S_override` geometry validation.
- `src/superglm/solvers/_structured/geometry.py`: explicit numerical
  certification failure contract.
- `src/superglm/inference/covariance.py`: NumPy-compatible boolean selector
  validation.
- `src/superglm/reml/gradient.py`: remove dead cross-trace implementation.
- `src/superglm/reml/w_derivatives.py`: correct compact derivative typing.
- `examples/fremtpl2_credibility.py`: stop teaching private model APIs.
- `docs/guide/credibility.md`: clarify REML versus selection and qualify the
  dated OpenML result snapshot.
- `.github/workflows/ci.yml`: run the complete Python 3.10 compatibility lane
  on pull requests while retaining the full master matrix.
- Focused tests live in `tests/test_structured_factor.py`,
  `tests/test_random_effect_inference.py`,
  `tests/test_factor_smooth_structured_parity.py`,
  `tests/test_structured_irls.py`, `tests/test_nb2.py`,
  `tests/test_sum_to_zero_structured_factor.py`,
  `tests/test_shape_reml.py`, `tests/test_shape_postfit.py`, and
  `tests/test_fremtpl2_credibility_demo.py`.

### Task 1: Reject zero-penalty aliased structured geometry

**Files:**

- Modify: `tests/test_random_effect_inference.py`
- Modify: `tests/test_factor_smooth_structured_parity.py`
- Modify: `tests/test_structured_factor.py`
- Modify: `src/superglm/solvers/_structured/selection.py`
- Modify: `src/superglm/solvers/_structured/factors.py`

- [ ] **Step 1: Add failing public RandomEffect dispatch tests**

Add tests that fit an all-level `RandomEffect` with
`LambdaPolicy.off()` and assert:

```python
auto = SuperGLM(
    family="gaussian",
    features={"group": RandomEffect(lambda_policy=LambdaPolicy.off())},
    direct_solve="auto",
).fit_reml(X, y, runtime_validation="skip")

assert auto.result.direct_backend == "gram"
assert "zero penalty" in auto.result.direct_fallback_reason

with pytest.raises(
    ValueError,
    match=r"direct_solve='structured'.*zero penalty.*intercept",
):
    SuperGLM(
        family="gaussian",
        features={"group": RandomEffect(lambda_policy=LambdaPolicy.off())},
        direct_solve="structured",
    ).fit_reml(X, y, runtime_validation="skip")
```

Also fit the same data with `LambdaPolicy.fixed(1e-8)` and assert that automatic
dispatch remains structured and agrees with Gram predictions and fitted
lambdas.

- [ ] **Step 2: Run the RandomEffect tests and verify RED**

Run:

```bash
rtk pytest tests/test_random_effect_inference.py -k "unpenalized or zero_penalty"
```

Expected: the auto case reports `structured`, and forced structured does not
raise at dispatch.

- [ ] **Step 3: Add failing FS/SZ dispatch tests**

In `tests/test_factor_smooth_structured_parity.py`, use explicit
`LambdaPolicy.off()` for all ordinary FS components. Assert auto falls back,
forced structured rejects, and forced Gram fits. Add an SZ control with its
wiggle fixed at zero and assert structured dispatch still succeeds and matches
Gram predictions.

- [ ] **Step 4: Run the FS/SZ tests and verify RED**

Run:

```bash
rtk pytest tests/test_factor_smooth_structured_parity.py -k "zero_penalty"
```

Expected: ordinary FS still selects structured or fails inside factorization.

- [ ] **Step 5: Add failing factor-level cancellation tests**

Construct a scalar Schur system with:

```python
rng = np.random.default_rng(0)
d = 10.0 ** rng.uniform(-6.0, 6.0, 42)
C = d[:, None]
A = np.array([[np.sum(d)]])
```

The exact Schur complement is zero while floating-point subtraction leaves a
positive pivot. Assert construction raises a `LinAlgError` mentioning a
coupled rank-deficient Schur null space instead of publishing full rank.

Add the block analogue with one local basis direction reproducing the
intercept. Keep the existing `C == 0` singular-small-block test passing to
prove uncoupled singular Schur geometry remains supported.

- [ ] **Step 6: Run the factor tests and verify RED**

Run:

```bash
rtk pytest tests/test_structured_factor.py -k "cancellation or singular_schur"
```

Expected: cancellation constructions currently publish a factor instead of
raising.

- [ ] **Step 7: Implement zero-penalty eligibility**

In `selection.py`:

1. Move the forced-structured success return after dynamic safety checks.
2. Add a helper that either returns a dense fallback decision for `auto` or
   raises:

```python
def _backend_ineligibility(
    reason: str,
    mode: Literal["auto", "structured"],
    selection: StructuredGroupSelection,
) -> StructuredBackendDecision:
    if mode == "structured":
        raise ValueError(f"direct_solve='structured' is ineligible: {reason}")
    return StructuredBackendDecision(
        use_structured=False,
        group_index=selection.group_index,
        group_name=selection.group_name,
        fallback_reason=reason,
    )
```

3. Preserve the existing zero-weight-level reason before applying the general
   RE intercept-alias reason.
4. For ordinary FS only, inspect every repeated component lambda using the
   same component-name fallback as `_factor_smooth_singular_local_level`; any
   explicitly zero component is ineligible. Do not apply this rule to SZ.
5. Apply cost crossover only after all safety checks, and then return forced
   structured success when requested.

- [ ] **Step 8: Implement Schur pivot and coupled-null certification**

For both scalar and block factors:

1. Compute the reference scale from the pre-subtraction and eliminated blocks.
2. Define the absolute cutoff as:

```python
absolute_cutoff = (
    np.finfo(np.float64).eps * schur_reference_scale * max(q, 1) * 10.0
)
```

3. After Cholesky, reject it when the smallest squared pivot is no larger than
   that cutoff.
4. Use `max(relative_cutoff, absolute_cutoff)` in the SVD fallback.
5. If SVD discards directions, form the null basis from `Vh[~positive].T` and
   reject when `_F @ null_basis` (or its block equivalent) is non-negligible
   relative to `_F`. This prevents invalid congruence-based pseudo-determinants
   and generalized inverses.

- [ ] **Step 9: Run focused tests and verify GREEN**

Run:

```bash
rtk pytest tests/test_structured_factor.py tests/test_random_effect_inference.py tests/test_factor_smooth_structured_parity.py -k "zero_penalty or unpenalized or cancellation or singular_schur"
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit Task 1**

```bash
rtk git add src/superglm/solvers/_structured/selection.py src/superglm/solvers/_structured/factors.py tests/test_structured_factor.py tests/test_random_effect_inference.py tests/test_factor_smooth_structured_parity.py
rtk git commit -m "Guard zero-penalty structured geometry"
```

### Task 2: Preserve compact NB2 automatic-theta fitting

**Files:**

- Modify: `tests/test_structured_irls.py`
- Modify: `tests/test_nb2.py`
- Modify: `src/superglm/solvers/irls_direct.py`
- Modify: `src/superglm/profiling/nb.py`

- [ ] **Step 1: Add a failing low-level auto-fallback test**

Call `fit_irls_direct()` with a `RandomEffectGroupMatrix`,
`direct_solve="auto"`, no `S_override`, and no `reml_penalties`. Assert it
returns a Gram result with a fallback reason containing `compact
reml_penalties`. Keep a forced-structured case asserting the existing clear
`ValueError`.

- [ ] **Step 2: Verify the low-level test fails**

Run:

```bash
rtk pytest tests/test_structured_irls.py -k "missing_compact_penalties"
```

Expected: auto raises the same `ValueError` as forced structured.

- [ ] **Step 3: Add failing public NB2 tests**

Create deterministic overdispersed counts with at least 60 levels and test:

```python
model = SuperGLM(
    family=families.nb2(),
    features={"group": RandomEffect()},
    direct_solve="auto",
).fit_reml(X, y, max_reml_iter=2, runtime_validation="skip")

assert np.isfinite(model.theta_)
assert model.result.direct_backend == "structured"
```

Add a small FS case and compare conditional predictions against forced Gram
within the existing structured parity tolerance.

- [ ] **Step 4: Verify the public tests fail**

Run:

```bash
rtk pytest tests/test_nb2.py -k "credibility and theta"
```

Expected: auto raises `requires compact reml_penalties`.

- [ ] **Step 5: Implement defensive direct-solver fallback**

Immediately after `resolve_structured_backend()` and before constructing the
structured layout, detect:

```python
if (
    structured_decision.use_structured
    and S_override is None
    and reml_penalties is None
):
```

For `auto`, replace the decision with a Gram fallback decision and retain the
dominant group and explicit reason. For forced structured, raise the current
error.

- [ ] **Step 6: Build compact penalty context in NB profiling**

After `dm`, `groups`, and `_use_direct` are known:

```python
reml_penalties = None
if _use_direct:
    from superglm.model.reml_setup import collect_reml_groups
    from superglm.reml.penalty_algebra import build_penalty_context

    reml_groups = collect_reml_groups(groups, dm.group_matrices)
    if reml_groups:
        reml_penalties, _caches, _ranks = build_penalty_context(
            dm.group_matrices,
            reml_groups,
        )
```

Pass `reml_penalties=reml_penalties` to each theta-iteration
`fit_irls_direct()` call. Build the context once, not once per theta.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run:

```bash
rtk pytest tests/test_structured_irls.py -k "missing_compact_penalties"
rtk pytest tests/test_nb2.py -k "credibility and theta"
```

Expected: all selected tests pass, and the public auto model reports the
structured backend.

- [ ] **Step 8: Commit Task 2**

```bash
rtk git add src/superglm/solvers/irls_direct.py src/superglm/profiling/nb.py tests/test_structured_irls.py tests/test_nb2.py
rtk git commit -m "Keep NB2 credibility profiling compact"
```

### Task 3: Replace the unsupported SciPy batch call and add minimum-Python CI

**Files:**

- Modify: `tests/test_sum_to_zero_structured_factor.py`
- Modify: `src/superglm/solvers/sum_to_zero.py`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add deficient-batch parity tests**

Exercise `_decompose_local_psd_batch()` with:

- 300 full-rank `10 x 10` PSD blocks;
- a mixed batch containing exact rank-deficient blocks;
- a scaled `129 x 4 x 4` batch.

For every block, assert the returned rank matches `np.linalg.matrix_rank` under
the solver cutoff, `block @ pinv @ block` reconstructs the block, null vectors
annihilate the block, and the retained log determinant agrees with a direct
`np.linalg.eigh` reference.

- [ ] **Step 2: Run the batch tests before implementation**

Run:

```bash
rtk pytest tests/test_sum_to_zero_structured_factor.py -k "local_psd_batch"
```

Expected: existing numerical tests pass on the current environment; the new
test establishes the invariant before the dependency-compatible swap.

- [ ] **Step 3: Swap the single stacked eigensolve**

Replace:

```python
eigenvalues, eigenvectors = scipy.linalg.eigh(
    symmetric,
    driver="evr",
    check_finite=False,
)
```

with:

```python
eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
```

Do not alter the explicit finite check, threshold, rank, null-space, or
pseudo-inverse logic.

- [ ] **Step 4: Verify numerical parity and Python 3.10 execution**

Run:

```bash
rtk pytest tests/test_sum_to_zero_structured_factor.py -k "local_psd_batch or local_psd"
rtk proxy uv run --python 3.10 pytest tests/test_sum_to_zero_structured_factor.py -q -k "local_psd_batch"
```

Expected: both environments pass.

- [ ] **Step 5: Add the pull-request compatibility trigger**

Add `pull_request` for `master` with the same paths as the existing push
trigger. Use an event-dependent compatibility matrix:

```yaml
matrix:
  python-version: >-
    ${{ fromJSON(
      github.event_name == 'pull_request'
      && '["3.10"]'
      || '["3.10","3.11","3.12","3.14"]'
    ) }}
```

Add `if: github.event_name == 'push'` to coverage shards, combined coverage,
lint, and browser/frontend jobs so PRs run only the Python 3.10 compatibility
job from this workflow. Do not add secrets or write permissions.

- [ ] **Step 6: Validate workflow syntax**

Run:

```bash
rtk proxy pre-commit run check-yaml --files .github/workflows/ci.yml
rtk git diff --check
```

Expected: both checks pass.

- [ ] **Step 7: Commit Task 3**

```bash
rtk git add src/superglm/solvers/sum_to_zero.py tests/test_sum_to_zero_structured_factor.py .github/workflows/ci.yml
rtk git commit -m "Keep SZ compatible with minimum dependencies"
```

### Task 4: Define shape-constraint contracts for structured terms

**Files:**

- Modify: `tests/test_shape_reml.py`
- Modify: `tests/test_shape_postfit.py`
- Modify: `src/superglm/model/fit_ops.py`
- Modify: `src/superglm/model/shape_ops.py`

- [ ] **Step 1: Add failing fit-time preflight tests**

Parameterize RE, FS, and SZ models containing a spline with
`Constraint.fit.increasing`. Assert `fit_reml()` raises before NB profiling,
design fitting, or cloning:

```python
with pytest.raises(
    NotImplementedError,
    match=r"fit-time shape constraints.*RandomEffect|FactorSmooth",
):
    model.fit_reml(X, y)
```

Use a sentinel around `_maybe_estimate_nb_theta` in one NB2 case to prove the
public preflight runs first.

- [ ] **Step 2: Verify fit-time tests fail**

Run:

```bash
rtk pytest tests/test_shape_reml.py -k "structured"
```

Expected: current code enters SCOP EFS and crashes on `omega_ssp=None`.

- [ ] **Step 3: Add failing post-fit repair tests**

For RE, ordinary FS, and SZ, fit a Gaussian model with a deliberately violated
`Constraint.postfit.increasing` population spline. Call
`apply_shape_postfit()` and assert:

- the constrained spline coefficients change;
- repaired predictions satisfy the constraint;
- structured-term predictions remain finite;
- fitted state revision is transactional;
- `penalty_component_dense_matrix()` is never called.

- [ ] **Step 4: Verify post-fit tests fail**

Run:

```bash
rtk pytest tests/test_shape_postfit.py -k "structured"
```

Expected: RE accesses missing `R_inv`; FS/SZ use incompatible matrix shapes.

- [ ] **Step 5: Implement the fit-time preflight**

Add `_reject_structured_fit_constraints(model)` in `fit_ops.py`. Inspect
configured `_SplineBase` specs for a non-null constraint whose mode is
`"fit"`, and inspect configured `RandomEffect` main specs and `FactorSmooth`
interaction specs. If both sets are nonempty, raise a precise
`NotImplementedError` naming the terms.

Call it immediately after `_auto_detect_specs_if_needed()` and before
`_maybe_estimate_nb_theta()` in the REML attempt.

- [ ] **Step 6: Implement compact post-fit penalty terms**

Introduce an internal frozen value object:

```python
@dataclass(frozen=True)
class _CompactPenaltyTerms:
    lambdas: float | dict[str, float]
    penalties: tuple[PenaltyComponent, ...]
    group_matrices: tuple[GroupMatrix, ...]
```

When `_reml_penalties` exists, `_build_smooth_penalty_terms()` returns this
object rather than treating `omega_raw`/`R_inv` as universal. In
`_smooth_penalty_value()`, dispatch this object to
`total_penalty_quadratic()`. Retain the legacy tuple path for ordinary
non-REML fits.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run:

```bash
rtk pytest tests/test_shape_reml.py tests/test_shape_postfit.py -k "structured"
```

Expected: fit-time combinations reject clearly and post-fit combinations
repair successfully without dense structured penalties.

- [ ] **Step 8: Commit Task 4**

```bash
rtk git add src/superglm/model/fit_ops.py src/superglm/model/shape_ops.py tests/test_shape_reml.py tests/test_shape_postfit.py
rtk git commit -m "Define structured shape constraint contracts"
```

### Task 5: Validate authoritative scalar penalty overrides

**Files:**

- Modify: `tests/test_structured_irls.py`
- Modify: `src/superglm/solvers/_structured/assembly.py`

- [ ] **Step 1: Add the failing correlated-override test**

Build a scalar structured problem and an `S_override` whose dominant RE block
contains a symmetric off-diagonal value. Assert:

```python
with pytest.raises(
    ValueError,
    match="S_override.*dominant RandomEffect block.*diagonal",
):
    fit_irls_direct(..., direct_solve="structured", S_override=penalty)
```

Retain a diagonal override control that matches Gram.

- [ ] **Step 2: Verify RED**

Run:

```bash
rtk pytest tests/test_structured_irls.py -k "correlated_override"
```

Expected: structured fitting silently succeeds and disagrees with Gram.

- [ ] **Step 3: Implement full dominant-block validation**

Before adding `np.diag(penalty)` to `d`, extract the dominant block, subtract
its diagonal, and reject any absolute residual above `1e-12`. Keep the existing
cross-partition validation unchanged.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
rtk pytest tests/test_structured_irls.py -k "override"
```

Expected: correlated override rejects and all canonical overrides pass.

- [ ] **Step 5: Commit Task 5**

```bash
rtk git add src/superglm/solvers/_structured/assembly.py tests/test_structured_irls.py
rtk git commit -m "Reject unsupported scalar penalty overrides"
```

### Task 6: Make estimability and covariance selector failures explicit

**Files:**

- Modify: `tests/test_structured_factor.py`
- Modify: `tests/test_random_effect_inference.py`
- Modify: `src/superglm/solvers/_structured/geometry.py`
- Modify: `src/superglm/inference/covariance.py`

- [ ] **Step 1: Add failing estimability failure-contract tests**

Monkeypatch the independent compact certifier to raise:

- `ValueError("contract bug")`: assert the same `ValueError` propagates;
- `np.linalg.LinAlgError("non-convergence")` at width 128: assert exact dense
  fallback is used;
- the same numerical error at width 513: assert a `RuntimeError` naming compact
  estimability certification and preserving the original exception as cause.

- [ ] **Step 2: Verify estimability RED**

Run:

```bash
rtk pytest tests/test_structured_factor.py -k "certification_failure"
```

Expected: `ValueError` is swallowed and the wide case returns all false.

- [ ] **Step 3: Implement the failure contract**

Catch only numerical linear-algebra/ARPACK exceptions. If width is at most the
dense bound, call `_bounded_centered_estimability()`. Otherwise raise:

```python
raise RuntimeError(
    "Compact structured estimability certification failed for a system wider "
    "than the bounded dense fallback; coefficient standard errors cannot be "
    "reported safely."
) from error
```

Do not catch `ValueError`.

- [ ] **Step 4: Add failing boolean-selector tests**

Index both structured covariance accessors with a wrong-length boolean mask and
assert NumPy-compatible `IndexError`. Add a correct-length mask control.

- [ ] **Step 5: Verify selector RED**

Run:

```bash
rtk pytest tests/test_random_effect_inference.py -k "boolean_selector"
```

Expected: the wrong-length mask silently selects positions.

- [ ] **Step 6: Implement boolean mask validation**

In `_selector_indices()`, require boolean selectors to be one-dimensional and
exactly `size` elements before calling `np.flatnonzero`.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run:

```bash
rtk pytest tests/test_structured_factor.py -k "certification_failure"
rtk pytest tests/test_random_effect_inference.py -k "boolean_selector"
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 6**

```bash
rtk git add src/superglm/solvers/_structured/geometry.py src/superglm/inference/covariance.py tests/test_structured_factor.py tests/test_random_effect_inference.py
rtk git commit -m "Expose structured inference contract failures"
```

### Task 7: Apply review cleanup and compatibility documentation

**Files:**

- Modify: `src/superglm/reml/gradient.py`
- Modify: `src/superglm/reml/w_derivatives.py`
- Modify: `examples/fremtpl2_credibility.py`
- Modify: `tests/test_fremtpl2_credibility_demo.py`
- Modify: `tests/test_factor_smooth_mgcv_parity.py`
- Modify: `tests/test_factor_smooth_sz_mgcv_parity.py`
- Modify: `docs/guide/credibility.md`

- [ ] **Step 1: Remove dead and incorrect internal declarations**

Delete `_penalty_component_cross_trace()` and any imports used only by it.
Change:

```python
dH_extra: dict[int, NDArray] = {}
```

to:

```python
dH_extra: dict[int, NDArray | CompactSymmetricOperator] = {}
```

- [ ] **Step 2: Replace private example reads**

Use:

```python
fit_diagnostics = model.diagnostics()["_model"]
```

for REML convergence and iteration count. Select the reporting model from the
known variant dictionary:

```python
re_model = models.get("re_fs") or models.get("re")
fs_model = models.get("re_fs") or models.get("fs")
```

Call the public `random_effects()` and `factor_smooth()` methods directly;
do not inspect `_specs` or `_interaction_specs`.

- [ ] **Step 3: Update the demo test**

Use `model.features` for public main-feature assertions. Test FS construction
through the known variant behavior and public fitted report in the existing
small demo fixture rather than inspecting `_interaction_specs` or
`_direct_solve`.

- [ ] **Step 4: Clarify numerical and product contracts**

In the credibility guide:

- explain that explicit `selection_penalty=0.0` documents the required
  no-selection REML mode and that sparse selection belongs to `fit()` or
  `fit_path()`;
- label the OpenML held-out table as a dated seeded snapshot that must be
  regenerated when dataset/preprocessing versions change.

Add comments beside the broad null-lambda mgcv tolerances explaining that
predictions/deviance are tightly pinned while the optimum is flat in those
penalty coordinates.

- [ ] **Step 5: Run focused checks**

Run:

```bash
rtk pytest tests/test_fremtpl2_credibility_demo.py tests/test_factor_smooth_mgcv_parity.py tests/test_factor_smooth_sz_mgcv_parity.py
rtk ruff check src/superglm/reml/gradient.py src/superglm/reml/w_derivatives.py examples/fremtpl2_credibility.py tests/test_fremtpl2_credibility_demo.py
```

Expected: all tests and lint checks pass.

- [ ] **Step 6: Commit Task 7**

```bash
rtk git add src/superglm/reml/gradient.py src/superglm/reml/w_derivatives.py examples/fremtpl2_credibility.py tests/test_fremtpl2_credibility_demo.py tests/test_factor_smooth_mgcv_parity.py tests/test_factor_smooth_sz_mgcv_parity.py docs/guide/credibility.md
rtk git commit -m "Clarify credibility compatibility contracts"
```

### Task 8: Run regression and release verification

**Files:**

- Modify only if a verified regression requires a focused correction.

- [ ] **Step 1: Run the complete focused review surface**

```bash
rtk pytest tests/test_structured_factor.py tests/test_random_effect_inference.py tests/test_factor_smooth_structured_parity.py tests/test_structured_irls.py tests/test_nb2.py tests/test_sum_to_zero_structured_factor.py tests/test_shape_reml.py tests/test_shape_postfit.py tests/test_fremtpl2_credibility_demo.py
```

Expected: all pass.

- [ ] **Step 2: Run the wider REML and inference surface**

```bash
rtk pytest tests/test_reml.py tests/test_reml_newton_fixes.py tests/test_random_effect_reml.py tests/test_factor_smooth_inference.py tests/test_sum_to_zero_structured_factor.py tests/test_metrics.py
```

Expected: all pass.

- [ ] **Step 3: Run formatting and static checks**

```bash
rtk ruff check src/ tests/ examples/
rtk proxy uv run ruff format --check src/ tests/ examples/
rtk proxy pre-commit run check-yaml --files .github/workflows/ci.yml
rtk git diff --check
```

Expected: all pass.

- [ ] **Step 4: Run the complete test suite**

```bash
rtk pytest tests/
```

Expected: no failures.

- [ ] **Step 5: Build and smoke-test the package**

```bash
rtk proxy uv build
rtk proxy uv run python run_test.py
```

Expected: wheel and source distribution build; smoke test ends with
`END-TO-END COMPLETE`.

- [ ] **Step 6: Update PR compatibility and validation evidence**

Update PR #165's body to include:

- zero-lambda dispatch and Schur certification;
- compact NB2 theta profiling;
- the NumPy SZ batch change with unchanged dependency floors;
- explicit constraint and inference failure contracts;
- discrete-REML trust-region, frozen-pair, line-search, and history semantics;
- exact final-head local verification.

- [ ] **Step 7: Push and inspect exact-head CI**

```bash
rtk git push origin feature/structured-credibility
rtk gh pr checks 165 --repo StrudelDoodleS/superglm --watch
```

Expected: all checks, including the new Python 3.10 PR lane, pass.

- [ ] **Step 8: Request and process exact-head review**

Post a new PR comment tagging `@codex` with attention to the review findings,
wait for the review result using state-based polling, fix any verified issue
test-first, and use the thread-aware GitHub review script to confirm no
unresolved actionable threads remain.

- [ ] **Step 9: Final branch audit**

```bash
rtk git status --short --branch
rtk git rev-parse HEAD
rtk git rev-parse origin/feature/structured-credibility
rtk gh pr view 165 --repo StrudelDoodleS/superglm --json isDraft,mergeable,mergeStateStatus,headRefOid,baseRefOid
```

Expected: clean worktree, local and remote head equal, PR remains draft and
mergeable, and no merge or release action has occurred.
