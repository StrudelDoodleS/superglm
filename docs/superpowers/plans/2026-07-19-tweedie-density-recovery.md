# Tweedie Density Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve PR #158's independently verified Tweedie `p`/`phi` correctness while eliminating fit crashes and restoring ordinary `fit`/`fit_reml` time to the PR #156 envelope.

**Architecture:** Keep the existing Wright-Bessel fast path. Replace the naïve term-1 series walk with bounded mode-centered Dunn-Smyth summation, route infeasible rows to the existing diagnosed saddlepoint approximation, and give post-fit statistics a stricter exact-work budget than likelihood profiling. Use one scaled Pearson implementation everywhere.

**Tech Stack:** Python 3.13, NumPy, SciPy special functions, pytest, cProfile/pstats, GitHub Actions.

---

## File map

- Modify `src/superglm/_tweedie_series.py`: bounded mode-centered compound-Poisson series and exact-row mask.
- Modify `src/superglm/profiling/tweedie.py`: row routing, large-argument `p=1.5` Bessel asymptotic, fit-stat series budget, stable paired likelihood, and shared Pearson contributions.
- Modify `src/superglm/model/fit_ops.py`: stable Tweedie Pearson fit statistic.
- Modify `tests/test_tweedie_numerics.py`: numerical, crash, work-bound, and Pearson regressions.
- Modify `tests/test_tweedie_profile_performance.py`: structural fit-stat budget and benchmark coverage for `p != 1.5`.
- Verify `tests/test_tweedie_profile_reference.py`: neutral joint `p`/`phi` reference.
- Verify `tests/test_tweedie_reml_reference.py`: mgcv/R terminal-scale reference.

### Task 0: Establish clean-room mgcv and saddlepoint baselines

**Files:**
- Document: `docs/superpowers/specs/2026-07-19-tweedie-density-recovery-design.md`

- [x] **Step 1: Inspect official mgcv 1.9-4 behavior and source**

Use the installed package as a black box and the official CRAN source only to
record algorithmic behavior. Do not translate or copy implementation code.
Confirm mode-centered Dunn-Smyth summation, shared buffering, transformed
`p`/`phi` derivatives, and mgcv's explicit index/buffer failure limits.

- [x] **Step 2: Benchmark comparable mgcv fits**

Measure warm medians for fixed- and estimated-power linear and spline fits, then
run mgcv on the shared 800-row neutral profile fixture. Record the environment,
timings, `p`, and `phi` in the design evidence.

- [x] **Step 3: Quantify saddlepoint error**

Compare saddlepoint log density with mgcv exact density over a broad positive
grid and compare exact versus saddlepoint-only known-mean `p`/`phi` estimates.
Use the result to justify the fallback boundary: exact for ordinary profiling,
saddlepoint only when exact work is pathological.

### Task 1: Center exact series work on the contributing mode

**Files:**
- Modify: `src/superglm/_tweedie_series.py`
- Modify: `tests/test_tweedie_numerics.py`

- [ ] **Step 1: Write failing distant-mode and budget tests**

Add imports and tests that require a three-value result `(log_sum, expected_j, exact_mask)`, prove a mode near 90,000 does not require walking 90,000 indices, and prove an astronomically distant mode is rejected before allocation:

```python
import superglm._tweedie_series as series_module


def _log_t_with_series_mode(a: float, mode: int) -> float:
    return float(
        np.log(mode + 1.0)
        + gammaln(a * (mode + 1.0))
        - gammaln(a * mode)
    )


def test_exact_series_starts_near_distant_mode(monkeypatch) -> None:
    calls = 0
    elements = 0
    real_gammaln = series_module.gammaln

    def counted(values):
        nonlocal calls, elements
        calls += 1
        elements += int(np.size(values))
        return real_gammaln(values)

    monkeypatch.setattr(series_module, "gammaln", counted)
    log_sum, expected_j, exact = series_module.tweedie_log_series(
        np.array([_log_t_with_series_mode(1.5, 90_000)]),
        1.5,
    )

    assert exact.tolist() == [True]
    assert np.isfinite(log_sum[0])
    assert expected_j[0] == pytest.approx(90_000.7, rel=2e-9)
    assert calls < 100
    assert elements < 20_000


def test_exact_series_rejects_impossible_work_without_raising() -> None:
    log_sum, expected_j, exact = series_module.tweedie_log_series(
        np.array([70.0]),
        1.5,
        max_total_terms=1_000,
    )

    assert exact.tolist() == [False]
    assert np.isnan(log_sum[0])
    assert np.isnan(expected_j[0])
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
rtk pytest -q tests/test_tweedie_numerics.py::test_exact_series_starts_near_distant_mode tests/test_tweedie_numerics.py::test_exact_series_rejects_impossible_work_without_raising
```

Expected: FAIL because the current function returns two arrays and raises when the mode exceeds index 100,000.

- [ ] **Step 3: Implement bounded mode-centered summation**

Replace the term-1 loop with these units:

```python
_SERIES_RTOL = 5.0e-15
_SERIES_LOG_CUTOFF = 37.0
_SERIES_MAX_TERMS = 100_000
_SERIES_MAX_TOTAL_TERMS = 1_000_000
_SERIES_BATCH_TERMS = 262_144


def _log_series_term(log_t: NDArray, a: float, j: NDArray) -> NDArray:
    j_float = np.asarray(j, dtype=np.float64)
    return j_float * log_t - gammaln(j_float + 1.0) - gammaln(a * j_float)


def tweedie_log_series(
    log_t: NDArray,
    a: float,
    *,
    rtol: float = _SERIES_RTOL,
    max_terms: int = _SERIES_MAX_TERMS,
    max_total_terms: int = _SERIES_MAX_TOTAL_TERMS,
    batch_terms: int = _SERIES_BATCH_TERMS,
) -> tuple[NDArray, NDArray, NDArray]:
    """Return log mass, E[J], and the rows evaluated exactly around their modes."""
    values = np.asarray(log_t, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise FloatingPointError("Tweedie exact series requires finite one-dimensional log(t)")
    if not np.isfinite(a) or a <= 0.0:
        raise FloatingPointError("Tweedie exact series requires finite a > 0")

    log_sum = np.full(values.shape, np.nan, dtype=np.float64)
    expected_j = np.full(values.shape, np.nan, dtype=np.float64)
    exact = np.zeros(values.shape, dtype=np.bool_)
    cutoff = max(_SERIES_LOG_CUTOFF, -float(np.log(rtol)))
    max_mode = ((max_terms - 1.0) / 2.0) ** 2 * (a + 1.0) / (2.0 * cutoff)
    log_mode = (values - a * np.log(a)) / (a + 1.0)
    candidates = np.flatnonzero(log_mode <= np.log(max_mode))
    if candidates.size == 0 or max_total_terms <= 0:
        return log_sum, expected_j, exact

    guesses = np.maximum(1, np.floor(np.exp(log_mode[candidates])).astype(np.int64))
    adjacent = np.stack((np.maximum(1, guesses - 1), guesses, guesses + 1))
    adjacent_log_terms = _log_series_term(values[candidates][None, :], a, adjacent)
    selected = np.argmax(adjacent_log_terms, axis=0)
    modes = adjacent[selected, np.arange(candidates.size)]
    peaks = adjacent_log_terms[selected, np.arange(candidates.size)]
    radii = np.maximum(
        2,
        np.ceil(np.sqrt(2.0 * cutoff * modes / (a + 1.0)) + 2.0).astype(np.int64),
    )
    lower = np.maximum(1, modes - radii)
    upper = modes + radii
    for _ in range(32):
        low_large = (lower > 1) & (
            _log_series_term(values[candidates], a, lower) > peaks - cutoff
        )
        high_large = _log_series_term(values[candidates], a, upper) > peaks - cutoff
        expand = low_large | high_large
        if not np.any(expand):
            break
        radii[expand] *= 2
        lower[expand] = np.maximum(1, modes[expand] - radii[expand])
        upper[expand] = modes[expand] + radii[expand]

    counts = upper - lower + 1
    feasible = counts <= max_terms
    ordered = np.flatnonzero(feasible)[np.argsort(log_mode[candidates][feasible])]
    cumulative = np.cumsum(counts[ordered])
    chosen = ordered[cumulative <= max_total_terms]
    chosen_rows = candidates[chosen]
    exact[chosen_rows] = True

    position = 0
    while position < chosen.size:
        remaining = chosen[position:]
        cumulative = np.cumsum(counts[remaining])
        take = max(1, int(np.searchsorted(cumulative, batch_terms, side="right")))
        local_rows = remaining[:take]
        local_counts = counts[local_rows]
        starts = np.cumsum(local_counts) - local_counts
        total = int(np.sum(local_counts))
        repeated_starts = np.repeat(starts, local_counts)
        j = np.repeat(lower[local_rows], local_counts) + np.arange(total) - repeated_starts
        repeated_rows = np.repeat(local_rows, local_counts)
        relative = np.exp(
            _log_series_term(values[candidates][repeated_rows], a, j)
            - peaks[repeated_rows]
        )
        mass = np.add.reduceat(relative, starts)
        moment = np.add.reduceat(relative * j, starts)
        output_rows = candidates[local_rows]
        log_sum[output_rows] = peaks[local_rows] + np.log(mass)
        expected_j[output_rows] = moment / mass
        position += take

    return log_sum, expected_j, exact
```

During implementation, preserve permutation invariance by resolving equal-work rows using `log_mode`/`log_t`, not original row position. Keep all mode/work checks before `exp(log_mode)` and before ragged allocation.

- [ ] **Step 4: Run the new and existing exact-density/score tests**

Run:

```bash
rtk pytest -q tests/test_tweedie_numerics.py -k 'series or density or score'
```

Expected: PASS, including neutral `p=1.05` density and analytic-score finite difference.

- [ ] **Step 5: Commit Task 1**

```bash
rtk git add src/superglm/_tweedie_series.py tests/test_tweedie_numerics.py
rtk git commit -m "perf: center Tweedie exact series on its mode"
```

### Task 2: Route exact and asymptotic density work without crashes

**Files:**
- Modify: `src/superglm/profiling/tweedie.py`
- Modify: `tests/test_tweedie_numerics.py`
- Modify: `tests/test_tweedie_profile_performance.py`

- [ ] **Step 1: Add failing large-Bessel, near-perfect-fit, and fit-budget tests**

Add:

```python
@pytest.mark.parametrize("p", [1.2, 1.4, 1.5, 1.8])
def test_near_perfect_tweedie_fit_does_not_fail_in_fit_statistics(p: float) -> None:
    x = np.linspace(-1.0, 1.0, 40)
    y = np.exp(0.3 + 0.5 * x)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family=Tweedie(p=p),
        selection_penalty=0,
        features={"x": Numeric()},
    ).fit(frame, y)

    assert np.isfinite(model.result.phi)
    assert np.isfinite(model._fit_stats.log_likelihood)
    assert np.isfinite(model._fit_stats.null_log_likelihood)


def test_p15_large_argument_uses_finite_scaled_asymptotic() -> None:
    y = np.array([1.35])
    mu = np.array([1.3500001])
    weights = np.array([4.0])
    prepared = _prepare_tweedie_density(y, mu, 1.5, weights=weights)
    evaluated = _evaluate_tweedie_density(prepared, 1.0e-14, compute_score=True)

    assert np.isfinite(evaluated.logpdf[0])
    assert evaluated.score_valid
    assert evaluated.diagnostics.n_saddlepoint == 0
```

In the performance test, wrap `tweedie_log_series` and assert `_tweedie_logpdf_pair` passes a strict `max_total_terms` no greater than the fit-stat constant. Use `p=1.4`, not `p=1.5`, so the test cannot pass through the closed-form Bessel branch.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
rtk pytest -q tests/test_tweedie_numerics.py::test_near_perfect_tweedie_fit_does_not_fail_in_fit_statistics tests/test_tweedie_numerics.py::test_p15_large_argument_uses_finite_scaled_asymptotic tests/test_tweedie_profile_performance.py -k 'fit_stat and series'
```

Expected: near-perfect fits raise the current series/Bessel errors; the structural fit-budget assertion fails because no budget is passed.

- [ ] **Step 3: Implement row-level routing and the Bessel asymptotic**

Add internal policy constants:

```python
_PROFILE_SERIES_MAX_TOTAL_TERMS = 1_000_000
_FIT_STATS_SERIES_MAX_TOTAL_TERMS = 4_096
```

Extend `_evaluate_tweedie_density` with `series_max_total_terms`, unpack the series exact mask, and leave rejected rows for the existing saddlepoint assignment:

```python
series_log_sum, expected_j, series_success = tweedie_log_series(
    log_t[series_candidates],
    prepared.a,
    max_total_terms=series_max_total_terms,
)
candidate_indices = np.flatnonzero(series_candidates)
series_indices = candidate_indices[series_success]
use_series[series_indices] = True
```

For failed `ive` calls at finite positive argument `z`, use:

```python
inverse_z = 1.0 / z
correction = (
    -3.0 * inverse_z / 8.0
    - 15.0 * inverse_z**2 / 128.0
    - 315.0 * inverse_z**3 / 3072.0
)
log_scaled_i1 = -0.5 * (np.log(2.0 * np.pi) + np.log(z)) + np.log1p(correction)
score_component = (
    0.5
    + 3.0 * inverse_z / 8.0
    + 3.0 * inverse_z**2 / 8.0
    + 63.0 * inverse_z**3 / 128.0
)
```

Use the direct SciPy values where finite and these expansions only outside SciPy's range. Any remaining non-finite candidate becomes saddlepoint instead of raising.

- [ ] **Step 4: Make paired fit statistics bounded and stable**

Call `_evaluate_tweedie_density` from `_tweedie_logpdf_pair` with `_FIT_STATS_SERIES_MAX_TOTAL_TERMS`, remove the exact-only exception, and compute the null likelihood by the deviance identity:

```python
fit_deviance = _tweedie_positive_unit_deviance(prepared.y, prepared.mu, prepared.p)
null_deviance = _tweedie_positive_unit_deviance(prepared.y, null_array, prepared.p)
null_logpdf = evaluation.logpdf - (
    0.5 * prepared.weights / phi * (null_deviance - fit_deviance)
)
```

This identity is valid for both the exact and saddlepoint exponential-family base measures and avoids subtracting/re-adding huge canonical terms.

- [ ] **Step 5: Verify GREEN and the original crash manually**

Run the focused tests, followed by the original 40-row fit reproduction across powers. Expected: all fits complete with finite statistics; no series-cap or Bessel exception.

- [ ] **Step 6: Commit Task 2**

```bash
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_numerics.py tests/test_tweedie_profile_performance.py
rtk git commit -m "fix: bound Tweedie density work without fit crashes"
```

### Task 3: Remove Pearson underflow and duplicate floor behavior

**Files:**
- Modify: `src/superglm/profiling/tweedie.py`
- Modify: `src/superglm/model/fit_ops.py`
- Modify: `tests/test_tweedie_numerics.py`

- [ ] **Step 1: Add failing public and profile Pearson tests**

```python
def test_pearson_phi_is_zero_for_equal_subnormal_scale_values() -> None:
    value = np.array([1.0e-300])
    assert estimate_phi(value, value, 1.5) == 0.0


def test_pearson_phi_preserves_finite_subnormal_scale_residual() -> None:
    mu = np.array([1.0e-300])
    y = np.array([1.0e-300 + 1.0e-310])
    expected = float(np.square((y - mu) / np.power(mu, 0.75))[0])
    assert estimate_phi(y, mu, 1.5) == pytest.approx(expected, rel=2e-15)


def test_profile_pearson_uses_same_unfloored_contributions() -> None:
    y = np.array([1.0e-12, 2.0e-12])
    mu = np.array([1.0e-20, 2.0e-20])
    expected = estimate_phi(y, mu, 1.5)
    actual = _profile_phi_detailed(y, mu, 1.5, phi_method="pearson")
    assert actual.phi == pytest.approx(expected, rel=2e-15)
```

- [ ] **Step 2: Run and verify RED**

Expected: public subnormal tests raise `FloatingPointError`; the profile test returns approximately `2.5e-9` instead of `1.207e6`.

- [ ] **Step 3: Add one scaled Pearson helper and use it everywhere**

```python
def _tweedie_pearson_contributions(y: NDArray, mu: NDArray, p: float) -> NDArray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return np.square((y - mu) / np.power(mu, 0.5 * p))
```

Use it in `estimate_phi`, `_pearson_phi_from_prepared`, and the Tweedie branch of `_compute_fit_stats`. Remove the `1e-10` private floor.

- [ ] **Step 4: Verify GREEN and commit Task 3**

```bash
rtk pytest -q tests/test_tweedie_numerics.py -k pearson
rtk git add src/superglm/profiling/tweedie.py src/superglm/model/fit_ops.py tests/test_tweedie_numerics.py
rtk git commit -m "fix: stabilize Tweedie Pearson scaling"
```

### Task 4: Prove reliable profiling and tune only from evidence

**Files:**
- Modify only if a measured bound requires tuning: `src/superglm/profiling/tweedie.py`, `src/superglm/_tweedie_series.py`
- Verify: `tests/test_tweedie_profile_reference.py`
- Verify: `tests/test_tweedie_reml_reference.py`
- Verify: `tests/test_tweedie_profile_performance.py`

- [ ] **Step 1: Run neutral correctness references**

```bash
rtk pytest -q tests/test_tweedie_profile_reference.py tests/test_tweedie_reml_reference.py tests/test_tweedie_numerics.py
```

Required: joint profile remains within `abs(p)=2e-4`, `rel(phi)=5e-4`; REML terminal scale stays within 2% of the R/mgcv reference; all density references pass.

- [ ] **Step 2: Re-run the measured profile and ordinary-fit cProfiles**

Use the exact scripts from the diagnosis:

- 1,000-row `p=1.4` ordinary fit with 2.5% log noise;
- 300-row fixed-`mu`, `p=1.4` MLE dispersion profile;
- one-row modes 10,000 and 90,000;
- 40-row near-perfect fits across `p=1.2,1.4,1.5,1.8`.

Measure seven warm runs and compare medians. Hard gates:

- ordinary fit is at most `max(1.5 * base, base + 0.005 seconds)` against the
  same-process PR #156 baseline and is not dominated by post-fit density;
- the 300-row MLE profile completes within 0.25 seconds locally, remains inside
  its neutral-reference tolerance, and never enters an unbounded lower-`phi`
  sum;
- a one-row mode near 90,000 completes within 0.02 seconds locally.

The provisional 65,536-term fit-stat budget missed the ordinary-fit gate.
Measured warm medians on the 1,000-row reproduction were 0.0122 seconds at
65,536 terms, 0.0084 seconds at 4,096 terms, and 0.0052 seconds with no exact
series work, versus 0.0048 seconds on PR #156. Keep the measured 4,096-term
ceiling unless later evidence shows a correctness failure.

Also report the shared 800-row profile beside mgcv's 0.023-second clean-room
median. mgcv is a directional performance reference, not an equality gate for
this bounded recovery patch; any remaining multiple must be explained by a
profile rather than hidden by the smaller 300-row gate.

Measured recovery results:

- 1,000-row noisy ordinary fit: about 0.0071 seconds versus PR #156's 0.0048;
- routine simulated 1,000-row fit: about 0.0126 seconds versus PR #156's 0.0143;
- 300-row small-dispersion exact profile: about 0.16 seconds, down from 1.58;
- one-row mode 90,000 exact series: below 0.001 seconds;
- shared 800-row neutral profile: about 0.21 seconds, down from 1.42, versus
  mgcv's 0.023 seconds.

The 800-row profile remains exact and inside the neutral `p`/`phi` tolerances.
Its remaining gap to mgcv is architectural: the current public profile performs
about 277 prepared density passes across nested scalar `p` and `phi` searches,
whereas mgcv optimizes transformed parameters jointly with density derivatives.
Do not expand this recovery patch into a profiler rewrite.

- [ ] **Step 3: Tune only the two work budgets if evidence requires it**

Keep the fit-stat budget strict enough that post-fit likelihood is not the dominant ordinary-fit frame. Increase the profile budget only if an independent `p`/`phi` reference fails because a materially biased saddlepoint row was selected. Do not add a public option.

- [ ] **Step 4: Run focused performance tests and commit any measured tuning**

```bash
rtk pytest -q tests/test_tweedie_profile_performance.py
rtk git add src/superglm/_tweedie_series.py src/superglm/profiling/tweedie.py tests/test_tweedie_profile_performance.py
rtk git commit -m "perf: tune bounded Tweedie profile work"
```

Skip the commit if no tuning changes are needed.

### Task 5: Full verification and PR update

**Files:**
- Verify all modified source and test files.

- [ ] **Step 1: Run formatting and static checks**

```bash
rtk ruff check src/superglm/_tweedie_series.py src/superglm/profiling/tweedie.py src/superglm/model/fit_ops.py tests/test_tweedie_numerics.py tests/test_tweedie_profile_performance.py
rtk ruff format --check src/superglm/_tweedie_series.py src/superglm/profiling/tweedie.py src/superglm/model/fit_ops.py tests/test_tweedie_numerics.py tests/test_tweedie_profile_performance.py
```

- [ ] **Step 2: Run the focused Tweedie suite**

```bash
rtk pytest -q tests/test_tweedie_numerics.py tests/test_tweedie_profile.py tests/test_tweedie_profile_reference.py tests/test_tweedie_reml_reference.py tests/test_tweedie_profile_performance.py
```

- [ ] **Step 3: Run the complete test suite**

```bash
rtk pytest -q
```

Required: zero failures; inspect warnings rather than relying only on the exit code.

- [ ] **Step 4: Re-run original black-box reproductions after the full suite**

Required evidence:

- no crash for valid near-perfect fits;
- neutral joint profile remains `p≈1.196897`, `phi≈0.806814`;
- R/mgcv terminal `phi≈0.374165`;
- ordinary fit timing remains within the accepted PR #156 envelope;
- public Pearson subnormal cases return finite mathematical results.

- [ ] **Step 5: Inspect the final diff and push**

```bash
rtk git diff --check 3656b50e94ad7713dc6ec4b7d7d4a2c4d3a1cd76..HEAD
rtk git status --short --branch
rtk git push
rtk gh pr checks 158
```

Do not mark the goal complete until all requirements have fresh evidence and PR checks are green.
