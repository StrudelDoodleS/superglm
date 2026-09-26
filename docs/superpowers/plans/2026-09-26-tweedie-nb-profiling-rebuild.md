# Tweedie and NB2 Profiling Rebuild Implementation Plan

> **For agentic workers:** execute through Workflow scripts. Max chose this method: few agents, with effort set explicitly per agent (see "Execution"). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ~10,200-line Tweedie/NB2 estimation cluster with one compiled Dunn–Smyth series, one φ solver, one Brent p search and one likelihood-ratio interval (≤ 3,000 source lines), recovering p at least as well and running faster.

**Architecture:** `_tweedie_series.py` (compiled per-row series) → `_tweedie.py` (positive-row state `TweedieRows`, density, deviance, the φ solver `solve_log_phi`, simulation) → `profiling/_scalar.py` (a recorded bounded search and the LR interval, shared by p and θ) → `profiling/tweedie.py` and `profiling/nb.py` (estimators and result objects) → `model/profile_ops.py` (publication). `reml/scale.py` and `distributions.py` depend only on `_tweedie`.

**Tech Stack:** Python 3.12–3.14, NumPy ≥ 2.3, SciPy, numba (`@njit(cache=True)`), pandas, pytest with xdist, uv.

**Spec:** `docs/superpowers/specs/2026-09-26-tweedie-nb-profiling-rebuild-design.md`. Read it before any task. It holds the method citations, the tolerances derivation and success criteria 1–9.

## Global Constraints

- Production numerics are portable IEEE float64. No `longdouble`, `float96` or `float128`. Higher precision appears only in test oracles and fixture generators.
- Internal source lines are justified only by complete-fit time or by deleting complexity. Tests may be verbose. Source and test lines are reported separately.
- CODE SHAPE (Max, standing):
  1. No guard that cannot fire or that repeats a boundary check.
  2. No nested loops in Python-level code. A nested loop is allowed only inside a compiled kernel whose algorithm is a nested loop, and a comment must say so.
  3. Flat control flow: early returns, helpers of about 40 lines or fewer, nesting depth at most 3.
  4. No speculative configurability and no flags nobody sets.
  5. Names say what the thing is.
  6. Comments say WHY, never narrate.
  7. No O(n) pass repeated inside a loop over candidates when one pass can serve.
  8. The diff is as small as the behaviour requires.
- Edit existing files surgically (the Edit tool or in-place replacement). Whole-file writes are only for new files, and for `profiling/tweedie.py` / `profiling/nb.py` in Tasks 6 and 8, which are replaced wholesale.
- Every mgcv oracle test passes with its tolerances unchanged:
  - `tests/test_tweedie_reml_exact_scale.py`
  - `tests/test_nb_theta_estimation_correctness.py`
  - `tests/test_tweedie_lss_kernel.py`
  - `tests/test_negative_binomial_lss_kernel.py`
- No wall-clock assertions in tests. Assert counts, envelopes and mathematics.
- Tolerances derive from float64 error analysis or from a recorded measurement, never from what passed locally. Any loosened check carries its derivation in a comment.
- The private validation dataset may be read, fitted and timed, but never named, pathed or described by feature names in any committed file, commit message, comment or PR text. Its results live in the session scratchpad only.
- Feature PRs never touch `pyproject.toml`'s version, `superglm.__version__` or `uv.lock`'s own version pin. The PR declares `release:minor`.
- Git:
  - Never `git stash`.
  - Never `cd` into `/home/max/projects/superglm` (another session's checkout). Work only in this worktree.
  - Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. When the agent's model is Fable, use `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` instead.
  - `docs/superpowers/` is gitignored, so use `git add -f` there.
- Tests:
  - Each task runs its focused tests with `uv run pytest <files> -n 8 -q`.
  - The full suite runs once, in Task 10: `uv run python scripts/run_test_suite.py`.
  - Plotly-dependent tests need `uv run --with plotly`.
- Timing runs pin all six thread pools. Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1`, report process CPU and wall, and interleave the before and after runs.

## Review Focus

1. **A candidate power at which the response has no positive rows, or D = 0.** Expect a `ValueError` naming the cause before any Newton step, not a hang or NaN. Pinned in Task 3 (`test_solve_log_phi_refuses_no_interior_optimum`).
2. **A profile optimum at a p bound** (rounded data make a spurious maximum as p→1, per Dunn & Smyth 2005). Expect `p_hat` at the bound, a warning in `result.warnings`, and the CI side censored; no exception. Pinned in Task 6 (`test_boundary_optimum_is_reported_not_raised`).
3. **Extreme rows: large y with tiny φ near p = 1.95** (peak index around 1e9 or beyond). Expect a finite log W, or a `FloatingPointError` naming p, φ and the row count; never NaN. Pinned in Task 2 (`test_rows_past_the_work_bound_are_refused_not_nan`).
4. **A REML candidate power whose penalized mode cannot be certified.** Expect it to be skipped as infeasible, the search to continue, and a CI side that ends against it to be reported censored. Pinned in Task 5 (`test_interval_side_ending_at_infeasible_region_is_censored`) and Task 6 (`test_uncertifiable_power_is_skipped`).
5. **Frequency weight semantics.** REML-scale replication counts must equal literally replicated rows, and `estimate_p` must refuse non-unit frequency weights. Pinned in Task 3 (`test_frequency_counts_equal_replicated_rows`) and Task 7 (`test_estimate_p_refuses_nonunit_frequency_weights`).

---

## File map

| Path | Action | Responsibility |
|---|---|---|
| `src/superglm/_tweedie_series.py` | Create (Task 2) | Compiled per-row Dunn–Smyth series and warmup |
| `src/superglm/_tweedie.py` | Create (Task 3) | `TweedieRows`, `tweedie_logpdf`, `tweedie_logpdf_pair`, `tweedie_unit_deviance`, `solve_log_phi`, `generate_tweedie_cpg` |
| `src/superglm/_tweedie_profile_kernel.py` | Delete (Task 9) | Replaced by `_tweedie_series.py` |
| `src/superglm/distributions.py` | Modify (Task 4) | Import from `_tweedie` |
| `src/superglm/stats/model_tests.py` | Modify (Task 4) | Import from `_tweedie` |
| `src/superglm/model/fit_ops.py` | Modify (Tasks 4, 6, 8) | Fit stats via `tweedie_logpdf_pair`; warm start in `_solve_coefficients`; NB alternation |
| `src/superglm/reml/scale.py` | Modify (Task 4) | Tweedie part on `TweedieRows` + `solve_log_phi` |
| `src/superglm/reml/objective.py` | Modify (Task 4) | Type import `TweedieRows` |
| `src/superglm/profiling/_scalar.py` | Create (Task 5) | `RecordedObjective`, `minimize_profile`, `Interval`, `likelihood_ratio_interval`, `profile_plot` |
| `src/superglm/profiling/tweedie.py` | Replace (Task 6) | `_PowerProfile`, `search_power`, `TweedieProfileResult` |
| `src/superglm/profiling/nb.py` | Replace (Task 8) | θ score and solve, `estimate_nb_theta` (private use), `NBProfileResult` |
| `src/superglm/profiling/_reporting.py` | Modify (Task 7) | Summary labels for the slim results |
| `src/superglm/profiling/__init__.py` | Modify (Task 9) | Exports |
| `src/superglm/profiling/harness.py` | Move (Task 9) | → `benchmarks/_harness.py` |
| `src/superglm/model/profile_ops.py` | Modify (Tasks 7, 8) | Publication for p and θ through one helper |
| `src/superglm/model/api.py` | Modify (Tasks 7, 8) | `estimate_p` / `estimate_theta` signatures |
| `src/superglm/model/report_ops.py`, `src/superglm/inference/metrics.py` | Modify (Task 7) | Summary fields |
| `src/superglm/editor/{widget.py,server.py,app/index.html,app/summary.js}` | Modify (Task 7) | Drop the method pickers; read `evaluations` |
| `src/superglm/__init__.py` | Modify (Task 9) | Exports; `warmup()` |
| `benchmarks/tweedie_nb_characterisation.py` | Create (Task 1) | Fixture generator run on master code |
| `benchmarks/tweedie_series_oracle.py` | Create (Task 2) | 50-digit oracle generator (`uv run --with mpmath`) |
| `benchmarks/nb_auto_theta_reml.py` | Create (Task 1) | NB auto-θ timing case |
| `benchmarks/tweedie_nb_rebuild_receipts.json` | Create (Task 1), update (Task 10) | Before/after performance receipts |
| `tests/fixtures/tweedie_nb_characterisation.json` | Create (Task 1) | Master-code outputs |
| `tests/fixtures/tweedie_series_oracle.json` | Create (Task 2) | 50-digit log W values |
| `tests/test_tweedie_series.py` | Create (Task 2) | Kernel oracle and work-bound tests |
| `tests/test_tweedie_density.py` | Create (Task 3) | Density, pair, solver and simulation tests |
| `tests/test_tweedie_nb_characterisation.py` | Create (Task 3; extended in 6, 8) | New vs master fixtures |
| `tests/test_profile_scalar.py` | Create (Task 5) | Search and interval tests |
| `tests/test_tweedie_p_recovery.py` | Create (Task 6) | Criterion 8 |
| Old tests pinning removed internals | Delete or rewrite (Task 9) | See Task 9 |

## Execution

Two workflow runs; each run keeps within Max's cap of about 8 agents. Every agent name carries model and effort. Every `agent()` passes `model` and `effort` explicitly. The session model is Opus 5.5. Fable only where named.

| Run | Agent (label) | Tasks | Model / effort |
|---|---|---|---|
| A | `baseline_opus_xhigh` | 1 | opus / xhigh |
| A | `kernel_phi_fable_max` | 2, 3 | fable / max (the single hardest numerics stage) |
| A | `switch_opus_xhigh` | 4 | opus / xhigh |
| A | `scalar_power_opus_xhigh` | 5, 6 | opus / xhigh |
| A | `publish_editor_opus_xhigh` | 7 | opus / xhigh |
| A | `critic_a_opus_xhigh` | reviews the tree after Task 7 against spec and plan | opus / xhigh |
| A | `repair_a_opus_xhigh` | only if the critic finds defects | opus / xhigh |
| B | `nb_opus_xhigh` | 8 | opus / xhigh |
| B | `delete_docs_opus_xhigh` | 9 | opus / xhigh |
| B | `evidence_opus_xhigh` | 10 | opus / xhigh |
| B | `critic_b_opus_xhigh` | reviews the whole branch | opus / xhigh |
| B | `lss_gate_opus_xhigh` | 11 (gated) | opus / xhigh |

- Agents run sequentially: every task consumes the previous task's interfaces.
- The main session consolidates results and owns the final answer.
- Run A stops after its critic. The main session reports Task 1's stage-0 decisions and the critic verdict to Max before Run B starts.
- Every agent prompt carries three items:
  - this plan's Global Constraints;
  - the CODE SHAPE block;
  - the private-dataset path and model spec. Only the main session knows these, and they go in prompts only.

---

### Task 1: Baseline fixtures, receipts and stage 0 measurements

**Files:**
- Create: `benchmarks/tweedie_nb_characterisation.py`, `benchmarks/nb_auto_theta_reml.py`, `tests/fixtures/tweedie_nb_characterisation.json`, `benchmarks/tweedie_nb_rebuild_receipts.json`
- Read-only reference: `benchmarks/tweedie_profile_end_to_end.py`, `benchmarks/profile_tweedie_reml_fit.py`, `benchmarks/tweedie_reml_search_cost.py`, `tests/test_tweedie_reml_exact_scale.py` (model construction for the `re_*.csv` fixtures), `tests/test_nb_theta_estimation_correctness.py` (the NB fixtures)

**Interfaces:**
- Consumes: the unmodified master code (`superglm` at 9e0fe6f9 plus the spec commit).
- Produces:
  - A fixture JSON with the schema below, consumed by Tasks 3, 6 and 8.
  - A receipts JSON consumed by Task 10.
  - Four recorded decisions (a)–(d) in the receipts under `"decisions"`.

Fixture schema (all floats are JSON numbers written with `repr` precision):

```json
{
  "provenance": {"master_sha": "...", "script": "benchmarks/tweedie_nb_characterisation.py", "generated": "YYYY-MM-DD"},
  "old_route_max_rel_error": 0.0,
  "logpdf": [{"y": 0.0, "mu": 0.0, "phi": 0.0, "p": 0.0, "w": 1.0, "logpdf": 0.0, "saddlepoint": false}],
  "reml_phi": [{"case": "re_ident_p15", "p": 1.5, "phi": 0.0}],
  "estimate_p": [{"case": "re_ident_p15", "fit_mode": "fit", "p_hat": 0.0, "phi_hat": 0.0, "nll": 0.0,
                  "search_nll": 0.0, "ci95": [0.0, 0.0], "n_evaluations": 0}],
  "estimate_theta": [{"case": "nb_worst", "fit_mode": "fit", "theta_hat": 0.0, "nll": 0.0, "ci95": [0.0, 0.0]}]
}
```

- [x] **Step 1: Write the characterisation generator.** In `benchmarks/tweedie_nb_characterisation.py`:
  - **`logpdf` grid:**
    - Axes: p ∈ {1.01, 1.05, 1.1, 1.3, 1.5, 1.7, 1.9, 1.95, 1.99}; φ ∈ {1e-3, 1e-2, 0.1, 1, 10, 100}; y ∈ {0, 1e-3, 0.1, 1, 10, 1e3}; μ ∈ {0.5·y, y, 2·y}, with μ = 1 when y = 0; w ∈ {1, 3.5}.
    - Evaluate through `superglm.profiling.tweedie._prepare_tweedie_density` / `_evaluate_tweedie_density`, one row at a time, and record `saddlepoint = bool(evaluation.positive_saddlepoint_mask[0])` for positive rows.
    - Skip rows whose `tweedie_logpdf` raises, and record them under `"logpdf_refused"`.
  - **`old_route_max_rel_error`:**
    - Recompute log W for every non-saddlepoint positive row with mpmath at 50 digits (the reference sum in Task 2, Step 1) and take max |old − ref| / max(1, |logpdf|).
    - Rows whose peak index exceeds 3e5 are left out (mpmath cost), and the count left out is recorded.
    - This part runs only under `uv run --with mpmath`.
  - **`reml_phi`:** for each `tests/fixtures/re_*.csv`, build the model exactly as `tests/test_tweedie_reml_exact_scale.py` does, `fit_reml` at the file's power, and record `model.result.phi`.
  - **`estimate_p` cases:**
    - The three `re_*.csv` fixtures.
    - A synthetic book `zeros90`: n = 30,000, about 90% zeros.
    - A synthetic book `positive96`: n = 30,000, about 96% positive.
    - An unpenalized design `unpen`: categorical features only, `spline_penalty=None`.
    - Both synthetic books use `generate_tweedie_cpg` with seed 20260926, a `Spline(n_knots=10)` on one uniform covariate and a 5-level categorical.
    - Run each case under `fit_mode="fit"` and `"reml"` with default arguments and `ci_alpha=0.05`. Record `search_nll` from `result.search_nll`, or from `result.nll` when that is `None`, and `n_evaluations` from `len(result.search_trace)`.
  - **`estimate_theta`:** `nb_clamp005.csv` and `nb_worst.csv` built as in `tests/test_nb_theta_estimation_correctness.py`, plus a Poisson-limit case (`generate` Poisson counts, n = 5,000, seed 7). Run under `fit` and `reml`, and record `result.ci(0.05)`.
  - Write the JSON with `json.dump(..., indent=1)`.

- [x] **Step 2: Generate the fixture on master code.** Run: `uv run --with mpmath python benchmarks/tweedie_nb_characterisation.py --out tests/fixtures/tweedie_nb_characterisation.json`. Expected: the file is written; `"logpdf"` has at least 1,500 rows; `old_route_max_rel_error` is finite.

- [x] **Step 3: Run the private dataset through the same script into scratch.** Build the model from the real-book spec given in your prompt, run `estimate_p` under reml with default arguments, and write to `$SCRATCH/private_characterisation.json`. Never write it under the repository.

- [x] **Step 4: Write `benchmarks/nb_auto_theta_reml.py`.** Replicate the `nb_worst.csv` design to 200,000 rows (np.tile, seed-fixed jitter on the continuous column), fit `families.nb2()` (auto θ) with one spline and one factor under `fit_reml`, and print wall time, CPU time, peak RSS (`resource.getrusage`), θ̂, refit count and the REML iteration count.

- [x] **Step 5: Baseline receipts and the time breakdown.** With the six thread pools pinned, run each §9 case from the spec three times interleaved. The cases are:
  - `benchmarks/tweedie_profile_end_to_end.py` on the `zeros90` and `positive96` shapes, under fit and under reml;
  - `benchmarks/profile_tweedie_reml_fit.py`;
  - `benchmarks/tweedie_reml_search_cost.py`;
  - `benchmarks/nb_auto_theta_reml.py`;
  - the private dataset under reml (scratch only).
  
  Record per case: median wall, CPU, peak RSS, candidate-fit count, and the backend dispatched (the `result.search_trace` length and `model._last_fit_meta`).
  
  Then, in a separate profiling run (`cProfile`, not the timing run), attribute estimate_p time to three buckets:
  - candidate fits (`fit_pirls`, `fit_irls_direct` and `fit_reml` cumulative);
  - φ/density (`_profile_phi_detailed` and `_evaluate_tweedie_density` cumulative);
  - everything else.
  
  Write all of it to `benchmarks/tweedie_nb_rebuild_receipts.json` under `"baseline"`. Private-dataset numbers go only to scratch.

- [x] **Step 6: The four stage 0 measurements. Record each under `"decisions"`.**
  - **(a) Series refusals.** Instrument by calling `series_moments(log_t, a)` from `_tweedie_profile_kernel` directly:
    - rows: every positive row of every fixture case, the private dataset and the `re_*.csv` files;
    - p ∈ {1.05, 1.2, 1.5, 1.8, 1.95};
    - φ ∈ {φ̂/1e3, φ̂/10, φ̂, 10·φ̂, 1e3·φ̂}, where φ̂ is the Pearson estimate.
    
    Record the largest per-row term count needed, 2·sqrt(2·37·mode/(a+1)), and how many rows exceed 1e5 and 1e6 terms. Decision: if no row exceeds 1e6, the kernel's `_MAX_ROW_TERMS = 1_000_000` stands and refusal is an error. Otherwise record the (p, φ) where refusal happens and stop for Max.
  - **(b) `search_fit_mode` gain.** Time `estimate_p(fit_mode="reml", search_fit_mode="fit")` against `estimate_p(fit_mode="reml")` on the `positive96` shape and the private dataset. Decision: keep the parameter if and only if the speed-up is ≥ 1.5× on both.
  - **(c) p = 1.5 Bessel vs series inside `fit_reml`.** Time `benchmarks/profile_tweedie_reml_fit.py` at p = 1.5 as-is, then with `TweedieScaleProfileData._bessel_saturated_log_likelihood` and `_bessel_saturated_score` monkeypatched to return `None` (forcing the series). Decision: keep a Bessel path if and only if the complete fit is ≥ 1.1× faster with it.
  - **(d) `joint_ml` vs Brent on `unpen`.** Time `estimate_p(method="joint_ml")` against `method="brent"`. This is informational only: record the regression that removing `joint_ml` accepts.

- [x] **Step 7: Commit.**

```bash
git add benchmarks/tweedie_nb_characterisation.py benchmarks/nb_auto_theta_reml.py benchmarks/tweedie_nb_rebuild_receipts.json tests/fixtures/tweedie_nb_characterisation.json
git commit -m "bench: characterise the Tweedie/NB2 profilers before the rebuild

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

**Stage 0 outcome** (measured on master code; numbers in `benchmarks/tweedie_nb_rebuild_receipts.json` under `"decisions"`):

- **(a) Series refusals: none.** The largest per-row work on the probed (p, φ) grid is 3,415 terms (positive96, p = 1.95, φ̂/10³); the logpdf grid's largest is 10,483. No row passes 10⁵ terms or a peak index of 2⁵². `MAX_ROW_TERMS = 1_000_000` stands and a refusal is an error. In the suite's Tweedie tests, rows past 10⁶ terms occur only at the old φ search's lower bound 1e-12 and in adversarial tests of deleted internals, with one behavioural exception for Tasks 3–4: `test_tweedie_numerics.py::test_near_perfect_tweedie_fit_does_not_fail_in_fit_statistics` (exact-curve data, Pearson φ ≈ 2.6e-26, every row's peak index past 2⁵², evaluated by saddlepoint on master).
- **(b) `search_fit_mode` stays** (spec §6): the decoupled search is 2.9× faster on positive96 and 3.1× on the REML search-cost benchmark, with p̂ moving by 2e-6 and 1e-6; the private validation dataset agrees.
- **(c) No Bessel path in `fit_reml`.** The REML-scale Bessel methods ran 0 times in complete p = 1.5 fits (the Newton solve uses the series; only the bounded fallback reaches them), and disabling them changes wall time by noise (0.96–0.98×).
- **(d) `joint_ml` removal costs 2.3× on `unpen`** at master's φ cost (0.18 s vs 0.42 s; 4 vs 11 candidate fits; p̂ within 3.3e-5). Informational; Task 10 re-measures Brent there.

---

### Task 2: Compiled per-row series kernel

**Files:**
- Create: `src/superglm/_tweedie_series.py`, `benchmarks/tweedie_series_oracle.py`, `tests/fixtures/tweedie_series_oracle.json`, `tests/test_tweedie_series.py`

**Interfaces:**
- Produces:
  - `series_moments(log_t: NDArray[float64], a: float) -> tuple[NDArray[bool], NDArray, NDArray, NDArray]`. It returns `(ok, log_w, mean_j, var_j)`: per row, log W(t) = log Σ_{j≥1} t^j / (j! Γ(a j)), and the mean and variance of j under the normalised terms. Rows are independent.
  - `warmup() -> None`.
  - Module constant `MAX_ROW_TERMS: int = 1_000_000`.

- [x] **Step 1: Write the oracle generator** `benchmarks/tweedie_series_oracle.py` (run with `uv run --with mpmath`):

```python
"""50-digit reference values of the Dunn-Smyth series log W(t) = log sum_j t^j / (j! Gamma(a j))."""

import json
import math
import sys

import mpmath as mp

mp.mp.dps = 50


def reference_log_w(log_t: float, a: float) -> tuple[str, int]:
    lt, am = mp.mpf(log_t), mp.mpf(a)

    def term(j: int):
        return j * lt - mp.loggamma(j + 1) - mp.loggamma(am * j)

    mode = max(1, int(math.exp((log_t - a * math.log(a)) / (a + 1))))
    peak = max(term(j) for j in range(max(1, mode - 2), mode + 3))
    total = mp.mpf(0)
    for step in (1, -1):
        j = mode if step == 1 else mode - 1
        while j >= 1:
            q = term(j)
            total += mp.exp(q - peak)
            if q < peak - 120:
                break
            j += step
    return mp.nstr(peak + mp.log(total), 40), mode


rows = []
for p in (1.01, 1.05, 1.1, 1.3, 1.5, 1.7, 1.9, 1.95, 1.99):
    a = (2 - p) / (p - 1)
    for phi in (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0):
        for y in (1e-3, 0.1, 1.0, 10.0, 1e3):
            log_t = a * (math.log(y) - math.log(p - 1)) - math.log(2 - p) - (a + 1) * math.log(phi)
            mode = math.exp((log_t - a * math.log(a)) / (a + 1))
            if mode > 3e5:
                continue
            value, mode = reference_log_w(log_t, a)
            rows.append({"p": p, "a": a, "log_t": log_t, "mode": mode, "log_w": value})
json.dump({"generator": "benchmarks/tweedie_series_oracle.py", "digits": 50, "rows": rows},
          open(sys.argv[1], "w"), indent=1)
```

Run: `uv run --with mpmath python benchmarks/tweedie_series_oracle.py tests/fixtures/tweedie_series_oracle.json`. Expected: about 237 rows.

- [x] **Step 2: Write the failing tests** in `tests/test_tweedie_series.py`:

```python
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import i1e

from superglm._tweedie_series import MAX_ROW_TERMS, series_moments

ORACLE = json.loads((Path(__file__).parent / "fixtures" / "tweedie_series_oracle.json").read_text())
EPS = np.finfo(np.float64).eps


def _peak_magnitude(log_t: float, a: float, mode: int) -> float:
    # |q(j_max)| components: the float64 error of each term's log is a few eps times these.
    return abs(mode * log_t) + abs(math.lgamma(mode + 1.0)) + abs(math.lgamma(a * mode))


@pytest.mark.parametrize("row", ORACLE["rows"], ids=lambda r: f"p{r['p']}-lt{r['log_t']:.2f}")
def test_log_w_matches_50_digit_reference(row):
    ok, log_w, _, _ = series_moments(np.array([row["log_t"]]), row["a"])
    assert ok[0]
    reference = float(row["log_w"])
    # 16 eps per unit of peak-term magnitude covers the lgamma evaluations, the
    # exp/log of the log-sum-exp and the summation of <= MAX_ROW_TERMS terms each
    # at most 1 relative to the peak (Higham 2002, sec. 4.2).
    bound = 16.0 * EPS * max(1.0, _peak_magnitude(row["log_t"], row["a"], row["mode"]))
    assert abs(log_w[0] - reference) <= bound


def test_p15_matches_bessel_closed_form_at_large_modes():
    # At p = 1.5, a = 1 and W = sqrt(t) I_1(2 sqrt t) (DLMF 10.46.2); i1e is Cephes, ~1e-15.
    log_t = np.linspace(5.0, 30.0, 26)
    ok, log_w, _, _ = series_moments(log_t, 1.0)
    z = 2.0 * np.exp(0.5 * log_t)
    reference = 0.5 * log_t + np.log(i1e(z)) + z
    assert ok.all()
    np.testing.assert_allclose(log_w, reference, rtol=64 * EPS, atol=0.0)


def test_moments_are_row_local():
    rng = np.random.default_rng(3)
    log_t = rng.uniform(-3.0, 12.0, 200)
    alone = np.array([series_moments(log_t[i : i + 1], 0.7)[1][0] for i in range(log_t.size)])
    together = series_moments(log_t, 0.7)[1]
    np.testing.assert_array_equal(alone, together)


def test_mean_and_variance_match_finite_differences_of_log_w():
    # d log W / d log t = E[J]; d2 log W / d log t2 = Var[J].
    log_t, a, h = 4.0, 0.6, 1e-4
    _, lw, mean_j, var_j = series_moments(np.array([log_t - h, log_t, log_t + h]), a)
    assert mean_j[1] == pytest.approx((lw[2] - lw[0]) / (2 * h), rel=1e-7)
    assert var_j[1] == pytest.approx((lw[2] - 2 * lw[1] + lw[0]) / h**2, rel=1e-4)


def test_rows_past_the_work_bound_are_refused_not_nan():
    a = (2 - 1.95) / 0.95
    # Mode ~1e14: needs ~2 sqrt(74 mode/(a+1)) >> MAX_ROW_TERMS terms.
    log_t = (a + 1) * math.log(1e14) + a * math.log(a)
    ok, log_w, mean_j, var_j = series_moments(np.array([0.0, log_t]), a)
    assert ok.tolist() == [True, False]
    assert np.isnan(log_w[1]) and np.isnan(mean_j[1]) and np.isnan(var_j[1])
    assert MAX_ROW_TERMS == 1_000_000
```

- [x] **Step 3: Run them to confirm they fail.** Run: `uv run pytest tests/test_tweedie_series.py -q`. Expected: ERROR on import (`No module named 'superglm._tweedie_series'`).

- [x] **Step 4: Implement `src/superglm/_tweedie_series.py`:**

```python
"""Compiled Dunn-Smyth (2005) series for the Tweedie density, 1 < p < 2.

A positive response has density W(t) / y * exp(c w / phi) with
W(t) = sum_{j>=1} t^j / (j! Gamma(a j)) and a = (2 - p) / (p - 1). Every term
is positive, so starting at the peak term and summing outward until a term
falls _LOG_CUTOFF below it evaluates log W to machine accuracy; only the term
count grows, about 2 sqrt(2 * 37 * j_max / (a + 1)).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

# Terms 37 log-units below the peak change log W by < 1e-16 (Dunn & Smyth 2005).
_LOG_CUTOFF = 37.0
# Beyond 2**52 consecutive integers are no longer exact in float64.
_MAX_SAFE_MODE = float(2**52)
# Covers peak indices up to ~3e9 (a + 1); stage 0 of the rebuild found no real row
# needing more (benchmarks/tweedie_nb_rebuild_receipts.json, decision a).
MAX_ROW_TERMS = 1_000_000
# lgamma(j + 1) + lgamma(a j) is shared by every row of one call; cap its size.
_TABLE_LIMIT = 1 << 20


@njit(cache=True)
def _term(j: int, log_t: float, a: float, log_base: NDArray) -> float:
    if j < log_base.size:
        return j * log_t - log_base[j]
    return j * log_t - (math.lgamma(j + 1.0) + math.lgamma(a * j))


@njit(cache=True)
def _row_moments(log_t: float, a: float, mode: int, log_base: NDArray):
    """(ok, log W, E[J], Var[J]) for one row, summed outward from the peak term.

    The terms are log-concave in j, so climbing from the estimated mode reaches
    the peak. The climb matters near p = 1: a is large there and one step off
    the peak already takes exp(q - peak) out of range. Two directions times a
    term walk is the algorithm itself, hence the nested loop.
    """
    peak = _term(mode, log_t, a, log_base)
    for direction in (1, -1):
        while mode + direction >= 1:
            neighbour = _term(mode + direction, log_t, a, log_base)
            if not neighbour > peak:
                break
            mode += direction
            peak = neighbour
    mass, first, second, n_terms = 1.0, 0.0, 0.0, 1
    for direction in (1, -1):
        j = mode + direction
        while j >= 1:
            if n_terms >= MAX_ROW_TERMS:
                return False, math.nan, math.nan, math.nan
            q = _term(j, log_t, a, log_base)
            relative = math.exp(q - peak)
            offset = float(j - mode)
            mass += relative
            first += relative * offset
            second += relative * offset * offset
            n_terms += 1
            if q <= peak - _LOG_CUTOFF:
                break
            j += direction
    mean_offset = first / mass
    variance = second / mass - mean_offset * mean_offset
    log_w = peak + math.log(mass)
    if not (math.isfinite(log_w) and math.isfinite(variance)):
        return False, math.nan, math.nan, math.nan
    return True, log_w, mode + mean_offset, variance


@njit(cache=True)
def _series_moments_kernel(log_t, a, ok, log_w, mean_j, var_j) -> None:
    a_plus_one = a + 1.0
    a_log_a = a * math.log(a)
    log_safe_mode = math.log(_MAX_SAFE_MODE)
    modes = np.zeros(log_t.size, dtype=np.int64)  # 0 marks a row past the work bound
    table_size = 64
    for row in range(log_t.size):
        log_mode = (log_t[row] - a_log_a) / a_plus_one
        if log_mode > log_safe_mode:
            continue
        mode = math.exp(log_mode)
        radius = math.sqrt(2.0 * _LOG_CUTOFF * mode / a_plus_one)
        if 2.0 * radius >= MAX_ROW_TERMS:
            continue
        modes[row] = max(1, int(math.floor(mode)))
        table_size = max(table_size, int(min(mode + 4.0 * radius + 64.0, _TABLE_LIMIT)))
    log_base = np.empty(table_size, dtype=np.float64)
    log_base[0] = 0.0
    for j in range(1, table_size):
        log_base[j] = math.lgamma(j + 1.0) + math.lgamma(a * j)
    for row in range(log_t.size):
        if modes[row] == 0:
            ok[row], log_w[row], mean_j[row], var_j[row] = False, math.nan, math.nan, math.nan
            continue
        ok[row], log_w[row], mean_j[row], var_j[row] = _row_moments(
            log_t[row], a, modes[row], log_base
        )


def series_moments(
    log_t: NDArray, a: float
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per row: whether the series evaluated, log W, E[J] and Var[J]."""
    log_t = np.ascontiguousarray(log_t, dtype=np.float64)
    ok = np.empty(log_t.size, dtype=np.bool_)
    log_w = np.empty(log_t.size, dtype=np.float64)
    mean_j = np.empty(log_t.size, dtype=np.float64)
    var_j = np.empty(log_t.size, dtype=np.float64)
    _series_moments_kernel(log_t, float(a), ok, log_w, mean_j, var_j)
    return ok, log_w, mean_j, var_j


def warmup() -> None:
    """Compile the kernel (called by superglm.warmup)."""
    if not series_moments(np.array([-1.0, 2.0]), 1.0)[0].all():
        raise RuntimeError("Tweedie series warmup failed")
```

- [x] **Step 5: Run the tests to confirm they pass.** Run: `uv run pytest tests/test_tweedie_series.py -q`. Expected: all pass. If an oracle row fails, report the row, the bound and the error; do not widen the bound without a derivation.

- [x] **Step 6: Commit.**

```bash
git add src/superglm/_tweedie_series.py benchmarks/tweedie_series_oracle.py tests/fixtures/tweedie_series_oracle.json tests/test_tweedie_series.py
git commit -m "feat: row-local compiled Dunn-Smyth series with a 50-digit oracle

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Density, logpdf pair, φ solver and simulation (`_tweedie.py`)

**Files:**
- Create: `src/superglm/_tweedie.py`, `tests/test_tweedie_density.py`, `tests/test_tweedie_nb_characterisation.py`

**Interfaces:**
- Consumes: `series_moments` (Task 2).
- Produces:
  - `TweedieRows` (frozen dataclass): fields `p`, `a`, `log_t_unit_phi`, `log_y`, `saturated_canonical`, `count`. Property `size`. Classmethod `prepare(y, weights, p, *, frequency=False)`. Methods `row_saturated(phi) -> (value, score, slope)` (per-row arrays) and `saturated(phi) -> (float, float, float)`.
  - `tweedie_unit_deviance(y, mu, p) -> NDArray`: the body of `profiling/tweedie.py:_tweedie_positive_unit_deviance` (lines 458–562), moved verbatim with the recursive call renamed.
  - `tweedie_logpdf(y, mu, phi, p, weights=None) -> NDArray`.
  - `tweedie_logpdf_pair(y, mu, null_mu, phi, p, *, weights=None) -> tuple[NDArray, NDArray]`.
  - `PhiSolve(phi, criterion, curvature, n_passes)` (frozen dataclass).
  - `solve_log_phi(rows, deviance, nullity=0.0) -> PhiSolve`.
  - `generate_tweedie_cpg(n, mu, phi, p, rng=None) -> NDArray`.

- [x] **Step 1: Write the failing tests** in `tests/test_tweedie_density.py`:

```python
import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import i1e

from superglm._tweedie import (
    TweedieRows,
    generate_tweedie_cpg,
    solve_log_phi,
    tweedie_logpdf,
    tweedie_logpdf_pair,
    tweedie_unit_deviance,
)

EPS = np.finfo(np.float64).eps


def _book(p=1.5, phi=2.0, n=4000, seed=11):
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.normal(0.0, 0.5, n))
    return generate_tweedie_cpg(n, mu, phi, p, rng=rng), mu


def test_logpdf_equals_bessel_closed_form_at_p15():
    y, mu = _book()
    y, mu = y[y > 0][:200], mu[y > 0][:200]
    phi = 2.0
    # p = 1.5, w = 1: log f = log(2 / phi) - log(y) / 2 + log I_1(z) - z - d / (2 phi), z = 4 sqrt(y) / phi.
    # The saturated canonical term c(y, y) / phi = -z cancels the e^z of the Bessel
    # function, so the scaled i1e appears alone.
    z = 4.0 * np.sqrt(y) / phi
    deviance = tweedie_unit_deviance(y, mu, 1.5)
    reference = np.log(2.0 / phi) - 0.5 * np.log(y) + np.log(i1e(z)) - deviance / (2 * phi)
    np.testing.assert_allclose(tweedie_logpdf(y, mu, phi, 1.5), reference, rtol=0, atol=64 * EPS * np.maximum(1, np.abs(reference)))


def test_zero_rows_are_the_exact_atom():
    mu = np.array([0.3, 2.0])
    value = tweedie_logpdf(np.zeros(2), mu, 0.7, 1.3, weights=np.array([1.0, 2.5]))
    np.testing.assert_allclose(value, -np.array([1.0, 2.5]) * mu**0.7 / (0.7 * 0.7), rtol=4 * EPS)


def test_logpdf_pair_null_shares_the_saturated_term():
    y, mu = _book(p=1.3)
    null_mu = np.full_like(mu, y.mean())
    fitted, null = tweedie_logpdf_pair(y, mu, null_mu, 1.7, 1.3)
    np.testing.assert_array_equal(fitted, tweedie_logpdf(y, mu, 1.7, 1.3))
    np.testing.assert_allclose(null, tweedie_logpdf(y, null_mu, 1.7, 1.3), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("p", [1.05, 1.3, 1.5, 1.8, 1.95])
def test_solve_log_phi_is_the_profile_minimiser(p):
    y, mu = _book(p=p)
    rows = TweedieRows.prepare(y, np.ones_like(y), p)
    deviance = float(np.sum(tweedie_unit_deviance(y, mu, p)))
    solved = solve_log_phi(rows, deviance)

    def criterion(u):
        return 0.5 * deviance * math.exp(-u) - rows.saturated(math.exp(u))[0]

    brute = minimize_scalar(criterion, bounds=(math.log(solved.phi) - 1, math.log(solved.phi) + 1),
                            method="bounded", options={"xatol": 1e-10})
    assert math.log(solved.phi) == pytest.approx(brute.x, abs=1e-7)
    assert solved.criterion == pytest.approx(criterion(math.log(solved.phi)), rel=1e-12)
    # Score at the root: Q'(u) = -D e^{-u}/2 + T(u); zero up to its evaluation round-off.
    score = -0.5 * deviance / solved.phi + rows.saturated(solved.phi)[1]
    assert abs(score) <= 1e-9 * rows.size
    assert solved.curvature > 0 and solved.n_passes <= 12


def test_solve_log_phi_refuses_no_interior_optimum():
    y = np.zeros(50)
    with pytest.raises(ValueError, match="no finite interior optimum"):
        solve_log_phi(TweedieRows.prepare(y, np.ones(50), 1.5), 3.0)
    y, mu = _book()
    with pytest.raises(ValueError, match="positive finite deviance"):
        solve_log_phi(TweedieRows.prepare(y, np.ones_like(y), 1.5), 0.0)


def test_frequency_counts_equal_replicated_rows():
    y, _ = _book(p=1.4, n=300)
    counts = np.random.default_rng(2).integers(1, 4, y.size).astype(float)
    counted = TweedieRows.prepare(y, counts, 1.4, frequency=True).saturated(1.3)
    replicated = TweedieRows.prepare(np.repeat(y, counts.astype(int)), np.ones(int(counts.sum())), 1.4).saturated(1.3)
    np.testing.assert_allclose(counted, replicated, rtol=1e-13)
```

The generator's draws are already pinned bitwise against a frozen copy of the pre-rebuild generator (`tests/test_tweedie_generator.py`, `_legacy_generate_tweedie_cpg`). Point that file's import at `superglm._tweedie.generate_tweedie_cpg`; its bitwise tests must pass unchanged in Step 4. Its message-text assertions are Task 9's business.

And create `tests/test_tweedie_nb_characterisation.py` with its logpdf arm:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from superglm._tweedie import tweedie_logpdf

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "tweedie_nb_characterisation.json").read_text())
EPS = np.finfo(np.float64).eps


@pytest.mark.parametrize("row", [r for r in FIXTURE["logpdf"] if not r["saddlepoint"]])
def test_logpdf_matches_master_within_old_route_error(row):
    value = tweedie_logpdf(np.array([row["y"]]), np.array([row["mu"]]), row["phi"], row["p"],
                           weights=np.array([row["w"]]))[0]
    scale = max(1.0, abs(row["logpdf"]))
    # The master routes (Wright, Bessel) differ from exact by at most the measured
    # old_route_max_rel_error; the new series adds at most ~16 eps per unit magnitude.
    bound = (FIXTURE["old_route_max_rel_error"] + 64 * EPS) * scale
    assert abs(value - row["logpdf"]) <= bound
```

- [x] **Step 2: Run the tests to confirm they fail.** Run: `uv run pytest tests/test_tweedie_density.py tests/test_tweedie_nb_characterisation.py -q`. Expected: ERROR on import of `superglm._tweedie`.

- [x] **Step 3: Implement `src/superglm/_tweedie.py`:**

```python
"""Tweedie (1 < p < 2) density, dispersion profile and simulation on one series.

Every EDM satisfies log f(y; mu, phi/w) = log f(y; y, phi/w) - w d(y, mu) / (2 phi).
The saturated term depends on (y, w, p) and phi only, and a zero response
contributes nothing to it, so one prepared set of positive rows serves the
density, the fitted/null pair and the dispersion profile.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._tweedie_series import series_moments

_LOG_TWO_PI = math.log(2.0 * math.pi)
_LOG_PHI_LIMIT = 45.0
_NEWTON_MAX_STEPS = 60
_NEWTON_MAX_STEP = 2.0
_NEWTON_STEP_TOL = 1e-8
_POISSON_LAM_MAX = float(np.iinfo(np.int64).max) - 10.0 * math.sqrt(float(np.iinfo(np.int64).max))


@dataclass(frozen=True)
class TweedieRows:
    """Phi-invariant saturated state of the positive responses at one power."""

    p: float
    a: float
    log_t_unit_phi: NDArray
    log_y: NDArray
    saturated_canonical: NDArray
    count: NDArray | None

    @classmethod
    def prepare(cls, y: NDArray, weights: NDArray, p: float, *, frequency: bool = False) -> TweedieRows:
        """Hoist the positive rows; prior weights enter the density, frequency counts multiply it."""
        positive = (y > 0.0) & (weights > 0.0) if frequency else y > 0.0
        y_positive = y[positive]
        density_weight = np.ones_like(y_positive) if frequency else weights[positive]
        a = (2.0 - p) / (p - 1.0)
        log_y = np.log(y_positive)
        log_t = (
            (a + 1.0) * np.log(density_weight)
            + a * (log_y - math.log(p - 1.0))
            - math.log(2.0 - p)
        )
        canonical = density_weight * np.power(y_positive, 2.0 - p) / ((1.0 - p) * (2.0 - p))
        return cls(p, a, log_t, log_y, canonical, weights[positive] if frequency else None)

    @property
    def size(self) -> float:
        return float(self.log_y.size) if self.count is None else float(np.sum(self.count))

    def row_saturated(self, phi: float) -> tuple[NDArray, NDArray, NDArray]:
        """Per row: l_sat, T = d(-l_sat)/d log phi and dT/d log phi at ``phi``."""
        ok, log_w, mean_j, var_j = series_moments(
            self.log_t_unit_phi - (self.a + 1.0) * math.log(phi), self.a
        )
        if not ok.all():
            raise FloatingPointError(
                f"Tweedie series cannot evaluate {int(np.count_nonzero(~ok))} of {ok.size} "
                f"positive rows at p={self.p:.6g}, phi={phi:.6g}"
            )
        canonical = self.saturated_canonical / phi
        inverse_r = self.a + 1.0
        return (
            log_w - self.log_y + canonical,
            mean_j * inverse_r + canonical,
            -var_j * inverse_r**2 - canonical,
        )

    def saturated(self, phi: float) -> tuple[float, float, float]:
        value, score, slope = self.row_saturated(phi)
        if self.count is None:
            return float(np.sum(value)), float(np.sum(score)), float(np.sum(slope))
        return float(self.count @ value), float(self.count @ score), float(self.count @ slope)


@dataclass(frozen=True)
class PhiSolve:
    phi: float
    criterion: float
    curvature: float
    n_passes: int


def solve_log_phi(rows: TweedieRows, deviance: float, nullity: float = 0.0) -> PhiSolve:
    """Minimise Q(u) = D e^-u / 2 - l_sat(e^u) - (M / 2)(log 2 pi + u) over u = log phi.

    M = 0 gives the maximum-likelihood dispersion at a fitted mean; D = Dp and
    M = Mp give Wood (2011) Eq. 4's REML scale term. Newton on the analytic
    score Q'(u) = -D e^-u / 2 + T(u) - M / 2, one series pass per step,
    safeguarded by bisection inside the sign-change bracket (Press et al.,
    rtsafe): no concavity result is known for the Tweedie dispersion profile.
    """
    size = rows.size
    if not (math.isfinite(deviance) and deviance > 0.0):
        raise ValueError("Tweedie dispersion needs a positive finite deviance")
    # l_sat decays like phi^(-1/(p-1)) per positive row, so Q has an interior
    # minimum only if the upper-tail slope N/(p-1) - M/2 is positive.
    if 2.0 * size <= (rows.p - 1.0) * nullity or size == 0.0:
        raise ValueError("Tweedie dispersion profile has no finite interior optimum")
    lower, upper = -_LOG_PHI_LIMIT, _LOG_PHI_LIMIT
    # The saddlepoint density's root, where every positive row adds 1/2 to T.
    u = min(max(math.log(deviance / max(size - nullity, 0.5 * size)), lower + 1.0), upper - 1.0)
    for n_passes in range(1, _NEWTON_MAX_STEPS + 1):
        saturated, saturated_score, saturated_slope = rows.saturated(math.exp(u))
        half_deviance = 0.5 * deviance * math.exp(-u)
        score = saturated_score - half_deviance - 0.5 * nullity
        curvature = saturated_slope + half_deviance
        if score > 0.0:
            upper = u
        else:
            lower = u
        step = -score / curvature if curvature > 0.0 else math.copysign(_NEWTON_MAX_STEP, -score)
        if curvature > 0.0 and abs(step) <= _NEWTON_STEP_TOL:
            # Quadratic convergence puts u + step within O(step^2) of the root;
            # l_sat moves by -T per unit u, so carrying it keeps Q to O(step^2).
            u += step
            criterion = (
                half_deviance * math.exp(-step)
                - (saturated - saturated_score * step)
                - 0.5 * nullity * (_LOG_TWO_PI + u)
            )
            return PhiSolve(math.exp(u), criterion, curvature, n_passes)
        proposal = u + max(-_NEWTON_MAX_STEP, min(step, _NEWTON_MAX_STEP))
        u = proposal if lower < proposal < upper else 0.5 * (lower + upper)
    raise FloatingPointError(
        f"Tweedie dispersion Newton did not settle in {_NEWTON_MAX_STEPS} steps at p={rows.p:.6g}"
    )


def _density_arrays(y, mu, phi, p, weights):
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    weights = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64)
    if y.ndim != 1 or mu.shape != y.shape or weights.shape != y.shape:
        raise ValueError("y, mu and weights must be one-dimensional with the same shape")
    if not (np.all(np.isfinite(y)) and np.all(y >= 0.0)):
        raise ValueError("y must be finite and non-negative")
    if not (np.all(np.isfinite(mu)) and np.all(mu > 0.0)):
        raise ValueError("mu must be finite and strictly positive")
    if not (np.all(np.isfinite(weights)) and np.all(weights > 0.0)):
        raise ValueError("weights must be finite and strictly positive")
    if not (math.isfinite(phi) and phi > 0.0):
        raise ValueError("phi must be finite and strictly positive")
    if not 1.0 < p < 2.0:
        raise ValueError("p must be in the open interval (1, 2)")
    return y, mu, float(phi), float(p), weights


def _saturated_rows(y, weights, phi, p) -> NDArray:
    """Per-row saturated log density, with zero rows at their exact 0."""
    saturated = np.zeros_like(y)
    positive = y > 0.0
    saturated[positive] = TweedieRows.prepare(y, weights, p).row_saturated(phi)[0]
    return saturated


def tweedie_logpdf(y, mu, phi, p, weights=None) -> NDArray:
    """Row log densities of Tweedie(mu, phi / w, p), 1 < p < 2 (Dunn & Smyth 2005)."""
    y, mu, phi, p, weights = _density_arrays(y, mu, phi, p, weights)
    return _saturated_rows(y, weights, phi, p) - weights * tweedie_unit_deviance(y, mu, p) / (2.0 * phi)


def tweedie_logpdf_pair(y, mu, null_mu, phi, p, *, weights=None) -> tuple[NDArray, NDArray]:
    """Fitted and null row log densities from one saturated pass."""
    y, mu, phi, p, weights = _density_arrays(y, mu, phi, p, weights)
    null_mu = np.asarray(null_mu, dtype=np.float64)
    if null_mu.shape != y.shape or not (np.all(np.isfinite(null_mu)) and np.all(null_mu > 0.0)):
        raise ValueError("null_mu must match mu and be finite and strictly positive")
    saturated = _saturated_rows(y, weights, phi, p)
    scale = weights / (2.0 * phi)
    return (
        saturated - scale * tweedie_unit_deviance(y, mu, p),
        saturated - scale * tweedie_unit_deviance(y, null_mu, p),
    )


# tweedie_unit_deviance: moved verbatim from profiling/tweedie.py lines 458-562
# (_tweedie_positive_unit_deviance, with its module constants
# _TWEEDIE_DEVIANCE_SERIES_THRESHOLD and _TWEEDIE_DEVIANCE_SERIES_TERMS), renamed.


def generate_tweedie_cpg(n: int, mu, phi, p: float, rng=None) -> NDArray:
    """Simulate Tweedie(mu, phi, p) as compound Poisson-gamma: N ~ Poisson, Y | N ~ Gamma."""
    if isinstance(n, bool | np.bool_) or operator.index(n) < 0:
        raise ValueError("n must be a non-negative integer")
    n = operator.index(n)
    p = float(p)
    if not 1.0 < p < 2.0:
        raise ValueError("p must be in the open interval (1, 2)")
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (n,))
    phi = np.broadcast_to(np.asarray(phi, dtype=np.float64), (n,))
    if not (np.all(np.isfinite(mu)) and np.all(mu > 0.0) and np.all(np.isfinite(phi)) and np.all(phi > 0.0)):
        raise ValueError("mu and phi must be finite and strictly positive")
    with np.errstate(over="ignore", divide="ignore"):
        rate = np.power(mu, 2.0 - p) / ((2.0 - p) * phi)
        scale = phi * (p - 1.0) * np.power(mu, p - 1.0)
    if not (np.all(rate > 0.0) and np.all(rate <= _POISSON_LAM_MAX) and np.all(np.isfinite(scale)) and np.all(scale > 0.0)):
        raise ValueError("the Poisson rate or gamma scale is not representable for these mu, phi and p")
    rng = np.random.default_rng() if rng is None else rng
    counts = rng.poisson(rate)
    y = np.zeros(n, dtype=np.float64)
    positive = counts > 0
    if positive.any():
        y[positive] = rng.gamma((2.0 - p) / (p - 1.0) * counts[positive], scale=scale[positive])
    return y
```

Move `_tweedie_positive_unit_deviance` (with its two constants) into this file as `tweedie_unit_deviance`, verbatim except for the name and its self-call.

- [x] **Step 4: Run the tests to confirm they pass.** Run: `uv run pytest tests/test_tweedie_density.py tests/test_tweedie_nb_characterisation.py tests/test_tweedie_series.py -n 8 -q` and `uv run pytest tests/test_tweedie_generator.py -k "legacy or bitwise" -q`. Expected: all pass. A characterisation row that fails is reported with its p, φ, y and error. A row that the old code evaluated by `wright_bessel` and whose new value agrees with the Task 2 oracle better than the fixture does is **evidence for the new value**. List those rows in the commit message; do not loosen the bound.

- [x] **Step 5: Commit.**

```bash
git add src/superglm/_tweedie.py tests/test_tweedie_density.py tests/test_tweedie_nb_characterisation.py
git commit -m "feat: one Tweedie density and dispersion solver on the compiled series

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Switch the density callers and the REML scale onto `_tweedie`

**Files:**
- Modify:
  - `src/superglm/distributions.py` (`Tweedie.deviance_unit` ≈ line 531, `Tweedie.log_likelihood` ≈ 537, `prior_weight_log_density` ≈ 776, `_frequency_weight_log_likelihood` ≈ 795)
  - `src/superglm/stats/model_tests.py:148-157`
  - `src/superglm/model/fit_ops.py:538-560` (`_compute_fit_stats` Tweedie arm)
  - `src/superglm/reml/scale.py` (Tweedie part, lines 847–1465, plus the imports at 10–13)
  - `src/superglm/reml/objective.py:28-34, 78`
- Test: `tests/test_tweedie_reml_exact_scale.py`, `tests/test_reml_scale.py`, `tests/test_reml_scale_integration.py`, `tests/test_discrete_reml_scale.py`, `tests/test_tweedie_reml_reference.py`, `tests/test_pearson_scale_weights.py`, `tests/test_nb2.py`, `tests/test_tweedie_nb_characterisation.py`

**Interfaces:**
- Consumes: `TweedieRows`, `solve_log_phi`, `tweedie_logpdf`, `tweedie_logpdf_pair`, `tweedie_unit_deviance` (Task 3).
- Produces:
  - `reml.scale.prepare_tweedie_reml_scale_data(y, sample_weight, power, *, weight_semantics) -> TweedieRows`
  - `reml.scale.profile_tweedie_reml_scale(profile_data: TweedieRows, penalized_deviance, penalty_nullity) -> ProfiledScaleTerm`

  Both keep their current names and signatures. `TweedieScaleProfileData` is removed.

- [x] **Step 1: Add the REML-phi arm to the characterisation test** in `tests/test_tweedie_nb_characterisation.py`:

```python
@pytest.mark.parametrize("row", FIXTURE["reml_phi"], ids=lambda r: r["case"])
def test_reml_phi_matches_master(row):
    from tests.test_tweedie_reml_exact_scale import fit_fixture_model  # the file's own builder

    model = fit_fixture_model(row["case"])
    # The master saturated likelihood used the series (or i1e at p=1.5) to <=1e-13;
    # phi moves by that over the profile curvature, well under 1e-9 relative.
    assert model.result.phi == pytest.approx(row["phi"], rel=1e-9)
```

If `tests/test_tweedie_reml_exact_scale.py` has no reusable builder, extract its model construction into a module-level `fit_fixture_model(case: str)` in that test file first. That is a test-only refactor; its assertions stay unchanged.

- [x] **Step 2: Run to confirm the new arm passes on the current code** (the fixture came from it). Run: `uv run pytest tests/test_tweedie_nb_characterisation.py -k reml_phi -q`. Expected: PASS. This is the baseline the switch must preserve.

- [x] **Step 3: Rewire the density callers.**
  - In `distributions.py`, replace each `from superglm.profiling.tweedie import tweedie_logpdf` with `from superglm._tweedie import tweedie_logpdf`, and `_tweedie_positive_unit_deviance` with `from superglm._tweedie import tweedie_unit_deviance`.
  - In `stats/model_tests.py`, the same.
  - In `fit_ops._compute_fit_stats`, import `tweedie_logpdf_pair` from `superglm._tweedie` and call it with the same arguments.
  - Fix the docstring at `distributions.py:537` to read "Tweedie log-likelihood via the Dunn–Smyth series."

- [x] **Step 4: Rewrite the Tweedie part of `reml/scale.py`.**
  1. Delete `_ScoreUnavailableInBracketError`; the whole `TweedieScaleProfileData` class (850–1113); `_newton_tweedie_log_phi`; `_bounded_tweedie_log_phi`; all `_TWEEDIE_*` constants; and the `_P15_BESSEL_ASYMPTOTIC_MIN_ARGUMENT` import, `i0e`, `i1e` and `brentq` (if Gamma no longer uses them; check with grep).
  2. Replace `prepare_tweedie_reml_scale_data` with:

```python
def prepare_tweedie_reml_scale_data(
    y: NDArray, sample_weight: NDArray, power: float, *, weight_semantics: str
) -> TweedieRows:
    """Hoist the positive rows once per fit: the saturated likelihood's only phi-dependent part.

    Under "prior" the weight is an EDM precision inside the density and must be
    strictly positive; under "frequency" it is a replication count that
    multiplies the unit-weight row, so zero counts simply drop out.
    """
    y = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    frequency = weight_semantics == FREQUENCY_WEIGHTS
    if not frequency and np.any(sample_weight <= 0.0):
        raise ValueError("Tweedie scale profiling requires strictly positive prior weights")
    rows = TweedieRows.prepare(y, sample_weight, float(power), frequency=frequency)
    if rows.log_y.size == 0:
        raise ValueError(
            "Tweedie scale profiling requires at least one positive response; "
            "an all-zero response has no estimable dispersion"
        )
    return rows
```

  3. Replace `profile_tweedie_reml_scale` with:

```python
def profile_tweedie_reml_scale(
    profile_data: TweedieRows, penalized_deviance: float, penalty_nullity: float
) -> ProfiledScaleTerm:
    """Profile phi in Wood (2011) Eq. 4 / Wood, Pya & Saefken (2016) Sec. 3.3 with the exact saturated likelihood.

    Q(phi) = Dp / (2 phi) - l_sat(phi) - (Mp / 2) log(2 pi phi), solved by
    `solve_log_phi`. Implicit differentiation of Q'(xi) = 0 at xi = 1/phi gives
    d(xi)/d(Dp) = -1/2 / Q''(xi) = -exp(-2 log phi) / (2 Q''(log phi)).
    """
    solved = solve_log_phi(profile_data, float(penalized_deviance), float(penalty_nullity))
    log_phi = math.log(solved.phi)
    log_magnitude = math.log(0.5) - 2.0 * log_phi - math.log(solved.curvature)
    if log_magnitude > math.log(np.finfo(np.float64).max):
        raise FloatingPointError("Tweedie REML scale derivative is not representable")
    derivative = -0.0 if log_magnitude < math.log(np.nextafter(0.0, 1.0)) else -math.exp(log_magnitude)
    return ProfiledScaleTerm(
        phi=solved.phi,
        inverse_phi=math.exp(-log_phi),
        criterion=solved.criterion,
        d_inverse_phi_d_penalized_deviance=derivative,
    )
```

  4. In `prepare_reml_scale_data`, update the Tweedie arm's return annotation to `TweedieRows | None`, and `__all__` to drop `TweedieScaleProfileData`.
  5. In `reml/objective.py`, import `TweedieRows` from `superglm._tweedie` in place of `TweedieScaleProfileData` (import list and the annotation at line 78).
  6. Decision (c) from Task 1: if it kept the Bessel path, re-add it as a `TweedieRows`-independent fast path inside `profile_tweedie_reml_scale` for `p == 1.5` only, with the measured speed-up cited in its comment. Otherwise add nothing.

- [x] **Step 5: Run the focused tests.** Run: `uv run pytest tests/test_tweedie_reml_exact_scale.py tests/test_reml_scale.py tests/test_reml_scale_integration.py tests/test_discrete_reml_scale.py tests/test_tweedie_reml_reference.py tests/test_pearson_scale_weights.py tests/test_nb2.py tests/test_tweedie_nb_characterisation.py tests/test_tweedie_density.py -n 8 -q`. Expected: the mgcv oracles pass unchanged, `reml_phi` passes, and the `logpdf` arm passes.
  - Tests that import `TweedieScaleProfileData` or its private methods fail with `ImportError`/`AttributeError`. List them in the commit message; Task 9 rewrites them.
  - Run `tests/test_tweedie_profile.py` too, and report only the failures that are not private-name imports.

- [x] **Step 6: Commit.**

```bash
git add -u src/superglm tests/test_tweedie_nb_characterisation.py tests/test_tweedie_reml_exact_scale.py
git commit -m "refactor: REML scale and Tweedie densities on the single series solver

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

**Task 4 outcome:**

- `solve_log_phi` stops on the move rtsafe actually takes, Newton or bisection. Testing only the Newton step never settled on the weighted SCOP fit in `test_pearson_scale_weights.py`. There φ ≈ 1.5e-7, every peak index is about 1e7, and the score's round-off (up to 5e-4, against a curvature of 17) kept Newton steps above 1e-8 after the bracket had collapsed. Master's Newton failed the same way and fell through to the bounded search this rebuild deletes. The bracket test is inclusive, so a score of exactly 0.0 stops the solve where it is. A strict test bisected away from the root and cost one p = 1.95 REML solve 29 passes instead of 5.
- `TweedieRows.phi_solves` holds the solves by (D, M). The direct REML optimizer re-profiles each accepted line-search point with identical (Dp, Mp) at the next iteration's start, 4 of 12 calls on positive96. Master's per-φ cache absorbed these. Without the memo a fit makes 48 series passes against master's 32; with it, 32.
- Decision (c) is honoured: `fit_reml` has no Bessel path.
- The characterisation arm reuses `benchmarks/tweedie_nb_characterisation.build_case`, the builder that wrote the fixture, and also covers `reml_phi_books`. `tests/test_tweedie_reml_exact_scale.py` is unchanged.
- **Open (needs a decision):** `test_tweedie_numerics.py::test_near_perfect_tweedie_fit_does_not_fail_in_fit_statistics` now raises the §10 refusal from `fit()`. The cause is fit statistics at φ ≈ 2.6e-26, where all 40 rows' modes are past 2^52.

---

### Task 5: One-parameter profile machinery (`profiling/_scalar.py`)

**Files:**
- Create: `src/superglm/profiling/_scalar.py`, `tests/test_profile_scalar.py`

**Interfaces:**
- Produces:
  - `RecordedObjective(objective: Callable[[float], float])`: callable, caches by exact float key and records insertion order in `.values: dict[float, float]`. Method `best() -> tuple[float, float]` returns the minimum over finite values.
  - `minimize_profile(objective: RecordedObjective, bounds: tuple[float, float], *, xatol: float, maxiter: int) -> bool` returns converged. The argmin comes from `objective.best()`.
  - `Interval(lower, upper, lower_censored, upper_censored)` (frozen dataclass).
  - `likelihood_ratio_interval(objective, x_hat, nll_hat, bounds, *, alpha, scale, rtol) -> Interval`.
  - `profile_plot(values: dict[float, float], x_hat, nll_hat, *, scale, alpha, interval: Interval | None, label: str, ax=None)`, returning the matplotlib `Axes`.

- [x] **Step 1: Write the failing tests** in `tests/test_profile_scalar.py`:

```python
import math

import pytest
from scipy.stats import chi2

from superglm.profiling._scalar import (
    RecordedObjective,
    likelihood_ratio_interval,
    minimize_profile,
)

N = 1000.0
CURVATURE = 3.0


def quadratic(x):
    return 0.5 * CURVATURE * (x - 0.3) ** 2


def test_minimize_profile_records_bounds_and_finds_minimum():
    objective = RecordedObjective(quadratic)
    assert minimize_profile(objective, (0.0, 1.0), xatol=1e-6, maxiter=50)
    x_hat, _ = objective.best()
    assert x_hat == pytest.approx(0.3, abs=1e-5)
    assert 0.0 in objective.values and 1.0 in objective.values


def test_boundary_minimum_is_the_recorded_bound():
    objective = RecordedObjective(lambda x: x)
    minimize_profile(objective, (0.2, 1.0), xatol=1e-6, maxiter=50)
    assert objective.best()[0] == 0.2


def test_interval_is_exact_for_a_quadratic_profile():
    half_width = math.sqrt(chi2.ppf(0.95, 1) / (N * CURVATURE))
    interval = likelihood_ratio_interval(quadratic, 0.3, 0.0, (0.0, 1.0), alpha=0.05, scale=N, rtol=1e-12)
    assert interval.lower == pytest.approx(0.3 - half_width, rel=1e-9)
    assert interval.upper == pytest.approx(0.3 + half_width, rel=1e-9)
    assert not (interval.lower_censored or interval.upper_censored)


def test_side_inside_the_acceptance_region_is_censored_at_the_bound():
    interval = likelihood_ratio_interval(quadratic, 0.3, 0.0, (0.29, 0.31), alpha=0.05, scale=N, rtol=1e-12)
    assert (interval.lower, interval.upper) == (0.29, 0.31)
    assert interval.lower_censored and interval.upper_censored


def test_interval_side_ending_at_infeasible_region_is_censored():
    def objective(x):
        return math.inf if x > 0.32 else quadratic(x)

    interval = likelihood_ratio_interval(objective, 0.3, 0.0, (0.0, 1.0), alpha=0.05, scale=N, rtol=1e-12)
    assert interval.upper_censored and interval.upper == pytest.approx(0.32, abs=1e-9)
    assert not interval.lower_censored
```

- [x] **Step 2: Run them to confirm they fail.** Run: `uv run pytest tests/test_profile_scalar.py -q`. Expected: ERROR on import.

- [x] **Step 3: Implement `src/superglm/profiling/_scalar.py`:**

```python
"""One-parameter profile likelihood: recorded bounded search and likelihood-ratio interval."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2

# Stand-in for an infeasible point's excess: finite so brentq's interpolation
# stays finite, far above any likelihood-ratio excess that can occur.
_BARRIER = 1e6
# Stand-in objective for an infeasible point inside Brent: finite because the
# parabolic step differences objective values (inf - inf is NaN); its square
# stays far below overflow.
_INFEASIBLE = 1e50


class RecordedObjective:
    """Cache an objective by exact argument and keep every evaluation in call order."""

    def __init__(self, objective: Callable[[float], float]):
        self._objective = objective
        self.values: dict[float, float] = {}

    def __call__(self, x: float) -> float:
        key = float(x)
        if key not in self.values:
            self.values[key] = float(self._objective(key))
        return self.values[key]

    def best(self) -> tuple[float, float]:
        finite = {x: v for x, v in self.values.items() if math.isfinite(v)}
        if not finite:
            raise RuntimeError("no evaluated point has a finite profile objective")
        x_best = min(finite, key=finite.__getitem__)
        return x_best, finite[x_best]


def minimize_profile(
    objective: RecordedObjective, bounds: tuple[float, float], *, xatol: float, maxiter: int
) -> bool:
    """Bounded Brent (Brent 1973) after evaluating both bounds.

    Brent's bounded method never evaluates the bounds themselves, so they are
    recorded first and a boundary optimum is found by `objective.best()`.
    Infeasible points (inf) are replaced by a finite barrier inside the search
    so the parabolic steps stay finite; they never win `best()`.
    """
    objective(bounds[0])
    objective(bounds[1])

    def finite_objective(x: float) -> float:
        value = objective(x)
        return value if math.isfinite(value) else _INFEASIBLE

    result = minimize_scalar(
        finite_objective, bounds=bounds, method="bounded", options={"xatol": xatol, "maxiter": maxiter}
    )
    return bool(result.success)


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    lower_censored: bool
    upper_censored: bool


def likelihood_ratio_interval(
    objective: Callable[[float], float],
    x_hat: float,
    nll_hat: float,
    bounds: tuple[float, float],
    *,
    alpha: float,
    scale: float,
    rtol: float,
) -> Interval:
    """{x : 2 scale (nll(x) - nll_hat) <= chi2_1(1 - alpha)} around x_hat (Venzon & Moolgavkar 1988)."""
    cutoff = float(chi2.ppf(1.0 - alpha, 1))

    def excess(x: float) -> float:
        value = objective(x)
        return min(2.0 * scale * (value - nll_hat) - cutoff, _BARRIER) if math.isfinite(value) else _BARRIER

    lower, lower_censored = _interval_side(excess, bounds[0], x_hat, rtol)
    upper, upper_censored = _interval_side(excess, bounds[1], x_hat, rtol)
    return Interval(lower, upper, lower_censored, upper_censored)


def _interval_side(excess, bound: float, x_hat: float, rtol: float) -> tuple[float, bool]:
    """Root of the excess between x_hat and the bound, or the bound when none exists.

    A genuine likelihood-ratio crossing has |excess| <= |slope| * xtol at the
    root, well under 1 for any realistic n (slope ~ 2 sqrt(chi2 n curvature),
    about 4e3 at n = 1e6, times the ~1.5e-6 root tolerance). A root that brentq
    places at a jump into the infeasible barrier does not, and is reported
    censored.
    """
    if excess(bound) <= 0.0:
        return bound, True
    root = float(brentq(excess, min(bound, x_hat), max(bound, x_hat), xtol=1e-12, rtol=rtol))
    return root, abs(excess(root)) > 1.0


def profile_plot(values, x_hat, nll_hat, *, scale, alpha, interval, label, ax=None):
    """Likelihood-ratio statistic against the parameter, with the cutoff and the interval."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    points = sorted((x, v) for x, v in values.items() if math.isfinite(v))
    ax.plot([x for x, _ in points], [2.0 * scale * (v - nll_hat) for _, v in points], "o-", ms=3)
    ax.axhline(chi2.ppf(1.0 - alpha, 1), ls="--", c="grey", lw=1)
    ax.axvline(x_hat, c="k", lw=1)
    if interval is not None:
        ax.axvspan(interval.lower, interval.upper, alpha=0.15)
    ax.set_xlabel(label)
    ax.set_ylabel("likelihood-ratio statistic")
    return ax
```

- [x] **Step 4: Run the tests to confirm they pass.** Run: `uv run pytest tests/test_profile_scalar.py -q`. Expected: all pass.

- [x] **Step 5: Commit.**

```bash
git add src/superglm/profiling/_scalar.py tests/test_profile_scalar.py
git commit -m "feat: shared one-parameter profile search and likelihood-ratio interval

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Tweedie power search and result (`profiling/tweedie.py` replaced)

**Files:**
- Replace: `src/superglm/profiling/tweedie.py`. The old file is removed from the import graph in this task and its callers are switched in Task 7; tests that import its private names are handled in Task 9.
- Modify: `src/superglm/model/fit_ops.py` (`_solve_coefficients`, ≈1198: add `beta_init=None, intercept_init=None` and pass them to both solvers)
- Test: `tests/test_tweedie_p_recovery.py` (create), `tests/test_tweedie_nb_characterisation.py` (extend)

**Interfaces:**
- Consumes: `TweedieRows`, `solve_log_phi`, `tweedie_unit_deviance` (Task 3); `RecordedObjective`, `minimize_profile`, `likelihood_ratio_interval`, `Interval`, `profile_plot` (Task 5); `fit_ops._solve_coefficients(..., beta_init, intercept_init)`; `model.base.model_build_design_matrix`, `resolve_selection_penalty_for_fit`, `model_has_lambda1_targets`; `ObservedModeNotCertifiedError`.
- Produces:
  - `search_power(model, X, y, sample_weight, offset, *, fit_mode: Literal["fit", "fit_reml"], p_bounds=(1.05, 1.95), xatol=1e-3, maxiter=30) -> TweedieProfileResult`.
  - `TweedieProfileResult` with fields:
    - `p_hat: float`
    - `phi_hat: float`
    - `nll: float` (mean NLL at the published fit; equals `search_nll` until publication)
    - `converged: bool`
    - `fit_mode: str`
    - `evaluations: pd.DataFrame` (columns `p`, `nll`, `phi`, `fit_converged`, in search order; infeasible powers carry `nll = inf`)
    - `warnings: list[str]`
    - `search_nll: float`
    - private: `_objective`, `_ll_scale`, `_ci_bounds`, `_ci_cache: dict[float, Interval]`
    
    Methods: `ci(alpha=0.05) -> tuple[float, float]`, `interval(alpha=0.05) -> Interval`, `profile_plot(alpha=0.05, ax=None)`.
  - `profile_phi_at(y, mu, weights, p) -> PhiSolve` (ML dispersion at a fitted mean; used by publication in Task 7).

- [x] **Step 1: Write the failing tests.** Create `tests/test_tweedie_p_recovery.py`:

```python
"""Spec criterion 8: estimate_p recovers the true power on constant-phi data."""

import math

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM, families
from superglm.features import Categorical, Spline
from superglm._tweedie import generate_tweedie_cpg

pytestmark = pytest.mark.slow


def _simulate(p: float, seed: int, n: int = 20_000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    level = rng.integers(0, 5, n)
    mu = np.exp(0.3 + np.sin(2 * np.pi * x) * 0.6 + np.array([0.0, 0.2, -0.3, 0.1, 0.4])[level])
    y = generate_tweedie_cpg(n, mu, 1.8, p, rng=rng)
    return pd.DataFrame({"x": x, "level": level.astype(str)}), y


@pytest.mark.parametrize("fit_mode", ["fit", "reml"])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("true_p", [1.2, 1.5, 1.8])
def test_estimate_p_recovers_true_power(true_p, seed, fit_mode):
    X, y = _simulate(true_p, seed)
    model = SuperGLM(
        family=families.tweedie(p=1.5),
        features={"x": Spline(n_knots=10), "level": Categorical()},
    )
    result = model.estimate_p(X, y, fit_mode=fit_mode)
    h = 0.01
    curvature = (result._objective(result.p_hat + h) - 2 * result.search_nll + result._objective(result.p_hat - h)) / h**2
    standard_error = 1.0 / math.sqrt(len(y) * curvature)
    assert abs(result.p_hat - true_p) <= 3.0 * standard_error
```

Before running, check how existing tests construct `SuperGLM` with features (for example `tests/test_tweedie_profile.py`), and use that constructor form exactly: the `features=` keyword or `add_feature`.

Extend `tests/test_tweedie_nb_characterisation.py`:

```python
@pytest.mark.slow
@pytest.mark.parametrize("row", FIXTURE["estimate_p"], ids=lambda r: f"{r['case']}-{r['fit_mode']}")
def test_estimate_p_matches_master(row, characterisation_case):
    model, X, y = characterisation_case(row["case"])  # conftest fixture building the Task 1 cases
    result = model.estimate_p(X, y, fit_mode=row["fit_mode"], ci_alpha=0.05)
    # Brent's path can differ only where a comparison flips at the density's
    # ~1e-11 change, which moves p_hat by at most its final bracket, xatol.
    assert abs(result.p_hat - row["p_hat"]) <= 1e-3
    # The objective moves by at most the summed per-row logpdf change.
    assert result.search_nll == pytest.approx(row["search_nll"], rel=1e-8)
    assert result.phi_hat == pytest.approx(row["phi_hat"], rel=1e-6)
    lower, upper = result.ci(0.05)
    assert lower == pytest.approx(row["ci95"][0], abs=2e-3) and upper == pytest.approx(row["ci95"][1], abs=2e-3)


def test_boundary_optimum_is_reported_not_raised(characterisation_case):
    model, X, y = characterisation_case("zeros90")
    result = model.estimate_p(X, y, p_bounds=(1.6, 1.9))  # true p = 1.5 lies below the window
    assert result.p_hat == 1.6
    assert any("bound" in w for w in result.warnings)
    assert result.interval(0.05).lower_censored
```

Add a `characterisation_case` fixture to `tests/conftest.py` (or a local conftest) that rebuilds each Task 1 case by name with the same seeds and constructors as `benchmarks/tweedie_nb_characterisation.py`. Import its builder function instead of duplicating it.

The phi and CI tolerances above are placeholders to derive, not to accept. Before asserting, compute the actual bounds from the fixture's recorded curvature: phi tolerance = nll bound / (Q'' · phi); CI tolerance = rtol + nll bound / |profile slope at the endpoint|. Replace the literals with the derived expressions, with the derivation in a comment.

- [x] **Step 2: Run to confirm they fail.** Run: `uv run pytest tests/test_tweedie_p_recovery.py tests/test_tweedie_nb_characterisation.py -k "recover or estimate_p or boundary" -q`. Expected: FAIL. The recovery test fails on `result._objective`/`result.search_nll` attribute shape, or passes on master; record which. The boundary test fails on `result.interval`.

- [x] **Step 3: Add warm starts to `_solve_coefficients`.** In `fit_ops._solve_coefficients`, add the keyword-only parameters `beta_init=None, intercept_init=None` and pass `beta_init=beta_init, intercept_init=intercept_init` to both `fit_irls_direct(...)` and `fit_pirls(...)`. Existing callers are unchanged.

- [x] **Step 4: Write the new `src/superglm/profiling/tweedie.py`.** Core code below. Keep the whole module ≤ 600 lines.

```python
"""Profile-likelihood estimation of the Tweedie power (Dunn & Smyth 2005).

At each candidate p the mean is refitted (warm-started) and phi is profiled
at that mean by `solve_log_phi`; the power is the bounded-Brent minimiser of
the resulting mean negative log-likelihood, and its interval inverts the
likelihood-ratio test on the same curve.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from superglm._tweedie import PhiSolve, TweedieRows, solve_log_phi, tweedie_unit_deviance
from superglm.profiling._scalar import (
    Interval,
    RecordedObjective,
    likelihood_ratio_interval,
    minimize_profile,
    profile_plot,
)

# Candidate REML fits only rank powers; see the measurement recorded with
# _SEARCH_REML_TOL on master (p_hat agrees with tight-bar searches to ~1e-11).
_SEARCH_REML_TOL = 1e-6
_CI_BOUNDS = (1.02, 1.98)
_CI_RTOL = 1e-6


def profile_phi_at(y, mu, weights, p) -> PhiSolve:
    """Maximum-likelihood phi at a fitted mean: Q with M = 0."""
    deviance = float(np.sum(weights * tweedie_unit_deviance(y, mu, p)))
    return solve_log_phi(TweedieRows.prepare(y, weights, p), deviance)


@dataclass
class _Candidate:
    nll: float
    phi: float
    fit_converged: bool


class _PowerProfile:
    """Mean NLL of the profile at a power: refit mu(p), then phi(p) at that mean."""

    def __init__(self, model, X, y, sample_weight, offset, fit_mode: str):
        ...  # clone (see _clone_profile_model on master), build design once for "fit";
        # set clone._profile_design_cache = {} for "fit_reml"
        self.candidates: dict[float, _Candidate] = {}
        self.infeasible: dict[float, str] = {}

    def __call__(self, p: float) -> float:
        try:
            mu, fit_converged = self._fit(p)
        except ObservedModeNotCertifiedError as exc:  # REML only: skipped, not fatal
            self.infeasible[p] = str(exc)
            return math.inf
        solved = profile_phi_at(self.y, mu, self.w, p)
        nll = solved.criterion / self.n
        self.candidates[p] = _Candidate(nll, solved.phi, fit_converged)
        return nll
```

Implementation requirements for `_PowerProfile`. No placeholder survives; each is concrete:

- **Clone.** Reuse master's `_clone_profile_model` body verbatim (master `profiling/tweedie.py:4161-4192`) as a module-private helper, and snapshot inputs as master's `_snapshot_profile_inputs` did.
- **ML (`"fit"`) setup.**
  - Once: set `clone._distribution = Tweedie(1.5)` temporarily, then call `y, w, offset = model_build_design_matrix(clone, X, y, w, offset)`, then `resolve_selection_penalty_for_fit(clone, configured_penalty(clone), y, w)`.
  - Keep master's two monotone `NotImplementedError` guards: they fire on real configurations and are tested.
  - Keep `_reject_random_effect_selection_fit(model, "fit")` and `_reject_lambda_policy_fit`, which `estimate_tweedie_p` called at entry on master.
- **ML (`"fit"`) per p.**
  - Set `clone._distribution = Tweedie(p)`.
  - Call `result = _solve_coefficients(clone, y, w, offset, penalty=..., lambda2=configured_lambda2(clone), has_lambda1_targets=model_has_lambda1_targets(clone), max_iter=clone._max_iter, tol=clone._tol, record_diagnostics=False, convergence=clone._convergence, beta_init=self._warm_beta, intercept_init=self._warm_intercept)`.
  - Then `mu = clip_mu(link.inverse(stabilize_eta(dm.matvec(beta) + intercept + offset, link)), dist)`, and store the warm start.
- **REML (`"fit_reml"`) per p.**
  - Set `clone.family = Tweedie(p)`.
  - Call `clone.fit_reml(X, y, sample_weight=..., offset=..., runtime_validation="skip", reml_tol=_SEARCH_REML_TOL, lambda2_init=self._warm_lambdas)`.
  - Then `mu = clone._fit_mu`, and store `self._warm_lambdas = dict(clone._reml_result.lambdas)`.
  - Measure `lambda2_init` in Task 10. If it does not speed up the §9 REML case, or moves p̂ beyond `xatol` against the fixture, remove it (it is one line).
- **ML size.** `self.n = float(len(y))`. Prior semantics only: frequency weights with non-unit values are refused in `profile_ops.estimate_p` (Task 7), exactly as master refused them.

`search_power`:

```python
def search_power(model, X, y, sample_weight, offset, *, fit_mode, p_bounds=(1.05, 1.95), xatol=1e-3, maxiter=30):
    profile = _PowerProfile(model, X, y, sample_weight, offset, fit_mode)
    objective = RecordedObjective(profile)
    search_converged = minimize_profile(objective, p_bounds, xatol=xatol, maxiter=maxiter)
    p_hat, nll_hat = objective.best()
    best = profile.candidates[p_hat]
    warnings = [f"p={p:.4g} skipped: {reason}" for p, reason in profile.infeasible.items()]
    if min(p_hat - p_bounds[0], p_bounds[1] - p_hat) <= xatol:
        warnings.append(
            f"p_hat={p_hat:.4g} is at the search bound {p_bounds}; the optimum may lie beyond it "
            "(a maximum as p -> 1 can be an artefact of rounded responses, Dunn & Smyth 2005)."
        )
    evaluations = pd.DataFrame(
        [(p, v, getattr(profile.candidates.get(p), "phi", math.nan), getattr(profile.candidates.get(p), "fit_converged", False))
         for p, v in objective.values.items()],
        columns=["p", "nll", "phi", "fit_converged"],
    )
    return TweedieProfileResult(
        p_hat=p_hat, phi_hat=best.phi, nll=nll_hat, converged=search_converged and best.fit_converged,
        fit_mode=fit_mode, evaluations=evaluations, warnings=warnings, search_nll=nll_hat,
        _objective=objective, _ll_scale=profile.n,
        _ci_bounds=(min(_CI_BOUNDS[0], p_bounds[0]), max(_CI_BOUNDS[1], p_bounds[1])),
    )
```

`TweedieProfileResult`:

```python
@dataclass
class TweedieProfileResult:
    """Profile-likelihood estimate of the Tweedie power p and dispersion phi."""

    p_hat: float
    phi_hat: float
    nll: float
    converged: bool
    fit_mode: str
    evaluations: pd.DataFrame
    warnings: list[str]
    search_nll: float
    _objective: Any = field(default=None, repr=False)
    _ll_scale: float = field(default=1.0, repr=False)
    _ci_bounds: tuple[float, float] = field(default=_CI_BOUNDS, repr=False)
    _ci_cache: dict[float, Interval] = field(default_factory=dict, repr=False)

    def interval(self, alpha: float = 0.05) -> Interval:
        """Likelihood-ratio interval for p on the searched curve, with censoring flags."""
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if alpha not in self._ci_cache:
            if self._objective is None:
                raise RuntimeError("this result no longer carries its profile objective")
            self._ci_cache[alpha] = likelihood_ratio_interval(
                self._objective, self.p_hat, self.search_nll, self._ci_bounds,
                alpha=alpha, scale=self._ll_scale, rtol=_CI_RTOL,
            )
        return self._ci_cache[alpha]

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        interval = self.interval(alpha)
        return interval.lower, interval.upper

    def profile_plot(self, alpha: float = 0.05, ax=None):
        values = self._objective.values if self._objective is not None else dict(zip(self.evaluations.p, self.evaluations.nll))
        interval = self._ci_cache.get(alpha)
        return profile_plot(values, self.p_hat, self.search_nll, scale=self._ll_scale,
                            alpha=alpha, interval=interval, label="p")
```

- [x] **Step 5: Run the tests.** Run: `uv run pytest tests/test_tweedie_p_recovery.py tests/test_tweedie_nb_characterisation.py tests/test_profile_scalar.py -n 8 -q`. Expected: pass (`estimate_p` itself is rewired in Task 7; until then, call `search_power` through a temporary test helper, or run this step at the end of Task 7). Add `test_uncertifiable_power_is_skipped`:

```python
def test_uncertifiable_power_is_skipped(characterisation_case, monkeypatch):
    from superglm.reml.observed_geometry import ObservedModeNotCertifiedError
    model, X, y = characterisation_case("positive96")
    original = type(model).fit_reml

    def flaky(self, *args, **kwargs):
        if self._family_config.p > 1.9:
            raise ObservedModeNotCertifiedError("synthetic")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "fit_reml", flaky)
    result = model.estimate_p(X, y, fit_mode="reml")
    assert math.isinf(result.evaluations.set_index("p").loc[1.95, "nll"])
    assert any("skipped" in w for w in result.warnings)
```

Check `ObservedModeNotCertifiedError`'s constructor signature in `reml/observed_geometry.py` and construct it accordingly.

- [x] **Step 6: Commit.**

```bash
git add -A src/superglm/profiling/tweedie.py src/superglm/model/fit_ops.py tests/test_tweedie_p_recovery.py tests/test_tweedie_nb_characterisation.py tests/conftest.py
git commit -m "feat: Brent power search on the single dispersion solver

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

**Task 5 and 6 outcome:**

- `search_power` reproduces master's Brent path on every fixture row that master searched by Brent away from a certification wall. p̂ agrees within 4e-12, `search_nll` within 6e-15 relative, and the CI endpoints within 4.9e-6, which is inside master's own accepted root residual. On the private validation dataset p̂ and `search_nll` are identical in both modes and the CI agrees within 1.1e-8.
- The REML `lambda2_init` warm start is removed, which settles Task 10 Step 3. It was inert: candidate fits were bitwise identical with and without it, because the direct and discrete engines bootstrap their own lambdas.
- `test_estimate_p_matches_master` has derived tolerances:
  - p̂: scipy's bounded Brent leaves each estimate within 2·(√ε·p + xatol/3) of the minimiser it brackets, so two searches differ by at most twice that (1.33·xatol, not xatol).
  - `search_nll`: c·reach²/2 plus the density bound, plus, for ML candidates, the warm-started fit's determination tol·D/(2φn). The curvature c is recovered from master's interval.
  - φ: checked as the ML dispersion at master's p̂ on the published mean, where only the density and the fit move it. The published φ̂ itself moves with p̂: on `unpen/fit`, d log φ/dp ≈ 4, so no bound derivable from the fixture covers it.
  - CI: master's accepted LR residual (10⁻³·χ²) over 2n·slope, plus brentq's tolerance, plus the NLL shift over the slope.
  - Measured against an emulated publication, every ratio is ≤ 0.53. Seven source mutations each redden at least one test.
- Listed flip: `re_flat_p18/reml`. Master scored p = 1.76433 uncertifiable. The rebuilt REML scale certifies it, because a mode score against the 10⁻⁹ bar is a knife edge. Brent then routes on to p̂ = 1.76493, with a mean NLL 0.0066 lower (20 log-likelihood units) and still censored between uncertifiable powers. The test checks that the profile agrees at master's p̂ (2e-13), that the new estimate is no worse, and that the censoring is disclosed.
- Warnings:
  - p̂ is at a search bound when it is the first or last evaluated power.
  - An infeasible evaluated neighbour marks p̂ censored.
  - Every skipped power is listed.
  - The interval's search is cut to the search bound on the side p̂ sits on. The plan's widened bounds would find a root beyond the bound and fail the plan's own boundary test.
- **Open, criterion 8 REML arm:** plain `fit_reml` at p = 1.8 fails mode certification on all three recovery seeds, with scores 1.3e-9 to 1.1e-7 against the 10⁻⁹ bar. The failure is identical on master. The REML search is then censored: seed 3 is trapped at p̂ = 1.691, the same as master. Its curvature probes also land on uncertifiable powers. With publication emulated, the recovery test passes 15 of 18 cases; the three REML p = 1.8 cases fail. This needs a decision; it is not a search defect.

---

### Task 7: `estimate_p` publication, summary and editor

**Files:**
- Modify: `src/superglm/model/profile_ops.py` (`estimate_p`, `_reprofile_published_dispersion`, `_synchronize_tweedie_profile_refit`, `_installed_tweedie_profile_copy`, the payload helpers); `src/superglm/model/api.py` (`estimate_p` ≈1290); `src/superglm/profiling/_reporting.py`; `src/superglm/model/report_ops.py:259-275`; `src/superglm/inference/metrics.py:1593-1609`; `src/superglm/editor/widget.py` (`_profile_trace_rows` ≈1110, `_profile_estimate_payload` ≈1129, and the `profile_options` path ≈782); `src/superglm/editor/server.py:378-400`; `src/superglm/editor/app/index.html` (≈530, the profile method and φ-method controls); `src/superglm/editor/app/summary.js` (≈155–156)
- Test: `tests/test_tweedie_nb_characterisation.py`, the editor tests that reference `method`/`phi_method`, and a new publication test in `tests/test_tweedie_nb_characterisation.py`

**Interfaces:**
- Consumes: `search_power`, `profile_phi_at`, `TweedieProfileResult`, `Interval` (Task 6).
- Produces:
  - `SuperGLM.estimate_p(X, y, sample_weight=None, offset=None, *, fit_mode="fit", p_bounds=(1.05, 1.95), xatol=1e-3, ci_alpha=None, max_reml_iter=None, progress_callback=None) -> TweedieProfileResult`. It also takes `search_fit_mode=None` only if Task 1's decision (b) kept it.
  - `profile_ops._publish_profiled_family(model, X, y, sample_weight, offset, *, fit_mode, family, parameter, value, progress) -> final_model`. This is the shared search → final refit → install skeleton; Task 8 reuses it for θ.
  - `_reporting.cached_tweedie_profile_ci(result, alpha) -> (interval | None, status)`, where status ∈ {"available", "censored", "not computed"}.

- [ ] **Step 1: Write the failing publication tests** in `tests/test_tweedie_nb_characterisation.py`:

```python
def test_published_phi_is_profiled_at_the_published_mean(characterisation_case):
    from superglm.profiling.tweedie import profile_phi_at
    model, X, y = characterisation_case("zeros90")
    result = model.estimate_p(X, y, fit_mode="reml")
    expected = profile_phi_at(np.asarray(y, float), model.predict(X), np.ones(len(y)), result.p_hat)
    assert result.phi_hat == pytest.approx(expected.phi, rel=1e-12)
    assert model.result.phi == result.phi_hat
    assert result.nll == pytest.approx(expected.criterion / len(y), rel=1e-12)


def test_estimate_p_refuses_nonunit_frequency_weights(characterisation_case):
    model, X, y = characterisation_case("zeros90", weight_semantics="frequency")
    with pytest.raises(ValueError, match="frequency"):
        model.estimate_p(X, y, sample_weight=np.full(len(y), 2.0))


def test_estimate_p_signature_is_the_slim_one():
    import inspect
    from superglm import SuperGLM
    parameters = set(inspect.signature(SuperGLM.estimate_p).parameters)
    assert {"method", "phi_method", "kwargs"}.isdisjoint(parameters)
```

The `characterisation_case` fixture accepts `weight_semantics` and forwards it to the `SuperGLM` constructor.

- [ ] **Step 2: Run to confirm they fail.** Run: `uv run pytest tests/test_tweedie_nb_characterisation.py -k "published or frequency or signature" -q`. Expected: FAIL.

- [ ] **Step 3: Rewire `profile_ops.estimate_p`.**
  - Keep master's input validation (`_validate_entrypoint_input`, `_resolve_profile_fit_mode`, `_validate_profile_selection_mode`, the `max_reml_iter` checks) and the frequency-weight refusal from master's `estimate_tweedie_p`.
  - Replace the call to `estimate_tweedie_p` with `search_power(profile_workspace.model, X, y, sample_weight, offset, fit_mode=resolved_search_mode, p_bounds=p_bounds, xatol=xatol)`.
  - Extract the search → final refit → install sequence (master lines 186–301, and the parallel block in `estimate_theta`) into `_publish_profiled_family`. The ONLY differences between p and θ are the family, the parameter name used by `_publication_mode_failure`, and the post-refit synchronisation callback.
  - Replace `_reprofile_published_dispersion` with:

```python
def _reprofile_published_dispersion(result, y_arr, weights, mu) -> None:
    """Profile phi at the PUBLISHED mean; the searched curve keeps search_nll for the CI."""
    solved = profile_phi_at(y_arr, mu, weights, result.p_hat)
    result.phi_hat = solved.phi
    result.nll = solved.criterion / float(len(y_arr))
```

  - `_synchronize_tweedie_profile_refit` keeps its body minus the reprofile plumbing, and calls the function above with `public_mu`.
  - Delete `_installed_tweedie_profile_copy` and `_TWEEDIE_PROFILE_SHARED_RUNTIME_FIELDS`. Install `copy.copy(result)` with `_ci_cache` copied as a new dict and `_objective` shared.
  - Delete the `phi_method` and `method` plumbing.

- [ ] **Step 4: Update `api.py` `estimate_p`** to the Produces signature, with its docstring listing only the surviving parameters.

- [ ] **Step 5: Update the summary and reporting.**
  - `_reporting.py` shrinks to three functions:
    - `cached_tweedie_profile_ci(result, alpha)`: reads `result._ci_cache.get(float(alpha))` and returns `((lower, upper), "censored" if either flag else "available")`, or `(None, "not computed")`;
    - `tweedie_profile_report_identity(result, alpha)`: `(id(result), p_hat, phi_hat, nll, status, interval)`;
    - one small float-hex helper if needed.
  - Delete `tweedie_profile_method_label` and `_uses_density_approximation`.
  - In `report_ops.py` and `metrics.py`, delete the `tweedie_p_method` and `nb_theta_method` keys.

- [ ] **Step 6: Update the editor.**
  - `widget._profile_trace_rows` returns `result.evaluations.to_dict("records")` for Tweedie. For NB it reads `result.evaluations`, the same frame from Task 8. Until Task 8 lands, keep the NB `cache` branch.
  - `server._profile_options` allows only `{"fit_mode", "xatol", "p_bounds", "theta_bounds"}`, plus `"search_fit_mode"` only if decision (b) kept it.
  - Remove the profile method and φ-method `<select>` elements from `index.html` and the two `payload.method`/`payload.phi_method` lines from `summary.js`.
  - Search the editor for any remaining reference: `grep -rn "profileMethod\|profilePhiMethod\|phi_method\|grid_refine" src/superglm/editor` must return nothing.

- [ ] **Step 7: Run the focused tests.** Run: `uv run pytest tests/test_tweedie_nb_characterisation.py tests/test_tweedie_p_recovery.py -n 8 -q`, then `uv run --with plotly pytest tests -k "editor and (profile or estimate)" -n 8 -q`. Expected: the characterisation and recovery tests pass. Editor tests that assert removed options are updated to the surviving ones, and each change is listed in the commit.

- [ ] **Step 8: Commit.**

```bash
git add -u src/superglm tests
git commit -m "feat: slim estimate_p publication, summary and editor controls

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: NB2 θ on the shared machinery

**Files:**
- Replace: `src/superglm/profiling/nb.py` (≤ 320 lines)
- Modify: `src/superglm/model/fit_ops.py` (`_maybe_estimate_nb_theta` ≈1007, `_refine_nb_theta_to_reml_fixed_point` ≈1038, `_nb_joint_nll` ≈1181); `src/superglm/model/profile_ops.py` (`estimate_theta`, `_theta_estimate_payload`, `_cached_ci`); `src/superglm/model/api.py` (`estimate_theta` ≈1390); `src/superglm/model/report_ops.py`; `src/superglm/inference/metrics.py`; `src/superglm/editor/widget.py` (the NB trace branch)
- Test: `tests/test_nb_theta_estimation_correctness.py` (unchanged oracles), `tests/test_nb2.py`, `tests/test_tweedie_nb_characterisation.py` (the θ arm)

**Interfaces:**
- Consumes: `RecordedObjective`, `likelihood_ratio_interval`, `Interval`, `profile_plot` (Task 5); `fit_ops._solve_coefficients(..., beta_init, intercept_init)` (Task 6); `profile_ops._publish_profiled_family` (Task 7); `superglm.distributions.NegativeBinomial.log_likelihood` and `prior_weight_log_density`.
- Produces:
  - `theta_score(y, mu, weights, theta, *, weight_semantics) -> float`: master's `_theta_profile_score` verbatim, including the asymptotic branch and its docstring derivation.
  - `solve_theta(y, mu, weights, theta_start, *, weight_semantics, bounds) -> ThetaSolve(theta, at_lower, at_upper)`: master's `_theta_ml`, with `n_score_evaluations` dropped.
  - `nb_nll(y, mu, weights, theta, *, weight_semantics) -> float`: the mean NLL from the FAMILY density only. It uses `prior_weight_log_density` under prior and `NegativeBinomial(theta).log_likelihood(y, mu, weights)` under frequency, divided by `dispersion_likelihood_size`.
  - `estimate_nb_theta(model, X, y, sample_weight, offset, *, theta_bounds=(1e-8, 1e8), xatol=1e-2, maxiter=30) -> NBProfileResult`: private use by fit_ops/profile_ops; not exported from the root.
  - `NBProfileResult(theta_hat, nll, converged, evaluations: DataFrame[theta, nll], warnings, _y, _mu, _weights, _weight_semantics, _ci_cache)` with `ci(alpha)`, `interval(alpha)` and `profile_plot(alpha, ax)`.

- [ ] **Step 1: Write the failing tests** (the θ arm) in `tests/test_tweedie_nb_characterisation.py`:

```python
@pytest.mark.parametrize("row", FIXTURE["estimate_theta"], ids=lambda r: f"{r['case']}-{r['fit_mode']}")
def test_estimate_theta_matches_master(row, characterisation_case):
    model, X, y = characterisation_case(row["case"])
    result = model.estimate_theta(X, y, fit_mode=row["fit_mode"])
    # theta_hat is published to six significant digits on both sides.
    assert result.theta_hat == pytest.approx(row["theta_hat"], rel=2e-6)
    # Master's frequency NLL used naive gammaln (drift <= 5e-8 relative at theta = 1e8, inventory);
    # the family density replaces it, so the NLL agrees to that drift, not to round-off.
    assert result.nll == pytest.approx(row["nll"], rel=1e-7)
    lower, upper = result.ci(0.05)
    assert lower == pytest.approx(row["ci95"][0], rel=1e-5) and upper == pytest.approx(row["ci95"][1], rel=1e-5)


def test_nb_nll_agrees_with_family_log_likelihood_at_large_theta():
    from superglm.distributions import NegativeBinomial
    from superglm.profiling.nb import nb_nll
    rng = np.random.default_rng(5)
    mu = rng.uniform(0.5, 3.0, 400)
    y = rng.poisson(mu).astype(float)
    w = rng.integers(1, 4, 400).astype(float)
    for theta in (1e2, 1e6, 1e8):
        expected = -NegativeBinomial(theta).log_likelihood(y, mu, w) / w.sum()
        assert nb_nll(y, mu, w, theta, weight_semantics="frequency") == expected
```

- [ ] **Step 2: Run to confirm they fail.** Run: `uv run pytest tests/test_tweedie_nb_characterisation.py -k "theta or nb_nll" -q`. Expected: FAIL (`nb_nll` is not defined).

- [ ] **Step 3: Write the new `profiling/nb.py`.**
  - Keep `NBThetaBoundWarning` and master's bound-hit warning text.
  - `estimate_nb_theta` keeps master's alternation (moment start on the first pass, warm-started θ thereafter, six-significant-digit publication). Replace the duplicated solver dispatch (master 799–861) with `fit_ops._solve_coefficients(model, y, w, offset, penalty=..., lambda2=configured_lambda2(model), has_lambda1_targets=model_has_lambda1_targets(model), max_iter=model._max_iter, tol=model._tol, record_diagnostics=False, convergence=model._convergence, beta_init=warm_beta, intercept_init=warm_intercept)`, with `model._distribution = NegativeBinomial(theta)` set before each call.
  - Master passed `reml_penalties` into `fit_irls_direct` on the direct path, which `_solve_coefficients` does not. Compare θ̂ on both mgcv NB fixtures before and after. If θ̂ moves beyond the fixture tolerance, add `reml_penalties=None` passthrough to `_solve_coefficients` and pass it; otherwise drop it and say so in the commit.
  - Fix master's `_theta_moment_start` so the prior-weight arm uses the prior-weight moment equation: solve Σ w((y−μ)² − μ/w) = Σ μ²/θ; the frequency arm stays as it is. It remains a start value only.
  - Build `NBProfileResult.interval(alpha)` with `likelihood_ratio_interval(lambda t: nb_nll(...), theta_hat, nll, (min(0.01, theta_hat/100), max(500, theta_hat*100)), alpha=alpha, scale=dispersion_likelihood_size(...), rtol=1e-6)`. Those are master's range and tolerance. `ci(alpha)` returns `(lower, upper)`.
  - `profile_plot` uses the shared `profile_plot` over a 40-point log grid between the interval bounds (evaluated on demand, since it is O(n) each).
  - The `n_evaluations` and `cache` fields are replaced by `evaluations`.

- [ ] **Step 4: Update fit_ops and profile_ops.**
  - `_maybe_estimate_nb_theta` drops `contract_already_checked`: `estimate_nb_theta` no longer checks the weight contract, because every caller has already run `check_weight_contract` at its entry point (`fit` via `validate_fit_input`, `estimate_theta` via `_validate_entrypoint_input`). Verify that with grep, and note it in the commit.
  - `_refine_nb_theta_to_reml_fixed_point` uses `solve_theta` and `nb_nll`, and builds the refreshed `NBProfileResult` with `evaluations` extended by one row per refit.
  - Delete `_nb_joint_nll`: inline `nb_nll(...)` at its one call site.
  - `profile_ops.estimate_theta` goes through `_publish_profiled_family`.
  - `api.estimate_theta` gets the explicit signature `(X, y, sample_weight=None, offset=None, *, fit_mode="fit", theta_bounds=(1e-8, 1e8), ci_alpha=None, progress_callback=None)`.

- [ ] **Step 5: Run the focused tests.** Run: `uv run pytest tests/test_nb_theta_estimation_correctness.py tests/test_nb2.py tests/test_tweedie_nb_characterisation.py tests/test_profile_scalar.py -n 8 -q`. Expected: the mgcv θ oracles pass unchanged and the θ arm passes. Tests referencing `_nb2_nll`, `_theta_ml`, `profile_ci_theta` or `cache` are listed for Task 9.

- [ ] **Step 6: Commit.**

```bash
git add -u src/superglm tests
git commit -m "feat: NB2 theta on the shared profile machinery and the family density

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Delete the old code; exports, harness, docs and tests

**Files:**
- Delete: `src/superglm/_tweedie_profile_kernel.py`
- Move: `src/superglm/profiling/harness.py` → `benchmarks/_harness.py`. Update the imports in `benchmarks/profile_superbooster_interactions.py`, `benchmarks/profile_structured_credibility.py` and `tests/test_profiling_harness.py`, and delete the unreferenced `dataclass_payload`.
- Modify:
  - `src/superglm/__init__.py:133-150` (exports) and `:189-198` (`warmup()` calls `superglm._tweedie_series.warmup`)
  - `src/superglm/profiling/__init__.py`
  - docs: `docs/api/families-and-links.md`, `docs/api/internals.md`, `docs/api/model/fit.md`, `docs/api/warnings-and-exceptions.md`, `docs/explanation/families-and-weights.md`
  - notebooks: `docs/examples/tweedie_profile_estimation.ipynb`, `docs/examples/editor_demo.ipynb`
  - `tests/test_tweedie_profile_docs.py`
- Tests: delete or rewrite every test that pins removed internals (rule below).

**Interfaces:**
- Produces:
  - Root exports: `TweedieProfileResult`, `NBProfileResult`, `NBThetaBoundWarning`, `PublicationModeError`, `tweedie_logpdf`, `generate_tweedie_cpg`, `TweedieLSS`, `NegativeBinomialLS`.
  - `superglm.profiling` exports: `TweedieProfileResult`, `NBProfileResult`, `NBThetaBoundWarning`.

- [ ] **Step 1: Exports.**
  - `superglm/__init__.py` imports `tweedie_logpdf` and `generate_tweedie_cpg` from `superglm._tweedie`, and the two results and the warning from `superglm.profiling`.
  - Remove from the import block and `__all__`:
    - `estimate_tweedie_p`, `estimate_nb_theta`, `estimate_phi`;
    - `TweedieProfileCIDetails`, `TweedieProfileCIEndpoint`, `TweedieProfileCIEvaluation`, `TweedieProfileCIDensityProvenance`;
    - `profile_ci_p`, `profile_ci_theta`.

- [ ] **Step 2: Sweep for the old rule's fingerprints.** Every pattern below must be gone from `src/`, `docs/` and `benchmarks/`. Tests are handled in Step 3. Run:

```bash
grep -rnE "_tweedie_profile_kernel|_evaluate_tweedie_density|_prepare_tweedie_density|_profile_phi|_profile_ci_p_detailed|estimate_tweedie_p|estimate_phi\b|profile_ci_p\b|profile_ci_theta|TweedieProfileCI|phi_method|grid_refine|profile_opt|joint_ml|wright_bessel|t_arg_limit|search_trace|density_exact|saddlepoint|_nb2_nll|_theta_ml\b|TweedieScaleProfileData|contract_already_checked" src docs benchmarks --include=*.py --include=*.md --include=*.ipynb --include=*.js --include=*.html
```

Expected: no hits in `src/`. In `benchmarks/`, only the Task 1 characterisation script may still name master's internals: it documents what it measured and runs only against a master checkout. Say so in its module docstring. Old benchmarks that call removed APIs (`tweedie_profile_end_to_end.py`, `profile_tweedie_reml_fit.py`, `tweedie_reml_search_cost.py`) are updated to the new API so Task 10 can run them.

- [ ] **Step 3: Tests.**
  - For every test file matched by `grep -lE "<the Step 2 pattern>" tests/`, delete a test when the behaviour it pins no longer exists (search methods, density provenance, branch masks, CI record classes, Pearson `phi_method`, trace columns). Rewrite it against the public maths when the behaviour survives (logpdf values, φ̂, p̂, CI endpoints, θ̂, warnings on real states).
  - Each rewritten behavioural test records, in its commit message, the mutation that reddens it: a sign flip in `solve_log_phi`'s score, a dropped `count` multiply, an off-by-one peak in the series, and so on. Run each mutation once to confirm, then revert.
  - Heavy files: `test_tweedie_profile.py` (6,200 lines), `test_profile_ci.py`, `test_tweedie_numerics.py`, `test_tweedie_profile_performance.py`, `test_tweedie_profile_reference.py`, `test_reml_search_infeasible_mode.py`, `test_reml_search_dm_cache.py`, `test_profile_structured_retention.py`, `test_constrained_fit_profile.py`, `test_tweedie_generator.py` (its bitwise legacy pin stays; message-text pins become `pytest.raises(ValueError)` without `match`).
  - `test_tweedie_profile_reference.py` pins p̂ = 1.1968971098776182 ± 2e-4. Keep the value and the tolerance; drop its `method == "joint_ml"` and evaluation-count assertions.

- [ ] **Step 4: Docs and notebooks.**
  - Replace every `method=`, `phi_method=`, `search_fit_mode=` (unless kept), `trace_plot`, `ci_details` and `search_trace` mention with the surviving API.
  - In `families-and-weights.md`, the estimation section says in plain language:
    - p is chosen by maximising the profile likelihood, refitting the mean at each candidate p;
    - φ is the maximum-likelihood dispersion at that mean;
    - the interval inverts the likelihood-ratio test;
    - and it cites Dunn & Smyth (2005).
  - `internals.md` drops the CI record classes. Re-run the notebooks' affected cells to refresh their outputs: `uv run jupyter nbconvert --to notebook --execute --inplace <nb>`.

- [ ] **Step 5: Durations and lint.** Run `uv run python scripts/run_test_suite.py --store-durations`, or whatever this repo uses to regenerate `.test_durations`; check `scripts/run_test_suite.py --help`. Then run `uv run ruff check src/ tests/ benchmarks/` and `uv run ruff format --check src/ tests/`.

- [ ] **Step 6: Build the docs in strict off-mode.** Use the repo's docs build command in its off (no-execution) mode, as the dev-ci docs job runs it. See `.github/workflows` for the exact invocation. Expected: no warnings.

- [ ] **Step 7: Commit.**

```bash
git add -A src benchmarks docs tests .test_durations
git commit -m "refactor: delete the old Tweedie/NB2 profilers and their internal-pinning tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Evidence — performance, line counts and the full suite

**Files:**
- Modify: `benchmarks/tweedie_nb_rebuild_receipts.json` (add `"after"`)
- Output: the PR-body evidence section (a text block returned to the main session)

- [ ] **Step 1: Timing.** With pinned thread pools, re-run every Task 1 Step 5 case three times, interleaved with a checkout of the baseline code built in a second worktree:
  - Run `git worktree add ../tweedie-baseline <baseline commit>` from this worktree, using a relative path under `.claude/worktrees/`.
  - Use the baseline worktree's own venv (`uv run` inside it), or `PYTHONPATH=<worktree>/src`. Verify which tree was imported by printing `superglm.__file__` in each run: the shared `.venv` editable install points at the main checkout.
  
  Record median wall, CPU, peak RSS, candidate-fit count, φ-solver passes per candidate, and backend. Also the private dataset (scratch only).

- [ ] **Step 2: The time breakdown after.** Re-profile `estimate_p` on the §9 cases into the same three buckets: candidate fits, φ/density, everything else. Criterion 9: every case is faster than baseline, and the φ/density plus bookkeeping share is reported before and after.

- [ ] **Step 3: The `lambda2_init` measurement from Task 6.** Time the §9 REML case with and without the warm lambdas, and compare p̂. Keep the warm lambdas only if they are faster and p̂ moves by less than `xatol`; otherwise remove the line and re-run the recovery tests.

- [ ] **Step 4: The lgamma table.** If the profile shows the kernel's lgamma table build at more than 10% of φ-solver time on any case, hoist it into `TweedieRows` (build once per (p, max mode)) and re-measure. Otherwise leave it.

- [ ] **Step 5: Line counts.** Run:

```bash
git diff --stat 9e0fe6f9 -- src | tail -1
git diff --stat 9e0fe6f9 -- tests | tail -1
wc -l src/superglm/_tweedie_series.py src/superglm/_tweedie.py src/superglm/profiling/*.py src/superglm/model/profile_ops.py
```

Also count the Tweedie part of `reml/scale.py` and the NB functions in `fit_ops.py`, using the same `ast` measurement as the spec §2 table. Criterion 1: ≤ 3,000.

- [ ] **Step 6: Full checks.** `uv run python scripts/run_test_suite.py` (the full suite), `uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/`, `uv lock --check`, `uv pip check`, `uv run python run_test.py`. Also run the freMTPL2-anchored suites with `SUPERGLM_REQUIRE_DATA=1` if `data/` holds the freMTPL2 files; fetch them with `uv run python scripts/fetch_fremtpl.py --dest data/` otherwise. Expected: all green. Any red is diagnosed to a mechanism before it is called pre-existing.

- [ ] **Step 7: Commit the receipts.**

```bash
git add benchmarks/tweedie_nb_rebuild_receipts.json
git commit -m "bench: receipts after the Tweedie/NB2 profiling rebuild

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11 (gated): TweedieLSS onto the shared series

**Files:**
- Modify: `src/superglm/_tweedie_series.py` (add p-derivative channels), `src/superglm/distributional/kernels/_tweedie_numba.py`, `src/superglm/distributional/kernels/tweedie.py`

**Gate, all three required, or revert the task and record the follow-up:**
1. `tests/test_tweedie_lss_kernel.py`, `tests/test_tweedie_lss_family.py`, `tests/test_tweedie_lss_distribution_functions.py` and `tests/test_tweedie_lss_vertical_slice.py` pass with their tolerances unchanged.
2. The TweedieLSS complete-fit wall time on the LSS benchmark (the fit in `tests/test_tweedie_lss_vertical_slice.py`, scaled to 100k rows in a scratch script) is ≤ 1.05× baseline.
3. The net source lines of the two LSS kernel files fall.

- [ ] **Step 1: Map the LSS kernel's outputs.** For each `KERNEL_*` status and each derivative channel `_positive_row` returns, write down the series quantity it needs: log W, E[J], Var[J], and for p the per-term digamma/trigamma sums that master's `_exact_profile_statistics_kernel` computed (`_series_term_derivatives`). Write the mapping into the commit message.
- [ ] **Step 2: Extend `_tweedie_series.py`** with `series_p_moments(log_t, a, dlog_t_dp, d2log_t_dp2) -> (ok, log_w, dlogw_dp, d2logw_dp2, d2logw_dp_dlogphi)`. Accumulate term derivatives in the same outward walk (reuse `_row_moments`'s structure, adding the per-term digamma/trigamma channels from master's `_series_term_derivatives`). Add a finite-difference test against `series_moments` in `tests/test_tweedie_series.py`.
- [ ] **Step 3: Route `_positive_row`'s series summary through the new kernel.** Keep the LSS status codes that describe states of the LSS point evaluation; delete those that described the old series' internal failures.
- [ ] **Step 4: Run the gate.** If it fails, `git revert` this task's commits and write the follow-up into the PR body.
- [ ] **Step 5: Commit.**

```bash
git add -u src/superglm tests/test_tweedie_series.py
git commit -m "refactor: TweedieLSS kernel on the shared Dunn-Smyth series

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Self-review record

- **Spec coverage, § by §:**
  - §1 criteria 1–9: Task 10 (1, 6, 7, 9); Tasks 2–4 (2, 3); Tasks 4 and 8 (4); Tasks 3, 6 and 8 (5); Task 6 (8).
  - §3 methods: Tasks 2, 3, 5, 6 and 8.
  - §4 identity: Task 3 (`solve_log_phi`, `tweedie_logpdf`) and Task 4.
  - §5 architecture: the file map.
  - §6 API: Tasks 7, 8 and 9.
  - §7 evidence: Tasks 1, 2, 3, 6, 8 and 9.
  - §8 stages: Tasks 1 to 11.
  - §9 performance: Tasks 1 and 10.
  - §10 errors: Review Focus plus Tasks 3, 5 and 6.
  - §11 scope: Task 11 is gated.
- **Consistency:**
  - `TweedieRows.prepare(y, weights, p, *, frequency)` is the same in Tasks 3, 4 and 6.
  - `solve_log_phi(rows, deviance, nullity)` returns `PhiSolve(phi, criterion, curvature, n_passes)` everywhere.
  - `likelihood_ratio_interval(..., alpha=, scale=, rtol=)` is the same in Tasks 5, 6 and 8.
  - `_solve_coefficients(..., beta_init, intercept_init)` is the same in Tasks 6 and 8.
- **Placeholders left deliberately:** Task 6's φ and CI tolerances are marked "derive before asserting", with the formula given, because their numeric value depends on the fixture's recorded curvature.
