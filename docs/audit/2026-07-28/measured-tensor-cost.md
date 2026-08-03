# Measured tensor-interaction cost — real freMTPL2, exact vs discrete

Script: `benchmarks/benchmark_tensor_cost.py`. Model: 4 × `Spline(ps, k=10)` (DrivAge,
VehAge, BonusMalus, VehPower) + `Categorical(Area)`, Poisson/log, exposure as
`sample_weight`. Tensor variant adds one `ti(DrivAge, BonusMalus)` via
`_add_interaction` (spline×spline → TensorInteraction factory).

## Local performance certification

Wall-time certification is deliberately local-only. Hosted CI runs synthetic
tests of the gate logic, but it must never execute these benchmark producers or
use hosted-runner timing/RSS as a performance decision.

The committed `superglm-reference-box-2026-07-29` baseline has
`certification_enabled: false`. Its machine and software identity is not
sufficiently established, so all of its timing numbers and thresholds are
historical, non-certifying provenance. Even supplying its matching profile ID
must be refused before measurement artifacts are read. Typing an ID is only an
operator assertion; the gate does not inspect or authenticate hardware.

Certification becomes available only after a fresh calibration on a
deliberately selected, stable local machine and software/thread environment.
Choose a new profile ID and run the producers sequentially on an otherwise
quiet system:

```bash
export LOCAL_PERF_PROFILE="replace-with-new-local-profile-id"

uv run --with pyarrow python benchmarks/fetch_mtpl2.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --with pyarrow python benchmarks/benchmark_tensor_cost.py \
  --n 30000 --reps 5 --json-out /tmp/tensor_cost_local.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --with pyarrow python benchmarks/timing_30rep_superglm.py \
  --reps 30 --json-out /tmp/flagship_local.json
```

Use those fresh artifacts from that same environment to update the baseline
profile ID, timing references, ratios, repetition counts, numerical invariants,
and only facts actually known about the environment. Set
`certification_enabled` to `true` only after that calibration is reviewed.
Then—and only then—the local certification command is:

```bash
uv run python benchmarks/local_perf_gate.py \
  --machine-profile "$LOCAL_PERF_PROFILE" \
  --baselines benchmarks/results/local_perf_baselines.json \
  --tensor-json /tmp/tensor_cost_local.json \
  --flagship-json /tmp/flagship_local.json
```

The tensor references use five steady complete fits after one excluded warmup;
the flagship artifact uses its existing 30-repetition protocol and excludes
its first repetition from steady statistics. The JSON artifacts retain raw
complete-fit timings, numerical outputs, dimensions, backend dispatch, and
whole-process peak RSS where available. Peak RSS is provenance, not a gate.

A disabled, missing, mismatched, or unknown profile cannot certify performance.
Results from another machine must not be mixed into an enabled profile. The
gate also refuses to run when `GITHUB_ACTIONS` or the conventional `CI`
environment variable is truthy, before reading any baseline or measurement
file.

## Usage survey

| where | TensorInteraction refs |
|---|---|
| `benchmarks/` | **0 direct refs** (the tensor benchmark builds via `_add_interaction`) |
| `tests/` | **56 refs across 4 files** (test_interactions 34, test_discretize_fit 10, test_multi_penalty 9, test_tweedie_profile 3) |
| `src/` | 18 refs across 7 files |

**Tensors are first-class and actively worked on, not a corner feature.** Corroborating evidence:
`benchmarks/benchmark_tensor_ti_freq.py` exists *specifically* to track "the current `discrete=True fit_reml`
tensor-interaction failure mode", with `TIMEOUT_S = 120.0`, per-block instrumentation
(`block_diag_tensor_s`, `block_cross_tensor_own_margin_s`, `block_cross_tensor_main_s`,
`block_cross_tensor_tensor_s`, `block_cross_fallback_s`) and multi-tensor scaling cases. A dedicated REML mode
`interaction_mode="fast_candidate"` exists solely to cope with interaction cost.

## Wall time (n = 100,000)

| config | wall | p | vs baseline |
|---|---:|---:|---:|
| baseline, **exact** | 8.79 s | 41 | — |
| baseline, **discrete** | 1.41 s | 41 | — |
| + 1 tensor, **exact** | **63.33 s** | 122 | **7.2×** |
| + 1 tensor, **discrete** | **5.39 s** | 122 | **3.8×** |

**One tensor term costs 7.2× on the exact path and 3.8× on discrete. The exact/discrete gap widens from
6.2× (baseline) to 11.7× (with tensor).**

## Where the exact-path tensor cost goes (n=60k, 58.4 s, cProfile)

| cum s | % | site |
|---:|---:|---|
| 23.9 | 41% | `reml_w_correction` → `centered_signed_gram` (23.3) → `_moments_impl` (23.3) |
| 20.8 | 36% | └ **scipy sparse `__matmul__`** — the tensor block's Gram |
| 16.7 | 29% | `reml_linesearch` (overlaps: trials re-run full PIRLS) |
| 14.5 | 25% | `build_centered_system` → `_gram_any_sign` (14.4) |
| 14.4 | 25% | └ **`_group_matrix_core.py:676 gram` — 14.24 s of PURE SELF TIME** |
| 12.0 | 21% | `group_matrix.py:440 toarray` → `_group_matrix_core.py:681 toarray` (10.6) |
| 10.1 | 17% | `scipy.sparse._sparsetools.csr_matvecs` (built-in, tottime) |

**~38 s of 58 s (65%) is sparse-matrix Gram work on the tensor block** — split between the W-correction path
(23.3 s) and the PIRLS Gram path (14.4 s). This confirms the audit's dense-in-CSR row-Kronecker landmine
(`features/interaction.py:774-791`) is live and dominant on the exact path.

## Where the discrete-path tensor cost goes (n=60k, 3.47 s)

| cum s | % | site |
|---:|---:|---|
| 2.34 | 67% | `reml_pirls` |
| 1.46 | 42% | `irls_finalize` |
| **0.59** | **17%** | `irls_gram` |
| 0.09 | 2% | `reml_linesearch` |
| — | — | W-correction **absent by construction** |

**The discrete path's tensor Gram is already cheap (0.59 s of 3.47 s).** Bin-space accumulation is working
here. Discrete tensor cost is dominated by PIRLS iterations and terminal finalize, not Gram formation.

## Implications for the rerank

1. **The row-tensor identity is primarily an EXACT-path optimisation.** It attacks ~65% of exact-path tensor
   runtime. On the discrete path it can save at most ~0.5 s of a 5.4 s fit — the Gram is already 17%.
2. **The leverage-diagonal W-correction reformulation matters MORE with tensors**, because a tensor adds
   penalties (kron(S1,I) + kron(I,S2)) and q is the driver. It attacks the 41% W-correction directly.
3. **Interaction *discovery* does not need the row-tensor identity at all.** Screening via score statistics
   needs only the 2-D histograms, which the discrete path already builds cheaply. The identity is for
   *fitting* the winners on the exact path.
4. Projected combined effect on the 63.3 s exact tensor fit: leverage-diagonal (−~25 s), row-tensor Gram
   (−~9 s), cheap frozen-W line-search trials (−~10 s) → **roughly 12-18 s, ~4×**, narrowing the exact/discrete
   gap from 11.7× to ~3-4×.
