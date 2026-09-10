# Profile comparison for the numerical repairs

The [PR review follow-up](pr381_review_followup.md) records the later repairs and
fresh comparisons against the resumed baseline. Earlier measurements below
retain their original source identities.

The subsequent [family-arithmetic addition](family_arithmetic.md) uses the
strict candidate below as its baseline and records fresh complete-fit timing,
profile and dispatch comparisons. The measurements below remain attributed
to their listed sources.

## Final tensor candidate

The strict candidate `c7ee4477` takes 12.10698 seconds against the resumed
baseline `167873e1` at 11.27126 seconds. These are medians of five alternating
pairs in fresh processes with one native thread. The remaining difference is
0.83572 seconds, or 7.41%. Median absolute deviations are 0.36405 and 0.21715
seconds respectively. Median process peak RSS is 446.76 versus 404.67 MiB.

All ten fits converge in twelve smoothing iterations. The maximum prediction
difference is `4.14e-11`; relative objective difference is `4.37e-10`. Each side
has identical complete numerical output payloads across its five repetitions.
The profile reproduces every candidate result exactly, with matching source,
driver, helper and fixture hashes. The reported fit times come from separate
uninstrumented runs.

| Work in final profile | Calls | Cumulative seconds |
| --- | ---: | ---: |
| Complete fit | 1 | 13.435 |
| Certified penalty summaries | 16 from 141 context requests | 8.819 |
| Wide-product interface, including accepted exact native products | 199 | 5.677 |
| Dyadic product attempts | 155 | 5.571 |
| Dyadic slice preparation | 310 | 1.938 |
| Penalty-context construction | 2 | 2.264 |
| Group moments | 94 | 1.429 |
| Group diagonal Grams | 282 | 0.848 |
| Group cross-Grams | 282 | 0.438 |
| Coefficient fits | 14 | 1.184 |

Cumulative entries overlap. Strict certification remains the main cost;
the iteration count is unchanged. The changes are:

- Authenticate and reuse raw penalty support and the matching final summary
  across optimizer/finalizer contexts. Coordinate transport remains fresh.
- Calculate eligible wide products using sixteen exact float64 matrix products
  of 21-bit slices. Group equal exponents in native arithmetic and retain the
  original error allowance; unsupported cases use the original wide product.
- For the tensor's fully populated raw CSR basis, calculate the diagonal Gram
  through a zero-copy dense view and a native product. The previous route made
  50,850,000 scalar pair updates per tensor Gram. Genuine sparse and extreme
  cases keep their existing routes.
- Batch ordinary SSP cross-Grams as `Ri.T @ (Bi.T @ (W * Bj)) @ Rj` instead of
  4,230 column operations in each direction. The path uses the live raw basis,
  preserves range checks and limits temporary allocation.
- Use a native product to propose the initial factor on large positive penalty
  supports. Recompute all geometry and certificates under the strict policy.
  A numerical refusal retries the complete original proposal route. Small
  supports, zero faces and public full-result calculations keep that route.
- Request only the triangular QR result where its orthogonal factor was
  discarded. No transformed observation design is retained as a new cache.

The native proposal uses the original error bound. Experiments with a wider
bound were not adopted. This fixture still takes 0.84 seconds longer and uses
about 42 MiB more peak process memory than the resumed baseline.
The fixture explicitly uses REML tolerance `1e-6`; these times do not describe
the default `1e-9` tolerance or every model size.

The final receipts are under
`.superpowers/sdd/2026-09-10-numerical-resume/artifacts/complete-fit-tensor-v4-final/`
and `artifacts/profile-tensor-v4-final/`. The source digest is
`c7ee44778b9bf8cd424b48a21dbca6ba6ffeb8c88ef982ec0f743d86668c7aaf`.
The sections below retain the earlier diagnostic comparisons with their own
source identities.

## Earlier tensor checkpoint

The large scalar tensor slowdown came from the new penalty geometry and
certification path. Ordinary Gram and coefficient-fit work was almost unchanged
between these two earlier sources.
These profiles compare the resumed baseline (`167873e1`) with the repaired
checkpoint (`3292dc72`) on the same 2,000-row, 255-coefficient public REML fit.

Each side was profiled separately in a fresh process with one native thread.
The timing batch was paused until both profiles finished. Source, driver,
helper and data identities match the timed receipts, and the complete numerical
output payload from each profile equals all five corresponding timed fits.
The profiler itself is outside the production package.

| Work | Baseline calls | Repaired calls | Baseline cumulative seconds | Repaired cumulative seconds |
| --- | ---: | ---: | ---: | ---: |
| Complete profiled fit | 1 | 1 | 11.729 | 26.990 |
| Group Gram calculations | 282 | 282 | 8.458 | 8.497 |
| Moment evaluations | 94 | 94 | 9.823 | 9.829 |
| Coefficient fits | 14 | 14 | 5.207 | 5.100 |
| Centered-system construction | 46 | 46 | 4.974 | 4.870 |
| Penalty evaluations | 47 old evaluations | 17 certified summaries | 1.065 | 13.294 |
| Repaired penalty-context construction | — | 2 | — | 3.024 |

The candidate's 141 penalty requests produce 17 summaries: result reuse is
active. Both fits retain 12 smoothing iterations. The extra work is within
each evaluation and context construction, rather than extra optimizer steps.
These are diagnostic profile times. Nested cumulative times must not be added;
the separate five-pair timing medians are about 11.86 and 26.99 seconds.

The expensive repaired stack is:

```text
fit_reml
  optimize_direct_reml
    reml_laml_objective
      penalty context evaluate                 141 requests
        _evaluate_penalty_summary               17 evaluations
          _direct_candidate                     17 calls
          _reference_correct_once               17 calls
          fresh reference actions and certification
```

| Repaired function | Calls | Self seconds | Relevant callers |
| --- | ---: | ---: | --- |
| `_matmul_enclosed` | 147 | 6.580 | 68 materialization-check products; 17 correction products; 17 candidate products; 17 whitening products; 28 context/basis products |
| `_reference_root_actions` | 34 | 2.713 | Once before and once after the QR update per evaluation |
| `_direct_candidate` | 17 | 1.953 | Weighted-root projection and inverse-root construction, in addition to callees |
| `_triangular_solve` | 35 | 0.722 | Wide substitution for inverse-root actions |

`_matmul_enclosed` converts both operands to `longdouble` and performs the
signed product in that dtype on every call. Its magnitude-bound calculation
has a separate guarded native path. Materialization checks alone call the
enclosed product 68 times, consuming 3.055 cumulative seconds there.
The corrected geometry has different operands, so the before/after reference
actions are not automatically interchangeable. Any further reuse must preserve
that distinction and its error evidence.

The installed NumPy 2.5.2 reports 63 fraction bits and 16-byte storage for
`longdouble`, compared with 52 fraction bits and eight-byte storage for
float64. Its dtype display name is `float128`; that is not 128 bits of
significand precision on this machine. The matching
[NumPy source](https://raw.githubusercontent.com/numpy/numpy/v2.5.2/numpy/_core/src/umath/matmul.c.src)
disables BLAS for `LONGDOUBLE` and uses its scalar matrix-product loop.
This explains why many dense products through that path are expensive even
when the number of smoothing iterations is unchanged.

The profile calls neither `_dot2_value` nor `_inverse_gram_enclosed`. It therefore
does not attribute this regression to compiled compensated-dot fallback or
dense inverse output construction. The existing summary path already omits
those unused full outputs. Native QR is not a leading cost in this profile.

Raw profiles, callers, exact comparison data and receipts are preserved under
`.superpowers/sdd/2026-09-10-numerical-resume/artifacts/profile-paired-final-tensor/`.
The same files currently reside under
`/tmp/superglm-numerical-resume-9106rchs/profile-paired-final-tensor/`.
`comparison.json` records full source locations and unrounded values.

## Repeated SCOP penalty construction

The joint discrete SCOP fit has a separate cause. Both sources perform
42 coefficient fits and 140 Newton steps on the same 600-row, two-group
fixture. Its five-pair median grows from 0.3400 to 1.3410 seconds. Both
reach the configured 40-iteration smoothing limit; this comparison does not
claim smoothing convergence.

The repaired profile records 166 full penalty evaluations and 250 support
constructions. These counts follow directly from the callers:

- Two singleton groups, each used by 42 objectives and 41 derivative calls:
  `2 * (42 + 41) = 166` full evaluations.
- Two groups, each used by 42 nullity calculations: 84 more support builds.
- Total support constructions: `166 + 84 = 250`.

Each mode creates fresh penalty descriptors for the same latent `S_scop`.
That discards the ownership needed to retain checked support and unit-weight
summary results. The ordinary SSP penalty context describes different
coordinates and cannot substitute for this latent target.

The caller repair retains authenticated latent descriptors in the per-fit
SCOP context. It checks group placement, exact penalty contents,
reparameterization identity and descriptor integrity before reuse. Coefficient,
Jacobian and Hessian calculations stay current. A new inactive penalty does
not request a unit-weight summary; a refused replacement preserves the
previous valid entry.

The complete-fit profile now confirms two support builds, two unit-weight
summaries and no full-result evaluations. The single-group profile falls from
11 support builds and seven full evaluations to one support build and one
summary. Coefficient fits and Newton steps are unchanged in both cases.

Five alternating pairs compare the pre-cache source `3292dc72` with the
accepted cache source `6cd33c35`:

| Complete fit | Before median seconds | After median seconds | Before / after MAD seconds | Before / after peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: |
| Single SCOP | 0.07564 | 0.03810 | 0.00614 / 0.00367 | 335.57 / 335.71 |
| Joint SCOP, discrete | 1.21276 | 0.33774 | 0.03713 / 0.02131 | 377.53 / 377.38 |

Both joint SCOP fits stop at the 40-iteration smoothing limit without smoothing
convergence.

The joint median improves by 72.2% (3.59 times faster); the single median
improves by 49.6%. Each case's ten complete numerical output payloads are
identical, including coefficients, predictions, objective, selected smoothing
weights, work counts and convergence status. The existing joint smoothing
iteration-limit result is preserved. Every before/after profile also matches
all five corresponding timed fits. First-use work remains inside the measured
fit, and no timing or profile jobs overlap.

The change passes 564 affected integration tests. The 35 new tests also pass
on Python 3.12; independent review adds 34 checks on each Python version and
exact, high-precision, range and baseline-replay controls. Only the SCOP caller
and its new test file differ from the earlier numerical checkpoint.

The before profiles are in `artifacts/profiles-other-regressions/`; new profiles,
exact comparisons and timing receipts are in `artifacts/profiles-scop-v4/` and
`artifacts/complete-fit-scop-v4/` under the audit directory.
