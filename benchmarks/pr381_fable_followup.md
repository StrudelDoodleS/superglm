# Fable review follow-up

The [full-PR Fable review](https://github.com/StrudelDoodleS/superglm/pull/381#issuecomment-5634806983)
identified an unbounded weighted temporary in saturated SSP Grams. The fix
retains dense matrix products and bounds their observation-dependent workspace
to 64 MiB. It preserves the existing single-product order for small groups.
Coefficient-sized products and native BLAS workspace are outside that bound;
the pre-existing exceptional exact-arithmetic route remains unchanged.

The review covered base `66141d28` through `282bba39`. Its
[completed CI run](https://github.com/StrudelDoodleS/superglm/actions/runs/34601975527)
records Fable 5.1 at max effort, with no Opus substitution. It was a source
review, not a test execution or proof that every possible defect is absent.

## Complete-fit evidence

The baseline is the reviewed `282bba39`. Frozen candidate source adds only the
Gram fix and a working-weight docstring clarification; unrelated local QR work
is excluded. Full source digests, inputs, timings and observations are in
[the receipt](ssp_gram_memory_receipt.json).

All fits use the existing scalar Poisson tensor fixture: 255 coefficients plus
an intercept, including a 225-column raw tensor basis. Processes ran serially
with one native thread. Each source had one untimed REML warmup; three pairs
then alternated source order. The larger fixed-penalty case has one pair.
Timed fits contain no observers. RSS is the process high-water mark at fit
completion, including imports and fixture construction; it is not an isolated
Gram allocation measure.

| Complete fit | Baseline wall / CPU, s | Candidate wall / CPU, s | Baseline / candidate peak RSS, MiB |
| --- | ---: | ---: | ---: |
| 2,000 rows, REML; three-pair medians | 13.697 / 13.684 | 13.591 / 13.577 | 445.96 / 445.21 |
| 100,000 rows, fixed penalty 1; one pair | 7.287 / 7.246 | 7.356 / 7.327 | 1482.69 / 1482.66 |

All recorded small-fit numerical outputs and histories are bitwise identical.
The large fits both converge in five coefficient iterations with rank 255.
Maximum prediction and coefficient differences are `1.88e-13` and `8.39e-13`;
deviance differs by `1.46e-11` and EDF by `1.19e-12`. Other recorded outputs agree.
These measurements show no small-case timing regression and only a single
large-case timing observation. They demonstrate no reduction in whole-fit
peak memory and make no new C1 latency claim.

Separate untimed complete fits match their timed numerical outputs exactly.
The execution trace observes six completed `100000 × 225` weighted products
before the fix. After it, those six Grams execute eighteen products with at
most 37,282 rows each. The weighted temporary therefore changes from 171.66 MiB
to at most 64 MiB. Both fits retain twelve CSR Gram calls for the two marginal
groups; the tensor does not fall back to scalar CSR accumulation. Native pool
samples report one thread and no observer errors. The trace observes completed
NumPy products and pool state; it does not intercept native BLAS symbols.

Reproduce with frozen package source selected through `PYTHONPATH`:

```sh
python -m benchmarks.multi_penalty_support --case scalar_tensor --tensor-rows 2000 --mode reml --measure-time --label small --out small.json
python -m benchmarks.multi_penalty_support --case scalar_tensor --tensor-rows 100000 --mode fixed --measure-time --label large --out large.json
python -m benchmarks.saturated_ssp_gram_dispatch --case scalar_tensor --tensor-rows 100000 --mode fixed --label dispatch --out dispatch.json
```

Source and driver hashes bind the retained measurements. The full local receipts
remain under `.superpowers/sdd/2026-09-11-fable-followup/`; the tracked summary
contains their hashes. An initial observer recorded the products but sampled no
native pools; it is retained locally. The corrected observer adds sampling at
completed Gram products and was run separately from every timed fit.

## Disposition of the other findings

| Finding | Verified disposition |
| --- | --- |
| 2: nonfinite triple-product inputs | `NaturalLikelihoodEvaluation` rejects nonfinite Hessians, and inverse-link derivatives are checked before the helper. Existing family-contract and derivative tests cover these boundaries. The proposed inputs do not reach this private finite-input helper through those contracts. |
| 3: Poisson/sqrt clipping | `_fisher_rows` already uses the coherent unfloored response `(eta + y/eta)/2`, with dedicated zero and unrepresentable-response handling. `test_irls_working_rows.py` covers predictors down to `1e-150` with a floored supplied mean; `test_sqrt_irls_zero.py` covers complete fits. The shared-weight docstring now states this exception explicitly. |
| 4: result-time penalty reassessment | The cited saved Gaussian/shared, Gamma/shared and LSS tensor profiles have zero assessment-context evaluations. Their full result construction costs are 0.727, 0.821 and 1.011 ms. They do not attribute the cited slowdown to result replay. Repeated identical-weight evaluations already use a memo; transferable summary authority requires more than a scalar objective receipt. A wide fit with face transitions and artifact loading still needs a dedicated profile before changing this cache. |
| 5: Schur refusal | The certificate protects the pseudo-determinant. `test_structured_factor_extreme.py` includes acceptance at zero and `2**-30` null coupling, as well as refusal. An instrumented complete fit entering the rank-deficient Schur fallback during a collapsing-level trajectory remains a coverage item. No valid-fit regression was demonstrated. |
| 6: wide sum-to-zero refusal | The automatic IRLS route catches `SumToZeroIdentifiabilityError` and retries Gram. Existing factor-smooth tests cover automatic fallback, forced-structured refusal and public REML fallback. A complete wide near-boundary fit, including terminal reconstruction, remains a coverage item. The certificate is preserved. |
| 7: discarded observed geometry | An explicit Fisher fit can recompute observed geometry after a successful objective/step certificate. This is an optional reuse opportunity for that route; it does not affect the default observed-curvature C1 fixture. |
| 8: single-direction batching | The previously documented single-direction work and chunk-workspace accounting remain deferred. No new speed or whole-workspace bound is claimed. |
| 9: merit-error dictionary | A small dictionary survives for the fit's trial lifetime. The review found no incorrect lookup. Ownership cleanup remains optional and does not justify changing solver-state contracts in this memory fix. |
| 10: evidence and documentation | The audit now distinguishes checkpoint verification counts and states that the targeted comparator probe bundle is local. The opt-in manifest regression already exists in `test_distributional_history_schema.py::test_full_history_manifest_adds_only_the_explicit_retention_flag`. The roadmap now names initialization arrays among the remaining full-row owners. |

The profile values above come from the saved diagnostic profiles in
`.superpowers/sdd/2026-09-10-numerical-resume/artifacts/pr381-followup/complete-fits/`.
They are historical source-bound observations, not new timings or measurements
of wide fits with face transitions.

Validation covers 129 focused group-matrix cases, 153 family/derivative,
Poisson/sqrt and schema cases, and 32 benchmark-driver cases. The original
allocation regressions fail on the unfixed source; signed numerical cases use
exact Fraction targets and dimension-derived error bounds. The benchmark
driver's test double was updated for the new optional row-count argument;
its no-observer timing assertions are unchanged. Ruff and strict docs pass.
An independent source review approved the narrow production and regression diff.

## C1 completion remains separate work

Fable confirmed that compact history is implemented. The existing
[roadmap](../docs/ROADMAP.md) and
[performance plan](../docs/research/2026-09-discrete-performance-plan.md)
still require the following, in dependency order:

1. Replayable bounded-row input and prepared-likelihood storage, preserving one
   compiled model, weight semantics and live source authority.
2. Bounded initialization and live endpoint handling, with streamed comparisons
   and reuse that avoids unnecessary full-row passes.
3. Explicit backed or streamed result and serialization access, preserving
   requested eager outputs through deliberate materialization.
4. Matched complete-fit validation before 10-million- and 100-million-row pilots,
   recording numerical outputs, convergence, wall/CPU time, peak memory, actual
   dispatch, pass counts and storage traffic. Billion-row fitting remains unproven.
5. The separate 12.5-second million-row latency target and broader width/thread
   policy evidence. The compact-history pair was about 14.6 seconds.

Removing a prepared Gaussian carrier is a possible small next implementation
slice, worth `8*N` retained bytes. It leaves other full-row owners in place and
does not complete the end-to-end memory or latency requirements.
