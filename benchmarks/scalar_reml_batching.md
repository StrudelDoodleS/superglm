# Scalar REML derivative batching

The exact French planted-noise selection fit's median time fell from
59.106 to 42.246 seconds, a 28.525% reduction across three alternating
baseline/candidate pairs. Training and test predictions, lambdas, term EDFs
and objective histories were bitwise identical. Deviance, total EDF,
convergence, iteration counts and reported solver backend also matched.

The [measurement receipt](scalar_reml_batching_receipt.json) records every
fit's time, peak RSS, solver metadata, prediction-array hashes, source
digests and profile counts. These measurements isolate the scalar batching
change in commit `8da1fe0958c96a38731107dbbc6b56977651f3b5` from its parent
`6370652d3d8acb1ba4f87a1830cecf829f1258d2`.

## Complete-fit comparison

| Exact scalar model | Baseline, seconds | Candidate, seconds |
| --- | ---: | ---: |
| Selection, pair 1 | 59.106 | 42.269 |
| Selection, pair 2 | 60.449 | 42.246 |
| Selection, pair 3 | 58.362 | 42.146 |
| Selection, median | 59.106 | 42.246 |
| Without selection, one control pair | 24.900 | 24.037 |

The control showed no regression in this pair. One pair does not establish
a repeatable speedup for that model. Profiling was separate from these timings.

These are the supplied constrained Poisson models on 508,509 training
policies and 169,504 test policies. They retain the original splits, knots,
supports, exposure offsets, noise seed and increasing bonus-malus constraint.
Selection uses nine penalties; the control uses five. Both have 88
coefficients including the intercept. Data preparation and prediction are
outside the fit clock.

The selection model returned train deviance 158879.16405948185, test deviance
52859.53998312188 and EDF 67.45881165896706 in every run. Each used ten outer
iterations and nine line-search fits. Both convergence flags were true in
all eight fit receipts.

## Memory, reuse and dispatch

Selection peak process RSS was 1169.785 to 1170.035 MiB before the change and
1169.453 to 1169.949 MiB after it. The control measured 1169.547 and
1169.727 MiB. Process peak includes data preparation and can occur outside
the derivative calculation.

The nine derivative-weight arrays occupy 34.917 MiB. The batch allowance
also includes a Gram accumulator, compensation and symmetric output per
direction, bringing this case to 36.476 MiB. The 64 MiB limit applies to
retained batch arrays. Completed derivative matrices, shared row chunks and
single-direction product scratch are separate costs.

Each correction owns its pending weights and clears them after the batch.
Each row block is expanded and centered once, then reused across directions
with different weights. No full observation-by-coefficient design is retained.
The caller reduces the batch width when necessary and uses the existing
serial route if fewer than two directions fit. Single-direction corrections
also retain that route.

The separate profiles explain the complete-fit change:

| Operation | Baseline calls | Candidate calls |
| --- | ---: | ---: |
| Weight correction evaluations | 10 | 10 |
| Derivative design traversals | 90 | 10 |
| Derivative design-block expansions | 5,670 | 630 |
| Coefficient-system centered Gram/RHS builds | 39 | 39 |
| Total design-block expansions | 8,127 | 3,087 |
| First-order mean-derivative transpose products | 90 | 0 |

All 90 signed derivative matrices remain necessary. Each direction keeps
the same BLAS product, row order and compensated reduction. The measured
gain combines row-block reuse, removal of unused zero-response RHS products,
and removal of mean derivatives that have no first-order consumer. This
experiment does not separate those contributions.

Inclusive correction time in the profiles fell from 45.682 to 21.351 seconds;
inclusive design expansion time fell from 18.101 to 5.988 seconds. These
call-stack totals overlap and must not be added together.

Every timed fit reports the Gram backend in both the final result and the
internal REML profile. The separate profiles each record 13 coefficient
solver calls and zero calls to `certify_centered_factor`, the routine called
by every iteration of the local experimental coefficient-QR route. Other
QR factorizations still occur elsewhere in the solver.

## Source and environment limits

Both timed source snapshots include the same local compact coefficient-QR
experiment in `model/api.py` and `solvers/irls_direct.py`. That experiment is
excluded from the implementation commit. Reconstructing the candidate from
the frozen baseline plus only the two committed batching files reproduces
the candidate source digest in the receipt. The dispatch evidence above
supports attributing the paired difference to batching.

These timings therefore compare controlled source snapshots, not pristine
checkouts of the two commits. The 178 focused regression checks were also
run against exactly the proposed PR source without the local QR experiment;
all passed in 29.59 seconds.
After adding the stable-route guard, the same 178 checks passed again on
the proposed follow-up source in 30.68 seconds.

The runs used Python 3.13.14, NumPy 2.5.2 and SciPy 1.18.0. The installed
SuperGLM metadata version was 0.31.0. Every recorded native thread pool used
one thread. NumPy reported OpenBLAS 0.3.34.0.0 and SciPy reported OpenBLAS
0.3.31.dev. Dependencies were unchanged between variants.

## Reproduction and review

The external assurance runner is
`/home/max/superglm-assurance/benchmark/fremtpl/run_fremtpl.py`; the input is
its pinned `results/freMTPL2freq.csv.gz`. Their SHA-256 digests are in the
receipt. Use that runner's `preprocess(load_snapshot(...))`,
`split_indices`, `tutorial_edges` and `design_frame` functions, then call
`fit_superglm` with kind `gam_b_noise_select` or `gam_b_noise`,
`discrete=False`, response `ClaimNb` and offset `log(Exposure)`.
Measure the complete `fit_reml` call, then compare predictions on both splits,
deviance, EDF, lambdas, objective history, iterations and dispatch. Keep
outputs in scratch storage so the pinned assurance records remain intact.

The original measurement wrapper and raw artifacts remain locally under
`.superpowers/sdd/2026-09-11-select-performance/` and
`.superpowers/sdd/2026-09-11-scalar-reml-batch/`. The receipt retains their
hashes and the measurements needed to assess this result without those files.

The focused regressions check signed cancellation, translation, aliases,
chunk boundaries, compensation, scalar-gradient agreement and route
admission. Mutation controls expose raw-moment cancellation, repeated serial
materialization and loss of compensation. Second-order, structured,
gradient-only and well-scaled routes retain their existing behavior.

[Claude](https://github.com/StrudelDoodleS/superglm/pull/381#issuecomment-5632283530)
and [Codex](https://github.com/StrudelDoodleS/superglm/pull/381#issuecomment-5632310910)
reviewed the implementation commit and reported no correctness defects.
Claude's follow-up prompted this tracked evidence and
an explicit stable-route guard. Removing the unused serial RHS is deferred:
the approved first pass retains the serial route for single directions and
small batch budgets. Forcing a minimum batch width of one also needs an
explicit policy when one direction exceeds the batch budget. No full-fit
speedup is claimed for that suggestion.
