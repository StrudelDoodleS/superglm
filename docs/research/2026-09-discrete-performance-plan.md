# Discrete execution performance plan

Status: implementation in progress. Baseline: `0a15736e88a317088bfd01e56933d45c58e4ac9a`
(production source unchanged since `5f994c8f6ac0501606594e2f36bfc0cd24050ec1`).

The immediate priority is efficient discrete execution before additional roadmap
capabilities. Preserve the working grouped, signed, coupled distributional
solver and its bounded row processing. A compact representation alone does not
establish a performance improvement: complete fits must supply the evidence.

## Stage 1: establish the work and numerical contracts

The code audit found that ordinary discrete pair dispatch limits histogram
allocation to five million cells but ignores the current row count and the
cost of contracting the bin table. An existing bounded row-panel contraction
can evaluate the same stored support representation. The audit also found
that chunked likelihood, predictor-change and terminal prediction passes
construct execution plans that they discard.

Use independent synthetic fixtures to demonstrate unnecessary dispatch and
plan construction before changing either path. Separate work-count assertions
from numerical tests. Numerical oracles evaluate the stored design, including
signed weights, rectangular blocks, centering and coefficient maps. Agreement
with an exact feature basis is a different question when bins approximate
continuous covariates.

## Stage 2: implement bounded execution improvements

1. Choose histogram or stored-support row contraction using row count,
   support dimensions and contraction work, preserving the allocation ceiling
   and favorable histogram reuse.
2. Separate predictor-value evaluation from geometry-plan construction.
   Keep chunk temporaries bounded; do not retain all materialized row designs.
3. Inspect residual complete-fit profiles before adding further structural
   reuse. Weighted reductions must be recomputed for each changing curvature
   channel. Endpoint numerical certificates retain their existing provenance
   requirements.
4. Following the first profile, replace avoidable tensor Gram/prediction
   temporaries with equivalent bounded contractions, and certify strictly
   increasing row selections without sorting them again. Retain existing
   validation and provenance for general selections.

## Stage 3: demonstrate complete-fit behavior

Compare the starting source and candidate on reproducible public synthetic
and insurance fixtures. Record complete-fit wall time from the worker's own
clock, process peak RSS, fitted numerical outputs, convergence diagnostics and
actual backend dispatch. Compare discrete before/after separately from the
discrete/exact representation comparison. Include cases favorable to support
compression and cases where histogram setup dominates.

Use fresh serial subprocesses with numerical thread counts fixed to one.
Retain raw receipts and disclose compilation/cache policy, order, repetitions
and background activity. Headroom/Kompress tool timings are not fit timings.
Pause other numerical work during timing windows and distinguish noisy
measurements from supported improvements.

## Stage 4: verification and review

Run focused mathematical, dispatch, bounded-memory and complete-fit regression
tests, followed by the repository checks appropriate to the final changes.
Review public artifacts for independent reproducibility and source provenance.
Update this plan and the roadmap with measured outcomes and remaining limits.
No new capability is promoted merely because a microbenchmark improves.

## Evidence and decisions

- The baseline focused distributional assembly, grouped assembly, chunking and
  discrete fit suites passed before implementation.
- Keep the versioned automatic chunk-size policy unchanged initially; remove
  avoidable work before changing its memory/performance tradeoff.
- Do not rebuild exact feature bases to make a discrete benchmark faster.
- No release version change or publication is part of this task.
- The first three-repeat public Gaussian comparison (20,000 rows, four knots,
  256 bins) rejected the initial arithmetic-only dispatch estimate: median
  discrete fit time increased from 1.912 s to 2.087 s with unchanged iteration
  counts. Raw receipts remain under
  `.benchmark-artifacts/discrete-performance/timing-gaussian/`. Background CPU
  was audited; these are local measurements with Headroom active.
- Kernel diagnostics explain the rejection: gathering and weighting row
  panels costs much more than the initial arithmetic count represented.
  Dispatch tests must use independently demonstrated favorable cases, rather
  than treating fewer arithmetic operations as proof of faster execution.
- A separate complete-fit profile identifies factored tensor Gram products,
  tensor prediction and repeated chunk preparation as remaining work. Review
  these before claiming that the discrete execution gate has been resolved.
- The revised histogram estimate includes setup and gather overhead and keeps
  histograms near the estimated crossover. Kernel diagnostics support this
  revision, but its acceptance still depends on complete-fit measurements.
- Defer a shared chunk-index slicing context: profile evidence favors simpler
  tensor and row-selection improvements first. Do not persistently cache
  marginal outer products while the underlying support tables remain mutable.
- The reviewed candidate `aef4ee7e` improves the same public Gaussian case:
  within the final window, median discrete time is 1.522 s versus 2.342 s at
  baseline (35% lower). Iterations remain 9 smoothing / 26 inner; the largest
  saved training-parameter difference is 2.1e-15. Candidate discrete repeats
  span 1.167–1.562 s, so retain the raw variation and do not infer a precise
  universal speed ratio. Larger and insurance comparisons remain in progress.
- Separate instrumentation confirms 326 reassociated tensor Gram calls and
  666 fused tensor prediction calls. Timed runs disable this instrumentation.
- On 100,000 rows with 32 observed values per covariate, median discrete time
  decreases from 3.753 s to 2.116 s. Candidate exact fitting takes 4.331 s;
  peak process RSS medians are 483 MiB discrete and 695 MiB exact. This fixture
  preserves the fitted covariates exactly, with no unseen holdout support
  values. Saved exact/discrete training parameters differ by at most 3.8e-15.
- Retain timing ranges: guest process CPU audits cannot establish exclusive
  access to the physical host. Subsequent timed workers also record their own
  fit CPU time alongside wall time to expose scheduling-related variation.
- The insurance execution gate remains open. On freMTPL2 Gamma severity with
  four copies of the training partition, discrete median time is 12.010 s
  versus 12.490 s at baseline and 11.077 s for candidate exact fitting.
  Saved outputs and the 7 smoothing / 18 inner iteration counts are identical.
- Profile comparison identifies six extra initial derivative evaluations in
  chunked smoothing: existing observed endpoint reuse accepts only dense
  execution. Extend reuse to aggregated chunked score/curvature at a certified
  unchanged point, rebuilding penalty-dependent quantities for each lambda.
  Retain terminal retry and certification rules. Check fixed inputs with a
  bounded-memory digest; unsupported representations conservatively recompute.
- A separate Gamma profile identifies repeated scalar origin-series work in
  initialization. Compile the bounded batch calculation with independently
  checked truncation/rounding behavior; preserve general-domain fallbacks.
  This is an execution improvement within the current scope, not a new family.

### Follow-up implementation checkpoint

- Gamma's four small-shape series now execute in compiled batches. A stronger
  geometric tail bound preserves the existing EPS/8 error budget; a regression
  demonstrates the old derivative-tail underestimate. Initialization evaluates
  exactly identical shapes once and retains `math.fsum` over the same row terms.
  The 22 new tests, 123 Gamma family tests and 274 shared-consumer tests pass;
  the latter run includes the optional high-precision dependency.
- Chunked smoothing reuses the raw likelihood score and observed curvature at
  a certified unchanged endpoint, rebuilding penalty-dependent quantities for
  the next lambda. The certificate checks live responses, prepared likelihood
  arrays, weights and semantics, offsets, design, links and family configuration.
  It covers all nine built-in families, retains only O(p²) numeric state plus
  a bounded-memory digest, and conservatively refreshes unsupported extensions.
  Fresh terminal evaluation and convergence certification are unchanged.
- The 147 focused solver/chunk tests pass, including changed-input refusals,
  cancellation and lifetime checks, and a three-parameter Tweedie comparison
  against a fresh solve after changing lambda. Independent mathematical and
  certificate reviews report no unresolved material findings.
- Repeat complete-fit comparisons from this new source checkpoint before
  closing the insurance gate. Earlier timing receipts remain historical evidence;
  kernel diagnostics alone do not establish a complete-fit improvement.

### Representation and integration findings

- The freMTPL2 severity fixture is an exact training-support representation:
  its three covariates have 73, 21 and 82 distinct values, below the 256-bin
  budget, and reconstruct every supplied training value exactly. Prediction
  evaluates the learned basis at the supplied holdout values, including the
  one unseen marginal value. Its exact/discrete arithmetic differences must
  not be described as discretization error. The support-32 synthetic fixture
  is also exact on supplied covariates; the continuous synthetic fixture uses
  approximate training bins. The comparison harness now uses a neutral label.
- The next insurance window at `6e611349` gives discrete complete-fit times
  4.214, 2.976 and 3.821 seconds, versus 13.383, 13.115 and 14.926 seconds
  before these changes. Exact candidate times are 2.962, 2.804 and 3.929 seconds.
  Same-representation saved numerical outputs are identical in every repeat.
  These ranges overlap; repeat the final comparison before drawing a firm
  conclusion about the residual exact/discrete difference.
- Instrumentation confirms seven accepted chunked endpoint reuses, versus
  seven refusals at baseline. Remaining costs include repeated chunk-plan
  preparation and separate trial-value/derivative evaluation. Do not compare
  isolated assembly timers: dense derivative work is attributed to likelihood
  evaluation, while chunked derivative work is inside geometry assembly.
- Broad regression testing caught a forbidden solver-to-family dependency in
  the first certificate implementation. Built-in adapters now register their
  exact reuse schemas through the existing contract layer. All seven existing
  architecture checks and 71 reuse tests pass without relaxing that policy.
  Complete integration validation remains pending the final source freeze.
- The reuse certificate now also covers built-in categorical and random-effect
  groups and exact CSR matrices. It hashes codes, dimensions and sparse storage
  directly, including relevant cached CSR flags, without dense expansion or new
  row caches. Unknown matrix formats and custom subclasses still refresh.
  The 94 reuse regressions and 22 Gamma execution tests pass at this checkpoint;
  the existing architecture policy remains intact.

### Remaining ordinary-group execution stage

A further code audit found that ordinary grouped curvature still visits each
group pair separately. Column fallbacks and bin aggregation can therefore scan
the same observations many times in a fragmented design. Tensor factorization
does not address these routes. The independent `gaussian-fragmented` benchmark
adds small numeric, categorical, spline and supported grouped-curve terms to
exercise this behavior; its generated data and specifications are public.

1. Extend certified endpoint reuse to the built-in factor-smooth representation,
   covering its actual basis, transformations, row assignments and nested CSR
   state. Keep unsupported spline-by-category cache configurations conservative.
   The 215 focused checks pass, including 41 additional reuse regressions.
2. Implement and test a bounded row-panel workspace for small ordinary groups,
   using typed stored-design row rendering and signed rectangular matrix
   multiplication. The existing row renderer calls compressed `toarray()`;
   reusing it would violate the bounded cross-product contract. Count all
   simultaneously live panels, weighted scratch and
   rendering temporaries. Retain panels only for one likelihood chunk, preserve
   specialized tensor/sparse routes, and require fallback when the byte budget
   or numerical-domain conditions do not permit batching.
3. Establish a dispatch rule from the independent fragmented fixture and
   existing tensor/support controls. Validate reconstruction, signed moments,
   backward error and workspace lifetime before enabling the selected route.
   Avoid a general aggregate-result cache without demonstrated repeated keys.
4. Repeat complete-fit timing, peak memory, numerical comparisons and actual
   dispatch from the resulting source. Keep the discrete execution gate open
   until the remaining performance evidence is satisfactory.

### Independent call-stack review

The public fragmented profile separates two causes: 32 chunked geometry
evaluations versus 23 dense evaluations, and repeated ordinary group-pair work
within each evaluation. Extend the spline-by-categorical endpoint certificate
using its live basis, row-alignment and lazy-cache state so the extra evaluations
do not confound the batching comparison. This preserves the existing endpoint
acceptance rules; it adds representation coverage.

The generic factor-smooth/dense cross-product route also bypasses an existing
signed reduction kernel. Evaluate a narrow dispatch to that kernel with the
correct sum-to-zero adjoint, a bound on raw intermediate storage, and arithmetic
range guards. Singleton dense columns already need one factor scan and retain
their current path. Validate this route separately from the panel workspace.

Start panel integration with an internal opt-in byte budget. Select an automatic
route only after comparing complete fits on the independent fragmented fixture
and existing tensor/support controls. Interpret likelihood and geometry costs
together: derivative evaluation is charged to different phases by dense and
chunked execution. No additional roadmap capability starts during this work.

The optional panel implementation, signed factor-smooth dispatch and expanded
reuse certificates pass 395 focused checks together with existing chunk-reuse
and family-layering tests. Independent review resolved stored-buffer coherence,
subclass dispatch and mixed-CSR-index allocation issues. Automatic panel dispatch
is still disabled while a frozen-source public off/on pilot evaluates the
complete-fit tradeoff.

The first off/on pilot at `e23cd41c` does not support automatic admission:
one discrete fit takes 6.068 s with panels disabled and 6.423 s with panels
enabled; exact controls take 2.074 and 2.207 s. Inputs, outputs and iteration
counts agree within roundoff. Separate profiling confirms the intended route
(90,255 ordinary cross calls reduced to 210), but identifies 1.400 s of direct
Python range-check work. Replace those loops with the existing allocation-free
compiled predicate, preserving the same bounds and refusals. Warm writable and
readonly C, Fortran and strided layouts before repeating uninstrumented fits.
The 101 focused panel, integration and public warmup checks pass; keep this
unfavorable pilot and its profile receipts alongside the next comparison.

Three repetitions at `50b5e8bb` support a local panel benefit: discrete median
fit time falls from 6.847 to 5.475 s, with comparable process RSS and unchanged
iterations and numerical outputs. Exact controls take about 2.4 s. The remaining
gap therefore needs further execution evidence. A shared categorical subset
path now avoids redundant copying and sink-code reversal while retaining the
existing constructor's validation and owned storage; 128 focused checks pass.
Memory tests explicitly warm the compiled predicate before measuring workspace
allocations, so isolated tests exclude one-time compiler initialization.

Audit also found that the facade always couples `discrete=True` to chunked
execution. The existing lower-level dense backend can evaluate the identical
stored discrete basis. Compare this explicit execution override before changing
automatic policy. Include dense materialization, temporary allocations and
complete-process RSS; an additional workspace estimate is not a total-fit memory
cap. Preserve requested versus resolved execution metadata and specialized
support/tensor routes in any subsequently justified policy change.
