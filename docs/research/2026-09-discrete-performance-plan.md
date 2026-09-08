# Discrete execution performance plan

Status: production streamed moments are implemented, reviewed and validated.
Fifteen timed fits and five separate witnesses establish faster discrete
execution on the measured mixed layouts through one million rows. C1 performance
work remains active: the user considers the 8% one-thread time advantage over
exact at one million rows insufficient. A completed eight-fit BLAS/shape screen
shows the remaining comparison against threaded dense execution. Next work
attributes the production fit cost and checks scalar thread-policy behavior.
Baseline: `0a15736e88a317088bfd01e56933d45c58e4ac9a`
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
and background activity. Tool completion timings are not fit timings.
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
  within the final window, median discrete time is 1.521 s versus 2.342 s at
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
remained disabled for that checkpoint's frozen-source public off/on pilot of
the complete-fit tradeoff.

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

### Size, threading and execution follow-through

The fixed public fragmented layout now has exploratory complete fits at
65,536, 262,144 and 1,048,576 rows. Model terms, knots and bins stay fixed;
iteration counts agree across execution arms within each size, but change
between sizes. At one million rows, ordinary discrete chunks take 70.093 s,
optional panels 62.747 s, and the same stored discrete basis through dense
execution 26.748 s. Process high-water RSS captured at fit completion is
1,831.6, 1,832.5 and 3,177.7 MiB respectively; exact execution takes 30.014 s
and 3,883.3 MiB. These are single runs, not a statistically established
crossover. The larger book preserves the time/memory tradeoff. Fixed 8,065-row
chunks repeat preparation and dispatch as the book grows.

A separate two-repeat comparison at 262,144 rows uses the existing
`SUPERGLM_BLAS_THREADS` override, with NumPy/SciPy pools witnessed inside the
solver and other numerical thread pools held at one. BLAS one-to-four median
wall times are 18.241 to 18.121 s for ordinary chunks, 16.394 to 15.933 s for
panels, and 7.631 to 5.721 s for dense execution. Chunk/panel ranges overlap;
dense ranges do not. Increased CPU use is recorded separately. The override
also affects compilation: bin maps and raw support bases match exactly, while
ten transformation-array hashes change between thread settings. Fitted outputs
agree within floating-point error, but the receipts do not quantify the
transformation-array differences. Do not claim identical stored representation
hashes across those settings.

Call-stack profiles at `ed84669a` identify repeated row preparation and panel
rendering as substantial remaining costs. The panel matrix products are already
efficient; the range predicates are compiled, so their remaining scan cost is
not the earlier Python-loop defect. Profile cumulative times nest and are not
additive complete-fit timings. The detailed report separates their ownership.

The comparison with regular SuperGLM also corrects an algorithm assumption.
Scalar discrete REML selects a cached-working-weight optimizer and avoids
rebuilding weighted geometry for each lambda trial. SuperLSS retains its
distributional observed-curvature/EFS algorithm. Reusing scalar execution
techniques and adopting its optimizer are distinct changes; signed coupled
curvature needs its own valid contractions.

Those findings established these subsequent implementation stages:

1. Improve the existing bounded panel renderer with explicitly budgeted,
   chunk-local small support transformations and checked writes. Preserve
   numerical guards, unsupported-type fallback and workspace lifetime; do not
   cache numerical results across mutable designs by identity.
2. Compare installed tabmat primitives with a dedicated distributional execution
   plan that can reuse bounded row preparation for predictor, score and
   curvature work. Resolve contiguous-range lookup validity before introducing
   range shortcuts; a persistent sorted flag on mutable caches is insufficient.
3. Validate any selected execution change with mathematical, memory and actual
   dispatch regressions, then repeat serial complete-fit measurements. Keep
   automatic dense and panel policy changes on hold until these findings are
   incorporated. No further roadmap capability starts during this gate.

The first two execution primitives now pass 602 combined focused checks, with
four inapplicable exact-category permutation cases skipped. Independent review
resolved obsolete lookup-buffer retention on same-object storage replacement;
range authority is cleared on mismatch. Geometry snapshots retain bounded
owned arrays. The support-table reserve and checked-writer coverage are reviewed,
including public warmup and isolated allocation tests. Automatic policy was
unchanged at that checkpoint. Its comparison against frozen `ed84669a` and the
separate raw-basis tabmat evaluation are recorded below.

The three-repeat complete-fit comparison against `ed84669a` now records
ordinary chunk medians of 22.648 to 17.906 s and panel medians of 17.172 to
12.207 s, with nonoverlapping old/new ranges in this window. Saved same-route
arrays and stored representation hashes agree on the fixture. The raw-basis
tabmat prototype loses its constructor-inclusive geometry comparison (1.467 s
versus panels at 0.535 s), so it will not receive a full-fit adaptation.

Promote the bounded panel kernel automatically only within an explicit initial
ordinary mixed-layout envelope, preserving specialized and unsupported routes,
explicit off/budget overrides and per-chunk refusal. Keep existing row bounds
and requested/resolved backend metadata. A group-width envelope is a tested
scope limit, not a fitted speed crossover. Automatic dense execution remains
deferred; its memory tradeoff is documented separately. Validate the actual
default route and specialized controls, then complete final production suites.

Automatic panel integration is implemented and independently reviewed. It reads
only exact built-in types and dimensions, requiring the stated mixed layout in
every predictor with slopes, allowing intercept-only companions and widths up
to 32. It resolves the additional 64 MiB allowance once per geometry assembly.
The builder remains the numerical and actual-storage authority. The 172 focused
checks pass, including 44 new regressions for default dispatch, explicit
overrides, signed channels, refusal and cleanup. The original default-dispatch
case fails before the change. Source was then frozen for full-suite and
actual-default complete-fit validation; automatic dense execution remains deferred.

Production source `9f0e196c` completes full non-browser validation with 12,350
passed and 109 skipped unique cases, including 84 required real-data checks with
no skips. The only initial failure was duration-manifest coverage; 1,078 missing
entries were filled from this run's JUnit durations without changing previous
values, and all eight CI contract checks pass on rerun. Lint, formatting,
dependency and smoke checks pass. This completed the production suite before
the final default-route complete fits below.

Final default-route validation at `74ce13f3` completes 28 uninstrumented fits and
three separate dispatch witnesses against frozen post-C3 `5f994c8f`. Discrete
median wall times fall from 31.530 to 13.074 s for fragmented Gaussian, 3.684 to
1.864 s for support-32, and 12.601 to 2.950 s for public Gamma severity. All
before/after discrete ranges are disjoint in this window; the two-repeat
controls remain local evidence. Stored representations agree between sources
within each mode, and saved outputs agree exactly or at roundoff scale.
Representation effects are recorded separately. All fits report successful
practical convergence, without claiming strict smoothing certification.

Automatic panels are observed only in the intended mixed-layout fixture;
support/tensor and severity controls retain their existing routes. Source,
helper, thread and activity checks pass. The [report](2026-09-discrete-performance-report.md)
and [tracked receipt](../../benchmarks/discrete_performance_receipt.json) retain
complete-fit CPU, process RSS, ranges, numerical comparisons and actual dispatch.

The remaining public mixed-layout gap is explicit: current discrete fitting is
44.5% slower than exact fitting, with 499.5 MiB less fit high-water RSS. Automatic
dense selection remains deferred and the raw-basis tabmat prototype remains
unselected. The current implementation stages are validated; this residual C1
execution issue remains ahead of other roadmap capabilities. Cross-predictor
penalties stay deferred under the chosen scope.

## Next stage: compute on supports, then reduce to coefficients

Grouping covariates at the selected resolution is the computational objective.
Faster panel rendering alone does not deliver the full
benefit when curvature still expands every stored support row. Preserve the
current solver and observation-level response, weight and offset semantics;
accumulate their changing score and signed curvature contributions by support
indices where the term representation permits it.

1. Profile current production source, separately comparing exact execution,
   default discrete execution and the identical stored discrete basis through
   the existing dense backend. Separate disjoint call owners, iteration counts,
   row rendering, weighted products and memory traffic. Use the original
   controlled activity protocol, including all observed external processes.
2. Falsify the repeated-contraction hypothesis with three diagnostic complete
   fits: panels disabled and only the geometry batch changed from 8,065 to
   64,520 to all 262,144 rows. Preserve other pass sizes, family, optimizer and
   stored basis. Count actual histogram, directional and row dispatch. These
   instrumented fits identify mechanisms; their times are not replacement
   benchmark estimates. Large geometry batches are a diagnostic, not an
   automatic memory policy.
3. Select the smallest supported execution change from that evidence. Eligible
   pairs should accumulate weights or weighted columns before the final support
   contraction, with explicit memory limits and mixed-term fallback. Scalar
   full-design aggregation is reusable architectural evidence; its positive
   working-weight centering and alternative smoothing optimizer are separate.
4. Validate signed rectangular blocks, score reductions, cancellation/refusal,
   live-input authority and workspace accounting independently. Compare the
   stored design before measuring any resolution effects. Require a baseline
   failure or mutation check for the relevant new regressions.
5. After a selected implementation passes focused checks, evaluate complete fits
   with wall/CPU, fit-end RSS, outputs, iterations and actual dispatch. Retain
   the public insurance and favorable-support controls. Do not repeat the full
   historical matrix without a new uncertainty that requires it.

Steps 1 and 2 are complete at `748c8596`, whose production source matches the
validated checkpoint above. The default panel geometry still has row-space
quadratic coefficient work. Its disjoint profile attributes 2.969 s to panel
building and 2.398 s to likelihood-chunk iteration, against 3.486 s for curvature
channel calls. Only 0.046 s is spent transforming the small support tables;
hoisting those transformations alone cannot recover the gap.

Panel-off geometry batches of 8,065 / 64,520 / 262,144 rows produce diagnostic
fit times of 22.669 / 14.776 / 14.174 s. Histogram builds fall 33:5:1 while
weighted histogram rows and directional row-by-width work remain identical.
Larger batches therefore recover setup/contraction amortization, but the
whole-book grouped route still exceeds default panel geometry time. Mixed row
work remains substantial. No larger chunk default is selected from this probe.

Step 3 now has a specific bounded design to test: retain signed support-pair
and directional moments across derivative chunks, contract supports once per
geometry, and process the small numeric/categorical/intercept block together.
Do not allocate and add a fresh full histogram for every chunk, which would
preserve the repeated initialization just measured. Do not retain expanded
N-by-p spline panels. Batch the ordinary columns to remove the existing
singleton pair scans, preserving interaction masks and rectangular channels.
Use explicit state/scratch budgets, fit-local ownership and the existing
numerical/unsupported-layout fallback; full-design mutation authority cannot be
inferred from matching hashes on this fixture. The measurements below now test
this design directly.

The prototype now has an explicit evaluation contract. Construction validates
the layout and owns the small support bases/maps; each geometry owns fresh
accumulator state. Reset initializes scores, curvature and penalties. Existing
likelihood chunks provide observation-specific signed channels and owned local
row maps. Accumulation verifies their support authority, retains moments across
chunks, and finalization transforms each support block once. Unsupported
layouts, invalid numeric state and budget excess refuse explicitly. Ordinary
row panels remain bounded by the existing chunk size. No predictor equality or
full-N copied map is required.

An independent oracle compares distinct rectangular predictors against directly
materialized stored designs, including masks, categorical baseline codes,
signed channels, reset behavior and live-input mismatch. Its sign/mask mutation
checks must demonstrate sensitivity to wrong accumulation. Memory/dispatch
checks are separate from mathematical comparisons.

The initial performance gate uses three fresh public workers: current panels,
existing global grouped moments and the prototype. Every whole-geometry timing
includes construction, validation, reset, current family/predictor evaluation,
accumulation, support transformations and final outputs. One whole-pass warmup
precedes three within-worker repetitions. The global reference collects the
same chunk-produced channels before grouped contraction, charging collection
and allocation; this is distinct from the earlier C=N likelihood-batch probe.
Only a favorable and numerically correct result advances to a small repeated
complete-fit comparison under the existing serial activity protocol.

The frozen prototype `1028b157` passes the independent oracle, including 12
distinct-layout/chunk/cancellation cases with two reset geometries each, zero
channels, executable sign/mask mutations, live-input refusal and separate
workspace/dispatch checks. Its corrected loader comparison records median
whole-geometry wall times of 0.273 s for global moments, 0.448 s for current
panels and 0.510 s for existing global grouped assembly. The three within-worker
ranges are disjoint between the prototype and either reference; all geometry
and intercept comparisons pass on identical stored bases. This is a favorable
geometry diagnostic, not a complete-fit speed estimate.

Proceed with six fresh complete-fit workers: current exact, current default
discrete and prototype discrete, twice each with reversed second-repeat order.
Preserve the existing public model, optimizer, bins and tolerances. Construct
fresh prototype state for every geometry; any refusal fails that experimental
arm rather than silently timing fallback. All arms receive the same public
warmup and a declared tiny native-signature warmup. Record complete-fit clocks,
fit-end RSS before saved outputs, work/status, numerical outputs, representation
hashes and native execution. Keep the prototype outside production until this
evidence and the code review support promotion.

Environment correction: the user reports that Headroom and Kompress have been
uninstalled. Current comparisons do not assume their presence or depend on
their tools. Serial workers, fixed numerical thread counts, process activity
screens and raw worker wall/CPU/RSS evidence remain unchanged. Historical raw
receipts are preserved; inclusion in a process-audit policy does not establish
that a particular service was running.

The six-worker comparison is complete. Median wall times are 7.639 s exact,
10.821 s current discrete and 7.781 s prototype discrete; the prototype and
exact ranges overlap. Median fit-end process peaks are 1,280.88 / 778.36 /
784.09 MiB respectively. Both prototype fits execute 19 fresh geometry plans,
627 chunks and 855 support-pair finalizations without refusal. Inputs and stored
discrete designs match; maximum holdout difference versus current discrete is
8.88e-16. All fits retain 18 inner and seven smoothing iterations and the same
practical-plateau status. The exact/discrete holdout difference of 7.36e-4
remains a resolution effect, separate from the accumulator comparison.

### Production integration stages

1. Add independent stored-row mathematical regressions and observe the missing
   production capability. Cover signed rectangular channels, masks, cancellation,
   reset/cleanup, budget boundaries and executable sign/mask mutations. Keep
   dispatch/workspace assertions separate from numerical comparisons.
2. Implement the bounded accumulator in `solver/_global_moments.py`. Own the
   small solver-support tables `T = B @ R` once per geometry and contract the
   global moments with those tables. Keep original B/R copies for live authority
   checks. Require exact ndarray authority before accessing caller arrays and
   certify nonzero T, ordinary values and derivative channels within the existing
   broad exponent envelope. Zeros, signs and cancellation remain valid. This
   closes the prototype's raw-contraction underflow and ndarray-subclass gaps
   without changing the solver, likelihood or stored representation.
3. Integrate a narrow automatic selection into chunked assembly. Preserve
   explicit panel overrides and the existing row budget. Refusal discards partial
   state and replays the entire geometry through the existing fallback; arbitrary
   exceptions still propagate after cleanup. Establish an evidence-based size
   envelope before extending automatic dispatch to small models.
4. Independently review implementation and run focused mathematical, integration,
   endpoint-reuse and warmup regressions. Run the required broader checks once the
   production source is stable. Existing extreme-scale fallback behavior is not
   upgraded by refusing the new route.
5. Compare actual production defaults against the frozen starting implementation,
   including complete fits, fit-end RSS, saved outputs and actual dispatch. Use
   bounded size controls to assess selection and retain the public insurance and
   favorable-support controls. Update this plan, report and roadmap with the
   validated capability and any remaining crossover limitation.

Ownership and interface review: the mathematical tests consume the accumulator
API; the implementation owns that API and its explicit refusal; integration owns
selection and whole-stream replay; the benchmark consumes actual default dispatch.
No two implementers own the same production file. The original prototype and its
receipts remain frozen as evidence. The production plan adopts solver-support
tables because their one-time construction is small and avoids delayed map
rescaling; this choice requires new numerical and performance validation.

Stage 1 observed the missing assembler and eager fallback allocation before
implementation. The production core at SHA256 `838e9519f42674e065965ca8cc590470fe3c95d26b272dc9a991f054481d7c66`
now passes 45 independent behavioral tests, including both executable mutations,
signed cancellation, solver-support scaling, exact array authority, workspace
and state recovery. Static checks pass. Independent core review and integration
validation are in progress; this is not the final production source freeze.

The initial automatic admission is deliberately restricted to the measured
mixed layout, exact Gaussian/Gamma family and likelihood-plan contracts, at
least 262,144 observations, and histogram initialization no larger than one
quarter of the corresponding histogram row updates. This is an initial scope
rule rather than an established speed crossover. Explicit panel controls retain
their existing semantics. A private registry flag admits deterministic replay;
custom family or likelihood-plan types stay on their existing routes.

The planned production comparison has 15 timed workers: two repeats of the
262,144-row exact/discrete baseline and new discrete default, discrete baseline
and candidate pairs at 65,536 and 1,048,576 rows, and pairs for the existing
support-32 and public Gamma-severity controls. One exact control at 1,048,576
rows also directly checks the large-N time/memory frontier. Five candidate witnesses
check actual default selection/native dispatch. All timed arms receive the same
tiny global-kernel warmup, with module/source provenance recorded. The untouched
starting implementation is retained in `.worktrees/discrete-global-baseline` at
`299ab249`; benchmark helpers must not replace its assembly routines.

Independent core review found two admission issues: an early numerical refusal
could conceal a later same-chunk source error, and oversized live activity
indices could allocate an unbounded validation temporary. Three regressions
reproduce these failures against `838e9519`. The revised core at SHA256
`e7628669a4ed0a77c33115820dcf0c8f7144518701a0edb8552673a55d6785e2`
defers numerical refusal until the same structural pass completes and checks
activity length before comparison. Scoped re-review clears both findings with
no additional important issue. Integration review also requires exact certified
link types before replay admission; custom links retain baseline dispatch.

The link guard is implemented, with eight regressions demonstrated against the
unfixed policy. Final composition review also identified overridable slicing
on resolved-weight subclasses. Four Gaussian/Gamma regressions reproduce that
admission gap; exact resolved-weight authority now joins the exact family,
plan and link checks. Both policy changes pass scoped re-review. The final
48 assembler tests and 193 integration/reuse/architecture/warmup checks pass.

The first broad non-browser run on source tree `1ad3538e4190dc7ed8f39d35bbe9017e1e8331f7f67cb5e31919ed711f8b4f03`
records 12,134 passes, no assertion failures, 405 skips and 15 setup errors.
The setup errors are the required severity-data guard: the default cache has
frequency data but lacks severity. Of the skips, 296 require the absent optional
`mpmath` oracle. Restore that environment-only dependency and rerun its seven
affected modules plus real-data parity with the explicit existing public data
directory. The only production delta after the broad run is the reviewed
resolved-weight admission guard, covered by the focused checks above. Preserve
the initial receipts and report the combined latest outcomes without presenting
the broad run as a single successful check on the final source.

The corrected affected-suite run passes all 556 cases without skips; the final
50-case integration run and smoke check pass. Combined latest outcomes are
12,449 passes and 109 remaining expected/optional skips, including all 84
required real-data cases passing. Source-tree comparison confirms the policy
guard is the only production change after the broad run. Final source tree
`0da3cf592be4e26727059d42cdd2c2161f37b08649b5cbc522bf9bad6e4c06ea`
passes static/dependency checks, and all independent review findings are closed.
Proceed to actual-default timing and witnesses with this implementation frozen.

The production campaign at `5c1ce17e` is complete: the 262k discrete median
falls 12.036 to 7.884 s, and the 1m/P102 sample falls 39.439 to 29.321 s versus
31.956 s exact. The latter uses 1,869.37 MiB fit-end highwater versus 3,893.99
MiB exact. All same-discrete inputs/stored representations match; held-out
differences are at most 8.88e-16. Five separate witnesses confirm 19/17 global
plan lifecycles at 262k/1m without refusal, and existing-route bypass on all
three smaller controls. Raw receipts preserve the first 262k exact cold-warmup
highwater and separately label output memory. Full evidence and qualifications
are in the report and tracked performance receipt.

### Further latency work selected on 2026-09-09

The user explicitly requests more fitting-time reduction and prioritizes
single-fit wall time; higher CPU usage is acceptable when it saves time.
Therefore the modest million-row time advantage does not close the C1 gate.
Do not move to another roadmap capability or present this as task completion.

1. Completed: eight fresh forced-BLAS1/4 fits on 262k/P102 and 1m/P182. Exact
   wall times improve 8.179 to 6.296 s and 75.795 to 46.844 s respectively;
   discrete changes 8.085 to 8.078 s and 46.956 to 44.732 s. Inputs/raw supports
   agree across thread settings; transform roundoff and binning effects remain
   separately reported. These two coupled N/P shapes provide a targeted screen,
   not a full crossover study or a direct automatic-policy observation.
2. Audit complete: shared scalar/LSS threading is a fixed width-only cap at
   1,500 coefficients, with no row/backend or measured timing input. `-1`
   disables intervention and leaves native settings. Prepare a four-worker
   public scalar control of exact forced1/4/auto and discrete auto under known
   native BLAS4, capturing actual scopes and restored settings. This control
   does not establish discrete scalar 1/4 sensitivity.
3. Profile one unchanged production 1m/P102 discrete fit. Existing phase data
   places about 16.0 of 29.3 s in inclusive geometry and only 0.04 s in
   coefficient solves. Obtain disjoint attribution for predictor/preparation,
   family derivatives, validation/packing, native moments and final products
   before selecting another optimization. Preserve numerical/ownership/replay
   contracts; do not equate data-work counts with measured time.
4. Capture live `SuperLSS.diagnose()` after fit clocks, profiling and fit-end RSS.
   It reports retained phase/iteration/backtracking evidence but not kernel
   dispatch. Scalar controls use `training_telemetry()` and `reml_diagnostics()`.
   Keep diagnosis/output cost separate from fit timing.
5. Select the next bounded change from the new evidence, with focused
   mathematical regressions and before/after complete fits. Unknown iteration
   count does not prevent per-operation thread decisions, but calibration cost,
   actual backend and nested/concurrent pool behavior require evidence. Existing
   BLAS control does not parallelize the native moment loops.

Keep source `5c1ce17e` and the completed benchmark windows as the new comparison
checkpoint. Documentation-only commits may update HEAD while preserving the
production-tree hash; subsequent experiments must pin both explicitly.
