# Shared-penalty complete-fit measurements

`multi_penalty_support.py` fits deterministic Gaussian and Gamma models with
nonconstant scale and shared first/second derivative penalties, a scalar
Poisson model with three derivative penalties, or a Gaussian location-scale
model with shared tensor margins. A separate scalar tensor fixture has 2,000
rows and 12-knot marginal splines, exercising the larger penalty matrices used
by interaction refits. Each invocation performs one public fit.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python -m benchmarks.multi_penalty_support \
  --case gaussian --mode reml --out /tmp/candidate.json --label candidate
```

Use `--case gamma|scalar|tensor|scalar_tensor` for the other fixtures and `--discrete` for
exact-bin discrete execution. `--mode fixed` isolates a complete coefficient
fit. REML mode also exercises penalty determinants and smoothing derivatives.
The fixed LSS fits need no penalty-kernel call; an empty kernel count there
is expected. Actual fit convergence and diagnostics remain in the receipt.

Select an archived package through `PYTHONPATH=/path/to/snapshot/src`. The
receipt identifies the imported source by its path and a digest of every
Python source file. Its git fields are null for a snapshot outside git.
The driver, native-observation helper and generated data have separate hashes.
A label never substitutes for source identity.

Arrange exclusive numerical execution and follow
[the timing policy](../docs/development/cost-and-timing.md) before adding
`--measure-time`. Compare five fresh-process baseline/candidate pairs in
alternating order using the same interpreter, driver and data. Inspect both
the absolute difference and baseline median absolute deviation. A slowdown
exceeding 10 percent and three baseline deviations warrants investigation;
this is a diagnostic rule, not a unit-test assertion.

Receipts include process peak RSS for exactly one fit, actual retained ranks
and execution backend, coefficients,
predictions/natural parameters, objective and convergence evidence, and kernel
entry counts. Counts distinguish each entry point; nested entry counts must
not be added and called independent penalty evaluations. Instrumentation
retains counts and rank frequencies, with no growing list of result arrays.
The custom wide triangular solves and compensated dots have separate counters;
native QR or BLAS use alone does not establish that the whole kernel is native.
The Dot2 receipt also records compiled success/fallback counts and the actual
Numba signatures. Any first-use compilation occurs inside the measured fit.
Timed runs bypass all kernel wrappers and native-pool observation. Collect
dispatch in a separate fresh process without `--measure-time`, and compare
its numerical outputs with every matching timed run. Untimed native-pool
snapshots run synchronously on selected solver call/return events. They
record event frequencies, not elapsed-time dwell or continuous coverage.
Receipts state the observation status, attempts, samples, capacity limits,
drops and errors; missing observations do not count as observed dispatch.

This separation also applies to `rank_deficient_complete_fit.py` and
`lss_convergence_repair.py`: their default run is untimed; `--measure-time`
performs a separate fit without observation hooks. No background library
observer runs in either mode. An earlier background `threadpool_info` call
deadlocked against Numba's first-use dynamic loader during the final comparison
attempt. The native stack trace and partial receipts are retained as a harness
failure; its stalled wall time is not a solver timing measurement. Numba's
first-use work remains inside the complete-fit clock.

The final rank replay uses a separately hashed untimed launcher to include
the conditional and private Gram entry points in the observer registry.
The full-rank case uses those routes and otherwise produces no selected
events. The launcher refuses timing flags and changes no numerical function
or frozen benchmark driver. Its four receipts match all 20 corresponding
timed fits. The combined dispatch record validates 28 untimed executions
against all 140 complete fits; it is preserved as `dispatch-combined-v3.json`
in the numerical audit's artifact directory.

The LSS receipt also includes `SuperLSS.diagnose().to_dict()`, collected after
the fit clock and peak-RSS reading. Its diagnostic phase timings remain outside
the numerical outputs used for exact timed/instrumented comparisons.

`solver_repair_complete_fit.py` adds public REML fits for the other repaired
solver paths. Its cases are `qp` (inactive constraints), `qp_binding` (a negative
quadratic response against a convex constraint), `scop_single` (one SCOP group
using the default joint routine), `scop_joint_discrete` (two groups using the
discrete cross-product route), and `sum_to_zero` (a structured factor smooth
with a global spline). The untimed receipts require actual solver returns;
the binding QP must perform active-set iterations and retain active constraints.
The discrete SCOP receipt observes returned histogram products, and sum-to-zero
records its factor/solve calls and terminal structured backend. The one-group
case does not claim coverage of the separate private single-group kernel.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python -m benchmarks.solver_repair_complete_fit \
  --case qp_binding --out /tmp/qp-binding.json --label candidate
```

These receipts hash the actual inputs, package, driver and four imported
benchmark helpers. Timed mode omits solver observation. Compare every
instrumented receipt against all five matching timed receipts, including
helper identities and the complete numerical output/history payload. The
40-iteration REML limit is part of the fixture; the baseline discrete joint
case reaches that limit, so a completed invocation alone is not convergence.

The resumed baseline is archived at
`/tmp/superglm-numerical-resume-9106rchs/resume-baseline.tar.gz`, with its full
source/data manifest beside it. Gaussian, Gamma, scalar and tensor driver
smoke checks passed against that source, including REML and discrete cases.
Their wall time is unmeasured because implementation and numerical reviews
were active concurrently. Final candidate comparisons follow integration.
