# Observation and Dirichlet harness

`superlss_observations_dirichlet.py` implements Task8's executable runner.
It does **not** constitute the final validation receipt. The parent owns that
receipt, representative runs, full tests, supported-version/native checks and
review. No production code, dependency, version or other benchmark is changed.

## Invocation and cells

Run `uv run python benchmarks/superlss_observations_dirichlet.py --list-cases`
for the deterministic case order and complete specifications. No arguments
prints help. `--case` is repeatable; omission selects the full matrix.
`--rows` defaults to 1000 and `--repeats` to 1. These defaults are small
witnesses, not representative performance evidence.

- Seven existing bounded controls, each with unit and frequency weights;
  Gaussian/Gamma also have nonunit prior controls. Log-normal includes both
  location and mean parametrisations.
- Two-piece normal, two-piece log-normal (both parametrisations), NB and
  Tweedie each have complete, exact-with-truncation, mixed-censor-only and
  combined cells; frequency variants, and NB/Tweedie prior variants. Each
  also has a combined smoothing cell.
- Dirichlet K=2/3/5 has intercept, fixed spline, smoothing and frequency cells.
- Gaussian fixed/smoothing and Dirichlet K3 fixed cells declare a two-column
  or two-array offset sum, with coefficient one on the link scale.

The same seed, generated inputs and mathematical specification feed both
sources. There are 16 repeated covariate values by default. Fixed fits use
cubic regression splines (`k=5+j%2`, fixed penalty 0.8); smoothing uses the
family's public default outer method and initial lambda 1. All Dirichlet
parameters vary; the first parameter varies in scalar cases. Intercept cells
have no spline. Bounds mix lower-only, upper-only and two-sided windows.
Sampling uses the conditioned law, then records exact and left/right-censored
rows. NB prior data are represented counts divided by prior weight; Tweedie
prior data use dispersion divided by prior weight. Frequency multiplies the
whole joint row law once. No weight-plan duplicate take is introduced.

`--discrete`, `--n-bins`, `--compact` (`retain_rows=False`), `--profile`,
`--timeout` (180 seconds per worker by default), `--shard-index` and
`--shard-count` control execution. Sharding partitions the selected case order,
zero-based. Run shards **sequentially** on a machine used for timing.
`--max-inner-iter` and `--max-reml-iter` both default to 100 and are recorded;
increasing these creates a different solver specification.

`--phases timing,memory,profile,reference` is the default. Narrower phase
selections are useful for harness development but explicitly partial.
`--equivalence` adds untimed dense/discrete fits on both repeated covariates
(exact equivalence) and a continuous grid (approximation sensitivity). The two
data grids are never compared as if their data were equal. `--multiple-starts`
adds untimed smoothing starts 0.1, 1 and 10, keeping every endpoint. There is
no public coefficient-start argument, so coefficient multiple-start work is
limited to the independent small-reference optimizer.

## Sources and measurement

`--baseline-source` must be an existing clean checkout's `src` at exactly
`1b63f7c59e3f351fcb194f0eb25ac4e1106f2ca7`. The runner never prepares a
checkout. Imports are verified against the selected path and source contents
before/after work. Baseline/candidate use the same interpreter, dependencies,
single-thread controls, deterministic inputs and harness.

Each timing repeat executes baseline-before, candidate, baseline-after in
sequence where the baseline supports the **same law**. Each job is a fresh
process with a private empty Numba disk-cache directory. It times a first
complete fit, then a new-model warm complete fit after collecting the first
model. Import/data generation and output collection are outside the timer;
model construction, fitting and its normal null/inference work are inside.
Warm means in-process caches may be populated; it is not a warm-started fit.
First/warm numerical equality is checked. Failed first/warm observations and
their warnings survive worker exit/timeout. No averages hide bad repeats.

Memory and profiling run in separate fresh processes and compare numerical
outputs to the timing endpoint. Memory includes process high-water RSS and
unique reachable NumPy backing buffers, with aliases deduplicated before
classifying cache/prepared/log-share/window paths. Python headers, native
allocations and global JIT caches are excluded from buffer counts but included
in RSS. Peak RSS is the whole-process high-water mark, **not** an incremental
fit allocation. Corrected-covariance requests run separately after memory is
recorded; fallback kind, warnings and refusals remain explicit.

Python profiling reports actual kernel/assembly/chunk entry points and work,
including exposed compiled entry calls. It does not count inner native-loop
iterations. `--profile` additionally retains full pstats. Production
`diagnose()` and phase snapshots provide iteration, retry, smoothing, null-fit
and refit evidence. Numerical arrays include public predictions/parameters,
log scores, link SE, coefficients, covariance, coefficient SE and smoothing
parameters, plus response covariance for Dirichlet. Rank, EDF, objective,
conditioning and stop reasons accompany them. Prediction/scoring profiling
checks that no training solver ran.

## Staged interfaces and independent references

Candidate offsets use `bind_predictor(..., offset=[...])`; the runner requires
that public `fit` and `fit_reml` no longer have `offsets`. Missing Task9 returns
`missing_stage`, never a hidden fitting-keyword fallback. Only the approved
baseline uses the explicit old-API adapter `fit(offsets={parameter: a+b})`.
Numeric declaration predictions supply new-row sums through prediction-time
`offsets`; column declarations resolve the current frame.

Dirichlet calls public `predict` and requires a DataFrame with the component
labels and input index. Joint `scores(..., which=("log",))` must return one
score per row. Pending Task6 records `public_status=missing_stage` or a public
error without erasing the completed fitting-core evidence. Internal means
are never substituted. This harness's prediction/scoring/covariance checks do
not replace Task6's required CV, posterior, serialization and refusal tests.

Every Dirichlet public score is checked independently against SciPy's joint
density at the published alpha, including frequency mass. Public score sums
must equal the fitted likelihood for exact predictor evaluation. With discrete
fitting, training uses binned predictor values and prediction evaluates the
original inputs continuously, so that equality is inapplicable. The raw
difference, applicability flag and evaluation basis are retained; the exact
guard and independent public-density check remain enforced.

### Source-bound performance evidence

Synthetic fixtures use seed 62118. These measurements used Python 3.13.14,
NumPy 2.5.2, SciPy 1.18.0 and Numba 0.68.0 on Linux x86-64 / EPYC 9645,
with one thread. Each timing worker used a fresh interpreter and empty private
code cache, followed by a fresh-model warm fit. Separate before/after controls
imported c23b3569d1c758343dc5f214eabce394a6a7d4d2. Numerical outputs, covariance,
SE, EDF, convergence, null fits, work and actual backend agreed. All reported
cases dispatched `distributional-dense-v1`.

| Case | Candidate source | First seconds: control / candidate / control | Warm seconds: control / candidate / control |
| --- | --- | --- | --- |
| Dirichlet K3 smoothing, 20k rows | `1c9764a6a61aba75e43437ee73f45dcfa774ebcf` | 104.92 / 29.06 / 108.26 | 101.75 / 5.69 / 102.82 |
| Dirichlet K5 fixed, 100k rows | `1c9764a6a61aba75e43437ee73f45dcfa774ebcf` | 141.45 / 22.49 / 123.01 | 130.38 / 8.51 / 135.72 |
| Tweedie combined censoring/truncation fixed, 20k rows | `fc20c2b3b41d53dfb9d823591d61e07bd8bd03ee` | 214.21 / 76.81 / 214.51 | 203.10 / 46.59 / 198.61 |

Warm control drift was +1.05%, +4.10% and -2.21%, respectively. Single-core
review probes overlapped the Tweedie after-control period on the shared host.
That Tweedie candidate preceded the quadrature disk-cache repair; its first
fit includes the then-recurring compile. These figures are not measurements
of later integration commits. Separate short profiles confirmed the compiled
Dirichlet row/directional route and remaining Python gamma/mixture overhead.

At 150abfd345ed4abace97f4575bd9dac7892428ed, the candidate-only 20k combined
Tweedie smoothing witness took 155.58 seconds first and 127.56 seconds warm.
Both main/null fits converged, smoothing stopped stationary, and first/warm
outputs, inference, work and backend agreed. There is no same-session control
ratio for this cell. Bounded Tweedie remains an expensive numerical route.

Separate 20k Dirichlet K3 discrete/compact frequency memory fits at c23b3569d
and 1c9764a6a retained the same 1,943,976 NumPy backing bytes, with no retained
cache buffers, on `distributional-chunked-v1`. Whole-process cold peak RSS
was 388.80 / 447.28 MiB, including imports/JIT/native state. The increased JIT
footprint is material. Saved public scores match independent SciPy densities
within 1.34e-14. The original receipts retain the erroneous training-bin versus
continuous-score identity failure; separate corrected-contract adjudication
validates their unchanged outputs without refitting.

These cases do not certify every tail, platform or smoothing endpoint.
K2 Dirichlet smoothing retains an existing nonconvergence in the short stress
fixture. Other numerical admission and optional-derivative limitations remain
in the family reference. Failed, timed-out and unresolved historical cells are
retained and are not counted as successful timings.

New bounded laws and Dirichlet have no matching old-source control. They get
absolute costs and a separate deterministic 64-row intercept fit, compared to
an independent SciPy BFGS objective with three named starts. The reference
uses textbook piecewise normals, NB probabilities, a 240-term compound
Poisson/gamma sum, or the normalized Dirichlet density. It never calls a
production probability or derivative. The reference envelope is moderate:
NB finite endpoints below 128/prior weight; Tweedie rate at most 30 (omitted
Poisson mass below 1e-120); central float64 differences for gradient/Hessian.
It checks objective, normalization, stationarity and covariance. Unresolved
or nonstationary endpoints and reference-range refusals stay visible. This is
a small-law reference, not an independent representative smoothing fit.

## Receipt schema and interpretation

The top-level JSON uses `superlss-observations-dirichlet/v1`: session, source
heads/content hashes, harness hash, settings, selected case definitions, runs,
comparisons, unmatched-law cases and open concerns. It checkpoints after every
job and comparison. Raw worker JSON, logs, array NPZ files, JIT caches and
optional pstats live in git-ignored `.benchmark-artifacts/`. Output files must
be new names; old evidence is never overwritten.

Each run names its artifact, actual source, mathematical/data/weight/environment/
machine/thread identities, adapter, execution settings and **all** observations.
Array descriptors carry path, SHA256, shapes/dtypes and individual-array hashes.
Failed/refused/timed-out workers remain rows, even without any fit result.

Comparisons retain absolute deltas, scales and tolerances. Session, baseline
head, inputs/weights, specification, harness, environment and thread mismatches
prevent a time ratio. Backend changes are reported independently; a requested
backend is never evidence of dispatch. Coefficient/smoothing nonconvergence,
missing public output or conditioning at/above 1e10 prevents an accepted time
ratio. Observation tolerances are solver-scale (2e-6, covariance/SE 2e-5),
with a bounded binary64 conditioning term; objective 5e-8, EDF 5e-6 and exact
rank. These are comparison thresholds for moderate fixtures, not numerical
certificates. Untimed backend-equivalence checks permit backend differences
and never produce time ratios. Continuous-grid and multiple-start differences
are sensitivity results, never filtered endpoints.

Equivalence jobs also record `backend_coverage`: expected chunked/dense and
observed identifiers. Missing either route requires attention, including in a
continuous-grid sensitivity job. Dirichlet public checks record the mean's
maximum absolute difference from independently normalized alpha and a tolerance
of 64u, where u = eps/2; a mismatch retains those deltas and fails public validation.
Small references retain `mathematical_match` separately from public validation.
An unresolved nested `production.public_status` requires attention even in a
reference-only run with a mathematically matched likelihood endpoint.

`runner_status=executed` means the selected runner work executed without an
unresolved check. Exit 1 and `attention_required` preserve failures for action.
`validation_complete` is always false: final acceptance belongs to the parent.
Control drift is recorded beside each baseline/candidate timing comparison.
Absolute new-family timings cannot establish same-law overhead against the old
source, which cannot fit their bounded observations.

Keep the earlier Task7A GG/GPD short timing increases (about 25%/20%),
two-piece bounded-mean cost, and Tweedie work-limit/timing concerns in the
parent assessment. Neither this harness nor a tiny smoke clears them. Retain
failed/refused/timed-out/nonconverged Tweedie 20k/100k stress cells; never remove
them or extrapolate throughput from smaller cells.
