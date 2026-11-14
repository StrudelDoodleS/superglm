# Benchmark baselines

`benchmarks/books.py` runs complete fits on named books and compares a branch
with a baseline that is already stored, so a pull request times only its own
branch. The baselines live in benchboard, the machine coordinator: its
`run-timed --record` store holds the wall time per case, commit and hardware,
and its fit history (`history.sqlite3`) holds the full record of each fit. A
commit that has been timed for a dataset, case spec, environment, machine and
thread setting is not timed again for its work counts, numerical outputs and
backend.

Wall time drifts between sessions on a shared machine: the same commit and
case have measured 15-18% apart on different evenings, with identical work
counts. A wall-time claim therefore needs the baseline timed in the same
session as the branch, or a stored case re-run in that session as a control,
with the drift reported beside the result.

## Cases

`python benchmarks/books.py cases` lists the case sets:

- `standard`: the corrected freMTPL2 book (`examples/fremtpl2_data.py`, seed
  20261012, 80/20 split), Poisson, exact mains, 4 pairs and 10 pairs, plus 4
  pairs binned.
- `families`: 13 family/link cases on the same book, mains plus 4 pairs, exact
  and binned.
- `lss`: the negative binomial, Tweedie and gamma severity `SuperLSS` models.
- `smoke`: a tiny synthetic book, for developing the harness and as `check`'s default set.

## Run one fit

```
benchboard run-timed me-std_pairs4 5 -- uv run python benchmarks/books.py run std_pairs4 --threads default
```

`--threads pinned` sets every thread pool, `SUPERGLM_N_JOBS` included, to 1;
`--threads default` removes them all. A timed run refuses to start when the
pools it finds do not match the setting. Each run writes:

- one JSON record under `~/.cache/benchboard/books/<sha>/` (or
  `$SUPERGLM_BOOKS_DIR`), with a line in the append-only `index.jsonl` there;
- the same fit into benchboard's history: a row in `fits` (the work counts and
  held-out metrics in `summary_json`), its phases in `fit_phases`, and the full
  record, compressed, in `fit_diagnostics`. Inside a benchboard job that
  captures fits the record goes through the job's spool; otherwise the run
  records its own task with benchboard's writer.

The record holds:

- provenance: the git SHA and dirty flag of the tree superglm was imported
  from, the dataset content hash, the case-spec hash, the environment (Python,
  numpy, scipy, numba, pandas, tabmat, narwhals, scikit-learn, joblib,
  threadpoolctl and every BLAS threadpoolctl reports), the machine
  (CPU model, physical cores, RAM) and the thread setting;
- measurements: wall and CPU time of the fit, peak RSS, the REML profile's phase
  seconds, work counts (Gram builds, table and leverage passes, solved-row and
  stop passes, TSQR and data-rank factorisations, PIRLS and outer iterations,
  line-search refits), the backend, convergence flags and captured warnings;
- results through superglm's own `predict`: REML objective, EDF, test deviance,
  NLL, Gini and the worst decile's |A/E − 1|, or for `SuperLSS` the NLL,
  deviance, mean |A/E − 1| and 80% coverage.

A warm-up fit on the first training rows compiles the kernels before the clock
starts. Time only on a quiet machine; [Cost claims and the clock](cost-and-timing.md)
explains why.

`--counts-only` is for development: the fit runs, the record keeps the work
counts and results, and it drops every timing field. Counts settle most cost
questions without a quiet machine.

## Store a baseline

```
uv run python benchmarks/books.py baseline origin/master --set standard --threads pinned,default --label me
```

With no ref, the baseline is the merge base of `HEAD` with `origin/master`. The
command checks the commit out as a detached worktree under
`~/.cache/superglm-bench/trees/`, syncs its own environment there, and queues
one `benchboard run-timed` per fit against that tree's superglm. A pinned fit
runs with `--record`, so benchboard stores its wall time as a baseline; the
benchboard case name is `books/<case>/<threads>/<hash>`, where the hash covers
every part of the key except the commit, which benchboard keys itself.

A key whose successful timed record is already in benchboard's history is
reported as `reused` and not run. Work counts, numerical outputs and backend
may be compared against it. A wall-time claim needs the baseline timed in the
same session (AGENTS.md, *Performance evidence*): `--force` re-times the key, and
`compare` pairs with the most recent usable stored record, so it picks up that
run. `--force` also replaces a stored record known to be bad. `--prepare`
creates and syncs the worktree without fitting.

## Compare

```
uv run python benchmarks/books.py compare
```

With no arguments, `compare` takes the latest record per case and thread setting
for this tree's `HEAD` and looks up the stored baseline at the merge base in
benchboard's history. It prints the markdown tables pull requests use: one
summary row per case (wall, CPU, peak RSS, outer iterations, objective, EDF and
held-out metrics, base → branch with ratios), then the phases and work counts of
each case. For a pass/fail gate on a pinned run, `benchboard run-timed --vs` with
the same benchboard case name kills a run that passes the factor.

## Check fit outputs against master

`check` asks whether a branch moves what a user reads. It does not time anything, so it
runs on one core beside other work:

```
uv run python benchmarks/books.py baseline --set smoke --counts-only --label me
uv run python benchmarks/books.py check --set smoke --label me
```

`baseline --counts-only` stores untimed, pinned fits of the merge base, one `benchboard
heavy --cores 1` job per case, and reuses a key already stored (a timed baseline serves as
well). `check --set` fits the same cases at `HEAD` the same way and compares each with its
baseline under `compare`'s key rules: same dataset, spec, harness, environment, machine and
thread setting, so a pinned record is only ever checked against a pinned baseline. Given
record files, or none (the latest per case for `HEAD`), it fits nothing. It prints one
table row per case and a `FAIL` line naming the case, the output and the size of every
move beyond tolerance. It exits 1 when anything moves, 2 when a record has no usable
baseline, and 0 otherwise.

Pinned on one core, the smoke fit takes about 5 s, `std_mains` 8.5 s and `lss_gamma` 6 s;
the larger `standard`, `families` and `lss` fits take longer, so add those sets for changes
that reach them. Each `baseline` or `check` run also reads benchboard's history once per
baseline commit, which has no index for that read and takes 2 to 3 minutes on a 54 GB
history, and a first `baseline` at a commit creates and syncs its worktree (about a minute).

Every record carries an `outputs` block:

- the sample: 2,000 test rows (all of them on a smaller test frame) drawn with the harness
  seed, so a dataset hash names the same rows at every commit, and a hash of the rows;
- `SuperGLM`: the predicted means on those rows, the intercept, the summary's coefficient
  standard errors (per term, and the intercept's; `predict` gives no standard errors),
  total and per-term EDF, the dispersion and any estimated theta or Tweedie power, and the
  smoothing parameters;
- `SuperLSS`: every parameter's prediction on its natural scale and its linear
  predictor's standard error (`predict_link(se_fit=True)`) on those rows, each
  parameter's intercept, total and per-term EDF, and the smoothing parameters;
- decisions: the convergence flag and termination reason, and the dispatched backend.
  The warning categories come from the record's captured warnings.

The block is about 44 KB of JSON for a `SuperGLM` book case (2,000 predictions and 67
coefficient standard errors on `std_mains`) and 166 KB for a two-parameter `SuperLSS` case (a prediction and
a standard error per parameter and row). On disk, one float per line, the `std_mains`
record is 58 KB and the `lss_gamma` record 206 KB; compressed in benchboard's history they
are 23 KB and 76 KB.

### Tolerances

Decisions must match exactly: status, convergence flag, termination reason, backend,
the set of warning categories, the terms that carry EDF and smoothing parameters, and
which standard errors are finite. A warning's count may change.

Values are gated at `VALUE_RTOL = 2**-26 ≈ 1.49e-8` relative, entry by entry:
`|b - a| / max(|a|, |b|, floor)` against the baseline value `a`. The argument:

- Two fits under one key run the same arithmetic on the same machine with the same
  thread pools. A branch that leaves the numerical code alone reproduces its baseline bit
  for bit, so master against itself moves nothing.
- A branch that reorders that arithmetic (another summation order, a faster kernel)
  moves a converged output by rounding amplified by the problem's conditioning. Last-bit
  thread differences reach about 1e-9 after REML, and are accepted as rounding.
- A branch that changes where a fit stops moves its outputs by up to the fit's
  convergence slack, and that is not rounding. The slack is set by PIRLS stopping on a
  relative deviance change of `tol = 1e-6`, and by REML stopping on a projected
  log-lambda gradient and an objective change of `reml_tol * (1 + |V|)` with
  `reml_tol = 1e-9`, or on a Newton step of at most `1e-3` in log lambda. On these books,
  with `|V|` in the thousands or more, that moves outputs far beyond 1e-8. Such a move must
  be judged against a reference optimum, and passing it silently would hide it.
- Between the two sits `sqrt(eps)`: the accuracy to which rounding in an objective limits
  where its minimiser can be located (Gill, Murray and Wright, *Practical Optimization*,
  1981, section 8.2.2, written in machine precision, `eps = 2u` here). A move beyond it is
  a different answer, not rounding, and it is still far below anything priced.

The floor of each output's scale:

- Predictions and standard errors are read relative to themselves. An entry under a
  thousandth of the output's largest baseline entry is read against that thousandth,
  because a relative move of a near-zero entry says nothing a user reads.
- The intercept has a floor of 1 on the link scale. Moving it by `d` moves every linear
  predictor by `d`, so a log-link price moves by a factor of about `1 + d`.
- EDF has a floor of 1 effective parameter.
- The dispersion, theta and power are relative.

Smoothing parameters are reported, as the largest `|Δ log λ|`, and not gated. Along a flat
direction they are not identified, and rounding can move them by orders of magnitude while
the EDF and predictions stay put (AGENTS.md, *Fidelity and guards*). The EDF and the
predictions carry their effect.

## Every fit on a benchboard machine is recorded

`SuperGLM.fit`, `fit_path`, `fit_reml`, `estimate_p` and `estimate_theta`, and `SuperLSS.fit` and
`fit_reml`, each write one record as they end, on any machine where
benchboard's state directory exists (`BENCHBOARD_DIR`, default
`~/.cache/benchboard`). Inside a benchboard job the record goes to the job's
spool; outside one it goes to `fits/unowned/`, and benchboard imports it at the
end of its next job under one task per process. A small marker written as the
fit starts and removed with the record lets benchboard keep a fit that was
killed as a censored one.

A record describes the problem (rows, coefficients, terms and pairs, each
term's kind, width, cardinality and non-zeros per row, empty and
all-zero-response cells, the smallest level count, weight, offset and
response spread, per numeric input its distinct values, skew, ties at the mode
and share beyond 5 sd, family and family parameters such as a negative
binomial theta or Tweedie power, link, binning and the fit's options), the
conditions (the thread pools the process actually runs, the commit, packages,
machine, caches, CPU clock, free memory, load and steal at the start and end,
and whether it is the process's first fit) and the outcome (phase times, work
counts, iterations, smoothing parameters, convergence, termination reason,
certification flags, CPU seconds, numba compile seconds and peak RSS). Rows,
responses and coefficients are summarised, never written.

Without benchboard's directory the recorder is not imported and a fit pays a
cached directory check. `SUPERGLM_FIT_RECORD=0` turns recording off. A recorder
failure never changes a fit's result, exception or warnings.

## When a baseline is reused, and when it is not valid

A baseline is reused only when its key matches the branch record in everything
but the SHA:

- the dataset hash, computed from the prepared train and test frames, so a
  change to the data files or to their preparation invalidates it;
- the case-spec hash: family, link, features, interactions, binning, fit method
  and options, the warm-up, and the harness hash;
- the harness hash: the source of the harness code that builds the response,
  weights and offsets, computes the held-out metrics, counts the work and reads
  the fit's fields, with the counter tables, so editing that code retires the
  baselines stored under the old version, and `compare` names it as the cause;
- the environment: an upgrade of any of those packages or of BLAS, or a different
  OpenBLAS core type, needs a new baseline;
- the machine: CPU model, core count and RAM;
- the thread setting, and the BLAS and numba pool widths observed under it,
  since affinity, cgroups or a library can narrow a "default" pool. The widths
  are read in the fit's own process, so `baseline` reuse and benchboard's case
  name leave them out (the `key` step runs in another process), while
  `compare` refuses on them. When the widths are all that differs, `compare`'s
  refusal names the remedy: re-time the baseline with `--force`. A record whose
  key predates the widths is never reused.

superglm's own fit recorder is off during a `books.py` run, which records the
fit itself, so the timed fit carries no recorder work.

`compare` refuses when any of these differs, and names the component and both
values. Store a new baseline at the same commit in the new environment rather
than comparing across environments. Only a timed, successful fit of a tree with
no uncommitted changes to `src/`, `pyproject.toml` or `uv.lock` counts as a
baseline.
