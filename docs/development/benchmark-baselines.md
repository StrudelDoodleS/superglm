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
- `smoke`: a tiny synthetic book, for developing the harness itself.

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
  `compare` refuses on them.

superglm's own fit recorder is off during a `books.py` run, which records the
fit itself, so the timed fit carries no recorder work.

`compare` refuses when any of these differs, and names the component and both
values. Store a new baseline at the same commit in the new environment rather
than comparing across environments. Only a timed, successful fit of a tree with
no uncommitted changes to `src/`, `pyproject.toml` or `uv.lock` counts as a
baseline.
