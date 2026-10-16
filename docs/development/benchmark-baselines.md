# Benchmark baselines

`benchmarks/books.py` runs complete fits on named books and compares a branch
with a baseline that is already stored, so a pull request times only its own
branch. The baselines live in benchboard, the machine coordinator: its
`run-timed --record` store holds the wall time per case, commit and hardware,
and its fit history (`history.sqlite3`) holds the full record of each fit. A
commit that has been timed for a dataset, case spec, environment, machine and
thread setting is never timed again.

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
  numpy, scipy, numba, pandas and every BLAS threadpoolctl reports), the machine
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
reported as `reused` and not run. `--force` re-times it, which is only right
when the stored record is known to be bad. `--prepare` creates and syncs the
worktree without fitting.

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

## Every fit under benchboard is recorded

When a fit runs inside a benchboard job that captures fits, benchboard sets
`BENCHBOARD_FIT_DIR` and `BENCHBOARD_TASK_ID`, and `SuperGLM.fit`, `fit_path`
and `fit_reml` and `SuperLSS.fit` and `fit_reml` each write one record as they
end. A record describes the problem (rows, coefficients, terms and pairs, each
term's kind, width, cardinality and non-zeros per row, empty cells, the
smallest level count, weight and exposure spread, the share of zero
responses, family, link, binning and the fit's options), the conditions (the
thread pools the process actually runs, the commit, packages, machine, and
load and steal at the start and end) and the outcome (phase times, work
counts, iterations, convergence, termination reason, certification flags and
peak RSS). Rows, responses and coefficients are summarised, never written.
Without the variable the recorder is not imported and a fit pays one
environment lookup; a recorder failure never changes a fit's result or
exception.

## When a baseline is reused, and when it is not valid

A baseline is reused only when its key matches the branch record in everything
but the SHA:

- the dataset hash, computed from the prepared train and test frames, so a
  change to the data files or to their preparation invalidates it;
- the case-spec hash: family, link, features, interactions, binning, fit method
  and options, the warm-up and the metric definitions;
- the environment: a numpy, scipy, numba, pandas or BLAS upgrade, or a different
  OpenBLAS core type, needs a new baseline;
- the machine: CPU model, core count and RAM;
- the thread setting.

`compare` refuses when any of these differs, and names the component and both
values. Store a new baseline at the same commit in the new environment rather
than comparing across environments. Only a timed, successful fit of a tree with
no uncommitted changes to `src/`, `pyproject.toml` or `uv.lock` counts as a
baseline.
