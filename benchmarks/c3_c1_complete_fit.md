# C3 / C1 complete-fit benchmark

Run from the implementation worktree with its `.venv/bin/python`. The controller
starts **serial fresh workers**, each importing the requested worktree through an
explicit `PYTHONPATH`; each worker asserts the imported path. The same interpreter,
dependency installation, fixture code, and thread limits serve every arm. No legacy
assembler, solver, or Numba repair patches are installed.

```bash
.venv/bin/python benchmarks/c3_c1_complete_fit.py \
  --source baseline=/home/max/projects/superglm/.worktrees/c3-c1-baseline \
  --source candidate=/home/max/projects/superglm/.worktrees/c3-c1-completion \
  --fixture tweedie-stress --outer efs+newton --max-reml-iter 60 \
  --out .benchmark-artifacts/c3-c1/tweedie-newton
```

The frozen baseline is `8962c4520cad948aa20c480a238b7bb1e276e9cd` (v0.31.0).
`--fixture tweedie-stress` preserves seed 5915, 10,000 rows, mix .75, four knots
and q=192. `tweedie-friendly` preserves the same generator with 100,000 rows and
mix zero. `gpd-tail` preserves `marked_book(10000, tail=True)`, seed 9506, threshold
1000 and three knots: 1,401 training excesses and q=98. The recovered generator
retains its exact random draw order, including the Gaussian draw preceding
Tweedie sampling. `gaussian` is a smaller smooth/tensor smoke fixture.
`gaussian-fragmented` independently generates small numeric, categorical,
spline and spline-by-categorical terms in both Gaussian predictors. It exercises
ordinary group-pair assembly, with 65,536 training rows by default and seeds
28109/28110 for training/holdout. It uses the supported grouped-curve API;
it does not exercise factor-smooth interactions through the public predictor API.

`--fixture severity-gaussian` fits log aggregate policy claim amount, joined by
IDpol from verified freMTPL2 Parquet files in `--data`. `severity-gamma` fits the
positive amounts directly. `nb2` fits raw claim counts with the actual positive log exposure offset
in both training and validation predictions. Driver age, vehicle age and bonus
malus enter all parameter predictors. The default real-data validation excludes
every tenth sorted policy; `--n` truncates the remaining training set. `--n 0`
uses every available training row. `--holdout-every 0 --n 0` fits the full book
(about 24,944 severity policies or 678,013 frequency policies), and explicitly
labels the first 2,000 validation rows **in-sample**, not independent holdout.

`--replicate 10` copies each severity training row ten times after the holdout split
and `--n` truncation. This is a **synthetic row-replication scaling fixture**, with
the replication factor and original policy count explicitly recorded; it adds no
new independent real observations. Validation rows remain unchanged. Replication
changes the likelihood's information content and can change selected smoothing
parameters, so compare representations at the same replication factor.

`factor-smooth` fits Gaussian location and scale with a continuous spline,
categorical main effect and categorical-by-spline interaction in each predictor;
`--levels` controls category count. It exercises per-level row-subset extraction,
unlike a single `FactorSmoothGroupMatrix` direct gather.

For the C3 matrix vary `--initial-lambda` (e.g. .001, .1, 10), `--outer efs` versus
`efs+newton`, and `--no-practical-reml`. Retain converged/certified flags, gradient
certificates, endpoint direction evidence, objective and predictions together;
an iteration cap or practical plateau is not a stationarity certificate.

For C1, compare baseline dense and candidate dense using the command above, then
run candidate discrete separately with `--discrete --n-bins 256` and a different
`--out`. Options are common to all sources in one invocation: the baseline's
public discrete refusal is expected and saved as an error receipt. On continuous
covariates, sweep `--n-bins 64`, `256`, `1024`: this is **grid sensitivity**, because
quantization changes the model design. For repeated-support representation tests,
give both arms identical `--support-size 32` (or another explicit support size)
and an adequate bin count. This changes input covariates before both fits;
it is a separate fixture from the original continuous C3 stress case. Validate
actual compiled-design equality before treating a finite-support comparison as
exact, particularly if a grid has fewer cells than the support.

Wall time defaults to **unmeasured**. Coordinate an exclusive run and follow
[`cost-and-timing.md`](../docs/development/cost-and-timing.md), then add
`--measure-time --quiet-profile 'operator/session identifying a quiet host'`.
The worker refuses measured mode when load exceeds twice the available CPU count
and records load and all processes' CPU ticks around the
fit, with fixed Python/pytest/Pylance/node category labels derived
from command lines without recording their arguments. That threshold is necessary,
not sufficient: the operator must establish
quietness. If load exceeds the bound after the fit, elapsed time is discarded.
`--repeat 3` repeats the source sequence serially. Warmup runs by default before
the timed public `fit_reml`; use `--no-warmup` only for explicitly cold comparisons.
All timings come from `perf_counter` **inside the worker**. The tool proxy's clock
and transformed stdout are not receipts. Peak RSS belongs to that fresh worker,
including imports, fixture preparation, warmup, retained fit history and compiled
libraries; external processes are excluded from RSS but can contend CPU.

Category labels are heuristic command-line matches, not process identity. Exclude
the measured worker PID when auditing background activity: descriptive argument
text can label the worker itself as a proxy process. A quiet preflight does not
establish quietness during the fit. Inspect fit-period CPU deltas as well as load;
shared cache or memory-bandwidth interference can occur below oversubscription.
Retain rejected timing receipts as observations and mark the speed benefit
unmeasured. The [completion evidence](../docs/research/2026-09-c3-c1-completion-evidence.md)
records this distinction for the two local timing series.

Run `--instrument` separately, without measured mode. Python profiling counts
actual grouped row-subset/materialization and chunked solver calls, with operand
shapes where exposed, plus NumPy argsort calls. Published coefficient-fit backend
identifiers and chunk sizes independently record dispatch. The profile is not a
complete native allocation or FLOP counter and changes performance. An empty
relevant dispatch log cannot support a claimed backend benefit.

Each run writes raw JSON, compressed NPZ and a log directly from the worker.
The JSON records exact source SHA, tracked diff hash, per-source-file hashes,
interpreter/dependency/BLAS/thread metadata, environment, fixture/data fingerprints,
complete-fit stopping evidence, lambda/EDF/objective/log likelihood, coefficient
fit histories and backend dispatch. NPZ stores coefficients, covariance, training
and validation natural parameters, lambda vector, EDF, objective, log likelihood,
terminal score/curvature, and smoothing Hessian/certificates when available.
Errors and refusals also write JSON. The controller writes `manifest.json` with
raw JSON/log checksums; each successful JSON includes its NPZ checksum. Verify
these files directly rather than reconstructing them from transformed tool text.

Keep raw NPZ in the operator-selected ignored artifact directory; commit compact
receipts/report and checksums rather than large prediction arrays. Small smoke
runs establish harness plumbing only: short iteration budgets do not establish
convergence, numerical equivalence or a performance result.
