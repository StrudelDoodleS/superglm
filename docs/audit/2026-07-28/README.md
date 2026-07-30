# Architecture audit & research sweep — 2026-07-28

Evidence base for the `perf/cheap-interactions` line of work. Everything here is **analysis**: no source
changes accompany these documents. Audit target was `origin/master` @ `f082e9b`.

## Start here

| File | What it is |
|---|---|
| `architecture-audit.md` | The main report: architecture map (A), measured complexity map (B), confirmed decluttering findings (C), target architecture (D), ranked RFC backlog (E), implementation tranches (F), do-not-implement list (G), scaling ceiling (H), research deltas (I). |
| `measured-tensor-cost.md` | **The reason this branch exists.** Measured cost of one `ti()` term on real freMTPL2, exact vs discrete, with profile attribution. |
| `research/research-CONSOLIDATED.md` | Cross-track synthesis of the five literature/benchmark tracks, with conflicts resolved and rejected ideas listed with evidence. |

## How it was produced

Three staged multi-agent passes over `origin/master`, then five research tracks:

1. **Read** — 9 subsystem readers (`subsystems/`) + 2 cProfile baselines (`profiles/`).
2. **Audit** — 7 specialist auditors producing 73 structured findings.
3. **Verify** — 19 adversarial verifiers over 16 finding clusters, refute-first. This pass **refuted** one
   high-severity recommendation with counter-measurement and **corrected** several others; those corrections
   are carried in the report rather than the original claims.
4. **Research** — 5 literature tracks (`research/`), each grounded in a measured bottleneck.

All performance claims are cProfile-based, per project convention. Where a claim was re-measured and
disagreed with an earlier estimate, the **later measurement wins and the earlier one is marked corrected**.

## Reproducing the headline measurement

```bash
uv run python benchmarks/benchmark_tensor_cost.py --n 100000
```

Writes `benchmarks/results/tensor_cost.json`. Baseline recorded at `f082e9b` (n=100k, real freMTPL2):

| config | wall | p |
|---|---:|---:|
| baseline exact | 8.79 s | 41 |
| baseline discrete | 1.41 s | 41 |
| +1 tensor exact | **63.33 s** | 122 |
| +1 tensor discrete | **5.39 s** | 122 |

One `ti()` term costs **7.2× on exact, 3.8× on discrete**, and widens the exact/discrete gap from 6.2× to
11.7×. On the exact path ~65% of the fit is sparse-matrix Gram work on the tensor block.

## What this branch is aiming at

**Cheap interactions**, in two independent halves:

- **Discovery** — rank candidate interaction pairs from the 2-D weighted histograms the discrete path already
  builds, via an exact penalised score statistic rather than a GBM proxy. No prerequisites; the histograms
  cost ~0.6 s at n=60k today. See `research/research-CONSOLIDATED.md` §2.K.
- **Fitting** — the row-tensor identity `(A′WA) = G(Ã₁)′W̄G(Ã₂)`, which forms tensor Gram blocks from those
  same histograms and small marginal bases, never touching an `n×p` array. Measured 77-352× on the block in
  isolation; targets the ~65% of exact-path tensor runtime identified above. See
  `research/research-array-methods.md`.

Sequencing note: discovery does **not** depend on the fitting work. The expensive half is fitting what you
find, not finding it.

## Caveats carried forward

- The row-tensor identity is a **combination** of two published results (Currie–Durbán–Eilers' G-operator and
  Li & Wood's marginal bin space) not published as such; the reference implementation does not appear to do it. Validate carefully.
- The score-statistic derivation for discovery is **inferred, not verified** — check before building on it.
- Screening candidates invalidates naive p-values; discovery needs held-out validation or selective inference
  before anything ships to a pricing context.
