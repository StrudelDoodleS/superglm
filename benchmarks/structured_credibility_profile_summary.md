# Structured credibility profile

Profile date: 2026-07-24

Base revision: `origin/master` at `86b2a1c78c8f0c8333672bd07c3ff835dac199d6`.

## Environment

- AMD Ryzen 9 5950X, 16 cores / 32 threads
- Python 3.13.11
- NumPy 2.4.2
- SciPy 1.17.1
- tabmat 4.2.1
- psutil 7.2.2

All timings below are fresh-model `fit_reml` wall times. Data generation and model
construction are outside the timer. Medians use uninstrumented repetitions after warmup;
cProfile, tracemalloc, and system telemetry run in separate fits.

## Results

| Case | Backend selected | Median fit | REML iterations | Notes |
| --- | --- | ---: | ---: | --- |
| Poisson, n=1,000, K=20, q=4 | `auto -> gram` | 0.0895 s | 5 | Explicit Gram median 0.0919 s |
| Poisson, n=1,000, K=30, q=4 | `auto -> structured` | 0.0522 s | 5 | Explicit Gram median 3.6810 s; 70.6x speedup |
| Poisson, n=20,000, K=1,000, q=4 | exact structured | 0.1428 s | 5 | 5.0 MB traced allocation peak |
| Poisson, n=20,000, K=1,000, q=4 | discrete structured | 0.1164 s | 9 | 9 cached solves, zero cache data passes |
| Poisson, n=50,000, K=10,000, q=4 | exact structured | 3.0121 s | 5 | 24.2 MB traced allocation peak |

For the K=30 crossover case, dense and structured results agree to:

- maximum prediction difference: `2.22e-15`;
- maximum coefficient difference: `3.89e-16`;
- REML objective difference: `1.14e-13`;
- relative lambda difference: `5.39e-16`.

The K=20 auto case resolves to the existing Gram solver and is bit-identical to an
explicit Gram fit.

## Profile-guided changes

The first K=1,000 exact profile spent most of its time rebuilding a full tabmat execution
split and repeatedly dispatching small dense products. Caching a pruned structured layout,
fusing the dense-small weighted moments, and fusing structured design products reduced the
same deterministic case from a 3.5610 s development-profile median to 0.1428 s while
preserving the REML objective `11467.52359638592`.

The first K=10,000 allocation trace exposed an unrelated eager `p x p` identity in runtime
canonicalization: 817.8 MB traced peak. Replacing it with a lazy identity coefficient map
reduced the peak to 24.2 MB (97.0%) and canonicalization time from about 104 ms to
0.17 ms. The fitted objective, coefficients, deviance, EDF, and convergence path were
unchanged.

At K=10,000, the remaining cProfile cost is linear-width work: penalized-deviance
evaluation, compact diagonal-plus-low-rank trace products, selected inverse diagonals,
and tabmat matvecs. No `K x K` design, Hessian, covariance, or identity is constructed by
the structured fit.

## Reproduction

The harness is `benchmarks/profile_structured_credibility.py`. Representative commands:

```bash
uv run python benchmarks/profile_structured_credibility.py \
  --n 20000 --levels 1000 --family poisson --small-width 4 \
  --backend structured --repetitions 5 --warmups 2 --max-reml-iter 8

uv run python benchmarks/profile_structured_credibility.py \
  --n 20000 --levels 1000 --family poisson --discrete --small-width 4 \
  --backend structured --repetitions 5 --warmups 2 --max-reml-iter 30

uv run python benchmarks/profile_structured_credibility.py \
  --n 50000 --levels 10000 --family poisson --small-width 4 \
  --backend structured --repetitions 3 --warmups 1 --max-reml-iter 12
```

Use `--backend auto` with a level count within `--dense-max-levels` to record both the
automatic selection and dense parity. Raw generated profiles live under
`benchmarks/results/structured_credibility/` and are intentionally git-ignored.
