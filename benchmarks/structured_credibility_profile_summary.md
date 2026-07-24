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

### Scalar random effects

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

### Block factor smooths

All rows below converged at `reml_tol=1e-7`. `K` is the number of factor levels,
`k` is the marginal basis width, and the dominant coefficient width is `K * k`.

| Case | Backend selected | Median fit | REML iterations | Notes |
| --- | --- | ---: | ---: | --- |
| Gaussian, n=2,000, K=6, k=5, q=2 | exact structured | 0.0936 s | 6 | Explicit Gram median 3.1890 s; 34.1x speedup |
| Poisson, n=6,000, K=50, k=5, q=4 | exact structured | 1.8517 s | 7 | Includes a separate global spline |
| Poisson, n=10,000, K=100, k=5, q=4 | discrete structured | 0.2221 s | 10 | 500-coefficient factor-smooth block |
| Gamma, n=8,000, K=40, k=10, q=4 | exact structured | 0.5244 s | 6 | 400-coefficient factor-smooth block |
| Poisson, n=20,000, K=300, k=10, q=4 | discrete structured | 35.1557 s | 11 | Global spline plus a secondary 25-level random effect |

The converged K=6 Gaussian reference agrees with the dense solver to:

- maximum prediction difference: `2.89e-15`;
- maximum coefficient difference: `1.02e-14`;
- REML objective difference: `2.27e-13`;
- maximum relative lambda difference: `1.51e-14`;
- EDF difference: `2.13e-14`.

The block auto rule keeps a `K=5, k=5, q=2` term (`p=27`) on Gram and switches
`K=6` (`p=32`) to structured fitting. This is the same measured coefficient-width
crossover used for scalar random effects, with a block-aware cost estimate.

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

The first combined factor-smooth profile (`n=5,000`, `K=50`, `k=6`, global
spline, and secondary random effect) encoded penalty operators with structural
zero dense/cross blocks as unnecessary low-rank terms. Removing those zero
parts reduced the same eight-outer-iteration cProfile workload from 7.649 s to
1.754 s (4.4x). Its general block-diagonal-plus-low-rank trace kernel fell from
5.530 s to 0.280 s (19.8x).

On the real 30,000-policy `freMTPL2freq` example, the same change reduced the
fully converged combined random-effect plus factor-smooth fit from 57.0 s to
7.0 s without changing predictions or reported credibility. The smaller
single-term fits take 1.58 s for vehicle-brand random effects and 3.81 s for
driver-age-by-region factor smooths.

The deliberately wide `K=300, k=10` combined stress case shows the next
cProfile target honestly: about 15.0 s is penalty cross traces and 13.9 s is
tabmat cross-Gram aggregation in its 35.6 s instrumented fit. Those operations
scale with compact level blocks; the backend still avoids a dense
`3,038 x 3,038` Hessian and covariance.

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

uv run python benchmarks/profile_structured_credibility.py \
  --n 10000 --levels 100 --family poisson --discrete \
  --structured-term factor_smooth --block-size 5 --random-effects 0 \
  --backend structured --repetitions 3 --warmups 1 --max-reml-iter 20

uv run python benchmarks/profile_structured_credibility.py \
  --n 20000 --levels 300 --family poisson --discrete \
  --structured-term factor_smooth --block-size 10 --global-spline \
  --random-effects 1 --secondary-levels 25 --backend structured \
  --repetitions 3 --warmups 1 --max-reml-iter 20
```

Use `--backend auto` with a level count within `--dense-max-levels` to record both the
automatic selection and dense parity. Raw generated profiles live under
`benchmarks/results/structured_credibility/` and are intentionally git-ignored.
