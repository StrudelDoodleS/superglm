# Research track 4: subsampling & sketching — findings

## Headline corrections to the audit's §H.3 subsample-λ hypothesis

1. **λ̂ is NOT scale-free in n.** Measured on penalised cubic spline, Gaussian REML, mgcv parameterisation:
   d log λ̂ / d log n = **0.43** (λ̂: 16.5 at n=2k → 237.6 at n=1M; EDF 9.5 → 15.9). Transplanting λ̂ from m
   rows to n rows under-penalises by (n/m)^α, α ≈ 0.2–0.57. Systematic bias, not sampling noise — decays like
   (n/m)^0.43, NOT m^(−1/2).

   Deviation from full-data REML fit at n=1M, k=24, in SEs of the fitted curve:

   | m | m/n | raw λ̂ | rescaled (n/m)^0.43 |
   |---|---|---|---|
   | 2,000 | 0.2% | 2.06 SE | 0.56 SE |
   | 10,000 | 1% | 1.85 SE | 0.53 SE |
   | 50,000 | 5% | 1.52 SE | **0.28 SE** |
   | 200,000 | 20% | 1.03 SE | **0.10 SE** |

   Raw transplant is still 1.03 SE off at a 20% subsample. Note the premise "prediction is insensitive to λ"
   IS true (12× λ error → only 20-35% MISE change) — that flatness is what makes any of this viable — but
   insensitive ≠ invariant, and a 1-2 SE shift in a rate relativity is what a validator diffs.

2. **mgcv's `samfrac` does not do what the audit assumed.** Read from CRAN source (`bam.r` ~line 2680):
   it extracts **only `coefficients`**, never `sp`; runs the subsample fit at deliberately loose tolerance
   (epsilon 1e-2 vs 1e-7); and is **skipped entirely when `discrete=TRUE`**. Wood warm-starts β and re-runs the
   full λ search. He did not freeze λ.

3. **Codebase note that resizes the whole idea:** `reml/discrete.py::optimize_discrete_reml_cached_w` already
   implements cached-W — λ trials cost O(p³) with **no data pass**. So the ~14 measured passes are **W
   refreshes, not λ trials**. The right question is "how many times must W be rebuilt", not "how many λ trials".
   This reduces frozen-λ headroom and raises the value of better starting points and looser convergence.

## Key literature

- **Sun, Zhong & Ma (2021), Biometrika 108(1):149-166** (arXiv:2004.10271) — "asympirical" smoothing parameter
  selection. λ_ASP(n;b) = λ_GCV(b)·(n/b)^(−r/(pr+1)). Thm 2: = λ_RISK(n){1+o(1)}. Thm 3: optimal rate.
  **ASP-A variant estimates the exponent empirically** from λ_GCV(b_k) across several b_k — no prior knowledge
  of r,p needed. Beat `bam` on 893k rows: 1.6s vs 0.49s RMSE-competitive. Caveats: degrades at low SNR;
  theory is Gaussian SS-ANOVA, not weighted GLM.
- Kauermann, Krivobokova & Fahrmeir (2009, JRSS-B 71:487-503) — λ must adapt with n for generalized P-splines.
- Reiss & Ogden (2009, JRSS-B 71:505-523) — REML has fewer local minima than GCV ⇒ REML λ̂ is the less
  variable target to estimate from a subsample.
- Wang, Zhu & Ma (2018, JASA 113:829-844) — OSMAC. Ai, Yu, Zhang & Wang (Statistica Sinica, arXiv:1806.06761)
  — extends to Poisson. **Yu, Wang, Ai & Zhang (2022, JASA, arXiv:2005.10435) — quasi-likelihood, which is the
  one that covers exposure-weighted Poisson: π_i ∝ w_i|y_i − μ_i|·‖x_i‖ falls out of their Thm 4.**
  Guards required by the papers: residual threshold δ (1e-6) or zero-residual rows get π=0; shrinkage toward
  uniform. **Known failure: variance formula under-estimates for imbalanced data — validated only for event
  proportions 0.15-0.85. Insurance frequency is far outside. Treat subsampled asymptotic SEs as unreliable.**
- **Lie & Munteanu (arXiv:2410.22872)** — Poisson coresets: **Ω(n) lower bound**, and "subsampling for the
  log-link is not possible with multiplicative (1±ε) error guarantees". Lower bound extends to *arbitrary*
  data-reduction up to log(n) (Lemma 6.2, communication-complexity reduction). Only ID-link and sqrt-link admit
  sublinear coresets. **This is a theorem — kills any "(1±ε)-guaranteed compressed book" proposal.**
- Munteanu et al. (NeurIPS 2018, arXiv:1805.08571) — same story for logistic.
- Ma, Mahoney & Yu (2015, JMLR 16:861-911) — neither leverage nor uniform sampling dominates; superseded by
  OSMAC's shrinkage. **Skip leverage sampling.**
- RandNLA survey (arXiv:2302.11474 §3.2.2) — crossover m≥50n, m≥1e5; superglm formally qualifies BUT the win
  is avoiding O(mn²) dense QR, which superglm already avoids via structured kernels + O(p³) Cholesky.
  **Forming the sketch costs a full data pass — the scarce resource. Skip.**
- Zhang, Duchi & Wainwright (2013, JMLR 14:3321-3363); Shang & Cheng (2017, JMLR 18:3809-3845) —
  divide-and-conquer: low value in-process; averaging λ̂ inherits the n-scaling bias. Yu et al. Alg. 3 is the
  version to use IF superglm ever distributes.
- King & Zeng (2001); Fithian & Hastie (2014, Ann. Stat. 42:1693-1724); Chen/Blanchet/Dembczyński
  (arXiv:2410.08994) — **all binary-response theory. Poisson-with-exposure is NOT rare-event logistic**: a
  zero-claim row with 1.0 exposure carries real rate information (score contribution −w_i μ_i x_i, not small).
  Import the design insight (stratify on response), not the estimator.

## Ranked recommendations

1. **Warm-start λ from a rescaled subsample fit, converge on full data** (highest EV). Fit at m≈2-5%n, rescale
   by (n/m)^α̂ with α̂ from 2-3 subsample sizes (ASP-A), feed as the Fellner–Schall starting point. Because FS
   is a fixed point, **the limit is start-independent** — final λ is a genuine full-data REML estimate,
   identical up to tolerance regardless of seed. Wrong α̂ costs one iteration, not correctness.
   **~2-2.5×, zero statistical contract change, no reproducibility or parity exposure.**
2. **Change the convergence criterion from λ to EDF / linear predictor** (free — do first). Converging λ to
   1e-7 is meaningless when risk is flat to 12× in λ. Stop when EDF moves <~0.01/term or the linear predictor
   moves < a fixed fraction of its SE. No estimator-definition change. Likely 2-5 passes saved on its own.
3. **Opt-in `lambda_subsample=` frozen-λ path** (~3-3.5×), only with: (n/m)^α̂ rescale with α̂ estimated;
   m ≥ max(5%n, 2000×effective params); recorded seed + α̂ + m on the fitted object; documented discrepancy in
   SE units. **HARD BLOCKER: must be refused when `select=True`** — double-penalty λ selection IS term
   selection, so a term shrunk to zero on the subsample stays zero on full data: a model-structure decision
   made on 5% of the book. Not defensible in pricing.
4. **Exposure-aware stratified sampling for the λ subsample.** Critical distinction the literature doesn't
   state explicitly: **OSMAC-style optimal (residual-driven) subsampling deliberately distorts the design
   distribution — which is exactly what the REML criterion integrates over. Use optimal subsampling for the
   pilot β fit; use design-preserving stratified sampling for the λ subsample.** Mixing them silently biases λ̂.
5. OSMAC for the pilot/bootstrap fit only (see quasi-likelihood result above + guards).
6-10. Rare-event literature: context only. Leverage sampling, sketching, coresets, divide-and-conquer: skip
   (reasons above).

## Proposed user-facing contract if #3 ships

> `lambda_subsample=f` selects smoothing parameters on a random f-fraction of rows and applies them, after a
> sample-size correction, to a final full-data PIRLS fit. **Coefficients are always full-data maximum penalised
> likelihood estimates.** Only the smoothing parameters are subsample-derived. The subsample seed, fraction and
> fitted scaling exponent are recorded on the fitted model. Fits are reproducible given the seed but are *not*
> identical to `lambda_subsample=None`; expect fitted-value differences up to ~0.3 SE at f=0.05. Not available
> with `select=True`.
