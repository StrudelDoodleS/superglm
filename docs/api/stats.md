# Statistical helpers

Distribution functions used by the smooth-term tests, exposed for readers
who want to reproduce a p-value by hand.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.psum_chisq
   superglm.satterthwaite
   superglm.wood_test_smooth
```

## Simultaneous bands

A pointwise band at level $\alpha$ is crossed somewhere far more often than $\alpha$ once a plot has
many points or cells. The helpers below build simultaneous bands that hold for the whole plot at
once, so a well-specified model leaves its band anywhere with probability at most $\alpha$.

- `equal_local_levels_band` builds a `UniformBand` for the $n$ sorted values of a Q-Q or P-P plot
  on the probability scale. Each order statistic is tested against its `Beta(i, n + 1 - i)` law at
  one shared two-sided local level $\eta$, chosen so the global level is $\alpha$ (exact search for
  $2 \le n < 250$, asymptotic formula for $n \ge 250$ at $\alpha \in \{0.01, 0.05, 0.1\}$). Map its
  `lower`, `upper` and `median` arrays through a distribution's quantile function to draw the band
  on that distribution's scale.
- `bonferroni_binomial_band` bounds the counts of $m$ cells, each `Binomial(trials, probability)`,
  at per-tail level $\alpha / (2 m)$ or at an explicit `cut`. By Bonferroni's inequality the global
  level holds under any dependence between cells, including multinomial histogram bins and quantile
  exceedance counts.
- `order_statistic_level`, `binomial_cell_level`, `monte_carlo_level` and `band_at_level` calibrate
  a band by simulation when parameters are estimated rather than known: `order_statistic_level` and
  `binomial_cell_level` compute the plot-level test statistics, `monte_carlo_level` reads the
  critical level from $B$ null replicates using Barnard's rank $k = \lfloor \alpha (B + 1) \rfloor$,
  and `band_at_level` (or `bonferroni_binomial_band(..., cut=...)`) draws the band at that level.

`equal_local_levels_band` and `bonferroni_binomial_band` (with `cut=None`) assume independent,
identically distributed continuous values on the probability scale from a distribution whose
parameters are known. When parameters are fitted to the same rows, the fit absorbs part of the noise
and those bands are conservative (Durbin 1973). On rows held out from the fit, parameter estimation
error is shared across every validation row at once, making known-parameter bands anti-conservative
by an amount set by the ratio of held-out to fitted rows (West 1996); calibrating the level with
`monte_carlo_level` over posterior-predictive replicates accounts for that uncertainty, assuming
the observed statistic is exchangeable with the replicates under the null.

Sources: Berk and Jones (1979), *Z. Wahrscheinlichkeitstheorie verw. Gebiete* 47(1); Aldor-Noiman,
Brown, Buja, Rolke and Stine (2013), *The American Statistician* 67(4), 249–260; Gontscharuk and
Finner (2017), *Communications in Statistics – Theory and Methods* 46(5), 2332–2342; Weine, McPeek
and Abney (2023), *Journal of Statistical Software* 106(10); Bonferroni (1936), *Pubbl. R. Ist. Sup.
Sci. Econ. Comm. Firenze* 8, 1–62; Durbin (1973), *Ann. Statist.* 1(2), 279–290; West (1996),
*Econometrica* 64(5), 1067–1084; Barnard (1963), as described by Myllymäki, Mrkvička, Grabarnik,
Seijo and Hahn (2017), *JRSS B* 79(2), 381–404; Moran, Blei and Ranganath (2024), *JRSS B*; Li and
Huggins (2024).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   superglm.UniformBand
   superglm.equal_local_levels_band
   superglm.band_at_level
   superglm.order_statistic_level
   superglm.bonferroni_binomial_band
   superglm.binomial_cell_level
   superglm.monte_carlo_level
```
