# Migration: group penalty and df ledger now price a group's rank

*Refs issue #342. Ships with the first release containing `group_pricing`.*

## What changed

Two quantities used to price a penalized group at the width its term
**spans**; both now price the width it **emits** — the group's identifiable
rank — unless you opt out:

1. **The group-lasso weight.** `GroupSlice.weight = sqrt(p_g)` multiplies
   `selection_penalty` for every group. When a categorical interaction
   contains empty (or, in unpenalized fits, exactly aliased) cells, the
   build withholds those structurally-dead columns; `p_g` previously stayed
   at the full spanned grid, and now is the emitted width.
2. **The Breheny–Huang df fallback.** Fits whose penalty contributes no
   inference curvature (`SparseGroupLasso`, custom penalties) price each
   group's degrees of freedom at the same `p_g` — previously the spanned
   width, now the emitted width. `effective_df`, `phi`, AIC and BIC move
   with it.

Both sites read one build-time decision
(`dm_builder._priced_group_dimension`), so they cannot disagree.

The new constructor parameter is the seam:

```python
SuperGLM(..., group_pricing="rank")     # new default
SuperGLM(..., group_pricing="spanned")  # exact pre-change behaviour
```

## Why

The `sqrt(p_g)` convention is derived from the df of the group's score
statistic, with column count standing in for rank under a standing
full-rank assumption:

- Yuan & Lin (2006) orthonormalise every block (`X_j'X_j = I_{p_j}`), pick
  the `p_j` kernel to match ANOVA tests, and their own two-way ANOVA
  example carries `p = (I-1)(J-1)` — the rank after aliased directions are
  removed, never `I*J` cells.
- Simon & Tibshirani (2012) derive the weight from
  `E‖P_col(X_m) r‖² = p_m σ²`, which holds only at full column rank; their
  ridged group lasso generalises it to `sqrt(df_l)` with
  `df_l = Σ d_i²/(d_i²+δ)`, where a zero singular value contributes exactly
  zero.
- Lounici, Pontil, Tsybakov & van de Geer (2011) set the penalty level from
  trace/eigenvalue functionals of the group Gram; a structurally-zero
  column contributes 0 to all of them.
- Breheny & Huang (2015, §2.1) handle rank-deficient groups "by omitting
  the zero eigenvalues and their associated eigenvectors … we avoid the
  problem of incomplete rank by fitting the model in a lower-dimensional
  parameter space."
- Meier, van de Geer & Bühlmann (2008) name the quantity `df_g` outright
  and count a 4-level factor as 3.

No source supports counting never-estimable columns. Pricing the spanned
width made the same `lambda1` buy *more* shrinkage on a pruned block than
its identifiable dimension warrants, and reported df for parameters that do
not exist.

## Who is affected, and by how much

All magnitudes below are **measured**, not estimated.

### 1. An active group penalty (`selection_penalty > 0`, or an explicit `penalty=` with `lambda1 > 0`) with an interaction containing empty or nested cells

The affected group's weight drops from `sqrt(spanned)` to `sqrt(emitted)`,
so a pinned `lambda1` shrinks it slightly **less** (always in that
direction). Measured max-abs relative prediction shift, new default vs
`"spanned"`, same `lambda1`:

| lambda1 | synthetic fixture (4→3 cells) | real pricing workload (see below) |
|---|---|---|
| 0.05 | 1.21e-4 | 8.88e-5 |
| 0.5  | 9.67e-4 | 1.37e-3 |
| 3.2  | 5.68e-3 | 8.02e-3 |

Real workload: Tweedie(p=1.5), log link, a cost-per-exposure target with
exposure weight and offset, 120,883 positive-weight rows, seven banded
categorical rating factors plus the interaction of the 11-level regional
band with the 24-level usage band (the interaction spans 230 non-base
cells and emits 223 — all seven withheld cells are empty; weight ratio
`sqrt(230/223)` = 1.0157). The
weighted-L2 relative shift is an order smaller than the max-abs figures
(1.06e-5 / 1.51e-4 / 5.71e-4). No group's selection status flipped at any
of the three lambdas.

### 2. Fits on the Breheny–Huang df fallback (`SparseGroupLasso`, custom penalties)

On the same real workload with `SparseGroupLasso(lambda1=0.05, alpha=0.9)`:
the interaction's reported df falls from 206.88 to 200.93, `effective_df`
from 281.30 to 275.35 (−5.95), and `phi` from 215.7289 to 215.7183
(−4.9e-5 relative). Under `"spanned"` these ledgers are unchanged from the
previous release.

**Unpenalized and REML fits are not affected**: with `lambda1 = 0` the df
ledger comes from the exact rank-certified hat trace, which was measured
byte-identical between modes (`effective_df` 305.0, `phi`, AIC, BIC and
predictions all identical). That measurement also confirms the premise on
real data: the certified rank equals the emitted width plus intercept, so
pricing the emitted width *is* pricing the rank.

### 3. `selection_penalty="auto"` and `fit_path` grids

Both are anchored at `lambda_max = max_g ‖grad_g‖ / w_g`. Repricing raises
`‖grad_g‖/w_g` only for affected groups, so:

- **If an affected group is the argmax**, `lambda_max` — and with it the
  `"auto"` value (10% of it) and every `fit_path` grid point — rises by
  exactly `sqrt(spanned/emitted)` (pinned in
  `tests/test_group_pricing.py`; +15.5% on the synthetic fixture where the
  ratio is `sqrt(4/3)`).
- **Otherwise they do not move at all.** On the real workload the argmax
  is an unaffected 4-level rating factor, and the measured `lambda_max`,
  `"auto"` value and path anchor are identical to the last digit (ratio
  1.0000000000; the ceiling had an affected group been argmax would be
  `sqrt(230/223)` = +1.57%).

`"auto"` recalibrates each fit, so even when the resolved value shifts it
tracks the repriced geometry; a *pinned* `lambda1` does not.

## What to do

- **Cross-validated `lambda1`**: nothing. Re-running the selection
  reabsorbs the reweighting.
- **Pinned `lambda1` you must reproduce exactly** (regulatory refits,
  frozen-model operations): pass `group_pricing="spanned"` to get the
  previous fit bit-for-bit, or re-tune `lambda1` once under the new
  default. As a first-order patch for a single dominant interaction,
  scaling the pinned `lambda1` by `sqrt(spanned/emitted)` for that group's
  regime is *not* exact (the weight is per-group); prefer the seam.
- **Pipelines reading `effective_df`/`phi`/AIC/BIC from
  `SparseGroupLasso`-style fits**: expect the ledger to drop by roughly
  the pruned-cell count scaled by the group's shrink factor; the new
  numbers are the defensible ones.
- **Pickled models**: a model (or `ModelConfig`) pickled before this
  release restores with `group_pricing="spanned"` and keeps reproducing
  the results it recorded — refits, `clone_unfitted()` and
  `cross_validate` included. Only newly constructed models adopt the new
  default.

## Verification trail

- Sources verified against the papers themselves (arXiv 1007.1771;
  arXiv 1209.2160 §2.1; Statistica Sinica 22(3):983–1001; JRSS-B
  68(1):49–67; JRSS-B 70(1):53–71).
- New behaviour, its magnitude, and the seam are pinned in
  `tests/test_group_pricing.py`; the spanned-mode contract from #338 is
  pinned in `tests/test_alias_prune_adversarial.py`.
