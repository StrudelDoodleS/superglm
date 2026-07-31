# Mixed-type interaction screening (PSST v2) — design

**Date:** 2026-07-31
**Status:** approved in discussion; awaiting spec review
**Branch:** `mixed-interaction-screening` (cut from `origin/master` @ c1b1339)

## Goal

Extend `SuperGLM.screen_interactions` (PSST — Penalized Smooth Score Test)
beyond spline×spline pairs to every interaction kind the model can refit:
spline×categorical, numeric×categorical, categorical×categorical, and
numeric×numeric — plus OrderedCategorical (spline mode) margins, which
requires a small fit-side enabler because OC parents cannot build any
interaction today. One sweep, one z-ranked table, the confirmatory refit
stays the gate.

This is the follow-up the original plan deferred ("`by=`-style
varying-coefficient candidates … own plan",
`docs/superpowers/plans/2026-07-28-interaction-screening.md`).

## Decisions taken (with Max, 2026-07-31)

1. **spline×numeric: deferred.** There is no fit-time varying-coefficient
   term (`B(x)·z`) in the interaction dispatch table, so screening it would
   rank something a refit cannot confirm. It waits for a `SplineNumeric`
   term class in its own plan. The numeric-margin machinery built here
   makes the later screening delta small. Users who suspect spline×numeric
   structure respec the `Numeric` as a `Spline` and get ti() screening.
2. **OrderedCategorical: full fit-side enabler.** Spline-mode OC becomes a
   working interaction parent everywhere (it is already *dispatched* as
   "spline" by `_spec_kind` but every builder raises
   `TypeError: Expected a spline spec` — verified live). Step-mode OC
   (deprecated) is rejected as an interaction parent with a clear error.
3. **Polynomial margins: deferred.** poly×cat / poly×poly refit targets
   exist and would ride the same adapter, but they stay out of v1; the
   adapter leaves the slot open.
4. **One output table.** Single z-ranked frame with a new `kind` column;
   per-kind null noise floors are *measured* (null gauntlet re-run per
   kind) and documented, not assumed transferable from the smooth ladder.

## Repo facts the design rests on

- Screening pipeline (`src/superglm/model/screening_ops.py`,
  `src/superglm/screening/`): per-pair cell tables `(S_cell, W_cell)` from
  one fused O(n) bincount pass → score `U` / curvature `V` → overlap
  profiling against `[1 | A | B]` → `penalized_score_statistic` at an edf
  ladder → `z = (T/phi − edf0)/sqrt(2·edf0)`. Every stage downstream of
  the cell tables is menu-agnostic dense algebra.
- `penalized_score_statistic` already handles `S_ti=None`/zero: it returns
  the unpenalized Rao statistic with `edf0 = rank`, `lambda0 = 0`.
- Interaction dispatch (`src/superglm/dm_builder.py`,
  `_INTERACTION_FACTORIES`): spline+categorical → `SplineCategorical`,
  numeric+categorical → `NumericCategorical` (L−1 slope columns, one
  unpenalized group), categorical+categorical → `CategoricalInteraction`
  ((L1−1)(L2−1) non-base pair indicators, one unpenalized group),
  numeric+numeric → `NumericInteraction` (single product column),
  spline+spline → `TensorInteraction`; plus polynomial variants.
  `_spec_kind` maps spline-mode OC → "spline", step-mode OC →
  "categorical".
- `FactorSmooth` (fs/sz) is an interaction-side spec with `parent_names`,
  the penalized alternative refit for spline×cat structure.
- `Categorical` fitted state: `_levels`, `_non_base`, `_base_level`,
  optional `LevelGrouping` (screen must apply the same collapse);
  `Numeric` is a raw pass-through column; spline margins come from
  `TensorInteraction._marginal_from_spec` (centered menu + normalized
  penalty on the compact support).
- Current eligibility filter is `isinstance(spec, _SplineBase)`; fitted-
  pair exclusion covers only `TensorInteraction`.

## Architecture

### Margin adapters

Each screenable feature resolves, per pair, to a **margin**:
`(codes, support_size, menu, penalty | None)`. Three adapter families:

- **Spline** (unchanged): unique support → codes; menu = centered tensor
  marginal; penalty = normalized marginal penalty. Quantile-binning
  fallback, intermediate budgets, and lossy-discretization `approx`
  semantics untouched.
- **OrderedCategorical, spline mode**: `spec._map_to_numeric` maps levels
  to their scores, then the *inner* spline follows the spline adapter on
  that axis. Support is the ≤K distinct scores — grids stay tiny, never
  bin. Eligibility mirrors splines (inner `select=True` keeps the hard
  error).
- **Categorical** (incl. grouped): codes over the fitted spec's levels —
  raw column mapped through `LevelGrouping` where configured, validated
  against `_known_levels` (unseen levels raise via the spec's own
  validator), indexed in `_levels` order over ALL L levels. Menu = the
  non-base contrast identity: `(L, L−1)` with the base-level row zero.
  No penalty. Categorical margins never bin (support = L by
  construction) and never mark `approx`.
- **Numeric**: no grid. `z` enters every probe linearly, so the pair's
  sufficient statistics are z-weighted moments accumulated over the
  *other* margin's cells: reuse the existing cell kernels with
  transformed row vectors — channels `s`, `s·z`, `w`, `w·z`, `w·z²`
  (probe needs `s·z`/`w·z²`; the overlap span `[1 | other | z]` needs the
  rest). numeric×numeric degenerates to plain O(n) dot products
  (channels up to `w·z1²·z2²`). Numeric-margin pairs are therefore
  **exact by construction**: no cell budget pressure, no binning, no
  `approx`, regardless of cardinality.

Categorical margins need **no new kernels at all** — an `(L, L−1)` menu
with `S=0` flows through `pair_cell_moments`, `pair_score_curvature`,
`pair_overlap_moments`, and `tensor_penalty` verbatim (`tensor_penalty(S,
0) = kron(S, I)`). Only numeric margins add the transformed-row channels.

### Pair kinds

| kind | probe block | penalty | null complexity | refit target |
|---|---|---|---|---|
| `ti` | centered ⊗ centered | kron-sum | edf ladder (unchanged) | `TensorInteraction` |
| `spline_cat` | centered spline menu ⊗ contrasts | `kron(S_spline, I_{L−1})` | edf ladder, rank-clamped | `SplineCategorical` (docs note `FactorSmooth` as the penalized alternative) |
| `numeric_cat` | `z·I(g=level)`, non-base | none | fixed df = L−1 | `NumericCategorical` |
| `cat_cat` | contrasts ⊗ contrasts | none | fixed df = (L1−1)(L2−1) | `CategoricalInteraction` |
| `numeric_numeric` | `z1·z2` | none | fixed df = 1 | `NumericInteraction` |

Spline margins include OC-spline margins throughout (an OC×spline pair is
`ti`; OC×categorical is `spline_cat`; OC×OC is `ti` on the two score
axes).

**Probe == refit basis, per kind.** For the three unpenalized kinds the
probe columns are the refit term's columns *exactly*: the kron of contrast
menus evaluated on a level-pair cell is `I(g1=lev_j)·I(g2=lev_k)` —
`CategoricalInteraction`'s indicator — and the z-slope columns are
`NumericCategorical`'s columns. This is stronger than the spline case.
For `spline_cat` and `ti` the probe spans the refit term's *identifiable
deviation space* (the part its parents' mains do not absorb) — the same
relationship the shipped screen documents today.

**Ladder behavior.** `spline_cat` keeps the edf ladder on its spline
direction; rungs above the block's achievable rank clamp exactly as
today. Unpenalized kinds evaluate once (all rungs would be identical):
the statistic is the profiled Rao score statistic, `edf0` reports the
achieved rank, `lambda0 = 0`.

### OC fit-side enabler

One resolver at the interaction input boundary:

```
resolve_interaction_parent(spec, x) -> (effective_spec, effective_x)
```

Identity for every spec except spline-mode `OrderedCategorical`, which
resolves to `(inner Spline, mapped scores)` — applying grouping and level
validation before mapping, exactly as `OrderedCategorical.build` does.
Applied at every point interaction classes receive parent inputs: design-
matrix build, discrete build, `transform`, `score` (prediction). After
this, `interactions=[("age_band", "region")]` and `("age_band", "power")`
build, fit, predict, and clone in the editor. Step-mode OC parents raise
`NotImplementedError` naming the deprecation instead of today's opaque
`TypeError`. The interaction classes' `isinstance` guards accept the
resolved spec; dispatch (`_spec_kind`) is already correct.

Downstream consequences handled: `_pair_refits_discrete` and the
`approx` logic consult the *inner* spline's `discrete`/`n_bins` for OC
margins; tensor-parent validation (`select=True`, multi-`m`) applies to
the inner spline.

### Statistic, ranking, calibration

Unchanged: working score/weights at the fitted eta (stabilized predictor,
not `link(predict())`), Pearson `phi` on the `n − edf` exposure contract,
`z = (T/phi − edf0)/sqrt(2·edf0)`, best rung wins, rank by `z` only,
`attrs["phi"]`, `phi=` override.

New: the null gauntlet is re-run **per kind** — families from Bernoulli
to dispersed Gaussian, correlated parents, exposure spread, plus
rare-cell two-way tables and few-level factors — and each kind's measured
null floor is documented the way "best null z never exceeded ~4.5" is
today. Expectations: low-df kinds (`numeric_numeric` df=1; few-level
`numeric_cat`/`cat_cat`) have the most skewed chi² tails and the highest
floors; large-df `cat_cat` normalizes well. The release-gate bound stays
per-kind-generous and pinned in tests. No p-values anywhere: ranking-only
stays ranking-only.

### API and output

Same method, same signature.

- `candidates=None` sweeps all unordered pairs of eligible features —
  splines, spline-mode OC, categoricals, numerics — minus already-fitted
  pairs. Eligibility exclusions: `RandomEffect`, `Polynomial` (deferred),
  step-mode OC, single-level-after-grouping categoricals.
- Explicit `candidates=` entries must be an eligible kind; spline×numeric
  and polynomial pairs get a specific "deferred — respec the parent or
  see the screening guide" error, not a generic one.
- Fitted-pair exclusion extends from `TensorInteraction` to **all**
  interaction classes plus `FactorSmooth`, via `parent_names`.
- Output columns: `feature_a, feature_b, kind, statistic, z, edf0,
  lambda0, n_cells, approx`. `kind` uses the table above. `n_cells`
  reports the gridded cells actually used: the product of the two
  margins' grid sizes, where a spline/OC margin contributes its support
  size, a categorical margin contributes L, and a numeric margin
  contributes 1 (so numeric×numeric reports 1).
  `approx` can only be True for rows with at least one spline/OC margin
  (binning or lossy refit discretization) — categorical/numeric margins
  are always exact.
- Behavior change (intended): models with categorical/numeric features
  now get more rows from the default sweep than the previous
  splines-only sweep.

### Edge cases

- Rare cat×cat cells contribute zero rows and screen honestly weak — the
  same corner-support limit the guide documents; a refit faces it too.
- Unseen levels at screen time raise via the specs' own validators;
  missing values keep the existing finite-covariate error.
- Degenerate statistics keep NaN-row semantics and sort last.
- `select=True` parents (spline or OC-inner) keep the hard error listing
  offenders.
- Two-level factors and df=1 blocks are legal; their floor caveat lives in
  the docs.

## Testing

1. **Exactness pins** per kind: cell/moment assembly ≡ dense row-Kronecker
   assembly on random mixed-type data (including all z-moment channels),
   extending the existing release pin in
   `tests/test_interaction_screening.py`.
2. **Probe==refit df pins** for unpenalized kinds: profiled probe rank ==
   the refit term's identifiable dimension on the same data.
3. **Per-kind null gauntlet** with recorded floors (documented in the
   guide; generous bound pinned as a release gate).
4. **Power/recovery**: planted per-level slope (numeric_cat), planted
   two-way table (cat_cat), planted deviation curve (spline_cat), planted
   product tilt (numeric_numeric), planted OC×cat effect on banded data —
   each surfaced at material z and confirmed by its refit target; rank
   ordering sane against deviance gain on refit.
5. **OC enabler round-trip**: OC×cat and OC×spline fit_reml, predict,
   editor clone, discrete on/off, step-mode rejection message.
6. **End-to-end sanity** on a freMTPL2-style frequency book: the screen
   surfaces known structure (e.g. region×brand, age×density) and the
   confirmatory refits agree.

## Deliberately out of scope

- spline×numeric screening and the `SplineNumeric` varying-coefficient
  term (own plan; the numeric-margin adapter here is its enabler).
- Polynomial margins (slot open in the adapter).
- Step-mode OC as an interaction parent (deprecated mode).
- 3-way sweeps.
- FactorSmooth-specific probe penalties (fs/sz rungs; `spline_cat` rows
  note the FactorSmooth refit option in docs).
- Fusion-penalty ladders for many-level cat×cat probes (future work;
  SCOPE/fused-lasso literature is the hook).
- Any p-value or calibrated-significance claim.

## Provenance

The parts are published; the assembly is ours and empirically calibrated.

- Unpenalized kinds: classical Rao score tests for a block of regressors
  in a GLM with efficient-score profiling of the mains — textbook GLM
  theory; block-score screening over pairs is standard practice in
  large-scale epistasis screening.
- Penalized kinds: score tests for penalized smooth terms in the
  variance-component tradition (Lin 1997; Zhang & Lin 2003), with the
  fixed-edf budget scan following the adaptive score-test family already
  cited by the shipped screen (Eubank & Hart; Fan's adaptive Neyman;
  multiscale testing).
- Ordered factors as smooths on scores: score-choice literature
  (Graubard & Korn 1987), penalized ordinal predictors
  (Gertheiss & Tutz 2009), and current ordered-factor scoring work
  (Azzalini, Stat 2023; arXiv:2305.03634, arXiv:2406.15933).
- Complementary, non-overlapping neighbor: reluctant interaction
  modeling/inference (Yu, Bien & Tibshirani 2019; Huang, Panigrahi, Yu &
  Bien 2025, arXiv:2506.01219) — Gaussian, 1-df linear products,
  selective inference; different goal from a GLM block-ranking screen
  with a refit gate.
- The cross-kind z ranking has no single-paper anchor; it stands on the
  measured per-kind null gauntlet plus the confirm-by-refit contract,
  the same footing the shipped budget-scan z already stands on.

No R package internals were consulted or referenced anywhere in this
design or its lineage.
