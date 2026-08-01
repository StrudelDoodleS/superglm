# Screening Evaluation

This page records a head-to-head evaluation of PSST interaction screening
against the closest existing tool, and a comparison of the mains model the
screen is anchored to against a boosted alternative. It exists so the claims
in [Interaction Screening](screening.md) can be checked rather than taken on
trust, and so the places where the comparison goes against us are on the
record next to the places it does not.

It is a point-in-time study on one book and one family. Read
[Caveats](#caveats) before quoting any number here.

## What was compared

[FAST](https://doi.org/10.1145/2487575.2487579) (Lou, Caruana, Gehrke &
Hooker 2013), the pair-ranking algorithm behind GA2M, shipped as
`measure_interactions` in InterpretML. It solves the same problem shape as
PSST: rank candidate pairs against an already-fitted additive model from one
pass of cell tables, leaving a refit as the gate.

Both screens ranked the same candidate set against the same fitted baseline,
and every candidate pair was then **actually refit** to establish ground
truth — no top-k shortlist.

## Setup

freMTPL2 frequency, on the house exposure contract:

```python
df["Exposure"] = df["Exposure"].clip(lower=0.01)
exposure = df["Exposure"].to_numpy(float)
y = df["ClaimNb"].to_numpy(float) / exposure   # claim RATE
sample_weight = exposure
```

| | |
|---|---|
| Full book | 678,013 rows / 36,102 claims |
| Study sample | 200,000 rows (`random_state=0`) / 10,633 claims |
| Split | 70/30 hash of `IDpol` → 139,790 train / 60,210 holdout |

Both *screens* handle the full book comfortably. The binding constraint is the
gold standard: one confirmatory refit of the cheapest pair on 678k rows takes
**733.8 s**, and the study needed 36 refits. The subsample is applied
identically to both methods and to the gold standard, and every ranking was
re-run on the full 678k book to confirm no conclusion depends on it.

Two specifications were run, because the choice turns out to be load-bearing:

| spec | DrivAge | VehAge | BonusMalus | VehBrand | Region |
|---|---|---|---|---|---|
| A | `Spline(ps, k=8)` | `Spline(ps, k=6)` | `Numeric()` | `Categorical()` | `Categorical()` |
| B | `Spline(ps, k=8)` | `Spline(ps, k=6)` | `Spline(ps, k=8)` | `Categorical()` | `Categorical()` |

FAST was handed the fitted superglm baseline as `init_score` on the link
scale, verified equivalent by two independent checks: the Poisson deviance
recomputed by hand at that `eta` matches the fit's own to machine precision,
and re-parameterising as counts with a `log(exposure)` offset and unit weights
rescales every strength by exactly the mean exposure, leaving the ranking
identical.

## Result: ranking quality

Spearman correlation of each screen's ranking against the ranking by
**realized out-of-sample deviance gain** from the confirmatory refit.

| ranker | spec A (n=8) | spec B (n=10) |
|---|---:|---:|
| **PSST `z`** | **+0.810** | **+0.830** |
| FAST, default | +0.667 | +0.612 |
| FAST, `Purify` flag | +0.714 | +0.661 |
| FAST, on its own EBM baseline | +0.714 | +0.661 |
| FAST, no baseline at all | +0.667 | +0.564 |
| FAST, full-tensor gain | +0.524 | +0.321 |

PSST ranks better on every FAST configuration found, on both specs, on
Spearman and Kendall.

Against **in-sample** gain the result reverses — FAST +0.394 against PSST
+0.309 — and against a complexity-penalised in-sample view (`gain - 2*edf`)
`Purify`-flagged FAST is the best ranker in the whole study at +0.964, above
PSST's +0.855. All three are reported because the choice of gold standard, not
the screens, decides the winner.

### Why the two verdicts disagree

One row accounts for it. `VehBrand x Region` is an 11 x 22 factor block:

| | value |
|---|---|
| effective df | 208 |
| in-sample deviance gain | +237.6 (rank **2 of 10**) |
| out-of-sample deviance gain | **−903.7** (rank 10 of 10) |
| PSST rank | 10 of 10 (`z` = −5.89) |
| FAST rank | 10 of 10 |

It buys the second-largest training gain in the study and destroys nearly four
times that on the holdout. Both screens rank it last; the in-sample gold ranks
it second. **Both screens are right and the in-sample gold standard is
wrong** — and because PSST is maximally "wrong" against that gold, this single
row is most of the reason the in-sample comparison reverses.

## Mechanism

The two screens disagree materially on three pairs. For each, the realized
interaction surface was refit, the shift in linear predictor extracted, and
the fraction of its weighted variance reproducible by each screen's probe
shape measured directly.

**`BonusMalus x VehBrand`** — PSST ranks it 2nd of 8, FAST 7th; the refit puts
it **2nd out of sample** (+37.2 on 9.8 edf).

| | |
|---|---|
| reproducible by a per-level linear slope (PSST's `numeric_cat` probe) | **96.4%** |
| reproducible by the best 4-quadrant step (FAST's probe) | 72.1% |
| total weighted shift variance | **576** |
| …against `VehAge x Region` | 32,778 |

**`VehAge x Region`** is the mirror image and FAST's clearest false positive:
FAST ranks it 3rd–4th, PSST 7th–9th, and the refit shows a 30.8-edf term that
gains 88 in training and **loses** it out of sample. FAST's probe explains only
**1.7%** of that pair's shift — but 1.7% of 32,778 still exceeds 72.1% of 576.

That is the whole mechanism: **FAST reports a raw average Newton gain — an
effect size. PSST normalizes against the block's own noise floor — a
signal-to-noise ratio.** A small, tightly identified, 10-edf effect worth +37
out-of-sample deviance is exactly what an un-normalised measure buries.

Corroborating this: FAST's `Purify` flag, which strips main-effect-shaped
components out of the tensor, moves FAST *toward* PSST's ordering and is its
best-performing variant. Purification is a coarse version of what PSST's
efficient-score profiling does exactly.

The obvious alternative explanation was tested and **refuted**. 57% of rows
share a single `BonusMalus` value, so tie-heavy quantile binning was the
suspect; rank-uniformising the margins and raising `max_interaction_bins` to
256 and 1024 leave FAST's ordering completely unchanged. Binning resolution is
not the mechanism.

## Where FAST wins

`VehAge x BonusMalus`. FAST ranks it **2nd on every configuration and both
splits**, and the refit confirms it: gold rank 3 by both standards, +32.0 out
of sample on 10.1 edf. Under spec A, PSST cannot rank it at all — spline x
numeric is deferred.

Stated fairly in both directions: superglm cannot *fit* that interaction under
spec A either, so the deferral is internally consistent and the screen is not
hiding something the model could exploit. And the remedy works — under spec B
PSST ranks the same pair **1st** (`z` = 11.83), above FAST's 2nd.

But the deeper reading is that spec A was the problem. `BonusMalus` fitted as a
spline uses **edf 7.5 of 11** with chi2(8.2) = 1705.9; specifying it linearly
costs 195 deviance in sample and 37 on the holdout. It is a strongly curved
feature, and the only reason this pair was ever spline x numeric is that a
worked example declared an obviously non-linear margin linear. The screening
guide's example has since been respecified. What remains true is narrower: a
practitioner who mis-specifies a curved numeric gets a queue with a real
interaction silently absent, and FAST's queue does not have that hole.

## Cost

Ten candidate pairs, n = 200,000:

| | spec A | spec B |
|---|---:|---:|
| PSST `screen_interactions` | 2.77 s | 4.89 s |
| FAST `measure_interactions` | 0.106 s | 0.113 s |
| *one* confirmatory refit (full book, cheapest pair) | **733.8 s** | |

FAST is **26–43× faster** than PSST. Both are irrelevant beside what they
replace: on the full book the cheapest single refit costs 295× a whole PSST
sweep and 2,320× a whole FAST sweep.

## The baseline: deviance against shape

PSST is defined against the fitted mains model, so the quality of that model
bounds the screen. FAST turns out to be nearly insensitive to which additive
model it screens against — its ranking barely moves between the superglm
baseline, an EBM baseline, and *no baseline at all* — which is itself a
finding: on this book it behaves more like a marginal-dependence detector than
a residual-structure detector. PSST has no such property.

That makes the mains model worth measuring. Fitted on the same split
(measurements below use an independently constructed 139,752/60,248 split, so
they are internally consistent but not directly comparable to the refit gains
above):

| model | train deviance | **holdout deviance** | fit |
|---|---:|---:|---:|
| superglm (spec B, penalized splines) | 44,419.8 | 18,933.8 | 1.5 s |
| EBM mains (`interactions=0`, 8 bags) | 43,550.3 | **18,499.4** | 8.1 s |

**The EBM mains model is genuinely better on deviance** — by 870 in sample and
**434 out of sample**. It is not merely more flexible; it generalises better.
Stated plainly because it is the result.

Two things follow. First, the gap is resolution on the smooth margins, not
regularization of the factors:

| variant | holdout | gap to EBM closed |
|---|---:|---:|
| baseline (k=8/6/8, fixed factors) | 18,933.8 | — |
| richer splines (k=20 each) | 18,705.9 | **228 of 434** |
| credibility factors (`RandomEffect`) | 18,932.6 | 1 of 434 |

Shrinking the 33 factor levels buys nothing. Knot resolution buys half the gap.

Second — and this is why the deviance table is not the end of it — the shape
EBM buys it with is not one a pricing model can ship. Measuring monotonicity
violations along the fitted `BonusMalus` relativity over its observed range:

| fit | dips | **total violation** | relativity at BM=230 |
|---|---:|---:|---:|
| superglm, ps k=8 uniform | 41 | 0.175 | 2.530 |
| superglm, ps k=12 `quantile_tempered` | 8 | 0.233 | 2.405 |
| **EBM mains** | 18 | **4.094** | 1.995 |

EBM's total violation is **23× the penalized spline's**, and it is not diffuse
wiggle. Its fitted relativity runs 0.942 at BM=70 and **0.620 at BM=90** — the
model asserts that a driver at bonus-malus 90 has materially lower claim
frequency than one at 70. That is a reversal in the middle of the scale, on
the one variable in the book whose direction is a matter of contractual
definition rather than inference.

So the honest summary is that EBM wins 434 holdout deviance and loses the
shape, and on a rating factor like Bonus-Malus the shape is the binding
constraint. This is the ordinary penalized-GAM-against-boosting trade, and it
is stated here rather than left out because the deviance comparison alone
would be misleading in our favour.

Two practical notes fell out of the same measurements:

- **Tempered-quantile knots are worth using on Bonus-Malus**, as
  [Feature Types](features.md) already recommends: they improve holdout
  deviance and cut dips from 41 to 8. Plain `quantile_rows` is **bit-identical
  to uniform** on this feature and buys nothing — with 57% of mass at the
  scale minimum, every quantile up to the 57th *is* that value, so the interior
  knots collapse onto one point and de-duplicate back to uniform. Tempering is
  what rescues it.
- Enforcing monotonicity directly is the obvious response to the shape
  problem, and it does not currently work through `fit_reml()`. See
  [issue #189](https://github.com/StrudelDoodleS/superglm/issues/189).

## Caveats

Any of these could change a conclusion.

1. **Eight to ten pairs.** A one-place swap moves Spearman by 0.05–0.10, so
   the +0.83 against +0.61 margin is about two rank swaps wide. It reproduces
   on both specs, both splits and the full book, but it is not a large-sample
   claim.
2. **One holdout.** A single 60,210-row split, no repeated splitting or CV.
   `VehBrand x Region`'s −904 is unambiguous; the ordering among the small
   positives (+5.6, +2.5, +0.5, −0.25) is within holdout noise.
3. **The gold standard's definition decides the winner.** Out-of-sample gain
   favours PSST, in-sample gain favours FAST, `gain - 2*edf` favours
   `Purify`-FAST. The out-of-sample reading is taken as primary here; that
   choice should be quoted alongside the result.
4. **The gold standard is superglm's own refit** — by construction the exact
   basis PSST probes. A gold standard defined by EBM's own pair refit would
   likely favour FAST. That experiment was not run. Scoring on out-of-sample
   deviance mitigates this but does not remove it.
5. **One book, one family.** Poisson frequency on freMTPL2. The mechanism
   above predicts the ordering generalises to any book mixing small
   well-identified with large thinly-identified interactions — a prediction,
   not a result.
6. **Subsample.** Gold-standard refits are at 200k of 678k rows. Both
   rankings are stable at full scale; the refit gains themselves were not
   recomputed at 678k.
7. **`DisableNewton`** is not reachable through the public
   `measure_interactions` surface and was not swept.

## Reproducing

The FAST comparison requires `interpret-core`, which is not a dependency of
this library; it was supplied out of tree and nothing in the package was
modified to run it. The mains-model and relativity measurements in
[the baseline section](#the-baseline-deviance-against-shape) need only
superglm and the freMTPL2 parquet.
