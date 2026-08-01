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

## From rank to decision: does the pair pay for its own df?

Everything above ranks pairs. Ranking is a different question from whether the
top-ranked pair should be refit, and at wide factors the two answers come apart
far enough to invert the decision.

This section is a **simulation, not freMTPL2** — Gaussian, balanced parents, a
planted truth of known shape. The question needs the ground truth held fixed
while the block's df is varied over a wide range, which one real book cannot
supply. Read it as a mechanism study; the freMTPL2 numbers above are the
evidence on real data.

A 41×41 `cat_cat` pair carrying a genuine 6σ effect in 5 of its 1,681 cells
scores **z = 17.75** — unambiguous by any conventional reading — and costs
**+22.5% holdout MSE** when refit as a fixed interaction (three replicates:
+22.3%, +20.5%, +24.6%). In sample the refit looks like the best model
available: train MSE 1.1244 → 0.7249. It spends 1,635 effective df on 6,000
rows to recover five cells and memorises the other 1,676.

The same pair fitted as a `RandomEffect` on the cell — 645 effective df instead
of 1,635 — *improves* holdout by 3.2%. So the pair is real, the detection is
correct, and "add it as a fixed interaction" is still the wrong action.

### The threshold is not a constant

[Caveat 3](#caveats) already notes that scoring by `gain - 2*edf` changes which
screen wins. That is Mallows' Cp, and on PSST's own scale it is not a rescoring
but a threshold. Since `z = (T/φ − edf0) / sqrt(2·edf0)`:

```
T/φ > 2·edf0    ⟺    z > sqrt(edf0 / 2)
```

Both sides read the **same** `edf0`, and for an unpenalized `cat_cat` the value
`screen_interactions` returns is the block's *achieved rank*, not `(L−1)²` — it
drops below the nominal rank whenever a joint cell is empty in the training
split, which is routine at these widths. The `edf0` column below is therefore
the screen's own value, read off the same row as `z`; the nominal rank is shown
beside it.

The bar **grows with the block's df**: z > 4.95 at 8×8, z > 28.3 at 41×41.
Sweeping table width against effect size, taking the screen's own z and the
holdout change from actually refitting:

| levels | edf0 | threshold | z | z/threshold | holdout Δ |
|---:|---:|---:|---:|---:|---:|
| 8 | 49 | 4.95 | 0.02 | 0.00 | +0.7% |
| 8 | 49 | 4.95 | 0.72 | 0.15 | +0.4% |
| 8 | 49 | 4.95 | 3.28 | 0.66 | −0.3% ✗ |
| 8 | 49 | 4.95 | 12.02 | 2.43 | −2.2% |
| 16 | 225 | 10.61 | 2.13 | 0.20 | +3.4% |
| 16 | 225 | 10.61 | 5.95 | 0.56 | +2.1% |
| 16 | 225 | 10.61 | 13.31 | 1.26 | −0.4% |
| 16 | 225 | 10.61 | 28.92 | 2.73 | −5.9% |
| 25 | 576 | 16.97 | 1.84 | 0.11 | +10.5% |
| 25 | 576 | 16.97 | 7.16 | 0.42 | +6.9% |
| 25 | 576 | 16.97 | 16.76 | 0.99 | +0.4% |
| 25 | 576 | 16.97 | 32.94 | 1.94 | −10.8% |
| 32 | 961 | 21.92 | 1.76 | 0.08 | +18.9% |
| 32 | 961 | 21.92 | 7.57 | 0.35 | +12.5% |
| 32 | 961 | 21.92 | 18.30 | 0.84 | +0.7% |
| 32 | 961 | 21.92 | 30.53 | 1.39 | −12.8% |

**The rule agrees with the sign of the holdout change in 15/16.** Fixed cutoffs
on the same data: `z > 2` in 10/16, `z > 3` in 11/16, `z > 5` in 10/16. The one
disagreement is a −0.3% change, which is zero.

Every fixed-cutoff failure is the same kind — z = 5.95, 7.16, 16.76, 18.30, all
comfortably "significant" and all harmful. That population grows with the table,
which is why no constant can be chosen to replace the width term.

### The total does not say how to fit it

Every χ²-family score reads only the total: PSST's `T`, FAST's RSS gain,
Information Value, mutual information, deviance change. None reads the shape, so
none separates five live cells from 1,681 faintly live ones carrying the same
total. The participation ratio of the per-cell contributions does:

```
P = (Σ t_c)² / Σ t_c²        with  t_c = n_c · mean_c² / φ
```

For k independent χ²₁ contributions E[t] = 1 and E[t²] = 3, so the null sits at
`P = k/3` **for large k**. Reporting `P / (k/3)` makes it comparable across the
wide blocks this section is about, the same property that makes `z` comparable
across them — but `k/3` is a limit, not the finite-sample expectation, and the
null ratio sits above 1 at small k (measured: ≈1.15 at k = 25, ≈1.04 at k = 100,
and exactly 3 at k = 1, where a single occupied cell carries everything by
construction). See [caveat 14](#caveats).

Spiky and diffuse truths at 41×41, magnitudes chosen so their **z values
coincide** — the only honest way to ask whether `P` carries information `z`
lacks:

| truth | z | P/(k/3) |
|---|---:|---:|
| noise | −0.26 | 1.015 |
| 5 cells @ 4.0 | 7.38 | **0.154** |
| diffuse sd=0.20 | 7.15 | 1.015 |
| 5 cells @ 6.0 | 15.67 | **0.054** |
| diffuse sd=0.30 | 14.99 | 1.007 |
| 5 cells @ 8.0 | 26.09 | **0.030** |
| diffuse sd=0.41 | 25.88 | 0.989 |

At z = 26.09 against z = 25.88 the scores are indistinguishable and `P` differs
by 33×. The measured null, 568.2 against a predicted 560, calibrates as claimed.

It pays off in the fitting decision. Ranking cells on **training residuals
only** — ranking on the full sample is the target leakage that makes supervised
binning look better than it is — and adding only the top m as levels:

| truth | top-5 | top-10 | top-25 | top-50 | all 1,681 |
|---|---:|---:|---:|---:|---:|
| 5 cells @ 6.0 (P/(k/3) = 0.054) | **−7.2%** | −6.4% | −4.5% | −1.2% | +22.5% |
| diffuse sd=0.30 (P/(k/3) = 1.007) | +0.6% | +1.0% | +2.5% | +4.2% | +22.5% |

**Five parameters beat 1,680 by 29.7 points**, and the optimum sits exactly at
the true number of live cells. On the diffuse truth the identical procedure
degrades monotonically — there is nothing localised to find, so every cell added
is a memorised residual. `P` predicts which of the two you are in; `z` cannot,
because by construction it is the same in both.

Together the two readings give a decision rather than a score:

| | `P/(k/3)` ≈ 1 | `P/(k/3)` ≪ 1 |
|---|---|---|
| **z below threshold** | skip, or pool the cell | fit the few cells that carry it |
| **z above threshold** | refit the pair as a fixed interaction | fit the few cells that carry it, and check the full refit against it |

The right-hand column stays the same above and below the bar because the two
readings answer different questions. Clearing the Cp gate says a **full refit
beats leaving the pair out** — not that it beats fitting the handful of cells
that carry the signal. Raise the magnitude on a concentrated truth far enough
and z crosses any bar while `P` stays near zero; taking that as "refit as fixed"
throws away the shape information and spends the other 1,600-odd df on nothing.
The gate is a floor on the full refit, not a ranking of it against the sparse
one.

Neither reading is implemented in `screen_interactions`, and they do not cost
the same to obtain. The **gate is** arithmetic on the returned row: `z` and
`edf0` are both columns of it. **`P` is not** — `screen_interactions` returns
aggregates (`statistic`, `z`, `edf0`, `n_cells`), not the per-cell
contributions, so obtaining it takes one extra pass over the mains-model
residuals grouped by joint cell, which is what `cell_contributions` in the
benchmark does. Reproduce with
`uv run python benchmarks/screening_worth_gate.py`.

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

The remaining caveats apply to
[From rank to decision](#from-rank-to-decision-does-the-pair-pay-for-its-own-df)
only, which is a separate simulated study.

8. **Gaussian and balanced.** `2*edf` is a Gaussian-family argument. The
   constant has not been checked for Poisson with exposure — the family this
   library is actually aimed at — and there is no reason to assume it carries
   over unchanged. This is the first thing to check before the gate is quoted
   anywhere else.
9. **Sixteen points, three replicates, and a grid chosen to straddle the bar.**
   Enough to establish that the crossing tracks `sqrt(edf0/2)` across a 20×
   range of df, not enough to pin the constant. A true crossing at 1.2 rather
   than 1.0 would not be resolved by this data. Separately, the effect sizes at
   each width were **selected so z brackets `sqrt(edf0/2)` there**, which is the
   right design for locating the crossing but also guarantees a grid whose
   boundary moves with width — precisely the configuration no constant cutoff
   can track. Part of the margin over fixed cutoffs is therefore a property of
   that selection rather than of the data; a grid drawn from some plausible
   distribution over (width, effect) would give a different one.
10. **The gate is for a plain fixed refit.** A shrunk term spends less edf, so
    its bar is lower: the `RandomEffect` fit spent 645 df against the fixed
    fit's 1,635 and improved holdout where the fixed fit cost 22%. The
    threshold answers "gate this into `interactions=[...]`", not "is this pair
    ever usable".
11. **Two of the decision table's four cells are unmeasured.** Pooling on a
    diffuse truth was never run, so the "pool the cell" half of the lower-left
    cell is inference. And *no* diffuse truth in the study clears its own
    threshold, while every truth in the gate ladder is spiky, so all of the
    above-threshold evidence sits in the right-hand column. The upper-right
    cell ("z above threshold, `P` ≈ 1 → refit as fixed") is inference from the
    spiky rows, not a measurement. The two right-hand cells' shared advice is
    likewise reasoning, not a comparison: no run in the study puts a
    concentrated truth above its threshold and then fits the sparse and full
    models against each other there.
12. **Planted truths are scattered single cells.** That is the best case for
    top-m cell selection and the worst case for a group-structured penalty. A
    contiguous block of live cells — arguably the more realistic shape for a
    rating interaction — was not tested, and would likely reorder the last
    table.
13. **`P` is measured on the mains-model residuals**, so it inherits whatever
    the mains model failed to absorb. On a book where the additive fit is poor,
    concentration would read structure that belongs to a margin.
14. **`P/(k/3)` is calibrated for wide blocks only.** `k/3` is the large-k
    limit of the null, not its finite-sample expectation: the ratio's null mean
    is ≈1.39 at k = 8, ≈1.15 at k = 25, ≈1.04 at k = 100 and ≈1.003 at k = 1600
    (measured; pinned in `tests/test_screening_worth_gate.py`). At the bottom it
    inverts outright — one occupied cell gives `P` = 1 by construction and the
    ratio reads 3, the value that elsewhere means "as diffuse as noise". A
    narrow or thinly occupied block needs a finite-k calibration before its
    reading means anything.

## Reproducing

The FAST comparison requires `interpret-core`, which is not a dependency of
this library; it was supplied out of tree and nothing in the package was
modified to run it. The mains-model and relativity measurements in
[the baseline section](#the-baseline-deviance-against-shape) need only
superglm and the freMTPL2 parquet.

[From rank to decision](#from-rank-to-decision-does-the-pair-pay-for-its-own-df)
is simulated and needs neither, so it runs from a clean checkout:

```
uv run python benchmarks/screening_worth_gate.py
```

It prints four tables: the gate ladder, concentration at matched z, the sparse
payoff, and the three-model-class comparison the `+22.5%` and `−3.2%` figures
come from. Expect roughly 45 minutes at the defaults, nearly all of it in the
wide fixed refits — which is part of what the section is about. Their
wall-clock moves several-fold under CPU contention and should not be read as a
benchmark of the fitting paths; the holdout columns are unaffected. The
arithmetic underneath the two readings is guarded by
`tests/test_screening_worth_gate.py`.

One trap worth recording for anyone extending the FAST comparison:
InterpretML's `term_features_` is **sorted by arity then feature index**
(`order_terms`), not by FAST rank. Reading `term_features_[0]` as "the pair
FAST liked most" measures column order — on the setup above, relabelling so the
narrow pair sits at indices 0,1 rather than 2,3 moves it from "chosen first in
0/6" to "6/6" on identical data. Recovering the real ranking requires
instrumenting `calc_interaction_strength` and replaying the cross-bag
aggregation, which averages ordinal ranks and discards strength.
