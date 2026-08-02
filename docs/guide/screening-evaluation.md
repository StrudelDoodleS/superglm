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
scores **z = 8.40** on its own training split — clearing every constant cutoff
in the sweep below, `z > 2`, `z > 3` and `z > 5` — while sitting far under its
own width-scaled bar of **27.86**. Refit as a fixed interaction it costs
**+22.5% holdout MSE** (three replicates: +20.3%, +27.0%, +20.3%). In sample
that refit looks like the best model available: train MSE 1.1071 → 0.7292. It
spends **1,633.3 effective df** on 6,000 rows to recover five cells.

The same cell fitted as a `RandomEffect` spends **555.9 df** and *improves*
holdout by **3.7%**. So the pair is real, the detection is correct, and "add it
as a fixed interaction" is still the wrong action — which is the case every
constant cutoff gets wrong here and the width-scaled bar gets right.

Two things that follow are established and one is not, so it is worth
separating them up front.

**Established.** Mallows' Cp, written on PSST's own `z` scale, is an *exact*
restatement rather than an approximation — `z > sqrt(edf0/2)` is the same
statement as `T/φ > 2·edf0`, and the bar it implies grows with the block's df.
The *shape* of a score carries information its total does not: at deliberately
matched `z`, a truth concentrated in five cells and a diffuse one are separated
34-fold by the participation ratio. And that shape reading pays off in the
fitting decision — on the concentrated truth, the five best cells beat both the
mains model and the full refit, while the identical procedure on a diffuse
truth only degrades.

**Not established.** Whether thresholding on `sqrt(edf0/2)` actually beats a
well-chosen constant cutoff *in general*. The worked example above is one case;
on the sixteen-point sweep below the rule agrees with the sign of the holdout
change one row more often than the best constant does, which at that sample
size settles nothing. See [caveats 9 and 15](#caveats).

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

The bar **grows with the block's df**: z > 4.95 at 8×8, z > 28.27 at 41×41
(both measured — the 41×41 value is the achieved rank from the next table's
screen). Sweeping table width against effect size, taking the screen's own z
and the holdout change from actually refitting:

| levels | edf0 | nominal | threshold | z | z/threshold | holdout Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 49.0 | 49 | 4.95 | 0.80 | 0.16 | +0.8% |
| 8 | 49.0 | 49 | 4.95 | 2.04 | 0.41 | +0.6% |
| 8 | 49.0 | 49 | 4.95 | 5.46 | 1.10 | +0.0% ✗ |
| 8 | 49.0 | 49 | 4.95 | 15.83 | 3.20 | −1.7% |
| 16 | 225.0 | 225 | 10.61 | 2.32 | 0.22 | +3.1% |
| 16 | 225.0 | 225 | 10.61 | 6.17 | 0.58 | +1.8% |
| 16 | 225.0 | 225 | 10.61 | 13.49 | 1.27 | −0.7% |
| 16 | 225.0 | 225 | 10.61 | 28.96 | 2.73 | −6.2% |
| 25 | 576.0 | 576 | 16.97 | 0.68 | 0.04 | +9.0% |
| 25 | 576.0 | 576 | 16.97 | 5.73 | 0.34 | +5.2% |
| 25 | 576.0 | 576 | 16.97 | 15.01 | 0.88 | −1.5% ✗ |
| 25 | 576.0 | 576 | 16.97 | 30.88 | 1.82 | −12.8% |
| 32 | 957.7 | 961 | 21.88 | 0.74 | 0.03 | +15.9% |
| 32 | 957.7 | 961 | 21.88 | 5.86 | 0.27 | +9.9% |
| 32 | 957.7 | 961 | 21.88 | 15.54 | 0.71 | −1.1% ✗ |
| 32 | 957.7 | 961 | 21.88 | 26.97 | 1.23 | −13.4% |

Only the 32×32 rows show the achieved rank falling below the nominal one — at
the narrower widths 6,000 training rows fill every joint cell.

**The rule agrees with the sign of the holdout change in 13/16.** Fixed cutoffs
on the same data: `z > 2` in **10/16**, `z > 3` in **12/16**, `z > 5` in
**12/16** — the last two score identically because no row falls between them.

**This does not establish that the gate beats a fixed cutoff.** One row out of
sixteen is the entire margin, on a grid of sixteen points at three replicates,
with the effect sizes chosen to straddle the bar in the first place
([caveat 9](#caveats)). The comparison is inconclusive at this sample size and
should not be quoted as a result. What the sweep does show is that the crossing
tracks `sqrt(edf0/2)` across a 20× range of df, which is the shape the identity
predicts.

How the two rules fail is more informative than the count. Every fixed-cutoff
miss here is a **false positive**: `z > 3` admits z = 5.46, 6.17, 5.73 and 5.86,
all comfortably "significant" and all harmful, and that population appears at
every width in the sweep with the damage it admits growing from +0.0% at 8×8 to
+9.9% at 32×32. The width-scaled rule misses three rows, of two kinds: one where
the holdout change rounds to zero (z = 5.46 at 8×8), and two **false negatives**
at the wide end — z = 15.01 at 25×25 and z = 15.54 at 32×32, where refitting
actually helped by 1.5% and 1.1%. Its bar is too *high* there. That is the
opposite of a constant cutoff's failure, and it is direct evidence that the
constant in `sqrt(edf0/2)` is not pinned by this data.

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
lacks. These fit the full 12,000 rows, where the table above and the one below
fit a 6,000-row train split, so their `edf0` is the achieved rank on the full
sample: 1,598.7 of a nominal 1,600, for a threshold of 28.27.

| truth | z | P | P/(k/3) |
|---|---:|---:|---:|
| noise | −0.25 | 574.1 | 1.025 |
| 5 cells @ 4.0 | 8.02 | 81.1 | **0.145** |
| diffuse sd=0.20 | 7.48 | 571.3 | 1.020 |
| 5 cells @ 6.0 | 16.54 | 29.1 | **0.052** |
| diffuse sd=0.30 | 15.28 | 562.6 | 1.005 |
| 5 cells @ 8.0 | 27.15 | 16.3 | **0.029** |
| diffuse sd=0.41 | 26.15 | 548.8 | 0.980 |

At z = 27.15 against z = 26.15 the scores are indistinguishable and `P` differs
by 34×. The null checks out: 1,680 of the 1,681 cells are occupied at this
sample size, so `k/3` predicts 559.9, and the noise row measures 574.1 — 2.5%
high, in the direction the finite-k bias goes ([caveat 14](#caveats)).

Note also what the `z` column does *not* contain: no diffuse truth here clears
the 28.27 bar, the widest reaching 26.15. That gap is why the decision table
below is only partly measured: one cell on both axes but across two seeds
rather than by a single run, one half, and the two above-threshold cells not at
all ([caveat 11](#caveats)).

It pays off in the fitting decision. Ranking cells on **training residuals
only** — ranking on the full sample is the target leakage that makes supervised
binning look better than it is — and adding only the top m as levels. The last
column is the plain fixed `cat_cat` refit on the same seed and the same split,
so every cell in a row is one simulation. `P/(k/3)` here is the training-split
value, which is why it differs from the full-sample figure in the table above:

| truth | P/(k/3) | top-5 | top-10 | top-25 | top-50 | full refit |
|---|---:|---:|---:|---:|---:|---:|
| 5 cells @ 6.0 | 0.149 | **−9.2%** | −8.1% | −6.2% | −3.8% | +24.0% |
| diffuse sd=0.30 | 1.019 | +0.8% | +1.2% | +2.3% | +4.1% | +28.4% |

The df each arm buys over the mains model is 5, 10, 25, 50 and **1,552** — the
last being the achieved rank of the interaction block on this split, not the
nominal 1,600.

The **full refit** column is the same model on the same configuration as the
worked example's fixed arm — `spike` at 6.0, 41×41, `n` = 12,000, half-sample
split — differing only in seed, `31337 + rep` here against `4242 + rep` there.
So **+24.0%** and the worked example's **+22.5%** are two independent
three-replicate estimates of one quantity, 1.5 points apart. That gap is well
inside the worked example's own replicate spread of +20.3% to +27.0%, which is
the cheapest calibration this section has of what a three-replicate holdout
mean is worth at this width ([caveat 9](#caveats)).

**Five extra parameters beat 1,552 by 33.2 points of holdout MSE**, and the
optimum sits exactly at the true number of live cells. On the diffuse truth the
identical procedure degrades monotonically — there is nothing localised to find,
so every cell added is a memorised residual. `P` predicts which of the two you
are in; `z` cannot, because by construction it is nearly the same in both.

Together the two readings give a decision rather than a score:

| | `P/(k/3)` ≈ 1 | `P/(k/3)` ≪ 1 |
|---|---|---|
| **z above threshold** | refit the pair as a fixed interaction | fit the few cells that carry it, and check the full refit against it |
| **z below threshold** | skip, or pool the cell | fit the few cells that carry it |

The lower-right cell is the only one measured on both axes — but not by one
run, and the seam is worth naming. The worked example puts z = 8.40 against a
bar of 27.86 (table 4, seeded `4242 + rep`); the same configuration on a
different seed reads `P/(k/3)` = 0.149 and has its five best cells beat both
alternatives (table 3, seeded `31337 + rep`). Neither run spans the pair:
table 4 runs the screen but never computes `P` and has no sparse arm, and
table 3 has the sparse arm and `P` but no screen. So the chain screen → `P` →
sparse fit → full fit is walked by no single run, and the cell is measured on
two axes across two seeds rather than end to end. The other three are weaker —
see [caveat 11](#caveats) for exactly which parts are inference. The reasoning
behind the right-hand column is worth recording either way: it stays
the same above and below the bar because the two readings answer different
questions. Clearing the Cp gate says a full refit beats *leaving the pair out* —
not that it beats fitting the handful of cells that carry the signal. Raise the
magnitude on a concentrated truth far enough and z crosses any bar while `P`
stays near zero; taking that as "refit as fixed" throws away the shape
information the participation ratio just supplied. The gate is a floor on the
full refit, not a ranking of it against the sparse one.

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
   range of df, not enough to pin the constant — and the two false negatives at
   25×25 and 32×32 are direct evidence that it is not pinned, since both sit
   below a bar the refit went on to beat. A true crossing at 0.8 rather than
   1.0 would not be resolved by this data. Separately, the effect sizes at
   each width were **selected so z brackets `sqrt(edf0/2)` there**, which is the
   right design for locating the crossing but also guarantees a grid whose
   boundary moves with width — precisely the configuration no constant cutoff
   can track. Part of the margin over fixed cutoffs is therefore a property of
   that selection rather than of the data; a grid drawn from some plausible
   distribution over (width, effect) would give a different one. The margin is
   one row in sixteen, which is not a result.

   There is one directly measured statement of what three replicates buy at
   this width, and it is worth reading beside that margin. The same plain fixed
   `cat_cat` refit appears twice on two seeds — **+24.0%** in the sparse-payoff
   table and **+22.5%** in the worked example — and the worked example's own
   three replicates span **+20.3% to +27.0%**. So a 1.5-point gap between two
   three-replicate means of one quantity is comfortably inside the spread of a
   single such mean, and differences of that size between arms should not be
   read as separating them.
10. **The gate is for a plain fixed refit.** A shrunk term spends less edf, so
    its bar is lower, and the threshold answers "gate this into
    `interactions=[...]`", not "is this pair ever usable". Measured on the
    worked example: the `RandomEffect` fit spends 555.9 df against the fixed
    fit's 1,633.3 and improves holdout by 3.7% where the fixed fit costs 22.5%.
    So a pair the gate excludes can still be worth having — in another class.
    What is *not* measured is where the shrunk term's own bar sits.
11. **No cell of the decision table is measured by a single run.** The
    lower-right — z below threshold, `P` ≪ 1, fit the few cells — comes
    closest, and even it is two runs on two seeds: the worked example supplies
    the screen, and the spiky row of the sparse-payoff table supplies `P` and
    the sparse arms. Neither computes the other's half. The lower-left is
    half measured: "skip" is backed by the diffuse row, where every top-m arm
    degrades, but **pooling on a diffuse truth was never run**. Neither
    upper cell is measured at all, and not by accident: *no* diffuse truth here
    clears its own threshold — the widest, sd = 0.41, reaches z = 26.15 against
    28.27 — while every truth in the gate ladder is spiky, so nothing in the
    study puts a diffuse truth above the bar. The upper-right cell's advice to
    check the sparse fit against the full one is reasoning, not a comparison:
    no run puts a concentrated truth above its threshold and then fits both.
12. **Planted truths are scattered single cells.** That is the best case for
    top-m cell selection and the worst case for a group-structured penalty. A
    contiguous block of live cells — arguably the more realistic shape for a
    rating interaction — was not tested, and would probably reorder the
    sparse-payoff table.
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
15. **An earlier version of this section published figures the harness does not
    produce. Every number above has since been re-measured.** This is the most
    useful thing recorded here, so it is worth stating precisely. Nothing in
    this section is inherited: the four tables and the worked example were all
    taken from runs of the committed benchmark, and where the new number
    disagreed with the old one the new one stands.

    The earlier gate ladder published z = 0.02, 0.72, 3.28, 12.02 at 8×8 and
    claimed the width-scaled rule agreed with the holdout sign in **15/16**
    against 10/16, 11/16 and 10/16 for fixed cutoffs. Running the harness
    unchanged gives z = 0.80, 2.04, 5.46, 15.83 on that rung and **13/16**
    against 10/16, 12/16 and 12/16 — so the headline margin was five rows and
    is one. The concentration and sparse-payoff tables disagreed with the
    harness too, the latter by 2 points of holdout MSE per column on its spiky
    row.

    Version drift does not explain it. Both the original figures and the
    re-check ran against `src/` at **`a2611cc`** — the tip of `master`, and the
    merge base of the branch that published them, which touched no `src/` file
    at the time. No `--n` in {4k, 6k, 8k, 12k, 24k} reproduces the published
    row either. The figures were simply never produced by the committed
    benchmark.

    **The library has moved since, and the tables moved with it.** The branch
    now carries five changes to `src/superglm/solvers/rank.py` and its callers
    rather than one, and their claims are not equally strong — which matters
    here, because the rank-deficient path is exactly the path a wide `cat_cat`
    refit takes:

    - `b2de09d` replaced the alias-representative walk. Wherever the old walk
      resolved a block, the two choose the same columns — measured across 958
      deficient blocks with no disagreement — but on 126 of those the walk
      resolved nothing at all, and on 42 the new path retains a representative
      basis where the old one fell back to a spectral one.
    - `0fbef7e` ([#196](https://github.com/StrudelDoodleS/superglm/issues/196))
      certifies that choice on conditioning, and this one **does move which
      columns are selected**, on blocks whose earliest independent columns are
      near-duplicates.
    - `44e167a`, `4d2f321` and `18b06c3` stop building a Gram subspace that a
      certification is about to discard. Their claim is **byte-identical fitted
      output**, which is a stronger claim than equivalence and should not be
      read as the same one.
    - `e80b440` records that the deficient answer can differ, in
      `SHARED_RANK_POLICY.version`, now 2.
    - `6430239` replaced the conditioning fallback's criterion again, after
      review found it read individual components of a null basis that
      `eigh`/`svd` choose arbitrarily. It now selects on null-space
      leverage, `diag(N N.T)`, which no rotation of that basis can move.
      The selections it returns have identical conditioning to the rule it
      replaces on all 287 blocks where the two were compared, so this
      bought reproducibility rather than accuracy.

    Tables 1 and 2 were first taken at `a2611cc` and have been re-taken with
    `src/` at **`6430239`**, which is this branch's `src/` tree with all five
    changes in place: every published value is unchanged, to every digit
    printed. Tables 3 and 4 exist only on
    the fixed code and have not been re-taken since. The evidence that they are
    unaffected is indirect but specific: the 41×41 `cat_cat` refit they are
    built from was checked directly for #196 and produces bitwise-identical
    coefficients, predictions, edf and deviance across the conditioning
    certificate, because on that route every decomposition whose selection
    changes is superseded by a factor certification before the selection is
    used.

    That fix is also why the last two tables exist at all. On the old code one
    41×41 refit was measured at 668.55 s and the pair of tables needed twelve
    wide fits — a plausible reason nobody re-ran them, and not a reason to
    publish the output of a run that did not happen. The lesson is not "check
    your arithmetic"; it is that when re-running gets expensive enough,
    published numbers quietly stop being re-run, and the remedy is to make the
    rerun cheap and to name the command beside the figure.

    **No wall-clock figure is quoted anywhere in this guide as a property of
    this section's results.** Wall-clock does appear, and every occurrence is
    evidence about something else: the three figures below are evidence about
    measurement conditions, and the 668.55 s and 9.27 s above and in the
    Reproducing section are evidence about a fixed defect. None of them is
    offered as what this section's tables cost or as a finding of the study.

    The reason is worth recording as evidence rather than as a preference.
    Three consecutive runs of `--tables 4` — identical work, identical output —
    took **67 s, 100 s and 542 s** on a shared machine whose load average moved
    from 21 to 37 across them. An eight-fold spread on a fixed workload is what
    a contended wall-clock measurement is worth. The holdout and df columns are
    unaffected by any of it, which is why those are the only things quoted.

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

That prints all four tables at the defaults — three replicates, n = 12,000,
41×41 for the wide ones. No figure here came from that bare command,
though: each table was taken from its own `--tables` run, and they were not
all taken at the same `src/` state ([caveat 15](#caveats)). The values are
unaffected, because each runner opens its own generators on per-table seeds,
so a subset run and the full run produce identical rows.
`--tables` runs a subset when only one table needs refreshing, and which
command produced which table is:

| table | command |
|---|---|
| gate ladder | `--tables 1` |
| concentration at matched z | `--tables 2` |
| sparse payoff | `--tables 3` |
| three model classes | `--tables 4` |

Each was taken from its own run of that command. The tables are deterministic
given the seeds: three consecutive runs of `--tables 3` and of `--tables 4`
produced byte-identical output apart from the wall-clock columns.

**No runtime estimate for these commands is quoted here**, and
[caveat 15](#caveats) records the measurement that is the reason. The wide `cat_cat` refits dominate the cost —
twelve wide fits between tables 3 and 4, nine plain refits and three
`RandomEffect` fits on the 1,681-level cell. What can be said without a
stopwatch is structural: on the code before the alias-representative fix in
this branch one of the nine refits was measured at **668.55 s**, against
**9.27 s** after it (both figures from that commit, not from this benchmark).
That is the difference between a run nobody repeats and a run anyone can.

`--tables` exists so that a partial rerun is a quotable command rather than a
hand edit: anything taken from this section in future should be published with
the command and the commit that produced it. The benchmark also arms a
`faulthandler` watchdog by default, so a fit that hangs prints its own stack to
stderr every five minutes instead of merely looking expensive — which is how
the 668.55 s above stopped being mistaken for a big problem and started being a
fixable one.

Wall-clock here moves several-fold under CPU contention and should not be read
as a benchmark of the fitting paths; the holdout columns are unaffected. The
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
