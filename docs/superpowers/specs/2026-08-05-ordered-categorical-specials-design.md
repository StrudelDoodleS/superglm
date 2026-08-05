# OrderedCategorical `specials=` — design

**Date:** 2026-08-05
**Branch:** `oc-specials` (off `origin/master` @ `7109e7f`)
**Status:** design approved, ready for an implementation plan

## Problem

`OrderedCategorical` smooths a spline across every level of an ordered factor.
That is the right model when the levels genuinely lie on a scale, and the wrong
model when one of them does not. A `MISSING` band, or a structural `0` band, is
a different population: it has no position on the ordering, and forcing the
smooth to span it damages the estimate twice over — the curve bends to reach the
outlier, and the outlier is dragged toward the curve.

Today the only options are to smooth across the offending level anyway, or to
abandon the smooth entirely and fit `Categorical(...)`, losing the shape
regularisation on the levels that deserve it.

### Measured motivation

A spike (`oc_specials_spike.py`, 40k rows, ~2.6k claims, 10 ordered bands whose
true effect is a smooth saturating curve, plus an 18% `MISSING` population whose
true relativity is 0.577) compares treating `MISSING` as an ordered level
against holding it out as a free level. Mean over 12 seeds, in relativity points
×100:

| | ordered rmse | MISSING error |
|---|---|---|
| MISSING smoothed as an ordered level | 17.25 | 10.42 |
| MISSING held out as a free level | 10.25 | 3.99 |
| fully saturated, no smoothing | 12.45 | 3.98 |

The `MISSING` estimate is the robust win: 10.42 → 3.99, matching the saturated
fit, which is what "fit it free" should mean. The curve improvement is real on
average but noisy — holding out beat smoothing-across on the ordered levels in
8 of 12 seeds, and beat the saturated fit in 6 of 12. **The case for this
feature is that the special stops corrupting the curve and the curve stops
corrupting the special — not that it beats a saturated fit.**

Specials are for *structural* difference, never for mere sparsity. The penalty
already handles sparse bands better than free levels do.

## API

```python
OrderedCategorical(
    order=["1", ..., "10"],
    specials=["MISSING"],
    basis=Spline(kind="ps", k=8),
)
```

`specials` is a list of level labels, default `None`. Labels are matched exactly
as they appear in the data column, identically to `order`.

### Normalisation

A label appearing in both `order` and `specials` is removed from `order`, and
from `values` if given. Spline positions are then computed over the surviving
ordered levels. Specials never receive a numeric position.

### Validation

| Rule | Rationale |
|---|---|
| `base=` may never be a special; `_choose_base`'s `most_exposed` selects from ordered levels only | `MISSING` is often the most exposed level in a real book, so the default would silently anchor the term on the free level. `_base_log_effect` (`ordered_categorical.py:501`) would then `KeyError`. |
| A declared special absent from the training data is a fit-time `ValueError` | Its indicator column would be all-zero and its coefficient unidentifiable. Deliberately asymmetric with ordered levels, which may be declared-but-unobserved and still get a valid basis row from their position. |
| At least two ordered levels must survive normalisation | Otherwise there is no curve. The message points at `Categorical(...)`. |
| `specials` with `basis="step"` is a `ValueError` | Step mode is deprecated and already refuses interactions. |
| A level grouping may not mix a special with ordered levels | Merging a special into an ordered group would smooth it after all. Grouping ordered levels on a term that has specials is allowed. |

## Construction

`build()` returns **two** `GroupInfo` objects where it returned one:

| block | columns | rows for ordered levels | rows for specials | penalty |
|---|---|---|---|---|
| `<name>:spline` | centered spline basis | basis at the level's position | zero | `S_spline`, unchanged |
| `<name>:special` | one indicator per special | zero | 1 | none, `penalized=False` |

The spline is built on the **ordered rows only** and the result row-expanded
back to `n` rows with zeros. This is not a convenience: `build_identifiability_projection`
(`_spline_identifiability.py:23-29`) forms its constraint as a column sum over
the rows present. Build over all rows with specials mapped to some numeric value
and `1'(B@Z) = 0` no longer holds on the zero-filled matrix, the intercept stops
being the ordered-row baseline, and `_place_knots` (`_spline_build.py:70`) sees a
fabricated coordinate.

Given that, `[1 | zero-filled centered spline | special indicators]` is **not**
rank deficient — a centered basis cannot reproduce a constant, so no indicator is
recoverable from the other columns — provided every special is non-empty, which
validation guarantees.

The special block carries `projection=None, reparametrize=False,
penalty_matrix=None, penalized=False`. This matters: it is **not** the
`TensorInteraction` configuration. That precedent (`interaction.py:1351-1370`)
works because both its subgroups project into the same raw basis, so the
`np.hstack(r_inv_parts)` at `dm_builder.py:1055` has matching row counts. A
specials block is a disjoint column space; giving it a projection "for symmetry"
produces a bare numpy `ValueError`. With the settings above it falls into the
terminal branch at `dm_builder.py:676-677`, is skipped by the collector at
`dm_builder.py:1032-1033`, and `set_reparametrisation` still receives the
spline's `R_inv` unchanged.

### Block order is a contract

**Spline first, specials second.** `coef_tables.py:413` and `report_ops.py:405`
read `feature_groups[0].name` as "the spline group" for λ and knot metadata, and
report *absent* metadata with no error if it is not. Both the contract is
documented on `build()` and those two readers are changed to select by
`subgroup_type` rather than by position. The failure is silent, so both.

### Parking-position invariance (measured)

The spike fitted the emulated construction twice, parking special rows at the
first and last ordered position. Fitted relativities agreed to `4.9e-15` across
the ordered levels and `1.1e-16` on the special itself. A free level effect makes
those rows uninformative for every other coefficient, so no position is needed —
which is what justifies zero-filling rather than inventing a coordinate.

## Coefficient read-back

Roughly fifteen call sites build a feature's coefficient vector as
`np.concatenate([beta[g.sl] for g in feature_groups])` and hand it to
`OrderedCategorical.reconstruct`, `.score` or `._base_log_effect`, all of which
today equate "the feature's coefficients" with "the inner spline's
coefficients". `model/base.py:357-361` actively enforces the full-width contract.

**The split therefore lives entirely inside `OrderedCategorical`.** The vector
stays full-width everywhere outside the term. Correspondingly `transform` widens
to emit `[centered spline | special indicators]` in GroupSlice emission order.
Four existing width guards fire immediately if that ordering is wrong —
`_ordered_reference.py:72-77` (RuntimeError), `_metrics_design.py:136-139`
(ValueError), `model/base.py:357-361` (ValueError), and `_term_helpers.py:120-127`
(raw `IndexError`).

### Level bookkeeping

Specials go **into** `spec._ordered_levels`, so `coef_tables.py:441`,
`summary_levels.py:73`, `export/summary.py:234`, `_term_covariance.py:106` and
`comparison.py:112` pick them up without change. They stay **out** of
`_level_to_value`; a separate smooth-only level list drives base selection, the
knot clamp and the difference penalty. A special never receives a spline-axis
coordinate that could leak into `transform` or the export.

### Reported quantity

A special reports `beta_special − f(base)`, matching how ordered levels report
`f(x) − f(base)` (`_ordered_reference.py:27-50`), so the rating table can
reconstruct predictions. `transform([base])` must be zero over the special
columns, keeping the intercept SE at `coef_tables.py:184-189` correct.

## Reporting

### Summary

The term keeps **one** group row. Its whole-smooth Wood test remains a test of
the smooth alone, and the level table below marks which levels were fitted free.

This is not free. `coef_tables.py:340` selects a feature's groups with no
subgroup filter, so a second block under the same `feature_name` would silently
turn the reported p-value into a joint test of "curve is flat **and** all special
offsets are zero", and inflate `feature_edf` (`343-347`), `ref_df` (`374`) and
`n_params` (`421`). `coef_tables.py:335-467` is restructured to filter the smooth
row to the spline block. No sibling row is emitted.

The marker is a **field, never a label decoration** — the level column stays the
raw lookup key. This requires its own work: nothing in `coef_tables.py`,
`summary.py` or `summary_levels.py` consumes `TermInference`. Summary level rows
are rebuilt independently from `spec.reconstruct()` + `feature_se_from_cov()` and
keyed on `spec._ordered_levels`. A new `_CoefRow` field is threaded through
`coef_tables`, `summary_levels`, and both the ASCII (`summary.py:341-357,
411-415, 456-458, 552, 557`) and HTML (`summary.py:739-1103`) renderers.

Rendered as a `fit` column reading `smooth` / `free`, present only on terms that
have specials, so no existing OC output changes width.

### Export

The **rating sheet is correct with no change** — `export/rating_tables.py:152-171`
reads only `ti.levels` and `ti.relativity`, so a special appears as a row with
its own relativity and exposure weight, and a rater looking it up gets the right
factor.

The marker goes on the **Summary sheet only**, via the existing `SummaryTermRow`
kind machinery, and it goes on at **both** granularities. The Summary sheet does
not carry one row per term: `export/summary.py:301` iterates `summary._coef_rows`
and emits a `SummaryTermRow` per *coefficient* row, so every OC level already has
its own row there — `_canonical_level_row_names` (`228-242`) reads
`spec._ordered_levels` and therefore picks specials up unchanged, and
`tests/test_rating_table_export.py:1325-1348` pins one level row per level.

So the term's whole-smooth row is marked `smooth+free`, **and** each special's
level row is marked `free level` instead of `level`. Per-level provenance costs
one `kind` string at `export/summary.py:306-310`: no new column, no new sheet,
and nothing the rating sheet's fixed 3-column blocks ever see. The decision to
keep the rating sheet unmarked is unchanged and rests on its own reasons below,
not on any limit of the Summary sheet.

The rating sheet is explicitly **not** given a marker column:
`excel.py:176` hard-codes `start_col = 1 + idx * 3` with number formats keyed on
`cell.column % 3` (`186`, `188`), and `tests/test_rating_table_export.py:1310-1319`
pins block 2 to columns 4–6. A fourth column would overwrite the next block's
name column and desync formatting for every later block.

## Plotting

### Prerequisite commit

On `origin/master` the two backends already draw **different curves for the same
fitted term**. `_plot_ordered_spline_panel` (`main_effects.py:524-635`) never
references `ti.smooth_curve`; it invents a PCHIP through the level relativities at
integer positions (`543`, `587-588`). The plotly panel draws the genuine fit from
`smooth_curve` (`main_effects_plotly.py:1161-1173`). `_collapsed_smooth_curve`
(`group_display.py:162-183`) pushes the same integer-position PCHIP into plotly
through the collapsed display, which is the **auto default** for OC
(`group_display.py:186-190`).

A separate, prior commit switches the matplotlib panel to `ti.smooth_curve` and
removes the collapsed-display override. It changes every existing OC matplotlib
figure and breaks tests asserting PCHIP-through-levels, so it is committed alone,
attributable and revertable, before any specials work.

### Specials rendering

The fitted curve spans **ordered levels only**. Specials render as detached
points with error bars, set off from the curve on the x-axis, with their own
tick labels and exposure bars.

`SmoothCurve.level_x` stays ordered-only, with special positions carried in a
separate array. Two failure modes are avoided by this: a special absent from
`_level_to_value` would `KeyError` in `_term_ops.py:226` before any figure exists;
and a `level_x` left at length K while `ti.levels` is K+S makes plotly render
`min(len(x), len(y))`, **silently dropping** specials from markers
(`main_effects_plotly.py:1132-1133`), exposure bars (`1362-1363`) and tick labels
(`740-741`), while shifting hover `customdata` (`1365`) off by the offset. A
figure that quietly omits the level you added is harder to catch than a visibly
bent curve.

## Editor

Specials terms are **fully editable**.

`_apply_projected_term` (`apply.py:178-192`) currently builds
`B = spec.transform(x_values)` over the full level list; with today's
`transform`, specials map to NaN through `_map_to_numeric`
(`ordered_categorical.py:337-341`), `np.linalg.lstsq` returns all-NaN
coefficients, and those are written into the term block **and** the model
intercept (`_adjust_intercept`, `apply.py:320-327`). Widening `transform` removes
the NaN path, but leaves the solve min-norm over the special columns.

**Convention:** a special's coefficient is set **exactly** from its edited
effect; only the ordered levels go through the least-squares projection onto the
spline basis. Each special contributes exactly one row, so its effect determines
its coefficient outright — there is nothing to project and no min-norm ambiguity.

Collapse allows grouping ordered levels on a term that has specials, and refuses
any group mixing a special with ordered levels.

`_ordered_spec_with_grouping` (`collapse.py:347-388`) rebuilds the spec from an
explicit argument list and must learn `specials=`, or every collapse edit
silently turns free levels back into smoothed ones.

## Screening and interactions

Interactions involving a specials term are **out of scope**. The screening path
resolves an OC column to numeric scores, and a special has none, so support means
composite margins in PSST — a separate piece of work that changes nothing about
what a special means.

**Order matters, and both halves land in the same commit.** `_margin_kind`
(`screening_ops.py:245-263`) returns `"spline"` for any spline-mode OC, so a
specials term enters the automatic sweep and reaches `_margin_source` (`661-682`),
which calls `resolve_interaction_parent` unguarded at line `673` inside an eager
pre-read loop (`703-712`). Adding the `NotImplementedError` to
`resolve_interaction_parent` (`ordered_categorical.py:577-583`) *without first*
changing `_margin_kind` aborts the entire sweep before a single statistic is
computed.

`_margin_kind` returning `None` for a specials term buys both behaviours:
silent exclusion from the automatic sweep (`283-294`), and the purpose-built
`ValueError` when the term is explicitly named as a candidate (`656-671`).

Deferral is **reported**, not silent, via `table.attrs["deferred_features"]`
mapping feature name to reason — machine-readable, following the
`attrs["phi"]` precedent (`screening_ops.py:1354`). The mechanism records
whatever screening deferred, so `Polynomial` and step-mode OC — silently skipped
today, the same defect — are covered by the same code.

A specials OC as a `FactorSmooth` group column is **legal**: the OC main effect
handles the specials, and the FactorSmooth needs only a group identity.
`resolve_interaction_parent_of` (`ordered_categorical.py:596-611`) already passes
it through untouched by design. Recorded here because no resolver guard reaches
it, so the absence of a rule would otherwise be an accident.

## Out of scope

- Interactions and PSST screening support for specials terms (composite margins).
- NaN as a special. Specials are literal labels present in the column; the strict
  NaN-rejection contract is untouched.
- A marker column on the Excel rating sheet.
- Making `_map_to_numeric` strict. It is the common root of several
  silent-wrongness paths and deserves its own commit and test sweep, not to be
  folded in here.
- Rating-sheet `centering=` for OC, which is already a no-op (`_term_ops.py:285`,
  `330` return without `_recenter_term`). Pre-existing; noted, not fixed.

## Risks

Six sites degrade silently rather than failing, and will hide an incomplete
implementation. Each needs a test that asserts the *content*, not merely that the
call succeeded:

| Site | Silent failure |
|---|---|
| `coef_tables.py:445` | `i < len(se_levels) else None` → blank std-err on free levels |
| `summary_levels.py:222-232` | unmatched rows preserved but misplaced and unformatted |
| `plotting/data.py:126-130` | drops `x_position` for the whole term |
| `plotting/data.py:266-268` | width mismatch makes the basis-decomposition panel vanish |
| `_term_helpers.py:236-258` | rebuilds `TermInference` by hand-listing 19 fields; a new field vanishes |
| `collapse.py:367-372, 382-387` | rebuilds the spec by hand; a new constructor argument vanishes |

Two further notes carried from the probe:

- **SSP conditioning.** `dm_builder.py:123-134` normalises the Gram by the weight
  sum over all `n` rows while only ordered rows contribute, so the normalisation
  is off by `ordered_exposure / total_exposure` and the fixed `1e-8·I` ridge
  becomes relatively larger exactly when specials carry material exposure. Not a
  correctness bug — REML re-estimates λ — but it needs a conditioning test with a
  high-exposure special, and preferably the ordered-row weight sum passed through.
- **Intercept meaning shifts.** Restricting the identifiability constraint to
  ordered rows means the reported intercept changes when a special is added to or
  removed from an existing model. This is editor-visible and is documented rather
  than hidden.

## Testing

Following the existing flat `tests/test_ordered_categorical*.py` convention:

- **Construction** — two `GroupInfo`s in the documented order, penalty and
  `penalized` flags, zero-fill correctness, full rank of the assembled design.
- **Invariance** — the fitted curve over ordered levels is unchanged by adding a
  special, to within solver tolerance. This is the spike's finding turned into a
  regression test.
- **Validation** — each rule in the table above, asserting the message.
- **Read-back** — `reconstruct`/`score`/`transform` widths and the split, plus a
  test that trips each of the four width guards if the block order is reversed.
- **Reporting** — the whole-smooth p-value and edf are unchanged by adding a
  special (this is the `coef_tables.py:340` regression), the `fit` column in both
  ASCII and HTML, and the rating-table export containing the special's row.
- **Plotting** — the curve spans ordered levels only; specials appear in markers,
  exposure bars, tick labels and hover data in *both* backends.
- **Editor** — a specials term round-trips through edit and through collapse of
  ordered levels; a mixed group is refused; the spec clone preserves `specials=`.
- **Screening** — a specials term is excluded from the automatic sweep without
  aborting it, is recorded in `attrs["deferred_features"]`, and raises when named
  explicitly.

Every test must assert what the fix *changes*, not merely observe that the call
returned.

## Implementation order

1. Plot-backend convergence (prerequisite, standalone commit).
2. `specials=` constructor, normalisation and validation.
3. Two-block `build()`, ordered-row spline, block-order contract and readers.
4. `transform` widening and the in-term coefficient split.
5. `coef_tables` restructure: smooth row filtered to the spline block.
6. Level marker through summary ASCII + HTML, and the Summary sheet.
7. Plot rendering of specials in both backends.
8. Screening deferral and the interaction refusal, in one commit.
9. Editor: exact-assignment refit convention, collapse rules, spec clone.
