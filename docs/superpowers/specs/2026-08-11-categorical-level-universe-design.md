# Bound level universes for the categorical family

- **Date:** 2026-08-11
- **Status:** approved by Max (conversation, 2026-08-11); awaiting implementation plan
- **Branch:** `categorical-level-universe` (from `origin/master` @ 5243302, post-0.25.0)

## 1. Problem

The categorical family conflates "levels observed in this data" with "levels
that exist". Four concrete defects follow:

1. **CV folds invent their own universes.** `cross_validate`
   (`model_selection.py`) clones the model per fold and each clone re-infers
   levels via `pd.factorize(sort=True)` on the fold's training slice. A level
   present only in a test fold raises
   `ValueError("Encountered unseen categorical levels at predict time: ...")`,
   and under the default `error_score` the fold is warning-logged and
   NaN-scored — silent fold loss on books where thin levels are structural.
2. **The standard escape hatch is silently broken.** The frame adapter
   (`_frame.py`, `EagerFrame._extract_column`) flattens columns with
   `np.asarray`, which discards `pd.CategoricalDtype` categories. A user who
   declares the full universe on the dtype — the pandas/R-orthodox recipe —
   gets no benefit and no warning.
3. **`specials=` fails in the inverse direction.** A declared special level
   with zero (or zero-weight) training rows is a hard error at build
   (`ordered_categorical.py`), so a thin special kills a CV fold even though
   the level is declared.
4. **`base="most_exposed"` resolves per fit.** Coefficient identity reshuffles
   across folds. (For an unpenalized `Categorical`, predictions are invariant
   to base choice; the damage is to comparability and reporting.)

A latent hazard rides along: if the predict-time unseen guard were bypassed,
`pd.Categorical(x, categories=self._levels).codes` yields `-1` for unknown
values and `level_effects[codes]` negative-indexes the last level — silent
wraparound misindexing.

## 2. Decisions

Made with Max on 2026-08-11:

- **The universe is the feature.** Knowing every level up front is the fix;
  imputation-style workarounds are rejected.
- **Universe sources: three, one mechanism**, resolved per term with fixed
  precedence (§3.1). The features dict stays the single source of truth — no
  model-level `levels=` mapping, no standalone vocabulary/transformer class,
  no pre-one-hot-encoded input. The term is the transformer: `build()` is
  fit-transform, `score()` is transform, state lives on the spec.
- **A declared level with zero training rows pins to base** (warn + record),
  not an error, not a rank-deficient all-zero column.
- **Out-of-universe at predict is policy-controlled:** `unseen="error"`
  (default) or `unseen="base"` (opt-in). No data-derived universe is complete
  against production, so the policy is the necessary complement to the
  universe mechanism, not a band-aid.
- **Scope: the whole categorical family** — `Categorical` and its derived
  interaction terms, `OrderedCategorical` including `specials=`,
  `LevelGrouping`; `RandomEffect`/`FactorSmooth` gain the same universe
  sources and keep their existing `unseen` policies.

## 3. Design

Three rules. Every edge case reduces to one of them.

### 3.1 Rule 1 — every term gets a bound universe

Precedence per term: **explicit `levels=` > column dtype > full-frame CV bind
> per-fit inference** (the last is the status quo and remains the fallback for
a plain `fit` on a raw object column).

**Explicit channel.** `Categorical`, `RandomEffect`, and `FactorSmooth` gain a
`levels=` constructor argument accepting exactly three source shapes:

| Source | Universe becomes |
|---|---|
| `list` / `tuple` of labels | exactly those labels, in the given order (`base="first"` = first declared) |
| `pd.Series` / numpy array / polars `Series` | object/string data: sorted observed uniques; categorical-dtype data (`pd.Categorical`, polars `Enum`): the dtype's declared categories in dtype order — a superset of the observed values by construction, so declared-but-unobserved levels are captured |
| `pd.CategoricalDtype` | its `.categories`, in dtype order |

Resolution happens inside `__init__`: the source is consumed immediately into
a plain `list` and the data reference dropped. Specs stay cheap to deep-copy
through `ModelConfig`; the resolved list is the reviewable artifact.
Constraints: labels must be unique after resolution; NaN/None in a source is a
hard error (a level cannot be missing); an empty universe or a singleton
universe fails the existing `>= 2 levels` check at build.

`OrderedCategorical` keeps `order=` / `values=` as its declaration channel —
it already declares; it does not gain `levels=`.

**Dtype channel.** The frame adapter surfaces dtype-declared categories
(pandas `CategoricalDtype`; polars `Enum`) alongside the extracted values. A
term with no explicit `levels=` adopts them. This is what makes the
type-the-column-once recipe work and what protects the `features=None`
auto-detect path.

**Full-frame CV bind.** See §3.5.

**Fit-time contract (sklearn's):** training rows outside a bound universe are
a hard error at build — the user declared the world; data exceeding it is a
data bug, never silently grouped or dropped.

### 3.2 Rule 2 — in-universe level with no training data → pin to base

For unpenalized dummy structures (`Categorical` non-base levels,
`OrderedCategorical` specials):

- No design column is emitted for the empty level (no all-zero columns, no
  rank-deficiency roulette).
- Its integer codes route to the kernel's existing `-1` sink bin
  (`CategoricalGroupMatrix` already treats `-1` as zero-contribution; no
  kernel change).
- One warning per fit names the pinned levels and the term.
- The pin is recorded on the fitted term; `summary`/reporting shows the level
  as "pinned to base: no data in this fit" (§3.8). The level keeps its
  identity — it is *known*, merely unobserved here.

"No training data" means zero rows **or** rows only with zero weight (matches
the existing zero-weight special detection in `ordered_categorical.py`).

Penalized structures need no pin: an empty level in a `RandomEffect` /
`FactorSmooth` / `OrderedCategorical` smooth position shrinks to the
population/neighbour value through its penalty, exactly as an empty region of
a numeric spline is bridged. Consequently
`OrderedCategorical._require_two_smooth_levels` counts **declared** smooth
levels, not observed ones.

**Empty declared base.** If an explicit `base=` level has no (effective)
training rows, intercept and dummy-sum become collinear. The fit falls back
deterministically — to the most-exposed observed level by weight, or to the
first observed level in universe order when unweighted (matching the existing
`most_exposed`-without-weights demotion) — with a loud warning, and the swap
is recorded on the term. Predictions are invariant to this swap for
unpenalized terms.

### 3.3 Rule 3 — out-of-universe at predict → policy

New constructor argument on `Categorical` (and mirrored on the derived
terms via inheritance of the parent's state): `unseen="error" | "base"`,
default `"error"`.

- `"error"`: today's behavior and message, unchanged.
- `"base"`: route the offending rows' codes to the `-1` sink → they predict
  as the base level. One warning per predict call names the novel levels and
  the count of affected rows. Never silent.

The parameter is an enum so later policies (`"nan"`, an `"other"` bucket
composed with `LevelGrouping`) can be added without an API break; they are
explicitly **out of scope** now (§4).

Independent of policy, the `-1` wraparound at the predict mapping
(`categorical.py`, `level_effects[codes]`) gets an explicit guard so negative
codes can never fancy-index.

`RandomEffect`/`FactorSmooth` keep their existing
`unseen="population" | "error"` vocabulary untouched — `"population"` is the
penalized analogue of `"base"`.

### 3.4 Frame boundary

`EagerFrame._extract_column` (and the polars path) returns dtype-declared
categories alongside values when the column dtype carries them, instead of
flattening through `np.asarray` and losing them. Consumers that ignore the
extra information see current behavior; the categorical family consumes it as
the dtype universe source. This fixes defect §1.2 for every term at once.

### 3.5 `cross_validate` binds on the full frame, pre-split

Before splitting, `cross_validate`:

1. Resolves the universe for every cat-family term that lacks one (no
   `levels=`, no dtype categories) by inferring from the **full** frame's
   column.
2. Resolves `base="most_exposed"` once, on full-frame exposure.
3. Stamps both onto each fold's cloned spec at materialize time (via the
   existing `ModelConfig` override path used by `FitWorkspace.start`), so
   every fold shares one universe and one base identity.

Statistical note, stated here deliberately: sharing the level *set* across
folds is R factor semantics — the vocabulary is a property of the data column,
not of the training subset. No target information crosses folds; this is not
leakage. Per-fold quantities that legitimately depend on training rows (knots,
penalty scaling, coefficients) continue to bind per fold.

Docs recommendation (not enforcement): governance models declare `levels=`
and `base=` explicitly, making the term fully data-independent.

### 3.6 Family propagation

- **`SplineCategorical`, `CategoricalInteraction`, `NumericCategorical`,
  `PolynomialCategorical`:** copy the parent `Categorical`'s `_non_base` /
  `_base_level` at build and therefore inherit bound universes, zero-count
  pinning, and the `unseen` policy for free. Their build-time validation
  changes from "observed set" to "bound universe".
- **`OrderedCategorical` / `specials=`:** the declared-special-with-no-rows
  hard errors (zero rows; zero-weight rows) are replaced by Rule 2: indicator
  dropped, contribution pinned to zero, warned, recorded. The smooth-level
  minimum counts declared levels (§3.2). Out-of-universe data at build/predict
  keeps erroring per Rule 1/Rule 3.
- **`LevelGrouping` / `collapse_levels`:** `levels=` on the owning term
  declares the **raw** (pre-collapse) universe. The grouping must cover every
  declared level, else build errors — no silent identity fallthrough for
  declared-but-unmapped levels. `collapse_levels(data, ...)` fed the full
  column already yields a covering grouping.
- **`RandomEffect` / `FactorSmooth`:** gain `levels=` and the dtype/full-frame
  sources; empty levels shrink via the penalty; existing `unseen` policies
  unchanged.

### 3.7 Errors and warnings

All messages name the term, the offending levels (sorted with `key=str`, so
mixed int/str labels cannot crash the error path), and the fix.

New warnings (all `UserWarning`, once per fit or predict call, not per row):

| Event | When |
|---|---|
| Zero-count pin | bound level(s) with no effective training rows pinned to base |
| Base fallback | explicit `base=` empty in this fit; fallback level named |
| Specials pin | declared special(s) with no effective rows pinned to zero contribution |
| Unseen routed | `unseen="base"` routed novel level(s) at predict; levels + row count |

New errors:

| Event | When |
|---|---|
| Universe exceeded at fit | training rows outside a bound universe |
| Bad `levels=` source | NaN/None in source; duplicate labels; multi-feature encoder guidance (see §4); unsupported type |
| Grouping coverage | declared level not covered by the grouping |

Unchanged: predict-time unseen error under `unseen="error"` (same message);
NaN/None in the data column is still a hard error everywhere (missing-value
routing for categoricals remains a separate, unshipped idea).

### 3.8 Reporting / governance

The fitted model exposes, per categorical-family term: the bound universe, its
**source** (`declared` / `dtype` / `full-frame` / `inferred`), the resolved
base and whether it was fallback-swapped, and any pinned levels. This is the
audit trail that makes the mechanism defensible in a rating-governance
setting.

## 4. Explicitly rejected

- **Model-level `levels={col: [...]}` mapping** — parallel declaration surface
  competing with the features dict (Max, 2026-08-11).
- **Standalone vocabulary/transformer class** (`LevelUniverse`,
  sklearn-style encoder) — one piece of state wrapped in a second stateful
  object to fit/persist/version; the spec already persists through fit state.
- **Consuming fitted encoder objects or `get_dummies` output.** An encoder's
  vocabulary is already `ohe.categories_[0]` — a plain array the
  Series/array source accepts; the docs show that recipe. Parsing dummy
  column names is the brittle hack this design exists to avoid. The error for
  an accidentally passed encoder object says exactly this.
- **Pre-one-hot-encoded input** — bypasses the code-path kernel and every
  term-level semantic (base, exposure, grouping, specials).
- **`unseen="nan"` / `unseen="other"` policies** — deferred; enum leaves room.
- **Making `most_exposed` deterministic outside `cross_validate`** — plain
  `fit` keeps per-fit resolution; declare `base=` if identity must be pinned.

## 5. Compatibility

- Plain `fit` on raw object columns with no `levels=`, no categorical dtype:
  behavior unchanged (per-fit inference, same unseen error at predict).
- `cross_validate` results **change where folds previously failed**: folds
  that NaN-scored (or raised) on unseen levels now complete. That is the fix;
  the changelog entry says so explicitly.
- Columns that are already `pd.Categorical` **change behavior**: declared
  categories now count as levels (previously stripped). Models fit on such
  columns may gain pinned levels and emit the new warning. Documented as a
  bug fix (defect §1.2).
- `specials=` thin-fold hard errors become warnings + pins (§3.6).
- No version bump in the feature PR (release convention: bump at release).

## 6. Testing

Contract tests to keep green: predict-time unseen error under default policy
(`test_theory_invariants.py::TestPredictionTimeContracts`), NaN hard errors,
existing `RandomEffect`/`FactorSmooth` unseen-policy suites.

New coverage, one fixture per real shape (not just clean strings — mixed-type
labels, weighted rows with zero-exposure levels, a thin-level book shape):

1. Declared-but-unobserved level: fit succeeds, no column, warning, predicts
   base, reporting shows the pin, coefficients unchanged vs. the
   without-that-level fit.
2. `pd.CategoricalDtype` categories survive the frame boundary (regression on
   defect §1.2) — for pandas and polars `Enum`.
3. `cross_validate` with a pigeonhole-rare level: completes, no NaN folds,
   identical universe and base in every fold's fitted specs.
4. `levels=` source shapes: list order preserved (`base="first"` semantics);
   Series → sorted uniques ∪ dtype categories; `CategoricalDtype` order; NaN
   in source errors; duplicate labels error; encoder object errors with the
   `categories_[0]` guidance.
5. Fit-time universe-exceeded error.
6. Empty-declared-base fallback: deterministic choice, warning, predictions
   invariant (unpenalized), swap recorded.
7. Specials pin: thin special completes with warning; predictions for the
   pinned special equal zero contribution; genuinely unseen level still
   errors.
8. `unseen="base"`: routed rows predict base, warning names levels and count;
   wraparound guard: negative codes can never index (unit test at the mapping
   layer).
9. Grouping coverage error; grouping + declared raw universe round-trip.
10. Interaction terms inherit the parent's universe and pins (one test per
    derived term).
11. Equivalence control: on a frame with every level observed, all sources
    (declared / dtype / inferred) produce byte-identical designs and
    identical fits — with the declared list and dtype categories given in
    sorted order, so all three resolve to the same level ordering.

## 7. Prior art

- **R factor semantics** (public documentation: `?factor`, `?model.frame`):
  levels are an attribute of the data — "the unique values that x *might*
  have taken"; fitted models snapshot `xlevels` and re-impose them at
  predict, erroring on new levels. The design's universe/data separation and
  Rule 3 default mirror this.
- **scikit-learn encoders** (BSD-3): the two-knob pattern — `categories=`
  (vocabulary source) × `handle_unknown` (OOV policy) — and the fit-time
  contract that declared-but-unobserved categories are legal while
  fit-data-exceeding-declaration errors. Also documents the
  all-zeros-aliases-the-dropped-category trap that motivates loud naming of
  `unseen="base"`.
- **patsy / formulaic** (BSD-2 / MIT): stateful-transform architecture —
  vocabulary learned at build, stored on the spec/design-info, replayed on
  new data; patsy errors on OOV, formulaic warns-and-NaNs, evidence the two
  concerns are separate parameters.
- **pandas `CategoricalDtype`** (BSD-3): dtype-as-carrier; survives slicing
  (what makes fold slices safe); silent OOV→NaN at `astype` is the behavior
  we deliberately do *not* copy (Rule 3 is loud).
- In-house precedents: `RandomEffect(unseen=)` / `FactorSmooth(unseen=)`
  (policy shape), `OrderedCategorical(order=)` (declaration shape), the
  kernel's `-1` sink bin (mechanism).

## 8. Implementation anchors

For the planner; discovered 2026-08-11, verify at implementation time.

- Level inference: `src/superglm/features/categorical.py` build path
  (`pd.factorize` around `:162`), base selection `:174-192`, code remap
  `:194-208`, predict mapping `:219-233`, unseen validation `:17-37` /
  `:40-73` / `:86-123`.
- Frame boundary: `src/superglm/_frame.py` `_extract_column` `:126-130`,
  dtype routing `:90-118`.
- Kernel sink: `src/superglm/_group_matrix/_group_matrix_core.py:86-134`.
- Lifecycle: `src/superglm/model/fit_state.py` (`ModelConfig.capture`
  `:122`, `materialize` `:199`), `src/superglm/model/fit_workspace.py:79`.
- CV: `src/superglm/model_selection.py` (fold loop `:355`, clone `:110`,
  fold scoring/`error_score` `:418-437`).
- Specials: `src/superglm/features/ordered_categorical.py` (domain assembly
  `:645-655`, zero-row/zero-weight errors `:868-889`, two-smooth-levels
  `:194-201`).
- Derived terms: `src/superglm/features/interaction.py` (parent copy at
  `:315-316`, `:730-732`, `:803-806`).
- Screening guard to keep consistent: `src/superglm/model/screening_ops.py`
  `:390-396`.
