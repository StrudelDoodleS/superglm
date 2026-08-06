# Piecewise Variate Prototype — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> or `superpowers:executing-plans` to work this plan stage-by-stage. Steps use checkbox
> (`- [ ]`) syntax. **Checkbox state in a shipped plan lies** — read the appended stage
> narrative, never the tick count.

**Goal:** ship `Piecewise(breaks=[...])`, a continuous piecewise-linear feature whose
coefficients are log-relativities at stated breakpoints, reporting per-knot Wald rows and a
fixed `edf = J+1`, exporting to a rating table that reproduces `model.predict` to ~1e-12,
and editable one-handle-per-knot.

**Architecture:** degree-1 B-spline (hat) basis on the knot vector
`t_0 = lower, t_1..t_J = breaks, t_{J+1} = upper`, with the hat at a base knot `t_r`
dropped for identifiability. Group size `J+1`. Each retained coefficient is exactly
`v_j = f(t_j) − f(t_r)`, which makes the summary row, the editor handle and the workbook
cell the same number. Nothing else in the design matters as much as that one property, and
every stage below exists to keep it true at one more surface.

**Tech Stack:** Python 3.13, NumPy, pandas, openpyxl, pytest, `uv`, ruff.

**Spec:** `/home/max/2026-08-06-piecewise-variate-design.md`, implemented **AS AMENDED** by
the 2026-08-06 review amendments (reproduced inline below where they bite). Where spec and
amendment differ, the amendment wins. Where this plan differs from both, the deviation is
called out under **DEVIATION** and the implementer must not silently revert it.

---

## Global constraints

- Work only inside `/home/max/projects/superglm/.worktrees/piecewise-prototype` on branch
  `piecewise-prototype`. Never push, never touch master, never modify files outside the
  worktree.
- **Never run `git stash`.** The stash stack is shared across every worktree of this repo
  and other agents pop it.
- Environment: `uv sync --python 3.13 --extra dev` (already done once). **Before trusting
  any test result**, run
  `uv run python -c "import superglm; print(superglm.__file__)"` and confirm the path is
  inside this worktree. Verified at plan time: resolves to
  `/home/max/projects/superglm/.worktrees/piecewise-prototype/src/superglm/__init__.py`,
  version `0.19.0`.
- Standard checks, all three green before any commit:
  - `uv run pytest tests/ -q -m "not slow"`
  - `uv run ruff check src/ tests/`
  - `uv run ruff format --check src/ tests/`
  - Deterministic full run when a stage touches shared code:
    `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest tests/ -q -m "not slow" -p no:randomly`
- **Version stays `0.19.0`. Do not bump, tag, publish, or merge.** This is a prototype
  branch; the release decision is the user's and is not part of this plan.
- `docs/superpowers/` is gitignored — commit plan docs with `git add -f` and confirm the
  file appears in `git log --stat`.
- One commit per completed stage, message style matching `git log --oneline -10` (short
  imperative sentence, no scope prefix), body ending with:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  ```
- **Never cite R package source** (mgcv/scam file:line or internal identifiers) in code,
  comments, or docs. Naming mgcv in prose is fine.

### Recorded baseline (measured at plan time, clean worktree at `origin/master`)

```
1 failed, 6322 passed, 195 skipped, 137 deselected  (218s)
FAILED tests/test_structured_screening.py::test_a_thin_level_does_not_cost_the_pair_a_degree_of_freedom[1.0]
  Obtained: 207.9999554200152   Expected: 207.9999666796071 ± 1.0e-05
```

**That failure is PRE-EXISTING and deterministic** — it reproduces in isolation on a clean
tree with no Piecewise code present. It is a numerics tolerance miss in the 5th decimal,
unrelated to this feature. Do not chase it, do not "fix" it as part of this work, and do not
let it mask a real regression: the bar for every stage is **exactly this failure and no
other**, with the pass count at or above 6322 plus the stage's new tests.

### Sabotage-test discipline (applies to every stage)

Every **new** test must be verified by temporarily breaking the behaviour it pins, running
just that test, confirming it **FAILS**, restoring, and confirming it passes. Record each
one in the stage's narrative as a line:

```
test_name :: sabotage applied :: observed failure
```

A green test with no recorded sabotage evidence is treated as measuring nothing. This
repository has produced green suites that asserted nothing on three separate occasions,
twice because the test was aimed at a branch that was not the one under change. Two
specific traps here:

- Several validation rules **overlap** (the zero-hat-mass rule and the rank check both fire
  on the degenerate `breaks=[2]`, `x ∈ {1,3}` case). Sabotaging one rule will not fail a
  test that the other rule also catches. Each validation test must name the rule it pins
  and its sabotage must be aimed at *that* rule.
- The export exactness test compares against `model.predict`. Sabotaging the *basis* would
  move both sides. Sabotage the **workbook write path** (e.g. round the Log relativity
  column to 4 dp), which moves only the reconstruction.

---

## Verified against this worktree at plan time (do not re-derive)

These were checked by reading the code on `piecewise-prototype` @ `origin/master`. They are
inputs to the plan, not assumptions.

| Claim | Status | Evidence |
|---|---|---|
| `dm_builder` needs no change | **TRUE** | `_spec_kind` default branch returns `type(spec).__name__` (`dm_builder.py:339`); `should_discretize` returns False for non-`_SplineBase` (`:150`); main effects build polymorphically via `spec.build(x_col, sample_weight=...)` (`:997`). |
| Prediction needs no change | **TRUE** | `model/base.py:252 _score_feature` calls `spec.score(...)` when present, else `spec.transform(...) @ beta`. |
| PSST screening already defers unknown types | **TRUE** | `model/screening_ops.py:246 _margin_kind` returns None; `:275 _deferral_reason` falls through to `f"{type(spec).__name__} margins are deferred: no screenable margin"`. |
| `editor/controls.py` duck-types on `_raw_basis_matrix` | **TRUE** | `controls.py:86`. But see the next two rows — **the module itself is fine; its callers gate on `term_type == "spline"`.** |
| Editor session refuses non-spline control handles | **BLOCKER, spec §7 and amendment 2 both miss it** | `editor/session.py:1223 _require_control_term` raises `TypeError` unless `term_type == "spline"`. |
| Editor control payload refuses non-spline | **BLOCKER, same** | `editor/payloads.py:252 _controls_payload` returns `None` unless `term_type == "spline"`. |
| `term_inference` raises on an unknown spec | **TRUE — must be extended** | `inference/_term_ops.py` final `else: raise TypeError(f"Unknown feature type: ...")`. |
| `build_coef_rows` has a silent-wrong fallback | **TRUE — must be extended** | `inference/coef_tables.py` final `else:` branch emits **one** row from `b_g[0]`. A `J+1`-column Piecewise term would report a single coefficient and drop the rest without erroring. |
| Plotly main-effects would break on a new kind | **TRUE** | `plotting/main_effects_plotly.py:629` handles `("spline","polynomial")`, `:674` `"numeric"`, then falls through to `project_grouped_term_for_display`, which needs `ti.levels`. |
| `TermInference.to_dataframe` would mis-shape | **TRUE** | `_term_types.py` dispatches on `kind in ("spline","polynomial")`, else categorical, else the single-row numeric shape. |
| `_continuous_features` cannot simply "admit" Piecewise | **TRUE — amendment 7 needs interpreting** | `rating_tables.py:97` feeds `_continuous_features` into `model.discretization_impact(features=...)`, and `diagnostics/discretize.py:202 _is_continuous_feature` **raises** `ValueError` for anything that is not `_SplineBase | Polynomial`. Worse, `build_rating_table_payload` routes any name in `selected.tables` to the **binned** `_continuous_block` — the exact lossy path this feature exists to avoid. See stage 2, task 2.3. |
| Excel main-effect blocks are laid out on a fixed 3-column stride | **TRUE — constrains the block width** | `export/excel.py:176` `start_col = 1 + idx * 3`, and `:186-189` applies number formats globally by `cell.column % 3`. A 4-column block would overwrite its neighbour and mis-format. See stage 2, task 2.4. |
| `GroupInfo.penalized` defaults True | **TRUE** | `types.py`; `Polynomial`, `Numeric` and `Categorical` all leave it True, so uniform group-lasso selection applies. Piecewise matches them (amendment 3). |
| `weighted_quantile_knots` dedups | **TRUE** | `features/_spline_knots.py:103`, returns `np.unique(...)`. |
| `Categorical` falls back to the first level when `base="most_exposed"` and no weights | **TRUE** | `categorical.py:183-184`. Mirror it (amendment 5). |

---

## File structure

| File | Responsibility | Stage |
|---|---|---|
| `src/superglm/features/piecewise.py` | **new** — `Piecewise` class, hat basis, validation, `build/transform/score/reconstruct`, `_raw_basis_matrix` | 1 |
| `src/superglm/__init__.py` | export `Piecewise` (import + `__all__`) | 1 |
| `tests/_piecewise_cases.py` | **new** — the shared fixture matrix (amendment 9) | 1 |
| `tests/test_piecewise.py` | **new** — basis, every validation rule, §9 properties 2 and 4 | 1 |
| `src/superglm/inference/_term_ops.py` | `Piecewise` branch in `term_inference` | 2 |
| `src/superglm/inference/_term_covariance.py` | `Piecewise` branch in `feature_se_from_cov` | 2 |
| `src/superglm/inference/_term_types.py` | `to_dataframe` admits `"piecewise"` | 2 |
| `src/superglm/inference/coef_tables.py` | per-knot Wald rows + whole-term chi-square; **the four-condition comment lives here** | 2 |
| `src/superglm/export/rating_tables.py` | `_piecewise_block`, main-effects routing | 2 |
| `src/superglm/export/excel.py` | piecewise number formats + interpolation/boundary-slope note | 2 |
| `tests/test_piecewise_reporting.py` | **new** — inference rows, edf, §9 property 1 exactness | 2 |
| `tests/test_rating_table_export.py` | characterization test pinning existing block layout | 2 |
| `src/superglm/editor/apply.py` | `_apply_piecewise_term` | 3 |
| `src/superglm/editor/terms.py` | `term_type_from_spec` returns `"piecewise"` | 3 |
| `src/superglm/editor/session.py` | `_require_control_term` admits `"piecewise"` | 3 |
| `src/superglm/editor/payloads.py` | `_controls_payload` admits `"piecewise"` | 3 |
| `src/superglm/editor/controls.py` | opt-in full-handle default for piecewise specs | 3 |
| `src/superglm/plotting/main_effects_plotly.py` | one-word: `"piecewise"` joins the continuous branch | 3 |
| `src/superglm/model/screening_ops.py` | bespoke Piecewise deferral reason | 3 |
| `tests/test_piecewise_editor.py` | **new** — §9 property 3, re-base, fixture matrix sweep | 3 |

---

# Stage 1 — core feature

**Deliverable:** `Piecewise` builds, transforms, scores, reconstructs, validates, and is
importable from `superglm`. Spec §9 properties **2** (coefficient meaning) and **4** (linear
extrapolation) hold, and every validation rule has a test that fails when that rule alone is
disabled.

### Task 1.1 — `src/superglm/features/piecewise.py`

**Class shape** (mirror `features/numeric.py` and `features/polynomial.py` for the
`build/transform/score/reconstruct` contract; mirror `features/categorical.py` for base
resolution):

```python
class Piecewise:
    def __init__(self, breaks, *, base="most_exposed", strategy="quantile",
                 lower=None, upper=None): ...
```

Fitted state, all set in `build()`:

| Attribute | Meaning |
|---|---|
| `self._knots` | `(J+2,)` float64, `[lower, *breaks, upper]`, strictly increasing |
| `self._base_index` | `int` r, index into `_knots` |
| `self._non_base_indices` | `(J+1,)` intp, `_knots` indices in ascending order excluding r |
| `self._strategy_actual` | `"explicit"` or `"quantile"` |
| `self._n_breaks_requested` | int-mode only; `None` for a sequence |

**Hat evaluation — `_hat_basis(x) -> (n, J+2)`, no clipping.** Do not call scipy. The whole
construction is:

```
seg = clip(searchsorted(t, x, side="right") - 1, 0, J)     # J+1 segments, 0..J
w   = (x - t[seg]) / (t[seg+1] - t[seg])                   # UNCLIPPED: w<0 below t_0, w>1 above t_{J+1}
H[i, seg[i]]     = 1 - w[i]
H[i, seg[i] + 1] = w[i]
```

**All four consequences below were measured at plan time** by running that exact formula on
`t = [0, 1, 4, 10]` (unequal widths) — they are observations, not predictions. If your
implementation disagrees with any of them, the implementation is wrong.

Consequences the tests will pin, so get them right:

- Rows sum to exactly 1 everywhere including the tails → partition of unity holds under
  extrapolation, which is what makes the drop-a-knot algebra survive outside `[t_0, t_{J+1}]`.
- `_hat_basis(self._knots)` is the **identity** matrix. This is load-bearing for stage 3:
  it is what makes the editor's least-squares recovery return the coefficients exactly and
  place handles exactly on knots.
- Beyond the boundary the outer hat exceeds 1 and its neighbour goes negative. That is the
  linear tail, and it is why the zero-column test in task 1.2 must use `|h_j|` (see
  **DEVIATION 1**).

`_raw_basis_matrix(x)` returns `_hat_basis(x)` as a dense `(n, J+2)` float64 array — this is
the public duck-typed hook `editor/controls.py:86` looks for. Docstring must say it returns
**all** `J+2` raw columns, not the identifiable `J+1`.

`transform(x)` returns `_hat_basis(x)[:, self._non_base_indices]` → `(n, J+1)`.
`build(x, sample_weight=None)` resolves knots and base, runs validation, then returns
`GroupInfo(columns=transform(x), n_cols=J+1)` — no `penalty_matrix`, no `reparametrize`,
`penalized` left at its `True` default (amendment 3).
`score(x, beta)` returns `transform(x) @ beta`.
`reconstruct(beta)` returns
`{"knots", "base_knot", "base_index", "log_relativity", "relativity", "slopes", "boundary_slopes"}`
where `log_relativity` is the `J+2`-vector with `0.0` inserted at `base_index`,
`slopes[j] = (v[j+1] - v[j]) / (t[j+1] - t[j])` for `j = 0..J` (spec §4.1, derived, no
p-value), and `boundary_slopes = (slopes[0], slopes[-1])`.

Also add `__repr__` in the register of `Polynomial.__repr__`.

### Task 1.2 — validation (amendments 4 and 5)

All rules raise from `build()` with the offending values named. Ordered as they must run:

1. **`x` finite.** Any NaN or inf → `ValueError`. (Spec §3.1: a genuine missing band belongs
   in `OrderedCategorical(specials=[...])`, not here — say so in the message.)
2. **Empty `breaks`.** A zero-length sequence → `ValueError` pointing at `Numeric` (a
   piecewise term with no breaks *is* a straight line).
3. **`strategy` is an extension point.** Validate against a module-level
   `_STRATEGIES = frozenset({"quantile"})` and raise naming the set. **Do not add other
   strategies and do not change the default.** Rationale in a comment: the literature
   (Labovich 2025, arXiv:2505.12460) reports k-means binning beating equal-frequency by
   55%+ MSE on skewed data, *largest at small cut budgets* — exactly this feature's regime —
   but that is one single-author paper from histogram-GBT binning, and nothing located
   addresses **weighted** quantiles at all. So: reserve the seam, do not act on the
   evidence. `strategy` is only consulted when `breaks` is an int.
4. **Sequence mode:** `breaks` strictly increasing with no duplicates → `ValueError`. This
   is the user's own mistake, so it raises.
5. **Int mode:** place `breaks` breakpoints at exposure-weighted quantiles using
   `features/_spline_knots.py::weighted_quantile_knots(x, n, alpha=1.0, sample_weight=w)`,
   which already `np.unique`s. Then **drop any value equal to `lower` or `upper`**. If the
   realised `J` is below the requested count → `warnings.warn` naming **both** counts and
   stating that `edf` is therefore `J_realised + 1`. If the realised `J` is 0 → `ValueError`.
   Rationale comment: insurance rating variables are heaped (ages ending 0/5, whole-year
   tenures) so ties are the norm, not the exception; raising on a user who only typed
   `breaks=8` would be our mistake reported as theirs.
6. **`lower`/`upper`.** Default `min(x)`/`max(x)`. `lower < upper` or `ValueError`. Every
   break strictly inside `(lower, upper)` or `ValueError`. Pinning **narrower** than the data
   is allowed — the docstring must state that rows outside `[lower, upper]` load the linear
   tails and therefore dominate the boundary slopes.
7. **`base` resolves to exactly one knot.** `"first"` → index 0. `"most_exposed"` →
   `argmax_j Σ_i w_i · h_j(x_i)`; with `sample_weight=None` fall back to the **first** knot,
   mirroring `categorical.py:183-184`. A float must equal exactly one knot; zero matches or
   more than one → `ValueError` naming the knots. Reuse a fitted base across a re-`build()`
   the way `Categorical` does.
8. **Zero hat mass → ERROR.** For each of the `J+2` knots compute
   `colmass_j = Σ_i w_i · |h_j(x_i)|` and raise if any is exactly 0. Message names the knot
   and its value.
   > **DEVIATION 1 from amendment 4.** The amendment writes the condition as
   > `Σ_i w_i·h_j(x_i) == 0`. That is the true zero-column condition **only where the hats
   > are non-negative**. With `lower`/`upper` pinned narrower than the data (an explicitly
   > supported mode, amendment 5) the tail rows carry negative entries and a non-zero column
   > can sum to zero by cancellation, so the amendment's form is a false negative on exactly
   > the fixture matrix we are required to cover. `|h_j|` is the amendment's own stated
   > intent ("the true zero-column condition") expressed correctly. Keep the amendment's
   > wording in the docstring, use `|h_j|` in the code, and comment why.
9. **Per-segment positive weight → ERROR, justified as a data-support rule.** For each of
   the `J+1` segments, total the weight of rows whose `seg` index (the same index the basis
   uses, so tail rows count toward segment 0 / segment J) falls in it. Any segment with zero
   total → `ValueError` whose message **reports the weight of every segment**. Justify it in
   the comment as a data-support rule — a segment bracketing no data rates a distinction the
   data cannot support — **not** as an identifiability rule; positive weight per segment is
   not sufficient for identifiability, which is what rule 10 is for.
10. **Rank check → ERROR.** `matrix_rank(sqrt(w)[:, None] * transform(x))` must equal `J+1`;
    otherwise raise naming the rank and the deficiency. This is the rule that catches the
    genuinely degenerate configurations rules 8 and 9 miss.
11. **Small-weight WARNING.** Any segment whose weight is below
    `_SMALL_SEGMENT_WEIGHT_FRACTION = 0.005` of the total → `warnings.warn` reporting the
    weight of **every** segment. Keep the constant module-level and named so it is
    reviewable. This is the failure mode most likely to reach production silently, and it
    stays a warning because the conditioning of the *basis* is fine (the hat basis is
    provably well-conditioned uniformly over knot placements — Zhong 2026, arXiv:2606.12270)
    while the conditioning of `XᵀWX` is not. Do not conflate the two in the comment.

### Task 1.3 — `src/superglm/__init__.py`

Add `from superglm.features.piecewise import Piecewise` alongside the `Polynomial` import,
and `"Piecewise"` to `__all__` beside `"Polynomial"`. Nothing else.

### Task 1.4 — `tests/_piecewise_cases.py` (the fixture matrix)

A shared module the stage-2 and stage-3 suites import, so all three sweep the same shapes.
Amendment 9 requires **at minimum** these named cases; each returns
`(X: pd.DataFrame, y, sample_weight, spec: Piecewise)`:

| Case name | Shape it covers |
|---|---|
| `interior_base` | base at an interior knot |
| `end_base` | base at `t_0` (and a second variant at `t_{J+1}`) |
| `unequal_widths` | segments of visibly different widths |
| `pinned_wider` | `lower`/`upper` pinned **wider** than the data |
| `pinned_narrower` | `lower`/`upper` pinned **narrower** than the data (the tail-loading case; the one that breaks DEVIATION 1's naive form) |
| `heaped_int_x` | integer x heaped on multiples of 5 with `breaks=int` and tied quantiles |
| `many_knots` | `J + 2 > 12` knots, to exercise editor handle subsampling in stage 3 |

Prior incident to avoid: *"my own green suites keep measuring nothing — add a fixture per
real shape, and re-derive any case that flips BAD→OK."* If a case that should fail
validation starts passing, that is a finding, not a fixture to quietly adjust.

### Task 1.5 — `tests/test_piecewise.py`

Follow the `tests/test_polynomial.py` idiom (a `TestPiecewiseSpec` class of direct-spec
tests, then a model-level class).

**Basis and contract**
- `_hat_basis(knots)` is exactly the identity (`assert_array_equal`).
- Rows of `_hat_basis` sum to 1 at interior points **and** at two points below `t_0` and two
  above `t_{J+1}`.
- `build().n_cols == J + 1`; `columns.shape == (n, J+1)`; `penalty_matrix is None`;
  `reparametrize is False`; `penalized is True`.
- `transform(x)` equals `build(x).columns` on the same `x`.
- `score(x, beta)` equals `transform(x) @ beta`.
- `_raw_basis_matrix(x).shape[1] == J + 2` and its `_non_base_indices` columns equal
  `transform(x)`.

**§9 property 2 — coefficient meaning.** Fit a real model over the fixture matrix. For each
case assert `beta[k] == score(t_j) - score(t_r)` to `1e-12` for every non-base knot,
computed independently through `spec.score()`, and that `score(t_r) == 0`. Parametrize over
at least `interior_base`, both `end_base` variants and `unequal_widths`.

**§9 property 4 — linear extrapolation.** For `x` below `t_0` at **two** distinct points and
above `t_{J+1}` at **two** distinct points, assert the predictions lie on the boundary
segment's line: `score(x) == score(t_0) + slope_0 · (x - t_0)` to `1e-12`, and the same at
the top. Two points per side is the requirement — one point cannot distinguish a slope
error from an offset error.

**Validation — one test per rule, each naming its rule.** Rules 1-11 above. For the
overlapping pair, include the discriminating fixture:
- Rule 8 (zero mass) fires on `breaks=[2]` with data only at `{1, 3}` — *and so does rule
  10*. Assert on the rule-8 message.
- Rule 10 (rank) must have a case rules 8 and 9 do **not** catch. Construction
  `lower=0, breaks=[1, 2], upper=3` with `x ∈ {0, 1.5, 3}` and base index 0 was **measured
  at plan time** against the exact `_hat_basis` formula above: column masses
  `[1.0, 0.5, 0.5, 1.0]` so rule 8 stays silent; all three segments carry a row so rule 9
  stays silent; the `J+1 = 3` retained columns have rank **2**, so only rule 10 fires. Use
  it. If your implementation disagrees, the basis is wrong — do not adjust the fixture to
  make the test pass.
- Rule 5 warning: assert the realised `J`, the warning text naming both counts, and that
  `n_cols == J_realised + 1`.

**Sabotage plan for this stage** (each must be recorded):
- Basis: negate the `w` used for the upper tail → extrapolation tests fail.
- Identity: clip `w` to `[0, 1]` → tail tests and the pinned-narrower case fail.
- Coefficient meaning: drop the *last* hat instead of the base hat → property-2 tests fail.
- Each validation rule: comment out that rule's raise **only** and confirm its own test fails
  while the others still pass (this is what proves the tests are not all riding on one rule).

### Stage 1 acceptance criteria

- [ ] `uv run python -c "from superglm import Piecewise; print(Piecewise([1,2]))"` works.
- [ ] `uv run pytest tests/test_piecewise.py -q` green, and the full
      `uv run pytest tests/ -q -m "not slow"` shows the recorded baseline failure and no other.
- [ ] `uv run ruff check src/ tests/` and `uv run ruff format --check src/ tests/` clean.
- [ ] A fitted `SuperGLM` containing a Piecewise term completes `.fit()` and `.predict()`
      with no change to `dm_builder.py` (if a change turns out to be needed, that is a
      finding to report, not to make silently).
- [ ] Every new test has a recorded `test :: sabotage applied :: observed failure` line, and
      each validation-rule sabotage failed **only** its own test.
- [ ] Committed with `git add -f docs/superpowers/plans/2026-08-06-piecewise-prototype.md`,
      and the plan appears in `git log --stat`.

---

# Stage 2 — reporting and export

**Deliverable:** a Piecewise term reports per-knot Wald rows, a `J+1`-df whole-term
chi-square, and `edf = J+1`; it exports a `kind="piecewise"` rating-table block that a
consumer can reconstruct to `model.predict` at ~1e-12 by log-linear interpolation.

**Depends on:** stage 1 (`reconstruct` keys, `_knots`, `_base_index`, `_non_base_indices`).

### Task 2.1 — inference

`src/superglm/inference/_term_ops.py` — add an `elif isinstance(spec, Piecewise)` branch
before the final `else`, modelled on the `Categorical` branch:

- `kind="piecewise"`, `x=spec._knots`, `log_relativity` = the `J+2` vector with `0.0` at the
  base index, `relativity=_safe_exp(...)`, `absorbs_intercept=False`,
  `centering_mode="base_knot"`.
- `se`/`ci_lower`/`ci_upper` via `feature_se_from_cov` when `with_se and active`.
- `edf`: **use the measured `_compute_term_edf` value when it is not None, else `float(J+1)`.**
  > **DEVIATION 2 from spec §4.** §4 says "`edf` equals `J+1` exactly, not an estimated
  > trace." Reporting `J+1` unconditionally is a lie under group-lasso shrinkage
  > (`GroupInfo.penalized` is True — amendment 3). Reporting the measured value satisfies
  > §4's claim *by measurement* inside the contract's domain and stays honest outside it.
  > Stage 2 must include a test asserting the measured edf **equals** `J+1` to `1e-9` on an
  > unpenalized, unshrunk fit — that is what turns §4 from an assertion into an observation.
- Wrap in `_recenter_term(..., centering)` like the other branches.

`src/superglm/inference/_term_covariance.py::feature_se_from_cov` — add a `Piecewise` branch
mirroring `Categorical`: `M = spec._raw_basis_matrix(spec._knots)[:, spec._non_base_indices]`
(the identity, padded), `se = sqrt(diag(M @ Cov_g @ M.T))`, with `0.0` at the base row. Also
extend the `not active_subs` early return so a dropped Piecewise term returns `zeros(J+2)`,
not `zeros(1)`.

`src/superglm/inference/_term_types.py::TermInference.to_dataframe` — add `"piecewise"` to
the `kind in ("spline", "polynomial")` tuple so the x-bearing frame shape is used.

### Task 2.2 — coefficient rows and the four-condition comment

`src/superglm/inference/coef_tables.py` — add an `elif isinstance(spec, Piecewise)` branch
**before the final `else`** (the fallback emits one row from `b_g[0]` and would silently drop
`J` coefficients). Emit:

- One coefficient row per non-base knot, named `f"{g.name}[{knot:.10g}]"`, carrying
  `coef/se/z/p/ci_low/ci_high` via `_compute_coef_stats`, `edf` on the first row only
  (mirroring the `Categorical` branch).
- One whole-term group row modelled on the `PolynomialCategorical` branch:
  `is_spline=True, n_params=J+1, active, group_norm, wald_chi2, wald_p, ref_df=float(J+1)`,
  computed as `b_g @ solve(V_b_j, b_g)` against `chi2` with `J+1` df, guarded by the same
  `LinAlgError` try/except. `_group_test_kind` will classify it as `"group"` with
  `statistic_type="chi2"`, which is exactly spec §4's "`J+1` df Wald/chi-square on all
  coefficients jointly, testing that the term is flat. Not a Wood smooth test."

**The four-condition comment goes here, at this branch, in one place.** Amendment 3 asks for
the selection-shrinkage dependency; the literature review adds two more. All four:

```
Per-knot Wald rows, their CIs, and edf = J+1 are valid only when ALL FOUR hold:
  1. the term is unpenalized -- the slope penalty of design section 8 is deferred
     precisely because it forfeits the fixed-df contract (k=1 trend filtering has
     df = E[#knots] + k + 1, data-dependent);
  2. the group carries no selection shrinkage -- GroupInfo.penalized is True, so a
     group-lasso selection_penalty shrinks this block and breaks both the Wald rows
     and the fixed edf;
  3. the breakpoints are FIXED INPUTS, not selected on the response from the same
     data -- when a breakpoint is data-chosen the statistic converges to a sup of a
     nonstandard Gaussian process and nominal Wald calibration fails, even though df
     is still nominally J+1. breaks=int quantile placement is materially milder: it
     looks only at x, never at y. Reading kinks off a fitted Spline and re-fitting
     them here on the same data is the real offender;
  4. the term is unconstrained -- under an active monotone constraint the effective
     df becomes the size of the active face, which is data-dependent.
Withdraw the per-coefficient p-values if any of these stops holding.
```

> **DEVIATION 3 from spec §4.** §4 justifies the per-knot Wald tests "precisely because the
> df is fixed and known". That is the wrong reason and it licenses the mistake §1's own
> discovery workflow invites. Validity rests on the **breakpoints being fixed inputs**;
> df stays nominally `J+1` under response-selected breaks while the p-values go wrong.
> Condition 3 above is the correction.

### Task 2.3 — rating-table block

`src/superglm/export/rating_tables.py`:

- Add `_piecewise_features(model)` returning the Piecewise main-effect names.
- Add `_piecewise_block(model, name, centering) -> RatingTableBlock(kind="piecewise")` whose
  table is **exactly three columns**: `[name (knot value), "Relativity", "Log relativity"]`,
  one row per knot in ascending order. Three columns is not cosmetic — see task 2.4.
- In `build_rating_table_payload`'s main-effects loop, test `isinstance(spec, Piecewise)`
  **before** the `name in selected.tables` branch.

> **DEVIATION 4 / interpretation of amendment 7.** The amendment says
> "`_continuous_features` admits `Piecewise`". Taken literally that is a bug in two ways:
> `_continuous_features` is fed to `model.discretization_impact(features=...)`, whose
> `_is_continuous_feature` gate **raises** `ValueError` for a non-`_SplineBase|Polynomial`
> spec; and any name landing in `selected.tables` is routed to the **binned**
> `_continuous_block`, which is the lossy path this whole feature exists to remove. The
> functional intent — the export must not silently omit a Piecewise term, which is what
> today's code does since it matches no branch — is met by the dedicated routing above.
> `_continuous_features` and the discretization-impact sweep stay unchanged, and a Piecewise
> term correctly contributes **no** row to the impact sheet because its export has no
> discretisation error to measure.

### Task 2.4 — excel rendering

`src/superglm/export/excel.py::write_rating_table_workbook`. Two constraints discovered on
this worktree, both load-bearing:

1. `start_col = 1 + idx * 3` — main-effect blocks sit on a **fixed 3-column stride**. A
   4-column block overwrites its right-hand neighbour. This is why the piecewise block is
   3 columns wide and why `Weight` is not one of them (per-knot weight is not a rating-table
   quantity; per-segment weight belongs in the small-weight warning from stage 1).
2. Number formats are applied **globally** by `cell.column % 3`: `% 3 == 2` gets
   `"0.000000"`, `% 3 == 0` gets `"#,##0.00"`. The Log relativity column lands on `% 3 == 0`
   and would render at 2 dp — the values are stored exactly but a human reading or
   copy-pasting the sheet sees `0.00`, which defeats the column's entire purpose.

Required change, in this shape (minimal, layout-preserving):

- Leave the block placement loop and the global format loop **exactly as they are**.
- Afterwards, iterate `payload.main_effects` again; for each `kind == "piecewise"` block set
  `number_format = "0.000000000000"` on its Relativity and Log relativity cells, and write
  the header note into **row 6** of the block's start column (row 5 is the block title, row 7
  is the dataframe header — row 6 is currently unused).
- The note is one string and must state: interpolation is **geometric on Relativity /
  linear on Log relativity**; and both boundary slopes, so the extrapolation rule is
  reproducible outside the tabulated range. Suggested text:
  `"Interpolate linearly on Log relativity (equivalently, geometrically on Relativity).
  Below {t0:.10g} extend at {slope_lo:.10g} log-relativity per unit; above {tJ1:.10g} at
  {slope_hi:.10g}."`

**Framing note for the text (docs and comments alike):** do **not** present linear
extrapolation as self-evidently correct. This library's splines already linear-extrapolate
(`features/_spline_extrapolation.py`), so it is not a differentiator, and the established
alternative — hold flat beyond the boundary knots — has published support for this exact
model class. The defence is the combination: `lower`/`upper` let the user pin the boundary
knots at the tariff's rated range, and the sheet prints the boundary slopes so the rule is
auditable. State that, name the flat alternative, and do not write "novel" anywhere (the
literature sweep found zero arXiv prior art on exact rating-table export of a continuous
fitted effect — that is an unindexed practice area, so absence of evidence is not evidence
of novelty).

### Task 2.5 — tests

`tests/test_rating_table_export.py` — **characterization test first, before touching
`excel.py`.** Fit the existing `_fit_export_model()` workbook, and pin for every main-effect
block: its title cell coordinate, its header-row coordinates, and the `number_format` of its
Relativity and Weight columns. This must pass **before** and **after** the excel change,
proving the 3-column stride and the global format loop are untouched. Sabotage: change the
stride to `idx * 4` → the test fails.

`tests/test_piecewise_reporting.py` (new):

- **§9 property 1 — export exactness.** Fit a model whose terms are all exactly tabulable
  (intercept + one Piecewise + one Categorical + one Numeric — **no spline**, whose block is
  binned by construction). Export to a `BytesIO` workbook, read it back with
  `openpyxl.load_workbook`, and reconstruct predictions **from the workbook alone**: base
  relativity × the categorical lookup × the numeric per-unit power ×
  `exp(np.interp(x, knots, log_rel))` with the boundary slopes parsed from the note for rows
  outside `[t_0, t_{J+1}]`. Assert
  `np.allclose(reconstructed, model.predict(X), rtol=1e-12, atol=0)`. Run it over
  `pinned_narrower` too, so the extrapolating rows are genuinely exercised.
  **Sabotage: round the Log relativity column to 4 dp in `_piecewise_block`** — not the
  basis, which would move both sides of the comparison.
- **Per-knot rows.** `model.summary()` / `build_summary_export_payload` contains exactly
  `J+1` coefficient rows for the term, each with a finite `p_value` and CI, plus one
  `kind == "group"`, `statistic_type == "chi2"` row with `edf`-adjacent `ref_df == J+1`.
- **edf.** On an unpenalized unshrunk fit, the measured term edf equals `J+1` to `1e-9`
  (DEVIATION 2's evidence). Also a `selection_penalty="auto"` smoke fit that completes and
  reports without error (amendment 9) — assert only that it runs and produces rows, since
  the four-condition comment says the Wald numbers are not licensed there.
- **Slopes.** `spec.reconstruct(beta)["slopes"]` matches
  `(v[j+1]-v[j])/(t[j+1]-t[j])` computed independently, and `boundary_slopes` are its
  endpoints.
- **Model-level `discrete=True` fit containing a Piecewise term** completes and predicts
  identically to `discrete=False` for that term (amendment 9). `should_discretize` returns
  False for a non-spline, so this pins that the discrete path leaves the term alone.

### Stage 2 acceptance criteria

- [ ] `model.term_inference(name)` returns `kind="piecewise"` with `J+2` knot values and a
      zero at the base index; `to_dataframe()` returns the x-bearing shape.
- [ ] `model.summary()` shows `J+1` per-knot rows plus one chi-square group row; no
      `TypeError: Unknown feature type`.
- [ ] The four-condition comment is present at the `coef_tables.py` Piecewise branch and
      enumerates all four conditions.
- [ ] Round-trip exactness holds at `rtol=1e-12` on at least `interior_base` and
      `pinned_narrower`.
- [ ] The characterization test passes both before and after the `excel.py` change.
- [ ] The discretization-impact sheet contains **no** row for the Piecewise term.
- [ ] All three standard checks clean; the suite shows the recorded baseline failure and
      no other.
- [ ] Sabotage evidence recorded for every new test, with the export sabotage aimed at the
      **write path**, not the basis.

---

# Stage 3 — editor, plotting hookup, hardening

**Deliverable:** one editor handle per knot, sitting exactly on the knot, whose value is the
coefficient; edits are exact and local; the fixture matrix is swept end to end; ruff clean.

**Depends on:** stage 2 (`term_inference` must return `kind="piecewise"` with `x=knots`
before the editor can build an `EditableTerm` at all).

### Task 3.1 — unblock the editor path

Three gates outside `controls.py`, all found on this worktree, all required:

- `src/superglm/editor/terms.py::term_type_from_spec` → return `"piecewise"` for a
  `Piecewise` spec (it currently falls through to `type(spec).__name__`).
- `src/superglm/editor/session.py:1223 _require_control_term` → admit `"piecewise"` beside
  `"spline"`.
- `src/superglm/editor/payloads.py:252 _controls_payload` → same.

> **DEVIATION 5 / correction to amendment 2.** The amendment says `editor/controls.py`
> "likely needs ZERO changes". That is true of `controls.py` itself — `raw_control_components`
> duck-types on `_raw_basis_matrix` and needs nothing. It is **not** true of the editor as a
> whole: two callers refuse the term before `controls.py` is ever reached. Without these
> three edits, every stage-3 test fails with `TypeError: Term ... does not expose spline
> control handles`.

### Task 3.2 — handles land on knots, one per knot

With `term.x = spec._knots` (stage 2) the basis `raw_control_components` recovers is
`_raw_basis_matrix(knots)` = the **identity**, so:
`np.linalg.lstsq` returns the coefficient vector exactly; `_basis_support_centers` returns
the knots exactly. That is the property that makes "the handles are the coefficients"
literally true rather than approximately true — assert it.

Handle count: `_control_handle_count(n_basis, None)` returns `min(max_handles, 12)`, so a
term with more than 12 knots subsamples. Add a **3-line opt-in** in
`controls.py::raw_control_components`: when the spec exposes a truthy
`_editor_wants_all_handles` attribute (set it on `Piecewise`) and `n_handles is None`,
default `n_handles = basis.shape[1]`. `_control_handle_limits` still caps at 24 — document
in the `Piecewise` docstring that a term with more than 24 knots subsamples its handles, and
assert that in the `many_knots` case rather than pretending it does not happen.

### Task 3.3 — `editor/apply.py`

Add a `Piecewise` branch to `_apply_term_edit` before the final `raise NotImplementedError`,
implemented as `_apply_piecewise_term` on the `_apply_categorical_term` pattern — **local and
exact, no least squares**:

```python
target      = native_log_effect_values(term)      # one value per knot, term.x == spec._knots
base_value  = float(target[spec._base_index])
beta_new    = target[spec._non_base_indices] - base_value
_adjust_intercept(model, base_value)
_patch_beta_block(model, groups, beta_new)
```

This is the `Categorical` path's "#236 for free" property: `v_r = 0` holds by construction,
so a null edit yields `base_value == 0` and `beta_new == beta`, and nothing moves. Do **not**
route this through `_apply_projected_term`.

### Task 3.4 — plotting and screening hookups (both small, both required)

- `src/superglm/plotting/main_effects_plotly.py:629` — add `"piecewise"` to the
  `("spline", "polynomial")` tuple. Without it a Piecewise term falls through to
  `project_grouped_term_for_display`, which needs `ti.levels` and will break both
  `model.plot_main_effects()` and the editor's plotting payload. The knot-marker styling
  from spec §7 stays a stretch goal; this one word is the correctness fix.
- `src/superglm/model/screening_ops.py::_deferral_reason` — add a bespoke Piecewise string
  ahead of the generic fallback, e.g. *"Piecewise margins are deferred: the hat basis is not
  a penalized marginal smooth, so no interaction class refits the pair"*, mirroring the
  Polynomial wording. The generic fallback already works; this makes the report specific.

### Task 3.5 — `tests/test_piecewise_editor.py`

- **§9 property 3a — null edit.** Open an `EditorSession`, apply no change, materialize:
  every prediction is **bit-identical** (`assert_array_equal`, not `allclose`) and the
  intercept is unchanged. This is the #236 property.
- **§9 property 3b — locality.** Move one **non-base** handle by a known delta. Assert rows
  with `x` outside `(t_{j-1}, t_{j+1})` are bit-identical, rows inside moved, and that
  exactly one coefficient changed (by the delta) with the intercept unchanged.
- **§9 property 3c — the base-handle re-base.** Dragging the base handle by `d` re-bases:
  every reported coefficient shifts by `-d` and the intercept shifts by `+d`.
  **The implementer must MEASURE what happens to predictions before writing the assertion.**
  The algebra predicts predictions stay **local** — with `Σ_j h_j = 1`,
  `f'(x) = f(x) + d·h_r(x)`, which is supported only on `(t_{r-1}, t_{r+1})` — so the
  prediction-locality assertion should hold for the base handle too, while
  *coefficient*-locality does not. Amendment 2 says to exclude the base handle from the
  locality test; that exclusion is correct for the **coefficient** assertion and, per the
  algebra above, unnecessary for the **prediction** assertion. Run it, record what you
  observe, and assert what you measured. If predictions turn out non-local, that is a real
  finding about `_apply_piecewise_term` — report it, do not weaken the test.
- **Handles.** `session.control_points(name)` returns exactly `J+2` handles at exactly the
  knot values (`assert_allclose(..., spec._knots, atol=0)`), with `log_effect` equal to the
  reported knot log-relativities. For `many_knots`, assert the documented cap behaviour.
- **Fixture-matrix sweep.** Parametrize the null-edit and locality tests over every case in
  `tests/_piecewise_cases.py`.
- **Plot smoke.** `model.plot_main_effects()` (or the plotly builder directly) completes for
  a Piecewise term — this is what pins task 3.4's one-word fix.

**Sabotage plan:** null edit → make `_apply_piecewise_term` subtract `target[0]` instead of
`target[base_index]`; locality → replace the branch with `_apply_projected_term`; handles →
remove the `_editor_wants_all_handles` opt-in and check the `many_knots` assertion fails;
plotting → revert the one-word tuple change.

### Stage 3 acceptance criteria

- [ ] `EditorSession.control_points` on a Piecewise term returns `J+2` handles on the knots.
- [ ] Null edit is bit-identical; single-handle edit is local; base-handle edit re-bases with
      the measured prediction behaviour asserted (and recorded in the narrative).
- [ ] `model.plot_main_effects()` completes for a Piecewise term.
- [ ] PSST screening reports a bespoke Piecewise deferral string rather than the generic one.
- [ ] Full fixture matrix swept by at least the stage-1 property tests and the stage-3
      edit tests.
- [ ] `uv run ruff check src/ tests/` and `uv run ruff format --check src/ tests/` clean;
      the full suite shows the recorded baseline failure and no other.
- [ ] Sabotage evidence recorded for every new test.

---

## Explicitly deferred — record, never silently drop

| Item | Reason | Where it was scoped |
|---|---|---|
| `docs/guide/features.md` — a `Piecewise` section + the §1 join-behaviour table | Stretch goal per the prototype scope. If skipped, **say so in the stage-3 report**. When written: free joins = binned `Categorical` (`OrderedCategorical(basis="step")` is deprecated and `basis=Categorical` does not exist); splines here already linear-extrapolate, so the pitch is a **stated, hand-reproducible boundary slope**, not extrapolation splines lack; `transform` evaluates hats directly in about a dozen lines (scipy's `design_matrix` *does* support `extrapolate=True` — we simply do not need it); name flat-beyond-boundary as the established alternative; do not write "novel". | spec §7 |
| Knot markers / segment styling in `main_effects_plotly` | Stretch. The one-word correctness fix in task 3.4 is **not** stretch. | spec §7 |
| `TermInference` gaining a slopes field | Would widen a public frozen dataclass for a prototype. Slopes live in `reconstruct()` and in the workbook note. | spec §4.1 |
| Per-interior-knot kink contrast ("is this breakpoint needed") | Amendment 8 marks it optional-and-only-if-cheap. It is an exact linear contrast `c'Σc`, not a delta-method approximation — record that for whoever picks it up. | amendment 8 |
| Monotone `Piecewise` | At degree 1 `v_{j+1} ≥ v_j` is **exactly** monotonicity (a theorem, not an approximation — coefficient constraints are necessary *and* sufficient up to quadratic order). Still deferred: it would add a caller to the hand-rolled active-set QP whose termination is unproven. When it lands, align with the published cone-projection / active-set family rather than extending the existing QP. | spec §8 |
| Slope-change penalty | Forfeits the fixed-df contract: this model class under an ℓ1 slope-change penalty is k=1 trend filtering, whose df is `E[#knots] + k + 1` — data-dependent. | spec §8 |
| `Piecewise ×` anything (interactions, PSST screening) | Reported as deferred, not silently skipped — `_spec_kind` returns the class name, `add_interaction` raises `TypeError` naming supported pairs, `_deferral_reason` reports it. | spec §8 |
| `strategy="kmeans"` | Seam reserved in stage 1 rule 3; default unchanged; evidence recorded but not acted on. | literature P1 |

---

## Open risks

1. **`edf` reporting (DEVIATION 2).** The plan reports the measured edf and *tests* that it
   equals `J+1`. If the measured value turns out **not** to be `J+1` on an unpenalized
   unshrunk fit, that is a finding about how `group_edf` is computed for an unpenalized
   group — report it, do not switch to hardcoding `J+1` to make the test pass.
2. **Excel layout.** The 3-column stride is a real constraint discovered by reading
   `excel.py`, not a stylistic choice. If a later reviewer asks for a `Weight` column, the
   stride refactor and its characterization test come with it.
3. **Base-handle prediction locality (task 3.5c).** Asserted from algebra, must be confirmed
   by measurement before the assertion is written.
4. **Validation-rule overlap.** Rules 8, 9 and 10 all fire on some degenerate inputs. If a
   sabotage of one rule does not fail that rule's own test, the test is measuring another
   rule — fix the test, not the sabotage.

---

# Stage 1 narrative — completed 2026-08-06

**Status: complete.** `Piecewise` builds, transforms, scores, reconstructs, validates and is
exported from `superglm`. Spec §9 properties 2 and 4 hold on all eight fixture cases. 50 new
tests, every one with recorded sabotage evidence.

Files: `src/superglm/features/piecewise.py` (new), `src/superglm/__init__.py` (import +
`__all__`), `tests/_piecewise_cases.py` (new), `tests/test_piecewise.py` (new).
**`dm_builder.py` was not touched** — the claim in the pre-flight table held under test.

### Measured results

```
uv run pytest tests/test_piecewise.py -q          -> 50 passed
uv run pytest tests/ -q -m "not slow"             -> 1 failed, 6372 passed, 195 skipped,
                                                     137 deselected  (215s)
  FAILED tests/test_structured_screening.py::test_a_thin_level_does_not_cost_the_pair_a_
         degree_of_freedom[1.0]        <- the recorded baseline failure, and no other
uv run ruff check src/ tests/                     -> All checks passed!
uv run ruff format --check src/ tests/            -> 430 files already formatted
```

6372 = the recorded 6322 baseline + the 50 new tests. Version untouched at `0.19.0`.

### Deviations and findings

**DEVIATION 1 (`|h_j|` in rule 8) — implemented as planned.** Confirmed necessary: on
`pinned_narrower` the tail rows carry negative entries, so the amendment's signed form can
cancel to zero on a non-zero column.

**NEW DEVIATION 1a — int-mode breakpoints are snapped to observed x values before dedup.**
The plan says to place them with `weighted_quantile_knots` and let `np.unique` collapse ties.
Measured on the `heaped_int_x` geometry (x heaped on {10, 20, 30} plus a light scatter over
multiples of 5, `breaks=8`):

```
raw weighted quantiles : [6.3775, 7.9811, 9.5846, 15.8704, 17.4402, 19.0101, 25.7306, 28.2165]
per-segment weight     : [23.58, 0, 0, 211.67, 0, 0, 208.85, 0, 146.93]
```

Two consequences, both fatal to the plan as written:

1. That helper **interpolates a weighted CDF**, so it lands breakpoints in the *gaps between*
   the heaps. Four segments bracket no rows at all, so rule 9 raises — a bare `breaks=8`
   becomes a hard error. That is precisely the "library's mistake reported as theirs" the
   plan's own rule-5 rationale forbids.
2. Because the CDF is strictly increasing at every observed value, the interpolated
   quantiles are **essentially never equal**, so `np.unique` never collapses anything and the
   documented shrinkage warning is unreachable. A test for it would have been untestable, or
   worse, testable only against a contrived input.

Snapping each quantile to the nearest observed value fixes both: it realises 6 of 8 on the
same data (so the warning fires for the documented reason), every segment provably carries
the rows sitting on its lower knot, and a breakpoint is now a value that actually occurs —
a better property for a filed tariff than an interpolated 6.3775. Implemented in
`_resolve_knots` with the measurement recorded in the comment.

**Rule 6 is split.** `lower < upper` (6a) must run *before* int-mode placement, which needs
the range to know which rows are inside it; "breaks strictly inside" (6b) runs after. The
build docstring states the full order.

**Rule 6b generalised in int mode.** Int-mode placement drops any value not *strictly
inside* the range rather than only values equal to `lower`/`upper`; quantiles are also
computed on rows inside `[lower, upper]` only, mirroring how `resolve_interior_knots` treats
an explicit boundary. Both are no-ops when the range defaults to the data.

**`sample_weight` length is checked** as part of the rule-1 input-contract step. Without it a
mismatched weight vector reaches `np.bincount` and reports a confusing shape error.

**`strategy` is validated in sequence mode too**, where it is not consulted, so a typo'd
keyword fails loudly instead of silently doing nothing.

**Finding for stage 3: `SuperGLM` deep-copies the feature spec at fit time.** The fitted
knots, base index and non-base indices live on `model._specs[name]`, *not* on the spec object
the caller constructed (which stays unbuilt, `_knots.size == 0`). Every stage-2/3 test that
wants fitted state must read it off `model._specs`. Pinned by
`test_a_model_with_a_piecewise_term_fits_and_predicts`.

**Confirmed for stage 2/3:** `warnings.warn(..., stacklevel=3)` from inside `_resolve_knots`
points at the `spec.build(...)` call site (`dm_builder.py:997`), which is the intended frame.

### Fixture matrix as built (`tests/_piecewise_cases.py`)

| Case | knots | group size | base index | notes |
|---|---|---|---|---|
| `interior_base` | 5 | 4 | 2 | base at an interior knot |
| `end_base_lower` | 5 | 4 | 0 | `base="first"` |
| `end_base_upper` | 5 | 4 | 4 | base at `t_{J+1}` |
| `unequal_widths` | 5 | 4 | 2 | widths 5, 5, 50, 40 |
| `pinned_wider` | 5 | 4 | 2 | range [-20, 140] over data [0, 100] |
| `pinned_narrower` | 4 | 3 | 1 | range [20, 80] over data [0, 100]; tail rows at fit time |
| `heaped_int_x` | 8 | 7 | 4 | `breaks=8` realises 6, **warns** |
| `many_knots` | 15 | 14 | 6 | above the editor's 12-handle default |

`heaped_int_x` is the only case that warns at `build()`; anything else warning is a finding.

### Sabotage evidence

34 sabotages, each a single exact edit to `src/superglm/features/piecewise.py`, each run
against the whole new test file and then reverted. Every one of the 50 tests fails under at
least one sabotage; **no sabotage was inert**. Driver and raw JSON:
`scratchpad/sabotage.py`, `scratchpad/sabotage_results.json` (session scratch, not committed).

Named basis sabotages required by the brief:

```
negate the upper-tail w  :: 8 failed :: all 8 TestLinearExtrapolation cases
clip w to [0, 1]         :: 8 failed :: all 8 TestLinearExtrapolation cases (incl. pinned_narrower)
drop the LAST hat        :: 8 failed :: 7 of 8 TestCoefficientMeaning cases + reconstruct
```

`drop the LAST hat` leaves `end_base_upper` passing — for that case the last knot *is* the
base knot, so the sabotage is a no-op there. Recorded rather than papered over.

The eleven validation rules, each disabled alone (the overlap trap the plan flagged):

```
RULE 1 finite x                      :: 2 failed :: both rule-1 tests, nothing else
RULE 1 sample_weight length          :: 1 failed :: its own test
RULE 2 empty breaks (sequence)       :: 1 failed :: its own test
RULE 2 empty breaks (int)            :: 1 failed :: its own test
RULE 3 unknown strategy              :: 1 failed :: its own test
RULE 3 reserved seam opened          :: 2 failed :: both rule-3 tests
RULE 4 strictly increasing           :: 2 failed :: both rule-4 tests
RULE 5 dedup shrinkage warning       :: 1 failed :: its own test
RULE 5 nothing realised inside       :: 1 failed :: its own test
RULE 6a lower < upper                :: 1 failed :: its own test
RULE 6b breaks strictly inside       :: 1 failed :: its own test
RULE 7 float base names one knot     :: 1 failed :: its own test
RULE 7 unknown base keyword          :: 1 failed :: its own test
RULE 7 no-weight fallback            :: 2 failed :: its own test + the repr test (which
                                                    asserts ref=0, i.e. the fallback)
RULE 7 most_exposed argmax->argmin   :: 2 failed :: its own test + the refit-reuse test
RULE 8 zero hat mass                 :: 1 failed :: ONLY its own test
RULE 9 empty segment                 :: 1 failed :: ONLY its own test
RULE 10 rank check                   :: 1 failed :: ONLY its own test
RULE 11 thin-segment warning         :: 1 failed :: its own test
RULE 11 threshold constant           :: 1 failed :: its own test
```

Rules 8, 9 and 10 each fail **only** their own test, which is the property the plan's open
risk 4 demands: the rule-8 fixture (`breaks=[2]`, x in {1, 3}) also trips rule 10, and the
rule-9 fixture also trips rule 10, so each test asserts on its own rule's message text and
the rule-10 test uses the discriminating fixture measured at plan time (column masses
`[1, 0.5, 0.5, 1]`, all three segments occupied, retained rank 2 of 3 — reproduced exactly by
the implementation).

Contract sabotages covering the remaining tests:

```
build() scales its columns          :: 1 failed :: test_transform_equals_build_columns
build() keeps the base column       :: 21 failed :: incl. both integration tests
GroupInfo penalized=False           :: 1 failed :: test_build_reports_j_plus_one_unpenalised_columns
score() drifts from transform@beta  :: 9 failed :: test_score_equals_transform_at_beta + property 2
_raw_basis_matrix drops base column :: 2 failed :: raw-basis + partition-of-unity tests
reconstruct() skips the width       :: 1 failed :: test_reconstruct_returns_knot_relativities...
unfitted transform() guard          :: 1 failed :: test_transform_before_build_names_the_missing_step
repr loses its knot count           :: 1 failed :: test_repr_before_and_after_build
perturb w by 1e-9                   :: 19 failed :: incl. the identity test
drop the 1-w term                   :: 18 failed :: incl. the partition-of-unity test
```

### Stage 1 acceptance criteria — all met

- [x] `uv run python -c "from superglm import Piecewise; print(Piecewise([1,2]))"` prints
      `Piecewise(breaks=[1, 2], base='most_exposed')`.
- [x] `tests/test_piecewise.py` green (50); full suite shows the baseline failure and no other.
- [x] `ruff check` and `ruff format --check` clean.
- [x] A fitted `SuperGLM` with a Piecewise term fits and predicts with **no** `dm_builder.py`
      change.
- [x] Sabotage evidence recorded for every new test; each validation rule fails only its own.
- [x] Committed with `git add -f` on the plan doc.

### Handed to stage 2

- `spec._knots` (J+2), `spec._base_index`, `spec._base_knot`, `spec._non_base_indices`,
  `spec._strategy_actual`, `spec._n_breaks_requested` are all populated by `build()`.
- `reconstruct(beta)` returns `knots`, `base_knot`, `base_index`, `log_relativity`,
  `relativity`, `slopes`, `boundary_slopes` exactly as the plan specifies.
- `_raw_basis_matrix(knots)` is the identity, so stage 3's least-squares handle recovery is
  exact — verified by `test_hat_basis_at_the_knots_is_exactly_the_identity`.
- `_editor_wants_all_handles` is **not** set on `Piecewise`; stage 3 adds it with the
  `controls.py` opt-in, as scoped.

---

# Stage 2 narrative — completed 2026-08-06

**Status: complete.** A `Piecewise` term reports `J+1` per-knot Wald rows and one `J+1`-df
whole-term chi-square, reports its *measured* edf (which equals `J+1` on an unshrunk fit),
and exports a `kind="piecewise"` rating-table block that reproduces `model.predict` from the
workbook alone. 55 new tests, every one with recorded sabotage evidence.

Files: `inference/_term_ops.py`, `inference/_term_covariance.py`, `inference/_term_types.py`,
`inference/coef_tables.py`, `inference/summary.py` (one extra site, below),
`export/rating_tables.py`, `export/excel.py`, `tests/test_piecewise_reporting.py` (new),
`tests/test_rating_table_export.py` (characterization test).

### Measured results

```
uv run pytest tests/test_piecewise_reporting.py -q      -> 53 passed
uv run pytest tests/test_rating_table_export.py -q      -> 61 passed (60 existing + 1 new)
uv run pytest tests/ -q -m "not slow"                   -> 6427 passed, 195 skipped,
                                                           137 deselected, 0 FAILED  (261s)
uv run ruff check src/ tests/                           -> All checks passed!
uv run ruff format --check src/ tests/                  -> 431 files already formatted
uv lock --check / uv pip check                          -> clean
```

6427 = stage 1's 6372 + the 55 new tests, with **zero** failures.

**The stage-1 baseline failure did not reproduce.** A full-suite run on the stage-1 commit at
the start of this stage gave `6373 passed, 195 skipped, 137 deselected` with **zero**
failures, and `test_structured_screening.py::test_a_thin_level_does_not_cost_the_pair_a_
degree_of_freedom[1.0]` also passes in isolation. It is order- or thread-sensitive, not
deterministic as the plan's baseline note assumed. Nothing was changed to make that happen.

### Deviations

**DEVIATION 2 (measured edf) — implemented, and now backed by measurement rather than
assertion.** `test_the_measured_edf_equals_j_plus_one_on_an_unshrunk_fit` measures `J+1` to
`1e-9` on all eight fixture cases. `test_a_shrunk_fit_reports_less_than_j_plus_one` is the
other half and it is the one that settles the design question: under
`selection_penalty="auto"` the same `interior_base` term measures **edf = 3.622**, not 4.
Hardcoding `J+1` would have over-reported that fit's degrees of freedom by 10%.

**DEVIATION 3 (why the Wald rows are valid) — implemented as the four-condition comment** at
the `coef_tables.py` Piecewise branch, with condition 3 (breakpoints are fixed inputs) named
as the load-bearing one and the design's own "because the df is fixed and known" explicitly
corrected. The comment is pinned by a test that reads the source, and the sabotage for it
deletes the **whole 28-line block** — replacing only its first line leaves the other three
conditions standing and the sabotage measured INERT, which is exactly the trap the plan's
sabotage rules describe.

**DEVIATION 4 (`_continuous_features` unchanged) — implemented and confirmed necessary.**
Sabotage `RATING _continuous_features admits Piecewise` takes amendment 7 literally and fails
6 tests: `discretization_impact`'s continuity gate rejects the spec. The Piecewise routing
sits before the `selected.tables` branch instead, and the impact sheet correctly carries no
row for the term (asserted against a model that *also* contains a spline, so the sheet is
genuinely populated and the assertion cannot pass for the wrong reason).

### Smaller decisions, each measured

- **The block is three columns and the note lives in row 6.** Both are pinned by the
  characterization test, which passes byte-identically before and after `excel.py` changed.
  Sabotaging the stride to `idx * 4` fails it, and so does flipping the global format loop's
  residue — the second is extra, since the brief only required the first.
- **Boundary slopes are printed at full round-trip precision**, not at `:.10g`. Measured:
  `.10g` breaks the exactness test on `pinned_narrower` (the extrapolating rows pick up a
  ~1e-10 relative error), which is precisely the silent model-versus-tariff discrepancy this
  feature exists to remove. Recorded as sabotage `EXCEL note slopes rounded to 10 significant
  figures`.
- **`inference/summary.py` gained one `elif` in each of its two renderers** so the whole-term
  row prints `[piecewise, 4 params, ...]` rather than `[spline, ...]`. The group row has to
  set `is_spline=True` to be classified as a chi-square group test, and both renderers label
  every such row "spline" unless `subgroup_type` says otherwise. Not in the brief's file list,
  but calling a piecewise term a spline in the console summary is the kind of small lie this
  repository does not ship. The new branch fires only on `subgroup_type == "piecewise"`.
- **`_piecewise_block` reads `term_inference`**, mirroring `_categorical_block`, so the
  workbook cell and the summary row are the same number by construction rather than by
  agreement between two derivations. The note's boundary slopes are derived in `excel.py`
  from the block's own printed columns for the same reason.
- **`_piecewise_features()` was not added.** The plan lists it, but with `isinstance` routing
  in the main-effects loop it would have no caller.
- **edf is reported on the whole-term row only, not also on the first knot row.** The plan
  said to mirror `Categorical`, which puts edf on its first level row -- but it does that
  because a categorical term has no term-level row to put it on. This one has. Carrying it
  twice reads as two numbers about one term and lands the same degrees of freedom in the
  summary's parametric bucket *and* its smooth bucket. The ordered-spline branch already does
  it this way. Pinned by an assertion that exactly one row in the term carries edf, with the
  sabotage `COEF edf reported twice`.

### Measured round-trip error

```
interior_base    n_extrapolating = 0     max relative error = 6.64e-16
pinned_narrower  n_extrapolating = 226   max relative error = 5.27e-16
```

The test's tolerance is derived (`32 * eps = 7.1e-15`, a flop count over both compared paths)
and is asserted to sit inside the design's `1e-12`, so both bars are pinned.

### Sabotage evidence

29 sabotages, each one exact text edit, each run against
`tests/test_piecewise_reporting.py` + `tests/test_rating_table_export.py` and then reverted;
the campaign was re-run in full against the **final** source so the evidence certifies the
committed bytes. **0 inert.** All 55 new tests fail under at least one sabotage. Driver and
raw JSON: `scratchpad/sabotage_stage2.py`, `scratchpad/sabotage_stage2_results.json` (session
scratch, not committed).

```
TERM_OPS kind is no longer 'piecewise'                    :: 16 failed
TERM_OPS x drops the knot vector                          :: 20 failed
TERM_OPS edf hardcodes the nominal J+1                    ::  1 failed (only the shrunk-fit test)
TERM_OPS branch removed (Unknown feature type)            :: 32 failed
TERM_OPS base knot loses its exact zero                   :: 10 failed
TERM_COV piecewise SE branch removed                      :: 26 failed
TERM_COV inactive early return back to zeros(1)           ::  1 failed (only its own test)
TERM_TYPES to_dataframe drops 'piecewise'                 ::  8 failed
COEF piecewise branch removed (one-row fallback)          :: 11 failed
COEF whole-term row is not a group test                   :: 10 failed
COEF ref_df is not J+1                                    ::  8 failed
COEF four-condition comment block deleted                 ::  1 failed (only its own test)
COEF edf reported twice (also on the first knot row)      ::  8 failed
COEF group row no longer labelled piecewise               ::  1 failed
COEF per-knot rows named from column order                ::  8 failed
COEF per-knot CIs collapse onto the estimate              ::  8 failed
RATING write path rounds Log relativity to 4 dp           ::  3 failed  <- the required write-path sabotage
RATING piecewise routing removed (block omitted)          ::  6 failed
RATING _continuous_features admits Piecewise              ::  6 failed
RATING block emits a Weight column instead of the log     ::  5 failed
EXCEL main-effect stride widened to 4                     ::  8 failed  <- the required characterization sabotage
EXCEL global format loop keys on the wrong residue        ::  1 failed
EXCEL piecewise annotation pass never runs                ::  4 failed
EXCEL note slopes rounded to 10 significant figures       ::  2 failed
EXCEL log-relativity column left at the global 2 dp       ::  1 failed
EXCEL note written over the block title row               ::  3 failed
PIECEWISE reconstruct forgets the segment width           ::  8 failed
DM_BUILDER discretizes a Piecewise term                   ::  1 failed
CASES pinned_narrower stops extrapolating                 ::  1 failed (the fixture guard)
```

### `.test_durations` had to be extended — a real failure stage 1 nearly caused

`tests/test_ci_contracts.py::test_duration_manifest_covers_the_non_browser_suite` requires the
committed pytest-split duration manifest to cover **95%** of the collected non-browser suite.
Stage 1's 50 new tests took it to `6331/6634 = 0.9543`; stage 2's 55 took it to
`6331/6689 = 0.9465` and it **failed**. Fixed by measuring the three touched test files and
merging only the **111 node ids not already present** into `.test_durations`, so every
pre-existing entry keeps its recorded value byte for byte (the diff removes exactly one line,
the closing brace). Whoever adds tests in stage 3 must do the same:

```
uv run pytest <new files> -q --store-durations --durations-path /tmp/new.json
# then merge only the keys absent from .test_durations, and re-sort
```

A blanket `--store-durations` over the whole suite would rewrite all 6374 existing values and
bury the real change.

### Findings for stage 3

- **`plotting/data.py:213` is a SECOND plotting site the plan's task 3.4 does not list.**
  `_main_effect_density_dataframe` dispatches on `ti.kind in ("spline", "polynomial")` then
  `"numeric"`, then falls through to `list(ti.levels)` — which is `None` for a piecewise term
  and raises `TypeError`. It needs the same one-word addition as
  `main_effects_plotly.py:629`. Before this stage `term_inference` raised on a Piecewise spec,
  so neither site was reachable; they are reachable now.
- **`editor/summaries.py:235` labels any `is_spline` row `"spline"`** in the editor's summary
  payload, the same way `inference/summary.py` did. Stage 3 owns the editor; the fix mirrors
  the `subgroup_type == "piecewise"` branch added here.
- **The console summary's "smooth p-values use Wood (2013)" note fires for any `is_spline`
  row**, so a model whose only group-test row is a Piecewise term prints it. This is
  pre-existing (a `PolynomialCategorical` term does the same) and was left alone rather than
  changed as a side effect of this stage. The *export* payload is already correct: it keys on
  `kind.startswith("smooth")`, and the piecewise row's kind is `"group"`.
- `spec._knots`, `spec._base_index` and `spec._non_base_indices` live on `model._specs[name]`,
  as stage 1 recorded. `term_inference(name).x` is now the knot vector, which is the input
  stage 3's editor handles need.

### Stage 2 acceptance criteria — all met

- [x] `term_inference` returns `kind="piecewise"` with `J+2` knots and a zero at the base
      index; `to_dataframe()` returns the x-bearing shape.
- [x] `summary()` shows `J+1` per-knot rows plus one chi-square group row; no
      `TypeError: Unknown feature type`.
- [x] The four-condition comment is present at the `coef_tables.py` branch and is pinned by a
      test whose sabotage deletes the whole block.
- [x] Round-trip exactness holds on `interior_base` and `pinned_narrower`, measured at
      ~6e-16 against a derived tolerance asserted to be inside `1e-12`.
- [x] The characterization test passes before and after the `excel.py` change.
- [x] The impact sheet has no Piecewise row; `_continuous_features` unchanged.
- [x] `ruff check` and `ruff format --check` clean.
- [x] Sabotage evidence recorded for all 55 new tests; 29 sabotages, 0 inert.
