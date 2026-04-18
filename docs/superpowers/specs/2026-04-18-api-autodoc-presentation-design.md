# API Autodoc Presentation Cleanup Design

## Goal

Keep `mkdocstrings` as the source of truth for API reference pages, but make the rendered `Features` and `Families` pages read like intentional API docs instead of raw introspection dumps.

## Scope

This design covers only the first API-doc presentation pass:

- `docs/api/features.md`
- `docs/api/families.md`
- `mkdocs.yml`
- the small set of public docstrings whose current wording makes those two pages noisy or misleading

This pass is based on `origin/master` at `c133a82`.

## Non-goals

- Full site reorganization
- Guide-page rewrite beyond light consistency fixes required by the API cleanup
- Notebook cleanup
- Replacing `mkdocstrings` with handwritten API pages
- Broad CSS-only restyling as the primary solution

## Problems To Fix

### Rendering noise

The current API pages expose too much `mkdocstrings` chrome:

- symbol badges and symbol-type TOC noise
- large constructor-heavy headings dominating the page
- inconsistent flow between factories and concrete classes
- visual emphasis on implementation structure instead of the public API

### Content noise

The current pages mix good public material with wording that reads like internal transition notes:

- factory and class sections are not clearly separated
- family semantics need one short handwritten clarification instead of forcing users to infer behavior from signatures alone
- public spline docs need to read cleanly with the `PSpline` / `BSplineSmooth` split

## Chosen Approach

Use thin curation plus autodoc:

1. tune `mkdocstrings` defaults in `mkdocs.yml` to reduce repeated chrome
2. keep handwritten wrapper pages that group objects intentionally
3. clean only the public docstrings that materially affect these rendered pages

This keeps maintenance low while still producing readable pages.

## Page Design

### Features

The `Features` API page should render in this order:

1. Short intro explaining that the page is split between the public factory and the concrete feature classes.
2. `Factory` section:
   - `superglm.Spline`
   - one short handwritten note clarifying that `kind="ps"` and `kind="bs"` map to different concrete smooth classes
3. `Spline Classes` section:
   - `superglm.PSpline`
   - `superglm.BSplineSmooth`
   - `superglm.NaturalSpline`
   - `superglm.CubicRegressionSpline`
4. `Other Feature Classes` section:
   - `superglm.Categorical`
   - `superglm.Numeric`
   - `superglm.Polynomial`

Only concrete public classes should appear. Internal helpers and module attributes should not.

### Families

The `Families` API page should render in this order:

1. Short intro explaining the difference between family objects and convenience constructors.
2. `Factories` section:
   - `superglm.families.poisson`
   - `superglm.families.gaussian`
   - `superglm.families.gamma`
   - `superglm.families.binomial`
   - `superglm.families.nb2`
   - `superglm.families.tweedie`
3. `Family Classes` section:
   - `superglm.Poisson`
   - `superglm.Gaussian`
   - `superglm.Gamma`
   - `superglm.Binomial`
   - `superglm.NegativeBinomial`
   - `superglm.Tweedie`
4. One short handwritten note above the class list stating:
   - known-scale families keep `phi=1`
   - `NegativeBinomial` uses `theta` as the dispersion/overdispersion parameter instead of a meaningful fitted `phi`

## Rendering Rules

The rendered object blocks should emphasize:

- object name
- one-line summary
- arguments
- attributes or properties
- returns where relevant

The pages should de-emphasize or hide:

- private members
- internal helpers
- inherited clutter that is not part of the public surface
- symbol-type decoration that adds visual noise without adding meaning

## MkDocs / mkdocstrings Changes

The implementation should adjust `mkdocs.yml` so that the generated blocks on these API pages are quieter by default.

Expected changes:

- disable symbol-type headings
- disable symbol-type TOC entries
- keep source links disabled
- keep no-docstring objects hidden
- preserve merged constructor docs only if they still read better than a separate `__init__` block after the other cleanup; otherwise split them

If constructor merging still makes the page read poorly after the other changes, the implementation should switch off `merge_init_into_class`.

## Docstring Rules

Only public docstrings affecting these pages should be updated.

Required style:

- first sentence is a clean one-line summary
- parameter descriptions are short and consistent
- public behavioral distinctions are explicit
- no transition-history or compatibility-window prose on the public API pages

Examples:

- `PSpline` should clearly state “B-spline basis + discrete-difference penalty”
- `BSplineSmooth` should clearly state “B-spline basis + integrated-derivative penalty”
- `NegativeBinomial` should clearly state that overdispersion is controlled by `theta`

## Files Expected To Change

- `mkdocs.yml`
- `docs/api/features.md`
- `docs/api/families.md`
- selected public docstrings under:
  - `src/superglm/features/spline.py`
  - `src/superglm/distributions.py`
  - `src/superglm/families.py` if factory wording needs cleanup

## Acceptance Criteria

The first pass is successful when all of the following are true:

1. `Features` no longer shows module-attribute or alias junk in the public spline area.
2. `Features` visibly separates the factory from the concrete spline classes.
3. `Families` visibly separates factories from family classes.
4. `Families` explicitly communicates the `NB2 theta` vs `phi` distinction.
5. The generated API pages still come primarily from `mkdocstrings`, not large manual rewrites.
6. The rendered pages read as public API docs rather than raw introspection output.

## Verification Plan

Implementation should verify with:

- docs build using the docs dependency group
- inspection of generated `site/api/features/index.html`
- inspection of generated `site/api/families/index.html`
- targeted lint/tests only if touched docstring or config changes require them

## Risks

### Over-correcting globally

Changing `mkdocstrings` defaults in `mkdocs.yml` affects every API page, not just `Features` and `Families`.

Mitigation:

- prefer settings that improve readability generically
- keep page-specific structure in the Markdown wrapper pages
- if a setting makes other API pages worse, move the fix into page structure or docstrings instead of forcing a global style

### Constructor rendering tradeoff

`merge_init_into_class` may still be useful for some pages even if it is noisy here.

Mitigation:

- evaluate generated output after the other cleanup first
- only change constructor merging if it materially improves readability on the target pages without making the rest of the API worse

## Recommended Next Step

After spec approval, write an implementation plan for this focused subproject only, then execute it in the dedicated worktree from `origin/master`.
