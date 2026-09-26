# Shaped ranges: replace Breaks mode with range-pinned polynomials

Date: 2026-09-25. Branch `feat/editor-structural-tools` (the one editor PR).
Status: design agreed with Max ("go", 2026-09-25); detailed defaults below
were delegated and are recorded as such.

## 1. Problem

Breaks mode asks for the wrong thing. It splits the axis at clicked breaks and
then applies ONE form to the whole term: Piecewise (straight segments),
Polynomial (one polynomial over the whole axis, breaks ignored) or Spline
(knots at the breaks, which throws away the penalised smooth and its REML
control). Max's model is different: *ranges of the axis, each with its own
shape, while everything not touched stays the fitted smooth* — "quadratic over
a–b, linear over c–d, leave the rest alone". Breaks mode cannot express it and
was not intuitive to him.

## 2. The term (library)

A spline whose function is pinned to a polynomial of degree <= d on chosen
ranges and is the ordinary penalised spline elsewhere.

    Spline(kind="bs", k=12, polynomial_ranges=[
        PolynomialRange(18, 25, degree=1),
        PolynomialRange(70, 80, degree=0),
    ])

`PolynomialRange(lo, hi, degree, join="kink")`, exported from `superglm`.
`degree` in 0..3 (flat, line, quadratic, cubic) and at most the spline degree.
`join` is `"kink"` (continuous value, slope may change: C0) or `"smooth"` (the
spline's own continuity at the edge).

Construction — each step is standard and cited:

1. **Edge knots.** Each range edge becomes a knot, with multiplicity equal to
   the spline degree for `join="kink"` (C0 at the edge) and 1 for
   `"smooth"` (Curry–Schoenberg; de Boor, *A Practical Guide to Splines*).
   Without an edge knot the constraint would leak across the whole knot
   interval the edge falls in.
2. **Equality constraints.** "Polynomial of degree <= d on [lo, hi]" is linear
   in the coefficients: the (d+1)-th derivative vanishes on every knot interval
   inside the range (evaluated exactly with the B-spline derivative recursion),
   plus, when d equals the spline degree, continuity of the d-th derivative at
   the interior knots of the range. Rows are certified full rank.
3. **Absorption.** `C beta = 0` is absorbed by `beta = Z theta` with Z from the
   QR of C' — the same mechanism as the natural-spline constraints and
   sum-to-zero identifiability (Wood 2017, §1.8.1). Implemented by overriding
   `_SplineBase._apply_constraints`, which already flows through fit,
   `fit_reml`, covariance, predict and export unchanged.
4. **Penalty only outside the ranges.** For derivative penalties (`bs`, `cr`)
   the penalty is an exact sum of per-knot-interval blocks (Wood 2016,
   arXiv:1605.02446, §1); summing only the free intervals leaves the pinned
   polynomials unpenalised. REML then chooses one smoothing parameter for the
   free parts: global control stays. A range covering every interval leaves a
   zero penalty, and REML treats the group as unpenalised.
5. **Kinds.** `bs` and `cr` only. Difference penalties (`ps`, `ns`) assume
   equally spaced knots, which repeated edge knots break; the editor rebuilds
   a `ps`/`ns` term as `bs` with the same knots, degree and penalty order and
   says so in Help (decided during planning, 2026-09-25). `cr_cardinal`
   refuses by name (its penalty divides by knot gaps). Fit-time shape
   constraints and `select=True` together with ranges are refused in this
   version — named follow-ups.
6. **One piece per range.** The spline's own knots inside a range are
   dropped, so each range is a single polynomial piece and pinning it to
   degree d is `spline_degree - d` derivative rows. On `cr`, a Flat or Line
   range touching an end omits that end's natural-boundary row (implied, and
   dependent). Adjacent ranges must meet at a kink (a smooth shared edge makes
   the rows dependent).

Roadmap checked (`notes/ROADMAP.md`): no editor or shape-fitting entries; no
conflict.

Validation (refusals with named messages): lo < hi, inside the fitted range;
ranges non-overlapping (adjacent allowed, sharing the edge); at least
`degree + 1` distinct observed x values inside each range; degree bounds.

Reporting: `fitted_knots` reports the repeated knots so the ppform export is
exact; the rating-table/Excel paths need no new block kind. Summary, Python
`summary()` and the workbook carry the existing editor disclosure, reworded:
"Shaped ranges for X were chosen in the editor from this data; tests are
conditional on them — judge them on validation deviance." (Reuses the
`EDITOR_BREAKS_ATTRIBUTE` marker, renamed to `EDITOR_CHOSEN_SHAPE_ATTRIBUTE`.)

Ordered categoricals: an `OrderedCategorical(basis=Spline(...))` takes
`polynomial_ranges` on the level axis (band positions, or `values=` when
given); ranges snap to whole bands. An ordered term without a spline basis is
out of scope (its existing selection edits remain).

## 3. The editor

- **Gesture.** In Select mode, select a contiguous run of points or bands on a
  numeric spline term or an ordered term with a spline basis. The selection
  palette shows four shape icons: **Flat, Line, Quadratic, Cubic**. Choosing
  one refits immediately (a structural step, like collapse); **Restore**
  undoes it. No popup, no separate mode ([[editor-ui-must-not-intrude]]).
- **Range.** Numeric: [min, max] of the selected x, snapped as the old breaks
  were (3 significant figures of the fitted span). Ordered: whole bands.
- **Join.** Kink (C0) in this version. The library supports `"smooth"`; an
  editor toggle is a later addition if Max wants it.
- **Whole-axis polynomial.** Select all, then Quadratic. No separate form.
- **Several ranges.** Each shape is one step; a new range overlapping an
  existing one is refused by name ("overlaps the Line range 18–25").
  Choosing a shape on an existing range with a different degree replaces it.
- **Visible.** Pinned ranges are drawn as a light shaded band with a small
  label (Flat/Line/Quadratic/Cubic); hover names the range and degree.
- **Unavailable cases** show the icons disabled with the reason on hover:
  categorical terms, linear Numeric terms, `cr_cardinal`, a term with a
  fit-time shape constraint, an interaction parent, fewer than degree+1
  distinct x in the selection.
- **Removed.** Breaks mode (tool rail `B`), `breaks.js`,
  `chart/break_overlay.js`, `views/breaks_controls.js`, the transform form
  picker, `/transform_term` and `editor/transform.py`'s Piecewise/Spline/
  Polynomial forms, their tests and help/tutorial text. `/shape_range`
  replaces `/transform_term` on the shared structural path
  (`_refit_replacing`, `StructuralStep`, `structure_history`).

## 4. Also in this change (Max's request)

**Feature list with search.** Replace the context-bar `<select id="term">`
with a collapsible left-hand feature list: search box (filters as you type),
the current feature highlighted and scrolled into view, arrow keys step
between features, each row shows kind and EDF. Collapses to a thin strip so
the plot keeps its width.

## 5. Out of scope (named follow-ups)

Smooth-join editor toggle; shaped ranges with fit-time monotone/convex
constraints; ordered terms without a spline basis; `cr_cardinal`; Build
animation duration/scrub; fixed editor port for remote use.

## 6. Verification

Library: the pinned function is a polynomial of degree <= d on the range to
round-off (residual of a degree-d fit on a dense grid); continuity at edges
(value always; slope changes allowed only for `kink`); penalty excludes the
range (a pinned quadratic is not shrunk as lambda grows); a range covering the
whole axis reproduces an unpenalised polynomial fit; predictions, covariance
and ppform export agree (export exact to 1e-11); REML converges on the demo
book; refusals named. Editor: backend step/undo tests, browser test of the
gesture and Restore, frontend unit tests for the range snapping and palette
enablement. Complete-fit timing on the demo book vs the plain spline.
