# Migration: a spline × factor level is held where its responses are all zero

*Ships in 0.40.0.*

## What changed

In a spline-by-factor interaction (`interactions=[("x", "level")]` with a spline
`x`), each level of the factor has its own curve. Where a level's responses are
all zero over a range of `x`, and only that level's curve can lower the fitted
rate there, nothing in the data stops the curve from falling. A level that
REML smooths lightly can then extrapolate to the predictor's floor: on the
freMTPL2 frequency book's uncorrected claim counts, brand B12's vehicles older
than 22 years (21 rows, no claims) were priced at a log rate of -80, a premium
of essentially zero. On those counts B12's curve was lightly smoothed because
of a spike in claims at vehicle age 0, and that spike is an artefact: the
book's `ClaimNb` is wrong for policies with `IDpol` up to 24,500 (Wüthrich &
Merz, *Statistical Foundations of Actuarial Learning*, Section 13.1). With the
counts taken from the severity file (`examples/fremtpl2_data.py`) there is no
spike and B12 is smoothed far more, but its tail still has no claims (VehAge 23
to 49, 24 training rows) and is still held, as are ten other levels.

The build now finds those ranges and holds the curve there. A response "off
the boundary" is a nonzero one, or for a binomial response one strictly
between 0 and 1 (a range of all-1 responses is held the same way):

- Before the first and past the last value where the level has a response off
  the boundary, the curve keeps its value at that point, in the fit and in
  `predict()`. A spline does the same past its training range
  (`extrapolation="clip"`).
- Between two values with responses off the boundary, the curve takes the
  smoothest path that joins its two sides.
- The factor's base level has no curve of its own (its curve is the main
  effect's, which every level shares), so its ranges are found on the main
  effect alone and named as approximate, not held; `separation="error"` does
  not refuse on them.
- A range the fit cannot hold is still named, as "left to the smoothing
  penalty".

A `SeparationWarning` names the term, the level and the range, for example
`'VehAge:VehBrand', level 'B12': VehAge 23 to 48 (21 rows, total weight 15.04,
every response 0): held at its value at VehAge 22`. The solver's own separation
warning now names the term whose contribution drove the rows, where it used to
name the term with the largest coefficient.

## Who is affected

- **Spline-by-factor terms where a level's responses are all zero over a
  range of the spline**: the fit changes there, and the rest of the fit moves
  slightly with it. Terms without such a range are bitwise unchanged. The
  hold applies whatever penalty bounds the term; a selection penalty exempts
  only the categorical cells it targets.
- **Every other term**: nothing changes. Main-effect splines, tensor (`ti`)
  interactions, `FactorSmooth` terms and `SuperLSS` are not scanned.

## What to do

- **To keep 0.39's curves**: pass `separation="ignore"`, which now also leaves
  these curves free. `separation="error"` refuses the design instead.
- **To let the data shape the range**: merge the level into a similar one, or
  drop the interaction.

## Why a hold, not something else

Past a level's last nonzero response the data say only that the rate is low,
so the curve there is a convention. The alternatives were weighed against the
literature:

- **Warning only**, as mgcv does: the rate stays at the floor.
- **Firth or Jeffreys pseudo-observations** (Firth 1993; Kosmidis & Firth
  2021): their finiteness argument needs a full-rank design, which a penalised
  spline with empty knot spans is not; on B12's tail they add about 3.5
  pseudo-claims on 15 units of exposure, roughly four times the rate of the
  neighbouring ages.
- **Penalising the smoothing parameter** (Sterzinger & Kosmidis): the tail is
  not what drives the small smoothing parameter (on the uncorrected counts
  the age-0 artefact did), so it does not help.
- **One smoothing parameter for all levels** (mgcv's `id=`): treats the cause,
  a lightly smoothed level, but changes every spline-by-factor fit.

The ranges are found with the linear programs that decide whether a maximum
likelihood estimate exists (Geyer 2009, Theorem 4; Konis 2007).

## Measured

The freMTPL2 frequency book on its uncorrected claim counts (542,410 training
rows), Poisson with exposure, `discrete=True` at 256 bins, the screening
guide's main effects and ten screened pairs. This compares two versions of the
code on one fixed input, and the corrected counts give different numbers
(not re-measured here):

| | 0.39 rule | 0.40 |
|---|---|---|
| B12 log rate past age 22 (training rows) | -80 to -13.3 | -4.55 to -2.29 |
| Rows with a log rate below -10 | 14 training, 9 test | none |
| Test deviance | 42,456.29 | 42,454.48 |
| Levels held | none | 44 |
| Fit time, one thread | 83-92 s | 79-83 s |

## Verification trail

- `tests/test_spline_cat_separation.py` pins the held tail at 256 bins, 16 bins
  and on the exact path (through `predict()`), `"ignore"` and `"error"`, a fit
  without separation bitwise equal to the `"ignore"` fit, an interior gap, and a
  faintly separated level.
- `tests/test_separation.py` pins the warning's attribution.
- `tests/test_realdata_parity.py` pins B12 on the book (skips without it).
