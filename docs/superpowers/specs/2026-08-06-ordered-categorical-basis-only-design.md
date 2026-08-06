# OrderedCategorical goes basis-only

Design, 2026-08-06. Approved by Max in conversation the same day.

## Motivation

`OrderedCategorical` currently offers three ways to say what smooth to fit:

- `basis=Spline(...)` — the canonical form,
- `basis="spline"` plus the scalar shortcuts `kind`, `n_knots`, `degree`,
  `select`, `penalty` — deprecated since a6b5530 (2026-07-10), each emitting a
  `FutureWarning`,
- `basis="step"` — a one-hot encoding with a first-difference penalty,
  separately deprecated.

Three channels for one decision is the actual defect. It produced a reported
bug: a `Spline` object passed to `kind=` (the parameter next to `basis=`) was
stored unvalidated and rejected much later by the private spline factory,
against a list of *string* kinds, naming neither the parameter at fault nor the
one to use. PR #242 added a boundary guard for that, but a guard protecting a
parameter that should not exist is compensation, not a fix.

Removing the alternatives makes `basis=` the only way to configure the smooth
and deletes the class of confusion outright.

## Decisions

All three were Max's, made explicitly:

1. **The five scalar shortcuts are removed** from `__init__`.
2. **Step mode is removed** — "step is deprecated and marked for removal, basis
   is how we are going forward". `basis` stops accepting the `"spline"` and
   `"step"` strings.
3. **`basis` keeps a default.** `OrderedCategorical(order=[...])` must still
   work. Making `basis` mandatory was considered and rejected.

Strategy: **two sequenced PRs**, not one. The two removals are independent
concerns with different risk profiles — one is the configuration channel, the
other a fitting mode — and separating them keeps a mechanical change away from
a deletion of real numerics.

Both PRs stack on #242, because the removal deletes the parameters that PR's
guard protects.

## Target API

```python
OrderedCategorical(
    values=None, order=None,
    basis=None,             # a Spline object, or None
    base="most_exposed",
    grouping=None,
    specials=None,
)
```

`basis=None` must reproduce today's default exactly: `Spline(kind="ps",
n_knots=5, degree=3, penalty="ssp", select=False)`, with `n_knots` clamped to
`n_levels - 1` and the existing clamp warning preserved.

## PR1 — remove the scalar shortcuts

Delete from `OrderedCategorical.__init__`:

- the parameters `kind`, `n_knots`, `degree`, `select`, `penalty`;
- `shortcut_values` / `used_shortcuts` and the two `FutureWarning` branches
  that report ignored or deprecated shortcuts;
- the misplaced-Spline guard added in #242, which exists solely to protect
  these parameters;
- the `select=True`-with-`basis="step"` check, whose `select` operand is gone
  (step mode itself survives PR1; the check does not).

`basis` still accepts a `Spline`, `"spline"` or `"step"` after PR1. The
`basis="spline"`-is-deprecated warning stays until PR2.

`_init_spline`'s no-object branch stops reading `self.kind` etc. and constructs
the default spline directly.

### The derived attributes must become unconditional

This is the one non-mechanical part of PR1 and it is easy to get wrong.

Today `self.kind`, `.select`, `.penalty`, `.degree`, `.n_knots` are assigned
twice: first from the shortcut values (`self.kind = resolved_kind`, always),
then *re-derived* from the inner spline — but only under
`if self._spline_obj is not None`. Delete the shortcuts and the first
assignment disappears with them, so a spec built from `basis="spline"` (still
legal in PR1, since `_spline_obj` is `None`) would leave all five unset.
`editor/collapse.py` reads them and would raise `AttributeError`.

PR1 must therefore drop the `_spline_obj is not None` guard and derive all five
from `self._spline` whenever spline mode built one — which `_init_spline`
always does. In step mode, which survives PR1, `_spline` is `None`; set the
five to `None` there rather than leaving them absent, so no reader sees a
missing attribute. PR2 deletes that branch along with step mode.

### Migration

161 `OrderedCategorical(` call sites exist in `tests/`. By shape:

| shape | count | action |
| --- | --- | --- |
| `basis=Spline(...)` **and** a shortcut | 69 | delete the shortcut — it is already ignored, so this is a no-op |
| `basis="spline"` + shortcut | 31 | rewrite as `basis=Spline(...)` |
| no `basis=` + shortcut | 5 | rewrite as `basis=Spline(...)` |
| no `basis=`, clean | 8 | unchanged (default preserved) |
| `basis="spline"`, clean | 3 | unchanged in PR1 |
| `basis="step"` | 39 | unchanged in PR1 |
| `basis=<variable>` | 6 | inspect individually |

The 69 no-op deletions are the bulk of the diff and carry no risk: those
shortcuts are already discarded at runtime with a warning. The ~36 real
rewrites are the ones that need care.

Tests asserting the deprecation warnings themselves are deleted, not migrated.

## PR2 — remove the basis strings and step mode

`basis` accepts only a `Spline` instance or `None`. Anything else raises.

Delete:

- `_build_step`, `_reconstruct_step`;
- `_R_inv` on this class, its `set_reparametrisation` branch, and the step
  branches in `transform`, `score` and `reconstruct`;
- the D1/Z first-difference penalty construction;
- the step-mode `FutureWarning` and the `specials=`-with-step guard;
- the `basis="spline"` legacy-string deprecation warning;
- `editor/collapse.py`'s `basis="step"` reconstruction branch;
- the `NotImplementedError` in `resolve_interaction_parent` that refuses step
  parents, along with its `spec.basis != "spline"` test.

`_choose_base` and `_base_level` **stay**: spline mode calls them to anchor
reported relativities.

The ~39 `basis="step"` test sites are deleted rather than migrated. Step mode
has no spline-mode equivalent to migrate them *to* — a caller who wants
unsmoothed level effects wants `Categorical`, which is what the existing
deprecation message already says.

## What survives

`.kind`, `.n_knots`, `.degree`, `.select`, `.penalty` remain as **derived
attributes**, populated from the inner spline (see "The derived attributes must
become unconditional" above). `editor/collapse.py` reads them. Only the
constructor parameters are removed.

Once `basis` is always a `Spline`, `_spline_obj` is never `None`, so
`editor/collapse.py`'s fallback branch — which rebuilds a `Spline` from those
five attributes — becomes unreachable and is deleted with PR2.

`self.basis` is **kept as a vestigial constant** `"spline"`. It is read as a
string in nine source files (`dm_builder`, `model/screening_ops`,
`model/report_ops`, `export/summary`, `editor/apply`, `editor/collapse`,
`features/ordered_categorical`, and their tests). Removing the attribute would
mean collapsing every one of those checks, widening PR2 far beyond this class
and into inference, export and screening. That cleanup is deliberately deferred
and is not part of this design.

## Testing

The failure mode to defend against is a migrated call site that silently
changes the smooth and still passes. Green is not evidence here; equivalence
is.

- For each of the ~36 real rewrites, assert the resulting `_spline` matches
  what the shortcut form produced — same type, `n_knots`, `degree`, `penalty`,
  `select` — rather than only that the model fits.
- Cover each *shape* from the migration table with at least one case, not only
  the simplest one. A suite that exercises `order=` but never `values=`, or
  never the `grouping`/`specials` combinations, measures less than it appears
  to.
- Invert the existing deprecation tests: assert the removed parameters raise
  `TypeError` rather than warn.
- Assert `OrderedCategorical(order=[...])` with no `basis` still builds the
  same default spline it does today — this is the one behaviour the whole
  design promises to preserve.

## Risks and open items

**`_non_base` and `_R_inv` need a per-site audit before PR2.** Both names are
owned by `Categorical` as well as `OrderedCategorical`, and roughly ten source
files read them. Sampling shows the reads are either guarded by
`isinstance(spec, Categorical)` or keyed off `spec._levels`, which
`OrderedCategorical` does not have — so on OC they appear step-only and safe to
delete. That was sampled, not proven. Every site must be checked individually
during planning, in particular the `getattr(spec, "_non_base", None)` form in
`model/report_ops.py`, which is defensive and would silently accept an OC.

**Version.** Both PRs are breaking public API changes and need a minor bump.
v0.19.0 is tagged and on PyPI; #242 lands as 0.19.1. The exact target for PR1
and PR2 is a release decision to make when they are ready, not now.

## Out of scope

- Removing `self.basis` and simplifying the nine files that read it.
- Any change to `Categorical`.
- The `ns` / `cr_cardinal` fit-time constraint gap, which is a spline-engine
  limitation unrelated to this class.

## Status (2026-08-10, implementation)

Shelved unreviewed on 2026-08-06; un-shelved and implemented against master
`8453eb6` (v0.22.0) as **one branch and one bump (0.24.0)** rather than the two
sequenced PRs above — both #242-era guard commits this design stacked on had
already landed on master independently, so the PR1/PR2 sequencing had nothing
left to protect. Deviations from the letter of this document, each following
master's evolution:

- The five derived attributes became read-only **properties** over the inner
  spline rather than unconditionally re-assigned fields; writes now raise, and
  `.n_knots` reports the post-clamp effective value in the default path too.
- `basis=None` constructs the default `Spline` into `_spline_obj`, so
  `_spline_obj` is never `None` on a constructed spec; the clamp warning for
  the default path is now worded like the explicit-object one.
- `editor/collapse.py`'s five-attribute rebuild fallback is deleted as planned,
  replaced by a `_basis_spline` read that keeps an old spline-mode pickle
  cloneable and refuses an old step-mode pickle loudly.
- `resolve_interaction_parent`'s step `NotImplementedError` is deleted as
  planned; `dm_builder`'s registration-time step guard (added after this doc,
  0.19.x) is kept and reworded for removal, since it still gives pre-0.24
  pickles an early refusal.
- The `_non_base` audit resolved cleanly: `_choose_base` still populates it in
  spline mode, so every cross-file read is unchanged; only OC's own `_R_inv`
  was step-only, and no file outside the class reads it.
- The migration table was stale: master had already migrated the 69 no-op
  sites during 0.19–0.22, leaving 40 shortcut sites and 37 step sites.
