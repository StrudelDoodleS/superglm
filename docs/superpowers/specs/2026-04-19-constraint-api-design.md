# Constraint API Design

## Goal

Replace the current two-argument monotonicity API

- `monotone=...`
- `monotone_mode=...`

with a single typed public API based on constraint tokens such as:

- `constraint=Constraint.fit.increasing`
- `constraint=Constraint.postfit.decreasing`

The old public API should be removed, not kept as a long-lived compatibility layer.

## Scope

This design covers the public spline constraint interface only:

- public constructor API for spline specs and the `Spline(...)` factory
- validation and normalization of constraint tokens
- docs and tests for the new public surface
- explicit `NotImplementedError` behavior for `convex` / `concave`

This design does **not** change the underlying solver architecture:

- `SCOP` remains the fit-time monotone path for `PSpline`
- `QP` remains the fit-time monotone path for `BSplineSmooth` and `CubicRegressionSpline`
- post-fit monotone repair remains the existing `postfit` behavior

## Current Problem

The current API is explicit but clunky:

```python
PSpline(..., monotone="increasing", monotone_mode="fit")
```

Problems:

- users must supply two loosely related string arguments
- the API is stringly typed
- validation errors are weaker than they could be
- docs must explain a pairing rather than one concept
- it scales poorly to additional constraint kinds

## Chosen Direction

Introduce a single `constraint=` argument and remove `monotone` / `monotone_mode`
from the public constructor surface.

Target call sites:

```python
PSpline(..., constraint=Constraint.fit.increasing)
BSplineSmooth(..., constraint=Constraint.fit.increasing)
CubicRegressionSpline(..., constraint=Constraint.postfit.decreasing)
```

This keeps the public API:

- explicit
- compact
- typed
- extensible

## Naming

Use `Constraint`, not `Monotone` or `Shape`.

Reasoning:

- `Monotone` is too narrow if `convex` / `concave` are added later
- `Shape` is slightly more abstract than needed today
- `Constraint` is clear at the call site and still leaves room for future shape constraints

Recommended public namespace:

- `Constraint.fit.increasing`
- `Constraint.fit.decreasing`
- `Constraint.fit.convex`
- `Constraint.fit.concave`
- `Constraint.postfit.increasing`
- `Constraint.postfit.decreasing`
- `Constraint.postfit.convex`
- `Constraint.postfit.concave`

For now, only `increasing` and `decreasing` are implemented. `convex` and
`concave` should exist as public tokens but raise `NotImplementedError` when used.

## Object Model

The constraint token should be an immutable declarative value, not a stateful
service object.

Conceptually:

```python
Constraint.fit.increasing
-> ConstraintSpec(mode="fit", kind="increasing")
```

Required properties of the token:

- immutable
- no hidden state
- no model reference
- printable / debuggable repr
- comparable in tests

The constraint token should **not** know anything about the model object. It is
just intent. The spline constructor and build pipeline interpret that intent.

## Why It Does Not Need Model State

The constraint object should not “know” it is inside a model. The flow should be:

1. user passes `constraint=Constraint.fit.increasing`
2. spline spec stores normalized intent
3. build/validation code checks whether that spline family supports it
4. fit-time or post-fit code routes to the correct implementation

This keeps the constraint token simple and makes validation live in the correct
place: the spline family and the model build path.

## Internal Normalization

To minimize implementation risk, the first version should normalize the new
token back onto the existing internal fields:

- `spec.monotone`
- `spec.monotone_mode`

That means solver code can stay mostly unchanged in the first pass.

Example normalization:

```python
constraint=Constraint.fit.increasing
-> spec.monotone = "increasing"
-> spec.monotone_mode = "fit"
```

This keeps the internal transition small while giving users the new API.

## Public API Changes

### Constructors

Affected public constructors:

- `Spline(...)`
- `PSpline(...)`
- `BSplineSmooth(...)`
- `NaturalSpline(...)`
- `CubicRegressionSpline(...)`
- `CardinalCRSpline(...)` if it is meant to expose the same surface

Public change:

- add `constraint=...`
- remove `monotone=...`
- remove `monotone_mode=...`

### OrderedCategorical

If `OrderedCategorical(..., basis=<Spline object>)` already delegates to the
inner spline object, it should inherit the new API automatically.

This design does not require a separate `OrderedCategorical` constraint API.

## Validation Rules

### Implemented now

Supported now:

- `Constraint.fit.increasing`
- `Constraint.fit.decreasing`
- `Constraint.postfit.increasing`
- `Constraint.postfit.decreasing`

### Reserved now

Publicly constructible but not implemented:

- `Constraint.fit.convex`
- `Constraint.fit.concave`
- `Constraint.postfit.convex`
- `Constraint.postfit.concave`

Using those should raise:

```python
NotImplementedError
```

with a message that clearly says convex / concave constraints are not yet implemented.

### Existing spline-family guard rails remain

Examples:

- `NaturalSpline` fit-time monotonicity unsupported
- `select=True` with fit-time monotone constraints unsupported
- `selection_penalty > 0` with fit-time monotonicity unsupported
- mixed `SCOP` and `QP` monotone engines unsupported

These should remain family/model-level validation errors, just triggered via
the new `constraint=` path.

## Error Semantics

Desired error behavior:

- wrong type for `constraint=` -> `TypeError`
- unsupported token kind for the chosen spline family -> `NotImplementedError`
- convex/concave token used anywhere today -> `NotImplementedError`
- old `monotone` / `monotone_mode` args used -> explicit failure, not silent acceptance

Because this is repeal-and-replace, the old API should fail loudly rather than
silently mapping through.

## Repeal-And-Replace Policy

The old public API should not remain the recommended interface.

Preferred implementation posture:

- remove `monotone` / `monotone_mode` from public signatures
- reject them if passed as keyword arguments
- update all public docs/examples/tests to the new `constraint=` shape

If short-term plumbing requires the internal fields to remain in the spec
object, that is acceptable as an implementation detail. The public interface
should still be the new one only.

## File Areas Likely To Change

- `src/superglm/features/spline.py`
- `src/superglm/features/_spline_factory.py`
- `src/superglm/features/_spline_config.py`
- possibly `src/superglm/types.py` if a public `ConstraintSpec` type belongs there
- public docs and examples that currently use `monotone=` / `monotone_mode=`
- tests for monotone fitting and API validation

## Acceptance Criteria

1. Users can express supported monotone behavior with one argument:
   - `constraint=Constraint.fit.increasing`
   - `constraint=Constraint.postfit.decreasing`
2. The old two-argument public form is rejected.
3. Existing fit-time routing still works:
   - `PSpline` -> `SCOP`
   - `BSplineSmooth` / `CubicRegressionSpline` -> `QP`
4. `convex` / `concave` tokens exist but raise `NotImplementedError`.
5. Ordered categorical features using a spline basis inherit the new API naturally.
6. Docs and examples consistently use `constraint=...`.

## Risks

### Risk: over-generalizing too early

If `Constraint` is made too abstract now, the first implementation may become
heavier than necessary.

Mitigation:

- keep the first token model minimal: `mode` + `kind`
- normalize onto the existing internal monotone fields

### Risk: breaking too many call sites at once

A repeal-and-replace change will touch constructors, docs, and many tests.

Mitigation:

- keep solver behavior unchanged
- isolate the work to API/validation/docs/tests in the first pass

## Recommended Next Step

Write an implementation plan for a focused migration that:

1. introduces `Constraint`
2. normalizes it to the current internal monotone fields
3. removes/rejects the old public args
4. updates docs and tests
