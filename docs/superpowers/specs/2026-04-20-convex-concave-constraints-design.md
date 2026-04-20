# Convex And Concave Constraint Design

## Goal

Implement real convex and concave spline constraints across the public
`constraint=` API, with:

- fit-time support
- postfit repair support
- `fit_reml()` support
- `discrete=True` support
- no regressions to existing unconstrained or monotone performance paths

This work should extend the existing constraint system without collapsing
everything into a generic abstraction that slows ordinary models down.

## Scope

This design covers univariate spline families only:

- `PSpline`
- `BSplineSmooth`
- `CubicRegressionSpline`

The design includes:

- public `Constraint.fit.convex`
- public `Constraint.fit.concave`
- public `Constraint.postfit.convex`
- public `Constraint.postfit.concave`
- fit-time solver integration
- postfit repair integration
- `fit_reml()` semantics
- `discrete=True` semantics
- docs, tests, and benchmark coverage

This design does **not** include:

- mixed constraints in one token or one term
  - no monotone+convex / monotone+concave combinations in v1
- tensor-product shape constraints
- response-scale convexity/concavity

## Public Semantics

Convexity and concavity are defined on the spline term on the linear
predictor scale.

That applies to both:

- fit-time constraints
- postfit repair

This is the mathematically appropriate contract for the current spline API,
because the basis functions, penalties, linear constraints, and SCOP
reparameterizations all operate on the additive term itself.

Response-scale convexity is intentionally out of scope because it is
link-dependent and is not a clean term-local property in a GAM/GLM.

## User-Facing API

Target call sites:

```python
from superglm import Constraint, PSpline, BSplineSmooth, CubicRegressionSpline

PSpline(n_knots=10, constraint=Constraint.fit.convex)
PSpline(n_knots=10, constraint=Constraint.postfit.concave)

BSplineSmooth(n_knots=10, constraint=Constraint.fit.convex)
CubicRegressionSpline(n_knots=10, constraint=Constraint.fit.concave)
```

Publicly supported tokens after this work:

- `Constraint.fit.increasing`
- `Constraint.fit.decreasing`
- `Constraint.fit.convex`
- `Constraint.fit.concave`
- `Constraint.postfit.increasing`
- `Constraint.postfit.decreasing`
- `Constraint.postfit.convex`
- `Constraint.postfit.concave`

The public API stays as a single `constraint=` argument. No new public
`engine=` argument is added.

## Current State

Today the public token layer already exists, but only monotonicity is real.

Internally:

- `Constraint.*` tokens normalize to legacy internal shape fields
- fit-time `PSpline` monotonicity uses the SCOP path
- fit-time `BSplineSmooth` and `CubicRegressionSpline` monotonicity use
  projected QP inequality constraints
- postfit support is monotone-only and isotonic
- `fit_reml()` already has split semantics:
  - dedicated SCOP path for fit-time monotone PSplines
  - QP passthrough heuristic for auto-lambda monotone terms

This is a good base because the outer fitting and REML machinery is already
aware of constrained terms. The missing pieces are the convex/concave
constraint builders, repairers, and SCOP basis generalization.

## Reference Model

This design deliberately follows two external references:

- `mgcv::scasm`
  - reference for QP-style fit-time convex/concave constraints via linear
    inequalities on B-spline coefficients
- `scam`
  - reference for SCOP-style convex/concave basis constructions for
    P-splines

The relevant takeaway is:

- QP convex/concave is a linear-constraint problem
- SCOP convex/concave is a basis reparameterization problem

## Approaches Considered

### 1. QP-only convex/concave

Implement convex/concave only for the QP-capable spline families and leave
`PSpline` unsupported.

Pros:

- lowest implementation risk
- fastest to ship
- easiest performance story

Cons:

- fails the product requirement for `PSpline`
- creates an awkward API asymmetry

### 2. Dual-engine convex/concave

Implement convex/concave for both current fit engines:

- SCOP for `PSpline`
- QP for `BSplineSmooth` and `CubicRegressionSpline`

Add postfit convex/concave repair for all three.

Pros:

- matches the intended product
- preserves the current engine split instead of forcing users through one
  generic path
- fits naturally into existing `fit_reml()` and `discrete=True` architecture

Cons:

- requires a true SCOP convex/concave design, not a small patch

### 3. Fully generalized shape framework first

Introduce a broad internal shape system with mixed constraints and future
tensor support, then implement convex/concave on top.

Pros:

- future-proof

Cons:

- too broad for the current goal
- higher risk of perf regressions
- likely to slow shipping and muddy acceptance criteria

## Chosen Direction

Choose approach 2.

Implement convex/concave for both fit engines, but keep the internal design
minimal and direct:

- no mixed constraints in v1
- no tensor generalization in v1
- no new public engine knobs

## Engine Split

### PSpline

`PSpline` fit-time convex/concave uses a true SCOP-style second-order shape
reparameterization.

This should not silently fall back to the generic QP path. The point of
keeping SCOP here is:

- performance
- consistency with the current monotone `PSpline` path
- compatibility with the dedicated SCOP `fit_reml()` route

### BSplineSmooth

`BSplineSmooth` fit-time convex/concave uses projected QP linear inequality
constraints.

### CubicRegressionSpline

`CubicRegressionSpline` fit-time convex/concave uses projected QP linear
inequality constraints.

### Postfit

All three families support postfit convex/concave repair on the linear
predictor scale.

## Internal Representation

The public `ConstraintSpec(mode, kind)` token model can remain.

However, the implementation should stop assuming “shape = monotone”.
Internally the spec should carry generic shape intent:

- shape kind
- enforcement mode

It is acceptable for the first implementation wave to continue normalizing
onto legacy fields if that reduces risk, but the logic that branches on shape
must become generic enough to distinguish:

- increasing
- decreasing
- convex
- concave

The build path should decide:

- whether the spline family supports the requested shape
- which engine applies
- which build helpers / repair helpers are required

## QP Fit-Time Design

For QP-capable spline families, convex/concave should follow the
`mgcv::scasm` model.

Use linear inequality constraints on the raw coefficient vector:

- convex: second differences constrained to the appropriate sign
- concave: second differences constrained to the opposite sign

These constraints should be built in raw basis space, then composed through:

- family-specific basis constraints
- identifiability projection

just as the current monotone QP constraints are.

This means the existing group build contract remains intact:

- `GroupInfo.constraints`
- `GroupInfo.monotone_engine`
- `GroupInfo.raw_to_solver_map`

The engine label should stop being conceptually monotone-only. It can remain a
string field if that is the smallest implementation step, but it now means
"fit-time constrained engine" rather than only "monotone engine".

## SCOP Fit-Time Design

### High-Level Requirement

`PSpline` convex/concave must use a dedicated SCOP-style shape-preserving
reparameterization rather than generic inequality constraints.

### Conceptual Structure

Monotone SCOP today is:

- positive first differences via exponentiated parameters
- cumulative sum to recover coefficients

Convex/concave SCOP should generalize this to second-order shape:

- positive or negative second differences encoded through exponentiated latent
  variables
- integrated twice to recover coefficients
- affine null space for the linear component

So compared with monotone SCOP:

- null-space dimension becomes `2` instead of `1`
- the basis centering / identifiability story changes
- initialization changes
- penalty mapping changes

### Practical Contract

The outer Newton loop should stay reusable as far as possible.

The reusable pieces are:

- nonlinear forward map in solver space
- Jacobian
- penalty matrix in solver space
- QP-based initialization in transformed space

The existing SCOP solver objects are currently monotone-specific. They should
be generalized carefully so the solver consumes a shape-aware reparameterizer
interface rather than only the monotone exp-increment map.

The goal is to preserve:

- current monotone behavior and performance
- joint SCOP Newton support
- discrete/bin-index support
- dedicated SCOP REML/EFS integration

without adding extra overhead to ordinary monotone or unconstrained models.

## Postfit Convex/Concave Repair

Postfit convex/concave repair should be added as a sibling to the existing
monotone repair path.

The repair target is the reconstructed spline term on a dense linear-predictor
scale grid.

Expected pipeline:

1. reconstruct the fitted spline term on a grid
2. repair the grid values to satisfy convexity or concavity
3. project the repaired curve back onto spline coefficients
4. recenter to preserve identifiability
5. patch model coefficients and invalidate relevant caches

This should be a new repair path rather than trying to force everything
through isotonic regression. Convex/concave repair is a second-order shape
problem, not a monotonicity problem.

The public API should remain consistent with the current postfit style:

- the term is marked with `Constraint.postfit.convex` / `concave`
- model-level postfit application uses the same overall workflow as monotone
  repair

## fit_reml() Semantics

### QP Path

QP convex/concave should follow the same semantics as existing QP monotone
terms:

- fixed lambdas: fit directly under constraints
- auto lambdas: unconstrained REML first, then constrained refit at those
  lambdas

This is a heuristic passthrough, not exact joint constrained REML.

That should be documented explicitly.

### SCOP Path

SCOP convex/concave should get a dedicated integrated path analogous to the
current SCOP monotone REML/EFS route.

The outer architecture is already in place:

- fixed constrained REML path
- SCOP-specific EFS path for estimated lambdas

The new work should extend this architecture to the second-order SCOP basis,
not replace it.

## discrete=True Semantics

`discrete=True` is an explicit requirement for this feature.

That means:

- QP convex/concave constraints must compose correctly with discretized group
  matrices
- SCOP convex/concave must support bin-level design matrices and the existing
  discretized Newton setup
- no “implemented, but dense-only” first release

The design should reuse the same discrete-vs-dense branching strategy the
current monotone paths already use, rather than inventing a parallel codepath.

## Performance Guardrails

The performance requirement is not “convex/concave must be as cheap as
unconstrained”.

The real requirement is:

- existing unconstrained paths do not slow down
- existing monotone paths do not slow down
- the new convex/concave paths are performant enough to be usable rather than
  self-defeating

That implies the implementation should:

- avoid generic shape machinery on the hot path for ordinary models
- branch early on the requested constraint kind
- build only the matrices / cached state required for the active constraint
- preserve current fast paths for models with no convex/concave terms

Validation should include:

- no measurable slowdown on representative unconstrained benchmarks
- no measurable slowdown on representative monotone benchmarks
- bounded time-to-fit for new convex/concave benchmarks in dense and discrete
  modes

## Benchmark Expectations

Before merge, benchmark coverage should include at least:

- existing monotone `PSpline` with `fit_reml()` and `discrete=True`
- existing monotone QP spline with `fit_reml()`
- new convex `PSpline` with `fit_reml()` and `discrete=True`
- new convex QP spline with `fit_reml()` and `discrete=True`

The benchmark goal is not absolute speed parity with monotone. The goal is:

- no regression on old paths
- no pathological blow-up on new paths

## Documentation

Docs must cover:

- the new convex/concave tokens
- linear predictor scale semantics
- family/engine split
- `fit_reml()` semantics for QP vs SCOP
- `discrete=True` availability
- postfit repair semantics

Docs should also be explicit that mixed constraints are not included in v1.

## Testing

Required test areas:

- token normalization and validation
- family-level support / rejection
- fit-time convex/concave enforcement for all three target families
- postfit convex/concave repair
- `fit_reml()` fixed-lambda behavior
- `fit_reml()` auto-lambda behavior
- `discrete=True` parity with dense behavior
- regression tests showing no slowdown / no control-flow regression on existing
  monotone and unconstrained paths

## File Areas Likely To Change

- `src/superglm/features/constraint.py`
- `src/superglm/features/_spline_config.py`
- `src/superglm/features/_spline_build.py`
- `src/superglm/features/_spline_constraints.py`
- `src/superglm/features/_spline_subclass_ops.py`
- `src/superglm/features/spline.py`
- `src/superglm/solvers/scop.py`
- `src/superglm/solvers/scop_newton.py`
- `src/superglm/solvers/irls_direct.py`
- `src/superglm/model/fit_ops.py`
- `src/superglm/model/reml_execute.py`
- `src/superglm/model/monotone_ops.py` or a generalized shape-repair module
- tests, docs, and benchmark scripts

## Versioning

This should ship with a minor feature-version bump, not a patch bump.

Reason:

- it expands the public constraint feature set materially
- it adds new documented public behavior

The exact version target can be chosen during implementation planning.

## Risks

### Risk: SCOP generalization destabilizes existing monotone behavior

Mitigation:

- preserve the current monotone fast path while adding second-order support
- add regression benchmarks and tests before broad refactoring

### Risk: postfit convex/concave repair becomes mathematically loose

Mitigation:

- define the repair contract strictly on the linear predictor scale
- test both repaired curvature and projection residuals

### Risk: generic shape abstractions add overhead everywhere

Mitigation:

- keep v1 narrow
- no mixed constraints
- no tensor generalization
- branch early and keep ordinary hot paths unchanged

### Risk: QP and SCOP semantics drift apart

Mitigation:

- unify public semantics
- document that engine choice is family-driven, not user-chosen
- keep acceptance tests aligned across dense/discrete and fixed/auto-lambda

## Acceptance Criteria

1. `Constraint.fit.convex` and `Constraint.fit.concave` are real supported
   public behaviors.
2. `Constraint.postfit.convex` and `Constraint.postfit.concave` are real
   supported public behaviors.
3. `PSpline` uses a true SCOP fit-time path for convex/concave.
4. `BSplineSmooth` and `CubicRegressionSpline` use projected QP fit-time paths
   for convex/concave.
5. `fit_reml()` works for constrained convex/concave terms.
6. `discrete=True` works for constrained convex/concave terms.
7. Existing unconstrained and monotone benchmarks do not regress.
8. New convex/concave paths have bounded, acceptable time-to-fit.
9. Docs and tests reflect the new constraint surface clearly.

## Recommended Next Step

Write an implementation plan that splits the work into:

1. shape-token and build-path generalization
2. QP convex/concave constraints
3. SCOP convex/concave reparameterization
4. postfit convex/concave repair
5. REML/discrete integration and benchmark verification
