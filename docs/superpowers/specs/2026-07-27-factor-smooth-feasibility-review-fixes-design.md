# Factor-Smooth Feasibility Review Fixes

## Context

PR #165 adds structured direct solvers for random effects and factor smooths. Review of
head `a4a6b21` identified four related problems in factor-smooth backend resolution:

1. `direct_solve="auto"` can scan all SZ rows even when the cost gate will select Gram.
2. Override feasibility replaces every positive row weight with one, so numerical rank
   can differ from the factorization built from the requested weights.
3. Non-override feasibility adds active penalty components without their lambda scales
   and caches only component names and positive-row support.
4. Structurally valid override blocks retain harmless floating-point asymmetry and can
   pass dispatch before failing the structured operator's strict symmetry check.

All four failures arise because backend resolution is not evaluating exactly the same
weighted, penalized local blocks that structured factorization will receive.

## Chosen Design

Keep the current single-dominant structured architecture and make backend resolution
faithful to its downstream algebra. This is a focused correction, not a solver redesign.

### Auto Cost Gate

For `direct_solve="auto"`, calculate the existing FS/SZ crossover decision after cheap
topology and override-geometry validation but before any row-dependent feasibility scan.
If the measured crossover selects Gram, return the existing crossover fallback
immediately. Forced structured mode continues through all feasibility checks so invalid
local geometry is rejected at the public boundary.

Cheap structural validation remains ahead of the cost gate because it does not scan
training rows and preserves explicit diagnostics for malformed authoritative overrides.

### Weighted Local Feasibility

Both override and lambda-defined feasibility checks will call
`FactorSmoothGroupMatrix.factor_smooth_sufficient_stats()` with the actual float64 row
weights. They will not convert positive weights to Boolean support.

The resulting information blocks will be combined with the exact local penalties used
by assembly. Their rank decision will retain the structured factorization policy:

- symmetrize the local blocks;
- use batched symmetric eigenvalues;
- scale the cutoff by each block's largest absolute eigenvalue with a floor of one for
  combined information;
- use `eps * block_width * scale * 10`.

This keeps early dispatch aligned with `SumToZeroBlockFactor` and prevents a selected
structured path from failing later solely because feasibility used different weights.

### Lambda-Defined Penalties and Cache Identity

Without `S_override`, construct the local penalty as the sum of
`lambda_for_component * omega` for every repeated component. Merely positive but
numerically negligible lambdas must not be treated as unit penalties.

The feasibility cache key will include:

- the ordered component suffix and resolved float64 lambda for every component;
- a digest of the exact contiguous float64 row-weight bytes.

The cached value therefore cannot survive a change in lambda scale or weight magnitude.
The cache remains local to the factor-smooth matrix and stores only a small key and the
first singular level.

### Authoritative Override Symmetry

After `_structured_override_incompatibility()` accepts FS or SZ geometry,
`_factor_smooth_override_local_blocks()` will symmetrize every extracted local block as
`0.5 * (block + block.T)`. Selection and assembly already share this extractor, so they
will consume identical blocks.

The structural validator remains responsible for rejecting material noncanonical or
cross-block mass. Symmetrization only canonicalizes opposite-triangle differences that
the scale-relative validation already classified as floating-point roundoff. It matches
the symmetric penalty contract used by the Gram path.

## Alternatives Rejected

### Catch Structured Factorization Failures and Retry Gram

This would avoid some preflight logic but would construct row moments and partially
factor an invalid system before falling back. It also makes exceptions normal control
flow and weakens forced-structured error locality.

### Introduce a Shared Retained Moment Object

Passing a new feasibility/moment object from dispatch into assembly could eliminate
more repeated work. It changes ownership and cache lifetimes across several solver
modules, which is disproportionate to these four review findings. That larger
optimization can be evaluated separately with profiles.

## Regression Coverage

Targeted tests will prove:

1. a below-crossover automatic SZ candidate never calls
   `factor_smooth_sufficient_stats()`, while forced structured still performs the
   feasibility check;
2. tiny nonzero identifying weights make a rank-deficient SZ override fall back under
   auto and reject early when forced, matching Gram behavior;
3. a tiny positive lambda is applied at its real scale, and changing lambda magnitude
   on the same matrix invalidates the feasibility cache;
4. a scale-relative, slightly asymmetric canonical FS/SZ override dispatches
   successfully and matches the Gram coefficients;
5. the existing valid/singular authoritative override and broad FS/SZ parity suites
   remain green.

## Performance and Scope

The auto change removes an unnecessary `O(n * k^2)` scan when Gram is already selected.
Full-rank authoritative override penalties continue to skip row feasibility. Cases that
require rank certification perform one compact sufficient-statistics scan and allocate
only `O(K * k^2)` local blocks.

No public API, penalty semantics, solver layout, Tabmat boundary, release metadata, or
LSS code changes are in scope.
