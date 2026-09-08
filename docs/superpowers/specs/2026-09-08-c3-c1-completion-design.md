# Remaining C3 and C1 capabilities

The user explicitly selected C3 + C1 and authorized an audit, staged plan,
implementation, and complete-fit validation. This supersedes the draft C5-before-C1
ordering. Work starts at `8962c4520cad948aa20c480a238b7bb1e276e9cd`, matching
published v0.31.0 (GitHub release and PyPI checked 2026-09-08).

## Scope and preserved inputs

Implement trustworthy, usable grouped/discrete distributional fitting on CPU,
including observed-curvature families, smoothing certification, and uncertainty.
Close the two named convergence questions through reproduced evidence and focused
fixes where warranted. The exit gate allows a reproducible, mathematically
explained limitation; it does not permit relabeling a plateau as convergence.

Read and preserve the uncommitted documents in `.worktrees/roadmap-dossier`.
Their copies travel with this work; update this worktree's roadmap as facts change.
Source SHA256 receipts:

- AGENTS.md: `be7f17b8f0d16d9a7c3aac65a832fea0e9750d7a77ce27988747f220a221bbc8`
- docs/ROADMAP.md: `fa6dfd41c520960650c57e1047abf76920afef59f838e4b830361cea600df51f`
- dossier: `aa9edb8abf547e362274dec6075e7ee121474fd1e8f099c5155c95951a4c5a94`

## Audit and design decision

Preserve the existing coupled coefficient solver, safeguarded EFS/Newton/BFGS,
rank policy, exact penalty faces, grouped assembly and rectangular cross-products.
Replacing those with a new discrete subsystem or optimizer would duplicate working
architecture. Merely removing the public refusal would leave dense allocations in
certification and would exclude Tweedie/NB2 from chunked fitting.

Promote the existing grouped execution route. Keep the Fisher capability check
only where Fisher information is requested or actually needed. Signed observed
cross-blocks must retain their exact geometry. Use bounded row designs or grouped
operations for all LAML derivative and posterior replay paths. Reuse immutable
row-layout information where repeated factor-smooth chunk extraction currently
sorts full parent rows. Keep result/certification history semantics intact.

## Mathematical and production contract

- Target Python 3.12+; use the existing NumPy/SciPy/Numba/tabmat CPU stack.
- No changes to package versions, release tags, or publication state.
- Exact representation tests compare the SAME constructed design, response,
  weights, offsets, penalties and constraints. Tolerances follow dimensions,
  floating-point epsilon, norms and conditioning; test stable observables.
- Quantized covariates change the statistical design. Report dense versus
  discrete and finer-grid sensitivity separately from representation equivalence.
- Certification authority, practical stops, likelihood/weight semantics and
  refusal behavior remain explicit. No invented expected-information fallback.
- Preserve joint covariance and prediction semantics; include smoothing fits,
  Newton certification, infinity faces and optional smoothing uncertainty.
- Memory claims cover actual allocations and complete-fit RSS. Retained fit
  histories and coefficient-space dense factors remain explicit limitations.

## Acceptance evidence

Recover the exact #376 synthetic fixtures from the preserved math-audit worktree:
correlated Tweedie n=10,000, correlation mixture .75, q=192; GPD threshold excess
fixture n=1,401, q=98. Reproduce them on frozen v0.31.0 and the implementation,
check independent derivative/reference evidence and stable predictions across
starts where a fit is claimed. Do not infer an unbounded GPD likelihood from a
smoothing cap; its current shape link restricts shape to (0,1).

A subprocess benchmark harness uses frozen baseline source and current source
with identical inputs and environment. Record complete-fit elapsed time, peak
RSS, stopping status, objective, coefficients/predictions/covariance, smoothing
parameters, work counts and actual backend dispatch. Separate instrumentation
from timing. Measure serially on a quiet machine, reporting environmental load.
Demonstrate a reliable distributional model on real freMTPL2 plus a model-size
sweep or synthetic insurance book that establishes practical memory/time benefit.
Do not claim universal sub-second or 10^8-row capability.

Run focused mathematical regressions with demonstrated baseline failure or a
mutation, then ordinary repository checks and data-required suites. Record
pre-existing failures independently. Keep the plan, roadmap and final evidence
report synchronized with what was actually established.

## Established results

The implemented route retains the architecture selected above. Existing strict
EFS plus Newton resolves both named C3 trajectories under the existing
stationarity contract; no production C3 solver change was justified. Public
observed chunking, bounded derivative/posterior design replay and cached
categorical row lookup are covered by focused mathematical and mutation checks.

The [completion report](../../research/2026-09-c3-c1-completion-evidence.md)
separates the real freMTPL2 book, synthetic replication for scaling, recovered
synthetic stress fixtures, exact representation checks and continuous-grid
approximation. The positive memory result uses 449,000 replicated severity rows;
the smaller real book saves no RSS, and both versions refuse certification of
the tested large real-book NB2 smoothing fit.

The existing finite-difference refinement indicator is not a rigorous bound on
all coefficient-mode, linear-solve, truncation and roundoff error. Passing the
existing objective-scaled first-order stationarity checks does not certify a
local or global minimum. The [C3 evidence](../../research/2026-09-c3-stress-evidence.md)
records terminal authority, independent reference checks, sensitivity across
starts and unresolved probes without strengthening that contract.
