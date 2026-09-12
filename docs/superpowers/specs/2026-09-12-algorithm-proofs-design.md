# Proof programme for SuperGLM's distributional solver and PSST

Date: 2026-09-12. Status: proposed research scope.

The user selected proof planning after the LSS API refinement and Newton
completion repair, then included PSST's interpretation and calibration.
This document defines claims worth investigating; it does
not assert that the current implementation satisfies them.

The implementation baseline is master at
`c0ed3a6291213d5f6d2b969a51e3aabbaf4eb68b`, after merging PRs
[#386](https://github.com/StrudelDoodleS/superglm/pull/386) and
[#387](https://github.com/StrudelDoodleS/superglm/pull/387).
Its `src/superglm` Git tree is
`cb06e71ec9ae8dfda4d03c8cf7e0bd5ddd864c10`.
This matches the numerical source tree inspected during planning. Execution
must record its own revision and compare this source tree, since a proof about
an earlier algorithm does not cover later changes.

## Decision and scope

Build a collection of explicit claims about the implementation's departures
from the reference algorithms. Start with fixed compiled designs and smooth,
fixed-rank GaussianLS/GammaLS examples. Establish which arguments extend to
other families only after checking their regularity and numerical evaluation.

The first deliverable is a source-to-claim ledger, followed by assembly and
reuse arguments. Error bounds for strict stopping precede a theorem about the
combined EFS/Newton/BFGS controller. Exact penalty faces, changing numerical rank
and two-piece families require separate arguments.

PSST is a companion subproject in the same programme. Its first target is the
meaning of the local quadratic score, followed by Gaussian-reference moments
and the gap between that reference and the fitted-model sampling distribution.
It can proceed alongside the distributional assembly and reuse work.
The [PSST plan](../plans/2026-09-12-psst-proofs.md) keeps that execution scope
separate from the LSS controller proof.

A proof may reveal that code must change. Record a counterexample or restrict
the claim when assumptions fail; do not describe an idealized algorithm as a
proof of production code. Tests and numerical experiments support the mapping
from the argument to the implementation. They do not establish a theorem.

## Global constraints

- Target Python 3.12+; retain the existing NumPy/SciPy/Numba/tabmat CPU stack.
- No package-version, release-tag, or publication changes.
- This scope plans proofs; it does not authorize a new solver or public API.
- Keep exact-arithmetic, floating-point, optimization and statistical claims separate.
- Numerical tests use dimension-, epsilon-, norm- and conditioning-based bounds.
- Near-rank and cancellation fixtures check certification, refusal or stable observables.
- Adversarial regressions require an unfixed demonstration or a mutation check.
- Performance changes require complete-fit timing, peak RSS, numerical outputs and actual dispatch.
- A failed proof obligation may produce a counterexample or a narrower supported claim.

## Distributional mathematical target

Fix the response, likelihood weights, offsets, links, compiled predictor
matrices, constraints and penalty matrices. Eliminate fixed homogeneous
equality constraints in the coefficient coordinates and initially exclude
active inequality constraints. On this identifiable coefficient space and
one fixed-rank branch, let

\[
 Q(\beta,\rho)=-\ell(\beta)+\tfrac12\beta^\top S(\rho)\beta,
 \qquad
 S(\rho)=S_{\rm fixed}+\sum_j e^{\rho_j}S_j.
\]

The penalties are symmetric positive semidefinite. For positive finite
smoothing parameters their combined nullspace is fixed:

\[
 \ker S(\rho)=\ker S_{\rm fixed}\cap\bigcap_j\ker S_j.
\]

The individual penalties need not have identical nullspaces.
Here \(\rho\) contains only estimated, strictly positive, finite smoothing
parameters. Assume one continuously selected,
isolated coefficient solution
\(\hat\beta(\rho)\) of \(\nabla_\beta Q=0\), with positive-definite
\(H_\beta=\nabla_\beta^2Q\) on the identified space. Initially require \(Q\)
to be locally \(C^4\) for the classical profile derivatives; finite-difference
error bounds may require higher derivatives of the stenciled quantities.
The smooth-branch target, with determinants on that same coefficient space, is

\[
 F(\rho)=Q(\hat\beta(\rho),\rho)
       +\tfrac12\log\det H_\beta
       -\tfrac12\log\det{}^+ S(\rho).
\]

Match this expression to `joint_laplace_objective`, including its optimizing
likelihood and constant convention, before using it in a theorem.
The omitted constants do not change stationary points, but they do change a
stopping threshold proportional to \(1+|F|\). A local coefficient branch need
not be the globally best branch, and a Laplace criterion is itself an
approximation to an integrated likelihood.

The initial feasible set is a finite box \(C\) in the estimated \(\rho\)
coordinates. Fixed parameters are outside this optimization problem. A large
finite cap, an exactly infinite smoothing parameter and a heuristically frozen
coordinate are three different cases.

## Claims and source boundaries

All entries below start with status **proposed**.

| ID | Claim to investigate | Main implementation boundary | Deliverable |
| --- | --- | --- | --- |
| P1 | Dense, chunked and admitted grouped assembly represent the same fixed compiled model in exact arithmetic. | `solver/assembly.py`, `solver/chunks.py`, `solver/_global_moments.py`, `solver/_batched_moments.py` | Algebraic identities, admission assumptions and code mapping. |
| P2 | Floating-point assembly and solves have explicit error bounds or a defined refusal region. | The P1 modules, `solver/curvature.py`, `smoothing/penalty_geometry.py` | Forward/backward bounds with cancellation and rank qualifications. |
| P3 | Reused geometry and derivative state belongs to the requested model state; curvature-based decisions use valid provenance. | `solver/_reuse_digest.py`, `solver/solver.py`, `smoothing/derivatives.py`, `smoothing/newton.py` | State invariants, invalidation cases and regression witnesses. |
| P4 | Newton's `stationary` outcome bounds a residual for the exact profiled objective on the stated branch. | `smoothing/endpoint_direction.py`, `smoothing/derivatives.py`, `reml/convergence.py` | An error budget and a conditional residual bound. |
| P5 | A precisely specified safeguarded controller approaches first-order stationarity under verified assumptions. | `smoothing/newton.py`, `smoothing/loop.py`, `smoothing/objective.py` | A conditional theorem and a list of code obligations. |
| P6 | Rank changes, exact penalty faces and piecewise-smooth families have appropriate separate contracts. | `smoothing/penalty_face.py`, `smoothing/endpoint_laml.py`, `smoothing/faces.py`, `kernels/two_piece.py` | Applicability results, counterexamples and separate follow-up scopes. |
| P7 | PSST's local fitting interpretation, reference moments, calibration assumptions and ranking objective are explicit. | `screening/_pair_factor.py`, `screening/_score_stat.py`, `screening/_structured.py`, `model/screening_ops.py` | Component arguments, normalization repair scope, sampling counterexamples and an evaluation design. |

Paths in this table are relative to `src/superglm/distributional/`, except
`reml/convergence.py` and all P7 paths, which are relative to `src/superglm/`.
Shared scalar algebra may acquire a claim when the same assumptions and source
actually apply. This is not a proof of the entire scalar optimizer.

### P1: representation equivalence

For a row partition \(c\), predictor blocks \(a,b\), and diagonal row-weight
blocks \(W_{ab}\), derive

\[
 X_a^\top W_{ab}X_b
 =\sum_c X_{a,c}^\top W_{ab,c}X_{b,c}.
\]

Derive the grouped identity by collecting rows with identical relevant compiled
values, or by specifying the factorization that recovers those values from
group indices and retained row factors. Equal sparse support indices alone
are insufficient when their nonzero values differ. Cross-predictor observed
weights may be signed. Cover ordinary columns, intercepts, interactions, constraints
and likelihood weights, with each penalty added once.

This concerns the same compiled design. It does not establish bitwise equality
between reduction orders or bound the approximation introduced by quantizing
continuous covariates. Document every admitted grouped layout and keep fallback
correctness separate from dispatch and performance.

### P2: floating-point bounds

Use an explicit arithmetic model. For example, a serial sum with unit roundoff
\(u\), \(nu<1\), and no exceptional-range arithmetic satisfies an absolute
bound involving \(\gamma_n=nu/(1-nu)\) times the sum of absolute operands.
Use the actual reduction graph for parallel/native paths and include product,
histogram and contraction errors. A BLAS call needs a stated implementation
assumption or a separately checked residual; its name is not an error bound.

Cancellation makes a bound relative to the final small signed sum unsuitable.
The global-moment operand admission range does not by itself bound a histogram
sum. Account for repeated addition, overflow/underflow, matrix norms and rank
decisions. Residual/backward accuracy and coefficient-forward accuracy have
different conditioning requirements.

### P3: reuse and current-state invariants

List the state consumed by every reuse decision: coefficient/predictor values,
family and likelihood plan, weights and offsets, penalty/lambdas, compiled
layout, rank/face context, derivative step and ownership/mutation assumptions.
The exact key can differ by cached quantity; prove why omitted state is
irrelevant to that quantity.

Digest equality is not logical equality. State the collision assumption and
array mutation rules wherever a digest forms part of the guard. Separate
immutable structural reuse, data-curvature reuse under a changed penalty and
derivative reuse at an unchanged fit.

For the Newton completion repair, investigate the narrower property that an
old negative-curvature verdict cannot indefinitely prevent completion solely
because it is old, once the other convergence gates pass and the budget permits
a fresh evaluation. Do not claim this refresh makes the objective convex or
guarantees that the next Hessian is positive definite.

### P4: strict stopping and derivative error

Initially analyse Newton's `stationary` outcome. Strict EFS may instead stop
at `lambda_change` or `objective_plateau`; its fixed-point residual is not
the full LAML gradient. Extending P4 to those outcomes requires a separate
conversion argument.

For an interior point, a valid enclosure
\(\|\nabla F-\hat g\|_\infty\leq\delta\) gives

\[
 \|\nabla F\|_\infty\leq\|\hat g\|_\infty+\delta.
\]

For the finite box, use the projected residual

\[
 R(\rho)=\rho-\Pi_C(\rho-\nabla F(\rho)),\qquad
 \hat R(\rho)=\rho-\Pi_C(\rho-\hat g(\rho)).
\]

Projection onto a box is nonexpansive in the infinity norm, so
\(\|R\|_\infty\leq\|\hat R\|_\infty+\delta\).
Choose and record the coordinate scaling; this is a stationarity measure for
the declared constrained problem, not an unconstrained or infinity-face claim.

The production Newton stop tests a sign-projected score after heuristic
freezing, not \(\hat R\) directly. Its exact-bound projection uses
`bound_window=0.0`. Derive the bridge to the displayed residual using both the
active score bar and the current-state recheck on frozen coordinates. With
\(t=\mathtt{reml\_tol}\), the computed objective \(\hat F\), and

\[
 \tau=t(1+|\hat F|),\qquad
 \phi=\max(0.1t,10^{-7})(1+|\hat F|),
\]

the proposed implementation bound is
\(\|\hat R\|_\infty\leq\max(\tau,\phi)\), hence
\(\|R\|_\infty\leq\max(\tau,\phi)+\delta\) under the gradient enclosure.
Tightening `reml_tol` does not remove the freezing floor. Verify every
projection/masking path and its arithmetic before treating this as an
implementation theorem.

The triangle inequality is straightforward. Establishing \(\delta\) is the
substantial work: coefficient residual and local conditioning, linear solves,
implicit differentiation, family evaluation, finite-difference truncation,
roundoff, and penalty/log-determinant derivatives all contribute.
The coefficient residual-to-solution argument needs a certified neighborhood
or another explicit local regularity condition.

The current finite-difference refinement indicators do not enclose all these
errors. In particular, agreement between two stencils is not by itself a
truncation bound. Checking both an estimated gradient and an error indicator
against the same tolerance also does not bound their sum by that tolerance.
Preserve the existing numerical meaning of `smoothing_certified_` while the
stronger claim remains unproved.

### P5: a conditional controller theorem

The initial target is deliberately narrower than the current default policy:
a smooth, fixed-rank coefficient branch, finite smoothing box and a continued
iteration sequence with practical and finite-tolerance success exits disabled.
Treat refusal exits separately. An unbounded iteration budget alone does not
prevent finite-tolerance termination.

Seek a proof under explicit assumptions: a compact set containing the entire
sequence and its accumulation points within the branch's regular domain,
a Lipschitz profile gradient, uniformly nonsingular coefficient curvature, feasible
gradient-related steps, controlled inexact derivatives/objective evaluations,
and an acceptance rule yielding, in the Euclidean norm,

\[
 F(\rho_{k+1})\leq F(\rho_k)-c\|R(\rho_k)\|^2+e_k,
 \qquad c>0,\quad e_k\geq0,\quad \sum_k e_k<\infty.
\]

With a lower bound for \(F\), this implies \(R(\rho_k)\to0\); accumulation
points satisfy the box first-order conditions when the residual is continuous.
The regularity set must allow for positive slack: for example, use the
inflated sublevel set \(F\leq F(\rho_0)+\sum_k e_k\), or the whole finite box.
The research task is to derive the displayed decrease from the actual
controller, or identify the bounded changes needed to do so. Assuming that
inequality is not a convergence proof of the implementation.
P4's terminal error enclosure alone does not control errors at all iterations
and line-search trials; P5 needs those bounds wherever its decrease argument
uses them.

Audit these unresolved obligations explicitly:

- A coefficient solver's convergence flag does not supply the required profile
  derivative error bound or a shrinking inexactness forcing sequence.
- Positive-definite BFGS memory alone gives no uniform spectral bounds.
- Generalized Fellner--Schall proposals need assumptions of their own; in the
  general non-Gaussian case they need not be descent directions for the full
  profiled Laplace objective.
- Mixing EFS coordinates with another direction, clipping to bounds and using
  a pre-clipping directional product need a feasible-step argument.
- Fixed objective slack, derivative floors and heuristic coordinate freezing
  cannot silently become summable errors or exact stationarity.
- Rank/face changes must preserve the theorem's assumptions or leave its scope.

Fixed numerical tolerances can support a quantified residual neighborhood,
provided the required bounds are proved. A finite iteration limit can return
budget exhaustion. Practical plateau stopping establishes its stated stability
checks; it is not the asymptotic theorem above.

### P6: boundaries and nonsmooth cases

For exact infinite penalties, derive the limiting coefficient subspace and
Laplace normalization under fixed nullspace/rank assumptions. Do not substitute
a large finite lambda for that limit. Check transitions and covariance claims
separately from stationary points on a fixed face.

Two-piece likelihoods can be continuously differentiable with a Hessian jump
when a row crosses the join. The profiled log-determinant term can then lose
continuity; finite differences do not restore smoothness. Establish the actual
regularity before choosing a generalized derivative theorem or narrowing the
supported regime. Support boundaries and family-specific series evaluation
errors require their own analysis.

### P7: PSST interpretation and calibration

PSST screens candidate terms against one fitted scalar model. With 30 eligible
features there are at most 435 pairs to assess, subject to supported kinds and
resource limits. It does not enumerate subsets of interactions or solve the
distributional structure-discovery problem in C9.

Split P7 into four independently reviewable statements.

**P7.S1: the local fitting problem.** On the identified candidate space, let
\(U\) be the working score and \(V\) the Fisher working curvature after
profiling unpenalized adjustments to the intercept and the pair's two margin
blocks. Measure gain relative to that nuisance-only profiled optimum, rather
than the unchanged fitted coefficients. For fixed \(\lambda\ge0\), penalty
\(S\succeq0\), and positive-definite \(V+\lambda S\), derive

\[
 q_\lambda(b)=U^\top b-\tfrac12b^\top(V+\lambda S)b,\qquad
 \max_b q_\lambda(b)=\tfrac12U^\top(V+\lambda S)^{-1}U=\tfrac12T_\lambda.
\]

Here \(q_\lambda(b)=\widetilde q_\lambda(b)-\widetilde q_\lambda(0)\),
where \(\widetilde q_\lambda\) is the profiled Fisher working objective.
This is an exact statement about that working quadratic.
Its relationship to a complete likelihood refit is an approximation.
The screen does not jointly
reoptimize all mains coefficients, working weights and smoothing parameters
for each candidate. Its overlap projection is not projection against the
whole fitted nuisance space.

Map compiled term spans, centering, score scaling and factor assembly to this
identity. Record quantization and `approx` separately. Use the same retained
space and penalty conventions in the dense and structured routes.

**P7.S2: Gaussian-reference moments and ladder selection.** Under the explicit
assumption \(U\sim N(0,\phi V)\), with fixed geometry, known dispersion and
score-independent ladder choices, derive

\[
 T_\lambda/\phi\ \overset d=\ \sum_j a_{j\lambda}Z_j^2,\quad
 E(T_\lambda/\phi)=\sum_j a_{j\lambda},\quad
 \operatorname{Var}(T_\lambda/\phi)=2\sum_j a_{j\lambda}^2.
\]

For the stored pencil, \(a_j=v_j/(v_j+\lambda s_j)\), with
\(0\le a_j\le1\) for \(S\succeq0\) and \(\lambda\ge0\).
The current ranking denominator uses \(2\sum_j a_j\) under its square root.
The intended first corrective scope is to
establish and implement the appropriate Gaussian-reference normalization on
both dense and structured paths, with regression and ranking evidence.
The model-level `edf1` matrix is not the candidate's matrix.

Gaussian ladder simulation must use \(u_j^*=\sqrt{\phi v_j}Z_j\), sharing the
same normal draw across rungs. Unit variance alone does not equalize tails,
and a maximum over rungs needs its joint reference distribution. The structured
path does not expose one global GSVD, so its arithmetic and cost need a
separate derivation. Preserve duplicate-rung and numerical-refusal behavior.

**P7.S3: the fitted-model null.** Establish when the Gaussian score assumption
is justified. In common whitened coordinates, let
\(\widetilde y=X_0\beta_0+\sqrt\phi\,\varepsilon\), with
\(\varepsilon\sim N(0,I)\), fixed baseline smoother \(H_0\), fixed nuisance
design \(X_0\), and fixed candidate design \(A\) after pair projection. Analyse

\[
 E(U)=A^\top(I-H_0)X_0\beta_0,\qquad
 \operatorname{Cov}(U)=\phi A^\top(I-H_0)(I-H_0)^\top A.
\]

These generally differ from zero and \(\phi A^\top A\). Conditioning on a
response-estimated smoother does not establish this fixed-smoother model.
Full nuisance projection must also update the candidate curvature.
Include shrinkage bias, estimated smoothing, estimated dispersion,
non-Gaussian higher moments and sparse cells in the applicability analysis.
Conditioning on fitted Fisher weights does not make scores independent normal
variables. A parametric bootstrap must repeat the relevant fitting and screening
steps; its validity is an additional claim, not an automatic consequence of
simulation.

Define whether a null concerns a wholly additive model or one pair in the
presence of other interactions. Require valid marginal p-values before any
multiple-testing claim. Bonferroni/Holm, ordinary BH and joint maximum
calibration have different dependence requirements. Monte Carlo resolution,
including the \(1/(B+1)\) minimum for the usual exchangeable-rank construction,
must match the intended tail threshold. Retain ranking-only semantics until
the proposed stronger contract has an argument and enforcement.

**P7.S4: predictive ranking and provenance.** Define the intended objective
before changing the ranking: evidence against a null, expected held-out gain,
or expected gain per refit cost. They are different quantities. The guide's
43-versus-73 deviance example uses training gain; it does not identify which
pair offers greater predictive return. Small p-values do not answer that
question either.

Compare current ranking, reference-variance normalization and FAST on the same
baseline, candidate set, data split and refit classes. Refit every candidate,
report held-out gain and cost, and use repeated data/split evidence as well as
a wider candidate set. Overlapping pairs are not independent replicates.
Retain null and signal cases, correlated mains, thin cells and mixed kinds.

The theory has established antecedents. Rao score testing supplies the
unpenalized quadratic form; smooth and variance-component score methods supply
related testing constructions. These are not automatically the same statistic
or nuisance treatment. The original FAST scheme ranks a four-region step
approximation using cumulative tables. PSST combines term-specific candidate
spaces with a screening-EDF ladder and stable factor arithmetic. Mapping that
combination to the literature is required before any originality claim.

References for P7:

- [Lou et al. (2013), FAST and GA2M](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf).
- [Zhang and Lin (2003), smooth-component testing](https://doi.org/10.1093/biostatistics/4.1.57).
- [Lin et al. (2013), GESAT interaction testing](https://pubmed.ncbi.nlm.nih.gov/23462021/).
- [Phipson and Smyth, Monte Carlo p-values](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).
- [Wang, BH and BY dependence results](https://arxiv.org/abs/2201.09350).

## Evidence and review

Each claim record has: ID, status, precise statement, assumptions, pinned source
symbols, argument or counterexample, numerical obligations, regression evidence,
and review disposition. Allowed statuses are `proposed`, `proved-conditional`,
`refuted`, `unsupported` and `empirical-only`. A conditional proof must say
which assumptions the implementation checks and which it does not.

Require a mathematical review of the argument and a source review of its
mapping. Neither model-generated prose nor passing tests is sufficient review
evidence by itself. Record unresolved objections. New runtime certification
claims need both reviews and executable enforcement of their required bounds.

Global optimality, uniqueness across branches, statistical consistency,
uncertainty coverage, discretization rates and formal verification of the
Python/NumPy/LAPACK stack are outside this initial programme. They must not be
inferred from a successful stationarity result.

## Reference obligations

- [Wood, Pya and Säfken, generalized smooth models](https://arxiv.org/pdf/1511.03864):
  map the Laplace objective, implicit derivatives and outer Newton safeguards
  to the implementation. Their algorithm is the starting point, not a proof
  of every subsequent modification.
- [Wood and Fasiolo, generalized Fellner--Schall](https://arxiv.org/pdf/1606.04802):
  extract the definiteness, fixed-nullspace and step-control assumptions;
  distinguish the non-Gaussian approximation from an exact profile gradient.
- [Dembo, Eisenstat and Steihaug, inexact Newton methods](https://epubs.siam.org/doi/10.1137/0719025):
  use explicit residual forcing conditions as a reference for local
  inexactness analysis. Their theorem does not directly cover this hybrid.

The [execution plan](../plans/2026-09-12-algorithm-proofs.md) orders the
deliverables and records the tests and review gates for each.
