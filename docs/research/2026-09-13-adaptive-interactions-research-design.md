# Adaptive interactions and coefficient scaling research

Date: 2026-09-13

Base: `7c4e70ffac99c9a70adf90e5b21d9735109c4675`, `v0.33.0`

Worktree: `.worktrees/adaptive-interactions`

Status: research direction approved; the first tensor-support optimization is
implemented and measured in `306f12e0`. Adaptive guarantees require the gates below.

Mathematical details belong in the companion
[representation memo](2026-09-13-adaptive-interaction-representation.md) and
[Gaussian error-bound memo](2026-09-13-adaptive-gaussian-error-bounds.md).
The [cost audit](2026-09-13-adaptive-interaction-costs.md) records the source-level
allocation and reuse analysis.
The [performance follow-through](2026-09-13-tensor-support-handoff-performance.md)
records the first implemented change, numerical checks and memory tradeoffs.
The [tool setup](research-tools-setup.md) records the installed Lean and SymPy tools.
Keep future derivations, counterexamples and formal proof scope under
`docs/research` with their assumptions, source revisions and references.

## Objective and scope

Improve predictive accuracy per unit of complete-fit time and peak process RSS
by allocating interaction resolution where it is useful. Retain all observations.
Investigate both better representations and cheaper coefficient algebra, with
error bounds derived for the actual compiled problem and executed arithmetic.
Measure their composition instead of assuming the gains multiply.

The user selected a locally adaptive two-dimensional spline experiment, with a
fixed Gaussian problem as the first numerical validation and automatic smoothing
plus fresh held-out evaluation as the modelling test. The fixed rich model is an
engineering reference; matching it does not establish population accuracy.
Use `gpt-6-astra` at `max` for mathematical agents and no more than three concurrent
subagents. Keep benchmark workers sequential, separate from other heavy work.

This promotes bounded C16/C21 work and the necessary C15/C18 analysis. The proof
programme supplies requirements for these claims. C9 integration follows an
affordable, validated representation. Coupled LSS curvature, general structure
search, and 100-million-row/out-of-core fitting have separate scopes.

## First performance investigation

**Completed first step.** The optimizer-to-finalizer handoff now reduces the
expensive tensor support constructions from two to one while preserving the
selected numerical support and error ledger. Matched complete-fit medians
improve by 34.0% (rows20) and 44.8% (rows30); all recorded non-timing numerical
outputs match and retained model payload is unchanged. Small-case peak RSS
remains 2.4% higher, while large-case ranges overlap baseline. See the
[final report and receipt](2026-09-13-tensor-support-handoff-performance.md).
This removes repeated work without changing dense asymptotic scaling.

### Preserved baseline investigation at 7c4e70ff

The following records the investigation before implementation. Its proposed
handoff is now implemented as described above.

The committed housing runner supplies a frozen input/model workload, one owned
worker per invocation, a 180-second whole-worker deadline, source/input hashes,
actual dispatch and numerical comparisons. The
[measurement receipt](../../benchmarks/adaptive_interactions_baseline.json)
records runtime, hashes, numerical outputs and profile callers. The initial
serial measurements are:

| Case | Coefficients excluding intercept | Complete fit, seconds | Peak process RSS, MiB | REML iterations |
| --- | ---: | ---: | ---: | ---: |
| `rows20` | 513 | 10.053859 | 565.039 | 9 |
| `rows30` | 1,013 | 117.279187 | 900.609 | 10 |

Both runs converged at `score_objective_tolerance`, used the `gram` direct
backend with eight `DiscretizedSSPGroupMatrix` groups and one
`DiscretizedTensorGroupMatrix`, and matched committed validation/test predictions
exactly. These are one observation per case, not a speedup measurement or an
estimate of run variability. No implementation changed between the reference
checkout and these runs. Exact local replay does not define a portable tolerance.

The fit clock includes `fit_reml` construction, optimization and finalization.
It excludes imports, data loading and later prediction/export. Process RSS also
covers that later work and the runtime. The `rows30` optimizer timer recorded
61.956480 seconds. The separate `rows30` profile completed in 116.541042 seconds
and attributes the remaining cost:

| Profile entry | Calls | Cumulative seconds |
| --- | ---: | ---: |
| `_penalty_support` | 98 | 103.592 |
| `compute_total_penalty_rank` | 1 | 51.601 |
| `compute_penalty_nullity` | 22 | 51.737 |
| `finalize_reml_fit` | 1 | 55.433 |

These entries overlap. The rank consumers account for expensive support
preparation on optimizer and finalization paths. Source and call counts identify
96 singleton supports and two two-component tensor supports, not 98 expensive
tensor decompositions. Profiled clocks are diagnostic, not a timing comparison
with the unprofiled observations.

The first investigation ended with these baselines, a complete-fit profile,
producer/consumer analysis and one bounded optimization hypothesis. Any patch
must then demonstrate a complete-fit benefit with numerical evidence and actual
dispatch, including retained-memory and lifetime costs. A support-cache change
can remove repeated work without changing the asymptotic cost of its first build.

The measured first hypothesis is to carry the optimizer-owned positive tensor
penalty family into finalization. `reml/discrete.py:446` first builds its support
for bootstrap rank and retains that geometry across lambda iterations. The
discrete result at that base did not return that family; `fit_ops.py:2020` passed
the original entry family to finalization. `reml_finalize.py:404` built another
context, whose terminal objective requests nullity and builds support again.
Seeding from the entry family alone would miss the expensive producer. Existing
raw-family support transfer explicitly excludes discrete tensors; its other
supported cases are already implemented.

A bounded patch would transfer the admitted tensor family through the existing
result mechanism or a fit-local owner and verify identical basis, coordinate map,
component order/placement, penalty values, dtype and arithmetic/rank policy.
Weighted summaries have a separate lambda-key lifetime. Keep changed zero faces
outside the first positive-family transfer unless separately justified. Count
snapshots and overlapping contexts in retained and peak memory. No measured
speedup or production change had been established at that investigation stage.

## Representation contract

The [representation memo](2026-09-13-adaptive-interaction-representation.md)
defines the proposed construction and its proof obligations. Start with bicubic
THB splines on a fixed rectangle, an isotropic dyadic admissible hierarchy and
normalized Lebesgue centering measures. Freeze physical scaling, boundary and
extrapolation rules, knots and the maximum fine space. Product-centering the raw
hierarchy gives nested interaction spaces without requiring whole marginal
strips of additional fitted coefficients.

Apply a local raw function minus its two marginal means plus its overall mean.
The means must use probability measures. This operator preserves nestedness,
but its output columns can be a redundant frame. Their kernel is precisely the
raw hierarchy's additive functions; it can change with refinement. Product
centering does not imply sample orthogonality on correlated observed inputs.

Keep the physical penalty fixed:

\[
S_{\max}=\lambda_xK_x\otimes M_y+\lambda_yM_x\otimes K_y,\qquad
\lambda_x,\lambda_y>0.
\]

Here the mass matrices integrate basis products. The active penalty is the
pullback through the full centering/refinement map, separately for each
component. Fixed maps must preserve functions and penalties across refinement.
Define the injection into independent rich coordinates before invoking any
positive-definite rich Hessian; the raw centered frame itself is redundant.

For the first pilot, solve the consistent PSD frame system without an artificial
ridge and certify the lifted candidate independently. Audit rank on small
fixtures using the cellwise mixed-derivative map: its kernel is the additive
functions on a connected rectangle. This does not establish a cheap production
rank factorization or finite-precision Krylov convergence.

The proposed centered operator combines local THB evaluations and marginal
piecewise-cubic corrections accumulated over leaf cells in 1D interval trees.
Let N be leaf count, L depth, p raw active width and s raw overlap. A conservative
work target is O(n(s+1)+Ns+NL+p), with O(p+Ns+NL) hierarchy storage plus row state.
Record N independently of p. The published raw-overlap/mesh-closure results do
not prove these new centered-operator costs. Signed marginal corrections in the
physical penalty require cancellation-aware arithmetic bounds.

The housing tensor is a comparator, not this fixed rich target. It uses cardinal
natural cubic marginals, independently selected quantile knots and normalized
penalties with identity factors. Build matched uniform coarse/fine references
for the new hierarchy before comparing it with housing. An equivalence claim
requires a map that preserves boundary, centering and penalty semantics.

A concrete integration trap is cubic mass quadrature. Degree-six basis products
need four Gauss points per span or exact polynomial integration. Calling the
current derivative-penalty helper at order zero gives only three points and
would underintegrate the new mass matrix.

## Fixed Gaussian error analysis

The [Gaussian memo](2026-09-13-adaptive-gaussian-error-bounds.md) contains the
derivations, adversarial examples and formal-verification scope. The following
summary is an exact-arithmetic statement, not a completed floating-point
certificate.

Freeze the compiled target, its rank and scale conventions. Require W and S
positive semidefinite, and H positive definite in the already identifiable rich
coordinates:

\[
F(\beta)=\tfrac12\|W^{1/2}(y-X\beta)\|^2+\tfrac12\beta^TS\beta,\quad
H=X^TWX+S,\quad b=X^TWy.
\]

For any lifted approximate fit, including an inexact adaptive solve, put
r=b-H beta_tilde. With e=beta_star-beta_tilde,

\[
E^2=e^THe=r^TH^{-1}r,\qquad
F(\widetilde\beta)-F(\beta^\star)=E^2/2.
\]

Readjust mains and nuisance effects jointly. Their residuals vanish only at an
exact active optimum. Individual omitted-block scores do not sum to the full
remaining error.

The reviewed penalty-nullspace construction gives a positive-definite lower
operator G in transformed coordinates with G <= T.T H T. It uses a small
data-supported penalty-nullspace Gram and the inverse of the positive penalty
complement, retaining nuisance cross terms. Generalized modal coordinates are
valid through congruence; the residual transforms as T.T r. A full virtual
residual scan is still needed. Weak penalties can make this upper bound
arbitrarily pessimistic.

For an operational floating-point certificate, establish

\[
T^THT\succeq\alpha\widehat G,\quad \alpha>0,\quad\widehat G\succ0,
\quad d_{\rm up}\ge\|\widehat r_c\|_{\widehat G^{-1}},
\quad\zeta\ge\|r_c-\widehat r_c\|_{\widehat G^{-1}}.
\]

Then inverse order and the triangle inequality give

\[
E\le E_{\rm up}=(d_{\rm up}+\zeta)/\sqrt{\alpha}.
\]

This separates geometry/factorization evidence, residual error and norm
evaluation. Approximate nullspace defects must enter the enclosure. Computed
positive eigenvalues or the smallest Ritz value alone do not prove the required
lower bound. Include the actual reduction, transform, projection and inverse
errors, and outward control of the envelope computation.

For any linear prediction functional l, |l.T e| <= E sqrt(l.T H^-1 l).
With fixed phi>0, E/sqrt(phi) bounds numerical mean displacement in units of
the fixed rich covariance phi H^-1. This neither computes those standard
deviations nor establishes population accuracy.

Even a zero mean residual leaves possible omitted covariance, EDF and
trace/determinant contributions. The rich-minus-reduced covariance is PSD and
can remain nonzero. A response-adaptive numerical subspace can approximate a
fixed target, but ordinary inference for a newly selected statistical model
requires a separate argument.

The acceptable numerical approximation is an application accuracy budget.
Roundoff limits derive from audited arithmetic, dimensions, norms and
conditioning. Refuse certification if target identity, rank separation or the
error enclosure is unresolved. The compiled-target certificate does not include
row discretization or continuous-basis approximation unless separate perturbation
bounds account for them.

## Automatic smoothing and selection

Re-estimating smoothing changes the reference Hessian; profiling dispersion
changes objective/uncertainty conventions. A fixed-parameter mean certificate
does not certify the REML optimum. The next theoretical target is Gaussian REML
on a fixed-rank declared parameter neighbourhood, with errors in solves,
quadratic terms, traces, log determinants and the outer stopping decision.

For stochastic quantities, separate probability/confidence from deterministic
rounding and solve error. Account for repeated adaptive evaluations. Common
random probes can help comparisons but do not prove convergence. Local outer
error propagation requires established curvature and neighbourhood assumptions.
No global LSS or post-selection interval theorem is claimed.

Use a pilot with exact small reference quantities to test these errors before
introducing stochastic or iterative substitutes. Fresh held-out accuracy is the
empirical modelling criterion. The historical housing split, already inspected
during model selection, remains a numerical/performance workload.

## Complete-fit complexity and memory

Report row count `n`, active coefficient count `p`, virtual fine count `P`,
smoothing-parameter count `q`, hierarchy levels/closure, and actual local overlap.
Let `z` be the sum of row overlaps, `o` the sum of squared row overlaps, `e_S`
the stored penalty nonzeros and `c_T` the coordinate-transform application cost.
Under a genuinely sparse/structured representation one Hessian action costs
`O(n + p + z + e_S + c_T)`, plus any centered marginal/leaf work above.
Sparse Gram assembly involves `O(n + p + o)` work before
fill and coordinate changes. Dense coefficient factorization retains quadratic
storage and cubic work even when the raw design is sparse.

Sum construction, preconditioner setups, all coefficient iterations, all outer
evaluations and line-search trials, trace/logdet work, candidate searches,
failed proposals, refinement/closure, certification, finalization and requested
inference. Krylov iteration counts depend on conditioning and requested error;
probe counts depend on accuracy and confidence. Neither is a free constant.

Let `A` denote one Hessian action and `C_K` the cost of a Krylov iteration,
including that action, preconditioner application and vector/orthogonalization
work. Let `K` be the required Krylov iterations, `b_g/b_h/b_log` the respective
probe counts, `ell` Lanczos steps and `C_S` a component-penalty action. For
fixed-weight Gaussian fits, charge the following conditional costs:

| Operation | Work to charge |
| --- | --- |
| Coefficient solve | Preconditioner setup/update plus `K C_K` per solve |
| Gradient traces | `b_g K C_K + b_g q C_S` |
| Full outer-Hessian trace construction | `b_h (q+1) K C_K + b_h q C_S + b_h q^2 p` for a shared-probe construction |
| Log determinant | `b_log ell A`, plus vector work and reorthogonalization if used |
| Outer Newton solve | `q^3`, with `q^2` storage |
| Full covariance output | At least `p^2` output entries, plus computation |

These are algorithm-dependent work models, not bounds on convergence iterations.
General GLMs introduce weight-derivative work. Summing only gradient trace costs
would undercount a full smoothing Hessian. With refinement stages `r` and every
attempted outer evaluation `j`, the total is

\[
T_{\rm complete}=\sum_r\left[C_{{\rm build},r}+C_{{\rm search},r}
+C_{{\rm cert},r}+\sum_j(C_{{\rm coefficients},rj}+C_{{\rm outer},rj})\right]
+C_{\rm terminal}+C_{\rm requested}.
\]

Keep virtual-space certificate costs explicit. For tensor marginal widths
`p_x,p_y`, `P = p_x p_y`, generalized marginal decompositions can cost
`O(p_x^3+p_y^3)` setup and `O(P(p_x+p_y))` per separable inverse application,
with `O(P+p_x^2+p_y^2)` storage. This is still virtual-width-dependent and is
conditional on the required centered tensor structure and numerical certificates.
Add full rich residual scans and the data-supported nullspace Gram. For nullity
`k_0`, its setup can cost `O(n k_0^2 + k_0^3)`. Storing its rich cross terms can
add `O(P k_0)` memory; streaming them adds operator passes.

Peak RSS is the maximum simultaneously live process state. Count inputs,
evaluation tables, old and new hierarchy/context overlap, factors, high-precision
scratch, probes/recycled vectors, virtual residuals and exported predictions.
Report retained model memory separately. A full dense covariance output has an
unavoidable quadratic output size; selected uncertainty outputs require their
own cost and accuracy contracts.

Future runners should record both fit-phase peak and whole-process peak. The
current housing receipt has only the latter. Probes can stream, while recycling
and retained Lanczos bases can require `O(p d)` or `O(p ell)` state. Count aliases
once and independent snapshots separately, including the period where old and
new hierarchy contexts are both live.

Current dense tensor tables, SSP coordinate maps, component penalties, curvature
and inverse state prevent treating a new Hessian-vector method as sufficient for
large-coefficient scalability. Keep direct solvers when their complete cost wins.

## Bounded experiment and decision gates

1. **Completed:** unchanged housing baseline/profile, exact support handoff,
   focused numerical/mutation tests and matched complete-fit comparisons,
   including peak and retained memory. The measured tradeoff is recorded above.
2. Demonstrate nested transfer, product centering, penalty pullbacks, nullspace
   preservation and adjoint consistency for one small two-dimensional hierarchy.
   Keep a dense reference only at these small sizes. Report raw frame count,
   identified coefficient count, closure and all operator allocations.
3. Compare coarse, uniform fine and adaptive versions of that same model on a
   fixed Gaussian problem. Recompute the full omitted residual; compare error
   certificates with reference objective and prediction errors. Charge search,
   certification and finalization to the adaptive method.
4. Add automatic smoothing with small exact reference trace/determinant quantities
   and separate error checks, then assess fresh held-out accuracy. Freeze methods,
   budgets and splits before inspecting final results; account for failed fits.
5. After these gates, vary row count and coefficient count independently. Start
   with a five-cell cross: row counts 2,000/20,000/200,000 at target active width
   256, and widths 64/256/1,024 at 20,000 rows, recording actual identified widths.
   Use independently sampled rows rather than replication of identical rows.
   Use the
   proposed 1k/5k/20k coefficient ladder only after a storage/dispatch audit makes
   each step feasible. Do not escalate solely because a small HVP ran quickly.

Fixtures include smooth global structure, localized bumps, diagonal ridges,
boundary features, additive/null cases, nonuniform/correlated designs and signals
with cancelling coarse scores. For an analytic mutation, take omitted curvature
`[[1,1-delta],[1-delta,1]]` and residual `t*(1,-1)` with `t^2=delta`, `0<delta<1`.
Independent local scores vanish as `delta` shrinks while the full objective gap
stays one. Use a stable invariant/reference derivation, not coefficient-forward
accuracy near singularity. Also test zero mean residual with nonzero omitted
covariance, omitted nuisance correction, rank ambiguity and target mutations.

Stop or redesign a route if centering/closure destroys the local cost advantage,
the certificate requires an unaffordable rich solve, its bound forces essentially
full refinement on the declared workloads, or complete smoothing/inference cost
erases the saving. Record unfavorable and null cases. A measured speedup cannot
replace the mathematical claim it is supposed to satisfy.

## Research records and formal proofs

All mathematical findings, including counterexamples and rejected approaches,
belong under `docs/research`. Record definitions, assumptions, the derivation,
source/code revisions, primary references, verification commands and remaining
obligations. Distinguish published results, project derivations, implemented
numerical contracts and formalized statements. A derivation recorded here does
not by itself establish research novelty.

Lean 4.33.1 and Lake are installed user-locally in `/home/max/.elan/bin`, with
Mathlib pinned to `0df444a360eaa60ab8c11dca51a86af692955474`. Use Lean when a bounded
formal proof can check an important mathematical step. Keep successful sources,
dependency pins and exact theorem scope under `docs/research`; retain generated
caches in ignored directories. Name the precise checked statements and their
assumptions. Verifying an exact quadratic identity does not certify the Schur
bound, stochastic smoothing, floating-point implementation or statistical
coverage unless those results are separately formalized.

The [Lean proof tour](lean-gaussian-certificate/README.md) explains two beginner
examples and four general finite-dimensional Gaussian identities. The archived
project builds successfully; the [validation receipt](lean-gaussian-certificate/validation.json)
records accepted proofs, a deliberately rejected false identity, dependency
audits and source hashes. These are exact algebraic statements under explicit
hypotheses. The stronger numerical and adaptive-model claims remain separate
research obligations.
