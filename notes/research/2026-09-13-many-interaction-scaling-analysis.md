# Scaling GAMs with many tensor interactions

Date: 2026-09-13. Source audited at
**d7f231e35b196b1421d604596204549d0d28ea49** in
.worktrees/adaptive-interactions. This note extends the
[complete-fit cost audit](2026-09-13-adaptive-interaction-costs.md) and
[research design](2026-09-13-adaptive-interactions-research-design.md).
It contains source-derived operation counts, elementary exact-arithmetic
arguments, conditional alternatives and an experiment design. No fits, tests or
formal proof compilation were run for this note. The mathematical arguments
below are hand derivations, not new machine-checked theorems.

We have not reached a demonstrated mathematical limit. The measured support
handoff removes one repeated construction, while several different costs remain:
work inside each interaction, cross-term data geometry, the global coefficient
solve, smoothing optimization and requested inference. A method can improve one
of these while making another dominant.

## Dimensions and the distinction between the two graphs

Let the feature-interaction graph have \(M\) edges. Edge \(g=(u,v)\) has effective
marginal widths \(k_{g,u},k_{g,v}\), after marginal constraints, and tensor width
\(p_g=k_{g,u}k_{g,v}\). These are coefficient widths, not knot counts. If additional
constraints destroy the full product layout, use the actual retained width and
charge the constraint map separately. Let

\[
P_{\rm int}=\sum_{g=1}^{M}p_g,\qquad P=P_0+P_{\rm int},\qquad
q=q_0+\sum_g q_g
\]

for independent per-group smoothing parameters. \(P_0\) includes ordinary terms
and the intercept; a usual anisotropic tensor has \(q_g=2\). Shared smoothing
parameters change the expression for \(q\); record the actual active dimension
and any singular directions of the smoothing problem.

Let \(a_{ig}\) count the local basis entries visited on row \(i\) for group \(g\).
Including the ordinary terms, set

\[
a_i=\sum_g a_{ig},\qquad z=\sum_i a_i,\qquad o=\sum_i a_i^2.
\]

These counts refer to the local representation used by an operator. Centering
and coefficient maps have additional application cost \(t\); calling a matrix
sparse does not make a dense map free. Let \(e_S\) count stored penalty nonzeros,
or replace \(e_S\) by the actual structured penalty application cost.

For discretized groups, also record \(b_{g,u},b_{g,v}\), the marginal support
sizes, and \(h_g\leq\min(n,b_{g,u}b_{g,v})\), the number of observed pairs. Basis
width, local overlap, bin-grid size and observed-pair count are different axes.

The feature graph describes which variables enter each term. The coefficient
precision graph describes nonzero entries of \(X^\top W X+S\). They need not have
the same sparsity. Two interactions with no shared feature can still have a
nonzero cross-Gram block.

## What follows from the mathematics

Write \(X=[X_0,X_1,\ldots,X_M]\) and assume fixed nonnegative Gaussian weights,
with a block-diagonal penalty \(S\). The Hessian blocks are

\[
H_{gh}=X_g^\top W X_h+\mathbf 1_{\{g=h\}}S_g.
\]

**Cross-coupling argument.** Expanding
\(\|W^{1/2}(y-\sum_g X_g\beta_g)\|^2\) gives the cross products above.
Centering only removes specified lower-order directions; it does not establish
\(X_g^\top W X_h=0\). Even two centered columns can both equal
\((-1,1)^\top\), giving cross product \(2\). Population orthogonality under a
product measure and appropriate functional-ANOVA constraints is a special
case; it does not imply exact empirical orthogonality on correlated or finite
data, or under changing weights.

**Matrix-free identity.** For \(v=(v_g)_g\), form

\[
u=\sum_g X_gv_g,\qquad (Hv)_g=X_g^\top(Wu)+S_gv_g.
\]

This includes every cross interaction with one shared row vector; it does not
enumerate \(M^2\) pairs. With local evaluation, indexed accumulation and
structured maps throughout, its work is

\[
C_H=O(n+P+z+e_S+t).
\]

This is a conditional algorithmic bound. It is not the present tensor
implementation's bound: dense maps, dense penalties, full bin-grid histograms
and per-group full-row allocation can add work. The existing
[DesignMatrix operations](../../src/superglm/group_matrix.py#L423) already
express the forward/adjoint identity, but their consumers and representations
must also support an operator route.

**Identifiability argument.** For \(W\succeq0,S\succeq0\),

\[
H\succ0\quad\Longleftrightarrow\quad
\ker(S)\cap\ker(W^{1/2}X)=\{0\}.
\]

Indeed, \(v^\top Hv=\|W^{1/2}Xv\|^2+v^\top Sv\), and both summands are
nonnegative. If the total penalty nullity is \(k_0\) and \(n_+\) observations
have positive weight, \(k_0\leq n_+\) is necessary, though not sufficient.
For independent coefficient blocks, nullities add. With a positive two-sided
Kronecker-sum penalty, the tensor nullity is the product of marginal nullities.
Thus \(k_0\) may grow proportionally to \(M\), even when every tensor is small.
Zero smoothing faces can enlarge it. Penalizing these null directions or
removing constraints changes the model; it is not a free solver optimization.

**Output lower bounds.** Explicitly exporting a dense \(P\times P\) covariance
or Gram requires \(\Omega(P^2)\) entries and corresponding output work.
Streaming entries can reduce simultaneous storage, but a materialized dense
return value itself requires quadratic memory. Returning all \(M\) term
contributions on \(n_{\rm pred}\) rows similarly requires
\(\Omega(n_{\rm pred}M)\) output. An explicit coefficient vector requires
\(\Omega(P)\) entries. Reading arbitrary previously unseen responses requires
\(\Omega(n)\) work in the worst case. None of these arguments proves a cubic
lower bound for fitting, or a quadratic storage requirement for mean prediction
and selected uncertainty outputs. Nor is \(\Omega(nP)\) a universal input bound
when the design has an implicit representation.

## Source-derived costs at the audited head

The following are costs of identified algorithms and allocations, not empirical
power laws. Ordinary groups add analogous terms. Cubic counts use conventional
dense linear algebra; they are not lower bounds for every possible algorithm.

| Current work or state | Source-derived size/work model |
| --- | --- |
| Observed tensor tables | Retained \(O(\sum_g h_gp_g)\) entries; constructing those entries takes the same order of work, in addition to discretization, marginal preparation and observed-pair discovery. Comparison sorting for pair discovery gives a conservative \(O(Mn\log n)\) model. |
| Dense group penalties, maps and support evidence | \(O(\sum_g(q_g+1)p_g^2)\) entries for full-rank-sized layouts, with several independent arrays and precision-dependent constants. |
| First dense support construction | \(O(\sum_g q_gp_g^3)\) work when component ranks are proportional to widths. Rank-deficient cases can be cheaper; full QR and error arrays can still be quadratic. |
| Global Gram assembly | \(O(P^2)+\sum_g C_{gg}+\sum_{g<h}C_{gh}\), where each \(C_{gh}\) is the dispatched contraction, histogram or row-panel cost. A local row-product algorithm has \(O(n+P+o)\) raw arithmetic before coordinate changes, fill and dense-output initialization. |
| Dense coefficient factorization/inverse | Conventional \(O(P^3)\) work and \(O(P^2)\) entries each time the penalized system needs a new factor. |
| Dense-reference gradient penalty traces | The present block matrix products can cost \(O(\sum_g q_gp_g^3)\), even though a trace contraction need not materialize the full block product. |
| Compact fixed-weight outer-Hessian traces | Cross-group work bounded by \(O(\sum_{g<h}q_gq_h p_gp_h(p_g+p_h))\), plus same-group products/cached contractions. |
| Coefficient-response part of that outer Hessian | Present dense matrix products use \(O(P^2q+Pq^2)\) work and \(O(Pq+q^2)\) state, in addition to trace work. |
| Full dense derivative correction branch | Can retain \(q\) matrices of shape \(P\times P\); dense inverse/correction products can add \(O(qP^3)\) work and pair contractions \(O(q^2P^2)\), in addition to producing the corrections. This is a conditional branch, not a claim about the fixed-Gaussian housing profile. |

Construction occurs in
[interaction.py:1730](../../src/superglm/features/interaction.py#L1730):
observed-pair materialization is at line 1768 and dense Kronecker penalties at
1772. The
[tensor matrix](../../src/superglm/_group_matrix/_group_matrix_discretized.py#L351)
retains that table at line 387. Its Gram applies a dense coefficient sandwich at
447; forward products apply the dense map at 480/483; its adjoint at 495 builds
the full marginal-bin grid. Generic
[penalty actions](../../src/superglm/reml/penalty_algebra.py#L1178) still use the
dense component matrix for these groups.

[Penalty support](../../src/superglm/reml/penalty_support.py#L136) extracts
component roots, stacks them, runs SVD at 210 and full QR at 217, and computes
projection/error evidence in wider precision. The
[completed handoff](2026-09-13-tensor-support-handoff-performance.md) avoids
rebuilding admitted support in finalization; it does not remove the first
construction or change its exponent. Its exact ownership/invalidation checks
remain necessary.

Cached marginal spectral summaries already accelerate admitted two-component
penalty log determinants and their derivatives:
[penalty_algebra.py:1502](../../src/superglm/reml/penalty_algebra.py#L1502)
evaluates the marginal-eigenvalue sum grid in \(O(p_g)\) work per group and
lambda pair. This existing route does not supply all the dense selected-support
evidence above. Deferring unused dense support materialization and replacing
its rank construction with a structured certificate are distinct proposals.

The [group execution plan](../../src/superglm/_group_matrix/_group_matrix_execution.py#L352)
allocates a dense global Gram and visits cross-group pairs at line 430.
Shared-margin tensor products already have a
[specialized route](../../src/superglm/_group_matrix/_group_matrix_algebra.py#L390):
it forms a three-dimensional bin histogram, then a dense \(p_g\times p_h\)
output. A generic two-support histogram contraction at
[line 1407](../../src/superglm/_group_matrix/_group_matrix_algebra.py#L1407)
costs, for its displayed multiplication order,

\[
O(n+h_gh_h+h_gh_hp_g+h_hp_gp_h
  +p_g^2p_h+p_gp_h^2),
\]

when square dense coefficient maps apply. Its guarded row-panel alternative
has different costs. Therefore quoting only \(O(nP^2)\), or only the tensor
class's factored diagonal-Gram description, does not describe every current
dispatch. Record branch counters and support dimensions.

The dense inverse is materialized in
[irls_direct.py:2730](../../src/superglm/solvers/irls_direct.py#L2730).
Constant-weight Gaussian Gram reuse already exists at line 1122 and following;
do not charge every outer trial a fresh data Gram if it is actually reused.
The trace counts follow
[DenseHessianFactor](../../src/superglm/solvers/hessian_factor.py#L164):
the gradient computes a block matrix product before taking its trace; cross
traces multiply rectangular inverse/penalty blocks at 173.
[gradient.py:148](../../src/superglm/reml/gradient.py#L148) contains the compact
and full correction branches, and line 336 constructs the \(P\times q\)
coefficient-response matrix. Localized derivative supports could avoid some
operations on its explicit zeros, but that is a separate implementation claim.

### The first many-interaction profile changes the immediate priority

The parent-owned
[probe report](2026-09-13-many-interaction-probe.md) and
[measurement receipt](2026-09-13-many-interaction-measurements.json) include
the \(M=28\), effective marginal-width-five case: 740 coefficients excluding
intercept and 64 smoothing parameters. Its 8.094-second instrumented REML fit
spends 5.247 cumulative seconds in nine centered-system builds and 4.997 seconds
in 5,670 cross-Gram calls. Shared-margin dispatch accounts for 4.242 seconds;
96,768 contraction-path planning calls account for 2.343 seconds. Support work
is only 0.284 seconds. These clocks overlap and are not uninstrumented medians.
The measured bottleneck here is repeated cross-group assembly/planning, rather
than the first tensor-support build or a demonstrated dense-factor ceiling.

The existing raw profiles were inspected for the producer/caller count.
There are nine coefficient-solver invocations: eight from the discrete REML
optimizer and one from public finalization. The optimizer calls correspond to
bootstrap, six one-step candidates and its terminal refit. Each invocation
creates a fresh
[constant-weight cache](../../src/superglm/solvers/irls_direct.py#L1145),
then builds its centered system at line 1166. Thus the within-invocation cache
works; ownership does not persist across these invocations. With 36 groups,
one all-pairs pass has \(36\cdot35/2=630\) cross blocks, and nine passes give
the observed 5,670 calls. The separate fixed-lambda profile has one build and
630 cross calls.

There is also an exact graph explanation for the planning count. In a simple
interaction graph, the number of pairs of distinct edges sharing a vertex is
\(J_G=\sum_v\binom{\deg(v)}2\): each such pair has exactly one common vertex.
For the complete graph on eight features, \(J_G=8\binom{7}{2}=168\).
The admitted shared-margin route performs one optimized einsum per shared bin.
With 64 bins and nine builds, \(9\cdot168\cdot64=96{,}768\), matching the
profile's path-planning count. More generally, when margins match and the
three-dimensional-grid admission checks pass, that count per build is
\(\sum_v b_v\binom{\deg(v)}2\). Thus a matching has no such calls and a hub
with \(M\) incident edges has \(\binom{M}{2}\) shared-edge pairs. Other cross
routes still execute: this is a route-specific count, not a total-work formula
or a claim that graph sparsity decouples the solve.

Reuse must distinguish coordinates. Gaussian/identity explicitly has constant
working weights under the solver's
[family/link check](../../src/superglm/solvers/irls_direct.py#L281).
[dm_builder.py:1225](../../src/superglm/dm_builder.py#L1225) preserves the same
unprojected multi-penalty tensor group across lambda changes, while ordinary
SSP groups in the branch starting at line 1258 receive new lambda-dependent
coefficient maps. The
optimizer invokes that rebuild at
[discrete.py:1346](../../src/superglm/reml/discrete.py#L1346).
Consequently a whole transformed Gram cannot simply be reused unchanged.

The bounded reuse question is whether a fit-local owner can retain unchanged
tensor-by-tensor blocks, or raw \(B_g^\top W B_h\) moments with updated map
sandwiches, through those solver boundaries. Admission must bind ordered basis
values, bins/row assignments, weights, dimensions, coefficient placement,
centering and arithmetic/rank evidence; equal feature names or Gaussian labels
alone are insufficient. Reused data geometry still needs a new penalty/Hessian
and appropriate factor evidence when lambdas change. Updating main-effect
maps must not reuse a previous centered-system certificate without validating
the transformed target and cancellation bounds. Count retained raw blocks,
comparison snapshots and live old/new systems against the saved contractions.
Caching compatible contraction plans is another small hypothesis, subject to
shape/dtype and arithmetic-equivalence checks. Neither saving has been measured.

### A many-group cache lifetime that a one-tensor fit does not expose

[_BlockWeightCache](../../src/superglm/_group_matrix/_group_matrix_algebra.py#L72)
retains every admitted two-dimensional histogram until the assembly returns.
Generic tensor-pair histogram dispatch can therefore retain many distinct
\(h_gh_h\) arrays alongside the dense Gram. A per-histogram cell ceiling does
not bound their sum: up to \(O(M^2)\) admitted pair grids can coexist when
their index/weight identities differ. Reverse-key transposes are aliases and
must be counted once. The shared-margin three-dimensional grids are transient
in the inspected route, not retained by this cache.

This is a possible source-derived memory growth, not a measured bottleneck in
the current housing fit or the new many-interaction probe. Measure cache builds,
reuse, unique backing bytes and release time before proposing an eviction rule.
An assembly cache with little reuse could lose memory without much recomputation;
actual reuse and complete-fit cost must decide that.

## What changes as the number of interactions grows

For the following comparison assume equal effective marginal width \(k\),
so \(p_g=k^2\), bounded \(q_g\), fixed spline degree with at most \(s\) local
entries per margin, and \(P_0\ll Mk^2\). These are explicit simplifying
assumptions; a dense centered representation need not have this local overlap.

| Quantity | \(M\) equal tensors |
| --- | --- |
| Coefficients | \(P_{\rm int}=Mk^2\) |
| First dense support work | \(O(Mk^6)\) |
| Dense per-group state | \(O(Mk^4)\) |
| Global dense factor work/state | \(O(M^3k^6)\) / \(O(M^2k^4)\) |
| Raw local forward/adjoint work | \(O(nMs^2+P)\), before penalties/maps |
| Raw local pair-assembly arithmetic | \(O(nM^2s^4+P)\), before maps/fill/output |
| Smoothing dimension with two parameters per tensor | \(q=q_0+2M\) |

For all pairs of \(d\) features, \(M=d(d-1)/2\): the conventional dense global
factor count becomes \(O(d^6k^6)\), while the conditional local data action is
\(O(nd^2s^2+P)\). This difference motivates research; it does not establish
iteration counts or a complete-fit speedup.

At fixed \(P_{\rm int}\), heterogeneity matters:

\[
\sum_g p_g^2\geq P_{\rm int}^2/M,\qquad
\sum_g p_g^3\geq P_{\rm int}^3/M^2.
\]

These follow from convexity, with equality for equal widths; also
\(\sum_g p_g^r\leq p_{\max}^{r-1}P_{\rm int}\) for \(r\geq1\).
Thus many equal thin groups reduce dense per-group storage/work while leaving
the global dense \(P^2/P^3\) counts essentially unchanged. They can increase
row overlap, \(q\), nullity and bookkeeping. Equal-\(P\) models are not equal
statistical models, so prediction equality across such models is not an
acceptance requirement.

### Graph and marginal reuse

When all edges incident on a vertex really share a marginal, preparation cost
can move from \(\sum_v\deg(v)C_v\) to \(\sum_v C_v\). Shared bin assignments can
similarly move from repeated \(O(nM)\) storage toward \(O(n|V|)\), while
interaction-specific pair maps or tables remain. Dense marginal spectral
setups can conditionally cost \(\sum_v O(k_v^3)\), followed by per-edge work.
The actual reuse units are compatible marginal constructions, not feature
names: different knots, normalization, geometry weights, centering, dtype or
coordinate/rank policy create different units.

The current
[marginal preparation](../../src/superglm/features/interaction.py#L1564)
constructs/stores both marginal objects on each interaction. Existing shared
histograms and cross-product routes are useful starting points, not proof that
all marginal setup is already shared. A fit-local owner needs exact admission,
explicit lifetime and invalidation on relevant input changes; mutable identity
alone is insufficient.

Sharing a marginal does not share the arbitrary edge coefficient matrix.
For a separable physical penalty, dense marginal actions can cost
\(O(p_g(k_{g,u}+k_{g,v}))\) per group; bounded-bandwidth marginal factors can
reduce this to \(O(p_g)\). Both require a representation that preserves those
factors and handles its constraints. Diagonalizing each penalty does not
diagonalize the empirical cross-Gram. A block preconditioner might cost
\(\sum_g O(p_g^3)\) to set up and \(\sum_g O(p_g^2)\) to apply using dense
blocks, but its iteration count depends on the remaining coupling. The feature
graph alone gives no bound on that count.

## Complete smoothing, certification and memory costs

Let \(D_i=\partial S/\partial\rho_i\), where \(\rho_i=\log\lambda_i\).
Let \(r_i\) be the number of coordinates touched by \(D_i\),
\(L_D=\sum_{i=1}^{q}r_i\), and
\(C_D=\sum_i C(D_i v)\). For ordinary block-local components,
\(L_D=\sum_gq_gp_g\), including ordinary penalized groups. Let \(C_K\)
include one Hessian action, preconditioner application and Krylov vector work.

For fixed Gaussian weights, the following is one conditional operator design:

| Task | Work to include |
| --- | --- |
| Coefficient solve | Preconditioner setup/update plus \(K C_K\) |
| Gradient trace estimates | \(b_g K C_K+b_g C_D\) |
| Full smoothing-Hessian trace estimates | \(b_h(q+1)K C_K+b_h C_D+b_h qL_D\) when derivative supports are used; \(b_hq^2P\) replaces the last term for dense contractions |
| Coefficient-response terms | \(q\) solves, component actions and \(O(qL_D)\) contractions |
| Log determinant | \(b_{\log}\ell C_H\), plus vector/reorthogonalization work and any preconditioner determinant correction |
| Dense outer Newton step | \(O(q^3)\) work and \(O(q^2)\) state |

**Trace identity behind the counts.** For a random vector with
\(\mathbb E[vv^\top]=I\), solve \(u=H^{-1}v\).
Then \(\mathbb E[u^\top D_i v]=\operatorname{tr}(H^{-1}D_i)\).
For full Hessian traces also solve \(w_j=H^{-1}D_jv\) for each \(j\);
\(\mathbb E[(D_i u)^\top w_j]
=\operatorname{tr}(H^{-1}D_iH^{-1}D_j)\).
This needs \(q+1\) solves per probe, not one. The localized dot products have
total length \(qL_D\). The identities assume exact solves; numerical and
stochastic errors require separate budgets. Sharing probes does not remove
their error, and these counts do not assert an optimal estimator.

There is no dimension-independent constant \(K\) established here. For exact
arithmetic and an SPD preconditioned system with condition number \(\kappa\),
the standard CG energy bound is
\(\|e_K\|_H\leq2[(\sqrt\kappa-1)/(\sqrt\kappa+1)]^K\|e_0\|_H\).
It explains a sufficient \(O(\sqrt\kappa\log(1/\varepsilon))\) dependence;
finite precision requires verified residual/error handling, and clustered
spectra can behave better. [Saad, chapter 6.11](https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf)

Probe counts and Lanczos depth depend on spectra, error targets and confidence.
Budget confidence over all reported derivatives, log determinants, candidate
choices and adaptive trials; a full Hessian has \(O(q^2)\) entries. A fixed
probability per scalar estimate is not a complete-fit guarantee. General GLMs
add changing weights and weight-derivative terms. Gradient-only outer methods
can avoid explicit \(q^2\) output, but their complete work and stopping evidence
must be compared with the current optimizer.

For refinement stages \(r\) and every attempted outer evaluation \(e\), write

\[
T_{\rm complete}=
\sum_r\left[
 C_{{\rm build},r}+C_{{\rm search},r}+C_{{\rm cert},r}
 +\sum_{e=1}^{E_r}
 (C_{{\rm coeff},re}+C_{{\rm outer},re})
\right]+C_{\rm terminal}+C_{\rm requested}.
\]

Coefficient costs include every required inner iteration and factor/setup;
outer costs include its actual traces, log determinants and steps. Failed
candidate fits, rejected lambda trials and terminal refits belong in the sum.
The direct comparator receives its existing geometry/factor reuse. Changing
\(M\) changes both \(E_r\) and inner iteration distributions in ways these
algebraic counts do not predict.

### The many-interaction virtual certificate

Let \(P^\star=\sum_g p_g^\star+P_0^\star\) be the fixed rich width, and let
\(U,V\) be an orthogonal split of its penalty-null and positive subspaces.
For the rich normal-equation residual \(r=b^\star-H^\star\hat\beta\), use
\(r_0=U^\top r\) and \(r_z=V^\top r\). Define
\(A=W^{1/2}X^\star U\), \(B=W^{1/2}X^\star V\),
\(K_0=A^\top A\), \(L=V^\top S^\star V\), and
\(\tilde r_z=r_z-B^\top A K_0^{-1}r_0\).
The [certificate analysis](2026-09-13-adaptive-gaussian-error-bounds.md)
uses the conditional exact-arithmetic bound

\[
\|e\|_{H^\star}^2
\leq r_0^\top K_0^{-1}r_0+\tilde r_z^\top L^{-1}\tilde r_z.
\]

The positive penalty inverse separates by interaction, but \(K_0\) generally
does not. Once nullspace evaluations are available, a dense Gram/factor
construction costs \(O(nk_0^2+k_0^3)\); evaluating them is additional work.
If \(k_0=O(M)\), this reintroduces \(O(nM^2+M^3)\) work in this particular
certificate construction, even if coefficient iteration uses local operators.
It is not a lower bound for every possible certificate or coarse solve.

Subject to the centered tensor structure and valid numerical evidence,
separable positive-penalty inverse setup costs
\(\sum_g O((k_{g,u}^\star)^3+(k_{g,v}^\star)^3)\), reducible to distinct
compatible marginal setups, and one application costs
\(\sum_g O(p_g^\star(k_{g,u}^\star+k_{g,v}^\star))\).
Add full rich residual evaluation, all coordinate changes and the nullspace
cross correction. Caching that cross matrix costs \(O(P^\star k_0)\) entries;
applying it as \(B^\top(Ac)\) avoids the matrix and requires operator passes.
A rich residual retained in full costs \(O(P^\star)\) entries. Streaming
interaction blocks can reduce that particular storage if the whole certificate
permits it; no active-\(P\)-only memory guarantee follows without proving such
an execution plan and charging its extra passes. Near-null directions and zero
faces need their own rank/precision evidence.

### Peak memory

Use a live-object model, not a sum of all allocations ever made:

\[
M_{\rm peak}=\max_\tau\{
M_{\rm input}+M_{\rm design/maps/contexts}
+M_{\rm factors}+M_{\rm assembly\ cache}
+M_{\rm outer}+M_{\rm precision\ scratch}
+M_{\rm certificate}+M_{\rm requested\ outputs}\}_\tau.
\]

Count independent snapshots and old/new owners while both are live; count
aliases once. The principal possible terms are
\(\sum_g h_gp_g\), \(\sum_g q_gp_g^2\), \(P^2\), \(Pq\), \(q^2\), the sum
of cached grid cells, and the certificate state above. Full derivative
correction branches can add \(qP^2\); retained Krylov/recycling vectors can add
\(Pd_{\rm rec}\) for \(d_{\rm rec}\) vectors, while sequential probes need not
retain all probes.
Raw inputs, row maps and predictions also remain. Report fit-phase peak,
whole-worker peak RSS and retained model bytes separately.

## Accuracy and a bounded measurement design

Approximation errors across interactions also couple. If group errors are
\(\delta_g\), the triangle and Cauchy inequalities give

\[
\left\|\sum_g\delta_g\right\|_W
\leq\sum_g\|\delta_g\|_W,\qquad
\left\|\sum_g\delta_g\right\|_W^2
\leq M\sum_g\|\delta_g\|_W^2.
\]

Empirical orthogonality improves the second expression to equality without
the factor \(M\), but requires evidence. Consequently a per-term approximation
tolerance is not automatically a whole-model tolerance. No smoothness-dependent
approximation rate, sample-complexity law or fresh held-out accuracy guarantee
is derived here. Fixed-system numerical error, smoothing-optimum error and
predictive model error remain distinct.

The parent-owned first probe uses \(n=2048\), eight ordinary cardinal-cubic
terms with knot count six/effective width five, tensor marginal width five,
64 bins and \(M\in\{1,2,4,8,16,28\}\) in a balanced edge order.
Thus the coefficient count excluding intercept is \(40+25M\).
It compares fixed \(\lambda=0.1\) with automatic REML, with two sequential
repeats in forward/reverse order. Its
[source-bound report](2026-09-13-many-interaction-probe.md) also compares two
models with 320 coefficients excluding intercept: four width-64 tensors versus
18 width-16 tensors. The ordinary blocks differ (64 versus 32 coefficients),
as do \(q\), graph, resolution and optimizer iterations. This comparison shows
why \(P\) alone is insufficient; it does not identify a single causal mechanism.
Iteration-limit endpoints remain separate from converged fits, and these
observations do not establish empirical exponents.

After that probe, select the smallest additional slice that resolves its
dominant uncertainty; do not run a full Cartesian grid:

| Axis | Controlled comparison |
| --- | --- |
| Number of terms | The parent probe above; retain actual \(P,q,k_0\), iterations and dispatch. |
| Width | Three effective marginal widths at fixed \(M,n\), after checking predicted storage/deadline. |
| Rows | Three independently sampled row counts at fixed groups, widths and basis construction; repeated identical rows are not a generic row-scaling test. |
| Graph | Equal \(M\), widths and ordinary features; compare a matching/spread graph, a hub-heavy graph and a dense subgraph. Choose a common feature pool large enough for every graph and keep ordinary terms fixed. |
| Conditioning | Independent versus correlated predictors, and bounded weight/anisotropy ranges, with the same shape; include a near-null refusal case separately. |
| Equal coefficients | For example \((M,k)=(1,16),(4,8),(16,4)\) gives \(P_{\rm int}=256\) in every case. Use actual effective widths, not these numbers as knot settings; record the changing \(q\) and \(k_0\). |

For each shape, separate profiling from uninstrumented complete-fit repeats.
Record source/runtime/input identities, declared output contract, convergence
and refusal status, term widths, support counts, graph degrees, \(z,o\), selected
ranks, support-build counts, Gram/factor/trace calls, outer/inner iterations,
backend dispatch, histogram builds/reuse/live bytes and all three memory
measures. Candidate and comparator need the same numerical target within a
shape; different models across the shape grid need a common, untouched
predictive evaluation protocol if an accuracy comparison is claimed.

The next change should target the producer/consumer that the complete profile
actually identifies. Accept a structured/operator experiment only if its
claimed numerical budgets and requested inference pass, and its full-fit
time/memory compare favorably at the declared accuracy. Stop the pilot when
conditioning, rank refusal, virtual certification, outer work or required
outputs exhaust the agreed budget. Report a narrower useful regime when that
is what the evidence supports. Two widths, one tensor, or unconverged endpoints
cannot establish a general scaling law or a mathematical performance ceiling.
