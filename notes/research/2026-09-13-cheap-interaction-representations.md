# Cheap interaction representations relative to an additive fit

Date: 2026-09-13

Source baseline: `03766e8d5ce5b67a2ce1091ab5ae92cabf9076c3`.

Status: representation analysis and a bounded experiment proposal. This note
contains no new fit, measured speedup, implemented certificate or compiled
formal proof. Algebraic identities below are derived research claims unless a
publication is cited. The [existing representation note] and [Gaussian bounds]
give the centering and numerical contracts used here.

The parent-owned [additive baseline] supplies current timing and accuracy
controls. Its smaller cardinal interactions change the knot count, geometry
and penalty normalization. They are descriptive coarse-model controls, not the
nested pullback dictionaries or fixed-target certificates proposed here.

The selected workload also needs cheap discovery of a small unknown number of
valuable interactions. These representation analyses apply after an edge set
is declared, or within a separately specified discovery method. The
[discovery memo](2026-09-13-cheap-interaction-discovery.md) addresses proposals,
shortlist refinement and adaptive count selection. Dense all-pairs tensors are
not an intended production workload; shared-factor aggregate models below are
a distinct research alternative.

Start with a small dictionary of products of shared marginal basis functions.
It gives the smallest experiment against an unchanged rich Gaussian objective.
Retain local hierarchical refinement for functions that need resolution in a
small part of an edge's domain. Separately compare a shared-factor model when
many edges plausibly share a few marginal shapes. Its all-pairs evaluation can
cost about a rank-sized multiple of an additive evaluation. This is the most
direct representation-level route toward the user's 2, 5 or 10 times additive
cost aspiration, but it imposes stronger restrictions on the fitted functions.

None of these representations establishes a complete-fit time ratio. Their
advantages depend on different forms of compressibility:

| Representation | What must be small | Main limitation |
| --- | --- | --- |
| Selected products in shared marginal bases | Useful product coordinates per edge | A separable function can still use many product coordinates. |
| Projected hierarchical splines | Regions and levels needing fine resolution | Centering, mesh closure and certification cost extra; a ridge can require extensive refinement. |
| Shared latent marginal factors | Rank of a compatible completion across all edges | Individual low-rank edges need not admit a small shared factorization. |

An independently learned low-rank coefficient matrix on each edge is also
examined below. It sits between product selection and shared factors, with
nonconvex optimization but no sharing of edge parameters.

## Fixed target and reusable marginal geometry

Write the rich Gaussian objective as

\[
F(\beta)=\tfrac12\|W^{1/2}(y-X\beta)\|^2+
          \tfrac12\beta^\top S_\lambda\beta,
\qquad H=X^\top WX+S_\lambda\succ0.
\]

Observation weights are diagonal and nonnegative, and the penalty is PSD.
There are \(d\) features and a declared set of \(M\) edges. Feature \(j\)
has a centered marginal basis \(b_j\in\mathbb R^{k_j}\); an edge is
\(f_{j\ell}(x_j,x_\ell)=b_j(x_j)^\top C_{j\ell}b_\ell(x_\ell)\).
Freeze the data, weights, offsets, basis, centering measures, penalty
normalization, positive smoothing parameters and identified coordinates.
Every active fit readjusts the main effects and other nuisance terms jointly.

Reuse the additive model's basis when it has the required geometry. Reusing
only its fitted univariate function is too restrictive. With independent
centered \(x_1,x_2\), the signal \(x_1x_2\) has zero univariate conditional
means and a nonzero interaction. An additive coefficient or EDF screen can
therefore discard exactly the marginal directions an interaction needs.

One basis per feature means matching knots, constraints, scaling, centering,
arithmetic and rank policy across every incident edge. If these differ, count
separate compatible marginal constructions. A common feature name is
insufficient. See the [many-interaction cost analysis] for reuse ownership and
invalidation.

Two different penalty conventions must remain explicit. For centered
coefficient coordinates, a separable edge penalty can have the form

\[
\mathcal P_{j\ell}(C)=
\lambda_{j\ell,j}\operatorname{tr}(C^\top K_j C G_\ell)
+\lambda_{j\ell,\ell}\operatorname{tr}(C^\top G_j C K_\ell).
\tag{1}
\]

Here \(K_j\succeq0\) and \(G_j\succ0\).
The current cardinal tensor construction uses normalized marginal penalties
and identity factors in its declared coordinates. The separate physical
derivative model uses mass matrices \(G_j=M_j\). Replacing identity factors
with mass matrices changes the objective unless a congruence transports the
same bilinear form. The first dictionary experiment can keep the current
compiled target. The proposed local THB construction uses the separately
declared physical model; it is not already an equivalent current-model backend.

Product centering gives functional identification under a fixed product
measure. It does not give observed-sample orthogonality for correlated
predictors. The joint condition
\(\ker(W^{1/2}X)\cap\ker S_\lambda=\{0\}\) still needs evidence.

## Small product dictionaries and independent edge rank

For a compatible marginal pair \((K_j,G_j)\), choose coordinates satisfying

\[
V_j^\top G_jV_j=I,\qquad
V_j^\top K_jV_j=\operatorname{diag}(a_j).
\]

Use an ordinary orthogonal eigensystem for a current-model identity factor.
Use a generalized eigensystem for the physical mass model. Center before this
construction. The transformed edge penalty is diagonal with entries
\(s_{j\ell,uv}=\lambda_{j\ell,j}a_{j,u}
+\lambda_{j\ell,\ell}a_{\ell,v}\). This diagonalization concerns the penalty,
not the empirical joint Gram matrix.

Select \(\mathcal D_{j\ell}\subseteq[k_j]\times[k_\ell]\), and fit freely
the coefficients of those products. With \(r_{j\ell}=|\mathcal D_{j\ell}|\),
the active interaction width is \(R=\sum_{j<\ell}r_{j\ell}\) rather than
\(\sum_{j<\ell}k_jk_\ell\). A fixed dictionary is a linear subspace. Nested
dictionary growth with each component penalty pulled back by the same
embedding gives a numerical approximation to the original convex quadratic.
It does not introduce a new factor penalty or alternating coefficient solve.

Start with every rich penalty-null direction. For centered cubic marginals
with the usual second-derivative nullspace and both edge lambdas positive,
this includes the centered linear-by-linear product on every declared edge.
There is thus already an \(M\)-coefficient floor for that reference. Weak
smoothing and poorly identified joint null directions remain difficult even
if only one product per edge is active. Zero smoothing faces need a different
nullspace analysis.

A nested rectangle of the first \(h\) marginal modes gives \(h^2\) products
per edge. An adaptive dictionary can instead favor anisotropic or scattered
products, with optional downward closure. Such closure is a selection rule,
not a theorem about where the fitted signal lies. Smoothness ordering alone
does not establish that a small dictionary reaches a prescribed error.

After compatible marginal evaluations exist, a dense selected-product
forward or adjoint action costs \(O(nR)\), with product columns generated in
chunks if desired. Full marginal evaluation can cost \(O(n\sum_j k_j)\)
and dense modal setup \(O(\sum_j k_j^3)\); charge both. A dense joint solve
still has \(O((P_0+R)^3)\) factor work and quadratic factor state, where
\(P_0\) includes the intercept and ordinary terms. Reducing \(R\) can help
substantially without changing that solver. It does not remove global
coefficient coupling or the cost of searching omitted products.

Learned edge factorizations \(C_{j\ell}=UV^\top\) offer a different
compression. Rank \(r\) stores \(r(k_j+k_\ell)\) factor entries and a
generic prediction action costs \(O(nr(k_j+k_\ell))\), before any local-basis
optimization. Keeping (1) gives

\[
\mathcal P_{j\ell}(UV^\top)=
\lambda_{j\ell,j}\operatorname{tr}((U^\top K_jU)(V^\top G_\ell V))
+\lambda_{j\ell,\ell}\operatorname{tr}((U^\top G_jU)(V^\top K_\ell V)).
\tag{2}
\]

The objective is jointly nonconvex in the factors, though fixing one factor
leaves a Gaussian quadratic in the other. The transformation
\(U\mapsto UA, V\mapsto VA^{-\top}\) leaves the edge unchanged for
invertible \(A\), so factors are not identified. A sum of independent
factor roughness penalties is different from (2). Rescaling \(U\) by \(t\)
and \(V\) by \(1/t\) already demonstrates the difference.

Low matrix rank, few selected products and spatial locality are distinct.
A rank-one matrix with dense marginal factors can occupy every modal product;
a diagonal matrix with several nonzero entries has a small product dictionary
but that many independent rank directions. A separable localized bump is
rank one. Localization by itself is not evidence of high rank.

For physical mass matrices, an exact SVD truncation of
\(D=M_j^{1/2}CM_\ell^{1/2}\) has squared product-measure function error equal
to the discarded squared singular values. This is a useful oracle diagnostic
of an already computed reference, not a cheap fitting algorithm or a bound in
the joint empirical \(H\) norm. Derivatives, leverage and cross-edge coupling
can make that other norm much larger. A production candidate that uses the
rich fitted matrix to select factors must pay for that rich fit.

## Local hierarchical interactions

Use the existing proposal \(\mathcal I_A=QV_A\), where \(V_A\) is a raw
THB space, \(Q\) is frozen product centering and every raw space embeds in a
fixed fine tensor. Apply the centered embedding to both design and penalty.
Nested raw transfer then gives nested centered function spaces. Projected
columns can be redundant, and their fitted function needs an identified rich
coordinate map. The [existing representation note] specifies these maps and
the physical penalty pullback.

This route spends coefficients on localized detail instead of retaining all
fine tensor products. Buffa and coauthors establish admissibility preservation
and a cumulative refinement-closure bound for their stated dyadic hierarchy.
Their local overlap bound depends on degree and admissibility class. Those
results concern the raw mesh; they do not establish statistical marking
quality, centered-frame conditioning or a fit-time bound. [Primary paper,
Sections 2 and 3](https://arxiv.org/pdf/1509.05566).

Centering introduces marginal strips and a constant correction. The proposed
per-edge operator allowance is
\(O(n(s_e+1)+N_es_e+N_eL_e+p_e)\), using raw overlap \(s_e\), leaves
\(N_e\), depth \(L_e\) and active frame width \(p_e\). This bound depends
on the leaf and marginal data structures described in the existing note.
It is not a bound for materialized dense centered columns. Independent edges
still require their own work, even when they share marginal input data.

This is a useful candidate when difficult regions occupy a small fraction of
each edge domain. A narrow diagonal ridge can intersect many cells across the
domain; repeated localized structure, closure or global smooth detail can
also remove the saving. Report marked and inserted cells, marginal-strip
state, rank work and all refinement attempts. Near-additive cancellation in
centered prediction and physical energy needs an error enclosure or refusal.

Sparse grids select tensor levels rather than necessarily local regions.
Their classical efficiency results require mixed-regularity assumptions.
These assumptions are stronger than merely penalizing the two pure second
derivatives in (1); no sparse-grid approximation rate follows here from that
penalty. [Bungartz and Griebel, abstract](https://www.cambridge.org/core/journals/acta-numerica/article/abs/sparse-grids/47EA2993DB84C9D231BB96ECB26F615C).
Introducing a new sparse-grid or wavelet construction is unnecessary for the
first dictionary experiment.

## Shared factors across all edges

Let \(A_j\in\mathbb R^{k_j\times r}\),
\(g_{jt}(x_j)=b_j(x_j)^\top A_j[:,t]\), and choose
\(J=\operatorname{diag}(\sigma_1,\ldots,\sigma_r)\), with
\(\sigma_t\in\{-1,1\}\). The edge matrices are
\(C_{j\ell}=A_jJA_\ell^\top\). The all-pairs sum is

\[
\sum_{j<\ell}f_{j\ell}
=\frac12\sum_{t=1}^r\sigma_t
\left[\left(\sum_j g_{jt}\right)^2-\sum_j g_{jt}^2\right].
\tag{3}
\]

The \(J=I\) version is the additive factorization machine representation
in Rügamer's Lemma 1 and Proposition 1. The paper uses marginal factor
penalties and a degrees-of-freedom tuning scheme. Its optimization discussion
reports slow block-coordinate convergence in the experiments and uses
stochastic gradient methods. These are different smoothing and optimization
contracts from SuperGLM's rich tensor REML. [Primary paper, Sections 4.1 and
4.3](https://proceedings.mlr.press/v238/ruegamer24a/ruegamer24a.pdf).

Equation (3), its signed extension and the following cost/target analysis
are algebraic consequences of the stated model. With
\(K=\sum_j k_j\), dense marginal-factor evaluation plus aggregation costs
\(O(nKr+ndr)\) per full forward or gradient pass. An additive action costs
\(O(nK)\) in the same representation. Thus fixed small \(r\) can give
an additive-relative \(O(r)\) action cost while representing every pair.
Parameters use \(O(Kr)\) entries. Cached marginal bases still use
\(O(nK)\), or a chunked implementation must pay for their reevaluation.

This action count does not imply \(T_{\rm factor}/T_0\approx r\).
Charge factor initialization, all training epochs and restarts, penalty work,
rank and smoothing selection, convergence checks, certification and outputs.
The nonconvex objective can need many more passes than the additive fit.
Maintaining a small factor rank is an accuracy assumption.

The cross-edge restriction is stronger than requiring rank at most \(r\)
on each edge. With \(J=I\), stacking the \(A_j\) gives a PSD completion
of the whole block interaction matrix. Its diagonal feature blocks are
unobserved by the pairwise predictor. Arbitrary finite off-diagonal blocks
can be completed to a PSD matrix by a sufficiently large diagonal shift, but
that completion can require large rank. PSD therefore does not exclude
arbitrary interactions when rank is unrestricted; a cheap low-rank completion
is the substantive assumption. Large diagonal completions can also enlarge
the intermediate squared terms that cancel in (3).

For scalar margins at rank one, edge values satisfy
\(c_{12}c_{13}c_{23}\geq0\) in the PSD version, which excludes, for
example, \((1,1,-1)\). Signed factors permit indefinite completions and
broaden the class at a given rank. They still enforce compatibility. At
signed rank one on four vertices,
\(c_{12}c_{34}=c_{13}c_{24}=c_{14}c_{23}\). Arbitrary edge-specific
separable functions need not satisfy such relations. Orthogonal factor
rotations remain unidentified in the PSD version; signed factors have their
corresponding indefinite change-of-coordinates symmetry.

For an arbitrary selected-edge mask with symmetric adjacency \(A_E\),
the sum becomes \(\tfrac12\sum_t\sigma_t g_t^\top A_Eg_t\). A sparse
evaluation costs \(O(nMr)\) after computing the \(g_{jt}\). A mask with
a small exact low-rank-plus-diagonal representation can be cheaper, but a
generic mask has no such established saving. Similarly, exporting every
edge's predictions costs at least its \(nM\) output entries. Materializing
every edge coefficient matrix costs its \(\sum_{j<\ell}k_jk_\ell\)
entries. Equation (3) saves work for the aggregate predictor.

The rich quadratic penalty need not be replaced merely to evaluate factors.
Set \(R_j=A_j^\top K_jA_j\) and \(B_j=A_j^\top G_jA_j\). Equation (1)
evaluates through small traces such as
\(\operatorname{tr}(JR_jJB_\ell)\). For arbitrary edge lambdas the pair
trace sum costs \(O(Mr^2)\) after marginal Gram construction. If the declared
model includes all pairs and ties the edge-direction lambda to its feature,
that sum becomes

\[
\sum_j\lambda_j\operatorname{tr}
\left(JR_jJ\left[\sum_\ell B_\ell-B_j\right]\right).
\tag{4}
\]

Diagonal modal marginal penalties permit Gram construction in
\(O(Kr^2)\). Dense marginal matrices have additional multiplication costs.
This retains the given function penalty on a restricted, nonconvex mean
class. It does not make factor optimization equivalent to the unrestricted
rich solve. Tying previously independent lambdas changes the smoothing model.
Using the published factor penalty changes the penalty target as well.

The difference of squares in (3) can lose accuracy through cancellation.
An equivalent prefix recurrence accumulates
\(\sum_j g_{jt}\sum_{\ell<j}g_{\ell t}\) with the same action order.
Either kernel needs an error analysis based on absolute term sizes, actual
summation and factor-evaluation errors. A latent Hessian inverse is not the
current rich-model covariance, and factor symmetries can make it singular.

SymPy 1.14.0 expanded three finite-dimensional checks to zero remainders in
this investigation: (2) for symbolic 2-by-2 factors and symmetric marginal
matrices; (3) for four features and two signed factors; and (4) for four
features with symbolic symmetric 2-by-2 marginal factor Grams and feature
lambdas. These checks support the transposes and counting conventions. They
are not general formal proofs or floating-point certificates. The
[reproducible SymPy script](check_sympy_interaction_factors.py) and
[validation record](2026-09-13-sympy-interaction-factors-validation.json)
archive the command, versions, output and script hash.

## Certification and smoothing costs survive compression

Any mapped candidate, including a low-rank or inexact one, has rich residual
\(r=X^\top W(y-X\widehat\beta)-S_\lambda\widehat\beta\). The rich
energy error satisfies

\[
E^2=r^\top H^{-1}r,
\qquad F(\widehat\beta)-F(\beta^\star)=\tfrac12 E^2.
\]

The [Gaussian bounds] provide a penalty-null Schur correction and upper bound
without a full rich inverse. For an exact reduced fit containing every null
direction, modal positive-space terms reduce to
\(\sum_i r_i^2/s_i\). In finite precision, retain the active and null
residuals, their cross correction and arithmetic errors. Scores restricted to
the current refinement frontier cannot certify unexamined omitted products.

A full rich residual sees every declared interaction direction. Streaming
can limit retained arrays, but a dictionary, hierarchy or shared-factor
predictor does not automatically make this rich adjoint equally cheap.
The joint null Gram and its factor can also grow with \(M\). Charge residual
passes, transformations, null solves and precision increases. Weak penalties
can make the bound too loose even when held-out predictions are close.

The weighted training prediction norm satisfies
\(\|W^{1/2}X(\beta^\star-\widehat\beta)\|_2\leq E\).
The unweighted norm needs its own bound when \(W\ne I\). A held-out norm
also needs its own evaluation-operator bound, or a measured comparison with
the rich reference. Fixed-mean agreement does not certify covariance, EDF,
determinant or a smoothing optimum. Reduced-space REML uses a different
integration space unless omitted-direction corrections are established.
Smoothing remains a separate gate for every candidate, including the linear
dictionary.

## Smallest reproducible next experiment

Use one standalone research script and existing linear algebra, with no
production solver rewrite. Start with the parent's eight-feature Gaussian
fixture and all 28 edges, compiling one frozen compatible marginal basis per
feature. Effective rich marginal width five gives 25 products per edge.
Keep the current rich design and component penalties exactly. This is a small
representation check; success at that width would not establish cheap rich
interactions.

1. Use an additive fit, the full rich joint fit, and nested product dictionaries
   with the first 1, 2, 3, 4 and 5 marginal modes on both axes. Their per-edge
   widths are 1, 4, 9, 16 and 25. All arms use the same retained observations,
   nuisance model and fixed lambdas. A later adaptive allocation may redistribute
   the same total coefficient budget across edges using full residual evidence.
2. Freeze the split and seed before selection. Keep a separate validation set
   for any tuning and an untouched test set. Report test loss, the improvement
   over the additive fit, and the fraction of the rich model's positive
   improvement retained. That fraction is unstable when the rich improvement
   is near zero. Predeclare the desired retained improvement and numerical
   error budget before comparing the 2, 5 and 10 times additive time budgets.
3. Include smooth low-mode signals, edge-specific rank-one signals with
   different marginal directions and arbitrary signs, and full-rank edge
   matrices with a broad spectrum. The existing edge-separable fixture alone
   cannot establish the advantage of a shared rank-one model. Follow a positive
   result with one larger fixed width and a diagonal ridge plus boundary detail
   to test nonlocal refinement demands. Repeat the most informative adverse
   case with correlated predictors. The separately curated real-data panel
   should broaden coverage while each within-dataset comparison holds its
   declared target and split fixed.
4. Check the exact fixed-objective gap and predictions against the direct rich
   solve. Separately implement and test the proposed error enclosure before
   calling it a certificate. Include the null directions, cancellation of
   omitted scores, poorly identified data, and mutations of a mass factor,
   centering map or residual transpose transformation. Tolerances must follow
   dimensions, epsilon, norms and conditioning rather than coefficient cutoffs.
5. Record complete candidate time divided by the matching additive time,
   peak RSS, retained bytes, actual dispatch, dimensions, all search/refit
   attempts and certificate cost. Exclude the offline rich answer from candidate
   timing only if the candidate never uses that answer to choose its space.
   A small rank observed after an expensive rich fit is not a cheap candidate.

Across real datasets, inspect concentrated predictor density, correlated
features, sparsely sampled boundaries, repeated measurements and rare
categorical levels. A compact product-measure approximation can fail where
the test distribution places weight. Correlation can make individual edge
estimates unstable while aggregate prediction remains stable. A shared factor
useful for one subgroup can induce unwanted edges in another. Sparse local
sampling can make refinement fit noise. Use time, entity or spatial separation
when the intended prediction task requires it, and construct knots, centering
and category rules from training data only. These are failure modes to cover
in the separate dataset inventory, not grounds to reuse test data for tuning.

Use SVD spectra and local detail diagnostics from the rich answer only as
explicitly labeled oracle evidence for choosing the next research route. If
dictionary compression succeeds, implement its full-residual selection before
new nonlinear training machinery. If it fails on localized structure, return
to the existing THB representation gate. A bounded rank-1, rank-2 and rank-4
shared-factor comparator is justified when cross-edge compatibility appears
promising, with its distinct penalty, smoothing and convergence contract
reported alongside predictive performance.

[existing representation note]: 2026-09-13-adaptive-interaction-representation.md
[Gaussian bounds]: 2026-09-13-adaptive-gaussian-error-bounds.md
[many-interaction cost analysis]: 2026-09-13-many-interaction-scaling-analysis.md
[additive baseline]: 2026-09-13-cheap-interaction-additive-baseline.md
