# Further reductions in tensor penalty support cost

Source inspected at `d7f231e35b196b1421d604596204549d0d28ea49` on
2026-09-13. This memo contains source analysis and mathematical derivations.
It changes no production code and reports no new timing experiment. The
derivations have not been formalized in Lean or checked with SymPy. The
conditional numerical bounds below still require implementation evidence.

The support handoff is not the mathematical limit. An unprojected tensor
penalty has enough structure to compute its exact rank, null projector,
determinant and derivative sums using marginal matrices. There is also a
smaller opportunity before changing the support algorithm: scalar rank
callers currently cause construction of large roots, coordinate matrices and
error arrays that they do not consume. Keeping the present rank calculation
while making those outputs lazy can remove much of the measured work without
first proving that a marginal rank calculation reproduces the current SVD
decision.

These are separate proposals. Deferring unused outputs retains dense
factorization's cubic asymptotic cost. A representation that keeps penalties
and their support in marginal form can change that cost. Neither proposal
removes the coupled likelihood system for a model containing many
interactions.

The [completed handoff measurements](2026-09-13-tensor-support-handoff-performance.md)
record one remaining large two-component support build. In the diagnostic
profile, all 97 support builds together take 60.490 seconds of a 74.319-second
fit. The bootstrap `compute_total_penalty_rank` call accounts for 60.213
seconds. Across those support builds, `_component_root` takes 14.267 seconds
cumulatively and `_penalty_support_from_roots` takes 46.216 seconds. The
latter includes 10.976 seconds in its balanced-coordinate generator. These
are overlapping profile totals, not quantities to add. The source contains
large extended-precision reconstruction, coordinate and projection products
at those sites. The profile therefore points to materialization and its
evidence as the immediate target, rather than establishing that the SVD
itself costs 60 seconds. The diagnostic run had external CPU competition and
is unsuitable for estimating the next speedup. The committed
[measurement receipt](2026-09-13-tensor-support-handoff-measurements.json)
records that limitation.

The [many-interaction probe](2026-09-13-many-interaction-probe.md) locates a
different cost at the same source revision. With 28 tensor terms of width 25,
total coefficient width 740 and 64 smoothing parameters, its separate
8.094-second profile spends 5.247 seconds in nine centered-system builds,
including 4.997 seconds in 5,670 cross-Gram calls. Support construction takes
only 0.284 seconds. Those profile entries overlap and do not predict a
speedup. They establish why the distribution of term widths matters: the
single wide housing term exposes full-support materialization, while many
small terms expose repeated cross-group assembly. The proposed support work
addresses the first case. It must accompany the
[global scaling analysis](2026-09-13-many-interaction-scaling-analysis.md)
and its repeated-assembly and operator experiments for the second case.

The relevant source paths distinguish the mathematical inputs from the
objects currently built for them.

| Source and current location | Behavior relevant to this proposal |
| --- | --- |
| [Tensor marginal preparation](../../src/superglm/features/interaction.py), `_prepare_marginal_infos`, line 1564; [marginal builder](../../src/superglm/features/_spline_build.py), line 240 | Builds a constrained, centered marginal penalty. Support counts or geometry weights can change the centering projection. |
| [Tensor construction](../../src/superglm/features/interaction.py), `build_discrete`, line 1728 | Forms dense `kron(S1, I)` and `kron(I, S2)` at lines 1772 and 1773. The mathematical factors are available before that allocation. |
| [Fixed tensor design rebuilding](../../src/superglm/dm_builder.py), line 1225 | Preserves an unprojected multi-penalty tensor's coordinates while lambdas change. |
| [Penalty component construction](../../src/superglm/reml/penalty_algebra.py), `_tensor_marginal_rank_logdet`, line 1347, and `_canonicalize_ssp_penalty`, line 1744 | Obtains marginal rank metadata, then separately canonicalizes each full dense solver penalty using an eigendecomposition. |
| [Selected support](../../src/superglm/reml/penalty_support.py), `_component_root` and `_penalty_support_from_roots` | Applies the shared Gram policy to each solver component, balances and column-equilibrates its roots, selects rank by SVD, then constructs full support geometry and numerical evidence. |
| [Rank policy](../../src/superglm/solvers/rank.py), `RankPolicy`, `_eigensolver_relative_bar`, `_decompose_gram` | Uses `sqrt(eps)` for the factor cutoff and a dimension-dependent floor on the Gram cutoff. A marginal solve cannot silently substitute its smaller dimension. |
| [Tensor determinant summaries](../../src/superglm/reml/penalty_algebra.py), lines 1430 and 1502 | Already evaluates an eligible tensor pair with marginal mode sums. Its scalar bounds concern those selected marginal spectra. |

Let the centered marginal widths be \(a,b\), and write \(p=ab\). In exact
arithmetic, the unprojected pair is

\[
S_x=A\otimes I_b,\qquad S_y=I_a\otimes B,\qquad
S(\lambda)=\lambda_x S_x+\lambda_y S_y,
\]

where \(A,B\) are symmetric positive semidefinite. If
\(A=U\operatorname{diag}(\alpha)U^T\) and
\(B=V\operatorname{diag}(\beta)V^T\) with orthogonal \(U,V\), applying each
summand to \(u_i\otimes v_j\) gives the eigenvalue
\(d_{ij}=\lambda_x\alpha_i+\lambda_y\beta_j\). This is the Kronecker-sum
identity described in [Nick Higham's account of Kronecker products](https://nhigham.com/2020/08/25/what-is-the-kronecker-product/).
The consequences below follow directly by examining these modes.

For positive \(\lambda_x,\lambda_y\), a mode is null precisely when both
marginal eigenvalues vanish. If the marginal nullities are \(k_A,k_B\),

\[
\ker S=\ker A\otimes\ker B,\qquad
\operatorname{rank}S=ab-k_Ak_B.
\]

If \(N_A,N_B\) are orthonormal null bases, set
\(N=N_A\otimes N_B\). The penalized projector is
\(P_+=I-NN^T\). A caller needing an application of the projector does not
need a dense \(p\times(p-k_Ak_B)\) basis. For the usual centered
second-derivative tensor with one marginal null direction on each side,
\(N\) has one column. This is a structural statement about that marginal
configuration, not a fixed nullity for all spline orders or projections.

The zero faces have different answers. With only the left penalty active,
the rank is \(b\operatorname{rank}A\). With only the right active, it is
\(a\operatorname{rank}B\). With neither active, it is zero. Arbitrarily
small positive lambdas do not enter any new support cutoff. A full positive
pair's rank must never be reused as a one-component face's rank.

For a coefficient vector in the repository's row-major tensor ordering,
reshape it into \(C\in\mathbb R^{a\times b}\). Then

\[
S(\lambda)\operatorname{vec}_{\rm row}(C)
=\operatorname{vec}_{\rm row}
  \{\lambda_x AC+\lambda_y CB^T\}.
\]

This application costs \(O(ab(a+b))\) with dense marginal matrices. Applying
the eigenbasis or its inverse has the same order using two matrix products.
The component roots can remain \(R_A\otimes I_b\) and \(I_a\otimes R_B\),
where \(R_A^TR_A=A\) and \(R_B^TR_B=B\). They need not be assembled.

The selected positive modes give

\[
\log\det{}_+ S=\sum_{d_{ij}>0}\log d_{ij}.
\]

For positive smoothing parameters, let
\(f_{ij}=\lambda_x\alpha_i/d_{ij}\) and
\(g_{ij}=\lambda_y\beta_j/d_{ij}\). On active modes, \(f+g=1\). The
derivatives with respect to log smoothing parameters are

\[
\nabla_\rho\log\det{}_+S=
\begin{bmatrix}\sum f_{ij}\\\sum g_{ij}\end{bmatrix},\qquad
\nabla^2_\rho\log\det{}_+S=
\left(\sum f_{ij}g_{ij}\right)
\begin{bmatrix}1&-1\\-1&1\end{bmatrix}.
\]

The current tensor evaluator already uses log-domain sums and error bounds
for this calculation. A structured support proposal should reuse that
arithmetic. It should not reintroduce a cutoff on weighted eigenvalues,
overflow-prone direct products, or a determinant evaluation with a different
selected spectrum.

These identities require attention to coordinates. If raw coefficients equal
\(T\) times solver coefficients, the solver penalty is \(T^TST\). For an
invertible \(T\), its exact rank equals that of \(S\), and its nullspace is
\(T^{-1}\ker S\). Rank preservation alone does not preserve numerical
conditioning, orthogonal projectors or pseudodeterminants.

Writing \(S=Q_+\Lambda_+Q_+^T\) with an orthonormal retained basis gives
the exact volume identity

\[
\log\det{}_+(T^TST)=\log\det{}_+S+
\log\det(Q_+^T TT^TQ_+).
\]

This follows by taking the Gram determinant of
\(T^TQ_+\Lambda_+^{1/2}\). It agrees with the purpose of
[`_support_coordinate_volume`](../../src/superglm/reml/penalty_algebra.py),
line 226. That implementation also accounts for finite orthogonality of a
stored basis. Adding only \(2\log|\det T|\) would be wrong when the penalty
is deficient.

There is a useful smaller-nullspace form. Complete \(Q_+\) with an
orthonormal \(Q_0\). The Schur-complement determinant identity for
\(TT^T\), in this orthogonal basis, gives

\[
\log\det(Q_+^TTT^TQ_+)
=2\log|\det T|
+\log\det\{(T^{-1}Q_0)^T(T^{-1}Q_0)\}.
\]

For a separable map \(T=T_x\otimes T_y\) and product nullspace, the last
Gram matrix is itself a Kronecker product of two small nullspace Grams.
This can avoid constructing \(Q_+\) even for a deficient penalty. Its
implementation would still need inverse-action and log-volume enclosures,
and the active face determines which nullspace belongs in the formula.

A separable congruence also preserves a more general pair structure:

\[
T^TS_xT=(T_x^TAT_x)\otimes(T_y^TT_y),\qquad
T^TS_yT=(T_x^TT_x)\otimes(T_y^TBT_y).
\]

Generalized marginal eigensystems against the positive-definite Gram factors
diagonalize this pair by congruence. This is the symmetric-definite problem
supported by [`scipy.linalg.eigh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html).
The generalized eigenvalues do not by themselves provide the ordinary
Euclidean pseudodeterminant; the volume calculation still applies. Replacing
the identity factors in the existing penalty by mass matrices would instead
change the penalty model unless it comes from a consistent coefficient
transformation. The mass-matrix penalty in the
[adaptive research design](2026-09-13-adaptive-interactions-research-design.md)
must remain a separately specified model.

An arbitrary dense map usually loses cheap separability. For a rectangular
projection \(C\), the exact formula is

\[
\operatorname{rank}(C^TSC)
=\operatorname{rank}C-
\dim\{\operatorname{range}C\cap\ker S\}.
\]

That intersection has its own numerical boundary. Some small constraint
sets may admit a later compact treatment, but an unprojected tensor's null
count cannot certify it. The current determinant-summary exclusion for
projected tensors should remain in the first experiment.

The raw Kronecker rank is not yet a certificate for the current selected
numerical rank. There are several selection stages. The builder uses
`eps**(2/3)` marginal metadata and reconstructs each full solver component
independently. This can introduce rounding outside the exact Kronecker
structure and rotate their numerical nullspaces differently. The support
builder then calls `decompose_gram` on those actual solver matrices. The
Gram policy uses diagonal equilibration and an effective cutoff floored at
the active matrix order times epsilon. Finally, the selected component
roots undergo a second normalization and an SVD. A smaller marginal
eigensolve with a marginal dimension in its cutoff would change that policy
near the boundary.

Even ideal roots do not make the final column equilibration separable. Let
\(R_A^TR_A=A\), \(R_B^TR_B=B\), and assume both components are nonzero.
The exact Frobenius-normalized stacked factor has Gram matrix

\[
H_0=\frac{A\otimes I_b}{b\operatorname{tr}A}
    +\frac{I_a\otimes B}{a\operatorname{tr}B}.
\]

Its column norms are

\[
d_{ij}^2=
\frac{A_{ii}}{b\operatorname{tr}A}
+\frac{B_{jj}}{a\operatorname{tr}B}.
\]

In general these are not a product of one scale for each margin. With
\(D=\operatorname{diag}(d)\), the matrix governing the final SVD is
\(D^{-1}H_0D^{-1}\). Its nonzero eigenvalues are not the original marginal
sum eigenvalues. Its exact nullity is unchanged if all column norms are
positive, but its position relative to the numerical cutoff can change.
Also, the support code maps retained right singular vectors back through
\(D\) before its full QR. Using the equilibrated right singular vectors
directly as a Euclidean solver-space projector would be incorrect.

A conservative enclosure can make the stronger shortcut precise. The
following derivation is conditional on certified root and arithmetic
bounds; it does not claim those bounds already exist in the source.

Let \(R_x,R_y\) be the actual roots chosen by the current component
decompositions. Let \(s_x,s_y\) and \(D\) be the positive scale values used
by the current support normalization. Treat those scale values as fixed
numbers in the reference calculation. Assume no active column is lost to
underflow and all \(d_j>0\). Construct structured reference roots
\(Z_x,Z_y\) whose Grams are \(A\otimes I_b\) and \(I_a\otimes B\).
After any necessary zero-row padding, suppose orthogonal row alignments
\(O_x,O_y\) give certified bounds

\[
\|R_j-O_jZ_j\|_2\le e_j,\qquad j\in\{x,y\}.
\]

The alignments do not change a factor's singular values. They matter because
two valid roots of the same Gram matrix can have different row coordinates.
Comparing their entries without alignment is not a root-error estimate.
Marginal eigensystem errors and the full solver canonicalization must both
be covered by \(e_j\).

Define

\[
H=\frac{A\otimes I_b}{s_x^2}+\frac{I_a\otimes B}{s_y^2},\qquad
\eta=\frac{\sqrt{(e_x/s_x)^2+(e_y/s_y)^2}}{d_{\min}}
      +\eta_{\rm arithmetic}.
\]

Here \(\eta_{\rm arithmetic}\) encloses the divisions, normalization and
cast to the actual float64 SVD input. Using the actual scales in \(H\)
avoids an assumption that computed Frobenius norms equal exact traces.
The triangle inequality for stacked factors gives a perturbation of at
most \(\eta\) between that actual input \(F\) and a reference factor with
Gram \(D^{-1}HD^{-1}\).

Let \(h_{\max}\) be the largest eigenvalue of \(H\), and let
\(h_{\min+}\) be its smallest positive eigenvalue. Both come from the
small marginal spectra and mode sums. If the reference rank is \(r\),
singular-value inequalities for multiplication by \(D^{-1}\), followed
by the perturbation inequality, yield

\[
\begin{aligned}
\sigma_r(F)&\ge L_r=\sqrt{h_{\min+}}/d_{\max}-\eta,\\
\sigma_{r+1}(F)&\le U_0=\eta,\\
\sigma_1(F)&\le U_1=\sqrt{h_{\max}}/d_{\min}+\eta,\\
\sigma_1(F)&\ge L_1=\sqrt{h_{\max}}/d_{\max}-\eta.
\end{aligned}
\]

The \(\sigma_{r+1}\) condition is omitted for full rank. Outward bounds
on marginal eigenvalues, scales and scalar arithmetic belong in these
quantities. The lower bound on \(\sigma_1\) can also use an enclosed norm
of any actual column of \(F\). Its normalized columns should have norm
near one, but the implementation must bound that fact rather than assert it.

Let \(\tau\) be the existing `factor_rcond`, and let
\(\delta_{\rm svd}\) enclose error in the singular values returned by the
current SVD algorithm. Sufficient conditions for the same selected rank are

\[
L_r-\delta_{\rm svd}>\tau(U_1+\delta_{\rm svd}),\qquad
U_0+\delta_{\rm svd}\le\tau(L_1-\delta_{\rm svd}).
\]

The floating multiplication used for the cutoff needs its own outward
allowance. These inequalities certify a gap around the existing cutoff.
They do not move it. The
[LAPACK SVD error analysis](https://www.netlib.org/lapack/lug/node97.html)
provides the backward-stability form involving a dimension-dependent
multiple of epsilon. Its illustrative constant is not a ready-made strict
bound for this application. Supplying a defensible bound for the supported
arithmetic and backend remains an obligation of this structured shortcut.
The smaller experiment below avoids this extra equivalence obligation by
running the same SVD.

There is a useful refinement when the nullspace is small. Let
\(N=N_A\otimes N_B\), let \(Y=DN\), and suppose \(Y\) has full column
rank \(k=p-r\). The min-max characterization of singular values gives

\[
\sigma_{r+1}(F)\le \frac{\|FY\|_2}{\sigma_{\min}(Y)}.
\]

In exact normalization,
\(FY=[R_xN/s_x;R_yN/s_y]\). An implementation can compute an enclosed
residual using the actual selected roots and the small candidate nullspace,
including normalization and multiplication error. This costs
\(O(p^2k)\) for dense roots and is much cheaper than projecting every
root through an almost-full \(Q_+\). It needs only a small Gram enclosure
for the denominator. It does not require the computed null vectors to be
exactly orthonormal. This provides a direct bound on the discarded modes
when a global root perturbation bound is too pessimistic.

The positive-mode lower bound can instead use a Gram perturbation enclosure
\(\|F^TF-D^{-1}HD^{-1}\|_2\le\rho\), giving
\(\sigma_r(F)^2\ge h_{\min+}/d_{\max}^2-\rho\). This can be paired with
the small-nullspace residual bound. Using only a Gram perturbation bound
for the null modes is usually much weaker: a Gram error of order
\(p\epsilon\) permits root singular values of order
\(\sqrt{p\epsilon}\), which can exceed the existing
\(\sqrt\epsilon\) cutoff. The separate root-error and Gram-reconstruction
ledgers in the current implementation cannot be interchanged to avoid this
problem. In particular, zero input root-error entries mean that the selected
roots define the held target. They do not certify equality to an earlier
raw Kronecker root or account for the separately recorded reconstruction
and support-projection errors.

For a fully structured implementation, marginal root-error bounds lift
without a tensor-width product. For example,
\(\|(\widehat R_A-R_A)\otimes I_b\|_2=
\|\widehat R_A-R_A\|_2\), while its Frobenius norm gains a factor
\(\sqrt b\). An elementwise marginal Gram enclosure can remain an implicit
Kronecker enclosure. Likewise, marginal basis Gram defects bounded by
\(\epsilon_U,\epsilon_V\) give the product-basis defect bound
\(\epsilon_U+\epsilon_V+\epsilon_U\epsilon_V\). These identities reduce
the size of evidence as well as the size of numerical operands. They do
not prove that the full dense canonicalization currently performed has the
same retained representative. That comparison is still needed.

The scalar consumers provide a narrow place to start.

| Consumer | Does it need a materialized tensor support? |
| --- | --- |
| [`compute_total_penalty_rank`](../../src/superglm/reml/penalty_algebra.py), line 2260 | Needs one unweighted rank per supplied active family. It currently calls `get_support().rank` unless an external tensor evaluation is supplied. |
| [`compute_penalty_nullity`](../../src/superglm/reml/penalty_algebra.py), line 2326 | Needs ranks of the strictly positive active subsets for the identified Hessian dimension. It needs neither a penalty root nor a projector. |
| [Discrete bootstrap](../../src/superglm/reml/discrete.py), lines 446 and 456 | Requests the total rank, then estimated-scale families request active nullity. The total-rank variable is later used in profile telemetry at line 1511. Removing that telemetry request alone would leave Gaussian nullity's need for rank. |
| [Profiled objective](../../src/superglm/reml/objective.py), lines 310 and 320; [terminal scale calculation](../../src/superglm/model/reml_finalize.py), lines 241 and 282 | Consume scalar nullity. These routes need access to the same rank selection; optimizing only the bootstrap would move the dense build to a later call. |
| [`_compute_penalty_logdet_evaluation`](../../src/superglm/reml/penalty_algebra.py), line 2450 | Eligible tensor summaries bypass generic support geometry and already return scalar determinant and derivative evidence. |
| [Generic multi-penalty algebra](../../src/superglm/reml/multi_penalty.py), `_evaluate_penalty_geometry` | Uses roots, balanced coordinates and basis-volume calculations even for its summary mode. Its full result can also expose a dense inverse and square root. Keep lazy materialization available for this route. |
| [Raw-to-solver support transport](../../src/superglm/reml/penalty_algebra.py), `_ssp_component_roots`, `_support_coordinate_volume` | Consumes actual roots and bases. It is a separate coordinate contract and already excludes this fixed tensor handoff. |
| [Distributional penalty evaluation](../../src/superglm/distributional/smoothing/penalty_geometry.py), line 181, and [result evidence](../../src/superglm/distributional/results/solver.py), line 95 | Inspects root counts or retained roots and bases. A scalar certificate is not a substitute for these objects. |

The first implementation experiment should split rank selection from support
materialization for the unchanged, unprojected, identity-map discrete tensor
pair. Its proposed steps are deliberately bounded.

1. Extract shared internal operations for selecting component roots and for
   the balanced-factor SVD rank decision. Preserve the actual solver
   matrices, `decompose_gram` calls, returned-root construction, safe
   Frobenius and column norms, cast, SVD configuration and cutoff. A rank
   call would not construct dense component reconstruction bounds,
   balanced coordinates, a full QR basis or projected-root error arrays.
2. Keep rank-selection validation distinct from validation of a
   materialized root or inverse. Preserve the present balanced-factor
   residual check in the first experiment. Audit failures currently raised
   only while building the unused geometry, including singular coordinate
   maps and unrepresentable evidence. The admitted pilot domain must not
   silently turn ambiguous or invalid rank decisions into accepted ranks.
3. Route both total rank and positive active nullity to a small rank result
   owned by the complete fixed family. Bind any cached result to the exact
   ordered solver inputs and numerical arithmetic, using the existing
   context ownership rules. Keep zero-face and incomplete-family behavior
   separate. Do not populate `_PenaltySupport` with empty placeholder
   arrays or claim that scalar evidence authenticates a selected root.
4. Leave `get_support()` available for a later genuine geometry consumer.
   It must build its full evidence from the admitted inputs. A rank-only
   request must not retain all of the temporary roots and SVD arrays just
   to avoid a hypothetical later rebuild. Across finalization, either
   transfer a separately authenticated scalar result or recompute the
   cheaper rank selection once. The existing fixed-support receipt requires
   actual support, so it cannot be reused unchanged as scalar authority.
5. Prove the dispatch change independently of numerical correctness. The
   existing Gaussian public regression should observe no tensor full-support
   materialization while still observing the intended rank-selection calls.
   Compare against forced old support selection on well-conditioned tensor
   fixtures. Check selected ranks, nullity, predictions, objective,
   determinant and derivative bounds. Explicitly request full support after
   a scalar rank to verify that deferred materialization still works.
6. Add refusal controls for projection, nonidentity maps, changed component
   order or values, numerical policy and arithmetic changes, incomplete
   families, exact zero faces and near-cutoff matrices. A mutation that
   substitutes the marginal rank or union rank must fail a numerical or
   refusal test. A mutation that restores eager full support must fail the
   dispatch test. Then measure complete fits and retained/peak memory at
   matched source revisions, including a small many-interaction case.

The anticipated saving in this experiment is a hypothesis based on the
profile and consumer trace, not a measured result. Dense selected component
roots and a dense SVD remain. Its main advantage is that it removes unused
work while preserving the existing rank arithmetic. The surviving root and
SVD operations then give a cleaner baseline for the stronger structured
experiment.

The next structural experiment should carry explicit marginal penalty
provenance from tensor construction instead of recovering it by approximate
pattern matching. `_extract_tensor_marginal_eigvals` currently accepts a
relative reconstruction tolerance of `1e-10` and allocates dense Kronecker
comparisons. That is useful for its existing summary route, but it is not
exact authority for a new support representative. A compact descriptor
should bind marginal matrices, their selected spectral evidence, factor
ordering, coefficient map and numerical policy at construction. Identity
checks or a digest alone cannot authorize reuse of mutable numerical inputs.
Keep this authority within one fit, give each family its own mutable
evaluation state, and discard live authorization receipts on pickle or
deepcopy. Sharing immutable marginal operands must not create an unchecked
cross-fit cache or repeated tensor-sized snapshots on every interaction.

Admission would require both component selection and the final normalized
factor decision to agree with the current dense route, under the enclosures
above. Near a marginal cutoff, near a common-support cutoff, after an
uncertified map, with unrepresentable bounds, or when required provenance is
missing, the existing dense computation remains the fallback. A local
structured root or basis may use different floating-point entries than a
dense LAPACK result. That is acceptable only under an explicitly verified
representative/error contract; it must not be described as bitwise reuse.
This experiment should keep the penalty model and tolerance values fixed.

For many interactions, regard each distinct admitted marginal geometry as a
vertex and each tensor pair as an edge. Reusing a marginal eigensystem is
valid when its knots, constraints, centering projection, normalized penalty,
ordering, dtype and numerical policy agree. A feature name alone is
insufficient. The current `_prepare_marginal_infos` creates ingredients on
each interaction object, and the determinant-summary cache is keyed by
tensor group. There is no fit-wide shared marginal eigensystem in these
paths. The `_own_margin_cache` on a discretized tensor stores design-margin
matching information, not a penalty eigendecomposition.

Geometry weights matter when constructing centered marginals. Once an exact
penalty and coordinate interpretation have been established, full observed
row evaluations or `B_joint` snapshots are unnecessary for penalty-support
authority. This is the same distinction used by the completed handoff.
Interactions with knot overrides or different constraints may require
different marginal owners even when they mention the same input column.

The principal local costs can be separated as follows. These are algebraic
orders for dense marginal matrices; they do not include the global fit.

| Representation or operation | Work | Stored numerical operands |
| --- | --- | --- |
| Present dense tensor support, with component ranks proportional to \(p\) | \(O(p^3)\) | \(O(p^2)\) roots, bases and evidence |
| Proposed scalar rank using the current dense root/SVD selection | \(O(p^3)\), with most materialization omitted | \(O(p^2)\) temporary factor workspace; scalar retained result |
| Fully structured pair, before sharing marginals | \(O(a^3+b^3+ab)\) for spectra and scalar mode calculations | \(O(a^2+b^2+ab)\), or streamed mode sums |
| Shared marginal spectra over an interaction graph | \(O(\sum_v k_v^3+\sum_{(v,w)}k_vk_w)\) | Marginal eigensystems and per-pair scalar state or mode arrays |
| One structured penalty or spectral-basis application | \(O(ab(a+b))\) | Marginal factors plus coefficient-sized work arrays |
| Explicit full tensor inverse, root or almost-full basis | At least the cost of writing its entries | \(\Omega(p^2)\) output, regardless of implicit construction |

The fully structured orders require removing the full tensor
canonicalization and dense equality/reconstruction scans too. Retaining
those operations while adding a marginal cache does not achieve the stated
asymptotic reduction. A structured enclosure must also remain structured;
materializing every elementwise error array can restore quadratic storage.

There is a second possible local optimization if a later consumer really
needs projected roots. For an orthogonal complete basis,
\(RQ_+Q_+^T=R-RQ_0Q_0^T\). With a small nullspace the right-hand expression
costs \(O(p^2k)\) instead of \(O(p^3)\). In floating-point arithmetic,
the stored basis's orthogonality defect and the different subtraction error
must be included in the projection ledger. This identity does not by itself
eliminate root reconstruction or balanced-coordinate products, so it is a
separate fallback optimization rather than the first scalar experiment.

Finally, pairwise separability does not diagonalize the likelihood Hessian.
Observed tensor pairs can have arbitrary joint weights, and different
interaction columns generally have nonzero cross-products. The sum of
likelihood and penalty, its rank decisions, REML traces and requested
uncertainty therefore remain coupled across groups. Marginal penalty
eigensystems can support preconditioning or implicit penalty action; they
do not justify dropping cross-interaction blocks or treating them as
independent fits. The global scaling analysis and operator experiment must
remain alongside this local support work and the original adaptive
hierarchy research.

The immediate decision is whether the small rank/materialization split
passes the numerical and memory gates. The subsequent mathematical gate is
a certified equivalence between marginally structured evidence and the
current selected component and common-factor ranks. That equivalence,
including backend error allowances and ambiguous-state fallback, is the
unresolved proof obligation in this memo.
