# Adaptive interaction representation, centering and penalty geometry

Date: 2026-09-13

Source baseline: 7c4e70ffac99c9a70adf90e5b21d9735109c4675, published v0.33.0

Status: reviewed research derivations and a proposed Gaussian pilot; implementation and arithmetic certification remain open.

This note records the representation findings supporting the
[adaptive-interaction research design](2026-09-13-adaptive-interactions-research-design.md)
and candidates C16/C21 in the
[feature-roadmap additions](2026-09-superglm-feature-roadmap-additions.md).
It is intended to preserve assumptions and proof obligations for subsequent
implementation and possible paper development. None of the results in this
note is claimed to have been verified in Lean.

The centering, rank, coordinate and operator arguments below were derived in
this investigation. Published results are identified and cited separately.
There is no new fit, benchmark result or implemented floating-point certificate
in this note.

## 1. Recommended first representation

Use a fixed product-centering operator applied to a raw truncated hierarchical
B-spline frame. Solve the fixed Gaussian problem over the resulting interaction
space, keeping coefficient redundancies explicit in the mathematical model.
Certify the reconstructed function in independent coordinates of a predeclared
fine tensor space.

The initial hierarchy should be bicubic, dyadic, isotropic in fixed normalized
coordinates, and strictly admissible with a fixed admissibility class. Freeze
the physical rectangle, coordinate scaling, knot nesting, maximum level,
boundary/extrapolation policy, centering measures and positive smoothing
parameters before adaptation. A concrete minimal boundary choice is an open
endpoint knot vector with multiplicity four and otherwise unconstrained cubic
coefficients. Natural boundary restrictions would define a different declared
space and need their own compatible construction.

The current housing tensor is a separate comparator. Its cardinal natural cubic
marginals, quantile-selected knots and normalized penalties with identity factors
do not define the physical penalty below. Independently selected quantile knots
at two marginal widths need not nest. The adaptive experiment needs matched
coarse and fine uniform references with its own frozen semantics before an
equivalence claim with the existing model is possible.

## 2. Normalized product centering

Let \(D=D_x\times D_y\) be a nondegenerate rectangle and let
\(S_x,S_y\) be the finest marginal cubic spline spaces, both containing the
constant function. Write

\[
V_{\max}=S_x\otimes S_y,\qquad P=p_xp_y.
\]

Here \(p_j=\dim S_j\), and \(P\) counts raw fine tensor coefficients.
Fix probability measures \(\mu_x,\mu_y\) on their marginal domains.
They can be normalized Lebesgue measures or normalized, frozen training
geometry measures. A zero total mass is invalid. Re-estimating these measures
during refinement would change the target.

For a marginal column-vector basis \(b_j\), let \(e_j\) represent the constant
function and set

\[
c_j=\int b_j\,d\mu_j,\qquad
c_j^Te_j=1,\qquad
R_j=I-e_jc_j^T.
\]

Then \(R_j^2=R_j\), its range is \(\ker c_j^T\), and its kernel is
\(\operatorname{span}\{e_j\}\). In the fixed tensor coefficient ordering, define

\[
Q=R_x\otimes R_y.
\]

The same symbol denotes the corresponding function projection:

\[
(Qf)(x,y)=f(x,y)-a_x(x)-a_y(y)+a_0,
\]

\[
a_x(x)=\int f(x,t)\,d\mu_y(t),\qquad
a_y(y)=\int f(s,y)\,d\mu_x(s),\qquad
a_0=\iint f\,d\mu_xd\mu_y.
\]

Normalization is required for this four-term formula. Unnormalized measures
would require division by their marginal total masses.

Using the marginal bases to identify functions with coefficients, the range
is the interaction space

\[
\mathcal I_{\max}
=\{f\in V_{\max}:\int f(x,y)\,d\mu_y(y)=0,\
                         \int f(x,y)\,d\mu_x(x)=0\}
=(\ker c_x^T)\otimes(\ker c_y^T),
\]

of dimension \((p_x-1)(p_y-1)\). Its complementary additive space is

\[
\mathcal A=S_x\otimes1+1\otimes S_y.
\]

Indeed, \(Q\) annihilates additive functions and \(f-Qf\) is additive.
Consequently \(\ker Q=\mathcal A\).

These are functional constraints under the fixed product measure. They do not
make interaction columns orthogonal to nuisance columns in a correlated
observed sample. The actual joint design and its penalty-null directions must
still be checked for identification.

## 3. Projected spaces and the raw-constraint obstruction

Let \(V_A\subseteq V_{\max}\) have an independent raw THB basis. Denote its
coefficient embedding by \(E_A\), so

\[
B_A=B_{\max}E_A.
\]

Use \(\mathcal I_A=QV_A\). Its evaluated frame is \(B_{\max}QE_A\).
For any raw refinement satisfying

\[
E_A=E_BT,
\]

the projected embeddings satisfy

\[
QE_A=QE_BT,\qquad \mathcal I_A\subseteq\mathcal I_B.
\]

This requires no additional marginal mesh closure.

The alternative \(V_A\cap\ker C_{\max}\) is nested, but it can suppress isolated
fine directions. For example, enlarge a coarse tensor space by a localized
\(u(x)v(y)\), where \(u\) is outside the coarse marginal space and
\(\int v\,d\mu_y\ne0\). Enforcing a zero \(y\)-integral on a function in this
enlarged raw space can force the added coefficient to zero: its fine marginal
\(u(x)\int v\,d\mu_y\) cannot be cancelled by a coarse marginal function.
Applying \(Q\) instead gives

\[
Q(uv)=(u-\mu_xu)(v-\mu_yv).
\]

The marginal corrections are determined by the coefficient of \(uv\). They do
not introduce separately fitted marginal parameters. They do introduce global
strips and a constant correction into the function representation. Therefore
strict locality of the centered columns is not claimed.

## 4. Raw-to-identified coordinates

The matrix \(QE_A\) lives in redundant raw fine tensor coordinates. It must not
silently become the embedding into an independent rich coefficient vector.

Choose full-column-rank \(Z_j\) spanning \(\ker c_j^T\), and any left inverse
\(J_j\) with \(J_jZ_j=I\). Define

\[
Z_{\mathcal I}=Z_x\otimes Z_y,\qquad
J_{\mathcal I}=J_x\otimes J_y,\qquad
P_A=J_{\mathcal I}QE_A.
\]

Since the range of \(QE_A\) lies in that of \(Z_{\mathcal I}\),

\[
Z_{\mathcal I}P_A=QE_A.
\]

Thus the independent rich design and penalty are

\[
X_{\mathcal I}=X_{\max}^{\rm raw}Z_{\mathcal I},\qquad
S_{\mathcal I}=Z_{\mathcal I}^TS_{\max}^{\rm raw}Z_{\mathcal I},
\]

and the active quantities are

\[
X_A=X_{\mathcal I}P_A,\qquad S_A=P_A^TS_{\mathcal I}P_A.
\]

Left inverses and tensor transforms are mathematical definitions here; no dense
Kronecker matrix is prescribed. For arbitrary centering measures, \(R_j\) need
not be the physical-mass orthogonal projection. In particular,
\(Z_jJ_j=R_j\) is not justified for an arbitrary left inverse. The identity
actually used is \(Z_jJ_jR_j=R_j\).

Joint nuisance terms append their own fixed coordinates and embeddings.
Positive definiteness of the full rich Hessian is an additional hypothesis.
Functional ANOVA identification by itself does not establish it.

## 5. Frame rank and a consistent PSD solve

Because the raw THB columns are independent,

\[
\operatorname{rank}(QE_A)
=\dim V_A-\dim(V_A\cap\mathcal A).
\]

This additive kernel can change when refinements permit new additive
functions. Removing only a fixed coarse additive subspace is insufficient in
general.

There is a sparse rank identity for reference checks. On each leaf rectangle,
express \(\partial_{xy}f_A\) in its nine biquadratic Bernstein coefficients.
Let \(D_A\) map raw active coefficients to the stacked cell coefficients.
On the connected rectangle, a globally sufficiently smooth function with
\(\partial_{xy}f=0\) is additive. One can integrate the equality to obtain
\(f(x,y)=u(x)+v(y)\). Bicubic \(C^2\) splines meet the required smoothness.
Therefore

\[
\ker D_A=\ker(QE_A),\qquad
\operatorname{rank}D_A=\operatorname{rank}(QE_A).
\]

The rows are local, but a rank-revealing factorization of this matrix may
produce fill. Its production cost is unresolved. Use the identity for small
exact or well-conditioned audits, rather than assuming sparse rank detection
is cheap.

Suppose the identified rich quadratic has Hessian \(H\succ0\) and right-hand
side \(b\). The frame equations are

\[
G_A\alpha=g_A,\qquad
G_A=P_A^THP_A,\qquad g_A=P_A^Tb.
\]

They are PSD and consistent:
\(\ker G_A=\ker P_A\), and \(g_A\) is orthogonal to that kernel.
All coefficient solutions map to the same rich fitted coefficient vector.
For example, in exact arithmetic, conjugate gradients started at zero works
within the range of \(G_A\). A correction equation after a transferred warm
start is also consistent.

This provides a concrete initial solver route without adding a ridge or
constructing a dense nullspace basis. It does not prove finite-precision
behavior of the proposed implementation. A badly conditioned frame, roundoff
drift in null directions, or cancellation in its operators can still cause
failure. The separate full-rich residual certificate must assess the mapped
candidate, including inexact active solutions. A finite-precision frame solver
and a certified rank/identification contract remain proof and test obligations.

## 6. Physical penalty and integration

Fix

\[
a_\lambda(f,g)
=\lambda_x\int_D f_{xx}g_{xx}
 +\lambda_y\int_D f_{yy}g_{yy},
\qquad \lambda_x,\lambda_y>0.
\]

With physical marginal mass and second-derivative matrices,

\[
M_j=\int_{D_j}b_jb_j^T,\qquad
K_j=\int_{D_j}b_j''(b_j'')^T,
\]

the raw fine penalty is

\[
S_{\max}^{\rm raw}
=\lambda_xK_x\otimes M_y+\lambda_yM_x\otimes K_y.
\]

The centered active penalty is

\[
S_A=(QE_A)^TS_{\max}^{\rm raw}(QE_A).
\]

The uncentered pullback \(E_A^TS_{\max}^{\rm raw}E_A\) would penalize a different
function. Under refinement, \(S_A=T^TS_BT\), separately for both components.
Mass factors cannot be replaced by identity unless a consistent coordinate
transformation establishes that identity for the same bilinear form.

If physical axes have lengths \(L_x,L_y\) and normalized coordinates are
\(u=(x-a_x)/L_x\), \(v=(y-a_y)/L_y\), the two derivative energies acquire
factors \(L_y/L_x^3\) and \(L_x/L_y^3\). Dropping these factors changes the
physical smoothing convention.

On a leaf, products contributing to \(f_{xx}g_{xx}\) have degrees at most
two in \(x\) and six in \(y\). Four Gauss points in each direction therefore
integrate both bicubic derivative components exactly in exact arithmetic.
Exact Bernstein-product integrals are another option.

The existing
[integrated-derivative helper](../../src/superglm/features/_spline_penalties.py)
uses \(n_{\rm quad}=\max(\text{order}+1,\text{degree})\). Calling it with
degree three and order zero as a mass constructor would use only three Gauss
points. Cubic mass products have degree six and need four. This is a proposed
reuse hazard, not a claim that the current supported derivative-penalty path
is incorrect.

## 7. Leaf and marginal operators

Let \(N\) count raw mesh leaf cells, \(p\) raw active frame columns, \(L\) maximum
depth, and \(s\) maximum raw overlap. Store or compute each leaf's bicubic
coefficient map using its at most \(s\) contributing functions.

For a leaf \(I\times J\), integrate its polynomial over \(J\). The result is a
cubic contribution on \(I\). Accumulate these contributions in a one-dimensional
dyadic interval tree, splitting polynomial representations where the projected
partition requires it. Repeat in the other direction. This constructs
\(a_x,a_y,a_0\) for a coefficient vector without expanding that vector on the
entire finest tensor grid. The adjoint reverses the same linear operations.

There are at most \(2N\) distinct endpoints among all projected intervals on
one axis. A conservative implementation allowance for tree accumulation is
\(O(NL)\) time and storage. More compact implementations may improve this,
but that improvement is not assumed.

For empirical measures, freeze marginal interval moments through degree three.
Use a consistent half-open cell convention for atoms on boundaries, with the
outer endpoint included. Scaled local moments can avoid unnecessarily large
physical-coordinate powers. Sorting and moment construction, potentially
\(O(n\log n)\), belong to complete construction cost and retained state.

After point locations are available, a forward or adjoint operator has the
derived target cost

\[
O\!\left(n(s+1)+Ns+NL+p\right).
\]

Hierarchy and leaf state costs \(O(p+Ns+NL)\), in addition to input rows,
cached row evaluations or a bounded chunk workspace. Initial row location can
cost \(O(nL)\); rebuilding it during refinement is not free. This construction
keeps all observations.

These are proposed algorithmic bounds conditional on the stated data
structures and cell maps. They are not measurements or a published THB
centering theorem. In particular, report \(N\) as well as \(p\); no unproved
replacement \(N=O(p)\) is needed for this cost statement. Dense centered
columns and dense cross-products would defeat these operator bounds.

## 8. Penalty corrections and cancellation

The physical penalty can use the same marginal machinery. For the \(x\)
component define

\[
a(x)=\int f(x,y)\,d\mu_y(y),\qquad
b(x)=\int_{D_y} f(x,y)\,dy.
\]

Because \((Qf)_{xx}=f_{xx}-a''\),

\[
\int_D (Qf)_{xx}^2
=\int_D f_{xx}^2-2\int_{D_x}a''b''
 +L_y\int_{D_x}(a'')^2.
\]

There is an analogous \(y\) expression. Both the Lebesgue and centering-measure
marginal integrals are one-dimensional operators. For normalized Lebesgue
centering, \(b=L_ya\), so the expression reduces to

\[
\int_D (Qf)_{xx}^2
=\int_D f_{xx}^2-L_y\int_{D_x}(a'')^2.
\]

The quadratic expressions also determine bilinear actions by polarization or
direct differentiation. They avoid creating a two-dimensional mesh consisting
of all projected marginal knot crossings.

They are signed expressions. Near-additive functions can make their true
centered energy small while the individual terms remain large. Error bounds
must include the absolute sizes and computation errors of all terms, with
summation and multiplication bounds matching the actual kernel. A favorable
computed cancellation does not certify its own accuracy. The same issue applies
to subtracting marginal functions in prediction and in adjoints.

An implementation must enclose these errors or refuse certification when they
cannot be separated. Clipping a negative computed quadratic form to zero,
dropping a near-null direction, or adding a ridge changes neither this
obligation nor the identity of the target problem.

## 9. Generalized marginal modes and the rich certificate

First restrict to the fixed centering constraints:

\[
M_j^0=Z_j^TM_jZ_j,\qquad K_j^0=Z_j^TK_jZ_j.
\]

Here \(M_j^0\succ0\). Solve the reduced generalized eigenproblem using
\(V_j\) with

\[
V_j^TM_j^0V_j=I,\qquad
V_j^TK_j^0V_j=\operatorname{diag}(d_j).
\]

The tensor transform \(V_x\otimes V_y\) diagonalizes the identified penalty
by congruence, with diagonal entries

\[
\lambda_xd_{x,i}+\lambda_yd_{y,k}.
\]

For the stated cubic spaces, each marginal second-derivative kernel consists
of constants and linear functions. Restriction removes its constant direction.
The centered interaction penalty therefore has exactly one null direction,

\[
(x-\mu_xx)(y-\mu_yy),
\]

for positive \(\lambda_x,\lambda_y\). It must be available in the initial
interaction space. Joint nuisance null directions must also be represented.
Zero smoothing parameters would enlarge the kernel and require a revised
nullspace analysis.

Deleting a constant column from an uncentered generalized eigensystem is
insufficient for arbitrary centering measures. Generalized eigenvectors also
must not be treated as Euclidean orthonormal. With an invertible rich transform
\(T\), the correct Hessian and residual maps are

\[
H_c=T^THT,\qquad r_c=T^Tr.
\]

The residual is a covector. Mapping it with \(T^{-1}\) would give the wrong
certificate. Penalty-null coupling to the observed design still requires the
Schur correction in the Gaussian analysis. Diagonalizing the penalty alone
does not diagonalize that design or the active THB penalty.

Marginal dense eigensystems cost \(O(p_x^3+p_y^3)\) setup. Separable modal
applications cost \(O(P(p_x+p_y))\), and storage costs
\(O(P+p_x^2+p_y^2)\). Explicit \(P\)-by-\(P\) tensor eigenvector matrices are
unnecessary. The full virtual residual, coefficient lift and transformed
scratch still have \(P\)-dependent cost. Their storage is part of peak RSS.
Numerical certification of the marginal eigensystems, inverse applications
and null/range separation remains open.

## 10. What the published THB results establish

Buffa, Giannelli, Morgenstern and Peterseim analyze nested tensor spline spaces
with fixed degree and standard dyadic refinement. Their stated initial grids
are uniform hypercube grids in parameter coordinates. For strictly admissible
meshes, their refinement algorithm preserves the admissibility class.
Theorem 12 bounds cumulative inserted cells by a constant, depending on
dimension, degree and admissibility, times cumulative marked cells. The cited
local-overlap estimate is controlled by \(m\prod_j(d_j+1)\), hence \(16m\) for
bicubics. A single refinement step need not satisfy the same ratio bound.
[Primary source, Sections 2 and 3](https://arxiv.org/pdf/1509.05566).

These results do not establish closure for arbitrary anisotropic refinements,
product-centering constraints, observed-sample rank decisions, statistical
marking quality, or Gaussian Krylov iteration counts. They apply to the raw
mesh hierarchy. The proposed projection has globally supported corrections,
whose costs were derived separately above.

Giannelli and coauthors describe hierarchical domain trees, recursive
subdivision, local representations and precomputed THB coefficients. Their
implementation discussion describes a time/memory tradeoff and reports storage
behavior for graded examples. It supports using explicit lifetime and retained
memory accounting; it is not a proof of the projected-frame operator bound.
[Primary source, Section 2](https://www.ag.jku.at/pubs/2016gjkmss.pdf).

## 11. Alternatives and failure cases

- Constrained raw THB spaces remain a valid model, but they can suppress local
  additions or require expensive additive closure. Dense nullspace elimination
  can destroy local evaluation structure.
- Column selection using a rank-revealing factorization can give an independent
  projected basis. Rank decisions, fill and transfer into the newly selected
  basis need explicit cost and arithmetic analysis.
- Compact centered wavelet details could avoid a redundant projected frame,
  but require another basis construction, boundary treatment and stability
  argument. This note establishes none of those properties.
- Sparse-grid truncation remains a separate representation option with its
  mixed-regularity assumptions and diagonal-ridge failure cases. The isotropic
  THB closure result does not prove its efficiency.
- Materializing the centered fine tensor or a dense covariance during every
  active solve would reintroduce virtual-width costs that the operator design
  aims to avoid.

Weak smoothing can make the penalty-based certificate too loose. Frame
conditioning can require too many iterations. Refinement of a diagonal ridge
can approach uniform resolution. Marginal integration, closure, certification
or later automatic-smoothing work can consume the saved coefficient work.
Each is an admissible unfavorable outcome, not evidence to omit from a report.

## 12. Verification and decision gates

1. Check prolongation, prediction reconstruction, product centering, adjoints
   and each penalty congruence against small explicit fine tensors. Mutate the
   mass factors, measure normalization, physical scaling and one transfer to
   demonstrate detection of changed targets.
2. Compare the rank of \(D_A\) with the projected fine reference for coarse and
   uniform fine tensors, isolated patches, full marginal strips and boundary
   refinements. Assert subspaces and reconstructed functions rather than signs
   of roundoff eigenvalues.
3. Check additive annihilation and preservation of the centered bilinear
   penalty-null function. Include correlated and degenerate observed designs
   and the full joint nuisance-null identification check.
4. Compare fixed-Gaussian certificates with reference objective and prediction
   errors for smooth global structure, localized bumps, diagonal ridges,
   boundary features and cancelling omitted scores. A local marking score is
   not the full omitted residual.
5. Include near-additive cancellation and the largest permitted dyadic levels.
   Verify distinct knot endpoints, quadrature exactness at the polynomial level
   and error envelopes derived from dimensions, epsilon, norms and conditioning.
   Certification must refuse unresolved rank or SPD separation.
6. Record complete construction, measure setup, search, closure, all iterations,
   certificate passes and finalization. Report \(n,p,N,L,P\), identified rank,
   projected marginal knots, actual backend dispatch, retained model memory
   and peak live process state. Count simultaneous old/new hierarchies and
   high-precision or virtual-space scratch.

Proceed to an implementation experiment only with these contracts explicit.
The Gaussian pilot must first save complete-fit resources at its declared
approximation budget relative to the same fixed fine model. Automatic
smoothing, covariance approximation and fresh held-out predictive quality have
separate gates. The fixed-model approximation bound is not a population-error
or post-selection inference theorem.

## 13. Source-to-code pointers

At the pinned source baseline:

- [TensorInteraction](../../src/superglm/features/interaction.py) constructs
  centered marginal products, routes eligible cubic regression marginals through
  cardinal geometry and combines marginal penalties with identity factors.
  The normalization helper is near line 1367, marginal routing near line 1520,
  and ordinary/discrete tensor penalties near lines 1700 and 1728.
- [Marginal identifiability](../../src/superglm/features/_spline_identifiability.py)
  derives its centering constraint from geometry mass and uses a complete QR
  projection. Applying a similarly dense projection to a growing hierarchical
  coefficient space is not the proposed operator construction.
- [Derivative penalty integration](../../src/superglm/features/_spline_penalties.py)
  contains the quadrature helper discussed in Section 6.

The exact centering projector, raw-to-identified embedding, mixed-derivative
rank identity and leaf/marginal cost construction in this note are derived
results awaiting implementation tests and kernel-specific arithmetic bounds.
The two primary publications cited above do not prove those additional claims.
