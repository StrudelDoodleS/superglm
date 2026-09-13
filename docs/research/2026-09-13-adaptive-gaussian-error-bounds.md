# Fixed Gaussian error bounds for adaptive interaction spaces

Date: 2026-09-13.

Repository base: 7c4e70ffac99c9a70adf90e5b21d9735109c4675, SuperGLM 0.33.0.

This memo records the mathematical findings of the adaptive-interaction
certification review. The motivating directions are C15 and C16 in
[the numerical and function-space roadmap additions](2026-09-superglm-feature-roadmap-additions.md).
The representation investigation is a separate workstream. All fitting and
certification here retain every observation.

**Status.** Results labelled "Derived here" have the algebraic proofs below.
Published results are identified by their primary sources. Numerical obligations
are requirements for a future implementation, not completed numerical proofs.
Formal verification is recorded separately at the end of this memo. None of
these statements establishes an empirical SuperGLM speedup, lower peak RSS, or
improved population prediction.

## 1. Fixed target and permitted adaptation

Fix a rich identifiable coefficient space and the objective

\[
F(\beta)=\frac12\|W^{1/2}(y-X\beta)\|_2^2
       +\frac12\beta^\top S\beta,\qquad
H=X^\top WX+S\succ0,\qquad b=X^\top Wy.
\]

Assume \(W\succeq0\), \(S\succeq0\), and real arithmetic. Cost statements below
assume diagonal observation weights. The response \(y\) already incorporates
the declared offset convention. Freeze the observations, weights, response,
offset, basis, physical coordinate scaling, boundary and extrapolation rules,
centering, penalty components and normalization, smoothing parameters,
dispersion convention, and rank decisions. A posterior statement using
\(\phi H^{-1}\) additionally assumes \(\phi>0\) and that this covariance matches
the declared likelihood/prior scaling.

The rich target must already have a fixed identifiable representation. Product
ANOVA constraints remove functional overlap with additive terms, but correlated
observations can still leave unidentifiable penalty-null combinations. For this
quadratic problem, positive definiteness is equivalent to
\(\ker(W^{1/2}X)\cap\ker S=\{0\}\).

An adaptive map \(P\) defines \(\widetilde\beta=P\theta\). It must use
\(XP\) and \(P^\top SP\), separately for every smoothing component. All main
effects and nuisance directions readjust jointly. Adaptive selection may depend
on the response while remaining a numerical method for this fixed target.
Changing the penalty or treating the selected subspace as a new inferential
model has different semantics.

The columns of \(P\) may be redundant. Exact reduced minimization gives a unique
rich fitted vector but need not give unique frame coefficients. The algebraic
solution is
\[
\widetilde\beta=P(P^\top HP)^\dagger P^\top b.
\]
This identity does not prescribe a dense pseudoinverse. A quotient or rank-aware
solver needs its own numerical and complexity justification.

## 2. Residual, energy and objective identities

**Derived here.** For any approximate rich vector, including an inexact reduced
fit, let
\[
r=b-H\widetilde\beta,\qquad
e=\beta^\star-\widetilde\beta,\qquad
\beta^\star=H^{-1}b.
\]
Then
\[
He=r,\qquad
E^2=e^\top He=r^\top H^{-1}r,\qquad
F(\widetilde\beta)-F(\beta^\star)=\frac12 E^2.
\]

**Proof.** The stationary equation is \(H\beta^\star=b\), so \(He=r\).
Substitution gives the residual expression for the energy. Expanding the
quadratic around its stationary point cancels its linear term and leaves
\(\frac12(\widetilde\beta-\beta^\star)^\top
H(\widetilde\beta-\beta^\star)\). Symmetry makes this equal to \(E^2/2\).
The polynomial gap identity itself needs symmetry and stationarity, not positive
definiteness. Positive definiteness supplies uniqueness and a nonnegative error
norm.

At an exact reduced optimum, \(P^\top r=0\). This follows by differentiating
with respect to the frame coefficients and holds for redundant \(P\).
An inexact reduced solve must retain its active residual.

After choosing independent active and omitted coordinates, elimination of all
active directions gives
\[
C=H_{OO}-H_{OA}H_{AA}^{-1}H_{AO}.
\]
At an exact active optimum, \(E^2=r_O^\top C^{-1}r_O\). This expression requires
the entire omitted space and its cross terms. Individual local scores can guide
refinement but do not add to a general certificate.

## 3. A penalty-nullspace upper bound

**Derived here.** Let \(T=[N,Z]\) be invertible on the fixed rich identifiable
space, with \(N\) spanning \(\ker S\) and \(Z\) a direct complement. The columns
need not be Euclidean orthonormal. Define
\[
r_c=T^\top r,\quad
A=W^{1/2}XN,\quad B=W^{1/2}XZ,\quad
K=A^\top A,\quad C=A^\top B,\quad L=Z^\top SZ.
\]
Then \(K\succ0\) and \(L\succ0\). Put
\[
\Pi_A=AK^{-1}A^\top,\qquad
D=B^\top(I-\Pi_A)B\succeq0,\qquad
q=r_{c,z}-C^\top K^{-1}r_{c,a}.
\]
The exact identity and upper bound are
\[
E^2=r_{c,a}^\top K^{-1}r_{c,a}
       +q^\top(L+D)^{-1}q
\le
r_{c,a}^\top K^{-1}r_{c,a}+q^\top L^{-1}q.
\tag{1}
\]

**Proof.** Since \(SN=0\), the null block of \(T^\top HT\) is \(K\), its
cross block is \(C\), and its remaining block is \(B^\top B+L\).
For nonzero \(a\), \(a^\top Ka=(Na)^\top H(Na)>0\). For nonzero \(z\),
\(Zz\notin\ker S\), so \(z^\top Lz>0\). Schur elimination gives the equality.
The matrix \(I-\Pi_A\) is an orthogonal projector, so \(D\succeq0\), and inverse
order gives the inequality. Equivalently,
\[
G=
\begin{bmatrix}
K&C\\ C^\top&C^\top K^{-1}C+L
\end{bmatrix}\succ0,\qquad
T^\top HT-G=
\begin{bmatrix}0&0\\0&D\end{bmatrix}\succeq0.
\]
The upper bound in (1) is \(r_c^\top G^{-1}r_c\).

The formulas use congruence for Hessians and penalties, and transpose
transformation for residuals. In particular, the transformed residual is
\(T^\top r\), not \(T^{-1}r\). Generalized eigenvectors cannot be treated as
Euclidean orthonormal.

**Derived here.** The bound is independent of the choice of bases and complement.
The physical null correction
\[
\delta=N(N^\top HN)^{-1}N^\top r
\]
is invariant under a change of nullspace basis. The corrected residual
\(\bar r=r-H\delta\) annihilates \(\ker S\), and
\[
q^\top L^{-1}q=\bar r^\top S^\dagger\bar r.
\]
To see this, \(x=ZL^{-1}Z^\top\bar r\) satisfies \(Sx=\bar r\): testing the
difference against both \(N\) and \(Z\) gives zero, and \(T\) is invertible.
Solutions of this penalty equation differ only in \(\ker S\), which
\(\bar r\) annihilates. This also proves invariance when a complement column
is changed by adding nullspace columns.

Include all penalty-null directions in the initial active space. Their residual
vanishes at an exact active optimum, but finite-precision residuals and the
cross correction in \(q\) must remain in a numerical enclosure.

The upper bound can be arbitrarily loose. If \(D=\gamma L\) along the relevant
direction, its positive-space contribution exceeds the exact contribution by
a factor \(1+\gamma\). Weak penalties can therefore force almost full refinement
or further correction work.

## 4. Structured application and full costs

Compute the full residual as
\[
r=X^\top W(y-X\widetilde\beta)-S\widetilde\beta.
\]
The null cross correction can use the action
\(Z^\top X^\top WXN K^{-1}r_{c,a}\); it does not require storing the dense
matrix \(C\). With nullity \(k\), setup of the null data Gram and its factor costs
approximately \(O(nk^2+k^3)\), plus basis application. Stable small factorizations
and streaming can avoid a retained \(n\times k\) array. Their error bounds and
extra passes remain part of the cost.

For the fixed physical tensor penalty
\[
S=\lambda_x K_x\otimes M_y+\lambda_y M_x\otimes K_y,
\]
first restrict each marginal to its fixed centering constraint. For normalized
centering measures, this means a basis of \(\{u:c_j^\top u=0\}\), with
\(c_j=\int b_j\,d\mu_j\). Form the restricted mass and stiffness matrices before
solving their generalized eigensystems. Deleting a constant column from an
unrestricted eigensystem is insufficient for arbitrary centering measures.

The correctly transformed tensor penalty has modal entries
\[
\lambda_x d_{x,i}+\lambda_y d_{y,j}.
\]
With positive smoothing parameters, centered bilinear \(xy\) is the remaining
interaction null mode. Zero smoothing parameters expand the kernel. An
implementation must either handle that changed support explicitly or refuse
the positive-face formula.

For marginal widths \(p_x,p_y\) and \(P=p_xp_y\), separable setup costs
\(O(p_x^3+p_y^3)\), application costs \(O(P(p_x+p_y))\), and storage costs
\(O(P+p_x^2+p_y^2)\). These exclude the other terms, nullspace construction,
error enclosures and row workspace. An active hierarchical penalty is a
pullback of this operator and need not itself be a Kronecker sum.

Certification scans the virtual rich space. Dense designs cost \(O(nP)\) per
forward/transpose scan; a structured representation must state its actual
support and transform cost. Retain coefficient-sized residual vectors and
bounded row workspace. Dense error arrays or coordinate maps can erase the
claimed savings.

Complete-fit comparisons must include construction, refinement/search,
failed proposals, factor/preconditioner setup, all inner and outer iterations,
certificate scans, finalization and requested inference. Peak RSS counts
simultaneously live old/new contexts, numerical evidence, buffers and outputs.
A full covariance output has a quadratic output-size floor.

The existing housing interaction uses different marginal boundary, knot and
normalization conventions. It is a comparator, not automatically the rich target
of this theorem. Matching a fixed rich fit does not establish population accuracy.

## 5. Optional improvements without a rich inverse

**Derived here.** If observation-space \(Q\) satisfies
\(QQ^\top\preceq I-\Pi_A\), let \(J=B^\top Q\). Then \(JJ^\top\preceq D\),
so replacing \(L\) in (1) by \(L+JJ^\top\) strengthens the bound.
One admissible construction is \(Q=(I-\Pi_A)E\), where \(E\) selects observation
coordinate columns. This uses selected rows only to construct a conservative
curvature bound; every fit and residual still uses all observations.

For any vector \(a\),
\[
q^\top(L+JJ^\top)^{-1}q
\le (q-Ja)^\top L^{-1}(q-Ja)+a^\top a,
\]
with equality at the minimizer. Differentiating this positive quadratic in \(a\)
and substituting its stationary value gives the Woodbury identity. This form
avoids subtracting two large Woodbury terms. Accuracy of a small approximate
solve affects sharpness, not the exact-arithmetic upper-bound property.
Storage \(O(P\,\mathrm{cols}(Q))\), construction, factorization and error control
must be counted.

**Derived here.** For any correction \(z\), define \(s=r-Hz\). Expansion gives
\[
E^2=2r^\top z-z^\top Hz+s^\top H^{-1}s.
\tag{2}
\]
A certified bound on the final term permits correction iterations to sharpen
the result. Scalar cancellation in (2) needs its own enclosure.

More generally, a certified \(\alpha M\preceq H\), with \(\alpha>0\) and
\(M\succ0\), implies \(E^2\le r^\top M^{-1}r/\alpha\). A smallest Ritz value is
generally an upper bound on the smallest eigenvalue and cannot establish this
coercivity floor.

**Published result.** Gauss-Radau upper bounds for CG error norms require a
justified prescribed underestimate of the smallest eigenvalue. Their practical
sharpness can deteriorate as Ritz values converge. See
[Meurant and Tichý, The behaviour of the Gauss-Radau upper bound of the error norm in CG](https://arxiv.org/abs/2209.14601).
This reference does not by itself certify SuperGLM's finite-precision operators.

## 6. Numerical enclosure and refusal contract

**Derived here.** Suppose an implementation verifies
\[
T^\top HT\succeq\alpha\widehat G,\qquad
\alpha>0,\quad\widehat G\succ0,
\]
and obtains outward bounds
\[
d_{\rm up}\ge\|\widehat r_c\|_{\widehat G^{-1}},\qquad
\zeta\ge\|r_c-\widehat r_c\|_{\widehat G^{-1}}.
\]
Inverse order and the triangle inequality give the operational bound
\[
E\le E_{\rm up}
=\frac{d_{\rm up}+\zeta}{\sqrt\alpha}.
\tag{3}
\]
For the ideal exact construction, \(\widehat G=G\) and \(\alpha=1\).
Computed geometry requires an admitted lower bound after including its defects.

**Numerical obligations.**

- Recompute a residual using the fixed target operator. A small recursive
  Krylov residual is insufficient.
- Enclose prediction, subtraction, weighting, transpose accumulation, penalty
  application, coordinate maps, null projection, inverse application and norm
  evaluation. The enclosure's own arithmetic needs an outward bound.
- Computed kernel vectors must represent the target kernel, or their defects
  must enter the enclosure. Do not silently zero approximate null or cross
  blocks. Positive computed eigenvalues are not certified lower bounds.
- Prove that small Gram/factor and marginal generalized-eigen reconstruction
  errors compose into the lower operator bound in (3), without materializing an
  unaffordable rich matrix.
- Preserve evidence across reuse only when weights, basis, penalties, precision,
  rank policy and all relevant ownership conditions permit it.
- Refuse when target identity, rank separation, positive lower bounds, finite
  arithmetic or residual error remains unresolved. Higher precision or a
  verified fallback can be attempted. An added ridge or silently dropped
  direction changes the target.

**Published arithmetic framework.** Standard dot-product analysis bounds errors
using \(\gamma_j=ju/(1-ju)\) and sums of absolute products, where \(u\) is unit
roundoff. The chosen count must follow the executed reduction or a proved
conservative bound. Underflow, overflow and arithmetic assumptions require
explicit handling. See
[Higham, Accuracy and Stability of Numerical Algorithms, chapter 3](https://epubs.siam.org/doi/10.1137/1.9780898718027.ch3).
The repository's outward helpers in src/superglm/reml/multi_penalty.py are useful
patterns, not an existing certificate for these new kernels.

The desired approximation accuracy is an application budget. Roundoff
allowances derive from dimensions, arithmetic, norms and conditioning. These
are different quantities. For example, accepting
\(E_{\rm up}\le\epsilon_{\rm num}\sqrt\phi\) would implement a declared
posterior-unit numerical budget; this memo chooses no universal value for
\(\epsilon_{\rm num}\).

The theorem concerns the fixed compiled problem. Continuous-basis evaluation,
row discretization, offset preparation or coefficient-map approximations that
are outside that definition need separate perturbation bounds.

## 7. Predictions and the limit of mean certification

**Derived here.** Cauchy-Schwarz in the \(H\) geometry gives, for every linear
functional \(\ell\),
\[
|\ell^\top e|\le E\sqrt{\ell^\top H^{-1}\ell},\qquad
\|W^{1/2}Xe\|_2\le E.
\]
If the declared fixed-rich posterior covariance is \(\phi H^{-1}\), then
\(E_{\rm up}/\sqrt\phi\) bounds numerical mean displacement in the true
posterior-SD units, simultaneously for every functional with nonzero variance.
This does not calculate those standard deviations or establish population
calibration. Held-out absolute error bounds need the relevant evaluation norm.

**Derived here.** Mean certification does not certify reduced uncertainty.
Set \(R=H^{1/2}P\). Since \(R(R^\top R)^\dagger R^\top=\Pi_R\) is an
orthogonal projector,
\[
H^{-1}-P(P^\top HP)^\dagger P^\top
=H^{-1/2}(I-\Pi_R)H^{-1/2}\succeq0.
\]
It can be nonzero when \(r=0\). Reduced covariance, EDF, traces and determinants
can therefore differ at an exact mean fit. Frequentist sandwich covariance and
smoothing-parameter uncertainty require their own analyses.

## 8. Adversarial fixtures and target mutations

These are analytic mutation fixtures, not executed SuperGLM regressions.
Numerical versions must obey the repository policy on backward error, rank,
subspaces and stable observables.

1. For omitted curvature
   \[
   C=\begin{bmatrix}1&1-\delta\\1-\delta&1\end{bmatrix},\quad
   r=t(1,-1),\quad t^2=\delta,\quad0<\delta<1,
   \]
   the eigenvalues are \(\delta\) and \(2-\delta\). Separate local improvements
   sum to \(t^2=\delta\), while the full objective gap is
   \(\frac12r^\top C^{-1}r=t^2/\delta=1\). An aggregated coarse score cancels
   to zero. Do not test coefficient-forward accuracy near singularity.
2. With \(X=[1,c]\), \(S=\operatorname{diag}(0,L)\), \(L>0\), and residual
   \(r=(1,0)\), the exact squared error is \(1+c^2/L\). Dropping the null
   residual gives zero. Keeping that residual but omitting its cross correction
   gives one. This catches incomplete active-residual handling.
3. With \(X=I_2\), \(S=\operatorname{diag}(0,1)\), \(b=(1,0)\) and active
   \(P=(1,0)^\top\), the residual is zero. Omitted covariance is
   \(\operatorname{diag}(0,1/2)\), and EDF differs by \(1/2\).
4. For scalar \(X=W=y=1\), fixed \(S=1\) gives \(\beta^\star=1/2\).
   Mutating the penalty to \(S'=3\) gives \(\widetilde\beta=1/4\) and a zero
   mutated residual, while the original objective gap is \(1/16\). The target
   identity check must invalidate the mutated certificate.
5. Under nonorthogonal rescaling, transform design, penalty and residual
   consistently and require invariant energies/predictions. Mutating the
   residual transformation from \(T^\top r\) to \(T^{-1}r\) must fail.
6. Test nearly unresolved null modes, inaccurate inverse applications and
   misleading smallest Ritz values. Require a valid bound or refusal.
   A positive computed eigenvalue or a particular roundoff sign is not an
   invariant.

## 9. Automatic smoothing remains a separate proof

Re-estimating smoothing parameters changes the reference Hessian. Profiling
dispersion changes objective or uncertainty conventions. A fixed-parameter
coefficient certificate does not certify an optimum of REML.

**Derived here, conditional only.** Suppose exact profiled negative Gaussian
REML has an interior minimizer \(\rho^\star\), and its Hessian is at least
\(\kappa I\) on the segment connecting \(\rho^\star\) and \(\widehat\rho\).
If the computed gradient has norm at most \(\tau\), with deterministic gradient
error at most \(\eta\), strong monotonicity and Cauchy-Schwarz give
\[
\|\widehat\rho-\rho^\star\|_2\le(\tau+\eta)/\kappa.
\]
This requires fixed rank, an appropriate neighborhood, existence of the
relevant interior minimizer and a proved curvature bound. Those assumptions
have not been established for the proposed adaptive method.

[Wood, Pya and Säfken, Smoothing parameter and model selection for general smooth models](https://arxiv.org/abs/1511.03864)
is the primary smoothing/inference reference. No global LSS branch or
post-selection interval guarantee is asserted here.

**Published results and numerical obligations.** PSD Hutch++ guarantees apply
to suitable symmetric operators. For a smoothing trace, one such operator is
\(S_j^{1/2}H^{-1}S_j^{1/2}\); a nonsymmetric \(H^{-1}S_j\) representation does
not inherit every PSD theorem automatically.
[Meyer, Musco, Musco and Woodruff, Hutch++](https://arxiv.org/abs/2010.09649)
provides the trace-estimation result.
[Ubaru, Chen and Saad, stochastic Lanczos quadrature](https://epubs.siam.org/doi/10.1137/16M1104974)
treat analytic matrix functions of SPD matrices, including log determinants.
Separate stochastic confidence, quadrature error, inexact solves and rounding.
Adaptive repeated evaluations require joint error control. Common probes alone
do not certify convergence.

## 10. Formal verification status

**Lean verified, limited scope.** On 2026-09-13, the first compiler invocation
accepted four theorems in Certificate.lean. The original checked file is in the
ignored .benchmark-artifacts/lean-gaussian-certificate project. The same source
is archived as [Certificate.lean](lean-gaussian-certificate/Certificate.lean),
and was independently checked again in that archive. Its project also builds
the beginner examples described in the [proof tour](lean-gaussian-certificate/README.md).
The file's SHA-256 is
887356aa33dce90017c1fc7f125d9186332f8e939ccf792fd8a717c9ccfa4718.

The statements use Mathlib's finite matrices and vectors over the real numbers.
The dimension is an arbitrary finite index type. Define
\[
Q(x)=\tfrac12 x^\top Hx-b^\top x+c,
\]
where \(c\) is any real constant. The checked theorem names and assumptions are:

| Theorem | Explicit assumptions | Checked conclusion |
| --- | --- | --- |
| SuperGLM.residual_identity | \(Hx^\star=b\) | \(H(x^\star-x)=b-Hx\) |
| SuperGLM.quadratic_gap | \(H^\top=H\), \(Hx^\star=b\) | \(Q(x)-Q(x^\star)=\frac12(x^\star-x)^\top H(x^\star-x)\) |
| SuperGLM.residual_energy_of_left_inverse | \(MH=I\), \(Hx^\star=b\) | \((x^\star-x)^\top H(x^\star-x)=(b-Hx)^\top M(b-Hx)\) |
| SuperGLM.quadratic_gap_residual | \(H^\top=H\), \(MH=I\), \(Hx^\star=b\) | \(Q(x)-Q(x^\star)=\frac12(b-Hx)^\top M(b-Hx)\) |

The inverse is a supplied matrix with an explicit left-inverse hypothesis.
The file does not prove that a numerical inverse has this property. Stationarity
and symmetry are hypotheses, not computed certificates. The energy identity
holds algebraically without positive definiteness; interpreting it as a squared
error norm needs the separately stated positive-definiteness assumptions.
Expanding the original weighted least-squares objective into \(Q\), including
\(c=\frac12y^\top Wy\), remains the handwritten derivation in this memo.

For a readable example, the complete first proof is:

    theorem residual_identity
        (H : Matrix ι ι ℝ) (b xStar x : ι → ℝ)
        (hStationary : H *ᵥ xStar = b) :
        H *ᵥ (xStar - x) = b - H *ᵥ x := by
      rw [mulVec_sub, hStationary]

The declaration states the claim for every allowed matrix and vector satisfying
stationarity. The proof rewrites matrix multiplication over subtraction, then
uses the stationary equation. Lean checks the resulting proof term. This is
a universal exact algebraic proof, rather than a check on selected numerical
examples.

The general gap proof uses the transpose/dot-product identity to establish
symmetry of the cross terms, expands vector subtraction, and invokes Mathlib's
ring tactic for the remaining polynomial identity. The other theorems combine
these identities with the supplied exact left inverse.

The exact successful command, run from the ignored proof project, was:

    /home/max/.elan/bin/lake env lean Certificate.lean

It exited with status 0. All output came from the four explicit dependency
audits at the end of the source:

    'SuperGLM.residual_identity' depends on axioms: [propext, Classical.choice, Quot.sound]
    'SuperGLM.quadratic_gap' depends on axioms: [propext, Classical.choice, Quot.sound]
    'SuperGLM.residual_energy_of_left_inverse' depends on axioms: [propext, Classical.choice, Quot.sound]
    'SuperGLM.quadratic_gap_residual' depends on axioms: [propext, Classical.choice, Quot.sound]

These are standard foundational axioms used by Lean/Mathlib. The file introduces
no new axioms and contains no sorry or admit. The printed dependencies contain
no sorryAx. This audit does not remove the explicit hypotheses in the theorem
statements.

The checked environment was:

    Lean (version 4.33.1, x86_64-unknown-linux-gnu, commit 819816b2e0a3bf405af45ae5c7af2491d8f5bee6, Release)
    Lake version 5.0.0-src+819816b (Lean version 4.33.1)
    Mathlib Git commit 0df444a360eaa60ab8c11dca51a86af692955474

The project pins leanprover/lean4:v4.33.1 and the Mathlib commit above. Exact
dependency identities also appear in its Lake manifest. The original ignored
project's checked metadata SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| lean-toolchain | 3aac669c7a910ec2389f4e4f921b605adf6ebf2d1e0c9b9cd0be4d33f3f5db71 |
| lakefile.toml | ed8d09927d83cc67df5208f82d11c2bce19466c2bcb4a29fe62d129f34b2d2dd |
| lake-manifest.json | 2b50252b52b677e35c636f448788862d25e534bc1d74b4df3c01b60d4bcdf9e5 |

The archive has a different project name and includes the beginner proof target,
so its Lake metadata hashes differ. Its exact source and metadata hashes,
compiler commands, exit statuses and output are recorded in the
[archive validation receipt](lean-gaussian-certificate/validation.json).
The Lean/Mathlib pins and four-theorem source are unchanged.

The Schur bound, congruence invariance, inverse order, covariance theorem,
optional improvements, floating-point kernels, numerical enclosures and
implementation complexity are not Lean verified by this file. There is no
formal SuperGLM implementation proof or empirical validation claim.
