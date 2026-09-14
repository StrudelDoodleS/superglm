# Coupled interaction operators and cross-group terms

Date: 2026-09-13.

Repository audit base: d7f231e35b196b1421d604596204549d0d28ea49.

This note records exact algebra relevant to a GAM with many interaction groups.
It complements the
[fixed Gaussian error-bound memo](2026-09-13-adaptive-gaussian-error-bounds.md).
The accompanying source is
[InteractionOperator.lean](lean-gaussian-certificate/InteractionOperator.lean).
Lean accepted all four theorems after serial benchmark and profiling workers
finished. The exact command, output, hashes and scope are archived in
[the validation record](lean-gaussian-certificate/interaction-operator-validation.json).

## Full Hessian action

For finite real matrices with compatible dimensions,
\[
H=X^\top WX+S
\quad\Longrightarrow\quad
Hv=X^\top\{W(Xv)\}+Sv.
\]
This follows from distribution over matrix addition and associativity of matrix
and matrix-vector multiplication. Symmetry, positive definiteness and diagonal
weights are unnecessary for the algebraic equality.

The right side requires the actions of \(X\), \(X^\top\), \(W\) and \(S\).
It has no Gram-matrix operand. This permits an implementation based on composed
operator actions, subject to its own storage and numerical proofs.

## Every group receives all cross-group contributions

Let the finite group index be \(g\), let \(X_g\) have \(p_g\) coefficient
columns, and let \(v_g\in\mathbb R^{p_g}\). The widths may differ. Accumulate the
shared row vector
\[
t=\sum_h X_hv_h.
\]
For each output group,
\[
X_g^\top Wt
=\sum_h(X_g^\top WX_h)v_h.
\]
This is a finite distributive identity. Accumulating rows before applying each
group's transpose preserves every cross-group term, including \(h\ne g\).
It does not require storing those cross-product blocks.

The penalty need not be block diagonal. For arbitrary compatible blocks
\(S_{gh}\), the full group action is
\[
X_g^\top Wt+\sum_h S_{gh}v_h
=\sum_h(X_g^\top WX_h+S_{gh})v_h.
\]
The Lean statements use a dependent coefficient index type for each group, so
they do not impose equal group widths or pad groups to a common width.

## A cross-term counterexample

Take one observation, two groups with one coefficient each, \(W=[1]\),
\[
X=[1\;\;1],\qquad S=I_2,\qquad v=(1,1)^\top.
\]
Then
\[
X^\top X+S=
\begin{bmatrix}2&1\\1&2\end{bmatrix},\qquad
(X^\top X+S)v=(3,3)^\top.
\]
Discarding the cross-group data terms gives \(2I_2\), whose action on \(v\) is
\((2,2)^\top\). The identity penalty does not make dropping the data cross
terms exact. A block approximation used as a preconditioner needs a separate
convergence and stopping analysis.

## Scope and verification

The checked source has four theorems:

- SuperGLM.coupled_hessian_action.
- SuperGLM.group_row_accumulation.
- SuperGLM.group_penalized_row_accumulation.
- SuperGLM.dropping_cross_terms_changes_action.

These concern exact finite-real algebra. Mathlib's matrix type represents
mathematical functions, so a proof using that type is not an allocation or
runtime proof for Python, NumPy, BLAS or any production solver.

The source does not prove a complexity bound, memory saving, floating-point
accuracy, conditioning, positive definiteness, convergence, statistical
uncertainty, or a SuperGLM matrix-free implementation. In particular, an exact
operator identity does not make its group passes, transforms, penalty actions,
iterations or smoothing work free.

The successful command, run from docs/research/lean-gaussian-certificate, was:

    /home/max/.elan/bin/lake env lean InteractionOperator.lean

It exited with status 0 on its first invocation. Its complete output was:

    'SuperGLM.coupled_hessian_action' depends on axioms: [propext, Classical.choice, Quot.sound]
    'SuperGLM.group_row_accumulation' depends on axioms: [propext, Classical.choice, Quot.sound]
    'SuperGLM.group_penalized_row_accumulation' depends on axioms: [propext, Classical.choice, Quot.sound]
    'SuperGLM.dropping_cross_terms_changes_action' depends on axioms: [propext, Classical.choice, Quot.sound]

The file contains no sorry, admit or new axiom declarations. The reported
dependencies are standard Lean/Mathlib foundations and contain no sorryAx.

The verified environment was Lean 4.33.1, compiler commit
819816b2e0a3bf405af45ae5c7af2491d8f5bee6; Lake 5.0.0-src+819816b; and Mathlib
commit 0df444a360eaa60ab8c11dca51a86af692955474. The source SHA-256 is
70714fbb2e2e3594df71535f71d80c748665191d02446d401deb30b346b8c4e4.
The validation record includes the exact metadata hashes for this compiler
invocation. It predates the parent's addition of InteractionOperator as a
default Lake library target; those metadata hashes are a historical snapshot.
The source hash identifies the unchanged checked theorem file. After adding
the default target, the parent also ran `/home/max/.elan/bin/lake build`:
exit status 0, `Build completed successfully (8711 jobs).` This integrated check
is appended separately to the validation receipt with the updated Lake-file
hash. The four new theorems and the six earlier theorem declarations are all
included in the default build.
