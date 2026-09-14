import Mathlib

/-!
# Exact coupled interaction operators

These statements concern finite matrices over the real numbers. They show that
row accumulation retains every cross-group contribution in the full operator.
They do not prove execution complexity, floating-point accuracy, conditioning,
positive definiteness, convergence, or a production matrix-free implementation.

Repository audit base: d7f231e35b196b1421d604596204549d0d28ea49.
Date: 2026-09-13.
-/

open Matrix
open scoped BigOperators

namespace SuperGLM

variable {Obs Coef : Type*} [Fintype Obs] [Fintype Coef]

/-- A full penalized Hessian action is the composition of design actions. -/
theorem coupled_hessian_action
    (X : Matrix Obs Coef ℝ) (W : Matrix Obs Obs ℝ)
    (S : Matrix Coef Coef ℝ) (v : Coef → ℝ) :
    (Xᵀ * W * X + S) *ᵥ v =
      Xᵀ *ᵥ (W *ᵥ (X *ᵥ v)) + S *ᵥ v := by
  simp only [add_mulVec, mulVec_mulVec, Matrix.mul_assoc]

variable {Group : Type*} [Fintype Group]
variable {Width : Group → Type*} [∀ g, Fintype (Width g)]

/--
Each output group receives contributions from all input groups. The group
coefficient dimensions may differ.
-/
theorem group_row_accumulation
    (X : (g : Group) → Matrix Obs (Width g) ℝ)
    (W : Matrix Obs Obs ℝ)
    (v : (g : Group) → Width g → ℝ) (g : Group) :
    (X g)ᵀ *ᵥ (W *ᵥ (∑ h, X h *ᵥ v h)) =
      ∑ h, ((X g)ᵀ * W * X h) *ᵥ v h := by
  simp only [mulVec_sum, mulVec_mulVec, Matrix.mul_assoc]

/--
The same identity holds with arbitrary penalty blocks, including penalties
that themselves couple distinct groups.
-/
theorem group_penalized_row_accumulation
    (X : (g : Group) → Matrix Obs (Width g) ℝ)
    (W : Matrix Obs Obs ℝ)
    (S : (g h : Group) → Matrix (Width g) (Width h) ℝ)
    (v : (g : Group) → Width g → ℝ) (g : Group) :
    (X g)ᵀ *ᵥ (W *ᵥ (∑ h, X h *ᵥ v h)) +
        ∑ h, S g h *ᵥ v h =
      ∑ h, ((X g)ᵀ * W * X h + S g h) *ᵥ v h := by
  simp only [mulVec_sum, mulVec_mulVec, add_mulVec, Finset.sum_add_distrib,
    Matrix.mul_assoc]

/--
One observation and two one-coefficient groups suffice for a counterexample.
The full data Gram has cross terms equal to one. Replacing it by its diagonal
changes the action, even with an identity penalty.
-/
theorem dropping_cross_terms_changes_action :
    let X : Matrix (Fin 1) (Fin 2) ℝ := !![1, 1]
    let v : Fin 2 → ℝ := ![1, 1]
    (Xᵀ * X + (1 : Matrix (Fin 2) (Fin 2) ℝ)) *ᵥ v ≠
      ((1 : Matrix (Fin 2) (Fin 2) ℝ) + 1) *ᵥ v := by
  dsimp
  intro h
  have hFirst := congrFun h (0 : Fin 2)
  norm_num [Matrix.mulVec, Matrix.mul_apply, dotProduct,
    Fin.sum_univ_two, Fin.sum_univ_one] at hFirst

end SuperGLM

#print axioms SuperGLM.coupled_hessian_action
#print axioms SuperGLM.group_row_accumulation
#print axioms SuperGLM.group_penalized_row_accumulation
#print axioms SuperGLM.dropping_cross_terms_changes_action
