import Mathlib

/-!
# Exact finite-dimensional quadratic identities

This file proves exact algebra over the real numbers. It does not formalize
positive definiteness, statistical covariance, Schur bounds, floating-point
rounding, or an implementation of the SuperGLM solver.

Repository base: 7c4e70ffac99c9a70adf90e5b21d9735109c4675.
Date: 2026-09-13.
-/

open Matrix

namespace SuperGLM

variable {ι : Type*} [Fintype ι]

/-- A general real quadratic objective, including an arbitrary constant. -/
noncomputable def quadraticObjective
    (H : Matrix ι ι ℝ) (b x : ι → ℝ) (c : ℝ) : ℝ :=
  (1 / 2 : ℝ) * (x ⬝ᵥ (H *ᵥ x)) - b ⬝ᵥ x + c

/-- Stationarity makes the coefficient error map to the true residual. -/
theorem residual_identity
    (H : Matrix ι ι ℝ) (b xStar x : ι → ℝ)
    (hStationary : H *ᵥ xStar = b) :
    H *ᵥ (xStar - x) = b - H *ᵥ x := by
  rw [mulVec_sub, hStationary]

/-- A symmetric quadratic's gap is half its error energy at any stationary reference. -/
theorem quadratic_gap
    (H : Matrix ι ι ℝ) (b xStar x : ι → ℝ) (c : ℝ)
    (hSymmetric : Hᵀ = H) (hStationary : H *ᵥ xStar = b) :
    quadraticObjective H b x c - quadraticObjective H b xStar c =
      (1 / 2 : ℝ) * ((xStar - x) ⬝ᵥ (H *ᵥ (xStar - x))) := by
  have hCross : x ⬝ᵥ (H *ᵥ xStar) = xStar ⬝ᵥ (H *ᵥ x) := by
    simpa only [hSymmetric] using dotProduct_transpose_mulVec H x xStar
  unfold quadraticObjective
  rw [← hStationary]
  simp only [mulVec_sub, dotProduct_sub, sub_dotProduct]
  rw [dotProduct_comm (H *ᵥ xStar) x,
      dotProduct_comm (H *ᵥ xStar) xStar, hCross]
  ring

/-- An exact supplied left inverse converts error energy into residual energy. -/
theorem residual_energy_of_left_inverse [DecidableEq ι]
    (H M : Matrix ι ι ℝ) (b xStar x : ι → ℝ)
    (hLeftInverse : M * H = 1) (hStationary : H *ᵥ xStar = b) :
    (xStar - x) ⬝ᵥ (H *ᵥ (xStar - x)) =
      (b - H *ᵥ x) ⬝ᵥ (M *ᵥ (b - H *ᵥ x)) := by
  have hResidual := residual_identity H b xStar x hStationary
  have hSolve : M *ᵥ (b - H *ᵥ x) = xStar - x := by
    rw [← hResidual, mulVec_mulVec, hLeftInverse, one_mulVec]
  calc
    (xStar - x) ⬝ᵥ (H *ᵥ (xStar - x)) =
        (xStar - x) ⬝ᵥ (b - H *ᵥ x) := by rw [hResidual]
    _ = (b - H *ᵥ x) ⬝ᵥ (xStar - x) := dotProduct_comm _ _
    _ = (b - H *ᵥ x) ⬝ᵥ (M *ᵥ (b - H *ᵥ x)) := by rw [hSolve]

/-- Combine the symmetric objective gap with a supplied exact left inverse. -/
theorem quadratic_gap_residual [DecidableEq ι]
    (H M : Matrix ι ι ℝ) (b xStar x : ι → ℝ) (c : ℝ)
    (hSymmetric : Hᵀ = H) (hLeftInverse : M * H = 1)
    (hStationary : H *ᵥ xStar = b) :
    quadraticObjective H b x c - quadraticObjective H b xStar c =
      (1 / 2 : ℝ) * ((b - H *ᵥ x) ⬝ᵥ (M *ᵥ (b - H *ᵥ x))) := by
  rw [quadratic_gap H b xStar x c hSymmetric hStationary,
      residual_energy_of_left_inverse H M b xStar x hLeftInverse hStationary]

end SuperGLM

#print axioms SuperGLM.residual_identity
#print axioms SuperGLM.quadratic_gap
#print axioms SuperGLM.residual_energy_of_left_inverse
#print axioms SuperGLM.quadratic_gap_residual
