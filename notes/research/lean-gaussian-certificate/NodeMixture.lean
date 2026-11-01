import Mathlib

/-!
# Issue 544, stage 2: derivatives of a node mixture `log Σ ω_i exp(l_i)` to order 4

These statements are about real numbers. They certify, along one parameter direction `θ`,
the formula `quadrature_window_jets` in
`src/superglm/distributional/kernels/_response_tails.py` uses for a narrow window:

    F_I = sum over set partitions pi of I of kappa_p(l_B : B in pi),

the joint cumulants under the node weights `p_i ∝ ω_i exp(l_i)` of the blocks' node
derivatives. On the diagonal `I = (θ, θ, θ, θ)` the partitions of four positions with two or
more blocks are 4 of shape (3, 1), 3 of shape (2, 2), 6 of shape (2, 1, 1) and 1 of shape
(1, 1, 1, 1), so

    F'''' = E[l''''] + 4 Cov(l''', l') + 3 Var(l'') + 6 E[c'' c'^2] + (E[c'^4] - 3 E[c'^2]^2),

with `c = l - E[l]` the node values centred at their mixture means (`_node_cumulant`, whose
fourth cumulant subtracts the three pairings). Orders 1-3 likewise.

The statements are univariate. The code's multivariate entries are the polarisation of
these (a symmetric multilinear form is determined by its diagonal), which is not proved
here. The node functions `l_i` are assumed four times differentiable on all of `ℝ` with
the given derivatives; the weights `ω_i` positive and the node set nonempty. Gauss-Legendre
truncation and floating-point accuracy are not covered.

Repository base: ff2ed52d (branch `claude/issue-544-gamma-bounds`). Date: 2026-11-01.
-/

namespace SuperGLM.Issue544.Mixture

open Finset Real

variable {ι : Type*} (s : Finset ι) (ω : ι → ℝ) (l l1 l2 l3 l4 : ι → ℝ → ℝ)

/-- `Z_0 = Σ ω_i e^(l_i)`. -/
noncomputable def Z0 (θ : ℝ) : ℝ := ∑ i ∈ s, ω i * Real.exp (l i θ)

/-- `Z_1 = Z_0'`. -/
noncomputable def Z1 (θ : ℝ) : ℝ := ∑ i ∈ s, ω i * Real.exp (l i θ) * l1 i θ

/-- `Z_2 = Z_0''`. -/
noncomputable def Z2 (θ : ℝ) : ℝ :=
  ∑ i ∈ s, ω i * Real.exp (l i θ) * (l2 i θ + l1 i θ * l1 i θ)

/-- `Z_3 = Z_0'''`. -/
noncomputable def Z3 (θ : ℝ) : ℝ :=
  ∑ i ∈ s, ω i * Real.exp (l i θ) *
    (l3 i θ + 3 * l2 i θ * l1 i θ + l1 i θ * l1 i θ * l1 i θ)

/-- `Z_4 = Z_0''''`. -/
noncomputable def Z4 (θ : ℝ) : ℝ :=
  ∑ i ∈ s, ω i * Real.exp (l i θ) *
    (l4 i θ + 4 * l3 i θ * l1 i θ + 3 * l2 i θ * l2 i θ + 6 * l2 i θ * l1 i θ * l1 i θ +
      l1 i θ * l1 i θ * l1 i θ * l1 i θ)

/-- The node mixture `F(θ) = log Σ ω_i exp(l_i(θ))`. -/
noncomputable def F (θ : ℝ) : ℝ := Real.log (Z0 s ω l θ)

/-- The moment-cumulant forms of `F'` to `F''''` in `μ_k = Z_k/Z_0`. -/
noncomputable def F1 (θ : ℝ) : ℝ := Z1 s ω l l1 θ / Z0 s ω l θ

noncomputable def F2 (θ : ℝ) : ℝ :=
  Z2 s ω l l1 l2 θ / Z0 s ω l θ - (Z1 s ω l l1 θ / Z0 s ω l θ) ^ 2

noncomputable def F3 (θ : ℝ) : ℝ :=
  Z3 s ω l l1 l2 l3 θ / Z0 s ω l θ -
    3 * (Z2 s ω l l1 l2 θ / Z0 s ω l θ) * (Z1 s ω l l1 θ / Z0 s ω l θ) +
    2 * (Z1 s ω l l1 θ / Z0 s ω l θ) ^ 3

noncomputable def F4 (θ : ℝ) : ℝ :=
  Z4 s ω l l1 l2 l3 l4 θ / Z0 s ω l θ -
    4 * (Z3 s ω l l1 l2 l3 θ / Z0 s ω l θ) * (Z1 s ω l l1 θ / Z0 s ω l θ) -
    3 * (Z2 s ω l l1 l2 θ / Z0 s ω l θ) ^ 2 +
    12 * (Z2 s ω l l1 l2 θ / Z0 s ω l θ) * (Z1 s ω l l1 θ / Z0 s ω l θ) ^ 2 -
    6 * (Z1 s ω l l1 θ / Z0 s ω l θ) ^ 4

variable {s ω l l1 l2 l3 l4}

section Calculus

variable (hω : ∀ i ∈ s, 0 < ω i) (hs : s.Nonempty)
  (h0 : ∀ i θ, HasDerivAt (l i) (l1 i θ) θ) (h1 : ∀ i θ, HasDerivAt (l1 i) (l2 i θ) θ)
  (h2 : ∀ i θ, HasDerivAt (l2 i) (l3 i θ) θ) (h3 : ∀ i θ, HasDerivAt (l3 i) (l4 i θ) θ)

include hω hs in
lemma Z0_pos (θ : ℝ) : 0 < Z0 s ω l θ :=
  Finset.sum_pos (fun i hi => mul_pos (hω i hi) (Real.exp_pos _)) hs

include h0 in
lemma hasDerivAt_Z0 (θ : ℝ) : HasDerivAt (Z0 s ω l) (Z1 s ω l l1 θ) θ := by
  have h : HasDerivAt (fun y => ∑ i ∈ s, ω i * Real.exp (l i y))
      (∑ i ∈ s, ω i * (Real.exp (l i θ) * l1 i θ)) θ :=
    HasDerivAt.fun_sum fun i _ => ((h0 i θ).exp).const_mul (ω i)
  have e : (∑ i ∈ s, ω i * (Real.exp (l i θ) * l1 i θ)) = Z1 s ω l l1 θ :=
    Finset.sum_congr rfl fun i _ => by ring
  rw [e] at h
  exact h

include h0 h1 in
lemma hasDerivAt_Z1 (θ : ℝ) : HasDerivAt (Z1 s ω l l1) (Z2 s ω l l1 l2 θ) θ := by
  have h : HasDerivAt (fun y => ∑ i ∈ s, ω i * Real.exp (l i y) * l1 i y)
      (∑ i ∈ s, (ω i * (Real.exp (l i θ) * l1 i θ) * l1 i θ +
        ω i * Real.exp (l i θ) * l2 i θ)) θ :=
    HasDerivAt.fun_sum fun i _ => (((h0 i θ).exp).const_mul (ω i)).mul (h1 i θ)
  have e : (∑ i ∈ s, (ω i * (Real.exp (l i θ) * l1 i θ) * l1 i θ +
      ω i * Real.exp (l i θ) * l2 i θ)) = Z2 s ω l l1 l2 θ :=
    Finset.sum_congr rfl fun i _ => by ring
  rw [e] at h
  exact h

include h0 h1 h2 in
lemma hasDerivAt_Z2 (θ : ℝ) : HasDerivAt (Z2 s ω l l1 l2) (Z3 s ω l l1 l2 l3 θ) θ := by
  have h : HasDerivAt (fun y => ∑ i ∈ s, ω i * Real.exp (l i y) * (l2 i y + l1 i y * l1 i y))
      (∑ i ∈ s, (ω i * (Real.exp (l i θ) * l1 i θ) * (l2 i θ + l1 i θ * l1 i θ) +
        ω i * Real.exp (l i θ) * (l3 i θ + (l2 i θ * l1 i θ + l1 i θ * l2 i θ)))) θ :=
    HasDerivAt.fun_sum fun i _ =>
      (((h0 i θ).exp).const_mul (ω i)).mul ((h2 i θ).add ((h1 i θ).mul (h1 i θ)))
  have e : (∑ i ∈ s, (ω i * (Real.exp (l i θ) * l1 i θ) * (l2 i θ + l1 i θ * l1 i θ) +
      ω i * Real.exp (l i θ) * (l3 i θ + (l2 i θ * l1 i θ + l1 i θ * l2 i θ)))) =
      Z3 s ω l l1 l2 l3 θ :=
    Finset.sum_congr rfl fun i _ => by ring
  rw [e] at h
  exact h

include h0 h1 h2 h3 in
lemma hasDerivAt_Z3 (θ : ℝ) : HasDerivAt (Z3 s ω l l1 l2 l3) (Z4 s ω l l1 l2 l3 l4 θ) θ := by
  have hB : ∀ i, HasDerivAt
      (fun y => l3 i y + 3 * l2 i y * l1 i y + l1 i y * l1 i y * l1 i y)
      (l4 i θ + (3 * l3 i θ * l1 i θ + 3 * l2 i θ * l2 i θ) +
        ((l2 i θ * l1 i θ + l1 i θ * l2 i θ) * l1 i θ + l1 i θ * l1 i θ * l2 i θ)) θ :=
    fun i => ((h3 i θ).add (((h2 i θ).const_mul 3).mul (h1 i θ))).add
      (((h1 i θ).mul (h1 i θ)).mul (h1 i θ))
  have h : HasDerivAt (fun y => ∑ i ∈ s, ω i * Real.exp (l i y) *
        (l3 i y + 3 * l2 i y * l1 i y + l1 i y * l1 i y * l1 i y))
      (∑ i ∈ s, (ω i * (Real.exp (l i θ) * l1 i θ) *
          (l3 i θ + 3 * l2 i θ * l1 i θ + l1 i θ * l1 i θ * l1 i θ) +
        ω i * Real.exp (l i θ) *
          (l4 i θ + (3 * l3 i θ * l1 i θ + 3 * l2 i θ * l2 i θ) +
            ((l2 i θ * l1 i θ + l1 i θ * l2 i θ) * l1 i θ + l1 i θ * l1 i θ * l2 i θ)))) θ :=
    HasDerivAt.fun_sum fun i _ => (((h0 i θ).exp).const_mul (ω i)).mul (hB i)
  have e : (∑ i ∈ s, (ω i * (Real.exp (l i θ) * l1 i θ) *
          (l3 i θ + 3 * l2 i θ * l1 i θ + l1 i θ * l1 i θ * l1 i θ) +
        ω i * Real.exp (l i θ) *
          (l4 i θ + (3 * l3 i θ * l1 i θ + 3 * l2 i θ * l2 i θ) +
            ((l2 i θ * l1 i θ + l1 i θ * l2 i θ) * l1 i θ + l1 i θ * l1 i θ * l2 i θ)))) =
      Z4 s ω l l1 l2 l3 l4 θ :=
    Finset.sum_congr rfl fun i _ => by ring
  rw [e] at h
  exact h

/-- `(Z_k/Z_0)' = Z_{k+1}/Z_0 - (Z_k/Z_0)(Z_1/Z_0)`. -/
lemma hasDerivAt_ratio {A B : ℝ → ℝ} {θ : ℝ} (hA : HasDerivAt A (B θ) θ)
    (hZ : HasDerivAt (Z0 s ω l) (Z1 s ω l l1 θ) θ) (hpos : 0 < Z0 s ω l θ) :
    HasDerivAt (fun y => A y / Z0 s ω l y)
      (B θ / Z0 s ω l θ - (A θ / Z0 s ω l θ) * (Z1 s ω l l1 θ / Z0 s ω l θ)) θ := by
  have h : HasDerivAt (fun y => A y / Z0 s ω l y)
      ((B θ * Z0 s ω l θ - A θ * Z1 s ω l l1 θ) / Z0 s ω l θ ^ 2) θ := hA.div hZ hpos.ne'
  refine h.congr_deriv ?_
  have := hpos.ne'
  field_simp

include hω hs h0 in
/-- `F' = Z_1/Z_0`. -/
theorem hasDerivAt_F (θ : ℝ) : HasDerivAt (F s ω l) (F1 s ω l l1 θ) θ :=
  (hasDerivAt_Z0 (s := s) (ω := ω) h0 θ).log (Z0_pos hω hs θ).ne'

include hω hs h0 h1 in
theorem hasDerivAt_F1 (θ : ℝ) : HasDerivAt (F1 s ω l l1) (F2 s ω l l1 l2 θ) θ := by
  have hZ := hasDerivAt_Z0 (s := s) (ω := ω) h0 θ
  have hp := Z0_pos (l := l) hω hs θ
  have h : HasDerivAt (F1 s ω l l1) _ θ :=
    hasDerivAt_ratio (hasDerivAt_Z1 (s := s) (ω := ω) h0 h1 θ) hZ hp
  refine h.congr_deriv ?_
  simp only [F2]
  ring

include hω hs h0 h1 h2 in
theorem hasDerivAt_F2 (θ : ℝ) : HasDerivAt (F2 s ω l l1 l2) (F3 s ω l l1 l2 l3 θ) θ := by
  have hZ := hasDerivAt_Z0 (s := s) (ω := ω) h0 θ
  have hp := Z0_pos (l := l) hω hs θ
  have r1 := hasDerivAt_ratio (hasDerivAt_Z1 (s := s) (ω := ω) h0 h1 θ) hZ hp
  have r2 := hasDerivAt_ratio (hasDerivAt_Z2 (s := s) (ω := ω) h0 h1 h2 θ) hZ hp
  have h : HasDerivAt (F2 s ω l l1 l2) _ θ := r2.sub (r1.pow 2)
  refine h.congr_deriv ?_
  simp only [F3, Nat.cast_ofNat, Nat.reduceSub, pow_one]
  ring

include hω hs h0 h1 h2 h3 in
theorem hasDerivAt_F3 (θ : ℝ) : HasDerivAt (F3 s ω l l1 l2 l3) (F4 s ω l l1 l2 l3 l4 θ) θ := by
  have hZ := hasDerivAt_Z0 (s := s) (ω := ω) h0 θ
  have hp := Z0_pos (l := l) hω hs θ
  have r1 := hasDerivAt_ratio (hasDerivAt_Z1 (s := s) (ω := ω) h0 h1 θ) hZ hp
  have r2 := hasDerivAt_ratio (hasDerivAt_Z2 (s := s) (ω := ω) h0 h1 h2 θ) hZ hp
  have r3 := hasDerivAt_ratio (hasDerivAt_Z3 (s := s) (ω := ω) h0 h1 h2 h3 θ) hZ hp
  have h : HasDerivAt (F3 s ω l l1 l2 l3) _ θ :=
    (r3.sub ((r2.const_mul 3).mul r1)).add ((r1.pow 3).const_mul 2)
  refine h.congr_deriv ?_
  simp only [F4, Nat.cast_ofNat, Nat.reduceSub]
  ring

include hω hs h0 h1 h2 h3 in
/-- `F'''' = F4` at every `θ`. -/
theorem iteratedDeriv_four_F (θ : ℝ) : iteratedDeriv 4 (F s ω l) θ = F4 s ω l l1 l2 l3 l4 θ := by
  have d0 : deriv (F s ω l) = F1 s ω l l1 := funext fun t => (hasDerivAt_F hω hs h0 t).deriv
  have d1 : deriv (F1 s ω l l1) = F2 s ω l l1 l2 :=
    funext fun t => (hasDerivAt_F1 hω hs h0 h1 t).deriv
  have d2 : deriv (F2 s ω l l1 l2) = F3 s ω l l1 l2 l3 :=
    funext fun t => (hasDerivAt_F2 hω hs h0 h1 h2 t).deriv
  have d3 : deriv (F3 s ω l l1 l2 l3) = F4 s ω l l1 l2 l3 l4 :=
    funext fun t => (hasDerivAt_F3 hω hs h0 h1 h2 h3 t).deriv
  simp only [iteratedDeriv_succ, iteratedDeriv_zero, d0, d1, d2, d3]

end Calculus

/-! ## The cumulant form at one point -/

section Cumulants

variable (s ω l) (θ : ℝ)

/-- The node weights `p_i = ω_i e^(l_i)/Z_0` (the code's `share`). -/
noncomputable def p (i : ι) : ℝ := ω i * Real.exp (l i θ) / Z0 s ω l θ

/-- The mixture mean `E_p[X] = Σ p_i X_i`. -/
noncomputable def E (X : ι → ℝ) : ℝ := ∑ i ∈ s, p s ω l θ i * X i

/-- The mixture mean of a node derivative (the code's `means[m]`). -/
noncomputable def mean (L : ι → ℝ → ℝ) : ℝ := E s ω l θ (fun i => L i θ)

/-- A node derivative centred at its mixture mean (the code's `centred`). -/
noncomputable def cen (L : ι → ℝ → ℝ) (i : ι) : ℝ := L i θ - mean s ω l θ L

variable {s ω l θ}

lemma ratio_eq_E {B : ι → ℝ} :
    (∑ i ∈ s, ω i * Real.exp (l i θ) * B i) / Z0 s ω l θ = E s ω l θ B := by
  simp only [E, p, Finset.sum_div]
  exact Finset.sum_congr rfl fun i _ => by ring

lemma E_one (hpos : 0 < Z0 s ω l θ) : ∑ i ∈ s, p s ω l θ i = 1 := by
  simp only [p, ← Finset.sum_div]
  exact div_self hpos.ne'

/-- `E` is linear and `E[c] = c`. -/
lemma E_lin (hpos : 0 < Z0 s ω l θ) (n : ℕ) (X : ι → ℝ) (a : Fin n → ℝ) (f : Fin n → ι → ℝ)
    (c : ℝ) (hX : ∀ i, X i = ∑ k, a k * f k i + c) :
    E s ω l θ X = ∑ k, a k * E s ω l θ (f k) + c := by
  have h1 := E_one hpos (s := s) (ω := ω) (l := l)
  have e1 : E s ω l θ X =
      ∑ i ∈ s, ∑ k, p s ω l θ i * (a k * f k i) + c * ∑ i ∈ s, p s ω l θ i := by
    simp only [E, hX, mul_add, Finset.sum_add_distrib, Finset.mul_sum]
    congr 1
    exact Finset.sum_congr rfl fun i _ => by ring
  have e2 : ∑ k, a k * E s ω l θ (f k) = ∑ i ∈ s, ∑ k, p s ω l θ i * (a k * f k i) := by
    simp only [E, Finset.mul_sum]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun k _ => by ring
  rw [e1, e2, h1, mul_one]

/-- Shorthand inside proofs: the raw moment `E[f]` of a node expression. -/
local notation "𝔼" f => E s ω l θ f

/-- Order 1: `F' = E[l']`. -/
theorem F1_cumulants : F1 s ω l l1 θ = mean s ω l θ l1 := by
  rw [F1, Z1, ratio_eq_E, mean]

/-- Order 2: `F'' = E[l''] + Var(l')`. -/
theorem F2_cumulants (hpos : 0 < Z0 s ω l θ) :
    F2 s ω l l1 l2 θ = mean s ω l θ l2 + E s ω l θ (fun i => cen s ω l θ l1 i ^ 2) := by
  have b2 := E_lin hpos 2 (fun i => l2 i θ + l1 i θ * l1 i θ) ![1, 1]
    ![fun i => l2 i θ, fun i => l1 i θ * l1 i θ] 0
    (fun i => by simp only [Fin.sum_univ_succ, Fin.sum_univ_zero, Matrix.cons_val_zero,
      Matrix.cons_val_succ]; ring)
  have c11 := E_lin hpos 2 (fun i => cen s ω l θ l1 i ^ 2) ![1, -2 * (𝔼 fun i => l1 i θ)]
    ![fun i => l1 i θ * l1 i θ, fun i => l1 i θ] ((𝔼 fun i => l1 i θ) ^ 2)
    (fun i => by simp only [cen, mean, Fin.sum_univ_succ, Fin.sum_univ_zero,
      Matrix.cons_val_zero, Matrix.cons_val_succ]; ring)
  simp only [Fin.sum_univ_succ, Fin.sum_univ_zero, Matrix.cons_val_zero,
    Matrix.cons_val_succ] at b2 c11
  rw [F2, Z1, Z2, ratio_eq_E, ratio_eq_E, b2, c11, mean]
  ring

/-- Order 3: `F''' = E[l'''] + 3 Cov(l'', l') + E[c'^3]`. -/
theorem F3_cumulants (hpos : 0 < Z0 s ω l θ) :
    F3 s ω l l1 l2 l3 θ = mean s ω l θ l3 +
      3 * E s ω l θ (fun i => cen s ω l θ l2 i * cen s ω l θ l1 i) +
      E s ω l θ (fun i => cen s ω l θ l1 i ^ 3) := by
  have b2 := E_lin hpos 2 (fun i => l2 i θ + l1 i θ * l1 i θ) ![1, 1]
    ![fun i => l2 i θ, fun i => l1 i θ * l1 i θ] 0
    (fun i => by simp only [Fin.sum_univ_succ, Fin.sum_univ_zero, Matrix.cons_val_zero,
      Matrix.cons_val_succ]; ring)
  have b3 := E_lin hpos 3 (fun i => l3 i θ + 3 * l2 i θ * l1 i θ + l1 i θ * l1 i θ * l1 i θ)
    ![1, 3, 1] ![fun i => l3 i θ, fun i => l2 i θ * l1 i θ, fun i => l1 i θ * l1 i θ * l1 i θ] 0
    (fun i => by simp only [Fin.sum_univ_succ, Fin.sum_univ_zero, Matrix.cons_val_zero,
      Matrix.cons_val_succ]; ring)
  have c21 := E_lin hpos 3 (fun i => cen s ω l θ l2 i * cen s ω l θ l1 i)
    ![1, -(𝔼 fun i => l1 i θ), -(𝔼 fun i => l2 i θ)]
    ![fun i => l2 i θ * l1 i θ, fun i => l2 i θ, fun i => l1 i θ]
    ((𝔼 fun i => l1 i θ) * (𝔼 fun i => l2 i θ))
    (fun i => by simp only [cen, mean, Fin.sum_univ_succ, Fin.sum_univ_zero,
      Matrix.cons_val_zero, Matrix.cons_val_succ]; ring)
  have c111 := E_lin hpos 3 (fun i => cen s ω l θ l1 i ^ 3)
    ![1, -3 * (𝔼 fun i => l1 i θ), 3 * (𝔼 fun i => l1 i θ) ^ 2]
    ![fun i => l1 i θ * l1 i θ * l1 i θ, fun i => l1 i θ * l1 i θ, fun i => l1 i θ]
    (-(𝔼 fun i => l1 i θ) ^ 3)
    (fun i => by simp only [cen, mean, Fin.sum_univ_succ, Fin.sum_univ_zero,
      Matrix.cons_val_zero, Matrix.cons_val_succ]; ring)
  simp only [Fin.sum_univ_succ, Fin.sum_univ_zero, Matrix.cons_val_zero,
    Matrix.cons_val_succ] at b2 b3 c21 c111
  rw [F3, Z1, Z2, Z3, ratio_eq_E, ratio_eq_E, ratio_eq_E, b2, b3, c21, c111, mean]
  ring

/-- **Order 4**, the formula of `quadrature_window_jets` along one direction:
`F'''' = E[l''''] + 4 Cov(l''', l') + 3 Var(l'') + 6 E[c'' c'^2] + (E[c'^4] - 3 E[c'^2]^2)`,
`c^(k)` the node derivatives centred at their mixture means; the last bracket is the fourth
cumulant, which subtracts the three pairings. -/
theorem F4_cumulants (hpos : 0 < Z0 s ω l θ) :
    F4 s ω l l1 l2 l3 l4 θ = mean s ω l θ l4 +
      4 * E s ω l θ (fun i => cen s ω l θ l3 i * cen s ω l θ l1 i) +
      3 * E s ω l θ (fun i => cen s ω l θ l2 i ^ 2) +
      6 * E s ω l θ (fun i => cen s ω l θ l2 i * cen s ω l θ l1 i ^ 2) +
      (E s ω l θ (fun i => cen s ω l θ l1 i ^ 4) -
        3 * E s ω l θ (fun i => cen s ω l θ l1 i ^ 2) ^ 2) := by
  have b2 := E_lin hpos 2 (fun i => l2 i θ + l1 i θ * l1 i θ) ![1, 1]
    ![fun i => l2 i θ, fun i => l1 i θ * l1 i θ] 0
    (fun i => by simp only [Fin.sum_univ_succ, Fin.sum_univ_zero, Matrix.cons_val_zero,
      Matrix.cons_val_succ]; ring)
  have b3 := E_lin hpos 3 (fun i => l3 i θ + 3 * l2 i θ * l1 i θ + l1 i θ * l1 i θ * l1 i θ)
    ![1, 3, 1] ![fun i => l3 i θ, fun i => l2 i θ * l1 i θ, fun i => l1 i θ * l1 i θ * l1 i θ] 0
    (fun i => by simp only [Fin.sum_univ_succ, Fin.sum_univ_zero, Matrix.cons_val_zero,
      Matrix.cons_val_succ]; ring)
  have b4 := E_lin hpos 5
    (fun i => l4 i θ + 4 * l3 i θ * l1 i θ + 3 * l2 i θ * l2 i θ +
      6 * l2 i θ * l1 i θ * l1 i θ + l1 i θ * l1 i θ * l1 i θ * l1 i θ)
    ![1, 4, 3, 6, 1]
    ![fun i => l4 i θ, fun i => l3 i θ * l1 i θ, fun i => l2 i θ * l2 i θ,
      fun i => l2 i θ * l1 i θ * l1 i θ, fun i => l1 i θ * l1 i θ * l1 i θ * l1 i θ] 0
    (fun i => by simp only [Fin.sum_univ_succ, Fin.sum_univ_zero, Matrix.cons_val_zero,
      Matrix.cons_val_succ]; ring)
  have c31 := E_lin hpos 3 (fun i => cen s ω l θ l3 i * cen s ω l θ l1 i)
    ![1, -(𝔼 fun i => l1 i θ), -(𝔼 fun i => l3 i θ)]
    ![fun i => l3 i θ * l1 i θ, fun i => l3 i θ, fun i => l1 i θ]
    ((𝔼 fun i => l1 i θ) * (𝔼 fun i => l3 i θ))
    (fun i => by simp only [cen, mean, Fin.sum_univ_succ, Fin.sum_univ_zero,
      Matrix.cons_val_zero, Matrix.cons_val_succ]; ring)
  have c22 := E_lin hpos 2 (fun i => cen s ω l θ l2 i ^ 2) ![1, -2 * (𝔼 fun i => l2 i θ)]
    ![fun i => l2 i θ * l2 i θ, fun i => l2 i θ] ((𝔼 fun i => l2 i θ) ^ 2)
    (fun i => by simp only [cen, mean, Fin.sum_univ_succ, Fin.sum_univ_zero,
      Matrix.cons_val_zero, Matrix.cons_val_succ]; ring)
  have c211 := E_lin hpos 5 (fun i => cen s ω l θ l2 i * cen s ω l θ l1 i ^ 2)
    ![1, -2 * (𝔼 fun i => l1 i θ), (𝔼 fun i => l1 i θ) ^ 2, -(𝔼 fun i => l2 i θ),
      2 * (𝔼 fun i => l1 i θ) * (𝔼 fun i => l2 i θ)]
    ![fun i => l2 i θ * l1 i θ * l1 i θ, fun i => l2 i θ * l1 i θ, fun i => l2 i θ,
      fun i => l1 i θ * l1 i θ, fun i => l1 i θ]
    (-(𝔼 fun i => l2 i θ) * (𝔼 fun i => l1 i θ) ^ 2)
    (fun i => by simp only [cen, mean, Fin.sum_univ_succ, Fin.sum_univ_zero,
      Matrix.cons_val_zero, Matrix.cons_val_succ]; ring)
  have c1111 := E_lin hpos 4 (fun i => cen s ω l θ l1 i ^ 4)
    ![1, -4 * (𝔼 fun i => l1 i θ), 6 * (𝔼 fun i => l1 i θ) ^ 2, -4 * (𝔼 fun i => l1 i θ) ^ 3]
    ![fun i => l1 i θ * l1 i θ * l1 i θ * l1 i θ, fun i => l1 i θ * l1 i θ * l1 i θ,
      fun i => l1 i θ * l1 i θ, fun i => l1 i θ]
    ((𝔼 fun i => l1 i θ) ^ 4)
    (fun i => by simp only [cen, mean, Fin.sum_univ_succ, Fin.sum_univ_zero,
      Matrix.cons_val_zero, Matrix.cons_val_succ]; ring)
  have c11 := E_lin hpos 2 (fun i => cen s ω l θ l1 i ^ 2) ![1, -2 * (𝔼 fun i => l1 i θ)]
    ![fun i => l1 i θ * l1 i θ, fun i => l1 i θ] ((𝔼 fun i => l1 i θ) ^ 2)
    (fun i => by simp only [cen, mean, Fin.sum_univ_succ, Fin.sum_univ_zero,
      Matrix.cons_val_zero, Matrix.cons_val_succ]; ring)
  simp only [Fin.sum_univ_succ, Fin.sum_univ_zero, Matrix.cons_val_zero,
    Matrix.cons_val_succ] at b2 b3 b4 c31 c22 c211 c1111 c11
  rw [F4, Z1, Z2, Z3, Z4, ratio_eq_E, ratio_eq_E, ratio_eq_E, ratio_eq_E, b2, b3, b4, c31,
    c22, c211, c1111, c11, mean]
  ring

end Cumulants

end SuperGLM.Issue544.Mixture

#print axioms SuperGLM.Issue544.Mixture.hasDerivAt_F
#print axioms SuperGLM.Issue544.Mixture.hasDerivAt_F1
#print axioms SuperGLM.Issue544.Mixture.hasDerivAt_F2
#print axioms SuperGLM.Issue544.Mixture.hasDerivAt_F3
#print axioms SuperGLM.Issue544.Mixture.iteratedDeriv_four_F
#print axioms SuperGLM.Issue544.Mixture.F1_cumulants
#print axioms SuperGLM.Issue544.Mixture.F2_cumulants
#print axioms SuperGLM.Issue544.Mixture.F3_cumulants
#print axioms SuperGLM.Issue544.Mixture.F4_cumulants
