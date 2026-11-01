import Mathlib

/-!
# Issue 544, stage 2: the window combiner `log(-expm1 Δ)`

These statements are about real numbers. They certify the real-valued formulas in
`src/superglm/distributional/kernels/_response_tails.py`:

* `_window_log_derivatives(kappa, order)`, which returns `phi^(m)(Δ)` for
  `phi(Δ) = log(-expm1(Δ))` as polynomials in `kappa = 1/expm1(-Δ)`;
* the value step of `log_window_jets` and `log_window_channels`,
  `log D = log T(near) + log(-expm1(Δ))` with `Δ = log T(far) - log T(near)`;
* the branch that returns the near end unchanged when the far end is absent
  (`Δ = -∞`) or when `kappa` rounds to zero.

They do not cover floating-point accuracy, the multivariate set-partition chain
rule that composes `phi^(m)` with `Δ`'s jets (`compose_derivatives`), or the
quadrature branch for narrow windows.

Repository base: ff2ed52d (branch `claude/issue-544-gamma-bounds`). Date: 2026-11-01.
-/

namespace SuperGLM.Issue544.Window

open Real Filter Topology Set

/-- `phi(Δ) = log(-expm1(Δ))`: the code's `np.log(-np.expm1(delta))`. -/
noncomputable def phi (Δ : ℝ) : ℝ := Real.log (-(Real.exp Δ - 1))

/-- `kappa(Δ) = 1/expm1(-Δ)`: the code's `1.0 / np.expm1(-delta)`. -/
noncomputable def kappa (Δ : ℝ) : ℝ := 1 / (Real.exp (-Δ) - 1)

lemma expm1_neg_ne_zero {Δ : ℝ} (h : Δ ≠ 0) : Real.exp (-Δ) - 1 ≠ 0 := by
  intro h0
  have h1 : Real.exp (-Δ) = 1 := by linarith
  rw [Real.exp_eq_one_iff] at h1
  exact h (by linarith)

lemma exp_ne_one {Δ : ℝ} (h : Δ ≠ 0) : Real.exp Δ ≠ 1 := by
  intro h1
  rw [Real.exp_eq_one_iff] at h1
  exact h h1

/-- `kappa = e^Δ/(1 - e^Δ)`. -/
lemma kappa_eq {Δ : ℝ} (h : Δ ≠ 0) : kappa Δ = Real.exp Δ / (1 - Real.exp Δ) := by
  have h1 : 1 - Real.exp Δ ≠ 0 := sub_ne_zero.mpr (exp_ne_one h).symm
  have h2 := expm1_neg_ne_zero h
  have h3 : Real.exp Δ ≠ 0 := (Real.exp_pos Δ).ne'
  unfold kappa
  rw [Real.exp_neg] at h2 ⊢
  field_simp

/-- For `Δ < 0` (a two-ended window), `kappa > 0`: every factor of
`_window_log_derivatives` is a sum of positive terms. -/
theorem kappa_pos {Δ : ℝ} (h : Δ < 0) : 0 < kappa Δ := by
  have : 1 < Real.exp (-Δ) := Real.one_lt_exp_iff.mpr (by linarith)
  unfold kappa
  exact div_pos one_pos (by linarith)

/-- `d kappa/dΔ = kappa (1 + kappa)` for every `Δ ≠ 0`
(docstring of `_window_log_derivatives`). -/
theorem hasDerivAt_kappa {Δ : ℝ} (h : Δ ≠ 0) :
    HasDerivAt kappa (kappa Δ * (1 + kappa Δ)) Δ := by
  have hE := expm1_neg_ne_zero h
  have h1 : HasDerivAt (fun x => Real.exp (-x) - 1) (-Real.exp (-Δ)) Δ := by
    simpa using ((hasDerivAt_neg Δ).exp).sub_const 1
  have h2 := h1.inv hE
  have hk : kappa = (fun x => Real.exp (-x) - 1)⁻¹ := by
    funext x
    simp [kappa]
  have h3 : HasDerivAt kappa (-(-Real.exp (-Δ)) / (Real.exp (-Δ) - 1) ^ 2) Δ := by
    rw [hk]
    exact h2
  convert h3 using 1
  unfold kappa
  field_simp
  ring

/-- `phi' = -kappa` for every `Δ < 0` (`derivatives = {1: -kappa}`). -/
theorem hasDerivAt_phi {Δ : ℝ} (h : Δ < 0) : HasDerivAt phi (-kappa Δ) Δ := by
  have hlt : Real.exp Δ < 1 := Real.exp_lt_one_iff.mpr h
  have h1 : HasDerivAt (fun x => -(Real.exp x - 1)) (-Real.exp Δ) Δ :=
    ((Real.hasDerivAt_exp Δ).sub_const 1).neg
  have hpos : -(Real.exp Δ - 1) ≠ 0 := by linarith
  have h2 : HasDerivAt phi (-Real.exp Δ / -(Real.exp Δ - 1)) Δ := h1.log hpos
  convert h2 using 1
  rw [kappa_eq h.ne, show -(Real.exp Δ - 1) = 1 - Real.exp Δ by ring, neg_div]

/-- Agreement on the open half-line `Δ < 0` passes to derivatives there. -/
lemma eqOn_deriv {f g : ℝ → ℝ} (h : EqOn f g (Iio 0)) : EqOn (deriv f) (deriv g) (Iio 0) :=
  fun _ hx => Filter.EventuallyEq.deriv_eq (Filter.eventuallyEq_of_mem (Iio_mem_nhds hx) h)

lemma iteratedDeriv_one_eqOn : EqOn (iteratedDeriv 1 phi) (fun Δ => -kappa Δ) (Iio 0) := by
  intro Δ hΔ
  rw [iteratedDeriv_one]
  exact (hasDerivAt_phi hΔ).deriv

lemma iteratedDeriv_two_eqOn :
    EqOn (iteratedDeriv 2 phi) (fun Δ => -kappa Δ * (1 + kappa Δ)) (Iio 0) := by
  intro Δ hΔ
  rw [iteratedDeriv_succ, eqOn_deriv iteratedDeriv_one_eqOn hΔ]
  have hk := hasDerivAt_kappa (ne_of_lt hΔ)
  have h1 : HasDerivAt (fun x => -kappa x) (-(kappa Δ * (1 + kappa Δ))) Δ := hk.neg
  rw [h1.deriv]
  ring

lemma iteratedDeriv_three_eqOn :
    EqOn (iteratedDeriv 3 phi)
      (fun Δ => (-kappa Δ * (1 + kappa Δ)) * (1 + 2 * kappa Δ)) (Iio 0) := by
  intro Δ hΔ
  rw [iteratedDeriv_succ, eqOn_deriv iteratedDeriv_two_eqOn hΔ]
  have hk := hasDerivAt_kappa (ne_of_lt hΔ)
  have h1 : HasDerivAt (fun x => -kappa x * (1 + kappa x))
      (-(kappa Δ * (1 + kappa Δ)) * (1 + kappa Δ) + -kappa Δ * (kappa Δ * (1 + kappa Δ))) Δ :=
    hk.neg.mul (hk.const_add 1)
  rw [h1.deriv]
  ring

lemma iteratedDeriv_four_eqOn :
    EqOn (iteratedDeriv 4 phi)
      (fun Δ => (-kappa Δ * (1 + kappa Δ)) * (1 + 6 * kappa Δ * (1 + kappa Δ))) (Iio 0) := by
  intro Δ hΔ
  rw [iteratedDeriv_succ, eqOn_deriv iteratedDeriv_three_eqOn hΔ]
  have hk := hasDerivAt_kappa (ne_of_lt hΔ)
  have h1 : HasDerivAt (fun x => -kappa x * (1 + kappa x) * (1 + 2 * kappa x))
      ((-(kappa Δ * (1 + kappa Δ)) * (1 + kappa Δ) + -kappa Δ * (kappa Δ * (1 + kappa Δ))) *
          (1 + 2 * kappa Δ) +
        -kappa Δ * (1 + kappa Δ) * (2 * (kappa Δ * (1 + kappa Δ)))) Δ :=
    (hk.neg.mul (hk.const_add 1)).mul ((hk.const_mul 2).const_add 1)
  rw [h1.deriv]
  ring

/-- `phi'(Δ) = -kappa` (code: `derivatives[1] = -kappa`). -/
theorem phi_deriv_one {Δ : ℝ} (h : Δ < 0) : iteratedDeriv 1 phi Δ = -kappa Δ :=
  iteratedDeriv_one_eqOn h

/-- `phi''(Δ) = -kappa (1 + kappa)` (code: `derivatives[2] = -kappa * (1.0 + kappa)`). -/
theorem phi_deriv_two {Δ : ℝ} (h : Δ < 0) :
    iteratedDeriv 2 phi Δ = -kappa Δ * (1 + kappa Δ) :=
  iteratedDeriv_two_eqOn h

/-- `phi''' = phi'' (1 + 2 kappa)`
(code: `derivatives[3] = derivatives[2] * (1.0 + 2.0 * kappa)`). -/
theorem phi_deriv_three {Δ : ℝ} (h : Δ < 0) :
    iteratedDeriv 3 phi Δ = iteratedDeriv 2 phi Δ * (1 + 2 * kappa Δ) := by
  rw [iteratedDeriv_three_eqOn h, phi_deriv_two h]

/-- `phi'''' = phi'' (1 + 6 kappa (1 + kappa))`
(code: `derivatives[4] = derivatives[2] * (1.0 + 6.0 * kappa * (1.0 + kappa))`). -/
theorem phi_deriv_four {Δ : ℝ} (h : Δ < 0) :
    iteratedDeriv 4 phi Δ = iteratedDeriv 2 phi Δ * (1 + 6 * kappa Δ * (1 + kappa Δ)) := by
  rw [iteratedDeriv_four_eqOn h, phi_deriv_two h]

/-- Two ends with `0 < T_far < T_near` have `Δ = log T_far - log T_near < 0`. -/
theorem delta_neg {Tnear Tfar : ℝ} (hfar : 0 < Tfar) (hlt : Tfar < Tnear) :
    Real.log Tfar - Real.log Tnear < 0 :=
  sub_neg.mpr (Real.log_lt_log hfar hlt)

/-- The value step of `log_window_jets` and `log_window_channels`:
`log(T_near - T_far) = log T_near + log(-expm1(log T_far - log T_near))`
for every `0 < T_far < T_near`. -/
theorem log_window {Tnear Tfar : ℝ} (hfar : 0 < Tfar) (hlt : Tfar < Tnear) :
    Real.log (Tnear - Tfar) = Real.log Tnear + phi (Real.log Tfar - Real.log Tnear) := by
  have hnear : 0 < Tnear := hfar.trans hlt
  unfold phi
  rw [Real.exp_sub, Real.exp_log hfar, Real.exp_log hnear]
  have h1 : -(Tfar / Tnear - 1) = (Tnear - Tfar) / Tnear := by
    field_simp
    ring
  rw [h1, Real.log_div (by linarith) hnear.ne']
  ring

/-- An absent far end (`T_far = 0`, `Δ = -∞`): `phi(Δ) → 0` as `Δ → -∞`, so the window is
the near end's tail, the branch `far_value > -inf` excludes. -/
theorem phi_tendsto_atBot : Tendsto phi atBot (𝓝 0) := by
  have h1 : Tendsto (fun Δ => -(Real.exp Δ - 1)) atBot (𝓝 (-(0 - 1))) :=
    (Real.tendsto_exp_atBot.sub_const 1).neg
  have h2 := (Real.continuousAt_log (by norm_num : (-(0 - 1) : ℝ) ≠ 0)).tendsto.comp h1
  rw [show (-(0 - 1) : ℝ) = 1 by norm_num, Real.log_one] at h2
  exact h2

/-- The negligible-far-end branch (`np.abs(kappa) > 0.0` false): for `Δ < 0`,
`0 ≤ -phi(Δ) ≤ kappa(Δ)`, so the value correction is no larger than `kappa`, and every
derivative correction `phi^(m)` carries a factor `kappa` by `phi_deriv_one` to
`phi_deriv_four`. -/
theorem neg_phi_le_kappa {Δ : ℝ} (h : Δ < 0) : 0 ≤ -phi Δ ∧ -phi Δ ≤ kappa Δ := by
  have hlt : Real.exp Δ < 1 := Real.exp_lt_one_iff.mpr h
  have hpos := Real.exp_pos Δ
  have hx : 0 < -(Real.exp Δ - 1) := by linarith
  constructor
  · have : Real.log (-(Real.exp Δ - 1)) ≤ 0 := Real.log_nonpos hx.le (by linarith)
    unfold phi
    linarith
  · have h1 := Real.log_le_sub_one_of_pos (inv_pos.mpr hx)
    rw [Real.log_inv] at h1
    unfold phi
    rw [kappa_eq h.ne]
    have h2 : (-(Real.exp Δ - 1))⁻¹ - 1 = Real.exp Δ / (1 - Real.exp Δ) := by
      have h3 : 1 - Real.exp Δ ≠ 0 := by linarith
      rw [show -(Real.exp Δ - 1) = 1 - Real.exp Δ by ring, inv_eq_one_div, div_sub_one h3]
      ring
    linarith

end SuperGLM.Issue544.Window

#print axioms SuperGLM.Issue544.Window.hasDerivAt_kappa
#print axioms SuperGLM.Issue544.Window.hasDerivAt_phi
#print axioms SuperGLM.Issue544.Window.kappa_pos
#print axioms SuperGLM.Issue544.Window.phi_deriv_one
#print axioms SuperGLM.Issue544.Window.phi_deriv_two
#print axioms SuperGLM.Issue544.Window.phi_deriv_three
#print axioms SuperGLM.Issue544.Window.phi_deriv_four
#print axioms SuperGLM.Issue544.Window.delta_neg
#print axioms SuperGLM.Issue544.Window.log_window
#print axioms SuperGLM.Issue544.Window.phi_tendsto_atBot
#print axioms SuperGLM.Issue544.Window.neg_phi_le_kappa
